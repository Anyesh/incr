use std::sync::{Arc, Mutex};

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use incr_spreadsheet::coord::Coord;
use incr_spreadsheet::engine::Engine;
use incr_spreadsheet::value::CellValue;
use incr_spreadsheet::Sheet;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

const COLS: u32 = 10;
const ROWS: u32 = 20;

#[derive(Clone, Serialize)]
struct CellState {
    cell: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(flatten)]
    value: CellValue,
}

#[derive(Serialize)]
struct FullStateMsg {
    r#type: &'static str,
    cells: Vec<CellState>,
    node_count: usize,
}

#[derive(Serialize)]
struct UpdateMsg {
    r#type: &'static str,
    changed: Vec<CellState>,
}

#[derive(Deserialize)]
struct SetCellMsg {
    cell: String,
    content: String,
}

struct AppState {
    engine: Mutex<Engine>,
    tx: broadcast::Sender<String>,
}

impl AppState {
    /// Applies and broadcasts the edit under one hold of the lock: the
    /// runtime keeps one queue of dirtied cells, so two edits
    /// recalculating at once would each take part of the other's report,
    /// and two broadcasts racing could leave clients on the older value.
    fn edit(&self, at: Coord, content: String) {
        let mut engine = self.engine.lock().expect("engine lock");
        let update = apply(&mut engine, at, content);
        if let Ok(json) = serde_json::to_string(&update) {
            let _ = self.tx.send(json);
        }
    }
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    // Subscribed before the snapshot, so an edit after it is not missed.
    let mut rx = state.tx.subscribe();
    let msg = full_state(&state.engine.lock().expect("engine lock"));
    if let Ok(json) = serde_json::to_string(&msg) {
        let _ = sender.send(Message::Text(json.into())).await;
    }

    // Spawn a task to forward broadcasts to this client.
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            let Message::Text(text) = msg else { continue };
            let Ok(set_msg) = serde_json::from_str::<SetCellMsg>(&text) else {
                continue;
            };
            let Some(at) = Coord::parse(&set_msg.cell) else {
                continue;
            };
            let state = Arc::clone(&state);
            if let Err(e) =
                tokio::task::spawn_blocking(move || state.edit(at, set_msg.content)).await
            {
                eprintln!("edit failed: {e}");
            }
        }
    });

    // If either task completes, abort the other.
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
}

/// Sets the cell and recalculates. The edited cell is always sent, with
/// its content, since a new content can leave the value unchanged.
fn apply(engine: &mut Engine, at: Coord, content: String) -> UpdateMsg {
    engine.set(at, &content);
    let mut changed: Vec<CellState> = engine
        .recalc()
        .into_iter()
        .filter(|&(cell, _)| cell != at)
        .map(|(cell, value)| CellState {
            cell: cell.to_string(),
            content: None,
            value,
        })
        .collect();
    changed.push(CellState {
        cell: at.to_string(),
        content: Some(content),
        value: engine.value(at),
    });
    UpdateMsg {
        r#type: "update",
        changed,
    }
}

fn full_state(engine: &Engine) -> FullStateMsg {
    let cells = (1..=ROWS)
        .flat_map(|row| (0..COLS).map(move |col| Coord::new(col, row)))
        .map(|at| CellState {
            cell: at.to_string(),
            content: Some(engine.content(at)),
            value: engine.value(at),
        })
        .collect();
    FullStateMsg {
        r#type: "full_state",
        cells,
        node_count: engine.node_count(),
    }
}

fn seed_data(engine: &mut Engine) {
    let seeds: &[(&str, &str)] = &[
        ("A1", "Price"),
        ("B1", "Qty"),
        ("C1", "Total"),
        ("A2", "29.99"),
        ("B2", "5"),
        ("C2", "=A2*B2"),
        ("A3", "49.99"),
        ("B3", "3"),
        ("C3", "=A3*B3"),
        ("A4", "9.99"),
        ("B4", "12"),
        ("C4", "=A4*B4"),
        ("A6", "Subtotal"),
        ("C6", "=SUM(C2:C4)"),
        ("A7", "Tax (8%)"),
        ("C7", "=C6*0.08"),
        ("A8", "Total"),
        ("C8", "=C6+C7"),
    ];
    for (cell, content) in seeds {
        engine.set(Coord::parse(cell).expect("seed cell"), content);
    }
    engine.recalc();
}

#[tokio::main]
async fn main() {
    let mut engine = Engine::new();
    seed_data(&mut engine);

    println!("incr spreadsheet demo");
    println!("nodes: {}", engine.node_count());

    let (tx, _) = broadcast::channel::<String>(256);

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        tx,
    });

    let app = axum::Router::new()
        .route("/ws", get(ws_handler))
        .fallback(get(serve_static))
        .with_state(state);

    let addr = "0.0.0.0:3001";
    println!("listening on {addr}");
    println!("open http://localhost:3001 in your browser");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn serve_static() -> impl IntoResponse {
    let html = include_str!("../static/index.html");
    axum::response::Html(html)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn send(engine: &mut Engine, cell: &str, content: &str) -> serde_json::Value {
        let at = Coord::parse(cell).unwrap();
        serde_json::to_value(apply(engine, at, content.to_string())).unwrap()
    }

    #[test]
    fn updates_carry_the_edited_cell_last_with_its_content() {
        let engine = &mut Engine::new();
        send(engine, "A1", "=B1");
        let update = send(engine, "B1", "=A1");
        assert_eq!(
            update["changed"],
            serde_json::json!([
                {"cell": "A1", "error": "#CIRC!"},
                {"cell": "B1", "content": "=A1", "error": "#CIRC!"},
            ])
        );
        let update = send(engine, "B1", "5");
        assert_eq!(
            update["changed"],
            serde_json::json!([
                {"cell": "A1", "value": 5.0},
                {"cell": "B1", "content": "5", "value": 5.0},
            ])
        );
        let update = send(engine, "B1", "5.0");
        assert_eq!(
            update["changed"],
            serde_json::json!([{"cell": "B1", "content": "5.0", "value": 5.0}])
        );
    }

    #[test]
    fn full_state_covers_the_page_grid_with_seed_totals() {
        let mut engine = Engine::new();
        seed_data(&mut engine);
        let msg = serde_json::to_value(full_state(&engine)).unwrap();
        let cells = msg["cells"].as_array().unwrap();
        assert_eq!(cells.len(), (COLS * ROWS) as usize);
        let c8 = cells.iter().find(|c| c["cell"] == "C8").unwrap();
        assert_eq!(c8["content"], "=C6+C7");
        let subtotal = 29.99 * 5.0 + 49.99 * 3.0 + 9.99 * 12.0;
        assert_eq!(c8["value"], subtotal + subtotal * 0.08);
        let a1 = cells.iter().find(|c| c["cell"] == "A1").unwrap();
        assert_eq!(a1["text"], "Price");
    }
}
