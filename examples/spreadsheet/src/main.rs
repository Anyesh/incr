mod engine;
mod formula;

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::engine::{seed_data, SpreadsheetEngine};

#[derive(Clone, Serialize)]
struct CellState {
    cell: String,
    content: String,
    value: f64,
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
    engine: Arc<SpreadsheetEngine>,
    tx: broadcast::Sender<String>,
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    // Send full state on connect.
    let full = state.engine.full_state();
    let cells: Vec<CellState> = full
        .into_iter()
        .map(|(cell, content, value)| CellState {
            cell,
            content,
            value,
        })
        .collect();
    let msg = FullStateMsg {
        r#type: "full_state",
        cells,
        node_count: state.engine.node_count(),
    };
    if let Ok(json) = serde_json::to_string(&msg) {
        let _ = sender.send(Message::Text(json.into())).await;
    }

    // Subscribe to broadcast for updates from other clients.
    let mut rx = state.tx.subscribe();

    // Spawn a task to forward broadcasts to this client.
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    // Process incoming messages from this client.
    let engine = state.engine.clone();
    let tx = state.tx.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                if let Ok(set_msg) = serde_json::from_str::<SetCellMsg>(&text) {
                    let changed_values = engine.set_cell(&set_msg.cell, &set_msg.content);

                    // Build update with content for the edited cell and values for all changed.
                    let mut changed: Vec<CellState> = changed_values
                        .into_iter()
                        .map(|(cell, value)| {
                            let content = if cell.eq_ignore_ascii_case(&set_msg.cell) {
                                set_msg.content.clone()
                            } else {
                                String::new()
                            };
                            CellState {
                                cell,
                                content,
                                value,
                            }
                        })
                        .collect();

                    // Always include the edited cell even if its value didn't "change"
                    // (e.g., setting text content).
                    if !changed
                        .iter()
                        .any(|c| c.cell.eq_ignore_ascii_case(&set_msg.cell))
                    {
                        let full = engine.full_state();
                        if let Some((_, content, value)) = full
                            .iter()
                            .find(|(name, _, _)| name.eq_ignore_ascii_case(&set_msg.cell))
                        {
                            changed.push(CellState {
                                cell: set_msg.cell.to_uppercase(),
                                content: content.clone(),
                                value: *value,
                            });
                        }
                    }

                    let update = UpdateMsg {
                        r#type: "update",
                        changed,
                    };
                    if let Ok(json) = serde_json::to_string(&update) {
                        let _ = tx.send(json);
                    }
                }
            }
        }
    });

    // If either task completes, abort the other.
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
}

#[tokio::main]
async fn main() {
    let engine = SpreadsheetEngine::new();
    seed_data(&engine);

    println!("incr spreadsheet demo");
    println!("nodes: {}", engine.node_count());

    let (tx, _) = broadcast::channel::<String>(256);

    let state = Arc::new(AppState { engine, tx });

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
