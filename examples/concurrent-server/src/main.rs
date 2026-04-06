//! Concurrent computation server demo.
//!
//! Proves v2's concurrent access model: one writer thread feeds live
//! market data into an incr graph, multiple HTTP handler threads read
//! derived portfolio values simultaneously without blocking.
//!
//! Run modes:
//!   cargo run -p incr-concurrent-server                  # concurrent (default)
//!   cargo run -p incr-concurrent-server -- --serialized  # single-threaded baseline
//!
//! Then load-test:
//!   wrk -t4 -c100 -d10s http://localhost:3000/portfolio
//!
//! The concurrent mode should show higher throughput and lower latency
//! because readers proceed in parallel. The serialized mode funnels
//! all reads through one thread, forcing them to queue.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::routing::get;
use axum::Json;
use incr_core::{Incr, Runtime};
use rand::Rng;
use serde::Serialize;
use tokio::sync::{mpsc, oneshot};

const STOCKS: &[&str] = &[
    "AAPL", "GOOG", "MSFT", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC", "WFC",
];

const INITIAL_PRICES: &[f64] = &[
    185.0, 175.0, 420.0, 185.0, 500.0, 130.0, 250.0, 195.0, 35.0, 58.0,
];

const SHARES: &[f64] = &[
    100.0, 50.0, 30.0, 80.0, 20.0, 150.0, 40.0, 60.0, 200.0, 100.0,
];

const TECH_INDICES: &[usize] = &[0, 1, 2, 3, 4, 5, 6]; // AAPL..TSLA
const FINANCE_INDICES: &[usize] = &[7, 8, 9]; // JPM, BAC, WFC

struct Graph {
    rt: Runtime,
    prices: Vec<Incr<f64>>,
    positions: Vec<Incr<f64>>,
    portfolio_value: Incr<f64>,
    tech_value: Incr<f64>,
    finance_value: Incr<f64>,
    best_performer: Incr<f64>,
}

impl Graph {
    fn new() -> Self {
        let rt = Runtime::new();
        let mut prices = Vec::with_capacity(STOCKS.len());
        let mut positions = Vec::with_capacity(STOCKS.len());

        for (i, &initial) in INITIAL_PRICES.iter().enumerate() {
            let price = rt.create_input(initial);
            prices.push(price);

            let shares = SHARES[i];
            let pos = rt.create_query(move |rt| -> f64 { rt.get(price) * shares });
            positions.push(pos);
        }

        let pos_clone = positions.clone();
        let portfolio_value =
            rt.create_query(move |rt| -> f64 { pos_clone.iter().map(|p| rt.get(*p)).sum() });

        let tech_positions: Vec<Incr<f64>> = TECH_INDICES.iter().map(|&i| positions[i]).collect();
        let tech_value =
            rt.create_query(move |rt| -> f64 { tech_positions.iter().map(|p| rt.get(*p)).sum() });

        let fin_positions: Vec<Incr<f64>> = FINANCE_INDICES.iter().map(|&i| positions[i]).collect();
        let finance_value =
            rt.create_query(move |rt| -> f64 { fin_positions.iter().map(|p| rt.get(*p)).sum() });

        let prices_for_best = prices.clone();
        let best_performer = rt.create_query(move |rt| -> f64 {
            let mut best_pct = f64::NEG_INFINITY;
            for (i, &price_handle) in prices_for_best.iter().enumerate() {
                let current = rt.get(price_handle);
                let pct = (current - INITIAL_PRICES[i]) / INITIAL_PRICES[i] * 100.0;
                if pct > best_pct {
                    best_pct = pct;
                }
            }
            best_pct
        });

        // Force initial computation.
        let _ = rt.get(portfolio_value);

        Graph {
            rt,
            prices,
            positions,
            portfolio_value,
            tech_value,
            finance_value,
            best_performer,
        }
    }
}

#[derive(Serialize)]
struct PortfolioResponse {
    stocks: Vec<StockInfo>,
    portfolio_value: f64,
    tech_sector: f64,
    finance_sector: f64,
    best_performer_pct: f64,
    read_latency_ns: u64,
}

#[derive(Serialize)]
struct StockInfo {
    symbol: &'static str,
    price: f64,
    position: f64,
}

// ---------------------------------------------------------------------------
// Concurrent mode: readers call rt.get() directly from handler threads
// ---------------------------------------------------------------------------

async fn portfolio_concurrent(State(graph): State<Arc<Graph>>) -> Json<PortfolioResponse> {
    let start = Instant::now();

    let mut stocks = Vec::with_capacity(STOCKS.len());
    for (i, &sym) in STOCKS.iter().enumerate() {
        stocks.push(StockInfo {
            symbol: sym,
            price: graph.rt.get(graph.prices[i]),
            position: graph.rt.get(graph.positions[i]),
        });
    }

    let resp = PortfolioResponse {
        stocks,
        portfolio_value: graph.rt.get(graph.portfolio_value),
        tech_sector: graph.rt.get(graph.tech_value),
        finance_sector: graph.rt.get(graph.finance_value),
        best_performer_pct: graph.rt.get(graph.best_performer),
        read_latency_ns: start.elapsed().as_nanos() as u64,
    };

    Json(resp)
}

// ---------------------------------------------------------------------------
// Serialized mode: all reads funneled through one thread via channel
// ---------------------------------------------------------------------------

struct SerializedState {
    tx: mpsc::Sender<oneshot::Sender<PortfolioResponse>>,
}

async fn portfolio_serialized(
    State(state): State<Arc<SerializedState>>,
) -> Json<PortfolioResponse> {
    let (reply_tx, reply_rx) = oneshot::channel();
    let _ = state.tx.send(reply_tx).await;
    let resp = reply_rx.await.unwrap();
    Json(resp)
}

fn read_portfolio(graph: &Graph) -> PortfolioResponse {
    let start = Instant::now();

    let mut stocks = Vec::with_capacity(STOCKS.len());
    for (i, &sym) in STOCKS.iter().enumerate() {
        stocks.push(StockInfo {
            symbol: sym,
            price: graph.rt.get(graph.prices[i]),
            position: graph.rt.get(graph.positions[i]),
        });
    }

    PortfolioResponse {
        stocks,
        portfolio_value: graph.rt.get(graph.portfolio_value),
        tech_sector: graph.rt.get(graph.tech_value),
        finance_sector: graph.rt.get(graph.finance_value),
        best_performer_pct: graph.rt.get(graph.best_performer),
        read_latency_ns: start.elapsed().as_nanos() as u64,
    }
}

// ---------------------------------------------------------------------------
// Writer: simulates a market data feed
// ---------------------------------------------------------------------------

fn run_writer(graph: &Graph) {
    let mut rng = rand::thread_rng();
    loop {
        for &price_handle in &graph.prices {
            let current = graph.rt.get(price_handle);
            let delta: f64 = rng.gen_range(-0.5..0.5);
            let new_price = (current + delta).max(1.0);
            graph.rt.set(price_handle, new_price);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let serialized = std::env::args().any(|a| a == "--serialized");
    let mode = if serialized {
        "serialized"
    } else {
        "concurrent"
    };

    println!("incr concurrent server demo");
    println!("mode: {mode}");
    println!("stocks: {}", STOCKS.len());
    println!("writer interval: 10ms");
    println!();

    let graph = Arc::new(Graph::new());
    println!("graph built: {} nodes", graph.rt.node_count());

    // Start writer thread.
    let writer_graph = graph.clone();
    std::thread::spawn(move || run_writer(&writer_graph));

    let app = if serialized {
        // Serialized: one reader thread, channel-based.
        let (tx, mut rx) = mpsc::channel::<oneshot::Sender<PortfolioResponse>>(1024);
        let reader_graph = graph.clone();
        std::thread::spawn(move || {
            while let Some(reply_tx) = rx.blocking_recv() {
                let resp = read_portfolio(&reader_graph);
                let _ = reply_tx.send(resp);
            }
        });

        let state = Arc::new(SerializedState { tx });
        axum::Router::new()
            .route("/portfolio", get(portfolio_serialized))
            .with_state(state)
    } else {
        // Concurrent: handlers read directly.
        axum::Router::new()
            .route("/portfolio", get(portfolio_concurrent))
            .with_state(graph)
    };

    let addr = "0.0.0.0:3000";
    println!("listening on {addr}");
    println!();
    println!("load test with:");
    println!("  wrk -t4 -c100 -d10s http://localhost:3000/portfolio");
    println!();

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
