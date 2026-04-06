use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use incr_concurrent::{Incr, Runtime};

use crate::formula::{eval_expr, parse_formula};

const COLS: u8 = 10; // A-J
const ROWS: u32 = 20; // 1-20

pub fn cell_name(col: u8, row: u32) -> String {
    format!("{}{}", (b'A' + col) as char, row)
}

pub fn parse_cell_name(name: &str) -> Option<(u8, u32)> {
    let name = name.to_uppercase();
    let col_char = name.chars().next()?;
    if !col_char.is_ascii_alphabetic() {
        return None;
    }
    let col = col_char as u8 - b'A';
    if col >= COLS {
        return None;
    }
    let row: u32 = name[1..].parse().ok()?;
    if row < 1 || row > ROWS {
        return None;
    }
    Some((col, row))
}

struct CellNodes {
    content: Incr<String>,
    value: Incr<f64>,
}

pub struct SpreadsheetEngine {
    pub rt: Runtime,
    cells: HashMap<String, CellNodes>,
    /// Shared map of value nodes so query closures can look up references.
    value_nodes: Arc<RwLock<HashMap<String, Incr<f64>>>>,
    /// Cache of last-known cell values for diffing.
    prev_values: RwLock<HashMap<String, f64>>,
}

impl SpreadsheetEngine {
    pub fn new() -> Arc<Self> {
        let rt = Runtime::new();
        let value_nodes: Arc<RwLock<HashMap<String, Incr<f64>>>> =
            Arc::new(RwLock::new(HashMap::new()));

        let mut cells = HashMap::new();

        // Create all cells up front so every query closure can reference
        // any cell, even ones defined later in the grid.
        for row in 1..=ROWS {
            for col in 0..COLS {
                let name = cell_name(col, row);
                let content = rt.create_input(String::new());

                let vn = value_nodes.clone();
                let content_handle = content;
                let value = rt.create_query(move |rt| -> f64 {
                    let raw = rt.get(content_handle);
                    if raw.is_empty() {
                        return 0.0;
                    }
                    if let Some(stripped) = raw.strip_prefix('=') {
                        match parse_formula(stripped) {
                            Ok(ast) => {
                                let nodes = vn.read().expect("value_nodes lock poisoned");
                                eval_expr(&ast, rt, &nodes)
                            }
                            Err(_) => f64::NAN,
                        }
                    } else if let Ok(n) = raw.parse::<f64>() {
                        n
                    } else {
                        // Text content: display as NAN (the UI will show
                        // the raw text instead via the content field).
                        f64::NAN
                    }
                });

                rt.set_label(content.slot(), format!("{}.content", name));
                rt.set_label(value.slot(), format!("{}.value", name));

                cells.insert(name.clone(), CellNodes { content, value });
            }
        }

        // Populate the shared value_nodes map now that all cells exist.
        {
            let mut vn = value_nodes.write().expect("value_nodes lock poisoned");
            for (name, cell) in &cells {
                vn.insert(name.clone(), cell.value);
            }
        }

        let engine = Arc::new(Self {
            rt,
            cells,
            value_nodes,
            prev_values: RwLock::new(HashMap::new()),
        });

        // Force initial computation of all value nodes.
        for row in 1..=ROWS {
            for col in 0..COLS {
                let name = cell_name(col, row);
                if let Some(cell) = engine.cells.get(&name) {
                    let val = engine.rt.get(cell.value);
                    engine
                        .prev_values
                        .write()
                        .expect("prev_values lock")
                        .insert(name, val);
                }
            }
        }

        engine
    }

    /// Set a cell's raw content and return a map of cells whose computed
    /// value actually changed.
    pub fn set_cell(&self, name: &str, content: &str) -> HashMap<String, f64> {
        let name = name.to_uppercase();
        let cell = match self.cells.get(&name) {
            Some(c) => c,
            None => return HashMap::new(),
        };

        self.rt.set(cell.content, content.to_string());

        // Re-evaluate all cells and diff against previous values.
        let mut changed = HashMap::new();
        let mut prev = self.prev_values.write().expect("prev_values lock");

        for row in 1..=ROWS {
            for col in 0..COLS {
                let cname = cell_name(col, row);
                if let Some(c) = self.cells.get(&cname) {
                    let new_val = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.rt.get(c.value)
                    }))
                    .unwrap_or(f64::NAN);

                    let old_val = prev.get(&cname).copied().unwrap_or(0.0);
                    if !values_equal(old_val, new_val) {
                        changed.insert(cname.clone(), new_val);
                    }
                    prev.insert(cname, new_val);
                }
            }
        }

        changed
    }

    /// Get the full grid state: (cell_name, raw_content, computed_value).
    pub fn full_state(&self) -> Vec<(String, String, f64)> {
        let mut result = Vec::new();
        for row in 1..=ROWS {
            for col in 0..COLS {
                let name = cell_name(col, row);
                if let Some(cell) = self.cells.get(&name) {
                    let content = self.rt.get(cell.content);
                    let value = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.rt.get(cell.value)
                    }))
                    .unwrap_or(f64::NAN);
                    result.push((name, content, value));
                }
            }
        }
        result
    }

    pub fn node_count(&self) -> usize {
        self.rt.node_count()
    }
}

fn values_equal(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
        return true;
    }
    (a - b).abs() < f64::EPSILON * a.abs().max(b.abs()).max(1.0)
}

/// Load the seed data (invoice example).
pub fn seed_data(engine: &SpreadsheetEngine) {
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
        engine.set_cell(cell, content);
    }
}
