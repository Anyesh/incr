use std::time::{Duration, Instant};

use incr_spreadsheet::coord::Coord;
use incr_spreadsheet::engine::Engine;
use incr_spreadsheet::full::FullEngine;
use incr_spreadsheet::value::CellValue;
use incr_spreadsheet::Sheet;

const ROWS: u32 = 100_000;
const CHAINS: u32 = 1_000;
const CHAIN_LEN: u32 = 100;
/// Chain `j` runs along row `j` from this column, each cell its left
/// neighbour plus one.
const CHAIN_COL: u32 = 3;
const RUNS: u32 = 15;

fn build(sheet: &mut impl Sheet) {
    for row in 1..=ROWS {
        sheet.set(Coord::new(0, row), &row.to_string());
        let sum = if row == 1 {
            "=A1".to_string()
        } else {
            format!("=B{}+A{row}", row - 1)
        };
        sheet.set(Coord::new(1, row), &sum);
    }
    for row in 1..=CHAINS {
        sheet.set(Coord::new(CHAIN_COL, row), "1");
        for col in CHAIN_COL + 1..CHAIN_COL + CHAIN_LEN {
            let left = Coord::new(col - 1, row);
            sheet.set(Coord::new(col, row), &format!("={left}+1"));
        }
    }
}

fn every_cell() -> impl Iterator<Item = Coord> {
    (1..=ROWS)
        .flat_map(|row| [Coord::new(0, row), Coord::new(1, row)])
        .chain((1..=CHAINS).flat_map(|row| {
            (CHAIN_COL..CHAIN_COL + CHAIN_LEN).map(move |col| Coord::new(col, row))
        }))
}

fn same(full: &FullEngine, incr: &Engine) -> bool {
    every_cell().all(|at| full.value(at) == incr.value(at))
}

fn timed(f: impl FnOnce()) -> Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

fn median(mut times: Vec<Duration>) -> Duration {
    times.sort();
    times[times.len() / 2]
}

fn show(d: Duration) -> String {
    let us = d.as_secs_f64() * 1e6;
    if us < 1_000.0 {
        format!("{us:.1} us")
    } else if us < 1_000_000.0 {
        format!("{:.2} ms", us / 1e3)
    } else {
        format!("{:.2} s", us / 1e6)
    }
}

fn row(label: &str, full: Duration, incr: Duration, same: bool) {
    let speedup = full.as_secs_f64() / incr.as_secs_f64();
    let verdict = if same { "same" } else { "DIFFERENT" };
    println!(
        "{label:<28} {:>10} {:>10} {speedup:>9.1}x  {verdict}",
        show(full),
        show(incr)
    );
}

fn edit_row(label: &str, at: Coord, base: u32, full: &mut FullEngine, incr: &mut Engine) {
    let (mut full_times, mut incr_times) = (Vec::new(), Vec::new());
    for run in 1..=RUNS {
        let raw = (base + run).to_string();
        full_times.push(timed(|| {
            full.set(at, &raw);
            full.recalc();
        }));
        incr_times.push(timed(|| {
            incr.set(at, &raw);
            incr.recalc();
        }));
    }
    row(
        label,
        median(full_times),
        median(incr_times),
        same(full, incr),
    );
}

fn cycle_demo() {
    let mut full = FullEngine::new();
    let mut incr = Engine::new();
    let cells = ["D1", "E1", "F1"].map(|n| Coord::parse(n).expect("cell name"));
    let mut step = |label: &str, edits: &[(&str, &str)]| {
        for (name, raw) in edits {
            let at = Coord::parse(name).expect("cell name");
            full.set(at, raw);
            incr.set(at, raw);
        }
        full.recalc();
        incr.recalc();
        let shown: Vec<String> = cells
            .iter()
            .map(|&at| {
                let (f, i) = (full.value(at), incr.value(at));
                let v = format!("{at} {}", display(&i));
                if f == i {
                    v
                } else {
                    format!("{v} (full: {})", display(&f))
                }
            })
            .collect();
        println!("{label:<36} {}", shown.join("   "));
    };
    step(
        "A1 2, D1 =E1+A1, E1 =D1, F1 =A1*2",
        &[
            ("A1", "2"),
            ("D1", "=E1+A1"),
            ("E1", "=D1"),
            ("F1", "=A1*2"),
        ],
    );
    step("break the loop: E1 1", &[("E1", "1")]);
    step("close it again: E1 =D1", &[("E1", "=D1")]);
}

fn display(v: &CellValue) -> String {
    match v {
        CellValue::Empty => "(empty)".into(),
        CellValue::Number(n) => n.to_string(),
        CellValue::Text(s) => s.clone(),
        CellValue::Error(code) => (*code).into(),
    }
}

fn main() {
    println!(
        "Running sum over {ROWS} rows (A numbers, B =B(n-1)+A(n)) and {CHAINS} chains of {CHAIN_LEN}: {} cells.",
        2 * ROWS + CHAINS * CHAIN_LEN
    );
    println!("Each edit row is the median of {RUNS} edit-plus-recalc runs; \"same\" compares every cell.\n");
    println!(
        "{:<28} {:>10} {:>10} {:>10}  answer",
        "edit", "full", "incr", "speedup"
    );

    let mut full = FullEngine::new();
    let mut incr = Engine::new();
    let full_load = timed(|| {
        build(&mut full);
        full.recalc();
    });
    let incr_load = timed(|| {
        build(&mut incr);
        incr.recalc();
    });
    row(
        "load and first recalc",
        full_load,
        incr_load,
        same(&full, &incr),
    );

    let a = |r: u32| Coord::new(0, r);
    edit_row("A100000 (last row)", a(ROWS), ROWS, &mut full, &mut incr);
    edit_row(
        "A50000 (middle row)",
        a(ROWS / 2),
        ROWS / 2,
        &mut full,
        &mut incr,
    );
    edit_row("A1 (first row)", a(1), 1, &mut full, &mut incr);
    let head = Coord::new(CHAIN_COL, CHAINS / 2);
    edit_row(
        &format!("{head} (one chain head)"),
        head,
        1,
        &mut full,
        &mut incr,
    );

    println!("\nCircular references:");
    cycle_demo();
}
