use std::collections::BTreeMap;

use incr_spreadsheet::coord::Coord;
use incr_spreadsheet::engine::Engine;
use incr_spreadsheet::full::FullEngine;
use incr_spreadsheet::value::CellValue;
use incr_spreadsheet::Sheet;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const SEEDS: u64 = 200;
const EDITS: usize = 30;
const GRID_COLS: u32 = 6;
const GRID_ROWS: u32 = 40;
/// Past the release frame budget of 256, so reads down the chain replay.
const CHAIN: u32 = 600;
const CHAIN_COL: u32 = GRID_COLS;
const SUMMARY: Coord = Coord {
    col: GRID_COLS + 1,
    row: 1,
};

/// Grid cells are ordered row-major and a formula names only cells
/// before its own, chain cells name the cell above and grid cells, and
/// the summary names the chain's end and the grid, so no edit can close
/// a cycle.
fn grid_coord(index: u32) -> Coord {
    Coord::new(index % GRID_COLS, index / GRID_COLS + 1)
}

fn earlier_ref(rng: &mut StdRng, index: u32) -> Option<Coord> {
    (index > 0).then(|| grid_coord(rng.gen_range(0..index)))
}

/// A rectangle whose bottom-right corner precedes `index`, so every cell
/// in it does too.
fn earlier_range(rng: &mut StdRng, index: u32) -> Option<String> {
    let end = earlier_ref(rng, index)?;
    let start = Coord::new(rng.gen_range(0..=end.col), rng.gen_range(1..=end.row));
    Some(format!("{start}:{end}"))
}

fn number(rng: &mut StdRng) -> String {
    match rng.gen_range(0..4) {
        0 => "0".into(),
        1 => format!("{}", rng.gen_range(-5..=5)),
        _ => format!("{:.2}", rng.gen_range(-100.0..100.0)),
    }
}

fn operand(rng: &mut StdRng, index: u32) -> String {
    match earlier_ref(rng, index) {
        Some(at) if rng.gen_bool(0.7) => at.to_string(),
        _ => number(rng),
    }
}

fn formula(rng: &mut StdRng, index: u32) -> String {
    let a = operand(rng, index);
    let b = operand(rng, index);
    let body = match rng.gen_range(0..8) {
        0 => format!("{a}+{b}"),
        1 => format!("{a}*{b}-{}", operand(rng, index)),
        2 => format!("{a}/{b}"),
        3 => format!("-{a}"),
        4 => format!(
            "IF({a}>{b}, {}, {})",
            operand(rng, index),
            operand(rng, index)
        ),
        _ => {
            let f = ["SUM", "MIN", "MAX", "AVG"][rng.gen_range(0..4)];
            match earlier_range(rng, index) {
                Some(range) => format!("{f}({range}, {a})"),
                None => format!("{f}({a})"),
            }
        }
    };
    format!("={body}")
}

fn grid_content(rng: &mut StdRng, index: u32) -> String {
    match rng.gen_range(0..10) {
        0 => String::new(),
        1 => "label".into(),
        2 => "=1+".into(),
        3..=5 => number(rng),
        _ => formula(rng, index),
    }
}

fn chain_content(rng: &mut StdRng, row: u32) -> String {
    let above = Coord::new(CHAIN_COL, row - 1);
    match rng.gen_range(0..10) {
        0 => number(rng),
        1 => format!(
            "={above}+{}",
            grid_coord(rng.gen_range(0..GRID_COLS * GRID_ROWS))
        ),
        _ if row == 1 => format!("=A1+{}", rng.gen_range(1..4)),
        _ => format!("={above}+1"),
    }
}

fn summary_content() -> String {
    let last = Coord::new(GRID_COLS - 1, GRID_ROWS);
    let chain_end = Coord::new(CHAIN_COL, CHAIN);
    format!("={chain_end}+SUM(A1:{last})")
}

fn every_cell() -> impl Iterator<Item = Coord> {
    (0..GRID_COLS * GRID_ROWS)
        .map(grid_coord)
        .chain((1..=CHAIN).map(|row| Coord::new(CHAIN_COL, row)))
        .chain(std::iter::once(SUMMARY))
}

fn snapshot(sheet: &impl Sheet) -> BTreeMap<Coord, CellValue> {
    every_cell().map(|at| (at, sheet.value(at))).collect()
}

fn diff(
    before: &BTreeMap<Coord, CellValue>,
    after: &BTreeMap<Coord, CellValue>,
) -> BTreeMap<Coord, CellValue> {
    after
        .iter()
        .filter(|(at, v)| before.get(at) != Some(v))
        .map(|(&at, v)| (at, v.clone()))
        .collect()
}

fn reported(sheet: &mut impl Sheet) -> BTreeMap<Coord, CellValue> {
    let changes = sheet.recalc();
    let count = changes.len();
    let map: BTreeMap<Coord, CellValue> = changes.into_iter().collect();
    assert_eq!(map.len(), count, "a cell was reported twice");
    map
}

fn set_both(full: &mut FullEngine, incr: &mut Engine, at: Coord, raw: &str) {
    full.set(at, raw);
    incr.set(at, raw);
}

fn run(seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut full = FullEngine::new();
    let mut incr = Engine::new();
    let mut before = snapshot(&full);

    let mut builds: Vec<(Coord, String)> = (0..GRID_COLS * GRID_ROWS)
        .map(|i| (grid_coord(i), grid_content(&mut rng, i)))
        .collect();
    for row in 1..=CHAIN {
        builds.push((Coord::new(CHAIN_COL, row), chain_content(&mut rng, row)));
    }
    builds.push((SUMMARY, summary_content()));
    if seed % 2 == 1 {
        builds.reverse();
    }
    let mut log = Vec::new();
    for (at, raw) in &builds {
        set_both(&mut full, &mut incr, *at, raw);
    }

    for step in 0..=EDITS {
        if step > 0 {
            for _ in 0..rng.gen_range(1..=3) {
                let (at, raw) = if rng.gen_bool(0.7) {
                    let i = rng.gen_range(0..GRID_COLS * GRID_ROWS);
                    (grid_coord(i), grid_content(&mut rng, i))
                } else {
                    let row = rng.gen_range(1..=CHAIN);
                    (Coord::new(CHAIN_COL, row), chain_content(&mut rng, row))
                };
                log.push(format!("{at}={raw}"));
                set_both(&mut full, &mut incr, at, &raw);
            }
        }
        let full_changes = reported(&mut full);
        let incr_changes = reported(&mut incr);
        let after = snapshot(&full);
        let context = || format!("seed {seed} step {step} edits {log:?}");
        assert_eq!(snapshot(&incr), after, "values differ: {}", context());
        let want = diff(&before, &after);
        assert_eq!(full_changes, want, "full report: {}", context());
        assert_eq!(incr_changes, want, "incr report: {}", context());
        before = after;
    }
}

#[test]
fn random_acyclic_sheets_agree_after_every_edit() {
    for seed in 0..SEEDS {
        run(seed);
    }
}
