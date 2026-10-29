use std::collections::BTreeMap;

use incr_spreadsheet::coord::Coord;
use incr_spreadsheet::engine::Engine;
use incr_spreadsheet::full::FullEngine;
use incr_spreadsheet::value::{CellValue, CIRC};
use incr_spreadsheet::Sheet;

fn c(name: &str) -> Coord {
    Coord::parse(name).unwrap()
}

fn num(n: f64) -> CellValue {
    CellValue::Number(n)
}

fn circ() -> CellValue {
    CellValue::Error(CIRC)
}

fn set_all(sheet: &mut impl Sheet, cells: &[(&str, &str)]) {
    for (name, raw) in cells {
        sheet.set(c(name), raw);
    }
}

fn changed(sheet: &mut impl Sheet) -> BTreeMap<String, CellValue> {
    sheet
        .recalc()
        .into_iter()
        .map(|(at, v)| (at.to_string(), v))
        .collect()
}

fn expect(sheet: &impl Sheet, cells: &[(&str, CellValue)]) {
    for (name, want) in cells {
        assert_eq!(sheet.value(c(name)), *want, "{name}");
    }
}

fn invoice(mut s: impl Sheet) {
    set_all(
        &mut s,
        &[
            ("A1", "Price"),
            ("A2", "29.99"),
            ("B2", "5"),
            ("C2", "=A2*B2"),
            ("A3", "49.99"),
            ("B3", "3"),
            ("C3", "=A3*B3"),
            ("C6", "=SUM(C2:C4)"),
            ("C7", "=C6*0.08"),
            ("C8", "=C6+C7"),
        ],
    );
    let first = changed(&mut s);
    assert_eq!(first.len(), 10);
    assert_eq!(first["A1"], CellValue::Text("Price".into()));
    let subtotal = 29.99 * 5.0 + 49.99 * 3.0;
    expect(&s, &[("C6", num(subtotal)), ("C8", num(subtotal * 1.08))]);

    s.set(c("B2"), "6");
    let names: Vec<String> = changed(&mut s).into_keys().collect();
    assert_eq!(names, ["B2", "C2", "C6", "C7", "C8"]);
    assert!(s.recalc().is_empty());

    s.set(c("B3"), "3");
    assert!(s.recalc().is_empty());
}

fn clearing_a_cell_reports_empty(mut s: impl Sheet) {
    set_all(&mut s, &[("A1", "4"), ("B1", "=A1+1")]);
    s.recalc();
    s.set(c("A1"), "");
    let got = changed(&mut s);
    assert_eq!(got["A1"], CellValue::Empty);
    assert_eq!(got["B1"], num(1.0));
    assert_eq!(got.len(), 2);
}

fn two_cell_loop_breaks_and_closes(mut s: impl Sheet) {
    set_all(&mut s, &[("A1", "=B1"), ("B1", "=A1")]);
    s.recalc();
    expect(&s, &[("A1", circ()), ("B1", circ())]);

    s.set(c("B1"), "5");
    let got = changed(&mut s);
    assert_eq!(got.len(), 2);
    expect(&s, &[("A1", num(5.0)), ("B1", num(5.0))]);

    s.set(c("B1"), "=A1");
    s.recalc();
    expect(&s, &[("A1", circ()), ("B1", circ())]);
}

fn self_reference(mut s: impl Sheet) {
    set_all(&mut s, &[("A1", "=A1+1"), ("B1", "=A1*2")]);
    s.recalc();
    expect(&s, &[("A1", circ()), ("B1", circ())]);
    s.set(c("A1"), "3");
    s.recalc();
    expect(&s, &[("A1", num(3.0)), ("B1", num(6.0))]);
}

fn downstream_of_a_loop(mut s: impl Sheet) {
    set_all(
        &mut s,
        &[
            ("A1", "2"),
            ("D1", "=E1+A1"),
            ("E1", "=D1"),
            ("F1", "=D1*2"),
            ("G1", "=A1*2"),
        ],
    );
    s.recalc();
    expect(
        &s,
        &[
            ("D1", circ()),
            ("E1", circ()),
            ("F1", circ()),
            ("G1", num(4.0)),
        ],
    );

    s.set(c("E1"), "1");
    s.recalc();
    expect(&s, &[("D1", num(3.0)), ("F1", num(6.0))]);

    s.set(c("A1"), "10");
    let names: Vec<String> = changed(&mut s).into_keys().collect();
    assert_eq!(names, ["A1", "D1", "F1", "G1"]);

    s.set(c("E1"), "=D1");
    s.recalc();
    expect(&s, &[("D1", circ()), ("E1", circ()), ("F1", circ())]);
}

/// Sets the bottom row first, so the first cell an engine reads sits at
/// the end of a 100,000-deep chain.
fn running_sum_100k(mut s: impl Sheet) {
    const N: u32 = 100_000;
    for row in (1..=N).rev() {
        s.set(Coord::new(0, row), &row.to_string());
        let sum = if row == 1 {
            "=A1".to_string()
        } else {
            format!("=B{}+A{}", row - 1, row)
        };
        s.set(Coord::new(1, row), &sum);
    }
    s.recalc();
    let n = f64::from(N);
    assert_eq!(s.value(Coord::new(1, N)), num(n * (n + 1.0) / 2.0));

    s.set(c("A1"), "2");
    let got = s.recalc();
    assert_eq!(got.len(), N as usize + 1);
    assert_eq!(s.value(Coord::new(1, N)), num(n * (n + 1.0) / 2.0 + 1.0));
}

macro_rules! sheet_tests {
    ($engine:ident, $make:expr) => {
        mod $engine {
            use super::*;

            #[test]
            fn invoice() {
                super::invoice($make);
            }

            #[test]
            fn clearing_a_cell_reports_empty() {
                super::clearing_a_cell_reports_empty($make);
            }

            #[test]
            fn two_cell_loop_breaks_and_closes() {
                super::two_cell_loop_breaks_and_closes($make);
            }

            #[test]
            fn self_reference() {
                super::self_reference($make);
            }

            #[test]
            fn downstream_of_a_loop() {
                super::downstream_of_a_loop($make);
            }

            #[test]
            fn running_sum_100k() {
                super::running_sum_100k($make);
            }
        }
    };
}

sheet_tests!(full, FullEngine::new());
sheet_tests!(incr, Engine::new());
