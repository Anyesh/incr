use incr_spreadsheet::coord::Coord;
use incr_spreadsheet::formula::{
    eval_expr, parse_content, parse_formula, Content, Expr, Op, ParseError,
};
use incr_spreadsheet::value::{CellValue, CIRC, DIV0, PARSE, VALUE};

fn c(name: &str) -> Coord {
    Coord::parse(name).unwrap()
}

fn eval_with(formula: &str, cells: &[(&str, CellValue)]) -> (CellValue, Vec<Coord>) {
    let expr = parse_formula(formula).unwrap();
    let mut read = Vec::new();
    let value = eval_expr(&expr, &mut |at| {
        read.push(at);
        cells
            .iter()
            .find(|(name, _)| c(name) == at)
            .map(|(_, v)| v.clone())
            .unwrap_or(CellValue::Empty)
    });
    (value, read)
}

fn num(n: f64) -> CellValue {
    CellValue::Number(n)
}

#[test]
fn coords_parse_and_print_multi_letter_columns() {
    assert_eq!(c("A1"), Coord::new(0, 1));
    assert_eq!(c("z3"), Coord::new(25, 3));
    assert_eq!(c("AA10"), Coord::new(26, 10));
    assert_eq!(c("ALL1000"), Coord::new(999, 1000));
    for name in ["A1", "Z99", "AA10", "ALL1000", "XFD100000"] {
        assert_eq!(c(name).to_string(), name);
    }
    for bad in ["", "A", "1A", "A0", "A1B", "A-1", "AAAAAA1", "A99999999"] {
        assert_eq!(Coord::parse(bad), None, "{bad}");
    }
}

#[test]
fn refs_parse_to_coords_and_ranges_expand_row_major() {
    assert_eq!(parse_formula("b2"), Ok(Expr::CellRef(Coord::new(1, 2))));
    let Ok(Expr::FnCall(name, args)) = parse_formula("SUM(A1:B2)") else {
        panic!("expected a call");
    };
    assert_eq!(name, "SUM");
    let refs: Vec<Expr> = ["A1", "B1", "A2", "B2"]
        .iter()
        .map(|n| Expr::CellRef(c(n)))
        .collect();
    assert_eq!(args, refs);
    assert!(matches!(
        parse_formula("A1+B2*3"),
        Ok(Expr::BinOp(_, Op::Add, _))
    ));
}

#[test]
fn unknown_functions_and_bad_if_arity_fail_to_parse() {
    assert_eq!(parse_formula("FOO(A1)"), Err(ParseError));
    assert_eq!(parse_formula("IF(A1, B1)"), Err(ParseError));
    assert_eq!(parse_formula("FOO"), Err(ParseError));
    assert_eq!(parse_formula("SUM(1+2:A3)"), Err(ParseError));
    assert_eq!(parse_formula("SUM(A1:A2000000)"), Err(ParseError));
}

#[test]
fn contents_classify_as_empty_number_text_formula_or_invalid() {
    assert_eq!(parse_content(""), Content::Empty);
    assert_eq!(parse_content("29.99"), Content::Number(29.99));
    assert_eq!(parse_content("Price"), Content::Text("Price".into()));
    assert_eq!(
        parse_content("=A1"),
        Content::Formula(Expr::CellRef(c("A1")))
    );
    assert_eq!(parse_content("=1+"), Content::Invalid);
    assert_eq!(
        parse_content("=1+").evaluate(&mut |_| CellValue::Empty),
        CellValue::Error(PARSE)
    );
    assert_eq!(
        parse_content("Price").evaluate(&mut |_| CellValue::Empty),
        CellValue::Text("Price".into())
    );
}

#[test]
fn precedents_include_both_if_branches() {
    let content = parse_content("=IF(A1>0, B1, SUM(C1:C2))");
    let mut precedents = content.precedents();
    precedents.sort_by_key(|p| (p.row, p.col));
    assert_eq!(precedents, vec![c("A1"), c("B1"), c("C1"), c("C2")]);
    assert!(parse_content("5").precedents().is_empty());
}

#[test]
fn arithmetic_treats_an_empty_ref_as_zero() {
    let (v, _) = eval_with("A1*2+B1", &[("A1", num(3.0))]);
    assert_eq!(v, num(6.0));
    let (v, _) = eval_with("B1", &[]);
    assert_eq!(v, num(0.0));
}

#[test]
fn text_in_arithmetic_is_a_value_error() {
    let (v, _) = eval_with("A1+1", &[("A1", CellValue::Text("x".into()))]);
    assert_eq!(v, CellValue::Error(VALUE));
    let (v, _) = eval_with("A1", &[("A1", CellValue::Text("x".into()))]);
    assert_eq!(v, CellValue::Text("x".into()));
}

#[test]
fn division_by_zero_is_div0() {
    assert_eq!(eval_with("1/A1", &[]).0, CellValue::Error(DIV0));
}

#[test]
fn errors_propagate_and_both_operands_are_read() {
    let cells = [
        ("A1", CellValue::Error(CIRC)),
        ("B1", CellValue::Error(DIV0)),
    ];
    let (v, read) = eval_with("A1+B1", &cells);
    assert_eq!(v, CellValue::Error(CIRC));
    assert_eq!(read, vec![c("A1"), c("B1")]);
    let (v, _) = eval_with("-B1", &cells);
    assert_eq!(v, CellValue::Error(DIV0));
    let (v, read) = eval_with("SUM(A1:B1, C1)", &cells);
    assert_eq!(v, CellValue::Error(CIRC));
    assert_eq!(read.len(), 3);
}

#[test]
fn aggregates_skip_text_and_empty_refs() {
    let cells = [
        ("A1", num(4.0)),
        ("A2", CellValue::Text("n/a".into())),
        ("A4", num(-2.0)),
    ];
    assert_eq!(eval_with("SUM(A1:A4)", &cells).0, num(2.0));
    assert_eq!(eval_with("AVG(A1:A4)", &cells).0, num(1.0));
    assert_eq!(eval_with("MIN(A1:A4)", &cells).0, num(-2.0));
    assert_eq!(eval_with("MAX(A1:A4, 10)", &cells).0, num(10.0));
    assert_eq!(eval_with("AVG(B1:B3)", &cells).0, CellValue::Error(DIV0));
    assert_eq!(eval_with("MAX(B1:B3)", &cells).0, num(0.0));
    assert_eq!(eval_with("SUM(A2+1)", &cells).0, CellValue::Error(VALUE));
}

#[test]
fn if_reads_only_the_taken_branch() {
    let cells = [("A1", num(1.0)), ("B1", num(7.0)), ("C1", num(9.0))];
    let (v, read) = eval_with("IF(A1>0, B1, C1)", &cells);
    assert_eq!(v, num(7.0));
    assert_eq!(read, vec![c("A1"), c("B1")]);
    let (v, read) = eval_with("IF(A1==0, B1, C1)", &cells);
    assert_eq!(v, num(9.0));
    assert_eq!(read, vec![c("A1"), c("C1")]);
    let (v, read) = eval_with("IF(1/0, B1, C1)", &cells);
    assert_eq!(v, CellValue::Error(DIV0));
    assert!(read.is_empty());
}

#[test]
fn values_serialize_by_kind_and_nan_equals_nan() {
    let json = |v: &CellValue| serde_json::to_string(v).unwrap();
    assert_eq!(json(&num(1.5)), r#"{"value":1.5}"#);
    assert_eq!(json(&CellValue::Text("hi".into())), r#"{"text":"hi"}"#);
    assert_eq!(json(&CellValue::Error(CIRC)), r##"{"error":"#CIRC!"}"##);
    assert_eq!(json(&CellValue::Empty), "{}");
    assert_eq!(num(f64::NAN), num(f64::NAN));
    assert_ne!(num(0.0), CellValue::Empty);
}
