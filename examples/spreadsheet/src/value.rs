use serde::ser::{Serialize, SerializeMap, Serializer};

pub const DIV0: &str = "#DIV/0!";
pub const VALUE: &str = "#VALUE!";
pub const CIRC: &str = "#CIRC!";
pub const PARSE: &str = "#ERR";

#[derive(Clone, Debug)]
pub enum CellValue {
    Empty,
    Number(f64),
    Text(String),
    Error(&'static str),
}

/// NaN equals NaN, so a cell stuck at NaN still counts as unchanged.
impl PartialEq for CellValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Empty, Self::Empty) => true,
            (Self::Number(a), Self::Number(b)) => a == b || (a.is_nan() && b.is_nan()),
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Error(a), Self::Error(b)) => a == b,
            _ => false,
        }
    }
}

/// `{"value": n}`, `{"text": s}`, `{"error": code}`, or `{}` for an empty
/// cell, so a message can flatten it next to the cell name.
impl Serialize for CellValue {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        match self {
            Self::Empty => {}
            Self::Number(n) => map.serialize_entry("value", n)?,
            Self::Text(s) => map.serialize_entry("text", s)?,
            Self::Error(code) => map.serialize_entry("error", code)?,
        }
        map.end()
    }
}
