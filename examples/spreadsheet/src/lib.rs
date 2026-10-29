pub mod coord;
pub mod engine;
pub mod formula;
pub mod full;
pub mod value;

use coord::Coord;
use value::CellValue;

/// A spreadsheet engine: edits are applied by `set` and take effect at
/// the next `recalc`.
pub trait Sheet {
    fn set(&mut self, at: Coord, raw: &str);

    /// Brings every cell up to date and returns each cell whose value
    /// differs from the last recalc, a cleared cell as `Empty`.
    fn recalc(&mut self) -> Vec<(Coord, CellValue)>;

    /// The cell's value as of the last recalc.
    fn value(&self, at: Coord) -> CellValue;
}
