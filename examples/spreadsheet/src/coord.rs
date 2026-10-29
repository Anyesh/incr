use std::fmt;

/// A cell address: `col` 0 is column A, `row` 1 is the first row, as
/// the sheet shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Coord {
    pub col: u32,
    pub row: u32,
}

impl Coord {
    /// One more than the largest column, matching incr's `GridKey` bound.
    pub const COLUMNS: u32 = 1 << 16;
    /// One more than the largest row, matching incr's `GridKey` bound.
    pub const ROWS: u32 = 1 << 24;

    pub fn new(col: u32, row: u32) -> Self {
        Self { col, row }
    }

    /// Parses a name such as `B7` or `AA10`, case-insensitively.
    pub fn parse(name: &str) -> Option<Self> {
        let split = name.find(|ch: char| !ch.is_ascii_alphabetic())?;
        let (letters, digits) = name.split_at(split);
        if letters.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mut col: u32 = 0;
        for b in letters.bytes() {
            col = col
                .checked_mul(26)?
                .checked_add(u32::from(b.to_ascii_uppercase() - b'A') + 1)?;
            if col > Self::COLUMNS {
                return None;
            }
        }
        let row: u32 = digits.parse().ok()?;
        (1..Self::ROWS)
            .contains(&row)
            .then_some(Self::new(col - 1, row))
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut letters = Vec::new();
        let mut n = self.col + 1;
        while n > 0 {
            n -= 1;
            letters.push(b'A' + (n % 26) as u8);
            n /= 26;
        }
        letters.reverse();
        write!(f, "{}{}", String::from_utf8_lossy(&letters), self.row)
    }
}
