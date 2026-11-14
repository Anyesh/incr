//! Which live entries a checkpoint still has to save.

use std::collections::HashMap;

use incr_core::ErasedIncr;

/// Per table, the live entries computed since a checkpoint last saved
/// them, each with the mark of its latest compute. Marks rise by one per
/// compute; a checkpoint attempt notes the mark it started at, so an entry
/// whose mark is above it was computed since that attempt.
pub(crate) struct Dirty {
    marks: u64,
    attempt: u64,
    unsaved: u64,
    tables: Vec<HashMap<ErasedIncr, u64>>,
}

impl Dirty {
    pub(crate) fn new(tables: usize) -> Self {
        Dirty {
            marks: 0,
            attempt: 0,
            unsaved: 0,
            tables: vec![HashMap::new(); tables],
        }
    }

    /// Records a compute of `handle`; returns the unsaved count after it.
    pub(crate) fn computed(&mut self, table: u32, handle: ErasedIncr) -> u64 {
        self.marks += 1;
        let previous = self.tables[table as usize].insert(handle, self.marks);
        if previous.is_none_or(|mark| mark <= self.attempt) {
            self.unsaved += 1;
        }
        self.unsaved
    }

    /// Keeps `handle` dirty without counting it as unsaved: an entry the
    /// backend lost rather than one that computed.
    pub(crate) fn lost(&mut self, table: u32, handle: ErasedIncr) {
        self.tables[table as usize].entry(handle).or_insert(0);
    }

    /// Starts a checkpoint attempt: drops entries `alive` refuses and
    /// returns a copy of the rest.
    pub(crate) fn attempt(
        &mut self,
        alive: impl Fn(ErasedIncr) -> bool,
    ) -> Vec<HashMap<ErasedIncr, u64>> {
        self.attempt = self.marks;
        self.unsaved = 0;
        for table in &mut self.tables {
            table.retain(|&handle, _| alive(handle));
        }
        self.tables.clone()
    }

    /// `handle` reached the backend as of `mark`; a later compute keeps it.
    pub(crate) fn saved(&mut self, table: u32, handle: ErasedIncr, mark: u64) {
        let table = &mut self.tables[table as usize];
        if table.get(&handle) == Some(&mark) {
            table.remove(&handle);
        }
    }

    pub(crate) fn unsaved(&self) -> u64 {
        self.unsaved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use incr_core::{Local, Runtime};

    fn handles(n: u64) -> (Runtime<Local>, Vec<ErasedIncr>) {
        let rt = Runtime::<Local>::new();
        let handles = (0..n).map(|i| rt.create_input(i).erase()).collect();
        (rt, handles)
    }

    #[test]
    fn unsaved_counts_entries_computed_since_the_attempt() {
        let (_rt, h) = handles(3);
        let mut dirty = Dirty::new(1);
        assert_eq!(dirty.computed(0, h[0]), 1);
        assert_eq!(dirty.computed(0, h[0]), 1);
        assert_eq!(dirty.computed(0, h[1]), 2);
        let marks = dirty.attempt(|_| true);
        assert_eq!(dirty.unsaved(), 0);
        assert_eq!(dirty.computed(0, h[0]), 1);
        dirty.lost(0, h[2]);
        assert_eq!(dirty.unsaved(), 1);
        dirty.saved(0, h[0], marks[0][&h[0]]);
        dirty.saved(0, h[1], marks[0][&h[1]]);
        let left = dirty.attempt(|_| true);
        assert_eq!(left[0].len(), 2);
        assert!(left[0].contains_key(&h[0]) && left[0].contains_key(&h[2]));
    }
}
