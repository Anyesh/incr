use std::collections::HashMap;
use std::sync::Arc;

use incr_concurrent::{
    CycleStrategy, ErasedIncr, GridKey, GridTable, Incr, QueryOpts, Replay, Runtime,
};

use crate::coord::Coord;
use crate::formula::{parse_content, Content};
use crate::value::{CellValue, CIRC};
use crate::Sheet;

/// The incremental engine. Each cell is three nodes: its raw content, the
/// parse of it, and its value. A value reads the cells its formula
/// names, creating their nodes on first use, so the grid has no fixed
/// size and an edit recomputes only the values it reaches; an edit that
/// leaves a parse or a value equal stops there.
pub struct Engine {
    rt: Runtime,
    tables: Arc<Tables>,
    watched: HashMap<ErasedIncr, Coord>,
    shown: HashMap<Coord, CellValue>,
}

struct Tables {
    raw: GridTable<String>,
    parsed: GridTable<Arc<Content>>,
    values: GridTable<CellValue>,
}

fn key(at: Coord) -> GridKey {
    GridKey::new(0, at.row, at.col)
}

impl Tables {
    fn raw(&self, rt: &Runtime, at: Coord) -> Incr<String> {
        self.raw.get_or_insert_input(rt, key(at), String::new)
    }

    fn parsed(self: &Arc<Self>, rt: &Runtime, at: Coord) -> Incr<Arc<Content>> {
        self.parsed.get_or_insert_query_lazy(rt, key(at), || {
            let tables = Arc::clone(self);
            let parse = move |rt: &Runtime| Arc::new(parse_content(&rt.get(tables.raw(rt, at))));
            (parse, QueryOpts::default())
        })
    }

    /// A cycle lands every member at `#CIRC!`, and a chain deeper than
    /// the frame budget replays instead of overflowing the stack.
    fn value(self: &Arc<Self>, rt: &Runtime, at: Coord) -> Incr<CellValue> {
        self.values.get_or_insert_query_lazy(rt, key(at), || {
            let tables = Arc::clone(self);
            let compute = move |rt: &Runtime| {
                let content = rt.get(tables.parsed(rt, at));
                content.evaluate(&mut |r| rt.get(tables.value(rt, r)))
            };
            let opts = QueryOpts {
                cycle: CycleStrategy::fallback(|_| CellValue::Error(CIRC)),
                replay: Replay::Allowed,
            };
            (compute, opts)
        })
    }
}

impl Engine {
    pub fn new() -> Self {
        let rt = Runtime::new();
        // A compute cut short for replay reads, and makes table misses
        // that return, these placeholders until it reruns.
        rt.set_placeholder(String::new());
        rt.set_placeholder(Arc::new(Content::Empty));
        rt.set_placeholder(CellValue::Empty);
        let tables = Arc::new(Tables {
            raw: rt.create_grid_table(),
            parsed: rt.create_grid_table(),
            values: rt.create_grid_table(),
        });
        Self {
            rt,
            tables,
            watched: HashMap::new(),
            shown: HashMap::new(),
        }
    }

    /// The cell's raw content as last set.
    pub fn content(&self, at: Coord) -> String {
        self.tables
            .raw
            .get(&self.rt, key(at))
            .map_or_else(String::new, |raw| self.rt.get(raw))
    }

    pub fn node_count(&self) -> usize {
        self.rt.node_count()
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Sheet for Engine {
    /// Watches the cell's value, so `recalc` hears of every change to it.
    fn set(&mut self, at: Coord, raw: &str) {
        let input = self.tables.raw(&self.rt, at);
        self.rt.set(input, raw.to_string());
        let value = self.tables.value(&self.rt, at);
        if self.watched.insert(value.erase(), at).is_none() {
            self.rt.watch(value);
        }
    }

    /// Reads only the watched cells an edit dirtied since the last
    /// recalc; a dirtied cell whose value came back equal is not
    /// returned.
    fn recalc(&mut self) -> Vec<(Coord, CellValue)> {
        let mut changed = Vec::new();
        for node in self.rt.drain_dirtied_watched() {
            let at = self.watched[&node];
            let value = self.rt.get(self.tables.value(&self.rt, at));
            let old = self.shown.get(&at).unwrap_or(&CellValue::Empty);
            if *old != value {
                changed.push((at, value.clone()));
                self.shown.insert(at, value);
            }
        }
        changed
    }

    fn value(&self, at: Coord) -> CellValue {
        self.shown.get(&at).cloned().unwrap_or(CellValue::Empty)
    }
}
