use std::collections::HashMap;

use crate::coord::Coord;
use crate::formula::{parse_content, Content};
use crate::value::{CellValue, CIRC};
use crate::Sheet;

/// What a non-incremental engine does: every recalc evaluates every cell
/// in precedent order. The order and the cells on a cycle come from the
/// formulas' static references and are recomputed only after an edit
/// adds, removes or changes a formula.
#[derive(Default)]
pub struct FullEngine {
    contents: HashMap<Coord, Content>,
    values: HashMap<Coord, CellValue>,
    cleared: Vec<Coord>,
    plan: Option<Plan>,
}

/// Cells in evaluation order, each with whether it sits on a cycle.
struct Plan {
    order: Vec<(Coord, bool)>,
}

impl FullEngine {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Sheet for FullEngine {
    fn set(&mut self, at: Coord, raw: &str) {
        let content = parse_content(raw);
        let reshapes = matches!(content, Content::Formula(_) | Content::Empty);
        let old = if content == Content::Empty {
            self.cleared.push(at);
            self.contents.remove(&at)
        } else {
            self.contents.insert(at, content)
        };
        if reshapes || !matches!(old, Some(Content::Number(_) | Content::Text(_))) {
            self.plan = None;
        }
    }

    fn recalc(&mut self) -> Vec<(Coord, CellValue)> {
        let mut changed = Vec::new();
        for at in std::mem::take(&mut self.cleared) {
            if !self.contents.contains_key(&at) && self.values.remove(&at).is_some() {
                changed.push((at, CellValue::Empty));
            }
        }
        let plan = self.plan.take().unwrap_or_else(|| plan(&self.contents));
        for &(at, on_cycle) in &plan.order {
            let new = if on_cycle {
                CellValue::Error(CIRC)
            } else {
                let values = &self.values;
                self.contents[&at]
                    .evaluate(&mut |r| values.get(&r).cloned().unwrap_or(CellValue::Empty))
            };
            if self.values.get(&at) != Some(&new) {
                changed.push((at, new.clone()));
                self.values.insert(at, new);
            }
        }
        self.plan = Some(plan);
        changed
    }

    fn value(&self, at: Coord) -> CellValue {
        self.values.get(&at).cloned().unwrap_or(CellValue::Empty)
    }
}

/// Tarjan's strongly connected components over the precedent graph,
/// iterative so a 100,000-cell chain cannot overflow the stack. Tarjan
/// finishes a component only after every component it reaches, and
/// edges run from a cell to its precedents, so finishing order is
/// evaluation order.
fn plan(contents: &HashMap<Coord, Content>) -> Plan {
    let cells: Vec<Coord> = contents.keys().copied().collect();
    let index: HashMap<Coord, usize> = cells.iter().enumerate().map(|(i, &c)| (c, i)).collect();
    let edges: Vec<Vec<usize>> = cells
        .iter()
        .map(|c| {
            contents[c]
                .precedents()
                .iter()
                .filter_map(|p| index.get(p).copied())
                .collect()
        })
        .collect();

    const UNSEEN: usize = usize::MAX;
    let n = cells.len();
    let mut order = Vec::with_capacity(n);
    let mut num = vec![UNSEEN; n];
    let mut low = vec![0; n];
    let mut on_stack = vec![false; n];
    let mut stack = Vec::new();
    let mut walk: Vec<(usize, usize)> = Vec::new();
    let mut next = 0;

    for root in 0..n {
        if num[root] != UNSEEN {
            continue;
        }
        walk.push((root, 0));
        while let Some(&mut (v, ref mut edge)) = walk.last_mut() {
            if *edge == 0 {
                num[v] = next;
                low[v] = next;
                next += 1;
                stack.push(v);
                on_stack[v] = true;
            }
            if let Some(&w) = edges[v].get(*edge) {
                *edge += 1;
                if num[w] == UNSEEN {
                    walk.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(num[w]);
                }
                continue;
            }
            walk.pop();
            if let Some(&(parent, _)) = walk.last() {
                low[parent] = low[parent].min(low[v]);
            }
            if low[v] == num[v] {
                let start = stack
                    .iter()
                    .rposition(|&x| x == v)
                    .expect("v is on the stack");
                let component = stack.split_off(start);
                let on_cycle = component.len() > 1 || edges[v].contains(&v);
                for w in component {
                    on_stack[w] = false;
                    order.push((cells[w], on_cycle));
                }
            }
        }
    }
    Plan { order }
}
