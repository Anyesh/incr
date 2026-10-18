//! `NodeTable` against a `HashMap` model: random sequences of inserts,
//! sets, removes, reads and walks over keyed graphs whose queries create
//! the inputs they read through the table, under `Local` and `Shared`.
//! A query reads only smaller keys, so graphs stay acyclic and at most as
//! deep as the key range.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use incr_core::{Cells, DeleteError, Local, NodeTable, Runtime, Shared};
use proptest::prelude::*;

const KEYS: u64 = 24;

#[derive(Clone, Debug)]
enum Op {
    Input(u64, i64),
    Query(u64, Vec<u64>),
    Set(u64, i64),
    Remove(u64),
    Get(u64),
    ForEach,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0..KEYS, -50..50_i64).prop_map(|(k, v)| Op::Input(k, v)),
        (1..KEYS, prop::collection::vec(0..KEYS, 0..4))
            .prop_map(|(k, reads)| Op::Query(k, reads.into_iter().map(|r| r % k).collect())),
        (0..KEYS, -50..50_i64).prop_map(|(k, v)| Op::Set(k, v)),
        (0..KEYS).prop_map(Op::Remove),
        (0..KEYS).prop_map(Op::Get),
        Just(Op::ForEach),
    ]
}

#[derive(Clone, Debug)]
enum Entry {
    Input(i64),
    Query { reads: Vec<u64>, computed: bool },
}

#[derive(Default)]
struct Model {
    entries: HashMap<u64, Entry>,
}

impl Model {
    /// A query's value, creating each input it reads that is missing,
    /// as its closure does, and marking every query it computes.
    fn value_of(&mut self, k: u64) -> i64 {
        match self.entries.get(&k).cloned() {
            None => {
                self.entries.insert(k, Entry::Input(0));
                0
            }
            Some(Entry::Input(v)) => v,
            Some(Entry::Query { reads, .. }) => {
                let sum = 1 + reads.iter().map(|&r| self.value_of(r)).sum::<i64>();
                self.entries.insert(
                    k,
                    Entry::Query {
                        reads,
                        computed: true,
                    },
                );
                sum
            }
        }
    }

    fn has_dependents(&self, k: u64) -> bool {
        self.entries
            .values()
            .any(|e| matches!(e, Entry::Query { reads, computed: true } if reads.contains(&k)))
    }
}

fn run<C: Cells>(ops: &[Op])
where
    NodeTable<u64, i64, C>: Send + Sync,
{
    let rt = Runtime::<C>::new();
    let table = Arc::new(rt.create_table::<u64, i64>());
    let mut model = Model::default();
    for op in ops {
        match op {
            Op::Input(k, v) => {
                let h = table.get_or_insert_input(&rt, *k, || *v);
                model.entries.entry(*k).or_insert(Entry::Input(*v));
                assert_eq!(table.get(&rt, k), Some(h));
            }
            Op::Query(k, reads) => {
                let t = Arc::clone(&table);
                let r = reads.clone();
                table.get_or_insert_query(&rt, *k, move |rt| {
                    1 + r
                        .iter()
                        .map(|&key| rt.get(t.get_or_insert_input(rt, key, || 0)))
                        .sum::<i64>()
                });
                model.entries.entry(*k).or_insert(Entry::Query {
                    reads: reads.clone(),
                    computed: false,
                });
            }
            Op::Set(k, v) => {
                if let Some(Entry::Input(old)) = model.entries.get_mut(k) {
                    *old = *v;
                    rt.set(table.get(&rt, k).expect("a model input is live"), *v);
                }
            }
            Op::Remove(k) => {
                let expected = match model.entries.get(k) {
                    None => Err(DeleteError::NotAlive),
                    Some(_) if model.has_dependents(*k) => Err(DeleteError::HasDependents),
                    Some(_) => Ok(()),
                };
                assert_eq!(table.remove(&rt, k), expected, "remove {k}");
                if expected.is_ok() {
                    model.entries.remove(k);
                    assert_eq!(table.get(&rt, k), None);
                }
            }
            Op::Get(k) => {
                if model.entries.contains_key(k) {
                    let expected = model.value_of(*k);
                    let h = table.get(&rt, k).expect("a model entry is live");
                    assert_eq!(rt.get(h), expected, "value of {k}");
                }
            }
            Op::ForEach => {
                let mut seen = Vec::new();
                table.for_each(&rt, |k, _| seen.push(*k));
                let unique: HashSet<u64> = seen.iter().copied().collect();
                assert_eq!(unique.len(), seen.len(), "no key twice");
                assert_eq!(
                    unique,
                    model.entries.keys().copied().collect::<HashSet<_>>()
                );
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    #[cfg_attr(miri, ignore)]
    fn local_matches_the_model(ops in prop::collection::vec(op(), 1..120)) {
        run::<Local>(&ops);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn shared_matches_the_model(ops in prop::collection::vec(op(), 1..120)) {
        run::<Shared>(&ops);
    }
}
