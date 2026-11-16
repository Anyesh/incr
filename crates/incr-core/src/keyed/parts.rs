//! Key-set partitions: a user function groups a collection's keys, and each
//! partition keeps its own pairs inside the key table, so a query reading
//! one partition depends on that partition alone.

use std::any::Any;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::sync::Arc;

use super::Pair;
use crate::fx::FxBuildHasher;
use crate::value::Value;

/// A key whose presence changed in one write: the pair that is read for it
/// (inserted) or one it held (removed), and its index hash.
pub(super) struct Flip<'a, P> {
    pub pair: &'a P,
    pub hash: u64,
    pub inserted: bool,
}

/// One partitioning inside a [`super::KeyTable`], changed only under the
/// table's write lock.
pub(super) trait PartSink<P>: Send + Sync {
    /// Runs the partition function for each of `flips`, as value code, and
    /// finds where each lands, changing nothing: the plan is valid for the
    /// state it was made against.
    fn plan(&self, flips: &[Flip<'_, P>]) -> Box<dyn Any + Send>;
    /// The hashes of the partitions `plan` touches.
    fn touched(&self, plan: &dyn Any) -> Vec<u64>;
    /// Applies a plan made against this state and drops every partition it
    /// leaves without a key. Runs no user code; the pairs it lets go go to
    /// `gone`. Returns the hashes that no record has any more, whose inputs
    /// no write can bump until a partition of that hash returns.
    fn apply(&mut self, plan: Box<dyn Any + Send>, gone: &mut Vec<P>) -> Vec<u64>;
    /// Runs the partition function on `pair`'s key, discarding the result.
    fn check(&self, pair: &P);
    /// How many partition records the partitioning holds.
    fn records(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
}

struct Part<Q, P> {
    q: Q,
    qhash: u64,
    pairs: HashMap<u64, Vec<P>, FxBuildHasher>,
}

pub(super) struct PartState<K, V, Q, P> {
    f: Arc<dyn Fn(&K) -> Q + Send + Sync>,
    parts: Vec<Part<Q, P>>,
    by_qhash: HashMap<u64, Vec<usize>, FxBuildHasher>,
    _row: PhantomData<fn() -> V>,
}

enum Slot {
    Old(usize),
    New(usize),
}

enum Op<P> {
    Insert { part: Slot, hash: u64, pair: P },
    Remove { part: usize, hash: u64, at: usize },
}

struct Plan<Q, P> {
    fresh: Vec<(Q, u64)>,
    ops: Vec<Op<P>>,
}

impl<K, V, Q, P> PartState<K, V, Q, P>
where
    K: Value + Hash + Eq,
    V: Value + Hash + Eq,
    Q: Value + Hash + Eq,
    P: Pair<K, V>,
{
    /// The partitions of the pairs `existing` yields, one per key.
    pub(super) fn build<'a>(
        f: Arc<dyn Fn(&K) -> Q + Send + Sync>,
        existing: impl Iterator<Item = (u64, &'a P)>,
    ) -> Self
    where
        P: 'a,
    {
        let mut state = Self {
            f,
            parts: Vec::new(),
            by_qhash: HashMap::default(),
            _row: PhantomData,
        };
        for (hash, pair) in existing {
            let q = (state.f)(&pair.key());
            let qhash = FxBuildHasher.hash_one(&q);
            let at = match state.find(qhash, &q) {
                Some(at) => at,
                None => state.push(q, qhash),
            };
            state.parts[at]
                .pairs
                .entry(hash)
                .or_default()
                .push(pair.clone());
        }
        state
    }

    fn push(&mut self, q: Q, qhash: u64) -> usize {
        self.parts.push(Part {
            q,
            qhash,
            pairs: HashMap::default(),
        });
        let at = self.parts.len() - 1;
        self.by_qhash.entry(qhash).or_default().push(at);
        at
    }

    /// Removes the record at `at` by moving the last record into its place,
    /// and returns its hash.
    fn drop_record(&mut self, at: usize) -> u64 {
        let qhash = self.parts[at].qhash;
        let bucket = self
            .by_qhash
            .get_mut(&qhash)
            .expect("incr-core: a record is indexed");
        bucket.retain(|&i| i != at);
        if bucket.is_empty() {
            self.by_qhash.remove(&qhash);
        }
        let last = self.parts.len() - 1;
        self.parts.swap_remove(at);
        if at != last {
            let moved = self.parts[at].qhash;
            for i in self
                .by_qhash
                .get_mut(&moved)
                .expect("incr-core: a record is indexed")
            {
                if *i == last {
                    *i = at;
                }
            }
        }
        qhash
    }

    fn find(&self, qhash: u64, q: &Q) -> Option<usize> {
        self.by_qhash
            .get(&qhash)?
            .iter()
            .copied()
            .find(|&at| self.parts[at].q == *q)
    }

    /// A copy of each key in partition `q`, in no particular order.
    pub(super) fn keys_of(&self, q: &Q) -> Vec<K> {
        let qhash = FxBuildHasher.hash_one(q);
        let Some(at) = self.find(qhash, q) else {
            return Vec::new();
        };
        self.parts[at]
            .pairs
            .values()
            .flatten()
            .map(Pair::key)
            .collect()
    }
}

impl<K, V, Q, P> PartSink<P> for PartState<K, V, Q, P>
where
    K: Value + Hash + Eq,
    V: Value + Hash + Eq,
    Q: Value + Hash + Eq,
    P: Pair<K, V>,
{
    fn plan(&self, flips: &[Flip<'_, P>]) -> Box<dyn Any + Send> {
        let mut plan: Plan<Q, P> = Plan {
            fresh: Vec::new(),
            ops: Vec::new(),
        };
        for flip in flips {
            let key = flip.pair.key();
            let q = (self.f)(&key);
            let qhash = FxBuildHasher.hash_one(&q);
            if flip.inserted {
                let slot = match self.find(qhash, &q) {
                    Some(at) => Slot::Old(at),
                    None => match plan
                        .fresh
                        .iter()
                        .position(|(fresh, h)| *h == qhash && *fresh == q)
                    {
                        Some(j) => Slot::New(j),
                        None => {
                            plan.fresh.push((q, qhash));
                            Slot::New(plan.fresh.len() - 1)
                        }
                    },
                };
                plan.ops.push(Op::Insert {
                    part: slot,
                    hash: flip.hash,
                    pair: flip.pair.clone(),
                });
            } else if let Some(part) = self.find(qhash, &q) {
                let at = self.parts[part].pairs.get(&flip.hash).and_then(|bucket| {
                    if bucket.len() == 1 {
                        Some(0)
                    } else {
                        bucket.iter().position(|p| p.key_eq(&key))
                    }
                });
                if let Some(at) = at {
                    plan.ops.push(Op::Remove {
                        part,
                        hash: flip.hash,
                        at,
                    });
                }
            }
        }
        Box::new(plan)
    }

    fn touched(&self, plan: &dyn Any) -> Vec<u64> {
        let plan = plan
            .downcast_ref::<Plan<Q, P>>()
            .expect("incr-core: a partition plan fits its partitioning");
        plan.ops
            .iter()
            .map(|op| match op {
                Op::Insert {
                    part: Slot::Old(at),
                    ..
                }
                | Op::Remove { part: at, .. } => self.parts[*at].qhash,
                Op::Insert {
                    part: Slot::New(j), ..
                } => plan.fresh[*j].1,
            })
            .collect()
    }

    fn apply(&mut self, plan: Box<dyn Any + Send>, gone: &mut Vec<P>) -> Vec<u64> {
        let Plan { fresh, ops } = *plan
            .downcast::<Plan<Q, P>>()
            .expect("incr-core: a partition plan fits its partitioning");
        let base = self.parts.len();
        for (q, qhash) in fresh {
            self.push(q, qhash);
        }
        let mut removes = Vec::new();
        let mut inserts = Vec::new();
        for op in ops {
            match op {
                Op::Remove { part, hash, at } => removes.push((part, hash, at)),
                Op::Insert { part, hash, pair } => inserts.push((part, hash, pair)),
            }
        }
        removes.sort_unstable_by(|a, b| b.cmp(a));
        let mut shrunk: Vec<usize> = removes.iter().map(|&(part, _, _)| part).collect();
        shrunk.dedup();
        for (part, hash, at) in removes {
            let pairs = &mut self.parts[part].pairs;
            if let Some(bucket) = pairs.get_mut(&hash) {
                if at < bucket.len() {
                    gone.push(bucket.swap_remove(at));
                }
                if bucket.is_empty() {
                    pairs.remove(&hash);
                }
            }
        }
        for (part, hash, pair) in inserts {
            let at = match part {
                Slot::Old(at) => at,
                Slot::New(j) => base + j,
            };
            self.parts[at].pairs.entry(hash).or_default().push(pair);
        }
        shrunk.retain(|&at| self.parts[at].pairs.is_empty());
        let mut emptied = Vec::new();
        for at in shrunk {
            let qhash = self.drop_record(at);
            if !self.by_qhash.contains_key(&qhash) {
                emptied.push(qhash);
            }
        }
        emptied
    }

    fn check(&self, pair: &P) {
        let q = (self.f)(&pair.key());
        FxBuildHasher.hash_one(&q);
    }

    fn records(&self) -> usize {
        self.parts.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
