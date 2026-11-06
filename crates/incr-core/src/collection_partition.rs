use std::hash::Hash;
use std::sync::atomic::{AtomicI64, Ordering as MemOrdering};
use std::sync::Arc;

use super::{Delta, IncrCollection};
use crate::cells::Cells;
use crate::handle::Incr;
use crate::runtime::Runtime;
use crate::value::Value;
use crate::value_code;

impl<T: Value + Hash + Eq, C: Cells> IncrCollection<T, C> {
    /// `(matching, other)`: how many live elements `pred` holds for and how
    /// many it does not, kept as [`Self::count`] keeps its tally. `pred`
    /// runs once per element of the bootstrap snapshot and once per later
    /// delta, as value code and under the upstream log's read lock, so it
    /// must be a pure function of the element that never calls into the
    /// runtime. If it panics, nothing of that pass is counted.
    pub fn count_partition<P>(&self, rt: &Runtime<C>, pred: P) -> Incr<(u64, u64)>
    where
        P: Fn(&T) -> bool + Send + Sync + 'static,
    {
        value_code::check();
        let upstream_log = Arc::clone(&self.log);
        let upstream_version = self.version_node;

        let (cursor, bootstrap) = upstream_log.write().register_consumer();
        let matching = value_code::run(|| bootstrap.iter().filter(|v| pred(v)).count());
        let running = [
            AtomicI64::new(matching as i64),
            AtomicI64::new((bootstrap.len() - matching) as i64),
        ];
        drop(bootstrap);

        rt.create_operator_query(move |rt| -> (u64, u64) {
            let _uv = rt.get(upstream_version);
            let up = upstream_log.read();
            let from = cursor.load(MemOrdering::Acquire);
            if from < up.end() {
                let mut delta = [0_i64; 2];
                value_code::run(|| {
                    for d in up.pending_from(from) {
                        match d {
                            Delta::Insert(v) => delta[usize::from(!pred(v))] += 1,
                            Delta::Delete(v) => delta[usize::from(!pred(v))] -= 1,
                        }
                    }
                });
                for (count, change) in running.iter().zip(delta) {
                    count.fetch_add(change, MemOrdering::Relaxed);
                }
                cursor.store(up.end(), MemOrdering::Release);
            }
            let [matching, other] = &running;
            (
                matching.load(MemOrdering::Relaxed).max(0) as u64,
                other.load(MemOrdering::Relaxed).max(0) as u64,
            )
        })
    }
}
