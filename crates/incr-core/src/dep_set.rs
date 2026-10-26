//! Membership tests over the dep lists `publish_deps` diffs. A changed
//! list can hold tens of thousands of entries, so a scan per lookup
//! would make the diff quadratic.

use std::collections::HashSet;

use crate::fx::FxBuildHasher;
use crate::node::NodeId;

/// While either list is at most this long, a diff by scans is linear in
/// the other and beats building hash sets.
pub(crate) const SCAN_MAX: usize = 16;

#[inline]
pub(crate) fn needs_hashing(old_len: usize, new_len: usize) -> bool {
    old_len > SCAN_MAX && new_len > SCAN_MAX
}

pub(crate) trait DepLookup {
    fn has(&self, dep: NodeId) -> bool;
}

impl DepLookup for [NodeId] {
    #[inline]
    fn has(&self, dep: NodeId) -> bool {
        self.contains(&dep)
    }
}

pub(crate) struct DepHashSet(HashSet<NodeId, FxBuildHasher>);

impl DepHashSet {
    pub(crate) fn new(deps: &[NodeId]) -> Self {
        DepHashSet(deps.iter().copied().collect())
    }
}

impl DepLookup for DepHashSet {
    #[inline]
    fn has(&self, dep: NodeId) -> bool {
        self.0.contains(&dep)
    }
}

/// A node's installed deps and the ones it is about to install, each with
/// the lookup that answers membership in it.
pub(crate) struct DepDiff<'a, S: ?Sized> {
    pub(crate) old: &'a [NodeId],
    pub(crate) new: &'a [NodeId],
    pub(crate) old_set: &'a S,
    pub(crate) new_set: &'a S,
}

/// `old` as it is, then each recorded dep not already present, once, in
/// recorded order.
#[inline]
pub(crate) fn union(old: &[NodeId], recorded: &[(NodeId, u32)]) -> Vec<NodeId> {
    // Every recorded dep scans the growing union, not only `old`.
    if old.len() + recorded.len() > SCAN_MAX {
        return hashed_union(old, recorded);
    }
    let mut union = old.to_vec();
    for &(dep, _) in recorded {
        if !union.contains(&dep) {
            union.push(dep);
        }
    }
    union
}

#[cold]
#[inline(never)]
fn hashed_union(old: &[NodeId], recorded: &[(NodeId, u32)]) -> Vec<NodeId> {
    let mut union = old.to_vec();
    let mut have: HashSet<NodeId, FxBuildHasher> = old.iter().copied().collect();
    for &(dep, _) in recorded {
        if have.insert(dep) {
            union.push(dep);
        }
    }
    union
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(raw: impl IntoIterator<Item = u32>) -> Vec<NodeId> {
        raw.into_iter().map(NodeId).collect()
    }

    #[test]
    fn hashed_lookup_agrees_with_the_scan() {
        let deps = ids((0..500).map(|x| x * 3));
        let set = DepHashSet::new(&deps);
        for probe in 0..1502 {
            assert_eq!(set.has(NodeId(probe)), deps[..].has(NodeId(probe)));
        }
    }

    #[test]
    fn union_matches_the_scan_on_both_sides_of_the_threshold() {
        for (old_len, new_len) in [
            (3_u32, 4_u32),
            (SCAN_MAX as u32, 100),
            (100, SCAN_MAX as u32),
            (40, 60),
            (0, 50),
        ] {
            let old = ids((0..old_len).map(|x| x % 7 * 2));
            let recorded: Vec<(NodeId, u32)> = (0..new_len)
                .map(|x| (NodeId((new_len - x) % 11), 0))
                .collect();
            let mut expected = old.clone();
            for &(dep, _) in &recorded {
                if !expected.contains(&dep) {
                    expected.push(dep);
                }
            }
            assert_eq!(union(&old, &recorded), expected);
            assert_eq!(hashed_union(&old, &recorded), expected);
        }
    }
}
