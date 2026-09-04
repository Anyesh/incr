//! `incr-kube`: keyed-upsert layer over `incr-concurrent`, built for
//! Kubernetes-style watch-event state (replace-by-key, delete-by-key,
//! staleness rejection by version), plus a driver that feeds it from a
//! kube-rs watch stream.
//!
//! ## Scope
//!
//! [`KeyedCollection`] and [`Entry`] are the generic keyed-collection
//! engine layer. [`drive_reflector`] is the kube-rs integration built on
//! top of it: it consumes a raw `watcher()`/`reflector()` event stream
//! directly, not `store_shared()`/`ReflectHandle` (see the [`reflector`]
//! module docs for why). Still not in this crate: the milestone's
//! published 50k-object benchmark against `Store::state_filter`, and a
//! worked conversion of a real public operator; both need a live or
//! synthetic cluster-scale watch feed this crate's own test suite
//! doesn't set up.
//!
//! Hardcoded to `Shared` mode (`incr_concurrent`'s re-export), not
//! generic over strategy: kube-rs controllers are inherently
//! multi-threaded and async, so there's no single-threaded use case to
//! support here.
//!
//! ## Known gaps
//!
//! - **`ListSemantic::Any`**: if the caller configures their `watcher()`
//!   with cached, possibly-stale relists, a relist's `InitApply` can
//!   legitimately carry an older object state than a live `Apply`
//!   already applied for the same key. [`drive_reflector`]'s monotonic
//!   counter applies it as newest regardless, the same way kube's own
//!   `Store` does under that setting; not a regression, but a real
//!   caveat on the versioning design, not something silently assumed
//!   away.
//! - **A relist reconciles `KeyedCollection` only when the caller drives
//!   one.** [`KeyedCollection::retain`] is the primitive that lets a
//!   caller prune drift after an authoritative full listing;
//!   [`drive_reflector`] calls it on every `InitDone`, but a caller
//!   using `KeyedCollection` directly, without `drive_reflector`, gets
//!   no automatic reconciliation.
//!
//! Two gaps named after the previous chunk are now closed rather than
//! deferred again: `resourceVersion` parsing turned out not to be
//! needed at all (a loop-owned monotonic counter is sufficient, since a
//! single watch stream already delivers per-object events in order),
//! and the relist/tombstone staleness gap is closed by
//! [`drive_reflector`]'s `retain`-on-`InitDone` reconciliation.

#![doc(html_no_source)]

mod entry;
mod keyed;
mod reflector;

pub use entry::Entry;
pub use keyed::{KeyedCollection, UpsertOutcome};
pub use reflector::drive_reflector;
