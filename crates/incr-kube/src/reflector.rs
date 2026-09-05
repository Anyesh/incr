//! Drives a `KeyedCollection` from a raw kube-rs watch/reflector event
//! stream.
//!
//! Deliberately does *not* use `kube::runtime::reflector::ReflectHandle`
//! (`store_shared()`'s subscriber stream): reading kube-runtime's own
//! `Writer::dispatch_event` shows it only ever broadcasts on `Apply` and
//! `InitDone`, never `Delete`, so a `ReflectHandle` stream would silently
//! lose every delete. This module consumes the stream `reflector()`
//! itself wraps instead (`Stream<Item = Result<watcher::Event<K>,
//! watcher::Error>>`), which carries every event kube-rs's own `Store`
//! sees, and never touches kube's `Store`, `Writer`, or the
//! `unstable-runtime-subscribe` feature.

use std::collections::HashSet;
use std::hash::Hash;

use futures::{Stream, StreamExt};
use incr_concurrent::Runtime;
use kube::runtime::reflector::ObjectRef;
use kube::runtime::watcher;
use kube::{Resource, ResourceExt};

use crate::keyed::{KeyedCollection, UpsertOutcome};

/// Consume `stream` and drive `keyed` to match it, until the stream
/// ends.
///
/// `stream` is anything shaped like `watcher()`'s output; production
/// callers build one against a live `Api<K>` and pass it straight in,
/// tests pass a synthetic `futures::stream::iter([...])`, mirroring
/// kube-runtime's own test pattern for the same stream type. `on_event`
/// is invoked once per successfully received event, after `keyed` has
/// already been updated for it: a caller that needs to log every event
/// or detect `InitDone` to signal its own readiness (both common in a
/// real controller's startup sequence) has a hook for it without
/// needing its own copy of this loop. `on_error` is invoked for every
/// `Err` the stream yields instead, and does not also trigger
/// `on_event`; the loop keeps polling afterward rather than returning,
/// matching how a production controller logs and continues past a
/// transient watch error. Note that `drive_reflector` applies no
/// backoff of its own: if `stream` isn't wrapped with
/// `.default_backoff()` (or equivalent) by the caller, a persistent
/// error spins this loop hot with no delay.
///
/// Versioning is a loop-owned monotonic counter, not a parse of
/// Kubernetes' `resourceVersion` string: a single watch stream already
/// delivers events for a given object in one ordered sequence, so a
/// counter incremented once per processed event is a correct and
/// sufficient `Ver` for `KeyedCollection::upsert`'s replace machinery,
/// without relying on cross-event ordering the Kubernetes API doesn't
/// actually guarantee. One caveat: under a `watcher()` configured with
/// `ListSemantic::Any`, a relist's `InitApply` can legitimately carry
/// an older object state than a live `Apply` already applied for the
/// same key; the counter applies it as newest regardless, the same way
/// kube's own `Store` does under that setting.
///
/// The counter seeds from `keyed.max_version()` rather than `0`: a
/// caller that builds a fresh stream and calls `drive_reflector` again
/// against the same `keyed` (a watcher restart after reconfiguring the
/// `Api`, say) would otherwise have every event rejected as `Stale`
/// until the new counter climbed back past whatever high-water mark the
/// previous call left behind.
///
/// A relist (`Init`, then some `InitApply`, then `InitDone`) reconciles
/// `keyed` against the fresh listing via `KeyedCollection::retain`:
/// anything not observed in the relist is pruned, per `InitDone`'s own
/// documented contract. The seen-set is reset on every `Init`, not just
/// the first, since kube-rs's watcher state machine re-emits `Init`
/// after a failed or restarted relist attempt.
pub async fn drive_reflector<K, S>(
    keyed: &KeyedCollection<ObjectRef<K>, K>,
    rt: &Runtime,
    dyntype: K::DynamicType,
    mut stream: S,
    mut on_event: impl FnMut(&watcher::Event<K>),
    mut on_error: impl FnMut(watcher::Error),
) where
    K: Resource + Clone + Send + Sync + 'static,
    K::DynamicType: Clone + Eq + Hash + Send + Sync + 'static,
    S: Stream<Item = Result<watcher::Event<K>, watcher::Error>> + Unpin,
{
    let mut seen: Option<HashSet<ObjectRef<K>>> = None;
    let mut counter: u64 = keyed.max_version().unwrap_or(0);

    while let Some(event) = stream.next().await {
        match event {
            Ok(ev) => {
                // Matches on `&ev` (cloning `obj` into `apply`/`upsert`
                // rather than moving it out of `ev`) so `ev` is still
                // whole for `on_event` below, after the update it
                // describes has actually landed in `keyed`.
                match &ev {
                    watcher::Event::Apply(obj) => {
                        let key = ObjectRef::from_obj_with(obj, dyntype.clone());
                        apply(keyed, rt, &mut counter, key, obj.clone());
                    }
                    watcher::Event::Delete(obj) => {
                        let key = ObjectRef::from_obj_with(obj, dyntype.clone());
                        keyed.remove(rt, &key);
                    }
                    watcher::Event::Init => {
                        seen = Some(HashSet::new());
                    }
                    watcher::Event::InitApply(obj) => {
                        let key = ObjectRef::from_obj_with(obj, dyntype.clone());
                        // Unconditional: recorded as seen regardless of
                        // whether `apply` below skips the upsert as
                        // unchanged. Skipping this insert for unchanged
                        // objects would mean `retain` prunes everything
                        // InitDone didn't re-upsert, which on a healthy
                        // cluster is nearly the whole collection.
                        if let Some(seen) = seen.as_mut() {
                            seen.insert(key.clone());
                        }
                        apply(keyed, rt, &mut counter, key, obj.clone());
                    }
                    watcher::Event::InitDone => {
                        if let Some(seen) = seen.take() {
                            keyed.retain(rt, |k| seen.contains(k));
                        }
                    }
                }
                on_event(&ev);
            }
            Err(err) => on_error(err),
        }
    }
}

/// Shared `Apply`/`InitApply` path: skip the upsert when the incoming
/// object's `resourceVersion` string is present and equal to what's
/// already indexed (a routine relist re-listing unchanged objects would
/// otherwise still push a full delete+insert through every downstream
/// operator for zero real change). `None` on either side never counts
/// as equal: a real watch stream never emits an object with an empty
/// `resourceVersion` (it surfaces that as `Err(NoResourceVersion)`
/// before reaching here), so `None == None` would only ever fire on a
/// test fixture that forgot to set one, silently skipping every apply
/// in that test.
fn apply<K>(
    keyed: &KeyedCollection<ObjectRef<K>, K>,
    rt: &Runtime,
    counter: &mut u64,
    key: ObjectRef<K>,
    obj: K,
) where
    K: Resource + Clone + Send + Sync + 'static,
    K::DynamicType: Clone + Eq + Hash + Send + Sync + 'static,
{
    let incoming_rv = obj.resource_version();
    let stored_rv = keyed.get(&key).and_then(|stored| stored.resource_version());
    let unchanged = matches!((stored_rv, incoming_rv), (Some(a), Some(b)) if a == b);
    if unchanged {
        return;
    }
    *counter += 1;
    let outcome = keyed.upsert(rt, key, *counter, obj);
    debug_assert_ne!(
        outcome,
        UpsertOutcome::Stale,
        "incr-kube: a freshly incremented counter can never be stale against what's stored; \
         the counter and the collection's stored versions have drifted out of sync"
    );
}
