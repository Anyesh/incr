//! Integration tests for `drive_reflector`, mirroring kube-runtime's own
//! `reflector/dispatcher.rs` test pattern: a synthetic
//! `futures::stream::iter([...])` fed directly in, no live cluster
//! needed.

use futures::stream;
use incr_concurrent::Runtime;
use incr_kube::{drive_reflector, KeyedCollection};
use k8s_openapi::api::core::v1::Pod;
use kube::runtime::reflector::ObjectRef;
use kube::runtime::watcher::{self, Event};

fn testpod(name: &str, resource_version: &str) -> Pod {
    let mut pod = Pod::default();
    pod.metadata.name = Some(name.to_string());
    pod.metadata.resource_version = Some(resource_version.to_string());
    pod
}

#[tokio::test]
async fn apply_then_get() {
    let rt = Runtime::new();
    let keyed: KeyedCollection<ObjectRef<Pod>, Pod> = KeyedCollection::new(&rt);
    let foo = testpod("foo", "1");

    let events = stream::iter([Ok(Event::Apply(foo.clone()))]);
    drive_reflector(&keyed, &rt, (), events, |_| {}, |_| {}).await;

    assert!(keyed.get(&ObjectRef::from(&foo)).is_some());
}

#[tokio::test]
async fn delete_removes() {
    let rt = Runtime::new();
    let keyed: KeyedCollection<ObjectRef<Pod>, Pod> = KeyedCollection::new(&rt);
    let foo = testpod("foo", "1");

    let events = stream::iter([
        Ok(Event::Apply(foo.clone())),
        Ok(Event::Delete(foo.clone())),
    ]);
    drive_reflector(&keyed, &rt, (), events, |_| {}, |_| {}).await;

    assert!(keyed.get(&ObjectRef::from(&foo)).is_none());
}

#[tokio::test]
async fn relist_prunes_key_not_relisted() {
    let rt = Runtime::new();
    let keyed: KeyedCollection<ObjectRef<Pod>, Pod> = KeyedCollection::new(&rt);
    let foo = testpod("foo", "1");
    let bar = testpod("bar", "1");

    let events = stream::iter([
        Ok(Event::Apply(foo.clone())),
        Ok(Event::Apply(bar.clone())),
        Ok(Event::Init),
        Ok(Event::InitApply(bar.clone())),
        Ok(Event::InitDone),
    ]);
    drive_reflector(&keyed, &rt, (), events, |_| {}, |_| {}).await;

    assert!(
        keyed.get(&ObjectRef::from(&foo)).is_none(),
        "foo was not in the relist, should be pruned on InitDone"
    );
    assert!(keyed.get(&ObjectRef::from(&bar)).is_some());
}

#[tokio::test]
async fn relist_restart_resets_seen_set() {
    let rt = Runtime::new();
    let keyed: KeyedCollection<ObjectRef<Pod>, Pod> = KeyedCollection::new(&rt);
    let a = testpod("a", "1");
    let b = testpod("b", "1");

    // A relist starts, sees `a`, then the attempt errors out and
    // restarts with a fresh Init; the second attempt only relists `b`.
    // `a` must not survive on the strength of the aborted first
    // attempt's seen-set.
    let events = stream::iter([
        Ok(Event::Apply(a.clone())),
        Ok(Event::Init),
        Ok(Event::InitApply(a.clone())),
        Err(watcher::Error::NoResourceVersion),
        Ok(Event::Init),
        Ok(Event::InitApply(b.clone())),
        Ok(Event::InitDone),
    ]);

    let mut error_count = 0;
    drive_reflector(&keyed, &rt, (), events, |_| {}, |_| error_count += 1).await;

    assert_eq!(error_count, 1);
    assert!(
        keyed.get(&ObjectRef::from(&a)).is_none(),
        "a survived only in the aborted first relist attempt's seen-set"
    );
    assert!(keyed.get(&ObjectRef::from(&b)).is_some());
}

#[tokio::test]
async fn error_does_not_stop_later_events() {
    let rt = Runtime::new();
    let keyed: KeyedCollection<ObjectRef<Pod>, Pod> = KeyedCollection::new(&rt);
    let foo = testpod("foo", "1");

    let events = stream::iter([
        Err(watcher::Error::NoResourceVersion),
        Ok(Event::Apply(foo.clone())),
    ]);
    let mut error_count = 0;
    drive_reflector(&keyed, &rt, (), events, |_| {}, |_| error_count += 1).await;

    assert_eq!(error_count, 1);
    assert!(keyed.get(&ObjectRef::from(&foo)).is_some());
}

#[tokio::test]
async fn unchanged_relist_skips_upsert_but_still_counts_as_seen() {
    let rt = Runtime::new();
    let keyed: KeyedCollection<ObjectRef<Pod>, Pod> = KeyedCollection::new(&rt);
    let foo = testpod("foo", "1");
    let bar = testpod("bar", "1");

    // foo is applied once, then relisted at the *same* resourceVersion
    // (unchanged); bar is new in the relist. Expected log deltas: +1
    // for foo's initial Apply, +0 for foo's unchanged InitApply (the
    // equality-skip), +1 for bar's InitApply. If the skip didn't fire,
    // foo's InitApply would do a spurious replace of an unchanged
    // object (a delete+insert, +2 deltas), giving 4 instead of 2.
    let events = stream::iter([
        Ok(Event::Apply(foo.clone())),
        Ok(Event::Init),
        Ok(Event::InitApply(foo.clone())),
        Ok(Event::InitApply(bar.clone())),
        Ok(Event::InitDone),
    ]);
    drive_reflector(&keyed, &rt, (), events, |_| {}, |_| {}).await;

    let deltas = rt.get(keyed.collection().version_node());
    assert_eq!(
        deltas, 2,
        "unchanged relisted object should not push a delta"
    );

    // The real bug this guards against: if the equality-skip's
    // seen-set insert were gated by the skip instead of unconditional,
    // foo would never be recorded as seen and InitDone would prune it.
    assert!(
        keyed.get(&ObjectRef::from(&foo)).is_some(),
        "foo must survive InitDone even though its InitApply was skipped as unchanged"
    );
    assert!(keyed.get(&ObjectRef::from(&bar)).is_some());
}

#[tokio::test]
async fn restart_seeds_counter_above_stored_high_water_mark() {
    let rt = Runtime::new();
    let keyed: KeyedCollection<ObjectRef<Pod>, Pod> = KeyedCollection::new(&rt);
    let foo = testpod("foo", "1");
    let foo_v2 = testpod("foo", "2");

    // First watch stream ends (a reconnect, an `Api` reconfiguration,
    // whatever); the caller builds a fresh stream and calls
    // `drive_reflector` again against the same collection. The
    // second-call counter must not restart at 0, or every event here
    // gets rejected as `Stale` against what the first call already
    // stored.
    let first = stream::iter([Ok(Event::Apply(foo.clone()))]);
    drive_reflector(&keyed, &rt, (), first, |_| {}, |_| {}).await;

    let second = stream::iter([Ok(Event::Apply(foo_v2.clone()))]);
    drive_reflector(&keyed, &rt, (), second, |_| {}, |_| {}).await;

    let stored = keyed
        .get(&ObjectRef::from(&foo))
        .expect("foo still present");
    assert_eq!(
        stored.metadata.resource_version.as_deref(),
        Some("2"),
        "second call's Apply should have replaced the first call's stored value"
    );
}

#[tokio::test]
async fn on_event_fires_for_every_ok_event_not_for_errors() {
    let rt = Runtime::new();
    let keyed: KeyedCollection<ObjectRef<Pod>, Pod> = KeyedCollection::new(&rt);
    let foo = testpod("foo", "1");

    let events = stream::iter([
        Ok(Event::Apply(foo.clone())),
        Err(watcher::Error::NoResourceVersion),
        Ok(Event::Init),
        Ok(Event::InitApply(foo.clone())),
        Ok(Event::InitDone),
        Ok(Event::Delete(foo.clone())),
    ]);

    let mut seen_kinds = Vec::new();
    drive_reflector(
        &keyed,
        &rt,
        (),
        events,
        |ev| {
            seen_kinds.push(match ev {
                Event::Apply(_) => "Apply",
                Event::Delete(_) => "Delete",
                Event::Init => "Init",
                Event::InitApply(_) => "InitApply",
                Event::InitDone => "InitDone",
            });
        },
        |_| {},
    )
    .await;

    assert_eq!(
        seen_kinds,
        vec!["Apply", "Init", "InitApply", "InitDone", "Delete"],
        "on_event should fire once per Ok event, in order, skipping the Err in between"
    );
}
