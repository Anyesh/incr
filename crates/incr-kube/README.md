# incr-kube

A drop-in replacement for kube-rs's reflector `Store<K>`, built on `incr-core`'s incrementally-maintained collections instead of a `RwLock<HashMap>` you scan on every query.

kube-rs's `Store` holds its read lock for the entire duration of a predicate scan (`Store::state_filter`'s own docs say so), and the reflector's writer takes that same lock to apply the next watch event. Under load, a hot query and a busy writer queue behind each other. `incr-kube` keeps a `filter`/`group_by`/`join`/`aggregate` view up to date as each watch event lands, so a query reads an already-current result instead of triggering a scan.

This isn't a new idea: Go's Kubernetes ecosystem has had it for close to a decade (`client-go`'s `Indexer`, `controller-runtime`'s field indexes). `kube-rs` doesn't have it yet; there's an open issue asking for it ([kube-rs#970](https://github.com/kube-rs/kube/issues/970), 2022) and a stalled PR attempting a partial fix ([kube-rs#1681](https://github.com/kube-rs/kube/pull/1681)) where kube-rs's own maintainer names the exact problem this crate solves. `incr-kube` is that pattern, ported.

## Install

```
cargo add incr-kube
```

## Quick start

```rust
use incr_concurrent::Runtime;
use incr_kube::{drive_reflector, KeyedCollection};
use k8s_openapi::api::core::v1::Pod;
use kube::runtime::reflector::ObjectRef;
use kube::runtime::watcher;
use kube::Api;

let rt = Runtime::new();
let pods: KeyedCollection<ObjectRef<Pod>, Pod> = KeyedCollection::new(&rt);

// A live view: pods grouped by namespace, incrementally maintained.
let by_namespace = pods.collection().group_by(&rt, |entry| {
    entry.value().metadata.namespace.clone().unwrap_or_default()
});

let api: Api<Pod> = Api::all(client);
let stream = watcher(api, watcher::Config::default());

drive_reflector(
    &pods,
    &rt,
    (),
    stream,
    |_event| {},          // on_event: log, or signal readiness on InitDone
    |err| eprintln!("watch error: {err}"),
).await;

// Elsewhere, on any thread: reads the current state of the view,
// never a full scan of the source collection.
let ns_pods = by_namespace.get_group(&rt, "default").values(&rt);
```

`KeyedCollection` gives you `upsert`/`remove`/`get`/`retain` directly if you want to drive it from something other than a raw `watcher()` stream. `drive_reflector` is the kube-rs integration built on top: it consumes the same event stream `reflector()` wraps (not `store_shared()`/`ReflectHandle`, which never broadcasts deletes), applies `Apply`/`Delete`/`InitApply`/`InitDone` the same way kube-rs's own reflector does, and reconciles a relist via `retain` on `InitDone`.

## Numbers

`cargo bench -p incr-kube --bench store_vs_view`, 50,000 Pods, a writer applying events at 1,000/sec, 4 concurrent reader threads:

- **Writer:** kube-rs's `Store` sustains 444-506 events/sec against the 1,000 target (its write path queues behind readers holding the scan lock). `incr-kube`'s writer hits the target exactly, 7-8µs mean, roughly 300-500x faster.
- **Reader, 1% selectivity** (a targeted query, e.g. "pods not Running"): `incr-kube` ~100µs mean vs. kube-rs's ~2ms, since `state_filter` still scans all 50k Pods regardless of match count.
- **Reader, 50% selectivity:** `incr-kube` is *slower* here, 15.6ms vs. 4.8ms mean. Materializing a large view's current result set has its own clone cost, and a live writer keeps a large view dirty enough that reads queue behind recomputes, the same lock-during-scan shape as the problem above, just moved onto the reader path. Don't reach for a wide `incr-kube` view in place of a targeted `state_filter` call; the win is real for selective queries, not for "give me everything."

## Proven against a real operator

`pod-graceful-drain`'s five `Store<K>` reflectors and hand-rolled namespace scans were converted onto `KeyedCollection`/`drive_reflector`/`group_by`, independently reviewed, with the same non-cluster test suite passing unchanged.

## Known gaps

- **`ListSemantic::Any`:** if the caller's `watcher()` is configured for cached, possibly-stale relists, a relist's `InitApply` can carry an older object state than a live `Apply` already applied for the same key. The version counter applies it as newest regardless, the same way kube-rs's own `Store` does under that setting.
- **Relist reconciliation is opt-in.** `KeyedCollection::retain` prunes drift after a full listing; `drive_reflector` calls it on every `InitDone`, but using `KeyedCollection` directly gets no automatic reconciliation.
- **A relist mid-flight is visible as a mix of old and fresh state, not an atomic snapshot swap.** kube-rs's `Store` buffers a relist and swaps it in atomically on `InitDone`; `drive_reflector` applies each `InitApply` live and only prunes on `InitDone`, so a reader mid-relist (or on first start) can see a partially-updated collection. Harmless if you don't depend on relist snapshot atomicity; real if you do.

## When to use

Use this for a query your reconciler runs often, over a predicate that's selective (a small fraction of the cache matches). If you're reading the whole cache anyway, or reading rarely, `kube-rs`'s own `Store` is simpler and has no second cache to keep in sync.
