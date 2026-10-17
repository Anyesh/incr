//! `Value`: the user-type bound for everything stored in a `Runtime<C>`.
//!
//! Single trait, single mental model: `Value` = `Clone + PartialEq + Send + Sync + 'static`.
//! Local-strategy users pay no runtime cost for the `Send + Sync` bound
//! (those are zero-cost markers), but they cannot store `Rc<...>` or
//! other `!Send` types directly. This is a deliberate uniformity
//! decision for the v0.2 API: one bound across both strategies, identical
//! impl story, no per-strategy Value implementations.
//!
//! Users who genuinely need to embed a `!Send` type can wrap it in
//! `Arc<Mutex<T>>` or move the !Send state outside the graph and pass
//! values through. The consolidation plan's decision page covers the
//! tradeoff.
//!
//! The blanket impl auto-derives `Value` for every qualifying type, so
//! no `impl Value for MyType` boilerplate is required. This matches
//! production incr-compute's `T: Any + Clone + PartialEq + 'static` and
//! tightens it with `Send + Sync`.
//!
//! # Value and key methods must not call into the runtime
//!
//! The runtime calls a value's `Clone` and `PartialEq`, and a collection
//! element's or key's `Hash`, `Eq` and `Ord`, while it holds locks or a
//! compute claim. A method that calls back into any runtime from there
//! (a `get`, a `set`, a collection read or write, creating or deleting a
//! node) panics with "value and key methods must not call into the
//! runtime", under `Local` and `Shared` alike, instead of deadlocking or
//! corrupting the graph. Operator closures (`filter`'s predicate,
//! `map`'s function, key functions, `aggregate`'s `lift` and `combine`)
//! are compute code and may read the runtime.
//!
//! `Drop` is under the same contract but outside the check: under
//! `Shared` a displaced value may be reclaimed on any thread, inside any
//! later runtime call, so a `Drop` that calls into the runtime may panic
//! on a stale-handle or own-compute check or deadlock on a lock its
//! thread already holds. It never causes undefined behaviour.

pub trait Value: Clone + PartialEq + Send + Sync + 'static {}

impl<T> Value for T where T: Clone + PartialEq + Send + Sync + 'static {}
