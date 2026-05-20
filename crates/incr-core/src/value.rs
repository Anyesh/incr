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

pub trait Value: Clone + PartialEq + Send + Sync + 'static {}

impl<T> Value for T where T: Clone + PartialEq + Send + Sync + 'static {}
