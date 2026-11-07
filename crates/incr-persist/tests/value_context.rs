//! Every encode and decode of a user value runs inside a value context
//! naming its slot, which the value's serde impl reads. The probe types
//! record the context they see; an unwrapped site records `None`.

#![allow(clippy::arc_with_non_send_sync)]

use std::cell::RefCell;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

use incr_persist::memo::{BuildId, MemoBuilder, MemoOptions, MemoryMemo};
use incr_persist::{
    current_value_context, EncodeMismatch, Local, MemoPart, MemoryBackend, Options, Store,
    ValueContext,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Ser,
    De,
}

type Seen = Vec<(&'static str, Op, Option<ValueContext>)>;

thread_local! {
    static SEEN: RefCell<Seen> = const { RefCell::new(Vec::new()) };
}

fn record(label: &'static str, op: Op) {
    SEEN.with(|seen| seen.borrow_mut().push((label, op, current_value_context())));
}

fn take_seen() -> Seen {
    SEEN.with(|seen| std::mem::take(&mut *seen.borrow_mut()))
}

trait Label {
    const NAME: &'static str;
    fn decoded(n: u32) -> u32 {
        n
    }
}

#[derive(Clone, Copy, Debug)]
struct Probe<L>(u32, PhantomData<L>);

impl<L> Probe<L> {
    fn new(n: u32) -> Self {
        Probe(n, PhantomData)
    }
}

impl<L> PartialEq for Probe<L> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<L> Eq for Probe<L> {}

impl<L> Hash for Probe<L> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<L: Label> Serialize for Probe<L> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        record(L::NAME, Op::Ser);
        self.0.serialize(s)
    }
}

impl<'de, L: Label> Deserialize<'de> for Probe<L> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        record(L::NAME, Op::De);
        u32::deserialize(d).map(|n| Probe::new(L::decoded(n)))
    }
}

macro_rules! label {
    ($ty:ident, $name:literal) => {
        #[derive(Clone, Copy, Debug, Hash)]
        struct $ty;
        impl Label for $ty {
            const NAME: &'static str = $name;
        }
    };
}

label!(Plain, "plain");
label!(MKey, "mkey");
label!(MVal, "mval");

#[derive(Clone, Copy, Debug, Hash)]
struct Drifts;

impl Label for Drifts {
    const NAME: &'static str = "drifts";
    fn decoded(n: u32) -> u32 {
        n + 1
    }
}

fn key(name: &str) -> Option<ValueContext> {
    Some(ValueContext::Key(name.to_owned()))
}

fn memo(table: &str, part: MemoPart) -> Option<ValueContext> {
    Some(ValueContext::MemoTable {
        table: table.to_owned(),
        part,
    })
}

#[track_caller]
fn assert_all_key(step: &str, name: &str) {
    let seen = take_seen();
    assert!(!seen.is_empty(), "{step}: no value was encoded or decoded");
    for (label, op, ctx) in &seen {
        assert_eq!(ctx, &key(name), "{step}: {label} {op:?}");
    }
}

type P = Probe<Plain>;

fn open_store(log: &MemoryBackend) -> Store<Local, MemoryBackend> {
    Store::open(log.clone(), Options::default()).unwrap()
}

#[test]
fn input_sites_run_in_the_inputs_context() {
    let log = MemoryBackend::new();
    let store = open_store(&log);
    let x = store.input("x", P::new(0)).unwrap();
    assert_all_key("declare", "x");
    store.set(x, P::new(1)).unwrap();
    assert_all_key("set", "x");
    store
        .batch(|b| {
            b.set(x, P::new(2));
        })
        .unwrap();
    assert_all_key("batch set", "x");
}

#[test]
fn collection_sites_run_in_the_collections_context() {
    let log = MemoryBackend::new();
    let store = open_store(&log);
    let c = store.collection::<P>("c").unwrap();
    take_seen();
    c.insert(&store, P::new(1)).unwrap();
    assert_all_key("insert", "c");
    c.insert(&store, P::new(2)).unwrap();
    take_seen();
    c.delete(&store, &P::new(1)).unwrap();
    assert_all_key("delete", "c");
    c.replace(&store, Some(&P::new(2)), P::new(3)).unwrap();
    assert_all_key("replace", "c");
    c.replace(&store, None, P::new(4)).unwrap();
    assert_all_key("replace none", "c");
    store
        .batch(|b| {
            b.insert(&c, P::new(5));
            b.delete(&c, &P::new(4));
            b.replace(&c, Some(&P::new(3)), P::new(6));
        })
        .unwrap();
    assert_all_key("batch", "c");
}

#[test]
fn snapshot_fold_and_compaction_run_in_the_slots_context() {
    let log = MemoryBackend::new();
    let store = open_store(&log);
    let x = store.input("x", P::new(0)).unwrap();
    store.set(x, P::new(7)).unwrap();
    take_seen();
    store.checkpoint().unwrap();
    assert_all_key("checkpoint input", "x");

    let log = MemoryBackend::new();
    let store = open_store(&log);
    let c = store.collection::<P>("c").unwrap();
    c.insert(&store, P::new(7)).unwrap();
    c.insert(&store, P::new(8)).unwrap();
    c.delete(&store, &P::new(8)).unwrap();
    take_seen();
    store.checkpoint().unwrap();
    for (label, op, ctx) in take_seen() {
        assert_eq!(ctx, key("c"), "checkpoint collection: {label} {op:?}");
    }
}

#[test]
fn open_replay_runs_in_the_slots_context() {
    let log = MemoryBackend::new();
    {
        let store = open_store(&log);
        let x = store.input("x", P::new(0)).unwrap();
        let c = store.collection::<P>("c").unwrap();
        store.set(x, P::new(1)).unwrap();
        c.insert(&store, P::new(2)).unwrap();
        c.insert(&store, P::new(3)).unwrap();
        c.delete(&store, &P::new(3)).unwrap();
        c.replace(&store, Some(&P::new(2)), P::new(4)).unwrap();
    }
    take_seen();
    let store = open_store(&log);
    store.input("x", P::new(0)).unwrap();
    assert_all_key("replay input", "x");
    store.collection::<P>("c").unwrap();
    assert_all_key("replay collection", "c");

    store.checkpoint().unwrap();
    let after = open_store(&log);
    take_seen();
    after.input("x", P::new(0)).unwrap();
    assert_all_key("replay input over a snapshot", "x");
    after.collection::<P>("c").unwrap();
    assert_all_key("replay collection over a snapshot", "c");
}

#[test]
fn refresh_runs_in_the_slots_context() {
    let log = MemoryBackend::new();
    let leader = open_store(&log);
    let x = leader.input("x", P::new(0)).unwrap();
    let c = leader.collection::<P>("c").unwrap();
    let follower = open_store(&log);
    follower.input("x", P::new(0)).unwrap();
    let fc = follower.collection::<P>("c").unwrap();
    take_seen();

    leader.set(x, P::new(1)).unwrap();
    c.insert(&leader, P::new(2)).unwrap();
    c.insert(&leader, P::new(3)).unwrap();
    take_seen();
    follower.refresh().unwrap();
    let seen = take_seen();
    assert!(!seen.is_empty());
    for (label, op, ctx) in &seen {
        assert!(
            *ctx == key("x") || *ctx == key("c"),
            "refresh: {label} {op:?} saw {ctx:?}"
        );
    }
    assert!(seen.iter().any(|s| s.2 == key("x")));
    assert!(seen.iter().any(|s| s.2 == key("c")));

    c.delete(&leader, &P::new(3)).unwrap();
    c.replace(&leader, Some(&P::new(2)), P::new(5)).unwrap();
    leader.checkpoint().unwrap();
    take_seen();
    follower.refresh().unwrap();
    for (label, op, ctx) in take_seen() {
        assert!(
            ctx == key("x") || ctx == key("c"),
            "refresh across a compaction: {label} {op:?} saw {ctx:?}"
        );
    }
    drop(fc);
}

#[cfg(debug_assertions)]
mod bypass_check {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq)]
    struct NeverEqual(u32);

    impl Hash for NeverEqual {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.0.hash(state);
        }
    }

    impl PartialEq for NeverEqual {
        fn eq(&self, _: &Self) -> bool {
            false
        }
    }

    impl Serialize for NeverEqual {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            record("never", Op::Ser);
            self.0.serialize(s)
        }
    }

    impl<'de> Deserialize<'de> for NeverEqual {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            record("never", Op::De);
            u32::deserialize(d).map(NeverEqual)
        }
    }

    #[test]
    fn the_bypass_check_encodes_in_the_inputs_context() {
        let log = MemoryBackend::new();
        let store = open_store(&log);
        let x = store.input("x", NeverEqual(1)).unwrap();
        take_seen();
        store.set(x, NeverEqual(2)).unwrap();
        let seen = take_seen();
        assert!(
            seen.iter().filter(|s| s.1 == Op::Ser).count() >= 3,
            "the check encodes both values besides the write: {seen:?}"
        );
        assert!(seen.iter().all(|s| s.2 == key("x")), "{seen:?}");
    }
}

type MK = Probe<MKey>;
type MV = Probe<MVal>;

fn options() -> MemoOptions {
    MemoOptions::new(BuildId::Fixed(0x77))
}

struct Backends {
    log: MemoryBackend,
    memo: MemoryMemo,
}

impl Backends {
    fn new() -> Self {
        Backends {
            log: MemoryBackend::new(),
            memo: MemoryMemo::new(),
        }
    }
}

fn assert_memo_parts(step: &str, table: &str, seen: &Seen, key_label: &str, value_label: &str) {
    for (label, op, ctx) in seen {
        let part = if *label == key_label {
            MemoPart::Key
        } else {
            assert_eq!(*label, value_label, "{step}: unexpected type");
            MemoPart::Value
        };
        assert_eq!(ctx, &memo(table, part), "{step}: {label} {op:?}");
    }
}

fn has(seen: &Seen, label: &str, op: Op) -> bool {
    seen.iter().any(|s| s.0 == label && s.1 == op)
}

#[test]
fn memo_key_and_value_encode_and_restore_run_in_the_tables_context() {
    let backends = Backends::new();
    {
        let store = Arc::new(open_store(&backends.log));
        let mut builder = MemoBuilder::new(options());
        let t = builder.table::<MK, MV>("t", 1);
        builder.define(t, |_, k| MV::new(k.0 * 2));
        let m = builder.open(&store, backends.memo.clone()).unwrap();
        take_seen();
        let q = m.get_or_insert(t, MK::new(4));
        let seen = take_seen();
        assert!(has(&seen, "mkey", Op::Ser), "{seen:?}");
        assert_memo_parts("lookup", "t", &seen, "mkey", "mval");
        assert_eq!(m.runtime().get(q).0, 8);
        take_seen();
        assert_eq!(m.checkpoint().unwrap().written, 1);
        let seen = take_seen();
        for (label, op) in [
            ("mkey", Op::Ser),
            ("mval", Op::Ser),
            ("mkey", Op::De),
            ("mval", Op::De),
        ] {
            assert!(has(&seen, label, op), "checkpoint: {label} {op:?}");
        }
        assert_memo_parts("checkpoint", "t", &seen, "mkey", "mval");
        assert!(m.take_encode_mismatches().is_empty());
    }
    let store = Arc::new(open_store(&backends.log));
    let mut builder = MemoBuilder::new(options());
    let t = builder.table::<MK, MV>("t", 1);
    builder.define(t, |_, k| MV::new(k.0 * 2));
    let m = builder.open(&store, backends.memo.clone()).unwrap();
    take_seen();
    let q = m.get_or_insert(t, MK::new(4));
    assert_eq!(m.runtime().get(q).0, 8);
    let seen = take_seen();
    assert_eq!(m.stats().reused, 1);
    for (label, op) in [("mkey", Op::De), ("mval", Op::De), ("mkey", Op::Ser)] {
        assert!(has(&seen, label, op), "restore: {label} {op:?}");
    }
    assert_memo_parts("restore", "t", &seen, "mkey", "mval");
}

#[test]
fn a_mismatching_memo_value_is_recorded_and_not_persisted() {
    let backends = Backends::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let open = |runs: &Arc<AtomicUsize>| {
        let store = Arc::new(open_store(&backends.log));
        let mut builder = MemoBuilder::new(options());
        let good = builder.table::<MK, MV>("good", 1);
        let bad = builder.table::<MK, Probe<Drifts>>("bad", 1);
        builder.define(good, |_, k| MV::new(k.0));
        let counter = Arc::clone(runs);
        builder.define(bad, move |_, k| {
            counter.fetch_add(1, SeqCst);
            Probe::new(k.0)
        });
        let m = builder.open(&store, backends.memo.clone()).unwrap();
        (store, m, good, bad)
    };
    {
        let (_store, m, good, bad) = open(&runs);
        let rt = m.runtime();
        rt.get(m.get_or_insert(good, MK::new(1)));
        rt.get(m.get_or_insert(bad, MK::new(1)));
        rt.get(m.get_or_insert(bad, MK::new(2)));
        let stats = m.checkpoint().unwrap();
        assert_eq!(stats.written, 1, "only the good entry is persisted");
        assert_eq!(
            m.take_encode_mismatches(),
            [EncodeMismatch {
                table: "bad".to_owned(),
                part: MemoPart::Value
            }]
        );
        assert!(m.take_encode_mismatches().is_empty());
    }
    assert_eq!(runs.load(SeqCst), 2);
    let (_store, m, good, bad) = open(&runs);
    let rt = m.runtime();
    rt.get(m.get_or_insert(good, MK::new(1)));
    assert_eq!(rt.get(m.get_or_insert(bad, MK::new(1))).0, 1);
    assert_eq!(runs.load(SeqCst), 3, "the bad entry was recomputed");
    assert_eq!(m.stats().reused, 1);
}

#[test]
fn a_mismatching_memo_key_is_recorded() {
    let backends = Backends::new();
    let store = Arc::new(open_store(&backends.log));
    let mut builder = MemoBuilder::new(options());
    let t = builder.table::<Probe<Drifts>, MV>("t", 1);
    builder.define(t, |_, k| MV::new(k.0));
    let m = builder.open(&store, backends.memo.clone()).unwrap();
    m.runtime().get(m.get_or_insert(t, Probe::new(1)));
    assert_eq!(m.checkpoint().unwrap().written, 0);
    assert_eq!(
        m.take_encode_mismatches(),
        [EncodeMismatch {
            table: "t".to_owned(),
            part: MemoPart::Key
        }]
    );
}

#[test]
fn dependency_fingerprints_of_a_collection_run_in_its_context() {
    let backends = Backends::new();
    let leader = Arc::new(open_store(&backends.log));
    let mut builder = MemoBuilder::new(options());
    let t = builder.table::<MK, MV>("t", 1);
    builder.define(t, |_, k| MV::new(k.0));
    let _memo = builder.open(&leader, backends.memo.clone()).unwrap();
    let c = leader.collection::<P>("c").unwrap();
    take_seen();
    c.insert(&leader, P::new(1)).unwrap();
    c.insert(&leader, P::new(2)).unwrap();
    assert_all_key("insert", "c");
    c.replace(&leader, Some(&P::new(1)), P::new(3)).unwrap();
    assert_all_key("replace", "c");
    c.delete(&leader, &P::new(2)).unwrap();
    assert_all_key("delete", "c");

    let follower = Arc::new(open_store(&backends.log));
    let mut builder = MemoBuilder::new(options());
    let t = builder.table::<MK, MV>("t", 1);
    builder.define(t, |_, k| MV::new(k.0));
    let _memo = builder.open(&follower, MemoryMemo::new()).unwrap();
    let fc = follower.collection::<P>("c").unwrap();
    c.insert(&leader, P::new(9)).unwrap();
    leader.checkpoint().unwrap();
    take_seen();
    follower.refresh().unwrap();
    assert_all_key("refresh across a compaction", "c");
    drop(fc);
}
