//! `Store::staged_batch`: writes applied as they are made and appended as
//! one conditional run at commit.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use incr_persist::memo::{BuildId, Memo, MemoBuilder, MemoCells, MemoOptions, MemoryMemo};
use incr_persist::{
    Backend, Cells, ConditionalAppend, Error, Local, MemoryBackend, Options, Origin, PendingWrite,
    Position, ReadOutcome, Record, Shared, SnapshotId, Store,
};

fn open<C: Cells, B: Backend>(backend: B) -> Store<C, B> {
    Store::open(backend, Options::default()).unwrap()
}

fn log_after(backend: &impl Backend, after: Position) -> Vec<(Position, Record)> {
    let mut out = Vec::new();
    assert_eq!(
        backend.read_from(after, &mut out).unwrap(),
        ReadOutcome::Records
    );
    out
}

fn sorted<T: Ord>(mut values: Vec<T>) -> Vec<T> {
    values.sort();
    values
}

fn assert_refused<T: std::fmt::Debug>(what: &str, result: incr_persist::Result<T>) {
    match result {
        Err(Error::StagedBatchOpen) => {}
        other => panic!("{what}: expected StagedBatchOpen, got {other:?}"),
    }
}

fn assert_poisoned<T: std::fmt::Debug>(what: &str, result: incr_persist::Result<T>) {
    match result {
        Err(Error::Poisoned) => {}
        other => panic!("{what}: expected Poisoned, got {other:?}"),
    }
}

type Hook =
    Box<dyn FnOnce(&MemoryBackend, Position, &[Record]) -> incr_persist::Result<Position> + Send>;

/// A memory backend whose next `append_at` runs a hook in its place.
#[derive(Clone, Default)]
struct Hooked {
    inner: MemoryBackend,
    next_append_at: Arc<Mutex<Option<Hook>>>,
    append_ats: Arc<AtomicUsize>,
}

impl Hooked {
    fn on_append_at(
        &self,
        hook: impl FnOnce(&MemoryBackend, Position, &[Record]) -> incr_persist::Result<Position>
            + Send
            + 'static,
    ) {
        *self.next_append_at.lock().unwrap() = Some(Box::new(hook));
    }
}

fn ambiguous() -> Error {
    Error::Backend("connection reset".into())
}

impl Backend for Hooked {
    fn append(&self, records: &[Record]) -> incr_persist::Result<Position> {
        self.inner.append(records)
    }

    fn sync(&self, upto: Position) -> incr_persist::Result<()> {
        self.inner.sync(upto)
    }

    fn read_from(
        &self,
        after: Position,
        out: &mut Vec<(Position, Record)>,
    ) -> incr_persist::Result<ReadOutcome> {
        self.inner.read_from(after, out)
    }

    fn load_snapshot(&self) -> incr_persist::Result<Option<(Position, Vec<Record>)>> {
        self.inner.load_snapshot()
    }

    fn snapshot_id(&self) -> incr_persist::Result<Option<SnapshotId>> {
        self.inner.snapshot_id()
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> incr_persist::Result<()> {
        self.inner.compact(upto, folded)
    }

    fn writable(&self) -> bool {
        self.inner.writable()
    }

    fn conditional_append(&self) -> Option<&dyn ConditionalAppend> {
        Some(self)
    }
}

impl ConditionalAppend for Hooked {
    fn append_at(&self, after: Position, records: &[Record]) -> incr_persist::Result<Position> {
        self.append_ats.fetch_add(1, SeqCst);
        let hook = self.next_append_at.lock().unwrap().take();
        match hook {
            Some(hook) => hook(&self.inner, after, records),
            None => self.inner.append_at(after, records),
        }
    }
}

/// Forwards everything but offers no conditional append.
struct Plain(MemoryBackend);

impl Backend for Plain {
    fn append(&self, records: &[Record]) -> incr_persist::Result<Position> {
        self.0.append(records)
    }

    fn sync(&self, upto: Position) -> incr_persist::Result<()> {
        self.0.sync(upto)
    }

    fn read_from(
        &self,
        after: Position,
        out: &mut Vec<(Position, Record)>,
    ) -> incr_persist::Result<ReadOutcome> {
        self.0.read_from(after, out)
    }

    fn load_snapshot(&self) -> incr_persist::Result<Option<(Position, Vec<Record>)>> {
        self.0.load_snapshot()
    }

    fn snapshot_id(&self) -> incr_persist::Result<Option<SnapshotId>> {
        self.0.snapshot_id()
    }

    fn compact(&self, upto: Position, folded: Vec<Record>) -> incr_persist::Result<()> {
        self.0.compact(upto, folded)
    }

    fn writable(&self) -> bool {
        self.0.writable()
    }
}

fn staged_writes_apply_at_once_and_commit_appends_them_once_in_write_order<C: Cells>() {
    let log = MemoryBackend::new();
    let store = open::<C, _>(log.clone());
    let x = store.input("x", 0_u32).unwrap();
    let c = store.collection::<u32>("c").unwrap();
    let rt = store.runtime();
    let doubled = {
        let xh = x.handle();
        rt.create_query(move |ctx| ctx.get(xh) * 2)
    };
    let count = c.handle().count(rt);
    assert_eq!(rt.get(doubled), 0);
    let before = store.stats().unwrap();

    let mut staged = store.staged_batch().unwrap();
    staged.set(x, 1).unwrap();
    assert_eq!(rt.get(doubled), 2, "a staged write applies at once");
    staged.insert(&c, 5).unwrap();
    staged.insert(&c, 6).unwrap();
    staged.delete(&c, 5).unwrap();
    staged.replace(&c, Some(6), 7).unwrap();
    staged.set(x, 3).unwrap();
    assert_eq!(rt.get(doubled), 6);
    assert_eq!(rt.get(count), 1);
    assert_eq!(
        store.stats().unwrap().position,
        before.position,
        "staged writes do not advance the position"
    );
    assert!(log_after(&log, before.position).is_empty());
    let last = staged.commit().unwrap();

    let landed = log_after(&log, before.position);
    assert_eq!(last, Position(before.position.0 + 6));
    assert_eq!(landed.last().unwrap().0, last);
    let kinds: Vec<_> = landed
        .iter()
        .map(|(_, record)| match record {
            Record::Set { .. } => "set",
            Record::Insert { .. } => "insert",
            Record::Remove { .. } => "remove",
            Record::Replace { .. } => "replace",
            _ => "other",
        })
        .collect();
    assert_eq!(
        kinds,
        ["set", "insert", "insert", "remove", "replace", "set"]
    );
    let stats = store.stats().unwrap();
    assert_eq!(stats.position, last);
    assert_eq!(
        stats.records_since_checkpoint,
        before.records_since_checkpoint + 6
    );
    assert_eq!(rt.get(doubled), 6, "commit does not apply again");
    assert_eq!(sorted(c.handle().values(rt)), vec![7]);
    drop(staged);

    store.set(x, 4).unwrap();
    drop(store);
    let reopened = open::<C, _>(log);
    let c = reopened.collection::<u32>("c").unwrap();
    assert_eq!(sorted(c.handle().values(reopened.runtime())), vec![7]);
}

fn staged_keyed_writes_go_through_pending_writes<C: Cells>() {
    let log = MemoryBackend::new();
    let store = open::<C, _>(log.clone());
    let keyed = store
        .keyed_collection::<u32, String, (u32, String)>("k")
        .unwrap();
    let x = store.input("x", 0_u32).unwrap();
    store.upsert(&keyed, 1, "r0".to_owned()).unwrap();
    let before = store.stats().unwrap().position;

    let mut staged = store.staged_batch().unwrap();
    let mut first = PendingWrite::new();
    first.upsert(&keyed, 1, "r1".to_owned());
    first.upsert(&keyed, 2, "a".to_owned());
    first.set(x, 9);
    staged.write_pending(first).unwrap();
    assert_eq!(keyed.row(&1).as_deref(), Some("r1"));
    let mut second = PendingWrite::new();
    second.upsert(&keyed, 1, "r2".to_owned());
    second.remove(&keyed, &2);
    staged.write_pending(second).unwrap();
    assert_eq!(keyed.row(&1).as_deref(), Some("r2"));
    assert!(!keyed.contains(&2));
    let last = staged.commit().unwrap();
    assert!(last > before);
    assert_eq!(keyed.len(), 1);
    drop(staged);

    drop(store);
    let reopened = open::<C, _>(log);
    let keyed = reopened
        .keyed_collection::<u32, String, (u32, String)>("k")
        .unwrap();
    assert_eq!(keyed.row(&1).as_deref(), Some("r2"));
    assert_eq!(keyed.items(), vec![(1, "r2".to_owned())]);
    let x = reopened.input("x", 0_u32).unwrap();
    assert_eq!(reopened.runtime().get(x.handle()), 9);
}

fn every_other_store_call_refuses_while_a_batch_is_staged<C: Cells>() {
    let store = Arc::new(open::<C, _>(MemoryBackend::new()));
    let x = store.input("x", 0_u32).unwrap();
    let c = store.collection::<u32>("c").unwrap();
    let mut staged = store.staged_batch().unwrap();
    staged.set(x, 1).unwrap();

    assert_refused("set", store.set(x, 2));
    assert_refused("insert", c.insert(&store, 1));
    assert_refused("delete", store.delete(x));
    assert_refused(
        "batch",
        store.batch(|b| {
            b.set(x, 3);
        }),
    );
    assert_refused(
        "upsert via pending",
        store.write_pending({
            let mut pending = PendingWrite::new();
            pending.set(x, 4);
            pending
        }),
    );
    assert_refused("declare", store.input("y", 0_u32));
    assert_refused("undeclare", store.undeclare(x));
    assert_refused("refresh", store.refresh());
    assert_refused("restored_keys", store.restored_keys(""));
    assert_refused("checkpoint", store.checkpoint());
    assert_refused("unlogged scope", store.unlogged_scope().map(|_| ()));
    assert_refused("second staged batch", store.staged_batch().map(|_| ()));
    assert_eq!(
        store.input("x", 0_u32).unwrap(),
        x,
        "a bound key is handed back"
    );
    store.stats().unwrap();

    staged.commit().unwrap();
    assert!(matches!(staged.set(x, 5), Err(Error::StagedBatchClosed)));
    drop(staged);
    store.set(x, 6).unwrap();
    store.refresh().unwrap();
    store.checkpoint().unwrap();
}

fn a_staged_batch_refuses_while_a_scope_is_open<C: Cells>() {
    let store = open::<C, _>(MemoryBackend::new());
    let scope = store.unlogged_scope().unwrap();
    assert!(matches!(
        store.staged_batch().map(|_| ()),
        Err(Error::UnloggedScopeOpen)
    ));
    scope.close().unwrap();
    store.staged_batch().unwrap().commit().unwrap();
}

fn a_staged_batch_needs_a_writable_conditional_backend<C: Cells>() {
    let plain = open::<C, _>(Plain(MemoryBackend::new()));
    assert!(matches!(
        plain.staged_batch().map(|_| ()),
        Err(Error::ConditionalAppendUnsupported)
    ));
    let log = MemoryBackend::new();
    let read_only = open::<C, _>(log.read_only());
    assert!(matches!(
        read_only.staged_batch().map(|_| ()),
        Err(Error::ReadOnly)
    ));
}

fn opening_catches_up_and_an_empty_commit_appends_nothing<C: Cells>() {
    let log = MemoryBackend::new();
    let store = open::<C, _>(log.clone());
    let x = store.input("x", 0_u32).unwrap();
    let other = open::<C, _>(log.clone());
    let ox = other.input("x", 0_u32).unwrap();
    let foreign = other.set(ox, 8).unwrap();

    let mut staged = store.staged_batch().unwrap();
    assert_eq!(store.stats().unwrap().position, foreign);
    assert_eq!(store.runtime().get(x.handle()), 8);
    assert_eq!(staged.commit().unwrap(), foreign);
    assert!(log_after(&log, foreign).is_empty());
}

fn dropping_an_uncommitted_batch_poisons<C: Cells>() {
    let store = open::<C, _>(MemoryBackend::new());
    let x = store.input("x", 0_u32).unwrap();
    let mut staged = store.staged_batch().unwrap();
    staged.set(x, 1).unwrap();
    drop(staged);
    assert_poisoned("set after drop", store.set(x, 2));
}

fn a_panic_inside_the_batch_poisons<C: Cells>() {
    let store = open::<C, _>(MemoryBackend::new());
    let x = store.input("x", 0_u32).unwrap();
    let unwound = catch_unwind(AssertUnwindSafe(|| {
        let mut staged = store.staged_batch().unwrap();
        staged.set(x, 1).unwrap();
        panic!("user code fails inside the block");
    }));
    assert!(unwound.is_err());
    assert_poisoned("set after panic", store.set(x, 2));
}

fn subscribers_hear_each_record_at_commit<C: Cells>() {
    let store = open::<C, _>(MemoryBackend::new());
    let x = store.input("x", 0_u32).unwrap();
    let c = store.collection::<u32>("c").unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    store
        .subscribe(move |position, record, origin| {
            sink.lock()
                .unwrap()
                .push((position, record.clone(), origin));
        })
        .unwrap();
    let before = store.stats().unwrap().position;
    let mut staged = store.staged_batch().unwrap();
    staged.set(x, 1).unwrap();
    staged.insert(&c, 2).unwrap();
    assert!(
        seen.lock().unwrap().is_empty(),
        "nothing is heard before commit"
    );
    let last = staged.commit().unwrap();
    let seen = seen.lock().unwrap();
    let positions: Vec<_> = seen.iter().map(|(position, _, _)| *position).collect();
    assert_eq!(positions, vec![Position(before.0 + 1), last]);
    assert!(seen.iter().all(|(_, _, origin)| *origin == Origin::Local));
}

fn an_append_error_with_nothing_landed_keeps_the_run_for_retry<C: Cells>() {
    let backend = Hooked::default();
    let store = open::<C, _>(backend.clone());
    let x = store.input("x", 0_u32).unwrap();
    let before = store.stats().unwrap().position;
    let mut staged = store.staged_batch().unwrap();
    staged.set(x, 1).unwrap();
    staged.set(x, 2).unwrap();
    backend.on_append_at(|_, _, _| Err(ambiguous()));
    assert!(matches!(staged.commit(), Err(Error::Backend(_))));
    assert!(log_after(&backend.inner, before).is_empty());
    assert!(matches!(staged.set(x, 3), Err(Error::StagedBatchClosed)));
    assert_refused("set while awaiting retry", store.set(x, 4));

    let last = staged.retry().unwrap();
    assert_eq!(last, Position(before.0 + 2));
    assert_eq!(log_after(&backend.inner, before).len(), 2);
    drop(staged);
    assert_eq!(store.stats().unwrap().position, last);
    store.set(x, 5).unwrap();
}

fn an_append_error_after_the_run_landed_is_adopted_on_retry<C: Cells>() {
    let backend = Hooked::default();
    let store = open::<C, _>(backend.clone());
    let x = store.input("x", 0_u32).unwrap();
    let c = store.collection::<u32>("c").unwrap();
    let before = store.stats().unwrap().position;
    let mut staged = store.staged_batch().unwrap();
    staged.set(x, 1).unwrap();
    staged.insert(&c, 1).unwrap();
    backend.on_append_at(|inner, after, records| {
        inner.append_at(after, records)?;
        Err(ambiguous())
    });
    assert!(matches!(staged.commit(), Err(Error::Backend(_))));
    let last = staged.retry().unwrap();
    assert_eq!(last, Position(before.0 + 2));
    assert_eq!(backend.append_ats.load(SeqCst), 2);
    assert_eq!(log_after(&backend.inner, before).len(), 2, "landed once");
    drop(staged);
    assert_eq!(store.stats().unwrap().position, last);
    assert_eq!(c.handle().values(store.runtime()), vec![1]);
    store.set(x, 2).unwrap();
}

fn a_foreign_record_before_the_run_poisons<C: Cells>() {
    let log = MemoryBackend::new();
    let store = open::<C, _>(log.clone());
    let x = store.input("x", 0_u32).unwrap();
    let other = open::<C, _>(log.clone());
    let ox = other.input("x", 0_u32).unwrap();
    let mut staged = store.staged_batch().unwrap();
    staged.set(x, 1).unwrap();
    let foreign = other.set(ox, 7).unwrap();
    assert!(matches!(staged.commit(), Err(Error::TailMoved { .. })));
    assert_eq!(
        log_after(&log, Position(foreign.0 - 1)).len(),
        1,
        "the run did not land"
    );
    drop(staged);
    assert_poisoned("set after a foreign run", store.set(x, 2));
}

fn a_memo_checkpoint_refuses_while_a_batch_is_staged<C: MemoCells>() {
    let store = Arc::new(open::<C, _>(MemoryBackend::new()));
    let x = store.input("x", 1_u64).unwrap();
    let xh = x.handle();
    let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(0xb1d)));
    let scaled = builder.table::<u32, u64>("scaled", 1);
    builder.define(scaled, move |ctx, k| ctx.rt().get(xh) * u64::from(*k));
    let memo = builder.open(&store, MemoryMemo::new()).unwrap();
    let q = memo.get_or_insert(scaled, 3);
    assert_eq!(memo.runtime().get(q), 3);

    let mut staged = store.staged_batch().unwrap();
    staged.set(x, 2).unwrap();
    assert_eq!(memo.runtime().get(q), 6);
    assert_refused("memo checkpoint", memo.checkpoint());
    staged.commit().unwrap();
    drop(staged);
    assert_eq!(memo.checkpoint().unwrap().written, 1);
}

fn a_shared_staged_batch_keeps_the_store_alive<C: Cells>() {
    let log = MemoryBackend::new();
    let store = Arc::new(open::<C, _>(log.clone()));
    let x = store.input("x", 0_u32).unwrap();
    let mut staged = store.staged_batch_shared().unwrap();
    let weak = Arc::downgrade(&store);
    drop(store);
    staged.set(x, 1).unwrap();
    let last = staged.commit().unwrap();
    drop(staged);
    assert!(weak.upgrade().is_none());
    assert_eq!(log_after(&log, Position(last.0 - 1)).len(), 1);
}

fn a_rolled_away_value_misses_on_restore<C: MemoCells>() {
    let log = MemoryBackend::new();
    let memo_backend = MemoryMemo::new();
    let runs = Arc::new(AtomicUsize::new(0));
    let open_both = |runs: &Arc<AtomicUsize>| {
        let store = Arc::new(open::<C, _>(log.clone()));
        let x = store.input("x", 1_u64).unwrap();
        let xh = x.handle();
        let mut builder = MemoBuilder::new(MemoOptions::new(BuildId::Fixed(0xb1d)));
        let scaled = builder.table::<u32, u64>("scaled", 1);
        let runs = Arc::clone(runs);
        builder.define(scaled, move |ctx, k| {
            runs.fetch_add(1, SeqCst);
            ctx.rt().get(xh) * u64::from(*k)
        });
        let memo = builder.open(&store, memo_backend.clone()).unwrap();
        (store, x, memo, scaled)
    };

    let (store, x, memo, scaled) = open_both(&runs);
    let q3 = memo.get_or_insert(scaled, 3);
    assert_eq!(memo.runtime().get(q3), 3);
    memo.checkpoint().unwrap();
    let base = store.stats().unwrap().position;

    let mut staged = store.staged_batch().unwrap();
    staged.set(x, 2).unwrap();
    let q4 = memo.get_or_insert(scaled, 4);
    assert_eq!(memo.runtime().get(q4), 8);
    assert_eq!(memo.runtime().get(q3), 6);
    drop(staged);
    assert_poisoned("set after the batch rolled away", store.set(x, 5));
    assert_refused(
        "memo checkpoint after the batch rolled away",
        memo.checkpoint(),
    );
    assert!(log_after(&log, base).is_empty(), "nothing reached the log");
    drop((memo, store));

    runs.store(0, SeqCst);
    let (store, x, memo, scaled) = open_both(&runs);
    assert_eq!(store.runtime().get(x.handle()), 1);
    let q3 = memo.get_or_insert(scaled, 3);
    let q4 = memo.get_or_insert(scaled, 4);
    assert_eq!(memo.runtime().get(q3), 3);
    assert_eq!(memo.runtime().get(q4), 4);
    assert_eq!(runs.load(SeqCst), 1, "only the rolled-away entry runs");
}

type SharedMemo = (
    Arc<Store<Shared, MemoryBackend>>,
    Memo<Shared, MemoryBackend>,
);

fn shared_memo_store_with_auto_checkpoint() -> SharedMemo {
    let store = Arc::new(open::<Shared, _>(MemoryBackend::new()));
    let x = store.input("x", 1_u64).unwrap();
    let xh = x.handle();
    let mut builder =
        MemoBuilder::new(MemoOptions::new(BuildId::Fixed(0xb1d)).auto_checkpoint_secs(1));
    let scaled = builder.table::<u32, u64>("scaled", 1);
    builder.define(scaled, move |ctx, k| ctx.rt().get(xh) * u64::from(*k));
    let memo = builder.open(&store, MemoryMemo::new()).unwrap();
    let q = memo.get_or_insert(scaled, 3);
    assert_eq!(memo.runtime().get(q), 3);
    (store, memo)
}

fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !done() {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn the_auto_checkpoint_skips_a_staged_batch_without_counting_an_error() {
    let (store, memo) = shared_memo_store_with_auto_checkpoint();
    let staged = store.staged_batch().unwrap();
    std::thread::sleep(Duration::from_millis(2500));
    let stats = memo.stats();
    assert_eq!(stats.auto_checkpoint_errors, 0);
    assert_eq!(stats.checkpoints, 0);
    drop(staged);
}

#[test]
fn the_auto_checkpoint_skips_an_unlogged_scope_without_counting_an_error() {
    let (store, memo) = shared_memo_store_with_auto_checkpoint();
    let scope = store.unlogged_scope().unwrap();
    std::thread::sleep(Duration::from_millis(2500));
    assert_eq!(memo.stats().auto_checkpoint_errors, 0);
    scope.close().unwrap();
    wait_until("no automatic checkpoint ran after the scope closed", || {
        memo.stats().checkpoints > 0
    });
}

macro_rules! both {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(#[test] fn $name() { super::$name::<super::Local>(); })*
        }
        mod shared {
            $(#[test] fn $name() { super::$name::<super::Shared>(); })*
        }
    };
}

both!(
    staged_writes_apply_at_once_and_commit_appends_them_once_in_write_order,
    staged_keyed_writes_go_through_pending_writes,
    every_other_store_call_refuses_while_a_batch_is_staged,
    a_staged_batch_refuses_while_a_scope_is_open,
    a_staged_batch_needs_a_writable_conditional_backend,
    opening_catches_up_and_an_empty_commit_appends_nothing,
    dropping_an_uncommitted_batch_poisons,
    a_panic_inside_the_batch_poisons,
    subscribers_hear_each_record_at_commit,
    an_append_error_with_nothing_landed_keeps_the_run_for_retry,
    an_append_error_after_the_run_landed_is_adopted_on_retry,
    a_foreign_record_before_the_run_poisons,
    a_memo_checkpoint_refuses_while_a_batch_is_staged,
    a_shared_staged_batch_keeps_the_store_alive,
    a_rolled_away_value_misses_on_restore,
);
