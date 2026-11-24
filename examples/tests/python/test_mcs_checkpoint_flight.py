"""R24: `store.begin_checkpoint()` and `memo.begin_checkpoint()` start a
checkpoint whose copying runs in `poll` slices on the caller and whose
encode and write run on a worker thread. Local wheel only; a backend must
declare `concurrent_compaction = True` and have the streamed write."""

import gc
import json
import threading
import time

import incr.persist
import incr_concurrent.persist
import pytest

persist = incr.persist
DEADLINE = 60


class JsonCodec:
    def encode(self, obj):
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)


class Delegating:
    concurrent_compaction = True

    def __init__(self, inner=None):
        self.inner = inner or persist.MemoryBackend()

    def append(self, records):
        return self.inner.append(records)

    def sync(self, upto):
        return self.inner.sync(upto)

    def read_from(self, after):
        return self.inner.read_from(after)

    def load_snapshot(self):
        return self.inner.load_snapshot()

    def snapshot_id(self):
        return self.inner.snapshot_id()

    def writable(self):
        return self.inner.writable()

    def compact(self, upto, records):
        return self.inner.compact(upto, records)

    def compact_streamed(self, upto, records):
        return self.inner.compact_streamed(upto, records)


class Gated(Delegating):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.go = threading.Event()

    def compact_streamed(self, upto, records):
        self.entered.set()
        assert self.go.wait(DEADLINE)
        return self.inner.compact_streamed(upto, records)


class NoFlag(Delegating):
    concurrent_compaction = False


class FlagWithoutStream:
    concurrent_compaction = True

    def __init__(self):
        self.inner = persist.MemoryBackend()

    def __getattr__(self, name):
        if name == "compact_streamed":
            raise AttributeError(name)
        return getattr(self.inner, name)


def open_store(backend):
    store = persist.Store.open(backend, build_id="b1")
    base = store.input("base", 1, JsonCodec())
    rows = store.collection("rows", JsonCodec())
    keyed = store.keyed_collection("keyed", JsonCodec(), JsonCodec())
    return store, base, rows, keyed


def fill(store, base, rows, keyed, n):
    store.set(base, n)
    for i in range(n):
        store.insert(rows, f"row{i}")
        store.upsert(keyed, f"k{i}", {"i": i})


def contents(backend):
    store, base, rows, keyed = open_store(backend)
    rt = store.runtime
    snapshot = (
        rt.get(base),
        sorted(map(json.dumps, rt.get(rows.reduce(sorted)))),
        sorted((k, json.dumps(v)) for k, v in keyed.items()),
    )
    store.close()
    return snapshot


def poll_to_end(pending, budget=0.05):
    deadline = time.monotonic() + DEADLINE
    while not pending.poll(budget):
        assert time.monotonic() < deadline, "flight did not finish"
        time.sleep(0.001)


def settle_until_checkpoint(store):
    deadline = time.monotonic() + DEADLINE
    while True:
        try:
            return store.checkpoint()
        except persist.Busy:
            assert time.monotonic() < deadline, "store stayed busy"
            time.sleep(0.005)


def test_the_reference_backends_declare_concurrent_compaction():
    assert persist.MemoryBackend.concurrent_compaction is True
    assert persist.MemoryMemo.concurrent_compaction is True


def test_a_store_flight_writes_what_a_checkpoint_writes():
    backend = Delegating()
    store, base, rows, keyed = open_store(backend)
    fill(store, base, rows, keyed, 300)
    pending = store.begin_checkpoint()
    assert isinstance(pending, persist.PendingCheckpoint)
    store.insert(rows, "after-begin")
    poll_to_end(pending)
    assert pending.result() == store.stats()["snapshot_position"]
    expected = contents(backend)
    assert "after-begin" in [json.loads(r) for r in expected[1]]
    store.close()

    twin = Delegating()
    store, base, rows, keyed = open_store(twin)
    fill(store, base, rows, keyed, 300)
    store.insert(rows, "after-begin")
    store.checkpoint()
    store.close()
    assert contents(twin) == expected


def test_checkpoint_is_busy_while_a_flight_is_open():
    backend = Gated()
    store, base, rows, keyed = open_store(backend)
    fill(store, base, rows, keyed, 10)
    pending = store.begin_checkpoint()
    with pytest.raises(persist.Busy):
        store.checkpoint()
    with pytest.raises(persist.Busy):
        store.begin_checkpoint()
    backend.go.set()
    assert pending.result(timeout=DEADLINE) > 0
    store.checkpoint()


def test_result_with_a_timeout_that_expires_raises_and_a_later_poll_finishes():
    backend = Gated()
    store, base, rows, keyed = open_store(backend)
    fill(store, base, rows, keyed, 10)
    pending = store.begin_checkpoint()
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        pending.result(timeout=0.2)
    assert time.monotonic() - started < DEADLINE
    assert backend.entered.wait(DEADLINE)
    backend.go.set()
    poll_to_end(pending)
    assert store.stats()["records_since_checkpoint"] == 0


def test_leaving_a_with_block_abandons_the_flight():
    backend = Gated()
    store, base, rows, keyed = open_store(backend)
    fill(store, base, rows, keyed, 10)
    with store.begin_checkpoint() as pending:
        assert isinstance(pending, persist.PendingCheckpoint)
    backend.go.set()
    store.set(base, 99)
    settle_until_checkpoint(store)
    assert contents(backend)[0] == 99


def test_a_handle_dropped_on_a_second_thread_ends_the_flight():
    backend = Gated()
    store, base, rows, keyed = open_store(backend)
    fill(store, base, rows, keyed, 10)
    holder = [store.begin_checkpoint()]

    def drop_it():
        holder.pop()
        gc.collect()

    thread = threading.Thread(target=drop_it)
    thread.start()
    thread.join(DEADLINE)
    assert not holder
    backend.go.set()
    store.set(base, 5)
    settle_until_checkpoint(store)
    assert contents(backend)[0] == 5


def test_a_handle_is_used_only_on_the_thread_that_began_the_flight():
    store, base, rows, keyed = open_store(Delegating())
    fill(store, base, rows, keyed, 10)
    pending = store.begin_checkpoint()
    raised = []

    def use_it():
        calls = (
            lambda: pending.poll(),
            lambda: pending.result(),
            pending.cancel,
            lambda: pending.__exit__(None, None, None),
        )
        for call in calls:
            try:
                call()
            except RuntimeError as error:
                raised.append(str(error))

    thread = threading.Thread(target=use_it)
    thread.start()
    thread.join(DEADLINE)
    assert len(raised) == 4
    poll_to_end(pending)


def test_a_cancelled_flight_raises_and_the_store_checkpoints_after():
    backend = Gated()
    store, base, rows, keyed = open_store(backend)
    fill(store, base, rows, keyed, 10)
    pending = store.begin_checkpoint()
    pending.cancel()
    backend.go.set()
    with pytest.raises(persist.PersistError):
        poll_to_end(pending)
    assert store.checkpoint() > 0


@pytest.mark.parametrize("backend", [NoFlag, FlagWithoutStream])
def test_a_store_flight_refuses_a_backend_without_the_promise(backend):
    store, base, rows, keyed = open_store(backend())
    fill(store, base, rows, keyed, 3)
    with pytest.raises(persist.PersistError, match="concurrent"):
        store.begin_checkpoint()
    store.checkpoint()


def test_a_codec_failure_on_the_worker_raises_the_codecs_exception():
    class Refusing(JsonCodec):
        armed = False

        def encode(self, obj):
            if self.armed and obj == "poison":
                raise KeyError("refused on the worker")
            return super().encode(obj)

    codec = Refusing()
    store = persist.Store.open(Delegating(), build_id="b1")
    rows = store.collection("rows", codec)
    store.insert(rows, "poison")
    codec.armed = True
    pending = store.begin_checkpoint()
    with pytest.raises(KeyError, match="refused on the worker"):
        poll_to_end(pending)


def test_the_shared_wheel_has_no_flights():
    store = incr_concurrent.persist.Store.open(
        incr_concurrent.persist.MemoryBackend(), build_id="b1"
    )
    assert not hasattr(store, "begin_checkpoint")


class ThreadRecording:
    concurrent_compaction = True

    def __init__(self):
        self.inner = persist.MemoryMemo()
        self.local = threading.local()
        self.seen = []
        self.next_id = 0

    def note(self, method):
        if not hasattr(self.local, "id"):
            self.next_id += 1
            self.local.id = self.next_id
        self.seen.append((method, threading.get_ident(), self.local.id))

    def __getattr__(self, name):
        method = getattr(self.inner, name)

        def call(*args):
            self.note(name)
            return method(*args)

        return call


def memo_graph(backend, n):
    store = persist.Store.open(persist.MemoryBackend(), build_id="b1")
    base = store.input("base", 7, JsonCodec())

    def sq(ctx, k):
        return k * k + ctx.get(base)

    table = store.memo_table("sq", 1, JsonCodec(), JsonCodec(), sq)
    memo = store.open_memo(backend)
    for k in range(n):
        assert store.runtime.get(memo.get_or_insert(table, k)) == k * k + 7
    return store, memo, table


def test_a_memo_flight_writes_entries_a_reopen_restores():
    backend = persist.MemoryMemo()
    store, memo, _ = memo_graph(backend, 500)
    pending = memo.begin_checkpoint()
    poll_to_end(pending)
    stats = pending.result()
    assert stats["written"] == 500
    store2, memo2, table2 = memo_graph(backend, 0)
    for k in range(500):
        assert store2.runtime.get(memo2.get_or_insert(table2, k)) == k * k + 7
    assert memo2.stats()["entries"] == 500


def test_a_worker_keeps_one_thread_state_for_its_life():
    backend = ThreadRecording()
    store, memo, table = memo_graph(backend, 300)
    memo.checkpoint()
    for k in range(300, 600):
        store.runtime.get(memo.get_or_insert(table, k))
    backend.seen.clear()
    pending = memo.begin_checkpoint()
    poll_to_end(pending)
    worker = [(m, t, i) for m, t, i in backend.seen if t != threading.get_ident()]
    assert len({m for m, _, _ in worker}) >= 2, backend.seen
    assert len({(t, i) for _, t, i in worker}) == 1, worker


def test_a_memo_flight_refuses_a_memo_backend_without_the_flag():
    class Unflagged:
        def __init__(self):
            self.inner = persist.MemoryMemo()

        def __getattr__(self, name):
            return getattr(self.inner, name)

    store, memo, _ = memo_graph(Unflagged(), 3)
    with pytest.raises(persist.PersistError, match="concurrent"):
        memo.begin_checkpoint()
    memo.checkpoint()


def test_a_memo_handle_dropped_on_a_second_thread_lets_checkpoint_run():
    store, memo, _ = memo_graph(persist.MemoryMemo(), 200)
    holder = [memo.begin_checkpoint()]
    thread = threading.Thread(target=holder.pop)
    thread.start()
    thread.join(DEADLINE)
    deadline = time.monotonic() + DEADLINE
    while True:
        try:
            memo.checkpoint()
            break
        except persist.Busy:
            assert time.monotonic() < deadline
            time.sleep(0.005)
    assert memo.stats()["unsaved_entries"] == 0
