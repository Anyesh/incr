"""The Shared store across Python threads: a batch stays on the thread that
opened it, drops never wait on the store with the GIL held, and several
threads write and read one store."""

import faulthandler
import gc
import pickle
import sys
import threading
import time

import pytest
from incr_concurrent.persist import MemoryBackend, MemoryMemo, PersistError, Store


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


def in_thread(f):
    out = {}

    def run():
        try:
            out["result"] = f()
        except BaseException as e:
            out["error"] = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if "error" in out:
        raise out["error"]
    return out.get("result")


@pytest.mark.parametrize("apply", ["on_exit", "on_write"])
def test_a_batch_raises_on_a_thread_other_than_its_opener(apply):
    store = Store.open(MemoryBackend(), build_id=1)
    price = store.input("price", 0, PickleCodec())
    with store.batch(apply=apply) as b:
        with pytest.raises(RuntimeError, match="thread"):
            in_thread(lambda: b.set(price, 1))
        b.set(price, 2)
    assert store.runtime.get(price) == 2
    unopened = store.batch()
    with pytest.raises(RuntimeError, match="thread"):
        in_thread(unopened.__enter__)


class SlowMemo:
    """A memo backend whose `write_generation` signals `started` and waits
    for `release`, so a checkpoint thread is inside it, needing the GIL to
    finish, while the test drops the memo."""

    def __init__(self):
        self.inner = MemoryMemo()
        self.started = threading.Event()
        self.release = threading.Event()

    def lock_checkpoint(self):
        return self.inner.lock_checkpoint()

    def load(self):
        return self.inner.load()

    def read_entry(self, *args):
        return self.inner.read_entry(*args)

    def write_generation(self, *args):
        self.started.set()
        self.release.wait(10)
        return self.inner.write_generation(*args)

    def remove_generation(self, *args):
        return self.inner.remove_generation(*args)


@pytest.fixture
def watchdog():
    faulthandler.dump_traceback_later(30, exit=True, file=sys.__stderr__)
    yield
    faulthandler.cancel_dump_traceback_later()


def store_with_checkpointing_memo():
    memo_backend = SlowMemo()
    store = Store.open(MemoryBackend(), build_id=1)
    table = store.memo_table("sq", 1, PickleCodec(), PickleCodec(), lambda ctx, k: k * k)
    memo = store.open_memo(memo_backend, checkpoint_after_entries=1)
    assert store.runtime.get(memo.get_or_insert(table, 3)) == 9
    assert memo_backend.started.wait(10)
    return store, memo, table, memo_backend


@pytest.mark.parametrize("drop", ["close", "dealloc"])
def test_dropping_a_memo_whose_thread_is_checkpointing_does_not_deadlock(watchdog, drop):
    store, memo, table, memo_backend = store_with_checkpointing_memo()
    del memo, table

    def drop_store():
        nonlocal store
        if drop == "close":
            store.close()
        else:
            store = None
            gc.collect()

    dropper = threading.Thread(target=drop_store)
    dropper.start()
    time.sleep(0.2)
    memo_backend.release.set()
    dropper.join(10)
    assert not dropper.is_alive()


def test_dropping_an_open_on_write_batch_poisons_the_store(watchdog):
    store = Store.open(MemoryBackend(), build_id=1)
    price = store.input("price", 0, PickleCodec())
    b = store.batch(apply="on_write")
    b.__enter__()
    b.set(price, 1)
    del b
    gc.collect()
    with pytest.raises(PersistError):
        store.set(price, 2)


class SlowLoadMemo(SlowMemo):
    def load(self):
        self.started.set()
        self.release.wait(10)
        return self.inner.load()


def test_a_memo_table_declared_while_the_memo_opens_raises(watchdog):
    store = Store.open(MemoryBackend(), build_id=1)
    store.memo_table("sq", 1, PickleCodec(), PickleCodec(), lambda ctx, k: k * k)
    memo_backend = SlowLoadMemo()
    opener = threading.Thread(target=store.open_memo, args=(memo_backend,))
    opener.start()
    assert memo_backend.started.wait(10)
    outcome = {}

    def declare():
        try:
            outcome["table"] = store.memo_table(
                "cube", 1, PickleCodec(), PickleCodec(), lambda ctx, k: k**3
            )
        except RuntimeError as e:
            outcome["error"] = e

    declarer = threading.Thread(target=declare)
    declarer.start()
    time.sleep(0.2)
    memo_backend.release.set()
    opener.join(10)
    declarer.join(10)
    assert "open_memo" in str(outcome.get("error"))


def wait_for(condition, timeout=20):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


def test_the_checkpoint_thread_runs_a_due_memo_checkpoint():
    log, memo_backend = MemoryBackend(), MemoryMemo()
    calls = []

    def square(ctx, k):
        calls.append(k)
        return k * k

    def open_graph():
        store = Store.open(log, build_id=1)
        table = store.memo_table("sq", 1, PickleCodec(), PickleCodec(), square)
        memo = store.open_memo(memo_backend, checkpoint_after_entries=3)
        return store, table, memo

    store, table, memo = open_graph()
    for k in range(3):
        assert store.runtime.get(memo.get_or_insert(table, k)) == k * k
    wait_for(lambda: memo.stats()["entries"] == 3)
    assert memo.stats()["unsaved_entries"] == 0
    store.close()

    calls.clear()
    store, table, memo = open_graph()
    assert [store.runtime.get(memo.get_or_insert(table, k)) for k in range(3)] == [0, 1, 4]
    assert calls == []
    store.close()


def test_an_on_write_batch_is_visible_to_another_thread_before_it_lands():
    store = Store.open(MemoryBackend(), build_id=1)
    price = store.input("price", 0, PickleCodec())
    position = store.stats()["position"]
    with store.batch(apply="on_write") as b:
        b.set(price, 5)
        assert in_thread(lambda: store.runtime.get(price)) == 5
        assert store.stats()["position"] == position
    assert store.stats()["position"] == position + 1
    assert "other threads" in Store.batch.__doc__


def test_threads_writing_keys_batches_and_reading_memo_entries_finish(watchdog):
    store = Store.open(MemoryBackend(), build_id=1, auto_checkpoint_records=7)
    codec = PickleCodec()
    shifts = store.keyed_collection("shifts", codec, codec)
    hours = store.memo_table(
        "hours", 1, codec, codec, lambda ctx, k: (ctx.get_key(shifts, k) or 0) * 2
    )
    memo = store.open_memo(MemoryMemo(), checkpoint_after_entries=5)
    keys = [f"k{i}" for i in range(4)]
    errors = []

    def write(key):
        try:
            for value in range(1, 16):
                if value % 3:
                    store.upsert(shifts, key, value)
                else:
                    with store.batch() as b:
                        b.upsert(shifts, key, value)
                store.runtime.get(memo.get_or_insert(hours, key))
        except Exception as error:
            errors.append(error)

    def read():
        try:
            for _ in range(30):
                for key in keys:
                    store.runtime.get(memo.get_or_insert(hours, key))
                store.stats()
                memo.stats()
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=write, args=(key,)) for key in keys]
    threads += [threading.Thread(target=read) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert [store.runtime.get(memo.get_or_insert(hours, key)) for key in keys] == [30] * 4
    store.close()


def test_a_call_holding_the_memo_when_the_store_closes_does_not_deadlock(watchdog):
    memo_backend = SlowMemo()
    store = Store.open(MemoryBackend(), build_id=1)
    entered, gate = threading.Event(), threading.Event()

    def square(ctx, k):
        if k == 5:
            entered.set()
            gate.wait(10)
        return k * k

    table = store.memo_table("sq", 1, PickleCodec(), PickleCodec(), square)
    memo = store.open_memo(memo_backend, checkpoint_after_entries=1)
    rt = store.runtime
    assert rt.get(memo.get_or_insert(table, 0)) == 0
    assert memo_backend.started.wait(10)
    query = rt.create_query(lambda ctx: ctx.memo(table, 5))
    reader = threading.Thread(target=lambda: rt.get(query))
    reader.start()
    assert entered.wait(10)
    del memo
    store.close()
    gate.set()
    time.sleep(0.2)
    memo_backend.release.set()
    reader.join(10)
    assert not reader.is_alive()


def test_memo_calls_from_python_threads_while_the_checkpoint_thread_runs_finish(watchdog):
    store = Store.open(MemoryBackend(), build_id=1)
    codec = PickleCodec()
    table = store.memo_table("sq", 1, codec, codec, lambda ctx, k: k * k)
    memo = store.open_memo(MemoryMemo(), checkpoint_after_entries=2)
    rt = store.runtime
    computing = threading.Event()
    errors = []

    def compute(base):
        try:
            for k in range(base, base + 400):
                assert rt.get(memo.get_or_insert(table, k)) == k * k
        except Exception as error:
            errors.append(error)

    def poll():
        try:
            while not computing.is_set():
                memo.checkpoint_if_due()
                memo.stats()
        except Exception as error:
            errors.append(error)

    pollers = [threading.Thread(target=poll) for _ in range(2)]
    computers = [threading.Thread(target=compute, args=(i * 1000,)) for i in range(3)]
    for thread in pollers + computers:
        thread.start()
    for thread in computers:
        thread.join(30)
    computing.set()
    for thread in pollers:
        thread.join(30)
    assert not any(thread.is_alive() for thread in pollers + computers)
    assert errors == []
    store.close()


class UserError(Exception):
    pass


def test_abandoning_an_on_write_batch_raises_the_blocks_exception_not_a_stored_routing_failure(
    watchdog,
):
    store = Store.open(MemoryBackend(), build_id=1)
    codec = PickleCodec()
    rows = store.collection("rows", codec)

    def key(row):
        if row == "bad":
            raise UserError("no key")
        return row

    rt = store.runtime
    store.insert(rows, "ok")
    count = rows.group_by(key).get_group("ok").count()
    assert rt.get(count) == 1
    with pytest.raises(UserError, match="the block"):
        with store.batch(apply="on_write") as b:
            b.insert(rows, "bad")
            raise UserError("the block")
    with pytest.raises(RuntimeError, match="routing failed"):
        rt.node_count()
    with pytest.raises(PersistError):
        store.insert(rows, "ok")
