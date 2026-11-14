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
