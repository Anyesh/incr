"""R40: a store or memo that ran a checkpoint flight frees its codecs and
backend once every handle is deleted, with no later incr call. A checkpoint
thread can hold the last reference to them; released there without the
GIL, PyO3 would only queue the decref for the next thread that attaches."""

import gc
import json
import time
import weakref

import incr.persist
import pytest
from test_mcs_checkpoint_flight import DEADLINE, Gated, GatedMemo, poll_to_end

persist = incr.persist


class Codec:
    def encode(self, obj):
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)


class Raising:
    concurrent_compaction = True

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        if name in ("compact_streamed", "write_generation_streamed"):

            def refuse(*args):
                raise OSError("refused by the backend")

            return refuse
        return getattr(self.inner, name)


class RefusingGate(Gated):
    def compact_streamed(self, upto, records):
        self.entered.set()
        assert self.go.wait(DEADLINE)
        raise OSError("refused after the gate")


class RefusingMemoGate(GatedMemo):
    def write_generation_streamed(self, *args):
        self.entered.set()
        assert self.go.wait(DEADLINE)
        raise OSError("refused after the gate")


def assert_freed(refs):
    gc.collect()
    deadline = time.monotonic() + 10
    while any(ref() is not None for ref in refs):
        assert time.monotonic() < deadline, [ref() for ref in refs]
        time.sleep(0.001)


def open_store(backend, codec):
    store = persist.Store.open(backend, build_id="b1")
    base = store.input("base", 1, codec)
    store.set(base, 2)
    return store


def open_memo(backend, codec):
    store = persist.Store.open(persist.MemoryBackend(), build_id="b1")
    table = store.memo_table("sq", 1, codec, codec, lambda ctx, k: k * k)
    memo = store.open_memo(backend)
    for k in range(20):
        store.runtime.get(memo.get_or_insert(table, k))
    return store, memo, table


def wait_for_write(pending, backend):
    deadline = time.monotonic() + DEADLINE
    while not backend.entered.is_set():
        assert not pending.poll(0.001)
        assert time.monotonic() < deadline, "the worker did not reach the write"


@pytest.mark.parametrize(
    "how", ["completed", "cancelled", "abandoned", "abandoned_failing", "failed"]
)
def test_a_store_flight_frees_its_codec_and_backend(how):
    codec = Codec()
    backend = {
        "failed": lambda: Raising(Gated()),
        "abandoned_failing": RefusingGate,
    }.get(how, Gated)()
    go = (backend.inner if how == "failed" else backend).go
    store = open_store(backend, codec)
    pending = store.begin_checkpoint()
    if how == "completed":
        go.set()
        poll_to_end(pending)
        pending.result()
    elif how == "failed":
        with pytest.raises(OSError):
            poll_to_end(pending)
    elif how == "cancelled":
        wait_for_write(pending, backend)
        pending.cancel()
        go.set()
        with pytest.raises(persist.PersistError):
            poll_to_end(pending)
    else:
        wait_for_write(pending, backend)
    refs = [weakref.ref(codec), weakref.ref(backend)]
    del pending, store, codec, backend
    go.set()
    assert_freed(refs)


@pytest.mark.parametrize(
    "how", ["completed", "cancelled", "abandoned", "abandoned_failing", "failed"]
)
def test_a_memo_flight_frees_its_codec_and_backend(how):
    codec = Codec()
    backend = {
        "failed": lambda: Raising(GatedMemo()),
        "abandoned_failing": RefusingMemoGate,
    }.get(how, GatedMemo)()
    go = (backend.inner if how == "failed" else backend).go
    store, memo, table = open_memo(backend, codec)
    pending = memo.begin_checkpoint()
    if how == "completed":
        go.set()
        poll_to_end(pending)
        pending.result()
    elif how == "failed":
        with pytest.raises(OSError):
            poll_to_end(pending)
    elif how == "cancelled":
        pending.cancel()
        with pytest.raises(persist.PersistError):
            poll_to_end(pending)
    else:
        wait_for_write(pending, backend)
    refs = [weakref.ref(codec), weakref.ref(backend)]
    del pending, store, memo, table, codec, backend
    go.set()
    assert_freed(refs)
