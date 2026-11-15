"""R17: a compaction writes the next full generation through the backend's
`write_generation_streamed`, whose `frames` read the old chain back through
`read_entries` on the same handle while the call runs. On Local the
checkpoint runs on the calling thread. On Shared the checkpoint thread runs
it, and its first read of the chain mid-write waits until the main thread
has opened a second store on the same backend and restored entries from
it, so a lock taken in the wrong order against the GIL would hang. Each run
has a deadline, and a hang that holds the GIL ends the process."""

import faulthandler
import json
import threading
import time

import incr.persist
import incr_concurrent.persist
import pytest

PACKAGES = [
    pytest.param(incr.persist, id="local"),
    pytest.param(incr_concurrent.persist, id="shared"),
]

TIMEOUT_SECS = 60
STEP = 200
DEPTH = 8


@pytest.fixture(params=PACKAGES)
def persist(request):
    return request.param


class JsonCodec:
    def encode(self, obj):
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)


class Watched:
    """A Python `MemoryMemo` that counts its streamed writes that replace a
    chain holding deltas, and the frames such a write reads back. With
    `overlap` set to an Event, the write's first read of the chain sets
    `parked` and waits for it."""

    def __init__(self, persist):
        self.inner = persist.MemoryMemo()
        self.compactions = 0
        self.read_while_writing = 0
        self.writing = False
        self.writer = None
        self.overlap = None
        self.parked = threading.Event()
        self.overlapped = False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def read_entries(self, generation, items):
        if self.writing and threading.get_ident() == self.writer:
            if self.overlap is not None and not self.parked.is_set():
                self.parked.set()
                self.overlapped = self.overlap.wait(TIMEOUT_SECS)
            self.read_while_writing += len(items)
        return self.inner.read_entries(generation, items)

    def write_generation_streamed(self, base, generation, header, frames):
        chain = self.inner.load_chain()
        compacting = chain is not None and len(chain) > 1
        self.writer = threading.get_ident()
        self.writing = compacting
        try:
            items = self.inner.write_generation_streamed(
                base, generation, header, frames
            )
        finally:
            self.writing = False
        if compacting:
            self.compactions += 1
        return items


def graph(persist, watched, **memo_options):
    store = persist.Store.open(persist.MemoryBackend(), build_id="b1")
    base = store.input("base", 7, JsonCodec())
    calls = []

    def sq(ctx, k):
        calls.append(k)
        return k * k + ctx.get(base)

    table = store.memo_table("sq", 1, JsonCodec(), JsonCodec(), sq)
    memo = store.open_memo(watched, **memo_options)

    def read(k):
        value = store.runtime.get(memo.get_or_insert(table, k))
        assert value == k * k + 7, k
        return value

    return store, memo, read, calls


def compact_on_local(persist, watched):
    store, memo, read, _ = graph(persist, watched)
    for k in range(2 * STEP):
        read(k)
    memo.checkpoint()
    k = 2 * STEP
    for _ in range(DEPTH + 1):
        read(k)
        k += 1
        counts = memo.checkpoint()
    assert watched.compactions == 1
    assert counts["merged"] == 2 * STEP + DEPTH
    for k in range(2 * STEP + DEPTH + 1):
        read(k)
    store.close()


def compact_on_the_checkpoint_thread(persist, watched):
    watched.overlap = threading.Event()
    store, memo, read, _ = graph(persist, watched, checkpoint_after_entries=STEP)
    deadline = time.monotonic() + TIMEOUT_SECS
    k = 0
    while watched.compactions == 0:
        assert time.monotonic() < deadline, "the checkpoint thread never compacted"
        if watched.parked.is_set() and not watched.overlap.is_set():
            restore_during_the_write(persist, watched)
            read(0)
            read(k)
            k += 1
            watched.overlap.set()
        # Each checkpoint takes about STEP entries, so once the base holds
        # four times that, checkpoints write deltas and the chain grows.
        if memo.stats()["unsaved_entries"] < STEP:
            read(k)
            k += 1
        else:
            time.sleep(0.001)
    assert watched.overlapped
    for j in range(k):
        read(j)
    assert memo.stats()["auto_checkpoint_errors"] == 0
    store.close()


def restore_during_the_write(persist, watched):
    """Opens a second store on `watched` while a compaction is writing and
    restores the entries the chain holds, without running any of them."""
    store, _, read, calls = graph(persist, watched)
    for k in range(STEP):
        read(k)
    assert calls == []
    store.close()


def finishes(run, *args):
    failure = []

    def body():
        try:
            run(*args)
        except BaseException as error:
            failure.append(error)

    worker = threading.Thread(target=body, daemon=True)
    faulthandler.dump_traceback_later(2 * TIMEOUT_SECS, exit=True)
    try:
        worker.start()
        worker.join(TIMEOUT_SECS)
    finally:
        faulthandler.cancel_dump_traceback_later()
    assert not worker.is_alive(), f"{run.__name__} did not finish"
    if failure:
        raise failure[0]


def test_a_compaction_streams_the_chain_back_through_the_same_memory_memo(persist):
    watched = Watched(persist)
    if persist is incr.persist:
        finishes(compact_on_local, persist, watched)
    else:
        finishes(compact_on_the_checkpoint_thread, persist, watched)
    assert watched.compactions >= 1
    assert watched.read_while_writing >= 2 * STEP
