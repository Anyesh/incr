"""R17: a compaction writes the next full generation through the backend's
`write_generation_streamed`, whose `frames` read the old chain back through
`read_entries` on the same handle while the call runs. On Local the
checkpoint runs on the calling thread; on Shared the checkpoint thread runs
it while another thread keeps reading memo entries, so a lock taken in the
wrong order against the GIL would hang. Each run has a deadline."""

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
    chain holding deltas, and the frames read back while one runs."""

    def __init__(self, persist):
        self.inner = persist.MemoryMemo()
        self.compactions = 0
        self.read_while_writing = 0
        self.writing = False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def read_entries(self, generation, items):
        if self.writing:
            self.read_while_writing += len(items)
        return self.inner.read_entries(generation, items)

    def write_generation_streamed(self, base, generation, header, frames):
        chain = self.inner.load_chain()
        compacting = chain is not None and len(chain) > 1
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
    table = store.memo_table(
        "sq", 1, JsonCodec(), JsonCodec(), lambda ctx, k: k * k + ctx.get(base)
    )
    memo = store.open_memo(watched, **memo_options)

    def read(k):
        value = store.runtime.get(memo.get_or_insert(table, k))
        assert value == k * k + 7, k
        return value

    return store, memo, read


def compact_on_local(persist, watched):
    store, memo, read = graph(persist, watched)
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
    store, _, read = graph(persist, watched, checkpoint_after_entries=STEP)
    deadline = time.monotonic() + TIMEOUT_SECS
    k = 0
    while watched.compactions == 0:
        assert time.monotonic() < deadline, "the checkpoint thread never compacted"
        read(k)
        k += 1
    for j in range(k):
        read(j)
    store.close()


def finishes(run, *args):
    failure = []

    def body():
        try:
            run(*args)
        except BaseException as error:
            failure.append(error)

    worker = threading.Thread(target=body, daemon=True)
    worker.start()
    worker.join(TIMEOUT_SECS)
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
