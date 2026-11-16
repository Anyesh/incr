"""R17: memo checkpoints past chain depth 8 compact the delta chain into a new
base. Runs through the Python `MemoryMemo` and through a pure-Python backend
that implements `read_entries`, `write_generation_streamed` and the delta
pair, on both packages. Each run has a deadline, and the restored entries
must equal those of a store that only ever checkpointed in full."""

import faulthandler
import json
import threading

import incr.persist
import incr_concurrent.persist
import pytest
from test_mcs_persist_conformance import DictDeltaMemoFactory

PACKAGES = [
    pytest.param(incr.persist, id="local"),
    pytest.param(incr_concurrent.persist, id="shared"),
]

TIMEOUT_SECS = 60
BASE_KEYS = 50
DEPTH = 8
CHECKPOINTS = 3 * DEPTH


@pytest.fixture(params=PACKAGES)
def persist(request):
    return request.param


class JsonCodec:
    def encode(self, obj):
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)


class PythonBackend:
    def __init__(self, persist):
        self.factory = DictDeltaMemoFactory(persist)
        self.state = self.factory.new_store()

    def open(self):
        return self.factory.open(self.state)


class MemoryBackendFactory:
    def __init__(self, persist):
        self.memo = persist.MemoryMemo()

    def open(self):
        return self.memo


BACKENDS = [
    pytest.param(MemoryBackendFactory, id="memory-memo"),
    pytest.param(PythonBackend, id="python-streamed"),
]


class Graph:
    def __init__(self, persist, log, memo_backend):
        self.store = persist.Store.open(log, build_id="b1")
        self.calls = []

        def square(ctx, key):
            self.calls.append(key)
            return key * key

        self.table = self.store.memo_table("sq", 1, JsonCodec(), JsonCodec(), square)
        self.memo = self.store.open_memo(memo_backend)

    def read(self, key):
        return self.store.runtime.get(self.memo.get_or_insert(self.table, key))

    def read_all(self, keys):
        return [self.read(key) for key in keys]


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


def compact_and_restore(persist, backend):
    log = persist.MemoryBackend()
    graph = Graph(persist, log, backend.open())
    graph.read_all(range(BASE_KEYS))
    first = graph.memo.checkpoint()
    assert first["delta"] is False

    depths = []
    compactions = 0
    key = BASE_KEYS
    for _ in range(CHECKPOINTS):
        graph.read(key)
        key += 1
        counts = graph.memo.checkpoint()
        assert counts["fresh_written"] == 1
        depths.append(counts["chain_depth"])
        if counts["compacted"]:
            compactions += 1
            assert counts["delta"] is False
            assert counts["chain_depth"] == 0
    assert compactions >= 1
    assert max(depths) <= DEPTH
    stats = graph.memo.stats()
    assert stats["compactions"] == compactions
    assert stats["unsaved_entries"] == 0
    graph.store.close()

    again = Graph(persist, log, backend.open())
    expected = [k * k for k in range(key)]
    assert again.read_all(range(key)) == expected
    assert again.calls == []
    entries = again.memo.stats()["entries"]
    again.store.close()
    return key, entries


def full_restore(persist, count):
    log = persist.MemoryBackend()
    memo = persist.MemoryMemo()
    graph = Graph(persist, log, memo)
    graph.read_all(range(count))
    counts = graph.memo.checkpoint()
    assert counts["delta"] is False
    graph.store.close()

    again = Graph(persist, log, memo)
    values = again.read_all(range(count))
    assert again.calls == []
    entries = again.memo.stats()["entries"]
    again.store.close()
    return values, entries


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_a_chain_past_depth_eight_compacts_and_restores_like_a_full_checkpoint(
    persist, backend_class
):
    result = []

    def run():
        result.append(compact_and_restore(persist, backend_class(persist)))

    finishes(run)
    count, entries = result[0]
    values, full_entries = full_restore(persist, count)
    assert values == [k * k for k in range(count)]
    assert entries == full_entries == count


def test_the_python_backend_is_asked_for_a_streamed_write_on_compaction(persist):
    backend = PythonBackend(persist)

    def run():
        compact_and_restore(persist, backend)

    finishes(run)
    names = backend.state.calls
    assert "write_generation_streamed" in names
    assert names.count("write_generation_streamed") >= 2
