"""R17 C8: the delta fields of `memo.checkpoint()` and `memo.stats()`, and the
`known` argument of a Python backend's `load_chain`."""

import json
import time

import incr.persist
import incr_concurrent.persist
import pytest
from test_mcs_persist_conformance import (
    DictDeltaMemo,
    DictDeltaMemoFactory,
    DictMemoFactory,
)

PACKAGES = [
    pytest.param(incr.persist, id="local"),
    pytest.param(incr_concurrent.persist, id="shared"),
]

CHECKPOINT_INTS = {
    "written",
    "unkept",
    "merged",
    "corrupt",
    "evicted",
    "oversized",
    "unchanged",
    "fresh_written",
    "tombstones",
    "bytes_appended",
    "chain_depth",
}
CHECKPOINT_BOOLS = {"delta", "compacted"}
TIMES = {"snapshot_ns", "merge_ns", "write_ns"}
STATS_INTS = {
    "chain_depth",
    "chain_bytes",
    "delta_checkpoints",
    "compactions",
    "last_bytes_appended",
}


@pytest.fixture(params=PACKAGES)
def persist(request):
    return request.param


class JsonCodec:
    def encode(self, obj):
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)


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


def test_a_delta_checkpoint_writes_only_the_dirty_entries(persist):
    log, backend = persist.MemoryBackend(), persist.MemoryMemo()
    graph = Graph(persist, log, backend)
    keys = list(range(100))
    graph.read_all(keys)
    first = graph.memo.checkpoint()
    assert first["delta"] is False
    assert first["fresh_written"] == 100
    assert first["chain_depth"] == 0

    graph.read(100)
    second = graph.memo.checkpoint()
    assert second["delta"] is True
    assert second["fresh_written"] == 1
    assert second["written"] == 1
    assert second["tombstones"] == 0
    assert second["chain_depth"] == 1
    assert second["compacted"] is False
    assert second["bytes_appended"] > 0
    graph.store.close()

    again = Graph(persist, log, backend)
    assert again.read_all(range(101)) == [key * key for key in range(101)]
    assert again.calls == []


def test_a_five_method_backend_still_checkpoints_in_full(persist):
    factory = DictMemoFactory(persist)
    graph = Graph(persist, persist.MemoryBackend(), factory.open(factory.new_store()))
    graph.read_all(range(10))
    graph.memo.checkpoint()
    graph.read(10)
    stats = graph.memo.checkpoint()
    assert stats["delta"] is False
    assert stats["chain_depth"] == 0
    assert stats["fresh_written"] == 1
    assert graph.memo.stats()["delta_checkpoints"] == 0


def test_every_checkpoint_key_is_present_with_its_type(persist):
    graph = Graph(persist, persist.MemoryBackend(), persist.MemoryMemo())
    graph.read_all(range(5))
    for key in (5, 6):
        stats = graph.memo.checkpoint()
        assert set(stats) == CHECKPOINT_INTS | CHECKPOINT_BOOLS | {"times"}
        for name in CHECKPOINT_INTS:
            assert type(stats[name]) is int, name
        for name in CHECKPOINT_BOOLS:
            assert type(stats[name]) is bool, name
        assert set(stats["times"]) == TIMES
        assert all(type(v) is int and v >= 0 for v in stats["times"].values())
        graph.read(key)


def test_checkpoint_if_due_carries_the_same_keys(persist):
    log = persist.MemoryBackend()
    store = persist.Store.open(log, build_id="b1")
    table = store.memo_table("sq", 1, JsonCodec(), JsonCodec(), lambda ctx, k: k)
    memo = store.open_memo(persist.MemoryMemo(), checkpoint_after_entries=2)
    store.runtime.get(memo.get_or_insert(table, 1))
    assert memo.checkpoint_if_due() is None
    store.runtime.get(memo.get_or_insert(table, 2))
    stats = memo.checkpoint_if_due()
    if stats is None:
        assert persist is not incr.persist, "only Shared's checkpoint thread runs it"
        deadline = time.monotonic() + 60
        while memo.stats()["entries"] == 0:
            assert time.monotonic() < deadline, "the checkpoint thread did not save"
            time.sleep(0.001)
    else:
        assert set(stats) == CHECKPOINT_INTS | CHECKPOINT_BOOLS | {"times"}
    assert memo.stats()["unsaved_entries"] < 2


def test_stats_follow_deltas_and_a_compaction(persist):
    graph = Graph(persist, persist.MemoryBackend(), persist.MemoryMemo())
    graph.read_all(range(10))
    graph.memo.checkpoint()
    idle = graph.memo.stats()
    assert set(idle) >= STATS_INTS
    assert all(type(idle[name]) is int for name in STATS_INTS)
    assert idle["chain_depth"] == 0
    assert idle["delta_checkpoints"] == 0
    assert idle["compactions"] == 0

    graph.read(10)
    delta = graph.memo.checkpoint()
    stats = graph.memo.stats()
    assert stats["chain_depth"] == 1
    assert stats["chain_bytes"] == delta["bytes_appended"] > 0
    assert stats["delta_checkpoints"] == 1
    assert stats["last_bytes_appended"] == delta["bytes_appended"]

    compacted = None
    for key in range(11, 30):
        graph.read(key)
        done = graph.memo.checkpoint()
        if done["compacted"]:
            compacted = done
            break
    assert compacted is not None
    assert compacted["delta"] is False
    assert compacted["chain_depth"] == 0
    after = graph.memo.stats()
    assert after["compactions"] == 1
    assert after["chain_depth"] == 0
    assert after["chain_bytes"] == 0
    assert after["last_bytes_appended"] == compacted["bytes_appended"]


class Counting(DictDeltaMemo):
    """Takes `known`, the ids of the layers the engine already holds, and
    sends only the layers it does not as full tuples."""

    def __init__(self, state, persist, loaded):
        super().__init__(state, persist=persist)
        self.loaded = loaded

    def load_chain(self, known):
        self.record("load_chain")
        with self.state.lock:
            if not self.state.chain:
                return None
            layers = []
            for gen in self.state.chain:
                if gen in known:
                    layers.append((gen, None, None))
                    continue
                self.loaded.append(gen)
                header, items, _ = self.state.generations[gen]
                layers.append((gen, header, list(items)))
            return layers


class Plain(DictDeltaMemo):
    def __init__(self, state, persist, loaded):
        super().__init__(state, persist=persist)
        self.loaded = loaded

    def load_chain(self):
        layers = super().load_chain()
        self.loaded.append(len(layers or []))
        return layers


def test_a_backend_taking_known_loads_the_base_once_across_deltas(persist):
    factory = DictDeltaMemoFactory(persist)
    state = factory.new_store()
    log = persist.MemoryBackend()
    first = Graph(persist, log, Counting(state, persist, []))
    first.read_all(range(50))
    first.memo.checkpoint()
    first.store.close()
    base = state.chain[0]

    loaded = []
    graph = Graph(persist, log, Counting(state, persist, loaded))
    for key in range(50, 55):
        graph.read(key)
        assert graph.memo.checkpoint()["delta"] is True
    assert loaded.count(base) == 1
    assert loaded == [base]
    assert graph.read_all(range(55)) == [key * key for key in range(55)]


def test_a_known_backend_restores_the_same_chain(persist):
    factory = DictDeltaMemoFactory(persist)
    state = factory.new_store()
    log = persist.MemoryBackend()
    graph = Graph(persist, log, Counting(state, persist, []))
    graph.read_all(range(20))
    graph.memo.checkpoint()
    for key in range(20, 23):
        graph.read(key)
        graph.memo.checkpoint()
    graph.store.close()

    again = Graph(persist, log, Counting(state, persist, []))
    assert again.read_all(range(23)) == [key * key for key in range(23)]
    assert again.calls == []


class ViewHeaders(DictDeltaMemo):
    """Returns layer headers as a `memoryview` and a `bytearray`, as a
    driver's binary column type may."""

    def load_chain(self):
        layers = super().load_chain()
        if layers is None:
            return None
        kinds = [memoryview, bytearray]
        return [
            (gen, kinds[i % 2](header), items)
            for i, (gen, header, items) in enumerate(layers)
        ]


def test_a_layer_header_may_be_any_bytes_like_object(persist):
    factory = DictDeltaMemoFactory(persist)
    state = factory.new_store()
    log = persist.MemoryBackend()
    graph = Graph(persist, log, ViewHeaders(state, persist=persist))
    graph.read_all(range(10))
    graph.memo.checkpoint()
    for key in range(10, 13):
        graph.read(key)
        assert graph.memo.checkpoint()["delta"] is True
    graph.store.close()

    again = Graph(persist, log, ViewHeaders(state, persist=persist))
    assert again.read_all(range(13)) == [key * key for key in range(13)]
    assert again.calls == []


def test_a_zero_argument_load_chain_still_works(persist):
    factory = DictDeltaMemoFactory(persist)
    state = factory.new_store()
    loaded = []
    log = persist.MemoryBackend()
    graph = Graph(persist, log, Plain(state, persist, loaded))
    graph.read_all(range(10))
    graph.memo.checkpoint()
    for key in range(10, 13):
        graph.read(key)
        assert graph.memo.checkpoint()["delta"] is True
    assert loaded[-1] == 3
    graph.store.close()

    again = Graph(persist, log, Plain(state, persist, loaded))
    assert again.read_all(range(13)) == [key * key for key in range(13)]
    assert again.calls == []


def test_an_unknown_layer_id_in_a_known_placeholder_is_a_bad_return(persist):
    factory = DictDeltaMemoFactory(persist)
    state = factory.new_store()

    class Lying(Counting):
        def load_chain(self, known):
            layers = super().load_chain(known)
            if layers:
                return [(layers[0][0] + 1000, None, None)] + layers[1:]
            return layers

    graph = Graph(persist, persist.MemoryBackend(), Lying(state, persist, []))
    graph.read_all(range(5))
    graph.memo.checkpoint()
    graph.read(5)
    with pytest.raises(TypeError):
        graph.memo.checkpoint()


def test_memory_memo_sends_a_placeholder_for_each_known_layer(persist):
    backend = persist.MemoryMemo()
    graph = Graph(persist, persist.MemoryBackend(), backend)
    graph.read_all(range(10))
    graph.memo.checkpoint()
    for key in (10, 11):
        graph.read(key)
        graph.memo.checkpoint()

    layers = backend.load_chain()
    ids = [layer[0] for layer in layers]
    assert len(ids) == 3
    assert all(layer[1] is not None and layer[2] is not None for layer in layers)

    partial = backend.load_chain(frozenset(ids[:2]))
    assert partial[:2] == [(ids[0], None, None), (ids[1], None, None)]
    assert partial[2] == layers[2]
    assert backend.load_chain(frozenset()) == layers


def test_memory_memo_takes_known_when_the_store_opens_it(persist):
    backend = persist.MemoryMemo()
    graph = Graph(persist, persist.MemoryBackend(), backend)
    graph.read_all(range(10))
    graph.memo.checkpoint()
    state = graph.memo.stats()
    assert state["chain_depth"] == 0
    held = {layer[0] for layer in backend.load_chain()}
    again = Graph(persist, persist.MemoryBackend(), backend)
    again.read_all(range(10, 12))
    assert again.memo.checkpoint()["delta"] is True
    assert held <= {layer[0] for layer in backend.load_chain()}
