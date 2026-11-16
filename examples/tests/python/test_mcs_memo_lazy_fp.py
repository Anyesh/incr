"""R20: opening a memo encodes no element of a store collection, and a
memo entry reading keyed rows through `ctx.get_key` is checked without
encoding the whole collection. Counting codecs show which encodes ran."""

import importlib
import json
import threading

import incr.persist
import incr_concurrent.persist
import pytest

PACKAGES = [
    pytest.param(incr.persist, id="local"),
    pytest.param(incr_concurrent.persist, id="shared"),
]

ROWS = 200


@pytest.fixture(params=PACKAGES)
def persist(request):
    return request.param


class CountingCodec:
    def __init__(self):
        self.encoded = 0

    def encode(self, obj):
        self.encoded += 1
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)


class Graph:
    """A store with the plain collection `xs`, the keyed collection `rows`
    and the memo table `price` over `rows`, recording the keys it ran for."""

    def __init__(self, persist, log, memo_backend):
        self.store = persist.Store.open(log, build_id="b1")
        self.xs_codec = CountingCodec()
        self.row_codec = CountingCodec()
        self.xs = self.store.collection("xs", self.xs_codec)
        self.rows = self.store.keyed_collection("rows", CountingCodec(), self.row_codec)
        self.calls = []

        def price(ctx, key):
            self.calls.append(key)
            row = ctx.get_key(self.rows, key)
            return None if row is None else row[0]

        self.price = self.store.memo_table(
            "price", 1, CountingCodec(), CountingCodec(), price
        )

    def open_memo(self, memo_backend):
        self.memo = self.store.open_memo(memo_backend)

    def read(self, key):
        return self.store.runtime.get(self.memo.get_or_insert(self.price, key))

    def encoded(self):
        return self.xs_codec.encoded + self.row_codec.encoded


@pytest.fixture
def backends(persist):
    return persist.MemoryBackend(), persist.MemoryMemo()


def populated(persist, backends):
    log, memo_backend = backends
    graph = Graph(persist, log, memo_backend)
    graph.open_memo(memo_backend)
    for k in range(ROWS):
        graph.store.insert(graph.xs, k)
        graph.store.upsert(graph.rows, k, [k * 10])
    for k in range(4):
        graph.read(k)
    graph.memo.checkpoint()
    graph.store.close()


def test_opening_a_memo_encodes_no_collection_element(persist, backends):
    populated(persist, backends)
    log, memo_backend = backends
    graph = Graph(persist, log, memo_backend)
    graph.xs_codec.encoded = graph.row_codec.encoded = 0
    graph.open_memo(memo_backend)
    assert graph.encoded() == 0


def test_a_get_key_check_encodes_no_collection_and_a_changed_row_misses(
    persist, backends
):
    populated(persist, backends)
    log, memo_backend = backends
    graph = Graph(persist, log, memo_backend)
    graph.store.upsert(graph.rows, 2, [999])
    graph.open_memo(memo_backend)
    graph.xs_codec.encoded = graph.row_codec.encoded = 0
    assert [graph.read(k) for k in range(4)] == [0, 10, 999, 30]
    assert graph.encoded() == 0
    assert graph.calls == [2]


def test_a_memo_check_racing_mapped_row_writes_finishes():
    persist = incr_concurrent.persist
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    populated(persist, (log, memo_backend))
    for round in range(8):
        graph = Graph(persist, log, memo_backend)
        graph.open_memo(memo_backend)
        start = threading.Barrier(2)
        failures = []

        def check():
            try:
                start.wait()
                for k in range(4):
                    graph.read(k)
            except BaseException as e:
                failures.append(e)

        def write():
            try:
                start.wait()
                for k in range(4):
                    graph.store.upsert(graph.rows, k, [k * 10 + round + 1])
            except BaseException as e:
                failures.append(e)

        threads = [threading.Thread(target=check), threading.Thread(target=write)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not any(t.is_alive() for t in threads), "a thread hung"
        assert failures == []
        assert [graph.read(k) for k in range(4)] == [
            k * 10 + round + 1 for k in range(4)
        ]
        graph.memo.checkpoint()
        graph.store.close()
