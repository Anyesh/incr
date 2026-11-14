"""R12: keyed collection rows need not be hashable. A keyed write hashes
only keys and detects an unchanged row with `==`; a persisted collection
names a logged pair by the fingerprint of its bytes, so a row that is not
`==` to its own decoded copy (NaN, identity-compared) still replaces,
replays and converges to one pair."""

import importlib
import pickle

import incr
import incr.persist
import incr_concurrent.persist
import pytest

PACKAGES = [
    pytest.param(incr.persist, id="local"),
    pytest.param(incr_concurrent.persist, id="shared"),
]


@pytest.fixture(params=PACKAGES)
def persist(request):
    return request.param


@pytest.fixture
def pkg(persist):
    return importlib.import_module(persist.__name__.rpartition(".")[0])


# Decodes only bytes this test's own store wrote.
class PickleCodec:
    def __init__(self):
        self.encodes = 0

    def encode(self, obj):
        self.encodes += 1
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class ReorderingCodec(PickleCodec):
    """Encodes a dict's items in a rotating order, so a re-encode of an
    equal row gives other bytes, as a set under a new hash seed may."""

    def encode(self, obj):
        self.encodes += 1
        items = list(obj.items())
        turn = self.encodes % len(items)
        return pickle.dumps(items[turn:] + items[:turn], protocol=4)

    def decode(self, data):
        return dict(pickle.loads(data))


class RecordingBackend:
    def __init__(self, inner):
        self.inner = inner
        self.appends = 0

    def append(self, records):
        self.appends += 1
        return self.inner.append(records)

    def sync(self, upto):
        return self.inner.sync(upto)

    def read_from(self, after):
        return self.inner.read_from(after)

    def load_snapshot(self):
        return self.inner.load_snapshot()

    def snapshot_id(self):
        return self.inner.snapshot_id()

    def compact(self, upto, folded):
        return self.inner.compact(upto, folded)

    def writable(self):
        return self.inner.writable()


class IdentityCompared:
    def __init__(self, n):
        self.n = n


class EqRaisesValueError:
    def __init__(self, n):
        self.n = n

    def __eq__(self, other):
        raise ValueError("the truth value of an array is ambiguous")

    __hash__ = None


class Graph:
    """A store over `backend` with the keyed collection `rows`, whose row
    codec counts its encodes."""

    def __init__(self, persist, backend, codec=None):
        self.backend = backend
        self.codec = codec or PickleCodec()
        self.store = persist.Store.open(backend, build_id=1)
        self.rt = self.store.runtime
        self.kc = self.store.keyed_collection("rows", PickleCodec(), self.codec)

    def reopen(self, persist):
        self.store.close()
        return Graph(persist, self.backend, self.codec)

    def pairs_held(self):
        # A checkpoint encodes every held pair's row once.
        self.codec.encodes = 0
        self.store.checkpoint()
        return self.codec.encodes


def reader(rt, kc, key):
    runs = []

    def read(ctx):
        runs.append(key)
        return ctx.get_key(kc, key)

    return rt.create_query(read), runs


ROWS = {"d": {"a": 1, "b": [1, 2]}, "l": [{"x": 1}, [2, 3]]}


def assert_rows(kc, rows):
    assert len(kc) == len(rows)
    for key, row in rows.items():
        assert kc.get(key) == row
    assert sorted(kc.items(), key=lambda pair: pair[0]) == sorted(rows.items())


def write_in(batch, kc, rows):
    with batch as b:
        for key, row in rows.items():
            b.upsert(kc, key, row)


WRITES = {
    "upsert": lambda g, rows: [g.store.upsert(g.kc, k, v) for k, v in rows.items()],
    "reconcile": lambda g, rows: g.store.reconcile(g.kc, rows),
    "batch": lambda g, rows: write_in(g.store.batch(), g.kc, rows),
    "on_write": lambda g, rows: write_in(g.store.batch(apply="on_write"), g.kc, rows),
}


@pytest.mark.parametrize("write", WRITES)
def test_unhashable_rows_persist_and_reopen_equal(persist, write):
    g = Graph(persist, persist.MemoryBackend())
    WRITES[write](g, ROWS)
    assert_rows(g.kc, ROWS)

    g = g.reopen(persist)
    assert_rows(g.kc, ROWS)
    assert g.pairs_held() == 2

    g = g.reopen(persist)
    assert_rows(g.kc, ROWS)


def test_an_equal_dict_reruns_and_appends_nothing_and_a_changed_one_reruns_once(
    persist,
):
    g = Graph(persist, RecordingBackend(persist.MemoryBackend()))
    g.store.upsert(g.kc, "a", {"v": 1})
    query, runs = reader(g.rt, g.kc, "a")
    assert g.rt.get(query) == {"v": 1}
    appends = g.backend.appends

    g.store.upsert(g.kc, "a", {"v": 1})
    assert g.rt.get(query) == {"v": 1}
    assert (runs, g.backend.appends) == (["a"], appends)

    g.store.upsert(g.kc, "a", {"v": 2})
    assert g.rt.get(query) == {"v": 2}
    assert runs == ["a", "a"]


def test_a_memo_entry_over_a_dict_row_verifies_after_reopen(persist):
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    calls = []

    def open_graph():
        store = persist.Store.open(log, build_id="b1")
        kc = store.keyed_collection("rows", PickleCodec(), PickleCodec())

        def price(ctx, key):
            calls.append(key)
            return ctx.get_key(kc, key)["v"]

        table = store.memo_table("price", 1, PickleCodec(), PickleCodec(), price)
        return store, kc, table, store.open_memo(memo_backend)

    store, kc, table, memo = open_graph()
    store.upsert(kc, "a", {"v": 1})
    store.upsert(kc, "b", {"v": 2})
    assert store.runtime.get(memo.get_or_insert(table, "a")) == 1
    memo.checkpoint()
    store.upsert(kc, "b", {"v": 3})
    store.close()

    store, kc, table, memo = open_graph()
    calls.clear()
    assert store.runtime.get(memo.get_or_insert(table, "a")) == 1
    assert calls == []
    store.upsert(kc, "a", {"v": 5})
    assert store.runtime.get(memo.get_or_insert(table, "a")) == 5
    assert calls == ["a"]


def test_a_second_store_catches_up_dict_rows(persist):
    backend = persist.MemoryBackend()
    leader = Graph(persist, backend)
    standby = Graph(persist, backend)
    leader.store.upsert(leader.kc, "a", {"v": 1})
    leader.store.upsert(leader.kc, "b", {"v": [2]})
    standby.store.refresh()
    assert_rows(standby.kc, {"a": {"v": 1}, "b": {"v": [2]}})
    query, runs = reader(standby.rt, standby.kc, "a")
    standby.rt.get(query)

    leader.store.upsert(leader.kc, "b", {"v": [3]})
    leader.store.checkpoint()
    standby.store.refresh()
    assert_rows(standby.kc, {"a": {"v": 1}, "b": {"v": [3]}})
    assert standby.rt.get(query) == {"v": 1}
    assert runs == ["a"]

    standby.store.upsert(standby.kc, "a", {"v": 9})
    leader.store.refresh()
    assert leader.kc.get("a") == {"v": 9}


def test_overlay_upsert_of_dict_rows_is_undone():
    g = Graph(incr.persist, incr.persist.MemoryBackend())
    g.store.upsert(g.kc, "a", {"v": 1})
    plain = incr.KeyedCollection(g.rt)
    plain.upsert("a", {"v": 1})
    with g.rt.overlay():
        for kc in (g.kc, plain):
            kc.upsert("a", {"v": 2})
            kc.upsert("b", [1])
            assert (kc.get("a"), kc.get("b")) == ({"v": 2}, [1])
    for kc in (g.kc, plain):
        assert_rows(kc, {"a": {"v": 1}})
    assert g.pairs_held() == 1


def test_a_plain_collection_holds_unhashable_rows_and_its_rows_collection_hashes(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    for key, row in ROWS.items():
        kc.upsert(key, row)
    query, runs = reader(rt, kc, "d")
    assert rt.get(query) == ROWS["d"]
    kc.upsert("d", {"a": 1, "b": [1, 2]})
    assert rt.get(query) == ROWS["d"]
    assert runs == ["d"]
    assert kc.reconcile({"d": {"a": 2}, "l": ROWS["l"]}) == (0, 1, 0)
    assert rt.get(query) == {"a": 2}
    assert_rows(kc, {"d": {"a": 2}, "l": ROWS["l"]})
    with pytest.raises(TypeError, match="unhashable"):
        rt.get(kc.collection.count())


def nan_row(v):
    return {"x": float("nan"), "v": v}


def identity_row(v):
    return {"o": IdentityCompared(v), "v": v}


ROW_MAKERS = [
    pytest.param(nan_row, id="nan"),
    pytest.param(identity_row, id="identity"),
]


@pytest.mark.parametrize("make", ROW_MAKERS)
def test_a_row_unequal_to_its_decoded_copy_replaces_to_one_pair(persist, make):
    g = Graph(persist, persist.MemoryBackend())
    g.store.upsert(g.kc, "k", make(1))
    g.store.upsert(g.kc, "k", make(2))
    assert g.kc.get("k")["v"] == 2

    g = g.reopen(persist)
    assert g.kc.get("k")["v"] == 2
    assert g.pairs_held() == 1

    g.store.upsert(g.kc, "k", make(3))
    g = g.reopen(persist)
    assert g.kc.get("k")["v"] == 3
    assert g.pairs_held() == 1


@pytest.mark.parametrize("make", ROW_MAKERS)
def test_a_restored_row_unequal_to_its_decoded_copy_reruns_no_reader(persist, make):
    backend = persist.MemoryBackend()
    leader = Graph(persist, backend)
    standby = Graph(persist, backend)
    leader.store.upsert(leader.kc, "a", make(1))
    leader.store.upsert(leader.kc, "b", make(2))
    standby.store.refresh()
    query, runs = reader(standby.rt, standby.kc, "a")
    standby.rt.get(query)

    leader.store.upsert(leader.kc, "b", make(3))
    leader.store.checkpoint()
    standby.store.refresh()
    assert standby.rt.get(query)["v"] == 1
    assert standby.kc.get("b")["v"] == 3
    assert runs == ["a"]


def test_a_raising_row_eq_under_two_store_contention_does_not_poison(persist):
    backend = persist.MemoryBackend()
    first = Graph(persist, backend)
    second = Graph(persist, backend)
    first.store.upsert(first.kc, "k", EqRaisesValueError(0))
    second.store.refresh()

    with second.store.batch() as b:
        b.upsert(second.kc, "k", EqRaisesValueError(2))
        first.store.upsert(first.kc, "k", EqRaisesValueError(1))
    assert second.kc.get("k").n == 2
    assert second.pairs_held() == 1

    first.store.refresh()
    assert first.kc.get("k").n == 2
    second.store.upsert(second.kc, "k", EqRaisesValueError(3))
    second = second.reopen(persist)
    assert second.kc.get("k").n == 3
    assert second.pairs_held() == 1


def test_a_row_whose_re_encode_differs_replaces_to_one_pair(persist):
    g = Graph(persist, persist.MemoryBackend(), ReorderingCodec())
    g.store.upsert(g.kc, "k", {"a": 1, "b": 2, "c": 3})
    g.store.upsert(g.kc, "k", {"a": 1, "b": 2, "c": 4})
    g = g.reopen(persist)
    assert g.kc.get("k") == {"a": 1, "b": 2, "c": 4}
    assert g.pairs_held() == 1
