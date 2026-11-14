import importlib
import pickle

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


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class Boom(Exception):
    pass


class Key:
    def __init__(self, name, hash_raises=False, eq_raises=False):
        self.name = name
        self.hash_raises = hash_raises
        self.eq_raises = eq_raises

    def __hash__(self):
        if self.hash_raises:
            raise Boom(f"hash {self.name}")
        return hash(self.name)

    def __eq__(self, other):
        if self.eq_raises or getattr(other, "eq_raises", False):
            raise Boom(f"eq {self.name}")
        return isinstance(other, Key) and self.name == other.name

    def __repr__(self):
        return f"Key({self.name!r})"


class Row:
    hashes = 0
    eqs = 0

    def __init__(self, value, eq_raises=False):
        self.value = value
        self.eq_raises = eq_raises

    def __hash__(self):
        Row.hashes += 1
        return hash(self.value)

    def __eq__(self, other):
        Row.eqs += 1
        if self.eq_raises or getattr(other, "eq_raises", False):
            raise Boom(f"eq row {self.value}")
        return isinstance(other, Row) and self.value == other.value


def value_of(row):
    return row.value if isinstance(row, Row) else row


def readers(rt, kc):
    rows = kc.collection
    nodes = {
        "values": rows.reduce(lambda xs: sorted(value_of(x) for x in xs)),
        "count": rows.count(),
        "sum": rows.aggregate(0, value_of, lambda a, b: a + b),
    }
    for node in nodes.values():
        rt.get(node)
    return nodes


def read(rt, nodes):
    return {name: rt.get(node) for name, node in nodes.items()}


def state(rt, kc, nodes, keys):
    return (
        len(kc),
        {k: value_of(kc.get(k)) for k in keys},
        {k: k in kc for k in keys},
        read(rt, nodes),
    )


def test_keyed_collection_is_registered_and_its_rows_are_a_collection(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    assert isinstance(kc.collection, pkg.Collection)
    assert len(kc) == 0


def test_upsert_of_an_equal_row_recomputes_nothing_downstream(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a", 1)
    kc.upsert("b", 2)

    lifts = []

    def lift(x):
        lifts.append(x)
        return x

    total = kc.collection.aggregate(0, lift, lambda a, b: a + b)
    count = kc.collection.count()
    version = kc.collection.version_node
    reads = []

    def reader(ctx):
        reads.append(1)
        return (ctx.get(version), ctx.get(count), ctx.get(total))

    q = rt.create_query(reader)
    assert rt.get(q)[1:] == (2, 3)
    lifts.clear()
    reads.clear()

    kc.upsert("a", 1)
    kc.upsert("b", 2.0)

    assert rt.get(q)[1:] == (2, 3)
    assert reads == []
    assert lifts == []
    _, trace = rt.get_traced(count)
    assert trace["nodes_recomputed"] == 0

    kc.upsert("a", 4)
    assert rt.get(q)[1:] == (2, 6)
    assert reads == [1]


def test_upsert_of_a_changed_row_replaces_it(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a", 1)
    kc.upsert("b", 2)
    nodes = readers(rt, kc)
    assert read(rt, nodes) == {"values": [1, 2], "count": 2, "sum": 3}

    kc.upsert("a", 5)

    assert len(kc) == 2
    assert kc.get("a") == 5
    assert read(rt, nodes) == {"values": [2, 5], "count": 2, "sum": 7}


def test_get_and_remove_of_an_absent_key(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a", 1)
    nodes = readers(rt, kc)

    assert kc.get("missing") is None
    assert "missing" not in kc
    assert "a" in kc
    with pytest.raises(KeyError):
        kc.remove("missing")
    assert len(kc) == 1

    kc.remove("a")
    assert len(kc) == 0
    assert kc.get("a") is None
    assert read(rt, nodes) == {"values": [], "count": 0, "sum": 0}
    with pytest.raises(KeyError):
        kc.remove("a")


def test_keys_follow_python_equality_like_a_dict(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert(1, "one")
    kc.upsert(1.0, "uno")
    assert len(kc) == 1
    assert kc.get(1) == "uno"
    assert True in kc


def build(pkg, rt, snapshot):
    kc = pkg.KeyedCollection(rt)
    for key, row in snapshot.items():
        kc.upsert(key, row)
    return kc


def test_reconcile_from_a_fresh_snapshot_equals_building_from_scratch(pkg):
    rt = pkg.Runtime()
    kc = build(pkg, rt, {"a": 1, "b": 2, "c": 3, "e": 5})
    nodes = readers(rt, kc)
    version = kc.collection.version_node

    snapshot = {"a": 1, "b": 20, "d": 4, "e": 5, "f": 6}
    assert kc.reconcile(snapshot) == (2, 1, 1)

    fresh = build(pkg, rt, snapshot)
    fresh_nodes = readers(rt, fresh)
    assert read(rt, nodes) == read(rt, fresh_nodes)
    assert len(kc) == len(fresh) == 5
    assert {k: kc.get(k) for k in snapshot} == snapshot
    assert "c" not in kc

    before = rt.get(version)
    assert kc.reconcile(dict(snapshot)) == (0, 0, 0)
    assert rt.get(version) == before

    assert kc.reconcile([("a", 1), ("b", 7), ("a", 9)]) == (0, 2, 3)
    assert {k: kc.get(k) for k in "abdef"} == {
        "a": 9,
        "b": 7,
        "d": None,
        "e": None,
        "f": None,
    }
    assert read(rt, nodes) == read(rt, readers(rt, build(pkg, rt, {"a": 9, "b": 7})))

    assert kc.reconcile({}) == (0, 0, 2)
    assert len(kc) == 0
    assert read(rt, nodes) == {"values": [], "count": 0, "sum": 0}


def test_group_by_and_aggregate_over_rows_track_upserts_and_removes(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    groups = kc.collection.group_by(lambda row: row[0])

    def hours_by_worker():
        rt.get(groups.version_node)
        out = {}
        for worker in groups.keys():
            group = groups.get_group(worker)
            total = rt.get(group.aggregate(0, lambda r: r[1], lambda a, b: a + b))
            if rt.get(group.count()):
                out[worker] = total
        return out

    total = kc.collection.aggregate(0, lambda r: r[1], lambda a, b: a + b)
    kc.reconcile({"s1": ("w1", 8), "s2": ("w1", 4), "s3": ("w2", 6)})
    assert hours_by_worker() == {"w1": 12, "w2": 6}
    assert rt.get(total) == 18

    kc.upsert("s2", ("w2", 5))
    assert hours_by_worker() == {"w1": 8, "w2": 11}
    assert rt.get(total) == 19

    kc.upsert("s4", ("w3", 1))
    kc.remove("s1")
    assert hours_by_worker() == {"w2": 11, "w3": 1}
    assert rt.get(total) == 12

    kc.reconcile({"s3": ("w2", 6)})
    assert hours_by_worker() == {"w2": 6}
    assert rt.get(total) == 6


def seeded(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.reconcile({Key("a"): Row(1), Key("b"): Row(2), Key("x"): Row(3)})
    return rt, kc, readers(rt, kc)


KEYS = [Key("a"), Key("b"), Key("x"), Key("new")]


def test_reconcile_whose_row_eq_raises_leaves_everything_unchanged(pkg):
    rt, kc, nodes = seeded(pkg)
    before = state(rt, kc, nodes, KEYS)

    with pytest.raises(Boom):
        kc.reconcile(
            [
                (Key("new"), Row(9)),
                (Key("b"), Row(20)),
                (Key("a"), Row(1, eq_raises=True)),
            ]
        )

    assert state(rt, kc, nodes, KEYS) == before
    assert kc.reconcile({Key("a"): Row(5), Key("x"): Row(3)}) == (0, 1, 1)
    assert read(rt, nodes) == {"values": [3, 5], "count": 2, "sum": 8}


RAISING_KEYS = {
    "hash": lambda: Key("x", hash_raises=True),
    "eq": lambda: Key("x", eq_raises=True),
}

OPERATIONS = {
    "upsert": lambda kc, key: kc.upsert(key, Row(7)),
    "remove": lambda kc, key: kc.remove(key),
    "get": lambda kc, key: kc.get(key),
    "contains": lambda kc, key: key in kc,
    "reconcile": lambda kc, key: kc.reconcile(
        [(Key("new"), Row(9)), (Key("b"), Row(20)), (key, Row(7))]
    ),
}


@pytest.mark.parametrize("op", OPERATIONS)
@pytest.mark.parametrize("raising", RAISING_KEYS)
def test_a_key_whose_hash_or_eq_raises_leaves_everything_unchanged(pkg, raising, op):
    rt, kc, nodes = seeded(pkg)
    before = state(rt, kc, nodes, KEYS)

    with pytest.raises(Boom):
        OPERATIONS[op](kc, RAISING_KEYS[raising]())

    assert state(rt, kc, nodes, KEYS) == before
    kc.upsert(Key("x"), Row(30))
    assert kc.get(Key("x")).value == 30
    assert read(rt, nodes) == {"values": [1, 2, 30], "count": 3, "sum": 33}


@pytest.mark.parametrize("read_rows", [True, False], ids=["rows-read", "rows-unread"])
def test_row_hash_calls_per_upsert_meet_the_documented_bound(pkg, read_rows):
    # Pairs hash their key alone, so only `kc.collection`'s output hashes
    # rows, once per added or removed row, and only once it is read.
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    count = kc.collection.count() if read_rows else None
    kc.upsert("a", Row(1))
    if count is not None:
        rt.get(count)

    def calls(change):
        Row.hashes = Row.eqs = 0
        change()
        if count is not None:
            rt.get(count)
        return Row.hashes, Row.eqs

    per_row = int(read_rows)
    assert calls(lambda: kc.upsert("b", Row(2))) == (per_row, 0)
    assert calls(lambda: kc.upsert("b", Row(3))) == (2 * per_row, 4 + per_row)
    assert calls(lambda: kc.upsert("b", Row(3))) == (0, 1)
    assert calls(lambda: kc.remove("b")) == (per_row, per_row)


WRITES = {
    "upsert": lambda kc: kc.upsert("a", 99),
    "upsert-new": lambda kc: kc.upsert("z", 1),
    "remove": lambda kc: kc.remove("a"),
    "reconcile": lambda kc: kc.reconcile({"a": 5, "q": 6}),
}


@pytest.mark.parametrize("op", WRITES)
def test_a_keyed_write_from_inside_a_query_raises_and_leaves_the_rows(pkg, op):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.reconcile({"a": 1, "b": 2})
    q = rt.create_query(lambda ctx: WRITES[op](kc))
    with pytest.raises(RuntimeError, match="written from inside a compute"):
        rt.get(q)
    assert sorted(kc.items()) == [("a", 1), ("b", 2)]
    kc.upsert("c", 3)
    assert len(kc) == 3


PERSISTED_WRITES = {
    "upsert": lambda store, kc: store.upsert(kc, "a", 99),
    "remove": lambda store, kc: store.remove(kc, "a"),
    "reconcile": lambda store, kc: store.reconcile(kc, {"a": 5, "q": 6}),
}


@pytest.mark.parametrize("op", PERSISTED_WRITES)
def test_a_store_keyed_write_from_inside_a_query_raises_and_leaves_the_rows(
    persist, op
):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    codec = PickleCodec()
    kc = store.keyed_collection("rows", codec, codec)
    store.reconcile(kc, {"a": 1, "b": 2})
    position = store.stats()["position"]
    q = store.runtime.create_query(lambda ctx: PERSISTED_WRITES[op](store, kc))
    with pytest.raises(RuntimeError):
        store.runtime.get(q)
    assert sorted(kc.items()) == [("a", 1), ("b", 2)]
    assert store.stats()["position"] == position
