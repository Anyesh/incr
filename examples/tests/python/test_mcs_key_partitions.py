"""R15b: `kc.key_partitions(fn)` partitions a keyed collection's keys and
`ctx.keys(parts, p)` reads one partition inside a query, rerunning only when
a write inserts or removes a key of that partition."""

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


def first_letter(key):
    return key[0]


def declare(store):
    return store.keyed_collection("shifts", PickleCodec(), PickleCodec())


class Reader:
    def __init__(self, rt, parts, p):
        self.runs = 0
        self.rt = rt

        def read(ctx):
            self.runs += 1
            return sorted(ctx.keys(parts, p))

        self.node = rt.create_query(read)

    def value(self):
        return self.rt.get(self.node)


def test_a_partition_reader_reruns_only_for_its_own_partition(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a1", 1)
    kc.upsert("b1", 1)
    parts = kc.key_partitions(first_letter)
    reader = Reader(rt, parts, "a")
    assert reader.value() == ["a1"]
    kc.upsert("a1", 2)
    kc.upsert("b2", 1)
    kc.remove("b1")
    assert reader.value() == ["a1"]
    assert reader.runs == 1
    kc.upsert("a2", 1)
    assert reader.value() == ["a1", "a2"]
    assert reader.runs == 2
    kc.remove("a1")
    assert reader.value() == ["a2"]
    assert reader.runs == 3
    kc.reconcile({"a2": 5, "b3": 1, "a9": 1})
    assert reader.value() == ["a2", "a9"]
    assert reader.runs == 4
    kc.reconcile({"a2": 6, "b3": 2, "a9": 2})
    assert reader.value() == ["a2", "a9"]
    assert reader.runs == 4


def test_a_reader_of_a_partition_with_no_keys_wakes_when_one_arrives(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    parts = kc.key_partitions(first_letter)
    reader = Reader(rt, parts, "z")
    assert reader.value() == []
    kc.upsert("a1", 1)
    assert reader.value() == []
    assert reader.runs == 1
    kc.upsert("z1", 1)
    assert reader.value() == ["z1"]
    assert reader.runs == 2


def test_the_partition_may_be_any_hashable_value(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert((1, "x"), 1)
    kc.upsert((2, "x"), 1)
    kc.upsert((3, "y"), 1)
    parts = kc.key_partitions(lambda key: (key[1], None))
    reader = Reader(rt, parts, ("x", None))
    assert reader.value() == [(1, "x"), (2, "x")]


def test_a_partition_reader_agrees_with_a_key_reader_on_one_partition(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    parts = kc.key_partitions(first_letter)
    all_keys = rt.create_query(lambda ctx: sorted(ctx.keys(kc)))
    in_a = Reader(rt, parts, "a")
    for key in ["a1", "b1", "a2", "c1"]:
        kc.upsert(key, 0)
    kc.remove("a1")
    assert rt.get(all_keys) == ["a2", "b1", "c1"]
    assert in_a.value() == ["a2"]


def test_a_raising_partition_fn_refuses_the_write_and_changes_nothing(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a1", 1)

    def fn(key):
        if key == "bad":
            raise Boom("no")
        return key[0]

    parts = kc.key_partitions(fn)
    reader = Reader(rt, parts, "b")
    assert reader.value() == []
    with pytest.raises(Boom):
        kc.upsert("bad", 1)
    assert "bad" not in kc
    assert len(kc) == 1
    assert reader.value() == []
    assert reader.runs == 1
    kc.upsert("b1", 1)
    assert reader.value() == ["b1"]


def test_a_raising_partition_fn_refuses_a_reconcile_whole(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a1", 1)

    def fn(key):
        if key == "bad":
            raise Boom("no")
        return key[0]

    kc.key_partitions(fn)
    with pytest.raises(Boom):
        kc.reconcile({"a1": 2, "b1": 1, "bad": 1})
    assert len(kc) == 1
    assert kc.get("a1") == 1


def test_a_raising_fn_at_registration_leaves_the_collection_usable(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("bad", 1)

    def fn(key):
        raise Boom("no")

    with pytest.raises(Boom):
        kc.key_partitions(fn)
    kc.upsert("a1", 1)
    assert len(kc) == 2


def test_an_unhashable_partition_is_a_type_error(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a1", 1)
    with pytest.raises(TypeError):
        kc.key_partitions(lambda key: [key])


def test_key_partitions_of_another_runtime_is_refused(pkg):
    kc = pkg.KeyedCollection(pkg.Runtime())
    parts = kc.key_partitions(first_letter)
    rt = pkg.Runtime()
    node = rt.create_query(lambda ctx: ctx.keys(parts, "a"))
    with pytest.raises(ValueError):
        rt.get(node)


def test_ctx_keys_takes_a_collection_or_partitions_and_one_partition(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    parts = kc.key_partitions(first_letter)
    for args in [(parts,), (kc, "a"), (parts, "a", "b"), ("a", "a")]:
        node = rt.create_query(lambda ctx, args=args: ctx.keys(*args))
        with pytest.raises(TypeError):
            rt.get(node)


def test_a_persisted_partition_reader_reruns_only_for_its_own_partition(persist):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    kc = declare(store)
    store.upsert(kc, "a1", 1)
    parts = kc.key_partitions(first_letter)
    reader = Reader(store.runtime, parts, "a")
    assert reader.value() == ["a1"]
    store.upsert(kc, "a1", 2)
    store.upsert(kc, "b1", 1)
    assert reader.value() == ["a1"]
    assert reader.runs == 1
    store.upsert(kc, "a2", 1)
    assert reader.value() == ["a1", "a2"]
    store.remove(kc, "a1")
    assert reader.value() == ["a2"]
    assert reader.runs == 3


def test_a_raising_fn_refuses_a_persisted_write_and_the_store_stays_usable(persist):
    backend = persist.MemoryBackend()
    store = persist.Store.open(backend, build_id=1)
    kc = declare(store)
    store.upsert(kc, "a1", 1)

    def fn(key):
        if key == "bad":
            raise Boom("no")
        return key[0]

    parts = kc.key_partitions(fn)
    with pytest.raises(Boom):
        store.upsert(kc, "bad", 1)
    assert "bad" not in kc
    store.upsert(kc, "b1", 1)
    assert Reader(store.runtime, parts, "b").value() == ["b1"]
    del store
    store = persist.Store.open(backend, build_id=1)
    kc = declare(store)
    assert sorted(Reader(store.runtime, kc.key_partitions(first_letter), "a").value()) == [
        "a1"
    ]
    assert "bad" not in kc


def test_registering_on_a_reopened_persisted_collection_builds_from_its_keys(persist):
    backend = persist.MemoryBackend()
    store = persist.Store.open(backend, build_id=1)
    kc = declare(store)
    for key in ["a1", "a2", "b1"]:
        store.upsert(kc, key, 1)
    del store
    store = persist.Store.open(backend, build_id=1)
    kc = declare(store)
    parts = kc.key_partitions(first_letter)
    reader = Reader(store.runtime, parts, "a")
    assert reader.value() == ["a1", "a2"]
    store.upsert(kc, "b2", 1)
    assert reader.value() == ["a1", "a2"]
    assert reader.runs == 1
    store.upsert(kc, "a3", 1)
    assert reader.value() == ["a1", "a2", "a3"]
    assert reader.runs == 2
