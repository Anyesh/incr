"""R15: `ctx.keys(kc)` reads the key set of a keyed collection inside a
query, with a dependency on membership: the query reruns when a write
inserts or removes a key and not when it only changes a row."""

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


class Lister:
    def __init__(self, rt, kc):
        self.runs = 0
        self.rt = rt

        def read(ctx):
            self.runs += 1
            return sorted(ctx.keys(kc))

        self.node = rt.create_query(read)

    def value(self):
        return self.rt.get(self.node)


def declare(store):
    return store.keyed_collection("shifts", PickleCodec(), PickleCodec())


def test_a_lister_reruns_on_insert_and_remove_not_on_a_row_update(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a", 1)
    lister = Lister(rt, kc)
    assert lister.value() == ["a"]
    kc.upsert("a", 2)
    assert lister.value() == ["a"]
    assert lister.runs == 1
    kc.upsert("b", 3)
    assert lister.value() == ["a", "b"]
    assert lister.runs == 2
    kc.remove("b")
    assert lister.value() == ["a"]
    assert lister.runs == 3
    kc.reconcile({"a": 5, "c": 6})
    assert lister.value() == ["a", "c"]
    assert lister.runs == 4
    kc.reconcile({"a": 7, "c": 8})
    assert lister.value() == ["a", "c"]
    assert lister.runs == 4


def test_a_persisted_lister_reruns_on_insert_and_remove_not_on_a_row_update(persist):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    kc = declare(store)
    store.upsert(kc, "a", 1)
    lister = Lister(store.runtime, kc)
    assert lister.value() == ["a"]
    store.upsert(kc, "a", 2)
    assert lister.value() == ["a"]
    assert lister.runs == 1
    store.upsert(kc, "b", 3)
    assert lister.value() == ["a", "b"]
    assert lister.runs == 2
    store.remove(kc, "b")
    assert lister.value() == ["a"]
    assert lister.runs == 3


def test_a_key_reader_and_a_lister_do_not_wake_each_other(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a", 1)
    runs = []
    by_key = rt.create_query(lambda ctx: runs.append(1) or ctx.get_key(kc, "a"))
    lister = Lister(rt, kc)
    rt.get(by_key)
    lister.value()
    kc.upsert("b", 2)
    rt.get(by_key)
    assert lister.value() == ["a", "b"]
    assert (len(runs), lister.runs) == (1, 2)
    kc.upsert("a", 9)
    rt.get(by_key)
    assert lister.value() == ["a", "b"]
    assert (len(runs), lister.runs) == (2, 2)


def test_ctx_keys_refuses_another_runtimes_collection(pkg):
    other = pkg.Runtime()
    kc = pkg.KeyedCollection(other)
    rt = pkg.Runtime()
    node = rt.create_query(lambda ctx: ctx.keys(kc))
    with pytest.raises(ValueError):
        rt.get(node)


def test_a_persisted_lister_is_correct_after_reopen(persist):
    backend = persist.MemoryBackend()
    store = persist.Store.open(backend, build_id=1)
    kc = declare(store)
    store.upsert(kc, "a", 1)
    store.upsert(kc, "b", 2)
    assert Lister(store.runtime, kc).value() == ["a", "b"]
    del store
    store = persist.Store.open(backend, build_id=1)
    kc = declare(store)
    lister = Lister(store.runtime, kc)
    assert lister.value() == ["a", "b"]
    store.upsert(kc, "c", 3)
    assert lister.value() == ["a", "b", "c"]
    assert lister.runs == 2
