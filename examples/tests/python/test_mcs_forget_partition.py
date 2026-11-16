"""R15b-fix3: `kc.forget_partition(parts, p)` frees the input of a partition
that holds no key once no query depends on it. Shared never frees one, so it
returns False there."""

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


@pytest.fixture
def local(persist):
    if persist is not incr.persist:
        pytest.skip("Local only")


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


def first_letter(key):
    return key[0]


def reader(rt, parts, p):
    return rt.create_query(lambda ctx: sorted(ctx.keys(parts, p)))


def freed(pkg):
    return pkg is incr


def test_an_emptied_partition_is_freed_once_its_reader_is_gone(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    parts = kc.key_partitions(first_letter)
    node = reader(rt, parts, "a")
    kc.upsert("a1", 1)
    assert rt.get(node) == ["a1"]
    kc.remove("a1")
    assert kc.forget_partition(parts, "a") is False
    rt.delete_node(node)
    assert kc.forget_partition(parts, "a") is freed(pkg)
    assert kc.forget_partition(parts, "a") is False
    later = reader(rt, parts, "a")
    assert rt.get(later) == []
    kc.upsert("a2", 1)
    assert rt.get(later) == ["a2"]


def test_a_live_reader_keeps_its_partition_and_reruns_on_refill(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    parts = kc.key_partitions(first_letter)
    node = reader(rt, parts, "z")
    assert rt.get(node) == []
    assert kc.forget_partition(parts, "z") is False
    kc.upsert("z1", 1)
    assert kc.forget_partition(parts, "z") is False
    assert rt.get(node) == ["z1"]


def test_a_partition_read_but_never_filled_is_freed(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    parts = kc.key_partitions(first_letter)
    node = reader(rt, parts, "q")
    assert rt.get(node) == []
    rt.delete_node(node)
    assert kc.forget_partition(parts, "q") is freed(pkg)
    assert kc.forget_partition(parts, "never read") is False


def test_partitions_of_another_collection_are_refused(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    other = pkg.KeyedCollection(rt)
    parts = other.key_partitions(first_letter)
    with pytest.raises(ValueError, match="another keyed collection"):
        kc.forget_partition(parts, "a")


def test_forget_partition_refuses_an_overlay(local):
    rt = incr.Runtime()
    kc = incr.KeyedCollection(rt)
    parts = kc.key_partitions(first_letter)
    node = reader(rt, parts, "q")
    rt.get(node)
    rt.delete_node(node)
    with rt.overlay():
        with pytest.raises(RuntimeError, match="overlay"):
            kc.forget_partition(parts, "q")
    assert kc.forget_partition(parts, "q") is True


def test_a_persisted_collection_forgets_a_partition_like_a_plain_one(persist):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    kc = store.keyed_collection("shifts", PickleCodec(), PickleCodec())
    parts = kc.key_partitions(first_letter)
    node = reader(store.runtime, parts, "a")
    store.upsert(kc, "a1", 1)
    assert store.runtime.get(node) == ["a1"]
    store.remove(kc, "a1")
    assert kc.forget_partition(parts, "a") is False
    store.runtime.delete_node(node)
    assert kc.forget_partition(parts, "a") is (persist is incr.persist)
