import pickle
import time

import pytest

from incr import KeyedCollection, Runtime
from incr.persist import MemoryBackend, Store


# Decodes only bytes this test's own store wrote.
class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


def declare(store, name="rows"):
    codec = PickleCodec()
    return store.keyed_collection(name, codec, codec)


def test_keys_and_items_list_the_held_keys_after_upserts_and_a_remove():
    rt = Runtime()
    kc = KeyedCollection(rt)
    kc.upsert("a", 1)
    kc.upsert("b", 2)
    kc.upsert("c", 3)
    kc.remove("b")
    assert sorted(kc.keys()) == ["a", "c"]
    assert sorted(kc.items()) == [("a", 1), ("c", 3)]


def test_keys_of_a_persisted_keyed_collection_after_a_reopen():
    backend = MemoryBackend()
    store = Store.open(backend, build_id=1)
    kc = declare(store)
    for key in ("a", "b", "c"):
        store.upsert(kc, key, key.upper())
    store.remove(kc, "b")
    store.close()

    again = Store.open(backend, build_id=1)
    kc2 = declare(again)
    assert sorted(kc2.keys()) == ["a", "c"]
    assert sorted(kc2.items()) == [("a", "A"), ("c", "C")]


def test_mutating_while_iterating_a_returned_snapshot_does_not_raise():
    rt = Runtime()
    kc = KeyedCollection(rt)
    for i in range(10):
        kc.upsert(i, i)
    for key in kc.keys():
        kc.remove(key)
        kc.upsert(key + 100, key)
    for key, row in kc.items():
        kc.upsert(key, row + 1)
    assert sorted(kc.keys()) == list(range(100, 110))


def test_keys_are_a_list_snapshot():
    rt = Runtime()
    kc = KeyedCollection(rt)
    kc.upsert("a", 1)
    keys = kc.keys()
    kc.upsert("b", 2)
    assert keys == ["a"]
    assert isinstance(kc.items(), list)


def test_keys_raise_once_the_runtime_is_closed():
    rt = Runtime()
    kc = KeyedCollection(rt)
    kc.upsert("a", 1)
    rt.close()
    with pytest.raises(RuntimeError, match="closed"):
        kc.keys()
    with pytest.raises(RuntimeError, match="closed"):
        kc.items()


def test_keys_from_a_key_method_during_an_operation_raise():
    rt = Runtime()
    kc = KeyedCollection(rt)
    kc.upsert("held", 0)

    class Key:
        def __hash__(self):
            kc.keys()
            return 1

        def __eq__(self, other):
            return self is other

    with pytest.raises(RuntimeError, match="key or row method"):
        kc.upsert(Key(), 1)
    assert kc.items() == [("held", 0)]


def test_240000_keys_list_in_under_a_second():
    rt = Runtime()
    kc = KeyedCollection(rt)
    kc.reconcile({i: i for i in range(240_000)})
    start = time.perf_counter()
    keys = kc.keys()
    elapsed = time.perf_counter() - start
    assert len(keys) == 240_000
    assert elapsed < 1.0, f"keys() took {elapsed:.3f}s"
