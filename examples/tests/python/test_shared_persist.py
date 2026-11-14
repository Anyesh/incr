import pickle
import threading

import pytest
from incr_concurrent.persist import MemoryBackend, PersistError, Store


class PickleCodec:
    def __init__(self):
        self.threads = set()

    def encode(self, obj):
        self.threads.add(threading.get_ident())
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        self.threads.add(threading.get_ident())
        return pickle.loads(data)


class RefusingCodec(PickleCodec):
    def encode(self, obj):
        if obj == 5:
            raise ValueError(f"cannot encode {obj!r}")
        return super().encode(obj)


def test_a_set_input_reads_back_after_reopen():
    log = MemoryBackend()
    store = Store.open(log, build_id="v1")
    price = store.input("price", 1.0, PickleCodec())
    doubled = store.runtime.create_query(lambda rt: rt.get(price) * 2)
    assert store.runtime.get(doubled) == 2.0
    store.set(price, 5.0)
    assert store.runtime.get(doubled) == 10.0
    store.close()

    reopened = Store.open(log, build_id="v1")
    price = reopened.input("price", 1.0, PickleCodec())
    assert reopened.runtime.get(price) == 5.0


def test_writes_from_other_threads_find_the_store_codecs():
    log = MemoryBackend()
    store = Store.open(log, build_id="v1")
    codec = PickleCodec()
    nodes = [store.input(f"k{i}", 0, codec) for i in range(4)]
    errors = []

    def write(node, value):
        try:
            store.set(node, value)
        except Exception as error:
            errors.append(error)

    threads = [
        threading.Thread(target=write, args=(node, i + 1))
        for i, node in enumerate(nodes)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(codec.threads - {threading.get_ident()}) >= 1
    store.close()

    reopened = Store.open(log, build_id="v1")
    values = [reopened.runtime.get(reopened.input(f"k{i}", 0, codec)) for i in range(4)]
    assert values == [1, 2, 3, 4]


def test_a_codec_exception_reaches_the_caller_as_itself():
    store = Store.open(MemoryBackend(), build_id="v1")
    node = store.input("k", None, RefusingCodec())
    with pytest.raises(ValueError, match="cannot encode 5"):
        store.set(node, 5)


def test_a_value_for_a_slot_without_a_codec_raises_persist_error():
    store = Store.open(MemoryBackend(), build_id="v1")
    with pytest.raises(PersistError, match="no codec is registered"):
        store.input("k", 1, None)


def test_a_closed_store_refuses_calls():
    store = Store.open(MemoryBackend(), build_id="v1")
    node = store.input("k", 0, PickleCodec())
    store.close()
    with pytest.raises(RuntimeError, match="store is closed"):
        store.set(node, 1)
