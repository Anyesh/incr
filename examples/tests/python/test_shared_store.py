"""The Shared package's store surface: what it registers, how its runtime
refuses an overlay, what its close leaves, and store use from several
threads at once."""

import pickle
import threading
import time

import incr_concurrent.persist as persist
import pytest
from test_mcs_persist_conformance import DictBackend, DictLogState


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


def test_tail_moved_is_a_persist_error():
    assert issubclass(persist.TailMoved, persist.PersistError)


def test_overlay_raises_naming_the_missing_isolation():
    store = persist.Store.open(DictBackend(DictLogState()), build_id=1)
    with pytest.raises(RuntimeError, match="isolat"):
        store.runtime.overlay()


def test_close_refuses_store_calls_and_keeps_the_runtime_values():
    store = persist.Store.open(DictBackend(DictLogState()), build_id=1)
    node = store.input("x", 0, PickleCodec())
    store.set(node, 4)
    runtime = store.runtime
    store.close()
    with pytest.raises(RuntimeError):
        store.set(node, 5)
    with pytest.raises(RuntimeError):
        store.stats()
    with pytest.raises(RuntimeError):
        store.checkpoint()
    assert runtime.get(node) == 4


def test_refresh_applies_another_writers_records():
    state = DictLogState()
    writer = persist.Store.open(DictBackend(state), build_id=1)
    reader = persist.Store.open(DictBackend(state), build_id=1)
    codec = PickleCodec()
    written = writer.input("x", 0, codec)
    read = reader.input("x", 0, codec)
    writer.set(written, 7)
    assert reader.refresh() == writer.stats()["position"]
    assert reader.runtime.get(read) == 7


def test_threads_writing_and_reading_one_store_finish():
    store = persist.Store.open(
        DictBackend(DictLogState()), build_id=1, auto_checkpoint_records=5
    )
    codec = PickleCodec()
    nodes = [store.input(f"k{i}", 0, codec) for i in range(4)]
    total = store.runtime.create_query(
        lambda rt: sum(rt.get(node) for node in nodes)
    )
    errors = []

    def write(node):
        try:
            for value in range(1, 21):
                store.set(node, value)
                store.runtime.get(total)
                store.stats()
        except Exception as error:
            errors.append(error)

    def check():
        try:
            for _ in range(20):
                store.checkpoint()
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=write, args=(node,)) for node in nodes]
    threads.append(threading.Thread(target=check))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert store.runtime.get(total) == 80


def test_a_failed_declaration_does_not_unregister_a_concurrent_ones_codec():
    state = DictLogState()
    writer = persist.Store.open(DictBackend(state), build_id=1)
    writer.set(writer.input("x", 0, PickleCodec()), 9)
    writer.close()

    b_started = threading.Event()

    class FailsFirstDecode(PickleCodec):
        def __init__(self):
            self.decodes = 0

        def decode(self, data):
            self.decodes += 1
            if self.decodes == 1:
                b_started.wait(10)
                time.sleep(0.2)
                raise ValueError("first decode refused")
            return super().decode(data)

    store = persist.Store.open(DictBackend(state), build_id=1)
    codec = FailsFirstDecode()
    results = {}

    def declare(name):
        try:
            results[name] = store.runtime.get(store.input("x", 0, codec))
        except Exception as error:
            results[name] = error

    a = threading.Thread(target=declare, args=("a",))
    a.start()
    while codec.decodes == 0:
        time.sleep(0.01)
    b = threading.Thread(target=declare, args=("b",))
    b.start()
    b_started.set()
    a.join(30)
    b.join(30)
    assert isinstance(results["a"], ValueError)
    assert results["b"] == 9
