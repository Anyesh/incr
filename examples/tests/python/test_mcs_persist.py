import json
import pickle
import sys

import incr
import pytest
from incr.persist import MemoryBackend, Store


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class FlakyBackend:
    """Delegates to a MemoryBackend and raises from the methods named in
    `failing`."""

    def __init__(self, inner, failing=()):
        self.inner = inner
        self.failing = set(failing)

    def _call(self, name, *args):
        if name in self.failing:
            raise OSError(f"{name} refused")
        return getattr(self.inner, name)(*args)

    def append(self, records):
        return self._call("append", records)

    def sync(self, upto):
        return self._call("sync", upto)

    def read_from(self, after):
        return self._call("read_from", after)

    def load_snapshot(self):
        return self._call("load_snapshot")

    def snapshot_id(self):
        return self._call("snapshot_id")

    def compact(self, upto, folded):
        return self._call("compact", upto, folded)

    def writable(self):
        return self._call("writable")


def rows_of(rt, rows):
    return rt.get(rows.reduce(lambda rs: sorted(rs, key=repr)))


def test_submodule_is_importable_both_ways():
    import incr.persist

    from incr.persist import Store as FromImport

    assert "incr.persist" in sys.modules
    assert incr.persist.Store is FromImport


def test_input_and_collection_round_trip_through_a_backend():
    backend = MemoryBackend()
    codec = PickleCodec()
    store = Store.open(backend, build_id=1)
    count = store.input("count", 0, codec)
    rows = store.collection("rows", codec)
    store.set(count, 5)
    store.insert(rows, ("a", 1))
    store.extend(rows, [("b", 2), ("b", 2)])
    assert store.delete(rows, ("b", 2)) is None
    store.replace(rows, ("a", 1), ("a", 10))
    rt = store.runtime
    assert rt.get(count) == 5
    assert rows_of(rt, rows) == [("a", 10), ("b", 2)]
    store.close()

    again = Store.open(backend, build_id=1)
    count2 = again.input("count", 0, codec)
    rows2 = again.collection("rows", codec)
    assert again.runtime.get(count2) == 5
    assert rows_of(again.runtime, rows2) == [("a", 10), ("b", 2)]
    again.checkpoint()
    again.close()

    third = Store.open(backend, build_id=1)
    assert third.runtime.get(third.input("count", 0, codec)) == 5
    assert rows_of(third.runtime, third.collection("rows", codec)) == [
        ("a", 10),
        ("b", 2),
    ]


def test_persisted_nodes_feed_queries():
    backend = MemoryBackend()
    codec = PickleCodec()
    store = Store.open(backend, build_id="v1")
    count = store.input("count", 1, codec)
    rows = store.collection("rows", codec)
    rt = store.runtime
    q = rt.create_query(lambda r: r.get(count) * 10)
    total = rows.aggregate(0, lambda r: r, lambda a, b: a + b)
    assert rt.get(q) == 10
    store.set(count, 4)
    store.extend(rows, [1, 2, 3])
    assert rt.get(q) == 40
    assert rt.get(total) == 6


@pytest.mark.parametrize(
    "value",
    [
        incr.Pending("k"),
        incr.Pending(),
        incr.LookupFailed("gone", key=("a", 1)),
        incr.Unavailable("withheld", row=("id", 3)),
        incr.Incomplete(41, 2),
    ],
)
def test_not_ready_values_restore_as_themselves(value):
    backend = MemoryBackend()
    codec = PickleCodec()
    store = Store.open(backend, build_id=1)
    node = store.input("shift", None, codec)
    rows = store.collection("rows", codec)
    store.set(node, value)
    store.insert(rows, value)
    store.close()

    again = Store.open(backend, build_id=1)
    node2 = again.input("shift", None, codec)
    rows2 = again.collection("rows", codec)
    value_back, status = again.runtime.get_with_status(node2)
    assert again.runtime.status(node2) == status
    if isinstance(value, incr.Incomplete):
        assert status == value
        assert value_back == value.value
    else:
        assert status == value
    elements = again.runtime.get(rows2.reduce(lambda rs: list(rs)))
    assert elements == ([41] if isinstance(value, incr.Incomplete) else [])


def test_quarantined_input_equals_what_was_stored():
    backend = MemoryBackend()
    codec = PickleCodec()
    store = Store.open(backend, build_id=1)
    node = store.input("shift", None, codec)
    stored = incr.Unavailable("withheld", row=("emp", 7))
    store.set(node, stored)
    store.close()

    again = Store.open(backend, build_id=1)
    node2 = again.input("shift", None, codec)
    assert again.runtime.status(node2) == stored
    assert type(again.runtime.status(node2)) is incr.Unavailable


def test_nested_not_ready_fields_go_through_the_codec():
    class Marked(PickleCodec):
        def __init__(self):
            self.seen = []

        def encode(self, obj):
            self.seen.append(obj)
            return super().encode(obj)

    codec = Marked()
    store = Store.open(MemoryBackend(), build_id=1)
    node = store.input("shift", None, codec)
    codec.seen.clear()
    store.set(node, incr.Unavailable("why", row=("emp", 7)))
    assert ("emp", 7) in codec.seen
    assert "why" in codec.seen


def test_no_codec_for_a_slot_raises_naming_it():
    store = Store.open(MemoryBackend(), build_id=1)
    with pytest.raises(Exception, match="'count'"):
        store.input("count", 0, None)
    rows = store.collection("rows", None)
    with pytest.raises(Exception, match="'rows'"):
        store.insert(rows, 1)
    assert rows.snapshot_len() == 0


def test_codec_that_fails_to_decode_surfaces_its_error():
    backend = MemoryBackend()
    store = Store.open(backend, build_id=1)
    store.set(store.input("count", 0, PickleCodec()), 9)
    store.close()

    class Broken:
        def encode(self, obj):
            return json.dumps(obj).encode()

        def decode(self, data):
            raise ValueError("cannot read these bytes")

    again = Store.open(backend, build_id=1)
    with pytest.raises(ValueError, match="cannot read these bytes"):
        again.input("count", 0, Broken())
    node = again.input("count", 0, PickleCodec())
    assert again.runtime.get(node) == 9


def test_codec_that_fails_to_encode_surfaces_its_error_and_writes_nothing():
    class Broken:
        def encode(self, obj):
            if obj == 13:
                raise ValueError("unlucky")
            return PickleCodec().encode(obj)

        def decode(self, data):
            return PickleCodec().decode(data)

    backend = MemoryBackend()
    store = Store.open(backend, build_id=1)
    node = store.input("count", 0, Broken())
    with pytest.raises(ValueError, match="unlucky"):
        store.set(node, 13)
    assert store.runtime.get(node) == 0
    store.set(node, 14)
    assert store.runtime.get(node) == 14


def test_rt_set_on_a_persisted_input_raises():
    store = Store.open(MemoryBackend(), build_id=1)
    node = store.input("count", 0, PickleCodec())
    with pytest.raises(RuntimeError, match=r"store\.set"):
        store.runtime.set(node, 5)
    assert store.runtime.get(node) == 0
    plain = store.runtime.create_input(1)
    store.runtime.set(plain, 2)
    assert store.runtime.get(plain) == 2


def test_direct_collection_mutation_on_a_persisted_collection_raises():
    store = Store.open(MemoryBackend(), build_id=1)
    rows = store.collection("rows", PickleCodec())
    for call in (
        lambda: rows.insert(1),
        lambda: rows.delete(1),
        lambda: rows.extend([1, 2]),
        lambda: rows.replace(1, 2),
    ):
        with pytest.raises(RuntimeError, match=r"store\."):
            call()
    assert rows.snapshot_len() == 0
    store.insert(rows, 1)
    assert rows.snapshot_len() == 1


def test_store_set_rejects_a_node_that_is_not_a_persisted_input():
    store = Store.open(MemoryBackend(), build_id=1)
    plain = store.runtime.create_input(1)
    with pytest.raises(ValueError, match="not a persisted input"):
        store.set(plain, 2)
    other = store.runtime.create_collection()
    with pytest.raises(ValueError, match="not a persisted collection"):
        store.insert(other, 2)


def test_runtime_close_points_to_store_close():
    store = Store.open(MemoryBackend(), build_id=1)
    with pytest.raises(RuntimeError, match=r"store\.close"):
        store.runtime.close()
    node = store.input("count", 0, PickleCodec())
    assert store.runtime.get(node) == 0
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.runtime.get(node)
    with pytest.raises(RuntimeError, match="closed"):
        store.set(node, 1)


def test_failing_append_raises_the_backend_error_and_leaves_the_store_usable():
    inner = MemoryBackend()
    backend = FlakyBackend(inner)
    store = Store.open(backend, build_id=1)
    node = store.input("count", 0, PickleCodec())
    backend.failing = {"append"}
    with pytest.raises(OSError, match="append refused"):
        store.set(node, 5)
    assert store.runtime.get(node) == 0
    backend.failing = set()
    store.set(node, 6)
    assert store.runtime.get(node) == 6


def test_failing_sync_poisons_the_store():
    inner = MemoryBackend()
    backend = FlakyBackend(inner)
    store = Store.open(backend, build_id=1)
    node = store.input("count", 0, PickleCodec())
    backend.failing = {"sync"}
    with pytest.raises(OSError, match="sync refused"):
        store.set(node, 5)
    backend.failing = set()
    with pytest.raises(incr.persist.PersistError, match="poisoned"):
        store.set(node, 6)
    with pytest.raises(incr.persist.PersistError, match="poisoned"):
        store.checkpoint()


def test_build_id_is_an_int_or_a_string_and_does_not_gate_inputs():
    backend = MemoryBackend()
    codec = PickleCodec()
    store = Store.open(backend, build_id=7)
    store.set(store.input("count", 0, codec), 3)
    store.close()
    for build_id in ("other-build", 8, 2**100):
        again = Store.open(backend, build_id=build_id)
        assert again.runtime.get(again.input("count", 0, codec)) == 3
        again.close()
    for bad in (1.5, None, b"x", -1):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            Store.open(backend, build_id=bad)


def test_durability_option():
    backend = FlakyBackend(MemoryBackend(), failing={"sync"})
    store = Store.open(backend, build_id=1, durability="buffered")
    node = store.input("count", 0, PickleCodec())
    store.set(node, 2)
    assert store.runtime.get(node) == 2
    with pytest.raises(ValueError, match="durability"):
        Store.open(MemoryBackend(), build_id=1, durability="sometimes")


def test_checkpoint_compacts_and_a_reopen_sees_the_same_state():
    backend = MemoryBackend()
    codec = PickleCodec()
    store = Store.open(backend, build_id=1)
    count = store.input("count", 0, codec)
    rows = store.collection("rows", codec)
    for i in range(5):
        store.set(count, i)
        store.insert(rows, i)
    store.checkpoint()
    assert backend.snapshot_id() is not None
    store.close()
    again = Store.open(backend, build_id=1)
    assert again.runtime.get(again.input("count", 0, codec)) == 4
    assert rows_of(again.runtime, again.collection("rows", codec)) == [0, 1, 2, 3, 4]


def test_memory_backend_protocol():
    source = MemoryBackend()
    store = Store.open(source, build_id=1)
    store.set(store.input("count", 0, PickleCodec()), 3)
    store.close()
    entries = source.read_from(0)
    assert [at for at, _ in entries] == [1, 2]
    assert all(isinstance(record, bytes) for _, record in entries)

    backend = MemoryBackend()
    assert backend.writable() is True
    assert backend.snapshot_id() is None
    assert backend.load_snapshot() is None
    assert backend.read_from(0) == []
    records = [record for _, record in entries]
    assert backend.append(records) == 2
    assert backend.append([]) == 2
    backend.sync(2)
    assert backend.read_from(0) == entries
    assert backend.read_from(1) == entries[1:]
    backend.compact(2, records[:1])
    assert backend.snapshot_id() == (2, 1)
    assert backend.load_snapshot() == (2, records[:1])
    assert backend.read_from(0) == ("compacted", 2)
    assert backend.read_from(2) == []
    read_only = backend.read_only()
    assert read_only.writable() is False
    with pytest.raises(RuntimeError, match="read-only"):
        read_only.append(records)
    with pytest.raises(RuntimeError, match="decode"):
        backend.append([b"\x01"])
