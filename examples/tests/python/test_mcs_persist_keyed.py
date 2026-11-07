import pickle

import incr
import pytest
from incr.persist import MemoryBackend, Store


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class RecordingBackend:
    def __init__(self, inner, failing=()):
        self.inner = inner
        self.failing = set(failing)
        self.appends = []

    def _call(self, name, *args):
        if name in self.failing:
            raise OSError(f"{name} refused")
        return getattr(self.inner, name)(*args)

    def append(self, records):
        if "append" in self.failing:
            raise OSError("append refused")
        self.appends.append(len(records))
        return self.inner.append(records)

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


class Key:
    log = []

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        Key.log.append(("hash", self.name))
        return hash(self.name)

    def __eq__(self, other):
        Key.log.append(("eq", self.name))
        return isinstance(other, Key) and other.name == self.name

    def __reduce__(self):
        return (Key, (self.name,))

    def __repr__(self):
        return f"Key({self.name!r})"


def rows_of(rt, collection):
    return rt.get(collection.reduce(sorted))


def open_store(backend, build_id=1):
    return Store.open(backend, build_id=build_id)


def declare(store, name="rows"):
    codec = PickleCodec()
    return store.keyed_collection(name, codec, codec)


def test_batch_applies_every_operation_as_one_append():
    inner = MemoryBackend()
    backend = RecordingBackend(inner)
    codec = PickleCodec()
    store = open_store(backend)
    count = store.input("count", 0, codec)
    rows = store.collection("rows", codec)
    backend.appends.clear()
    with store.batch() as b:
        b.set(count, 7)
        b.insert(rows, "a")
        b.insert(rows, "b")
        b.insert(rows, "b")
        b.delete(rows, "b")
        b.replace(rows, "a", "c")
    assert backend.appends == [6]
    assert store.runtime.get(count) == 7
    assert rows_of(store.runtime, rows) == ["b", "c"]
    store.close()

    again = open_store(inner)
    assert again.runtime.get(again.input("count", 0, codec)) == 7
    assert rows_of(again.runtime, again.collection("rows", codec)) == ["b", "c"]


def test_a_batch_that_raises_applies_nothing():
    inner = MemoryBackend()
    backend = RecordingBackend(inner)
    codec = PickleCodec()
    store = open_store(backend)
    count = store.input("count", 0, codec)
    rows = store.collection("rows", codec)
    kc = declare(store, "pairs")
    backend.appends.clear()
    with pytest.raises(ZeroDivisionError):
        with store.batch() as b:
            b.set(count, 7)
            b.insert(rows, "a")
            b.upsert(kc, "k", 1)
            1 / 0
    assert backend.appends == []
    assert store.runtime.get(count) == 0
    assert rows_of(store.runtime, rows) == []
    assert len(kc) == 0
    store.set(count, 1)
    assert store.runtime.get(count) == 1


def test_an_empty_batch_writes_nothing():
    backend = RecordingBackend(MemoryBackend())
    store = open_store(backend)
    store.input("count", 0, PickleCodec())
    backend.appends.clear()
    with store.batch():
        pass
    assert backend.appends == []


def test_batch_operations_outside_the_with_block_raise():
    store = open_store(MemoryBackend())
    count = store.input("count", 0, PickleCodec())
    batch = store.batch()
    with pytest.raises(RuntimeError, match="batch"):
        batch.set(count, 1)
    with batch as b:
        b.set(count, 2)
    with pytest.raises(RuntimeError, match="batch"):
        b.set(count, 3)
    assert store.runtime.get(count) == 2


def test_batch_rejects_a_node_the_store_does_not_persist():
    store = open_store(MemoryBackend())
    plain = store.runtime.create_input(1)
    with pytest.raises(ValueError, match="not a persisted input"):
        with store.batch() as b:
            b.set(plain, 2)


def test_two_inputs_and_a_progress_input_land_in_one_append():
    inner = MemoryBackend()
    backend = RecordingBackend(inner)
    codec = PickleCodec()
    store = open_store(backend)
    a = store.input("a", 0, codec)
    second = store.input("b", 0, codec)
    progress = store.input("progress", 0, codec)
    backend.appends.clear()
    with store.batch() as b:
        b.set(a, 1)
        b.set(second, 2)
        b.set(progress, 99)
    assert backend.appends == [3]
    store.close()

    again = open_store(inner)
    values = [
        again.runtime.get(again.input(k, 0, codec)) for k in ("a", "b", "progress")
    ]
    assert values == [1, 2, 99]


def test_a_failing_append_in_a_batch_leaves_all_three_unwritten():
    inner = MemoryBackend()
    backend = RecordingBackend(inner)
    codec = PickleCodec()
    store = open_store(backend)
    a = store.input("a", 0, codec)
    progress = store.input("progress", 0, codec)
    backend.failing = {"append"}
    with pytest.raises(OSError, match="append refused"):
        with store.batch() as b:
            b.set(a, 1)
            b.set(progress, 99)
    assert store.runtime.get(a) == 0
    backend.failing = set()
    store.close()

    again = open_store(inner)
    assert again.runtime.get(again.input("a", 0, codec)) == 0
    assert again.runtime.get(again.input("progress", 0, codec)) == 0


def test_a_failing_sync_mid_batch_poisons_and_the_log_holds_all_or_none():
    inner = MemoryBackend()
    backend = RecordingBackend(inner)
    codec = PickleCodec()
    store = open_store(backend)
    a = store.input("a", 0, codec)
    progress = store.input("progress", 0, codec)
    kc = declare(store)
    backend.failing = {"sync"}
    with pytest.raises(OSError, match="sync refused"):
        with store.batch() as b:
            b.set(a, 1)
            b.set(progress, 99)
            b.upsert(kc, "k", 5)
    backend.failing = set()
    with pytest.raises(incr.persist.PersistError, match="poisoned"):
        store.set(a, 2)

    again = open_store(inner)
    seen = (
        again.runtime.get(again.input("a", 0, codec)),
        again.runtime.get(again.input("progress", 0, codec)),
        declare(again).get("k"),
    )
    assert seen in {(0, 0, None), (1, 99, 5)}


def test_batch_inside_an_overlay_raises():
    store = open_store(MemoryBackend())
    store.input("count", 0, PickleCodec())
    with store.runtime.overlay():
        with pytest.raises(RuntimeError, match="overlay"):
            store.batch()


def test_keyed_collection_reads_through_the_graph_and_the_index():
    store = open_store(MemoryBackend())
    kc = declare(store)
    store.upsert(kc, "a", 1)
    store.upsert(kc, "b", 2)
    assert kc.get("a") == 1
    assert "b" in kc
    assert len(kc) == 2
    assert rows_of(store.runtime, kc.collection) == [1, 2]
    store.upsert(kc, "a", 10)
    assert kc.get("a") == 10
    assert len(kc) == 2
    assert rows_of(store.runtime, kc.collection) == [2, 10]
    store.remove(kc, "b")
    assert kc.get("b") is None
    with pytest.raises(KeyError):
        store.remove(kc, "b")
    assert store.reconcile(kc, {"x": 1, "y": 2}) == (2, 0, 1)
    assert rows_of(store.runtime, kc.collection) == [1, 2]


def test_keyed_collection_restores_on_reopen_with_a_working_index():
    backend = MemoryBackend()
    store = open_store(backend)
    kc = declare(store)
    store.upsert(kc, Key("a"), 1)
    store.upsert(kc, Key("b"), 2)
    store.close()

    again = open_store(backend)
    kc2 = declare(again)
    assert len(kc2) == 2
    assert kc2.get(Key("a")) == 1
    again.upsert(kc2, Key("a"), 11)
    assert len(kc2) == 2
    assert kc2.get(Key("a")) == 11
    assert rows_of(again.runtime, kc2.collection) == [2, 11]
    again.checkpoint()
    again.close()

    third = open_store(backend)
    kc3 = declare(third)
    assert kc3.get(Key("a")) == 11
    third.upsert(kc3, Key("a"), 12)
    assert rows_of(third.runtime, kc3.collection) == [2, 12]


def test_restoring_the_index_runs_key_hash_in_the_declaration():
    backend = MemoryBackend()
    store = open_store(backend)
    kc = declare(store)
    store.upsert(kc, Key("a"), 1)
    store.close()
    again = open_store(backend)
    Key.log.clear()
    kc2 = declare(again)
    assert ("hash", "a") in Key.log
    Key.log.clear()
    assert kc2.get(Key("a")) == 1
    assert ("eq", "a") in Key.log


def test_not_ready_rows_persist_in_a_keyed_collection():
    backend = MemoryBackend()
    store = open_store(backend)
    kc = declare(store)
    store.upsert(kc, "k", incr.Unavailable("withheld", row=("id", 3)))
    store.close()
    again = open_store(backend)
    row = declare(again).get("k")
    assert isinstance(row, incr.Unavailable)
    assert row.reason == "withheld"


def test_direct_writes_to_a_persisted_keyed_collection_raise():
    store = open_store(MemoryBackend())
    kc = declare(store)
    store.upsert(kc, "a", 1)
    for call in (
        lambda: kc.upsert("a", 2),
        lambda: kc.remove("a"),
        lambda: kc.reconcile({"b": 3}),
    ):
        with pytest.raises(RuntimeError, match=r"store\."):
            call()
    assert kc.get("a") == 1
    assert len(kc) == 1
    assert rows_of(store.runtime, kc.collection) == [1]


def test_store_keyed_calls_reject_a_keyed_collection_of_another_store():
    store = open_store(MemoryBackend())
    other = incr.KeyedCollection(store.runtime)
    with pytest.raises(ValueError, match="not a persisted keyed collection"):
        store.upsert(other, "a", 1)
    with pytest.raises(ValueError, match="not a persisted keyed collection"):
        with store.batch() as b:
            b.upsert(other, "a", 1)


def test_keyed_ops_in_a_batch_compose_on_one_key():
    inner = MemoryBackend()
    backend = RecordingBackend(inner)
    store = open_store(backend)
    kc = declare(store)
    store.upsert(kc, "k", 0)
    backend.appends.clear()
    with store.batch() as b:
        b.upsert(kc, "k", 1)
        b.upsert(kc, "k", 2)
    assert backend.appends == [2]
    assert kc.get("k") == 2
    assert len(kc) == 1
    assert rows_of(store.runtime, kc.collection) == [2]

    with store.batch() as b:
        b.upsert(kc, "k", 3)
        b.remove(kc, "k")
        b.upsert(kc, "j", 4)
    assert kc.get("k") is None
    assert kc.get("j") == 4
    assert rows_of(store.runtime, kc.collection) == [4]

    with store.batch() as b:
        b.upsert(kc, "m", 5)
        b.upsert(kc, "m", 6)
        b.upsert(kc, "m", 6)
    store.close()

    again = open_store(inner)
    kc2 = declare(again)
    assert len(kc2) == 2
    assert kc2.get("m") == 6
    assert rows_of(again.runtime, kc2.collection) == [4, 6]


def test_a_batch_remove_of_an_absent_key_raises_and_sees_staged_keys():
    store = open_store(MemoryBackend())
    kc = declare(store)
    with pytest.raises(KeyError):
        with store.batch() as b:
            b.remove(kc, "a")
    with store.batch() as b:
        b.upsert(kc, "a", 1)
        b.remove(kc, "a")
        with pytest.raises(KeyError):
            b.remove(kc, "a")
    assert len(kc) == 0
    assert rows_of(store.runtime, kc.collection) == []


def test_a_batch_reconcile_composes_with_other_keyed_ops():
    store = open_store(MemoryBackend())
    kc = declare(store)
    store.reconcile(kc, {"a": 1, "b": 2})
    with store.batch() as b:
        b.upsert(kc, "c", 3)
        b.reconcile(kc, [("a", 10), ("c", 30)])
        b.upsert(kc, "d", 4)
    assert len(kc) == 3
    assert (kc.get("a"), kc.get("b"), kc.get("c"), kc.get("d")) == (10, None, 30, 4)
    assert rows_of(store.runtime, kc.collection) == [4, 10, 30]


def test_a_failing_batch_leaves_the_index_unchanged():
    inner = MemoryBackend()
    backend = RecordingBackend(inner)
    store = open_store(backend)
    kc = declare(store)
    store.upsert(kc, "a", 1)
    backend.failing = {"append"}
    with pytest.raises(OSError, match="append refused"):
        with store.batch() as b:
            b.upsert(kc, "a", 2)
            b.upsert(kc, "b", 3)
    assert kc.get("a") == 1
    assert kc.get("b") is None
    assert len(kc) == 1
    backend.failing = set()
    store.upsert(kc, "a", 5)
    assert rows_of(store.runtime, kc.collection) == [5]


def test_keyed_ops_in_one_batch_across_two_collections_and_a_plain_input():
    inner = MemoryBackend()
    backend = RecordingBackend(inner)
    store = open_store(backend)
    first = declare(store, "first")
    second = declare(store, "second")
    progress = store.input("progress", 0, PickleCodec())
    backend.appends.clear()
    with store.batch() as b:
        b.upsert(first, "k", 1)
        b.upsert(second, "k", 2)
        b.upsert(first, "k", 3)
        b.set(progress, 4)
    assert backend.appends == [3]
    assert (first.get("k"), second.get("k")) == (3, 2)
    store.close()

    again = open_store(inner)
    assert declare(again, "first").get("k") == 3
    assert declare(again, "second").get("k") == 2
    assert again.runtime.get(again.input("progress", 0, PickleCodec())) == 4


def test_refresh_returns_the_position_applied_through():
    backend = MemoryBackend()
    leader = open_store(backend)
    standby = open_store(backend)
    count = leader.input("count", 0, PickleCodec())
    leader.set(count, 3)
    assert standby.refresh() == 2
    assert standby.refresh() == 2


def test_refresh_makes_the_leaders_rows_visible_and_upsert_replaces_them():
    backend = MemoryBackend()
    leader = open_store(backend)
    standby = open_store(backend)
    kc_leader = declare(leader)
    kc_standby = declare(standby)
    leader.upsert(kc_leader, "a", 1)
    assert kc_standby.get("a") is None
    standby.refresh()
    assert kc_standby.get("a") == 1
    assert rows_of(standby.runtime, kc_standby.collection) == [1]

    leader.upsert(kc_leader, "a", 2)
    leader.upsert(kc_leader, "b", 3)
    leader.remove(kc_leader, "b")
    standby.refresh()
    assert kc_standby.get("a") == 2
    assert kc_standby.get("b") is None
    standby.upsert(kc_standby, "a", 9)
    assert len(kc_standby) == 1
    assert rows_of(standby.runtime, kc_standby.collection) == [9]
    leader.refresh()
    assert kc_leader.get("a") == 9
    assert rows_of(leader.runtime, kc_leader.collection) == [9]


def test_refresh_applies_a_leader_batch_to_the_index_with_key_code_in_refresh():
    backend = MemoryBackend()
    leader = open_store(backend)
    standby = open_store(backend)
    kc_leader = declare(leader)
    kc_standby = declare(standby)
    with leader.batch() as b:
        b.upsert(kc_leader, Key("a"), 1)
        b.upsert(kc_leader, Key("b"), 2)
        b.upsert(kc_leader, Key("a"), 3)
    Key.log.clear()
    standby.refresh()
    assert ("hash", "a") in Key.log
    assert kc_standby.get(Key("a")) == 3
    assert kc_standby.get(Key("b")) == 2
    assert len(kc_standby) == 2


def test_a_standby_that_refreshes_across_a_leader_compaction_then_upserts():
    backend = MemoryBackend()
    leader = open_store(backend)
    standby = open_store(backend)
    kc_leader = declare(leader)
    kc_standby = declare(standby)
    leader.reconcile(kc_leader, {"a": 1, "b": 2, "c": 3})
    standby.refresh()
    leader.upsert(kc_leader, "a", 10)
    leader.remove(kc_leader, "b")
    leader.upsert(kc_leader, "d", 4)
    leader.checkpoint()
    standby.refresh()
    assert kc_standby.get("a") == 10
    assert kc_standby.get("b") is None
    assert kc_standby.get("d") == 4
    assert len(kc_standby) == 3
    standby.upsert(kc_standby, "a", 11)
    standby.upsert(kc_standby, "d", 5)
    assert len(kc_standby) == 3
    assert rows_of(standby.runtime, kc_standby.collection) == [3, 5, 11]


def test_a_standby_behind_a_compaction_that_forgot_the_key_rebuilds_its_index():
    backend = MemoryBackend()
    leader = open_store(backend)
    standby = open_store(backend)
    kc_leader = declare(leader)
    kc_standby = declare(standby)
    leader.upsert(kc_leader, "a", 1)
    standby.refresh()
    assert len(kc_standby) == 1
    leader.remove(kc_leader, "a")
    leader.checkpoint()
    standby.refresh()
    assert len(kc_standby) == 0
    assert kc_standby.get("a") is None
    standby.upsert(kc_standby, "a", 2)
    assert rows_of(standby.runtime, kc_standby.collection) == [2]
    assert len(kc_standby) == 1


def test_a_write_that_catches_up_leaves_the_index_matching_the_collection():
    backend = MemoryBackend()
    leader = open_store(backend)
    standby = open_store(backend)
    kc_leader = declare(leader)
    kc_standby = declare(standby)
    leader.upsert(kc_leader, "a", 1)
    standby.upsert(kc_standby, "b", 2)
    assert kc_standby.get("a") == 1
    assert kc_standby.get("b") == 2
    assert len(kc_standby) == 2
    assert rows_of(standby.runtime, kc_standby.collection) == [1, 2]


def test_refresh_without_a_keyed_collection_applies_plain_nodes():
    backend = MemoryBackend()
    leader = open_store(backend)
    standby = open_store(backend)
    count_leader = leader.input("count", 0, PickleCodec())
    count_standby = standby.input("count", 0, PickleCodec())
    leader.set(count_leader, 4)
    standby.refresh()
    assert standby.runtime.get(count_standby) == 4
