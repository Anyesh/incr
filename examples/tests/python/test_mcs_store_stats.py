import pickle

import pytest
from incr.persist import Store
from test_mcs_persist_conformance import DictBackend, DictLogState
from test_mcs_persist_on_write import RecordingBackend


# Decodes only bytes this test's own store wrote.
class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class Refused(Exception):
    pass


class CountingBackend(DictBackend):
    def __init__(self, state, refuse_compact=False):
        super().__init__(state)
        self.refuse_compact = refuse_compact
        self.records_read = 0

    def read_from(self, after):
        read = super().read_from(after)
        if not isinstance(read, tuple):
            self.records_read += len(read)
        return read

    def compact(self, upto, records):
        if self.refuse_compact:
            raise Refused("compaction refused")
        return super().compact(upto, records)


def test_a_fresh_store_reports_no_snapshot():
    store = Store.open(DictBackend(DictLogState()), build_id=1)
    assert store.stats() == {
        "position": 0,
        "snapshot_position": 0,
        "records_since_checkpoint": 0,
    }


def test_the_write_after_n_records_checkpoints_first():
    store = Store.open(
        DictBackend(DictLogState()), build_id=1, auto_checkpoint_records=3
    )
    node = store.input("x", 0, PickleCodec())
    store.set(node, 1)
    store.set(node, 2)
    before = store.stats()
    assert before["records_since_checkpoint"] == 3
    store.set(node, 3)
    after = store.stats()
    assert after["snapshot_position"] == before["position"]
    assert after["records_since_checkpoint"] == 1
    store.checkpoint()
    assert store.stats()["records_since_checkpoint"] == 0


def test_a_failing_triggered_checkpoint_fails_the_write_and_leaves_the_log():
    state = DictLogState()
    backend = CountingBackend(state, refuse_compact=True)
    store = Store.open(backend, build_id=1, auto_checkpoint_records=2)
    node = store.input("x", 0, PickleCodec())
    store.set(node, 1)
    logged = list(state.records)
    with pytest.raises(Refused):
        store.set(node, 2)
    assert state.records == logged
    assert state.snapshot is None
    assert store.runtime.get(node) == 1


def test_a_reopen_after_a_triggered_checkpoint_replays_only_later_records():
    state = DictLogState()
    store = Store.open(DictBackend(state), build_id=1, auto_checkpoint_records=3)
    node = store.input("x", 0, PickleCodec())
    for value in range(1, 6):
        store.set(node, value)
    expected = store.stats()
    store.close()

    reader = CountingBackend(state)
    again = Store.open(reader, build_id=1, auto_checkpoint_records=3)
    assert again.stats() == expected
    assert reader.records_read == expected["records_since_checkpoint"]
    assert again.runtime.get(again.input("x", 0, PickleCodec())) == 5


def test_auto_checkpoint_records_must_be_positive():
    with pytest.raises(ValueError):
        Store.open(DictBackend(DictLogState()), build_id=1, auto_checkpoint_records=0)


def test_inside_an_on_write_batch_the_trigger_fires_before_the_append():
    state = DictLogState()
    backend = RecordingBackend(state)
    store = Store.open(backend, build_id=1, auto_checkpoint_records=3)
    node = store.input("x", 0, PickleCodec())
    store.set(node, 1)
    store.set(node, 2)
    due = store.stats()
    assert due["records_since_checkpoint"] == 3
    del backend.calls[:]
    with store.batch(apply="on_write") as b:
        assert [call[0] for call in backend.calls] == ["compact"]
        for value in range(3, 8):
            b.set(node, value)
    assert [call[0] for call in backend.calls] == ["compact", "append_at"]
    assert len(backend.calls[1][2]) == 5
    assert store.stats() == {
        "position": due["position"] + 5,
        "snapshot_position": due["position"],
        "records_since_checkpoint": 5,
    }

    del backend.calls[:]
    store.set(node, 8)
    assert [call[0] for call in backend.calls] == ["compact", "append"]
    assert store.stats()["records_since_checkpoint"] == 1
    store.close()

    again = Store.open(DictBackend(state), build_id=1, auto_checkpoint_records=3)
    assert again.runtime.get(again.input("x", 0, PickleCodec())) == 8
