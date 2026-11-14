"""`store.batch(apply="on_write")`: writes apply as they are made and land
as one conditional append when the block ends. Every test takes the
package's `persist` module from the `persist` fixture, so a Shared run can
add `incr_concurrent.persist` to `PACKAGES`."""

import gc
import pickle

import incr.persist
import pytest
from test_mcs_persist_conformance import DictBackend, DictLogState, EpochMismatch

PACKAGES = [pytest.param(incr.persist, id="local")]


@pytest.fixture(params=PACKAGES)
def persist(request):
    return request.param


# Decodes only bytes this test's own store wrote.
class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class Ambiguous(OSError):
    pass


class RecordingBackend(DictBackend):
    """A dict log that records each `append` and `append_at` call, and can
    fail `append_at`: `"before"` raises once without landing, `"after"`
    lands the run once and then raises, `"fence"` raises `EpochMismatch` on
    every call."""

    def __init__(self, state, fail=None):
        super().__init__(state)
        self.calls = []
        self.fail = fail

    def append(self, records):
        self.calls.append(("append", list(records)))
        return super().append(records)

    def append_at(self, after, records):
        self.calls.append(("append_at", after, list(records)))
        if self.fail == "fence":
            raise EpochMismatch("epoch moved on")
        if self.fail == "before":
            self.fail = None
            raise Ambiguous("connection reset before commit")
        landed = super().append_at(after, records)
        if self.fail == "after":
            self.fail = None
            raise Ambiguous("connection reset after commit")
        return landed

    def compact(self, upto, records):
        self.calls.append(("compact", upto))
        return super().compact(upto, records)

    def runs(self):
        return [call for call in self.calls if call[0] != "compact"]


class WithoutAppendAt(DictBackend):
    def __getattribute__(self, name):
        if name == "append_at":
            raise AttributeError(name)
        return super().__getattribute__(name)


class Graph:
    """A store with an input `members`, a query `total` over it, a
    collection `rows` and a keyed collection `pairs`, which `seed` fills
    with two keys."""

    def __init__(self, persist, backend, seed=True):
        codec = PickleCodec()
        self.store = persist.Store.open(backend, build_id=1)
        self.rt = self.store.runtime
        self.members = self.store.input("members", 0, codec)
        self.total = self.rt.create_query(lambda r: r.get(self.members) * 10)
        self.rows = self.store.collection("rows", codec)
        self.pairs = self.store.keyed_collection("pairs", codec, codec)
        if seed:
            self.store.upsert(self.pairs, "kept", 1)
            self.store.upsert(self.pairs, "dropped", 2)

    def state(self):
        return (
            self.rt.get(self.members),
            self.rt.get(self.rows.reduce(sorted)),
            sorted(self.pairs.items()),
        )


def on_write(store):
    return store.batch(apply="on_write")


def test_a_read_inside_the_block_sees_the_write_just_made(persist):
    g = Graph(persist, RecordingBackend(DictLogState()))
    assert g.rt.get(g.total) == 0
    with on_write(g.store) as b:
        b.set(g.members, 4)
        assert g.rt.get(g.total) == 40
        b.insert(g.rows, "a")
        assert g.rt.get(g.rows.reduce(sorted)) == ["a"]
        b.upsert(g.pairs, "new", 7)
        assert g.pairs.get("new") == 7
        b.remove(g.pairs, "kept")
        assert g.pairs.get("kept") is None
        b.set(g.members, 5)
        assert g.rt.get(g.total) == 50
    assert g.rt.get(g.total) == 50


def test_the_default_batch_still_applies_on_exit(persist):
    g = Graph(persist, RecordingBackend(DictLogState()))
    with g.store.batch() as b:
        b.set(g.members, 4)
        assert g.rt.get(g.total) == 0
    assert g.rt.get(g.total) == 40
    with g.store.batch(apply="on_exit") as b:
        b.set(g.members, 5)
        assert g.rt.get(g.total) == 40
    assert g.rt.get(g.total) == 50
    with pytest.raises(ValueError, match="apply"):
        g.store.batch(apply="later")


def write_everything(g, write):
    write.set(g.members, 3)
    write.insert(g.rows, "a")
    write.insert(g.rows, "b")
    write.delete(g.rows, "a")
    write.replace(g.rows, "b", "c")
    write.upsert(g.pairs, "new", 5)
    write.remove(g.pairs, "dropped")
    write.reconcile(g.pairs, {"new": 6, "x": 1})
    write.set(g.members, 4)


def test_the_backend_sees_one_append_of_every_record_in_write_order(persist):
    one_by_one = RecordingBackend(DictLogState())
    reference = Graph(persist, one_by_one)
    del one_by_one.calls[:]
    write_everything(reference, reference.store)
    expected = [record for call in one_by_one.calls for record in call[1]]

    backend = RecordingBackend(DictLogState())
    g = Graph(persist, backend)
    del backend.calls[:]
    tail = g.store.stats()["position"]
    with on_write(g.store) as b:
        write_everything(g, b)
        assert backend.calls == []
    assert backend.calls == [("append_at", tail, expected)]
    assert g.state() == reference.state()
    assert g.store.stats()["position"] == tail + len(expected)


def test_an_empty_block_appends_nothing(persist):
    backend = RecordingBackend(DictLogState())
    g = Graph(persist, backend)
    del backend.calls[:]
    with on_write(g.store):
        pass
    assert backend.calls == []
    g.store.set(g.members, 1)


def test_the_store_refuses_other_calls_inside_the_block(persist):
    g = Graph(persist, RecordingBackend(DictLogState()))
    memo = g.store.open_memo(persist.MemoryMemo())
    with on_write(g.store) as b:
        b.set(g.members, 1)
        with pytest.raises(persist.PersistError, match="staged batch is open"):
            g.store.set(g.members, 2)
        with pytest.raises(persist.PersistError, match="staged batch is open"):
            g.store.upsert(g.pairs, "k", 1)
        with pytest.raises(persist.PersistError, match="staged batch is open"):
            g.store.checkpoint()
        with pytest.raises(persist.PersistError, match="staged batch is open"):
            g.store.refresh()
        with pytest.raises(persist.PersistError, match="staged batch is open"):
            memo.checkpoint()
        with pytest.raises(persist.PersistError, match="staged batch is open"):
            with on_write(g.store):
                pass
        with pytest.raises(KeyError):
            b.remove(g.pairs, "absent")
    assert g.rt.get(g.members) == 1
    g.store.set(g.members, 2)


class Picky(PickleCodec):
    def encode(self, obj):
        if obj == "unencodable":
            raise ValueError("cannot encode")
        return super().encode(obj)


def test_a_write_whose_codec_raises_fails_alone_and_the_batch_goes_on(persist):
    backend = RecordingBackend(DictLogState())
    g = Graph(persist, backend)
    picky = g.store.input("picky", "", Picky())
    del backend.calls[:]
    with on_write(g.store) as b:
        b.set(g.members, 1)
        with pytest.raises(ValueError, match="cannot encode"):
            b.set(picky, "unencodable")
        b.set(picky, "fine")
    assert [call[0] for call in backend.calls] == ["append_at"]
    assert len(backend.calls[0][2]) == 2
    assert g.rt.get(picky) == "fine"
    g.store.set(g.members, 2)


def test_writes_after_the_block_raise(persist):
    g = Graph(persist, RecordingBackend(DictLogState()))
    batch = on_write(g.store)
    with pytest.raises(RuntimeError, match="batch"):
        batch.set(g.members, 1)
    with batch as b:
        b.set(g.members, 2)
    with pytest.raises(RuntimeError, match="batch"):
        b.set(g.members, 3)
    with pytest.raises(persist.PersistError, match="committed"):
        b.retry()
    assert g.rt.get(g.members) == 2


def test_an_exception_inside_the_block_poisons_the_store(persist):
    state = DictLogState()
    backend = RecordingBackend(state)
    g = Graph(persist, backend)
    del backend.calls[:]
    logged = list(state.records)
    with pytest.raises(ZeroDivisionError):
        with on_write(g.store) as b:
            b.set(g.members, 9)
            1 / 0
    assert backend.calls == []
    assert state.records == logged
    assert g.rt.get(g.members) == 9
    with pytest.raises(persist.PersistError, match="poisoned"):
        g.store.set(g.members, 1)
    with pytest.raises(RuntimeError, match="batch"):
        b.retry()
    g.store.close()

    again = Graph(persist, DictBackend(state), seed=False)
    assert again.rt.get(again.members) == 0


def test_a_batch_dropped_without_commit_poisons_the_store(persist):
    g = Graph(persist, RecordingBackend(DictLogState()))
    batch = on_write(g.store)
    batch.__enter__()
    batch.set(g.members, 9)
    del batch
    gc.collect()
    with pytest.raises(persist.PersistError, match="poisoned"):
        g.store.set(g.members, 1)


@pytest.mark.parametrize("fail", ["before", "after"])
def test_retry_after_an_ambiguous_append_lands_the_run_once(persist, fail):
    state = DictLogState()
    backend = RecordingBackend(state, fail=fail)
    g = Graph(persist, backend)
    tail = g.store.stats()["position"]
    with pytest.raises(Ambiguous):
        with on_write(g.store) as b:
            write_everything(g, b)
    run = backend.runs()[-1][2]
    with pytest.raises(persist.PersistError, match="staged batch is open"):
        g.store.set(g.members, 1)
    assert b.retry() == tail + len(run)
    assert backend.runs()[-1] == ("append_at", tail, run)
    assert [at for at, _ in state.records] == list(range(1, tail + len(run) + 1))
    with pytest.raises(persist.PersistError, match="committed"):
        b.retry()
    g.store.set(g.members, 8)
    live = g.state()
    g.store.close()

    again = Graph(persist, DictBackend(state), seed=False)
    assert again.state() == live


def test_a_fenced_append_leaves_nothing_in_the_log(persist):
    state = DictLogState()
    backend = RecordingBackend(state, fail="fence")
    g = Graph(persist, backend)
    logged = list(state.records)
    with pytest.raises(EpochMismatch):
        with on_write(g.store) as b:
            write_everything(g, b)
    with pytest.raises(EpochMismatch):
        b.retry()
    assert state.records == logged
    with pytest.raises(persist.PersistError, match="staged batch is open"):
        g.store.set(g.members, 1)
    g.store.close()
    del b
    gc.collect()

    again = Graph(persist, DictBackend(state), seed=False)
    assert again.rt.get(again.members) == 0


def test_a_backend_without_append_at_refuses_the_batch(persist):
    g = Graph(persist, WithoutAppendAt(DictLogState()))
    with pytest.raises(persist.PersistError, match="conditional append"):
        with on_write(g.store):
            pass
    g.store.set(g.members, 1)
    assert g.rt.get(g.total) == 10
