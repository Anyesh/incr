"""R3: `store.open_memo(backend, max_bytes=, checkpoint_after_entries=)`,
`memo.stats()` and `memo.checkpoint_if_due()`. On Local the trigger is
checked after each logged write and batch commit, never on a read. Every
test takes the package's `persist` module from the `persist` fixture, so a
Shared run can add `incr_concurrent.persist` to `PACKAGES`."""

import json

import incr.persist
import pytest

PACKAGES = [pytest.param(incr.persist, id="local")]

STATS = {
    "entries",
    "bytes",
    "unsaved_entries",
    "unsaveable",
    "evicted_last_checkpoint",
    "auto_checkpoint_errors",
    "last_auto_checkpoint_error",
}


@pytest.fixture(params=PACKAGES)
def persist(request):
    return request.param


class CountingCodec:
    def __init__(self):
        self.encoded = 0

    def encode(self, obj):
        self.encoded += 1
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)


class Graph:
    """A store with the inputs `base` and `tick` and the memo table `sq`,
    whose entry for `k` is `k * k + base` and records the keys it ran for.
    Writes to `tick` are units of work that change no entry."""

    def __init__(self, persist, log, memo_backend, **memo_options):
        self.store = persist.Store.open(log, build_id="b1")
        self.base = self.store.input("base", 0, CountingCodec())
        self.tick = self.store.input("tick", 0, CountingCodec())
        self.keys = CountingCodec()
        self.calls = []

        def sq(ctx, k):
            self.calls.append(k)
            return k * k + ctx.get(self.base)

        self.table = self.store.memo_table("sq", 1, self.keys, CountingCodec(), sq)
        self.memo = self.store.open_memo(memo_backend, **memo_options)

    def read(self, k):
        return self.store.runtime.get(self.memo.get_or_insert(self.table, k))

    def unsaved(self):
        return self.memo.stats()["unsaved_entries"]


@pytest.fixture
def backends(persist):
    return persist.MemoryBackend(), persist.MemoryMemo()


def reopen(persist, graph, backends, **memo_options):
    graph.store.close()
    return Graph(persist, *backends, **memo_options)


def test_stats_has_its_fields_and_counts_unsaved_entries(persist, backends):
    graph = Graph(persist, *backends)
    stats = graph.memo.stats()
    assert set(stats) == STATS
    assert stats["unsaved_entries"] == 0
    for k in range(5):
        graph.read(k)
    assert graph.memo.stats()["unsaved_entries"] == 5
    graph.memo.checkpoint()
    stats = graph.memo.stats()
    assert (stats["entries"], stats["unsaved_entries"]) == (5, 0)
    assert stats["bytes"] > 0


def test_max_bytes_below_the_live_entries_evicts_and_a_reopen_recomputes_those(persist, backends):
    graph = Graph(persist, *backends)
    for k in range(40):
        graph.read(k)
    full = graph.memo.stats()
    graph.memo.checkpoint()
    entry_bytes = graph.memo.stats()["bytes"] // 40
    graph = reopen(persist, graph, backends, max_bytes=entry_bytes * 10)
    for k in range(40, 60):
        graph.read(k)
    evicted = graph.memo.checkpoint()["evicted"]
    assert evicted > 0
    assert graph.memo.stats()["evicted_last_checkpoint"] == evicted
    assert full["unsaved_entries"] == 40

    graph = reopen(persist, graph, backends)
    for k in range(60):
        graph.read(k)
    assert len(graph.calls) == evicted


def test_max_bytes_above_the_live_entries_evicts_nothing(persist, backends):
    graph = Graph(persist, *backends, max_bytes=1 << 30)
    for k in range(40):
        graph.read(k)
    assert graph.memo.checkpoint()["evicted"] == 0
    assert graph.memo.stats()["evicted_last_checkpoint"] == 0
    graph = reopen(persist, graph, backends)
    for k in range(40):
        graph.read(k)
    assert graph.calls == []


def test_unsaved_entries_never_pass_the_trigger_by_more_than_one_unit(persist, backends):
    trigger, per_unit = 10, 4
    graph = Graph(persist, *backends, checkpoint_after_entries=trigger)
    k = 0
    for unit in range(1, 21):
        graph.store.set(graph.tick, unit)
        assert graph.unsaved() < trigger
        for _ in range(per_unit):
            graph.read(k)
            k += 1
            assert graph.unsaved() < trigger + per_unit
    assert graph.memo.stats()["entries"] >= k - trigger - per_unit


def test_a_read_never_checkpoints_and_checkpoint_if_due_does(persist, backends):
    graph = Graph(persist, *backends, checkpoint_after_entries=10)
    assert graph.memo.checkpoint_if_due() is None
    for k in range(30):
        graph.read(k)
    assert graph.unsaved() == 30
    assert graph.memo.stats()["entries"] == 0
    stats = graph.memo.checkpoint_if_due()
    assert stats["written"] == 30
    assert graph.unsaved() == 0
    assert graph.memo.checkpoint_if_due() is None


def test_checkpoint_if_due_without_a_trigger_returns_none(persist, backends):
    graph = Graph(persist, *backends)
    for k in range(30):
        graph.read(k)
    assert graph.memo.checkpoint_if_due() is None
    assert graph.unsaved() == 30


@pytest.mark.parametrize("write", ["set", "insert", "upsert", "on_exit", "on_write"])
def test_each_logged_write_checks_the_trigger(persist, backends, write):
    graph = Graph(persist, *backends, checkpoint_after_entries=5)
    rows = graph.store.collection("rows", CountingCodec())
    kc = graph.store.keyed_collection("kc", CountingCodec(), CountingCodec())
    for k in range(5):
        graph.read(k)
    assert graph.unsaved() == 5
    if write == "set":
        graph.store.set(graph.tick, 1)
    elif write == "insert":
        graph.store.insert(rows, 1)
    elif write == "upsert":
        graph.store.upsert(kc, "a", 1)
    else:
        with graph.store.batch(apply=write) as b:
            b.set(graph.tick, 1)
            assert graph.unsaved() == 5
    assert graph.unsaved() == 0
    assert graph.memo.stats()["entries"] == 5


def test_a_triggered_checkpoint_encodes_in_proportion_to_the_new_entries(persist, backends):
    base, new = 20_000, 1_000
    graph = Graph(persist, *backends)
    for k in range(base):
        graph.read(k)
    graph.memo.checkpoint()
    graph = reopen(persist, graph, backends, checkpoint_after_entries=new)
    for k in range(base, base + new):
        graph.read(k)
    graph.keys.encoded = 0
    graph.store.set(graph.tick, 1)
    assert graph.unsaved() == 0
    assert graph.memo.stats()["entries"] == base + new
    assert new <= graph.keys.encoded <= 3 * new


def test_a_crash_between_triggered_checkpoints_loses_only_the_unsaved(persist, backends):
    graph = Graph(persist, *backends, checkpoint_after_entries=10)
    for unit in range(1, 6):
        graph.store.set(graph.tick, unit)
        for k in range(unit * 7, unit * 7 + 7):
            graph.read(k)
    unsaved = graph.unsaved()
    assert unsaved > 0
    computed = sorted(graph.calls)
    graph = reopen(persist, graph, backends)
    for k in computed:
        graph.read(k)
    assert len(graph.calls) == unsaved


def test_checkpoint_after_entries_must_be_positive(persist, backends):
    log, memo_backend = backends
    store = persist.Store.open(log, build_id="b1")
    store.memo_table("t", 1, CountingCodec(), CountingCodec(), lambda ctx, k: k)
    with pytest.raises(ValueError, match="checkpoint_after_entries"):
        store.open_memo(memo_backend, checkpoint_after_entries=0)
    store.open_memo(memo_backend, checkpoint_after_entries=1)


class FailingMemo:
    """A memo backend whose `write_generation` raises while `fail` is set,
    counting the attempts."""

    def __init__(self, persist):
        self.inner = persist.MemoryMemo()
        self.fail = False
        self.writes = 0

    def lock_checkpoint(self):
        return self.inner.lock_checkpoint()

    def load(self):
        return self.inner.load()

    def read_entry(self, *args):
        return self.inner.read_entry(*args)

    def write_generation(self, *args):
        self.writes += 1
        if self.fail:
            raise OSError("memo disk full")
        return self.inner.write_generation(*args)

    def remove_generation(self, *args):
        return self.inner.remove_generation(*args)


@pytest.fixture
def failing(persist):
    memo_backend = FailingMemo(persist)
    graph = Graph(persist, persist.MemoryBackend(), memo_backend, checkpoint_after_entries=3)
    for k in range(3):
        graph.read(k)
    memo_backend.fail = True
    return graph, memo_backend


def test_a_failed_triggered_checkpoint_leaves_the_write_returning_once(failing):
    graph, memo_backend = failing
    position = graph.store.stats()["position"]
    assert graph.store.set(graph.tick, 1) is None
    assert graph.store.stats()["position"] == position + 1
    stats = graph.memo.stats()
    assert stats["auto_checkpoint_errors"] == 1
    assert "memo disk full" in stats["last_auto_checkpoint_error"]
    assert memo_backend.writes == 1


def test_a_failed_triggered_checkpoint_waits_for_a_new_compute(failing):
    graph, memo_backend = failing
    graph.store.set(graph.tick, 1)
    graph.store.set(graph.tick, 2)
    assert memo_backend.writes == 1
    assert graph.memo.stats()["auto_checkpoint_errors"] == 1
    graph.read(10)
    memo_backend.fail = False
    graph.store.set(graph.tick, 3)
    assert memo_backend.writes == 2
    stats = graph.memo.stats()
    assert (stats["unsaved_entries"], stats["entries"]) == (0, 4)
    assert stats["auto_checkpoint_errors"] == 1


def test_reconcile_returns_its_counts_when_the_triggered_checkpoint_fails(failing):
    graph, _ = failing
    kc = graph.store.keyed_collection("kc", CountingCodec(), CountingCodec())
    assert graph.store.reconcile(kc, {"a": 1, "b": 2}) == (2, 0, 0)
    assert graph.memo.stats()["auto_checkpoint_errors"] == 1


def test_an_on_write_batch_commits_when_the_triggered_checkpoint_fails(failing):
    graph, _ = failing
    position = graph.store.stats()["position"]
    with graph.store.batch(apply="on_write") as b:
        b.set(graph.tick, 1)
    assert graph.store.stats()["position"] == position + 1
    graph.store.set(graph.tick, 2)
    assert graph.store.runtime.get(graph.tick) == 2


def test_explicit_checkpoints_still_raise(failing):
    graph, _ = failing
    with pytest.raises(OSError, match="memo disk full"):
        graph.memo.checkpoint_if_due()
    with pytest.raises(OSError, match="memo disk full"):
        graph.memo.checkpoint()
    assert graph.memo.stats()["auto_checkpoint_errors"] == 0
