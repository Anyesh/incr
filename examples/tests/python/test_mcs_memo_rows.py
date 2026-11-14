"""R10: a memo entry that reads keys of a persisted keyed collection through
`ctx.get_key` depends on those keys alone across a checkpoint and a
reopen. Every test takes the package's `persist` module from the `persist`
fixture, so a Shared run can add `incr_concurrent.persist` to `PACKAGES`."""

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


# Decodes only bytes this test's own store wrote.
class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class Graph:
    """A store over `log` with the keyed collection `shifts` and the memo
    tables `price` (a row's first field, `None` for an absent key),
    `status` (the reason of an `Unavailable` row) and `total` (over
    `price`), each recording the keys it ran for."""

    def __init__(self, persist, log, memo_backend):
        self.store = persist.Store.open(log, build_id="b1")
        self.shifts = self.store.keyed_collection(
            "shifts", PickleCodec(), PickleCodec()
        )
        self.calls = []

        def price(ctx, key):
            self.calls.append(("price", key))
            row = ctx.get_key(self.shifts, key)
            return None if row is None else row[0]

        def status(ctx, key):
            self.calls.append(("status", key))
            row, status = ctx.get_key_status(self.shifts, key)
            return None if status is None else status.reason

        def total(ctx, key):
            self.calls.append(("total", key))
            return ctx.memo(self.price, key)

        self.price = self.store.memo_table(
            "price", 1, PickleCodec(), PickleCodec(), price
        )
        self.status = self.store.memo_table(
            "status", 1, PickleCodec(), PickleCodec(), status
        )
        self.total = self.store.memo_table(
            "total", 1, PickleCodec(), PickleCodec(), total
        )
        self.memo = self.store.open_memo(memo_backend)

    def read(self, table, key):
        return self.store.runtime.get(self.memo.get_or_insert(table, key))


@pytest.fixture
def backends(persist):
    return persist.MemoryBackend(), persist.MemoryMemo()


def seeded(persist, backends, rows):
    graph = Graph(persist, *backends)
    for key, row in rows.items():
        graph.store.upsert(graph.shifts, key, row)
    return graph


def reopen(persist, graph, backends):
    graph.store.close()
    return Graph(persist, *backends)


def test_an_entry_verifies_after_another_key_changes(persist, backends):
    graph = seeded(persist, backends, {"a": (1,), "b": (2,)})
    assert graph.read(graph.price, "a") == 1
    assert graph.memo.checkpoint()["written"] == 1
    graph.store.upsert(graph.shifts, "b", (3,))
    graph = reopen(persist, graph, backends)
    assert graph.read(graph.price, "a") == 1
    assert graph.calls == []


def test_an_entry_whose_key_changed_runs_once_and_cuts_off_an_equal_value(
    persist, backends
):
    graph = seeded(persist, backends, {"a": (1, "x")})
    assert graph.read(graph.total, "a") == 1
    graph.memo.checkpoint()
    graph.store.upsert(graph.shifts, "a", (1, "y"))
    graph = reopen(persist, graph, backends)
    assert graph.read(graph.total, "a") == 1
    assert graph.calls == [("price", "a")]

    graph.store.upsert(graph.shifts, "a", (5, "y"))
    graph.calls.clear()
    assert graph.read(graph.total, "a") == 5
    assert sorted(graph.calls) == [("price", "a"), ("total", "a")]


def test_an_absent_key_verifies_while_absent_and_runs_once_it_appears(
    persist, backends
):
    graph = seeded(persist, backends, {"a": (1,)})
    assert graph.read(graph.price, "z") is None
    graph.memo.checkpoint()
    graph = reopen(persist, graph, backends)
    assert graph.read(graph.price, "z") is None
    assert graph.calls == []
    graph.memo.checkpoint()

    graph.store.upsert(graph.shifts, "z", (9,))
    graph = reopen(persist, graph, backends)
    assert graph.read(graph.price, "z") == 9
    assert graph.calls == [("price", "z")]


def test_an_entry_whose_key_was_removed_runs_on_reopen(persist, backends):
    graph = seeded(persist, backends, {"a": (1,), "b": (2,)})
    assert graph.read(graph.price, "a") == 1
    graph.memo.checkpoint()
    graph.store.remove(graph.shifts, "a")
    graph = reopen(persist, graph, backends)
    assert graph.read(graph.price, "a") is None
    assert graph.calls == [("price", "a")]


def test_an_unavailable_row_fingerprints_like_any_row(persist, pkg, backends):
    graph = seeded(persist, backends, {"a": pkg.Unavailable("withheld")})
    assert graph.read(graph.status, "a") == "withheld"
    graph.memo.checkpoint()
    graph = reopen(persist, graph, backends)
    assert graph.read(graph.status, "a") == "withheld"
    assert graph.calls == []
    graph.memo.checkpoint()

    graph.store.upsert(graph.shifts, "a", pkg.Unavailable("late"))
    graph = reopen(persist, graph, backends)
    assert graph.read(graph.status, "a") == "late"
    assert graph.calls == [("status", "a")]


def test_entries_reading_only_rows_restore_without_running_after_a_log_checkpoint(
    persist, backends
):
    rows = {k: (k,) for k in range(50)}
    graph = seeded(persist, backends, rows)
    assert [graph.read(graph.total, k) for k in rows] == list(rows)
    graph.memo.checkpoint()
    graph.store.checkpoint()
    graph = reopen(persist, graph, backends)
    assert [graph.read(graph.total, k) for k in rows] == list(rows)
    assert graph.calls == []


def test_a_labelled_memo_tables_entries_are_one_group(persist, backends):
    log, memo_backend = backends
    store = persist.Store.open(log, build_id="b1")
    rt = store.runtime
    double = store.memo_table(
        "double",
        1,
        PickleCodec(),
        PickleCodec(),
        lambda ctx, key: key * 2,
        label="prices",
    )
    total = store.memo_table(
        "total",
        1,
        PickleCodec(),
        PickleCodec(),
        lambda ctx, key: ctx.memo(double, key) + 1,
    )
    memo = store.open_memo(memo_backend)
    assert rt.get(memo.get_or_insert(total, 3)) == 7
    assert rt.get(memo.get_or_insert(double, 4)) == 8
    entry = memo.get_or_insert(double, 3)
    assert rt.get(entry) == 6

    groups = {g["label"]: g for g in rt.graph_summary()["groups"]}
    assert groups["prices"]["nodes"] == 2
    assert rt.graph_snapshot(roots=[entry], depth=0)["nodes"][0]["label"] == "prices/3"
