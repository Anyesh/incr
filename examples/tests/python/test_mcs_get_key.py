"""R1: `ctx.get_key(kc, key)`, `ctx.get_key_status(kc, key)` and
`rt.get_key(kc, key)` read one key of a keyed collection, inside a query
with a dependency on that key alone. Every test takes the package's
`persist` module from the `persist` fixture, and the package itself from
it, so a Shared run can add `incr_concurrent.persist` to `PACKAGES`."""

import importlib
import pickle
import time

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
def with_overlay(persist):
    """Skips Shared, whose `rt.overlay()` raises: it has no isolation for
    an overlay's candidate values."""
    if persist is not incr.persist:
        pytest.skip("rt.overlay() raises on Shared")


@pytest.fixture
def pkg(persist):
    return importlib.import_module(persist.__name__.rpartition(".")[0])


# Decodes only bytes this test's own store wrote.
class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class Reader:
    def __init__(self, rt, kc, key):
        self.runs = 0

        def read(ctx):
            self.runs += 1
            return ctx.get_key(kc, key)

        self.node = rt.create_query(read)
        self.rt = rt

    def value(self):
        return self.rt.get(self.node)


def open_store(persist, backend):
    return persist.Store.open(backend, build_id=1)


def declare(store, label=None):
    return store.keyed_collection("shifts", PickleCodec(), PickleCodec(), label=label)


def test_a_reader_reruns_only_for_its_own_key(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a", 1)
    kc.upsert("b", 2)
    ra, rb = Reader(rt, kc, "a"), Reader(rt, kc, "b")
    assert (ra.value(), rb.value()) == (1, 2)
    kc.upsert("a", 10)
    assert (ra.value(), rb.value()) == (10, 2)
    assert (ra.runs, rb.runs) == (2, 1)
    kc.upsert("a", 10)
    assert (ra.value(), rb.value()) == (10, 2)
    assert (ra.runs, rb.runs) == (2, 1)


def test_an_absent_key_reads_none_until_upserted_and_again_after_remove(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    reader = Reader(rt, kc, "z")
    assert reader.value() is None
    kc.upsert("z", 5)
    assert reader.value() == 5
    kc.remove("z")
    assert reader.value() is None
    assert reader.runs == 3


def test_an_unavailable_row_reaches_the_reader_as_its_status(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a", pkg.Unavailable("withheld"))
    status_query = rt.create_query(lambda ctx: ctx.get_key_status(kc, "a"))
    row, status = rt.get(status_query)
    assert isinstance(status, pkg.Unavailable)
    assert status.reason == "withheld"
    assert row == status
    value_query = rt.create_query(lambda ctx: ctx.get_key(kc, "a"))
    assert isinstance(rt.get(value_query), pkg.Unavailable)
    kc.upsert("a", 3)
    assert rt.get(status_query) == (3, None)
    assert rt.get(value_query) == 3


def test_get_key_status_of_an_absent_key_is_none_and_ready(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    query = rt.create_query(lambda ctx: ctx.get_key_status(kc, "a"))
    assert rt.get(query) == (None, None)


def test_rt_get_key_reads_outside_a_query(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a", 1)
    assert rt.get_key(kc, "a") == 1
    assert rt.get_key(kc, "b") is None


def test_kc_get_inside_a_query_registers_no_dependency(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a", 1)
    query = rt.create_query(lambda ctx: kc.get("a"))
    assert rt.get(query) == 1
    kc.upsert("a", 2)
    assert rt.get(query) == 1
    assert "ctx.get_key" in pkg.KeyedCollection.get.__doc__


def test_get_key_refuses_a_collection_and_another_runtimes_keyed_collection(pkg):
    rt, other = pkg.Runtime(), pkg.Runtime()
    rows = rt.create_collection()
    foreign = pkg.KeyedCollection(other)
    with pytest.raises(TypeError):
        rt.get(rt.create_query(lambda ctx: ctx.get_key(rows, "a")))
    with pytest.raises(TypeError):
        rt.get_key(rows, "a")
    with pytest.raises(ValueError, match="runtime"):
        rt.get(rt.create_query(lambda ctx: ctx.get_key(foreign, "a")))
    with pytest.raises(ValueError, match="runtime"):
        rt.get_key(foreign, "a")


def test_a_labelled_collection_labels_each_read_key(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt, label="shifts")
    kc.upsert("a", 1)
    reader = Reader(rt, kc, "a")
    reader.value()
    snapshot = rt.graph_snapshot(roots=[reader.node], depth=1, label_prefix="shifts/")
    assert [n["label"] for n in snapshot["nodes"]] == ["shifts/a"]


def test_a_persisted_keyed_collection_after_reopen(persist):
    backend = persist.MemoryBackend()
    store = open_store(persist, backend)
    kc = declare(store)
    store.upsert(kc, "a", 1)
    store.upsert(kc, "b", 2)
    store.close()

    again = open_store(persist, backend)
    kc = declare(again)
    rt = again.runtime
    ra, rb, rz = Reader(rt, kc, "a"), Reader(rt, kc, "b"), Reader(rt, kc, "z")
    assert (ra.value(), rb.value(), rz.value()) == (1, 2, None)
    again.upsert(kc, "a", 11)
    again.upsert(kc, "b", 2)
    assert (ra.value(), rb.value(), rz.value()) == (11, 2, None)
    assert (ra.runs, rb.runs, rz.runs) == (2, 1, 1)
    again.upsert(kc, "z", 7)
    assert rz.value() == 7
    again.remove(kc, "z")
    assert rz.value() is None
    with again.batch() as b:
        b.upsert(kc, "b", 3)
    assert (ra.value(), rb.value()) == (11, 3)
    assert (ra.runs, rb.runs) == (2, 2)
    assert rt.get_key(kc, "a") == 11


def test_a_reader_sees_an_overlay_upsert_and_nothing_is_logged(with_overlay, persist):
    store = open_store(persist, persist.MemoryBackend())
    kc = declare(store)
    store.upsert(kc, "a", 1)
    rt = store.runtime
    reader = Reader(rt, kc, "a")
    assert reader.value() == 1
    position = store.stats()["position"]
    with rt.overlay():
        kc.upsert("a", 2)
        assert reader.value() == 2
    assert reader.value() == 1
    assert store.stats()["position"] == position


def test_a_plain_reader_sees_an_overlay_upsert_and_its_undo(with_overlay, pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a", 1)
    reader = Reader(rt, kc, "a")
    assert reader.value() == 1
    with rt.overlay():
        kc.upsert("a", 2)
        assert reader.value() == 2
    assert reader.value() == 1


def test_published_240k_keys_one_upsert_reruns_one_reader(pkg, capsys):
    keys = 240_000
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    runs = [0]

    def reader(key):
        def read(ctx):
            runs[0] += 1
            return ctx.get_key(kc, key)

        return read

    started = time.perf_counter()
    kc.reconcile({k: k for k in range(keys)})
    reconciled = time.perf_counter()
    queries = [rt.create_query(reader(k)) for k in range(keys)]
    for q in queries:
        rt.get(q)
    first_read = time.perf_counter()
    assert runs[0] == keys
    kc.upsert(17, -1)
    upserted = time.perf_counter()
    for q in queries:
        rt.get(q)
    reread = time.perf_counter()
    assert runs[0] == keys + 1
    assert rt.get(queries[17]) == -1
    with capsys.disabled():
        print(
            f"\nR1 {keys} keys: reconcile {reconciled - started:.3f}s, "
            f"create and first read of {keys} readers {first_read - reconciled:.3f}s, "
            f"one upsert {(upserted - first_read) * 1e6:.0f}us, "
            f"reread of every reader {reread - upserted:.3f}s"
        )
