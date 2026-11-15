"""R14: `kc.forget_key(key)` frees the version input of a key whose readers
are gone, and `kc.version_input_count()` counts a collection's inputs.
Shared never frees one, so `forget_key` returns False there. Every test
takes the package's `persist` module from the `persist` fixture."""

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


@pytest.fixture
def local(persist):
    if persist is not incr.persist:
        pytest.skip("Local only")


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


def reader(rt, kc, key):
    return rt.create_query(lambda ctx: ctx.get_key(kc, key))


def freed(pkg):
    return pkg is incr


def test_forget_key_frees_the_input_of_a_never_written_key(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    assert kc.version_input_count() == 0
    node = reader(rt, kc, "z")
    assert rt.get(node) is None
    assert kc.version_input_count() == 1

    assert kc.forget_key("z") is False
    assert kc.version_input_count() == 1

    rt.delete_node(node)
    assert kc.forget_key("z") is freed(pkg)
    assert kc.version_input_count() == (0 if freed(pkg) else 1)
    assert kc.forget_key("z") is False


def test_a_live_reader_keeps_its_input_and_is_invalidated_by_an_insert(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    node = reader(rt, kc, "z")
    assert rt.get(node) is None
    assert kc.forget_key("z") is False
    kc.upsert("z", 5)
    assert rt.get(node) == 5


def test_a_held_key_keeps_its_input(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("a", 1)
    node = reader(rt, kc, "a")
    rt.get(node)
    rt.delete_node(node)
    assert kc.forget_key("a") is False
    assert kc.version_input_count() == 1
    kc.remove("a")
    assert kc.version_input_count() == (0 if freed(pkg) else 1)


def test_forget_key_of_an_unread_key_is_false(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    assert kc.forget_key("nobody") is False


def test_forget_key_refuses_an_overlay(local):
    rt = incr.Runtime()
    kc = incr.KeyedCollection(rt)
    node = reader(rt, kc, "z")
    rt.get(node)
    rt.delete_node(node)
    with rt.overlay():
        with pytest.raises(RuntimeError, match="overlay"):
            kc.forget_key("z")
    assert kc.version_input_count() == 1
    assert kc.forget_key("z") is True


def test_a_persisted_collection_forgets_a_key_like_a_plain_one(persist):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    kc = store.keyed_collection("shifts", PickleCodec(), PickleCodec())
    rt = store.runtime
    node = reader(rt, kc, "z")
    assert rt.get(node) is None
    assert kc.version_input_count() == 1
    assert kc.forget_key("z") is False
    rt.delete_node(node)
    assert kc.forget_key("z") is (persist is incr.persist)
    store.upsert(kc, "z", 3)
    assert rt.get_key(kc, "z") == 3


class Graph:
    def __init__(self, persist, log, memo_backend):
        self.store = persist.Store.open(log, build_id="b1")
        self.shifts = self.store.keyed_collection(
            "shifts", PickleCodec(), PickleCodec()
        )
        self.calls = []

        def price(ctx, key):
            self.calls.append(key)
            row = ctx.get_key(self.shifts, key)
            return None if row is None else row[0]

        self.price = self.store.memo_table(
            "price", 1, PickleCodec(), PickleCodec(), price
        )
        self.memo = self.store.open_memo(memo_backend)


@pytest.mark.parametrize("key", ["a", "z"], ids=["held", "absent"])
def test_a_restored_entry_not_yet_run_survives_forget_key(persist, key):
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    graph = Graph(persist, log, memo_backend)
    graph.store.upsert(graph.shifts, "a", (1,))
    rt = graph.store.runtime
    expected = 1 if key == "a" else None
    assert rt.get(graph.memo.get_or_insert(graph.price, key)) == expected
    graph.memo.checkpoint()
    graph.store.close()

    graph = Graph(persist, log, memo_backend)
    node = graph.memo.get_or_insert(graph.price, key)
    graph.shifts.forget_key(key)
    rt = graph.store.runtime
    assert rt.get(node) == expected
    assert graph.calls == []

    graph.store.upsert(graph.shifts, key, (7,))
    assert rt.get(node) == 7
    assert graph.calls == [key]


def test_forget_key_from_inside_a_query_raises_and_leaves_the_input(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    node = reader(rt, kc, "z")
    rt.get(node)
    rt.delete_node(node)
    q = rt.create_query(lambda ctx: kc.forget_key("z"))
    with pytest.raises(RuntimeError, match="inside a compute"):
        rt.get(q)
    assert kc.version_input_count() == 1
