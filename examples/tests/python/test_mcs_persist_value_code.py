"""Value methods (`__eq__`, `__hash__`) that call back into incr: a store or
memo call gets a RuntimeError, a runtime call that would compute gets the
contract RuntimeError, and neither prints a Rust panic line on stderr."""

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


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class Probe:
    """Equal by `n` and hashed by `n`; `hook` runs inside the next `method`
    call of any probe, once."""

    hook = None
    method = None

    def __init__(self, n):
        self.n = n

    @classmethod
    def arm(cls, method, hook):
        cls.method = method
        cls.hook = hook

    @classmethod
    def fire(cls, method):
        if cls.method == method and cls.hook is not None:
            hook, cls.hook = cls.hook, None
            hook()

    def __hash__(self):
        Probe.fire("hash")
        return hash(self.n)

    def __eq__(self, other):
        Probe.fire("eq")
        return isinstance(other, Probe) and other.n == self.n

    def __reduce__(self):
        return (Probe, (self.n,))


@pytest.fixture(autouse=True)
def disarm():
    yield
    Probe.hook = None
    Probe.method = None


def assert_quiet(capfd):
    out, err = capfd.readouterr()
    assert err == "", err


def declare(store):
    codec = PickleCodec()
    return store.keyed_collection("rows", codec, codec)


def test_a_keyed_row_eq_that_calls_the_store_raises_and_prints_nothing(persist, capfd):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    kc = declare(store)
    store.upsert(kc, "a", Probe(1))
    capfd.readouterr()
    Probe.arm("eq", store.stats)
    with pytest.raises(RuntimeError):
        store.upsert(kc, "a", Probe(1))
    assert_quiet(capfd)
    store.upsert(kc, "b", Probe(2))
    assert kc.get("b").n == 2


@pytest.mark.parametrize("method", ["eq", "hash"])
def test_a_non_clean_runtime_get_from_value_code_raises_and_prints_nothing(
    persist, capfd, method
):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    rt = store.runtime
    base = store.input("base", 1, PickleCodec())
    doubled = rt.create_query(lambda r: r.get(base) * 2)
    rows = store.collection("rows", PickleCodec())
    store.insert(rows, Probe(1))
    assert rt.get(doubled) == 2
    store.set(base, 5)
    capfd.readouterr()
    Probe.arm(method, lambda: rt.get(doubled))
    with pytest.raises(RuntimeError, match="must not call into the runtime"):
        store.insert(rows, Probe(1))
    assert_quiet(capfd)
    assert rt.get(doubled) == 10


@pytest.mark.parametrize("method", ["eq", "hash"])
def test_a_memo_checkpoint_from_value_code_raises_and_prints_nothing(
    persist, capfd, method
):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    memo = store.open_memo(persist.MemoryMemo())
    rows = store.collection("rows", PickleCodec())
    store.insert(rows, Probe(1))
    capfd.readouterr()
    Probe.arm(method, memo.checkpoint)
    with pytest.raises(RuntimeError):
        store.insert(rows, Probe(1))
    assert_quiet(capfd)


@pytest.mark.parametrize("method", ["keys", "get"])
def test_a_keyed_read_from_a_key_method_raises_and_prints_nothing(
    persist, capfd, method
):
    pkg = importlib.import_module(persist.__name__.rpartition(".")[0])
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert("held", 0)
    capfd.readouterr()

    class Key:
        def __hash__(self):
            kc.keys() if method == "keys" else kc.get("held")
            return 1

        def __eq__(self, other):
            return self is other

    with pytest.raises(RuntimeError, match="key or row method"):
        kc.upsert(Key(), 1)
    assert_quiet(capfd)
    assert kc.items() == [("held", 0)]
