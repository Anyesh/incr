import incr
import incr_concurrent
import pytest

BOTH_RUNTIMES = pytest.mark.parametrize(
    "make_rt", [incr.Runtime, incr_concurrent.Runtime], ids=["local", "concurrent"]
)


class Element:
    def __init__(self, v, on_eq):
        self.v = v
        self.on_eq = on_eq
        self.armed = False

    def __hash__(self):
        return hash(self.v)

    def __eq__(self, other):
        if self.armed or getattr(other, "armed", False):
            self.on_eq()
        return isinstance(other, Element) and self.v == other.v


def raise_value_error():
    raise ValueError("element __eq__ raised")


def assert_unchanged(rt, col, count, expected):
    assert col.snapshot_len() == expected
    assert rt.get(count) == expected


@BOTH_RUNTIMES
def test_raising_element_eq_propagates_out_of_insert(make_rt):
    rt = make_rt()
    col = rt.create_collection()
    count = col.count()
    col.insert(Element(1, raise_value_error))
    assert_unchanged(rt, col, count, 1)

    probe = Element(1, raise_value_error)
    probe.armed = True
    with pytest.raises(ValueError, match="element __eq__ raised"):
        col.insert(probe)
    assert_unchanged(rt, col, count, 1)

    col.insert(2)
    assert_unchanged(rt, col, count, 2)


@BOTH_RUNTIMES
def test_element_eq_calling_into_incr_raises_the_contract_error(make_rt):
    rt = make_rt()
    a = rt.create_input(5)
    col = rt.create_collection()
    count = col.count()
    col.insert(Element(1, lambda: rt.get(a)))
    assert_unchanged(rt, col, count, 1)

    probe = Element(1, lambda: rt.get(a))
    probe.armed = True
    with pytest.raises(RuntimeError, match="must not call into the runtime"):
        col.insert(probe)
    assert_unchanged(rt, col, count, 1)
    assert rt.get(a) == 5


@BOTH_RUNTIMES
def test_raising_element_hash_propagates_out_of_insert(make_rt):
    rt = make_rt()
    col = rt.create_collection()
    count = col.count()
    with pytest.raises(TypeError, match="unhashable"):
        col.insert([1, 2])
    assert_unchanged(rt, col, count, 0)


class Key:
    def __init__(self, v):
        self.v = v

    def __eq__(self, other):
        return self.v == other.v

    def __lt__(self, other):
        raise ValueError("key __lt__ raised")


@BOTH_RUNTIMES
def test_raising_sort_key_lt_propagates_out_of_the_read(make_rt):
    rt = make_rt()
    col = rt.create_collection()
    ordered = col.sort_by_key(Key)
    version = ordered.version_node
    rt.get(version)
    col.insert(1)
    col.insert(2)
    with pytest.raises(ValueError, match="key __lt__ raised"):
        rt.get(version)


@BOTH_RUNTIMES
def test_raising_group_key_hash_propagates_out_of_get_group(make_rt):
    rt = make_rt()
    col = rt.create_collection()
    grouped = col.group_by(lambda x: x % 2)
    col.insert(1)
    rt.get(grouped.version_node)
    with pytest.raises(TypeError, match="unhashable"):
        grouped.get_group([1])
    assert grouped.group_count() == 1
