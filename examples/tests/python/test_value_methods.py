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


class Ambiguous:
    def __bool__(self):
        raise ValueError("predicate result __bool__ raised")


@BOTH_RUNTIMES
def test_raising_predicate_result_bool_propagates_out_of_the_read(make_rt):
    rt = make_rt()
    col = rt.create_collection()
    kept = col.filter(lambda x: Ambiguous())
    count = kept.count()
    col.insert(1)
    with pytest.raises(ValueError, match="__bool__ raised"):
        rt.get(count)


class AmbiguousTruth:
    def __init__(self, exc):
        self.exc = exc

    def __bool__(self):
        raise self.exc


class ArrayLike:
    def __init__(self, v, exc=ValueError):
        self.v = v
        self.exc = exc

    def __eq__(self, other):
        return AmbiguousTruth(self.exc("truth value is ambiguous"))

    __hash__ = None


@BOTH_RUNTIMES
def test_ambiguous_eq_truth_counts_as_not_equal_on_set(make_rt):
    rt = make_rt()
    a = rt.create_input(ArrayLike(1))
    q = rt.create_query(lambda r: r.get(a).v)
    assert rt.get(q) == 1
    rt.set(a, ArrayLike(2))
    assert rt.get(q) == 2


@BOTH_RUNTIMES
def test_ambiguous_eq_truth_counts_as_not_equal_from_a_compute(make_rt):
    rt = make_rt()
    a = rt.create_input(1)
    arr = rt.create_query(lambda r: ArrayLike(r.get(a)))
    q = rt.create_query(lambda r: r.get(arr).v)
    assert rt.get(q) == 1
    rt.set(a, 2)
    assert rt.get(q) == 2


@BOTH_RUNTIMES
def test_eq_truth_raising_other_errors_propagates_out_of_set(make_rt):
    rt = make_rt()
    a = rt.create_input(ArrayLike(1, TypeError))
    with pytest.raises(TypeError, match="ambiguous"):
        rt.set(a, ArrayLike(2, TypeError))


@BOTH_RUNTIMES
def test_eq_itself_raising_value_error_still_propagates_out_of_set(make_rt):
    class Raises:
        def __eq__(self, other):
            raise ValueError("eq itself raised")

        __hash__ = None

    rt = make_rt()
    a = rt.create_input(Raises())
    with pytest.raises(ValueError, match="eq itself raised"):
        rt.set(a, Raises())
