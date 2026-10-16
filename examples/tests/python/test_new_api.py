import incr
import incr_concurrent
import pytest
from incr import Runtime

BOTH_RUNTIMES = pytest.mark.parametrize(
    "make_rt", [incr.Runtime, incr_concurrent.Runtime], ids=["local", "concurrent"]
)


def test_query_exception_propagates_as_original():
    rt = Runtime()
    a = rt.create_input(0)
    q = rt.create_query(lambda r: 10 // r.get(a))

    with pytest.raises(ZeroDivisionError):
        rt.get(q)

    # Engine recovers once the input makes the query computable again.
    rt.set(a, 5)
    assert rt.get(q) == 2


def test_filter_predicate_exception_propagates():
    rt = Runtime()
    col = rt.create_collection()
    bad = col.filter(lambda x: x.missing_attribute)
    count = bad.count()
    col.insert(1)
    with pytest.raises(AttributeError):
        rt.get(count)


def test_delete_node_then_stale_handle_raises():
    rt = Runtime()
    a = rt.create_input(1)
    rt.delete_node(a)
    with pytest.raises(RuntimeError, match="stale"):
        rt.get(a)


def test_delete_node_with_dependents_raises():
    rt = Runtime()
    a = rt.create_input(1)
    q = rt.create_query(lambda r: r.get(a) + 1)
    assert rt.get(q) == 2
    with pytest.raises(RuntimeError, match="dependents"):
        rt.delete_node(a)


def test_observe_and_stabilize():
    rt = Runtime()
    a = rt.create_input(1)
    doubled = rt.create_query(lambda r: r.get(a) * 2)

    seen = []
    obs = rt.observe(doubled, seen.append)

    rt.stabilize()
    assert seen == [2]

    rt.stabilize()
    assert seen == [2]  # no change, no fire

    rt.set(a, 5)
    rt.stabilize()
    assert seen == [2, 10]

    rt.unobserve(obs)
    rt.set(a, 7)
    rt.stabilize()
    assert seen == [2, 10]


def test_aggregate_sum():
    rt = Runtime()
    col = rt.create_collection()
    total = col.aggregate(0, lambda x: x, lambda a, b: a + b)
    assert rt.get(total) == 0
    for i in range(1, 11):
        col.insert(i)
    assert rt.get(total) == 55
    col.delete(10)
    assert rt.get(total) == 45


def test_insert_on_derived_collection_raises():
    rt = Runtime()
    col = rt.create_collection()
    evens = col.filter(lambda x: x % 2 == 0)
    with pytest.raises(RuntimeError, match="derived"):
        evens.insert(2)


def test_cycle_detection_raises():
    rt = Runtime()
    holder = {}
    q = rt.create_query(lambda r: r.get(holder["q"]) + 1)
    holder["q"] = q
    with pytest.raises(RuntimeError, match="cycle"):
        rt.get(q)


@BOTH_RUNTIMES
def test_is_alive_tracks_delete_and_foreign_runtimes(make_rt):
    rt = make_rt()
    a = rt.create_input(1)
    assert rt.is_alive(a)
    assert not make_rt().is_alive(a)
    rt.delete_node(a)
    assert not rt.is_alive(a)


@BOTH_RUNTIMES
def test_delete_with_dependents_removes_the_subgraph(make_rt):
    rt = make_rt()
    x = rt.create_input(1)
    left = rt.create_query(lambda r: r.get(x) + 1)
    right = rt.create_query(lambda r: r.get(x) + 2)
    both = rt.create_query(lambda r: r.get(left) + r.get(right))
    other = rt.create_input(10)
    assert rt.get(both) == 5

    assert rt.delete_with_dependents(x) == 4
    assert not any(rt.is_alive(n) for n in (x, left, right, both))
    assert rt.is_alive(other) and rt.get(other) == 10
    with pytest.raises(RuntimeError, match="stale"):
        rt.get(both)
    assert rt.delete_with_dependents(x) == 0


@BOTH_RUNTIMES
def test_delete_with_dependents_inside_a_query_raises(make_rt):
    rt = make_rt()
    x = rt.create_input(1)
    q = rt.create_query(lambda r: rt.delete_with_dependents(x))
    with pytest.raises(RuntimeError, match="during compute"):
        rt.get(q)
    assert rt.is_alive(x)
