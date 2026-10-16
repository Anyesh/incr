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


def _versioned_wrappers(rt):
    col = rt.create_collection()
    col.insert(1)
    col.insert(2)
    grouped = col.group_by(lambda x: x % 2)
    rt.get(grouped.version_node)
    return {
        "collection": lambda: col,
        "grouped": lambda: grouped,
        "sorted": lambda: col.sort_by_key(lambda x: x),
        "group": lambda: grouped.get_group(1),
    }


@BOTH_RUNTIMES
@pytest.mark.parametrize("kind", ["collection", "grouped", "sorted", "group"])
def test_version_node_is_one_node_per_collection(make_rt, kind):
    rt = make_rt()
    wrapper = _versioned_wrappers(rt)[kind]
    target = wrapper()
    first = target.version_node
    rt.get(first)
    nodes = rt.node_count()

    for _ in range(5):
        again = wrapper().version_node if kind == "group" else target.version_node
        assert again == first
        assert hash(again) == hash(first)
        rt.get(again)
    assert rt.node_count() == nodes


@BOTH_RUNTIMES
def test_deleted_version_node_goes_stale_and_is_recreated(make_rt):
    rt = make_rt()
    col = rt.create_collection()
    col.insert(1)
    old = col.version_node
    before = rt.get(old)
    rt.delete_node(old)
    with pytest.raises(RuntimeError, match="stale"):
        rt.get(old)

    fresh = col.version_node
    assert fresh != old
    assert rt.is_alive(fresh)
    assert rt.get(fresh) == before
    assert col.version_node == fresh


@BOTH_RUNTIMES
def test_node_ids_compare_by_node(make_rt):
    rt = make_rt()
    a = rt.create_input(1)
    b = rt.create_input(1)
    assert a == a and a != b
    assert a != 1
    assert len({a, b, a}) == 2
