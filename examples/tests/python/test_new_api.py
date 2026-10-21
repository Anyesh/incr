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


@BOTH_RUNTIMES
def test_cycle_raises_cycle_error_with_participants(make_rt):
    rt = make_rt()
    module = incr if make_rt is incr.Runtime else incr_concurrent
    holder = {}
    q = rt.create_query(lambda r: r.get(holder["q"]) + 1)
    holder["q"] = q
    with pytest.raises(module.CycleError) as raised:
        rt.get(q)
    assert isinstance(raised.value, RuntimeError)
    message, ids = raised.value.args
    assert "dependency cycle" in message
    assert ids == [q.id]

    # A later, unrelated error is not labelled with the old cycle.
    a = rt.create_input(0)
    bad = rt.create_query(lambda r: 10 // r.get(a))
    with pytest.raises(ZeroDivisionError):
        rt.get(bad)


@BOTH_RUNTIMES
def test_cycle_error_caught_in_a_query_is_an_ordinary_result(make_rt):
    rt = make_rt()
    module = incr if make_rt is incr.Runtime else incr_concurrent
    holder = {}

    def reads_itself(r):
        try:
            return r.get(holder["q"]) + 1
        except module.CycleError:
            return -1

    q = rt.create_query(reads_itself)
    holder["q"] = q
    # Caught, the read is what `try_get` returns in Rust: `q` still reads
    # itself, so it lands Cyclic, and a top-level read raises.
    with pytest.raises(module.CycleError):
        rt.get(q)

    a = rt.create_input(0)
    bad = rt.create_query(lambda r: 10 // r.get(a))
    with pytest.raises(ZeroDivisionError):
        rt.get(bad)


@BOTH_RUNTIMES
def test_engine_unwind_crosses_a_closure_as_a_base_exception(make_rt):
    rt = make_rt()
    module = incr if make_rt is incr.Runtime else incr_concurrent
    holder = {}
    seen = []

    def reads_itself(r):
        try:
            return r.get(holder["q"]) + 1
        except Exception as e:
            seen.append(("exception", type(e).__name__))
            raise
        except BaseException as e:
            seen.append(("base", type(e).__name__))
            raise

    q = rt.create_query(reads_itself)
    holder["q"] = q
    with pytest.raises(module.CycleError):
        rt.get(q)
    assert seen == [("base", "_IncrEngineUnwind")]


@BOTH_RUNTIMES
@pytest.mark.parametrize("cleanup_read", ["get", "try_get"])
@pytest.mark.parametrize("top_read", ["get", "try_get"])
def test_clean_read_in_finally_keeps_the_cycle_error(make_rt, cleanup_read, top_read):
    rt = make_rt()
    module = incr if make_rt is incr.Runtime else incr_concurrent
    a = rt.create_input(1)
    assert rt.get(a) == 1
    holder = {}

    def reads_itself(r):
        try:
            return r.get(holder["q"]) + 1
        finally:
            getattr(r, cleanup_read)(a)

    q = rt.create_query(reads_itself)
    holder["q"] = q
    with pytest.raises(module.CycleError) as raised:
        getattr(rt, top_read)(q)
    _, ids = raised.value.args
    assert ids == [q.id]


@BOTH_RUNTIMES
@pytest.mark.parametrize("raises", [False, True], ids=["returns", "raises"])
@pytest.mark.parametrize("cleanup_read", ["get", "try_get"])
@pytest.mark.parametrize("top_read", ["get", "try_get"])
def test_compute_in_finally_keeps_its_outcome_and_the_cycle(
    make_rt, raises, cleanup_read, top_read
):
    rt = make_rt()
    module = incr if make_rt is incr.Runtime else incr_concurrent
    holder = {}
    runs = []
    seen = []

    def body(r):
        runs.append("b")
        if raises:
            raise ValueError("b failed")
        return 7

    b = rt.create_query(body)

    def reads_itself(r):
        try:
            return r.get(holder["q"]) + 1
        finally:
            try:
                seen.append(getattr(r, cleanup_read)(b))
            except BaseException as e:
                seen.append(e)
                raise

    q = rt.create_query(reads_itself)
    holder["q"] = q
    with pytest.raises(module.CycleError) as raised:
        getattr(rt, top_read)(q)
    assert raised.value.args[1] == [q.id]
    assert runs == ["b"]
    if raises:
        assert len(seen) == 1
        assert type(seen[0]) is ValueError
        assert str(seen[0]) == "b failed"
        # `b` landed Failed on its own error, outside the cycle.
        for read in (rt.get, rt.try_get):
            with pytest.raises(RuntimeError, match="last compute panicked") as failed:
                read(b)
            assert not isinstance(failed.value, module.CycleError)
    else:
        assert seen == [7]
        assert rt.get(b) == 7
        assert rt.try_get(b) == 7
    assert runs == ["b"]
    with pytest.raises(module.CycleError) as again:
        rt.get(q)
    assert again.value.args[1] == [q.id]


@BOTH_RUNTIMES
def test_python_exception_is_raised_as_itself_with_no_panic_message(make_rt, capfd):
    rt = make_rt()
    a = rt.create_input(0)
    bad = rt.create_query(lambda r: 10 // r.get(a))
    outer = rt.create_query(lambda r: r.get(bad) + 1)
    with pytest.raises(ZeroDivisionError):
        rt.get(outer)
    assert capfd.readouterr().err == ""


@BOTH_RUNTIMES
def test_runtime_ref_kept_by_a_raising_query_is_invalidated(make_rt):
    rt = make_rt()
    a = rt.create_input(1)
    kept = []

    def body(r):
        kept.append(r)
        raise ValueError("boom")

    q = rt.create_query(body)
    with pytest.raises(ValueError):
        rt.get(q)
    with pytest.raises(RuntimeError, match="no longer valid"):
        kept[0].get(a)
