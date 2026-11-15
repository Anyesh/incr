import incr
import incr_concurrent
import pytest


@pytest.fixture(params=[incr, incr_concurrent], ids=["local", "shared"])
def mod(request):
    return request.param


def entries(rt):
    return {n["id"]: n for n in rt.graph_snapshot()}


def named(rt, node):
    entry = entries(rt)[node.id]
    return entry["id"], entry["generation"]


def build(mod, calls=None):
    rt = mod.Runtime()
    a = rt.create_input(2)

    def double(r):
        if calls is not None:
            calls.append(1)
        return r.get(a) * 2

    q = rt.create_query(double)
    return rt, a, q


def steady(trace):
    return {
        **{k: v for k, v in trace.items() if k not in ("elapsed_ns", "node_traces")},
        "node_traces": [
            {k: v for k, v in e.items() if k != "elapsed_ns"}
            for e in trace["node_traces"]
        ],
    }


def test_snapshot_nodes_carry_a_generation(mod):
    rt, a, q = build(mod)
    rt.get(q)
    for entry in rt.graph_snapshot():
        assert isinstance(entry["generation"], int)
    bounded = rt.graph_snapshot(roots=[q], depth=0)
    assert bounded["nodes"][0]["generation"] == entries(rt)[q.id]["generation"]
    assert "status" not in bounded["nodes"][0]
    assert "value" not in bounded["nodes"][0]


def test_get_by_id_equals_the_handle_read(mod):
    rt, a, q = build(mod)
    assert rt.get_by_id(*named(rt, a)) == rt.get(a) == 2
    assert rt.get_by_id(*named(rt, q)) == rt.get(q) == 4
    rt.set(a, 5)
    assert rt.get_by_id(*named(rt, q)) == rt.get(q) == 10


def test_get_traced_by_id_equals_the_handle_trace(mod):
    by_handle, a1, q1 = build(mod)
    by_id, a2, q2 = build(mod)
    by_handle.get(q1)
    by_id.get(q2)
    by_handle.set(a1, 7)
    by_id.set(a2, 7)

    value, trace = by_handle.get_traced(q1)
    id_value, id_trace = by_id.get_traced_by_id(*named(by_id, q2))

    assert id_value == value == 14
    assert steady(id_trace) == steady(trace)
    assert id_trace["nodes_recomputed"] == 1


def test_a_deleted_id_raises_stale_node_error(mod):
    rt, a, q = build(mod)
    rt.get(q)
    name = named(rt, q)
    rt.delete_node(q)
    with pytest.raises(mod.StaleNodeError):
        rt.get_by_id(*name)
    with pytest.raises(mod.StaleNodeError):
        rt.get_traced_by_id(*name)
    assert issubclass(mod.StaleNodeError, LookupError)


def test_a_reused_slot_raises_stale_node_error_for_the_old_generation(mod):
    rt, a, q = build(mod)
    rt.get(q)
    old = named(rt, q)
    rt.delete_node(q)
    reused = rt.create_input(99)
    assert reused.id == old[0]
    new = named(rt, reused)
    assert new[1] != old[1]
    with pytest.raises(mod.StaleNodeError):
        rt.get_by_id(*old)
    with pytest.raises(mod.StaleNodeError):
        rt.get_traced_by_id(*old)
    assert rt.get_by_id(*new) == 99


def test_an_out_of_range_id_raises_stale_node_error(mod):
    rt, a, q = build(mod)
    for bad in [(10_000, 0), (2**32 - 1, 0)]:
        with pytest.raises(mod.StaleNodeError):
            rt.get_by_id(*bad)
        with pytest.raises(mod.StaleNodeError):
            rt.get_traced_by_id(*bad)
    with pytest.raises(mod.StaleNodeError):
        rt.get_by_id(a.id, named(rt, a)[1] + 1)


def test_a_raising_compute_re_raises_through_get_by_id(mod):
    rt = mod.Runtime()

    def boom(r):
        raise ValueError("boom")

    q = rt.create_query(boom)
    name = named(rt, q)
    with pytest.raises(ValueError, match="boom"):
        rt.get_by_id(*name)


def test_a_cycle_gives_none(mod):
    rt = mod.Runtime()
    holder = {}
    q = rt.create_query(lambda r: r.get(holder["q"]) + 1)
    holder["q"] = q
    assert rt.get_by_id(*named(rt, q)) is None


def test_values_gives_clean_values_and_never_recomputes(mod):
    calls = []
    rt, a, q = build(mod, calls)
    rt.get(q)
    assert len(calls) == 1

    snap = {n["id"]: n for n in rt.graph_snapshot(values=True)}

    assert snap[a.id]["status"] == "clean" or snap[a.id]["status"] == "new"
    assert snap[a.id]["value"] == 2
    assert snap[q.id]["status"] == "clean"
    assert snap[q.id]["value"] == 4
    assert len(calls) == 1


def test_values_leaves_a_dirty_node_with_a_status_and_no_value(mod):
    calls = []
    rt, a, q = build(mod, calls)
    rt.get(q)
    rt.set(a, 3)

    snap = {n["id"]: n for n in rt.graph_snapshot(values=True)}

    assert snap[q.id]["status"] == "dirty"
    assert "value" not in snap[q.id]
    assert snap[a.id]["value"] == 3
    assert len(calls) == 1


def test_values_with_roots_and_limit_reads_only_the_returned_nodes(mod):
    rt = mod.Runtime()
    nodes = [rt.create_input(1)]
    for _ in range(5):
        prev = nodes[-1]
        q = rt.create_query(lambda r, p=prev: r.get(p) + 1)
        rt.get(q)
        nodes.append(q)

    snap = rt.graph_snapshot(roots=[nodes[5]], limit=3, values=True)

    assert [n["id"] for n in snap["nodes"]] == [n.id for n in nodes[5:2:-1]]
    assert snap["truncated"] is True
    assert [n["value"] for n in snap["nodes"]] == [6, 5, 4]
    assert all(n["status"] == "clean" for n in snap["nodes"])


def test_values_reads_a_node_by_the_name_the_snapshot_gave(mod):
    rt, a, q = build(mod)
    rt.get(q)
    for entry in rt.graph_snapshot(values=True):
        assert rt.get_by_id(entry["id"], entry["generation"]) == entry["value"]
