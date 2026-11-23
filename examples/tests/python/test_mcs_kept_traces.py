import incr
import incr_concurrent
import pytest


@pytest.fixture(params=[incr, incr_concurrent], ids=["local", "shared"])
def mod(request):
    return request.param


def named(rt, node):
    entry = {n["id"]: n for n in rt.graph_snapshot()}[node.id]
    return entry["id"], entry["generation"]


def chain(rt):
    a = rt.create_input(1)
    p = rt.create_query(lambda r: r.get(a) * 3)
    q = rt.create_query(lambda r: r.get(p) + r.get(a) + 10)
    return a, p, q


def entries_by_id(trace):
    return {entry["id"]: entry for entry in trace["node_traces"]}


def test_the_last_trace_equals_the_traced_read_as_a_new_dict(mod):
    rt = mod.Runtime()
    a, p, q = chain(rt)
    _, trace = rt.get_traced(q)

    status, kept = rt.last_trace_by_id(*named(rt, q))
    assert status == "clean"
    assert kept == trace
    assert kept is not trace

    kept["node_traces"].clear()
    again = rt.last_trace_by_id(*named(rt, q))[1]
    assert len(again["node_traces"]) == 2
    assert again["target"] == q.id


def test_a_native_count_node_renders_equal_values():
    rt = incr.Runtime()
    c = rt.create_collection()
    count = c.count()
    rt.get(count)
    c.insert("x")
    c.insert("y")
    _, trace = rt.get_traced(count)

    kept = rt.last_trace_by_id(*named(rt, count))[1]
    assert kept == trace
    assert entries_by_id(kept)[count.id]["value"] == 2
    assert (2, 0) in [e["value"] for e in kept["node_traces"]]


def test_a_label_set_after_the_read_shows_in_the_last_trace():
    rt = incr.Runtime()
    a, p, q = chain(rt)
    rt.set_label(p, "before")
    _, trace = rt.get_traced(q)
    assert entries_by_id(trace)[p.id]["label"] == "before"

    rt.set_label(p, "after")
    rt.set_label(q, "root")
    kept = rt.last_trace_by_id(*named(rt, q))[1]
    assert entries_by_id(kept)[p.id]["label"] == "after"
    assert entries_by_id(kept)[q.id]["label"] == "root"
    assert entries_by_id(kept)[p.id]["value"] == 3


def test_a_step_deleted_and_its_slot_reused_shows_no_label():
    rt = incr.Runtime()
    i = rt.create_input(1)
    a = rt.create_input(1)
    x = rt.create_query(lambda r: r.get(i) * 10)
    rt.set_label(x, "x")
    reads_x = [True]
    q = rt.create_query(lambda r: (r.get(x) if reads_x[0] else 0) + r.get(a))
    rt.get(q)

    rt.set(i, 2)
    reads_x[0] = False
    _, trace = rt.get_traced(q)
    assert entries_by_id(trace)[x.id]["label"] == "x"
    assert entries_by_id(trace)[x.id]["action"] == "recomputed_changed"

    old = x.id
    rt.delete_node(x)
    y = rt.create_input(5)
    assert y.id == old
    rt.set_label(y, "y")

    status, kept = rt.last_trace_by_id(*named(rt, q))
    assert status == "clean"
    entry = entries_by_id(kept)[old]
    assert entry["label"] is None
    assert entry["value"] == 20
    entries_by_id(trace)[old]["label"] = None
    assert kept == trace


def test_memory_stats_count_kept_traces_apart_from_the_graph():
    plain = incr.Runtime()
    traced = incr.Runtime()
    roots = []
    for rt in (plain, traced):
        nodes = [chain(rt) for _ in range(20)]
        roots.append(nodes)
    for _, _, q in roots[0]:
        plain.get(q)
    for _, _, q in roots[1]:
        traced.get_traced(q)

    before = plain.memory_stats()
    after = traced.memory_stats()
    assert after["bytes"] == before["bytes"]
    assert after["py_refs"] == before["py_refs"]
    assert before.get("kept_traces") == {"count": 0, "bytes": 0, "py_refs": 0}
    kept = after["kept_traces"]
    assert kept["count"] == 20
    assert kept["py_refs"] == 40
    assert 0 < kept["bytes"] < 20 * 300

    traced.delete_node(roots[1][0][2])
    fewer = traced.memory_stats()["kept_traces"]
    assert fewer["count"] == 19
    assert fewer["py_refs"] == 38
    assert fewer["bytes"] < kept["bytes"]

    a, _, q = roots[1][1]
    traced.set(a, 5)
    traced.get_traced(q)
    replaced = traced.memory_stats()["kept_traces"]
    assert replaced == fewer
