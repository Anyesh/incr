import incr
import incr_concurrent
import pytest


@pytest.fixture(params=[incr, incr_concurrent], ids=["local", "shared"])
def mod(request):
    return request.param


def named(rt, node):
    entry = {n["id"]: n for n in rt.graph_snapshot()}[node.id]
    return entry["id"], entry["generation"]


def build(mod):
    calls = []
    rt = mod.Runtime()
    a = rt.create_input(1)

    def parity(r):
        calls.append("parity")
        return r.get(a) % 2

    p = rt.create_query(parity)

    def total(r):
        calls.append("total")
        return r.get(p) + 10

    q = rt.create_query(total)
    return rt, a, q, calls


def test_a_dirty_node_reads_dirty_without_a_compute(mod):
    rt, a, q, calls = build(mod)
    rt.get_traced(q)
    rt.set(a, 2)
    before = list(calls)

    assert rt.peek_by_id(*named(rt, q)) == ("dirty", None)
    assert rt.last_trace_by_id(*named(rt, q)) == ("dirty", None)
    assert calls == before


def test_a_new_node_reads_new_without_a_compute(mod):
    rt, a, q, calls = build(mod)
    assert rt.peek_by_id(*named(rt, q)) == ("new", None)
    assert rt.last_trace_by_id(*named(rt, q)) == ("new", None)
    assert calls == []


def test_peek_gives_a_clean_value_as_stored(mod):
    rt, a, q, calls = build(mod)
    rt.get(q)
    assert rt.peek_by_id(*named(rt, q)) == ("clean", 11)
    status, value = rt.peek_by_id(*named(rt, a))
    assert value == 1
    assert status == {n["id"]: n for n in rt.graph_snapshot(values=True)}[a.id]["status"]
    assert calls == ["total", "parity"]


def test_a_clean_node_gives_the_trace_that_produced_its_value(mod):
    rt, a, q, calls = build(mod)
    value, trace = rt.get_traced(q)
    assert value == 11
    status, kept = rt.last_trace_by_id(*named(rt, q))
    assert status == "clean"
    assert kept is trace

    rt.set(a, 2)
    value, trace = rt.get_traced_by_id(*named(rt, q))
    assert value == 10
    assert rt.last_trace_by_id(*named(rt, q))[1] is trace


def test_a_traced_verify_keeps_the_trace_and_a_plain_one_hides_it(mod):
    rt, a, q, calls = build(mod)
    _, first = rt.get_traced(q)
    rt.set(a, 3)
    _, verified = rt.get_traced(q)
    assert [e["action"] for e in verified["node_traces"] if e["id"] == q.id] == [
        "verified_clean"
    ]
    assert rt.last_trace_by_id(*named(rt, q)) == ("clean", first)
    assert rt.last_trace_by_id(*named(rt, q))[1] is first

    rt.set(a, 5)
    assert rt.get(q) == 11
    assert rt.last_trace_by_id(*named(rt, q)) == ("clean", None)


def test_a_node_never_traced_has_no_trace(mod):
    rt, a, q, calls = build(mod)
    rt.get(q)
    assert rt.last_trace_by_id(*named(rt, q)) == ("clean", None)


def test_stale_ids_raise_stale_node_error(mod):
    rt, a, q, calls = build(mod)
    rt.get_traced(q)
    old = named(rt, q)
    rt.delete_node(q)
    for name in [old, (10_000, 0), (2**32 - 1, 0)]:
        with pytest.raises(mod.StaleNodeError):
            rt.peek_by_id(*name)
        with pytest.raises(mod.StaleNodeError):
            rt.last_trace_by_id(*name)
    reused = rt.create_input(7)
    assert reused.id == old[0]
    with pytest.raises(mod.StaleNodeError):
        rt.last_trace_by_id(*old)
    assert rt.last_trace_by_id(*named(rt, reused))[1] is None


def test_a_failed_node_reads_failed_without_a_compute(mod):
    rt = mod.Runtime()
    calls = []

    def boom(r):
        calls.append(1)
        raise ValueError("boom")

    q = rt.create_query(boom)
    with pytest.raises(ValueError):
        rt.get_traced(q)
    assert rt.peek_by_id(*named(rt, q)) == ("failed", None)
    assert rt.last_trace_by_id(*named(rt, q)) == ("failed", None)
    assert calls == [1]
