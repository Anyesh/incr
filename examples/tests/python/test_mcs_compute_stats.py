import time

import incr
import incr_concurrent
import pytest

runtimes = pytest.mark.parametrize(
    "make_rt", [incr.Runtime, incr_concurrent.Runtime], ids=["local", "concurrent"]
)


def chain(rt):
    source = rt.create_input(1)
    a = rt.create_query(lambda r: r.get(source) + 1)
    b = rt.create_query(lambda r: r.get(a) * 2)
    return source, a, b


def entries_by_id(trace):
    return {entry["id"]: entry for entry in trace["node_traces"]}


@runtimes
def test_trace_entries_list_the_dependencies_inside_the_trace(make_rt):
    rt = make_rt()
    source, a, b = chain(rt)
    rt.get(b)
    rt.set(source, 5)
    _, trace = rt.get_traced(b)
    entries = entries_by_id(trace)
    assert entries[b.id]["dependencies"] == [a.id]
    assert entries[a.id]["dependencies"] == [source.id]


@runtimes
def test_compute_dependencies_outside_the_trace_are_left_out(make_rt):
    rt = make_rt()
    source = rt.create_input(1)
    constant = rt.create_query(lambda r: 10)
    a = rt.create_query(lambda r: r.get(source) + r.get(constant))
    rt.get(a)
    rt.set(source, 2)
    _, trace = rt.get_traced(a)
    assert entries_by_id(trace)[a.id]["dependencies"] == [source.id]


@runtimes
def test_a_recomputed_node_has_its_own_elapsed_time(make_rt):
    rt = make_rt()
    source = rt.create_input(1)

    def slow(r):
        value = r.get(source)
        time.sleep(0.02)
        return value

    a = rt.create_query(slow)
    b = rt.create_query(lambda r: r.get(a) + 1)
    rt.get(b)
    rt.set(source, 2)
    _, trace = rt.get_traced(b)
    entries = entries_by_id(trace)
    assert entries[a.id]["elapsed_ns"] >= 15_000_000
    assert entries[b.id]["elapsed_ns"] < entries[a.id]["elapsed_ns"]


@runtimes
def test_a_node_verified_clean_has_elapsed_zero(make_rt):
    rt = make_rt()
    source = rt.create_input(1)
    parity = rt.create_query(lambda r: r.get(source) % 2)
    out = rt.create_query(lambda r: r.get(parity) + 1)
    other = rt.create_query(lambda r: r.get(parity) * 10)
    rt.get(out)
    rt.get(other)
    rt.set(source, 3)
    rt.get(out)
    _, trace = rt.get_traced(other)
    entry = entries_by_id(trace)[other.id]
    assert entry["action"] == "verified_clean"
    assert entry["elapsed_ns"] == 0


@runtimes
def test_existing_trace_keys_keep_their_meaning(make_rt):
    rt = make_rt()
    source, a, b = chain(rt)
    rt.get(b)
    rt.set(source, 7)
    value, trace = rt.get_traced(b)
    assert value == 16
    assert trace["nodes_recomputed"] == 2
    assert entries_by_id(trace)[b.id]["action"] == "recomputed_changed"


def labelled(rt):
    left = rt.create_input(1)
    right = rt.create_input(1)
    left_q = rt.create_query(lambda r: r.get(left) * 2)
    right_q = rt.create_query(lambda r: r.get(right) * 2)
    rt.set_label(left_q, "left/one")
    rt.set_label(right_q, "right/one")
    return left, right, left_q, right_q


@runtimes
def test_stats_are_off_until_enabled(make_rt):
    rt = make_rt()
    left, _right, left_q, _right_q = labelled(rt)
    rt.get(left_q)
    assert rt.compute_stats() == {}


@runtimes
def test_only_the_touched_groups_recomputed_moves(make_rt):
    rt = make_rt()
    left, _right, left_q, right_q = labelled(rt)
    rt.get(left_q)
    rt.get(right_q)
    rt.enable_compute_stats(True)
    rt.set(left, 5)
    rt.get(left_q)
    rt.get(right_q)
    stats = rt.compute_stats()
    assert stats["left"]["recomputed"] == 1
    assert stats.get("right", {}).get("recomputed", 0) == 0


@runtimes
def test_an_equal_recompute_counts_as_cutoff(make_rt):
    rt = make_rt()
    source = rt.create_input(1)
    parity = rt.create_query(lambda r: r.get(source) % 2)
    rt.set_label(parity, "parity")
    rt.get(parity)
    rt.enable_compute_stats(True)
    rt.set(source, 3)
    rt.get(parity)
    group = rt.compute_stats()["parity"]
    assert group["recomputed"] == 1
    assert group["cutoff"] == 1
    assert group["elapsed_ns"] > 0


@runtimes
def test_verified_and_unlabelled_counts(make_rt):
    rt = make_rt()
    source = rt.create_input(1)
    parity = rt.create_query(lambda r: r.get(source) % 2)
    out = rt.create_query(lambda r: r.get(parity) + 1)
    rt.get(out)
    rt.enable_compute_stats(True)
    rt.set(source, 3)
    rt.get(out)
    group = rt.compute_stats()[None]
    assert group["recomputed"] == 1
    assert group["cutoff"] == 1
    assert group["verified"] == 1


@runtimes
def test_reset_zeroes_the_counters(make_rt):
    rt = make_rt()
    left, _right, left_q, _right_q = labelled(rt)
    rt.get(left_q)
    rt.enable_compute_stats(True)
    rt.set(left, 9)
    rt.get(left_q)
    assert rt.compute_stats(reset=True)["left"]["recomputed"] == 1
    assert rt.compute_stats() == {}


@runtimes
def test_a_deleted_nodes_counts_stay_with_its_label(make_rt):
    rt = make_rt()
    left, _right, left_q, _right_q = labelled(rt)
    rt.get(left_q)
    rt.enable_compute_stats(True)
    rt.set(left, 4)
    rt.get(left_q)
    rt.delete_node(left_q)
    replacement = rt.create_query(lambda r: r.get(left) + 1)
    rt.get(replacement)
    stats = rt.compute_stats()
    assert stats["left"]["recomputed"] == 1
    assert stats[None]["recomputed"] == 1


@runtimes
def test_disabling_stops_counting(make_rt):
    rt = make_rt()
    left, _right, left_q, _right_q = labelled(rt)
    rt.get(left_q)
    rt.enable_compute_stats(True)
    rt.enable_compute_stats(False)
    rt.set(left, 6)
    rt.get(left_q)
    assert rt.compute_stats() == {}


@runtimes
def test_stats_and_a_trace_run_together(make_rt):
    rt = make_rt()
    source, a, b = chain(rt)
    rt.set_label(b, "b")
    rt.get(b)
    rt.enable_compute_stats(True)
    rt.set(source, 2)
    _, trace = rt.get_traced(b)
    assert trace["nodes_recomputed"] == 2
    assert rt.compute_stats()["b"]["recomputed"] == 1
