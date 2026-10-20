import pathlib
import sys

import incr
import incr_concurrent
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "benches" / "python"))
import bench

BOTH = pytest.mark.parametrize(
    "module", [incr, incr_concurrent], ids=["local", "concurrent"]
)


@BOTH
def test_chains_end_at_head_plus_depth(module):
    w = bench.Chains(module, chains=10, depth=100)
    assert w.first_compute() == sum(range(10)) + 10 * 100


@BOTH
def test_running_sum_first_compute_and_edits(module):
    w = bench.RunningSum(module, n=1000)
    assert w.first_compute() == sum(range(1, 1001))
    assert w.edit_head() == sum(range(1, 1001)) + 1
    assert w.edit_last() == sum(range(1, 1001)) + 2


@BOTH
def test_prefix_sum_dag_set_at_base_doubles_every_node(module):
    w = bench.PrefixSumDag(module, n=50)
    assert w.last() == 50
    assert w.set_base() == 2 * 50
    assert w.set_base() == 3 * 50


@BOTH
def test_table_hit_finds_every_key_without_creating(module):
    w = bench.TableHit(module, keys=8)
    before = w.rt.node_count()
    handles = [w.hit() for _ in range(16)]
    assert w.rt.node_count() == before
    assert handles[:8] == handles[8:]


@BOTH
def test_table_miss_creates_one_entry_each(module):
    w = bench.TableMiss(module)
    before = w.rt.node_count()
    for _ in range(5):
        w.miss()
    assert w.rt.node_count() == before + 5


@BOTH
def test_children_rerun_keeps_every_child(module):
    w = bench.ChildrenRerun(module, n=20)
    before = w.rt.node_count()
    assert w.rerun() == 1
    assert w.rerun() == 2
    assert w.rt.node_count() == before


@BOTH
def test_children_churn_creates_and_sweeps_one(module):
    w = bench.ChildrenChurn(module)
    before = w.rt.node_count()
    assert w.churn() == 1
    assert w.rt.node_count() == before


def test_measure_reports_median_and_p99():
    ticks = iter(range(100))
    result = bench.measure(lambda: next(ticks), runs=11, check=lambda v: v >= 0)
    assert result.runs == 11
    assert result.median_ms >= 0
    assert result.p99_ms >= result.median_ms
