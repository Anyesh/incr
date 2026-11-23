import json
import math
import os
import subprocess
import sys
from operator import add

import pytest

from incr import KeyedCollection, Runtime


def graph():
    rt = Runtime()
    roots = []
    for k in range(200):
        i = rt.create_input(k)
        rt.set_label(i, f"in{k % 4}/{k}")
        a = rt.create_query(lambda r, i=i: r.get(i) + 1)
        roots.append(rt.create_query(lambda r, a=a: r.get(a) * 2, label=f"pay{k % 3}/{k}"))
    for root in roots[:150]:
        rt.get_traced(root)
    for root in roots[150:]:
        rt.get(root)
    rows = KeyedCollection(rt, label="shifts/rows")
    rows.reconcile([(j, (j % 7, j)) for j in range(500)])
    shifts = rt.create_collection()
    shifts.extend((j % 5, j, j % 3) for j in range(300))
    rt.set_label(shifts.version_node, "shifts/all")
    groups = shifts.group_by(lambda row: row[0])
    totals = [shifts.aggregate(0, lambda row: row[2], add)]
    for w in range(5):
        g = groups.get_group(w)
        totals.append(g.sort_by_key(lambda row: row[1]).pairwise().count())
    for n in totals:
        rt.get(n)
    rt.delete_with_dependents(roots[-1])
    return rt, (roots, rows, shifts, groups, totals)


def walk(walker, budget=0.0):
    steps = 1
    while not walker.step(budget):
        steps += 1
    return steps


def test_a_walk_in_steps_equals_memory_stats_on_a_quiet_runtime():
    rt, _keep = graph()
    walker = rt.memory_walker()

    steps = walk(walker)
    result = walker.result()

    assert steps > 1, "a zero budget returns after one part of the walk"
    assert result == rt.memory_stats()
    assert result["kept_traces"]["count"] == 150
    assert {"pay0", "pay1", "pay2", "in0", "shifts"} <= set(result["by_label"])
    assert walker.step(0.0)
    assert walker.result() == result


def test_an_unbudgeted_step_finishes_the_walk():
    rt, _keep = graph()
    walker = rt.memory_walker()

    assert walker.step(math.inf)
    assert walker.result() == rt.memory_stats()


def test_result_before_the_walk_ends_raises():
    rt, _keep = graph()
    walker = rt.memory_walker()
    assert not walker.step(0.0)

    with pytest.raises(RuntimeError, match="step"):
        walker.result()


@pytest.mark.parametrize("budget", [-0.1, math.nan])
def test_a_budget_must_be_a_non_negative_number_of_seconds(budget):
    rt = Runtime()
    walker = rt.memory_walker()

    with pytest.raises(ValueError):
        walker.step(budget)


def test_a_walk_across_changes_finishes_and_later_walks_match_again():
    rt, (roots, rows, shifts, _groups, _totals) = graph()
    walker = rt.memory_walker()
    assert not walker.step(0.0)

    for root in roots[:50]:
        rt.delete_with_dependents(root)
    fresh = [rt.create_query(lambda r: 1, label="late/q") for _ in range(30)]
    rows.reconcile([(j, (j,)) for j in range(100)])
    shifts.extend((9, j, 1) for j in range(50))
    walk(walker)
    mixed = walker.result()

    assert mixed["nodes"] > 0
    assert set(mixed) == set(rt.memory_stats())
    again = rt.memory_walker()
    walk(again)
    assert again.result() == rt.memory_stats()
    assert fresh


def test_stepping_a_closed_runtime_raises():
    rt, _keep = graph()
    walker = rt.memory_walker()
    rt.close()

    with pytest.raises(RuntimeError):
        walker.step(0.0)


L_SHAPE = r"""
import json
import sys
import time
from functools import partial
from operator import add

import incr

ROOTS, KEYED_ROWS, SHIFTS, WORKERS, BUDGET = 270_967, 2_614_122, 240_000, 5_000, 0.05


def s0(i, r):
    return r.get(i) + 1000


def s1(a, r):
    return r.get(a) + 1


def s2(a, b, r):
    return r.get(a) + r.get(b)


rt = incr.Runtime()
roots = []
for k in range(ROOTS):
    i = rt.create_input(k)
    rt.set_label(i, f"in{k % 50}/{k}")
    a = rt.create_query(partial(s0, i))
    b = rt.create_query(partial(s1, a))
    c = rt.create_query(partial(s2, a, b))
    d = rt.create_query(partial(s2, b, c))
    roots.append(rt.create_query(partial(s2, c, d), label=f"pay{k % 100}/{k}"))
for root in roots:
    rt.get_traced(root)

families = []
for f in range(47):
    kc = incr.KeyedCollection(rt, label=f"fam{f}/rows")
    n = KEYED_ROWS * 8 // 70 if f < 7 else KEYED_ROWS * 2 // 400
    kc.reconcile([((f, j), (j, j * 3, "s")) for j in range(n)])
    families.append(kc)
whole_family = families[0].collection.aggregate(0, lambda row: row[1], add)

shifts = rt.create_collection()
shifts.extend([(j % WORKERS, j, (j * 7) % 13) for j in range(SHIFTS)])
groups = shifts.group_by(lambda row: row[0])
totals = [whole_family]
for w in range(WORKERS):
    g = groups.get_group(w)
    totals.append(g.aggregate(0, lambda row: row[2], add))
    totals.append(g.sort_by_key(lambda row: row[1]).pairwise().aggregate(0, lambda p: p[1][1] - p[0][1], add))
    totals.append(g.count())
for n in totals:
    rt.get(n)

t = time.perf_counter()
whole = rt.memory_stats()
whole_s = time.perf_counter() - t
walker = rt.memory_walker()
gaps = []
while True:
    t = time.perf_counter()
    done = walker.step(BUDGET)
    gaps.append(time.perf_counter() - t)
    if done:
        break
t = time.perf_counter()
result = walker.result()
gaps.append(time.perf_counter() - t)
print(json.dumps({
    "nodes": whole["nodes"],
    "memory_stats_s": whole_s,
    "steps": len(gaps),
    "max_step_s": max(gaps),
    "equal": result == whole,
}))
"""


@pytest.mark.skipif(
    not os.environ.get("INCR_SCALE_TESTS"),
    reason="builds the client's L shape (about 2 GB); set INCR_SCALE_TESTS=1",
)
def test_every_step_at_the_l_shape_returns_within_its_budget_plus_50_ms():
    out = subprocess.run(
        [sys.executable, "-c", L_SHAPE], check=True, capture_output=True, text=True
    )
    figures = json.loads(out.stdout)
    print(figures)

    assert figures["equal"]
    assert figures["steps"] > 1
    assert figures["max_step_s"] <= 0.05 + 0.05
