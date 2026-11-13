import time

import incr
import incr_concurrent
import pytest


@pytest.fixture(params=[incr, incr_concurrent], ids=["local", "shared"])
def mod(request):
    return request.param


def build_chain(rt, length):
    nodes = [rt.create_input(1)]
    for _ in range(length - 1):
        prev = nodes[-1]
        q = rt.create_query(lambda r, p=prev: r.get(p) + 1)
        rt.get(q)
        nodes.append(q)
    return nodes


def ids(snapshot):
    return [n["id"] for n in snapshot["nodes"]]


def test_one_root_with_depth_one(mod):
    rt = mod.Runtime()
    chain = build_chain(rt, 5)
    snap = rt.graph_snapshot(roots=[chain[3]], depth=1)
    assert ids(snap) == [chain[3].id, chain[2].id]
    assert snap["truncated"] is False
    tip = snap["nodes"][0]
    assert tip["dependencies"] == [chain[2].id]
    assert tip["dependents"] == [chain[4].id]
    assert snap["nodes"][1]["dependencies"] == [chain[1].id]


def test_depth_zero_returns_only_the_roots(mod):
    rt = mod.Runtime()
    chain = build_chain(rt, 4)
    snap = rt.graph_snapshot(roots=[chain[3]], depth=0)
    assert ids(snap) == [chain[3].id]


def test_limit_caps_and_sets_truncated(mod):
    rt = mod.Runtime()
    chain = build_chain(rt, 6)
    snap = rt.graph_snapshot(roots=[chain[5]], limit=3)
    assert ids(snap) == [chain[5].id, chain[4].id, chain[3].id]
    assert snap["truncated"] is True
    snap = rt.graph_snapshot(roots=[chain[5]], limit=6)
    assert len(snap["nodes"]) == 6
    assert snap["truncated"] is False


def test_label_prefix_excludes_but_walks_through(mod):
    rt = mod.Runtime()
    chain = build_chain(rt, 3)
    rt.set_label(chain[0], "a/in")
    rt.set_label(chain[1], "x/mid")
    rt.set_label(chain[2], "a/out")
    snap = rt.graph_snapshot(roots=[chain[2]], label_prefix="a/")
    assert ids(snap) == [chain[2].id, chain[0].id]
    assert snap["truncated"] is False
    assert snap["nodes"][0]["dependencies"] == [chain[1].id]


def test_direction_dependents_and_both(mod):
    rt = mod.Runtime()
    chain = build_chain(rt, 4)
    down = rt.graph_snapshot(roots=[chain[1]], direction="dependents")
    assert ids(down) == [chain[1].id, chain[2].id, chain[3].id]
    both = rt.graph_snapshot(roots=[chain[1]], direction="both", depth=1)
    assert sorted(ids(both)) == sorted([chain[0].id, chain[1].id, chain[2].id])


def test_no_argument_call_is_unchanged(mod):
    rt = mod.Runtime()
    chain = build_chain(rt, 4)
    full = rt.graph_snapshot()
    assert isinstance(full, list)
    assert [n["id"] for n in full] == [n.id for n in chain]
    assert set(full[0]) == {"id", "kind", "label", "dependencies", "dependents"}
    bounded = rt.graph_snapshot(roots=[chain[3]])
    assert sorted(n["id"] for n in bounded["nodes"]) == sorted(n["id"] for n in full)
    by_id = {n["id"]: n for n in full}
    assert all(by_id[n["id"]] == n for n in bounded["nodes"])


def test_collection_as_a_root(mod):
    rt = mod.Runtime()
    rows = rt.create_collection()
    for v in (1, 2, 3):
        rows.insert(v)
    version = rows.version_node
    watcher = rt.create_query(lambda r: r.get(version))
    rt.get(watcher)
    snap = rt.graph_snapshot(roots=[rows], direction="dependents")
    assert snap["nodes"][0]["id"] == rows.version_node.id
    assert watcher.id in ids(snap)
    assert snap == rt.graph_snapshot(roots=[rows.version_node], direction="dependents")


def test_value_errors(mod):
    rt = mod.Runtime()
    chain = build_chain(rt, 2)
    with pytest.raises(ValueError):
        rt.graph_snapshot(roots=[chain[1]], direction="sideways")
    with pytest.raises(ValueError):
        rt.graph_snapshot(depth=1)
    with pytest.raises(ValueError):
        rt.graph_snapshot(limit=5)
    with pytest.raises(ValueError):
        rt.graph_snapshot(label_prefix="a")
    with pytest.raises(ValueError):
        rt.graph_snapshot(direction="both")
    with pytest.raises(ValueError):
        rt.graph_snapshot(roots=[object()])


def test_empty_roots_return_an_empty_snapshot(mod):
    rt = mod.Runtime()
    build_chain(rt, 2)
    assert rt.graph_snapshot(roots=[]) == {"nodes": [], "truncated": False}


def test_bounded_call_is_much_faster_than_the_full_snapshot(mod):
    rt = mod.Runtime()
    chain = build_chain(rt, 200_000)
    start = time.perf_counter()
    full = rt.graph_snapshot()
    full_time = time.perf_counter() - start
    assert len(full) == 200_000
    start = time.perf_counter()
    snap = rt.graph_snapshot(roots=[chain[-1]], limit=300)
    bounded_time = time.perf_counter() - start
    assert len(snap["nodes"]) == 300
    assert snap["truncated"] is True
    assert bounded_time < full_time / 10
