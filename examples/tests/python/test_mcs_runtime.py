import gc
import weakref

import incr
import pytest


class Box:
    pass


def test_closed_runtime_raises_on_every_method():
    rt = incr.Runtime()
    a = rt.create_input(1)
    q = rt.create_query(lambda r: r.get(a) + 1)
    assert rt.get(q) == 2
    rt.close()
    calls = [
        lambda: rt.get(q),
        lambda: rt.try_get(q),
        lambda: rt.set(a, 2),
        lambda: rt.create_input(0),
        lambda: rt.create_query(lambda r: 0),
        lambda: rt.create_collection(),
        lambda: rt.create_table(),
        lambda: rt.delete_node(q),
        lambda: rt.delete_with_dependents(a),
        lambda: rt.is_alive(a),
        lambda: rt.watch(q),
        lambda: rt.drain_dirtied_watched(),
        lambda: rt.stabilize(),
        lambda: rt.node_count(),
        lambda: rt.graph_snapshot(),
        lambda: rt.get_traced(q),
        lambda: rt.set_label(a, "x"),
    ]
    for call in calls:
        with pytest.raises(RuntimeError, match="runtime is closed"):
            call()


def test_collection_created_before_close_raises():
    rt = incr.Runtime()
    c = rt.create_collection()
    c.insert(1)
    n = c.count()
    t = rt.create_table()
    rt.close()
    with pytest.raises(RuntimeError, match="runtime is closed"):
        c.insert(2)
    with pytest.raises(RuntimeError, match="runtime is closed"):
        c.snapshot_len()
    with pytest.raises(RuntimeError, match="runtime is closed"):
        rt.get(n)
    with pytest.raises(RuntimeError, match="runtime is closed"):
        t.get_or_insert_input("k", 0)


def test_close_releases_captured_objects():
    rt = incr.Runtime()
    box = Box()
    ref = weakref.ref(box)
    a = rt.create_input(1)
    q = rt.create_query(lambda r, b=box: r.get(a))
    v = rt.create_input(box)
    assert rt.get(q) == 1
    del box
    gc.collect()
    assert ref() is not None
    rt.close()
    gc.collect()
    assert ref() is None
    del q, v


def test_close_is_idempotent():
    rt = incr.Runtime()
    rt.create_input(1)
    rt.close()
    rt.close()


def test_close_inside_a_compute_raises_and_leaves_runtime_usable():
    rt = incr.Runtime()
    holder = {}

    def body(r):
        with pytest.raises(RuntimeError):
            holder["rt"].close()
        return 7

    holder["rt"] = rt
    q = rt.create_query(body)
    assert rt.get(q) == 7
    a = rt.create_input(3)
    assert rt.get(a) == 3
