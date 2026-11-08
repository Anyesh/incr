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


def test_keyed_collection_and_table_reads_raise_after_close():
    rt = incr.Runtime()
    kc = incr.KeyedCollection(rt)
    kc.upsert("a", 1)
    t = rt.create_table()
    t.get_or_insert_input("k", lambda: 0)
    rt.close()
    calls = [
        lambda: kc.get("a"),
        lambda: "a" in kc,
        lambda: len(kc),
        lambda: kc.upsert("b", 2),
        lambda: kc.remove("a"),
        lambda: kc.reconcile({}),
        lambda: kc.collection,
        lambda: t.get("k"),
        lambda: t.remove("k"),
        lambda: t.items(),
    ]
    for call in calls:
        with pytest.raises(RuntimeError, match="runtime is closed"):
            call()


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


def test_try_delete_node_refuses_while_a_dependent_exists():
    rt = incr.Runtime()
    a = rt.create_input(1)
    q = rt.create_query(lambda r: r.get(a) * 2)
    assert rt.get(q) == 2
    with pytest.raises(incr.HasDependentsError):
        rt.try_delete_node(a)
    assert issubclass(incr.HasDependentsError, RuntimeError)
    assert rt.is_alive(a)
    rt.set(a, 4)
    assert rt.get(q) == 8


def test_try_delete_node_succeeds_after_the_dependent_is_gone():
    rt = incr.Runtime()
    a = rt.create_input(1)
    q = rt.create_query(lambda r: r.get(a) * 2)
    assert rt.get(q) == 2
    rt.try_delete_node(q)
    rt.try_delete_node(a)
    assert not rt.is_alive(a)
    with pytest.raises(KeyError):
        rt.try_delete_node(a)


def test_table_remove_with_a_dependent_raises_has_dependents():
    rt = incr.Runtime()
    t = rt.create_table()
    n = t.get_or_insert_input("k", lambda: 1)
    q = rt.create_query(lambda r: r.get(n))
    assert rt.get(q) == 1
    with pytest.raises(incr.HasDependentsError):
        t.remove("k")
