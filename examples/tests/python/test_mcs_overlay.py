import random

import incr
import pytest


def total_of(rows):
    return rows.aggregate(0, lambda r: r, lambda a, b: a + b)


def test_input_set_is_restored_and_dependents_recompute():
    rt = incr.Runtime()
    a = rt.create_input(1)
    q = rt.create_query(lambda r: r.get(a) * 10)
    assert rt.get(q) == 10
    with rt.overlay() as ov:
        ov.set(a, 5)
        ov.set(a, 7)
        assert ov.get(q) == 70
    assert rt.get(a) == 1
    assert rt.get(q) == 10


def test_overlay_restores_when_the_body_raises():
    rt = incr.Runtime()
    a = rt.create_input(1)
    with pytest.raises(ValueError):
        with rt.overlay() as ov:
            ov.set(a, 2)
            raise ValueError("boom")
    assert rt.get(a) == 1
    with rt.overlay() as ov:
        ov.set(a, 3)
    assert rt.get(a) == 1


def test_collection_mutations_are_restored():
    rt = incr.Runtime()
    rows = rt.create_collection()
    rows.extend([1, 2, 3, 3])
    total = total_of(rows)
    count = rows.count()
    assert (rt.get(total), rt.get(count)) == (9, 4)
    with rt.overlay() as ov:
        rows.insert(10)
        assert rows.delete(3) is True
        assert rows.delete(99) is False
        assert rows.replace(1, 100) is True
        rows.extend([5, 5])
        assert (ov.get(total), ov.get(count)) == (10 + 2 + 3 + 100 + 10, 6)
    assert (rt.get(total), rt.get(count)) == (9, 4)
    assert rows.snapshot_len() == 4


def test_keyed_mutations_are_restored_with_their_index():
    rt = incr.Runtime()
    kc = incr.KeyedCollection(rt)
    kc.reconcile({"a": 1, "b": 2, "c": 3})
    total = total_of(kc.collection)
    assert rt.get(total) == 6
    with rt.overlay() as ov:
        kc.upsert("a", 10)
        kc.upsert("d", 4)
        kc.remove("b")
        assert ov.get(total) == 10 + 3 + 4
        assert kc.reconcile({"z": 1}) == (1, 0, 3)
        assert len(kc) == 1
    assert rt.get(total) == 6
    assert len(kc) == 3
    assert (kc.get("a"), kc.get("b"), kc.get("c"), kc.get("d")) == (1, 2, 3, None)
    kc.upsert("a", 5)
    kc.remove("b")
    assert rt.get(total) == 5 + 3


def test_nodes_created_inside_are_deleted_at_exit():
    rt = incr.Runtime()
    a = rt.create_input(1)
    before = rt.node_count()
    with rt.overlay() as ov:
        inner = rt.create_input(2)
        q = rt.create_query(lambda r: r.get(a) + r.get(inner))
        assert ov.get(q) == 3
    assert rt.node_count() == before
    assert not rt.is_alive(inner)
    assert not rt.is_alive(q)


def test_collection_created_inside_is_dead_after_exit():
    rt = incr.Runtime()
    base = rt.create_collection()
    base.extend([1, 2])
    with rt.overlay():
        made = rt.create_collection()
        made.insert(1)
        derived = base.map(lambda v: v + 1)
        kc = incr.KeyedCollection(rt)
        kc.upsert("k", 1)
        ordered = base.sort_by_key(lambda v: v)
        grouped = base.group_by(lambda v: v % 2)
    for use in (
        lambda: made.insert(2),
        lambda: made.snapshot_len(),
        lambda: derived.snapshot_len(),
        lambda: kc.upsert("k", 2),
        lambda: made.version_node,
        lambda: derived.version_node,
        lambda: ordered.version_node,
        lambda: grouped.version_node,
        lambda: kc.collection.version_node,
    ):
        with pytest.raises(RuntimeError, match="created inside an overlay"):
            use()
    base.insert(3)
    assert base.snapshot_len() == 3


def test_a_group_is_dead_after_exit_only_when_its_grouped_collection_is():
    rt = incr.Runtime()
    base = rt.create_collection()
    base.extend([1, 2, 3])
    base_grouped = base.group_by(lambda v: v % 2)
    with rt.overlay():
        made_grouped = base.group_by(lambda v: v % 2)
        made_group = made_grouped.get_group(1)
        base_group = base_grouped.get_group(1)
        assert made_group.snapshot_len() == 2
    with pytest.raises(RuntimeError, match="created inside an overlay"):
        made_group.snapshot_len()
    assert base_group.snapshot_len() == 2
    assert base_grouped.get_group(1).snapshot_len() == 2


def test_a_version_node_read_inside_an_overlay_leaves_node_count_unchanged():
    rt = incr.Runtime()
    base = rt.create_collection()
    base.extend([1, 2])
    ordered = base.sort_by_key(lambda v: v)
    grouped = base.group_by(lambda v: v % 2)
    before = rt.node_count()
    with rt.overlay():
        for collection in (base, ordered, grouped):
            collection.version_node
    assert rt.node_count() == before
    assert rt.get(base.version_node) is not None


def test_labels_are_restored():
    rt = incr.Runtime()
    a = rt.create_input(1)
    rt.set_label(a, "base/a")
    with rt.overlay():
        rt.set_label(a, "overlay/a")
    traced = rt.graph_snapshot()
    assert "base/a" in repr(traced)
    assert "overlay/a" not in repr(traced)


@pytest.mark.parametrize(
    "forbidden",
    [
        lambda rt, n: rt.delete_node(n),
        lambda rt, n: rt.delete_with_dependents(n),
        lambda rt, n: rt.try_delete_node(n),
        lambda rt, n: rt.close(),
        lambda rt, n: rt.drain_dirtied_watched(),
        lambda rt, n: rt.stabilize(),
        lambda rt, n: rt.watch(n),
        lambda rt, n: rt.unwatch(n),
        lambda rt, n: rt.observe(n, lambda v: None),
        lambda rt, n: rt.overlay().__enter__(),
        lambda rt, n: rt.resolve_lookups({("k",): 1}),
        lambda rt, n: rt.fail_lookup(("k",), "no"),
        lambda rt, n: rt.retry_lookups([("k",)]),
    ],
)
def test_forbidden_operations_raise_and_leave_the_overlay_restorable(forbidden):
    rt = incr.Runtime()
    a = rt.create_input(1)
    rt.create_lookup(("k",))
    with rt.overlay() as ov:
        ov.set(a, 2)
        with pytest.raises(RuntimeError, match="overlay"):
            forbidden(rt, a)
        assert ov.get(a) == 2
    assert rt.get(a) == 1


def test_drain_after_exit_reports_only_base_changes():
    rt = incr.Runtime()
    rows = rt.create_collection()
    rows.extend([1, 2])
    total = total_of(rows)
    other = rt.create_query(lambda r: r.get(total) + 1)
    assert rt.get(other) == 4
    rt.watch(total)
    rt.watch(other)
    assert rt.drain_dirtied_watched() == []
    with rt.overlay() as ov:
        rows.insert(10)
        assert ov.get(total) == 13
    assert rt.drain_dirtied_watched() == []
    rows.insert(5)
    assert {n for n in rt.drain_dirtied_watched()} == {total, other}


def test_a_node_dirty_and_unreported_at_entry_stays_unreported():
    rt = incr.Runtime()
    rows = rt.create_collection()
    rows.extend([1, 2])
    total = total_of(rows)
    assert rt.get(total) == 3
    rt.watch(total)
    rows.insert(4)
    assert rt.drain_dirtied_watched() == [total]
    with rt.overlay() as ov:
        assert ov.get(total) == 7
        rows.insert(8)
    assert rt.drain_dirtied_watched() == []
    assert rt.get(total) == 7


def test_a_base_child_keeps_its_value_and_handle_across_an_overlay():
    rt = incr.Runtime()
    flag = rt.create_input(True)
    handles = []

    def creator(r):
        use_child = r.get(flag)
        if not use_child:
            return -1
        child = r.child_input("c", lambda: 0)
        handles.append(child)
        return r.get(child)

    q = rt.create_query(creator)
    assert rt.get(q) == 0
    child = handles[-1]
    rt.set(child, 41)
    assert rt.get(q) == 41
    with rt.overlay() as ov:
        ov.set(flag, False)
        assert ov.get(q) == -1
    assert rt.get(q) == 41
    assert rt.get(child) == 41
    assert rt.is_alive(child)


def test_overlays_in_a_row_leave_node_count_unchanged_when_a_creator_makes_children():
    rt = incr.Runtime()
    k = rt.create_input(0)

    def creator(r):
        return r.get(r.child_input(("c", r.get(k)), lambda: 0))

    q = rt.create_query(creator)
    rt.watch(q)
    assert rt.get(q) == 0
    rt.drain_dirtied_watched()
    before = rt.node_count()
    for i in range(1, 301):
        with rt.overlay() as ov:
            ov.set(k, i)
            assert ov.get(q) == 0
    assert rt.node_count() == before
    assert rt.get(q) == 0
    assert rt.drain_dirtied_watched() == []


def test_lookup_created_inside_stays_and_resolves_between_overlays():
    rt = incr.Runtime()
    leg = rt.create_lookup(("leg", 1))
    rt.drain_pending_lookups()
    with rt.overlay():
        rt.create_lookup(("leg", 2))
    assert rt.drain_pending_lookups() == [("leg", 2)]
    rt.resolve_lookups({("leg", 2): 20})
    inside = rt.create_lookup(("leg", 2))
    with rt.overlay() as ov:
        assert ov.get(inside) == 20
    assert rt.get(inside) == 20
    assert rt.is_alive(leg)


class Poison:
    armed = False

    def __init__(self, n):
        self.n = n

    def __hash__(self):
        if Poison.armed:
            raise RuntimeError("hash failed")
        return hash(self.n)

    def __eq__(self, other):
        return isinstance(other, Poison) and self.n == other.n


def test_a_failed_restore_poisons_the_runtime_and_close_still_releases_it():
    rt = incr.Runtime()
    rows = rt.create_collection()
    rows.insert(Poison(1))
    try:
        with pytest.raises(RuntimeError):
            with rt.overlay():
                rows.delete(Poison(1))
                Poison.armed = True
    finally:
        Poison.armed = False
    with pytest.raises(RuntimeError, match="poisoned"):
        rt.create_input(0)
    with pytest.raises(RuntimeError, match="poisoned"):
        rows.snapshot_len()
    rt.close()
    with pytest.raises(RuntimeError, match="closed"):
        rt.create_input(0)


def test_every_operator_created_inside_leaves_node_count_unchanged():
    rt = incr.Runtime()
    base = rt.create_collection()
    other = rt.create_collection()
    base.extend([(1, "a"), (2, "b"), (3, "a")])
    other.extend([(1, "x"), (2, "y")])
    before = rt.node_count()
    with rt.overlay() as ov:
        made = [
            base.filter(lambda r: r[0] > 1),
            base.map(lambda r: r),
            base.join(other, lambda r: r[0], lambda r: r[0]),
            base.sort_by_key(lambda r: r[0]).pairwise(),
            base.sort_by_key(lambda r: r[0]).window(2),
        ]
        nodes = [
            base.count(),
            base.reduce(lambda rows: len(rows)),
            base.aggregate(0, lambda r: r[0], lambda a, b: a + b),
        ]
        grouped = base.group_by(lambda r: r[1])
        for node in nodes:
            ov.get(node)
        for collection in made:
            collection.snapshot_len()
        grouped.group_count()
    assert rt.node_count() == before


def build_graph(rt, rows):
    col = rt.create_collection()
    col.extend(rows)
    ordered = col.sort_by_key(lambda r: r[0])
    version = ordered.version_node
    gaps = ordered.pairwise().aggregate(
        0, lambda p: p[1][0] - p[0][0], lambda a, b: a + b
    )
    window_sums = ordered.window(3).aggregate(
        0, lambda w: sum(r[0] for r in w), lambda a, b: a + b
    )
    grouped = col.group_by(lambda r: r[1])
    return (
        col,
        {
            "total": col.aggregate(0, lambda r: r[0], lambda a, b: a + b),
            "count": col.count(),
            "gaps": gaps,
            "windows": window_sums,
            "sorted": rt.create_query(
                lambda r: (r.get(version), tuple(ordered.snapshot()))[1]
            ),
        },
        grouped,
    )


def read_all(rt, nodes, grouped):
    values = {name: rt.get(node) for name, node in nodes.items()}
    values["groups"] = {
        k: grouped.get_group(k).snapshot_len()
        for k in grouped.keys()
        if grouped.get_group(k).snapshot_len()
    }
    return values


def test_random_candidates_match_from_scratch_graphs():
    rng = random.Random(7)
    base_rows = [(i * 3, rng.choice("abc")) for i in range(30)]
    rt = incr.Runtime()
    col, nodes, grouped = build_graph(rt, base_rows)
    base_values = read_all(rt, nodes, grouped)
    for round_ in range(40):
        rows = list(base_rows)
        with rt.overlay() as ov:
            for _ in range(rng.randint(1, 6)):
                kind = rng.choice(["insert", "delete", "replace"])
                if kind == "insert":
                    row = (rng.randrange(1000) * 3 + 1, rng.choice("abc"))
                    if all(r[0] != row[0] for r in rows):
                        col.insert(row)
                        rows.append(row)
                elif kind == "delete" and rows:
                    row = rng.choice(rows)
                    assert col.delete(row)
                    rows.remove(row)
                elif rows:
                    old = rng.choice(rows)
                    new = (old[0], rng.choice("abc"))
                    assert col.replace(old, new)
                    rows[rows.index(old)] = new
            fresh_rt = incr.Runtime()
            _, fresh_nodes, fresh_grouped = build_graph(fresh_rt, rows)
            inside = {n: ov.get(node) for n, node in nodes.items()}
            expected = {n: fresh_rt.get(node) for n, node in fresh_nodes.items()}
            assert inside == expected
        assert read_all(rt, nodes, grouped) == base_values


def test_a_tie_under_sort_by_key_keeps_the_sorted_keys_after_exit():
    rt = incr.Runtime()
    col = rt.create_collection()
    col.extend([(1, "a"), (1, "b"), (2, "c")])
    ordered = col.sort_by_key(lambda r: r[0])
    version = ordered.version_node
    keys = rt.create_query(
        lambda r: (r.get(version), tuple(k for k, _ in ordered.snapshot()))[1]
    )
    assert rt.get(keys) == (1, 1, 2)
    with rt.overlay():
        col.delete((1, "a"))
        col.insert((1, "z"))
    assert rt.get(keys) == (1, 1, 2)
    assert sorted(ordered.snapshot()) == [(1, "a"), (1, "b"), (2, "c")]


class Boom(Exception):
    pass


def routing_failure_source(rt):
    rows = rt.create_collection()
    state = {"fail": False}

    def key(row):
        if state["fail"] and row == "bad":
            raise Boom("no key")
        return row

    grouped = rows.group_by(key)
    return rows, grouped, state


def test_a_retried_set_after_a_raised_routing_failure_is_restored():
    rt = incr.Runtime()
    rows, _, state = routing_failure_source(rt)
    n = rt.create_input(1)
    with rt.overlay() as ov:
        state["fail"] = True
        rows.insert("bad")
        with pytest.raises(RuntimeError, match="routing failed"):
            ov.set(n, 5)
        state["fail"] = False
        ov.set(n, 5)
        assert ov.get(n) == 5
    assert rt.get(n) == 1
