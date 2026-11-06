import pytest
from incr import KeyedCollection, Pending, Runtime


class Boom(Exception):
    pass


def shifts(rt):
    rows = rt.create_collection()
    rows.extend([("ann", 8), ("ann", 4), ("bob", 6)])
    return rows


def count_of(rt, grouped, key):
    return rt.get(grouped.get_group(key).count())


def test_an_insert_routes_with_no_read_of_the_version_node():
    rt = Runtime()
    rows = rt.create_collection()
    by_worker = rows.group_by(lambda row: row[0])
    rows.insert(("ann", 8))
    assert count_of(rt, by_worker, "ann") == 1
    rows.insert(("ann", 4))
    rows.insert(("bob", 6))
    assert count_of(rt, by_worker, "ann") == 2
    assert sorted(by_worker.keys()) == ["ann", "bob"]
    assert count_of(rt, by_worker, "bob") == 1


def test_only_the_touched_groups_watched_total_is_reported():
    rt = Runtime()
    rows = shifts(rt)
    by_worker = rows.group_by(lambda row: row[0])
    ann = by_worker.get_group("ann").aggregate(
        0, lambda row: row[1], lambda a, b: a + b
    )
    bob = by_worker.get_group("bob").aggregate(
        0, lambda row: row[1], lambda a, b: a + b
    )
    assert (rt.get(ann), rt.get(bob)) == (12, 6)
    rt.watch(ann)
    rt.watch(bob)
    assert rt.drain_dirtied_watched() == []
    rows.insert(("ann", 3))
    assert rt.drain_dirtied_watched() == [ann]
    assert rt.get(ann) == 15
    assert rt.drain_dirtied_watched() == []


def test_a_keyed_upsert_routes_the_group_by_it_feeds():
    rt = Runtime()
    keyed = KeyedCollection(rt)
    by_worker = keyed.collection.group_by(lambda row: row[0])
    keyed.upsert("s1", ("ann", 8))
    assert count_of(rt, by_worker, "ann") == 1
    keyed.upsert("s1", ("bob", 8))
    assert count_of(rt, by_worker, "bob") == 1
    assert count_of(rt, by_worker, "ann") == 0


def test_no_key_function_runs_inside_a_write_and_the_next_read_routes():
    rt = Runtime()
    flag = rt.create_input(0)
    calls = []

    def key(row):
        calls.append(row)
        rt.get(flag)
        return row[0]

    rows = shifts(rt)
    keyed = KeyedCollection(rt)
    by_worker = rows.group_by(key)
    by_keyed = keyed.collection.group_by(key)
    assert count_of(rt, by_worker, "ann") == 2
    calls.clear()

    writes = [
        lambda: rows.insert(("cat", 1)),
        lambda: rows.delete(("ann", 4)),
        lambda: rows.extend([("dan", 2), ("eve", 3)]),
        lambda: rows.replace(("bob", 6), ("bob", 7)),
        lambda: rt.set(flag, 1),
        lambda: keyed.upsert("k1", ("fay", 1)),
        lambda: keyed.reconcile({"k1": ("fay", 2), "k2": ("gus", 1)}),
        lambda: keyed.remove("k2"),
    ]
    for write in writes:
        write()
        assert calls == []

    assert count_of(rt, by_worker, "cat") == 1
    assert calls
    assert count_of(rt, by_keyed, "fay") == 1
    assert sorted(by_worker.keys()) == ["ann", "bob", "cat", "dan", "eve"]
    assert sorted(by_keyed.keys()) == ["fay", "gus"]
    assert count_of(rt, by_keyed, "gus") == 0


def test_a_key_function_failure_is_raised_by_the_next_read_chained():
    rt = Runtime()
    rows = shifts(rt)

    def key(row):
        if row[0] == "bad":
            raise Boom("no worker")
        return row[0]

    by_worker = rows.group_by(key)
    ann = by_worker.get_group("ann").count()
    assert rt.get(ann) == 2
    rows.insert(("bad", 1))
    with pytest.raises(RuntimeError, match="routing failed") as raised:
        rt.get(ann)
    assert isinstance(raised.value.__cause__, Boom)
    assert str(raised.value.__cause__) == "no worker"
    rows.delete(("bad", 1))
    rows.insert(("ann", 1))
    assert rt.get(ann) == 3


def test_drain_dirtied_watched_raises_a_routing_failure_before_taking_ids():
    rt = Runtime()
    rows = shifts(rt)

    def key(row):
        if row[0] == "bad":
            raise Boom("no worker")
        return row[0]

    by_worker = rows.group_by(key)
    a = rt.create_input(1)
    q = rt.create_query(lambda r: r.get(a) + 1)
    assert rt.get(q) == 2
    rt.watch(q)
    rt.set(a, 2)
    rows.insert(("bad", 1))
    with pytest.raises(RuntimeError, match="routing failed") as raised:
        rt.drain_dirtied_watched()
    assert isinstance(raised.value.__cause__, Boom)
    rows.delete(("bad", 1))
    assert rt.drain_dirtied_watched() == [q]
    assert count_of(rt, by_worker, "ann") == 2


def test_a_not_ready_element_without_a_row_fails_the_read_that_routes_it():
    rt = Runtime()
    rows = shifts(rt)
    by_worker = rows.group_by(lambda row: row[0])
    assert count_of(rt, by_worker, "ann") == 2
    rows.insert(Pending(key="s9"))
    with pytest.raises(RuntimeError, match="routing failed") as raised:
        by_worker.keys()
    assert isinstance(raised.value.__cause__, TypeError)
    assert "no row" in str(raised.value.__cause__)


def test_an_observer_that_writes_leaves_later_observers_reading_current_groups():
    rt = Runtime()
    rows = shifts(rt)
    by_worker = rows.group_by(lambda row: row[0])
    ann = by_worker.get_group("ann").count()
    assert rt.get(ann) == 2
    tick = rt.create_input(0)
    tock = rt.create_query(lambda r: r.get(tick))
    seen = []
    rt.observe(tock, lambda t: rows.insert(("ann", 100 + t)))
    rt.observe(tock, lambda t: seen.append(rt.get(ann)))
    rt.stabilize()
    assert seen == [3]
    rt.set(tick, 1)
    rt.stabilize()
    assert seen == [3, 4]


@pytest.mark.skip(
    reason="the pending list's length is not exposed to Python; "
    "incr-core's collection_routing_tests bound it through queued_routers"
)
def test_the_pending_list_stays_bounded_under_writes_with_no_read():
    pass
