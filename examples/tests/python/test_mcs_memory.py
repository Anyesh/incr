import time

from incr import KeyedCollection, Runtime

ENTRY = {"nodes", "bytes", "py_refs"}


def chain(rt, length):
    root = rt.create_input(1)
    nodes = [root]
    for _ in range(length):
        prev = nodes[-1]
        nodes.append(rt.create_query(lambda r, p=prev: r.get(p) + 1))
    rt.get(nodes[-1])
    return nodes


def test_stats_have_the_documented_shape():
    stats = Runtime().memory_stats()

    assert set(stats) == {"bytes", "nodes", "py_refs", "by_kind", "by_label"}
    assert set(stats["by_kind"]) == {"input", "query", "collection"}
    for entry in stats["by_kind"].values():
        assert set(entry) == ENTRY
    assert stats["by_label"] == {}


def test_stats_grow_with_nodes_and_shrink_after_delete_with_dependents():
    rt = Runtime()
    before = rt.memory_stats()

    nodes = chain(rt, 50)
    grown = rt.memory_stats()

    assert grown["nodes"] == before["nodes"] + 51
    assert grown["by_kind"]["input"]["nodes"] == before["by_kind"]["input"]["nodes"] + 1
    assert (
        grown["by_kind"]["query"]["nodes"] == before["by_kind"]["query"]["nodes"] + 50
    )
    assert grown["bytes"] > before["bytes"]
    # 51 values and 50 query callables.
    assert grown["py_refs"] == before["py_refs"] + 101

    assert rt.delete_with_dependents(nodes[0]) == 51
    shrunk = rt.memory_stats()

    assert shrunk["nodes"] == before["nodes"]
    assert shrunk["py_refs"] == before["py_refs"]
    assert shrunk["bytes"] < grown["bytes"]


def test_ten_thousand_rows_into_a_collection_raise_py_refs_by_at_least_as_many():
    rt = Runtime()
    rows = rt.create_collection()
    before = rt.memory_stats()

    rows.extend((i, f"row {i}") for i in range(10_000))
    after = rt.memory_stats()

    assert after["py_refs"] - before["py_refs"] >= 10_000
    collection = after["by_kind"]["collection"]
    assert collection["py_refs"] - before["by_kind"]["collection"]["py_refs"] >= 10_000
    assert collection["bytes"] > before["by_kind"]["collection"]["bytes"]


def test_ten_thousand_rows_into_a_keyed_collection_raise_py_refs_by_at_least_as_many():
    rt = Runtime()
    keyed = KeyedCollection(rt)
    before = rt.memory_stats()

    keyed.reconcile({i: (i, f"row {i}") for i in range(10_000)})
    after = rt.memory_stats()

    assert after["py_refs"] - before["py_refs"] >= 10_000


def test_keyed_index_counts_its_pairs():
    rt = Runtime()
    keyed = KeyedCollection(rt)
    before = rt.memory_stats()["py_refs"]

    keyed.reconcile({i: (i,) for i in range(1_000)})
    rt.get(keyed.collection.version_node)
    after = rt.memory_stats()["py_refs"]

    # Each pair is held by the index and, with its insert delta, by the
    # pair log; each row by the row log and its delta.
    assert after - before >= 5 * 1_000


def test_operators_hold_their_callables_until_deleted():
    rt = Runtime()
    rows = rt.create_collection()
    rows.extend(range(100))
    before = rt.memory_stats()["py_refs"]

    evens = rows.filter(lambda v: v % 2 == 0)
    total = evens.aggregate(0, lambda v: v, lambda a, b: a + b)
    assert rt.get(total) == sum(range(0, 100, 2))
    after = rt.memory_stats()["py_refs"]

    # The filter's output holds 50 rows, the aggregate its tree, and three
    # callables are held.
    assert after - before >= 50 + 3


def test_a_labelled_subgraph_prefix_accounts_for_its_nodes():
    rt = Runtime()
    base = rt.create_input(1)
    rt.set_label(base, "pricing/base")
    shifts = [rt.create_query(lambda r, i=i: r.get(base) + i) for i in range(10)]
    for i, shift in enumerate(shifts):
        rt.set_label(shift, f"pricing/shift/{i}")
    pay = rt.create_input(2)
    rt.set_label(pay, "pay/base")
    solo = rt.create_input(3)
    rt.set_label(solo, "solo")
    rt.create_input(4)
    for shift in shifts:
        rt.get(shift)

    stats = rt.memory_stats()
    by_label = stats["by_label"]

    assert set(by_label) == {"pricing", "pay", "solo"}
    for entry in by_label.values():
        assert set(entry) == ENTRY
    assert by_label["pricing"]["nodes"] == 11
    assert by_label["pricing"]["py_refs"] == 11
    assert by_label["pay"]["nodes"] == 1
    assert by_label["solo"]["nodes"] == 1
    assert by_label["pricing"]["bytes"] > 10 * by_label["pay"]["bytes"] // 2
    assert sum(e["nodes"] for e in by_label.values()) == stats["nodes"] - 1


def test_a_labelled_collection_prefix_accounts_for_its_rows():
    rt = Runtime()
    rows = rt.create_collection()
    rt.set_label(rows.version_node, "travel/rows")
    rows.extend(range(1_000))

    travel = rt.memory_stats()["by_label"]["travel"]

    assert travel["nodes"] == 1
    assert travel["py_refs"] >= 1_000


def test_a_closed_runtime_raises():
    rt = Runtime()
    rt.close()
    try:
        rt.memory_stats()
    except RuntimeError as e:
        assert "closed" in str(e)
    else:
        raise AssertionError("memory_stats on a closed runtime returned")


def test_a_hundred_thousand_node_call_takes_under_fifty_ms():
    rt = Runtime()
    base = rt.create_input(0)
    rt.set_label(base, "pricing/base")
    for i in range(50_000):
        node = rt.create_input(i)
        if i % 2:
            rt.set_label(node, f"pay/{i}")
    for i in range(49_999):
        node = rt.create_query(lambda r, i=i: r.get(base) + i)
        if i % 3 == 0:
            rt.set_label(node, f"travel/{i}")
        rt.get(node)

    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        stats = rt.memory_stats()
        best = min(best, time.perf_counter() - start)

    assert stats["nodes"] == 100_000
    assert best < 0.050, f"memory_stats took {best * 1000:.1f} ms"
