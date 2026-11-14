import gc
import pickle
import threading
import time

import incr
import incr_concurrent
import pytest

PACKAGES = [
    pytest.param(incr, id="local"),
    pytest.param(incr_concurrent, id="concurrent"),
]

runtimes = pytest.mark.parametrize(
    "make_rt", [incr.Runtime, incr_concurrent.Runtime], ids=["local", "concurrent"]
)


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


def labelled_input(rt, label, value=1):
    node = rt.create_input(value)
    rt.set_label(node, label)
    return node


def labelled_query(rt, label, fn):
    node = rt.create_query(fn)
    rt.set_label(node, label)
    return node


def groups(summary):
    return {g["label"]: g for g in summary["groups"]}


def edges(summary):
    return {(e["from"], e["to"]): e["count"] for e in summary["edges"]}


def chain(rt):
    a1 = labelled_input(rt, "shifts/1")
    a2 = labelled_input(rt, "shifts/2")
    b = labelled_query(rt, "pricing/1", lambda r: r.get(a1) + r.get(a2))
    c = labelled_query(rt, "totals/1", lambda r: r.get(b) * 2)
    rt.get(c)
    return a1, a2, b, c


@runtimes
def test_three_groups_in_a_chain_report_counts_and_edges(make_rt):
    rt = make_rt()
    chain(rt)
    summary = rt.graph_summary()
    g = groups(summary)
    assert set(g) == {"shifts", "pricing", "totals"}
    assert (g["shifts"]["nodes"], g["shifts"]["kind"]) == (2, "input")
    assert (g["pricing"]["nodes"], g["pricing"]["kind"]) == (1, "compute")
    assert g["totals"]["nodes"] == 1
    assert edges(summary) == {("shifts", "pricing"): 2, ("pricing", "totals"): 1}


@runtimes
def test_counts_move_after_add_and_delete(make_rt):
    rt = make_rt()
    a1, _, b, _ = chain(rt)
    extra = labelled_query(rt, "pricing/2", lambda r: r.get(a1))
    rt.get(extra)
    summary = rt.graph_summary()
    assert groups(summary)["pricing"]["nodes"] == 2
    assert edges(summary)[("shifts", "pricing")] == 3
    rt.delete_node(extra)
    summary = rt.graph_summary()
    assert groups(summary)["pricing"]["nodes"] == 1
    assert edges(summary)[("shifts", "pricing")] == 2


@runtimes
def test_an_edge_inside_one_group_is_a_self_edge(make_rt):
    rt = make_rt()
    a = labelled_input(rt, "g/in")
    b = labelled_query(rt, "g/mid", lambda r: r.get(a) + 1)
    rt.get(b)
    summary = rt.graph_summary()
    assert edges(summary) == {("g", "g"): 1}
    assert groups(summary)["g"]["kind"] == "mixed"


@runtimes
def test_unlabelled_nodes_are_under_none(make_rt):
    rt = make_rt()
    a = rt.create_input(1)
    q = rt.create_query(lambda r: r.get(a))
    rt.get(q)
    summary = rt.graph_summary()
    assert groups(summary)[None]["nodes"] == 2
    assert edges(summary) == {(None, None): 1}


@runtimes
def test_create_query_takes_a_label(make_rt):
    rt = make_rt()
    a = rt.create_input(1)
    q = rt.create_query(lambda r: r.get(a), label="named/q")
    rt.get(q)
    g = groups(rt.graph_summary())
    assert g["named"]["nodes"] == 1
    assert g[None]["nodes"] == 1


@pytest.fixture(params=PACKAGES)
def pkg(request):
    return request.param


def test_labelled_keyed_collection_reports_its_key_count(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt, label="shifts/rows")
    for key in range(5):
        kc.upsert(key, key)
    kc.remove(0)
    g = groups(rt.graph_summary())
    assert g["shifts"]["keys"] == len(kc) == 4
    assert g["shifts"]["nodes"] >= 2
    kc.upsert(10, 10)
    assert groups(rt.graph_summary())["shifts"]["keys"] == 5


def test_two_labelled_keyed_collections_in_one_group_add_their_keys(pkg):
    rt = pkg.Runtime()
    a = pkg.KeyedCollection(rt, label="shifts/a")
    b = pkg.KeyedCollection(rt, label="shifts/b")
    a.upsert(1, 1)
    b.upsert(1, 1)
    b.upsert(2, 2)
    assert groups(rt.graph_summary())["shifts"]["keys"] == 3


def test_unlabelled_keyed_collection_adds_no_keys(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt)
    kc.upsert(1, 1)
    assert all("keys" not in g for g in rt.graph_summary()["groups"])


def test_a_dropped_keyed_collection_stops_reporting_keys(pkg):
    rt = pkg.Runtime()
    kc = pkg.KeyedCollection(rt, label="shifts/rows")
    kc.upsert(1, 1)
    assert groups(rt.graph_summary())["shifts"]["keys"] == 1
    del kc
    gc.collect()
    assert "keys" not in groups(rt.graph_summary()).get("shifts", {})


def test_persisted_keyed_collection_reports_its_key_count(pkg):
    store = pkg.persist.Store.open(pkg.persist.MemoryBackend(), build_id=1)
    codec = PickleCodec()
    kc = store.keyed_collection("rows", codec, codec, label="pay/rows")
    for key in "abc":
        store.upsert(kc, key, key.upper())
    store.remove(kc, "b")
    g = groups(store.runtime.graph_summary())
    assert g["pay"]["keys"] == len(kc) == 2


def test_summary_does_not_block_a_concurrent_set():
    rt = incr_concurrent.Runtime()
    inputs = [rt.create_input(i) for i in range(200_000)]
    target = inputs[0]
    done = threading.Event()
    longest = 0.0

    def writer():
        nonlocal longest
        n = 0
        while not done.is_set():
            n += 1
            t = time.perf_counter()
            rt.set(target, n)
            longest = max(longest, time.perf_counter() - t)

    thread = threading.Thread(target=writer)
    thread.start()
    time.sleep(0.05)
    t = time.perf_counter()
    summary = rt.graph_summary()
    took = time.perf_counter() - t
    done.set()
    thread.join()
    assert groups(summary)[None]["nodes"] == 200_000
    assert longest < took


def test_a_keyed_collection_made_in_an_overlay_stops_counting_at_exit():
    rt = incr.Runtime()
    base = incr.KeyedCollection(rt, label="kept")
    base.upsert("a", 1)
    with rt.overlay():
        made = incr.KeyedCollection(rt, label="gone")
        made.upsert("x", 1)
        inside = {g["label"]: g for g in rt.graph_summary()["groups"]}
        assert inside["gone"]["keys"] == 1
    groups = {g["label"]: g for g in rt.graph_summary()["groups"]}
    assert "keys" not in groups.get("gone", {})
    assert groups["kept"]["keys"] == 1
    assert made is not None
