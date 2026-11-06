from incr import Runtime


def labelled_graph():
    rt = Runtime()
    a = rt.create_input(1)
    double = rt.create_query(lambda r: r.get(a) * 2)
    plain = rt.create_query(lambda r: r.get(a) + 100)
    total = rt.create_query(lambda r: r.get(double) + r.get(plain))
    rt.set_label(a, "a")
    rt.set_label(double, "double")
    rt.set_label(total, "total")
    return rt, a, double, plain, total


def entries_by_id(trace):
    return {entry["id"]: entry for entry in trace["node_traces"]}


def test_trace_entries_carry_labels_and_new_values():
    rt, a, double, _plain, total = labelled_graph()
    assert rt.get(total) == 103

    rt.set(a, 5)
    value, trace = rt.get_traced(total)

    assert value == 115
    entries = entries_by_id(trace)
    assert entries[double.id]["label"] == "double"
    assert entries[double.id]["value"] == 10
    assert entries[total.id]["label"] == "total"
    assert entries[total.id]["value"] == 115
    assert entries[total.id]["action"] == "recomputed_changed"


def test_unlabelled_node_has_label_none():
    rt, a, _double, plain, total = labelled_graph()
    rt.get(total)

    rt.set(a, 2)
    _, trace = rt.get_traced(total)

    entry = entries_by_id(trace)[plain.id]
    assert entry["label"] is None
    assert entry["value"] == 102


def test_trace_entries_keep_id_and_action():
    rt, a, _double, _plain, total = labelled_graph()
    rt.get(total)

    rt.set(a, 3)
    _, trace = rt.get_traced(total)

    for entry in trace["node_traces"]:
        assert set(entry) == {"id", "action", "label", "value"}


def test_engine_u64_node_shows_its_value():
    rt = Runtime()
    c = rt.create_collection()
    count = c.count()
    assert rt.get(count) == 0

    c.insert("x")
    c.insert("y")
    _, trace = rt.get_traced(count)

    internal = [e for e in trace["node_traces"] if e["id"] != count.id]
    assert len(internal) == 1
    assert internal[0]["value"] == 2
    assert entries_by_id(trace)[count.id]["value"] == 2
