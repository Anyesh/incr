import incr
import incr_concurrent
import pytest

BOTH = pytest.mark.parametrize(
    "module", [incr, incr_concurrent], ids=["local", "concurrent"]
)


@BOTH
def test_a_change_reports_a_watched_node_once(module):
    rt = module.Runtime()
    a = rt.create_input(1)
    q = rt.create_query(lambda r: r.get(a) + 1)
    assert rt.get(q) == 2
    rt.watch(q)
    assert rt.drain_dirtied_watched() == []
    rt.set(a, 2)
    drained = rt.drain_dirtied_watched()
    assert drained == [q]
    assert rt.get(drained[0]) == 3
    assert rt.drain_dirtied_watched() == []


@BOTH
def test_a_node_without_a_value_is_reported_when_watched(module):
    rt = module.Runtime()
    a = rt.create_input(1)
    q = rt.create_query(lambda r: r.get(a) * 10)
    rt.watch(q)
    assert rt.drain_dirtied_watched() == [q]


@BOTH
def test_unwatch_drops_the_pending_report_and_later_ones(module):
    rt = module.Runtime()
    a = rt.create_input(1)
    q = rt.create_query(lambda r: r.get(a) + 1)
    p = rt.create_query(lambda r: r.get(a) + 2)
    rt.get(q)
    rt.get(p)
    rt.watch(q)
    rt.watch(p)
    rt.set(a, 2)
    rt.unwatch(q)
    assert rt.drain_dirtied_watched() == [p]
    rt.get(q)
    rt.get(p)
    rt.set(a, 3)
    assert rt.drain_dirtied_watched() == [p]


@BOTH
def test_a_deleted_watched_node_is_not_reported(module):
    rt = module.Runtime()
    a = rt.create_input(1)
    q = rt.create_query(lambda r: r.get(a) + 1)
    rt.get(q)
    rt.watch(q)
    rt.set(a, 2)
    rt.delete_node(q)
    assert rt.drain_dirtied_watched() == []
