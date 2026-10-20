import incr
import incr_concurrent
import pytest

BOTH = pytest.mark.parametrize(
    "module", [incr, incr_concurrent], ids=["local", "concurrent"]
)


@BOTH
def test_child_input_is_created_once_per_key(module):
    rt = module.Runtime()
    a = rt.create_input(1)
    inits = []
    children = []

    def init():
        inits.append(1)
        return 10

    def parent(r):
        child = r.child_input("k", init)
        children.append(child)
        return r.get(a) + r.get(child)

    p = rt.create_query(parent)
    assert rt.get(p) == 11
    rt.set(a, 2)
    assert rt.get(p) == 12
    assert inits == [1]
    assert children[0] == children[1]
    rt.set(children[0], 20)
    assert rt.get(p) == 22


@BOTH
def test_child_queries_not_asked_for_are_swept(module):
    rt = module.Runtime()
    n = rt.create_input(5)

    def parent(r):
        kids = [r.child_query(i, lambda r, i=i: i * i) for i in range(r.get(n))]
        return sum(r.get(k) for k in kids)

    p = rt.create_query(parent)
    assert rt.get(p) == 30
    before = rt.node_count()
    rt.set(n, 2)
    assert rt.get(p) == 1
    assert rt.node_count() == before - 3


@BOTH
def test_child_query_takes_a_cycle_strategy(module):
    rt = module.Runtime()
    holder = {}

    def parent(r):
        holder["c"] = r.child_query(
            "c", lambda r: r.get(holder["c"]) + 1, fallback=lambda ids: -1
        )
        return r.get(holder["c"])

    p = rt.create_query(parent)
    assert rt.get(p) == -1
