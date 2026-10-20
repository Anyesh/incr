import incr
import incr_concurrent
import pytest

BOTH = pytest.mark.parametrize(
    "module", [incr, incr_concurrent], ids=["local", "concurrent"]
)


@BOTH
def test_try_get_returns_the_value_or_raises_cycle_error(module):
    rt = module.Runtime()
    a = rt.create_input(3)
    assert rt.try_get(a) == 3
    holder = {}
    q = rt.create_query(lambda r: r.get(holder["q"]) + 1)
    holder["q"] = q
    with pytest.raises(module.CycleError) as raised:
        rt.try_get(q)
    _, ids = raised.value.args
    assert ids == [q.id]


@BOTH
def test_try_get_in_a_query_is_an_ordinary_exception(module):
    rt = module.Runtime()
    nodes = {}

    def top(r):
        try:
            return r.try_get(nodes["f"])
        except module.CycleError:
            return 0

    nodes["t"] = rt.create_query(top)
    nodes["f"] = rt.create_query(lambda r: r.get(nodes["u"]) + 1)
    nodes["u"] = rt.create_query(lambda r: r.get(nodes["f"]) + 1)
    assert rt.get(nodes["t"]) == 0
    with pytest.raises(module.CycleError):
        rt.try_get(nodes["f"])


@BOTH
def test_fallback_gives_each_member_its_value(module):
    rt = module.Runtime()
    nodes = {}
    seen = []

    def fallback(ids):
        seen.append(sorted(ids))
        return -1

    nodes["a"] = rt.create_query(lambda r: r.get(nodes["b"]) + 1, fallback=fallback)
    nodes["b"] = rt.create_query(lambda r: r.get(nodes["a"]) + 1, fallback=fallback)
    assert rt.get(nodes["a"]) == -1
    assert rt.get(nodes["b"]) == -1
    assert sorted([nodes["a"].id, nodes["b"].id]) in seen


@BOTH
def test_fixpoint_iterates_to_a_fixed_point(module):
    rt = module.Runtime()
    nodes = {}
    fix = module.Fixpoint(lambda: 0)
    nodes["a"] = rt.create_query(lambda r: min(r.get(nodes["b"]) + 1, 5), fixpoint=fix)
    nodes["b"] = rt.create_query(lambda r: r.get(nodes["a"]), fixpoint=fix)
    assert rt.get(nodes["a"]) == 5
    assert rt.get(nodes["b"]) == 5


@BOTH
def test_fixpoint_callbacks(module):
    rt = module.Runtime()
    nodes = {}
    steps, limits = [], []

    def step(prev, out, k):
        steps.append(k)
        return out

    def on_limit(ids, value):
        limits.append(sorted(ids))
        return -value

    fix = module.Fixpoint(
        lambda: 0,
        converged=lambda prev, cur: prev == cur,
        step=step,
        max_iterations=3,
        on_limit=on_limit,
    )
    nodes["a"] = rt.create_query(lambda r: r.get(nodes["b"]) + 1, fixpoint=fix)
    nodes["b"] = rt.create_query(lambda r: r.get(nodes["a"]) + 1, fixpoint=fix)
    assert rt.get(nodes["a"]) < 0
    assert min(steps) == 1 and max(steps) == 3
    assert sorted([nodes["a"].id, nodes["b"].id]) in limits


@BOTH
def test_strategy_arguments_are_checked(module):
    rt = module.Runtime()
    fix = module.Fixpoint(lambda: 0)
    with pytest.raises(ValueError):
        rt.create_query(lambda r: 0, fallback=lambda ids: 0, fixpoint=fix)
    with pytest.raises(ValueError):
        module.Fixpoint(lambda: 0, max_iterations=0)


@BOTH
def test_strategy_callback_exception_is_raised_as_itself(module):
    rt = module.Runtime()
    nodes = {}
    nodes["a"] = rt.create_query(
        lambda r: r.get(nodes["b"]), fallback=lambda ids: 1 // 0
    )
    nodes["b"] = rt.create_query(
        lambda r: r.get(nodes["a"]), fallback=lambda ids: 1 // 0
    )
    with pytest.raises(ZeroDivisionError):
        rt.get(nodes["a"])
