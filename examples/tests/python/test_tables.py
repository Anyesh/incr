import threading
import time

import incr
import incr_concurrent
import pytest

BOTH = pytest.mark.parametrize(
    "module", [incr, incr_concurrent], ids=["local", "concurrent"]
)


@BOTH
def test_input_entries(module):
    rt = module.Runtime()
    table = rt.create_table()
    calls = []

    def init():
        calls.append(1)
        return 7

    x = table.get_or_insert_input("x", init)
    assert table.get_or_insert_input("x", init) == x
    assert calls == [1]
    assert rt.get(x) == 7
    assert table.get("x") == x
    assert table.get("y") is None
    assert table.items() == [("x", x)]
    table.remove("x")
    assert table.get("x") is None
    assert not rt.is_alive(x)
    with pytest.raises(KeyError):
        table.remove("x")


@BOTH
def test_query_entries_memoize_a_recursion(module):
    rt = module.Runtime()
    table = rt.create_table()
    runs = []

    def fib(n):
        def body(r):
            runs.append(n)
            if n < 2:
                return n
            return r.get(fib(n - 1)) + r.get(fib(n - 2))

        return table.get_or_insert_query(n, body)

    assert rt.get(fib(30)) == 832040
    assert sorted(runs) == list(range(31))
    assert len(table.items()) == 31


@BOTH
def test_query_entries_take_a_cycle_strategy(module):
    rt = module.Runtime()
    table = rt.create_table()
    holder = {}
    holder["a"] = table.get_or_insert_query(
        "a", lambda r: r.get(holder["a"]) + 1, fallback=lambda ids: -1
    )
    assert rt.get(holder["a"]) == -1


@BOTH
def test_get_is_refused_inside_a_query(module):
    rt = module.Runtime()
    table = rt.create_table()
    table.get_or_insert_input("x", lambda: 1)
    q = rt.create_query(lambda r: table.get("x"))
    with pytest.raises(RuntimeError, match="called during compute"):
        rt.get(q)


class Key:
    """Equal by `name`, with one hash for every key, so every lookup
    compares. The `arm_at`-th comparison across all keys sharing `state`
    calls into incr: with the lock-free probe first, the second runs
    under the table's miss mutex."""

    def __init__(self, name, state):
        self.name, self.state = name, state

    def __hash__(self):
        return 0

    def __eq__(self, other):
        self.state["calls"] += 1
        if self.state["calls"] == self.state.get("arm_at"):
            self.state["rt"].create_input(0)
        return isinstance(other, Key) and self.name == other.name


@BOTH
def test_key_eq_that_calls_into_incr_raises_the_contract_error(module):
    rt = module.Runtime()
    table = rt.create_table()
    state = {"calls": 0, "rt": rt}
    table.get_or_insert_input(Key("first", state), lambda: 1)
    state["arm_at"] = state["calls"] + 2
    with pytest.raises(RuntimeError, match="must not call into the runtime"):
        table.get_or_insert_input(Key("second", state), lambda: 2)
    assert state["calls"] == state["arm_at"]
    assert [k.name for k, _ in table.items()] == ["first"]


class SlowKey:
    """One hash for every key, and an `__eq__` that sleeps, so a second
    thread's miss waits on the miss mutex while the first compares."""

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return 0

    def __eq__(self, other):
        time.sleep(0.05)
        return isinstance(other, SlowKey) and self.name == other.name


def test_a_table_miss_waits_without_the_gil():
    rt = incr_concurrent.Runtime()
    table = rt.create_table()
    table.get_or_insert_input(SlowKey("a"), lambda: 0)
    threads = [
        threading.Thread(
            target=table.get_or_insert_input, args=(SlowKey(name), lambda: 1), daemon=True
        )
        for name in "bcd"
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a miss waited holding the GIL"
    assert sorted(k.name for k, _ in table.items()) == ["a", "b", "c", "d"]
