"""A Local runtime hands the GIL to other Python threads during long native
work (`Runtime(gil_yield_ms=100)`, the default), so a thread that sleeps
wakes on time while the engine reconciles, marks, restores or checkpoints.

The workloads use C-implemented values and codecs only, so no Python frame
runs inside the measured calls and the interpreter's own switching never
hands the GIL over there. `Calls` reads the clock right before and after a
call with no bytecode between, so a wake stamped inside a call's window
happened while the call was running.
"""

import functools
import gc
import operator
import os
import pickle
import sys
import threading
import time

import incr
import incr.persist as persist
import pytest

MS = 1_000_000


class CCodec:
    encode = staticmethod(functools.partial(pickle.dumps, protocol=5))
    decode = staticmethod(pickle.loads)


class Sleeper:
    def __init__(self, period):
        self.period = int(period * 1e9)
        self.wakes = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        deadline = time.monotonic_ns() + self.period
        while not self.stop.wait(max(0, deadline - time.monotonic_ns()) / 1e9):
            self.wakes.append((deadline, time.monotonic_ns()))
            deadline += self.period

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


class Calls:
    def __init__(self):
        self.windows = []

    def __call__(self, f, *args):
        start, result, end = map(
            operator.methodcaller("__call__"),
            (time.monotonic_ns, functools.partial(f, *args), time.monotonic_ns),
        )
        self.windows.append((start, end))
        return result

    def busy_ns(self):
        return sum(end - start for start, end in self.windows)

    def wakes_during(self, sleeper):
        return [
            (deadline, woke)
            for deadline, woke in sleeper.wakes
            if any(start < deadline < end for start, end in self.windows)
        ]

    def wakes_inside(self, sleeper):
        return [
            sum(1 for _, woke in sleeper.wakes if start < woke < end)
            for start, end in self.windows
        ]


def until(seconds, timed):
    rounds = 0
    while rounds < 2 or timed.busy_ns() < seconds * 1e9:
        yield rounds
        rounds += 1


def keyed_reconcile(options, n, seconds, timed):
    rt = incr.Runtime(**options)
    keyed = incr.KeyedCollection(rt)
    keyed_count = keyed.collection.count()
    plain = rt.create_collection()
    tables = [{k: (k, j) for k in range(n)} for j in range(2)]
    rows = [(k, k * 7) for k in range(n)]
    results = []
    for rounds in until(seconds, timed):
        results.append(timed(keyed.reconcile, tables[rounds % 2]))
        results.append(timed(rt.get, keyed_count))
        timed(plain.extend, rows)
        results.append(timed(rt.get, timed(plain.count)))
    return results


def fan_out(options, n, seconds, timed):
    rt = incr.Runtime(**options)
    a = rt.create_input(0)
    cut = rt.create_query(lambda r: r.get(a) < 0)

    def read(r):
        return r.get(cut)

    reads = [rt.create_query(read) for _ in range(n)]
    total = rt.create_query(lambda r: sum(map(r.get, reads)))
    results = [rt.get(total)]
    for rounds in until(seconds, timed):
        timed(rt.set, a, rounds + 1)
        results.append(rt.get(cut))
        results.append(timed(rt.get, total))
    return results


def seeded_store(options, n):
    log = persist.MemoryBackend()
    store = persist.Store.open(log, build_id=1, **options)
    keyed = store.keyed_collection("rows", CCodec(), CCodec())
    store.reconcile(keyed, {k: (k, 0) for k in range(n)})
    store.checkpoint()
    store.reconcile(keyed, {k: (k, 1) for k in range(0, n, 2)})
    store.close()
    return log


def store_restore(options, n, seconds, timed):
    log = seeded_store(options, n)
    open_store = functools.partial(persist.Store.open, log, build_id=1, **options)
    results = []
    for _ in until(seconds, timed):
        store = timed(open_store)
        keyed = timed(store.keyed_collection, "rows", CCodec(), CCodec())
        results.append(sorted(keyed.items()))
        timed(store.checkpoint)
        store.close()
    return results


def declare_memo(store, n):
    item = store.memo_table("item", 1, CCodec(), CCodec(), lambda ctx, k: (k, k * 3))
    return store.memo_table(
        "total",
        1,
        CCodec(),
        CCodec(),
        lambda ctx, k: sum(ctx.memo(item, i)[1] for i in range(n)),
    )


def memo_restore(options, n, seconds, timed):
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    store = persist.Store.open(log, build_id=1, **options)
    total = declare_memo(store, n)
    memo = store.open_memo(memo_backend)
    store.runtime.get(memo.get_or_insert(total, 0))
    timed(memo.checkpoint)
    store.close()
    results = []
    for _ in until(seconds, timed):
        store = persist.Store.open(log, build_id=1, **options)
        total = declare_memo(store, n)
        memo = store.open_memo(memo_backend)
        node = timed(memo.get_or_insert, total, 0)
        results.append(timed(store.runtime.get, node))
        store.close()
    return results


# Sized so that calls run from a few hundred ms to a few seconds each.
LARGE = {
    keyed_reconcile: 400_000,
    fan_out: 2_500_000,
    store_restore: 300_000,
    memo_restore: 300_000,
}

SMALL = {
    keyed_reconcile: 20_000,
    fan_out: 200_000,
    store_restore: 10_000,
    memo_restore: 10_000,
}

WORKLOADS = [
    pytest.param(keyed_reconcile, id="keyed_reconcile_and_count"),
    pytest.param(fan_out, id="dirty_marking_fan_out"),
    pytest.param(store_restore, id="store_restore_and_checkpoint"),
    pytest.param(memo_restore, id="memo_restore"),
]

# At the default 100 ms the hook yields at its first check 100 ms after the
# last yield, so a call this long hands the GIL over at least once.
LONG_CALL = 250 * MS


@pytest.mark.parametrize("workload", WORKLOADS)
def test_a_long_native_call_lets_another_thread_run(workload):
    calls = Calls()
    with Sleeper(0.002) as sleeper:
        workload({}, LARGE[workload], 0, calls)
    long_calls = [
        (end - start, inside)
        for (start, end), inside in zip(calls.windows, calls.wakes_inside(sleeper))
        if end - start >= LONG_CALL
    ]
    assert long_calls, "no call ran 250 ms; raise the workload's size"
    held = [duration // MS for duration, inside in long_calls if inside == 0]
    assert not held, f"calls of {held} ms never let the sleeper run"


@pytest.fixture
def fast_switching():
    interval = sys.getswitchinterval()
    sys.setswitchinterval(0.0001)
    yield
    sys.setswitchinterval(interval)


@pytest.mark.parametrize("workload", WORKLOADS)
def test_results_equal_with_the_hook_off_and_yielding_every_ms(
    workload, fast_switching
):
    off, every_ms = Calls(), Calls()
    with Sleeper(0.001) as sleeper:
        expected = workload({"gil_yield_ms": None}, SMALL[workload], 0, off)
    assert sum(off.wakes_inside(sleeper)) == 0
    with Sleeper(0.001) as sleeper:
        got = workload({"gil_yield_ms": 1}, SMALL[workload], 0, every_ms)
    assert sum(every_ms.wakes_inside(sleeper)) > 0
    assert got == expected


# Yielding at every check would release and retake the GIL every few
# microseconds, and CPython lets a waiting thread ask for the GIL only
# when no switch happened during its wait, so it would starve.
@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: incr.Runtime(gil_yield_ms=0), id="runtime"),
        pytest.param(
            lambda: persist.Store.open(
                persist.MemoryBackend(), build_id=1, gil_yield_ms=0
            ),
            id="store",
        ),
    ],
)
def test_a_zero_interval_is_refused(make):
    with pytest.raises(ValueError, match="gil_yield_ms"):
        make()


# A cyclic collection over the millions of objects the workloads build
# holds the GIL for its whole pass, which is CPython's pause, not incr's.
@pytest.fixture
def no_cyclic_gc():
    gc.collect()
    gc.freeze()
    gc.disable()
    yield
    gc.enable()
    gc.unfreeze()


@pytest.mark.skipif(
    not os.environ.get("INCR_TIMING_TESTS"),
    reason="5 s of work per form against a wall-clock bar; set INCR_TIMING_TESTS=1",
)
@pytest.mark.parametrize("workload", WORKLOADS)
def test_a_sleeping_thread_wakes_within_200_ms_during_5_s_of_native_work(
    workload, no_cyclic_gc
):
    calls = Calls()
    with Sleeper(1.0) as sleeper:
        workload({}, LARGE[workload], 5.0, calls)
    during = calls.wakes_during(sleeper)
    assert len(during) >= 4
    late = [(woke - deadline) // MS for deadline, woke in during]
    assert max(late) <= 200, f"wakes late by {late} ms"
