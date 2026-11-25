"""R24: how long a checkpoint worker keeps the GIL, measured from outside.

A probe thread asks for the GIL every 200 us and records each wait and the
time the process's threads spent runnable but not scheduled during it, and
the gate records every worker hold of 1 ms or more with its site. The scenarios
drive store and memo flights with `poll` alone while both run, and every
gate site declared for a worker has to be reached by one of them.

The default run uses small sizes and checks reach only. With
`INCR_GIL_PROBE=1` the scenarios run at scale, five times each on a
machine with a core free for each of the caller, the worker and the
probe, and hold both bars: no probe wait reaches 10 ms once the caller's
own polls and the collection passes are taken off, and no gate hold
reaches 10 ms once the collection passes are taken off; every unbounded
worker sequence site must also carry 100,000 elements in one call. A wait
over the bar that the threads' wait for a CPU would explain fails as
inconclusive. A backend that keeps or wraps its returns costs one copy in
a single hold, a known limit, so those scenarios report their figures
without the bars. Set `INCR_GIL_PROBE_REPORT=<path>` to write the measured
figures as JSON."""

import gc
import json
import os
import threading
import time

import incr.persist
import pytest
from test_gil_gate_scan import declared_sites

persist = incr.persist
checks = getattr(persist, "_gil_checks", None)
pytestmark = pytest.mark.skipif(checks is None, reason="needs wheels built with gil-checks")

SCALE = bool(os.environ.get("INCR_GIL_PROBE"))
RUNS = 5 if SCALE else 1
BAR_NS = 10_000_000
UNBOUNDED_MIN = 100_000
DEADLINE = 900 if SCALE else 60


def size(full, small):
    return full if SCALE else small


KEYED_ROWS = size(3_000_000, 3_000)
PLAIN_ROWS = size(1_000_000, 1_000)
MEMO_ENTRIES = size(531_000, 2_000)
DELTA_ENTRIES = size(100_000, 400)
DEPENDENTS = size(100_000, 300)

REACHED = {}
REPORT = []
FAILURES = []


class BytesCodec:
    def encode(self, obj):
        return obj

    def decode(self, data):
        return bytes(data)


class JsonCodec:
    def encode(self, obj):
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def overlap(start, end, merged):
    return sum(max(0, min(end, b) - max(start, a)) for a, b in merged)


class Passes:
    """Every collection pass while installed, on any thread, as `(start, end)`
    on the gate's clock."""

    def __init__(self):
        self.open = {}
        self.done = []

    def __call__(self, phase, info):
        now = checks._gil_now_ns()
        if phase == "start":
            self.open[threading.get_ident()] = now
        else:
            started = self.open.pop(threading.get_ident(), None)
            if started is not None:
                self.done.append((started, now))


def probe(scenario, start, on_poll=None, expect=None, bars=True):
    """Runs the flight `start()` opens under the probe, polling with a 1 ms
    sleep between calls until it settles, its handle is freed (which starts
    a memo flight's removal of superseded generations) and no worker is
    left. Returns the flight's result, or the exception it raised when
    `expect` names its type."""
    gc.collect()
    gc.freeze()
    passes = Passes()
    gc.callbacks.append(passes)
    checks._gil_reset()
    checks._gil_probe_start(200)
    calls = []
    outcome = None
    began = checks._gil_now_ns()
    try:
        pending = start()
        calls.append((began, checks._gil_now_ns()))
        deadline = time.monotonic() + DEADLINE
        done = False
        while not done:
            assert time.monotonic() < deadline, f"{scenario}: the flight did not settle"
            time.sleep(0.001)
            asked = checks._gil_now_ns()
            if on_poll is not None:
                on_poll(pending)
            try:
                done = pending.poll()
            except Exception as error:
                if expect is None or not isinstance(error, expect):
                    raise
                outcome, done = error, True
            finally:
                calls.append((asked, checks._gil_now_ns()))
        asked = checks._gil_now_ns()
        if outcome is None:
            assert expect is None, f"{scenario}: the flight succeeded"
            outcome = pending.result()
        del pending
        calls.append((asked, checks._gil_now_ns()))
        while checks._checkpoint_workers_running():
            assert time.monotonic() < deadline, f"{scenario}: a worker outlived its flight"
            time.sleep(0.001)
        ended = checks._gil_now_ns()
    finally:
        waits = checks._gil_probe_stop()
        gc.callbacks.remove(passes)
        gc.unfreeze()
    note(scenario, began, ended, calls, passes.done, waits, bars)
    return outcome


def note(scenario, began, ended, calls, passes, waits, bars):
    holds = checks._gil_holds()
    sites = {
        name: [entered, most, longest / 1e6]
        for name, _, _, entered, most, longest in checks._gil_sites()
        if entered or most
    }
    for name, (entered, most, _) in sites.items():
        seen = REACHED.setdefault(name, [0, 0])
        seen[0] += entered
        seen[1] = max(seen[1], most)
    collections = union(passes)
    set_aside = union(calls + passes)
    wait, asked, raw, unscheduled = max(
        ((w - overlap(a, a + w, set_aside), a, w, u) for a, w, u in waits), default=(0, 0, 0, 0)
    )
    during = [
        {"site": s, "hold_ms": (e - b) / 1e6} for b, e, s in holds if b < asked + raw and e > asked
    ]
    hold, site = max(
        ((e - b - overlap(b, e, collections), s) for b, e, s in holds), default=(0, None)
    )
    REPORT.append(
        {
            "scenario": scenario,
            "flight_ms": (ended - began) / 1e6,
            "polls": len(calls),
            "probe_wait_ms": wait / 1e6,
            "probe_wait_raw_ms": raw / 1e6,
            "probe_unscheduled_ms": unscheduled / 1e6,
            "probe_wait_at_ms": (asked - began) / 1e6,
            "most_unscheduled_ms": max((u for _, _, u in waits), default=0) / 1e6,
            "holds_during_wait": during,
            "gate_hold_ms": hold / 1e6,
            "gate_hold_site": site,
            "collection_ms": sum(e - b for b, e in collections) / 1e6,
            "collections": len(passes),
            "sites": sites,
        }
    )
    write_report()
    if SCALE and bars:
        if wait >= BAR_NS and wait - unscheduled < BAR_NS:
            FAILURES.append(
                f"{scenario}: inconclusive, probe waited {wait / 1e6:.2f} ms of which "
                f"{unscheduled / 1e6:.2f} ms the process's threads were runnable but not "
                "scheduled: the machine is busy"
            )
        elif wait >= BAR_NS:
            FAILURES.append(f"{scenario}: probe waited {wait / 1e6:.2f} ms; gate holds {during}")
        if hold >= BAR_NS:
            FAILURES.append(f"{scenario}: gate held {hold / 1e6:.2f} ms at {site}")


def assert_bars(prefix):
    assert [f for f in FAILURES if f.startswith(prefix)] == []


def write_report():
    path = os.environ.get("INCR_GIL_PROBE_REPORT")
    if path:
        with open(path, "w") as out:
            json.dump(REPORT, out, indent=1)


class Delegating:
    concurrent_compaction = True

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)


class Truncating(Delegating):
    def compact_streamed(self, upto, records):
        next(iter(records))


class CountingCodec(BytesCodec):
    fail_at = None

    def __init__(self):
        self.encodes = 0

    def encode(self, obj):
        if self.fail_at is not None:
            self.encodes += 1
            if self.encodes == self.fail_at:
                raise KeyError("refused in the middle of the fold")
        return obj


def open_store(backend, codec):
    store = persist.Store.open(backend, build_id="probe")
    keyed = store.keyed_collection("keyed", codec, codec)
    rows = store.collection("rows", codec)
    return store, keyed, rows


def fill_store(store, keyed, rows, keyed_rows, plain_rows):
    with store.batch() as batch:
        for i in range(keyed_rows):
            batch.upsert(keyed, i.to_bytes(8, "little"), b"v%d" % (i % 997))
    store.extend(rows, [i.to_bytes(8, "big") for i in range(plain_rows)])


@pytest.fixture(scope="class")
def big_store():
    backend = persist.MemoryBackend()
    codec = CountingCodec()
    store, keyed, rows = open_store(backend, codec)
    fill_store(store, keyed, rows, KEYED_ROWS, PLAIN_ROWS)
    yield backend, codec, store
    store.close()


def worker_encodes():
    return next((s[3] for s in checks._gil_sites() if s[0] == "codec.encode_value"), 0)


class TestStoreFlights:
    def test_a_store_flight_over_millions_of_rows(self, big_store):
        _, _, store = big_store
        for _ in range(RUNS):
            probe("store.flight", store.begin_checkpoint)
            figures = REPORT[-1]
            records = figures["sites"]["mem.MemoryBackend.compact_streamed.records"][1]
            assert records >= KEYED_ROWS + PLAIN_ROWS
            figures["records_per_second"] = records / (figures["flight_ms"] / 1e3)
        assert_bars("store.flight")

    def test_a_store_flight_cancelled_in_the_middle_of_the_fold(self, big_store):
        _, _, store = big_store

        def cancel_halfway(pending):
            if not SCALE or worker_encodes() >= KEYED_ROWS // 2:
                pending.cancel()

        for _ in range(RUNS):
            probe("store.cancelled", store.begin_checkpoint, cancel_halfway, persist.PersistError)
        assert_bars("store.cancelled")

    def test_a_store_flight_failed_by_a_codec_in_the_middle_of_the_fold(self, big_store):
        _, codec, store = big_store
        for _ in range(RUNS):
            codec.encodes, codec.fail_at = 0, KEYED_ROWS // 2
            try:
                raised = probe("store.codec_failed", store.begin_checkpoint, expect=KeyError)
            finally:
                codec.fail_at = None
            assert "middle of the fold" in str(raised)
        assert_bars("store.codec_failed")

    def test_a_store_flight_on_a_backend_that_returns_after_one_record_poisons(self, big_store):
        backend, _, store = big_store
        store.checkpoint()
        store.close()
        for _ in range(RUNS):
            reopened, _, _ = open_store(Truncating(backend), CountingCodec())
            try:
                raised = probe(
                    "store.truncated", reopened.begin_checkpoint, expect=persist.PersistError
                )
            finally:
                reopened.close()
            assert "truncated" in str(raised)
        assert_bars("store.truncated")


class Memo:
    """The reference memo backend behind Python methods a scenario can
    switch: `refuse_reads` makes `read_entries` and `read_entry` raise and
    `keep_generations` turns `remove_generation` into a no-op."""

    concurrent_compaction = True
    hidden = ()

    def __init__(self):
        self.inner = persist.MemoryMemo()
        self.refuse_reads = False
        self.keep_generations = False
        self.kept = []

    def __getattr__(self, name):
        if name in self.hidden:
            raise AttributeError(name)
        method = getattr(self.inner, name)
        if name in ("read_entries", "read_entry"):

            def read(*args):
                if self.refuse_reads:
                    raise OSError("read refused inside the streamed pull")
                return method(*args)

            return read
        if name == "remove_generation":

            def remove(generation):
                if self.keep_generations:
                    self.kept.append(generation)
                else:
                    method(generation)

            return remove
        return method


FIRST_ITEM = (0, 0, 0, 1)


class LoadChainWithoutKnown(Memo):
    def load_chain(self):
        return self.inner.load_chain(frozenset())


class WithoutLoadChain(Memo):
    hidden = ("load_chain",)


class WithoutReadEntries(Memo):
    hidden = ("read_entries",)


class KeepsItsLists(Memo):
    def write_generation_streamed(self, *args):
        self.returned = self.inner.write_generation_streamed(*args)
        return self.returned


class ReturnsTuples(Memo):
    def write_generation_streamed(self, *args):
        return tuple(self.inner.write_generation_streamed(*args))


class MemoWorld:
    def __init__(self, backend):
        store = persist.Store.open(persist.MemoryBackend(), build_id="probe")
        base = store.input("base", 7, JsonCodec())
        width = store.input("width", 1, JsonCodec())
        big = store.memo_table(
            "big", 1, JsonCodec(), JsonCodec(), lambda ctx, k: "x" * ctx.get(width)
        )
        self.sq = store.memo_table(
            "sq", 1, JsonCodec(), JsonCodec(), lambda ctx, k: k * k + ctx.get(base)
        )
        self.dep = store.memo_table(
            "dep", 1, JsonCodec(), JsonCodec(), lambda ctx, k: len(ctx.memo(big, 0)) + k
        )
        rows = store.keyed_collection("rows", JsonCodec(), JsonCodec())
        self.row = store.memo_table(
            "row", 1, JsonCodec(), JsonCodec(), lambda ctx, k: ctx.get_key(rows, k)
        )
        self.store, self.width, self.big, self.rows = store, width, big, rows
        self.memo = store.open_memo(backend)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.store.close()

    def read_rows(self, count):
        with self.store.batch() as batch:
            for k in range(count):
                batch.upsert(self.rows, k, k)
        return self.read(self.row, range(count))

    def read(self, table, keys):
        get, entry = self.store.runtime.get, self.memo.get_or_insert
        return [get(entry(table, k)) for k in keys]

    def fresh_keys(self, count):
        start = getattr(self, "next_key", 10**9)
        self.next_key = start + count
        return range(start, start + count)

    def fill_chain(self, scenario, chains):
        """Takes small deltas until the next checkpoint has to rewrite the
        chain in full."""
        while True:
            self.read(self.sq, self.fresh_keys(10))
            if not chains or self.memo.stats()["chain_depth"] >= 8:
                return
            self.flight(scenario)

    def grow(self, width):
        self.store.set(self.width, width)
        self.read(self.big, [0])

    def flight(self, scenario, **kwargs):
        return probe(scenario, self.memo.begin_checkpoint, **kwargs)


def cancel_once_encoding(entries):
    def cancel(pending):
        if not SCALE or worker_encodes() >= entries:
            pending.cancel()

    return cancel


@pytest.mark.parametrize(
    "variant", [Memo, LoadChainWithoutKnown, WithoutLoadChain, WithoutReadEntries]
)
def test_memo_flights_from_a_full_generation_to_a_rewrite_over_its_deltas(variant):
    name = f"memo.{variant.__name__}"
    chains = variant is not WithoutLoadChain
    for _ in range(RUNS):
        backend = variant()
        with MemoWorld(backend) as world:
            world.read(world.sq, range(MEMO_ENTRIES))
            world.read_rows(100)
            if variant is Memo:
                world.flight(
                    f"{name}.cancelled_after_seal",
                    on_poll=cancel_once_encoding(MEMO_ENTRIES // 2),
                    expect=persist.PersistError,
                )
            stats = world.flight(f"{name}.full")
            assert stats["fresh_written"] == MEMO_ENTRIES + 100

            world.read(world.sq, range(MEMO_ENTRIES, MEMO_ENTRIES + DELTA_ENTRIES))
            stats = world.flight(f"{name}.delta")
            assert stats["delta"] is chains
            assert stats["fresh_written"] == DELTA_ENTRIES

            world.fill_chain(f"{name}.small_delta", chains)
            backend.refuse_reads = True
            world.flight(f"{name}.rewrite_refused", expect=OSError)
            backend.refuse_reads = False
            stats = world.flight(f"{name}.rewrite")
            assert not stats["delta"] and stats["merged"]
            assert stats["written"] >= MEMO_ENTRIES + DELTA_ENTRIES

            world.fill_chain(f"{name}.small_delta", chains)
            backend.keep_generations = True
            stats = world.flight(f"{name}.rewrite_keeping_generations")
            assert not stats["delta"] and stats["merged"]
            backend.keep_generations = False
            assert backend.kept
            assert all(backend.inner.read_entry(g, FIRST_ITEM) for g in backend.kept)
            world.read(world.sq, world.fresh_keys(10))
            world.flight(f"{name}.stale_dropped")
            assert not any(backend.inner.read_entry(g, FIRST_ITEM) for g in backend.kept)
    assert_bars(name)


def test_a_memo_delta_with_many_tombstones():
    limit = 4096
    for _ in range(RUNS):
        with MemoWorld(Memo()) as world:
            world.read(world.sq, range(MEMO_ENTRIES))
            world.read(world.dep, range(DEPENDENTS))
            world.flight("memo.tombstones.base")
            checks._set_memo_frame_limit(limit)
            try:
                world.grow(2 * limit)
                world.read(world.dep, range(DEPENDENTS))
                stats = world.flight("memo.tombstones")
            finally:
                checks._set_memo_frame_limit(None)
        assert stats["delta"]
        assert stats["tombstones"] >= DEPENDENTS
    assert_bars("memo.tombstones")


@pytest.mark.parametrize("variant", [KeepsItsLists, ReturnsTuples])
def test_a_full_generation_on_a_backend_whose_returns_incr_must_copy(variant):
    name = f"memo.{variant.__name__}"
    for _ in range(RUNS):
        backend = variant()
        with MemoWorld(backend) as world:
            world.read(world.sq, range(MEMO_ENTRIES))
            world.flight(name, bars=False)
        figures = REPORT[-1]
        figures["copy_ms"] = figures["sites"]["memo.write_generation_streamed"][2]

        with MemoWorld(backend) as restored:
            assert restored.memo.stats()["entries"] == MEMO_ENTRIES
            values = restored.read(restored.sq, range(MEMO_ENTRIES))
        assert values == [k * k + 7 for k in range(MEMO_ENTRIES)]


@pytest.mark.skipif(not SCALE, reason="set INCR_GIL_PROBE=1 for the scale run")
def test_flight_time_under_a_busy_python_caller_is_reported():
    store, keyed, rows = open_store(persist.MemoryBackend(), BytesCodec())
    fill_store(store, keyed, rows, 500_000, 0)
    with MemoWorld(persist.MemoryMemo()) as world:
        world.read(world.sq, range(90_000))
        for scenario, begin in [
            ("busy_caller.store", store.begin_checkpoint),
            ("busy_caller.memo", world.memo.begin_checkpoint),
        ]:
            started = time.monotonic()
            pending = begin()
            while not pending.poll():
                sum(i * i for i in range(20_000))
            REPORT.append({"scenario": scenario, "flight_ms": (time.monotonic() - started) * 1e3})
    store.close()
    write_report()


def test_every_worker_site_was_reached_on_a_worker():
    if not REACHED:
        pytest.skip("no scenario ran")
    sites, _ = declared_sites()
    worker = {name: kind for name, (kind, _) in sites.items() if kind != "caller"}
    assert sorted(set(worker) - set(REACHED)) == []
    if SCALE:
        short = {
            name: REACHED[name][1]
            for name, kind in worker.items()
            if kind == "unbounded" and REACHED[name][1] < UNBOUNDED_MIN
        }
        assert short == {}
