"""Overlay cost at the L-tenant shape: workers with shifts, a per-worker
prefix chain and a per-day travel chain, watched per-worker totals.

Usage: python bench_overlay.py [--persisted] [workers] [shifts_per_worker] [candidates]

With --persisted the same shape also runs with the shifts in a persisted
collection on incr.persist.MemoryBackend with a memo table open, and the
numbers are printed beside the in-memory ones.
"""

import pickle
import random
import statistics
import sys
import time

import incr
import incr.persist as persist


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


def percentile(samples, q):
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


class InMemory:
    def __init__(self):
        self.rt = incr.Runtime()
        self.shifts = self.rt.create_collection()

    def extend(self, rows):
        self.shifts.extend(rows)

    def insert(self, row):
        self.shifts.insert(row)

    def delete(self, row):
        self.shifts.delete(row)

    def close(self):
        self.rt.close()


class Persisted:
    def __init__(self):
        self.store = persist.Store.open(persist.MemoryBackend(), build_id=1)
        self.rt = self.store.runtime
        self.shifts = self.store.collection("shifts", PickleCodec())
        self.table = self.store.memo_table(
            "scaled", 1, PickleCodec(), PickleCodec(), lambda ctx, key: key * 2
        )
        self.memo = self.store.open_memo(persist.MemoryMemo())
        self.rt.get(self.memo.get_or_insert(self.table, 1))

    def extend(self, rows):
        self.store.extend(self.shifts, rows)

    def insert(self, row):
        self.store.insert(self.shifts, row)

    def delete(self, row):
        self.store.delete(self.shifts, row)

    def close(self):
        self.store.close()


def candidate_row(rng, w, per_worker):
    return (
        w,
        w * 1_000_000 + rng.randrange(per_worker * 10) * 10 + 5,
        rng.randrange(1, 9),
    )


def measure(tenant, workers, per_worker, candidates, rng):
    rt = tenant.rt
    rows = [
        (w, w * 1_000_000 + s * 10, rng.randrange(1, 9))
        for w in range(workers)
        for s in range(per_worker)
    ]
    tenant.extend(rows)
    by_worker = tenant.shifts.group_by(lambda r: r[0])

    started = time.perf_counter()
    totals = {}
    for w in range(workers):
        group = by_worker.get_group(w)
        ordered = group.sort_by_key(lambda r: r[1])
        prefix = ordered.pairwise().aggregate(
            0, lambda p: p[1][1] - p[0][1], lambda a, b: a + b
        )
        hours = group.aggregate(0, lambda r: r[2], lambda a, b: a + b)
        totals[w] = (prefix, hours)
    for prefix, hours in totals.values():
        rt.get(prefix)
        rt.get(hours)
        rt.watch(hours)
    rt.drain_dirtied_watched()
    built = time.perf_counter() - started

    overlay_times = []
    for _ in range(candidates):
        w = rng.randrange(workers)
        row = candidate_row(rng, w, per_worker)
        started = time.perf_counter()
        with rt.overlay() as ov:
            tenant.shifts.insert(row)
            ov.get(totals[w][0])
            ov.get(totals[w][1])
        overlay_times.append((time.perf_counter() - started) * 1000)
    assert rt.drain_dirtied_watched() == []

    waits = []
    for _ in range(candidates):
        w = rng.randrange(workers)
        row = candidate_row(rng, w, per_worker)
        with rt.overlay() as ov:
            tenant.shifts.insert(row)
            ov.get(totals[w][1])
        event = (w, w * 1_000_000 + 7, 1)
        started = time.perf_counter()
        tenant.insert(event)
        rt.get(totals[w][1])
        waits.append((time.perf_counter() - started) * 1000)
        tenant.delete(event)
        rt.get(totals[w][1])
    return len(rows), built, overlay_times, waits


def report(label, workers, result):
    shifts, built, overlay_times, waits = result
    print(f"[{label}] built {shifts} shifts, {workers} workers in {built:.1f}s")
    print(
        "[%s] overlay ms p50 %.3f p95 %.3f mean %.3f"
        % (
            label,
            statistics.median(overlay_times),
            percentile(overlay_times, 0.95),
            statistics.fmean(overlay_times),
        )
    )
    print(
        "[%s] event after overlay ms p50 %.3f p95 %.3f"
        % (label, statistics.median(waits), percentile(waits, 0.95))
    )


def main():
    args = [a for a in sys.argv[1:] if a != "--persisted"]
    run_persisted = len(args) != len(sys.argv) - 1
    workers = int(args[0]) if len(args) > 0 else 2000
    per_worker = int(args[1]) if len(args) > 1 else 120
    candidates = int(args[2]) if len(args) > 2 else 300

    flavors = [("in-memory", InMemory)]
    if run_persisted:
        flavors.append(("persisted", Persisted))
    for label, flavor in flavors:
        tenant = flavor()
        result = measure(tenant, workers, per_worker, candidates, random.Random(1))
        report(label, workers, result)
        tenant.close()


if __name__ == "__main__":
    main()
