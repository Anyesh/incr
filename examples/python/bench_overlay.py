"""Overlay cost at the L-tenant shape: workers with shifts, a per-worker
prefix chain and a per-day travel chain, watched per-worker totals.

Usage: python bench_overlay.py [workers] [shifts_per_worker] [candidates]
"""

import random
import statistics
import sys
import time

import incr


def percentile(samples, q):
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    per_worker = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    candidates = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    rng = random.Random(1)

    rt = incr.Runtime()
    shifts = rt.create_collection()
    rows = [
        (w, w * 1_000_000 + s * 10, rng.randrange(1, 9))
        for w in range(workers)
        for s in range(per_worker)
    ]
    shifts.extend(rows)
    by_worker = shifts.group_by(lambda r: r[0])

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
    print(f"built {len(rows)} shifts, {workers} workers in {time.perf_counter() - started:.1f}s")

    overlay_times = []
    for _ in range(candidates):
        w = rng.randrange(workers)
        row = (w, w * 1_000_000 + rng.randrange(per_worker * 10) * 10 + 5, rng.randrange(1, 9))
        started = time.perf_counter()
        with rt.overlay() as ov:
            shifts.insert(row)
            ov.get(totals[w][0])
            ov.get(totals[w][1])
        overlay_times.append((time.perf_counter() - started) * 1000)
    assert rt.drain_dirtied_watched() == []

    waits = []
    for _ in range(candidates):
        w = rng.randrange(workers)
        row = (w, w * 1_000_000 + rng.randrange(per_worker * 10) * 10 + 5, rng.randrange(1, 9))
        started = time.perf_counter()
        with rt.overlay() as ov:
            shifts.insert(row)
            ov.get(totals[w][1])
        event_at = time.perf_counter()
        shifts.insert((w, w * 1_000_000 + 7, 1))
        rt.get(totals[w][1])
        waits.append((time.perf_counter() - event_at) * 1000)
        shifts.delete((w, w * 1_000_000 + 7, 1))
        rt.get(totals[w][1])
    print(
        "overlay ms p50 %.3f p95 %.3f mean %.3f"
        % (
            statistics.median(overlay_times),
            percentile(overlay_times, 0.95),
            statistics.fmean(overlay_times),
        )
    )
    print(
        "event after overlay ms p50 %.3f p95 %.3f"
        % (statistics.median(waits), percentile(waits, 0.95))
    )


if __name__ == "__main__":
    main()
