"""GIL release gates for the Local binding: how a second Python thread and
the engine thread affect each other. Run with and without a build that
detaches and compare.

Usage: python bench_gil.py
"""

import statistics
import threading
import time

import incr


class Spinner:
    """A second Python thread: `light=False` never releases the GIL on its
    own, `light=True` works in short bursts between sleeps, as a request
    handler does."""

    light = False

    def __init__(self):
        self.count = 0
        self.stop = False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        count = 0
        while not self.stop:
            count += 1
            if Spinner.light and count % 2000 == 0:
                time.sleep(0.0001)
        self.count = count

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop = True
        self.thread.join()


def fanout_writer(fanout=20000, sets=60):
    rt = incr.Runtime()
    a = rt.create_input(0)
    nodes = [rt.create_query(lambda r: r.get(a) + 1) for _ in range(fanout)]
    for n in nodes[::50]:
        rt.get(n)
    for n in nodes:
        rt.get(n)
    start = time.perf_counter()
    for i in range(1, sets + 1):
        rt.set(a, i)
    return time.perf_counter() - start, rt, nodes


def gate_b():
    alone, _, _ = fanout_writer()
    with Spinner() as spin:
        busy, _, _ = fanout_writer()
    return alone, busy, spin.count


def gate_c(queries=300, rounds=5):
    rt = incr.Runtime()
    a = rt.create_input(0)
    nodes = [rt.create_query(lambda r: r.get(a) + 1) for _ in range(queries)]
    top = rt.create_query(lambda r: sum(r.get(n) for n in nodes))
    rt.get(top)

    def work():
        start = time.perf_counter()
        for i in range(1, rounds + 1):
            rt.set(a, i)
            rt.get(top)
        return time.perf_counter() - start

    alone = work()
    with Spinner():
        busy = work()
    return alone, busy


def latency(samples=2000):
    rt = incr.Runtime()
    a = rt.create_input(0)
    q = rt.create_query(lambda r: r.get(a) + 1)
    rt.get(q)
    box = rt.create_collection()
    box.extend(range(1000))

    def measure():
        gets, sets, applies = [], [], []
        for i in range(samples):
            t = time.perf_counter_ns()
            rt.get(q)
            gets.append(time.perf_counter_ns() - t)
            t = time.perf_counter_ns()
            rt.set(a, i)
            sets.append(time.perf_counter_ns() - t)
            rt.get(q)
            t = time.perf_counter_ns()
            box.insert(10_000 + i)
            applies.append(time.perf_counter_ns() - t)
        return [
            (statistics.median(x) / 1000, sorted(x)[int(len(x) * 0.99)] / 1000)
            for x in (gets, sets, applies)
        ]

    alone = measure()
    with Spinner():
        busy = measure()
    return alone, busy


def main():
    alone, busy, progress = gate_b()
    print(
        f"b fanout set x60: alone {alone * 1000:.1f} ms, with spinner {busy * 1000:.1f} ms, spinner iterations {progress}"
    )
    alone, busy = gate_c()
    print(
        f"c closure-heavy: alone {alone * 1000:.1f} ms, with spinner {busy * 1000:.1f} ms"
    )
    alone, busy = latency()
    for name, a, b in zip(("clean get", "small set", "insert"), alone, busy):
        print(
            f"d/e {name}: alone p50 {a[0]:.2f} p99 {a[1]:.2f} us, with spinner p50 {b[0]:.2f} p99 {b[1]:.2f} us"
        )


if __name__ == "__main__":
    for light in (False, True):
        Spinner.light = light
        print("second thread:", "light bursts" if light else "busy loop")
        main()
