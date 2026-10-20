"""Python benches for `incr` (Local) and `incr_concurrent` (Shared).

    python examples/benches/python/bench.py [--runs-scale X] [--only NAME ...]

Each line is one workload on one module: the median and p99 of its timed
operation in milliseconds, over the runs listed beside it. Every run checks
the value it computed, so a bench never times a wrong answer.
"""

import argparse
import math
import statistics
import time
from dataclasses import dataclass


@dataclass
class Result:
    runs: int
    median_ms: float
    p99_ms: float


def measure(op, runs, check, setup=None):
    """Times `op` `runs` times. With `setup`, each run calls it untimed
    and passes its result to `op`."""
    times = []
    for _ in range(runs):
        state = setup() if setup is not None else None
        start = time.perf_counter_ns()
        value = op(state) if setup is not None else op()
        times.append(time.perf_counter_ns() - start)
        if not check(value):
            raise AssertionError(f"bench computed a wrong value: {value!r}")
    times.sort()
    p99 = times[max(0, math.ceil(0.99 * len(times)) - 1)]
    return Result(runs, statistics.median(times) / 1e6, p99 / 1e6)


class Chains:
    """`chains` inputs, each under a chain of `depth` queries, so reading a
    chain's end nests `depth` levels: 100 by default, well under CPython's
    recursion limit."""

    def __init__(self, module, chains=1000, depth=100):
        self.rt = module.Runtime()
        self.ends = []
        for head in range(chains):
            end = self.rt.create_input(head)
            for _ in range(depth):
                end = self.rt.create_query(lambda r, below=end: r.get(below) + 1)
            self.ends.append(end)
        self.expected = sum(range(chains)) + chains * depth

    def first_compute(self):
        return sum(self.rt.get(end) for end in self.ends)


class RunningSum:
    """s_k = s_(k-1) + a_k over inputs a_k = k. A top-down first read would
    nest `n` levels natively and in Python, so the first compute reads the
    sums in ascending order, each a one-level compute."""

    def __init__(self, module, n=100_000):
        self.rt = module.Runtime()
        self.inputs = [self.rt.create_input(k) for k in range(1, n + 1)]
        self.sums = []
        below = None
        for a in self.inputs:
            if below is None:
                s = self.rt.create_query(lambda r, a=a: r.get(a))
            else:
                s = self.rt.create_query(lambda r, a=a, b=below: r.get(b) + r.get(a))
            self.sums.append(s)
            below = s
        self.values = list(range(1, n + 1))
        self.expected = n * (n + 1) // 2

    def first_compute(self):
        for s in self.sums:
            self.rt.get(s)
        return self.rt.get(self.sums[-1])

    def _edit(self, index):
        self.values[index] += 1
        self.expected += 1
        self.rt.set(self.inputs[index], self.values[index])
        return self.rt.get(self.sums[-1])

    def edit_head(self):
        return self._edit(0)

    def edit_last(self):
        return self._edit(-1)


class PrefixSumDag:
    """Node k reads nodes 0 .. k-1 in ascending order and computes
    x + max of them, so node k is (k + 1) * x and a set of x changes every
    node. Built by ascending reads, as T1's test of the same DAG is, because
    a top-down first compute through Python closures would nest `n` Python
    frames. The timed operation is the walk and recompute after a set."""

    def __init__(self, module, n=2000):
        self.rt = module.Runtime()
        self.n = n
        self.x_value = 1
        self.x = self.rt.create_input(self.x_value)
        nodes = []
        for _ in range(n):

            def node(r, below=tuple(nodes), x=self.x):
                best = 0
                for s in below:
                    best = max(best, r.get(s))
                return r.get(x) + best

            q = self.rt.create_query(node)
            nodes.append(q)
            self.rt.get(q)
        self.top = nodes[-1]

    @property
    def expected(self):
        return self.x_value * self.n

    def last(self):
        return self.rt.get(self.top)

    def set_base(self):
        self.x_value += 1
        self.rt.set(self.x, self.x_value)
        return self.rt.get(self.top)


def _never():
    raise AssertionError("every key is present")


class TableHit:
    """`get_or_insert_input` on one of `keys` present keys, in turn, so the
    probe is not served from one bucket."""

    def __init__(self, module, keys=1024):
        self.rt = module.Runtime()
        self.table = self.rt.create_table()
        self.keys = keys
        self.next = 0
        for k in range(keys):
            self.table.get_or_insert_input(k, lambda k=k: k)

    def hit(self):
        k = self.next
        self.next = (k + 1) % self.keys
        return self.table.get_or_insert_input(k, _never)


class TableMiss:
    """`get_or_insert_input` of a key not yet present, which creates its
    input node; the table grows by one entry per run."""

    def __init__(self, module):
        self.rt = module.Runtime()
        self.table = self.rt.create_table()
        self.next = 0

    def miss(self):
        k = self.next
        self.next = k + 1
        return self.table.get_or_insert_input(k, lambda: k)


class ChildrenRerun:
    """A creator with `n` child inputs whose input moves every run: the
    rerun finds every child and the sweep deletes nothing."""

    def __init__(self, module, n=1000):
        self.rt = module.Runtime()
        self.x = self.rt.create_input(0)
        self.value = 0

        def creator(r):
            v = r.get(self.x)
            for k in range(n):
                r.child_input(k, lambda k=k: k)
            return v

        self.creator = self.rt.create_query(creator)
        self.rt.get(self.creator)

    @property
    def expected(self):
        return self.value

    def rerun(self):
        self.value += 1
        self.rt.set(self.x, self.value)
        return self.rt.get(self.creator)


class ChildrenChurn:
    """A creator whose children are the 16 keys from its input up: each run
    moves the window by one, so the rerun creates one child and the sweep
    deletes one."""

    def __init__(self, module):
        self.rt = module.Runtime()
        self.x = self.rt.create_input(0)
        self.value = 0

        def creator(r):
            v = r.get(self.x)
            for k in range(v, v + 16):
                r.child_input(k, lambda k=k: k)
            return v

        self.creator = self.rt.create_query(creator)
        self.rt.get(self.creator)

    @property
    def expected(self):
        return self.value

    def churn(self):
        self.value += 1
        self.rt.set(self.x, self.value)
        return self.rt.get(self.creator)


def _scaled(runs, scale):
    return max(1, round(runs * scale))


def _first_compute(workload, runs):
    def run(module, scale=1.0):
        built = []

        def setup():
            built.append(workload(module))
            return built[-1]

        return measure(
            lambda w: w.first_compute(),
            _scaled(runs, scale),
            lambda v: v == built[-1].expected,
            setup,
        )

    return run


def _steady(workload, method, runs):
    def run(module, scale=1.0):
        w = workload(module)
        if hasattr(w, "first_compute"):
            w.first_compute()
        op = getattr(w, method)
        return measure(op, _scaled(runs, scale), lambda v: v == w.expected)

    return run


def _handles(workload, method, runs):
    def run(module, scale=1.0):
        w = workload(module)
        return measure(
            getattr(w, method), _scaled(runs, scale), lambda v: v is not None
        )

    return run


BENCHES = [
    ("chains_1000x100_first_compute", _first_compute(Chains, 5)),
    ("running_sum_100k_first_compute", _first_compute(RunningSum, 5)),
    ("running_sum_100k_edit_head", _steady(RunningSum, "edit_head", 10)),
    ("running_sum_100k_edit_last", _steady(RunningSum, "edit_last", 200)),
    ("prefix_sum_dag_2k_set_base", _steady(PrefixSumDag, "set_base", 5)),
    ("table_hit", _handles(TableHit, "hit", 2000)),
    ("table_miss", _handles(TableMiss, "miss", 2000)),
    ("children_rerun_1000", _steady(ChildrenRerun, "rerun", 200)),
    ("children_churn", _steady(ChildrenChurn, "churn", 2000)),
]


def main():
    import incr
    import incr_concurrent

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="run only these benches")
    parser.add_argument(
        "--runs-scale", type=float, default=1.0, help="multiply every run count"
    )
    args = parser.parse_args()
    for name, bench in BENCHES:
        if args.only and name not in args.only:
            continue
        for module in (incr, incr_concurrent):
            result = bench(module, args.runs_scale)
            print(
                f"{name:34} {module.__name__:16} runs {result.runs:5}"
                f"  median {result.median_ms:10.4f} ms  p99 {result.p99_ms:10.4f} ms",
                flush=True,
            )


if __name__ == "__main__":
    main()
