"""End to end run of the mycareshyft round 2 features on one package.

On a MemoryBackend it reconciles KEYS rows into a persisted keyed
collection, creates one reader per key with ctx.get_key, opens a memo whose
entries read their key and has checkpoint_after_entries set, applies units of
work as batch(apply="on_write") blocks that read a dependent mid-block,
checkpoints the log and the memo, closes, reopens, and checks the values,
that no memo entry ran again, and the stores' stats. With
--package incr_concurrent, --readers threads read during the units. It ends
with graph_summary and a bounded graph_snapshot. Prints each step's time.

Usage: python mcs_round2_e2e.py --package incr|incr_concurrent [--keys 240000]
"""

import argparse
import importlib
import pickle
import random
import threading
import time

BUMP = 1_000_000
KEYED_LABEL = "shifts"


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class Timings:
    def __init__(self):
        self.rows = []
        self.mark = time.perf_counter()

    def lap(self, name):
        now = time.perf_counter()
        self.rows.append((name, now - self.mark))
        self.mark = time.perf_counter()

    def report(self):
        width = max(len(name) for name, _ in self.rows)
        for name, seconds in self.rows:
            print(f"  {name:<{width}}  {seconds:9.3f} s")


class Graph:
    """A store over `log` with the keyed collection `shifts` and the memo
    table `price`, whose entry for a key is twice its row."""

    def __init__(self, persist, log, memo_backend, memo_options):
        self.store = persist.Store.open(log, build_id="r2-e2e")
        self.rt = self.store.runtime
        self.kc = self.store.keyed_collection(
            KEYED_LABEL, PickleCodec(), PickleCodec(), label=KEYED_LABEL
        )
        self.reader_runs = 0
        self.memo_runs = 0

        def price(ctx, key):
            self.memo_runs += 1
            return ctx.get_key(self.kc, key) * 2

        self.table = self.store.memo_table(
            "price", 1, PickleCodec(), PickleCodec(), price
        )
        self.memo = self.store.open_memo(memo_backend, **memo_options)

    def reader(self, key):
        def read(ctx):
            self.reader_runs += 1
            return ctx.get_key(self.kc, key)

        return read

    def entry(self, key):
        return self.memo.get_or_insert(self.table, key)

    def price(self, key):
        return self.rt.get(self.entry(key))


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAILED: {message}")


def run_units(graph, args, queries, units, changed):
    """Each unit upserts `per_unit` keys inside an on_write block and reads
    the dependents of the first one before the block ends."""
    appended = 0
    for unit in range(units):
        keys = [(unit * args.per_unit + i) % args.keys for i in range(args.per_unit)]
        before = graph.store.stats()["position"]
        with graph.store.batch(apply="on_write") as b:
            for key in keys:
                b.upsert(graph.kc, key, key + BUMP)
                changed.add(key)
            check(
                graph.rt.get(queries[keys[0]]) == keys[0] + BUMP,
                "a query did not see the write made earlier in its block",
            )
            check(
                graph.price(keys[0]) == (keys[0] + BUMP) * 2,
                "a memo entry is stale mid-block",
            )
        appended += graph.store.stats()["position"] - before
    return appended


def reader_threads(graph, args, queries, stop, failures, counts, transient):
    """A row replaced inside a block is logged as a removal then an insert,
    which a Shared reader can see between the two, so None is counted, not
    refused."""

    def loop(seed):
        rng = random.Random(seed)
        reads = 0
        try:
            while not stop.is_set():
                key = rng.randrange(args.keys)
                row = graph.rt.get(queries[key])
                if row is None:
                    transient[0] += 1
                elif row not in (key, key + BUMP):
                    failures.append((key, row))
                    return
                reads += 1
        except Exception as error:
            failures.append(error)
        counts.append(reads)

    return [threading.Thread(target=loop, args=(seed,)) for seed in range(args.readers)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, choices=["incr", "incr_concurrent"])
    parser.add_argument("--keys", type=int, default=240_000)
    parser.add_argument("--units", type=int, default=20)
    parser.add_argument("--per-unit", type=int, default=5)
    parser.add_argument("--readers", type=int, default=4)
    args = parser.parse_args()
    args.keys = max(args.keys, args.units * args.per_unit)

    persist = importlib.import_module(f"{args.package}.persist")
    shared = args.package == "incr_concurrent"
    times = Timings()
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    trigger = max(1, args.keys // 2)
    graph = Graph(persist, log, memo_backend, {"checkpoint_after_entries": trigger})
    times.lap("open store and memo")

    counts = graph.store.reconcile(graph.kc, {k: k for k in range(args.keys)})
    check(counts == (args.keys, 0, 0), f"reconcile returned {counts}")
    times.lap(f"reconcile {args.keys} keys")

    queries = [graph.rt.create_query(graph.reader(k)) for k in range(args.keys)]
    times.lap("create per-key readers")
    for k, q in enumerate(queries):
        check(graph.rt.get(q) == k, f"reader {k} read the wrong row")
    check(graph.reader_runs == args.keys, "a reader ran more or less than once")
    times.lap("first read of every reader")

    for k in range(args.keys):
        check(graph.price(k) == 2 * k, f"memo entry {k} is wrong")
    check(graph.memo_runs == args.keys, "a memo entry did not run exactly once")
    times.lap("first read of every memo entry")

    changed = set()
    stop = threading.Event()
    failures, reads, transient = [], [], [0]
    threads = []
    if shared:
        threads = reader_threads(graph, args, queries, stop, failures, reads, transient)
        for thread in threads:
            thread.start()
    runs_before = graph.reader_runs
    appended = run_units(graph, args, queries, args.units, changed)
    stop.set()
    for thread in threads:
        thread.join()
    check(not failures, f"reader threads failed: {failures[:3]}")
    check(
        args.units * args.per_unit <= appended <= 2 * args.units * args.per_unit,
        f"the log advanced by {appended} records for {args.units * args.per_unit} upserts",
    )
    for k, q in enumerate(queries):
        check(
            graph.rt.get(q) == (k + BUMP if k in changed else k),
            f"reader {k} after the units",
        )
    reruns = graph.reader_runs - runs_before
    check(
        reruns >= len(changed) if shared else reruns == len(changed),
        f"{reruns} readers ran for {len(changed)} changed keys",
    )
    for k in range(args.keys):
        check(
            graph.price(k) == 2 * (k + BUMP if k in changed else k),
            f"memo entry {k} after the units",
        )
    suffix = (
        f", {sum(reads)} reads on {len(threads)} threads, {transient[0]} saw a key mid-replace"
        if shared
        else ""
    )
    times.lap(f"{args.units} on_write units of {args.per_unit}{suffix}")

    stats = graph.memo.stats()
    check(
        stats["entries"] > 0 or stats["unsaved_entries"] > 0, "the memo holds nothing"
    )
    graph.store.checkpoint()
    times.lap("log checkpoint")
    graph.memo.checkpoint()
    times.lap("memo checkpoint")
    check(
        graph.memo.stats()["unsaved_entries"] == 0,
        "unsaved entries after a memo checkpoint",
    )
    position = graph.store.stats()["position"]
    graph.store.close()
    times.lap("close")

    again = Graph(persist, log, memo_backend, {"checkpoint_after_entries": trigger})
    times.lap("reopen store and memo")
    check(len(again.kc.keys()) == args.keys, "keys() after reopen is not every key")
    times.lap("keys() after reopen")
    for k in range(args.keys):
        expected = 2 * (k + BUMP if k in changed else k)
        check(again.price(k) == expected, f"memo entry {k} after reopen is wrong")
    check(again.memo_runs == 0, f"{again.memo_runs} memo entries ran after reopen")
    times.lap("first read of every memo entry after reopen")

    stats = again.store.stats()
    check(stats["position"] == position, "the reopened store is at another position")
    check(
        stats["records_since_checkpoint"] == 0,
        "records since checkpoint after a checkpoint",
    )
    check(again.memo.stats()["entries"] == args.keys, "memo entries after reopen")
    times.lap("stats")

    rt = again.rt
    live = [rt.create_query(again.reader(k)) for k in range(min(args.keys, 1000))]
    for k, q in enumerate(live):
        check(
            rt.get(q) == (k + BUMP if k in changed else k), f"reader {k} after reopen"
        )
    summary = rt.graph_summary()
    check(summary["groups"], "graph_summary has no groups")
    times.lap("graph_summary")
    snapshot = rt.graph_snapshot(
        roots=[live[0]], depth=1, label_prefix=f"{KEYED_LABEL}/", limit=50
    )
    check(len(snapshot["nodes"]) <= 50, "graph_snapshot ignored its limit")
    times.lap("bounded graph_snapshot")

    print(f"{args.package}: {args.keys} keys")
    times.report()
    print("OK")


if __name__ == "__main__":
    main()
