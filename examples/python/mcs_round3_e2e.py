"""End to end run of the mycareshyft round 3 features on one package.

On a persisted store (MemoryBackend log, delta-capable MemoryMemo) it checks
together: default-valued sets surviving a checkpoint and a reopen with
another default (R19), value code that calls the store raising RuntimeError
with nothing on stderr (R18), forget_key freeing an orphan version input
(R14), ctx.keys over the whole key set and over partitions rerunning only on
inserts and removals of their keys (R15, R15b), delta memo checkpoints that
write only the dirty entries (R17), a close and reopen that restores equal
values without running any memo entry, and get_by_id with a generation,
StaleNodeError and graph_snapshot(values=True) on the restored graph (R16).
It exits non-zero on the first mismatch and prints each step's time.

Usage: python mcs_round3_e2e.py --package incr|incr_concurrent [--keys 2000]
"""

import argparse
import importlib
import os
import pickle
import tempfile
import time

DEFAULT = 3
REOPEN_DEFAULT = 9
DIRTY = 7
BUMP = 1_000_000


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class Probe:
    hook = None

    def __init__(self, n):
        self.n = n

    def __hash__(self):
        return hash(self.n)

    def __eq__(self, other):
        if Probe.hook is not None:
            hook, Probe.hook = Probe.hook, None
            hook()
        return isinstance(other, Probe) and other.n == self.n

    def __reduce__(self):
        return (Probe, (self.n,))


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


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAILED: {message}")


def first_letter(key):
    return key[0]


def stderr_of(call):
    with tempfile.TemporaryFile() as sink:
        saved = os.dup(2)
        os.dup2(sink.fileno(), 2)
        error = None
        try:
            call()
        except Exception as raised:
            error = raised
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        sink.seek(0)
        return error, sink.read().decode()


class Counted:
    def __init__(self, rt, body):
        self.rt = rt
        self.runs = 0

        def run(ctx):
            self.runs += 1
            return body(ctx)

        self.node = rt.create_query(run)

    def value(self):
        return self.rt.get(self.node)


class Graph:
    def __init__(self, persist, log, memo_backend, default):
        self.store = persist.Store.open(log, build_id="r3-e2e")
        self.rt = self.store.runtime
        self.limit = self.store.input("limit", default, PickleCodec())
        self.kc = self.store.keyed_collection(
            "shifts", PickleCodec(), PickleCodec(), label="shifts"
        )
        self.probes = self.store.keyed_collection(
            "probes", PickleCodec(), PickleCodec(), label="probes"
        )
        self.parts = self.kc.key_partitions(first_letter)
        self.memo_runs = 0

        def price(ctx, key):
            self.memo_runs += 1
            return ctx.get_key(self.kc, key) * 2

        self.table = self.store.memo_table(
            "price", 1, PickleCodec(), PickleCodec(), price
        )
        self.memo = self.store.open_memo(memo_backend)

    def price(self, key):
        return self.rt.get(self.memo.get_or_insert(self.table, key))


def read_all(*readers):
    return [reader.value() for reader in readers]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, choices=["incr", "incr_concurrent"])
    parser.add_argument("--keys", type=int, default=2000)
    args = parser.parse_args()
    args.keys = max(args.keys, 4 * DIRTY)

    pkg = importlib.import_module(args.package)
    persist = importlib.import_module(f"{args.package}.persist")
    local = args.package == "incr"
    times = Timings()
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    graph = Graph(persist, log, memo_backend, DEFAULT)
    store, rt, kc = graph.store, graph.rt, graph.kc
    rows = {f"{'ab'[i % 2]}{i}": i for i in range(args.keys)}
    check(
        store.reconcile(kc, dict(rows)) == (len(rows), 0, 0),
        "reconcile did not insert every key",
    )
    times.lap(f"open and reconcile {len(rows)} keys")

    store.set(graph.limit, DEFAULT)
    store.checkpoint()
    times.lap("R19 set equal to the default, log checkpoint")

    everything = Counted(rt, lambda ctx: sorted(ctx.keys(kc)))
    in_a = Counted(rt, lambda ctx: sorted(ctx.keys(graph.parts, "a")))
    in_b = Counted(rt, lambda ctx: sorted(ctx.keys(graph.parts, "b")))
    check(everything.value() == sorted(rows), "ctx.keys(kc) is not every key")
    check(
        in_a.value() == sorted(k for k in rows if k[0] == "a"),
        "ctx.keys(parts, 'a') is wrong",
    )
    check(
        in_b.value() == sorted(k for k in rows if k[0] == "b"),
        "ctx.keys(parts, 'b') is wrong",
    )
    times.lap("R15 first reads of whole-set and partition readers")

    store.upsert(kc, "a0", rows["a0"] + BUMP)
    rows["a0"] += BUMP
    read_all(everything, in_a, in_b)
    check(
        (everything.runs, in_a.runs, in_b.runs) == (1, 1, 1),
        "a row replacement reran a key-set reader",
    )
    store.upsert(kc, "a_new", 1)
    rows["a_new"] = 1
    read_all(everything, in_a, in_b)
    check(
        (everything.runs, in_a.runs, in_b.runs) == (2, 2, 1),
        "an insert into partition a reran the wrong readers",
    )
    store.remove(kc, "b1")
    del rows["b1"]
    read_all(everything, in_a, in_b)
    check(
        (everything.runs, in_a.runs, in_b.runs) == (3, 2, 2),
        "a remove from partition b reran the wrong readers",
    )
    check(everything.value() == sorted(rows), "ctx.keys(kc) after writes")
    times.lap("R15 inserts, removes and row replacements")

    ghost = rt.create_query(lambda ctx: ctx.get_key(kc, "ghost"))
    check(rt.get(ghost) is None, "a never written key is not None")
    before = kc.version_input_count()
    check(kc.forget_key("ghost") is False, "forget_key freed an input a reader holds")
    rt.delete_node(ghost)
    freed = kc.forget_key("ghost")
    check(freed is local, f"forget_key returned {freed} on {args.package}")
    check(
        kc.version_input_count() == before - (1 if local else 0),
        "version_input_count did not drop by the freed input",
    )
    times.lap("R14 forget_key frees an orphan version input")

    store.upsert(graph.probes, "p", Probe(1))
    Probe.hook = store.stats
    error, noise = stderr_of(lambda: store.upsert(graph.probes, "p", Probe(1)))
    check(
        isinstance(error, RuntimeError), f"row __eq__ calling the store gave {error!r}"
    )
    check(noise == "", f"row __eq__ calling the store wrote to stderr: {noise!r}")
    store.upsert(graph.probes, "q", Probe(2))
    check(graph.probes.get("q").n == 2, "the store is unusable after the refusal")
    times.lap("R18 value code calling the store raises quietly")

    for key in rows:
        check(graph.price(key) == 2 * rows[key], f"price {key}")
    check(graph.memo_runs == len(rows), "a memo entry did not run exactly once")
    first = graph.memo.checkpoint()
    check(first["delta"] is False, "the first memo checkpoint was a delta")
    check(first["fresh_written"] == len(rows), f"first checkpoint {first}")
    times.lap("R17 memo entries and full checkpoint")

    candidates = sorted(key for key in rows if key != "a_new")
    for round_number in range(3):
        dirty = candidates[round_number * DIRTY : (round_number + 1) * DIRTY]
        for key in dirty:
            store.upsert(kc, key, rows[key] + BUMP)
            rows[key] += BUMP
        for key in dirty:
            check(graph.price(key) == 2 * rows[key], f"dirty price {key}")
        unsaved = graph.memo.stats()["unsaved_entries"]
        counts = graph.memo.checkpoint()
        check(counts["delta"] is True, f"round {round_number} not a delta: {counts}")
        check(
            counts["fresh_written"] == len(dirty) == unsaved,
            f"fresh_written {counts['fresh_written']} for {len(dirty)} dirty, "
            f"{unsaved} unsaved",
        )
        check(counts["tombstones"] == 0, f"tombstones in {counts}")
        check(counts["chain_depth"] == round_number + 1, f"chain depth in {counts}")
    stats = graph.memo.stats()
    check(stats["delta_checkpoints"] == 3, f"memo stats {stats}")
    times.lap("R17 three delta checkpoints writing only the dirty entries")

    store.checkpoint()
    position = store.stats()["position"]
    store.close()
    times.lap("log checkpoint and close")

    again = Graph(persist, log, memo_backend, REOPEN_DEFAULT)
    check(
        again.rt.get(again.limit) == DEFAULT, "R19: a set equal to the default was lost"
    )
    check(
        sorted(again.kc.keys()) == sorted(rows),
        "keys() after reopen is not the written key set",
    )
    reopened = again.store.stats()["position"]
    check(
        position <= reopened <= position + 1,
        f"reopened at {reopened}, not at {position} or its redeclared input",
    )
    for key in rows:
        check(again.price(key) == 2 * rows[key], f"price {key} after reopen")
    check(again.memo_runs == 0, f"{again.memo_runs} memo entries ran after reopen")
    check(
        again.memo.stats()["entries"] == len(rows),
        "memo entries after reopen differ from the rows",
    )
    times.lap("reopen with another default, restore equal, no memo entry ran")

    rt = again.rt
    in_a = Counted(rt, lambda ctx: sorted(ctx.keys(again.parts, "a")))
    check(
        in_a.value() == sorted(k for k in rows if k[0] == "a"),
        "partition a after reopen",
    )
    again.store.upsert(again.kc, "a_new2", 1)
    rows["a_new2"] = 1
    check("a_new2" in in_a.value() and in_a.runs == 2, "partition a missed an insert")
    again.store.upsert(again.kc, "b_new2", 1)
    rows["b_new2"] = 1
    in_a.value()
    check(in_a.runs == 2, "an insert into partition b reran the a reader")
    times.lap("R15b partitions after reopen")

    again.price("a_new2")
    delta = again.memo.checkpoint()
    check(delta["delta"] is True, f"no delta after a restore: {delta}")
    check(delta["fresh_written"] == 1, f"fresh_written after a restore: {delta}")
    times.lap("R17 delta checkpoint after a restore")

    reader = Counted(rt, lambda ctx: ctx.get_key(again.kc, "a2") + 1)
    reader.value()
    snapshot = {n["id"]: n for n in rt.graph_snapshot(values=True)}
    entry = snapshot[reader.node.id]
    check(entry["status"] == "clean", f"snapshot status {entry['status']}")
    check(entry["value"] == rows["a2"] + 1, "snapshot value is not the node's value")
    runs = reader.runs
    check(
        rt.get_by_id(entry["id"], entry["generation"]) == entry["value"],
        "get_by_id differs from the snapshot value",
    )
    check(reader.runs == runs, "get_by_id recomputed a clean node")
    rt.delete_node(reader.node)
    stale_error = None
    try:
        rt.get_by_id(entry["id"], entry["generation"])
    except pkg.StaleNodeError as raised:
        stale_error = raised
    check(stale_error is not None, "get_by_id of a deleted node did not raise")
    check(isinstance(stale_error, LookupError), "StaleNodeError is not a LookupError")
    times.lap("R16 get_by_id, StaleNodeError and graph_snapshot(values=True)")

    print(f"{args.package}: {len(rows)} keys")
    times.report()
    print("OK")


if __name__ == "__main__":
    main()
