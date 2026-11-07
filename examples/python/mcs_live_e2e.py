"""End to end run of one scheduling tenant on a persisted store over the
stdlib sqlite3 backend: batches with a progress input, keyed reconcile,
lookups, a quarantined shift, 50 overlays checked against from-scratch graphs,
memory stats, checkpoints and a restart in a subprocess.

Usage: python mcs_live_e2e.py
"""

import json
import os
import pickle
import random
import sqlite3
import subprocess
import sys
import tempfile
import threading

import incr
import incr.persist as persist
from incr.persist import Busy, InvalidCompaction, Store

WORKERS = 6
SHIFTS = 8
QUARANTINED_WORKER = 2
FAILED_LEG_WORKER = 1
BUILD_ID = "e2e-1"
CALLS = {"scaled": [], "pair": []}


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


CODEC = PickleCodec()


class SqliteLog:
    SCHEMA = """
        create table if not exists meta(k text primary key, v integer not null);
        create table if not exists log(pos integer primary key, rec blob not null);
        create table if not exists snap(idx integer primary key, rec blob not null);
        insert or ignore into meta values ('tail', 0);
        insert or ignore into meta values ('snap_pos', 0);
        insert or ignore into meta values ('generation', 0);
    """

    def __init__(self, path):
        self.path = path
        self.local = threading.local()
        self.db().executescript(self.SCHEMA)

    def db(self):
        conn = getattr(self.local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=60)
            conn.execute("pragma journal_mode = wal")
            conn.execute("pragma synchronous = full")
            self.local.conn = conn
        return conn

    def _meta(self, db, key):
        return db.execute("select v from meta where k = ?", (key,)).fetchone()[0]

    def _set_meta(self, db, key, value):
        db.execute("update meta set v = ? where k = ?", (value, key))

    def _write(self, body):
        db = self.db()
        db.execute("begin immediate")
        try:
            result = body(db)
        except BaseException:
            db.execute("rollback")
            raise
        db.execute("commit")
        return result

    def append(self, records):
        def body(db):
            tail = self._meta(db, "tail")
            for record in records:
                tail += 1
                db.execute("insert into log values (?, ?)", (tail, record))
            self._set_meta(db, "tail", tail)
            return tail

        return self._write(body)

    def sync(self, upto):
        pass

    def read_from(self, after):
        db = self.db()
        db.execute("begin")
        try:
            snap_pos = self._meta(db, "snap_pos")
            if self._meta(db, "generation") and after < snap_pos:
                return ("compacted", snap_pos)
            return db.execute(
                "select pos, rec from log where pos > ? order by pos", (after,)
            ).fetchall()
        finally:
            db.execute("commit")

    def load_snapshot(self):
        db = self.db()
        db.execute("begin")
        try:
            if not self._meta(db, "generation"):
                return None
            rows = db.execute("select rec from snap order by idx").fetchall()
            return self._meta(db, "snap_pos"), [row[0] for row in rows]
        finally:
            db.execute("commit")

    def snapshot_id(self):
        db = self.db()
        db.execute("begin")
        try:
            generation = self._meta(db, "generation")
            if not generation:
                return None
            return self._meta(db, "snap_pos"), generation
        finally:
            db.execute("commit")

    def compact(self, upto, records):
        def body(db):
            floor = self._meta(db, "snap_pos")
            tail = self._meta(db, "tail")
            if upto < floor or upto > tail:
                raise InvalidCompaction(upto, floor, tail)
            db.execute("delete from log where pos <= ?", (upto,))
            db.execute("delete from snap")
            for idx, record in enumerate(records):
                db.execute("insert into snap values (?, ?)", (idx, record))
            self._set_meta(db, "snap_pos", upto)
            self._set_meta(db, "generation", self._meta(db, "generation") + 1)

        self._write(body)

    def writable(self):
        return True


class SqliteMemoGuard:
    def __init__(self, memo):
        self.memo = memo

    def release(self):
        self.memo.db().execute("delete from meta where k = 'lock'")


class SqliteMemo:
    def __init__(self, path):
        self.path = path
        self.local = threading.local()
        self.db().executescript(
            """
            create table if not exists meta(k text primary key, v integer);
            create table if not exists gens(id integer primary key, header blob not null);
            create table if not exists items(
                gen integer, loc integer, tbl integer, key_hash text, len integer,
                frame blob not null, primary key (gen, loc));
            """
        )

    def db(self):
        conn = getattr(self.local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=60)
            conn.execute("pragma journal_mode = wal")
            conn.execute("pragma synchronous = full")
            self.local.conn = conn
        return conn

    def _current(self, db):
        row = db.execute("select v from meta where k = 'current'").fetchone()
        return row[0] if row else None

    def lock_checkpoint(self):
        db = self.db()
        db.execute("begin immediate")
        try:
            if db.execute("select 1 from meta where k = 'lock'").fetchone():
                raise Busy("checkpoint lock held")
            db.execute("insert into meta values ('lock', 1)")
            current = self._current(db)
            db.execute("delete from items where gen is not ?", (current,))
            db.execute("delete from gens where id is not ?", (current,))
        except BaseException:
            db.execute("rollback")
            raise
        db.execute("commit")
        return SqliteMemoGuard(self)

    def load(self):
        db = self.db()
        db.execute("begin")
        try:
            current = self._current(db)
            if current is None:
                return None
            (header,) = db.execute(
                "select header from gens where id = ?", (current,)
            ).fetchone()
            rows = db.execute(
                "select tbl, key_hash, loc, len from items where gen = ? order by loc",
                (current,),
            ).fetchall()
        finally:
            db.execute("commit")
        return current, header, [(t, int(h), loc, n) for t, h, loc, n in rows]

    def read_entry(self, generation, item):
        row = (
            self.db()
            .execute(
                "select frame from items where gen = ? and loc = ?",
                (generation, item[2]),
            )
            .fetchone()
        )
        return row[0] if row else None

    def write_generation(self, base, generation, header, entries):
        db = self.db()
        db.execute("begin immediate")
        try:
            if self._current(db) != base:
                raise Busy("another handle wrote a generation")
            db.execute("insert into gens values (?, ?)", (generation, header))
            items = []
            for loc, (table, key_hash, frame) in enumerate(entries):
                db.execute(
                    "insert into items values (?, ?, ?, ?, ?, ?)",
                    (generation, loc, table, str(key_hash), len(frame), frame),
                )
                items.append((table, key_hash, loc, len(frame)))
            db.execute(
                "insert or replace into meta values ('current', ?)", (generation,)
            )
        except BaseException:
            db.execute("rollback")
            raise
        db.execute("commit")
        return items

    def remove_generation(self, generation):
        db = self.db()
        db.execute("delete from items where gen = ?", (generation,))
        db.execute("delete from gens where id = ?", (generation,))


def add(a, b):
    return a + b


def build_totals(rt, collection, rate):
    by_worker = collection.group_by(lambda row: row[0])
    graph = {}
    for w in range(WORKERS):
        group = by_worker.get_group(w)
        hours = group.aggregate(0, lambda row: row[2], add)
        gaps = (
            group.sort_by_key(lambda row: row[1])
            .pairwise()
            .aggregate(0, lambda pair: pair[1][1] - pair[0][1], add)
        )
        pay = rt.create_query(
            lambda ctx, hours=hours, w=w: (
                ctx.get(hours) * ctx.get(rate) + ctx.lookup(("leg", w))
            )
        )
        graph[w] = {"hours": hours, "gaps": gaps, "pay": pay}
    return graph


def read_totals(read, graph, names):
    return {(w, name): read(graph[w][name]) for w in range(WORKERS) for name in names}


def reference_totals(rows, rate_value, legs, names):
    rt = incr.Runtime()
    collection = rt.create_collection()
    collection.extend(list(rows))
    rate = rt.create_input(rate_value)
    graph = build_totals(rt, collection, rate)
    rt.resolve_lookups(legs)
    totals = read_totals(rt.get_with_status, graph, names)
    rt.close()
    return totals


def missing_of(status):
    return None if status is None else status.missing


class Tenant:
    def __init__(self, db_path):
        self.log = SqliteLog(db_path)
        self.store = Store.open(self.log, build_id=BUILD_ID)
        self.rt = self.store.runtime
        self.rate = self.store.input("rate", 1, CODEC)
        self.progress = self.store.input("progress", 0, CODEC)
        self.shifts = self.store.keyed_collection("shifts", CODEC, CODEC)
        self.scaled = self.store.memo_table(
            "scaled", 1, CODEC, CODEC, self.compute_scaled
        )
        self.pair = self.store.memo_table("pair", 1, CODEC, CODEC, self.compute_pair)
        self.memo = self.store.open_memo(SqliteMemo(db_path + ".memo"))
        self.graph = None

    def build_graph(self):
        self.graph = build_totals(self.rt, self.shifts.collection, self.rate)

    def compute_scaled(self, ctx, key):
        CALLS["scaled"].append(key)
        return ctx.get(self.rate) * key

    def compute_pair(self, ctx, key):
        CALLS["pair"].append(key)
        return ctx.memo(self.scaled, key) + ctx.memo(self.scaled, key + 1)

    def memo_value(self, table, key):
        return self.rt.get(self.memo.get_or_insert(table, key))


def initial_rows(rng):
    return {
        (w, s * 10): (w, s * 10, rng.randrange(1, 9))
        for w in range(WORKERS)
        for s in range(SHIFTS)
    }


def candidate_for(rng, model):
    w = rng.randrange(WORKERS)
    if rng.random() < 0.5:
        start = rng.choice([k[1] for k in model if k[0] == w])
    else:
        start = rng.randrange(SHIFTS * 10) * 10 + 5
    return (w, start), (w, start, rng.randrange(1, 9))


def hydrate(tenant, rng):
    model = initial_rows(rng)
    keys = sorted(model)
    half = len(keys) // 2
    with tenant.store.batch() as b:
        b.reconcile(tenant.shifts, [(k, model[k]) for k in keys[:half]])
        b.set(tenant.progress, 1)
    assert tenant.rt.get(tenant.progress) == 1

    changed = {k: (k[0], k[1], model[k][2] + 1) for k in keys[:3]}
    model.update(changed)
    dropped = keys[half - 2 : half]
    for key in dropped:
        del model[key]
    with tenant.store.batch() as b:
        b.reconcile(tenant.shifts, [(k, model[k]) for k in sorted(model)])
        b.set(tenant.progress, 2)
        b.set(tenant.rate, 3)
    assert tenant.store.reconcile(tenant.shifts, sorted(model.items())) == (0, 0, 0)
    assert tenant.rt.get(tenant.rate) == 3
    return model


def resolve_legs(tenant):
    rt = tenant.rt
    pay = {w: tenant.graph[w]["pay"] for w in range(WORKERS)}
    for w in range(WORKERS):
        assert rt.get(pay[w]) == incr.Pending(key=("leg", w))
    pending = rt.drain_pending_lookups()
    assert sorted(pending) == [("leg", w) for w in range(WORKERS)]
    assert rt.drain_pending_lookups() == []

    legs = {("leg", w): 100 + w for w in range(WORKERS)}
    failed_key = ("leg", FAILED_LEG_WORKER)
    rt.resolve_lookups({k: v for k, v in legs.items() if k != failed_key})
    rt.fail_lookup(failed_key, "timeout")
    assert rt.get(pay[FAILED_LEG_WORKER]) == incr.LookupFailed(
        "timeout", key=failed_key
    )
    rt.retry_lookups([failed_key])
    assert rt.get(pay[FAILED_LEG_WORKER]) == incr.Pending(key=failed_key)
    assert rt.drain_pending_lookups() == [failed_key]
    rt.resolve_lookups({failed_key: legs[failed_key]})
    return legs


def quarantine(tenant, model, worker):
    key = sorted(k for k in model if k[0] == worker)[0]
    row = model[key]
    tenant.store.upsert(tenant.shifts, key, incr.Unavailable("quarantined", row=row))
    return key, row


def reference_rows(model, quarantined_key):
    rows = []
    for key, row in model.items():
        if key == quarantined_key:
            rows.append(incr.Unavailable("quarantined", row=row))
        else:
            rows.append(row)
    return rows


def run_overlays(tenant, model, quarantined_key, legs, rng, count):
    rt = tenant.rt
    names = ("hours", "gaps", "pay")
    graph = tenant.graph
    for w in range(WORKERS):
        rt.watch(graph[w]["hours"])
    rt.drain_dirtied_watched()
    rate_value = rt.get(tenant.rate)
    base_expected = reference_totals(
        reference_rows(model, quarantined_key), rate_value, legs, names
    )
    assert read_totals(rt.get_with_status, graph, names) == base_expected

    inserts = upserts = 0
    for _ in range(count):
        key, row = candidate_for(rng, model)
        if key == quarantined_key:
            continue
        upserts += key in model
        inserts += key not in model
        candidate_model = dict(model)
        candidate_model[key] = row
        expected = reference_totals(
            reference_rows(candidate_model, quarantined_key), rate_value, legs, names
        )
        with rt.overlay() as ov:
            tenant.shifts.upsert(key, row)
            assert read_totals(ov.get_with_status, graph, names) == expected
            assert tenant.shifts.get(key) == row
        assert read_totals(rt.get_with_status, graph, names) == base_expected
        assert tenant.shifts.get(key) == model.get(key)
    assert rt.drain_dirtied_watched() == []
    return inserts, upserts


def expectations(tenant, model, quarantined_key):
    totals = read_totals(tenant.rt.get_with_status, tenant.graph, ("hours", "gaps"))
    return {
        "rate": tenant.rt.get(tenant.rate),
        "progress": tenant.rt.get(tenant.progress),
        "rows": sorted(
            list(row) for key, row in model.items() if key != quarantined_key
        ),
        "quarantined": list(quarantined_key),
        "totals": [
            [w, name, value, missing_of(status)]
            for (w, name), (value, status) in sorted(totals.items())
        ],
        "scaled": {str(k): tenant.memo_value(tenant.scaled, k) for k in (3, 4, 5)},
        "pair": {str(k): tenant.memo_value(tenant.pair, k) for k in (3, 4)},
    }


def first_process(db_path):
    rng = random.Random(7)
    tenant = Tenant(db_path)
    rt = tenant.rt
    model = hydrate(tenant, rng)
    tenant.build_graph()
    print(f"hydrated {len(model)} shifts, progress {rt.get(tenant.progress)}")

    legs = resolve_legs(tenant)
    print(f"lookups: {len(legs)} created, drained, resolved; 1 failed and retried")

    quarantined_key, _ = quarantine(tenant, model, QUARANTINED_WORKER)
    value, status = rt.get_with_status(tenant.graph[QUARANTINED_WORKER]["hours"])
    assert status == incr.Incomplete(value, 1), status
    assert rt.get_with_status(tenant.graph[0]["hours"])[1] is None
    print(f"quarantined shift {quarantined_key}: worker total {value} reads {status}")

    inserts, upserts = run_overlays(tenant, model, quarantined_key, legs, rng, 50)
    print(f"overlays: {inserts} inserts, {upserts} upserts matched from-scratch graphs")

    stats = rt.memory_stats()
    print(
        "memory_stats: nodes %d py_refs %d bytes %d kinds %s"
        % (
            stats["nodes"],
            stats["py_refs"],
            stats["bytes"],
            sorted(stats["by_kind"]),
        )
    )

    tenant.memo_value(tenant.pair, 3)
    tenant.memo_value(tenant.pair, 4)
    tenant.memo_value(tenant.scaled, 5)
    tenant.memo.checkpoint()
    tenant.store.checkpoint()
    print("checkpoint and memo checkpoint written")

    tail_key = (0, 1000)
    tail_row = (0, 1000, 5)
    with tenant.store.batch() as b:
        b.upsert(tenant.shifts, tail_key, tail_row)
        b.set(tenant.progress, 3)
    model[tail_key] = tail_row

    saved = expectations(tenant, model, quarantined_key)
    with open(db_path + ".json", "w") as handle:
        json.dump(saved, handle)
    tenant.store.close()
    print(f"saved resume position {saved['progress']} and {len(saved['rows'])} rows")


def restart(db_path):
    with open(db_path + ".json") as handle:
        saved = json.load(handle)
    tenant = Tenant(db_path)
    tenant.build_graph()
    rt = tenant.rt

    assert rt.get(tenant.rate) == saved["rate"]
    assert rt.get(tenant.progress) == saved["progress"], "resume position differs"
    rows = rt.get(tenant.shifts.collection.reduce(sorted))
    assert [list(row) for row in rows] == saved["rows"]
    held = tenant.shifts.get(tuple(saved["quarantined"]))
    assert type(held) is incr.Unavailable and held.reason == "quarantined"

    totals = read_totals(rt.get_with_status, tenant.graph, ("hours", "gaps"))
    got = [
        [w, name, value, missing_of(status)]
        for (w, name), (value, status) in sorted(totals.items())
    ]
    assert got == saved["totals"], "totals differ"

    for table, name in ((tenant.scaled, "scaled"), (tenant.pair, "pair")):
        for key, value in saved[name].items():
            assert tenant.memo_value(table, int(key)) == value
    assert CALLS == {"scaled": [], "pair": []}, "restored keys were recomputed"

    assert tenant.memo_value(tenant.scaled, 9) == saved["rate"] * 9
    assert CALLS == {"scaled": [9], "pair": []}
    tenant.store.close()
    print(
        f"restart: inputs, {len(rows)} rows, totals and resume position "
        f"{saved['progress']} match; memo hits without recompute"
    )


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--restart":
        restart(sys.argv[2])
        return
    assert hasattr(persist, "Store")
    with tempfile.TemporaryDirectory() as directory:
        db_path = os.path.join(directory, "tenant.db")
        first_process(db_path)
        out = subprocess.run(
            [sys.executable, __file__, "--restart", db_path],
            capture_output=True,
            text=True,
        )
        sys.stdout.write(out.stdout)
        sys.stderr.write(out.stderr)
        if out.returncode != 0:
            sys.exit(out.returncode)
    print("e2e ok")


if __name__ == "__main__":
    main()
