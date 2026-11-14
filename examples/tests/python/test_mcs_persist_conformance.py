import json
import os
import pickle
import sqlite3
import subprocess
import sys
import tempfile
import threading

import incr
import incr.persist as persist
import pytest
from incr.persist import Busy, InvalidCompaction, ReadOnly, Store

HERE = os.path.dirname(os.path.abspath(__file__))

HAS_CONFORMANCE = hasattr(persist, "_conformance")
SKIP_REASON = (
    "incr.persist._conformance is missing: the wheel was built without "
    "--features conformance"
)


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class EpochMismatch(Exception):
    pass


class DictLogState:
    def __init__(self):
        self.lock = threading.Lock()
        self.records = []
        self.tail = 0
        self.snapshot = None
        self.generation = 0


class DictBackend:
    def __init__(self, state, writable=True):
        self.state = state
        self._writable = writable

    def append(self, records):
        if not self._writable:
            raise ReadOnly("handle is read-only")
        with self.state.lock:
            for record in records:
                self.state.tail += 1
                self.state.records.append((self.state.tail, record))
            return self.state.tail

    def append_at(self, after, records):
        if not self._writable:
            raise ReadOnly("handle is read-only")
        with self.state.lock:
            if self.state.tail != after:
                raise persist.TailMoved(f"tail is {self.state.tail}, not {after}")
            for record in records:
                self.state.tail += 1
                self.state.records.append((self.state.tail, record))
            return self.state.tail

    def sync(self, upto):
        pass

    def read_from(self, after):
        with self.state.lock:
            if self.state.snapshot is not None and after < self.state.snapshot[0]:
                return ("compacted", self.state.snapshot[0])
            return [(at, rec) for at, rec in self.state.records if at > after]

    def load_snapshot(self):
        with self.state.lock:
            if self.state.snapshot is None:
                return None
            at, records = self.state.snapshot
            return at, list(records)

    def snapshot_id(self):
        with self.state.lock:
            if self.state.snapshot is None:
                return None
            return self.state.snapshot[0], self.state.generation

    def compact(self, upto, records):
        if not self._writable:
            raise ReadOnly("handle is read-only")
        with self.state.lock:
            floor = self.state.snapshot[0] if self.state.snapshot else 0
            if upto < floor or upto > self.state.tail:
                raise InvalidCompaction(upto, floor, self.state.tail)
            self.state.records = [(at, r) for at, r in self.state.records if at > upto]
            self.state.snapshot = (upto, list(records))
            self.state.generation += 1

    def writable(self):
        return self._writable


class DictLogFactory:
    def new_log(self):
        return DictLogState()

    def open(self, log):
        return DictBackend(log)

    def open_read_only(self, log):
        return DictBackend(log, writable=False)

    def multi_writer(self):
        return True


class SqliteLog:
    """A log in a sqlite3 file standing in for Postgres: every write is one
    transaction, `epoch` (when given) is checked inside `append`'s, and each
    thread uses its own connection."""

    SCHEMA = """
        create table if not exists meta(k text primary key, v integer not null);
        create table if not exists log(pos integer primary key, rec blob not null);
        create table if not exists snap(idx integer primary key, rec blob not null);
        insert or ignore into meta values ('tail', 0);
        insert or ignore into meta values ('snap_pos', 0);
        insert or ignore into meta values ('generation', 0);
        insert or ignore into meta values ('epoch', 0);
    """

    def __init__(self, path, writable=True, epoch=None):
        self.path = path
        self._writable = writable
        self.epoch = epoch
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
        if not self._writable:
            raise ReadOnly("handle is read-only")

        def body(db):
            if self.epoch is not None and self._meta(db, "epoch") != self.epoch:
                raise EpochMismatch(f"epoch {self.epoch} is no longer current")
            tail = self._meta(db, "tail")
            for record in records:
                tail += 1
                db.execute("insert into log values (?, ?)", (tail, record))
            self._set_meta(db, "tail", tail)
            return tail

        return self._write(body)

    def append_at(self, after, records):
        if not self._writable:
            raise ReadOnly("handle is read-only")

        def body(db):
            if self.epoch is not None and self._meta(db, "epoch") != self.epoch:
                raise EpochMismatch(f"epoch {self.epoch} is no longer current")
            tail = self._meta(db, "tail")
            if tail != after:
                raise persist.TailMoved(f"tail is {tail}, not {after}")
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
        if not self._writable:
            raise ReadOnly("handle is read-only")

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
        return self._writable

    def fence(self, epoch):
        self._write(lambda db: self._set_meta(db, "epoch", epoch))


class SqliteLogFactory:
    def __init__(self):
        self.dir = tempfile.TemporaryDirectory()
        self.count = 0

    def new_log(self):
        self.count += 1
        return os.path.join(self.dir.name, f"log{self.count}.db")

    def open(self, log):
        return SqliteLog(log)

    def open_read_only(self, log):
        return SqliteLog(log, writable=False)

    def multi_writer(self):
        return True


class DictMemoState:
    def __init__(self):
        self.lock = threading.Lock()
        self.locked = False
        self.current = None
        self.generations = {}


class DictMemoGuard:
    def __init__(self, state):
        self.state = state

    def release(self):
        with self.state.lock:
            self.state.locked = False


class DictMemo:
    def __init__(self, state):
        self.state = state

    def lock_checkpoint(self):
        with self.state.lock:
            if self.state.locked:
                raise Busy("checkpoint lock held")
            self.state.locked = True
            for gen in [g for g in self.state.generations if g != self.state.current]:
                del self.state.generations[gen]
        return DictMemoGuard(self.state)

    def load(self):
        with self.state.lock:
            current = self.state.current
            if current is None:
                return None
            header, items, _ = self.state.generations[current]
            return current, header, list(items)

    def read_entry(self, generation, item):
        with self.state.lock:
            found = self.state.generations.get(generation)
            return found[2].get(item[2]) if found else None

    def write_generation(self, base, generation, header, entries):
        with self.state.lock:
            if self.state.current != base:
                raise Busy("another handle wrote a generation")
            items = [
                (table, key_hash, loc, len(frame))
                for loc, (table, key_hash, frame) in enumerate(entries)
            ]
            frames = {loc: entry[2] for loc, entry in enumerate(entries)}
            self.state.generations[generation] = (header, items, frames)
            self.state.current = generation
            return list(items)

    def remove_generation(self, generation):
        with self.state.lock:
            self.state.generations.pop(generation, None)


class DictMemoFactory:
    def new_store(self):
        return DictMemoState()

    def open(self, store):
        return DictMemo(store)

    def exclusive_lock(self):
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


class SqliteMemoFactory:
    def __init__(self):
        self.dir = tempfile.TemporaryDirectory()
        self.count = 0

    def new_store(self):
        self.count += 1
        return os.path.join(self.dir.name, f"memo{self.count}.db")

    def open(self, store):
        return SqliteMemo(store)

    def exclusive_lock(self):
        return True


STORE_FLAVORS = {"dict": DictLogFactory, "sqlite3": SqliteLogFactory}
MEMO_FLAVORS = {"dict": DictMemoFactory, "sqlite3": SqliteMemoFactory}


def case_params(kind):
    if not HAS_CONFORMANCE:
        return [pytest.param("-", "-", id="without-the-conformance-feature")]
    flavors = STORE_FLAVORS if kind == "store" else MEMO_FLAVORS
    return [
        pytest.param(flavor, case, id=f"{flavor}-{case}")
        for flavor in flavors
        for case in persist._conformance_cases(kind)
    ]


def run_case(kind, flavors, flavor, case):
    if not HAS_CONFORMANCE:
        pytest.skip(SKIP_REASON)
    outcome = persist._conformance(kind, flavors[flavor](), case)
    assert outcome == {case: None}


@pytest.mark.parametrize(("flavor", "case"), case_params("store"))
def test_store_backend_conformance(flavor, case):
    run_case("store", STORE_FLAVORS, flavor, case)


@pytest.mark.parametrize(("flavor", "case"), case_params("memo"))
def test_memo_backend_conformance(flavor, case):
    run_case("memo", MEMO_FLAVORS, flavor, case)


@pytest.mark.skipif(not HAS_CONFORMANCE, reason=SKIP_REASON)
def test_conformance_reports_a_failing_case_by_name():
    class Broken(DictLogFactory):
        def open(self, log):
            backend = DictBackend(log)
            backend.append = lambda records: 1
            return backend

    outcome = persist._conformance("store", Broken(), "empty_append_returns_the_tail")
    assert list(outcome) == ["empty_append_returns_the_tail"]
    assert outcome["empty_append_returns_the_tail"]
    with pytest.raises(ValueError):
        persist._conformance("store", Broken(), "no_such_case")
    with pytest.raises(ValueError):
        persist._conformance("elsewhere", Broken())


class Graph:
    """A store with a persisted input `base`, a persisted collection `rows`,
    a progress input and a memo table `scaled` that multiplies `base` by its
    key and records each call."""

    def __init__(self, log, memo_backend, build_id="b1"):
        self.calls = []
        self.store = Store.open(log, build_id=build_id)
        self.rt = self.store.runtime
        codec = PickleCodec()
        self.base = self.store.input("base", 1, codec)
        self.progress = self.store.input("progress", 0, codec)
        self.rows = self.store.collection("rows", codec)
        self.table = self.store.memo_table("scaled", 1, codec, codec, self.compute)
        self.memo = self.store.open_memo(memo_backend)

    def compute(self, ctx, key):
        self.calls.append(key)
        return ctx.get(self.base) * key

    def value(self, key):
        return self.rt.get(self.memo.get_or_insert(self.table, key))

    def snapshot(self):
        return {
            "base": self.rt.get(self.base),
            "progress": self.rt.get(self.progress),
            "rows": self.rt.get(self.rows.reduce(sorted)),
        }


def child_main(db_path, memo_path, keys):
    graph = Graph(SqliteLog(db_path), SqliteMemo(memo_path))
    values = [graph.value(key) for key in keys]
    return {**graph.snapshot(), "values": values, "calls": graph.calls}


def run_child(db_path, memo_path, keys):
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]);"
        "import test_mcs_persist_conformance as t;"
        "print(json.dumps(t.child_main(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, HERE, db_path, memo_path, json.dumps(keys)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.splitlines()[-1])


@pytest.fixture
def paths(tmp_path):
    return str(tmp_path / "log.db"), str(tmp_path / "memo.db")


def test_a_fresh_process_restores_inputs_collections_and_memo_hits(paths):
    db_path, memo_path = paths
    graph = Graph(SqliteLog(db_path), SqliteMemo(memo_path))
    graph.store.set(graph.base, 10)
    graph.store.insert(graph.rows, 7)
    graph.store.insert(graph.rows, 3)
    assert [graph.value(3), graph.value(4)] == [30, 40]
    graph.memo.checkpoint()
    graph.store.checkpoint()
    graph.store.set(graph.progress, 5)
    graph.store.close()

    child = run_child(db_path, memo_path, [3, 4])
    assert child["base"] == 10
    assert child["rows"] == [3, 7]
    assert child["progress"] == 5
    assert child["values"] == [30, 40]
    assert child["calls"] == []

    child = run_child(db_path, memo_path, [3, 5])
    assert child["values"] == [30, 50]
    assert child["calls"] == [5]


def test_a_batch_with_a_progress_input_resumes_exactly(paths):
    db_path, memo_path = paths
    log = SqliteLog(db_path)
    graph = Graph(log, SqliteMemo(memo_path))
    with graph.store.batch() as b:
        b.set(graph.base, 2)
        b.insert(graph.rows, 1)
        b.set(graph.progress, 1)
    log.fence(1)
    other = Graph(SqliteLog(db_path, epoch=0), SqliteMemo(memo_path))
    with pytest.raises(EpochMismatch):
        with other.store.batch() as b:
            b.set(other.base, 99)
            b.insert(other.rows, 2)
            b.set(other.progress, 2)
    graph.store.close()

    again = Graph(SqliteLog(db_path), SqliteMemo(memo_path))
    assert again.snapshot() == {"base": 2, "progress": 1, "rows": [1]}


def test_a_fenced_append_leaves_the_store_usable(paths):
    db_path, _ = paths
    log = SqliteLog(db_path, epoch=0)
    store = Store.open(log, build_id=1)
    node = store.input("count", 0, PickleCodec())
    store.set(node, 1)
    log.fence(1)
    with pytest.raises(EpochMismatch):
        store.set(node, 2)
    assert store.runtime.get(node) == 1
    log.epoch = 1
    store.set(node, 3)
    assert store.runtime.get(node) == 3
    store.close()

    again = Store.open(SqliteLog(db_path), build_id=1)
    assert again.runtime.get(again.input("count", 0, PickleCodec())) == 3


def test_a_standby_refresh_picks_up_the_leaders_batch_on_sqlite3(paths):
    db_path, _ = paths
    codec = PickleCodec()
    leader = Store.open(SqliteLog(db_path), build_id=1)
    standby = Store.open(SqliteLog(db_path), build_id=1)
    rows_leader = leader.keyed_collection("rows", codec, codec)
    rows_standby = standby.keyed_collection("rows", codec, codec)
    progress_leader = leader.input("progress", 0, codec)
    progress_standby = standby.input("progress", 0, codec)

    with leader.batch() as b:
        b.upsert(rows_leader, "a", 1)
        b.upsert(rows_leader, "b", 2)
        b.set(progress_leader, 9)
    assert rows_standby.get("a") is None
    standby.refresh()
    assert rows_standby.get("a") == 1
    assert rows_standby.get("b") == 2
    assert standby.runtime.get(progress_standby) == 9

    standby.upsert(rows_standby, "a", 5)
    assert len(rows_standby) == 2
    leader.refresh()
    assert rows_leader.get("a") == 5


def test_quarantined_inputs_and_incomplete_memo_values_restore_on_sqlite3(paths):
    db_path, memo_path = paths
    codec = PickleCodec()

    def build():
        store = Store.open(SqliteLog(db_path), build_id="b1")
        shift = store.input("shift", None, codec)
        partial = store.input("partial", None, codec)
        calls = []

        def compute(ctx, key):
            calls.append(key)
            return ctx.get(partial) + 1

        table = store.memo_table("t", 1, codec, codec, compute)
        memo = store.open_memo(SqliteMemo(memo_path))
        return store, shift, partial, table, memo, calls

    store, shift, partial, table, memo, _ = build()
    quarantined = incr.Unavailable("withheld", row=("emp", 7))
    store.set(shift, quarantined)
    store.set(partial, incr.Incomplete(5, 2))
    expected = store.runtime.get_with_status(memo.get_or_insert(table, "k"))
    assert expected[1] == incr.Incomplete(6, 2)
    memo.checkpoint()
    store.close()

    store, shift, partial, table, memo, calls = build()
    assert store.runtime.status(shift) == quarantined
    assert type(store.runtime.status(shift)) is incr.Unavailable
    assert store.runtime.get_with_status(memo.get_or_insert(table, "k")) == expected
    assert calls == []
