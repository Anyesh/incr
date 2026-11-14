import importlib
import json
import os
import pickle
import sqlite3
import subprocess
import sys

import incr.persist
import incr_concurrent.persist
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

PACKAGES = [
    pytest.param(incr.persist, id="local"),
    pytest.param(incr_concurrent.persist, id="shared"),
]


@pytest.fixture(params=PACKAGES)
def persist(request):
    return request.param


@pytest.fixture
def pkg(persist):
    return importlib.import_module(persist.__name__.rpartition(".")[0])


class JsonCodec:
    def encode(self, obj):
        return json.dumps(obj, sort_keys=True).encode()

    def decode(self, data):
        return json.loads(data)


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class SqliteGuard:
    def __init__(self, db):
        self.db = db

    def release(self):
        self.db.execute("delete from meta where k = 'lock'")


class SqliteMemo:
    """A memo backend whose generations live in a sqlite file, so they
    cross processes. `busy` is the `Busy` class of the package whose store
    it serves."""

    busy = incr.persist.Busy

    def __init__(self, path):
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.executescript(
            """
            create table if not exists meta(k text primary key, v integer);
            create table if not exists gens(id integer primary key, header blob not null);
            create table if not exists items(
                gen integer, loc integer, tbl integer, key_hash text, len integer,
                frame blob not null, primary key (gen, loc));
            """
        )

    def _current(self):
        row = self.db.execute("select v from meta where k = 'current'").fetchone()
        return row[0] if row else None

    def lock_checkpoint(self):
        try:
            self.db.execute("insert into meta values ('lock', 1)")
        except sqlite3.IntegrityError:
            raise self.busy("checkpoint lock held") from None
        current = self._current()
        self.db.execute("delete from items where gen is not ?", (current,))
        self.db.execute("delete from gens where id is not ?", (current,))
        return SqliteGuard(self.db)

    def load(self):
        current = self._current()
        if current is None:
            return None
        (header,) = self.db.execute(
            "select header from gens where id = ?", (current,)
        ).fetchone()
        rows = self.db.execute(
            "select tbl, key_hash, loc, len from items where gen = ? order by loc",
            (current,),
        )
        return current, header, [(t, int(h), loc, n) for t, h, loc, n in rows]

    def read_entry(self, generation, item):
        row = self.db.execute(
            "select frame from items where gen = ? and loc = ?", (generation, item[2])
        ).fetchone()
        return row[0] if row else None

    def write_generation(self, base, generation, header, entries):
        self.db.execute("begin immediate")
        try:
            if self._current() != base:
                raise self.busy("another handle wrote a generation")
            self.db.execute("insert into gens values (?, ?)", (generation, header))
            items = []
            for loc, (table, key_hash, frame) in enumerate(entries):
                self.db.execute(
                    "insert into items values (?, ?, ?, ?, ?, ?)",
                    (generation, loc, table, str(key_hash), len(frame), frame),
                )
                items.append((table, key_hash, loc, len(frame)))
            self.db.execute(
                "insert or replace into meta values ('current', ?)", (generation,)
            )
            self.db.execute("commit")
        except BaseException:
            self.db.execute("rollback")
            raise
        return items

    def remove_generation(self, generation):
        self.db.execute("delete from items where gen = ?", (generation,))
        self.db.execute("delete from gens where id = ?", (generation,))


class Wrapped:
    """Delegates to a memo backend, with `hooks` replacing methods and
    `release` replacing the guard's release."""

    def __init__(self, inner, hooks=None, release=None):
        self.inner = inner
        self.hooks = hooks or {}
        self.release = release

    def _call(self, name, *args):
        if name in self.hooks:
            return self.hooks[name](*args)
        return getattr(self.inner, name)(*args)

    def lock_checkpoint(self):
        guard = self._call("lock_checkpoint")
        if self.release is None:
            return guard
        release = self.release

        class Guard:
            def release(self):
                guard.release()
                release()

        return Guard()

    def load(self):
        return self._call("load")

    def read_entry(self, generation, item):
        return self._call("read_entry", generation, item)

    def write_generation(self, base, generation, header, entries):
        return self._call("write_generation", base, generation, header, entries)

    def remove_generation(self, generation):
        return self._call("remove_generation", generation)


class Graph:
    """A store with a persisted input `base` and memo table `scaled`, which
    multiplies `base` by its key and records each call."""

    def __init__(self, persist, log, memo_backend, build_id="b1"):
        if isinstance(memo_backend, SqliteMemo):
            memo_backend.busy = persist.Busy
        self.calls = []
        self.store = persist.Store.open(log, build_id=build_id)
        self.rt = self.store.runtime
        self.base = self.store.input("base", 1, JsonCodec())
        self.table = self.store.memo_table(
            "scaled", 1, JsonCodec(), JsonCodec(), self.compute
        )
        self.memo = self.store.open_memo(memo_backend)

    def compute(self, ctx, key):
        self.calls.append(key)
        return ctx.get(self.base) * key

    def value(self, key):
        return self.rt.get(self.memo.get_or_insert(self.table, key))


def log_records(log):
    return [record for _, record in log.read_from(0)]


def child_main(memo_path, log_path, set_base, keys):
    with open(log_path, "rb") as f:
        records = pickle.load(f)
    log = incr.persist.MemoryBackend()
    log.append(records)
    graph = Graph(incr.persist, log, SqliteMemo(memo_path))
    if set_base is not None:
        graph.store.set(graph.base, set_base)
    values = [graph.value(key) for key in keys]
    return {"base": graph.rt.get(graph.base), "values": values, "calls": graph.calls}


def run_child(memo_path, log_path, set_base, keys):
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]);"
        "import test_mcs_persist_memo as t;"
        "print(json.dumps(t.child_main(sys.argv[2], sys.argv[3],"
        " json.loads(sys.argv[4]), json.loads(sys.argv[5]))))"
    )
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            HERE,
            memo_path,
            log_path,
            json.dumps(set_base),
            json.dumps(keys),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.splitlines()[-1])


@pytest.fixture
def first_run(tmp_path):
    memo_path = str(tmp_path / "memo.db")
    log_path = str(tmp_path / "log.pkl")
    log = incr.persist.MemoryBackend()
    graph = Graph(incr.persist, log, SqliteMemo(memo_path))
    graph.store.set(graph.base, 10)
    assert [graph.value(3), graph.value(4)] == [30, 40]
    assert graph.calls == [3, 4]
    graph.memo.checkpoint()
    with open(log_path, "wb") as f:
        pickle.dump(log_records(log), f)
    graph.store.close()
    return memo_path, log_path


def test_a_fresh_process_restores_inputs_and_memo_hits(first_run):
    memo_path, log_path = first_run
    result = run_child(memo_path, log_path, None, [3, 4])
    assert result == {"base": 10, "values": [30, 40], "calls": []}


def test_a_memo_whose_input_changed_recomputes(first_run):
    memo_path, log_path = first_run
    result = run_child(memo_path, log_path, 11, [3, 4])
    assert result == {"base": 11, "values": [33, 44], "calls": [3, 4]}


def test_a_memo_restores_in_process_and_a_new_key_computes(persist):
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    graph = Graph(persist, log, memo_backend)
    graph.store.set(graph.base, 5)
    assert graph.value(2) == 10
    stats = graph.memo.checkpoint()
    assert stats["written"] == 1
    graph.store.close()

    again = Graph(persist, log, memo_backend)
    assert again.value(2) == 10
    assert again.calls == []
    assert again.value(7) == 35
    assert again.calls == [7]


def test_a_build_id_change_drops_the_entries(persist):
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    graph = Graph(persist, log, memo_backend, build_id="old")
    assert graph.value(2) == 2
    graph.memo.checkpoint()
    graph.store.close()

    again = Graph(persist, log, memo_backend, build_id="new")
    assert again.value(2) == 2
    assert again.calls == [2]


def test_a_codec_that_does_not_round_trip_raises_after_the_checkpoint(persist):
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    calls = []

    def declare(store):
        def pair(ctx, key):
            calls.append(key)
            return (key, key)

        table = store.memo_table("pairs", 1, JsonCodec(), JsonCodec(), pair)
        return table, store.open_memo(memo_backend)

    store = persist.Store.open(log, build_id="b1")
    table, memo = declare(store)
    assert store.runtime.get(memo.get_or_insert(table, 1)) == (1, 1)
    with pytest.raises(ValueError, match="'pairs'"):
        memo.checkpoint()
    store.close()

    again = persist.Store.open(log, build_id="b1")
    table, memo = declare(again)
    assert again.runtime.get(memo.get_or_insert(table, 1)) == (1, 1)
    assert calls == [1, 1]


def test_every_checkpoint_raises_while_the_unencodable_entry_is_live(persist):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    table = store.memo_table(
        "pairs", 1, JsonCodec(), JsonCodec(), lambda ctx, key: (key, key)
    )
    memo = store.open_memo(persist.MemoryMemo())
    store.runtime.get(memo.get_or_insert(table, 1))
    with pytest.raises(ValueError):
        memo.checkpoint()
    with pytest.raises(ValueError, match="'pairs'"):
        memo.checkpoint()


@pytest.mark.parametrize(
    ("codecs", "slot"),
    [
        ((None, JsonCodec()), "keys of memo table 't'"),
        ((JsonCodec(), None), "values of memo table 't'"),
    ],
)
def test_a_memo_table_with_no_codec_raises_naming_the_slot(persist, codecs, slot):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    with pytest.raises(ValueError, match=slot):
        store.memo_table("t", 1, *codecs, lambda ctx, key: key)


def test_memo_table_declaration_rules(persist):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    codec = JsonCodec()
    table = store.memo_table("t", 1, codec, codec, lambda ctx, key: key)
    with pytest.raises(ValueError, match="'t'"):
        store.memo_table("t", 2, codec, codec, lambda ctx, key: key)
    memo = store.open_memo(persist.MemoryMemo())
    with pytest.raises(RuntimeError, match="open_memo"):
        store.memo_table("u", 1, codec, codec, lambda ctx, key: key)
    with pytest.raises(RuntimeError, match="already"):
        store.open_memo(persist.MemoryMemo())
    other = persist.Store.open(persist.MemoryBackend(), build_id=1)
    foreign = other.memo_table("t", 1, codec, codec, lambda ctx, key: key)
    other.open_memo(persist.MemoryMemo())
    with pytest.raises(ValueError, match="another store"):
        memo.get_or_insert(foreign, 1)
    assert store.runtime.get(memo.get_or_insert(table, 5)) == 5


def test_a_raising_release_is_logged_not_raised(persist, monkeypatch):
    reports = []
    monkeypatch.setattr(sys, "unraisablehook", reports.append)

    def fail():
        raise RuntimeError("release failed")

    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    table = store.memo_table("t", 1, JsonCodec(), JsonCodec(), lambda ctx, key: key)
    memo = store.open_memo(Wrapped(persist.MemoryMemo(), release=fail))
    store.runtime.get(memo.get_or_insert(table, 1))
    memo.checkpoint()
    assert [str(report.exc_value) for report in reports] == ["release failed"]


def test_a_backend_exception_surfaces_with_its_message(persist):
    inner = persist.MemoryMemo()

    def refuse(*args):
        raise OSError("disk gone")

    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    table = store.memo_table("t", 1, JsonCodec(), JsonCodec(), lambda ctx, key: key)
    with pytest.raises(OSError, match="disk gone"):
        store.open_memo(Wrapped(inner, hooks={"load": refuse}))

    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    table = store.memo_table("t", 1, JsonCodec(), JsonCodec(), lambda ctx, key: key)
    memo = store.open_memo(Wrapped(inner, hooks={"write_generation": refuse}))
    store.runtime.get(memo.get_or_insert(table, 1))
    with pytest.raises(OSError, match="disk gone"):
        memo.checkpoint()


def test_a_busy_backend_raises_busy_and_the_next_checkpoint_succeeds(persist):
    inner = persist.MemoryMemo()
    busy = {"on": True}

    def lock():
        if busy["on"]:
            raise persist.Busy("held elsewhere")
        return inner.lock_checkpoint()

    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    table = store.memo_table("t", 1, JsonCodec(), JsonCodec(), lambda ctx, key: key)
    memo = store.open_memo(Wrapped(inner, hooks={"lock_checkpoint": lock}))
    store.runtime.get(memo.get_or_insert(table, 1))
    with pytest.raises(persist.Busy):
        memo.checkpoint()
    assert issubclass(persist.Busy, persist.PersistError)
    busy["on"] = False
    assert memo.checkpoint()["written"] == 1


def test_a_memo_backend_of_the_wrong_shape_raises_type_error(persist):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    store.memo_table("t", 1, JsonCodec(), JsonCodec(), lambda ctx, key: key)
    with pytest.raises(TypeError, match="load"):
        store.open_memo(Wrapped(persist.MemoryMemo(), hooks={"load": lambda: 5}))


def test_incomplete_and_pending_memo_values_restore_as_themselves(pkg, persist):
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()

    def build():
        store = persist.Store.open(log, build_id="b1")
        partial = store.input("partial", None, PickleCodec())
        waiting = store.input("waiting", None, PickleCodec())
        calls = []

        def compute(ctx, key):
            calls.append(key)
            return ctx.get(partial if key == "partial" else waiting) + 1

        table = store.memo_table("t", 1, PickleCodec(), PickleCodec(), compute)
        return store, partial, waiting, table, store.open_memo(memo_backend), calls

    store, partial, waiting, table, memo, calls = build()
    store.set(partial, pkg.Incomplete(5, 2))
    store.set(waiting, pkg.Pending("k"))
    for key in ("partial", "waiting"):
        store.runtime.get(memo.get_or_insert(table, key))
    expected = {
        key: store.runtime.get_with_status(memo.get_or_insert(table, key))
        for key in ("partial", "waiting")
    }
    assert expected["partial"][1] == pkg.Incomplete(6, 2)
    assert expected["waiting"][1] == pkg.Pending("k")
    memo.checkpoint()
    store.close()

    store, partial, waiting, table, memo, calls = build()
    for key in ("partial", "waiting"):
        got = store.runtime.get_with_status(memo.get_or_insert(table, key))
        assert got == expected[key]
    assert calls == []


def test_ctx_memo_inside_a_query_and_a_memo_function(persist):
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    calls = []

    def build():
        store = persist.Store.open(log, build_id="b1")
        offset = store.input("offset", 0, JsonCodec())

        def fib(ctx, n):
            calls.append(n)
            if n < 2:
                return n + ctx.get(offset)
            return ctx.memo(table, n - 1) + ctx.memo(table, n - 2)

        table = store.memo_table("fib", 1, JsonCodec(), JsonCodec(), fib)
        memo = store.open_memo(memo_backend)
        query = store.runtime.create_query(lambda ctx: ctx.memo(table, 10) + 1)
        return store, offset, memo, query

    store, offset, memo, query = build()
    assert store.runtime.get(query) == 56
    assert sorted(calls) == list(range(11))
    store.set(offset, 1)
    assert store.runtime.get(query) == 144 + 1
    memo.checkpoint()
    store.close()

    calls.clear()
    store, offset, memo, query = build()
    assert store.runtime.get(query) == 144 + 1
    assert calls == []


def test_ctx_memo_rejects_a_table_of_another_runtime(pkg, persist):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    table = store.memo_table("t", 1, JsonCodec(), JsonCodec(), lambda ctx, key: key)
    store.open_memo(persist.MemoryMemo())
    rt = pkg.Runtime()
    node = rt.create_query(lambda ctx: ctx.memo(table, 1))
    with pytest.raises(ValueError, match="runtime"):
        rt.get(node)


def test_a_closed_store_refuses_memo_calls(persist):
    store = persist.Store.open(persist.MemoryBackend(), build_id=1)
    table = store.memo_table("t", 1, JsonCodec(), JsonCodec(), lambda ctx, key: key)
    memo = store.open_memo(persist.MemoryMemo())
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        memo.get_or_insert(table, 1)
    with pytest.raises(RuntimeError, match="closed"):
        memo.checkpoint()


def test_memory_memo_protocol(persist):
    backend = persist.MemoryMemo()
    assert backend.load() is None
    guard = backend.lock_checkpoint()
    with pytest.raises(persist.Busy):
        backend.lock_checkpoint()
    guard.release()
    guard.release()
    backend.lock_checkpoint().release()


class LoadFailsOnce:
    def __init__(self, inner):
        self.inner = inner
        self.failed = False

    def lock_checkpoint(self):
        return self.inner.lock_checkpoint()

    def load(self):
        if not self.failed:
            self.failed = True
            raise OSError("load refused")
        return self.inner.load()

    def read_entry(self, generation, item):
        return self.inner.read_entry(generation, item)

    def write_generation(self, base, generation, header, entries):
        return self.inner.write_generation(base, generation, header, entries)

    def remove_generation(self, generation):
        return self.inner.remove_generation(generation)


def test_a_failed_open_memo_keeps_the_declared_tables_for_the_retry(persist):
    store = persist.Store.open(persist.MemoryBackend(), build_id="b1")
    base = store.input("base", 2, JsonCodec())
    first = store.memo_table(
        "first", 1, JsonCodec(), JsonCodec(), lambda ctx, key: ctx.get(base) * key
    )
    second = store.memo_table(
        "second", 1, JsonCodec(), JsonCodec(), lambda ctx, key: ctx.memo(first, key) + 1
    )
    backend = LoadFailsOnce(persist.MemoryMemo())
    with pytest.raises(OSError, match="load refused"):
        store.open_memo(backend)
    memo = store.open_memo(backend)
    rt = store.runtime
    assert rt.get(memo.get_or_insert(first, 5)) == 10
    assert rt.get(memo.get_or_insert(second, 5)) == 11
    assert rt.get(rt.create_query(lambda ctx: ctx.memo(first, 3))) == 6
    store.close()


class CountedKey:
    strs = 0

    def __init__(self, n, raises=False):
        self.n = n
        self.raises = raises

    def __hash__(self):
        return hash(self.n)

    def __eq__(self, other):
        return isinstance(other, CountedKey) and self.n == other.n

    def __str__(self):
        CountedKey.strs += 1
        if self.raises:
            raise RuntimeError("no str")
        return f"k{self.n}"


class KeyCodec:
    def encode(self, obj):
        return str(obj.n).encode()

    def decode(self, data):
        return CountedKey(int(data))


@pytest.fixture
def unraisable(monkeypatch):
    monkeypatch.setattr(sys, "unraisablehook", lambda info: None)


def labelled_memo(persist, log, memo_backend):
    store = persist.Store.open(log, build_id="b1")
    double = store.memo_table(
        "double",
        1,
        KeyCodec(),
        JsonCodec(),
        lambda ctx, key: key.n * 2,
        label="prices",
    )
    total = store.memo_table(
        "total",
        1,
        KeyCodec(),
        JsonCodec(),
        lambda ctx, key: ctx.memo(double, key) + 1,
    )
    return store, double, total, store.open_memo(memo_backend)


def node_label(rt, entry):
    return rt.graph_snapshot(roots=[entry], depth=0)["nodes"][0]["label"]


def test_a_labelled_entry_is_labelled_from_its_first_compute(persist):
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    store, double, _, memo = labelled_memo(persist, log, memo_backend)
    rt = store.runtime
    rt.enable_compute_stats(True)
    assert rt.get(memo.get_or_insert(double, CountedKey(3))) == 6
    stats = rt.compute_stats()
    assert stats["prices"]["recomputed"] == 1
    assert None not in stats
    assert node_label(rt, memo.get_or_insert(double, CountedKey(3))) == "prices/k3"


def test_a_labelled_entry_formats_its_key_once(persist):
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    store, double, _, memo = labelled_memo(persist, log, memo_backend)
    rt = store.runtime
    CountedKey.strs = 0
    for _ in range(3):
        assert rt.get(memo.get_or_insert(double, CountedKey(3))) == 6
    assert CountedKey.strs == 1
    CountedKey.strs = 0
    query = rt.create_query(
        lambda ctx: ctx.memo(double, CountedKey(4)) + ctx.memo(double, CountedKey(4))
    )
    assert rt.get(query) == 16
    assert CountedKey.strs == 1


def test_an_entry_restored_as_a_dependency_is_labelled_once_read_directly(persist):
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    store, _, total, memo = labelled_memo(persist, log, memo_backend)
    assert store.runtime.get(memo.get_or_insert(total, CountedKey(5))) == 11
    memo.checkpoint()
    store.close()

    store, double, total, memo = labelled_memo(persist, log, memo_backend)
    rt = store.runtime
    assert rt.get(memo.get_or_insert(total, CountedKey(5))) == 11
    entry = memo.get_or_insert(double, CountedKey(5))
    assert node_label(rt, entry) == "prices/k5"
    assert rt.get(entry) == 10


def test_a_key_whose_str_raises_leaves_the_entry_unlabelled(persist, unraisable):
    log, memo_backend = persist.MemoryBackend(), persist.MemoryMemo()
    store, double, _, memo = labelled_memo(persist, log, memo_backend)
    rt = store.runtime
    entry = memo.get_or_insert(double, CountedKey(6, raises=True))
    assert rt.get(entry) == 12
    assert node_label(rt, entry) is None
    query = rt.create_query(lambda ctx: ctx.memo(double, CountedKey(7, raises=True)))
    assert rt.get(query) == 14
