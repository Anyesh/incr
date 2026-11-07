import pickle

import pytest
from incr.persist import MemoryBackend, MemoryMemo, Store


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


class FlakyDecode(PickleCodec):
    """Raises the second time it decodes the same bytes: the store decodes
    a foreign record first, so the binding's own decode of it fails."""

    def __init__(self):
        self.seen = {}
        self.raised = 0
        self.armed = True

    def decode(self, data):
        self.seen[data] = self.seen.get(data, 0) + 1
        if self.armed and self.seen[data] == 2:
            self.raised += 1
            raise RuntimeError("second decode refused")
        return super().decode(data)


class Poison:
    armed = False

    def __init__(self, n):
        self.n = n

    def __hash__(self):
        if Poison.armed:
            raise RuntimeError("hash failed")
        return hash(self.n)

    def __eq__(self, other):
        return isinstance(other, Poison) and self.n == other.n

    def __reduce__(self):
        return (Poison, (self.n,))


CODEC = PickleCodec()


def total_of(rows):
    return rows.aggregate(0, lambda r: r, lambda a, b: a + b)


def rows_of(rt, collection):
    return rt.get(collection.reduce(sorted))


def keyed(store, name="pairs", codec=CODEC):
    return store.keyed_collection(name, CODEC, codec)


def snapshot(backend):
    return sorted(backend.load_snapshot()[1])


def test_overlay_over_a_persisted_input_and_keyed_collection_restores_the_base():
    backend = MemoryBackend()
    store = Store.open(backend, build_id=1)
    rt = store.runtime
    count = store.input("count", 0, CODEC)
    kc = keyed(store)
    store.set(count, 5)
    store.reconcile(kc, {"a": 1, "b": 2})
    doubled = rt.create_query(lambda ctx: ctx.get(count) * 2)
    total = total_of(kc.collection)
    store.checkpoint()
    base = snapshot(backend)

    with rt.overlay() as ov:
        ov.set(count, 9)
        rt.set(count, 11)
        kc.upsert("a", 10)
        kc.upsert("c", 3)
        kc.remove("b")
        assert kc.reconcile({"a": 10, "c": 3, "d": 4}) == (1, 0, 0)
        assert ov.get(count) == 11
        assert ov.get(doubled) == 22
        assert (kc.get("a"), kc.get("b"), kc.get("d"), len(kc)) == (10, None, 4, 3)
        assert ov.get(total) == 17
    assert (rt.get(count), rt.get(doubled), rt.get(total)) == (5, 10, 3)
    assert (kc.get("a"), kc.get("b"), kc.get("c"), len(kc)) == (1, 2, None, 2)
    assert rows_of(rt, kc.collection) == [1, 2]

    store.checkpoint()
    assert snapshot(backend) == base
    store.close()

    again = Store.open(backend, build_id=1)
    assert again.runtime.get(again.input("count", 0, CODEC)) == 5
    reopened = keyed(again)
    assert (reopened.get("a"), reopened.get("b"), len(reopened)) == (1, 2, 2)


def test_overlay_over_a_persisted_collection_restores_the_base():
    backend = MemoryBackend()
    store = Store.open(backend, build_id=1)
    rt = store.runtime
    rows = store.collection("rows", CODEC)
    store.extend(rows, [1, 2, 3, 3])
    total = total_of(rows)
    store.checkpoint()
    base = snapshot(backend)

    with rt.overlay() as ov:
        rows.insert(10)
        assert rows.delete(3) is True
        assert rows.replace(1, 100) is True
        rows.extend([5, 5])
        assert ov.get(total) == 2 + 3 + 10 + 100 + 10
    assert rt.get(total) == 9
    assert rows_of(rt, rows) == [1, 2, 3, 3]

    store.checkpoint()
    assert snapshot(backend) == base
    with pytest.raises(RuntimeError, match="store.insert"):
        rows.insert(1)


@pytest.mark.parametrize("reopen", [False, True])
def test_removing_a_base_keyed_pair_in_an_overlay_restores_index_and_collection(reopen):
    backend = MemoryBackend()
    store = Store.open(backend, build_id=1)
    kc = keyed(store)
    store.reconcile(kc, {"a": 1, "b": 2})
    if reopen:
        store.close()
        store = Store.open(backend, build_id=1)
        kc = keyed(store)
    rt = store.runtime

    with rt.overlay():
        kc.remove("a")
        assert (kc.get("a"), len(kc)) == (None, 1)
        assert rows_of(rt, kc.collection) == [2]
    with rt.overlay():
        kc.remove("a")
        kc.upsert("a", 7)
        kc.remove("a")
        kc.upsert("a", 8)
        assert kc.get("a") == 8
        kc.reconcile({})
        assert len(kc) == 0
    assert (kc.get("a"), kc.get("b"), len(kc)) == (1, 2, 2)
    assert rows_of(rt, kc.collection) == [1, 2]

    store.upsert(kc, "a", 7)
    assert len(kc) == 2
    assert rows_of(rt, kc.collection) == [2, 7]
    store.remove(kc, "b")
    assert (kc.get("b"), len(kc)) == (None, 1)
    assert rows_of(rt, kc.collection) == [7]
    store.checkpoint()
    store.close()
    again = keyed(Store.open(backend, build_id=1))
    assert (again.get("a"), again.get("b"), len(again)) == (7, None, 1)


def test_a_body_that_raises_restores_and_leaves_the_store_usable():
    backend = MemoryBackend()
    store = Store.open(backend, build_id=1)
    count = store.input("count", 1, CODEC)
    with pytest.raises(ValueError), store.runtime.overlay() as ov:
        ov.set(count, 2)
        raise ValueError("boom")
    assert store.runtime.get(count) == 1
    store.set(count, 3)
    store.checkpoint()
    assert store.runtime.get(count) == 3


def test_a_failed_restore_poisons_the_runtime_and_every_store_call():
    backend = MemoryBackend()
    store = Store.open(backend, build_id=1)
    count = store.input("count", 0, CODEC)
    rows = store.collection("rows", CODEC)
    store.insert(rows, Poison(1))
    try:
        with pytest.raises(RuntimeError), store.runtime.overlay():
            rows.delete(Poison(1))
            Poison.armed = True
    finally:
        Poison.armed = False
    with pytest.raises(RuntimeError, match="poisoned"):
        store.runtime.get(count)
    for call in (
        lambda: store.checkpoint(),
        lambda: store.refresh(),
        lambda: store.set(count, 1),
        lambda: store.insert(rows, Poison(2)),
        lambda: store.delete(rows, Poison(1)),
        lambda: store.replace(rows, Poison(1), Poison(2)),
        lambda: store.extend(rows, [Poison(3)]),
        lambda: store.batch(),
        lambda: store.input("other", 0, CODEC),
        lambda: store.collection("other", CODEC),
    ):
        with pytest.raises(RuntimeError):
            call()
    store.close()
    again = Store.open(backend, build_id=1)
    assert rows_of(again.runtime, again.collection("rows", CODEC)) == [Poison(1)]


def test_a_dropped_overlay_scope_does_not_leave_the_store_usable():
    store = Store.open(MemoryBackend(), build_id=1)
    count = store.input("count", 0, CODEC)
    overlay = store.runtime.overlay()
    overlay.__enter__()
    with pytest.raises(RuntimeError):
        store.set(count, 1)
    del overlay


def test_every_store_call_raises_inside_an_overlay_and_the_overlay_still_restores():
    store = Store.open(MemoryBackend(), build_id=1)
    rt = store.runtime
    count = store.input("count", 1, CODEC)
    rows = store.collection("rows", CODEC)
    kc = keyed(store)
    store.memo_table("t", 1, CODEC, CODEC, lambda ctx, key: key)
    memo = store.open_memo(MemoryMemo())
    store.insert(rows, 1)
    batch = store.batch()
    batch.__enter__()
    calls = {
        "set": lambda: store.set(count, 2),
        "insert": lambda: store.insert(rows, 2),
        "delete": lambda: store.delete(rows, 1),
        "replace": lambda: store.replace(rows, 1, 2),
        "extend": lambda: store.extend(rows, [2]),
        "batch": lambda: store.batch(),
        "batch.set": lambda: batch.set(count, 2),
        "batch.insert": lambda: batch.insert(rows, 2),
        "batch.delete": lambda: batch.delete(rows, 1),
        "batch.replace": lambda: batch.replace(rows, 1, 2),
        "batch.upsert": lambda: batch.upsert(kc, "a", 1),
        "batch.remove": lambda: batch.remove(kc, "a"),
        "batch.reconcile": lambda: batch.reconcile(kc, {}),
        "upsert": lambda: store.upsert(kc, "a", 1),
        "remove": lambda: store.remove(kc, "a"),
        "reconcile": lambda: store.reconcile(kc, {}),
        "refresh": lambda: store.refresh(),
        "checkpoint": lambda: store.checkpoint(),
        "close": lambda: store.close(),
        "input": lambda: store.input("other", 0, CODEC),
        "collection": lambda: store.collection("other", CODEC),
        "keyed_collection": lambda: keyed(store, "other"),
        "memo_table": lambda: store.memo_table("u", 1, CODEC, CODEC, lambda c, k: k),
        "open_memo": lambda: store.open_memo(MemoryMemo()),
        "memo.checkpoint": lambda: memo.checkpoint(),
    }
    with rt.overlay() as ov:
        ov.set(count, 5)
        for call in calls.values():
            with pytest.raises(RuntimeError, match="overlay"):
                call()
    assert rt.get(count) == 1
    batch.__exit__(None, None, None)
    store.set(count, 2)
    assert rt.get(count) == 2


class Graph:
    """A store with persisted inputs `base` and `other` and memo table
    `scaled`, which multiplies `base` by its key and records each call."""

    def __init__(self, log, memo_backend, open_memo=True):
        self.calls = []
        self.store = Store.open(log, build_id="b1")
        self.rt = self.store.runtime
        self.base = self.store.input("base", 1, CODEC)
        self.other = self.store.input("other", 0, CODEC)
        self.table = self.store.memo_table("scaled", 1, CODEC, CODEC, self.compute)
        self.memo = self.store.open_memo(memo_backend) if open_memo else None

    def open_memo(self, memo_backend):
        self.memo = self.store.open_memo(memo_backend)

    def compute(self, ctx, key):
        self.calls.append(key)
        return ctx.get(self.base) * key

    def value(self, key):
        return self.rt.get(self.memo.get_or_insert(self.table, key))


def seeded(log, memo_backend):
    graph = Graph(log, memo_backend)
    graph.store.set(graph.base, 10)
    assert graph.value(2) == 20
    assert graph.memo.checkpoint()["written"] == 1
    graph.store.close()
    return Graph(log, memo_backend)


def test_memo_inside_an_overlay_computes_fresh_what_it_would_restore():
    log, memo_backend = MemoryBackend(), MemoryMemo()
    graph = seeded(log, memo_backend)
    with graph.rt.overlay() as ov:
        ov.set(graph.other, 5)
        assert graph.value(2) == 20
        assert graph.calls == [2]
    assert graph.value(2) == 20
    graph.store.close()

    again = seeded(log, memo_backend)
    assert again.value(2) == 20
    assert again.calls == []


def test_a_memo_value_computed_in_an_overlay_never_reaches_the_memo():
    log, memo_backend = MemoryBackend(), MemoryMemo()
    graph = seeded(log, memo_backend)
    with graph.rt.overlay() as ov:
        ov.set(graph.base, 20)
        assert graph.value(2) == 40
        assert graph.value(4) == 80
        assert graph.calls == [2, 4]
    assert (graph.value(2), graph.value(4)) == (20, 40)
    graph.memo.checkpoint()
    graph.store.close()

    again = Graph(log, memo_backend)
    assert (again.value(2), again.value(4)) == (20, 40)
    assert again.calls == []


def test_ctx_memo_inside_an_overlay_computes_fresh():
    log, memo_backend = MemoryBackend(), MemoryMemo()
    seen = []

    def declare(store, base):
        def inner(ctx, key):
            seen.append(("inner", key))
            return ctx.get(base) * key

        def outer(ctx, key):
            seen.append(("outer", key))
            return ctx.memo(inner_table, key) + 1

        inner_table = store.memo_table("inner", 1, CODEC, CODEC, inner)
        outer_table = store.memo_table("outer", 1, CODEC, CODEC, outer)
        return outer_table, store.open_memo(memo_backend)

    store = Store.open(log, build_id="b1")
    base = store.input("base", 1, CODEC)
    other = store.input("other", 0, CODEC)
    store.set(base, 10)
    table, memo = declare(store, base)
    assert store.runtime.get(memo.get_or_insert(table, 3)) == 31
    memo.checkpoint()
    store.close()

    seen.clear()
    store = Store.open(log, build_id="b1")
    base = store.input("base", 1, CODEC)
    other = store.input("other", 0, CODEC)
    table, memo = declare(store, base)
    rt = store.runtime
    with rt.overlay() as ov:
        ov.set(other, 1)
        assert rt.get(memo.get_or_insert(table, 3)) == 31
    assert seen == [("outer", 3), ("inner", 3)]


def test_a_memo_entry_over_an_input_the_overlay_touched_survives_a_checkpoint():
    log, memo_backend = MemoryBackend(), MemoryMemo()
    graph = seeded(log, memo_backend)
    assert graph.value(2) == 20
    assert graph.calls == []
    with graph.rt.overlay() as ov:
        ov.set(graph.base, 50)
        ov.set(graph.base, 60)
    stats = graph.memo.checkpoint()
    assert (stats["merged"], stats["unkept"]) == (1, 0)
    graph.store.close()

    again = Graph(log, memo_backend)
    assert again.value(2) == 20
    assert again.calls == []


def test_a_fresh_entry_over_an_input_the_overlay_touched_is_written():
    log, memo_backend = MemoryBackend(), MemoryMemo()
    graph = Graph(log, memo_backend)
    graph.store.set(graph.base, 10)
    assert graph.value(3) == 30
    with graph.rt.overlay() as ov:
        ov.set(graph.base, 99)
    assert graph.value(3) == 30
    assert graph.memo.checkpoint()["written"] == 1
    graph.store.close()

    again = Graph(log, memo_backend)
    assert again.value(3) == 30
    assert again.calls == []


def test_a_memo_opened_after_an_overlay_vouches_for_the_inputs_it_touched():
    log, memo_backend = MemoryBackend(), MemoryMemo()
    graph = Graph(log, memo_backend, open_memo=False)
    graph.store.set(graph.base, 10)
    with graph.rt.overlay() as ov:
        ov.set(graph.base, 99)
    graph.open_memo(memo_backend)
    assert graph.value(3) == 30
    assert graph.memo.checkpoint()["written"] == 1
    graph.store.close()

    again = Graph(log, memo_backend)
    assert again.value(3) == 30
    assert again.calls == []


def leader_and_standby():
    backend = MemoryBackend()
    leader = Store.open(backend, build_id=1)
    standby = Store.open(backend, build_id=1)
    return leader, keyed(leader), standby, keyed(standby)


def settled(leader, kc_leader, standby, kc_standby):
    leader.refresh()
    standby.refresh()
    return [
        (len(kc), rows_of(store.runtime, kc.collection))
        for store, kc in ((leader, kc_leader), (standby, kc_standby))
    ]


def test_a_standby_upsert_of_a_key_the_leader_just_wrote_leaves_one_pair():
    leader, kc_leader, standby, kc_standby = leader_and_standby()
    leader.upsert(kc_leader, "k", 1)
    standby.upsert(kc_standby, "k", 2)
    assert (len(kc_standby), kc_standby.get("k")) == (1, 2)
    assert rows_of(standby.runtime, kc_standby.collection) == [2]
    assert settled(leader, kc_leader, standby, kc_standby) == [(1, [2]), (1, [2])]
    assert (kc_leader.get("k"), kc_standby.get("k")) == (2, 2)


def test_a_standby_upsert_over_a_row_the_leader_replaced_leaves_one_pair():
    leader, kc_leader, standby, kc_standby = leader_and_standby()
    leader.upsert(kc_leader, "k", 1)
    standby.refresh()
    leader.upsert(kc_leader, "k", 5)
    standby.upsert(kc_standby, "k", 9)
    assert (len(kc_standby), kc_standby.get("k")) == (1, 9)
    assert settled(leader, kc_leader, standby, kc_standby) == [(1, [9]), (1, [9])]


def test_a_standby_remove_of_a_key_the_leader_replaced_removes_it_everywhere():
    leader, kc_leader, standby, kc_standby = leader_and_standby()
    leader.upsert(kc_leader, "k", 1)
    standby.refresh()
    leader.upsert(kc_leader, "k", 5)
    standby.remove(kc_standby, "k")
    assert (len(kc_standby), kc_standby.get("k")) == (0, None)
    assert settled(leader, kc_leader, standby, kc_standby) == [(0, []), (0, [])]


def test_a_standby_reconcile_after_the_leader_wrote_leaves_exactly_its_rows():
    leader, kc_leader, standby, kc_standby = leader_and_standby()
    leader.upsert(kc_leader, "k", 1)
    leader.upsert(kc_leader, "z", 7)
    standby.reconcile(kc_standby, {"k": 2, "m": 3})
    assert settled(leader, kc_leader, standby, kc_standby) == [
        (2, [2, 3]),
        (2, [2, 3]),
    ]
    assert (kc_standby.get("z"), kc_leader.get("z")) == (None, None)


def test_a_batch_open_across_a_refresh_composes_with_the_leaders_rows():
    leader, kc_leader, standby, kc_standby = leader_and_standby()
    with standby.batch() as b:
        b.upsert(kc_standby, "k", 2)
        leader.upsert(kc_leader, "k", 1)
        standby.refresh()
    assert settled(leader, kc_leader, standby, kc_standby) == [(1, [2]), (1, [2])]


def test_a_standby_write_that_catches_up_other_keys_changes_nothing_else():
    leader, kc_leader, standby, kc_standby = leader_and_standby()
    leader.upsert(kc_leader, "a", 1)
    standby.upsert(kc_standby, "b", 2)
    assert settled(leader, kc_leader, standby, kc_standby) == [(2, [1, 2]), (2, [1, 2])]


def test_the_index_is_rebuilt_when_applying_a_foreign_record_fails():
    backend = MemoryBackend()
    leader = Store.open(backend, build_id=1)
    standby = Store.open(backend, build_id=1)
    kc_leader = keyed(leader)
    flaky = FlakyDecode()
    kc_standby = keyed(standby, codec=flaky)
    leader.upsert(kc_leader, "a", 1)
    leader.upsert(kc_leader, "b", 2)
    standby.refresh()
    assert flaky.raised == 1
    flaky.armed = False
    assert (kc_standby.get("a"), kc_standby.get("b"), len(kc_standby)) == (1, 2, 2)
    standby.upsert(kc_standby, "a", 5)
    assert len(kc_standby) == 2
    assert rows_of(standby.runtime, kc_standby.collection) == [2, 5]
    standby.remove(kc_standby, "b")
    assert (kc_standby.get("b"), len(kc_standby)) == (None, 1)


def routing_failure_source(rt):
    rows = rt.create_collection()
    state = {"fail": False}

    def key(row):
        if state["fail"] and row == "bad":
            raise RuntimeError("no key")
        return row

    grouped = rows.group_by(key)
    return rows, grouped, state


def test_a_retried_set_of_a_persisted_input_after_a_routing_failure_is_restored():
    store = Store.open(MemoryBackend(), build_id=1)
    rt = store.runtime
    rows, _, state = routing_failure_source(rt)
    n = store.input("n", 1, CODEC)
    with rt.overlay() as ov:
        state["fail"] = True
        rows.insert("bad")
        with pytest.raises(RuntimeError, match="routing failed"):
            ov.set(n, 5)
        state["fail"] = False
        ov.set(n, 5)
        assert ov.get(n) == 5
    assert rt.get(n) == 1
    store.checkpoint()
    store.close()


def test_an_overlay_that_cannot_enter_on_a_stored_routing_failure_leaves_the_store_usable():
    store = Store.open(MemoryBackend(), build_id=1)
    rt = store.runtime
    rows, _, state = routing_failure_source(rt)
    n = store.input("n", 1, CODEC)
    state["fail"] = True
    rows.insert("bad")
    with pytest.raises(RuntimeError, match="routing failed"):
        with rt.overlay():
            pass
    state["fail"] = False
    store.set(n, 2)
    store.checkpoint()
    with rt.overlay() as ov:
        ov.set(n, 9)
        assert ov.get(n) == 9
    assert rt.get(n) == 2
    store.set(n, 3)
    store.close()


def test_an_overlay_exit_whose_settle_raises_still_closes_the_scope():
    store = Store.open(MemoryBackend(), build_id=1)
    rt = store.runtime
    rows, _, state = routing_failure_source(rt)
    n = store.input("n", 1, CODEC)
    with pytest.raises(RuntimeError, match="routing failed"):
        with rt.overlay() as ov:
            ov.set(n, 9)
            state["fail"] = True
            rows.insert("bad")
    state["fail"] = False
    assert rt.get(n) == 1
    store.set(n, 4)
    store.checkpoint()
    assert rt.get(n) == 4
    store.close()


class Probe:
    """Hashes like every other probe and, when compared, runs `hook` once:
    a store call made while a lookup holds the index of its collection."""

    hook = None

    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return 7

    def __eq__(self, other):
        hook, Probe.hook = Probe.hook, None
        if hook:
            hook()
        return False

    def __reduce__(self):
        return (Probe, (self.name,))


def standby_whose_drain_hits_a_held_index():
    backend = MemoryBackend()
    leader = Store.open(backend, build_id=1)
    standby = Store.open(backend, build_id=1)
    kc_leader = keyed(leader, "a")
    held = keyed(standby, "a")
    other = keyed(standby, "b")
    standby.upsert(held, Probe("x"), 1)
    leader.upsert(kc_leader, "y", 2)
    Probe.hook = lambda: standby.upsert(other, "new", 1)
    with pytest.raises(RuntimeError, match="during an operation"):
        held.get(Probe("probe"))
    return standby, held, other


def test_a_drain_error_after_the_append_leaves_the_index_holding_the_logged_keys():
    standby, held, other = standby_whose_drain_hits_a_held_index()
    assert (other.get("new"), len(other)) == (1, 1)
    assert rows_of(standby.runtime, other.collection) == [1]


def test_an_index_a_failed_drain_left_behind_is_rebuilt_by_the_next_store_call():
    standby, held, other = standby_whose_drain_hits_a_held_index()
    standby.refresh()
    assert (held.get("y"), len(held)) == (2, 2)
    assert rows_of(standby.runtime, held.collection) == [1, 2]
