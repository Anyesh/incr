"""R39: a Local object freed on a thread other than the one that made it is
dropped on its own thread, or on the main thread once its own has exited:
never leaked, and never reported as unraisable."""

import gc
import json
import os
import sys
import threading
import time
import weakref

import incr
import incr.persist as persist
import pytest

DEADLINE = 60


class Codec:
    collect_on_worker = False

    def encode(self, obj):
        if Codec.collect_on_worker and threading.current_thread() is not threading.main_thread():
            gc.collect()
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)


class Query:
    def __init__(self, node, escaped):
        self.node = node
        self.escaped = escaped

    def __call__(self, r):
        self.escaped.append(r)
        return r.get(self.node) + 1


@pytest.fixture
def unraisable(monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "unraisablehook", lambda u: seen.append(str(u.exc_value)))
    return seen


@pytest.fixture
def no_gc():
    gc.collect()
    gc.disable()
    yield
    gc.enable()


def every_local_class():
    """One object of each Local class, and weakrefs to Python objects only
    the runtime and the store keep."""
    rt = incr.Runtime()
    a = rt.create_input(1)
    escaped = []
    query = Query(a, escaped)
    q = rt.create_query(query)
    assert rt.get(q) == 2
    observer = rt.observe(q, lambda v: None)
    table = rt.create_table()
    col = rt.create_collection()
    col.insert(1)
    ordered = col.sort_by_key(lambda x: x)
    grouped = col.group_by(lambda x: x)
    kc = incr.KeyedCollection(rt)
    kc.upsert("a1", 1)
    parts = kc.key_partitions(lambda key: key[0])
    walker = rt.memory_walker()
    codec = Codec()
    store = persist.Store.open(persist.MemoryBackend(), build_id="b1")
    memo_table = store.memo_table("sq", 1, codec, codec, lambda ctx, k: k * k)
    memo = store.open_memo(persist.MemoryMemo())
    assert store.runtime.get(memo.get_or_insert(memo_table, 3)) == 9
    batch = store.batch()
    overlay = rt.overlay()
    objects = [rt, a, q, escaped, observer, table, col, ordered, grouped, kc, parts]
    objects += [walker, store, memo_table, memo, batch, overlay]
    return objects, [weakref.ref(query), weakref.ref(codec)]


def make_garbage():
    """A reference cycle holding one object of each Local class, which only
    a collection frees."""
    objects, refs = every_local_class()
    objects.append(objects)
    return refs


def on_thread(f):
    thread = threading.Thread(target=f)
    thread.start()
    thread.join(DEADLINE)
    assert not thread.is_alive()


def freed(refs):
    """Whether every referent dies within a few seconds of bytecode on this
    thread, where the main thread runs its pending calls."""
    deadline = time.monotonic() + 5
    while any(r() is not None for r in refs):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.001)
    return True


def live_owner(engine_call):
    """Garbage made on a thread that stays alive, collected on this one,
    and whether that thread's next call, `engine_call`, freed it."""
    made = threading.Event()
    collected = threading.Event()
    result = []

    def owner():
        refs = make_garbage()
        made.set()
        assert collected.wait(DEADLINE)
        engine_call()
        result.append(all(r() is None for r in refs))

    thread = threading.Thread(target=owner)
    thread.start()
    assert made.wait(DEADLINE)
    gc.collect()
    collected.set()
    thread.join(DEADLINE)
    assert not thread.is_alive()
    return result


def test_garbage_collected_on_another_thread_is_freed_without_a_report(unraisable, no_gc):
    refs = make_garbage()
    on_thread(gc.collect)
    assert unraisable == []
    assert freed(refs)


def test_garbage_collected_on_a_flight_worker_is_freed_without_a_report(unraisable, no_gc):
    refs = make_garbage()
    store = persist.Store.open(persist.MemoryBackend(), build_id="b1")
    table = store.memo_table("sq", 1, Codec(), Codec(), lambda ctx, k: k)
    memo = store.open_memo(persist.MemoryMemo())
    store.runtime.get(memo.get_or_insert(table, 1))
    Codec.collect_on_worker = True
    try:
        pending = memo.begin_checkpoint()
        while not pending.poll(0.05):
            pass
    finally:
        Codec.collect_on_worker = False
    assert unraisable == []
    assert freed(refs)


class Streamed:
    concurrent_compaction = True

    def __init__(self):
        self.inner = persist.MemoryBackend()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_a_pending_checkpoint_holding_the_last_store_reference_frees_it_off_thread(unraisable):
    codec = Codec()
    store = persist.Store.open(Streamed(), build_id="b1")
    base = store.input("base", 1, codec)
    store.set(base, 2)
    holder = [store.begin_checkpoint()]
    while not holder[0].poll(0.05):
        pass
    refs = [weakref.ref(codec)]
    del store, base, codec
    on_thread(holder.pop)
    assert unraisable == []
    assert freed(refs)


def test_a_pending_checkpoint_holding_the_last_memo_reference_drops_it_off_thread(unraisable):
    store = persist.Store.open(persist.MemoryBackend(), build_id="b1")
    table = store.memo_table("sq", 1, Codec(), Codec(), lambda ctx, k: k * k)
    memo = store.open_memo(persist.MemoryMemo())
    for k in range(20):
        store.runtime.get(memo.get_or_insert(table, k))
    holder = [memo.begin_checkpoint()]
    del store, memo, table
    on_thread(holder.pop)
    assert unraisable == []


def wait_exited(native_id):
    """Waits until the OS thread has run its thread-local destructors, which
    `join` does not wait for, where the platform shows it."""
    task = f"/proc/self/task/{native_id}"
    deadline = time.monotonic() + DEADLINE
    while os.path.exists(task):
        assert time.monotonic() < deadline
        time.sleep(0.001)


def test_objects_whose_thread_has_exited_are_freed_on_the_main_thread(unraisable, no_gc):
    made = []

    def make():
        made.append(make_garbage())
        made.append(every_local_class())
        made.append(threading.get_native_id())

    on_thread(make)
    wait_exited(made.pop())
    cyclic, (objects, refs) = made
    del made, objects
    gc.collect()
    assert unraisable == []
    assert freed(cyclic + refs)


def test_garbage_of_a_live_thread_is_freed_at_its_next_engine_call(unraisable, no_gc):
    def engine_call():
        rt = incr.Runtime()
        rt.get(rt.create_input(1))

    result = live_owner(engine_call)
    assert unraisable == []
    assert result == [True]


def test_garbage_of_a_live_thread_is_freed_at_its_next_store_call(unraisable, no_gc):
    def store_call():
        store = persist.Store.open(persist.MemoryBackend(), build_id="b1")
        base = store.input("base", 1, Codec())
        store.set(base, 2)

    result = live_owner(store_call)
    assert unraisable == []
    assert result == [True]


def test_a_local_object_used_on_another_thread_raises():
    rt = incr.Runtime()
    node = rt.create_input(1)
    raised = []

    def use():
        try:
            rt.get(node)
        except BaseException as error:
            raised.append(str(error))

    on_thread(use)
    assert len(raised) == 1
    assert "is unsendable, but sent to another thread" in raised[0]


def test_a_foreign_node_leaves_the_runtime_unchanged():
    rt = incr.Runtime()
    node = rt.create_input(1)
    doubled = rt.create_query(lambda r: r.get(node) * 2)
    assert rt.get(doubled) == 2
    made = []
    on_thread(lambda: made.append(incr.Runtime().create_input(5)))
    with pytest.raises(BaseException, match="unsendable, but sent to another thread"):
        rt.set(made[0], 7)
    rt.set(node, 3)
    assert rt.get(doubled) == 6
