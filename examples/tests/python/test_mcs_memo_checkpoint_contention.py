"""A Shared memo's automatic checkpoint, running on its own thread while
the caller keeps reading memo entries, costs the caller about what the
checkpoint's own work costs. Each entry reads a row of a keyed collection,
so the checkpoint clones, encodes and decodes every entry's key and value
and maps its row; doing that with one GIL acquisition per value made the
caller's pass several times slower than the same pass without the
trigger. Local checkpoints on the writing thread, so only Shared runs."""

import pickle
import time

import incr_concurrent.persist as persist

ENTRIES = 60_000
SLOWDOWN_LIMIT = 3.0
ATTEMPTS = 2
TIMEOUT_SECS = 30


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


def first_pass_secs(memo_options):
    store = persist.Store.open(persist.MemoryBackend(), build_id="contention")
    rows = store.keyed_collection("rows", PickleCodec(), PickleCodec(), label="rows")
    store.reconcile(rows, {k: k for k in range(ENTRIES)})
    table = store.memo_table(
        "price",
        1,
        PickleCodec(),
        PickleCodec(),
        lambda ctx, k: ctx.get_key(rows, k) * 2,
    )
    memo = store.open_memo(persist.MemoryMemo(), **memo_options)
    rt = store.runtime
    started = time.perf_counter()
    for k in range(ENTRIES):
        assert rt.get(memo.get_or_insert(table, k)) == k * 2
    elapsed = time.perf_counter() - started
    if memo_options:
        deadline = time.monotonic() + TIMEOUT_SECS
        while memo.stats()["entries"] == 0:
            assert time.monotonic() < deadline, "the trigger never checkpointed"
            time.sleep(0.01)
    store.close()
    return elapsed


def test_an_automatic_checkpoint_does_not_slow_the_reading_thread_down_severalfold():
    plain = min(first_pass_secs({}) for _ in range(ATTEMPTS))
    triggered = min(
        first_pass_secs({"checkpoint_after_entries": ENTRIES // 2})
        for _ in range(ATTEMPTS)
    )
    assert triggered <= SLOWDOWN_LIMIT * plain, (
        f"first pass took {triggered:.2f} s with the trigger, {plain:.2f} s without"
    )
