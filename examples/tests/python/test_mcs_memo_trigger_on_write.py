"""A Shared memo whose computes all run inside `store.batch(apply="on_write")`
blocks: its checkpoint thread, woken by the trigger while a block is open,
skips that checkpoint and runs it once the block closes, so unsaved entries
end below the trigger instead of growing without bound. Local runs the trigger on the writing thread
after each commit, so it is skipped here."""

import json
import time

import incr.persist
import incr_concurrent.persist
import pytest

PACKAGES = [
    pytest.param(incr.persist, id="local"),
    pytest.param(incr_concurrent.persist, id="shared"),
]

TRIGGER = 5
BLOCKS = 20
KEYS_PER_BLOCK = 3
TIMEOUT_SECS = 10


@pytest.fixture(params=PACKAGES)
def persist(request):
    if request.param is incr.persist:
        pytest.skip("Local runs the trigger on the writing thread")
    return request.param


class JsonCodec:
    def encode(self, obj):
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)


def test_on_write_blocks_that_compute_get_their_entries_checkpointed(persist):
    store = persist.Store.open(persist.MemoryBackend(), build_id="b1")
    tick = store.input("tick", 0, JsonCodec())
    base = store.input("base", 0, JsonCodec())
    table = store.memo_table(
        "sq", 1, JsonCodec(), JsonCodec(), lambda ctx, k: k * k + ctx.get(base)
    )
    memo = store.open_memo(persist.MemoryMemo(), checkpoint_after_entries=TRIGGER)
    k = 0
    for block in range(BLOCKS):
        with store.batch(apply="on_write") as b:
            b.set(tick, block + 1)
            for _ in range(KEYS_PER_BLOCK):
                assert store.runtime.get(memo.get_or_insert(table, k)) == k * k
                k += 1
    deadline = time.monotonic() + TIMEOUT_SECS
    while not settled(memo.stats()):
        assert time.monotonic() < deadline, (
            f"a due checkpoint never ran: {memo.stats()}"
        )
        time.sleep(0.01)
    assert memo.stats()["auto_checkpoint_errors"] == 0


def settled(stats):
    """Whether fewer entries than the trigger are unsaved and every other
    entry is saved, so no checkpoint is due or running."""
    unsaved = stats["unsaved_entries"]
    return unsaved < TRIGGER and stats["entries"] + unsaved == BLOCKS * KEYS_PER_BLOCK
