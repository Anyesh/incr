"""The Shared store across Python threads: a batch stays on the thread that
opened it, drops never wait on the store with the GIL held, and several
threads write and read one store."""

import pickle
import threading

import pytest
from incr_concurrent.persist import MemoryBackend, Store


class PickleCodec:
    def encode(self, obj):
        return pickle.dumps(obj, protocol=4)

    def decode(self, data):
        return pickle.loads(data)


def in_thread(f):
    out = {}

    def run():
        try:
            out["result"] = f()
        except BaseException as e:
            out["error"] = e

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    if "error" in out:
        raise out["error"]
    return out.get("result")


@pytest.mark.parametrize("apply", ["on_exit", "on_write"])
def test_a_batch_raises_on_a_thread_other_than_its_opener(apply):
    store = Store.open(MemoryBackend(), build_id=1)
    price = store.input("price", 0, PickleCodec())
    with store.batch(apply=apply) as b:
        with pytest.raises(RuntimeError, match="thread"):
            in_thread(lambda: b.set(price, 1))
        b.set(price, 2)
    assert store.runtime.get(price) == 2
    unopened = store.batch()
    with pytest.raises(RuntimeError, match="thread"):
        in_thread(unopened.__enter__)
