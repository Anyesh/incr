"""A store's checkpoint hands a backend with `compact_streamed(upto, records)`
a lazy iterator of record `bytes`, valid only during the call and on its
thread; a backend without it gets `compact(upto, list)` as before."""

import json
import threading

import incr.persist
import incr_concurrent.persist
import pytest

PACKAGES = [
    pytest.param(incr.persist, id="local"),
    pytest.param(incr_concurrent.persist, id="shared"),
]


@pytest.fixture(params=PACKAGES)
def persist(request):
    return request.param


class JsonCodec:
    def __init__(self):
        self.fail_on_encode = None

    def encode(self, obj):
        if self.fail_on_encode is not None and obj == self.fail_on_encode:
            raise ValueError("encode refused")
        return json.dumps(obj).encode()

    def decode(self, data):
        return json.loads(data)


class Recorder:
    """Delegates the backend protocol to a `MemoryBackend` and records what
    `compact` receives."""

    def __init__(self, persist):
        self.inner = persist.MemoryBackend()
        self.compactions = []

    def append(self, records):
        return self.inner.append(records)

    def sync(self, upto):
        return self.inner.sync(upto)

    def read_from(self, after):
        return self.inner.read_from(after)

    def load_snapshot(self):
        return self.inner.load_snapshot()

    def snapshot_id(self):
        return self.inner.snapshot_id()

    def writable(self):
        return self.inner.writable()

    def compact(self, upto, records):
        self.compactions.append(("compact", upto, records))
        return self.inner.compact(upto, records)


class Streaming(Recorder):
    def compact_streamed(self, upto, records):
        self.compactions.append(("streamed", upto, records))
        return self.inner.compact_streamed(upto, records)


class Graph:
    def __init__(self, persist, backend, codec=None):
        self.codec = codec or JsonCodec()
        self.store = persist.Store.open(backend, build_id="b1")
        self.rt = self.store.runtime
        self.base = self.store.input("base", 1, self.codec)
        self.rows = self.store.collection("rows", self.codec)

    def fill(self, rows=6):
        self.store.set(self.base, 41)
        for row in range(rows):
            self.store.insert(self.rows, row)

    def state(self):
        return self.rt.get(self.base), self.rt.get(self.rows.reduce(sorted))


def checkpointed(persist, backend_type):
    backend = backend_type(persist)
    graph = Graph(persist, backend)
    graph.fill()
    graph.store.checkpoint()
    return backend, graph


def test_a_streaming_backend_gets_a_lazy_iterator_of_the_records_compact_gets(persist):
    seen = []

    class Collecting(Streaming):
        def compact_streamed(self, upto, records):
            seen.append(iter(records) is records and not isinstance(records, list))
            collected = list(records)
            self.compactions.append(("streamed", upto, collected))
            return self.inner.compact(upto, collected)

    plain, _ = checkpointed(persist, Recorder)
    streamed, _ = checkpointed(persist, Collecting)
    assert seen == [True]
    [(kind, upto, records)] = streamed.compactions
    [(plain_kind, plain_upto, plain_records)] = plain.compactions
    assert (kind, plain_kind) == ("streamed", "compact")
    assert isinstance(plain_records, list)
    assert upto == plain_upto
    assert records == plain_records
    assert records and all(isinstance(record, bytes) for record in records)


def test_a_backend_without_compact_streamed_still_gets_a_list(persist):
    backend, _ = checkpointed(persist, Recorder)
    [(kind, _, records)] = backend.compactions
    assert kind == "compact"
    assert type(records) is list


def test_the_iterator_is_invalid_after_the_call_and_on_other_threads(persist):
    outcomes = {}

    class Probing(Streaming):
        def compact_streamed(self, upto, records):
            def pull():
                try:
                    next(records)
                except RuntimeError as error:
                    outcomes["thread"] = str(error)

            worker = threading.Thread(target=pull)
            worker.start()
            worker.join()
            self.compactions.append(("streamed", upto, records))
            return self.inner.compact_streamed(upto, records)

    backend, _ = checkpointed(persist, Probing)
    assert "thread" in outcomes
    [(_, _, records)] = backend.compactions
    with pytest.raises(RuntimeError, match="only valid during"):
        next(records)


def test_a_codec_that_raises_while_encoding_fails_the_checkpoint_and_changes_nothing(
    persist,
):
    backend = Streaming(persist)
    graph = Graph(persist, backend)
    graph.fill()
    graph.store.checkpoint()
    before = (backend.load_snapshot(), backend.snapshot_id())
    graph.store.insert(graph.rows, 99)
    graph.codec.fail_on_encode = 3
    with pytest.raises(ValueError, match="encode refused"):
        graph.store.checkpoint()
    assert (backend.load_snapshot(), backend.snapshot_id()) == before
    graph.codec.fail_on_encode = None
    graph.store.checkpoint()
    assert backend.snapshot_id() != before[1]


def test_restore_after_a_streamed_compaction_equals_restore_after_compact(persist):
    restored = []
    for backend_type in (Recorder, Streaming):
        backend, graph = checkpointed(persist, backend_type)
        graph.store.insert(graph.rows, 100)
        graph.store.close()
        restored.append(Graph(persist, backend).state())
        assert backend.snapshot_id() is not None
    assert restored[0] == restored[1] == (41, [0, 1, 2, 3, 4, 5, 100])


def test_the_reference_memory_backend_streams_a_python_iterable(persist):
    source = persist.MemoryBackend()
    graph = Graph(persist, source)
    graph.fill()
    graph.store.checkpoint()
    _, records = source.load_snapshot()
    other = persist.MemoryBackend()
    tail = other.append(records)
    other.compact_streamed(tail, (record for record in records))
    assert other.load_snapshot() == (tail, records)


def test_a_failure_while_iterating_reaches_the_caller_unchanged_and_changes_nothing(
    persist,
):
    backend = persist.MemoryBackend()
    graph = Graph(persist, backend)
    graph.fill()
    graph.store.checkpoint()
    before = (backend.load_snapshot(), backend.snapshot_id())
    tail = backend.read_from(before[0][0])

    class Boom(Exception):
        pass

    def records():
        yield before[0][1][0]
        raise Boom("pulled")

    with pytest.raises(Boom, match="pulled"):
        backend.compact_streamed(before[0][0], records())
    assert (backend.load_snapshot(), backend.snapshot_id()) == before
    assert backend.read_from(before[0][0]) == tail


def test_a_backend_returning_with_records_left_fails_the_checkpoint_and_poisons(
    persist,
):
    class Truncating(Streaming):
        def compact_streamed(self, upto, records):
            next(records)

    backend = Truncating(persist)
    graph = Graph(persist, backend)
    graph.fill()
    with pytest.raises(persist.PersistError):
        graph.store.checkpoint()
    with pytest.raises(persist.PersistError, match="poisoned"):
        graph.store.set(graph.base, 2)


def test_a_backend_committing_after_a_pull_raised_fails_the_checkpoint_and_poisons(
    persist,
):
    class Swallowing(Streaming):
        def compact_streamed(self, upto, records):
            kept = []
            try:
                for record in records:
                    kept.append(record)
            except RuntimeError:
                pass
            self.inner.compact(upto, kept)

    backend = Swallowing(persist)
    graph = Graph(persist, backend)
    graph.fill()
    graph.codec.fail_on_encode = 3
    with pytest.raises(ValueError, match="encode refused"):
        graph.store.checkpoint()
    graph.codec.fail_on_encode = None
    with pytest.raises(persist.PersistError, match="poisoned"):
        graph.store.set(graph.base, 2)
