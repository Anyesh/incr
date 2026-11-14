"""The conformance API of each package's `persist` module: `persist` is the
package's module, and a backend raises that module's exceptions."""

import incr.persist
import incr_concurrent.persist
import pytest
from test_mcs_persist_conformance import (
    DictBackend,
    DictLogFactory,
    DictMemoFactory,
    EpochMismatch,
    PickleCodec,
    SqliteLog,
    SqliteLogFactory,
)

PACKAGES = [
    pytest.param(incr.persist, id="local"),
    pytest.param(incr_concurrent.persist, id="shared"),
]


@pytest.fixture(params=PACKAGES)
def persist(request):
    return request.param


@pytest.fixture
def conformance(persist):
    return persist.conformance


class MemoryLogFactory:
    def __init__(self, persist):
        self.persist = persist

    def new_log(self):
        return self.persist.MemoryBackend()

    def open(self, log):
        return log

    def open_read_only(self, log):
        return log.read_only()

    def multi_writer(self):
        return True


class MemoryMemoFactory:
    def __init__(self, persist):
        self.persist = persist

    def new_store(self):
        return self.persist.MemoryMemo()

    def open(self, store):
        return store

    def exclusive_lock(self):
        return True


class FencedSqliteLogFactory(SqliteLogFactory):
    def __init__(self, persist):
        super().__init__(persist)
        self.epochs = {}
        self.fenced_appends = 0

    def open(self, log):
        handle = SqliteLog(
            log, epoch=self.epochs.setdefault(log, 0), persist=self.persist
        )
        append = handle.append

        def counted(records):
            try:
                return append(records)
            except EpochMismatch:
                self.fenced_appends += 1
                raise

        handle.append = counted
        return handle

    def fence(self, log):
        self.epochs[log] = self.epochs.setdefault(log, 0) + 1
        SqliteLog(log, persist=self.persist).fence(self.epochs[log])


def test_the_built_in_memory_backends_pass(persist, conformance):
    conformance.run_log_backend(MemoryLogFactory(persist))
    conformance.run_memo_backend(MemoryMemoFactory(persist))


def test_python_backends_pass(persist, conformance):
    conformance.run_log_backend(DictLogFactory(persist))
    conformance.run_memo_backend(DictMemoFactory(persist))


def test_a_backend_that_reorders_records_fails_naming_the_case(persist, conformance):
    class Reordering(DictLogFactory):
        def open(self, log):
            backend = DictBackend(log, persist=self.persist)
            read_from = backend.read_from

            def reversed_read(after):
                read = read_from(after)
                return read if isinstance(read, tuple) else read[::-1]

            backend.read_from = reversed_read
            return backend

    with pytest.raises(conformance.ConformanceFailure) as failure:
        conformance.run_log_backend(Reordering(persist))
    assert failure.value.case == "append_assigns_contiguous_increasing_positions"
    assert "append_assigns_contiguous_increasing_positions" in str(failure.value)


def test_a_backend_accepting_a_compaction_past_its_tail_fails_naming_the_case(
    persist, conformance
):
    class Overreaching(DictLogFactory):
        def open(self, log):
            backend = DictBackend(log, persist=self.persist)
            compact = backend.compact

            def lenient(upto, records):
                with log.lock:
                    upto = min(upto, log.tail)
                return compact(upto, records)

            backend.compact = lenient
            return backend

    with pytest.raises(conformance.ConformanceFailure) as failure:
        conformance.run_log_backend(Overreaching(persist))
    assert (
        failure.value.case == "compaction_rejects_positions_outside_snapshot_and_tail"
    )


def test_a_backend_whose_append_raises_a_callers_exception_passes_the_fenced_case(
    persist, conformance
):
    factory = FencedSqliteLogFactory(persist)
    conformance.run_log_backend(factory)
    assert factory.fenced_appends == 1


def test_a_fenced_handle_whose_append_lands_fails_the_fenced_case(persist, conformance):
    class Leaky(FencedSqliteLogFactory):
        def open(self, log):
            handle = SqliteLog(log, persist=self.persist)
            handle.epoch = None
            return handle

    with pytest.raises(conformance.ConformanceFailure) as failure:
        conformance.run_log_backend(Leaky(persist))
    assert failure.value.case == "fenced_append_lands_nothing"


def test_cases_lists_each_kind(persist, conformance):
    assert "fenced_append_lands_nothing" in conformance.cases("log")
    assert {
        "append_at_a_stale_position_writes_nothing",
        "racing_append_at_one_position_lands_exactly_one",
        "append_at_right_after_compacting_to_the_tail_lands",
    } <= set(conformance.cases("log"))
    assert conformance.cases("memo")
    with pytest.raises(ValueError):
        conformance.cases("elsewhere")


def test_tail_moved_is_a_persist_error(persist):
    assert issubclass(persist.TailMoved, persist.PersistError)


def test_memory_backend_append_at_lands_only_after_the_tail_it_names(persist):
    written = persist.MemoryBackend()
    store = persist.Store.open(written, build_id=1)
    store.set(store.input("x", 0, PickleCodec()), 1)
    run = [record for _, record in written.read_from(0)]

    backend = persist.MemoryBackend()
    with pytest.raises(persist.TailMoved):
        backend.append_at(1, run)
    assert backend.read_from(0) == []
    assert backend.append_at(0, run) == len(run)
    with pytest.raises(persist.TailMoved):
        backend.append_at(0, run)
    assert [record for _, record in backend.read_from(0)] == run


def test_python_backends_with_append_at_pass_its_cases(persist):
    for factory in (DictLogFactory(persist), SqliteLogFactory(persist)):
        for case in (
            "append_at_a_stale_position_writes_nothing",
            "racing_append_at_one_position_lands_exactly_one",
            "append_at_right_after_compacting_to_the_tail_lands",
        ):
            assert persist._conformance("store", factory, case) == {case: None}


def test_a_backend_whose_append_at_ignores_the_tail_fails_naming_the_case(
    persist, conformance
):
    class Unconditional(DictLogFactory):
        def open(self, log):
            backend = DictBackend(log, persist=self.persist)
            backend.append_at = lambda after, records: backend.append(records)
            return backend

    with pytest.raises(conformance.ConformanceFailure) as failure:
        conformance.run_log_backend(Unconditional(persist))
    assert failure.value.case == "append_at_a_stale_position_writes_nothing"


def test_a_backend_whose_append_at_raises_another_error_on_a_moved_tail_fails(
    persist, conformance
):
    class Mislabelled(DictLogFactory):
        def open(self, log):
            backend = DictBackend(log, persist=self.persist)
            conditional = backend.append_at

            def append_at(after, records):
                try:
                    return conditional(after, records)
                except self.persist.TailMoved:
                    raise ValueError("stale") from None

            backend.append_at = append_at
            return backend

    with pytest.raises(conformance.ConformanceFailure) as failure:
        conformance.run_log_backend(Mislabelled(persist))
    assert failure.value.case == "append_at_a_stale_position_writes_nothing"
