import incr.persist as persist
import pytest
from incr.persist import (
    MemoryBackend,
    MemoryMemo,
    PersistError,
    Store,
    TailMoved,
    conformance,
)
from test_mcs_persist_conformance import (
    DictBackend,
    DictLogFactory,
    DictMemoFactory,
    EpochMismatch,
    PickleCodec,
    SqliteLog,
    SqliteLogFactory,
)


class MemoryLogFactory:
    def new_log(self):
        return MemoryBackend()

    def open(self, log):
        return log

    def open_read_only(self, log):
        return log.read_only()

    def multi_writer(self):
        return True


class MemoryMemoFactory:
    def new_store(self):
        return MemoryMemo()

    def open(self, store):
        return store

    def exclusive_lock(self):
        return True


class FencedSqliteLogFactory(SqliteLogFactory):
    def __init__(self):
        super().__init__()
        self.epochs = {}
        self.fenced_appends = 0

    def open(self, log):
        handle = SqliteLog(log, epoch=self.epochs.setdefault(log, 0))
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
        SqliteLog(log).fence(self.epochs[log])


def test_the_built_in_memory_backends_pass():
    conformance.run_log_backend(MemoryLogFactory())
    conformance.run_memo_backend(MemoryMemoFactory())


def test_python_backends_pass():
    conformance.run_log_backend(DictLogFactory())
    conformance.run_memo_backend(DictMemoFactory())


def test_a_backend_that_reorders_records_fails_naming_the_case():
    class Reordering(DictLogFactory):
        def open(self, log):
            backend = DictBackend(log)
            read_from = backend.read_from

            def reversed_read(after):
                read = read_from(after)
                return read if isinstance(read, tuple) else read[::-1]

            backend.read_from = reversed_read
            return backend

    with pytest.raises(conformance.ConformanceFailure) as failure:
        conformance.run_log_backend(Reordering())
    assert failure.value.case == "append_assigns_contiguous_increasing_positions"
    assert "append_assigns_contiguous_increasing_positions" in str(failure.value)


def test_a_backend_accepting_a_compaction_past_its_tail_fails_naming_the_case():
    class Overreaching(DictLogFactory):
        def open(self, log):
            backend = DictBackend(log)
            compact = backend.compact

            def lenient(upto, records):
                with log.lock:
                    upto = min(upto, log.tail)
                return compact(upto, records)

            backend.compact = lenient
            return backend

    with pytest.raises(conformance.ConformanceFailure) as failure:
        conformance.run_log_backend(Overreaching())
    assert (
        failure.value.case == "compaction_rejects_positions_outside_snapshot_and_tail"
    )


def test_a_backend_whose_append_raises_a_callers_exception_passes_the_fenced_case():
    factory = FencedSqliteLogFactory()
    conformance.run_log_backend(factory)
    assert factory.fenced_appends == 1


def test_a_fenced_handle_whose_append_lands_fails_the_fenced_case():
    class Leaky(FencedSqliteLogFactory):
        def open(self, log):
            handle = SqliteLog(log)
            handle.epoch = None
            return handle

    with pytest.raises(conformance.ConformanceFailure) as failure:
        conformance.run_log_backend(Leaky())
    assert failure.value.case == "fenced_append_lands_nothing"


def test_cases_lists_each_kind():
    assert "fenced_append_lands_nothing" in conformance.cases("log")
    assert {
        "append_at_a_stale_position_writes_nothing",
        "racing_append_at_one_position_lands_exactly_one",
        "append_at_right_after_compacting_to_the_tail_lands",
    } <= set(conformance.cases("log"))
    assert conformance.cases("memo")
    with pytest.raises(ValueError):
        conformance.cases("elsewhere")


def test_tail_moved_is_a_persist_error():
    assert issubclass(TailMoved, PersistError)


def test_memory_backend_append_at_lands_only_after_the_tail_it_names():
    written = MemoryBackend()
    store = Store.open(written, build_id=1)
    store.set(store.input("x", 0, PickleCodec()), 1)
    run = [record for _, record in written.read_from(0)]

    backend = MemoryBackend()
    with pytest.raises(TailMoved):
        backend.append_at(1, run)
    assert backend.read_from(0) == []
    assert backend.append_at(0, run) == len(run)
    with pytest.raises(TailMoved):
        backend.append_at(0, run)
    assert [record for _, record in backend.read_from(0)] == run


def test_python_backends_with_append_at_pass_its_cases():
    for factory in (DictLogFactory(), SqliteLogFactory()):
        for case in (
            "append_at_a_stale_position_writes_nothing",
            "racing_append_at_one_position_lands_exactly_one",
            "append_at_right_after_compacting_to_the_tail_lands",
        ):
            assert persist._conformance("store", factory, case) == {case: None}


def test_a_backend_whose_append_at_ignores_the_tail_fails_naming_the_case():
    class Unconditional(DictLogFactory):
        def open(self, log):
            backend = DictBackend(log)
            backend.append_at = lambda after, records: backend.append(records)
            return backend

    with pytest.raises(conformance.ConformanceFailure) as failure:
        conformance.run_log_backend(Unconditional())
    assert failure.value.case == "append_at_a_stale_position_writes_nothing"


def test_a_backend_whose_append_at_raises_another_error_on_a_moved_tail_fails():
    class Mislabelled(DictLogFactory):
        def open(self, log):
            backend = DictBackend(log)
            conditional = backend.append_at

            def append_at(after, records):
                try:
                    return conditional(after, records)
                except TailMoved:
                    raise ValueError("stale") from None

            backend.append_at = append_at
            return backend

    with pytest.raises(conformance.ConformanceFailure) as failure:
        conformance.run_log_backend(Mislabelled())
    assert failure.value.case == "append_at_a_stale_position_writes_nothing"
