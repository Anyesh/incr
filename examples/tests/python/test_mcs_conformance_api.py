import pytest
from incr.persist import MemoryBackend, MemoryMemo, conformance
from test_mcs_persist_conformance import (
    DictBackend,
    DictLogFactory,
    DictMemoFactory,
    EpochMismatch,
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
    assert conformance.cases("memo")
    with pytest.raises(ValueError):
        conformance.cases("elsewhere")
