//! `incr.persist._conformance`: runs incr-persist's backend conformance
//! cases against Python backend objects. Compiled only with the
//! `conformance` feature.
//!
//! A factory object stands in for the Rust harness. Store cases call
//! `new_log()`, `open(log)`, `open_read_only(log)` (an object or `None`)
//! and `multi_writer()`; memo cases call `new_store()`, `open(store)` and
//! `exclusive_lock()`.

use std::collections::BTreeMap;
use std::panic::{self, AssertUnwindSafe};

use incr_persist::conformance::{self as store_cases, Harness};
use incr_persist::memo::conformance::{self as memo_cases, MemoHarness};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::backend::PyBackend;
use super::memo_backend::PyMemoBackend;

struct StoreFactory(Py<PyAny>);

impl StoreFactory {
    fn call(&self, method: &str, arg: Option<&Py<PyAny>>) -> Py<PyAny> {
        Python::attach(|py| {
            let factory = self.0.bind(py);
            let result = match arg {
                Some(arg) => factory.call_method1(method, (arg.bind(py),)),
                None => factory.call_method0(method),
            };
            result
                .unwrap_or_else(|e| panic!("factory.{method} raised: {e}"))
                .unbind()
        })
    }
}

impl Harness for StoreFactory {
    type Backend = PyBackend;
    type Log = Py<PyAny>;

    fn new_log(&mut self) -> Self::Log {
        self.call("new_log", None)
    }

    fn open(&mut self, log: &Self::Log) -> PyBackend {
        PyBackend::new(self.call("open", Some(log)))
    }

    fn open_read_only(&mut self, log: &Self::Log) -> Option<PyBackend> {
        let handle = self.call("open_read_only", Some(log));
        Python::attach(|py| (!handle.is_none(py)).then(|| PyBackend::new(handle)))
    }

    fn multi_writer(&self) -> bool {
        Python::attach(|py| {
            self.0
                .bind(py)
                .call_method0("multi_writer")
                .and_then(|v| v.is_truthy())
                .unwrap_or_else(|e| panic!("factory.multi_writer raised: {e}"))
        })
    }
}

struct MemoFactory(StoreFactory);

impl MemoHarness for MemoFactory {
    type Backend = PyMemoBackend;
    type Store = Py<PyAny>;

    fn new_store(&mut self) -> Self::Store {
        self.0.call("new_store", None)
    }

    fn open(&mut self, store: &Self::Store) -> PyMemoBackend {
        PyMemoBackend::new(self.0.call("open", Some(store)))
    }

    fn exclusive_lock(&self) -> bool {
        Python::attach(|py| {
            self.0
                 .0
                .bind(py)
                .call_method0("exclusive_lock")
                .and_then(|v| v.is_truthy())
                .unwrap_or_else(|e| panic!("factory.exclusive_lock raised: {e}"))
        })
    }
}

macro_rules! case_table {
    ($module:ident, $harness:ident, $($case:ident),+ $(,)?) => {
        &[$((stringify!($case), |factory| $module::$case($harness(factory)))),+]
    };
}

type Case = (&'static str, fn(Py<PyAny>));

const STORE_CASES: &[Case] = case_table!(
    store_cases,
    StoreFactory,
    append_assigns_contiguous_increasing_positions,
    empty_append_returns_the_tail,
    read_from_returns_only_later_records,
    synced_records_survive_reopen,
    records_roundtrip_unchanged,
    writers_share_one_total_order,
    readers_never_skip_records_under_concurrent_appends,
    compaction_keeps_later_records_and_records_its_position,
    compacted_when_behind_snapshot_with_no_log_records,
    positions_are_never_reused_after_compaction,
    compaction_survives_reopen,
    compaction_rejects_positions_outside_snapshot_and_tail,
    forgotten_keys_leave_snapshot_and_log,
    read_only_handle_rejects_writes_and_follows_writer,
    snapshot_id_names_each_compaction,
    snapshot_id_survives_reopen,
    read_only_handle_sees_the_snapshot_id,
);

fn memo_harness(factory: Py<PyAny>) -> MemoFactory {
    MemoFactory(StoreFactory(factory))
}

const MEMO_CASES: &[Case] = case_table!(
    memo_cases,
    memo_harness,
    fresh_store_loads_nothing,
    written_generation_loads_and_reads_back,
    empty_generation_roundtrips,
    generation_survives_reopen,
    write_from_a_stale_base_is_busy,
    one_of_two_racing_writers_publishes,
    older_generation_stays_readable_until_removed,
    checkpoint_lock_excludes_other_handles,
    checkpoint_lock_removes_generations_below_current,
);

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_else(|| "panic with a non-string payload".to_owned())
}

/// Runs the cases of `kind` (`"store"` or `"memo"`), or only `case`, with
/// the GIL released so the cases' own threads can attach. Returns each
/// case's name mapped to `None` on a pass or the panic message.
#[pyfunction]
#[pyo3(signature = (kind, factory, case=None))]
fn _conformance(
    py: Python<'_>,
    kind: &str,
    factory: Py<PyAny>,
    case: Option<&str>,
) -> PyResult<BTreeMap<&'static str, Option<String>>> {
    let table = match kind {
        "store" => STORE_CASES,
        "memo" => MEMO_CASES,
        _ => return Err(PyValueError::new_err("kind must be 'store' or 'memo'")),
    };
    let selected: Vec<&Case> = table
        .iter()
        .filter(|(name, _)| case.is_none_or(|wanted| wanted == *name))
        .collect();
    if selected.is_empty() {
        return Err(PyValueError::new_err("no such conformance case"));
    }
    Ok(py.detach(|| {
        selected
            .into_iter()
            .map(|(name, run)| {
                let handle = Python::attach(|py| factory.clone_ref(py));
                let outcome = panic::catch_unwind(AssertUnwindSafe(|| run(handle)));
                (*name, outcome.err().map(panic_message))
            })
            .collect()
    }))
}

#[pyfunction]
fn _conformance_cases(kind: &str) -> PyResult<Vec<&'static str>> {
    match kind {
        "store" => Ok(STORE_CASES.iter().map(|(name, _)| *name).collect()),
        "memo" => Ok(MEMO_CASES.iter().map(|(name, _)| *name).collect()),
        _ => Err(PyValueError::new_err("kind must be 'store' or 'memo'")),
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_conformance, module)?)?;
    module.add_function(wrap_pyfunction!(_conformance_cases, module)?)
}
