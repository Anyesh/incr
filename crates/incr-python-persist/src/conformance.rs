//! Runs incr-persist's backend conformance cases against Python backend
//! objects, for each package's `persist.conformance` module, which
//! [`conformance_functions!`](crate::conformance_functions) defines.
//! Compiled only with the `conformance` feature, which the wheels enable.
//!
//! A factory object stands in for the Rust harness. Log cases call
//! `new_log()`, `open(log)`, `open_read_only(log)` (an object or `None`),
//! `multi_writer()` and, if the factory has it, `fence(log)`; memo cases
//! call `new_store()`, `open(store)` and `exclusive_lock()`.

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::panic::{self, AssertUnwindSafe};

use incr_persist::conformance::{self as store_cases, Harness};
use incr_persist::memo::conformance::{self as memo_cases, MemoHarness};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::backend::PyBackend;
use crate::memo_backend::PyMemoBackend;
use crate::PersistExceptions;

struct StoreFactory<E>(Py<PyAny>, PhantomData<fn() -> E>);

impl<E> StoreFactory<E> {
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

impl<E: PersistExceptions> Harness for StoreFactory<E> {
    type Backend = PyBackend<E>;
    type Log = Py<PyAny>;

    fn new_log(&mut self) -> Self::Log {
        self.call("new_log", None)
    }

    fn open(&mut self, log: &Self::Log) -> PyBackend<E> {
        wrap(self.call("open", Some(log)))
    }

    fn open_read_only(&mut self, log: &Self::Log) -> Option<PyBackend<E>> {
        let handle = self.call("open_read_only", Some(log));
        let none = Python::attach(|py| handle.is_none(py));
        (!none).then(|| wrap(handle))
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

    fn fence(&mut self, log: &Self::Log) -> bool {
        let has_fence = Python::attach(|py| {
            self.0
                .bind(py)
                .hasattr("fence")
                .unwrap_or_else(|e| panic!("factory lookup of fence raised: {e}"))
        });
        if has_fence {
            self.call("fence", Some(log));
        }
        has_fence
    }
}

fn wrap<E: PersistExceptions>(handle: Py<PyAny>) -> PyBackend<E> {
    Python::attach(|py| {
        PyBackend::new(py, handle)
            .unwrap_or_else(|e| panic!("looking up the handle's append_at raised: {e}"))
    })
}

struct MemoFactory<E>(StoreFactory<E>);

impl<E: PersistExceptions> MemoHarness for MemoFactory<E> {
    type Backend = PyMemoBackend<E>;
    type Store = Py<PyAny>;

    fn new_store(&mut self) -> Self::Store {
        self.0.call("new_store", None)
    }

    fn open(&mut self, store: &Self::Store) -> PyMemoBackend<E> {
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
        vec![$((stringify!($case), |factory| $module::$case($harness::<E>(factory)))),+]
    };
}

/// A conformance case: its name and the function running it on a factory.
pub type Case = (&'static str, fn(Py<PyAny>));

fn store_harness<E: PersistExceptions>(factory: Py<PyAny>) -> StoreFactory<E> {
    StoreFactory(factory, PhantomData)
}

fn store_table<E: PersistExceptions>() -> Vec<Case> {
    case_table!(
        store_cases,
        store_harness,
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
        fenced_append_lands_nothing,
        append_at_a_stale_position_writes_nothing,
        racing_append_at_one_position_lands_exactly_one,
        append_at_right_after_compacting_to_the_tail_lands,
    )
}

fn memo_harness<E: PersistExceptions>(factory: Py<PyAny>) -> MemoFactory<E> {
    MemoFactory(store_harness(factory))
}

fn memo_table<E: PersistExceptions>() -> Vec<Case> {
    case_table!(
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
    )
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_else(|| "panic with a non-string payload".to_owned())
}

/// The cases of `kind` (`"log"`, its alias `"store"`, or `"memo"`), in run
/// order.
pub fn table<E: PersistExceptions>(kind: &str) -> PyResult<Vec<Case>> {
    match kind {
        "store" | "log" => Ok(store_table::<E>()),
        "memo" => Ok(memo_table::<E>()),
        _ => Err(PyValueError::new_err("kind must be 'log' or 'memo'")),
    }
}

/// Runs each case on a fresh handle to `factory` with the GIL released, so
/// the cases' own threads can attach, until one fails if `stop` is set.
/// Returns each case run mapped to `None` on a pass or the panic message.
fn run(
    py: Python<'_>,
    cases: Vec<Case>,
    factory: &Py<PyAny>,
    stop: bool,
) -> Vec<(&'static str, Option<String>)> {
    py.detach(|| {
        let mut outcomes = Vec::with_capacity(cases.len());
        for (name, case) in cases {
            let handle = Python::attach(|py| factory.clone_ref(py));
            let outcome = panic::catch_unwind(AssertUnwindSafe(|| case(handle)));
            let failed = outcome.is_err();
            outcomes.push((name, outcome.err().map(panic_message)));
            if stop && failed {
                break;
            }
        }
        outcomes
    })
}

/// Runs the cases of `kind`, or only `case`. Returns each case's name
/// mapped to `None` on a pass or the panic message.
pub fn run_selected<E: PersistExceptions>(
    py: Python<'_>,
    kind: &str,
    factory: &Py<PyAny>,
    case: Option<&str>,
) -> PyResult<BTreeMap<&'static str, Option<String>>> {
    let selected: Vec<Case> = table::<E>(kind)?
        .into_iter()
        .filter(|(name, _)| case.is_none_or(|wanted| wanted == *name))
        .collect();
    if selected.is_empty() {
        return Err(PyValueError::new_err("no such conformance case"));
    }
    Ok(run(py, selected, factory, false).into_iter().collect())
}

/// The names of the `kind` cases, in run order.
pub fn names<E: PersistExceptions>(kind: &str) -> PyResult<Vec<&'static str>> {
    Ok(table::<E>(kind)?.iter().map(|(name, _)| *name).collect())
}

/// Runs every `kind` case until one fails. Returns the first failing case
/// and its panic message, if any.
pub fn first_failure<E: PersistExceptions>(
    py: Python<'_>,
    kind: &str,
    factory: &Py<PyAny>,
) -> PyResult<Option<(&'static str, String)>> {
    let cases = table::<E>(kind)?;
    Ok(run(py, cases, factory, true)
        .into_iter()
        .find_map(|(name, failure)| failure.map(|message| (name, message))))
}

/// Defines a package's conformance entry points for the exceptions `$exc`:
/// the `ConformanceFailure` exception under `$module`, and
/// `register_conformance(persist)`, which adds `_conformance` and
/// `_conformance_cases` to the `persist` module and creates its
/// `conformance` submodule, registered in `sys.modules` as `$name`.
#[macro_export]
macro_rules! conformance_functions {
    ($exc:ty, $module:expr, $name:literal) => {
        $crate::pyo3::create_exception!(
            $module,
            ConformanceFailure,
            $crate::pyo3::exceptions::PyAssertionError,
            "A backend failed a conformance case; `case` names it."
        );

        /// Runs the cases of `kind` (`"store"` or `"memo"`), or only `case`.
        /// Returns each case's name mapped to `None` on a pass or the panic
        /// message.
        #[$crate::pyo3::pyfunction]
        #[pyo3(signature = (kind, factory, case=None))]
        fn _conformance(
            py: $crate::pyo3::Python<'_>,
            kind: &str,
            factory: $crate::pyo3::Py<$crate::pyo3::PyAny>,
            case: Option<&str>,
        ) -> $crate::pyo3::PyResult<::std::collections::BTreeMap<&'static str, Option<String>>> {
            $crate::conformance::run_selected::<$exc>(py, kind, &factory, case)
        }

        #[$crate::pyo3::pyfunction]
        fn _conformance_cases(kind: &str) -> $crate::pyo3::PyResult<Vec<&'static str>> {
            $crate::conformance::names::<$exc>(kind)
        }

        /// The names of the `kind` (`"log"` or `"memo"`) cases, in run order.
        #[$crate::pyo3::pyfunction]
        fn cases(kind: &str) -> $crate::pyo3::PyResult<Vec<&'static str>> {
            $crate::conformance::names::<$exc>(kind)
        }

        fn run_all(
            py: $crate::pyo3::Python<'_>,
            kind: &str,
            factory: $crate::pyo3::Py<$crate::pyo3::PyAny>,
        ) -> $crate::pyo3::PyResult<()> {
            use $crate::pyo3::types::PyAnyMethods;
            match $crate::conformance::first_failure::<$exc>(py, kind, &factory)? {
                None => Ok(()),
                Some((name, message)) => {
                    let error = ConformanceFailure::new_err(format!(
                        "conformance case {name} failed: {message}"
                    ));
                    error.value(py).setattr("case", name)?;
                    Err(error)
                }
            }
        }

        /// Runs every log backend case against handles from `factory`, raising
        /// `ConformanceFailure` naming the first case that fails.
        #[$crate::pyo3::pyfunction]
        fn run_log_backend(
            py: $crate::pyo3::Python<'_>,
            factory: $crate::pyo3::Py<$crate::pyo3::PyAny>,
        ) -> $crate::pyo3::PyResult<()> {
            run_all(py, "log", factory)
        }

        /// Runs every memo backend case against handles from `factory`, raising
        /// `ConformanceFailure` naming the first case that fails.
        #[$crate::pyo3::pyfunction]
        fn run_memo_backend(
            py: $crate::pyo3::Python<'_>,
            factory: $crate::pyo3::Py<$crate::pyo3::PyAny>,
        ) -> $crate::pyo3::PyResult<()> {
            run_all(py, "memo", factory)
        }

        pub(crate) fn register_conformance(
            module: &$crate::pyo3::Bound<'_, $crate::pyo3::types::PyModule>,
        ) -> $crate::pyo3::PyResult<()> {
            use $crate::pyo3::prelude::*;
            use $crate::pyo3::wrap_pyfunction;
            let py = module.py();
            module.add_function(wrap_pyfunction!(_conformance, module)?)?;
            module.add_function(wrap_pyfunction!(_conformance_cases, module)?)?;
            let public = $crate::pyo3::types::PyModule::new(py, "conformance")?;
            public.setattr("__name__", $name)?;
            public.add("ConformanceFailure", py.get_type::<ConformanceFailure>())?;
            public.add_function(wrap_pyfunction!(run_log_backend, &public)?)?;
            public.add_function(wrap_pyfunction!(run_memo_backend, &public)?)?;
            public.add_function(wrap_pyfunction!(cases, &public)?)?;
            module.add("conformance", &public)?;
            py.import("sys")?
                .getattr("modules")?
                .set_item($name, &public)?;
            Ok(())
        }
    };
}
