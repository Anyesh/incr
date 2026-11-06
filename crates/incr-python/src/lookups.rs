//! Lookups: one input node per key, created as `Pending(key)` and queued
//! for the caller to fetch, then resolved, failed or retried. The nodes sit
//! in a table no creator owns, so a query that stops reading a lookup never
//! deletes it, and a later read reuses the value.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::mem::size_of;
use std::rc::Rc;
use std::sync::Arc;

use incr_compute::{Held, Incr, NodeTable, Runtime, StateKind, StateMemory, StateStats};
use incr_python_protocol::notready::{pending, PyPending};
use incr_python_protocol::{or_stash, PyValue};
use pyo3::prelude::*;

type Table = NodeTable<PyValue, PyValue>;
type Queued = (PyValue, Incr<PyValue>);

/// A runtime's lookup table, created on first use, and the keys created or
/// retried since the last drain.
#[derive(Default)]
pub(crate) struct Lookups {
    table: RefCell<Option<Rc<Table>>>,
    queue: RefCell<Vec<Queued>>,
    keys: Cell<usize>,
}

// SAFETY: the state registry needs `Send + Sync`, but it lives in a
// `Runtime<Local>`, which is neither, and the lookups otherwise only in the
// binding state of that runtime's `unsendable` classes and closures, so they
// are only ever reached from the one thread that runtime lives on.
unsafe impl Send for Lookups {}
// SAFETY: see `Send`.
unsafe impl Sync for Lookups {}

impl StateStats for Lookups {
    fn state_stats(&self) -> StateMemory {
        let keys = self.keys.get();
        let (queued, queue_bytes) = self
            .queue
            .try_borrow()
            .map_or((0, 0), |q| (q.len(), q.capacity() * size_of::<Queued>()));
        let table_bytes = keys * size_of::<Queued>();
        StateMemory::new(
            StateKind::Other("lookups"),
            keys,
            vec![Held::of::<PyValue>(keys + queued)],
            table_bytes + queue_bytes,
        )
    }
}

impl Lookups {
    /// The table, created and registered for `memory_stats` on first use.
    fn table(self: &Arc<Self>, rt: &Runtime) -> Rc<Table> {
        if let Some(table) = &*self.table.borrow() {
            return Rc::clone(table);
        }
        let table = Rc::new(rt.create_table::<PyValue, PyValue>());
        *self.table.borrow_mut() = Some(Rc::clone(&table));
        let weak: std::sync::Weak<Lookups> = Arc::downgrade(self);
        rt.register_state(None, weak);
        table
    }

    /// `key`'s node and whether this call created it, with `init()` as its
    /// value; a created node is queued when `queue` is set.
    fn node_with(
        self: &Arc<Self>,
        rt: &Runtime,
        py: Python<'_>,
        key: &Py<PyAny>,
        queue: bool,
        init: impl FnOnce() -> Py<PyAny>,
    ) -> (Incr<PyValue>, bool) {
        let created = Cell::new(false);
        let node = self
            .table(rt)
            .get_or_insert_input(rt, PyValue(key.clone_ref(py)), || {
                created.set(true);
                PyValue(init())
            });
        if created.get() {
            self.keys.set(self.keys.get() + 1);
            if queue {
                self.queue
                    .borrow_mut()
                    .push((PyValue(key.clone_ref(py)), node));
            }
        }
        (node, created.get())
    }

    /// `key`'s node, created as `Pending(key)` and queued on first use.
    pub(crate) fn node(
        self: &Arc<Self>,
        rt: &Runtime,
        py: Python<'_>,
        key: &Py<PyAny>,
    ) -> Incr<PyValue> {
        self.node_with(rt, py, key, true, || pending_value(py, key))
            .0
    }

    /// Sets `key`'s node to `value`, creating it with that value if absent.
    pub(crate) fn set(
        self: &Arc<Self>,
        rt: &Runtime,
        py: Python<'_>,
        key: &Py<PyAny>,
        value: Py<PyAny>,
    ) {
        let mut value = Some(value);
        let (node, _) = self.node_with(rt, py, key, false, || {
            value.take().expect("init runs at most once")
        });
        if let Some(value) = value {
            rt.set(node, PyValue(value));
        }
    }

    /// Sets `key`'s node back to `Pending(key)` and queues it.
    pub(crate) fn retry(self: &Arc<Self>, rt: &Runtime, py: Python<'_>, key: &Py<PyAny>) {
        let (node, created) = self.node_with(rt, py, key, true, || pending_value(py, key));
        if !created {
            rt.set(node, PyValue(pending_value(py, key)));
            self.queue
                .borrow_mut()
                .push((PyValue(key.clone_ref(py)), node));
        }
    }

    /// The queued keys whose node is still alive and `Pending`, each once,
    /// in the order they were queued. Empties the queue.
    pub(crate) fn drain(&self, rt: &Runtime, py: Python<'_>) -> Vec<PyValue> {
        let queued = std::mem::take(&mut *self.queue.borrow_mut());
        let mut seen = HashSet::new();
        let mut keys = Vec::new();
        for (key, node) in queued {
            if !rt.is_alive(node) || !seen.insert(node.slot()) {
                continue;
            }
            if rt.get(node).0.bind(py).is_exact_instance_of::<PyPending>() {
                keys.push(key);
            }
        }
        keys
    }

    /// Drops the table and the queue, for a runtime being closed.
    pub(crate) fn clear(&self) -> (Option<Rc<Table>>, Vec<Queued>) {
        self.keys.set(0);
        (
            self.table.borrow_mut().take(),
            std::mem::take(&mut *self.queue.borrow_mut()),
        )
    }
}

fn pending_value(py: Python<'_>, key: &Py<PyAny>) -> Py<PyAny> {
    or_stash(pending(py, key.clone_ref(py)))
}
