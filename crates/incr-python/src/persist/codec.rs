//! How a value becomes bytes in a persisted slot.
//!
//! A slot is a persisted key (`Key`), or the key or value of a memo table.
//! Each slot has one user codec: any object with `encode(obj) -> bytes` and
//! `decode(bytes) -> obj`. The hook incr-python installs reads the slot
//! incr-persist is encoding or decoding for from `current_value_context`
//! and calls that slot's codec.
//!
//! Two contracts bind a codec. `decode(encode(x)) == x`, or a restart does
//! not restore what was written. And `encode` is deterministic, giving equal
//! values equal bytes, because collection fingerprints and memo reuse hash
//! the bytes: a codec that orders a set or a dict arbitrarily makes a
//! restart look like a change.
//!
//! Every value is framed with a one-byte tag. Plain values (tag 0) are the
//! user codec's bytes. The not-ready values a user codec cannot know
//! (`Pending`, `LookupFailed`, `Unavailable`, `Incomplete`) encode their own
//! fields under their own tag, and each field that is not `None` goes
//! through the user codec, so a `row`, `key`, `reason` or `value` is
//! whatever the codec handles.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use incr_persist::{current_value_context, MemoPart, ValueContext};
use incr_python_protocol::notready::{
    self, PyIncomplete, PyLookupFailed, PyPending, PyUnavailable,
};
use incr_python_protocol::persist::{install, Hook};
use incr_python_protocol::PyValue;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};

use super::PersistError;

const PLAIN: u8 = 0;
const PENDING: u8 = 1;
const LOOKUP_FAILED: u8 = 2;
const UNAVAILABLE: u8 = 3;
const INCOMPLETE: u8 = 4;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Slot {
    Key(String),
    MemoKey(String),
    MemoValue(String),
}

impl Slot {
    fn of(context: ValueContext) -> Slot {
        match context {
            ValueContext::Key(key) => Slot::Key(key),
            ValueContext::MemoTable {
                table,
                part: MemoPart::Key,
            } => Slot::MemoKey(table),
            ValueContext::MemoTable {
                table,
                part: MemoPart::Value,
            } => Slot::MemoValue(table),
        }
    }

    fn describe(&self) -> String {
        match self {
            Slot::Key(key) => format!("persisted key '{key}'"),
            Slot::MemoKey(table) => format!("keys of memo table '{table}'"),
            Slot::MemoValue(table) => format!("values of memo table '{table}'"),
        }
    }
}

/// One store's codecs, and the exception a codec raised during the call in
/// progress, which incr-persist can only report as a bare serde error.
#[derive(Default)]
pub(crate) struct Registry {
    codecs: RefCell<HashMap<Slot, Py<PyAny>>>,
    failure: RefCell<Option<PyErr>>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Rc<Registry>>> = const { RefCell::new(None) };
}

struct Restore(Option<Rc<Registry>>);

impl Drop for Restore {
    fn drop(&mut self) {
        let previous = self.0.take();
        let _ = ACTIVE.try_with(|active| active.replace(previous));
    }
}

impl Registry {
    /// Registers `codec` for `slot`. Returns whether it is new: a slot keeps
    /// the codec it was declared with, and a different one is an error.
    pub(crate) fn register(&self, slot: Slot, codec: Py<PyAny>) -> PyResult<bool> {
        let mut codecs = self.codecs.borrow_mut();
        match codecs.get(&slot) {
            Some(held) if held.is(&codec) => Ok(false),
            Some(_) => Err(PyValueError::new_err(format!(
                "{} is already declared with another codec",
                slot.describe()
            ))),
            None => {
                codecs.insert(slot, codec);
                Ok(true)
            }
        }
    }

    pub(crate) fn unregister(&self, slot: &Slot) {
        self.codecs.borrow_mut().remove(slot);
    }

    pub(crate) fn is_registered(&self, slot: &Slot) -> bool {
        self.codecs.borrow().contains_key(slot)
    }

    /// Runs a store call with this registry's codecs active. A failure is
    /// raised as the exception a codec raised during it, else through
    /// `describe`.
    pub(crate) fn run<R>(
        self: &Rc<Self>,
        py: Python<'_>,
        f: impl FnOnce() -> incr_persist::Result<R>,
    ) -> PyResult<R> {
        self.failure.borrow_mut().take();
        let result = {
            let _restore = Restore(ACTIVE.with(|active| active.replace(Some(Rc::clone(self)))));
            f()
        };
        let failure = self.failure.borrow_mut().take();
        match result {
            Ok(value) => Ok(value),
            Err(error) => Err(failure.unwrap_or_else(|| super::persist_error(py, error))),
        }
    }

    fn stash(&self, err: PyErr) -> String {
        let message = Python::attach(|py| err.value(py).to_string());
        self.failure.borrow_mut().get_or_insert(err);
        message
    }

    fn codec_for(&self, py: Python<'_>) -> Result<Py<PyAny>, String> {
        let context = current_value_context().ok_or_else(|| {
            self.stash(PersistError::new_err(
                "a value was encoded outside any persisted slot",
            ))
        })?;
        let slot = Slot::of(context);
        match self.codecs.borrow().get(&slot) {
            Some(codec) => Ok(codec.clone_ref(py)),
            None => Err(self.stash(PersistError::new_err(format!(
                "no codec is registered for {}",
                slot.describe()
            )))),
        }
    }
}

fn active() -> Result<Rc<Registry>, String> {
    ACTIVE
        .with(|active| active.borrow().clone())
        .ok_or_else(|| "a persisted value was used outside a store call".to_owned())
}

pub(crate) fn install_hook() {
    let _ = install(Hook { encode, decode });
}

fn encode(py: Python<'_>, value: &PyValue) -> Result<Vec<u8>, String> {
    let registry = active()?;
    let codec = registry.codec_for(py)?;
    frame(py, &codec, value.0.bind(py)).map_err(|e| registry.stash(e))
}

fn decode(py: Python<'_>, bytes: &[u8]) -> Result<PyValue, String> {
    let registry = active()?;
    let codec = registry.codec_for(py)?;
    unframe(py, &codec, bytes)
        .map(PyValue)
        .map_err(|e| registry.stash(e))
}

fn user_encode(py: Python<'_>, codec: &Py<PyAny>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let encoded = codec.bind(py).call_method1(intern!(py, "encode"), (obj,))?;
    if let Ok(bytes) = encoded.downcast::<PyBytes>() {
        return Ok(bytes.as_bytes().to_vec());
    }
    if let Ok(bytes) = encoded.downcast::<PyByteArray>() {
        return Ok(bytes.to_vec());
    }
    Err(PyTypeError::new_err(format!(
        "codec.encode must return bytes, not {}",
        encoded.get_type().name()?
    )))
}

fn user_decode(py: Python<'_>, codec: &Py<PyAny>, bytes: &[u8]) -> PyResult<Py<PyAny>> {
    Ok(codec
        .bind(py)
        .call_method1(intern!(py, "decode"), (PyBytes::new(py, bytes),))?
        .unbind())
}

fn push_field(out: &mut Vec<u8>, bytes: &[u8]) -> PyResult<()> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| PyValueError::new_err("an encoded field exceeds 4 GiB"))?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn push_value(
    py: Python<'_>,
    codec: &Py<PyAny>,
    out: &mut Vec<u8>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if value.is_none() {
        out.push(0);
        return Ok(());
    }
    out.push(1);
    push_field(out, &user_encode(py, codec, value)?)
}

fn field(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyAny>> {
    Ok(obj.getattr(name)?.unbind())
}

fn frame(py: Python<'_>, codec: &Py<PyAny>, value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let mut out = Vec::new();
    match notready::status(value) {
        notready::Status::Ready => {
            out.push(PLAIN);
            out.extend_from_slice(&user_encode(py, codec, value)?);
        }
        notready::Status::Incomplete(incomplete) => {
            let incomplete = incomplete.get();
            out.push(INCOMPLETE);
            out.extend_from_slice(&incomplete.missing().to_le_bytes());
            push_value(py, codec, &mut out, incomplete.value().bind(py))?;
        }
        notready::Status::NotReady => {
            if let Ok(pending) = value.downcast_exact::<PyPending>() {
                out.push(PENDING);
                push_value(py, codec, &mut out, field(pending, "key")?.bind(py))?;
            } else if let Ok(failed) = value.downcast_exact::<PyLookupFailed>() {
                out.push(LOOKUP_FAILED);
                push_value(py, codec, &mut out, field(failed, "reason")?.bind(py))?;
                push_value(py, codec, &mut out, field(failed, "key")?.bind(py))?;
            } else if let Ok(unavailable) = value.downcast_exact::<PyUnavailable>() {
                let unavailable = unavailable.get();
                out.push(UNAVAILABLE);
                push_value(py, codec, &mut out, unavailable.reason().bind(py))?;
                push_value(py, codec, &mut out, unavailable.row().bind(py))?;
            } else {
                return Err(PyTypeError::new_err(format!(
                    "cannot persist a {} value: only Pending, LookupFailed and Unavailable are persistable not-ready values",
                    value.get_type().name()?
                )));
            }
        }
    }
    Ok(out)
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> PyResult<&'a [u8]> {
        if self.0.len() < n {
            return Err(PyValueError::new_err("a persisted value is truncated"));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn value(&mut self, py: Python<'_>, codec: &Py<PyAny>) -> PyResult<Py<PyAny>> {
        match self.take(1)?[0] {
            0 => Ok(py.None()),
            1 => {
                let len = u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"));
                let bytes = self.take(len as usize)?;
                user_decode(py, codec, bytes)
            }
            other => Err(PyValueError::new_err(format!(
                "a persisted field has unknown presence byte {other}"
            ))),
        }
    }

    fn finish(self) -> PyResult<()> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "a persisted value has trailing bytes",
            ))
        }
    }
}

fn unframe(py: Python<'_>, codec: &Py<PyAny>, bytes: &[u8]) -> PyResult<Py<PyAny>> {
    let (&tag, rest) = bytes
        .split_first()
        .ok_or_else(|| PyValueError::new_err("a persisted value is empty"))?;
    if tag == PLAIN {
        return user_decode(py, codec, rest);
    }
    let mut reader = Reader(rest);
    let value = match tag {
        PENDING => notready::pending(py, reader.value(py, codec)?)?,
        LOOKUP_FAILED => {
            let reason = reader.value(py, codec)?;
            notready::lookup_failed(py, reason, reader.value(py, codec)?)?
        }
        UNAVAILABLE => {
            let reason = reader.value(py, codec)?;
            notready::unavailable(py, reason, reader.value(py, codec)?)?
        }
        INCOMPLETE => {
            let missing = u64::from_le_bytes(reader.take(8)?.try_into().expect("eight bytes"));
            let inner = reader.value(py, codec)?;
            py.get_type::<PyIncomplete>()
                .call1((inner, missing))?
                .unbind()
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "a persisted value has unknown tag {other}"
            )))
        }
    };
    reader.finish()?;
    Ok(value)
}
