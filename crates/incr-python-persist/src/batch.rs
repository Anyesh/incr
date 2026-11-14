//! A store batch: the writes of a `with` block, appended as one run when
//! the block ends without an exception.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::binding::PyBinding;
use crate::keyed::Pending;

#[derive(PartialEq)]
enum Phase {
    New,
    Open,
    Done,
}

/// A batch's phase and the writes staged in it.
pub struct Batch<B: PyBinding> {
    phase: Phase,
    pending: Pending<B>,
}

impl<B: PyBinding> Default for Batch<B> {
    fn default() -> Self {
        Batch {
            phase: Phase::New,
            pending: Pending::default(),
        }
    }
}

impl<B: PyBinding> Batch<B> {
    /// Opens the batch; `RuntimeError` if it was entered before.
    pub fn enter(&mut self) -> PyResult<()> {
        if self.phase != Phase::New {
            return Err(PyRuntimeError::new_err("a batch can be entered once"));
        }
        self.phase = Phase::Open;
        Ok(())
    }

    /// Ends the open batch and returns its writes, which the caller appends
    /// unless the block raised.
    pub fn exit(&mut self) -> PyResult<Pending<B>> {
        self.pending()?;
        self.phase = Phase::Done;
        Ok(std::mem::take(&mut self.pending))
    }

    /// The staged writes of the open batch; `RuntimeError` outside the
    /// `with` block.
    pub fn pending(&mut self) -> PyResult<&mut Pending<B>> {
        if self.phase == Phase::Open {
            Ok(&mut self.pending)
        } else {
            Err(PyRuntimeError::new_err(
                "the batch is not open: use it as `with store.batch() as b`",
            ))
        }
    }
}
