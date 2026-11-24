use std::collections::VecDeque;

use crate::memo::VALUE_BATCH;
use crate::value_context::hold;
use crate::ValueScope;

/// Captured values on their way to a checkpoint worker or on it. Every value
/// leaves a parcel inside its scope's `hold`, [`VALUE_BATCH`] at a time,
/// whether the flight code takes it or the parcel drops it, on any path and
/// while a panic unwinds. A worker runs without what `hold` takes, so a
/// value dropped outside a parcel would be released outside it.
pub(crate) struct Parcel<T> {
    items: VecDeque<T>,
    scope: Option<ValueScope>,
}

impl<T> Parcel<T> {
    pub(crate) fn new(items: Vec<T>, scope: Option<ValueScope>) -> Self {
        Parcel {
            items: items.into(),
            scope,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn push(&mut self, item: T) {
        self.items.push_back(item);
    }

    /// Moves every value of `other` to the end of this parcel.
    pub(crate) fn append(&mut self, mut other: Parcel<T>) {
        self.items.append(&mut other.items);
    }

    /// The values in order, to reorder them in place by keys whose
    /// comparison runs no value code, since this takes no `hold`.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        self.items.make_contiguous()
    }

    /// Takes up to [`VALUE_BATCH`] values from the front and runs `f` on
    /// each, in order and inside one `hold`, where each value ends. Returns
    /// what `f` returned, empty once the parcel is.
    pub(crate) fn take_batch<R>(&mut self, mut f: impl FnMut(T) -> R) -> Vec<R> {
        let Parcel { items, scope } = self;
        let n = items.len().min(VALUE_BATCH);
        let mut out = Vec::with_capacity(n);
        if n > 0 {
            hold(scope.as_ref(), || out.extend(items.drain(..n).map(&mut f)));
        }
        out
    }
}

impl<T> Drop for Parcel<T> {
    fn drop(&mut self) {
        let Parcel { items, scope } = self;
        while !items.is_empty() {
            let n = items.len().min(VALUE_BATCH);
            hold(scope.as_ref(), || drop(items.drain(..n)));
        }
    }
}
