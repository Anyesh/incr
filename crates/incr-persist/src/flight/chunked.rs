use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::{Control, Verdict};
use crate::value_context::idle;
use crate::{Error, Result, ValueScope};

const CHUNK_ITEMS: usize = 256;
const CHUNK_TIME: Duration = Duration::from_millis(2);

/// The iterator a flight's worker hands its backend. Each `next` pops one
/// item from a buffer, refilling it first, when empty, with up to 256 items
/// or 2 ms of pulls from `source` inside the value scope's `idle`. A pull
/// that fails is buffered at its position, after the items pulled before
/// it, and a cancel or abandon noticed at a `next` is handed out as
/// [`Error::Cancelled`]; after an `Err` is handed out, `next` returns
/// `None`.
pub(crate) struct Chunked<T, I> {
    source: I,
    buffer: VecDeque<Result<T>>,
    control: Arc<Control>,
    scope: Option<ValueScope>,
    source_ended: bool,
    source_failed: bool,
    handed_err: bool,
    handed_cancel: bool,
    refilling: bool,
}

impl<T, I: Iterator<Item = Result<T>>> Chunked<T, I> {
    pub(crate) fn new(source: I, control: Arc<Control>, scope: Option<ValueScope>) -> Self {
        Chunked {
            source,
            buffer: VecDeque::with_capacity(CHUNK_ITEMS),
            control,
            scope,
            source_ended: false,
            source_failed: false,
            handed_err: false,
            handed_cancel: false,
            refilling: false,
        }
    }

    /// The verdict on the flight once the backend returned `result`, and
    /// the source. Failure is tested first, since a source returns `None`
    /// after its `Err`: `Ok` after any `Err` was handed out is
    /// [`Verdict::Truncated`]. After a plain `Ok` the run is whole when the
    /// buffer is empty and the source is at its end; when the end was not
    /// seen, this pulls the source once more, inside `idle`, to tell.
    pub(crate) fn finish<R>(mut self, result: Result<R>) -> (Verdict<R>, I) {
        let verdict = match result {
            Ok(_) if self.handed_err || self.refilling => Verdict::Truncated,
            Err(_) if self.handed_cancel => Verdict::Cancelled,
            Err(error) => Verdict::Failed(error),
            Ok(committed) if self.pulled_to_end() => Verdict::Committed(committed),
            Ok(_) => Verdict::Truncated,
        };
        (verdict, self.source)
    }

    fn pulled_to_end(&mut self) -> bool {
        if !self.buffer.is_empty() {
            return false;
        }
        if self.source_ended {
            return true;
        }
        let Chunked { source, scope, .. } = self;
        let mut more = false;
        idle(scope.as_ref(), || more = source.next().is_some());
        !more
    }

    fn refill(&mut self) {
        let Chunked {
            source,
            buffer,
            scope,
            source_ended,
            source_failed,
            ..
        } = self;
        let deadline = Instant::now() + CHUNK_TIME;
        idle(scope.as_ref(), || {
            while buffer.len() < CHUNK_ITEMS {
                match source.next() {
                    None => {
                        *source_ended = true;
                        return;
                    }
                    Some(Err(error)) => {
                        buffer.push_back(Err(error));
                        *source_failed = true;
                        return;
                    }
                    Some(Ok(item)) => buffer.push_back(Ok(item)),
                }
                if Instant::now() >= deadline {
                    return;
                }
            }
        });
    }
}

impl<T, I: Iterator<Item = Result<T>>> Iterator for Chunked<T, I> {
    type Item = Result<T>;

    fn next(&mut self) -> Option<Result<T>> {
        if self.handed_err || self.refilling {
            return None;
        }
        if self.control.stopped() {
            self.handed_err = true;
            self.handed_cancel = true;
            return Some(Err(Error::Cancelled));
        }
        if self.buffer.is_empty() && !self.source_ended && !self.source_failed {
            self.refilling = true;
            self.refill();
            self.refilling = false;
        }
        let item = self.buffer.pop_front()?;
        if item.is_err() {
            self.handed_err = true;
        }
        Some(item)
    }
}
