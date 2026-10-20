//! Hooks for incr-persist's mapped writes. Unstable: incr-persist pins
//! this crate's exact version, and nothing else should call them.

use crate::cells::Cells;
use crate::handle::Incr;
use crate::runtime::Runtime;
use crate::value::Value;
use crate::value_slot::{self, Compared};

/// [`Runtime::set`] that hands back what it would drop, for the caller to
/// drop after releasing its own locks: `Replaced` with the displaced value
/// if it wrote, `Equal` with the rejected value if it did not.
pub fn set_returning_displaced<T: Value, C: Cells>(
    rt: &Runtime<C>,
    handle: Incr<T>,
    value: T,
) -> Compared<T, value_slot::Displaced<T, C>> {
    rt.set_returning_displaced(handle, value)
}

#[cfg(all(test, not(loom)))]
#[path = "persist_hooks_tests.rs"]
mod tests;
