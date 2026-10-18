//! `SlotVec<T>`: a per-node table behind the runtime's `inner` lock,
//! indexed by node slot.
//!
//! Entries live in boxed segments that never move, so an append never
//! copies existing entries under `inner.write()`, and a pointer into an
//! entry stays valid across later appends. That lets `compute_one` call a
//! node's closure after it releases the lock (see `Runtime::compute_one`).
//! An index is a shift, a mask and two loads.

use std::ops::{Index, IndexMut};

// Four-entry segments under Miri, so its tests cross segments.
#[cfg(miri)]
const SEGMENT_SHIFT: u32 = 2;
#[cfg(not(miri))]
const SEGMENT_SHIFT: u32 = 10;
const SEGMENT_SIZE: usize = 1 << SEGMENT_SHIFT;
const SEGMENT_MASK: usize = SEGMENT_SIZE - 1;

pub(crate) struct SlotVec<T> {
    segments: Vec<Box<[T; SEGMENT_SIZE]>>,
    len: usize,
}

impl<T: Default> SlotVec<T> {
    pub(crate) const fn new() -> Self {
        Self {
            segments: Vec::new(),
            len: 0,
        }
    }

    pub(crate) fn push(&mut self, value: T) {
        if self.len & SEGMENT_MASK == 0 {
            self.segments.push(Self::new_segment());
        }
        let at = self.len;
        self.len += 1;
        self[at] = value;
    }

    #[cold]
    fn new_segment() -> Box<[T; SEGMENT_SIZE]> {
        let segment: Box<[T]> = (0..SEGMENT_SIZE).map(|_| T::default()).collect();
        segment
            .try_into()
            .unwrap_or_else(|_| unreachable!("a segment has SEGMENT_SIZE entries"))
    }

    #[cfg(all(test, not(loom)))]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn get(&self, index: usize) -> Option<&T> {
        (index < self.len).then(|| &self[index])
    }
}

impl<T> Index<usize> for SlotVec<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &T {
        &self.segments[index >> SEGMENT_SHIFT][index & SEGMENT_MASK]
    }
}

impl<T> IndexMut<usize> for SlotVec<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.segments[index >> SEGMENT_SHIFT][index & SEGMENT_MASK]
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn entries_stay_put_across_appends() {
        let mut v: SlotVec<Option<Box<u32>>> = SlotVec::new();
        v.push(Some(Box::new(7)));
        let first: *const Option<Box<u32>> = &v[0];
        for i in 1..3 * SEGMENT_SIZE as u32 {
            v.push(Some(Box::new(i)));
        }
        assert!(std::ptr::eq(first, &v[0]));
        assert_eq!(v.len(), 3 * SEGMENT_SIZE);
        for i in 0..v.len() {
            let want = if i == 0 { 7 } else { i as u32 };
            assert_eq!(v[i].as_deref(), Some(&want));
        }
        assert!(v.get(v.len()).is_none());
        v[1] = None;
        assert!(v[1].is_none());
    }
}
