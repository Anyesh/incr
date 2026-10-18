//! `DependentList`: one node's reverse edges, in two machine words.
//!
//! Up to `2·W` ids (`W = size_of::<usize>() / 4`, so 4 on 64-bit and 2 on
//! wasm32) live inline, packed left to right and padded with `NONE`.
//! Past that the ids move to one heap block, `{len, cap, ids[cap]}`,
//! which doubles with `realloc` and is freed when it empties.
//!
//! The mode is read from `head[0]`: not `NONE` means inline, and the
//! tail holds ids; `NONE` means the tail holds the block pointer, null
//! for an empty list. Every write into a mode writes the tail variant
//! that mode reads, so no union field is read before it was written.
//!
//! The runtime keeps each id at most once per list, so `remove` stops at
//! the first match it finds scanning from the back. Order is kept.

use std::alloc::{alloc, dealloc, handle_alloc_error, realloc, Layout};
use std::ptr;

use crate::node::NodeId;

const W: usize = std::mem::size_of::<usize>() / 4;
const INLINE: usize = 2 * W;
const NONE: u32 = NodeId::SENTINEL.0;

#[repr(C)]
struct Header {
    len: u32,
    cap: u32,
}

#[repr(C)]
union Tail {
    ids: [u32; W],
    block: *mut Header,
}

#[repr(C)]
pub(crate) struct DependentList {
    head: [u32; W],
    tail: Tail,
}

// The inline ids are read as one `[u32; INLINE]` starting at `head`.
const _: () = assert!(std::mem::offset_of!(DependentList, tail) == 4 * W);
const _: () = assert!(std::mem::size_of::<DependentList>() == 2 * std::mem::size_of::<usize>());

// SAFETY: the list owns its block outright and has no interior
// mutability, so moving or sharing it is as safe as for a `Vec<u32>`.
unsafe impl Send for DependentList {}
unsafe impl Sync for DependentList {}

impl Default for DependentList {
    fn default() -> Self {
        Self::new()
    }
}

impl DependentList {
    pub(crate) const fn new() -> Self {
        Self {
            head: [NONE; W],
            tail: Tail {
                block: ptr::null_mut(),
            },
        }
    }

    #[inline]
    fn is_inline(&self) -> bool {
        self.head[0] != NONE
    }

    #[inline]
    fn block(&self) -> *mut Header {
        debug_assert!(!self.is_inline());
        // SAFETY: outside inline mode the tail holds the block variant.
        unsafe { self.tail.block }
    }

    fn inline_ids(&self) -> &[u32] {
        // SAFETY: in inline mode `head` and the tail's ids are written and
        // form one `[u32; INLINE]` (the asserts above); the length counts
        // the entries before the first `NONE`.
        unsafe {
            let all = std::slice::from_raw_parts(ptr::from_ref(self).cast::<u32>(), INLINE);
            &all[..all.iter().position(|&x| x == NONE).unwrap_or(INLINE)]
        }
    }

    fn inline_slot(&mut self, i: usize) -> &mut u32 {
        debug_assert!(i < INLINE);
        // SAFETY: as in `inline_ids`, and `i < INLINE`.
        unsafe { &mut *ptr::from_mut(self).cast::<u32>().add(i) }
    }

    fn block_ids<'a>(block: *mut Header) -> &'a mut [u32] {
        // SAFETY: a block holds `len` initialized ids after its header, and
        // the caller borrows it through the list that owns it.
        unsafe { std::slice::from_raw_parts_mut(Self::ids_of(block), (*block).len as usize) }
    }

    fn ids_of(block: *mut Header) -> *mut u32 {
        // SAFETY: the ids start right after the 8-byte, 4-aligned header.
        unsafe { block.add(1).cast::<u32>() }
    }

    fn layout(cap: usize) -> Layout {
        Layout::new::<Header>()
            .extend(Layout::array::<u32>(cap).expect("incr-core: dependent list too long"))
            .expect("incr-core: dependent list too long")
            .0
            .pad_to_align()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        if self.is_inline() {
            self.inline_ids().len()
        } else if self.block().is_null() {
            0
        } else {
            // SAFETY: a non-null block is live.
            unsafe { (*self.block()).len as usize }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        !self.is_inline() && self.block().is_null()
    }

    fn as_slice(&self) -> &[u32] {
        if self.is_inline() {
            self.inline_ids()
        } else if self.block().is_null() {
            &[]
        } else {
            Self::block_ids(self.block())
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.as_slice().iter().map(|&x| NodeId(x))
    }

    pub(crate) fn last(&self) -> Option<NodeId> {
        self.as_slice().last().map(|&x| NodeId(x))
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, id: NodeId) -> bool {
        self.as_slice().contains(&id.0)
    }

    pub(crate) fn push(&mut self, id: NodeId) {
        debug_assert_ne!(id.0, NONE, "NodeId::SENTINEL is never a dependent");
        if self.is_inline() {
            let n = self.inline_ids().len();
            if n < INLINE {
                *self.inline_slot(n) = id.0;
            } else {
                self.spill(id.0);
            }
        } else if self.block().is_null() {
            self.head = [NONE; W];
            self.head[0] = id.0;
            self.tail = Tail { ids: [NONE; W] };
        } else {
            let block = self.block();
            // SAFETY: a non-null block is live and owned by this list.
            unsafe {
                if (*block).len == (*block).cap {
                    self.tail.block = Self::grow(block);
                }
                let block = self.tail.block;
                Self::ids_of(block).add((*block).len as usize).write(id.0);
                (*block).len += 1;
            }
        }
    }

    #[cold]
    fn spill(&mut self, id: u32) {
        let cap = 4 * W;
        let layout = Self::layout(cap);
        // SAFETY: the layout has a non-zero size.
        let block = unsafe { alloc(layout) }.cast::<Header>();
        if block.is_null() {
            handle_alloc_error(layout);
        }
        // SAFETY: the block fits a header and `cap > INLINE` ids.
        unsafe {
            ptr::copy_nonoverlapping(self.inline_ids().as_ptr(), Self::ids_of(block), INLINE);
            Self::ids_of(block).add(INLINE).write(id);
            block.write(Header {
                len: INLINE as u32 + 1,
                cap: cap as u32,
            });
        }
        self.head = [NONE; W];
        self.tail = Tail { block };
    }

    #[cold]
    unsafe fn grow(block: *mut Header) -> *mut Header {
        let cap = (*block).cap as usize;
        let new_cap = cap
            .checked_mul(2)
            .expect("incr-core: dependent list too long");
        let new_layout = Self::layout(new_cap);
        let grown = realloc(block.cast(), Self::layout(cap), new_layout.size()).cast::<Header>();
        if grown.is_null() {
            handle_alloc_error(new_layout);
        }
        (*grown).cap = u32::try_from(new_cap).expect("incr-core: dependent list too long");
        grown
    }

    /// Remove `id`, scanning from the back. Returns whether it was there.
    pub(crate) fn remove(&mut self, id: NodeId) -> bool {
        if self.is_inline() {
            let ids = self.inline_ids();
            let n = ids.len();
            let Some(at) = ids.iter().rposition(|&x| x == id.0) else {
                return false;
            };
            for i in at..n - 1 {
                *self.inline_slot(i) = *self.inline_slot(i + 1);
            }
            *self.inline_slot(n - 1) = NONE;
            if n == 1 {
                self.set_empty();
            }
            true
        } else if self.block().is_null() {
            false
        } else {
            let block = self.block();
            let ids = Self::block_ids(block);
            let Some(at) = ids.iter().rposition(|&x| x == id.0) else {
                return false;
            };
            ids.copy_within(at + 1.., at);
            // SAFETY: a non-null block is live and owned by this list.
            unsafe {
                (*block).len -= 1;
                if (*block).len == 0 {
                    Self::free(block);
                    self.tail.block = ptr::null_mut();
                }
            }
            true
        }
    }

    pub(crate) fn clear(&mut self) {
        if !self.is_inline() && !self.block().is_null() {
            // SAFETY: a non-null block is live and owned by this list.
            unsafe { Self::free(self.block()) };
        }
        self.set_empty();
    }

    // Field writes, not `*self = Self::new()`: that assignment would drop
    // the old value, and `Drop` calls `clear`.
    fn set_empty(&mut self) {
        self.head = [NONE; W];
        self.tail = Tail {
            block: ptr::null_mut(),
        };
    }

    /// Whether the list holds a heap block.
    pub(crate) fn has_block(&self) -> bool {
        !self.is_inline() && !self.block().is_null()
    }

    unsafe fn free(block: *mut Header) {
        dealloc(block.cast(), Self::layout((*block).cap as usize));
    }
}

impl Drop for DependentList {
    fn drop(&mut self) {
        self.clear();
    }
}

impl std::fmt::Debug for DependentList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[derive(Clone, Debug)]
    enum Op {
        Push(usize, u32),
        Remove(usize, u32),
        Clear(usize),
    }

    const LISTS: usize = 4;

    fn op() -> impl Strategy<Value = Op> {
        // Ids from a small range, so removes often hit.
        prop_oneof![
            6 => (0..LISTS, 0..24_u32).prop_map(|(l, x)| Op::Push(l, x)),
            4 => (0..LISTS, 0..24_u32).prop_map(|(l, x)| Op::Remove(l, x)),
            1 => (0..LISTS).prop_map(Op::Clear),
        ]
    }

    fn run(ops: &[Op]) {
        let mut lists: Vec<DependentList> = (0..LISTS).map(|_| DependentList::new()).collect();
        let mut model: Vec<Vec<u32>> = vec![Vec::new(); LISTS];
        for op in ops {
            match *op {
                Op::Push(l, x) => {
                    // The runtime never pushes an id a list already holds.
                    if !model[l].contains(&x) {
                        lists[l].push(NodeId(x));
                        model[l].push(x);
                    }
                }
                Op::Remove(l, x) => {
                    let at = model[l].iter().rposition(|&y| y == x);
                    if let Some(at) = at {
                        model[l].remove(at);
                    }
                    assert_eq!(lists[l].remove(NodeId(x)), at.is_some(), "{op:?}");
                }
                Op::Clear(l) => {
                    lists[l].clear();
                    model[l].clear();
                }
            }
            for (list, want) in lists.iter().zip(&model) {
                let got: Vec<u32> = list.iter().map(|n| n.0).collect();
                assert_eq!(&got, want, "after {op:?}");
                assert_eq!(list.len(), want.len());
                assert_eq!(list.is_empty(), want.is_empty());
                assert_eq!(list.last().map(|n| n.0), want.last().copied());
                if want.len() > INLINE {
                    assert!(list.has_block(), "{} ids fit inline", want.len());
                }
                if want.is_empty() {
                    assert!(!list.has_block(), "an empty list kept its block");
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: if cfg!(miri) { 6 } else { 512 },
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        #[test]
        fn matches_a_vec_model(ops in proptest::collection::vec(op(), 0..if cfg!(miri) { 60 } else { 300 })) {
            run(&ops);
        }
    }

    #[test]
    fn spills_grows_and_frees() {
        let mut ops: Vec<Op> = (0..20).map(|x| Op::Push(0, x)).collect();
        ops.extend((0..20).rev().map(|x| Op::Remove(0, x)));
        ops.extend((0..20).map(|x| Op::Push(1, x)));
        ops.extend((0..20).map(|x| Op::Remove(1, x)));
        run(&ops);
    }

    #[test]
    fn stays_inline_up_to_two_words() {
        let mut list = DependentList::new();
        for x in 0..INLINE as u32 {
            list.push(NodeId(x));
        }
        assert!(!list.has_block());
        list.push(NodeId(99));
        assert!(list.has_block());
    }
}
