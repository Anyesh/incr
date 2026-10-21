//! `DependentList`: one node's reverse edges, in two machine words.
//!
//! Up to `2·W` ids (`W = size_of::<usize>() / 4`, so 4 on 64-bit and 2 on
//! wasm32) live inline, packed left to right and padded with `NONE`.
//! Past that the ids move to one heap block, `{len, cap, dead, index,
//! ids[cap]}`, which doubles with `realloc` and is freed when it empties.
//!
//! A remove that had to scan past `INDEX_AT` ids from the back gives the
//! block an index from id to position, so later removes from a list of
//! many dependents (a shared input read by a whole column) cost a hash
//! lookup rather than a scan and a shift. Lists torn down back to front
//! never scan far and never pay for one. An indexed remove writes `NONE`
//! over the id instead of shifting, counted in `dead`; the block
//! compacts once the dead outnumber the live, and the last id is never
//! `NONE`.
//!
//! The mode is read from `head[0]`: not `NONE` means inline, and the
//! tail holds ids; `NONE` means the tail holds the block pointer, null
//! for an empty list. Every write into a mode writes the tail variant
//! that mode reads, so no union field is read before it was written.
//!
//! The runtime keeps each id at most once per list, so `remove` stops at
//! the first match it finds scanning from the back. Order is kept.

use std::alloc::{alloc, dealloc, handle_alloc_error, realloc, Layout};
use std::collections::HashMap;
use std::ptr;

use crate::fx::FxBuildHasher;
use crate::node::NodeId;

const W: usize = std::mem::size_of::<usize>() / 4;
const INLINE: usize = 2 * W;
const NONE: u32 = NodeId::SENTINEL.0;
const INDEX_AT: usize = 64;

type Index = HashMap<u32, u32, FxBuildHasher>;

#[repr(C)]
struct Header {
    len: u32,
    cap: u32,
    dead: u32,
    index: *mut Index,
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

    #[inline(always)]
    fn is_inline(&self) -> bool {
        self.head[0] != NONE
    }

    #[inline(always)]
    fn block(&self) -> *mut Header {
        debug_assert!(!self.is_inline());
        // SAFETY: outside inline mode the tail holds the block variant.
        unsafe { self.tail.block }
    }

    #[inline(always)]
    fn inline_ids(&self) -> &[u32] {
        // SAFETY: in inline mode `head` and the tail's ids are written and
        // form one `[u32; INLINE]` (the asserts above).
        let all = unsafe { &*ptr::from_ref(self).cast::<[u32; INLINE]>() };
        // Entries are packed before the `NONE` padding, so counting them
        // gives the length without a scan that stops early.
        let len = all.iter().map(|&x| usize::from(x != NONE)).sum::<usize>();
        &all[..len]
    }

    fn inline_slot(&mut self, i: usize) -> &mut u32 {
        debug_assert!(i < INLINE);
        // SAFETY: as in `inline_ids`, and `i < INLINE`.
        unsafe { &mut *ptr::from_mut(self).cast::<u32>().add(i) }
    }

    fn block_ids<'a>(block: *mut Header) -> &'a [u32] {
        // SAFETY: as in `block_ids_mut`. Shared, so readers on several
        // threads under the read lock never hold overlapping `&mut`.
        unsafe { std::slice::from_raw_parts(Self::ids_of(block), (*block).len as usize) }
    }

    fn block_ids_mut<'a>(block: *mut Header) -> &'a mut [u32] {
        // SAFETY: a block holds `len` initialized ids after its header, and
        // the caller borrows it through the list that owns it.
        unsafe { std::slice::from_raw_parts_mut(Self::ids_of(block), (*block).len as usize) }
    }

    fn ids_of(block: *mut Header) -> *mut u32 {
        // SAFETY: the ids start right after the header, whose size is a
        // multiple of its alignment and so of `u32`'s.
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
            unsafe { ((*self.block()).len - (*self.block()).dead) as usize }
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        !self.is_inline() && self.block().is_null()
    }

    #[inline]
    fn as_slice(&self) -> &[u32] {
        if self.is_inline() {
            self.inline_ids()
        } else if self.block().is_null() {
            &[]
        } else {
            Self::block_ids(self.block())
        }
    }

    #[inline]
    pub(crate) fn iter(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.as_slice()
            .iter()
            .filter(|&&x| x != NONE)
            .map(|&x| NodeId(x))
    }

    /// Append the ids to `out`. Inline entries are pushed until the
    /// first `NONE`, so a one-entry list costs two compares.
    #[inline(always)]
    pub(crate) fn extend_into(&self, out: &mut Vec<NodeId>) {
        if self.is_inline() {
            // SAFETY: as in `inline_ids`.
            let all = unsafe { &*ptr::from_ref(self).cast::<[u32; INLINE]>() };
            out.push(NodeId(all[0]));
            for &x in &all[1..] {
                if x == NONE {
                    return;
                }
                out.push(NodeId(x));
            }
        } else if !self.block().is_null() {
            self.extend_from_block(out);
        }
    }

    #[cold]
    #[inline(never)]
    fn extend_from_block(&self, out: &mut Vec<NodeId>) {
        let ids = Self::block_ids(self.block()).iter();
        // SAFETY: a non-null block is live.
        if unsafe { (*self.block()).dead } == 0 {
            out.extend(ids.map(|&x| NodeId(x)));
        } else {
            out.extend(ids.filter(|&&x| x != NONE).map(|&x| NodeId(x)));
        }
    }

    #[inline]
    pub(crate) fn last(&self) -> Option<NodeId> {
        self.as_slice().last().map(|&x| NodeId(x))
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, id: NodeId) -> bool {
        debug_assert_ne!(id.0, NONE);
        self.as_slice().contains(&id.0)
    }

    /// Append `id` unless it is already the last entry. The runtime
    /// pushes each dependent's id once per dep it read, and a dep read
    /// twice is pushed twice in a row, so this keeps every id unique.
    #[inline]
    pub(crate) fn push_unique(&mut self, id: NodeId) {
        debug_assert_ne!(id.0, NONE, "NodeId::SENTINEL is never a dependent");
        if self.is_inline() {
            let n = self.inline_ids().len();
            if self.inline_ids()[n - 1] == id.0 {
                return;
            }
            if n < INLINE {
                *self.inline_slot(n) = id.0;
            } else {
                self.spill(id.0);
            }
        } else if self.block().is_null() {
            self.head = [NONE; W];
            self.head[0] = id.0;
            self.tail = Tail { ids: [NONE; W] };
        } else if self.last() != Some(id) {
            self.push_to_block(id);
        }
    }

    #[inline(never)]
    fn push_to_block(&mut self, id: NodeId) {
        let block = self.block();
        // SAFETY: a non-null block is live and owned by this list.
        unsafe {
            if (*block).len == (*block).cap {
                self.tail.block = Self::grow(block);
            }
            let block = self.tail.block;
            let at = (*block).len;
            Self::ids_of(block).add(at as usize).write(id.0);
            (*block).len += 1;
            if let Some(index) = (*block).index.as_mut() {
                let had = index.insert(id.0, at);
                debug_assert!(had.is_none(), "{id:?} is already a dependent");
            }
        }
    }

    #[cold]
    unsafe fn build_index(block: *mut Header) {
        let ids = Self::block_ids(block);
        let mut index = Index::with_capacity_and_hasher(ids.len(), FxBuildHasher);
        index.extend((0..).zip(ids).map(|(at, &x)| (x, at)));
        debug_assert_eq!(index.len(), ids.len(), "a dependent is listed twice");
        (*block).index = Box::into_raw(Box::new(index));
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
                dead: 0,
                index: ptr::null_mut(),
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
    #[inline]
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
            self.remove_from_block(id)
        }
    }

    #[inline(never)]
    fn remove_from_block(&mut self, id: NodeId) -> bool {
        let block = self.block();
        // SAFETY: a non-null block is live and owned by this list.
        if !unsafe { (*block).index }.is_null() {
            return self.remove_indexed(id);
        }
        let ids = Self::block_ids_mut(block);
        let Some(at) = ids.iter().rposition(|&x| x == id.0) else {
            return false;
        };
        ids.copy_within(at + 1.., at);
        let scanned = ids.len() - at;
        // SAFETY: as above.
        unsafe {
            (*block).len -= 1;
            if (*block).len == 0 {
                Self::free(block);
                self.tail.block = ptr::null_mut();
            } else if scanned > INDEX_AT {
                Self::build_index(block);
            }
        }
        true
    }

    /// Mark `id`'s entry removed. Trailing `NONE`s are dropped, so the
    /// last id stays live, and the block compacts once more than half its
    /// entries are `NONE`.
    #[inline(never)]
    fn remove_indexed(&mut self, id: NodeId) -> bool {
        let block = self.block();
        // SAFETY: the caller saw this live block's index non-null.
        let Some(at) = (unsafe { &mut *(*block).index }).remove(&id.0) else {
            return false;
        };
        let ids = Self::block_ids_mut(block);
        ids[at as usize] = NONE;
        let mut len = ids.len();
        // SAFETY: as above.
        let mut dead = unsafe { (*block).dead } + 1;
        while len > 0 && ids[len - 1] == NONE {
            len -= 1;
            dead -= 1;
        }
        // SAFETY: as above.
        unsafe {
            (*block).len = len as u32;
            (*block).dead = dead;
            if len == 0 {
                Self::free(block);
                self.tail.block = ptr::null_mut();
            } else if 2 * dead as usize > len {
                Self::compact(block);
            }
        }
        true
    }

    #[cold]
    unsafe fn compact(block: *mut Header) {
        let ids = Self::block_ids_mut(block);
        let mut live = 0;
        for at in 0..ids.len() {
            if ids[at] != NONE {
                ids[live] = ids[at];
                live += 1;
            }
        }
        (*block).len = live as u32;
        (*block).dead = 0;
        let index = (*block).index;
        if live < INDEX_AT / 2 {
            drop(Box::from_raw(index));
            (*block).index = ptr::null_mut();
        } else {
            let index = &mut *index;
            for (at, &x) in (0..).zip(&ids[..live]) {
                index.insert(x, at);
            }
        }
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        if !self.is_inline() && !self.block().is_null() {
            self.free_block();
        }
        self.set_empty();
    }

    #[cold]
    #[inline(never)]
    fn free_block(&mut self) {
        // SAFETY: a non-null block is live and owned by this list.
        unsafe { Self::free(self.block()) };
    }

    // Field writes, not `*self = Self::new()`: that assignment would drop
    // the old value, and `Drop` calls `clear`.
    #[inline(always)]
    fn set_empty(&mut self) {
        self.head = [NONE; W];
        self.tail = Tail {
            block: ptr::null_mut(),
        };
    }

    #[cfg(all(test, not(loom)))]
    fn is_indexed(&self) -> bool {
        // SAFETY: a non-null block is live.
        self.has_block() && !unsafe { (*self.block()).index }.is_null()
    }

    /// Whether the list holds a heap block.
    pub(crate) fn has_block(&self) -> bool {
        !self.is_inline() && !self.block().is_null()
    }

    unsafe fn free(block: *mut Header) {
        if !(*block).index.is_null() {
            drop(Box::from_raw((*block).index));
        }
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

    fn wide_op() -> impl Strategy<Value = Op> {
        // Enough distinct ids that lists pass `INDEX_AT`, and removes
        // often enough that indexed blocks compact and drop their index.
        prop_oneof![
            7 => (0..2_usize, 0..160_u32).prop_map(|(l, x)| Op::Push(l, x)),
            4 => (0..2_usize, 0..160_u32).prop_map(|(l, x)| Op::Remove(l, x)),
            1 => Just(Op::Clear(0)),
        ]
    }

    fn run(ops: &[Op]) {
        let mut lists: Vec<DependentList> = (0..LISTS).map(|_| DependentList::new()).collect();
        let mut model: Vec<Vec<u32>> = vec![Vec::new(); LISTS];
        for op in ops {
            match *op {
                Op::Push(l, x) => {
                    // The runtime pushes an id a list holds only right after
                    // pushing it, which `push_unique` skips.
                    if model[l].last() == Some(&x) {
                        lists[l].push_unique(NodeId(x));
                    } else if !model[l].contains(&x) {
                        lists[l].push_unique(NodeId(x));
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
                let mut out = vec![NodeId(7)];
                list.extend_into(&mut out);
                let out: Vec<u32> = out.iter().map(|n| n.0).collect();
                assert_eq!(out[1..], want[..]);
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

        #[test]
        fn indexed_lists_match_a_vec_model(ops in proptest::collection::vec(wide_op(), 0..if cfg!(miri) { 300 } else { 1500 })) {
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
    fn indexes_compacts_and_drops_the_index() {
        let n = 4 * INDEX_AT as u32;
        let mut ops: Vec<Op> = (0..n).map(|x| Op::Push(0, x)).collect();
        ops.extend((0..n).step_by(2).map(|x| Op::Remove(0, x)));
        ops.push(Op::Remove(0, n - 1));
        ops.extend((0..n).map(|x| Op::Push(0, n + x)));
        ops.extend((1..2 * n).step_by(2).map(|x| Op::Remove(0, x)));
        ops.extend((n..2 * n).rev().step_by(2).map(|x| Op::Remove(0, x)));
        run(&ops);
    }

    #[test]
    fn only_a_long_scan_builds_the_index() {
        let n = 4 * INDEX_AT as u32;
        let mut list = DependentList::new();
        for x in 0..n {
            list.push_unique(NodeId(x));
        }
        for x in (n / 2..n).rev() {
            assert!(list.remove(NodeId(x)));
        }
        assert!(!list.is_indexed(), "back-to-front removes built an index");
        assert!(list.remove(NodeId(0)));
        assert!(list.is_indexed());
        assert_eq!(list.last(), Some(NodeId(n / 2 - 1)));
    }

    #[test]
    fn stays_inline_up_to_two_words() {
        let mut list = DependentList::new();
        for x in 0..INLINE as u32 {
            list.push_unique(NodeId(x));
        }
        assert!(!list.has_block());
        list.push_unique(NodeId(99));
        assert!(list.has_block());
    }
}
