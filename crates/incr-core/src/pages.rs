//! Zeroed memory for table storage that can be given back in pieces.
//!
//! A table that grows by migration (see [`crate::grow`]) empties its old
//! storage front to back over many writes. On Linux a large block is an
//! anonymous mapping, so its pages arrive untouched, the ones a migration
//! has emptied go back to the system as it passes them, and freeing the
//! block at the end is not one long call. Elsewhere, and for small blocks,
//! the memory comes from the global allocator and leaves in one piece.

use std::alloc::{self, Layout};
use std::ptr::NonNull;

/// A zeroed block of `layout`, which must not be zero-sized.
pub(crate) fn alloc_zeroed(layout: Layout) -> NonNull<u8> {
    debug_assert!(layout.size() != 0);
    if sys::maps(layout) {
        return sys::map(layout);
    }
    // SAFETY: the size is not zero.
    let ptr = unsafe { alloc::alloc_zeroed(layout) };
    NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout))
}

/// A block of `layout`, which must not be zero-sized, whose bytes from
/// `from` on are zero. [`dealloc`] and [`release`] take it as they take
/// one from [`alloc_zeroed`].
pub(crate) fn alloc_zeroed_from(layout: Layout, from: usize) -> NonNull<u8> {
    debug_assert!(layout.size() != 0 && from <= layout.size());
    if sys::maps(layout) {
        return sys::map(layout);
    }
    // SAFETY: the size is not zero.
    let ptr = unsafe { alloc::alloc(layout) };
    let ptr = NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout));
    // SAFETY: bytes `from..layout.size()` lie inside the block.
    unsafe { ptr.add(from).write_bytes(0, layout.size() - from) };
    ptr
}

/// # Safety
/// `ptr` must come from [`alloc_zeroed`] with this `layout` and not have
/// been passed here before.
pub(crate) unsafe fn dealloc(ptr: NonNull<u8>, layout: Layout) {
    if sys::maps(layout) {
        // SAFETY: mapped by `alloc_zeroed` with this size.
        unsafe { sys::unmap(ptr, layout) }
    } else {
        // SAFETY: allocated by `alloc_zeroed` with this layout.
        unsafe { alloc::dealloc(ptr.as_ptr(), layout) }
    }
}

/// Gives back the memory of bytes `from..to` of the block, where the block
/// can: whole release units only, rounding both ends down, so calls over
/// consecutive ranges starting at 0 release every unit below the last
/// `to`. The bytes given back read as zero afterwards.
///
/// # Safety
/// `ptr` must come from [`alloc_zeroed`] with this `layout`, `from <= to <=
/// layout.size()`, and nothing may rely on the contents of the bytes below
/// `to`.
pub(crate) unsafe fn release(ptr: NonNull<u8>, layout: Layout, from: usize, to: usize) {
    if sys::maps(layout) {
        // SAFETY: the range lies in the mapping.
        unsafe { sys::release(ptr, from, to) }
    }
}

#[cfg(all(target_os = "linux", not(miri)))]
mod sys {
    use std::alloc::{self, Layout};
    use std::ptr::{self, NonNull};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Smaller blocks are not worth a mapping each: the allocator frees
    /// one in well under a millisecond.
    const MAP_MIN: usize = 2 << 20;
    /// Pages go back this many bytes at a time, so a migration makes a
    /// system call every few dozen writes and not on each.
    const RELEASE_UNIT: usize = 256 << 10;

    fn page_size() -> usize {
        static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);
        match PAGE_SIZE.load(Ordering::Relaxed) {
            0 => {
                // SAFETY: `sysconf` reads a system constant.
                let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
                let size = usize::try_from(size).expect("incr-core: the system has a page size");
                PAGE_SIZE.store(size, Ordering::Relaxed);
                size
            }
            size => size,
        }
    }

    pub(super) fn maps(layout: Layout) -> bool {
        layout.size() >= MAP_MIN
    }

    pub(super) fn map(layout: Layout) -> NonNull<u8> {
        assert!(
            layout.align() <= page_size(),
            "incr-core: table storage is aligned to more than a page"
        );
        // SAFETY: an anonymous private mapping at an address the system
        // picks touches no existing memory.
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                layout.size(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            alloc::handle_alloc_error(layout)
        }
        NonNull::new(ptr.cast()).unwrap_or_else(|| alloc::handle_alloc_error(layout))
    }

    /// # Safety
    /// `ptr` must be a live mapping from [`map`] with this `layout`.
    pub(super) unsafe fn unmap(ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the caller's.
        let rc = unsafe { libc::munmap(ptr.as_ptr().cast(), layout.size()) };
        debug_assert_eq!(rc, 0, "incr-core: munmap of table storage failed");
    }

    /// # Safety
    /// `from..to` must lie in a live mapping from [`map`] at `ptr`, and
    /// nothing may rely on the contents of the bytes below `to`.
    pub(super) unsafe fn release(ptr: NonNull<u8>, from: usize, to: usize) {
        let unit = RELEASE_UNIT.max(page_size());
        let (from, to) = (from / unit * unit, to / unit * unit);
        if from == to {
            return;
        }
        // SAFETY: the unit-aligned range lies in the mapping, whose start
        // is page-aligned. A refused request leaves the pages resident,
        // which costs memory and nothing else, so its result is unused.
        let rc = unsafe {
            libc::madvise(
                ptr.as_ptr().add(from).cast(),
                to - from,
                libc::MADV_DONTNEED,
            )
        };
        debug_assert_eq!(rc, 0, "incr-core: madvise of table storage failed");
    }
}

#[cfg(not(all(target_os = "linux", not(miri))))]
mod sys {
    use std::alloc::Layout;
    use std::ptr::NonNull;

    pub(super) fn maps(_layout: Layout) -> bool {
        false
    }

    pub(super) fn map(layout: Layout) -> NonNull<u8> {
        unreachable!("incr-core: no mapping for {layout:?} on this target")
    }

    /// # Safety
    /// Never called: `maps` is false.
    pub(super) unsafe fn unmap(_ptr: NonNull<u8>, layout: Layout) {
        unreachable!("incr-core: no mapping for {layout:?} on this target")
    }

    /// # Safety
    /// Never called: `maps` is false.
    pub(super) unsafe fn release(_ptr: NonNull<u8>, _from: usize, _to: usize) {}
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    /// One block the allocator serves and one large enough to map, which
    /// Miri neither maps nor walks in reasonable time.
    fn sizes() -> [usize; 2] {
        [4096, if cfg!(miri) { 16 << 10 } else { 4 << 20 }]
    }

    fn block(size: usize) -> (NonNull<u8>, Layout) {
        let layout = Layout::from_size_align(size, 8).unwrap();
        (alloc_zeroed(layout), layout)
    }

    #[test]
    fn a_block_of_either_size_arrives_zeroed_and_holds_what_is_written() {
        for size in sizes() {
            let (ptr, layout) = block(size);
            // SAFETY: the block is `size` bytes, written and read in range.
            unsafe {
                let bytes = std::slice::from_raw_parts_mut(ptr.as_ptr(), size);
                assert!(bytes.iter().all(|b| *b == 0));
                bytes.fill(0xA5);
                assert!(bytes.iter().all(|b| *b == 0xA5));
                dealloc(ptr, layout);
            }
        }
    }

    #[test]
    fn releasing_keeps_every_byte_from_the_end_of_the_range_on() {
        for size in sizes() {
            let (ptr, layout) = block(size);
            let mut released = 0;
            // SAFETY: the block is `size` bytes; the ranges released are
            // consecutive from 0 and nothing below them is relied on.
            unsafe {
                std::slice::from_raw_parts_mut(ptr.as_ptr(), size).fill(0xA5);
                for to in (0..=size).step_by(size / 16) {
                    release(ptr, layout, released, to);
                    released = to;
                    let kept = std::slice::from_raw_parts(ptr.as_ptr().add(to), size - to);
                    assert!(kept.iter().all(|b| *b == 0xA5), "after releasing {to}");
                }
                dealloc(ptr, layout);
            }
        }
    }

    #[cfg(all(target_os = "linux", not(miri)))]
    #[test]
    fn a_mapped_block_reads_zero_where_it_was_released() {
        let size = 4 << 20;
        let (ptr, layout) = block(size);
        // SAFETY: as in the test above.
        unsafe {
            std::slice::from_raw_parts_mut(ptr.as_ptr(), size).fill(0xA5);
            release(ptr, layout, 0, size / 2);
            let bytes = std::slice::from_raw_parts(ptr.as_ptr(), size);
            assert!(bytes[..size / 2].iter().all(|b| *b == 0));
            assert!(bytes[size / 2..].iter().all(|b| *b == 0xA5));
            dealloc(ptr, layout);
        }
    }
}
