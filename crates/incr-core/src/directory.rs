//! `Directory<T, C>`: a growable array of `C::Ptr<T>` entries that
//! readers index with no lock while one serialized writer grows it.
//!
//! The entries live in one allocation behind a length header, published
//! through a single `C::Ptr`, so a reader pays one Acquire load for the
//! directory, one bounds compare and one load for the entry. Growth
//! allocates a directory twice the size, copies the entries, publishes it
//! with a Release store and keeps the old one until the `Directory`
//! drops, so a reader still holding the old one never reads freed memory.
//! The retired directories add up to less than the final one.
//!
//! A reader may hold a stale directory. It still finds every entry it can
//! name: it reached its index through a happens-before chain from the
//! write that filled the entry, so its Acquire load of the directory
//! returns the one that write went into or a later one, and every later
//! directory was copied from it after that write.

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};

use crate::cells::{Cells, PtrCell};
use crate::locks::Lock;

/// The front of a directory allocation; `len` entries follow it.
#[repr(C)]
pub(crate) struct Header {
    len: usize,
}

/// A directory a growth replaced, freed when its [`Directory`] drops.
struct Retired(NonNull<Header>);

// SAFETY: a retired directory is owned by its `Directory` alone, and no
// thread touches it again until that `Directory` drops it.
unsafe impl Send for Retired {}
unsafe impl Sync for Retired {}

pub(crate) struct Directory<T: 'static, C: Cells> {
    current: C::Ptr<Header>,
    // Growth is rare (one per doubling), so a lock here costs nothing on
    // any path that does not grow.
    retired: C::Lock<Vec<Retired>>,
    _entries: PhantomData<T>,
}

impl<T: 'static, C: Cells> Directory<T, C> {
    const ENTRIES_OFFSET: usize =
        std::mem::size_of::<Header>().next_multiple_of(std::mem::align_of::<C::Ptr<T>>());

    pub(crate) fn with_len(len: usize) -> Self {
        assert!(
            len > 0,
            "incr-core: a directory starts with at least one entry"
        );
        Self {
            current: C::Ptr::new(Self::alloc(len).as_ptr()),
            retired: <C::Lock<Vec<Retired>> as Lock<_>>::new(Vec::new()),
            _entries: PhantomData,
        }
    }

    /// The entry at `index`, or `None` past the end.
    #[inline(always)]
    pub(crate) fn get(&self, index: usize) -> Option<&C::Ptr<T>> {
        let dir = self.current.load_acquire();
        // SAFETY: `current` always holds a live directory: it starts
        // non-null, a growth stores only another live one, and every
        // replaced one is kept in `retired` until `self` drops, so the
        // header and the `len` entries after it outlive the borrow of
        // `self`.
        unsafe {
            if index >= (*dir).len {
                return None;
            }
            Some(&*Self::entries(dir).add(index))
        }
    }

    pub(crate) fn len(&self) -> usize {
        // SAFETY: see `get`.
        unsafe { (*self.current.load_acquire()).len }
    }

    /// Make `index` addressable, doubling the directory until it is.
    /// Callers must serialize every `grow_to` on one directory; readers
    /// may run concurrently.
    pub(crate) fn grow_to(&self, index: usize) {
        let old = self.current.load_relaxed();
        // SAFETY: see `get`; the caller's serialization means no other
        // growth replaces `old` meanwhile.
        let old_len = unsafe { (*old).len };
        if index < old_len {
            return;
        }
        let mut len = old_len;
        while len <= index {
            len = len
                .checked_mul(2)
                .expect("incr-core: directory length overflow");
        }
        let new = Self::alloc(len);
        for i in 0..old_len {
            // SAFETY: both directories are live with at least `old_len`
            // entries, and only this serialized writer stores entries.
            unsafe {
                let entry = (*Self::entries(old).add(i)).load_relaxed();
                (*Self::entries(new.as_ptr()).add(i)).store_relaxed(entry);
            }
        }
        self.current.store_release(new.as_ptr());
        // SAFETY: `old` came from `alloc` and was current until the store
        // above.
        self.retired
            .write()
            .push(Retired(unsafe { NonNull::new_unchecked(old) }));
    }

    fn layout(len: usize) -> Layout {
        let (layout, offset) = Layout::new::<Header>()
            .extend(Layout::array::<C::Ptr<T>>(len).expect("incr-core: directory too large"))
            .expect("incr-core: directory too large");
        debug_assert_eq!(offset, Self::ENTRIES_OFFSET);
        layout.pad_to_align()
    }

    /// The first entry of the directory at `dir`. Derived from `dir`
    /// itself, not from a reference to the header, so it keeps the
    /// provenance of the whole allocation.
    #[inline(always)]
    fn entries(dir: *mut Header) -> *mut C::Ptr<T> {
        // SAFETY: the entries start at this offset in every directory
        // allocation (`layout` asserts it).
        unsafe { dir.cast::<u8>().add(Self::ENTRIES_OFFSET).cast() }
    }

    fn alloc(len: usize) -> NonNull<Header> {
        let layout = Self::layout(len);
        // SAFETY: the layout has a non-zero size (it holds the header).
        let Some(dir) = NonNull::new(unsafe { alloc(layout) }.cast::<Header>()) else {
            handle_alloc_error(layout)
        };
        // SAFETY: the allocation fits the header and `len` entries.
        unsafe {
            ptr::write(dir.as_ptr(), Header { len });
            let entries = Self::entries(dir.as_ptr());
            for i in 0..len {
                ptr::write(entries.add(i), C::Ptr::new_null());
            }
        }
        dir
    }

    /// # Safety
    /// `dir` came from [`Self::alloc`] and nothing reads it again.
    unsafe fn free(dir: NonNull<Header>) {
        let len = (*dir.as_ptr()).len;
        let entries = Self::entries(dir.as_ptr());
        for i in 0..len {
            ptr::drop_in_place(entries.add(i));
        }
        dealloc(dir.as_ptr().cast(), Self::layout(len));
    }
}

impl<T: 'static, C: Cells> Drop for Directory<T, C> {
    fn drop(&mut self) {
        let current = self.current.load_relaxed();
        // SAFETY: `&mut self` rules out readers; `current` and every
        // retired directory came from `alloc` and are freed once each.
        unsafe {
            Self::free(NonNull::new_unchecked(current));
            for Retired(dir) in self.retired.write().drain(..) {
                Self::free(dir);
            }
        }
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::cells::{Local, Shared};

    fn grows_and_keeps_entries<C: Cells>() {
        let dir: Directory<u64, C> = Directory::with_len(2);
        let mut boxes: Vec<*mut u64> = Vec::new();
        for i in 0..9usize {
            dir.grow_to(i);
            let b = Box::into_raw(Box::new(i as u64));
            dir.get(i).unwrap().store_release(b);
            boxes.push(b);
        }
        assert_eq!(dir.len(), 16);
        assert!(dir.get(16).is_none());
        for (i, b) in boxes.iter().enumerate() {
            assert_eq!(dir.get(i).unwrap().load_acquire(), *b);
        }
        assert!(dir.get(9).unwrap().load_acquire().is_null());
        for b in boxes {
            // SAFETY: each came from Box::into_raw above.
            drop(unsafe { Box::from_raw(b) });
        }
    }

    #[test]
    fn local_directory_grows_and_keeps_entries() {
        grows_and_keeps_entries::<Local>();
    }

    #[test]
    fn shared_directory_grows_and_keeps_entries() {
        grows_and_keeps_entries::<Shared>();
    }

    #[test]
    fn shared_reader_during_growth() {
        use std::sync::Arc;
        let dir: Arc<Directory<u64, Shared>> = Arc::new(Directory::with_len(1));
        let first = Box::into_raw(Box::new(7u64));
        dir.get(0).unwrap().store_release(first);
        let reader = {
            let dir = Arc::clone(&dir);
            let first = first as usize;
            std::thread::spawn(move || {
                for _ in 0..if cfg!(miri) { 20 } else { 10_000 } {
                    assert_eq!(dir.get(0).unwrap().load_acquire() as usize, first);
                }
            })
        };
        for i in 1..if cfg!(miri) { 64 } else { 4096 } {
            dir.grow_to(i);
        }
        reader.join().unwrap();
        // SAFETY: came from Box::into_raw above.
        drop(unsafe { Box::from_raw(first) });
    }
}
