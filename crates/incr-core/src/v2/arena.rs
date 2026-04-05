//! Typed value arenas.
//!
//! Per section 5.2 of the concurrent core rewrite spec, node values live in
//! per-type arenas indexed by a type tag stored in `NodeData`. This module
//! defines the type-erased arena trait stored in the runtime's arena
//! registry, and the concrete arena implementations the runtime uses to
//! store values without paying the `Box<dyn Any>` tax.
//!
//! ## Arena implementations
//!
//! - [`AtomicPrimitiveArena`] holds `Copy` primitive values (`u32`, `i32`,
//!   `u64`, `i64`, `f32`, `f64`, `bool`) inline as atomic cells. Reads and
//!   writes are tear-free by construction, so the arena does not coordinate
//!   with any state machine for the value slot itself. The node state
//!   machine is still the authority on whether the value is *meaningful*
//!   (Clean vs Dirty), but the raw bytes can be loaded any time without
//!   undefined behavior.
//! - [`GenericArena`] holds everything else (`Clone + Send + Sync` types)
//!   in `UnsafeCell<Option<T>>` slots. Access is coordinated by the node
//!   state machine: readers only touch a slot when they have observed
//!   `Clean` via an Acquire load on the node's state, and writers only
//!   touch a slot when they own `Computing` state. The arena does not
//!   enforce this at the type level; the runtime is responsible for
//!   the invariant.
//!
//! ## Why `Option<T>` instead of the spec's `MaybeUninit<T>`
//!
//! Section 5.2 of the spec presents `UnsafeCell<MaybeUninit<T>>` as a
//! sketch. `Option<T>` is a cleaner equivalent that preserves the spec's
//! intent while avoiding two costs that `MaybeUninit` would impose:
//!
//! 1. **Drop correctness comes for free.** `Option::drop` destructs the
//!    contained `T` if `Some` and does nothing if `None`. With
//!    `MaybeUninit` the arena would have to track per-slot initialization
//!    (via an extra `AtomicBool` or a bitmap) just to know what to drop
//!    when the arena itself is dropped, because the node state machine
//!    is not available at `Drop` time.
//! 2. **Query nodes can defer initialization.** A query slot is allocated
//!    before the first compute runs and starts as `None`. With
//!    `MaybeUninit` the caller would have to provide a placeholder `T`
//!    at reserve time, which would force `T: Default` on the user-facing
//!    query API.
//!
//! The cost is one discriminant byte per slot (often niche-optimized
//! away for `Box`, `Vec`, `String`, `&T`, etc.) and one predictable
//! branch on the read path. Negligible.
//!
//! ## Segmented storage and lock-free growth
//!
//! Both arena kinds use a segmented storage layout to allow lock-free reads
//! while still supporting dynamic growth. A fixed-size top-level array of
//! `AtomicPtr<Segment>` indexes segments, and segments are allocated on
//! demand and never deallocated for the arena's lifetime. Readers access
//! a slot via two atomic loads: Acquire-load the segment pointer, then
//! Relaxed-access the slot within the segment. Growth never moves existing
//! slots, so concurrent readers never observe dangling references.
//!
//! Per-arena capacity is [`MAX_SLOTS`] = `MAX_SEGMENTS * SEGMENT_SIZE` (1M
//! slots at the current sizing). Exhaustion panics loudly at the reserve
//! call so we find out early. The numbers can be tuned after real workloads.
//!
//! ## Who calls what
//!
//! Arenas do not own the concurrency policy. The runtime is the authority
//! on when a slot may be written and when a value is safe to read. Concretely:
//!
//! - `reserve` may be called concurrently by multiple threads; the
//!   implementation handles the segment-allocation race with a CAS. In
//!   practice the runtime serializes node creation through its write
//!   mutex, but the arena does not rely on that.
//! - `write` for a query node's value is called by the thread that owns
//!   the node's `Computing` state (guaranteed by the state machine CAS).
//!   It uses `Relaxed` ordering because the accompanying state transition
//!   to `Clean` is the `Release` publish point.
//! - `write` for an input node's value is called under the runtime's write
//!   mutex. The runtime is responsible for issuing a `Release` store on
//!   the input node's state after calling `arena.write`, so that subsequent
//!   readers that Acquire-load the input's state observe the updated slot.
//!   See spec section 6.4; this publish step is implied by the memory
//!   ordering contract even though the spec's pseudocode does not spell
//!   it out for the input node itself (only for its dependents).
//! - `read` uses `Relaxed` ordering. The caller must have established
//!   happens-before with the writer via an Acquire load on the node's state
//!   (or an equivalent synchronization point) before calling `read`.

use std::any::TypeId;
use std::sync::atomic::{
    AtomicBool, AtomicI32, AtomicI64, AtomicPtr, AtomicU32, AtomicU64, Ordering,
};

/// Top-level segment count. Combined with [`SEGMENT_SIZE`] this fixes the
/// maximum number of slots per arena instance.
const MAX_SEGMENTS: usize = 1024;

/// Slots per segment. Power of two so that slot-to-segment math is a shift
/// and a mask.
const SEGMENT_SIZE: usize = 1024;
const SEGMENT_SHIFT: u32 = 10;
const SEGMENT_MASK: u32 = (SEGMENT_SIZE as u32) - 1;

/// Maximum number of slots that a single arena can hold. Reserving beyond
/// this panics. At the current sizing this is one million slots per value
/// type, which covers realistic workloads by a wide margin.
pub(crate) const MAX_SLOTS: u32 = (MAX_SEGMENTS * SEGMENT_SIZE) as u32;

const _: () = assert!(SEGMENT_SIZE.is_power_of_two());
const _: () = assert!(1 << SEGMENT_SHIFT == SEGMENT_SIZE);

/// Type-erased arena trait stored in the runtime's arena registry.
///
/// The runtime holds `Box<dyn ErasedArena>` keyed by `TypeId`, and downcasts
/// to the concrete arena type at each `get::<T>` call site, which carries
/// the type parameter statically. The trait surface is intentionally
/// minimal: type identification only. Concrete operations (reserve, read,
/// write) live on the concrete arena types and are reached via downcast.
pub(crate) trait ErasedArena: Send + Sync {
    /// Returns the `TypeId` of the value type this arena holds.
    fn erased_type_id(&self) -> TypeId;

    /// Upcast helper so the registry can downcast through `Any`-like
    /// machinery without pulling in `std::any::Any` directly on every
    /// concrete arena.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// A `Copy` primitive type that can be stored tear-free in an atomic cell.
///
/// Implemented for the fixed set of primitive types below. The trait is
/// crate-private so adding a variant is a deliberate act requiring choice
/// of backing atomic and a tear-free read justification.
///
/// Floats are stored in their bit-pattern-equivalent integer atomic
/// (`AtomicU32` for `f32`, `AtomicU64` for `f64`) via `to_bits` / `from_bits`.
/// This is sound because `f32::to_bits` and `f64::to_bits` are pure
/// reinterpret-casts and `from_bits` accepts every bit pattern (including
/// NaN payloads).
pub(crate) trait AtomicPrimitive:
    Copy + PartialEq + std::fmt::Debug + Send + Sync + 'static
{
    /// The atomic cell type used to store a value of this primitive.
    type Atomic: Send + Sync;

    /// A well-defined zero value used to initialize fresh segment slots
    /// before the first real reservation touches them.
    fn zero() -> Self;

    /// Construct a new atomic cell holding `value`.
    fn new_atomic(value: Self) -> Self::Atomic;

    /// Relaxed load of the current value. Tear-free by construction.
    fn load(atomic: &Self::Atomic) -> Self;

    /// Relaxed store. Caller is responsible for the surrounding Release
    /// publish (on state or on another field) that makes the new value
    /// visible to readers who need a happens-before guarantee.
    fn store(atomic: &Self::Atomic, value: Self);
}

macro_rules! impl_atomic_primitive_int {
    ($t:ty, $atomic:ty, $zero:expr) => {
        impl AtomicPrimitive for $t {
            type Atomic = $atomic;
            #[inline]
            fn zero() -> Self {
                $zero
            }
            #[inline]
            fn new_atomic(value: Self) -> Self::Atomic {
                <$atomic>::new(value)
            }
            #[inline]
            fn load(atomic: &Self::Atomic) -> Self {
                atomic.load(Ordering::Relaxed)
            }
            #[inline]
            fn store(atomic: &Self::Atomic, value: Self) {
                atomic.store(value, Ordering::Relaxed);
            }
        }
    };
}

impl_atomic_primitive_int!(u32, AtomicU32, 0);
impl_atomic_primitive_int!(i32, AtomicI32, 0);
impl_atomic_primitive_int!(u64, AtomicU64, 0);
impl_atomic_primitive_int!(i64, AtomicI64, 0);

impl AtomicPrimitive for bool {
    type Atomic = AtomicBool;
    #[inline]
    fn zero() -> Self {
        false
    }
    #[inline]
    fn new_atomic(value: Self) -> Self::Atomic {
        AtomicBool::new(value)
    }
    #[inline]
    fn load(atomic: &Self::Atomic) -> Self {
        atomic.load(Ordering::Relaxed)
    }
    #[inline]
    fn store(atomic: &Self::Atomic, value: Self) {
        atomic.store(value, Ordering::Relaxed);
    }
}

impl AtomicPrimitive for f32 {
    type Atomic = AtomicU32;
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn new_atomic(value: Self) -> Self::Atomic {
        AtomicU32::new(value.to_bits())
    }
    #[inline]
    fn load(atomic: &Self::Atomic) -> Self {
        f32::from_bits(atomic.load(Ordering::Relaxed))
    }
    #[inline]
    fn store(atomic: &Self::Atomic, value: Self) {
        atomic.store(value.to_bits(), Ordering::Relaxed);
    }
}

impl AtomicPrimitive for f64 {
    type Atomic = AtomicU64;
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn new_atomic(value: Self) -> Self::Atomic {
        AtomicU64::new(value.to_bits())
    }
    #[inline]
    fn load(atomic: &Self::Atomic) -> Self {
        f64::from_bits(atomic.load(Ordering::Relaxed))
    }
    #[inline]
    fn store(atomic: &Self::Atomic, value: Self) {
        atomic.store(value.to_bits(), Ordering::Relaxed);
    }
}

/// One contiguous block of atomic slots. Segments are heap-allocated once
/// and never moved or freed until the arena is dropped.
struct Segment<A> {
    slots: Box<[A]>,
}

/// Arena for `Copy` primitive values stored inline as atomic cells.
///
/// See the module-level docs for the overall design. Tear-free reads are
/// guaranteed by the atomic cell choice; staleness is the state machine's
/// concern and is not enforced here.
pub(crate) struct AtomicPrimitiveArena<T: AtomicPrimitive> {
    /// Top-level segment directory. All entries start null; segments are
    /// allocated on demand by `reserve`. Size is fixed at construction
    /// and never reallocated, so readers may index safely without locking.
    segments: Box<[AtomicPtr<Segment<T::Atomic>>]>,

    /// Number of logically-reserved slots. Monotonically increasing via
    /// `fetch_add`, used to hand out slot indices and to bound-check
    /// against [`MAX_SLOTS`].
    len: AtomicU32,
}

impl<T: AtomicPrimitive> AtomicPrimitiveArena<T> {
    /// Construct an empty arena. Segments are not allocated until the
    /// first reservation lands in each one.
    pub(crate) fn new() -> Self {
        let segments = (0..MAX_SEGMENTS)
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            segments,
            len: AtomicU32::new(0),
        }
    }

    /// Reserve a new slot holding `initial`, returning its index.
    ///
    /// Atomically increments `len` to claim an index, allocates the
    /// containing segment if it is not yet present (racing safely with
    /// any other concurrent reservation into the same segment), and
    /// stores `initial` into the claimed slot with `Relaxed` ordering.
    ///
    /// Panics if the arena has already exhausted [`MAX_SLOTS`].
    pub(crate) fn reserve(&self, initial: T) -> u32 {
        let idx = self.len.fetch_add(1, Ordering::Relaxed);
        if idx >= MAX_SLOTS {
            // Undo the over-increment so repeated panics don't leave
            // `len` unboundedly large in case the caller catches the panic.
            self.len.fetch_sub(1, Ordering::Relaxed);
            panic!(
                "AtomicPrimitiveArena<{}> exhausted at {} slots",
                std::any::type_name::<T>(),
                MAX_SLOTS
            );
        }
        let seg_idx = (idx >> SEGMENT_SHIFT) as usize;
        let within = (idx & SEGMENT_MASK) as usize;
        let segment = self.get_or_allocate_segment(seg_idx);
        // SAFETY: `segment` is a non-null pointer to a Segment owned by
        // this arena. Segments are never freed until `Drop`, and the
        // slot index `within` is in `0..SEGMENT_SIZE` by construction.
        unsafe {
            T::store(&(*segment).slots[within], initial);
        }
        idx
    }

    /// Read the value at `slot` with `Relaxed` ordering.
    ///
    /// Tear-free. The caller is responsible for establishing happens-before
    /// with the writer via an Acquire load on the owning node's state (or
    /// equivalent). Reading an unreserved slot is a logic error; in debug
    /// builds this is caught by the `debug_assert`, in release builds it
    /// returns the slot's zero-initialized value (no undefined behavior).
    #[inline]
    pub(crate) fn read(&self, slot: u32) -> T {
        debug_assert!(
            slot < self.len.load(Ordering::Relaxed),
            "read of unreserved slot {} (len {})",
            slot,
            self.len.load(Ordering::Relaxed)
        );
        let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
        let within = (slot & SEGMENT_MASK) as usize;
        let seg_ptr = self.segments[seg_idx].load(Ordering::Acquire);
        debug_assert!(
            !seg_ptr.is_null(),
            "read of slot {} in unallocated segment {}",
            slot,
            seg_idx
        );
        // SAFETY: `seg_ptr` was published via `Release` by `reserve`, is
        // non-null for any reserved slot, points to a Segment owned by
        // this arena that lives until Drop, and `within` is in range.
        unsafe { T::load(&(*seg_ptr).slots[within]) }
    }

    /// Write `value` to `slot` with `Relaxed` ordering.
    ///
    /// Used during compute completion (under Computing-state ownership)
    /// or for input updates (under the runtime's write mutex, with a
    /// Release publish on the node's state afterwards). The arena does
    /// not enforce exclusivity; it is the caller's responsibility to
    /// avoid conflicting concurrent writes to the same slot.
    #[inline]
    pub(crate) fn write(&self, slot: u32, value: T) {
        debug_assert!(
            slot < self.len.load(Ordering::Relaxed),
            "write to unreserved slot {} (len {})",
            slot,
            self.len.load(Ordering::Relaxed)
        );
        let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
        let within = (slot & SEGMENT_MASK) as usize;
        let seg_ptr = self.segments[seg_idx].load(Ordering::Acquire);
        debug_assert!(
            !seg_ptr.is_null(),
            "write to slot {} in unallocated segment {}",
            slot,
            seg_idx
        );
        // SAFETY: same as `read`.
        unsafe {
            T::store(&(*seg_ptr).slots[within], value);
        }
    }

    /// Current number of reserved slots.
    #[cfg(test)]
    pub(crate) fn len(&self) -> u32 {
        self.len.load(Ordering::Relaxed)
    }

    /// Return a pointer to the segment at `seg_idx`, allocating it if
    /// necessary. Safe to call concurrently: at most one allocation wins
    /// the compare-exchange, and losers drop their speculative allocation
    /// and use the winner's pointer.
    fn get_or_allocate_segment(&self, seg_idx: usize) -> *const Segment<T::Atomic> {
        let existing = self.segments[seg_idx].load(Ordering::Acquire);
        if !existing.is_null() {
            return existing;
        }
        let slots: Vec<T::Atomic> = (0..SEGMENT_SIZE)
            .map(|_| T::new_atomic(T::zero()))
            .collect();
        let segment = Box::new(Segment {
            slots: slots.into_boxed_slice(),
        });
        let ptr = Box::into_raw(segment);
        match self.segments[seg_idx].compare_exchange(
            std::ptr::null_mut(),
            ptr,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            Ok(_) => ptr,
            Err(winner) => {
                // Another thread already published a segment here.
                // Drop our speculative allocation and use theirs.
                // SAFETY: `ptr` came from `Box::into_raw` in this call
                // and was never published anywhere else; we are the sole
                // owner and can reclaim it.
                unsafe {
                    drop(Box::from_raw(ptr));
                }
                winner
            }
        }
    }
}

impl<T: AtomicPrimitive> Drop for AtomicPrimitiveArena<T> {
    fn drop(&mut self) {
        // Reclaim every segment that was allocated via `Box::into_raw`.
        for entry in self.segments.iter() {
            let ptr = entry.load(Ordering::Acquire);
            if !ptr.is_null() {
                // SAFETY: the pointer came from `Box::into_raw` in
                // `get_or_allocate_segment` and is uniquely owned by this
                // arena. `&mut self` guarantees no concurrent access.
                unsafe {
                    drop(Box::from_raw(ptr));
                }
            }
        }
    }
}

impl<T: AtomicPrimitive> ErasedArena for AtomicPrimitiveArena<T> {
    fn erased_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// GenericArena: non-primitive values coordinated by the state machine.
// ---------------------------------------------------------------------------

use std::cell::UnsafeCell;

/// One slot in a `GenericArena`. The inner `Option<T>` starts `None` at
/// segment allocation and becomes `Some(value)` once the arena's user
/// writes a value into it. Interior mutability is granted by `UnsafeCell`
/// and exclusivity is enforced at runtime by the node state machine: at
/// any instant, a slot is being read by zero-or-more threads that have
/// observed `Clean` via Acquire, or being written by exactly one thread
/// that owns `Computing` via CAS, never both.
struct GenericSlot<T> {
    cell: UnsafeCell<Option<T>>,
}

impl<T> GenericSlot<T> {
    fn none() -> Self {
        Self {
            cell: UnsafeCell::new(None),
        }
    }
}

// SAFETY: `GenericSlot<T>` wraps an `UnsafeCell<Option<T>>`, which is
// `!Sync` by default. We share slots across threads behind the invariant
// that the node state machine linearizes access: a slot is either being
// read (zero-or-more Acquire observers of `Clean`) or written (exactly
// one `Computing` owner), never both. Under that invariant, concurrent
// access to the underlying `Option<T>` bytes is coordinated by
// Release-Acquire synchronization on the node's state field, which
// establishes happens-before for the slot's plain memory operations.
// This is the same coordination the spec describes for atomic slots;
// here we apply it to non-atomic memory.
//
// `T: Send + Sync` is required because (a) a reader on thread A may
// call `Clone` on the `T`, which briefly creates an `&T` on thread A,
// requiring `T: Sync`, and (b) the cloned value may be moved across
// threads by the caller, requiring `T: Send`.
unsafe impl<T: Send + Sync> Sync for GenericSlot<T> {}
unsafe impl<T: Send> Send for GenericSlot<T> {}

/// Arena for `Clone + Send + Sync` values whose access is coordinated by
/// the node state machine.
///
/// Layout and growth mirror [`AtomicPrimitiveArena`]: a fixed-size
/// top-level directory of `AtomicPtr<Segment<GenericSlot<T>>>` with lazy
/// segment allocation via CAS. Only the slot body differs: generic values
/// live in `UnsafeCell<Option<T>>` cells, and access safety depends on
/// external coordination rather than atomicity.
pub(crate) struct GenericArena<T: Clone + Send + Sync + 'static> {
    segments: Box<[AtomicPtr<Segment<GenericSlot<T>>>]>,
    len: AtomicU32,
}

impl<T: Clone + Send + Sync + 'static> GenericArena<T> {
    /// Construct an empty arena.
    pub(crate) fn new() -> Self {
        let segments = (0..MAX_SEGMENTS)
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            segments,
            len: AtomicU32::new(0),
        }
    }

    /// Reserve a new slot, leaving it `None` (uninitialized).
    ///
    /// The caller is expected to follow up with [`GenericArena::write`]
    /// once they have a value to store. For query nodes the first write
    /// happens at the end of the first compute; for input nodes it
    /// happens immediately after reservation (and both happen under the
    /// runtime's write mutex, so there is no window where another thread
    /// could observe the handle before the slot is populated).
    ///
    /// Panics on exhaustion, same contract as `AtomicPrimitiveArena`.
    pub(crate) fn reserve(&self) -> u32 {
        let idx = self.len.fetch_add(1, Ordering::Relaxed);
        if idx >= MAX_SLOTS {
            self.len.fetch_sub(1, Ordering::Relaxed);
            panic!(
                "GenericArena<{}> exhausted at {} slots",
                std::any::type_name::<T>(),
                MAX_SLOTS
            );
        }
        let seg_idx = (idx >> SEGMENT_SHIFT) as usize;
        // Ensure segment exists. We don't need to touch the slot here;
        // segments are allocated with all slots `None`.
        let _ = self.get_or_allocate_segment(seg_idx);
        idx
    }

    /// Convenience: reserve and immediately initialize to `Some(initial)`.
    ///
    /// Used by the runtime's `create_input` path where the initial value
    /// is known at node creation time. Equivalent to `reserve()` followed
    /// by `write(slot, initial)`, but saves one segment lookup.
    pub(crate) fn reserve_with(&self, initial: T) -> u32 {
        let idx = self.reserve();
        self.write(idx, initial);
        idx
    }

    /// Clone the value at `slot`.
    ///
    /// The caller must have established happens-before with the most
    /// recent writer via an Acquire load on the owning node's state;
    /// typically the caller observed the node in state `Clean`. Reading
    /// a slot whose state is `New` (still `None` in the Option) is a
    /// logic error and panics with a clear message; in release builds
    /// this may still panic via `Option::expect`.
    pub(crate) fn read(&self, slot: u32) -> T {
        debug_assert!(
            slot < self.len.load(Ordering::Relaxed),
            "read of unreserved slot {} (len {})",
            slot,
            self.len.load(Ordering::Relaxed)
        );
        let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
        let within = (slot & SEGMENT_MASK) as usize;
        let seg_ptr = self.segments[seg_idx].load(Ordering::Acquire);
        debug_assert!(
            !seg_ptr.is_null(),
            "read of slot {} in unallocated segment {}",
            slot,
            seg_idx
        );
        // SAFETY: `seg_ptr` was published via `Release` by a prior
        // `reserve`; segments are never moved or freed until Drop; and
        // `within` is in range. The read of the `Option<T>` is
        // non-atomic but is synchronized with the writer via the
        // caller's Acquire load on the node state, which happens-after
        // the writer's Release store on the same state field. Under
        // that happens-before the plain memory of the slot is visible
        // consistently. `Clone::clone` may panic; that is fine, the
        // slot is unchanged.
        unsafe {
            let slot_ref = &(*seg_ptr).slots[within];
            (*slot_ref.cell.get())
                .as_ref()
                .expect("GenericArena::read on uninitialized slot; caller must check state first")
                .clone()
        }
    }

    /// Clone the value at `slot` if it exists, returning `None` if the
    /// slot has never been written (still `Option::None` inside the
    /// cell). Used by callers that cannot guarantee the slot has been
    /// initialized, such as the Runtime's early-cutoff path during a
    /// retry-from-Failed recompute where the first compute panicked
    /// before the arena was written.
    ///
    /// The same happens-before requirements as `read` apply: the
    /// caller must have established synchronization with the most
    /// recent writer via an Acquire load on the owning node's state.
    pub(crate) fn try_read(&self, slot: u32) -> Option<T> {
        if slot >= self.len.load(Ordering::Relaxed) {
            return None;
        }
        let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
        let within = (slot & SEGMENT_MASK) as usize;
        let seg_ptr = self.segments[seg_idx].load(Ordering::Acquire);
        if seg_ptr.is_null() {
            return None;
        }
        // SAFETY: same as `read`, plus we now also check for the
        // `None` variant of the inner `Option` and return `None`
        // from this method in that case rather than panicking.
        unsafe {
            let slot_ref = &(*seg_ptr).slots[within];
            (*slot_ref.cell.get()).as_ref().cloned()
        }
    }

    /// Overwrite the value at `slot` with `Some(value)`.
    ///
    /// If the slot previously held `Some(old)`, the old value is dropped
    /// in place. This is called exclusively by the writer that owns the
    /// slot (the compute thread holding Computing state for a query
    /// node, or the runtime under the write mutex for an input node).
    /// The arena does not enforce exclusivity; the state machine does.
    pub(crate) fn write(&self, slot: u32, value: T) {
        debug_assert!(
            slot < self.len.load(Ordering::Relaxed),
            "write to unreserved slot {} (len {})",
            slot,
            self.len.load(Ordering::Relaxed)
        );
        let seg_idx = (slot >> SEGMENT_SHIFT) as usize;
        let within = (slot & SEGMENT_MASK) as usize;
        let seg_ptr = self.segments[seg_idx].load(Ordering::Acquire);
        debug_assert!(
            !seg_ptr.is_null(),
            "write to slot {} in unallocated segment {}",
            slot,
            seg_idx
        );
        // SAFETY: the state machine invariant guarantees no concurrent
        // readers or writers are touching this slot. `UnsafeCell::get`
        // yields a raw pointer we may write through without creating a
        // Rust &mut reference, so no aliasing rule is violated even if
        // compiler-inserted reads existed briefly. Dropping the old
        // `Option` (possibly a `Some(T)`) runs `T::drop` if a value was
        // present; the state machine ensures no reader has an `&T` into
        // the slot at this moment.
        unsafe {
            let slot_ref = &(*seg_ptr).slots[within];
            *slot_ref.cell.get() = Some(value);
        }
    }

    /// Current number of reserved slots.
    #[cfg(test)]
    pub(crate) fn len(&self) -> u32 {
        self.len.load(Ordering::Relaxed)
    }

    fn get_or_allocate_segment(&self, seg_idx: usize) -> *const Segment<GenericSlot<T>> {
        let existing = self.segments[seg_idx].load(Ordering::Acquire);
        if !existing.is_null() {
            return existing;
        }
        let slots: Vec<GenericSlot<T>> = (0..SEGMENT_SIZE).map(|_| GenericSlot::none()).collect();
        let segment = Box::new(Segment {
            slots: slots.into_boxed_slice(),
        });
        let ptr = Box::into_raw(segment);
        match self.segments[seg_idx].compare_exchange(
            std::ptr::null_mut(),
            ptr,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            Ok(_) => ptr,
            Err(winner) => {
                // SAFETY: `ptr` came from `Box::into_raw` above and was
                // never published anywhere else; we own it.
                unsafe {
                    drop(Box::from_raw(ptr));
                }
                winner
            }
        }
    }
}

impl<T: Clone + Send + Sync + 'static> Drop for GenericArena<T> {
    fn drop(&mut self) {
        // Reclaim every allocated segment. Each segment's `Drop` in
        // turn drops its `Box<[GenericSlot<T>]>`, which drops each
        // `GenericSlot<T>`, which drops the inner `UnsafeCell<Option<T>>`,
        // which drops the `Option<T>` (calling `T::drop` if `Some`).
        // This is why we do not need a separate "which slots are
        // initialized" tracker: the `Option` discriminant is the tracker.
        for entry in self.segments.iter() {
            let ptr = entry.load(Ordering::Acquire);
            if !ptr.is_null() {
                // SAFETY: pointer came from `Box::into_raw` in
                // `get_or_allocate_segment`; uniquely owned by this
                // arena; `&mut self` guarantees no concurrent access.
                unsafe {
                    drop(Box::from_raw(ptr));
                }
            }
        }
    }
}

impl<T: Clone + Send + Sync + 'static> ErasedArena for GenericArena<T> {
    fn erased_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn reserve_then_read_u64() {
        let arena: AtomicPrimitiveArena<u64> = AtomicPrimitiveArena::new();
        let slot = arena.reserve(42);
        assert_eq!(slot, 0);
        assert_eq!(arena.read(slot), 42);
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn reserve_hands_out_sequential_indices() {
        let arena: AtomicPrimitiveArena<u64> = AtomicPrimitiveArena::new();
        for i in 0..100 {
            let slot = arena.reserve(i);
            assert_eq!(slot, i as u32);
            assert_eq!(arena.read(slot), i);
        }
        assert_eq!(arena.len(), 100);
    }

    #[test]
    fn write_overwrites_existing_value() {
        let arena: AtomicPrimitiveArena<u64> = AtomicPrimitiveArena::new();
        let slot = arena.reserve(10);
        arena.write(slot, 20);
        assert_eq!(arena.read(slot), 20);
        arena.write(slot, 30);
        assert_eq!(arena.read(slot), 30);
    }

    #[test]
    fn reservations_span_segment_boundary() {
        // Force crossing from segment 0 into segment 1 and back into
        // segment 1 several times. Verifies both segment-allocation and
        // slot addressing across the boundary.
        let arena: AtomicPrimitiveArena<u64> = AtomicPrimitiveArena::new();
        let count = (SEGMENT_SIZE as u64) + 50;
        let mut slots = Vec::with_capacity(count as usize);
        for i in 0..count {
            slots.push(arena.reserve(i * 7 + 1));
        }
        for (i, slot) in slots.into_iter().enumerate() {
            assert_eq!(arena.read(slot), (i as u64) * 7 + 1);
        }
        assert_eq!(arena.len(), count as u32);
    }

    #[test]
    fn bool_arena_read_write() {
        let arena: AtomicPrimitiveArena<bool> = AtomicPrimitiveArena::new();
        let a = arena.reserve(true);
        let b = arena.reserve(false);
        assert!(arena.read(a));
        assert!(!arena.read(b));
        arena.write(a, false);
        arena.write(b, true);
        assert!(!arena.read(a));
        assert!(arena.read(b));
    }

    #[test]
    fn f64_arena_round_trips_bits() {
        let arena: AtomicPrimitiveArena<f64> = AtomicPrimitiveArena::new();
        let values = [
            0.0,
            -0.0,
            1.5,
            -1.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN,
            f64::MAX,
            std::f64::consts::PI,
        ];
        let slots: Vec<u32> = values.iter().map(|&v| arena.reserve(v)).collect();
        for (slot, expected) in slots.iter().zip(values.iter()) {
            assert_eq!(arena.read(*slot).to_bits(), expected.to_bits());
        }
        // NaN payloads round trip too.
        let nan = f64::from_bits(0x7ff8_dead_beef_cafe);
        let slot = arena.reserve(nan);
        assert_eq!(arena.read(slot).to_bits(), 0x7ff8_dead_beef_cafe);
    }

    #[test]
    fn f32_arena_round_trips_bits() {
        let arena: AtomicPrimitiveArena<f32> = AtomicPrimitiveArena::new();
        let slot = arena.reserve(std::f32::consts::E);
        assert_eq!(arena.read(slot).to_bits(), std::f32::consts::E.to_bits());
    }

    #[test]
    fn erased_type_id_matches() {
        let u64_arena: AtomicPrimitiveArena<u64> = AtomicPrimitiveArena::new();
        let bool_arena: AtomicPrimitiveArena<bool> = AtomicPrimitiveArena::new();
        assert_eq!(u64_arena.erased_type_id(), TypeId::of::<u64>());
        assert_eq!(bool_arena.erased_type_id(), TypeId::of::<bool>());
        assert_ne!(u64_arena.erased_type_id(), bool_arena.erased_type_id());
    }

    #[test]
    fn erased_as_any_downcasts_to_concrete_type() {
        let arena: Box<dyn ErasedArena> = Box::new(AtomicPrimitiveArena::<u64>::new());
        let concrete = arena
            .as_any()
            .downcast_ref::<AtomicPrimitiveArena<u64>>()
            .expect("downcast to concrete type");
        let slot = concrete.reserve(99);
        assert_eq!(concrete.read(slot), 99);
    }

    #[test]
    fn concurrent_reservers_never_produce_duplicate_indices() {
        // Many threads race to reserve. Each thread records its returned
        // indices. At the end, every index in [0..total) must appear
        // exactly once across all threads. This exercises the segment
        // allocation CAS.
        const THREADS: usize = 16;
        const PER_THREAD: usize = 2000;
        let arena: Arc<AtomicPrimitiveArena<u64>> = Arc::new(AtomicPrimitiveArena::new());
        let handles: Vec<_> = (0..THREADS)
            .map(|tid| {
                let arena = arena.clone();
                thread::spawn(move || {
                    let mut mine = Vec::with_capacity(PER_THREAD);
                    for i in 0..PER_THREAD {
                        // Value encodes (tid, i) so we can verify later.
                        let v = (tid as u64) * 1_000_000 + i as u64;
                        let slot = arena.reserve(v);
                        mine.push((slot, v));
                    }
                    mine
                })
            })
            .collect();

        let mut all: Vec<(u32, u64)> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        all.sort_by_key(|(slot, _)| *slot);

        assert_eq!(all.len(), THREADS * PER_THREAD);
        for (i, (slot, v)) in all.iter().enumerate() {
            assert_eq!(*slot as usize, i, "slot indices must be a dense range");
            assert_eq!(
                arena.read(*slot),
                *v,
                "every slot must hold the value its reserver wrote"
            );
        }
    }

    #[test]
    fn concurrent_reader_sees_tear_free_values() {
        // One writer rewrites a single slot with alternating bit patterns.
        // Many readers spin-read the slot and assert they always observe
        // one of the two known patterns (never a torn combination).
        // This is a smoke test for the u64 atomic-store path; atomicity
        // is guaranteed by `AtomicU64` on all supported targets.
        const READERS: usize = 8;
        const ITERS: usize = 50_000;
        const PATTERN_A: u64 = 0x0000_0000_DEAD_BEEF;
        const PATTERN_B: u64 = 0xCAFE_BABE_0000_0000;

        let arena: Arc<AtomicPrimitiveArena<u64>> = Arc::new(AtomicPrimitiveArena::new());
        let slot = arena.reserve(PATTERN_A);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let reader_handles: Vec<_> = (0..READERS)
            .map(|_| {
                let arena = arena.clone();
                let stop = stop.clone();
                thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let v = arena.read(slot);
                        assert!(
                            v == PATTERN_A || v == PATTERN_B,
                            "observed torn value 0x{:016x}",
                            v
                        );
                    }
                })
            })
            .collect();

        // Writer toggles the value ITERS times.
        for i in 0..ITERS {
            let v = if i & 1 == 0 { PATTERN_A } else { PATTERN_B };
            arena.write(slot, v);
        }
        stop.store(true, Ordering::Relaxed);
        for h in reader_handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn segments_are_reclaimed_on_drop() {
        // Exercise Drop by allocating across several segments and letting
        // the arena fall out of scope. Without Drop, the segment boxes
        // would leak; we rely on Miri (or valgrind) to catch that.
        // Here we at least exercise the code path.
        let arena: AtomicPrimitiveArena<u64> = AtomicPrimitiveArena::new();
        for i in 0..(SEGMENT_SIZE * 3 + 7) {
            arena.reserve(i as u64);
        }
        drop(arena);
    }

    // -----------------------------------------------------------------
    // GenericArena tests
    // -----------------------------------------------------------------

    #[test]
    fn generic_reserve_starts_none_and_write_populates() {
        let arena: GenericArena<String> = GenericArena::new();
        let slot = arena.reserve();
        assert_eq!(slot, 0);
        arena.write(slot, "hello".to_string());
        assert_eq!(arena.read(slot), "hello");
    }

    #[test]
    fn generic_reserve_with_initializes_immediately() {
        let arena: GenericArena<String> = GenericArena::new();
        let slot = arena.reserve_with("world".to_string());
        assert_eq!(arena.read(slot), "world");
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn generic_write_overwrites_and_drops_old_value() {
        // Use a struct that tracks its drops, so we can verify that
        // overwriting a Some(old) with Some(new) runs old's destructor.
        use std::sync::atomic::AtomicUsize;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        #[derive(Clone)]
        struct DropCounter(#[allow(dead_code)] u64);
        impl Drop for DropCounter {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROPS.store(0, Ordering::SeqCst);
        let arena: GenericArena<DropCounter> = GenericArena::new();
        let slot = arena.reserve_with(DropCounter(1));
        // One DropCounter briefly existed as the argument to reserve_with,
        // but the value is moved into the slot so its destructor is not
        // called yet.
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);

        arena.write(slot, DropCounter(2));
        // The write replaced Some(DropCounter(1)) with Some(DropCounter(2));
        // the old DropCounter(1) was dropped as part of the Option
        // assignment.
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);

        arena.write(slot, DropCounter(3));
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);

        drop(arena);
        // The arena's Drop runs DropCounter(3).
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn generic_reservations_span_segment_boundary() {
        let arena: GenericArena<Vec<u32>> = GenericArena::new();
        let count = SEGMENT_SIZE + 25;
        let mut slots = Vec::with_capacity(count);
        for i in 0..count {
            slots.push(arena.reserve_with(vec![i as u32, (i as u32) * 2]));
        }
        for (i, slot) in slots.into_iter().enumerate() {
            let v = arena.read(slot);
            assert_eq!(v, vec![i as u32, (i as u32) * 2]);
        }
    }

    #[test]
    fn generic_read_clones_independent_copies() {
        // Mutating the cloned value must not affect the arena's copy.
        let arena: GenericArena<Vec<u32>> = GenericArena::new();
        let slot = arena.reserve_with(vec![1, 2, 3]);
        let mut clone_a = arena.read(slot);
        clone_a.push(999);
        let clone_b = arena.read(slot);
        assert_eq!(clone_b, vec![1, 2, 3]);
        assert_eq!(clone_a, vec![1, 2, 3, 999]);
    }

    #[test]
    fn generic_arena_drops_all_initialized_slots() {
        // Cross several segments with values that track Drop, then let
        // the arena fall out of scope and verify every value was dropped
        // exactly once.
        use std::sync::atomic::AtomicUsize;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        #[derive(Clone)]
        struct Tracked;
        impl Drop for Tracked {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROPS.store(0, Ordering::SeqCst);
        let arena: GenericArena<Tracked> = GenericArena::new();
        let populated = SEGMENT_SIZE + 10;
        for _ in 0..populated {
            arena.reserve_with(Tracked);
        }
        // Also reserve some slots that remain uninitialized (None).
        // Their Drop should not count, because Option::None has no T.
        for _ in 0..7 {
            arena.reserve();
        }
        drop(arena);
        assert_eq!(DROPS.load(Ordering::SeqCst), populated);
    }

    #[test]
    fn generic_erased_type_id_and_downcast() {
        let arena: Box<dyn ErasedArena> = Box::new(GenericArena::<String>::new());
        assert_eq!(arena.erased_type_id(), TypeId::of::<String>());
        let concrete = arena
            .as_any()
            .downcast_ref::<GenericArena<String>>()
            .expect("downcast to GenericArena<String>");
        let slot = concrete.reserve_with("downcast".to_string());
        assert_eq!(concrete.read(slot), "downcast");
    }

    #[test]
    fn generic_concurrent_readers_see_stable_value() {
        // Many readers clone the same slot in parallel while we hold
        // the value constant. This exercises the Sync impl and the
        // read path's non-atomic access under a shared reference.
        // Correctness for concurrent read-and-write is the state
        // machine's job, not the arena's; here we only verify that
        // concurrent reads of a stable value work.
        const READERS: usize = 16;
        const ITERS: usize = 10_000;

        let arena: Arc<GenericArena<String>> = Arc::new(GenericArena::new());
        let slot = arena.reserve_with("stable".to_string());

        let handles: Vec<_> = (0..READERS)
            .map(|_| {
                let arena = arena.clone();
                thread::spawn(move || {
                    for _ in 0..ITERS {
                        assert_eq!(arena.read(slot), "stable");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn generic_concurrent_reservers_get_unique_indices() {
        // Same invariant as the primitive arena: concurrent reserve()
        // hands out a dense range of unique indices. This exercises
        // the segment-allocation CAS on the GenericArena path.
        const THREADS: usize = 16;
        const PER_THREAD: usize = 1500;

        let arena: Arc<GenericArena<u64>> = Arc::new(GenericArena::new());
        let handles: Vec<_> = (0..THREADS)
            .map(|tid| {
                let arena = arena.clone();
                thread::spawn(move || {
                    let mut mine = Vec::with_capacity(PER_THREAD);
                    for i in 0..PER_THREAD {
                        let v = (tid as u64) * 1_000_000 + i as u64;
                        let slot = arena.reserve_with(v);
                        mine.push((slot, v));
                    }
                    mine
                })
            })
            .collect();

        let mut all: Vec<(u32, u64)> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        all.sort_by_key(|(slot, _)| *slot);

        assert_eq!(all.len(), THREADS * PER_THREAD);
        for (i, (slot, v)) in all.iter().enumerate() {
            assert_eq!(*slot as usize, i);
            assert_eq!(arena.read(*slot), *v);
        }
    }
}
