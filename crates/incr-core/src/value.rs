//! Type dispatch for the arena hierarchy.
//!
//! `Value` is the crate-private trait that decides which concrete
//! arena a type lives in. Primitive types (u32, i32, u64, i64, f32,
//! f64, bool) route to [`AtomicPrimitiveArena`] where reads are a
//! single atomic load and writes are tear-free. Non-primitive types
//! (String, Vec, user structs) route to [`GenericArena`] where
//! values live in `UnsafeCell<Option<T>>` cells gated by the node
//! state machine.
//!
//! This trait exists because Rust's stable surface does not support
//! specialization: you cannot write a blanket impl plus overrides
//! for specific concrete types. So instead the runtime's generic
//! bounds use `Value`, and the trait's methods perform the concrete
//! downcast at each call site. The downcast is `downcast_ref` which
//! costs ~2 ns (a `TypeId` compare). For primitives this is
//! outweighed by the atomic-load fast path being ~3-5 ns cheaper
//! than GenericArena's `Option<T>::as_ref().unwrap().clone()`.
//!
//! ## Why not specialization
//!
//! Nightly `#[feature(min_specialization)]` would let us write
//!
//! ```ignore
//! default impl<T: Clone + PartialEq + ...> Value for T { /* generic */ }
//! impl Value for u64 { /* primitive override */ }
//! ```
//!
//! which is strictly cleaner. v2 targets stable. We accept the
//! small per-call downcast overhead as the price of stability and
//! pay it back in primitive performance.
//!
//! ## User-facing implications
//!
//! Because there is no blanket impl, a user who wants to store a
//! custom type `MyStruct` in an incr Runtime must provide an
//! explicit `Value` impl for `MyStruct`. The [`impl_value_generic`]
//! macro generates the boilerplate: `impl_value_generic!(MyStruct);`.
//! Every `impl Value` for a generic type routes to
//! `GenericArena<MyStruct>`; primitive dispatch is only for the
//! sealed list of primitives in this module.

use super::arena::{AtomicPrimitive, AtomicPrimitiveArena, ErasedArena, GenericArena};

/// A type that can be stored in an incr Runtime. Dispatches between
/// `AtomicPrimitiveArena` (for primitives) and `GenericArena` (for
/// everything else).
///
/// All methods take `&dyn ErasedArena` and internally downcast to
/// the concrete arena type. The downcast is guaranteed to succeed
/// when the arena was originally constructed via `Self::create_arena`
/// (the registry enforces this by keying on `TypeId::of::<T>()`).
/// A panic from the downcast indicates a library bug, not a user
/// error.
pub trait Value: Clone + PartialEq + Send + Sync + 'static {
    /// Construct the concrete arena for this value type. Called once
    /// per type by `ArenaRegistry::ensure_arena` the first time the
    /// runtime sees a node of this type.
    fn create_arena() -> Box<dyn ErasedArena>;

    /// Reserve a new slot populated with `initial`. Used by
    /// `create_input` where the value is known at node creation.
    fn reserve_with(arena: &dyn ErasedArena, initial: Self) -> u32;

    /// Reserve an empty slot. Used by `create_query` where the slot
    /// will be populated on the first compute. For primitive arenas
    /// the slot is zero-initialized; for generic arenas it is `None`.
    fn reserve_empty(arena: &dyn ErasedArena) -> u32;

    /// Read the value at `slot`. Caller is responsible for
    /// establishing happens-before with the most recent writer via
    /// an Acquire load on the node's state before calling.
    fn read(arena: &dyn ErasedArena, slot: u32) -> Self;

    /// Try to read the value at `slot`, returning `None` if the slot
    /// has not yet been populated. For primitive arenas this is
    /// always `Some` (slots are initialized to zero on reserve).
    /// For generic arenas this is `None` when the slot is an
    /// uninitialized `Option::None` (the first-compute-panicked
    /// case; see commit L's retry path).
    fn try_read(arena: &dyn ErasedArena, slot: u32) -> Option<Self>;

    /// Overwrite the value at `slot`. Caller must own exclusive
    /// access to the slot (via Computing state ownership or the
    /// runtime's write mutex).
    fn write(arena: &dyn ErasedArena, slot: u32, value: Self);
}

/// Internal: unified implementation of the `Value` trait body for
/// primitive types backed by `AtomicPrimitiveArena<Self>`. The
/// macro expands to a full trait impl; see `impl_value_primitive`
/// usages below for the concrete applications.
macro_rules! impl_value_primitive {
    ($t:ty) => {
        impl Value for $t {
            #[inline]
            fn create_arena() -> Box<dyn ErasedArena> {
                Box::new(AtomicPrimitiveArena::<$t>::new())
            }

            #[inline]
            fn reserve_with(arena: &dyn ErasedArena, initial: Self) -> u32 {
                downcast_primitive::<$t>(arena).reserve(initial)
            }

            #[inline]
            fn reserve_empty(arena: &dyn ErasedArena) -> u32 {
                downcast_primitive::<$t>(arena).reserve(<$t as AtomicPrimitive>::zero())
            }

            #[inline]
            fn read(arena: &dyn ErasedArena, slot: u32) -> Self {
                downcast_primitive::<$t>(arena).read(slot)
            }

            #[inline]
            fn try_read(arena: &dyn ErasedArena, slot: u32) -> Option<Self> {
                // Primitive slots are initialized to zero on reserve,
                // so there is no "uninitialized" state to return None
                // for. Always Some.
                Some(downcast_primitive::<$t>(arena).read(slot))
            }

            #[inline]
            fn write(arena: &dyn ErasedArena, slot: u32, value: Self) {
                downcast_primitive::<$t>(arena).write(slot, value);
            }
        }
    };
}

/// Internal: concrete downcast helper for the primitive Value impls.
/// Factored out of the macro so the downcast site has one code path
/// and one panic message across all primitive types.
#[inline]
fn downcast_primitive<T: AtomicPrimitive>(arena: &dyn ErasedArena) -> &AtomicPrimitiveArena<T> {
    arena
        .as_any()
        .downcast_ref::<AtomicPrimitiveArena<T>>()
        .expect("Value impl invariant violated: primitive arena type mismatch")
}

impl_value_primitive!(u32);
impl_value_primitive!(i32);
impl_value_primitive!(u64);
impl_value_primitive!(i64);
impl_value_primitive!(f32);
impl_value_primitive!(f64);
impl_value_primitive!(bool);

/// Internal: unified implementation of the `Value` trait body for
/// non-primitive types backed by `GenericArena<Self>`. Parallels
/// `impl_value_primitive` but routes to the generic arena.
macro_rules! impl_value_generic {
    ($t:ty) => {
        impl Value for $t {
            #[inline]
            fn create_arena() -> Box<dyn ErasedArena> {
                Box::new(GenericArena::<$t>::new())
            }

            #[inline]
            fn reserve_with(arena: &dyn ErasedArena, initial: Self) -> u32 {
                downcast_generic::<$t>(arena).reserve_with(initial)
            }

            #[inline]
            fn reserve_empty(arena: &dyn ErasedArena) -> u32 {
                downcast_generic::<$t>(arena).reserve()
            }

            #[inline]
            fn read(arena: &dyn ErasedArena, slot: u32) -> Self {
                downcast_generic::<$t>(arena).read(slot)
            }

            #[inline]
            fn try_read(arena: &dyn ErasedArena, slot: u32) -> Option<Self> {
                downcast_generic::<$t>(arena).try_read(slot)
            }

            #[inline]
            fn write(arena: &dyn ErasedArena, slot: u32, value: Self) {
                downcast_generic::<$t>(arena).write(slot, value);
            }
        }
    };
}

/// Concrete downcast helper for the generic Value impls. Public so that
/// `impl_value!` expansions in downstream crates can call it.
#[inline]
pub fn downcast_generic<T: Clone + Send + Sync + 'static>(
    arena: &dyn ErasedArena,
) -> &GenericArena<T> {
    arena
        .as_any()
        .downcast_ref::<GenericArena<T>>()
        .expect("Value impl invariant violated: generic arena type mismatch")
}

impl_value_generic!(String);

/// Blanket impl for `Vec<T>`. This is the one place where a generic
/// impl is OK: `Vec<T>` is a distinct type from any primitive or
/// other named type, so there is no conflict with the primitive
/// impls. Note that this does NOT require `T: Value`; only that `T`
/// has the underlying bounds the arena needs (`Clone + PartialEq +
/// Send + Sync + 'static`). A nested Incr-aware Vec would be
/// pathological and is not a supported pattern.
impl<T> Value for Vec<T>
where
    T: Clone + PartialEq + Send + Sync + 'static,
{
    #[inline]
    fn create_arena() -> Box<dyn ErasedArena> {
        Box::new(GenericArena::<Vec<T>>::new())
    }

    #[inline]
    fn reserve_with(arena: &dyn ErasedArena, initial: Self) -> u32 {
        downcast_generic::<Vec<T>>(arena).reserve_with(initial)
    }

    #[inline]
    fn reserve_empty(arena: &dyn ErasedArena) -> u32 {
        downcast_generic::<Vec<T>>(arena).reserve()
    }

    #[inline]
    fn read(arena: &dyn ErasedArena, slot: u32) -> Self {
        downcast_generic::<Vec<T>>(arena).read(slot)
    }

    #[inline]
    fn try_read(arena: &dyn ErasedArena, slot: u32) -> Option<Self> {
        downcast_generic::<Vec<T>>(arena).try_read(slot)
    }

    #[inline]
    fn write(arena: &dyn ErasedArena, slot: u32, value: Self) {
        downcast_generic::<Vec<T>>(arena).write(slot, value);
    }
}

impl<T> Value for Option<T>
where
    T: Clone + PartialEq + Send + Sync + 'static,
{
    #[inline]
    fn create_arena() -> Box<dyn ErasedArena> {
        Box::new(GenericArena::<Option<T>>::new())
    }

    #[inline]
    fn reserve_with(arena: &dyn ErasedArena, initial: Self) -> u32 {
        downcast_generic::<Option<T>>(arena).reserve_with(initial)
    }

    #[inline]
    fn reserve_empty(arena: &dyn ErasedArena) -> u32 {
        downcast_generic::<Option<T>>(arena).reserve()
    }

    #[inline]
    fn read(arena: &dyn ErasedArena, slot: u32) -> Self {
        downcast_generic::<Option<T>>(arena).read(slot)
    }

    #[inline]
    fn try_read(arena: &dyn ErasedArena, slot: u32) -> Option<Self> {
        downcast_generic::<Option<T>>(arena).try_read(slot)
    }

    #[inline]
    fn write(arena: &dyn ErasedArena, slot: u32, value: Self) {
        downcast_generic::<Option<T>>(arena).write(slot, value);
    }
}

impl<A, B> Value for (A, B)
where
    A: Clone + PartialEq + Send + Sync + 'static,
    B: Clone + PartialEq + Send + Sync + 'static,
{
    #[inline]
    fn create_arena() -> Box<dyn ErasedArena> {
        Box::new(GenericArena::<(A, B)>::new())
    }

    #[inline]
    fn reserve_with(arena: &dyn ErasedArena, initial: Self) -> u32 {
        downcast_generic::<(A, B)>(arena).reserve_with(initial)
    }

    #[inline]
    fn reserve_empty(arena: &dyn ErasedArena) -> u32 {
        downcast_generic::<(A, B)>(arena).reserve()
    }

    #[inline]
    fn read(arena: &dyn ErasedArena, slot: u32) -> Self {
        downcast_generic::<(A, B)>(arena).read(slot)
    }

    #[inline]
    fn try_read(arena: &dyn ErasedArena, slot: u32) -> Option<Self> {
        downcast_generic::<(A, B)>(arena).try_read(slot)
    }

    #[inline]
    fn write(arena: &dyn ErasedArena, slot: u32, value: Self) {
        downcast_generic::<(A, B)>(arena).write(slot, value);
    }
}

/// Public macro for implementing Value for user-defined types.
/// Routes the type to GenericArena.
///
/// Usage: `incr_core::impl_value!(MyStruct);`
#[macro_export]
macro_rules! impl_value {
    ($t:ty) => {
        impl $crate::Value for $t {
            #[inline]
            fn create_arena() -> Box<dyn $crate::arena::ErasedArena> {
                Box::new($crate::arena::GenericArena::<$t>::new())
            }

            #[inline]
            fn reserve_with(arena: &dyn $crate::arena::ErasedArena, initial: Self) -> u32 {
                $crate::value::downcast_generic::<$t>(arena).reserve_with(initial)
            }

            #[inline]
            fn reserve_empty(arena: &dyn $crate::arena::ErasedArena) -> u32 {
                $crate::value::downcast_generic::<$t>(arena).reserve()
            }

            #[inline]
            fn read(arena: &dyn $crate::arena::ErasedArena, slot: u32) -> Self {
                $crate::value::downcast_generic::<$t>(arena).read(slot)
            }

            #[inline]
            fn try_read(arena: &dyn $crate::arena::ErasedArena, slot: u32) -> Option<Self> {
                $crate::value::downcast_generic::<$t>(arena).try_read(slot)
            }

            #[inline]
            fn write(arena: &dyn $crate::arena::ErasedArena, slot: u32, value: Self) {
                $crate::value::downcast_generic::<$t>(arena).write(slot, value);
            }
        }
    };
}
