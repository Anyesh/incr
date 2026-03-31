use std::marker::PhantomData;

/// Index into the node arena. Cheap to copy and compare.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct NodeId(pub(crate) u32);

/// Monotonically increasing counter. Incremented on every input mutation.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Default)]
pub struct Revision(pub(crate) u64);

impl Revision {
    pub(crate) fn increment(&mut self) {
        self.0 += 1;
    }
}

/// Typed handle to a node in the incremental graph. `T` is the value type.
/// Cheap to copy — it's just a u32 index + phantom type.
#[derive(Debug)]
pub struct Incr<T> {
    pub(crate) id: NodeId,
    pub(crate) _phantom: PhantomData<T>,
}

// Manual impls because derive would add T: Copy/Clone bounds
impl<T> Copy for Incr<T> {}
impl<T> Clone for Incr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
