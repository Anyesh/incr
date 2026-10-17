//! Test hook that forces a haphazard reclamation at one named retire site,
//! so a pending value whose `Drop` panics is dropped exactly there.

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum RetireSite {
    /// The compute wrapper's value write, before the landing.
    ComputeWrite = 1,
    /// A displaced dep list's retire at a `Deferred` drain.
    DepListRetire = 2,
    /// A delete's retires after it released `inner`.
    DeleteRetire = 3,
    /// `set`'s drop of its displaced value, after its set-stack pop.
    SetRetire = 4,
}

#[cfg(test)]
static FORCED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(all(test, not(loom)))]
pub(crate) fn force(site: RetireSite) {
    FORCED.store(site as u8, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(test)]
pub(crate) fn reached(site: RetireSite) {
    if FORCED.load(std::sync::atomic::Ordering::SeqCst) == site as u8 {
        haphazard::Domain::global().eager_reclaim();
    }
}

#[cfg(not(test))]
#[inline(always)]
pub(crate) fn reached(_site: RetireSite) {}
