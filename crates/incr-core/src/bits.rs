//! A build-time check that the flags and fields packed into one word do
//! not share a bit: branches that each pick the next free flag otherwise
//! collide silently.

/// Whether `fields`, the flags and multi-bit fields of one word, are each
/// nonzero and pairwise disjoint. Each word asserts it once, beside its
/// constants, listing every flag the word holds.
#[allow(
    dead_code,
    reason = "only `const _` asserts call it, which rustc 1.88, the MSRV, does not count as a use"
)]
pub(crate) const fn disjoint(fields: &[u64]) -> bool {
    let mut seen = 0;
    let mut i = 0;
    while i < fields.len() {
        if fields[i] == 0 || seen & fields[i] != 0 {
            return false;
        }
        seen |= fields[i];
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::disjoint;

    #[test]
    fn disjoint_flags_and_fields_pass() {
        assert!(disjoint(&[1, 2, 4, 0x30, 1 << 63]));
        assert!(disjoint(&[]));
    }

    #[test]
    fn a_shared_bit_or_an_empty_flag_fails() {
        assert!(!disjoint(&[2048, 1024, 2048]));
        assert!(!disjoint(&[3 << 61, 1 << 62]));
        assert!(!disjoint(&[1, 0]));
    }
}
