//! The Fx hash, written in-crate: a multiply-and-add per word, fast for
//! the small keys node tables hold. It is not resistant to chosen
//! collisions; a table keyed by untrusted input takes another hasher
//! through [`crate::Runtime::create_table_with_hasher`].

use std::hash::{BuildHasher, Hasher};

const K: u64 = 0xf135_7aea_2e62_a9c5;

#[derive(Clone, Copy, Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline(always)]
    fn add(&mut self, word: u64) {
        self.hash = self.hash.wrapping_add(word).wrapping_mul(K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rest) = bytes.as_chunks::<8>();
        for &chunk in chunks {
            self.add(u64::from_le_bytes(chunk));
        }
        if !rest.is_empty() {
            let mut last = [0u8; 8];
            last[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(last));
        }
        // The length keeps "ab" + "c" apart from "a" + "bc".
        self.add(bytes.len() as u64);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    /// The multiply leaves a product's low bits depending only on the
    /// input's low bits, and tables index by the low bits, so the result
    /// folds the high bits down before and after one more multiply.
    #[inline]
    fn finish(&self) -> u64 {
        let h = (self.hash ^ (self.hash >> 32)).wrapping_mul(K);
        h ^ (h >> 29)
    }
}

#[derive(Clone, Copy, Default)]
pub struct FxBuildHasher;

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;

    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_keys_hash_equal_and_split_strings_differ() {
        let b = FxBuildHasher;
        assert_eq!(b.hash_one("abc"), b.hash_one("abc"));
        assert_ne!(b.hash_one(("ab", "c")), b.hash_one(("a", "bc")));
        assert_ne!(b.hash_one(1_u64), b.hash_one(2_u64));
    }

    /// Pinned byte-string hashes: a table's iteration order, and anything
    /// keyed through it, depends on them, so a rewrite of `write` must keep
    /// every one.
    #[test]
    fn byte_hashes_keep_their_values() {
        for (len, want) in [
            (0, 0x0000_0000_0000_0000_u64),
            (1, 0x5479_45e6_76e1_bbbc),
            (7, 0xf74d_4a3d_93de_bae2),
            (8, 0x74ee_0a78_ee68_3e42),
            (9, 0x8da5_80d2_5022_a1b0),
            (15, 0xf71b_ddf7_d699_d3a5),
            (16, 0xa414_a2ae_74a1_b721),
            (17, 0x00e7_d7d9_e2f6_e63b),
            (64, 0x36af_e208_b9ac_47bb),
        ] {
            let bytes: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
            let mut h = FxHasher::default();
            h.write(&bytes);
            assert_eq!(h.finish(), want, "{len} bytes");
        }
    }

    #[test]
    fn consecutive_integers_spread_over_the_low_bits() {
        let b = FxBuildHasher;
        let mut seen = std::collections::HashSet::new();
        for i in 0..256_u64 {
            seen.insert(b.hash_one(i << 12) & 255);
        }
        assert!(
            seen.len() > 128,
            "only {} of 256 low-byte values",
            seen.len()
        );
    }
}
