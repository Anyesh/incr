//! Sizes for large tests (500 nodes or more). Integration tests cannot see
//! the lib's `cfg(test)` items, so they include this file with
//! `#[path = "../src/test_size.rs"] mod test_size;`.

/// `full`, or a reduced size when `INCR_TEST_REDUCED` is set (the TSan job
/// sets it). The reduced size keeps at least 1,000 nodes, or `full` if that
/// is smaller, so each test still reaches the depth its mechanism needs.
pub(crate) fn test_size(full: usize) -> usize {
    if std::env::var_os("INCR_TEST_REDUCED").is_some() {
        reduced(full)
    } else {
        full
    }
}

fn reduced(full: usize) -> usize {
    (full / 10).max(full.min(1_000))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_is_a_tenth_with_a_floor_of_a_thousand() {
        assert_eq!(reduced(1_100_000), 110_000);
        assert_eq!(reduced(100_000), 10_000);
        assert_eq!(reduced(10_000), 1_000);
        assert_eq!(reduced(2_000), 1_000);
        assert_eq!(reduced(700), 700);
    }

    #[test]
    fn test_size_follows_the_environment() {
        let expected = if std::env::var_os("INCR_TEST_REDUCED").is_some() {
            10_000
        } else {
            100_000
        };
        assert_eq!(test_size(100_000), expected);
    }
}
