//! A list that keeps its first item inline, for the replay indexes, where
//! almost every key holds one entry and a `Vec` each would allocate.

#[derive(Clone, Debug)]
pub(crate) enum Few<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> Few<T> {
    pub(crate) fn push(&mut self, x: T) {
        match self {
            Few::Many(v) => v.push(x),
            Few::One(_) => {
                if let Few::One(a) = std::mem::replace(self, Few::Many(Vec::new())) {
                    *self = Few::Many(vec![a, x]);
                }
            }
        }
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            Few::One(a) => std::slice::from_ref(a),
            Few::Many(v) => v,
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Few::One(a) => std::slice::from_mut(a),
            Few::Many(v) => v,
        }
    }

    /// Keeps the items `f` accepts, returning whether none is left; the
    /// caller then drops the list.
    pub(crate) fn retain(&mut self, mut f: impl FnMut(&T) -> bool) -> bool {
        match self {
            Few::One(a) => !f(a),
            Few::Many(v) => {
                v.retain(|x| f(x));
                v.is_empty()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Few;

    #[test]
    fn pushes_keep_order_and_retain_reports_empty() {
        let mut l = Few::One(1);
        assert_eq!(l.as_slice(), &[1]);
        l.push(2);
        l.push(3);
        assert_eq!(l.as_slice(), &[1, 2, 3]);
        l.as_mut_slice()[1] = 5;
        assert!(!l.retain(|&x| x != 1));
        assert_eq!(l.as_slice(), &[5, 3]);
        assert!(l.retain(|_| false));
        let mut one = Few::One(7);
        assert!(!one.retain(|&x| x == 7));
        assert!(one.retain(|&x| x != 7));
    }
}
