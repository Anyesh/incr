//! A list that keeps its first item inline, for the replay indexes, where
//! almost every key holds one entry and a `Vec` each would allocate.

#[derive(Clone, Debug, Default)]
pub(crate) enum Few<T> {
    #[default]
    None,
    One(T),
    Many(Vec<T>),
}

impl<T> Few<T> {
    pub(crate) fn push(&mut self, x: T) {
        match std::mem::take(self) {
            Few::None => *self = Few::One(x),
            Few::One(a) => *self = Few::Many(vec![a, x]),
            Few::Many(mut v) => {
                v.push(x);
                *self = Few::Many(v);
            }
        }
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        match self {
            Few::None => &[],
            Few::One(a) => std::slice::from_ref(a),
            Few::Many(v) => v,
        }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Few::None => &mut [],
            Few::One(a) => std::slice::from_mut(a),
            Few::Many(v) => v,
        }
    }

    /// Keeps the items `f` accepts, returning whether none is left.
    pub(crate) fn retain(&mut self, mut f: impl FnMut(&T) -> bool) -> bool {
        match self {
            Few::None => {}
            Few::One(a) => {
                if !f(a) {
                    *self = Few::None;
                }
            }
            Few::Many(v) => v.retain(|x| f(x)),
        }
        self.as_slice().is_empty()
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
        assert!(one.as_slice().is_empty());
        one.push(8);
        assert_eq!(one.as_slice(), &[8]);
    }
}
