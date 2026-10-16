use std::collections::{BTreeMap, HashMap};

use crate::{Kind, Record};

/// The first live `Declare` of a key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Declared {
    pub kind: Kind,
    pub default: Vec<u8>,
}

/// What the log says one key holds, folded from its records in position
/// order, as serialized bytes. Kept for every key the log knows, bound in
/// this process or not, so binding a key later, folding a snapshot and
/// listing restored keys all read the same state.
///
/// The elements of a collection bound in this process are held by its
/// binding, by decoded value, so `elements` stays empty for such a key.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LoggedKey {
    pub declared: Option<Declared>,
    /// The latest `Set` since the key was declared or last deleted.
    pub value: Option<Vec<u8>>,
    pub elements: Elements,
}

/// A collection's elements folded without knowing their type.
///
/// Two equal elements may serialize to different bytes (a `HashMap` field
/// iterates in an arbitrary order), so a `Remove` whose bytes match no live
/// element might still remove one by value. Up to the first such `Remove`
/// the byte multiset is exact, because equal bytes always decode to equal
/// values; from it on, element records are kept raw in `tail` until a
/// binding decodes them or a `Delete` clears them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Elements {
    live: HashMap<Vec<u8>, usize>,
    tail: Vec<ElementOp>,
}

/// One element change kept raw in [`Elements`]' tail.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ElementOp {
    Insert(Vec<u8>),
    Remove(Vec<u8>),
}

impl LoggedKey {
    /// Folds one record in. A `Declare` for a key that already has a live
    /// one is ignored, which is what makes the first in log order win.
    pub fn apply(&mut self, record: Record) {
        match record {
            Record::Declare { kind, default, .. } => {
                if self.declared.is_none() {
                    self.declared = Some(Declared { kind, default });
                }
            }
            Record::Set { value, .. } => self.value = Some(value),
            Record::Insert { element, .. } => self.elements.insert(element),
            Record::Remove { element, .. } => self.elements.remove(element),
            Record::Replace { old, new, .. } => {
                if let Some(old) = old {
                    self.elements.remove(old);
                }
                self.elements.insert(new);
            }
            Record::Delete { .. } => {
                self.value = None;
                self.elements = Elements::default();
            }
        }
    }

    /// The bytes an input currently holds: its latest value, else its
    /// logged default. `None` when the log has neither.
    pub fn current(&self) -> Option<&[u8]> {
        self.value
            .as_deref()
            .or(self.declared.as_ref().map(|d| d.default.as_slice()))
    }
}

impl Elements {
    /// Live elements by serialized bytes with their multiplicity. Apply
    /// [`Elements::tail`] after them to reach the current state.
    pub fn live(&self) -> impl Iterator<Item = (&[u8], usize)> {
        self.live
            .iter()
            .map(|(bytes, &count)| (bytes.as_slice(), count))
    }

    pub fn tail(&self) -> &[ElementOp] {
        &self.tail
    }

    fn insert(&mut self, element: Vec<u8>) {
        if self.tail.is_empty() {
            *self.live.entry(element).or_insert(0) += 1;
        } else {
            self.tail.push(ElementOp::Insert(element));
        }
    }

    fn remove(&mut self, element: Vec<u8>) {
        if !self.tail.is_empty() {
            self.tail.push(ElementOp::Remove(element));
        } else if let Some(count) = self.live.get_mut(&element) {
            *count -= 1;
            if *count == 0 {
                self.live.remove(&element);
            }
        } else if !self.live.is_empty() {
            self.tail.push(ElementOp::Remove(element));
        }
    }
}

/// Whether `record` changes a collection's elements, as opposed to its
/// declaration, its value or the whole key.
pub(crate) fn is_element_change(record: &Record) -> bool {
    matches!(
        record,
        Record::Insert { .. } | Record::Remove { .. } | Record::Replace { .. }
    )
}

/// Every key the log knows, by name.
pub(crate) type LoggedKeys = BTreeMap<String, LoggedKey>;

/// Folds `record` into the entry for its key, creating the entry if needed,
/// and returns the key.
pub(crate) fn fold(keys: &mut LoggedKeys, mut record: Record) -> String {
    let key = take_key(&mut record);
    if !keys.contains_key(&key) {
        keys.insert(key.clone(), LoggedKey::default());
    }
    keys.get_mut(&key).expect("inserted above").apply(record);
    key
}

// Moves the key out instead of cloning it, because every applied record
// passes through here; `apply` never reads the key.
fn take_key(record: &mut Record) -> String {
    match record {
        Record::Declare { key, .. }
        | Record::Set { key, .. }
        | Record::Insert { key, .. }
        | Record::Remove { key, .. }
        | Record::Replace { key, .. }
        | Record::Delete { key } => std::mem::take(key),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert(element: u8) -> Record {
        Record::Insert {
            key: "c".into(),
            element: vec![element],
        }
    }

    #[test]
    fn first_declare_wins_and_delete_resets_to_it() {
        let mut keys = LoggedKeys::new();
        for record in [
            Record::Declare {
                key: "k".into(),
                kind: Kind::Input,
                default: vec![1],
            },
            Record::Declare {
                key: "k".into(),
                kind: Kind::Input,
                default: vec![2],
            },
            Record::Set {
                key: "k".into(),
                value: vec![9],
            },
        ] {
            fold(&mut keys, record);
        }
        assert_eq!(keys["k"].current(), Some(&[9][..]));
        fold(&mut keys, Record::Delete { key: "k".into() });
        assert_eq!(keys["k"].current(), Some(&[1][..]));
    }

    #[test]
    fn collections_fold_to_a_multiset() {
        let mut keys = LoggedKeys::new();
        for record in [
            insert(1),
            insert(1),
            insert(2),
            Record::Remove {
                key: "c".into(),
                element: vec![2],
            },
            Record::Replace {
                key: "c".into(),
                old: Some(vec![1]),
                new: vec![3],
            },
            Record::Replace {
                key: "c".into(),
                old: None,
                new: vec![3],
            },
        ] {
            fold(&mut keys, record);
        }
        let expected = HashMap::from([(vec![1], 1), (vec![3], 2)]);
        assert_eq!(keys["c"].elements.live, expected);
        assert!(keys["c"].elements.tail.is_empty());
        fold(&mut keys, Record::Delete { key: "c".into() });
        assert_eq!(keys["c"].elements, Elements::default());
    }

    #[test]
    fn a_remove_matching_no_live_bytes_keeps_later_changes_raw() {
        let mut keys = LoggedKeys::new();
        let remove = |element: u8| Record::Remove {
            key: "c".into(),
            element: vec![element],
        };
        for record in [insert(1), insert(2), remove(3), insert(1), remove(1)] {
            fold(&mut keys, record);
        }
        let elements = &keys["c"].elements;
        assert_eq!(elements.live, HashMap::from([(vec![1], 1), (vec![2], 1)]));
        assert_eq!(
            elements.tail,
            [
                ElementOp::Remove(vec![3]),
                ElementOp::Insert(vec![1]),
                ElementOp::Remove(vec![1]),
            ]
        );

        fold(&mut keys, Record::Delete { key: "c".into() });
        assert_eq!(keys["c"].elements, Elements::default());
    }

    #[test]
    fn a_remove_from_no_elements_is_dropped() {
        let mut keys = LoggedKeys::new();
        fold(
            &mut keys,
            Record::Remove {
                key: "c".into(),
                element: vec![5],
            },
        );
        assert_eq!(keys["c"].elements, Elements::default());
    }
}
