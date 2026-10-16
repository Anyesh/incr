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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LoggedKey {
    pub declared: Option<Declared>,
    /// The latest `Set` since the key was declared or last deleted.
    pub value: Option<Vec<u8>>,
    /// The element multiset of a collection, by serialized element.
    pub elements: HashMap<Vec<u8>, usize>,
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
            Record::Insert { element, .. } => self.insert(element),
            Record::Remove { element, .. } => self.remove(&element),
            Record::Replace { old, new, .. } => {
                if let Some(old) = old {
                    self.remove(&old);
                }
                self.insert(new);
            }
            Record::Delete { .. } => {
                self.value = None;
                self.elements.clear();
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

    fn insert(&mut self, element: Vec<u8>) {
        *self.elements.entry(element).or_insert(0) += 1;
    }

    fn remove(&mut self, element: &[u8]) {
        if let Some(count) = self.elements.get_mut(element) {
            *count -= 1;
            if *count == 0 {
                self.elements.remove(element);
            }
        }
    }
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
            Record::Remove {
                key: "c".into(),
                element: vec![7],
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
        assert_eq!(keys["c"].elements, expected);
        fold(&mut keys, Record::Delete { key: "c".into() });
        assert!(keys["c"].elements.is_empty());
    }
}
