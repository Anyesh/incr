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

    pub fn is_empty(&self) -> bool {
        self.declared.is_none() && self.value.is_none() && self.elements.is_empty()
    }

    /// The bytes an input currently holds: its latest value, else its
    /// logged default. `None` when the log has neither.
    pub fn current(&self) -> Option<&[u8]> {
        self.value
            .as_deref()
            .or(self.declared.as_ref().map(|d| d.default.as_slice()))
    }

    /// Whether the key holds nothing beyond its logged default, compared
    /// by bytes, so that compaction may forget it. A key without a
    /// `Declare` has no default and never qualifies.
    pub fn at_default(&self) -> bool {
        let Some(declared) = &self.declared else {
            return false;
        };
        self.elements.is_empty()
            && self
                .value
                .as_ref()
                .is_none_or(|value| *value == declared.default)
    }

    /// The records a snapshot keeps for this key: its `Declare`, its latest
    /// `Set`, then one `Insert` per live element occurrence followed by the
    /// raw tail. Folding them back reproduces this state.
    pub fn records(&self, key: &str) -> Vec<Record> {
        let mut out = Vec::new();
        if let Some(Declared { kind, default }) = &self.declared {
            out.push(Record::Declare {
                key: key.to_owned(),
                kind: *kind,
                default: default.clone(),
            });
        }
        if let Some(value) = &self.value {
            out.push(Record::Set {
                key: key.to_owned(),
                value: value.clone(),
            });
        }
        self.elements.records(key, &mut out);
        out
    }
}

impl Elements {
    /// The multiset of `elements`, one occurrence per item.
    pub fn from_encoded(elements: impl IntoIterator<Item = Vec<u8>>) -> Self {
        let mut out = Elements::default();
        for element in elements {
            out.insert(element);
        }
        out
    }

    pub fn is_empty(&self) -> bool {
        self.live.is_empty() && self.tail.is_empty()
    }

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

    /// Pushes one `Insert` per live element occurrence, then the raw tail,
    /// the records that fold back to these elements from none.
    fn records(&self, key: &str, out: &mut Vec<Record>) {
        for (element, &count) in &self.live {
            out.extend((0..count).map(|_| Record::Insert {
                key: key.to_owned(),
                element: element.clone(),
            }));
        }
        out.extend(self.tail.iter().map(|op| match op {
            ElementOp::Insert(element) => Record::Insert {
                key: key.to_owned(),
                element: element.clone(),
            },
            ElementOp::Remove(element) => Record::Remove {
                key: key.to_owned(),
                element: element.clone(),
            },
        }));
    }

    /// Pushes a `Remove` or `Insert` per occurrence that differs between
    /// `self` and `new`. Exact only while neither has a raw tail.
    fn diff(&self, key: &str, new: &Elements, out: &mut Vec<Record>) {
        for (element, &have) in &self.live {
            let want = new.live.get(element).copied().unwrap_or(0);
            out.extend((want..have).map(|_| Record::Remove {
                key: key.to_owned(),
                element: element.clone(),
            }));
        }
        for (element, &want) in &new.live {
            let have = self.live.get(element).copied().unwrap_or(0);
            out.extend((have..want).map(|_| Record::Insert {
                key: key.to_owned(),
                element: element.clone(),
            }));
        }
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

/// Pushes records onto `out` that take a subscriber from `old`, one key's
/// state, to `new`, its state in a snapshot; `None` is a key the log does
/// not know. Nothing is pushed for an unchanged key. `elements` is false
/// for a collection bound in this process, whose binding reports its own
/// element changes and whose value is meaningless, so only its `Declare`
/// is compared here.
pub(crate) fn changes_between(
    key: &str,
    old: Option<&LoggedKey>,
    new: Option<&LoggedKey>,
    elements: bool,
    out: &mut Vec<Record>,
) {
    let none = LoggedKey::default();
    let (old, new) = (old.unwrap_or(&none), new.unwrap_or(&none));
    if new.declared != old.declared {
        if let Some(Declared { kind, default }) = &new.declared {
            out.push(Record::Declare {
                key: key.to_owned(),
                kind: *kind,
                default: default.clone(),
            });
        }
    }
    if !elements {
        return;
    }
    let elements_changed = old.elements != new.elements;
    let raw = !old.elements.tail.is_empty() || !new.elements.tail.is_empty();
    // A value going back to unset has no Set to express it, and raw tails
    // cannot be diffed by bytes; both are a Delete and the full new state.
    if (old.value.is_some() && new.value.is_none()) || (elements_changed && raw) {
        out.push(Record::Delete {
            key: key.to_owned(),
        });
        if let Some(value) = &new.value {
            out.push(Record::Set {
                key: key.to_owned(),
                value: value.clone(),
            });
        }
        new.elements.records(key, out);
        return;
    }
    if new.value != old.value {
        if let Some(value) = &new.value {
            out.push(Record::Set {
                key: key.to_owned(),
                value: value.clone(),
            });
        }
    }
    if elements_changed {
        old.elements.diff(key, &new.elements, out);
    }
}

/// Every key the log knows, by name.
pub(crate) type LoggedKeys = BTreeMap<String, LoggedKey>;

/// Folds `record` into the entry for its key, creating the entry if needed
/// and dropping it when the record leaves it holding nothing, as a `Delete`
/// of a key without a `Declare` does, so the key stops counting as known.
pub(crate) fn fold(keys: &mut LoggedKeys, mut record: Record) {
    let key = take_key(&mut record);
    if let Some(logged) = keys.get_mut(&key) {
        logged.apply(record);
        if logged.is_empty() {
            keys.remove(&key);
        }
    } else {
        let mut logged = LoggedKey::default();
        logged.apply(record);
        if !logged.is_empty() {
            keys.insert(key, logged);
        }
    }
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
        assert!(!keys.contains_key("c"), "nothing is left without a Declare");
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

        let mut refolded = LoggedKeys::new();
        for record in keys["c"].records("c") {
            fold(&mut refolded, record);
        }
        assert_eq!(refolded["c"], keys["c"]);

        fold(&mut keys, Record::Delete { key: "c".into() });
        assert!(!keys.contains_key("c"), "nothing is left without a Declare");
    }

    #[test]
    fn a_record_that_leaves_a_key_holding_nothing_drops_it() {
        let mut keys = LoggedKeys::new();
        fold(
            &mut keys,
            Record::Remove {
                key: "c".into(),
                element: vec![5],
            },
        );
        assert!(keys.is_empty());
        fold(&mut keys, insert(1));
        fold(&mut keys, Record::Delete { key: "c".into() });
        assert!(keys.is_empty());
        fold(&mut keys, Record::Delete { key: "c".into() });
        assert!(keys.is_empty());
    }

    fn folded(records: &[Record]) -> LoggedKey {
        let mut keys = LoggedKeys::new();
        for record in records {
            fold(&mut keys, record.clone());
        }
        keys.remove("c").unwrap_or_default()
    }

    // Folding the reported changes over the old state must reach the new
    // one, which is what lets a subscriber mirror a catch-up it never saw
    // record by record.
    #[test]
    fn changes_between_take_the_old_state_to_the_new() {
        let declare = Record::Declare {
            key: "c".into(),
            kind: Kind::Collection,
            default: vec![],
        };
        let set = |value: u8| Record::Set {
            key: "c".into(),
            value: vec![value],
        };
        let remove = |element: u8| Record::Remove {
            key: "c".into(),
            element: vec![element],
        };
        let with_tail = [insert(1), insert(2), remove(3), insert(4)];
        let cases: Vec<(Vec<Record>, Vec<Record>)> = vec![
            (vec![set(1)], vec![set(2)]),
            (vec![set(1)], vec![]),
            (
                vec![insert(1), insert(1), insert(2)],
                vec![insert(1), insert(3)],
            ),
            (vec![insert(1)], with_tail.to_vec()),
            (with_tail.to_vec(), vec![insert(1)]),
            (vec![], vec![declare.clone(), set(4), insert(5)]),
            (vec![declare.clone(), insert(5)], vec![]),
        ];
        for (old, new) in cases {
            let (old, new) = (folded(&old), folded(&new));
            let mut changes = Vec::new();
            changes_between("c", Some(&old), Some(&new), true, &mut changes);
            let mut replayed = old.clone();
            for record in changes {
                replayed.apply(record);
            }
            assert_eq!(replayed.value, new.value, "{old:?} -> {new:?}");
            assert_eq!(replayed.elements, new.elements, "{old:?} -> {new:?}");
            if new.declared.is_some() {
                assert_eq!(replayed.declared, new.declared);
            }
        }
        let same = folded(&[declare, set(1), insert(2)]);
        let mut changes = Vec::new();
        changes_between("c", Some(&same), Some(&same), true, &mut changes);
        assert!(changes.is_empty());
    }

    #[test]
    fn at_default_needs_a_declare_and_nothing_beyond_its_default() {
        let declare = Record::Declare {
            key: "k".into(),
            kind: Kind::Input,
            default: vec![1],
        };
        let set = |value: u8| Record::Set {
            key: "k".into(),
            value: vec![value],
        };
        let mut keys = LoggedKeys::new();
        fold(&mut keys, set(1));
        assert!(!keys["k"].at_default(), "no Declare, so no default");
        fold(&mut keys, declare);
        assert!(keys["k"].at_default());
        fold(&mut keys, set(2));
        assert!(!keys["k"].at_default());
        fold(&mut keys, Record::Delete { key: "k".into() });
        assert!(keys["k"].at_default());
        fold(
            &mut keys,
            Record::Insert {
                key: "k".into(),
                element: vec![0],
            },
        );
        assert!(!keys["k"].at_default());
    }
}
