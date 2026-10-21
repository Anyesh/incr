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
/// iterates in an arbitrary order), and the runtime keeps the first
/// inserted of equal elements as the one it stores. So live bytes keep the
/// order they were first inserted in, and a change is folded only when the
/// result holds whatever `T`'s `Eq` is: an insert, a remove from no
/// elements, or a remove of bytes with more than one live occurrence or of
/// the only occurrence of the latest inserted bytes. Any other remove might remove an equal element
/// stored under other bytes, or one an earlier equal element must outlive;
/// from it on, element records are kept raw in `tail` until a binding
/// decodes them or a `Delete` clears them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Elements {
    live: HashMap<Vec<u8>, Live>,
    tail: Vec<ElementOp>,
}

/// One live element's occurrences and its place in insertion order. Only
/// the latest inserted bytes are ever dropped, so the places are always
/// `0..live.len()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Live {
    count: usize,
    at: usize,
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
            Record::Forget { .. } => *self = LoggedKey::default(),
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

    /// Live elements by serialized bytes with their multiplicity, in the
    /// order first inserted. Apply [`Elements::tail`] after them to reach
    /// the current state.
    pub fn live(&self) -> impl DoubleEndedIterator<Item = (&[u8], usize)> {
        let mut live = vec![(&[][..], 0); self.live.len()];
        for (bytes, entry) in &self.live {
            live[entry.at] = (bytes.as_slice(), entry.count);
        }
        live.into_iter()
    }

    pub fn tail(&self) -> &[ElementOp] {
        &self.tail
    }

    /// Pushes one `Insert` per live element occurrence, then the raw tail,
    /// the records that fold back to these elements from none.
    fn records(&self, key: &str, out: &mut Vec<Record>) {
        for (element, count) in self.live() {
            out.extend((0..count).map(|_| Record::Insert {
                key: key.to_owned(),
                element: element.to_vec(),
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

    /// Pushes a `Remove` per occurrence `new` lacks, latest inserted first,
    /// then an `Insert` per occurrence it adds, in its order. Pushes nothing
    /// and returns false when those records do not fold `self` into exactly
    /// `new`: either has a raw tail, or `new`'s order is not one they reach.
    fn diff(&self, key: &str, new: &Elements, out: &mut Vec<Record>) -> bool {
        if !self.tail.is_empty() || !new.tail.is_empty() {
            return false;
        }
        let count = |elements: &Elements, element: &[u8]| {
            elements.live.get(element).map_or(0, |entry| entry.count)
        };
        let mut changes = Vec::new();
        for (element, have) in self.live().rev() {
            let want = count(new, element);
            changes.extend((want..have).map(|_| ElementOp::Remove(element.to_vec())));
        }
        for (element, want) in new.live() {
            let have = count(self, element);
            changes.extend((have..want).map(|_| ElementOp::Insert(element.to_vec())));
        }
        let mut reached = self.clone();
        for op in &changes {
            match op {
                ElementOp::Insert(element) => reached.insert(element.clone()),
                ElementOp::Remove(element) => reached.remove(element.clone()),
            }
        }
        if reached != *new {
            return false;
        }
        out.extend(changes.into_iter().map(|op| match op {
            ElementOp::Insert(element) => Record::Insert {
                key: key.to_owned(),
                element,
            },
            ElementOp::Remove(element) => Record::Remove {
                key: key.to_owned(),
                element,
            },
        }));
        true
    }

    fn insert(&mut self, element: Vec<u8>) {
        if self.tail.is_empty() {
            let at = self.live.len();
            self.live
                .entry(element)
                .or_insert(Live { count: 0, at })
                .count += 1;
        } else {
            self.tail.push(ElementOp::Insert(element));
        }
    }

    fn remove(&mut self, element: Vec<u8>) {
        if self.tail.is_empty() {
            let latest = self.live.len().checked_sub(1);
            match self.live.get_mut(&element) {
                Some(entry) if entry.count > 1 => {
                    entry.count -= 1;
                    return;
                }
                Some(entry) if Some(entry.at) == latest => {
                    self.live.remove(&element);
                    return;
                }
                None if latest.is_none() => return,
                _ => {}
            }
        }
        self.tail.push(ElementOp::Remove(element));
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
///
/// A first-wins fold never replaces a live `Declare`, so a key the snapshot
/// dropped, or whose `Declare` it dropped or changed, is reported as a
/// `Forget` followed by the key's full new state, elements included even
/// when `elements` is false. Returns whether it did so, in which case a
/// binding must not report its own element changes on top.
pub(crate) fn changes_between(
    key: &str,
    old: Option<&LoggedKey>,
    new: Option<&LoggedKey>,
    elements: bool,
    out: &mut Vec<Record>,
) -> bool {
    let none = LoggedKey::default();
    let (old, new) = (old.unwrap_or(&none), new.unwrap_or(&none));
    let undeclared = old.declared.is_some() && new.declared != old.declared;
    if !old.is_empty() && (new.is_empty() || undeclared) {
        out.push(Record::Forget {
            key: key.to_owned(),
        });
        out.extend(new.records(key));
        return true;
    }
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
        return false;
    }
    let elements_changed = old.elements != new.elements;
    let mut diff = Vec::new();
    let diffed = elements_changed && old.elements.diff(key, &new.elements, &mut diff);
    // A value going back to unset has no Set to express it, and elements
    // no diff reaches have no records that do; both are a Delete and the
    // full new state.
    if (old.value.is_some() && new.value.is_none()) || (elements_changed && !diffed) {
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
        return false;
    }
    if new.value != old.value {
        if let Some(value) = &new.value {
            out.push(Record::Set {
                key: key.to_owned(),
                value: value.clone(),
            });
        }
    }
    out.extend(diff);
    false
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
        | Record::Delete { key }
        | Record::Forget { key } => std::mem::take(key),
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
        assert_eq!(live(&keys["c"].elements), [(vec![1], 1), (vec![3], 2)]);
        assert!(keys["c"].elements.tail.is_empty());
        fold(&mut keys, Record::Delete { key: "c".into() });
        assert!(!keys.contains_key("c"), "nothing is left without a Declare");
    }

    fn live(elements: &Elements) -> Vec<(Vec<u8>, usize)> {
        elements
            .live()
            .map(|(bytes, count)| (bytes.to_vec(), count))
            .collect()
    }

    fn remove(element: u8) -> Record {
        Record::Remove {
            key: "c".into(),
            element: vec![element],
        }
    }

    // Equal elements may encode as 1 and 2, so the remove of 1's only
    // occurrence might have to leave 1 as what an equal 2 is stored as.
    #[test]
    fn live_bytes_keep_insertion_order_and_fold_only_removes_any_eq_allows() {
        let mut keys = LoggedKeys::new();
        for record in [
            insert(2),
            insert(1),
            insert(2),
            insert(3),
            remove(3),
            remove(2),
        ] {
            fold(&mut keys, record);
        }
        assert_eq!(live(&keys["c"].elements), [(vec![2], 1), (vec![1], 1)]);
        assert!(keys["c"].elements.tail.is_empty());
        fold(&mut keys, remove(2));
        assert_eq!(live(&keys["c"].elements), [(vec![2], 1), (vec![1], 1)]);
        assert_eq!(keys["c"].elements.tail, [ElementOp::Remove(vec![2])]);
        let records = keys["c"].records("c");
        assert_eq!(records[..2], [insert(2), insert(1)]);
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
        assert_eq!(live(elements), [(vec![1], 1), (vec![2], 1)]);
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
    // one, declaration included, which is what lets a subscriber mirror a
    // catch-up it never saw record by record.
    #[test]
    fn changes_between_take_the_old_state_to_the_new() {
        let declare = Record::Declare {
            key: "c".into(),
            kind: Kind::Collection,
            default: vec![],
        };
        let declare_input = |default: u8| Record::Declare {
            key: "c".into(),
            kind: Kind::Input,
            default: vec![default],
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
            (vec![declare_input(1)], vec![]),
            (vec![declare_input(1), set(3)], vec![declare_input(2)]),
            (vec![declare_input(1)], vec![declare_input(2), set(3)]),
            (vec![declare.clone(), insert(5)], vec![insert(5), insert(6)]),
            (vec![declare_input(1)], vec![declare.clone(), insert(5)]),
            (vec![set(1), insert(2)], vec![]),
            (vec![insert(1), insert(2)], vec![insert(2)]),
            (vec![insert(1)], vec![insert(2), insert(1)]),
            (vec![insert(1), insert(2)], vec![insert(1), insert(3)]),
        ];
        for (old, new) in cases {
            let (old, new) = (folded(&old), folded(&new));
            let mut changes = Vec::new();
            changes_between("c", Some(&old), Some(&new), true, &mut changes);
            let mut replayed = old.clone();
            for record in changes {
                replayed.apply(record);
            }
            assert_eq!(replayed, new, "{old:?} -> {new:?}");
        }
        let (old, new) = (
            folded(&[insert(1), insert(1), insert(2)]),
            folded(&[insert(1), insert(3)]),
        );
        let mut changes = Vec::new();
        changes_between("c", Some(&old), Some(&new), true, &mut changes);
        assert_eq!(
            changes,
            [remove(2), remove(1), insert(3)],
            "diffed, not rebuilt"
        );
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
