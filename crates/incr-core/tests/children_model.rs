//! Owned children against a model: random sequences of key-set changes,
//! reads, reader attachments and reader deletes over a chain of creators
//! (creator `i` reads creator `i - 1`), under `Local` and `Shared`. After
//! every step a creator's live children are the keys of its last run plus
//! the children that run swept while a reader still read them.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use incr_core::{Cells, Incr, Local, Runtime, Shared};
use proptest::prelude::*;

const CREATORS: usize = 5;
const KEYS: u32 = 6;

#[derive(Clone, Debug)]
enum Op {
    Mask(usize, u8),
    Get(usize),
    AddReader(usize, u32),
    DeleteReader(usize),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        (0..CREATORS, 0..64_u8).prop_map(|(c, m)| Op::Mask(c, m)),
        (0..CREATORS).prop_map(Op::Get),
        (0..CREATORS, 0..KEYS).prop_map(|(c, k)| Op::AddReader(c, k)),
        (0..16_usize).prop_map(Op::DeleteReader),
    ]
}

fn keys(mask: u8) -> impl Iterator<Item = u32> {
    (0..KEYS).filter(move |k| mask & (1 << k) != 0)
}

#[derive(Default)]
struct Creator {
    /// The mask its input holds now.
    mask: u8,
    /// A set changed the input since the last run.
    input_changed: bool,
    /// The value of its last run, and how many runs changed its value:
    /// a dependent reruns when the count moved since it last ran, even if
    /// the value came back.
    value: Option<u8>,
    changes: u64,
    /// The upstream's change count its last run saw.
    seen_upstream: u64,
    live: BTreeSet<u32>,
}

#[derive(Default)]
struct Model {
    creators: Vec<Creator>,
    /// Each reader's creator and key.
    readers: Vec<(usize, u32)>,
}

impl Model {
    /// Brings creator `i` up to date as a pull of it does, and returns
    /// its value, the mask of its last run.
    fn pull(&mut self, i: usize) -> u8 {
        let upstream = if i == 0 {
            0
        } else {
            self.pull(i - 1);
            self.creators[i - 1].changes
        };
        let c = &self.creators[i];
        if c.value.is_none() || c.input_changed || c.seen_upstream != upstream {
            let mask = c.mask;
            self.run(i, mask);
            let c = &mut self.creators[i];
            if c.value != Some(mask) {
                c.changes += 1;
            }
            c.value = Some(mask);
            c.input_changed = false;
            c.seen_upstream = upstream;
        }
        self.creators[i].value.unwrap()
    }

    fn run(&mut self, i: usize, mask: u8) {
        let read: BTreeSet<u32> = self
            .readers
            .iter()
            .filter(|(c, _)| *c == i)
            .map(|&(_, k)| k)
            .collect();
        let kept: BTreeSet<u32> = keys(mask).collect();
        let creator = &mut self.creators[i];
        creator
            .live
            .retain(|k| kept.contains(k) || read.contains(k));
        creator.live.extend(kept);
    }
}

type Handles = Arc<Mutex<HashMap<(usize, u32), Incr<u32>>>>;

fn check<C: Cells>(rt: &Runtime<C>, model: &Model, handles: &Handles) {
    let handles = handles.lock().unwrap();
    for (i, creator) in model.creators.iter().enumerate() {
        for k in 0..KEYS {
            let live = handles.get(&(i, k)).is_some_and(|&h| rt.is_alive(h));
            assert_eq!(
                live,
                creator.live.contains(&k),
                "creator {i} key {k}: live {live}, model {:?}",
                creator.live
            );
        }
    }
}

fn run_ops<C: Cells>(ops: Vec<Op>) {
    let rt = Runtime::<C>::new();
    let handles: Handles = Arc::default();
    let mut model = Model::default();
    let mut masks = Vec::new();
    let mut creators: Vec<Incr<u8>> = Vec::new();
    for i in 0..CREATORS {
        let mask = rt.create_input(0_u8);
        let upstream = creators.last().copied();
        let handles = handles.clone();
        let creator = rt.create_query(move |rt| {
            if let Some(up) = upstream {
                let _ = rt.get(up);
            }
            let m = rt.get(mask);
            for k in keys(m) {
                let h = rt.child_input(k, move || k * 100);
                handles.lock().unwrap().insert((i, k), h);
            }
            m
        });
        masks.push(mask);
        creators.push(creator);
        model.creators.push(Creator::default());
    }
    let mut readers: Vec<Incr<u32>> = Vec::new();

    for op in ops {
        match op {
            Op::Mask(i, m) => {
                rt.set(masks[i], m);
                let c = &mut model.creators[i];
                c.input_changed |= c.mask != m;
                c.mask = m;
            }
            Op::Get(i) => {
                assert_eq!(rt.get(creators[i]), model.pull(i));
            }
            Op::AddReader(i, k) => {
                if !model.creators[i].live.contains(&k) {
                    continue;
                }
                let h = handles.lock().unwrap()[&(i, k)];
                let reader = rt.create_query(move |rt| rt.get(h) + 1);
                assert_eq!(rt.get(reader), k * 100 + 1);
                readers.push(reader);
                model.readers.push((i, k));
            }
            Op::DeleteReader(r) => {
                if r >= readers.len() {
                    continue;
                }
                rt.delete_node(readers.remove(r));
                model.readers.remove(r);
            }
        }
        check(&rt, &model, &handles);
    }

    for mask in masks {
        rt.delete_with_dependents(mask);
    }
    // A creator that never ran has no edge from its input, so no cascade
    // from the input reaches it.
    for creator in creators {
        rt.delete_with_dependents(creator);
    }
    assert_eq!(
        rt.node_count(),
        0,
        "the cascades took every creator, child and reader"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn local_children_match_the_model(ops in prop::collection::vec(op(), 1..60)) {
        run_ops::<Local>(ops);
    }

    #[test]
    fn shared_children_match_the_model(ops in prop::collection::vec(op(), 1..60)) {
        run_ops::<Shared>(ops);
    }
}
