//! `Runtime::memory_stats`: the walk over nodes and registered state.

use std::any::TypeId;
use std::sync::Arc;

use crate::cells::{Cells, Local, Shared};
use crate::collection::IncrCollection;
use crate::memory::{Held, MemoryStats, NodeKind, StateKind, StateMemory, StateStats};
use crate::runtime::Runtime;

macro_rules! both {
    ($($name:ident),* $(,)?) => {
        mod local {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Local>();
                }
            )*
        }
        mod shared {
            $(
                #[test]
                fn $name() {
                    super::$name::<super::Shared>();
                }
            )*
        }
    };
}

both!(
    memory_stats_grow_with_nodes,
    memory_stats_shrink_after_delete_with_dependents,
    memory_stats_count_collection_and_operator_state,
    memory_stats_prune_dead_weaks,
    memory_stats_registry_stays_bounded_without_walks,
    memory_stats_report_registered_external_state,
);

fn kind_count(stats: &MemoryStats, kind: NodeKind) -> usize {
    stats.nodes.iter().filter(|n| n.kind == kind).count()
}

fn states_of(stats: &MemoryStats, kind: StateKind) -> Vec<&StateMemory> {
    stats.states.iter().filter(|s| s.kind == kind).collect()
}

fn memory_stats_grow_with_nodes<C: Cells>() {
    let rt = Runtime::<C>::new();
    let before = rt.memory_stats();
    let inputs: Vec<_> = (0..10_u64).map(|i| rt.create_input(i)).collect();
    let queries: Vec<_> = inputs
        .iter()
        .map(|&i| rt.create_query(move |rt| rt.get(i) + 1))
        .collect();
    for &q in &queries {
        rt.get(q);
    }
    rt.set_label(queries[0].slot(), "pricing/total".to_string());
    let after = rt.memory_stats();

    assert_eq!(after.nodes.len(), before.nodes.len() + 20);
    assert_eq!(kind_count(&after, NodeKind::Input), 10);
    assert_eq!(kind_count(&after, NodeKind::Query), 10);
    assert!(after.bytes() > before.bytes());
    assert_eq!(after.held(TypeId::of::<u64>()), 20);

    let q = after
        .nodes
        .iter()
        .find(|n| n.slot == queries[0].slot())
        .expect("the query is walked");
    assert!(q.has_compute);
    assert!(q.has_value);
    assert_eq!(q.label.as_deref(), Some("pricing/total"));
    assert_eq!(
        after.value_type(q.type_tag).map(|t| t.type_id),
        Some(TypeId::of::<u64>())
    );
    let plain = after
        .nodes
        .iter()
        .find(|n| n.slot == queries[1].slot())
        .expect("the query is walked");
    assert!(q.bytes > plain.bytes, "the label's bytes count");
}

fn memory_stats_shrink_after_delete_with_dependents<C: Cells>() {
    let rt = Runtime::<C>::new();
    let root = rt.create_input(1_u64);
    let mid = rt.create_query(move |rt| rt.get(root) * 2);
    let top = rt.create_query(move |rt| rt.get(mid) + 1);
    let other = rt.create_input(7_u64);
    rt.get(top);
    let before = rt.memory_stats();

    assert_eq!(rt.delete_with_dependents(root), 3);
    let after = rt.memory_stats();

    assert_eq!(after.nodes.len(), before.nodes.len() - 3);
    assert!(after.bytes() < before.bytes());
    assert_eq!(after.held(TypeId::of::<u64>()), 1);
    assert!(after.nodes.iter().any(|n| n.slot == other.slot()));
    assert_eq!(after.free_slots, 3);
}

fn memory_stats_count_collection_and_operator_state<C: Cells>() {
    let rt = Runtime::<C>::new();
    let col: IncrCollection<u64, C> = rt.create_collection();
    let other: IncrCollection<u64, C> = rt.create_collection();
    col.extend(&rt, 0..100_u64);
    other.extend(&rt, 0..10_u64);

    let grouped = col.group_by(&rt, |v| v % 4);
    let total = col.aggregate(&rt, 0_u64, |v| *v, |a, b| a + b);
    let joined = col.join(&rt, &other, |v| *v, |v| *v);
    let sorted = col.sort_by_key(&rt, |v| *v);
    let pairs = sorted.pairwise(&rt);
    rt.get(grouped.version_node());
    rt.get(total);
    rt.get(joined.version_node());
    rt.get(pairs.version_node());

    let stats = rt.memory_stats();
    let source = stats
        .states
        .iter()
        .find(|s| s.node == Some(crate::node::NodeId(col.version_node().slot())))
        .expect("the source collection is registered with its version node");
    assert_eq!(source.kind, StateKind::Collection);
    assert_eq!(source.elements, 100);
    assert!(source.bytes > 0);
    assert!(source
        .held
        .contains(&Held::of::<u64>(100 + source.pending_deltas)));

    let groups = states_of(&stats, StateKind::GroupBy);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].elements, 4);
    assert_eq!(
        groups[0].node,
        Some(crate::node::NodeId(grouped.version_node().slot()))
    );
    assert_eq!(states_of(&stats, StateKind::KeyCache)[0].elements, 100);
    assert_eq!(states_of(&stats, StateKind::Aggregate)[0].elements, 100);
    assert_eq!(
        states_of(&stats, StateKind::Aggregate)[0].node,
        Some(crate::node::NodeId(total.slot()))
    );
    assert_eq!(states_of(&stats, StateKind::Join)[0].elements, 110);
    assert_eq!(states_of(&stats, StateKind::Sort)[0].elements, 100);
    assert_eq!(states_of(&stats, StateKind::SortMirror)[0].elements, 100);

    // The source, `other`, four groups, and the join, pairwise outputs.
    let collections = states_of(&stats, StateKind::Collection);
    assert_eq!(collections.len(), 8);
    assert_eq!(kind_count(&stats, NodeKind::Collection), 8);
    let group_elements: usize = collections
        .iter()
        .filter(|s| s.held.iter().any(|h| h.type_id == TypeId::of::<u64>()))
        .map(|s| s.elements)
        .sum();
    assert_eq!(group_elements, 100 + 10 + 100);
    assert!(stats.held(TypeId::of::<u64>()) >= 100 * 5);
    assert!(stats.state_bytes() > 0);
    assert!(stats.bytes() > stats.state_bytes());
}

fn memory_stats_prune_dead_weaks<C: Cells>() {
    let rt = Runtime::<C>::new();
    let kept: IncrCollection<u64, C> = rt.create_collection();
    {
        let dropped: IncrCollection<u64, C> = rt.create_collection();
        dropped.extend(&rt, 0..5_u64);
        rt.delete_node(dropped.version_node());
    }
    assert_eq!(rt.registered_states(), 2);
    let stats = rt.memory_stats();
    assert_eq!(states_of(&stats, StateKind::Collection).len(), 1);
    assert_eq!(rt.registered_states(), 1);
    drop(kept);
}

fn memory_stats_registry_stays_bounded_without_walks<C: Cells>() {
    let rt = Runtime::<C>::new();
    for _ in 0..10_000 {
        let col: IncrCollection<u64, C> = rt.create_collection();
        rt.delete_node(col.version_node());
    }
    assert!(
        rt.registered_states() <= 128,
        "dead registrations pile up: {}",
        rt.registered_states()
    );
}

struct External(usize);

impl StateStats for External {
    fn state_stats(&self) -> StateMemory {
        StateMemory::new(
            StateKind::Other("external"),
            self.0,
            vec![Held::of::<String>(self.0)],
            self.0 * 24,
        )
    }
}

fn memory_stats_report_registered_external_state<C: Cells>() {
    let rt = Runtime::<C>::new();
    let anchor = rt.create_input(0_u64);
    let state: Arc<External> = Arc::new(External(3));
    let weak = Arc::downgrade(&state);
    rt.register_state(Some(anchor.erase()), weak);
    let stats = rt.memory_stats();
    let ext = states_of(&stats, StateKind::Other("external"));
    assert_eq!(ext.len(), 1);
    assert_eq!(ext[0].node, Some(crate::node::NodeId(anchor.slot())));
    assert_eq!(stats.held(TypeId::of::<String>()), 3);

    rt.delete_node(anchor);
    let stats = rt.memory_stats();
    assert_eq!(
        states_of(&stats, StateKind::Other("external"))[0].node,
        None
    );

    drop(state);
    let stats = rt.memory_stats();
    assert!(states_of(&stats, StateKind::Other("external")).is_empty());
}

#[test]
fn shared_memory_stats_survives_concurrent_collection_creation() {
    let rt = Runtime::<Shared>::new();
    let live_creators = std::sync::atomic::AtomicUsize::new(4);
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                let mut kept = Vec::new();
                for _ in 0..5000 {
                    let col: IncrCollection<u64, Shared> = rt.create_collection();
                    kept.push(col);
                }
                live_creators.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
            });
        }
        for _ in 0..4 {
            s.spawn(|| {
                while live_creators.load(std::sync::atomic::Ordering::Acquire) > 0 {
                    rt.memory_stats();
                }
            });
        }
    });
}
