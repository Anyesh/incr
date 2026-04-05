//! Proptest suite for the v2 Runtime.
//!
//! Mirrors `crates/incr-core/tests/property.rs` (which targets v1)
//! but runs against `v2::Runtime` through crate-private access. The
//! goal is spec Gate 2: v2 passes the same property tests as v1 in
//! single-threaded mode, establishing correctness equivalence
//! between the two engines for every case the proptest suite can
//! generate.
//!
//! Why in-crate rather than in `tests/`: the v2 module is
//! `pub(crate)` until spec Gate 5, so external integration tests
//! cannot see `v2::Runtime` or `v2::Incr`. Putting this proptest in
//! a `#[cfg(test)]` submodule of `v2/` gives it direct crate-
//! private access without exposing the v2 API publicly before it
//! is ready.
//!
//! Collection operator proptests (`tests/collection_property.rs`)
//! are deliberately not ported here. The collection API
//! (`IncrCollection`, filter/map/count/reduce/sort/pairwise) is a
//! separate piece of work that will be rewritten against v2 in its
//! own chunk per the spec's section 3 scope notes.

use super::handle::Incr;
use super::runtime::Runtime;
use proptest::prelude::*;

/// Build a layered graph of the given shape, run it incrementally,
/// then rebuild from scratch and compare results.
///
/// This is the same function body as `tests/property.rs::verify_
/// incremental_matches_batch`, mechanically ported to use
/// `v2::Runtime` and `v2::Incr<i64>`. The core correctness
/// contract (incremental result equals batch recomputation result
/// for every mutation sequence) is the property the proptest
/// proves across thousands of generated graph shapes.
fn verify_incremental_matches_batch(
    num_inputs: usize,
    input_values: Vec<i64>,
    layers: Vec<Vec<(usize, usize)>>, // Each layer: vec of (dep_a_idx, dep_b_idx) pairs
    mutations: Vec<(usize, i64)>,     // (input_index, new_value) pairs
) {
    assert!(num_inputs >= 2);
    assert_eq!(input_values.len(), num_inputs);

    // --- Build incremental graph ---
    let rt = Runtime::new();
    let mut all_nodes: Vec<Incr<i64>> = Vec::new();

    // Create inputs.
    for &val in &input_values {
        let node = rt.create_input::<i64>(val);
        all_nodes.push(node);
    }

    // Create compute layers. Each query sums two existing nodes via
    // indices into the running `all_nodes` list, with modular
    // indexing so proptest-generated offsets always pick valid
    // predecessors.
    for layer in &layers {
        let mut layer_nodes = Vec::new();
        for &(dep_a_rel, dep_b_rel) in layer {
            let available = all_nodes.len();
            if available < 2 {
                continue;
            }
            let idx_a = dep_a_rel % available;
            let idx_b = dep_b_rel % available;
            let a = all_nodes[idx_a];
            let b = all_nodes[idx_b];
            let node = rt.create_query::<i64, _>(move |rt| rt.get(a).wrapping_add(rt.get(b)));
            layer_nodes.push(node);
        }
        all_nodes.extend(layer_nodes);
    }

    if all_nodes.len() <= num_inputs {
        return; // No compute nodes generated
    }

    // Read all compute nodes once to force the initial compute.
    let last = *all_nodes.last().unwrap();
    let _ = rt.get(last);

    // Apply mutations. Each mutation flips an input value; the
    // dirty walk propagates and subsequent reads trigger recomputes
    // with early cutoff on any node whose value happens to be
    // unchanged.
    for &(input_rel, new_val) in &mutations {
        let idx = input_rel % num_inputs;
        rt.set(all_nodes[idx], new_val);
    }

    // Get incremental result.
    let incremental_result = rt.get(last);

    // --- Build batch (from scratch) with final input values ---
    let mut final_values = input_values.clone();
    for &(input_rel, new_val) in &mutations {
        let idx = input_rel % num_inputs;
        final_values[idx] = new_val;
    }

    let rt2 = Runtime::new();
    let mut all_nodes2: Vec<Incr<i64>> = Vec::new();

    for &val in &final_values {
        let node = rt2.create_input::<i64>(val);
        all_nodes2.push(node);
    }

    for layer in &layers {
        let mut layer_nodes = Vec::new();
        for &(dep_a_rel, dep_b_rel) in layer {
            let available = all_nodes2.len();
            if available < 2 {
                continue;
            }
            let idx_a = dep_a_rel % available;
            let idx_b = dep_b_rel % available;
            let a = all_nodes2[idx_a];
            let b = all_nodes2[idx_b];
            let node = rt2.create_query::<i64, _>(move |rt| rt.get(a).wrapping_add(rt.get(b)));
            layer_nodes.push(node);
        }
        all_nodes2.extend(layer_nodes);
    }

    let last2 = *all_nodes2.last().unwrap();
    let batch_result = rt2.get(last2);

    // --- The critical assertion ---
    assert_eq!(
        incremental_result,
        batch_result,
        "v2: Incremental result {} != batch result {} with {} inputs, {} layers, {} mutations",
        incremental_result,
        batch_result,
        num_inputs,
        layers.len(),
        mutations.len()
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// The main correctness property: for every generated graph
    /// shape and mutation sequence, v2's incremental result must
    /// equal a from-scratch rebuild with the final input values.
    /// This covers thousands of random dep topologies (diamonds,
    /// chains, wide fan-out, wide fan-in, mixed) and checks the
    /// full commit H-M stack: dep tracking, dirty walk, early
    /// cutoff, and dynamic dep updates.
    #[test]
    fn v2_incremental_matches_batch(
        num_inputs in 2_usize..20,
        input_values in prop::collection::vec(-1000_i64..1000, 2..20),
        layers in prop::collection::vec(
            prop::collection::vec((0_usize..100, 0_usize..100), 1..5),
            1..8
        ),
        mutations in prop::collection::vec((0_usize..100, -1000_i64..1000), 1..20),
    ) {
        let num_inputs = num_inputs.min(input_values.len()).max(2);
        let input_values = input_values[..num_inputs].to_vec();
        verify_incremental_matches_batch(num_inputs, input_values, layers, mutations);
    }
}

/// Specific regression case from the v1 proptest suite: a shallow
/// diamond where one mutation is a no-op (same value) and another
/// actually changes. Exercises the early cutoff fast path combined
/// with a live dirty walk. This case was originally a shrunk
/// failure from the v1 proptest; keeping the concrete case as a
/// named test catches regressions without waiting for proptest to
/// re-discover the shape.
#[test]
fn v2_property_specific_diamond_cutoff() {
    verify_incremental_matches_batch(
        3,
        vec![10, 20, 30],
        vec![
            vec![(0, 1), (1, 2)], // Layer 1: node3=in0+in1, node4=in1+in2
            vec![(0, 1)],         // Layer 2: node5=node3+node4
        ],
        vec![(0, 10), (1, 25)], // Change input 0 to same value (no-op), change input 1
    );
}

/// Deeper chain regression case from the v1 proptest suite. Ten
/// layers, each a single query that sums two earlier nodes. Three
/// mutations at different depths exercise the dirty walk's
/// transitive reach.
#[test]
fn v2_property_deep_chain() {
    verify_incremental_matches_batch(
        5,
        vec![1, 2, 3, 4, 5],
        vec![
            vec![(0, 1)],
            vec![(2, 0)],
            vec![(0, 1)],
            vec![(1, 0)],
            vec![(0, 1)],
            vec![(2, 0)],
            vec![(0, 1)],
            vec![(1, 0)],
            vec![(0, 1)],
            vec![(2, 0)],
        ],
        vec![(0, 100), (2, 50), (4, 75)],
    );
}
