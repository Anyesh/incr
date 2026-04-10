use incr_compute::{Incr, Runtime};
use proptest::prelude::*;

/// Build a layered graph of the given shape, run it incrementally,
/// then rebuild from scratch and compare results.
fn verify_incremental_matches_batch(
    num_inputs: usize,
    input_values: Vec<i64>,
    layers: Vec<Vec<(usize, usize)>>, // Each layer: vec of (dep_a_idx, dep_b_idx) pairs
    mutations: Vec<(usize, i64)>,     // (input_index, new_value) pairs
) {
    assert!(num_inputs >= 2);
    assert_eq!(input_values.len(), num_inputs);

    let rt = Runtime::new();
    let mut all_nodes: Vec<Incr<i64>> = Vec::new();

    for &val in &input_values {
        let node = rt.create_input(val);
        all_nodes.push(node);
    }

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
            let node = rt.create_query(move |rt| rt.get(a).wrapping_add(rt.get(b)));
            layer_nodes.push(node);
        }
        all_nodes.extend(layer_nodes);
    }

    if all_nodes.len() <= num_inputs {
        return; // No compute nodes generated
    }

    // Read all compute nodes to initialize
    let last = *all_nodes.last().unwrap();
    let _ = rt.get(last);

    // Apply mutations
    for &(input_rel, new_val) in &mutations {
        let idx = input_rel % num_inputs;
        rt.set(all_nodes[idx], new_val);
    }

    let incremental_result = rt.get(last);

    let mut final_values = input_values.clone();
    for &(input_rel, new_val) in &mutations {
        let idx = input_rel % num_inputs;
        final_values[idx] = new_val;
    }

    let rt2 = Runtime::new();
    let mut all_nodes2: Vec<Incr<i64>> = Vec::new();

    for &val in &final_values {
        let node = rt2.create_input(val);
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
            let node = rt2.create_query(move |rt| rt.get(a).wrapping_add(rt.get(b)));
            layer_nodes.push(node);
        }
        all_nodes2.extend(layer_nodes);
    }

    let last2 = *all_nodes2.last().unwrap();
    let batch_result = rt2.get(last2);

    assert_eq!(
        incremental_result,
        batch_result,
        "Incremental result {} != batch result {} with {} inputs, {} layers, {} mutations",
        incremental_result,
        batch_result,
        num_inputs,
        layers.len(),
        mutations.len()
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn incremental_matches_batch(
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

#[test]
fn property_specific_diamond_cutoff() {
    verify_incremental_matches_batch(
        3,
        vec![10, 20, 30],
        vec![
            vec![(0, 1), (1, 2)], // Layer 1: node3=in0+in1, node4=in1+in2
            vec![(0, 1)],         // Layer 2: node5=node3+node4
        ],
        vec![(0, 10), (1, 25)], // Change input 0 (same!), change input 1
    );
}

#[test]
fn property_deep_chain() {
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
