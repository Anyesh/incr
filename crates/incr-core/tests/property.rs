//! Property-test suite for `incr-core`. Generates random function-DAG
//! graphs, applies random mutations, and asserts that the incremental
//! result matches the batch-recompute result.
//!
//! The same generators and verifier run under both `Local` and `Shared`
//! strategies. Each strategy gets its own proptest! block so failures
//! shrink in the right type context.

use incr_core::{Incr, Local, Runtime, Shared};
use proptest::prelude::*;

/// Verify that an incremental run on a randomly-shaped graph produces
/// the same final values as a from-scratch rebuild with the mutated
/// inputs in place.
fn verify_incremental_matches_batch<C: incr_core::Cells>(
    num_inputs: usize,
    input_values: Vec<i64>,
    layers: Vec<Vec<(usize, usize)>>,
    mutations: Vec<(usize, i64)>,
) where
    Runtime<C>: Default,
{
    assert!(num_inputs >= 2);
    assert_eq!(input_values.len(), num_inputs);

    // Pass 1: incremental.
    let rt: Runtime<C> = Runtime::new();
    let mut all_nodes: Vec<Incr<i64>> = Vec::new();
    for &v in &input_values {
        all_nodes.push(rt.create_input(v));
    }
    for layer in &layers {
        let mut layer_nodes = Vec::new();
        for &(a_rel, b_rel) in layer {
            let avail = all_nodes.len();
            if avail < 2 {
                continue;
            }
            let a = all_nodes[a_rel % avail];
            let b = all_nodes[b_rel % avail];
            layer_nodes.push(rt.create_query(move |rt| rt.get(a).wrapping_add(rt.get(b))));
        }
        all_nodes.extend(layer_nodes);
    }
    if all_nodes.len() <= num_inputs {
        return;
    }
    let last = *all_nodes.last().unwrap();
    let _ = rt.get(last);

    for &(input_rel, new_val) in &mutations {
        let idx = input_rel % num_inputs;
        rt.set(all_nodes[idx], new_val);
    }
    let incremental_result = rt.get(last);

    // Pass 2: batch rebuild with the mutated input values baked in.
    let mut final_values = input_values.clone();
    for &(input_rel, new_val) in &mutations {
        let idx = input_rel % num_inputs;
        final_values[idx] = new_val;
    }

    let rt2: Runtime<C> = Runtime::new();
    let mut all_nodes2: Vec<Incr<i64>> = Vec::new();
    for &v in &final_values {
        all_nodes2.push(rt2.create_input(v));
    }
    for layer in &layers {
        let mut layer_nodes = Vec::new();
        for &(a_rel, b_rel) in layer {
            let avail = all_nodes2.len();
            if avail < 2 {
                continue;
            }
            let a = all_nodes2[a_rel % avail];
            let b = all_nodes2[b_rel % avail];
            layer_nodes.push(rt2.create_query(move |rt| rt.get(a).wrapping_add(rt.get(b))));
        }
        all_nodes2.extend(layer_nodes);
    }
    let last2 = *all_nodes2.last().unwrap();
    let batch_result = rt2.get(last2);

    assert_eq!(
        incremental_result,
        batch_result,
        "Incremental {} != batch {} with {} inputs, {} layers, {} mutations (strategy = {})",
        incremental_result,
        batch_result,
        num_inputs,
        layers.len(),
        mutations.len(),
        std::any::type_name::<C>(),
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn local_incremental_matches_batch(
        num_inputs in 2_usize..16,
        input_values in prop::collection::vec(-1000_i64..1000, 2..16),
        layers in prop::collection::vec(
            prop::collection::vec((0_usize..100, 0_usize..100), 1..5),
            1..6,
        ),
        mutations in prop::collection::vec((0_usize..100, -1000_i64..1000), 1..15),
    ) {
        let num_inputs = num_inputs.min(input_values.len()).max(2);
        let input_values = input_values[..num_inputs].to_vec();
        verify_incremental_matches_batch::<Local>(
            num_inputs,
            input_values,
            layers,
            mutations,
        );
    }

    #[test]
    fn shared_incremental_matches_batch(
        num_inputs in 2_usize..16,
        input_values in prop::collection::vec(-1000_i64..1000, 2..16),
        layers in prop::collection::vec(
            prop::collection::vec((0_usize..100, 0_usize..100), 1..5),
            1..6,
        ),
        mutations in prop::collection::vec((0_usize..100, -1000_i64..1000), 1..15),
    ) {
        let num_inputs = num_inputs.min(input_values.len()).max(2);
        let input_values = input_values[..num_inputs].to_vec();
        verify_incremental_matches_batch::<Shared>(
            num_inputs,
            input_values,
            layers,
            mutations,
        );
    }
}

#[test]
fn regression_diamond_with_cutoff() {
    verify_incremental_matches_batch::<Local>(
        3,
        vec![10, 20, 30],
        vec![vec![(0, 1), (1, 2)], vec![(0, 1)]],
        vec![(0, 10), (1, 25)],
    );
    verify_incremental_matches_batch::<Shared>(
        3,
        vec![10, 20, 30],
        vec![vec![(0, 1), (1, 2)], vec![(0, 1)]],
        vec![(0, 10), (1, 25)],
    );
}

#[test]
fn regression_deep_chain() {
    verify_incremental_matches_batch::<Local>(
        5,
        vec![1, 2, 3, 4, 5],
        vec![vec![(0, 1)], vec![(2, 0)], vec![(0, 1)], vec![(1, 0)]],
        vec![(0, 100), (2, 50), (4, 75)],
    );
    verify_incremental_matches_batch::<Shared>(
        5,
        vec![1, 2, 3, 4, 5],
        vec![vec![(0, 1)], vec![(2, 0)], vec![(0, 1)], vec![(1, 0)]],
        vec![(0, 100), (2, 50), (4, 75)],
    );
}
