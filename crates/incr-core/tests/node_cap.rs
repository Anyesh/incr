//! A runtime holds more nodes than the old fixed segment directory
//! could address (1024 segments of 1024 nodes).
#![cfg(not(miri))]

use incr_core::{Cells, Local, Runtime, Shared};

const PAST_THE_OLD_CAP: u64 = 1_100_000;

fn creates_and_reads_back<C: Cells>() {
    let rt: Runtime<C> = Runtime::new();
    let handles: Vec<_> = (0..PAST_THE_OLD_CAP).map(|k| rt.create_input(k)).collect();
    for (k, h) in handles.into_iter().enumerate() {
        assert_eq!(rt.get(h), k as u64);
    }
    assert_eq!(rt.node_count(), PAST_THE_OLD_CAP as usize);
}

#[test]
fn local_holds_past_the_old_node_cap() {
    creates_and_reads_back::<Local>();
}

#[test]
fn shared_holds_past_the_old_node_cap() {
    creates_and_reads_back::<Shared>();
}
