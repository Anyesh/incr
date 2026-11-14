use super::{Options, Store};
use crate::{Cells, Error, Local, MemoryBackend, Record, Shared};

fn a_record_that_fails_to_apply_poisons<C: Cells>() {
    let store: Store<C, _> = Store::open(MemoryBackend::new(), Options::default()).unwrap();
    let x = store.input("x", 0_u32).unwrap();
    let mut staged = store.staged_batch().unwrap();
    staged.set(x, 1).unwrap();
    let bad = Record::Set {
        key: "x".to_owned(),
        value: vec![0xff; 8],
    };
    let failed = {
        let mut state = store.state().unwrap();
        store.apply_staged(&mut state, bad, &mut Vec::new())
    };
    assert!(matches!(failed, Err(Error::TypeMismatch { .. })));
    assert!(matches!(staged.set(x, 2), Err(Error::Poisoned)));
    assert!(matches!(staged.commit(), Err(Error::Poisoned)));
}

#[test]
fn a_record_that_fails_to_apply_poisons_local() {
    a_record_that_fails_to_apply_poisons::<Local>();
}

#[test]
fn a_record_that_fails_to_apply_poisons_shared() {
    a_record_that_fails_to_apply_poisons::<Shared>();
}
