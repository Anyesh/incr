use incr_persist::memo::{conformance::MemoHarness, MemoryMemo};

struct MemoryHarness;

impl MemoHarness for MemoryHarness {
    type Backend = MemoryMemo;
    type Store = MemoryMemo;

    fn new_store(&mut self) -> MemoryMemo {
        MemoryMemo::new()
    }

    fn open(&mut self, store: &MemoryMemo) -> MemoryMemo {
        store.clone()
    }

    fn exclusive_lock(&self) -> bool {
        true
    }
}

incr_persist::memo_backend_conformance!(MemoryHarness);
