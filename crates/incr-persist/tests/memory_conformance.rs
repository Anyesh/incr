use incr_persist::conformance::Harness;
use incr_persist::MemoryBackend;

struct MemoryHarness;

impl Harness for MemoryHarness {
    type Backend = MemoryBackend;
    type Log = MemoryBackend;

    fn new_log(&mut self) -> MemoryBackend {
        MemoryBackend::new()
    }

    fn open(&mut self, log: &MemoryBackend) -> MemoryBackend {
        log.clone()
    }

    fn open_read_only(&mut self, log: &MemoryBackend) -> Option<MemoryBackend> {
        Some(log.read_only())
    }

    fn multi_writer(&self) -> bool {
        true
    }
}

incr_persist::backend_conformance!(MemoryHarness);
