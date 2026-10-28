# incr-persist

Durable, shareable state for [incr](https://github.com/Anyesh/incr) runtimes. A `Store` owns an incr runtime and keeps its inputs in sync with an ordered operation log, so a restarted process, or several processes sharing the log, see the same inputs and compute the same results.

The log lives behind the `Backend` trait: a totally ordered sequence of records plus one snapshot that folds a prefix of it. The crate ships three backends:

- `FileBackend`: a log in a directory, with checksummed records that crash recovery sees whole or not at all, one writer process at a time, and any number of read-only processes following it.
- `MemoryBackend`: an in-process log several writers can share.
- `SqliteBackend` (feature `sqlite`): the log in a SQLite database, which several writer processes can share.

The `backend-tests` feature exports `conformance`, the suite every backend must pass, so a new backend can be checked against the same contract.

## Install

```
cargo add incr-persist
```

## Quick start

```rust
use incr_persist::{FileBackend, Local, Options, Store};

let store: Store<Local, FileBackend> =
    Store::open(FileBackend::open("data/log")?, Options::default())?;
let price = store.input("price", 100u64)?;
let qty = store.input("qty", 1u64)?;

let rt = store.runtime();
let (p, q) = (price.handle(), qty.handle());
let total = rt.create_query(move |rt| rt.get(p) * rt.get(q));

store.set(qty, 3)?; // logged, then applied to the runtime
assert_eq!(rt.get(total), 300);
// A process that opens "data/log" later starts with qty at 3.
```

`Store::batch` appends several sets and collection changes with one sync, `Store::refresh` applies records other writers appended, and `PersistedCollection` persists an incr collection the same way. [`examples/persist-config`](https://github.com/Anyesh/incr/tree/main/examples/persist-config) is a complete program with a revenue query over a persisted config and order log.

## License

Apache-2.0
