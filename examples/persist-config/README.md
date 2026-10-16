# persist-config

A shop's pricing config and order log kept in an incr-persist file log.
Config entries (`price.<sku>`, `discount_pct`) are persisted inputs, orders
are a persisted collection, and revenue is an incr query over both.

```sh
cargo run -p incr-persist-config -- state set price.apple 120
cargo run -p incr-persist-config -- state set discount_pct 10
cargo run -p incr-persist-config -- state order apple 3
cargo run -p incr-persist-config -- state show --read-only
```

Each command opens the log, replays it, applies one change and prints the
result. One process at a time can open the log for writing; `show
--read-only` takes no lock and works while a writer is running.

`churn <from>` writes a fixed workload until killed. `tests/sigkill.rs`
uses it to kill a writer with SIGKILL mid-write many times on one log and
check that every acknowledged write survives, that revenue matches what
the writer reported, and that a read-only process sees the same state.
The test runs on Unix only.
