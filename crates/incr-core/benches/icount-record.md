# icount record

Per-iteration user-space instruction counts from `benches/icount.sh`:
`perf stat -e instructions:u` on the `icount` driver at N iterations,
minus the same run at 0, divided by N, with 100 warm-up iterations in
both. Each task of the spreadsheet-grade plan appends its section here
with, per workload, the count, the delta from the previous section and
the cost sources it runs.

Reading the counts:

- The fan-out and delete workloads rebuild per iteration, so their counts
  include the setup and teardown that criterion leaves untimed: building
  and first-computing the subgraph, and for `fanout_*` and
  `delete_owned_tree_*` the `delete_with_dependents` that tears it down.
- `fanout_*_1000` spends more than half of its count in
  `release_claimed` during teardown, because removing each of the 1000
  leaves from the input's dependents list is a linear `retain`
  (`runtime.rs:916`), so tearing down a fan-in of n is quadratic in n.
- The collection workloads grow their collection by one element per
  iteration, so their counts hold only at the N recorded.
- `delete_owned_tree_*` routes through `HashMap`s with per-process random
  seeds, and `get_contended_shared` (4 readers of 10,000 `get`s each, the
  thread spawns included) depends on scheduling; their spreads are the
  widest and set how far apart two runs of the same binary can land.

## T0 baseline

- Engine: `c61be87` (the bench and driver commits change no engine code).
- Machine: Intel Core i7-9750H (Coffee Lake, 6 cores, 12 threads), Linux
  7.0.0-38-generic, glibc 2.39, perf 7.0.14.
- Toolchain: `rustc 1.96.1 (31fca3adb 2026-06-26)`, `cargo bench` profile
  (release defaults, no workspace overrides).

| workload | N | run 1 | run 2 | spread | spread % |
|---|--:|--:|--:|--:|--:|
| `chain_local/propagate/4` | 1,000,000 | 4,582.01 | 4,582.01 | 0.00 | 0.0000 |
| `chain_local/propagate/10` | 1,000,000 | 10,508.03 | 10,508.02 | 0.01 | 0.0001 |
| `chain_local/propagate/100` | 100,000 | 88,523.18 | 88,523.21 | 0.03 | 0.0000 |
| `chain_shared/propagate/4` | 1,000,000 | 9,601.32 | 9,601.30 | 0.02 | 0.0002 |
| `chain_shared/propagate/10` | 1,000,000 | 21,907.42 | 21,907.40 | 0.02 | 0.0001 |
| `chain_shared/propagate/100` | 100,000 | 191,601.54 | 191,601.86 | 0.32 | 0.0002 |
| `diamond_local` | 1,000,000 | 3,995.00 | 3,995.01 | 0.01 | 0.0003 |
| `diamond_shared` | 1,000,000 | 8,115.63 | 8,115.63 | 0.00 | 0.0000 |
| `early_cutoff_local` | 1,000,000 | 2,284.00 | 2,284.00 | 0.00 | 0.0000 |
| `early_cutoff_shared` | 1,000,000 | 3,693.92 | 3,693.95 | 0.03 | 0.0008 |
| `collection_insert_then_read/local_incremental/1000` | 100,000 | 4,647.53 | 4,647.29 | 0.24 | 0.0052 |
| `collection_insert_then_read/shared_incremental/1000` | 100,000 | 7,500.08 | 7,500.16 | 0.08 | 0.0011 |
| `collection_insert_then_read/batch/1000` | 10,000 | 92,283.98 | 92,284.16 | 0.18 | 0.0002 |
| `collection_insert_then_read/local_incremental/10000` | 100,000 | 4,601.88 | 4,601.95 | 0.07 | 0.0015 |
| `collection_insert_then_read/shared_incremental/10000` | 100,000 | 7,431.86 | 7,431.86 | 0.00 | 0.0000 |
| `collection_insert_then_read/batch/10000` | 1,000 | 158,885.99 | 158,886.77 | 0.78 | 0.0005 |
| `collection_insert_then_read/local_incremental/100000` | 100,000 | 4,653.66 | 4,653.86 | 0.20 | 0.0043 |
| `collection_insert_then_read/shared_incremental/100000` | 100,000 | 7,479.41 | 7,479.34 | 0.07 | 0.0009 |
| `collection_insert_then_read/batch/100000` | 1,000 | 1,490,571.84 | 1,490,569.57 | 2.27 | 0.0002 |
| `simple_count/local/1000` | 100,000 | 2,117.78 | 2,117.65 | 0.13 | 0.0061 |
| `simple_count/shared/1000` | 100,000 | 4,217.92 | 4,217.82 | 0.10 | 0.0024 |
| `simple_count/local/10000` | 100,000 | 2,094.92 | 2,094.81 | 0.11 | 0.0053 |
| `simple_count/shared/10000` | 100,000 | 4,185.43 | 4,185.44 | 0.01 | 0.0002 |
| `get_clean_local` | 10,000,000 | 198.00 | 198.00 | 0.00 | 0.0000 |
| `get_clean_shared` | 10,000,000 | 309.00 | 309.00 | 0.00 | 0.0000 |
| `get_contended_shared` | 2,000 | 13,580,141.65 | 13,637,093.01 | 56,951.36 | 0.4194 |
| `fanout_local_4` | 100,000 | 18,167.06 | 18,167.03 | 0.03 | 0.0002 |
| `fanout_local_5` | 100,000 | 22,590.67 | 22,590.60 | 0.07 | 0.0003 |
| `fanout_local_1000` | 1,000 | 7,348,468.64 | 7,348,463.26 | 5.38 | 0.0001 |
| `fanout_shared_4` | 100,000 | 25,215.22 | 25,215.14 | 0.08 | 0.0003 |
| `fanout_shared_5` | 100,000 | 30,480.76 | 30,480.84 | 0.08 | 0.0003 |
| `fanout_shared_1000` | 1,000 | 8,870,378.99 | 8,870,392.92 | 13.93 | 0.0002 |
| `delete_single_local` | 200,000 | 1,331.00 | 1,331.00 | 0.00 | 0.0000 |
| `delete_single_shared` | 200,000 | 2,139.42 | 2,139.59 | 0.17 | 0.0079 |
| `delete_cascade_local` | 1,000 | 3,251,003.31 | 3,250,998.02 | 5.29 | 0.0002 |
| `delete_cascade_shared` | 1,000 | 4,427,382.89 | 4,427,380.46 | 2.43 | 0.0001 |
| `delete_owned_tree_local` | 10,000 | 602,595.26 | 601,803.35 | 791.91 | 0.1314 |
| `delete_owned_tree_shared` | 10,000 | 1,001,322.01 | 1,002,070.91 | 748.90 | 0.0748 |
