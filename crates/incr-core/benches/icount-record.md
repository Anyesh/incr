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

## T1 (final)

- Engine: `951ec82` (branch `t1-review-fixes`, the same tree as
  `spreadsheet-t1`). Its last engine change is `289c1a5`; `951ec82`
  changes only comments. Machine and toolchain as for T0.
- T0 is the T0 driver (`3d58829`) run again in the same session as T1;
  both columns are the mean of two runs of `icount.sh` at the default N.
  The rerun is within 0.003% of the T0 baseline above, except
  `delete_owned_tree_*` (up to 0.05%, seeds) and `get_contended_shared`
  (7.5% lower, scheduling on a loaded machine). The two T1 runs differ by
  at most 0.016%, except `delete_owned_tree_local` (0.24%) and
  `get_contended_shared` (4.1%).
- Attribution: Intel PT on line-table builds of both drivers
  (`CARGO_PROFILE_BENCH_DEBUG=line-tables-only`; their counts equal the
  release drivers' to within 1 per iteration), every instruction decoded,
  N iterations minus 0, divided by N, per symbol, with inlined code counted
  in its caller. N is about 8 million instructions' worth per workload;
  the collection workloads were traced at their recorded N with one PT
  sample every 100 instructions, and `delete_owned_tree_*` at N = 200.
- "alloc" in the sources column is the recorded delta minus the engine's
  share, where the engine is every symbol outside glibc, the Rust
  allocator shims, `RawVec` growth and haphazard. Under Shared it carries
  the haphazard reclamation and glibc work that changes with N, so it is
  taken at the recorded N. Most of the negative alloc on the chains is
  T0's walk vector growing through `realloc`; T1 grows it once through
  `Walk::grow` to 16 entries ("walk growth", engine code) and then
  doubles.

| workload | N | T0 | T1 | delta | delta % | sources |
|---|--:|--:|--:|--:|--:|---|
| `chain_local/propagate/4` | 1,000,000 | 4,582.00 | 4,849.00 | 267.00 | +5.83 | set 62, slow 40, walk growth 62, 4 × step 88, alloc -248 |
| `chain_local/propagate/10` | 1,000,000 | 10,508.00 | 10,519.00 | 11.00 | +0.10 | set 62, slow 40, walk growth 62, 10 × step 86, alloc -1,014 |
| `chain_local/propagate/100` | 100,000 | 88,523.03 | 96,328.02 | 7,804.99 | +8.82 | set 62, slow 40, walk growth 248, 100 × step 86, alloc -1,152 |
| `chain_shared/propagate/4` | 1,000,000 | 9,601.28 | 9,934.42 | 333.14 | +3.47 | set 38, slow 44, walk growth 62, 4 × step 86, alloc -154 |
| `chain_shared/propagate/10` | 1,000,000 | 21,907.32 | 21,700.28 | -207.03 | -0.95 | set 38, slow 44, walk growth 62, 10 × step 84, alloc -1,192 |
| `chain_shared/propagate/100` | 100,000 | 191,600.90 | 199,196.11 | 7,595.20 | +3.96 | set 38, slow 44, walk growth 248, 100 × step 84, alloc -1,142 |
| `diamond_local` | 1,000,000 | 3,995.00 | 4,160.00 | 165.00 | +4.13 | set 62, slow 40, walk growth 62, 3 × step 83, alloc -248 |
| `diamond_shared` | 1,000,000 | 8,115.61 | 8,346.12 | 230.51 | +2.84 | set 38, slow 44, walk growth 62, 3 × step 88, alloc -177 |
| `early_cutoff_local` | 1,000,000 | 2,284.00 | 2,350.00 | 66.00 | +2.89 | set 62, slow 40, walk growth 62, recompute and verify-Clean steps 150, alloc -248 |
| `early_cutoff_shared` | 1,000,000 | 3,693.93 | 3,743.99 | 50.07 | +1.36 | set 38, slow 44, walk growth 62, recompute and verify-Clean steps 141, alloc -235 |
| `collection_insert_then_read/local_incremental/1000` | 100,000 | 4,647.55 | 4,645.65 | -1.90 | -0.04 | set 62, slow 40, walk growth 62, 3 steps 266, collection -183, alloc -249 |
| `collection_insert_then_read/shared_incremental/1000` | 100,000 | 7,500.08 | 7,507.92 | 7.84 | +0.10 | set 38, slow 44, walk growth 62, 3 steps 255, collection -184, alloc -207 |
| `collection_insert_then_read/batch/1000` | 10,000 | 92,283.73 | 92,283.26 | -0.47 | -0.00 | none |
| `collection_insert_then_read/local_incremental/10000` | 100,000 | 4,601.88 | 4,639.21 | 37.33 | +0.81 | set 62, slow 40, walk growth 63, 3 steps 264, collection -144, alloc -247 |
| `collection_insert_then_read/shared_incremental/10000` | 100,000 | 7,431.78 | 7,484.15 | 52.37 | +0.70 | set 38, slow 44, walk growth 62, 3 steps 255, collection -146, alloc -200 |
| `collection_insert_then_read/batch/10000` | 1,000 | 158,885.84 | 158,886.08 | 0.23 | +0.00 | none |
| `collection_insert_then_read/local_incremental/100000` | 100,000 | 4,653.55 | 4,647.55 | -5.99 | -0.13 | set 62, slow 40, walk growth 62, 3 steps 266, collection -190, alloc -246 |
| `collection_insert_then_read/shared_incremental/100000` | 100,000 | 7,479.60 | 7,489.05 | 9.44 | +0.13 | set 38, slow 44, walk growth 62, 3 steps 252, collection -187, alloc -200 |
| `collection_insert_then_read/batch/100000` | 1,000 | 1,490,571.21 | 1,490,570.31 | -0.90 | -0.00 | none |
| `simple_count/local/1000` | 100,000 | 2,117.65 | 2,191.00 | 73.35 | +3.46 | set 62, slow 40, 1 step 86, collection -115, alloc 0 |
| `simple_count/shared/1000` | 100,000 | 4,217.93 | 4,282.95 | 65.02 | +1.54 | set 38, slow 44, 1 step 91, collection -117, alloc 9 |
| `simple_count/local/10000` | 100,000 | 2,094.82 | 2,187.23 | 92.41 | +4.41 | set 62, slow 40, 1 step 87, collection -96, alloc 0 |
| `simple_count/shared/10000` | 100,000 | 4,185.27 | 4,264.98 | 79.70 | +1.90 | set 38, slow 44, 1 step 89, collection -98, alloc 7 |
| `get_clean_local` | 10,000,000 | 198.00 | 178.00 | -20.00 | -10.10 | get -20 |
| `get_clean_shared` | 10,000,000 | 309.00 | 302.00 | -7.00 | -2.27 | get -7 |
| `get_contended_shared` | 2,000 | 12,587,106.04 | 12,515,381.03 | -71,725.01 | -0.57 | get; within the T1 runs' spread (scheduling) |
| `fanout_local_4` | 100,000 | 18,167.01 | 19,522.00 | 1,354.99 | +7.46 | 5 × first 96, 6 × delete 146, alloc 0 |
| `fanout_local_5` | 100,000 | 22,590.58 | 24,186.50 | 1,595.92 | +7.06 | 6 × first 97, 7 × delete 145, alloc -1 |
| `fanout_local_1000` | 1,000 | 7,348,001.73 | 7,590,807.71 | 242,805.98 | +3.30 | 1,001 × first 98, 1,002 × delete 145, alloc -634 |
| `fanout_shared_4` | 100,000 | 25,215.10 | 26,509.11 | 1,294.01 | +5.13 | 5 × first 87, 6 × delete 144, alloc 0 |
| `fanout_shared_5` | 100,000 | 30,480.67 | 32,005.07 | 1,524.40 | +5.00 | 6 × first 87, 7 × delete 143, alloc -4 |
| `fanout_shared_1000` | 1,000 | 8,870,366.36 | 9,101,544.88 | 231,178.52 | +2.61 | 1,001 × first 91, 1,002 × delete 142, alloc -2,230 |
| `delete_single_local` | 200,000 | 1,331.00 | 1,349.00 | 18.00 | +1.35 | strict 158, alloc -140 |
| `delete_single_shared` | 200,000 | 2,139.43 | 2,158.24 | 18.81 | +0.88 | strict 166, alloc -147 |
| `delete_cascade_local` | 1,000 | 3,250,990.79 | 3,493,116.82 | 242,126.03 | +7.45 | 1,000 × first 96, 1,001 × delete 146, alloc 2 |
| `delete_cascade_shared` | 1,000 | 4,427,362.26 | 4,660,485.05 | 233,122.80 | +5.27 | 1,000 × first 91, 1,001 × delete 142, alloc 0 |
| `delete_owned_tree_local` | 10,000 | 602,112.15 | 644,420.94 | 42,308.80 | +7.03 | 200 × set 62, collection -362, owned-tree delete and other 22,279, alloc 7,992 |
| `delete_owned_tree_shared` | 10,000 | 1,002,226.00 | 1,032,661.26 | 30,435.26 | +3.04 | 200 × set 38, collection 5,748, owned-tree delete and other 16,554, alloc 534 |

Measured units, Local / Shared, in instructions against T0:

- **set**, per value-changing `set`: +62 / +38 (Local: the `set` symbol,
  119 at T0 and 181 at T1; Shared: `set` +70 and its slot work, where
  `compare_and_write` and one `write_if_changed` session, 141, replace
  T0's `eq_current` and `write`, 173). Inside it at T1: the claim's take
  (a load, the flag test and a CAS on `verified_at`) and release (one
  `fetch_add`) 13 and 13; the set-stack push and pop 34 and 35 inside
  `set_stack`; the
  `VALUE_CODE` check and the wrap around the one `PartialEq` call; the
  generation compares. No icount workload makes a no-op `set`, so its
  cost (the same claim, compare and release, no write and no walk) is not
  measured here.
- **slow**, per top-level non-Clean `get`: +40 / +44: `get_slow`'s frame,
  its generation compares and the `Gone` test after the walk, net of
  T0's out-of-line `ensure_clean` entry. Under
  Shared `try_read` costs 8 more than T0's `read`.
- **step**, per walk-driven recompute that reads one Clean dep: +86 / +84
  on the chains (walk growth excluded), 83 to 91 on the diamond and the
  collection pipelines. Isolated pieces at T1, per compute: the
  `Deferred` bag 13 (construction 4, the emptiness test at the drain 5,
  the drain call 4), the `g0` and post-claim generation compares 4, the
  memo-bit test at frame pop 3, the poisoned-landing branch 3, `wrote`
  and its argument 1, the first-visit counter and exact-mode test 7. The rest is the walk's restructure (`clean_walk` with
  `compute_one` inlined, against T0's `ensure_clean`), the compute
  wrapper's `VALUE_CODE` wrap and `Compared` return (+19 / +23 across
  the wrapper and the arena write) and the nested Clean `get` with its dep record (+17 /
  +20).
- **get**, per Clean `get`: -20 / -7. Its named pieces cost 19: the
  `VALUE_CODE` check 5, the wrap around `clone` 5, the pre-check and
  post-read generation compares 7, one segment null test 2. T0 called
  `ensure_clean` out of line on every `get` (22) and indexed the node
  three times; T1 tests Clean inline and indexes once.
- **first**, per query created and first computed through a non-Clean
  `get`: 96 to 98 Local, 87 to 91 Shared.
- **delete**, per node removed through `delete_erased`: 145 Local, 142 to
  144 Shared. At T1, per node: the set-stack check before the claim 36
  (`set_stack::holds` out of line 31, of which the thread-local scan is
  21, and its call site 5), `mark_deleted`'s `fetch_or` and branch 7 with
  the node index it needs 11, the `DeleteBag` construction 14 and its
  drain 10 outside the drops it runs, the drain's `catch_unwind`.
  `fd7eb95` tests the input's claim bit first, so the scan runs only
  while a set holds it: `delete_single` went to 1,335.00 Local and
  2,140.24 Shared, and `delete_cascade` to 3,477,099.84 Local (-16 per
  delete) and 4,640,478.62 Shared (-20), means of two runs.
  `delete_owned_tree` stays inside its seed spread.
- **strict**, per strict delete of a node that owns nothing: +158 / +166
  engine, against -140 / -147 alloc because T0's `claim_tree` allocated
  its walk stack on every call.
- **collection**: the multiset's one user `Hash` per element instead of
  two, net of its staging and `VALUE_CODE` wraps: -144 to -190 on the
  three logs of `*_incremental`, -96 to -117 on `simple_count`.

### Since the trimmed T1 engine

Counts at each engine commit after `08c0ea7`, the last engine before
the set claim moved, for every workload that moved by 1 or more. Each
commit ran once, which is enough because these workloads repeat to
within 1 instruction between runs.
`62f29d7` moves the claim to `verified_at`, `df5ee20` looks up the
arena before the claim, `289c1a5` refuses a delete under this thread's
set.

| workload | `08c0ea7` | `62f29d7` | `df5ee20` | `289c1a5` | T1 | T1 - `08c0ea7` |
|---|--:|--:|--:|--:|--:|--:|
| `chain_local/propagate/4` | 4,881.00 | 4,852.00 | 4,849.00 | 4,849.00 | 4,849.00 | -32.00 |
| `chain_local/propagate/10` | 10,539.00 | 10,522.00 | 10,519.00 | 10,519.00 | 10,519.00 | -20.00 |
| `chain_local/propagate/100` | 96,168.04 | 96,331.03 | 96,328.02 | 96,328.02 | 96,328.02 | +159.98 |
| `chain_shared/propagate/4` | 10,091.47 | 9,936.42 | 9,934.44 | 9,934.42 | 9,934.42 | -157.05 |
| `chain_shared/propagate/10` | 21,869.40 | 21,702.28 | 21,700.29 | 21,700.29 | 21,700.28 | -169.12 |
| `chain_shared/propagate/100` | 199,546.22 | 199,198.02 | 199,196.58 | 199,196.14 | 199,196.11 | -350.11 |
| `diamond_local` | 4,194.00 | 4,163.00 | 4,160.00 | 4,160.00 | 4,160.00 | -34.00 |
| `diamond_shared` | 8,500.14 | 8,348.10 | 8,346.14 | 8,346.11 | 8,346.12 | -154.02 |
| `early_cutoff_local` | 2,386.00 | 2,353.00 | 2,350.00 | 2,350.00 | 2,350.00 | -36.00 |
| `early_cutoff_shared` | 3,898.01 | 3,745.99 | 3,743.99 | 3,743.99 | 3,743.99 | -154.02 |
| `collection_insert_then_read/local_incremental/1000` | 4,679.87 | 4,648.58 | 4,645.66 | 4,645.72 | 4,645.65 | -34.22 |
| `collection_insert_then_read/shared_incremental/1000` | 7,663.31 | 7,510.25 | 7,508.59 | 7,508.45 | 7,507.92 | -155.39 |
| `collection_insert_then_read/local_incremental/10000` | 4,673.40 | 4,642.89 | 4,639.30 | 4,638.56 | 4,639.21 | -34.19 |
| `collection_insert_then_read/shared_incremental/10000` | 7,640.05 | 7,485.98 | 7,484.34 | 7,483.78 | 7,484.15 | -155.90 |
| `collection_insert_then_read/local_incremental/100000` | 4,680.06 | 4,648.29 | 4,646.97 | 4,646.89 | 4,647.55 | -32.51 |
| `collection_insert_then_read/shared_incremental/100000` | 7,644.93 | 7,491.32 | 7,488.88 | 7,488.50 | 7,489.05 | -155.89 |
| `simple_count/local/1000` | 2,228.65 | 2,193.76 | 2,190.48 | 2,190.65 | 2,191.00 | -37.65 |
| `simple_count/shared/1000` | 4,434.50 | 4,284.33 | 4,283.66 | 4,282.97 | 4,282.95 | -151.55 |
| `simple_count/local/10000` | 2,225.40 | 2,190.82 | 2,187.14 | 2,187.26 | 2,187.23 | -38.17 |
| `simple_count/shared/10000` | 4,415.57 | 4,266.53 | 4,263.48 | 4,264.64 | 4,264.98 | -150.59 |
| `fanout_local_4` | 19,210.01 | 19,301.01 | 19,301.00 | 19,522.01 | 19,522.00 | +311.99 |
| `fanout_local_5` | 23,823.59 | 23,929.59 | 23,929.57 | 24,186.50 | 24,186.50 | +362.91 |
| `fanout_local_1000` | 7,539,003.62 | 7,553,998.92 | 7,554,231.99 | 7,592,205.25 | 7,590,807.71 | +51,804.09 |
| `fanout_shared_4` | 26,174.48 | 26,290.36 | 26,290.37 | 26,509.07 | 26,509.11 | +334.63 |
| `fanout_shared_5` | 31,619.15 | 31,753.05 | 31,753.03 | 32,005.06 | 32,005.07 | +385.92 |
| `fanout_shared_1000` | 9,048,097.19 | 9,066,123.46 | 9,066,127.04 | 9,101,544.05 | 9,101,544.88 | +53,447.69 |
| `delete_single_local` | 1,253.00 | 1,316.00 | 1,316.00 | 1,349.00 | 1,349.00 | +96.00 |
| `delete_single_shared` | 2,045.30 | 2,124.24 | 2,124.24 | 2,158.24 | 2,158.24 | +112.94 |
| `delete_cascade_local` | 3,441,063.28 | 3,456,078.96 | 3,456,078.74 | 3,493,116.03 | 3,493,116.82 | +52,053.54 |
| `delete_cascade_shared` | 4,606,441.83 | 4,624,451.11 | 4,624,455.94 | 4,660,489.54 | 4,660,485.05 | +54,043.22 |

Where `set` and delete moved, from Intel PT per symbol and per source
line, `08c0ea7` against `289c1a5`:

- **set**, per value-changing `set`, `62f29d7`: -37 Local, -141 Shared.
  The precheck before the claim is gone: under Local its `eq_current`,
  slot compare, `VALUE_CODE` wrap, arena borrow, node index and
  generation compare, about 39; under Shared the out-of-line
  `eq_current` symbols with their hazard-pointer read, 116, and the
  arena read lock. The claim's take and release on `verified_at` cost 13
  under both strategies, in place of the state CAS and the landing's
  CAS, and `claim_input` returning the claim, arena and slot costs 11 /
  9. `df5ee20` takes 3 / 2 more off `set`.
- **per compute**, `62f29d7`: masking the claim and delete bits off each
  `verified_at` load (`& REVISION_BITS`) adds 2 per verified compute
  under Local, so across the three commits `chain_local/propagate/100`
  rose by 160 while its `set` fell by 40. Under Shared the walk came out 4 lower per compute and
  haphazard's `push_list` 2 higher.
- **delete**, per node through `delete_erased`: +15 Local, +18 Shared at
  `62f29d7` (`mark_deleted`'s `fetch_or` and branch 7, and the node
  index it reads 11), and +37 / +36 at `289c1a5`, the set-stack check
  before the claim (`set_stack::holds` 31 out of line, its call site 5).
- **strict delete**: +63 Local, +79 Shared at `62f29d7` (the mark loop
  over the root and its owned members, `once(&root).chain(&owned)`, 30,
  plus the mark, a node index and, under Shared, the `wait_out_set`
  test), and +33 / +34 at `289c1a5` (the same set-stack check).
- The set-stack check runs on every delete, with no set in progress.
  This thread's set always holds `SET_CLAIM` while its stack entry
  exists, so testing that bit first would confine the scan to inputs some
  set holds.

Gates against T0 (T0 **Gate method** and T1 **Gate**):

- Clean `get`: 178 / 302 against 198 / 309, below T0. Holds.
- The workloads that run no T1 source, `collection_insert_then_read/batch/*`,
  moved by -0.47, +0.23 and -0.90, within T0's spreads (0.18, 0.78 and
  2.27). Holds.
- Every other rise falls in the units above, each of them a source named
  in T1's section of the plan, which now carries these measured counts in
  place of its estimates. `get_contended_shared` is within the spread of
  the two T1 runs; readers no longer wait on a set's claim.

## T2

- Engine: `abb7540` (branch `spreadsheet-t2`); later T2 commits change
  only tests. Machine and toolchain as for T0.
- T1 is `main` at `7d8faef` run again in the same session. Both columns
  are the mean of two runs of `icount.sh` at the default N; the two runs
  of each differ by at most 0.02%, except `delete_owned_tree_*` (0.28%,
  seeds) and the T1 `get_contended_shared` (4.3%, scheduling; T2's two
  runs agree to 0.001%).
- Attribution: Intel PT on line-table builds of both drivers, as for T1,
  on `get_clean_*`, `chain_*/propagate/10`, `delete_single_*` and
  `fanout_shared_4`; the other rows are sums of the same units.

| workload | N | T1 | T2 | delta | delta % | sources |
|---|--:|--:|--:|--:|--:|---|
| `chain_local/propagate/4` | 1,000,000 | 4,849.00 | 4,577.00 | -272.00 | -5.61 | read and step units, set |
| `chain_local/propagate/10` | 1,000,000 | 10,519.00 | 9,977.00 | -542.00 | -5.15 | read and step units, set |
| `chain_local/propagate/100` | 100,000 | 96,328.02 | 91,736.01 | -4,592.01 | -4.77 | read and step units, set |
| `chain_shared/propagate/4` | 1,000,000 | 9,934.44 | 9,668.43 | -266.01 | -2.68 | read and step units, set |
| `chain_shared/propagate/10` | 1,000,000 | 21,700.35 | 21,182.30 | -518.05 | -2.39 | read and step units, set |
| `chain_shared/propagate/100` | 100,000 | 199,196.71 | 194,898.41 | -4,298.30 | -2.16 | read and step units, set |
| `diamond_local` | 1,000,000 | 4,160.00 | 3,881.00 | -279.00 | -6.71 | read and step units, set |
| `diamond_shared` | 1,000,000 | 8,346.12 | 8,061.10 | -285.02 | -3.41 | read and step units, set |
| `early_cutoff_local` | 1,000,000 | 2,350.00 | 2,227.00 | -123.00 | -5.23 | read and step units, set |
| `early_cutoff_shared` | 1,000,000 | 3,743.98 | 3,607.99 | -135.99 | -3.63 | read and step units, set |
| `collection_insert_then_read/local_incremental/1000` | 100,000 | 4,645.88 | 4,474.50 | -171.38 | -3.69 | read and step units, set |
| `collection_insert_then_read/shared_incremental/1000` | 100,000 | 7,508.07 | 7,319.07 | -189.00 | -2.52 | read and step units, set |
| `collection_insert_then_read/batch/1000` | 10,000 | 92,283.31 | 92,283.06 | -0.25 | -0.00 | none |
| `collection_insert_then_read/local_incremental/10000` | 100,000 | 4,638.95 | 4,468.35 | -170.61 | -3.68 | read and step units, set |
| `collection_insert_then_read/shared_incremental/10000` | 100,000 | 7,484.38 | 7,296.00 | -188.37 | -2.52 | read and step units, set |
| `collection_insert_then_read/batch/10000` | 1,000 | 158,885.93 | 158,883.05 | -2.89 | -0.00 | none |
| `collection_insert_then_read/local_incremental/100000` | 100,000 | 4,646.66 | 4,475.51 | -171.15 | -3.68 | read and step units, set |
| `collection_insert_then_read/shared_incremental/100000` | 100,000 | 7,488.56 | 7,299.48 | -189.08 | -2.52 | read and step units, set |
| `collection_insert_then_read/batch/100000` | 1,000 | 1,490,573.37 | 1,490,572.54 | -0.83 | -0.00 | none |
| `simple_count/local/1000` | 100,000 | 2,190.80 | 2,055.54 | -135.26 | -6.17 | read and step units, set |
| `simple_count/shared/1000` | 100,000 | 4,282.83 | 4,138.08 | -144.76 | -3.38 | read and step units, set |
| `simple_count/local/10000` | 100,000 | 2,187.18 | 2,052.27 | -134.91 | -6.17 | read and step units, set |
| `simple_count/shared/10000` | 100,000 | 4,264.53 | 4,119.24 | -145.29 | -3.41 | read and step units, set |
| `get_clean_local` | 10,000,000 | 178.00 | 125.00 | -53.00 | -29.78 | read unit -53 |
| `get_clean_shared` | 10,000,000 | 302.00 | 245.00 | -57.00 | -18.87 | read unit -57 |
| `get_contended_shared` | 2,000 | 13,172,516.03 | 9,753,051.50 | -3,419,464.53 | -25.96 | read unit; T1 runs spread 4.3% (scheduling) |
| `fanout_local_4` | 100,000 | 19,424.00 | 17,964.00 | -1,460.00 | -7.52 | create, read and step units, delete |
| `fanout_local_5` | 100,000 | 24,072.58 | 22,187.44 | -1,885.14 | -7.83 | create, read and step units, delete |
| `fanout_local_1000` | 1,000 | 7,574,040.92 | 7,305,714.19 | -268,326.73 | -3.54 | create, read and step units, delete |
| `fanout_shared_4` | 100,000 | 26,386.38 | 24,636.07 | -1,750.31 | -6.63 | create, read and step units, delete |
| `fanout_shared_5` | 100,000 | 31,865.12 | 29,790.06 | -2,075.07 | -6.51 | create, read and step units, delete |
| `fanout_shared_1000` | 1,000 | 9,082,157.38 | 8,759,125.55 | -323,031.83 | -3.56 | create, read and step units, delete |
| `delete_single_local` | 200,000 | 1,335.00 | 1,160.00 | -175.00 | -13.11 | create, delete |
| `delete_single_shared` | 200,000 | 2,140.24 | 1,954.24 | -186.00 | -8.69 | create, delete |
| `delete_cascade_local` | 1,000 | 3,477,099.22 | 3,206,925.54 | -270,173.68 | -7.77 | create, read and step units, delete |
| `delete_cascade_shared` | 1,000 | 4,640,476.78 | 4,315,275.24 | -325,201.53 | -7.01 | create, read and step units, delete |
| `delete_owned_tree_local` | 10,000 | 641,506.11 | 618,633.56 | -22,872.55 | -3.57 | create, set, delete; seeds |
| `delete_owned_tree_shared` | 10,000 | 1,028,126.38 | 1,001,632.51 | -26,493.86 | -2.58 | create, set, delete; seeds |

Measured units, Local / Shared, against T1:

- **read**, per Clean `get` (a top-level read, a dep read inside a
  compute, or `set`'s arena lookup): the registry lookup, out of line in
  T1 with an `inner` borrow or read lock and an `Arc` clone and drop, -35
  / -35; the vtable downcast, now a `TypeId` compare, -7 / -7; `get`
  itself -10 / -6. Under Shared the arena read lock goes too (-29), and
  the slot read's hazard-pool access stopped inlining into it and runs in
  `LocalKey::with` (+20 net). `get_clean_*`: -53 / -57.
- **step**, per compute: Local, the write without the arena borrow -4,
  and `clean_walk` +11, one instruction on each of about ten lines of the
  walk (register allocation; no T2 code runs there). Shared, the read and
  write sessions with no arena lock -7 net of the hazard pool moving out
  of line, the compute closure +5, `clean_walk` +6.
- **set**: its arena lookup is now inline, +10 / +18 in `set`, against
  the read unit's -42 it no longer pays out of line.
- **create**, per node: the `TypeId`-to-tag map hashed with SipHash in
  T1 (`hash_one` and `DefaultHasher::write`, 142 per create); it now uses
  the `TypeId`'s own bits. With the registry's `Arc` clone and downcast
  gone and the lookup inlined into `create_input` (+42 / +45) and
  `reserve_with` (+12 / +4), a `create_input` is -172 / -177. A
  `create_query` takes the arena's `Arc` for its closure
  (`ensure_tagged_shared`, 71 per query), so it saves less.
- **delete**, per node: the bag borrows the arena instead of cloning its
  `Arc` (-8 / -11 in `delete_strict`, -16 per node in `delete_erased`
  under Shared, -5 / -5 in the drain), and the release looks the slot up
  before it takes the arena lock (+10 / +7).
- **value-code arm**: `get_slow`'s value-code test replaces the
  `checked_word` test it had, and the arm is out of line; no workload
  takes it.

Gates against T1 and T0 (plan T2 **Gate**):

- Clean `get`: 125 / 245 against T0's 198 / 309. Holds.
- Every other workload is below T1; the `delete_*` benches needed none of
  the allowance for the Local `reserve` and `clear` checks, which the
  create unit absorbs. `get_contended_shared` is 26% below T1. Holds.

## T3

- Engine: `561774b` (branch `spreadsheet-t3`). Machine and toolchain as
  for T0.
- T2 is `main` at `52fc36a` run again in the same session. Both columns
  are the mean of two runs of `icount.sh` at the default N; the two runs
  of each agree to 0.001% except the `batch` rows (up to 8 instructions
  in 1.49M) and `delete_owned_tree_*` (seeds).
- Attribution: Intel PT on `get_clean_shared` in both drivers; every
  other row falls, and each node access in it goes through the same
  indexing.

| workload | N | T2 | T3 | delta | delta % | sources |
|---|--:|--:|--:|--:|--:|---|
| `chain_local/propagate/4` | 1,000,000 | 4,577.00 | 4,380.00 | -197.00 | -4.30 | node reads |
| `chain_local/propagate/10` | 1,000,000 | 9,977.00 | 9,504.00 | -473.00 | -4.74 | node reads |
| `chain_local/propagate/100` | 100,000 | 91,736.01 | 87,123.01 | -4,613.00 | -5.03 | node reads |
| `chain_shared/propagate/4` | 1,000,000 | 9,668.42 | 9,554.42 | -114.00 | -1.18 | node reads |
| `chain_shared/propagate/10` | 1,000,000 | 21,182.28 | 20,906.29 | -275.99 | -1.30 | node reads |
| `chain_shared/propagate/100` | 100,000 | 194,898.10 | 192,192.35 | -2,705.74 | -1.39 | node reads |
| `diamond_local` | 1,000,000 | 3,881.00 | 3,714.00 | -167.00 | -4.30 | node reads |
| `diamond_shared` | 1,000,000 | 8,061.09 | 7,965.10 | -95.99 | -1.19 | node reads |
| `early_cutoff_local` | 1,000,000 | 2,227.00 | 2,139.00 | -88.00 | -3.95 | node reads |
| `early_cutoff_shared` | 1,000,000 | 3,607.98 | 3,549.98 | -58.00 | -1.61 | node reads |
| `collection_insert_then_read/local_incremental/1000` | 100,000 | 4,475.10 | 4,333.45 | -141.65 | -3.17 | node reads |
| `collection_insert_then_read/shared_incremental/1000` | 100,000 | 7,319.41 | 7,234.35 | -85.06 | -1.16 | node reads |
| `collection_insert_then_read/batch/1000` | 10,000 | 92,283.62 | 92,283.09 | -0.54 | -0.00 | none |
| `collection_insert_then_read/local_incremental/10000` | 100,000 | 4,468.09 | 4,326.25 | -141.84 | -3.17 | node reads |
| `collection_insert_then_read/shared_incremental/10000` | 100,000 | 7,295.22 | 7,210.48 | -84.74 | -1.16 | node reads |
| `collection_insert_then_read/batch/10000` | 1,000 | 158,885.96 | 158,886.15 | +0.18 | +0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/100000` | 100,000 | 4,475.06 | 4,333.56 | -141.50 | -3.16 | node reads |
| `collection_insert_then_read/shared_incremental/100000` | 100,000 | 7,300.51 | 7,214.84 | -85.67 | -1.17 | node reads |
| `collection_insert_then_read/batch/100000` | 1,000 | 1,490,568.62 | 1,490,573.05 | +4.42 | +0.00 | none (run spread) |
| `simple_count/local/1000` | 100,000 | 2,055.41 | 1,995.68 | -59.73 | -2.91 | node reads |
| `simple_count/shared/1000` | 100,000 | 4,137.93 | 4,104.60 | -33.32 | -0.81 | node reads |
| `simple_count/local/10000` | 100,000 | 2,052.19 | 1,992.28 | -59.91 | -2.92 | node reads |
| `simple_count/shared/10000` | 100,000 | 4,119.42 | 4,086.39 | -33.03 | -0.80 | node reads |
| `get_clean_local` | 10,000,000 | 125.00 | 117.00 | -8.00 | -6.40 | node read -8 |
| `get_clean_shared` | 10,000,000 | 245.00 | 246.00 | +1.00 | +0.41 | node read -5; `record_dep` +6 (codegen units, below) |
| `get_contended_shared` | 2,000 | 9,753,041.40 | 9,793,050.97 | +40,009.57 | +0.41 | as `get_clean_shared` |
| `fanout_local_4` | 100,000 | 17,964.00 | 17,367.00 | -597.00 | -3.32 | node reads |
| `fanout_local_5` | 100,000 | 22,187.44 | 21,490.72 | -696.72 | -3.14 | node reads |
| `fanout_local_1000` | 1,000 | 7,305,725.69 | 7,189,549.70 | -116,175.99 | -1.59 | node reads |
| `fanout_shared_4` | 100,000 | 24,635.29 | 24,150.04 | -485.25 | -1.97 | node reads |
| `fanout_shared_5` | 100,000 | 29,790.06 | 29,210.02 | -580.04 | -1.95 | node reads |
| `fanout_shared_1000` | 1,000 | 8,758,749.94 | 8,666,647.73 | -92,102.20 | -1.05 | node reads |
| `delete_single_local` | 200,000 | 1,160.00 | 1,103.00 | -57.00 | -4.91 | node reads |
| `delete_single_shared` | 200,000 | 1,954.24 | 1,909.24 | -45.00 | -2.30 | node reads |
| `delete_cascade_local` | 1,000 | 3,206,925.76 | 3,100,879.98 | -106,045.79 | -3.31 | node reads |
| `delete_cascade_shared` | 1,000 | 4,315,279.22 | 4,232,556.88 | -82,722.35 | -1.92 | node reads |
| `delete_owned_tree_local` | 10,000 | 620,502.71 | 613,307.09 | -7,195.62 | -1.16 | node reads |
| `delete_owned_tree_shared` | 10,000 | 1,002,149.55 | 999,631.55 | -2,518.00 | -0.25 | node reads |

- **node read**, per `SegmentedNodes::get`: the directory's length now
  sits in its header, next to the entries, and a segment is a fixed-size
  array, so the segment's own slice bounds check (a length load, a
  compare and a branch) and the load of its slice pointer are gone.
  Shared `get` is -5 per Clean read; Local `get_clean_local` is -8.
- **`record_dep` (Shared, +6 per call):** the thread-local
  closure behind `SharedDepStack::record_dep`, which T3 does not touch,
  now spills its arguments to the stack on every call. Its source is
  unchanged; the build's codegen-unit partitioning moved it. With
  `CARGO_PROFILE_BENCH_CODEGEN_UNITS=1` both drivers compile it the same
  way, and T3 is below T2: `get_clean_shared` 212 against 215,
  `get_clean_local` 118 against 125, `get_contended_shared`
  8,432,857 against 8,552,949 (-1.4%), `delete_single_shared` 1,893
  against 1,953, `chain_shared/propagate/10` 19,820 against 20,437.

Gate against T2 (plan T3 **Gate**): every workload is at or below T2
except `get_clean_shared` (+1) and `get_contended_shared` (+0.41%), both
the `record_dep` codegen-unit shift named above, with T3 below T2 on
both in a single-unit build. The drops come from the directory's
indexing.

## T4

- Engine: `ca0d327` (branch `spreadsheet-t4`). Machine and toolchain as
  for T0.
- T3 is `main` at `e972737` run again in the same session. Both columns
  are the mean of two runs of `icount.sh` at the default N.
- Attribution: Intel PT on both drivers, counted per symbol at steady
  state: each workload runs a warm-up of 10,000 or more iterations, and
  the per-iteration count is the difference between two traced N. At
  small N without a warm-up the per-iteration counts differ from the
  default N, so only steady-state traces are used.

| workload | N | T3 | T4 | delta | delta % | sources |
|---|--:|--:|--:|--:|--:|---|
| `chain_local/propagate/4` | 1,000,000 | 4,380.00 | 4,326.00 | -54.00 | -1.23 | compute: no closure `Arc` clone, dependent-list walk |
| `chain_local/propagate/10` | 1,000,000 | 9,504.00 | 9,360.00 | -144.00 | -1.52 | compute: no closure `Arc` clone, dependent-list walk |
| `chain_local/propagate/100` | 100,000 | 87,123.01 | 85,629.01 | -1,494.01 | -1.71 | compute: no closure `Arc` clone, dependent-list walk |
| `chain_shared/propagate/4` | 1,000,000 | 9,554.42 | 9,227.19 | -327.23 | -3.42 | compute: no closure `Arc` clone, dependent-list walk |
| `chain_shared/propagate/10` | 1,000,000 | 20,906.32 | 20,146.80 | -759.52 | -3.63 | compute: no closure `Arc` clone, dependent-list walk |
| `chain_shared/propagate/100` | 100,000 | 192,192.39 | 184,949.85 | -7,242.54 | -3.77 | compute: no closure `Arc` clone, dependent-list walk |
| `diamond_local` | 1,000,000 | 3,714.00 | 3,682.00 | -32.00 | -0.86 | compute: no closure `Arc` clone, dependent-list walk |
| `diamond_shared` | 1,000,000 | 7,965.10 | 7,693.93 | -271.18 | -3.40 | compute: no closure `Arc` clone, dependent-list walk |
| `early_cutoff_local` | 1,000,000 | 2,139.00 | 2,122.00 | -17.00 | -0.79 | compute: no closure `Arc` clone, dependent-list walk |
| `early_cutoff_shared` | 1,000,000 | 3,549.99 | 3,426.84 | -123.15 | -3.47 | compute: no closure `Arc` clone, dependent-list walk |
| `collection_insert_then_read/local_incremental/1000` | 100,000 | 4,333.70 | 4,301.85 | -31.85 | -0.74 | compute: no closure `Arc` clone, dependent-list walk |
| `collection_insert_then_read/shared_incremental/1000` | 100,000 | 7,234.60 | 7,038.18 | -196.42 | -2.71 | compute: no closure `Arc` clone, dependent-list walk |
| `collection_insert_then_read/batch/1000` | 10,000 | 92,283.36 | 92,283.50 | +0.14 | +0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/10000` | 100,000 | 4,326.05 | 4,294.44 | -31.61 | -0.73 | compute: no closure `Arc` clone, dependent-list walk |
| `collection_insert_then_read/shared_incremental/10000` | 100,000 | 7,210.60 | 7,011.88 | -198.72 | -2.76 | compute: no closure `Arc` clone, dependent-list walk |
| `collection_insert_then_read/batch/10000` | 1,000 | 158,886.68 | 158,886.48 | -0.20 | -0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/100000` | 100,000 | 4,334.05 | 4,302.80 | -31.25 | -0.72 | compute: no closure `Arc` clone, dependent-list walk |
| `collection_insert_then_read/shared_incremental/100000` | 100,000 | 7,216.27 | 7,016.37 | -199.90 | -2.77 | compute: no closure `Arc` clone, dependent-list walk |
| `collection_insert_then_read/batch/100000` | 1,000 | 1,490,570.40 | 1,490,570.29 | -0.11 | -0.00 | none (run spread) |
| `simple_count/local/1000` | 100,000 | 1,995.85 | 1,986.57 | -9.28 | -0.46 | compute: no closure `Arc` clone, dependent-list walk |
| `simple_count/shared/1000` | 100,000 | 4,105.03 | 3,981.82 | -123.21 | -3.00 | compute: no closure `Arc` clone, dependent-list walk |
| `simple_count/local/10000` | 100,000 | 1,992.53 | 1,983.59 | -8.94 | -0.45 | compute: no closure `Arc` clone, dependent-list walk |
| `simple_count/shared/10000` | 100,000 | 4,086.91 | 3,961.43 | -125.48 | -3.07 | compute: no closure `Arc` clone, dependent-list walk |
| `get_clean_local` | 10,000,000 | 117.00 | 118.00 | +1.00 | +0.85 | register allocation in `get` (+1) |
| `get_clean_shared` | 10,000,000 | 246.00 | 220.00 | -26.00 | -10.57 | closure `Arc` gone from the read path; `dep_stack` at offset 0 |
| `get_contended_shared` | 2,000 | 9,793,057.19 | 8,752,974.64 | -1,040,082.54 | -10.62 | closure `Arc` gone from the read path; `dep_stack` at offset 0 |
| `fanout_local_4` | 100,000 | 17,367.00 | 17,786.00 | +419.00 | +2.41 | create and delete tables (below) |
| `fanout_local_5` | 100,000 | 21,490.72 | 22,432.15 | +941.43 | +4.38 | create and delete tables (below) |
| `fanout_local_1000` | 1,000 | 7,189,574.39 | 3,520,628.27 | -3,668,946.12 | -51.03 | linear fan-in teardown |
| `fanout_shared_4` | 100,000 | 24,150.10 | 24,377.89 | +227.79 | +0.94 | create and delete tables (below) |
| `fanout_shared_5` | 100,000 | 29,210.03 | 29,802.78 | +592.75 | +2.03 | create and delete tables (below) |
| `fanout_shared_1000` | 1,000 | 8,665,752.26 | 4,865,016.29 | -3,800,735.97 | -43.86 | linear fan-in teardown |
| `delete_single_local` | 200,000 | 1,103.00 | 1,106.00 | +3.00 | +0.27 | create and delete tables (below) |
| `delete_single_shared` | 200,000 | 1,909.26 | 1,933.24 | +23.98 | +1.26 | create and delete tables (below) |
| `delete_cascade_local` | 1,000 | 3,100,880.06 | 3,329,606.48 | +228,726.42 | +7.38 | allocator consolidation (below), create and delete tables |
| `delete_cascade_shared` | 1,000 | 4,237,599.57 | 4,313,815.06 | +76,215.49 | +1.80 | allocator consolidation (below), create and delete tables |
| `delete_owned_tree_local` | 10,000 | 614,953.89 | 602,068.39 | -12,885.50 | -2.10 | owned-map flag; seeds |
| `delete_owned_tree_shared` | 10,000 | 1,000,073.06 | 984,464.28 | -15,608.78 | -1.56 | owned-map flag; seeds |

- **compute**, per compute: the closure is borrowed under the claim, so
  the `Arc` clone and drop go; `Runtime` is `repr(C)` with `dep_stack`
  first, so `record_dep` takes `self` as its pointer, as it happened to
  in T3. A dirty walk reads each dependents list inline (a mode test and
  pushes up to the first `NONE`), and its queue is allocated at 8
  entries only when the first list is non-empty. Local compute workloads
  fall 0.6-1.7%, Shared 1.3-3.7%.
- **`get_clean_local` (+1):** one `mov` more in `get`, a register the
  handle pointer is kept in; no T4 code runs on that path.
- **create and delete tables (`fanout_*_4`, `fanout_*_5`,
  `delete_single_*`):** at steady state on `fanout_local_4` (+427):
  `delete_erased` +199, which now drops the boxed closure itself
  (`Arc::drop_slow` -165 is gone); `delete_with_dependents` +157, a
  `SlotVec` index and a dependent-list `last()` per cascade step;
  `publish_deps` +121, a `SlotVec` index and an inline push per new
  edge; allocator +79. A fan-in of five spills its list to a heap block,
  which is freed when the list empties, where T3 kept every recycled
  slot's `Vec` capacity; that is `fanout_*_5`'s larger rise.
- **allocator consolidation (`delete_cascade_*`):** each cascade grows
  its stack to about 2 KiB, a large-bin request in both builds. With
  closures in 24-byte boxes instead of 40-byte `Arc`s, glibc finds
  fastbin chunks to consolidate at that request: of
  `delete_cascade_local`'s +228,727, `_int_malloc` +92,013,
  `malloc_consolidate` +63,137 and `unlink_chunk` +23,226 are the
  allocator, and the rest is the table costs above. Taking the cascade's
  first dependent instead of its last leaves the count unchanged
  (3,314,959), so the teardown order is not the cause.
- **linear fan-in teardown:** each dependent is in a list once, a
  delete removes it scanning from the back, and the cascade takes the
  last dependent, so `fanout_*_1000` halves.

Gate against T3 (plan T4 **Gate**): every compute workload is at or
below T3 except `get_clean_local` (+1, register allocation). The
`compute_one` flags-byte load the plan allows is not taken: no flag
`compute_one` decides on exists before T5. The rises on `fanout_*_4`,
`fanout_*_5`, `delete_single_*` and `delete_cascade_*` are the create
and delete table costs and the allocator effect named above.

## T5

- Engine: `82f19f6` (branch `spreadsheet-t5`). Machine and toolchain as
  for T0.
- T4 is `main` at `26e14c0`, run twice in the same session as T5's two
  runs; both columns are the mean of two runs of `icount.sh` at the
  default N.
- Attribution: Intel PT on both drivers at steady state (a warm-up of
  10,000 iterations, or 2,000 and 10 on the fan-out and cascade
  workloads), per symbol, the per-iteration count being the difference
  between two traced N. Each traced total matches its `icount.sh` row,
  except the collection workloads', whose count depends on N.

| workload | N | T4 | T5 | delta | delta % | sources |
|---|--:|--:|--:|--:|--:|---|
| `chain_local/propagate/4` | 1,000,000 | 4,326.00 | 4,443.00 | +117.00 | +2.70 | compute and entry (below) |
| `chain_local/propagate/10` | 1,000,000 | 9,360.00 | 9,591.00 | +231.00 | +2.47 | compute and entry (below) |
| `chain_local/propagate/100` | 100,000 | 85,629.01 | 87,570.01 | +1,941.00 | +2.27 | compute and entry (below) |
| `chain_shared/propagate/4` | 1,000,000 | 9,227.22 | 9,268.41 | +41.19 | +0.45 | compute and entry (below) |
| `chain_shared/propagate/10` | 1,000,000 | 20,146.81 | 20,176.27 | +29.46 | +0.15 | compute and entry (below) |
| `chain_shared/propagate/100` | 100,000 | 184,950.35 | 184,811.96 | -138.39 | -0.07 | compute and entry (below) |
| `diamond_local` | 1,000,000 | 3,682.00 | 3,773.00 | +91.00 | +2.47 | compute and entry (below) |
| `diamond_shared` | 1,000,000 | 7,693.95 | 7,737.09 | +43.14 | +0.56 | compute and entry (below) |
| `early_cutoff_local` | 1,000,000 | 2,122.00 | 2,209.00 | +87.00 | +4.10 | compute and entry (below) |
| `early_cutoff_shared` | 1,000,000 | 3,426.84 | 3,502.89 | +76.05 | +2.22 | compute and entry (below) |
| `collection_insert_then_read/local_incremental/1000` | 100,000 | 4,301.35 | 4,489.77 | +188.43 | +4.38 | compute and entry (below) |
| `collection_insert_then_read/shared_incremental/1000` | 100,000 | 7,037.67 | 7,193.55 | +155.88 | +2.21 | compute and entry (below) |
| `collection_insert_then_read/batch/1000` | 10,000 | 92,283.65 | 92,272.86 | -10.79 | -0.01 | none (run spread) |
| `collection_insert_then_read/local_incremental/10000` | 100,000 | 4,293.49 | 4,483.61 | +190.12 | +4.43 | compute and entry (below) |
| `collection_insert_then_read/shared_incremental/10000` | 100,000 | 7,011.93 | 7,167.06 | +155.13 | +2.21 | compute and entry (below) |
| `collection_insert_then_read/batch/10000` | 1,000 | 158,886.35 | 158,876.07 | -10.29 | -0.01 | none (run spread) |
| `collection_insert_then_read/local_incremental/100000` | 100,000 | 4,301.99 | 4,491.07 | +189.08 | +4.40 | compute and entry (below) |
| `collection_insert_then_read/shared_incremental/100000` | 100,000 | 7,016.07 | 7,173.01 | +156.93 | +2.24 | compute and entry (below) |
| `collection_insert_then_read/batch/100000` | 1,000 | 1,490,574.65 | 1,490,560.76 | -13.89 | -0.00 | none (run spread) |
| `simple_count/local/1000` | 100,000 | 1,986.97 | 2,049.49 | +62.52 | +3.15 | compute and entry (below) |
| `simple_count/shared/1000` | 100,000 | 3,981.57 | 4,030.98 | +49.42 | +1.24 | compute and entry (below) |
| `simple_count/local/10000` | 100,000 | 1,983.26 | 2,045.96 | +62.70 | +3.16 | compute and entry (below) |
| `simple_count/shared/10000` | 100,000 | 3,961.20 | 4,010.60 | +49.39 | +1.25 | compute and entry (below) |
| `get_clean_local` | 10,000,000 | 118.00 | 98.00 | -20.00 | -16.95 | thread-local address outside `with`; `record_dep` inline |
| `get_clean_shared` | 10,000,000 | 220.00 | 200.00 | -20.00 | -9.09 | thread-local address outside `with`; `record_dep` inline |
| `get_contended_shared` | 2,000 | 8,753,056.38 | 7,953,180.26 | -799,876.12 | -9.14 | thread-local address outside `with`; `record_dep` inline |
| `fanout_local_4` | 100,000 | 17,786.00 | 18,173.00 | +387.00 | +2.18 | compute and entry (below); delete entry, strategy and own-held checks |
| `fanout_local_5` | 100,000 | 22,432.14 | 22,889.93 | +457.79 | +2.04 | compute and entry (below); delete entry, strategy and own-held checks |
| `fanout_local_1000` | 1,000 | 3,527,052.35 | 3,587,732.03 | +60,679.68 | +1.72 | compute and entry (below); delete entry, strategy and own-held checks |
| `fanout_shared_4` | 100,000 | 24,377.32 | 24,957.61 | +580.29 | +2.38 | compute and entry (below); delete entry, strategy and own-held checks |
| `fanout_shared_5` | 100,000 | 29,802.73 | 30,476.01 | +673.28 | +2.26 | compute and entry (below); delete entry, strategy and own-held checks |
| `fanout_shared_1000` | 1,000 | 4,864,336.72 | 4,961,266.86 | +96,930.14 | +1.99 | compute and entry (below); delete entry, strategy and own-held checks |
| `delete_single_local` | 200,000 | 1,106.00 | 1,156.00 | +50.00 | +4.52 | compute and entry (below); delete entry, strategy and own-held checks |
| `delete_single_shared` | 200,000 | 1,933.25 | 2,013.24 | +79.99 | +4.14 | compute and entry (below); delete entry, strategy and own-held checks |
| `delete_cascade_local` | 1,000 | 3,328,988.05 | 3,400,636.00 | +71,647.95 | +2.15 | delete entry, strategy and own-held checks; compute and entry |
| `delete_cascade_shared` | 1,000 | 4,666,877.80 | 4,766,672.11 | +99,794.31 | +2.14 | delete entry, strategy and own-held checks; compute and entry |
| `delete_owned_tree_local` | 10,000 | 602,328.28 | 605,446.12 | +3,117.84 | +0.52 | delete entry, strategy and own-held checks; compute and entry |
| `delete_owned_tree_shared` | 10,000 | 986,251.05 | 989,330.76 | +3,079.71 | +0.31 | delete entry, strategy and own-held checks; compute and entry |

- **compute**, per compute: the `ComputeCtl` stores, the frame's `rt`,
  `ctl` and `cidx`, and the wrapper's flag test; the flags reload after
  the claim and the `IN_GROUP` tests at verify and landing; the
  `verified_at == 0` compare; the walk's `IN_GROUP` test per Dirty visit
  and its Cyclic and New arms; the entry-depth counter around
  `compute_one`; and at the walk's drop, the one test of the box that
  holds its New and group sets. The dep-stack methods are now inlined,
  so T4's out-of-line `push_frame`, `pop_frame` and `record_dep` move
  into `clean_walk` and `get`, and `for_each_dep` moves out of
  `clean_walk` into its own symbol. On `chain_local/propagate/4`
  (+117): `clean_walk` with `for_each_dep` and the walk's drop +399
  against `push_frame` and `pop_frame` -240, `get` +103 against
  `record_dep` -180, `get_slow` +38. Net per compute on
  `chain_local/propagate/100` is +19.4; under Shared it is -1.4, because
  `record_dep`'s thread-local access got cheaper (below).
- **entry**, per outermost entry (a top-level non-Clean `get`, a `set`,
  a delete): the entry-depth increment and decrement, the test of the
  flag that says the outermost reset has anything to clear, and the
  `catch_unwind` call, whose landing pad is out of line. `get_slow` +38
  Local.
- **Clean `get`** (hard gate: equal to T4's): 98 Local and 200 Shared,
  against 118 and 220. Under Shared, `record_dep` and the hazard-pool
  pop and push each take only the thread-local's address inside
  `LocalKey::with` and work on it outside; with the work inside the
  closure, T5's larger callers left `LocalKey::with` and its liveness
  check out of line, which had put the count at 269. Under Local,
  `record_dep` is inlined into `get`. `get_contended_shared` falls by
  the same path.
- **delete entry, strategy and own-held checks** (`fanout_*`,
  `delete_*`): each delete is a public entry (depth counter and
  `catch_unwind` call, `delete_node` +36), `delete_with_dependents`
  enters once per node it deletes (+145 on `fanout_local_4`), and
  `delete_erased` and `delete_strict` test `HAS_STRATEGY` and check the
  wider own-held set, which compares `rt` per frame and tests a
  held-map flag before the map (`delete_erased` +82, `delete_strict`
  +23, Local). `create_query` passes no strategy, so it builds and
  erases no `CycleStrategy`; `create_query_with` erases the one it is
  given.
- **`delete_cascade_*`, `delete_owned_tree_*`, `fanout_*_1000`:** each
  iteration also builds and first-computes its subgraph, which pays the
  compute and entry sources per node. On `delete_cascade_local`
  (+71,648): the build's `create_query`, closure and `clean_walk`
  symbols account for most of it, and the teardown for
  `delete_with_dependents` +23,030 (an entry per cascaded node),
  `delete_erased` +14,012 and `cycle_state::holds` +10,010 (the
  own-held check, about 10 per node). The glibc symbols move by under
  0.4%. `delete_owned_tree_*` rises by about one T4 run spread.

Gate against T4 (plan T5 **Gate**): the Clean `get` is below T4's on
both strategies. Every other rise is in a symbol that holds one of the
named sources above, apart from `for_each_dep`, which is code that was
inlined into `clean_walk` in T4 and is a call now. The collection
`batch` rows are within run spread.

## T6

- Engine: `91e83c8` (branch `spreadsheet-t6`). Machine and toolchain as
  for T0.
- T5 is `main` at `964d070`, built in a worktree at a path of the same
  length as T6's: built from the main checkout's shorter path, the same
  commit reads `fanout_local_1000` 6,400 higher, so drivers are compared
  only at equal path lengths. Both columns are the mean of two runs of
  `icount.sh` at the default N.

| workload | N | T5 | T6 | delta | delta % | sources |
|---|--:|--:|--:|--:|--:|---|
| `chain_local/propagate/4` | 1,000,000 | 4,443.00 | 4,437.00 | -6.00 | -0.14 | none |
| `chain_local/propagate/10` | 1,000,000 | 9,591.00 | 9,579.00 | -12.00 | -0.13 | none |
| `chain_local/propagate/100` | 100,000 | 87,570.01 | 87,468.01 | -102.01 | -0.12 | none |
| `chain_shared/propagate/4` | 1,000,000 | 9,268.41 | 9,251.28 | -17.13 | -0.18 | none |
| `chain_shared/propagate/10` | 1,000,000 | 20,176.27 | 20,140.93 | -35.34 | -0.18 | none |
| `chain_shared/propagate/100` | 100,000 | 184,811.96 | 184,504.31 | -307.65 | -0.17 | none |
| `diamond_local` | 1,000,000 | 3,773.00 | 3,767.00 | -6.00 | -0.16 | none |
| `diamond_shared` | 1,000,000 | 7,737.09 | 7,721.97 | -15.12 | -0.20 | none |
| `early_cutoff_local` | 1,000,000 | 2,209.00 | 2,206.00 | -3.00 | -0.14 | none |
| `early_cutoff_shared` | 1,000,000 | 3,502.89 | 3,496.86 | -6.03 | -0.17 | none |
| `collection_insert_then_read/local_incremental/1000` | 100,000 | 4,489.77 | 4,486.07 | -3.70 | -0.08 | none |
| `collection_insert_then_read/shared_incremental/1000` | 100,000 | 7,193.55 | 7,182.96 | -10.59 | -0.15 | none |
| `collection_insert_then_read/batch/1000` | 10,000 | 92,272.86 | 92,273.42 | +0.56 | +0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/10000` | 100,000 | 4,483.61 | 4,479.75 | -3.86 | -0.09 | none |
| `collection_insert_then_read/shared_incremental/10000` | 100,000 | 7,167.06 | 7,158.23 | -8.83 | -0.12 | none |
| `collection_insert_then_read/batch/10000` | 1,000 | 158,876.07 | 158,875.90 | -0.16 | -0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/100000` | 100,000 | 4,491.07 | 4,487.65 | -3.43 | -0.08 | none |
| `collection_insert_then_read/shared_incremental/100000` | 100,000 | 7,173.01 | 7,162.91 | -10.10 | -0.14 | none |
| `collection_insert_then_read/batch/100000` | 1,000 | 1,490,560.76 | 1,490,562.31 | +1.55 | +0.00 | none (run spread) |
| `simple_count/local/1000` | 100,000 | 2,049.49 | 2,046.72 | -2.76 | -0.13 | none |
| `simple_count/shared/1000` | 100,000 | 4,030.98 | 4,022.86 | -8.12 | -0.20 | none |
| `simple_count/local/10000` | 100,000 | 2,045.96 | 2,043.47 | -2.49 | -0.12 | none |
| `simple_count/shared/10000` | 100,000 | 4,010.60 | 4,002.32 | -8.28 | -0.21 | none |
| `get_clean_local` | 10,000,000 | 98.00 | 97.00 | -1.00 | -1.02 | mismatch test inline in `get` |
| `get_clean_shared` | 10,000,000 | 200.00 | 199.00 | -1.00 | -0.50 | mismatch test inline in `get` |
| `get_contended_shared` | 2,000 | 7,953,180.26 | 7,913,174.82 | -40,005.44 | -0.50 | mismatch test inline in `get` |
| `fanout_local_4` | 100,000 | 18,173.00 | 18,167.01 | -5.99 | -0.03 | none |
| `fanout_local_5` | 100,000 | 22,889.93 | 22,875.12 | -14.81 | -0.06 | none |
| `fanout_local_1000` | 1,000 | 3,587,732.03 | 3,591,107.84 | +3,375.81 | +0.09 | allocator drift over the run (below) |
| `fanout_shared_4` | 100,000 | 24,957.61 | 24,925.97 | -31.65 | -0.13 | none |
| `fanout_shared_5` | 100,000 | 30,476.01 | 30,437.77 | -38.24 | -0.13 | none |
| `fanout_shared_1000` | 1,000 | 4,961,266.86 | 4,953,640.85 | -7,626.01 | -0.15 | none |
| `delete_single_local` | 200,000 | 1,156.00 | 1,155.00 | -1.00 | -0.09 | none |
| `delete_single_shared` | 200,000 | 2,013.24 | 2,006.30 | -6.94 | -0.34 | none |
| `delete_cascade_local` | 1,000 | 3,400,636.00 | 3,398,638.59 | -1,997.41 | -0.06 | none |
| `delete_cascade_shared` | 1,000 | 4,766,672.11 | 4,761,802.05 | -4,870.06 | -0.10 | none |
| `delete_owned_tree_local` | 10,000 | 605,446.12 | 601,208.88 | -4,237.24 | -0.70 | none |
| `delete_owned_tree_shared` | 10,000 | 989,330.76 | 982,317.73 | -7,013.03 | -0.71 | none |

New workloads (mean of two runs; the table counts include the driver's
key arithmetic):

| workload | N | T6 |
|---|--:|--:|
| `table_hit_local` | 10,000,000 | 155.14 |
| `table_hit_shared` | 10,000,000 | 218.14 |
| `table_hit_sip_local` | 10,000,000 | 300.84 |
| `table_miss_local` | 100,000 | 866.18 |
| `table_miss_shared` | 100,000 | 2,175.11 |
| `value_slot_read_local` | 10,000,000 | 5.00 |
| `value_slot_read_shared` | 10,000,000 | 87.00 |

- **Hit gate, in cycles** (`icount.sh -e cycles:u`, three runs on a quiet
  machine): `table_hit_shared − table_hit_local` was 50.05, 46.55 and
  49.14 against 1.1 × (`value_slot_read_shared − value_slot_read_local`)
  of 52.81, 56.80 and 52.32.
- **Probe bound:** a hit's probe loop counts its steps against the
  capacity, an increment and a compare per step, which the
  `table_hit_*` counts include.
- **`get_clean_*`, `get_contended_shared` (−1):** `get` and `try_get` test
  the handle's runtime id inline and branch to the cold `read_mismatch`,
  which reads a pending handle's placeholder; T5 called
  `check_runtime`.
- **`fanout_local_1000` (+3,376, +0.09%):** no engine symbol rises. Intel
  PT after 10 warm-up runs puts T6 1,830 below T5 (`get` −2,001), and
  `perf stat` at N = 100 and 300 puts it 1,950 and 2,560 below; only the
  default 1,100-iteration run reads higher, so the rise builds up in
  glibc's allocator over the run, whose layout moved with
  `GenericArena`'s new placeholder field.

Gate against T5 (plan T6 **Gate**): the Clean `get` is one below T5's
on both strategies, every other workload is at or below T5 except
`fanout_local_1000` (allocator, above) and the collection `batch` rows,
which are within run spread, and the hit gate holds in cycles.

## T7

- Engine: `8a8820f` (branch `spreadsheet-t7`). Machine and toolchain as
  for T0.
- T6 is `main` at `015829f`, built in a worktree at a path of the same
  length as T7's. The "T7, T6 benches" column is T7's library built with
  T6's bench sources, so it differs from the T7 column only in the bench
  binary; one run of `icount.sh` at the default N for each column.

| workload | N | T6 | T7, T6 benches | T7 | delta | delta % | sources |
|---|--:|--:|--:|--:|--:|--:|---|
| `chain_local/propagate/4` | 1,000,000 | 4,437.00 | 4,428.00 | 4,428.00 | -9.00 | -0.20 | none |
| `chain_local/propagate/10` | 1,000,000 | 9,579.00 | 9,558.00 | 9,558.00 | -21.00 | -0.22 | none |
| `chain_local/propagate/100` | 100,000 | 87,468.01 | 87,267.01 | 87,267.02 | -200.99 | -0.23 | none |
| `chain_shared/propagate/4` | 1,000,000 | 9,251.25 | 9,269.19 | 9,294.23 | +42.98 | +0.46 | state decode, `push_list`, `clean_walk` spills; bench build (below) |
| `chain_shared/propagate/10` | 1,000,000 | 20,140.92 | 20,188.87 | 20,243.86 | +102.94 | +0.51 | state decode, `push_list`, `clean_walk` spills; bench build (below) |
| `chain_shared/propagate/100` | 100,000 | 184,504.12 | 185,001.74 | 185,506.29 | +1,002.17 | +0.54 | state decode, `push_list`, `clean_walk` spills; bench build (below) |
| `diamond_local` | 1,000,000 | 3,767.00 | 3,760.00 | 3,760.00 | -7.00 | -0.19 | none |
| `diamond_shared` | 1,000,000 | 7,721.97 | 7,734.95 | 7,756.93 | +34.96 | +0.45 | state decode, `push_list`, `clean_walk` spills; bench build (below) |
| `early_cutoff_local` | 1,000,000 | 2,206.00 | 2,202.00 | 2,202.00 | -4.00 | -0.18 | none |
| `early_cutoff_shared` | 1,000,000 | 3,496.85 | 3,505.86 | 3,515.85 | +19.00 | +0.54 | state decode, `push_list`, `clean_walk` spills; bench build (below) |
| `collection_insert_then_read/local_incremental/1000` | 100,000 | 4,486.55 | 4,480.34 | 4,480.04 | -6.51 | -0.15 | none |
| `collection_insert_then_read/shared_incremental/1000` | 100,000 | 7,182.81 | 7,195.30 | 7,211.37 | +28.56 | +0.40 | state decode, `push_list`, `clean_walk` spills; bench build (below) |
| `collection_insert_then_read/batch/1000` | 10,000 | 92,273.83 | 92,273.66 | 92,273.84 | +0.01 | +0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/10000` | 100,000 | 4,479.35 | 4,473.59 | 4,473.94 | -5.41 | -0.12 | none |
| `collection_insert_then_read/shared_incremental/10000` | 100,000 | 7,158.26 | 7,169.41 | 7,184.97 | +26.71 | +0.37 | state decode, `push_list`, `clean_walk` spills; bench build (below) |
| `collection_insert_then_read/batch/10000` | 1,000 | 158,876.43 | 158,876.78 | 158,876.66 | +0.23 | +0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/100000` | 100,000 | 4,487.02 | 4,481.25 | 4,480.22 | -6.80 | -0.15 | none |
| `collection_insert_then_read/shared_incremental/100000` | 100,000 | 7,161.14 | 7,174.10 | 7,189.50 | +28.36 | +0.40 | state decode, `push_list`, `clean_walk` spills; bench build (below) |
| `collection_insert_then_read/batch/100000` | 1,000 | 1,490,560.08 | 1,490,559.45 | 1,490,550.38 | -9.70 | -0.00 | none (run spread) |
| `simple_count/local/1000` | 100,000 | 2,046.15 | 2,043.82 | 2,043.93 | -2.22 | -0.11 | none |
| `simple_count/shared/1000` | 100,000 | 4,023.35 | 4,025.82 | 4,035.19 | +11.84 | +0.29 | state decode, `push_list`, `clean_walk` spills; bench build (below) |
| `simple_count/local/10000` | 100,000 | 2,043.14 | 2,040.65 | 2,040.81 | -2.33 | -0.11 | none |
| `simple_count/shared/10000` | 100,000 | 4,002.73 | 4,005.37 | 4,016.15 | +13.42 | +0.34 | state decode, `push_list`, `clean_walk` spills; bench build (below) |
| `get_clean_local` | 10,000,000 | 97.00 | 97.00 | 97.00 | +0.00 | +0.00 | none |
| `get_clean_shared` | 10,000,000 | 199.00 | 199.00 | 201.00 | +2.00 | +1.01 | bench build (below) |
| `get_contended_shared` | 2,000 | 7,913,153.66 | 7,913,095.60 | 7,993,011.08 | +79,857.42 | +1.01 | bench build (below) |
| `fanout_local_4` | 100,000 | 18,167.01 | 16,683.01 | 16,683.00 | -1,484.01 | -8.17 | single-node claim and cascade units (below) |
| `fanout_local_5` | 100,000 | 22,875.13 | 21,151.00 | 21,151.00 | -1,724.13 | -7.54 | single-node claim and cascade units (below) |
| `fanout_local_1000` | 1,000 | 3,591,107.95 | 3,348,168.06 | 3,348,168.20 | -242,939.75 | -6.77 | single-node claim and cascade units (below) |
| `fanout_shared_4` | 100,000 | 24,925.97 | 23,488.21 | 23,492.05 | -1,433.92 | -5.75 | single-node claim and cascade units (below) |
| `fanout_shared_5` | 100,000 | 30,437.83 | 28,818.75 | 28,824.73 | -1,613.10 | -5.30 | single-node claim and cascade units (below) |
| `fanout_shared_1000` | 1,000 | 4,953,630.56 | 4,728,876.88 | 4,730,860.11 | -222,770.45 | -4.50 | single-node claim and cascade units (below) |
| `delete_single_local` | 200,000 | 1,155.00 | 726.00 | 726.00 | -429.00 | -37.14 | single-node claim and cascade units (below) |
| `delete_single_shared` | 200,000 | 2,006.30 | 1,574.25 | 1,570.24 | -436.06 | -21.73 | single-node claim and cascade units (below) |
| `delete_cascade_local` | 1,000 | 3,398,639.09 | 3,163,256.77 | 3,163,256.69 | -235,382.40 | -6.93 | single-node claim and cascade units (below) |
| `delete_cascade_shared` | 1,000 | 4,761,810.24 | 4,540,105.55 | 4,542,096.61 | -219,713.63 | -4.61 | single-node claim and cascade units (below) |
| `delete_owned_tree_local` | 10,000 | 602,559.81 | 606,699.94 | 602,402.29 | -157.52 | -0.03 | owned-set claim, offset (below) |
| `delete_owned_tree_shared` | 10,000 | 982,753.05 | 979,276.17 | 984,057.16 | +1,304.11 | +0.13 | owned-set claim (below) |
| `table_hit_local` | 10,000,000 | 155.14 | 155.14 | 155.14 | +0.00 | +0.00 | none |
| `table_hit_shared` | 10,000,000 | 218.14 | 218.14 | 219.14 | +1.00 | +0.46 | bench build (below) |
| `table_hit_sip_local` | 10,000,000 | 300.99 | 299.98 | 301.75 | +0.76 | +0.25 | bench build (below) |
| `table_miss_local` | 100,000 | 866.18 | 866.18 | 865.88 | -0.30 | -0.03 | none |
| `table_miss_shared` | 100,000 | 2,175.11 | 2,179.10 | 2,178.24 | +3.13 | +0.14 | state decode (below) |
| `value_slot_read_local` | 10,000,000 | 5.00 | 5.00 | 5.00 | +0.00 | +0.00 | none |
| `value_slot_read_shared` | 10,000,000 | 87.00 | 87.00 | 90.00 | +3.00 | +3.45 | bench build (below) |

New workloads:

| workload | N | T7 |
|---|--:|--:|
| `children_rerun_1k_local` | 10,000 | 217,910.27 |
| `children_rerun_1k_shared` | 10,000 | 236,896.61 |
| `children_churn_local` | 100,000 | 16,465.46 |
| `children_churn_shared` | 100,000 | 19,599.39 |

`children_rerun_1k_*` reruns a creator of 1,000 children that keeps
every key, so it measures the creator-only costs the plan names: the
index lookups, the touch per child and the Clean landing's sweep scan.
`children_churn_*` replaces one child per run, so it adds a sweep that
deletes.

- **Bench build (`get_clean_shared` +2, `value_slot_read_shared` +3,
  `table_hit_shared` +1, and about half of each Shared compute rise):**
  with T6's bench sources the T7 library reads 199, 87 and 218.14, equal
  to T6. Adding the children workloads to the bench binary moves how
  `SharedValueSlot::read` and its callers are partitioned and register
  allocated (Intel PT puts `value_slot_read_shared`'s +3 in `read`, 74 to
  77, with only register renames in the code). No library code differs
  between the two columns.
- **Shared computes, library part (`chain_shared/propagate/10` +47.95 per
  10 computes with T6 benches; `diamond_shared` +12.98,
  `early_cutoff_shared` +9.01, `simple_count/shared` +2.5,
  `shared_incremental` +12.5, `table_miss_shared` +4):** Intel PT on
  `chain_shared/propagate/10` puts the whole rise (+49.5) in three
  places. `clean_walk` +51 and its closure +18, less
  `mark_dependents_dirty` −41 whose code moved into it, nets +28: about
  20 of it is `NodeState::from_u8` decoding a ninth state, `Deleting`
  (the plan's state; removing the arm reads 20,167.86, and the code shows
  the jump table's added bound check), and the rest is spills and
  register renames in the walk. haphazard's `Domain::push_list`
  +21.9 is the bulk-reclaim cadence inside the crate: its loop
  instructions run 12,013 and 24,026 times against 104 and 97 in T6, as
  the shard it picks by each retired box's address changes with heap
  layout. Removing T7's other per-compute pieces one at a time (the
  walk's and the compute's `Deleting` arms, the children landing test
  and the control block's `children` field) left the count within 2 of
  20,188. Local computes pay the same decode (without the arm
  `chain_local/propagate/10` reads 9,538 against 9,558) and still read
  below T6.
- **Single-node claim and cascade units (`delete_single_*` −430,
  `delete_cascade_*` and `fanout_*` −4.5% to −8%):** `delete_node` and
  `delete_erased` claim one node with `claim_single` inline, where T6
  built a tree claim for every delete; the cascade's unit test, stack
  segments and bucket counts are the plan's named per-node costs, and the
  `OWNED` flag lets `unhook_retired` skip the `owner_of` and `labels`
  lookups for nodes that were never owned, which more than pays for them.
- **Owned-set claim (`delete_owned_tree_shared` +1,304, +0.13%;
  `delete_owned_tree_local` +4,140 with T6 benches, −158 with T7's):**
  `claim_set` takes the owned closure under one write hold, CASes each
  member to `Deleting` and then `Dead` (two CASes where T6's tree claim
  made one) and checks every member for a dependent outside the set, the
  costs the plan's gate names for this bench.

Gate against T6 (plan T7 **Gate**): every Local workload is at or below
T6; every Shared rise is the bench binary's partitioning, the ninth
state's decode, haphazard's reclaim cadence or the walk's register
allocation, each measured above; the delete benches fall except the
owned-tree pair, whose rise is the plan's named `claim_set` cost.
