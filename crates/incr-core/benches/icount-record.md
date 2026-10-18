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
