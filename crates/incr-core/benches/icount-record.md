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

## T8

- Engine: `92521e2` (branch `spreadsheet-t8`). Machine and toolchain as
  for T0.
- T7 is `main` at `9346b5a`, built in a worktree at a path of the same
  length as T8's. The "T8, T7 benches" column is T8's library built with
  T7's bench sources, which T8 does not change, so the two T8 columns
  differ only by run; one run of `icount.sh` at the default N for each
  column. T8 adds no workloads.

| workload | N | T7 | T8, T7 benches | T8 | delta | delta % | sources |
|---|--:|--:|--:|--:|--:|--:|---|
| `chain_local/propagate/4` | 1,000,000 | 4,428.00 | 4,612.00 | 4,612.00 | +184.00 | +4.16 | per-compute and per-get sources (below) |
| `chain_local/propagate/10` | 1,000,000 | 9,558.00 | 9,976.00 | 9,976.00 | +418.00 | +4.37 | per-compute and per-get sources (below) |
| `chain_local/propagate/100` | 100,000 | 87,267.03 | 91,195.06 | 91,195.11 | +3,928.08 | +4.50 | per-compute and per-get sources (below) |
| `chain_shared/propagate/4` | 1,000,000 | 9,294.23 | 9,349.25 | 9,349.29 | +55.06 | +0.59 | per-compute and per-get sources (below) |
| `chain_shared/propagate/10` | 1,000,000 | 20,243.86 | 20,352.98 | 20,352.88 | +109.02 | +0.54 | per-compute and per-get sources (below) |
| `chain_shared/propagate/100` | 100,000 | 185,505.97 | 186,426.40 | 186,426.13 | +920.16 | +0.50 | per-compute and per-get sources (below) |
| `diamond_local` | 1,000,000 | 3,760.00 | 3,903.01 | 3,903.01 | +143.01 | +3.80 | per-compute and per-get sources (below) |
| `diamond_shared` | 1,000,000 | 7,756.93 | 7,787.96 | 7,787.98 | +31.05 | +0.40 | per-compute and per-get sources (below) |
| `early_cutoff_local` | 1,000,000 | 2,202.00 | 2,280.01 | 2,280.00 | +78.00 | +3.54 | per-compute and per-get sources (below) |
| `early_cutoff_shared` | 1,000,000 | 3,515.85 | 3,559.86 | 3,559.85 | +44.00 | +1.25 | per-compute and per-get sources (below) |
| `collection_insert_then_read/local_incremental/1000` | 100,000 | 4,480.84 | 4,710.52 | 4,710.67 | +229.83 | +5.13 | per-compute and per-get sources (below) |
| `collection_insert_then_read/shared_incremental/1000` | 100,000 | 7,210.94 | 7,349.15 | 7,349.72 | +138.78 | +1.92 | per-compute and per-get sources (below) |
| `collection_insert_then_read/batch/1000` | 10,000 | 92,272.75 | 92,272.30 | 92,273.35 | +0.60 | +0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/10000` | 100,000 | 4,473.69 | 4,704.78 | 4,703.76 | +230.07 | +5.14 | per-compute and per-get sources (below) |
| `collection_insert_then_read/shared_incremental/10000` | 100,000 | 7,184.80 | 7,323.85 | 7,324.04 | +139.24 | +1.94 | per-compute and per-get sources (below) |
| `collection_insert_then_read/batch/10000` | 1,000 | 158,876.70 | 158,876.29 | 158,875.51 | -1.19 | -0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/100000` | 100,000 | 4,481.54 | 4,710.89 | 4,711.25 | +229.71 | +5.13 | per-compute and per-get sources (below) |
| `collection_insert_then_read/shared_incremental/100000` | 100,000 | 7,190.73 | 7,327.33 | 7,329.18 | +138.45 | +1.93 | per-compute and per-get sources (below) |
| `collection_insert_then_read/batch/100000` | 1,000 | 1,490,561.29 | 1,490,551.57 | 1,490,561.74 | +0.45 | +0.00 | none (run spread) |
| `simple_count/local/1000` | 100,000 | 2,043.51 | 2,109.70 | 2,109.56 | +66.05 | +3.23 | per-compute and per-get sources (below) |
| `simple_count/shared/1000` | 100,000 | 4,035.66 | 4,062.80 | 4,062.55 | +26.89 | +0.67 | per-compute and per-get sources (below) |
| `simple_count/local/10000` | 100,000 | 2,040.15 | 2,106.48 | 2,106.37 | +66.22 | +3.25 | per-compute and per-get sources (below) |
| `simple_count/shared/10000` | 100,000 | 4,015.35 | 4,043.02 | 4,042.69 | +27.34 | +0.68 | per-compute and per-get sources (below) |
| `get_clean_local` | 10,000,000 | 97.00 | 102.00 | 102.00 | +5.00 | +5.15 | thread-wide frame stack (below) |
| `get_clean_shared` | 10,000,000 | 201.00 | 196.00 | 196.00 | -5.00 | -2.49 | frame stack thread-local (below) |
| `get_contended_shared` | 2,000 | 7,993,284.39 | 7,791,636.96 | 7,791,679.35 | -201,605.04 | -2.52 | frame stack thread-local (below) |
| `fanout_local_4` | 100,000 | 16,683.01 | 17,205.04 | 17,205.00 | +521.99 | +3.13 | per-compute and per-get sources; delete path (below) |
| `fanout_local_5` | 100,000 | 21,151.01 | 21,781.05 | 21,781.01 | +630.00 | +2.98 | per-compute and per-get sources; delete path (below) |
| `fanout_local_1000` | 1,000 | 3,348,168.25 | 3,455,272.78 | 3,455,269.25 | +107,101.00 | +3.20 | per-compute and per-get sources; delete path (below) |
| `fanout_shared_4` | 100,000 | 23,492.03 | 23,633.18 | 23,633.12 | +141.09 | +0.60 | per-compute and per-get sources; delete path (below) |
| `fanout_shared_5` | 100,000 | 28,824.78 | 29,003.74 | 29,003.71 | +178.93 | +0.62 | per-compute and per-get sources; delete path (below) |
| `fanout_shared_1000` | 1,000 | 4,730,861.58 | 4,767,399.36 | 4,767,388.41 | +36,526.83 | +0.77 | per-compute and per-get sources; delete path (below) |
| `delete_single_local` | 200,000 | 726.00 | 730.00 | 730.00 | +4.00 | +0.55 | delete path (below) |
| `delete_single_shared` | 200,000 | 1,570.25 | 1,546.24 | 1,546.24 | -24.01 | -1.53 | delete path (below) |
| `delete_cascade_local` | 1,000 | 3,163,258.32 | 3,246,233.20 | 3,246,233.65 | +82,975.33 | +2.62 | per-compute and per-get sources; delete path (below) |
| `delete_cascade_shared` | 1,000 | 4,542,105.65 | 4,550,927.86 | 4,550,918.71 | +8,813.06 | +0.19 | per-compute and per-get sources; delete path (below) |
| `delete_owned_tree_local` | 10,000 | 603,427.40 | 604,532.82 | 607,202.29 | +3,774.89 | +0.63 | run spread (below) |
| `delete_owned_tree_shared` | 10,000 | 982,399.12 | 980,140.43 | 979,479.05 | -2,920.07 | -0.30 | none |
| `table_hit_local` | 10,000,000 | 155.14 | 155.14 | 155.14 | +0.00 | +0.00 | none |
| `table_hit_shared` | 10,000,000 | 219.14 | 218.14 | 218.14 | -1.00 | -0.46 | none |
| `table_hit_sip_local` | 10,000,000 | 301.08 | 300.32 | 301.84 | +0.76 | +0.25 | none (run spread) |
| `table_miss_local` | 100,000 | 865.87 | 871.17 | 871.18 | +5.31 | +0.61 | per-compute and per-get sources (below) |
| `table_miss_shared` | 100,000 | 2,178.10 | 2,164.21 | 2,164.22 | -13.88 | -0.64 | none |
| `value_slot_read_local` | 10,000,000 | 5.00 | 5.00 | 5.00 | +0.00 | +0.00 | none |
| `value_slot_read_shared` | 10,000,000 | 90.00 | 87.00 | 87.00 | -3.00 | -3.33 | none |
| `children_rerun_1k_local` | 10,000 | 217,910.10 | 217,977.58 | 217,977.16 | +67.06 | +0.03 | per-compute and per-get sources (below) |
| `children_rerun_1k_shared` | 10,000 | 236,896.63 | 224,925.55 | 224,925.11 | -11,971.52 | -5.05 | haphazard reclaim cadence (below) |
| `children_churn_local` | 100,000 | 16,465.47 | 16,532.24 | 16,530.69 | +65.22 | +0.40 | per-compute and per-get sources (below) |
| `children_churn_shared` | 100,000 | 19,602.39 | 19,420.17 | 19,417.11 | -185.28 | -0.95 | none |

- **Per-compute and per-get sources (every workload that computes).**
  A walk-driven compute costs +39 instructions under Local and +9 under
  Shared, from the slope of `chain_*/propagate/100` against `/10`
  ((3,928.08 − 418.00) / 90 and (920.16 − 109.02) / 90). Intel PT per
  instruction address on `chain_local/propagate/10`, with line-table
  builds of T7 and T8 and each source line mapped across the T7-to-T8
  diff, splits the Local +39 as follows:
  - named T8 lines, about +6 per compute: the frame stack's cached
    pointer and borrow (+1, below), the first-push capacity test that
    registers the stack's guard (+2), the frame index bias the replay
    driver needs (+2), the `DOOMED` test folded into the `LOW` test at
    the pop (+1) and the `top.rt` compare (+1); folding `REPLAY_BIT`
    into the post-pop word test saves 1;
  - `for_each_dep`, out of line in T7 (380 per 10 computes), is inlined
    into `clean_walk`; its body lands on the iterator, cell and range
    lines inside the walk and nets about zero;
  - the rest, about +33 per compute, is a2077e7's larger inlined compute
    body. `clean_walk` inlines `compute_attempts`, and with T8's replay
    arms in it the walk executes per compute +16 stack loads, +6 stack
    stores, +14 compares and +11 branches on lines whose source did not
    change. Of that, +4 Local and +8 Shared came with the `Skipped` result
    (579f0a4, merged in bb737b6), whose arms run only under a replay
    context; a `Skipped` without its payload measures −1 per 10 computes,
    so the rise is the variant's effect on the walk's register
    allocation, not executed work.

  By commit (first-parent sweep of `spreadsheet-t8` built with T7's
  bench sources, `chain_local/propagate/10` and `chain_shared/propagate/10`,
  per 10 computes): commits before a2077e7 −1 and +10; a2077e7 (the
  replay feature) +655 and +739, of which an out-of-line `Frames::frame`
  was 310 under Local; 1a57101 (entry, frame push and landing paths
  inline) −342 and −449; 49cbc8a +70 and +40; bb737b6 +41 and +78;
  aba5fe3, dfb67d4, bdc8ebe and 42b8c2b together −37 and −60; the
  thread-wide frame stack (8c8a521 with fe65b0e) +59 and −141, measured
  against fe65b0e with 8c8a521 reverted; the other commits up to
  fe65b0e −26 and −67; b393429 −2 and −40. The steps sum to the table's
  +418 and +109.

  A codegen pass (T8-walk-codegen) tried to get the +33 back: one cold
  call for the post-pop replay checks (b393429, kept: −2 Local, −40
  Shared per 10 computes); `compute_attempts` out of line (+99 and +322,
  rejected); `for_each_dep` out of line (+366 Local, rejected);
  `walk_stopped` deriving its visit itself, so the loop keeps fewer
  values live (0, dropped).

  Every top-level non-Clean `get` adds +28 Local and +21 Shared in
  `get_slow`: about +11 on T8 lines (the entry guard, the active-context
  test, the frame checks after the walk) and +16 from two more node
  lookups (`segmented_nodes` 8 to 21, `directory` 6 to 9 per call) that
  re-derive node pointers T7 kept across the walk call. A nested
  non-Clean read (`fanout_*`) adds +45 per `get_slow`: +12 on T8 lines
  (`record_read`'s `reading` store and `ActiveRead`, the depth compare
  and the `DOOMED` test), +16 from the same lookups and +17 of glue.

  The workloads compose from these: `diamond_local` +143,
  `early_cutoff_local` +78, `simple_count/local` +66 and the children
  pair (+67 and +65 Local) are one top-level get plus their computes;
  `collection_insert_then_read/local_incremental` +230 adds to that
  `Provisional::drop` moving out of `insert_staged` (+69 and −104) and
  +44 in the collection's compute closure; `delete_cascade_local`
  +82,975 is its 1,000 setup gets and computes (about +67 each, so about
  +67,000) and about +16 per node of the cascade's delete path, which
  this record does not split further: Intel PT loses AUX data on the
  cascade benches at every N. Shared pays the same sources at its
  smaller per-compute cost.
- **Thread-wide frame stack (`get_clean_local` +5):** Local Clean get
  +5 is the thread-wide frame stack's TLS address and the `top.rt`
  compare the cross-runtime rule needs. The stack's thread-local has no
  destructor (its buffer is freed by a guard registered on the first
  push) and each Local runtime caches its address at `Runtime::new`
  (fe65b0e), which took the count from 109 to 102.
- **Frame stack thread-local (`get_clean_shared` −5,
  `get_contended_shared` −2.52%):** the same destructor-free
  thread-local takes Shared's Clean get from 201 to 196; the contended
  bench is Clean gets on four threads and falls by the same fraction
  (−2.49% and −2.52%).
- **Delete path (`delete_single_*`, and the delete half of
  `fanout_*` and `delete_cascade_*`):** every delete tests `REPLAY_BIT`
  in the thread word before the virtual-frame scan (483a1c8), and the
  scan sits in a cold function so the own-compute assert stays inline
  (d37613e): `delete_single_local` +4 (726 to 730), `delete_single_shared`
  −24. Before those two fixes the scan ran on every delete
  (`delete_single_local` 807).
- **Haphazard reclaim cadence (`children_rerun_1k_shared` −11,972):**
  the drop is in the allocator and haphazard's `Domain::push_list`, the
  bulk-reclaim cadence T7's note describes: a two-iteration Intel PT
  trace puts `push_list` at 41,156 per iteration in T7 against 94 in T8.
  That workload's traces lose AUX data at every N, so the split is not
  exact.
- **Run spread (`delete_owned_tree_local` +3,775, +0.63%):** the two T8
  columns, built from the same library and bench sources, read 604,533
  and 607,202, and T7 read 602,402 in T7's record and 603,427 here, so
  the bench moves by up to about 2,700 between runs of one binary;
  against the "T8, T7 benches" column the rise is +1,105 (+0.18%).

Gate against T7 (plan T8 **Gate**): the Clean get is 102 Local (+5,
the thread-wide frame stack, a recorded departure from "unchanged") and
196 Shared (−5). Every other rise is the named per-read and per-delete
costs plus a2077e7's larger inlined compute body in `clean_walk`, about
+33 per walk-driven compute under Local (Shared's whole per-compute
rise is +9), which the gate's "adds nothing per visit" did not foresee;
the codegen pass above recovered under 10 of it.

## T-core-publish-deps-quadratic

- Engine: `aebe6b6` (branch `w/T-core-publish-deps-quadratic`), against
  its base `c63f074` (`spreadsheet-t9`), both built in the same
  worktree. One run of `icount.sh` at the default N for each column.
  No workloads added.

| workload | N | base | after | delta | delta % | sources |
|---|--:|--:|--:|--:|--:|---|
| `chain_local/propagate/4` | 1,000,000 | 4,646.02 | 4,646.01 | -0.01 | -0.00 | none |
| `chain_local/propagate/10` | 1,000,000 | 10,034.02 | 10,034.03 | +0.01 | +0.00 | none |
| `chain_local/propagate/100` | 100,000 | 91,613.31 | 91,613.14 | -0.17 | -0.00 | none |
| `chain_shared/propagate/4` | 1,000,000 | 9,382.26 | 9,382.27 | +0.01 | +0.00 | none |
| `chain_shared/propagate/10` | 1,000,000 | 20,415.96 | 20,415.91 | -0.05 | -0.00 | none |
| `chain_shared/propagate/100` | 100,000 | 186,939.55 | 186,939.17 | -0.38 | -0.00 | none |
| `diamond_local` | 1,000,000 | 3,934.01 | 3,934.00 | -0.01 | -0.00 | none |
| `diamond_shared` | 1,000,000 | 7,818.00 | 7,817.99 | -0.01 | -0.00 | none |
| `early_cutoff_local` | 1,000,000 | 2,306.00 | 2,306.00 | +0.00 | +0.00 | none |
| `early_cutoff_shared` | 1,000,000 | 3,583.86 | 3,583.86 | +0.00 | +0.00 | none |
| `collection_insert_then_read/local_incremental/1000` | 100,000 | 4,740.95 | 4,741.07 | +0.12 | +0.00 | none |
| `collection_insert_then_read/shared_incremental/1000` | 100,000 | 7,379.06 | 7,378.89 | -0.17 | -0.00 | none |
| `collection_insert_then_read/batch/1000` | 10,000 | 92,273.88 | 92,273.13 | -0.75 | -0.00 | none |
| `collection_insert_then_read/local_incremental/10000` | 100,000 | 4,734.58 | 4,734.16 | -0.42 | -0.01 | none |
| `collection_insert_then_read/shared_incremental/10000` | 100,000 | 7,353.25 | 7,353.53 | +0.28 | +0.00 | none |
| `collection_insert_then_read/batch/10000` | 1,000 | 158,876.88 | 158,875.49 | -1.39 | -0.00 | none |
| `collection_insert_then_read/local_incremental/100000` | 100,000 | 4,742.19 | 4,740.45 | -1.74 | -0.04 | none |
| `collection_insert_then_read/shared_incremental/100000` | 100,000 | 7,357.87 | 7,357.75 | -0.12 | -0.00 | none |
| `collection_insert_then_read/batch/100000` | 1,000 | 1,490,564.46 | 1,490,574.47 | +10.01 | +0.00 | none |
| `simple_count/local/1000` | 100,000 | 2,131.77 | 2,131.88 | +0.11 | +0.01 | none |
| `simple_count/shared/1000` | 100,000 | 4,080.93 | 4,080.65 | -0.28 | -0.01 | none |
| `simple_count/local/10000` | 100,000 | 2,127.98 | 2,129.07 | +1.09 | +0.05 | none |
| `simple_count/shared/10000` | 100,000 | 4,060.52 | 4,060.35 | -0.17 | -0.00 | none |
| `get_clean_local` | 10,000,000 | 102.00 | 102.00 | +0.00 | +0.00 | none |
| `get_clean_shared` | 10,000,000 | 196.00 | 196.00 | +0.00 | +0.00 | none |
| `get_contended_shared` | 2,000 | 7,791,665.07 | 7,791,645.85 | -19.22 | -0.00 | none |
| `fanout_local_4` | 100,000 | 17,067.01 | 17,140.02 | +73.01 | +0.43 | changed-dep publish (below) |
| `fanout_local_5` | 100,000 | 21,620.04 | 21,708.01 | +87.97 | +0.41 | changed-dep publish (below) |
| `fanout_local_1000` | 1,000 | 3,432,230.77 | 3,447,237.82 | +15,007.05 | +0.44 | changed-dep publish (below) |
| `fanout_shared_4` | 100,000 | 23,502.23 | 23,510.07 | +7.84 | +0.03 | none |
| `fanout_shared_5` | 100,000 | 28,850.83 | 28,858.69 | +7.86 | +0.03 | none |
| `fanout_shared_1000` | 1,000 | 4,745,362.78 | 4,745,364.66 | +1.88 | +0.00 | none |
| `delete_single_local` | 200,000 | 707.00 | 707.00 | +0.00 | +0.00 | none |
| `delete_single_shared` | 200,000 | 1,527.25 | 1,527.24 | -0.01 | -0.00 | none |
| `delete_cascade_local` | 1,000 | 3,223,212.18 | 3,237,210.01 | +13,997.83 | +0.43 | changed-dep publish (below) |
| `delete_cascade_shared` | 1,000 | 4,528,908.39 | 4,532,910.59 | +4,002.20 | +0.09 | changed-dep publish (below) |
| `delete_owned_tree_local` | 10,000 | 604,062.54 | 604,814.43 | +751.89 | +0.12 | run spread (below) |
| `delete_owned_tree_shared` | 10,000 | 985,127.63 | 985,033.41 | -94.22 | -0.01 | none |
| `table_hit_local` | 10,000,000 | 155.14 | 155.14 | +0.00 | +0.00 | none |
| `table_hit_shared` | 10,000,000 | 218.14 | 218.14 | +0.00 | +0.00 | none |
| `table_hit_sip_local` | 10,000,000 | 300.22 | 300.91 | +0.69 | +0.23 | none (run spread) |
| `table_miss_local` | 100,000 | 871.16 | 871.17 | +0.01 | +0.00 | none |
| `table_miss_shared` | 100,000 | 2,164.22 | 2,164.23 | +0.01 | +0.00 | none |
| `value_slot_read_local` | 10,000,000 | 5.00 | 5.00 | +0.00 | +0.00 | none |
| `value_slot_read_shared` | 10,000,000 | 87.00 | 87.00 | +0.00 | +0.00 | none |
| `children_rerun_1k_local` | 10,000 | 217,999.08 | 217,999.21 | +0.13 | +0.00 | none |
| `children_rerun_1k_shared` | 10,000 | 224,944.41 | 224,944.96 | +0.55 | +0.00 | none |
| `children_churn_local` | 100,000 | 16,532.53 | 16,531.70 | -0.83 | -0.01 | none |
| `children_churn_shared` | 100,000 | 19,416.99 | 19,417.68 | +0.69 | +0.00 | none |

- **Changed-dep publish (every `publish_deps` that misses the static
  fast path; under Local +14 per call, under Shared +4).** Every first
  compute takes it, which is all of `delete_cascade_*`'s setup (1,000
  per iteration) and `fanout_*`'s (`width + 1` per iteration); the
  static-dep fast path is untouched, so `chain_*`, `diamond_*` and
  `early_cutoff_*` read the same. Intel PT on `fanout_local_4` (20
  iterations, no lost data) puts the whole Local rise in
  `publish_deps`: 169 straight-line instructions per call against 156,
  plus one more per pass of a loop that runs 1.6 times per call. About 5 are the
  length test that picks the scan or the hashed path (two lengths
  loaded from the stack, two `setae`, a branch); the rest is register
  reallocation around the new cold call (+4 stack moves, a loop bound
  compared on the stack). Variants measured and dropped: an enum
  holding either a slice or a hash set (about +80 Local per call), the
  length test and the generic diff left to the inliner (+17 Local, +14
  Shared), and deciding
  from the dep-count cell before the lists are built (+55 Local).
- **Run spread (`delete_owned_tree_local` +752):** inside T8's
  recorded spread of about 2,700 between runs of one binary.

## T9

- Engine: `9fc5a3f` (branch `spreadsheet-t9`). Machine and toolchain as
  for T0.
- T8 is `main` at `3127150`, built in a worktree at a path of the same
  length as T9's. T9 changes no bench source (only this file under
  `benches/`), so the "T9, T8 benches" column is the same binary as
  "T9" and the two differ only by run; one run of `icount.sh` at the
  default N for each column. T9 adds no workloads.
- The split by merge comes from two more runs at the same path length:
  `c63f074`, which is T8 (all but a test-only commit) plus T9's watch
  queue, `observer_read`, `stabilize` and the failed-observer fix, and
  `2e10c62`, which adds the `get_slow` node pass-through. The rest,
  up to `9fc5a3f`, is `publish_deps` (its own section above); the other
  merges in between change no incr-core source but tests.

| workload | N | T8 | T9, T8 benches | T9 | delta | delta % | sources |
|---|--:|--:|--:|--:|--:|--:|---|
| `chain_local/propagate/4` | 1,000,000 | 4,612.00 | 4,638.00 | 4,638.00 | +26.00 | +0.56 | watch sources; `get_slow` node pass-through (below) |
| `chain_local/propagate/10` | 1,000,000 | 9,976.01 | 10,026.02 | 10,026.00 | +49.99 | +0.50 | watch sources; `get_slow` node pass-through (below) |
| `chain_local/propagate/100` | 100,000 | 91,195.13 | 91,605.11 | 91,605.10 | +409.97 | +0.45 | watch sources; `get_slow` node pass-through (below) |
| `chain_shared/propagate/4` | 1,000,000 | 9,349.26 | 9,375.24 | 9,375.28 | +26.02 | +0.28 | watch sources; `get_slow` node pass-through (below) |
| `chain_shared/propagate/10` | 1,000,000 | 20,352.89 | 20,408.97 | 20,408.95 | +56.06 | +0.28 | watch sources; `get_slow` node pass-through (below) |
| `chain_shared/propagate/100` | 100,000 | 186,426.01 | 186,931.85 | 186,932.63 | +506.62 | +0.27 | watch sources; `get_slow` node pass-through (below) |
| `diamond_local` | 1,000,000 | 3,903.01 | 3,926.02 | 3,926.00 | +22.99 | +0.59 | watch sources; `get_slow` node pass-through (below) |
| `diamond_shared` | 1,000,000 | 7,787.97 | 7,810.97 | 7,810.98 | +23.01 | +0.30 | watch sources; `get_slow` node pass-through (below) |
| `early_cutoff_local` | 1,000,000 | 2,280.00 | 2,298.01 | 2,298.00 | +18.00 | +0.79 | watch sources; `get_slow` node pass-through (below) |
| `early_cutoff_shared` | 1,000,000 | 3,559.85 | 3,576.85 | 3,576.86 | +17.01 | +0.48 | watch sources; `get_slow` node pass-through (below) |
| `collection_insert_then_read/local_incremental/1000` | 100,000 | 4,711.21 | 4,732.34 | 4,732.70 | +21.49 | +0.46 | watch sources; `get_slow` node pass-through (below) |
| `collection_insert_then_read/shared_incremental/1000` | 100,000 | 7,349.83 | 7,371.86 | 7,372.40 | +22.57 | +0.31 | watch sources; `get_slow` node pass-through (below) |
| `collection_insert_then_read/batch/1000` | 10,000 | 92,274.01 | 92,273.56 | 92,273.82 | -0.19 | -0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/10000` | 100,000 | 4,703.99 | 4,726.00 | 4,726.00 | +22.01 | +0.47 | watch sources; `get_slow` node pass-through (below) |
| `collection_insert_then_read/shared_incremental/10000` | 100,000 | 7,324.14 | 7,346.28 | 7,345.86 | +21.72 | +0.30 | watch sources; `get_slow` node pass-through (below) |
| `collection_insert_then_read/batch/10000` | 1,000 | 158,876.64 | 158,875.23 | 158,875.61 | -1.03 | -0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/100000` | 100,000 | 4,712.72 | 4,733.94 | 4,731.98 | +19.26 | +0.41 | watch sources; `get_slow` node pass-through (below) |
| `collection_insert_then_read/shared_incremental/100000` | 100,000 | 7,329.27 | 7,351.01 | 7,352.31 | +23.04 | +0.31 | watch sources; `get_slow` node pass-through (below) |
| `collection_insert_then_read/batch/100000` | 1,000 | 1,490,566.08 | 1,490,567.19 | 1,490,559.01 | -7.07 | -0.00 | none (run spread) |
| `simple_count/local/1000` | 100,000 | 2,109.99 | 2,123.80 | 2,124.31 | +14.32 | +0.68 | watch sources; `get_slow` node pass-through (below) |
| `simple_count/shared/1000` | 100,000 | 4,062.83 | 4,073.62 | 4,073.28 | +10.45 | +0.26 | watch sources; `get_slow` node pass-through (below) |
| `simple_count/local/10000` | 100,000 | 2,105.87 | 2,120.22 | 2,120.30 | +14.43 | +0.69 | watch sources; `get_slow` node pass-through (below) |
| `simple_count/shared/10000` | 100,000 | 4,042.49 | 4,053.84 | 4,053.79 | +11.30 | +0.28 | watch sources; `get_slow` node pass-through (below) |
| `get_clean_local` | 10,000,000 | 102.00 | 102.00 | 102.00 | +0.00 | +0.00 | none |
| `get_clean_shared` | 10,000,000 | 196.00 | 196.00 | 196.00 | +0.00 | +0.00 | none |
| `get_contended_shared` | 2,000 | 7,791,623.01 | 7,791,671.33 | 7,791,677.22 | +54.21 | +0.00 | none (run spread) |
| `fanout_local_4` | 100,000 | 17,205.06 | 17,112.06 | 17,112.06 | -93.00 | -0.54 | delete path; `get_slow` node pass-through; changed-dep publish (below) |
| `fanout_local_5` | 100,000 | 21,781.00 | 21,675.01 | 21,675.00 | -106.00 | -0.49 | delete path; `get_slow` node pass-through; changed-dep publish (below) |
| `fanout_local_1000` | 1,000 | 3,456,502.41 | 3,443,455.17 | 3,443,461.98 | -13,040.43 | -0.38 | delete path; `get_slow` node pass-through; changed-dep publish (below) |
| `fanout_shared_4` | 100,000 | 23,633.11 | 23,467.12 | 23,467.11 | -166.00 | -0.70 | delete path; `get_slow` node pass-through; changed-dep publish (below) |
| `fanout_shared_5` | 100,000 | 29,002.87 | 28,805.91 | 28,806.04 | -196.83 | -0.68 | delete path; `get_slow` node pass-through; changed-dep publish (below) |
| `fanout_shared_1000` | 1,000 | 4,769,160.10 | 4,738,126.07 | 4,738,137.44 | -31,022.66 | -0.65 | delete path; `get_slow` node pass-through; changed-dep publish (below) |
| `delete_single_local` | 200,000 | 730.01 | 707.00 | 707.00 | -23.01 | -3.15 | delete path (below) |
| `delete_single_shared` | 200,000 | 1,546.27 | 1,527.24 | 1,527.24 | -19.03 | -1.23 | delete path (below) |
| `delete_cascade_local` | 1,000 | 3,246,002.48 | 3,228,976.97 | 3,228,976.45 | -17,026.03 | -0.52 | delete path; `get_slow` node pass-through; changed-dep publish (below) |
| `delete_cascade_shared` | 1,000 | 4,550,304.43 | 4,525,268.31 | 4,525,277.50 | -25,026.93 | -0.55 | delete path; `get_slow` node pass-through; changed-dep publish (below) |
| `delete_owned_tree_local` | 10,000 | 607,278.40 | 603,752.12 | 603,848.16 | -3,430.24 | -0.56 | run spread (below) |
| `delete_owned_tree_shared` | 10,000 | 979,617.18 | 984,745.36 | 985,570.65 | +5,953.47 | +0.61 | run spread (below) |
| `table_hit_local` | 10,000,000 | 155.14 | 155.14 | 155.14 | +0.00 | +0.00 | none |
| `table_hit_shared` | 10,000,000 | 218.14 | 218.14 | 218.14 | +0.00 | +0.00 | none |
| `table_hit_sip_local` | 10,000,000 | 301.68 | 301.24 | 300.64 | -1.04 | -0.34 | none |
| `table_miss_local` | 100,000 | 871.17 | 871.18 | 871.18 | +0.01 | +0.00 | none |
| `table_miss_shared` | 100,000 | 2,164.22 | 2,164.22 | 2,164.23 | +0.01 | +0.00 | none |
| `value_slot_read_local` | 10,000,000 | 5.00 | 5.00 | 5.00 | +0.00 | +0.00 | none |
| `value_slot_read_shared` | 10,000,000 | 87.00 | 87.00 | 87.00 | +0.00 | +0.00 | none |
| `children_rerun_1k_local` | 10,000 | 217,976.94 | 217,991.05 | 217,991.11 | +14.17 | +0.01 | watch sources; `get_slow` node pass-through (below) |
| `children_rerun_1k_shared` | 10,000 | 224,925.35 | 224,938.15 | 224,937.25 | +11.90 | +0.01 | watch sources; `get_slow` node pass-through (below) |
| `children_churn_local` | 100,000 | 16,530.70 | 16,523.74 | 16,523.71 | -6.99 | -0.04 | `get_slow` node pass-through; layout (below) |
| `children_churn_shared` | 100,000 | 19,417.93 | 19,410.01 | 19,408.46 | -9.47 | -0.05 | `get_slow` node pass-through; layout (below) |

- **Watch sources (every workload that sets or computes; `c63f074`).**
  A walk-driven compute costs +4 Local and +5 Shared, from the slope of
  `chain_*/propagate/100` against `/10` ((417.96 − 58.00) / 90 and
  (512.67 − 63.04) / 90), and each `set` with its walk +18 Local and +13
  Shared. Intel PT per source line, with line-table builds of T8 and
  `c63f074` and each line mapped across the diff
  (`chain_local/propagate/10` and `/100`, `chain_shared/propagate/10`;
  the Shared `/100` trace loses AUX data), splits them:
  - per node the dirty walk flips, the `WATCHED` byte load and branch
    the plan names (Local +2, Shared +3) and the flags pointer that
    `get_with_flags` derives for it (Local +1, Shared +2);
  - per landing, the test that the target is neither Clean nor Cyclic
    before `report_landing` (+2); every icount landing is Clean or
    Cyclic, so `report_landing`'s `inner.read()` never runs;
  - per compute, register reallocation in the compute path: `land_run`'s
    call line −2, the `verifying` test +1 Local, the `compute` call +1
    Shared; under Local the walk's `inner.read()` reloads its borrow
    flag once more per flipped node (`cell.rs:955`, +1), offset by one
    fewer instruction with no line information in `runtime.rs` (−1);
  - per `set`, the own-input `WATCHED` test the plan names (+2), passing
    the input's flags through `claim_input`, `SetClaim` and
    `finish_written` into the walk (+4), the walk queue's
    `Vec::with_capacity` and its inlined std lines, whose source did not
    change (+8 Local), and +2 in `arena_registry.rs` with no line
    information. Under Shared the `/10` trace adds +9 indexing the
    dependents' `SlotVec` and −14 of moves across `cycle.rs`,
    `directory.rs` and lines with no line information, with the
    workload's total exact (+63).

  The workloads compose from these: `early_cutoff_local` +26 is one
  `set` and two computes, and `simple_count/*` and
  `children_rerun_1k_*` (+22 and +22 Local, +18 and +20 Shared) one
  `set` and one compute. A first compute,
  which no walk flips, nets 0 Local and −3 Shared (`fanout_*_1000`'s
  1,001 computes after its deletes are taken out).
- **`get_slow` node pass-through (every non-Clean `get`; `2e10c62`):**
  −8 Local and −7 Shared per top-level non-Clean `get`, as recorded for
  T8's follow-up; a nested one (`fanout_*_1000`, 1,000 per iteration)
  saves about 5 Local and 9 Shared.
- **Delete path (`delete_single_*`, and the teardown of `fanout_*` and
  `delete_cascade_*`):** −23 Local and −19 Shared per deleted node. The
  plan's `WATCHED`/`QUEUED` clear and conditional dequeue
  (`unwatch_deleted`) costs +2 Local and +1 Shared, but T8 collected a
  deleted node's observers with `extract_if` over the observer
  `HashMap` into a `Vec` on every delete (46 Local, the empty map scan
  and the `collect`), where T9's `Observers::take_slot` and its `Vec`
  take 25; `unhook_retired` and `delete_strict` move by +4 and −8 Local
  (Intel PT per function, `delete_single_*`). Six
  deletes per `fanout_local_4` iteration give its −138, and 1,001 per
  `delete_cascade_local` iteration its −23,029.
- **Changed-dep publish (`fanout_*`, `delete_cascade_*`):** the
  `publish_deps` length test, +14 Local and +4 Shared per first
  compute, as its section above records (`fanout_local_1000` +15,009,
  `delete_cascade_local` +14,004 here).
- **Layout (`children_churn_*` −3 Local, −2 Shared after the
  `publish_deps` merge):** the T9 runs agree to 0.03, but the
  `publish_deps` section measured this workload unchanged across the
  same commit at another path length, so it is placement, not executed
  work.
- **Run spread (`delete_owned_tree_*`):** −3,430 Local and +5,953
  Shared over three merges whose steps change sign (Local −6,748,
  +7,698, −4,380), inside the recorded spread between runs of one
  binary.
- **Clean get, table and value-slot reads:** unchanged. Observers and
  `stabilize` run only in workloads that use them, which none of these
  do.

## Dependent-list index

- Engine: branch `w/T-core-dependents-linear` against its base
  `spreadsheet-t10` at `be61773`, built in a detached worktree; one run
  of `icount.sh` at the default N for each column. No workloads added.
- Change: a block-mode `DependentList` whose remove scans past 64 ids
  from the back builds an id-to-position index, and later removes take
  a hash lookup and leave a `NONE` hole (compacting once holes outnumber
  ids), so rewiring a dependent off a shared input no longer scans and
  shifts the whole list. The block header grows from 8 to 24 bytes on
  64-bit for the hole count and the index pointer. Inline lists (up to 4
  ids on 64-bit) are unchanged.
- Attribution: Intel PT (`intel_pt/tsc=0,mtc=0,noretcomp=0/u`) on
  line-table builds of both drivers, per symbol, N minus 0:
  `fanout_*_5` at N = 20 with 20 warm-up and `fanout_local_1000` at N = 1
  with none, where the traced totals match `perf stat` to within 100 per
  iteration. The cascade traces lose data at every N, so the allocator's
  share comes from a third driver: the base with only its block header
  padded to the new 24 bytes ("padded base" below), counted with
  `perf stat`.

| workload | N | base | index | delta | delta % | sources |
|---|--:|--:|--:|--:|--:|---|
| `chain_local/propagate/4` | 1,000,000 | 4,639.00 | 4,639.00 | +0.00 | +0.00 | none |
| `chain_local/propagate/10` | 1,000,000 | 10,027.00 | 10,027.00 | +0.00 | +0.00 | none |
| `chain_local/propagate/100` | 100,000 | 91,606.12 | 91,606.03 | -0.09 | -0.00 | none (run spread) |
| `chain_shared/propagate/4` | 1,000,000 | 9,356.41 | 9,356.45 | +0.04 | +0.00 | none (run spread) |
| `chain_shared/propagate/10` | 1,000,000 | 20,354.26 | 20,354.30 | +0.04 | +0.00 | none (run spread) |
| `chain_shared/propagate/100` | 100,000 | 186,339.97 | 186,340.25 | +0.28 | +0.00 | none (run spread) |
| `diamond_local` | 1,000,000 | 3,927.00 | 3,927.00 | +0.00 | +0.00 | none |
| `diamond_shared` | 1,000,000 | 7,796.09 | 7,796.09 | +0.00 | +0.00 | none |
| `early_cutoff_local` | 1,000,000 | 2,299.00 | 2,299.00 | +0.00 | +0.00 | none |
| `early_cutoff_shared` | 1,000,000 | 3,582.89 | 3,582.89 | +0.00 | +0.00 | none |
| `collection_insert_then_read/local_incremental/1000` | 100,000 | 4,739.08 | 4,738.65 | -0.43 | -0.01 | none (run spread) |
| `collection_insert_then_read/shared_incremental/1000` | 100,000 | 7,365.62 | 7,365.69 | +0.07 | +0.00 | none (run spread) |
| `collection_insert_then_read/batch/1000` | 10,000 | 92,282.72 | 92,283.51 | +0.79 | +0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/10000` | 100,000 | 4,730.71 | 4,731.70 | +0.99 | +0.02 | none (run spread) |
| `collection_insert_then_read/shared_incremental/10000` | 100,000 | 7,339.54 | 7,339.99 | +0.45 | +0.01 | none (run spread) |
| `collection_insert_then_read/batch/10000` | 1,000 | 158,886.10 | 158,886.20 | +0.10 | +0.00 | none (run spread) |
| `collection_insert_then_read/local_incremental/100000` | 100,000 | 4,739.46 | 4,739.92 | +0.46 | +0.01 | none (run spread) |
| `collection_insert_then_read/shared_incremental/100000` | 100,000 | 7,346.23 | 7,344.35 | -1.88 | -0.03 | none (run spread) |
| `collection_insert_then_read/batch/100000` | 1,000 | 1,490,571.62 | 1,490,567.99 | -3.63 | -0.00 | none (run spread) |
| `simple_count/local/1000` | 100,000 | 2,129.84 | 2,129.44 | -0.40 | -0.02 | none (run spread) |
| `simple_count/shared/1000` | 100,000 | 4,076.30 | 4,076.68 | +0.38 | +0.01 | none (run spread) |
| `simple_count/local/10000` | 100,000 | 2,126.80 | 2,126.20 | -0.60 | -0.03 | none (run spread) |
| `simple_count/shared/10000` | 100,000 | 4,056.15 | 4,056.54 | +0.39 | +0.01 | none (run spread) |
| `get_clean_local` | 10,000,000 | 102.00 | 102.00 | +0.00 | +0.00 | none |
| `get_clean_shared` | 10,000,000 | 196.00 | 196.00 | +0.00 | +0.00 | none |
| `get_contended_shared` | 2,000 | 7,791,704.26 | 7,792,709.64 | +1,005.38 | +0.01 | run spread |
| `fanout_local_4` | 100,000 | 17,111.01 | 17,111.02 | +0.01 | +0.00 | none (run spread) |
| `fanout_local_5` | 100,000 | 21,674.00 | 21,719.02 | +45.02 | +0.21 | per-remove and spill (below) |
| `fanout_local_1000` | 1,000 | 3,442,229.18 | 3,449,740.20 | +7,511.02 | +0.22 | per-remove and per-push; block size (below) |
| `fanout_shared_4` | 100,000 | 23,467.47 | 23,467.32 | -0.15 | -0.00 | none (run spread) |
| `fanout_shared_5` | 100,000 | 28,806.12 | 28,809.43 | +3.31 | +0.01 | per-remove and spill; block size (below) |
| `fanout_shared_1000` | 1,000 | 4,735,368.88 | 4,751,561.99 | +16,193.11 | +0.34 | per-remove and per-push; block size (below) |
| `delete_single_local` | 200,000 | 706.00 | 706.00 | +0.00 | +0.00 | none |
| `delete_single_shared` | 200,000 | 1,530.24 | 1,530.24 | +0.00 | +0.00 | none |
| `delete_cascade_local` | 1,000 | 3,229,254.71 | 3,228,572.90 | -681.81 | -0.02 | block size (below) |
| `delete_cascade_shared` | 1,000 | 4,523,990.43 | 4,533,605.93 | +9,615.50 | +0.21 | block size (below) |
| `delete_owned_tree_local` | 10,000 | 601,878.67 | 601,269.84 | -608.83 | -0.10 | run spread |
| `delete_owned_tree_shared` | 10,000 | 984,913.95 | 977,903.88 | -7,010.07 | -0.71 | run spread |
| `table_hit_local` | 10,000,000 | 155.14 | 155.14 | +0.00 | +0.00 | none |
| `table_hit_shared` | 10,000,000 | 218.14 | 218.14 | +0.00 | +0.00 | none |
| `table_hit_sip_local` | 10,000,000 | 301.06 | 301.52 | +0.46 | +0.15 | none (run spread) |
| `table_miss_local` | 100,000 | 871.18 | 871.18 | +0.00 | +0.00 | none |
| `table_miss_shared` | 100,000 | 2,170.29 | 2,170.27 | -0.02 | -0.00 | none (run spread) |
| `value_slot_read_local` | 10,000,000 | 5.00 | 5.00 | +0.00 | +0.00 | none |
| `value_slot_read_shared` | 10,000,000 | 87.00 | 87.00 | +0.00 | +0.00 | none |
| `children_rerun_1k_local` | 10,000 | 217,992.01 | 217,992.97 | +0.96 | +0.00 | none (run spread) |
| `children_rerun_1k_shared` | 10,000 | 224,936.74 | 224,936.34 | -0.40 | -0.00 | none (run spread) |
| `children_churn_local` | 100,000 | 16,525.48 | 16,513.72 | -11.76 | -0.07 | outside the list (below) |
| `children_churn_shared` | 100,000 | 19,413.17 | 19,401.97 | -11.20 | -0.06 | outside the list (below) |

- **Per-remove and spill (`fanout_*_5`):** the teardown removes 5 leaves
  from the input's block list, each +9 in `remove_from_block` (the index
  test, and the scan-length test that decides whether to build one), and
  the spill writes the header's two new fields (+2): +47 per iteration
  in both traces. Shared nets +3 because glibc runs fewer instructions
  for the larger block: the padded base reads −36 against the base.
- **Per-remove and per-push (`fanout_*_1000`):** +9,000 in
  `remove_from_block` (1,000 removes at +9) and +2,985 in `push_to_block`
  (995 pushes at +3, the index test), plus about +1,100 in glibc's
  malloc: +13,056 traced against +12,964 by `perf stat` at N = 1. The
  teardown removes back to front, so no remove scans past 64 and no
  index is built. At the recorded N the padded base reads −5,206 Local
  and +407 Shared against the base, which with the engine's +12,000
  accounts for both rises; the same Shared binaries read up to 3,000
  apart between passes.
- **Block size (`delete_cascade_*`):** the padded base alone reads
  +10,678 Shared and −501 Local against the base, matching the index
  build's +10,759 and −414 in the same `perf stat` pass, so both are
  glibc's response to the larger header; the engine part traces at
  +105 (10 removes, 5 pushes).
- **Outside the list (`children_churn_*`, −12 and −11):** the padded
  base reads within 2 of the base, and the N = 1 trace has no
  `DependentList` symbol; the fall is in `HashMap::insert` and
  `Vec::retain`.

Gate (board item: the small case must not grow): every inline-list
workload (`chain_*`, `diamond_*`, `early_cutoff_*`, `fanout_*_4`,
`delete_single_*`, the Clean get, table hit and value slot) is unchanged.
The engine's only rise is +9 per remove and +3 per push on a list that
has spilled to a block, plus 16 bytes per block.

## Walk containment

- Engine: branch `w/T8-walk-contain-failed-dep` against its base
  `spreadsheet-t10` at `978b407`, built in a detached worktree; one run
  of `icount.sh` at the default N for each column. No workloads added.
- Change: a dep that the dirty walk computes and that panics or closes
  a cycle counts as changed for its recorder, and its payload stays with
  the Failed node for the first reader as a ticket. The walk passes
  `walked` to `compute_one`, which sets a `WALKED` bit in `ComputeCtl`
  (flags widen from `u8` to `u16`). The dirty walk's Failed arm, now apart
  from the Clean arm, drops the node's ticket; a delete buries the
  tickets of the node it deletes and its bag drops them, each behind one
  relaxed flag.
- Attribution: Intel PT on line-table builds, N = 10 with 100 warm-up,
  traced totals within 0.3 of `perf stat` per iteration, on a build that
  still had one `Clean | Failed` arm; the arm split and the delete flag
  were attributed by their diff, not traced.

| workload | base | tip | delta |
|---|---|---|---|
| `get_clean_local` / `get_clean_shared` | 102 / 196 | 102 / 196 | 0 |
| `table_hit_local`, `value_slot_read_local` | 155.14, 5 | 155.14, 5 | 0 |
| `diamond_local` | 3,921 | 3,918 | −3 |
| `diamond_shared` | 7,811 | 7,849 | +38 |
| `chain_local/propagate/100` | 91,406 | 91,306 | −100 |
| `chain_shared/propagate/10` | 20,404 | 20,525 | +121 |
| `chain_shared/propagate/100` | 186,837 | 188,038 | +1,201 |
| `delete_single_local` / `_shared` | 706 / 1,530 | 712 / 1,534 | +6 / +4 |

- **`walked` and `WALKED`:** `ComputeCtl::new` ORs the bit in and
  `has`/`raise` test a `u16`: up to +10 per iteration on the Shared
  chain, about 1 per walk visit.
- **Codegen in `clean_walk`:** with `compute_one` taking another
  parameter, the frame push and pop, the `catch_unwind` wrapper and the
  node lookup shift: +33 (`diamond_shared`) and +110
  (`chain_shared/propagate/10`) in the trace, which with the line above
  accounts for the Shared rise. Local nets about +2 in the trace and
  falls below base once the arm is split.
- **The Failed arm:** with one `Clean | Failed` arm, a `cur == Failed`
  test after every flip cost +9, +28 and +80 in `mark_dependents_dirty`;
  the split arm takes it off the Clean path, which is where the Local
  fall and most of the Shared drop between the two tip builds come from.
- **Deletes:** two relaxed loads of the ticket flag and their branches,
  one where the node is unhooked and one at the bag's drain: +6 Local and
  +4 Shared per delete, and the matching share of `fanout_*` and
  `delete_*` (+0.3 to +0.6 percent).
- **No cost:** `NOTES_BIT` in the compute pop's mask (one mask, nets 0
  to −20 in the trace) and the `Failed if walked` arm, which the clean
  path never reaches.

Gate: the Clean get, table hit and value slot are unchanged; the dirty
walk falls 0.1 percent Local and rises 0.5 to 0.6 percent Shared.

## Episode containment

- Engine: branch `w/T8-walk-contain-episode-panics` against its base
  `spreadsheet-t10` at `a06016d`, each built in its own worktree; one run
  of `icount.sh` at the default N for each column. No workloads added.
- Change: an episode rooted at a walk's entry runs under a catch at
  `run_episode`, and its abort decides containment as `compute_panicked`
  does; every contained unwind's noted landings become its ticket in one
  function. All of it is on the episode and panic paths, and the walk's
  path gains only an argument to `group_verify` and a `PopEnd` arm.
- Attribution: Intel PT on line-table builds of `chain_local/propagate/10`
  and `chain_shared/propagate/10`, N = 10 with 100 warm-up, traced totals
  equal to `perf stat`.

| workload | base | tip | delta |
|---|---|---|---|
| `get_clean_local` / `get_clean_shared` | 102 / 196 | 102 / 196 | 0 |
| `table_hit_local`, `value_slot_read_local` | 155.14, 5 | 155.14, 5 | 0 |
| `diamond_local` / `diamond_shared` | 3,918 / 7,855 | 3,915 / 7,849 | −3 / −6 |
| `chain_local/propagate/100` | 91,306 | 91,206 | −100 |
| `chain_shared/propagate/100` | 188,238 | 188,038 | −200 |
| `fanout_shared_1000` | 4,776,649 | 4,773,642 | −3,007 |
| `delete_single_local` / `_shared` | 712 / 1,534 | 712 / 1,534 | 0 |

- **Codegen in the walk's inlined `compute_attempts`:** no line of the
  change runs on the traced path. Local drops one `directory.rs:76`
  bounds compare per compute (−10 per iteration); Shared shifts ±1 across
  a dozen lines of the same body and nets −2 per compute.

Gate: the Clean get, table hit and value slot are unchanged; the dirty
walk falls about 0.1 percent under both strategies.

## Ticket drops

- Engine: `w/T8-contain-ticket-drop-panic` (`2b6fc1f`) against its base
  `spreadsheet-t10` at `03f279e`, and `675ac65` (the swept-ticket fix)
  against its parent `7dbde7c`, each in a detached worktree; one run of
  `icount.sh` at the default N per column, `getrandom` fixed by
  `fixed_getrandom.so`. No workloads added.
- Change: a set walk's Failed→Dirty flip parks the tickets it released
  with the buried ones, and one flag check after the walk drops them; a
  compute's failed landing hands its released tickets to its bag; a
  delete's payload panic drops the bag's items on a cold path; a sweep
  hands buried tickets to its compute's bag. Attributed by the diff, not
  traced.

| workload | base | tip | delta |
|---|---|---|---|
| `get_clean_local` / `get_clean_shared` | 102 / 196 | 102 / 196 | 0 |
| `table_hit_local`, `value_slot_read_local` | 155.14, 5 | 155.14, 5 | 0 |
| `diamond_local` / `_shared` | 3,915 / 7,849 | 3,919 / 7,852 | +4 / +3 |
| `chain_local/propagate/100` | 91,206 | 91,210 | +4 |
| `chain_shared/propagate/100` | 188,038 | 188,040 | +2 |
| `delete_single_local` / `_shared` | 712 / 1,534 | 712 / 1,534 | 0 |
| `delete_owned_tree_local` / `_shared` | 600,829 / 980,844 | 601,629 / 981,443 | +800 / +599 |
| `children_churn_local` / `_shared` (`675ac65`) | 16,523 / 19,429 | 16,526 / 19,433 | +3 / +4 |

- **After the walk:** `drop_released_tickets` loads the ticket flag and
  branches once per `set`: +4 Local and +3 Shared on every workload that
  sets, independent of walk length (`chain_*/propagate/4` to `/100` all
  move by the same amount). `delete_owned_tree_*` moves only through
  its setup's 200 collection sets per iteration.
- **The walk loop:** unchanged in source and in count. Each version that
  edited the Failed arm (a returned `Option`, the queue by `&mut`, a
  cold continuation holding the rest of the walk) cost +1 per visited
  node on Shared or +4 to +8 per set on Local, though the arm never runs
  here.
- **Deletes:** the buried-ticket check before the bag's items is the
  same load and branch as before; the panic path lives in the cold
  `drain_after` and `DeleteBag::unwind`.
- **Sweeps (`675ac65`):** `defer_buried_tickets` loads the ticket flag
  once per `sweep_units` call: +3 Local and +4 Shared on
  `children_churn_*`, the only workload that sweeps children.

Gate: the Clean get, table hit, value slot and single delete are
unchanged; a set costs 3 to 4 more instructions.

## Arena lock spills

- Engine: the four `spreadsheet-t10` merges from `c0fddc04` to
  `0433749e`, each built in one detached worktree; `icount.sh` at the
  default N, `getrandom` fixed. Intel PT line diffs through `pt-lines.sh`
  at N = 10. No workloads added.

| merge | `delete_single_shared` | `fanout_shared_4` |
|---|---|---|
| `c0fddc04`, `1d97b37f`, `f2e77b8b` | 1,530.24 | 23,577.1 |
| `0433749e` (`w/T8-remove-path-take`) | 1,534.24 | 23,591.3 |
| both with `codegen-units = 1` | 1,554.33 | 23,515.3 |

- **Where:** the whole rise is in `GenericArena::reserve_with` and
  `ErasedArena::release` under `Shared`, +2 instructions per call, at the
  arena lock's write guard: the lock pointer and poison flag are stored
  to the stack and `RwLockWriteGuard::drop` is called through a pointer
  to the guard, where before they went in registers. One reserve and one
  release per single delete, six releases and one reserve per
  `fanout_shared_4` iteration.
- **Why:** `0433749e` touches only replay and provenance code, and with
  one codegen unit both trees count the same on every measured workload,
  so the source of the arena is not the cause. `RwLockWriteGuard::drop`
  carries no `#[inline]` in std (`MutexGuard::drop` does), so it is one
  out-of-line instance shared by every arena type; LLVM passes its guard
  in registers only when every caller lands in that instance's codegen
  unit, and any change elsewhere in the crate can move a caller out.
  `w/T8-unbuilt-value-drop-rest` alone moved the same two calls the same
  way against `1bbf96ed`.

Gate: the Clean get, table hit and value slot are unchanged. The two
calls cost 0 or 2 instructions by partitioning, like every other
`RwLock` write in the engine; a lock whose unlock inlines would remove
the dependence for that lock only.

## Foreign reads

- Engine: `w/T8-foreign-read-stack-overflow` (`16a6164f`) against its
  base `spreadsheet-t10` at `98d7c337`, the base in a detached worktree;
  one run of `icount.sh` at the default N per column, and the deletes
  again with `getrandom` fixed by `fixed_getrandom.so`. Intel PT line
  diffs through `pt-lines.sh` at N = 10. No workloads added.
- Change: a non-Clean read that another runtime's frame makes takes the
  cold path, which keeps its outcome for that frame's retry; a Clean one
  keeps nothing; an uncopyable outcome holds the reader undoomable within
  its own runtime's budget (`FOREIGN_HOLD`).

| workload | base | tip | delta |
|---|---|---|---|
| `get_clean_local` / `get_clean_shared` | 102 / 196 | 102 / 196 | 0 |
| `table_hit_local` / `_shared` | 155.14 / 218.14 | 155.14 / 218.14 | 0 |
| `value_slot_read_local` / `_shared` | 5 / 87 | 5 / 87 | 0 |
| `diamond_local` / `_shared` | 3,919 / 7,850.9 | 3,919 / 7,850.0 | 0 / -1 |
| `chain_local/propagate/100` | 91,210.0 | 91,210.0 | 0 |
| `fanout_local_4` / `fanout_shared_4` | 17,195.0 / 23,591.1 | 17,195.0 / 23,591.1 | 0 |
| `delete_owned_tree_local` / `_shared` (fixed seed) | 601,628 / 981,243 | 601,638 / 981,053 | +10 / -190 |
| `children_churn_local` / `_shared` | 16,528.7 / 19,434.1 | 16,498.7 / 19,402.4 | -30 / -32 |

- **The foreign arm of `record_read`:** returning the other runtime's
  frame as a second `Some` exit cost +3 per framed non-Clean read under
  Local (+7 on `fanout_local_4`, +9 on `fanout_shared_4`): LLVM merged
  the RefCell release of both exits, which spilled two registers in
  `get_slow`. Building that frame out of line, or returning the whole
  `Option` from a cold call, cost +4 to +5. The arm now returns `None`
  like the base and its cold call sets `value_code::FOREIGN_BIT`, which
  joins the mask `read_slow` already tests, so the own-runtime path is
  the base's instruction for instruction.
- **The Clean arm of `record_dep`:** an empty arm spilled two more
  registers in the inlined Clean `get`, 2 to 10 instructions per read; a
  cold out-of-line call in the arm keeps the base's code.
- **Deletes:** `icount.sh` does not fix `getrandom`, and its
  `delete_owned_tree_*` columns moved +1,318 / +608 between runs of the
  same trees. With the seed fixed the Local rise is +10, all in drop glue
  this branch does not touch (`drop_in_place`, `DeleteBag`), and Shared
  falls by 190.
- **`children_churn_*`:** `HashMap::insert` is now inlined into
  `children::IndexState::insert` (-82 for the call, +52 inlined).
- **Partitioning:** with `codegen-units = 1` and the seed fixed, both
  trees count the same on `delete_owned_tree_local` (610,684.3),
  `children_churn_local` (16,410.4) and `diamond_local` (3,986.0), so
  the delete rise and the churn fall are where codegen units split.

Gate: every hard gate and every single-thread compute case counts the
same as the base.

## Foreign budget fixes

- Engine: `w/T8-wave8-foreign-budget-fixes` (`759cf000`) against its
  base `spreadsheet-t10` at `ba660f66`, the base in a detached worktree;
  one run of `icount.sh` at the default N per column, the fanouts again
  twice. Intel PT line diffs through `pt-lines.sh` at N = 10 and 1. No
  workloads added.
- Change: a runtime's unbudgeted listing is decided under the list's
  lock, and a frame's second foreign hold keeps the address nearer the
  stack's base; both are cold paths.

| workload | base | tip | delta |
|---|---|---|---|
| `get_clean_local` / `get_clean_shared` | 102 / 196 | 102 / 196 | 0 |
| `table_hit_local` / `_shared` | 155.14 / 218.14 | 155.14 / 218.14 | 0 |
| `value_slot_read_local` / `_shared` | 5 / 87 | 5 / 87 | 0 |
| `diamond_local` / `_shared` | 3,919 / 7,850.9 | 3,919 / 7,850.9 | 0 |
| `chain_local/propagate/100` | 91,210.0 | 91,210.0 | 0 |
| `fanout_local_4` / `fanout_shared_4` | 17,195.0 / 23,590.9 | 17,195.0 / 23,590.9 | 0 |
| `fanout_local_5` / `fanout_shared_5` | 21,817.0 / 28,957.1 | 21,839.0 / 28,979.1 | +22 / +22 |
| `fanout_local_1000` / `fanout_shared_1000` | 3,463,747 / 4,775,102 | 3,468,745 / 4,780,100 | +4,998 / +4,998 |

- **Fanouts:** the whole rise is `DependentList::remove_from_block`, +5
  per removed dependent at its entry, exit and empty-block test: it now
  saves four registers and aligns the stack where the base saved three,
  in a function neither commit touches. With `codegen-units = 1`,
  `getrandom` fixed and ASLR off, the PT trace of `fanout_local_1000`
  counts 3,515,762 per iteration on both trees, line for line.
  `icount.sh` on those builds moves by -40, -1,092 and +614 at N = 1, 10
  and the default, in both directions with N, so what is left there
  depends on the run's state, not on the change.
- **`table_hit_*` in cycles:** three runs per tree overlap (Local 100.5
  to 107.9 base, 106.6 to 117.8 tip; Shared 185.8 to 200.4 base, 165.7
  to 228.0 tip), with the instruction counts equal.

Gate: every hard gate and every single-thread compute case outside the
fanouts counts the same as the base; the fanout rise is partitioning.

## Unsettled stand-in landings

- Engine: `w/T8-finding3-unsettled` (`88a53c99`) against its base
  `spreadsheet-t10` at `5bd221cc`, the base in a detached worktree; one
  run of `icount.sh` at the default N per column. No workloads added.
- Change: a stand-in's `land_alone` returns the state it landed, and a
  virtual frame's pop that lands its node not Clean reports it to the
  walk as unsettled; both are replay paths no workload reaches.

| workload | base | tip | delta |
|---|---|---|---|
| `get_clean_local` / `get_clean_shared` | 102 / 196 | 102 / 196 | 0 |
| `table_hit_local` / `_shared` | 155.14 / 218.14 | 155.14 / 218.14 | 0 |
| `value_slot_read_local` / `_shared` | 5 / 87 | 5 / 87 | 0 |
| `diamond_local` / `_shared` | 3,919 / 7,850.99 | 3,919 / 7,850.94 | 0 / -0.05 |
| `chain_local/propagate/100` | 91,210.02 | 91,210.02 | 0 |
| `fanout_local_1000` / `fanout_shared_1000` | 3,463,128.8 / 4,773,608.4 | 3,463,128.9 / 4,773,609.6 | +0.1 / +1.2 |

Gate: every hard gate and every single-thread compute case counts the
same as the base. The rest move by run-to-run noise in both directions
(`delete_owned_tree_local` -1,490, `get_contended_shared` +249, the
collection workloads under 2).

## Member try_get of its own member

- Engine: `w/T8-lead-a18d-adds-failed` (`7c14ca1e`) against its base
  `spreadsheet-t10` at `ab33f929`, the base in a detached worktree; one
  run of `icount.sh` at the default N per column, the fanouts again
  twice. Intel PT line diff of `fanout_local_1000` through `pt-lines.sh`
  at N = 1. No workloads added.
- Change: a delivered cycle unwind carrying a closure's own panic stays
  the cycle's for a member run's `try_get` of a member of its episode, a
  test on the cold delivery path only.

| workload | base | tip | delta |
|---|---|---|---|
| `get_clean_local` / `get_clean_shared` | 102 / 198 | 102 / 198 | 0 |
| `table_hit_local` / `_shared` | 155.14 / 219.14 | 155.14 / 219.14 | 0 |
| `value_slot_read_local` / `_shared` | 5 / 90 | 5 / 90 | 0 |
| `fanout_local_5` | 21,817.0 | 21,817.0 | 0 |
| `fanout_local_1000` / `fanout_shared_1000` | 3,463,128 / 4,777,611 | 3,463,742 / 4,779,099 | +614 / +1,488 |

- **`fanout_*_1000`:** each build repeats its count run to run, but the
  PT trace with `getrandom` fixed and ASLR off counts 3,505,756 per
  iteration on both trees, line for line, and `fanout_*_5` is equal. The
  two counts are the values `fanout_local_1000` also took on the trees of
  **Foreign budget fixes**, so the difference is the run's state, not
  the change.

Gate: every hard gate and every single-thread compute case counts the
same as the base.

## Replayed entries pop only what their stop pushed

- Engine: `w/T8-seed-922f-adds-failed` against its base `spreadsheet-t10`
  at `c355284e`, the base in a detached worktree; one run of `icount.sh`
  at the default N per column. No workloads added.
- Change: a walk's stashed delivery carries the entry its stop pushed
  back, and a context's run of a doomed entry no longer pops a repeat
  of its node; replay paths only.

| workload | base | tip | delta |
|---|---|---|---|
| `get_clean_local` / `get_clean_shared` | 102 / 196 | 102 / 196 | 0 |
| `table_hit_local` / `_shared` | 155.14 / 218.14 | 155.14 / 218.14 | 0 |
| `value_slot_read_local` / `_shared` | 5 / 87 | 5 / 87 | 0 |
| `diamond_local` | 3,919 | 3,919 | 0 |
| `chain_local/propagate/100` | 91,210.02 | 91,210.01 | -0.01 |
| `fanout_local_4` / `fanout_local_5` | 17,195.01 / 21,817 | 17,195.01 / 21,817 | 0 |

Gate: every hard gate and every single-thread compute case counts the
same as the base. The rest move by run-to-run noise in both directions
(`delete_owned_tree_local` -1,938, `delete_owned_tree_shared` +5,132,
`get_contended_shared` +249, the others under 15).

## Kept unwinds out of replay loops

- Engine: `w/T8-seed-3da5-extra-failed` against its base
  `spreadsheet-t10` at `1b77f852`, the base in a detached worktree; one
  run of `icount.sh` at the default N per column. No workloads added.
- Change: a context closing on an unwind, and an R5 member run's unwind,
  keep the read they unwind into for its frame's retry; replay paths
  only.

| workload | base | tip | delta |
|---|---|---|---|
| `get_clean_local` / `get_clean_shared` | 102 / 196 | 102 / 196 | 0 |
| `table_hit_local` / `_shared` | 155.14 / 218.14 | 155.14 / 218.14 | 0 |
| `value_slot_read_local` / `_shared` | 5 / 87 | 5 / 87 | 0 |
| `diamond_local` | 3,919 | 3,919 | 0 |
| `chain_local/propagate/100` | 91,210.01 | 91,210.01 | 0 |
| `fanout_local_4` | 17,195 | 17,195 | 0 |

Gate: every hard gate and every single-thread compute case counts the
same as the base. The rest move by run-to-run noise in both directions
(`delete_owned_tree_*` +244 / -349, `fanout_shared_1000` +9, the others
under 9).
