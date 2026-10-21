# memo record

Wall-clock times from `benches/memo.rs` (its header says what each group
measures), per entry unless marked. Restores read a `FileMemo`
generation on ext4 under `target/` with the page cache warm; the store
log is a `MemoryBackend` and the runtime is `Local`.

## T10.7

- Code: `ea8fb35` (spreadsheet-t10 at `bc10b87` plus the benches).
- Machine: Intel Core i7-9750H (6 cores, 12 threads), Linux
  7.0.0-38-generic, glibc 2.39, `rustc 1.96.1 (31fca3adb 2026-06-26)`,
  `cargo bench` profile.
- Command: `cargo bench -p incr-persist --bench memo` with criterion's
  `--output-format bencher`, once for every group (run 1) and once more
  filtered to `memo_sweep` (run 2).
- Other builds shared the box, holding the load average between 12 and
  20 on 12 threads through both runs. Restore never runs a closure, so
  its sweep column should be flat, yet it ranges from 2.6 to 9.7 µs; that
  spread is the noise floor here, and a quiet-box rerun should come
  before anyone quotes these tighter than the ranges below.

### 100k shapes

| | chain | wide |
|---|---|---|
| cold | 1.09 µs | 0.68 µs |
| restore | 4.59 µs | 5.90 µs |
| checkpoint (fsyncs included) | 12.4 µs | 10.1 µs |
| open, whole table | 12.7 ms | 12.8 ms |

The chain's restore reads only the top entry, which restores all 100k
under it on the restore walk's explicit stack and then verifies the top
without running anything. Before timing, each shape checks that a
restart restores every entry (`reused == 100000`, `cold == 0`).

### Closure-cost sweep, 1000 entries

| closure | spin 1 | cold 1 | restore 1 | spin 2 | cold 2 | restore 2 |
|---|---|---|---|---|---|---|
| 100 ns | 0.18 | 0.77 | 2.82 | 0.23 | 1.61 | 9.69 |
| 300 ns | 0.40 | 1.09 | 2.94 | 0.52 | 1.58 | 4.56 |
| 1 µs | 1.12 | 2.70 | 5.48 | 1.17 | 1.69 | 2.92 |
| 3 µs | 3.84 | 5.02 | 5.48 | 3.19 | 3.91 | 2.92 |
| 10 µs | 12.99 | 11.20 | 2.61 | 10.51 | 11.54 | 5.63 |
| 30 µs | 37.0 | 37.2 | 5.08 | 31.5 | 31.8 | 5.12 |
| 100 µs | 101.0 | 102.4 | 4.53 | 104.1 | 104.8 | 3.96 |
| 300 µs | 303 | 304 | 3.13 | 304 | 309 | 4.56 |
| 1 ms | 1003 | 1009 | 2.98 | 1001 | 1007 | 3.54 |

All values are µs per entry. `spin` times the closure's busy-wait alone,
so cold minus spin is the memo's own cold overhead, 0.5 to 0.7 µs in the
least noisy points.

### Break-even

In a 1000-entry table, restoring an entry beats running it once the
closure costs about 2 to 3.5 µs. Run 2's cold and restore cross at
2.1 µs of nominal closure cost (between its 1 µs and 3 µs points) and
run 1's at 3.4 µs (between 3 µs and 10 µs), while the lowest restore
(2.6 µs) minus the lowest cold overhead (0.5 µs) also gives 2.1 µs. The
100k tables restore more slowly per entry, which moves their break-even
to about 3.5 µs for the chain and 5.2 µs for the wide table. Past the
break-even a restore stays at 3 to 5 µs per entry whatever the closure
costs, so a 1 ms closure restores about 300 times faster than it runs.

### Where restore spends its time

In a `perf record -g` of `memo_wide_100k/restore` (whose samples also
cover every group's untimed setup, cold runs included), 13.8% sit in
`Generation::find`, the binary search over the generation's sorted
index that a restore runs twice per entry (`IndexGen::holds` and then
`IndexGen::read`, `memo_core.rs:101` and `:117`). At 100k items each
search is about 17 dependent cache misses, part of why the 100k wide
table restores more slowly than the 1000-entry sweep. `malloc`
and `free` take about 8% (per-entry key and frame buffers and the walk's
`HashMap`) and the kernel about 10% (per-entry frame reads).
