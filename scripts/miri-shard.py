#!/usr/bin/env python3
"""Pick one CI shard of incr-core's lib tests for Miri.

Usage: miri-shard.py <times.tsv> <shards> <index> <tests> <ignored>

<tests> is the `--list` output of every lib test (one name per line) and
<ignored> the `--list --ignored` output. Tests are packed longest first onto
the least loaded shard (then the one with fewest tests) by their Miri
seconds in <times.tsv>; ignored tests count as 0. Tests the times file does
not know go on the least loaded shard at UNKNOWN_SECS each, so a new or
renamed test still runs. Every job packs
the same list the same way, so the shards cover each test exactly once.
Prints the chosen shard's tests, and each shard's size and estimate to
stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path

UNKNOWN_SECS = 30


def read_names(path: str) -> list[str]:
    return [
        line.strip() for line in Path(path).read_text().splitlines() if line.strip()
    ]


def read_times(path: str) -> dict[str, int]:
    times: dict[str, int] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        secs, name = line.split("\t")[:2]
        times[name] = max(times.get(name, 0), int(secs))
    return times


def pack(
    tests: list[str], ignored: set[str], times: dict[str, int], shards: int
) -> tuple[list[list[str]], list[int]]:
    def cost(test: str) -> int:
        return 0 if test in ignored else times[test]

    known = sorted((t for t in tests if t in times), key=lambda t: (-cost(t), t))
    unknown = sorted(t for t in tests if t not in times)
    picked: list[list[str]] = [[] for _ in range(shards)]
    load = [0] * shards
    for test, secs in [(t, cost(t)) for t in known] + [
        (t, 0 if t in ignored else UNKNOWN_SECS) for t in unknown
    ]:
        k = min(range(shards), key=lambda i: (load[i], len(picked[i]), i))
        picked[k].append(test)
        load[k] += secs
    return picked, load


def main() -> int:
    if len(sys.argv) != 6:
        print(__doc__, file=sys.stderr)
        return 2
    times = read_times(sys.argv[1])
    shards, index = int(sys.argv[2]), int(sys.argv[3])
    tests, ignored = read_names(sys.argv[4]), set(read_names(sys.argv[5]))
    if not tests or len(set(tests)) != len(tests):
        print(
            f"miri-shard: {len(tests)} tests listed, or a name listed twice",
            file=sys.stderr,
        )
        return 1
    picked, load = pack(tests, ignored, times, shards)
    if not all(picked):
        print(f"miri-shard: {len(tests)} tests leave a shard empty", file=sys.stderr)
        return 1
    if sorted(t for shard in picked for t in shard) != sorted(tests):
        print("miri-shard: shards do not cover each test once", file=sys.stderr)
        return 1
    unknown = sum(t not in times and t not in ignored for t in tests)
    for k, (shard, secs) in enumerate(zip(picked, load)):
        mark = " <-" if k == index else ""
        print(
            f"shard {k}: {len(shard)} tests, about {secs // 60} min{mark}",
            file=sys.stderr,
        )
    print(
        f"{len(tests)} tests, {unknown} without a time at {UNKNOWN_SECS} s",
        file=sys.stderr,
    )
    print("\n".join(picked[index]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
