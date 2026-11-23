import json
import os
import subprocess
import sys

import pytest

PROGRAM = r"""
import gc
import json
import sys
from functools import partial

import incr

n, mode = int(sys.argv[1]), sys.argv[2]


def s0(i, r):
    return r.get(i) + 1000


def s1(a, r):
    return r.get(a) + 1


def s2(a, b, r):
    return r.get(a) + r.get(b)


rt = incr.Runtime()
roots = []
for k in range(n):
    i = rt.create_input(k)
    a = rt.create_query(partial(s0, i))
    b = rt.create_query(partial(s1, a))
    c = rt.create_query(partial(s2, a, b))
    d = rt.create_query(partial(s2, b, c))
    roots.append(rt.create_query(partial(s2, c, d)))

read = rt.get_traced if mode == "traced" else rt.get
for root in roots:
    read(root)
gc.collect()

with open("/proc/self/status") as status:
    rss_kb = next(int(line.split()[1]) for line in status if line.startswith("RssAnon:"))
print(json.dumps({"rss_kb": rss_kb, "kept_traces": rt.memory_stats().get("kept_traces")}))
"""

CLIENT_ROOTS = 270_967
JUST_PAST_RESIZE = 7 * 2**18 // 8 + 1


def run(n, mode):
    out = subprocess.run(
        [sys.executable, "-c", PROGRAM, str(n), mode],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout)


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or not os.environ.get("INCR_RSS_TESTS"),
    reason="RssAnon is Linux-only; set INCR_RSS_TESTS=1 for these multi-GB runs",
)
@pytest.mark.parametrize("n", [CLIENT_ROOTS, JUST_PAST_RESIZE])
def test_kept_five_step_traces_cost_at_most_300_bytes_per_root(n):
    plain = run(n, "plain")
    traced = run(n, "traced")
    per_root = (traced["rss_kb"] - plain["rss_kb"]) * 1024 / n
    print(f"n={n} rss_per_root={per_root:.1f} B plain={plain} traced={traced}")
    assert per_root <= 300
    assert traced["kept_traces"]["count"] == n
    assert traced["kept_traces"]["bytes"] / n <= 300
