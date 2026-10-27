#!/usr/bin/env bash
set -uo pipefail
(($# == 1)) || { echo "usage: rerun.sh <release-sha>" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
beta5=c61be873c4d6681cd0b59333441267749f4776cc
rounds=${FIGURE_ROUNDS:-3}
for t in b5:$beta5 rel:$1; do
  n=${t%%:*}
  s=${t#*:}
  git fetch -q --depth=1 origin "$s"
  git worktree add -q --detach "../fig-$n" "$s" || exit 1
  (cd "../fig-$n" && git log --oneline -1 && cp "$here"/probes/*.rs crates/incr-core/tests/ &&
    cargo test --release -p incr-core --tests --no-run 2>&1 | grep -aE '^error' -A6 | head -20)
done
for p in "$here"/probes/*.rs; do
  probe=$(basename "$p" .rs)
  for r in $(seq "$rounds"); do
    for n in b5 rel; do
      (cd "../fig-$n" && FIGURE_TREE=r$r-$n NC_TREE=r$r-$n cargo test -q --release -p incr-core --test "$probe" -- --nocapture 2>&1 | grep -a '^NC \|^FIG \|panicked\|FAILED')
    done
  done
done
echo "runner=${RUNNER_NAME:-unknown}"
