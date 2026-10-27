#!/usr/bin/env bash
set -uo pipefail
(($# == 1)) || { echo "usage: rerun.sh <release-sha>" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
rounds=${FIGURE_ROUNDS:-3}
declare -A sha=([rel]=$1 [b5]=c61be873c4d6681cd0b59333441267749f4776cc [main]=b9f0d0bf3b11adec48b81eee123f9921a8ef694c)
declare -A against built
probes=()
while IFS=$'\t' read -r probe vs _; do
  [[ -n $probe ]] || continue
  [[ -n ${sha[$vs]:-} || $vs == none ]] || { echo "probes.tsv: $probe has unknown against tree $vs" >&2; exit 2; }
  [[ -f $here/probes/$probe.rs ]] || { echo "probes.tsv: no probes/$probe.rs" >&2; exit 2; }
  probes+=("$probe")
  against[$probe]=$vs
done < <(tail -n +2 "$here/probes/probes.tsv")
logs=$(mktemp -d)

trees_of() {
  echo rel
  [[ ${against[$1]} == none ]] || echo "${against[$1]}"
}

in_tree() {
  local n=$1
  shift
  if [[ $n == b5 ]]; then
    (cd "../fig-$n" && env -u CARGO_ENCODED_RUSTFLAGS RUSTFLAGS="${RUSTFLAGS:-} --cfg fig_b5" CARGO_TARGET_DIR="$PWD/target" "$@")
  else
    (cd "../fig-$n" && CARGO_TARGET_DIR="$PWD/target" "$@")
  fi
}

for n in rel b5 main; do
  mine=()
  for p in "${probes[@]}"; do
    trees_of "$p" | grep -qx "$n" && mine+=("$p")
  done
  ((${#mine[@]})) || continue
  git fetch -q --depth=1 origin "${sha[$n]}"
  git worktree add -q --detach "../fig-$n" "${sha[$n]}" || exit 1
  in_tree "$n" git log --oneline -1
  for p in "${mine[@]}"; do
    cp "$here/probes/$p.rs" "../fig-$n/crates/incr-core/tests/"
    if in_tree "$n" cargo test --release -p incr-core --test "$p" --no-run >"$logs/$n-$p.log" 2>&1; then
      built[$n:$p]=1
    else
      echo "FIG-BUILD-FAILED $n $p"
      grep -a -E '^error' -A6 "$logs/$n-$p.log" | head -20
    fi
  done
done

for p in "${probes[@]}"; do
  for r in $(seq "$rounds"); do
    for n in $(trees_of "$p" | tac); do
      [[ -n ${built[$n:$p]:-} ]] || continue
      in_tree "$n" env FIGURE_TREE="r$r-$n" NC_TREE="r$r-$n" cargo test -q --release -p incr-core --test "$p" -- --nocapture 2>&1 |
        grep -a '^NC \|^FIG \|panicked\|FAILED'
    done
  done
done
echo "runner=${RUNNER_NAME:-unknown}"
