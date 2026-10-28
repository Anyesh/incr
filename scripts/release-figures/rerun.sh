#!/usr/bin/env bash
set -uo pipefail
(($# == 1)) || { echo "usage: rerun.sh <release-sha>" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
. "$here/lib.sh" || { echo "release-figures: cannot load $here/lib.sh" >&2; exit 2; }
rounds=${FIGURE_ROUNDS:-3}
declare -A sha=([rel]=$1 [b5]=c61be873c4d6681cd0b59333441267749f4776cc [main]=b9f0d0bf3b11adec48b81eee123f9921a8ef694c)
declare -A against built kind
failed=0
probes=()
while IFS=$'\t' read -r probe vs _ src; do
  [[ -n $probe ]] || continue
  [[ -z ${FIGURE_PROBES:-} || " $FIGURE_PROBES " == *" $probe "* ]] || continue
  [[ -n ${sha[$vs]:-} || $vs == none ]] || { echo "probes.tsv: $probe has unknown against tree $vs" >&2; exit 2; }
  src=${src:-kit}
  [[ $src == kit || $src == tree ]] || { echo "probes.tsv: $probe has unknown source $src" >&2; exit 2; }
  [[ $src == tree || -f $here/probes/$probe.rs ]] || { echo "probes.tsv: no probes/$probe.rs" >&2; exit 2; }
  probes+=("$probe")
  against[$probe]=$vs
  kind[$probe]=$src
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
  checkout_sha "${sha[$n]}" "../fig-$n" || { echo "FIG-FETCH-FAILED $n ${sha[$n]}"; echo "FIG-DONE failed=1"; exit 1; }
  in_tree "$n" git log --oneline -1
  for p in "${mine[@]}"; do
    if [[ ${kind[$p]} == kit ]]; then
      cp "$here/probes/$p.rs" "../fig-$n/crates/incr-core/tests/"
    elif [[ ! -f ../fig-$n/crates/incr-core/tests/$p.rs ]]; then
      echo "FIG-MISSING $n $p"
      failed=1
      continue
    fi
    if in_tree "$n" cargo test --release -p incr-core --test "$p" --no-run >"$logs/$n-$p.log" 2>&1; then
      built[$n:$p]=1
    else
      echo "FIG-BUILD-FAILED $n $p"
      failed=1
      grep -a -E '^error' -A6 "$logs/$n-$p.log" | head -20
    fi
  done
done

for p in "${probes[@]}"; do
  for r in $(seq "$rounds"); do
    for n in $(trees_of "$p" | tac); do
      [[ -n ${built[$n:$p]:-} ]] || continue
      out=$logs/run-$r-$n-$p.log
      in_tree "$n" env FIGURE_TREE="r$r-$n" NC_TREE="r$r-$n" cargo test -q --release -p incr-core --test "$p" -- --nocapture >"$out" 2>&1
      rc=$?
      if [[ ${kind[$p]} == tree ]]; then
        grep -a ' B per \|panicked\|FAILED' "$out" | sed "s/^/FIG r$r-$n $p /"
      else
        grep -a '^NC \|^FIG \|panicked\|FAILED' "$out"
      fi
      if ((rc != 0)); then
        echo "FIG-RUN-FAILED r$r-$n $p rc=$rc"
        failed=1
      elif ! grep -a -q ' B per \|^NC \|^FIG ' "$out"; then
        echo "FIG-RUN-EMPTY r$r-$n $p"
        failed=1
      fi
    done
  done
done
echo "runner=${RUNNER_NAME:-unknown}"
echo "FIG-DONE failed=$failed"
exit "$failed"
