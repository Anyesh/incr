#!/usr/bin/env bash
set -uo pipefail
(($# == 2)) || { echo "usage: rates.sh <release-sha> <row>" >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
root=$PWD
row=$2
declare -A sha=([rel]=$1 [b5]=c61be873c4d6681cd0b59333441267749f4776cc [main]=b9f0d0bf3b11adec48b81eee123f9921a8ef694c)
spec=$(awk -F'\t' -v r="$row" 'NR > 1 && $1 == r' "$here/rates/rates.tsv")
[[ -n $spec ]] || { echo "rates.tsv: no row $row" >&2; exit 2; }
IFS=$'\t' read -r _ seed mode profile runs trees match _ <<<"$spec"
runs=${RATE_RUNS:-$runs}
case $mode in
  two) envs="REPLAY_SHARED=1 REPLAY_TWO=1" ;;
  crosstwo) envs="REPLAY_SHARED=1 REPLAY_TWO=1 REPLAY_CROSS=1" ;;
  *) echo "rates.tsv: $row has unknown mode $mode" >&2; exit 2 ;;
esac
flags=()
[[ $profile == release ]] && flags=(--release)
runner=${RUNNER_NAME:-unknown}
failed=0
names=() bins=()
for t in ${trees//,/ }; do
  s=${sha[$t]:-$t}
  wt=$root/../rate-$row-$t
  if ! { git fetch -q --depth=1 origin "$s" || git fetch -q --depth=1 https://github.com/Anyesh/incr.git "$s"; } ||
    ! git worktree add -q --detach "$wt" FETCH_HEAD; then
    echo "RATE-FETCH-FAILED $row $t $s"
    failed=1
    continue
  fi
  echo "point $row $t $(git -C "$wt" log -1 --format='%H %s')"
  b=$(cd "$wt" && CARGO_TARGET_DIR=$wt/target cargo test -q -p incr-core "${flags[@]}" --lib --no-run --message-format=json 2>/dev/null |
    jq -r 'select(.reason == "compiler-artifact" and .executable != null and .target.kind == ["lib"]) | .executable' | tail -n 1)
  if [[ ! -x $b ]]; then
    echo "RATE-BUILD-FAILED $row $t"
    failed=1
    continue
  fi
  names+=("$t") bins+=("$b")
done
export ENVS=$envs SEED=$seed MATCH=$match
# One run: "<tree index> pass", "<k> match | <text>" (a failure whose text
# holds MATCH, or any failure when MATCH is *), "<k> fail | <text>" (another
# failure text) or "<k> other <panic or timeout>" (no differential verdict).
one() {
  local o rc t
  o=$(env $ENVS REPLAY_CHECK=$SEED timeout 600 "$2" --ignored --exact runtime::replay_differential::replay_scan --test-threads=1 2>&1)
  rc=$?
  if ((rc == 0)); then echo "$1 pass"; return; fi
  if ((rc == 124)); then echo "$1 other timeout"; return; fi
  t=$(grep -aE -A3 '^seed 0x[0-9a-f]+[,:]' <<<"$o" | head -n 4 | sed -E 's/^seed 0x[0-9a-f]+, //' | cut -c1-600 | paste -sd'|')
  if [[ -z $t ]]; then
    echo "$1 other $(grep -a -E -A2 "panicked" <<<"$o" | head -n 3 | paste -sd'|' | cut -c1-300)"
  elif [[ $MATCH == '*' || $t == *"$MATCH"* ]]; then
    echo "$1 match | $t"
  else
    echo "$1 fail | $t"
  fi
}
export -f one
if ((${#bins[@]})); then
  for _ in $(seq 1 "$runs"); do for k in "${!bins[@]}"; do echo "$k ${bins[$k]}"; done; done |
    xargs -P 2 -L 1 bash -c 'one "$0" "$1"' > "$root/rate-$row.out"
fi
for k in "${!names[@]}"; do
  t=${names[$k]}
  n=$(grep -cE "^$k (pass|match|fail|other)" "$root/rate-$row.out")
  m=$(grep -cE "^$k match" "$root/rate-$row.out")
  f=$(grep -cE "^$k fail" "$root/rate-$row.out")
  x=$(grep -cE "^$k other" "$root/rate-$row.out")
  echo "RATE $row $t ${sha[$t]:-$t} $mode $profile seed=0x$seed matched $m/$n other-text $f other $x runner=$runner"
  ((n == runs)) || { echo "RATE-SHORT $row $t $n of $runs"; failed=1; }
  for c in match fail other; do
    grep -E "^$k $c" "$root/rate-$row.out" | cut -d' ' -f2- | sort | uniq -c | sort -rn | head -n 10 | sed "s/^/TEXT $row $t /"
  done
done
echo "RATE-DONE $row failed=$failed"
exit "$failed"
