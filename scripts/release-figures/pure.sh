#!/usr/bin/env bash
set -uo pipefail
(($# >= 2)) || { echo "usage: pure.sh <release-sha> <case>..." >&2; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
root=$PWD
rel=$1
shift
runs=${PURE_RUNS:-30000}
secs=${PURE_SECS:-1200}
end=$((SECONDS + ${PURE_TOTAL_SECS:-6000}))
runner=${RUNNER_NAME:-unknown}
wt=$root/../pure-$rel
if ! { git fetch -q --depth=1 origin "$rel" || git fetch -q --depth=1 https://github.com/Anyesh/incr.git "$rel"; } ||
  ! git worktree add -q --detach "$wt" FETCH_HEAD; then
  echo "PURE-FETCH-FAILED $rel"
  echo "PURE-DONE failed=1"
  exit 1
fi
echo "point pure $(git -C "$wt" log -1 --format='%H %s')"
t=$wt/crates/incr-core/tests
mkdir -p "$t/b7"
cp "$here/pure/b7_runner.rs" "$t/b7_runner.rs"
cp "$here/pure/api.rs" "$here/pure/cases.rs" "$t/b7/"
b=$(cd "$wt" && CARGO_TARGET_DIR=$wt/target cargo test -q -p incr-core --test b7_runner --no-run --message-format=json 2>/dev/null |
  jq -r 'select(.reason == "compiler-artifact" and .executable != null and .target.name == "b7_runner") | .executable' | tail -n 1)
if [[ ! -x $b ]]; then
  echo "PURE-BUILD-FAILED $rel"
  echo "PURE-DONE failed=1"
  exit 1
fi
failed=0
for c in "$@"; do
  start=$SECONDS
  left=$((end - start))
  if ((left <= 0)); then
    echo "PURE-SKIPPED $c $rel past PURE_TOTAL_SECS runner=$runner"
    failed=1
    continue
  fi
  cap=$((left < secs ? left : secs))
  o=$(B7_MODE=two B7_RUNS=$runs B7_PROJECT=1 B7_CASE=$c timeout "$cap" "$b" --ignored --exact b7_run --nocapture 2>&1)
  rc=$?
  w=$((SECONDS - start))
  s=$(grep -aP "^\Q$c\E\ttwo-summary\t" <<<"$o" | cut -f3)
  h=$(grep -aP "^\Q$c\E\ttwo-HANG\t" <<<"$o" | cut -f3)
  if ((rc == 124)); then
    echo "PURE-TIMEOUT $c $rel after $cap s runner=$runner"
    failed=1
  elif ! grep -aqx '# api full projected true' <<<"$o"; then
    echo "PURE-RUN-FAILED $c rc=$rc not projected: $(grep -a '^# api' <<<"$o" | head -n 1)"
    failed=1
  elif [[ -n $s ]] && ((rc == 0)); then
    echo "PURE $c $rel $s wall-s $w runner=$runner"
    [[ $s == "runs $runs "* ]] || { echo "PURE-SHORT $c $s of $runs"; failed=1; }
  elif [[ -n $h ]]; then
    echo "PURE-HANG $c $rel $h runner=$runner"
    failed=1
  else
    echo "PURE-RUN-FAILED $c rc=$rc $(grep -a -A2 panicked <<<"$o" | head -n 3 | paste -sd'|' | cut -c1-300)"
    failed=1
  fi
  grep -aP "^\Q$c\E\ttwo\t" <<<"$o" | cut -f3- | sed "s/^/PURE-TEXT $c /"
done
echo "PURE-DONE failed=$failed"
exit "$failed"
