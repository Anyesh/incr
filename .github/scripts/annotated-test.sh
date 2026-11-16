#!/usr/bin/env bash
# usage: annotated-test.sh <timeout-seconds> <label> <command...>
# Runs the command under a timeout, tees its output, and on failure or
# timeout emits two ::error:: annotations (job logs need sign-in, check-run
# annotations are public).
set -uo pipefail
limit=$1
label=$2
shift 2
log=$(mktemp)
rc=0
timeout -k 30 "$limit" "$@" 2>&1 | tee "$log" || rc=${PIPESTATUS[0]}
if [ "$rc" -eq 0 ]; then
  exit 0
fi

sed -i 's/\x1b\[[0-9;]*[mK]//g' "$log"

escape() {
  awk 'BEGIN{ORS=""} {gsub(/%/,"%25"); gsub(/\r/,"%0D"); print (NR>1?"%0A":"") $0}'
}

stuck=$(awk '
  /has been running for over/ { n=$2; started[n]=1 }
  /^test .* \.\.\. (ok|FAILED|ignored)/ { done[$2]=1 }
  END { for (n in started) if (!(n in done)) print n }' "$log" | sort)
failed=$(grep -E '^test .* \.\.\. FAILED' "$log" | awk '{print $2}' | sort -u)
seeds=$(grep -E 'minimal failing input|^Test failed: seed|seed 0x' "$log" | cut -c1-300 | sort -u | head -8)
panics=$(grep -E "panicked at" "$log" | cut -c1-300 | sort -u | head -12)
binary=$(grep -E '^ +(Running|Doc-tests)' "$log" | tail -1)
slow=$(grep -E 'has been running for over' "$log" | sort -u | head -12)

{
  echo "$label: exit $rc$([ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] && echo " (timed out after ${limit}s)")"
  echo "last binary started: ${binary:-none}"
  echo "still running at the end: ${stuck:-none}"
  echo "failed tests: ${failed:-none}"
  echo "seeds:"
  echo "$seeds"
} | escape | { printf '::error title=%s summary::' "$label"; cat; echo; }

{
  echo "panics:"
  echo "$panics"
  echo "running over 60s:"
  echo "$slow"
  echo "last 20 lines:"
  tail -20 "$log" | cut -c1-300
} | escape | { printf '::error title=%s detail::' "$label"; cat; echo; }

exit "$rc"
