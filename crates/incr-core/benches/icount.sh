#!/usr/bin/env bash
# Per-iteration user-space instruction counts for the icount driver:
# (instructions:u at N iterations - instructions:u at 0) / N, per workload.
# Prints CSV: workload,iterations,count_n,count_0,per_iteration.
#
# usage: icount.sh [-n iterations] [-w warm-up] [-e event] [workload ...]
#   -n  iterations for every workload, instead of each one's default
#   -w  warm-up iterations, run at both N and 0 (default 100)
#   -e  the perf event to count instead of instructions:u, such as
#       cycles:u for T6's table-hit gate
#   With no workloads named, runs every one the driver lists.
set -euo pipefail

usage() {
	echo "usage: icount.sh [-n iterations] [-w warm-up] [-e event] [workload ...]" >&2
	exit 2
}

iterations=""
warmup=100
event=instructions:u
while getopts "n:w:e:h" opt; do
	case $opt in
	n) iterations=$OPTARG ;;
	w) warmup=$OPTARG ;;
	e) event=$OPTARG ;;
	*) usage ;;
	esac
done
shift $((OPTIND - 1))

manifest="$(cd "$(dirname "$0")/.." && pwd)/Cargo.toml"
driver=$(cargo bench --manifest-path "$manifest" --bench icount --no-run --message-format=json 2>/dev/null |
	jq -r 'select(.reason == "compiler-artifact" and .target.name == "icount") | .executable')
if [[ ! -x $driver ]]; then
	echo "icount.sh: could not build the icount driver" >&2
	exit 1
fi

declare -A default_n
while read -r name n; do
	default_n[$name]=$n
done < <("$driver" --list)

if (($# == 0)); then
	mapfile -t workloads < <("$driver" --list | cut -d' ' -f1)
else
	workloads=("$@")
fi

out=$(mktemp)
trap 'rm -f "$out"' EXIT

count() {
	perf stat -x, -e "$event" -o "$out" -- "$driver" "$1" "$2" "$warmup" >/dev/null
	awk -F, -v e="${event%%:*}" '$3 ~ "^" e { print $1 }' "$out"
}

echo "workload,iterations,count_n,count_0,per_iteration"
for w in "${workloads[@]}"; do
	if [[ -z ${default_n[$w]+set} ]]; then
		echo "icount.sh: unknown workload $w" >&2
		exit 1
	fi
	n=${iterations:-${default_n[$w]}}
	((n > 0)) || usage
	full=$(count "$w" "$n")
	empty=$(count "$w" 0)
	awk -v w="$w" -v n="$n" -v a="$full" -v b="$empty" \
		'BEGIN { printf "%s,%d,%d,%d,%.2f\n", w, n, a, b, (a - b) / n }'
done
