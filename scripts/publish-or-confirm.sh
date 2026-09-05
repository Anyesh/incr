#!/usr/bin/env bash
set -euo pipefail

crate="$1"
version="${GITHUB_REF_NAME#v}"

if cargo publish -p "$crate"; then
    exit 0
fi

echo "cargo publish reported failure for $crate; checking crates.io directly" \
    "in case the upload landed but the post-publish availability wait timed out"

for _ in 1 2 3 4 5; do
    sleep 15
    if curl -sf "https://crates.io/api/v1/crates/$crate/$version" >/dev/null; then
        echo "$crate $version confirmed on crates.io"
        exit 0
    fi
done

echo "$crate $version is not on crates.io; the publish genuinely failed"
exit 1
