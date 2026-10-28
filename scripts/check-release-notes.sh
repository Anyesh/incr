#!/usr/bin/env bash
set -euo pipefail

# Release prep marks figures and conditional lines that must be settled before
# the tag with a DO NOT SHIP comment; the tag must not publish one of them.

cd "$(dirname "$0")/.."

tag="${1:-${GITHUB_REF_NAME:?pass the tag, e.g. v0.2.0-beta.6}}"
version="${tag#v}"
status=0

heading=$(grep -m1 -F "## [$version]" CHANGELOG.md || true)
if [[ -z "$heading" ]]; then
    echo "CHANGELOG.md has no ## [$version] section"
    status=1
elif [[ ! "$heading" =~ [0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "CHANGELOG.md section for $version is not dated: $heading"
    status=1
fi

markers=$(grep -n -o 'DO NOT SHIP' CHANGELOG.md | cut -d: -f1 | paste -sd, - || true)
if [[ -n "$markers" ]]; then
    echo "CHANGELOG.md has DO NOT SHIP markers on lines $markers"
    status=1
fi
exit $status
