#!/usr/bin/env bash
set -euo pipefail

# The pyproject versions are static, so a tag can reach PyPI with a version
# the bump missed; maturin publish --skip-existing would then upload nothing.

cd "$(dirname "$0")/.."

tag="${1:-${GITHUB_REF_NAME:?pass the tag, e.g. v0.2.0-beta.6}}"
version="${tag#v}"
pep440=$(sed -E 's/-alpha\./a/; s/-beta\./b/; s/-rc\./rc/' <<<"$version")

workspace=$(sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"/\1/p' Cargo.toml)
status=0
if [[ "$workspace" != "$version" ]]; then
    echo "Cargo.toml workspace version is $workspace, the tag is $version"
    status=1
fi
for p in crates/incr-python/pyproject.toml crates/incr-concurrent-python/pyproject.toml; do
    v=$(sed -n 's/^version = "\(.*\)"/\1/p' "$p")
    if [[ "$v" != "$pep440" ]]; then
        echo "$p version is $v, the tag needs $pep440"
        status=1
    fi
done
exit $status
