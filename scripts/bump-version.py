#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RELEASE_FILES = [
    "crates/incr-compute/Cargo.toml",
    "crates/incr-concurrent/Cargo.toml",
    "crates/incr-python/Cargo.toml",
    "crates/incr-concurrent-python/Cargo.toml",
    "crates/incr-python/pyproject.toml",
    "crates/incr-concurrent-python/pyproject.toml",
]

# Anchored to start-of-line so dependency lines like
# `pyo3 = { version = "0.23", ... }` are never touched.
VERSION_LINE_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


class BumpError(RuntimeError):
    pass


def run(
    cmd: list[str], check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=capture,
    )


def preflight(target_version: str) -> str:
    if not SEMVER_RE.match(target_version):
        raise BumpError(
            f"{target_version!r} is not a valid semver (expected MAJOR.MINOR.PATCH)"
        )

    status = run(["git", "status", "--porcelain"], capture=True).stdout
    if status.strip():
        raise BumpError(
            "working tree is not clean; commit or stash before bumping:\n" + status
        )

    branch = run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True
    ).stdout.strip()
    if branch != "main":
        raise BumpError(f"must be on main branch (currently on {branch!r})")

    versions: dict[str, str] = {}
    for rel in RELEASE_FILES:
        content = (REPO_ROOT / rel).read_text()
        match = VERSION_LINE_RE.search(content)
        if not match:
            raise BumpError(f"no top-level version line found in {rel}")
        versions[rel] = match.group(1)

    distinct = set(versions.values())
    if len(distinct) != 1:
        detail = "\n".join(f"  {rel}: {v}" for rel, v in versions.items())
        raise BumpError("release files are out of sync:\n" + detail)

    current = distinct.pop()
    if current == target_version:
        raise BumpError(f"all files already at {target_version}; nothing to bump")

    existing_tag = run(
        ["git", "tag", "--list", f"v{target_version}"], capture=True
    ).stdout.strip()
    if existing_tag:
        raise BumpError(f"tag v{target_version} already exists")

    return current


def rewrite_file(path: Path, current: str, target: str) -> None:
    content = path.read_text()
    match = VERSION_LINE_RE.search(content)
    if not match or match.group(1) != current:
        raise BumpError(
            f"{path}: expected current version {current!r}, "
            f"found {match.group(1) if match else None!r}"
        )
    # subn count=1 replaces only the top-level [package] version. The regex
    # is anchored to start-of-line so inline dependency tables like
    # `pyo3 = { version = "0.23", ... }` cannot match.
    new_content, n = VERSION_LINE_RE.subn(f'version = "{target}"', content, count=1)
    if n != 1:
        raise BumpError(f"failed to rewrite version in {path}")
    path.write_text(new_content)


def verify_post(target: str) -> None:
    for rel in RELEASE_FILES:
        content = (REPO_ROOT / rel).read_text()
        match = VERSION_LINE_RE.search(content)
        if not match or match.group(1) != target:
            raise BumpError(f"{rel} did not pick up the new version")


def sanity_check_build() -> None:
    print("[check] cargo check --workspace")
    try:
        run(["cargo", "check", "--workspace", "--quiet"])
    except subprocess.CalledProcessError as exc:
        raise BumpError(
            f"cargo check failed after bump (exit {exc.returncode}); aborting"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="new version, e.g. 0.2.0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the diff that would be applied and exit without modifying anything",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip the cargo check sanity pass (use for docs-only bumps)",
    )
    args = parser.parse_args()

    try:
        current = preflight(args.version)
    except BumpError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    print(f"Bumping {current} -> {args.version}")
    print("Files:")
    for rel in RELEASE_FILES:
        print(f"  {rel}")

    if args.dry_run:
        print("\n(dry run) no files modified")
        return 0

    try:
        for rel in RELEASE_FILES:
            rewrite_file(REPO_ROOT / rel, current, args.version)
        verify_post(args.version)
    except BumpError as err:
        print(f"error: {err}", file=sys.stderr)
        print("rolling back with `git checkout --`", file=sys.stderr)
        run(["git", "checkout", "--"] + RELEASE_FILES, check=False)
        return 3

    if not args.no_check:
        try:
            sanity_check_build()
        except BumpError as err:
            print(f"error: {err}", file=sys.stderr)
            print("rolling back with `git checkout --`", file=sys.stderr)
            run(["git", "checkout", "--"] + RELEASE_FILES, check=False)
            return 4

    run(["git", "add", "--"] + RELEASE_FILES)
    run(
        [
            "git",
            "commit",
            "-m",
            f"chore: bump version to v{args.version}",
        ]
    )
    run(
        [
            "git",
            "tag",
            "-a",
            f"v{args.version}",
            "-m",
            f"Release v{args.version}",
        ]
    )

    print(f"\n[ok] commit + tag v{args.version} created locally")
    return 0


if __name__ == "__main__":
    sys.exit(main())
