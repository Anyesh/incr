#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every quoted occurrence of the current version in these files is a release
# version: workspace version and pins, Python pins, pyproject (PEP 440), README.
VERSION_FILES = [
    "Cargo.toml",
    "crates/incr-python/Cargo.toml",
    "crates/incr-concurrent-python/Cargo.toml",
    "crates/incr-python/pyproject.toml",
    "crates/incr-concurrent-python/pyproject.toml",
    "README.md",
]
CHANGELOG = "CHANGELOG.md"

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
WORKSPACE_VERSION_RE = re.compile(
    r'^\[workspace\.package\]\n(?:[^\[\n].*\n)*?version = "([^"]+)"$', re.MULTILINE
)
PRERELEASE_TAGS = {"alpha": "a", "beta": "b", "rc": "rc"}


class BumpError(RuntimeError):
    pass


def run(
    cmd: list[str], check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=REPO_ROOT, check=check, text=True, capture_output=capture
    )


def pep440(version: str) -> str:
    return re.sub(r"-(alpha|beta|rc)\.", lambda m: PRERELEASE_TAGS[m.group(1)], version)


def current_version() -> str:
    match = WORKSPACE_VERSION_RE.search((REPO_ROOT / "Cargo.toml").read_text())
    if not match:
        raise BumpError("no version in Cargo.toml [workspace.package]")
    return match.group(1)


def preflight(target: str) -> str:
    if not SEMVER_RE.match(target):
        raise BumpError(f"{target!r} is not a valid semver")
    status = run(
        ["git", "status", "--porcelain", "--untracked-files=no"], capture=True
    ).stdout
    if status.strip():
        raise BumpError("tracked files have changes; commit or stash first:\n" + status)
    branch = run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True
    ).stdout.strip()
    if branch != "main":
        raise BumpError(f"must be on main branch (currently on {branch!r})")
    current = current_version()
    if current == target:
        raise BumpError(f"already at {target}; nothing to bump")
    if run(["git", "tag", "--list", f"v{target}"], capture=True).stdout.strip():
        raise BumpError(f"tag v{target} already exists")
    return current


def bumped_versions(current: str, target: str) -> dict[str, str]:
    pairs = [
        (f'"={current}"', f'"={target}"'),
        (f'"{current}"', f'"{target}"'),
        (f'"{pep440(current)}"', f'"{pep440(target)}"'),
    ]
    out = {}
    for rel in VERSION_FILES:
        content = (REPO_ROOT / rel).read_text()
        new = content
        for old, repl in pairs:
            new = new.replace(old, repl)
        if new == content:
            raise BumpError(f"{rel} has no quoted {current} or {pep440(current)}")
        out[rel] = new
    return out


def dated_changelog(target: str, date: str) -> str:
    content = (REPO_ROOT / CHANGELOG).read_text()
    heading = re.compile(
        rf"^## \[(?:Unreleased|{re.escape(target)})\](?: - Unreleased)?$",
        re.MULTILINE,
    )
    new, n = heading.subn(f"## [{target}] - {date}", content)
    if n != 1:
        raise BumpError(
            f"{CHANGELOG} needs exactly one '## [Unreleased]' or "
            f"'## [{target}] - Unreleased' heading, found {n}"
        )
    return new


def release_checks(target: str) -> None:
    for cmd in (
        ["bash", "scripts/check-release-versions.sh", f"v{target}"],
        ["bash", "scripts/check-release-notes.sh", f"v{target}"],
        ["cargo", "check", "--workspace", "--quiet"],
    ):
        print("[check] " + " ".join(cmd))
        if run(cmd, check=False).returncode != 0:
            raise BumpError(f"{' '.join(cmd)} failed after the bump")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bump every release version, date the CHANGELOG entry, "
        "then commit and tag locally."
    )
    parser.add_argument("version", help="new version, e.g. 0.2.0-beta.13")
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip the release scripts and cargo check",
    )
    args = parser.parse_args()
    touched = VERSION_FILES + [CHANGELOG]

    try:
        current = preflight(args.version)
        rewrites = bumped_versions(current, args.version)
        rewrites[CHANGELOG] = dated_changelog(args.version, args.date)
    except BumpError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    print(f"Bumping {current} -> {args.version}, dated {args.date}")
    if args.dry_run:
        print("\n".join(f"  {rel}" for rel in touched))
        return 0

    for rel, content in rewrites.items():
        (REPO_ROOT / rel).write_text(content)
    if not args.no_check:
        try:
            release_checks(args.version)
        except BumpError as err:
            print(f"error: {err}; restoring the files", file=sys.stderr)
            run(["git", "checkout", "--"] + touched, check=False)
            return 3

    run(["git", "add", "--"] + touched)
    run(
        [
            "git",
            "commit",
            "-q",
            "-m",
            f"chore: bump to {args.version} and date its changelog entry",
        ]
    )
    run(["git", "tag", "-a", f"v{args.version}", "-m", args.version])
    print(f"[ok] commit and tag v{args.version} created locally; nothing pushed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
