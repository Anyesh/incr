import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "bump-version.py"

FIXTURE = {
    "Cargo.toml": """[workspace]
members = []

[workspace.package]
version = "0.2.0-beta.11"
license = "AGPL-3.0-only"

[workspace.dependencies]
incr-core = { version = "=0.2.0-beta.11", path = "crates/incr-core" }
incr-concurrent = { version = "=0.2.0-beta.11", path = "crates/incr-concurrent" }
pyo3 = { version = "0.23" }
""",
    "crates/incr-compute/Cargo.toml": """[package]
name = "incr-compute"
version.workspace = true
""",
    "crates/incr-python/Cargo.toml": """[package]
name = "incr-python"
version.workspace = true

[dependencies]
incr_compute = { package = "incr-compute", version = "=0.2.0-beta.11", path = "../incr-compute" }
""",
    "crates/incr-concurrent-python/Cargo.toml": """[package]
name = "incr-concurrent-python"
version.workspace = true

[dependencies]
incr_concurrent = { package = "incr-concurrent", version = "=0.2.0-beta.11", path = "../incr-concurrent" }
""",
    "crates/incr-python/pyproject.toml": """[project]
name = "incr-compute"
version = "0.2.0b11"
""",
    "crates/incr-concurrent-python/pyproject.toml": """[project]
name = "incr-concurrent"
version = "0.2.0b11"
""",
    "README.md": """```toml
incr-compute = "0.2.0-beta.11"      # single-threaded
incr-concurrent = "0.2.0-beta.11"   # multi-threaded (Send + Sync)
```

Versions 0.2.0-beta.11 and earlier were Apache-2.0.
""",
    "CHANGELOG.md": """# Changelog

## [Unreleased]

### Changed

- Something.

## [0.2.0-beta.11] - 2026-11-17
""",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for rel, content in FIXTURE.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "bump-version.py")
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    (tmp_path / "untracked.txt").write_text("not part of the release\n")
    return tmp_path


def bump(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "bump-version.py"), *args],
        cwd=repo,
        text=True,
        capture_output=True,
    )


def test_bump_rewrites_every_version_and_dates_the_changelog(repo: Path) -> None:
    result = bump(repo, "0.2.0-beta.12", "--no-check", "--date", "2026-11-18")
    assert result.returncode == 0, result.stderr

    cargo = (repo / "Cargo.toml").read_text()
    assert 'version = "0.2.0-beta.12"\nlicense' in cargo
    assert cargo.count('"=0.2.0-beta.12"') == 2
    assert 'pyo3 = { version = "0.23" }' in cargo
    assert (repo / "crates/incr-compute/Cargo.toml").read_text() == FIXTURE[
        "crates/incr-compute/Cargo.toml"
    ]
    assert '"=0.2.0-beta.12"' in (repo / "crates/incr-python/Cargo.toml").read_text()
    assert (
        '"=0.2.0-beta.12"'
        in (repo / "crates/incr-concurrent-python/Cargo.toml").read_text()
    )
    for rel in (
        "crates/incr-python/pyproject.toml",
        "crates/incr-concurrent-python/pyproject.toml",
    ):
        assert 'version = "0.2.0b12"' in (repo / rel).read_text()
    readme = (repo / "README.md").read_text()
    assert 'incr-compute = "0.2.0-beta.12"' in readme
    assert 'incr-concurrent = "0.2.0-beta.12"' in readme
    assert "Versions 0.2.0-beta.11 and earlier" in readme
    changelog = (repo / "CHANGELOG.md").read_text()
    assert "## [0.2.0-beta.12] - 2026-11-18\n" in changelog
    assert "[Unreleased]" not in changelog

    assert (
        git(repo, "log", "-1", "--format=%s").strip()
        == "chore: bump to 0.2.0-beta.12 and date its changelog entry"
    )
    assert git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert (
        git(repo, "tag", "-l", "--format=%(contents:subject)", "v0.2.0-beta.12").strip()
        == "0.2.0-beta.12"
    )


def test_bump_accepts_a_changelog_heading_marked_unreleased(repo: Path) -> None:
    path = repo / "CHANGELOG.md"
    path.write_text(
        path.read_text().replace("## [Unreleased]", "## [0.2.0-beta.12] - Unreleased")
    )
    git(repo, "commit", "-q", "-am", "name the section")
    result = bump(repo, "0.2.0-beta.12", "--no-check", "--date", "2026-11-18")
    assert result.returncode == 0, result.stderr
    assert "## [0.2.0-beta.12] - 2026-11-18\n" in path.read_text()


def test_bump_without_an_unreleased_section_changes_nothing(repo: Path) -> None:
    path = repo / "CHANGELOG.md"
    path.write_text(path.read_text().replace("## [Unreleased]", "## [0.2.0-beta.10]"))
    git(repo, "commit", "-q", "-am", "no unreleased section")
    head = git(repo, "rev-parse", "HEAD")
    result = bump(repo, "0.2.0-beta.12", "--no-check", "--date", "2026-11-18")
    assert result.returncode != 0
    assert "CHANGELOG" in result.stderr
    assert git(repo, "status", "--porcelain", "--untracked-files=no") == ""
    assert git(repo, "rev-parse", "HEAD") == head
