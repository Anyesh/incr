#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = EXAMPLE_DIR.parent.parent
VENV_DIR = EXAMPLE_DIR / ".venv"


def run(cmd, **kwargs):
    print(f"  > {' '.join(str(c) for c in cmd)}")
    subprocess.check_call(cmd, **kwargs)


def main():
    print("incr travel premium demo\n")

    python = str(VENV_DIR / "bin" / "python")

    if not VENV_DIR.exists():
        print("[1/4] Creating virtual environment...")
        run(["uv", "venv", str(VENV_DIR), "--python", "3.12"])
    else:
        print("[1/4] Virtual environment exists")

    print("[2/4] Installing dependencies...")
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            python,
            "maturin",
            "fastapi",
            "uvicorn[standard]",
        ]
    )

    print("[3/4] Building incr Python bindings...")
    run(
        [python, "-m", "maturin", "develop"],
        cwd=str(REPO_ROOT),
        env={**os.environ, "VIRTUAL_ENV": str(VENV_DIR)},
    )

    print("[4/4] Starting server at http://127.0.0.1:8001\n")
    os.chdir(EXAMPLE_DIR)
    os.execvp(
        python,
        [
            python,
            "-m",
            "uvicorn",
            "server:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
        ],
    )


if __name__ == "__main__":
    main()
