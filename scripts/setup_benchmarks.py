#!/usr/bin/env python3
"""Build one virtual environment per Avengers benchmark subject.

One environment per subject, not one shared one: the EMSE profile pins versions
from 2021 whose dependency ranges genuinely conflict with each other, and a
single environment would silently resolve them to something that is no longer
the pinned corpus.

The environments live outside the repository -- they are a cache, not a
deliverable -- under ``.benchmark-venvs`` by default.  Each holds the subject
plus every backend, so Pynguin and CrossHair analyse the subject with exactly
the interpreter that imports it.

Usage::

    python scripts/setup_benchmarks.py --profile current-3.12
    python scripts/setup_benchmarks.py --profile emse-3.10 --only codetiming
    python scripts/setup_benchmarks.py --profile current-3.12 --smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "benchmarks" / "avengers" / "manifest.json"


def build(
    name: str,
    requirement: str,
    python: str,
    root: Path,
    backends: list[str],
    flatting: Path | None,
) -> bool:
    environment = root / name
    interpreter = environment / "bin" / "python"
    if not shutil.which("uv"):
        print("uv is required: https://docs.astral.sh/uv/", file=sys.stderr)
        return False
    steps = [
        ["uv", "venv", "--python", python, str(environment)],
        ["uv", "pip", "install", "--python", str(interpreter), requirement, *backends],
        # Tharkun itself, so the worker the parent spawns in this environment
        # can import it. Editable, so a change here is a change there.
        ["uv", "pip", "install", "--python", str(interpreter), "-e", str(REPO)],
    ]
    if flatting is not None:
        # State coverage is one of the measured objectives, and the flatting
        # digest distinguishes more states than the dependency-free fallback
        # does. Without it the benchmark systematically UNDER-reports state
        # coverage for every configuration alike.
        steps.append(
            ["uv", "pip", "install", "--python", str(interpreter), "-e", str(flatting)]
        )
    for command in steps:
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            print(f"  {name}: FAILED ({' '.join(command[:3])})", file=sys.stderr)
            print("   ", completed.stderr.strip().splitlines()[-1:], file=sys.stderr)
            return False
    print(f"  {name}: {interpreter}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="current-3.12")
    parser.add_argument("--root", default=str(REPO / ".benchmark-venvs"))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--smoke", action="store_true", help="only the smoke subjects")
    parser.add_argument(
        "--flatting",
        default=str(REPO.parent / "py_obj_flatting"),
        help="checkout of py_obj_flatting to install; '' to skip it",
    )
    arguments = parser.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text())
    profile = manifest["profiles"].get(arguments.profile)
    if profile is None:
        print(f"unknown profile {arguments.profile!r}", file=sys.stderr)
        return 2
    wanted = set(arguments.only) or (set(manifest["smoke"]) if arguments.smoke else None)
    root = Path(arguments.root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"profile {arguments.profile} (python {profile['python']}) -> {root}")
    if profile.get("comparable_to"):
        print(f"comparable to: {profile['comparable_to']}")
    else:
        print("NOT comparable to the published baseline; this profile is for iteration")

    flatting = Path(arguments.flatting) if arguments.flatting else None
    if flatting is not None and not (flatting / "pyproject.toml").is_file():
        print(f"py_obj_flatting not found at {flatting}; state coverage will be coarser")
        flatting = None

    failures = 0
    for name, requirement in sorted(profile["subjects"].items()):
        if wanted is not None and name not in wanted:
            continue
        if not build(
            name, requirement, profile["python"], root, manifest["backends"], flatting
        ):
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
