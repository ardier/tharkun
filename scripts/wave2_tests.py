"""Run the two real timing-sensitive tests on an exact source archive.

The watcher owns queue admission, timeout and descendant cleanup. This script
never kills process groups or treats skipped/missing tests as a successful run.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET



def source_hashes(root: Path, manifest: dict, nodes: list[str] = ()) -> None:
    files = manifest.get("files")
    required = {"pytest.ini", "scripts/wave2_tests.py",
                *(node.split("::", 1)[0] for node in nodes)}
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("source manifest omits test/config/runner bytes")
    for name, digest in files.items():
        path = root / name
        if path.resolve().is_relative_to(root.resolve()) is False:
            raise ValueError("source manifest escapes the owned checkout")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source changed: {name}")


def collected(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()
            if line.startswith("tests/") and "::" in line]


def junit_status(path: Path, nodes: list[str]) -> tuple[str, dict]:
    try:
        cases = list(ET.parse(path).getroot().iter("testcase"))
    except (OSError, ET.ParseError):
        return "INCOMPLETE", {"reason": "missing or malformed JUnit"}
    identities = [(case.attrib.get("classname"), case.attrib.get("name")) for case in cases]
    wanted = []
    for node in nodes:
        parts = node.split("::")
        module = parts[0].removesuffix(".py").replace("/", ".")
        wanted.append((".".join([module, *parts[1:-1]]), parts[-1]))
    counts = {kind: sum(case.find(kind) is not None for case in cases)
              for kind in ("failure", "error", "skipped")}
    counts["tests"] = len(cases)
    if sorted(identities) != sorted(wanted) or counts["skipped"]:
        return "INCOMPLETE", dict(counts, reason="missing, unexpected or skipped test")
    if counts["failure"] or counts["error"]:
        return "FAIL", counts
    return "PASS", counts


def run(root: Path, out: Path, manifest_path: Path) -> int:
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"status": "INCOMPLETE", "critical_status": "COLLECTED_ONLY"}
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        source_hashes(root, manifest)
        result.update(source_sha=manifest["sha"], source_tree=manifest["tree"],
                      manifest_sha256=hashlib.sha256(raw).hexdigest(),
                      python=sys.version, executable=sys.executable,
                      affinity=sorted(os.sched_getaffinity(0)))
        try:
            result["packages"] = {name: version(name) for name in ("pytest", "pytest-xdist", "hypothesis")}
        except PackageNotFoundError as error:
            raise ValueError("missing test dependency: " + str(error)) from error
        env = os.environ.copy()
        env.update(PYTHONPATH=str(root / "src"), PYTEST_ADDOPTS="",
                   PYTHONDONTWRITEBYTECODE="1")
        prefix = [sys.executable, "-B", "-m", "pytest", "-q", "-n", "0",
                  "-o", "addopts="]
        inventories = {}
        for label, extra in (("full", []), ("critical", ["-m", "not wave2"]),
                             ("wave2", ["-m", "wave2"])):
            log = out / f"collect-{label}.log"
            with log.open("w") as stream:
                proc = subprocess.run(prefix + ["--collect-only", *extra, "tests"],
                                      cwd=root, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT)
            nodes = collected(log.read_text())
            if proc.returncode or not nodes or len(nodes) != len(set(nodes)):
                raise ValueError(f"incomplete {label} collection")
            inventories[label] = nodes
        full, critical, deferred = (set(inventories[key]) for key in ("full", "critical", "wave2"))
        if critical & deferred or full != critical | deferred:
            raise ValueError("full collection differs from disjoint critical/deferred collections")
        source_hashes(root, manifest, inventories["wave2"])
        result.update(full_nodes=inventories["full"],
                      critical_nodes=inventories["critical"], deferred_nodes=inventories["wave2"],
                      critical_count=len(critical), deferred_count=len(deferred), full_count=len(full))
        junit = out / "wave2.xml"
        command = prefix + ["-m", "wave2", "--junitxml", str(junit), *inventories["wave2"]]
        result["command"] = command
        with (out / "pytest.log").open("w") as stream:
            proc = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT)
        status, counts = junit_status(junit, inventories["wave2"])
        if proc.returncode not in (0, 1) or (proc.returncode == 1 and status == "PASS"):
            status = "INCOMPLETE"
        result.update(status=status, pytest_rc=proc.returncode, counts=counts)
        source_hashes(root, manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status="INCOMPLETE", reason=str(exc))
    result["elapsed_seconds"] = time.monotonic() - started
    result["logs"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in out.iterdir() if path.is_file()}
    (out / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("full_nodes", "critical_nodes")}), flush=True)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[result["status"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    return run(args.root, args.out, args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
