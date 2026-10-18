"""Coverage and fail-closed receipts for the owned Wave 2 runner."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/wave2_tests.py"
spec = importlib.util.spec_from_file_location("wave2_tests", SCRIPT)
wave2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wave2)

NODES = ("tests/test_search.py::Search::test_reaches", "tests/test_cpu.py::test_budget")
CRITICAL = ["tests/test_scheduler.py::critical"]


def fixture(tmp_path):
    root = tmp_path / "source"
    files = {"pytest.ini", "scripts/wave2_tests.py",
             *(node.split("::", 1)[0] for node in NODES)}
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sha": "a" * 40, "tree": "b" * 40,
        "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                  for name in files}}))
    return root, manifest


def mocked_pytest(monkeypatch, *, skipped=False, failed=False, missing=False,
                  wrong_inventory=False, mutate=None):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "--collect-only" in command:
            nodes = list(NODES)
            marker = command[command.index("-m", command.index("pytest") + 1) + 1] if command.count("-m") > 1 else None
            if marker is None:
                nodes += CRITICAL
            elif marker == "not wave2":
                nodes = CRITICAL
            elif wrong_inventory:
                nodes = nodes[:1]
            kwargs["stdout"].write("\n".join(nodes) + "\n")
            return SimpleNamespace(returncode=0)
        if mutate:
            mutate()
        if not missing:
            path = Path(command[command.index("--junitxml") + 1])
            body = "<testsuites><testsuite>"
            for i, node in enumerate(NODES):
                tag = "<skipped/>" if skipped and i == 0 else "<failure/>" if failed and i == 0 else ""
                parts = node.split("::")
                classname = ".".join([parts[0].removesuffix(".py").replace("/", "."), *parts[1:-1]])
                body += f'<testcase classname="{classname}" name="{parts[-1]}">{tag}</testcase>'
            path.write_text(body + "</testsuite></testsuites>")
        return SimpleNamespace(returncode=1 if failed else 0)

    monkeypatch.setattr(wave2.subprocess, "run", run)
    return calls


def test_complete_roster_records_full_and_actual_critical_collection(tmp_path, monkeypatch):
    root, manifest = fixture(tmp_path)
    calls = mocked_pytest(monkeypatch)
    out = tmp_path / "out"
    assert wave2.run(root, out, manifest) == 0
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "PASS" and result["critical_status"] == "COLLECTED_ONLY"
    assert result["full_count"] == 3 and result["critical_count"] == 1
    assert set(result["full_nodes"]) == set(result["critical_nodes"]) | set(NODES)
    assert set(result["critical_nodes"]).isdisjoint(NODES)
    assert calls[-1][-2:] == list(NODES)
    assert calls[-1][calls[-1].index("-m", calls[-1].index("pytest") + 1) + 1] == "wave2"


@pytest.mark.parametrize("option", ["skipped", "missing", "wrong_inventory"])
def test_missing_or_skipped_coverage_is_incomplete(tmp_path, monkeypatch, option):
    root, manifest = fixture(tmp_path)
    mocked_pytest(monkeypatch, **{option: True})
    out = tmp_path / "out"
    assert wave2.run(root, out, manifest) == 2
    assert json.loads((out / "result.json").read_text())["status"] == "INCOMPLETE"


def test_real_assertion_failure_remains_failure(tmp_path, monkeypatch):
    root, manifest = fixture(tmp_path)
    mocked_pytest(monkeypatch, failed=True)
    out = tmp_path / "out"
    assert wave2.run(root, out, manifest) == 1
    assert json.loads((out / "result.json").read_text())["status"] == "FAIL"


@pytest.mark.parametrize("before", [True, False])
def test_source_change_before_or_after_run_refuses(tmp_path, monkeypatch, before):
    root, manifest = fixture(tmp_path)
    change = lambda: (root / "pytest.ini").write_text("changed")
    calls = mocked_pytest(monkeypatch, mutate=None if before else change)
    if before:
        change()
    out = tmp_path / "out"
    assert wave2.run(root, out, manifest) == 2
    if before:
        assert calls == []


def test_manifest_cannot_omit_runner_or_escape_root(tmp_path):
    root, manifest = fixture(tmp_path)
    data = json.loads(manifest.read_text())
    del data["files"]["scripts/wave2_tests.py"]
    with pytest.raises(ValueError, match="omits"):
        wave2.source_hashes(root, data)
    data = json.loads(manifest.read_text())
    data["files"]["../outside.py"] = "0" * 64
    with pytest.raises(ValueError, match="escapes"):
        wave2.source_hashes(root, data)


def test_junit_same_function_name_in_wrong_module_refuses(tmp_path):
    xml = tmp_path / "forged.xml"
    xml.write_text('<testsuites><testsuite><testcase classname="wrong.Search" name="test_reaches"/>'
                   '<testcase classname="tests.test_cpu" name="test_budget"/></testsuite></testsuites>')
    assert wave2.junit_status(xml, list(NODES))[0] == "INCOMPLETE"


def test_collected_deferred_source_must_be_in_manifest(tmp_path):
    root, manifest = fixture(tmp_path)
    data = json.loads(manifest.read_text())
    del data["files"][NODES[0].split("::", 1)[0]]
    with pytest.raises(ValueError, match="omits"):
        wave2.source_hashes(root, data, list(NODES))


def test_default_marker_partition_keeps_honesty_control(tmp_path):
    import subprocess
    root = tmp_path / "selection"
    root.mkdir()
    root.joinpath("pytest.ini").write_bytes(SCRIPT.parents[1].joinpath("pytest.ini").read_bytes())
    (root / "tests").mkdir()
    root.joinpath("tests/test_selection.py").write_text("""
import pytest
def test_no_search_honesty(): pass
@pytest.mark.wave2
def test_real_search(): pass
@pytest.mark.wave2
def test_physical_contract(): pass
""")
    base = [__import__("sys").executable, "-m", "pytest", "-q", "-n", "0", "--collect-only"]
    ordinary = subprocess.run(base, cwd=root, capture_output=True, text=True)
    deferred = subprocess.run(base + ["-o", "addopts=", "-m", "wave2"], cwd=root,
                              capture_output=True, text=True)
    assert ordinary.returncode == deferred.returncode == 0
    assert "test_no_search_honesty" in ordinary.stdout
    assert "test_real_search" not in ordinary.stdout and "test_physical_contract" not in ordinary.stdout
    assert "test_no_search_honesty" not in deferred.stdout
    assert "test_real_search" in deferred.stdout and "test_physical_contract" in deferred.stdout
