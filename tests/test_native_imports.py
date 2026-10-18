"""Native controls for bootstrap cache observations and task stdlib shadows ([internal ref])."""
from __future__ import annotations

import ast
import os
import subprocess
import sys

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.census import site_census
from tharkun.solve.determinism import DET_MODULES, determinism_census
from tharkun.solve.extern import native_import_reason
from tharkun.solve.pyresolve import Resolver, Top
from tharkun.sv import nondet_harness, product_proof, verify
from tharkun.sv.engine import replay, run_once
from tharkun.sv.prove import run_probe

AR = ("AssertionError",)


def program(tmp_path, source):
    path = tmp_path / "main.py"
    path.write_text(source)
    return str(path)


def native(path):
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONSTARTUP"}}
    return subprocess.run([sys.executable, path], env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("source", [
    "import sys\nassert 'json' not in sys.modules\n",
    "from sys import modules as cache\nassert 'json' not in cache\n",
    "import sys as s\nassert 'json' not in getattr(s, 'modules')\n",
])
def test_bootstrap_cache_observations_are_unknown_and_never_witnesses(tmp_path, monkeypatch, source):
    path = program(tmp_path, source)
    assert native(path).returncode == 0
    reason = native_import_reason(source, path)
    assert "sys.modules" in reason
    assert prove(source, AR, filename=path).verdict == "UNKNOWN"
    assert not site_census(source, AR, path).no_site
    assert not determinism_census(source, path).ok
    assert product_proof.prove(source, AR, program=path) is None
    out = run_once(sys.executable, path, [], timeout=30)
    assert "import_state_unmodelled" in out.flags and out.violated(AR) is None
    ok, out = replay(path, {"vector": []}, AR, timeout=30)
    assert not ok and "import_state_unmodelled" in out.flags
    probe = run_probe(sys.executable, path, [], [], [], timeout=30)
    assert "import_state_unmodelled" in probe["flags"]
    assert "import_state_unmodelled" in nondet_harness.drive(path, []).flags
    # Unsupported native semantics is checked before any solving/search stage runs.
    from tharkun.sv import stages
    monkeypatch.setattr(stages, "select", lambda *a, **k: pytest.fail("unsupported task entered stages"))
    for do_prove in (False, True):
        verdict = verify(path, AR, budget_seconds=5, jobs=1, prove=do_prove, canary_seconds=0)
        assert verdict.verdict == "unknown" and "sys.modules" in verdict.reason
        assert verdict.runs == 0


@pytest.mark.parametrize("package", [False, True])
@pytest.mark.parametrize("task_files", [False, True])
def test_task_stdlib_shadows_are_never_certified(tmp_path, package, task_files):
    source = "import json\nassert True\n"
    path = program(tmp_path, source)
    shadow = tmp_path / "json.py"
    if package:
        shadow = tmp_path / "json" / "__init__.py"
        shadow.parent.mkdir()
    shadow.write_text("raise AssertionError('native sibling import')\n")
    run = native(path)
    assert run.returncode == 1 and "native sibling import" in run.stderr
    reason = native_import_reason(source, path)
    assert "stdlib shadow 'json'" in reason
    assert prove(source, AR, filename=path).verdict == "UNKNOWN"
    with pytest.raises(Top, match="stdlib shadow"):
        Resolver(ast.parse(source), DET_MODULES, filename=path)
    task = tmp_path / "main.yml"
    task.write_text(f"format_version: '2.0'\ninput_files:\n  - main.py\n  - {shadow.relative_to(tmp_path)}\n")
    verdict = verify(path, AR, task=str(task) if task_files else None,
                     budget_seconds=5, jobs=1, canary_seconds=0)
    assert verdict.verdict == "unknown" and "stdlib shadow 'json'" in verdict.reason
    ok, out = replay(path, {"vector": []}, AR, timeout=30, task=str(task) if task_files else None)
    assert not ok and "import_state_unmodelled" in out.flags


def test_no_shadow_import_and_plain_violation_keep_native_verdicts(tmp_path):
    source = "import json\nassert True\n"
    path = program(tmp_path, source)
    assert native(path).returncode == 0 and native_import_reason(source, path) is None
    verdict = verify(path, AR, budget_seconds=5, jobs=1, canary_seconds=0)
    assert verdict.verdict == "true"
    source = "assert False\n"
    path = program(tmp_path, source)
    assert native(path).returncode == 1 and native_import_reason(source, path) is None
    verdict = verify(path, AR, budget_seconds=5, jobs=1, canary_seconds=0)
    assert verdict.verdict == "false" and replay(path, verdict.witness, AR, timeout=30)[0]


@pytest.mark.parametrize("source", ["import local\n", "from pkg import helper\n"])
def test_cache_observations_inside_imported_code_are_refused(tmp_path, source):
    path = program(tmp_path, source)
    (tmp_path / "local.py").write_text("import sys\nassert 'json' not in sys.modules\n")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from . import helper\n")
    (pkg / "helper.py").write_text("import sys\nassert 'json' not in sys.modules\n")
    assert native(path).returncode == 0
    assert "sys.modules" in native_import_reason(source, path)
    assert "import_state_unmodelled" in run_once(sys.executable, path, [], timeout=30).flags
    assert verify(path, AR, budget_seconds=5, jobs=1, canary_seconds=0).verdict == "unknown"


def test_explicit_root_shadow_is_refused_and_unused_shadow_is_a_control(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "json.py").write_text("raise AssertionError('explicit root shadow')\n")
    source = f"import sys\nsys.path.insert(0, {str(lib)!r})\nimport json\nassert True\n"
    path = program(tmp_path, source)
    assert native(path).returncode == 1
    assert "stdlib shadow 'json'" in native_import_reason(source, path)
    source = "import json\nassert True\n"
    assert native_import_reason(source, path) is None


def test_a_safe_local_package_and_cycle_do_not_block_native_execution(tmp_path):
    path = program(tmp_path, "import pkg\nassert pkg.VALUE == 7\n")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("VALUE = 7\nfrom . import helper\n")
    (pkg / "helper.py").write_text("import pkg\n")
    assert native(path).returncode == 0
    assert native_import_reason(open(path).read(), path) is None
    out = run_once(sys.executable, path, [], timeout=30)
    assert out.status == "clean" and "import_state_unmodelled" not in out.flags


def test_dynamic_import_cache_is_explicitly_unsupported():
    assert "dynamic code/import" in native_import_reason("__import__('json')\n")


@pytest.mark.parametrize("source", [
    "exec(\"import sys\\nassert 'json' not in sys.modules\\n\")\n",
    "from importlib import import_module as load\nload('json')\n",
    "import importlib\nload = importlib.import_module\nload('json')\n",
])
def test_dynamic_code_and_import_aliases_cannot_escape_the_refusal(tmp_path, source):
    path = program(tmp_path, source)
    assert native_import_reason(source, path) is not None
    out = run_once(sys.executable, path, [], timeout=30)
    assert "import_state_unmodelled" in out.flags and out.violated(AR) is None
    assert verify(path, AR, budget_seconds=5, jobs=1, canary_seconds=0).verdict == "unknown"


def test_relative_program_paths_check_sibling_shadows(tmp_path, monkeypatch):
    source = "import json.decoder\nassert True\n"
    program(tmp_path, source)
    (tmp_path / "json.py").write_text("raise AssertionError('dotted sibling shadow')\n")
    monkeypatch.chdir(tmp_path)
    assert native("main.py").returncode == 1
    assert "stdlib shadow 'json'" in native_import_reason(source, "main.py")
    assert verify("main.py", AR, budget_seconds=5, jobs=1, canary_seconds=0).verdict == "unknown"


def test_later_roots_cannot_replace_the_context_of_an_earlier_local_import(tmp_path):
    later = tmp_path / "later"
    later.mkdir()
    (tmp_path / "helper.py").write_text("import other\n")
    (tmp_path / "other.py").write_text("import sys\nassert 'json' not in sys.modules\n")
    (later / "other.py").write_text("VALUE = 7\n")
    source = f"import helper\nimport sys\nsys.path.insert(0, {str(later)!r})\n"
    path = program(tmp_path, source)
    assert native(path).returncode == 0
    assert native_import_reason(source, path) is not None
    assert "import_state_unmodelled" in run_once(sys.executable, path, [], timeout=30).flags


def test_a_package_initializer_that_changes_global_roots_is_refused(tmp_path):
    later = tmp_path / "later"
    later.mkdir()
    (later / "json.py").write_text("MARKER = True\n")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(f"import sys\nsys.path.insert(0, {str(later)!r})\n")
    (pkg / "helper.py").write_text("import json\nassert hasattr(json, 'MARKER')\n")
    source = "import pkg.helper\n"
    path = program(tmp_path, source)
    assert native(path).returncode == 0
    assert native_import_reason(source, path) is not None
    out = run_once(sys.executable, path, [], timeout=30)
    assert "import_state_unmodelled" in out.flags and out.violated(AR) is None


def test_an_imported_package_directory_is_not_an_absolute_import_root(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from . import helper\n")
    (pkg / "helper.py").write_text("import json\nassert hasattr(json, 'loads')\n")
    (pkg / "json.py").write_text("raise AssertionError('unused relative shadow')\n")
    source = "import pkg\n"
    path = program(tmp_path, source)
    assert native(path).returncode == 0 and native_import_reason(source, path) is None
    out = run_once(sys.executable, path, [], timeout=30)
    assert out.status == "clean" and "import_state_unmodelled" not in out.flags


@pytest.mark.parametrize("mutation", [
    "sys.path.append({root!r})",
    "sys.path = [{root!r}] + sys.path",
    "p = sys.path; p.insert(0, {root!r})",
    "getattr(sys, 'path').insert(0, {root!r})",
])
def test_unmodelled_path_mutations_cannot_bypass_shadow_detection(tmp_path, mutation):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "json.py").write_text("MARKER = True\n")
    source = "import sys\n" + mutation.format(root=str(lib)) + "\nimport json\n"
    path = program(tmp_path, source)
    assert native_import_reason(source, path) is not None
    assert "import_state_unmodelled" in run_once(sys.executable, path, [], timeout=30).flags


@pytest.mark.parametrize("source", [
    "import sys as s\ncache = s.__dict__['mod' + 'ules']\nassert 'json' not in cache\n",
    "import sys\nfrom operator import attrgetter\ncache = attrgetter('mod' + 'ules')(sys)\nassert 'json' not in cache\n",
    "import sys\ncache = sys.__getattribute__('mod' + 'ules')\nassert 'json' not in cache\n",
])
def test_reflected_cache_observations_are_never_bootstrap_witnesses(tmp_path, source):
    path = program(tmp_path, source)
    assert native(path).returncode == 0
    assert native_import_reason(source, path) is not None
    out = run_once(sys.executable, path, [], timeout=30)
    assert "import_state_unmodelled" in out.flags and out.violated(AR) is None


@pytest.mark.parametrize("lookup", [
    "sys.modules['_sv_verifier'].nondet_int()",
    "globals()['_sv_verifier'].nondet_int()",
    "__import__('_sv_verifier').nondet_int()",
    "getattr(_sv_verifier, 'nondet_int')()",
])
def test_known_helper_lookups_do_not_hide_other_cache_observations(tmp_path, lookup):
    (tmp_path / "_sv_verifier.py").write_text("def nondet_int(): return 0\n")
    prefix = "import sys\nimport _sv_verifier\n" + lookup + "\n"
    path = program(tmp_path, prefix)
    assert native(path).returncode == 0 and native_import_reason(prefix, path) is None
    source = prefix + "assert 'json' not in sys.modules\n"
    path = program(tmp_path, source)
    assert native(path).returncode == 0
    assert "sys.modules" in native_import_reason(source, path)
    out = run_once(sys.executable, path, [], timeout=30)
    assert "import_state_unmodelled" in out.flags and out.violated(AR) is None


@pytest.mark.parametrize("source,reason", [
    ("import sys\nsys.monitoring.get_tool(4)\n", "sys.monitoring"),
    ("import sys as s\nm = s.monitoring\n", "sys.monitoring"),
    ("from sys import monitoring as m\n", "sys.monitoring"),
    ("import sys.monitoring\n", "sys.monitoring"),
    ("import sys\ngetattr(sys, 'monitoring')\n", "reflection"),
])
def test_monitoring_access_is_refused_before_search(tmp_path, monkeypatch, source, reason):
    path = program(tmp_path, source)
    assert reason in native_import_reason(source, path)
    from tharkun.sv import stages
    monkeypatch.setattr(stages, "select", lambda *a, **k: pytest.fail("monitoring task entered search"))
    verdict = verify(path, AR, budget_seconds=60, run_timeout=30, jobs=1, canary_seconds=0)
    assert verdict.verdict == "unknown" and reason in verdict.reason
    assert verdict.runs == 0


@pytest.mark.parametrize("package", [False, True])
def test_monitoring_in_declared_imported_siblings_is_refused(tmp_path, package):
    source = "import pkg.helper\n" if package else "import helper\n"
    path = program(tmp_path, source)
    root = tmp_path / "pkg" if package else tmp_path
    if package:
        root.mkdir()
        (root / "__init__.py").write_text("")
    helper = root / "helper.py"
    helper.write_text("from sys import monitoring as m\nassert False\n")
    task = tmp_path / "main.yml"
    task.write_text("format_version: '2.0'\ninput_files:\n  - main.py\n  - " +
                    str(helper.relative_to(tmp_path)) + "\n")
    assert "sys.monitoring" in native_import_reason(source, path)
    from tharkun.solve.environment import TaskEnvironment
    out = run_once(sys.executable, path, [], timeout=30, files=TaskEnvironment.from_task(str(task)).files)
    assert "import_state_unmodelled" in out.flags and out.violated(AR) is None
    verdict = verify(path, AR, task=str(task), budget_seconds=60, run_timeout=30, jobs=1, canary_seconds=0)
    assert verdict.verdict == "unknown" and verdict.runs == 0


def test_unrelated_sys_usage_keeps_native_admission():
    assert native_import_reason("import sys\nassert sys.getrecursionlimit() > 0\n") is None
