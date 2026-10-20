"""CLI-level invariants for tharkun.sv ([internal ref]).

``tharkun.sv.__main__`` is what BenchExec actually invokes in competition; it, not any
test, is the thing that must keep production behavior. This file pins invariants about
that entry point directly, separate from tests that exercise ``verify``/``replay`` as a
library.
"""
from __future__ import annotations

import inspect
import json

import pytest

from tharkun.sv import __main__ as sv_main
from tharkun.sv.engine import Verdict


def _task(tmp_path):
    program = tmp_path / "program.py"
    program.write_text("assert True\n")
    prop = tmp_path / "property.prp"
    prop.write_text("CHECK( init(main()), ! uncaught(AssertionError))\n")
    return program, prop, [str(program), "--property", str(prop)]


def test_cli_never_sets_cpu_time_only():
    """``verify(..., cpu_time_only=True)`` (engine.py) is a TEST SEAM: it drops the
    wall-clock half of the search budget so a test asserting a fully deterministic
    outcome is not flaky under host CPU contention. Default ``False`` is production
    behavior, and the SV CLI -- what BenchExec actually runs -- must never pass it,
    or it silently becomes a runtime option instead of a seam a test opts into on
    purpose. Source-inspects ``main()`` rather than calling it (calling it would need
    a real task/property file and a subprocess) -- this fails the moment ``main()``'s
    own ``verify(...)`` call site starts passing the keyword, keeping the docstring's
    claim in engine.py honest.
    """
    source = inspect.getsource(sv_main.main)
    assert "cpu_time_only" not in source, (
        "tharkun.sv.__main__.main() must never pass cpu_time_only to verify() -- "
        "it is a test seam, not a runtime option (see engine.py::verify's docstring)"
    )


@pytest.mark.parametrize("kind", ["true", "unknown", "false", "crash"])
@pytest.mark.parametrize("stale", [False, True])
def test_witness_output_belongs_only_to_this_cli_invocation(tmp_path, monkeypatch, kind, stale):
    _, _, args = _task(tmp_path)
    witness = tmp_path / "witness.json"
    if stale:
        witness.write_text('{"old": true}\n')
    fresh = {"format": "sv-py-witness/1", "draws": []}

    def verify(*args, **kwargs):
        assert not witness.exists(), "previous witness must be gone before verification"
        if kind == "crash":
            raise RuntimeError("interrupted verification")
        return Verdict(kind, witness=fresh if kind == "false" else None, reason="control")

    monkeypatch.setattr(sv_main, "verify", verify)
    if kind == "crash":
        with pytest.raises(RuntimeError, match="interrupted verification"):
            sv_main.main([*args, "--witness-out", str(witness)])
    else:
        assert sv_main.main([*args, "--witness-out", str(witness)]) == 0
    if kind == "false":
        assert json.loads(witness.read_text()) == fresh
    else:
        assert not witness.exists()


@pytest.mark.parametrize("collision", ["program", "property", "task", "required"])
def test_witness_output_cannot_replace_a_declared_input(tmp_path, monkeypatch, collision):
    program, prop, _ = _task(tmp_path)
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 1\n")
    task = tmp_path / "task.yml"
    task.write_text("format_version: '2.0'\ninput_files: program.py\nrequired_files: helper.py\n")
    output = {"program": program, "property": prop, "task": task, "required": helper}[collision]
    original = output.read_bytes()
    monkeypatch.setattr(sv_main, "verify", lambda *args, **kwargs: pytest.fail("verification must not start"))
    with pytest.raises(SystemExit) as error:
        sv_main.main([str(task), "--property", str(prop), "--witness-out", str(output)])
    assert error.value.code == 2 and output.read_bytes() == original


def test_witness_output_removal_failure_stops_verification(tmp_path, monkeypatch):
    _, _, args = _task(tmp_path)
    output = tmp_path / "directory"
    output.mkdir()
    monkeypatch.setattr(sv_main, "verify", lambda *args, **kwargs: pytest.fail("verification must not start"))
    with pytest.raises(IsADirectoryError):
        sv_main.main([*args, "--witness-out", str(output)])
    assert output.is_dir()


@pytest.mark.parametrize("options", ['options:\n  language_version: "3.15"\n', 'options:\n  language_version: null\n'])
def test_unsupported_task_removes_stale_output_before_refusal(tmp_path, monkeypatch, capsys, options):
    from tharkun.sv import engine

    program, prop, _ = _task(tmp_path)
    task = tmp_path / "task.yml"
    task.write_text("input_files: program.py\n" + options)
    witness = tmp_path / "witness.json"
    witness.write_text('{"old": true}\n')
    monkeypatch.setattr(engine, "read_source", lambda *a: pytest.fail("must refuse before source analysis"))
    assert sv_main.main([str(task), "--property", str(prop), "--witness-out", str(witness),
                         "--language-version", "3.14"]) == 0
    assert not witness.exists() and "UNKNOWN(unsupported" in capsys.readouterr().out
    original = program.read_bytes()
    with pytest.raises(SystemExit) as error:
        sv_main.main([str(task), "--property", str(prop), "--witness-out", str(program)])
    assert error.value.code == 2 and program.read_bytes() == original


def test_unsupported_task_cannot_skip_output_removal_failure(tmp_path, monkeypatch):
    program, prop, _ = _task(tmp_path)
    task = tmp_path / "task.yml"
    task.write_text('input_files: program.py\noptions:\n  language_version: "3.15"\n')
    output = tmp_path / "directory"
    output.mkdir()
    monkeypatch.setattr(sv_main, "verify", lambda *a, **k: pytest.fail("must not verify"))
    with pytest.raises(IsADirectoryError):
        sv_main.main([str(task), "--property", str(prop), "--witness-out", str(output)])


def test_cli_validation_refuses_before_source_or_checker(tmp_path, monkeypatch, capsys):
    program, prop, args = _task(tmp_path)
    monkeypatch.setattr(sv_main, "read_source", lambda *a: pytest.fail("must not read target"))
    assert sv_main.main([*args, "--language-version", "3.15",
                         "--validate-witness", str(tmp_path / "nonexistent.yml")]) == 1
    assert "UNKNOWN(unsupported language version" in capsys.readouterr().out


@pytest.mark.parametrize("metadata", [None, 'options:\n  language_version: "3.15"\n'])
def test_cli_keeps_raw_declaration_for_verifier(tmp_path, monkeypatch, metadata):
    program, prop, args = _task(tmp_path)
    if metadata is not None:
        task = tmp_path / "task.yml"
        task.write_text("input_files: program.py\n" + metadata)
        args = [str(task), "--property", str(prop)]

    def verify(program, classes, **kwargs):
        assert kwargs["language_version"] == ("3.14" if metadata is None else None)
        assert kwargs["task"] == args[0]
        return Verdict("unknown", reason="control")
    monkeypatch.setattr(sv_main, "verify", verify)
    assert sv_main.main(args) == 0


def test_cli_replay_keeps_unsupported_reason_and_verdict(tmp_path, capsys):
    program, prop, args = _task(tmp_path)
    witness = tmp_path / "witness.json"
    witness.write_text('{"vector": []}')
    assert sv_main.main([*args, "--language-version", "3.15", "--replay", str(witness)]) == 1
    output = capsys.readouterr().out
    assert "unsupported language version" in output and output.splitlines()[-1] == "NOT CONFIRMED"


def test_replay_does_not_remove_its_input_when_witness_out_is_also_given(tmp_path, capsys):
    _, _, args = _task(tmp_path)
    witness = tmp_path / "witness.json"
    witness.write_text('{"vector": []}')
    original = witness.read_bytes()
    assert sv_main.main([*args, "--language-version", "3.15", "--replay", str(witness),
                         "--witness-out", str(witness)]) == 1
    assert witness.read_bytes() == original

def test_quoted_primary_input_is_protected_before_witness_cleanup(tmp_path, monkeypatch):
    program, prop, _ = _task(tmp_path)
    task = tmp_path / "task.yml"
    task.write_text('"input_files": program.py\n')
    original = program.read_bytes()
    monkeypatch.setattr(sv_main, "verify", lambda *a, **k: pytest.fail("must not verify unresolved inputs"))
    with pytest.raises(SystemExit) as error:
        sv_main.main([str(task), "--property", str(prop), "--witness-out", str(program)])
    assert error.value.code == 2 and program.read_bytes() == original


def test_unreadable_task_metadata_preserves_unknown_inputs(tmp_path, monkeypatch, capsys):
    program, prop, _ = _task(tmp_path)
    task = tmp_path / "task.yml"
    task.write_bytes(b"input_files: program.py\n# \xff\n")
    original = program.read_bytes()
    monkeypatch.setattr(sv_main, "verify", lambda *a, **k: pytest.fail("must not verify unreadable metadata"))
    assert sv_main.main([str(task), "--property", str(prop), "--witness-out", str(program)]) == 0
    assert capsys.readouterr().out.startswith("UNKNOWN(unsupported task metadata:")
    assert program.read_bytes() == original

@pytest.mark.parametrize("quoted_field", ["input_files", "required_files"])
def test_partial_task_input_projection_preserves_unknown_files(tmp_path, monkeypatch, quoted_field):
    program, prop, _ = _task(tmp_path)
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 1\n")
    task = tmp_path / "task.yml"
    task.write_text(f'{chr(34) if quoted_field == "input_files" else ""}input_files'
                    f'{chr(34) if quoted_field == "input_files" else ""}: program.py\n'
                    f'{chr(34) if quoted_field == "required_files" else ""}required_files'
                    f'{chr(34) if quoted_field == "required_files" else ""}: helper.py\n')
    output = program if quoted_field == "input_files" else helper
    original = output.read_bytes()
    monkeypatch.setattr(sv_main, "verify", lambda *a, **k: pytest.fail("must not verify unknown input ownership"))
    with pytest.raises(SystemExit) as error:
        sv_main.main([str(task), "--property", str(prop), "--witness-out", str(output)])
    assert error.value.code == 2 and output.read_bytes() == original


@pytest.mark.parametrize("declaration", [
    "input_files: [program.py, missing.py]\nrequired_files: helper.py\n",
    "input_files: ['program.py', 'missing*.py']\nrequired_files: helper.py\n",
    "input_files: program.py\ninput_files: helper.py\n",
])
@pytest.mark.parametrize("output", [None, "program.py", "helper.py"])
def test_unresolved_task_files_refuse_before_analysis_and_cleanup(tmp_path, monkeypatch, capsys, declaration, output):
    program, prop, _ = _task(tmp_path)
    helper = tmp_path / "helper.py"
    helper.write_text("pass\n")
    task = tmp_path / "task.yml"
    task.write_text(declaration)
    originals = {p: p.read_bytes() for p in (program, helper, task, prop)}
    monkeypatch.setattr(sv_main, "verify", lambda *a, **k: pytest.fail("must not analyze unresolved task inputs"))
    args = [str(task), "--property", str(prop)]
    if output is not None:
        args += ["--witness-out", str(tmp_path / output)]
    assert sv_main.main(args) == 0
    assert capsys.readouterr().out.startswith("UNKNOWN(unsupported task metadata:")
    assert all(path.read_bytes() == original for path, original in originals.items())


def test_quoted_primary_and_helper_bind_the_real_program_without_output(tmp_path, capsys):
    program, prop, _ = _task(tmp_path)
    program.write_text("raise IndexError\n")
    prop.write_text("CHECK( init(main()), ! uncaught(IndexError))\n")
    helper = tmp_path / "a_helper.py"
    helper.write_text("pass\n")
    task = tmp_path / "task.yml"
    task.write_text('"input_files": program.py\nrequired_files: a_helper.py\n')
    assert sv_main.main([str(task), "--property", str(prop), "--canary-seconds", "0",
                         "--budget-seconds", "60", "--max-runs", "5", "--jobs", "1",
                         "--proof-dir", str(tmp_path / "proofs")]) == 0
    assert capsys.readouterr().out.rstrip().endswith("FALSE(IndexError)")


# ---------------------------------------------------------------- outer limits (--cpu-limit-seconds, --memory-limit-bytes)

def _tool_info_020(cputime, memory):
    """What the BenchExec tool-info module of 0.2.0 passed for BenchExec's limits (seconds, bytes)."""
    budget = max(5, int(cputime * 0.9)) if cputime else None
    run_memory = max(64, int(memory * 0.6) // 2**20) if memory else None
    return budget, run_memory


@pytest.mark.parametrize("cputime", [None, 1, 5, 6, 60, 899, 900, 901, 1800, 3600, 86400])
@pytest.mark.parametrize("memory", [None, 1, 100 * 10**6, 2**30, 7 * 10**9, 15 * 10**9, 15 * 2**30, 64 * 10**9])
def test_limits_give_the_budgets_the_020_tool_info_module_passed(cputime, memory):
    budget, run_memory = _tool_info_020(cputime, memory)
    assert sv_main.effective_limits(None, None, cputime and float(cputime), memory) == (
        600.0 if budget is None else float(budget), 0 if run_memory is None else run_memory)


def test_limits_at_sv_comp_values():
    assert sv_main.effective_limits(None, None, 900.0, 15 * 10**9) == (810.0, 8583)


def test_explicit_budgets_take_precedence_over_the_limits():
    assert sv_main.effective_limits(120.0, 1000, 900.0, 15 * 10**9) == (120.0, 1000)
    assert sv_main.effective_limits(None, 0, 900.0, 15 * 10**9) == (810.0, 0)  # 0 stays "no cap"
    assert sv_main.effective_limits(2000.0, None, 900.0, None) == (2000.0, 0)  # used as given, not capped


def _captured_limits(tmp_path, monkeypatch, *flags):
    _, _, args = _task(tmp_path)
    seen = {}

    def verify(*args, **kwargs):
        seen.update(budget=kwargs["budget_seconds"], memory=kwargs["run_memory_bytes"])
        return Verdict("true")

    monkeypatch.setattr(sv_main, "verify", verify)
    assert sv_main.main([*args, *flags, "--proof-dir", str(tmp_path / "proofs")]) == 0
    return seen["budget"], seen["memory"]


@pytest.mark.parametrize("flags, budget, memory", [
    ((), 600.0, None),
    (("--budget-seconds", "810", "--run-memory-mb", "8583"), 810.0, 8583 * 2**20),
    (("--cpu-limit-seconds", "900", "--memory-limit-bytes", "15000000000"), 810.0, 8583 * 2**20),
    (("--cpu-limit-seconds", "900"), 810.0, None),
    (("--memory-limit-bytes", "15000000000"), 600.0, 8583 * 2**20),
    (("--cpu-limit-seconds", "900", "--budget-seconds", "100"), 100.0, None),
    (("--memory-limit-bytes", "15000000000", "--run-memory-mb", "0"), 600.0, None),
    (("--budget-seconds", "100", "--cpu-limit-seconds", "900"), 100.0, None),
])
def test_cli_resolves_the_limits(tmp_path, monkeypatch, flags, budget, memory):
    assert _captured_limits(tmp_path, monkeypatch, *flags) == (budget, memory)


@pytest.mark.parametrize("flag, value", [
    ("--cpu-limit-seconds", "0"), ("--cpu-limit-seconds", "-1"), ("--cpu-limit-seconds", "nan"),
    ("--cpu-limit-seconds", "inf"), ("--memory-limit-bytes", "0"), ("--memory-limit-bytes", "1.5e10"),
])
def test_cli_rejects_a_limit_that_is_not_positive(tmp_path, monkeypatch, flag, value):
    _, _, args = _task(tmp_path)
    monkeypatch.setattr(sv_main, "verify", lambda *a, **k: pytest.fail("must not verify"))
    with pytest.raises(SystemExit) as exit_info:
        sv_main.main([*args, flag, value])
    assert exit_info.value.code == 2
