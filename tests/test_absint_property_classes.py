"""Unsupported property classes must not become proofs with zero relevant checks."""
import builtins
import hashlib
import subprocess
import sys

import pytest

from tharkun.solve.absint import PROPERTY_CLASSES, prove

CASES = [
    ("missing_name", "missing_name\n", ("NameError",), "NameError"),
    ("parameter_other_scope", "def f(n):\n    return n\nf(3)\ni = 0\nwhile i < n:\n    i += 1\n", ("NameError",), "NameError"),
    ("unbound_local", "def f():\n    x = x + 1\nf()\n", ("UnboundLocalError",), "UnboundLocalError"),
    ("exception_superclass", "assert False\n", ("Exception",), "AssertionError"),
    ("baseexception_superclass", "assert False\n", ("BaseException",), "AssertionError"),
    ("value_error", "int('invalid')\n", ("ValueError",), "ValueError"),
    ("overflow_error", "float(10 ** 1000)\n", ("OverflowError",), "OverflowError"),
    ("mixed_name", "missing_name\n", ("AssertionError", "NameError"), "NameError"),
    ("mixed_broad", "assert False\n", ("ZeroDivisionError", "Exception"), "AssertionError"),
]


def _native_raises(tmp_path, name, source, classes, raised):
    path = tmp_path / (name + ".py")
    path.write_text(source)
    native = subprocess.run([sys.executable, str(path)], text=True, capture_output=True, timeout=30)
    assert native.returncode != 0
    assert native.stderr.splitlines()[-1].split(":", 1)[0] == raised
    assert any(issubclass(getattr(builtins, raised), getattr(builtins, cls)) for cls in classes)
    return path


@pytest.mark.parametrize("name,source,classes,raised", CASES)
def test_unsupported_property_classes_refuse_before_analysis(tmp_path, monkeypatch, name, source, classes, raised):
    _native_raises(tmp_path, name, source, classes, raised)
    import tharkun.solve.slicing as slicing

    monkeypatch.setattr(slicing, "analysis_slice", lambda *a, **k: pytest.fail("unsupported property reached slicing"))
    result = prove(source, classes)
    assert result.verdict == "UNKNOWN"
    assert "unsupported property classes" in result.reason
    assert not result.complete and result.records == [] and result.sites == []
    assert result.source_sha == hashlib.sha256(source.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("name,source,classes,raised", CASES)
def test_public_verifier_does_not_promote_unsupported_property_to_true(tmp_path, name, source, classes, raised):
    from tharkun.sv import verify

    path = _native_raises(tmp_path, name, source, classes, raised)
    verdict = verify(str(path), classes, python=sys.executable, budget_seconds=60, run_timeout=30,
                     max_runs=12, jobs=1, prove=True, canary_seconds=0, early_stop_seconds=0,
                     witness_dir=tmp_path / "witness")
    assert verdict.verdict == "false", (verdict.component, verdict.reason)


@pytest.mark.parametrize("cls", [cls for classes in PROPERTY_CLASSES.values() for cls in classes])
def test_audited_property_classes_keep_their_safe_proof(cls):
    result = prove("x = 1\n", (cls,))
    assert result.verdict == "TRUE", result.reason
    assert result.complete
