"""Generic definitions and PEP 695 aliases are outside the absint model."""

from __future__ import annotations

import ast
import sys
import types

import pytest

from tharkun.solve.absint import prove
from tharkun.sv import verify

pytestmark = pytest.mark.skipif(
    not hasattr(ast, "TypeAlias"), reason="PEP 695 syntax requires Python 3.12 or newer"
)


GUARDED_FUNCTION = """\
import _sv_verifier
def f[int](v):
    return int(v)
n = _sv_verifier.nondet_int()
m = _sv_verifier.nondet_int()
if n * 2654435761 % 4294967291 == 987654 and m == n + 77:
    a = f(2)
"""

GUARDED_CLASS = """\
import _sv_verifier
class Box[float]:
    def get(self, v):
        return float(v)
n = _sv_verifier.nondet_int()
m = _sv_verifier.nondet_int()
if n * 2654435761 % 4294967291 == 987654 and m == n + 77:
    b = Box().get(1)
"""


def _native_exception(source: str, draws: tuple[int, ...]) -> str | None:
    fake = types.ModuleType("_sv_verifier")
    values = iter(draws)
    fake.nondet_int = lambda: next(values)
    previous = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = fake
    try:
        exec(compile(source, "<generic-native-control>", "exec"), {"__name__": "__main__"})
    except Exception as exc:  # noqa: BLE001 - capture the target exception class
        return type(exc).__name__
    finally:
        if previous is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = previous
    return None


@pytest.mark.parametrize("source", [GUARDED_FUNCTION, GUARDED_CLASS])
def test_generic_function_and_class_repros_are_native_type_errors_and_refused(source):
    assert _native_exception(source, (2577420508, 2577420585)) == "TypeError"
    result = prove(source, ("TypeError",), filename="<generic-type-parameter>")
    assert result.verdict == "UNKNOWN", result.explain()
    assert result.reason.startswith("unsupported:") and "type parameters" in result.reason


@pytest.mark.parametrize("source", [GUARDED_FUNCTION, GUARDED_CLASS])
def test_public_canary_zero_does_not_promote_generic_definition_to_true(tmp_path, source):
    path = tmp_path / "generic.py"
    path.write_text(source)
    result = verify(str(path), ("TypeError",), python=sys.executable,
                    budget_seconds=60, run_timeout=30, max_runs=12, jobs=1,
                    prove=True, canary_seconds=0, early_stop_seconds=0,
                    witness_dir=tmp_path / "witness")
    assert result.verdict != "true", (result.component, result.reason)


def test_async_generic_definition_is_explicitly_refused():
    source = "async def f[T](x):\n    return x\n"
    result = prove(source, ("TypeError",), filename="<async-generic>")
    assert result.verdict == "UNKNOWN"
    assert "type parameters" in result.reason


def test_type_alias_statement_is_explicitly_refused():
    source = "type Alias = int\nx = 1 // 0\n"
    result = prove(source, ("ZeroDivisionError",), filename="<type-alias>")
    assert result.verdict == "UNKNOWN"
    assert result.reason.startswith("unsupported:") and "type alias" in result.reason
