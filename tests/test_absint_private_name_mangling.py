"""Class-scope private identifiers use CPython's name-mangling rule."""

from __future__ import annotations

import sys
import types
import ast

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.absint.interp import _class_has_private_name_mangling
from tharkun.sv import verify


GUARDED = {
    "name": """\
import _sv_verifier
__x = 1
_C__x = 's'
class C:
    def m(self):
        return __x + 1
n = _sv_verifier.nondet_int()
m = _sv_verifier.nondet_int()
if n * 2654435761 % 4294967291 == 987654 and m == n + 77:
    r = C().m()
""",
    "global": """\
import _sv_verifier
__g = 1
_C__g = 's'
class C:
    def m(self):
        global __g
        return __g + 1
n = _sv_verifier.nondet_int()
m = _sv_verifier.nondet_int()
if n * 2654435761 % 4294967291 == 987654 and m == n + 77:
    r = C().m()
""",
    "store": """\
import _sv_verifier
class C:
    def m(self):
        __t = 's'
        return _C__t + 1
n = _sv_verifier.nondet_int()
m = _sv_verifier.nondet_int()
if n * 2654435761 % 4294967291 == 987654 and m == n + 77:
    r = C().m()
""",
}


def _native_exception(source: str, draws: tuple[int, ...]) -> str | None:
    fake = types.ModuleType("_sv_verifier")
    values = iter(draws)
    fake.nondet_int = lambda: next(values)
    previous = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = fake
    try:
        exec(compile(source, "<private-mangle-native>", "exec"), {"__name__": "__main__"})
    except Exception as exc:  # noqa: BLE001 - capture target exception class
        return type(exc).__name__
    finally:
        if previous is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = previous
    return None


@pytest.mark.parametrize("source", list(GUARDED.values()))
def test_guarded_private_name_repros_are_native_type_errors_and_never_prove(source):
    assert _native_exception(source, (2577420508, 2577420585)) == "TypeError"
    result = prove(source, ("TypeError",), filename="<guarded-private-name>")
    assert result.verdict == "UNKNOWN", result.explain()
    assert result.reason.startswith("unsupported:") and "private name mangling" in result.reason


@pytest.mark.parametrize("source", list(GUARDED.values()))
def test_public_canary_zero_does_not_promote_guarded_private_name_to_true(tmp_path, source):
    path = tmp_path / "private-name.py"
    path.write_text(source)
    result = verify(str(path), ("TypeError",), python=sys.executable,
                    budget_seconds=60, run_timeout=30, max_runs=12, jobs=1,
                    prove=True, canary_seconds=0, early_stop_seconds=0,
                    witness_dir=tmp_path / "witness")
    assert result.verdict != "true", (result.component, result.reason)


PRIVATE_CLASS_SOURCES = (
    "class C:\n    __value = 1\n",
    "class C:\n    def f(self, __value):\n        return __value\n",
    "class C:\n    def f(self):\n        global __value\n",
    "class C:\n    def outer(self):\n        __value = 1\n        def inner():\n            nonlocal __value\n            return __value\n",
    "class C:\n    import math as __module\n",
    "class C:\n    import __private_module\n",
    "class C:\n    def __method(self):\n        return 1\n",
    "class C:\n    class __Nested:\n        pass\n",
    "class C:\n    def f(self):\n        return dict(__value=1)\n",
    "class C:\n    def f(self):\n        try:\n            raise ValueError()\n        except ValueError as __error:\n            return __error\n",
    "class C:\n    def f(self, value):\n        match value:\n            case __capture:\n                return __capture\n",
    "class Outer:\n    class Inner:\n        __value = 1\n",
)


@pytest.mark.parametrize("source", PRIVATE_CLASS_SOURCES)
def test_all_private_identifier_binding_forms_are_refused_before_class_execution(source):
    result = prove(source, ("TypeError",), filename="<private-name-shape>")
    assert result.verdict == "UNKNOWN", result.explain()
    assert result.reason.startswith("unsupported:") and "private name mangling" in result.reason


def test_only_non_attribute_names_that_cpython_mangles_are_refused():
    no_mangling = (
        "class ___:\n    __value = 1\n",
        "class C:\n    __value__ = 1\n",
        "class C:\n    def f(self):\n        return self.__attribute\n",
    )
    for source in no_mangling:
        tree = ast.parse(source)
        cls = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        assert not _class_has_private_name_mangling(cls), source
    assert prove("class ___:\n    __value = 1\n", ("TypeError",)).verdict == "TRUE"
    assert prove("class C:\n    __value__ = 1\n", ("TypeError",)).verdict == "TRUE"


def test_module_level_dunder_control_remains_proven_safe():
    source = """\
import _sv_verifier
__x = 1
class C:
    def m(self):
        return __x__ + 1
__x__ = 2
n = _sv_verifier.nondet_int()
if n == 5:
    r = C().m()
"""
    assert _native_exception(source, (5,)) is None
    result = prove(source, ("TypeError",), filename="<unmangled-dunder-safe>")
    assert result.verdict == "TRUE", result.explain()
