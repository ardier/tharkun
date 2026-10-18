"""Narrow check_type support for inline builtin int/str union hints."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES

ALL = tuple(ALL_PROPERTY_CLASSES)


def _prove(source, exceptions=ALL):
    return prove(source, exceptions, time_budget=60.0)


def _peer_runtime_check_type(source):
    """Run the pinned peer runtime model in a fresh interpreter."""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-B", "-c",
         "import sys\nfrom tharkun.sv._child import _make_sv_verifier\n"
         "sv = _make_sv_verifier()\nsys.modules['_sv_verifier'] = sv\n" + source],
        capture_output=True, text=True, timeout=30, env=env, check=False,
    )
    if completed.returncode == 0:
        return ""
    lines = completed.stderr.strip().splitlines()
    assert lines, completed
    return lines[-1].split(":", 1)[0]


@pytest.mark.parametrize("source", [
    "import _sv_verifier as sv\nsv.check_type(7, int | str)\n",
    "import _sv_verifier as sv\nsv.check_type('text', int | str)\n",
    "import _sv_verifier as sv\nsv.check_type(True, int | str)\n",
    "from _sv_verifier import check_type as check\ncheck('text', str | int)\n",
    "import _sv_verifier as sv\nsv.check_type(value=True, hint=str | int)\n",
])
def test_inline_int_str_hint_accepts_covered_builtin_values(source):
    assert _peer_runtime_check_type(source) == ""
    result = _prove(source)
    assert result.verdict == "TRUE", result.explain()


def test_nondeterministic_int_or_str_value_is_covered_in_both_arms():
    source = (
        "import _sv_verifier as sv\n"
        "value = 7 if sv.nondet_bool() else 'text'\n"
        "sv.check_type(value, int | str)\n"
    )
    result = _prove(source)
    assert result.verdict == "TRUE", result.explain()


@pytest.mark.parametrize("alternative", [
    "1.5", "None", "Custom()",
])
def test_uncovered_value_alternatives_never_prove_typeerror_absent(alternative):
    source = (
        "import _sv_verifier as sv\n"
        "class Custom: pass\n"
        f"value = 7 if sv.nondet_bool() else {alternative}\n"
        "sv.check_type(value, int | str)\n"
    )
    result = _prove(source, ("TypeError",))
    assert result.verdict != "TRUE", result.explain()
    later_assert = _prove(source + "raise AssertionError('normal branch survives')\n",
                          ("AssertionError",))
    assert later_assert.verdict != "TRUE", later_assert.explain()


@pytest.mark.parametrize("exception,tail", [
    ("AssertionError", "raise AssertionError('after check')"),
    ("ZeroDivisionError", "value = 1 // 0"),
])
def test_later_property_exception_remains_visible_after_matching_union(exception, tail):
    source = (
        "import _sv_verifier as sv\n"
        "value = 7 if sv.nondet_bool() else 'text'\n"
        "sv.check_type(value, int | str)\n"
        f"{tail}\n"
    )
    result = _prove(source, (exception,))
    assert result.verdict != "TRUE", result.explain()


@pytest.mark.parametrize("source", [
    "import _sv_verifier as sv\nclass Meta(type):\n    def __or__(cls, other): raise AssertionError('or hook')\nclass int(metaclass=Meta): pass\nsv.check_type(1, int | str)\n",
    "import _sv_verifier as sv\nclass Meta(type):\n    def __instancecheck__(cls, value): raise AssertionError('instance hook')\nclass int(metaclass=Meta): pass\nsv.check_type(1, int | str)\n",
    "import _sv_verifier as sv\nclass Custom:\n    def __repr__(self): raise AssertionError('repr hook')\nsv.check_type(Custom(), int | str)\n",
    "import _sv_verifier as sv\nsv.check_type([1], list[int] | str)\n",
    "import _sv_verifier as sv\nclass Scope:\n    int = float\n    sv.check_type(1, int | str)\n",
    "import _sv_verifier as sv\nhint = int | str\nsv.check_type(1, hint)\n",
    "import _sv_verifier as sv\nhint = int | str\nprint(hint)\n",
])
def test_shadowed_effectful_or_non_inline_union_hints_remain_refused(source):
    result = _prove(source, ("AssertionError", "TypeError"))
    assert result.verdict != "TRUE", result.explain()


def test_class_scope_user_check_type_consumer_does_not_admit_union():
    source = (
        "from _sv_verifier import check_type\n"
        "def other(value, hint): return None\n"
        "class Scope:\n"
        "    check_type = other\n"
        "    check_type(1, int | str)\n"
    )
    result = _prove(source)
    assert result.verdict == "UNKNOWN", result.explain()


def test_class_scope_joined_shadow_rejects_uncertain_int_name():
    source = (
        "import _sv_verifier as sv\n"
        "class Scope:\n"
        "    int = int if sv.nondet_bool() else 1\n"
        "    sv.check_type(1, int | str)\n"
    )
    type_error = _prove(source, ("TypeError",))
    assert type_error.verdict != "TRUE", type_error.explain()
    later_assertion = _prove(source + "raise AssertionError('normal branch survives')\n",
                             ("AssertionError",))
    assert later_assertion.verdict != "TRUE", later_assertion.explain()


def test_peer_runtime_checker_confirms_match_mismatch_and_callback_controls():
    assert _peer_runtime_check_type("sv.check_type(True, int | str)\n") == ""
    assert _peer_runtime_check_type("sv.check_type('s', str | int)\n") == ""
    assert _peer_runtime_check_type("sv.check_type(1.5, int | str)\n") == "TypeError"
    assert _peer_runtime_check_type(
        "class Meta(type):\n"
        "    def __or__(cls, other): raise AssertionError('or hook')\n"
        "class int(metaclass=Meta): pass\n"
        "sv.check_type(1, int | str)\n"
    ) == "AssertionError"
    assert _peer_runtime_check_type(
        "class Meta(type):\n"
        "    def __instancecheck__(cls, value): raise AssertionError('instance hook')\n"
        "class int(metaclass=Meta): pass\n"
        "sv.check_type(1, int | str)\n"
    ) == "AssertionError"
    assert _peer_runtime_check_type(
        "class Scope:\n"
        "    int = float\n"
        "    sv.check_type(1, int | str)\n"
    ) == "TypeError"
    assert _peer_runtime_check_type(
        "class Custom:\n"
        "    def __repr__(self): raise AssertionError('repr hook')\n"
        "sv.check_type(Custom(), int | str)\n"
    ) == "AssertionError"
