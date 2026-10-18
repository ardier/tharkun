"""A bound method names its function by address: the older function of a re-run ``def``/lambda is the site's summary.

``State.redirect`` rewrites the values that name a demoted object, but ``BoundMethod.func`` is an address in a field, so
the bound method of the OLDER function kept naming the site's most recent function and called that one:
``ms[0]() == 1`` was proven for a loop whose first bound method returns 0.
"""
from __future__ import annotations

import textwrap

import pytest

from tharkun.solve.absint import prove

PROGRAMS = {
    "def_in_loop": """
        class A:
            pass
        a = A()
        ms = []
        for i in range(2):
            def f(self, x=i):
                return x
            A.f = f
            ms.append(a.f)
        assert ms[0]() == 1
    """,
    "lambda_in_loop": """
        class A:
            pass
        a = A()
        ms = []
        for i in range(2):
            A.f = lambda self, x=i: x
            ms.append(a.f)
        assert ms[0]() == 1
    """,
    "stored_in_a_dict_and_called_through_it": """
        class A:
            pass
        a = A()
        d = {}
        for i in range(2):
            A.f = lambda self, x=i: x
            d[i] = a.f
        assert d[0]() == 1
    """,
    "older_one_kept_in_a_dict_under_a_name": """
        class A:
            pass
        a = A()
        d = {}
        for i in range(2):
            A.f = lambda self, x=i: x
            if i == 0:
                d["old"] = a.f
        assert d["old"]() == 1
    """,
    "while_loop": """
        class A:
            pass
        a = A()
        ms = []
        i = 0
        while i < 2:
            def f(self, x=i):
                return x
            A.f = f
            ms.append(a.f)
            i += 1
        assert ms[0]() == 1
    """,
}


def _native(source: str) -> str:
    try:
        exec(compile(source, "<native>", "exec"), {"__name__": "__main__"})
    except Exception as exc:  # noqa: BLE001 -- how the program ends is the point
        return type(exc).__name__
    return ""


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_a_bound_method_of_an_older_function_is_not_the_newer_function(name):
    source = textwrap.dedent(PROGRAMS[name]).strip() + "\n"
    assert _native(source) == "AssertionError"
    result = prove(source, ("AssertionError",), time_budget=20.0)
    assert result.verdict != "TRUE", result.explain()


def test_a_bound_method_of_a_function_made_once_is_still_proven():
    source = textwrap.dedent("""
        class A:
            def f(self, x=3):
                return x
        a = A()
        ms = []
        for i in range(2):
            ms.append(a.f)
        assert ms[0]() == 3
    """).strip() + "\n"
    assert _native(source) == ""
    result = prove(source, ("AssertionError",), time_budget=20.0)
    assert result.verdict == "TRUE", result.explain()
