"""A class statement that runs more than once names two classes of one allocation site.

``Analyzer.alloc`` turns the older class of a site into the site's summary and redirects the VALUES that name it, but an
instance names its class by address (``InstObj.cls``) and a subclass its base (``ClassObj.bases``), and those were not
redirected: after the second run an instance of the FIRST class read as an instance of the second.  ``isinstance(objs[0], C)``
was proven True for a program in which it is False.  A class statement whose site already holds a class is refused.
"""
from __future__ import annotations

import textwrap

import pytest

from tharkun.solve.absint import prove

PROGRAMS = {
    "isinstance_of_first": """
        objs = []
        for i in range(2):
            class C:
                pass
            objs.append(C())
        assert isinstance(objs[0], C)
    """,
    "isinstance_of_first_while": """
        objs = []
        i = 0
        while i < 2:
            class C:
                pass
            objs.append(C())
            i += 1
        assert isinstance(objs[0], C)
    """,
    "subclass_of_first": """
        subs = []
        for i in range(2):
            class Base:
                pass
            class Sub(Base):
                pass
            subs.append(Sub())
        assert isinstance(subs[0], Base)
    """,
}


def _native(source: str) -> str:
    try:
        exec(compile(source, "<native>", "exec"), {"__name__": "__main__"})
    except Exception as exc:  # noqa: BLE001 -- how the program ends is the point
        return type(exc).__name__
    return ""


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_a_class_made_in_a_loop_never_proves_isinstance_of_an_older_instance(name):
    source = textwrap.dedent(PROGRAMS[name]).strip() + "\n"
    assert _native(source) == "AssertionError"
    result = prove(source, ("AssertionError",), time_budget=20.0)
    assert result.verdict != "TRUE", result.explain()


def test_a_class_made_once_is_still_proven():
    source = textwrap.dedent("""
        class C:
            pass
        objs = []
        for i in range(2):
            objs.append(C())
        assert isinstance(objs[0], C)
    """).strip() + "\n"
    assert _native(source) == ""
    result = prove(source, ("AssertionError",), time_budget=20.0)
    assert result.verdict == "TRUE", result.explain()
