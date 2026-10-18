"""One regression test per site changed by [internal ref]: an unmodelled kind is a NAMED REFUSAL (UNKNOWN), never an infeasible path.

The sites, each of which treated a kind it does not model as "bad" (a recorded, unproven ``TypeError``) and, when no alternative
produced a value, ended the path -- so everything after it was unchecked and a program that raises there was proven TRUE:

| site (``solve/absint/interp.py::``)        | kind                             | now |
|--------------------------------------------|----------------------------------|-----|
| ``Analyzer.unpack``                        | array                            | exact: the elements of the array |
| ``Analyzer.m_random_choice``               | array                            | exact: the length check (``IndexError``) and the elements |
| ``Analyzer.subscript_load`` (str)          | a non-int index or slice bound   | a ``TypeError`` check, and no value comes of it |
| ``Analyzer.subscript_load`` (dict)         | a slice (hashable since 3.12)    | a ``KeyError`` check (never a ``TypeError``) |
| ``Analyzer._seq_binop`` / ``binop``        | float * sequence, sequence * float | a ``TypeError`` check |
| ``Analyzer.check_in``                      | a static object ``in`` a str     | a ``TypeError`` check |
| ``Analyzer.attr_store``                    | list, tuple, dict, range, array, iterator, pattern, bound method | an ``AttributeError`` check; a StringIO or file handle (it has a ``__dict__``): refused |
| ``Analyzer.call_static``                   | an exception INSTANCE            | not callable: a ``TypeError`` check (it was the class's own value) |
| ``Analyzer.static_attr``                   | a name no module table models    | an ``AttributeError`` check (``math.foo``) |
| ``Analyzer.gc``                            | a dict view's dict               | kept alive (the view died with a ``KeyError`` crash) |
| ``Analyzer.check_iterable`` / ``unpack``   | StringIO                         | refused (its lines are not modelled) |

Every program here is small and safe to run natively.  ``tests/test_absint_kind_matrix.py`` is the sweep over every kind and every
consumer.
"""

from __future__ import annotations

import io
import sys
import textwrap
import warnings

import pytest

from tharkun.solve.absint import prove

HEAD = "import math\nimport re\nimport io\nimport random\nfrom array import array\nclass A:\n    pass\n"
SENTINEL = "reached = True\nc = 1 // 0\n"


def native(src: str):
    """``(sentinel reached, the exception class names a native run raises before it)``."""
    ns: dict = {"__name__": "__main__"}
    warnings.simplefilter("ignore")
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        exec(compile(src, "<program>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        reached = bool(ns.get("reached"))
        return reached, None if reached else {c.__name__ for c in type(e).__mro__}
    finally:
        sys.stdout = old
    return False, None


def verdict(src: str, cls: str) -> str:
    return prove(src, [cls], filename="<program>", time_budget=20.0).verdict


def never_true(consumer: str, cls: str | None = None) -> None:
    """``consumer`` then the sentinel: native reaches the sentinel (or raises ``cls`` in ``consumer``) and absint must NOT say
    TRUE for it."""
    src = HEAD + textwrap.dedent(consumer) + "\n" + SENTINEL
    reached, raised = native(src)
    if cls is None:
        assert reached, ("the consumer must succeed natively", consumer, raised)
        cls = "ZeroDivisionError"
    else:
        assert not reached and cls in raised, ("native must raise " + cls, consumer, reached, raised)
    assert verdict(src, cls) != "TRUE", src


def refused(consumer: str, why: str) -> None:
    src = HEAD + textwrap.dedent(consumer) + "\n" + SENTINEL
    r = prove(src, ["ZeroDivisionError"], filename="<program>", time_budget=20.0)
    assert r.verdict == "UNKNOWN" and r.reason.startswith("unsupported:") and why in r.reason, (consumer, r.reason)


# ---------------------------------------------------------------------------------------------------------------------------
# array: unpack and random.choice are exact
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("body", ["x, y = array('i', [1, 2])", "a = array('f', [0.5, 1.5])\nx, y = a", "(x, y), z = array('i', [1, 2]), 1",
                                  "d = {1: array('i', [1, 2])}\ny, z = d[1]", "def g(a):\n    x, y = a\ng(array('i', [1, 2]))"])
def test_unpacking_an_array_continues_the_path(body):
    never_true(body)


def test_unpacking_an_array_binds_its_elements():
    assert verdict(HEAD + "a = array('i', [1, 2])\nx, y = a\nc = 10 // x\n", "ZeroDivisionError") == "TRUE"
    assert verdict(HEAD + "a = array('i', [0, 2])\nx, y = a\nc = 10 // x\n", "ZeroDivisionError") == "UNKNOWN"
    never_true("a = array('i', [0, 2])\nx, y = a\nc = 10 // x", "ZeroDivisionError")


def test_random_choice_of_an_array_continues_and_checks_its_length():
    never_true("a = array('i', [1, 2])\nx = random.choice(a)")
    never_true("a = array('i')\nx = random.choice(a)", "IndexError")
    assert verdict(HEAD + "a = array('i', [1, 2])\nx = random.choice(a)\n", "IndexError") == "TRUE"
    assert verdict(HEAD + "a = array('i', [1, 2])\nx = random.choice(a)\nc = 10 // x\n", "ZeroDivisionError") == "TRUE"
    assert verdict(HEAD + "a = array('i', [0])\nx = random.choice(a)\nc = 10 // x\n", "ZeroDivisionError") == "UNKNOWN"


# ---------------------------------------------------------------------------------------------------------------------------
# str: the index and the slice bounds
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("index", ["2.5", "'a'", "None", "[1]", "(1,)", "A()", "len", "math"])
def test_a_str_subscripted_by_a_non_int_is_a_type_error(index):
    never_true(f"x = 'ab'[{index}]", "TypeError")


@pytest.mark.parametrize("bound", ["1.5", "'a'", "[1]"])
def test_a_str_slice_with_a_non_int_bound_is_a_type_error(bound):
    never_true(f"x = 'ab'[{bound}:]", "TypeError")
    never_true(f"x = 'ab'[:{bound}]", "TypeError")


def test_a_str_subscripted_by_an_int_is_still_proven_type_safe():
    assert verdict(HEAD + "x = 'ab'[0]\n", "TypeError") == "TRUE"
    assert verdict(HEAD + "x = 'ab'[True]\n", "TypeError") == "TRUE"
    assert verdict(HEAD + "x = 'ab'[0:1]\n", "TypeError") == "TRUE"
    assert verdict(HEAD + "x = 'ab'[None:None]\n", "TypeError") == "TRUE"


# ---------------------------------------------------------------------------------------------------------------------------
# dict[slice]
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("sl", ["0:1", ":", "::2", "1:"])
def test_a_dict_subscripted_by_a_slice_is_a_key_error_not_a_type_error(sl):
    never_true(f"x = {{1: 2}}[{sl}]", "KeyError")
    never_true(f"x = {{}}[{sl}]", "KeyError")


# ---------------------------------------------------------------------------------------------------------------------------
# float * sequence
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("seq", ["[1]", "(1,)", "'ab'", "array('i', [1])", "[]", "''"])
def test_a_sequence_repeated_by_a_float_is_a_type_error(seq):
    never_true(f"x = 2.5 * {seq}", "TypeError")
    never_true(f"x = {seq} * 2.5", "TypeError")
    never_true(f"x = random.random() * {seq}", "TypeError")


def test_a_sequence_repeated_by_an_int_is_still_proven_type_safe():
    for seq in ("[1]", "(1,)", "'ab'", "array('i', [1])"):
        assert verdict(HEAD + f"x = 2 * {seq}\n", "TypeError") == "TRUE"
        assert verdict(HEAD + f"x = {seq} * 2\n", "TypeError") == "TRUE"
        assert verdict(HEAD + f"x = True * {seq}\n", "TypeError") == "TRUE"


# ---------------------------------------------------------------------------------------------------------------------------
# `x in <str>`
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("left", ["len", "int", "math", "ValueError", "1", "None", "2.5", "[1]", "A()"])
def test_a_non_str_in_a_str_is_a_type_error(left):
    never_true(f"x = {left} in 'ab'", "TypeError")


def test_a_str_in_a_str_is_still_proven_type_safe():
    assert verdict(HEAD + "x = 'a' in 'ab'\n", "TypeError") == "TRUE"


# ---------------------------------------------------------------------------------------------------------------------------
# attribute stores
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("obj", ["[1]", "(1,)", "{1: 2}", "range(3)", "array('i', [1])", "iter([1])", "re.compile('a')", "{1: 2}.keys()",
                                 "A().__init__", "[].append", "5", "'ab'", "None", "2.5"])
def test_an_attribute_store_on_a_builtin_object_is_an_attribute_error(obj):
    never_true(f"v = {obj}\nv.foo = 1", "AttributeError")


def test_an_attribute_store_on_a_stringio_or_a_file_is_refused_not_a_raise():
    refused("v = io.StringIO()\nv.foo = 1", "stringio")
    refused("v = open('/dev/null', 'w')\nv.foo = 1", "outfile")


def test_an_attribute_store_on_an_instance_is_still_modelled():
    assert verdict(HEAD + "v = A()\nv.foo = 1\nx = v.foo\n", "AttributeError") == "TRUE"


# ---------------------------------------------------------------------------------------------------------------------------
# an exception instance is not callable
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("call", ["()", "(1)", "(1, 2)"])
def test_an_exception_instance_is_not_callable(call):
    never_true(f"v = ValueError('x')\nx = v{call}", "TypeError")
    never_true(f"x = ValueError('x'){call}", "TypeError")


def test_an_exception_class_is_still_callable():
    assert verdict(HEAD + "x = ValueError('x')\n", "TypeError") == "TRUE"
    assert verdict(HEAD + "v = ValueError\nx = v('x')\n", "TypeError") == "TRUE"


# ---------------------------------------------------------------------------------------------------------------------------
# a module attribute no table models
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["foo", "x", "nothing_here"])
def test_a_module_attribute_the_tables_do_not_know_may_not_exist(name):
    never_true(f"v = math\nx = v.{name}", "AttributeError")
    never_true(f"x = math.{name}", "AttributeError")


def test_a_modelled_module_attribute_is_still_proven():
    assert verdict(HEAD + "x = math.pi\ny = math.sqrt(4.0)\n", "AttributeError") == "TRUE"


# ---------------------------------------------------------------------------------------------------------------------------
# a dict view outlives its dict's last name
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("use", ["for x in v:\n    pass", "n = len(v)", "b = 1 in v", "x = list(v)", "x, = v", "x = max(v)", "x = sum(v)"])
def test_a_dict_view_keeps_its_dict_alive(use):
    src = HEAD + f"v = {{1: 2}}.keys()\n{use}\n" + SENTINEL
    r = prove(src, ["ZeroDivisionError"], filename="<program>", time_budget=20.0)        # used to raise KeyError out of prove()
    assert r.verdict != "TRUE"


def test_a_dict_view_of_a_dict_built_in_a_call_keeps_it_alive():
    src = HEAD + "def f():\n    return {1: 2}.values()\nv = f()\nfor x in v:\n    c = 1 // 0\n"
    assert prove(src, ["ZeroDivisionError"], filename="<program>", time_budget=20.0).verdict != "TRUE"


# ---------------------------------------------------------------------------------------------------------------------------
# a StringIO is iterable and its lines are not modelled
# ---------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("body", ["for x in v:\n    reached = True\n    c = 1 // 0", "x, y = v", "xs = [0]\nxs.extend(v)", "x = list(v)",
                                  "x = sorted(v)", "x = max(v)", "xs = [0]\nxs[0:1] = v"])
def test_iterating_a_stringio_is_refused_not_an_empty_loop(body):
    src = HEAD + "v = io.StringIO('a\\nb\\n')\n" + body + "\n" + SENTINEL
    reached, _ = native(src)
    assert reached
    r = prove(src, ["ZeroDivisionError"], filename="<program>", time_budget=20.0)
    assert r.verdict == "UNKNOWN", r.reason
