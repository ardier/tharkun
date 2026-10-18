"""``str.__getitem__`` on an abstract str: the proven cases of ``AbsintVisitor._abstract_string`` ([internal ref] regression of #151).

Since #151 ``Analyzer.subscript_load`` routes a definite-str subscript through the rule ``str.__getitem__``.  ``_abstract_string`` had
proven-cases for ``str.__mod__``, ``__contains__``, the zero-argument methods, ``repr``/``str``/``format`` but NONE for ``__getitem__``, so every
``s[i]`` and ``s[a:b]`` on an abstract str left its ``TypeError`` and ``AttributeError`` (and the rest of the contract's classes) unproven:
``dynamic-typing`` was never TRUE for a program that indexes a ``nondet_str()``, and five of MIN's mini tasks lost their TRUE.

The rule's proven cases (``visitor.py::str_key_parts``; the table is in docs/ABSINT.md "A str subscript through the rule"): a receiver that is a
definite builtin str and a key none of whose parts runs user code.  An exact int or bool proves every class of the contract except the
``IndexError`` of the index; an int-bound slice proves them all; a float, str, None, builtin container or a slice with a float bound is a
certain ``TypeError`` and nothing else (every other class is proven: no user code runs); a key that may be an int or not leaves the
``TypeError`` and the ``IndexError``.  An object, unknown or static key is refused.  The last section is MG's precision finding on e5580b2.
Each program here is small and safe to run natively; the native run is the control.
"""

from __future__ import annotations

import io
import sys
import warnings

import pytest

from tharkun.solve.absint import PROPERTY_CLASSES, prove

HEAD = "import _sv_verifier\n"
MAIN = 'if __name__ == "__main__":\n'


def _body(lines: str) -> str:
    return HEAD + "\n" + MAIN + "".join("    " + l + "\n" for l in lines.strip("\n").splitlines())


def verdict(src: str, prop: str) -> str:
    return prove(src, PROPERTY_CLASSES[prop], filename="<program>", time_budget=20.0).verdict


def verdict_for(src: str, cls: str) -> str:
    return prove(src, [cls], filename="<program>", time_budget=20.0).verdict


def native(src: str, draws: dict):
    """The exception class names a native run raises, or ``()`` -- with ``_sv_verifier`` drawing from ``draws`` in order."""
    import types

    values = {"str": iter(draws.get("str", [])), "int": iter(draws.get("int", [])), "float": iter(draws.get("float", []))}
    m = types.ModuleType("_sv_verifier")
    m.nondet_str = lambda: next(values["str"])
    m.nondet_int = lambda: next(values["int"])
    m.nondet_float = lambda: next(values["float"])
    old_mod, old_out = sys.modules.get("_sv_verifier"), sys.stdout
    sys.modules["_sv_verifier"] = m
    sys.stdout = io.StringIO()
    warnings.simplefilter("ignore")
    try:
        exec(compile(src, "<program>", "exec"), {"__name__": "__main__"})
    except Exception as e:  # noqa: BLE001
        return tuple(c.__name__ for c in type(e).__mro__)
    finally:
        sys.stdout = old_out
        if old_mod is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = old_mod
    return ()


# ---------------------------------------------------------------------------------------------------------------------------
# the five MIN mini tasks that lost their TRUE (expected verdicts from the task files)
# ---------------------------------------------------------------------------------------------------------------------------

MIN_TASKS = {
    "ai_str_eq_refines_len_true": ("s = _sv_verifier.nondet_str()\nif s == 'abc':\n    print(s[2])",
                                   {"data-lookup": True, "arithmetic": True, "dynamic-typing": True, "assertion-safety": True},
                                   {"str": ["abc"]}),
    "ai_str_first_unguarded_false": ("s = _sv_verifier.nondet_str()\nprint(s[0])",
                                     {"data-lookup": False, "arithmetic": True, "dynamic-typing": True, "assertion-safety": True},
                                     {"str": [""]}),
    "ai_str_index_guard_le_false": ("s = _sv_verifier.nondet_str()\ni = _sv_verifier.nondet_int()\nif 0 <= i <= len(s):\n    print(s[i])",
                                    {"data-lookup": False, "arithmetic": True, "dynamic-typing": True, "assertion-safety": True},
                                    {"str": ["ab"], "int": [2]}),
    "ai_str_index_guard_true": ("s = _sv_verifier.nondet_str()\ni = _sv_verifier.nondet_int()\nif 0 <= i < len(s):\n    print(s[i])",
                                {"data-lookup": True, "arithmetic": True, "dynamic-typing": True, "assertion-safety": True},
                                {"str": ["ab"], "int": [1]}),
    "ai_str_nonempty_truthiness_true": ("s = _sv_verifier.nondet_str()\nif s:\n    print(s[0], s[-1])",
                                        {"data-lookup": True, "arithmetic": True, "dynamic-typing": True, "assertion-safety": True},
                                        {"str": ["x"]}),
}


@pytest.mark.parametrize("task", sorted(MIN_TASKS))
def test_the_mini_task_is_proven_for_the_properties_absint_proves(task):
    """``arithmetic``, ``dynamic-typing`` and ``assertion-safety`` are TRUE on sv_2027 (``cfb8505``, whose own str subscript had a
    check) and on the PR head with this fix; #151 alone lost all three (UNKNOWN).  ``data-lookup`` (the ``IndexError``) was UNKNOWN on
    sv_2027 too, for every one of the five (``string length not tracked``): the TRUE ones get theirs from another component, and the
    two ``_false`` tasks raise it natively, so it must never be TRUE for them."""
    body, expected, _draws = MIN_TASKS[task]
    src = _body(body)
    for prop in ("arithmetic", "dynamic-typing", "assertion-safety"):
        assert expected[prop] and verdict(src, prop) == "TRUE", (task, prop)
    if not expected["data-lookup"]:
        assert verdict(src, "data-lookup") != "TRUE", task


@pytest.mark.parametrize("task", ["ai_str_first_unguarded_false", "ai_str_index_guard_le_false"])
def test_the_false_tasks_really_raise_index_error_natively(task):
    """The control: the two ``_false`` tasks raise ``IndexError`` for a draw, so ``data-lookup`` must not be TRUE."""
    body, _expected, draws = MIN_TASKS[task]
    assert "IndexError" in native(_body(body), draws)


@pytest.mark.parametrize("task", ["ai_str_eq_refines_len_true", "ai_str_index_guard_true", "ai_str_nonempty_truthiness_true"])
def test_the_true_tasks_run_clean_natively(task):
    body, _expected, draws = MIN_TASKS[task]
    assert native(_body(body), draws) == ()


# ---------------------------------------------------------------------------------------------------------------------------
# an index: IndexError stays a check; every other class is proven
# ---------------------------------------------------------------------------------------------------------------------------

def test_an_unguarded_index_leaves_only_the_index_error_unproven():
    src = _body("s = _sv_verifier.nondet_str()\nprint(s[0])")
    assert verdict_for(src, "IndexError") != "TRUE"
    for cls in ("TypeError", "AttributeError", "KeyError", "ZeroDivisionError", "AssertionError"):
        assert verdict_for(src, cls) == "TRUE", cls


@pytest.mark.parametrize("key", ["True", "-1", "0", "10 ** 30"])
def test_a_constant_int_or_bool_index_is_an_exact_int(key):
    src = _body(f"s = _sv_verifier.nondet_str()\nprint(s[{key}])")
    assert verdict_for(src, "TypeError") == "TRUE"
    assert verdict_for(src, "IndexError") != "TRUE"


# ---------------------------------------------------------------------------------------------------------------------------
# a slice never raises
# ---------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["1:3", ":2", "1:", "::2", "::-1", "-3:-1", ":", "1:3:2", "10 ** 30:", "True:", "False:True"])
def test_an_int_slice_never_raises_for_any_class(key):
    src = _body(f"s = _sv_verifier.nondet_str()\nprint(s[{key}])")
    assert native(src, {"str": ["abcdef"]}) == ()
    for prop in PROPERTY_CLASSES:
        assert verdict(src, prop) == "TRUE", (key, prop)
    assert verdict_for(src, "IndexError") == "TRUE"


@pytest.mark.parametrize("key", [":n", "n:", "n:n + 2", "::n", "-n:", "1:n:2"])
def test_a_slice_with_a_bound_that_is_an_int_but_not_a_constant_never_raises(key):
    """sv_2027 proved these; #151 refused them (``non-finite slice bounds``): the bound is an exact int, so the slice is an int slice."""
    src = _body(f"s = _sv_verifier.nondet_str()\nn = len(s)\nprint(s[{key}])")
    assert verdict_for(src, "TypeError") == "TRUE", key
    assert verdict_for(src, "IndexError") == "TRUE", key


def test_a_slice_with_a_bound_that_may_not_be_an_int_is_not_proven():
    src = _body("s = _sv_verifier.nondet_str()\nn = _sv_verifier.nondet_float()\nprint(s[:n])")
    assert verdict_for(src, "TypeError") != "TRUE"


# ---------------------------------------------------------------------------------------------------------------------------
# the negative probes: a key that is not an exact int / int slice stays unproven for TypeError
# ---------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["1.5", "'a'", "None", "[0]", "(0,)", "1.5:2", "'a':", ":'b'", "::0.5", "None:None:'x'"])
def test_a_float_str_none_or_object_key_is_never_proven_free_of_type_error(key):
    src = _body(f"s = _sv_verifier.nondet_str()\nprint(s[{key}])")
    assert "TypeError" in native(src, {"str": ["abcdef"]}), key
    assert verdict_for(src, "TypeError") != "TRUE", key


def test_a_key_that_may_be_an_int_or_a_float_is_not_proven():
    src = _body("s = _sv_verifier.nondet_str()\nk = 1 if _sv_verifier.nondet_int() > 0 else 1.5\nprint(s[k])")
    assert native(src, {"str": ["abc"], "int": [0]}).count("TypeError") == 1
    assert verdict_for(src, "TypeError") != "TRUE"


def test_a_key_that_may_be_an_int_or_a_slice_is_not_proven_exact():
    """An int and a slice alternative together: the index alternative may be out of range.  (Today no key holds both: ``slice(...)`` is
    not modelled, so this is a refusal; the test pins that the join is never proven, should ``slice`` ever be.)"""
    src = _body("s = _sv_verifier.nondet_str()\nk = 5 if _sv_verifier.nondet_int() > 0 else slice(1, 2)\nprint(s[k])")
    assert verdict_for(src, "IndexError") != "TRUE"


def test_a_str_key_joined_with_an_int_is_not_proven():
    src = _body("s = _sv_verifier.nondet_str()\nk = 0 if _sv_verifier.nondet_int() > 0 else 'a'\nprint(s[k])")
    assert native(src, {"str": ["abc"], "int": [0]}).count("TypeError") == 1
    assert verdict_for(src, "TypeError") != "TRUE"


def test_a_none_key_joined_with_an_int_is_not_proven():
    src = _body("s = _sv_verifier.nondet_str()\nk = 0 if _sv_verifier.nondet_int() > 0 else None\nprint(s[k])")
    assert native(src, {"str": ["abc"], "int": [0]}).count("TypeError") == 1
    assert verdict_for(src, "TypeError") != "TRUE"


@pytest.mark.parametrize("key", ["len", "int", "print"])
def test_a_function_or_type_key_is_refused_never_proven(key):
    src = _body(f"s = _sv_verifier.nondet_str()\nprint(s[{key}])")
    assert "TypeError" in native(src, {"str": ["abc"]})
    assert verdict_for(src, "TypeError") != "TRUE"


def test_an_unknown_value_key_is_a_named_refusal_or_unproven_never_true():
    src = _body("s = _sv_verifier.nondet_str()\nk = _sv_verifier.nondet_float()\nprint(s[k])")
    assert verdict_for(src, "TypeError") != "TRUE"


def test_a_user_object_receiver_is_not_a_definite_str():
    src = ("import _sv_verifier\nclass A:\n    def __getitem__(self, i):\n        return 1 // 0\n\n"
           + MAIN + "    a = A()\n    print(a[0])\n")
    assert "ZeroDivisionError" in native(src, {})
    assert verdict_for(src, "ZeroDivisionError") != "TRUE"


# ---------------------------------------------------------------------------------------------------------------------------
# the precision ([internal ref], MG's finding on e5580b2): a key that is not an int or an int slice is a TypeError and nothing else
# ---------------------------------------------------------------------------------------------------------------------------
# A key that runs no user code (a float, str, None, a builtin container, a slice with a float bound) gave all seven classes
# "unproven", so a program that natively raises TypeError there lost its TRUE for arithmetic, data-lookup and assertion-safety
# (179 cells against ec3cee0 on MG's table).  Now only the TypeError is unproven, and the IndexError when some part of the key
# may be an int index.

OTHER_CLASSES = ("AttributeError", "KeyError", "ZeroDivisionError", "FloatingPointError", "AssertionError")
NO_INT_KEYS = ["1.5", "'a'", "None", "[0]", "(0,)", "{1: 2}", "1.5:2", "'a':", ":'b'", "::0.5", "None:None:'x'", "(1,):2"]


@pytest.mark.parametrize("key", NO_INT_KEYS)
def test_a_key_with_no_int_part_is_a_certain_type_error_and_nothing_else(key):
    src = _body(f"s = _sv_verifier.nondet_str()\nprint(s[{key}])")
    assert "TypeError" in native(src, {"str": ["abcdef"]}), key           # the control: it raises natively
    assert verdict_for(src, "TypeError") != "TRUE", key
    for cls in ("IndexError",) + OTHER_CLASSES:
        assert verdict_for(src, cls) == "TRUE", (key, cls)
    for prop in ("arithmetic", "data-lookup", "assertion-safety"):
        assert verdict(src, prop) == "TRUE", (key, prop)
    assert verdict(src, "dynamic-typing") != "TRUE", key


@pytest.mark.parametrize("key", [":x", "x:", "::x", "x:x", "1:x", ":x:2", "-x:"])
def test_a_slice_bound_that_may_be_a_float_is_a_type_error_and_nothing_else(key):
    """``x`` is a ``nondet_float()``: not finitely many values, so the slice is the ``("any-slice",)`` key; a slice never raises IndexError."""
    src = _body(f"s = _sv_verifier.nondet_str()\nx = _sv_verifier.nondet_float()\nprint(s[{key}])")
    assert "TypeError" in native(src, {"str": ["abcdef"], "float": [1.5]}), key
    assert verdict_for(src, "TypeError") != "TRUE", key
    for cls in ("IndexError",) + OTHER_CLASSES:
        assert verdict_for(src, cls) == "TRUE", (key, cls)


@pytest.mark.parametrize("other", ["1.5", "None", "'a'"])
def test_a_key_that_may_be_an_int_or_not_leaves_the_type_error_and_the_index_error(other):
    src = _body(f"s = _sv_verifier.nondet_str()\nk = 0 if _sv_verifier.nondet_int() > 0 else {other}\nprint(s[k])")
    assert native(src, {"str": ["abc"], "int": [0]}).count("TypeError") == 1      # the other branch is clean natively:
    assert native(src, {"str": ["abc"], "int": [1]}) == ()                        # the key really may be either
    assert verdict_for(src, "TypeError") != "TRUE"
    assert verdict_for(src, "IndexError") != "TRUE"
    for cls in OTHER_CLASSES:
        assert verdict_for(src, cls) == "TRUE", (other, cls)


def test_a_key_that_may_be_a_float_or_a_float_slice_has_no_index_error():
    """A float is a TypeError and a slice never raises IndexError: with no int part there is no IndexError."""
    src = _body("s = _sv_verifier.nondet_str()\nx = _sv_verifier.nondet_float()\nk = x if _sv_verifier.nondet_int() > 0 else 1.5\nprint(s[k])")
    assert verdict_for(src, "IndexError") == "TRUE"
    assert verdict_for(src, "TypeError") != "TRUE"


# the refusals stay: a key or a bound that may run user code, or is not understood, is a named refusal -- never proven

REFUSED_KEYS = ["B()", ":B()", "B():", "::B()"]       # a user instance may define __index__ or be a metaclass trick: never proven


@pytest.mark.parametrize("key", REFUSED_KEYS)
def test_a_key_that_may_run_user_code_or_is_not_understood_is_refused(key):
    src = ("import math\nimport _sv_verifier\nclass B:\n    pass\n\n" + MAIN + f"    s = _sv_verifier.nondet_str()\n    print(s[{key}])\n")
    assert "TypeError" in native(src, {"str": ["abc"]}), key
    for cls in ("TypeError", "IndexError") + OTHER_CLASSES:
        r = prove(src, [cls], filename="<program>", time_budget=20.0)
        assert r.verdict == "UNKNOWN" and r.reason.startswith("unsupported:"), (key, cls, r.verdict, r.reason)


def test_a_float_or_str_receiver_is_not_a_definite_str():
    src = _body("x = _sv_verifier.nondet_float()\nprint(x[0])")
    assert "TypeError" in native(src, {"float": [1.5]})
    assert verdict_for(src, "TypeError") != "TRUE"


# ---------------------------------------------------------------------------------------------------------------------------
# a certain TypeError ends the path (MIN's suite v3: 28 precision regressions of #151 on main fca4fb9 against ec3cee0)
# ---------------------------------------------------------------------------------------------------------------------------
# ``s[1.5]``, ``s[len]``, ``s[[1]]``, ``s[range(2)]``, ``s[:[1]]`` ... raise TypeError on every run: the exception is recorded as an
# unproven check, and nothing after the subscript is reached.  A finite str ended the path there (``_native_string``: no output) and
# ec3cee0 did too; the abstract-str case let the path go on, so ``reached = True; 1 // 0`` after it made ``arithmetic`` UNKNOWN.  Static
# objects (builtin functions, types, modules) and builtin objects (a range) were refused outright, and a slice bound that is an int or
# None (``s[:(1 if c else None)]``) was treated as one that may not be an int.

CERTAIN_KEYS = ["1.0", "1.5", "None", "'a'", "[1]", "(1,)", "{1: 2}", "{1, 2}", "frozenset([1])", "range(2)", "len", "int", "print", "math", "ValueError",
                "nd() / 2", "nd() * 0.5", "1.5:2", ":[1]", "[1]:", "::[1]", "range(2):", ":{1, 2}", "::frozenset()", "range(2):", "::range(2)", ":len", ":None.__class__"]


def _reach(key, receiver="_sv_verifier.nondet_str()"):
    return ("import math\nimport _sv_verifier\n\ndef nd():\n    return _sv_verifier.nondet_int()\n\n"
            f"s = {receiver}\nx = s[{key}]\nreached = True\nz = 1 // 0\n")


@pytest.mark.parametrize("key", [k for k in CERTAIN_KEYS if k != ":None.__class__"])
@pytest.mark.parametrize("receiver", ["_sv_verifier.nondet_str()", "'abc'", "_sv_verifier.nondet_str() + 'a'", "_sv_verifier.nondet_str().upper()",
                                      "_sv_verifier.nondet_str() * 2"])
def test_a_certain_type_error_ends_the_path(receiver, key):
    src = _reach(key, receiver)
    raised = native(src, {"str": ["abc"], "int": [3]})
    assert "TypeError" in raised, (key, raised)                       # the control: it raises natively, before the sentinel
    assert verdict_for(src, "TypeError") != "TRUE", (receiver, key)   # ... and the TypeError stays a recorded, unproven check
    for cls in ("ZeroDivisionError", "IndexError", "KeyError", "AssertionError", "AttributeError", "FloatingPointError"):
        assert verdict_for(src, cls) == "TRUE", (receiver, key, cls)  # the sentinel is never reached
    for prop in ("arithmetic", "data-lookup", "assertion-safety"):
        assert verdict(src, prop) == "TRUE", (receiver, key, prop)
    assert verdict(src, "dynamic-typing") != "TRUE"


@pytest.mark.parametrize("bound", ["(1 if nd() > 0 else None)", "None", "nd()", "(nd() if nd() > 0 else None)", "len(s)", "True"])
@pytest.mark.parametrize("slot", ["s[:{b}]", "s[{b}:]", "s[::{b}]", "s[{b}:{b}]"])
@pytest.mark.parametrize("receiver", ["_sv_verifier.nondet_str()", "'abc'"])
def test_an_int_or_none_slice_bound_never_raises_and_the_path_goes_on(receiver, slot, bound):
    """``s[:(1 if c else None)]``: an int or None is a valid bound; the path reaches the sentinel natively and absint says so."""
    key = slot.format(b=bound)
    src = ("import _sv_verifier\n\ndef nd():\n    return _sv_verifier.nondet_int()\n\n"
           f"s = {receiver}\nx = {key}\nreached = True\nz = 1 // 0\n")
    if "step" and slot == "s[::{b}]" and bound in ("nd()", "(nd() if nd() > 0 else None)", "True"):
        pass
    raised = native(src, {"str": ["abc"], "int": [1, 2, 3, 4, 5, 6]})
    if "ValueError" in raised:
        pytest.skip("a zero slice step is a ValueError, not a property class")
    assert "ZeroDivisionError" in raised, (key, raised)                # the control: natively the sentinel is reached
    assert verdict_for(src, "TypeError") == "TRUE", key
    assert verdict_for(src, "IndexError") == "TRUE", key
    assert verdict_for(src, "ZeroDivisionError") != "TRUE", key        # the path is NOT ended: the sentinel is reachable


def test_a_key_that_may_be_valid_does_not_end_the_path():
    """``k`` is an int or a float: the int branch reaches the sentinel natively, so a TRUE for ZeroDivisionError would be wrong."""
    src = _reach("0 if nd() > 0 else 1.5")
    assert "ZeroDivisionError" in native(src, {"str": ["abc"], "int": [1, 1]}) or "ZeroDivisionError" in native(src, {"str": ["abc"], "int": [1, 5]})
    assert verdict_for(src, "ZeroDivisionError") != "TRUE"
    src = _reach(":(nd() if nd() > 0 else 1.5)")
    assert verdict_for(src, "ZeroDivisionError") != "TRUE"           # an int bound is valid: the path may go on


def test_a_slice_whose_bound_may_be_a_float_does_not_end_the_path_but_a_list_bound_does():
    src_may = ("import _sv_verifier\nx = _sv_verifier.nondet_float()\ns = _sv_verifier.nondet_str()\ny = s[:x]\nreached = True\nz = 1 // 0\n")
    assert verdict_for(src_may, "TypeError") != "TRUE"
    # nd() is an int, so the bound is an int: valid
    src_int = ("import _sv_verifier\nn = _sv_verifier.nondet_int()\ns = _sv_verifier.nondet_str()\ny = s[:n]\nreached = True\nz = 1 // 0\n")
    assert "ZeroDivisionError" in native(src_int, {"str": ["abc"], "int": [1]})
    assert verdict_for(src_int, "ZeroDivisionError") != "TRUE"
    assert verdict_for(_reach(":[1]"), "ZeroDivisionError") == "TRUE"


@pytest.mark.parametrize("key", [
    "s[:(None if nd() > 0 else [1])]", "s[(None if nd() > 0 else [1]):]", "s[::(None if nd() > 0 else [1])]",
    "s[(None if nd() > 0 else [1]):(None if nd() > 0 else [1])]", "s[:(None if nd() > 0 else 1.5)]",
    "s[(None if nd() > 0 else (1,)):]", "s[(1 if nd() > 0 else [1]):]"])
@pytest.mark.parametrize("receiver", ["_sv_verifier.nondet_str()", "'abc'"])
def test_a_slice_bound_that_is_none_or_a_list_may_be_valid_so_the_path_is_not_ended(receiver, key):
    """MG's review of #159 (mutant M5: ``non_int_value`` ignoring ``None`` made ``None | [1]`` a certain TypeError, which no test caught): a
    bound that is None (or an int) for some draws is a valid slice bound for those, the sentinel is reached natively, and a list, float or
    tuple bound for the others is a TypeError; neither may be proven away."""
    src = ("import _sv_verifier\n\ndef nd():\n    return _sv_verifier.nondet_int()\n\n"
           f"s = {receiver}\nx = {key}\nreached = True\nz = 1 // 0\n")
    assert "ZeroDivisionError" in native(src, {"str": ["abc"], "int": [1] * 6}), key     # the valid bound: the sentinel is reached
    assert "TypeError" in native(src, {"str": ["abc"], "int": [0] * 6}), key             # the other bound: a TypeError, before it
    assert verdict_for(src, "ZeroDivisionError") != "TRUE", (receiver, key)
    assert verdict_for(src, "TypeError") != "TRUE", (receiver, key)
    assert verdict(src, "arithmetic") != "TRUE", (receiver, key)


def test_a_certain_type_error_in_a_length_probe_is_still_refused_not_swallowed_away():
    """CPython swallows a ``TypeError`` raised by ``__len__`` in ``list(x)``; ending the path at the subscript inside it would hide the
    code after ``list(x)``.  The probe refuses a callee with a recorded TypeError check, so this stays UNKNOWN, never TRUE."""
    for body in ("        x = 'abc'[1.5]\n        return 1\n", "        x = _sv_verifier.nondet_str()[len]\n        return 1\n"):
        src = ("import _sv_verifier\nclass A:\n    def __len__(self):\n" + body +
               "    def __iter__(self):\n        return iter([1])\nx = list(A())\nreached = True\nz = 1 // 0\n")
        assert "ZeroDivisionError" in native(src, {"str": ["abc"]}), body
        assert verdict_for(src, "ZeroDivisionError") != "TRUE", body


def test_a_regex_flag_is_an_int_and_a_valid_index():
    """``re.I`` is a ``RegexFlag`` (an int): ``s[re.I]`` is valid, so a static tag must not make it a certain TypeError."""
    src = ("import re\nimport _sv_verifier\ns = _sv_verifier.nondet_str()\nx = s[re.I]\nreached = True\nz = 1 // 0\n")
    assert "ZeroDivisionError" in native(src, {"str": ["abcdef"]})
    assert verdict_for(src, "ZeroDivisionError") != "TRUE"
