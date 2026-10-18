"""[internal ref] (richards, chaos, scimark): ``isinstance`` against user classes, ``list``, ``tuple`` and tuples of classes.

richards asserts ``isinstance(h, HandlerTaskRec)`` four times, chaos checks ``isinstance(other, GVector)`` in
``__add__``, scimark ``isinstance(idx, tuple)``: all refused ("isinstance with an unsupported type argument"), since
only ``str``/``int``/``float``/``bool`` were modelled.  ``Analyzer.b_isinstance`` now takes the classinfo a program
writes (a builtin class name, a user class, a tuple of those) and decides each alternative of the value: a scalar by its
kind, an object by its class chain (single inheritance only: ``s_ClassDef`` refuses the rest) and heap kind, anything
else (a function, a module, a class) as an instance of ``object`` only.

Exactness tests, refusal tests, and a differential: a program draws one of several values and asserts
``isinstance`` (or its negation) over a classinfo; a TRUE the native run contradicts fails.
"""

from __future__ import annotations

import itertools
import random
import sys
import textwrap
import types

import pytest

from tharkun.solve.absint import prove

CLASSES = """
class A:
    def __init__(self):
        self.v = 1
class B(A):
    pass
class C(A):
    pass
class D:
    pass
class E(B):
    pass
"""
HEAD = "import _sv_verifier\ndef nd():\n    return _sv_verifier.nondet_int()\n" + CLASSES


def verdict(body: str, classes=("AssertionError",)) -> str:
    return prove(HEAD + textwrap.dedent(body), list(classes), filename="<program>").verdict


def test_an_object_is_an_instance_of_its_class_and_the_chain_above_it():
    assert verdict("x = E()\nassert isinstance(x, E)\nassert isinstance(x, B)\nassert isinstance(x, A)\nassert isinstance(x, object)\n") == "TRUE"
    assert verdict("x = E()\nassert not isinstance(x, C)\nassert not isinstance(x, D)\nassert not isinstance(x, int)\n") == "TRUE"
    assert verdict("x = A()\nassert isinstance(x, B)\n") == "UNKNOWN"          # a B is an A, not the other way round


def test_a_tuple_of_classes_is_any_of_them():
    assert verdict("x = C()\nassert isinstance(x, (B, C))\nassert not isinstance(x, (B, D))\nassert isinstance(x, (D, (A, int)))\n") == "TRUE"


def test_a_value_that_may_be_two_classes_is_decided_by_both():
    body = """
        x = B() if nd() > 0 else C()
        assert isinstance(x, A)
        assert not isinstance(x, D)
    """
    assert verdict(body) == "TRUE"
    assert verdict("x = B() if nd() > 0 else C()\nassert isinstance(x, B)\n") == "UNKNOWN"
    assert verdict("x = B() if nd() > 0 else 7\nassert isinstance(x, (A, int))\n") == "TRUE"


def test_builtin_values_and_names():
    assert verdict("assert isinstance(1, int)\nassert isinstance(True, int)\nassert not isinstance(1, bool)\nassert isinstance(2.5, float)\n") == "TRUE"
    assert verdict("assert isinstance([1], list)\nassert isinstance((1, 2), tuple)\nassert not isinstance((1,), list)\nassert not isinstance([1], tuple)\n") == "TRUE"
    assert verdict("assert isinstance('a', str)\nassert not isinstance(None, int)\nassert isinstance(None, object)\n") == "TRUE"
    assert verdict("assert not isinstance(1, A)\nassert not isinstance('s', A)\nassert not isinstance([1], A)\nassert not isinstance(A, A)\n") == "TRUE"


def test_a_function_or_a_class_is_an_object_and_nothing_here():
    assert verdict("def f():\n    pass\nassert isinstance(f, object)\nassert not isinstance(f, A)\nassert not isinstance(f, int)\n") == "TRUE"


def test_the_classinfo_shapes_it_does_not_know_stay_refused():
    for body in ("assert isinstance(1, 5)\n",                                   # not a class: a TypeError
                 "assert isinstance(1, 'int')\n",
                 "assert isinstance(1, dict)\n",                                # a builtin the model has no name for
                 "assert isinstance(1, (int, [A]))\n",
                 "k = int if nd() > 0 else A\nassert isinstance(1, k)\n",
                 "assert isinstance(1, ValueError)\n"):
        assert verdict(body) == "UNKNOWN", body


def test_an_object_that_carries_its_own_class_is_refused():
    src = textwrap.dedent("""
        class F:
            pass
        x = F()
        x.__class__ = int
        assert isinstance(x, F)
    """)
    assert prove(src, ["AssertionError"], filename="<program>").verdict == "UNKNOWN"


# ---------------------------------------------------------------------------
# the differential
# ---------------------------------------------------------------------------

VALUES = ["A()", "B()", "C()", "D()", "E()", "7", "True", "2.5", "'s'", "None", "[1]", "(1, 2)", "len", "A"]
INFOS = ["A", "B", "C", "D", "E", "int", "bool", "float", "str", "list", "tuple", "object", "(B, C)", "(D, int)",
         "(list, tuple)", "(A, (int, str))"]
DRAWS = [0, 1, 2, 3, 4, 5, 6]


def native_raises(src: str) -> bool:
    code = compile(src, "<program>", "exec")
    for d in DRAWS:
        it = iter([d, d, d])
        mod = types.ModuleType("_sv_verifier")
        mod.nondet_int = lambda: next(it)
        saved = sys.modules.get("_sv_verifier")
        sys.modules["_sv_verifier"] = mod
        try:
            exec(code, {"__name__": "__main__"})
        except AssertionError:
            return True
        except Exception:  # noqa: BLE001
            pass
        finally:
            if saved is None:
                sys.modules.pop("_sv_verifier", None)
            else:
                sys.modules["_sv_verifier"] = saved
    return False


@pytest.mark.parametrize("seed", range(10))
def test_isinstance_never_makes_a_wrong_true(seed):
    rng = random.Random(1121_900 + seed)
    proved = 0
    for _ in range(40):
        a, b, c = rng.sample(VALUES, 3)
        pick = f"x = [{a}, {b}, {c}][nd() % 3]"
        info = rng.choice(INFOS)
        negate = rng.choice(["", "not "])
        src = HEAD + pick + f"\nassert {negate}isinstance(x, {info})\n"
        if prove(src, ["AssertionError"], filename="<program>").verdict == "TRUE":
            proved += 1
            assert not native_raises(src), f"WRONG TRUE:\n{src}"
    assert proved > 0
