"""[internal ref]: the iterable-reducing builtins in ``tharkun.sv._symexec`` (the search executor), the sibling of the absint
models in ``solve/absint/reducers.py``.

``min``/``max`` over an iterable (``default=`` and ``key=None``), ``any``, ``all``, ``sum`` with a start and
``sorted(reverse=)`` run on the unrolled elements, symbolic ones included (an If chain, an Or, an And, a sum
of terms).  A claim here is advisory (a vector is replayed natively before it is believed), so the tests check the
two things that matter: a ``sat`` vector reaches the site natively, and the executor does not claim ``sat`` for a
site no draw in the box reaches.
"""

from __future__ import annotations

import itertools
import sys
import textwrap

import pytest

z3 = pytest.importorskip("z3")

from tharkun.sv import _symexec  # noqa: E402
from tharkun.sv.sites import build_harness  # noqa: E402

HEAD = "import _sv_verifier\nif __name__ == \"__main__\":\n"
DRAWS = "a = _sv_verifier.nondet_int()\nb = _sv_verifier.nondet_int()\n"


def _prog(body: str) -> str:
    return HEAD + textwrap.indent(DRAWS + textwrap.dedent(body), "    ")


def _solve(src: str):
    h = build_harness(src, ("KeyError",), None)
    assert h is not None, src
    mod = {"__name__": "sv_harness_test"}
    sys.modules.setdefault("_sv_verifier", type(sys)("_sv_verifier"))
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    got = _symexec.solve_targets(h.source, mod, "__sv_harness", [s["line"] for s in h.sites],
                                 h.params, h.kinds, {})
    return got[h.sites[0]["line"]]


def _raises_key_error(src: str, a, b) -> bool:
    import types

    it = iter([a, b])
    mod = types.ModuleType("_sv_verifier")
    mod.nondet_int = lambda: next(it)
    saved = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = mod
    try:
        exec(compile(src, "<native>", "exec"), {"__name__": "__main__"})
        return False
    except KeyError:
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        if saved is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = saved


def _reaches(src: str, box=range(-6, 9)) -> list:
    """The ``(a, b)`` in the box for which the program raises its KeyError natively."""
    return [(a, b) for a, b in itertools.product(box, box) if _raises_key_error(src, a, b)]


CASES = {
    "min of two": "if min([a, b]) == 7:\n    raise KeyError(1)\n",
    "max of three": "if max([a, b, 3]) == 3:\n    raise KeyError(1)\n",
    "max over a tuple": "if max((a, b)) < 0:\n    raise KeyError(1)\n",
    "min default": "if min([], default=a) == 4:\n    raise KeyError(1)\n",
    "sum": "if sum([a, b]) == 5:\n    raise KeyError(1)\n",
    "sum with a start": "if sum([a, b], 10) == 12:\n    raise KeyError(1)\n",
    "sum with start=": "if sum([a, b], start=3) == 3:\n    raise KeyError(1)\n",
    "any": "if any([a > 5, b > 5]):\n    raise KeyError(1)\n",
    "all": "if all([a > 5, b > 5]):\n    raise KeyError(1)\n",
    "all empty": "if all([]) and not any([]):\n    raise KeyError(1)\n",
    "sorted reverse": "if sorted([3, 1, 2], reverse=True)[0] == 3:\n    raise KeyError(1)\n",
    "unsat max": "if max([a, b]) < min([a, b]):\n    raise KeyError(1)\n",
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_claim_about_a_reducer_matches_the_native_run(name):
    src = _prog(CASES[name])
    outcome, vector = _solve(src)
    if outcome == "sat":
        # the vector reaches the site natively
        assert _raises_key_error(src, vector["a"], vector["b"]), (name, vector)
    elif outcome == "unsat":
        assert not _reaches(src), (name, outcome)


def test_the_family_is_solved_not_refused():
    solved = {name: _solve(_prog(body))[0] for name, body in CASES.items()}
    for name in ("min of two", "max of three", "max over a tuple", "min default", "sum", "sum with a start",
                 "sum with start=", "any", "all", "all empty", "sorted reverse"):
        assert solved[name] == "sat", (name, solved[name])
    assert solved["unsat max"] != "sat"


def test_a_key_function_and_an_empty_min_are_refused_not_guessed():
    for body in ("if min([a, b], key=abs) == 1:\n    raise KeyError(1)\n",
                 "if min([]) == 1:\n    raise KeyError(1)\n",
                 "if sorted([a, b], key=abs)[0] == 1:\n    raise KeyError(1)\n"):
        outcome, _vector = _solve(_prog(body))
        assert outcome != "sat", body
