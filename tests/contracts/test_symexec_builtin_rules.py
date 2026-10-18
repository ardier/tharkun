"""The symbolic executor reads the contracts of the builtins it evaluates itself ([internal ref] step 2, PR "symexec builtins").

``len``, ``abs``, ``min``, ``max``, ``any``, ``all``, ``isinstance``, ``sum``, ``range`` and ``sorted`` were evaluated by inline code of
``_Exec.ev_Call`` that never read their contracts (the audit's ``M`` cell); ``int``, ``float``, ``divmod``, ``pow`` and ``print`` were
refused or only half modelled (``R``).  All fifteen now go through their rule: ``_Exec._through_rule`` -> ``Rule.accept`` ->
``SymexecVisitor._builtin`` (a TESTED contract, else a refusal) -> ``_Exec._model_<name>``.

What this file pins, against CPython 3.14.4:

* DIFFERENTIAL: for every concrete call of the table, the executor either returns exactly what CPython returns (same class, same
  value) or refuses; it never returns a different value.  The table includes the shapes where a hand model drifts (``sum`` of floats:
  CPython 3.12+ sums with Neumaier compensation, so ``sum([0.1] * 10)`` is ``1.0``, not the left fold's ``0.999...``);
* a call that raises natively is refused or ends the path, never answered;
* symbolic: a guard on the value CPython gives for a symbolic argument is solved and replays natively; a guard on a wrong value is
  not reached;
* the contract is what admits the call: untested, the call is a refusal, not the model of old;
* NEGATIVE CONTROLS: the left-fold ``sum`` (the model before this change) is caught by the differential table.
"""

from __future__ import annotations

import ast
import dataclasses
import math

import pytest

from tests.contracts._native import run_native
from tests.contracts.test_engine_audit_consumed import SYMEXEC_RULE_BUILTINS, _solve, untested
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.sv import _symexec

BUILTINS = tuple(sorted(SYMEXEC_RULE_BUILTINS))

#: concrete calls: the executor returns what CPython returns, or refuses
CALLS = [
    "len([1, 2, 3])", "len('abc')", "len((1, 2))", "len({1: 2, 3: 4})", "len([])", "len(5)", "len(None)",
    "abs(-3)", "abs(3)", "abs(-2.5)", "abs(True)", "abs(-0.0)", "abs('a')",
    "min(3, 1)", "min(1, 3)", "min(2, 2.0)", "min(2.0, 2)", "max(3, 1)", "max(1, 3)", "max(2, 2.0)", "max(2.0, 2)",
    "min([3, 1, 2])", "max([3, 1, 2])", "min([])", "min([], default=7)", "max([2.5, 1])", "min(1.0, True)",
    "any([0, 0])", "any([0, 1])", "all([1, 1])", "all([1, 0])", "any([])", "all([])", "all([1, 'a'])", "any([0, ''])",
    "isinstance(3, int)", "isinstance(True, int)", "isinstance(3, bool)", "isinstance(2.5, float)", "isinstance('a', str)",
    "isinstance(3, (str, int))", "isinstance([1], list)", "isinstance((1,), tuple)", "isinstance(3, float)",
    "sum([1, 2, 3])", "sum([])", "sum([1, 2], 10)", "sum([1.5, 2.5])", "sum([0.1] * 10)", "sum([0.1, 0.2, 0.3])",
    "sum([1e100, 1.0, -1e100])", "sum([True, True])", "sum([0.1] * 3, 1)", "sum([-0.0])", "sum([], -0.0)",
    "range(3)", "range(1, 8, 2)", "range(5, 0, -2)", "range(0)", "range(2.5)",
    "sorted([3, 1, 2])", "sorted((2, 1))", "sorted([3, 1], reverse=True)", "sorted([1, 'a'])", "sorted({3: 1, 1: 2})",
    "int('5')", "int(' 7 ')", "int('x')", "int(2.7)", "int(-2.7)", "int(True)", "int(5)", "int(b'12')", "int(float('inf'))",
    "float(3)", "float('2.5')", "float(True)", "float(2.5)", "float('inf')", "float('x')", "float(10 ** 400)",
    "divmod(7, 2)", "divmod(-7, 2)", "divmod(7, -2)", "divmod(-7, -2)", "divmod(7.5, 2)", "divmod(-7.5, 2)", "divmod(7, 0)",
    "divmod(7.0, 0.0)", "divmod(True, 2)", "divmod('a', 2)", "divmod(0, 5)",
    "pow(2, 10)", "pow(2, -1)", "pow(2.0, 3)", "pow(0, 0)", "pow(-8, 1)", "pow(0, -1)", "pow(2, 0.5)", "pow(-8, 0.5)", "pow(True, 2)",
    "print(1)", "print(1, 'a', sep='-')", "print(1, end='')", "print(1, sep=1)", "print(1, file=None)",
]


def _executor():
    return _symexec._Exec({}, {}, {})


def _executor_runner():
    tree = ast.parse("def __sv_harness():\n    pass\n")
    return _symexec._Runner(tree, {}, {}, {}, [])


def _native(expr: str):
    """``(True, value)`` or ``(False, exception class)`` on CPython."""
    try:
        return True, eval(expr, {"__builtins__": __builtins__}, {})
    except BaseException as e:  # noqa: BLE001
        return False, type(e)


def _same(a, b) -> bool:
    if isinstance(b, float) and isinstance(a, float):
        return a == b and math.copysign(1, a) == math.copysign(1, b) or (math.isnan(a) and math.isnan(b))
    if isinstance(b, (list, tuple)) and type(a) is type(b):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


@pytest.mark.parametrize("make", [_executor, _executor_runner], ids=["exec", "runner"])
@pytest.mark.parametrize("expr", CALLS)
def test_the_executor_returns_what_cpython_returns_or_refuses(expr, make):
    ok, native = _native(expr)
    ex = make()
    node = ast.parse(expr, mode="eval").body
    try:
        got = ex.ev(node)
    except (_symexec.Unsupported, _symexec._Dead):
        return                          # a refusal says nothing false
    assert ok, (expr, "CPython raises", native, "but the executor answered", got)
    if isinstance(native, range):
        native = list(native)
    if isinstance(got, _symexec._LazyIter):
        got = list(got)
    assert _same(got, native), (expr, got, native)


def test_the_table_has_calls_that_the_executor_answers_for_every_builtin():
    """The differential table is not vacuous: each builtin has at least one call the executor really answers."""
    answered = set()
    for expr in CALLS:
        try:
            _executor_runner().ev(ast.parse(expr, mode="eval").body)
        except (_symexec.Unsupported, _symexec._Dead):
            continue
        answered.add(expr.split("(")[0])
    assert set(BUILTINS) <= answered, set(BUILTINS) - answered


def test_sum_of_floats_is_cpythons_compensated_sum_not_the_left_fold():
    """CPython 3.12+ adds floats with Neumaier compensation: the left fold the executor used drifts."""
    assert sum([0.1] * 10) == 1.0 and not (0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 == 1.0)
    got = _executor_runner().ev(ast.parse("sum([0.1] * 10)", mode="eval").body)
    assert got == 1.0


def test_negative_control_the_left_fold_sum_is_caught_by_the_differential(monkeypatch):
    """The control: the model before this change (``total + v`` over the elements) answers ``0.9999999999999999`` for
    ``sum([0.1] * 10)`` where CPython answers ``1.0``: the differential above fails for it."""
    def left_fold(self, n):
        vals = self._iterate_expr(n.args[0])
        total = _symexec._num(self.ev(n.args[1]) if len(n.args) == 2 else 0)
        for v in vals:
            total = total + _symexec._num(v)
        return total

    monkeypatch.setattr(_symexec._Exec, "_model_sum", left_fold)
    got = _executor_runner().ev(ast.parse("sum([0.1] * 10)", mode="eval").body)
    assert not _same(got, sum([0.1] * 10)), got


# --------------------------------------------------------------------------- the contract admits the call
def _answers(expr: str) -> bool:
    try:
        _executor_runner().ev(ast.parse(expr, mode="eval").body)
        return True
    except (_symexec.Unsupported, _symexec._Dead):
        return False


@pytest.mark.parametrize("name", BUILTINS)
def test_an_untested_contract_is_a_refusal_not_the_model_of_old(name, monkeypatch):
    expr = next(e for e in CALLS if e.split("(")[0] == name and _native(e)[0] and _answers(e))
    load_all()
    monkeypatch.setitem(REGISTRY, name, untested(REGISTRY[name]))
    assert not _answers(expr), (name, expr)


@pytest.mark.parametrize("name", BUILTINS)
def test_a_withdrawn_contract_is_a_refusal_too(name, monkeypatch):
    expr = next(e for e in CALLS if e.split("(")[0] == name and _native(e)[0] and _answers(e))
    load_all()
    monkeypatch.delitem(REGISTRY, name)
    assert not _answers(expr), (name, expr)


# --------------------------------------------------------------------------- symbolic: solved, and replayed on CPython
#: (builtin, the sample with a draw ``v``, the guard on what it computed): a guard on the value CPython gives for the value the solver picks
SYMBOLIC = [
    ("abs", "x = abs(v - 7)\n", "x == 4"),
    ("min", "x = min(v, 5)\n", "x == 2"),
    ("max", "x = max(v, 5)\n", "x == 9"),
    ("sum", "x = sum([v, 1, 2])\n", "x == 6"),
    ("divmod", "q, r = divmod(v, 4)\n", "q == 2 and r == 3"),
    ("pow", "x = pow(v, 2)\n", "x == 49"),
    ("int", "x = int(v)\n", "x == 7"),
    ("isinstance", "x = isinstance(v, int)\n", "x"),
]
_DRAW = "import _sv_verifier\nv = _sv_verifier.nondet_int()\n"


@pytest.mark.parametrize("name,src,guard", SYMBOLIC, ids=[c[0] for c in SYMBOLIC])
def test_a_guard_on_a_symbolic_call_is_solved_and_replays(name, src, guard):
    got, h = _solve(_DRAW + src, guard)
    sat = [m for o, m in got.values() if o == "sat"]
    assert sat and sat[0]["n"] == 3, (name, got)
    vector = [sat[0][p] for p in h.params]
    full = (_DRAW + src + "import _sv_verifier\nn = _sv_verifier.nondet_int()\n"
            f"if {guard} and n == 3:\n    raise IndexError('hit')\n")
    assert run_native(full, vector) == "IndexError", (name, vector)


@pytest.mark.parametrize("name,src,guard", [("abs", "x = abs(v - 7)\n", "x == -4"), ("min", "x = min(v, 5)\n", "x == 6"),
                                            ("max", "x = max(v, 5)\n", "x == 4"), ("divmod", "q, r = divmod(v, 4)\n", "r == 4")],
                         ids=["abs", "min", "max", "divmod"])
def test_a_guard_on_a_value_cpython_cannot_give_is_not_reached(name, src, guard):
    got, _h = _solve(_DRAW + src, guard)
    assert not [m for o, m in got.values() if o == "sat"], (name, got)


def test_a_print_is_answered_but_the_run_is_not_exact():
    """``print`` is modelled (the statement is not skipped), but what it writes is not tracked: a search does not mind, the exact product
    proof (``run_exact``) must: the run is inexact and says why."""
    from tharkun.sv._symexec import run_exact

    tree = ast.parse("print(1)\n")
    r = run_exact(tree, {}, {}, list(tree.body), [])
    assert r["exact"] is False and "print" in r["reason"], r
    runner = _executor_runner()
    runner.run(list(tree.body))
    assert not runner.skipped and runner.inexact, (runner.skipped, runner.inexact)
