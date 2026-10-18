"""Case splits on a boolean draw (docs/ABSINT.md "Case splits on a boolean draw").

``prove`` runs the joined analysis first and, only when that does not prove TRUE, runs it again with every
``_sv_verifier.nondet_bool()`` of an eligible statement case-split.  The programs under
``tests/fixtures/absint/draw_split`` are the adversarial set: each one is run natively over EVERY draw vector (the
draws are booleans, so the draw tree is finite or cut at ``MAX_DRAWS``), and a TRUE for a property class is a failure
whenever some vector makes an exception of that class escape.
"""
from __future__ import annotations

import ast
import contextlib
import io
import pathlib
import sys
import types

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.absint import interp
from tharkun.solve.absint.interp import Analyzer
from tharkun.solve.environment import TaskEnvironment
from tharkun.solve.obligations import source_points

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "absint" / "draw_split"
PROGRAMS = sorted(p.stem for p in FIXTURES.glob("ds*.py"))
CLASSES = {
    "dynamic-typing": ("TypeError", "AttributeError"),
    "arithmetic": ("ZeroDivisionError", "FloatingPointError"),
    "data-lookup": ("IndexError", "KeyError"),
    "assertion-safety": ("AssertionError",),
}
MAX_DRAWS = 12      # an unbounded draw loop is cut here; an escape found below the cut is real


def _stub(vector: list, used: list) -> types.ModuleType:
    """A stand-in ``_sv_verifier`` whose booleans are ``vector`` (False past its end) and whose check_type is isinstance."""
    mod = types.ModuleType("_sv_verifier")

    def nondet_bool():
        i = len(used)
        used.append(True)
        return vector[i] if i < len(vector) else False

    def check_type(value, hint):
        if not isinstance(value, hint):
            raise TypeError(f"expected {hint!r}")

    mod.nondet_bool, mod.check_type = nondet_bool, check_type
    return mod


def native_escapes(source: str) -> set:
    """Exception class names that escape some run of ``source`` over the booleans it draws."""
    escapes: set = set()
    seen = set()
    todo = [()]
    while todo:
        prefix = todo.pop()
        if prefix in seen:
            continue
        seen.add(prefix)
        used: list = []
        saved = sys.modules.get("_sv_verifier")
        sys.modules["_sv_verifier"] = _stub(list(prefix), used)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, "<draw_split>", "exec"), {"__name__": "__main__"})
        except BaseException as exc:  # noqa: BLE001 -- how the run ended is the point
            escapes.add(type(exc).__name__)
        finally:
            if saved is None:
                sys.modules.pop("_sv_verifier", None)
            else:
                sys.modules["_sv_verifier"] = saved
        for j in range(len(prefix), min(len(used), MAX_DRAWS)):
            todo.append(prefix + (False,) * (j - len(prefix)) + (True,))
    return escapes


def _source(name: str) -> str:
    return (FIXTURES / f"{name}.py").read_text()


def _prove(name: str, classes, **kw):
    path = FIXTURES / f"{name}.py"
    return prove(path.read_text(), classes, filename=str(path), time_budget=30.0,
                 environment=TaskEnvironment((str(path),)), **kw)


@pytest.mark.parametrize("name", PROGRAMS)
def test_never_true_where_a_draw_vector_escapes(name):
    escapes = native_escapes(_source(name))
    for prop, classes in CLASSES.items():
        for mode in (None, True):
            result = _prove(name, classes, split_draws=mode)
            if set(classes) & escapes:
                assert result.verdict != "TRUE", (name, prop, mode, escapes, result.explain())


@pytest.mark.parametrize("budget", [0, 1, 2, 5])
@pytest.mark.parametrize("name", ["ds03_two_draws_independent", "ds04_two_draws_one_statement",
                                  "ds15_flag_flipped_between", "ds20_global_redrawn", "ds23_eight_draws",
                                  "ds24_three_draws_ok", "ds34_inner_draw_in_function",
                                  "ds48_split_inside_called_in_loop"])
def test_a_spent_budget_leaves_the_rest_joined_and_sound(monkeypatch, name, budget):
    """Draws past the budget are analysed joined, as before: sound at every budget, never a partial partition."""
    monkeypatch.setattr(interp, "MAX_DRAW_SPLITS", budget)
    escapes = native_escapes(_source(name))
    for prop, classes in CLASSES.items():
        result = _prove(name, classes, split_draws=True)
        assert result.draw_splits <= budget
        if set(classes) & escapes:
            assert result.verdict != "TRUE", (name, prop, budget, escapes, result.explain())


@pytest.mark.parametrize("name", ["ds01_flag_correlated_ok", "ds13_alias_import_ok", "ds24_three_draws_ok",
                                  "ds32_draw_in_called_function_ok"])
def test_the_retry_proves_what_the_joined_run_cannot(name):
    assert native_escapes(_source(name)) == set()
    for prop, classes in CLASSES.items():
        joined = _prove(name, classes, split_draws=False)
        retried = _prove(name, classes)
        assert joined.verdict != "TRUE", (name, prop, joined.explain())
        assert retried.verdict == "TRUE", (name, prop, retried.explain())
        assert retried.draw_splits >= 1
        assert "case splits on a boolean draw" in retried.explain()


def test_the_retry_reports_the_joined_run_when_it_does_not_prove():
    name = "ds23_eight_draws"          # the all-True vector fails the assertion
    classes = CLASSES["assertion-safety"]
    assert "AssertionError" in native_escapes(_source(name))
    joined = _prove(name, classes, split_draws=False)
    retried = _prove(name, classes)
    assert retried.verdict == joined.verdict == "UNKNOWN"
    assert retried.reason == joined.reason and retried.draw_splits == 0


@pytest.mark.parametrize("name", ["ds43_one_partition_returns", "ds45_one_partition_raises",
                                  "ds47_return_values_differ", "ds48_split_inside_called_in_loop",
                                  "ds50_exception_in_method_then_rest"])
def test_an_exit_of_one_outcome_is_still_seen(name):
    """A return or raise in only one outcome of the rest of the block reaches the caller's joined result."""
    escapes = native_escapes(_source(name))
    assert escapes
    for prop, classes in CLASSES.items():
        if set(classes) & escapes:
            assert _prove(name, classes, split_draws=True).verdict != "TRUE", (name, prop)


@pytest.mark.parametrize("name,prop", [("ds44_one_partition_returns_ok", "dynamic-typing"),
                                       ("ds46_one_partition_raises_other_class", "assertion-safety"),
                                       ("ds49_split_in_called_in_loop_ok", "assertion-safety")])
def test_an_exit_of_one_outcome_does_not_hide_a_proof(name, prop):
    assert not (set(CLASSES[prop]) & native_escapes(_source(name)))
    assert _prove(name, CLASSES[prop]).verdict == "TRUE"


# ---------------------------------------------------------------- the force is the split statement's own
class _Probe(Analyzer):
    """An analyzer that watches the force: what a callee sees, and every write to a memo."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pending_at_call: list = []
        self.pending_in_callee: list = []
        self.writes_while_pending: list = []
        outer = self

        class Guarded(dict):
            def __setitem__(self, k, v):
                outer.writes_while_pending.append(bool(outer._forced_draws))
                super().__setitem__(k, v)

            def setdefault(self, k, d=None):
                if k not in self:
                    outer.writes_while_pending.append(bool(outer._forced_draws))
                return super().setdefault(k, d)

        self._fix_cache = Guarded()
        self.class_contracts = Guarded()

    def call_func(self, st, faddr, args, kwargs, node, **kw):
        self.pending_at_call.append(len(self._forced_draws))
        return super().call_func(st, faddr, args, kwargs, node, **kw)

    def exec_block(self, stmts, st, fctx, lctx):
        if not fctx.is_module and not fctx.is_class:
            self.pending_in_callee.append(len(self._forced_draws))
        return super().exec_block(stmts, st, fctx, lctx)


def _run_probe(source: str, **kw) -> _Probe:
    from tharkun.solve.absint.errors import Unsupported
    an = _Probe(source, "<probe>", time_budget=30.0, split_draws=True, **kw)
    try:
        an.run()
    except Unsupported:
        pass    # a refusal ends the run; what the force did until then is what is checked
    return an


def test_a_callee_never_sees_the_pending_force_and_nothing_is_memoized_meanwhile():
    # callee() runs INSIDE the split statement, before its draw is evaluated, and again afterwards; its own draw is
    # another node, its loop is a fixpoint, and the second call decides a check
    source = _source("ds41_callee_draw_before_force")
    an = _run_probe(source)
    assert any(n > 0 for n in an.pending_at_call), "the first callee() must run with the force pending"
    assert all(n == 0 for n in an.pending_in_callee)
    assert not any(an.writes_while_pending)
    assert an.splits_done >= 1


@pytest.mark.parametrize("name", PROGRAMS)
def test_no_memo_entry_is_written_while_a_force_is_pending(name):
    an = _run_probe(_source(name))
    assert all(n == 0 for n in an.pending_in_callee), name
    assert not any(an.writes_while_pending), name


def test_a_draw_in_a_callee_is_not_the_split_statements_draw():
    """The same function reached twice: each activation splits (or not) on its own draws."""
    name = "ds34_inner_draw_in_function"
    assert "AssertionError" in native_escapes(_source(name))
    assert _prove(name, CLASSES["assertion-safety"], split_draws=True).verdict != "TRUE"
    assert _prove(name, CLASSES["dynamic-typing"], split_draws=True).verdict == "TRUE"


# ---------------------------------------------------------------- invariants come from unsplit states only
def test_invariants_are_never_taken_inside_the_outcomes_of_a_split():
    source = ("import _sv_verifier\n\nx = 0\nflag = _sv_verifier.nondet_bool()\ny = 5 if flag else 7\n"
              "z = y + 1\nassert z >= 6\n")
    points = source_points(ast.parse(source), source, "p.py")
    locations = {entry[1] for entry in points.values()}
    joined = Analyzer(source, "p.py", split_draws=False, invariant_locations=locations)
    joined.run()
    split = Analyzer(source, "p.py", split_draws=True, invariant_locations=locations)
    split.run()
    assert split.splits_done == 1
    lines = lambda an: {loc[1] for (loc, _kind), parts in an.invariants.items() if parts}   # noqa: E731
    assert lines(joined) == {4, 5, 6, 7}
    # the entry of the split statement (line 4: a single state) is kept; the lines whose entries are the outcomes' are not
    assert lines(split) == {4}
    for (loc, _kind), parts in split.invariants.items():
        assert not any(p.startswith(("y >=", "y <=", "z >=", "z <=")) for p in parts)


def test_result_records_keep_only_facts_that_hold_in_every_outcome():
    source = ("import _sv_verifier\n\nflag = _sv_verifier.nondet_bool()\nd = [1] if flag else [2, 3]\n"
              "n = len(d)\nx = d[0]\ny = 10 // x\n")
    result = prove(source, CLASSES["arithmetic"] + CLASSES["data-lookup"], split_draws=True)
    assert result.verdict == "TRUE", result.explain()
    for record in result.records:
        for fact in record["scope"]:
            assert "== 1" not in fact and "== 2" not in fact, record
