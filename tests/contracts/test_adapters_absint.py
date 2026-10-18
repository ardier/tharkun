"""Tests for the Engine 2 (static analyser) adapter ([internal ref] step 2).

``solve/absint/interp.py``'s own ``MODULES`` table names only math/random/
_sv_verifier; any other call currently makes the WHOLE proof abort with
UNKNOWN (``Unsupported`` is caught once, at the top of ``prove()``). The
worked example, ``heapq.heappop``, is exactly such a call -- so it doubles
as the clearest demonstration that consulting contracts turns a total abort
into a specific, honest, per-class check.

2026-09-28 owner direction: no feature flag -- this engine is always on.
Correctness now rests on three things tested here: applicability (a claim
is trusted only for a call whose exact shape was hand-verified), applying
(or conservatively havocing) a mutation effect instead of leaving a stale
abstract list, and never trusting a declared return type when the contract
may run an unmodelled callback.
"""

from __future__ import annotations

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.contracts.adapters import common as _common
from tharkun.solve.contracts.adapters.absint import (
    applicable_call_shape, contract_for_module_func, raise_checks, simple_return_kind,
)
from tharkun.solve.contracts.rules import RULES, Rule, register_rule
from tharkun.solve.absint.visitor import AbsintVisitor
from tharkun.solve.contracts.registry import REGISTRY
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim, ReturnSpec


def _install(contract):
    class Synthetic(Rule):
        names = (contract.qualname,)
        CONTRACT = contract
        kind = "synthetic_" + contract.qualname.replace(".", "_")
    register_rule(Synthetic)
    setattr(AbsintVisitor, "visit_" + Synthetic.kind, AbsintVisitor._module_function)


def _register(qualname, *, mutates_receiver=False, may_execute_callback=False, returns=None):
    c = Contract(
        qualname=qualname,
        raises=tuple(RaiseClaim(e, "never", tested=True) for e in ALL_PROPERTY_CLASSES),
        may_raise_other=False,
        returns=returns,
        effects=Effects(mutates_receiver=mutates_receiver, may_execute_callback=may_execute_callback),
        source=("synthetic",),
        tested=True,
    )
    _install(c)
    return c


HEAPPOP_SRC = "import heapq\nh = [3, 1, 2]\nheapq.heapify(h)\nx = heapq.heappop(h)\n"


class TestRaiseChecks:
    def test_heapq_heappop_worked_example(self):
        contract = contract_for_module_func("heapq", "heappop")
        assert contract is not None
        checks = dict((exc, (ok, fact)) for exc, ok, fact in raise_checks(contract))
        assert set(checks) == set(ALL_PROPERTY_CLASSES)
        ok, fact = checks["IndexError"]
        assert ok is False  # unproven: this class really can happen
        assert "the heap (list) is empty" in fact
        # KeyError is untouched by heappop's own logic AND not a dunder-propagation
        # target of __lt__ -- IndexError is the only class it names outside "propagates".
        assert all(exc in checks for exc in ("KeyError", "TypeError", "ZeroDivisionError"))

    def test_a_tested_never_claim_is_proven(self):
        contract = Contract(
            qualname="math._test_never_fn",
            raises=tuple(RaiseClaim(e, "never", tested=True) for e in ALL_PROPERTY_CLASSES),
            may_raise_other=False,
            returns=None,
            effects=Effects(),
            source=("synthetic",),
            tested=True,
        )
        checks = dict((exc, ok) for exc, ok, _ in raise_checks(contract))
        assert all(ok is True for ok in checks.values())

    def test_negative_control_an_untested_never_claim_is_not_proven(self):
        # RaiseClaim.condition == "never" but tested=False (Contract-level tested
        # must then be False too, __post_init__ enforces this): the TEXT says
        # "never", but the gate is `tested`, not the words -- so this must NOT
        # come out as a proven check, or a wrong/unfinished contract could make
        # absint claim TRUE where it should say UNKNOWN.
        contract = Contract(
            qualname="math._test_unfinished_fn",
            raises=tuple(RaiseClaim(e, "never", tested=False) for e in ALL_PROPERTY_CLASSES),
            may_raise_other=False,
            returns=None,
            effects=Effects(),
            source=("synthetic",),
            tested=False,
        )
        checks = dict((exc, ok) for exc, ok, _ in raise_checks(contract))
        assert all(ok is False for ok in checks.values())


class TestSimpleReturnKind:
    def test_float_str_bool_pass_through(self):
        mk = lambda t: Contract(
            qualname="x", raises=tuple(RaiseClaim(e, "never", tested=True) for e in ALL_PROPERTY_CLASSES),
            may_raise_other=False, returns=ReturnSpec(t, tested=True),
            effects=Effects(may_execute_callback=False),
            source=("s",), tested=True,
        )
        assert simple_return_kind(mk("float")) == "float"
        assert simple_return_kind(mk("str")) == "str"
        assert simple_return_kind(mk("bool")) == "bool"

    def test_int_and_untested_returns_are_left_alone(self):
        mk = lambda t, tested: Contract(
            qualname="x", raises=tuple(RaiseClaim(e, "never", tested=True) for e in ALL_PROPERTY_CLASSES),
            may_raise_other=False, returns=ReturnSpec(t, tested=tested),
            effects=Effects(may_execute_callback=False),
            source=("s",), tested=tested,
        )
        assert simple_return_kind(mk("int", True)) is None
        assert simple_return_kind(mk("float", False)) is None

    def test_a_callback_capable_contract_never_yields_a_return_type(self):
        # 2026-09-28 owner review: "never trust a declared return type when
        # effects.may_execute_callback is True -- any object can come back."
        # Even a TESTED float return must not be trusted when the callable can
        # run unmodelled code (a key=/default= callback, a dunder).
        c = Contract(
            qualname="x", raises=tuple(RaiseClaim(e, "never", tested=True) for e in ALL_PROPERTY_CLASSES),
            may_raise_other=True, returns=ReturnSpec("float", tested=True),
            effects=Effects(may_execute_callback=True),
            source=("s",), tested=True,
        )
        assert simple_return_kind(c) is None


class TestApplicableCallShape:
    def test_exact_shape_is_applicable(self):
        assert applicable_call_shape("heapq.heappop", 1, False) is True

    def test_wrong_arity_is_not_applicable(self):
        assert applicable_call_shape("heapq.heappop", 2, False) is False
        assert applicable_call_shape("heapq.heappop", 0, False) is False

    def test_any_keywords_are_not_applicable(self):
        assert applicable_call_shape("heapq.heappop", 1, True) is False

    def test_an_unlisted_qualname_is_never_applicable(self):
        assert applicable_call_shape("heapq.heapreplace", 2, False) is False


class TestAbsintWiring:
    def test_does_not_abort_and_reports_the_precise_condition(self):
        r = prove(HEAPPOP_SRC, ("IndexError",))
        assert r.verdict == "UNKNOWN"  # still unproven -- the heap MIGHT be empty
        assert "unsupported" not in r.reason
        assert any("heapq.heappop" in s[2] for s in r.sites)

    def test_a_provably_nonempty_list_proves_true(self):
        # 2026-09-28 review finding, fixed: absint's OWN interval reasoning
        # (the same len_lin/lin_bounds machinery list.pop() already uses) now
        # decides the contract's precondition when the state already knows
        # enough, instead of always recording a blanket unproven check.
        r = prove("import heapq\nh = [3, 1, 2]\nx = heapq.heappop(h)\n", ("IndexError",))
        assert r.verdict == "TRUE"

    def test_applies_the_receiver_length_effect_not_just_the_precondition(self):
        # A length-1 list: the FIRST pop is provably safe, but the analysis
        # must not forget the effect (length -1) -- the SECOND pop must NOT be
        # provable safe, or a real double-pop-on-one-element bug would be
        # missed.
        r = prove("import heapq\nh = [5]\nx = heapq.heappop(h)\ny = heapq.heappop(h)\n", ("IndexError",))
        assert r.verdict == "UNKNOWN"
        by_line = {s[0]: s for s in r.sites}
        assert "UNPROVEN" not in by_line[3][4]
        assert "UNPROVEN" in by_line[4][4]

    def test_two_pops_from_a_len_3_list_both_prove_true(self):
        r = prove("import heapq\nh = [5, 3, 8]\nx = heapq.heappop(h)\ny = heapq.heappop(h)\n",
                   ("IndexError",))
        assert r.verdict == "TRUE"

    def test_regression_a_different_property_class_at_the_same_site_does_not_drag_down_this_one(self):
        # 2026-09-28: every property class was recorded under the SAME check
        # key (the bare qualname), so one class's unprovenness silently
        # dragged down every OTHER class checked at the same call site --
        # defeating the precise IndexError proof above the moment any other
        # class (e.g. the __lt__ propagation classes heapq.heappop always
        # carries) was also requested. Now fixed by keying each class
        # separately; this pins it so it can't quietly regress.
        r = prove("import heapq\nh = [3, 1, 2]\nx = heapq.heappop(h)\n",
                   ("IndexError", "AssertionError"))
        assert r.verdict == "UNKNOWN"  # AssertionError genuinely unproven (via __lt__)
        rel = [s for s in r.sites if "IndexError" in s[3]]
        assert rel and all("UNPROVEN" not in s[4] for s in rel)

    def test_negative_control_an_untracked_list_is_not_wrongly_proven_safe(self):
        # A slice by a nondet-controlled bound: absint's own length reasoning
        # cannot pin this list's length down to >= 1, so this must stay exactly
        # as conservative as before -- no false TRUE from guessing.
        r = prove(
            "import heapq\nimport _sv_verifier\nn = _sv_verifier.nondet_int()\n"
            "h = [1, 2, 3][:n]\nx = heapq.heappop(h)\n",
            ("IndexError",),
        )
        assert r.verdict == "UNKNOWN"
        assert any("UNPROVEN" in s[4] and "IndexError" in s[3] for s in r.sites)

    def test_a_safe_call_proves_true(self):
        _register("math._test_harmless", returns=ReturnSpec("float", tested=True))
        try:
            r = prove("import math\nx = math._test_harmless(1)\n", ("IndexError", "AssertionError"))
            assert r.verdict == "TRUE"
        finally:
            del REGISTRY["math._test_harmless"]
            del RULES["math._test_harmless"]

    def test_negative_control_a_wrong_untested_contract_cannot_manufacture_true(self):
        list(map(_install, [
            Contract(
                qualname="math._test_unfinished",
                raises=tuple(RaiseClaim(e, "never", tested=False) for e in ALL_PROPERTY_CLASSES),
                may_raise_other=False,
                returns=None,
                effects=Effects(),
                source=("synthetic",),
                tested=False,
            )
        ]))
        try:
            r = prove("import math\nx = math._test_unfinished(1)\n", ("IndexError",))
            assert r.verdict == "UNKNOWN"
        finally:
            del REGISTRY["math._test_unfinished"]
            del RULES["math._test_unfinished"]


class TestApplicabilityGate:
    """2026-09-28 owner review: 'check applicability... arity, keyword names'
    before trusting ANY precise fact -- including a bare TESTED 'never raises
    TypeError' claim, since Python itself raises TypeError for a call whose
    shape the contract never accounted for, before the callable's own logic
    (what the claim actually describes) ever runs.

    A real stdlib contract can't isolate this: every currently-registered
    ``mutates_receiver`` module-function contract already carries an honestly
    untested (non-``"never"``) claim for every class via ``__lt__``
    propagation, which masks the gate below it regardless of arity. A
    synthetic, cleanly-``"never"``-claimed contract isolates the mechanism.
    """

    def test_a_tested_never_typeerror_claim_is_trusted_at_the_verified_shape(self, monkeypatch):
        monkeypatch.setitem(_common.CALL_SHAPE, "math._test_strict", 1)
        _register("math._test_strict")
        try:
            r = prove("import math\nx = math._test_strict(1)\n", ("TypeError",))
            assert r.verdict == "TRUE"
        finally:
            del REGISTRY["math._test_strict"]
            del RULES["math._test_strict"]

    def test_a_tested_never_typeerror_claim_is_not_trusted_when_unshaped(self):
        # Not in CALL_SHAPE at all -- conservative by default, whatever a
        # contract's own claim says.
        _register("math._test_unshaped")
        try:
            r = prove("import math\nx = math._test_unshaped(1)\n", ("TypeError",))
            assert r.verdict == "UNKNOWN"
            assert any("call shape not hand-verified" in s[4] for s in r.sites)
        finally:
            del REGISTRY["math._test_unshaped"]
            del RULES["math._test_unshaped"]

    def test_wrong_arity_downgrades_an_otherwise_trusted_claim(self, monkeypatch):
        monkeypatch.setitem(_common.CALL_SHAPE, "math._test_strict2", 1)
        _register("math._test_strict2")
        try:
            r = prove("import math\nx = math._test_strict2(1, 2)\n", ("TypeError",))
            assert r.verdict == "UNKNOWN"
            assert any("call shape not hand-verified" in s[4] for s in r.sites)
        finally:
            del REGISTRY["math._test_strict2"]
            del RULES["math._test_strict2"]

    def test_any_keywords_downgrade_an_otherwise_trusted_claim(self, monkeypatch):
        monkeypatch.setitem(_common.CALL_SHAPE, "math._test_strict3", 1)
        _register("math._test_strict3")
        try:
            r = prove("import math\nx = math._test_strict3(n=1)\n", ("TypeError",))
            assert r.verdict == "UNKNOWN"
        finally:
            del REGISTRY["math._test_strict3"]
            del RULES["math._test_strict3"]

    def test_keywords_also_defeat_a_real_contracts_precise_precondition(self):
        # heapq.heappop itself: applicable_call_shape gates empty_first_arg_
        # precondition the same way, and this one IS observable end to end
        # since IndexError's claim is native+tested (not masked by __lt__
        # propagation the way TypeError/AssertionError/etc. are).
        r = prove("import heapq\nh = [1]\nx = heapq.heappop(h)\n", ("IndexError",))
        assert r.verdict == "TRUE"
        r2 = prove("import heapq\nh = [1]\nx = heapq.heappop(heap=h)\n", ("IndexError",))
        assert r2.verdict == "UNKNOWN"  # keyword call: never matches CALL_SHAPE


class TestReceiverEffects:
    """2026-09-28 owner review: 'apply (or conservatively havoc) mutation
    EFFECTS on arguments and receivers... a stale abstract list is unsound.'"""

    def test_a_mutates_receiver_contract_without_a_precise_entry_conservatively_havocs(self):
        # STALE (the bug) would leave h's tracked length at its literal bound
        # (3, still >= 1), so a later pop would WRONGLY prove TRUE even though
        # the mutator could have emptied h. Havoc must prevent that.
        _register("math._test_mutator", mutates_receiver=True)
        try:
            r = prove(
                "import math\nh = [1, 2, 3]\nmath._test_mutator(h)\n"
                "import heapq\nx = heapq.heappop(h)\n",
                ("IndexError",),
            )
            assert r.verdict == "UNKNOWN"
            assert any("UNPROVEN" in s[4] and "IndexError" in s[3] for s in r.sites)
        finally:
            del REGISTRY["math._test_mutator"]
            del RULES["math._test_mutator"]

    def test_a_precise_length_effect_lets_a_later_check_prove_true(self, monkeypatch):
        # heapq.heappush/bisect.insort's OWN contracts always carry an untested
        # __lt__-propagation claim for every class (real user code can raise
        # anything), which masks this effect from prove()'s aggregate verdict
        # for the real contracts. A synthetic push-shaped contract with clean
        # "never" claims isolates the length-effect mechanism itself.
        monkeypatch.setitem(_common.CALL_SHAPE, "math._test_push", 2)
        monkeypatch.setitem(_common.RECEIVER_LENGTH_EFFECT, "math._test_push", (0, 1, 1))
        _register("math._test_push", mutates_receiver=True, returns=ReturnSpec("None", tested=True))
        try:
            r = prove(
                "import math\nh = []\nmath._test_push(h, 1)\nimport heapq\nx = heapq.heappop(h)\n",
                ("IndexError",),
            )
            assert r.verdict == "TRUE"
        finally:
            del REGISTRY["math._test_push"]
            del RULES["math._test_push"]

    def test_a_new_element_is_joined_into_the_receivers_element_facts(self, monkeypatch):
        # The precise mutation-effect regression: a length-only update (no
        # element join) leaves the list's element facts at BOT (empty list's
        # "nothing possible"), which can make a later read+use of the pushed
        # value look unreachable instead of the real TypeError it is -- a
        # WRONG TRUE. Confirmed both ways: the assertion below fails if
        # new_elem_idx (the 3rd tuple element) is dropped.
        monkeypatch.setitem(_common.CALL_SHAPE, "math._test_push", 2)
        monkeypatch.setitem(_common.RECEIVER_LENGTH_EFFECT, "math._test_push", (0, 1, 1))
        _register("math._test_push", mutates_receiver=True, returns=ReturnSpec("None", tested=True))
        try:
            r = prove(
                "import math\nh = []\nmath._test_push(h, 'oops')\nx = h[0]\ny = x + 1\n",
                ("TypeError",),
            )
            assert r.verdict == "UNKNOWN"
            assert any("str and int" in s[4] for s in r.sites)
        finally:
            del REGISTRY["math._test_push"]
            del RULES["math._test_push"]

    def test_negative_control_length_only_without_the_element_join_would_wrongly_prove_true(self, monkeypatch):
        # Pins the exact failure mode the previous test guards against: with
        # new_elem_idx=None (length-only, the shape this table used before the
        # 2026-09-28 fix), the same program WRONGLY proves TRUE.
        monkeypatch.setitem(_common.CALL_SHAPE, "math._test_push", 2)
        monkeypatch.setitem(_common.RECEIVER_LENGTH_EFFECT, "math._test_push", (0, 1, None))
        _register("math._test_push", mutates_receiver=True, returns=ReturnSpec("None", tested=True))
        try:
            r = prove(
                "import math\nh = []\nmath._test_push(h, 'oops')\nx = h[0]\ny = x + 1\n",
                ("TypeError",),
            )
            assert r.verdict == "TRUE"  # demonstrates the bug this table shape would reintroduce
        finally:
            del REGISTRY["math._test_push"]
            del RULES["math._test_push"]

    def test_a_non_mutating_contract_never_touches_the_receiver(self):
        _register("math._test_harmless_receiver", returns=ReturnSpec("float", tested=True))
        try:
            r = prove(
                "import math\nh = [1, 2, 3]\nmath._test_harmless_receiver(h)\n"
                "import heapq\nx = heapq.heappop(h)\n",
                ("IndexError",),
            )
            assert r.verdict == "TRUE"  # h was never claimed to be mutated -- stays precise
        finally:
            del REGISTRY["math._test_harmless_receiver"]
            del RULES["math._test_harmless_receiver"]
