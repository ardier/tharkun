"""Tests for the Engine 3 (concolic/z3 solver) adapter ([internal ref] step 2).

``sv/_symexec.py``'s whole-harness ``_Runner`` currently raises ``Unsupported``
(and the whole symbolic run aborts with an ``"unsupported"`` outcome at every
target line) for any call it doesn't specially recognise -- ``heapq.heappop``
and ``random.random`` included. The worked example (heapq.heappop) exercises
the PRECONDITION-becomes-branch-condition half; ``random.random`` exercises
the RETURN-becomes-z3-fact half the ticket names explicitly.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tharkun.solve.contracts.adapters import common as _common
from tharkun.solve.contracts.adapters.concolic import empty_first_arg_precondition, range_return
from tharkun.solve.contracts.registry import REGISTRY
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim, ReturnSpec
from tharkun.sv.sites import build_harness

pytest.importorskip("z3")


def _run(src, classes=("IndexError",)):
    from tharkun.sv import _symexec

    h = build_harness(textwrap.dedent(src), classes, None)
    mod = {"__name__": "sv_harness_test"}
    sys.modules.setdefault("_sv_verifier", type(sys)("_sv_verifier"))
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    return _symexec.solve_targets(h.source, mod, "__sv_harness", [s["line"] for s in h.sites],
                                   h.params, h.kinds, {})


class TestEmptyFirstArgPrecondition:
    def test_heapq_heappop_worked_example(self):
        assert empty_first_arg_precondition("heapq.heappop", arg_count=1, has_keywords=False) == "IndexError"

    def test_unknown_qualname_is_none(self):
        assert empty_first_arg_precondition("os.path.exists", arg_count=1, has_keywords=False) is None

    def test_wrong_arity_or_keywords_does_not_apply(self):
        # 2026-09-28 owner review: applicability (exact arity, no keywords) is
        # now checked before ANY precise fact, including this one.
        assert empty_first_arg_precondition("heapq.heappop", arg_count=2, has_keywords=False) is None
        assert empty_first_arg_precondition("heapq.heappop", arg_count=1, has_keywords=True) is None

    def test_negative_control_an_untested_contract_does_not_apply(self):
        REGISTRY.update((c.qualname, c) for c in [
            Contract(
                qualname="heapq._test_heappop_like",
                raises=tuple(RaiseClaim(e, "the heap (list) is empty" if e == "IndexError" else "never",
                                        tested=False) for e in ALL_PROPERTY_CLASSES),
                may_raise_other=False, returns=None, effects=Effects(),
                source=("synthetic",), tested=False,
            )
        ])
        try:
            from tharkun.solve.contracts.adapters import concolic as _c
            _c._EMPTY_FIRST_ARG_RAISES["heapq._test_heappop_like"] = "IndexError"
            _common.CALL_SHAPE["heapq._test_heappop_like"] = 1
            assert empty_first_arg_precondition(
                "heapq._test_heappop_like", arg_count=1, has_keywords=False) is None
        finally:
            del REGISTRY["heapq._test_heappop_like"]
            del _c._EMPTY_FIRST_ARG_RAISES["heapq._test_heappop_like"]
            del _common.CALL_SHAPE["heapq._test_heappop_like"]


class TestRangeReturn:
    def test_random_random_worked_example(self):
        assert range_return("random.random") == (0.0, 1.0, False)

    def test_unknown_qualname_is_none(self):
        assert range_return("random.randint") is None

    def test_negative_control_an_untested_contract_does_not_apply(self):
        REGISTRY.update((c.qualname, c) for c in [
            Contract(
                qualname="random._test_random_like",
                raises=tuple(RaiseClaim(e, "never", tested=True) for e in ALL_PROPERTY_CLASSES),
                may_raise_other=False, returns=ReturnSpec("float", "in [0, 1)", tested=False),
                effects=Effects(), source=("synthetic",), tested=False,
            )
        ])
        try:
            from tharkun.solve.contracts.adapters import concolic as _c
            _c._RANGE_RETURN["random._test_random_like"] = (0.0, 1.0, False)
            assert range_return("random._test_random_like") is None
        finally:
            del REGISTRY["random._test_random_like"]
            del _c._RANGE_RETURN["random._test_random_like"]


class TestConcolicWiring:
    HEAPPOP_REACHABLE = """
        import heapq
        import _sv_verifier

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            h = []
            if n > 0:
                h = [1] * n
            x = heapq.heappop(h)
            if x == 1:
                raise IndexError("reached")
    """

    HEAPPOP_ALWAYS_EMPTY = """
        import heapq
        import _sv_verifier

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            h = [1] * 0
            x = heapq.heappop(h)
            raise IndexError("reached")
    """

    RANDOM_OUT_OF_RANGE = """
        import random
        import _sv_verifier

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            r = random.random()
            if r >= 1.0:
                raise IndexError("unreachable if [0, 1) is respected")
    """

    def test_worked_example_finds_a_reaching_input(self):
        got = _run(self.HEAPPOP_REACHABLE)
        (outcome, model), = got.values()
        assert outcome == "sat"
        assert model["n"] > 0  # the precondition (len(h) > 0) had to be solved for

    def test_an_always_empty_heap_prunes_the_dead_path(self):
        # heappop on a heap that is ALWAYS empty raises before the site: the whole
        # path is dead, so the target must be unreached, never "sat" -- the
        # _Dead()/precondition mechanism must not be bypassed by this hook.
        got = _run(self.HEAPPOP_ALWAYS_EMPTY)
        (outcome, _reason), = got.values()
        assert outcome == "unreached"

    def test_random_random_range_proves_the_site_unreachable(self):
        # random.random() is TESTED to return a value in [0, 1) -- so r >= 1.0
        # must be proven UNSAT, not merely left unsupported.
        got = _run(self.RANDOM_OUT_OF_RANGE)
        (outcome, _reason), = got.values()
        assert outcome == "unsat"

    DOUBLE_POP_FIXED_LENGTH_1 = """
        import heapq
        import _sv_verifier

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            h = [1] * 1
            x = heapq.heappop(h)
            y = heapq.heappop(h)
            raise IndexError("reached only if BOTH pops succeed")
    """

    def test_applies_the_receiver_length_effect_not_just_the_precondition(self):
        # 2026-09-28 review finding: the first cut checked the precondition but
        # left the container's tracked length stale, so a SECOND pop against
        # the same (now-empty) heap was wrongly modelled as still possible.
        # A heap fixed at length 1 must make the second pop UNREACHED, proving
        # heappop's own effect (removes exactly one element) was applied, not
        # just its precondition checked once.
        got = _run(self.DOUBLE_POP_FIXED_LENGTH_1)
        (outcome, _reason), = got.values()
        assert outcome == "unreached"

    def test_a_heap_that_can_be_length_2_lets_both_pops_through(self):
        src = """
            import heapq
            import _sv_verifier

            if __name__ == "__main__":
                n = _sv_verifier.nondet_int()
                h = []
                if n > 0:
                    h = [1] * n
                x = heapq.heappop(h)
                y = heapq.heappop(h)
                raise IndexError("reached only if BOTH pops succeed")
        """
        got = _run(src)
        (outcome, model), = got.values()
        assert outcome == "sat"
        assert model["n"] >= 2  # one element is not enough for two pops
