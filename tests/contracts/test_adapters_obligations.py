"""Tests for the Engine 1 (Obligations) adapter ([internal ref] step 2, 2026-09-28 redesign).

The redesign's rule, in the ticket's own words: obligations are ONE format
(OB's witness-shape ``location_invariant`` with a checked ``python_expression``
``value``), never English strings, never sent to a solver as free text.
Anything the adapter cannot express that way is an explicit :class:`Gap`, never
silently dropped.
"""

from __future__ import annotations

import ast

from tharkun.solve import census
from tharkun.solve.contracts.adapters.obligations import (
    Gap, Invariant, Obligation, obligation_for_call,
)


def _call(src: str) -> ast.Call:
    """The sole top-level Call expression in ``src``."""
    tree = ast.parse(src)
    return next(n for n in ast.walk(tree) if isinstance(n, ast.Call))


class TestObligationForCallWorkedExample:
    def test_heapq_heappop_produces_a_checked_python_expression(self):
        node = _call("heapq.heappop(h)")
        ob = obligation_for_call("heapq.heappop", node, file_name="t.py", function="<module>")
        assert isinstance(ob, Obligation)
        assert ob.gap is None
        inv = ob.invariant
        assert isinstance(inv, Invariant)
        assert inv.value == "len(h) > 0"
        assert inv.type == "location_invariant"
        # The exact shape OB's own witness format expects.
        assert inv.entry() == {"invariant": {
            "type": "location_invariant",
            "location": {"file_name": "t.py", "line": 1, "column": 1, "function": "<module>"},
            "value": "len(h) > 0", "format": "python_expression"}}

    def test_the_actual_argument_text_is_substituted_not_a_fixed_name(self):
        # The call site's own variable name, not the template's placeholder name.
        node = _call("heapq.heappop(my_heap)")
        ob = obligation_for_call("heapq.heappop", node, file_name="t.py", function="f")
        assert ob.invariant.value == "len(my_heap) > 0"

    def test_negative_control_an_attribute_argument_cannot_be_expressed_and_gaps(self):
        # self._data is an Attribute -- outside the python_expression grammar
        # (no Attribute node allowed) -- must gap, never emit malformed Python.
        node = _call("heapq.heappop(self._data)")
        ob = obligation_for_call("heapq.heappop", node, file_name="t.py", function="pop")
        assert ob.invariant is None
        assert isinstance(ob.gap, Gap)
        assert "grammar" in ob.gap.reason

    def test_negative_control_wrong_arity_gaps_instead_of_guessing(self):
        node = _call("heapq.heappop(h, extra)")
        ob = obligation_for_call("heapq.heappop", node, file_name="t.py", function="f")
        assert ob.invariant is None
        assert "arity" in ob.gap.reason

    def test_negative_control_keyword_argument_gaps(self):
        node = ast.parse("heapq.heappop(h=h)").body[0].value
        ob = obligation_for_call("heapq.heappop", node, file_name="t.py", function="f")
        assert ob.invariant is None
        assert "arity" in ob.gap.reason


class TestObligationForCallCoverage:
    def test_a_contracted_qualname_without_a_template_becomes_a_gap_not_silence(self):
        # random.shuffle: real, tested IndexError-propagation evidence, but no
        # hand-verified safety template exists for it -- must still surface as
        # a gap, never just vanish.
        node = _call("random.shuffle(x)")
        ob = obligation_for_call("random.shuffle", node, file_name="t.py", function="<module>")
        assert ob is not None
        assert ob.invariant is None
        assert isinstance(ob.gap, Gap)
        assert ob.gap.qualname == "random.shuffle"
        assert "no hand-verified" in ob.gap.reason

    def test_a_qualname_with_no_contract_evidence_at_all_yields_none(self):
        node = _call("os.getcwd()")
        assert obligation_for_call("os.getcwd", node, file_name="t.py", function="<module>") is None

    def test_a_never_claim_yields_none_not_a_gap(self):
        # dict.clear is TESTED to never raise IndexError -- genuinely nothing to
        # say about this site for that class, so None (not a gap).
        node = _call("d.clear()")
        ob = obligation_for_call("dict.clear", node, file_name="t.py", function="<module>",
                                  classes=("IndexError",))
        assert ob is None


class TestRecordObligation:
    """``_record_obligation``/``Census.obligations`` on a call that IS a site
    under census's own (native-only, [internal ref]) site table -- the widening this
    file used to also test (``contract_call_sites``, a ``census.py``-side
    "add a builtin to the site table by contract" mechanism) was removed
    ([internal ref], 2026-09-28 owner review): every hostile construction it could
    ever catch is already caught by census's existing, unconditional scan of
    a user dunder's/method's own body (see ``adapters/obligations.py``'s
    module docstring and ``docs/SV.md``'s ``proof:no-site`` section for the
    argument and the empirical probe). It added no soundness, only
    imprecision, so there is no widening left for this class to exercise --
    only the (unaffected, still-valuable) obligation-recording plumbing."""

    def test_heappop_produces_a_real_checked_invariant_via_the_scan_plumbing(self):
        # heapq is not in census's MODULES allowlist, so a real `import heapq`
        # program is refused before any site table is consulted (documented,
        # unrelated limitation -- see module docstring). This exercises the
        # SAME _record_obligation/_enclosing_function plumbing directly on a
        # Call node, the way visit_call reaches it, without fighting the
        # Resolver's unrelated module-attribute gate.
        from tharkun.solve.census import _Scan
        from tharkun.solve.pyresolve import Resolver
        tree = ast.parse("def pop_one(h):\n    return heapq.heappop(h)\n")
        scan = _Scan(tree, ("IndexError",), Resolver(tree, frozenset()), "t.py")
        scan.parent = {ch: n for n in ast.walk(tree) for ch in ast.iter_child_nodes(n)}
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
        scan._record_obligation(call, "heapq.heappop")
        assert len(scan.obligations) == 1
        _line, _col, ob = scan.obligations[0]
        assert ob.invariant.value == "len(h) > 0"
        assert ob.invariant.function == "pop_one"

    def test_negative_control_a_wrong_contract_cannot_manufacture_a_false_no_site(self):
        src = "x = [1, 2, 3]\ny = x[5]\n"
        c = census.site_census(src, ("IndexError",))
        assert c.no_site is False
