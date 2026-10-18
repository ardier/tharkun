"""Visitors for the smaller contract consumers ([internal ref]).

Public adapters retain their import paths. Their existing facts and tested gates
are the visit bodies; no new semantics are introduced by this dispatch.
"""
from __future__ import annotations

from tharkun.solve.contracts.registry import get
from tharkun.solve.contracts.rules import Refused, Rule, Visitor, all_rules


class CensusVisitor(Visitor):
    engine = "census"

    def _closed(self, rule: Rule, ctx):
        if ctx is not None:
            from tharkun.solve.contracts.adapters.obligations import _obligation_for_call_model
            node, kwargs = ctx
            return _obligation_for_call_model(rule.qualname, node, **kwargs)
        from tharkun.solve.contracts.reachability import _closed_call_hooks_model
        return _closed_call_hooks_model(rule.qualname)

    visit_heapq_heappop = _closed


class StationaryVisitor(Visitor):
    engine = "stationary"

    def _class_only(self, rule: Rule, ctx):
        from tharkun.solve.contracts.adapters.common import _result_depends_only_on_class_model
        count, keywords = ctx
        return _result_depends_only_on_class_model(rule.qualname, arg_count=count, has_keywords=keywords)

    visit_type = visit_isinstance = _class_only


def _receiver_evidence(visitor, rule: Rule, exc):
    from tharkun.solve.contracts.adapters.common import native_raise_condition
    from tharkun.solve.contracts.registry import get
    contract = get(rule.qualname)
    return contract is not None and native_raise_condition(contract, exc) is not None


def _constructor(visitor, rule: Rule, ctx):
    from tharkun.solve.contracts.adapters.common import _constructor_contract_model
    return _constructor_contract_model(rule.qualname)


class AvengersVisitor(Visitor):
    engine = "avengers"
    visit_list = visit_tuple = visit_set = _constructor


class DrawsVisitor(Visitor):
    engine = "draws"
    visit_list = visit_tuple = visit_set = _constructor

    def visit_random_random(self, rule: Rule, ctx):
        from tharkun.solve.contracts.adapters.common import closed_range_return
        return closed_range_return(rule.qualname)

    def visit_heapq_heapify(self, rule: Rule, ctx):
        from tharkun.solve.contracts.registry import get
        contract = get(rule.qualname)
        return contract is not None and contract.tested


class ProductVisitor(Visitor):
    """The exact product proof (``sv/product_proof.py``, allowlist ``sv/exact_subset.py``) admits no call into a library, by design; it
    admits ``abs(x)``, ``int(x)``, ``min(a, b)`` and ``max(a, b)`` on int/bool operands, whose ``_symexec`` semantics were audited exact.  The
    visit answers the audited ARITY of the call, while the contract of the builtin is TESTED ([internal ref] step 2): the allowlist asks the
    rule instead of keeping a table of names, so an untested or withdrawn contract takes the builtin out of the subset."""

    engine = "product"
    #: the audited call shapes (``exact_subset.py`` docstring): builtin -> its only accepted positional arity
    EXACT_ARITY = {"abs": 1, "int": 1, "min": 2, "max": 2}

    def _exact_arity(self, rule: Rule, ctx):
        contract = get(rule.qualname)
        if contract is None or not contract.tested:
            return Refused(self.engine, rule.qualname, "no tested contract")
        return self.EXACT_ARITY[rule.qualname]

    visit_abs = visit_int = visit_min = visit_max = _exact_arity


class EregionVisitor(Visitor):
    engine = "eregion"


# Derive the supported kinds from the existing audited rows, never a second
# per-qualname visitor registry.
def _install():
    from tharkun.solve.contracts.reachability import _CLOSED
    from tharkun.solve.contracts.adapters.common import native_raise_condition
    for rule in all_rules():
        if rule.qualname in _CLOSED or rule.qualname in {"dict", "dict.update", "random.seed"}:
            setattr(CensusVisitor, "visit_" + rule.kind, CensusVisitor._closed)
        if "." in rule.qualname and rule.qualname.split(".", 1)[0] in {"list", "dict", "set", "tuple", "str", "bytes"}:
            if any(native_raise_condition(rule.contract, exc) is not None for exc in ("IndexError", "KeyError")):
                for visitor in (AvengersVisitor, DrawsVisitor):
                    setattr(visitor, "visit_" + rule.kind, _receiver_evidence)


_install()
