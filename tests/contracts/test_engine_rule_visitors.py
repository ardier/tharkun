"""The visitor declaration must agree with every measured consumed cell."""
import pytest
from tests.contracts import _engine_audit as A
from tests.contracts._engine_audit_expected import EXPECTED
from tharkun.solve.contracts.rules import Refused, all_rules


def test_every_consumed_cell_has_a_visit_and_every_missing_visit_names_its_rule():
    for rule in all_rules():
        for engine, visitor in A.VISITORS.items():
            if EXPECTED[rule.qualname][engine] == "C":
                assert visitor.cell(rule) == "C", (rule.qualname, engine)
            if visitor.cell(rule) == "R":
                out = rule.accept(visitor.__new__(visitor))
                assert isinstance(out, Refused)
                assert out.engine == engine and out.qualname == rule.qualname
                assert rule.kind in out.reason


def test_every_rule_carries_its_valid_sample_and_fixed_twin():
    for rule in all_rules():
        assert rule.sample(rule.qualname) == A.SAMPLES[rule.qualname]
        twin = rule.twin(rule.qualname)
        assert twin is None or isinstance(twin, str)


@pytest.mark.parametrize("engine,qualname", [
    (engine, q) for q, row in EXPECTED.items()
    for engine in ("absint", "symexec") if row[engine] == "C"
])
def test_consumption_really_passes_through_accept(engine, qualname, monkeypatch):
    from tharkun.solve.contracts.rules import Rule
    seen = []
    original = Rule.accept
    def accept(rule, visitor, ctx=None):
        seen.append((visitor.engine, rule.qualname))
        return original(rule, visitor, ctx)
    monkeypatch.setattr(Rule, "accept", accept)
    probe = A.absint_cell if engine == "absint" else A.symexec_cell
    probe(qualname, A.SAMPLES[qualname])
    assert (engine, qualname) in seen
