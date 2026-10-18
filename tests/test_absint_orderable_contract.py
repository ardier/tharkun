"""Differential checks for the protocol template beside legacy ordering checks."""

from __future__ import annotations

import ast
import itertools
import operator
from pathlib import Path

from tharkun.solve.absint import prove
from tharkun.solve.absint.aval import ANY_STR, FloatInfo, V
from tharkun.solve.absint.interp import Analyzer
from tharkun.solve.absint.linear import IntInfo
from tharkun.solve.absint.state import Frame, State


TAGS = ("int", "bool", "float", "str", "NoneType")
OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
NATIVE = {ast.Lt: operator.lt, ast.LtE: operator.le,
          ast.Gt: operator.gt, ast.GtE: operator.ge}


def _value(tag):
    return {
        "int": lambda: V(i=IntInfo.interval(None, None)),
        "bool": lambda: V(b=frozenset({False, True})),
        "float": lambda: V(f=FloatInfo()),
        "str": lambda: V(s=ANY_STR),
        "NoneType": lambda: V(n=True),
    }[tag]()


def _union(tags):
    return V(
        i=IntInfo.interval(None, None) if "int" in tags else None,
        b=frozenset({False, True}) if "bool" in tags else None,
        f=FloatInfo() if "float" in tags else False,
        s=ANY_STR if "str" in tags else None,
        n="NoneType" in tags,
    )


def _masks():
    for size in range(1, len(TAGS) + 1):
        yield from (frozenset(items) for items in itertools.combinations(TAGS, size))


def _diagnostic(filename, node, op, left, right, legacy, candidate):
    op_name = op.__name__ if isinstance(op, type) else type(op).__name__
    return (f"{filename}:{node.lineno}: {op_name} "
            f"{sorted(left)} × {sorted(right)} legacy={legacy!r} contract={candidate!r}")


def _fixture():
    path = Path(__file__).parent / "fixtures" / "absint_ordering_matrix.py"
    source = path.read_text()
    filename = str(path)
    analyzer = Analyzer(source, filename=filename)
    nodes = {type(statement.value.ops[0]): statement.value for statement in analyzer.tree.body}
    return analyzer, source, filename, nodes


def _observe_compare(analyzer, node, op, left, right):
    """Run the real compare_split/_orderable path and capture its check result."""
    checks = []
    old_orderable = analyzer._orderable
    old_check = analyzer.check

    def watched_orderable(st, lv, rv):
        safe = old_orderable(st, lv, rv)
        checks.append((safe, analyzer._orderable_contract_candidate(op(), lv, rv)))
        return safe

    def watched_check(_node, check_op, exceptions, ok, *args, **kwargs):
        if check_op == "compare" and "TypeError" in exceptions:
            checks.append(("check", ok))

    analyzer._orderable = watched_orderable
    analyzer.check = watched_check
    try:
        state = State()
        state.frames.append(Frame("module"))
        analyzer.compare_split(state, op(), left, right,
                               node.left, node.comparators[0], node)
    finally:
        analyzer._orderable = old_orderable
        analyzer.check = old_check
    decision = next((entry[1] for entry in checks if entry[0] == "check"), None)
    observed = next((entry for entry in checks if entry[0] != "check"), None)
    if observed is None or decision is None or observed[0] != decision:
        raise AssertionError(f"{analyzer.filename}:{node.lineno}: compare_split did not expose one legacy check: {checks!r}")
    return observed[0], observed[1]


def _compare_rows(rows, filename):
    analyzer, _source, actual_filename, nodes = _fixture()
    assert filename == actual_filename
    differences = []
    for op, left_tags, right_tags in rows:
        node = nodes[op]
        legacy, candidate = _observe_compare(analyzer, node, op, _union(left_tags), _union(right_tags))
        if type(candidate) is not bool:
            differences.append(_diagnostic(actual_filename, node, op, left_tags, right_tags,
                                           legacy, "unavailable"))
        elif candidate != legacy:
            differences.append(_diagnostic(actual_filename, node, op, left_tags, right_tags,
                                           legacy, candidate))
    return differences


def test_all_31_kind_subsets_crossed_match_legacy_compare_split():
    rows = [(op, left, right)
            for op in OPS for left, right in itertools.product(_masks(), repeat=2)]
    fixture = str(Path(__file__).parent / "fixtures" / "absint_ordering_matrix.py")
    differences = _compare_rows(rows, fixture)
    assert len(rows) == 4 * 31 * 31
    assert not differences, "\n".join(differences)


def test_exact_scalar_pairs_match_legacy_and_native_typeerror_behavior():
    samples = {
        "int": (-3, 0, 8),
        "bool": (False, True),
        "float": (-1.5, 0.0, 4.25),
        "str": ("", "abc", "z"),
        "NoneType": (None,),
    }
    rows = [(op, left_tag, right_tag, left, right)
            for op in OPS
            for left_tag, left_values in samples.items()
            for right_tag, right_values in samples.items()
            for left, right in itertools.product(left_values, right_values)]
    analyzer, _source, filename, nodes = _fixture()
    differences = []
    for op, left_tag, right_tag, left, right in rows:
        node = nodes[op]
        legacy, candidate = _observe_compare(analyzer, node, op, _value(left_tag), _value(right_tag))
        try:
            NATIVE[op](left, right)
            native_safe = True
        except TypeError:
            native_safe = False
        if type(candidate) is not bool or candidate != legacy or candidate != native_safe:
            differences.append(_diagnostic(filename, node, op,
                                           {left_tag}, {right_tag}, legacy,
                                           (candidate, f"native_safe={native_safe}")))
    assert not differences, "\n".join(differences)


def test_candidate_refuses_values_outside_its_scalar_domain():
    from tharkun.solve.contracts import safety
    from tharkun.solve.terms import PY, and_, literal, typeis, var
    analyzer = Analyzer("")
    for op in (ast.Eq(), ast.NotEq(), ast.Is(), ast.In()):
        assert analyzer._orderable_contract_candidate(op, _value("int"), _value("int")) is None
    assert analyzer._orderable_contract_candidate(ast.Lt(), V(), _value("int")) is None
    assert analyzer._orderable_contract_candidate(ast.Lt(), V(top=True), _value("int")) is None
    assert analyzer._orderable_contract_candidate(
        ast.Lt(), V(i=IntInfo.interval(0, 1), refs=frozenset({("heap", 1)})), _value("int")
    ) is None
    assert analyzer._orderable_contract_candidate(
        ast.Lt(), V(i=IntInfo.interval(0, 1), st=frozenset({("static", "x")})), _value("int")
    ) is None
    old_cases = safety.cases
    try:
        safety.cases = lambda *_args: safety.Refused("test refusal")
        assert analyzer._orderable_contract_candidate(ast.Lt(), _value("int"), _value("int")) is None
        safety.cases = lambda *_args: ()
        assert analyzer._orderable_contract_candidate(ast.Lt(), _value("int"), _value("int")) is None
        safety.cases = lambda *_args: (
            safety.Case(literal(True), None),
            safety.Case(literal(True), None),
        )
        assert analyzer._orderable_contract_candidate(ast.Lt(), _value("int"), _value("int")) is None
        safety.cases = lambda *_args: (safety.Case(literal(False), None),)
        assert analyzer._orderable_contract_candidate(ast.Lt(), _value("int"), _value("int")) is None
        partial = and_(typeis(var("_left", PY), "int"),
                       typeis(var("_right", PY), "int"))
        safety.cases = lambda *_args: (safety.Case(partial, literal(True)),)
        mixed_left = V(i=IntInfo.interval(None, None), s=ANY_STR)
        assert analyzer._orderable_contract_candidate(ast.Lt(), mixed_left, _value("int")) is None
    finally:
        safety.cases = old_cases


def test_withdrawn_comparison_claim_is_unavailable(monkeypatch):
    from tharkun.solve.contracts import safety

    analyzer = Analyzer("")
    monkeypatch.setattr(safety, "get", lambda _qualname: None)
    assert analyzer._orderable_contract_candidate(ast.Lt(), _value("int"), _value("int")) is None


def test_refused_template_is_unavailable_and_differential_collects_real_sites(monkeypatch):
    from tharkun.solve.contracts import safety
    from tharkun.solve.terms import literal

    rows = [(op, left, right)
            for op in OPS for left, right in itertools.product(_masks(), repeat=2)]
    analyzer, _source, filename, nodes = _fixture()
    # All pairs are evaluated through the real compare_split/_orderable path.  The
    # deliberately wrong case claims every pair is safe, so all unsafe pairs must
    # be reported, with their actual fixture source location, before the assert.
    monkeypatch.setattr(safety, "cases", lambda *_args: (
        safety.Case(literal(True), literal(False)),
    ))
    mismatches = []
    expected = 0
    for op, left_tags, right_tags in rows:
        node = nodes[op]
        legacy, candidate = _observe_compare(analyzer, node, op, _union(left_tags), _union(right_tags))
        if not legacy:
            expected += 1
        if type(candidate) is not bool or candidate != legacy:
            mismatches.append(_diagnostic(filename, node, op,
                                          left_tags, right_tags, legacy, candidate))
    assert expected > 1
    assert len(mismatches) == expected
    assert all(line.startswith(filename + ":") for line in mismatches)
    lines = {int(line.split(":", 2)[1]) for line in mismatches}
    assert lines == {nodes[op].lineno for op in OPS}


def test_corrupt_template_differential_keeps_real_prove_result_and_legacy_records(monkeypatch):
    from tharkun.solve.contracts import safety
    from tharkun.solve.terms import literal

    source = "left = 1\nright = 2\nsmall = left < right\ntext = 'a' < 'b'\nlast = left <= right\n"
    before = prove(source, ("TypeError",), filename="comparison_public_result.py")
    assert before.verdict == "TRUE" and before.complete and before.records
    monkeypatch.setattr(safety, "cases", lambda *_args: (
        safety.Case(literal(True), literal(True)),
    ))
    real_compare_split = Analyzer.compare_split
    mismatches = []

    def instrumented(self, st, op, lv, rv, lnode, rnode, node):
        real_orderable = self._orderable

        def compared(state, left, right):
            legacy = real_orderable(state, left, right)
            candidate = self._orderable_contract_candidate(op, left, right)
            if type(candidate) is not bool or candidate != legacy:
                mismatches.append(_diagnostic(self.filename, node, op,
                                              set(left.kinds()), set(right.kinds()),
                                              legacy, candidate))
            return legacy

        self._orderable = compared
        try:
            return real_compare_split(self, st, op, lv, rv, lnode, rnode, node)
        finally:
            self._orderable = real_orderable

    monkeypatch.setattr(Analyzer, "compare_split", instrumented)
    after = prove(source, ("TypeError",), filename="comparison_public_result.py")
    assert len(mismatches) == 3, "\n".join(mismatches)
    assert all(line.startswith("comparison_public_result.py:") for line in mismatches)
    assert {int(line.split(":", 2)[1]) for line in mismatches} == {3, 4, 5}
    for field in ("verdict", "component", "sites", "reason", "lemmas", "checks_total",
                  "records", "complete", "source_sha", "invariants"):
        assert getattr(after, field) == getattr(before, field), field

