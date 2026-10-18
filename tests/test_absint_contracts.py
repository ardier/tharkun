"""Class candidates need establishment, preservation and live preconditions."""
from textwrap import dedent

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.absint.interp import Analyzer

PROPS = ("TypeError", "AttributeError", "ZeroDivisionError", "AssertionError")


def test_hidden_method_state_preserves_the_existing_arithmetic_proof():
    source = dedent("""
        import _sv_verifier
        class Ring:
            def __init__(self):
                self.k = 1
            def step(self):
                self.k = self.k % 4 + 1
                return 100 // self.k
        def main():
            r = Ring()
            for _ in range(abs(_sv_verifier.nondet_int())):
                r.step()
        main()
    """)
    for draw in (0, 1, 7, 32):
        exec(source.replace("import _sv_verifier", "")
             .replace("_sv_verifier.nondet_int()", str(draw)), {})
    result = prove(source, ("ZeroDivisionError", "FloatingPointError"), time_budget=10)
    assert result.verdict == "TRUE", result.explain()
    assert result.component == "proof:interval"
    assert result.complete and result.sites


@pytest.mark.parametrize("body,exception", [("return 7", None), ("raise KeyError()", KeyError)])
def test_context_free_function_analysis_never_drops_an_escaping_exception(body, exception):
    from tharkun.solve.absint.interp import FuncCtx, Unsupported
    from tharkun.solve.absint.state import Frame, State

    source = "def f():\n    " + body + "\n"
    namespace = {}
    exec(source, namespace)
    if exception is None:
        assert namespace["f"]() == 7
    else:
        with pytest.raises(exception):
            namespace["f"]()
    analysis = Analyzer(source, time_budget=10)
    state = State()
    state.frames.append(Frame("g"))
    context = FuncCtx(analysis.tree, {"f"}, set(), is_module=True)
    analysis.exec_block(analysis.tree.body, state, context, None)
    address, = state.frame.env["f"].refs
    assert not analysis._fctx_stack
    if exception is None:
        result = analysis.call_func(state, address, [], {}, analysis.tree.body[0])
        assert not state.bottom and analysis.ibounds(state, result.i) == (7, 7)
    else:
        with pytest.raises(Unsupported, match="context-free function analysis"):
            analysis.call_func(state, address, [], {}, analysis.tree.body[0])


@pytest.mark.parametrize("initial,update,relation", [
    ("1", "+= 1", ">= 1"),
    ("-2", "-= 1", "<= -1"),
])
def test_inductive_class_query_contract(initial, update, relation):
    source = dedent(f"""
        from _sv_verifier import nondet_bool
        class Meter:
            def __init__(self):
                self.count = {initial}
            def advance(self):
                self.count {update}
            def reciprocal(self):
                return 1 / self.count
        meter = Meter()
        first = meter.reciprocal()
        while nondet_bool():
            meter.advance()
            value = meter.reciprocal()
        assert meter.count {relation}
    """)
    for program in (source, source.replace("Meter", "Gauge").replace("count", "level")
                    .replace("meter", "gauge").replace("advance", "step").replace("reciprocal", "inverse")):
        analysis = Analyzer(program, time_budget=10)
        analysis.run()
        assert "class-invariant" in analysis.features
        result = prove(program, PROPS, time_budget=10)
        assert result.verdict == "TRUE", result.explain()


@pytest.mark.parametrize("change", [
    "meter.clear()",
    "alias = meter\nalias.count = 0",
    "Meter.reciprocal = broken",
])
def test_broken_method_alias_or_class_change_never_proves(change):
    source = dedent("""
        class Meter:
            def __init__(self):
                self.count = 1
            def clear(self):
                self.count = 0
            def reciprocal(self):
                return 1 / self.count
        def broken(self):
            return 1 / 0
        meter = Meter()
        first = meter.reciprocal()
    """) + change + "\nvalue = meter.reciprocal()"
    for program in (source, source.replace("Meter", "Gauge").replace("count", "level")
                    .replace("meter", "gauge").replace("reciprocal", "inverse")):
        with pytest.raises(ZeroDivisionError):
            exec(program, {})
        result = prove(program, PROPS, time_budget=10)
        assert result.verdict != "TRUE", result.explain()


def test_external_alias_invalidates_an_established_contract():
    source = dedent("""
        class Meter:
            def __init__(self):
                self.count = 1
            def advance(self):
                self.count += 1
            def reciprocal(self):
                return 1 / self.count
        meter = Meter()
        first = meter.reciprocal()
        alias = meter
        alias.count = 0
        value = meter.reciprocal()
    """)
    analysis = Analyzer(source, time_budget=10)
    analysis.run()
    assert "class-invariant" in analysis.features
    with pytest.raises(ZeroDivisionError):
        exec(source, {})
    assert prove(source, PROPS, time_budget=10).verdict != "TRUE"


@pytest.mark.parametrize("initial,update,offset", [("5", "+= 1", "- 4"), ("-3", "-= 1", "+ 2")])
def test_observed_candidates_are_proved_statically(tmp_path, initial, update, offset):
    import sys

    from tharkun.solve.absint.contracts import demands
    from tharkun.sv.engine import run_once

    source = dedent(f"""
        from _sv_verifier import nondet_bool
        class Gauge:
            def __init__(self):
                self.level = {initial}
            def step(self):
                self.level {update}
            def inverse(self):
                return 1 / (self.level {offset})
        gauge = Gauge()
        while nondet_bool():
            gauge.step()
        value = gauge.inverse()
    """)
    for program in (source, source.replace("Gauge", "Meter").replace("level", "amount")
                    .replace("gauge", "meter").replace("inverse", "reciprocal")):
        path = tmp_path / "candidate.py"
        path.write_text(program)
        out = run_once(sys.executable, str(path), [], timeout=10, observe_fields=demands(program))
        assert out.status == "clean" and out.class_hints
        analysis = Analyzer(program, time_budget=10, class_hints=out.class_hints)
        analysis.run()
        assert "class-invariant" in analysis.features
        assert prove(program, PROPS, time_budget=10, class_hints=out.class_hints).verdict == "TRUE"


@pytest.mark.parametrize("hints", [
    {"Gauge": {"level": [1, 1]}},
    {"Gauge": {"level": [20, 40]}},
    {"Gauge": "invalid"},
])
def test_false_samples_never_establish_a_constructor_invariant(hints):
    source = dedent("""
        class Gauge:
            def __init__(self):
                self.level = 0
            def inverse(self):
                return 1 / self.level
        gauge = Gauge()
        answer = gauge.inverse()
    """)
    with pytest.raises(ZeroDivisionError):
        exec(source, {})
    assert prove(source, PROPS, time_budget=10, class_hints=hints).verdict != "TRUE"


def test_observation_does_not_execute_attribute_hooks(tmp_path):
    import sys

    from tharkun.sv.engine import run_once

    source = dedent("""
        class Gauge:
            def __init__(self):
                self.level = 5
            @property
            def __dict__(self):
                return 1 / 0
        gauge = Gauge()
    """)
    path = tmp_path / "hooks.py"
    path.write_text(source)
    out = run_once(sys.executable, str(path), [], timeout=10, observe_fields={"Gauge": ["level"]})
    assert out.status == "clean"
    assert not out.class_hints


@pytest.mark.parametrize('initial,update,index', [
    ('[1]', 'self.items[0] += 1', '0'),
    ('[2, 3]', 'self.items = [4]', '-1'),
])
def test_nonempty_integer_list_contract(initial, update, index):
    source = dedent(f'''
        from _sv_verifier import nondet_bool
        class Sample:
            def __init__(self):
                self.items = {initial}
            def advance(self):
                {update}
            def query(self):
                return self.items[{index}] + 1
        sample = Sample()
        first = sample.query()
        while nondet_bool():
            sample.advance()
            result = sample.query()
    ''')
    for program in (source, source.replace('Sample', 'Buffer').replace('sample', 'buffer')
                    .replace('items', 'values').replace('query', 'read')):
        analysis = Analyzer(program, time_budget=10)
        analysis.run()
        assert 'class-invariant' in analysis.features
        result = prove(program, PROPS + ('IndexError',), time_budget=10)
        assert result.verdict == 'TRUE', result.explain()


@pytest.mark.parametrize('change', [
    'alias = sample.items\nalias.clear()',
    "alias = sample.items\nalias[0] = 'text'",
    'sample.items = []',
    'sample.break_shape()',
])
def test_list_shape_contract_invalidated_by_mutation(change):
    source = dedent('''
        class Sample:
            def __init__(self):
                self.items = [1]
            def query(self):
                return self.items[0] + 1
    ''')
    if 'break_shape' in change:
        source += '    def break_shape(self):\n        self.items = []\n'
    source += 'sample = Sample()\nfirst = sample.query()\n' + change + '\nresult = sample.query()'
    for program in (source, source.replace('Sample', 'Buffer').replace('sample', 'buffer')
                    .replace('items', 'values').replace('query', 'read')):
        with pytest.raises((IndexError, TypeError)):
            exec(program, {})
        result = prove(program, PROPS + ('IndexError',), time_budget=10)
        assert result.verdict != 'TRUE', result.explain()


def test_list_identity_is_not_inferred_from_separate_contract_objects():
    source = dedent('''
        class Pair:
            def __init__(self):
                self.left = [1]
                self.right = [1]
            def same(self):
                return self.left is self.right
        pair = Pair()
        before = pair.same()
        pair.right = pair.left
        assert not pair.same()
    ''')
    with pytest.raises(AssertionError):
        exec(source, {})
    for program in (source, source.replace('Sample', 'Buffer').replace('sample', 'buffer')
                    .replace('items', 'values').replace('grow', 'expand')):
        assert prove(program, PROPS, time_budget=10).verdict != 'TRUE'


def test_inplace_local_alias_is_not_a_pure_query():
    source = dedent('''
        class Sample:
            def __init__(self):
                self.items = [1]
            def grow(self):
                alias = self.items
                alias += [2]
                return 0
        sample = Sample()
        result = sample.grow()
        assert len(sample.items) == 1
    ''')
    with pytest.raises(AssertionError):
        exec(source, {})
    for program in (source, source.replace('Sample', 'Buffer').replace('sample', 'buffer')
                    .replace('items', 'values').replace('grow', 'expand')):
        assert prove(program, PROPS, time_budget=10).verdict != 'TRUE'
