"""Ordered Boolean projection, guard-refusal parity and public case safety."""
import math

import pytest
from hypothesis import given, settings, strategies as st

from tharkun.solve import terms as t
from tharkun.solve import terms_z3 as tz
from tharkun.solve.contracts.safety import Case, case_safety
from tharkun.solve.predicates import check_predicate


BUILTINS = {"type", "int", "float", "bool"}
X, Y = t.var("x"), t.var("y")
GX, GY = t.typeis(X, "int"), t.typeis(Y, "int")
P = t.eq(t.as_int(Y), t.literal(0))


def ground(env):
    facts = []
    for name, value in env.items():
        source = t.var(name)
        facts.append(t.typeis(source, type(value).__name__))
        cast = {int: t.as_int, float: t.as_f64, bool: t.as_bool}.get(type(value))
        if cast is not None:
            if type(value) is float and math.isnan(value):
                facts.append(t.ne(cast(source), cast(source)))
            else:
                facts.append(t.eq(cast(source), t.literal(value)))
    return facts


def assert_three_way(term, env):
    pytest.importorskip("z3")
    text = t.to_python(term, builtin_names=BUILTINS)
    assert isinstance(text, str)
    check_predicate(text, {name: "object" for name in t._variables(term)}, builtin_names=BUILTINS)
    expected = t.eval_term(term, env)
    assert type(expected) is bool
    assert eval(text, {}, env) is expected
    facts = ground(env)
    assert tz.implies(facts, term) == ("yes" if expected else "no")
    assert tz.implies(facts, t.not_(term)) == ("no" if expected else "yes")


@pytest.mark.parametrize("kind,cast,zero,values", [
    ("int", t.as_int, 0, [0, 2, -3, True, 2.5, None]),
    ("float", t.as_f64, 0.0, [0.0, -0.0, 2.5, float("nan"), float("inf"), 2, True, None]),
    ("bool", t.as_bool, False, [False, True, 0, 1, 0.0, None]),
])
def test_public_grouped_case_matches_flat_projection_and_three_way(kind, cast, zero, values):
    guard = t.typeis(Y, kind)
    raises = t.eq(cast(Y), t.literal(zero))
    nested = case_safety(Case(t.and_(GX, guard), raises))
    flat = t.and_(GX, guard, t.not_(raises))
    before = t.canon(nested)
    assert before != t.canon(flat)
    assert t.to_python(nested, builtin_names=BUILTINS) == t.to_python(flat, builtin_names=BUILTINS)
    assert t.canon(nested) == before  # projection does not normalize the stored term
    assert t.from_canon(before) == nested
    for value in values:
        assert_three_way(nested, {"x": 1, "y": value})


def test_same_or_nesting_splices_only_or_and_keeps_and_branches_independent():
    a = t.and_(GX, t.gt(t.as_int(X), t.literal(0)))
    b = t.and_(GY, P)
    nested = t.or_(t.or_(a, t.literal(False)), t.or_(b, t.literal(True)))
    flat = t.or_(a, t.literal(False), b, t.literal(True))
    assert t.canon(nested) != t.canon(flat)
    assert t.to_python(nested, builtin_names=BUILTINS) == t.to_python(flat, builtin_names=BUILTINS)
    for env in ({"x": 1, "y": None}, {"x": 0, "y": 0}, {"x": None, "y": 2}):
        assert_three_way(nested, env)


@pytest.mark.parametrize("term", [
    t.and_(P, t.and_(GX, GY)),
    t.and_(t.and_(GX, t.not_(GY)), P),
    t.and_(t.or_(GY, GX), P),
    t.and_(t.and_(GX, t.literal(True)), P),
    t.and_(t.and_(GX, t.typeis(Y, "float")), P),
    t.and_(t.or_(t.and_(GY, t.literal(True)), t.literal(False)), P),
    t.or_(t.and_(GY, t.literal(False)), P),
])
def test_late_negated_or_other_name_and_wrong_tag_guards_still_refuse(term):
    pytest.importorskip("z3")
    rendered, lowered = t.to_python(term, builtin_names=BUILTINS), tz.lower(term)
    assert isinstance(rendered, t.Refused) and "guard" in rendered.reason
    assert isinstance(lowered, t.Refused) and "guard" in lowered.reason


def test_splicing_preserves_builtin_identity_and_unsupported_value_refusals():
    nested = case_safety(Case(t.and_(GX, GY), P))
    for names in (set(), {"type"}, {"int"}):
        assert isinstance(t.to_python(nested, builtin_names=names), t.Refused)
    for name in ("type", "int"):
        assert isinstance(t.to_python(nested, types={name: "object"}, builtin_names=BUILTINS), t.Refused)
    for value in (t.floordiv(t.as_int(Y), t.literal(1)), t.i2f(t.as_int(Y))):
        zero = t.literal(0.0) if value.sort == t.F64 else t.literal(0)
        term = case_safety(Case(t.and_(GX, GY), t.eq(value, zero)))
        assert isinstance(t.to_python(term, builtin_names=BUILTINS), t.Refused)
    assert isinstance(t.to_python(t.and_(t.and_(GY, t.literal(True)), t.typeis(Y, "NoneType")),
                                  builtin_names=BUILTINS), t.Refused)


def test_wrong_type_short_circuit_does_not_invoke_callbacks_after_splicing():
    class Poison:
        def __eq__(self, other): raise AssertionError("user equality")
        def __bool__(self): raise AssertionError("user truth")

    term = case_safety(Case(t.and_(GX, GY), P))
    text = t.to_python(term, builtin_names=BUILTINS)
    assert isinstance(text, str)
    for env in ({"x": Poison(), "y": 1}, {"x": 1, "y": Poison()}):
        assert t.eval_term(term, env) is eval(text, {}, env) is False


SHAPES = st.recursive(st.sampled_from(["gx", "gy", "p", "true", "false"]),
    lambda child: st.one_of(
        st.tuples(st.sampled_from(["and", "or"]), st.lists(child, min_size=1, max_size=3)),
        st.tuples(st.just("not"), child)), max_leaves=10)


def build(shape):
    if isinstance(shape, str):
        return {"gx": GX, "gy": GY, "p": P, "true": t.literal(True), "false": t.literal(False)}[shape]
    op, children = shape
    if op == "not":
        return t.not_(build(children))
    return (t.and_ if op == "and" else t.or_)(*(build(child) for child in children))


def hand_flat(term):
    if term.op not in {"and", "or", "not"}:
        return term
    if term.op == "not":
        return t.not_(hand_flat(term.args[0]))
    args = []
    for child in term.args:
        flat = hand_flat(child)
        args.extend(flat.args if flat.op == term.op else (flat,))
    return (t.and_ if term.op == "and" else t.or_)(*args)


@settings(max_examples=35, deadline=None)
@given(SHAPES)
def test_generated_nesting_guard_refusal_parity_and_native_python_z3_agreement(shape):
    pytest.importorskip("z3")
    term = build(shape)
    before = t.canon(term)
    rendered, lowered = t.to_python(term, builtin_names=BUILTINS), tz.lower(term)
    assert isinstance(rendered, t.Refused) == isinstance(lowered, t.Refused)
    flat_text = t.to_python(hand_flat(term), builtin_names=BUILTINS)
    if isinstance(rendered, t.Refused):
        assert "guard" in rendered.reason and "guard" in lowered.reason
        assert isinstance(flat_text, t.Refused)
    else:
        assert rendered == flat_text
        for env in ({"x": 1, "y": 0}, {"x": 1, "y": 2}, {"x": 0, "y": -3},
                    {"x": 1.5, "y": 2}, {"x": 1, "y": True}, {"x": 1, "y": None}):
            assert_three_way(term, env)
    assert t.canon(term) == before
    assert t.from_canon(before) == term


def test_flat_projection_bytes_are_unchanged():
    term = t.and_(GX, GY, t.not_(P))
    assert t.to_python(term, builtin_names=BUILTINS) == (
        "((type(x) is int) and (type(y) is int) and (not (y == 0)))")
    assert t.canon(term) == (
        '(term 1 (and Bool (typeis Bool (var Py "x") (tag Tag "int")) '
        '(typeis Bool (var Py "y") (tag Tag "int")) '
        '(not Bool (eq Bool (as_int Int (var Py "y")) (lit Int 0)))))')


def test_frozen_checker_accepts_grouped_case_projection_at_real_statement():
    from tharkun.solve.obligations import Invariant, witness
    from tharkun.sv.validate import validate
    from tharkun.sv.witness import dumps

    source = "x = 1\ny = 2\nz = x // y\n"
    specification = "CHECK( init(main()), ! uncaught(ZeroDivisionError))\n"
    term = case_safety(Case(t.and_(GX, GY), P))
    text = t.to_python(term, builtin_names=BUILTINS)
    assert isinstance(text, str)
    invariant = Invariant("nested.py", 3, 1, "<module>", text)
    document = dumps(witness(source, "nested.py", specification, [invariant]))
    assert validate(source, "nested.py", specification, document).confirmed
    unsafe = source.replace("y = 2", "y = 0")
    mutated = dumps(witness(unsafe, "nested.py", specification, [invariant]))
    assert not validate(unsafe, "nested.py", specification, mutated).confirmed


@pytest.mark.parametrize("value", [0, 1, -1, True, False, 0.0, -0.0,
                                  float("inf"), -float("inf"), float("nan")])
def test_literal_type_tests_fold_exact_kind_without_value_spelling_or_builtin_lookup(value):
    for tag in ("int", "bool", "float", "NoneType", "str"):
        term = t.typeis(t.literal(value), tag)
        before = t.canon(term)
        expected = type(value).__name__ == tag
        assert t.to_python(term) == repr(expected)
        assert t.to_python(term, types={"type": "object", "int": "object"}) == repr(expected)
        assert_three_way(term, {})
        assert t.canon(term) == before and t.from_canon(before) == term


def test_owner_literal_guard_examples_remove_only_the_literal_guard():
    sent = t.var("sent")
    mixed = t.and_(t.typeis(sent, "int"), t.typeis(t.literal(0), "int"))
    literal_only = t.and_(t.typeis(t.literal(1), "int"), t.typeis(t.literal(30), "int"))
    assert t.to_python(mixed, builtin_names=BUILTINS) == "(type(sent) is int)"
    assert t.to_python(literal_only) == "True"
    for value in (1, 0, True, 0.0, float("nan"), None, object()):
        assert_three_way(mixed, {"sent": value})
    class Poison:
        def __bool__(self): raise AssertionError("user truth")
        def __eq__(self, other): raise AssertionError("user equality")
    env = {"sent": Poison()}
    assert t.eval_term(mixed, env) is eval(t.to_python(mixed, builtin_names=BUILTINS), {}, env) is False
    assert_three_way(literal_only, {})


@pytest.mark.parametrize("op", [t.and_, t.or_])
@pytest.mark.parametrize("matches", [(True, True), (True, False), (False, True), (False, False)])
def test_all_literal_guard_groups_reduce_consistently_including_cross_operator_groups(op, matches):
    guards = [t.typeis(t.literal(1), "int" if match else "bool") for match in matches]
    nested = op(op(guards[0]), op(guards[1]))
    expected = all(matches) if op is t.and_ else any(matches)
    assert t.to_python(nested) == repr(expected)
    assert_three_way(nested, {})
    # The opposite Boolean group can reduce to a neutral identity in its parent.
    neutral = t.or_(*guards) if op is t.and_ else t.and_(*guards)
    parent = op(GX, neutral)
    text = t.to_python(parent, builtin_names=BUILTINS)
    is_neutral = any(matches) if op is t.and_ else not all(matches)
    assert text == ("(type(x) is int)" if is_neutral else (
        "((type(x) is int) and False)" if op is t.and_ else "((type(x) is int) or True)"))
    for value in (1, True, None):
        assert_three_way(parent, {"x": value})


def test_plain_boolean_literal_bytes_and_singleton_groups_remain_unchanged():
    assert t.to_python(t.and_(t.literal(True))) == "(True)"
    assert t.to_python(t.or_(t.literal(False))) == "(False)"
    assert t.to_python(t.and_(GX), builtin_names=BUILTINS) == "((type(x) is int))"
    assert t.to_python(t.or_(GX), builtin_names=BUILTINS) == "((type(x) is int))"
    assert t.to_python(t.and_(GX, t.literal(True)), builtin_names=BUILTINS) == (
        "((type(x) is int) and True)")
    assert t.to_python(t.or_(GX, t.literal(False)), builtin_names=BUILTINS) == (
        "((type(x) is int) or False)")


def test_literal_folding_does_not_authenticate_variable_or_computed_type_tests():
    truth = t.typeis(t.literal(1), "int")
    for names in (set(), {"type"}, {"int"}):
        assert isinstance(t.to_python(t.and_(truth, GX), builtin_names=names), t.Refused)
    for name in ("type", "int"):
        assert isinstance(t.to_python(t.and_(truth, GX), types={name: "object"},
                                      builtin_names=BUILTINS), t.Refused)
    for operand in (X, t.add(t.literal(1), t.literal(2))):
        refused = t.to_python(t.typeis(operand, "NoneType"), builtin_names=BUILTINS)
        assert isinstance(refused, t.Refused) and refused.reason == "none_test_outside_witness_grammar"
    for operand in (t.b2i(t.literal(True)), t.i2f(t.literal(1))):
        assert isinstance(t.to_python(t.typeis(operand, "int"), builtin_names=BUILTINS), t.Refused)
    # Computed exact kinds still evaluate their operands; this is not general type folding.
    assert isinstance(t.to_python(t.typeis(t.floordiv(t.literal(1), t.literal(0)), "int"),
                                  builtin_names=BUILTINS), t.Refused)


@pytest.mark.parametrize("form,tag", [(t.and_, "bool"), (t.or_, "int")])
@pytest.mark.parametrize("unsupported", [
    t.eq(t.floordiv(t.literal(1), t.literal(0)), t.literal(0)),
    t.typeis(X, "NoneType"),
    P,
    t.eq(t.literal(float("nan")), t.literal(0.0)),
])
def test_absorbing_literal_guard_never_hides_unsupported_or_undefined_syntax(form, tag, unsupported):
    first = t.typeis(t.literal(1), tag)
    term = form(first, unsupported)
    assert isinstance(t.to_python(term, builtin_names=BUILTINS), t.Refused)
    # Native short-circuiting is retained, but projection screens the entire term.
    assert t.eval_term(term, {}) is (form is t.or_)


@settings(max_examples=30, deadline=None)
@given(st.lists(st.tuples(st.booleans(), st.booleans()), min_size=1, max_size=6))
def test_generated_literal_guard_nesting_preserves_native_render_and_smt_semantics(pairs):
    groups = [t.and_(t.typeis(t.literal(value), "bool" if match else "int"),
                     t.typeis(t.literal(0.0), "float")) for value, match in pairs]
    term = t.or_(t.or_(*groups), t.and_(GX, t.gt(t.as_int(X), t.literal(0))))
    before = t.canon(term)
    assert t.to_python(term, builtin_names=BUILTINS) == t.to_python(hand_flat(term), builtin_names=BUILTINS)
    for value in (-1, 1, True, None):
        assert_three_way(term, {"x": value})
    assert t.canon(term) == before and t.from_canon(before) == term
