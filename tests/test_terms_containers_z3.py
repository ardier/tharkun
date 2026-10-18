"""Conservative container models, primitive key identity and proof-context fences."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from hypothesis import given, settings, strategies as st

from tharkun.solve import terms as t
from tharkun.solve import terms_z3 as tz


z3 = pytest.importorskip("z3")
BUILTINS = {"type", "len", "int", "bool", "str", "bytes", "list", "tuple", "dict"}


def view(kind, domain=None, name="xs"):
    return (t.as_map(t.var(name), primitive_domain=domain) if kind == "dict"
            else t.as_seq(t.var(name), kind, primitive_domain=domain))


def checked(kind, condition):
    return t.and_(t.typeis(t.var("xs"), kind), condition)


@pytest.mark.parametrize("kind", ["list", "tuple", "dict"])
def test_container_lengths_are_nonnegative_without_content_assumptions(kind):
    value = view(kind)
    guard = t.typeis(t.var("xs"), kind)
    goal = t.ge(t.length(value), t.literal(0))
    assert tz.implies([guard], goal) == "yes"
    assert tz.implies([guard], t.gt(t.length(value), t.literal(0))) == "no"
    assert isinstance(tz.lower(t.length(value)), t.Refused)
    assert not isinstance(tz.lower(checked(kind, goal)), t.Refused)
    for actual in ([], [object()]) if kind == "list" else (((), (object(),)) if kind == "tuple" else ({}, {object(): object()})):
        native = checked(kind, goal)
        text = t.to_python(native, builtin_names=BUILTINS)
        assert isinstance(text, str)
        assert t.eval_term(native, {"xs": actual}) is eval(text, {}, {"xs": actual}) is True


@pytest.mark.parametrize("kind", ["list", "tuple", "dict"])
@pytest.mark.parametrize("domain", ["int", "bool", "str", "bytes"])
def test_empty_length_and_membership_imply_only_safe_length_relations(kind, domain):
    value = view(kind, domain)
    types = {"xs": f"{kind}[{domain}]"}
    key = t.key_literal("x") if domain == "str" else (t.key_literal(b"x") if domain == "bytes" else t.literal(1))
    member = t.contains(value, key)
    guard = t.typeis(t.var("xs"), kind)
    assert tz.implies([guard, t.eq(t.length(value), t.literal(0))], t.not_(member), types=types) == "yes"
    assert tz.implies([guard, member], t.gt(t.length(value), t.literal(0)), types=types) == "yes"
    # A positive size does not assert any particular value is present.
    assert tz.implies([guard, t.gt(t.length(value), t.literal(0))], member, types=types) == "no"


@pytest.mark.parametrize("kind", ["list", "tuple", "dict"])
@pytest.mark.parametrize("domain", ["int", "bool"])
def test_bool_int_membership_keys_share_python_equality_identity(kind, domain):
    value = view(kind, domain)
    types = {"xs": f"{kind}[{domain}]"}
    guard = t.typeis(t.var("xs"), kind)
    for flag in (False, True):
        integer = t.contains(value, t.literal(int(flag)))
        boolean = t.contains(value, t.literal(flag))
        assert tz.implies([guard], t.eq(integer, boolean), types=types) == "yes"
        assert tz.implies([guard, integer], boolean, types=types) == "yes"
        assert tz.implies([guard, t.not_(integer)], t.not_(boolean), types=types) == "yes"
        items = [int(flag)] if domain == "int" else [flag]
        actual = dict.fromkeys(items) if kind == "dict" else (tuple(items) if kind == "tuple" else items)
        native = checked(kind, t.eq(integer, boolean))
        text = t.to_python(native, types=types, builtin_names=BUILTINS)
        assert isinstance(text, str)
        assert t.eval_term(native, {"xs": actual}) is eval(text, {}, {"xs": actual}) is True


@pytest.mark.parametrize("domain,key", [("int", t.key_literal("1")),
    ("bool", t.key_literal(b"1")), ("str", t.literal(1)),
    ("bytes", t.key_literal("x"))])
def test_distinct_primitive_key_families_are_reflexively_safe_but_not_equal(domain, key):
    value = view("dict", domain)
    guard = t.typeis(t.var("xs"), "dict")
    assert tz.implies([guard], t.not_(t.contains(value, key)), types={"xs": f"dict[{domain}]"}) == "yes"


@pytest.mark.parametrize("literal", ["", "😀", "e\u0301", "\ud800K\x00", b"", b"\x00\xff"])
def test_literal_string_and_bytes_length_three_way(literal):
    value = t.length(t.key_literal(literal))
    expected = len(literal)
    assert t.eval_term(value, {}) == expected
    assert z3.simplify(tz.lower(value)).as_long() == expected
    text = t.to_python(t.eq(value, t.literal(expected)), builtin_names={"len"})
    assert isinstance(text, str)
    assert eval(text) is True


@pytest.mark.parametrize("kind", ["str", "bytes"])
def test_string_lengths_require_exact_guard_and_never_invent_numeric_content(kind):
    name, value = t.var("s"), t.length(t.var("s"))
    assert tz.implies([t.typeis(name, kind)], t.ge(value, t.literal(0))) == "yes"
    assert tz.implies([t.typeis(name, kind)], t.eq(value, t.literal(0))) == "no"
    assert isinstance(tz.implies([], t.ge(value, t.literal(0)), types={"s": kind}), t.Refused)


def test_membership_does_not_trust_constructor_labels_or_annotation_like_context():
    value = view("dict", "int")
    guard, goal = t.typeis(t.var("xs"), "dict"), t.contains(value, t.literal(1))
    for types in (None, {}, {"xs": "dict"}, {"xs": "dict[bool]"}, {"xs": "list[int]"},
                  {"xs": "dict[float]"}, {"xs": "typing.Dict[int,int]"}):
        assert isinstance(tz.implies([guard], goal, types=types), t.Refused)
        assert isinstance(tz.lower(checked("dict", goal), types=types), t.Refused)
    assert isinstance(tz.implies([], goal, types={"xs": "dict[int]"}), t.Refused)
    assert isinstance(tz.implies([t.typeis(t.var("other"), "dict")], goal, types={"xs": "dict[int]"}), t.Refused)


@pytest.mark.parametrize("make", [lambda g, p: p, lambda g, p: t.and_(p, g),
    lambda g, p: t.or_(g, p), lambda g, p: t.and_(t.not_(g), p),
    lambda g, p: t.or_(t.and_(g, t.literal(False)), p)])
def test_guard_order_and_disjunction_do_not_certify_container_views(make):
    g = t.typeis(t.var("xs"), "dict")
    p = t.contains(view("dict", "int"), t.literal(1))
    assert isinstance(tz.lower(make(g, p), types={"xs": "dict[int]"}), t.Refused)
    assert isinstance(tz.implies([], make(g, p), types={"xs": "dict[int]"}), t.Refused)


def test_reachable_partial_keys_remain_refused_even_for_wrong_key_family():
    zero = t.floordiv(t.literal(1), t.literal(0))
    member = t.contains(view("dict", "str"), zero)
    guard = t.typeis(t.var("xs"), "dict")
    assert isinstance(tz.implies([guard], t.not_(member), types={"xs": "dict[str]"}), t.Refused)
    assert isinstance(tz.implies([t.literal(False), guard], t.not_(member), types={"xs": "dict[str]"}), t.Refused)
    assert isinstance(t.to_python(checked("dict", member), types={"xs": "dict[str]"}, builtin_names=BUILTINS), t.Refused)


@pytest.mark.parametrize("kind", ["list", "tuple", "dict"])
@settings(max_examples=30, deadline=None)
@given(st.lists(st.integers(), max_size=12))
def test_native_render_smt_bool_int_key_equivalence_three_way(kind, items):
    value = view(kind, "int")
    types = {"xs": f"{kind}[int]"}
    predicate = t.eq(t.contains(value, t.literal(True)), t.contains(value, t.literal(1)))
    actual = dict.fromkeys(items) if kind == "dict" else (tuple(items) if kind == "tuple" else items)
    term = checked(kind, predicate)
    text = t.to_python(term, types=types, builtin_names=BUILTINS)
    assert isinstance(text, str)
    assert t.eval_term(term, {"xs": actual}) is eval(text, {}, {"xs": actual}) is True
    assert tz.implies([t.typeis(t.var("xs"), kind)], predicate, types=types) == "yes"


def test_guarded_variable_primitive_keys_use_one_source_binding():
    source = "type(xs) is dict and type(k) is str and k in xs"
    types = {"xs": "dict[str]", "k": "object"}
    term = t.from_ast(source, types, builtin_names=BUILTINS)
    assert isinstance(term, t.Term)
    assert not isinstance(tz.lower(term, types=types), t.Refused)
    text = t.to_python(term, types=types, builtin_names=BUILTINS)
    assert isinstance(text, str)
    for k in ("x", "\ud800", b"x", None):
        env = {"xs": {"x": 1, "\ud800": 2}, "k": k}
        assert t.eval_term(term, env) is eval(text, {}, env) is eval(source, {}, env)
    member = t.contains(view("dict", "str"), t.var("k"))
    assert isinstance(tz.implies([t.typeis(t.var("xs"), "dict")], member, types=types), t.Refused)


def test_views_remain_overapproximations_not_sequence_value_or_cardinality_claims():
    value = view("list", "int")
    guard, length = t.typeis(t.var("xs"), "list"), t.length(value)
    # A key-set abstraction loses order, duplicates and exact size relations.
    facts = [guard, t.eq(length, t.literal(1)), t.contains(value, t.literal(1))]
    assert tz.implies(facts, t.not_(t.contains(value, t.literal(2))), types={"xs": "list[int]"}) == "no"
    assert isinstance(tz.lower(value), t.Refused)
    assert isinstance(tz.lower(t.key_literal("x")), t.Refused)


def test_concurrent_container_implications_keep_checked_context_and_bindings_isolated():
    def query(domain):
        value = view("dict", domain)
        types = {"xs": f"dict[{domain}]"}
        key = t.key_literal("x")
        goal = t.not_(t.contains(value, key))
        expected = "no" if domain == "str" else "yes"
        for _ in range(3):
            assert tz.implies([t.typeis(t.var("xs"), "dict")], goal, types=types) == expected
        return expected

    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(query, ["str", "int", "bool", "bytes", "str", "int"])) == [
            "no", "yes", "yes", "yes", "no", "yes"]
