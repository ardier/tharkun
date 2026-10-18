"""CPython 3.14.4 parity for typed, tested safety cases (plan item 1.4)."""

import ast
import operator
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tharkun.solve.contracts.registry import get
from tharkun.solve.contracts import registry
from tharkun.solve.contracts.safety.rules_protocols import _INSTANCE_ATTRS
from tharkun.solve.contracts.safety import _runtime
from tharkun.solve.contracts.safety.rules_protocols import protocol_cases
from tharkun.solve.contracts.safety import (
    Binding,
    admission,
    binding_from_record,
    case_safety,
    cases,
    raise_condition,
    safety,
)
from tharkun.solve.terms import BOOL, INT, PY, Refused, Term, add, as_int, canon, eval_term, literal, to_python, var

_VALUES = {
    "int": 2,
    "bool": True,
    "float": 2.5,
    "str": "x",
    "NoneType": None,
}
_TAGS = tuple(_VALUES)
_REPRESENTATIVES = {
    **{tag: (value,) for tag, value in _VALUES.items()},
    "float": (2.5, -2.5, 0.0, -0.0, float("inf"), -float("inf"), float("nan")),
}
_COMPARE = {
    "Lt": operator.lt, "LtE": operator.le, "Gt": operator.gt,
    "GtE": operator.ge, "Eq": operator.eq, "NotEq": operator.ne,
}
_BINARY = {
    "Add": operator.add, "Sub": operator.sub, "Mult": operator.mul,
    "MatMult": operator.matmul, "Div": operator.truediv,
    "FloorDiv": operator.floordiv, "Mod": operator.mod, "Pow": operator.pow,
    "LShift": operator.lshift, "RShift": operator.rshift,
    "BitAnd": operator.and_, "BitXor": operator.xor, "BitOr": operator.or_,
}
_UNARY = {"UAdd": operator.pos, "USub": operator.neg, "Invert": operator.invert}
_NUMERIC_ZERO_OPS = {
    "FloorDiv": operator.floordiv,
    "Mod": operator.mod,
    "Div": operator.truediv,
}


def _binding(op: str, arity: int) -> Binding:
    return Binding(var("left", PY), tuple(var(f"arg{i}", PY) for i in range(arity)), op)


def _exception_type(func, *args):
    try:
        func(*args)
    except Exception as exc:
        return type(exc)
    return None


@pytest.mark.parametrize(
    "qualname,op,arity,func",
    [
        ("protocol.comparison", op, 1, func)
        for op, func in _COMPARE.items()
    ] + [
        ("protocol.binary_operator", op, 1, func)
        for op, func in _BINARY.items()
    ] + [
        ("protocol.unary_operator", op, 0, func)
        for op, func in _UNARY.items()
    ],
)
def test_native_typeerror_iff_term_on_each_admitted_scalar_case(qualname, op, arity, func):
    contract = get(qualname)
    assert contract is not None and contract.tested is True
    assert contract.raise_claim("TypeError").tested is True

    binding = _binding(op, arity)
    case_rows = cases(qualname, "TypeError", binding)
    assert isinstance(case_rows, tuple) and case_rows
    assert all(isinstance(row.domain, Term) and row.domain.sort == BOOL for row in case_rows)
    assert all(row.raises is None or (isinstance(row.raises, Term) and row.raises.sort == BOOL)
               for row in case_rows)
    term_admission = admission(qualname, "TypeError", binding)
    term_raise = raise_condition(qualname, "TypeError", binding)
    term_safety = safety(qualname, "TypeError", binding)
    assert isinstance(term_admission, Term) and isinstance(term_raise, Term) and isinstance(term_safety, Term)

    if arity == 0:
        for tag in _TAGS:
            for value in _REPRESENTATIVES[tag]:
                env = {"left": value}
                active = [eval_term(row.domain, env) for row in case_rows]
                assert sum(value is True for value in active) == 1
                assert eval_term(term_admission, env) is True
                native_raises = _exception_type(func, env["left"]) is TypeError
                assert eval_term(term_raise, env) is native_raises
                assert eval_term(term_safety, env) is (not native_raises)
                assert sum(eval_term(case_safety(row), env) for row in case_rows) == (not native_raises)
        return

    for left_tag in _TAGS:
        for left in _REPRESENTATIVES[left_tag]:
            for right_tag in _TAGS:
                for right in _REPRESENTATIVES[right_tag]:
                    env = {"left": left, "arg0": right}
                    active = [eval_term(row.domain, env) for row in case_rows]
                    expected_admitted = not (op == "Mod" and left_tag == "str")
                    assert sum(value is True for value in active) == int(expected_admitted)
                    assert eval_term(term_admission, env) is expected_admitted
                    native_raises = _exception_type(func, env["left"], env["arg0"]) is TypeError
                    if expected_admitted:
                        assert eval_term(term_raise, env) is native_raises
                        assert eval_term(term_safety, env) is (not native_raises)
                        assert sum(eval_term(case_safety(row), env) for row in case_rows) == (not native_raises)
                    else:
                        # The native TypeError result varies with string format contents;
                        # no tag-only claim is made for the excluded left-string domain.
                        assert eval_term(term_safety, env) is False


def test_cases_are_in_fixed_disjoint_tag_order():
    binding = _binding("Add", 1)
    rows = cases("protocol.binary_operator", "TypeError", binding)
    observed = []
    for row in rows:
        matches = []
        for left_tag in _TAGS:
            for right_tag in _TAGS:
                env = {"left": _VALUES[left_tag], "arg0": _VALUES[right_tag]}
                if eval_term(row.domain, env) is True:
                    matches.append((left_tag, right_tag))
        assert len(matches) == 1
        observed.append(matches[0])
    assert observed == [(left, right) for left in _TAGS for right in _TAGS]


def test_custom_and_subclass_hooks_are_outside_admission():
    class Hook:
        def __lt__(self, other):
            raise TypeError("custom comparison")

        def __eq__(self, other):
            raise TypeError("custom equality")

        def __add__(self, other):
            raise TypeError("custom addition")

    class IntChild(int):
        def __add__(self, other):
            raise TypeError("subclass addition")

    for op, left in (("Lt", Hook()), ("Eq", Hook()), ("Add", Hook()), ("Add", IntChild(2))):
        qualname = "protocol.comparison" if op in {"Lt", "Eq"} else "protocol.binary_operator"
        term = admission(qualname, "TypeError", _binding(op, 1))
        assert isinstance(term, Term)
        assert eval_term(term, {"left": left, "arg0": 1}) is False
        assert _exception_type(_COMPARE[op] if op in _COMPARE else _BINARY[op], left, 1) is TypeError


def test_value_and_non_typeerror_boundaries_remain_outside_scalar_claims():
    # String interpolation depends on format contents and the right-hand value.
    binding = _binding("Mod", 1)
    admitted = admission("protocol.binary_operator", "TypeError", binding)
    assert isinstance(admitted, Term)
    assert eval_term(admitted, {"left": "%s", "arg0": "x"}) is False
    assert eval_term(admitted, {"left": "%s", "arg0": (1,)}) is False
    assert "%s" % "x" == "x"
    assert "%s" % (1,) == "1"
    for value in _VALUES.values():
        assert eval_term(admitted, {"left": "%s", "arg0": value}) is False
        assert _exception_type(operator.mod, "%s", value) is not TypeError
    for tag in ("int", "bool", "float"):
        assert _exception_type(operator.mod, "%d", _VALUES[tag]) is not TypeError
    for value in ("x", None):
        assert _exception_type(operator.mod, "%d", value) is TypeError
    for value in _VALUES.values():
        assert _exception_type(operator.mod, "literal", value) is TypeError

    # Other native exceptions are not relabeled as protocol TypeError.
    assert _exception_type(operator.truediv, 1, 0) is ZeroDivisionError
    with pytest.raises(ZeroDivisionError):
        operator.truediv(1, 0)
    with pytest.raises(ValueError):
        operator.mod("%q", "x")
    assert isinstance(raise_condition("protocol.binary_operator", "ValueError", _binding("Mod", 1)), Refused)


def test_iteration_typeerror_cases_cover_creation_for_exact_scalar_types():
    contract = get("protocol.iteration")
    assert contract is not None and contract.raise_claim("TypeError").tested is True
    binding = Binding(var("value", PY), (), None)
    rows = cases("protocol.iteration", "TypeError", binding)
    assert isinstance(rows, tuple) and len(rows) == len(_TAGS)

    representatives = {
        "int": (0, -7, 10**40),
        "bool": (False, True),
        "float": (0.0, -0.0, 1.5, float("nan"), float("inf"), -float("inf")),
        "str": ("", "plain", "λ"),
        "NoneType": (None,),
    }
    type_builtins = ("type", "int", "bool", "float", "str")
    namespace = {"type": type, "int": int, "bool": bool, "float": float, "str": str}
    for tag in _TAGS:
        for value in representatives[tag]:
            env = {"value": value}
            active = [eval_term(row.domain, env) for row in rows]
            assert sum(result is True for result in active) == 1
            native_raises = _exception_type(iter, value) is TypeError
            assert native_raises is (tag != "str")
            assert eval_term(admission("protocol.iteration", "TypeError", binding), env) is True
            assert eval_term(raise_condition("protocol.iteration", "TypeError", binding), env) is native_raises
            assert eval_term(safety("protocol.iteration", "TypeError", binding), env) is (not native_raises)
            active_row = rows[active.index(True)]
            rendered = to_python(case_safety(active_row), builtin_names=type_builtins)
            if tag == "NoneType":
                assert isinstance(rendered, Refused)
                assert rendered.reason == "none_test_outside_witness_grammar"
            else:
                assert isinstance(rendered, str)
                assert eval(rendered, namespace, env) is eval_term(case_safety(active_row), env)


@pytest.mark.parametrize("value", [0, -7, 10**40, False, True, 0.0, -0.0, 2.5])
def test_iteration_numeric_literal_bindings_are_unconditionally_noniterable(value):
    contract = get("protocol.iteration")
    assert contract is not None and contract.raise_claim("TypeError").tested is True
    binding = Binding(literal(value), (), None)
    rows = cases("protocol.iteration", "TypeError", binding)
    assert isinstance(rows, tuple) and len(rows) == len(_TAGS)
    active = [row for row in rows if eval_term(row.domain, {}) is True]
    assert len(active) == 1
    assert eval_term(active[0].raises, {}) is True
    rendered = to_python(case_safety(active[0]), builtin_names=("type", "int", "bool", "float", "str"))
    assert isinstance(rendered, str)
    assert eval(rendered, {"type": type, "int": int, "bool": bool, "float": float, "str": str}, {}) is False
    assert _exception_type(iter, value) is TypeError


def test_iteration_builder_refuses_other_shapes_and_does_not_claim_hook_objects():
    base = Binding(var("value", PY), (), None)
    assert isinstance(cases("protocol.iteration", "TypeError", Binding(None, (), None)), Refused)
    assert isinstance(cases("protocol.iteration", "TypeError", Binding(var("value", PY), (var("arg", PY),), None)), Refused)
    assert isinstance(cases("protocol.iteration", "TypeError", Binding(var("value", PY), (), "Iter")), Refused)
    assert isinstance(cases("protocol.iteration", "TypeError", Binding(var("value", INT), (), None)), Refused)
    assert isinstance(cases("protocol.iteration", "TypeError", Binding(add(var("left", INT), var("right", INT)), (), None)), Refused)

    class Hook:
        def __iter__(self):
            raise TypeError("user hook")

    class IntChild(int):
        def __iter__(self):
            return iter((int(self),))

    admitted = admission("protocol.iteration", "TypeError", base)
    assert isinstance(admitted, Term)
    for value, native_raises in ((Hook(), True), (IntChild(1), False)):
        assert eval_term(admitted, {"value": value}) is False
        assert (_exception_type(iter, value) is TypeError) is native_raises


def test_iteration_creation_is_distinct_from_old_sequence_and_next_errors():
    class OldStyleSequence:
        def __getitem__(self, index):
            raise TypeError("sequence callback")

    class BadIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise TypeError("next callback")

    admitted = admission("protocol.iteration", "TypeError", Binding(var("value", PY), (), None))
    assert isinstance(admitted, Term)
    for value in (OldStyleSequence(), BadIterator()):
        assert eval_term(admitted, {"value": value}) is False
        iterator = iter(value)
        assert _exception_type(next, iterator) is TypeError


def test_truthiness_typeerror_cases_cover_exact_scalar_values():
    contract = get("protocol.truthiness")
    assert contract is not None and contract.raise_claim("TypeError").tested is True
    binding = Binding(var("value", PY), (), None)
    rows = cases("protocol.truthiness", "TypeError", binding)
    assert isinstance(rows, tuple) and len(rows) == len(_TAGS)
    assert all(row.raises is None for row in rows)
    representatives = {
        "int": (0, -7, 10**40),
        "bool": (False, True),
        "float": (0.0, -0.0, 1.5, float("nan"), float("inf"), -float("inf")),
        "str": ("", "plain", "λ"),
        "NoneType": (None,),
    }
    namespace = {"type": type, "int": int, "bool": bool, "float": float, "str": str}
    for tag in _TAGS:
        for value in representatives[tag]:
            env = {"value": value}
            active = [row for row in rows if eval_term(row.domain, env) is True]
            assert len(active) == 1
            assert type(bool(value)) is bool
            assert _exception_type(bool, value) is None
            assert eval_term(admission("protocol.truthiness", "TypeError", binding), env) is True
            assert eval_term(raise_condition("protocol.truthiness", "TypeError", binding), env) is False
            assert eval_term(safety("protocol.truthiness", "TypeError", binding), env) is True
            rendered = to_python(case_safety(active[0]), builtin_names=tuple(namespace))
            if tag == "NoneType":
                assert isinstance(rendered, Refused)
                assert rendered.reason == "none_test_outside_witness_grammar"
            else:
                assert isinstance(rendered, str)
                assert eval(rendered, namespace, env) is eval_term(case_safety(active[0]), env)


@pytest.mark.parametrize("value", [0, -7, 10**40, False, True, 0.0, -0.0, 2.5])
def test_truthiness_numeric_literal_binding_keeps_full_five_cases(value):
    contract = get("protocol.truthiness")
    assert contract is not None and contract.raise_claim("TypeError").tested is True
    binding = Binding(literal(value), (), None)
    rows = cases("protocol.truthiness", "TypeError", binding)
    assert isinstance(rows, tuple) and len(rows) == len(_TAGS)
    active = [row for row in rows if eval_term(row.domain, {}) is True]
    assert len(active) == 1 and active[0].raises is None
    assert _exception_type(bool, value) is None
    rendered = to_python(case_safety(active[0]), builtin_names=("type", "int", "bool", "float", "str"))
    assert isinstance(rendered, str)
    namespace = {"type": type, "int": int, "bool": bool, "float": float, "str": str}
    assert eval(rendered, namespace, {}) is eval_term(case_safety(active[0]), {})


def test_truthiness_hooks_and_subclasses_are_outside_exact_scalar_admission():
    class BadBool:
        def __bool__(self):
            return 1

    class IntChild(int):
        def __bool__(self):
            raise TypeError("subclass bool hook")

    binding = Binding(var("value", PY), (), None)
    admitted = admission("protocol.truthiness", "TypeError", binding)
    assert isinstance(admitted, Term)
    for value in (BadBool(), IntChild(1)):
        assert eval_term(admitted, {"value": value}) is False
        assert _exception_type(bool, value) is TypeError
    assert isinstance(cases("protocol.truthiness", "ValueError", binding), Refused)


@pytest.mark.parametrize(
    "binding",
    [
        Binding(None, (), None),
        Binding(var("value", PY), (var("arg", PY),), None),
        Binding(var("value", PY), (), "Not"),
        Binding(var("value", INT), (), None),
        Binding(as_int(var("value", PY)), (), None),
        Binding(add(var("value", INT), literal(1)), (), None),
    ],
)
def test_truthiness_template_refuses_non_source_bindings(binding):
    result = cases("protocol.truthiness", "TypeError", binding)
    assert isinstance(result, Refused)
    assert result.reason == "truthiness binding requires one source receiver, no args, and no operator"


@pytest.mark.parametrize("op,func", _NUMERIC_ZERO_OPS.items())
def test_numeric_zero_cases_match_native_exact_scalar_boundary(op, func):
    contract = get("protocol.binary_operator")
    assert contract is not None
    assert contract.raise_claim("ZeroDivisionError").tested is True
    binding = _binding(op, 1)
    rows = cases("protocol.binary_operator", "ZeroDivisionError", binding)
    assert isinstance(rows, tuple) and len(rows) == 26
    term_admission = admission("protocol.binary_operator", "ZeroDivisionError", binding)
    term_raise = raise_condition("protocol.binary_operator", "ZeroDivisionError", binding)
    term_safety = safety("protocol.binary_operator", "ZeroDivisionError", binding)
    assert isinstance(term_admission, Term)
    assert isinstance(term_raise, Term)
    assert isinstance(term_safety, Term)

    boundary = (1 << 1024) - (1 << 970)
    representatives = {
        "int": (0, 1, -3, boundary - 1, boundary, -boundary, 10**1000),
        "bool": (False, True),
        "float": (0.0, -0.0, 3.5, float("inf"), -float("inf"), float("nan")),
        "str": ("", "x", "%s", "%d", "%q"),
        "NoneType": (None,),
    }
    for left_tag, left_values in representatives.items():
        for right_tag, right_values in representatives.items():
            for left in left_values:
                for right in right_values:
                    env = {"left": left, "arg0": right}
                    active = [row for row in rows if eval_term(row.domain, env) is True]
                    assert len(active) == 1, (op, left_tag, right_tag, left, right)
                    native_raises = _exception_type(func, left, right) is ZeroDivisionError
                    row = active[0]
                    abstract_raises = row.raises is not None and eval_term(row.raises, env) is True
                    assert abstract_raises is native_raises, (op, left_tag, right_tag, left, right)
                    assert eval_term(term_admission, env) is True
                    assert eval_term(term_raise, env) is native_raises
                    assert eval_term(term_safety, env) is (not native_raises)
                    assert eval_term(case_safety(row), env) is (not native_raises)

    rendered = to_python(case_safety(rows[0]), builtin_names=("type", "int", "bool", "float", "str"))
    if isinstance(rendered, Refused):
        assert rendered.reason == "cast lacks dominating exact type guard"
    else:
        assert isinstance(rendered, str)
        namespace = {"type": type, "int": int, "bool": bool, "float": float, "str": str}
        for left_tag, left_values in representatives.items():
            for right_tag, right_values in representatives.items():
                for left in left_values:
                    for right in right_values:
                        env = {"left": left, "arg0": right}
                        expected = eval_term(case_safety(rows[0]), env)
                        assert eval(rendered, namespace, env) is expected


@pytest.mark.parametrize(
    "op,func,left,right",
    [
        (op, func, left, right)
        for op, func in _NUMERIC_ZERO_OPS.items()
        for left, right in (
            (1, 0), (True, False), (2.0, -0.0),
            (9, 2), (True, True), (9.0, 2.5),
            ((1 << 1024) - (1 << 970) - 1, 0.0),
            ((1 << 1024) - (1 << 970), 0.0),
        )
    ],
)
def test_numeric_zero_cases_handle_literal_operands(op, func, left, right):
    binding = Binding(literal(left), (literal(right),), op)
    rows = cases("protocol.binary_operator", "ZeroDivisionError", binding)
    assert isinstance(rows, tuple) and len(rows) == 26
    env = {}
    active = [row for row in rows if eval_term(row.domain, env) is True]
    assert len(active) == 1
    native_raises = _exception_type(func, left, right) is ZeroDivisionError
    row = active[0]
    assert (row.raises is not None and eval_term(row.raises, env) is True) is native_raises
    assert eval_term(safety("protocol.binary_operator", "ZeroDivisionError", binding), env) is (not native_raises)


@pytest.mark.parametrize("op,func", _NUMERIC_ZERO_OPS.items())
@pytest.mark.parametrize(
    "left,right",
    [
        (1, 0), (1, 2), (True, False), (True, True),
        (2.0, -0.0), (2.0, 3.5),
    ],
)
def test_numeric_zero_cases_handle_mixed_variable_and_literal_operands(op, func, left, right):
    binding = Binding(var("left", PY), (literal(right),), op)
    rows = cases("protocol.binary_operator", "ZeroDivisionError", binding)
    assert isinstance(rows, tuple) and len(rows) == 26
    env = {"left": left}
    active = [row for row in rows if eval_term(row.domain, env) is True]
    assert len(active) == 1
    native_raises = _exception_type(func, left, right) is ZeroDivisionError
    row = active[0]
    assert (row.raises is not None and eval_term(row.raises, env) is True) is native_raises
    assert eval_term(safety("protocol.binary_operator", "ZeroDivisionError", binding), env) is (not native_raises)


def test_numeric_zero_templates_refuse_other_shapes_and_property_classes():
    binding = _binding("FloorDiv", 1)
    assert isinstance(protocol_cases("protocol.binary_operator", "TypeError", binding), tuple)
    assert isinstance(protocol_cases("protocol.comparison", "ZeroDivisionError", binding), Refused)
    assert isinstance(cases("protocol.binary_operator", "TypeError", binding), tuple)
    assert isinstance(cases("protocol.binary_operator", "ZeroDivisionError", _binding("Add", 1)), Refused)
    assert isinstance(cases("protocol.binary_operator", "ZeroDivisionError", _binding("Div", 0)), Refused)
    assert isinstance(cases("protocol.binary_operator", "ZeroDivisionError", "not a Binding"), Refused)


@pytest.mark.parametrize(
    "receiver",
    [var("left", INT), as_int(var("left", PY)), add(var("left", INT), var("right", INT))],
)
def test_numeric_zero_binding_refuses_non_source_or_computed_operands(receiver):
    binding = Binding(receiver, (var("arg0", PY),), "FloorDiv")
    assert isinstance(cases("protocol.binary_operator", "ZeroDivisionError", binding), Refused)


def test_unsupported_shapes_and_unregistered_claims_refuse():
    assert isinstance(raise_condition("protocol.binary_operator", "TypeError", _binding("Unknown", 1)), Refused)
    assert isinstance(raise_condition("protocol.binary_operator", "TypeError", _binding("Add", 0)), Refused)
    assert isinstance(raise_condition("protocol.binary_operator", "IndexError", _binding("Add", 1)), Refused)
    assert isinstance(cases("protocol.binary_operator", "TypeError", "not a Binding"), Refused)


def test_binding_record_uses_only_the_shared_bounded_term_decoder():
    encoded = {
        "receiver": canon(var("left", PY)),
        "args": [canon(var("right", PY))],
        "op": "Add",
    }
    decoded = binding_from_record(encoded)
    assert decoded == Binding(var("left", PY), (var("right", PY),), "Add")
    assert isinstance(binding_from_record({**encoded, "extra": 1}), Refused)
    assert isinstance(binding_from_record({**encoded, "receiver": "(term 999)"}), Refused)


@pytest.mark.skipif(not _runtime.supports_cpython_314(), reason="pinned attribute inventory is for CPython 3.14.x")
@pytest.mark.parametrize("tag,value", [
    ("int", 0), ("int", -3), ("int", 10**1000),
    ("bool", False), ("bool", True),
    ("float", 0.0), ("float", -0.0), ("float", float("inf")),
    ("float", -float("inf")), ("float", float("nan")),
    ("str", ""), ("str", "λ"), ("NoneType", None),
])
def test_attribute_cases_match_exact_builtin_instance_attribute_inventory(tag, value):
    binding = Binding(var("receiver", PY), (), "Attribute:__m2_missing_attribute__")
    sentinels = {"__m2_missing_attribute__", "not_an_attribute"}
    actual_names = set(dir(value))
    assert _INSTANCE_ATTRS[tag] == actual_names - sentinels
    names = actual_names | _INSTANCE_ATTRS[tag] | sentinels | {"another_missing_name"}
    for name in names:
        bound = Binding(binding.receiver, (), f"Attribute:{name}")
        attr_rows = cases("protocol.attribute_access", "AttributeError", bound)
        type_rows = cases("protocol.attribute_access", "TypeError", bound)
        assert isinstance(attr_rows, tuple) and len(attr_rows) == 5
        assert isinstance(type_rows, tuple) and len(type_rows) == 5
        env = {"receiver": value}
        assert sum(eval_term(row.domain, env) is True for row in attr_rows) == 1
        try:
            getattr(value, name)
        except AttributeError:
            native_error = True
        except Exception as exc:  # exact builtins must not take custom hooks
            pytest.fail(f"unexpected {type(exc).__name__} for {tag}.{name}")
        else:
            native_error = False
        active = next(row for row in attr_rows if eval_term(row.domain, env) is True)
        assert (active.raises is not None and eval_term(active.raises, env) is True) is native_error, (tag, name, native_error)
        assert all(row.raises is None for row in type_rows)
        assert eval_term(raise_condition("protocol.attribute_access", "TypeError", bound), env) is False


@pytest.mark.skipif(not _runtime.supports_cpython_314(), reason="pinned attribute inventory is for CPython 3.14.x")
def test_attribute_binding_refuses_non_ast_shapes_and_keeps_none_rendering_store_only():
    receiver = var("receiver", PY)
    assert isinstance(cases("protocol.attribute_access", "AttributeError", Binding(receiver, (receiver,), "Attribute:real")), Refused)
    assert isinstance(cases("protocol.attribute_access", "AttributeError", Binding(receiver, (), "real")), Refused)
    assert isinstance(cases("protocol.attribute_access", "AttributeError", Binding(receiver, (), "Attribute:bad-name")), Refused)
    none_case = cases("protocol.attribute_access", "TypeError", Binding(receiver, (), "Attribute:__class__"))[4]
    assert eval_term(none_case.domain, {"receiver": None}) is True
    assert isinstance(to_python(case_safety(none_case), builtin_names=("type", "int", "bool", "float", "str")), Refused)


@pytest.mark.parametrize("implementation,version", [("pypy", (3, 14, 0)), ("cpython", (3, 13, 10))])
def test_attribute_templates_refuse_outside_cpython_314(monkeypatch, implementation, version):
    monkeypatch.setattr(_runtime, "sys", SimpleNamespace(
        implementation=SimpleNamespace(name=implementation), version_info=version,
    ))
    refused = cases(
        "protocol.attribute_access", "AttributeError",
        Binding(var("receiver", PY), (), "Attribute:real"),
    )
    assert isinstance(refused, Refused)
    assert refused.reason == "attribute template requires CPython 3.14"


def test_raiseclaim_tested_flag_is_a_required_template_gate(monkeypatch):
    contract = get("protocol.binary_operator")
    assert contract is not None
    withdrawn = replace(
        contract,
        raises=tuple(
            replace(claim, tested=False) if claim.exc == "TypeError" else claim
            for claim in contract.raises
        ),
        tested=False,
    )
    monkeypatch.setitem(registry.REGISTRY, contract.qualname, withdrawn)
    binding = _binding("Add", 1)
    assert isinstance(cases(contract.qualname, "TypeError", binding), Refused)
    assert isinstance(raise_condition(contract.qualname, "TypeError", binding), Refused)


def test_tested_never_claim_keeps_domain_but_has_no_raise_condition(monkeypatch):
    contract = get("protocol.binary_operator")
    assert contract is not None
    binding = _binding("Add", 1)
    original = cases(contract.qualname, "TypeError", binding)
    never = replace(
        contract,
        raises=tuple(
            replace(claim, condition="never") if claim.exc == "TypeError" else claim
            for claim in contract.raises
        ),
    )
    monkeypatch.setitem(registry.REGISTRY, contract.qualname, never)
    rows = cases(contract.qualname, "TypeError", binding)
    assert isinstance(rows, tuple)
    assert tuple(row.domain for row in rows) == tuple(row.domain for row in original)
    assert all(row.raises is None for row in rows)
    assert eval_term(raise_condition(contract.qualname, "TypeError", binding),
                     {"left": 1, "arg0": 2}) is False
    assert eval_term(safety(contract.qualname, "TypeError", binding),
                     {"left": 1, "arg0": 2}) is True
    assert isinstance(cases(contract.qualname, "TypeError", _binding("Unknown", 1)), Refused)


@pytest.mark.parametrize(
    "receiver",
    [var("left", INT), as_int(var("left", PY)), add(var("left", INT), var("right", INT))],
)
def test_operator_bindings_refuse_non_source_or_computed_operands(receiver):
    binding = Binding(receiver, (var("arg0", PY),), "Add")
    assert isinstance(cases("protocol.binary_operator", "TypeError", binding), Refused)


def test_heappop_stays_byte_for_byte_compatible_until_sequence_terms_exist():
    from tharkun.solve.contracts.adapters.obligations import obligation_for_call

    node = ast.parse("heapq.heappop(h)", mode="eval").body
    out = obligation_for_call("heapq.heappop", node, file_name="t.py", function="f")
    assert out is not None and out.invariant is not None
    assert out.invariant.value == "len(h) > 0"
    heap = Binding(var("h"), (), None)
    assert isinstance(cases("heapq.heappop", "IndexError", heap), Refused)

    invalid = ast.parse("heapq.heappop(h=h)", mode="eval").body
    gap = obligation_for_call("heapq.heappop", invalid, file_name="t.py", function="f")
    assert gap is not None and gap.gap is not None
    assert gap.gap.reason.startswith("call shape does not match")
