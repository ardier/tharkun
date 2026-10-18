"""Native parity for exact-builtin ``len`` safety cases."""

import builtins

import pytest

from tharkun.solve.contracts.registry import get
from tharkun.solve.contracts.safety import (
    Binding, Refused, admission, case_safety, cases, raise_condition, safety,
)
from tharkun.solve.terms import INT, PY, Term, add, eval_term, literal, to_python, var


_TAGS = (
    "int", "bool", "float", "str", "NoneType", "list", "tuple", "bytes",
    "dict", "set", "frozenset",
)
_BUILTIN_NAMES = ("len", "type", "int", "bool", "float", "str", "list", "tuple",
                  "bytes", "dict", "set", "frozenset")
_BUILTINS = {name: getattr(builtins, name) for name in _BUILTIN_NAMES}


def _exception_type(func, *args):
    try:
        func(*args)
    except Exception as exc:
        return type(exc)
    return None


def _binding(receiver=None):
    return Binding(var("value", PY) if receiver is None else receiver, (), None)


@pytest.mark.parametrize("tag,values", [
    ("int", (0, -7, 10**1000)),
    ("bool", (False, True)),
    ("float", (0.0, -0.0, float("nan"), float("inf"))),
    ("str", ("", "hé𝄞")),
    ("NoneType", (None,)),
    ("list", ([], [object(), object()])),
    ("tuple", ((), (object(), object()))),
    ("bytes", (b"", b"abc")),
    ("dict", ({}, {"key": object()})),
    ("set", (set(), {object()})),
    ("frozenset", (frozenset(), frozenset({object()}))),
])
def test_len_typeerror_cases_match_native_and_rendered_predicates(tag, values):
    contract = get("len")
    assert contract is not None and contract.tested is True
    assert contract.raise_claim("TypeError").tested is True

    binding = _binding()
    rows = cases("len", "TypeError", binding)
    assert isinstance(rows, tuple) and len(rows) == len(_TAGS)
    assert isinstance(admission("len", "TypeError", binding), Term)
    assert isinstance(raise_condition("len", "TypeError", binding), Term)
    safe_term = safety("len", "TypeError", binding)
    assert isinstance(safe_term, Term)

    for value in values:
        env = {"value": value}
        active = [row for row in rows if eval_term(row.domain, env) is True]
        assert len(active) == 1 and _TAGS[rows.index(active[0])] == tag
        native_raises = _exception_type(len, value) is TypeError
        assert native_raises is (tag in {"int", "bool", "float", "NoneType"})
        assert eval_term(admission("len", "TypeError", binding), env) is True
        assert eval_term(raise_condition("len", "TypeError", binding), env) is native_raises
        assert eval_term(safe_term, env) is (not native_raises)
        assert sum(eval_term(case_safety(row), env) for row in rows) == (not native_raises)

        rendered = to_python(
            case_safety(active[0]), types={"value": tag}, builtin_names=_BUILTIN_NAMES,
        )
        if tag == "NoneType":
            assert isinstance(rendered, Refused)
            assert rendered.reason == "none_test_outside_witness_grammar"
        else:
            assert isinstance(rendered, str)
            assert eval(rendered, {"__builtins__": {}}, {**_BUILTINS, "value": value}) is (not native_raises)


@pytest.mark.parametrize("receiver,tag", [
    (literal(4), "int"), (literal(False), "bool"), (literal(3.5), "float"),
])
def test_len_numeric_literal_keeps_full_cases_and_rendered_parity(receiver, tag):
    binding = _binding(receiver)
    rows = cases("len", "TypeError", binding)
    assert isinstance(rows, tuple) and len(rows) == len(_TAGS)
    active = [row for row in rows if eval_term(row.domain, {}) is True]
    assert len(active) == 1 and _TAGS[rows.index(active[0])] == tag
    value = eval_term(receiver, {})
    native_raises = _exception_type(len, value) is TypeError
    assert native_raises
    rendered = to_python(case_safety(active[0]), builtin_names=_BUILTIN_NAMES)
    assert isinstance(rendered, str)
    assert eval(rendered, {"__builtins__": {}}, _BUILTINS) is (not native_raises)


def test_len_cases_exclude_custom_and_subclass_len_hooks():
    calls = []

    class Hook:
        def __len__(self):
            calls.append("hook")
            raise TypeError("custom len")

    class ListSubclass(list):
        def __len__(self):
            calls.append("subclass")
            raise TypeError("custom list len")

    binding = _binding()
    rows = cases("len", "TypeError", binding)
    term_admission = admission("len", "TypeError", binding)
    for value in (Hook(), ListSubclass()):
        assert _exception_type(len, value) is TypeError
        assert eval_term(term_admission, {"value": value}) is False
        assert not any(eval_term(row.domain, {"value": value}) for row in rows)
    assert calls == ["hook", "subclass"]


@pytest.mark.parametrize("binding", [
    Binding(None, (), None),
    Binding(var("value", PY), (var("arg", PY),), None),
    Binding(var("value", PY), (), "Add"),
    Binding(add(literal(1), literal(2)), (), None),
    Binding(var("value", INT), (), None),
])
def test_len_cases_refuse_non_source_bindings(binding):
    assert isinstance(cases("len", "TypeError", binding), Refused)
