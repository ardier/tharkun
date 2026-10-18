"""The admitted callable TypeError cases and the explicit current Terms gap."""

import pytest

from tharkun.solve.contracts.safety import Binding, admission, cases, raise_condition
from tharkun.solve.terms import INT, PY, Refused, add, as_int, eval_term, key_literal, var

_NONCALLABLES = {
    "int": 1,
    "bool": True,
    "float": 1.0,
    "str": "x",
    "NoneType": None,
}
_UNMODELED = (
    "AttributeError", "AssertionError", "FloatingPointError", "IndexError",
    "KeyError", "ZeroDivisionError",
)


def _binding(receiver=None, args=(), op="Call"):
    return Binding(var("callee", PY) if receiver is None else receiver, tuple(args), op)


@pytest.mark.parametrize("tag,value", _NONCALLABLES.items())
def test_exact_noncallable_scalar_case_matches_native_typeerror(tag, value):
    binding = _binding(args=(var("arg", PY), key_literal("stable")))
    rows = cases("protocol.call", "TypeError", binding)
    assert isinstance(rows, tuple) and len(rows) == 5
    active = [eval_term(row.domain, {"callee": value, "arg": object()}) for row in rows]
    assert active.count(True) == 1
    assert next(row for row, chosen in zip(rows, active) if chosen).raises is not None
    assert eval_term(admission("protocol.call", "TypeError", binding), {"callee": value, "arg": 0}) is True
    assert eval_term(raise_condition("protocol.call", "TypeError", binding), {"callee": value, "arg": 0}) is True
    with pytest.raises(TypeError):
        value(object(), "stable")


def test_callable_targets_are_outside_the_scalar_admission():
    binding = _binding()
    domain = admission("protocol.call", "TypeError", binding)
    assert not isinstance(domain, Refused)
    assert eval_term(domain, {"callee": lambda: 1}) is False
    assert eval_term(domain, {"callee": object()}) is False


@pytest.mark.parametrize("exception", _UNMODELED)
def test_each_propagation_builder_is_registered_with_the_specific_terms_gap(exception):
    result = cases("protocol.call", exception, _binding())
    assert isinstance(result, Refused)
    assert "tp_call" in result.reason and "metaclass" in result.reason


def test_computed_or_cast_call_bindings_refuse_the_partial_case():
    computed_arg = add(var("left", INT), var("right", INT))
    assert isinstance(cases("protocol.call", "TypeError", _binding(args=(computed_arg,))), Refused)
    assert isinstance(cases("protocol.call", "TypeError", _binding(receiver=as_int(var("casted", PY)))), Refused)
    assert isinstance(cases("protocol.call", "TypeError", _binding(op="NotCall")), Refused)
