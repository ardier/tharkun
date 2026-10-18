"""Native parity for exact-list, zero-argument ``pop`` safety cases."""

import builtins

import pytest

from tharkun.solve.contracts.registry import get
from tharkun.solve.contracts.safety import (
    Binding, Refused, admission, case_safety, cases, raise_condition, safety,
)
from tharkun.solve.terms import INT, PY, Term, add, eval_term, literal, to_python, var


_BUILTIN_NAMES = ("len", "list", "type")
_BUILTINS = {name: getattr(builtins, name) for name in _BUILTIN_NAMES}


def _binding(receiver=None, args=(), op=None):
    return Binding(var("xs", PY) if receiver is None else receiver, args, op)


def _exception_type(func, *args):
    try:
        func(*args)
    except Exception as exc:
        return type(exc)
    return None


@pytest.mark.parametrize("exception", ("IndexError", "TypeError"))
def test_zero_arg_list_pop_cases_match_native_and_rendered_predicates(exception):
    contract = get("list.pop")
    assert contract is not None and contract.tested is True
    assert contract.raise_claim(exception).tested is True
    binding = _binding()
    rows = cases("list.pop", exception, binding)
    assert isinstance(rows, tuple) and len(rows) == 1
    assert isinstance(admission("list.pop", exception, binding), Term)
    assert isinstance(raise_condition("list.pop", exception, binding), Term)
    safe_term = safety("list.pop", exception, binding)
    assert isinstance(safe_term, Term)

    class Poison:
        calls = 0

        def __index__(self):
            self.calls += 1
            raise AssertionError("pop invoked element __index__")

        def __len__(self):
            self.calls += 1
            raise AssertionError("pop invoked element __len__")

        def __eq__(self, other):
            self.calls += 1
            raise AssertionError("pop compared elements")

        def __hash__(self):
            self.calls += 1
            raise AssertionError("pop hashed an element")

    poison = Poison()
    for values in ([], [poison], [poison, poison]):
        env = {"xs": values}
        native_exception = _exception_type(list.pop, list(values))
        native_raises = native_exception is getattr(builtins, exception)
        assert native_raises is (exception == "IndexError" and not values)
        assert eval_term(admission("list.pop", exception, binding), env) is True
        assert eval_term(raise_condition("list.pop", exception, binding), env) is native_raises
        assert eval_term(safe_term, env) is (not native_raises)
        assert eval_term(case_safety(rows[0]), env) is (not native_raises)

        rendered = to_python(
            case_safety(rows[0]), types={"xs": "list"}, builtin_names=_BUILTIN_NAMES,
        )
        assert isinstance(rendered, str)
        assert eval(rendered, {"__builtins__": {}}, {**_BUILTINS, "xs": values}) is (not native_raises)

    assert poison.calls == 0


def test_zero_arg_list_pop_excludes_subclasses_and_custom_receivers():
    class PopSubclass(list):
        def pop(self):
            raise TypeError("custom pop")

    binding = _binding()
    rows = cases("list.pop", "IndexError", binding)
    assert isinstance(rows, tuple) and len(rows) == 1
    assert eval_term(admission("list.pop", "IndexError", binding), {"xs": PopSubclass([1])}) is False
    assert _exception_type(lambda: PopSubclass([1]).pop()) is TypeError

    custom = object()
    assert eval_term(admission("list.pop", "IndexError", binding), {"xs": custom}) is False


@pytest.mark.parametrize("binding", [
    Binding(None, (), None),
    Binding(var("xs", PY), (var("index", PY),), None),
    Binding(var("xs", PY), (), "Add"),
    Binding(literal(1), (), None),
    Binding(add(literal(1), literal(2)), (), None),
    Binding(var("xs", INT), (), None),
])
@pytest.mark.parametrize("exception", ("IndexError", "TypeError"))
def test_zero_arg_list_pop_refuses_non_exact_receiver_shape(binding, exception):
    assert isinstance(cases("list.pop", exception, binding), Refused)
