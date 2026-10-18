"""Native parity for exact-builtin sequence subscription safety cases."""

import builtins
import operator

import pytest

from tharkun.solve.contracts.registry import get
from tharkun.solve.contracts.safety import (
    Binding, Refused, admission, case_safety, cases, raise_condition, safety,
)
from tharkun.solve.terms import INT, PY, Term, add, as_int, eval_term, literal, to_python, var


_FAMILIES = {
    "list.__getitem__": (list, operator.getitem, ([], [object(), object(), object()])),
    "tuple.__getitem__": (tuple, operator.getitem, ((), (object(), object(), object()))),
    "str.__getitem__": (str, operator.getitem, ("", "aé𝄞")),
}
_KEYS = (
    ("int", (-10**1000, -4, -3, -1, 0, 2, 3, 10**1000)),
    ("bool", (False, True)),
    ("float", (0.0, -0.0, 2.5, float("nan"))),
    ("str", ("", "index")),
    ("NoneType", (None,)),
)
_BUILTIN_NAMES = ("len", "type", "list", "tuple", "str", "int", "bool", "float")
_BUILTINS = {name: getattr(builtins, name) for name in _BUILTIN_NAMES}


def _binding(key=None):
    return Binding(var("receiver", PY), (var("key", PY) if key is None else key,), None)


def _exception_type(func, *args):
    try:
        func(*args)
    except Exception as exc:
        return type(exc)
    return None


@pytest.mark.parametrize("qualname", _FAMILIES)
@pytest.mark.parametrize("exception", ("IndexError", "TypeError"))
def test_sequence_getitem_cases_match_native_and_rendered_predicates(qualname, exception):
    receiver_type, getitem, receivers = _FAMILIES[qualname]
    kind = receiver_type.__name__
    contract = get(qualname)
    assert contract is not None and contract.tested is True
    assert contract.raise_claim(exception).tested is True

    binding = _binding()
    rows = cases(qualname, exception, binding)
    assert isinstance(rows, tuple) and len(rows) == 5
    assert all(row.domain.sort.value == "Bool" for row in rows)
    term_admission = admission(qualname, exception, binding)
    term_raise = raise_condition(qualname, exception, binding)
    term_safety = safety(qualname, exception, binding)
    assert all(isinstance(term, Term) for term in (term_admission, term_raise, term_safety))

    for receiver in receivers:
        for key_tag, representatives in _KEYS:
            for key in representatives:
                env = {"receiver": receiver, "key": key}
                active = [eval_term(row.domain, env) for row in rows]
                assert sum(value is True for value in active) == 1
                native_raises = _exception_type(getitem, receiver, key) is getattr(builtins, exception)
                assert eval_term(term_admission, env) is True
                assert eval_term(term_raise, env) is native_raises
                assert eval_term(term_safety, env) is (not native_raises)
                assert sum(eval_term(case_safety(row), env) for row in rows) == (not native_raises)

                rendered = to_python(
                    case_safety(rows[active.index(True)]),
                    types={"receiver": kind}, builtin_names=_BUILTIN_NAMES,
                )
                if key_tag == "NoneType":
                    assert isinstance(rendered, Refused)
                    assert rendered.reason == "none_test_outside_witness_grammar"
                else:
                    assert isinstance(rendered, str)
                    assert eval(rendered, {"__builtins__": {}}, {**_BUILTINS, **env}) is (not native_raises)


@pytest.mark.parametrize("qualname", _FAMILIES)
@pytest.mark.parametrize("key", (literal(-4), literal(True), literal(2.5)))
def test_sequence_getitem_numeric_literals_keep_all_five_cases(qualname, key):
    receiver_type, getitem, (_, receiver) = _FAMILIES[qualname]
    binding = _binding(key)
    for exception in ("IndexError", "TypeError"):
        rows = cases(qualname, exception, binding)
        assert isinstance(rows, tuple) and len(rows) == 5
        active = [row for row in rows if eval_term(row.domain, {"receiver": receiver}) is True]
        assert len(active) == 1
        native_key = eval_term(key, {})
        native_raises = _exception_type(getitem, receiver, native_key) is getattr(builtins, exception)
        assert eval_term(raise_condition(qualname, exception, binding), {"receiver": receiver}) is native_raises
        rendered = to_python(
            case_safety(active[0]), types={"receiver": receiver_type.__name__},
            builtin_names=_BUILTIN_NAMES,
        )
        assert isinstance(rendered, str)
        assert eval(rendered, {"__builtins__": {}}, {**_BUILTINS, "receiver": receiver}) is eval_term(
            case_safety(active[0]), {"receiver": receiver}
        )


@pytest.mark.parametrize("qualname", _FAMILIES)
def test_sequence_length_does_not_inspect_elements_or_call_hooks(qualname):
    receiver_type, getitem, _ = _FAMILIES[qualname]
    if qualname == "str.__getitem__":
        receiver = "é𝄞"
    else:
        class Hook:
            calls = 0

            def __index__(self):
                self.calls += 1
                raise AssertionError("sequence length invoked an element's __index__")

            def __len__(self):
                self.calls += 1
                raise AssertionError("sequence length invoked an element's __len__")

            def __eq__(self, other):
                self.calls += 1
                raise AssertionError("sequence length compared elements")

        items = [Hook(), Hook()]
        receiver = tuple(items) if receiver_type is tuple else items

    bound = _binding()
    term = safety(qualname, "IndexError", bound)
    assert isinstance(term, Term)
    assert eval_term(term, {"receiver": receiver, "key": 1}) is True
    if qualname == "str.__getitem__":
        assert getitem(receiver, 1) == receiver[1]
    else:
        assert all(item.calls == 0 for item in items)
        assert getitem(receiver, 1) is receiver[1]


@pytest.mark.parametrize("qualname", _FAMILIES)
def test_sequence_getitem_excludes_custom_indices_slices_and_subclass_receivers(qualname):
    receiver_type, getitem, (_, receiver) = _FAMILIES[qualname]

    class IndexLike:
        def __index__(self):
            return 0

    class IntChild(int):
        pass

    class ReceiverChild(receiver_type):
        def __getitem__(self, key):
            raise TypeError("subclass hook")

    binding = _binding()
    admitted = admission(qualname, "TypeError", binding)
    assert isinstance(admitted, Term)
    assert eval_term(admitted, {"receiver": receiver, "key": IndexLike()}) is False
    assert eval_term(admitted, {"receiver": receiver, "key": IntChild(0)}) is False
    if qualname == "str.__getitem__":
        assert getitem(receiver, IndexLike()) == getitem(receiver, 0)
        assert getitem(receiver, IntChild(0)) == getitem(receiver, 0)
    else:
        assert getitem(receiver, IndexLike()) is getitem(receiver, 0)
        assert getitem(receiver, IntChild(0)) is getitem(receiver, 0)
    assert eval_term(admitted, {"receiver": receiver, "key": slice(None)}) is False
    assert eval_term(admitted, {"receiver": ReceiverChild(receiver), "key": 0}) is False


@pytest.mark.parametrize("qualname", _FAMILIES)
@pytest.mark.parametrize(
    "binding",
    [
        Binding(None, (var("key", PY),), None),
        Binding(var("receiver", PY), (), None),
        Binding(var("receiver", PY), (var("key", PY),), "Subscript"),
        Binding(var("receiver", INT), (var("key", PY),), None),
        Binding(add(var("left", INT), literal(1)), (var("key", PY),), None),
        Binding(var("receiver", PY), (var("key", INT),), None),
        Binding(var("receiver", PY), (as_int(var("key", PY)),), None),
    ],
)
def test_sequence_getitem_refuses_non_source_binding_shapes(qualname, binding):
    assert isinstance(cases(qualname, "IndexError", binding), Refused)
    assert isinstance(cases(qualname, "TypeError", binding), Refused)
