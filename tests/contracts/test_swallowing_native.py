"""Protocol-level exception swallowing, pinned against CPython 3.14.4 ([internal ref] batch 5, [internal ref]).

CPython silently swallows, redirects or converts exceptions inside a dozen protocols (the length
hint, sequence-protocol iteration, ``hasattr``/``getattr``/``__getattr__``, ``__exit__``, ``NotImplemented``
dispatch, ``__missing__``, ...).  A model that gets one wrong gives a wrong TRUE.  Every case of the catalog
(``_swallowing_cases.py``) names its rule, its CPython source and its exact native outcome; this module
runs each one on the interpreter and pins the outcome, then sweeps the one mechanism the engines act on
(the length hint) across every consumer and every exception class.

The catalog is also what ``test_swallowing_engines.py`` hands to Tharkun's engines.
"""

from __future__ import annotations

import array
import collections
import functools
import itertools
import operator
import sys

import pytest

from tests.contracts._propagation import PROPERTY_CLASSES
from tests.contracts._native import run_native
from tests.contracts._swallowing_cases import AR, AS, CASES, DL, DT
from tharkun.solve.contracts.adapters import common
from tharkun.solve.contracts.registry import REGISTRY, load_all

BY_ID = {c.id: c for c in CASES}


# --------------------------------------------------------------------------- the catalog
class TestCatalog:
    def test_ids_are_unique_and_the_groups_are_the_mechanisms_named_by_the_audit(self):
        assert len(BY_ID) == len(CASES)
        assert {c.group for c in CASES} == {
            "length_hint", "iteration", "contains", "attribute", "with", "truthiness", "notimplemented",
            "hashing", "conversion"}

    def test_every_case_names_its_rule_and_its_cpython_source(self):
        for c in CASES:
            assert len(c.mechanism) > 20, c.id
            assert "::" in c.cpython or ".py" in c.cpython or "/" in c.cpython, c.id

    def test_the_property_classes_are_the_four_sv_properties(self):
        assert {c.classes for c in CASES} <= {AR, DL, DT, AS}

    def test_the_catalog_has_violating_and_clean_twins_in_every_group(self):
        for group in {c.group for c in CASES}:
            members = [c for c in CASES if c.group == group]
            assert any(c.violates for c in members) and any(not c.violates for c in members), group

    def test_a_clean_case_has_no_property_exception_and_a_violating_one_has_one(self):
        builtins = sys.modules["builtins"]
        for c in CASES:
            if not c.raises:
                assert not c.violates, c.id
                continue
            exc = getattr(builtins, c.raises)
            hits = [k for k in c.classes if issubclass(exc, getattr(builtins, k))]
            assert c.violates == bool(hits), c.id


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning",
                            "ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.id)
def test_the_native_outcome(case):
    """The uncaught exception class of the run, or "" for a clean exit, is exactly the one the catalog states."""
    assert run_native(case.src, case.vector) == case.raises, case.mechanism


# --------------------------------------------------------------------------- the length hint
def _probe(exc=None, ret=None):
    """An object iterating 1, 2 whose ``__len__`` records the call, then raises ``exc`` or returns ``ret``."""
    calls = []

    class P:
        def __iter__(self):
            return iter([1, 2])

        def __len__(self):
            calls.append("len")
            if exc is not None:
                raise exc("len")
            return ret

    return P(), calls


def _bytes_ok(x):
    return bytes(x)


CONSUMERS = {
    "list": lambda x: list(x),
    "list.extend": lambda x: [].extend(x),
    "list.__iadd__": lambda x: [].__iadd__(x),
    "sorted": lambda x: sorted(x),
    "bytes": _bytes_ok,
    "bytearray.extend": lambda x: bytearray().extend(x),
    "operator.length_hint": lambda x: operator.length_hint(x),
    "tuple": lambda x: tuple(x),
    "set": lambda x: set(x),
    "frozenset": lambda x: frozenset(x),
    "array.array": lambda x: array.array("i", x),
    "array.array.extend": lambda x: array.array("i").extend(x),
}
#: callables and constructs that the table does not list because no engine consumes them as a unit; none asks
OTHERS_THAT_NEVER_ASK = {
    "min": lambda x: min(x), "max": lambda x: max(x), "sum": lambda x: sum(x), "any": lambda x: any(x),
    "all": lambda x: all(x), "enumerate": lambda x: list(enumerate(x)), "zip": lambda x: list(zip(x)),
    "iter": lambda x: list(iter(x)), "for": lambda x: [i for i in x], "unpack": lambda x: (lambda a, b: 0)(*x),
    "xs[:] = x": lambda x: [0].__setitem__(slice(None), x), "dict.fromkeys": lambda x: dict.fromkeys(x),
    "deque": lambda x: collections.deque(x), "deque.extend": lambda x: collections.deque().extend(x),
    "chain": lambda x: list(itertools.chain(x)), "map": lambda x: list(map(abs, x)),
    "filter": lambda x: list(filter(None, x)), "reduce": lambda x: functools.reduce(operator.add, x),
    "set.update": lambda x: set().update(x), "f(*x)": lambda x: (lambda *a: a)(*x),
    "math.fsum": lambda x: __import__("math").fsum(x), "bytearray": lambda x: bytearray(x),
    "str.join": lambda x: ",".join(str(i) for i in x),
}
#: syntax that DOES ask, and that no engine models (absint refuses a starred element)
SYNTAX_THAT_ASKS = {"[*x]": lambda x: [*x], "(*x,)": lambda x: (*x,)}
SWALLOWED = {"TypeError"}


@pytest.mark.parametrize("name", sorted(common.LENGTH_HINT_RULES))
def test_the_table_rule_is_what_cpython_does(name):
    obj, calls = _probe(KeyError)
    try:
        CONSUMERS[name](obj)
    except KeyError:
        pass
    assert (calls == ["len"]) == (common.LENGTH_HINT_RULES[name] == "probe"), name


@pytest.mark.parametrize("name", sorted(OTHERS_THAT_NEVER_ASK))
def test_a_callable_outside_the_table_that_never_asks(name):
    obj, calls = _probe(KeyError)
    OTHERS_THAT_NEVER_ASK[name](obj)
    assert calls == [], name


@pytest.mark.parametrize("name", sorted(SYNTAX_THAT_ASKS))
def test_a_starred_display_asks(name):
    obj, calls = _probe(TypeError)
    SYNTAX_THAT_ASKS[name](obj)
    assert calls == ["len"]


def test_the_table_covers_every_consumer_the_catalog_names():
    assert set(CONSUMERS) == set(common.LENGTH_HINT_RULES)


@pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES) + ["ValueError", "OverflowError", "StopIteration"])
@pytest.mark.parametrize("name", sorted(n for n, r in common.LENGTH_HINT_RULES.items() if r == "probe"))
def test_a_probing_consumer_swallows_exactly_typeerror(name, exc_name):
    exc = PROPERTY_CLASSES.get(exc_name) or getattr(sys.modules["builtins"], exc_name)
    obj, calls = _probe(exc)
    if exc_name in SWALLOWED:
        CONSUMERS[name](obj)                       # swallowed: the consumer iterates with no hint
    else:
        with pytest.raises(exc):
            CONSUMERS[name](obj)
    assert calls == ["len"]


@pytest.mark.parametrize("name", sorted(common.LENGTH_HINT_RULES))
@pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
def test_a_skipping_consumer_never_calls_len_so_nothing_is_raised(name, exc_name):
    if common.LENGTH_HINT_RULES[name] != "skip":
        pytest.skip("probes")
    obj, calls = _probe(PROPERTY_CLASSES[exc_name])
    CONSUMERS[name](obj)
    assert calls == []


@pytest.mark.parametrize("name", sorted(n for n, r in common.LENGTH_HINT_RULES.items() if r == "probe"))
class TestWhatTheProbeAccepts:
    def test_a_non_int_result_is_a_swallowed_typeerror(self, name):
        for ret in (None, 1.5, "x", [1]):
            obj, calls = _probe(ret=ret)
            CONSUMERS[name](obj)
            assert calls == ["len"]

    def test_a_negative_result_is_a_valueerror(self, name):
        obj, _ = _probe(ret=-1)
        with pytest.raises(ValueError):
            CONSUMERS[name](obj)

    def test_a_result_beyond_maxsize_is_an_overflowerror(self, name):
        obj, _ = _probe(ret=2 ** 70)
        with pytest.raises(OverflowError):
            CONSUMERS[name](obj)

    def test_a_len_that_cannot_be_called_is_a_swallowed_typeerror(self, name):
        class Arity:
            def __iter__(self):
                return iter([1, 2])

            def __len__(self, extra):
                return 1

        CONSUMERS[name](Arity())

    def test_iter_runs_before_the_probe(self, name):
        order = []

        class Both:
            def __iter__(self):
                order.append("iter")
                return iter([1, 2])

            def __len__(self):
                order.append("len")
                return 2

        if name == "operator.length_hint":
            pytest.skip("length_hint does not iterate")
        CONSUMERS[name](Both())
        assert order == ["iter", "len"]


def test_the_probe_asks_the_object_not_its_iterator():
    """str.join builds its list from iter(x): the hint is asked of the iterator, which has no __len__."""
    obj, calls = _probe(KeyError)
    ",".join(str(i) for i in obj)
    assert calls == []

    class Joiner:
        def __iter__(self):
            return iter(["a", "b"])

        def __len__(self):
            raise KeyError("never")

    assert ",".join(Joiner()) == "a,b"


def test_length_hint_dunder_is_the_other_half_of_the_probe():
    class Hint:
        def __iter__(self):
            return iter([1, 2])

        def __length_hint__(self):
            raise TypeError("no hint")

    assert list(Hint()) == [1, 2]

    class Bad(Hint):
        def __length_hint__(self):
            raise KeyError("propagates")

    with pytest.raises(KeyError):
        list(Bad())

    class NotImpl(Hint):
        def __length_hint__(self):
            return NotImplemented

    assert list(NotImpl()) == [1, 2]


# --------------------------------------------------------------------------- the rule table and its gate
class TestRuleTable:
    def test_the_rules_are_probe_or_skip(self):
        assert set(common.LENGTH_HINT_RULES.values()) == {"probe", "skip"}

    @pytest.mark.parametrize("name", ["list", "sorted", "list.extend"])
    def test_a_probing_consumer_with_a_tested_contract(self, name):
        assert common.length_hint_rule(name) == "probe"

    @pytest.mark.parametrize("name", ["tuple", "set", "array.array"])
    def test_a_skipping_consumer(self, name):
        assert common.length_hint_rule(name) == "skip"

    @pytest.mark.parametrize("name", ["min", "xs[:] = x", "no_such_callable"])
    def test_an_unknown_consumer_has_no_rule_and_the_engine_refuses(self, name):
        assert common.length_hint_rule(name) is None

    @pytest.mark.parametrize("name", ["list", "sorted", "list.extend", "tuple", "set"])
    def test_withdrawing_the_contract_withdraws_the_rule(self, monkeypatch, name):
        import dataclasses

        load_all()
        monkeypatch.setitem(REGISTRY, name, dataclasses.replace(REGISTRY[name], tested=False))
        assert common.length_hint_rule(name) is None


# --------------------------------------------------------------------------- protocol.length_hint, the contract
class TestLengthHintContract:
    """``protocol.length_hint`` (``contracts/rules_protocols.py``): every claim, against the interpreter."""

    def setup_method(self):
        load_all()

    def test_it_is_total_tested_and_flags_its_callbacks(self):
        c = REGISTRY["protocol.length_hint"]
        assert c.tested and c.may_raise_other and c.effects.may_execute_callback
        assert {r.exc for r in c.raises} == set(PROPERTY_CLASSES)
        assert all(r.tested for r in c.raises)

    def test_typeerror_is_the_swallowed_class_and_the_others_propagate(self):
        c = REGISTRY["protocol.length_hint"]
        assert "SWALLOWED" in c.raise_claim("TypeError").condition
        for exc in set(PROPERTY_CLASSES) - {"TypeError"}:
            assert "propagates uncaught" in c.raise_claim(exc).condition, exc

    def test_the_machinery_itself_raises_typeerror_for_a_non_int_length_hint(self):
        class Hint:
            def __iter__(self):
                return iter([1])

            def __length_hint__(self):
                return "two"

        with pytest.raises(TypeError):
            list(Hint())
        assert "returns a non-int" in REGISTRY["protocol.length_hint"].raise_claim("TypeError").condition

    def test_operator_length_hint_returns_the_default_where_the_hint_is_swallowed_or_absent(self):
        class Swallowed:
            def __len__(self):
                raise TypeError("x")

        class Absent:
            pass

        class NotImpl:
            def __length_hint__(self):
                return NotImplemented

        for obj in (Swallowed(), Absent(), NotImpl()):
            assert operator.length_hint(obj, 7) == 7

    def test_the_result_is_non_negative(self):
        class Neg:
            def __len__(self):
                return -1

        with pytest.raises(ValueError):
            operator.length_hint(Neg(), 7)
        assert operator.length_hint([1, 2, 3]) == 3

    def test_the_side_effects_of_len_stay_on_the_swallowing_path(self):
        log = []

        class Writer:
            def __iter__(self):
                return iter([1])

            def __len__(self):
                log.append("wrote")
                raise TypeError("x")

        assert list(Writer()) == [1] and log == ["wrote"]

    def test_withdrawing_the_protocol_contract_withdraws_the_consumer_rules(self, monkeypatch):
        import dataclasses

        load_all()
        monkeypatch.setitem(REGISTRY, "protocol.length_hint",
                            dataclasses.replace(REGISTRY["protocol.length_hint"], tested=False))
        assert common.length_hint_rule("list") is None and common.length_hint_rule("tuple") is None
        monkeypatch.delitem(REGISTRY, "protocol.length_hint")
        assert common.length_hint_rule("sorted") is None
