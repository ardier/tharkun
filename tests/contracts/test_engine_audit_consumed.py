"""Every CONSUMED cell of the audit table, run end to end against CPython 3.14.4 ([internal ref]).

``tests/contracts/frozen/<qualname>.json`` (via ``_engine_audit_expected.EXPECTED``) freezes which (contract, engine) cells are ``C``: the engine's handling of the
call is driven by the contract.  Owner rule: "for every cell that claims consumed, a test exercising that
engine on the contract end to end against native CPython; a cell that cannot pass its test is reported, not
hidden".  So per engine:

* **absint**: the sample proves with all seven classes recorded at the call as the contract's ops; withdrawing
  the contract withdraws the consumption; and a call-shape TWIN (the same call with the wrong number of
  arguments, which CPython rejects with TypeError) is never TRUE for the dynamic-typing property;
* **symbolic executor**: a guard on the call's result, built from the value CPYTHON gives, is solved and the
  witness replayed natively raises; a guard on a wrong value is not reached; the contract-driven tables
  (random range, heap length, constant) are checked against CPython directly;
* **census**: the closed-hooks rows against a HOSTILE argument that logs every ordinary attribute lookup on it
  (a row says the call touches no ordinary user attribute except the listed ones); the obligation template
  against CPython's IndexError;
* **stationary**: ``type``/``isinstance`` of a trip value;
* **Avengers / draws**: the empty-value evidence against CPython (the contract's method really raises on the
  empty container) and the constructor-built seeds against the constructors.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import re
import sys

import pytest

from tests.contracts import _engine_audit as A
from tests.contracts._engine_audit_expected import EXPECTED
from tests.contracts._engine_audit import SAMPLES
from tests.contracts._native import peer_runtime_check_type, run_native, stationary
from tharkun.solve.absint import prove
from tharkun.solve.contracts import closed_call_hooks, get
from tharkun.solve.contracts.adapters import common
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.sv.symexec_visitor import NATIVE_METHODS, NATIVE_MODULE_FUNCTIONS
from tharkun.solve.contracts.rules import get_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES

ALL = tuple(ALL_PROPERTY_CLASSES)
DT = ("TypeError", "AttributeError")


def untested(contract):
    """The contract with every claim, its return and itself marked untested: what un-testing really means (the
    adapters read a CLAIM's ``tested`` flag, so flipping the contract's alone would withdraw nothing)."""
    return dataclasses.replace(
        contract, tested=False, raises=tuple(dataclasses.replace(r, tested=False) for r in contract.raises),
        returns=None if contract.returns is None else dataclasses.replace(contract.returns, tested=False))


def cells(engine):
    return sorted(q for q, row in EXPECTED.items() if row[engine] == "C")


# --------------------------------------------------------------------------- absint
#: the absint cells that are an attribute READ, not a call (``math.pi`` ...): no ``<name>:<class>`` op is recorded at a call
_ABSINT_CONSTANTS = [q for q in cells("absint") if get_rule(q).kind == "math_constant"]


@pytest.mark.parametrize("qualname", [q for q in cells("absint") if not q.startswith("protocol.") and q not in _ABSINT_CONSTANTS])
class TestAbsintConsumes:
    def test_the_seven_classes_are_recorded_at_the_call_as_the_contracts_ops(self, qualname):
        r = prove(SAMPLES[qualname], ALL, time_budget=30.0)
        ops = {s[2] for s in r.sites}
        assert {f"{A.op_name(qualname)}:{exc}" for exc in ALL} <= ops, (qualname, sorted(ops))

    def test_withdrawing_the_contract_withdraws_the_consumption(self, qualname, monkeypatch):
        load_all()
        monkeypatch.delitem(REGISTRY, qualname)
        assert A.absint_cell(qualname, SAMPLES[qualname])[0] != "C"

    def test_an_untested_contract_proves_nothing(self, qualname, monkeypatch):
        """A builtin or a math function is gated on a TESTED contract (not consumed otherwise); a module function's
        contract is read either way, but an untested claim is never a proof: no class stays proven."""
        load_all()
        monkeypatch.setitem(REGISTRY, qualname, untested(REGISTRY[qualname]))
        if A.absint_cell(qualname, SAMPLES[qualname])[0] != "C":
            return
        r = prove(SAMPLES[qualname], ALL, time_budget=30.0)
        proven = [s for s in r.sites if s[2].startswith(f"{A.op_name(qualname)}:") and "UNPROVEN" not in s[4]]
        assert not proven, (qualname, proven)

    def test_the_sample_is_clean_on_cpython_and_its_call_shape_twin_is_not(self, qualname):
        if qualname.startswith("random."):
            pytest.skip("nondeterministic")
        assert run_native(SAMPLES[qualname], check_type_oracle=peer_runtime_check_type() if qualname == "_sv_verifier.check_type" else None) == ""
        twin = A.arity_twin(SAMPLES[qualname], qualname)
        if twin is None:
            pytest.skip("a constant or a protocol has no call shape")
        assert run_native(twin, check_type_oracle=peer_runtime_check_type() if qualname == "_sv_verifier.check_type" else None) == "TypeError"
        r = prove(twin, DT, time_budget=30.0)
        assert r.verdict != "TRUE", (qualname, twin, r.explain())


@pytest.mark.parametrize("qualname", _ABSINT_CONSTANTS)
class TestAbsintConsumesConstants:
    """A constant is consumed when withdrawing its contract changes what absint does (it is a refusal, not the old table), and an
    untested one is refused too: the value is the contract's (``AbsintVisitor.visit_math_constant``)."""

    def test_withdrawing_the_contract_changes_the_outcome(self, qualname):
        r = prove(SAMPLES[qualname], ALL, time_budget=30.0)
        assert r.verdict == "TRUE"
        assert A._outcome_without(qualname, SAMPLES[qualname]) != (r.verdict, r.reason)

    def test_an_untested_constant_is_a_named_refusal(self, qualname, monkeypatch):
        load_all()
        monkeypatch.setitem(REGISTRY, qualname, untested(REGISTRY[qualname]))
        r = prove(SAMPLES[qualname], ALL, time_budget=30.0)
        assert r.verdict == "UNKNOWN" and "no tested contract" in r.reason, (qualname, r.reason)


_LH_PREFIX = ("class A:\n    def __len__(self):\n        raise {exc}('no length')\n    def __iter__(self):\n"
              "        return iter([1, 2])\nx = list(A())\n")


@pytest.mark.parametrize("qualname", [q for q in cells("absint") if q.startswith("protocol.")])
class TestAbsintConsumesProtocols:
    """A protocol leaves no op named after it: absint consumes it when withdrawing it changes what absint does."""

    def test_withdrawing_the_protocol_changes_the_outcome(self, qualname):
        r = prove(SAMPLES[qualname], ALL, time_budget=30.0)
        assert A._outcome_without(qualname, SAMPLES[qualname]) != (r.verdict, r.reason)

    def test_an_untested_protocol_is_not_consumed(self, qualname, monkeypatch):
        load_all()
        monkeypatch.setitem(REGISTRY, qualname, untested(REGISTRY[qualname]))
        r = prove(SAMPLES[qualname], ALL, time_budget=30.0)
        # the length probe says "no verified rule"; the dispatches of a user instance ([internal ref] step 2) say "has no tested contract"
        assert r.verdict == "UNKNOWN" and ("no verified rule" in r.reason or f"{qualname} has no tested contract" in r.reason), (qualname, r.reason)

    def test_the_length_probe_follows_cpython_where_the_contract_applies(self, qualname):
        """``list(x)`` asks ``x.__len__`` and swallows ONLY a TypeError (PEP 424): the program that raises one is
        clean on CPython and absint proves it; the one that raises a ValueError dies on CPython, and absint must
        not prove it."""
        if qualname != "protocol.length_hint":
            pytest.skip("the length probe is the only protocol absint reads")
        clean = _LH_PREFIX.format(exc="TypeError")
        dies = _LH_PREFIX.format(exc="ValueError")
        assert run_native(clean) == "" and run_native(dies) == "ValueError"
        # the raise of a TypeError is itself a site of the dynamic-typing classes (over-approximated, sound), so
        # the proof asks about two classes the swallowed path cannot reach
        assert prove(clean, ("IndexError", "ZeroDivisionError"), time_budget=30.0).verdict == "TRUE"
        assert prove(dies, ("ValueError",), time_budget=30.0).verdict != "TRUE"


# --------------------------------------------------------------------------- the symbolic executor
def _assignment(src: str):
    """``(name, expression)`` when the sample's last statement assigns one name, else None."""
    last = ast.parse(src).body[-1]
    if isinstance(last, ast.Assign) and len(last.targets) == 1 and isinstance(last.targets[0], ast.Name):
        return last.targets[0].id, ast.unparse(last.value)
    return None


def _value(src: str, name: str):
    ns: dict = {}
    exec(compile(src, "<sample>", "exec"), ns)
    return ns[name]


def _solve(src: str, guard: str):
    runner, h = A.symexec_harness(src, extra=f"if {guard} and n == 3:\n    raise IndexError('audit-result-guard')\n")
    from tharkun.sv import _symexec

    import ast as _ast
    import z3  # noqa: F401

    tree = _ast.parse(h.source)
    fn = next(n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name == "__sv_harness")
    mod = {"__name__": "sv_harness_audit"}
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    env = {p: _symexec._var(p, k) for p, k in zip(h.params, h.kinds)}
    lines = [s["line"] for s in h.sites]
    out = _symexec.solve_targets(h.source, mod, "__sv_harness", lines, h.params, h.kinds, {})
    # A subscript in the sample or guard adds an earlier synthetic IndexError site.
    line = next(node.lineno for node in ast.walk(fn) if isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call) and any(isinstance(arg, ast.Constant)
                and arg.value == "audit-result-guard" for arg in node.exc.args))
    return {line: out[line]}, h


#: the symbolic cells whose consumed fact is not the call's VALUE (a range, a length precondition, a class)
TABLE_DRIVEN = {"type", "random.random", "random.choice", "heapq.heappop"} | A._OUTPUT_FILE_CONTRACTS
#: the builtins the executor evaluates itself, read through their rules (``_Exec._through_rule``; [internal ref] step 2): withdrawn or
#: untested, the call is a refusal (``tests/contracts/test_symexec_builtin_rules.py`` pins each against CPython)
SYMEXEC_RULE_BUILTINS = frozenset({"len", "abs", "min", "max", "any", "all", "isinstance", "sum", "range", "sorted", "int", "float",
                                   "divmod", "pow", "print"})
#: the container operations ``_Runner`` evaluates itself, read through their rules the same way (``test_symexec_container_rules.py``)
SYMEXEC_RULE_CONTAINERS = frozenset({"list.append", "list.__getitem__", "list.__setitem__", "dict.get", "dict.keys",
                                     "dict.__getitem__", "dict.__setitem__", "tuple.__getitem__"}) | frozenset(NATIVE_METHODS) | frozenset(NATIVE_MODULE_FUNCTIONS)


@pytest.mark.parametrize("qualname", cells("symexec"))
class TestSymbolicConsumes:
    def test_a_guard_on_the_value_cpython_gives_is_solved_and_replays(self, qualname):
        src = SAMPLES[qualname]
        shape = _assignment(src)
        if shape is None or qualname in TABLE_DRIVEN:
            pytest.skip("covered by the table-specific tests below")
        name, expr = shape
        value = _value(src, name)
        if isinstance(value, tuple) or isinstance(value, float) and not math.isfinite(value):
            pytest.skip("a tuple or non-finite result has no literal guard; covered by the math tests")
        if isinstance(value, re.Match):
            guard = f"{name}.group() == {value.group()!r} and {name}.start() == {value.start()} and {name}.end() == {value.end()}"
        elif qualname.endswith("finditer"):
            guard = f"[(m.start(), m.end()) for m in {name}] == {[m.span() for m in value]!r}"
        else:
            guard = f"{name} == {value!r}"
        got, h = _solve(src, guard)
        sat = [m for o, m in got.values() if o == "sat"]
        assert sat and sat[0]["n"] == 3, (qualname, got)
        # replay on CPython: the witness reaches the raise
        full = src + f"import _sv_verifier\nn = _sv_verifier.nondet_int()\nif {guard} and n == 3:\n    raise IndexError('hit')\n"
        assert run_native(full, (3,)) == "IndexError"

    def test_a_guard_on_a_wrong_value_is_not_reached(self, qualname):
        src = SAMPLES[qualname]
        shape = _assignment(src)
        if shape is None or qualname in TABLE_DRIVEN:
            pytest.skip("covered by the table-specific tests below")
        name, _ = shape
        value = _value(src, name)
        if isinstance(value, (tuple, list)) or isinstance(value, float) and not math.isfinite(value):
            pytest.skip("no literal guard")
        wrong = (not value) if isinstance(value, bool) else (value + 1 if isinstance(value, (int, float)) else repr(value) + "x")
        got, _h = _solve(src, f"{name} == {wrong!r}")
        assert not [m for o, m in got.values() if o == "sat"], (qualname, got)

    def test_withdrawing_the_contract_withdraws_the_consumption(self, qualname, monkeypatch):
        from tharkun.solve.contracts.adapters import concolic

        load_all()
        monkeypatch.setitem(REGISTRY, qualname, untested(REGISTRY[qualname]))
        if qualname.startswith("math."):
            name = qualname.split(".", 1)[1]
            assert concolic.math_function(name) is None and concolic.math_constant(name) is None
        elif qualname in concolic.CONCRETE_FOLD_ARITY:
            assert concolic.concrete_fold(qualname, arg_count=concolic.CONCRETE_FOLD_ARITY[qualname][0],
                                          has_keywords=False) is None
        elif qualname == "type":
            assert not concolic.type_of_one_argument(arg_count=1, has_keywords=False)
        elif qualname == "random.random":
            assert concolic.range_return(qualname) is None
        elif qualname == "heapq.heappop":
            assert concolic.empty_first_arg_precondition(qualname, arg_count=1, has_keywords=False) is None
        elif (qualname in A._OUTPUT_FILE_CONTRACTS or qualname in SYMEXEC_RULE_BUILTINS or qualname in SYMEXEC_RULE_CONTAINERS
              or qualname == "random.choice"):
            assert A.symexec_cell(qualname, SAMPLES[qualname])[0] != "C"
        elif qualname.startswith(("str.", "re.", "syntax.")) or qualname in {"format", "repr", "ascii", "str"}:
            assert A.symexec_cell(qualname, SAMPLES[qualname])[0] != "C"
        else:
            pytest.fail(f"no withdrawal check for the symbolic cell {qualname}")


_TEXT_OPEN = 'f = open("/dev/null", "w")\n'
_BIN_OPEN = 'f = open("/dev/null", "wb")\n'
#: a call of each output-file method with an argument of the wrong type: a TypeError on CPython (the contract's
#: claim), which ends the path in the executor before anything after it
_WRONG_TYPE = {
    "io.TextIOWrapper.write": _TEXT_OPEN + "f.write(1)\n",
    "io.BufferedWriter.write": _BIN_OPEN + 'f.write("a")\n',
}


@pytest.mark.parametrize("qualname", sorted(A._OUTPUT_FILE_CONTRACTS))
class TestSymbolicOutputFiles:
    """The write side of an output file ([internal ref]): ``_symexec`` reads ``open.output`` and the handle methods'
    contracts; the program after the call is reached with a model that replays on CPython."""

    def test_the_path_after_the_call_is_reached_and_replays(self, qualname):
        got, _h = _solve(SAMPLES[qualname], "True")
        sat = [m for o, m in got.values() if o == "sat"]
        assert sat and sat[0]["n"] == 3, (qualname, got)
        full = (SAMPLES[qualname] + "import _sv_verifier\nn = _sv_verifier.nondet_int()\n"
                "if n == 3:\n    raise IndexError('hit')\n")
        assert run_native(full, (3,)) == "IndexError"

    def test_a_wrong_typed_argument_ends_the_path_as_cpython_does(self, qualname):
        if qualname not in _WRONG_TYPE:
            pytest.skip("only write has a wrong-typed argument shape of its own")
        src = _WRONG_TYPE[qualname]
        assert run_native(src + "raise IndexError('after')\n") == "TypeError"
        got, _h = _solve(src, "True")
        assert not [m for o, m in got.values() if o == "sat"], (qualname, got)


class TestSymbolicTables:
    def test_type_of_a_value_is_its_class(self):
        got, _ = _solve("x = type(3)\n", "x is int")
        assert [m for o, m in got.values() if o == "sat"]
        assert type(3) is int

    def test_random_random_is_in_the_unit_interval_on_cpython_and_in_the_model(self):
        import random

        assert all(0.0 <= random.random() < 1.0 for _ in range(2000))
        got, _ = _solve("import random\nx = random.random()\n", "x >= 1.0")
        assert not [m for o, m in got.values() if o == "sat"]
        got, _ = _solve("import random\nx = random.random()\n", "x < 0.5")
        assert [m for o, m in got.values() if o == "sat"]

    def test_heappop_needs_a_non_empty_heap_on_cpython_and_in_the_model(self):
        import heapq

        with pytest.raises(IndexError):
            heapq.heappop([])
        got, _ = _solve("import heapq\nh = [1, 2]\nv = heapq.heappop(h)\n", "v == 1")
        assert [m for o, m in got.values() if o == "sat"]


# --------------------------------------------------------------------------- the census
class _Log(list):
    pass


def _hostile(log):
    """An object that does everything a native call may ask of an argument through the TYPE (dunders) and logs
    every ORDINARY attribute lookup made on the instance: the thing a closed-hooks row says never happens."""
    class Hostile:
        def __getattribute__(self, name):
            if not (name.startswith("__") and name.endswith("__")):
                log.append(name)
            return object.__getattribute__(self, name)

        def __float__(self):
            return 2.0

        def __int__(self):
            return 3

        def __index__(self):
            return 3

        def __iter__(self):
            return iter([1, 2])

        def __len__(self):
            return 2

        def __repr__(self):
            return "H"

        def __hash__(self):
            return 7

        def __eq__(self, other):
            return self is other

        def __lt__(self, other):
            return False

        def __bool__(self):
            return True

        def __abs__(self):
            return 1

        def __mul__(self, other):
            return 1

        __rmul__ = __mul__

        def __add__(self, other):
            return 1

        __radd__ = __add__

    return Hostile()


#: one native call per closed-hooks row, with the hostile object as the argument that could call something back
def _hook_call(qualname, h):
    owner, _, method = qualname.rpartition(".")
    if qualname.startswith("math."):
        fn = getattr(math, method)
        n_args = {"atan2": 2, "copysign": 2, "fmod": 2, "remainder": 2, "pow": 2, "ldexp": 2, "nextafter": 2,
                  "isclose": 2, "comb": 2, "dist": 2, "sumprod": 2, "fma": 3}.get(method, 1)
        if method in ("fsum", "prod", "dist", "sumprod"):
            return lambda: fn(*([[h]] * n_args))
        if method in ("gcd", "lcm", "hypot"):
            return lambda: fn(h, h)
        return lambda: fn(*([h] * n_args))
    table = {
        "abs": lambda: abs(h), "divmod": lambda: divmod(h, 1), "float": lambda: float(h), "int": lambda: int(h),
        "len": lambda: len(h), "list": lambda: list(h), "range": lambda: range(h), "repr": lambda: repr(h),
        "set": lambda: set(h), "dict": lambda: dict(h), "random.random": lambda: __import__("random").random(),
        "random.randrange": lambda: __import__("random").randrange(h), "random.seed": lambda: __import__("random").seed(h),
        "list.append": lambda: [].append(h), "list.clear": lambda: [h].clear(), "list.copy": lambda: [h].copy(),
        "list.pop": lambda: [h].pop(), "list.remove": lambda: [h].remove(h),
        "dict.clear": lambda: {h: 1}.clear(), "dict.copy": lambda: {h: 1}.copy(), "dict.pop": lambda: {h: 1}.pop(h),
        "dict.update": lambda: {}.update(h),
        "set.add": lambda: set().add(h), "set.clear": lambda: {h}.clear(), "set.copy": lambda: {h}.copy(),
        "set.pop": lambda: {h}.pop(), "set.remove": lambda: {h}.remove(h), "set.update": lambda: set().update(h),
        "frozenset.copy": lambda: frozenset([h]).copy(),
        "str.join": lambda: ",".join([h]), "str.find": lambda: "abc".find(h),
        "bytes.join": lambda: b",".join([h]), "bytes.find": lambda: b"abc".find(h),
        "bytearray.append": lambda: bytearray().append(h), "bytearray.clear": lambda: bytearray(b"a").clear(),
        "bytearray.copy": lambda: bytearray(b"a").copy(), "bytearray.pop": lambda: bytearray(b"a").pop(),
        "bytearray.remove": lambda: bytearray(b"a").remove(h), "bytearray.join": lambda: bytearray(b",").join([h]),
        "bytearray.find": lambda: bytearray(b"abc").find(h),
        "_sv_verifier.nondet_int": lambda: None,
    }
    return table[qualname]


_CLOSED_ROWS = sorted(q for q in EXPECTED if closed_call_hooks(q) is not None)


@pytest.mark.parametrize("qualname", _CLOSED_ROWS)
def test_a_closed_hooks_row_holds_against_a_hostile_argument(qualname):
    """The row's claim on CPython: the call looks up no ordinary attribute of its argument except the listed ones."""
    log = _Log()
    h = _hostile(log)
    try:
        _hook_call(qualname, h)()
    except Exception:  # noqa: BLE001 -- a TypeError for the hostile argument is fine; what is logged is the claim
        pass
    assert set(log) <= set(closed_call_hooks(qualname)), (qualname, log)


def test_a_closed_row_that_lists_a_hook_really_looks_it_up():
    for q, hook in (("dict", "keys"), ("dict.update", "keys")):
        log = _Log()
        try:
            _hook_call(q, _hostile(log))()
        except Exception:  # noqa: BLE001
            pass
        assert hook in log, q


@pytest.mark.parametrize("qualname", cells("census"))
def test_the_census_consumes_it(qualname):
    cell = A.census_cell(qualname, SAMPLES[qualname])
    assert cell[0] == "C"
    if closed_call_hooks(qualname) is None:
        assert "obligation" in cell[1]


def test_a_closed_row_keeps_an_unreferenced_method_dead_and_a_listed_hook_live():
    """The census drops a method nothing references when every native call in the program has a closed row; a
    row that lists a hook keeps that method live.  Both against CPython."""
    from tharkun.solve.census import site_census

    dead = "class A:\n    def hook(self):\n        return [1 // 0]\nx = len([1, 2])\ny = abs(-3)\n"
    assert run_native(dead) == ""
    c = site_census(dead, ("ZeroDivisionError",), "<t>")
    assert c.no_site, c.reason
    # a call WITHOUT a row (str.upper has none) keeps every method live, as CPython may call any of them
    unknown = "class A:\n    def hook(self):\n        return [1 // 0]\nx = 'a'.upper()\n"
    assert not site_census(unknown, ("ZeroDivisionError",), "<t>").no_site
    # random.seed lists encode/decode: isinstance sees a spoofed __class__, so CPython DOES call them
    live = ("import random\nclass A:\n    __class__ = property(lambda self: str)\n    def encode(self):\n"
            "        return 1 // 0\nrandom.seed(A())\n")
    assert run_native(live) == "ZeroDivisionError"
    assert not site_census(live, ("ZeroDivisionError",), "<t>").no_site


# --------------------------------------------------------------------------- stationary
@pytest.mark.parametrize("qualname", cells("stationary"))
class TestStationaryConsumes:
    TRIP = "import _sv_verifier\nn = abs(_sv_verifier.nondet_int())\nfor _ in range(n):\n    pass\n"

    def _program(self, qualname, tail):
        return self.TRIP + tail

    def test_a_trip_value_may_flow_into_it_and_the_proof_is_true_like_cpython(self, qualname):
        tail = "if type(n) is int:\n    k = 1\n" if qualname == "type" else "if isinstance(n, int):\n    k = 1\n"
        src = self._program(qualname, tail)
        for n in (0, 1, 5):
            assert run_native(src, (n,)) == ""
        out = stationary(src, ("ZeroDivisionError",), sys.executable)
        assert out[0] == "TRUE", out

    def test_a_violating_twin_is_found_not_proved(self, qualname):
        tail = "if type(n) is int:\n    k = 1 // 0\n" if qualname == "type" else "if isinstance(n, int):\n    k = 1 // 0\n"
        src = self._program(qualname, tail)
        assert run_native(src, (0,)) == "ZeroDivisionError"
        assert stationary(src, ("ZeroDivisionError",), sys.executable)[0] != "TRUE"

    def test_a_shadowed_name_is_not_the_builtin(self, qualname):
        name = qualname
        tail = f"def {name}(x, *a):\n    return True\nif {name}(n, int):\n    k = 1\n"
        src = self._program(qualname, tail)
        assert run_native(src, (0,)) == ""
        # the planner must not read a user's function as the builtin: the plan either refuses or the proof is
        # for the program it really sees; either way never a TRUE the builtin's contract made up
        out = stationary(src, ("ZeroDivisionError",), sys.executable)
        assert out[0] in ("TRUE", "refuses")


# --------------------------------------------------------------------------- Avengers and draws
@pytest.mark.parametrize("engine", ["avengers", "draws"])
@pytest.mark.parametrize("qualname", sorted(set(cells("avengers")) | set(cells("draws"))))
def test_the_empty_value_evidence_holds_on_cpython(qualname, engine):
    if EXPECTED[qualname][engine] != "C":
        pytest.skip("not a consumed cell of this engine")
    if qualname in ("list", "tuple", "set"):
        from tharkun.solve.contracts.adapters import avengers as av
        from tharkun.solve.contracts.adapters import draws as dr

        ctor = {"list": list, "tuple": tuple, "set": set}[qualname]
        assert dr.boundary_value_for_type(qualname) == ctor() and av.nonempty_seed_for_type(qualname) == ctor([0])
        return
    empty = A.empty_variant(SAMPLES[qualname])
    assert empty is not None, qualname
    contract = get(qualname)
    classes = [e for e in ("IndexError", "KeyError") if contract.raise_claim(e).tested
               and common.native_raise_condition(contract, e) is not None]
    assert classes
    assert run_native(empty) in classes, (qualname, empty, run_native(empty))
