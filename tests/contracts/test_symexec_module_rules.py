"""The symbolic executor runs functions of ``bisect``, ``heapq``, ``copy`` and ``itertools`` NATIVELY by their contract rules ([internal ref] step 2).

``bisect.bisect_left`` / ``insort``, ``heapq.heappush`` / ``heapify`` / ``nlargest`` / ``nsmallest``, ``copy.copy`` / ``deepcopy`` and
``itertools.chain`` / ``product`` / ``combinations`` / ``permutations`` / ``islice`` were refused (the audit's ``R`` cell: "name bisect": a
module is not a value of the executor).  They go through the rule now (``_Runner._contract_call`` -> ``_through_rule`` ->
``SymexecVisitor._builtin`` -> ``_Runner._native_module_function``): CPython's own function (``Rule.native``) on concrete operands, a function
that mutates its first argument on a COPY with the name rebound (``_mutable``), an iterator back as a one-shot ``_LazyIter`` capped at 4096 items.
The module is the one the NAME is bound to in the harness (identity: ``import heapq as hq`` works, a local called ``heapq`` does not).

Pinned against CPython 3.14.4: the differential table (every name the program binds ends with CPython's value, or the executor refuses, or ends
the path where CPython raises ``IndexError`` / ``KeyError`` / ``ValueError``); module identity; withdrawal; symbolic elements refused where the
function compares; NEGATIVE CONTROL: a model that forgets to rebind the mutated name is caught by the table.
"""

from __future__ import annotations

import ast
import bisect
import copy
import heapq
import itertools

import pytest

from tests.contracts.test_engine_audit_consumed import untested
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.sv import _symexec
from tharkun.sv.symexec_visitor import NATIVE_MODULE_FUNCTIONS

MODULES = {"bisect": bisect, "heapq": heapq, "hq": heapq, "copy": copy, "itertools": itertools}

PROGRAMS = [
    "i = bisect.bisect_left([1, 3, 5], 3)\n", "i = bisect.bisect_left([1, 3, 5], 4)\n", "i = bisect.bisect_left([1, 3, 5], 0)\n",
    "i = bisect.bisect_left([1, 3, 5], 3, 1)\n", "i = bisect.bisect_left([1, 3, 5], 'a')\n", "xs = [1, 3]\nbisect.insort(xs, 2)\n",
    "xs = [1, 3]\nbisect.insort(xs, 3)\n", "xs = []\nbisect.insort(xs, 1)\n", "xs = [1, 2]\nys = xs\nbisect.insort(xs, 0)\n",
    "h = []\nheapq.heappush(h, 3)\nheapq.heappush(h, 1)\n", "h = [3, 1, 2]\nheapq.heapify(h)\n", "h = [3, 1, 2]\nhq.heapify(h)\n",
    "h = [3, 1, 2]\nheapq.heappush(h, 'a')\n", "a = heapq.nlargest(2, [3, 1, 2])\n", "a = heapq.nsmallest(2, [3, 1, 2])\n",
    "a = heapq.nsmallest(0, [3, 1, 2])\n", "a = heapq.nlargest(5, (1, 2))\n",
    "ys = copy.copy([1, 2])\n", "ys = copy.copy({1: 2})\n", "ys = copy.copy((1, 2))\n", "ys = copy.copy(5)\n", "xs = [1, [2]]\nys = copy.copy(xs)\n",
    "xs = [1, [2]]\nys = copy.deepcopy(xs)\n", "ys = copy.deepcopy({1: (2, 3)})\n", "ys = copy.deepcopy('ab')\n",
    "a = list(itertools.chain([1, 2], (3,), 'ab'))\n", "a = list(itertools.chain())\n", "a = list(itertools.product([1, 2], 'ab'))\n",
    "a = list(itertools.product([1, 2], repeat=2))\n", "a = list(itertools.combinations([1, 2, 3], 2))\n", "a = list(itertools.combinations([1, 2], 3))\n",
    "a = list(itertools.combinations([1, 2], -1))\n", "a = list(itertools.permutations([1, 2, 3]))\n", "a = list(itertools.permutations([1, 2, 3], 2))\n",
    "a = list(itertools.islice([1, 2, 3, 4], 2))\n", "a = list(itertools.islice([1, 2, 3, 4], 1, 3))\n", "a = list(itertools.islice([1, 2, 3, 4], 0, None, 2))\n",
    "a = list(itertools.islice([1, 2, 3], -1))\n", "a = list(itertools.product(range(70), range(70)))\n",
]


def _names(src: str) -> list:
    return sorted({n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)})


def _native(src: str):
    ns: dict = dict(MODULES)
    try:
        exec(compile(src, "<case>", "exec"), ns)
    except BaseException as e:  # noqa: BLE001
        return False, type(e)
    return True, {k: ns[k] for k in _names(src) if k in ns}


def _run(src: str, globals_=None):
    tree = ast.parse("def __sv_harness():\n    pass\n")
    runner = _symexec._Runner(tree, dict(MODULES if globals_ is None else globals_), {}, {}, [])
    runner.run(list(ast.parse(src).body))
    return runner


def _force(v):
    return v.take() if isinstance(v, _symexec._LazyIter) else v


@pytest.mark.parametrize("src", PROGRAMS)
def test_the_executor_agrees_with_cpython_or_refuses_or_ends_the_path(src):
    ok, native = _native(src)
    try:
        runner = _run(src)
    except _symexec._Dead:
        assert not ok, ("the executor ended the path but CPython runs the program to its end", src)
        return
    if runner.skipped or runner.poisoned:
        return                                  # refused by name: says nothing false
    assert ok, f"CPython raises {native.__name__} but the executor ran the program to its end: {src!r}"
    for name, value in native.items():
        got = _force(runner.env.get(name, _symexec.POISON))
        if got is _symexec.POISON:
            continue
        assert type(got) is type(value) and got == value, (src, name, got, value)


def test_the_table_is_not_vacuous():
    answered = 0
    for src in PROGRAMS:
        if not _native(src)[0]:
            continue
        try:
            r = _run(src)
        except _symexec._Dead:
            continue
        answered += 0 if (r.skipped or r.poisoned) else 1
    assert answered >= 30, answered


def test_every_function_has_a_rule_a_tested_contract_and_is_answered_by_some_program():
    from tharkun.solve.contracts.rules import get_rule

    load_all()
    for q in NATIVE_MODULE_FUNCTIONS:
        assert get_rule(q) is not None and REGISTRY[q].tested, q
        module, name = q.split(".")
        assert any(f"{module}.{name}(" in p and _native(p)[0] and not _run_skipped(p) for p in PROGRAMS), q


def _run_skipped(src):
    try:
        r = _run(src)
    except _symexec._Dead:
        return False
    return bool(r.skipped or r.poisoned)


def test_the_module_is_the_one_the_name_is_bound_to_not_the_spelling():
    # an alias of the real module works (``hq``, in the table above)...
    assert not _run_skipped("h = [3, 1, 2]\nhq.heapify(h)\n")
    # ...a name that is NOT the module never is: a local list called ``bisect``, a name bound to another module, a name the harness lacks
    assert _run_skipped("bisect = [1, 3]\ni = bisect.bisect_left([1, 3, 5], 3)\n")
    assert _run_skipped("i = bisect.bisect_left([1, 3, 5], 3)\n") is False       # control: the real one is answered
    r = _run("i = bisect.bisect_left([1, 3, 5], 3)\n", globals_={"bisect": itertools})
    assert r.skipped, "a name bound to another module is refused"
    r = _run("i = bisect.bisect_left([1, 3, 5], 3)\n", globals_={})
    assert r.skipped, "an unbound name is refused"


@pytest.mark.parametrize("qualname", sorted(NATIVE_MODULE_FUNCTIONS))
def test_an_untested_or_withdrawn_contract_is_a_refusal(qualname, monkeypatch):
    module, name = qualname.split(".")
    src = next(p for p in PROGRAMS if f"{module}.{name}(" in p and _native(p)[0] and not _run_skipped(p))
    load_all()
    monkeypatch.setitem(REGISTRY, qualname, untested(REGISTRY[qualname]))
    assert _run_skipped(src), (qualname, "untested: refused")
    monkeypatch.undo()
    load_all()
    monkeypatch.delitem(REGISTRY, qualname)
    assert _run_skipped(src), (qualname, "withdrawn: refused")


def test_a_symbolic_element_is_refused_where_the_function_compares_and_allowed_where_it_does_not():
    import z3

    tree = ast.parse("def __sv_harness():\n    pass\n")
    sym = z3.Int("v")
    runner = _symexec._Runner(tree, dict(MODULES), {"v": sym}, {}, [])
    runner.run(list(ast.parse("i = bisect.bisect_left([1, v], 1)\n").body))
    assert runner.skipped, "bisect over a symbolic element must not be answered"
    runner = _symexec._Runner(tree, dict(MODULES), {"v": sym}, {}, [])
    runner.run(list(ast.parse("a = list(itertools.chain([1, v], [2]))\nys = copy.copy([v, 2])\n").body))
    assert not runner.skipped and len(runner.env["a"]) == 3 and runner.env["ys"][0] is sym, runner.skipped


def test_negative_control_a_model_that_forgets_to_rebind_the_mutated_name_is_caught(monkeypatch):
    """The control: ``bisect.insort`` run on the copy but never bound back leaves ``xs`` as it was; the table's ``xs == [1, 2, 3]``
    check fails for such a model."""
    real = _symexec._Runner._native_module_function

    def forgetful(self, qualname, n):
        before = dict(self.env)
        out = real(self, qualname, n)
        self.env.update(before)
        return out

    monkeypatch.setattr(_symexec._Runner, "_native_module_function", forgetful)
    runner = _run("xs = [1, 3]\nbisect.insort(xs, 2)\n")
    assert runner.env["xs"] == [1, 3] != [1, 2, 3]
