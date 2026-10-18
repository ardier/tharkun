"""The symbolic executor runs the methods of the builtin containers NATIVELY by their contract rules ([internal ref] step 2, PR "symexec methods").

``list.count`` / ``index`` / ``copy`` / ``clear`` / ``extend`` / ``insert`` / ``pop`` / ``remove`` / ``reverse`` / ``sort``,
``dict.items`` / ``values`` / ``copy`` / ``clear`` / ``pop`` / ``popitem`` / ``setdefault`` / ``update`` and ``tuple.count`` / ``index`` had no
model (the audit's ``R`` cell).  They go through the rule now (``_Runner._ev_call`` -> ``_through_rule`` -> ``SymexecVisitor._builtin`` ->
``_Exec._native_method``): CPython's own method (``Rule.native``) on a COPY of the receiver, concrete operands only, the receiver's name rebound
when the method mutates it, and only where the name is the frame's own and the object is held nowhere else (``_Exec._mutable``): an aliased list is
a named refusal, which is also the fix of the divergence PR "symexec containers" pinned as a strict xfail.

Pinned against CPython 3.14.4: the DIFFERENTIAL table (the executor ends with exactly the values CPython computes for every name, or refuses, or
ends the path where CPython raises ``IndexError`` / ``KeyError`` / ``ValueError``); aliasing (``ys = xs`` then a mutation) is refused, never applied
to one name; a symbolic element where the method compares is refused; symbolic guards on a value the method moved are solved and replayed; the
contract admits the call (untested and withdrawn refuse); NEGATIVE CONTROL: a model that mutates the receiver in place and rebinds nothing is
caught by the table.
"""

from __future__ import annotations

import ast

import pytest

from tests.contracts._native import run_native
from tests.contracts.test_engine_audit_consumed import _solve, untested
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.sv import _symexec
from tharkun.sv.symexec_visitor import NATIVE_METHODS

PROGRAMS = [
    # list
    "xs = [3, 1, 2, 1]\nn = xs.count(1)\n", "xs = [3, 1, 2]\ni = xs.index(2)\n", "xs = [3, 1, 2]\ni = xs.index(5)\n",
    "xs = [3, 1, 2]\nys = xs.copy()\nys.append(4)\n", "xs = [3, 1]\nxs.clear()\n", "xs = [3, 1]\nxs.extend([4, 5])\n",
    "xs = [3, 1]\nxs.extend('ab')\n", "xs = [3, 1]\nxs.extend({7: 1})\n", "xs = [3, 1]\nxs.insert(0, 9)\n", "xs = [3, 1]\nxs.insert(-5, 9)\n",
    "xs = [3, 1, 2]\nv = xs.pop()\n", "xs = [3, 1, 2]\nv = xs.pop(0)\n", "xs = []\nv = xs.pop()\n", "xs = [3, 1]\nv = xs.pop(5)\n",
    "xs = [3, 1, 2]\nxs.remove(1)\n", "xs = [3, 1, 2]\nxs.remove(7)\n", "xs = [3, 1, 2]\nxs.reverse()\n", "xs = [3, 1, 2]\nxs.sort()\n",
    "xs = [3, 1, 2]\nxs.sort(reverse=True)\n", "xs = [3, 'a']\nxs.sort()\n", "xs = [3.5, 1, 2]\nxs.sort()\n", "xs = ['b', 'a']\nxs.sort()\n",
    "xs = [1, 2]\nys = xs\nxs.append(3)\n", "xs = [1, 2]\nys = xs\nxs.pop()\n", "xs = [1, 2]\nys = [xs]\nxs.append(3)\n",
    "xs = [1, 2]\nxs.extend(xs)\n",
    # dict
    "d = {1: 2, 3: 4}\nit = d.items()\n", "d = {1: 2, 3: 4}\nvs = d.values()\n", "d = {1: 2}\ne = d.copy()\ne[5] = 6\n", "d = {1: 2}\nd.clear()\n",
    "d = {1: 2, 3: 4}\nv = d.pop(1)\n", "d = {1: 2}\nv = d.pop(7)\n", "d = {1: 2}\nv = d.pop(7, 0)\n", "d = {1: 2}\nkv = d.popitem()\n",
    "d = {}\nkv = d.popitem()\n", "d = {1: 2}\nv = d.setdefault(1, 9)\n", "d = {1: 2}\nv = d.setdefault(5, 9)\n", "d = {1: 2}\nd.update({3: 4, 1: 0})\n",
    "d = {1: 2}\ne = d\nd.update({3: 4})\n",
    # tuple
    "t = (1, 2, 1)\nn = t.count(1)\n", "t = (1, 2, 1)\ni = t.index(2)\n", "t = (1, 2, 1)\ni = t.index(7)\n",
    "xs = [(1, 2), (3, 4), (1, 2)]\nn = xs.count((1, 2))\n", "xs = [(1, (2, 3)), (4, 5)]\ni = xs.index((4, 5))\n",
]
ALIASED = {"xs = [1, 2]\nys = xs\nxs.append(3)\n", "xs = [1, 2]\nys = xs\nxs.pop()\n", "xs = [1, 2]\nys = [xs]\nxs.append(3)\n",
           "d = {1: 2}\ne = d\nd.update({3: 4})\n"}


def _names(src: str) -> list:
    return sorted({n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)})


def _native(src: str):
    ns: dict = {}
    try:
        exec(compile(src, "<case>", "exec"), ns)
    except BaseException as e:  # noqa: BLE001
        return False, type(e)
    return True, {k: ns[k] for k in _names(src) if k in ns}


def _run(src: str):
    tree = ast.parse("def __sv_harness():\n    pass\n")
    runner = _symexec._Runner(tree, {}, {}, {}, [])
    runner.run(list(ast.parse(src).body))
    return runner


def _norm(v):
    return list(v) if isinstance(v, (type({}.keys()), type({}.values()), type({}.items()))) else v


@pytest.mark.parametrize("src", PROGRAMS)
def test_the_executor_agrees_with_cpython_or_refuses_or_ends_the_path(src):
    ok, native = _native(src)
    try:
        runner = _run(src)
    except _symexec._Dead:
        assert not ok, ("the executor ended the path but CPython runs the program to its end", src)
        return
    if runner.skipped or runner.poisoned:
        return                                   # refused by name: says nothing false
    assert ok, f"CPython raises {native.__name__} but the executor ran the program to its end: {src!r}"
    assert src not in ALIASED, ("an aliased mutation was applied to one name", src)
    for name, value in native.items():
        got = runner.env.get(name, _symexec.POISON)
        if got is _symexec.POISON:
            continue
        value = _norm(value)
        assert type(got) is type(value) and got == value, (src, name, got, value)


def test_the_aliased_mutations_are_refused_by_name_not_applied_to_one_name():
    for src in ALIASED:
        runner = _run(src)
        assert runner.skipped, ("an aliased mutation must be refused", src)
        assert any("aliasing" in s or "reachable" in s for s in runner.skipped), runner.skipped


def test_the_table_is_not_vacuous():
    answered = 0
    for src in PROGRAMS:
        ok, _ = _native(src)
        try:
            r = _run(src)
        except _symexec._Dead:
            continue
        if ok and not (r.skipped or r.poisoned):
            answered += 1
    assert answered >= 30, answered


@pytest.mark.parametrize("qualname", sorted(NATIVE_METHODS))
def test_every_native_method_has_a_rule_and_a_tested_contract(qualname):
    from tharkun.solve.contracts.rules import get_rule

    assert get_rule(qualname) is not None
    load_all()
    assert REGISTRY[qualname].tested


@pytest.mark.parametrize("qualname,src", [
    ("list.count", "xs = [3, 1, 2, 1]\nn = xs.count(1)\n"), ("list.index", "xs = [3, 1, 2]\ni = xs.index(2)\n"),
    ("list.copy", "xs = [3, 1, 2]\nys = xs.copy()\n"), ("list.clear", "xs = [3, 1]\nxs.clear()\n"),
    ("list.extend", "xs = [3, 1]\nxs.extend([4, 5])\n"), ("list.insert", "xs = [3, 1]\nxs.insert(0, 9)\n"),
    ("list.pop", "xs = [3, 1, 2]\nv = xs.pop()\n"), ("list.remove", "xs = [3, 1, 2]\nxs.remove(1)\n"),
    ("list.reverse", "xs = [3, 1, 2]\nxs.reverse()\n"), ("list.sort", "xs = [3, 1, 2]\nxs.sort()\n"),
    ("dict.items", "d = {1: 2}\nit = d.items()\n"), ("dict.values", "d = {1: 2}\nvs = d.values()\n"),
    ("dict.copy", "d = {1: 2}\ne = d.copy()\n"), ("dict.clear", "d = {1: 2}\nd.clear()\n"), ("dict.pop", "d = {1: 2}\nv = d.pop(1)\n"),
    ("dict.popitem", "d = {1: 2}\nkv = d.popitem()\n"), ("dict.setdefault", "d = {1: 2}\nv = d.setdefault(5, 9)\n"),
    ("dict.update", "d = {1: 2}\nd.update({3: 4})\n"), ("tuple.count", "t = (1, 2, 1)\nn = t.count(1)\n"),
    ("tuple.index", "t = (1, 2, 1)\ni = t.index(2)\n"),
])
def test_an_untested_or_withdrawn_contract_is_a_refusal(qualname, src, monkeypatch):
    r = _run(src)
    if qualname in ("dict.items", "dict.values"):
        assert r.skipped, "live views require alias tracking even with a tested contract"
    else:
        assert not (r.skipped or r.poisoned), (qualname, "answered while tested", r.skipped)
    load_all()
    monkeypatch.setitem(REGISTRY, qualname, untested(REGISTRY[qualname]))
    assert _run(src).skipped, (qualname, "untested: refused")
    monkeypatch.undo()
    load_all()
    monkeypatch.delitem(REGISTRY, qualname)
    assert _run(src).skipped, (qualname, "withdrawn: refused")


def test_a_symbolic_element_is_refused_where_the_method_compares_and_allowed_where_it_does_not():
    cmp_src = "import _sv_verifier\nv = _sv_verifier.nondet_int()\nxs = [1, v]\nn = xs.count(1)\n"
    keep_src = "import _sv_verifier\nv = _sv_verifier.nondet_int()\nxs = [1, 2]\nxs.append(v)\nxs.reverse()\nys = xs.copy()\n"
    got, _ = _solve(cmp_src, "n == 2")
    assert not [m for o, m in got.values() if o == "sat"], "count over a symbolic element must not be answered"
    got, h = _solve(keep_src, "ys[0] == 5")
    sat = [m for o, m in got.values() if o == "sat"]
    assert sat and sat[0]["n"] == 3, got
    vector = [sat[0][p] for p in h.params]
    full = (keep_src + "import _sv_verifier\nn = _sv_verifier.nondet_int()\nif ys[0] == 5 and n == 3:\n    raise IndexError('hit')\n")
    assert run_native(full, vector) == "IndexError", vector


def test_negative_control_without_the_sharing_guard_the_aliased_mutation_is_applied_to_one_name(monkeypatch):
    """The control: with ``_mutable`` returning the object whoever else holds it, ``ys = xs; xs.pop()`` is no longer refused and is applied to
    ``xs`` only (``ys`` keeps both elements, CPython's ``ys`` is ``[1]``): ``test_the_aliased_mutations_are_refused_...`` fails for such a model."""
    monkeypatch.setattr(_symexec._Exec, "_mutable", lambda self, name, kinds: self.env.get(name))
    runner = _run("xs = [1, 2]\nys = xs\nxs.pop()\n")
    assert not runner.skipped and runner.env["xs"] == [1] and runner.env["ys"] == [1, 2], (runner.skipped, runner.env)
