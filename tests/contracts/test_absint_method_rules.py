"""absint reads the contracts of the methods of the builtin heap kinds it models by hand ([internal ref] step 2, PR "absint containers").

``list.append``, ``list.extend``, ``list.insert``, ``list.pop``, ``list.reverse``, ``dict.get``, ``dict.keys``, ``dict.values`` and
``io.StringIO.write`` / ``read`` / ``getvalue`` / ``seek`` had a transfer function (``Analyzer.list_method`` / ``dict_method`` /
``stringio_method``) that never read the contract of the callable (the audit's ``M`` cell).  They now run through their rule:
``Analyzer.<kind>_method`` -> ``visit_method_rule`` -> ``AbsintVisitor.visit_<kind>`` (a :class:`MethodCtx`) ->
``Analyzer._contract_method`` -> the unchanged model under ``Analyzer._modelled``, which records the SEVEN classes as the contract
states them and upgrades a class only on an argument the model made (``_METHOD_HOOKS`` for the user-hook classes, the model's own
checks for the call's native ones: ``list.pop``'s ``IndexError``).

What this file pins, against CPython 3.14.4: for every program of the table, absint is never TRUE for a class the program raises
natively and is TRUE for every class of a clean program; the call site records the contract's seven ops (``<qualname>:<class>``),
with the native class UNPROVEN where the call raises it; the contract admits the model (withdrawn, the call is a named refusal);
NEGATIVE CONTROLS: a ``_modelled`` that proves every class whatever the model checked is caught.
"""

from __future__ import annotations

import pytest

from tests.contracts._native import run_native
from tharkun.solve.absint import prove
from tharkun.solve.absint.interp import Analyzer
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES

ALL = tuple(ALL_PROPERTY_CLASSES)
#: the subscripts record through ``_proven_classes`` after the model ran once (``_through_subscript_rules``), not through ``_modelled``
SUBSCRIPTS = frozenset({"list.__getitem__", "list.__setitem__", "tuple.__getitem__", "dict.__getitem__", "dict.__setitem__"})
METHODS = ("list.append", "list.extend", "list.insert", "list.pop", "list.reverse", "dict.get", "dict.keys", "dict.values",
           "io.StringIO.write", "io.StringIO.read", "io.StringIO.getvalue", "io.StringIO.seek",
           "list.__getitem__", "list.__setitem__", "tuple.__getitem__", "dict.__getitem__", "dict.__setitem__",
           "list.copy", "list.clear", "list.count", "list.index", "list.remove", "list.sort", "tuple.count", "tuple.index")
L = "xs = [3, 1, 2]\n"
D = "d = {1: 2}\n"
IO = "import io\nf = io.StringIO()\n"
T = "t = (4, 5, 6)\n"

#: the call's own class, besides TypeError, of the methods that have one
NATIVE_OTHER = {"list.pop": ("IndexError",), "list.index": (), "list.sort": (), "list.__getitem__": ("IndexError",), "list.__setitem__": ("IndexError",),
                "tuple.__getitem__": ("IndexError",), "dict.__getitem__": ("KeyError",)}

#: (method, program, the class CPython raises ("" = clean))
CASES = [
    ("list.append", L + "xs.append(4)\n", ""),
    ("list.append", L + "xs.append()\n", "TypeError"),
    ("list.append", L + "xs.append(1, 2)\n", "TypeError"),
    ("list.extend", L + "xs.extend([4, 5])\n", ""),
    ("list.extend", L + "xs.extend(range(2))\n", ""),
    ("list.extend", L + "xs.extend()\n", "TypeError"),
    ("list.extend", L + "xs.extend(5)\n", "TypeError"),
    ("list.insert", L + "xs.insert(0, 9)\n", ""),
    ("list.insert", L + "xs.insert(10, 9)\n", ""),
    ("list.insert", L + "xs.insert(9)\n", "TypeError"),
    ("list.insert", L + "xs.insert('a', 9)\n", "TypeError"),
    ("list.pop", L + "v = xs.pop()\n", ""),
    ("list.pop", L + "v = xs.pop(0)\n", ""),
    ("list.pop", L + "v = xs.pop(-3)\n", ""),
    ("list.pop", "xs = []\nv = xs.pop()\n", "IndexError"),
    ("list.pop", L + "v = xs.pop(5)\n", "IndexError"),
    ("list.pop", L + "v = xs.pop('a')\n", "TypeError"),
    ("list.pop", L + "v = xs.pop(1, 2)\n", "TypeError"),
    ("list.reverse", L + "xs.reverse()\n", ""),
    ("list.reverse", L + "xs.reverse(1)\n", "TypeError"),
    ("dict.get", D + "v = d.get(1)\n", ""),
    ("dict.get", D + "v = d.get(7, 0)\n", ""),
    ("dict.get", D + "v = d.get()\n", "TypeError"),
    ("dict.get", D + "v = d.get([1])\n", "TypeError"),
    ("dict.keys", D + "ks = list(d.keys())\n", ""),
    ("dict.keys", D + "ks = d.keys(1)\n", "TypeError"),
    ("dict.values", D + "vs = list(d.values())\n", ""),
    ("dict.values", D + "vs = d.values(1)\n", "TypeError"),
    ("io.StringIO.write", IO + 'n = f.write("a")\n', ""),
    ("io.StringIO.write", IO + "n = f.write(1)\n", "TypeError"),
    ("io.StringIO.write", IO + "n = f.write()\n", "TypeError"),
    ("io.StringIO.getvalue", IO + 'f.write("ab")\nx = f.getvalue()\n', ""),
    ("io.StringIO.getvalue", IO + "x = f.getvalue(0)\n", "TypeError"),
    ("io.StringIO.read", IO + "x = f.read()\n", ""),
    ("io.StringIO.read", IO + "x = f.read(2)\n", ""),
    ("io.StringIO.read", IO + "x = f.read('a')\n", "TypeError"),
    ("io.StringIO.seek", IO + "n = f.seek(0)\n", ""),
    ("io.StringIO.seek", IO + "n = f.seek()\n", "TypeError"),
    ("io.StringIO.seek", IO + "n = f.seek('a')\n", "TypeError"),
    ("list.__getitem__", L + "v = xs[0]\n", ""),
    ("list.__getitem__", L + "v = xs[-3]\n", ""),
    ("list.__getitem__", L + "v = xs[1:2]\n", ""),
    ("list.__getitem__", L + "v = xs[9:12]\n", ""),
    ("list.__getitem__", L + "v = xs[3]\n", "IndexError"),
    ("list.__getitem__", L + "v = xs[-4]\n", "IndexError"),
    ("list.__getitem__", L + "v = xs['a']\n", "TypeError"),
    ("list.__getitem__", L + "v = xs[1.5]\n", "TypeError"),
    ("list.__setitem__", L + "xs[0] = 9\n", ""),
    ("list.__setitem__", L + "xs[0:1] = [4, 5]\n", ""),
    ("list.__setitem__", L + "xs[5] = 9\n", "IndexError"),
    ("list.__setitem__", L + "xs['a'] = 9\n", "TypeError"),
    ("tuple.__getitem__", T + "v = t[0]\n", ""),
    ("tuple.__getitem__", T + "v = t[-3]\n", ""),
    ("tuple.__getitem__", T + "v = t[0:2]\n", ""),
    ("tuple.__getitem__", T + "v = t[3]\n", "IndexError"),
    ("tuple.__getitem__", T + "v = t['a']\n", "TypeError"),
    ("dict.__getitem__", D + "v = d[1]\n", ""),
    ("dict.__getitem__", D + "v = d[7]\n", "KeyError"),
    ("dict.__getitem__", D + "v = d[[1]]\n", "TypeError"),
    ("dict.__setitem__", D + "d[3] = 4\n", ""),
    ("dict.__setitem__", D + "d[1] = 5\n", ""),
    ("dict.__setitem__", D + "d[[1]] = 2\n", "TypeError"),
    ("list.copy", L + "ys = xs.copy()\n", ""),
    ("list.copy", L + "ys = xs.copy(1)\n", "TypeError"),
    ("list.clear", L + "xs.clear()\n", ""),
    ("list.clear", L + "xs.clear(1)\n", "TypeError"),
    ("list.count", L + "n = xs.count(1)\n", ""),
    ("list.count", L + "n = xs.count()\n", "TypeError"),
    ("list.index", L + "i = xs.index(1)\n", ""),
    ("list.index", L + "i = xs.index(1, 0, 3)\n", ""),
    ("list.index", L + "i = xs.index(1, 'a')\n", "TypeError"),
    ("list.index", L + "i = xs.index()\n", "TypeError"),
    ("list.remove", L + "xs.remove(1)\n", ""),
    ("list.remove", L + "xs.remove()\n", "TypeError"),
    ("list.sort", L + "xs.sort()\n", ""),
    ("list.sort", L + "xs.sort(reverse=True)\n", ""),
    ("list.sort", L + "xs.sort(1)\n", "TypeError"),
    ("tuple.count", T + "n = t.count(4)\n", ""),
    ("tuple.count", T + "n = t.count()\n", "TypeError"),
    ("tuple.index", T + "i = t.index(5)\n", ""),
    ("tuple.index", T + "i = t.index(5, 'a')\n", "TypeError"),
]

#: programs whose later subscript depends on what the method did to the list: (program, the class CPython raises ("" = clean)).  absint is
#: never TRUE for the raised class and TRUE for every class of a clean one -- the abstract length and elements after the call
SEMANTIC = [
    ("xs = [3, 1, 2]\nxs.clear()\nv = xs[0]\n", "IndexError"), ("xs = [3, 1, 2]\nxs.clear()\nassert len(xs) == 0\n", ""),
    ("xs = [3, 1, 2]\nxs.remove(1)\nv = xs[2]\n", "IndexError"), ("xs = [3, 1, 2]\nxs.remove(1)\nv = xs[1]\n", ""),
    ("xs = [3, 1, 2]\nys = xs.copy()\nys.append(5)\nv = xs[3]\n", "IndexError"), ("xs = [3, 1, 2]\nys = xs.copy()\nv = ys[2]\n", ""),
    ("xs = [3, 1, 2]\nys = xs.copy()\nxs.clear()\nv = ys[2]\n", ""),
    ("xs = [3, 1, 2]\nxs.sort()\nv = xs[3]\n", "IndexError"), ("xs = [3, 1, 2]\nxs.sort()\nv = xs[2]\n", ""),
    ("xs = [3, 1, 2]\nn = xs.count(1)\nv = xs[n + 3]\n", "IndexError"), ("xs = [3, 1, 2]\ni = xs.index(2)\nv = xs[i]\n", ""),
    ("xs = [3, 1, 2]\ni = xs.index(3)\nv = xs[i + 3]\n", "IndexError"),
    ("t = (4, 5, 6)\ni = t.index(6)\nv = t[i]\n", ""), ("t = (4, 5, 6)\ni = t.index(6)\nv = t[i + 1]\n", "IndexError"),
    ("t = (4, 5, 6)\nn = t.count(4)\nv = t[n + 2]\n", "IndexError"),
]
SEM_IDS = [f"sem-{i}" for i in range(len(SEMANTIC))]


@pytest.mark.parametrize("src,native", SEMANTIC, ids=SEM_IDS)
def test_the_abstract_length_and_elements_after_the_method_follow_cpython(src, native):
    assert run_native(src) == native, src
    for exc in ALL:
        verdict = prove(src, (exc,), time_budget=30.0).verdict
        if exc == native:
            assert verdict != "TRUE", (src, exc)
        elif native == "":
            assert verdict == "TRUE", (src, exc)

IDS = [f"{name}-{i}" for i, (name, _s, _n) in enumerate(CASES)]


@pytest.mark.parametrize("name,src,native", CASES, ids=IDS)
def test_the_table_is_native(name, src, native):
    assert run_native(src) == native, (name, src)


@pytest.mark.parametrize("name,src,native", CASES, ids=IDS)
def test_never_true_for_a_class_cpython_raises_and_true_for_every_class_of_a_clean_program(name, src, native):
    for exc in ALL:
        verdict = prove(src, (exc,), time_budget=30.0).verdict
        if exc == native:
            assert verdict != "TRUE", (name, src, exc)
        elif native == "":
            assert verdict == "TRUE", (name, src, exc)


def _status(src: str, name: str) -> dict:
    """``{class: "proven" | "UNPROVEN"}`` of the ``<name>:<class>`` ops recorded at the call.  A call the model refuses
    (``list(5)``: "several possible iterables") records nothing and is an UNKNOWN: ``None`` then, which the callers accept
    only where the table says the call raises natively."""
    r = prove(src, ALL, time_budget=30.0)
    if r.verdict == "UNKNOWN" and r.reason.startswith("unsupported:"):
        return None
    return {s[3]: ("UNPROVEN" if s[4].startswith("UNPROVEN") else "proven") for s in r.sites if s[2] == f"{name}:{s[3]}"}


@pytest.mark.parametrize("name,src,native", [c for c in CASES if c[2]], ids=[i for c, i in zip(CASES, IDS) if c[2]])
def test_a_native_class_is_unproven_and_the_other_six_are_proven(name, src, native):
    status = _status(src, name)
    if status is None:      # refused by name: no op was recorded, nothing is proven
        assert run_native(src) == native
        return
    assert set(status) == set(ALL), (name, src, status)
    assert status[native] == "UNPROVEN", (name, src, status)
    # a TypeError that ends the path before the call's own logic ran leaves that logic's class unargued: list.pop's IndexError
    free = {native, *NATIVE_OTHER.get(name, ())} if native == "TypeError" else {native}
    assert all(v == "proven" for k, v in status.items() if k not in free), (name, src, status)


@pytest.mark.parametrize("name,src,native", [c for c in CASES if c[2] == ""], ids=[i for c, i in zip(CASES, IDS) if c[2] == ""])
def test_a_clean_call_records_all_seven_classes_proven(name, src, native):
    status = _status(src, name)
    assert status is not None and set(status) == set(ALL), (name, src, status)
    assert set(status.values()) == {"proven"}, (name, src, status)


@pytest.mark.parametrize("name", METHODS)
def test_withdrawing_the_contract_withdraws_the_model(name, monkeypatch):
    """The contract is what admits the model: with it gone the call is a named refusal, not the hand model of old."""
    load_all()
    src = next(s for n, s, native in CASES if n == name and native == "")
    assert prove(src, ALL, time_budget=30.0).verdict == "TRUE"
    monkeypatch.delitem(REGISTRY, name)
    r = prove(src, ALL, time_budget=30.0)
    assert r.verdict == "UNKNOWN" and f"{name} has no tested contract" in r.reason, (name, r.reason)


# --------------------------------------------------------------------------- negative controls
def _lax(real):
    def modelled(self, st, node, qualname, model, args, kwargs, hooks, **kw):
        result, proven = real(self, st, node, qualname, model, args, kwargs, hooks, **kw)
        proven["TypeError"] = "lax: proven whatever the model checked"
        return result, proven
    return modelled


@pytest.mark.parametrize("name,src,native", [c for c in CASES if c[2] == "TypeError" and c[0] not in SUBSCRIPTS],
                         ids=[i for c, i in zip(CASES, IDS) if c[2] == "TypeError" and c[0] not in SUBSCRIPTS])
def test_negative_control_a_modelled_that_proves_typeerror_unconditionally_is_caught(name, src, native, monkeypatch):
    """The control: with ``_modelled`` proving ``TypeError`` whatever the model recorded, the op of a call that raises it
    natively comes out proven -- the very thing ``test_a_native_typeerror_is_unproven_...`` forbids."""
    monkeypatch.setattr(Analyzer, "_modelled", _lax(Analyzer._modelled))
    status = _status(src, name)
    if status is None:
        pytest.skip("the model refuses this call by name: there is no op to prove")
    assert status.get("TypeError") == "proven", (name, src, status)


def test_negative_control_a_proven_classes_that_proves_the_native_class_is_caught_for_the_subscripts(monkeypatch):
    """The control for the subscripts: a ``_proven_classes`` that proves every class of the contract whatever the model checked makes
    ``xs[3]`` (an ``IndexError`` on CPython) come out proven at ``list.__getitem__:IndexError``."""
    real = Analyzer._proven_classes

    def lax(contract, seen, hooks, native=None, type_free=None):
        out = real(contract, seen, hooks, native, type_free)
        out["IndexError"] = "lax"
        return out

    monkeypatch.setattr(Analyzer, "_proven_classes", staticmethod(lax))
    assert _status(L + "v = xs[3]\n", "list.__getitem__")["IndexError"] == "proven"
