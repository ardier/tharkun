"""The symbolic executor reads the contracts of the container operations it evaluates itself ([internal ref] step 2, PR "symexec containers").

``list.append``, ``list.__getitem__``, ``list.__setitem__``, ``dict.get``, ``dict.keys``, ``dict.__getitem__``, ``dict.__setitem__``
and ``tuple.__getitem__`` were evaluated by inline code of ``_Runner`` that never read their contracts (the audit's ``M`` cell).
They now go through their rule: ``_Exec._through_rule`` -> ``Rule.accept`` -> ``SymexecVisitor._builtin`` (a TESTED contract,
else a refusal) -> ``_Exec._model_<qualname>`` on the operands already evaluated.

What this file pins, against CPython 3.14.4:

* DIFFERENTIAL: for every program of the table the executor ends with exactly the value CPython computes for each name the program
  binds (same class, same value), or refuses, or ends the path where CPython raises (``KeyError`` / ``IndexError``);
* symbolic: a guard on what a symbolic index or key selects is solved and replays natively; a guard on a value no index selects is
  not reached;
* the contract is what admits the operation: untested or withdrawn, it is a refusal, not the model of old;
* NEGATIVE CONTROL: a model that ignores its operand (``xs[0]`` for every index) is caught by the symbolic guard.
"""

from __future__ import annotations

import ast

import pytest

from tests.contracts._native import run_native
from tests.contracts.test_engine_audit_consumed import _solve, untested
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.sv import _symexec

CONTAINER_RULES = ("list.append", "list.__getitem__", "list.__setitem__", "dict.get", "dict.keys", "dict.__getitem__",
                   "dict.__setitem__", "tuple.__getitem__")

#: programs whose names CPython and the executor must agree on (or the executor refuses / ends the path)
PROGRAMS = [
    "xs = [1, 2, 3]\nxs.append(4)\nys = xs\n",
    "xs = []\nxs.append(1)\nxs.append(2)\n",
    "xs = [1, 2, 3]\nv = xs[1]\n", "xs = [1, 2, 3]\nv = xs[-1]\n", "xs = [1, 2, 3]\nv = xs[3]\n", "xs = [1, 2, 3]\nv = xs[-4]\n",
    "xs = []\nv = xs[0]\n", "xs = [1, 2, 3]\nxs[0] = 9\n", "xs = [1, 2, 3]\nxs[-1] = 9\n", "xs = [1, 2, 3]\nxs[5] = 9\n",
    "t = (4, 5, 6)\nv = t[0]\n", "t = (4, 5, 6)\nv = t[-1]\n", "t = (4, 5, 6)\nv = t[3]\n",
    "d = {1: 2, 3: 4}\nv = d[1]\n", "d = {1: 2, 3: 4}\nv = d[7]\n", "d = {1: 2}\nd[3] = 4\n", "d = {1: 2}\nd[1] = 5\n",
    "d = {1: 2}\nv = d.get(1)\n", "d = {1: 2}\nv = d.get(7)\n", "d = {1: 2}\nv = d.get(7, 0)\n", "d = {1: 2}\nv = d.get(1, 0)\n",
    "d = {'a': 1, 'b': 2}\nks = d.keys()\n", "d = {}\nks = d.keys()\n",
    "xs = [1, 2]\nys = xs\nxs.append(3)\n",
]


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
    tree = ast.parse("def __sv_harness():\n    pass\n" + "".join(""))
    runner = _symexec._Runner(tree, {}, {}, {}, [])
    runner.run(list(ast.parse(src).body))
    return runner


def _norm(v):
    if isinstance(v, (type({}.keys()),)):
        return list(v)
    return v


#: a list reachable by two names: the executor binds the object itself and its mutations are functional, so a mutation of an aliased list is
#: REFUSED by name (``_Exec._mutable``) -- it used to be applied to one name only (``ys = xs; xs.append(3)`` left ``ys`` unchanged); pinned
#: in ``test_symexec_method_rules.py``
ALIASING = {"xs = [1, 2]\nys = xs\nxs.append(3)\n"}


@pytest.mark.parametrize("src", PROGRAMS)
def test_the_executor_agrees_with_cpython_or_refuses(src):
    ok, native = _native(src)
    try:
        runner = _run(src)
    except _symexec._Dead:
        assert not ok, ("the executor ended the path but CPython runs the program to its end", src)
        return
    if runner.skipped or runner.poisoned:
        return                                  # refused by name: says nothing false
    if not ok:
        pytest.fail(f"CPython raises {native.__name__} but the executor ran the program to its end: {src!r}")
    for name, value in native.items():
        got = runner.env.get(name, _symexec.POISON)
        if got is _symexec.POISON:
            continue
        if isinstance(got, _symexec._LazyIter):
            got = list(got)
        value = _norm(value)
        assert type(got) is type(value) and got == value, (src, name, got, value)


def test_the_table_is_not_vacuous():
    answered = sum(1 for src in PROGRAMS if _native(src)[0] and not (_run_or_none(src) is None))
    assert answered >= 15, answered


def _run_or_none(src):
    try:
        r = _run(src)
    except _symexec._Dead:
        return None
    return None if (r.skipped or r.poisoned) else r


# --------------------------------------------------------------------------- the contract admits the operation
@pytest.mark.parametrize("qualname", CONTAINER_RULES)
def test_an_untested_or_withdrawn_contract_is_a_refusal_not_the_model_of_old(qualname, monkeypatch):
    src = {"list.append": "xs = [1]\nxs.append(2)\n", "list.__getitem__": "xs = [1, 2]\nv = xs[1]\n",
           "list.__setitem__": "xs = [1, 2]\nxs[0] = 9\n", "dict.get": "d = {1: 2}\nv = d.get(1)\n",
           "dict.keys": "d = {1: 2}\nks = list(d.keys())\n", "dict.__getitem__": "d = {1: 2}\nv = d[1]\n",
           "dict.__setitem__": "d = {1: 2}\nd[3] = 4\n", "tuple.__getitem__": "t = (1, 2)\nv = t[1]\n"}[qualname]
    assert _run_or_none(src) is not None, "the call is answered while the contract is tested"
    load_all()
    monkeypatch.setitem(REGISTRY, qualname, untested(REGISTRY[qualname]))
    assert _run_or_none(src) is None, (qualname, "untested: refused")
    monkeypatch.undo()
    load_all()
    monkeypatch.delitem(REGISTRY, qualname)
    assert _run_or_none(src) is None, (qualname, "withdrawn: refused")


# --------------------------------------------------------------------------- symbolic: solved, and replayed on CPython
_DRAW = "import _sv_verifier\nv = _sv_verifier.nondet_int()\n"
SYMBOLIC = [
    ("list.__getitem__", "xs = [10, 20, 30]\nx = xs[v]\n", "x == 20"),
    ("tuple.__getitem__", "t = (10, 20, 30)\nx = t[v]\n", "x == 30"),
    ("dict.__getitem__", "d = {1: 10, 2: 20}\nx = d[v]\n", "x == 20"),
    ("dict.get", "d = {1: 10, 2: 20}\nx = d.get(v, 0)\n", "x == 10"),
    ("dict.get", "d = {1: 10, 2: 20}\nx = d.get(v, 0)\n", "x == 0"),
    ("list.append", "xs = [1, 2]\nxs.append(v)\nx = xs[2]\n", "x == 7"),
    ("list.__setitem__", "xs = [1, 2, 3]\nxs[1] = v\nx = xs[1]\n", "x == 8"),
]


@pytest.mark.parametrize("name,src,guard", SYMBOLIC, ids=[f"{n}-{i}" for i, (n, _s, _g) in enumerate(SYMBOLIC)])
def test_a_guard_on_a_symbolic_operation_is_solved_and_replays(name, src, guard):
    got, h = _solve(_DRAW + src, guard)
    sat = [m for o, m in got.values() if o == "sat"]
    assert sat and sat[0]["n"] == 3, (name, got)
    vector = [sat[0][p] for p in h.params]
    full = (_DRAW + src + "import _sv_verifier\nn = _sv_verifier.nondet_int()\n"
            f"if {guard} and n == 3:\n    raise IndexError('hit')\n")
    assert run_native(full, vector) == "IndexError", (name, vector)


@pytest.mark.parametrize("name,src,guard", [
    ("list.__getitem__", "xs = [10, 20, 30]\nx = xs[v]\n", "x == 99"),
    ("dict.__getitem__", "d = {1: 10, 2: 20}\nx = d[v]\n", "x == 99"),
    ("dict.get", "d = {1: 10, 2: 20}\nx = d.get(v, 0)\n", "x == 99"),
], ids=["list", "dict", "dict.get"])
def test_a_guard_on_a_value_no_index_selects_is_not_reached(name, src, guard):
    got, _h = _solve(_DRAW + src, guard)
    assert not [m for o, m in got.values() if o == "sat"], (name, got)


def test_negative_control_a_model_that_ignores_the_index_is_caught(monkeypatch):
    """The control: a ``list.__getitem__`` that answers ``c[0]`` whatever the index makes ``xs[v] == 20`` unsolvable (the real model
    solves it with ``v == 1``): the symbolic test above fails for it."""
    monkeypatch.setattr(_symexec._Runner, "_model_list___getitem__", lambda self, n, c, k: c[0])
    got, _h = _solve(_DRAW + "xs = [10, 20, 30]\nx = xs[v]\n", "x == 20")
    assert not [m for o, m in got.values() if o == "sat"], got


def _store_executor(kind, env):
    if kind == "base":
        return _symexec._Exec({}, env, {})
    return _symexec._Runner(ast.parse(""), {}, env, {}, [])


@pytest.mark.parametrize("kind", ["base", "runner"])
@pytest.mark.parametrize("index_value", [-3, -2, -1, 0, 1, 2])
def test_symbolic_list_store_matches_native_negative_and_positive_indices(kind, index_value):
    z3 = _symexec.z3
    index = z3.Int("store_index")
    runner = _store_executor(kind, {"xs": [10, 20, 30], "index": index})
    target = ast.parse("xs[index] = 9").body[0].targets[0]
    runner.assign(target, 9)
    solver = z3.Solver()
    solver.add(*runner.pc, index == index_value)
    assert solver.check() == z3.sat
    model = solver.model()
    actual = [model.eval(x).as_long() if z3.is_expr(x) else x for x in runner.env["xs"]]
    ok, native = _native(f"xs = [10, 20, 30]\nxs[{index_value}] = 9\n")
    assert ok and actual == native["xs"]


@pytest.mark.parametrize("kind", ["base", "runner"])
@pytest.mark.parametrize("values,index_value", [([10, 20, 30], -4), ([10, 20, 30], 3), ([], 0)])
def test_symbolic_list_store_rejects_native_out_of_range_paths(kind, values, index_value):
    z3 = _symexec.z3
    index = z3.Int("store_index")
    runner = _store_executor(kind, {"xs": values, "index": index})
    runner.assign(ast.parse("xs[index] = 9").body[0].targets[0], 9)
    solver = z3.Solver()
    solver.add(*runner.pc, index == index_value)
    assert solver.check() == z3.unsat
    ok, native = _native(f"xs = {values!r}\nxs[{index_value}] = 9\n")
    assert not ok and native is IndexError


@pytest.mark.parametrize("kind", ["base", "runner"])
def test_symbolic_list_store_keeps_alias_refusal(kind):
    values = [10, 20, 30]
    runner = _store_executor(kind, {"xs": values, "alias": values, "index": _symexec.z3.Int("store_index")})
    with pytest.raises(_symexec.Unsupported, match="aliasing"):
        runner.assign(ast.parse("xs[index] = 9").body[0].targets[0], 9)


@pytest.mark.parametrize("kind", ["base", "runner"])
@pytest.mark.parametrize("status", ["untested", "withdrawn"])
def test_symbolic_list_store_keeps_tested_contract_gate(kind, status, monkeypatch):
    load_all()
    if status == "untested":
        monkeypatch.setitem(REGISTRY, "list.__setitem__", untested(REGISTRY["list.__setitem__"]))
    else:
        monkeypatch.delitem(REGISTRY, "list.__setitem__")
    runner = _store_executor(kind, {"xs": [10, 20, 30], "index": _symexec.z3.Int("store_index")})
    with pytest.raises(_symexec.Unsupported, match="no tested contract"):
        runner.assign(ast.parse("xs[index] = 9").body[0].targets[0], 9)


@pytest.mark.parametrize("kind", ["base", "runner"])
def test_list_store_refuses_index_subclass_without_comparison_hooks(kind):
    calls = []
    class Noisy(int):
        def __lt__(self, other):
            calls.append("compare")
            return super().__lt__(other)
    runner = _store_executor(kind, {"xs": [10, 20, 30], "index": Noisy(1)})
    with pytest.raises(_symexec.Unsupported, match="not an int"):
        runner.assign(ast.parse("xs[index] = 9").body[0].targets[0], 9)
    assert calls == []


@pytest.mark.parametrize("kind", ["base", "runner"])
def test_list_store_refuses_symbolic_real_index(kind):
    runner = _store_executor(kind, {"xs": [10, 20, 30], "index": _symexec.z3.Real("real_index")})
    with pytest.raises(_symexec.Unsupported, match="index that is not an int"):
        runner.assign(ast.parse("xs[index] = 9").body[0].targets[0], 9)


@pytest.mark.parametrize("values", [[10, 20, 30], (10, 20, 30)])
def test_sequence_read_refuses_symbolic_real_index(values):
    runner = _store_executor("runner", {"xs": values, "index": _symexec.z3.Real("real_index")})
    with pytest.raises(_symexec.Unsupported, match="index that is not an int"):
        runner.ev(ast.parse("xs[index]", mode="eval").body)


@pytest.mark.parametrize("kind", ["base", "runner"])
@pytest.mark.parametrize("index_value", [False, True])
def test_symbolic_bool_store_keeps_native_index(kind, index_value):
    z3 = _symexec.z3
    index = z3.Bool("bool_index")
    runner = _store_executor(kind, {"xs": [10, 20, 30], "index": index})
    runner.assign(ast.parse("xs[index] = 9").body[0].targets[0], 9)
    solver = z3.Solver()
    solver.add(*runner.pc, index == index_value)
    assert solver.check() == z3.sat
    model = solver.model()
    actual = [model.eval(x).as_long() if z3.is_expr(x) else x for x in runner.env["xs"]]
    ok, native = _native(f"xs = [10, 20, 30]\nxs[{index_value!r}] = 9\n")
    assert ok and actual == native["xs"]


@pytest.mark.parametrize("source", [
    "xs = [1, 2]\nxs[0] += 5\n",
    "xs = [1, 2]\ndef pick():\n    return 0\nxs[pick()] += 5\n",
    "xs = [1, 2]\ndef pick():\n    xs.append(3)\n    return 0\nxs[pick()] += 5\n",
    "xs = [[1, 2]]\nxs[0][0] += 5\n",
])
def test_augmented_list_store_matches_native_or_refuses(source):
    ok, native = _native(source)
    assert ok
    runner = _run_or_none(source)
    if runner is not None:
        assert runner.env["xs"] == native["xs"]
