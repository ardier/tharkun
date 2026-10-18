"""A dict lookup whose key is a read of a list that is provably still empty never builds a KeyError.

``table[stack[-1]]`` shows the key's text in a ``KeyError``.  When ``stack`` is a function-local
``[]`` that nothing can fill, ``stack[-1]`` raises ``IndexError("list index out of range")`` first, so
the lookup never has a key to show.  The certificate is syntactic and private
(``Provenance.inert_empty_index``); every case below is checked against what CPython actually does.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap

import pytest

from tharkun.solve.determinism import DET_MODULES, determinism_census
from tharkun.solve.provenance import Provenance
from tharkun.solve.pyresolve import Resolver
from tharkun.sv import verify


def _native(tmp_path, source: str, seed: int):
    path = tmp_path / "case.py"
    path.write_text(source, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONSTARTUP"}}
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-B", str(path)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30, check=False)


def _certified(source: str, target: str = "s[-1]"):
    """The certificate for every Load subscript spelled ``target`` in ``source``."""
    tree = ast.parse(source)
    prov = Provenance(Resolver(tree, DET_MODULES))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Subscript)
             and isinstance(n.ctx, ast.Load) and ast.unparse(n) == target]
    assert nodes, f"no {target!r} in the program"
    return [prov.inert_empty_index(n) for n in nodes]


def _src(text: str) -> str:
    return textwrap.dedent(text)


# -- what is certified ----------------------------------------------------------------

#: (source, text of the native error or None when the read is never reached)
SHUNTING = _src("""\
    def f(tokens):
        precedence = {'+': 1, '*': 2}
        out = []
        s = []
        for t in tokens:
            while s and precedence[t] <= precedence[s[-1]]:
                out.append(s.pop())
        while s:
            out.append(s.pop())
        return out
    assert f(['+', '*', '+']) == []
""")


def test_the_shunting_yard_shape_is_certified_census_clean_and_native_clean(tmp_path):
    assert _certified(SHUNTING) == [True]
    assert determinism_census(SHUNTING).ok
    for seed in (0, 1, 2, 3):
        run = _native(tmp_path, SHUNTING, seed)
        assert run.returncode == 0, run.stderr


@pytest.mark.parametrize("index", ["-1", "0", "+3", "-2", "123456789012345678901234567890"])
def test_every_literal_int_index_raises_the_same_address_free_error(tmp_path, index):
    source = _src(f"""\
        def f():
            s = []
            table = {{1: 2}}
            return table[s[{index}]]
        f()
    """)
    assert _certified(source, f"s[{index}]") == [True]
    assert determinism_census(source).ok
    texts = set()
    for seed in range(12):
        run = _native(tmp_path, source, seed)
        assert run.returncode == 1 and "IndexError" in run.stderr and "KeyError" not in run.stderr
        assert "0x" not in run.stderr
        texts.add(run.stderr.strip().splitlines()[-1])
    assert len(texts) == 1


def test_pop_truth_tests_and_not_are_the_only_other_reads(tmp_path):
    for body in ("while s:\n        s.pop()", "if not s:\n        pass", "x = not s",
                 "y = 1 if s else 2", "assert not s", "if s or s:\n        s.pop()",
                 "if (s and 1) or (1 and s):\n        s.pop()"):
        source = f"def f():\n    s = []\n    {body}\n    return {{1: 2}}[s[-1]]\nf()\n"
        assert _certified(source) == [True], body
        run = _native(tmp_path, source, 0)
        assert "IndexError" in run.stderr and "0x" not in run.stderr, (body, run.stderr)


def test_the_empty_list_display_needs_no_variable():
    assert _certified("{1: 2}[[][0]]\n", "[][0]") == [True]
    assert determinism_census("{1: 2}[[][0]]\n").ok
    assert _certified("{1: 2}[[1][0]]\n", "[1][0]") == [False]
    assert not determinism_census("{1: 2}[[object()][0]]\n").ok


def test_same_spelled_names_in_other_functions_are_independent(tmp_path):
    source = _src("""\
        def good():
            s = []
            return {1: 2}[s[-1]]
        def bad(s):
            s.append(1)
            return {1: 2}[s[-1]]
        def nested():
            def inner():
                s = []
                return {1: 2}[s[-1]]
            return inner
    """)
    assert _certified(source) == [True, False, True]


# -- what is refused: each of these makes the read return a value, natively ----------------

#: statements placed between ``s = []`` and the lookup that fill the list natively
POPULATING = {
    "append": "s.append(1)",
    "extend": "s.extend([1])",
    "insert": "s.insert(0, 1)",
    "iadd": "s += [1]",
    "slice store": "s[:] = [1]",
    "item store into a grown list": "s.append(0)\n    s[0] = 1",
    "rebinding": "s = s + [1]",
    "rebinding to a display": "s = [1]",
    "list call": "s = list([1])",
    "alias append": "t = s\n    t.append(1)",
    "bound method": "put = s.append\n    put(1)",
    "unbound method": "list.append(s, 1)",
    "call argument": "fill(s)",
    "nested def": "def g():\n        s.append(1)\n    g()",
    "comprehension": "[s.append(1) for _ in range(1)]",
    "for target": "for s in [[1]]:\n        pass",
    "keyword argument": "fill(box=s)",
    "star argument": "fill(*[s])",
    "container": "box = [s]\n    box[0].append(1)",
    "attribute": "holder.items = s\n    holder.items.append(1)",
    "return and caller": "s = grow(s)",
    "class capture": "class K:\n        s.append(1)",
}


@pytest.mark.parametrize("name", sorted(POPULATING))
def test_every_way_of_filling_the_list_is_refused_and_really_fills_it(tmp_path, name):
    source = _src("""\
        class Holder: pass
        holder = Holder()
        def fill(*boxes, box=None):
            for target in boxes + ((box,) if box is not None else ()):
                target.append(1)
        def grow(items):
            items.append(1)
            return items
        def f():
            s = []
            BODY
            return {1: 'hit'}[s[-1]]
        print(f())
    """).replace("BODY", POPULATING[name])
    assert _certified(source) == [False], name
    for seed in (0, 1):
        run = _native(tmp_path, source, seed)
        assert run.returncode == 0 and run.stdout.strip() == "hit", (name, run.stderr)


#: (source, the read under test): shapes the grammar allows that make the local something else
STRUCTURAL = {
    "parameter": ("def f(s):\n    return {1: 'hit'}[s[-1]]\nprint(f([1]))\n", "s[-1]"),
    "default parameter": ("def f(s=[1]):\n    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
    "module list": ("s = [1]\nprint({1: 'hit'}[s[-1]])\n", "s[-1]"),
    "module empty list": ("s = []\ndef g():\n    s.append(1)\ng()\nprint({1: 'hit'}[s[-1]])\n", "s[-1]"),
    "conditional init": ("import sys\ndef f(c):\n    if c:\n        s = []\n    else:\n"
                         "        s = [1]\n    return {1: 'hit'}[s[-1]]\nprint(f(len(sys.argv) > 5))\n", "s[-1]"),
    "global declaration": ("def g():\n    s.append(1)\ndef f():\n    global s\n    s = []\n    g()\n"
                           "    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
    "closure capture": ("def f():\n    s = []\n    def g():\n        s.append(1)\n    g()\n"
                        "    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
    "generator handoff": ("def gen():\n    s = []\n    yield s\n    print({1: 'hit'}[s[-1]])\n"
                          "it = gen()\nnext(it).append(1)\nlist(it)\n", "s[-1]"),
    "value use of a bool op": ("def f():\n    s = []\n    t = s or [1]\n    return {1: 'hit'}[t[-1]]\n"
                               "print(f())\n", "t[-1]"),
    "bool op hands the list on": ("def f():\n    s = []\n    t = s and 1\n    t.append(1)\n"
                                  "    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
    "bool op in a call argument": ("def f():\n    s = []\n    (s and 1).append(1)\n"
                                   "    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
    "conditional expression value": (
        "def f(c):\n    s = []\n    t = 1 if c else s\n    t.append(1)\n"
        "    return {1: 'hit'}[s[-1]]\nprint(f(0))\n", "s[-1]"),
    "assert message": (
        "def f():\n    s = []\n    try:\n        assert 0, s\n    except AssertionError as error:\n"
        "        error.args[0].append(1)\n    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
    "nonempty display": ("def f():\n    s = [1]\n    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
    "nonempty tuple": ("def f():\n    s = (1,)\n    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
    "list call": ("def f():\n    s = list((1,))\n    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
    "chained target": ("def f():\n    s = t = []\n    t.append(1)\n    return {1: 'hit'}[s[-1]]\n"
                       "print(f())\n", "s[-1]"),
    "second chained target": ("def f():\n    t = s = []\n    t.append(1)\n    return {1: 'hit'}[s[-1]]\n"
                              "print(f())\n", "s[-1]"),
    "tuple target": ("def f():\n    (s, t) = ([1], [])\n    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
    "conditional expression hands the list on": (
        "def f():\n    s = []\n    t = s if s == [] else 1\n    t.append(1)\n"
        "    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
    "annotated rebinding": ("def f():\n    s = []\n    s: list = [1]\n    return {1: 'hit'}[s[-1]]\n"
                            "print(f())\n", "s[-1]"),
    "walrus rebinding": ("def f():\n    s = []\n    (s := [1])\n    return {1: 'hit'}[s[-1]]\n"
                         "print(f())\n", "s[-1]"),
    "del and rebind": ("def f():\n    s = []\n    del s\n    s = [1]\n    return {1: 'hit'}[s[-1]]\n"
                       "print(f())\n", "s[-1]"),
    "loop that fills after the lookup": (
        "def f():\n    s = []\n    out = []\n    for i in range(2):\n        if s:\n"
        "            out.append({1: 'hit'}[s[-1]])\n        s.append(1)\n    return out\n"
        "print(f()[0])\n", "s[-1]"),
    "element store": ("def f():\n    s = []\n    s.append(0)\n    s[0] = 1\n"
                      "    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
    "with target": ("import io\ndef f():\n    s = []\n    with open(__file__) as s:\n        pass\n"
                    "    return {1: 'hit'}[s.fileno() * 0 + 1]\nprint(f())\n", "s.fileno() * 0 + 1"),
    "decorator mention": ("def deco(x):\n    return x\ndef f():\n    s = []\n    s.append(1)\n"
                          "    @deco\n    def g(a=s[-1]):\n        return a\n"
                          "    return {1: 'hit'}[s[-1]]\nprint(f())\n", "s[-1]"),
}


@pytest.mark.parametrize("name", sorted(STRUCTURAL))
def test_structural_hazards_are_never_certified(tmp_path, name):
    source, target = STRUCTURAL[name]
    if name != "with target":  # that case reads a file descriptor, only the grammar shape matters
        assert _certified(source, target) == [False] * len(_certified(source, target)), name
        assert not any(_certified(source, target)), name
        run = _native(tmp_path, source, 0)
        assert run.returncode == 0 and "hit" in run.stdout, (name, run.stderr)
    else:
        tree = ast.parse(source)
        prov = Provenance(Resolver(tree, DET_MODULES))
        assert not any(prov.inert_empty_index(n) for n in ast.walk(tree)
                       if isinstance(n, ast.Subscript))


def test_refusals_that_are_harmless_natively_are_still_refusals(tmp_path):
    # the list is never filled, so the read does raise IndexError; the certificate still declines
    # because it does not reason about which binding is live
    for source, target in (
            ("def f(s):\n    s = []\n    return {1: 'hit'}[s[-1]]\nf([1])\n", "s[-1]"),
            ("def f():\n    s = []\n    s *= 2\n    return {1: 'hit'}[s[-1]]\nf()\n", "s[-1]"),
            ("def f():\n    s = []\n    n = len(s)\n    return {1: 'hit'}[s[-1]]\nf()\n", "s[-1]"),
            ("def f():\n    s = []\n    t = s\n    return {1: 'hit'}[t[-1]]\nf()\n", "t[-1]"),
            ("def f():\n    s = []\n    return {1: 'hit'}[s[-1]], s\nf()\n", "s[-1]"),
            ("def f():\n    s = []\n    def g():\n        return s[-1]\n    return {1: 'hit'}[g()]\nf()\n",
             "s[-1]"),
            ("def f():\n    s = []\n    def g():\n        return s[-1]\n    return {1: 'hit'}[s[-1]], g\nf()\n",
             "s[-1]"),
            ("def f():\n    s = []\n    x = ~s\n    return {1: 'hit'}[s[-1]]\nf()\n", "s[-1]"),
            ("def f():\n    s = []\n    s.pop(0)\n    return {1: 'hit'}[s[-1]]\nf()\n", "s[-1]"),
            ("def f(i):\n    s = []\n    s.pop(i)\n    return {1: 'hit'}[s[-1]]\nf(0)\n", "s[-1]")):
        assert not any(_certified(source, target)), source
        assert _native(tmp_path, source, 0).returncode == 1, source


def test_an_import_binding_the_name_does_not_resolve():
    assert not determinism_census("def f():\n    s = []\n    import math as s\n    return {1: 2}[s[-1]]\nf()\n").ok


def test_use_before_init_is_certified_only_if_it_can_never_run(tmp_path):
    # the read comes BEFORE the binding in the function: an unbound local, never a list read
    source = "def f():\n    x = {1: 'hit'}\n    y = x[s[-1]]\n    s = []\nf()\n"
    assert _certified(source) == [False]
    run = _native(tmp_path, source, 0)
    assert "UnboundLocalError" in run.stderr


READS_THAT_ARE_NOT_INDEX_LITERALS = (
    "s[i]", "s[i + 1]", "s[True]", "s[1.0]", "s[None]", "s[-i]", "s[--1]", "s[1:]", "s[::-1]",
    "s[1, 2]", "s[Weird()]",
)


@pytest.mark.parametrize("read", READS_THAT_ARE_NOT_INDEX_LITERALS)
def test_only_exact_int_literals_are_certified(read):
    source = _src(f"""\
        class Weird:
            def __index__(self): return 0
        def f(i):
            s = []
            return {{1: 2}}[{read}]
        f(0)
    """)
    assert _certified(source, read) == [False]


def test_computed_receivers_and_stores_are_not_reads():
    assert _certified("def f():\n    s = []\n    s[0] = 1\n    return {1: 2}[s[0]]\nf()\n", "s[0]") == [False]
    assert _certified("def f():\n    s = []\n    return {1: 2}[(s + [1])[-1]]\nf()\n", "(s + [1])[-1]") == [False]
    assert _certified("def f():\n    s = []\n    del s[0]\n    return {1: 2}[s[0]]\nf()\n", "s[0]") == [False]
    assert _certified("def f():\n    s = []\n    s[0] += 1\n    return {1: 2}[s[0]]\nf()\n", "s[0]") == [False]


# -- the census: the exemption is exactly one conversion, and the address cases still refuse ----

ADDRESS_HAZARDS = {
    "appended object": "s.append(object())",
    "extended object": "s.extend([object()])",
    "inserted object": "s.insert(0, object())",
    "alias appended object": "t = s\n    t.append(object())",
    "stored through a call": "fill(s)",
    "slice-stored object": "s[:] = [object()]",
}


@pytest.mark.parametrize("name", sorted(ADDRESS_HAZARDS))
def test_a_key_that_can_be_an_address_still_refuses_and_really_shows_one(tmp_path, name):
    source = _src("""\
        def fill(box):
            box.append(object())
        def f():
            s = []
            BODY
            return {1: 2}[s[-1]]
        f()
    """).replace("BODY", ADDRESS_HAZARDS[name])
    result = determinism_census(source)
    assert not result.ok, name
    for seed in range(12):
        run = _native(tmp_path, source, seed)
        assert run.returncode == 1 and "KeyError: <object object at 0x" in run.stderr, (name, run.stderr)


def test_the_exemption_is_only_for_the_dict_lookup_conversion(tmp_path):
    # other conversions of the same never-computed read keep their own checks
    for statement in ("raise KeyError(s[-1])", "print(s[-1])", "str(s[-1])", "assert 0, s[-1]",
                      "'%s' % (s[-1],)", "f'{s[-1]}'", "print({1: 2}[s[-1]], {}[object()])"):
        source = f"def f():\n    s = []\n    {statement}\nf()\n"
        assert not determinism_census(source).ok, statement
        run = _native(tmp_path, source, 0)
        assert run.returncode == 1, (statement, run.stderr)


def test_effects_of_the_receiver_and_of_earlier_siblings_are_still_scanned(tmp_path):
    for statement in ("return {print(object()): 1}[s[-1]]",
                      "print(object())\n    return {1: 2}[s[-1]]",
                      "return [print(object()), {1: 2}[s[-1]]]",
                      "return {1: 2}[s[-1]], {}[object()]"):
        source = f"def f():\n    s = []\n    {statement}\nf()\n"
        assert not determinism_census(source).ok, statement
        run = _native(tmp_path, source, 0)
        assert run.returncode == 1, (statement, run.stderr)


def test_a_handler_that_could_read_the_error_keeps_the_whole_file_refused():
    source = _src("""\
        def f():
            s = []
            try:
                return {1: 2}[s[-1]]
            except IndexError as error:
                return str(error)
        f()
    """)
    assert not determinism_census(source).ok


def test_unsupported_bindings_elsewhere_in_the_file_keep_the_veto():
    # ``print(1 + 1)`` is a text sink with a non-literal operand, so the unsafe veto is consulted
    clean = "def f():\n    s = []\n    return {1: 2}[s[-1]]\nprint(1 + 1)\nf()\n"
    assert determinism_census(clean).ok
    for extra in ("f2 = lambda: 0\nf2()", "async def g(): pass\ng",
                  "class C(object, metaclass=type): pass\nC()",
                  "try:\n    f()\nexcept ValueError:\n    pass", "(w := 1)"):
        source = clean.replace("f()\n", f"{extra}\nf()\n", 1) if "try" not in extra else clean + extra + "\n"
        assert not determinism_census(source).ok, extra


def test_existing_library_models_keep_their_precedence():
    deque_then_lookup = _src("""\
        from collections import deque
        def f():
            s = []
            q = deque()
            q.append(object())
            return {1: 2}[s[-1]]
        f()
    """)
    assert not determinism_census(deque_then_lookup).ok


# -- end to end: the property is decided by the run, the certificate only licenses the single run ----

def _verdict(tmp_path, source: str, *classes: str):
    path = tmp_path / "program.py"
    path.write_text(source, encoding="utf-8")
    return verify(str(path), classes, python=sys.executable, budget_seconds=60, run_timeout=30,
                  max_runs=1, jobs=1, prove=False, concolic=False, canary_seconds=0)


def test_a_lookup_that_is_never_reached_is_true_and_one_that_is_reached_is_false(tmp_path):
    guarded = _src("""\
        def f(tokens):
            table = {'+': 1}
            s = []
            for t in tokens:
                while s and table[t] <= table[s[-1]]:
                    s.pop()
            return tokens
        assert f(['+', '+']) == ['+', '+']
    """)
    verdict = _verdict(tmp_path, guarded, "KeyError", "IndexError")
    assert verdict.verdict == "true", verdict
    reached = _src("""\
        def f(tokens):
            table = {'+': 1}
            s = []
            for t in tokens:
                if table[t] <= table[s[-1]]:
                    s.pop()
            return tokens
        f(['+'])
    """)
    verdict = _verdict(tmp_path, reached, "KeyError", "IndexError")
    assert verdict.verdict == "false" and verdict.exception == "IndexError", verdict
    # the other classes of the same program: the IndexError is not a ZeroDivisionError
    verdict = _verdict(tmp_path, reached, "ZeroDivisionError")
    assert verdict.verdict != "false", verdict
