"""``determinism_census(class_only_exceptions=True)``: exceptions judged by class, only while nothing observes one.

The default census refuses a dict lookup whose key could show an address in its ``KeyError`` text and a heap
ordered by ``(number, instance)`` tuples.  The opt-in mode admits exactly those two, and only in a program
in which no code can see an exception in flight, so the text can only reach the stderr of the process that
dies of it and no verdict reads it.  Every adversary below is checked against what CPython does natively,
under many hash seeds with ASLR on.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from tharkun.solve.determinism import determinism_census, observer_reason
from tharkun.sv import verify

SEEDS = range(1, 25)


def _native(tmp_path, source: str, seed: int):
    path = tmp_path / "case.py"
    path.write_text(source, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONSTARTUP"}}
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-B", str(path)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=30, check=False)


def _outcomes(tmp_path, source: str) -> dict:
    """{last stderr line with addresses masked | stdout: how many seeds}"""
    seen: dict = {}
    for seed in SEEDS:
        run = _native(tmp_path, source, seed)
        tail = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else ""
        tail = tail.split(" at 0x")[0]
        key = (run.returncode, tail, run.stdout.strip())
        seen[key] = seen.get(key, 0) + 1
    return seen


def _verdict(tmp_path, source: str, *classes: str):
    path = tmp_path / "program.py"
    path.write_text(source, encoding="utf-8")
    return verify(str(path), classes, python=sys.executable, budget_seconds=60, run_timeout=30,
                  max_runs=1, jobs=1, prove=False, concolic=False, canary_seconds=0)


def _tree(source: str):
    return SimpleNamespace(tree=ast.parse(source))


def strict(source: str) -> bool:
    return determinism_census(source).ok


def relaxed(source: str) -> bool:
    return determinism_census(source, class_only_exceptions=True).ok


UNCAUGHT_KEYERROR = textwrap.dedent("""\
    class Node:
        pass


    table = {}
    table[Node()]
""")


# -- the positive cases ---------------------------------------------------------------------

def test_an_uncaught_keyerror_on_an_object_key_is_class_only_and_exits_keyerror_on_every_run(tmp_path):
    assert not strict(UNCAUGHT_KEYERROR)
    assert relaxed(UNCAUGHT_KEYERROR)
    outcomes = _outcomes(tmp_path, UNCAUGHT_KEYERROR)
    assert {code for code, _, _ in outcomes} == {1}
    assert {tail for _, tail, _ in outcomes} == {"KeyError: <__main__.Node object"}  # address masked above
    # the verdict is the class: the program raises KeyError and nothing else, on every run
    assert _verdict(tmp_path, UNCAUGHT_KEYERROR, "AssertionError").verdict == "true"
    assert _verdict(tmp_path, UNCAUGHT_KEYERROR, "TypeError", "AttributeError").verdict == "true"
    found = _verdict(tmp_path, UNCAUGHT_KEYERROR, "KeyError", "IndexError")
    assert found.verdict == "false" and found.exception == "KeyError"


DIJKSTRA = textwrap.dedent("""\
    from heapq import heappush, heappop


    class Vertex:
        def __init__(self, name, out=()):
            self.name = name
            self.out = list(out)


    def shortest(costs, start, goal):
        frontier = []
        heappush(frontier, (0, start))
        seen = set()
        while len(frontier) > 0:
            dist, vertex = heappop(frontier)
            if vertex is goal:
                return dist
            seen.add(vertex)
            for nxt in vertex.out:
                if nxt in seen:
                    continue
                heappush(frontier, (dist + costs[vertex, nxt], nxt))
        return float('inf')


    c = Vertex('c')
    b = Vertex('b', [c])
    a = Vertex('a', [b, c])
    costs = {(a, b): 1, (a, c): 5, (b, c): 2}
    assert shortest(costs, a, c) == 3
    assert shortest(costs, c, a) == float('inf')
""")


def test_a_dijkstra_over_instances_is_class_only_and_clean_natively(tmp_path):
    assert not strict(DIJKSTRA)
    assert relaxed(DIJKSTRA)
    assert {code for code, _, _ in _outcomes(tmp_path, DIJKSTRA)} == {0}
    verdict = _verdict(tmp_path, DIJKSTRA, "KeyError", "IndexError")
    assert verdict.verdict == "true", verdict
    assert "exceptions judged by class" in verdict.reason


def test_the_default_verdict_reason_is_unchanged_for_a_strictly_certified_program(tmp_path):
    verdict = _verdict(tmp_path, "assert 1 + 1 == 2\n", "AssertionError")
    assert verdict.verdict == "true"
    assert verdict.reason.startswith("deterministic: one execution (certified: every import")
    assert "exceptions judged by class" not in verdict.reason


def test_a_tie_between_instances_raises_typeerror_on_every_run_and_is_a_class_decision(tmp_path):
    source = textwrap.dedent("""\
        import heapq


        class Node:
            pass


        heap = []
        heapq.heappush(heap, (1, Node()))
        heapq.heappush(heap, (1, Node()))
    """)
    assert not strict(source)
    assert relaxed(source)
    outcomes = _outcomes(tmp_path, source)
    assert {tail for _, tail, _ in outcomes} == {
        "TypeError: '<' not supported between instances of 'Node' and 'Node'"}
    assert _verdict(tmp_path, source, "AssertionError").verdict == "true"
    found = _verdict(tmp_path, source, "TypeError", "AttributeError")
    assert found.verdict == "false" and found.exception == "TypeError"


# -- the five adversaries the exemption must never certify -------------------------------------

ADDRESS = 'bin(int(str({}).split("0x")[1].rstrip(">"), 16) >> 24).count("1") % 2'

ADVERSARIES = {
    "exit_reads_text_and_suppresses": (textwrap.dedent("""\
        class Node:
            pass


        class Guard:
            def __enter__(self):
                return self

            def __exit__(self, kind, value, trace):
                if %s:
                    assert False, "address digit is odd"
                return True


        table = {}
        with Guard():
            table[Node()]
    """) % ADDRESS.format("value"), "AssertionError"),
    "finally_reads_exc_info": (textwrap.dedent("""\
        import sys


        class Node:
            pass


        table = {}
        try:
            try:
                table[Node()]
            finally:
                seen = sys.exc_info()[1]
                if %s:
                    assert False, "address digit is odd"
        except KeyError:
            pass
    """) % ADDRESS.format("seen"), "AssertionError"),
    "del_reads_the_exception_during_unwinding": (textwrap.dedent("""\
        import sys


        class Node:
            pass


        odd = []


        class Reader:
            def __del__(self):
                seen = sys.exc_info()[1]
                if seen is not None and %s:
                    odd.append(1)


        def run():
            table = {}
            reader = Reader()
            try:
                table[Node()]
            finally:
                del reader


        try:
            run()
        except KeyError:
            pass
        assert not odd, "address digit is odd"
    """) % ADDRESS.format("seen"), "AssertionError"),
    "excepthook_prints_the_text": (textwrap.dedent("""\
        import sys


        class Node:
            pass


        def hook(kind, value, trace):
            print("digit", %s)


        sys.excepthook = hook
        table = {}
        table[Node()]
    """) % ADDRESS.format("value"), "KeyError"),
    "keyerror_chained_as_context_of_a_later_exception": (textwrap.dedent("""\
        class Node:
            pass


        def inner():
            table = {}
            try:
                table[Node()]
            finally:
                raise ValueError("replaced")


        try:
            inner()
        except ValueError as error:
            if %s:
                assert False, "address digit is odd"
    """) % ADDRESS.format("error.__context__"), "AssertionError"),
}


@pytest.mark.parametrize("name", sorted(ADVERSARIES))
def test_an_adversary_that_reads_the_exception_really_varies_and_is_never_certified(tmp_path, name):
    source, varying = ADVERSARIES[name]
    # native, ASLR on, 24 hash seeds: the observable behaviour depends on the address
    outcomes = _outcomes(tmp_path, source)
    assert len(outcomes) >= 2, (name, outcomes)
    if varying == "AssertionError":
        assert any("AssertionError" in tail for _, tail, _ in outcomes)
        assert any(code == 0 for code, _, _ in outcomes)
    # the census refuses in BOTH modes; the observer scan is independently sufficient
    assert not strict(source)
    assert not relaxed(source)
    assert observer_reason([_tree(source)]) is not None, name
    verdict = _verdict(tmp_path, source, varying)
    assert verdict.verdict != "true", (name, verdict)


@pytest.mark.parametrize("source", [
    "class Node:\n    pass\n\ndef f():\n    pass\ntry:\n    f()\nfinally:\n    pass\n",
    "class Node:\n    pass\ntry:\n    pass\nexcept KeyError:\n    pass\n",
    "class Q:\n    pass\nwith Q() as q:\n    pass\n",
    "async def f():\n    pass\n",
    "def g():\n    yield 1\n",
    "def g():\n    yield from [1]\n",
    "xs = list(i for i in range(3))\n",
    "async def f():\n    await g()\n",
    "class D:\n    def __del__(self):\n        pass\n",
    "class C:\n    def __enter__(self):\n        return self\n",
    "class C:\n    def __exit__(self, a, b, c):\n        return False\n",
    "import sys\nsys.stderr.write('x')\n",
    "import sys\nsys.excepthook = None\n",
    "import sys\nsys.exc_info()\n",
    "import sys\nsys.exception()\n",
    "import sys\nsys.settrace(None)\n",
    "from sys import exc_info\n",
    "import traceback\n",
    "import logging\n",
    "import atexit\n",
    "import warnings\n",
    "import threading\n",
    "import contextlib\n",
    "from contextlib import contextmanager\n",
    "import signal\n",
    "e = ValueError()\ne.__traceback__\n",
    "e = ValueError()\ne.with_traceback(None)\n",
    "e = ValueError()\ne.add_note('x')\n",
    "e = ValueError()\ne.__context__\n",
    "async def f():\n    async for x in y:\n        pass\n",
])
def test_every_observer_is_found_in_the_raw_tree(source):
    assert observer_reason([_tree(source)]) is not None


@pytest.mark.parametrize("source", [
    "x = 1\n",
    UNCAUGHT_KEYERROR,
    DIJKSTRA,
    "import sys\nsys.setrecursionlimit(5000)\nprint(sys.maxsize)\n",
    "import math\nprint(math.sqrt(4))\n",
    "def f(x):\n    return x + 1\nf(1)\n",
    "class A:\n    def __init__(self):\n        self.x = 1\n    def m(self):\n        return self.x\nA().m()\n",
    "lam = lambda v: v + 1\nlam(1)\n",
])
def test_observer_free_programs_have_no_observer_reason(source):
    assert observer_reason([_tree(source)]) is None


@pytest.mark.parametrize("extra", [
    "try:\n    pass\nfinally:\n    pass\n",
    "class Q:\n    def __enter__(self): return self\n    def __exit__(self, *a): return False\nwith Q():\n    pass\n",
    "def g():\n    yield 1\n",
    "xs = list(i for i in range(3))\n",
    "class D:\n    def __del__(self): pass\n",
    "import sys\nsys.stderr.write('x')\n",
    "def never_called():\n    try:\n        pass\n    except KeyError:\n        pass\n",
])
def test_any_observer_anywhere_returns_the_census_to_strict(extra):
    assert relaxed(UNCAUGHT_KEYERROR)
    assert not relaxed(UNCAUGHT_KEYERROR + extra)


def test_an_observer_in_another_file_of_the_program_counts(tmp_path):
    main = tmp_path / "main.py"
    for name, helper_source, expected in (
            ("quiet_helper", "def noop():\n    return 1\n", True),
            ("try_helper", "def noop():\n    try:\n        return 1\n    finally:\n        pass\n", False),
            ("with_helper", "class Q:\n    def __enter__(self): return self\n    def __exit__(self, *a): return False\n"
                            "def noop():\n    with Q():\n        return 1\n", False),
            ("dead_helper", "def never():\n    try:\n        pass\n    finally:\n        pass\n"
                            "def noop():\n    return 1\n", False)):
        helper = tmp_path / f"{name}.py"
        helper.write_text(helper_source, encoding="utf-8")
        source = f"import {name}\n" + UNCAUGHT_KEYERROR.replace(
            "table = {}", f"{name}.noop()\ntable = {{}}")
        main.write_text(source, encoding="utf-8")
        result = determinism_census(source, str(main), declared_files=(str(main), str(helper)),
                                    class_only_exceptions=True)
        assert result.ok is expected, (name, result)
        assert not determinism_census(source, str(main), declared_files=(str(main), str(helper))).ok


# -- the key must hash and compare without running program code (the class stays KeyError) ------

@pytest.mark.parametrize(("name", "source"), [
    ("unknown object", "table = {}\ntable[object()]\n"),
    ("list", "class Node: pass\ntable = {}\ntable[[Node()]]\n"),
    ("tuple holding a list", "class Node: pass\ntable = {}\ntable[(Node(), [1])]\n"),
    ("slice", "table = {}\ntable[1:2]\n"),
    ("hash raises", "class Bad:\n    def __hash__(self):\n        raise TypeError('unhashable')\n"
                    "table = {}\ntable[Bad()]\n"),
    ("eq defined", "class Eq:\n    def __eq__(self, other):\n        return True\n    def __hash__(self):\n"
                   "        return 1\ntable = {}\ntable[Eq()]\n"),
    ("eq defined on the base", "class Base:\n    def __eq__(self, other):\n        return True\n"
                               "    def __hash__(self):\n        return 1\nclass Child(Base):\n    pass\n"
                               "table = {}\ntable[Child()]\n"),
    ("eq defined as a class attribute", "def same(self, other):\n    return True\nclass Eq:\n"
                                        "    __eq__ = same\n    __hash__ = None\ntable = {}\ntable[Eq()]\n"),
    ("slots on the base", "class Base:\n    __slots__ = ('x',)\nclass Child(Base):\n    pass\n"
                          "table = {}\ntable[Child()]\n"),
    ("stored key with a raising eq", "class Eq:\n    def __eq__(self, other):\n        raise AssertionError('eq')\n"
                                     "    def __hash__(self):\n        return 1\nclass Node:\n    pass\n"
                                     "table = {Eq(): 1}\ntable[Node()]\n"),
    ("a stored key added later with a raising eq",
     "class Eq:\n    def __eq__(self, other):\n        raise AssertionError('eq')\n    def __hash__(self):\n"
     "        return 1\nclass Node:\n    pass\ntable = {}\ntable[Eq()] = 1\ntable[Node()]\n"),
    ("slots", "class Node:\n    __slots__ = ('x',)\ntable = {}\ntable[Node()]\n"),
    ("receiver mixes a display and an unknown", "class Node: pass\nimport sys\n"
                                                 "table = {} if sys.maxsize else dict()\ntable[Node()]\n"),
])
def test_a_key_that_may_run_program_code_or_raise_another_class_is_never_exempted(tmp_path, name, source):
    assert not relaxed(source), name


#: programs the STRICT census already accepts, because an unsupported binding (class keywords or decorators, a
#: protocol hook, a decorated def) or a receiver that is not a tracked dict display makes the provenance drop the
#: lookup's text conversion altogether.  The class-only mode changes nothing for them, in either direction.
ALREADY_UNCHECKED = [
    ("metaclass", "class Meta(type):\n    pass\nclass Node(metaclass=Meta):\n    pass\ntable = {}\ntable[Node()]\n"),
    ("decorated class", "def deco(cls):\n    return cls\n@deco\nclass Node:\n    pass\ntable = {}\ntable[Node()]\n"),
    ("getattr hook", "class Node:\n    def __getattr__(self, name):\n        return 1\ntable = {}\ntable[Node()]\n"),
    ("property on the class", "class Node:\n    @property\n    def x(self):\n        return 1\ntable = {}\ntable[Node()]\n"),
    ("receiver is not a display", "class Node: pass\nkeys = [1]\ntable = dict.fromkeys(keys)\ntable[Node()]\n"),
]


@pytest.mark.parametrize(("name", "source"), ALREADY_UNCHECKED)
def test_programs_whose_lookup_text_the_strict_census_never_examines_are_unchanged(name, source):
    assert strict(source) is relaxed(source) is True, name


def test_keys_of_every_plain_shape_are_exempted_and_really_raise_keyerror(tmp_path):
    for source in (
            "class Node:\n    pass\na = Node()\ntable = {(a, a): 1}\ntable[(Node(), a)]\n",
            "class Node:\n    pass\ntable = {(1, 'x', None, b'y', True): 1}\ntable[(Node(), 2)]\n",
            "class Base:\n    pass\nclass Child(Base):\n    pass\ntable = {}\ntable[Child()]\n",
            "class Node:\n    def __init__(self, v):\n        self.v = v\ntable = {}\ntable[Node(1)]\n",
            "class Node:\n    def __init__(self, v):\n        self.v = v\n    def f(self):\n        return self.v\n"
            "table = {}\ntable[(Node(1), (Node(2), 3))]\n"):
        assert not strict(source), source
        assert relaxed(source), source
        outcomes = _outcomes(tmp_path, source)
        assert {code for code, _, _ in outcomes} == {1}
        assert all(tail.startswith("KeyError: ") and "object" in tail for _, tail, _ in outcomes)


# -- the heap: ordered by number, and by identity for the instance slot ----------------------

HEAP_HEAD = "import heapq\n\n\nclass Node:\n    pass\n\n\na, b = Node(), Node()\nheap = []\n"


@pytest.mark.parametrize(("name", "tail", "expected"), [
    ("distinct numbers", "heapq.heappush(heap, (1, a))\nheapq.heappush(heap, (2, b))\nheapq.heappop(heap)\n", True),
    ("int and float numbers", "heapq.heappush(heap, (1, a))\nheapq.heappush(heap, (float('inf'), b))\n", True),
    ("a direct write of the same shape", "heapq.heappush(heap, (1, a))\nheap[0] = (2, b)\n", True),
    ("bare instances", "heapq.heappush(heap, a)\nheapq.heappush(heap, b)\n", True),
    ("a tie", "heapq.heappush(heap, (1, a))\nheapq.heappush(heap, (1, b))\n", True),
    ("a different tuple length", "heapq.heappush(heap, (1, a))\nheapq.heappush(heap, (1, a, 2))\n", False),
    ("the slots swapped", "heapq.heappush(heap, (1, a))\nheapq.heappush(heap, (a, 1))\n", False),
    ("a string beside a number", "heapq.heappush(heap, (1, a))\nheapq.heappush(heap, ('x', a))\n", False),
    ("None in the number slot", "heapq.heappush(heap, (None, a))\n", False),
    ("a direct write of another shape", "heapq.heappush(heap, (1, a))\nheap[0] = ('s', b)\n", False),
    ("a complex number", "heapq.heappush(heap, (1j, a))\n", False),
    ("nsmallest over such a heap", "heapq.heappush(heap, (1, a))\nheapq.nsmallest(1, heap)\n", True),
])
def test_heaps_of_numbers_and_identity_only_instances(name, tail, expected):
    source = HEAP_HEAD + tail
    assert not strict(source), name
    assert relaxed(source) is expected, name


@pytest.mark.parametrize(("name", "classes"), [
    ("an ordering method", "class Node:\n    def __lt__(self, other):\n        return True\n"),
    ("an equality method", "class Node:\n    def __eq__(self, other):\n        return True\n"),
    ("a hash method", "class Node:\n    def __hash__(self):\n        return 1\n"),
    ("a repr method", "class Node:\n    def __repr__(self):\n        return 'n'\n"),
    ("a bool method", "class Node:\n    def __bool__(self):\n        return False\n"),
    ("an ordering method on the base", "class Base:\n    def __lt__(self, other):\n        return True\n"
                                       "class Node(Base):\n    pass\n"),
    ("an ordering alias", "def less(self, other):\n    return True\nclass Node:\n    __lt__ = less\n"),
    ("a reflected ordering method", "class Node:\n    def __gt__(self, other):\n        return True\n"),
    ("slots", "class Node:\n    __slots__ = ('x',)\n"),
    ("a class decorator", "def deco(cls):\n    return cls\n@deco\nclass Node:\n    pass\n"),
    ("a metaclass", "class Meta(type):\n    pass\nclass Node(metaclass=Meta):\n    pass\n"),
    ("a descriptor attribute", "class Node:\n    @property\n    def x(self):\n        return 1\n"),
])
def test_an_instance_class_with_any_protocol_hook_keeps_the_heap_refused(name, classes):
    source = ("import heapq\n" + classes + "heap = []\nheapq.heappush(heap, (1, Node()))\n"
              "heapq.heappush(heap, (1, Node()))\n")
    assert not relaxed(source), name
    assert not strict(source), name


def test_a_heap_ordered_by_a_user_method_really_depends_on_the_address(tmp_path):
    # the refused shape, natively: ordering by an address-derived key changes the popped item between runs
    source = textwrap.dedent("""\
        import heapq


        class Node:
            def __lt__(self, other):
                return bin(int(object.__repr__(self).split("0x")[1].rstrip(">"), 16) >> 24).count("1") % 2 == 0


        first, second = Node(), Node()
        first.name, second.name = "first", "second"
        heap = [first, second]
        heapq.heapify(heap)
        assert heap[0].name == "first"
    """)
    outcomes = _outcomes(tmp_path, source)
    assert len(outcomes) >= 2
    assert not relaxed(source)
    assert _verdict(tmp_path, source, "AssertionError").verdict != "true"


def test_a_set_takes_only_plain_members_and_an_enumerate_only_a_builtin_container():
    heap_with_set = HEAP_HEAD + "seen = set()\nseen.add(a)\nheapq.heappush(heap, (1, b))\n"
    assert relaxed(heap_with_set)
    hooked = ("import heapq\nclass Node:\n    def __hash__(self):\n        return 1\n"
              "seen = set()\nseen.add(Node())\nheap = []\nheapq.heappush(heap, (1, Node()))\n")
    assert not relaxed(hooked)
    escaping = textwrap.dedent("""\
        class Hooked:
            def __init__(self):
                self.v = 'x'

            def __hash__(self):
                return 1

            def __eq__(self, other):
                return True


        seen = set()
        item = Hooked()
        seen.add(item)
        print(item.v)
    """)
    plain = escaping.replace("    def __hash__(self):\n        return 1\n\n    def __eq__(self, other):\n"
                             "        return True\n\n", "")
    assert not strict(escaping) and not relaxed(escaping)   # a hooked member escapes: its fields are unknown again
    assert not strict(plain) and relaxed(plain)             # a plain member does not
    over_instance = HEAP_HEAD + "for i, item in enumerate(a):\n    heapq.heappush(heap, item)\n"
    assert not relaxed(over_instance)


def test_a_function_that_can_fall_off_its_end_still_returns_none():
    falls = HEAP_HEAD + ("def best(h):\n    for d, n in h:\n        return d\n"
                         "heapq.heappush(heap, (1, a))\nheapq.heappush(heap, (best(heap), b))\n")
    returns = falls.replace("        return d\n", "        return d\n    return 0\n")
    assert not relaxed(falls)
    assert relaxed(returns)


def test_a_function_that_falls_off_its_end_returns_none_natively():
    def best(h):
        for d, n in h:
            return d
    assert best([]) is None


# -- scope: only the deterministic-TRUE route asks, and the default census does not move -------------

PINNED_STRICT = [
    "values = {}; value = values[object()]",
    "class Node: pass\nh = [(1, Node()), (1, Node())]\nimport heapq\nheapq.heapify(h)\n",
    "import heapq\nclass Node: pass\nh = [(1, Node()), (1, Node())]\nheapq.heapify(h)\n",
]


@pytest.mark.parametrize("source", PINNED_STRICT)
def test_the_default_census_still_refuses_what_it_always_refused(source):
    assert not strict(source)


def test_the_flag_is_keyword_only_and_defaults_to_strict():
    import inspect

    parameter = inspect.signature(determinism_census).parameters["class_only_exceptions"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is False
    engine = inspect.signature(__import__("tharkun.sv.engine", fromlist=["x"])._certify_determinism)
    assert engine.parameters["class_only_exceptions"].default is False


def test_only_the_deterministic_true_route_of_the_scheduler_asks_for_it():
    root = Path(__file__).resolve().parents[1] / "src" / "tharkun"
    hits = {}
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        count = text.count("class_only_exceptions=True")
        if count:
            hits[str(path.relative_to(root))] = count
    assert hits == {"sv/scheduler.py": 1}, hits
    scheduler = (root / "sv" / "scheduler.py").read_text(encoding="utf-8")
    at = scheduler.index("class_only_exceptions=True")
    start = scheduler.index("Two hash seeds are samples")
    route = scheduler[start:scheduler.index("return done(_engine.Verdict(\"unknown\", reason=f\"one execution observed", start)]
    assert "class_only_exceptions=True" in route and start < at
    found = scheduler[scheduler.index("def found("):scheduler.index("def replay_solved(")]
    assert "class_only_exceptions" not in found


def test_a_false_witness_is_confirmed_with_the_strict_census_only(tmp_path):
    # the KeyError is a violation of the data-lookup property; its confirmation never relies on the flag
    verdict = _verdict(tmp_path, UNCAUGHT_KEYERROR, "KeyError")
    assert verdict.verdict == "false" and verdict.exception == "KeyError"
    assert verdict.witness is not None


def test_observer_construct_in_the_program_keeps_every_other_verdict_as_before(tmp_path):
    source = UNCAUGHT_KEYERROR + "try:\n    pass\nfinally:\n    pass\n"
    assert not relaxed(source)
    assert _verdict(tmp_path, source, "AssertionError").verdict != "true"
