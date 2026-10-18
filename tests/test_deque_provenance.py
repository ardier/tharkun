"""Closed deque transport: native counterexamples precede certificate checks."""
import ast
import os
import re
import subprocess
import sys
from textwrap import dedent

import pytest

from tharkun.solve.determinism import determinism_census
from tharkun.solve.provenance import Provenance
from tharkun.solve.pyresolve import Resolver
from tests.test_address_provenance import renamed


IMPORT = "from collections import deque as Queue\n"


def native(source, seed=0):
    return subprocess.run([sys.executable, "-c", source], text=True, capture_output=True,
                          timeout=5, env=dict(os.environ, PYTHONHASHSEED=str(seed)))


CALLBACKS = [
    ("iter", """
        class Values:
            def __iter__(self):
                print('iter')
                return iter((1,))
        q = Queue()
        q.extend(Values())
    """, "iter\n"),
    ("next", """
        class Values:
            def __iter__(self): return self
            def __next__(self):
                print('next')
                raise StopIteration
        q = Queue()
        q.extend(Values())
    """, "next\n"),
    ("length_hint_wrapper", """
        class Values:
            def __iter__(self): return iter((1,))
            def __length_hint__(self):
                print('hint')
                return 1
        q = Queue()
        q.extend(list(Values()))
    """, "hint\n"),
    ("hash", """
        class Key:
            def __hash__(self):
                print('hash')
                return 0
        q = Queue()
        q.append(Key())
        seen = set()
        seen.add(q.popleft())
    """, "hash\n"),
    ("equality", """
        class Key:
            def __hash__(self): return 0
            def __eq__(self, other):
                print('eq')
                return False
        seen = set()
        seen.add(Key())
        q = Queue()
        q.append(Key())
        seen.add(q.popleft())
    """, "eq\n"),
    ("installed_equality", """
        def equal(self, other):
            print('eq')
            return False
        class Key:
            def __hash__(self): return 0
            __eq__ = equal
        seen = set()
        seen.add(Key())
        q = Queue()
        q.append(Key())
        seen.add(q.popleft())
    """, "eq\n"),
    ("descriptor", """
        class Field:
            def __get__(self, obj, owner):
                print('field')
                return [1]
        class Key:
            items = Field()
        q = Queue()
        q.append(Key())
        q.extend(q.popleft().items)
    """, "field\n"),
    ("attribute_hook", """
        class Key:
            def __getattribute__(self, name):
                print('field')
                return [1]
        q = Queue()
        q.append(Key())
        q.extend(q.popleft().items)
    """, "field\n"),
    ("generator_filter", """
        def keep(value):
            print('filter')
            return True
        q = Queue()
        q.extend(x for x in [1] if keep(x))
    """, "filter\n"),
]


@pytest.mark.parametrize("name,body,output", CALLBACKS, ids=[c[0] for c in CALLBACKS])
def test_native_callback_effects_are_refused(name, body, output):
    source = IMPORT + dedent(body)
    result = native(source)
    assert result.returncode == 0 and result.stdout == output, result.stderr
    for program in (source, renamed(source)):
        assert not determinism_census(program).ok


def test_native_callback_without_text_sink_is_refused():
    source = IMPORT + dedent("""
        class Key:
            def __hash__(self):
                assert False
        q = Queue()
        q.append(Key())
        seen = set()
        seen.add(q.popleft())
    """)
    result = native(source)
    assert result.returncode != 0 and "AssertionError" in result.stderr
    assert not determinism_census(source).ok


@pytest.mark.parametrize("carrier", ["values", "list(values)", "tuple(values)",
                                    "iter(values)", "list(tuple(values))"])
def test_native_set_order_is_not_laundered(carrier):
    source = IMPORT + ("values = {'apple', 'banana', 'cherry', 'date', 'fig', 'grape'}\n"
                       f"q = Queue()\nq.extend({carrier})\n")
    runs = [native(source + "print(tuple(q))\n", seed) for seed in range(4)]
    assert all(r.returncode == 0 for r in runs), [r.stderr for r in runs]
    assert len({r.stdout for r in runs}) > 1
    assert not determinism_census(source).ok  # no output sink is needed


@pytest.mark.parametrize("expression", ["q", "[q]", "(q,)", "q.popleft()", "[q.popleft()]"])
def test_native_addresses_are_not_transport_permissions(expression):
    source = IMPORT + "class Item: pass\nq = Queue()\nq.append(Item())\n"
    source += f"print(repr({expression}))\n"
    result = native(source)
    assert result.returncode == 0 and re.search(r"object at 0x[0-9a-f]+", result.stdout)
    assert not determinism_census(source).ok


SAFE = [
    "q = Queue([1, 2]); q.extend((3, 4)); q.append(5)\n"
    "print(q.popleft(), q.popleft(), q.popleft(), q.popleft(), q.popleft())",
    "q = Queue(); q.extend([]); q.extend(())\nassert not q\nassert len(q) == 0",
    "def fill(target, values=(2, 3), /, *, first=1):\n"
    "    target.append(first)\n    target.extend(values)\n"
    "q = Queue()\nfill(q)\nalias = q\nalias.append(4)\n"
    "print(q.popleft(), q.popleft(), q.popleft(), q.popleft())",
    "def value(): return 17\nq = Queue(); q.append(value()); print(q.popleft())",
    "def put(value, /, *, target): target.append(value)\n"
    "q = Queue()\nput(*(3,), target=q)\nprint(q.popleft())",
    "q = Queue(); q.extend(x for x in [1, 2]); print(q.popleft(), q.popleft())",
]


@pytest.mark.parametrize("body", SAFE)
def test_ordered_primitive_transport_and_aliases(body):
    source = IMPORT + body + "\n"
    for program in (source, renamed(source), source.replace(IMPORT, "import collections as c\n")
                    .replace("Queue(", "c.deque(")):
        result = native(program)
        assert result.returncode == 0, result.stderr
        certificate = determinism_census(program)
        assert certificate.ok, certificate.reason


GRAPH = dedent("""
    class Item:
        def __init__(self, value=None, links=[]):
            self.value = value
            self.links = links
    def walk(start, goal):
        q = Queue()
        q.append(start)
        seen = set()
        seen.add(start)
        while q:
            current = q.popleft()
            if current is goal:
                return True
            q.extend(current for current in current.links if current not in seen)
            seen.update(current.links)
        return False
    goal = Item('goal')
    left = Item('left', [goal])
    start = Item('start', [left])
    left.links = [goal, start]
    print(walk(start, goal), walk(start, start), walk(goal, start))
""")


@pytest.mark.parametrize("distinct_target", [False, True])
def test_plain_record_identity_fields_and_generator_scope(distinct_target):
    source = IMPORT + GRAPH
    if distinct_target:
        source = source.replace("current for current in current.links if current not in seen",
                                "neighbor for neighbor in current.links if neighbor not in seen")
    for program in (source, renamed(source)):
        result = native(program)
        assert result.returncode == 0 and result.stdout == "True True False\n", result.stderr
        certificate = determinism_census(program)
        assert certificate.ok, certificate.reason


@pytest.mark.parametrize("tail", [
    "q.append(Item())",
    "alias = q; alias.append(Item())",
    "def add(target, value=Item()): target.append(value)\nadd(q)",
    "def add(target, value): target.append(value)\nadd(q, 2)\nadd(q, Item())",
])
def test_later_alias_default_and_second_call_values_join(tail):
    source = IMPORT + "class Item: pass\nq = Queue(); q.append(1)\nprint(q.popleft())\n" + tail
    result = native(source)
    assert result.returncode == 0 and result.stdout == "1\n", result.stderr
    assert not determinism_census(source).ok


def test_later_field_write_is_not_hidden_by_an_earlier_primitive_observation():
    source = IMPORT + dedent("""
        class Item:
            def __init__(self, value): self.value = value
        item = Item(1)
        q = Queue()
        q.append(item)
        print(q.popleft().value)
        item.value = Item(2)
    """)
    result = native(source)
    assert result.returncode == 0 and result.stdout == "1\n", result.stderr
    assert not determinism_census(source).ok


REFUSED = [
    "q = Queue([], maxlen=1)", "q = Queue([], 1)", "q = Queue(*([],))",
    "factory = Queue; q = factory()", "class Child(Queue): pass\nq = Child()",
    "q = Queue(); method = q.append; method(1)",
    "q = Queue([1]); method = q.popleft; method()",
    "q = Queue(); method = q.extend; method([1])",
    "q = Queue(); q.append(value=1)", "q = Queue(); q.extend([1], [2])",
    "q = Queue([1]); q.remove(1)", "q = Queue([1]); q.count(1)",
    "q = Queue([1]); q.index(1)", "q = Queue(); q.clear()",
    "q = Queue([1]); assert q == Queue([1])",
    "q = Queue([1, 2]); left, right = q",
    "q = Queue([1, 2]); left, *rest = q",
    "items = [1]; alias = items; alias.append(2); q = Queue(items)",
    "items = [1]; alias = items; alias[0] = 2; q = Queue(items)",
    "q = Queue(list((1,)))", "q = Queue(); q.extend(iter([1]))",
    "q = Queue(); q.extend(reversed([1]))",
    "q = Queue(); gen = (x for x in [1]); q.extend(gen)",
    "q = Queue(); q.extend(x + 1 for x in [1])",
    "q = Queue(); q.extend(x for x in [1] for y in [2])",
    "q = Queue(); q.extend(x for x in [1] if x)",
    "class Item: pass\nfactory = Item\nq = Queue(); q.append(factory())",
    "class Item: pass\nclass Child(Item): pass\nq = Queue(); q.append(Child())",
    "q = Queue(); values = missing; q.extend(values)",
    "def fill(q, values=missing): q.extend(values)\nq = Queue(); fill(q, [])",
    "q = Queue(); seen = set(); seen.add(1); q.extend(seen)",
    "q = Queue(); seen = set(); seen.add(1); seen.pop()",
    "q = Queue(); seen = set(); seen.add(1); print(seen)",
    "q = Queue(); q = []; q.append(1)",
    "q = Queue(); print(q)", "q = Queue([1]); print([q])",
]


@pytest.mark.parametrize("body", REFUSED)
def test_unsupported_protocols_and_unknown_bindings_refuse(body):
    source = IMPORT + body + "\n"
    assert not determinism_census(source).ok


@pytest.mark.parametrize("body", [
    "from collections import defaultdict\nd = defaultdict(int)\nd.default_factory = float",
    "from collections import defaultdict\nd = defaultdict(int)\nprint([d])",
    "import heapq\nclass Item: pass\nheapq.heapify([Item(), Item()])",
    "import heapq\npush = heapq.heappush\npush([], 1)",
])
def test_deque_does_not_waive_other_library_effect_boundaries(body):
    assert not determinism_census(IMPORT + "q = Queue()\n" + body).ok


def test_set_transfer_requires_the_deque_effect_check():
    tree = ast.parse("seen = set()")
    p = Provenance(Resolver(tree, ()))
    assert not p.deque_calls
    assert p.value(tree.body[0].value) == {("unknown",)}


def test_comprehension_provenance_keeps_outer_and_inner_bindings_separate():
    source = "def f():\n    x = 'outer'\n    items = (x for x in [1])\n    print(x)\nf()"
    tree = ast.parse(source)
    p = Provenance(Resolver(tree, ()))
    gen = next(n for n in ast.walk(tree) if isinstance(n, ast.GeneratorExp))
    assert p.value(gen.elt) == {("int",)}
    assert p.get(("name", tree.body[0], "x")) == {("str",)}


def test_empty_deque_pop_remains_a_deterministic_violation(tmp_path):
    from tharkun.sv.engine import verify

    source = IMPORT + "q = Queue()\nq.popleft()\n"
    result = native(source)
    assert result.returncode != 0 and "IndexError" in result.stderr
    assert determinism_census(source).ok
    path = tmp_path / "empty_queue.py"
    path.write_text(source)
    verdict = verify(str(path), ("IndexError",), budget_seconds=10, jobs=1, prove=False)
    assert verdict.line() == "FALSE(IndexError)", verdict
