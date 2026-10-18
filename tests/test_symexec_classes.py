"""User classes in the whole-harness symbolic executor ([internal ref]; ``sv/_symexec_objects.py``, docs/SV.md "The
call_target fact").

Every modelled construct is a program of draws and ``assert`` sites run twice: through ``solve_targets`` with the
program's own ``call_target`` records, and natively.  A ``sat`` model must make exactly that assert fail natively;
an ``unsat`` / ``unreached`` site must never fail natively on a box of inputs; a refusal names its construct.
That makes each test a differential on the heap (aliasing, per-path forks, class vs instance attributes,
``super()``, ``isinstance``); the generator of ``scripts/class_differential.py`` is the same check at scale.
"""
from __future__ import annotations

import ast
import itertools
import sys
import textwrap
import traceback

import pytest

pytest.importorskip("z3")

from tharkun.solve.call_targets import call_targets  # noqa: E402
from tharkun.solve.module_namespace import module_constants  # noqa: E402
from tharkun.sv import _symexec  # noqa: E402
from tharkun.sv.sites import build_harness  # noqa: E402

HEADER = "import _sv_verifier\n"
MAIN = 'if __name__ == "__main__":\n'


def program(classes: str, main: str, draws: tuple = ("x",)) -> str:
    """A task: the class definitions, then a main block drawing ``draws`` and running ``main``."""
    body = "".join(f"    {d} = _sv_verifier.nondet_int()\n" for d in draws) + textwrap.indent(
        textwrap.dedent(main).strip("\n"), "    ") + "\n"
    return HEADER + "\n" + textwrap.dedent(classes).strip("\n") + "\n\n" + MAIN + body


class Spy(_symexec._Runner):
    instances: list = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        Spy.instances.append(self)


def solve(src: str, tmp_path, *, targets="auto"):
    """``(outcomes in site order, runner, harness)``; ``targets`` is ``"auto"`` (the producer's records), a list or None."""
    path = tmp_path / "task.py"
    path.write_text(src)
    constants = module_constants(str(path))
    records = call_targets(str(path), constants) if targets == "auto" else targets
    proven = [c["name"] for c in constants if c["proven"] and c["origin"] == "class"]
    h = build_harness(src, ("AssertionError",), None)
    mod = {"__name__": "sv_harness_test"}
    sys.modules.setdefault("_sv_verifier", type(sys)("_sv_verifier"))
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    Spy.instances = []
    orig = _symexec._Runner
    _symexec._Runner = Spy
    try:
        got = _symexec.solve_targets(h.source, mod, "__sv_harness", [s["line"] for s in h.sites], h.params, h.kinds, {},
                                     call_targets=records, proven_classes=proven)
    finally:
        _symexec._Runner = orig
    sites = sorted(h.sites, key=lambda s: s["line"])
    return [got[s["line"]] for s in sites], Spy.instances[-1], h


def native(src: str, draws: list):
    """The line of the ``assert`` that fails natively with these draws, or ``None``."""
    queue = list(draws)
    stub = type(sys)("_sv_verifier")
    stub.nondet_int = lambda: queue.pop(0)
    old = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = stub
    try:
        exec(compile(src, "<native>", "exec"), {"__name__": "__main__"})
    except AssertionError as e:
        return traceback.extract_tb(e.__traceback__)[-1].lineno
    except Exception:           # noqa: BLE001 -- AttributeError and friends end the run without a violation
        return None
    finally:
        if old is None:
            del sys.modules["_sv_verifier"]
        else:
            sys.modules["_sv_verifier"] = old
    return None


def check(src: str, tmp_path, expected: list, box=range(-3, 4)):
    """Run ``src`` both ways and compare, site by site.  ``expected`` per assert: ``sat``, ``unsat`` (``unsat`` or
    ``unreached``), or ``unsupported`` -- a refusal names its reason through ``runner.skipped``/the outcome."""
    out, runner, h = solve(src, tmp_path)
    asserts = sorted(n.lineno for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Assert))
    assert len(out) == len(asserts) == len(expected), (out, asserts)
    fired = set()
    for combo in itertools.product(box, repeat=len(h.params)):
        line = native(src, list(combo))
        if line is not None:
            fired.add(line)
    for (outcome, payload), line, want in zip(out, asserts, expected):
        if want == "sat":
            assert outcome == "sat", (line, outcome, payload)
            assert native(src, [payload[p] for p in h.params]) == line, (line, payload)
        elif want == "unsat":
            assert outcome in ("unsat", "unreached"), (line, outcome, payload)
            assert line not in fired, f"the assert at line {line} fails natively but the model said {outcome}"
        else:
            assert outcome == "unsupported", (line, outcome, payload)
    return out, runner


# ---------------------------------------------------------------------------------------------------------
# construction, attributes, methods
# ---------------------------------------------------------------------------------------------------------

COUNTER = """
class Counter:
    start = 10
    def __init__(self, c):
        self.c = c
    def inc(self, v):
        self.c += v
        return self.c
"""


def test_construction_instance_attributes_and_a_method_that_mutates_self(tmp_path):
    src = program(COUNTER, """
        k = Counter(x)
        r = k.inc(5)
        assert r != 12
        assert k.c != 20
        assert k.c != Counter.start + 1 + 5
    """)
    out, _ = check(src, tmp_path, ["sat", "sat", "sat"])
    assert out[0][1]["x"] == 7 and out[1][1]["x"] == 15


def test_a_class_attribute_is_read_through_the_class_and_the_instance_and_shadowed_by_the_instance(tmp_path):
    src = program("""
        class A:
            limit = 10
            def __init__(self, v):
                self.v = v
            def over(self):
                return self.v > self.limit
    """, """
        a = A(x)
        b = A(1)
        b.limit = 0
        assert not (a.over() and a.limit == 10 and A.limit == 10 and b.over() and x == 11)
        assert A.limit == 10
        assert not (A.limit == 0 or a.limit == 0 and not b.over())
    """)
    check(src, tmp_path, ["sat", "unsat", "unsat"])


def test_a_class_attribute_stored_at_import_time_is_what_the_program_reads(tmp_path):
    src = program("""
        class A:
            zero = 0
        A.zero = 5
        A.extra = 7
    """, """
        assert A.zero != 5 + x - x
        assert A.extra == 7
    """)
    check(src, tmp_path, ["sat", "unsat"])


def test_two_references_to_one_object_see_each_others_writes(tmp_path):
    src = program("""
        class Box:
            def __init__(self, v):
                self.v = v
    """, """
        a = Box(x)
        b = a
        b.v = b.v + 1
        c = Box(x)
        assert a.v != 5
        assert c.v != 4
    """)
    check(src, tmp_path, ["sat", "unsat"])


def test_objects_are_compared_by_identity_unless_the_class_says_otherwise(tmp_path):
    src = program("""
        class A:
            def __init__(self, v):
                self.v = v
    """, """
        a = A(x)
        b = A(x)
        c = a
        assert a is not b and a is c and a == c and a != b and bool(c) and not (a is None)
        assert a is b
    """)
    check(src, tmp_path, ["unsat", "sat"])


def test_a_missing_attribute_ends_the_path(tmp_path):
    src = program("""
        class A:
            def __init__(self):
                self.v = 1
    """, """
        a = A()
        y = a.w
        assert x != 3
    """)
    out, _ = check(src, tmp_path, ["unsat"])
    assert out[0][0] == "unreached"


def test_an_assert_inside_a_method_is_a_site(tmp_path):
    src = program("""
        class A:
            def __init__(self, v):
                self.v = v
            def check(self):
                assert self.v != 42
                return self.v
    """, """
        a = A(x)
        a.check()
    """)
    out, _ = check(src, tmp_path, ["sat"])
    assert out[0][1]["x"] == 42


def test_methods_calling_methods_and_objects_passed_to_functions(tmp_path):
    src = program("""
        class P:
            def __init__(self, a, b):
                self.a = a
                self.b = b
            def total(self):
                return self.a + self.b
            def scaled(self, k):
                return P(self.a * k, self.b * k)

        def grow(p, k):
            return p.scaled(k).total() + 1
    """, """
        p = P(x, 2)
        assert grow(p, 3) != 13
        assert p.total() != 5
    """)
    check(src, tmp_path, ["sat", "sat"])


# ---------------------------------------------------------------------------------------------------------
# the per-path heap
# ---------------------------------------------------------------------------------------------------------

def test_a_store_in_one_arm_of_a_symbolic_if_does_not_leak_into_the_other(tmp_path):
    src = program("""
        class Box:
            def __init__(self, v):
                self.v = v
    """, """
        a = Box(0)
        b = Box(0)
        if x > 0:
            a.v = 1
        else:
            b.v = 2
        assert not (a.v == 0 and b.v == 0)
        assert not (a.v == 1 and b.v == 2)
        assert not (a.v == 1 and b.v == 0)
        assert not (a.v == 0 and b.v == 2)
    """)
    check(src, tmp_path, ["unsat", "unsat", "sat", "sat"])


def test_an_attribute_stored_in_one_arm_only_merges_with_the_class_default(tmp_path):
    src = program("""
        class A:
            d = 7
    """, """
        a = A()
        if x > 0:
            a.d = 1
        assert a.d != 1
        assert a.d != 7
        assert a.d != 9
    """)
    check(src, tmp_path, ["sat", "sat", "unsat"])


def test_a_class_attribute_stored_in_one_arm_only_merges_with_its_old_value(tmp_path):
    src = program("""
        class Cnt:
            n = 0
    """, """
        if x > 0:
            Cnt.n = Cnt.n + 5
        assert Cnt.n != 5
        assert Cnt.n != 0
        assert Cnt.n != 3
    """)
    check(src, tmp_path, ["sat", "sat", "unsat"])


def test_a_method_with_early_returns_forks_the_heap(tmp_path):
    src = program("""
        class C:
            def __init__(self):
                self.n = 0
            def bump(self, c):
                if c > 0:
                    self.n += 1
                    return 1
                self.n += 10
                return 2
    """, """
        o = C()
        r = o.bump(x)
        assert not (r == 1 and o.n == 1)
        assert not (r == 2 and o.n == 10)
        assert not (r == 1 and o.n == 10)
        assert not (r == 2 and o.n == 1)
    """)
    check(src, tmp_path, ["sat", "sat", "unsat", "unsat"])


def test_a_conditional_expression_applies_its_effects_only_where_it_runs(tmp_path):
    src = program("""
        class C:
            def __init__(self):
                self.n = 0
            def inc(self):
                self.n += 1
                return self.n
            def get(self):
                return self.n
    """, """
        o = C()
        y = o.inc() if x > 0 else o.get()
        assert not (x > 0 and o.n == 0)
        assert not (x <= 0 and o.n == 1)
        assert not (x > 0 and y == 1 and o.n == 1)
    """)
    check(src, tmp_path, ["unsat", "unsat", "sat"])


def test_a_short_circuit_applies_its_effects_only_where_it_runs(tmp_path):
    src = program("""
        class C:
            def __init__(self):
                self.n = 0
            def inc(self):
                self.n += 1
                return True
    """, """
        o = C()
        ok = x > 0 and o.inc()
        assert not (x > 0 and o.n == 0)
        assert not (x <= 0 and o.n == 1)
        assert not (x > 0 and o.n == 1)
    """)
    check(src, tmp_path, ["unsat", "unsat", "sat"])


def test_a_loop_of_constructions_and_stores(tmp_path):
    src = program("""
        class A:
            def __init__(self, v):
                self.v = v
    """, """
        objs = []
        for i in range(3):
            objs.append(A(i + x))
        objs[1].v = 100
        assert objs[0].v == x
        assert objs[2].v == x + 2
        assert objs[1].v != 100
    """)
    check(src, tmp_path, ["unsat", "unsat", "sat"])


def test_objects_born_on_one_arm_are_unusable_after_the_merge(tmp_path):
    src = program("""
        class A:
            def __init__(self, v):
                self.v = v
    """, """
        if x > 0:
            p = A(1)
        else:
            p = A(2)
        assert p.v != 1
    """)
    out = solve(src, tmp_path)[0]
    assert out[0][0] == "unsupported" and "p was not modelled" in out[0][1]


# ---------------------------------------------------------------------------------------------------------
# static and class methods, inheritance, super, isinstance
# ---------------------------------------------------------------------------------------------------------

def test_static_and_class_methods_through_the_class_and_the_instance(tmp_path):
    src = program("""
        class K:
            total = 5
            def __init__(self, v):
                self.v = v
            @staticmethod
            def twice(v):
                return v * 2
            @classmethod
            def base(cls):
                return cls.total
    """, """
        k = K(x)
        assert K.twice(x) != 14
        assert k.twice(x) != 20
        assert K.base() == 5
        assert k.base() == 5
    """)
    check(src, tmp_path, ["sat", "sat", "unsat", "unsat"])


INHERIT = """
class Base:
    def __init__(self, v):
        self.v = v
        self.tag = 1
    def val(self):
        return self.v
    def who(self):
        return 1

class Child(Base):
    def __init__(self, v, w):
        super().__init__(v)
        self.w = w
        self.tag = 2
    def val(self):
        return super().val() + self.w
    def who(self):
        return 2

class Grand(Child):
    def who(self):
        return 3 + super().who()
"""


def test_single_inheritance_override_and_super_resolve_like_python(tmp_path):
    src = program(INHERIT, """
        c = Child(x, y)
        g = Grand(x, 1)
        b = Base(x)
        assert c.val() != 10
        assert g.who() == 5
        assert g.val() == x + 1 and c.tag == 2 and b.tag == 1 and g.tag == 2 and Base.who(c) == 1
    """, draws=("x", "y"))
    check(src, tmp_path, ["sat", "unsat", "unsat"])


def test_isinstance_follows_inheritance(tmp_path):
    src = program(INHERIT, """
        c = Child(x, 1)
        b = Base(x)
        assert isinstance(c, Base) and isinstance(c, Child) and not isinstance(b, Child)
        assert isinstance(b, (int, Base)) and not isinstance(x, Base) and not isinstance(3, Child)
        assert not isinstance(c, Child)
    """)
    check(src, tmp_path, ["unsat", "unsat", "sat"])


def test_class_attributes_are_shared_through_the_hierarchy_and_a_subclass_store_shadows(tmp_path):
    src = program("""
        class Cnt:
            count = 0
            def __init__(self):
                Cnt.count = Cnt.count + 1

        class Sub(Cnt):
            pass
    """, """
        a = Cnt()
        b = Cnt()
        s = Sub()
        assert Cnt.count == 3 and Sub.count == 3 and s.count == 3 and a.count == 3
        Sub.count = 100
        assert Cnt.count == 3 and a.count == 3 and s.count == 100 and Sub.count == 100
        assert Cnt.count != 3
    """)
    check(src, tmp_path, ["unsat", "unsat", "sat"])


def test_an_explicit_unbound_call_runs_the_named_class_method(tmp_path):
    src = program("""
        class A:
            def __init__(self, v):
                self.v = v
            def get(self):
                return self.v
        class B(A):
            def __init__(self, v):
                A.__init__(self, v + 1)
            def get(self):
                return A.get(self) * 2
    """, """
        b = B(x)
        assert b.get() != 8
    """)
    out, _ = check(src, tmp_path, ["sat"])
    assert out[0][1]["x"] == 3


# ---------------------------------------------------------------------------------------------------------
# every named refusal
# ---------------------------------------------------------------------------------------------------------

def refused(src: str, tmp_path, needle: str, *, targets="auto"):
    out, runner, _ = solve(src, tmp_path, targets=targets)
    reasons = "\n".join(runner.skipped) + "\n" + "\n".join(str(p) for _o, p in out)
    assert needle in reasons, (needle, runner.skipped, out)
    assert all(o != "sat" for o, _p in out), out


REFUSALS = {
    "metaclass": ("""
        class M(type):
            pass
        class A(metaclass=M):
            pass
    """, "a = A()\nassert a is None", "has a metaclass"),
    "multiple inheritance": ("""
        class P:
            pass
        class Q:
            pass
        class A(P, Q):
            pass
    """, "a = A()\nassert a is None", "multiple inheritance"),
    "base outside the program": ("""
        class A(Exception):
            pass
    """, "a = A()\nassert a is None", "is not a class of the program"),
    "__slots__": ("""
        class A:
            __slots__ = ("v",)
    """, "a = A()\nassert a is None", "defines __slots__"),
    "__new__": ("""
        class A:
            def __new__(cls):
                return object.__new__(cls)
    """, "a = A()\nassert a is None", "defines __new__"),
    "__setattr__": ("""
        class A:
            def __setattr__(self, k, v):
                pass
    """, "a = A()\nassert a is None", "defines __setattr__"),
    "property": ("""
        class A:
            @property
            def p(self):
                return 1
    """, "a = A()\nassert a.p != 1", "a descriptor (property)"),
    "descriptor": ("""
        class D:
            def __get__(self, obj, owner):
                return 1
        class A:
            d = D()
    """, "a = A()\nassert a.d != 1", "a descriptor (D)"),
    "generator method": ("""
        class A:
            def g(self):
                yield 1
    """, "a = A()\nassert a is None", "generator or async"),
    "decorated class": ("""
        def deco(c):
            return c
        @deco
        class A:
            pass
    """, "a = A()\nassert a is None", "decorated"),
    "dunder attribute": ("""
        class A:
            def __init__(self):
                self.v = 1
    """, "a = A()\nassert a.__class__ is None", "dunder and private-mangled"),
    "private-mangled attribute": ("""
        class A:
            def __init__(self):
                self.__v = 1
    """, "a = A()\nassert a is None", "dunder and private-mangled"),
    "method as a value": ("""
        class A:
            def m(self):
                return 1
    """, "a = A()\nf = a.m\nassert f() != 1", "a method used as a value"),
    "list in an attribute": ("""
        class A:
            def __init__(self):
                self.items = []
    """, "a = A()\nassert a is None", "a mutable container"),
    "dict in an attribute": ("""
        class A:
            def __init__(self):
                self.d = {}
    """, "a = A()\nassert a is None", "a mutable container"),
    "class attribute list read": ("""
        class A:
            registry = []
    """, "a = A()\nassert len(a.registry) != 1", "holds a list"),
    "operator on objects": ("""
        class A:
            def __init__(self, v):
                self.v = v
            def __add__(self, o):
                return A(self.v + o.v)
    """, "a = A(x)\nb = a + a\nassert b.v != 2", "applied to a A object"),
    "ordering of objects": ("""
        class A:
            def __lt__(self, o):
                return True
    """, "a = A()\nb = A()\nassert not (a < b)", "applied to a A object"),
    "format of an object": ("""
        class A:
            pass
    """, "a = A()\ns = f'{a}'\nassert s != 'x'", "applied to a A object"),
    "user __eq__": ("""
        class A:
            def __eq__(self, o):
                return True
    """, "a = A()\nb = A()\nassert not (a == b)", "the user-defined __eq__"),
    "user __bool__": ("""
        class A:
            def __bool__(self):
                return False
    """, "a = A()\nassert not a", "the user-defined __bool__"),
    "keyword arguments at a constructor": ("""
        class A:
            def __init__(self, v):
                self.v = v
    """, "a = A(v=x)\nassert a.v != 1", "call shape of __init__"),
    "defaults at a constructor": ("""
        class A:
            def __init__(self, v=1):
                self.v = v
    """, "a = A()\nassert a.v != 1", "call shape of __init__"),
    "super outside an instance method": ("""
        class A:
            def m(self):
                return 1
        class B(A):
            @classmethod
            def c(cls):
                return super().m()
    """, "assert B.c() != 1", "super() outside an instance method"),
    "type of an object": ("""
        class A:
            pass
    """, "a = A()\nassert type(a) is None", "type() of an unmodelled value"),
    "any over a call": ("""
        class A:
            def ok(self):
                return True
    """, "a = A()\nassert not any(a.ok() for i in range(2))", "any()/all() over an expression that calls code"),
    "store through a call": ("""
        class A:
            pass
        def make():
            return A()
    """, "make().v = 1\nassert x != 3 + x", "an attribute store through a call expression"),
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_every_class_construct_the_model_does_not_follow_is_a_named_refusal(tmp_path, name):
    classes, main, needle = REFUSALS[name]
    refused(program(classes, main), tmp_path, needle)


def test_an_attribute_store_on_a_number_is_a_named_refusal_and_costs_nothing_else(tmp_path):
    src = program("""
        class A:
            pass
    """, "x.v = 1\nassert x != 3")
    out, runner, _ = solve(src, tmp_path)
    assert any("attribute store on" in s for s in runner.skipped)
    assert out[0][0] == "sat" and out[0][1]["x"] == 3        # the statement is skipped, the assert is independent


def test_an_unproven_class_name_is_not_constructed(tmp_path):
    src = program("""
        class A:
            pass
        class B:
            pass
        A = B
    """, "a = A()\nassert a is None")
    refused(src, tmp_path, "the call_target is user_class, proven=False")


def test_a_class_without_a_call_target_fact_is_not_constructed(tmp_path):
    src = program("""
        class A:
            pass
    """, "a = A()\nassert a is None")
    refused(src, tmp_path, "call A: no call_target fact for it", targets=[])


def test_without_call_targets_a_call_of_a_class_stays_unmodelled_as_before(tmp_path):
    src = program("""
        class A:
            pass
    """, "a = A()\nassert a is None")
    refused(src, tmp_path, "call A", targets=None)


def test_a_method_patched_at_import_time_is_refused_and_a_wrapped_one_too(tmp_path):
    patched = program("""
        def other(self):
            return 2
        class A:
            def m(self):
                return 1
        A.m = other
    """, "a = A()\nassert a.m() != 2")
    refused(patched, tmp_path, "the live method is not the def of the class body")
    wrapped = program("""
        import functools
        def logged(f):
            @functools.wraps(f)
            def w(*a):
                return f(*a)
            return w
        class A:
            @logged
            def m(self):
                return 1
    """, "a = A()\nassert a.m() != 1")
    refused(wrapped, tmp_path, "the live method is not the def of the class body")


def test_the_class_name_must_be_proven_even_when_only_isinstance_reads_it(tmp_path):
    src = program("""
        class A:
            pass
        class B:
            pass
        def swap():
            global A
            A = B
    """, "a = B()\nassert isinstance(a, A)")
    refused(src, tmp_path, "its name is not proven constant")


# ---------------------------------------------------------------------------------------------------------
# skipped statements and the heap
# ---------------------------------------------------------------------------------------------------------

STALE = """
class A:
    def __init__(self, v):
        self.v = v

G = 0

def sneaky(o):
    global G
    G = 1
    o.v = 99
"""


def test_a_skipped_statement_that_may_have_written_makes_existing_objects_unknown(tmp_path):
    src = program(STALE, """
        a = A(x)
        sneaky(a)
        assert a.v != 99
    """)
    out, runner, _ = solve(src, tmp_path)
    assert out[0][0] == "unsupported"
    assert any("object state is unknown after an unmodelled statement" in s or "writes a global" in s
               for s in runner.skipped), runner.skipped
    assert out[0][0] != "sat"


def test_a_skipped_print_does_not_make_objects_unknown(tmp_path):
    src = program(STALE, """
        a = A(x)
        print("v", a.v, len([x]))
        assert a.v != 5
    """)
    out, runner, _ = solve(src, tmp_path)
    assert out[0][0] == "sat" and out[0][1]["x"] == 5


def test_after_such_a_statement_nothing_of_the_heap_is_trusted_not_even_a_construction(tmp_path):
    src = program(STALE, """
        a = A(x)
        sneaky(a)
        b = A(x)
        assert b.v != 5
    """)
    out, runner, _ = solve(src, tmp_path)
    assert out[0][0] == "unsupported"
    assert any("class state is unknown after an unmodelled statement" in s for s in runner.skipped)


def test_class_state_is_unknown_after_a_skipped_statement_that_may_have_rebound_it(tmp_path):
    src = program(STALE + "\nclass K:\n    n = 1\n", """
        a = A(x)
        sneaky(a)
        assert K.n != 1
    """)
    out, runner, _ = solve(src, tmp_path)
    assert out[0][0] == "unsupported"


# ---------------------------------------------------------------------------------------------------------
# the proof side stays out of it
# ---------------------------------------------------------------------------------------------------------

def test_run_exact_never_models_a_class_call(tmp_path):
    """``run_exact`` is the whole-run exactness a PROOF needs; it is handed no call_target records, so a class
    call is unmodelled there and the run is not exact (class support is search-only)."""
    tree = ast.parse("class A:\n    pass\na = A()\n")
    res = _symexec.run_exact(tree, {}, {}, [tree.body[1]], [])
    assert res["exact"] is False


def test_the_exact_product_proof_refuses_a_program_with_a_class(tmp_path):
    from tharkun.sv import product_proof

    src = program("""
        class A:
            def __init__(self, v):
                self.v = v
    """, "a = A(x)\nassert a.v != 3")
    assert product_proof.prove(src, ("AssertionError",), program=str(tmp_path / "task.py")) is None


# ---------------------------------------------------------------------------------------------------------
# calls the object check looks at but does not take
# ---------------------------------------------------------------------------------------------------------

def test_a_receiver_that_runs_code_is_evaluated_once_when_it_is_not_an_object(tmp_path):
    """``mk(s).startswith('ab')``: the object check evaluates the receiver ``mk(s)`` (it runs code: it bumps a class
    counter), finds a string, and hands the value to the generic dispatch -- which must not run ``mk`` again."""
    src = HEADER + """
class Cnt:
    n = 0

def mk(s):
    Cnt.n = Cnt.n + 1
    return s

""" + MAIN + """    s = _sv_verifier.nondet_str()
    t = mk(s).startswith("ab")
    assert Cnt.n == 1
    assert not t
"""
    path = tmp_path / "task.py"
    path.write_text(src)
    constants = module_constants(str(path))
    records = call_targets(str(path), constants)
    h = build_harness(src, ("AssertionError",), None)
    mod = {"__name__": "sv_harness_test"}
    sys.modules.setdefault("_sv_verifier", type(sys)("_sv_verifier"))
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    got = _symexec.solve_targets(h.source, mod, "__sv_harness", [s["line"] for s in h.sites], h.params, h.kinds, {},
                                 call_targets=records, proven_classes=["Cnt"])
    first, second = (got[s["line"]] for s in sorted(h.sites, key=lambda s: s["line"]))
    assert first[0] in ("unsat", "unreached"), first          # the counter is 1: ``mk`` ran exactly once
    assert second[0] == "sat" and second[1]["s"].startswith("ab"), second


def test_a_recursive_method_beyond_the_call_depth_is_refused_not_unrolled_forever(tmp_path):
    src = program("""
        class A:
            def down(self, n):
                return self.down(n + 1)
    """, "a = A()\nassert a.down(x) != 1")
    out, runner, _ = solve(src, tmp_path)
    assert any("call depth" in s for s in runner.skipped), runner.skipped


def test_ordinary_calls_on_non_objects_are_untouched_by_the_object_check(tmp_path):
    src = program("""
        class A:
            pass
    """, """
        xs = [x, 2]
        xs.append(3)
        d = {1: 5}
        assert d.get(1, 0) + len(xs) != 8
    """)
    out, runner, _ = solve(src, tmp_path)
    assert out[0][0] == "sat", (out, runner.skipped)


# ---------------------------------------------------------------------------------------------------------
# negative control: the heap tests must FAIL when the fix they pin is undone
# ---------------------------------------------------------------------------------------------------------

def _undo_store(monkeypatch):
    from tharkun.sv import _symexec_objects as objects

    monkeypatch.setattr(objects.ObjectModel, "store", lambda self, base, name, value: None)


def _undo_fork(monkeypatch):
    from tharkun.sv import _symexec_objects as objects

    monkeypatch.setattr(objects.Heap, "fork", lambda self: self)


def _undo_merge(monkeypatch):
    from tharkun.sv import _symexec_objects as objects

    monkeypatch.setattr(objects.ObjectModel, "merge", lambda self, guard, a, b: a)


def _undo_super(monkeypatch):
    from tharkun.sv import _symexec_objects as objects

    real = objects.ObjectModel._chain
    monkeypatch.setattr(objects.ObjectModel, "_chain",
                        lambda self, info, name, heap, skip=False: real(self, info.base or info, name, heap, skip)
                        if getattr(self, "_in_super", False) else real(self, info, name, heap, skip))
    orig = objects.ObjectModel._call_super

    def wrapped(self, n, f):
        self._in_super = True
        try:
            return orig(self, n, f)
        finally:
            self._in_super = False

    monkeypatch.setattr(objects.ObjectModel, "_call_super", wrapped)


#: (what is undone, the undo, the tests that pin it)
UNDONE = [
    ("attribute stores", _undo_store, [test_construction_instance_attributes_and_a_method_that_mutates_self,
                                       test_two_references_to_one_object_see_each_others_writes]),
    ("one heap per path", _undo_fork, [test_a_store_in_one_arm_of_a_symbolic_if_does_not_leak_into_the_other,
                                       test_a_method_with_early_returns_forks_the_heap]),
    ("the merge of the arms' heaps", _undo_merge, [test_a_store_in_one_arm_of_a_symbolic_if_does_not_leak_into_the_other,
                                                   test_a_conditional_expression_applies_its_effects_only_where_it_runs]),
]


@pytest.mark.parametrize("what, undo, tests", UNDONE, ids=[u[0] for u in UNDONE])
def test_the_heap_tests_fail_when_the_fix_is_undone(tmp_path_factory, monkeypatch, what, undo, tests):
    for pinned in tests:
        pinned(tmp_path_factory.mktemp("clean"))                  # green with the fix in place
    undo(monkeypatch)
    failed = 0
    for pinned in tests:
        try:
            pinned(tmp_path_factory.mktemp("broken"))
        except AssertionError:
            failed += 1
    assert failed == len(tests), f"undoing {what} must fail every test that pins it ({failed} of {len(tests)} did)"


def test_class_attribute_stores_and_super_are_pinned_too(tmp_path_factory, monkeypatch):
    clean = [test_class_attributes_are_shared_through_the_hierarchy_and_a_subclass_store_shadows,
             test_single_inheritance_override_and_super_resolve_like_python]
    for pinned in clean:
        pinned(tmp_path_factory.mktemp("clean"))
    from tharkun.sv import _symexec_objects as objects

    real = objects.ObjectModel.store

    def no_class_stores(self, base, name, value):
        return None if self.table.is_user_class(base) else real(self, base, name, value)

    monkeypatch.setattr(objects.ObjectModel, "store", no_class_stores)
    with pytest.raises(AssertionError):
        clean[0](tmp_path_factory.mktemp("broken"))
    monkeypatch.undo()
    _undo_super(monkeypatch)
    with pytest.raises(AssertionError):
        clean[1](tmp_path_factory.mktemp("broken2"))
