"""``solve/call_targets.py::call_targets``: the ``call_target`` fact ([internal ref]).

What a proven record promises is pinned here: ``proven`` only from the module-namespace fact, never through a
local binding, never when anything rebinds the name or the import graph is incomplete."""
from __future__ import annotations

import textwrap

import pytest

from tharkun.solve.call_targets import call_target_index, call_targets
from tharkun.solve.module_namespace import module_constants


def _produce(tmp_path, source: str, **modules: str):
    entry = tmp_path / "main.py"
    entry.write_text(textwrap.dedent(source))
    for name, text in modules.items():
        (tmp_path / f"{name}.py").write_text(textwrap.dedent(text))
    constants = module_constants(str(entry))
    return call_targets(str(entry), constants)


def _one(records, callee, scope=None):
    hits = [r for r in records if r["callee"] == callee and (scope is None or r["scope"] == scope)]
    assert len(hits) == 1, [(r["callee"], r["scope"]) for r in records]
    return hits[0]


BASIC = """
    class Canvas:
        def __init__(self, w):
            self.w = w

    def helper(x):
        return x

    def main():
        c = Canvas(3)
        return helper(c)

    if __name__ == "__main__":
        main()
"""


def test_a_class_and_a_function_resolve_and_are_proven(tmp_path):
    recs = _produce(tmp_path, BASIC)
    ctor = _one(recs, "Canvas")
    assert (ctor["target"], ctor["name"], ctor["proven"], ctor["scope"]) == ("user_class", "Canvas", True, "main")
    assert ctor["def"] == {"module": "main.py", "line": 2, "col": 0}
    assert ctor["dispatch"] == "static" and ctor["kind"] == "call_target" and ctor["module"] == "main.py"
    fn = _one(recs, "helper")
    assert (fn["target"], fn["name"], fn["proven"]) == ("user_function", "helper", True)
    assert fn["def"]["line"] == 6
    main = _one(recs, "main")
    assert (main["target"], main["scope"], main["proven"]) == ("user_function", "<main>", True)
    assert main["provenance"] == {"producer": "tharkun", "method": "static-graph", "complete": True}


def test_every_call_site_gets_one_record_in_source_order(tmp_path):
    recs = _produce(tmp_path, BASIC)
    assert [(r["line"], r["col"]) for r in recs] == sorted((r["line"], r["col"]) for r in recs)
    assert len({(r["line"], r["col"], r["end_line"], r["end_col"]) for r in recs}) == len(recs)
    assert [r["callee"] for r in recs] == ["Canvas", "helper", "main"]


def test_a_module_level_rebinding_makes_the_name_unproven_but_keeps_its_declared_target(tmp_path):
    recs = _produce(tmp_path, """
        def f(): return 1
        def g(): return 2
        f = g
        def main(): return f()
    """)
    r = _one(recs, "f")
    assert r["target"] == "user_function" and r["proven"] is False
    assert "not proven" in r["reason"]


def test_a_conditional_definition_is_not_proven(tmp_path):
    recs = _produce(tmp_path, """
        import sys
        if len(sys.argv) > 1:
            def f(): return 1
        else:
            def f(): return 2
        def main(): return f()
    """)
    assert _one(recs, "f")["proven"] is False


def test_a_global_store_from_a_function_rebinds(tmp_path):
    recs = _produce(tmp_path, """
        def f(): return 1
        def swap():
            global f
            f = lambda: 2
        def main(): return f()
    """)
    assert _one(recs, "f", "main")["proven"] is False


def test_a_class_attribute_store_does_not_unprove_the_class_name(tmp_path):
    # a fact about NAMES: the class is still the class; a consumer checks the live class (docs/SV.md)
    recs = _produce(tmp_path, """
        class A:
            x = 1
        A.x = 2
        def main(): return A()
    """)
    assert _one(recs, "A")["proven"] is True


@pytest.mark.parametrize("reflection", [
    "exec('f = 1')",
    "import builtins",
    "setattr(__import__('sys').modules[__name__], 'f', 1)",
])
def test_reflection_makes_the_graph_incomplete_so_nothing_is_proven(tmp_path, reflection):
    recs = _produce(tmp_path, f"""
        def f(): return 1
        {reflection}
        def main(): return f() + len([])
    """)
    assert _one(recs, "f")["proven"] is False
    assert _one(recs, "len")["target"] == "builtin" and _one(recs, "len")["proven"] is False
    assert all(r["provenance"]["complete"] is False for r in recs)


def test_a_namespace_write_unproves_every_name_and_the_builtins(tmp_path):
    recs = _produce(tmp_path, """
        def f(): return 1
        globals()['f'] = lambda: 2
        def main(): return f() + len([])
    """)
    assert _one(recs, "f", "main")["proven"] is False
    assert _one(recs, "len")["proven"] is False


def test_an_imported_function_reports_the_module_that_defines_it(tmp_path):
    recs = _produce(tmp_path, """
        from other import helper, Thing
        def main(): return helper(Thing())
    """, other="""
        class Thing: pass
        def helper(x): return x
    """)
    h = _one(recs, "helper")
    assert (h["target"], h["name"], h["proven"]) == ("user_function", "helper", True)
    assert h["def"] == {"module": "other.py", "line": 3, "col": 0}
    t = _one(recs, "Thing")
    assert (t["target"], t["def"]["module"], t["proven"]) == ("user_class", "other.py", True)


def test_an_import_from_a_library_is_imported_not_a_user_function(tmp_path):
    recs = _produce(tmp_path, """
        import math
        import os.path as osp
        from math import sqrt
        def main(): return sqrt(math.sin(1)) + osp.getsize("x")
    """)
    s = _one(recs, "sqrt")
    assert (s["target"], s["name"], s["proven"]) == ("imported", "math.sqrt", True)
    m = _one(recs, "math.sin")
    assert (m["target"], m["name"], m["proven"]) == ("imported", "math.sin", True)
    assert _one(recs, "osp.getsize")["name"] == "os.path.getsize"


def test_a_rebound_import_is_unproven(tmp_path):
    recs = _produce(tmp_path, """
        from math import sqrt
        sqrt = abs
        def main(): return sqrt(4)
    """)
    assert _one(recs, "sqrt")["proven"] is False


def test_a_builtin_is_proven_only_when_nothing_binds_the_name(tmp_path):
    recs = _produce(tmp_path, """
        def main(xs): return len(xs) + abs(1)
    """)
    r = _one(recs, "len")
    assert (r["target"], r["name"], r["proven"]) == ("builtin", "len", True)


def test_a_shadowed_builtin_is_not_a_builtin(tmp_path):
    recs = _produce(tmp_path, """
        def len(x): return 7
        def main(xs): return len(xs)
    """)
    r = _one(recs, "len", "main")
    assert (r["target"], r["name"]) == ("user_function", "len")


def test_a_builtin_rebound_by_assignment_is_unknown(tmp_path):
    recs = _produce(tmp_path, """
        len = lambda x: 7
        def main(xs): return len(xs)
    """)
    r = _one(recs, "len", "main")
    assert r["target"] == "unknown" and r["proven"] is False


def test_a_parameter_or_local_that_shadows_a_global_is_unknown(tmp_path):
    recs = _produce(tmp_path, """
        class Canvas: pass
        def make(Canvas):
            return Canvas()
        def local():
            Canvas = int
            return Canvas()
        def fine():
            return Canvas()
        def lam():
            return (lambda Canvas: Canvas())(int)
    """)
    assert _one(recs, "Canvas", "make")["target"] == "unknown"
    assert "bound in a function scope" in _one(recs, "Canvas", "make")["reason"]
    assert _one(recs, "Canvas", "local")["target"] == "unknown"
    assert _one(recs, "Canvas", "fine")["proven"] is True
    assert _one(recs, "Canvas", "lam")["target"] == "unknown"


def test_a_global_declaration_reaches_the_module_binding(tmp_path):
    recs = _produce(tmp_path, """
        class Canvas: pass
        def f():
            global Canvas
            return Canvas()
    """)
    # ``global Canvas`` with no store: the module binding answers
    r = _one(recs, "Canvas")
    assert r["target"] == "user_class" and r["proven"] is True


def test_a_comprehension_variable_shadows_inside_the_comprehension_only(tmp_path):
    recs = _produce(tmp_path, """
        class Canvas: pass
        def f(makers):
            a = Canvas()
            b = [Canvas() for Canvas in makers]
            return a, b
    """)
    kinds = sorted(r["target"] for r in recs if r["callee"] == "Canvas")
    assert kinds == ["unknown", "user_class"]
    # two sites spell the callee alike in one scope and resolve differently: the index must not pick one
    idx = call_target_index(recs)
    amb = idx[("f", "Canvas")]
    assert amb["target"] == "unknown" and amb["proven"] is False and "alike" in amb["reason"]


def test_the_comprehension_iterable_is_evaluated_outside_its_scope(tmp_path):
    recs = _produce(tmp_path, """
        class Canvas: pass
        def f():
            return [x for x in Canvas()]
    """)
    assert _one(recs, "Canvas")["target"] == "user_class"


def test_a_class_scope_name_is_visible_only_in_the_class_body(tmp_path):
    recs = _produce(tmp_path, """
        def helper(): return 1
        class K:
            def helper(self): return 2
            def m(self): return helper()
    """)
    assert _one(recs, "helper", "K.m")["target"] == "user_function"


def test_decorated_defs_and_classes_are_unknown(tmp_path):
    recs = _produce(tmp_path, """
        import functools
        @functools.lru_cache
        def f(x): return x
        @functools.total_ordering
        class C: pass
        def main(): return f(1), C()
    """)
    assert _one(recs, "f", "main")["target"] == "unknown" and "decorated" in _one(recs, "f", "main")["reason"]
    assert _one(recs, "C", "main")["target"] == "unknown"


def test_an_assignment_bound_callee_is_unknown(tmp_path):
    recs = _produce(tmp_path, """
        g = lambda x: x
        def main(): return g(1)
    """)
    r = _one(recs, "g")
    assert r["target"] == "unknown" and r["proven"] is False


def test_scopes_are_qualified_names(tmp_path):
    recs = _produce(tmp_path, """
        def helper(): return 1
        class A:
            def m(self):
                def inner(): return helper()
                return inner()
        if __name__ == "__main__":
            A().m()
    """)
    assert _one(recs, "helper")["scope"] == "A.m.inner"
    assert _one(recs, "inner")["scope"] == "A.m"
    assert _one(recs, "A")["scope"] == "<main>"
    assert _one(recs, "A().m")["scope"] == "<main>"


def test_a_method_through_the_class_name_resolves_along_the_single_inheritance_chain(tmp_path):
    recs = _produce(tmp_path, """
        class Base:
            def __init__(self, v): self.v = v
            @staticmethod
            def make(): return 1
        class Child(Base):
            def __init__(self, v):
                Base.__init__(self, v)
                super().__init__(v)
        def main(): return Child.make()
    """)
    base_init = _one(recs, "Base.__init__")
    assert (base_init["target"], base_init["name"], base_init["proven"]) == ("user_method", "Base.__init__", True)
    assert base_init["def"]["line"] == 3
    sup = _one(recs, "super().__init__")
    assert (sup["target"], sup["name"], sup["scope"]) == ("user_method", "Base.__init__", "Child.__init__")
    inherited = _one(recs, "Child.make")
    assert (inherited["target"], inherited["name"]) == ("user_method", "Base.make")


def test_a_method_on_a_receiver_of_unknown_type_is_dynamic(tmp_path):
    recs = _produce(tmp_path, """
        class A:
            def m(self): return self.n()
            def n(self): return 1
        def main(a): return a.m()
    """)
    for callee in ("self.n", "a.m"):
        r = _one(recs, callee)
        assert (r["target"], r["dispatch"], r["proven"]) == ("unknown", "dynamic", False)


def test_a_class_attribute_that_is_not_a_def_is_not_a_method(tmp_path):
    recs = _produce(tmp_path, """
        class A:
            m = 5
        class B:
            def n(self): return 1
            n = 3
        def main(): return A.m() + B.n()
    """)
    assert _one(recs, "A.m")["target"] == "unknown"
    assert _one(recs, "B.n")["target"] == "unknown"


def test_multiple_inheritance_or_a_missing_method_is_unknown(tmp_path):
    recs = _produce(tmp_path, """
        class A:
            def a(self): return 1
        class B:
            def b(self): return 1
        class C(A, B): pass
        def main(): return C.a() + A.zzz()
    """)
    assert _one(recs, "C.a")["target"] == "unknown"
    assert _one(recs, "A.zzz")["target"] == "unknown"


def test_super_outside_a_method_or_with_two_bases_is_unknown(tmp_path):
    recs = _produce(tmp_path, """
        class A:
            def m(self): return 1
        class B:
            def m(self): return 2
        class C(A, B):
            def m(self): return super().m()
        def f(): return super().m()
    """)
    assert _one(recs, "super().m", "C.m")["target"] == "unknown"
    assert _one(recs, "super().m", "f")["target"] == "unknown"


def test_a_callee_that_is_not_a_name_or_attribute_is_unknown(tmp_path):
    recs = _produce(tmp_path, """
        def make(): return lambda: 1
        def main(fs): return make()() + fs[0]()
    """)
    assert _one(recs, "make()")["target"] == "unknown"
    assert _one(recs, "fs[0]")["target"] == "unknown"


def test_an_unreadable_entry_produces_nothing(tmp_path):
    assert call_targets(str(tmp_path / "missing.py"), []) == []


def test_the_index_maps_scope_and_callee_to_the_record(tmp_path):
    recs = _produce(tmp_path, BASIC)
    idx = call_target_index(recs)
    assert idx[("main", "Canvas")]["target"] == "user_class"
    assert idx[("<main>", "main")]["target"] == "user_function"
    assert ("main", "nothing") not in idx


def test_a_call_chain_gives_every_call_its_own_extent(tmp_path):
    # ``a.f().g()`` starts one position for both calls: only the extent tells them apart
    recs = _produce(tmp_path, """
        class A:
            def f(self): return self
            def g(self): return 1
        def main(): return A().f().g()
    """)
    chain = [r for r in recs if r["line"] == 5 and r["scope"] == "main"]
    assert len(chain) == 3
    assert len({(r["line"], r["col"], r["end_line"], r["end_col"]) for r in chain}) == 3


# --- the type of a receiver variable ([internal ref]) ----------------------------------------------------------------

RECEIVER = """
    class Richards:
        def run(self, n):
            return n

    def helper():
        return 1
"""


def _method(recs, callee, scope=None):
    r = _one(recs, callee, scope)
    return r["target"], r["name"], r["proven"], r.get("receiver")


def test_a_module_level_receiver_under_the_main_guard_is_typed_from_its_one_binding(tmp_path):
    recs = _produce(tmp_path, RECEIVER + """
    if __name__ == "__main__":
        richard = Richards()
        richard.run(3)
    """)
    r = _one(recs, "richard.run")
    assert (r["target"], r["name"], r["proven"], r["dispatch"]) == ("user_method", "Richards.run", True, "static")
    assert r["receiver"] == {"name": "richard", "class": "Richards", "line": 10}
    assert r["def"]["line"] == 3 and "one binding" in r["reason"]


def test_a_function_local_receiver_and_a_read_from_a_nested_function_are_typed(tmp_path):
    recs = _produce(tmp_path, RECEIVER + """
    def main():
        r = Richards()
        def inner():
            return r.run(1)
        return r.run(2) + inner()
    """)
    assert _method(recs, "r.run", "main")[:3] == ("user_method", "Richards.run", True)
    assert _method(recs, "r.run", "main.inner")[:3] == ("user_method", "Richards.run", True)


def test_a_module_level_receiver_is_typed_inside_a_function_that_reads_it(tmp_path):
    recs = _produce(tmp_path, RECEIVER + """
    richard = Richards()
    def go():
        return richard.run(1)
    """)
    assert _method(recs, "richard.run", "go")[:3] == ("user_method", "Richards.run", True)


@pytest.mark.parametrize("why, body", [
    ("two bindings", "r = Richards()\n    r = Richards()\n    r.run(1)"),
    ("a rebinding in a branch", "r = Richards()\n    if helper():\n        r = Other()\n    r.run(1)"),
    ("a loop target", "r = Richards()\n    for r in []:\n        pass\n    r.run(1)"),
    ("a with target", "r = Richards()\n    with open('f') as r:\n        pass\n    r.run(1)"),
    ("a deletion", "r = Richards()\n    r.run(1)\n    del r"),
    ("an augmented assignment", "r = Richards()\n    r += 1\n    r.run(1)"),
    ("a walrus", "r = Richards()\n    (r := Other())\n    r.run(1)"),
    ("an import", "r = Richards()\n    import os as r\n    r.run(1)"),
    ("a tuple binding", "r, s = Richards(), 1\n    r.run(1)"),
    ("an annotated binding next to the assignment", "r = Richards()\n    r: int = 3\n    r.run(1)"),
    ("a binding by something that is not a class call", "r = helper()\n    r.run(1)"),
    ("a binding by a method call", "r = Richards().run(1)\n    r.run(2)"),
    ("a binding by an attribute", "r = Other.make()\n    r.run(1)"),
    ("a binding by a call of a function", "r = make()\n    r.run(1)"),
])
def test_a_receiver_with_anything_but_one_class_construction_binding_is_not_typed(tmp_path, why, body):
    recs = _produce(tmp_path, textwrap.dedent(RECEIVER) + textwrap.dedent("""
        class Other:
            @staticmethod
            def make():
                return 1
        def make():
            return Richards()
        def main():
        """) + textwrap.indent(body.replace("\n    ", "\n"), "    ") + "\n")
    got = _method(recs, "r.run", "main")
    assert got[0] == "unknown" and got[2] is False, (why, got)


def test_a_parameter_receiver_is_not_typed_and_a_class_body_binding_neither(tmp_path):
    recs = _produce(tmp_path, RECEIVER + """
    def main(r):
        return r.run(1)
    class K:
        r = Richards()
        v = r.run(1)
    """)
    assert _method(recs, "r.run", "main")[0] == "unknown"
    assert _method(recs, "r.run", "K")[0] == "unknown"


def test_a_global_or_nonlocal_declaration_in_a_nested_scope_hides_the_binding(tmp_path):
    recs = _produce(tmp_path, RECEIVER + """
    richard = Richards()
    def swap():
        global richard
        richard = Richards()
    def go():
        return richard.run(1)
    def outer():
        r = Richards()
        def mutate():
            nonlocal r
            r = Richards()
        return r.run(1)
    """)
    assert _method(recs, "richard.run", "go")[0] == "unknown"
    assert _method(recs, "r.run", "outer")[0] == "unknown"


def test_a_module_receiver_stored_into_from_a_function_or_through_the_namespace_is_not_typed(tmp_path):
    stored = _produce(tmp_path, RECEIVER + """
    richard = Richards()
    def swap():
        global richard
        richard = None
    def go():
        return richard.run(1)
    """)
    assert _method(stored, "richard.run", "go")[0] == "unknown"
    ns = _produce(tmp_path, RECEIVER + """
    richard = Richards()
    globals()['richard'] = None
    def go():
        return richard.run(1)
    """)
    assert _method(ns, "richard.run", "go")[0] == "unknown"


def test_a_receiver_of_an_unproven_or_shadowed_class_is_not_typed(tmp_path):
    unproven = _produce(tmp_path, RECEIVER + """
    Richards = Other = 3
    richard = Richards()
    richard.run(1)
    """)
    got = _method(unproven, "richard.run")
    assert got[0] == "user_method" and got[2] is False       # the declared target, never proven
    shadowed = _produce(tmp_path, RECEIVER + """
    def main(Richards):
        r = Richards()
        return r.run(1)
    """)
    assert _method(shadowed, "r.run", "main")[0] == "unknown"


def test_a_typed_receiver_follows_the_class_chain_and_names_the_defining_class(tmp_path):
    recs = _produce(tmp_path, """
        class Base:
            def run(self, n):
                return n
        class Child(Base):
            pass
        def main():
            c = Child()
            return c.run(1)
    """)
    got = _method(recs, "c.run", "main")
    assert got[:3] == ("user_method", "Base.run", True) and got[3]["class"] == "Child"


def test_a_typed_receiver_whose_method_is_missing_or_decorated_gets_no_method(tmp_path):
    recs = _produce(tmp_path, """
        import functools
        class A:
            @functools.lru_cache
            def cached(self):
                return 1
        def main():
            a = A()
            return a.cached() + a.missing()
    """)
    assert _method(recs, "a.cached", "main")[0] == "unknown"
    assert _method(recs, "a.missing", "main")[0] == "unknown"


def test_a_receiver_in_an_incomplete_graph_is_not_typed(tmp_path):
    recs = _produce(tmp_path, RECEIVER + """
    import builtins
    richard = Richards()
    richard.run(1)
    """)
    assert _method(recs, "richard.run")[0] == "unknown"
