"""[internal ref] follow-up: ``nondet_str`` draws used only for truthiness and as ``open()``'s own
path argument (``drawflow.py``'s ``str_truthy`` classification), including INTERPROCEDURAL
propagation through a resolvable function/method call chain (``_Planner.str_restricted``,
``_resolve_str_callee``, ``_recv_class``) -- bm_raytrace_nondet.py's own shape
(``args_filename`` -> ``bench_raytrace``'s ``filename`` parameter -> ``Canvas.write_ppm``'s
own ``filename`` parameter -> ``open()``).
"""

import textwrap

from tharkun.solve.drawflow import plan_draws


def _plan(src):
    return plan_draws(textwrap.dedent(src))


# -------------------------------------------------------------- single-hop str_truthy

def test_truthiness_and_open_path_same_variable_is_accepted():
    plan = _plan("""
        import _sv_verifier
        name = _sv_verifier.nondet_str()
        if name:
            with open(name, "wb") as fp:
                fp.write(b"x")
    """)
    assert plan.ok, plan.reason
    assert any(d.kind == "str_truthy" for d in plan.draws)


def test_not_operator_is_a_truthiness_position():
    plan = _plan("""
        import _sv_verifier
        name = _sv_verifier.nondet_str()
        if not name:
            print("empty")
        else:
            with open(name, "wb") as fp:
                fp.write(b"x")
    """)
    assert plan.ok, plan.reason


def test_bool_call_is_a_truthiness_position():
    plan = _plan("""
        import _sv_verifier
        name = _sv_verifier.nondet_str()
        ok = bool(name)
        if ok:
            with open(name, "wb") as fp:
                fp.write(b"x")
    """)
    assert plan.ok, plan.reason


def test_concatenation_refuses():
    plan = _plan("""
        import _sv_verifier
        name = _sv_verifier.nondet_str()
        print(name + ".txt")
    """)
    assert not plan.ok
    assert "str draw" in plan.reason


def test_used_as_second_open_argument_refuses():
    plan = _plan("""
        import _sv_verifier
        mode = _sv_verifier.nondet_str()
        with open("out.txt", mode) as fp:
            fp.write(b"x")
    """)
    assert not plan.ok


def test_rebinding_the_draw_variable_refuses():
    plan = _plan("""
        import _sv_verifier
        name = _sv_verifier.nondet_str()
        if name:
            name = name + "x"
            with open(name, "wb") as fp:
                fp.write(b"x")
    """)
    assert not plan.ok


# ----------------------------------------------------- interprocedural propagation

def test_propagation_through_a_plain_function_is_accepted():
    plan = _plan("""
        import _sv_verifier

        def write_it(path):
            if path:
                with open(path, "wb") as fp:
                    fp.write(b"x")

        name = _sv_verifier.nondet_str()
        write_it(name)
    """)
    assert plan.ok, plan.reason


def test_raytrace_shape_function_then_method_is_accepted():
    """bm_raytrace_nondet.py's exact shape: a str draw flows into a function's own
    parameter, which is checked for truthiness, then passed to a METHOD call on a
    receiver whose single, unambiguous class this program defines plainly."""
    plan = _plan("""
        import _sv_verifier

        class Canvas(object):
            def __init__(self, w):
                self.w = w

            def write_ppm(self, filename):
                with open(filename, "wb") as fp:
                    fp.write(b"x")

        def bench(filename):
            c = Canvas(3)
            if filename:
                c.write_ppm(filename)

        args_filename = _sv_verifier.nondet_str()
        bench(args_filename)
    """)
    assert plan.ok, plan.reason


def test_propagated_value_used_for_more_than_truthiness_in_the_callee_refuses():
    plan = _plan("""
        import _sv_verifier

        def write_it(path):
            print(path + ".bak")

        name = _sv_verifier.nondet_str()
        write_it(name)
    """)
    assert not plan.ok


def test_propagation_to_an_unresolvable_callee_refuses():
    # a class with real inheritance -- never guessed
    plan = _plan("""
        import _sv_verifier

        class Base(object):
            pass

        class Canvas(Base):
            def write_ppm(self, filename):
                with open(filename, "wb") as fp:
                    fp.write(b"x")

        def bench(filename):
            c = Canvas()
            if filename:
                c.write_ppm(filename)

        args_filename = _sv_verifier.nondet_str()
        bench(args_filename)
    """)
    assert not plan.ok


def test_propagation_to_a_rebound_receiver_refuses():
    plan = _plan("""
        import _sv_verifier

        class Canvas(object):
            def write_ppm(self, filename):
                with open(filename, "wb") as fp:
                    fp.write(b"x")

        def bench(filename):
            c = Canvas()
            c = Canvas()
            if filename:
                c.write_ppm(filename)

        args_filename = _sv_verifier.nondet_str()
        bench(args_filename)
    """)
    assert not plan.ok


def test_propagation_to_a_class_with_getattr_refuses():
    plan = _plan("""
        import _sv_verifier

        class Canvas(object):
            def __getattr__(self, name):
                return None

            def write_ppm(self, filename):
                with open(filename, "wb") as fp:
                    fp.write(b"x")

        def bench(filename):
            c = Canvas()
            if filename:
                c.write_ppm(filename)

        args_filename = _sv_verifier.nondet_str()
        bench(args_filename)
    """)
    assert not plan.ok


def test_propagation_through_starargs_refuses():
    plan = _plan("""
        import _sv_verifier

        def inner(path):
            if path:
                with open(path, "wb") as fp:
                    fp.write(b"x")

        def bench(*args):
            inner(*args)

        name = _sv_verifier.nondet_str()
        bench(name)
    """)
    assert not plan.ok


def test_a_three_hop_chain_is_accepted():
    plan = _plan("""
        import _sv_verifier

        def f3(x):
            if x:
                with open(x, "wb") as fp:
                    fp.write(b"x")

        def f2(x):
            f3(x)

        def f1(x):
            f2(x)

        name = _sv_verifier.nondet_str()
        f1(name)
    """)
    assert plan.ok, plan.reason


def test_an_excessively_long_chain_refuses_rather_than_recursing_forever():
    plan = _plan("""
        import _sv_verifier

        def f7(x):
            if x:
                with open(x, "wb") as fp:
                    fp.write(b"x")

        def f6(x):
            f7(x)

        def f5(x):
            f6(x)

        def f4(x):
            f5(x)

        def f3(x):
            f4(x)

        def f2(x):
            f3(x)

        def f1(x):
            f2(x)

        name = _sv_verifier.nondet_str()
        f1(name)
    """)
    assert not plan.ok
    assert "too many calls" in plan.reason


def test_mutual_recursion_terminates_and_refuses():
    """Two functions that keep passing the draw back and forth: str_seen's memoization
    (mirrors trip_variable's own ``seen`` set) must terminate this, not loop forever --
    and the actual OTHER use (a concatenation) still refuses the whole plan."""
    plan = _plan("""
        import _sv_verifier

        def a(x):
            if x:
                b(x)
                print(x + "y")

        def b(x):
            a(x)

        name = _sv_verifier.nondet_str()
        a(name)
    """)
    assert not plan.ok


# --------------------------------------------------- MG's PR 108 review: method rebinding
#
# _resolve_str_callee used to trust a resolved method's CLASS-BODY definition even though
# ordinary Python can replace what a later `c.run(...)` actually invokes -- `c.run = bad`,
# `C.run = bad2`, `self.run = bad` (in `__init__` or anywhere else), or `setattr(c, 'run',
# bad)` -- none of which are reflection (no __getattr__/__setattr__ override) and none
# rebind the RECEIVER (_recv_class's own "bound exactly once" check stays satisfied). MG's
# attacks ([host]:[local path]): `bad` tests a condition with no
# literal for a search to find (len(s)==6 and s[1]==chr(113) and s[4]==chr(90)) before
# indexing an empty list -- PR 108 (pre-fix) said TRUE for data-lookup; natively an
# IndexError is reachable.

def test_instance_attribute_reassignment_refuses():
    """MG's n01_instance_attr_shadow.py, ``c.run(s)`` -- a DIRECT method call on the str
    draw's own variable, no propagation hop needed to hit the bug."""
    plan = _plan("""
        import _sv_verifier

        class C:
            def run(self, s):
                if s:
                    pass

        def bad(s):
            if len(s) == 6 and s[1] == chr(113) and s[4] == chr(90):
                [][0]

        def main():
            s = _sv_verifier.nondet_str()
            c = C()
            c.run = bad
            c.run(s)

        main()
    """)
    assert not plan.ok


def test_class_attribute_reassignment_refuses():
    """MG's n02_class_attr_rebind.py: ``C.run = bad2`` at module scope."""
    plan = _plan("""
        import _sv_verifier

        class C:
            def run(self, s):
                if s:
                    pass

        def bad2(self, s):
            if len(s) == 6 and s[1] == chr(113) and s[4] == chr(90):
                [][0]

        C.run = bad2

        def main():
            s = _sv_verifier.nondet_str()
            c = C()
            c.run(s)

        main()
    """)
    assert not plan.ok


def test_init_setting_the_same_named_attribute_refuses():
    """MG's n03_init_sets_attr.py: ``__init__`` shadows its own class's ``run`` method
    with an instance attribute of the same name."""
    plan = _plan("""
        import _sv_verifier

        def bad(s):
            if len(s) == 6 and s[1] == chr(113) and s[4] == chr(90):
                [][0]

        class C:
            def __init__(self):
                self.run = bad

            def run(self, s):
                if s:
                    pass

        def main():
            s = _sv_verifier.nondet_str()
            c = C()
            c.run(s)

        main()
    """)
    assert not plan.ok


def test_setattr_anywhere_in_the_program_refuses():
    """MG's n05_setattr.py: ``setattr(c, 'run', bad)`` -- no ordinary ``ast.Attribute``
    store at all, caught instead by ``setattr`` now being in ``FORBIDDEN_NAMES``."""
    plan = _plan("""
        import _sv_verifier

        class C:
            def run(self, s):
                if s:
                    pass

        def bad(s):
            if len(s) == 6 and s[1] == chr(113) and s[4] == chr(90):
                [][0]

        def main():
            s = _sv_verifier.nondet_str()
            c = C()
            setattr(c, "run", bad)
            c.run(s)

        main()
    """)
    assert not plan.ok


def test_a_method_never_reassigned_anywhere_is_still_accepted():
    # the fix must not be overly strict: an ordinary, never-rebound method still works.
    plan = _plan("""
        import _sv_verifier

        class Canvas(object):
            def write_ppm(self, filename):
                with open(filename, "wb") as fp:
                    fp.write(b"x")

        def bench(filename):
            c = Canvas()
            if filename:
                c.write_ppm(filename)

        args_filename = _sv_verifier.nondet_str()
        bench(args_filename)
    """)
    assert plan.ok, plan.reason


def test_an_unrelated_attribute_reassignment_does_not_taint_a_different_name():
    # only the SPECIFIC attribute name being trusted as a method is checked -- rebinding
    # some OTHER attribute anywhere in the program must not refuse this plan.
    plan = _plan("""
        import _sv_verifier

        class Canvas(object):
            def write_ppm(self, filename):
                with open(filename, "wb") as fp:
                    fp.write(b"x")

        class Other(object):
            def __init__(self):
                self.unrelated = 1

        def bench(filename):
            c = Canvas()
            if filename:
                c.write_ppm(filename)

        args_filename = _sv_verifier.nondet_str()
        bench(args_filename)
    """)
    assert plan.ok, plan.reason


# -------------------------------- MG's PR 108 second review: n07 (object.__setattr__),
# -------------------------------- n11 (a class-body rebinding after the def)

def test_class_body_rebinding_after_the_def_refuses():
    """MG's n11_class_body_alias.py: ``def run(self, s): ...`` followed, later in the SAME
    class body, by ``run = staticmethod(bad)`` -- the class ends up with ``bad``, not the
    def; counting only ``FunctionDef`` nodes (the previous ``_class_method``) missed this
    plain rebinding entirely."""
    plan = _plan("""
        import _sv_verifier

        def bad(s):
            if len(s) == 6 and s[1] == chr(113) and s[4] == chr(90):
                [][0]

        class C:
            def run(self, s):
                if s:
                    pass
            run = staticmethod(bad)

        def main():
            s = _sv_verifier.nondet_str()
            c = C()
            c.run(s)

        main()
    """)
    assert not plan.ok


def test_class_body_rebinding_inside_an_if_refuses():
    # the rebinding does not have to be a bare top-level statement in the class body --
    # _bound_in_class_body still descends into an `if` at class scope.
    plan = _plan("""
        import _sv_verifier

        def bad(s):
            if len(s) == 6 and s[1] == chr(113) and s[4] == chr(90):
                [][0]

        class C:
            def run(self, s):
                if s:
                    pass
            if True:
                run = staticmethod(bad)

        def main():
            s = _sv_verifier.nondet_str()
            c = C()
            c.run(s)

        main()
    """)
    assert not plan.ok


def test_a_coincidentally_named_local_inside_the_method_does_not_taint_it():
    # _bound_in_class_body must NOT descend into the method's own body -- a local variable
    # there that happens to share the method's name is a different binding entirely.
    plan = _plan("""
        import _sv_verifier

        class Canvas:
            def write_ppm(self, filename):
                write_ppm = None
                with open(filename, "wb") as fp:
                    fp.write(b"x")

        def bench(filename):
            c = Canvas()
            if filename:
                c.write_ppm(filename)

        args_filename = _sv_verifier.nondet_str()
        bench(args_filename)
    """)
    assert plan.ok, plan.reason


def test_object_dunder_setattr_refuses():
    """MG's n07_object_setattr.py: ``object.__setattr__(c, 'run', bad)`` rebinds exactly
    like the plain ``setattr`` builtin, just spelled as an attribute access -- FORBIDDEN_
    NAMES's own ``setattr``/``delattr`` entries do not see an attribute access."""
    plan = _plan("""
        import _sv_verifier

        class C:
            def run(self, s):
                if s:
                    pass

        def bad(s):
            if len(s) == 6 and s[1] == chr(113) and s[4] == chr(90):
                [][0]

        def main():
            s = _sv_verifier.nondet_str()
            c = C()
            object.__setattr__(c, "run", bad)
            c.run(s)

        main()
    """)
    assert not plan.ok


def test_instance_dunder_setattr_refuses():
    plan = _plan("""
        import _sv_verifier

        class C:
            def run(self, s):
                if s:
                    pass

        def bad(s):
            if len(s) == 6 and s[1] == chr(113) and s[4] == chr(90):
                [][0]

        def main():
            s = _sv_verifier.nondet_str()
            c = C()
            c.__setattr__("run", bad)
            c.run(s)

        main()
    """)
    assert not plan.ok


def test_class_swap_via_dunder_class_refuses():
    """MG's n13_subclass_override.py: ``c.__class__ = D`` swaps ``c``'s ENTIRE class to
    one whose own ``run`` is ``bad`` -- after ``_recv_class`` already resolved ``c`` as a
    ``C`` from its own construction. The CLI happened to stay UNKNOWN by luck (no other
    check caught it), but the planner itself must refuse this, not rely on that luck."""
    plan = _plan("""
        import _sv_verifier

        def bad(s):
            if len(s) == 6 and s[1] == chr(113) and s[4] == chr(90):
                [][0]

        class C:
            def run(self, s):
                if s:
                    pass

        class D(C):
            run = staticmethod(bad)

        def main():
            s = _sv_verifier.nondet_str()
            c = C()
            c.__class__ = D
            c.run(s)

        main()
    """)
    assert not plan.ok
