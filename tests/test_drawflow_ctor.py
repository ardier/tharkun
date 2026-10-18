"""``plan_draws`` follows a trip value into a user class's ``__init__`` through the ``call_target`` fact ([internal ref],
consumer 2 of the fact; ``solve/drawflow.py::_Planner._ctor``, docs/SV.md "The call_target fact").

The stationarity proof is a PROOF, so this only widens what it follows when the fact says the callee IS that class
(``user_class``, ``proven``) and the class is a plain one with a single undecorated own ``__init__``; every other
shape stays the "unresolved callee" refusal it always was, and a refusal here names why.
"""
from __future__ import annotations

import textwrap

import pytest

from tharkun.solve.call_targets import call_targets
from tharkun.solve.drawflow import plan_draws
from tharkun.solve.module_namespace import module_constants

PLAIN = """
    import _sv_verifier

    class Canvas:
        def __init__(self, n, label):
            self.label = label
            for i in range(n):
                pass

    def make(count):
        return Canvas(count, "x")

    if __name__ == "__main__":
        n = abs(_sv_verifier.nondet_int())
        make(n)
"""


def plan(tmp_path, src: str, *, facts=True):
    src = textwrap.dedent(src)
    path = tmp_path / "task.py"
    path.write_text(src)
    records = call_targets(str(path), module_constants(str(path))) if facts else None
    return plan_draws(src, str(path), records)


def test_a_trip_value_is_followed_into_the_constructor_the_fact_resolves(tmp_path):
    p = plan(tmp_path, PLAIN)
    assert p.ok, p.reason
    assert [d.kind for d in p.draws] == ["trip"] and len(p.loops) == 1
    assert p.loops[0].dead == ("i",)


def test_a_keyword_argument_reaches_the_named_parameter_without_the_self_shift(tmp_path):
    p = plan(tmp_path, PLAIN.replace('Canvas(count, "x")', 'Canvas(label="x", n=count)'))
    assert p.ok, p.reason
    assert len(p.loops) == 1


def test_without_the_fact_the_constructor_is_an_unresolved_callee_as_before(tmp_path):
    p = plan(tmp_path, PLAIN, facts=False)
    assert not p.ok and "unresolved callee" in p.reason


def test_a_trip_value_the_constructor_reads_is_still_refused(tmp_path):
    p = plan(tmp_path, PLAIN.replace("self.label = label", "self.size = n * 2"))
    assert not p.ok and "n (a trip value) is read by" in p.reason


def with_class(tmp_path, definition: str, extra: str = ""):
    """A program whose ``Canvas`` is ``definition`` (it must define ``__init__(self, n, label)`` with the loop)."""
    return plan(tmp_path, f"""
        import _sv_verifier
        {definition}
        def make(count):
            return Canvas(count, "x")
        {extra}
        if __name__ == "__main__":
            n = abs(_sv_verifier.nondet_int())
            make(n)
    """)


BODY = """
            def __init__(self, n, label):
                for i in range(n):
                    pass
"""


@pytest.mark.parametrize("name, header, why", [
    ("a base class", "class Base:\n            pass\n        class Canvas(Base):", "a metaclass, a decorator, a base"),
    ("a metaclass", "class Canvas(metaclass=type):", "a metaclass, a decorator, a base"),
    ("a __new__", "class Canvas:\n            def __new__(cls, *a):\n                return object.__new__(cls)", "a metaclass, a decorator, a base"),
    ("a __setattr__", "class Canvas:\n            def __setattr__(self, k, v):\n                pass", "a metaclass, a decorator, a base"),
])
def test_a_class_that_could_redirect_construction_is_refused(tmp_path, name, header, why):
    p = with_class(tmp_path, header + BODY.replace("\n            def __init__", "\n            def __init__"))
    assert not p.ok and why in p.reason, p.reason


def test_a_decorated_class_is_not_the_class_the_fact_says(tmp_path):
    p = with_class(tmp_path, "def deco(c):\n            return c\n        @deco\n        class Canvas:" + BODY)
    assert not p.ok and "call_target is unknown" in p.reason, p.reason


def test_a_rebound_class_name_is_refused_as_unproven(tmp_path):
    p = with_class(tmp_path, "class Canvas:" + BODY, extra="class Other:\n            pass\n        Canvas = Other")
    assert not p.ok and "proven=False" in p.reason, p.reason


def test_a_constructor_replaced_by_an_attribute_store_is_refused(tmp_path):
    p = plan(tmp_path, PLAIN + """
    def bad(self, n, label):
        pass

    Canvas.__init__ = bad
    """)
    assert not p.ok
    assert "stored into elsewhere" in p.reason or "proven=False" in p.reason or "unresolved" in p.reason


def test_an_inherited_or_missing_init_is_refused(tmp_path):
    p = plan(tmp_path, PLAIN.replace("def __init__(self, n, label):", "def setup(self, n, label):"))
    assert not p.ok and "no single undecorated own __init__" in p.reason


def test_a_local_name_that_hides_the_class_is_not_the_class(tmp_path):
    p = plan(tmp_path, PLAIN.replace("def make(count):", "def make(count, Canvas=None):"))
    assert not p.ok


# --- a method on a receiver the fact types ([internal ref]) ------------------------------------------------------------

METHOD = """
    import _sv_verifier

    class Richards:
        def run(self, iterations):
            for i in range(iterations):
                pass
            return True

    if __name__ == "__main__":
        richard = Richards()
        richard.run(abs(_sv_verifier.nondet_int()))
"""


def test_a_trip_value_is_followed_into_a_method_of_the_typed_receiver(tmp_path):
    p = plan(tmp_path, METHOD)
    assert p.ok, p.reason
    assert [d.kind for d in p.draws] == ["trip"] and len(p.loops) == 1 and p.loops[0].dead == ("i",)


def test_the_method_form_needs_the_fact_without_it_the_callee_is_unresolved(tmp_path):
    p = plan(tmp_path, METHOD, facts=False)
    assert not p.ok and "unresolved callee" in p.reason


def test_a_keyword_argument_reaches_its_parameter_and_self_cannot_be_named(tmp_path):
    ok = plan(tmp_path, METHOD.replace("richard.run(abs(_sv_verifier.nondet_int()))",
                                        "richard.run(iterations=abs(_sv_verifier.nondet_int()))"))
    assert ok.ok, ok.reason
    bad = plan(tmp_path, METHOD.replace("richard.run(abs(_sv_verifier.nondet_int()))",
                                         "richard.run(self=abs(_sv_verifier.nondet_int()))"))
    assert not bad.ok and "unknown keyword" in bad.reason


def test_a_trip_value_the_method_reads_is_still_refused(tmp_path):
    p = plan(tmp_path, METHOD.replace("pass\n            return True", "x = iterations * 2\n            return True"))
    assert not p.ok and "iterations (a trip value) is read by" in p.reason


@pytest.mark.parametrize("name, change, why", [
    ("an attribute store binding the method", "richard.run = None\n", "some attribute store binds .run"),
    ("a class attribute store", "Richards.run = None\n", "some attribute store binds .run"),
    ("a rebound receiver", "richard = Richards()\n", "whose call_target is unknown"),
])
def test_a_method_that_could_be_replaced_or_a_receiver_that_could_change_is_refused(tmp_path, name, change, why):
    src = METHOD.replace('        richard = Richards()\n', f'        richard = Richards()\n        {change}')
    p = plan(tmp_path, src)
    assert not p.ok and why in p.reason, (name, p.reason)


def test_a_class_with_a_base_or_a_hook_or_a_decorated_method_is_refused(tmp_path):
    base = plan(tmp_path, METHOD.replace("class Richards:", "class Base:\n        pass\n\n    class Richards(Base):"))
    assert not base.ok and "has a metaclass, a decorator, a base or a hook" in base.reason
    hook = plan(tmp_path, METHOD.replace("        def run(self, iterations):", "        def __getattr__(self, n):\n            return 1\n\n        def run(self, iterations):"))
    assert not hook.ok and "has a metaclass, a decorator, a base or a hook" in hook.reason
    static = plan(tmp_path, METHOD.replace("        def run(self, iterations):", "        @staticmethod\n        def run(iterations):"))
    assert not static.ok and "single undecorated own instance method" in static.reason or "unresolved" in static.reason


def test_a_receiver_that_is_a_parameter_or_has_two_bindings_is_not_typed(tmp_path):
    param = plan(tmp_path, """
        import _sv_verifier
        class Richards:
            def run(self, iterations):
                for i in range(iterations):
                    pass
        def go(r):
            r.run(abs(_sv_verifier.nondet_int()))
        go(Richards())
    """)
    assert not param.ok and "whose call_target is unknown" in param.reason
    two = plan(tmp_path, METHOD.replace("        richard = Richards()\n", "        richard = Richards()\n        richard = Richards()\n"))
    assert not two.ok and "whose call_target is unknown" in two.reason


def test_a_builtin_callee_is_refused_by_name(tmp_path):
    p = plan(tmp_path, """
        import _sv_verifier
        SIZE = 1 + abs(_sv_verifier.nondet_int())
        def to_xy(pos):
            y, x = divmod(pos, SIZE)
            return x, y
        to_xy(3)
    """)
    assert not p.ok and "builtin divmod" in p.reason and "varies with the value" in p.reason, p.reason


def test_a_receiver_of_an_unproven_class_name_is_refused_with_the_declared_target(tmp_path):
    src = METHOD.replace('    if __name__ == "__main__":', "    class Other:\n        pass\n\n    Richards = Other\n\n    if __name__ == \"__main__\":")
    p = plan(tmp_path, src)
    assert not p.ok and "whose call_target is user_method (proven=False" in p.reason, p.reason
