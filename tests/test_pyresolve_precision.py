"""Closed receivers recover dead debug methods without pruning implicit hooks."""
import ast
import textwrap

import pytest

from tharkun.solve.census import site_census
from tharkun.solve.determinism import determinism_census
from tharkun.solve.pyresolve import Resolver, module_attrs
from tharkun.solve.contracts import closed_call_hooks


DEAD = "class C:\n    def check(self):\n        assert False\n"


def resolver(source):
    return Resolver(ast.parse(source), ("math", "random", "io", "_sv_verifier"))


def test_verifier_check_type_is_available_through_module_and_from_imports():
    assert "check_type" in module_attrs("_sv_verifier")
    imported = resolver("from _sv_verifier import check_type as check\n")
    assert imported.imports.names["check"] == ("_sv_verifier", "check_type")


@pytest.mark.parametrize("tail", [
    "C()",
    "xs = []; xs.append(C()); xs.pop()",
    "s = set([C()]); s.copy(); s.clear()",
    "x = {'x': C()}; x.copy(); x.pop('x')",
    "import math\nx = math.sqrt(4); y = math.log(x)",
    "import random\nrandom.seed(1); random.randrange(5); random.random()",
    "from _sv_verifier import nondet_int\nx = abs(nondet_int())",
    "def pick(x): return x\nx = pick(C()); xs = [x]; xs.pop()",
    "f = [].append\nf(C())",  # escaping native callable still has its contract
])
def test_unreferenced_ordinary_method_is_dead_in_closed_program(tail):
    # [internal ref]: an unreferenced def's HEADER (the ``def`` statement itself, run at
    # class-definition time to create the function object -- decorators, defaults
    # and annotations included) stays live.  Only its BODY statements, which never
    # execute because the method is never called, are dead.
    source = DEAD + tail + "\n"
    r = resolver(source)
    assert r._closed_ordinary_hooks() is not None
    method = next(n for n in ast.walk(r.tree) if isinstance(n, ast.FunctionDef) and n.name == "check")
    assert r.live(method)
    assert method.body and all(not r.live(stmt) for stmt in method.body)
    assert site_census(source, ["AssertionError"]).no_site


def test_unreferenced_method_body_is_pruned_from_the_error_site_census():
    # [internal ref] end-to-end: the census must not surface a site for the ``assert False``
    # sitting in the dead body of an unreferenced, closed-receiver method -- this is
    # what protects #55's +10 go gain (a whole never-called debug/check method no
    # longer forces ``! uncaught(AssertionError)`` open).  A raising statement placed
    # directly in ``check``'s (dead) body must not appear in ``census.sites``.
    source = DEAD + "C()\n"
    census = site_census(source, ["AssertionError"])
    assert census.no_site
    assert census.sites == ()
    dead_assert_line = DEAD.splitlines()[2]  # "        assert False"
    assert dead_assert_line.strip() == "assert False"
    assert not any(line == 3 for line, _col, _what in census.sites)


def test_unreferenced_method_default_argument_is_still_reported():
    # [internal ref] negative control: a default-argument expression is part of the def's
    # HEADER (``FunctionDef.args``, not ``FunctionDef.body``), so it runs at
    # class-definition time even though the method itself is never called and is
    # otherwise a "closed, unreferenced ordinary method" candidate for dead-code
    # pruning (no decorator, no reference by name).  The census must still report
    # it, and it must still be live -- unlike the body statement right after it.
    source = "class C:\n    def check(self, x=1/0):\n        assert False\n" + "C()\n"
    with pytest.raises(ZeroDivisionError):
        exec(source, {})
    r = resolver(source)
    method = next(n for n in ast.walk(r.tree) if isinstance(n, ast.FunctionDef) and n.name == "check")
    assert r.live(method)
    default = method.args.defaults[0]
    assert r.live(default)
    assert not r.live(method.body[0])  # the "assert False" right after it stays dead
    census = site_census(source, ["ZeroDivisionError"])
    assert not census.no_site
    assert census.sites and census.sites[0][0] == 2


@pytest.mark.parametrize("tail", [
    "import io\nx = io.StringIO()",  # external class, even if apparently harmless
    "import random\nx = random.Random()",
    "import random\nx = random.choice([C()])",  # absent callback contract
    "getattr(C(), 'che' + 'ck')()",  # computed reflection
    "class D(C): pass",  # inheritance/external protocols not admitted
    "class D(list): pass",  # never apply exact-native contracts to subclasses
    "C.mro()",  # source class values also expose native type attributes
    "x = '{0.che' + 'ck}'; x.format(C())",  # hidden attribute lookup
    "f = lambda x: x\n@f\nclass D: pass",
    "x = (v for v in [C()])",  # generator send/throw/close protocols deferred
    "x = __loader__",  # runtime-injected value, not a source-local class
    "try:\n    1/0\nexcept Exception as e:\n    pass",
    "class D:\n    def method(self, x: C): pass",  # annotation scope is separate
])
def test_unknown_provenance_keeps_all_methods(tail):
    source = DEAD + tail + "\n"
    try:
        r = resolver(source)
    except Exception as error:
        # Reflection can also be rejected before the liveness refinement.
        from tharkun.solve.pyresolve import Top
        assert isinstance(error, Top)
        return
    assert r._closed_ordinary_hooks() is None
    assert not site_census(source, ["AssertionError"]).no_site


@pytest.mark.parametrize("operation", ["dict(**C())", "{**C()}", "dict(C())", "{}.update(C())"])
def test_closed_mapping_hook_is_still_live(operation):
    source = ("class C:\n    def keys(self):\n        assert False\n        return []\n"
              "    def __getitem__(self, k): return 0\n" + operation + "\n")
    assert resolver(source)._closed_ordinary_hooks() == {"keys"}
    with pytest.raises(AssertionError):
        exec(source, {})
    assert not site_census(source, ["AssertionError"]).no_site


@pytest.mark.parametrize("source", [
    "class C:\n    def pop(self): assert False\nC().pop()",
    "class C:\n    def __len__(self):\n        assert False\n        return 0\nlen(C())",
    "class C:\n    def __float__(self):\n        assert False\n        return 0.0\nfloat(C())",
    "class C:\n    def __iter__(self):\n        assert False\n        return self\nlist(C())",
    "class C:\n    def check(self): assert False\nf = C().check\nf()",
    "class C:\n    def check(self): assert False\n    def __repr__(self):\n        self.check()\n        return ''\nrepr(C())",
])
def test_user_receivers_and_dunders_remain_possible(source):
    assert resolver(source)._closed_ordinary_hooks() is not None
    with pytest.raises(AssertionError):
        exec(source, {})
    assert not site_census(source, ["AssertionError"]).no_site


def test_dead_debug_loop_does_not_block_determinism():
    source = textwrap.dedent("""\
        import random
        class Board:
            def check(self):
                members = set([self])
                for member in members:
                    assert False
            def run(self):
                xs = []
                xs.append(random.randrange(5))
                return xs.pop()
        random.seed(1)
        Board().run()
    """)
    assert determinism_census(source).ok
    assert site_census(source, ["AssertionError"]).no_site


def test_missing_contract_is_unknown():
    assert closed_call_hooks("future.method") is None
    assert closed_call_hooks("dict.update") == frozenset({"keys"})
    assert closed_call_hooks("list.pop") == frozenset()


def test_random_seed_spoofed_class_has_ordinary_callback():
    import random

    class Spoof:
        def __getattribute__(self, name):
            if name == "__class__":
                return str
            return object.__getattribute__(self, name)

        def encode(self):
            raise AssertionError("encode reached")

    with pytest.raises(AssertionError, match="encode reached"):
        random.seed(Spoof())
    assert closed_call_hooks("random.seed") == frozenset({"encode", "decode"})
