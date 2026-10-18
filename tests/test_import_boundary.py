"""[internal ref]: the import boundary -- "read-and-recurse, not allow-by-name" (docs/SV.md).

Two gates share one predicate (``solve/extern.py``, wired into
``solve/pyresolve.py::Resolver``): an import that does not resolve via a stdlib allowlist may
resolve instead through a directory the PROGRAM ITSELF adds to ``sys.path`` (a literal,
static ``sys.path.insert(0, D)``), never anything ambient. Soundness conditions (a)-(g) are
cited by letter, matching ``extern.py``'s own module docstring and [internal ref].

Conditions (a)/(b)/(c)/(e) are ELIGIBILITY -- tested directly against ``Resolver`` (shared by
``census.py``/``determinism.py``): the file-finding logic is exactly the same regardless of
which allowlist/caller uses it, so a Resolver-level test covers both gates at once. Condition
(d) (imported code is program code) and (f) (a never-TRUE probe per condition, plus a real
crash negative control) and the is_string/is_full_string positive validation are tested
against ``absint.prove`` directly, since census.py's own MODULES table lacks "sys" and
determinism.py's SYS_OK excludes "sys.path" -- BOTH pre-existing, unrelated to this change --
so neither can ever fully certify a program that contains the required sys.path.insert
bootstrap itself; absint is the only gate that can demonstrate the full positive/negative
picture end-to-end (see this module's own docstring note on that, and the handoff).
"""

from __future__ import annotations

import ast
import os
import textwrap
from pathlib import Path

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.determinism import DET_MODULES
from tharkun.solve.pyresolve import Resolver, Top

CLASSES = ("TypeError", "AttributeError", "IndexError", "KeyError",
           "ZeroDivisionError", "FloatingPointError", "AssertionError")


def _src(body: str) -> str:
    return textwrap.dedent(body).lstrip("\n")


def _resolve(source: str, filename: str):
    """Build a Resolver over ``source`` (as the top-level program) against DET_MODULES,
    returning it, or the :class:`Top` it raised."""
    tree = ast.parse(source, filename)
    try:
        return Resolver(tree, DET_MODULES, filename=filename)
    except Top as e:
        return e


# =========================================================== (a)/(b): path literal-ness
def test_absolute_literal_path_resolves(tmp_path: Path):
    (tmp_path / "lib.py").write_text(_src("""
        def safe_len(x):
            return len(x)
    """))
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from lib import safe_len
    """)
    res = _resolve(prog, str(tmp_path / "harness.py"))
    assert isinstance(res, Resolver), res
    assert "safe_len" in res.imports.names


def test_file_dirname_abspath_joined_literal_recognised(tmp_path: Path):
    """Condition (a)'s second accepted form: a literal joined to
    os.path.dirname(os.path.abspath(__file__)). Tested against extern.py's own root-finding
    directly, not the full Resolver: DET_MODULES does not allow `import os` itself (a
    separate, pre-existing, deliberate restriction unrelated to this change) so a full
    top-level program combining "needs os to resolve" with "needs read-and-recurse to
    resolve" cannot fully resolve either way today -- this checks the (a) pattern-match
    itself resolves to the RIGHT directory, which is the part [internal ref] owns."""
    from tharkun.solve import extern

    (tmp_path / "libdir").mkdir()
    prog = _src("""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "libdir"))
    """)
    tree = ast.parse(prog, str(tmp_path / "harness.py"))
    snaps, final = extern.root_snapshots(tree, str(tmp_path / "harness.py"))
    assert final == [str(tmp_path / "libdir")]


@pytest.mark.parametrize("insert_expr", [
    '"relative/lib"',              # (a) probe: a relative literal
    "os.getcwd()",                 # (a) probe: a computed, non-__file__ expression
    'os.path.join(os.getcwd(), "lib")',
    '__file__ + "/../lib"',        # (a) probe: contains ".."
])
def test_a_non_literal_or_relative_insert_never_resolves(tmp_path: Path, insert_expr):
    (tmp_path / "lib.py").write_text("def f(x):\n    return x\n")
    prog = _src(f"""
        import sys, os
        sys.path.insert(0, {insert_expr})
        from lib import f
    """)
    res = _resolve(prog, str(tmp_path / "harness.py"))
    assert isinstance(res, Top), "must refuse, not resolve via an ineligible path form"


@pytest.mark.parametrize("wrapped", [
    "if True:\n    sys.path.insert(0, {D!r})\n",         # (b) probe: inside an if
    "def _setup():\n    sys.path.insert(0, {D!r})\n_setup()\n",  # inside a function
    "try:\n    sys.path.insert(0, {D!r})\nexcept Exception:\n    pass\n",  # inside a try
])
def test_b_a_conditional_or_nested_insert_never_resolves(tmp_path: Path, wrapped):
    (tmp_path / "lib.py").write_text("def f(x):\n    return x\n")
    body = wrapped.format(D=str(tmp_path))
    prog = "import sys\n" + body + "from lib import f\n"
    res = _resolve(prog, str(tmp_path / "harness.py"))
    assert isinstance(res, Top), "an insert that is not a bare, unconditional module-level " \
        "statement must never establish a root"


def test_b_an_insert_after_the_import_it_would_enable_never_resolves(tmp_path: Path):
    (tmp_path / "lib.py").write_text("def f(x):\n    return x\n")
    prog = _src(f"""
        import sys
        from lib import f
        sys.path.insert(0, {str(tmp_path)!r})
    """)
    res = _resolve(prog, str(tmp_path / "harness.py"))
    assert isinstance(res, Top)


# ===================================================== (c): runtime search-order fidelity
def test_c_a_root_shadowing_a_used_stdlib_name_refuses_the_whole_analysis(tmp_path: Path):
    (tmp_path / "re.py").write_text("SURPRISE = 1\n")  # shadows the real stdlib re
    (tmp_path / "lib.py").write_text("def f(x):\n    return x\n")
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import re
        from lib import f
    """)
    res = _resolve(prog, str(tmp_path / "harness.py"))
    assert isinstance(res, Top)
    assert "shadow" in str(res)


def test_c_an_import_before_the_insert_keeps_todays_rules(tmp_path: Path):
    (tmp_path / "lib.py").write_text("def f(x):\n    return x\n")
    prog = _src(f"""
        import re
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from lib import f
    """)
    res = _resolve(prog, str(tmp_path / "harness.py"))
    # the EARLIER `import re` resolved via the real stdlib re (unaffected by a root
    # established only afterwards); the later `from lib import f` still resolves too.
    assert isinstance(res, Resolver), res
    assert res.imports.modules.get("re") == "re"
    assert "f" in res.imports.names


# ==================================================================== (e): package semantics
def test_e_package_init_runs_and_relative_imports_resolve_against_it(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")  # inert: nothing further to open
    (pkg / "sibling.py").write_text("HELPER = 1\n")
    (pkg / "leaf.py").write_text(_src("""
        from .sibling import HELPER
        def f(x):
            return x if HELPER else x
    """))
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from pkg.leaf import f
    """)
    res = _resolve(prog, str(tmp_path / "harness.py"))
    assert isinstance(res, Resolver), res
    sub = res.externs["f"]
    assert sub.filename.endswith("leaf.py")
    assert "HELPER" in sub.imports.names


def test_e_a_non_inert_package_init_must_itself_fully_resolve(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    # "os" itself is no longer an unresolvable stdlib module ([internal ref]) -- "xml" still is. The
    # `print(1)` call (not just the import) is what makes this init NON-inert
    # (extern.toplevel_inert: "imports/defs/literal-assigns only" -- a bare `import xml`
    # alone would be skipped, unopened, never reaching its own unresolvable import at all).
    (pkg / "__init__.py").write_text("import xml\nprint(1)\n")  # a REAL top-level effect
    (pkg / "leaf.py").write_text("def f(x):\n    return x\n")
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from pkg.leaf import f
    """)
    res = _resolve(prog, str(tmp_path / "harness.py"))
    assert isinstance(res, Top), "a non-inert __init__.py using an unresolvable stdlib " \
        "module must refuse the whole import, not be silently skipped"


def test_e_c_extension_refuses(tmp_path: Path):
    (tmp_path / "clib.cpython-314-x86_64-linux-gnu.so").write_bytes(b"\x00")
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from clib import f
    """)
    res = _resolve(prog, str(tmp_path / "harness.py"))
    assert isinstance(res, Top)
    assert "extension" in str(res)


def test_e_namespace_package_ambiguity_refuses(tmp_path: Path):
    ns = tmp_path / "nspkg"
    ns.mkdir()
    (ns / "mod.py").write_text("def f(x):\n    return x\n")  # NO __init__.py
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from nspkg.mod import f
    """)
    res = _resolve(prog, str(tmp_path / "harness.py"))
    assert isinstance(res, Top)
    assert "namespace" in str(res)


# ============================================================ (d): imported code runs too
def test_d_a_library_that_can_raise_a_property_exception_never_proves_true(tmp_path: Path):
    (tmp_path / "lib.py").write_text(_src("""
        def maybe_bad(s):
            return s[0]
    """))
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import _sv_verifier
        from lib import maybe_bad
        if __name__ == "__main__":
            s = _sv_verifier.nondet_str()
            maybe_bad(s)
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=15.0)
    assert r.verdict != "TRUE"


def test_d_a_library_toplevel_monkeypatch_is_refused_not_ignored(tmp_path: Path):
    (tmp_path / "lib.py").write_text(_src("""
        def _bad(self, *a):
            return 0
        int.__add__ = _bad
        def f(x):
            return x
    """))
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import _sv_verifier
        from lib import f
        if __name__ == "__main__":
            f(_sv_verifier.nondet_str())
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=15.0)
    assert r.verdict != "TRUE"


# ============================================================================ (f): probes
def test_f_relative_insert_probe_never_true(tmp_path: Path):
    (tmp_path / "lib.py").write_text("def f(x):\n    return x\n")
    prog = _src("""
        import sys
        sys.path.insert(0, "lib_relative_dir")
        import _sv_verifier
        from lib import f
        if __name__ == "__main__":
            f(_sv_verifier.nondet_str())
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=10.0)
    assert r.verdict != "TRUE"


def test_f_conditional_insert_probe_never_true(tmp_path: Path):
    (tmp_path / "lib.py").write_text("def f(x):\n    return x\n")
    prog = _src(f"""
        import sys
        if True:
            sys.path.insert(0, {str(tmp_path)!r})
        import _sv_verifier
        from lib import f
        if __name__ == "__main__":
            f(_sv_verifier.nondet_str())
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=10.0)
    assert r.verdict != "TRUE"


def test_f_shadowing_probe_never_true(tmp_path: Path):
    (tmp_path / "string.py").write_text("SURPRISE = 1\n")  # shadows stdlib string
    (tmp_path / "lib.py").write_text("import string\ndef f(x):\n    return x\n")
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import string
        import _sv_verifier
        from lib import f
        if __name__ == "__main__":
            f(_sv_verifier.nondet_str())
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=10.0)
    assert r.verdict != "TRUE"


def test_f_unproven_site_probe_never_true(tmp_path: Path):
    (tmp_path / "lib.py").write_text(_src("""
        def f(s, i):
            return s[i]
    """))
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import _sv_verifier
        from lib import f
        if __name__ == "__main__":
            s = _sv_verifier.nondet_str()
            i = _sv_verifier.nondet_int()
            f(s, i)
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=10.0)
    assert r.verdict != "TRUE"


def test_f_package_c_extension_probe_never_true(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "native.cpython-314-x86_64-linux-gnu.so").write_bytes(b"\x00")
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import _sv_verifier
        from pkg.native import f
        if __name__ == "__main__":
            f(_sv_verifier.nondet_str())
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=10.0)
    assert r.verdict != "TRUE"


def test_f_real_crash_negative_control_never_true(tmp_path: Path):
    """A library function bundled the same way that genuinely raises for some reachable
    input -- must stay UNKNOWN (absint never guesses), never a wrong TRUE."""
    (tmp_path / "lib.py").write_text(_src("""
        def first(s):
            return s[0]
    """))
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import _sv_verifier
        from lib import first
        if __name__ == "__main__":
            s = _sv_verifier.nondet_str()
            first(s)
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=10.0)
    assert r.verdict != "TRUE"
    assert r.verdict == "UNKNOWN"


# ============================================================ (g)/positive: real crossing
def test_g_a_provably_safe_library_function_gets_a_real_true(tmp_path: Path):
    (tmp_path / "lib.py").write_text(_src("""
        def is_str(obj):
            return isinstance(obj, str)
    """))
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import _sv_verifier
        from lib import is_str
        if __name__ == "__main__":
            is_str(_sv_verifier.nondet_str())
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=15.0)
    assert r.verdict == "TRUE"
    # the check must be attributed to lib.py's own line (module_key plumbing), not the
    # harness's -- a vacuous 0-site TRUE from a modelling gap would not name a real site.
    assert any("call" in site[2] or "attr" in site[2] for site in r.sites)


def test_g_no_op_for_a_program_without_sys_path_insert(tmp_path: Path):
    """Condition (g): a program that never touches sys.path is byte-for-byte unaffected --
    an unresolvable import still refuses exactly as before read-and-recurse existed."""
    prog = _src("""
        from nonexistent_package import whatever
    """)
    tree = ast.parse(prog, "<program>")
    with pytest.raises(Top):
        Resolver(tree, DET_MODULES, filename="<program>")


# ================================================ MG's PR 86 review (7 wrong TRUEs): regressions
#
# [host]:[local path] (SUMMARY.txt), MG's PR 86 review comment.
# Three bugs, one per group below:
#
# (1) pyresolve.py::Resolver._check_inits resolved a non-inert package __init__.py but
#     discarded the child Resolver instead of linking it into self.externs, so
#     census.py/determinism.py (which reach an extern file only through
#     Resolver.all_resolvers()) never saw a division/assert/call/attribute-access at a
#     package init's own top level, although CPython always runs it on import
#     (case01, d02, d03, d04, d18, e01 below).
# (2) extern.py::find_in_roots checked a same-named ``.py`` FILE before a package
#     DIRECTORY, the opposite of CPython's own FileFinder order, so a safe module file
#     could mask a dangerous package of the same name (c12 below).
# (3) absint/interp.py::Analyzer.capture_invariant indexed ``self.invariant_points``
#     (built only from the top-level program's own tree) with a node from a
#     recursively-read extern file's tree once read-and-recurse crossed an extern
#     FUNCTION boundary -- an unconditional ``self.invariant_points[node]`` raised
#     KeyError whenever ``invariant_locations`` was non-empty (every real SV run: the
#     engine always derives it from the program's own obligations/while-loops), which is
#     what masked (1)/(2) at the CLI level: absint crashed rather than returning its
#     wrong TRUE, and a lucky blind concolic run supplied the correct FALSE instead of a
#     soundness argument.
def test_pkginit_division_effect_is_not_dropped(tmp_path: Path):
    """(1)/case01: an unconditional division at a package __init__.py's own top level."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("_denom = 0\n_boom = 1 / _denom\n")
    (pkg / "leaf.py").write_text("def safe_value():\n    return 5\n")
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from pkg.leaf import safe_value
        def main():
            x = safe_value()
            print(1 / x)
        if __name__ == "__main__":
            main()
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=15.0)
    assert r.verdict != "TRUE"


def test_pkginit_assert_effect_is_not_dropped(tmp_path: Path):
    """(1)/d02: an AssertionError behind an ``if`` on a module global at a package
    __init__.py's own top level -- non-inert because of the ``If``, not the assert alone."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(_src("""
        _flag = False
        if not _flag:
            assert False, "package init assertion"
    """))
    (pkg / "leaf.py").write_text("def f(x):\n    return x\n")
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from pkg.leaf import f
        if __name__ == "__main__":
            f(1)
    """)
    r = prove(prog, ("AssertionError",), filename=str(tmp_path / "harness.py"), time_budget=15.0)
    assert r.verdict != "TRUE"


def test_pkginit_call_raises_effect_is_not_dropped(tmp_path: Path):
    """(1)/d03: the "registration pattern" -- a package __init__.py calls a function at
    import time that raises, rather than raising directly at module level."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(_src("""
        def _register():
            return 1 / 0
        _registered = _register()
    """))
    (pkg / "leaf.py").write_text("def f(x):\n    return x\n")
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from pkg.leaf import f
        if __name__ == "__main__":
            f(1)
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=15.0)
    assert r.verdict != "TRUE"


def test_nested_pkginit_effect_is_not_dropped(tmp_path: Path):
    """(1)/d04: a two-level ``a.b`` import -- the OUTER a/__init__.py is (correctly)
    inert and skipped, the INNER a/b/__init__.py is not and must still be analysed."""
    a = tmp_path / "a"
    a.mkdir()
    (a / "__init__.py").write_text("")  # inert
    b = a / "b"
    b.mkdir()
    (b / "__init__.py").write_text("_denom = 0\n_boom = 1 / _denom\n")
    (b / "leaf.py").write_text("def f(x):\n    return x\n")
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from a.b.leaf import f
        if __name__ == "__main__":
            f(1)
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=15.0)
    assert r.verdict != "TRUE"


def test_pkginit_attributeerror_effect_is_not_dropped(tmp_path: Path):
    """(1)/d18: an AttributeError (dynamic-typing property) at a package __init__.py's
    own top level -- confirms the dropped-resolver bug is not arithmetic-specific."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("_boom = (1).nonexistent_attr\n")
    (pkg / "leaf.py").write_text("def f(x):\n    return x\n")
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from pkg.leaf import f
        if __name__ == "__main__":
            f(1)
    """)
    r = prove(prog, ("TypeError", "AttributeError"), filename=str(tmp_path / "harness.py"),
              time_budget=15.0)
    assert r.verdict != "TRUE"


def test_pkginit_conditionally_gated_effect_is_not_dropped(tmp_path: Path):
    """(1)/e01: the property-violating effect is behind a condition -- shows the
    dropped-resolver bug is not limited to an unconditional top-level statement."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(_src("""
        _mode = "boom"
        if _mode == "boom":
            _denom = 0
            _boom = 1 / _denom
    """))
    (pkg / "leaf.py").write_text("def f(x):\n    return x\n")
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from pkg.leaf import f
        if __name__ == "__main__":
            f(1)
    """)
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=15.0)
    assert r.verdict != "TRUE"


def test_pkg_directory_wins_over_a_same_named_module_file(tmp_path: Path):
    """(2)/c12: a root containing BOTH thing.py (safe) and thing/ (a package, dangerous)
    -- real CPython's FileFinder resolves the PACKAGE; find_in_roots used to check the
    leaf .py file first and would have resolved to the safe module instead."""
    (tmp_path / "thing.py").write_text("def get():\n    return 5\n")
    thing_pkg = tmp_path / "thing"
    thing_pkg.mkdir()
    (thing_pkg / "__init__.py").write_text(_src("""
        def get():
            return 0
    """))
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from thing import get
        def main():
            x = get()
            print(1 / x)
        if __name__ == "__main__":
            main()
    """)
    res = _resolve(prog, str(tmp_path / "harness.py"))
    assert isinstance(res, Resolver), res
    # The resolved binding must be the PACKAGE's own __init__.py, not the sibling module
    # file -- confirms find_in_roots now checks the directory first, matching CPython.
    assert res.externs["get"].filename.endswith(os.path.join("thing", "__init__.py"))
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=15.0)
    assert r.verdict != "TRUE"


def test_capture_invariant_does_not_crash_across_an_extern_function_boundary(tmp_path: Path):
    """(3): Analyzer.invariant_points only covers the top-level program's own tree, but
    exec_stmt calls capture_invariant for EVERY statement it executes, including a
    recursively-read extern file's own top level once read-and-recurse crosses a
    function boundary. A non-empty invariant_locations (every real tharkun.sv run
    derives one from the program's own obligations/while-loops -- scheduler.py's
    _obligation_knowledge/source_points) used to hit an unconditional
    ``self.invariant_points[node]`` and raise KeyError on the FIRST extern statement
    executed, regardless of whether that statement's OWN location was ever requested.
    Must simply run to a verdict, never raise -- reuses case01's shape (a function
    reached through an extern boundary) so the assertion below is meaningful too, not
    just "didn't crash": the crash used to happen before a verdict was ever reached."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("_denom = 0\n_boom = 1 / _denom\n")
    (pkg / "leaf.py").write_text("def safe_value():\n    return 5\n")
    prog = _src(f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        from pkg.leaf import safe_value
        def main():
            x = safe_value()
            print(1 / x)
        if __name__ == "__main__":
            main()
    """)
    # Any non-empty set reproduces the crash -- capture_invariant's own KeyError happens
    # before it ever checks whether the LOOKED-UP location is a member of this set.
    r = prove(prog, CLASSES, filename=str(tmp_path / "harness.py"), time_budget=15.0,
              invariant_locations={("<dummy>", 1, 0, "main")})
    assert r.verdict != "TRUE"
