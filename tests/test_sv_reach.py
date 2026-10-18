"""[internal ref]: what a program reaches and calls (``sv/reach.py``) and what the hybrid stage does with it.

A product over a library -- ``assert f_orig(s) == f_mut(s)`` with the library shipped next to the program -- used to
give the fuzzer coverage of the product file only and seeds/dictionary from its (literal-free) AST.  The module
reads the imports, follows the calls, and hands the stage the guards' tokens, the code's regexes and the packages
to instrument.  Nothing here runs a task.
"""
from __future__ import annotations

import ast
import textwrap

import pytest

from tharkun.hybrid.fdp_codec import atheris_available
from tharkun.sv import hybrid, reach
from tharkun.sv.engine import replay, verify


def write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return str(path)


# --- resolving imports ----------------------------------------------------------------------------

def test_sibling_modules_and_packages_are_reached_transitively_the_stdlib_is_not(tmp_path):
    prog = write(tmp_path, "main.py", """
        import os, json
        import lib.core as core
        from lib import extra
    """)
    write(tmp_path, "lib/__init__.py", "from . import core\n")
    write(tmp_path, "lib/core.py", "from .util import helper\nimport re\n")
    write(tmp_path, "lib/util.py", "def helper(): return 1\n")
    write(tmp_path, "lib/extra.py", "X = 1\n")
    names = [m.name for m in reach.reached(prog)]
    assert set(names) == {"lib", "lib.core", "lib.util", "lib.extra"}
    assert names.index("lib") < names.index("lib.core")                    # a package before its submodule
    assert reach.packages(reach.reached(prog)) == ["lib"]                  # what atheris.instrument_imports takes


def test_a_directory_the_program_adds_to_sys_path_is_a_root(tmp_path):
    (tmp_path / "vendor").mkdir()
    write(tmp_path, "vendor/pkg.py", "Y = 2\n")
    prog = write(tmp_path, "main.py", f"import sys\nsys.path.insert(0, {str(tmp_path / 'vendor')!r})\nimport pkg\n")
    assert [m.name for m in reach.reached(prog)] == ["pkg"]


def test_the_yml_declared_files_directories_are_roots(tmp_path):
    write(tmp_path, "src/helpers.py", "Z = 3\n")
    prog = write(tmp_path, "task/main.py", "import helpers\n")
    yml = write(tmp_path, "task/main.yml", """
        format_version: '2.0'
        input_files: 'main.py'
        required_files:
          - ../src/helpers.py
    """)
    assert [m.name for m in reach.reached(prog, yml)] == ["helpers"]
    assert reach.reached(prog) == []                                       # without the yml it is not on the path


def test_cycles_unreadable_files_and_the_program_itself_are_survived(tmp_path):
    prog = write(tmp_path, "main.py", "import a\nimport main\n")
    write(tmp_path, "a.py", "import b\n")
    write(tmp_path, "b.py", "import a\nimport main\n")
    write(tmp_path, "broken.py", "def (:\n")
    assert [m.name for m in reach.reached(prog)] == ["a", "b"]
    prog2 = write(tmp_path, "m2.py", "import broken\n")
    assert reach.reached(prog2) == []


def test_relative_imports_beyond_the_top_level_package_are_ignored(tmp_path):
    prog = write(tmp_path, "main.py", "import pkg\n")
    write(tmp_path, "pkg/__init__.py", "from ... import nothing\nfrom .. import also_nothing\nfrom . import real\n")
    write(tmp_path, "pkg/real.py", "A = 1\n")
    assert [m.name for m in reach.reached(prog)] == ["pkg", "pkg.real"]


# --- the call graph -------------------------------------------------------------------------------

LIB = """
    import re
    from ._regex import PATTERN

    def check(s):
        if not isinstance(s, str) or helper_strip(s) == "":
            return False
        head, tail = s.split("@")
        if head.startswith('"') and head.endswith('"'):
            head = head[1:-1]
        return PATTERN.match(head + "@" + tail) is not None

    def helper_strip(s):
        return s.strip()

    def never_called(s):
        return s.startswith("NEVER-CALLED-TOKEN")

    class Unused:
        def method(self):
            return "unused-class-literal"
"""


def library(tmp_path):
    write(tmp_path, "lib/__init__.py", "from .valid import check\n")
    write(tmp_path, "lib/valid.py", LIB)
    write(tmp_path, "lib/_regex.py", """
        import re
        RAW = r"[a-z]+@[a-z]+\\.[a-z]{2,3}"
        PATTERN = re.compile(r"^{}$".format(RAW))
        OTHER = re.compile(r"^unrelated-[0-9]+$")
    """)


def test_the_call_graph_follows_aliases_dotted_chains_reexports_and_helpers(tmp_path):
    library(tmp_path)
    for body in ("import lib\nprint(lib.check('x'))\n", "from lib import check\ncheck('x')\n",
                 "import lib.valid as v\nv.check('x')\n", "import lib.valid\nlib.valid.check('x')\n"):
        prog = write(tmp_path, "main.py", body)
        tree = ast.parse(body)
        units = reach.called(tree, reach.reached(prog))
        names = {getattr(u.node, "name", None) for u in units if u.module}
        assert {"check", "helper_strip"} <= names, body                    # the callee and what it calls
        assert "never_called" not in names and "Unused" not in names, body
        assigns = {t.id for u in units if u.module and isinstance(u.node, ast.Assign)
                   for t in u.node.targets if isinstance(t, ast.Name)}
        assert {"PATTERN", "RAW"} <= assigns and "OTHER" not in assigns, body   # the pattern it reads, and its source


def test_a_class_the_program_names_brings_its_methods(tmp_path):
    write(tmp_path, "lib/__init__.py", "")
    write(tmp_path, "lib/k.py", "class K:\n    def go(self, s):\n        return s.startswith('kk')\n")
    body = "from lib.k import K\nK().go('x')\n"
    prog = write(tmp_path, "main.py", body)
    units = reach.called(ast.parse(body), reach.reached(prog))
    toks = reach.dictionary(units, reach.folders(reach.reached(prog), ast.parse(body)))
    assert "kk" in toks


# --- what the code carries ------------------------------------------------------------------------

def test_guard_tokens_cover_every_arity_form_and_skip_the_script_guard():
    tree = ast.parse(textwrap.dedent("""
        if __name__ == "__main__":
            pass
        def f(s, t):
            if s.startswith(("http://", "https://")) or s.endswith('.py'):
                pass
            a, b = s.split('@')
            c = s.partition(':')
            if '..' in s or s in ('x', 'y') or s == 'exact' or t != "other":
                pass
            s.replace(' ', '').strip('/').count('ab').removeprefix('pre')
            s.join(['not', 'a', 'guard'])
    """))
    toks = reach.guard_tokens(tree)
    assert {"http://", "https://", ".py", "@", ":", "..", "x", "y", "exact", "other", " ", "/", "ab", "pre"} <= set(toks)
    assert "__main__" not in toks and "not" not in toks


def test_docstrings_and_annotations_are_not_literals():
    tree = ast.parse('"""module doc"""\ndef f(x: "annotated") -> "ret":\n    """doc"""\n    return "real"\n')
    assert reach.literals(tree) == ["real"]


def test_regex_patterns_fold_through_names_format_fstrings_concatenation_and_flags():
    tree = ast.parse(textwrap.dedent("""
        import re
        BASE = r"[a-z]+"
        TAIL = r"\\d{2}"
        A = re.compile(BASE + "-" + TAIL)
        B = re.compile(r"^{}$".format(BASE), re.IGNORECASE | re.DOTALL)
        C = re.compile(f"^{TAIL}$", flags=re.M)
        re.sub(r"x+", "y", "s")
        D = re.compile(some_function())
    """))
    import re
    assert set(reach.regex_patterns(tree)) == {
        (r"[a-z]+-\d{2}", 0), ("^[a-z]+$", int(re.I | re.S)), (r"^\d{2}$", int(re.M)), ("x+", 0)}


def test_regex_tokens_are_literal_runs_and_class_members():
    toks = reach.regex_tokens(r"^([a-z-]+://)(user@)?[!#$]+host\.com$")
    assert "://" in toks and "user@" in toks and "host.com" in toks
    assert {"!", "#", "$"} <= set(toks)


def test_the_dictionary_orders_guards_first_is_small_and_printable_only(tmp_path):
    library(tmp_path)
    body = "import lib\nlib.check('x')\n"
    prog = write(tmp_path, "main.py", body)
    tree, mods = ast.parse(body), reach.reached(prog)
    units = [reach.Unit("", tree)] + [u for u in reach.called(tree, mods) if u.module]
    toks = reach.dictionary(units, reach.folders(mods, tree))
    assert {"@", '"'} <= set(toks[:4])                                       # the guards' arguments lead
    assert "NEVER-CALLED-TOKEN" not in toks and "unused-class-literal" not in toks
    assert len(toks) <= 64 and all(t.isprintable() for t in toks)
    assert reach.dictionary(units, reach.folders(mods, tree), limit=2) == toks[:2]


def test_the_rendered_dictionary_escapes_backslash_quote_and_non_ascii():
    text = reach.render_dictionary(["a", '"', "\\ ", "\n", "é"])
    assert text.splitlines() == ['k0="a"', 'k1="\\""', 'k2="\\\\ "', 'k3="\\x0a"', 'k4="\\xe9"']


# --- the hybrid stage's use of it -----------------------------------------------------------------

def test_composed_seeds_wrap_the_leading_segment_and_the_whole_sample():
    seeds = hybrid._composed_seeds(["a@bc.de"], ['"', "@"])
    assert '"a"@bc.de' in seeds and '"a@bc.de"' in seeds and '"a@bc.de' in seeds
    assert "a@bc.de" not in seeds and len(seeds) == len(set(seeds))
    assert len(hybrid._composed_seeds(["a@bc.de"] * 3 + ["x.y.z@q.rs"], ['"', "@", ".", " "], limit=10)) == 10


def test_composed_seeds_ignore_long_tokens_and_empty_inputs():
    assert hybrid._composed_seeds([], ['"']) == [] and hybrid._composed_seeds(["abc"], []) == []
    assert all("LONGTOKEN" not in s for s in hybrid._composed_seeds(["a@b"], ["LONGTOKEN"]))


def test_regex_samples_are_the_shortest_drawn_and_match_the_pattern():
    import re
    pattern = r"^[a-z]+@[a-z]+\.[a-z]{2,3}$"
    got = hybrid._regex_samples([(pattern, 0)])
    assert 0 < len(got) <= hybrid.REGEX_SAMPLES_KEPT
    assert all(re.match(pattern, g) for g in got) and got == sorted(got, key=lambda v: (len(v), v))


def test_the_plan_of_a_single_file_program_reaches_nothing(tmp_path):
    prog = write(tmp_path, "solo.py", "import _sv_verifier\nif __name__ == '__main__':\n    s = _sv_verifier.nondet_str()\n")

    class Ctx:
        program, task = prog, None

    plan = hybrid._Plan(Ctx, ast.parse(open(prog).read()))
    assert plan.called == [] and plan.units == [] and plan.patterns() == []


def test_the_reach_plan_writes_the_dictionary_and_names_the_packages_to_instrument(tmp_path):
    library(tmp_path)
    body = "import _sv_verifier\nimport lib\nif __name__ == '__main__':\n    lib.check(_sv_verifier.nondet_str())\n"
    prog = write(tmp_path, "main.py", body)

    class Ctx:
        program, task = prog, None

    root = tmp_path / "scratch"
    root.mkdir()
    plan, fields, args = hybrid._reach_plan(Ctx, ast.parse(body), root)
    assert fields == {"reach": ["lib"]}
    (dict_arg,) = args
    assert dict_arg == f"-dict={root / 'tokens.dict'}"
    assert '"\\""' in (root / "tokens.dict").read_text()
    assert plan.patterns() == [(r"^[a-z]+@[a-z]+\.[a-z]{2,3}$", 0)]


# --- end to end: the product over a library -------------------------------------------------------

def product(tmp_path, offset_orig, offset_mut):
    for tag, off in (("orig", offset_orig), ("mut", offset_mut)):
        write(tmp_path, f"pkg_{tag}/__init__.py", "from .v import check\n")
        write(tmp_path, f"pkg_{tag}/v.py", f"""
            import re
            PATTERN = re.compile(r"^[a-z]+@[a-z]+\\.[a-z]{{2,3}}$")

            def check(s):
                head, _, tail = s.partition("@")
                if head.startswith('"') and head.endswith('"'):
                    head = head[{off}:-1]
                return PATTERN.match(head + "@" + tail) is not None
        """)
    return write(tmp_path, "product.py", """
        import _sv_verifier
        import pkg_orig
        import pkg_mut

        def safe(fn, *a):
            try:
                return ("ok", fn(*a))
            except Exception as e:
                return ("exc", type(e).__name__)

        if __name__ == "__main__":
            s = _sv_verifier.nondet_str()
            assert safe(pkg_orig.check, s) == safe(pkg_mut.check, s)
    """)


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_the_hybrid_stage_kills_a_mutant_inside_a_library_the_product_ships(tmp_path):
    """The shape of m131 (a quoted local part, a slice offset moved by one): enumeration cannot reach it, the
    product file has no literal, the guards live in the library.  The fuzzer instruments the library, starts from
    regex samples with the guards' tokens spliced in, and the witness replays."""
    p = product(tmp_path, 1, 2)
    v = verify(p, ("AssertionError",), jobs=1, budget_seconds=60, prove=False, concolic=False, stages=["hybrid"])
    assert v.verdict == "false" and v.component.startswith("hybrid:")
    assert replay(p, v.witness, ("AssertionError",))[0]


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_equal_copies_are_never_reported_killed(tmp_path):
    p = product(tmp_path, 1, 1)
    v = verify(p, ("AssertionError",), jobs=1, budget_seconds=12, prove=False, concolic=False, stages=["hybrid"])
    assert v.verdict == "unknown"
