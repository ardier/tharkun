"""[internal ref] (the "os bug", [internal ref]): ``os``/``os.path`` used to be refused wholesale
(``import os`` alone made every prover refuse, whatever it was used for). This suite locks
in the three-way split now in place: pure ``os.path`` string functions are modelled EXACTLY
when their arguments are known; reads of real process/OS/filesystem state become a sound
HAVOC value (never a refusal, never a false certainty); everything else (``os.system``,
``os.remove``, ``os.environ`` used bare, ...) still refuses, on purpose.
"""

from __future__ import annotations

from tharkun.solve.absint import prove
from tharkun.solve.census import site_census
from tharkun.solve.determinism import determinism_census

DL = ("KeyError", "IndexError")
AR = ("ZeroDivisionError",)
AS = ("AssertionError",)


# ---------------------------------------------------------------- (a) pure os.path -- exact

def test_join_is_computed_exactly():
    src = 'import os\np = os.path.join("/a", "b", "c.txt")\nassert p == "/a/b/c.txt"\n'
    r = prove(src, AS, filename="t.py", time_budget=10)
    assert r.verdict == "TRUE", r.reason


def test_dirname_basename_splitext_are_computed_exactly():
    src = ('import os\n'
           'assert os.path.dirname("/x/y/z.py") == "/x/y"\n'
           'assert os.path.basename("/x/y/z.py") == "z.py"\n'
           'root, ext = os.path.splitext("/x/y/z.tar.gz")\n'
           'assert ext == ".gz"\n')
    r = prove(src, AS, filename="t.py", time_budget=10)
    assert r.verdict == "TRUE", r.reason


def test_abspath_of_an_absolute_literal_is_exact_but_of_a_relative_one_is_havoc():
    exact = 'import os\np = os.path.abspath("/a/b/../c.txt")\nassert p == "/a/c.txt"\n'
    r = prove(exact, AS, filename="t.py", time_budget=10)
    assert r.verdict == "TRUE", r.reason
    # a RELATIVE literal's abspath depends on the real cwd -- never guessed
    relative = 'import os\np = os.path.abspath("rel.txt")\nassert p == "/anything"\n'
    r2 = prove(relative, AS, filename="t.py", time_budget=10)
    assert r2.verdict != "TRUE", r2.reason


def test_pure_os_path_is_deterministic():
    src = 'import os\nx = os.path.join(os.path.dirname("/a/b.py"), "c.py")\n'
    assert determinism_census(src).ok


def test_os_constants_are_deterministic_and_exact():
    src = 'import os\nassert os.sep == "/"\n'
    r = prove(src, AS, filename="t.py", time_budget=10)
    assert r.verdict == "TRUE", r.reason
    assert determinism_census(src).ok


# ------------------------------------------------------------ (b) environment reads -- havoc

def test_getenv_never_certifies_a_property_that_depends_on_it():
    src = 'import os\nv = os.getenv("X")\nif v == "boom":\n    assert False\n'
    r = prove(src, AS, filename="t.py", time_budget=10)
    assert r.verdict != "TRUE", r.reason
    assert not determinism_census(src).ok


def test_getenv_with_default_joins_the_defaults_own_value():
    # real os.getenv(key, default) returns `default` UNCHANGED when unset -- a literal
    # int default must still be able to flow into an int-typed site without refusing.
    src = 'import os\nv = os.getenv("X", 5)\nx = 1 // v\n'
    r = prove(src, ("ZeroDivisionError", "TypeError"), filename="t.py", time_budget=10)
    assert r.verdict != "TRUE", r.reason  # v could be "0" (a str) -- 1 // "0" is TypeError,
    # or the real env value (unknown) -- never a wrong TRUE either way; UNKNOWN is correct


def test_getcwd_listdir_getpid_cpu_count_never_refuse_the_call_itself():
    for src, classes in [
        ('import os\nc = os.getcwd()\nprint(c.upper())\n', ("AttributeError",)),
        ('import os\nfor name in os.listdir():\n    print(name.upper())\n', ("AttributeError",)),
        ('import os\np = os.getpid()\nx = 1 / p\n', AR),   # p >= 1 always: TRUE is a real proof
        ('import os\nn = os.cpu_count()\nx = 1 / n\n', ("ZeroDivisionError", "TypeError")),
    ]:
        r = prove(src, classes, filename="t.py", time_budget=10)
        assert "not modelled" not in r.reason and "imports os" not in r.reason, (src, r.reason)


def test_getpid_is_provably_never_a_zero_divisor():
    # a real, non-trivial gain: a POSITIVE-int-only havoc value the interval prover can use.
    src = 'import os\np = os.getpid()\nx = 1 / p\n'
    r = prove(src, AR, filename="t.py", time_budget=10)
    assert r.verdict == "TRUE", r.reason


def test_urandom_never_certifies_a_property_that_depends_on_it():
    src = 'import os\nx = os.urandom(1)[0]\na = [0] * 255\nprint(a[x])\n'
    r = prove(src, DL, filename="t.py", time_budget=10)
    assert r.verdict != "TRUE", r.reason
    assert not determinism_census(src).ok


def test_path_exists_isfile_isdir_never_certify_a_property_that_depends_on_them():
    for name in ("exists", "isfile", "isdir"):
        src = f'import os\nd = {{True: 1}}\nprint(d[os.path.{name}("/nonexistent-marker")])\n'
        r = prove(src, DL, filename="t.py", time_budget=10)
        assert r.verdict != "TRUE", (name, r.reason)
        assert not determinism_census(src).ok


# --------------------------------------------------------- (c) side-effecting -- still refuses

def test_os_environ_bare_still_refuses():
    src = 'import os\nv = os.environ["X"]\nassert v != "boom"\n'
    r = prove(src, AS, filename="t.py", time_budget=10)
    assert r.verdict != "TRUE"
    assert "os.environ" in r.reason
    assert not determinism_census(src).ok
    assert "os.environ" in determinism_census(src).reason


def test_side_effecting_os_calls_still_refuse():
    for call in ("os.system('echo hi')", "os.remove('/tmp/x')", "os.chdir('/tmp')",
                 "os.mkdir('/tmp/x')", "os.putenv('X', 'Y')", "os.execv('/bin/sh', [])"):
        src = f"import os\n{call}\n"
        r = prove(src, AS, filename="t.py", time_budget=10)
        assert r.verdict != "TRUE", (call, r.reason)
        assert not determinism_census(src).ok, call


# ------------------------------------------------- MG's PR 106 review: argument-type checks

def test_havoc_functions_reject_a_definitely_wrong_argument_type():
    """MG's PR 106 review (attacks [host]:[local path]): the HAVOC handlers
    checked arity and ``top`` but never the argument's TYPE, so a definitely-wrong-typed
    argument -- one that natively raises TypeError -- sailed through to a havoc RESULT
    instead of a checked violation: a wrong TRUE for a program whose only possible outcome
    is that native TypeError."""
    cases = [
        ('import os\nos.getenv(5)\n', "os.getenv: int"),                 # o01
        ("import os\nos.urandom('x')\n", "os.urandom: str"),             # o03
        ('import os\nos.path.isfile(None)\n', "os.path.isfile: None"),   # o05
    ]
    for src, why in cases:
        r = prove(src, ("TypeError",), filename="t.py", time_budget=10)
        assert r.verdict != "TRUE", (src, r.reason)
        assert why in r.reason, (src, r.reason)


def test_havoc_functions_still_accept_the_types_they_document():
    # a genuinely valid call of each still gets the ordinary havoc treatment, not a
    # spurious new refusal -- the type check above must not be overly strict.
    for src in (
        'import os\nv = os.getenv("X")\n',
        'import os\nv = os.getenv("X", "d")\n',
        'import os\nos.urandom(16)\n',
        'import os\nos.urandom(True)\n',                 # bool is an int (real Python accepts it)
        'import os\nos.path.isfile("/x")\n',
        'import os\nos.path.isfile(3)\n',                 # a file descriptor -- real Python accepts it
        'import os\nfor n in os.listdir():\n    pass\n',
        'import os\nfor n in os.listdir("/x"):\n    pass\n',
        'import os\nfor n in os.listdir(None):\n    pass\n',
    ):
        r = prove(src, AS, filename="t.py", time_budget=10)
        assert "arg-type TypeError" not in r.reason, (src, r.reason)
        assert "of unknown type" not in r.reason, (src, r.reason)
        assert "unmodelled arity/keywords" not in r.reason, (src, r.reason)


def test_listdir_rejects_a_definitely_wrong_argument_type():
    src = 'import os\nos.listdir(3.5)\n'
    r = prove(src, ("TypeError",), filename="t.py", time_budget=10)
    assert r.verdict != "TRUE", r.reason


# --------------------------------------------------------- MG's PR 106 review: realpath scope

def test_realpath_is_havoc_not_pure_even_for_an_absolute_literal():
    """MG's PR 106 review: UNLIKE abspath, realpath dereferences symlinks against the REAL,
    LIVE filesystem (o06: os.path.realpath("/var/run") == "/run" on [host], a real symlink)
    -- its result for the exact same literal can differ host to host, so it must never be
    computed exactly, even for an already-absolute path."""
    src = 'import os\np = os.path.realpath("/var/run")\nassert p == "/var/run"\n'
    r = prove(src, AS, filename="t.py", time_budget=10)
    assert r.verdict != "TRUE", r.reason
    # abspath (no filesystem access at all for an absolute path) is unaffected -- still exact
    src2 = 'import os\np = os.path.abspath("/a/b/../c.txt")\nassert p == "/a/c.txt"\n'
    r2 = prove(src2, AS, filename="t.py", time_budget=10)
    assert r2.verdict == "TRUE", r2.reason


# ------------------------------------------------------------------- census (proof:no-site)

def test_no_site_census_still_resolves_pure_os_path_programs():
    src = 'import os\nx = os.path.join("/a", "b")\nprint(x)\n'
    r = site_census(src, {"ZeroDivisionError"})
    assert r.no_site, r.reason


def test_a_bare_import_of_an_unaudited_stdlib_module_still_refuses():
    # "os" itself is no longer an example of a refusing import ([internal ref]) -- something never
    # allowlisted anywhere still is, confirming the change is scoped to os/os.path only.
    assert not determinism_census("import xml\n").ok
    r = site_census("import xml\nx = 1\n", {"ZeroDivisionError"})
    assert not r.no_site
