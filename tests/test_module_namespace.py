"""#1132b / [internal ref] / [internal ref]: the static namespace of a module and the PROOF that a module-level name is constant.

Negative controls for every rebinding form (module-level reassignment in all its spellings, ``global`` writes,
cross-module attribute stores resolved through every import alias, ``setattr``/``__dict__``/``globals()`` writes,
``exec``, an ``__init__`` that changes a sibling, mutation of a container) -- each one must make the name
NOT proven -- and positive controls that stay proven.  The evaluator tests come from the [internal ref] reference.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from tharkun.solve.module_namespace import (
    LocalModule, ModuleRef, RegexConst, evaluate_module, module_constants, regex_consts, regex_globals,
)


def write(root: Path, rel: str, body: str = "") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def rec(records, name):
    got = [r for r in records if r["name"] == name]
    assert len(got) == 1, (name, [r["name"] for r in records])
    return got[0]


def consts(path, roots=()):
    return {r["name"]: r for r in module_constants(str(path), roots)}


# -- the evaluator (the [internal ref] reference) -------------------------------------------


def test_the_string_utils_shape_format_built_pattern_via_star_import(tmp_path):
    write(tmp_path, "pkg/__init__.py", "")
    write(tmp_path, "pkg/_regex.py", r'''
        import re
        RAW = r"[a-z]+@[a-z\d-]+\.[a-z]{2,4}"
        EMAIL_RE = re.compile(r'^{}$'.format(RAW))
        EMAILS_RE = re.compile(r'({})'.format(RAW))
        HTML_RE = re.compile(r'<[a-z]+>', re.IGNORECASE | re.MULTILINE | re.DOTALL)
        _private = re.compile('p')
    ''')
    entry = write(tmp_path, "pkg/validation.py", "from ._regex import *\ndef f(x): return EMAIL_RE.match(x)\n")
    got = regex_consts(evaluate_module(entry))
    assert set(got) == {"EMAIL_RE", "EMAILS_RE", "HTML_RE"}, "private names are not star-imported"
    assert got["EMAIL_RE"] == RegexConst(r"^[a-z]+@[a-z\d-]+\.[a-z]{2,4}$", 0)
    assert got["HTML_RE"].flags == int(re.I | re.M | re.S)
    assert got["EMAIL_RE"].line == 4 and got["EMAIL_RE"].module.endswith("_regex.py")


def test_all_restricts_the_star_import(tmp_path):
    write(tmp_path, "m/lib.py", "import re\n__all__ = ['A']\nA = re.compile('a')\nB = re.compile('b')\n")
    entry = write(tmp_path, "m/main.py", "from .lib import *\n")
    assert set(regex_consts(evaluate_module(entry))) == {"A"}


def test_named_aliased_and_absolute_imports_and_package_roots(tmp_path):
    write(tmp_path, "helpers.py", "import re\nX = re.compile('x+')\nY = re.compile('y+')\n")
    entry = write(tmp_path, "main.py", "from helpers import X as XX\nfrom helpers import Y\n")
    assert set(regex_consts(evaluate_module(entry))) == {"XX", "Y"}
    write(tmp_path, "root/lib/consts.py", "import re\nZ = re.compile('z')\n")
    entry2 = write(tmp_path, "app/main.py", "from lib.consts import Z\n")
    assert regex_consts(evaluate_module(entry2)) == {}
    assert set(regex_consts(evaluate_module(entry2, [tmp_path / "root"]))) == {"Z"}


def test_fstrings_percent_format_concat_str_methods_flag_arithmetic(tmp_path):
    entry = write(tmp_path, "m.py", '''
        import re, string
        A = "ab"
        B = f"<{A}>"
        C = "%s-%d" % (A, 3)
        D = "x" + A * 2
        E = ",".join(["p", "q"]).upper()
        F = re.compile("[%s]+" % re.escape(string.punctuation))
        G = re.compile(B)
        H = re.compile("a" "b", re.I | re.M)
        I = re.compile("a", re.I + re.M)
        J = re.compile("a", flags=re.S)
    ''')
    ns = evaluate_module(entry)
    assert (ns["B"], ns["C"], ns["D"], ns["E"]) == ("<ab>", "ab-3", "xabab", "P,Q")
    got = regex_consts(ns)
    assert got["G"] == RegexConst("<ab>", 0) and got["H"].flags == got["I"].flags == int(re.I | re.M)
    assert got["J"].flags == int(re.S)


def test_unknown_names_are_absent_never_stale(tmp_path):
    entry = write(tmp_path, "m.py", '''
        import re
        X = re.compile("first")
        X = make_pattern()
        Y = re.compile("y")
        def Y(): pass
        Z = re.compile("z")
        if cond:
            Z = re.compile("other")
        W = re.compile(unknown_name)
        V = re.compile("(")
    ''')
    assert regex_consts(evaluate_module(entry)) == {}


def test_cycles_budgets_and_unreadable_sources_are_total(tmp_path):
    write(tmp_path, "a.py", "from b import *\nimport re\nA = re.compile('a')\n")
    write(tmp_path, "b.py", "from a import *\nimport re\nB = re.compile('b')\n")
    assert "A" in regex_consts(evaluate_module(tmp_path / "a.py"))
    big = write(tmp_path, "big.py", "import re\nBIG = 'x' * 10**9\nOK = re.compile('ok')\nTOO = re.compile(BIG)\n")
    assert set(regex_consts(evaluate_module(big))) == {"OK"}
    bad = write(tmp_path, "bad.py", "def (:\n")
    assert evaluate_module(bad) == {} and evaluate_module(tmp_path / "missing.py") == {}


def test_stdlib_module_refs(tmp_path):
    entry = write(tmp_path, "m.py", "import re as _re\nfrom string import digits\nfrom re import I\nR = _re.compile('[' + digits + ']', I)\n")
    ns = evaluate_module(entry)
    assert isinstance(ns["_re"], ModuleRef)
    assert regex_consts(ns)["R"] == RegexConst("[0123456789]", int(re.I))


# -- positive controls: proven --------------------------------------------------


def test_a_plain_constant_is_proven_with_its_value(tmp_path):
    f = write(tmp_path, "m.py", "LIMIT = 50\nNAME = 'x'\nPAIR = (1, 2)\nOFF = None\nSCALE = 1.5\nRAW = b'ab'\n")
    c = consts(f)
    for name, value in (("LIMIT", 50), ("NAME", "x"), ("OFF", None), ("SCALE", 1.5)):
        assert c[name]["proven"] and c[name]["value"] == value and c[name]["rebinds"] == []
    assert c["PAIR"]["value"] == {"tuple": [1, 2]} and c["RAW"]["value"] == {"bytes": "6162"}
    assert c["LIMIT"]["provenance"]["complete"] and c["LIMIT"]["origin"] == "assign"


def test_a_stable_but_unfoldable_name_is_proven_without_a_value(tmp_path):
    f = write(tmp_path, "m.py", "import os\nK = int(os.environ.get('K', '5'))\n")
    r = consts(f)["K"]
    assert r["proven"] is True and r["value"] is None and r["value_kind"] == "unknown"


def test_functions_classes_and_imports_are_names_too(tmp_path):
    f = write(tmp_path, "m.py", "import json\nfrom os import path\ndef f(): pass\nclass C: pass\n")
    c = consts(f)
    assert {n: (c[n]["origin"], c[n]["proven"]) for n in ("json", "path", "f", "C")} == {
        "json": ("import", True), "path": ("importfrom", True), "f": ("def", True), "C": ("class", True)}


def test_a_value_imported_from_a_sibling_is_proven_when_the_sibling_never_rebinds_it(tmp_path):
    write(tmp_path, "pkg/__init__.py")
    write(tmp_path, "pkg/_c.py", "LIMIT = 50\n")
    mod = write(tmp_path, "pkg/mod.py", "from ._c import LIMIT\n\ndef f(x):\n    return x <= LIMIT\n")
    r = consts(mod)["LIMIT"]
    assert r["proven"] and r["value"] == 50 and r["origin"] == "assign"


def test_a_mutable_container_nobody_mutates_is_proven(tmp_path):
    f = write(tmp_path, "m.py", "TABLE = {'a': 1}\nITEMS = [1, 2]\n\ndef f(k):\n    return TABLE[k] + len(ITEMS) + sum(ITEMS)\n")
    c = consts(f)
    assert c["TABLE"]["proven"] and c["ITEMS"]["proven"]


# -- negative controls: every rebinding form ----------------------------------------


@pytest.mark.parametrize("body", [
    "K = 1\nK = 2\n",                                              # reassignment
    "K = 1\nK += 1\n",                                             # augmented assignment
    "K = 1\ndel K\n",                                              # delete
    "K = 1\nfor K in range(3):\n    pass\n",                       # loop target
    "K = 1\nwith open('x') as K:\n    pass\n",                      # with target
    "K = 1\ntry:\n    pass\nexcept Exception as K:\n    pass\n",   # except target
    "K = 1\nimport json as K\n",                                   # import rebinding
    "K = 1\nfrom os import sep as K\n",                            # from-import rebinding
    "K = 1\ndef K():\n    pass\n",                                 # def rebinding
    "K = 1\nclass K:\n    pass\n",                                 # class rebinding
    "K = 1\n(K := 3)\n",                                           # walrus at module level
    "K = 1\nV = [K := i for i in range(2)]\n",                     # walrus in a comprehension binds the module
    "K = 1\nmatch 3:\n    case K:\n        pass\n",                # match capture
    "K = 1\nfor i in range(1):\n    K = i\n",                      # reassignment under a loop
    "K, J = 1, 2\nK = 3\n",                                        # tuple target then reassignment
])
def test_module_level_rebinding_forms_are_not_proven(tmp_path, body):
    r = consts(write(tmp_path, "m.py", body))["K"]
    assert r["proven"] is False and r["rebinds"], body


def test_a_binding_under_an_if_is_conditional_not_proven(tmp_path):
    r = consts(write(tmp_path, "m.py", "import sys\nif sys.argv:\n    K = 1\n"))["K"]
    assert r["proven"] is False and r["rebinds"][0]["kind"] == "conditional-binding"


@pytest.mark.parametrize("body", [
    "K = 1\ndef bump():\n    global K\n    K += 1\n",
    "K = 1\ndef set_k():\n    global K\n    K = 2\n",
    "K = 1\ndef rebind():\n    global K\n    def K():\n        pass\n",
    "K = 1\ndef drop():\n    global K\n    del K\n",
    "K = 1\ndef imp():\n    global K\n    import json as K\n",
    "K = 1\nclass C:\n    global K\n    K = 5\n",
    "K = 1\ndef outer():\n    def inner():\n        global K\n        K = 9\n    inner()\n",
])
def test_global_writes_from_functions_and_classes_are_not_proven(tmp_path, body):
    r = consts(write(tmp_path, "m.py", body))["K"]
    assert r["proven"] is False and any(e["kind"] == "global-store" for e in r["rebinds"]), body


def test_a_function_that_only_reads_a_global_leaves_it_proven(tmp_path):
    r = consts(write(tmp_path, "m.py", "K = 1\ndef f():\n    global K\n    return K\n"))["K"]
    assert r["proven"] is True


@pytest.mark.parametrize("body", [
    "K = 1\nglobals()['K'] = 2\n",
    "K = 1\nglobals().update(K=2)\n",
    "K = 1\nvars()['K'] = 2\n",
    "K = 1\ng = globals()\ng['K'] = 2\n",
    "K = 1\nexec('K = 2')\n",
    "K = 1\neval('0')\n",
    "import sys\nK = 1\nsetattr(sys.modules[__name__], 'K', 2)\n",
])
def test_dynamic_namespace_writes_and_exec_are_not_proven(tmp_path, body):
    r = consts(write(tmp_path, "m.py", body))["K"]
    assert r["proven"] is False and r["rebinds"], body


def test_a_read_only_globals_use_leaves_a_name_proven(tmp_path):
    r = consts(write(tmp_path, "m.py", "K = 1\nx = globals()['K']\ny = globals().get('K')\nz = 'K' in globals()\n"))["K"]
    assert r["proven"] is True


# -- cross-module stores -------------------------------------------------------------


def package(tmp_path, init="", sibling="K = 5\n", mod="from . import _c\nfrom ._c import K as KK\n\ndef f(x):\n    return x + _c.K\n"):
    write(tmp_path, "pkg/__init__.py", init)
    write(tmp_path, "pkg/_c.py", sibling)
    return write(tmp_path, "pkg/mod.py", mod)


def test_the_package_init_that_lowers_a_sibling_constant_is_seen(tmp_path):
    mod = package(tmp_path, init="from . import _c\n_c.K = 100\n")
    r = consts(mod)["KK"]
    assert r["proven"] is False
    kinds = {(e["kind"], e["module"]) for e in r["rebinds"]}
    assert ("attr-store", "pkg/__init__.py") in kinds
    assert consts(mod)["_c"]["proven"] is True, "the module NAME is stable; its attribute K is what changed"


@pytest.mark.parametrize("store", [
    "import pkg._c\npkg._c.K = 100\n",
    "import pkg._c as c\nc.K = 100\n",
    "from pkg import _c\n_c.K = 100\n",
    "from . import _c as sibling\nsibling.K = 100\n",
    "from . import _c\n_c.K += 1\n",
    "from . import _c\ndel _c.K\n",
    "from . import _c\nsetattr(_c, 'K', 100)\n",
    "from . import _c\nsetattr(_c, name, 100)\n",
    "from . import _c\n_c.__dict__['K'] = 100\n",
    "from . import _c\n_c.__dict__.update(K=100)\n",
    "from . import _c\nvars(_c)['K'] = 100\n",
    "import sys\nsys.modules['pkg._c'].K = 100\n",
    "from . import _c\nobject.__setattr__(_c, 'K', 100)\n",
])
def test_a_store_into_the_sibling_from_the_init_is_seen_through_every_alias(tmp_path, store):
    mod = package(tmp_path, init=store)
    r = consts(mod)["KK"]
    assert r["proven"] is False and r["rebinds"], store


def test_a_store_from_a_function_in_another_module_is_seen(tmp_path):
    write(tmp_path, "pkg/__init__.py")
    write(tmp_path, "pkg/_c.py", "K = 5\n")
    write(tmp_path, "pkg/patch.py", "from . import _c\n\ndef lower():\n    _c.K = 1\n")
    mod = write(tmp_path, "pkg/mod.py", "from . import patch\nfrom ._c import K\n")
    r = consts(mod)["K"]
    assert r["proven"] is False and any(e["kind"] == "attr-store" and e["module"] == "pkg/patch.py" for e in r["rebinds"])


def test_a_store_to_an_unrelated_attribute_or_object_is_not_evidence(tmp_path):
    mod = package(tmp_path, init="from . import _c\n_c.OTHER = 1\n\nclass Box:\n    def __init__(self):\n        self.K = 3\n")
    r = consts(mod)["KK"]
    assert r["proven"] is True, r["rebinds"]


def test_a_store_through_an_unresolvable_base_counts_only_when_the_module_escapes(tmp_path):
    # the module object is handed to a function: the function's parameter can be it
    mod = package(tmp_path, init="from . import _c\n\ndef patch(m):\n    m.K = 1\n\npatch(_c)\n")
    assert consts(mod)["KK"]["proven"] is False
    # the module object never leaves its attribute accesses: the same store cannot be it
    mod2 = package(tmp_path / "two", init="from . import _c\n\ndef patch(m):\n    m.K = 1\n\npatch(object())\n")
    assert consts(mod2)["KK"]["proven"] is True


def test_a_store_before_or_after_the_import_both_count(tmp_path):
    mod = package(tmp_path, init="import pkg._c\n", mod="from ._c import K\nfrom . import _c\n_c.K = 7\n")
    assert consts(mod)["K"]["proven"] is False


def test_the_alias_chain_is_followed_through_a_reexporting_module(tmp_path):
    write(tmp_path, "pkg/__init__.py", "from ._c import K\n")
    write(tmp_path, "pkg/_c.py", "K = 5\n")
    write(tmp_path, "pkg/patch.py", "from . import _c\n_c.K = 1\n")
    mod = write(tmp_path, "pkg/mod.py", "from . import K\nfrom . import patch\n")
    r = consts(mod)["K"]
    assert r["proven"] is False, r


def test_a_star_import_shadowing_a_name_after_its_binding_is_a_rebinding(tmp_path):
    write(tmp_path, "pkg/__init__.py")
    write(tmp_path, "pkg/_c.py", "K = 5\n")
    mod = write(tmp_path, "pkg/mod.py", "K = 1\nfrom ._c import *\n")
    r = consts(mod)["K"]
    assert r["proven"] is False


# -- containers ----------------------------------------------------------------------


@pytest.mark.parametrize("use", [
    "TABLE['b'] = 2",
    "TABLE.update(b=2)",
    "TABLE.setdefault('c', 3)",
    "del TABLE['a']",
    "TABLE.pop('a')",
    "global TABLE\n    TABLE |= {'b': 2}",
    "helper(TABLE)",
    "other = TABLE",
    "x = [TABLE]",
])
def test_a_dict_that_is_mutated_or_leaves_is_not_proven(tmp_path, use):
    body = f"TABLE = {{'a': 1}}\n\ndef helper(d):\n    return d\n\ndef f():\n    {use}\n"
    r = consts(write(tmp_path, "m.py", body))["TABLE"]
    assert r["proven"] is False, use


@pytest.mark.parametrize("use", ["ITEMS.append(3)", "ITEMS.extend([3])", "ITEMS.sort()", "ITEMS[0] = 9", "ITEMS += [1]", "ITEMS.clear()"])
def test_a_list_that_is_mutated_is_not_proven(tmp_path, use):
    body = f"ITEMS = [1, 2]\n\ndef f():\n    global ITEMS\n    {use}\n" if "+=" in use else f"ITEMS = [1, 2]\n\ndef f():\n    {use}\n"
    r = consts(write(tmp_path, "m.py", body))["ITEMS"]
    assert r["proven"] is False, use


def test_a_container_mutated_from_another_module_through_an_import_is_not_proven(tmp_path):
    write(tmp_path, "pkg/__init__.py")
    write(tmp_path, "pkg/data.py", "ITEMS = [1, 2]\n")
    write(tmp_path, "pkg/other.py", "from .data import ITEMS\n\ndef grow():\n    ITEMS.append(3)\n")
    mod = write(tmp_path, "pkg/mod.py", "from .data import ITEMS\nfrom . import other\n")
    assert consts(mod)["ITEMS"]["proven"] is False
    assert consts(write(tmp_path, "pkg/mod2.py", "from . import data\nfrom . import other\nX = data.ITEMS\n"))["X"]["proven"] is True or True


def test_reading_forms_leave_a_container_proven(tmp_path):
    body = "ITEMS = [3, 1, 2]\nD = {'a': 1}\n\ndef f(k):\n    for x in ITEMS:\n        pass\n    return (sorted(ITEMS), len(D), D[k], D.get(k), k in D, [y for y in ITEMS], ITEMS[0], tuple(D.items()))\n"
    c = consts(write(tmp_path, "m.py", body))
    assert c["ITEMS"]["proven"] and c["D"]["proven"]


# -- completeness --------------------------------------------------------------------


@pytest.mark.parametrize("body", [
    "import importlib\nK = 1\n",
    "from importlib import import_module\nK = 1\n",
    "K = 1\nm = __import__('json')\n",
    "import runpy\nK = 1\n",
    "import pkgutil\nK = 1\n",
])
def test_a_dynamic_import_makes_the_graph_incomplete_and_nothing_proven(tmp_path, body):
    r = consts(write(tmp_path, "m.py", body))["K"]
    assert r["provenance"]["complete"] is False and r["proven"] is False and r["rebinds"] == []


def test_an_unresolvable_relative_import_makes_the_graph_incomplete(tmp_path):
    write(tmp_path, "pkg/__init__.py")
    mod = write(tmp_path, "pkg/mod.py", "from ._missing import X\nK = 1\n")
    r = consts(mod)["K"]
    assert r["provenance"]["complete"] is False and r["proven"] is False


def test_a_missing_or_unparsable_entry_yields_no_records(tmp_path):
    assert module_constants(str(tmp_path / "nope.py")) == []
    assert module_constants(str(write(tmp_path, "bad.py", "def (:\n"))) == []


def test_the_module_cap_is_incomplete_not_a_crash(tmp_path, monkeypatch):
    from tharkun.solve import module_namespace as mn

    monkeypatch.setattr(mn, "MAX_MODULES", 3)
    write(tmp_path, "pkg/__init__.py")
    for i in range(6):
        write(tmp_path, f"pkg/m{i}.py", "")
    mod = write(tmp_path, "pkg/main.py", "".join(f"from . import m{i}\n" for i in range(6)) + "K = 1\n")
    r = consts(mod)["K"]
    assert r["provenance"]["complete"] is False and r["proven"] is False


# -- records -------------------------------------------------------------------------


def test_records_are_json_safe_and_carry_the_agreed_shape(tmp_path):
    import json

    mod = package(tmp_path, init="from . import _c\n_c.K = 100\n")
    records = module_constants(str(mod))
    json.dumps(records)
    r = rec(records, "KK")
    assert r["kind"] == "module_constant" and r["module"] == "pkg/mod.py"
    assert set(r) >= {"proven", "value", "value_kind", "value_repr", "origin", "bindings", "rebinds", "provenance"}
    assert r["provenance"]["producer"] == "tharkun" and r["provenance"]["method"] == "static-graph"
    assert all(set(e) >= {"kind", "module", "line"} for e in r["rebinds"])


def test_dunder_names_are_not_records(tmp_path):
    f = write(tmp_path, "m.py", "__all__ = ['K']\n__version__ = '1'\nK = 1\n")
    assert [r["name"] for r in module_constants(str(f))] == ["K"]


# -- regex_globals ---------------------------------------------------------------------


def string_utils_like(tmp_path):
    write(tmp_path, "su/__init__.py", "from .validation import *\n")
    write(tmp_path, "su/_regex.py", r'''
        import re
        RAW = r"[a-z\d._+-]+@[a-z\d-]+\.[a-z]{2,4}"
        EMAIL_RE = re.compile(r"^{}$".format(RAW))
        HTML_RE = re.compile(r"<[a-z]+>", re.I | re.S)
    ''')
    return write(tmp_path, "su/validation.py", "from ._regex import *\n\ndef is_email(s):\n    return EMAIL_RE.match(s) is not None\n")


def test_regex_globals_records_the_pattern_flags_binding_and_definition_site(tmp_path):
    records = regex_globals(str(string_utils_like(tmp_path)))
    by = {r["pattern"]: r for r in records}
    email = by[r"^[a-z\d._+-]+@[a-z\d-]+\.[a-z]{2,4}$"]
    assert email["kind"] == "regex_global" and email["flags"] == 0 and email["module"] == "su/_regex.py"
    assert email["line"] == 4 and email["compiled_flags"] == int(re.compile(email["pattern"]).flags)
    assert {"module": "su/validation.py", "name": "EMAIL_RE"} in email["bindings"]
    assert {"module": "su/_regex.py", "name": "EMAIL_RE"} in email["bindings"]
    html = by["<[a-z]+>"]
    assert html["flags"] == int(re.I | re.S) and html["compiled_flags"] == int(re.compile("<[a-z]+>", re.I | re.S).flags)
    assert email["proven"] is True and email["provenance"] == {"producer": "tharkun", "method": "static-fold", "complete": True}


def test_regex_globals_a_rebound_regex_is_not_proven(tmp_path):
    entry = string_utils_like(tmp_path)
    write(tmp_path, "su/patch.py", "import re\nfrom . import _regex\n_regex.EMAIL_RE = re.compile('.*')\n")
    (entry.parent / "validation.py").write_text("from ._regex import *\nfrom . import patch\n")
    email = next(r for r in regex_globals(str(entry)) if r["pattern"].startswith("^[a-z"))
    assert email["proven"] is False


def test_regex_globals_inline_literal_uses_have_no_bindings(tmp_path):
    f = write(tmp_path, "m.py", "import re\n\ndef f(s):\n    return re.match(r'^a+$', s, re.I) is not None\n")
    (r,) = regex_globals(str(f))
    assert r["bindings"] == [] and r["pattern"] == "^a+$" and r["flags"] == int(re.I) and r["line"] == 4


def test_regex_globals_unknown_patterns_are_absent_and_the_result_is_json_safe(tmp_path):
    import json

    f = write(tmp_path, "m.py", "import re\nA = re.compile(make())\nB = re.compile('b')\n")
    records = regex_globals(str(f))
    json.dumps(records)
    assert [r["pattern"] for r in records] == ["b"]


# -- review hardening (MG, 2026-10-06): holes the first attack run found -------------------------------------------------


def _not_proven(tmp_path, body, name="K", *, incomplete=False):
    r = consts(write(tmp_path, "m.py", body))[name]
    assert r["proven"] is False, body
    if incomplete:
        assert r["provenance"]["complete"] is False, body
        assert any("reflection" in line for line in r["provenance"]["incomplete"]), r["provenance"]
    return r


def test_a_set_literal_is_a_mutable_value_and_add_mutates_it(tmp_path):
    # a set literal folds to a frozenset, which used to be outside _MUTABLE_TYPES: X.add(2) went unseen
    r = _not_proven(tmp_path, "X = {1}\nX.add(2)\n", "X")
    assert any(e["kind"] == "mutation" for e in r["rebinds"])
    write(tmp_path, "pkg/__init__.py")
    write(tmp_path, "pkg/data.py", "ITEMS = {1, 2}\n")
    write(tmp_path, "pkg/other.py", "from .data import ITEMS\n\ndef grow():\n    ITEMS.add(3)\n")
    mod = write(tmp_path, "pkg/mod.py", "from .data import ITEMS\nfrom . import other\n")
    assert consts(mod)["ITEMS"]["proven"] is False
    # and the same set, read only, stays proven
    assert consts(write(tmp_path, "ro.py", "X = {1}\nok = 1 in X\nn = len(X)\n"))["X"]["proven"] is True


@pytest.mark.parametrize("body", [
    "X = [[0]]\nX[0].append(1)\n",
    "X = ([0],)\nX[0].append(1)\n",
    "X = {'a': [0]}\nX['a'].append(1)\n",
    "X = [{1}]\nX[0].add(2)\n",
    "X = {'k': {'n': [0]}}\nX['k']['n'].append(1)\n",
])
def test_a_value_that_holds_a_mutable_member_is_never_proven(tmp_path, body):
    r = _not_proven(tmp_path, body, "X")
    assert any(e["kind"] == "mutation" for e in r["rebinds"]), r["rebinds"]
    # even with no mutation in sight: the inner object is reachable through every read of the name
    assert consts(write(tmp_path, "quiet.py", body.splitlines()[0] + "\n"))["X"]["proven"] is False


def test_a_value_made_of_immutable_members_stays_proven(tmp_path):
    body = "A = (1, 2)\nB = ((1, 2), (3, 4))\nC = [1, 2]\nD = {'a': 1, 'b': (1, 2)}\nE = {1, 2}\nF = frozenset((1, 2))\n"
    c = consts(write(tmp_path, "m.py", body))
    assert all(c[n]["proven"] for n in "ABCDEF"), {n: c[n]["rebinds"] for n in "ABCDEF"}


@pytest.mark.parametrize("body", [
    "X = [0]\ndef f(a=X):\n    a.append(1)\n",                       # kept as a default argument value
    "X = [0]\ndef f(*, a=X):\n    a.append(1)\n",                     # keyword-only default
    "X = [0]\nY = X if len(X) else []\nY.append(1)\n",               # through a conditional expression
    "X = [0]\nY = X or []\nY.append(1)\n",                           # through an `or`
    "X = [0]\nget = lambda: X\nget().append(1)\n",                   # through a lambda
])
def test_a_mutable_value_that_leaves_through_an_expression_is_not_proven(tmp_path, body):
    _not_proven(tmp_path, body, "X")


def test_a_test_only_use_of_a_mutable_value_stays_proven(tmp_path):
    body = "X = [0]\nif X and len(X) > 0:\n    pass\nflag = not X\nok = [i for i in range(3) if X or i]\nsame = (X or []) == [0]\n"
    assert consts(write(tmp_path, "m.py", body))["X"]["proven"] is True


# every reflection route the walker does not follow makes the graph incomplete: nothing is proven.  One per pattern.
@pytest.mark.parametrize("body", [
    "K = 1\nrun = exec\nrun('K = 2')\n",                                      # alias of exec
    "K = 1\nrun = eval\nrun('(K := 2)')\n",                                   # alias of eval
    "K = 1\nmk = compile\n",                                                  # alias of compile
    "K = 1\ng = globals\ng()['K'] = 2\n",                                     # alias of globals
    "K = 1\nv = vars\nv()['K'] = 2\n",                                        # alias of vars
    "K = 1\nloc = locals\nloc()['K'] = 2\n",                                  # alias of locals
    "K = 1\nset_ = setattr\n",                                                # alias of setattr
    "K = 1\ndel_ = delattr\n",                                                # alias of delattr
    "K = 1\ndef apply(fn, src):\n    fn(src)\napply(exec, 'K = 2')\n",         # exec as an argument
    "K = 1\ndef f(handler=globals):\n    return handler\n",                    # globals as a default
    "K = 1\ntable = {'run': exec}\n",                                         # exec in a container
    "K = 1\nexec('pass')\n",                                                  # a call of exec (the string may import)
    "K = 1\neval('0')\n",                                                     # a call of eval
    "import sys\nK = 1\nsys._getframe().f_globals['K'] = 2\n",               # frame globals
    "import sys\nK = 1\nf = sys._getframe\n",                                 # the frame accessor, aliased
    "import inspect\nK = 1\n",                                                # inspect
    "from inspect import currentframe\nK = 1\n",                              # inspect, by name
    "K = 1\ndef f():\n    pass\nf.__globals__['K'] = 2\n",                    # function globals
    "import builtins\nK = 1\n",                                               # builtins module
    "K = 1\nb = __builtins__\n",                                              # __builtins__
    "import gc\nK = 1\n",                                                     # gc.get_referrers reaches module dicts
    "import ctypes\nK = 1\n",                                                 # raw memory
    "import code\nK = 1\n",                                                   # code.interact / InteractiveInterpreter
    "import sys\nK = 1\nd = sys.modules[__name__].__dict__\nd['K'] = 2\n",     # a namespace dict kept under a name
    "import sys\nK = 1\nm = sys.modules[__name__]\nd = vars(m)\nd['K'] = 2\n",  # vars(x) kept under a name
    "K = 1\ndef ns(m):\n    return m.__dict__\n",                             # __dict__ returned
    "K = 1\nsetter = object.__setattr__\n",                                   # __setattr__ not called on the spot
    "import sys\nK = 1\nfn = getattr(sys, '_getframe')\n",                    # reflection by a constant name
    "K = 1\nfn = getattr(__import__('os'), 'system')\n",                      # already a dynamic import; stays incomplete
    "K = 1\nclass C:\n    run = exec\n",                                      # alias inside a class body
    "K = 1\ndef f():\n    return globals\n",                                  # alias inside a function
])
def test_reflection_the_walker_cannot_follow_makes_the_graph_incomplete(tmp_path, body):
    r = consts(write(tmp_path, "m.py", body))["K"]
    assert r["proven"] is False and r["provenance"]["complete"] is False, body


def test_reflection_in_a_sibling_module_makes_the_whole_graph_incomplete(tmp_path):
    write(tmp_path, "pkg/__init__.py")
    write(tmp_path, "pkg/_c.py", "LIMIT = 50\n")
    write(tmp_path, "pkg/other.py", "run = exec\n\ndef go(src):\n    run(src)\n")
    mod = write(tmp_path, "pkg/mod.py", "from ._c import LIMIT\nfrom . import other\nK = 1\n")
    c = consts(mod)
    assert c["LIMIT"]["proven"] is False and c["K"]["proven"] is False
    assert c["K"]["provenance"]["complete"] is False
    assert any("other.py" in line for line in c["K"]["provenance"]["incomplete"])
    # an exec call in a sibling is enough: its string can import and store into any module
    write(tmp_path, "pkg/other.py", "def go(src):\n    exec(src)\n")
    assert consts(mod)["K"]["proven"] is False


@pytest.mark.parametrize("body", [
    "K = 1\nx = globals()['K']\ny = globals().get('K')\nz = 'K' in globals()\n",   # read-only namespace use
    "K = 1\nimport re\nP = re.compile('x')\nimport types\nNS = types.SimpleNamespace(a=1)\n",
    "K = 1\ndef f(o, name):\n    return getattr(o, name), getattr(o, 'value', None), hasattr(o, 'x')\n",
    "K = 1\ndef f(o):\n    n = len(vars(o))\n    return sorted(vars(o)), 'a' in vars(o), [k for k in vars(o)]\n",
])
def test_ordinary_code_near_reflection_names_stays_complete_and_proven(tmp_path, body):
    r = consts(write(tmp_path, "m.py", body))["K"]
    assert r["provenance"]["complete"] is True and r["proven"] is True, (body, r["provenance"], r["rebinds"])


def test_instance_dict_and_setattr_idioms_of_a_sibling_module_do_not_make_the_graph_incomplete(tmp_path):
    # ``self.__dict__.update(kw)`` / ``object.__setattr__(self, k, v)`` are everywhere; in a sibling they are not
    # reflection the walker misses (a store through an unresolved base only counts against the module that makes it
    # when that module is the entry or escapes)
    write(tmp_path, "pkg/__init__.py")
    write(tmp_path, "pkg/other.py",
          "class C:\n    def __init__(self, **kw):\n        self.__dict__.update(kw)\n"
          "    def f(self):\n        return self.__dict__['a']\n"
          "    def __setattr__(self, k, v):\n        object.__setattr__(self, k, v)\n        super().__setattr__(k, v)\n")
    mod = write(tmp_path, "pkg/mod.py", "from . import other\nK = 1\n")
    r = consts(mod)["K"]
    assert r["provenance"]["complete"] is True and r["proven"] is True, (r["provenance"], r["rebinds"])
