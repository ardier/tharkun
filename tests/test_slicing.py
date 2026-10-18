"""Two non-benchmark programs and reachability/effect controls for [internal ref]."""
import ast
import subprocess
import sys
import textwrap

import pytest

from tharkun.solve.slicing import slice_for, slice_obligation


def obligation(source, marker="assert", function="<module>", value="True"):
    line = next(i for i, s in enumerate(source.splitlines(), 1) if marker in s)
    column = len(source.splitlines()[line - 1]) - len(source.splitlines()[line - 1].lstrip()) + 1
    return {"exception": "AssertionError", "invariant": {
        "type": "location_invariant", "format": "python_expression",
        "location": {"file_name": "toy.py", "line": line, "column": column, "function": function},
        "value": value}}


def run(source):
    p = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=3)
    return p.returncode, p.stdout, p.stderr.splitlines()[-1:]  # fresh ORIGINAL/slice processes


def sliced(source, **kwargs):
    return slice_obligation(source, obligation(source, **kwargs), filename="toy.py")


def test_alias_mutation_negative_control():
    source = textwrap.dedent("""\
        dead = 19
        def mutate(alias):
            alias[0] = 0
        box = [1]
        mutate(box)
        assert box[0] != 0
    """)
    result = sliced(source, value="box[0] != 0")
    assert result.removed == ((1, 0),)
    assert run(source) == run(result.source)
    assert run(result.source)[0] != 0
    # Removing the aliasing call changes the verdict. It is a dependency even
    # though the call binds no name read by the safety condition.
    assert run(source.replace("mutate(box)", "pass"))[0] == 0


def test_exception_negative_control():
    source = textwrap.dedent("""\
        dead = 23
        box = [1]
        try:
            unused = 1 // 0
        except ZeroDivisionError:
            box[0] = 0
        assert box[0] != 0
    """)
    result = sliced(source, value="box[0] != 0")
    assert result.removed == ((1, 0),)
    assert run(source) == run(result.source)
    assert run(result.source)[0] != 0
    assert run(source.replace("unused = 1 // 0", "unused = 1"))[0] == 0


def test_two_small_programs_control_calls_loops_and_exceptions():
    # Unrelated mutually recursive functions can go; effectful call/control
    # closure, including the global read BEFORE its assignment in source, stays.
    first = textwrap.dedent("""\
        def spare_a():
            return spare_b()
        def spare_b():
            return spare_a()
        def step(x):
            return x + increment
        increment = 1
        unused = 7
        total = 0
        for i in range(5):
            if i == 2:
                continue
            total = step(total)
            if total == 3:
                break
        assert total == 3
    """)
    second = textwrap.dedent("""\
        def unused_function():
            raise RuntimeError()
        def checked(x):
            junk = 17
            try:
                return 10 // x
            except ZeroDivisionError:
                return 9
            finally:
                events.append(x)
        events = []
        a = checked(0)
        b = checked(2)
        assert (a, b, events) == (9, 5, [0, 2])
    """)
    for source in (first, second):
        result = sliced(source)
        assert result.removed
        assert run(source) == run(result.source) == (0, "", [])


def test_infinite_loop_and_early_raise_remain():
    for prefix in ("while True:\n    pass\n", "raise ValueError()\n"):
        source = "dead = 3\n" + prefix + "assert False\n"
        result = sliced(source)
        assert prefix in result.source
        assert result.removed == ((1, 0),)


def test_anchor_in_uncalled_function_and_safety_names_stay():
    source = "keep = 3\ndrop = 4\ndef uncalled():\n    assert keep == 3\n"
    result = sliced(source, function="uncalled", value="keep == 3")
    assert result.removed == ((2, 0),)
    assert "def uncalled" in result.source


@pytest.mark.parametrize("effect", [
    "unused = touch()", "unused = (1 // 0)", "unused = thing.value",
    "unused = thing[0]", "thing[0] = 0", "thing.append(0)",
    "for unused in range(5):\n    pass", "while True:\n    pass",
    "try:\n    raise ValueError()\nexcept ValueError:\n    pass",
    "def unused(x=touch()):\n    pass", "@touch\ndef unused():\n    pass",
])
def test_uncertain_effects_are_roots(effect):
    source = effect + "\nassert True\n"
    result = sliced(source)
    assert effect in result.source


def test_reassignment_destructor_class_namespace_and_parameters_stay():
    source = textwrap.dedent("""\
        events = []
        class C:
            member = 7
            def __del__(self):
                events.append(1)
        def f(x):
            x = 0
        old = C()
        old = 0
        f(C())
        assert events == [1, 1]
    """)
    result = sliced(source)
    assert "member = 7" in result.source
    assert "x = 0" in result.source
    assert "old = 0" in result.source
    assert run(source) == run(result.source)


@pytest.mark.parametrize("prefix", [
    "import inspect\n", "import sys\n", "names = dir()\n",
    "names = locals()\n", "f = globals\n", "names = object.__dict__\n",
    "text = open(__file__).read()\n",
])
def test_namespace_inspection_refuses(prefix):
    source = prefix + "dead = 7\nassert True\n"
    result = sliced(source)
    assert result.source == source
    assert result.reason


def test_input_untouched_flat_api_and_single_implementation():
    from tharkun.sv._symexec import slice_for as legacy
    assert legacy is slice_for
    tree = ast.parse("unused = 3\nused = 4\nassert used == 4\n")
    before = ast.dump(tree, include_attributes=True)
    result = slice_for(tree.body, {"used"})
    assert result == tree.body  # a prefix cannot certify its incoming bindings
    assert result[0] is tree.body[0]
    assert ast.dump(tree, include_attributes=True) == before
    slice_for(tree.body, {"used"}, anchors=[tree.body[-1]], closed=True)
    assert ast.dump(tree, include_attributes=True) == before


def test_coordinates_and_wrong_location():
    source = "dead = 3; assert True\n"
    entry = obligation(source)
    entry["invariant"]["location"]["column"] = 11
    result = slice_obligation(source, entry, filename="toy.py")
    assert result.source == "0       ; assert True\n"
    assert result.source.index("assert") == source.index("assert")
    entry["invariant"]["location"]["column"] = 2
    with pytest.raises(ValueError):
        slice_obligation(source, entry, filename="toy.py")


def test_unicode_location_refuses_pruning_without_moving_the_site():
    source = "é = 3; assert True\n"
    entry = obligation(source)
    entry["invariant"]["location"]["column"] = 8
    result = slice_obligation(source, entry, filename="toy.py")
    assert result.source == source


def test_absint_and_search_consume_the_slice(tmp_path, monkeypatch):
    from tharkun.solve.absint import prove
    from tharkun.solve import slicing
    from tharkun.sv import concolic

    source = "dead = 7\nassert True\n"
    entry = obligation(source)
    real = slicing.slice_obligation
    calls = []
    def observed(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result)
        return result
    monkeypatch.setattr(slicing, "slice_obligation", observed)
    result = prove(source, ("AssertionError",), filename="toy.py", obligations=[entry])
    assert result.verdict == "TRUE"
    assert result.source_sha == calls[-1].source_sha256
    assert calls[-1].removed == ((1, 0),)
    # Harness construction consumes the sliced text, without changing the
    # original file which the engine uses for fresh FALSE replay.
    path = tmp_path / "toy.py"
    path.write_text(source)
    entry["invariant"]["location"]["file_name"] = str(path)
    received = []
    monkeypatch.setattr(concolic, "build_harness", lambda text, *a: received.append(text))
    concolic.solve(str(path), ("AssertionError",), budget_seconds=1, obligations=[entry])
    assert received == [calls[-1].source]
    assert path.read_text() == source


def test_implicit_dunder_bindings_are_retained():
    source = "__annotations__ = 3\nassert True\n"
    assert sliced(source).source == source


def test_finalizer_scheduling_refuses_even_an_unused_function():
    source = ("class C:\n    def __del__(self):\n        raise ValueError()\n"
              "def unused():\n    pass\nassert True\n")
    result = sliced(source)
    assert result.source == source
    assert "finalizer" in result.reason


def test_multiline_assignment_with_semicolon_falls_back_to_original():
    source = "dead = (\n    3\n); assert True\n"
    entry = obligation(source)
    entry["invariant"]["location"]["column"] = 4
    result = slice_obligation(source, entry, filename="toy.py")
    assert result.source == source
    assert not result.removed
    assert "statement boundaries" in result.reason
    assert run(source) == run(result.source) == (0, "", [])


def test_implicit_protocol_call_via_dict_star_unpack_retains_class_method():
    # Regression for a soundness bug reported against a sibling census/resolver
    # ([internal ref] hotfix #51): `dict(**C())` invokes `C.keys()` with no source-level
    # name reference to "keys". This slicer never makes class methods
    # candidates in the first place (every ClassDef child is popped from
    # `candidates`, and `defs` only ever looks at the module-level body), so
    # `keys` cannot be pruned regardless of whether anything names it.
    source = textwrap.dedent("""\
        class C:
            def keys(self):
                return ["a"]
            def __getitem__(self, key):
                return 1
        merged = dict(**C())
        assert merged == {"a": 1}
    """)
    result = sliced(source)
    assert "def keys" in result.source
    assert "def __getitem__" in result.source
    assert run(source) == run(result.source)


def test_decorated_function_with_definition_time_effect_is_never_pruned():
    # Regression for the sibling decorator lexical-scope defect (assigned to
    # SA's 1057/decorator-scope-fix): this slicer never resolves a decorator
    # expression's scope to decide pruning at all -- `_inert_def` refuses any
    # function with a non-empty decorator_list outright, so a decorated
    # definition's own effect (and the decorator name it reads) always stays,
    # whether or not the decorated function is ever called again.
    source = textwrap.dedent("""\
        events = []
        def track(fn):
            events.append(fn.__name__)
            return fn
        @track
        def unused():
            pass
        assert events == ["unused"]
    """)
    result = sliced(source)
    assert "@track" in result.source
    assert "def unused" in result.source
    assert run(source) == run(result.source)


def test_negative_control_absint_broken_assertion_never_produces_true():
    # Negative control: when a guarantee is broken, slicing must not prune
    # the failure path and absint prove() must never produce TRUE.
    from tharkun.solve.absint import prove

    source = textwrap.dedent("""        dead = 7
        x = 0
        assert x != 0
    """)
    entry = obligation(source, value="x != 0")
    result = prove(source, ("AssertionError",), filename="toy.py", obligations=[entry])
    assert result.verdict != "TRUE"


def test_negative_control_absint_broken_reassignment_never_produces_true():
    from tharkun.solve.absint import prove

    source = textwrap.dedent("""        dead = 7
        x = 1
        x = 0
        assert x != 0
    """)
    entry = obligation(source, value="x != 0")
    result = prove(source, ("AssertionError",), filename="toy.py", obligations=[entry])
    assert result.verdict != "TRUE"


def test_negative_control_absint_broken_aliasing_never_produces_true():
    from tharkun.solve.absint import prove

    source = textwrap.dedent("""        dead = 7
        def mutate(box):
            box[0] = 0
        b = [1]
        mutate(b)
        assert b[0] != 0
    """)
    entry = obligation(source, value="b[0] != 0")
    result = prove(source, ("AssertionError",), filename="toy.py", obligations=[entry])
    assert result.verdict != "TRUE"


def test_negative_control_prove_dynamic_broken_guarantee_never_produces_true(tmp_path):
    from tharkun.sv.prove import prove_dynamic

    p = tmp_path / "broken.py"
    p.write_text(textwrap.dedent("""        dead = 7
        x = 0
        assert x != 0
    """))
    result = prove_dynamic(str(p), ("AssertionError",), python=sys.executable, remaining=lambda: 10.0)
    assert result is None or result.verdict != "TRUE"


def test_analysis_slice_accepts_exception_types_and_strings():
    from tharkun.solve.slicing import analysis_slice

    source = "dead = 7\nassert True\n"
    s1 = analysis_slice(source, ("AssertionError",), filename="toy.py")
    s2 = analysis_slice(source, (AssertionError,), filename="toy.py")
    assert s1.source == s2.source


# -- [internal ref]: analysis_slice must never anchor on (or raise from) the census's own
# -- "$coverage" module placeholder (line 1, col 1, "<module>", value "False").
# -- A leading comment, docstring or license header pushes the real site off
# -- line 1; the old code treated whichever entry obligation_census listed
# -- first as the anchor and crashed when it was the placeholder and line 1
# -- was not a single statement (MG-S, on the real merge result).

_REAL_SITE_AFTER = "\ndef f(x):\n    return 1 / x\n\nf(0)\n"


@pytest.mark.parametrize("prefix", [
    "# a leading comment\n",
    '"""Module docstring."""\n',
    "# Copyright 2026 Example Corp.\n# Licensed under the MIT License.\n# SPDX-License-Identifier: MIT\n",
])
def test_analysis_slice_skips_the_placeholder_anchor_for_a_real_site_below(prefix):
    from tharkun.solve.slicing import analysis_slice

    source = prefix + _REAL_SITE_AFTER
    s = analysis_slice(source, ("ZeroDivisionError",), filename="toy.py")
    # Anchored on the real division site, not the line-1 placeholder: a
    # successful slice, never a raise (this used to raise ValueError when
    # the prefix pushed the real site past line 1).
    assert s.reason == ""
    assert "1 / x" in s.source


def test_analysis_slice_falls_back_when_only_the_placeholder_is_open():
    from tharkun.solve.slicing import analysis_slice

    source = "x = 1\nprint(x)\n"  # no ZeroDivisionError site anywhere
    s = analysis_slice(source, ("ZeroDivisionError",), filename="toy.py")
    assert s.source == source
    assert s.reason != ""


def test_analysis_slice_falls_back_on_a_non_valueerror_anchor_failure():
    from tharkun.solve.slicing import analysis_slice

    source = "def f(x):\n    return 1 / x\n\nf(0)\n"
    # slice_obligation's own predicate parse (ast.parse(value, mode="eval"))
    # raises SyntaxError, not ValueError, for a malformed predicate -- exactly
    # the "any other unexpected failure" _slice_or_fall_back must also catch.
    entry = {"invariant": {"type": "location_invariant", "format": "python_expression",
             "location": {"file_name": "toy.py", "line": 2, "column": 5, "function": "f"},
             "value": "1 +"}}
    s = analysis_slice(source, ("ZeroDivisionError",), filename="toy.py", obligations=[entry])
    assert s.source == source
    assert "SyntaxError" in s.reason
