import ast

import pytest

from tharkun.solve.census import obligation_census, site_census
from tharkun.solve.obligations import Invariant, witness
from tharkun.solve.predicates import InvalidPredicate, check_predicate, parse_predicate
from tharkun.sv.witness import dumps, loads


def test_witness_round_trip():
    src = "a = 7\nb = 2\nx = a / b\n"
    census = obligation_census(src, ["ZeroDivisionError"], "task.py")
    assert census.complete and len(census.obligations) == 1
    ob = census.obligations[0]
    assert ob.identity[1:] == ("$.body[2].value", "ZeroDivisionError")
    assert ob.invariant.line == 3 and ob.invariant.column == 1
    check_predicate(ob.invariant.value, {"a": "object", "b": "object"})
    doc = witness(src, "task.py", "CHECK( init(main()), ! uncaught(ZeroDivisionError))", [ob.invariant])
    assert loads(dumps(doc)) == doc
    assert doc[0]["metadata"]["format_version"] == "2.3"
    assert doc[0]["metadata"]["task"]["language"] == "Python"
    assert "data_model" not in doc[0]["metadata"]["task"]
    assert doc[0]["metadata"]["creation_time"].endswith("Z")
    assert Invariant.from_entry(ob.invariant.entry()) == ob.invariant
    assert doc[0]["metadata"]["task"]["input_file_hashes"]["task.py"] == ob.source_sha256


def test_census_inventory_is_untruncated_class_specific_and_deduplicated():
    src = "a = [1]\ni = 0\n" + "x = a[i]\n" * 12
    census = obligation_census(src, ["KeyError", "IndexError"])
    assert census.complete
    assert len(census.obligations) == len({o.identity for o in census.obligations}) == 24
    assert len(site_census(src, ["IndexError"]).sites) == 8
    for ob in census.obligations:
        if ob.exception == "IndexError":
            check_predicate(ob.invariant.value, {"a": "object", "i": "object"})


def test_refused_empty_census_is_a_gap():
    # "import xml" ([internal ref]: "os" itself is no longer an example of an unresolvable import --
    # it is now a census.py::MODULES entry, same as math/random/re/...)
    for src, exc in [("x = 1", "TypeError"), ("import xml", "IndexError"), ("!", "AssertionError")]:
        result = obligation_census(src, [exc])
        assert not result.complete and result.coverage_gaps
        assert result.obligations[0].invariant.value == "False"
        assert result.obligations[0].gap
    assert obligation_census("x = 1", ["ZeroDivisionError"]).complete
    assert not obligation_census("x = 1", ["ZeroDivisionError"]).obligations


def test_original_statement_locations_and_nested_evaluation_gaps():
    src = "def f():\n    é = 1; x = 1 / é\nf()\n"
    ob, = obligation_census(src, ["ZeroDivisionError"]).obligations
    assert (ob.invariant.line, ob.invariant.column, ob.invariant.function) == (2, 12, "f")
    src = "def f():\n    return 1\nb = 1\nx = f() + 1 / b\n"
    ob, = obligation_census(src, ["ZeroDivisionError"]).obligations
    assert ob.gap and ob.invariant.value == "False"


@pytest.mark.parametrize("value,types", [
    ("x >= -100000000000000000000000000000000", {"x": "int"}),
    ("x != 0.0 and x * x >= 0.0", {"x": "float"}),
    ("type(x) is int and x > 0", {"x": "object"}),
    ("isinstance(x, int)", {"x": "bool"}),
    ("-len(xs) <= i < len(xs)", {"xs": "list", "i": "int"}),
    ("k in d", {"k": "int", "d": "dict[int]"}),
    ("flag or not flag", {"flag": "bool"}),
])
def test_checked_pure_subset(value, types):
    assert isinstance(check_predicate(value, types), ast.expr)


@pytest.mark.parametrize("value,types", [
    ("(x := 1) == 1", {}), ("__import__('os')", {}), ("x.f == 1", {"x": "object"}),
    ("x[0] == 1", {"x": "list"}), ("1 / x > 0", {"x": "int"}),
    ("len(x) > 0", {"x": "object"}), ("x == 0", {"x": "object"}),
    ("k in d", {"k": "int", "d": "dict"}),
    ("isinstance(x, int)", {"x": "object"}),
    ("isinstance(x, int) and x > 0", {"x": "object"}),
    ("x or True", {"x": "int"}), ("x > 0", {}), ("1e999 == 1e999", {}),
    ("type(x) is int", {"x": "object", "type": "int"}),
    ("x + y > 0", {"x": "int", "y": "float"}),
])
def test_refuse_partial_or_overloaded_predicates(value, types):
    with pytest.raises(InvalidPredicate):
        check_predicate(value, types)


def test_yaml_rejects_unsafe_constructors_duplicates_and_aliases():
    for text in ["!!python/object:builtins.str {}", "- entry_type: invariant_set\n  entry_type: other\n", "- &a [*a]"]:
        with pytest.raises(Exception):
            loads(text)
    doc = witness("x = 1", "t.py", "prp", [Invariant("t.py", 1, 1, "<module>", "True")])
    doc[0]["content"][0]["invariant"]["format"] = "c_expression"
    with pytest.raises(ValueError):
        dumps(doc)


def test_rebound_builtins_and_unmodeled_calls_stay_open():
    for src in ["type = 3\nb = 1\nx = 1 / b", "b = 1\nx = pow(0, b)"]:
        ob, = obligation_census(src, ["ZeroDivisionError"]).obligations
        assert ob.invariant.value == "False" and ob.gap


def test_isinstance_unknown_object_can_run_user_code():
    class Hook:
        @property
        def __class__(self):
            raise RuntimeError("user hook")
    with pytest.raises(RuntimeError, match="user hook"):
        isinstance(Hook(), int)
    with pytest.raises(InvalidPredicate):
        check_predicate("isinstance(x, int)", {"x": "object"})


def test_nonfinite_source_constant_leaves_an_explicit_gap():
    ob, = obligation_census("b = 2\nx = 1e999 / b\n", ["ZeroDivisionError"]).obligations
    assert ob.gap and ob.invariant.value == "False"


def test_unmodelled_implicit_calls_and_keys_negative_control():
    from pathlib import Path
    keys_src = (Path(__file__).parent / "fixtures/census_implicit_keys.py").read_text()
    for exc in ["AssertionError", "KeyError", "ZeroDivisionError"]:
        census = obligation_census(keys_src, [exc])
        assert not census.complete
        assert any("keys" in gap for gap in census.coverage_gaps)
        assert any("keys" in ob.gap or ob.invariant.function.endswith("keys") for ob in census.obligations)

    plain_src = "class C:\n    def keys(self):\n        return []\ndict(**C())\n"
    census = obligation_census(plain_src, ["KeyError"])
    assert not census.complete
    assert any("keys" in gap for gap in census.coverage_gaps)
    assert any("keys" in ob.gap for ob in census.obligations)

    for src in [
        "class C:\n    pass\nwith C():\n    pass\n",
        "@decorator\ndef f():\n    pass\n",
        "[*C()]\n",
    ]:
        census = obligation_census(src, ["ZeroDivisionError"])
        assert not census.complete
        assert len(census.coverage_gaps) > 0

@pytest.mark.parametrize("version", ["2.0", "2.3"])
def test_supported_witness_versions_round_trip(version):
    doc = witness("x = 1\n", "t.py", "prp", [Invariant("t.py", 1, 1, "<module>", "True")])
    doc[0]["metadata"]["format_version"] = version
    if version == "2.0":
        doc[0]["metadata"]["task"]["data_model"] = "CPython-3.14-binary64"
    assert loads(dumps(doc)) == doc


@pytest.mark.parametrize("version", ["2.1", "2.2", "2.4", 2.3, None])
def test_unsupported_witness_versions_are_rejected(version):
    doc = witness("x = 1\n", "t.py", "prp", [Invariant("t.py", 1, 1, "<module>", "True")])
    doc[0]["metadata"]["format_version"] = version
    with pytest.raises(ValueError, match="unsupported witness version"):
        dumps(doc)


def test_original_tab_and_unicode_columns_are_character_positions():
    from tharkun.solve.obligations import source_points

    source = "def f():\n\té = 1; x = 1 / é\nf()\n"
    points = source_points(ast.parse(source), source, "t.py")
    loc = next(loc for node, (_, loc) in points.items()
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == "x")
    assert loc == ("t.py", 2, 9, "f")
    entry = witness(source, "t.py", "prp", [Invariant(*loc, "True")])[0]
    assert entry["content"][0]["invariant"]["location"]["column"] == 9
