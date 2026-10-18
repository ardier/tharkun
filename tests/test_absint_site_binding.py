"""Source-bound site and operand metadata on absint's legacy records."""

from __future__ import annotations

import ast
import hashlib
from types import SimpleNamespace

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.absint.interp import Analyzer
from tharkun.solve.absint.state import Frame, State
from tharkun.solve.sitekey import site_keys


def _records(source, exc="TypeError"):
    result = prove(source, (exc,))
    return result, [row for row in result.records if row["exc"] == exc]


def test_record_keeps_legacy_fields_and_adds_original_source_identity():
    source = "x = 1\ny = 's'\nz = x + y\n"
    result, rows = _records(source)
    assert rows, result.explain()
    row = next(r for r in rows if r["kind"] == "binop")
    assert row["source_sha"] == hashlib.sha256(source.encode()).hexdigest()
    assert row["site"][0] == row["source_sha"]
    assert row["site"][2] == "TypeError"
    assert row["binding"] == {
        "receiver": '(term 1 (var Py "x"))',
        "args": ['(term 1 (var Py "y"))'],
        "op": "Add",
    }
    assert row["facts"] and row["facts"] == sorted(row["facts"])
    assert any("typeis" in fact for fact in row["facts"])
    assert row["contract"] == {"qualname": "protocol.binary_operator", "exception": "TypeError"}
    assert set(row) == {
        "line", "col", "exc", "kind", "function", "guards", "status", "fact", "operand",
        "residual", "scope", "types", "context", "loop_iteration", "site", "binding", "facts",
        "contract", "builtins", "source_sha", "fact_scope",
    }
    assert row["fact_scope"] == "site"
    assert row["status"] == "may_raise" and row["fact"] == "" and row["operand"] == ""
    assert row["residual"] is None and row["scope"] == [] and row["types"] == {}
    assert row["context"] == [] and row["loop_iteration"] == "any" and row["guards"] == []


def _builtin_names(source, *, site_source=None):
    analyzer = Analyzer(source, site_source=site_source)
    node = next(n for n in ast.walk(analyzer.tree) if isinstance(n, ast.BinOp))
    site = analyzer._source_sites.get(node)
    if site is None and site_source is not None:
        original_tree = ast.parse(site_source)
        original_node = next(n for n in ast.walk(original_tree) if isinstance(n, ast.BinOp))
        site = site_keys(original_tree, site_source, "<program>")[original_node]
    state = State()
    state.frames = [Frame("module")]
    analyzer._fctx_stack.append(SimpleNamespace(
        is_module=True, is_class=False, local_names=set(), global_names=set(), module_key=""
    ))
    return analyzer._record_builtins(state, node, site)


def test_untouched_source_records_all_five_builtin_identities():
    assert _builtin_names("result = left + right\n") == ["bool", "float", "int", "str", "type"]


def test_any_original_source_binding_vetoes_that_builtin_name():
    # A later assignment, another function, a comprehension target, or an import
    # can change lexical/global lookup independently of the sliced execution.
    for source in (
        "result = left + right\nint = 1\n",
        "def later():\n    int = 1\nresult = left + right\n",
        "def unrelated():\n    int = 1\nresult = left + right\n",
        "def f(int):\n    return left + right\nresult = left + right\n",
        "import int\nresult = left + right\n",
    ):
        names = _builtin_names(source)
        assert "int" not in names, (source, names)
    source = "items = [type for type in values]\nresult = left + right\n"
    assert "type" not in _builtin_names(source)


def test_reflection_anywhere_vetoes_all_builtin_identities():
    for source in (
        "result = left + right\nlocals()\n",
        "def f():\n    return globals()\nresult = left + right\n",
        "import builtins\nresult = left + right\n",
        "from builtins import len\nresult = left + right\n",
        "from unknown_module import *\nresult = left + right\n",
        "obj.__dict__\nresult = left + right\n",
    ):
        assert _builtin_names(source) == [], source


def test_type_parameter_names_veto_builtin_identities_when_supported():
    if not all(hasattr(ast, name) for name in ("TypeVar", "ParamSpec", "TypeVarTuple")):
        pytest.skip("type-parameter AST nodes require Python 3.12+")
    tree = ast.parse("class Generic[T, **int, *U]:\n    pass\nresult = left + right\n")
    bound, reflective = Analyzer._builtin_source_vetoes(tree)
    assert "T" in bound and "int" in bound and "U" in bound
    assert not reflective


def test_original_source_binding_veto_survives_slicing():
    sliced = "result = left + right\n"
    originals = (
        "int = 1\n" + sliced,
        sliced + "def later():\n    int = 1\n",
        "def f():\n    result = left + right\n    int = 1\n",
    )
    for original in originals:
        analyzer = Analyzer(sliced, site_source=original)
        bound, reflective = analyzer._record_builtin_vetoes
        assert "int" in bound and not reflective
        # The sliced operation has no matching path in these originals; use the
        # exact original site identity to check its global source veto directly.
        names = _builtin_names(sliced, site_source=original)
        assert "int" not in names and set(names) == {"bool", "float", "str", "type"}


def test_same_line_operations_keep_distinct_site_identity():
    source = "a = 1; b = 2; left = a + b; right = b + a\n"
    result, rows = _records(source)
    operations = [r for r in rows if r["kind"] == "binop"]
    assert len(operations) == 2, result.explain()
    assert all(r["status"] == "safe" for r in operations)
    assert len({tuple(r["site"]) for r in operations}) == 2
    assert all(r["binding"]["op"] == "Add" for r in operations)


def test_sliced_analysis_uses_original_source_sha_and_exact_path():
    source = "unused = 0\nx = 1\ny = 's'\nz = x + y\n"
    result, rows = _records(source)
    assert rows, result.explain()
    row = next(r for r in rows if r["kind"] == "binop")
    assert row["source_sha"] == hashlib.sha256(source.encode()).hexdigest()
    assert row["site"][0] == hashlib.sha256(source.encode()).hexdigest()
    assert row["site"][1] == "$.body[3].value"


def test_changed_statement_location_does_not_map_to_original_site():
    source = "x = a + b\n"
    shifted = "\n" + source
    mapped = Analyzer._map_source_sites(ast.parse(shifted), shifted, source, "<test>")
    assert mapped == {}


def test_analyzer_reuses_site_keys_for_each_source_text(monkeypatch):
    import tharkun.solve.sitekey as sitekey

    real_site_keys = sitekey.site_keys
    calls = []

    def counted(tree, source, filename, *args, **kwargs):
        calls.append(source)
        return real_site_keys(tree, source, filename, *args, **kwargs)

    monkeypatch.setattr(sitekey, "site_keys", counted)
    original = "x = 1\ny = 's'\nz = x + y\nunused = 0\n"
    same_source = Analyzer(original)
    assert calls == [original]
    assert same_source._record_source_parents is same_source._site_parents

    calls.clear()
    sliced = "x = 1\ny = 's'\nz = x + y\n"
    distinct_source = Analyzer(sliced, site_source=original)
    assert calls == [original, sliced]
    assert distinct_source._record_source_parents is not distinct_source._site_parents
    assert distinct_source._source_sites


def test_repeated_visits_intersect_facts_and_clear_disagreeing_bindings():
    source = """\
x = 1
while x:
    y = 's' if x else 2
    z = x + y
    x = 0
"""
    result, rows = _records(source)
    assert rows, result.explain()
    row = next(r for r in rows if r["kind"] == "binop")
    assert row["site"] is not None
    # The loop revisits the same syntactic site under different bindings; a
    # representative visit must not masquerade as the only abstract state.
    assert row["binding"] == {
        "receiver": '(term 1 (var Py "x"))',
        "args": ['(term 1 (var Py "y"))'],
        "op": "Add",
    }
    assert row["facts"] == sorted(row["facts"])


def test_nested_boolean_atom_records_entry_facts_from_prefix_and_site_operands():
    source = "a = 2\nb = 1\nc = 3\nd = 2\nout = (a > b) and (c > d)\n"
    result, rows = _records(source, "TypeError")
    nested = max((row for row in rows if row["line"] == 5), key=lambda row: row["col"])
    assert nested["binding"] is not None, result.explain()
    assert nested["fact_scope"] == "statement_entry"
    assert any('"c"' in fact for fact in nested["facts"])
    assert any('"d"' in fact for fact in nested["facts"])
    assert not any(any(f'"{name}"' in fact for name in ("a", "b")) for fact in nested["facts"])


@pytest.mark.parametrize("statement", [
    "if left + right < 0:\n    pass",
    "assert left + right < 0",
])
def test_binary_left_operand_of_direct_comparison_records_entry_facts(statement):
    source = f'''\
import _sv_verifier
left = _sv_verifier.nondet_int()
right = _sv_verifier.nondet_str()
{statement}
'''
    result, rows = _records(source, "TypeError")
    row = next(record for record in rows if record["kind"] == "binop")
    assert row["binding"] == {
        "receiver": '(term 1 (var Py "left"))',
        "args": ['(term 1 (var Py "right"))'],
        "op": "Add",
    }, result.explain()
    assert row["fact_scope"] == "statement_entry"
    assert any('"left"' in fact and '"int"' in fact for fact in row["facts"])
    assert any('"right"' in fact and '"str"' in fact for fact in row["facts"])


@pytest.mark.parametrize("calls", [
    'f("skip", draw)\nf(0, draw)',
    'f(0, draw)\nf("skip", draw)',
])
def test_statement_entry_facts_include_skipped_call_entries(calls):
    source = f'''\
import _sv_verifier
def f(x, y):
    return (x == 0) and (1 / y)
draw = _sv_verifier.nondet_int()
{calls}
'''
    result, rows = _records(source, "ZeroDivisionError")
    row = next(record for record in rows if record["kind"] == "divide")
    assert row["binding"] is not None, result.explain()
    assert row["fact_scope"] == "statement_entry"
    # x is str in one entry and int in the other; neither tag is a must-fact.
    assert not any('"x"' in fact and "typeis" in fact for fact in row["facts"])
    # y is an exact int at every entry, including the entry that skipped the atom.
    assert any('"y"' in fact and "typeis" in fact for fact in row["facts"])


def test_later_unsafe_prefix_entry_clears_existing_nested_record_metadata():
    source = '''\
import _sv_verifier
def f(x, y):
    return (x < 0) and (1 / y)
draw = _sv_verifier.nondet_int()
f(-1, draw)
f("skip", draw)
'''
    result, rows = _records(source, "ZeroDivisionError")
    row = next(record for record in rows if record["kind"] == "divide")
    assert row["binding"] is None, result.explain()
    assert row["facts"] == []
    assert "fact_scope" not in row


def test_final_skipped_while_call_entry_clears_nested_record_metadata():
    source = '''\
flag = True
denominator = 1
while (flag == True) and 10 // denominator:
    flag = False
    denominator = None
'''
    result, rows = _records(source, "ZeroDivisionError")
    row = next(record for record in rows if record["kind"] == "divide")
    assert row["binding"] is not None, result.explain()
    assert row["fact_scope"] == "statement_entry"
    assert not any('"denominator"' in fact and
                   ('"int"' in fact or "as_int" in fact)
                   for fact in row["facts"]), result.explain()


def test_nested_entry_observation_uses_class_namespace_values():
    source = '''\
import _sv_verifier
class C:
    flag = _sv_verifier.nondet_bool()
    text = "s"
    result = (flag == True) and (text + 1)
'''
    result, rows = _records(source, "TypeError")
    row = next((record for record in rows if record["kind"] == "binop"), None)
    assert row is not None, (result.explain(), rows)
    assert row["binding"] is not None, result.explain()
    assert row["fact_scope"] == "statement_entry"
    assert any('"text"' in fact and "typeis" in fact for fact in row["facts"])
    assert not any('"flag"' in fact for fact in row["facts"])


def test_original_function_local_veto_survives_sliced_statement_mapping():
    original = '''\
def f(y):
    result = (x < 0) and (1 / y)
    x = 1
'''
    sliced = '''\
def f(y):
    result = (x < 0) and (1 / y)
'''
    analyzer = Analyzer(sliced, site_source=original)
    node = next(node for node in ast.walk(analyzer.tree)
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div))
    assert analyzer._source_sites.get(node) is not None
    function = analyzer.tree.body[0]
    fctx = SimpleNamespace(node=function, local_names={"y"}, global_names=set(),
                           is_module=False, is_class=False, module_key="")
    assert not analyzer._source_scope_matches(node, fctx, ("x", "y"))


def test_metadata_merge_intersects_facts_and_stickily_clears_changed_bindings():
    source = "x = 1\ny = 's'\nz = x + y\n"
    analyzer = Analyzer(source)
    node = next(n for n in ast.walk(analyzer.tree) if isinstance(n, ast.BinOp))

    facts = iter([(["fact-int"], True), (["fact-str"], True)])
    builtins = iter([["int", "str"], ["int"]])
    analyzer._record_terms = lambda st, names: next(facts)
    analyzer._record_builtins = lambda *args: next(builtins)
    analyzer._record(node, "binop", {"TypeError"}, False, "", None, None, State())
    analyzer._record(node, "binop", {"TypeError"}, False, "", None, None, State())
    row = next(iter(analyzer.records.values()))
    assert row["binding"] is not None  # syntax binding survives exact value/type changes
    assert row["facts"] == []  # facts from distinct visits are intersected
    assert row["builtins"] == ["int"]

    analyzer = Analyzer(source)
    node = next(n for n in ast.walk(analyzer.tree) if isinstance(n, ast.BinOp))
    facts = iter([(["must-fact"], True), ([], False), (["must-fact"], True)])
    builtins = iter([["int"], [], ["int"]])
    analyzer._record_terms = lambda st, names: next(facts)
    analyzer._record_builtins = lambda *args: next(builtins)
    for _ in range(3):
        analyzer._record(node, "binop", {"TypeError"}, False, "", None, None, State())
    row = next(iter(analyzer.records.values()))
    assert row["binding"] is None and row["facts"] == []
    assert row["builtins"] == []


def test_non_scalar_heap_names_do_not_receive_source_binding_terms():
    source = "x = []\ny = x + 1\n"
    result, rows = _records(source)
    assert rows, result.explain()
    row = next(r for r in rows if r["kind"] == "binop")
    assert row["binding"] is None and row["facts"] == []
    assert "fact_scope" not in row


def test_lambda_body_records_keep_contract_but_fail_closed_on_source_bindings():
    source = '''\
import _sv_verifier
x = "module value, not the parameter"
f = lambda x: x + 1
f(_sv_verifier.nondet_int())
'''
    result, rows = _records(source)
    row = next(r for r in rows if r["kind"] == "binop")
    assert result.complete, result.explain()
    assert row["contract"] == {"qualname": "protocol.binary_operator", "exception": "TypeError"}
    assert row["binding"] is None and row["facts"] == []
    assert "fact_scope" not in row


def test_nonempty_division_record_preserves_legacy_residual_scope_and_types():
    source = """\
import _sv_verifier
n = abs(_sv_verifier.nondet_int())
k = n + 3
print(10 // (k - n - 3))
"""
    result, rows = _records(source, "ZeroDivisionError")
    record = next(r for r in rows if r["kind"] == "divide")
    assert result.complete
    assert record["residual"] == "(k - n - 3) == 0"
    assert "k - n <= 3" in record["scope"] and "n - k <= -3" in record["scope"]
    assert record["types"] == {"k": "int", "n": "int"}
    assert record["site"][0] == hashlib.sha256(source.encode()).hexdigest()
    assert record["binding"] is None and record["facts"] == []


def test_membership_does_not_claim_the_rich_comparison_contract():
    analyzer = Analyzer("x = a in b\ny = a < b\n")
    membership = ast.parse("a in b").body[0].value
    ordering = ast.parse("a < b").body[0].value
    assert analyzer._record_contract("compare", "TypeError", membership) is None
    assert analyzer._record_contract("compare", "TypeError", ordering) == {
        "qualname": "protocol.comparison", "exception": "TypeError"
    }


def test_legacy_record_collision_clears_all_source_bound_metadata():
    source = "result = (a + b) + (c + d)\n"
    analyzer = Analyzer(source)
    ops = [node for node in ast.walk(analyzer.tree) if isinstance(node, ast.BinOp)]
    assert len(ops) == 3
    first, second = ops[0], ops[1]
    second.lineno, second.col_offset = first.lineno, first.col_offset
    analyzer._record(first, "binop", {"TypeError"}, False, "", None, None, None)
    analyzer._record(second, "binop", {"TypeError"}, False, "", None, None, None)
    assert len(analyzer.records) == 1
    row = next(iter(analyzer.records.values()))
    assert row["site"] is None
    assert row["binding"] is None
    assert row["contract"] is None
    assert row["facts"] == []
    assert row["builtins"] == []


def test_lowered_operations_keep_legacy_record_but_decline_unmapped_site_metadata():
    source = "xs = [1]\ny = [x + 1 for x in xs]\n"
    result, rows = _records(source)
    assert rows, result.explain()
    row = next(r for r in rows if r["kind"] == "binop")
    assert row["source_sha"] == hashlib.sha256(source.encode()).hexdigest()
    assert row["site"] is None
    assert row["binding"] is None
    assert row["facts"] == []


def test_sitekey_path_mapping_requires_exact_location_attributes():
    source = "x = a + b\n"
    original = ast.parse(source)
    shifted = ast.parse("\n" + source)
    original_keys = {key.ast_path: key for key in site_keys(original, source, "<test>").values()}
    shifted_keys = {key.ast_path: key for key in site_keys(shifted, "\n" + source, "<test>").values()}
    assert "$.body[0].value" in original_keys
    assert "$.body[0].value" in shifted_keys
    assert original_keys["$.body[0].value"].source_sha256 != shifted_keys["$.body[0].value"].source_sha256
