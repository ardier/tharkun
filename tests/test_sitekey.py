import ast
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from tharkun.knowledge import program_point
from tharkun.solve.census import SITE_KINDS, obligation_census
from tharkun.solve.obligations import source_points
from tharkun.solve.sitekey import prestate_info, prestate_scope, site_keys, stable_at_prestate
from tharkun.sv.obligation_table import subject_of


fixtures = Path(__file__).parent / "fixtures"
census_fixtures = [fixtures / "census_implicit_keys.py", *sorted((fixtures / "census").rglob("*.py"))]


@pytest.mark.parametrize("path", census_fixtures, ids=lambda p: p.name)
def test_every_fixture_census_site_keeps_its_subject_and_location(path):
    source = path.read_text()
    keys = site_keys(ast.parse(source), source, str(path))
    by_path = {key.ast_path: key for key in keys.values()}
    for row in obligation_census(source, SITE_KINDS, str(path)).obligations:
        if row.ast_path == "$coverage":
            assert row.gap  # A whole-source refusal is not a physical site.
            continue
        key = replace(by_path[row.ast_path], exception=row.exception)
        assert key.row_subject() == subject_of(row)
        assert key.source_sha256 == row.source_sha256
        assert key.location() == (row.invariant.file_name, row.invariant.line,
                                  row.invariant.column, row.invariant.function)


def test_legacy_points_and_dotted_function_scope_are_preserved():
    source = "class C:\n    def f(self, x=1 / y):\n        def g():\n            return x / y\n        return g()\n"
    tree = ast.parse(source)
    keys = site_keys(tree, source, "task.py", "ZeroDivisionError")
    assert source_points(tree, source, "task.py") == {
        node: (key.ast_path, key.location()) for node, key in keys.items()}
    operations = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp)]
    default, body = operations
    assert keys[default].location() == ("task.py", 2, 5, "C")
    assert keys[body].location() == ("task.py", 4, 13, "C.f.g")
    assert keys[body].point() == program_point("task.py", 4)
    assert keys[body].source_sha256 == hashlib.sha256(source.encode()).hexdigest()


def test_absint_byte_columns_join_separate_operations_exactly_once():
    from tharkun.solve.absint import prove

    source = "def f():\n\té = 2; a = 8 / é; b = 12 / é\n\treturn a + b\nf()\n"
    tree = ast.parse(source)
    keys = site_keys(tree, source, "task.py", "ZeroDivisionError")
    divisions = {node: key for node, key in keys.items()
                 if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)}
    result = prove(source, ("ZeroDivisionError",), filename="task.py")
    assert result.complete and result.records
    records = [record for record in result.records if record["kind"] == "divide"]
    assert len(records) == len(divisions) == 2
    for record in records:
        matches = [key for key in divisions.values()
                   if (key.span[0], key.byte_column) == (record["line"], record["col"])]
        assert len(matches) == 1
        key, = matches
        assert key.function == "f"
        assert key.span[1] == len(source.splitlines()[1].encode()[:record["col"]].decode()) + 1
        assert key.span[1] != record["col"] + 1
    assert {key.location()[2] for key in divisions.values()} == {9, 20}


def test_full_paths_and_spans_distinguish_nested_same_start_operations():
    source = "result = a / b / c\n"
    keys = site_keys(ast.parse(source), source, "task.py", "ZeroDivisionError")
    outer, inner = [key for node, key in keys.items() if isinstance(node, ast.BinOp)]
    assert (outer.span[0], outer.byte_column) == (inner.span[0], inner.byte_column)
    assert outer.record_key() != inner.record_key()
    assert outer.span != inner.span
    assert outer.row_subject() != inner.row_subject()
    assert outer.location() == inner.location()


@pytest.mark.parametrize("source, expected", [
    ("out = a / b", True),
    ("a = a / b", True),
    ("a += b", True),
    ("out = -a", True),
    ("if a < b:\n    pass", True),
    ("while a < b:\n    pass", True),
    ("out = a[b]", True),
    ("out = f() + a", False),
    ("out = a.value + b", False),
    ("out = a[0] + b", False),
    ("out = (a := 1) + b", False),
    ("out = mutate() or a / b", False),
    ("out = (a / b) if flag else 0", False),
    ("out = sink(mutate(), a / b)", False),
    ("out = lambda: a / b", False),
    ("out = [a / b for a in xs]", False),
    ("out = a < b < c", False),
    ("a[b] += c", False),
])
def test_prestate_stability_rejects_effects_aliases_and_delayed_evaluation(source, expected):
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    node = next(node for node in ast.walk(tree)
                if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.Subscript, ast.AugAssign)))
    assert stable_at_prestate(node, parents) is expected


@pytest.mark.parametrize("source, expected", [
    ("out = (a < b) and (c < d)", "statement_entry"),
    ("if not ((a < b) or (c < d)):\n    pass", "statement_entry"),
    ("out = flag and (a < b)", None),
    ("out = ((a + 1) < b) and (c < d)", None),
    ("out = ready() and (a < b)", None),
    ("out = (a < b) if flag else False", None),
    ("out = (a is b) and (c < d)", None),
    ("out = (a in b) and (c < d)", None),
    ("out = (a < b < c) and (d < e)", None),
])
def test_nested_atom_scope_requires_measured_pure_boolean_prefix(source, expected):
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    atoms = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)]
    atom = max(atoms, key=lambda node: (node.lineno, node.col_offset))
    assert prestate_scope(atom, parents) == expected


def test_prestate_info_returns_shared_ordered_prefix_and_statement():
    tree = ast.parse("out = (a < b) and ((c < d) or (e < f))\n")
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    atom = max((node for node in ast.walk(tree) if isinstance(node, ast.Compare)),
               key=lambda node: (node.lineno, node.col_offset))
    scope, statement, prefix = prestate_info(atom, parents)
    assert scope == "statement_entry"
    assert statement is tree.body[0]
    assert [ast.unparse(item) for item in prefix] == ["a < b", "c < d"]


@pytest.mark.parametrize("source, statement_type", [
    ("assert (a + b) < c", ast.Assert),
    ("if (a + b) < c:\n    pass", ast.If),
])
def test_leftmost_binary_operand_of_direct_comparison_is_statement_entry(source, statement_type):
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    node = next(item for item in ast.walk(tree) if isinstance(item, ast.BinOp))
    scope, statement, prefix = prestate_info(node, parents)
    assert scope == "statement_entry"
    assert isinstance(statement, statement_type)
    assert prefix == ()


@pytest.mark.parametrize("source", [
    "assert c < (a + b)",
    "assert (a + b) < c and c < d",
    "out = (a + b) < c",
    "if c == (a + b):\n    pass",
    "assert (f() + b) < c",
])
def test_nested_or_later_binary_comparison_operands_remain_unclassified(source):
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    node = next(item for item in ast.walk(tree) if isinstance(item, ast.BinOp))
    assert prestate_scope(node, parents) is None


def test_prestate_scope_keeps_direct_site_classification():
    tree = ast.parse("out = a / b\n")
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    atom = next(node for node in ast.walk(tree) if isinstance(node, ast.BinOp))
    assert prestate_scope(atom, parents) == "site"
