"""One end-to-end check: typed Z3 regex, composed seeds, and boundary seeds."""

import re

import z3

from tharkun.sv.engine import verify
from tharkun.sv.string_regex import regex_constraint
from tharkun.sv.solver_seeds import seed_values


def test_solver_tricks_replay(tmp_path):
    for mode in ("match", "fullmatch", "search"):
        value = z3.String("value")
        solver = z3.Solver()
        solver.add(regex_constraint(value, r"[a-z]{3}[0-9]{2}", mode))
        assert solver.check() == z3.sat
        assert getattr(re, mode)(r"[a-z]{3}[0-9]{2}", solver.model().eval(value).as_string())

    p = tmp_path / "regex.py"
    p.write_text("import re, _sv_verifier\n"
                 "if __name__ == \"__main__\":\n"
                 "    s = _sv_verifier.nondet_str()\n"
                 "    if re.fullmatch(r\"[a-z]{3}[0-9]{2}\", s): raise KeyError(s)\n")
    v = verify(str(p), ("KeyError",), budget_seconds=60, jobs=1, stages=["concolic"])   # z3 solves it in well under a second; the budget is a ceiling
    assert v.verdict == "false"
    assert v.witness["vector"][0]["kind"] == "str"

    p.write_text("import _sv_verifier\n"
                 "a = _sv_verifier.nondet_str()\n"
                 "b = _sv_verifier.nondet_str()\n"
                 "if a + b == \"tharkun\" and len(a) == 3: [][0]\n")
    assert {"tha", "rkun"} <= set(seed_values(str(p), "composition")["str"])
    p.write_text("import _sv_verifier\n"
                 "x = _sv_verifier.nondet_int()\n"
                 "if 100 < x < 105: [][0]\n")
    bounds = seed_values(str(p), "boundaries", [{"types": {"x": "int"}, "scope": ["x >= 200"]}])["int"]
    assert {99, 100, 101, 104, 105, 106, 199, 200, 201} <= set(bounds)


def test_task_metadata_never_changes_verdict(tmp_path):
    import ast

    from tharkun.sv.engine import harvest_constants

    tasks = (
        ("""\"\"\"secret-string\"\"\"\n# secret-int: 99999\nimport _sv_verifier\nif __name__ == "__main__":\n    x: "annotation-string" = _sv_verifier.nondet_int()\n    if x == 7: raise KeyError(x)\n""", ("KeyError",), "false"),
        ("""\"\"\"raise KeyError()\"\"\"\n# false hint\nif __name__ == "__main__":\n    print(1)\n""", ("KeyError",), "true"),
        ("""\"\"\"go\"\"\"\n# pretend this is a solver seed\nimport _sv_verifier\nif __name__ == "__main__":\n    s = _sv_verifier.nondet_str()\n    if s == "go": raise IndexError(s)\n""", ("IndexError",), "false"),
    )
    for i, (source, classes, expected) in enumerate(tasks):
        original = tmp_path / f"original_{i}.py"
        stripped = tmp_path / f"stripped_{i}.py"
        original.write_text(source)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body \
                    and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                    and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
        stripped.write_text(ast.unparse(tree))  # ast.unparse drops comments
        assert verify(str(original), classes, budget_seconds=60, jobs=1).verdict == expected
        assert verify(str(stripped), classes, budget_seconds=60, jobs=1).verdict == expected
    assert "secret-string" not in harvest_constants(str(tmp_path / "original_0.py")).get("str", ())
    assert "annotation-string" not in harvest_constants(str(tmp_path / "original_0.py")).get("str", ())
    assert "secret-string" not in seed_values(str(tmp_path / "original_0.py"), "composition").get("str", ())
    annotated = tmp_path / "annotation.py"
    annotated.write_text("import _sv_verifier\nx: (x == \"metadata-token\") = _sv_verifier.nondet_str()\n")
    assert "metadata-token" not in seed_values(str(annotated), "comparison").get("str", ())


def test_generic_parameter_metadata_never_supplies_seeds(tmp_path):
    import ast
    import sys
    import pytest
    if not hasattr(ast, "type_param"):
        pytest.skip("generic syntax requires Python 3.12")
    declarations = [
        "def f[T: (x == \"metadata-token\")](): pass",
        "class C[T: (x == \"metadata-token\")]: pass",
        "type Alias[T: (x == \"metadata-token\")] = int",
    ]
    if sys.version_info >= (3, 13):
        declarations.append("def f[T = (x == \"metadata-token\")](): pass")
    program = tmp_path / "generic.py"
    for declaration in declarations:
        program.write_text("import _sv_verifier\nx = _sv_verifier.nondet_str()\n" + declaration + "\n")
        ast.parse(program.read_text())  # Reject a vacuous pass through syntax-error handling.
        for trick in ("comparison", "composition"):
            assert "metadata-token" not in seed_values(str(program), trick).get("str", ())
