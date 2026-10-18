"""Declared task sibling modules are a bounded fallback for absint imports."""

from pathlib import Path
import ast
import sys

from tharkun.solve.absint import prove
from tharkun.solve.census import obligation_census, site_census
from tharkun.solve.determinism import determinism_census
from tharkun.solve.environment import TaskEnvironment
from tharkun.solve.pyresolve import Resolver, Top
from tharkun.sv.prove import prove_static
from tharkun.sv.engine import run_once
from tharkun.sv import verify


def _task(tmp_path: Path, source: str, helper: str | None = None):
    main = tmp_path / "program.py"
    main.write_text(source)
    files = [main]
    if helper is not None:
        sibling = tmp_path / f"{helper[0]}.py"
        sibling.write_text(helper[1])
        files.append(sibling)
    return main, TaskEnvironment(tuple(str(path) for path in files))


def _prove(main: Path, environment, source: str):
    return prove(source, ("AssertionError",), filename=str(main), environment=environment, time_budget=60.0)


def _native(main: Path, environment):
    return run_once(sys.executable, str(main), [], timeout=30, files=environment.files)


def _manifest_path(tmp_path, names):
    task = tmp_path / "task.yml"
    task.write_text("format_version: '2.0'\ninput_files:\n  - program.py\nrequired_files:\n" +
                    "".join(f"  - {name}\n" for name in names))
    return task


def _manifest(tmp_path, names):
    return TaskEnvironment.from_task(str(_manifest_path(tmp_path, names)))


def test_declared_sibling_function_uses_its_globals_and_definition_time_default(tmp_path):
    source = "BASE = 99\nfrom helper import check\ncheck()\n"
    helper = ("helper", "BASE = 3\ndef check(value=BASE):\n    if value != 3 or BASE != 4:\n        raise AssertionError()\nBASE = 4\n")
    main, environment = _task(tmp_path, source, helper)

    native = _native(main, environment)
    result = _prove(main, environment, source)

    assert native.status == "clean", native
    assert result.verdict == "TRUE", result.explain()


def test_declared_module_qualified_function_call_is_deterministic(tmp_path):
    source = "import helper as imported_helper\nimported_helper.check()\n"
    helper = ("helper", "def check():\n    return None\n")
    main, environment = _task(tmp_path, source, helper)

    native = _native(main, environment)
    result = determinism_census(source, str(main), declared_files=environment.files)

    assert native.status == "clean", native
    assert result.ok, result.reason


def test_bare_declared_source_import_executes_and_proves_clean(tmp_path):
    source = "import helper\n"
    helper = ("helper", "VALUE = 1\n")
    main, environment = _task(tmp_path, source, helper)

    native = _native(main, environment)
    result = _prove(main, environment, source)

    assert native.status == "clean", native
    assert result.verdict == "TRUE", result.explain()


def test_bare_declared_source_import_cannot_hide_top_level_property_raise(tmp_path):
    source = "import helper\n"
    helper = ("helper", "raise AssertionError()\n")
    main, environment = _task(tmp_path, source, helper)

    native = _native(main, environment)
    result = _prove(main, environment, source)

    assert native.status == "raised" and native.exc == "AssertionError", native
    assert result.verdict != "TRUE", result.explain()


def test_package_import_precedence_beats_declared_sibling_file(tmp_path):
    source = "import helper\n"
    for declared_package in (False, True):
        case = tmp_path / ("declared-package" if declared_package else "ambient-package")
        case.mkdir()
        main = case / "program.py"
        main.write_text(source)
        sibling = case / "helper.py"
        sibling.write_text("VALUE = 1\n")
        package = case / "helper"
        package.mkdir()
        init = package / "__init__.py"
        init.write_text("raise AssertionError()\n")
        files = [str(main), str(sibling)]
        if declared_package:
            files.append(str(init))
        environment = TaskEnvironment(tuple(files))

        native = run_once(sys.executable, str(main), [], timeout=30)
        census = site_census(source, ("AssertionError",), str(main),
                             declared_files=environment.files)
        result = _prove(main, environment, source)

        assert native.status == "raised" and native.exc == "AssertionError", native
        assert not census.no_site, census.reason
        assert result.verdict != "TRUE", result.explain()


def test_declared_function_value_is_not_address_safe(tmp_path):
    source = "import helper\nsaved = helper.check\n"
    helper = ("helper", "def check():\n    return None\n")
    main, environment = _task(tmp_path, source, helper)

    result = determinism_census(source, str(main), declared_files=environment.files)

    assert not result.ok, result.reason


def test_repr_of_declared_function_result_is_not_address_safe(tmp_path):
    source = "import helper\ntext = repr(helper.value())\n"
    helper = ("helper", "def value():\n    return object()\n")
    main, environment = _task(tmp_path, source, helper)

    result = determinism_census(source, str(main), declared_files=environment.files)

    assert not result.ok, result.reason


def test_undeclared_sibling_and_bare_prove_keep_the_existing_refusal(tmp_path):
    source = "from helper import check\ncheck()\n"
    helper = ("helper", "def check():\n    raise AssertionError()\n")
    main, _declared_environment = _task(tmp_path, source, helper)
    no_helper = TaskEnvironment((str(main),))

    assert _prove(main, no_helper, source).verdict != "TRUE"
    assert prove(source, ("AssertionError",), filename=str(main), time_budget=60.0).verdict != "TRUE"


def test_literal_program_import_root_precedes_declared_sibling(tmp_path):
    root = tmp_path / "literal-root"
    root.mkdir()
    (root / "helper.py").write_text("def check():\n    raise AssertionError()\n")
    source = f"import sys\nsys.path.insert(0, {str(root)!r})\nfrom helper import check\ncheck()\n"
    main, environment = _task(tmp_path, source, ("helper", "def check():\n    return None\n"))
    environment = TaskEnvironment((*environment.files, str(root / "helper.py")))

    native = _native(main, environment)
    result = _prove(main, environment, source)

    assert native.status == "raised" and native.exc == "AssertionError", native
    assert result.verdict != "TRUE"


def test_declared_stdlib_name_collision_fails_closed(tmp_path):
    source = "import math\nx = math.sqrt(4)\n"
    helper = ("math", "def sqrt(value):\n    raise AssertionError()\n")
    main, environment = _task(tmp_path, source, helper)

    result = _prove(main, environment, source)

    assert result.verdict != "TRUE"
    assert "math" in result.reason


def test_declared_files_require_the_exact_primary_source(tmp_path):
    source = "from helper import check\ncheck()\n"
    helper = ("helper", "def check():\n    return None\n")
    main, environment = _task(tmp_path, source, helper)
    other_primary = tmp_path / "other.py"
    other_primary.write_text(source)
    mismatched = TaskEnvironment((str(other_primary), *environment.files[1:]))

    assert _prove(main, mismatched, source).verdict != "TRUE"


def test_declared_siblings_do_not_enable_relative_or_dotted_imports(tmp_path):
    helper = ("helper", "def check():\n    return None\n")
    cases = (
        "from .helper import check\ncheck()\n",
        "from helper.child import check\ncheck()\n",
        "import helper.child\n",
    )
    for source in cases:
        main, environment = _task(tmp_path, source, helper)
        result = _prove(main, environment, source)
        assert result.verdict != "TRUE", (source, result)


def test_same_basename_declared_only_outside_primary_directory_is_not_resolved(tmp_path):
    source = "from helper import check\ncheck()\n"
    main = tmp_path / "program.py"
    main.write_text(source)
    nested = tmp_path / "nested"
    nested.mkdir()
    helper = nested / "helper.py"
    helper.write_text("def check():\n    return None\n")
    environment = TaskEnvironment((str(main), str(helper)))

    assert _prove(main, environment, source).verdict != "TRUE"


def test_declared_flat_sibling_manifest_resolves_transitively_in_certificates(tmp_path):
    source = "from first import check\ncheck()\n"
    main = tmp_path / "program.py"
    main.write_text(source)
    (tmp_path / "first.py").write_text("from second import good\ndef check():\n    return good()\n")
    (tmp_path / "second.py").write_text("def good():\n    return True\n")
    task = _manifest_path(tmp_path, ["first.py", "second.py"])
    environment = TaskEnvironment.from_task(str(task))

    native = _native(main, environment)
    census = determinism_census(source, str(main), declared_files=environment.files)
    static = prove_static(str(main), ("AssertionError",), files=environment.files)
    verdict = verify(str(main), ("AssertionError",), task=str(task), budget_seconds=60,
                     jobs=1, canary_seconds=0)

    assert native.status == "clean", native
    assert census.ok, census.reason
    assert static is not None and static.verdict == "true"
    assert verdict.verdict == "true", verdict


def test_transitive_sibling_property_sites_remain_visible(tmp_path):
    source = "from first import check\ncheck()\n"
    main = tmp_path / "program.py"
    main.write_text(source)
    (tmp_path / "first.py").write_text("from second import fail\ndef check():\n    fail()\n")
    (tmp_path / "second.py").write_text("def fail():\n    raise AssertionError()\n")
    environment = _manifest(tmp_path, ["first.py", "second.py"])

    native = _native(main, environment)
    sites = site_census(source, ("AssertionError",), str(main), declared_files=environment.files)
    obligations = obligation_census(source, ("AssertionError",), str(main),
                                    declared_files=environment.files)

    assert native.status == "raised" and native.exc == "AssertionError", native
    assert not sites.no_site, sites.reason
    assert not obligations.complete
    assert any("imported-module obligation locations are not represented" in gap
               for gap in obligations.coverage_gaps)


def test_transitive_sibling_nondeterminism_is_not_certified(tmp_path):
    source = "from first import value\nx = value()\n"
    main = tmp_path / "program.py"
    main.write_text(source)
    (tmp_path / "first.py").write_text("from second import value\n")
    (tmp_path / "second.py").write_text("import random\ndef value():\n    return random.SystemRandom().random()\n")
    environment = _manifest(tmp_path, ["first.py", "second.py"])

    result = determinism_census(source, str(main), declared_files=environment.files)

    assert not result.ok and "random.SystemRandom" in result.reason


def test_transitive_controlled_random_draw_is_deterministic(tmp_path):
    source = "from first import value\nx = value()\n"
    main = tmp_path / "program.py"
    main.write_text(source)
    (tmp_path / "first.py").write_text("from second import value\n")
    (tmp_path / "second.py").write_text("import random\ndef value():\n    return random.random()\n")
    environment = _manifest(tmp_path, ["first.py", "second.py"])

    result = determinism_census(source, str(main), declared_files=environment.files)

    assert result.ok, result.reason


def test_declared_module_rejects_missing_private_and_stored_attributes(tmp_path):
    cases = (
        ("helper.count()\n", "not a public attribute"),
        ("helper._hidden\n", "not a public attribute"),
        ("helper.value = 1\n", "stores to module"),
    )
    for index, (tail, expected_reason) in enumerate(cases):
        case = tmp_path / str(index)
        case.mkdir()
        source = "import helper\n" + tail
        main = case / "program.py"
        main.write_text(source)
        (case / "helper.py").write_text("value = 0\ndef check():\n    return None\ndef _hidden():\n    return None\n")
        environment = _manifest(case, ["helper.py"])

        result = determinism_census(source, str(main), declared_files=environment.files)

        assert not result.ok and expected_reason in result.reason, result.reason


def test_declared_transitive_import_still_requires_the_second_file(tmp_path):
    source = "from first import check\ncheck()\n"
    main = tmp_path / "program.py"
    main.write_text(source)
    (tmp_path / "first.py").write_text("from second import check\n")
    (tmp_path / "second.py").write_text("def check():\n    return None\n")
    environment = _manifest(tmp_path, ["first.py"])

    result = determinism_census(source, str(main), declared_files=environment.files)

    assert not result.ok and "second" in result.reason


def test_declared_root_name_blocks_dotted_allowed_module_but_not_leaf_collision(tmp_path):
    source = "import os.path as osp\nosp.basename('leaf')\n"
    main = tmp_path / "program.py"
    main.write_text(source)
    (tmp_path / "os.py").write_text("value = 1\n")
    environment = _manifest(tmp_path, ["os.py"])
    try:
        Resolver(ast.parse(source), {"os", "os.path"}, filename=str(main),
                 _declared_files=environment.files)
    except Top as error:
        assert "os" in str(error)
    else:
        raise AssertionError("declared os.py bypassed os.path root collision")

    (tmp_path / "os.py").unlink()
    (tmp_path / "path.py").write_text("value = 1\n")
    environment = _manifest(tmp_path, ["path.py"])
    Resolver(ast.parse(source), {"os", "os.path"}, filename=str(main),
             _declared_files=environment.files)


def test_declared_files_do_not_create_fallback_roots_for_off_directory_modules(tmp_path):
    main = tmp_path / "program.py"
    main.write_text("from first import check\n")
    subdir = tmp_path / "package"
    subdir.mkdir()
    first = subdir / "first.py"
    second = subdir / "second.py"
    first.write_text("from second import check\n")
    second.write_text("def check():\n    return None\n")
    environment = TaskEnvironment((str(main), str(first), str(second)))

    result = determinism_census(main.read_text(), str(main), declared_files=environment.files)
    child_result = determinism_census(first.read_text(), str(first), declared_files=environment.files)

    assert not result.ok and "first" in result.reason
    assert not child_result.ok and "second" in child_result.reason
