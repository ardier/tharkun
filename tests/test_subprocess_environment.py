"""Inherited import paths keep their parent meaning when confined children change cwd."""
import os
from pathlib import Path
import sys

import pytest

from tharkun.solve.environment import subprocess_environment


@pytest.mark.parametrize("case", ["absent", "empty", "relative", "absolute", "empty-fields", "mixed"])
def test_subprocess_environment_preserves_path_entries(tmp_path, monkeypatch, case):
    monkeypatch.chdir(tmp_path)
    cwd = os.getcwd()
    absolute = str(tmp_path / "absolute")
    value, expected = {
        "absent": (None, None),
        "empty": ("", ""),
        "relative": ("src", cwd + os.sep + "src"),
        "absolute": (absolute, absolute),
        "empty-fields": (os.pathsep, os.pathsep.join([cwd, cwd])),
        "mixed": (os.pathsep.join(["src", "", "link/../libs", absolute, "src", ""]),
                  os.pathsep.join([cwd + "/src", cwd, cwd + "/link/../libs", absolute, cwd + "/src", cwd])),
    }[case]
    if value is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", value)
    before, path = dict(os.environ), list(sys.path)
    result = subprocess_environment()
    assert result.get("PYTHONPATH") == expected
    assert ("PYTHONPATH" in result) is (value is not None)
    assert {k: v for k, v in result.items() if k != "PYTHONPATH"} == {
        k: v for k, v in before.items() if k != "PYTHONPATH"}
    assert dict(os.environ) == before and sys.path == path
    assert result is not os.environ


@pytest.mark.parametrize("minimal_env", [False, True])
def test_main_child_runs_with_relative_pythonpath_in_scratch(tmp_path, monkeypatch, minimal_env):
    from tharkun.sv import engine

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setenv("PYTHONPATH", "src")
    data = tmp_path / "declared.txt"
    data.write_text("declared value")
    target = tmp_path / "target.py"
    target.write_text("with open('declared.txt') as stream:\n    value = stream.read()\n"
                      "with open('scratch-output', 'w') as stream:\n    stream.write(value)\n"
                      "raise KeyError(value)\n")
    scratches = []
    native_run = engine.subprocess.run

    def launch(*args, **kwargs):
        scratch = Path(kwargs["cwd"])
        assert scratch.is_dir() and scratch != root and scratch != tmp_path
        assert "PYTHONPATH" not in kwargs["env"]
        scratches.append(scratch)
        return native_run(*args, **kwargs)

    monkeypatch.setattr(engine.subprocess, "run", launch)
    out = engine.run_once(sys.executable, str(target), [], timeout=30,
                          files=[str(target), str(data)], minimal_env=minimal_env)
    assert out.violated(("KeyError",)) == "KeyError" and "declared value" in out.msg, out
    assert set(out.flags) == {"file_effect"}, out
    assert len(scratches) == 1 and not scratches[0].exists()
    assert not (tmp_path / "scratch-output").exists() and not (root / "scratch-output").exists()
    assert data.read_text() == "declared value" and os.environ["PYTHONPATH"] == "src"


def test_isolated_import_preserves_relative_dependency_path(tmp_path, monkeypatch):
    from tharkun.solve.loader import import_project_isolated

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "libs")
    libraries = tmp_path / "libs"
    libraries.mkdir()
    helper = libraries / "relative_path_dependency.py"
    helper.write_text("VALUE = 42\n")
    project = tmp_path / "project"
    project.mkdir()
    target = project / "target.py"
    target.write_text("from relative_path_dependency import VALUE\nassert VALUE == 42\n")
    report = import_project_isolated(str(target), files=[str(target), str(helper)])
    assert report["ok"] and "VALUE" in report["names"], report
    assert not report["flags"]
    assert "relative_path_dependency" not in sys.modules


def test_confined_pynguin_preserves_relative_pythonpath(tmp_path, monkeypatch):
    from tharkun.avengers import ObjectGoal, backends

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([os.path.relpath(root / "src", tmp_path), "libs"]))
    monkeypatch.setattr(backends, "pynguin_available", lambda interpreter: True)
    libraries = tmp_path / "libs"
    libraries.mkdir()
    generated = "def test_generated():\n    assert True\n"
    fake = libraries / "pynguin.py"
    fake.write_text("import sys\nfrom pathlib import Path\n"
                    "output = Path(sys.argv[sys.argv.index('--output-path') + 1])\n"
                    f"(output / 'test_generated.py').write_text({generated!r})\n")
    project = tmp_path / "project"
    project.mkdir()
    target = project / "target.py"
    target.write_text("class Box: pass\n")
    previous = sys.modules.get("pynguin")
    artifacts = backends.PynguinBackendPass(files=(str(target), str(fake)), timeout_s=30).generate(
        ObjectGoal(str(target), "Box"))
    assert [(artifact.tool, artifact.source) for artifact in artifacts] == [("pynguin", generated)]
    assert not (project / "test_generated.py").exists()
    assert sys.modules.get("pynguin") is previous
