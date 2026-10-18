"""The shipped SV command never loads an external object-generation engine.

Pynguin/Ghostwriter live inside Avengers as an availability-gated backend
(``tharkun.avengers.backends``, the ``backends``/``bench`` extras) -- that is
allowed and exercised by ``tests/test_avengers_backends.py``. What is not
allowed is SV (or anything else outside ``avengers/``) loading either engine
directly: the SV entry point never requests Avengers' ``backends=`` argument,
so a cold `python -m tharkun.sv` run never imports ``pynguin`` or ``black``.
"""

import os
import subprocess
import sys
from pathlib import Path


def test_sv_never_imports_pynguin_or_black(tmp_path):
    (tmp_path / "sitecustomize.py").write_text("""
import sys
class RejectOutsideEngine:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pynguin', 'black'}:
            raise RuntimeError(f'forbidden SV import: {fullname}')
sys.meta_path.insert(0, RejectOutsideEngine())
""")
    prop = tmp_path / "index.prp"
    prop.write_text("CHECK( init(main()), ! uncaught(IndexError))\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((
        str(tmp_path), str(Path(__file__).resolve().parents[1] / "src"),
        os.environ.get("PYTHONPATH", ""))))
    for name in ("pynguin", "black"):
        blocked = subprocess.run([sys.executable, "-c", f"import {name}"],
                                 env=env, capture_output=True, text=True)
        assert "forbidden SV import" in blocked.stderr
    for source in (
        "if __name__ == '__main__':\n    pass\n",
        "if __name__ == '__main__':\n    [][0]\n",
    ):
        program = tmp_path / "task.py"
        program.write_text(source)
        result = subprocess.run(
            [sys.executable, "-m", "tharkun.sv", str(program), "--property", str(prop),
             "--budget-seconds", "60", "--max-runs", "1", "--jobs", "1"],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert "forbidden SV import" not in result.stderr


def test_backends_are_an_avengers_only_extra_not_a_core_dependency():
    """Pynguin/black stay an opt-in ``avengers``-scoped extra, never core deps."""

    project = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert '"hypothesis>=6.0", "z3-solver>=4.12", "PyYAML>=6.0"]' in project, "core deps changed shape"
    backends_line = next(
        line for line in project.splitlines() if line.strip().startswith("backends")
    )
    assert "pynguin" in backends_line and "black" in backends_line
    assert (Path(__file__).resolve().parents[1] / "src/tharkun/avengers/backends.py").is_file()


def test_avengers_backends_are_never_imported_outside_avengers():
    """``sv/`` (and everything else outside ``avengers/``) may only reach the
    object-construction surface through ``tharkun.avengers``'s public API --
    never by importing the backend module, ``pynguin`` or ``black`` itself."""

    import ast

    root = Path(__file__).resolve().parents[1] / "src" / "tharkun"
    forbidden = {"tharkun.avengers.backends", "pynguin", "black"}
    bad = []
    for path in sorted(root.rglob("*.py")):
        if "avengers" in path.relative_to(root).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = set()
            if isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
            if names & forbidden:
                bad.append((path.relative_to(root).as_posix(), sorted(names & forbidden)))
    assert bad == [], bad
