"""``tharkun.__version__`` is a literal (editable installs freeze
``importlib.metadata`` at install time, so it cannot be the source of truth);
this keeps it in step with ``pyproject.toml`` and the changelog (#0608).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tharkun

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert tharkun.__version__ == data["project"]["version"]


def test_the_sv_verifier_and_the_witness_producer_report_the_package_version() -> None:
    # ``tharkun-sv --version`` and the witness producer name one version, the package's
    import ast
    from tharkun.sv.__main__ import VERSION
    assert VERSION == f"tharkun-sv {tharkun.__version__}"
    source = (ROOT / "src" / "tharkun" / "solve" / "obligations.py").read_text(encoding="utf-8")
    producers = [ast.literal_eval(n) for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Dict)
                 and any(isinstance(k, ast.Constant) and k.value == "name" for k in n.keys)
                 and any(isinstance(v, ast.Constant) and v.value == "Tharkun" for v in n.values)]
    assert producers and all(p["version"] == tharkun.__version__ for p in producers), producers


def test_changelog_has_an_entry_for_the_version() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{tharkun.__version__}]" in text


def test_layer_extras_are_declared() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    # the release drops the kb (egglog) and tools (MCP) extras with the modules that used them
    assert {"search", "avengers", "fuzz"} <= set(extras)
    assert not {"kb", "tools", "mcp", "egglog", "bench"} & set(extras)
    # one-release alias of the pre-0.2.0 name
    assert extras["hybrid"] == extras["fuzz"]
