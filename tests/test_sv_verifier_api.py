"""Missing verifier-model APIs cannot establish a property verdict."""

import sys
from pathlib import Path

import pytest

from tharkun.sv import _child, engine, scheduler


MISSING = [
    "import _sv_verifier\n_sv_verifier.missing_api_forever(1, int)\n",
    "import _sv_verifier as sv\nsv.missing_api_forever(1, int)\n",
    "import _sv_verifier\na = _sv_verifier\nb = a\nb.missing_api_forever(1, int)\n",
    "import _sv_verifier as sv\ngetattr(sv, 'missing_api_forever')(1, int)\n",
    "import _sv_verifier as sv\ngetattr(sv, 'missing_api_forever', None)\n",
    "import _sv_verifier as sv\nname = 'missing_api_' + 'forever'\ngetattr(sv, name)(1, int)\n",
    "from _sv_verifier import missing_api_forever as check\ncheck(1, int)\n",
    "import _sv_verifier as sv\nfrom builtins import getattr as get\nget(sv, 'missing_api_forever')\n",
    "import _sv_verifier as sv\nimport builtins as bi\nbi.getattr(sv, 'missing_api_forever')\n",
    "import _sv_verifier as sv\nget = getattr\nget(sv, 'missing_api_forever')\n",
    "import _sv_verifier as sv\ndef call():\n    sv.missing_api_forever(1, int)\ncall()\n",
]

STAR = [
    ("nondet_int\nraise AttributeError('ordinary target failure')\n", "AttributeError"),
    ("missing_api_forever(1, int)\n", "NameError"),
    ("future_verifier_api()\n", "NameError"),
    ("try:\n    missing_api_forever(1, int)\nexcept NameError:\n    pass\nraise TypeError\n", "TypeError"),
]


def program(tmp_path, source):
    path = tmp_path / "program.py"
    path.write_text(source)
    return str(path)


@pytest.mark.parametrize("source", MISSING)
def test_missing_api_refused_before_analysis_or_search(tmp_path, monkeypatch, source):
    from tharkun.solve import census

    path = program(tmp_path, source)

    def forbidden(*args, **kwargs):
        pytest.fail("missing API admission must precede analysis and search")

    for module, name in [(engine, "run_once"), (engine, "_static_proof"),
                         (engine, "native_import_reason"), (census, "obligation_census"),
                         (scheduler, "publish_module_namespace")]:
        monkeypatch.setattr(module, name, forbidden)
    verdict = engine.verify(path, ("AttributeError", "TypeError"), witness_dir=tmp_path / "witnesses")
    assert verdict.verdict == "unknown" and verdict.runs == 0
    assert verdict.witness is None and verdict.exception is None
    assert verdict.reason.startswith("unsupported _sv_verifier attribute:")
    assert not (tmp_path / "witnesses").exists()


@pytest.mark.parametrize("name", ["args_assigned_call", "args_call", "args_assigned_call_unsafe", "args_nested_call_unsafe"])
def test_pending_type_eval_programs_are_admissible(name):
    path = Path(__file__).parent / "fixtures" / "sv_missing_verifier" / (name + ".py")
    assert engine.verifier_api_reason(engine.read_source(str(path))) is None


@pytest.mark.parametrize("source", MISSING + [
    "import _sv_verifier as sv\ntry:\n    sv.missing_api_forever(1, int)\nexcept AttributeError:\n    pass\nraise TypeError('after caught missing API')\n",
    "import _sv_verifier as sv\ngetattr(sv, 'missing_api_forever', None)\nraise TypeError('after defaulted lookup')\n",
])
def test_missing_lookup_taints_fresh_replay_even_when_caught(tmp_path, source):
    path = program(tmp_path, source)
    classes = ("AttributeError", "TypeError", "ImportError")
    out = engine.run_once(sys.executable, path, [], timeout=30)
    assert "model_error" in out.flags
    assert out.violated(classes) is None
    confirmed, replayed = engine.replay(path, {"vector": []}, classes, timeout=30)
    assert not confirmed and "model_error" in replayed.flags


@pytest.mark.parametrize("style", ["direct", "from", "getattr"])
def test_every_defined_export_remains_available_without_taint(tmp_path, style):
    names = [name for name in vars(_child._make_sv_verifier()) if not name.startswith("__")]
    assert names
    if style == "direct":
        source = "import _sv_verifier as sv\n" + "\n".join("sv." + name for name in names)
    elif style == "from":
        source = "from _sv_verifier import " + ", ".join(names)
    else:
        source = "import _sv_verifier as sv\n" + "\n".join(f"getattr(sv, {name!r})" for name in names)
    assert engine.verifier_api_reason(source) is None
    path = program(tmp_path, source + "\nraise AttributeError('ordinary target failure')\n")
    out = engine.run_once(sys.executable, path, [], timeout=30)
    assert "model_error" not in out.flags and out.violated(("AttributeError",)) == "AttributeError"


@pytest.mark.parametrize("body,exception", STAR)
def test_star_import_is_refused_before_analysis_or_search(tmp_path, monkeypatch, body, exception):
    from tharkun.solve import census

    path = program(tmp_path, "from _sv_verifier import *\n" + body)
    monkeypatch.setattr(census, "obligation_census", lambda *a, **k: pytest.fail("must refuse before census"))
    monkeypatch.setattr(engine, "run_once", lambda *a, **k: pytest.fail("must refuse before search"))
    verdict = engine.verify(path, (exception,), witness_dir=tmp_path / "witnesses")
    assert verdict.verdict == "unknown" and verdict.runs == 0 and verdict.witness is None
    assert verdict.reason == "unsupported _sv_verifier star import"
    assert not (tmp_path / "witnesses").exists()


@pytest.mark.parametrize("body,exception", STAR)
@pytest.mark.parametrize("in_helper", [False, True])
def test_star_import_taints_direct_and_helper_runs_even_when_nameerror_is_caught(tmp_path, body, exception, in_helper):
    source = "from _sv_verifier import *\n" + body
    path = program(tmp_path, "import helper\n" if in_helper else source)
    files = [path]
    if in_helper:
        helper = tmp_path / "helper.py"
        helper.write_text(source)
        files.append(str(helper))
    out = engine.run_once(sys.executable, path, [], timeout=30, files=files)
    assert out.status == "raised" and out.exc == exception
    assert "model_error" in out.flags and out.violated((exception,)) is None


def test_defined_module_attributes_do_not_set_model_error(monkeypatch):
    monkeypatch.setattr(_child._State, "flags", set())
    module = _child._make_sv_verifier()
    for name in dir(module):
        getattr(module, name)
        assert engine.verifier_api_reason(f"import _sv_verifier\n_sv_verifier.{name}\n") is None
    assert not _child._State.flags


@pytest.mark.parametrize("source,exception", [
    ("object().missing\n", "AttributeError"), ("missing_bare_name\n", "NameError"),
])
def test_an_ordinary_lookup_error_is_still_a_witness(tmp_path, source, exception):
    path = program(tmp_path, source)
    assert engine.verifier_api_reason(source) is None
    confirmed, out = engine.replay(path, {"vector": []}, (exception,), timeout=30)
    assert confirmed and out.violated((exception,)) == exception


def test_failure_before_missing_lookup_remains_a_genuine_replay(tmp_path):
    path = program(tmp_path, "import _sv_verifier\nobject().missing\n_sv_verifier.missing_api_forever(1, int)\n")
    confirmed, out = engine.replay(path, {"vector": []}, ("AttributeError",), timeout=30)
    assert confirmed and "model_error" not in out.flags
    assert out.violated(("AttributeError",)) == "AttributeError"


@pytest.mark.parametrize("name,vector,exception", [
    ("args_assigned_call_unsafe", [{"kind": "bool", "value": False}], "AttributeError"),
    ("args_nested_call_unsafe", [{"kind": "int", "value": 0}, {"kind": "bool", "value": False}], "TypeError"),
])
def test_pending_unsafe_failure_precedes_missing_lookup(name, vector, exception):
    path = Path(__file__).parent / "fixtures" / "sv_missing_verifier" / (name + ".py")
    confirmed, out = engine.replay(str(path), {"vector": vector}, (exception,), timeout=30)
    assert confirmed and "model_error" not in out.flags
    assert out.violated((exception,)) == exception


@pytest.mark.parametrize("name", ["__path__", "__all__"])
@pytest.mark.parametrize("lookup", ["direct", "from", "computed", "caught"])
def test_explicit_metadata_lookup_in_imported_helper_taints_replay(tmp_path, name, lookup):
    helper = tmp_path / "helper.py"
    source = "import _sv_verifier as sv\n"
    if lookup == "direct":
        source += f"sv.{name}\n"
    elif lookup == "from":
        source += f"from _sv_verifier import {name}\n"
    else:
        source += f"name = '__' + {name[2:]!r}\n"
        if lookup == "caught":
            source += "try:\n    getattr(sv, name)\nexcept AttributeError:\n    pass\nraise TypeError\n"
        else:
            source += "getattr(sv, name)\n"
    helper.write_text(source)
    path = program(tmp_path, "import helper\n")
    out = engine.run_once(sys.executable, path, [], timeout=30, files=(path, str(helper)))
    expected = "TypeError" if lookup == "caught" else "ImportError" if lookup == "from" else "AttributeError"
    assert out.status == "raised" and out.exc == expected
    assert "model_error" in out.flags
    assert out.violated(("AttributeError", "ImportError", "TypeError")) is None


def test_shared_in_process_model_rejects_and_restores_taint(tmp_path):
    from tharkun.sv.nondet_harness import drive

    path = program(tmp_path, "import _sv_verifier\ngetattr(_sv_verifier, 'missing_api_forever', None)\nraise TypeError\n")
    previous = _child._State.flags
    out = drive(path, [])
    assert "model_error" in out.flags and out.violated(("TypeError",)) is None
    assert _child._State.flags is previous


def test_other_modules_and_relative_imports_are_not_the_verifier_api():
    assert engine.verifier_api_reason("import other as _sv_verifier\n_sv_verifier.missing_api_forever(1, int)\n") is None
    assert engine.verifier_api_reason("from ._sv_verifier import missing_api_forever\n") is None
