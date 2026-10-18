"""Negative-first boundary checks for the standalone witness differential."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace

import pytest

from tharkun.solve.obligations import Invariant, witness
from tharkun.solve.environment import subprocess_environment
from tharkun.sv.witness import dumps

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("store_witness_diff", ROOT / "scripts/store_witness_diff.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SPECIFICATION = "CHECK( init(main()), ! uncaught(ZeroDivisionError))\n"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row(tmp_path, source="x = 1\npass\n", values=((2, 1, "<module>", "True", "location_invariant"),), *, name="task", metrics=None, records=None):
    program = tmp_path / f"{name}.py"
    program.write_text(source)
    prop = tmp_path / "arithmetic.prp"
    prop.write_text(SPECIFICATION)
    invariants = [Invariant(str(program), line, column, function, value, kind)
                  for line, column, function, value, kind in values]
    artifact = tmp_path / f"{name}.witness.yml"
    artifact.write_text(dumps(witness(source, str(program), SPECIFICATION, invariants,
                                      source_bytes=program.read_bytes())))
    return {"task": name, "source": str(program), "source_sha256": sha(program),
            "property_file": str(prop), "property_sha256": sha(prop),
            "witness": {"path": str(artifact), "sha256": sha(artifact)},
            "metrics": metrics or {"checked_sites": 0, "safe_sites": 0,
                                    "established_sites": 0, "sites_established_and_rendered": 0, "reasons": {}},
            "records": records or []}


def packet(tmp_path, rows, name="packet.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"rows": rows}))
    return path


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"rows": [], "rows": []}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        MODULE.load_rows(path)


def test_empty_reference_and_missing_candidate_never_green(tmp_path):
    with pytest.raises(ValueError, match="empty reference"):
        MODULE.compare([], [])
    reference = MODULE.load_rows(packet(tmp_path, [row(tmp_path)]))
    result = MODULE.compare(reference, [])
    assert not result["task_inclusion"] and len(result["missing_store"]) == 1
    output = tmp_path / "result.json"
    empty = packet(tmp_path, [], "empty.json")
    assert MODULE.main([str(empty), str(empty), "--output", str(output)]) == 2


@pytest.mark.parametrize("field", ["source", "property_file", "witness"])
def test_raw_hash_mismatch_rejected(tmp_path, field):
    item = row(tmp_path)
    path = item["witness"]["path"] if field == "witness" else item[field]
    with Path(path).open("a") as file:
        file.write("\nchanged\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        MODULE.load_rows(packet(tmp_path, [item]))


def test_matching_task_name_cannot_override_pair_hash_binding(tmp_path):
    item = row(tmp_path)
    assert MODULE.pair_key(item) == ("task", item["source_sha256"], item["property_sha256"])
    other = {**item, "property_sha256": "0" * 64}
    assert MODULE.pair_key(item) != MODULE.pair_key(other)


def test_witness_metadata_is_authenticated_not_only_file_hash(tmp_path):
    item = row(tmp_path)
    text = Path(item["witness"]["path"]).read_text().replace(item["source_sha256"], "0" * 64)
    Path(item["witness"]["path"]).write_text(text)
    item["witness"]["sha256"] = sha(item["witness"]["path"])
    with pytest.raises(ValueError, match="source/property/semantics mismatch"):
        MODULE.load_rows(packet(tmp_path, [item]))


def test_duplicate_pair_and_identical_invariant_rejected(tmp_path):
    item = row(tmp_path)
    with pytest.raises(ValueError, match="duplicate packet pair"):
        MODULE.load_rows(packet(tmp_path, [item, item]))
    duplicate = row(tmp_path, values=((2, 1, "<module>", "True", "location_invariant"),) * 2)
    with pytest.raises(ValueError, match="duplicate invariant"):
        MODULE.load_rows(packet(tmp_path, [duplicate]))


def test_distinct_same_location_claims_form_conjunction(tmp_path):
    item = row(tmp_path, values=((2, 1, "<module>", "type(x) is int and x > 0", "location_invariant"),
                                  (2, 1, "<module>", "type(x) is int and x < 10", "location_invariant")))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    target = Invariant(item["source"], 2, 1, "<module>", "type(x) is int and x < 20")
    assert len(loaded["invariants"]) == 2
    assert MODULE.implication(loaded["invariants"], target, loaded["tree"], builtin_names={"type", "int"})["status"] == "yes"
    assert MODULE.implication([loaded["invariants"][0]], target, loaded["tree"], builtin_names={"type", "int"})["status"] == "no"


def test_original_character_columns_and_dotted_function_are_required(tmp_path):
    source = "def outer():\n    def inner():\n        a = 'é'; x = 1; pass\n    inner()\nouter()\n"
    item = row(tmp_path, source, ((3, 25, "outer.inner", "True", "location_invariant"),))
    # The semicolon's pass begins at character column 25, not UTF-8 byte 26.
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    assert loaded["invariants"][0].column == 25
    bad = row(tmp_path, source, ((3, 26, "outer.inner", "True", "location_invariant"),))
    with pytest.raises(ValueError, match="original statement"):
        MODULE.load_rows(packet(tmp_path, [bad]))


def test_extra_candidate_invariants_and_tasks_are_not_dropped(tmp_path, monkeypatch):
    ref = MODULE.load_rows(packet(tmp_path, [row(tmp_path)], "ref.json"))
    extra = row(tmp_path, name="extra", values=((2, 1, "<module>", "False", "location_invariant"),))
    stores = MODULE.load_rows(packet(tmp_path, [row(tmp_path), extra], "store.json"))
    monkeypatch.setattr(MODULE, "_independent", lambda *args: {"confirmed": False, "reason": "unsupported", "open_obligations": 0})
    monkeypatch.setattr(MODULE, "native_check", lambda candidate, **kwargs: {"visits": [1], "failures": {"0": {"kind": "false", "reason": "false"}} if candidate["invariants"][0].value == "False" else {}, "outcomes": []})
    result = MODULE.compare(ref, stores)
    assert len(result["per_task"]) == 2 and result["oracle_false"] == 1
    assert result["categories"]["REFUTED"] == 1


def test_true_reference_implication_never_confirms_candidate_truth(tmp_path, monkeypatch):
    ref = MODULE.load_rows(packet(tmp_path, [row(tmp_path)], "ref.json"))
    item = row(tmp_path, values=((2, 1, "<module>", "False", "location_invariant"),))
    stores = MODULE.load_rows(packet(tmp_path, [item], "store.json"))
    # The input source hashes remain identical: only witness predicate differs.
    assert MODULE.pair_key(ref[0]) == MODULE.pair_key(stores[0])
    monkeypatch.setattr(MODULE, "_independent", lambda *args: {"confirmed": False, "reason": "unknown", "open_obligations": 0})
    monkeypatch.setattr(MODULE, "native_check", lambda *args, **kwargs: {"visits": [0], "failures": {}, "outcomes": []})
    result = MODULE.compare(ref, stores)
    assert result["per_task"][0]["implications"][0]["status"] == "yes"
    assert result["categories"]["confirmed"] == 0 and result["categories"]["not_proved"] == 1
    assert not result["m6_ready"]


def test_unknown_is_reported_but_not_milestone1_red(tmp_path, monkeypatch):
    item = row(tmp_path)
    ref = packet(tmp_path, [item], "ref.json")
    store = packet(tmp_path, [item], "store.json")
    monkeypatch.setattr(MODULE, "_independent", lambda *args: {"confirmed": False, "reason": "unknown", "open_obligations": 0})
    monkeypatch.setattr(MODULE, "native_check", lambda *args, **kwargs: {"visits": [0], "failures": {}, "outcomes": []})
    output = tmp_path / "result.json"
    assert MODULE.main([str(ref), str(store), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["categories"]["not_proved"] == 1 and result["unproved"] == 1
    assert not result["m6_ready"]


def test_native_category_stays_bounded_and_unconfirmed(tmp_path, monkeypatch):
    rows = MODULE.load_rows(packet(tmp_path, [row(tmp_path)]))
    monkeypatch.setattr(MODULE, "_independent", lambda *args: {"confirmed": False, "reason": "unsupported", "open_obligations": 0})
    monkeypatch.setattr(MODULE, "native_check", lambda *args, **kwargs: {"visits": [4], "failures": {}, "outcomes": [], "run_count": 2, "draw_count": 3})
    result = MODULE.compare(rows, rows)
    assert result["categories"]["confirmed"] == 0
    assert result["categories"]["reference_cannot_visit_or_express"] == 1
    assert result["per_task"][0]["invariants"][0]["bounded_native"] is True


def test_bare_true_fails_only_when_established_site_renders(tmp_path, monkeypatch):
    metrics = {"checked_sites": 1, "safe_sites": 1, "established_sites": 1, "sites_established_and_rendered": 0,
               "reasons": {"store_only:none_test_outside_witness_grammar": 1}}
    item = row(tmp_path, metrics=metrics, records=[{"status": "safe", "binding": {}, "contract": {}, "facts": []}])
    ref = packet(tmp_path, [item], "ref.json")
    store = packet(tmp_path, [item], "store.json")
    monkeypatch.setattr(MODULE, "_independent", lambda *args: {"confirmed": True, "reason": "", "open_obligations": 0})
    output = tmp_path / "result.json"
    assert MODULE.main([str(ref), str(store), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert len(result["true_only"]) == 1 and not result["bare_true_with_check_sites"]
    assert result["record_counts"]["facts"] == {"with": 0, "without": 1}
    item["metrics"]["sites_established_and_rendered"] = 1
    store.write_text(json.dumps({"rows": [item]}))
    assert MODULE.main([str(ref), str(store), "--output", str(output)]) == 2


def test_closed_predicates_do_not_inflate_nontrivial_rate(tmp_path):
    item = row(tmp_path, values=((2, 1, "<module>", "(type(1) is int)", "location_invariant"),
                                  (2, 1, "<module>", "1 < 2", "location_invariant")))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    counts = MODULE._nontrivial(loaded["invariants"], loaded["tree"])
    assert counts == {"source_variable": 0, "literal_true": 0, "closed_predicates": 2, "closed_true": 1}


def test_missing_store_capture_is_an_evidence_error(tmp_path):
    loaded = MODULE.load_rows(packet(tmp_path, [row(tmp_path)]))
    del loaded[0]["metrics"]
    with pytest.raises(ValueError, match="captured"):
        MODULE.compare(loaded, loaded)


def test_reference56_mapping_reads_and_binds_raw_witnesses(tmp_path):
    rows = {}
    for index in range(56):
        item = row(tmp_path, name=f"task{index}")
        key = list(MODULE.pair_key(item))
        rows[json.dumps(key)] = {"task": item["task"], "key_tuple": key,
            "source": {"path": item["source"], "sha256": item["source_sha256"]},
            "property": {"file": item["property_file"], "sha256": item["property_sha256"]},
            "witness": {**item["witness"], "exact_yaml_utf8": Path(item["witness"]["path"]).read_text()},
            "reference_confirmation": {"actual_verify_proof_checked": True,
                                       "actual_export_external": {"confirmed": True}}}
    path = tmp_path / "REFERENCE56.json"
    path.write_text(json.dumps({"format": "reference56-static-inventory-v1", "rows": rows}))
    assert len(MODULE.load_rows(path)) == 56
    next(iter(rows.values()))["witness"]["exact_yaml_utf8"] += "\n"
    path.write_text(json.dumps({"format": "reference56-static-inventory-v1", "rows": rows}))
    with pytest.raises(ValueError, match="raw reference witness"):
        MODULE.load_rows(path)


def test_min_loader_uses_exact_pair_and_authenticated_checked_artifacts(tmp_path, monkeypatch):
    item = row(tmp_path)
    pair = {key: item[key] for key in ("task", "source", "source_sha256", "property_file", "property_sha256")}
    pair.update({"key": "task|arithmetic", "index": 0})
    public = tmp_path / "PUBLIC-VERDICT.json"
    public.write_text(json.dumps({"pair": pair, "public_verdict": {"verdict": "true", "detail": {"proof_checked": True}}, "serialization_complete": True}))
    # Retained result artifacts use paths relative to this public-verdict directory.
    proof = tmp_path / "proofs"
    proof.mkdir()
    witness_path = proof / "witness.yml"
    witness_path.write_bytes(Path(item["witness"]["path"]).read_bytes())
    result = {"results": [{"key": pair["key"], "index": 0, "proof_checked_state": True,
              "status": "RETURNED", "verdict": "true", "timed_out": False, "rc": 0,
              "public_verdict_path": str(public), "artifacts": {"PUBLIC-VERDICT.json": {"sha256": sha(public)},
              "proofs/witness.yml": {"sha256": sha(witness_path)}}}]}
    reads = []
    original_read = Path.read_bytes

    def capture(path):
        if path == public:
            reads.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", capture)
    assert len(MODULE._load_min([pair], result)) == 1
    assert reads == [public]
    result["results"][0]["key"] = "task|dynamic-typing"
    with pytest.raises(ValueError, match="unknown MIN"):
        MODULE._load_min([pair], result)


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
@pytest.mark.parametrize("value", ["type(x) is int and x < 2", "False"])
def test_native_assert_detects_later_and_final_false_loop_visit(tmp_path, value):
    source = "def main():\n    x = 0\n    while x < 2:\n        x += 1\nmain()\n"
    item = row(tmp_path, source, ((3, 5, "main", value, "loop_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["failures"]["0"]["kind"] == "false"
    assert result["visits"][0] == 3  # falsity is recorded without changing source control flow


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_caught_native_assert_cannot_hide_refutation(tmp_path):
    source = "def main():\n    try:\n        x = 1\n    except AssertionError:\n        pass\nmain()\n"
    item = row(tmp_path, source, ((3, 9, "main", "False", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["outcomes"][0]["status"] == "clean"
    assert result["failures"]["0"]["kind"] == "false"


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_unvisited_and_shadowed_builtins_are_not_native_proof(tmp_path):
    item = row(tmp_path, "if False:\n    pass\n", ((2, 5, "<module>", "False", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    assert MODULE.native_check(loaded, runs=1)["visits"] == [0]
    source = "type = lambda x: int\nx = 1\npass\n"
    item = row(tmp_path, source, ((3, 1, "<module>", "type(x) is int", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["failures"]["0"]["kind"] == "not_proved"


def test_trace_selection_leaves_original_else_break_continue_and_preamble_untouched(tmp_path):
    source = ('"module doc"\nfrom __future__ import annotations\n'
              'def outer():\n    "outer doc"\n    def inner():\n'
              '        "inner doc"\n        x = 0\n        while x < 3:\n'
              '            x += 1\n            if x == 1: continue\n'
              '            if x == 2: break\n        else:\n            x = 100\n'
              '        return x\n    return inner()\nassert outer() == 2\n')
    item = row(tmp_path, source, ((8, 9, "outer.inner", "type(x) is int and x >= 0", "loop_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    before = ast.dump(loaded["tree"], include_attributes=True)
    assert MODULE._trace_targets(loaded) == {("outer.inner", 8): [0]}
    assert ast.dump(loaded["tree"], include_attributes=True) == before
    instrumented = loaded["tree"]
    assert ast.get_docstring(instrumented) == "module doc"
    assert isinstance(instrumented.body[1], ast.ImportFrom) and instrumented.body[1].module == "__future__"
    loop = next(node for node in ast.walk(instrumented) if isinstance(node, ast.While))
    assert isinstance(loop.test, ast.Compare) and len(loop.orelse) == 1
    assert any(isinstance(node, ast.Continue) for node in ast.walk(loop))
    assert any(isinstance(node, ast.Break) for node in ast.walk(loop))
    assert ast.get_docstring(instrumented.body[2]) == "outer doc"


def test_strength_builtin_trust_requires_proof_record_intersection(tmp_path):
    item = row(tmp_path, values=((2, 1, "<module>", "type(x) is int", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    target = loaded["invariants"][0]
    assert MODULE.implication([target], target, loaded["tree"])["status"] == "Refused"
    path = next(path for node, (path, location) in loaded["points"].items() if isinstance(node, ast.Pass))
    loaded["records"] = [{"site": [loaded["source_sha256"], path, "TypeError"], "builtins": ["int", "type"]},
                         {"site": [loaded["source_sha256"], path, "AttributeError"], "builtins": ["type"]}]
    assert MODULE._record_builtins(loaded, target) == frozenset({"type"})
    loaded["records"][1]["builtins"] = ["int", "type"]
    trusted = MODULE._record_builtins(loaded, target)
    assert MODULE.implication([target], target, loaded["tree"], builtin_names=trusted)["status"] == "yes"
    del loaded["records"][1]["builtins"]
    assert MODULE._record_builtins(loaded, target) == frozenset()


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_bounded_native_reuses_draw_boot_with_recorded_counts(tmp_path):
    source = "from _sv_verifier import nondet_int\ndef main():\n    x = nondet_int()\n    pass\nmain()\n"
    item = row(tmp_path, source, ((4, 5, "main", "type(x) is int", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=2)
    assert result["visits"] == [2] and not result["failures"]
    assert result["run_count"] == 2 and result["draw_count"] == 2
    assert result["bounded_native"] is True and result["universal_proof"] is False


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_native_original_nested_loop_else_and_docstrings_execute(tmp_path):
    source = ('"module doc"\nfrom __future__ import annotations\n'
              'def outer():\n    "outer doc"\n    def inner():\n'
              '        "inner doc"\n        x = 0\n        while x < 3:\n'
              '            x += 1\n            if x == 1: continue\n'
              '            if x == 2: break\n        else:\n            x = 100\n'
              '        return x\n    return inner()\nassert outer() == 2\n')
    item = row(tmp_path, source, ((8, 9, "outer.inner", "type(x) is int and x >= 0", "loop_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["outcomes"][0]["status"] == "clean" and result["visits"] == [2]
    assert not result["failures"]


@pytest.mark.parametrize("invalid_draw,flags,expected", [
    (True, [], "not_proved"),
    (False, ["file_unmodelled"], "not_proved"),
    (False, ["orig_argv"], "not_proved"),
    (False, ["model_error"], "not_proved"),
    (False, ["random_state_unmodelled"], "not_proved"),
    (False, ["stack_unmodelled"], "not_proved"),
    (False, ["import_state_unmodelled"], "not_proved"),
    (False, [], "false"),
])
def test_native_false_requires_valid_draws_and_modelled_run(tmp_path, monkeypatch, invalid_draw, flags, expected):
    loaded = MODULE.load_rows(packet(tmp_path, [row(tmp_path)]))[0]
    report = {"status": "raised", "exc": "AssertionError", "invalid_draw": invalid_draw,
              "flags": flags, "draws": [], "visits": [1],
              "failures": {"0": {"kind": "false", "reason": "native assertion false"}}}
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr=""))
    result = MODULE.native_check(loaded, runs=1)
    assert result["failures"]["0"]["kind"] == expected
    assert result["observed_visits"] == [1]
    assert result["visits"] == ([1] if expected == "false" else [0])
    assert result["outcomes"][0]["eligible_for_refutation"] is (expected == "false")
    monkeypatch.setattr(MODULE, "_independent", lambda *args: {"confirmed": False, "reason": "unknown", "open_obligations": 0})
    monkeypatch.setattr(MODULE, "native_check", lambda *args, **kwargs: result)
    compared = MODULE.compare([loaded], [loaded])
    assert compared["oracle_false"] == (1 if expected == "false" else 0)
    assert compared["categories"]["REFUTED"] == (1 if expected == "false" else 0)


def test_store_canonical_rendered_metric_is_required_without_alias(tmp_path):
    loaded = MODULE.load_rows(packet(tmp_path, [row(tmp_path)]))[0]
    metrics, _ = MODULE._metrics(loaded)
    assert metrics["sites_established_and_rendered"] == 0
    loaded["metrics"]["rendered_sites"] = loaded["metrics"].pop("sites_established_and_rendered")
    with pytest.raises(ValueError, match="captured"):
        MODULE._metrics(loaded)


def test_authenticated_json_bytes_share_duplicate_key_rejection():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        MODULE._loads_json(b'{"pair": {}, "pair": {}}')


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_changed_target_source_is_rejected_before_execution(tmp_path):
    request = row(tmp_path)
    request.update({"draws": [], "seconds": 1.0})
    marker = tmp_path / "wrong-source-executed"
    code = r"""
import importlib.util, json, sys
from pathlib import Path
from tharkun.sv import _child
payload = json.load(sys.stdin)
spec = importlib.util.spec_from_file_location('oracle', payload['script'])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def changed_run(request):
    program = request['program']
    malicious = ("open(" + repr(payload['marker']) + ", 'w').write('executed')\n").encode()
    Path(program).write_bytes(malicious)
    exec(_child.compile(Path(program).read_bytes(), program, 'exec'))
    return {'status': 'clean'}
_child.run = changed_run
try:
    module._native_child(payload['request'])
except ValueError as exc:
    print(json.dumps({'reason': str(exc)}))
else:
    raise AssertionError('changed source accepted')
"""
    with tempfile.TemporaryDirectory(dir=tmp_path) as scratch:
        request["scratch"] = str(Path(scratch).resolve())
        child = subprocess.run([sys.executable, "-B", "-c", code],
                               input=json.dumps({"script": str(ROOT / "scripts/store_witness_diff.py"),
                                                 "request": request, "marker": str(marker)}),
                               capture_output=True, text=True, timeout=5, cwd=request["scratch"],
                               env=subprocess_environment())
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["reason"] == "target source changed before compilation"
    assert not marker.exists()


def test_native_scope_type_whitelist_never_calls_metaclass_equality():
    calls = []

    class Meta(type):
        def __eq__(cls, other):
            calls.append(other)
            raise AssertionError("user equality hook was called")

    class Custom(metaclass=Meta):
        pass

    assert MODULE._native_type(Custom()) == "object"
    assert not calls
    assert MODULE._native_type(1) == "int"
    assert MODULE._native_type(True) == "bool"


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_trace_docstring_future_and_class_source_semantics_are_original(tmp_path):
    source = ('"module doc"\nfrom __future__ import annotations\n'
              'class Box:\n    "class doc"\n    def method(self, x: Missing):\n'
              '        "method doc"\n        pass\n'
              'assert __doc__ == "module doc"\nassert Box.__doc__ == "class doc"\n'
              'assert Box.method.__doc__ == "method doc"\n'
              'assert Box.method.__annotations__["x"] == "Missing"\nBox().method(1)\n')
    item = row(tmp_path, source, ((1, 1, "<module>", "True", "location_invariant"),
                                 (2, 1, "<module>", "True", "location_invariant"),
                                 (6, 9, "Box.method", "True", "location_invariant"),
                                 (7, 9, "Box.method", "type(x) is int", "location_invariant")))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["outcomes"][0]["status"] == "clean"
    assert result["visits"][3] == 1 and result["visits"][2] == 0  # compiler omits function docstring
    assert not result["failures"]
    assert Path(item["source"]).read_text() == source


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_trace_refuses_ambiguous_statement_line(tmp_path):
    source = "x = 1; pass\n"
    item = row(tmp_path, source, ((1, 8, "<module>", "True", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["visits"] == [0]
    assert result["outcomes"][0]["status"] == "unsupported"
    assert "multi-statement line" in result["outcomes"][0]["reason"]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
@pytest.mark.parametrize("scope", ["local", "cell", "free", "global"])
def test_trace_missing_lexical_names_cannot_fall_back_to_global(tmp_path, scope):
    sources = {
        "local": ("x = 1\ndef main():\n    pass\n    x = 2\nmain()\n", 3, "main"),
        "cell": ("x = 1\ndef main():\n    def inner():\n        return x\n    pass\n    x = 2\nmain()\n", 5, "main"),
        "free": ("x = 1\ndef outer():\n    def inner():\n        pass\n        return x\n    inner()\n    x = 2\nouter()\n", 4, "outer.inner"),
        "global": ("x = 1\ndef main():\n    global x\n    pass\n    x = 2\nmain()\n", 4, "main"),
    }
    source, line, function = sources[scope]
    item = row(tmp_path, source, ((line, 5 if scope != "free" else 9, function, "type(x) is int", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["visits"] == [1]
    if scope == "global":
        assert not result["failures"]
    else:
        assert "native lexical name unbound: x" in result["failures"]["0"]["reason"]
        assert result["failures"]["0"]["kind"] == "not_proved"


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
@pytest.mark.parametrize("error,bound", [("AssertionError", False), ("ZeroDivisionError", True)])
def test_trace_source_exception_is_distinct_from_invariant_falsity(tmp_path, error, bound):
    source = "x = 1\npass\nraise " + error + "()\n"
    item = row(tmp_path, source, ((2, 1, "<module>", "type(x) is int", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    outcome = result["outcomes"][0]
    assert result["visits"] == [1] and "0" not in result["failures"]
    assert outcome["source_stop"] == ("bound_property" if bound else "outside_property")
    assert outcome["bound_property_violation"] == ("ZeroDivisionError" if bound else None)
    if bound:
        assert "run" in result["failures"]
    else:
        assert not result["failures"]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_trace_line_budget_stops_honestly_without_refutation(tmp_path):
    source = "x = 0\nwhile x < 100:\n    x += 1\n"
    item = row(tmp_path, source, ((2, 1, "<module>", "type(x) is int", "loop_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1, line_budget=5)
    assert result["outcomes"][0]["status"] == "budget_exhausted"
    assert result["outcomes"][0]["oracle_stop"] == "line_budget"
    assert result["failures"]["run"]["reason"] == "native line budget exhausted"
    assert all(failure["kind"] != "false" for failure in result["failures"].values())


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_trace_cpu_budget_returns_structured_partial_evidence(tmp_path):
    source = "x = 0\nwhile True:\n    x += 1\n"
    item = row(tmp_path, source, ((2, 1, "<module>", "type(x) is int", "loop_invariant"),))
    request = {**item, "draws": [], "seconds": 1.0, "line_budget": 400000}
    with tempfile.TemporaryDirectory(dir=tmp_path) as scratch:
        request["scratch"] = str(Path(scratch).resolve())
        child = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/store_witness_diff.py"), "--native-child"],
                               input=json.dumps(request), capture_output=True, text=True, timeout=10, cwd=request["scratch"],
                               env=subprocess_environment())
    assert child.returncode == 0, (child.returncode, child.stderr)
    outcome = json.loads(child.stdout)
    assert outcome["status"] == "budget_exhausted" and outcome["oracle_stop"] == "cpu_budget"
    assert outcome["reason"] == "native cpu budget exhausted"
    assert 0 < outcome["visits"][0] and outcome["line_events"] < outcome["line_budget"]
    assert not outcome["invalid_draw"] and not outcome["bound_property_violation"]
    assert all(failure["kind"] != "false" for failure in outcome["failures"].values())
    assert Path(item["source"]).read_text() == source


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_budget_stop_does_not_erase_an_earlier_valid_false_predicate(tmp_path):
    source = "x = 0\nwhile True:\n    x += 1\n"
    item = row(tmp_path, source, ((2, 1, "<module>", "x < 0", "loop_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1, line_budget=5)
    assert result["outcomes"][0]["status"] == "budget_exhausted"
    assert result["outcomes"][0]["eligible_for_refutation"] is True
    assert result["failures"]["0"]["kind"] == "false" and result["visits"][0] > 0


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
@pytest.mark.parametrize("tamper", ["trace_changed", "foreign_code"])
def test_cpu_stop_cannot_mask_prior_model_refusal(tmp_path, monkeypatch, tamper):
    suffix = ("sys.settrace(None)\nwhile True:\n    x += 1\n" if tamper == "trace_changed" else
              "exec(compile('while True:\\n    pass\\n', __file__, 'exec'))\n")
    source = "import sys\nx = 0\npass\n" + suffix
    item = row(tmp_path, source, ((3, 1, "<module>", "False", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    with tempfile.TemporaryDirectory(dir=tmp_path) as scratch:
        scratch = str(Path(scratch).resolve())
        child = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/store_witness_diff.py"), "--native-child"],
            input=json.dumps({**item, "draws": [], "seconds": 1.0, "line_budget": 400000, "scratch": scratch}),
            capture_output=True, text=True, timeout=10, cwd=scratch, env=subprocess_environment())
    assert child.returncode == 0, (child.returncode, child.stderr)
    outcome = json.loads(child.stdout)
    assert outcome["oracle_stop"] == tamper and "model_error" in outcome["flags"]
    assert outcome["failures"]["0"]["kind"] == "false"  # raw observation survives without becoming qualified
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: child)
    result = MODULE.native_check(loaded, runs=1)
    assert result["visits"] == [0] and result["observed_visits"] == [1]
    assert not result["outcomes"][0]["eligible_for_refutation"]
    assert result["failures"]["0"]["kind"] == "not_proved"


@pytest.mark.parametrize("returncode", [-9, 3])
def test_other_nonzero_native_exit_remains_transport_failure(tmp_path, monkeypatch, returncode):
    loaded = MODULE.load_rows(packet(tmp_path, [row(tmp_path)]))[0]
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=returncode, stdout=json.dumps({"status": "budget_exhausted"}), stderr=""))
    result = MODULE.native_check(loaded, runs=1)
    outcome = result["outcomes"][0]
    assert outcome["status"] == "child_failed" and outcome["returncode"] == returncode
    assert outcome["signal"] == (-returncode if returncode < 0 else None)
    assert not outcome["eligible_for_refutation"] and result["visits"] == [0]


def test_malformed_native_output_remains_transport_failure(tmp_path, monkeypatch):
    loaded = MODULE.load_rows(packet(tmp_path, [row(tmp_path)]))[0]
    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout="not JSON", stderr=""))
    result = MODULE.native_check(loaded, runs=1)
    assert result["outcomes"][0]["status"] == "timeout_or_transport_failure"
    assert not result["outcomes"][0]["eligible_for_refutation"] and result["visits"] == [0]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_trace_guard_only_object_never_calls_user_hooks(tmp_path):
    source = ('class Custom:\n    def __eq__(self, other):\n        raise ZeroDivisionError\n'
              'x = Custom()\npass\n')
    item = row(tmp_path, source, ((5, 1, "<module>", "type(x) is not int", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["visits"] == [1] and not result["failures"]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_trace_refuses_impure_custom_comparison_without_calling_it(tmp_path):
    source = ('class Custom:\n    def __eq__(self, other):\n        raise ZeroDivisionError\n'
              'x = Custom()\npass\n')
    item = row(tmp_path, source, ((5, 1, "<module>", "x == 1", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["outcomes"][0]["status"] == "clean"
    assert result["failures"]["0"]["kind"] == "not_proved"
    assert "comparison may invoke user code" in result["failures"]["0"]["reason"]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_trace_source_can_import_stdlib_without_blanket_builtin_refusal(tmp_path):
    source = "import math\nx = 1\npass\n"
    item = row(tmp_path, source, ((3, 1, "<module>", "type(x) is int", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["outcomes"][0]["status"] != "unsupported"
    assert result["observed_visits"] == [1]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_trace_builtin_mutation_cannot_replace_identity_baseline(tmp_path):
    source = "import builtins\nbuiltins.len = lambda x: 0\nx = 'a'\npass\n"
    item = row(tmp_path, source, ((4, 1, "<module>", "len(x) == 0", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["observed_visits"] == [1]
    assert result["failures"]["0"]["kind"] == "not_proved"
    failure = result["failures"]["0"]
    assert "builtin identity changed" in failure.get("observed_failure", failure)["reason"]
    assert result["outcomes"][0]["oracle_stop"] == "builtin_changed"


def test_trace_multiline_while_tests_do_not_claim_repeated_head_coverage(tmp_path):
    source = "x = 0\nwhile (\n    x < 2\n):\n    x += 1\n"
    item = row(tmp_path, source, ((2, 1, "<module>", "type(x) is int", "loop_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    with pytest.raises(ValueError, match="multiline while test"):
        MODULE._trace_targets(loaded)


def test_native_wall_budget_is_unavailable_not_transport_refutation(tmp_path, monkeypatch):
    loaded = MODULE.load_rows(packet(tmp_path, [row(tmp_path)]))[0]
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(MODULE.subprocess, "run", timeout)
    result = MODULE.native_check(loaded, runs=1)
    assert result["outcomes"][0]["status"] == "budget_exhausted"
    assert result["failures"]["run"]["kind"] == "not_proved" and result["visits"] == [0]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
@pytest.mark.parametrize("definition,bound", [("class E(ZeroDivisionError):", True), ("class ZeroDivisionError(Exception):", False)])
def test_trace_property_exceptions_use_actual_builtin_mro_not_exc_name(tmp_path, definition, bound):
    name = "E" if bound else "ZeroDivisionError"
    source = definition + "\n    pass\nx = 1\npass\nraise " + name + "()\n"
    item = row(tmp_path, source, ((4, 1, "<module>", "type(x) is int", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    outcome = result["outcomes"][0]
    assert outcome["source_stop"] == ("bound_property" if bound else "outside_property")
    assert ("ZeroDivisionError" in outcome["mro"]) is bound
    assert outcome["bound_property_violation"] == ("ZeroDivisionError" if bound else None)


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_source_disabling_trace_is_a_model_refusal(tmp_path):
    source = "import sys\nx = 1\npass\nsys.settrace(None)\npass\n"
    item = row(tmp_path, source, ((3, 1, "<module>", "type(x) is int", "location_invariant"),
                                 (5, 1, "<module>", "False", "location_invariant")))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    outcome = result["outcomes"][0]
    assert outcome["oracle_stop"] == "trace_changed"
    assert "model_error" in outcome["flags"]
    assert result["visits"] == [0, 0] and result["observed_visits"] == [1, 0]
    assert result["failures"]["run"]["kind"] == "not_proved"


def test_valid_bound_property_exception_hard_blocks_compare_and_cli_without_falsifying_invariants(tmp_path, monkeypatch):
    item = row(tmp_path)
    ref, store = packet(tmp_path, [item], "ref.json"), packet(tmp_path, [item], "store.json")
    output = tmp_path / "result.json"
    monkeypatch.setattr(MODULE, "_independent", lambda *args: {"confirmed": False, "reason": "Unchecked: unsupported", "open_obligations": 0})
    native = {"visits": [1], "failures": {"run": {"kind": "not_proved", "reason": "raised"}},
              "outcomes": [{"status": "raised", "eligible_for_refutation": True,
                            "bound_property_violation": "ZeroDivisionError", "exc": "E",
                            "mro": ["ZeroDivisionError", "ArithmeticError", "Exception", "BaseException", "object"],
                            "oracle_stop": None}]}
    monkeypatch.setattr(MODULE, "native_check", lambda *args, **kwargs: native)
    assert MODULE.main([str(ref), str(store), "--output", str(output)]) == 2
    result = json.loads(output.read_text())
    assert result["oracle_false"] == 0 and result["categories"]["REFUTED"] == 0
    assert result["property_violations"][0]["class"] == "ZeroDivisionError" and not result["m6_ready"]
    native["outcomes"][0]["eligible_for_refutation"] = False
    assert MODULE.main([str(ref), str(store), "--output", str(output)]) == 0
    assert not json.loads(output.read_text())["property_violations"]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_trace_custom_class_locals_cannot_invoke_mapping_callbacks(tmp_path):
    source = ('class Local(dict):\n    def __getitem__(self, key):\n'
              '        if key == "x":\n            raise ZeroDivisionError\n'
              '        return dict.__getitem__(self, key)\n'
              'class Meta(type):\n    @classmethod\n    def __prepare__(cls, name, bases):\n'
              '        return Local()\n'
              'class Box(metaclass=Meta):\n    x = 1\n    pass\n')
    item = row(tmp_path, source, ((12, 5, "Box", "type(x) is int", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["outcomes"][0]["status"] == "clean" and result["observed_visits"] == [1]
    failure = result["failures"]["0"]
    assert "custom scope mapping" in failure.get("observed_failure", failure)["reason"]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_trace_same_filename_spoofed_code_cannot_supply_visits(tmp_path):
    source = ('if False:\n    pass\n'
              'exec(compile("\\npass\\n", __file__, "exec"))\n')
    item = row(tmp_path, source, ((2, 5, "<module>", "False", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["outcomes"][0]["oracle_stop"] == "foreign_code"
    assert "model_error" in result["outcomes"][0]["flags"]
    assert result["observed_visits"] == [0] and result["visits"] == [0]
    assert all(failure["kind"] != "false" for failure in result["failures"].values())


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
@pytest.mark.parametrize("name", ["str", "getattr"])
def test_trace_builtin_mutation_cannot_trigger_observer_callbacks(tmp_path, name):
    source = ('import builtins\ncalls = []\n'
              'def bad(*args, **kwargs):\n    calls.append(1)\n    raise ZeroDivisionError\n'
              'original = builtins.' + name + '\n'
              'builtins.' + name + ' = bad\nx = 1\npass\n'
              'builtins.' + name + ' = original\nassert not calls\n')
    item = row(tmp_path, source, ((9, 1, "<module>", "type(x) is int", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["outcomes"][0]["status"] == "clean"  # original source asserts callback list is empty
    assert result["observed_visits"] == [1]
    failure = result["failures"]["0"]
    assert failure["kind"] == "not_proved"
    assert "builtin identity changed" in failure.get("observed_failure", failure)["reason"]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
@pytest.mark.parametrize("scope", ["globals", "class_locals", "function_builtins"])
def test_trace_hostile_nonstring_scope_keys_never_trigger_observer_equality(tmp_path, scope):
    collision = "len" if scope == "function_builtins" else "x"
    prefix = ('calls = []\nclass Key:\n    def __hash__(self):\n'
              '        return hash(' + repr(collision) + ')\n'
              '    def __eq__(self, other):\n        calls.append(other)\n'
              '        raise ZeroDivisionError\nkey = Key()\n')
    if scope == "globals":
        source = prefix + 'globals()[key] = 1\npass\ndel globals()[key]\nassert not calls\n'
        function, value = "<module>", "type(x) is int"
    elif scope == "class_locals":
        source = prefix + 'class Box:\n    locals()[key] = 1\n    pass\n    del locals()[key]\nassert not calls\n'
        function, value = "Box", "type(x) is int"
    else:
        source = prefix + ('original = __builtins__\nreplacement = original.__dict__.copy()\n'
                           'del replacement["len"]\nreplacement[key] = 1\n__builtins__ = replacement\n'
                           'def main():\n    pass\nmain()\n__builtins__ = original\n'
                           'del replacement[key]\nassert not calls\n')
        function, value = "main", "len('a') == 1"
    lines = source.splitlines()
    line = next(i + 1 for i, text in enumerate(lines) if text.strip() == "pass")
    column = len(lines[line - 1]) - len(lines[line - 1].lstrip()) + 1
    item = row(tmp_path, source, ((line, column, function, value, "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    assert result["outcomes"][0]["status"] == "clean"  # original source removes the identical key, then asserts no callbacks
    assert result["observed_visits"] == [1]
    failure = result["failures"]["0"]
    assert failure["kind"] == "not_proved"
    assert "non-string scope key" in failure.get("observed_failure", failure)["reason"]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_native_draw_attempts_have_fresh_parent_owned_scratch(tmp_path, monkeypatch):
    source = "import _sv_verifier\nx = _sv_verifier.nondet_int()\npass\n"
    item = row(tmp_path, source, ((3, 1, "<module>", "type(x) is int", "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    native_run, scratches = subprocess.run, []

    def observed_run(*args, **kwargs):
        request = json.loads(kwargs["input"])
        scratch = Path(kwargs["cwd"])
        assert request["scratch"] == str(scratch) and scratch.is_dir()
        assert not list(scratch.iterdir()) and scratch != Path(item["source"]).parent
        scratches.append(scratch)
        return native_run(*args, **kwargs)

    monkeypatch.setattr(MODULE.subprocess, "run", observed_run)
    result = MODULE.native_check(loaded)
    assert result["run_count"] == 4 and result["draw_count"] == 4
    assert result["visits"] == [4] and not result["failures"]
    assert all(outcome["status"] == "clean" and outcome["flags"] == [] for outcome in result["outcomes"])
    assert len(set(scratches)) == 4 and not any(path.exists() for path in scratches)


def test_native_timeout_cleans_only_parent_owned_scratch(tmp_path, monkeypatch):
    loaded = MODULE.load_rows(packet(tmp_path, [row(tmp_path)]))[0]
    scratches = []

    def timeout(*args, **kwargs):
        scratch = Path(kwargs["cwd"])
        assert scratch.is_dir() and json.loads(kwargs["input"])["scratch"] == str(scratch)
        (scratch / "owned-output").write_text("partial")
        scratches.append(scratch)
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(MODULE.subprocess, "run", timeout)
    result = MODULE.native_check(loaded, runs=1)
    assert result["outcomes"][0]["status"] == "budget_exhausted"
    assert scratches and not any(path.exists() for path in scratches)
    assert Path(loaded["source"]).is_file()


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_native_guarded_open_is_still_part_of_builtin_identity_check(tmp_path):
    source = "import builtins\nbuiltins.open = lambda *args, **kwargs: None\nx = 1\npass\n"
    item = row(tmp_path, source, ((4, 1, "<module>", "type(x) is int", "location_invariant"),))
    result = MODULE.native_check(MODULE.load_rows(packet(tmp_path, [item]))[0], runs=1)
    assert result["observed_visits"] == [1] and result["visits"] == [0]
    assert result["outcomes"][0]["oracle_stop"] == "builtin_changed"
    assert "model_error" in result["outcomes"][0]["flags"]
    failure = result["failures"]["0"]
    assert failure["kind"] == "not_proved"
    assert "builtin identity changed" in failure["observed_failure"]["reason"]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
@pytest.mark.parametrize("effect", ["filesystem", "spawn", "thread"])
def test_native_oracle_confines_target_effects(tmp_path, effect):
    victim = tmp_path / "victim"
    victim.write_text("intact")
    imports, action = {
        "filesystem": ("import os\n", f"os.unlink({str(victim)!r})"),
        "spawn": ("import subprocess, sys\n", "subprocess.run([sys.executable, '-c', "
                  + repr(f"import os; os.unlink({str(victim)!r})") + "], check=True)"),
        "thread": ("import os, threading\n", f"threading.Thread(target=lambda: os.unlink({str(victim)!r})).start()"),
    }[effect]
    source = imports + "blocked = False\ntry:\n    " + action + "\nexcept OSError:\n    blocked = True\nassert blocked\npass\n"
    item = row(tmp_path, source, ((8, 1, "<module>", "True", "location_invariant"),))
    result = MODULE.native_check(MODULE.load_rows(packet(tmp_path, [item]))[0], runs=1)
    outcome = result["outcomes"][0]
    assert victim.read_text() == "intact"
    assert outcome["status"] == "clean" and result["observed_visits"] == [1]
    assert {"file_effect", "file_unmodelled"} <= set(outcome["flags"])
    assert not outcome["eligible_for_refutation"] and result["visits"] == [0]
    if effect == "thread":
        assert "stack_unmodelled" in outcome["flags"]


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
def test_native_oracle_keeps_guards_for_exit_callbacks(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("intact")
    source = ("import atexit, os\n"
              "def later():\n"
              "    with open('callback-ran', 'w') as stream:\n"
              "        stream.write('yes')\n"
              "    try:\n"
              f"        os.unlink({str(victim)!r})\n"
              "    except OSError:\n"
              "        pass\n"
              "atexit.register(later)\nx = 1\npass\n")
    item = row(tmp_path, source, ((11, 1, "<module>", "type(x) is int", "location_invariant"),))
    with tempfile.TemporaryDirectory(dir=tmp_path) as scratch:
        scratch = str(Path(scratch).resolve())
        child = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/store_witness_diff.py"), "--native-child"],
            input=json.dumps({**item, "draws": [], "seconds": 1.0, "scratch": scratch}),
            capture_output=True, text=True, timeout=5, cwd=scratch, env=subprocess_environment())
        assert child.returncode == 0, child.stderr
        assert json.loads(child.stdout)["status"] == "clean"
        assert (Path(scratch) / "callback-ran").read_text() == "yes"
        assert victim.read_text() == "intact"
    assert not Path(scratch).exists()


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
@pytest.mark.parametrize("binding", ["missing", "mismatch", "nonempty"])
def test_native_oracle_refuses_unbound_scratch_before_target(tmp_path, binding):
    item = row(tmp_path, "open('target-ran', 'w').write('ran')\npass\n")
    with tempfile.TemporaryDirectory(dir=tmp_path) as scratch:
        scratch = str(Path(scratch).resolve())
        request = {**item, "draws": [], "seconds": 1.0}
        if binding != "missing":
            request["scratch"] = str(tmp_path) if binding == "mismatch" else scratch
        if binding == "nonempty":
            (Path(scratch) / "occupied").write_text("existing")
        child = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/store_witness_diff.py"), "--native-child"],
            input=json.dumps(request), capture_output=True, text=True, timeout=5, cwd=scratch,
            env=subprocess_environment())
        assert child.returncode == 0, child.stderr
        result = json.loads(child.stdout)
        assert result["status"] == "unsupported" and "empty bound scratch" in result["reason"]
        assert not (Path(scratch) / "target-ran").exists()


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="native oracle requires CPython 3.14")
@pytest.mark.parametrize("pythonpath", ["src", os.pathsep.join(["", "src", "missing", "src", ""])])
@pytest.mark.parametrize("predicate,refuted", [("x == 1", False), ("x == 2", True)])
def test_native_oracle_preserves_relative_pythonpath(tmp_path, monkeypatch, pythonpath, predicate, refuted):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    item = row(tmp_path, values=((2, 1, "<module>", predicate, "location_invariant"),))
    loaded = MODULE.load_rows(packet(tmp_path, [item]))[0]
    result = MODULE.native_check(loaded, runs=1)
    outcome = result["outcomes"][0]
    assert outcome["status"] == "clean" and outcome["flags"] == [], outcome
    assert outcome["eligible_for_refutation"] and not outcome["invalid_draw"]
    assert result["observed_visits"] == result["visits"] == [1]
    if refuted:
        assert result["failures"]["0"]["kind"] == "false"
    else:
        assert not result["failures"]
    assert os.environ["PYTHONPATH"] == pythonpath
