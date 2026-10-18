"""Source-bound witness differential; native fallback is bounded evidence, not proof.

Candidates are JSON ``rows`` with task, source/source_sha256,
property_file/property_sha256 and witness {path, sha256}. References are either
that format, REFERENCE56.json, or retained MIN PAIRS.json + --min-result.
Every candidate invariant is checked, including extra tasks and locations.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tokenize
from typing import Any
from collections import Counter


class _NativeStop(BaseException):
    pass


def _loads_json(text: str | bytes) -> Any:
    def mapping(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=mapping)


def load_json(path: Path) -> Any:
    return _loads_json(path.read_bytes())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bound_file(path, digest) -> bytes:
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError("expected absolute artifact path")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("missing/invalid SHA256")
    raw = Path(path).read_bytes()
    if _sha(raw) != digest:
        raise ValueError(f"hash mismatch: {path}")
    return raw


def pair_key(row) -> tuple[str, str, str]:
    return row["task"], row["source_sha256"], row["property_sha256"]


def invariant_key(inv) -> tuple:
    return inv.type, inv.file_name, inv.line, inv.column, inv.function


def _unique(rows, label):
    result = {}
    for row in rows:
        key = pair_key(row)
        if key in result:
            raise ValueError(f"duplicate {label} pair: {key}")
        result[key] = row
    return result


def _read_row(row):
    """Authenticate raw bytes, witness metadata and original statement locations."""
    from tharkun.solve.obligations import Invariant, source_points
    from tharkun.sv.witness import loads

    if not isinstance(row.get("task"), str) or not row["task"]:
        raise ValueError("missing task identity")
    source_raw = _bound_file(row["source"], row["source_sha256"])
    encoding, _ = tokenize.detect_encoding(io.BytesIO(source_raw).readline)
    source = source_raw.decode(encoding)
    spec = _bound_file(row["property_file"], row["property_sha256"]).decode("utf-8")
    artifact = row["witness"]
    raw = _bound_file(artifact["path"], artifact["sha256"])
    if "exact_yaml_utf8" in artifact and artifact["exact_yaml_utf8"].encode("utf-8") != raw:
        raise ValueError("raw reference witness mismatch")
    documents = loads(raw.decode("utf-8"))
    invariants = []
    for document in documents:
        meta = document["metadata"]
        if set(meta) != {"format_version", "uuid", "creation_time", "producer", "task"}:
            raise ValueError("missing/unknown witness metadata")
        expected = {"input_files": [row["source"]],
                    "input_file_hashes": {row["source"]: row["source_sha256"]},
                    "specification": spec, "language": "Python"}
        if meta["format_version"] == "2.0":
            expected["data_model"] = "CPython-3.14-binary64"
        if meta["task"] != expected:
            raise ValueError("witness source/property/semantics mismatch")
        invariants.extend(Invariant.from_entry(item) for item in document["content"])
    if not invariants:
        raise ValueError("empty witness")
    tree = ast.parse(source, row["source"])
    points = source_points(tree, source, row["source"])
    statements = {}
    for node, (_, location) in points.items():
        if isinstance(node, ast.stmt):
            statements.setdefault(location, []).append(node)
    seen = set()
    for inv in invariants:
        full = invariant_key(inv), inv.value
        if full in seen:
            raise ValueError(f"duplicate invariant: {full}")
        seen.add(full)
        location = (inv.file_name, inv.line, inv.column, inv.function)
        nodes = statements.get(location, [])
        # The empty-module location is the frozen checker's own convention.
        if not tree.body and location == (row["source"], 1, 1, "<module>"):
            nodes = [None]
        if len(nodes) != 1:
            raise ValueError("invariant is not at one original statement")
        if inv.type == "loop_invariant" and not isinstance(nodes[0], ast.While):
            raise ValueError("loop invariant is not at an original while head")
    return {**row, "source_text": source, "specification": spec,
            "invariants": invariants, "tree": tree, "points": points,
            "source_bytes": source_raw, "witness_text": raw.decode("utf-8")}


def load_rows(path: Path, *, min_result: Path | None = None) -> list[dict]:
    packet = load_json(path)
    if min_result is not None:
        rows = _load_min(packet, load_json(min_result))
        if len(rows) != 731:
            raise ValueError("retained MIN reference must have 731 checked witnesses")
        return rows
    if isinstance(packet, dict) and packet.get("format") == "reference56-static-inventory-v1":
        rows = []
        if not isinstance(packet["rows"], dict):
            raise ValueError("REFERENCE56 rows must be a mapping")
        for key, row in packet["rows"].items():
            expected = [row["task"], row["source"]["sha256"], row["property"]["sha256"]]
            if json.loads(key) != expected or row["key_tuple"] != expected:
                raise ValueError("REFERENCE56 key binding mismatch")
            confirmation = row["reference_confirmation"]
            if not (confirmation["actual_verify_proof_checked"] is True
                    and confirmation["actual_export_external"]["confirmed"] is True):
                raise ValueError("unchecked REFERENCE56 row")
            rows.append({"task": row["task"], "source": row["source"]["path"],
                         "source_sha256": expected[1], "property_file": row["property"]["file"],
                         "property_sha256": expected[2], "witness": row["witness"]})
        if len(rows) != 56:
            raise ValueError("REFERENCE56 must have 56 checked witnesses")
    else:
        rows = packet["rows"] if isinstance(packet, dict) else packet
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty/malformed witness packet")
    bound = [_read_row(row) for row in rows]
    _unique(bound, "packet")
    return bound


def _load_min(pairs, result):
    if not isinstance(pairs, list) or not pairs or not isinstance(result.get("results"), list):
        raise ValueError("malformed MIN pairs/results")
    by_key = {}
    for pair in pairs:
        if pair["key"] in by_key:
            raise ValueError("duplicate MIN pair key")
        by_key[pair["key"]] = pair
    seen, rows = set(), []
    for item in result["results"]:
        key = item["key"]
        if key in seen or key not in by_key:
            raise ValueError("duplicate/unknown MIN result key")
        seen.add(key)
        pair = by_key[key]
        if item["index"] != pair["index"]:
            raise ValueError("MIN result index mismatch")
        if item["proof_checked_state"] is not True:
            continue
        if (item["status"] != "RETURNED" or item["verdict"] != "true"
                or item["timed_out"] is not False or item["rc"] != 0):
            raise ValueError("invalid checked MIN result")
        public_path = item["public_verdict_path"]
        public_raw = _bound_file(public_path, item["artifacts"]["PUBLIC-VERDICT.json"]["sha256"])
        public = _loads_json(public_raw)
        if (public["pair"] != pair or public["public_verdict"]["verdict"] != "true"
                or public["public_verdict"]["detail"]["proof_checked"] is not True
                or public["serialization_complete"] is not True):
            raise ValueError("MIN public verdict binding mismatch")
        artifacts = [(name, info) for name, info in item["artifacts"].items()
                     if name.endswith("/witness.yml")]
        if len(artifacts) != 1:
            raise ValueError("MIN checked row must bind one raw witness")
        name, info = artifacts[0]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("MIN artifact escapes pair directory")
        rows.append({"task": pair["task"], "source": pair["source"],
                     "source_sha256": pair["source_sha256"],
                     "property_file": pair["property_file"], "property_sha256": pair["property_sha256"],
                     "witness": {"path": str(Path(public_path).parent / relative), "sha256": info["sha256"]}})
    if seen != set(by_key) or not rows:
        raise ValueError("missing MIN results or empty checked reference")
    bound = [_read_row(row) for row in rows]
    _unique(bound, "MIN reference")
    return bound


def _builtin_names(tree):
    """Conservative reuse of census bindings; reflection/wildcards stay unsupported."""
    from tharkun.solve.census import _bound
    from tharkun.solve.predicates import BUILTINS
    bound = {name for node in ast.walk(tree) for name in _bound(node)}
    bound.update(node.rest for node in ast.walk(tree)
                 if isinstance(node, ast.MatchMapping) and node.rest)
    bound.update(node.name for node in ast.walk(tree)
                 if isinstance(node, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple)))
    reflective = {"eval", "exec", "globals", "locals", "vars", "setattr", "getattr", "compile", "__import__"}
    if ("*" in bound or "__builtins__" in bound
            or any(isinstance(node, ast.Name) and node.id in reflective | {"__builtins__"}
                   for node in ast.walk(tree))):
        return frozenset()
    # Imported code can mutate builtins. Only the existing draw boot is allowed.
    if any(isinstance(node, (ast.Import, ast.ImportFrom))
           and (not isinstance(node, ast.ImportFrom) or node.module not in {"_sv_verifier", "__future__"})
           and (not isinstance(node, ast.Import) or any(a.name != "_sv_verifier" for a in node.names))
           for node in ast.walk(tree)):
        return frozenset()
    return BUILTINS - bound


def _term(inv, builtin_names):
    from tharkun.solve import terms
    node = ast.parse(inv.value, mode="eval").body
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} - set(builtin_names)
    term = terms.from_ast(node, {name: "object" for name in names}, builtin_names=builtin_names)
    if isinstance(term, terms.Refused) or term.sort != terms.BOOL:
        return terms.Refused("unsupported Boolean predicate")
    return term


def implication(candidates, reference, tree, *, builtin_names=(), resource_limit=10000):
    """Distinct same-site claims are a conjunction. Only 'yes' is a proof."""
    from tharkun.solve import terms
    try:
        from tharkun.solve.terms_z3 import implies
        builtins = frozenset(builtin_names)
        goal = _term(reference, builtins)
        if isinstance(goal, terms.Refused):
            return {"status": "Refused", "reason": goal.reason}
        # True needs no assumptions; candidate truth and full property proof are
        # still checked independently below, so this cannot bless a false claim.
        if goal == terms.literal(True):
            facts = []
        else:
            facts = [_term(inv, builtins) for inv in candidates]
            refusal = next((fact for fact in facts if isinstance(fact, terms.Refused)), None)
            if refusal is not None:
                return {"status": "Refused", "reason": refusal.reason}
        result = implies(facts, goal, resource_limit=resource_limit)
        if isinstance(result, terms.Refused):
            return {"status": "Refused", "reason": result.reason}
        return {"status": result if result in {"yes", "no", "unknown"} else "unknown", "reason": "typed implication"}
    except Exception as exc:
        return {"status": "unknown", "reason": f"{type(exc).__name__}: {exc}"}


def _independent(row, seconds):
    from tharkun.solve.census import obligation_census
    from tharkun.sv.spec import parse_property
    from tharkun.sv.validate import validate
    _, classes = parse_property(row["specification"])
    census = obligation_census(row["source_text"], classes, row["source"])
    result = validate(row["source_text"], row["source"], row["specification"],
                      row["witness_text"], seconds=seconds, census=census, source_bytes=row["source_bytes"])
    return {"confirmed": result.confirmed, "reason": result.reason,
            "open_obligations": len(result.open_obligations)}


def native_check(row, *, seconds=1.0, runs=4, line_budget=400000):
    """Each draw vector runs in a fresh bounded CPython child, never this process."""
    if (not math.isfinite(seconds) or seconds <= 0 or not 1 <= runs <= 16
            or type(line_budget) is not int or not 1 <= line_budget <= 400000):
        raise ValueError("invalid native budget")
    from tharkun.solve.environment import subprocess_environment
    from tharkun.sv.engine import NEVER_A_WITNESS_FLAGS
    vectors, outcomes = [[]], []
    entries = [inv.entry() for inv in row["invariants"]]
    request = {key: row[key] for key in ("task", "source", "source_sha256", "property_file", "property_sha256", "witness")}
    visits = [0] * len(entries)
    observed_visits = [0] * len(entries)
    failures = {}
    for index in range(runs):
        vector = vectors[index] if index < len(vectors) else []
        request.update({"draws": vector, "seconds": seconds, "line_budget": line_budget})
        try:
            with tempfile.TemporaryDirectory(prefix="store-witness-native-") as temporary:
                request["scratch"] = os.path.realpath(temporary)
                child = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--native-child"],
                                       input=json.dumps(request), text=True, capture_output=True, cwd=request["scratch"],
                                       timeout=seconds + 2, env={**subprocess_environment(), "PYTHONOPTIMIZE": "0",
                                       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
            report = json.loads(child.stdout) if child.returncode == 0 else {
                "status": "child_failed", "reason": child.stderr[-1000:] or f"native child exited with return code {child.returncode}",
                "returncode": child.returncode, "signal": -child.returncode if child.returncode < 0 else None}
        except subprocess.TimeoutExpired:
            report = {"status": "budget_exhausted", "reason": "native wall budget exhausted"}
        except ValueError as exc:
            report = {"status": "timeout_or_transport_failure", "reason": str(exc)}
        flags = report.get("flags", [])
        forbidden = sorted(set(flags) & set(NEVER_A_WITNESS_FLAGS))
        eligible = (report.get("invalid_draw") is False and not forbidden
                    and (report.get("status") in {"clean", "raised", "exit"}
                         or report.get("status") == "budget_exhausted" and report.get("oracle_stop") in {"cpu_budget", "line_budget"}))
        invalid_reason = ("invalid native draw" if report.get("invalid_draw") else
                          "native model refused: " + ", ".join(forbidden) if forbidden else
                          report.get("reason", "native run eligibility unavailable"))
        report["eligible_for_refutation"] = eligible
        outcomes.append(report)
        counts = report.get("visits", [])
        if len(counts) == len(visits):
            observed_visits = [a + b for a, b in zip(observed_visits, counts)]
            if eligible:
                visits = [a + b for a, b in zip(visits, counts)]
        for label, failure in report.get("failures", {}).items():
            failures[label] = failure if eligible else {"kind": "not_proved", "reason": invalid_reason,
                                                         "observed_failure": failure}
        legal_stop = eligible and report.get("source_stop") == "outside_property"
        if not eligible or (report.get("status") != "clean" and not legal_stop) or flags:
            reason = invalid_reason if not eligible else report.get("reason", report.get("status"))
            failures.setdefault("run", {"kind": "not_proved", "reason": reason})
        if index == 0:
            for value in (-1, 1, 2):
                changed = []
                for draw in report.get("draws", []):
                    kind = draw["kind"]
                    replacement = value if kind == "int" else (value > 0 if kind == "bool" else float(value).hex() if kind == "float" else draw["value"])
                    changed.append({"kind": kind, "value": replacement})
                vectors.append(changed)
        if any(item.get("kind") == "false" for item in failures.values()) or not report.get("draws"):
            break
    return {"bounded_native": True, "universal_proof": False, "visits": visits, "observed_visits": observed_visits,
            "failures": failures, "outcomes": outcomes, "run_count": len(outcomes),
            "draw_count": sum(len(outcome.get("draws", [])) for outcome in outcomes)}


def _trace_targets(row):
    """Line events identify pre-states only when one original statement owns a line."""
    per_line = Counter(node.lineno for node in ast.walk(row["tree"]) if isinstance(node, ast.stmt))
    targets = {}
    loops = {location: node for node, (_, location) in row["points"].items() if isinstance(node, ast.While)}
    for index, inv in enumerate(row["invariants"]):
        if per_line[inv.line] > 1:
            raise ValueError("native trace at a multi-statement line is unsupported")
        loop = loops.get((inv.file_name, inv.line, inv.column, inv.function))
        if loop is not None and (loop.test.lineno != loop.lineno or loop.test.end_lineno != loop.lineno):
            raise ValueError("native trace at a multiline while test is unsupported")
        targets.setdefault((inv.function, inv.line), []).append(index)
    return targets


def _native_type(value, type_fn=type, classes=((int, "int"), (bool, "bool"), (float, "float"),
        (str, "str"), (bytes, "bytes"), (list, "list"), (tuple, "tuple"), (dict, "dict"),
        (set, "set"), (frozenset, "frozenset"))):
    actual = type_fn(value)
    for builtin_class, name in classes:
        if actual is builtin_class:
            return name
    return "object"


def _native_child(request):
    """Trace untouched source in one bounded child using the existing draw boot."""
    import builtins
    import resource
    import signal
    from types import CodeType
    from tharkun.solve.predicates import BUILTINS, check_predicate, parse_predicate
    from tharkun.sv import _child
    from tharkun.sv.engine import Outcome
    from tharkun.sv.spec import parse_property
    if sys.version_info[:2] != (3, 14) or sys.flags.optimize:
        raise ValueError("native oracle requires unoptimized CPython 3.14")
    scratch = request.get("scratch")
    if (type(scratch) is not str or not os.path.isabs(scratch)
            or scratch != os.path.realpath(scratch) or scratch != os.getcwd() or os.listdir(scratch)):
        raise ValueError("native oracle requires its parent's empty bound scratch directory")
    failures, oracle_stop = {}, None
    native_signal = signal.signal
    cpu_signal = signal.SIGXCPU
    native_gettrace = sys.gettrace
    trace_started = False

    def cpu_exhausted(signum, frame):
        nonlocal oracle_stop
        # Check before raising: CPython disables tracing on our own trace exception.
        if oracle_stop is None:
            if trace_started and native_gettrace() is not tracer:
                oracle_stop = "trace_changed"
                failures.setdefault("run", {"kind": "not_proved", "reason": "native trace changed by source"})
            else:
                oracle_stop = "cpu_budget"
        failures.setdefault("run", {"kind": "not_proved", "reason": "native cpu budget exhausted"})
        raise _NativeStop("cpu_budget")

    previous_cpu_handler = native_signal(cpu_signal, cpu_exhausted)
    # The soft limit reports exhaustion; the hard limit remains a kill-switch.
    resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(request["seconds"]) + 1, math.ceil(request["seconds"]) + 2))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024,) * 2)
    row = _read_row(request)
    targets = _trace_targets(row)
    line_budget = request.get("line_budget", 400000)
    if type(line_budget) is not int or not 1 <= line_budget <= 400000:
        raise ValueError("invalid native line budget")
    # Capture identities before the target can import or mutate anything.
    identities = {name: builtins.__dict__[name] for name in BUILTINS}
    builtin_snapshot = None
    native_compile, native_eval, native_type = builtins.compile, builtins.eval, builtins.type
    native_error, native_exception, native_id = ValueError, Exception, id
    native_len, native_str = len, str
    native_locals_type = native_type(sys._getframe().f_locals)
    missing_builtin = object()
    native_settrace = sys.settrace
    predicates = [parse_predicate(inv.value) for inv in row["invariants"]]
    used_names = [{n.id for n in ast.walk(predicate) if isinstance(n, ast.Name)} for predicate in predicates]
    expressions = [native_compile(inv.value, "<store-witness-predicate>", "eval", dont_inherit=True)
                   for inv in row["invariants"]]
    labels = [native_str(index) for index in range(len(row["invariants"]))]
    visits = [0] * len(row["invariants"])
    line_events = 0
    original_codes = {}

    def builtins_changed():
        current = builtins.__dict__
        if builtin_snapshot is None or native_len(current) != native_len(builtin_snapshot):
            return True
        for name, value in current.items():
            if native_type(name) is not native_str or builtin_snapshot.get(name, missing_builtin) is not value:
                return True
        return False

    def check(index, frame):
        visits[index] += 1
        inv, names = row["invariants"][index], used_names[index]
        local, glob = frame.f_locals, frame.f_globals
        lexical = identities["set"](frame.f_code.co_varnames + frame.f_code.co_cellvars + frame.f_code.co_freevars)
        try:
            if builtins_changed():
                raise native_error("native checker builtin identity changed")
            if (native_type(glob) is not identities["dict"] or native_type(frame.f_builtins) is not identities["dict"]
                    or (native_type(local) is not identities["dict"] and native_type(local) is not native_locals_type)):
                raise native_error("native custom scope mapping is unsupported")
            for mapping in (glob, local, frame.f_builtins):
                for name in mapping:
                    if native_type(name) is not native_str:
                        raise native_error("native non-string scope key is unsupported")
            # eval's mapping lookup must not invent a value for an unbound LOAD_FAST/DEREF.
            for name in names:
                if name in lexical and name not in local:
                    raise native_error("native lexical name unbound: " + name)
            builtin_names = identities["set"]()
            for name, original in identities.items():
                actual = local[name] if name in local else glob[name] if name in glob else frame.f_builtins.get(name)
                if actual is original:
                    builtin_names.add(name)
            env = {**glob, **local}
            types = {name: _native_type(env[name]) for name in names - builtin_names if name in env}
            check_predicate(inv.value, types, builtin_names=builtin_names)
            if native_eval(expressions[index], glob, local) is not True:
                failures[labels[index]] = {"kind": "false", "reason": "native assertion false"}
        except native_exception as exc:
            failures.setdefault(labels[index], {"kind": "not_proved", "reason": native_type(exc).__name__ + ": " + native_str(exc)})

    def tracer(frame, event, arg):
        nonlocal line_events, oracle_stop
        if frame.f_code.co_filename != row["source"]:
            return None
        if native_id(frame.f_code) not in original_codes:
            oracle_stop = "foreign_code"
            failures.setdefault("run", {"kind": "not_proved", "reason": "native same-filename code is not original compiled source"})
            return None
        if event == "line":
            line_events += 1
            if line_events > line_budget:
                oracle_stop = "line_budget"
                failures.setdefault("run", {"kind": "not_proved", "reason": "native line budget exhausted"})
                raise _NativeStop("line_budget")
            function = ".".join(part for part in frame.f_code.co_qualname.split(".") if part != "<locals>")
            for index in targets.get((function, frame.f_lineno), ()):
                check(index, frame)
        return tracer

    previous_compile = _child.__dict__.get("compile")
    previous_trace = native_gettrace()
    consumed = False

    def compile_target(source, filename, mode, flags=0, dont_inherit=False, optimize=-1, **kwargs):
        nonlocal consumed, builtin_snapshot
        if filename == row["source"] and mode == "exec" and not flags & ast.PyCF_ONLY_AST:
            if not isinstance(source, bytes) or _sha(source) != row["source_sha256"]:
                raise ValueError("target source changed before compilation")
            if consumed:
                raise ValueError("target compiled more than once")
            consumed = True
            # The child has installed its guards, but no target instruction has run.
            builtin_snapshot = builtins.__dict__.copy()
            code = native_compile(source, filename, mode, flags, dont_inherit, optimize, **kwargs)
            pending = [code]
            while pending:
                original = pending.pop()
                original_codes[native_id(original)] = original
                pending.extend(item for item in original.co_consts if native_type(item) is CodeType)
            return code
        return native_compile(source, filename, mode, flags, dont_inherit, optimize, **kwargs)

    _child.compile = compile_target
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            native_settrace(tracer)
            trace_started = True
            try:
                report = _child.run({"program": row["source"], "vector": request["draws"],
                                     "files": [row["source"]]})
            finally:
                if native_gettrace() is not tracer and oracle_stop is None:
                    oracle_stop = "trace_changed"
                    failures.setdefault("run", {"kind": "not_proved", "reason": "native trace changed by source"})
                native_settrace(previous_trace)
                if builtin_snapshot is not None:
                    changed = builtins_changed()
                    builtins.__dict__.clear()
                    builtins.__dict__.update(builtin_snapshot)
                    if changed:
                        oracle_stop = "builtin_changed"
                        failures.setdefault("run", {"kind": "not_proved", "reason": "native builtin identities changed"})
        if not consumed:
            raise ValueError("original target compilation was not intercepted")
        result = Outcome(report["status"], report.get("exc"), report.get("mro", []),
                         draws=report["draws"], flags=report["flags"], invalid_draw=report["invalid_draw"])
        if oracle_stop in {"trace_changed", "builtin_changed", "foreign_code"}:
            result.flags = sorted(set(result.flags) | {"model_error"})
        _, classes = parse_property(row["specification"])
        violation = result.violated(classes) if oracle_stop is None else None
        source_stop = ("bound_property" if set(classes).intersection(result.mro) else "outside_property") if result.status == "raised" and oracle_stop is None else None
        budget_stop = oracle_stop in {"cpu_budget", "line_budget"}
        return {"status": "budget_exhausted" if budget_stop else result.status,
                **({"reason": failures["run"]["reason"]} if budget_stop else {}),
                "exc": result.exc, "mro": result.mro, "draws": result.draws,
                "flags": result.flags, "invalid_draw": result.invalid_draw,
                "visits": visits, "failures": failures, "line_events": line_events, "line_budget": line_budget,
                "oracle_stop": oracle_stop, "source_stop": source_stop, "bound_property_violation": violation}
    finally:
        native_settrace(previous_trace)
        native_signal(cpu_signal, previous_cpu_handler)
        if previous_compile is None:
            _child.__dict__.pop("compile", None)
        else:
            _child.compile = previous_compile


def _record_builtins(row, inv, *, independently_confirmed=False):
    from tharkun.solve.predicates import BUILTINS
    location = (inv.file_name, inv.line, inv.column, inv.function)
    paths = {path for _, (path, point) in row["points"].items() if point == location}
    matches = [record for record in row["records"]
               if isinstance(record.get("site"), list) and len(record["site"]) == 3
               and record["site"][0] == row["source_sha256"] and record["site"][1] in paths]
    if matches:
        values = []
        for record in matches:
            names = record.get("builtins")
            if (not isinstance(names, list) or any(not isinstance(name, str) for name in names)
                    or names != sorted(set(names)) or not set(names) <= {"type", "int", "bool", "float", "str"}):
                return frozenset()
            values.append(set(names))
        return frozenset.intersection(*(frozenset(value) for value in values))
    if independently_confirmed:
        # The frozen checker authenticated only names actually used by supplied
        # predicates at this point; its verdict is not blanket builtin trust.
        return frozenset(n.id for candidate in row["invariants"] if invariant_key(candidate) == invariant_key(inv)
                         for n in ast.walk(ast.parse(candidate.value, mode="eval"))
                         if isinstance(n, ast.Name) and n.id in BUILTINS)
    return frozenset()


def _metrics(row):
    metrics, records = row.get("metrics"), row.get("records")
    fields = ("checked_sites", "safe_sites", "established_sites", "sites_established_and_rendered")
    if not isinstance(metrics, dict) or any(type(metrics.get(key)) is not int or metrics[key] < 0 for key in fields):
        raise ValueError("candidate needs captured checked/safe/established/rendered site counts")
    if metrics["sites_established_and_rendered"] > metrics["established_sites"] or metrics["established_sites"] > metrics["safe_sites"]:
        raise ValueError("inconsistent established/rendered/safe site counts")
    reasons = metrics.get("reasons")
    if not isinstance(reasons, dict) or any(not isinstance(key, str) or type(value) is not int or value < 0
                                           for key, value in reasons.items()):
        raise ValueError("candidate needs captured store_only reasons")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("candidate needs plain captured site records")
    counts = {}
    for key in ("binding", "contract", "facts", "builtins"):
        with_value = sum(bool(record.get(key)) for record in records)
        counts[key] = {"with": with_value, "without": len(records) - with_value}
    return metrics, counts


def _nontrivial(invariants, tree):
    """Only predicates with source variables count; closed true terms are separate."""
    from tharkun.solve import terms
    from tharkun.solve.predicates import BUILTINS
    from tharkun.solve.census import _bound
    bound = {name for node in ast.walk(tree) for name in _bound(node)}
    literal_true = closed_true = closed_predicates = informative = 0
    for inv in invariants:
        node = ast.parse(inv.value, mode="eval").body
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} - (BUILTINS - bound)
        if inv.value == "True":
            literal_true += 1
        elif names:
            informative += 1
        else:
            closed_predicates += 1
            term = _term(inv, _builtin_names(tree))
            if not isinstance(term, terms.Refused) and terms.eval_term(term, {}) is True:
                closed_true += 1
    return {"source_variable": informative, "literal_true": literal_true,
            "closed_predicates": closed_predicates, "closed_true": closed_true}


def compare(reference_rows, store_rows, *, checker_seconds=3.0, native_seconds=1.0, native_runs=4):
    if not reference_rows:
        raise ValueError("empty reference cannot establish inclusion")
    refs, stores = _unique(reference_rows, "reference"), _unique(store_rows, "candidate")
    per_task, unproved, refuted, nontrivial = [], 0, 0, 0
    property_violations = []
    categories = {key: 0 for key in ("confirmed", "reference_cannot_visit_or_express", "not_proved", "REFUTED")}
    site_counts = {key: 0 for key in ("safe_sites", "established_sites", "sites_established_and_rendered")}
    true_only = []
    histogram, availability, bare_true = {}, {key: {"with": 0, "without": 0} for key in ("binding", "contract", "facts", "builtins")}, []
    for key, store in sorted(stores.items()):
        metrics, counts = _metrics(store)
        for reason, count in metrics["reasons"].items():
            histogram[reason] = histogram.get(reason, 0) + count
        for field in availability:
            for kind in ("with", "without"):
                availability[field][kind] += counts[field][kind]
        if key in refs:
            for field in site_counts:
                site_counts[field] += metrics[field]
        if all(inv.value == "True" for inv in store["invariants"]):
            if metrics["sites_established_and_rendered"]:
                bare_true.append(key)
            else:
                true_only.append({"key": key, "reasons": metrics["reasons"]})
        independent = _independent(store, checker_seconds)
        native = None if independent["confirmed"] else native_check(store, seconds=native_seconds, runs=native_runs)
        if native is not None:
            for run_index, outcome in enumerate(native["outcomes"]):
                violation = outcome.get("bound_property_violation")
                if (violation and outcome.get("eligible_for_refutation") is True
                        and outcome.get("status") == "raised" and outcome.get("oracle_stop") is None):
                    property_violations.append({"key": key, "run_index": run_index, "class": violation,
                                                "exc": outcome.get("exc"), "mro": outcome.get("mro"),
                                                "reason": "P0: valid native bound-property exception in correctness-witness candidate"})
        classified = []
        for index, inv in enumerate(store["invariants"]):
            failure = {} if native is None else native["failures"].get(str(index), {})
            if failure.get("kind") == "false":
                category = "REFUTED"
                refuted += 1
            elif independent["confirmed"]:
                category = "confirmed"
            elif (native["visits"][index] and not failure and not native["failures"].get("run")):
                category = "reference_cannot_visit_or_express"
            else:
                category = "not_proved"
            unproved += category == "not_proved"
            categories[category] += 1
            classified.append({"index": index, "site": invariant_key(inv), "value": inv.value,
                               "category": category, "bounded_native": category == "reference_cannot_visit_or_express",
                               "reason": failure.get("reason", independent["reason"])})
        by_site = {}
        for inv in store["invariants"]:
            by_site.setdefault(invariant_key(inv), []).append(inv)
        reference = refs.get(key)
        implications, strengths = [], []
        if reference:
            for inv in reference["invariants"]:
                candidates = by_site.get(invariant_key(inv), [])
                trusted = _record_builtins(store, inv, independently_confirmed=independent["confirmed"])
                result = implication(candidates, inv, store["tree"], builtin_names=trusted) if candidates else {
                    "status": "unknown", "reason": "missing reference location"}
                implications.append({"site": invariant_key(inv), **result})
            reference_groups = {}
            for inv in reference["invariants"]:
                reference_groups.setdefault(invariant_key(inv), []).append(inv)
            for site, goals in reference_groups.items():
                candidates = by_site.get(site, [])
                forward = [item["status"] for item in implications if tuple(item["site"]) == site]
                trusted = _record_builtins(store, goals[0], independently_confirmed=independent["confirmed"])
                reverse = [implication(goals, inv, store["tree"], builtin_names=trusted)["status"] for inv in candidates]
                if candidates and all(status == "yes" for status in forward + reverse):
                    relation = "equivalent"
                elif candidates and all(status == "yes" for status in forward) and "no" in reverse:
                    relation = "stronger"
                else:
                    relation = "incomparable"
                strengths.append({"site": site, "relation": relation, "forward": forward, "reverse": reverse})
        rate = _nontrivial(store["invariants"], store["tree"])
        nontrivial += key in refs and rate["source_variable"] > 0
        per_task.append({"key": key, "status": "compared" if reference else "extra_candidate",
                         "independent": independent, "invariants": classified,
                         "implications": implications, "strength": strengths, "native": native, "predicate_counts": rate,
                         "checked_sites": metrics["checked_sites"], "record_counts": counts})
    missing = sorted(set(refs) - set(stores))
    return {"format": "store-witness-differential-v1", "milestone": 1,
            "reference_tasks": len(refs), "store_tasks": len(stores), "task_inclusion": not missing,
            "missing_store": missing, "bare_true_with_check_sites": bare_true,
            "oracle_false": refuted, "unproved": unproved, "categories": categories,
            "property_violations": property_violations,
            "m6_ready": not missing and not bare_true and not refuted and not unproved and not property_violations
                        and all(row["independent"]["confirmed"] and all(i["status"] == "yes" for i in row["implications"]) for row in per_task),
            "nontrivial_tasks": nontrivial, "nontrivial_rate": nontrivial / len(refs), "nontrivial_denominator": len(refs),
            "sites": site_counts, "true_only": true_only,
            "nontrivial_rule": "at least one exported predicate mentions a source variable; literal and closed true predicates excluded",
            "reference_nontrivial_tasks": sum(_nontrivial(row["invariants"], row["tree"])["source_variable"] > 0 for row in refs.values()),
            "store_only_reasons": histogram, "record_counts": availability,
            "native_evidence": "bounded fresh executions; never counted as confirmed or a universal proof",
            "per_task": per_task}


def main(argv=None) -> int:
    if argv is None and sys.argv[1:] == ["--native-child"]:
        try:
            result = _native_child(json.load(sys.stdin))
        except _NativeStop:
            result = {"status": "budget_exhausted", "reason": "native cpu budget exhausted", "oracle_stop": "cpu_budget"}
        except Exception as exc:
            result = {"status": "unsupported", "reason": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, sort_keys=True))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("--min-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checker-seconds", type=float, default=3.0)
    parser.add_argument("--native-seconds", type=float, default=1.0)
    parser.add_argument("--native-runs", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.checker_seconds) or args.checker_seconds <= 0:
            raise ValueError("invalid checker budget")
        result = compare(load_rows(args.reference, min_result=args.min_result), load_rows(args.store),
                         checker_seconds=args.checker_seconds, native_seconds=args.native_seconds,
                         native_runs=args.native_runs)
        inputs = [args.reference, args.store] + ([args.min_result] if args.min_result else [])
        result["inputs"] = {str(path): _sha(path.read_bytes()) for path in inputs}
    except Exception as exc:
        result = {"format": "store-witness-differential-v1", "error": f"{type(exc).__name__}: {exc}",
                  "task_inclusion": False, "oracle_false": 0, "unproved": 1}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result.get(key) for key in ("error", "nontrivial_tasks", "nontrivial_denominator", "sites", "categories", "oracle_false", "property_violations", "true_only", "store_only_reasons")}, sort_keys=True))
    return 0 if ("error" not in result and result["task_inclusion"] and result["oracle_false"] == 0
                 and not result["bare_true_with_check_sites"] and not result["property_violations"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
