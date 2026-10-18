"""Data-only responses from the disposable Avengers worker.

The request is produced by the trusted parent and may use pickle inside the
child.  The response is produced after target code has run, so the parent
parses only this bounded JSON vocabulary and constructs known value objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tharkun.avengers.flatting import FlatObjectTemplate
from tharkun.avengers.model import ReplayOutcome

PROTOCOL = "tharkun-avengers-response/2"
# Version 1 carried no coverage. Both are accepted on read: the parent and the
# worker always ship together, but a store or a captured payload may predate
# the change, and refusing it would lose evidence that is still correct.
ACCEPTED_PROTOCOLS = frozenset({"tharkun-avengers-response/1", PROTOCOL})
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
# A traced replay reports at most this many lines and arcs. A target with a
# hot loop can execute a lot of distinct lines; the cap keeps one runaway
# module from filling the response budget that the template payload shares.
MAX_COVERAGE_ITEMS = 100_000


def _hex(value: bytes) -> str:
    return value.hex()


def _read_hex(value: Any, name: str) -> bytes:
    if not isinstance(value, str) or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    if len(value) % 2:
        raise ValueError(f"{name} has an odd hexadecimal length")
    return bytes.fromhex(value)


def _optional_string(value: Any, name: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{name} must be a string or null")


def _non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _encode_replay(value: ReplayOutcome) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": value.kind,
        "phase": value.phase,
        "detail": value.detail,
        "exception_type": value.exception_type,
        "signal": value.signal,
        "return_types": list(value.return_types),
        "state_digest": value.state_digest,
    }
    # Absent rather than empty when nothing was measured, so a reader can tell
    # "not traced" from "traced and executed nothing".
    if value.lines:
        payload["lines"] = list(value.lines)
    if value.arcs:
        payload["arcs"] = [[start, end] for start, end in value.arcs]
    return payload


def _coverage_lines(value: Any, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if type(value) is not list or len(value) > MAX_COVERAGE_ITEMS:
        raise ValueError(f"{name} must be a bounded list")
    for item in value:
        if type(item) is not int or item < 0:
            raise ValueError(f"{name} must hold non-negative integers")
    return tuple(value)


def _coverage_arcs(value: Any, name: str) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if type(value) is not list or len(value) > MAX_COVERAGE_ITEMS:
        raise ValueError(f"{name} must be a bounded list")
    result: list[tuple[int, int]] = []
    for item in value:
        if type(item) is not list or len(item) != 2:
            raise ValueError(f"{name} must hold two-element pairs")
        start, end = item
        if type(start) is not int or type(end) is not int or start < 0 or end < 0:
            raise ValueError(f"{name} pairs must be non-negative integers")
        result.append((start, end))
    return tuple(result)


def _decode_replay(value: Any) -> ReplayOutcome:
    if type(value) is not dict:
        raise ValueError("replay response must be an object")
    kind = value.get("kind")
    signal = value.get("signal")
    return_types = value.get("return_types", [])
    if not isinstance(kind, str):
        raise ValueError("replay kind must be a string")
    if signal is not None and type(signal) is not int:
        raise ValueError("replay signal must be an integer or null")
    if type(return_types) is not list or any(not isinstance(item, str) for item in return_types):
        raise ValueError("replay return_types must be a list of strings")
    return ReplayOutcome(
        kind=kind,
        phase=_optional_string(value.get("phase"), "replay phase"),
        detail=_optional_string(value.get("detail"), "replay detail"),
        exception_type=_optional_string(value.get("exception_type"), "replay exception_type"),
        signal=signal,
        return_types=tuple(return_types),
        state_digest=_optional_string(value.get("state_digest"), "replay state_digest"),
        lines=_coverage_lines(value.get("lines"), "replay lines"),
        arcs=_coverage_arcs(value.get("arcs"), "replay arcs"),
    )


def _encode_template(value: FlatObjectTemplate) -> dict[str, Any]:
    return {
        "recipe_blob": _hex(value.recipe_blob),
        "recipe_sha": value.recipe_sha,
        "total_budget": value.total_budget,
        "n_leaves": value.n_leaves,
        "baseline_data": _hex(value.baseline_data),
        "exemplar_data": _hex(value.exemplar_data),
        "exemplar_json": value.exemplar_json,
        "leaf_paths": list(value.leaf_paths),
    }


def _decode_template(value: Any) -> FlatObjectTemplate:
    if type(value) is not dict:
        raise ValueError("flat template response must be an object")
    recipe_sha = value.get("recipe_sha")
    exemplar_json = value.get("exemplar_json")
    leaf_paths = value.get("leaf_paths")
    if (
        not isinstance(recipe_sha, str)
        or len(recipe_sha) != 64
        or any(ch not in "0123456789abcdef" for ch in recipe_sha)
    ):
        raise ValueError("flat template recipe_sha must be a SHA-256 hex digest")
    if not isinstance(exemplar_json, str):
        raise ValueError("flat template exemplar_json must be a string")
    if type(leaf_paths) is not list or any(not isinstance(path, str) for path in leaf_paths):
        raise ValueError("flat template leaf_paths must be a list of strings")
    return FlatObjectTemplate(
        recipe_blob=_read_hex(value.get("recipe_blob"), "recipe_blob"),
        recipe_sha=recipe_sha,
        total_budget=_non_negative_int(value.get("total_budget"), "total_budget"),
        n_leaves=_non_negative_int(value.get("n_leaves"), "n_leaves"),
        baseline_data=_read_hex(value.get("baseline_data"), "baseline_data"),
        exemplar_data=_read_hex(value.get("exemplar_data"), "exemplar_data"),
        exemplar_json=exemplar_json,
        leaf_paths=tuple(leaf_paths),
    )


@dataclass(frozen=True)
class GeneratedSource:
    """Python test source a backend produced inside the worker."""

    tool: str
    source: str


@dataclass(frozen=True)
class ConstructorCandidates:
    """Constructor-call records a backend found inside the worker.

    They cross the boundary as documents, never as live calls, so the parent
    rebuilds them through the same closed literal vocabulary a stored recipe
    goes through.
    """

    records: tuple[dict[str, Any], ...]


MAX_GENERATED_SOURCE_BYTES = 4 * 1024 * 1024
MAX_CANDIDATES = 256


def _encode_source(value: GeneratedSource) -> dict[str, Any]:
    return {"tool": value.tool, "source": value.source}


def _decode_source(value: Any) -> GeneratedSource:
    if type(value) is not dict:
        raise ValueError("generated source response must be an object")
    tool = value.get("tool")
    source = value.get("source")
    if not isinstance(tool, str) or not tool:
        raise ValueError("generated source names its tool")
    if not isinstance(source, str):
        raise ValueError("generated source must be a string")
    if len(source.encode("utf-8")) > MAX_GENERATED_SOURCE_BYTES:
        raise ValueError("generated source exceeds the size limit")
    return GeneratedSource(tool, source)


def _encode_candidates(value: ConstructorCandidates) -> list[Any]:
    return list(value.records)


def _decode_candidates(value: Any) -> ConstructorCandidates:
    if type(value) is not list or len(value) > MAX_CANDIDATES:
        raise ValueError("candidate response must be a bounded list")
    for item in value:
        if type(item) is not dict:
            raise ValueError("each candidate is a constructor-call record")
    return ConstructorCandidates(tuple(value))


def encode_response(
    value: ReplayOutcome
    | FlatObjectTemplate
    | GeneratedSource
    | ConstructorCandidates
    | tuple[ReplayOutcome, ...],
) -> bytes:
    """Encode one known worker response without serialising live objects."""

    payload: Any
    if type(value) is ReplayOutcome:
        kind = "replay"
        payload = _encode_replay(value)
    elif type(value) is FlatObjectTemplate:
        kind = "flat-template"
        payload = _encode_template(value)
    elif type(value) is GeneratedSource:
        kind = "generated-source"
        payload = _encode_source(value)
    elif type(value) is ConstructorCandidates:
        kind = "candidates"
        payload = _encode_candidates(value)
    elif type(value) is tuple and all(type(item) is ReplayOutcome for item in value):
        kind = "batch"
        payload = [_encode_replay(item) for item in value]
    else:
        raise TypeError(f"unsupported worker response {type(value).__name__}")
    document = {"protocol": PROTOCOL, "type": kind, "value": payload}
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_response(
    data: bytes,
) -> ReplayOutcome | FlatObjectTemplate | tuple[ReplayOutcome, ...]:
    """Parse a bounded worker response using a closed, data-only vocabulary."""

    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("worker response exceeds the size limit")
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("worker response is not valid JSON") from error
    if type(document) is not dict or document.get("protocol") not in ACCEPTED_PROTOCOLS:
        raise ValueError("unknown worker response protocol")
    kind = document.get("type")
    if kind == "replay":
        return _decode_replay(document.get("value"))
    if kind == "flat-template":
        return _decode_template(document.get("value"))
    if kind == "batch":
        return _decode_batch(document.get("value"))
    if kind == "generated-source":
        return _decode_source(document.get("value"))
    if kind == "candidates":
        return _decode_candidates(document.get("value"))
    raise ValueError("unknown worker response type")


def _decode_batch(value: Any) -> tuple[ReplayOutcome, ...]:
    if type(value) is not list:
        raise ValueError("batch response must be a list")
    return tuple(_decode_replay(item) for item in value)
