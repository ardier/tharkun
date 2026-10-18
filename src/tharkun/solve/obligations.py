"""Obligations and facts share the Python invariant-witness entry shape ([internal ref]).

A serialized entry is a candidate claim, never a certificate of its own truth.
The source-bound ID and diagnostic gap stay in memory; no private proof syntax
is hidden in ``value``. A False placeholder is an explicit unimplemented safety
rule, not a demand that the producer prove a reachable operation unreachable.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from tharkun.solve.predicates import parse_predicate


@dataclass(frozen=True)
class Invariant:
    file_name: str
    line: int
    column: int
    function: str
    value: str
    type: str = "location_invariant"

    def __post_init__(self):
        if self.type not in {"location_invariant", "loop_invariant"}:
            raise ValueError("unknown invariant type")
        if (not isinstance(self.file_name, str) or not self.file_name
                or not isinstance(self.function, str) or not self.function
                or type(self.line) is not int or self.line < 1
                or type(self.column) is not int or self.column < 1):
            raise ValueError("invalid source location")
        parse_predicate(self.value)

    def entry(self) -> dict:
        return {"invariant": {"type": self.type, "location": {
            "file_name": self.file_name, "line": self.line, "column": self.column,
            "function": self.function}, "value": self.value, "format": "python_expression"}}

    @classmethod
    def from_entry(cls, entry: dict) -> Invariant:
        if not isinstance(entry, dict) or set(entry) != {"invariant"}:
            raise ValueError("expected invariant wrapper")
        obj = entry["invariant"]
        if not isinstance(obj, dict) or set(obj) != {"type", "location", "value", "format"} or obj["format"] != "python_expression":
            raise ValueError("invalid invariant fields/format")
        loc = obj["location"]
        if not isinstance(loc, dict) or set(loc) != {"file_name", "line", "column", "function"}:
            raise ValueError("invalid location fields")
        return cls(**loc, value=obj["value"], type=obj["type"])


@dataclass(frozen=True)
class Obligation:
    source_sha256: str
    ast_path: str
    exception: str
    invariant: Invariant
    gap: str = ""

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.source_sha256, self.ast_path, self.exception


@dataclass(frozen=True)
class ObligationCensus:
    obligations: tuple[Obligation, ...]
    coverage_gaps: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Coverage closed, independently of whether any obligation is discharged."""
        return not self.coverage_gaps


def witness(source: str, filename: str, specification: str, invariants, *,
           source_bytes: bytes | None = None) -> list[dict]:
    """Make a format-2.3 Python invariant_set, for facts OR still-open obligations.

    ``source_bytes``, when given, is what ``input_file_hashes`` binds instead
    of a UTF-8 re-encoding of ``source`` ([internal ref]): those differ whenever the
    file's own PEP 263 encoding isn't UTF-8, and a witness must identify the
    file as it is on disk, not as tharkun happened to decode it. Callers that
    hold the real file (the finalizer) pass it; a caller with no file backing
    ``filename`` (a synthetic path in a unit test) leaves it out."""
    digest = hashlib.sha256(source_bytes if source_bytes is not None
                            else source.encode("utf-8")).hexdigest()
    return [{"entry_type": "invariant_set", "metadata": {
        "format_version": "2.3", "uuid": str(uuid4()),
        "creation_time": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "producer": {"name": "Tharkun", "version": "0.2.0"},
        "task": {"input_files": [filename], "input_file_hashes": {filename: digest},
            "specification": specification, "language": "Python"}},
        "content": [i.entry() for i in invariants]}]


def source_points(tree: ast.AST, source: str, filename: str):
    """AST identities and original statement boundaries; columns count characters."""
    from tharkun.solve.sitekey import site_keys

    return {node: (key.ast_path, key.location())
            for node, key in site_keys(tree, source, filename).items()}
