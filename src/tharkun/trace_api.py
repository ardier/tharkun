from __future__ import annotations

import dis
import os
import sys
import time
from dataclasses import dataclass, field
from types import FrameType
from typing import Any


@dataclass(frozen=True)
class TracePolicy:
    enabled: bool = True
    spans: bool = False
    # Optional real paths allowlist. When set, only these file paths are recorded.
    paths: set[str] | None = None
    # When set, capture frame.f_locals at these specific lines.
    # {realpath: {line_numbers}} — forces sys.settrace mode (monitoring can't access frames).
    snapshot_lines: dict[str, set[int]] | None = None
    # When True, also capture real (non-serialized) objects in snapshots_real.
    # Used by minleaf for leaf projection on live objects.
    capture_real: bool = False
    # When True, record (previous line -> line) transitions per file, which is
    # what branch coverage counts. Forces sys.settrace mode: sys.monitoring's
    # LINE callback has no frame, so two live frames of the same code object
    # would share one "previous line" and invent arcs neither of them took.
    arcs: bool = False


def _safe_snapshot(v: Any) -> Any:
    """Capture a value safely for JSON serialization."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v[:256] if len(v) > 256 else v
    if isinstance(v, bytes):
        return {"__type__": "bytes", "repr": repr(v[:256]), "len": len(v)}
    if isinstance(v, (list, tuple)):
        return {"__type__": type(v).__name__, "len": len(v), "repr": repr(v)[:200]}
    if isinstance(v, dict):
        return {"__type__": "dict", "len": len(v), "keys": list(v.keys())[:20]}
    if isinstance(v, set):
        return {"__type__": "set", "len": len(v), "repr": repr(v)[:200]}
    if isinstance(v, type):
        return {"__type__": "type", "name": v.__name__}
    # Objects: type + truncated repr
    try:
        r = repr(v)[:200]
    except Exception:
        r = f"<{type(v).__name__}>"
    return {"__type__": type(v).__name__, "repr": r}


@dataclass
class TraceResult:
    executed: dict[str, set[int]] = field(default_factory=dict)
    # Executed (from line, to line) transitions per file. Populated only when
    # TracePolicy.arcs is set. A frame's first line is recorded as (0, line),
    # matching coverage.py's convention for "entered here".
    arcs: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    # Executed source spans (line/col) for evaluation-ish opcodes only (best-effort).
    executed_spans: dict[str, set[tuple[int, int, int | None, int | None]]] = field(default_factory=dict)
    # Variable snapshots at specific lines: {realpath: {lineno: [locals_dict, ...]}}
    # Safe snapshots: JSON-serializable, for diagnostics and condition_solver re-solve.
    snapshots: dict[str, dict[int, list[dict[str, Any]]]] = field(default_factory=dict)
    # Real snapshots: actual live Python objects, for minleaf leaf projection.
    # Only populated when capture_real=True on TracePolicy.
    snapshots_real: dict[str, dict[int, list[dict[str, Any]]]] = field(default_factory=dict)
    duration_s: float = 0.0


class Tracer:
    def __init__(self, policy: TracePolicy) -> None:
        self.policy = policy
        self.result = TraceResult()
        self._t0 = 0.0
        self._prev = None
        self._mode: str | None = None
        self._tool_id: int | None = None
        self._ins_cache: dict[Any, dict[int, tuple[str, dis.Positions | None]]] = {}
        # Per-frame last line, keyed by id(frame). Frames are held alive by the
        # interpreter for as long as they can emit events, and the entry is
        # dropped on "return", so a recycled id cannot inherit a stale line.
        self._last_line: dict[int, int] = {}

    def _want_file(self, fn: str) -> bool:
        if not self.policy.paths:
            return True
        rp = os.path.realpath(fn)
        return rp in self.policy.paths

    def _record_span_op(self, opname: str) -> bool:
        # Heuristic: only record spans for opcodes that correspond to evaluation, not control flow.
        if opname.startswith("LOAD_"):
            return True
        if opname.startswith("STORE_"):
            return True
        if opname.startswith("BINARY_") or opname.startswith("UNARY_"):
            return True
        if opname in {"CALL", "CALL_FUNCTION_EX", "COMPARE_OP", "IS_OP", "CONTAINS_OP"}:
            return True
        if opname.startswith("BUILD_") or opname in {"FORMAT_VALUE", "BUILD_STRING"}:
            return True
        return False

    def _on_line(self, code: Any, line: int) -> None:
        fn = os.path.realpath(getattr(code, "co_filename", ""))
        if not fn or not self._want_file(fn):
            return
        self.result.executed.setdefault(fn, set()).add(int(line))

    def _on_instruction(self, code: Any, offset: int) -> None:
        fn = os.path.realpath(getattr(code, "co_filename", ""))
        if not fn or not self._want_file(fn):
            return

        cache = self._ins_cache.get(code)
        if cache is None:
            cache = {}
            for ins in dis.get_instructions(code):
                cache[int(ins.offset)] = (str(ins.opname), getattr(ins, "positions", None))
            self._ins_cache[code] = cache

        opname, pos = cache.get(int(offset), ("", None))
        if not opname or pos is None or pos.lineno is None:
            return
        if not self._record_span_op(opname):
            return

        sl = int(pos.lineno)
        el = int(pos.end_lineno) if pos.end_lineno is not None else sl
        sc = int(pos.col_offset) if pos.col_offset is not None else None
        ec = int(pos.end_col_offset) if pos.end_col_offset is not None else None
        self.result.executed_spans.setdefault(fn, set()).add((sl, el, sc, ec))

    def __enter__(self) -> "Tracer":
        if self.policy.enabled:
            self._t0 = time.time()
            # When snapshot_lines is set, force sys.settrace mode since
            # sys.monitoring callbacks don't receive frame objects.
            if self.policy.snapshot_lines or self.policy.arcs:
                self._mode = "trace"
                self._prev = sys.gettrace()
                sys.settrace(self._trace)
                return self
            if self.policy.spans and hasattr(sys, "monitoring"):
                self._mode = "monitoring"
                m = sys.monitoring
                tool = None
                # Prefer non-reserved tool ids.
                for tid in (4, 3, 5, 2, 1, 0):
                    try:
                        m.use_tool_id(tid, "tulkas_v0")
                        tool = tid
                        break
                    except ValueError:
                        continue
                if tool is None:
                    # Fall back to line-only tracing if we cannot claim a tool id.
                    self._mode = "trace"
                else:
                    self._tool_id = tool
                    m.register_callback(tool, m.events.LINE, self._on_line)
                    m.register_callback(tool, m.events.INSTRUCTION, self._on_instruction)
                    m.set_events(tool, m.events.LINE | m.events.INSTRUCTION)
                    return self

            self._mode = "trace"
            self._prev = sys.gettrace()
            sys.settrace(self._trace)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.policy.enabled:
            if self._mode == "monitoring" and self._tool_id is not None and hasattr(sys, "monitoring"):
                m = sys.monitoring
                try:
                    m.set_events(self._tool_id, m.events.NO_EVENTS)
                    m.register_callback(self._tool_id, m.events.INSTRUCTION, None)
                    m.register_callback(self._tool_id, m.events.LINE, None)
                finally:
                    m.free_tool_id(self._tool_id)
                self._tool_id = None
            else:
                sys.settrace(self._prev)
            self.result.duration_s = time.time() - self._t0

    def _trace(self, frame: FrameType, event: str, arg):
        if event == "line":
            fn = os.path.realpath(frame.f_code.co_filename)
            if not self._want_file(fn):
                return self._trace
            ln = frame.f_lineno
            self.result.executed.setdefault(fn, set()).add(ln)
            if self.policy.arcs:
                previous = self._last_line.get(id(frame), 0)
                self.result.arcs.setdefault(fn, set()).add((previous, ln))
                self._last_line[id(frame)] = ln
            # Snapshot capture: if this line is in the snapshot set, capture locals.
            if self.policy.snapshot_lines:
                snap_lines = self.policy.snapshot_lines.get(fn)
                if snap_lines and ln in snap_lines:
                    try:
                        locals_snap = {
                            k: _safe_snapshot(v)
                            for k, v in frame.f_locals.items()
                        }
                    except Exception:
                        locals_snap = {"__snapshot_error__": True}
                    self.result.snapshots.setdefault(fn, {}).setdefault(ln, []).append(locals_snap)
                    # Real object capture for minleaf leaf projection.
                    if self.policy.capture_real:
                        try:
                            locals_real = dict(frame.f_locals)
                        except Exception:
                            locals_real = {}
                        self.result.snapshots_real.setdefault(fn, {}).setdefault(ln, []).append(locals_real)
        elif event == "return" and self.policy.arcs:
            self._last_line.pop(id(frame), None)
        return self._trace
