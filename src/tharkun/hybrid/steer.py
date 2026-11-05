"""The STEER half of limit-then-steer: feedback signals a coverage-guided fuzzer can follow.

Atheris/libFuzzer sees one thing: which code edges an input reached.  Two other facts the owner names as feedback (a
target line fired; how far the observed value is from the wanted value y) are invisible to it unless they are turned
into edges.  This module does that, and nothing more:

* :class:`LineWatch` -- did a target line execute in this run (``sys.monitoring``, one tool id, only the watched
  line stays enabled, so the cost after warm-up is nil).
* :func:`edit_distance` / :func:`numeric_distance` -- how far a value is from y.
* :class:`Ladder` -- a function with one ``elif`` per closeness rung and Atheris-instrumented.  Calling it with a rung
  reaches that rung's edge (:func:`closeness_level` maps a distance onto a rung); the first input that gets closer
  reaches a NEW edge, so libFuzzer keeps it in the corpus and mutates from it.  Distance becomes coverage, and
  hill-climbing on distance becomes what the fuzzer already does.

Nothing here generates inputs; the domain comes from ``tharkun.strategies`` (see :mod:`tharkun.hybrid.domain_codec`).
"""

from __future__ import annotations

import math
import sys
from typing import Any, Sequence


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Levenshtein distance between two sequences (str, bytes, list)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def numeric_distance(a: float, b: float) -> float:
    """``|a - b|``; infinite when either side is NaN (no progress can be read from it)."""
    if a != a or b != b:
        return math.inf
    return abs(a - b)


def closeness_level(distance: float, levels: int, *, kind: str = "log", scale: float = 1.0) -> int:
    """The rung of a ``levels``-rung ladder for ``distance``: ``levels - 1`` is distance 0, 0 is "no progress readable".

    ``kind="log"`` puts the rung at ``log2(1 + d * scale)`` below the top (a distance spanning many orders of magnitude,
    such as an int32 difference, becomes ~``levels`` steps); ``kind="linear"`` puts it at ``d`` (small integral distances,
    such as an edit distance).  A distance past the bottom rung is clamped onto it."""
    top = levels - 1
    if distance != distance or distance == math.inf:
        return 0
    if distance <= 0:
        return top
    steps = int(math.log2(1 + distance * scale)) if kind == "log" else int(distance)
    return max(0, top - 1 - steps)


class Ladder:
    """``levels`` distinct Atheris-instrumented edges, one per rung; ``mark`` walks onto the rung for a distance."""

    def __init__(self, levels: int = 48) -> None:
        import atheris

        self.levels = levels
        src = ["def ladder(level):"]
        for i in range(levels):
            src += [f"    {'if' if i == 0 else 'elif'} level == {i}:", f"        return {i}"]
        src.append("    return -1")
        ns: dict[str, Any] = {}
        exec("\n".join(src), ns)  # noqa: S102 -- generated from an int, no input reaches it
        self._ladder = atheris.instrument_func(ns["ladder"])

    def mark(self, distance: float | None, *, kind: str = "log", scale: float = 1.0) -> int:
        """Reach the edge of ``distance``'s rung (a None distance, such as a call that raised, reaches rung 0)."""
        level = 0 if distance is None else closeness_level(distance, self.levels, kind=kind, scale=scale)
        self._ladder(level)
        return level


class LineWatch:
    """True after the watched ``(filename, line)`` executes; :meth:`reset` between runs.  Python 3.12+ (``sys.monitoring``).

    Every other location returns ``DISABLE`` on first sight, so after warm-up only the watched line costs anything."""

    def __init__(self, filename: str, line: int, *, tool_id: int = 4) -> None:
        mon = sys.monitoring
        self._mon, self._tool = mon, tool_id
        self.filename, self.line, self.fired = filename, line, False
        mon.use_tool_id(tool_id, "tharkun-linewatch")
        mon.register_callback(tool_id, mon.events.LINE, self._on_line)
        mon.set_events(tool_id, mon.events.LINE)

    def _on_line(self, code: Any, line: int) -> Any:
        if code.co_filename == self.filename and line == self.line:
            self.fired = True
            return None
        return self._mon.DISABLE

    def reset(self) -> None:
        self.fired = False

    def close(self) -> None:
        self._mon.set_events(self._tool, 0)
        self._mon.register_callback(self._tool, self._mon.events.LINE, None)
        self._mon.free_tool_id(self._tool)
