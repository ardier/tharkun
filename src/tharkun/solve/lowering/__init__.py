"""Exact source lowerings for the analyses that do not model a construct ([internal ref]).

``lower_comprehensions`` turns list / set / dict comprehensions into loops with the SAME
semantics (``comprehension.py``).  Every node of a lowered program carries the path and
position of the original node it stands for (``provenance.py``), so findings on the
lowered program are findings about original sites.  Stdlib only: nothing here imports the
rest of tharkun, and a consumer in tharkun (``solve/absint``) or outside it (Eregion's own
comprehension pass, later) imports this package, never a copy.

Documentation: ``docs/LOWERING.md``.
"""

from __future__ import annotations

from .comprehension import Lowered, Refusal, lower_comprehensions
from .provenance import Origin, annotate, format_path, origin_of, resolve

__all__ = [
    "Lowered",
    "Origin",
    "Refusal",
    "annotate",
    "format_path",
    "lower_comprehensions",
    "origin_of",
    "resolve",
]
