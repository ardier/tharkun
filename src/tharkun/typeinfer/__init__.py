"""``tharkun.typeinfer`` -- the types of a function's parameters, from numbered sources (docs/TYPEINFER.md).

``infer_params(source_path, func)`` gives every parameter a status and ranked candidates, each a ``find_inputs``
kind with the source that gave it, and the structure its uses imply (``Shape``).  ``core/`` is stdlib only and never runs the target (the SOLVE layer,
docs/LAYERS.md); ``checkers/`` (external type checkers as subprocesses) is ruled, not built, and nothing outside
this package may import it (``tests/test_layering.py``).

The package root is lazy (PEP 562): ``from tharkun.typeinfer import annotation_kind`` loads the kind vocabulary and
nothing else.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "infer_params": "tharkun.typeinfer.core.params",
    "candidate_typings": "tharkun.typeinfer.core.params",
    "Candidate": "tharkun.typeinfer.core.params",
    "Evidence": "tharkun.typeinfer.core.params",
    "Report": "tharkun.typeinfer.core.params",
    "SOURCES": "tharkun.typeinfer.core.params",
    "Shape": "tharkun.typeinfer.core.shape",
    "infer_shapes": "tharkun.typeinfer.core.shape",
    "synthesize": "tharkun.typeinfer.core.shape",
    "usage_type": "tharkun.typeinfer.core.usage",
    "annotation_kind": "tharkun.typeinfer.core.kinds",
    "value_kind": "tharkun.typeinfer.core.kinds",
    "type_name": "tharkun.typeinfer.core.kinds",
    "strategy_constraints": "tharkun.typeinfer.core.kinds",
    "drawable": "tharkun.typeinfer.core.kinds",
    "str_or_int": "tharkun.typeinfer.core.kinds",
    "optional": "tharkun.typeinfer.core.kinds",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    home = _EXPORTS.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(home), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
