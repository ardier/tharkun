"""``Unsupported``: raised by the absint interpreter (and its mixins) for any construct it does not model.

Its own module so that ``interp.py`` and the mixins it inherits from (``reducers.py``) can both raise it
without importing each other.  ``interp.Unsupported`` is this class.
"""

from __future__ import annotations

import ast


class Unsupported(Exception):
    def __init__(self, reason: str, node: ast.AST | None = None):
        line = getattr(node, "lineno", None)
        super().__init__(f"{reason}" + (f" (line {line})" if line else ""))
        self.reason = reason
        self.line = line
