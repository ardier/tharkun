"""Where every node of a lowered program came from ([internal ref]).

A lowering rewrites a program so an analysis that does not model a construct can
still run on it (``comprehension.py`` turns a list comprehension into a loop).
The analysis reports its findings on SITES -- ``(line, col, op)`` -- and every
other artifact of a run (the obligation table, the site census, ``witness.yml``,
the records lanes A and B read) names sites of the ORIGINAL program.  So the
rewrite has to say, for every node it hands out, which node of the original
program it is.  Two rules make that true by construction:

* a node taken over from the original keeps its own ``lineno`` / ``col_offset`` /
  ``end_*`` and carries the AST PATH it had in the original program;
* a node the lowering invents (the accumulator, a loop, a temporary) is placed at
  the position of the original node it stands for, carries that node's path, and is
  marked with a ROLE (``"acc"``, ``"loop"``, ``"temp"``, ...), so a consumer can
  tell "this operation is in the program" from "this operation implements a
  construct of the program".

The path of a node is the sequence of ``(field, index)`` steps from the module:
``body[2].body[0].value.elt`` is ``(("body", 2), ("body", 0), ("value", None),
("elt", None))``.  It is stored on the node (``_lw_path`` / ``_lw_role``) so it
survives ``copy.copy`` and ``copy.deepcopy``, which is how analyses copy trees.

Stdlib only; imports nothing from tharkun.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional, Tuple

Path = Tuple[Tuple[str, Optional[int]], ...]

PATH_ATTR = "_lw_path"
ROLE_ATTR = "_lw_role"


@dataclass(frozen=True)
class Origin:
    """Where one node of a lowered program came from."""

    path: Path
    line: int
    col: int
    end_line: int
    end_col: int
    #: ``None`` for a node taken over from the original program, else what the
    #: lowering invented it for.
    role: Optional[str] = None

    @property
    def synthetic(self) -> bool:
        return self.role is not None

    def text(self) -> str:
        return format_path(self.path)


def annotate(tree: ast.AST) -> None:
    """Stamp every node of ``tree`` with its own path (iteratively; deep trees are common)."""
    stack = [(tree, ())]
    while stack:
        node, path = stack.pop()
        setattr(node, PATH_ATTR, path)
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for i, child in enumerate(value):
                    if isinstance(child, ast.AST):
                        stack.append((child, path + ((field, i),)))
            elif isinstance(value, ast.AST):
                stack.append((value, path + ((field, None),)))


def format_path(path: Path) -> str:
    out = []
    for field, index in path:
        out.append(field if index is None else f"{field}[{index}]")
    return ".".join(out) or "<module>"


def resolve(tree: ast.AST, path: Path) -> ast.AST:
    """The node ``path`` names in ``tree`` (the ORIGINAL program); ``KeyError`` if none."""
    node = tree
    for field, index in path:
        try:
            value = getattr(node, field)
            node = value if index is None else value[index]
        except (AttributeError, IndexError, TypeError) as exc:
            raise KeyError(format_path(path)) from exc
    return node


def path_of(node: ast.AST) -> Optional[Path]:
    return getattr(node, PATH_ATTR, None)


def role_of(node: ast.AST) -> Optional[str]:
    return getattr(node, ROLE_ATTR, None)


def origin_of(node: ast.AST) -> Optional[Origin]:
    """The :class:`Origin` of ``node``, or ``None`` for a node the lowering never saw."""
    path = path_of(node)
    if path is None:
        return None
    return Origin(
        path=path,
        line=getattr(node, "lineno", 0),
        col=getattr(node, "col_offset", 0),
        end_line=getattr(node, "end_lineno", None) or getattr(node, "lineno", 0),
        end_col=getattr(node, "end_col_offset", None) or getattr(node, "col_offset", 0),
        role=role_of(node),
    )


def stand_for(new: ast.AST, anchor: ast.AST, role: str) -> ast.AST:
    """Make ``new`` (a node the lowering invented) stand for ``anchor``: its position
    and path, plus ``role``.  Returns ``new``."""
    ast.copy_location(new, anchor)
    setattr(new, PATH_ATTR, getattr(anchor, PATH_ATTR, ()))
    setattr(new, ROLE_ATTR, role)
    return new
