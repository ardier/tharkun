"""#0814 — which family owns a lowered loop frame.

Owner ruling 2026-09-22 ("the inversion CARRIES the family it belongs to"):
a recursive loop frame is classified ONCE, syntactically, IN THE FORWARD
PASSES, BEFORE THE LIFT -- concretely in
``branch_to_call.py::_try_extract_if``, which runs before extraction -- and
the tag is CARRIED into the IR recursion-table row as a program fact rather
than re-derived downstream.  (#0815: name the pass.  The 2026-09-14 recon
uses "Gate A" for something else, so this module says which pass it means.)
The lift takes only ``kernel``; synthesis dispatches by tag and no route
"claims" a row; codegen renders the tagged family and the codegen doctor
fails a mismatch.  A family that fails on its own normal form is a bug in
that family, not a reason for the kernel to poach the row.

The classifier deliberately depends on NOTHING but :mod:`ast`.  It must be
callable from ``eregion-forward-passes`` (Gate A) and from this package's own
extraction, and it must not be able to consult synthesis, whose answers it
would then be encoding rather than the program's shape.

The tags
--------
``kernel``  branchy: the frame does more than run a straight body per step.
``scan``    an element read indexed by the loop's own counter.
``affine``  non-additive arithmetic on a state var (``* ^ & | % << >> // /``).
``ladder``  a counter stepped over a straight body; nothing of the above.
``none``    no loop-exit guard and no branching -- not a loop frame this
            vocabulary describes.  ``none`` is a LOUD decline, never a silent
            route: nothing may invert a frame nobody claims.

Predicate ORDER is part of the contract
---------------------------------------
Branching is tested FIRST, before the loop-exit guard is required.  A
``while True`` frame (0343-0347, 0087, 0075) has no ``__loop_cond_`` guard at
all and is nothing but its branch; an earlier draft of this classifier
required the guard first and put 23 such frames in ``none``.  Recorded here
because it is the trap the next reader falls into.
"""

from __future__ import annotations

import ast
from typing import Optional

KERNEL = "kernel"
SCAN = "scan"
AFFINE = "affine"
LADDER = "ladder"
NONE = "none"

#: Every tag this vocabulary defines, in the order the docs table lists them.
LOOP_FAMILIES = (AFFINE, LADDER, SCAN, KERNEL, NONE)

#: Non-additive operators.  ``+``/``-`` alone is a ladder's own stepping; an
#: accumulator folded with any of these is an affine/mixing chain (0280).
_NON_ADDITIVE = (
    ast.Mult, ast.BitXor, ast.BitAnd, ast.BitOr, ast.Mod,
    ast.LShift, ast.RShift, ast.FloorDiv, ast.Div, ast.Pow,
)

_LOOP_COND_PREFIX = "__loop_cond_"


def loop_exit_guard_var(node: ast.AST) -> Optional[str]:
    """The counter of a loop-exit guard, or ``None``.

    The shape ``loop_to_recursion`` emits for a frame's base case:
    ``if not __loop_cond_N(i): return (...)`` -- a no-else ``if`` whose test
    is the negation of the lowered loop condition.  This is the exact shape
    ``branch_to_call.py::_try_extract_if`` looks for, which is why the lift
    and the tag agree by construction.
    """

    if not isinstance(node, ast.If) or node.orelse:
        return None
    test = node.test
    if not (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)):
        return None
    call = test.operand
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id.startswith(_LOOP_COND_PREFIX)
    ):
        return None
    if not call.args or not isinstance(call.args[0], ast.Name):
        return None
    return call.args[0].id


def _mentions_lifted_arm(nodes) -> bool:
    """True iff an already-lifted ``__then_*`` / ``__else_*`` arm is called.

    The ordinary (non-Gate-A) lift extracts an EXPLICIT ``if``/``else`` into
    MainState arm helpers before this runs, so a frame can be branchy without
    holding an ``ast.If`` any more.  Missing this reads a branchy frame as a
    ladder.
    """

    for node in nodes:
        if isinstance(node, ast.Name) and ("__then_" in node.id or "__else_" in node.id):
            return True
    return False


def classify_loop_step(counter: Optional[str], step_body) -> str:
    """Classify a frame from its loop counter and the body of ONE step.

    The shared core.  :func:`classify_loop_family` finds the guard in a whole
    frame and calls this; Gate A (#0719a) already HAS the guard and the tail
    it is about to lift and calls this directly with them.  One predicate
    order, so the tag the lift gates on and the tag the IR row carries cannot
    disagree -- they are the same computation.

    *counter* is ``None`` when the frame has no loop-exit guard.
    """

    nodes = [node for stmt in step_body for node in ast.walk(stmt)]

    # (1) branchy -- checked FIRST; see the module docstring's trap note.
    if any(isinstance(node, (ast.If, ast.IfExp)) for node in nodes):
        return KERNEL
    if _mentions_lifted_arm(nodes):
        return KERNEL

    # (2) no guard and no branch: outside this vocabulary.
    if counter is None:
        return NONE

    # (3) an element read indexed by the loop's OWN counter.
    for node in nodes:
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Name)
            and node.slice.id == counter
        ):
            return SCAN

    # (4) non-additive arithmetic anywhere in the step body.
    if any(
        isinstance(node, ast.BinOp) and isinstance(node.op, _NON_ADDITIVE)
        for node in nodes
    ):
        return AFFINE

    # (5) a counter over a straight body.
    return LADDER


def classify_loop_family(fn: ast.FunctionDef | ast.For | ast.While) -> str:
    """Classify a lowered recursive frame or an original Python loop."""

    if isinstance(fn, (ast.For, ast.While)):
        return classify_source_loop(fn)
    guard_index = None
    counter: Optional[str] = None
    for index, stmt in enumerate(fn.body):
        found = loop_exit_guard_var(stmt)
        if found is not None:
            guard_index, counter = index, found
            break
    step_body = fn.body if guard_index is None else fn.body[guard_index + 1:]
    return classify_loop_step(counter, step_body)

# Source-loop adaptation for Tharkun.  The lowered-frame classifier above is
# copied from py-inv/eregion-ir, src/eregion_ir/program_extract/loop_family.py
# (main, accessed 2026-09-27); this adapter operates on Python's original loops.
COUNTER = "counter"
ACCUMULATOR = "accumulator"
SEARCH = "search"
ITERATOR = "iterator"


def classify_source_loop(loop: ast.For | ast.While) -> str:
    """Pick candidate templates from source syntax, never a proof verdict."""
    nodes = [node for stmt in loop.body for node in ast.walk(stmt)]
    if any(isinstance(n, ast.Break) for n in nodes):
        return SEARCH
    assigned = {n.target.id for n in nodes
                if isinstance(n, (ast.AugAssign, ast.AnnAssign))
                and isinstance(n.target, ast.Name)}
    assigned.update(t.id for n in nodes if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name))
    if isinstance(loop, ast.For):
        counter = {loop.target.id} if isinstance(loop.target, ast.Name) else set()
    else:
        counter = {n.id for n in ast.walk(loop.test) if isinstance(n, ast.Name)} & assigned
    if any(isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Name)
           and n.slice.id in counter for n in nodes):
        return SCAN
    if any(isinstance(n, ast.BinOp) and isinstance(n.op, _NON_ADDITIVE)
           for n in nodes):
        return AFFINE
    if any(isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)
           and (isinstance(loop, ast.For) or not any(
               isinstance(t, ast.Name) and t.id == n.target.id
               for t in ast.walk(loop.test))) for n in nodes):
        return ACCUMULATOR
    if isinstance(loop, ast.For):
        return ITERATOR
    if any(isinstance(n, (ast.If, ast.IfExp)) for n in nodes):
        return KERNEL
    return COUNTER
