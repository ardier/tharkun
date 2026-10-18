"""Condition-satisfying API for branch guard expressions.

Given a guard expression (from a branch condition), infers the types of
free variables and computes concrete values that satisfy the guard.
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from tharkun.solve.constants import BUILTIN_NAMES
from tharkun.solve.types import CallableInfo, ParamInfo
from tharkun.solve.guard import (  # #0606: shared guard-AST helpers; old private names kept
    find_branch_at_line as _find_branch_at_line,
    free_names as _free_names,
    try_extract_constant as _try_extract_constant,
    unparse as _unparse,
    unwrap_named_expr as _unwrap_named_expr,
)
from tharkun.solve.loop_exit import (  # #0606: loop-exit reasoning split out; names kept for callers
    _LOOP_EXIT_UNKNOWN,
    _collect_break_exits,
    _collect_loop_exit_facts,
    _loop_touched_names,
    _single_loop_exits,
    collect_loop_exit_facts,
    loop_exit_factenv,
)


class _UniversalSet:
    """A set-like object where `x in _UniversalSet()` is always True.

    Used when building self objects for membership guards like
    ``encoding in self.exclude_encodings`` where we want the guard to pass
    regardless of what ``encoding`` is.
    """
    def __contains__(self, item: Any) -> bool:
        return True

    def __repr__(self) -> str:
        return "{*}"

    def __bool__(self) -> bool:
        return True

    def __iter__(self):
        return iter([])

    def __len__(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VarConstraint:
    """Type and value constraint for one variable in a guard."""

    name: str
    inferred_type: str
    inference_source: str
    concrete_values: list[Any] = field(default_factory=list)
    strategy_hint: str | None = None


@dataclass
class ConditionSolution:
    """Complete solution for a guard expression."""

    guard_source: str
    variables: list[VarConstraint] = field(default_factory=list)
    satisfying_inputs: list[dict[str, Any]] = field(default_factory=list)
    strategy: str = "none"
    confidence: float = 0.0
    desired_branch: str = "then"  # "then" | "else" — records requested polarity
    explanation: str = ""                           # Human-readable rationale for agents
    input_constraints: dict[str, dict] | None = None  # Structured constraints for agents


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Type inference rules
# ---------------------------------------------------------------------------

_STR_METHODS = frozenset({
    "startswith", "endswith", "upper", "lower", "strip", "lstrip", "rstrip",
    "split", "rsplit", "join", "replace", "find", "rfind", "index", "rindex",
    "count", "encode", "format", "isdigit", "isalpha", "isalnum", "title",
    "capitalize", "swapcase", "zfill", "center", "ljust", "rjust",
})
_BYTES_METHODS = frozenset({"decode", "hex"})
_LIST_METHODS = frozenset({"append", "extend", "pop", "insert", "remove", "sort", "reverse", "copy", "clear"})
_SET_METHODS = frozenset({"add", "discard", "remove", "update", "intersection", "union", "difference"})
_DICT_METHODS = frozenset({"keys", "values", "items", "get", "setdefault", "update", "pop", "popitem"})


def _type_of_constant(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, bytes):
        return "bytes"
    return "Any"


def _resolve_int_constant(node: ast.expr, source_path: str) -> int | None:
    """Resolve *node* to an int, following a dotted name into its defining module.

    Protocol code writes its masks as named flags — ``NegotiateFlags.NTLMSSP_NEGOTIATE_
    EXTENDED_SESSIONSECURITY`` — so a bitmask guard is unsolvable until the name is
    turned back into the number.  Looked up in the enclosing module first, then through
    ``import_resolver`` into the module the class was imported from.
    """
    value, ok = _try_extract_constant(node)
    if ok and isinstance(value, int):
        return value

    # Split a dotted reference into (owner class or None, attribute name).
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        owner, attr = node.value.id, node.attr
    elif isinstance(node, ast.Name):
        owner, attr = None, node.id
    else:
        return None

    if not source_path or not os.path.exists(source_path):
        return None

    def _lookup(path: str, depth: int = 0) -> int | None:
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError):
            return None
        # Narrow to the owning class body when the reference was dotted, so an
        # identically-named constant elsewhere in the module cannot shadow it.
        bodies: list[list[ast.stmt]] = []
        if owner is None:
            bodies.append(tree.body)
        else:
            for node2 in tree.body:
                if isinstance(node2, ast.ClassDef) and node2.name == owner:
                    bodies.append(node2.body)
        for body in bodies:
            for stmt in body:
                targets = (stmt.targets if isinstance(stmt, ast.Assign)
                           else [stmt.target] if isinstance(stmt, ast.AnnAssign) and stmt.value
                           else [])
                for tgt in targets:
                    if isinstance(tgt, ast.Name) and tgt.id == attr:
                        val, ok2 = _try_extract_constant(stmt.value)
                        if ok2 and isinstance(val, int):
                            return val
        if depth > 0:
            return None
        # Not here — follow the import that brought `owner` (or the bare name) in.
        try:
            from tharkun.solve.frontend.import_resolver import parse_imports

            imports = parse_imports(path)
        except Exception:
            return None
        target = imports.get(owner or attr)
        if target and os.path.exists(target) and os.path.realpath(target) != os.path.realpath(path):
            return _lookup(target, depth + 1)
        return None

    return _lookup(os.path.realpath(source_path))


def _extract_bitmask_values(name: str, expr: ast.expr, source_path: str = "") -> list[Any]:
    """Values satisfying a bitmask flag test on *name*, e.g. ``if flags & MASK:``.

    ``_extract_comparison_values`` walks ``ast.Compare`` nodes, so a bare
    ``if flags & MASK:`` yields nothing — there is no comparison to read. Yet the guard
    is trivially invertible: any value with the mask's bits set passes it. This is the
    dominant guard shape in protocol parsers (it gates three of five reachable frontiers
    in ntlm_auth's key exchange alone), so it is worth reading directly.

    **Polarity matters.** Asking for the ``else`` edge negates the guard to
    ``not (flags & MASK)``, and walking through the ``not`` to return MASK answers the
    opposite question — the value that makes the guard TRUE. That is not a near miss: in
    an ``if/elif`` chain on flags, reaching the second arm requires *clearing* the first
    arm's bit, so a solver that can only set bits can never leave the first arm. Under an
    odd number of ``not``s the answer is 0.

    Handles the truthiness form and the two explicit forms — ``x & M == M`` and
    ``x & M != 0`` — returning the mask itself, which satisfies all three.
    """
    values: list[Any] = []

    def _mask_of(node: ast.expr) -> int | None:
        """The mask in ``name & <const>`` (either operand order), or None."""
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd)):
            return None
        for var_side, mask_side in ((node.left, node.right), (node.right, node.left)):
            if isinstance(var_side, ast.Name) and var_side.id == name:
                m = _resolve_int_constant(mask_side, source_path)
                if m:                    # a zero mask can never be made truthy
                    return m
        return None

    def walk(node: ast.expr, negated: bool) -> None:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            walk(node.operand, not negated)
            return
        if isinstance(node, ast.BoolOp):
            for v in node.values:
                walk(v, negated)
            return
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            mask = _mask_of(node.left)
            # The right-hand side is named as often as the mask is — the all-bits form
            # is written `x & Flags.SEAL == Flags.SEAL`, not `== 0x20`.
            other = _resolve_int_constant(node.comparators[0], source_path)
            if mask is None or other is None:
                return
            op = node.ops[0]
            # `x & M != 0` and `x & M == M` are both satisfied by M itself; under a
            # `not` each flips to wanting the bits clear.
            want_set: bool | None = None
            if isinstance(op, ast.NotEq) and other == 0:
                want_set = True
            elif isinstance(op, ast.Eq) and other == mask:
                want_set = True
            elif isinstance(op, ast.Eq) and other == 0:
                want_set = False
            if want_set is None:
                return
            if negated:
                want_set = not want_set
            values.append(mask if want_set else 0)
            return
        # A bare `x & M` used as a truth value.
        mask = _mask_of(node)
        if mask is not None:
            values.append(0 if negated else mask)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                walk(child, negated)

    walk(expr, False)

    # Preserve order, drop duplicates.
    seen: set[Any] = set()
    return [v for v in values if not (v in seen or seen.add(v))]


def _infer_from_bitwise(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Rule: a name used as an operand of a bitwise operator is an int."""
    for node in ast.walk(expr):
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)
        ):
            for side in (node.left, node.right):
                if isinstance(side, ast.Name) and side.id == name:
                    return "int", "bitwise_operand"
    return None


def _infer_from_comparison(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Rule 1: Infer type from comparison with literal."""
    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for op in operands:
            unwrapped = _unwrap_named_expr(op)
            if isinstance(unwrapped, ast.Name) and unwrapped.id == name:
                for other in operands:
                    if other is op:
                        continue
                    val, ok = _try_extract_constant(other)
                    if ok:
                        return _type_of_constant(val), "comparison_literal"
        # Subscript comparison: data[:2] == b"\xfe\xff"
        # The variable is a container (list/bytes/str), NOT the element type.
        # We return the container type inferred from the comparison value.
        for op in operands:
            if isinstance(op, ast.Subscript) and isinstance(op.value, ast.Name) and op.value.id == name:
                for other in operands:
                    if other is op:
                        continue
                    if isinstance(other, ast.Constant):
                        val = other.value
                        if isinstance(val, bytes):
                            return "bytes", "subscript"
                        if isinstance(val, str):
                            return "list", "subscript"
                        if isinstance(val, int) and not isinstance(val, bool):
                            # data[k] == <int> → byte indexing; defer to the
                            # byte-input rule (Rule 4b) so it infers ``bytes``.
                            continue
                        return "list", "subscript"
    return None


def _infer_from_isinstance(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Rule 2: Infer type from isinstance() call."""
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "isinstance"):
            continue
        if len(node.args) < 2:
            continue
        arg0 = node.args[0]
        # Direct Name match, or Attribute chain rooted at name.
        # isinstance(func.value, ast.Name) → root is "func"
        if isinstance(arg0, ast.Name) and arg0.id == name:
            pass  # direct match
        else:
            root = arg0
            while isinstance(root, ast.Attribute):
                root = root.value
            if not (isinstance(root, ast.Name) and root.id == name):
                continue
        # Second arg: single type or tuple of types.
        arg1 = node.args[1]
        if isinstance(arg1, ast.Name):
            return arg1.id, "isinstance"
        if isinstance(arg1, ast.Attribute):
            s = _unparse(arg1)
            return (s or "Any"), "isinstance"
        if isinstance(arg1, ast.Tuple):
            types = []
            for elt in arg1.elts:
                if isinstance(elt, ast.Name):
                    types.append(elt.id)
                else:
                    s = _unparse(elt)
                    if s:
                        types.append(s)
            if types:
                return types[0], "isinstance"
    return None


def _build_self_from_guard_constraints(
    name: str,
    expr: ast.expr,
) -> list[object]:
    """Build a SimpleNamespace for self that satisfies guard constraints.

    Scans the entire guard expression for patterns that constrain self's
    attributes and builds an object that satisfies all of them:

    - ``x in self.collection`` → self.collection = {x_default}
    - ``len(self.contents) != N`` → self.contents = [0]*(N+1) or []
    - ``self._method(args)`` → self._method = lambda *a,**k: True
    - ``self.attr is not None and self._usable(self.attr, tried)``
      → self.attr = "value", self._usable = lambda *a,**k: True
    """
    import types as _types

    attrs: dict[str, Any] = {}

    for node in ast.walk(expr):
        # Pattern: x in self.collection
        if isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, ast.In) and isinstance(comp, ast.Attribute):
                    if _attr_chain_root(comp) == name:
                        parts = _attr_chain_parts(comp)
                        if parts and len(parts) == 1:
                            # Build collection that accepts any member
                            attrs[parts[0]] = _UniversalSet()
                elif isinstance(op, ast.NotIn) and isinstance(comp, ast.Attribute):
                    if _attr_chain_root(comp) == name:
                        parts = _attr_chain_parts(comp)
                        if parts and len(parts) == 1:
                            attrs[parts[0]] = set()  # empty set for "not in"

        # Pattern: len(self.attr) comparison — extract the required length
        if isinstance(node, ast.Compare):
            left = node.left
            if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                    and left.func.id == "len" and left.args):
                arg = left.args[0]
                if isinstance(arg, ast.Attribute) and _attr_chain_root(arg) == name:
                    parts = _attr_chain_parts(arg)
                    if parts and len(parts) == 1 and parts[0] not in attrs:
                        # Determine the right length from the comparison
                        for op, comp in zip(node.ops, node.comparators):
                            if isinstance(comp, ast.Constant) and isinstance(comp.value, int):
                                n = comp.value
                                if isinstance(op, ast.NotEq):
                                    attrs[parts[0]] = list(range(n + 1))  # len != n → use n+1
                                elif isinstance(op, (ast.Gt, ast.GtE)):
                                    target = n + 1 if isinstance(op, ast.Gt) else n
                                    attrs[parts[0]] = list(range(target))
                                elif isinstance(op, (ast.Lt, ast.LtE)):
                                    target = n - 1 if isinstance(op, ast.Lt) else n
                                    attrs[parts[0]] = list(range(max(0, target)))
                                elif isinstance(op, ast.Eq):
                                    attrs[parts[0]] = list(range(n))
                                else:
                                    attrs[parts[0]] = ["item"]
                                break
                        else:
                            attrs[parts[0]] = ["item"]
        # Fallback: bare len(self.attr) used as truthiness
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "len" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Attribute) and _attr_chain_root(arg) == name:
                    parts = _attr_chain_parts(arg)
                    if parts and len(parts) == 1 and parts[0] not in attrs:
                        attrs[parts[0]] = ["item"]

        # Pattern: self.method(args) as truthiness or in BoolOp
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if _attr_chain_root(node.func) == name:
                parts = _attr_chain_parts(node.func)
                if parts:
                    method_name = parts[-1]
                    if method_name not in attrs:
                        attrs[method_name] = lambda *a, **k: True

        # Pattern: self.attr is not None (for compound guards)
        if isinstance(node, ast.Compare):
            left = node.left
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(left, ast.Attribute) and _attr_chain_root(left) == name:
                    parts = _attr_chain_parts(left)
                    if parts and isinstance(comp, ast.Constant) and comp.value is None:
                        attr_name = parts[0]
                        if isinstance(op, ast.IsNot) and attr_name not in attrs:
                            attrs[attr_name] = "value"
                        elif isinstance(op, ast.Is) and attr_name not in attrs:
                            attrs[attr_name] = None
                left = comp

    if not attrs:
        return []

    # Also check for `key in self` (self implements __contains__)
    for node in ast.walk(expr):
        if isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, ast.In) and isinstance(comp, ast.Name) and comp.id == name:
                    # self needs __contains__, build a dict-like object
                    ns = _types.SimpleNamespace(**attrs)
                    ns.__contains__ = lambda self_val: True
                    return [ns]

    return [_types.SimpleNamespace(**attrs)]


def _extract_special_pattern_values(
    name: str,
    expr: ast.expr,
) -> list[Any]:
    """Extract values for special guard patterns that the main extractor misses.

    Handles:
    - ``any(x.startswith(p) for p in {"10.", "192.168."})`` → ``"10.xxx"``
    - ``value % N in (a, b, c)`` → ``a`` (first value in the membership set)
    - ``callable(x)`` → ``lambda: None``
    - ``x.count(y) != N`` → string with correct count
    """
    values: list[Any] = []
    _dict_keys_found: set[str] = set()

    for node in ast.walk(expr):
        # Pattern: any(name.method(x) for x in {literals})
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("any", "all") and node.args):
            gen = node.args[0]
            if isinstance(gen, ast.GeneratorExp):
                # Check if the generator references our variable
                elt = gen.elt
                if isinstance(elt, ast.Call) and isinstance(elt.func, ast.Attribute):
                    if isinstance(elt.func.value, ast.Name) and elt.func.value.id == name:
                        method = elt.func.attr
                        # Get the iteration set
                        for comp in gen.generators:
                            if isinstance(comp.iter, (ast.Set, ast.List, ast.Tuple)):
                                elts = [e.value for e in comp.iter.elts
                                        if isinstance(e, ast.Constant)]
                                if elts:
                                    if method == "startswith" and isinstance(elts[0], str):
                                        values.append(elts[0] + "xxx")
                                    elif method == "endswith" and isinstance(elts[0], str):
                                        values.append("xxx" + elts[0])
                                    elif method == "startswith" and isinstance(elts[0], bytes):
                                        values.append(elts[0] + b"\x00" * 4)
                                    else:
                                        values.append(elts[0])

        # Pattern: value % N in (a, b, c) → just use a
        if isinstance(node, ast.Compare):
            left = node.left
            if (isinstance(left, ast.BinOp) and isinstance(left.op, ast.Mod)
                    and isinstance(left.left, ast.Name) and left.left.id == name):
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(op, ast.In) and isinstance(comp, (ast.Tuple, ast.List)):
                        for elt in comp.elts:
                            if isinstance(elt, ast.Constant):
                                values.append(elt.value)
                                break

        # Pattern: callable(name) → lambda: None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "callable" and node.args):
            arg = node.args[0]
            if isinstance(arg, ast.Name) and arg.id == name:
                values.append(lambda: None)

        # Pattern: name.count(char) != N → string with correct count
        if isinstance(node, ast.Compare):
            left = node.left
            if (isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute)
                    and isinstance(left.func.value, ast.Name) and left.func.value.id == name
                    and left.func.attr == "count" and left.args):
                char_arg = left.args[0]
                if isinstance(char_arg, ast.Constant) and isinstance(char_arg.value, str):
                    char = char_arg.value
                    for op, comp in zip(node.ops, node.comparators):
                        if isinstance(comp, ast.Constant) and isinstance(comp.value, int):
                            target = comp.value
                            if isinstance(op, ast.Eq):
                                values.append(char * target + "x")
                            elif isinstance(op, ast.NotEq):
                                # Need count != target, so use count = target + 1
                                values.append(char * (target + 1) + "x")
                            elif isinstance(op, (ast.Gt, ast.GtE)):
                                values.append(char * (target + 1) + "x")

        # Pattern: len(name.split(sep)) == N → name must contain (N-1) instances of sep
        # Also handles >=, >, etc.
        if isinstance(node, ast.Compare):
            left = node.left
            if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                    and left.func.id == "len" and len(left.args) == 1):
                inner = left.args[0]
                if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "split"
                        and isinstance(inner.func.value, ast.Name)
                        and inner.func.value.id == name
                        and inner.args):
                    sep_arg = inner.args[0]
                    if isinstance(sep_arg, ast.Constant) and isinstance(sep_arg.value, str):
                        sep = sep_arg.value
                        for op, comp in zip(node.ops, node.comparators):
                            if isinstance(comp, ast.Constant) and isinstance(comp.value, int):
                                n = comp.value
                                if isinstance(op, ast.Eq):
                                    count = n - 1
                                elif isinstance(op, ast.GtE):
                                    count = n - 1
                                elif isinstance(op, ast.Gt):
                                    count = n
                                else:
                                    continue
                                if count > 0:
                                    values.append(sep.join(["x"] * (count + 1)))

        # Pattern: type coercion — int(name) or float(name) compared to literal
        # int(x) > 10 → x = "11"; float(x) > 0.5 → x = "0.6"
        if isinstance(node, ast.Compare):
            left = node.left
            if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                    and left.func.id in ("int", "float")
                    and len(left.args) == 1):
                arg = left.args[0]
                if isinstance(arg, ast.Name) and arg.id == name:
                    coerce_fn = left.func.id
                    for op, comp in zip(node.ops, node.comparators):
                        if isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float)):
                            v = comp.value
                            if isinstance(op, ast.Gt):
                                result = v + 1 if coerce_fn == "int" else v + 0.1
                            elif isinstance(op, ast.GtE):
                                result = v
                            elif isinstance(op, ast.Lt):
                                result = v - 1 if coerce_fn == "int" else v - 0.1
                            elif isinstance(op, ast.LtE):
                                result = v
                            elif isinstance(op, ast.Eq):
                                result = v
                            elif isinstance(op, ast.NotEq):
                                result = v + 1 if coerce_fn == "int" else v + 0.1
                            else:
                                continue
                            values.append(str(result))

        # Pattern: dict subscript — name["key"] in guard implies name must be
        # a dict with that key.  Collect all keys accessed on name.
        if isinstance(node, ast.Subscript):
            if (isinstance(node.value, ast.Name) and node.value.id == name
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                _dict_keys_found.add(node.slice.value)

        # Pattern: name.get("key") — dict.get implies name is a dict with that key.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == name
                and node.args):
            key_arg = node.args[0]
            if isinstance(key_arg, ast.Constant) and isinstance(key_arg.value, str):
                _dict_keys_found.add(key_arg.value)

    # Synthesize dict from collected keys.
    if _dict_keys_found and not values:
        d: dict[str, Any] = {}
        for key in sorted(_dict_keys_found):
            d[key] = "test"  # Default string value; comparisons may refine.
        # Try to refine values from comparisons on subscripts.
        for node in ast.walk(expr):
            if not isinstance(node, ast.Compare):
                continue
            left = node.left
            if (isinstance(left, ast.Subscript)
                    and isinstance(left.value, ast.Name) and left.value.id == name
                    and isinstance(left.slice, ast.Constant)
                    and isinstance(left.slice.value, str)):
                key = left.slice.value
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(comp, ast.Constant):
                        if isinstance(op, (ast.Eq, ast.GtE, ast.LtE)):
                            d[key] = comp.value
                        elif isinstance(op, ast.Gt) and isinstance(comp.value, (int, float)):
                            d[key] = comp.value + 1
                        elif isinstance(op, ast.Lt) and isinstance(comp.value, (int, float)):
                            d[key] = comp.value - 1
        values.append(d)

    return values


def _extract_membership_from_scope(
    expr: ast.expr,
    scope_resolved: dict[str, Any],
) -> dict[str, list[Any]]:
    """For `x in SCOPE_VAR` where SCOPE_VAR is a resolved collection,
    extract member values for x.

    Returns {var_name: [member_values]} for each membership test found.
    """
    result: dict[str, list[Any]] = {}
    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, ast.In) and isinstance(comp, ast.Name):
                coll_name = comp.id
                if coll_name in scope_resolved:
                    coll = scope_resolved[coll_name]
                    if isinstance(left, ast.Name):
                        var_name = left.id
                        # Extract first few members from the collection
                        members: list[Any] = []
                        try:
                            if isinstance(coll, dict):
                                members = list(coll.keys())[:3]
                            elif isinstance(coll, (list, tuple, set, frozenset)):
                                members = list(coll)[:3]
                        except Exception:
                            pass
                        if members:
                            result[var_name] = members
            elif isinstance(op, ast.NotIn) and isinstance(comp, ast.Name):
                coll_name = comp.id
                if coll_name in scope_resolved:
                    coll = scope_resolved[coll_name]
                    if isinstance(left, ast.Name):
                        var_name = left.id
                        # For "not in", produce a value NOT in the collection
                        if isinstance(coll, (dict, set, frozenset)):
                            result[var_name] = ["__not_a_member__"]
                        elif isinstance(coll, (list, tuple)):
                            result[var_name] = ["__not_a_member__"]
            left = comp
    return result


def _resolve_scope_variables(
    var_names: list[str],
    source_path: str,
    lineno: int,
) -> dict[str, Any]:
    """Resolve free variables that are defined in scope but aren't parameters.

    Looks for:
    1. Module-level assignments (constants like ``powers = [...]``)
    2. Prior assignments in the same function body

    Returns a dict of {name: resolved_value} for variables that were
    successfully evaluated to a concrete value.
    """
    try:
        source = Path(source_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {}

    # Find the enclosing function and its parameter names
    param_names: set[str] = set()
    func_body: list[ast.stmt] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start + 1000)
            if start <= lineno <= end:
                param_names = {arg.arg for arg in node.args.args}
                func_body = node.body
                break

    # Only try to resolve non-parameter free variables
    candidates = [n for n in var_names if n not in param_names]
    if not candidates:
        return {}

    resolved: dict[str, Any] = {}

    # Check module-level assignments (both regular and annotated)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in candidates:
                    val = _try_eval_assignment(node.value, source_path)
                    if val is not None:
                        resolved[target.id] = val
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name) and node.target.id in candidates:
                val = _try_eval_assignment(node.value, source_path)
                if val is not None:
                    resolved[node.target.id] = val

    # Function-level statements BEFORE the guard line, walked in control-flow order
    # so the LATEST reaching definition wins.  Crucially this includes loops: the
    # value a variable carries PAST a loop -- not its pre-loop init -- is what holds
    # at a post-loop point (e.g. the return).  A loop every break-exits with the same
    # constant (and no natural fall-through) carries that constant; any other loop
    # mutation makes it unknown afterwards.
    for stmt in func_body:
        if getattr(stmt, "lineno", 0) >= lineno:
            break
        if isinstance(stmt, ast.Assign):
            val = _try_eval_assignment(stmt.value, source_path)
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id in candidates:
                    if val is not None:
                        resolved[target.id] = val
                    else:
                        resolved.pop(target.id, None)  # reassigned non-constant -> unknown
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            if isinstance(stmt.target, ast.Name) and stmt.target.id in candidates:
                val = _try_eval_assignment(stmt.value, source_path)
                if val is not None:
                    resolved[stmt.target.id] = val
                else:
                    resolved.pop(stmt.target.id, None)
        elif isinstance(stmt, (ast.For, ast.While)):
            exits = _single_loop_exits(stmt)
            breaks = [e for e in exits if e["kind"] == "break"]
            has_natural = any(e["kind"] == "natural" for e in exits)
            touched = _loop_touched_names(stmt)
            for name in candidates:
                if name not in touched:
                    continue  # loop doesn't write it -> prior reaching def survives
                vals = {b["assignments"].get(name, _LOOP_EXIT_UNKNOWN) for b in breaks}
                if breaks and not has_natural and len(vals) == 1 and _LOOP_EXIT_UNKNOWN not in vals:
                    resolved[name] = next(iter(vals))  # uniform break value carries past the loop
                else:
                    resolved.pop(name, None)  # mutated unpredictably / natural fall-through -> unknown

    return resolved


def _try_eval_assignment(value_node: ast.expr, source_path: str) -> Any:
    """Try to evaluate an assignment RHS to a concrete value.

    Only evaluates safe, constant expressions (literals, list comprehensions
    over constants, simple arithmetic).
    """
    source = _unparse(value_node)
    if source is None:
        return None

    try:
        # Safe eval with limited builtins
        safe_builtins = {
            "True": True, "False": False, "None": None,
            "int": int, "float": float, "str": str, "bool": bool,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "range": range, "len": len, "abs": abs,
            "min": min, "max": max, "sum": sum,
            "sorted": sorted, "reversed": reversed,
        }
        result = eval(source, {"__builtins__": safe_builtins})
        # Only accept simple types (not huge objects)
        if isinstance(result, (bool, int, float, str, bytes, tuple, list, set, dict, type(None))):
            return result
    except Exception:
        pass
    return None


def _is_bare_truthiness(name: str, expr: ast.expr) -> bool:
    """Check if *name* appears as a bare truthiness check.

    Returns True for patterns like:
    - ``name`` as the whole expression
    - ``not name``
    - ``name`` as a value in BoolOp(And/Or)
    """
    if isinstance(expr, ast.Name) and expr.id == name:
        return True
    if (isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not)
            and isinstance(expr.operand, ast.Name) and expr.operand.id == name):
        return True
    if isinstance(expr, ast.BoolOp):
        for value in expr.values:
            if isinstance(value, ast.Name) and value.id == name:
                return True
            if (isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not)
                    and isinstance(value.operand, ast.Name) and value.operand.id == name):
                return True
    return False


def _infer_type_from_branch_body(
    name: str,
    source_path: str,
    lineno: int,
) -> str | None:
    """Infer a variable's type by examining how it's used in the branch body.

    When the guard is ``if x:`` and the body does ``x.startswith(...)``,
    we infer ``x`` is a ``str``. This is a static analog of RecordingProxy.
    """
    try:
        source = Path(source_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    # Find the if/while node at this line
    branch_node = None
    for node in ast.walk(tree):
        if (isinstance(node, (ast.If, ast.While, ast.For))
                and getattr(node, "lineno", None) == lineno):
            branch_node = node
            break
    if branch_node is None:
        return None

    # Scan the branch body for usage patterns
    body = branch_node.body if hasattr(branch_node, "body") else []

    str_signals = 0
    list_signals = 0
    dict_signals = 0
    int_signals = 0
    float_signals = 0
    bool_signals = 0

    _STR_METHODS_SET = {
        "startswith", "endswith", "upper", "lower", "strip", "split",
        "replace", "find", "index", "count", "encode", "format",
        "join", "isdigit", "isalpha", "rstrip", "lstrip",
    }
    _LIST_METHODS_SET = {"append", "extend", "insert", "remove", "pop", "sort", "reverse"}
    _DICT_METHODS_SET = {"keys", "values", "items", "get", "setdefault", "update"}

    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        # Method calls: x.method()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == name:
                method = node.func.attr
                if method in _STR_METHODS_SET:
                    str_signals += 2
                elif method in _LIST_METHODS_SET:
                    list_signals += 2
                elif method in _DICT_METHODS_SET:
                    dict_signals += 2

        # Attribute access (without call): x.attr
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == name:
                attr = node.attr
                if attr in _STR_METHODS_SET:
                    str_signals += 1
                elif attr in _LIST_METHODS_SET:
                    list_signals += 1
                elif attr in _DICT_METHODS_SET:
                    dict_signals += 1

        # Subscript: x[0], x["key"]
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == name:
                if isinstance(node.slice, ast.Constant):
                    if isinstance(node.slice.value, int):
                        list_signals += 1
                    elif isinstance(node.slice.value, str):
                        dict_signals += 1

        # BinOp: x + something
        if isinstance(node, ast.BinOp):
            if isinstance(node.left, ast.Name) and node.left.id == name:
                if isinstance(node.right, ast.Constant):
                    if isinstance(node.right.value, str):
                        str_signals += 1
                    elif isinstance(node.right.value, int):
                        int_signals += 1
                    elif isinstance(node.right.value, float):
                        float_signals += 1

        # len(x)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "len" and node.args):
            if isinstance(node.args[0], ast.Name) and node.args[0].id == name:
                str_signals += 1
                list_signals += 1

    scores = {
        "str": str_signals,
        "list": list_signals,
        "dict": dict_signals,
        "int": int_signals,
        "float": float_signals,
        "bool": bool_signals,
    }
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    # If no usage signals found, check parameter annotations
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start + 1000)
            if start <= lineno <= end:
                for arg in node.args.args:
                    if arg.arg == name and arg.annotation:
                        ann = _unparse(arg.annotation)
                        if ann:
                            return ann
                break

    # Last resort: bare Name truthiness with no body signals → likely bool
    return "bool"


def _find_isinstance_module_refs(expr: ast.expr) -> set[str]:
    """Find names used as type references in isinstance arguments.

    These should NOT be treated as free variables to solve for:
    - ``isinstance(node, nodes.Expr)`` → ``nodes`` is a module ref
    - ``isinstance(x, Formatter)`` → ``Formatter`` is a type ref
    - ``isinstance(x, (Tag, NavigableString))`` → both are type refs
    """
    type_refs: set[str] = set()
    # Builtins that are already in BUILTIN_NAMES
    _KNOWN_BUILTINS = {"str", "int", "float", "bool", "list", "dict", "set",
                       "tuple", "bytes", "type", "object", "frozenset",
                       "bytearray", "memoryview", "complex"}

    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "isinstance"):
            continue
        if len(node.args) < 2:
            continue
        type_arg = node.args[1]
        # isinstance(x, mod.Class) — mod is a module ref
        if isinstance(type_arg, ast.Attribute):
            root = _attr_chain_root(type_arg)
            if root:
                type_refs.add(root)
        # isinstance(x, TypeName) — bare class name (starts with uppercase)
        if isinstance(type_arg, ast.Name):
            name = type_arg.id
            if name not in _KNOWN_BUILTINS and name[0:1].isupper():
                type_refs.add(name)
        # isinstance(x, (mod.A, BareName, ...)) — tuple of types
        if isinstance(type_arg, ast.Tuple):
            for elt in type_arg.elts:
                if isinstance(elt, ast.Attribute):
                    root = _attr_chain_root(elt)
                    if root:
                        type_refs.add(root)
                if isinstance(elt, ast.Name):
                    name = elt.id
                    if name not in _KNOWN_BUILTINS and name[0:1].isupper():
                        type_refs.add(name)
    return type_refs


def _infer_from_method_call(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Rule 3: Infer type from method call on the variable."""
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == name):
            continue
        method = func.attr
        # Check for bytes-arg startswith/endswith specifically
        if method in ("startswith", "endswith") and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, bytes):
                return "bytes", "method_call"
        if method in _STR_METHODS:
            return "str", "method_call"
        if method in _BYTES_METHODS:
            return "bytes", "method_call"
        if method in _LIST_METHODS:
            return "list", "method_call"
        if method in _SET_METHODS:
            return "set", "method_call"
        if method in _DICT_METHODS:
            return "dict", "method_call"
    return None


def _infer_from_hasattr(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Detect ``hasattr(name, "attr")`` and return ``("object", "hasattr")``."""
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "hasattr"):
            continue
        if len(node.args) < 2:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.Name) and arg0.id == name:
            return "object", "hasattr"
    return None


def _extract_hasattr_attr_name(name: str, expr: ast.expr) -> str | None:
    """Extract the attribute name from ``hasattr(name, "attr")``."""
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "hasattr"):
            continue
        if len(node.args) < 2:
            continue
        arg0, arg1 = node.args[0], node.args[1]
        if isinstance(arg0, ast.Name) and arg0.id == name:
            if isinstance(arg1, ast.Constant) and isinstance(arg1.value, str):
                return arg1.value
    return None


def _infer_from_method_chain(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Rule 3b: Detect method chains like ``obj.method().attr`` or ``obj.method().method2()``.

    Traverses ``Call.func.value`` chains to find if the root is our variable.
    Returns ``("object", "method_chain")`` if found.
    """
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        # Walk down the chain: func.value might be another Call or Attribute
        cur: ast.AST = node.func
        depth = 0
        while depth < 10:
            if isinstance(cur, ast.Attribute):
                if isinstance(cur.value, ast.Name) and cur.value.id == name:
                    return "object", "method_chain"
                if isinstance(cur.value, ast.Call):
                    cur = cur.value.func
                    depth += 1
                    continue
                break
            break
    return None


def _attr_chain_root(node: ast.AST) -> str | None:
    """Return the root ``Name.id`` of an attribute chain, or ``None``.

    ``c.kind``          -> ``"c"``
    ``node.value.id``   -> ``"node"``
    ``foo()``           -> ``None``
    """
    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id
    return None


def _attr_chain_parts(node: ast.AST) -> list[str] | None:
    """Return the attribute chain as a list of attribute names (excluding root).

    ``c.kind``        -> ``["kind"]``
    ``node.value.id`` -> ``["value", "id"]``
    """
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.reverse()
        return parts
    return None


def _build_attr_satisfying_value(attr_parts: list[str], value: object) -> object:
    """Construct a ``SimpleNamespace`` with nested attributes set to *value*.

    ``_build_attr_satisfying_value(["kind"], "method")``
    -> ``SimpleNamespace(kind="method")``

    ``_build_attr_satisfying_value(["value", "id"], "self")``
    -> ``SimpleNamespace(value=SimpleNamespace(id="self"))``
    """
    import types as _types
    result = value
    for attr in reversed(attr_parts):
        result = _types.SimpleNamespace(**{attr: result})
    return result


def _build_typed_attr_value(
    parts: list[str],
    leaf_val: object,
    chain_types: list[str],
    source_path: str,
) -> object:
    """Build a nested object satisfying an attribute chain, using real instances where possible.

    For ``parts=["config", "timeout"]``, ``leaf_val=61``,
    ``chain_types=["Config", "int"]``:

    1. Tries ``construct_instance(source_path, "Config", overrides={"timeout": 61})``
       → real ``Config(timeout=61)`` instance.
    2. Falls back to ``SimpleNamespace(timeout=61)`` if construction fails.
    3. Wraps the result at each level going out to the root.

    Falls back entirely to ``_build_attr_satisfying_value`` on any error.
    """
    if not chain_types or len(chain_types) != len(parts):
        return _build_attr_satisfying_value(parts, leaf_val)

    try:
        from tharkun.solve.frontend.constructor import construct_instance
        from tharkun.solve.frontend.field_resolver import _is_builtin_type
        from tharkun.solve.frontend.import_resolver import parse_imports

        current: object = leaf_val

        # Walk from innermost (leaf) to outermost, building each level.
        # parts[-1] is the leaf attr; chain_types[-1] is the leaf type (builtin).
        # parts[-2] is the parent attr; chain_types[-2] is the parent class, etc.
        for i in range(len(parts) - 1, -1, -1):
            attr = parts[i]
            # The type at this level is one step back in chain_types:
            # chain_types[i] is the type OF parts[i], so its parent class
            # is chain_types[i-1] (when i > 0) or the root_class (when i == 0).
            # We need to construct chain_types[i-1] with override {attr: current}
            # when i > 0.  At i == 0 the caller wraps it in the root var.
            if i == 0:
                # outermost: just wrap in SimpleNamespace for the root var
                # (root class construction happens in the caller if needed)
                import types as _t
                current = _t.SimpleNamespace(**{attr: current})
            else:
                parent_type = chain_types[i - 1]
                if _is_builtin_type(parent_type):
                    import types as _t
                    current = _t.SimpleNamespace(**{attr: current})
                else:
                    try:
                        # Find source file for parent_type
                        parent_path = source_path
                        imports = parse_imports(source_path)
                        if parent_type in imports:
                            parent_path = imports[parent_type]
                        instance = construct_instance(
                            parent_path, parent_type, overrides={attr: current}
                        )
                        current = instance
                    except Exception:
                        import types as _t
                        current = _t.SimpleNamespace(**{attr: current})

        return current
    except Exception:
        return _build_attr_satisfying_value(parts, leaf_val)


def _infer_from_attr_comparison(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Rule 3c: Infer type from ``obj.attr == literal`` or ``obj.attr in {literals}``.

    Detects patterns where an attribute of the variable is compared with a
    literal.  Returns ``("object", "attr_comparison")`` so the solver knows
    *name* is an object whose attribute has a constrained value.
    """
    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if isinstance(left, ast.Attribute) and _attr_chain_root(left) == name:
            return "object", "attr_comparison"
        for comp in node.comparators:
            if isinstance(comp, ast.Attribute) and _attr_chain_root(comp) == name:
                return "object", "attr_comparison"
    return None


def _extract_attr_comparison_values(
    name: str,
    expr: ast.expr,
) -> list[object]:
    """Extract ``SimpleNamespace`` objects satisfying attribute comparisons.

    Handles:
    - ``obj.attr == literal``
    - ``obj.attr in {literal, ...}``
    """
    values: list[object] = []

    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(left, ast.Attribute) and _attr_chain_root(left) == name:
                parts = _attr_chain_parts(left)
                if parts is None:
                    left = comparator
                    continue
                if isinstance(op, (ast.Eq, ast.Is)):
                    val, ok = _try_extract_constant(comparator)
                    if ok:
                        values.append(_build_attr_satisfying_value(parts, val))
                elif isinstance(op, ast.In):
                    if isinstance(comparator, (ast.Tuple, ast.List, ast.Set)):
                        for elt in comparator.elts:
                            if isinstance(elt, ast.Constant):
                                values.append(
                                    _build_attr_satisfying_value(parts, elt.value)
                                )
            left = comparator

    return values


def _leaf_val_for_op(
    op: ast.cmpop,
    comparator: ast.expr,
    leaf_type: str,
    flipped: bool = False,
) -> Any:
    """Generate a satisfying value for *leaf_type* given a comparison operator.

    ``flipped=True`` means the literal is on the left (``5 > var`` → effective ``var < 5``).
    Returns ``None`` if unable to produce a value.
    """
    val, is_const = _try_extract_constant(comparator)
    if not is_const:
        return None

    if flipped:
        _FLIP: dict[type, type] = {
            ast.Gt: ast.Lt, ast.Lt: ast.Gt,
            ast.GtE: ast.LtE, ast.LtE: ast.GtE,
        }
        op = _FLIP.get(type(op), type(op))()

    if isinstance(op, ast.Eq):
        return val
    if isinstance(op, ast.NotEq):
        if leaf_type == "int" and isinstance(val, int):
            return val + 1
        if leaf_type == "str" and isinstance(val, str):
            return val + "_x"
        return None
    if isinstance(op, (ast.Gt, ast.GtE)):
        if leaf_type in ("int", "Any") and isinstance(val, int):
            return val + 1
        if leaf_type == "float" and isinstance(val, (int, float)):
            return float(val) + 0.1
        return None
    if isinstance(op, (ast.Lt, ast.LtE)):
        if leaf_type in ("int", "Any") and isinstance(val, int):
            return val - 1
        if leaf_type == "float" and isinstance(val, (int, float)):
            return float(val) - 0.1
        return None
    if isinstance(op, ast.In):
        if isinstance(comparator, (ast.Tuple, ast.List, ast.Set)):
            for elt in comparator.elts:
                if isinstance(elt, ast.Constant):
                    return elt.value
        return val
    return None


def _extract_field_resolved_attr_values(
    name: str,
    test_expr: ast.expr,
    root_class: str,
    source_path: str,
) -> list[object]:
    """Build satisfying values for ``name.a.b.c OP literal`` using field resolution.

    Unlike ``_extract_attr_comparison_values`` (which only handles ``==``, ``is``,
    and ``in``), this function:

    1. Resolves the attribute chain type via ``field_resolver.resolve_field_chain``
       to determine the leaf's actual type.
    2. Generates a satisfying leaf value for *any* comparison operator
       (``>``, ``<``, ``>=``, ``<=``, ``!=``, ``==``, ``in``).
    3. Falls back gracefully if resolution fails.

    Returns a list of ``SimpleNamespace`` objects with the leaf value set.
    """
    values: list[object] = []
    try:
        from tharkun.solve.frontend.field_resolver import resolve_field_chain, _is_builtin_type
    except Exception:
        return values

    for node in ast.walk(test_expr):
        if not isinstance(node, ast.Compare):
            continue

        left = node.left
        for op, comparator in zip(node.ops, node.comparators):
            attr_node = None
            flipped = False

            if isinstance(left, ast.Attribute) and _attr_chain_root(left) == name:
                attr_node = left
            elif isinstance(comparator, ast.Attribute) and _attr_chain_root(comparator) == name:
                attr_node = comparator
                flipped = True

            if attr_node is None:
                left = comparator
                continue

            parts = _attr_chain_parts(attr_node)
            if not parts:
                left = comparator
                continue

            try:
                resolution = resolve_field_chain(root_class, parts, source_path)
            except Exception:
                left = comparator
                continue

            if resolution is None:
                left = comparator
                continue

            leaf_type = resolution.leaf_type
            leaf_val = _leaf_val_for_op(op, comparator if not flipped else left, leaf_type, flipped=flipped)
            if leaf_val is None:
                # Try default for the type
                leaf_val = _default_value_for_type(leaf_type)
            if leaf_val is None:
                left = comparator
                continue

            values.append(_build_typed_attr_value(
                parts, leaf_val, resolution.chain_types, source_path
            ))
            left = comparator

    return values


def _infer_from_attr_chain_truthiness(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Rule 3e: Detect bare attribute chain truthiness.

    Patterns:
    - ``obj.a.b`` as standalone (truthiness)
    - ``obj.a.b`` as a value in ``BoolOp(And/Or)``
    - ``not obj.a.b`` (negated truthiness)

    Returns ``("object", "attr_truthiness")`` if a multi-level attribute
    chain rooted at *name* is used for truthiness (not in a comparison).
    """
    # Check if the whole expression is an attribute chain
    if isinstance(expr, ast.Attribute) and _attr_chain_root(expr) == name:
        parts = _attr_chain_parts(expr)
        if parts and len(parts) >= 1:
            return "object", "attr_truthiness"

    # Check UnaryOp(Not, obj.a.b)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        if isinstance(expr.operand, ast.Attribute) and _attr_chain_root(expr.operand) == name:
            parts = _attr_chain_parts(expr.operand)
            if parts and len(parts) >= 1:
                return "object", "attr_truthiness_negated"

    # Check inside BoolOp values
    for node in ast.walk(expr):
        if not isinstance(node, ast.BoolOp):
            continue
        for value in node.values:
            if isinstance(value, ast.Attribute) and _attr_chain_root(value) == name:
                parts = _attr_chain_parts(value)
                if parts and len(parts) >= 1:
                    return "object", "attr_truthiness"
            # not obj.a.b inside BoolOp
            if (isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not)
                    and isinstance(value.operand, ast.Attribute)
                    and _attr_chain_root(value.operand) == name):
                parts = _attr_chain_parts(value.operand)
                if parts and len(parts) >= 1:
                    return "object", "attr_truthiness_negated"

    return None


def _extract_attr_truthiness_value(
    name: str,
    expr: ast.expr,
) -> object | None:
    """Build a SimpleNamespace satisfying a bare attribute chain truthiness check.

    For ``frame.eval_ctx.volatile``: returns
    ``SimpleNamespace(eval_ctx=SimpleNamespace(volatile=True))``

    For ``not self.allow_unicode``: returns
    ``SimpleNamespace(allow_unicode=False)``
    """
    # Find the attribute chain
    candidates: list[tuple[list[str], bool]] = []  # (parts, negated)

    if isinstance(expr, ast.Attribute) and _attr_chain_root(expr) == name:
        parts = _attr_chain_parts(expr)
        if parts:
            candidates.append((parts, False))

    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        if isinstance(expr.operand, ast.Attribute) and _attr_chain_root(expr.operand) == name:
            parts = _attr_chain_parts(expr.operand)
            if parts:
                candidates.append((parts, True))

    for node in ast.walk(expr):
        if not isinstance(node, ast.BoolOp):
            continue
        for value in node.values:
            if isinstance(value, ast.Attribute) and _attr_chain_root(value) == name:
                parts = _attr_chain_parts(value)
                if parts:
                    candidates.append((parts, False))
            if (isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not)
                    and isinstance(value.operand, ast.Attribute)
                    and _attr_chain_root(value.operand) == name):
                parts = _attr_chain_parts(value.operand)
                if parts:
                    candidates.append((parts, True))

    if not candidates:
        return None

    # Use the first (longest) chain
    parts, negated = max(candidates, key=lambda x: len(x[0]))
    leaf_value = False if negated else True
    return _build_attr_satisfying_value(parts, leaf_value)


def _extract_self_identity_values(
    name: str,
    expr: ast.expr,
) -> list[object]:
    """Extract SimpleNamespace values for self.attr identity checks.

    Handles:
    - ``self.attr is not None`` → SimpleNamespace(attr="value")
    - ``self.attr is None`` → SimpleNamespace(attr=None)
    - Also handles compound guards with multiple self.attr checks
    """
    values: list[object] = []
    import types as _types

    attrs_needed: dict[str, Any] = {}  # attr_name → required_value

    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        for op, comp in zip(node.ops, node.comparators):
            # self.attr is [not] None
            if isinstance(left, ast.Attribute) and _attr_chain_root(left) == name:
                parts = _attr_chain_parts(left)
                if parts and isinstance(comp, ast.Constant) and comp.value is None:
                    if isinstance(op, ast.IsNot):
                        # self.attr is not None → need a non-None value
                        attrs_needed[parts[0]] = "sentinel"
                    elif isinstance(op, ast.Is):
                        # self.attr is None → need None
                        attrs_needed[parts[0]] = None
            left = comp

    if attrs_needed:
        kwargs = {}
        for attr, val in attrs_needed.items():
            kwargs[attr] = "value" if val == "sentinel" else val
        values.append(_types.SimpleNamespace(**kwargs))

    return values


def _infer_from_type_is(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Detect ``type(name) is T`` or ``type(name) is not T`` patterns."""
    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        for op, comp in zip(node.ops, node.comparators):
            # type(name) is T
            if (isinstance(op, (ast.Is, ast.IsNot))
                    and isinstance(left, ast.Call)
                    and isinstance(left.func, ast.Name) and left.func.id == "type"
                    and left.args and isinstance(left.args[0], ast.Name)
                    and left.args[0].id == name
                    and isinstance(comp, ast.Name)):
                type_name = comp.id
                if type_name in ("tuple", "list", "dict", "set", "str", "int", "float", "bytes", "bool"):
                    if isinstance(op, ast.Is):
                        return type_name, "type_is"
            left = comp
    return None


def _infer_from_membership_target(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Rule 3f: If name is the RHS of `x in name` or `x not in name`, it's a collection.

    Also: if name is the LHS of `name in collection`, infer its type from
    the collection's known type if possible.
    """
    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        for op, comp in zip(node.ops, node.comparators):
            # x in name / x not in name → name is a set/list
            if isinstance(op, (ast.In, ast.NotIn)):
                if isinstance(comp, ast.Name) and comp.id == name:
                    return "set", "membership_target"
            # name in x → name needs to match x's element type
            # (handled elsewhere — skip here)
            left = comp
    return None


def _infer_from_subscript(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Rule 4: Infer from subscript access.

    Returns ``"dict"`` when the slice is a string constant (``data["key"]``),
    ``"list"`` otherwise (``data[0]``).
    """
    for node in ast.walk(expr):
        if not isinstance(node, ast.Subscript):
            continue
        if isinstance(node.value, ast.Name) and node.value.id == name:
            # data["key"] → dict; data[0] → list
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                return "dict", "subscript"
            return "list", "subscript"
    return None


def _iter_byte_input_atoms(name: str, expr: ast.expr):
    """Yield ``(kind, subscript_node, op, target_int, signed)`` for byte-input atoms.

    A *byte-input atom* is a comparison whose variable operand is either:

    * ``name[k]`` (subscript of the bytes variable) compared to an int, or
    * ``int.from_bytes(name[i:j], order)`` compared to an int.

    ``kind`` is ``"index"`` or ``"from_bytes"``. ``op`` is the (un-flipped,
    variable-on-the-left) comparison operator. ``target_int`` is the int the
    atom is compared against (a plain or *negated* literal), or ``None`` for
    ``==``/``!=`` without a usable constant. ``signed`` is the ``from_bytes``
    call's ``signed=`` kwarg (always False for an index atom). Only int
    comparators are considered — bytes/str literal comparisons are handled by
    the existing subscript path.
    """
    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        for op, comparator in zip(node.ops, node.comparators):
            left_u = _unwrap_named_expr(left)
            comp_u = _unwrap_named_expr(comparator)
            for var_side, other_side, eff_op in (
                (left_u, comp_u, op),
                (comp_u, left_u, _flip_compare_op(op)),
            ):
                target = _const_int(other_side)   # handles plain and negated literals
                # name[k] — direct subscript of the bytes variable.
                if (isinstance(var_side, ast.Subscript)
                        and isinstance(var_side.value, ast.Name)
                        and var_side.value.id == name):
                    yield ("index", var_side, eff_op, target, False)
                    break
                # int.from_bytes(name[i:j], order [, signed=...])
                if _is_from_bytes_of(var_side, name):
                    yield ("from_bytes", var_side.args[0], eff_op, target,
                           _from_bytes_signed(var_side))
                    break
            left = comparator


def _flip_compare_op(op: ast.cmpop) -> ast.cmpop:
    """Return the operator with operands swapped (``a < b`` ⇔ ``b > a``)."""
    return {
        ast.Lt: ast.Gt(), ast.Gt: ast.Lt(),
        ast.LtE: ast.GtE(), ast.GtE: ast.LtE(),
        ast.Eq: ast.Eq(), ast.NotEq: ast.NotEq(),
    }.get(type(op), op)


def _is_from_bytes_of(node: ast.expr, name: str) -> bool:
    """True if *node* is ``int.from_bytes(name[i:j], ...)`` (any byteorder)."""
    if not (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_bytes"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "int"
            and node.args):
        return False
    arg0 = node.args[0]
    return (isinstance(arg0, ast.Subscript)
            and isinstance(arg0.value, ast.Name)
            and arg0.value.id == name)


def _infer_from_byte_input(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Rule 4b: ``name`` is ``bytes`` when tested via byte index or from_bytes.

    Recognizes ``name[k] <op> <int>`` and
    ``int.from_bytes(name[i:j], order) <op> <int>``. Returns ``("bytes",
    "byte_input")`` so value generation synthesizes a satisfying bytes object.
    Runs before the generic subscript rule (which would say ``list``).
    """
    for _atom in _iter_byte_input_atoms(name, expr):
        return "bytes", "byte_input"
    return None


def _infer_from_param_hint(
    name: str,
    callable_info: CallableInfo | None,
) -> tuple[str, str] | None:
    """Rule 5: Use parameter type hint if available."""
    if callable_info is None:
        return None
    for p in callable_info.params:
        if p.name == name and p.type_hint.rendered and p.type_hint.rendered != "Any":
            return p.type_hint.rendered, "param_hint"
    return None


def infer_var_type(
    name: str,
    expr: ast.expr,
    source_path: str,
    lineno: int,
    callable_info: CallableInfo | None = None,
    facts: Any = None,
) -> tuple[str, str]:
    """Infer the type of a variable from its usage in the guard.

    Returns ``(type_string, inference_source)``.

    When *facts* is a ``FactEnv`` (from ``tharkun.minleaf.types``), external
    constraints can bias type inference.  Facts are consulted after all local
    rules but before the fallback.
    """
    # Rule 6: self attribute
    if name == "self":
        return "object", "self_attr"
    if name == "cls":
        return "type", "self_attr"

    # Rule 2: isinstance (highest signal)
    r = _infer_from_isinstance(name, expr)
    if r:
        return r

    # Rule 1: comparison with literal
    r = _infer_from_comparison(name, expr)
    if r:
        return r

    # Rule 1b: bitwise operand — `flags & MASK` makes flags an int
    r = _infer_from_bitwise(name, expr)
    if r:
        return r

    # Rule 3: method call
    r = _infer_from_method_call(name, expr)
    if r:
        return r

    # Rule 3b: method chain (obj.method().attr)
    r = _infer_from_method_chain(name, expr)
    if r:
        return r

    # Rule 3c: hasattr(name, "attr")
    r = _infer_from_hasattr(name, expr)
    if r:
        return r

    # Rule 3d: attribute comparison (obj.attr == literal / obj.attr in {..})
    r = _infer_from_attr_comparison(name, expr)
    if r:
        return r

    # Rule 3e: bare attribute chain truthiness (obj.a.b.c or not obj.a.b)
    r = _infer_from_attr_chain_truthiness(name, expr)
    if r:
        return r

    # Rule 3f: type(x) is T → x is T
    r = _infer_from_type_is(name, expr)
    if r:
        return r

    # Rule 3g: membership target type — `x in name` or `x not in name` → name is a collection
    r = _infer_from_membership_target(name, expr)
    if r:
        return r

    # Rule 4b: byte-input (data[k] / int.from_bytes(data[i:j])) — before the
    # generic subscript rule, which would mis-infer ``list``.
    r = _infer_from_byte_input(name, expr)
    if r:
        return r

    # Rule 4: subscript
    r = _infer_from_subscript(name, expr)
    if r:
        return r

    # Rule 5: param hint
    r = _infer_from_param_hint(name, callable_info)
    if r:
        return r

    # Rule 5b: external facts — infer type from eq values or bounds.
    if facts is not None:
        vf = getattr(facts, "facts", {}).get(name)
        if vf is not None:
            eq = getattr(vf, "eq", None)
            if eq and len(eq) > 0:
                first_val = next(iter(eq))
                return type(first_val).__name__, "facts_eq"
            lo = getattr(vf, "lo", None)
            hi = getattr(vf, "hi", None)
            if lo is not None or hi is not None:
                if isinstance(lo, float) or isinstance(hi, float):
                    return "float", "facts_bounds"
                return "int", "facts_bounds"

    # Rule 7: fallback
    return "Any", "fallback"


# ---------------------------------------------------------------------------
# Value generation
# ---------------------------------------------------------------------------


def _extract_len_constraint(
    name: str,
    expr: ast.expr,
) -> int | None:
    """Detect a ``len(name)`` comparison and return a length that satisfies it.

    Handles ``>``/``>=``/``==``, and also ``!=`` (any length other than N — return N+1,
    always non-zero and distinct) and ``<``/``<=`` (a length below the bound). Returns
    ``None`` if no such constraint is found.
    """
    def _for_op(op: ast.cmpop, n: int, flipped: bool) -> int | None:
        # `flipped` means the comparison is `N OP len(x)`, so the effective op on len(x)
        # is the mirror.
        eff = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE,
               ast.GtE: ast.LtE}.get(type(op), type(op)) if flipped else type(op)
        if eff is ast.Gt:
            return n + 1
        if eff is ast.GtE or eff is ast.Eq:
            return n
        if eff is ast.NotEq:
            return n + 1                      # length N+1 always differs from N
        if eff is ast.Lt:
            return n - 1 if n - 1 >= 1 else None
        if eff is ast.LtE:
            return n if n >= 1 else None
        return None

    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        for op, comparator in zip(node.ops, node.comparators):
            # len(x) OP N
            if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                    and left.func.id == "len" and len(left.args) == 1
                    and isinstance(left.args[0], ast.Name) and left.args[0].id == name
                    and isinstance(comparator, ast.Constant) and isinstance(comparator.value, int)):
                got = _for_op(op, comparator.value, flipped=False)
                if got is not None:
                    return got
            # N OP len(x) (flipped)
            if (isinstance(comparator, ast.Call) and isinstance(comparator.func, ast.Name)
                    and comparator.func.id == "len" and len(comparator.args) == 1
                    and isinstance(comparator.args[0], ast.Name) and comparator.args[0].id == name
                    and isinstance(left, ast.Constant) and isinstance(left.value, int)):
                got = _for_op(op, left.value, flipped=True)
                if got is not None:
                    return got
            left = comparator
    return None


def _extract_comparison_values(
    name: str,
    expr: ast.expr,
    facts: Any = None,
    source_path: str = "",
    lineno: int = 0,
) -> list[Any]:
    """Extract concrete values satisfying comparisons for *name*.

    When *facts* is provided (a ``FactEnv``), values are filtered by the
    external whitelist (``eq``), exclusions (``neq``), and bounds (``lo``/``hi``).
    """
    values: list[Any] = []

    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue

        # Identify which operand is our variable and which is the literal.
        left = node.left
        for op, comparator in zip(node.ops, node.comparators):
            var_node = None
            lit_node = None
            flipped = False

            left_u = _unwrap_named_expr(left)
            comp_u = _unwrap_named_expr(comparator)
            left_val, left_is_const = _try_extract_constant(left_u)
            comp_val, comp_is_const = _try_extract_constant(comp_u)
            if isinstance(left_u, ast.Name) and left_u.id == name and comp_is_const:
                var_node = left_u
                # Create a synthetic Constant node for the extracted value.
                lit_node = ast.Constant(value=comp_val)
            elif isinstance(comp_u, ast.Name) and comp_u.id == name and left_is_const:
                var_node = comp_u
                lit_node = ast.Constant(value=left_val)
                flipped = True
            # Subscript: data[:2] == b"\xfe\xff"
            elif (isinstance(left, ast.Subscript) and isinstance(left.value, ast.Name)
                  and left.value.id == name and isinstance(comparator, ast.Constant)):
                # Can't directly generate a value for the subscript case easily,
                # but we can use the full literal padded.
                v = comparator.value
                if isinstance(v, bytes):
                    values.append(v + b"\x00" * 10)
                elif isinstance(v, str):
                    values.append(v + "x" * 10)
                left = comparator
                continue
            else:
                left = comparator
                continue

            v = lit_node.value
            actual_op = op

            # When flipped, invert the operator semantics:
            #   lit > var → var < lit   (flipped Gt = effective Lt)
            #   lit >= var → var <= lit  (flipped GtE = effective LtE)
            #   lit < var → var > lit    (flipped Lt = effective Gt)
            #   lit <= var → var >= lit  (flipped LtE = effective GtE)
            effective_op = actual_op
            if flipped:
                _FLIP_MAP = {ast.Gt: ast.Lt, ast.Lt: ast.Gt, ast.GtE: ast.LtE, ast.LtE: ast.GtE}
                effective_op = _FLIP_MAP.get(type(actual_op), actual_op)
                if isinstance(effective_op, type):
                    effective_op = effective_op()

            if isinstance(effective_op, ast.Eq):
                values.append(v)
            elif isinstance(effective_op, ast.NotEq):
                if isinstance(v, int):
                    values.append(v + 1)
                elif isinstance(v, str):
                    values.append(v + "_")
            elif isinstance(effective_op, ast.Gt):
                if isinstance(v, (int, float)):
                    values.append(v + 1 if isinstance(v, int) else v + 0.1)
            elif isinstance(effective_op, ast.GtE):
                if isinstance(v, (int, float)):
                    values.append(v)
            elif isinstance(effective_op, ast.Lt):
                if isinstance(v, (int, float)):
                    values.append(v - 1 if isinstance(v, int) else v - 0.1)
            elif isinstance(effective_op, ast.LtE):
                if isinstance(v, (int, float)):
                    values.append(v)
            elif isinstance(actual_op, ast.In):
                if isinstance(comparator, (ast.Tuple, ast.List, ast.Set)):
                    for elt in comparator.elts:
                        if isinstance(elt, ast.Constant):
                            values.append(elt.value)

            left = comparator

    # Handle `In` with tuple/list on right side directly.
    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        left_u2 = _unwrap_named_expr(node.left)
        if isinstance(left_u2, ast.Name) and left_u2.id == name:
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, ast.In) and isinstance(comp, (ast.Tuple, ast.List, ast.Set)):
                    for elt in comp.elts:
                        if isinstance(elt, ast.Constant) and elt.value not in values:
                            values.append(elt.value)

    # Handle `is None` / `is not None` identity checks.
    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        left_u3 = _unwrap_named_expr(node.left)
        if isinstance(left_u3, ast.Name) and left_u3.id == name:
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(comp, ast.Constant) and comp.value is None:
                    if isinstance(op, ast.Is):
                        values.append(None)
                    elif isinstance(op, ast.IsNot):
                        values.append(1)  # any truthy non-None value

    # Handle known function call predicates: math.isnan(x), math.isinf(x), etc.
    # When the guard IS the call (truthiness) or NOT the call (negated), produce
    # the value that makes the predicate return the expected boolean.
    func_vals = _extract_known_predicate_values(name, expr)
    if func_vals:
        values.extend(func_vals)

    # Handle `x in "chars"` or `char in value` string/bytes containment.
    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        left_u4 = _unwrap_named_expr(node.left)
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, ast.In):
                # x in "abc" → x = "a"
                if (isinstance(left_u4, ast.Name) and left_u4.id == name
                        and isinstance(comp, ast.Constant) and isinstance(comp.value, str)
                        and comp.value):
                    values.append(comp.value[0])
                # "x" in value → value = "x..."
                if (isinstance(comp, ast.Name) and comp.id == name
                        and isinstance(left_u4, ast.Constant) and isinstance(left_u4.value, str)):
                    values.append(left_u4.value + "abc")
                # b"\x1b" in data → data = b"...\x1b..."  (byte containment)
                if (isinstance(comp, ast.Name) and comp.id == name
                        and isinstance(left_u4, ast.Constant) and isinstance(left_u4.value, bytes)):
                    values.append(b"\x00" * 4 + left_u4.value + b"\x00" * 4)
                # x in b"..." → x = first byte
                if (isinstance(left_u4, ast.Name) and left_u4.id == name
                        and isinstance(comp, ast.Constant) and isinstance(comp.value, bytes)
                        and comp.value):
                    values.append(comp.value[0:1])
            left_u4 = comp

    # Apply external facts filtering.
    if facts is not None:
        vf = getattr(facts, "facts", {}).get(name)
        if vf is not None:
            eq = getattr(vf, "eq", None)
            neq = getattr(vf, "neq", None) or set()
            lo = getattr(vf, "lo", None)
            hi = getattr(vf, "hi", None)
            # Whitelist: prefer eq values that satisfy the constraints.
            if eq is not None:
                filtered = [v for v in values if v in eq]
                if filtered:
                    values = filtered
                else:
                    # No overlap — use whitelist values directly (external
                    # evidence overrides heuristic generation).
                    values = list(eq)[:5]
            if neq:
                values = [v for v in values if v not in neq]
            if lo is not None:
                values = [v for v in values if not isinstance(v, (int, float)) or v >= lo]
            if hi is not None:
                values = [v for v in values if not isinstance(v, (int, float)) or v <= hi]

    return values


# ---------------------------------------------------------------------------
# Byte-input value generation (int.from_bytes / byte indexing)
# ---------------------------------------------------------------------------
#
# Forward solving for byte-input fuzz gates: synthesize a concrete ``bytes``
# value for a variable tested via ``data[k]`` or
# ``int.from_bytes(data[i:j], order)``.  This is what byte-input parsers
# (ASN.1 DER header tag, length-prefixed wire formats) gate the function body
# on.  It is *forward* value generation only — backward inversion of a deep
# constraint on the lifted integer onto raw bytes is out of scope (the SPEC
# defers expression inversion to glamdring).


def _const_int(node: ast.expr) -> int | None:
    """Return the int value of a Constant / negated-Constant node, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
            and not isinstance(node.operand.value, bool)):
        return -node.operand.value
    return None


def _slice_bounds(sub: ast.Subscript) -> tuple[int, int, bool] | None:
    """Resolve a subscript slice into ``(start, length, is_single_index)``.

    Handles ``data[k]`` (single index → length 1) and ``data[i:j]`` with
    non-negative integer literal bounds.  Returns None for negative indices,
    open-ended/dynamic slices, or stepped slices (out of scope — these need
    runtime length, which forward static synthesis can't pin down safely).
    """
    sl = sub.slice
    idx = _const_int(sl)
    if idx is not None:
        if idx < 0:
            return None
        return idx, 1, True
    if isinstance(sl, ast.Slice):
        if sl.step is not None:
            return None
        lower = 0 if sl.lower is None else _const_int(sl.lower)
        if lower is None or lower < 0:
            return None
        if sl.upper is None:
            return None  # open-ended slice → unknown width
        upper = _const_int(sl.upper)
        if upper is None or upper < lower:
            return None
        return lower, upper - lower, False
    return None


def _from_bytes_order(call: ast.Call) -> str:
    """Extract the byteorder of an ``int.from_bytes`` call (default ``big``)."""
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) \
            and isinstance(call.args[1].value, str):
        return call.args[1].value
    for kw in call.keywords:
        if kw.arg == "byteorder" and isinstance(kw.value, ast.Constant) \
                and isinstance(kw.value.value, str):
            return kw.value.value
    return "big"  # Python's default since 3.11


def _from_bytes_signed(call: ast.Call) -> bool:
    """True if an ``int.from_bytes`` call passes ``signed=True`` (default False)."""
    for kw in call.keywords:
        if kw.arg == "signed" and isinstance(kw.value, ast.Constant):
            return bool(kw.value.value)
    return False


def _target_int_for_op(op: ast.cmpop, target: int | None, width: int,
                       signed: bool = False) -> int | None:
    """Pick an integer in the decode's range satisfying ``value <op> target``.

    ``op`` is normalized so the byte-derived value is on the left. The range is
    ``[0, 256**width)`` for an unsigned decode and ``[-2**(8w-1), 2**(8w-1))``
    for a signed one. Returns None if no in-range value satisfies the constraint.
    """
    if signed:
        lo = -(1 << (8 * width - 1))
        hi = (1 << (8 * width - 1)) - 1
    else:
        lo, hi = 0, (1 << (8 * width)) - 1
    if target is None:
        # ==/!= without a usable constant, or unknown op → pick a nonzero in-range value.
        return 1 if hi >= 1 else lo
    if isinstance(op, ast.Eq):
        return target if lo <= target <= hi else None
    if isinstance(op, ast.NotEq):
        return lo if target != lo else (lo + 1 if lo + 1 <= hi else None)
    if isinstance(op, ast.Gt):
        cand = target + 1
        return cand if cand <= hi else None
    if isinstance(op, ast.GtE):
        cand = max(target, lo)
        return cand if cand <= hi else None
    if isinstance(op, ast.Lt):
        cand = target - 1
        return cand if cand >= lo else None
    if isinstance(op, ast.LtE):
        cand = min(target, hi)
        return cand if cand >= lo else None
    return None


def _extract_byte_input_values(name: str, expr: ast.expr) -> list[Any]:
    """Synthesize a ``bytes`` value satisfying byte-index / from_bytes atoms.

    Collects all ``data[k] <op> int`` and
    ``int.from_bytes(data[i:j], order) <op> int`` atoms for *name*, writes each
    target into the right slice positions of a single bytearray (sized to cover
    the widest atom), and returns ``[bytes(...)]``.  Returns ``[]`` when no
    byte-input atom is present or a constraint is unsatisfiable in range.

    When multiple atoms touch overlapping bytes (rare), the last write wins;
    this is best-effort forward synthesis, and the caller verifies the result
    against the full guard before accepting it.
    """
    writes: list[tuple[int, bytes]] = []  # (start_offset, encoded_bytes)
    max_end = 0
    found = False

    for kind, sub, op, target, signed in _iter_byte_input_atoms(name, expr):
        found = True
        bounds = _slice_bounds(sub)
        if bounds is None:
            continue
        start, length, is_single = bounds
        if length <= 0:
            continue

        if kind == "index":
            # data[k] yields one int in [0, 255] (a single byte is never signed).
            val = _target_int_for_op(op, target, 1)
            if val is None or not (0 <= val <= 255):
                continue
            writes.append((start, bytes([val])))
            max_end = max(max_end, start + 1)
        else:  # from_bytes over data[start:start+length]
            order = _from_bytes_order(_from_bytes_call_for(sub, expr, name))
            val = _target_int_for_op(op, target, length, signed)
            if val is None:
                continue
            try:
                encoded = int(val).to_bytes(length, order if order in ("big", "little") else "big",
                                            signed=signed)
            except (OverflowError, ValueError):
                continue
            writes.append((start, encoded))
            max_end = max(max_end, start + length)

    if not found or not writes:
        return []

    buf = bytearray(max(max_end, 1))
    for start, encoded in writes:
        buf[start:start + len(encoded)] = encoded
    return [bytes(buf)]


def _byte_guard_has_rescue(name: str, expr: ast.expr) -> bool:
    """True if the guard has a leaf on *name* that could satisfy it apart from byte atoms.

    ``_byte_atoms_unsatisfiable`` reasons only about byte-index / from_bytes atoms, so it
    may conclude "unsatisfiable" while a NON-byte clause on the same variable — a
    ``data.startswith(b'AB')``, a ``data != b''``, or a disjunction hiding either — would
    have satisfied the guard. Any of those makes the byte-atom analysis incomplete, so it
    must not claim the guard dead. (A disjunction is a rescue even between byte atoms: an
    out-of-range OR out-of-range is still unsat, but mixing with a non-byte disjunct is
    not, and the cheap, sound move is to bail on any OR.)
    """
    for node in ast.walk(expr):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            return True
        # A method call on the variable — data.startswith(...), data.endswith(...) — is a
        # non-byte satisfiable leaf.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == name):
            return True
        # A bare `name <op> ...` comparison (e.g. data != b'') is satisfiable independently
        # of the byte atoms — the variable appears whole, not as name[i:j] / name[k].
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                u = _unwrap_named_expr(operand)
                if isinstance(u, ast.Name) and u.id == name:
                    return True
    return False


def _byte_atoms_unsatisfiable(name: str, expr: ast.expr) -> bool:
    """True when the guard is nothing but byte atoms and every one is out of range.

    If no in-range value can satisfy any atom AND there is no other leaf on *name* that
    could satisfy the guard (see :func:`_byte_guard_has_rescue`), the guard is
    unsatisfiable. Conservative: an atom with an unresolvable slice, any satisfiable atom,
    or any rescuing structure returns False — this proves unsatisfiability, never guesses.
    """
    if _byte_guard_has_rescue(name, expr):
        return False
    saw = False
    for kind, sub, op, target, signed in _iter_byte_input_atoms(name, expr):
        bounds = _slice_bounds(sub)
        if bounds is None:
            return False                      # unresolvable width → cannot claim unsat
        _start, length, _single = bounds
        width = 1 if kind == "index" else length
        sgn = signed if kind == "from_bytes" else False
        if _target_int_for_op(op, target, width, sgn) is not None:
            return False                      # this atom IS satisfiable in range
        saw = True
    return saw


def _from_bytes_call_for(sub_arg: ast.Subscript, expr: ast.expr, name: str) -> ast.Call:
    """Find the ``int.from_bytes`` Call whose first arg *is* ``sub_arg``.

    ``_iter_byte_input_atoms`` yields the subscript (the call's arg); we walk
    the expression to recover the enclosing call so we can read its byteorder.
    """
    for node in ast.walk(expr):
        if (isinstance(node, ast.Call) and _is_from_bytes_of(node, name)
                and node.args and node.args[0] is sub_arg):
            return node
    # Fallback: a synthetic big-endian call (byteorder default).
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="int"), attr="from_bytes"),
        args=[sub_arg], keywords=[],
    )


# Known function call predicates → satisfying argument values.
_KNOWN_PREDICATES: dict[str, Any] = {
    "math.isnan": float("nan"),
    "math.isinf": float("inf"),
    "math.isfinite": 1.0,   # finite → True
    "math.isinf_neg": float("-inf"),
    "str.isdigit": "123",
    "str.isalpha": "abc",
    "str.isalnum": "abc123",
    "str.isupper": "ABC",
    "str.islower": "abc",
    "str.isspace": "   ",
}

# Negated predicate values (when `not func(x)` is the guard).
_KNOWN_PREDICATES_NEGATED: dict[str, Any] = {
    "math.isnan": 0.0,      # not nan → any finite float
    "math.isinf": 0.0,      # not inf → any finite float
    "math.isfinite": float("nan"),  # not finite → nan or inf
}


def _extract_known_predicate_values(name: str, expr: ast.expr) -> list[Any]:
    """Extract values for known function call predicates.

    Handles:
    - ``math.isnan(x)`` → ``float('nan')``
    - ``not math.isfinite(x)`` → ``float('nan')``
    - ``math.isinf(x) and x < 0`` → ``float('-inf')``
    """
    values: list[Any] = []

    for node in ast.walk(expr):
        call = None
        negated = False

        # Direct call: math.isnan(x)
        if isinstance(node, ast.Call):
            call = node
        # Negated call: not math.isnan(x)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            if isinstance(node.operand, ast.Call):
                call = node.operand
                negated = True

        if call is None:
            continue

        # Check if our variable is the argument
        if not call.args:
            continue
        arg = call.args[0]
        if not (isinstance(arg, ast.Name) and arg.id == name):
            continue

        # Resolve the function name
        func = call.func
        func_name = None
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            func_name = f"{func.value.id}.{func.attr}"
        elif isinstance(func, ast.Name):
            func_name = func.id

        if func_name is None:
            continue

        registry = _KNOWN_PREDICATES_NEGATED if negated else _KNOWN_PREDICATES
        val = registry.get(func_name)
        if val is not None and val not in values:
            values.append(val)

    # Special: math.isinf(x) and x < 0 → float('-inf')
    # Check if there's a compound guard with both isinf and x < 0
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And):
        has_isinf = False
        has_neg = False
        for clause in expr.values:
            if isinstance(clause, ast.Call):
                func = clause.func
                if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                        and func.value.id == "math" and func.attr == "isinf"
                        and clause.args and isinstance(clause.args[0], ast.Name)
                        and clause.args[0].id == name):
                    has_isinf = True
            if isinstance(clause, ast.Compare):
                left = _unwrap_named_expr(clause.left)
                if isinstance(left, ast.Name) and left.id == name:
                    for op, comp in zip(clause.ops, clause.comparators):
                        if isinstance(op, ast.Lt) and isinstance(comp, ast.Constant) and comp.value == 0:
                            has_neg = True
        if has_isinf and has_neg:
            values = [float("-inf")]  # Override with negative infinity

    return values


def _resolve_compiled_pattern(
    pat_name: str, source_path: str,
) -> tuple[Any, int, bool] | None:
    """If ``pat_name = re.compile(<literal>[, flags])`` in the module, return
    ``(pattern, flags, is_bytes)``.

    Reads the pattern straight from the AST rather than evaluating it — a compiled
    ``re.Pattern`` is not one of the simple types ``_try_eval_assignment`` returns, so
    ``scope_resolved`` never carries it. Flags (``re.I | re.M``) are evaluated with only
    the ``re`` module in scope; on any surprise the flags default to 0."""
    import re as _re

    # Fast path: a literal ``re.compile("...", flags)`` read straight from the AST. While
    # walking, note whether *pat_name* is assigned here at all and whether any assignment is
    # a ``re.compile(...)`` — that gates the import fallback below, so a genuinely non-regex
    # ``obj.match(x)`` (``obj = SomeMatcher()``) never triggers a module import.
    assigned_here = False
    evidence_compiled = False
    try:
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == pat_name for t in node.targets)):
                continue
            assigned_here = True
            call = node.value
            is_re_compile = (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                             and call.func.attr == "compile"
                             and isinstance(call.func.value, ast.Name)
                             and call.func.value.id == "re" and bool(call.args))
            if not is_re_compile:
                continue
            evidence_compiled = True
            if not (isinstance(call.args[0], ast.Constant)
                    and isinstance(call.args[0].value, (str, bytes))):
                continue  # computed pattern — resolved by the import fallback
            pat = call.args[0].value
            flags = 0
            if len(call.args) >= 2:
                try:
                    flags = int(eval(_unparse(call.args[1]),  # noqa: S307
                                     {"re": _re, "__builtins__": {}}))
                except Exception:
                    flags = 0
            return pat, flags, isinstance(pat, bytes)

    # Fallback: the pattern is COMPUTED (``re.compile(r"[%s]+" % re.escape(CONST))`` — real
    # code, e.g. gunicorn's TOKEN_RE), so read the actual compiled object by importing the
    # module. Consistent with how the driver already loads targets for execution. Only run
    # when the AST gave evidence this name is a compiled pattern (``evidence_compiled``) or
    # is not assigned in this file at all (likely imported) — never for a name we saw bound
    # to a non-``re.compile`` value, which would import the module to no purpose.
    if assigned_here and not evidence_compiled:
        return None
    try:
        from tharkun.solve.loader import load_module_for_execution

        mod, _ = load_module_for_execution(source_path)
        obj = getattr(mod, pat_name, None) if mod is not None else None
        if isinstance(obj, _re.Pattern):
            return obj.pattern, int(obj.flags), isinstance(obj.pattern, bytes)
    except Exception:
        pass
    return None


def _extract_regex_pattern(
    name: str,
    expr: ast.expr,
    source_path: str = "",
) -> tuple[Any, int, bool, str | None] | None:
    """Detect a regex guard on *name* and return ``(pattern, flags, is_bytes, const_name)``.

    ``const_name`` is the compiled-pattern constant's name (so a verifier can bind it) or
    ``None`` for the inline ``re.match(literal, x)`` shape, where ``re`` alone suffices.

    Two shapes:

    * ``re.match|search|fullmatch(<literal>, name)`` — the pattern is a str/bytes literal
      (the historic case), the variable is arg 1.
    * ``PAT.match|search|fullmatch(name)`` — ``PAT`` is a module-level ``re.compile(...)``
      constant and the variable is arg 0. The pattern/flags are resolved from ``PAT``'s
      definition (``_resolve_compiled_pattern``), which is the dominant shape in real
      parsers (``VERSION_RE.fullmatch(...)``, ``TOKEN_RE.fullmatch(...)``).
    """
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method not in ("match", "search", "fullmatch"):
            continue
        base = node.func.value
        # re.match(<literal>, name) — pattern literal, variable is arg 1.
        if isinstance(base, ast.Name) and base.id == "re" and len(node.args) >= 2:
            arg1 = node.args[1]
            arg0 = node.args[0]
            if (isinstance(arg1, ast.Name) and arg1.id == name
                    and isinstance(arg0, ast.Constant)
                    and isinstance(arg0.value, (str, bytes))):
                return arg0.value, 0, isinstance(arg0.value, bytes), None
        # PAT.method(name) — compiled-pattern constant, variable is arg 0.
        elif isinstance(base, ast.Name) and base.id != "re" and node.args:
            arg0 = node.args[0]
            if isinstance(arg0, ast.Name) and arg0.id == name and source_path:
                spec = _resolve_compiled_pattern(base.id, source_path)
                if spec is not None:
                    pattern, flags, is_bytes = spec
                    return pattern, flags, is_bytes, base.id
    return None


def _generate_from_regex(
    pattern: Any, *, flags: int = 0, is_bytes: bool = False, prefer_ascii: bool = True,
) -> Any:
    """Generate a value matching *pattern* using hypothesis if available.

    A ``.fullmatch`` witness (the whole value matches) also satisfies ``.match`` (matches
    at the start) and ``.search`` (a match exists), so one ``fullmatch=True`` draw serves
    every method — a possibly stronger value than needed, never a wrong one.

    ``\\d`` / ``\\w`` / ``\\s`` are *Unicode-aware* on ``str`` patterns, so a plain draw for
    ``[a-fA-F\\d]{32}`` can return Devanagari digits: a legal match but a poor witness. For
    str patterns we first try under ``re.ASCII`` and fall back only if that draws nothing.
    Bytes patterns are ASCII by construction, so they skip that step.
    """
    import re as _re

    try:
        from hypothesis import strategies as st
    except Exception:
        return None

    if not is_bytes and prefer_ascii:
        try:
            # re.ASCII and re.UNICODE are mutually exclusive; a compiled str pattern carries
            # UNICODE by default, so strip it before forcing ASCII.
            ascii_pat = _re.compile(pattern, (flags & ~_re.UNICODE) | _re.ASCII)
            value = st.from_regex(ascii_pat, fullmatch=True).example()
            if value is not None and value.isascii():
                return value
        except Exception:
            pass   # pattern rejects re.ASCII, or draws nothing — fall through
    try:
        compiled = _re.compile(pattern, flags)
        return st.from_regex(compiled, fullmatch=True).example()
    except Exception:
        return None


def _extract_str_method_values(
    name: str,
    expr: ast.expr,
    facts: Any = None,
) -> str | None:
    """Synthesise a string satisfying startswith/endswith constraints on *name*.

    For ``x.startswith("pre") and x.endswith("suf")``, returns ``"presuf"``.
    """
    prefix: str | None = None
    suffix: str | None = None
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == name):
            continue
        if func.attr == "startswith" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                prefix = arg.value
        if func.attr == "endswith" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                suffix = arg.value
    # Prefer facts whitelist values that satisfy prefix/suffix constraints.
    if facts is not None and (prefix is not None or suffix is not None):
        vf = getattr(facts, "facts", {}).get(name)
        if vf is not None:
            eq = getattr(vf, "eq", None)
            neq = getattr(vf, "neq", None) or set()
            if eq:
                for candidate in eq:
                    if not isinstance(candidate, str):
                        continue
                    if candidate in neq:
                        continue
                    ok = True
                    if prefix is not None and not candidate.startswith(prefix):
                        ok = False
                    if suffix is not None and not candidate.endswith(suffix):
                        ok = False
                    if ok:
                        return candidate

    if prefix is not None and suffix is not None:
        synth = prefix + suffix
        # Check against neq exclusion.
        if facts is not None:
            vf = getattr(facts, "facts", {}).get(name)
            neq = getattr(vf, "neq", None) if vf else None
            if neq and synth in neq:
                # Try adding filler.
                synth = prefix + "X" + suffix
        return synth
    if prefix is not None:
        return prefix + "x"
    if suffix is not None:
        return "x" + suffix
    return None


def _is_negated_isinstance(name: str, expr: ast.expr) -> str | None:
    """Detect ``not isinstance(name, T)`` and return the type string T.

    Returns ``None`` if no negated isinstance is found for *name*.
    """
    for node in ast.walk(expr):
        if not isinstance(node, ast.UnaryOp):
            continue
        if not isinstance(node.op, ast.Not):
            continue
        inner = node.operand
        if isinstance(inner, ast.Call):
            func = inner.func
            if isinstance(func, ast.Name) and func.id == "isinstance" and len(inner.args) >= 2:
                arg0 = inner.args[0]
                if isinstance(arg0, ast.Name) and arg0.id == name:
                    arg1 = inner.args[1]
                    if isinstance(arg1, ast.Name):
                        return arg1.id
    return None


def _construct_for_isinstance_type(type_str: str) -> Any:
    """Construct a real instance for a known type string.

    For stdlib AST types: build minimal valid nodes.
    For unknown types: return None (caller falls back to SimpleNamespace).
    """
    import ast as _ast
    import types as _types

    _CONSTRUCTORS: dict[str, Any] = {
        "ast.Call": lambda: _ast.Call(func=_ast.Name(id="f", ctx=_ast.Load()), args=[], keywords=[]),
        "ast.Name": lambda: _ast.Name(id="x", ctx=_ast.Load()),
        "ast.Constant": lambda: _ast.Constant(value=0),
        "ast.Attribute": lambda: _ast.Attribute(value=_ast.Name(id="x", ctx=_ast.Load()), attr="y", ctx=_ast.Load()),
        "ast.FunctionDef": lambda: _ast.FunctionDef(
            name="f", args=_ast.arguments(posonlyargs=[], args=[], vararg=None,
            kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=[_ast.Pass()], decorator_list=[], returns=None),
        "ast.AsyncFunctionDef": lambda: _ast.AsyncFunctionDef(
            name="f", args=_ast.arguments(posonlyargs=[], args=[], vararg=None,
            kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=[_ast.Pass()], decorator_list=[], returns=None),
        "ast.ClassDef": lambda: _ast.ClassDef(name="C", bases=[], keywords=[], body=[_ast.Pass()], decorator_list=[]),
        "ast.If": lambda: _ast.If(test=_ast.Constant(value=True), body=[_ast.Pass()], orelse=[]),
        "ast.Compare": lambda: _ast.Compare(left=_ast.Constant(value=0), ops=[_ast.Eq()], comparators=[_ast.Constant(value=0)]),
        "ast.BoolOp": lambda: _ast.BoolOp(op=_ast.And(), values=[_ast.Constant(value=True)]),
        "ast.UnaryOp": lambda: _ast.UnaryOp(op=_ast.Not(), operand=_ast.Constant(value=True)),
        "ast.Return": lambda: _ast.Return(value=_ast.Constant(value=None)),
        "ast.Assign": lambda: _ast.Assign(targets=[_ast.Name(id="x", ctx=_ast.Store())], value=_ast.Constant(value=0)),
        "ast.Subscript": lambda: _ast.Subscript(value=_ast.Name(id="x", ctx=_ast.Load()), slice=_ast.Constant(value=0), ctx=_ast.Load()),
        "ast.Module": lambda: _ast.Module(body=[], type_ignores=[]),
    }
    # Also handle without "ast." prefix (isinstance resolves bare names).
    for key in list(_CONSTRUCTORS):
        bare = key.split(".")[-1]
        if bare not in _CONSTRUCTORS:
            _CONSTRUCTORS[bare] = _CONSTRUCTORS[key]
    # Handle _ast.XXX prefixed names (from `import ast as _ast`)
    for key in list(_CONSTRUCTORS):
        if key.startswith("ast."):
            alt = "_ast." + key[4:]
            if alt not in _CONSTRUCTORS:
                _CONSTRUCTORS[alt] = _CONSTRUCTORS[key]

    constructor = _CONSTRUCTORS.get(type_str)
    if constructor:
        try:
            return constructor()
        except Exception:
            return None
    return None


# Map of method names to required argument types for cross-variable propagation.
_METHOD_ARG_TYPES: dict[str, str] = {
    "startswith": "str",
    "endswith": "str",
    "find": "str",
    "rfind": "str",
    "index": "str",
    "rindex": "str",
    "count": "str",
    "replace": "str",
    "split": "str",
    "join": "str",
    "append": "Any",
    "extend": "list",
    "insert": "Any",
    "remove": "Any",
    "get": "Any",
    "setdefault": "Any",
    "update": "dict",
}


def _propagate_arg_types(
    var_names: list[str],
    test_expr: ast.expr,
) -> dict[str, str]:
    """Infer types for variables used as arguments to known methods.

    For ``s.startswith(prefix)``, infer that ``prefix`` is ``str`` because
    ``str.startswith`` takes a ``str`` argument.
    """
    inferred: dict[str, str] = {}
    for node in ast.walk(test_expr):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        method_name = func.attr
        arg_type = _METHOD_ARG_TYPES.get(method_name)
        if arg_type is None or arg_type == "Any":
            continue
        # Check each positional arg — if it's a free variable, propagate type.
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id in var_names and arg.id not in inferred:
                inferred[arg.id] = arg_type
    return inferred


_NEGATED_ISINSTANCE_ALTERNATIVES: dict[str, Any] = {
    "str": 0,
    "int": "",
    "float": "",
    "bool": 0,
    "bytes": 0,
    "list": 0,
    "dict": 0,
    "set": 0,
    "tuple": 0,
}


def _default_value_for_type(type_str: str) -> Any:
    """Return a sensible default value for a type string."""
    defaults: dict[str, Any] = {
        "int": 0,
        "float": 0.0,
        "str": "",
        "bytes": b"",
        "bool": True,
        "list": [],
        "dict": {},
        "set": set(),
        "tuple": (),
        "object": None,
        "type": type,
        "Any": 0,
    }
    return defaults.get(type_str, None)


def _ensure_truthy(value: Any, type_str: str) -> Any:
    """Return a truthy version of *value*.

    When a variable appears as a bare name in ``BoolOp(And)``, the runtime
    requires it to be truthy for the ``and`` chain to continue.  Default
    values like ``0``, ``""``, ``[]`` are falsy — this function upgrades
    them.
    """
    if value:
        return value
    _TRUTHY: dict[str, Any] = {
        "int": 1,
        "float": 1.0,
        "str": "x",
        "bytes": b"x",
        "list": [0],
        "dict": {"k": 0},
        "set": {0},
        "tuple": (0,),
        "Any": 1,
    }
    return _TRUTHY.get(type_str, value if value else 1)


def _construct_for_isinstance_type_with_imports(
    type_str: str,
    source_path: str | None = None,
) -> Any:
    """Construct a concrete instance for a non-primitive isinstance type.

    Attempts in order:
    1. Known stdlib AST constructors (via the first _construct_for_isinstance_type)
    2. Cross-module resolution: parse imports from source_path, resolve the
       module, import the class, and attempt to instantiate it
    3. Fallback to SimpleNamespace
    """
    import types as _types

    # Try the AST/known-type constructors first
    result = _construct_for_isinstance_type(type_str)
    if result is not None:
        return result

    # Try cross-module resolution for qualified names like "nodes.Expr"
    if "." in type_str and source_path:
        try:
            result = _resolve_and_construct_cross_module(type_str, source_path)
            if result is not None:
                return result
        except Exception:
            pass

    # Try direct-import resolution for bare names like "Formatter"
    # These are imported via `from bs4.formatter import Formatter`
    if source_path and "." not in type_str:
        try:
            result = _resolve_and_construct_bare_import(type_str, source_path)
            if result is not None:
                return result
        except Exception:
            pass

    # Fallback: generic object for unknown types
    return _types.SimpleNamespace()


def _resolve_and_construct_bare_import(
    class_name: str,
    source_path: str,
) -> Any:
    """Resolve a bare class name and try to construct an instance.

    Handles both imported classes and classes defined in the same file:
    1. Check imports: ``from bs4.formatter import Formatter``
    2. Check same-file class defs: ``class NavigableString(str, PageElement)``
    3. Import the actual module, get the class, try to instantiate
    """
    from tharkun.solve.frontend.import_resolver import parse_imports
    imports = parse_imports(source_path)

    module_path = imports.get(class_name)

    # If not found in imports, try the source file itself (same-file class)
    if module_path is None:
        module_path = os.path.realpath(source_path)

    # Try to import the module and get the class -- through the core's one
    # loader (#0607), so ``no_targets()`` turns this into a static-only answer.
    from tharkun.solve.loader import import_dotted, load_from_file

    dotted = _file_path_to_module_name(module_path)
    cls = None

    try:
        mod = import_dotted(dotted)
        cls = getattr(mod, class_name, None) if mod is not None else None
    except Exception:
        pass

    if cls is None:
        try:
            mod = load_from_file(module_path, f"_tharkun_{class_name}")
            cls = getattr(mod, class_name, None) if mod is not None else None
        except Exception:
            pass

    if cls is None or not isinstance(cls, type):
        return None

    # Try to construct — multiple strategies
    try:
        return cls()
    except (TypeError, Exception):
        pass

    # Try with common single-arg defaults (for str/bytes subclasses)
    for default_arg in ["test", b"test", 0, True, []]:
        try:
            obj = cls(default_arg)
            if isinstance(obj, cls):
                return obj
        except (TypeError, Exception):
            continue

    # Try constructor.py
    try:
        from tharkun.solve.frontend.constructor import construct_instance
        obj = construct_instance(module_path, class_name)
        if isinstance(obj, cls):
            return obj
    except Exception:
        pass

    # Try concrete subclass
    try:
        mod_obj = import_dotted(dotted)
        if mod_obj:
            for attr_name in dir(mod_obj):
                sub = getattr(mod_obj, attr_name, None)
                if isinstance(sub, type) and issubclass(sub, cls) and sub is not cls:
                    try:
                        return sub()
                    except (TypeError, Exception):
                        continue
    except Exception:
        pass

    # Dynamic subclass
    try:
        dynamic = type(f"_Tharkun{class_name}", (cls,), {})
        return dynamic.__new__(dynamic)
    except (TypeError, Exception):
        pass

    return None


def _file_path_to_module_name(file_path: str) -> str | None:
    """Infer a dotted module name from a file path.

    ``/site-packages/jinja2/nodes.py`` → ``jinja2.nodes``
    ``/site-packages/jinja2/__init__.py`` → ``jinja2``
    """
    import sys
    rp = os.path.realpath(file_path)
    # Try each sys.path entry as a potential root
    for sp in sys.path:
        if not sp:
            continue
        sp_real = os.path.realpath(sp)
        if rp.startswith(sp_real + os.sep):
            rel = rp[len(sp_real) + 1:]
            # Remove .py extension
            if rel.endswith(".py"):
                rel = rel[:-3]
            # Convert path separators to dots
            parts = rel.split(os.sep)
            # Remove __init__ from the end
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            if parts:
                return ".".join(parts)
    return None


def _resolve_and_construct_cross_module(
    type_str: str,
    source_path: str,
) -> Any:
    """Resolve a module-qualified type and try to construct an instance.

    For ``nodes.Expr`` in a file that has ``from jinja2 import nodes``:
    1. Parse imports from source_path → find "nodes" maps to jinja2/nodes.py
    2. Import the actual module
    3. Get the class object
    4. Try to instantiate with no args, then with constructor analysis
    """
    parts = type_str.split(".")
    if len(parts) < 2:
        return None

    module_alias = parts[0]  # e.g. "nodes"
    class_name = parts[-1]    # e.g. "Expr"

    # Parse imports from the source file
    from tharkun.solve.frontend.import_resolver import parse_imports
    imports = parse_imports(source_path)

    # Find the module path for the alias
    module_path = imports.get(module_alias)
    if module_path is None:
        return None

    # Try to import the module and get the class -- through the core's one
    # loader (#0607).  Strategy 1: the package system (handles relative
    # imports); strategy 2: spec_from_file_location (standalone modules).
    from tharkun.solve.loader import import_dotted, load_from_file

    cls = None
    try:
        # Infer the dotted module name from the file path
        # e.g. /site-packages/jinja2/nodes.py → jinja2.nodes
        mod = import_dotted(_file_path_to_module_name(module_path))
        cls = getattr(mod, class_name, None) if mod is not None else None
    except Exception:
        pass

    if cls is None:
        try:
            mod = load_from_file(module_path, f"_tharkun_resolve_{module_alias}")
            cls = getattr(mod, class_name, None) if mod is not None else None
        except Exception:
            pass

    if cls is None or not isinstance(cls, type):
        return None

    # Try to construct an instance
    # Strategy 1: no-arg constructor
    try:
        return cls()
    except (TypeError, Exception):
        pass

    # Strategy 2: use constructor.py to analyze __init__ and synthesize args
    try:
        from tharkun.solve.frontend.constructor import construct_instance
        obj = construct_instance(module_path, class_name)
        if isinstance(obj, cls):
            return obj
    except Exception:
        pass

    # Strategy 3: find a concrete subclass and instantiate that
    try:
        for attr_name in dir(mod):
            sub = getattr(mod, attr_name, None)
            if (isinstance(sub, type) and issubclass(sub, cls)
                    and sub is not cls):
                try:
                    return sub()
                except (TypeError, Exception):
                    continue
    except Exception:
        pass

    # Strategy 4: create a dynamic subclass that passes isinstance
    try:
        dynamic = type(f"_Tharkun{class_name}", (cls,), {})
        return dynamic.__new__(dynamic)
    except (TypeError, Exception):
        pass

    return None


def _is_negated_bare_name(name: str, expr: ast.expr) -> bool:
    """Return True if *name* appears as ``not name`` in the expression."""
    # Direct: not name
    if (isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not)
            and isinstance(expr.operand, ast.Name) and expr.operand.id == name):
        return True
    # Inside BoolOp: ... and not name and ...
    for node in ast.walk(expr):
        if not isinstance(node, ast.BoolOp):
            continue
        for value in node.values:
            if (isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not)
                    and isinstance(value.operand, ast.Name) and value.operand.id == name):
                return True
    return False


def _ensure_falsy(value: Any, type_str: str) -> Any:
    """Return a falsy version of *value* for the given type."""
    if not value:
        return value  # Already falsy
    _FALSY: dict[str, Any] = {
        "int": 0,
        "float": 0.0,
        "str": "",
        "bytes": b"",
        "list": [],
        "dict": {},
        "set": set(),
        "tuple": (),
        "bool": False,
        "Any": 0,
    }
    return _FALSY.get(type_str, False)


def _name_in_iterator_context(name: str, expr: ast.expr) -> bool:
    """Return True if *name* is used in a for-loop iterator expression.

    For-loop iterators must be non-empty for the body to execute.
    Detects patterns like:
    - ``list(x.items())`` — x needs to be a non-empty dict
    - ``x.keys()`` — x needs to be non-empty
    - ``enumerate(x)`` — x needs to be non-empty
    """
    for node in ast.walk(expr):
        # x.items(), x.keys(), x.values()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == name:
                return True
            # list(x.items()) — x is nested inside a call
            for arg in node.args:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
                    if isinstance(arg.func.value, ast.Name) and arg.func.value.id == name:
                        return True
        # Direct: for item in x
        if isinstance(node, ast.Name) and node.id == name:
            # Check if this Name is the whole iterator expression
            if node is expr:
                return True
    return False


def _name_in_boolop_and(name: str, expr: ast.expr) -> bool:
    """Return True if *name* appears as a bare ``Name`` node directly
    inside a ``BoolOp(And)`` in *expr*.

    This detects the truthiness requirement: ``x and ...`` needs ``x`` truthy.
    """
    for node in ast.walk(expr):
        if not isinstance(node, ast.BoolOp):
            continue
        if not isinstance(node.op, ast.And):
            continue
        for value in node.values:
            if isinstance(value, ast.Name) and value.id == name:
                return True
    return False


def _strategy_hint_for_type(type_str: str, values: list[Any]) -> str | None:
    """Return a Hypothesis strategy hint string."""
    if type_str == "int":
        if values:
            nums = [v for v in values if isinstance(v, int)]
            if nums:
                return f"st.integers(min_value={min(nums) - 10}, max_value={max(nums) + 10})"
        return "st.integers()"
    if type_str == "float":
        return "st.floats(allow_nan=False, allow_infinity=False)"
    if type_str == "str":
        return "st.text(max_size=64)"
    if type_str == "bytes":
        return "st.binary(max_size=64)"
    if type_str == "bool":
        return "st.booleans()"
    if type_str == "list":
        return "st.lists(st.integers(), max_size=10)"
    if type_str == "dict":
        return "st.dictionaries(st.text(max_size=8), st.integers(), max_size=6)"
    # Common ABCs / protocols.
    _ABC_STRATEGIES: dict[str, str] = {
        "PathLike": "st.builds(pathlib.PurePosixPath, st.text(min_size=1, max_size=32))",
        "os.PathLike": "st.builds(pathlib.PurePosixPath, st.text(min_size=1, max_size=32))",
        "Sequence": "st.lists(st.integers(), max_size=10)",
        "Iterable": "st.lists(st.integers(), max_size=10)",
        "Iterator": "st.lists(st.integers(), max_size=10).map(iter)",
        "Mapping": "st.dictionaries(st.text(max_size=8), st.integers(), max_size=6)",
        "MutableMapping": "st.dictionaries(st.text(max_size=8), st.integers(), max_size=6)",
        "Callable": "st.just(lambda: None)",
        "Collection": "st.lists(st.integers(), max_size=10)",
        "Sized": "st.lists(st.integers(), max_size=10)",
        "object": "st.just(object())",
    }
    return _ABC_STRATEGIES.get(type_str)


def _try_intersect_int_constraints(
    name: str,
    expr: ast.expr,
    facts: Any = None,
) -> list[int]:
    """For compound int guards like ``x > 0 and x < 10 and x != 5``,
    compute the intersection of all int constraints on *name*.

    When *facts* is provided, bounds and exclusions from external analysis
    are incorporated.
    """
    # Collect all constraints as (op, value).
    constraints: list[tuple[str, int]] = []

    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        for op, comparator in zip(node.ops, node.comparators):
            comp_val, comp_ok = _try_extract_constant(comparator)
            left_val, left_ok = _try_extract_constant(left)
            if isinstance(left, ast.Name) and left.id == name and comp_ok and isinstance(comp_val, int):
                constraints.append((type(op).__name__, comp_val))
            elif isinstance(comparator, ast.Name) and comparator.id == name and left_ok and isinstance(left_val, int):
                flip = {"Gt": "Lt", "Lt": "Gt", "GtE": "LtE", "LtE": "GtE", "Eq": "Eq", "NotEq": "NotEq"}
                op_name = flip.get(type(op).__name__, type(op).__name__)
                constraints.append((op_name, left_val))
            left = comparator

    if not constraints:
        return []

    # Compute range — initialize from external facts if available.
    from tharkun.solve.config import config
    lo, hi = config.int_range_default
    if facts is not None:
        vf = getattr(facts, "facts", {}).get(name)
        if vf is not None:
            # If facts has a whitelist, return only those values that satisfy
            # the constraints (short-circuit the range computation).
            feq = getattr(vf, "eq", None)
            if feq:
                candidates = [int(v) for v in feq if isinstance(v, (int, float))]
                if candidates:
                    # Filter by the constraints we already collected.
                    valid = []
                    for c in candidates:
                        ok = True
                        for op_name, v in constraints:
                            if op_name == "Gt" and not (c > v):
                                ok = False
                            elif op_name == "GtE" and not (c >= v):
                                ok = False
                            elif op_name == "Lt" and not (c < v):
                                ok = False
                            elif op_name == "LtE" and not (c <= v):
                                ok = False
                            elif op_name == "Eq" and c != v:
                                ok = False
                            elif op_name == "NotEq" and c == v:
                                ok = False
                        if ok:
                            valid.append(c)
                    if valid:
                        return valid[:5]
    # #0611 (plan #0600): the interval intersection is the fact carrier's own
    # arithmetic (``fact_env.refine``), shared with ``guard_facts`` -- one
    # implementation of "lo/hi/excluded under a conjunction of int atoms".
    from tharkun.solve.fact_env import FactEnv as _FE, VarFacts as _VF, refine as _refine

    env = _FE(facts={name: _VF(lo=int(lo), hi=int(hi))})
    if facts is not None:
        vf = getattr(facts, "facts", {}).get(name)
        if vf is not None:
            flo = getattr(vf, "lo", None)
            fhi = getattr(vf, "hi", None)
            fneq = getattr(vf, "neq", None)
            if flo is not None:
                env = _refine(env, name, ">=", int(flo))
            if fhi is not None:
                env = _refine(env, name, "<=", int(fhi))
            for v in (fneq or ()):
                if isinstance(v, (int, float)):
                    env = _refine(env, name, "!=", int(v))
    _OPS = {"Gt": ">", "GtE": ">=", "Lt": "<", "LtE": "<=", "Eq": "==", "NotEq": "!="}
    for op_name, v in constraints:
        env = _refine(env, name, _OPS[op_name], int(v))
    if not env.possible:
        return []
    rvf = env.facts[name]
    lo, hi = int(rvf.lo), int(rvf.hi)
    excluded = {int(v) for v in rvf.neq if isinstance(v, (int, float))}

    # Return up to 5 values from the valid range.
    candidates = [i for i in range(lo, min(hi + 1, lo + 20)) if i not in excluded]
    return candidates[:5]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _solve_single_expr(
    test_expr: ast.expr,
    rp: str,
    lineno: int,
    callable_info: CallableInfo | None,
    builtins_names: set[str],
    facts: Any = None,
) -> tuple[list[VarConstraint], dict[str, Any], bool]:
    """Solve a single (non-Or) expression. Returns (variables, satisfying, all_direct).

    When *facts* is a ``FactEnv``, external constraints narrow type inference
    and value generation.
    """
    free = _free_names(test_expr)
    var_names = sorted(free - builtins_names)

    # Filter out module references used only in isinstance type args.
    # e.g. isinstance(node, nodes.Expr) — "nodes" is a module, not an input.
    isinstance_module_refs = _find_isinstance_module_refs(test_expr)
    var_names = [n for n in var_names if n not in isinstance_module_refs]

    # Resolve scope variables: names that aren't parameters but are defined
    # as module-level constants or earlier in the function body.
    scope_resolved = _resolve_scope_variables(var_names, rp, lineno)
    # For membership tests `x in RESOLVED_COLLECTION`, extract member values.
    scope_membership_values: dict[str, list[Any]] = {}
    if scope_resolved:
        var_names = [n for n in var_names if n not in scope_resolved]
        scope_membership_values = _extract_membership_from_scope(test_expr, scope_resolved)

    # If ALL free variables were scope-resolved and their values are truthy,
    # the branch is always reachable (e.g. `for c in COMMENTCHARS` where
    # COMMENTCHARS="#;"). Return immediately with empty inputs.
    if not var_names and scope_resolved:
        all_truthy = all(bool(v) for v in scope_resolved.values())
        if all_truthy:
            return [], {}, True, scope_resolved

    variables: list[VarConstraint] = []
    satisfying: dict[str, Any] = {}
    all_direct = True

    # Fix D: cross-variable type propagation.
    # If s.startswith(prefix), infer prefix is str because startswith takes str.
    propagated_types = _propagate_arg_types(var_names, test_expr)

    for name in var_names:
        inferred_type, source_kind = infer_var_type(
            name, test_expr, rp, lineno, callable_info, facts=facts
        )
        # Apply propagated type if inference fell to fallback.
        if source_kind == "fallback" and name in propagated_types:
            inferred_type = propagated_types[name]
            source_kind = "arg_propagation"

        # For bare truthiness guards (var is the entire expression or in BoolOp),
        # look at the branch body to infer what type the variable should be.
        if source_kind == "fallback" and _is_bare_truthiness(name, test_expr):
            body_type = _infer_type_from_branch_body(name, rp, lineno)
            if body_type:
                inferred_type = body_type
                source_kind = "body_usage"

        if source_kind == "self_attr":
            # Check if this is a bare attribute chain (self.a.b) — if so,
            # we can build a nested SimpleNamespace instead of bailing.
            attr_val = _extract_attr_truthiness_value(name, test_expr)
            if attr_val is not None:
                variables.append(VarConstraint(
                    name=name, inferred_type="object",
                    inference_source="attr_truthiness",
                    concrete_values=[attr_val],
                ))
                satisfying[name] = attr_val
                continue
            # Check attr_comparison (self.state == "connected") and
            # identity checks (self.attr is not None / self.attr is None)
            # and comprehensive self-building (collections, len, methods)
            attr_comp_vals = _extract_attr_comparison_values(name, test_expr)
            if not attr_comp_vals:
                # Field-resolver enhanced extraction: handles >, <, >=, <=, != on
                # nested fields when we know the enclosing class.
                _self_class = ""
                if callable_info:
                    _self_class = (callable_info.qualname or "").split(".")[0]
                if not _self_class:
                    # Fallback: scan AST for the enclosing class at this line
                    try:
                        from tharkun.solve.frontend.field_resolver import get_enclosing_class
                        _self_class = get_enclosing_class(rp, lineno) or ""
                    except Exception:
                        pass
                if _self_class:
                    try:
                        attr_comp_vals = _extract_field_resolved_attr_values(
                            name, test_expr, _self_class, rp
                        )
                    except Exception:
                        pass
            if not attr_comp_vals:
                # Comprehensive self-builder: merges identity checks, collection
                # membership, len constraints, and method calls into one object
                attr_comp_vals = _build_self_from_guard_constraints(name, test_expr)
            if not attr_comp_vals:
                attr_comp_vals = _extract_self_identity_values(name, test_expr)
            if attr_comp_vals:
                variables.append(VarConstraint(
                    name=name, inferred_type="object",
                    inference_source="attr_comparison",
                    concrete_values=attr_comp_vals,
                ))
                satisfying[name] = attr_comp_vals[0]
                continue
            # Fallback: can't solve self — skip
            variables.append(VarConstraint(
                name=name, inferred_type=inferred_type, inference_source=source_kind,
            ))
            all_direct = False
            continue

        # hasattr: produce SimpleNamespace with the required attribute.
        if source_kind == "hasattr":
            import types as _types
            attr_name = _extract_hasattr_attr_name(name, test_expr)
            if attr_name:
                ns = _types.SimpleNamespace(**{attr_name: True})
                variables.append(VarConstraint(
                    name=name, inferred_type="object",
                    inference_source="hasattr",
                    concrete_values=[ns],
                    strategy_hint=None,
                ))
                satisfying[name] = ns
                continue

        # attr_truthiness: build nested SimpleNamespace for bare attribute chains
        if source_kind in ("attr_truthiness", "attr_truthiness_negated"):
            attr_val = _extract_attr_truthiness_value(name, test_expr)
            if attr_val is not None:
                variables.append(VarConstraint(
                    name=name, inferred_type="object",
                    inference_source=source_kind,
                    concrete_values=[attr_val],
                ))
                satisfying[name] = attr_val
                continue

        neg_type = _is_negated_isinstance(name, test_expr)
        if neg_type is not None:
            alt = _NEGATED_ISINSTANCE_ALTERNATIVES.get(neg_type)
            if alt is not None:
                variables.append(VarConstraint(
                    name=name, inferred_type=inferred_type,
                    inference_source="negated_isinstance",
                    concrete_values=[alt],
                    strategy_hint=_strategy_hint_for_type(inferred_type, [alt]),
                ))
                satisfying[name] = alt
                continue

        direct_values = _extract_comparison_values(name, test_expr, facts=facts)

        # Bitmask flag test (`if flags & MASK:`) — no Compare node for the extractor
        # above to read, but the mask itself is the answer.
        if not direct_values:
            bit_values = _extract_bitmask_values(name, test_expr, rp)
            if bit_values:
                direct_values = bit_values
                if inferred_type in ("Any", "object"):
                    inferred_type = "int"
                    source_kind = "bitwise_operand"

        # If facts provides a whitelist and no direct values were extracted,
        # use the whitelist values directly.
        if not direct_values and facts is not None:
            vf = getattr(facts, "facts", {}).get(name)
            if vf is not None:
                eq = getattr(vf, "eq", None)
                if eq:
                    direct_values = list(eq)[:5]

        # Inject membership values from scope-resolved collections
        if not direct_values and name in scope_membership_values:
            direct_values = scope_membership_values[name]
            if inferred_type == "Any" and direct_values:
                inferred_type = type(direct_values[0]).__name__
                source_kind = "scope_membership"

        # Attribute comparison: obj.attr == literal / obj.attr in {..}
        if source_kind == "attr_comparison" and not direct_values:
            attr_vals = _extract_attr_comparison_values(name, test_expr)
            _BASIC_BUILTINS = frozenset({
                "int", "float", "str", "bytes", "bool",
                "list", "dict", "set", "tuple", "None",
            })
            if not attr_vals and inferred_type not in ("Any", "object") and inferred_type not in _BASIC_BUILTINS:
                # Field-resolver enhanced: handles >, <, >=, <= on typed params
                # e.g. config: Config, guard: config.timeout > 60
                try:
                    attr_vals = _extract_field_resolved_attr_values(
                        name, test_expr, inferred_type, rp
                    )
                except Exception:
                    pass
            if attr_vals:
                direct_values = attr_vals

        if inferred_type == "int" and not direct_values:
            direct_values = _try_intersect_int_constraints(name, test_expr, facts=facts)
        elif inferred_type == "int" and len(direct_values) == 1:
            intersected = _try_intersect_int_constraints(name, test_expr, facts=facts)
            if intersected:
                direct_values = intersected

        # Byte-input gates: data[k] / int.from_bytes(data[i:j]) → synthesize bytes.
        # Set when the byte synthesis produced candidates but verification proved every one
        # fails the guard — the guard is unsatisfiable, so the type-default placeholder must
        # NOT be injected below (that would report a non-satisfying b'' as type_inferred).
        byte_unsat = False
        if inferred_type == "bytes" and not direct_values:
            byte_vals = _extract_byte_input_values(name, test_expr)
            single_var = (_free_names(test_expr) - set(builtins_names)) <= {name}
            if byte_vals:
                # The synthesis merges every comparison atom across the whole guard into
                # one last-write-wins buffer, ignoring and/or structure — so a produced
                # value need not actually satisfy the guard (`data[0]==1 and
                # int.from_bytes(data[0:2])==2` is unsatisfiable, yet it emits b"\x00\x02").
                # When this is the guard's only variable, verify each value against the FULL
                # guard and keep only those that hold; if none hold the guard is
                # unsatisfiable (byte_unsat). A MULTI-variable byte guard cannot be verified
                # here (the other variables are unresolved), so `all_direct` is cleared to
                # force the full-guard verify at the return.
                if single_var:
                    guard_txt = ast.unparse(test_expr)
                    verified = [v for v in byte_vals
                                if _verify_guard_satisfaction(guard_txt, {name: v})]
                    byte_unsat = not verified
                    byte_vals = verified
                if byte_vals:
                    direct_values = byte_vals
                    source_kind = "byte_input"
                    if not single_var:
                        all_direct = False
            elif single_var and _byte_atoms_unsatisfiable(name, test_expr):
                # No value could be synthesized because every atom is out of range (e.g. a
                # signed decode compared to a value outside the signed domain) — the guard
                # is unsatisfiable, so report "none" rather than a placeholder.
                byte_unsat = True

        strategy_hint = _strategy_hint_for_type(inferred_type, direct_values)

        if inferred_type == "str" and not direct_values:
            synth = _extract_str_method_values(name, test_expr, facts=facts)
            if synth is not None:
                direct_values = [synth]

        # Try special patterns: any() generators, modular membership, callable()
        if not direct_values:
            special = _extract_special_pattern_values(name, test_expr)
            if special:
                direct_values = special
                if inferred_type == "Any":
                    # Infer type from the produced value
                    inferred_type = type(special[0]).__name__
                    source_kind = "special_pattern"

        # Try regex pattern inversion — an inline re.match(literal, x) OR a compiled
        # module-level pattern used via a bound method (PAT.fullmatch(x)); str or bytes.
        if inferred_type in ("str", "bytes", "Any") and not direct_values:
            spec = _extract_regex_pattern(name, test_expr, rp)
            if spec is not None:
                pattern, flags, is_bytes, const_name = spec
                regex_val = _generate_from_regex(pattern, flags=flags, is_bytes=is_bytes)
                # A drawn value is only accepted if it actually satisfies the guard — a bad
                # draw, a negated guard (`if not PAT.match(x)`, where a match is wrong), or a
                # same-named non-regex PAT is rejected here rather than returned as a false
                # `direct`. A compiled-pattern constant is bound into the verify env under
                # its own name so the guard evaluates.
                verify_env: dict[str, Any] = {name: regex_val}
                if const_name is not None:
                    import re as _re
                    try:
                        compiled = _re.compile(pattern, flags)
                        verify_env[const_name] = compiled
                        # Also make the constant resolvable to the RETURN-path verifier, which
                        # re-evaluates the whole guard from scope_resolved — otherwise the
                        # value verifies here but is downgraded there for an unbound name.
                        scope_resolved[const_name] = compiled
                    except Exception:
                        pass
                guard_src = _unparse(test_expr)
                if regex_val is not None and guard_src is not None \
                        and _verify_guard_satisfaction(guard_src, verify_env):
                    direct_values = [regex_val]
                    inferred_type = "bytes" if is_bytes else "str"
                    source_kind = "regex_pattern"
                    strategy_hint = f"st.from_regex({pattern!r}, fullmatch=True)"

        if direct_values:
            # When multiple candidates exist, prefer the one that actually
            # satisfies the full guard (handles compound guards like
            # math.isinf(x) and x < 0 where -1 fails but -inf works).
            best_val = direct_values[0]
            if len(direct_values) > 1:
                guard_src = _unparse(test_expr)
                if guard_src:
                    for candidate in direct_values:
                        if _verify_guard_satisfaction(guard_src, {name: candidate}):
                            best_val = candidate
                            break
            satisfying[name] = best_val
        else:
            len_min = _extract_len_constraint(name, test_expr)
            if len_min is not None and len_min >= 0:
                # len_min == 0 is a real answer (a `len(x) == 0` / `<= 0` guard is satisfied
                # by the EMPTY value), not "no constraint" — build the empty str/bytes/list.
                if inferred_type == "str" or inferred_type == "Any":
                    direct_values = ["x" * len_min]
                    if inferred_type == "Any":
                        inferred_type = "str"
                        source_kind = "len_constraint"
                elif inferred_type in ("list", "bytes"):
                    if inferred_type == "bytes":
                        direct_values = [b"\x00" * len_min]
                    else:
                        direct_values = [list(range(len_min))]
                if direct_values:
                    satisfying[name] = direct_values[0]

            if not direct_values and not byte_unsat:
                all_direct = False
                default = _default_value_for_type(inferred_type)
                if default is None and source_kind == "isinstance":
                    default = _construct_for_isinstance_type_with_imports(inferred_type, rp)
                # For bare truthiness with no default, provide a truthy value
                if default is None and _is_bare_truthiness(name, test_expr):
                    default = True
                if default is not None:
                    satisfying[name] = default
            elif byte_unsat:
                # The byte guard is unsatisfiable: no placeholder — leave `name` unbound so
                # `solve_condition` reports strategy "none" rather than a false type_inferred.
                all_direct = False

        # Negated truthiness: if the guard is `not x`, the value should be falsy.
        if name in satisfying and _is_negated_bare_name(name, test_expr):
            satisfying[name] = _ensure_falsy(satisfying[name], inferred_type)
            if not direct_values:
                direct_values = [satisfying[name]]

        # Truthiness: ensure value is truthy when used in truthiness context.
        # Covers: `if x:` (standalone), `if x and ...:` (in BoolOp And),
        # and for-loop iterators like `for k,v in list(x.items()):`
        needs_truthy = (
            (isinstance(test_expr, ast.Name) and test_expr.id == name)
            or _name_in_boolop_and(name, test_expr)
            or _name_in_iterator_context(name, test_expr)
        )
        if name in satisfying and needs_truthy:
            satisfying[name] = _ensure_truthy(satisfying[name], inferred_type)
            if not direct_values:
                direct_values = [satisfying[name]]

        variables.append(VarConstraint(
            name=name, inferred_type=inferred_type, inference_source=source_kind,
            concrete_values=direct_values, strategy_hint=strategy_hint,
        ))

    return variables, satisfying, all_direct, scope_resolved


def _solve_or_branches(
    test_expr: ast.BoolOp,
    guard_source: str,
    rp: str,
    lineno: int,
    callable_info: CallableInfo | None,
    facts: Any = None,
    desired_branch: str = "then",
) -> ConditionSolution:
    """Solve a top-level ``Or`` expression by solving each branch independently.

    Generates one ``satisfying_inputs`` entry per branch that can be solved,
    giving coverage of multiple ``or`` paths.
    """
    builtins_names = BUILTIN_NAMES

    all_variables: dict[str, VarConstraint] = {}
    satisfying_inputs: list[dict[str, Any]] = []
    any_direct = False

    for branch in test_expr.values:
        vars_b, sat_b, direct_b, _scope_b = _solve_single_expr(
            branch, rp, lineno, callable_info, builtins_names, facts=facts
        )
        for vc in vars_b:
            if vc.name not in all_variables or len(vc.concrete_values) > len(all_variables[vc.name].concrete_values):
                all_variables[vc.name] = vc
        if sat_b:
            satisfying_inputs.append(sat_b)
        if direct_b:
            any_direct = True

    variables = sorted(all_variables.values(), key=lambda v: v.name)

    from tharkun.solve.config import config as _cfg

    if satisfying_inputs:
        # Verification: check if any branch's inputs satisfy the full or-guard
        if not any_direct:
            for inp in satisfying_inputs:
                if _verify_guard_satisfaction(guard_source, inp):
                    any_direct = True
                    break
        strategy = "direct" if any_direct else "type_inferred"
        confidence = _cfg.confidence_or_branch_direct if any_direct else _cfg.confidence_or_branch_inferred
    else:
        strategy = "none"
        confidence = 0.0

    return ConditionSolution(
        guard_source=guard_source,
        variables=variables,
        satisfying_inputs=satisfying_inputs,
        strategy=strategy,
        confidence=confidence,
        desired_branch=desired_branch,
    )


def solve_condition(
    source_path: str,
    lineno: int,
    *,
    callable_info: CallableInfo | None = None,
    budget_ms: int = 2000,
    facts: Any = None,
    desired_branch: str = "then",
    var_types: dict[str, str] | None = None,
) -> ConditionSolution:
    """Solve a branch condition at the given line.

    1. Parse the source file, find the if/while/for at lineno
    2. Extract the test expression as AST
    3. Find free variables in the expression
    4. Infer types for each variable
    5. Attempt to find concrete satisfying values

    When *facts* is a ``FactEnv`` (from ``tharkun.minleaf.types``), external
    constraints narrow type inference and value generation.  This enables
    backward constraint propagation through call chains.

    *var_types* names the type of a free variable the caller already knows
    (``{"x": "int"}``; e.g. the kind of an SV draw).  When the inference below
    leaves the guard unsolved and every free name is a known integer, the
    native integer solver (:mod:`tharkun.solve.arith`: linear elimination,
    modular and polynomial guards) gets the guard; its witness is checked
    against the guard before it is returned (strategy ``arith``).
    """
    # Early return for statically infeasible constraints.
    if facts is not None and not getattr(facts, "possible", True):
        return ConditionSolution(
            guard_source="", strategy="infeasible", confidence=0.0,
            desired_branch=desired_branch,
            explanation=_build_explanation("infeasible", "", []),
        )

    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
    except (OSError, IOError):
        return ConditionSolution(
            guard_source="", strategy="none", confidence=0.0,
            desired_branch=desired_branch,
        )

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ConditionSolution(
            guard_source="", strategy="none", confidence=0.0,
            desired_branch=desired_branch,
        )

    test_expr = _find_branch_at_line(tree, lineno)
    if test_expr is None:
        return ConditionSolution(
            guard_source="", strategy="none", confidence=0.0,
            desired_branch=desired_branch,
        )

    guard_source = _unparse(test_expr) or ""

    # Literal True/False: always/never reachable — no inputs needed.
    if isinstance(test_expr, ast.Constant) and isinstance(test_expr.value, bool):
        lit_val = test_expr.value
        # For "else" branch, we want the constant to be False.
        target_val = (desired_branch == "then")
        satisfies = (lit_val == target_val)
        _lit_strategy = "direct" if satisfies else "none"
        return ConditionSolution(
            guard_source=guard_source,
            satisfying_inputs=[{}] if satisfies else [],
            strategy=_lit_strategy,
            confidence=1.0 if satisfies else 0.0,
            desired_branch=desired_branch,
            explanation=_build_explanation(_lit_strategy, guard_source, []),
        )

    # Normalize the guard: expand chained comparisons, push negation inward.
    # This is a canonical pre-processing step that makes all downstream
    # analysis more reliable.
    try:
        from tharkun.solve.normalize import normalize_guard
        test_expr = normalize_guard(test_expr)
        # Re-unparse after normalization (may change source representation)
        normalized_source = _unparse(test_expr)
        if normalized_source:
            guard_source = normalized_source
    except Exception:
        pass

    # user_guard_source is the normalized (but not negated) guard — always
    # returned to the caller regardless of which edge we're solving for.
    user_guard_source = guard_source

    # ---- False-branch negation (Phase C) ----
    # When the caller wants inputs that make the guard FALSE ("else" edge),
    # we negate the guard and solve for the negated form being True.
    # De Morgan pushdown is handled by normalize_guard, so
    #   not (x > 5 and y == "a")  →  x <= 5 or y != "a"
    # which _solve_single_expr handles natively.
    if desired_branch == "else":
        # Detect whether the guard is a for-loop iterator (not a boolean).
        # for-else means the iterator produced no items on the first evaluation.
        _is_for_loop = any(
            getattr(n, "lineno", None) == lineno and isinstance(n, ast.For)
            for n in ast.walk(tree)
        )
        if _is_for_loop:
            # Generate empty-iterator inputs.
            return ConditionSolution(
                guard_source=user_guard_source,
                satisfying_inputs=[{"__iter_empty__": True}],
                strategy="for_else",
                confidence=0.9,
                desired_branch=desired_branch,
                explanation=_build_explanation("for_else", user_guard_source, []),
            )

        # if/while "else": negate the guard and let the existing solver
        # produce inputs that satisfy the negation (= make original guard False).
        try:
            from tharkun.solve.normalize import normalize_guard as _ng
            negated = _ng(ast.UnaryOp(op=ast.Not(), operand=test_expr))
            test_expr = negated
            negated_src = _unparse(test_expr)
            if negated_src:
                guard_source = negated_src  # internal verification uses negated form
        except Exception:
            pass  # fallback: proceed with original guard (inputs may have wrong polarity)

    # Detect top-level `or` (BoolOp(Or)) and solve each branch independently.
    if isinstance(test_expr, ast.BoolOp) and isinstance(test_expr.op, ast.Or):
        sol = _solve_or_branches(
            test_expr, guard_source, rp, lineno, callable_info,
            facts=facts, desired_branch=desired_branch,
        )
        if sol.strategy not in ("direct", "infeasible"):
            arith = _arith_fallback(test_expr, user_guard_source, desired_branch,
                                    callable_info, facts,
                                    _with_annotated_ints(tree, lineno, var_types), sol.variables)
            if arith is not None:
                return arith
        # Always expose the original/normalized guard source to the caller.
        return ConditionSolution(
            guard_source=user_guard_source,
            variables=sol.variables,
            satisfying_inputs=sol.satisfying_inputs,
            strategy=sol.strategy,
            confidence=sol.confidence,
            desired_branch=sol.desired_branch,
            explanation=_build_explanation(sol.strategy, user_guard_source, sol.variables),
            input_constraints=_build_input_constraints(sol.variables),
        )

    builtins_names = BUILTIN_NAMES

    variables, satisfying, all_direct, scope_vars = _solve_single_expr(
        test_expr, rp, lineno, callable_info, builtins_names, facts=facts
    )

    satisfying_inputs = [satisfying] if satisfying else []

    from tharkun.solve.config import config as _cfg

    # Special case: all free variables were scope-resolved constants.
    # The branch is always reachable — return direct with empty inputs.
    if all_direct and not satisfying and not variables and scope_vars:
        return ConditionSolution(
            guard_source=user_guard_source,
            satisfying_inputs=[{}],
            strategy="direct",
            confidence=_cfg.confidence_direct,
            desired_branch=desired_branch,
            explanation=_build_explanation("direct", user_guard_source, []),
        )

    if all_direct and satisfying:
        strategy = "direct"
        confidence = _cfg.confidence_direct
    elif satisfying:
        # Verification pass: if the inferred inputs actually satisfy the
        # guard when evaluated, promote to "direct" strategy.
        # Merge scope-resolved variables into the eval context.
        verify_ctx = dict(satisfying)
        if scope_vars:
            verify_ctx.update(scope_vars)
        if _verify_guard_satisfaction(guard_source, verify_ctx):
            strategy = "direct"
            confidence = _cfg.confidence_verified
        else:
            # Joint satisfaction repair: for compound And guards, try
            # alternative values for each variable to find a combination
            # that satisfies all clauses together.
            repaired = _repair_compound_satisfaction(
                test_expr, guard_source, variables, satisfying, scope_vars or {}
            )
            if repaired:
                satisfying = repaired
                satisfying_inputs = [satisfying]
                strategy = "direct"
                confidence = _cfg.confidence_verified
            else:
                strategy = "type_inferred"
                confidence = _cfg.confidence_type_inferred
    else:
        strategy = "none"
        confidence = 0.0

    # Annotate strategy when external facts contributed.
    if facts is not None and strategy != "none" and strategy != "infeasible":
        has_facts_var = any(
            getattr(vc, "inference_source", "").startswith("facts_")
            for vc in variables
        )
        if has_facts_var:
            strategy = strategy + "+facts"

    if not strategy.startswith("direct") and strategy != "infeasible":
        arith = _arith_fallback(test_expr, user_guard_source, desired_branch,
                                callable_info, facts,
                                _with_annotated_ints(tree, lineno, var_types), variables)
        if arith is not None:
            return arith

    return ConditionSolution(
        guard_source=user_guard_source,
        variables=variables,
        satisfying_inputs=satisfying_inputs,
        strategy=strategy,
        confidence=confidence,
        desired_branch=desired_branch,
        explanation=_build_explanation(strategy, user_guard_source, variables),
        input_constraints=_build_input_constraints(variables),
    )


def solve_with_snapshot(
    source_path: str,
    lineno: int,
    snapshot: dict[str, Any],
    *,
    callable_info: CallableInfo | None = None,
    param_names: set[str] | None = None,
    desired_branch: str = "then",
) -> ConditionSolution:
    """Re-solve a guard condition using runtime snapshot data.

    Takes the snapshot of ``f_locals`` captured at the branch line and uses
    it to simplify the guard:

    1. Identify which free variables in the guard have known values from
       the snapshot (these are "resolved" — e.g. loop counters, config,
       computed intermediates). Function parameters are NOT resolved —
       they remain as unknowns to solve for.
    2. Evaluate sub-expressions that only reference resolved variables
       to determine their truthiness.
    3. For compound ``and`` guards: if some clauses are already True at
       runtime, focus solving on the unsatisfied clause.
    4. For any remaining unknowns: use the snapshot type info to improve
       type inference (e.g. if snapshot says ``threshold=100``, and guard
       is ``x > threshold``, substitute to get ``x > 100``).

    Args:
        source_path:    Path to the Python source file.
        lineno:         Line number of the branch guard.
        snapshot:       Dict of variable names to runtime values (from f_locals).
        callable_info:  Optional callable info for additional type context.

    Returns:
        A ``ConditionSolution`` with improved satisfying inputs.
    """
    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return ConditionSolution(guard_source="", strategy="none", confidence=0.0)

    test_expr = _find_branch_at_line(tree, lineno)
    if test_expr is None:
        return ConditionSolution(guard_source="", strategy="none", confidence=0.0)

    guard_source = _unparse(test_expr) or ""

    from tharkun.solve.config import config as _cfg

    # Identify which variables are free (need solving) vs resolved (in snapshot)
    builtins_names = BUILTIN_NAMES
    free = _free_names(test_expr)
    var_names = sorted(free - builtins_names)

    # Determine function parameter names — these should NOT be resolved
    # from snapshots, because they're the inputs we want to SOLVE for.
    if param_names is None:
        # Auto-detect from the enclosing function definition
        param_names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, "lineno", 0)
                end = getattr(node, "end_lineno", start + 1000)
                if start <= lineno <= end:
                    param_names = {
                        arg.arg for arg in node.args.args
                        if arg.arg not in ("self", "cls")
                    }
                    break

    # Partition into resolved (have snapshot value and NOT a parameter)
    # and unresolved (parameters or missing from snapshot)
    resolved: dict[str, Any] = {}
    unresolved: list[str] = []
    for name in var_names:
        if name in param_names:
            # Function parameter — solve for it, don't resolve from snapshot
            unresolved.append(name)
        elif name in snapshot:
            snap_val = snapshot[name]
            # _safe_snapshot wraps some values — unwrap scalars
            if isinstance(snap_val, (bool, int, float, str, type(None))):
                resolved[name] = snap_val
            elif isinstance(snap_val, dict) and "__type__" in snap_val:
                # Wrapped complex value — use it for type info but not substitution
                unresolved.append(name)
            else:
                resolved[name] = snap_val
        else:
            unresolved.append(name)

    # Try to simplify compound AND guards
    simplified_expr = _simplify_with_resolved(test_expr, resolved)

    # If simplification produced a constant True, the guard is already satisfied
    if isinstance(simplified_expr, ast.Constant) and simplified_expr.value is True:
        # All clauses are satisfied at runtime — the snapshot values ARE the inputs
        satisfying = {name: val for name, val in resolved.items() if name in var_names}
        variables = [
            VarConstraint(
                name=name, inferred_type=type(val).__name__,
                inference_source="snapshot", concrete_values=[val],
            )
            for name, val in satisfying.items()
        ]
        return ConditionSolution(
            guard_source=guard_source,
            variables=variables,
            satisfying_inputs=[satisfying] if satisfying else [],
            strategy="snapshot_resolved",
            confidence=_cfg.confidence_snapshot_resolved,
        )

    # Substitute resolved values as constants in the simplified expression.
    # This turns `x > threshold` with threshold=100 into `x > 100`.
    substituted_expr = _substitute_constants(simplified_expr, resolved)
    substituted_source = _unparse(substituted_expr)

    # Re-solve the substituted expression for remaining unknowns
    if substituted_source and substituted_source != guard_source:
        try:
            new_expr = ast.parse(substituted_source, mode="eval").body
            ast.fix_missing_locations(new_expr)
            variables, satisfying, all_direct, _scope = _solve_single_expr(
                new_expr, rp, lineno, callable_info, builtins_names
            )
            # Merge resolved values into satisfying inputs
            for name, val in resolved.items():
                if name in free and name not in satisfying:
                    satisfying[name] = val

            if satisfying:
                return ConditionSolution(
                    guard_source=guard_source,
                    variables=variables,
                    satisfying_inputs=[satisfying],
                    strategy="snapshot_simplified",
                    confidence=_cfg.confidence_snapshot_simplified,
                )
        except (SyntaxError, Exception):
            pass

    # Fallback: solve normally but with snapshot values as type hints
    # Use snapshot to improve type inference for unresolved vars
    variables, satisfying, all_direct, _scope2 = _solve_single_expr(
        test_expr, rp, lineno, callable_info, builtins_names
    )

    # Override with snapshot values where we have better concrete data
    for name, val in resolved.items():
        if name in free and isinstance(val, (bool, int, float, str)):
            satisfying[name] = val

    from tharkun.solve.config import config as _cfg

    satisfying_inputs = [satisfying] if satisfying else []
    strategy = "snapshot_assisted" if satisfying else "none"
    confidence = _cfg.confidence_snapshot_assisted if satisfying else 0.0

    return ConditionSolution(
        guard_source=guard_source,
        variables=variables,
        satisfying_inputs=satisfying_inputs,
        strategy=strategy,
        confidence=confidence,
    )


def _simplify_with_resolved(
    expr: ast.expr,
    resolved: dict[str, Any],
) -> ast.expr:
    """Simplify an AST expression by substituting resolved variable values.

    For ``BoolOp(And)`` expressions, evaluates each clause with resolved
    values. Clauses that evaluate to True are removed. If a clause evaluates
    to False, returns ``ast.Constant(value=False)``.

    For comparisons where both sides are resolved, evaluates them directly.
    """
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And):
        remaining: list[ast.expr] = []
        for clause in expr.values:
            result = _try_eval_clause(clause, resolved)
            if result is True:
                continue  # Clause is satisfied — remove it
            elif result is False:
                return ast.Constant(value=False)
            else:
                remaining.append(clause)

        if not remaining:
            return ast.Constant(value=True)  # All clauses satisfied
        if len(remaining) == 1:
            return remaining[0]  # Simplified to single clause
        return ast.BoolOp(op=ast.And(), values=remaining)

    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
        for clause in expr.values:
            result = _try_eval_clause(clause, resolved)
            if result is True:
                return ast.Constant(value=True)
        # Can't simplify OR unless we know a clause is True
        return expr

    # Single expression — try to evaluate
    result = _try_eval_clause(expr, resolved)
    if result is True:
        return ast.Constant(value=True)
    if result is False:
        return ast.Constant(value=False)

    return expr


def _substitute_constants(
    expr: ast.expr,
    resolved: dict[str, Any],
) -> ast.expr:
    """Replace Name nodes in *expr* with Constant nodes for resolved variables.

    This turns ``x > threshold`` with ``resolved={"threshold": 100}``
    into ``x > 100``, allowing the condition solver to extract comparison
    values directly.
    """
    import copy

    class _Substitutor(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in resolved:
                val = resolved[node.id]
                if isinstance(val, (bool, int, float, str, bytes, type(None))):
                    new_node = ast.Constant(value=val)
                    return ast.copy_location(new_node, node)
            return node

    try:
        new_expr = copy.deepcopy(expr)
        result = _Substitutor().visit(new_expr)
        ast.fix_missing_locations(result)
        return result
    except Exception:
        return expr


def _try_eval_clause(clause: ast.expr, resolved: dict[str, Any]) -> bool | None:
    """Try to evaluate a clause with resolved variable values.

    Returns True/False if evaluation succeeds, None if the clause
    contains unresolved variables.
    """
    # Check if all free names in the clause are resolved
    free = _free_names(clause)
    builtins_filter = {"len", "isinstance", "type", "range", "int", "str", "float",
                       "bool", "list", "dict", "set", "tuple", "bytes", "print",
                       "True", "False", "None", "abs", "min", "max", "sum",
                       "any", "all", "hasattr", "getattr", "callable", "not",
                       "and", "or"}
    needed = free - builtins_filter
    if not needed.issubset(resolved.keys()):
        return None

    # Try to evaluate the clause
    source = _unparse(clause)
    if source is None:
        return None

    try:
        eval_locals = dict(resolved)
        result = eval(source, _safe_eval_globals(), eval_locals)
        return bool(result)
    except Exception:
        return None


def _decompose_and_merge(
    test_expr: ast.BoolOp,
    guard_source: str,
    satisfying: dict[str, Any],
    scope_vars: dict[str, Any],
) -> dict[str, Any] | None:
    """Decompose an And guard into clauses, solve each independently, merge.

    For ``len(x) == 1 and ord(x) < 128 and x not in '<>'``:
    1. Clause 1: ``len(x) == 1`` → x must have length 1 → x is a 1-char string
    2. Clause 2: ``ord(x) < 128`` → x must be ASCII → ord < 128
    3. Clause 3: ``x not in '<>'`` → x must not be '<' or '>'
    4. Merge: x is a 1-char ASCII string not in '<>' → x = 'a'
    """
    # Solve each clause independently, propagating resolved values forward.
    # Only substitute values NEWLY RESOLVED by earlier clauses, not the
    # original (possibly wrong) satisfying values.
    newly_resolved: dict[str, Any] = dict(scope_vars)
    clause_solutions: list[dict[str, Any]] = []

    for clause in test_expr.values:
        clause_src = _unparse(clause)
        if clause_src is None:
            continue

        # Only substitute values resolved by PRIOR clauses (not original satisfying)
        resolved_for_sub = {k: v for k, v in newly_resolved.items()
                            if isinstance(v, (bool, int, float, str, bytes, type(None)))}
        substituted = _substitute_constants(clause, resolved_for_sub)

        # Solve this clause
        try:
            vars_c, sat_c, direct_c, scope_c = _solve_single_expr(
                substituted, "", 0, None, BUILTIN_NAMES
            )
            if sat_c:
                clause_solutions.append(sat_c)
                # Propagate newly resolved values forward
                for k, v in sat_c.items():
                    newly_resolved[k] = v
            else:
                clause_solutions.append({})
        except Exception:
            clause_solutions.append({})

    if not clause_solutions:
        return None

    # Merge: for each variable, find a value that satisfies ALL clauses.
    # Start with the first clause's solution, then check against others.
    merged = dict(scope_vars)
    all_vars: set[str] = set()
    for sol in clause_solutions:
        all_vars.update(sol.keys())

    for name in all_vars:
        # Collect all candidate values from different clauses
        candidates_for_var: list[Any] = []
        for sol in clause_solutions:
            if name in sol and sol[name] not in candidates_for_var:
                candidates_for_var.append(sol[name])
        if name in satisfying and satisfying[name] not in candidates_for_var:
            candidates_for_var.append(satisfying[name])

        # Try each candidate — pick the first that satisfies the full guard
        best = candidates_for_var[0] if candidates_for_var else satisfying.get(name)
        for candidate in candidates_for_var:
            trial = dict(merged)
            trial.update(satisfying)
            trial[name] = candidate
            if _verify_guard_satisfaction(guard_source, trial):
                best = candidate
                break
        if best is not None:
            merged[name] = best

    # Verify the merged solution
    trial = dict(merged)
    trial.update({k: v for k, v in satisfying.items() if k not in merged})
    if _verify_guard_satisfaction(guard_source, trial):
        result = {k: v for k, v in trial.items() if k not in scope_vars}
        return result

    # Phase 2: Constraint accumulation — collect type + value constraints
    # from all clauses, then synthesize compatible values.
    var_types: dict[str, str] = {}
    var_lens: dict[str, int] = {}
    var_values: dict[str, list[Any]] = {}

    for csol in clause_solutions:
        for name, val in csol.items():
            if name not in var_values:
                var_values[name] = []
            var_values[name].append(val)
            # Track type
            if isinstance(val, tuple):
                var_types[name] = "tuple"
            elif isinstance(val, list):
                var_types[name] = "list"
            elif isinstance(val, str):
                var_types.setdefault(name, "str")
            elif isinstance(val, bytes):
                var_types.setdefault(name, "bytes")
            elif isinstance(val, dict):
                var_types.setdefault(name, "dict")
            elif isinstance(val, int):
                var_types.setdefault(name, "int")

    # Collect len constraints from all clauses
    for clause in test_expr.values:
        for node in ast.walk(clause):
            if not isinstance(node, ast.Compare):
                continue
            left = node.left
            if (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)
                    and left.func.id == "len" and left.args
                    and isinstance(left.args[0], ast.Name)):
                var_name = left.args[0].id
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(comp, ast.Constant) and isinstance(comp.value, int):
                        if isinstance(op, ast.Eq):
                            var_lens[var_name] = comp.value

    # Synthesize type-aware values respecting len constraints
    synth: dict[str, Any] = dict(satisfying)
    synth.update(scope_vars)
    for name in var_types:
        t = var_types[name]
        ln = var_lens.get(name)
        if t == "tuple" and ln:
            synth[name] = tuple(range(ln))
        elif t == "list" and ln:
            synth[name] = list(range(ln))
        elif t == "str" and ln:
            synth[name] = "x" * ln
        elif t == "bytes" and ln:
            synth[name] = b"\x00" * ln
        elif name in newly_resolved:
            synth[name] = newly_resolved[name]

    if _verify_guard_satisfaction(guard_source, synth):
        result = {k: v for k, v in synth.items() if k not in scope_vars}
        return result

    return None


def _repair_compound_satisfaction(
    test_expr: ast.expr,
    guard_source: str,
    variables: list[VarConstraint],
    satisfying: dict[str, Any],
    scope_vars: dict[str, Any],
) -> dict[str, Any] | None:
    """Try to repair inputs for compound And guards by exploring alternatives.

    Uses minleaf's guard_normalize for canonical form, then explores
    alternative values for each variable to find a joint combination.

    When the initial solve produces inputs that don't jointly satisfy the
    guard (e.g. `isinstance(x, bytes) and len(x) > 5` with x=b"" which
    fails the len check), this function:

    1. For each variable with multiple candidate values, tries alternatives
    2. For variables with only one value, tries type-appropriate upgrades
       (e.g. b"" → b"xxxxxx" to satisfy len constraints)
    3. Verifies each combination against the full guard
    """
    # Normalize the guard for better compound handling (expand chained comparisons)
    try:
        from tharkun.solve.normalize import normalize_guard
        test_expr = normalize_guard(test_expr)
    except Exception:
        pass

    if not isinstance(test_expr, ast.BoolOp) or not isinstance(test_expr.op, ast.And):
        return None

    # Phase 1: Analytical per-clause decomposition.
    # Solve each clause independently, merge per-variable constraints.
    decomposed = _decompose_and_merge(test_expr, guard_source, satisfying, scope_vars)
    if decomposed is not None:
        return decomposed

    # Phase 2: Brute-force candidate pool (fallback).
    # Build a candidate pool for each variable
    candidates: dict[str, list[Any]] = {}
    for var in variables:
        vals = list(var.concrete_values)
        if var.name in satisfying:
            current = satisfying[var.name]
            if current not in vals:
                vals.insert(0, current)
        # Add type-upgraded versions for common patterns
        for v in list(vals):
            if isinstance(v, bytes) and len(v) < 8:
                vals.append(v + b"\x00" * 8)
            if isinstance(v, str) and len(v) < 8:
                vals.append(v + "x" * 8)
            if isinstance(v, list) and len(v) < 4:
                vals.append(v + [0] * 4)
            if v is True:
                vals.append(1)
            if isinstance(v, bool):
                vals.append(not v)
        if vals:
            candidates[var.name] = vals

    if not candidates:
        return None

    # Try combinations (limited to avoid explosion)
    # Start with the original satisfying and try swapping one variable at a time
    base = dict(satisfying)
    base.update(scope_vars)

    # First: try upgrading each variable individually
    for name, vals in candidates.items():
        for candidate in vals:
            trial = dict(base)
            trial[name] = candidate
            if _verify_guard_satisfaction(guard_source, trial):
                result = {k: v for k, v in trial.items() if k not in scope_vars}
                return result

    # Second: for 2-3 variable guards, try pairwise combinations
    var_names = list(candidates.keys())
    if len(var_names) <= 3:
        import itertools
        pools = [candidates.get(n, [satisfying.get(n, 0)]) for n in var_names]
        # Limit to first 5 candidates per variable to avoid explosion
        pools = [p[:5] for p in pools]
        for combo in itertools.product(*pools):
            trial = dict(base)
            for name, val in zip(var_names, combo):
                trial[name] = val
            if _verify_guard_satisfaction(guard_source, trial):
                result = {k: v for k, v in trial.items() if k not in scope_vars}
                return result

    return None


def _safe_eval_globals() -> dict[str, Any]:
    """Build a safe eval context with common builtins for guard verification."""
    import re as _re
    import math as _math
    return {
        "__builtins__": {
            "len": len, "isinstance": isinstance, "type": type,
            "int": int, "str": str, "float": float, "bool": bool,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "bytes": bytes, "bytearray": bytearray, "memoryview": memoryview,
            "frozenset": frozenset, "complex": complex, "object": object,
            "hasattr": hasattr, "getattr": getattr, "callable": callable,
            "True": True, "False": False, "None": None,
            "abs": abs, "min": min, "max": max, "sum": sum,
            "any": any, "all": all, "sorted": sorted, "reversed": reversed,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "range": range, "repr": repr, "ord": ord, "chr": chr,
            "hex": hex, "bin": bin, "oct": oct, "round": round,
            "divmod": divmod, "pow": pow, "slice": slice,
        },
        "re": _re,
        "math": _math,
    }


def _verify_guard_satisfaction(guard_source: str, inputs: dict[str, Any]) -> bool:
    """Evaluate the guard expression with the given inputs.

    Uses minleaf's guard_trace when available for explicit cause tracking.
    Falls back to eval() if parsing/tracing fails.

    Returns True if the guard evaluates to True, False otherwise.
    If evaluation fails (missing imports, complex types), returns False
    to avoid false promotions.
    """
    if not guard_source or not inputs:
        return False

    # Try minleaf's explicit guard tracer first — gives cause tracking
    try:
        import ast as _ast
        from tharkun.solve.normalize import normalize_guard
        from tharkun.solve.guard_eval import trace_guard

        guard_expr = _ast.parse(guard_source, mode="eval").body
        normalized = normalize_guard(guard_expr)
        # Build eval env: merge inputs with safe builtins
        eval_env = dict(_safe_eval_globals().get("__builtins__", {}))
        eval_env.update(_safe_eval_globals())
        eval_env.update(inputs)
        traced = trace_guard(normalized, eval_env, desired_value=True)
        return traced.trace.guard_value
    except Exception:
        pass

    # Fallback: simple eval
    try:
        result = eval(guard_source, _safe_eval_globals(), dict(inputs))
        return bool(result)
    except Exception:
        return False


_STRATEGY_EXPLANATIONS: dict[str, str] = {
    "direct": "Guard '{guard}' was solved directly from the comparison/membership expression.",
    "comparison_inversion": "Guard '{guard}' is a comparison against a literal. Produced inputs by stepping past the boundary.",
    "isinstance_check": "Guard requires a variable to be an instance of a specific type. Constructed a minimal instance.",
    "for_else": "For-loop else branch requires an empty iterable. Returned __iter_empty__ sentinel.",
    "infeasible": "Guard is statically infeasible (always-false literal or contradictory constraints). No inputs can satisfy it.",
    "none": "No satisfying inputs were found for guard '{guard}'.",
    "type_inferred": "Guard '{guard}' was partially solved by type inference. Inputs may not fully satisfy all clauses.",
    "or_branches": "Guard '{guard}' was solved by exploring each branch of an or-expression independently.",
}


def _build_explanation(strategy: str, guard_source: str, variables: list[VarConstraint]) -> str:
    """Build a human-readable explanation string for a solving strategy."""
    base = strategy.split("+")[0]
    template = _STRATEGY_EXPLANATIONS.get(strategy) or _STRATEGY_EXPLANATIONS.get(base, "Solved using strategy '{strategy}'.")
    ctx: dict[str, str] = {"guard": guard_source or "<unknown>", "strategy": strategy}
    if variables:
        first = variables[0]
        ctx.update({"var": first.name, "type": first.inferred_type})
        if first.concrete_values:
            ctx["result"] = repr(first.concrete_values[0])
    try:
        return template.format(**ctx)
    except KeyError:
        return template


def _build_input_constraints(variables: list[VarConstraint]) -> dict[str, dict] | None:
    """Translate VarConstraints to agent-readable constraint dicts. Best-effort; never raises."""
    if not variables:
        return None
    result: dict[str, dict] = {}
    for vc in variables:
        try:
            c: dict[str, Any] = {"type": vc.inferred_type}
            vals = vc.concrete_values
            if not vals:
                result[vc.name] = c
                continue
            fv = vals[0]
            if vc.inferred_type == "int" and isinstance(fv, int) and not isinstance(fv, bool):
                c["gte" if vc.inference_source == "comparison_literal" else "eq"] = fv
            elif vc.inferred_type == "float" and isinstance(fv, float):
                c["gte"] = fv
            elif vc.inferred_type == "str" and isinstance(fv, str):
                c["eq"] = fv
            elif vc.inferred_type in ("list", "tuple", "set") and hasattr(fv, "__len__"):
                c["min_len"] = len(fv)
                if fv:
                    c["element_type"] = type(next(iter(fv))).__name__
            elif vc.inferred_type == "dict" and isinstance(fv, dict):
                c["required_keys"] = list(fv.keys())
            elif vc.inferred_type == "bool" and isinstance(fv, bool):
                c["eq"] = fv
            elif vc.inferred_type == "bytes" and isinstance(fv, bytes):
                if vc.inference_source == "comparison_literal":
                    c["eq"] = fv.hex()
                else:
                    c["min_len"] = len(fv)
            result[vc.name] = c
        except Exception:
            result[vc.name] = {"type": vc.inferred_type}
    return result or None


def serialize_solution(sol: ConditionSolution) -> dict[str, Any]:
    """Serialize a ConditionSolution for JSON output."""
    return {
        "guard_source": sol.guard_source,
        "desired_branch": sol.desired_branch,
        "strategy": sol.strategy,
        "confidence": sol.confidence,
        "explanation": sol.explanation,
        "input_constraints": sol.input_constraints,
        "variables": [
            {
                "name": v.name,
                "inferred_type": v.inferred_type,
                "inference_source": v.inference_source,
                "concrete_values": [repr(x) for x in v.concrete_values],
                "strategy_hint": v.strategy_hint,
            }
            for v in sol.variables
        ],
        "satisfying_inputs": [
            {k: repr(v) for k, v in inp.items()}
            for inp in sol.satisfying_inputs
        ],
    }


def _with_annotated_ints(tree: ast.AST, lineno: int, var_types: dict | None) -> dict | None:
    """*var_types* plus the names the innermost function around *lineno* annotates as
    ``int`` (a parameter ``x: int`` or a statement ``x: int = ...``)."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None) or node.lineno
            if node.lineno <= lineno <= end and (best is None or node.lineno > best.lineno):
                best = node
    if best is None:
        return var_types
    out = dict(var_types or {})

    def is_int(ann):
        return isinstance(ann, ast.Name) and ann.id == "int"

    a = best.args
    for arg in a.posonlyargs + a.args + a.kwonlyargs:
        if is_int(arg.annotation):
            out.setdefault(arg.arg, "int")
    for node in ast.walk(best):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and is_int(node.annotation):
            out.setdefault(node.target.id, "int")
    return out


def _known_int_names(callable_info, facts, var_types, variables) -> set:
    """Free names the caller (``var_types``, an ``int`` annotation, integer facts) or,
    when *variables* is given, the inference says are integers."""
    names = {n for n, t in (var_types or {}).items() if t == "int"}
    for p in getattr(callable_info, "params", None) or []:
        if getattr(p.type_hint, "rendered", "") == "int":
            names.add(p.name)
    for n, vf in (getattr(facts, "facts", None) or {}).items():
        vals = [v for v in (vf.lo, vf.hi) if v is not None] + list(vf.eq or ())
        if vals and all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            names.add(n)
    for vc in variables or []:
        if vc.inferred_type == "int":
            names.add(vc.name)
    return names


def _arith_fallback(test_expr, guard_source, desired_branch, callable_info, facts,
                    var_types, variables) -> ConditionSolution | None:
    """The native integer solver's answer for an unsolved integer guard, or ``None``.

    ``test_expr`` is already the polarity-adjusted guard (negated for ``else``)."""
    ints = _known_int_names(callable_info, facts, var_types, variables)
    if not ints:
        return None
    from tharkun.solve.arith import free_names, solve_int_guard
    free = free_names(test_expr)
    if not free or not free <= ints:
        return None
    res = solve_int_guard(test_expr, free)
    # An integer refutation is a claim only when the caller vouched for every name's
    # type -- an inferred ``int`` may be a float at run time.
    trusted = _known_int_names(callable_info, facts, var_types, None)
    if res.status == "unsat" and free <= trusted:
        return ConditionSolution(
            guard_source=guard_source, strategy="infeasible", confidence=0.0,
            desired_branch=desired_branch,
            explanation=f"Guard '{guard_source}' has no integer solution ({res.explanation}).",
        )
    if res.status != "sat":
        return None
    from tharkun.solve.config import config as _cfg
    arith_vars = [VarConstraint(name=n, inferred_type="int", inference_source="arith",
                                concrete_values=[v]) for n, v in sorted(res.witness.items())]
    return ConditionSolution(
        guard_source=guard_source,
        variables=arith_vars,
        satisfying_inputs=[dict(res.witness)],
        strategy="arith",
        confidence=_cfg.confidence_verified,
        desired_branch=desired_branch,
        explanation=(f"Guard '{guard_source}' was solved over the integers "
                     f"({res.explanation}); the witness satisfies it."),
        input_constraints=_build_input_constraints(arith_vars),
    )
