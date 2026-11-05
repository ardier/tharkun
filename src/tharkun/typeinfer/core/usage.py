"""The usage rules: what a name's uses in an expression say about its type (docs/TYPEINFER.md, source 3).

Each rule reads one expression and answers ``(type text, rule name)`` or ``None``: ``isinstance(x, T)`` says ``T``,
``x > 5`` says ``int``, ``x.startswith("p")`` says ``str``, ``x["k"]`` says ``dict``, and so on.  This is the one
home of these rules ([internal ref] A2).  Two readers call :func:`usage_type`:

* ``tharkun.solve.condition.infer_var_type`` types a free variable of a guard (the guard solver's inference);
* ``tharkun.typeinfer.core.params`` types a parameter from every expression of a function body (source 3).

They differ in two places, both named by ``parameter=True`` (see :func:`usage_type`).  Stdlib and
``tharkun.solve.guard`` only; it reads ASTs and runs nothing.
"""

from __future__ import annotations

import ast
from typing import Any

from tharkun.solve.guard import (
    attr_chain_parts,
    attr_chain_root,
    iter_byte_input_atoms,
    try_extract_constant,
    unparse,
    unwrap_named_expr,
)

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


def _infer_from_comparison(name: str, expr: ast.expr, *, typed_only: bool = False) -> tuple[str, str] | None:
    """Rule 1: Infer type from comparison with literal.

    ``typed_only``: a constant that is no type (``None``) is no evidence, so ``x is None or x > 3`` still reads
    ``x > 3``; without it the first constant decides, ``Any`` included (the guard solver's reading)."""
    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for op in operands:
            unwrapped = unwrap_named_expr(op)
            if isinstance(unwrapped, ast.Name) and unwrapped.id == name:
                for other in operands:
                    if other is op:
                        continue
                    val, ok = try_extract_constant(other)
                    if ok and not (typed_only and _type_of_constant(val) == "Any"):
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


def _infer_from_isinstance(name: str, expr: ast.expr, *, chains: bool = True) -> tuple[str, str] | None:
    """Rule 2: Infer type from isinstance() call.

    ``chains``: ``isinstance(x.y, T)`` types the root ``x`` (the guard solver's reading); without it only
    ``isinstance(x, T)`` does."""
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
        elif not chains:
            continue
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
            s = unparse(arg1)
            return (s or "Any"), "isinstance"
        if isinstance(arg1, ast.Tuple):
            types = []
            for elt in arg1.elts:
                if isinstance(elt, ast.Name):
                    types.append(elt.id)
                else:
                    s = unparse(elt)
                    if s:
                        types.append(s)
            if types:
                return types[0], "isinstance"
    return None


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
        if isinstance(left, ast.Attribute) and attr_chain_root(left) == name:
            return "object", "attr_comparison"
        for comp in node.comparators:
            if isinstance(comp, ast.Attribute) and attr_chain_root(comp) == name:
                return "object", "attr_comparison"
    return None


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
    if isinstance(expr, ast.Attribute) and attr_chain_root(expr) == name:
        parts = attr_chain_parts(expr)
        if parts and len(parts) >= 1:
            return "object", "attr_truthiness"

    # Check UnaryOp(Not, obj.a.b)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        if isinstance(expr.operand, ast.Attribute) and attr_chain_root(expr.operand) == name:
            parts = attr_chain_parts(expr.operand)
            if parts and len(parts) >= 1:
                return "object", "attr_truthiness_negated"

    # Check inside BoolOp values
    for node in ast.walk(expr):
        if not isinstance(node, ast.BoolOp):
            continue
        for value in node.values:
            if isinstance(value, ast.Attribute) and attr_chain_root(value) == name:
                parts = attr_chain_parts(value)
                if parts and len(parts) >= 1:
                    return "object", "attr_truthiness"
            # not obj.a.b inside BoolOp
            if (isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not)
                    and isinstance(value.operand, ast.Attribute)
                    and attr_chain_root(value.operand) == name):
                parts = attr_chain_parts(value.operand)
                if parts and len(parts) >= 1:
                    return "object", "attr_truthiness_negated"

    return None


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


def _infer_from_byte_input(name: str, expr: ast.expr) -> tuple[str, str] | None:
    """Rule 4b: ``name`` is ``bytes`` when tested via byte index or from_bytes.

    Recognizes ``name[k] <op> <int>`` and
    ``int.from_bytes(name[i:j], order) <op> <int>``. Returns ``("bytes",
    "byte_input")`` so value generation synthesizes a satisfying bytes object.
    Runs before the generic subscript rule (which would say ``list``).
    """
    for _atom in iter_byte_input_atoms(name, expr):
        return "bytes", "byte_input"
    return None


#: the rules, in priority order (the first to answer wins)
_RULES = (
    _infer_from_isinstance,
    _infer_from_comparison,
    _infer_from_bitwise,
    _infer_from_method_call,
    _infer_from_method_chain,
    _infer_from_hasattr,
    _infer_from_attr_comparison,
    _infer_from_attr_chain_truthiness,
    _infer_from_type_is,
    _infer_from_membership_target,
    _infer_from_byte_input,  # before the generic subscript rule, which would say ``list``
    _infer_from_subscript,
)


#: the keyword arguments a rule takes when the name is a parameter of a function (see :func:`usage_type`)
_AS_PARAMETER: dict = {_infer_from_isinstance: {"chains": False}, _infer_from_comparison: {"typed_only": True}}


def usage_type(name: str, expr: ast.expr, *, parameter: bool = False) -> tuple[str, str] | None:
    """The type that *name*'s uses in *expr* imply, as ``(type text, rule name)``; ``None`` when no rule applies.

    ``parameter=False`` is the guard solver's reading: a variable of a guard, whose first applicable rule decides.
    ``parameter=True`` types a function's parameter from one expression of its body, and differs in two places:

    * ``isinstance(x.y, T)`` says something about ``x.y`` and nothing about ``x``, so an attribute chain does not
      type its root (the guard solver does type it: the chain is the variable's own guard);
    * a comparison with a constant that is no type (``x is None``) is no evidence, and does not stop the later
      comparisons and rules from reading the same expression.
    """
    for rule in _RULES:
        found = rule(name, expr, **(_AS_PARAMETER.get(rule, {}) if parameter else {}))
        if found:
            return found
    return None
