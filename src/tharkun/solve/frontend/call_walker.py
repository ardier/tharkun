"""Shared call-chain walking and callee resolution infrastructure.

Provides the core building blocks for traversing call chains in Python ASTs:
function/class indexing, callee resolution (including method calls), and
return condition extraction.

Both ``guard_decomposer`` (extracts return conditions) and ``call_mapper``
(extracts argument mappings) are thin wrappers over this shared module.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CallIndex:
    """Cached function/class index for a module.

    Attributes:
        func_defs:   ``{name: FunctionDef}`` for top-level functions and
                     class methods.  Methods are indexed twice: as
                     ``"ClassName.method"`` and as bare ``"method"``.
        class_defs:  ``{name: ClassDef}`` for top-level classes.
        imports:     ``{local_name: resolved_module_path}`` from
                     ``import_resolver.parse_imports``.
        source_path: The source file this index was built from.
    """
    func_defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = field(
        default_factory=dict,
    )
    class_defs: dict[str, ast.ClassDef] = field(default_factory=dict)
    imports: dict[str, str] = field(default_factory=dict)
    source_path: str | None = None


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def build_call_index(
    tree: ast.Module,
    source_path: str | None = None,
) -> CallIndex:
    """Build a :class:`CallIndex` from a module AST.

    Walks top-level functions and class methods (one level deep).
    When *source_path* is given, also resolves imports via
    ``import_resolver`` and includes cross-module function definitions.
    """
    func_defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    class_defs: dict[str, ast.ClassDef] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_defs[node.name] = node
        elif isinstance(node, ast.ClassDef):
            class_defs[node.name] = node
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_defs[f"{node.name}.{child.name}"] = child
                    if child.name not in func_defs:
                        func_defs[child.name] = child

    imports: dict[str, str] = {}
    if source_path:
        from tharkun.solve.frontend.import_resolver import load_function_def, parse_imports

        imports = parse_imports(source_path)
        for local_name, mod_path in imports.items():
            if local_name in func_defs:
                continue  # Local definition takes precedence.
            func_def = load_function_def(mod_path, local_name)
            if func_def is not None:
                func_defs[local_name] = func_def

    return CallIndex(
        func_defs=func_defs,
        class_defs=class_defs,
        imports=imports,
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# Callee resolution
# ---------------------------------------------------------------------------


def _resolve_class(
    class_name: str,
    index: CallIndex,
) -> ast.ClassDef | None:
    """Resolve a class name to its AST definition (same-module or cross-module)."""
    cls = index.class_defs.get(class_name)
    if cls is not None:
        return cls

    # Try cross-module.
    mod_path = index.imports.get(class_name)
    if mod_path is not None:
        from tharkun.solve.frontend.import_resolver import load_class_def
        return load_class_def(mod_path, class_name)

    return None


def _find_method_in_class(
    cls: ast.ClassDef,
    method_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Find a method by name in a ClassDef's body."""
    for node in cls.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == method_name:
                return node
    return None


def resolve_callee(
    call_expr: ast.Call,
    index: CallIndex,
    *,
    enclosing_class: ast.ClassDef | None = None,
    scope_isinstance: dict[str, str] | None = None,
    scope_annotations: dict[str, str] | None = None,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, str | None]:
    """Resolve a call expression to its function definition.

    Returns ``(func_def, qualified_name)`` or ``(None, None)`` if
    resolution fails.

    Resolution strategy (priority order):

    1. ``ast.Name`` — simple call like ``foo(x)``.  Looked up in
       ``index.func_defs``.
    2. ``self.method()`` / ``cls.method()`` — looked up in
       *enclosing_class*.
    3. ``super().method()`` — resolves parent class from
       ``enclosing_class.bases[0]``, looks up method.
    4. Typed receiver via ``isinstance`` — uses *scope_isinstance*
       to find the class, then the method.
    5. Typed receiver via annotation — uses *scope_annotations*.
    6. Known class name in index — ``ClassName.method()`` where
       ``ClassName`` is in ``index.class_defs``.
    7. Everything else → ``(None, None)``.
    """
    func = call_expr.func

    # --- 1. Simple name: foo() ---
    if isinstance(func, ast.Name):
        name = func.id
        func_def = index.func_defs.get(name)
        if func_def is not None:
            return func_def, name
        return None, None

    # --- Attribute calls: obj.method() ---
    if isinstance(func, ast.Attribute):
        method_name = func.attr

        # --- 2. self.method() / cls.method() ---
        if isinstance(func.value, ast.Name) and func.value.id in ("self", "cls"):
            if enclosing_class is not None:
                meth = _find_method_in_class(enclosing_class, method_name)
                if meth is not None:
                    qname = f"{enclosing_class.name}.{method_name}"
                    return meth, qname
            # Fallback: try index with qualified name.
            for cls_name, cls_def in index.class_defs.items():
                meth = _find_method_in_class(cls_def, method_name)
                if meth is not None:
                    return meth, f"{cls_name}.{method_name}"
            return None, None

        # --- 3. super().method() ---
        if (isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name)
                and func.value.func.id == "super"):
            if enclosing_class is not None and enclosing_class.bases:
                base = enclosing_class.bases[0]
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name is not None:
                    parent_cls = _resolve_class(base_name, index)
                    if parent_cls is not None:
                        meth = _find_method_in_class(parent_cls, method_name)
                        if meth is not None:
                            return meth, f"{base_name}.{method_name}"
            return None, None

        # --- Receiver is ast.Name("obj") — try typed resolution ---
        if isinstance(func.value, ast.Name):
            receiver_name = func.value.id

            # --- 4. isinstance-typed receiver ---
            if scope_isinstance and receiver_name in scope_isinstance:
                cls_name = scope_isinstance[receiver_name]
                cls_def = _resolve_class(cls_name, index)
                if cls_def is not None:
                    meth = _find_method_in_class(cls_def, method_name)
                    if meth is not None:
                        return meth, f"{cls_name}.{method_name}"

            # --- 5. Annotation-typed receiver ---
            if scope_annotations and receiver_name in scope_annotations:
                cls_name = scope_annotations[receiver_name]
                cls_def = _resolve_class(cls_name, index)
                if cls_def is not None:
                    meth = _find_method_in_class(cls_def, method_name)
                    if meth is not None:
                        return meth, f"{cls_name}.{method_name}"

            # --- 6. Known class name: ClassName.method() ---
            cls_def = index.class_defs.get(receiver_name)
            if cls_def is not None:
                meth = _find_method_in_class(cls_def, method_name)
                if meth is not None:
                    return meth, f"{receiver_name}.{method_name}"

            # Last resort: try qualified name in func_defs index.
            qname = f"{receiver_name}.{method_name}"
            func_def = index.func_defs.get(qname)
            if func_def is not None:
                return func_def, qname
            # Try bare method name.
            func_def = index.func_defs.get(method_name)
            if func_def is not None:
                return func_def, method_name

        return None, None

    # --- Everything else (subscript calls, nested calls, etc.) ---
    return None, None


# ---------------------------------------------------------------------------
# Return condition extraction
# ---------------------------------------------------------------------------


def is_falsy_return(node: ast.Return) -> bool:
    """Return True if this is ``return False``, ``return None``, or bare ``return``."""
    if node.value is None:
        return True
    if isinstance(node.value, ast.Constant) and node.value.value in (False, None, 0, ""):
        return True
    return False


def is_truthy_return(node: ast.Return) -> bool:
    """Return True if this is ``return True`` or ``return <non-trivial expr>``."""
    if node.value is None:
        return False
    if isinstance(node.value, ast.Constant) and node.value.value in (False, None, 0, ""):
        return False
    return True


def extract_guard_clause_condition(stmt: ast.stmt) -> ast.expr | None:
    """If *stmt* is a guard clause (``if X: return False/raise``), return
    the **negation** of X (i.e. the condition to pass through the guard).

    Returns ``None`` if *stmt* is not a guard clause.
    """
    if not isinstance(stmt, ast.If):
        return None
    if len(stmt.body) != 1:
        return None
    action = stmt.body[0]
    is_guard = False
    if isinstance(action, ast.Return) and is_falsy_return(action):
        is_guard = True
    if isinstance(action, ast.Raise):
        is_guard = True
    if not is_guard:
        return None
    return ast.UnaryOp(op=ast.Not(), operand=stmt.test)


def extract_return_condition(
    func_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.expr | None:
    """Extract the effective boolean condition from a function's return logic.

    Handles these patterns:

    **Pattern 1 — single expression return:**
    ``def f(x): return x > 0 and x < 100``
    → returns the expression ``x > 0 and x < 100``

    **Pattern 2 — guard clauses + final return:**
    ``def f(x):``
    ``    if not x: return False``
    ``    if x < 0: return False``
    ``    return True``
    → returns ``x and not (x < 0)`` (conjunction of negated guard conditions)

    **Pattern 3 — if-true/else-false:**
    ``def f(x): if cond: return True; return False``
    → returns ``cond``

    Returns ``None`` if the function body is too complex to extract.
    """
    body = func_def.body
    if not body:
        return None

    # Skip leading docstring.
    stmts = body
    if (isinstance(stmts[0], ast.Expr)
            and isinstance(stmts[0].value, ast.Constant)
            and isinstance(stmts[0].value.value, str)):
        stmts = stmts[1:]
    if not stmts:
        return None

    # Pattern 1: single return statement.
    if len(stmts) == 1 and isinstance(stmts[0], ast.Return) and stmts[0].value is not None:
        return stmts[0].value

    # Pattern 3: if/return True followed by return False.
    if (len(stmts) >= 2
            and isinstance(stmts[-1], ast.Return) and is_falsy_return(stmts[-1])
            and isinstance(stmts[-2], ast.If)
            and len(stmts[-2].body) == 1
            and isinstance(stmts[-2].body[0], ast.Return)
            and is_truthy_return(stmts[-2].body[0])):
        conditions: list[ast.expr] = []
        for s in stmts[:-2]:
            guard_cond = extract_guard_clause_condition(s)
            if guard_cond is not None:
                conditions.append(guard_cond)
            else:
                return None
        conditions.append(stmts[-2].test)
        if len(conditions) == 1:
            return conditions[0]
        return ast.BoolOp(op=ast.And(), values=conditions)

    # Pattern 2: sequence of guard clauses + final return.
    if isinstance(stmts[-1], ast.Return) and is_truthy_return(stmts[-1]):
        conditions = []
        for s in stmts[:-1]:
            guard_cond = extract_guard_clause_condition(s)
            if guard_cond is not None:
                conditions.append(guard_cond)

        ret_expr = stmts[-1].value
        if isinstance(ret_expr, ast.Constant) and ret_expr.value is True:
            pass
        elif ret_expr is not None:
            conditions.append(ret_expr)

        if conditions:
            if len(conditions) == 1:
                return conditions[0]
            return ast.BoolOp(op=ast.And(), values=conditions)
        return ret_expr

    return None


# ---------------------------------------------------------------------------
# Chained comparison negation
# ---------------------------------------------------------------------------

_NEGATE_OP: dict[type, type] = {
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}


def negate_chained_compare(node: ast.Compare) -> list[ast.Compare] | None:
    """Decompose ``not (A <= x <= B)`` into ``[x < A, x > B]``.

    For a chained comparison ``left op0 comp0 op1 comp1 ...``, the negation
    is the OR of each individual comparison negated:
    ``not (a <= x <= b)``  →  ``x < a  OR  x > b``

    Returns a list of negated individual comparisons, or ``None`` if the
    chain can't be decomposed.
    """
    parts: list[ast.Compare] = []
    operands = [node.left] + list(node.comparators)

    for i, op in enumerate(node.ops):
        neg_op_type = _NEGATE_OP.get(type(op))
        if neg_op_type is None:
            return None
        parts.append(ast.Compare(
            left=operands[i],
            ops=[neg_op_type()],
            comparators=[operands[i + 1]],
        ))

    return parts if parts else None


# ---------------------------------------------------------------------------
# Scope analysis helpers
# ---------------------------------------------------------------------------


def find_enclosing_class(
    tree: ast.Module,
    func_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.ClassDef | None:
    """Find the ClassDef in *tree* that directly contains *func_def*.

    Uses identity comparison on the AST node.  Returns ``None`` if
    *func_def* is a module-level function.
    """
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if child is func_def:
                    return node
    return None


def extract_scope_isinstance(
    func_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    """Scan a function body for ``isinstance(name, Type)`` calls.

    Returns ``{variable_name: "TypeName"}`` for each detected pattern.
    Only considers calls where the first arg is ``ast.Name`` and the
    second arg is ``ast.Name`` (not tuples of types).
    """
    result: dict[str, str] = {}
    for node in ast.walk(func_def):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "isinstance"):
            continue
        if len(node.args) < 2:
            continue
        arg0, arg1 = node.args[0], node.args[1]
        if isinstance(arg0, ast.Name) and isinstance(arg1, ast.Name):
            result[arg0.id] = arg1.id
        elif isinstance(arg0, ast.Name) and isinstance(arg1, ast.Attribute):
            # isinstance(x, module.Type) — use the attribute name.
            result[arg0.id] = arg1.attr
    return result


def extract_scope_annotations(
    func_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    """Extract parameter type annotations from a function definition.

    Returns ``{param_name: "TypeName"}`` for parameters with simple
    ``ast.Name`` or ``ast.Attribute`` annotations.
    """
    result: dict[str, str] = {}
    for arg in func_def.args.args:
        ann = arg.annotation
        if ann is None:
            continue
        if isinstance(ann, ast.Name):
            result[arg.arg] = ann.id
        elif isinstance(ann, ast.Attribute):
            result[arg.arg] = ann.attr
        elif isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            # PEP 563 string annotations: "ClassName"
            result[arg.arg] = ann.value
    return result
