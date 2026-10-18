"""Recursive field type resolution across class hierarchies.

Given a root class name and an attribute chain, walks class definitions
(possibly across files) to determine the type of the leaf field.

This fills the gap where condition_solver builds SimpleNamespace for guards
like ``obj.config.timeout > 5`` without knowing that ``config: Config`` and
``timeout: int``.

Usage::

    from tharkun.solve.frontend.field_resolver import resolve_field_chain

    # Resolve "config.timeout" on class "Client"
    result = resolve_field_chain(
        root_class="Client",
        chain=["config", "timeout"],
        source_path="/path/to/module.py",
    )
    if result:
        print(result.leaf_type)    # "int"
        print(result.chain_types)  # ["Config", "int"]
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path

from tharkun.solve.frontend.import_resolver import load_class_def, parse_imports


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldResolution:
    """Result of recursively resolving an attribute chain from a class.

    Attributes:
        leaf_type:      The inferred type of the last field in the chain,
                        e.g. ``"int"`` for ``["config", "timeout"]`` on a class
                        where ``config: Config`` and ``Config.timeout: int``.
        leaf_path:      File path where the leaf type's class is defined.
                        ``None`` for builtins (int, str, ...).
        chain_types:    Resolved type at each step.  May be shorter than the
                        chain if resolution stops early.
        resolved_depth: ``len(chain_types)`` — how many chain steps resolved.
    """
    leaf_type: str
    leaf_path: str | None
    chain_types: list[str]
    resolved_depth: int


# ---------------------------------------------------------------------------
# Builtin type recognition
# ---------------------------------------------------------------------------

_BUILTIN_TYPES = frozenset({
    "int", "float", "str", "bytes", "bool",
    "list", "dict", "set", "tuple", "frozenset",
    "None", "Any", "object", "type",
    # typing aliases (bare names only — generics stripped before checking)
    "List", "Dict", "Set", "Tuple", "FrozenSet",
    "Optional", "Union", "Sequence", "Iterable", "Iterator",
    "Callable", "Type", "ClassVar",
})


def _is_builtin_type(type_str: str) -> bool:
    """Return True if *type_str* is a Python builtin or standard typing alias."""
    base = type_str.split("[")[0].strip()
    # Handle dotted names: typing.Optional → Optional
    if "." in base:
        base = base.rsplit(".", 1)[-1]
    return base in _BUILTIN_TYPES


def _strip_annotation(type_str: str) -> str:
    """Clean an annotation string for class lookup.

    Handles:
    - Forward-reference string quotes: ``"'NodeB'"`` → ``"NodeB"``
    - Optional wrapper: ``"Optional[Config]"`` → ``"Config"``
    - PEP 604 union: ``"Config | None"`` → ``"Config"``
    - Generic parameters (when base is a user class): ``"List[int]"`` → ``"List"``

    Returns the unwrapped base type name suitable for class lookup.
    """
    s = type_str.strip()

    # Strip outer string quotes (forward references): 'NodeB' or "NodeB"
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        s = s[1:-1].strip()

    # Unwrap Optional[X] → X
    if s.startswith("Optional[") and s.endswith("]"):
        inner = s[len("Optional["):-1].strip()
        # Recurse to handle Optional["NodeB"] etc.
        return _strip_annotation(inner)

    # Unwrap Union[X, None] or Union[None, X] → X
    if s.startswith("Union[") and s.endswith("]"):
        parts = [p.strip() for p in s[len("Union["):-1].split(",")]
        non_none = [p for p in parts if p not in ("None", "type[None]")]
        if len(non_none) == 1:
            return _strip_annotation(non_none[0])

    # PEP 604: X | None  or  None | X
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
        non_none = [p for p in parts if p not in ("None", "type[None]")]
        if len(non_none) == 1:
            return _strip_annotation(non_none[0])

    # Strip generic params from user types: MyClass[int] → MyClass
    if "[" in s:
        s = s.split("[")[0].strip()

    return s


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_class(
    class_name: str,
    source_path: str,
) -> tuple[ast.ClassDef, str] | None:
    """Locate a class definition by name, starting from *source_path*.

    Search order:
    1. Class defined in *source_path* itself.
    2. Class imported into *source_path* (``from foo import Bar``).

    Returns ``(class_def, resolved_file_path)`` or ``None``.
    """
    rp = os.path.realpath(source_path)

    # 1. Same file
    cls = load_class_def(rp, class_name)
    if cls is not None:
        return cls, rp

    # 2. Imports
    imports = parse_imports(source_path)
    mod_path = imports.get(class_name)
    if mod_path:
        cls = load_class_def(mod_path, class_name)
        if cls is not None:
            return cls, mod_path

    return None


def _annotation_to_str(node: ast.expr | None) -> str | None:
    """Unparse an annotation node to a clean type string.

    Handles forward references: ``"NodeB"`` (stored as ``ast.Constant``)
    unparsed by ``ast.unparse`` → ``"'NodeB'"`` — we strip the outer quotes
    so callers see ``"NodeB"`` instead.
    """
    if node is None:
        return None
    try:
        raw = ast.unparse(node)
    except Exception:
        return None
    # Strip outer string quotes from forward-reference annotations
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] in ('"', "'") and raw[-1] == raw[0]:
        raw = raw[1:-1].strip()
    return raw


def _infer_type_from_value(node: ast.expr) -> str | None:
    """Infer a type string from an AST value expression (for unannotated fields).

    Only covers concrete cases:
    - Constant literals → int / float / str / bytes / bool / None
    - Empty containers → list / dict / set / tuple
    - Constructor call ``Foo(...)`` → ``"Foo"``
    """
    if isinstance(node, ast.Constant):
        v = node.value
        if v is None:
            return None  # ambiguous
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
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.Set):
        return "set"
    if isinstance(node, ast.Tuple):
        return "tuple"
    # Constructor call: Foo() or module.Foo()
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            try:
                return ast.unparse(func)
            except Exception:
                pass
    return None


def _get_field_type(
    class_def: ast.ClassDef,
    field_name: str,
) -> str | None:
    """Extract the declared type of *field_name* in *class_def*.

    Priority:
    1. Class-body ``ast.AnnAssign``: ``timeout: int`` or ``timeout: int = 30``
    2. ``__init__`` body ``ast.AnnAssign``: ``self.timeout: int = 30``
    3. ``__init__`` param annotation + ``self.field = param`` assignment
    4. ``self.field = SomeClass(...)`` constructor call (unambiguous type)
    5. ``self.field = literal`` constant value inference
    """
    init_def: ast.FunctionDef | ast.AsyncFunctionDef | None = None

    # Pass 1: class body
    for stmt in class_def.body:
        # class-level annotation: timeout: int  OR  timeout: int = 30
        if isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            if isinstance(target, ast.Name) and target.id == field_name:
                ann = _annotation_to_str(stmt.annotation)
                if ann:
                    return ann

        # locate __init__
        if (isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and stmt.name == "__init__"):
            init_def = stmt

    if init_def is None:
        return None

    # Pass 2: __init__ parameter annotations → self.field = param
    param_types: dict[str, str] = {}
    for arg in init_def.args.args:
        if arg.arg == "self":
            continue
        ann = _annotation_to_str(arg.annotation)
        if ann:
            param_types[arg.arg] = ann

    # Pass 3: walk __init__ body for self.field assignments
    for stmt in ast.walk(init_def):
        # AnnAssign: self.field: int = ...
        if isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == field_name):
                ann = _annotation_to_str(stmt.annotation)
                if ann:
                    return ann

        # Regular assign: self.field = <expr>
        elif isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if not (isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                        and tgt.attr == field_name):
                    continue
                rhs = stmt.value
                # self.field = param_name → use param annotation
                if isinstance(rhs, ast.Name) and rhs.id in param_types:
                    return param_types[rhs.id]
                # self.field = SomeClass(...) or constant
                inferred = _infer_type_from_value(rhs)
                if inferred:
                    return inferred

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_enclosing_class(source_path: str, lineno: int) -> str | None:
    """Find the class name that encloses the method containing *lineno*.

    Walks the top-level ``ClassDef`` nodes and checks whether any method
    spans the given line.  Returns the class name, or ``None``.
    """
    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start = item.lineno
            end = getattr(item, "end_lineno", start)
            if start <= lineno <= end:
                return node.name
    return None


def resolve_field_chain(
    root_class: str,
    chain: list[str],
    source_path: str,
    max_depth: int = 5,
) -> FieldResolution | None:
    """Resolve the type of each field in *chain* starting from *root_class*.

    For ``chain=["config", "timeout"]`` on ``root_class="Client"`` where::

        class Client:
            def __init__(self, config: Config):
                self.config = config

        class Config:
            timeout: int

    Returns::

        FieldResolution(
            leaf_type="int",
            leaf_path=None,
            chain_types=["Config", "int"],
            resolved_depth=2,
        )

    Returns ``None`` if *root_class* cannot be found at all.
    Partial results (``resolved_depth < len(chain)``) are returned when
    resolution stops mid-chain (missing annotation, class not found, etc.).

    Args:
        root_class:  Name of the class owning the first attribute in the chain.
        chain:       Attribute names to walk, e.g. ``["config", "timeout"]``.
        source_path: Python source file from which to start import resolution.
        max_depth:   Maximum chain steps to recurse (cycle + depth guard).
    """
    if not chain:
        return None

    # Cycle detection: (class_name, file_path) pairs already visited
    visited: set[tuple[str, str]] = set()

    current_class = root_class
    current_path = os.path.realpath(source_path)
    chain_types: list[str] = []

    for depth, field_name in enumerate(chain):
        if depth >= max_depth:
            break

        # Locate the current class
        found = _find_class(current_class, current_path)
        if found is None:
            break

        class_def, resolved_path = found
        key = (current_class, resolved_path)
        if key in visited:
            break  # cycle
        visited.add(key)

        # Determine the type of this field
        field_type = _get_field_type(class_def, field_name)
        if field_type is None:
            break

        chain_types.append(field_type)

        # Unwrap Optional/Union/forward-refs before deciding if it's terminal
        unwrapped = _strip_annotation(field_type)

        # If this is a terminal builtin, we're done
        if _is_builtin_type(unwrapped):
            return FieldResolution(
                leaf_type=unwrapped,
                leaf_path=None,
                chain_types=chain_types,
                resolved_depth=len(chain_types),
            )

        # Handle dotted qualification: "models.Config" → "Config"
        if "." in unwrapped:
            unwrapped = unwrapped.rsplit(".", 1)[-1]

        # Continue from the resolved class file
        current_class = unwrapped
        current_path = resolved_path

    if not chain_types:
        return None

    leaf_type = _strip_annotation(chain_types[-1])
    leaf_path = current_path if not _is_builtin_type(leaf_type) else None

    return FieldResolution(
        leaf_type=leaf_type,
        leaf_path=leaf_path,
        chain_types=chain_types,
        resolved_depth=len(chain_types),
    )
