"""Closed, non-executing syntax checks for verifier ``check_type`` hints.

This helper only recognizes names whose meaning the existing Resolver proves.
It never evaluates an annotation or calls a user object.  The returned effect
tags describe operations the verifier may perform while checking a value.
"""
from __future__ import annotations

import ast


_SCALAR_HINTS = frozenset({"int", "bool", "float", "str", "bytes", "complex"})
_CONTAINER_HINTS = frozenset({"list", "tuple", "dict", "set", "frozenset"})


def builtin_hint_shape(node: ast.AST, resolver):
    """Describe an exact builtin hint without evaluating it; unknown syntax refuses.

    The recursive tuples are inert descriptions, never runtime GenericAlias objects.
    Callers must additionally check the current values of the resolved names.
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return ("class", "NoneType")
    if isinstance(node, ast.Name) and resolver.is_builtin(node):
        if node.id in _SCALAR_HINTS | _CONTAINER_HINTS:
            return ("class", node.id)
    if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and resolver.is_builtin(node.value) and node.value.id in _CONTAINER_HINTS):
        return None
    family = node.value.id
    args = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
    variadic = (family == "tuple" and len(args) == 2
                and isinstance(args[1], ast.Constant) and args[1].value is Ellipsis)
    if variadic:
        args = args[:1]
    if ((family == "dict" and len(args) != 2)
            or (family in {"list", "set", "frozenset"} and len(args) != 1)):
        return None
    children = tuple(builtin_hint_shape(arg, resolver) for arg in args)
    if any(child is None for child in children):
        return None
    return ("generic", family, children, variadic)


def check_type_arguments(node: ast.Call):
    """Bind the verifier's two ordinary parameters without evaluating either argument."""
    if len(node.args) > 2 or any(isinstance(arg, ast.Starred) for arg in node.args):
        return None
    bound = dict(zip(("value", "hint"), node.args))
    for keyword in node.keywords:
        if keyword.arg not in {"value", "hint"} or keyword.arg in bound:
            return None
        bound[keyword.arg] = keyword.value
    return (bound["value"], bound["hint"]) if set(bound) == {"value", "hint"} else None


def check_type_contract():
    """The tested checker and None return on which the closed transfers depend."""
    from .contracts import get

    contract = get("_sv_verifier.check_type")
    if (contract is None or not contract.tested or contract.returns is None
            or not contract.returns.tested or contract.returns.type != "None"):
        return None
    return contract


def _other_binding(tree: ast.AST, local: str) -> bool:
    """Whether a spelling has any binding besides its one resolved import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == local and isinstance(node.ctx, (ast.Store, ast.Del)):
            return True
        if isinstance(node, ast.arg) and node.arg == local:
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == local:
            return True
        if isinstance(node, ast.ExceptHandler) and node.name == local:
            return True
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == local:
            return True
        if isinstance(node, (ast.Global, ast.Nonlocal)) and local in node.names:
            return True
    return False


def _typing_name(node: ast.AST, resolver, attr: str) -> bool:
    if isinstance(node, ast.Name):
        imported = resolver.imports.names.get(node.id)
        if imported != ("typing", attr) or _other_binding(resolver.tree, node.id):
            return False
        return sum(1 for n in ast.walk(resolver.tree) if isinstance(n, ast.alias)
                   and (n.asname or n.name) == node.id and n.name == attr) == 1
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        alias = node.value.id
        if node.attr != attr or resolver.imports.modules.get(alias) != "typing":
            return False
        if _other_binding(resolver.tree, alias):
            return False
        return sum(1 for n in ast.walk(resolver.tree) if isinstance(n, ast.alias)
                   and n.name == "typing" and (n.asname or n.name) == alias) == 1
    return False


def closed_hint_effects(node: ast.AST, resolver, plain_class=None) -> frozenset[str] | None:
    """Return verifier effects for a closed hint, or ``None`` when unproved.

    Effects are ``iterate`` for sequence hints and ``items`` for mapping hints.
    Only built-in scalar/container classes, ``None``, and the closed typing
    forms Any/Union/Optional are admitted.  Tuple- and mapping-generic arities
    match the verifier's supported runtime shapes.

    ``plain_class`` (default None): a predicate for a bare ``Name`` that the caller has proved names a plain
    user class (``isinstance(x, it)`` is ``type.__instancecheck__``); such a name is then a closed hint with
    no effects, alone or inside ``Union`` / ``Optional`` / ``|``.
    """
    shape = builtin_hint_shape(node, resolver)
    if shape is not None:
        def effects_of(hint):
            if hint[0] == "class":
                return frozenset()
            effects = frozenset({"items" if hint[1] == "dict" else "iterate"})
            for child in hint[2]:
                effects |= effects_of(child)
            return effects
        return effects_of(shape)
    if isinstance(node, ast.Constant) and node.value is None:
        return frozenset()
    if isinstance(node, ast.Name):
        if resolver.is_builtin(node) and node.id in _SCALAR_HINTS | _CONTAINER_HINTS:
            return frozenset()
        if node.id == "None" and resolver.is_builtin(node):
            return frozenset()
        if node.id in {"type", "object"} and resolver.is_builtin(node):
            return frozenset()      # ``isinstance(v, type)``: consults types only once no isinstance hook exists (the caller's plain_isinstance)
        if _typing_name(node, resolver, "Any"):
            return frozenset()
        if plain_class is not None and plain_class(node):
            return frozenset()
        return None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "type" and resolver.is_builtin(node.func)
            and len(node.args) == 1 and not node.keywords and isinstance(node.args[0], ast.Constant) and node.args[0].value is None):
        return frozenset()          # ``type(None)``: the class NoneType
    if isinstance(node, ast.Attribute) and _typing_name(node, resolver, "Any"):
        return frozenset()
    if isinstance(node, ast.Attribute) and plain_class is not None and plain_class(node):
        return frozenset()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = closed_hint_effects(node.left, resolver, plain_class)
        right = closed_hint_effects(node.right, resolver, plain_class)
        return None if left is None or right is None else left | right
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Name) and resolver.is_builtin(base) and base.id in _CONTAINER_HINTS:
            family = base.id
        elif _typing_name(base, resolver, "Union"):
            family = "Union"
        elif _typing_name(base, resolver, "Optional"):
            family = "Optional"
        else:
            return None
        args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if family == "dict" and len(args) != 2:
            return None
        if family in {"list", "set", "frozenset"} and len(args) != 1:
            return None
        if family == "tuple":
            if not args or (len(args) == 2 and isinstance(args[1], ast.Constant)
                            and args[1].value is Ellipsis):
                args = args[:1]
            elif any(isinstance(a, ast.Constant) and a.value is Ellipsis for a in args):
                return None
        if family == "Optional" and len(args) != 1:
            return None
        if family == "Union" and len(args) < 2:
            return None
        effects: frozenset[str] = frozenset()
        for arg in args:
            child = closed_hint_effects(arg, resolver, plain_class)
            if child is None:
                return None
            effects |= child
        if family in {"list", "tuple", "set", "frozenset"}:
            effects |= {"iterate"}
        elif family == "dict":
            effects |= {"items"}
        return effects
    return None


def _unrebound_class(tree: ast.AST, name: str):
    """The one module-level ``class name`` of ``tree`` that nothing rebinds or shadows anywhere in the file (a parameter, a
    local, an import, a ``global`` ... of that spelling refuses), or ``None``."""
    classes = [c for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name == name]
    if len(classes) != 1 or classes[0] not in tree.body:
        return None
    for n in ast.walk(tree):
        if ((isinstance(n, ast.Name) and n.id == name and not isinstance(n.ctx, ast.Load))
                or (isinstance(n, ast.arg) and n.arg == name)
                or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
                or (isinstance(n, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and n.name == name)
                or (isinstance(n, ast.alias) and name in {n.asname, n.name.partition(".")[0]})
                or (isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names)):
            return None
    return classes[0]


def _plain(cls: ast.ClassDef, resolver, seen=()) -> bool:
    """No keyword (a metaclass), no decorator, every base ``object`` or a plain class of the same file: then
    ``isinstance(x, cls)`` is ``type.__instancecheck__``, which consults only the types (a user ``__class__`` is a program
    dunder the census scans anyway)."""
    if cls in seen or cls.keywords or cls.decorator_list or getattr(cls, "type_params", ()):
        return False
    for base in cls.bases:
        if not isinstance(base, ast.Name):
            return False
        if base.id == "object" and resolver.is_builtin(base):
            continue
        parent = _unrebound_class(resolver.tree, base.id)
        if parent is None or not _plain(parent, resolver, (*seen, cls)):
            return False
    return True


def plain_class_hint(node: ast.AST, resolver) -> bool:
    """Is a ``check_type`` hint a plain program class: a name (``Person``), a class nested in one (``A.B``) or a class of a
    source module of the task (``mod.A``)?  A hint like that adds no effect to the check beyond ``isinstance``."""
    if isinstance(node, ast.Name):
        cls = _unrebound_class(resolver.tree, node.id)
        return cls is not None and _plain(cls, resolver)
    if not isinstance(node, ast.Attribute):
        return False
    base = node.value
    child = resolver.source_module(base)
    if child is not None:
        cls = _unrebound_class(child.tree, node.attr)
        return cls is not None and _plain(cls, child)
    if isinstance(base, ast.Name):
        outer = _unrebound_class(resolver.tree, base.id)
        if outer is None or not _plain(outer, resolver):
            return False
        inner = [st for st in outer.body if isinstance(st, ast.ClassDef) and st.name == node.attr]
        others = [st for st in outer.body if st not in inner and any(
            (isinstance(n, ast.Name) and n.id == node.attr and not isinstance(n.ctx, ast.Load))
            or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == node.attr)
            or isinstance(n, ast.alias) and node.attr in {n.asname, n.name} for n in ast.walk(st))]
        return len(inner) == 1 and not others and _plain(inner[0], resolver)
    return False
