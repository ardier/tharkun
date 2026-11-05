"""The kind vocabulary of type inference (docs/TYPEINFER.md "Kinds").

A kind is a ``find_inputs`` draw spec, nested exactly as docs/INPUTS.md "Kinds and draw domains" spells it
(``{"kind": "list", "of": {"kind": "int"}}``, ``one_of`` with ``of``, ``literal`` with ``values``, ``int`` with
``min``/``max`` ...), holding Python values (the normalized form ``inputs.py::_spec`` returns).  This module turns
an annotation (:func:`annotation_kind`, the table of docs/INPUTS.md "Mapping a Python annotation onto a kind") or a
value (:func:`value_kind`) into a kind, and a kind into a plain type name (:func:`type_name`) or into
``tharkun.strategies`` constraints (:func:`strategy_constraints`).

Stdlib only; it reads ASTs and values and executes nothing.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Callable, Mapping

#: the kinds ``find_inputs`` draws: the one list, read by ``inputs.py::_spec`` ([internal ref]) and by :func:`drawable`
BUILT_KINDS = frozenset({"int", "bool", "float", "complex", "decimal", "fraction", "str", "bytes", "bytearray",
                         "none", "list", "tuple", "dict", "set", "frozenset", "one_of", "literal"})



def str_or_int() -> dict:
    """Owner ruling 4: what nothing types is drawn as -- ``str | int`` (a fresh dict each call)."""
    return {"kind": "one_of", "of": [{"kind": "str"}, {"kind": "int"}]}


#: Default strategy-row work sizes, not domains. A regex that requires more text bypasses the text sample cut
#: (docs/INPUTS.md "Strategy rows"); a declared max_len remains authoritative.
ROW_MAX_LEN, ROW_MAX_TEXT = 8, 32

#: annotation names that spell one kind (bare builtins, ``typing`` aliases, stdlib value types); ported from
#: Eregion's ``tharkun_backend.py::_NAME_KIND``, plus ``Callable``
_NAME_KIND = {
    "int": "int", "float": "float", "bool": "bool", "str": "str", "bytes": "bytes", "None": "none",
    "NoneType": "none", "bytearray": "bytearray", "complex": "complex",
    "list": "list", "List": "list", "Sequence": "list", "tuple": "tuple", "Tuple": "tuple",
    "dict": "dict", "Dict": "dict", "Mapping": "dict", "set": "set", "Set": "set",
    "frozenset": "frozenset", "FrozenSet": "frozenset",
    "Decimal": "decimal", "decimal.Decimal": "decimal", "Fraction": "fraction",
    "fractions.Fraction": "fraction", "date": "date", "datetime.date": "date",
    "Callable": "callable", "Any": "any",
    "Iterable": "list", "Collection": "list", "MutableSequence": "list", "MutableMapping": "dict",
}
_UNION_NAMES = ("Optional", "Union")
_COLLECTIONS = ("list", "tuple", "set", "frozenset")


def _bare(kind: str) -> dict:
    """A kind with nothing said about its parts: a collection's parts are ``str | int`` (docs/INPUTS.md: bare
    ``list`` is ``of`` a ``one_of`` of primitives)."""
    if kind in _COLLECTIONS:
        return {"kind": kind, "of": str_or_int()}
    if kind == "dict":
        return {"kind": "dict", "key": str_or_int(), "value": str_or_int()}
    return {"kind": kind}


def _union(members: list) -> dict:
    flat: list = []
    for spec in members:
        for arm in spec["of"] if spec["kind"] == "one_of" else [spec]:
            if arm not in flat:
                flat.append(arm)
    return flat[0] if len(flat) == 1 else {"kind": "one_of", "of": flat}


def optional(spec: dict) -> dict:
    """``spec`` with a ``None`` arm: the kind :func:`annotation_kind` gives ``Optional[X]`` for X's kind."""
    return _union([spec, {"kind": "none"}])


def typing_imports(tree: ast.Module) -> dict[str, str]:
    """The local names a module binds to ``typing`` (or ``typing_extensions``, the same names) or to one of its names:
    ``import typing as T`` -> ``{"T": "typing"}``, ``from typing import List as L`` -> ``{"L": "typing.List"}``.  Read
    from the module's own imports, so a name that is not bound to typing there (a user module called ``T``) is not in it."""
    found: dict[str, str] = {}
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Import):
            found.update({a.asname: "typing" for a in stmt.names if a.asname and a.name in _TYPING_MODULES})
        elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module in _TYPING_MODULES:
            found.update({a.asname or a.name: f"typing.{a.name}" for a in stmt.names if a.name != "*"})
    return found


_TYPING_MODULES = ("typing", "typing_extensions")


def _resolve(text: str, imports: Mapping[str, str]) -> str:
    head, dot, rest = text.partition(".")
    return imports[head] + dot + rest if head in imports else text


def annotation_kind(annotation: ast.expr | str, aliases: Mapping[str, ast.expr] | None = None,
                    imports: Mapping[str, str] | None = None) -> dict:
    """The kind an annotation spells (an AST node, or its text).  Never raises: an annotation it cannot name is
    ``any``, a class is ``{"kind": "object", "class": <text>}`` (docs/INPUTS.md: refused by ``find_inputs`` as
    ``object_param_not_built``).  Ported from Eregion's ``tharkun_backend.py::_annotation_spec``.

    ``aliases`` maps a name to the type expression a module binds it to (``JsonData = Union[str, bytes]``, from
    ``params.py::_aliases``); such a name is read as the expression.  A name that leads back to itself is ``any``.
    ``imports`` (:func:`typing_imports` of the annotation's module) says which names are typing's: ``T.List[str]`` is
    ``List[str]`` when the module did ``import typing as T``; without it only the ``typing.`` / ``t.`` spellings are."""
    return _kind(annotation, aliases or {}, (), imports or {})


def _kind(annotation: ast.expr | str, aliases: Mapping[str, ast.expr], active: tuple, imports: Mapping[str, str]) -> dict:
    if isinstance(annotation, str):
        try:
            annotation = ast.parse(annotation.strip(), mode="eval").body
        except SyntaxError:
            return {"kind": "any"}
    node = annotation

    def sub(child: ast.expr | str) -> dict:
        return _kind(child, aliases, active, imports)

    if isinstance(node, ast.Constant):
        if node.value is None:
            return {"kind": "none"}
        if isinstance(node.value, str):  # a forward reference: the text is the annotation
            try:
                return sub(ast.parse(node.value, mode="eval").body)
            except SyntaxError:
                return {"kind": "object", "class": node.value}
        return {"kind": "any"}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):  # X | Y
        return _union([sub(node.left), sub(node.right)])
    if isinstance(node, (ast.Name, ast.Attribute)):
        text = ast.unparse(node)
        if text in aliases:
            return {"kind": "any"} if text in active else _kind(aliases[text], aliases, active + (text,), imports)
        spelled = _resolve(text, imports)
        short = spelled.rsplit(".", 1)[-1] if spelled.startswith(("typing.", "t.", "builtins.")) else spelled
        kind = _NAME_KIND.get(short) or _NAME_KIND.get(spelled)
        return _bare(kind) if kind else {"kind": "object", "class": text}
    if isinstance(node, ast.Subscript):
        base = _resolve(ast.unparse(node.value), imports)
        short = base.rsplit(".", 1)[-1] if base.startswith(("typing.", "t.")) else base
        args = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        if short in _UNION_NAMES:
            union = _union([sub(a) for a in args])
            return optional(union) if short == "Optional" else union
        if short == "Literal":
            try:
                return {"kind": "literal", "values": [ast.literal_eval(a) for a in args]}
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                return {"kind": "any"}  # ``Literal[Color.RED]``: not a value this reader can know
        kind = _NAME_KIND.get(short)
        if kind in ("list", "set", "frozenset") and len(args) == 1:
            return {"kind": kind, "of": sub(args[0])}
        if kind == "tuple":
            if len(args) == 2 and isinstance(args[1], ast.Constant) and args[1].value is Ellipsis:
                return {"kind": "tuple", "of": sub(args[0])}
            return {"kind": "tuple", "items": [sub(a) for a in args]}
        if kind == "dict" and len(args) == 2:
            return {"kind": "dict", "key": sub(args[0]), "value": sub(args[1])}
        if kind is not None:
            return _bare(kind)
        return {"kind": "object", "class": ast.unparse(node)}
    return {"kind": "any"}


_VALUE_KIND = {bool: "bool", int: "int", float: "float", complex: "complex", str: "str", bytes: "bytes",
               bytearray: "bytearray", type(None): "none"}


def kind_key(spec: dict) -> str:
    """A deterministic identity of a kind (dedupe and order; a literal may hold bytes)."""
    return json.dumps(spec, sort_keys=True, default=repr)


#: the deepest nesting :func:`value_kind` follows (``inputs.py::MAX_SPEC_DEPTH``); deeper parts are ``str | int``
_MAX_DEPTH = 32


def _join(values: Any, depth: int) -> dict:
    if depth >= _MAX_DEPTH:  # a cyclic or very deep value
        return str_or_int()
    arms = {kind_key(k): k for k in (value_kind(v, depth + 1) for v in values)}
    if not arms:
        return str_or_int()
    ordered = [arms[k] for k in sorted(arms)]
    return ordered[0] if len(ordered) == 1 else {"kind": "one_of", "of": ordered}


def value_kind(value: Any, depth: int = 0) -> dict:
    """The kind of one value (a default literal, an observed runtime value): its class, with a collection's parts
    joined over its elements (an empty one's are ``str | int``).  A class outside the vocabulary is an
    ``object`` kind naming it."""
    cls = type(value)
    if cls in _VALUE_KIND:
        return {"kind": _VALUE_KIND[cls]}
    if (cls.__module__, cls.__name__) in (("decimal", "Decimal"), ("fractions", "Fraction")):
        return {"kind": cls.__name__.lower()}
    if cls is dict:
        return {"kind": "dict", "key": _join(value.keys(), depth), "value": _join(value.values(), depth)}
    if cls in (list, tuple, set, frozenset):
        return {"kind": cls.__name__, "of": _join(value, depth)}
    return {"kind": "object", "class": cls.__qualname__}


def _parts(spec: dict) -> list:
    kind = spec["kind"]
    if kind == "one_of":
        return list(spec["of"])
    if kind == "dict":
        return [spec["key"], spec["value"]]
    if kind == "tuple" and "items" in spec:
        return list(spec["items"])
    return [spec["of"]] if kind in _COLLECTIONS else []


def drawable(spec: dict) -> bool:
    """Can ``find_inputs`` draw this kind (every part of it a built kind)?"""
    return spec.get("kind") in BUILT_KINDS and all(drawable(p) for p in _parts(spec))


_TYPE_NAME = {"none": "None", "decimal": "Decimal", "fraction": "Fraction", "any": "Any"}


def type_name(spec: dict) -> str:
    """The plain type name of a kind, as ``strategies.py::strategy_for_type`` and ``condition.infer_var_type``
    spell one (``int``, ``list``, ``None``, ``Decimal``, a class's text): an ``Optional[X]`` is X's name, any other
    union ``Any``, a literal its values' type."""
    kind = spec["kind"]
    if kind == "object":
        return spec.get("class", "object")
    if kind == "one_of":
        arms = [arm for arm in spec["of"] if arm["kind"] != "none"]
        return type_name(arms[0]) if len(arms) == 1 else "Any"
    if kind == "literal":
        names = {type(v).__name__ for v in spec["values"]}
        return names.pop() if len(names) == 1 else "Any"
    return _TYPE_NAME.get(kind, kind)


#: ``prefer(leaf, of, convert, hashable)``: the values a leaf of kind ``leaf`` prefers (``of``: the literal type
#: to take them from, ``convert`` applied to each)
Prefer = Callable[[dict, str, Callable[[Any], Any], bool], list]


def strategy_constraints(spec: dict, hashable: bool = False, prefer: Prefer | None = None) -> dict:
    """``spec`` as a ``tharkun.strategies`` constraint dict (docs/INPUTS.md "Strategy rows"): the kind IS the contract
    of the value. Lengths normally use the row work sizes (:data:`ROW_MAX_LEN`, :data:`ROW_MAX_TEXT`); an
    string's regex may require a longer row. ``prefer`` names the values a leaf tries first (``inputs.py``: the
    program's own literals of its type); ``hashable`` says the value is a set element or a dict key."""
    kind = spec["kind"]

    def own(c: dict, of: str, convert: Callable[[Any], Any] = lambda v: v, leaf: dict = spec) -> dict:
        values = prefer(leaf, of, convert, hashable) if prefer is not None else []
        return dict(c, prefer=values) if values else c

    def length(cap: int) -> dict:
        return {"min": 0, "max": min(spec.get("max_len", cap), cap)}

    finite = spec.get("finite", False)
    if kind == "int":
        lo, hi = spec.get("min"), spec.get("max")
        if lo is None and hi is None:
            return own({"type": "int", "unbounded": True}, "int")
        return own({"type": "int", "value_range": [lo, hi]}, "int")
    if kind in ("bool", "none", "fraction"):
        return {"type": kind}
    if kind == "float":
        return own({"type": "float", "unbounded": True, "allow_nan": not finite, "allow_infinity": not finite},
                   "float")
    if kind == "complex":
        part = {"type": "float", "unbounded": True, "allow_nan": not finite, "allow_infinity": not finite}
        return {"type": "complex", "part_constraints": own(part, "float", leaf={"kind": "float", "finite": finite})}
    if kind == "decimal":
        return own({"type": "decimal", "allow_nan": not finite, "allow_infinity": not finite,
                    "allow_snan": not hashable}, "decimal")
    if kind in ("str", "bytes"):
        # a str's ``regex`` / ``how`` / ``flags`` / ``contains`` are a HINT ([internal ref]: its rows are members)
        hint = {k: (list(spec[k]) if k == "contains" else spec[k])
                for k in ("regex", "how", "flags", "contains") if k in spec} if kind == "str" else {}
        if "regex" in hint:
            import re

            try:
                flags = hint.get("flags", 0)
                if isinstance(flags, (list, tuple)):
                    bits = 0
                    for name in flags:
                        bits |= getattr(re, name)
                    flags = bits
                # A sampling cut cannot exclude every member of the required language.
                if re._parser.parse(hint["regex"], flags & ~re.DEBUG).getwidth()[0] > ROW_MAX_TEXT:
                    row = {"type": kind, **hint}
                    if "max_len" in spec:
                        row["length"] = {"min": 0, "max": spec["max_len"]}
                    return own(row, kind)
            except (re.error, TypeError, ValueError, OverflowError, RecursionError, AttributeError):
                pass  # The strategy owns validation and the named refusal for an unsupported hint.
        cap = max(ROW_MAX_TEXT, sum(len(s) for s in hint.get("contains", ()))) if "regex" not in hint else ROW_MAX_TEXT
        return own({"type": kind, "length": length(cap), **hint}, kind)
    if kind == "bytearray":
        return own({"type": "bytearray", "length": length(ROW_MAX_TEXT)}, "bytes", bytearray)
    if kind == "literal":
        return {"type": "literal", "values": list(spec["values"])}
    if kind == "one_of":
        return {"type": "one_of", "options": [strategy_constraints(a, hashable, prefer) for a in spec["of"]]}
    if kind == "tuple" and "items" in spec:
        return {"type": "tuple", "items": [strategy_constraints(s, hashable, prefer) for s in spec["items"]]}
    if kind == "dict":
        return {"type": "dict", "length": length(ROW_MAX_LEN),
                "key_constraints": strategy_constraints(spec["key"], True, prefer),
                "value_constraints": strategy_constraints(spec["value"], False, prefer)}
    extra = {"contains": list(spec["contains"])} if kind == "list" and "contains" in spec else {}
    cap = max(ROW_MAX_LEN, len(extra.get("contains", ())))
    return {"type": kind, "length": length(cap),
            "element_constraints": strategy_constraints(spec["of"], hashable or kind in ("set", "frozenset"), prefer),
            **extra}
