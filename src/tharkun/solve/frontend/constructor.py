"""Real object construction for analyzed codebase classes.

When the condition solver resolves a parameter's type to a class defined in
the analyzed codebase, this module builds a real instance via ``__init__``
with synthesized arguments, instead of falling back to ``SimpleNamespace``.

Usage::

    from tharkun.solve.frontend.constructor import construct_instance

    obj = construct_instance("/path/to/module.py", "MyClass")
    # Returns a real MyClass instance, or SimpleNamespace as fallback
"""

from __future__ import annotations

import ast
import copy
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from tharkun.solve.frontend.import_resolver import load_class_def


# ---------------------------------------------------------------------------
# Default value synthesis for __init__ parameters
# ---------------------------------------------------------------------------

_ANNOTATION_DEFAULTS: dict[str, Any] = {
    "int": 0,
    "float": 0.0,
    "str": "",
    "bytes": b"",
    "bool": False,
    "list": [],
    "dict": {},
    "set": set(),
    "tuple": (),
    "None": None,
    "Optional": None,
    "Any": None,
}


_MISSING = object()


@dataclass(frozen=True)
class ConstructorParameter:
    """One statically discovered ``__init__`` parameter.

    ``kind`` is one of ``positional_only``, ``positional_or_keyword``,
    ``var_positional``, ``keyword_only``, or ``var_keyword``.  A declared
    default is recorded even when it is not a Python literal; consumers can
    distinguish that case with ``default_is_literal`` and still inspect its
    source text.
    """

    name: str
    annotation: str | None
    kind: str
    has_default: bool = False
    default_is_literal: bool = False
    default: Any = None
    default_source: str | None = None


@dataclass(frozen=True)
class ConstructorTemplate:
    """A static, execution-free description of a class constructor."""

    source_path: str
    class_name: str
    parameters: tuple[ConstructorParameter, ...]


def _synthesize_arg(annotation: str | None, default: Any = _MISSING) -> Any:
    """Synthesize a value for an __init__ parameter.

    Uses the annotation if available, otherwise the default value.
    """
    if default is not _MISSING:
        return default

    if annotation is None:
        return None

    # Strip Optional[...] wrapper
    ann = annotation.strip()
    if ann.startswith("Optional["):
        return None
    if ann.endswith("| None"):
        return None

    # Check direct type matches
    for type_name, val in _ANNOTATION_DEFAULTS.items():
        if ann == type_name or ann.endswith(f".{type_name}"):
            # Return a copy for mutable types
            if isinstance(val, (list, dict, set)):
                return type(val)()
            return val

    # List[...], Dict[...], etc.
    if ann.startswith("list[") or ann.startswith("List["):
        return []
    if ann.startswith("dict[") or ann.startswith("Dict["):
        return {}
    if ann.startswith("set[") or ann.startswith("Set["):
        return set()
    if ann.startswith("tuple[") or ann.startswith("Tuple["):
        return ()

    return None


def _annotation_text(annotation: ast.expr | None) -> str | None:
    if annotation is None:
        return None
    try:
        return ast.unparse(annotation)
    except Exception:
        return None


def _parameter(
    arg: ast.arg,
    kind: str,
    default_node: ast.expr | None = None,
    *,
    has_default: bool = False,
) -> ConstructorParameter:
    default_source = None
    default_is_literal = False
    default: Any = None
    if has_default and default_node is not None:
        try:
            default_source = ast.unparse(default_node)
        except Exception:
            pass
        try:
            default = ast.literal_eval(default_node)
            default_is_literal = True
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            pass

    return ConstructorParameter(
        name=arg.arg,
        annotation=_annotation_text(arg.annotation),
        kind=kind,
        has_default=has_default,
        default_is_literal=default_is_literal,
        default=default,
        default_source=default_source,
    )


def _signature_comment_types(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str | None] | None:
    """The argument types a PEP 484 signature comment (``# type: (A, B) -> R``) gives, or ``None``.

    Aligned to EVERY parameter of the ``def`` in order (positional, ``*args``,
    keyword-only, ``**kwargs``).  PEP 484 leaves the receiver out of a method's
    comment, so a list one shorter than the parameters is aligned after the
    first; any other length is not trusted.  ``(...) -> R`` gives no types.
    """
    comment = getattr(node, "type_comment", None)
    if not comment:
        return None
    try:
        parsed = ast.parse(comment, mode="func_type")
    except SyntaxError:
        return None
    if len(parsed.argtypes) == 1 and isinstance(parsed.argtypes[0], ast.Constant) and parsed.argtypes[0].value is Ellipsis:
        return None
    args = node.args
    total = (
        len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
        + (args.vararg is not None) + (args.kwarg is not None)
    )
    texts = [_annotation_text(item) for item in parsed.argtypes]
    if len(texts) == total:
        return texts
    if len(texts) == total - 1:
        return [None, *texts]
    return None


def function_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    drop_receiver: bool = False,
) -> tuple[ConstructorParameter, ...]:
    """The parameters of one ``def``, statically, in declaration order.

    ``drop_receiver`` drops the first positional parameter whatever it is named
    (``self`` of a method, ``cls`` of a classmethod); a staticmethod or a free
    function keeps it.  Shared by the constructor template and Avengers'
    method-call templates ([internal ref]), so both read a signature the same way.

    A parameter with no annotation takes its type from a PEP 484 type comment
    when the source has one -- the per-argument ``# type: A`` or the function's
    ``# type: (A, B) -> R`` -- provided the tree was parsed with
    ``type_comments=True`` (``import_resolver.py::parse_with_type_comments``).
    An annotation always wins over a comment.
    """
    parameters: list[ConstructorParameter] = []
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    default_start = len(positional) - len(args.defaults)
    commented = _signature_comment_types(node)
    position = 0

    def typed(parameter: ConstructorParameter, arg: ast.arg, index: int) -> ConstructorParameter:
        if parameter.annotation is not None:
            return parameter
        text = getattr(arg, "type_comment", None) or (commented[index] if commented else None)
        if text is None:
            return parameter
        if not isinstance(text, str):
            return parameter
        try:
            ast.parse(text, mode="eval")
        except SyntaxError:
            return parameter
        return ConstructorParameter(
            name=parameter.name,
            annotation=text.strip(),
            kind=parameter.kind,
            has_default=parameter.has_default,
            default_is_literal=parameter.default_is_literal,
            default=parameter.default,
            default_source=parameter.default_source,
        )

    for index, arg in enumerate(positional):
        here = position
        position += 1
        if drop_receiver and index == 0:
            continue
        has_default = index >= default_start
        default_node = args.defaults[index - default_start] if has_default else None
        kind = (
            "positional_only"
            if index < len(args.posonlyargs)
            else "positional_or_keyword"
        )
        parameters.append(
            typed(_parameter(arg, kind, default_node, has_default=has_default), arg, here)
        )

    if args.vararg is not None:
        parameters.append(typed(_parameter(args.vararg, "var_positional"), args.vararg, position))
        position += 1

    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(
            typed(
                _parameter(
                    arg,
                    "keyword_only",
                    default_node,
                    has_default=default_node is not None,
                ),
                arg,
                position,
            )
        )
        position += 1

    if args.kwarg is not None:
        parameters.append(typed(_parameter(args.kwarg, "var_keyword"), args.kwarg, position))

    return tuple(parameters)


def _constructor_parameters(cls_def: ast.ClassDef) -> tuple[ConstructorParameter, ...]:
    """Extract a complete, immutable parameter template from *cls_def*."""
    for node in cls_def.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "__init__":
            continue
        # ``self`` by name, as before: an ``__init__`` whose receiver is spelled
        # differently keeps reporting it, which no caller relies on changing.
        positional = [*node.args.posonlyargs, *node.args.args]
        return function_parameters(
            node, drop_receiver=bool(positional) and positional[0].arg == "self"
        )

    # No __init__ in the source. It may still have one: a dataclass writes it
    # from the annotated fields in its body, and reporting an empty template for
    # such a class says "takes no arguments", which is wrong in exactly the way
    # that stops anything from varying its state.
    generated = _dataclass_parameters(cls_def)
    if generated is not None:
        return generated

    # Otherwise the class really does use the inherited/default constructor.
    return ()


_DATACLASS_DECORATORS = frozenset({"dataclass", "attrs", "define", "frozen", "mutable"})


def _decorator_name(node: ast.expr) -> str | None:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _decorator_says_init(node: ast.expr) -> bool:
    """False when a decorator explicitly asks for ``init=False``."""

    if not isinstance(node, ast.Call):
        return True
    for keyword in node.keywords:
        if keyword.arg == "init" and isinstance(keyword.value, ast.Constant):
            return bool(keyword.value.value)
    return True


def _field_call_excludes_init(node: ast.expr | None) -> bool:
    """True for ``field(init=False)`` -- a field the constructor never takes."""

    if not isinstance(node, ast.Call):
        return False
    if _decorator_name(node) not in {"field", "Field"}:
        return False
    for keyword in node.keywords:
        if keyword.arg == "init" and isinstance(keyword.value, ast.Constant):
            return not bool(keyword.value.value)
    return False


def _dataclass_parameters(
    cls_def: ast.ClassDef,
) -> tuple[ConstructorParameter, ...] | None:
    """The constructor a dataclass-style decorator will generate, or ``None``.

    Read statically from the annotated assignments in the class body, in
    declaration order, which is the order the generated ``__init__`` takes
    them.  ``ClassVar`` fields are not constructor parameters and neither is a
    ``field(init=False)``; a field whose default is a call (``field(...)``,
    ``factory(...)``) is recorded as having one without claiming to know what
    it is, exactly as an unparseable ``__init__`` default would be.
    """

    decorated = any(
        _decorator_name(item) in _DATACLASS_DECORATORS and _decorator_says_init(item)
        for item in cls_def.decorator_list
    )
    if not decorated:
        return None

    parameters: list[ConstructorParameter] = []
    for node in cls_def.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        annotation = _annotation_text(node.annotation)
        if annotation is not None and annotation.replace("typing.", "").startswith("ClassVar"):
            continue
        if _field_call_excludes_init(node.value):
            continue
        default_node = node.value
        has_default = default_node is not None
        literal: Any = None
        is_literal = False
        if has_default:
            try:
                literal = ast.literal_eval(default_node)
                is_literal = True
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                is_literal = False
        parameters.append(
            ConstructorParameter(
                name=node.target.id,
                annotation=annotation,
                kind="positional_or_keyword",
                has_default=has_default,
                default_is_literal=is_literal,
                default=literal,
                default_source=(
                    _annotation_text(default_node) if default_node is not None else None
                ),
            )
        )
    return tuple(parameters)


def declares_constructor(cls_def: ast.ClassDef) -> bool:
    """Whether *cls_def*'s OWN source says what its constructor takes.

    True for a body that defines ``__init__`` or ``__new__`` and for a
    dataclass-style decorator that generates one.  False otherwise -- the class
    inherits its constructor, or a base or metaclass makes one at run time
    (``NamedTuple``, ``collections.namedtuple``, ``Enum``, an exception or
    ``ModuleType`` subclass) -- and then the empty parameter list
    :func:`constructor_template` reports is a statement about the SOURCE, not
    about the constructor: a caller that can run the class asks it instead.
    """
    for node in cls_def.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"__init__", "__new__"}:
            return True
    return _dataclass_parameters(cls_def) is not None


def class_declares_constructor(source_path: str, class_name: str) -> bool | None:
    """:func:`declares_constructor` for a class found by (dotted) name; ``None`` when it is not found."""
    cls_def = load_class_def(os.path.realpath(source_path), class_name)
    return None if cls_def is None else declares_constructor(cls_def)


def constructor_template(
    source_path: str,
    class_name: str,
) -> ConstructorTemplate | None:
    """Return a static constructor template without executing target code.

    ``None`` means the source could not be parsed or the named class was not
    found.  A class that relies on the inherited constructor has an empty
    ``parameters`` tuple.
    """
    rp = os.path.realpath(source_path)
    cls_def = load_class_def(rp, class_name)
    if cls_def is None:
        return None
    return ConstructorTemplate(
        source_path=rp,
        class_name=class_name,
        parameters=_constructor_parameters(cls_def),
    )


def _extract_init_params(cls_def: ast.ClassDef) -> list[tuple[str, str | None, Any]]:
    """Extract __init__ parameters as (name, annotation_str, default_or_MISSING).

    Skips 'self'.
    """
    params: list[tuple[str, str | None, Any]] = []
    for parameter in _constructor_parameters(cls_def):
        if parameter.kind in {"var_positional", "var_keyword"}:
            continue
        default = (
            parameter.default
            if parameter.has_default and parameter.default_is_literal
            else _MISSING
        )
        params.append((parameter.name, parameter.annotation, default))
    return params


def construct_instance(
    source_path: str,
    class_name: str,
    overrides: dict[str, Any] | None = None,
) -> Any:
    """Construct a real instance of a class from the analyzed codebase.

    Reads the class's ``__init__`` signature, synthesizes arguments,
    and attempts to construct a real instance. Falls back to
    ``SimpleNamespace`` if construction fails.

    Args:
        source_path: Path to the Python source file containing the class.
        class_name:  Name of the class to construct.
        overrides:   Optional field values to set after synthesis, overriding
                     the synthesized defaults.  E.g. ``{"timeout": 61}`` ensures
                     the constructed instance has ``instance.timeout == 61``.

    Returns:
        A real instance of the class, or ``SimpleNamespace`` as fallback.
    """
    template = constructor_template(source_path, class_name)
    if template is None:
        return SimpleNamespace(**(overrides or {}))

    # Synthesize arguments
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    fallback_attrs: dict[str, Any] = {}
    supplied_overrides = dict(overrides or {})
    for parameter in template.parameters:
        if parameter.kind in {"var_positional", "var_keyword"}:
            continue
        if parameter.name in supplied_overrides:
            value = supplied_overrides.pop(parameter.name)
        elif parameter.has_default and parameter.default_is_literal:
            value = copy.deepcopy(parameter.default)
        else:
            value = _synthesize_arg(parameter.annotation)
        fallback_attrs[parameter.name] = value

        if parameter.kind == "positional_only":
            args.append(value)
        else:
            kwargs[parameter.name] = value

    # Apply caller-supplied overrides before construction
    kwargs.update(supplied_overrides)
    fallback_attrs.update(supplied_overrides)

    # Try to load and instantiate the actual class -- through the core's one
    # loader (#0607); under ``no_targets()`` this step is skipped.
    from tharkun.solve.loader import exec_source

    module_globals = exec_source(template.source_path)
    if module_globals is not None:
        try:
            cls = module_globals.get(class_name)
            if cls is not None and isinstance(cls, type):
                return cls(*args, **kwargs)
        except Exception:
            pass

    # Fallback: SimpleNamespace with the synthesized attributes
    return SimpleNamespace(**fallback_attrs)
