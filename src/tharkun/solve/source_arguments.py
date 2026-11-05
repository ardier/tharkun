"""Bind source AST arguments with Python's signature rules, without target execution."""
from __future__ import annotations

import ast
from inspect import Parameter, Signature


def bind_source_arguments(target, positional, keywords=()):
    """Return (parameter -> AST/tuple/dict, defaulted names, refusal).

    Defaults retain their definition-time AST identity. The consumer must not
    interpret a default's free names in the caller's namespace.
    """
    if any(isinstance(value, ast.Starred) for value in positional) or any(key.arg is None for key in keywords):
        return None, frozenset(), "star/mapping argument expansion is unresolved"
    if len({key.arg for key in keywords}) != len(keywords):
        return None, frozenset(), "duplicate keyword arguments"
    args = target.args
    positionals = [*args.posonlyargs, *args.args]
    defaults = dict(zip((arg.arg for arg in positionals[len(positionals) - len(args.defaults):]), args.defaults))
    parameters = []
    for number, argument in enumerate(positionals):
        kind = Parameter.POSITIONAL_ONLY if number < len(args.posonlyargs) else Parameter.POSITIONAL_OR_KEYWORD
        parameters.append(Parameter(argument.arg, kind, default=defaults.get(argument.arg, Parameter.empty)))
    if args.vararg is not None:
        parameters.append(Parameter(args.vararg.arg, Parameter.VAR_POSITIONAL))
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(Parameter(argument.arg, Parameter.KEYWORD_ONLY,
                                    default=default if default is not None else Parameter.empty))
    if args.kwarg is not None:
        parameters.append(Parameter(args.kwarg.arg, Parameter.VAR_KEYWORD))
    try:
        signature = Signature(parameters)
        bound = signature.bind(*positional, **{key.arg: key.value for key in keywords})
    except (TypeError, ValueError) as error:
        return None, frozenset(), str(error)
    supplied = set(bound.arguments)
    bound.apply_defaults()
    defaulted = frozenset(name for name in bound.arguments if name not in supplied
                          and signature.parameters[name].default is not Parameter.empty)
    return dict(bound.arguments), defaulted, None


def bind_source_call(target, call):
    """Class attribute calls use only the descriptors certified by the source index."""
    implicit = []
    if isinstance(call.func, ast.Attribute):
        decorators = target.decorator_list
        if len(decorators) > 1 or any(not isinstance(node, ast.Name)
                                    or node.id not in ("classmethod", "staticmethod") for node in decorators):
            return None, frozenset(), "descriptor binding is unresolved"
        if decorators and decorators[0].id == "classmethod":
            implicit.append(call.func.value)
    return bind_source_arguments(target, [*implicit, *call.args], call.keywords)


def immutable_default(node):
    """A scalar/tuple literal is stable; mutable and namespace-dependent defaults are not."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return None

    def immutable(value):
        return type(value) in (int, bool, float, complex, str, bytes, type(None), type(Ellipsis)) or (
            type(value) is tuple and all(immutable(element) for element in value))

    return ast.unparse(node) if immutable(value) else None
