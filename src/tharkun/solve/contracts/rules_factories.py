"""Contracts for standard-library VALUE FACTORIES ([internal ref]): the constructors and functions that build a
ready-to-use object from literal arguments and touch nothing outside the process.

Why they are contracts.  Avengers builds an object for a parameter the project does not define (a queue, an
in-memory stream, a parsed tree, a date).  It must not name any standard-library attribute a recipe asks for -- a
recipe value is handed to the target, and a target that calls ``os.system`` runs it -- so what it MAY name is
decided by this registry, not by a table of its own: a callable is admitted when its :class:`Contract` is tested
and says no file, process, network, clock, environment or randomness effect (``adapters/avengers.py::stdlib_admission``).
A callable the registry does not cover is refused, with that reason.  The family below is the set of value
factories covered so far; adding one is one ``_FACTORIES`` row, and the tests (``tests/contracts/test_factories.py``)
sweep every row against CPython 3.14.4: each listed call returns an instance of the declared type, the call opens no
file, process or socket (an audit hook), and a call with too many arguments raises ``TypeError``.

What a row says. ``returns`` is the class returned by the listed recipe calls (the default-mode ``ast.parse``
recipes return ``ast.Module``); the callable contract separately covers all accepted modes. ``calls`` are VALID calls with
literal arguments, cheapest first (``()`` is the zero-argument call): the only calls Avengers makes, the engine
audit's sample (the first one) and what the sweep above runs.  All seven property classes are ``unknown`` and
``may_raise_other`` is true: a constructor raises on a bad argument, and no engine reads a raise claim of a factory.

SOLVING by ``docs/LAYERS.md``'s rule: stdlib and ``tharkun.solve`` only; nothing here runs a target.
"""

from __future__ import annotations

import abc
import ast
import collections
import collections.abc
import datetime
import decimal
import fractions
import io
import numbers
import os
import pathlib
import queue
import typing
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar

from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/"

#: the callables the family can call, by qualname: static imports (``solve/`` loads no module by name)
_NATIVE: dict[str, Any] = {
    "io.StringIO": io.StringIO, "io.BytesIO": io.BytesIO,
    "queue.Queue": queue.Queue, "queue.LifoQueue": queue.LifoQueue, "queue.PriorityQueue": queue.PriorityQueue,
    "queue.SimpleQueue": queue.SimpleQueue,
    "collections.OrderedDict": collections.OrderedDict, "collections.Counter": collections.Counter,
    "collections.deque": collections.deque,
    "ast.parse": ast.parse,
    "decimal.Decimal": decimal.Decimal, "fractions.Fraction": fractions.Fraction,
    "datetime.date": datetime.date, "datetime.datetime": datetime.datetime, "datetime.time": datetime.time,
    "datetime.timedelta": datetime.timedelta,
    "uuid.UUID": uuid.UUID, "typing.NewType": typing.NewType,
    "pathlib.PurePath": pathlib.PurePath, "pathlib.Path": pathlib.Path,
}


@dataclass(frozen=True)
class FactorySpec:
    """What a consumer needs to build a value: the callable, the class its call returns, the valid literal calls."""

    qualname: str
    module: str
    name: str
    returns: str
    is_class: bool
    calls: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...]


@dataclass(frozen=True)
class Factory:
    """One row: what the call returns, the valid literal calls, and the evidence."""

    returns: str
    calls: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...]
    doc: str
    cpython: str
    notes: str = ""


_ZERO: tuple[tuple[tuple[Any, ...], dict[str, Any]], ...] = (((), {}),)

#: qualname -> row.  ``returns`` is a class (its instance is what the call gives).
_FACTORIES: dict[str, Factory] = {
    "io.StringIO": Factory("io.StringIO", (((), {}), (("a\nb\n",), {})), "io.html#io.StringIO", "Modules/_io/stringio.c",
                           "an in-memory text stream; no file descriptor"),
    "io.BytesIO": Factory("io.BytesIO", (((), {}), ((b"a\nb\n",), {})), "io.html#io.BytesIO", "Modules/_io/bytesio.c",
                          "an in-memory binary stream; no file descriptor (fileno() raises UnsupportedOperation)"),
    "queue.Queue": Factory("queue.Queue", _ZERO, "queue.html#queue.Queue", "Lib/queue.py", "an unbounded FIFO queue"),
    "queue.LifoQueue": Factory("queue.LifoQueue", _ZERO, "queue.html#queue.LifoQueue", "Lib/queue.py"),
    "queue.PriorityQueue": Factory("queue.PriorityQueue", _ZERO, "queue.html#queue.PriorityQueue", "Lib/queue.py"),
    "queue.SimpleQueue": Factory("queue.SimpleQueue", _ZERO, "queue.html#queue.SimpleQueue", "Modules/_queuemodule.c"),
    "collections.OrderedDict": Factory("collections.OrderedDict", _ZERO, "collections.html#collections.OrderedDict",
                                       "Objects/odictobject.c"),
    "collections.Counter": Factory("collections.Counter", _ZERO, "collections.html#collections.Counter",
                                   "Lib/collections/__init__.py"),
    "collections.deque": Factory("collections.deque", _ZERO, "collections.html#collections.deque",
                                 "Modules/_collectionsmodule.c"),
    "ast.parse": Factory("ast.Module", ((("x = 1\n",), {}), (("",), {})), "ast.html#ast.parse", "Lib/ast.py::parse",
                         "parses source text into a tree; compiles nothing and runs nothing"),
    "decimal.Decimal": Factory("decimal.Decimal", _ZERO, "decimal.html#decimal.Decimal", "Modules/_decimal"),
    "fractions.Fraction": Factory("fractions.Fraction", _ZERO, "fractions.html#fractions.Fraction",
                                  "Lib/fractions.py"),
    "datetime.date": Factory("datetime.date", (((2000, 1, 1), {}),), "datetime.html#datetime.date",
                             "Modules/_datetimemodule.c"),
    "datetime.datetime": Factory("datetime.datetime", (((2000, 1, 1), {}),), "datetime.html#datetime.datetime",
                                 "Modules/_datetimemodule.c", "naive: no clock is read"),
    "datetime.time": Factory("datetime.time", _ZERO, "datetime.html#datetime.time", "Modules/_datetimemodule.c"),
    "datetime.timedelta": Factory("datetime.timedelta", _ZERO, "datetime.html#datetime.timedelta",
                                  "Modules/_datetimemodule.c"),
    "uuid.UUID": Factory("uuid.UUID", (((), {"int": 0}),), "uuid.html#uuid.UUID", "Lib/uuid.py",
                         "built from an integer: no randomness (uuid4 and uuid1 are not factories of this family)"),
    "typing.NewType": Factory("typing.NewType", ((("Id", int), {}),), "typing.html#typing.NewType", "Lib/typing.py::NewType",
                              "a distinct type over a builtin type (its `__supertype__`); no I/O (#1386)"),
    "pathlib.PurePath": Factory("pathlib.PurePath", _ZERO, "pathlib.html#pathlib.PurePath", "Lib/pathlib/_local.py",
                                "pure path arithmetic: no file-system access"),
    "pathlib.Path": Factory("pathlib.Path", _ZERO, "pathlib.html#pathlib.Path", "Lib/pathlib/_local.py",
                            "constructing a Path does no I/O; calling its methods does (the target's own business)"),
}

# Callable results can be broader than a factory's selected literal recipes.
_CONTRACT_RETURNS = {"ast.parse": "ast.AST"}


def _literal(value: Any) -> str:
    # a builtin class is written by its name (`int`): the one non-literal argument a row may hold (`typing.NewType('Id', int)`)
    return value.__name__ if isinstance(value, type) and value.__module__ == "builtins" else repr(value)


def literal_source(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, str]:
    """``(args, kwargs)`` as Python source text that ``eval`` reads back (a builtin class by its name)."""
    return ("(" + "".join(f"{_literal(a)}, " for a in args) + ")",
            "{" + ", ".join(f"{k!r}: {_literal(v)}" for k, v in kwargs.items()) + "}")


def call_source(qualname: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """The program ``import m; x = m.f(args)`` for one valid call (the engine audit's sample)."""
    module, _, name = qualname.rpartition(".")
    shown = [*(_literal(a) for a in args), *(f"{k}={_literal(v)}" for k, v in kwargs.items())]
    return f"import {module}\nx = {module}.{name}({', '.join(shown)})\n"


_CONTRACTS: dict[str, Contract] = {}


@register_rule
class StdlibFactory(Rule):
    """The rule of every standard-library value factory in :data:`_FACTORIES` (one parametrized family)."""

    names = tuple(sorted(_FACTORIES))
    SAMPLES = {q: call_source(q, *row.calls[0]) for q, row in _FACTORIES.items()}
    # nine positional arguments: more than any row accepts (``datetime.datetime`` takes eight)
    TWINS = {q: f"import {q.rpartition('.')[0]}\nx = {q}(0, 1, 2, 3, 4, 5, 6, 7, 8)\n" for q in _FACTORIES}
    SHAPE: ClassVar[Shape | None] = None

    @classmethod
    def contract_for(cls, qualname: str) -> Contract:
        # one object per qualname: the registry serves the rule's own contract (``test_rules_registry.py``)
        if qualname not in _CONTRACTS:
            _CONTRACTS[qualname] = cls._contract(qualname)
        return _CONTRACTS[qualname]

    @classmethod
    def _contract(cls, qualname: str) -> Contract:
        row = _FACTORIES[qualname]
        return Contract(
            qualname=qualname,
            raises=tuple(RaiseClaim(exc, "unknown") for exc in ALL_PROPERTY_CLASSES),
            may_raise_other=True,
            returns=ReturnSpec(_CONTRACT_RETURNS.get(qualname, row.returns), tested=True),
            effects=Effects(
                notes=(row.notes + "; " if row.notes else "") + "no file, process, network, clock, environment or randomness effect"),
            source=(f"{_DOCS}{row.doc}", f"CPython {row.cpython}"),
            tested=True,
            notes="a value factory (#1280): Avengers calls it with the literal arguments of `calls` only",
        )

    def native(self, *args: Any, **kwargs: Any) -> Any:
        return _NATIVE[self.qualname](*args, **kwargs)

    def spec(self) -> FactorySpec:
        """The facts a consumer builds a value from (the Avengers visitor hands this out once the contract is admitted)."""
        module, _, name = self.qualname.rpartition(".")
        row = _FACTORIES[self.qualname]
        return FactorySpec(self.qualname, module, name, row.returns, isinstance(_NATIVE[self.qualname], type), row.calls)


def resolve_returns(spec: FactorySpec) -> type | None:
    """The class *spec*'s call returns an instance of (``ast.Module``), from the modules this file imports; else ``None``."""
    module, _, name = spec.returns.rpartition(".")
    found = _NATIVE_MODULES.get(module)
    return getattr(found, name, None) if found is not None else None


_NATIVE_MODULES = {
    "io": io, "queue": queue, "collections": collections, "ast": ast, "decimal": decimal, "fractions": fractions,
    "datetime": datetime, "uuid": uuid, "pathlib": pathlib, "typing": typing,
}

#: the homes of the abstract types a parameter is annotated with, beside the modules above
_TYPE_MODULES = {**_NATIVE_MODULES, "typing": typing, "collections.abc": collections.abc, "numbers": numbers,
                 "os": os, "abc": abc}


def type_named(module: str, name: str) -> Any:
    """The type object ``module.name`` when *module* is one this file imports statically (the factories' modules and
    the homes of the abstract types: ``typing``, ``collections.abc``, ``numbers``, ``os``, ``abc``), else ``None``.
    Nothing is imported by name: ``solve/`` may not, and a result that depended on what the process had loaded
    would make a search irreproducible."""
    found: Any = _TYPE_MODULES.get(module)
    for part in name.split("."):
        found = getattr(found, part, None) if found is not None else None
    return found


__all__ = ["Factory", "FactorySpec", "StdlibFactory", "call_source", "literal_source", "resolve_returns", "type_named"]
