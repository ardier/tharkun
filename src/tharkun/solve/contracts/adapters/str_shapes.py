"""The argument binding of the ``str`` methods whose rules reach a user hook only through an argument ([internal ref], ticket [internal ref]).

``solve/contracts/rules_str.py`` states, per method, that every property class but ``TypeError`` can be raised only by a
USER hook run on an argument: the ``__index__`` of a position (``start``, ``end``, ``maxsplit``, ``count``, ``width``,
``tabsize``), the ``__iter__``/``__next__`` of the iterable ``join`` consumes, the ``__bool__`` of ``keepends``.  ``TypeError`` is
the call's own logic: an argument of a kind the method does not take, or a call that does not bind.

This module is the engine-neutral half of "no user hook runs": which parameters a method has, which of them an engine may pass
by keyword, and the KIND an argument must have for the call to run no user code and raise no property class.  An engine maps a
kind to its own values (``absint/visitor.py::_str_kind_ok`` for the abstract interpreter; the CPython three-way tests in
``tests/contracts/test_str_shapes.py`` for concrete ones) and calls :func:`bind` to learn which value answers to which kind.

Kinds (an argument of ANY other shape -- an instance of a user class, a subclass of a builtin, a float where an int is
taken, an unknown value -- is outside the admitted shape, and the engine keeps the contract's unproven claims):

* ``f`` a one-character ``str``;            * ``s`` a ``str``;                         * ``S`` a ``str`` or a ``tuple`` of ``str`` (``prefix``/``suffix``);
* ``c`` a ``str`` or ``None``;             * ``i`` an ``int`` or a ``bool``;
* ``n`` an ``int``, a ``bool`` or ``None``; * ``L`` a ``list`` or ``tuple`` all of whose elements are ``str`` (``join``).

``OverflowError``, ``MemoryError`` and ``ValueError`` are not property classes and are not claimed away: an engine that admits a
call has argued the seven property classes only (``NON_PROPERTY_RAISES`` says which methods can still raise one of them, and
``tests/contracts/test_str_shapes.py`` measures it).  The signatures are those of CPython 3.14 (``str.<m>.__text_signature__``);
a keyword is admitted only where 3.14 takes it by keyword, and a name outside that is a call that does not bind.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    """One parameter after the receiver: its name, its admitted kind, whether a call must pass it, and whether it may be keyword."""

    name: str
    kind: str
    required: bool = True
    keyword: bool = False


KINDS = {
    "s": "a str",
    "f": "a one-character str",
    "S": "a str or a tuple of str",
    "c": "a str or None",
    "i": "an int or a bool",
    "n": "an int, a bool or None",
    "L": "a list or a tuple of str",
}

#: method -> its parameters after the receiver, in call order (positional order; ``keyword`` marks the ones 3.14 takes by name)
SIGNATURES: dict[str, tuple[Param, ...]] = {
    "center": (Param("width", "i"), Param("fillchar", "f", False)),
    "ljust": (Param("width", "i"), Param("fillchar", "f", False)),
    "rjust": (Param("width", "i"), Param("fillchar", "f", False)),
    "startswith": (Param("prefix", "S"), Param("start", "n", False), Param("end", "n", False)),
    "endswith": (Param("suffix", "S"), Param("start", "n", False), Param("end", "n", False)),
    "count": (Param("sub", "s"), Param("start", "n", False), Param("end", "n", False)),
    "find": (Param("sub", "s"), Param("start", "n", False), Param("end", "n", False)),
    "rfind": (Param("sub", "s"), Param("start", "n", False), Param("end", "n", False)),
    "index": (Param("sub", "s"), Param("start", "n", False), Param("end", "n", False)),
    "rindex": (Param("sub", "s"), Param("start", "n", False), Param("end", "n", False)),
    "split": (Param("sep", "c", False, True), Param("maxsplit", "i", False, True)),
    "rsplit": (Param("sep", "c", False, True), Param("maxsplit", "i", False, True)),
    "strip": (Param("chars", "c", False),),
    "lstrip": (Param("chars", "c", False),),
    "rstrip": (Param("chars", "c", False),),
    "replace": (Param("old", "s"), Param("new", "s"), Param("count", "i", False, True)),
    "partition": (Param("sep", "s"),),
    "rpartition": (Param("sep", "s"),),
    "removeprefix": (Param("prefix", "s"),),
    "removesuffix": (Param("suffix", "s"),),
    "join": (Param("iterable", "L"),),
    "zfill": (Param("width", "i"),),
    "splitlines": (Param("keepends", "i", False, True),),
    "expandtabs": (Param("tabsize", "i", False, True),),
}

#: the methods that can still raise a NON-property class on an admitted call, and the classes (measured on CPython 3.14.4):
#: ``ValueError`` for an empty separator or an absent substring; ``OverflowError`` for a position wider than the C integer the method
#: converts it to (``maxsplit`` and ``count`` at 2**63, ``width`` at 2**63, ``tabsize`` at 2**31); ``MemoryError`` for a ``width`` or
#: ``tabsize`` whose result cannot be allocated.  ``start``/``end`` of ``startswith``/``endswith``/``count``/``find``/``index`` clamp:
#: they never raise.  Any str result can in principle fail to allocate; the table names the methods where ONE small argument asks
#: for the allocation.  None of these is a property class, and an engine that models a rule call inside a ``try`` must refuse it
#: (``Analyzer.visit_rule``: the handlers would have to see them).
NON_PROPERTY_RAISES: dict[str, frozenset[str]] = {
    "center": frozenset({"OverflowError", "MemoryError"}),
    "ljust": frozenset({"OverflowError", "MemoryError"}),
    "rjust": frozenset({"OverflowError", "MemoryError"}),
    "partition": frozenset({"ValueError"}),
    "rpartition": frozenset({"ValueError"}),
    "index": frozenset({"ValueError"}),
    "rindex": frozenset({"ValueError"}),
    "split": frozenset({"ValueError", "OverflowError"}),
    "rsplit": frozenset({"ValueError", "OverflowError"}),
    "replace": frozenset({"OverflowError"}),
    "zfill": frozenset({"OverflowError", "MemoryError"}),
    "expandtabs": frozenset({"OverflowError", "MemoryError"}),
}

#: the lower bound of an admitted call's int result, or of its list result's length (CPython: ``count`` and ``index`` are never
#: negative, ``find`` returns -1 for absent, and ``split(sep)`` with a str separator has at least one part)
INT_RESULT_LOWER = {"count": 0, "index": 0, "rindex": 0, "find": -1, "rfind": -1}


def bind(method: str, args, kwargs) -> dict[str, tuple[str, object]] | None:
    """Bind the call ``<receiver>.<method>(*args, **kwargs)`` the way CPython's argument parser does.

    ``args`` are the positional arguments WITHOUT the receiver, ``kwargs`` the keywords.  Returns ``{name: (kind, value)}`` for
    every parameter the call passes, in parameter order, or ``None`` when the method has no entry here or the call does not bind
    (too many positional arguments, a keyword the method does not take by keyword, a parameter given twice, a required
    parameter missing, an unknown keyword).  ``None`` is not "raises": the engine simply keeps the contract's claims."""
    params = SIGNATURES.get(method)
    if params is None:
        return None
    args = tuple(args)
    if len(args) > len(params):
        return None
    bound = {p.name: (p.kind, v) for p, v in zip(params, args)}
    by_name = {p.name: p for p in params}
    for name, value in dict(kwargs).items():
        p = by_name.get(name)
        if p is None or not p.keyword or name in bound:
            return None
        bound[name] = (p.kind, value)
    if any(p.required and p.name not in bound for p in params):
        return None
    return {p.name: bound[p.name] for p in params if p.name in bound}


__all__ = ["Param", "KINDS", "SIGNATURES", "NON_PROPERTY_RAISES", "INT_RESULT_LOWER", "bind"]
