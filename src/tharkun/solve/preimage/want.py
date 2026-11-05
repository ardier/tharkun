"""What a caller wants of a call's RESULT ([internal ref]): the demand vocabulary of the preimage visitor.

A :class:`Want` is a predicate over one value, written in a small closed vocabulary so that it is data (hashable,
printable, comparable), can be transferred backward through an operation exactly when the operation allows it, and can
be EVALUATED natively on a candidate (:meth:`Want.holds`) -- the forward check every preimage candidate must pass before
it is yielded (``visitor.py``).  ``holds`` runs only CPython's own builtin str / int / re code on the value: nothing here
runs a target or a user hook, and a value that is not of the builtin kind a predicate reads simply does not hold.

The vocabulary (each a frozen dataclass):

* :class:`Anything` -- every value;
* :class:`Eq` / :class:`OneOf` -- equal to one of the given builtin values (exact type: ``True`` is not ``1``);
* :class:`Truth` -- ``bool(value) is flag`` for a value whose truth is builtin (str, int, bool, None, list, tuple, a Match);
* :class:`IsNone` -- ``(value is None) is flag`` (``PAT.match(s) is not None``);
* :class:`Compare` -- ``value <op> const`` for an int or a str (``len(x) > 320``, ``x.find('@') == 1``);
* :class:`Length` -- ``lo <= len(value) <= hi`` (``hi`` None: unbounded) for a str, list or tuple;
* :class:`Prefix` / :class:`Suffix` / :class:`Contains` -- str-only structural demands;
* :class:`Member` -- ``re.compile(pattern, flags).<how>(value) is not None`` (``how`` is ``match``, ``search`` or
  ``fullmatch``), checked only when ``regex_members.admitted`` admits the pattern at that length (a work bound or a killable probe, so a
  forward check can never be a catastrophic backtrack; an inadmissible check does not hold);
* :class:`IsInstance` / :class:`IsSubclass` -- NOMINAL type demands ([internal ref]): the value's class has the given class in its MRO /
  the value is a class with it in its MRO (``isinstance(p, X)`` / ``issubclass(p, X)`` of a guard).  Read off ``type.__mro__``
  without calling ``__instancecheck__`` / ``__subclasscheck__``, so an ABC registration or a metaclass hook does not count and
  nothing a user class defines runs;
* :class:`Items` -- a demand on a list/tuple result: its length and a want per index (``h, t = s.split('@')``);
* :class:`All` / :class:`AnyOf` / :class:`Not` -- conjunction, disjunction, negation.

``seeds()`` gives a few values that satisfy the want BY CONSTRUCTION when that is cheap (an Eq's value, a prefix, a
regex's shortest members); the visitor starts its search from them.  They are still forward-checked like everything
else.  SOLVING by ``docs/LAYERS.md``: stdlib and ``tharkun.solve`` only.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Iterator

__all__ = ["Want", "Anything", "ANY", "Eq", "OneOf", "CaseMap", "Truth", "IsNone", "Compare", "Length", "Prefix", "Suffix",
           "Contains", "Member", "Items", "All", "AnyOf", "Not", "IsInstance", "IsSubclass", "is_builtin_value"]

_SCALARS = (str, int, bool, float, type(None), bytes)
_CMP = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq,
        "!=": operator.ne}


def is_builtin_value(value: Any, depth: int = 0) -> bool:
    """An exact builtin scalar, or a list/tuple/dict of them (depth-bounded): a value whose comparison, hashing, truth and
    length run no user code.  ``type(value) is X`` -- never ``isinstance`` -- so a subclass with hooks is not one."""
    if any(type(value) is t for t in _SCALARS):
        return True
    if (type(value) is list or type(value) is tuple) and depth < 4 and len(value) <= 4096:
        return all(is_builtin_value(v, depth + 1) for v in value)
    if type(value) is dict and depth < 4 and len(value) <= 4096:       # an exact dict of builtins (json.loads results)
        return all(is_builtin_value(k, depth + 1) and is_builtin_value(v, depth + 1) for k, v in value.items())
    return False


class Want:
    """A predicate over one value.  Subclasses define :meth:`holds` and, where cheap, :meth:`seeds`."""

    def holds(self, value: Any) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def seeds(self) -> Iterator[Any]:
        """Values that satisfy the want by construction, simplest first (possibly none)."""
        return iter(())

    def __and__(self, other: "Want") -> "Want":
        return All.of(self, other)


@dataclass(frozen=True)
class Anything(Want):
    def holds(self, value: Any) -> bool:
        return True


ANY = Anything()


@dataclass(frozen=True)
class Eq(Want):
    value: Any

    def holds(self, value: Any) -> bool:
        return is_builtin_value(value) and type(value) is type(self.value) and value == self.value

    def seeds(self):
        yield self.value


@dataclass(frozen=True)
class OneOf(Want):
    values: tuple

    def holds(self, value: Any) -> bool:
        return is_builtin_value(value) and any(type(value) is type(v) and value == v for v in self.values)

    def seeds(self):
        yield from self.values


@dataclass(frozen=True)
class CaseMap(Want):
    """The complete demand through a builtin case map; samples cannot enumerate mixed-case preimages."""

    method: str
    want: Want

    def holds(self, value: Any) -> bool:
        return (type(value) is str and len(value) <= 4096
                and self.method in ("lower", "upper", "casefold", "swapcase", "title", "capitalize")
                and self.want.holds(getattr(str, self.method)(value)))

    def seeds(self):
        for value in self.want.seeds():
            if type(value) is str and len(value) <= 4096:
                for variant in (value, value.lower(), value.upper(), value.swapcase(), value.title(), value.capitalize()):
                    if self.holds(variant):
                        yield variant


@dataclass(frozen=True)
class Truth(Want):
    flag: bool

    def holds(self, value: Any) -> bool:
        import re
        if is_builtin_value(value) or type(value) is re.Match:
            return bool(value) is self.flag
        return False

    def seeds(self):
        if self.flag:
            yield from ("a", "0", " ")
        else:
            yield ""


@dataclass(frozen=True)
class IsNone(Want):
    flag: bool

    def holds(self, value: Any) -> bool:
        return (value is None) is self.flag


@dataclass(frozen=True)
class Compare(Want):
    op: str
    const: Any

    def __post_init__(self):
        if self.op not in _CMP:
            raise ValueError(f"unknown comparison {self.op!r}")

    def holds(self, value: Any) -> bool:
        if not (is_builtin_value(value) and is_builtin_value(self.const)):
            return False
        if type(value) is bool or type(self.const) is bool:
            ok = type(value) is type(self.const) or self.op in ("==", "!=")
        else:
            ok = (type(value) in (int, float) and type(self.const) in (int, float)) or type(value) is type(self.const)
        if not ok:
            return self.op == "!=" and type(value) is not type(self.const)
        try:
            return bool(_CMP[self.op](value, self.const))
        except TypeError:
            return False

    def seeds(self):
        c = self.const
        if type(c) is int:
            yield from (v for v in (c, c + 1, c - 1) if self.holds(v))


@dataclass(frozen=True)
class Length(Want):
    lo: int = 0
    hi: int | None = None

    def holds(self, value: Any) -> bool:
        if type(value) not in (str, list, tuple, bytes, set, frozenset, dict):
            return False
        n = len(value)
        return n >= self.lo and (self.hi is None or n <= self.hi)

    def seeds(self):
        # a str of each boundary length (a hint like every seed: the visitor forward-checks what it builds from it)
        for n in dict.fromkeys((self.lo, self.lo + 1, self.hi if self.hi is not None else self.lo)):
            if self.holds("a" * n) and n <= 4096:
                yield "a" * n


@dataclass(frozen=True)
class Prefix(Want):
    text: str

    def holds(self, value: Any) -> bool:
        return type(value) is str and value.startswith(self.text)

    def seeds(self):
        yield self.text


@dataclass(frozen=True)
class Suffix(Want):
    text: str

    def holds(self, value: Any) -> bool:
        return type(value) is str and value.endswith(self.text)

    def seeds(self):
        yield self.text


@dataclass(frozen=True)
class Contains(Want):
    text: str

    def holds(self, value: Any) -> bool:
        return type(value) is str and self.text in value

    def seeds(self):
        yield self.text


@dataclass(frozen=True)
class Member(Want):
    pattern: str
    flags: int = 0
    how: str = "match"

    def __post_init__(self):
        if self.how not in ("match", "search", "fullmatch"):
            raise ValueError(f"unknown regex method {self.how!r}")

    def holds(self, value: Any) -> bool:
        if type(value) is not str:
            return False
        from tharkun.solve.preimage import regex_members
        return regex_members.native_member(self.pattern, self.flags, self.how, value) is True

    def seeds(self):
        from tharkun.solve.preimage import regex_members
        yield from regex_members.members(self.pattern, self.flags, how=self.how, limit=8)


def _mro(cls: Any) -> tuple:
    """``cls.__mro__`` read through ``type``'s own descriptor: no ``__getattribute__`` of a metaclass runs."""
    try:
        return type.__dict__["__mro__"].__get__(cls, type(cls))
    except TypeError:
        return ()


def _classes(cls: Any) -> tuple:
    classes = cls if type(cls) is tuple else (cls,)
    if not classes or not all(isinstance(c, type) for c in classes):
        raise ValueError(f"a type demand names a class or a tuple of classes, not {cls!r}")
    return classes


@dataclass(frozen=True)
class IsInstance(Want):
    """``value`` is an instance of ``cls`` (a class, or a tuple of classes for any of them), NOMINALLY: ``cls`` is in the MRO of
    ``type(value)``.  ``isinstance`` itself is not called (an ABC's ``register`` or a metaclass ``__instancecheck__`` would run user
    code): a virtual subclass does not hold."""

    cls: Any

    def __post_init__(self):
        _classes(self.cls)

    def holds(self, value: Any) -> bool:
        mro = _mro(type(value))
        return any(c in mro for c in _classes(self.cls))

    def seeds(self):
        # a builtin class builds its own empty value with no user code; any other class is built by Avengers, not here
        for c in _classes(self.cls):
            if c.__module__ == "builtins" and c not in (object, type, BaseException):
                try:
                    v = c()
                except TypeError:
                    continue
                if self.holds(v):
                    yield v


@dataclass(frozen=True)
class IsSubclass(Want):
    """``value`` is a class with ``cls`` (a class, or a tuple of classes for any of them) in its MRO; the class itself counts, as
    ``issubclass`` has it.  Nominal like :class:`IsInstance`: no ``__subclasscheck__`` runs."""

    cls: Any

    def __post_init__(self):
        _classes(self.cls)

    def holds(self, value: Any) -> bool:
        if type not in _mro(type(value)):
            return False
        mro = _mro(value)
        return any(c in mro for c in _classes(self.cls))

    def seeds(self):
        yield from _classes(self.cls)


@dataclass(frozen=True)
class Items(Want):
    """A list/tuple result: ``length`` (a Want on the int length, or None) and ``(index, Want)`` pairs."""

    length: Want | None = None
    items: tuple = ()

    def holds(self, value: Any) -> bool:
        if type(value) not in (list, tuple):
            return False
        if self.length is not None and not self.length.holds(len(value)):
            return False
        for i, w in self.items:
            if not -len(value) <= i < len(value) or not w.holds(value[i]):
                return False
        return True


@dataclass(frozen=True)
class All(Want):
    parts: tuple

    @staticmethod
    def of(*parts: Want) -> Want:
        flat = []
        for p in parts:
            if isinstance(p, All):
                flat.extend(p.parts)
            elif not isinstance(p, Anything):
                flat.append(p)
        if not flat:
            return ANY
        return flat[0] if len(flat) == 1 else All(tuple(flat))

    def holds(self, value: Any) -> bool:
        return all(p.holds(value) for p in self.parts)

    def seeds(self):
        # a seed of one part is a seed of the conjunction only if it satisfies the others: checked here, cheaply
        for p in self.parts:
            for s in p.seeds():
                if self.holds(s):
                    yield s
        # strings composed from the structural parts: prefix + middle + suffix, padded to the length bounds, and
        # regex members of an exact length when a Length pins it
        pres = [p.text for p in self.parts if isinstance(p, Prefix)]
        sufs = [p.text for p in self.parts if isinstance(p, Suffix)]
        mids = [p.text for p in self.parts if isinstance(p, Contains)]
        lens = [p for p in self.parts if isinstance(p, Length)]
        lo = max([p.lo for p in lens], default=0)
        pre = max(pres, key=len, default="")
        suf = max(sufs, key=len, default="")
        for mid in ["", *mids]:
            base = pre + mid + suf
            for pad in ("", "a" * max(0, lo - len(base)), "a"):
                cand = pre + mid + pad + suf
                if self.holds(cand):
                    yield cand
        members = [p for p in self.parts if isinstance(p, Member)]
        exact = [p.lo for p in lens if p.hi == p.lo]
        if members and exact:
            from tharkun.solve.preimage import regex_members
            for m in regex_members.members(members[0].pattern, members[0].flags, how=members[0].how, limit=4,
                                           length=exact[0]):
                if self.holds(m):
                    yield m


@dataclass(frozen=True)
class AnyOf(Want):
    parts: tuple

    def holds(self, value: Any) -> bool:
        return any(p.holds(value) for p in self.parts)

    def seeds(self):
        for p in self.parts:
            yield from p.seeds()


@dataclass(frozen=True)
class Not(Want):
    inner: Want

    def holds(self, value: Any) -> bool:
        return not self.inner.holds(value)

    def seeds(self):
        for s in ("", "a", "0", " ", "\n"):
            if self.holds(s):
                yield s
