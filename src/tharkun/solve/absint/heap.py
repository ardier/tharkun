"""Heap objects of the absint abstract state (docs/ABSINT.md).

Addresses are ``(site, 'r')`` (the most recent object allocated at that site:
strong updates) or ``(site, 's')`` (the summary of the older ones: weak
updates) -- the recency abstraction.  Objects are immutable: an update builds a
new one.  The values they hold are :class:`~tharkun.solve.absint.aval.V`.
"""

from __future__ import annotations

from typing import Iterable

from .aval import BOT, V


class Obj:
    """A heap object.  Objects are immutable: an update builds a new one."""

    kind = "obj"
    __slots__ = ()

    def _all_slots(self):
        out = []
        for klass in type(self).__mro__:
            out.extend(getattr(klass, "__slots__", ()))
        return out

    def replace(self, **kw):
        new = object.__new__(type(self))
        for k in self._all_slots():
            object.__setattr__(new, k, kw.get(k, getattr(self, k)))
        return new

    def values(self) -> Iterable[V]:
        return ()

    def zvars(self) -> set[str]:
        return set()


class ListObj(Obj):
    """A list.  ``length`` is a zone variable (recent) or a ``(lo, hi)`` interval
    (summary); ``elem`` over-approximates every element (``BOT``: none yet).

    **Segmentation** (``pre`` is not None, recent lists only): the zone variable
    ``P.<sid>`` (:func:`seg_var`) splits the indices: every element at an index
    ``< P`` is in ``pre``, every element at an index ``>= P`` is in ``elem``.
    ``P >= 0`` always.  This is what proves "every slot of ``[None] * n`` was
    written by ``for i in range(n)``": each store at index exactly ``P`` moves
    the boundary up by one."""

    kind = "list"
    __slots__ = ("length", "elem", "pre")

    def __init__(self, length, elem: V, pre: V | None = None):
        self.length = length
        self.elem = elem
        self.pre = pre

    def values(self):
        return (self.elem,) if self.pre is None else (self.elem, self.pre)

    def zvars(self):
        out = {self.length} if isinstance(self.length, str) else set()
        if self.pre is not None and isinstance(self.length, str):
            out.add(seg_var(self.length))
        return out


class TupleObj(ListObj):
    """A tuple: ``items`` when the arity is known, else ``length`` + ``elem``."""

    kind = "tuple"
    __slots__ = ("items",)

    def __init__(self, items, length=None, elem: V = BOT):
        self.items = items
        self.length = length
        self.elem = elem
        self.pre = None

    def values(self):
        return tuple(self.items) if self.items is not None else (self.elem,)


class ArrayObj(ListObj):
    """``array.array(typecode, ...)`` (docs/ABSINT.md Slice 3, [internal ref]).  Reuses
    ``ListObj``'s ``length``/``elem`` machinery verbatim -- indexing, slicing, ``len``
    and iteration are IDENTICAL to a list's, so every generic helper that reads those
    two fields (``len_lin``, ``index_ok``, ``elem_at``, ``all_elems``, ``store_elem``,
    ``set_len``, ``update_obj``) works on an array unmodified.  ``pre`` (segmentation)
    is never set for an array (always ``None``): the ``[None] * n``-style prefix proof
    is list-only future work, not required by any known SV-Benchmarks array use.

    ``typecode`` is the one field a list doesn't have, and it is what every array-only
    operation (``m_array_array``, ``array_method``, the ``subscript_store``/``binop``
    array branches in interp.py) checks: real ``array`` enforces a TypeError/
    OverflowError on every element write that ``list`` does not, and the SAME typecode
    on both sides of ``+``/``extend``/construction-from-another-array."""

    kind = "array"
    __slots__ = ("typecode",)

    def __init__(self, length, elem: V, typecode: str, pre: V | None = None):
        self.length = length
        self.elem = elem
        self.pre = pre
        self.typecode = typecode


def seg_var(length_var: str) -> str:
    """The segmentation boundary variable of the list whose length is ``length_var``."""
    return "P." + length_var[2:]


class InstObj(Obj):
    """An instance.  ``attrs``: name -> ``(V, definite, ver)`` where ``ver`` counts
    the attribute's writes since the object was last refreshed (None: unknown).
    ``qf``: frozenset of attribute names -> ``(lo, hi, why)``: for every object this
    abstract object stands for, the real sum of the squares of those (float)
    attributes, when none is NaN, lies in ``[lo, hi]`` -- a per-object invariant
    (docs/ABSINT.md, "float invariants")."""

    kind = "inst"
    __slots__ = ("cls", "attrs", "qf", "exc_args")

    def __init__(self, cls, attrs: dict, qf: dict | None = None, exc_args: V | None = None):
        self.exc_args = exc_args  # BaseException.__new__ storage, separate from the instance dict
        self.cls = cls  # a class address
        self.attrs = attrs
        self.qf = qf if qf is not None else {}

    def values(self):
        return tuple(e[0] for e in self.attrs.values()) + (() if self.exc_args is None else (self.exc_args,))


class ExceptionObj(Obj):
    """A canonical builtin exception, with GC-visible constructor arguments."""

    kind = 'exception'
    __slots__ = ('cls', 'args')

    def __init__(self, cls: str, args: V):
        self.cls, self.args = cls, args

    def values(self):
        return (self.args,)


class ClassObj(Obj):
    kind = "class"
    __slots__ = ("name", "node", "bases", "attrs", "slots", "abstracts")

    def __init__(self, name, node, bases, attrs, slots, abstracts=None):
        self.name = name
        self.node = node
        self.bases = bases  # tuple of class addresses
        self.attrs = attrs  # name -> V
        self.slots = slots  # tuple of names, or None (has __dict__)
        self.abstracts = abstracts  # None: type; frozenset: closed ABCMeta abstract names

    def values(self):
        return tuple(self.attrs.values())


class NamedTupleClassObj(ClassObj):
    """A closed stdlib-generated tuple type, never an executed target class."""
    __slots__ = ("fields", "defaults")

    def __init__(self, name, node, fields, defaults):
        super().__init__(name, node, (), {}, ())
        self.fields, self.defaults = fields, defaults

    def values(self):
        return (*super().values(), *self.defaults)


class NamedTupleObj(TupleObj):
    __slots__ = ("type_v",)

    def __init__(self, items, type_v):
        super().__init__(items)
        self.type_v = type_v

    def values(self):
        return (*super().values(), self.type_v)


class FuncObj(Obj):
    kind = "func"
    __slots__ = ("node", "defaults", "kwdefaults", "qualname", "flavor", "module_key", "captures", "capture_frames", "abstract")

    def __init__(self, node, defaults, kwdefaults, qualname, flavor="plain", module_key="", captures=None, capture_frames=None):
        self.node = node
        self.defaults = defaults  # tuple of V
        self.kwdefaults = kwdefaults  # dict name -> V
        self.qualname = qualname
        self.flavor = flavor  # plain | staticmethod | classmethod
        # [internal ref] (import boundary): "" is the program's own module (today's exact behaviour,
        # unchanged); a function read in from another file (read-and-recurse) carries THAT
        # file's own key, so a free name in its body resolves against ITS OWN globals, not
        # the caller's -- see interp.py's module_key plumbing.
        self.module_key = module_key
        self.captures = {} if captures is None else captures
        self.capture_frames = {} if capture_frames is None else capture_frames
        self.abstract = False

    def values(self):
        return tuple(self.defaults) + tuple(self.kwdefaults.values()) + tuple(self.captures.values())


class BoundMethod(Obj):
    """A user function bound to ``self_v``."""

    kind = "bmeth"
    __slots__ = ("func", "self_v")

    def __init__(self, func, self_v: V):
        self.func = func  # a FuncObj address
        self.self_v = self_v

    def values(self):
        return (self.self_v,)


class BuiltinMethod(Obj):
    """A builtin method (``list.insert`` ...) bound to receiver ``recv``."""

    kind = "bimeth"
    __slots__ = ("name", "recv")

    def __init__(self, name, recv: V):
        self.name = name
        self.recv = recv

    def values(self):
        return (self.recv,)


class RangeObj(Obj):
    """``range(lo, hi, step)``: endpoints are zone variables or summary intervals."""

    kind = "range"
    __slots__ = ("lo", "hi", "step")

    def __init__(self, lo: str, hi: str, step: int):
        self.lo = lo
        self.hi = hi
        self.step = step

    def zvars(self):
        return {value for value in (self.lo, self.hi) if isinstance(value, str)}



class DictObj(Obj):
    """A dict whose keys are hashable constants.

    ``items``: ``{key: V}`` for the keys it definitely has (a key once present
    stays present: deletion is not modelled and aborts).  ``exact``: no other
    key can be present.  Otherwise ``okeys`` / ``ovals`` over-approximate the
    keys and values beyond ``items`` (``okeys`` is a V of the possible keys)."""

    kind = "dict"
    __slots__ = ("items", "exact", "okeys", "ovals")

    def __init__(self, items: dict, exact: bool = True, okeys: V = BOT, ovals: V = BOT):
        self.items = items
        self.exact = exact
        self.okeys = okeys
        self.ovals = ovals

    def values(self):
        return tuple(self.items.values()) + (self.okeys, self.ovals)


class DefaultDictObj(DictObj):
    """An exact ``collections.defaultdict`` with a modeled factory.

    It keeps ``kind == "dict"`` so the existing dictionary shape transfers are
    shared, while the distinct Python class and ``factory`` tag retain the
    subclass identity.  Only ``"list"`` and ``None`` are admitted by the
    constructor model; arbitrary factories execute user code and stay refused.
    """

    __slots__ = ("factory",)

    def __init__(self, items: dict, exact: bool, okeys: V, ovals: V, factory: str | None):
        super().__init__(items, exact, okeys, ovals)
        self.factory = factory


class SetObj(Obj):
    """A ``set`` or a ``frozenset`` (docs/ABSINT.md, "Sets").

    ``length`` is the interval ``(lo, hi)`` of the number of distinct elements (``hi`` is ``None``
    when unbounded; it is never a zone variable, so a set has no recent/summary difference in
    shape, only in how an update joins).  ``elem`` over-approximates EVERY element: whatever the
    set holds is one of the values ``elem`` allows (``BOT``: none yet, which :meth:`make` keeps
    equal to ``length == (0, 0)``).  Element values are the ones that were inserted: two equal
    values (``1``, ``1.0``, ``True``) are one element and either object may be the one kept, so
    ``elem`` joins both.  ``frozen`` is True for a ``frozenset``: the two kinds differ in the
    methods they have, in hashability, and in what an operator returns, never in their elements."""

    kind = "set"
    __slots__ = ("length", "elem", "frozen")

    def __init__(self, length, elem: V, frozen: bool = False):
        self.length = length
        self.elem = elem
        self.frozen = frozen

    @classmethod
    def make(cls, lo, hi, elem: V, frozen: bool = False) -> "SetObj":
        """A set of ``lo`` to ``hi`` elements whose elements are all in ``elem``, with the facts that
        must agree made to: a length is never negative, no element means length 0 (and the other
        way round: a set that cannot hold an element has none to describe), and a lower bound
        never exceeds the upper one."""
        lo = 0 if lo is None or lo < 0 else lo
        if hi is not None and hi < lo:
            lo = hi = max(hi, 0)
        if elem.is_bot() or hi == 0:
            return cls((0, 0), BOT, frozen)
        return cls((lo, hi), elem, frozen)

    def values(self):
        return (self.elem,)


class DictView(Obj):
    """``d.keys()`` / ``d.values()`` / ``d.items()``: a live view of dict ``dict_addrs``."""

    kind = "dictview"
    __slots__ = ("dicts", "which")

    def __init__(self, dicts: frozenset, which: str):
        self.dicts = dicts
        self.which = which

class TextWrapperObj(Obj):
    """Canonical textwrap.TextWrapper settings, mutable through heap updates."""

    kind = 'textwrapper'
    __slots__ = ('attrs',)

    def __init__(self, attrs):
        self.attrs = attrs

    def values(self):
        return tuple(self.attrs.values())


class StringIOObj(Obj):
    """Text or binary program stream, with exact WorldObject facts when available."""

    kind = "stringio"
    __slots__ = ("content", "stream", "position", "binary", "closed")

    def __init__(self, content, stream=None, position=None, binary=False, closed=frozenset({False})):
        self.content, self.stream, self.position = content, stream, position
        self.binary, self.closed = binary, closed

    def values(self):
        return (self.content,)


class ReadFileObj(Obj):
    """Remaining text and possible closed states of a world file ([internal ref])."""

    kind = "readfile"
    __slots__ = ("content", "closed")

    def __init__(self, content, closed=frozenset({False})):
        self.content, self.closed = content, closed

    def values(self):
        return (self.content,)


class OutFileObj(Obj):
    """A file opened by a write-only ``open()`` ([internal ref], ``contracts/rules_fileio.py``).

    It carries nothing but its kind: ``binary`` is True for a binary mode (an
    ``io.BufferedWriter``), False for a text mode (an ``io.TextIOWrapper``).  What was written
    cannot reach the program again except through a read, and a handle opened for output has
    none, so no content is tracked; whether it is closed is not tracked either, because the
    only exceptions that depends on (``ValueError`` on a closed file, ``OSError``) are not
    property classes and a path that raises them simply ends.  An allocation site is one
    ``open`` call node, whose mode is a constant, so one site is one kind."""

    kind = "outfile"
    __slots__ = ("binary",)

    def __init__(self, binary: bool):
        self.binary = binary


class GeneratorFrame:
    """An immutable-by-convention saved local environment at one continuation."""

    __slots__ = ("bindings", "maybe_unbound")

    def __init__(self, bindings, maybe_unbound=()):
        self.bindings = bindings
        self.maybe_unbound = frozenset(maybe_unbound)


class GeneratorObj(Obj):
    """Source generator states; None is unstarted, negative one is closed."""

    kind = "generator"
    __slots__ = ("function", "states")

    def __init__(self, function, bindings=None, *, states=None):
        self.function = function
        self.states = {None: GeneratorFrame(bindings)} if states is None else states

    @property
    def bindings(self):
        return self.states[None].bindings

    def values(self):
        return (self.function, *(value for frame in self.states.values()
                                 for value in frame.bindings.values()))


class IteratorObj(Obj):
    """Builtin iterator over a known source; consumption is overapproximated."""
    # ponytail: no cursor precision; add cursor intervals for exhaustion proofs.
    kind = "iterator"
    __slots__ = ("source",)

    def __init__(self, source):
        self.source = source

    def values(self):
        return (self.source,)


class ZipObj(Obj):
    """A native zip: sources stay live; remaining consumption is overapproximated."""
    kind = "iterator"
    __slots__ = ("sources", "strict", "fresh")

    def __init__(self, sources, strict=False, fresh=True):
        self.sources = tuple(sources)
        self.strict = strict
        self.fresh = fresh  # True guarantees no consumption; False makes no cursor claim.

    def values(self):
        return self.sources


class EnumerateObj(Obj):
    """Native enumerate with a live source and immutable construction-time start bounds."""
    kind = "iterator"
    __slots__ = ("source", "start")

    def __init__(self, source, start):
        self.source = source
        self.start = start

    def values(self):
        return self.source, self.start
