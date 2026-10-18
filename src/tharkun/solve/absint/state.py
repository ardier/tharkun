"""The abstract state (frames, heap, zone) and its lattice operations.

Soundness notes, checked by ``tests/test_absint.py``:

* ``join`` / ``widen`` / ``leq`` are computed per component; a symbolic bound
  ``(var, c)`` is compared in the zone of the side it came from.
* A zone variable is never dropped while a summary bound names it:
  ``State.kill_var`` first rewrites every such bound to an equal variable, or
  to the variable's constant bound, or to infinity.
* Heap objects are immutable; a state copy shares them.
"""

from __future__ import annotations

from .aval import (ANY_STR, BOT, FloatInfo, V, f_join, f_leq, f_widen)
from .heap import (ArrayObj, DefaultDictObj, DictObj, DictView, BoundMethod, BuiltinMethod, ClassObj, FuncObj, InstObj, IteratorObj, ListObj, Obj, RangeObj, SetObj, StringIOObj, TupleObj)
from .linear import (IntInfo, Lin, lin_bounds)
from . import lemmas as _lem
from .aval import _merge_why
from .zone import ZERO, Zone


# ------------------------------------------------------------ symbolic bounds
def _const_hi(b, z: Zone):
    if b is None:
        return None
    v, c = b
    if v == ZERO:
        return c
    hi = z.get(v, ZERO)
    return None if hi is None else hi + c


def _const_lo(b, z: Zone):
    if b is None:
        return None
    v, c = b
    if v == ZERO:
        return c
    lo = z.get(ZERO, v)
    return None if lo is None else c - lo


def hi_join(a, za: Zone, b, zb: Zone):
    """Least-ish upper bound of two upper bounds ``e <= a`` (in za), ``e <= b`` (in zb)."""
    if a is None or b is None:
        return None
    (va, ca), (vb, cb) = a, b
    if va == vb:
        return (va, max(ca, cb))
    cands = []
    # b expressed with va, in zb: vb - va <= d  ==> e <= va + cb + d
    d = zb.get(vb, va)
    if d is not None and va in zb.vars:
        cands.append((va, max(ca, cb + d)))
    d = za.get(va, vb)
    if d is not None and vb in za.vars:
        cands.append((vb, max(cb, ca + d)))
    if cands:
        # prefer a non-constant anchor
        cands.sort(key=lambda t: (t[0] == ZERO, t[1]))
        return cands[0]
    ha, hb = _const_hi(a, za), _const_hi(b, zb)
    if ha is None or hb is None:
        return None
    return (ZERO, max(ha, hb))


def lo_join(a, za: Zone, b, zb: Zone):
    if a is None or b is None:
        return None
    (va, ca), (vb, cb) = a, b
    if va == vb:
        return (va, min(ca, cb))
    cands = []
    # e >= vb + cb ; va - vb <= d (zb) ==> vb >= va - d ==> e >= va + cb - d
    d = zb.get(va, vb)
    if d is not None and va in zb.vars:
        cands.append((va, min(ca, cb - d)))
    d = za.get(vb, va)
    if d is not None and vb in za.vars:
        cands.append((vb, min(cb, ca - d)))
    if cands:
        cands.sort(key=lambda t: (t[0] == ZERO, -t[1]))
        return cands[0]
    la, lb = _const_lo(a, za), _const_lo(b, zb)
    if la is None or lb is None:
        return None
    return (ZERO, min(la, lb))


def hi_leq(a, za: Zone, b):
    """In za, does ``e <= a`` imply ``e <= b``?"""
    if b is None:
        return True
    if a is None:
        return False
    (va, ca), (vb, cb) = a, b
    if va == vb:
        return ca <= cb
    d = za.get(va, vb)
    return d is not None and ca + d <= cb


def lo_leq(a, za: Zone, b):
    """In za, does ``e >= a`` imply ``e >= b``?"""
    if b is None:
        return True
    if a is None:
        return False
    (va, ca), (vb, cb) = a, b
    if va == vb:
        return ca >= cb
    d = za.get(vb, va)  # vb - va <= d  ==> va >= vb - d ==> e >= vb - d + ca
    return d is not None and ca - d >= cb


# ------------------------------------------------------------------ V lattice
def int_join(a: IntInfo | None, za, b: IntInfo | None, zb) -> IntInfo | None:
    if a is None:
        return b
    if b is None:
        return a
    lin = a.lin if a.lin == b.lin else None
    if lin is not None and a.lo is None and a.hi is None and b.lo is None and b.hi is None:
        return IntInfo(lin)
    return IntInfo(lin, lo_join(a.lo, za, b.lo, zb), hi_join(a.hi, za, b.hi, zb))


def int_leq(a: IntInfo | None, za, b: IntInfo | None) -> bool:
    if a is None:
        return True
    if b is None:
        return False
    if b.lin is not None and a.lin != b.lin:
        return False
    return lo_leq(a.lo, za, b.lo) and hi_leq(a.hi, za, b.hi)


def int_widen(a: IntInfo | None, za, b: IntInfo | None, zb) -> IntInfo | None:
    if a is None:
        return b
    if b is None:
        return a
    j = int_join(a, za, b, zb)
    lo = a.lo if lo_leq(j.lo, zb, a.lo) else None
    hi = a.hi if hi_leq(j.hi, zb, a.hi) else None
    return IntInfo(j.lin if a.lin == j.lin else None, lo, hi)


def _str_join(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a == ANY_STR or b == ANY_STR:
        return ANY_STR
    u = a | b
    return u if len(u) <= 8 else ANY_STR


def _str_leq(a, b):
    if a is None:
        return True
    if b is None:
        return False
    if b == ANY_STR:
        return True
    if a == ANY_STR:
        return False
    return a <= b


def vjoin(a: V, za: Zone, b: V, zb: Zone) -> V:
    if a is b:
        return a
    return V(
        i=int_join(a.i, za, b.i, zb),
        b=(a.b or frozenset()) | (b.b or frozenset()) or None,
        f=f_join(a.f, b.f),
        s=_str_join(a.s, b.s),
        n=a.n or b.n,
        refs=a.refs | b.refs,
        st=a.st | b.st,
        top=a.top or b.top,
    )


def vleq(a: V, za: Zone, b: V) -> bool:
    if a is b:
        return True
    if b.top:
        return True
    if a.top:
        return False
    return (
        int_leq(a.i, za, b.i)
        and (not a.b or (b.b is not None and a.b <= b.b))
        and f_leq(a.f, b.f)
        and _str_leq(a.s, b.s)
        and (not a.n or b.n)
        and a.refs <= b.refs
        and a.st <= b.st
    )


def vwiden(a: V, za: Zone, b: V, zb: Zone) -> V:
    j = vjoin(a, za, b, zb)
    return j.replace(i=int_widen(a.i, za, j.i, zb) if a.i is not None else j.i, f=f_widen(a.f, j.f))


# -------------------------------------------------------------- heap lattice
def _len_join(a, za, b, zb):
    if isinstance(a, str) and isinstance(b, str) and a == b:
        return a
    ia = za.bounds(a) if isinstance(a, str) else a
    ib = zb.bounds(b) if isinstance(b, str) else b
    lo = None if ia[0] is None or ib[0] is None else min(ia[0], ib[0])
    hi = None if ia[1] is None or ib[1] is None else max(ia[1], ib[1])
    return (lo, hi)


def _len_leq(a, za, b):
    if isinstance(b, str):
        return a == b
    ia = za.bounds(a) if isinstance(a, str) else a
    lo_ok = b[0] is None or (ia[0] is not None and ia[0] >= b[0])
    hi_ok = b[1] is None or (ia[1] is not None and ia[1] <= b[1])
    return lo_ok and hi_ok


def _len_widen(a, za, b, zb):
    if isinstance(a, str) and isinstance(b, str) and a == b:
        return a
    j = _len_join(a, za, b, zb)
    ia = za.bounds(a) if isinstance(a, str) else a
    lo = ia[0] if (j[0] is not None and ia[0] is not None and j[0] >= ia[0]) else None
    hi = ia[1] if (j[1] is not None and ia[1] is not None and j[1] <= ia[1]) else None
    return (lo, hi)


class Incompatible(Exception):
    """Two heap objects at one address cannot be joined (analysis gives up)."""


def ojoin(a: Obj, za, b: Obj, zb, widen=False) -> Obj:
    if a is b:
        return a
    vj = vwiden if widen else vjoin
    if a.kind != b.kind:
        raise Incompatible(f"{a.kind} vs {b.kind}")
    if a.kind == "tuple":
        if a.items is not None and b.items is not None and len(a.items) == len(b.items):
            return TupleObj(tuple(vj(x, za, y, zb) for x, y in zip(a.items, b.items)))
        la = (len(a.items), len(a.items)) if a.items is not None else a.length
        lb = (len(b.items), len(b.items)) if b.items is not None else b.length
        ea = _elem_of_tuple(a, za)
        eb = _elem_of_tuple(b, zb)
        lf = _len_widen if widen else _len_join
        return TupleObj(None, lf(la, za, lb, zb), vj(ea, za, eb, zb))
    if a.kind == "list":
        lf = _len_widen if widen else _len_join
        length = lf(a.length, za, b.length, zb)
        if a.pre is not None and b.pre is not None and isinstance(length, str):
            return ListObj(length, vj(a.elem, za, b.elem, zb), vj(a.pre, za, b.pre, zb))
        return ListObj(length, vj(_all(a, za), za, _all(b, zb), zb))
    if a.kind == "array":
        if a.typecode != b.typecode:
            raise Incompatible(f"array of typecode {a.typecode!r} vs {b.typecode!r}")
        lf = _len_widen if widen else _len_join
        length = lf(a.length, za, b.length, zb)
        return ArrayObj(length, vj(_all(a, za), za, _all(b, zb), zb), a.typecode)
    if a.kind == "set":
        if a.frozen != b.frozen:
            raise Incompatible("a set and a frozenset at one site")
        lf = _len_widen if widen else _len_join
        lo, hi = lf(a.length, za, b.length, zb)
        return SetObj.make(lo, hi, vj(a.elem, za, b.elem, zb), a.frozen)
    if a.kind == "inst":
        if a.cls != b.cls:
            raise Incompatible("instance of two classes at one site")
        attrs = {}
        for k in set(a.attrs) | set(b.attrs):
            if k in a.attrs and k in b.attrs:
                (va, da, ra), (vb, db, rb) = a.attrs[k], b.attrs[k]
                attrs[k] = (vj(va, za, vb, zb), da and db, ra if ra == rb else None)
            else:
                v, _, _ = a.attrs.get(k) or b.attrs.get(k)
                attrs[k] = (v, False, None)
        qf = {}
        for g in set(a.qf) & set(b.qf):
            (la, ha, wa), (lb, hb, wb) = a.qf[g], b.qf[g]
            lo, hi = min(la, lb), max(ha, hb)
            if widen:
                lo, hi = _lem.widen_lo(la, lb), _lem.widen_hi(ha, hb)
            qf[g] = (lo, hi, _merge_why(wa, wb))
        return InstObj(a.cls, attrs, qf)
    if a.kind == "class":
        if a.node is not b.node:
            raise Incompatible("two classes at one site")
        attrs = {}
        for k in set(a.attrs) | set(b.attrs):
            if k in a.attrs and k in b.attrs:
                attrs[k] = vj(a.attrs[k], za, b.attrs[k], zb)
            else:
                raise Incompatible("class attribute set on one path only")
        return a.replace(attrs=attrs)
    if a.kind == "func":
        if a.node is not b.node or len(a.defaults) != len(b.defaults) or set(a.kwdefaults) != set(b.kwdefaults):
            raise Incompatible("two functions at one site")
        return a.replace(
            defaults=tuple(vj(x, za, y, zb) for x, y in zip(a.defaults, b.defaults)),
            kwdefaults={k: vj(a.kwdefaults[k], za, b.kwdefaults[k], zb) for k in a.kwdefaults},
        )
    if a.kind == "bmeth":
        if a.func != b.func:
            raise Incompatible("bound methods of two functions at one site")
        return BoundMethod(a.func, vj(a.self_v, za, b.self_v, zb))
    if a.kind == "bimeth":
        if a.name != b.name:
            raise Incompatible("two builtin methods at one site")
        return BuiltinMethod(a.name, vj(a.recv, za, b.recv, zb))
    if a.kind == "iterator":
        return IteratorObj(vj(a.source, za, b.source, zb))
    if a.kind == "range":
        if a.lo == b.lo and a.hi == b.hi and a.step == b.step:
            return a
        raise Incompatible("two ranges at one site")
    if a.kind == "dict":
        if type(a) is not type(b):
            raise Incompatible("plain dict and defaultdict at one allocation site")
        if isinstance(a, DefaultDictObj) and a.factory != b.factory:
            raise Incompatible("defaultdicts with different factories at one allocation site")
        items = {k: vj(a.items[k], za, b.items[k], zb) for k in a.items.keys() & b.items.keys()}
        okeys = vj(a.okeys, za, b.okeys, zb)
        ovals = vj(a.ovals, za, b.ovals, zb)
        for side, z in ((a, za), (b, zb)):
            for k, v in side.items.items():
                if k not in items:
                    okeys = vj(okeys, z, _key_value(k), z)
                    ovals = vj(ovals, z, v, z)
        exact = a.exact and b.exact and a.items.keys() == b.items.keys()
        if isinstance(a, DefaultDictObj):
            return DefaultDictObj(items, exact, okeys, ovals, a.factory)
        return DictObj(items, exact, okeys, ovals)
    if a.kind == "dictview":
        if a.which != b.which:
            raise Incompatible("two dict views at one site")
        return DictView(a.dicts | b.dicts, a.which)
    if a.kind == "stringio":
        # [internal ref]: join the one field a StringIOObj carries, the same
        # single-field shape as IteratorObj's ``source`` above.
        return StringIOObj(vj(a.content, za, b.content, zb))
    if a.kind == "outfile":
        # [internal ref]: an output handle is only its binary-ness, fixed by the constant mode of its one
        # ``open`` call site; two kinds at one site cannot happen, and is refused if it ever does.
        if a.binary != b.binary:
            raise Incompatible("a text and a binary output file at one site")
        return a
    raise Incompatible(a.kind)


def _key_value(k) -> V:
    from .aval import const_bool, const_int, const_str, NONE
    if isinstance(k, bool):
        return const_bool(k)
    if isinstance(k, int):
        return const_int(k)
    if isinstance(k, str):
        return const_str(k)
    return NONE


def _all(ob, z) -> V:
    """Every element of a list, ignoring its segmentation."""
    return ob.elem if ob.pre is None else vjoin(ob.elem, z, ob.pre, z)


def _elem_of_tuple(t: TupleObj, z) -> V:
    if t.items is None:
        return t.elem
    e = BOT
    for x in t.items:
        e = vjoin(e, z, x, z)
    return e


def oleq(a: Obj, za, b: Obj) -> bool:
    if a is b:
        return True
    if a.kind != b.kind:
        return False
    if a.kind == "tuple":
        if b.items is not None:
            return a.items is not None and len(a.items) == len(b.items) and all(
                vleq(x, za, y) for x, y in zip(a.items, b.items))
        la = (len(a.items), len(a.items)) if a.items is not None else a.length
        return _len_leq(la, za, b.length) and vleq(_elem_of_tuple(a, za), za, b.elem)
    if a.kind == "list":
        if not _len_leq(a.length, za, b.length):
            return False
        if b.pre is not None:
            return a.pre is not None and vleq(a.elem, za, b.elem) and vleq(a.pre, za, b.pre)
        return vleq(_all(a, za), za, b.elem)
    if a.kind == "array":
        if a.typecode != b.typecode:
            return False
        if not _len_leq(a.length, za, b.length):
            return False
        return vleq(_all(a, za), za, b.elem)
    if a.kind == "set":
        return a.frozen == b.frozen and _len_leq(a.length, za, b.length) and vleq(a.elem, za, b.elem)
    if a.kind == "inst":
        if a.cls != b.cls:
            return False
        for k, (vb, db, rb) in b.attrs.items():
            if k not in a.attrs:
                if db:
                    return False
                continue
            va, da, ra = a.attrs[k]
            if db and not da:
                return False
            if rb is not None and ra != rb:
                return False
            if not vleq(va, za, vb):
                return False
        for g, (lb, hb, _w) in b.qf.items():
            if g not in a.qf or a.qf[g][0] < lb or a.qf[g][1] > hb:
                return False
        return all(k in b.attrs for k in a.attrs)
    if a.kind == "class":
        return set(a.attrs) == set(b.attrs) and all(vleq(a.attrs[k], za, b.attrs[k]) for k in a.attrs)
    if a.kind == "func":
        return all(vleq(x, za, y) for x, y in zip(a.defaults, b.defaults)) and all(
            vleq(a.kwdefaults[k], za, b.kwdefaults[k]) for k in a.kwdefaults)
    if a.kind == "bmeth":
        return a.func == b.func and vleq(a.self_v, za, b.self_v)
    if a.kind == "bimeth":
        return a.name == b.name and vleq(a.recv, za, b.recv)
    if a.kind == "iterator":
        return vleq(a.source, za, b.source)
    if a.kind == "range":
        return a.lo == b.lo and a.hi == b.hi and a.step == b.step
    if a.kind == "dict":
        if type(a) is not type(b):
            return False
        if isinstance(a, DefaultDictObj) and a.factory != b.factory:
            return False
        if b.exact and (not a.exact or a.items.keys() != b.items.keys()):
            return False
        if not b.items.keys() <= a.items.keys():
            return False
        for k, v in a.items.items():
            w = b.items.get(k, b.ovals)
            if not vleq(v, za, w):
                return False
        return vleq(a.okeys, za, b.okeys) and vleq(a.ovals, za, b.ovals)
    if a.kind == "dictview":
        return a.which == b.which and a.dicts <= b.dicts
    if a.kind == "pattern":
        # #1102b: no fields to compare -- same kind already checked above.
        return True
    if a.kind == "stringio":
        return vleq(a.content, za, b.content)
    if a.kind == "outfile":
        return a.binary == b.binary
    return False


def _proof_text(ob) -> tuple:
    """The ``why`` of every value an object holds, and the lemma text of an instance's float invariants,
    in an order that does not depend on how the object was built."""
    if ob.kind == "inst":
        return (tuple(sorted((k, e[0].why) for k, e in ob.attrs.items())),
                tuple(sorted((tuple(sorted(g)), w) for g, (_lo, _hi, w) in ob.qf.items())))
    if isinstance(ob, DefaultDictObj):
        return ("defaultdict", ob.factory, tuple(v.why for v in ob.values()))
    return tuple(v.why for v in ob.values())


# ------------------------------------------------------------------ the state
class Frame:
    __slots__ = ("fid", "env", "maybe_unbound")

    def __init__(self, fid: str, env: dict | None = None, maybe_unbound=()):
        self.fid = fid
        self.env: dict[str, V] = env if env is not None else {}
        self.maybe_unbound: set[str] = set(maybe_unbound) & self.env.keys()

    def copy(self) -> "Frame":
        return Frame(self.fid, dict(self.env), self.maybe_unbound)


class State:
    __slots__ = ("frames", "heap", "zone", "pins")

    def __init__(self):
        self.frames: list[Frame] = []
        self.heap: dict = {}
        self.zone = Zone()
        self.pins: tuple = ()

    @property
    def bottom(self) -> bool:
        return self.zone.bottom

    def set_bottom(self) -> None:
        self.zone.bottom = True
        self.zone.m.clear()

    def copy(self) -> "State":
        s = State.__new__(State)
        s.frames = [f.copy() for f in self.frames]
        s.heap = dict(self.heap)
        s.zone = self.zone.copy()
        s.pins = self.pins
        return s

    @property
    def frame(self) -> Frame:
        return self.frames[-1]

    # ------------------------------------------------------------ lattice
    def join(self, o: "State") -> "State":
        if self.bottom:
            return o.copy()
        if o.bottom:
            return self.copy()
        return self._combine(o, widen=False)

    def widen(self, o: "State") -> "State":
        if self.bottom:
            return o.copy()
        if o.bottom:
            return self.copy()
        return self._combine(o, widen=True)

    def _combine(self, o: "State", widen: bool) -> "State":
        assert len(self.frames) == len(o.frames), "frame stacks differ at a join"
        za, zb = self.zone, o.zone
        vj = vwiden if widen else vjoin
        s = State.__new__(State)
        s.frames = []
        for fa, fb in zip(self.frames, o.frames):
            assert fa.fid == fb.fid
            env = {}
            maybe_unbound = set(fa.maybe_unbound) | set(fb.maybe_unbound)
            for k in fa.env.keys() | fb.env.keys():
                if k in fa.env and k in fb.env:
                    env[k] = vj(fa.env[k], za, fb.env[k], zb)
                else:
                    env[k] = fa.env.get(k) or fb.env[k]
                    maybe_unbound.add(k)
            s.frames.append(Frame(fa.fid, env, maybe_unbound))
        heap = {}
        for k in self.heap.keys() | o.heap.keys():
            if k in self.heap and k in o.heap:
                heap[k] = ojoin(self.heap[k], za, o.heap[k], zb, widen=widen)
            else:
                heap[k] = self.heap.get(k) or o.heap[k]
        s.heap = heap
        if widen:
            s.zone = za.widen(zb)
            s.zone.close()
        else:
            s.zone = za.join(zb)
        s.pins = self.pins if self.pins == o.pins else tuple(dict.fromkeys(self.pins + o.pins))
        return s

    def leq(self, o: "State") -> bool:
        if self.bottom:
            return True
        if o.bottom:
            return False
        if not self.zone.leq(o.zone):
            return False
        za = self.zone
        if len(self.frames) != len(o.frames):
            return False
        for fa, fb in zip(self.frames, o.frames):
            if fa.fid != fb.fid:
                return False
            for k in fa.env.keys() | fb.env.keys():
                av, bv = fa.env.get(k), fb.env.get(k)
                a_absent = av is None or k in fa.maybe_unbound
                b_absent = bv is None or k in fb.maybe_unbound
                if a_absent and not b_absent:
                    return False
                if av is not None and (bv is None or not vleq(av, za, bv)):
                    return False
        for k, ob in self.heap.items():
            w = o.heap.get(k)
            if w is None or not oleq(ob, za, w):
                return False
        return True

    def equiv(self, o: "State") -> bool:
        """Semantic equality, the test behind fixpoint reuse (``interp.py::Analyzer._fixpoint``): each
        state is included in the other (``leq`` both ways) AND everything ``leq`` does not look at agrees:
        the frame stack (same frames, same variables), the pins, and the proof text of every value
        (``V.why``, which ``leq`` and ``==`` ignore but a site record's fact can quote).  A bottom state
        equals nothing, so a result is never reused for an unreachable entry."""
        if self.bottom or o.bottom or len(self.frames) != len(o.frames) or self.pins != o.pins:
            return False
        if self.heap.keys() != o.heap.keys():
            return False
        for fa, fb in zip(self.frames, o.frames):
            if (fa.fid != fb.fid or fa.env.keys() != fb.env.keys()
                    or fa.maybe_unbound != fb.maybe_unbound):
                return False
            if any(v.why != fb.env[k].why for k, v in fa.env.items()):
                return False
        if any(a.why != b.why for a, b in zip(self.pins, o.pins)):
            return False
        for k, ob in self.heap.items():
            w = o.heap[k]
            if ob is not w and _proof_text(ob) != _proof_text(w):
                return False
        return self.leq(o) and o.leq(self)

    # --------------------------------------------------- symbolic references
    def _map_values(self, fn) -> None:
        """Apply ``fn: V -> V`` to every value stored in frames and the heap."""
        for f in self.frames:
            for k, v in list(f.env.items()):
                nv = fn(v)
                if nv is not v:
                    f.env[k] = nv
        for addr, ob in list(self.heap.items()):
            nob = _map_obj(ob, fn)
            if nob is not ob:
                self.heap[addr] = nob
        if self.pins:
            self.pins = tuple(fn(v) for v in self.pins)

    def rewrite_bounds_away(self, zv: str, ob=None):
        """Rewrite every summary bound naming ``zv`` to one that does not (and drop
        every float term naming ``zv``'s value, which is about to change).

        ``ob`` is an object that is not in the heap yet -- the replacement a caller built from the one
        in the heap and is about to store (``Analyzer.set_len``).  Its summaries were derived while
        ``zv`` still had its old value, so they name it exactly as the stored ones do: they are
        rewritten the same way and the new object is returned (``None`` when ``ob`` is ``None``).

        A variable the zone does not have is named by nothing: a bound or a term that names a variable is
        rewritten away before the variable is forgotten (this method and ``kill_var``), and a variable
        enters the zone with no value naming it.  Most calls are of that kind (``write_var`` kills the variable of
        every non-int assignment): 48,034 of 62,358 in a chaos run, none of which changed a value, and each
        one walked every value of every frame and object."""
        if zv not in self.zone.vars:
            return ob
        return self._rewrite_bounds_away(zv, ob)

    def _rewrite_bounds_away(self, zv: str, ob):
        """The walk of ``rewrite_bounds_away``: one pass over the values that drops the float terms naming
        ``zv`` and rewrites the bounds naming it."""
        strip = _term_stripper(lambda t: t == ("iv", zv))
        z = self.zone
        if z.bottom:
            self._map_values(strip)
            return None if ob is None else _map_obj(ob, strip)
        # the best replacement anchor: an equal variable (not a temporary), else ZERO
        eq = None
        for y in sorted(z.vars):
            if y == zv or y == ZERO or y.startswith("$"):
                continue
            if z.get(zv, y) == 0 and z.get(y, zv) == 0:
                eq = y
                break

        def fix_hi(b):
            if b is None or b[0] != zv:
                return b
            if eq is not None:
                return (eq, b[1])
            hi = z.get(zv, ZERO)
            return None if hi is None else (ZERO, hi + b[1])

        def fix_lo(b):
            if b is None or b[0] != zv:
                return b
            if eq is not None:
                return (eq, b[1])
            lo = z.get(ZERO, zv)
            return None if lo is None else (ZERO, b[1] - lo)

        def fn(v: V) -> V:
            ii = v.i
            if ii is not None and ((ii.lo is not None and ii.lo[0] == zv) or (ii.hi is not None and ii.hi[0] == zv)):
                v = v.replace(i=IntInfo(ii.lin, fix_lo(ii.lo), fix_hi(ii.hi)))
            return strip(v)

        self._map_values(fn)
        return None if ob is None else _map_obj(ob, fn)

    def shift_bounds(self, zv: str, k: int, ob=None):
        """``zv := zv + k``: keep every summary bound naming ``zv`` true.  ``ob`` is an object that is not
        in the heap yet (see :meth:`rewrite_bounds_away`): it is shifted too and returned.  A variable the
        zone does not have is named by nothing (see :meth:`rewrite_bounds_away`)."""
        if zv not in self.zone.vars:
            return ob
        return self._shift_bounds(zv, k, ob)

    def _shift_bounds(self, zv: str, k: int, ob):
        strip = _term_stripper(lambda t: t == ("iv", zv))

        def fn(v: V) -> V:
            ii = v.i
            if ii is not None:
                lo, hi = ii.lo, ii.hi
                ch = False
                if lo is not None and lo[0] == zv:
                    lo = (zv, lo[1] - k)
                    ch = True
                if hi is not None and hi[0] == zv:
                    hi = (zv, hi[1] - k)
                    ch = True
                if ch:
                    v = v.replace(i=IntInfo(ii.lin, lo, hi))
            return strip(v)

        self._map_values(fn)
        return None if ob is None else _map_obj(ob, fn)

    def kill_var(self, zv: str) -> None:
        self.rewrite_bounds_away(zv)
        self.zone.forget(zv)

    def strip_terms(self, pred, ob=None):
        """Drop every float value term for which ``pred(term)`` holds (what it names changed).  ``ob`` is an
        object that is not in the heap yet (a replacement about to be stored): it is rewritten the same
        way and returned."""
        fn = _term_stripper(pred)
        self._map_values(fn)
        return None if ob is None else _map_obj(ob, fn)

    def redirect(self, old, new) -> None:
        def fn(v: V) -> V:
            if old in v.refs:
                return v.replace(refs=(v.refs - {old}) | {new})
            return v

        self._map_values(fn)
        for addr, ob in list(self.heap.items()):
            if ob.kind == "dictview" and old in ob.dicts:
                self.heap[addr] = DictView((ob.dicts - {old}) | {new}, ob.which)
            elif ob.kind == "bmeth" and ob.func == old:
                # a bound method names its function by ADDRESS, not by value: when the function's site is allocated
                # again (a ``def`` or lambda in a loop) the older function becomes the site's summary and the bound
                # method of the older one must follow it, or it would call the newer function
                self.heap[addr] = BoundMethod(new, ob.self_v)


def _map_obj(ob: Obj, fn) -> Obj:
    k = ob.kind
    if k == "list":
        e = fn(ob.elem)
        p = fn(ob.pre) if ob.pre is not None else None
        return ob if e is ob.elem and p is ob.pre else ListObj(ob.length, e, p)
    if k == "array":
        e = fn(ob.elem)
        return ob if e is ob.elem else ArrayObj(ob.length, e, ob.typecode)
    if k == "set":
        e = fn(ob.elem)
        return ob if e is ob.elem else SetObj(ob.length, e, ob.frozen)
    if k == "tuple":
        if ob.items is not None:
            items = tuple(fn(x) for x in ob.items)
            return ob if all(a is b for a, b in zip(items, ob.items)) else TupleObj(items)
        e = fn(ob.elem)
        return ob if e is ob.elem else TupleObj(None, ob.length, e)
    if k == "inst":
        ch = False
        attrs = {}
        for a, (v, d, r) in ob.attrs.items():
            nv = fn(v)
            ch = ch or nv is not v
            attrs[a] = (nv, d, r)
        return InstObj(ob.cls, attrs, ob.qf) if ch else ob
    if k == "class":
        attrs = {a: fn(v) for a, v in ob.attrs.items()}
        return ob.replace(attrs=attrs) if any(attrs[a] is not ob.attrs[a] for a in attrs) else ob
    if k == "func":
        d = tuple(fn(x) for x in ob.defaults)
        kd = {a: fn(v) for a, v in ob.kwdefaults.items()}
        return ob.replace(defaults=d, kwdefaults=kd)
    if k == "bmeth":
        s = fn(ob.self_v)
        return ob if s is ob.self_v else BoundMethod(ob.func, s)
    if k == "bimeth":
        s = fn(ob.recv)
        return ob if s is ob.recv else BuiltinMethod(ob.name, s)
    if k == "iterator":
        return IteratorObj(fn(ob.source))
    if k == "dict":
        items = {a: fn(v) for a, v in ob.items.items()}
        ok, ov = fn(ob.okeys), fn(ob.ovals)
        return ob.replace(items=items, okeys=ok, ovals=ov)
    return ob


def _term_stripper(pred):
    """``V -> V`` dropping the float value term when ``pred`` holds for any part of it."""
    def fn(v: V) -> V:
        f = v.f
        if f and f.term is not None and _mentions(f.term, pred):
            return v.replace(f=FloatInfo(f.fact, None, f.why))
        return v

    return fn


def _mentions(t, pred) -> bool:
    if pred(t):
        return True
    if isinstance(t, tuple):
        return any(_mentions(x, pred) for x in t)
    return False


def iv_of_len(length, z: Zone):
    return z.bounds(length) if isinstance(length, str) else length


__all__ = [
    "State", "Frame", "vjoin", "vleq", "vwiden", "ojoin", "oleq", "Incompatible",
    "hi_join", "lo_join", "iv_of_len", "lin_bounds", "Lin",
]
