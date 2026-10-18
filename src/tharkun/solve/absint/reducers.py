"""The iterable-reducing builtins for ``Analyzer``: ``min``/``max`` over an iterable, ``sum``,
``any``, ``all`` and the keywords of ``sorted`` (docs/ABSINT.md, "The iterable-reducing builtins").

One family, one shape: the argument is a modelled builtin iterable (list, tuple, array, range, dict
keys, a str), the call reads it without changing it, and the result is a function of two facts the
analysis already holds about it: the bounds of its length and the join of its elements
(``Analyzer.elem_summary``).  Each model is entered through ``Analyzer._contract_builtin_call`` (the
``c_<name>`` convention of ``interp.py``): the TESTED contract in ``solve/contracts/rules_builtins.py`` gives the
seven property classes, and a model upgrades a class to proven only for the reason it states.

What the family never does: run a user hook.  An element that is an object (an instance with
``__lt__``/``__add__``/``__bool__``, any reference, a function, an unknown value) is ``Unsupported``,
because an element-wise call would have to be analysed once per element, in order, and the elements are
summarised.  So every class the contracts admit only through a hook (all but ``TypeError``) is proven for
the elements that remain: numbers, strs and None.

Exact or refuse: a ``key=`` function (a call per element) and an iterator or dict view (consumed by the
call) are ``Unsupported``; ``key=None`` is the call without a key.
"""

from __future__ import annotations

from .aval import ANY_STR, BOT, FLOAT, V, const_bool, const_int, int_iv
from .errors import Unsupported
from .state import vjoin

#: the property classes the contracts of these builtins admit only through a user hook (``__iter__``,
#: ``__next__``, ``__lt__``, ``__add__``, ``__radd__``, ``__bool__``, the key function).  A model whose
#: elements are numbers, strs and None runs none of them.
NO_HOOK_CLASSES = ("AttributeError", "IndexError", "KeyError", "ZeroDivisionError", "FloatingPointError",
                   "AssertionError")
_NO_HOOK = {exc: "numbers and strs only: no user hook runs" for exc in NO_HOOK_CLASSES}


def _sum_bounds(lo, hi, a, b):
    """The bounds of a sum of ``n`` elements, ``n`` in ``[lo, hi]``, each element in ``[a, b]``
    (``None`` is unbounded): the lower bound is the smallest ``n * a`` and the upper bound the largest
    ``n * b``, over the counts the length allows."""
    if a is None:
        lower = 0 if hi == 0 else None
    elif a >= 0:
        lower = lo * a
    else:
        lower = None if hi is None else hi * a
    if b is None:
        upper = 0 if hi == 0 else None
    elif b <= 0:
        upper = lo * b
    else:
        upper = None if hi is None else hi * b
    return lower, upper


def _is_none_only(v: V) -> bool:
    return v.n and v.i is None and not v.b and not v.f and v.s is None and not v.refs and not v.st and not v.top


class ReducersMixin:
    """Mixed into ``interp.py::Analyzer``; every method here uses only Analyzer methods and the shared value types."""

    # ------------------------------------------------------------ the source
    def _reduce_source(self, st, v: V, node, name: str):
        """``(lo, hi, iterable_ok)`` for the iterable argument ``v`` of ``name``: the bounds of its length
        over every possible iterable (``hi`` is ``None`` when unbounded) and whether it is certainly
        iterable (the TypeError check is recorded).  ``None`` when no alternative of ``v`` is iterable, so
        the call always raises and no path continues."""
        if v.top:
            raise Unsupported(f"{name}() of an unknown value", node)
        ok = self.check_iterable(st, v, node)
        los: list = []
        his: list = []
        for a in sorted(v.refs, key=repr):
            ob = st.heap[a]
            kind = ob.kind
            if kind in ("list", "tuple", "array"):
                lo, hi = self.len_lin(st, ob)[1]
            elif kind == "range":
                lo, hi = self._range_len(st, ob)[1]
            elif kind == "dict":
                lo, hi = self.dict_len(st, {a})
            elif kind == "set":
                lo, hi = ob.length
            else:
                # an iterator is consumed by the call and a dict view is not modelled here
                raise Unsupported(f"{name}() of a {kind}", node)
            los.append(lo)
            his.append(hi)
        if v.s is not None:
            if v.s == ANY_STR:
                los.append(0)
                his.append(None)
            else:
                los.append(min(len(x) for x in v.s))
                his.append(max(len(x) for x in v.s))
        if not los:
            return None
        lo = min(0 if x is None else x for x in los)
        hi = None if any(x is None for x in his) else max(his)
        return lo, hi, ok

    def _elems(self, st, v: V, node, name: str) -> V | None:
        """The join of every element ``v`` may hold; ``None`` when it can hold none.  An element that is an
        object, a function or an unknown value refuses: it may carry a hook the family does not run."""
        e = self.elem_summary(st, v)
        if e is not None and (e.refs or e.st or e.top):
            raise Unsupported(f"{name}() over elements that are objects", node)
        return e

    # ------------------------------------------------------------ min / max over an iterable
    def _minmax_iterable(self, st, v: V, kwargs, node, is_min: bool):
        name = "min" if is_min else "max"
        default, has_default = None, False
        for k, kv in kwargs.items():
            if k == "default":
                default, has_default = kv, True
            elif k == "key":
                if not _is_none_only(kv):
                    raise Unsupported(f"{name}() with key=: a call per element, not modelled", node)
            else:
                raise Unsupported(f"{name}() with the keyword {k}", node)
        if has_default and default.top:
            raise Unsupported(f"{name}() with an unknown default", node)
        src = self._reduce_source(st, v, node, name)
        if src is None:
            st.set_bottom()
            return BOT, {}
        lo, hi, iter_ok = src
        elems = self._elems(st, v, node, name)
        empty_only = hi == 0 or elems is None
        cmp_ok = elems is None or self._elem_sortable(elems)
        # one element is never compared with another
        ok = iter_ok and (cmp_ok or (hi is not None and hi <= 1))
        self.check(node, f"{name}-iterable", {"TypeError"}, ok, f"{name}() over elements: " + (",".join(elems.kinds()) if elems else "none"))
        proven = dict(_NO_HOOK)
        if ok:
            proven["TypeError"] = "the elements are mutually orderable (or there is at most one)"
        if empty_only:
            # an empty iterable: the default if there is one, else ValueError (not a property class)
            if has_default:
                return default, proven
            st.set_bottom()
            return BOT, proven
        if has_default and lo == 0:
            # the iterable may be empty: the result may be the default; no element is known to exist
            out = vjoin(self._const_bounds(st, elems, ""), st.zone, default, st.zone)
            return self.mat(st, out), proven
        # no default, or the iterable is certainly non-empty: the call continues only on a non-empty one
        elem = self.iter_elem(st, v, node)
        if st.bottom or elem is None:
            st.set_bottom()
            return BOT, {}
        return elem, proven

    # ------------------------------------------------------------ sum
    def c_sum(self, st, args, kwargs, node):
        """``sum(iterable, start=0)``: ``start`` plus the elements, left to right.  Integers (and bools) are
        exact: the sum of ``n`` elements in ``[a, b]`` lies in ``[n_lo*a, n_hi*b]`` (the products
        with the length bounds that fit), plus ``start``.  A float anywhere gives a float of unknown
        value: CPython's ``sum`` compensates float rounding (3.12+), so the plain left fold is not the same
        number and no interval of it is claimed.  A str, None or object element is a TypeError (strs and
        None) or a hook (objects, refused); ``start`` must be a number."""
        if not 1 <= len(args) <= 2 or (len(args) == 2 and kwargs):
            self.check(node, "call-args", {"TypeError"}, False, "sum() arity")
            st.set_bottom()
            return BOT, {}
        start = args[1] if len(args) == 2 else const_int(0)
        for k, kv in kwargs.items():
            if k != "start":
                raise Unsupported(f"sum() with the keyword {k}", node)
            start = kv
        if start.top or start.s is not None or start.n or start.refs or start.st:
            raise Unsupported("sum() with a start that is not a number", node)
        src = self._reduce_source(st, args[0], node, "sum")
        if src is None:
            st.set_bottom()
            return BOT, {}
        lo, hi, iter_ok = src
        elems = self._elems(st, args[0], node, "sum")
        empty_only = hi == 0 or elems is None
        bad = elems is not None and (elems.s is not None or elems.n)
        ok = iter_ok and not bad
        self.check(node, "sum", {"TypeError"}, ok, "sum() elements: " + (",".join(elems.kinds()) if elems else "none"))
        proven = dict(_NO_HOOK)
        if ok:
            proven["TypeError"] = "the argument is iterable and every element is a number"
        out = BOT
        if lo == 0:
            # the iterable may be empty: the result is start itself (kind and value as given)
            out = vjoin(out, st.zone, start, st.zone)
        if not empty_only:
            e_int = self._as_int(st, elems)
            s_int = self._as_int(st, start)
            if e_int is not None and s_int is not None:
                a, b = self.ibounds(st, e_int)
                s1, s2 = self.ibounds(st, s_int)
                lower, upper = _sum_bounds(lo, hi, a, b)
                lower = None if s1 is None or lower is None else s1 + lower
                upper = None if s2 is None or upper is None else s2 + upper
                out = vjoin(out, st.zone, int_iv(lower, upper), st.zone)
            if elems.f or start.f:
                out = vjoin(out, st.zone, FLOAT, st.zone)
        if out.is_bot():
            st.set_bottom()
            return BOT, proven
        return self.mat(st, out), proven

    # ------------------------------------------------------------ any / all
    def c_any(self, st, args, kwargs, node):
        return self._any_all(st, args, kwargs, node, "any")

    def c_all(self, st, args, kwargs, node):
        return self._any_all(st, args, kwargs, node, "all")

    def _any_all(self, st, args, kwargs, node, name: str):
        """``any(x)``/``all(x)`` over an iterable: a bool.  ``any([])`` is False and ``all([])`` is True; a
        truthy or falsy ELEMENT decides the rest, so the result is the truth values the length bounds and
        the elements' truth values leave possible.  An instance element refuses (a ``__bool__`` or
        ``__len__`` runs per element)."""
        if not self._arity(node, args, kwargs, 1, 1, name):
            st.set_bottom()
            return BOT, {}
        src = self._reduce_source(st, args[0], node, name)
        if src is None:
            st.set_bottom()
            return BOT, {}
        lo, hi, iter_ok = src
        elems = self._elems(st, args[0], node, name)
        if elems is not None and self.instance_value(st, elems):
            raise Unsupported(f"{name}() over instances", node)
        can_t = can_f = False
        if elems is not None:
            t, f = self.truth_split(st.copy(), elems, node)
            can_t, can_f = not t.bottom, not f.bottom
        nonempty = hi != 0 and elems is not None
        may_be_empty = lo == 0 or elems is None
        if name == "any":
            true_possible = nonempty and can_t
            false_possible = may_be_empty or can_f
        else:
            true_possible = may_be_empty or can_t
            false_possible = nonempty and can_f
        out = BOT
        if true_possible:
            out = vjoin(out, st.zone, const_bool(True), st.zone)
        if false_possible:
            out = vjoin(out, st.zone, const_bool(False), st.zone)
        proven = dict(_NO_HOOK)
        if iter_ok:
            proven["TypeError"] = "the argument is iterable"
        if out.is_bot():
            st.set_bottom()
            return BOT, {}
        return out, proven

    # ------------------------------------------------------------ sorted's keywords
    def _sorted_keywords(self, st, kwargs, node) -> None:
        """``sorted(x, key=None, reverse=<scalar>)``: ``reverse`` only changes the order (any scalar
        truth value is computed without a hook) and ``key=None`` is the call without a key.  A key function
        is a call per element (refused); another keyword is a TypeError the model does not take."""
        for k, kv in kwargs.items():
            if k == "key":
                if not _is_none_only(kv):
                    raise Unsupported("sorted() with key=: a call per element, not modelled", node)
            elif k == "reverse":
                if kv.top or kv.refs or kv.st:
                    raise Unsupported("sorted() with a reverse= that is not a scalar", node)
            else:
                raise Unsupported(f"sorted() with the keyword {k}", node)

