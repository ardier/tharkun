"""The ``random`` module models of the absint pass ([internal ref], [internal ref]).

``random.random``, ``randint``, ``uniform``, ``choice`` and ``seed``: the module-level functions are bound methods of one
``random.Random`` instance, so a call binds its arguments exactly as CPython binds a call of a plain Python function
(``Random.randint(self, a, b)``, ``Random.seed(self, a=None, version=2)``): each parameter may be given by its NAME, and a call
that does not bind (too many or too few arguments, an unknown or repeated keyword) is a certain ``TypeError``.  ``random()`` is the
one C method: no arguments and no keywords at all.  Before this module an ARGUMENT COUNT other than the positional one (every
keyword call, and ``seed(a, version)``) was recorded as a ``TypeError`` and the path ended, so a program that calls
``random.randint(a=0, b=5)`` and then divides by zero was proven free of ``ZeroDivisionError`` ([internal ref]: a wrong TRUE).

The rule of the pass ([internal ref]) holds here: a shape error is a certain, recorded ``TypeError`` and ends the path; a value the model
does not know (an unknown value, a user object whose methods could run) is a named refusal, never an infeasible path.

``uniform(a, b)`` is the formula CPython evaluates, ``a + (b - a) * random()`` (``Lib/random.py::Random.uniform``), evaluated by the
ordinary operators of this analyser over the draw ``[0.0, 1.0]``: the result is NOT inside ``[a, b]`` when ``b - a`` overflows
(``uniform(-1e308, 1e308)`` is ``inf``: ``(1e308 - -1e308)`` is ``inf``), is ``nan`` when it is ``inf * 0.0`` or an infinite bound
meets its own infinity, and an int too large for a float is an ``OverflowError`` (not a property class: that path ends)
([internal ref]).  The operators already carry the transfer functions of ``solve/values.py`` (outward-rounded intervals, ``inf`` and ``nan``).
"""

from __future__ import annotations

import ast

from .. import values as fv
from .aval import ANYSTR, BOT, FloatInfo, NONE, V, int_iv
from .errors import Unsupported
from .linear import lin_bounds
from .state import vjoin

#: ``name -> (parameters in order, the ones with no default)`` of the functions below, as the Python source of
#: ``Lib/random.py`` (CPython 3.14) declares them; ``random()`` is a C method: no parameters at all
_SHAPES = {
    "random": ((), ()),
    "getstate": ((), ()),
    "uniform": (("a", "b"), ("a", "b")),
    "randint": (("a", "b"), ("a", "b")),
    "choice": (("seq",), ("seq",)),
    "seed": (("a", "version"), ()),
    "randrange": (("start", "stop", "step"), ("start",)),
    "getrandbits": (("k",), ("k",)),  # a C method: positional only (``_POSITIONAL_ONLY``)
}

#: the C methods: a keyword is a ``TypeError`` even for a parameter that has a name
_POSITIONAL_ONLY = frozenset({"random", "getrandbits"})

#: the draw of ``random.random()``: a float in ``[0.0, 1.0)``, kept as the closed hull
_DRAW = FloatInfo(fv.FloatFact(0.0, 1.0, False))


class RandomMixin:
    """``Analyzer.m_random_*``: dispatched through ``MODULES["random"]`` (``interp.py``)."""

    def _random_args(self, st, n, a, k, name):
        """The call's arguments by parameter name, or ``None`` when the call does not bind (a recorded, certain ``TypeError``: the
        path has ended).  An unknown value, or a user object (its ``__index__``, ``__add__``, ``__eq__`` ... could run), is a
        named refusal."""
        params, required = _SHAPES[name]
        bound = self._bind_call(st, n, a, k, f"random.{name}", params, required, positional_only=name in _POSITIONAL_ONLY)
        if bound is None:
            return None
        for v in bound.values():
            if v.top:
                raise Unsupported(f"random.{name} of an unknown value", n)
            if self.instance_value(st, v):
                raise Unsupported(f"random.{name} of a user object", n)
        return bound

    def m_random_choice(self, st, a, k, n):
        bound = self._random_args(st, n, a, k, "choice")
        if bound is None:
            # the call does not bind: its TypeError is raised before any sequence is read, so no empty one can raise ``IndexError``
            self.check(n, "choice", {"IndexError"}, True, "the call does not bind: no sequence is read")
            return BOT
        v = bound["seq"]
        out = BOT
        bad = bool(v.i is not None or v.b or v.f or v.n or v.st)
        for r in v.refs:
            ob = st.heap[r]
            if ob.kind in ("list", "tuple", "array"):
                llin, liv = self.len_lin(st, ob)
                ok = liv[0] is not None and liv[0] >= 1
                if not ok and llin is not None:
                    lo, _ = lin_bounds(llin, st.zone)
                    ok = lo is not None and lo >= 1
                self.check(n, "choice", {"IndexError"}, ok,
                           f"choice from a sequence of length {self.plin(llin)} in [{liv[0]}, {liv[1]}]"
                           + (" >= 1" if ok else ": may be empty"))
                out = vjoin(out, st.zone, self.all_elems(st, ob), st.zone)
            elif ob.kind in ("dict", "dictview", "range", "set"):
                raise Unsupported(f"random.choice of a {ob.kind}", n)
            else:
                bad = True
        if v.s is not None:
            self.check(n, "choice", {"IndexError"}, False, "choice from a str of unknown length")
            out = vjoin(out, st.zone, ANYSTR, st.zone)
        if bad:
            self.check(n, "choice", {"IndexError"}, True, "a part that is not a sequence raises TypeError before any length is read")
        self.check(n, "arg-type", {"TypeError"}, not bad, "choice from " + ",".join(v.kinds()))
        if out.is_bot():
            st.set_bottom()
            return BOT
        return self.mat(st, out)  # the non-raising path: an element was chosen

    def m_random_random(self, st, a, k, n):
        if self._random_args(st, n, a, k, "random") is None:
            return BOT
        return V(f=_DRAW)

    def m_random_getstate(self, st, a, k, n):
        if self._random_args(st, n, a, k, "getstate") is None:
            return BOT
        # CPython seeding/draws expose index 1..624; setstate, which permits zero, is refused.
        state = self.new_list(st, n, None, BOT, tag="random-state", kind="tuple",
                              items=[int_iv(0, 2**32 - 1)] * 624 + [int_iv(1, 624)])
        result = self.new_list(st, n, None, BOT, tag="random-getstate", kind="tuple",
                               items=[int_iv(3, 3), V(refs=frozenset({state})), V(top=True)])
        return V(refs=frozenset({result}))

    def m_random_randint(self, st, a, k, n):
        bound = self._random_args(st, n, a, k, "randint")
        if bound is None:
            return BOT
        for x in (bound["a"], bound["b"]):
            self._num_arg(st, x, n, "randint", allow_float=False)
        lo = self._as_int(st, bound["a"])
        hi = self._as_int(st, bound["b"])
        if lo is None or hi is None:
            st.set_bottom()
            return BOT
        l, _ = self.ibounds(st, lo)
        _, h = self.ibounds(st, hi)
        if l is not None and h is not None and l > h:
            st.set_bottom()  # a > b is a ValueError on every run (not a property class): the path ends
            return BOT
        return int_iv(l, h)  # a > b on some runs: the same ValueError, and the other runs are inside [a, b]

    def m_random_uniform(self, st, a, k, n):
        """``a + (b - a) * random()``, evaluated with the analyser's own operators (see the module docstring): never the hull
        ``[a, b]``, which is wrong exactly when ``b - a`` overflows."""
        bound = self._random_args(st, n, a, k, "uniform")
        if bound is None:
            return BOT
        return self.uniform_formula(st, bound["a"], bound["b"], n)

    def uniform_formula(self, st, x: V, y: V, n) -> V:
        """``x + (y - x) * random()`` with the analyser's own operators: the value set of ``random.uniform(x, y)``, and ([internal ref]) of the
        code of ``_sv_verifier.nondet_float_range(x, y)``.  The state is bottom afterwards when an operator certainly raises."""
        span = self.binop(st, ast.Sub(), y, x, n)
        if st.bottom:
            return BOT
        step = self.binop(st, ast.Mult(), span, V(f=_DRAW), n)
        if st.bottom:
            return BOT
        return self.binop(st, ast.Add(), x, step, n)

    def m_random_seed(self, st, a, k, n):
        bound = self._random_args(st, n, a, k, "seed")
        if bound is None:
            return BOT
        if "a" in bound:
            v = bound["a"]
            bad = bool(v.refs or v.st)  # None, int, float, str (and bytes) are accepted
            self.check(n, "arg-type", {"TypeError"}, not bad, "seed of " + ",".join(v.kinds()))
        return NONE

    # ------------------------------------------------------------------ randrange and getrandbits ([internal ref] follow-up)
    @staticmethod
    def _hull(parts):
        """The union (as bounds, ``None`` unbounded) of ``(lo, hi)`` intervals."""
        lo = None if any(p[0] is None for p in parts) else min(p[0] for p in parts)
        hi = None if any(p[1] is None for p in parts) else max(p[1] for p in parts)
        return lo, hi

    def m_random_randrange(self, st, a, k, n):
        """``randrange(start, stop=None, step=1)`` (``Lib/random.py::Random.randrange``, 3.14)::

            istart = _index(start)                      # TypeError: only int and bool have __index__
            if stop is None:
                if step is not _ONE: raise TypeError    # ``step`` given, and not the very int 1
                if istart > 0: return randbelow(istart) # [0, istart - 1]
                raise ValueError
            istop, istep = _index(stop), _index(step)
            if istep == 1: width > 0 -> istart + randbelow(width), else ValueError
            istep > 0 -> n = (width + istep - 1) // istep;  istep < 0 -> n = (width + istep + 1) // istep;  istep == 0 -> ValueError
            n <= 0 -> ValueError;  return istart + istep * randbelow(n)

        The result is the hull of the draws over the bounds of the arguments: ``[istart, istop - 1]`` for a positive step (needs
        ``istop > istart`` for some pair of values), ``[istop + 1, istart]`` for a negative one, the two joined when the step may be
        either, and nothing (the path ends: a ``ValueError``, not a property class) when no pair of values makes the range non-empty.
        A float, a str, ``None`` and an object are a ``TypeError`` (recorded), as is a ``step`` given with ``stop=None``."""
        bound = self._random_args(st, n, a, k, "randrange")
        if bound is None:
            return BOT
        start, stop, step = bound["start"], bound.get("stop"), bound.get("step")
        bad = False

        def index(v, allow_none=False):
            nonlocal bad
            bad = bad or bool(v.f or v.s is not None or v.refs or v.st or (v.n and not allow_none))
            ii = self._as_int(st, v)
            return None if ii is None else self.ibounds(st, ii)

        s_b = index(start)
        t_none = stop is None or bool(stop.n)  # stop=None (the default, or given)
        t_b = index(stop, allow_none=True) if stop is not None else None
        p_b = index(step) if step is not None else (1, 1)
        if step is not None and t_none:
            # ``step is not _ONE``: only the very int 1 passes (a bool, a float, any other int is the TypeError)
            one = step.i is not None and not (step.b or step.f or step.s is not None or step.n or step.refs or step.st) \
                and p_b is not None and p_b == (1, 1)
            bad = bad or not one
        self.check(n, "arg-type", {"TypeError"}, not bad, "randrange of " + ",".join(start.kinds()) + (
            "" if stop is None else "," + ",".join(stop.kinds())) + ("" if step is None else "," + ",".join(step.kinds())))
        if s_b is None:
            st.set_bottom()  # `_index(start)` is a TypeError on every run
            return BOT
        s_lo, s_hi = s_b
        parts = []
        if stop is None or t_none:
            # randbelow(istart): istart > 0; the step, if given, has to be the int 1 for this alternative to continue
            step_may_be_one = step is None or (p_b is not None and (p_b[0] is None or p_b[0] <= 1)
                                               and (p_b[1] is None or p_b[1] >= 1) and step.i is not None)
            if step_may_be_one and (s_hi is None or s_hi >= 1):
                parts.append((0, None if s_hi is None else s_hi - 1))
        if t_b is not None and p_b is not None:
            t_lo, t_hi = t_b
            p_lo, p_hi = p_b
            if p_hi is None or p_hi >= 1:  # a positive step: istart, istart + step, ... < istop
                if t_hi is None or s_lo is None or t_hi > s_lo:
                    parts.append((s_lo, None if t_hi is None else t_hi - 1))
            if p_lo is None or p_lo <= -1:  # a negative step: istart, istart + step, ... > istop
                if t_lo is None or s_hi is None or t_lo < s_hi:
                    parts.append((None if t_lo is None else t_lo + 1, s_hi))
        if not parts:
            st.set_bottom()  # an empty range, a zero step, or a TypeError: on every run (a ValueError is not a property class)
            return BOT
        lo, hi = self._hull(parts)
        return int_iv(lo, hi)

    def m_random_getrandbits(self, st, a, k, n):
        """``getrandbits(k, /)``: an int in ``[0, 2**k - 1]`` (``0`` for ``k == 0``); a negative ``k`` is a ``ValueError``, a float,
        str or ``None`` a ``TypeError`` (the argument is read with ``__index__``).  A ``k`` beyond 4096 is given no upper bound
        (a result of that many bits is never built)."""
        bound = self._random_args(st, n, a, k, "getrandbits")
        if bound is None:
            return BOT
        v = bound["k"]
        bad = bool(v.f or v.s is not None or v.n or v.refs or v.st)
        self.check(n, "arg-type", {"TypeError"}, not bad, "getrandbits of " + ",".join(v.kinds()))
        ii = self._as_int(st, v)
        if ii is None:
            st.set_bottom()
            return BOT
        _, hi = self.ibounds(st, ii)
        if hi is not None and hi < 0:
            st.set_bottom()  # a negative k is a ValueError on every run
            return BOT
        return int_iv(0, None if hi is None or hi > 4096 else (1 << hi) - 1)
