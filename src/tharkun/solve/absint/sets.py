"""``set`` and ``frozenset`` for ``Analyzer``: the heap kind of ``heap.py::SetObj`` and every operation on it
(docs/ABSINT.md, "Sets").

A set is a length interval and ONE summary of its elements, nothing more: ``x in s`` is not decided by the
elements except at the two ends (an empty set holds nothing; a set that certainly has an element and whose
elements are all one constant holds that constant), because a summary says what every element satisfies and
never which ones exist.  Everything else is exact over those two facts, including the places where CPython's
own behaviour is easy to get wrong:

* **Equal values are one element.**  ``{1, 1.0, True}`` has one element and either object may be the one kept
  (a union keeps the receiver's, an intersection the argument's), so an element summary joins the values of
  every operand that can supply one.
* **Hashing.**  Every element is hashed when it enters a set, so ``add([])`` is a ``TypeError`` (and the path
  past it does not exist), an instance of a class that defines ``__eq__`` is unhashable (``__hash__`` is then
  ``None``; a class that defines ``__hash__`` is refused when it is defined), and only ``remove``, ``discard``
  and ``in`` convert an unhashable SET to a frozenset first (``tests/contracts/test_set_.py`` pins which
  methods do).  No other user code can run: an element is a scalar, a tuple or frozenset of those, a class, a
  function, or an instance whose class has neither hook, so the six classes a contract admits only through a
  hook (``__hash__``, ``__eq__``, ``__iter__``) are proven wherever the arguments are in that set of kinds.
* **``remove`` and ``pop`` raise ``KeyError``** when the element is absent or the set is empty.  Both are
  recorded checks; ``pop`` is proven only when the length has a lower bound of 1, ``remove`` only when the
  element is certainly there (a set of at least one element whose elements are all the one constant asked for).
* **A frozenset** has the non-mutating methods only (a mutating one is an ``AttributeError``), is hashable,
  and ``frozenset(f)`` and ``f.copy()`` are ``f`` itself.

The contracts (``solve/contracts/rules_set.py``) are the authority on WHICH of the seven property classes each call
can raise, read through ``adapters/absint.py::contract_for_method`` and ``Analyzer._contract_checks`` exactly as
the reducers read theirs: every class starts as the contract states it (proven only for a TESTED ``never``),
and a model upgrades a class only for the reason it states.  A model refuses (``Unsupported``) what it does not
understand rather than guess.

What stays refused: an iterator is consumed by the call in CPython and not here (so a set built from one has
no lower bound), a dict view of ``items``, a key function, a set operator with a dict view operand (the
reflected operator of a view), an element of a kind whose hashability is not modelled (a bound method, a file,
a pattern), a starred element, and a ``set`` and a ``frozenset`` that may be the same operand.
"""

from __future__ import annotations

import ast

from .aval import ANY_STR, ANYSTR, BOOL, BOT, NONE, V, const_bool, int_iv
from .errors import Unsupported
from .heap import SetObj
from .reducers import NO_HOOK_CLASSES
from .state import vjoin
from .visitor import SetCtx

#: the property classes a contract admits only through a user hook: none can run on a set of the supported
#: element kinds (see the module docstring), so a model may prove them for the calls it understands
_HOOK_FREE = ("no user hook runs: the elements are scalars, tuples, frozensets, classes, functions or "
              "instances of a class with neither __hash__ nor __eq__")
_NO_HOOK = {exc: _HOOK_FREE for exc in NO_HOOK_CLASSES}

#: the non-dunder attributes of the two types (``tests/contracts/test_set_.py`` pins them against ``dir``)
FROZENSET_METHODS = frozenset({"copy", "difference", "intersection", "isdisjoint", "issubset", "issuperset",
                               "symmetric_difference", "union"})
SET_METHODS = FROZENSET_METHODS | {"add", "clear", "difference_update", "discard", "intersection_update", "pop",
                                   "remove", "symmetric_difference_update", "update"}

#: binary operator -> (contract name of the operator, of its in-place form)
_SET_OPERATORS = {ast.BitOr: ("or", "ior"), ast.BitAnd: ("and", "iand"), ast.Sub: ("sub", "isub"),
                  ast.BitXor: ("xor", "ixor")}
_ORDERINGS = {ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge"}
_MANY = 10 ** 9


def _add(a, b):
    return None if a is None or b is None else a + b


def _sub(a, b):
    """``a - b`` for bounds, ``None`` (unbounded) when either is."""
    return None if a is None or b is None else a - b


def _min_hi(a, b):
    """The smaller of two upper bounds (``None`` is unbounded)."""
    if a is None:
        return b
    return a if b is None else min(a, b)


def _lo(x) -> int:
    return 0 if x is None else x


class SetsMixin:
    """Mixed into ``interp.py::Analyzer``; every method uses only Analyzer methods and the shared value types."""

    # ------------------------------------------------------------------ facts about a set
    def _set_objs(self, st, v: V):
        """The ``SetObj`` of every alternative of ``v`` when ``v`` is nothing but sets, else ``None``."""
        if v.top or v.st or v.s is not None or v.i is not None or v.b or v.f or v.n or not v.refs:
            return None
        obs = [st.heap[a] for a in v.refs]
        return obs if all(ob.kind == "set" for ob in obs) else None

    @staticmethod
    def _set_bounds(obs):
        """The length bounds over every set in ``obs``: ``(lo, hi)``, ``hi`` ``None`` when unbounded."""
        lo = min(ob.length[0] for ob in obs)
        his = [ob.length[1] for ob in obs]
        return lo, (None if any(h is None for h in his) else max(his))

    def _set_elem(self, st, obs) -> V:
        out = BOT
        for ob in obs:
            out = vjoin(out, st.zone, ob.elem, st.zone)
        return out

    def _const_value(self, st, v: V):
        """``(True, value)`` when ``v`` is exactly one int, bool, str or None constant, else ``(False, None)``.
        A float is never one: ``nan`` is a member of a set only as the very same object, so no abstract value
        of it is a certain member."""
        if v.top or v.refs or v.st or v.f:
            return False, None
        vals = []
        if v.i is not None:
            lo, hi = self.ibounds(st, v.i)
            if lo is None or lo != hi:
                return False, None
            vals.append(lo)
        if v.b:
            if len(v.b) != 1:
                return False, None
            vals.append(next(iter(v.b)))
        if v.s is not None:
            if v.s == ANY_STR or len(v.s) != 1:
                return False, None
            vals.append(next(iter(v.s)))
        if v.n:
            vals.append(None)
        if len(vals) != 1:
            return False, None
        return True, vals[0]

    def _certain_member(self, st, ob, x: V) -> bool:
        """``x`` is certainly an element of ``ob``: the set has an element and every element is the one
        constant ``x`` is (``True`` equals ``1`` and hashes alike, so they are one element)."""
        if ob.length[0] < 1:
            return False
        cx, a = self._const_value(st, x)
        ce, b = self._const_value(st, ob.elem)
        return cx and ce and a == b

    # ------------------------------------------------------------------ hashing
    def _hashable(self, st, v: V, node, *, allow_set: bool = False, _depth: int = 0):
        """``(the alternatives of v that can be hashed, whether every alternative can)``.  A scalar, a class, a
        function and a static object hash; a list, a dict, an array, a dict keys or items view and a (non-frozen) set do not
        (``allow_set``: the call converts a set to a frozenset first); a tuple hashes when its elements do; an
        instance hashes unless its class defines ``__eq__`` (a class that defines ``__hash__`` never gets this
        far: ``s_ClassDef`` refuses it); every other kind is refused rather than guessed at.

        An alternative that hashes only for SOME values of its parts (a tuple one of whose parts is ``list | int``) is
        neither: it stays in the value (the path where its parts hash exists) and the second result is ``False`` (a
        ``TypeError`` is possible, never proven away).  Only an alternative that CERTAINLY cannot hash is dropped."""
        if v.top:
            raise Unsupported("a set element of unknown value", node)
        verdicts = {a: self._ref_hashable(st, a, node, allow_set, _depth) for a in v.refs}
        keep = [a for a, r in verdicts.items() if r is not False]     # False: certainly unhashable; None: may or may not (a tuple with a part that may not hash)
        all_ok = all(r is True for r in verdicts.values())
        if len(keep) == len(v.refs):
            return v, all_ok
        return v.replace(refs=frozenset(keep)), False

    def _ref_hashable(self, st, a, node, allow_set: bool, depth: int) -> bool | None:
        """Whether the object at ``a`` hashes: ``True`` always, ``False`` never, ``None`` for some values of its parts
        (a tuple with a part that may or may not hash)."""
        ob = st.heap[a]
        k = ob.kind
        if k in ("list", "dict", "array"):
            return False
        if k == "dictview":
            return ob.which == "values"       # keys and items views are set-like (they define __eq__): unhashable; values hash by identity
        if k == "set":
            return ob.frozen or allow_set
        if k == "tuple":
            if depth > 4:
                raise Unsupported("a set element of a deeply nested tuple", node)
            if ob.items is not None:
                parts, lo = ob.items, len(ob.items)
            else:
                parts, lo = (ob.elem,), _lo(self.len_lin(st, ob)[1][0])       # a summary: the join of the elements, and how few there are
            res = [(p, *self._hashable(st, p, node, _depth=depth + 1)) for p in parts]
            if any(kept.is_bot() and not p.is_bot() for p, kept, _ok in res):
                # a part that is a value and certainly cannot hash.  The tuple certainly cannot hash only if it has that
                # part: a summary that may be empty (``tuple(l)`` of a list grown in a loop) hashes when it is.
                return False if lo >= 1 else None
            return True if all(ok for _p, _kept, ok in res) else None    # None: it hashes only for some values of its parts
        if k in ("range", "class", "func"):
            return True
        if k == "inst":
            return self.class_lookup(st, ob.cls, "__eq__") is None
        raise Unsupported(f"a {k} as a set element: its hashability is not modelled", node)

    # ------------------------------------------------------------------ the iterables a set method takes
    def _source(self, st, v: V, node, consumer: str):
        """What a set method reads from an iterable argument: ``(lo, hi, dlo, elem, all_sets, ok)`` -- the bounds
        of how many elements it yields, the fewest DISTINCT values it can yield (a set, a dict's keys and a range
        yield distinct ones; a list or a dict's values may repeat one), the join of the elements (``BOT`` when it
        yields none), whether every alternative is a set, and whether it is certainly iterable.  ``None`` when no
        alternative is iterable (the call always raises).  An iterator yields what is left of its source: none,
        or any number of them."""
        if v.top:
            raise Unsupported(f"{consumer}() of an unknown value", node)
        v = self.iterable(st, v, node, consumer=consumer)
        ok = self.check_iterable(st, v, node)
        los, his, dlos = [], [], []
        elem = BOT
        for a in sorted(v.refs, key=repr):
            ob = st.heap[a]
            k = ob.kind
            if k in ("list", "tuple", "array"):
                lo, hi = self.len_lin(st, ob)[1]
                lo = _lo(lo)
                dlo = min(lo, 1)
                e = self.all_elems(st, ob)
            elif k == "range":
                lo, hi = self._range_len(st, ob)[1]
                lo = _lo(lo)
                dlo = lo
                e = self.elem_summary(st, V(refs=frozenset({a})))
            elif k == "dict":
                lo, hi = self.dict_len(st, {a})
                dlo = lo
                e = self.dict_keys_v(st, ob)
            elif k == "dictview":
                lo, hi = self.dict_len(st, ob.dicts)
                dlo = lo if ob.which == "keys" else min(lo, 1)
                e = self.elem_summary(st, V(refs=frozenset({a})))
            elif k == "set":
                lo, hi = ob.length
                dlo = lo
                e = ob.elem
            elif k == "iterator":
                lo, hi, dlo = 0, None, 0
                e = self.elem_summary(st, V(refs=frozenset({a})))
            elif k in ("stringio", "outfile"):
                raise Unsupported(f"{consumer}() of a {k}", node)
            else:
                continue                      # not iterable: a TypeError on this alternative, no path through it
            los.append(lo)
            his.append(hi)
            dlos.append(dlo)
            if e is not None:
                elem = vjoin(elem, st.zone, e, st.zone)
        if v.s is not None:
            if v.s == ANY_STR:
                los.append(0)
                his.append(None)
                dlos.append(0)
            else:
                shortest = min(len(x) for x in v.s)
                los.append(shortest)
                his.append(max(len(x) for x in v.s))
                dlos.append(min(shortest, 1))
            elem = vjoin(elem, st.zone, ANYSTR, st.zone)
        if not los:
            return None
        hi = None if any(h is None for h in his) else max(his)
        all_sets = (v.s is None and not (v.i is not None or v.b or v.f or v.n or v.st)
                    and all(st.heap[a].kind == "set" for a in v.refs))
        return min(los), hi, min(dlos), elem, all_sets, ok

    def _sources(self, st, vs, node, consumer: str):
        """``_source`` of each argument: ``(rows, hash_ok, ok)`` where ``rows`` are ``(lo, hi, dlo, elem)`` with
        the elements that cannot be hashed removed, ``hash_ok`` says every element of every non-set argument can
        be hashed, ``ok`` that every argument is iterable.  ``None`` when some argument is never iterable."""
        rows, hash_ok, ok = [], True, True
        for v in vs:
            src = self._source(st, v, node, consumer)
            if src is None:
                return None
            lo, hi, dlo, elem, all_sets, iter_ok = src
            ok = ok and iter_ok
            if not all_sets and not elem.is_bot():
                kept, h = self._hashable(st, elem, node)
                hash_ok = hash_ok and h
                if kept.is_bot():
                    # everything it could yield is certainly unhashable: only an empty argument gets past it
                    if lo >= 1:
                        return None
                    lo = dlo = hi = 0
                elem = kept
            rows.append((lo, hi, dlo, elem))
        return rows, hash_ok, ok

    # ------------------------------------------------------------------ allocation
    @staticmethod
    def _tag(base: str, node) -> str:
        """An allocation-site tag that also names where ``node`` ends: ``a | b | c`` and ``a.union(b).union(c)``
        are nested nodes that START at the same column, and one site reached twice in a statement is refused."""
        return f"{base}@{getattr(node, 'end_lineno', 0)}:{getattr(node, 'end_col_offset', 0)}"

    def _alloc_set(self, st, node, lo, hi, elem: V, frozen: bool, tag: str):
        """A new set at the site of ``node``.  What the caller read from the heap names the old recent object
        of this site (``elem`` may hold a reference to it) and the allocation is about to turn that object into
        the site's summary: ``State.redirect`` rewrites the heap, not a Python local, so the reference follows."""
        site = self.site(node, tag)
        r, s = (site, "r"), (site, "s")

        def make(sid):
            e = elem.replace(refs=(elem.refs - {r}) | {s}) if r in elem.refs else elem
            return SetObj.make(lo, hi, e, frozen)

        return V(refs=frozenset({self.alloc(st, node, make, tag)}))

    def _contract_of(self, name: str, node):
        from tharkun.solve.contracts.adapters import absint as _contracts

        contract = _contracts.contract_for_method("set", name)
        if contract is None:
            raise Unsupported(f"set.{name} has no tested contract", node)
        return contract

    @staticmethod
    def _grounded(contract, proven: dict) -> dict:
        """``proven`` without the classes the generic "no user hook runs" argument cannot prove for this contract:
        it covers a class the contract admits only by PROPAGATION from a hook, never one the callable raises by
        its own logic (``KeyError`` of ``remove`` and ``pop``, ``TypeError`` of an unhashable element).  A model
        proves those with an argument of its own, which this keeps; a model that forgot to is held to the
        contract."""
        from tharkun.solve.contracts.adapters.common import native_raise_condition

        return {exc: why for exc, why in proven.items()
                if why != _HOOK_FREE or native_raise_condition(contract, exc) is None}

    def _via_rule(self, qualname: str, node, run):
        """The one entry through which absint reads a set rule (``visitor.py``): the rule ``qualname`` accepts this analyzer's
        visitor, whose ``visit_<kind>`` runs ``run`` (the transfer function of the operation).  A rule the visitor has no visit
        for would be a NAMED refusal (:class:`Refused`) reported as ``Unsupported``."""
        from tharkun.solve.contracts.rules import Refused, get_rule

        rule = get_rule(qualname)
        if rule is None:
            raise Unsupported(f"{qualname} has no rule", node)
        out = rule.accept(self.visitor, SetCtx(node, run))
        if isinstance(out, Refused):
            raise Unsupported(str(out), node)
        return out

    def _set_checks(self, node, name: str, proven: dict) -> None:
        """The seven checks of ``set.<name>`` (the contract, upgraded by ``proven``)."""
        contract = self._contract_of(name, node)
        self._contract_checks(node, f"set.{name}", contract, self._grounded(contract, proven))

    # ------------------------------------------------------------------ construction
    def e_Set(self, e, st, fctx):
        """A set display ``{a, b, c}``: each element is evaluated, hashed and added in order (a ``TypeError``
        for an unhashable one), and equal elements collapse: the length is exact when every element is a
        constant, else between 1 and the number of elements."""
        elem = BOT
        consts, ok = [], True
        for x in e.elts:
            if isinstance(x, ast.Starred):
                raise Unsupported("starred set item", e)
            v = self.eval(x, st, fctx)
            if st.bottom:
                return BOT
            v, hash_ok = self._hashable(st, v, x)
            ok = ok and hash_ok
            if v.is_bot():
                self._set_checks(e, "add", dict(_NO_HOOK))
                st.set_bottom()
                return BOT
            consts.append(self._const_value(st, v))
            elem = self.join_into(st, elem, v)
        proven = dict(_NO_HOOK)
        if ok:
            proven["TypeError"] = "every element is hashable"
        self._set_checks(e, "add", proven)
        n = len(e.elts)
        if n and all(c for c, _ in consts):
            lo = hi = len({val for _, val in consts})
        else:
            lo, hi = min(n, 1), n
        return self._alloc_set(st, e, lo, hi, elem, False, self._tag("set", e))

    def c_set(self, st, args, kwargs, node):
        return self._construct(st, args, kwargs, node, "set")

    def c_frozenset(self, st, args, kwargs, node):
        return self._construct(st, args, kwargs, node, "frozenset")

    def _construct(self, st, args, kwargs, node, name: str):
        """``set(iterable)`` / ``frozenset(iterable)`` (the contracts ``set`` and ``frozenset``): a NEW set of the
        distinct elements; ``frozenset(f)`` of an exact frozenset is ``f`` itself.  The elements of a non-set
        argument are hashed (a ``TypeError`` for an unhashable one); the length is between 1 (when the argument
        yields anything) and the number of elements it yields, and the exact count of distinct values when the
        argument is a set, a range or a dict's keys."""
        frozen = name == "frozenset"
        if not self._arity(node, args, kwargs, 0, 1, name):
            st.set_bottom()
            return BOT, self._grounded(self._builtin_contract(name, node), dict(_NO_HOOK))
        tag = self._tag(name, node)
        if not args:
            return self._alloc_set(st, node, 0, 0, BOT, frozen, tag), dict(_NO_HOOK, TypeError="no argument")
        v = args[0]
        if frozen and v.refs and not v.replace(refs=frozenset()).kinds():
            exact = {(st.heap[a].kind == "set" and st.heap[a].frozen) for a in v.refs}
            if exact == {True}:
                return v, dict(_NO_HOOK, TypeError="the argument is an exact frozenset, returned unchanged")
            if True in exact:
                raise Unsupported("frozenset() of a frozenset or something else", node)
        got = self._sources(st, [v], node, name)
        if got is None:
            st.set_bottom()
            return BOT, self._grounded(self._builtin_contract(name, node), dict(_NO_HOOK))
        rows, hash_ok, ok = got
        lo, hi, dlo, elem = rows[0]
        proven = dict(_NO_HOOK)
        if ok and hash_ok:
            proven["TypeError"] = "the argument is iterable and every element hashes"
        return self._alloc_set(st, node, dlo, hi, elem, frozen, tag), self._grounded(self._builtin_contract(name, node), proven)

    def _builtin_contract(self, name: str, node):
        from tharkun.solve.contracts.adapters import absint as _contracts

        contract = _contracts.contract_for_builtin(name)
        if contract is None:
            raise Unsupported(f"{name}() has no tested contract", node)
        return contract

    # ------------------------------------------------------------------ methods
    def set_method(self, st, addrs, name: str, args, kwargs, node) -> V:
        """``<set or frozenset>.<name>(...)`` on the receivers ``addrs`` (one kind among them: ``call_method``
        and ``attr_load`` have already answered the ``AttributeError`` of a mutating method on a frozenset)."""
        model = getattr(self, "sm_" + name, None)
        if model is None:
            raise Unsupported(f"set.{name} is not modelled", node)
        addrs = frozenset(addrs)

        def run():
            result, proven = model(st, addrs, args, kwargs, node)
            self._set_checks(node, name, proven)
            return result

        return self._via_rule(f"set.{name}", node, run)

    def _receivers(self, st, addrs):
        obs = [st.heap[a] for a in sorted(addrs, key=repr)]
        lo, hi = self._set_bounds(obs)
        return obs, lo, hi, self._set_elem(st, obs)

    def _no_args(self, st, args, kwargs, node, name: str, lo: int = 0, hi: int = 0):
        """``_arity``; a wrong count or a keyword is a ``TypeError`` raised before anything runs, so the path
        ends (the caller returns ``(BOT, dict(_NO_HOOK))``: no hook ran either)."""
        if self._arity(node, args, kwargs, lo, hi, name):
            return True
        st.set_bottom()
        return False

    # ---- element methods
    def sm_add(self, st, addrs, args, kwargs, node):
        if not self._no_args(st, args, kwargs, node, "add", 1, 1):
            return BOT, dict(_NO_HOOK)
        x, ok = self._hashable(st, args[0], node)
        if x.is_bot():
            st.set_bottom()                                   # an unhashable element: always a TypeError
            return BOT, dict(_NO_HOOK)

        def upd(ob, strong):
            lo, hi = ob.length
            return SetObj.make(max(lo, 1), _add(hi, 1), self.join_into(st, ob.elem, x), ob.frozen)

        self.update_obj(st, addrs, upd)
        proven = dict(_NO_HOOK)
        if ok:
            proven["TypeError"] = "the element is hashable"
        return NONE, proven

    def sm_discard(self, st, addrs, args, kwargs, node):
        if not self._no_args(st, args, kwargs, node, "discard", 1, 1):
            return BOT, dict(_NO_HOOK)
        x, ok = self._hashable(st, args[0], node, allow_set=True)
        if x.is_bot():
            st.set_bottom()                                   # an unhashable element: always a TypeError
            return BOT, dict(_NO_HOOK)
        self.update_obj(st, addrs, lambda ob, strong: SetObj.make(max(ob.length[0] - 1, 0), ob.length[1], ob.elem,
                                                                   ob.frozen))
        proven = dict(_NO_HOOK)
        if ok:
            proven["TypeError"] = "the element is hashable (or a set, which is converted)"
        return NONE, proven

    def sm_remove(self, st, addrs, args, kwargs, node):
        if not self._no_args(st, args, kwargs, node, "remove", 1, 1):
            return BOT, dict(_NO_HOOK)
        x, ok = self._hashable(st, args[0], node, allow_set=True)
        if x.is_bot():
            st.set_bottom()                                   # an unhashable element: always a TypeError
            return BOT, dict(_NO_HOOK)
        obs, lo, hi, elem = self._receivers(st, addrs)
        proven = dict(_NO_HOOK)
        if ok:
            proven["TypeError"] = "the element is hashable (or a set, which is converted)"
        if all(self._certain_member(st, ob, x) for ob in obs):
            proven["KeyError"] = "the set certainly has the element: every element is that one constant"
        if all(hi_ == 0 for hi_ in (ob.length[1] for ob in obs)):
            # no element to remove: every path raises KeyError (recorded unproven by the contract's own claim)
            st.set_bottom()
            return BOT, proven
        # the path that continues had the element: one fewer, and still a set of at least one when it had two
        self.update_obj(st, addrs, lambda ob, strong: SetObj.make(
            max(ob.length[0] - 1, 0), None if ob.length[1] is None else max(ob.length[1] - 1, 0), ob.elem,
            ob.frozen))
        return NONE, proven

    def sm_pop(self, st, addrs, args, kwargs, node):
        if not self._no_args(st, args, kwargs, node, "pop"):
            return BOT, dict(_NO_HOOK)
        obs, lo, hi, elem = self._receivers(st, addrs)
        proven = dict(_NO_HOOK)
        proven["TypeError"] = "pop() takes no arguments and there are none"
        if lo >= 1:
            proven["KeyError"] = f"the set has at least {lo} element(s)"
        if hi == 0:
            st.set_bottom()                                   # always empty: always a KeyError
            return BOT, proven
        # an element exists on the path that continues, so its bounds may become constraints (as list.pop's)
        out = self.mat(st, elem)
        self.update_obj(st, addrs, lambda ob, strong: SetObj.make(
            max(ob.length[0] - 1, 0), None if ob.length[1] is None else max(ob.length[1] - 1, 0), ob.elem,
            ob.frozen))
        return out, proven

    def sm_clear(self, st, addrs, args, kwargs, node):
        if not self._no_args(st, args, kwargs, node, "clear"):
            return BOT, dict(_NO_HOOK)
        self.update_obj(st, addrs, lambda ob, strong: SetObj.make(0, 0, BOT, ob.frozen))
        return NONE, dict(_NO_HOOK, TypeError="clear() takes no arguments and there are none")

    def sm_copy(self, st, addrs, args, kwargs, node):
        if not self._no_args(st, args, kwargs, node, "copy"):
            return BOT, dict(_NO_HOOK)
        obs, lo, hi, elem = self._receivers(st, addrs)
        proven = dict(_NO_HOOK, TypeError="copy() takes no arguments and there are none")
        frozen = {ob.frozen for ob in obs}
        if frozen == {True}:
            return V(refs=addrs), proven                      # an exact frozenset is its own copy
        if frozen != {False}:
            raise Unsupported("copy() of a set or a frozenset", node)
        return self._alloc_set(st, node, lo, hi, elem, False, self._tag("set.copy", node)), proven

    # ---- methods that take iterables
    def _iterables(self, st, addrs, args, kwargs, node, name: str, lo: int, hi: int):
        """The receivers and the argument iterables of a method that takes ``lo``..``hi`` of them: ``(obs,
        recv_lo, recv_hi, recv_elem, rows, proven)`` or ``None`` when the call always raises (``st`` is then
        bottom).  ``proven`` has the six hook-free classes, and ``TypeError`` when every argument is iterable and
        hashes."""
        if not self._arity(node, args, kwargs, lo, hi, name):
            st.set_bottom()
            return BOT, dict(_NO_HOOK)
        got = self._sources(st, list(args), node, f"set.{name}")
        if got is None:
            st.set_bottom()
            return BOT, dict(_NO_HOOK)
        rows, hash_ok, ok = got
        obs, rlo, rhi, relem = self._receivers(st, addrs)
        proven = dict(_NO_HOOK)
        if ok and hash_ok:
            proven["TypeError"] = "every argument is iterable and every element hashes"
        return obs, rlo, rhi, relem, rows, proven

    def _union_facts(self, rlo, rhi, relem, rows, st):
        """Bounds and elements of ``receiver | each argument``."""
        lo, hi, elem = rlo, rhi, relem
        for alo, ahi, adlo, aelem in rows:
            lo = max(lo, adlo)
            hi = _add(hi, ahi)
            elem = vjoin(elem, st.zone, aelem, st.zone)
        return lo, hi, elem

    def sm_union(self, st, addrs, args, kwargs, node):
        got = self._iterables(st, addrs, args, kwargs, node, "union", 0, _MANY)
        if got[0] is BOT:
            return got
        obs, rlo, rhi, relem, rows, proven = got
        lo, hi, elem = self._union_facts(rlo, rhi, relem, rows, st)
        frozen = self._one_kind(obs, node, "union")
        return self._alloc_set(st, node, lo, hi, elem, frozen, self._tag("set.union", node)), proven

    def sm_update(self, st, addrs, args, kwargs, node):
        got = self._iterables(st, addrs, args, kwargs, node, "update", 0, _MANY)
        if got[0] is BOT:
            return got
        obs, rlo, rhi, relem, rows, proven = got
        extra_lo = max((dlo for _lo, _hi, dlo, _e in rows), default=0)
        extra_hi = None if any(r[1] is None for r in rows) else sum(r[1] for r in rows)
        added = BOT
        for r in rows:
            added = vjoin(added, st.zone, r[3], st.zone)
        added = self.summarize(st, added)

        def upd(ob, strong):
            return SetObj.make(max(ob.length[0], extra_lo), _add(ob.length[1], extra_hi),
                               vjoin(ob.elem, st.zone, added, st.zone), ob.frozen)

        self.update_obj(st, addrs, upd)
        return NONE, proven

    def sm_intersection(self, st, addrs, args, kwargs, node):
        got = self._iterables(st, addrs, args, kwargs, node, "intersection", 0, _MANY)
        if got[0] is BOT:
            return got
        obs, rlo, rhi, relem, rows, proven = got
        lo, hi, elem = self._intersection_facts(rlo, rhi, relem, rows, st)
        frozen = self._one_kind(obs, node, "intersection")
        return self._alloc_set(st, node, lo, hi, elem, frozen, self._tag("set.intersection", node)), proven

    def _intersection_facts(self, rlo, rhi, relem, rows, st):
        """``receiver & each argument``: the result is a subset of each, and an element may be the argument's
        object (``{1}.intersection([1.0])`` is ``{1.0}``), so the elements are joined.  With no argument it is a
        copy."""
        if not rows:
            return rlo, rhi, relem
        hi, elem = rhi, relem
        for _alo, ahi, _adlo, aelem in rows:
            hi = _min_hi(hi, ahi)
            elem = vjoin(elem, st.zone, aelem, st.zone)
        return 0, hi, elem

    def sm_intersection_update(self, st, addrs, args, kwargs, node):
        got = self._iterables(st, addrs, args, kwargs, node, "intersection_update", 0, _MANY)
        if got[0] is BOT:
            return got
        obs, rlo, rhi, relem, rows, proven = got
        if rows:
            hi_arg, elem_arg = None, BOT
            for _alo, ahi, _adlo, aelem in rows:
                hi_arg = _min_hi(hi_arg, ahi)
                elem_arg = vjoin(elem_arg, st.zone, aelem, st.zone)
            elem_arg = self.summarize(st, elem_arg)

            def upd(ob, strong):
                return SetObj.make(0, _min_hi(ob.length[1], hi_arg), vjoin(ob.elem, st.zone, elem_arg, st.zone),
                                   ob.frozen)

            self.update_obj(st, addrs, upd)
        return NONE, proven

    def sm_difference(self, st, addrs, args, kwargs, node):
        got = self._iterables(st, addrs, args, kwargs, node, "difference", 0, _MANY)
        if got[0] is BOT:
            return got
        obs, rlo, rhi, relem, rows, proven = got
        lo = rlo
        for _alo, ahi, _adlo, _aelem in rows:
            lo = 0 if ahi is None else max(0, lo - ahi)
        frozen = self._one_kind(obs, node, "difference")
        return self._alloc_set(st, node, lo, rhi, relem, frozen, self._tag("set.difference", node)), proven

    def sm_difference_update(self, st, addrs, args, kwargs, node):
        got = self._iterables(st, addrs, args, kwargs, node, "difference_update", 0, _MANY)
        if got[0] is BOT:
            return got
        obs, rlo, rhi, relem, rows, proven = got
        his = [r[1] for r in rows]

        def upd(ob, strong):
            lo = ob.length[0]
            for ahi in his:
                lo = 0 if ahi is None else max(0, lo - ahi)
            return SetObj.make(lo, ob.length[1], ob.elem, ob.frozen)

        self.update_obj(st, addrs, upd)
        return NONE, proven

    @staticmethod
    def _symmetric_facts(rlo, rhi, relem, row, join):
        """``receiver ^ argument``: elements from either side; at least the difference of the lengths (the
        argument has between ``dlo`` distinct values and ``hi``), at most their sum."""
        _alo, ahi, adlo, aelem = row
        lo = max(0, 0 if ahi is None else rlo - ahi, 0 if rhi is None else adlo - rhi)
        return lo, _add(rhi, ahi), join(relem, aelem)

    def sm_symmetric_difference(self, st, addrs, args, kwargs, node):
        got = self._iterables(st, addrs, args, kwargs, node, "symmetric_difference", 1, 1)
        if got[0] is BOT:
            return got
        obs, rlo, rhi, relem, rows, proven = got
        lo, hi, elem = self._symmetric_facts(rlo, rhi, relem, rows[0], lambda a, b: vjoin(a, st.zone, b, st.zone))
        frozen = self._one_kind(obs, node, "symmetric_difference")
        return self._alloc_set(st, node, lo, hi, elem, frozen, self._tag("set.symmetric_difference", node)), proven

    def sm_symmetric_difference_update(self, st, addrs, args, kwargs, node):
        got = self._iterables(st, addrs, args, kwargs, node, "symmetric_difference_update", 1, 1)
        if got[0] is BOT:
            return got
        obs, rlo, rhi, relem, rows, proven = got
        row = rows[0]

        def upd(ob, strong):
            lo, hi, elem = self._symmetric_facts(ob.length[0], ob.length[1], ob.elem, row,
                                                 lambda a, b: vjoin(a, st.zone, self.summarize(st, b), st.zone))
            return SetObj.make(lo, hi, elem, ob.frozen)

        self.update_obj(st, addrs, upd)
        return NONE, proven

    # ---- predicates
    def _predicate(self, st, addrs, args, kwargs, node, name: str, decide):
        got = self._iterables(st, addrs, args, kwargs, node, name, 1, 1)
        if got[0] is BOT:
            return got
        obs, rlo, rhi, relem, rows, proven = got
        alo, ahi, adlo, _aelem = rows[0]
        truth = decide(rlo, rhi, alo, ahi, adlo)
        return (const_bool(truth) if truth is not None else BOOL), proven

    def sm_issubset(self, st, addrs, args, kwargs, node):
        # the receiver is inside the argument: certainly true for an empty receiver, certainly false when it is
        # longer than the argument can be
        return self._predicate(st, addrs, args, kwargs, node, "issubset",
                               lambda rlo, rhi, alo, ahi, adlo: True if rhi == 0 else (
                                   False if ahi is not None and rlo > ahi else None))

    def sm_issuperset(self, st, addrs, args, kwargs, node):
        return self._predicate(st, addrs, args, kwargs, node, "issuperset",
                               lambda rlo, rhi, alo, ahi, adlo: True if ahi == 0 else (
                                   False if rhi is not None and adlo > rhi else None))

    def sm_isdisjoint(self, st, addrs, args, kwargs, node):
        return self._predicate(st, addrs, args, kwargs, node, "isdisjoint",
                               lambda rlo, rhi, alo, ahi, adlo: True if rhi == 0 or ahi == 0 else None)

    @staticmethod
    def _one_kind(obs, node, name: str) -> bool:
        """Whether the result is a frozenset: the receiver's own type.  Receivers of both kinds at one call are
        not one site's one object."""
        kinds = {ob.frozen for ob in obs}
        if len(kinds) != 1:
            raise Unsupported(f"{name}() of a set or a frozenset", node)
        return kinds.pop()

    # ------------------------------------------------------------------ operators
    def set_binop(self, st, op, l: V, r: V, node, inplace: bool):
        """``l <op> r`` for the alternatives that are sets.  Returns ``(value, TypeError possible)``: ``value`` is
        the result of every (set, set) pair, ``BOT`` when there is none; every other pair with a set in it is a
        ``TypeError`` (a set operator accepts only a set or a frozenset, ``tests/contracts/test_set_.py``).  A
        dict view as the other operand is refused by ``binop`` before it gets here (its reflected operator)."""
        lobs = [(a, st.heap[a]) for a in sorted(l.refs, key=repr) if st.heap[a].kind == "set"]
        robs = [(b, st.heap[b]) for b in sorted(r.refs, key=repr) if st.heap[b].kind == "set"]
        if not lobs and not robs:
            return BOT, False

        def has_other(v: V) -> bool:
            return bool(v.s is not None or v.i is not None or v.b or v.f or v.n or v.st
                        or any(st.heap[x].kind != "set" for x in v.refs))

        bad = (bool(lobs) and has_other(r)) or (bool(robs) and has_other(l))
        names = _SET_OPERATORS.get(type(op))
        if not lobs or not robs or names is None:
            return BOT, True
        kinds = {ob.frozen for _a, ob in lobs}
        if len(kinds) != 1:
            raise Unsupported("a set operator whose left operand may be a set or a frozenset", node)
        frozen = kinds.pop()
        in_place = inplace and not frozen
        contract_name = f"__{names[1] if in_place else names[0]}__"
        aobs = [ob for _a, ob in lobs]
        bobs = [ob for _b, ob in robs]

        def run():
            blo, bhi = self._set_bounds(bobs)
            aelem, belem = self._set_elem(st, aobs), self._set_elem(st, bobs)
            symbol = type(op)
            elem_b = belem if symbol is not ast.Sub else BOT     # a difference keeps the receiver's own elements
            proven = dict(_NO_HOOK)
            if not bad:
                proven["TypeError"] = "both operands are sets"
            self._set_checks(node, contract_name, proven)
            if in_place:
                added = self.summarize(st, elem_b)
                self.update_obj(st, frozenset(a for a, _ob in lobs), lambda ob, strong: SetObj.make(
                    *self._operator_bounds(symbol, ob.length[0], ob.length[1], blo, bhi),
                    vjoin(ob.elem, st.zone, added, st.zone), ob.frozen))
                return V(refs=frozenset(a for a, _ob in lobs)), bad
            alo, ahi = self._set_bounds(aobs)
            lo, hi = self._operator_bounds(symbol, alo, ahi, blo, bhi)
            elem = self.summarize(st, vjoin(aelem, st.zone, elem_b, st.zone))
            return self._alloc_set(st, node, lo, hi, elem, frozen, self._tag("set" + names[0], node)), bad

        return self._via_rule(f"set.{contract_name}", node, run)

    @staticmethod
    def _operator_bounds(symbol, alo, ahi, blo, bhi):
        """The length bounds of ``a <op> b`` for sets of ``alo``..``ahi`` and ``blo``..``bhi`` elements."""
        if symbol is ast.BitOr:
            return max(alo, blo), _add(ahi, bhi)
        if symbol is ast.BitAnd:
            return 0, _min_hi(ahi, bhi)
        if symbol is ast.Sub:
            return (0 if bhi is None else max(0, alo - bhi)), ahi
        return max(0, 0 if bhi is None else alo - bhi, 0 if ahi is None else blo - ahi), _add(ahi, bhi)

    # ------------------------------------------------------------------ comparisons and membership
    def set_ordering(self, st, op, lv: V, rv: V, node, t, f) -> bool:
        """``<``, ``<=``, ``>``, ``>=`` between two sets: a subset test, decided at the ends the lengths decide
        (``a <= b`` is false when ``a`` is certainly longer, true when ``a`` is empty).  Returns whether it
        handled the comparison (both operands nothing but sets)."""
        a, b = self._set_objs(st, lv), self._set_objs(st, rv)
        if a is None or b is None:
            return False
        name = _ORDERINGS[type(op)]

        def run():
            alo, ahi = self._set_bounds(a)
            blo, bhi = self._set_bounds(b)
            if name in ("gt", "ge"):                          # a > b is b < a
                alo, ahi, blo, bhi = blo, bhi, alo, ahi
            strict = name in ("lt", "gt")
            truth = None
            if ahi == 0 and (blo >= 1 or not strict):
                truth = True
            elif bhi is not None and (alo >= bhi if strict else alo > bhi):
                truth = False
            proven = dict(_NO_HOOK, TypeError="both operands are sets")
            self._set_checks(node, f"__{name}__", proven)
            if truth is True:
                f.set_bottom()
            elif truth is False:
                t.set_bottom()
            return True

        return self._via_rule(f"set.__{name}__", node, run)

    def set_equality(self, st, op, lv: V, rv: V, node, t, f) -> bool:
        """``==`` and ``!=`` between two sets: certainly unequal when the lengths cannot match, certainly equal
        when both are empty.  Never an error for any operand.  Returns whether both are nothing but sets."""
        a, b = self._set_objs(st, lv), self._set_objs(st, rv)
        if a is None or b is None:
            return False
        name = "__eq__" if isinstance(op, ast.Eq) else "__ne__"

        def run():
            alo, ahi = self._set_bounds(a)
            blo, bhi = self._set_bounds(b)
            equal = None
            if ahi == 0 and bhi == 0:
                equal = True
            elif (bhi is not None and alo > bhi) or (ahi is not None and blo > ahi):
                equal = False
            proven = dict(_NO_HOOK, TypeError="no operand has an __eq__ of its own to run")
            self._set_checks(node, name, proven)
            if equal is not None:
                truth = equal if isinstance(op, ast.Eq) else not equal
                (f if truth else t).set_bottom()
            return True

        return self._via_rule(f"set.{name}", node, run)

    def set_membership(self, st, lv: V, rv: V, node):
        """``lv in rv`` for ``rv`` nothing but sets: the TypeError check of the contract (``lv`` hashes, or is a
        set, which is converted), and the truth value when it is decided: false for an empty set, true when ``lv``
        is a certain member.  Returns ``None`` when ``rv`` is not nothing but sets, else ``True``/``False``/
        ``"unknown"``."""
        obs = self._set_objs(st, rv)
        if obs is None:
            return None

        def run():
            x, ok = self._hashable(st, lv, node, allow_set=True)
            proven = dict(_NO_HOOK)
            if ok:
                proven["TypeError"] = "the element is hashable (or a set, which is converted)"
            self._set_checks(node, "__contains__", proven)
            if all(ob.length[1] == 0 for ob in obs):
                return False
            if all(self._certain_member(st, ob, lv) for ob in obs):
                return True
            return "unknown"

        return self._via_rule("set.__contains__", node, run)
