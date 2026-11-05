"""Finite builtin values for str/format/re rule visits; never execute target objects."""
from __future__ import annotations

import itertools
import re
from .aval import ANY_STR, BOT, FloatInfo, NONE, V, const_bool, const_int, const_str
from .heap import DictObj, TupleObj
from .linear import IntInfo
from .state import vjoin
from .. import values as facts


# Optional precision shares one allocation-work allowance across the analysis.
NATIVE_WORK_BUDGET = 200_000


class _NativeWorkExhausted(Exception):
    pass


def _spend_work(analyzer, units=1):
    remaining = getattr(analyzer, '_native_work_remaining', NATIVE_WORK_BUDGET) - units
    analyzer._native_work_remaining = remaining
    if remaining < 0:
        raise _NativeWorkExhausted


def _check_work(analyzer):
    if analyzer._clock() > analyzer.deadline:
        from .errors import Unsupported
        raise Unsupported("native materialization time budget exhausted")


def _work_items(analyzer, values, *, charge=True):
    _check_work(analyzer)
    iterator = iter(values)
    while True:
        _check_work(analyzer)
        try:
            value = next(iterator)
        except StopIteration:
            return
        if charge:
            _spend_work(analyzer)
        yield value


def product(pools, *, analyzer):
    """Complete finite alternatives, or exhaust the caller's shared analysis budget."""
    try:
        _check_work(analyzer)
        _spend_work(analyzer)
        checked = []
        for pool in _work_items(analyzer, pools):
            if pool is None:
                return None
            checked.append(_work_items(analyzer, pool))
        rows = []
        for row in _work_items(analyzer, itertools.product(*checked)):
            _spend_work(analyzer, len(row))
            rows.append(row)
        _check_work(analyzer)
        return rows
    except _NativeWorkExhausted:
        return None
    except MemoryError:
        from .errors import Unsupported
        raise Unsupported("native materialization memory exhausted") from None


def concrete(analyzer, state, value, seen=frozenset()):
    _check_work(analyzer)
    try:
        _spend_work(analyzer)
        return _concrete(analyzer, state, value, seen)
    except _NativeWorkExhausted:
        return None
    except MemoryError:
        from .errors import Unsupported
        raise Unsupported("native materialization memory exhausted") from None


def _concrete(analyzer, state, value, seen):
    if value.top:
        return None
    out = []
    if value.i is not None:
        lo, hi = analyzer.ibounds(state, value.i)
        if lo is None or hi is None:
            return None
        out.extend(_work_items(analyzer, range(lo, hi + 1)))
    if value.b:
        out.extend(value.b)
    if value.s is not None:
        if value.s == ANY_STR:
            return None
        out.extend(_work_items(analyzer, value.s))
    if value.n:
        out.append(None)
    if value.f:
        ff = value.f.fact
        if ff.nan or ff.lo is None or ff.lo != ff.hi:
            return None
        if ff.lo == 0.0:
            term = value.f.term
            if (isinstance(term, tuple) and len(term) == 2
                    and term[0] == "zero-sign" and type(term[1]) is bool):
                out.append(-0.0 if term[1] else 0.0)
            else:
                # Numeric zero bounds alone do not identify the IEEE sign bit.
                out.extend((0.0, -0.0))
        else:
            out.append(ff.lo)
    for tag in _work_items(analyzer, value.st):
        if tag[0] not in {"native-regex", "native-slice", "native-flag"}:
            return None
        out.append(tag[1])
    for addr in _work_items(analyzer, value.refs):
        if addr in seen:
            return None
        obj = state.heap[addr]
        if type(obj) is TupleObj and obj.items is not None:
            options = product([concrete(analyzer, state, x, seen | {addr})
                               for x in _work_items(analyzer, obj.items)], analyzer=analyzer)
            if options is None:
                return None
            out.extend(_work_items(analyzer, options))
        elif type(obj) is DictObj and obj.exact:
            keys = list(_work_items(analyzer, obj.items))
            options = product([concrete(analyzer, state, obj.items[k], seen | {addr})
                               for k in _work_items(analyzer, keys)], analyzer=analyzer)
            if options is None:
                return None
            out.extend(dict(zip(keys, xs)) for xs in _work_items(analyzer, options))
        elif obj.kind == "list":
            lo, hi = analyzer.len_lin(state, obj)[1]
            if lo != hi or lo is None or obj.pre is not None:
                return None
            elems = concrete(analyzer, state, obj.elem, seen | {addr}) if lo else []
            if lo and not elems:
                return None
            if lo and len(elems) == 1:
                out.append([elems[0] for _ in _work_items(analyzer, range(lo))])
            else:
                pools = [elems for _ in _work_items(analyzer, range(lo))]
                options = product(pools, analyzer=analyzer)
                if options is None:
                    return None
                out.extend(list(xs) for xs in _work_items(analyzer, options))
        else:
            return None
    _check_work(analyzer)
    return out


def abstract(analyzer, state, outputs, node, path="result"):
    out = BOT
    containers = {}
    for value in _work_items(analyzer, outputs, charge=False):
        t = type(value)
        if value is None:
            part = NONE
        elif t is bool:
            part = const_bool(value)
        elif t is int:
            part = const_int(value)
        elif t is float:
            term = ("zero-sign", value.hex().startswith("-")) if value == 0.0 else None
            part = V(f=FloatInfo(facts.FloatFact.const(value), term))
        elif t is str:
            part = const_str(value)
        elif t in (list, tuple, dict):
            containers.setdefault(t, []).append(value)
            continue
        elif t is re.RegexFlag:
            # [internal ref]: numeric view plus enum alternative; never erase repr/str/invert semantics.
            part = const_int(int(value)).replace(st=frozenset({("native-flag", value)}))
        elif t in (re.Pattern, re.Match):
            part = V(st=frozenset({("native-regex", value)}))
        else:
            return None
        out = vjoin(out, state.zone, part, state.zone)
    for t, values in _work_items(analyzer, containers.items(), charge=False):
        tag = "native:" + path + ":" + t.__name__
        if t is dict:
            keys = set(values[0])
            if any(set(x) != keys for x in values):
                return None
            items = {k: abstract(analyzer, state, [x[k] for x in values], node, path + repr(k)) for k in keys}
            if any(x is None for x in items.values()):
                return None
            addr = analyzer.alloc(state, node, lambda sid: DictObj(items), tag)
        elif t is tuple:
            if len({len(x) for x in values}) != 1:
                return None
            items = [abstract(analyzer, state, [x[i] for x in values], node, path + str(i)) for i in range(len(values[0]))]
            if any(x is None for x in items):
                return None
            addr = analyzer.new_list(state, node, None, BOT, tag=tag, kind="tuple", items=items)
        else:
            elem = abstract(analyzer, state, [e for x in values for e in x], node, path + "element")
            if elem is None:
                return None
            length = IntInfo.interval(min(map(len, values)), max(map(len, values)))
            addr = analyzer.new_list(state, node, length, elem, tag=tag)
        out = vjoin(out, state.zone, V(refs=frozenset({addr})), state.zone)
    return out
