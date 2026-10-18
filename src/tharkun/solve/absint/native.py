"""Finite builtin values for str/format/re rule visits; never execute target objects."""
from __future__ import annotations

import itertools
import re
from .aval import ANY_STR, BOT, FloatInfo, NONE, V, const_bool, const_int, const_str
from .heap import DictObj
from .linear import IntInfo
from .state import vjoin
from .. import values as facts

# ponytail: enumerate at most 64 alternatives; add a relational string domain if this ceiling matters.
LIMIT = 64


def product(pools):
    if any(x is None for x in pools):
        return None
    size = 1
    for x in pools:
        size *= len(x)
        if size > LIMIT:
            return None
    return list(itertools.product(*pools))


def concrete(analyzer, state, value, seen=frozenset()):
    if value.top:
        return None
    out = []
    if value.i is not None:
        lo, hi = analyzer.ibounds(state, value.i)
        if lo is None or hi is None or hi - lo >= LIMIT:
            return None
        out.extend(range(lo, hi + 1))
    if value.b:
        out.extend(value.b)
    if value.s is not None:
        if value.s == ANY_STR:
            return None
        out.extend(value.s)
    if value.n:
        out.append(None)
    if value.f:
        ff = value.f.fact
        if ff.nan or ff.lo is None or ff.lo != ff.hi:
            return None
        out.append(ff.lo)
    for tag in value.st:
        if tag[0] not in {"native-regex", "native-slice", "native-flag"}:
            return None
        out.append(tag[1])
    for addr in value.refs:
        if addr in seen:
            return None
        obj = state.heap[addr]
        if obj.kind == "tuple" and obj.items is not None:
            options = product([concrete(analyzer, state, x, seen | {addr}) for x in obj.items])
            if options is None:
                return None
            out.extend(options)
        elif type(obj) is DictObj and obj.exact:
            keys = list(obj.items)
            options = product([concrete(analyzer, state, obj.items[k], seen | {addr}) for k in keys])
            if options is None:
                return None
            out.extend(dict(zip(keys, xs)) for xs in options)
        elif obj.kind == "list":
            lo, hi = analyzer.len_lin(state, obj)[1]
            if lo != hi or lo is None or lo > 8 or obj.pre is not None:
                return None
            elems = concrete(analyzer, state, obj.elem, seen | {addr}) if lo else []
            options = product([elems] * lo)
            if options is None:
                return None
            out.extend(list(xs) for xs in options)
        else:
            return None
    return out if len(out) <= LIMIT else None


def abstract(analyzer, state, outputs, node, path="result"):
    out = BOT
    containers = {}
    for value in outputs:
        t = type(value)
        if value is None:
            part = NONE
        elif t is bool:
            part = const_bool(value)
        elif t is int:
            part = const_int(value)
        elif t is float:
            part = V(f=FloatInfo(facts.FloatFact.const(value)))
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
    for t, values in containers.items():
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
