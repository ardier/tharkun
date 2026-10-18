#!/usr/bin/env python3
"""Differential: eregion's ``reduce_domain`` vs tharkun's ``solve.guard_domain`` ([internal ref]).

This runs over every guard assume in the eregion toy corpus.  The eregion side is
``eregion_ir.program_extract.domain_capture.reduce_domain`` applied to the IR expr
dict in ``fixtures/02_ir.json``.  The tharkun side is ``guard_domain.domain_of``
applied to the matching test in the toy's ``source.py``: the ``if`` / ``while`` /
``assert`` at the assume's ``loc``, with its polarity taken from concrete
agreement with the IR predicate.  When no source test matches (the IR predicate
is a rewrite rather than the test itself), the IR predicate rendered back to Python
is used and the case is counted as ``rendered``.

Each (predicate, variable) case is compared over a pool of concrete values:

* ``tharkun_unsound``: a value takes the arm but tharkun's domain drops it.  Must be 0.
* ``narrower``: eregion keeps a value that takes the arm and tharkun drops it.
  This implies tharkun is unsound, so it must also be 0.
* ``tighter``: eregion keeps a value that does NOT take the arm (it raises or it
  is false) and tharkun drops it.  That is a precision gain.
* ``wider``: tharkun keeps a value eregion drops.  Each case gets a reason:
  ``eregion_unsound`` (the value takes the arm, so eregion is wrong to drop it),
  ``other_type`` (the value is outside eregion's type family; tharkun is open world),
  or ``precision`` (in the family and not taking the arm; listed in full).
* ``eregion_unsound``: a value that takes the arm is outside eregion's domain.  These
  are split into ``by_type`` (outside the domain's type family, e.g. ``3.0`` for an
  int domain) and ``in_family`` (a real bug in eregion's own family).

A guard that calls anything but a small set of pure builtins is ``undecided``: the
claims are still compared, but no value is known to take the arm.

Usage::

    python scripts/guard_domain_differential.py --eregion /path/to/[internal ref] \
        [--out tests/fixtures/guard_domain_differential.json]
"""

from __future__ import annotations

import argparse
import ast
import glob
import importlib.util
import itertools
import json
import math
import os
import signal
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tharkun.solve.guard_domain import domain_of  # noqa: E402

NAN = float("nan")
POOL = [
    -(10 ** 20), -1000, -256, -101, -100, -11, -10, -5, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 12, 15, 16, 20, 31, 32, 42, 50, 63, 64, 99, 100, 101, 127, 128, 255, 256, 1000,
    1023, 1024, 65535, 65536, 2 ** 31, 2 ** 32, 2 ** 53 + 1, 10 ** 20,
    -2.5, -1.0, -0.0, 0.0, 5e-324, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 9.5, 10.0, 100.0, 1e300,
    math.inf, -math.inf, NAN,
    False, True, None,
    "", "a", "b", "z", "A", "Z", "0", "1", "ab", "abc", "hello", "go", "stop", "x", "+", "-",
    "red", "GET", "POST", "0x", "{", "}", " ",
    b"", b"a", b"ab", b"\x00", b"\x01", b"\xff", b"PK", b"%PDF",
    [], [0], [1], [1, 2], [1, 2, 3], ["a"], (), (1,), (1, 2), (1, 2, 3), {1}, {1, 2},
    frozenset({"a"}), {"a": 1}, {},
]
OTHER_POOL = [0, 1, -1, 2, 5, 10, 100, 0.5, "a", "", None, True, False, [], [1, 2], (1, 2), b"a"]
SAFE_CALLS = {"len", "abs", "int", "float", "str", "bool", "ord", "chr", "min", "max", "range",
              "isinstance", "type", "sorted", "sum", "any", "all", "tuple", "list", "set",
              "round", "divmod", "hex", "bin", "repr"}
SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                 for n in SAFE_CALLS}
SAFE_BUILTINS.update({"True": True, "False": False, "None": None})

BIN = {"add": "+", "sub": "-", "mul": "*", "div": "/", "truediv": "/", "floordiv": "//",
       "mod": "%", "pow": "**", "bitand": "&", "bitor": "|", "bitxor": "^", "lshift": "<<",
       "rshift": ">>"}
CMP = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!=", "in": "in",
       "not_in": "not in", "is": "is", "is_not": "is not"}


class Unrenderable(Exception):
    pass


def render(e) -> str:
    """An IR expr dict back to Python source."""
    if not isinstance(e, dict):
        raise Unrenderable(repr(e))
    k = e.get("kind")
    if k == "var":
        return e["name"]
    if k == "const":
        return repr(e.get("value"))
    if k == "const_none":
        return "None"
    if k == "const_float":
        return repr(float(e["value"]))
    if k == "const_str":
        raw = e.get("name")
        return raw if isinstance(raw, str) else repr(e.get("value"))
    if k == "const_bytes":
        return repr(bytes.fromhex(e["name"]))
    if k == "cmp":
        return f"({render(e['left'])} {CMP[e['name']]} {render(e['right'])})"
    if k == "not":
        return f"(not {render(e['left'])})"
    if k in ("and", "or"):
        return f"({render(e['left'])} {k} {render(e['right'])})"
    if k in BIN:
        return f"({render(e['left'])} {BIN[k]} {render(e['right'])})"
    if k in ("tuple", "list"):
        items = [render(i) for i in e.get("items", [])]
        if k == "list":
            return "[" + ", ".join(items) + "]"
        return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
    if k == "call":
        return f"{e['name']}(" + ", ".join(render(i) for i in e.get("items", [])) + ")"
    if k == "call_attr":
        return f"{render(e['left'])}.{e['name']}(" + ", ".join(render(i) for i in e.get("items", [])) + ")"
    if k == "attr":
        return f"{render(e['left'])}.{e['name']}"
    if k == "index":
        return f"{e['name']}[{e['value']!r}]"
    if k == "slice":
        lo, hi, st = (render(i) if i.get("kind") != "const_none" else "" for i in e["items"])
        return f"{render(e['left'])}[{lo}:{hi}" + (f":{st}]" if st else "]")
    raise Unrenderable(k)


def ir_vars(e, out):
    if isinstance(e, dict):
        if e.get("kind") == "var":
            out.add(e["name"])
        for v in e.values():
            ir_vars(v, out)
    elif isinstance(e, list):
        for v in e:
            ir_vars(v, out)
    return out


def safe(tree: ast.AST) -> bool:
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            if not (isinstance(n.func, ast.Name) and n.func.id in SAFE_CALLS):
                return False
        if isinstance(n, (ast.Attribute, ast.Lambda, ast.NamedExpr, ast.Await, ast.Yield)):
            return False
    return True


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)


def evaluate(code, env):
    """``bool(code)`` under ``env``; ``None`` if it raises or runs too long."""
    try:
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        try:
            r = bool(eval(code, {"__builtins__": SAFE_BUILTINS}, env))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        return r
    except BaseException:           # noqa: BLE001 -- includes the timeout
        return None


def assignments(names, var, v, cap=400):
    others = [n for n in names if n != var]
    combos = itertools.product(OTHER_POOL, repeat=len(others)) if len(others) <= 2 else \
        itertools.islice(itertools.product(OTHER_POOL, repeat=len(others)), cap)
    for combo in itertools.islice(combos, cap):
        env = dict(zip(others, combo))
        env[var] = v
        yield env


# --------------------------------------------------------------------------- #
# eregion domain semantics over concrete values
# --------------------------------------------------------------------------- #

def _real_int(v):
    return type(v) is int


def _eq_any(v, vals):
    for x in vals or ():
        try:
            if v == x:
                return True
        except Exception:
            pass
    return False


def family(kind):
    """The type test eregion's domain kind implies (its enumeration family)."""
    if kind.startswith("int_"):
        return _real_int
    if kind.startswith("float_"):
        return lambda v: type(v) in (int, float)
    if kind in ("str_set", "str_neq"):
        return lambda v: type(v) is str
    if kind in ("bytes_set", "bytes_neq"):
        return lambda v: type(v) is bytes
    if kind == "lex_range":
        return lambda v: type(v) in (str, bytes)
    if kind == "bool_set":
        return lambda v: type(v) is bool
    if kind == "is_none":
        return lambda v: v is None
    if kind in ("container_length", "set_membership"):
        return lambda v: hasattr(v, "__len__")
    return lambda v: True


def ere_value_ok(d, v):
    """The value part of eregion's claim, ignoring the type family."""
    k = d["domain_kind"]
    lo, hi, lo_open, hi_open = d.get("lo"), d.get("hi"), d.get("lo_open"), d.get("hi_open")
    vals, neq = d.get("values"), d.get("neq")
    try:
        if k in ("top",):
            return True
        if k == "guard_only":
            return vals is None or _eq_any(v, vals)
        if k in ("int_set", "float_set", "str_set", "bytes_set", "bool_set"):
            return _eq_any(v, vals) and not _eq_any(v, neq)
        if k == "is_none":
            return v is None
        if k in ("int_neq", "str_neq", "bytes_neq"):
            return not _eq_any(v, neq)
        if k in ("int_range_closed", "int_range_low_open", "int_range_high_open", "float_range",
                 "lex_range"):
            if isinstance(v, float) and math.isnan(v):
                return False
            if lo is not None and (v < lo or (lo_open and v == lo)):
                return False
            if hi is not None and (v > hi or (hi_open and v == hi)):
                return False
            return not _eq_any(v, neq)
        if k == "container_length":
            n = len(v)
            return (lo is None or n >= lo) and (hi is None or n <= hi)
        if k == "set_membership":
            if not all(r in v for r in d.get("required") or ()):
                return False
            if any(f in v for f in d.get("forbidden") or ()):
                return False
            n = len(v)
            return (lo is None or n >= lo) and (hi is None or n <= hi)
    except Exception:
        return False
    return True


def ere_admits(d, v):
    return family(d["domain_kind"])(v) and ere_value_ok(d, v)


# --------------------------------------------------------------------------- #
# the corpus
# --------------------------------------------------------------------------- #

def guard_buckets(obj, out):
    """Every JSON list that directly holds guard assumes: one arm's bucket."""
    if isinstance(obj, dict):
        for v in obj.values():
            guard_buckets(v, out)
    elif isinstance(obj, list):
        g = [x for x in obj if isinstance(x, dict) and x.get("kind") == "assume"
             and x.get("assume_kind") == "guard"]
        if g:
            out.append(g)
        for v in obj:
            guard_buckets(v, out)
    return out


def source_test(tree, loc):
    ln, col = loc.get("lineno"), loc.get("col_offset")
    for n in ast.walk(tree):
        if getattr(n, "lineno", None) == ln and getattr(n, "col_offset", None) == col \
                and isinstance(getattr(n, "test", None), ast.expr):
            return n.test
    return None


def polarity(ir_code, src_test, names):
    """+1 when the source test agrees with the IR predicate on every decidable
    assignment, -1 when it is the negation, else 0."""
    src_code = compile(ast.Expression(src_test), "<src>", "eval")
    names = set(names) | {n.id for n in ast.walk(src_test) if isinstance(n, ast.Name)}
    names -= SAFE_CALLS
    agree = disagree = 0
    for var in sorted(names) or ["_"]:
        for v in POOL:
            for env in itertools.islice(assignments(names, var, v), 30):
                a, b = evaluate(ir_code, env), evaluate(src_code, env)
                if a is None or b is None:
                    continue
                if a == b:
                    agree += 1
                else:
                    disagree += 1
    if agree and not disagree:
        return 1
    if disagree and not agree:
        return -1
    return 0


def _vclass(v):
    """A short class for a concrete value, for the by-type breakdown."""
    if isinstance(v, float):
        if math.isnan(v):
            return "float-nan"
        return "float-integral" if math.isfinite(v) and v.is_integer() else "float"
    return type(v).__name__


def compare_case(reduce_domain, ir_exprs, my_test, var, decide_code, names):
    ere = reduce_domain(ir_exprs, var)
    mine = domain_of(my_test, var)
    fam = family(ere["domain_kind"])
    res = Counter()
    wider_samples, unsound_samples = [], []
    for v in POOL:
        e_in, m_in = ere_admits(ere, v), mine.admits(v)
        takes = None
        if decide_code is not None:
            takes = False
            for env in assignments(names, var, v):
                r = evaluate(decide_code, env)
                if r:
                    takes = True
                    break
        if takes and not m_in:
            res["tharkun_unsound"] += 1
            unsound_samples.append(repr(v))
        if takes and not e_in:
            res["eregion_unsound_in_family" if fam(v) else "eregion_unsound_by_type"] += 1
            if not fam(v):
                res[f"bytype:{ere['domain_kind']}:{_vclass(v)}"] += 1
        if e_in and not m_in:
            res["narrower" if takes else ("tighter" if takes is False else "narrower_undecided")] += 1
        if m_in and not e_in:
            if takes:
                why = "eregion_unsound"
            elif not fam(v):
                why = "other_type"
            elif takes is None:
                why = "undecided"
            else:
                why = "precision"
            res["wider_" + why] += 1
            if why in ("precision", "undecided") and len(wider_samples) < 6:
                wider_samples.append(repr(v))
        if takes and not e_in and fam(v) and len(unsound_samples) < 6:
            unsound_samples.append("eregion drops " + repr(v))
    return ere, mine, res, wider_samples, unsound_samples


# --------------------------------------------------------------------------- #
# random guards: the unit-test grammar, converted to IR dicts for eregion
# --------------------------------------------------------------------------- #

_IR_CMP = {v: k for k, v in CMP.items()}


def to_ir(node):
    """A Python guard AST as an eregion IR expr dict (the subset the grammar uses)."""
    if isinstance(node, ast.Name):
        return {"kind": "var", "name": node.id}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return {"kind": "not", "left": to_ir(node.operand)}
    if isinstance(node, ast.BoolOp):
        k = "and" if isinstance(node.op, ast.And) else "or"
        out = to_ir(node.values[0])
        for v in node.values[1:]:
            out = {"kind": k, "left": out, "right": to_ir(v)}
        return out
    if isinstance(node, ast.Compare):
        ops = [node.left, *node.comparators]
        out = None
        for i, o in enumerate(node.ops):
            sym = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==",
                   ast.NotEq: "!=", ast.In: "in", ast.NotIn: "not in", ast.Is: "is",
                   ast.IsNot: "is not"}[type(o)]
            if sym == "is not":
                c = {"kind": "not", "left": {"kind": "cmp", "name": "is", "left": to_ir(ops[i]),
                                              "right": to_ir(ops[i + 1])}}
            else:
                c = {"kind": "cmp", "name": _IR_CMP[sym], "left": to_ir(ops[i]),
                     "right": to_ir(ops[i + 1])}
            out = c if out is None else {"kind": "and", "left": out, "right": c}
        return out
    if isinstance(node, (ast.Tuple, ast.List)):
        return {"kind": "tuple" if isinstance(node, ast.Tuple) else "list",
                "items": [to_ir(e) for e in node.elts]}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
            and isinstance(node.operand, ast.Constant):
        return to_ir(ast.Constant(-node.operand.value))
    if isinstance(node, ast.Constant):
        v = node.value
        if v is None:
            return {"kind": "const_none"}
        if isinstance(v, str):
            return {"kind": "const_str", "name": repr(v)}
        if isinstance(v, bytes):
            return {"kind": "const_bytes", "name": v.hex()}
        if isinstance(v, float):
            return {"kind": "const_float", "value": v}
        return {"kind": "const", "value": v}
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        return {"kind": "call", "name": node.func.id, "items": [to_ir(a) for a in node.args]}
    raise Unrenderable(type(node).__name__)


def random_report(reduce_domain, n):
    """eregion's reduce_domain checked for soundness on ``n`` random guards."""
    sys.path.insert(0, os.path.join(HERE, "..", "tests"))
    import random as _random

    from test_solve_guard_domain import _atom, _guard  # the unit-test grammar
    rng = _random.Random(1021)
    tot = Counter()
    examples = {}
    for _ in range(n):
        src = _atom(rng, "x") if rng.random() < 0.5 else _guard(rng)
        tree = ast.parse(src, mode="eval").body
        try:
            ir = to_ir(tree)
        except Unrenderable:
            tot["unconvertible"] += 1
            continue
        tot["guards"] += 1
        code = compile(src, "<g>", "eval")
        names = {x.id for x in ast.walk(tree) if isinstance(x, ast.Name)} - SAFE_CALLS
        for negate in (False, True):
            pred = {"kind": "not", "left": ir} if negate else ir
            for var in sorted(names):
                tot["cases"] += 1
                try:
                    ere = reduce_domain([pred], var)
                except Exception as exc:   # reduce_domain documents "never raises"
                    tot["reduce_domain_raised"] += 1
                    raised = examples.setdefault("RAISED", [])
                    if len(raised) < 4:
                        raised.append({"guard": ("not " if negate else "") + f"({src})",
                                       "var": var, "error": repr(exc)})
                    continue
                fam = family(ere["domain_kind"])
                bad_in, bad_type = [], 0
                for v in POOL:
                    takes = False
                    for env in assignments(names, var, v, cap=40):
                        r = evaluate(code, env)
                        if r is not None and r != negate:
                            takes = True
                            break
                    if takes and not ere_admits(ere, v):
                        if fam(v):
                            bad_in.append(repr(v))
                        else:
                            bad_type += 1
                if bad_type:
                    tot["unsound_by_type_cases"] += 1
                if bad_in:
                    tot["unsound_in_family_cases"] += 1
                    key = ere["domain_kind"]
                    examples.setdefault(key, [])
                    if len(examples[key]) < 4:
                        examples[key].append({"guard": ("not " if negate else "") + f"({src})",
                                              "var": var,
                                              "domain": {k: v for k, v in ere.items()
                                                         if k != "guards" and v is not None},
                                              "dropped": bad_in[:6]})
    out = dict(sorted(tot.items()))
    out["in_family_examples"] = examples
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eregion", required=True, help="the py-inv umbrella checkout")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--random", type=int, default=0,
                    help="also check reduce_domain on N random guards (the unit-test grammar)")
    args = ap.parse_args(argv)
    repos = os.path.join(args.eregion, "repos")
    sys.path.insert(0, os.path.join(repos, "eregion-commons"))
    spec = importlib.util.spec_from_file_location(
        "_ere_domain_capture",
        os.path.join(repos, "eregion-ir/src/eregion_ir/program_extract/domain_capture.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    reduce_domain = mod.reduce_domain

    packs = sorted(glob.glob(os.path.join(repos, "eregion-toys/toys/packs/*/fixtures/02_ir.json")))
    if args.limit:
        packs = packs[:args.limit]
    totals = Counter()
    kinds = Counter()
    cases_out, ere_bugs, wider_precision = [], [], []
    mismatch = {}
    bytype = Counter()
    for path in packs:
        toy = path.split(os.sep)[-3]
        src_path = os.path.join(os.path.dirname(os.path.dirname(path)), "source.py")
        try:
            tree = ast.parse(open(src_path).read())
        except (OSError, SyntaxError):
            tree = None
        buckets = guard_buckets(json.load(open(path)), [])
        seen = set()
        for bucket in buckets:
            singles = [[a] for a in bucket]
            units = singles + ([bucket] if len(bucket) > 1 else [])
            for unit in units:
                key = json.dumps([a["expr"] for a in unit], sort_keys=True) + json.dumps(
                    [a.get("loc") for a in unit], sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                totals["units_multi" if len(unit) > 1 else "units_single"] += 1
                try:
                    codes = [render(a["expr"]) for a in unit]
                except Unrenderable as exc:
                    totals["unrenderable"] += 1
                    continue
                names = set()
                for a in unit:
                    ir_vars(a["expr"], names)
                my_parts, how = [], "source"
                for a, code in zip(unit, codes):
                    ir_code = compile(code, "<ir>", "eval")
                    st = source_test(tree, a.get("loc") or {}) if tree is not None else None
                    pol = polarity(ir_code, st, ir_vars(a["expr"], set())) if st is not None else 0
                    if pol == 1:
                        my_parts.append(st)
                    elif pol == -1:
                        my_parts.append(ast.UnaryOp(op=ast.Not(), operand=st))
                    else:
                        my_parts.append(ast.parse(code, mode="eval").body)
                        how = "rendered"
                totals["from_" + how] += 1
                my_test = my_parts[0] if len(my_parts) == 1 else ast.BoolOp(op=ast.And(), values=my_parts)
                ir_whole = " and ".join(codes)
                decidable = safe(ast.parse(ir_whole, mode="eval"))
                decide_code = compile(ir_whole, "<ir>", "eval") if decidable else None
                totals["decidable" if decidable else "undecided"] += 1
                for var in sorted(names):
                    ere, mine, res, wsamp, usamp = compare_case(
                        reduce_domain, [a["expr"] for a in unit], my_test, var, decide_code, names)
                    totals["cases"] += 1
                    kinds[(ere["domain_kind"], mine.kind)] += 1
                    for k, n in res.items():
                        if k.startswith("bytype:"):
                            bytype[k[7:]] += 1
                            continue
                        totals[k + "_values"] += n
                        totals[k + "_cases"] += 1
                    rec = {"toy": toy, "loc": unit[0].get("loc"), "pred": ir_whole,
                           "tharkun_test": ast.unparse(my_test), "var": var,
                           "eregion_kind": ere["domain_kind"], "tharkun_kind": mine.kind,
                           "counts": dict(res)}
                    if res.get("tharkun_unsound") or res.get("narrower"):
                        rec["samples"] = usamp
                        cases_out.append(rec)
                    if res.get("eregion_unsound_in_family"):
                        ere_bugs.append(dict(rec, samples=[s for s in usamp if s.startswith("eregion")],
                                             eregion_domain={k: v for k, v in ere.items()
                                                             if k != "guards" and v is not None}))
                    if ere["domain_kind"] != mine.kind:
                        mismatch.setdefault(f'{ere["domain_kind"]} -> {mine.kind}', []).append(
                            f"{toy}: {var}: {ast.unparse(my_test)}")
                    if res.get("wider_precision") or res.get("wider_undecided"):
                        wider_precision.append(dict(rec, samples=wsamp))
    agree = sum(n for (a, b), n in kinds.items() if a == b)
    report = {
        "corpus": "eregion-toys/toys/packs/*/fixtures/02_ir.json (guard assumes) + source.py",
        "pool_size": len(POOL),
        "totals": dict(sorted(totals.items())),
        "label_agreement": {"same": agree, "cases": totals["cases"]},
        "label_pairs": {f"{a} -> {b}": n for (a, b), n in kinds.most_common()},
        "label_mismatch_examples": {k: v[:5] for k, v in sorted(mismatch.items())},
        "eregion_unsound_by_type_breakdown": {
            "doc": "eregion domain kind : class of a value that takes the arm but is outside the "
                   "kind's type family -> number of cases",
            "counts": dict(bytype.most_common())},
        "random_guards": random_report(reduce_domain, args.random) if args.random else None,
        "tharkun_unsound_or_narrower": cases_out,
        "eregion_unsound_in_family": ere_bugs,
        "wider_precision_or_undecided": wider_precision,
    }
    text = json.dumps(report, indent=1, sort_keys=False, default=repr)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(json.dumps({"totals": report["totals"], "label_agreement": report["label_agreement"],
                      "tharkun_unsound_or_narrower": len(cases_out),
                      "eregion_unsound_in_family": len(ere_bugs),
                      "wider_precision_or_undecided": len(wider_precision),
                      "random_guards": report["random_guards"] and {
                          k: v for k, v in report["random_guards"].items() if k != "in_family_examples"}},
                     indent=1))
    return 1 if cases_out else 0


if __name__ == "__main__":
    sys.exit(main())
