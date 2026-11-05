"""Constraint dicts to Hypothesis strategies — narrow constructively, do not filter.

The driver's search tiers built strategies from a type name alone, so everything the
static solver had already worked out — the bounds a comparison implies, the literal set a
membership test names, the mask a bitwise guard demands — was thrown away and rediscovered
by sampling. A guard that says ``x > 1000 and x < 1010`` deserves
``st.integers(1001, 1009)``, not ``st.integers()`` filtered nine million ways.

The governing rule throughout is **narrow, do not filter**. A filtered strategy still
draws from the wide domain and throws work away; a narrowed one only ever draws answers.
Where filtering is unavoidable it is used only when the rejection rate is provably low,
and the reason is stated at the site. Two cases are worth reading for the pattern:

* a **contiguous bitmask** (``0b111…``) needs no filter at all, because every value in
  ``[0, mask]`` already satisfies ``x & ~mask == 0``;
* a **left shift** is enumerated over the quotient and multiplied back, so every draw is
  a multiple of the step — instead of drawing integers and rejecting the ones that are not.

The schema is deliberately a plain dict rather than a class: it is what a static analysis
naturally produces, it serializes, and it recurses for structured values.

    {
      "type": "int" | "float" | "str" | "bytes" | "list" | "tuple" | "dict"
            | "bool" | "none" | "complex" | "decimal" | "fraction" | "bytearray"
            | "set" | "frozenset" | "one_of" | "literal" | "namespace" | "callable",

      # any type ([internal ref]: one schema for tharkun.inputs, Avengers' object leaves and the str/re regex members)
      "values": [v1, v2, ...],        # membership; dominates every other constraint
      "prefer": [v1, ...],            # also these values: FIRST in examples(), and a strategy's first option

      # int
      "bitmask_mask": <int>,          # only bits set in mask may be set
      "shift_direction": "left" | "right",
      "shift_bits": <int>,

      # int / float
      "value_range": [min, max],      # inclusive; None on a side means the default bound
      "unbounded": <bool>,            # no default range: every int, every float (-0.0, subnormals)

      # float / decimal
      "allow_nan": <bool>, "allow_infinity": <bool>,          # default False
      "allow_snan": <bool>,           # decimal; default True (a set element / dict key cannot be one)

      # complex
      "part_constraints": {...},      # the float dict of both parts

      # str
      "alphabet": "printable" | "ascii" | "<chars>",          # narrows (a regex too), never filters
      "contains": "<sub1>" | ["<sub1>", ...],                 # substrings every str must hold (constructive)
      "regex": "<pattern>",           # sample the language rather than filter for it
      "flags": <int> | [names],       # re flags; re.DEBUG masked, re.LOCALE refused
      "how": "fullmatch" | "match" | "search",                # default fullmatch ("fullmatch": False = search)
      "prefer_chars": "<chars>",      # regex: leading members over these characters
      "segments": {"split": "@", "lengths": [64, 255]},       # an int, {"min","max"} or None per stretch
      "flag_difference": <int> | [names],                     # regex: members where the two flag sets differ

      # str / bytes / bytearray / list / tuple / set / frozenset / dict
      "length": {"min": a, "max": b}, # with a regex: the TOTAL length, by rewriting the pattern

      # list / tuple / set / frozenset
      "element_constraints": {...},   # recursive
      # list
      "contains": [e1, e2, ...],      # elements every list must hold (interleaved constructively)
      # tuple
      "items": [{...}, ...],          # one dict per position (a FIXED arity)

      # dict
      "key_constraints": {...},
      "value_constraints": {...},

      # one_of / literal
      "options": [{...}, ...],        # the union (Optional[X] is [X, {"type": "none"}])
      "values": [...],                # literal: any values, sampled

      # namespace (duck typing) / callable
      "attributes": {name: {...}}, "class": "types.SimpleNamespace" | "argparse.Namespace",
      "returns": {...}, "allow": [builtin names],             # a contract-matched builtin, else ConstantFunction
    }

A regex is parsed (``re._parser``), rewritten (:func:`_render`; exact repeat counts and branch choices for
``length`` / ``segments``, one variant per flag-reading node for ``flag_difference``) and handed to Hypothesis's
``from_regex``: the regex fields narrow by construction.  Rows are drawn with :func:`examples`, the ONE
deterministic driver (derandomized, no database; the leading values, then the simplest value, then the drawn
ones).

Two functions narrow a finished dict without building a strategy ([internal ref]): :func:`small` cuts the open sides of a
domain (ints, floats, lengths) for a search that replays every draw, and :func:`truthy` keeps the members whose ``bool``
is True, so ``examples(truthy(c), 1)`` is the simplest truthy value of ``c`` and ``examples(c, 1)`` the simplest value.
:func:`contains` tests whether a Python value is a member of a constraint dict without building a strategy.

:func:`domain` is the SYNTHESIS-side door of the same subsystem ([internal ref], docs/INPUTS.md "Baked domains"): a
``find_inputs`` kind, the guards that must hold, contract facts and the program's literals in; a narrowed kind, its
constraint dict, deterministic example values and a ``tharkun-strategies:<rule>`` tag out.  It never runs the target.

Any contradiction, empty domain, or unusable field raises
:class:`UnsatisfiableConstraints` rather than returning a strategy that cannot produce
anything — an empty domain is a fact worth surfacing, not a search to run.

Lifted from the eregion codegen project's ``hypothesis_enum`` and reimplemented here;
tharkun does not depend on eregion.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
import itertools
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

# A guard's literals are usually small; these bound the search when nothing says otherwise.
INT_DEFAULT_MIN = -(2 ** 31)
INT_DEFAULT_MAX = 2 ** 31 - 1
FLOAT_DEFAULT_MIN = float(-(2 ** 31))
FLOAT_DEFAULT_MAX = float(2 ** 31 - 1)

_VALID_TYPES = frozenset({"int", "float", "str", "bytes", "list", "tuple", "dict",
                          "bool", "none", "complex", "decimal", "fraction", "bytearray", "set", "frozenset",
                          "one_of", "literal", "namespace", "callable"})
#: the types whose builder reads ``values`` itself (the rest sample ``values`` as a literal)
_TYPED_VALUES = frozenset({"int", "float", "str", "bytes"})


class UnsatisfiableConstraints(ValueError):
    """The constraints describe an empty domain, or one that cannot be sampled."""


# Static/runtime inference spells a handful of types differently than make_strategy's
# vocabulary; fold the common aliases so a prior's type actually selects a strategy.
_TYPE_ALIASES = {
    "int": "int", "integer": "int", "bool": "int",
    "float": "float", "complex": "float",
    "str": "str", "string": "str", "unicode": "str",
    "bytes": "bytes", "bytearray": "bytes",
    "list": "list", "List": "list",
    "tuple": "tuple", "Tuple": "tuple",
    "dict": "dict", "Dict": "dict", "mapping": "dict",
}


def constraints_from_state(state: Any, subject: str) -> dict | None:
    """Translate a lattice ``FactState``'s knowledge of *subject* into a make_strategy dict.

    The bridge that lets the BEFORE warmup and the observed lattice actually AIM the search:
    a prior that typed a parameter ``str`` should hand the search a text strategy, even when
    the seed happens to hold ``b''`` and the value-shape guess would say bytes.

    Duck-typed on purpose — reads ``state.vars[subject]`` and its ``types`` / ``values`` /
    ``interval`` components by attribute, so this module needs no import of ``lattice`` (which
    would import canonicalize, guard_normalize, …). Returns ``None``, never raises, whenever
    the state says nothing usable — an unknown subject, an ambiguous or unmappable type — so
    the caller cleanly falls back to its value-shape strategy.
    """
    vs = getattr(state, "vars", {}).get(subject) if state is not None else None
    if vs is None:
        return None

    # A single, mappable type is required to pick ONE strategy. Several types (a union the
    # analysis could not resolve) or an unmappable one (a custom class) → fall back.
    type_names = getattr(getattr(vs, "types", None), "types", None)
    if not type_names or len(type_names) != 1:
        return None
    mapped = _TYPE_ALIASES.get(next(iter(type_names)))
    if mapped is None:
        return None

    constraints: dict = {"type": mapped}

    # Membership is the sharpest fact; make_strategy honours "values" for int. For other
    # types the value set is advisory only and better delivered as the search alphabet, so
    # it is not forced into the strategy here.
    values = getattr(getattr(vs, "values", None), "values", None)
    if mapped == "int" and values:
        ints = [v for v in values if _is_int(v)]
        if ints:
            constraints["values"] = sorted(ints)
            return constraints

    # A numeric interval narrows the range. Excluded points are dropped: make_strategy has
    # no exclusion field and filtering them is not worth it for a search alphabet.
    interval = getattr(vs, "interval", None)
    if mapped in ("int", "float") and interval is not None and not getattr(interval, "empty", False):
        lo, hi = getattr(interval, "lo", None), getattr(interval, "hi", None)
        if lo is not None or hi is not None:
            constraints["value_range"] = [lo, hi]

    return constraints


def strategy_from_state(state: Any, subject: str) -> Any | None:
    """A Hypothesis strategy for *subject* built from a lattice state, or ``None``.

    ``None`` on anything unusable — no facts, an unmappable type, or an empty domain the
    constraints describe — so a caller can always fall back to a value-shape strategy.
    """
    constraints = constraints_from_state(state, subject)
    if constraints is None:
        return None
    try:
        return make_strategy(constraints)
    except UnsatisfiableConstraints:
        return None


def strategy_for_type(
    type_name: str,
    *,
    int_range: tuple[int, int] = (INT_DEFAULT_MIN, INT_DEFAULT_MAX),
    max_size: int = 50,
) -> Any | None:
    """Build the shared broad strategy for an inferred primitive type.

    Branch search and Avengers previously maintained separate type tables.
    This is the fallback before either caller adds facts with
    :func:`make_strategy`; it deliberately preserves the historical
    ``hypothesis_adapter`` domains.
    """

    try:
        import hypothesis.strategies as st
    except ImportError:  # pragma: no cover - optional dependency
        return None

    if not isinstance(type_name, str):
        return None
    # A bare name collapses to the plain name the kind table gives it (docs/TYPEINFER.md "Kinds"): ``List``,
    # ``Dict``, ``Mapping``, ``Sequence``, ``decimal.Decimal``, ``FrozenSet``, ``NoneType``.  A name the table does
    # not know (``Any`` and a class are not a kind of it), and anything that is not a bare name (``List[int]``,
    # ``Optional[int]``, a quoted name), stays as written.
    from tharkun.typeinfer.core.kinds import annotation_kind, type_name as plain_name

    normalized = type_name.removeprefix("typing.")
    if re.fullmatch(r"[A-Za-z_][\w.]*", type_name) and (spelled := annotation_kind(type_name))["kind"] != "any":
        normalized = plain_name(spelled).removeprefix("typing.")
    lo, hi = int_range
    if normalized in {"int", "integer"}:
        return st.integers(min_value=lo, max_value=hi)
    if normalized == "float":
        return st.floats(min_value=float(lo), max_value=float(hi), allow_nan=False)
    if normalized in {"str", "string", "unicode"}:
        return st.text(min_size=0, max_size=max_size)
    if normalized in {"bytes", "bytearray"}:
        return st.binary(min_size=0, max_size=max_size)
    if normalized == "bool":
        return st.booleans()
    if normalized == "list":
        return st.lists(st.integers(), max_size=10)
    if normalized in {"dict", "mapping"}:
        return st.dictionaries(st.text(max_size=5), st.integers(), max_size=5)
    if normalized == "set":
        return st.frozensets(st.integers(), max_size=10).map(set)
    if normalized == "tuple":
        return st.tuples(st.integers(), st.integers())
    if normalized == "object":
        return st.fixed_dictionaries({})
    if normalized in {"Any", "any"}:
        return st.one_of(st.integers(), st.text(max_size=20), st.booleans())
    # [internal ref]: the rest of the builtin value types, through the constraint table (names that returned None before)
    rest = {"complex": "complex", "Decimal": "decimal", "Fraction": "fraction", "frozenset": "frozenset",
            "None": "none"}.get(normalized)
    if rest is not None:
        return make_strategy({"type": rest, "length": {"min": 0, "max": 10}} if rest == "frozenset" else {"type": rest})
    return None


def _is_int(v: Any) -> bool:
    """True for a real int. ``bool`` is an ``int`` subclass and is never one here."""
    return isinstance(v, int) and not isinstance(v, bool)


def make_strategy(constraints: dict) -> Any:
    """A Hypothesis strategy for *constraints*, as narrow as the facts allow."""
    try:
        import hypothesis.strategies as st
    except ImportError as exc:                        # pragma: no cover - env dependent
        raise UnsatisfiableConstraints("hypothesis is not installed") from exc

    if not isinstance(constraints, dict):
        raise UnsatisfiableConstraints(f"constraints must be a dict, got {type(constraints).__name__}")
    kind = constraints.get("type")
    if kind not in _VALID_TYPES:
        raise UnsatisfiableConstraints(f"unknown or missing type {kind!r}")

    if "values" in constraints and kind not in _TYPED_VALUES:
        # membership dominates ([internal ref]: any type -- a class's constants, a recipe's literals)
        base = _literal_strategy(constraints)
    else:
        if "contains" in constraints and kind not in ("str", "list"):
            raise UnsatisfiableConstraints(f"'contains' is only valid on 'str' and 'list' domains, not {kind!r}")
        base = _BUILDERS[kind](constraints)
    prefer = constraints.get("prefer")
    if prefer is None:
        return base
    if not isinstance(prefer, (list, tuple)):
        raise UnsatisfiableConstraints(f"'prefer' must be a sequence of values, got {type(prefer).__name__}")
    # the caller's own values (a program's literals) first: Hypothesis draws the first option of a one_of most
    # often and shrinks toward it, so a derandomized run tries them early; the type's strategy keeps the rest
    return st.one_of(st.sampled_from(list(prefer)), base) if prefer else base


def _length_bounds(constraints: dict, default_max: int) -> tuple[int, int]:
    """``(min, max)`` size, validated. Absent means ``[0, default_max]``."""
    length = constraints.get("length")
    if length is None:
        return 0, default_max
    if not isinstance(length, dict) or "min" not in length or "max" not in length:
        raise UnsatisfiableConstraints("'length' must be a dict with 'min' and 'max'")
    lo, hi = length["min"], length["max"]
    if not _is_int(lo) or lo < 0:
        raise UnsatisfiableConstraints(f"length.min must be a non-negative int, got {lo!r}")
    if not _is_int(hi) or hi < 0:
        raise UnsatisfiableConstraints(f"length.max must be a non-negative int, got {hi!r}")
    if lo > hi:
        raise UnsatisfiableConstraints(f"length.min ({lo}) > length.max ({hi}); empty domain")
    return lo, hi


def _range_bounds(constraints: dict, lo_default: Any, hi_default: Any, numeric) -> tuple[Any, Any]:
    rng = constraints.get("value_range")
    if rng is None:
        return lo_default, hi_default
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        raise UnsatisfiableConstraints(f"'value_range' must be [min, max], got {rng!r}")
    lo, hi = rng
    # an open side is unbounded: the default must not cross the given side (a lower bound above the
    # default max is not an empty domain)
    lo = (min(lo_default, hi) if numeric(hi) else lo_default) if lo is None else lo
    hi = (max(hi_default, lo) if numeric(lo) else hi_default) if hi is None else hi
    if not numeric(lo) or not numeric(hi):
        raise UnsatisfiableConstraints(f"value_range bounds must be numeric, got {rng!r}")
    if lo > hi:
        raise UnsatisfiableConstraints(f"value_range min ({lo}) > max ({hi}); empty domain")
    return lo, hi


def _int_strategy(constraints: dict) -> Any:
    import hypothesis.strategies as st

    # Membership dominates: a named set of values is the sharpest fact there is, and
    # nothing else can narrow it further.
    if "values" in constraints:
        values = constraints["values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise UnsatisfiableConstraints("int 'values' must be a non-empty sequence")
        for v in values:
            if not _is_int(v):
                raise UnsatisfiableConstraints(
                    f"int 'values' contains a non-int element {v!r} ({type(v).__name__})")
        return st.sampled_from(list(values))

    lo, hi = _range_bounds(constraints, INT_DEFAULT_MIN, INT_DEFAULT_MAX, _is_int)

    if "bitmask_mask" in constraints:
        mask = constraints["bitmask_mask"]
        if not _is_int(mask) or mask < 0:
            raise UnsatisfiableConstraints(f"'bitmask_mask' must be a non-negative int, got {mask!r}")
        # `x & ~mask == 0` says only the mask's bits may be set, so the domain is at
        # most [0, mask]. Narrowing the range is what makes this cheap; the filter below
        # then only has to reject the sparse leftovers.
        lo, hi = max(lo, 0), min(hi, mask)
        if lo > hi:
            raise UnsatisfiableConstraints(
                f"bitmask_mask={mask} leaves an empty range given the other constraints")
        base = st.integers(min_value=lo, max_value=hi)
        if mask == 0 or (mask & (mask + 1)) == 0:
            # A contiguous mask (0b1, 0b11, 0b111 …) is satisfied by EVERY value in
            # [0, mask], so the filter would reject nothing and cost everything.
            return base
        return base.filter(lambda x, m=mask: (x & ~m) == 0)

    if "shift_direction" in constraints or "shift_bits" in constraints:
        direction = constraints.get("shift_direction", "left")
        bits = constraints.get("shift_bits", 0)
        if not _is_int(bits) or bits < 0:
            raise UnsatisfiableConstraints(f"'shift_bits' must be a non-negative int, got {bits!r}")
        if direction == "left":
            # `value = q << bits` means value is a multiple of 2**bits. Enumerate the
            # QUOTIENT and multiply back, so every draw is valid by construction —
            # rather than drawing integers and rejecting the ones that are not multiples.
            step = 1 << bits
            lo_q = -(-lo // step) * step
            hi_q = (hi // step) * step
            if lo_q > hi_q:
                raise UnsatisfiableConstraints(
                    f"left shift by {bits} leaves no multiple of {step} in [{lo}, {hi}]")
            return st.integers(min_value=lo_q // step, max_value=hi_q // step).map(
                lambda q, s=step: q * s)
        # A right shift discards the low bits, so nothing about the source is recoverable.
        return st.integers(min_value=lo, max_value=hi)

    if constraints.get("unbounded") and "value_range" not in constraints:
        return st.integers()
    # a None side of an explicit range stays open (Hypothesis takes None): no 32-bit cap on a one-sided bound
    open_lo, open_hi = (constraints.get("value_range") or (lo, hi))
    return st.integers(min_value=None if open_lo is None else lo, max_value=None if open_hi is None else hi)


def _float_strategy(constraints: dict) -> Any:
    import hypothesis.strategies as st

    def _num(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    if "values" in constraints:
        values = constraints["values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise UnsatisfiableConstraints("float 'values' must be a non-empty sequence")
        return st.sampled_from([float(v) for v in values if _num(v)] or
                               _raise("float 'values' contains no numeric element"))

    allow_nan, allow_inf = bool(constraints.get("allow_nan")), bool(constraints.get("allow_infinity"))
    if constraints.get("unbounded") and "value_range" not in constraints:
        # every float ([internal ref]: a parameter's whole domain, -0.0 and the subnormals included), the special values
        # only when asked for
        return st.floats(allow_nan=allow_nan, allow_infinity=allow_inf)
    lo, hi = _range_bounds(constraints, FLOAT_DEFAULT_MIN, FLOAT_DEFAULT_MAX, _num)
    # Bound an open side rather than leave it open: an unbounded float strategy spends
    # most of its draws in magnitudes no guard cares about. NaN and infinity are excluded
    # because they satisfy no ordered comparison and so are never a useful answer to one.
    return st.floats(min_value=float(lo), max_value=float(hi),
                     allow_nan=False, allow_infinity=False)


def _raise(msg: str):
    raise UnsatisfiableConstraints(msg)


def _str_contains(constraints: dict) -> list[str]:
    """Required substrings on a ``str`` constraint dict (a ``str``, or a sequence of ``str``), normalized:
    ``""`` and substrings already inside another required substring dropped, order preserved."""
    raw = constraints.get("contains")
    if raw is None:
        return []
    items = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, (list, tuple)) else None
    if items is None or not all(isinstance(s, str) for s in items):
        raise UnsatisfiableConstraints(f"str 'contains' must be a str or sequence of strs, got {raw!r}")
    uniq = list(dict.fromkeys(s for s in items if s))
    return [s for i, s in enumerate(uniq) if not any(s in t for j, t in enumerate(uniq) if i != j)]


def _merge_overlap(a: str, b: str) -> str:
    if b in a:
        return a
    for k in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + b


def _shortest_superstrings(subs: list[str], max_len: int | None = None) -> list[str]:
    """Merged superstrings of ``subs`` (exact permutations up to 6 substrings, greedy max-overlap above),
    filtered to ``<= max_len`` when given."""
    if not subs:
        return [""]
    if len(subs) == 1:
        return [subs[0]] if max_len is None or len(subs[0]) <= max_len else []
    if len(subs) <= 6:
        seen: list[str] = []
        for perm in itertools.permutations(subs):
            acc = perm[0]
            for s in perm[1:]:
                acc = _merge_overlap(acc, s)
                if max_len is not None and len(acc) > max_len:
                    break
            else:
                if acc not in seen:
                    seen.append(acc)
        return seen
    work = list(subs)
    while len(work) > 1:
        best = (-1, 0, 1, work[0] + work[1])
        for i, a in enumerate(work):
            for j, b in enumerate(work):
                if i != j:
                    merged = _merge_overlap(a, b)
                    saved = len(a) + len(b) - len(merged)
                    if saved > best[0]:
                        best = (saved, i, j, merged)
        _, i, j, merged = best
        work = [w for k, w in enumerate(work) if k not in (i, j)] + [merged]
    return [work[0]] if max_len is None or len(work[0]) <= max_len else []


def _alphabet_allows(name: Any, text: str) -> bool:
    """Does every character of ``text`` belong to ``alphabet`` (validated)?"""
    import string

    if name is None:
        return True
    if name == "printable":
        return set(text) <= set(string.printable)
    if name == "ascii":
        return text.isascii()
    if not isinstance(name, str):
        raise UnsatisfiableConstraints(f"'alphabet' is 'printable', 'ascii' or a str of characters, got {name!r}")
    return set(text) <= set(name)


def _str_strategy(constraints: dict) -> Any:
    import hypothesis.strategies as st

    if "values" in constraints:
        values = constraints["values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise UnsatisfiableConstraints("str 'values' must be a non-empty sequence")
        return st.sampled_from([v for v in values if isinstance(v, str)] or
                               _raise("str 'values' contains no string"))

    subs = _str_contains(constraints)
    alphabet = constraints.get("alphabet")
    for sub in subs:
        if not _alphabet_allows(alphabet, sub):
            raise UnsatisfiableConstraints(f"str 'contains' {sub!r} has a character outside 'alphabet'")

    # A regex guard is the case where filtering is hopeless: no amount of sampling
    # st.text() will produce "32hex:32hex". Sample the pattern's LANGUAGE instead.
    # fullmatch defaults True because a captured pattern is normally ^…$-anchored;
    # a caller matching only a prefix passes fullmatch=False.
    if constraints.get("regex") is not None:
        if subs and "length" in constraints:
            _, hi = _length_bounds(constraints, default_max=64)
            if not _shortest_superstrings(subs, hi):
                raise UnsatisfiableConstraints(f"str 'contains' cannot fit {subs!r} within length.max ({hi})")
        out = _regex_strategy(constraints)
        if subs:
            # Regex generation already restricts draws to the pattern's language; filter only for any extra substrings.
            out = out.filter(lambda v, req=tuple(subs): all(s in v for s in req))
        return out
    for field in ("flags", "how", "flag_difference", "prefer_chars"):
        if field in constraints:
            raise UnsatisfiableConstraints(f"{field!r} needs a 'regex'")

    if "segments" in constraints:
        out = _segments_strategy(constraints, alphabet)
        return out.filter(lambda v, req=tuple(subs): all(s in v for s in req)) if subs else out
    sum_len = sum(len(s) for s in subs)
    lo, hi = _length_bounds(constraints, default_max=max(64, sum_len))
    chars = _alphabet(alphabet)
    if not subs:
        if chars is not None:
            if chars == "" and lo > 0:
                raise UnsatisfiableConstraints("an empty 'alphabet' has no text of length >= 1")
            return st.text(alphabet=chars, min_size=lo, max_size=hi)
        return st.text(min_size=lo, max_size=hi)

    def filler_text(mn: int, mx: int) -> Any:
        return st.text(alphabet=chars, min_size=mn, max_size=mx) if chars is not None else st.text(min_size=mn, max_size=mx)

    if sum_len <= hi:
        if chars == "" and lo > sum_len:
            raise UnsatisfiableConstraints("an empty 'alphabet' cannot pad 'contains' to length.min")

        @st.composite
        def with_substrings(draw):
            order = list(draw(st.permutations(subs))) if len(subs) > 1 else subs
            extra_hi = 0 if chars == "" else hi - sum_len
            filler = draw(filler_text(max(0, lo - sum_len), extra_hi)) if extra_hi > 0 else ""
            if not filler:
                return "".join(order)
            cuts = sorted(draw(st.integers(min_value=0, max_value=len(filler))) for _ in order)
            parts, prev = [], 0
            for cut, sub in zip(cuts, order):
                parts.append(filler[prev:cut])
                parts.append(sub)
                prev = cut
            parts.append(filler[prev:])
            return "".join(parts)

        return with_substrings()

    merged_candidates = _shortest_superstrings(subs, hi)
    if not merged_candidates:
        raise UnsatisfiableConstraints(f"str 'contains' cannot fit {subs!r} within length.max ({hi})")
    if chars == "":
        valid = [m for m in merged_candidates if lo <= len(m) <= hi]
        if not valid:
            raise UnsatisfiableConstraints("an empty 'alphabet' cannot pad 'contains' to length.min")
        return st.sampled_from(valid)

    @st.composite
    def with_merged(draw):
        core = draw(st.sampled_from(merged_candidates))
        extra_hi = hi - len(core)
        filler = draw(filler_text(max(0, lo - len(core)), extra_hi)) if extra_hi > 0 else ""
        if not filler:
            return core
        cut = draw(st.integers(min_value=0, max_value=len(filler)))
        return filler[:cut] + core + filler[cut:]

    return with_merged()


def _bytes_strategy(constraints: dict) -> Any:
    import hypothesis.strategies as st

    if "values" in constraints:
        values = constraints["values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise UnsatisfiableConstraints("bytes 'values' must be a non-empty sequence")
        return st.sampled_from([bytes(v) for v in values if isinstance(v, (bytes, bytearray))] or
                               _raise("bytes 'values' contains no bytes"))
    lo, hi = _length_bounds(constraints, default_max=64)
    return st.binary(min_size=lo, max_size=hi)


def _same_member(a: Any, b: Any) -> bool:
    ka = canonical_key(a)
    if ka is not None:
        return ka == canonical_key(b)
    return type(a) is type(b) and bool(a == b)


def _list_contains(constraints: dict, elem: dict | None) -> list:
    """Required elements on a ``list`` constraint dict, validated against ``elem`` and deduplicated."""
    raw = constraints.get("contains")
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise UnsatisfiableConstraints(
            f"list 'contains' must be a list or tuple of elements, got {type(raw).__name__}")
    uniq: list = []
    for item in raw:
        if elem is not None and not contains(elem, item):
            raise UnsatisfiableConstraints(f"list 'contains' element {item!r} is outside element_constraints")
        if not any(_same_member(item, w) for w in uniq):
            uniq.append(item)
    return uniq


def _sequence_strategy(constraints: dict, *, as_tuple: bool) -> Any:
    import hypothesis.strategies as st

    if as_tuple and "items" in constraints:
        items = constraints["items"]
        if not isinstance(items, (list, tuple)) or "element_constraints" in constraints or "length" in constraints:
            raise UnsatisfiableConstraints("tuple 'items' is a list of dicts, and excludes element_constraints/length")
        return st.tuples(*[make_strategy(item) for item in items])
    elem = constraints.get("element_constraints")
    element_strategy = make_strategy(elem) if elem else st.integers()
    required = _list_contains(constraints, elem) if not as_tuple else []
    lo, hi = _length_bounds(constraints, default_max=max(16, len(required)))
    if len(required) > hi:
        raise UnsatisfiableConstraints(
            f"list 'contains' needs at least {len(required)} elements, > length.max ({hi})")
    if required:
        @st.composite
        def with_elements(draw):
            extra_hi = hi - len(required)
            extra = (draw(st.lists(element_strategy, min_size=max(0, lo - len(required)), max_size=extra_hi))
                     if extra_hi > 0 else [])
            out = list(extra)
            order = list(draw(st.permutations(required))) if len(required) > 1 else required
            for item in reversed(order):
                at = draw(st.integers(min_value=0, max_value=len(out)))
                out.insert(at, copy.deepcopy(item) if isinstance(item, (list, dict, set, bytearray)) else item)
            return out

        return with_elements()
    base = st.lists(element_strategy, min_size=lo, max_size=hi)
    # A tuple is built by mapping a list rather than st.tuples, because the length here
    # is a BOUND rather than a fixed arity.
    return base.map(tuple) if as_tuple else base


def _dict_strategy(constraints: dict) -> Any:
    import hypothesis.strategies as st

    keys = constraints.get("key_constraints")
    values = constraints.get("value_constraints")
    key_strategy = make_strategy(keys) if keys else st.text(max_size=8)
    value_strategy = make_strategy(values) if values else st.integers()
    lo, hi = _length_bounds(constraints, default_max=8)
    return st.dictionaries(key_strategy, value_strategy, min_size=lo, max_size=hi)


# --------------------------------------------------------------------------- [internal ref]: the rest of the builtin kinds
def _bool_strategy(constraints: dict) -> Any:
    import hypothesis.strategies as st
    return st.booleans()


def _none_strategy(constraints: dict) -> Any:
    import hypothesis.strategies as st
    return st.none()


def _complex_strategy(constraints: dict) -> Any:
    """A complex as two floats (``part_constraints``, default the float type's own), built with ``complex(re, im)``,
    which keeps both parts exactly (a ``-0.0`` imaginary part too)."""
    import hypothesis.strategies as st
    part = constraints.get("part_constraints") or {"type": "float"}
    if not isinstance(part, dict) or part.get("type") != "float":
        raise UnsatisfiableConstraints("complex 'part_constraints' must be a float dict")
    p = make_strategy(part)
    return st.builds(complex, p, p)


def _decimal_strategy(constraints: dict) -> Any:
    import hypothesis.strategies as st
    out = st.decimals(allow_nan=bool(constraints.get("allow_nan")),
                      allow_infinity=bool(constraints.get("allow_infinity")))
    if constraints.get("allow_nan") and constraints.get("allow_snan", True) is False:
        # an sNaN is one of the four NaN values decimals() draws among every other Decimal: the filter rejects
        # few, and hash() refuses an sNaN, so a set element or a dict key cannot be one
        out = out.filter(lambda d: not d.is_snan())
    return out


def _fraction_strategy(constraints: dict) -> Any:
    import hypothesis.strategies as st
    return st.fractions()


def _bytearray_strategy(constraints: dict) -> Any:
    return _bytes_strategy(constraints).map(bytearray)


def _set_strategy(constraints: dict, *, frozen: bool) -> Any:
    import hypothesis.strategies as st
    elem = constraints.get("element_constraints")
    element_strategy = make_strategy(elem) if elem else st.integers()
    lo, hi = _length_bounds(constraints, default_max=16)
    out = st.frozensets(element_strategy, min_size=lo, max_size=hi)
    return out if frozen else out.map(set)


def _one_of_strategy(constraints: dict) -> Any:
    import hypothesis.strategies as st
    options = constraints.get("options")
    if not isinstance(options, (list, tuple)) or not options:
        raise UnsatisfiableConstraints("one_of 'options' must be a non-empty list of dicts")
    return st.one_of(*[make_strategy(o) for o in options])


def _literal_strategy(constraints: dict) -> Any:
    import hypothesis.strategies as st
    values = constraints.get("values")
    if not isinstance(values, (list, tuple)) or not values:
        raise UnsatisfiableConstraints("literal 'values' must be a non-empty sequence")
    return st.sampled_from(list(values))


_BUILDERS = {
    "namespace": lambda c: _namespace_strategy(c), "callable": lambda c: _callable_strategy(c),
    "int": _int_strategy, "float": _float_strategy, "str": _str_strategy, "bytes": _bytes_strategy,
    "list": lambda c: _sequence_strategy(c, as_tuple=False), "tuple": lambda c: _sequence_strategy(c, as_tuple=True),
    "dict": _dict_strategy, "bool": _bool_strategy, "none": _none_strategy, "complex": _complex_strategy,
    "decimal": _decimal_strategy, "fraction": _fraction_strategy, "bytearray": _bytearray_strategy,
    "set": lambda c: _set_strategy(c, frozen=False), "frozenset": lambda c: _set_strategy(c, frozen=True),
    "one_of": _one_of_strategy, "literal": _literal_strategy,
}


# --------------------------------------------------------------------------- the regex fields (TK-STR R1-R5, [internal ref])
#: the flag names a ``flags`` / ``flag_difference`` list may hold
REGEX_FLAGS = {"IGNORECASE": re.IGNORECASE, "DOTALL": re.DOTALL, "MULTILINE": re.MULTILINE, "ASCII": re.ASCII,
               "VERBOSE": re.VERBOSE, "UNICODE": re.UNICODE}
#: the named alphabets (``alphabet``): ``printable`` is SV's ``nondet_str`` domain ([internal ref])
ALPHABETS = {"printable": None, "ascii": None}
_SINGLE = ("LITERAL", "NOT_LITERAL", "ANY", "IN", "CATEGORY")
_ZERO = ("AT", "ASSERT", "ASSERT_NOT")
_REPEATS = ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT")
_CATEGORY_TEXT = {"CATEGORY_DIGIT": r"\d", "CATEGORY_NOT_DIGIT": r"\D", "CATEGORY_SPACE": r"\s",
                  "CATEGORY_NOT_SPACE": r"\S", "CATEGORY_WORD": r"\w", "CATEGORY_NOT_WORD": r"\W"}
_AT_TEXT = {"AT_BEGINNING": "^", "AT_BEGINNING_STRING": r"\A", "AT_END": "$", "AT_END_STRING": r"\Z",
            "AT_BOUNDARY": r"\b", "AT_NON_BOUNDARY": r"\B"}
_INLINE = (("i", re.IGNORECASE), ("m", re.MULTILINE), ("s", re.DOTALL), ("x", re.VERBOSE), ("a", re.ASCII),
           ("u", re.UNICODE))
#: what ASCII changes: a non-ASCII member of each class (``\w`` / ``\d`` / ``\s`` take it under UNICODE only)
_NON_ASCII = {"CATEGORY_WORD": "é", "CATEGORY_NOT_WORD": "é", "CATEGORY_DIGIT": "٣",
              "CATEGORY_NOT_DIGIT": "٣", "CATEGORY_SPACE": " ", "CATEGORY_NOT_SPACE": " "}
#: The initial length-table work window; empty windows widen to the declared bound.
MAX_REGEX_LENGTH = 1 << 16


def _regex_flags(names) -> int:
    """``flags`` / ``flag_difference``: an int of ``re`` flags, or a list of names; ``re.DEBUG`` masked off (it prints
    on compile, TK-STR R5)."""
    if names is None:
        return 0
    if isinstance(names, int) and not isinstance(names, bool):        # an int, or a ``re.RegexFlag``
        return int(names) & ~re.DEBUG
    if not isinstance(names, (list, tuple)) or any(n not in REGEX_FLAGS for n in names):
        raise UnsatisfiableConstraints(f"'flags' is an int or a list of {sorted(REGEX_FLAGS)}, got {names!r}")
    out = 0
    for n in names:
        out |= REGEX_FLAGS[n]
    return out


def _alphabet(name):
    """``alphabet``: ``"printable"``, ``"ascii"``, or the str of the allowed characters (``None``: every one)."""
    import string

    import hypothesis.strategies as st
    if name is None:
        return None
    if name == "printable":
        return string.printable
    if name == "ascii":
        return st.characters(codec="ascii")
    if not isinstance(name, str):
        raise UnsatisfiableConstraints(f"'alphabet' is 'printable', 'ascii' or a str of characters, got {name!r}")
    return name


def _name(x) -> str:
    return str(x)


def _refusal(seq) -> str | None:
    """Why a parsed pattern is not generated here (TK-STR R5), or ``None``."""
    for op, av in seq:
        n = _name(op)
        if n == "GROUPREF":
            return "a backreference is not built"
        if n == "GROUPREF_EXISTS":
            return "a conditional group is not built"
        if n in ("ASSERT", "ASSERT_NOT") and av[0] < 0:
            return "a lookbehind is not built"
        for sub in _subsequences(n, av):
            why = _refusal(sub)
            if why:
                return why
    return None


def _subsequences(n: str, av) -> list:
    if n == "SUBPATTERN":
        return [av[3]]
    if n == "BRANCH":
        return list(av[1])
    if n in _REPEATS:
        return [av[2]]
    if n in ("ASSERT", "ASSERT_NOT"):
        return [av[1]]
    if n == "ATOMIC_GROUP":
        return [av]
    return []


def _esc(c: int) -> str:
    return re.escape(chr(c))


def _class_text(items) -> str:
    parts, negate = [], False
    for op, av in items:
        n = _name(op)
        if n == "NEGATE":
            negate = True
        elif n == "LITERAL":
            parts.append(_esc(av))
        elif n == "RANGE":
            parts.append(_esc(av[0]) + "-" + _esc(av[1]))
        elif n == "CATEGORY" and _name(av) in _CATEGORY_TEXT:
            parts.append(_CATEGORY_TEXT[_name(av)])
        else:
            raise UnsatisfiableConstraints(f"a class member {n} is not built")
    return "[" + ("^" if negate else "") + "".join(parts) + "]"


def _quant(lo: int, hi) -> str:
    from re import _parser
    if hi is None or hi == _parser.MAXREPEAT:
        return "{%d,}" % lo
    return "{%d}" % lo if lo == hi else "{%d,%d}" % (lo, hi)


def _inline(add: int, delete: int) -> str:
    on = "".join(c for c, f in _INLINE if add & f)
    off = "".join(c for c, f in _INLINE if delete & f)
    return on + ("-" + off if off else "")


def _render(seq) -> str:
    """The source of a parsed pattern (``re._parser``'s tree): what is compiled again, with the pattern's own global
    flags, after a node is rewritten (an exact count, a chosen branch, a flag variant)."""
    return "".join(_render_node(op, av) for op, av in seq)


def _render_node(op, av) -> str:
    n = _name(op)
    if n == "LITERAL":
        return _esc(av)
    if n == "NOT_LITERAL":
        return "[^" + _esc(av) + "]"
    if n == "ANY":
        return "."
    if n == "IN":
        return _class_text(av)
    if n == "CATEGORY":
        return _CATEGORY_TEXT[_name(av)]
    if n == "AT":
        return _AT_TEXT[_name(av)]
    if n == "SUBPATTERN":
        return "(?" + _inline(av[1], av[2]) + ":" + _render(av[3]) + ")"
    if n == "BRANCH":
        return "(?:" + "|".join(_render(alt) for alt in av[1]) + ")"
    if n in _REPEATS:
        suffix = {"MIN_REPEAT": "?", "POSSESSIVE_REPEAT": "+"}.get(n, "")
        return "(?:" + _render(av[2]) + ")" + _quant(av[0], av[1]) + suffix
    if n == "ATOMIC_GROUP":
        return "(?>" + _render(av) + ")"
    if n == "ASSERT":
        return "(?=" + _render(av[1]) + ")"
    if n == "ASSERT_NOT":
        return "(?!" + _render(av[1]) + ")"
    raise UnsatisfiableConstraints(f"a pattern node {n} is not built")


def _conv(a: int, b: int, cap: int) -> int:
    """The lengths ``x + y`` (``x`` in bitset ``a``, ``y`` in ``b``), up to ``cap``."""
    out, i = 0, 0
    while b:
        if b & 1:
            out |= a << i
        b >>= 1
        i += 1
    return out & ((1 << (cap + 1)) - 1)


def _bits(x: int) -> list:
    out, i = [], 0
    while x:
        if x & 1:
            out.append(i)
        x >>= 1
        i += 1
    return out


def _counts(sub: int, lo: int, hi, cap: int) -> list:
    """``[(count, lengths of count iterations)]`` for a repeat of a node of lengths ``sub``, count in [lo, hi]."""
    from re import _parser
    if hi is None or hi == _parser.MAXREPEAT:
        hi = max(lo, cap) if sub & 1 else cap
    elif sub & 1 == 0:
        hi = min(hi, cap)
    out, reach, c = [], 1, 0
    while c <= hi:
        if c >= lo:
            out.append((c, reach))
        nxt = _conv(reach, sub, cap)
        if nxt == 0:
            break
        if nxt == reach:
            if c < lo <= hi:
                c = lo
                continue
            break
        reach, c = nxt, c + 1
    return out


def _widths(seq, cap: int) -> int:
    out = 1
    for op, av in seq:
        out = _conv(out, _node_widths(op, av, cap), cap)
    return out


def _length_options(seq, lo: int, hi: int) -> list[int]:
    """Admitted widths; a work window cannot establish that the declared interval is empty."""
    from re import _parser

    cap = min(hi, MAX_REGEX_LENGTH)
    while True:
        options = [w for w in _bits(_widths(seq, cap)) if lo <= w <= hi]
        if options or cap == hi:
            return options
        minimum, maximum = _parser.SubPattern(None, list(seq)).getwidth()
        hi = min(hi, maximum)
        if lo > hi or minimum > hi:
            return []
        cap = min(hi, max(cap * 2, lo, minimum))


def _node_widths(op, av, cap: int) -> int:
    """The lengths a node can match, as a bitset up to ``cap``."""
    n = _name(op)
    if n in _SINGLE:
        return 1 << 1 if cap >= 1 else 0
    if n in _ZERO:
        return 1
    if n == "SUBPATTERN":
        return _widths(av[3], cap)
    if n == "ATOMIC_GROUP":
        return _widths(av, cap)
    if n == "BRANCH":
        out = 0
        for alt in av[1]:
            out |= _widths(alt, cap)
        return out
    if n in _REPEATS:
        out = 0
        for _, reach in _counts(_widths(av[2], cap), av[0], av[1], cap):
            out |= reach
        return out
    raise UnsatisfiableConstraints(f"a pattern node {n} is not built")


#: how the slack of an exact length is spread over a sequence: to the right, to the left, evenly, or drawn
SLACK = ("right", "left", "even", "drawn")


def _choose(options: list, mode: str, draw, target=None):
    import hypothesis.strategies as st
    if mode == "right":
        return options[0]
    if mode == "left":
        return options[-1]
    if mode == "even" and target is not None:
        return min(options, key=lambda o: (abs(o - target), o))
    return draw(st.sampled_from(options))


def _fix_seq(seq, length: int, draw, mode: str) -> str:
    """``seq`` rendered with every repeat count and branch fixed so that it matches exactly ``length`` characters (the
    choices drawn, the slack spread by ``mode``)."""
    tables = [_node_widths(op, av, length) for op, av in seq]
    suffix = [1] * (len(seq) + 1)
    for i in range(len(seq) - 1, -1, -1):
        suffix[i] = _conv(tables[i], suffix[i + 1], length)
    if not (suffix[0] >> length) & 1:
        raise UnsatisfiableConstraints(f"no member has length {length}")
    out, rem = [], length
    for i, (op, av) in enumerate(seq):
        feasible = [w for w in _bits(tables[i]) if w <= rem and (suffix[i + 1] >> (rem - w)) & 1]
        free = sum(1 for t in tables[i:] if t & (t - 1))          # the nodes left whose length can vary
        least = sum(min(_bits(t)) for t in tables[i:])
        w = _choose(feasible, mode, draw, min(_bits(tables[i])) + (rem - least) / max(free, 1))
        out.append(_fix_node(op, av, w, draw, mode))
        rem -= w
    return "".join(out)


def _fix_node(op, av, length: int, draw, mode: str) -> str:
    n = _name(op)
    if n in _SINGLE or n in _ZERO:
        return _render_node(op, av)
    if n == "SUBPATTERN":
        return "(?" + _inline(av[1], av[2]) + ":" + _fix_seq(av[3], length, draw, mode) + ")"
    if n == "ATOMIC_GROUP":
        return "(?:" + _fix_seq(av, length, draw, mode) + ")"
    if n == "BRANCH":
        alts = [alt for alt in av[1] if (_widths(alt, length) >> length) & 1]
        return "(?:" + _fix_seq(_choose(alts, "drawn" if mode == "drawn" else "right", draw), length, draw, mode) + ")"
    lo, hi, sub = av
    sub_widths = _widths(sub, length)
    if sub_widths == 1 << 1:                                       # one character a time: the count IS the length
        return "(?:" + _render(sub) + ")" + _quant(length, length)
    counts = [c for c, reach in _counts(sub_widths, lo, hi, length) if (reach >> length) & 1]
    count = _choose(counts, mode, draw)
    return _fix_seq([(_SUBPATTERN_OP(), (None, 0, 0, sub))] * count, length, draw, mode)


def _SUBPATTERN_OP():
    from re import _constants
    return _constants.SUBPATTERN


def _parse(constraints: dict):
    """``(tree, flags, how)`` of a ``regex`` constraint, every refusal named (TK-STR R5)."""
    from re import _parser
    regex = constraints["regex"]
    if isinstance(regex, (bytes, bytearray)):
        raise UnsatisfiableConstraints("a bytes pattern is not a str pattern (not built)")
    if not isinstance(regex, str):
        raise UnsatisfiableConstraints(f"'regex' must be a pattern string, got {type(regex).__name__}")
    flags = _regex_flags(constraints.get("flags"))
    if flags & re.LOCALE:
        raise UnsatisfiableConstraints("re.LOCALE is refused: its classes depend on the process's locale")
    how = constraints.get("how", "fullmatch" if constraints.get("fullmatch", True) else "search")
    if how not in ("fullmatch", "match", "search"):
        raise UnsatisfiableConstraints(f"'how' is 'fullmatch', 'match' or 'search', not {how!r}")
    try:
        tree = _parser.parse(regex, flags)
    except (re.error, OverflowError, RecursionError) as exc:
        raise UnsatisfiableConstraints(f"the pattern does not parse: {exc}") from None
    why = _refusal(list(tree))
    if why:
        raise UnsatisfiableConstraints(why)
    if tree.state.flags & re.LOCALE:
        raise UnsatisfiableConstraints("re.LOCALE is refused: its classes depend on the process's locale")
    return list(tree), (tree.state.flags & ~(re.DEBUG | re.VERBOSE)), how


def _from_text(text: str, flags: int, fullmatch: bool, alphabet):
    import hypothesis.strategies as st
    try:
        pattern = re.compile(text, flags)
    except re.error as exc:                                        # pragma: no cover - a rendering bug
        raise UnsatisfiableConstraints(f"the rewritten pattern does not compile: {exc}") from None
    if alphabet is None and any(0xD800 <= ord(char) <= 0xDFFF for char in text):
        # CPython accepts these literals; Hypothesis's default UTF-8 alphabet does not.
        alphabet = st.characters()
    return st.from_regex(pattern, fullmatch=fullmatch, alphabet=alphabet)


def _regex_strategy(constraints: dict) -> Any:
    """Members of a pattern's language (TK-STR R1-R4): narrowed by ``alphabet`` (Hypothesis's own
    ``from_regex(alphabet=...)``) and, for ``length`` / ``segments``, by rewriting the pattern to exact repeat counts
    and branch choices before ``from_regex``; ``flag_difference`` gives the variants of :func:`_flag_variants`."""
    import hypothesis.strategies as st

    seq, flags, how = _parse(constraints)
    alphabet = _alphabet(constraints.get("alphabet"))
    if "flag_difference" in constraints:
        variants = _flag_variants(seq, flags, _regex_flags(constraints["flag_difference"]))
        return st.one_of(*[_from_text(v, flags, True, alphabet) for v in variants])
    if "segments" in constraints:
        return _regex_segments(seq, flags, constraints, alphabet)
    if "length" in constraints:
        lo, hi = _length_bounds(constraints, default_max=64)
        lengths = _length_options(seq, lo, hi)
        if not lengths:
            raise UnsatisfiableConstraints(f"no member of the pattern has a length in [{lo}, {hi}]")

        @st.composite
        def exact(draw):
            mode = draw(st.sampled_from(SLACK))
            text = _fix_seq(seq, draw(st.sampled_from(lengths)), draw, mode)
            return draw(_from_text(text, flags, True, alphabet))

        return exact()
    if how == "match":
        return _from_text(r"\A(?:" + _render(seq) + ")", flags, False, alphabet)
    return _from_text(_render(seq), flags, how == "fullmatch", alphabet)


def _segment_length(entry) -> tuple:
    if entry is None:
        return None
    if type(entry) is int and entry >= 0:
        return entry, entry
    lo, hi = _length_bounds({"length": entry}, default_max=0)
    return lo, hi


def _regex_segments(seq, flags: int, constraints: dict, alphabet) -> Any:
    """``segments`` with a ``regex`` (TK-STR R3): the stretches of the TOP-LEVEL sequence between occurrences of the
    literal split character, each fixed to its length."""
    import hypothesis.strategies as st

    split, lengths = _segments_fields(constraints)
    if len(split) != 1:
        raise UnsatisfiableConstraints("a regex's 'split' is one literal character")
    stretches, current = [], []
    for op, av in seq:
        if _name(op) == "LITERAL" and av == ord(split):
            stretches.append(current)
            current = []
        else:
            current.append((op, av))
    stretches.append(current)
    if len(stretches) != len(lengths):
        raise UnsatisfiableConstraints(f"the pattern has {len(stretches)} stretches around {split!r}, "
                                       f"not {len(lengths)}")
    bounds = [_segment_length(e) for e in lengths]
    options = []
    for part, b in zip(stretches, bounds):
        if b is None:
            options.append(None)
            continue
        ok = _length_options(part, *b)
        if not ok:
            raise UnsatisfiableConstraints(f"no member of a stretch has a length in [{b[0]}, {b[1]}]")
        options.append(ok)

    @st.composite
    def joined(draw):
        mode = draw(st.sampled_from(SLACK))
        texts = [_render(part) if ok is None else _fix_seq(part, draw(st.sampled_from(ok)), draw, mode)
                 for part, ok in zip(stretches, options)]
        return draw(_from_text(re.escape(split).join(texts), flags, True, alphabet))

    return joined()


def _segments_fields(constraints: dict) -> tuple:
    seg = constraints["segments"]
    split = seg.get("split", seg.get("sep")) if isinstance(seg, dict) else None
    lengths = seg.get("lengths") if isinstance(seg, dict) else None
    if not isinstance(split, str) or not split or not isinstance(lengths, (list, tuple)) or not lengths:
        raise UnsatisfiableConstraints("'segments' is {'split': non-empty str, 'lengths': [int | {'min', 'max'} | None]}")
    return split, list(lengths)


def _segments_strategy(constraints: dict, alphabet) -> Any:
    """``segments`` without a regex: the segments joined by ``split``, segment i of its length, no segment holding a
    character of ``split`` (so ``s.split(split)`` gives exactly those segments)."""
    import hypothesis.strategies as st

    if "length" in constraints:
        raise UnsatisfiableConstraints("'segments' with a total 'length' is not built")
    split, lengths = _segments_fields(constraints)
    chars = _alphabet(alphabet)
    if chars is None:
        char = st.characters(exclude_characters=set(split))
    elif isinstance(chars, str):
        allowed = "".join(c for c in chars if c not in split)
        char = st.sampled_from(allowed) if allowed else None
    else:
        char = chars.filter(lambda c: c not in split)              # an ASCII alphabet minus a few separators
    parts = []
    for entry in lengths:
        b = _segment_length(entry) or (0, 64)
        if char is None and b[0] > 0:
            raise UnsatisfiableConstraints("the alphabet holds only separator characters")
        parts.append(st.text(alphabet=char, min_size=b[0], max_size=b[1]) if char is not None else st.just(""))
    return st.tuples(*parts).map(split.join)


def _swapped_class(items) -> str | None:
    """The cased members of a class with their case swapped (``[a-z]`` -> ``[A-Z]``), or ``None``."""
    out = []
    for op, av in items:
        n = _name(op)
        if n == "NEGATE":
            return None
        if n == "LITERAL" and chr(av).swapcase() != chr(av) and len(chr(av).swapcase()) == 1:
            out.append(re.escape(chr(av).swapcase()))
        elif n == "RANGE":
            a, b = chr(av[0]), chr(av[1])
            if a.isascii() and b.isascii() and a.isalpha() and b.isalpha() and a.islower() == b.islower():
                out.append(re.escape(a.swapcase()) + "-" + re.escape(b.swapcase()))
    return "[" + "".join(out) + "]" if out else None


def _node_variants(op, av, differ: int, forced: int) -> list:
    """The renderings of one node that exercise a flag of ``differ`` (one per way), ``forced`` being the flags an
    inline group fixes there (a ``(?s:.)`` reads no global DOTALL)."""
    n, live = _name(op), differ & ~forced
    out = []
    if n == "ANY" and live & re.DOTALL:
        out.append(r"\n")
    if n == "LITERAL" and live & re.IGNORECASE and chr(av).swapcase() != chr(av) and len(chr(av).swapcase()) == 1:
        out.append(re.escape(chr(av).swapcase()))
    if n == "IN" and live & re.IGNORECASE:
        swapped = _swapped_class(av)
        if swapped:
            out.append(swapped)
    if live & re.ASCII:
        cats = [_name(av)] if n == "CATEGORY" else [_name(x) for o, x in av if _name(o) == "CATEGORY"] if n == "IN" else []
        out += [_NON_ASCII[c] for c in cats if c in _NON_ASCII][:1]
    if n == "AT" and live & re.MULTILINE:
        if _name(av) == "AT_BEGINNING":
            out.append(r"\n")
        elif _name(av) == "AT_END":
            out.append(r"\n[^\n]")
    return out


def _variants_seq(seq, differ: int, forced: int) -> list:
    out = []
    for i, (op, av) in enumerate(seq):
        head, tail = _render(seq[:i]), _render(seq[i + 1:])
        out += [head + v + tail for v in _variants_node(op, av, differ, forced)]
    return out


def _variants_node(op, av, differ: int, forced: int) -> list:
    n = _name(op)
    if n == "SUBPATTERN":
        inner = forced | av[1] | av[2]
        return ["(?" + _inline(av[1], av[2]) + ":" + v + ")" for v in _variants_seq(av[3], differ, inner)]
    if n == "BRANCH":                                              # only the alternative that holds the node
        return ["(?:" + v + ")" for alt in av[1] for v in _variants_seq(alt, differ, forced)]
    if n in _REPEATS:                                              # one iteration is the variant
        lo, hi, sub = av
        from re import _parser
        rest = None if hi is None or hi == _parser.MAXREPEAT else hi - 1
        return ["(?:" + v + ")(?:" + _render(sub) + ")" + _quant(max(lo - 1, 0), rest)
                for v in _variants_seq(sub, differ, forced) if rest is None or rest >= 0]
    if n == "ATOMIC_GROUP":
        return ["(?:" + v + ")" for v in _variants_seq(av, differ, forced)]
    return _node_variants(op, av, differ, forced)


def _flag_variants(seq, flags: int, other: int) -> list:
    """TK-STR R4: one rendering per node whose meaning differs between ``flags`` and ``other`` (DOTALL, IGNORECASE,
    MULTILINE, ASCII), in pattern order, each restricted to the path through that node (a branch is its alternative,
    a repeat takes the variant once).  Their members are NOT membership-checked: the caller compares the two
    flag sets' native results."""
    differ = (flags ^ other) & (re.DOTALL | re.IGNORECASE | re.MULTILINE | re.ASCII)
    variants = list(dict.fromkeys(_variants_seq(seq, differ, 0))) if differ else []
    if not variants:
        raise UnsatisfiableConstraints("no node reads the differing flags")
    return variants


def _regex_leading(constraints: dict) -> list:
    """The strategies whose SIMPLEST member leads ``examples()`` for a regex: each ``flag_difference`` variant (TK-STR
    R4: the differing member first), each top-level alternation branch (R1), the members over ``prefer_chars``."""
    seq, flags, how = _parse(constraints)
    alphabet = _alphabet(constraints.get("alphabet"))
    out = []
    if "flag_difference" in constraints:
        return [_from_text(v, flags, True, alphabet)
                for v in _flag_variants(seq, flags, _regex_flags(constraints["flag_difference"]))]
    if "length" in constraints or "segments" in constraints:
        return out
    top = seq
    while len(top) == 1 and _name(top[0][0]) == "SUBPATTERN":
        top = list(top[0][1][3])
    if len(top) == 1 and _name(top[0][0]) == "BRANCH":
        out += [_from_text(_render(alt), flags, True, alphabet) for alt in top[0][1][1]]
    prefer = constraints.get("prefer_chars")
    if prefer:
        if not isinstance(prefer, str):
            raise UnsatisfiableConstraints("'prefer_chars' is a str of characters")
        out.append(_from_text(_render(seq), flags, True, prefer))
    return out


# --------------------------------------------------------------------------- object leaves (AV R3, R4)
#: the attribute-bag classes a ``namespace`` may be (AV R3: the parameter's declared bag type)
NAMESPACE_CLASSES = ("types.SimpleNamespace", "argparse.Namespace")


def _namespace_class(name: str):
    if name == "argparse.Namespace":
        import argparse
        return argparse.Namespace
    import types
    return types.SimpleNamespace


def _namespace_strategy(constraints: dict) -> Any:
    """Duck typing: an object with exactly the given attributes, of ``class`` (``types.SimpleNamespace`` or
    ``argparse.Namespace``); ``vars()`` of it is the attributes."""
    import hypothesis.strategies as st
    attributes = constraints.get("attributes")
    if not isinstance(attributes, dict) or not all(isinstance(k, str) and k.isidentifier() for k in attributes):
        raise UnsatisfiableConstraints("namespace 'attributes' maps identifiers to dicts")
    cls = constraints.get("class", "types.SimpleNamespace")
    if cls not in NAMESPACE_CLASSES:
        raise UnsatisfiableConstraints(f"namespace 'class' is one of {list(NAMESPACE_CLASSES)}, not {cls!r}")
    return st.builds(_namespace_class(cls), **{k: make_strategy(v) for k, v in attributes.items()})


class ConstantFunction:
    """A callable that returns ``value`` on every call, whatever the arguments: the one drawn value that is not a
    plain literal (AV records it as ``{"const_fn": <literal>}`` and renders ``lambda *a, **k: <value>``)."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __call__(self, *args, **kwargs) -> Any:
        return self.value

    def __repr__(self) -> str:
        return f"ConstantFunction({self.value!r})"


#: a contract's result-type word -> the constraint types it is
_CONTRACT_TYPES = {"str": {"str"}, "None": {"none"}, "int": {"int"}, "bool": {"bool", "int"}, "list": {"list"},
                   "float": {"float"}, "complex": {"complex"}, "dict": {"dict"}, "tuple": {"tuple"},
                   "set": {"set"}, "frozenset": {"frozenset"}, "bytes": {"bytes"}, "numeric": {"int", "float"}}


def contract_result_types(name: str) -> set:
    """The constraint types a builtin's result can be, read off its contract (``solve/contracts``:
    ``registry.get(name).returns.type``); empty when it has none."""
    from tharkun.solve.contracts import registry
    contract = registry.get(name)
    spec = getattr(contract, "returns", None)
    out: set = set()
    for word in str(getattr(spec, "type", "") or "").split("|"):
        word = word.strip().split(" ")[0].split("(")[0]
        out |= _CONTRACT_TYPES.get(word, set())
    return out


def _allowed_builtins(constraints: dict) -> list:
    """The ``allow``-listed builtins whose contract's result type is ``returns``'s, in ``allow`` order (AV R4.1)."""
    import builtins
    allow = constraints.get("allow", [])
    if not isinstance(allow, (list, tuple)) or not all(isinstance(a, str) for a in allow):
        raise UnsatisfiableConstraints("callable 'allow' is a list of builtin names")
    want = constraints["returns"].get("type")
    return [getattr(builtins, a) for a in allow
            if callable(getattr(builtins, a, None)) and want in contract_result_types(a)]


def _callable_strategy(constraints: dict) -> Any:
    """AV R4: an allowlisted builtin whose contract matches ``returns`` (the caller's ``allow`` vocabulary), else a
    :class:`ConstantFunction` of a drawn ``returns`` value.  (``st.functions`` is not used: what it makes can be
    called only inside its own ``@given``.)"""
    import hypothesis.strategies as st
    returns = constraints.get("returns")
    if not isinstance(returns, dict):
        raise UnsatisfiableConstraints("callable 'returns' is a constraint dict")
    const = make_strategy(returns).map(ConstantFunction)
    allowed = _allowed_builtins(constraints)
    return st.one_of(st.sampled_from(allowed), const) if allowed else const


# --------------------------------------------------------------------------- membership predicate
def _int_bounds_check(constraints: dict) -> tuple[int | None, int | None]:
    rng = constraints.get("value_range")
    if rng is not None:
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            raise UnsatisfiableConstraints(f"'value_range' must be [min, max], got {rng!r}")
        lo, hi = rng
        if (lo is not None and not _is_int(lo)) or (hi is not None and not _is_int(hi)):
            raise UnsatisfiableConstraints(f"value_range bounds must be numeric, got {rng!r}")
        if lo is not None and hi is not None and lo > hi:
            raise UnsatisfiableConstraints(f"value_range min ({lo}) > max ({hi}); empty domain")
        return lo, hi
    if constraints.get("unbounded"):
        return None, None
    return INT_DEFAULT_MIN, INT_DEFAULT_MAX


def _int_contains(constraints: dict, value: Any) -> bool:
    if "values" in constraints:
        values = constraints["values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise UnsatisfiableConstraints("int 'values' must be a non-empty sequence")
        for v in values:
            if not _is_int(v):
                raise UnsatisfiableConstraints(
                    f"int 'values' contains a non-int element {v!r} ({type(v).__name__})")
        return _is_int(value) and value in values

    lo, hi = _int_bounds_check(constraints)
    if "bitmask_mask" in constraints:
        mask = constraints["bitmask_mask"]
        if not _is_int(mask) or mask < 0:
            raise UnsatisfiableConstraints(f"'bitmask_mask' must be a non-negative int, got {mask!r}")
        lo_m = 0 if lo is None else max(lo, 0)
        hi_m = mask if hi is None else min(hi, mask)
        if lo_m > hi_m:
            raise UnsatisfiableConstraints(
                f"bitmask_mask={mask} leaves an empty range given the other constraints")
        return _is_int(value) and lo_m <= value <= hi_m and (value & ~mask) == 0

    if "shift_direction" in constraints or "shift_bits" in constraints:
        direction = constraints.get("shift_direction", "left")
        bits = constraints.get("shift_bits", 0)
        if not _is_int(bits) or bits < 0:
            raise UnsatisfiableConstraints(f"'shift_bits' must be a non-negative int, got {bits!r}")
        if direction == "left":
            step = 1 << bits
            lo_q = None if lo is None else -(-lo // step) * step
            hi_q = None if hi is None else (hi // step) * step
            if lo_q is not None and hi_q is not None and lo_q > hi_q:
                raise UnsatisfiableConstraints(
                    f"left shift by {bits} leaves no multiple of {step} in [{lo}, {hi}]")
            return (_is_int(value) and (lo is None or value >= lo)
                    and (hi is None or value <= hi) and value % step == 0)
        return _is_int(value) and (lo is None or value >= lo) and (hi is None or value <= hi)

    return _is_int(value) and (lo is None or value >= lo) and (hi is None or value <= hi)


def _float_contains(constraints: dict, value: Any) -> bool:
    def _num(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    if "values" in constraints:
        values = constraints["values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise UnsatisfiableConstraints("float 'values' must be a non-empty sequence")
        typed = [float(v) for v in values if _num(v)]
        if not typed:
            raise UnsatisfiableConstraints("float 'values' contains no numeric element")
        return type(value) is float and any(_same_member(value, w) for w in typed)

    allow_nan, allow_inf = bool(constraints.get("allow_nan")), bool(constraints.get("allow_infinity"))
    if constraints.get("unbounded") and "value_range" not in constraints:
        if type(value) is not float:
            return False
        if math.isnan(value):
            return allow_nan
        if math.isinf(value):
            return allow_inf
        return True
    lo, hi = _range_bounds(constraints, FLOAT_DEFAULT_MIN, FLOAT_DEFAULT_MAX, _num)
    return type(value) is float and not math.isnan(value) and not math.isinf(value) and float(lo) <= value <= float(hi)


def _regex_segments_contains(seq: list, flags: int, constraints: dict, value: Any) -> bool:
    split, lengths = _segments_fields(constraints)
    if len(split) != 1:
        raise UnsatisfiableConstraints("a regex's 'split' is one literal character")
    stretches, current = [], []
    for op, av in seq:
        if _name(op) == "LITERAL" and av == ord(split):
            stretches.append(current)
            current = []
        else:
            current.append((op, av))
    stretches.append(current)
    if len(stretches) != len(lengths):
        raise UnsatisfiableConstraints(f"the pattern has {len(stretches)} stretches around {split!r}, "
                                       f"not {len(lengths)}")
    bounds = [_segment_length(e) for e in lengths]
    for part, b in zip(stretches, bounds):
        if b is None:
            continue
        if not _length_options(part, *b):
            raise UnsatisfiableConstraints(f"no member of a stretch has a length in [{b[0]}, {b[1]}]")
    if type(value) is not str:
        return False
    pats = [re.compile(_render(part), flags) for part in stretches]

    def match_at(idx: int, pos: int) -> bool:
        b = bounds[idx]
        if idx == len(stretches) - 1:
            piece = value[pos:]
            return (b is None or b[0] <= len(piece) <= b[1]) and pats[idx].fullmatch(piece) is not None
        at = value.find(split, pos)
        while at != -1:
            piece = value[pos:at]
            if b is not None and len(piece) > b[1]:
                break
            if (b is None or len(piece) >= b[0]) and pats[idx].fullmatch(piece) is not None:
                if match_at(idx + 1, at + 1):
                    return True
            at = value.find(split, at + 1)
        return False

    return match_at(0, 0)


def _str_member(constraints: dict, value: Any) -> bool:
    if "values" in constraints:
        values = constraints["values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise UnsatisfiableConstraints("str 'values' must be a non-empty sequence")
        strs = [v for v in values if isinstance(v, str)]
        if not strs:
            raise UnsatisfiableConstraints("str 'values' contains no string")
        return type(value) is str and value in strs

    subs = _str_contains(constraints)
    alphabet = constraints.get("alphabet")
    _alphabet(alphabet)
    for sub in subs:
        if not _alphabet_allows(alphabet, sub):
            raise UnsatisfiableConstraints(f"str 'contains' {sub!r} has a character outside 'alphabet'")

    if constraints.get("regex") is not None:
        if subs and "length" in constraints:
            _, hi = _length_bounds(constraints, default_max=64)
            if not _shortest_superstrings(subs, hi):
                raise UnsatisfiableConstraints(f"str 'contains' cannot fit {subs!r} within length.max ({hi})")
        seq, flags, how = _parse(constraints)
        if "flag_difference" in constraints:
            variants = _flag_variants(seq, flags, _regex_flags(constraints["flag_difference"]))
            return (type(value) is str and _alphabet_allows(alphabet, value) and all(s in value for s in subs)
                    and any(re.compile(v, flags).fullmatch(value) is not None for v in variants))
        if "segments" in constraints:
            ok = _regex_segments_contains(seq, flags, constraints, value)
            return ok and _alphabet_allows(alphabet, value) and all(s in value for s in subs)
        if "length" in constraints:
            lo, hi = _length_bounds(constraints, default_max=64)
            if not _length_options(seq, lo, hi):
                raise UnsatisfiableConstraints(f"no member of the pattern has a length in [{lo}, {hi}]")
            return (type(value) is str and lo <= len(value) <= hi and _alphabet_allows(alphabet, value)
                    and all(s in value for s in subs)
                    and re.compile(_render(seq), flags).fullmatch(value) is not None)
        if type(value) is not str or not _alphabet_allows(alphabet, value) or not all(s in value for s in subs):
            return False
        return getattr(re.compile(_render(seq), flags), how)(value) is not None

    for field in ("flags", "how", "flag_difference", "prefer_chars"):
        if field in constraints:
            raise UnsatisfiableConstraints(f"{field!r} needs a 'regex'")

    if "segments" in constraints:
        if "length" in constraints:
            raise UnsatisfiableConstraints("'segments' with a total 'length' is not built")
        split, lengths = _segments_fields(constraints)
        bounds = [_segment_length(e) or (0, 64) for e in lengths]
        if isinstance(alphabet, str) and alphabet not in ALPHABETS and not any(c not in split for c in alphabet):
            if any(b[0] > 0 for b in bounds):
                raise UnsatisfiableConstraints("the alphabet holds only separator characters")
        if type(value) is not str or not _alphabet_allows(alphabet, value) or not all(s in value for s in subs):
            return False
        parts = value.split(split)
        return len(parts) == len(bounds) and all(
            b[0] <= len(p) <= b[1] and not (set(p) & set(split)) for p, b in zip(parts, bounds))

    sum_len = sum(len(s) for s in subs)
    lo, hi = _length_bounds(constraints, default_max=max(64, sum_len))
    if alphabet == "" and ((not subs and lo > 0) or (subs and sum_len <= hi and lo > sum_len)):
        raise UnsatisfiableConstraints("an empty 'alphabet' has no text of the required length")
    if subs and sum_len > hi:
        merged = _shortest_superstrings(subs, hi)
        if not merged or (alphabet == "" and not any(lo <= len(m) <= hi for m in merged)):
            raise UnsatisfiableConstraints(f"str 'contains' cannot fit {subs!r} within length.max ({hi})")
    if type(value) is not str or not _alphabet_allows(alphabet, value) or not all(s in value for s in subs):
        return False
    return "length" not in constraints or lo <= len(value) <= hi


def _base_contains(kind: str, constraints: dict, value: Any) -> bool:
    if kind == "int":
        return _int_contains(constraints, value)
    if kind == "float":
        return _float_contains(constraints, value)
    if kind == "str":
        return _str_member(constraints, value)
    if kind == "bytes":
        if "values" in constraints:
            values = constraints["values"]
            if not isinstance(values, (list, tuple)) or not values:
                raise UnsatisfiableConstraints("bytes 'values' must be a non-empty sequence")
            bs = [bytes(v) for v in values if isinstance(v, (bytes, bytearray))]
            if not bs:
                raise UnsatisfiableConstraints("bytes 'values' contains no bytes")
            return type(value) is bytes and value in bs
        lo, hi = _length_bounds(constraints, default_max=64)
        return type(value) is bytes and ("length" not in constraints or lo <= len(value) <= hi)
    if kind == "bytearray":
        lo, hi = _length_bounds(constraints, default_max=64)
        return type(value) is bytearray and ("length" not in constraints or lo <= len(value) <= hi)
    if kind in ("list", "tuple"):
        as_tuple = kind == "tuple"
        if as_tuple and "items" in constraints:
            items = constraints["items"]
            if not isinstance(items, (list, tuple)) or "element_constraints" in constraints or "length" in constraints:
                raise UnsatisfiableConstraints("tuple 'items' is a list of dicts, and excludes element_constraints/length")
            checks = [contains(item, value[i]) if type(value) is tuple and i < len(value) else (contains(item, None) and False)
                      for i, item in enumerate(items)]
            return type(value) is tuple and len(value) == len(items) and all(checks)
        elem = constraints.get("element_constraints")
        if elem is not None:
            contains(elem, None)
        required = _list_contains(constraints, elem) if not as_tuple else []
        lo, hi = _length_bounds(constraints, default_max=max(16, len(required)))
        if len(required) > hi:
            raise UnsatisfiableConstraints(
                f"list 'contains' needs at least {len(required)} elements, > length.max ({hi})")
        if type(value) is not (tuple if as_tuple else list):
            return False
        if "length" in constraints and not (lo <= len(value) <= hi):
            return False
        if not all(contains(elem, x) if elem is not None else _is_int(x) for x in value):
            return False
        return all(any(_same_member(req, x) for x in value) for req in required)
    if kind in ("set", "frozenset"):
        elem = constraints.get("element_constraints")
        if elem is not None:
            contains(elem, None)
        lo, hi = _length_bounds(constraints, default_max=16)
        if type(value) is not (frozenset if kind == "frozenset" else set):
            return False
        if "length" in constraints and not (lo <= len(value) <= hi):
            return False
        return all(contains(elem, x) if elem is not None else _is_int(x) for x in value)
    if kind == "dict":
        keys = constraints.get("key_constraints")
        vals = constraints.get("value_constraints")
        if keys is not None:
            contains(keys, None)
        if vals is not None:
            contains(vals, None)
        lo, hi = _length_bounds(constraints, default_max=8)
        if type(value) is not dict:
            return False
        if "length" in constraints and not (lo <= len(value) <= hi):
            return False
        return all((contains(keys, k) if keys is not None else type(k) is str)
                   and (contains(vals, v) if vals is not None else _is_int(v))
                   for k, v in value.items())
    if kind == "bool":
        return type(value) is bool
    if kind == "none":
        return value is None
    if kind == "complex":
        part = constraints.get("part_constraints") or {"type": "float"}
        if not isinstance(part, dict) or part.get("type") != "float":
            raise UnsatisfiableConstraints("complex 'part_constraints' must be a float dict")
        contains(part, None)
        return type(value) is complex and contains(part, value.real) and contains(part, value.imag)
    if kind == "decimal":
        import decimal

        if type(value) is not decimal.Decimal:
            return False
        if value.is_nan():
            if not bool(constraints.get("allow_nan")):
                return False
            return not (value.is_snan() and constraints.get("allow_snan", True) is False)
        if value.is_infinite():
            return bool(constraints.get("allow_infinity"))
        return True
    if kind == "fraction":
        import fractions

        return type(value) is fractions.Fraction
    if kind == "one_of":
        options = constraints.get("options")
        if not isinstance(options, (list, tuple)) or not options:
            raise UnsatisfiableConstraints("one_of 'options' must be a non-empty list of dicts")
        checks = [contains(o, value) for o in options]
        return any(checks)
    if kind == "literal":
        values = constraints.get("values")
        if not isinstance(values, (list, tuple)) or not values:
            raise UnsatisfiableConstraints("literal 'values' must be a non-empty sequence")
        return any(_same_member(value, w) for w in values)
    if kind == "namespace":
        attributes = constraints.get("attributes")
        if not isinstance(attributes, dict) or not all(isinstance(k, str) and k.isidentifier() for k in attributes):
            raise UnsatisfiableConstraints("namespace 'attributes' maps identifiers to dicts")
        cls = constraints.get("class", "types.SimpleNamespace")
        if cls not in NAMESPACE_CLASSES:
            raise UnsatisfiableConstraints(f"namespace 'class' is one of {list(NAMESPACE_CLASSES)}, not {cls!r}")
        for ac in attributes.values():
            contains(ac, None)
        if type(value) is not _namespace_class(cls):
            return False
        bag = vars(value)
        return set(bag) == set(attributes) and all(contains(attributes[k], bag[k]) for k in attributes)
    returns = constraints.get("returns")
    if not isinstance(returns, dict):
        raise UnsatisfiableConstraints("callable 'returns' is a constraint dict")
    contains(returns, None)
    allowed = _allowed_builtins(constraints)
    if any(value is fn for fn in allowed):
        return True
    return type(value) is ConstantFunction and contains(returns, value.value)


def contains(constraints: dict, value: Any) -> bool:
    """Whether ``value`` is a member of the domain ``constraints`` describes (:func:`make_strategy`'s schema),
    checked directly from the dict without building a strategy.  Raises :class:`UnsatisfiableConstraints` on a
    malformed or empty constraint dict, just as :func:`make_strategy` does."""
    if not isinstance(constraints, dict):
        raise UnsatisfiableConstraints(f"constraints must be a dict, got {type(constraints).__name__}")
    kind = constraints.get("type")
    if kind not in _VALID_TYPES:
        raise UnsatisfiableConstraints(f"unknown or missing type {kind!r}")
    if "values" in constraints and kind not in _TYPED_VALUES:
        base_ok = _base_contains("literal", constraints, value)
    else:
        if "contains" in constraints and kind not in ("str", "list"):
            raise UnsatisfiableConstraints(f"'contains' is only valid on 'str' and 'list' domains, not {kind!r}")
        base_ok = _base_contains(kind, constraints, value)
    prefer = constraints.get("prefer")
    if prefer is None:
        return base_ok
    if not isinstance(prefer, (list, tuple)):
        raise UnsatisfiableConstraints(f"'prefer' must be a sequence of values, got {type(prefer).__name__}")
    return base_ok or any(_same_member(value, p) for p in prefer)


# --------------------------------------------------------------------------- narrowing a constraint dict ([internal ref])
#: the sub-constraints of a dict, by field (a list of dicts, or one dict)
_CHILDREN_ONE = ("element_constraints", "key_constraints", "value_constraints", "part_constraints")
_SIZED = ("str", "bytes", "bytearray", "list", "tuple", "set", "frozenset", "dict")


def _contains_min_len(constraints: dict) -> int:
    kind = constraints.get("type")
    if "contains" not in constraints or "values" in constraints:
        return 0
    if kind == "str":
        subs = _str_contains(constraints)
        return min((len(m) for m in _shortest_superstrings(subs)), default=0) if subs else 0
    if kind == "list":
        return len(_list_contains(constraints, constraints.get("element_constraints")))
    return 0


def small(constraints: dict, *, ints: tuple = (-32, 32), floats: tuple = (-32.0, 32.0), text: int = 16,
          collection: int = 4) -> dict:
    """``constraints`` cut to a SMALL domain, as a new dict: an int with no bound of its own is drawn in ``ints``, a float
    in ``floats`` (finite), a text or bytes value is at most ``text`` long, a collection has at most ``collection``
    elements.  What a search that replays every draw in a child process wants: small values shrink and replay fast.
    It only NARROWS an open side or an open length; a bound the constraints already state is a fact and stays."""
    out = dict(constraints)
    kind = out.get("type")
    if kind in ("int", "float") and "values" not in out:
        lo, hi = ints if kind == "int" else floats
        have = out.get("value_range") or (None, None)
        out["value_range"] = [lo if have[0] is None else have[0], hi if have[1] is None else have[1]]
        for field in ("unbounded", "allow_nan", "allow_infinity"):
            out.pop(field, None)
    elif kind in _SIZED and "regex" not in out and "segments" not in out and "values" not in out:
        cap = text if kind in ("str", "bytes", "bytearray") else collection
        floor = _contains_min_len(out)
        have = out.get("length") or {"min": 0, "max": cap}
        if have["max"] > cap >= max(have["min"], floor):
            out["length"] = {"min": have["min"], "max": cap}
    for field in _CHILDREN_ONE:
        if isinstance(out.get(field), dict):
            out[field] = small(out[field], ints=ints, floats=floats, text=text, collection=collection)
    for field in ("items", "options"):
        if isinstance(out.get(field), (list, tuple)):
            out[field] = [small(c, ints=ints, floats=floats, text=text, collection=collection) for c in out[field]]
    if isinstance(out.get("attributes"), dict):
        out["attributes"] = {k: small(v, ints=ints, floats=floats, text=text, collection=collection)
                             for k, v in out["attributes"].items()}
    return out


def truthy(constraints: dict) -> dict | None:
    """``constraints`` narrowed to the values whose ``bool`` is True, as a new dict; ``None`` when there are none (a
    ``None``, a ``False``, an empty domain).  A number is drawn from its positive side, a text or collection has at
    least one element, a union keeps the arms that have a truthy member.  Its simplest value (:func:`examples`) is
    the simplest TRUTHY one: ``1``, ``1.0``, ``"0"``, ``[0]`` -- what ``if parameter:`` needs to go the other way."""
    kind = constraints.get("type")
    out = dict(constraints)
    if "values" in out:
        kept = [v for v in out["values"] if v]
        return dict(out, values=kept) if kept else None
    if kind == "none":
        return None
    if kind == "bool":
        return {"type": "literal", "values": [True]}
    if kind in ("int", "float"):
        one = 1 if kind == "int" else 1.0
        default = (INT_DEFAULT_MIN, INT_DEFAULT_MAX) if kind == "int" else (FLOAT_DEFAULT_MIN, FLOAT_DEFAULT_MAX)
        lo, hi = (out.get("value_range") or (None, None))
        lo, hi = default[0] if lo is None else lo, default[1] if hi is None else hi
        for field in ("unbounded", "allow_nan", "allow_infinity"):
            out.pop(field, None)
        if hi >= one:
            out["value_range"] = [max(lo, one), hi]
        elif lo <= -one:
            out["value_range"] = [lo, min(hi, -one)]
        else:
            return None
        return out
    if kind in _SIZED:
        if kind == "tuple" and "items" in out:
            return out if out["items"] else None
        floor = _contains_min_len(out)
        have = out.get("length") or {"min": 0, "max": max(16, floor)}
        if have["max"] < max(1, floor):
            return None
        out["length"] = {"min": max(have["min"], 1), "max": have["max"]}
        return out
    if kind == "one_of":
        options = [t for t in (truthy(o) for o in out.get("options") or ()) if t is not None]
        return dict(out, options=options) if options else None
    return out


# --------------------------------------------------------------------------- the one deterministic driver
def canonical_key(value: Any) -> Any:
    """A hashable key of ``value`` that two values share exactly when they are the same value of the same type
    (``-0.0`` is not ``0.0``, a NaN is a NaN, a set is its elements whatever their order, a dict its items in
    order); ``None`` for a value it has no key for (a function)."""
    import argparse
    import builtins
    import decimal
    import fractions
    import types

    t = type(value)
    if value is None:
        return ("none",)
    if t in (bool, int, str, bytes):
        return (t.__name__, value)
    if t is float:
        return ("float", "nan" if value != value else value.hex())
    if t is complex:
        return ("complex", canonical_key(value.real), canonical_key(value.imag))
    if t is bytearray:
        return ("bytearray", bytes(value))
    if t is decimal.Decimal:
        return ("decimal", str(value))
    if t is fractions.Fraction:
        return ("fraction", value.numerator, value.denominator)
    if t is re.Pattern:
        return ("regex", canonical_key(value.pattern), int(value.flags))
    if t in (list, tuple, set, frozenset):
        parts = [canonical_key(x) for x in value]
        if any(p is None for p in parts):
            return None
        return (t.__name__, tuple(sorted(parts, key=repr)) if t in (set, frozenset) else tuple(parts))
    if t is ConstantFunction:
        inner = canonical_key(value.value)
        return None if inner is None else ("const_fn", inner)
    if getattr(builtins, getattr(value, "__name__", "") or "", None) is value:
        return ("builtin", value.__name__)
    if t in (dict, types.SimpleNamespace, argparse.Namespace):
        pairs = []
        for k, v in (value.items() if t is dict else sorted(vars(value).items())):
            kk, vk = (canonical_key(k) if t is dict else k), canonical_key(v)
            if kk is None or vk is None:
                return None
            pairs.append((kk, vk))
        return (t.__name__, tuple(pairs))
    return None


def _size(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _leading(constraints: Any) -> tuple:
    """``(values, strategies)`` that lead ``examples()``: the explicit values (``prefer``; a callable's matching
    builtins) and the strategies whose SIMPLEST member comes next (a regex's flag variants, branches,
    ``prefer_chars``)."""
    if not isinstance(constraints, dict):
        return [], []
    values = list(constraints.get("prefer") or [])
    if constraints.get("type") == "callable" and isinstance(constraints.get("returns"), dict):
        values += _allowed_builtins(constraints)
    strategies = _regex_leading(constraints) if constraints.get("type") == "str" and constraints.get("regex") else []
    return values, strategies


class _Stop(Exception):
    """Raised inside the collecting test when ``seconds`` ran out."""


_LOCAL_CONSTANTS_LOCK = threading.RLock()


@contextlib.contextmanager
def without_local_constants():
    """Hypothesis mixes constants it harvests from every LOCAL module in ``sys.modules`` into its draws
    (``hypothesis/internal/conjecture/providers.py::_get_local_constants``: an editable tharkun's own modules among
    them), so a derandomized run depended on which modules the process had loaded.  :func:`examples` promises the
    same list in every process, so its draws run with that pool empty; Hypothesis's own global constants stay, and
    a program's literals reach a strategy only as ``prefer``.  A Hypothesis without the pool is left as it is."""
    with _LOCAL_CONSTANTS_LOCK:
        try:
            from hypothesis.internal.conjecture import providers
            harvest, cache, empty = providers._get_local_constants, providers.CONSTANTS_CACHE.cache, providers.Constants()
        except (ImportError, AttributeError):
            yield
            return
        providers._get_local_constants = lambda: empty
        cache.clear()  # the cache holds each choice's local constants as last computed
        try:
            yield
        finally:
            providers._get_local_constants = harvest
            cache.clear()


#: Backwards compatibility facade for without_local_constants
_without_local_constants = without_local_constants


def _draw(strategy: Any, n: int, deadline: float | None) -> list:
    """The first ``n`` values a derandomized run of ``strategy`` draws: the FIRST is the simplest one (Hypothesis
    runs the simplest choice sequence first), the rest are generated; whatever modules the process has loaded
    (:func:`without_local_constants`)."""
    from hypothesis import HealthCheck, Phase, Verbosity, given, settings

    drawn: list = []

    @settings(derandomize=True, database=None, max_examples=max(1, n), deadline=None, phases=[Phase.generate],
              suppress_health_check=list(HealthCheck), verbosity=Verbosity.quiet, report_multiple_bugs=False)
    @given(strategy)
    def collect(value):
        if deadline is not None and time.monotonic() > deadline:
            raise _Stop()
        drawn.append(value)

    try:
        with without_local_constants():
            collect()
    except Exception:  # noqa: BLE001 -- _Stop, a filter that kept nothing, a strategy that cannot draw here
        pass
    return drawn


def examples(source: Any, n: int, *, order: str = "draw", seconds: float | None = None) -> list:
    """Up to ``n`` DISTINCT values (by :func:`canonical_key`; a value without a key is never merged) of ``source``
    (a constraint dict or a strategy), deterministically: the LEADING values (:func:`_leading`: ``prefer`` in order,
    a callable's matching builtins, the simplest member of each regex flag variant, branch and ``prefer_chars``
    stream), then the drawn ones -- ``derandomize``, no example database, the generate phase only, the SIMPLEST
    value first.  The same constraints give the same list in every process and under every hash seed.
    ``order="shortest"`` sorts the list by size (then key), stably; ``seconds`` bounds the wall time.  The ONE way
    a lane draws rows from this module (docs/INPUTS.md "Strategy rows")."""
    if order not in ("draw", "shortest"):
        raise ValueError(f"order is 'draw' or 'shortest', not {order!r}")
    strategy = make_strategy(source) if isinstance(source, dict) else source
    deadline = None if seconds is None else time.monotonic() + seconds
    values, leading = _leading(source)
    for s in leading:
        values += _draw(s, 1, deadline)
    values += _draw(strategy, n, deadline)
    out, seen = [], set()
    for value in values:
        key = canonical_key(value)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        out.append(value)
        if len(out) >= n:
            break
    if order == "shortest":
        out.sort(key=lambda v: (_size(v), repr(canonical_key(v))))
    return out


# --------------------------------------------------------------------------- the synthesis-side door ([internal ref])
#: the deterministic example values a :class:`Domain` carries
DOMAIN_EXAMPLES = 8
#: the widest exact member set that becomes a ``literal`` kind (a wider set stays a bound or a length)
DOMAIN_MAX_MEMBERS = 64
#: at most this many of the caller's literals of each type are preferred
DOMAIN_LITERALS = 32
#: the rules a domain names, in tag order (``tharkun-strategies:bounds+prefer``)
DOMAIN_RULES = ("type", "bounds", "members", "length", "contains", "regex", "arms", "prefer")

_PY_TYPE = {"int": "int", "float": "float", "str": "str", "bytes": "bytes", "bytearray": "bytearray", "bool": "bool",
            "none": "NoneType", "list": "list", "tuple": "tuple", "dict": "dict", "set": "set",
            "frozenset": "frozenset", "complex": "complex"}


@dataclass(frozen=True)
class Domain:
    """What :func:`domain` says about ONE parameter.

    ``kind``: the ``find_inputs`` kind (docs/INPUTS.md "Kinds and draw domains") cut to what the guards and facts admit
    -- ``min`` / ``max`` of an int, a ``literal`` of an exact member set, a ``max_len``, a ``one_of`` without its
    impossible arms; the input kind itself when nothing narrows (``narrowed`` False).  ``constraints``: the same
    domain as a :func:`make_strategy` dict (a float's bounds live only here), with the caller's in-domain literals
    ``prefer``red.  ``values``: up to ``DOMAIN_EXAMPLES`` deterministic values of it, each in the domain
    (:func:`examples`; the preferred literals first).  ``rules``: which rules fired, in :data:`DOMAIN_RULES` order."""

    kind: dict
    constraints: dict
    values: tuple
    rules: tuple
    narrowed: bool
    declined: tuple = ()

    @property
    def open(self) -> bool:
        """True when no guard, fact or regex narrowed or generated the parameter (program literals may lead its
        examples; they do not constrain it): the domain is "anything of this kind", and a caller that wants more has to
        take it from outside the program (a policy file, a library's knowledge) -- the owner's rule of 2026-10-23,
        "report it as needing policy rather than invent a default"."""
        return set(self.rules) <= {"type", "prefer"}

    @property
    def tag(self) -> str:
        """The provenance tag a value drawn from this domain carries: ``tharkun-strategies:<rule>[+<rule>...]``."""
        return "tharkun-strategies:" + "+".join(self.rules)

    def as_record(self) -> dict:
        """The domain as plain data a caller can bake into a program (a Python literal and JSON): ``kind``,
        ``constraints``, ``values`` and ``tag``.  A value that is not a bool, int, str, finite float or None, or a list or
        tuple of them (a NaN, a ``Decimal``) is left out of ``values`` and ``prefer``: they are examples, not the domain."""
        constraints = _plain_constraints(self.constraints)
        return {"kind": self.kind, "constraints": constraints,
                "values": [v for v in self.values if _plain(v)], "tag": self.tag,
                **({"declined": list(self.declined)} if self.declined else {})}


def _plain(value: Any) -> bool:
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return value == value and value not in (float("inf"), float("-inf"))
    return type(value) in (list, tuple) and all(_plain(v) for v in value)


def _plain_constraints(c: Any) -> Any:
    if isinstance(c, dict):
        return {k: ([v for v in x if _plain(v)] if k in ("prefer", "values") else _plain_constraints(x))
                for k, x in c.items()}
    if isinstance(c, list):
        return [_plain_constraints(x) for x in c]
    return c


def _facts_value(facts: Any):
    """The :class:`tharkun.solve.values.Value` of per-variable facts (``tharkun.solve.fact_env.VarFacts``, or any object
    with the same ``eq`` / ``neq`` / ``lo`` / ``hi`` / ``container_len`` attributes, such as Eregion's own carrier)."""
    from tharkun.solve.fact_env import VarFacts, to_value

    if facts is None:
        return None
    eq = getattr(facts, "eq", None)
    return to_value(VarFacts(
        eq=set(eq) if eq is not None else None, neq=set(getattr(facts, "neq", None) or ()),
        lo=getattr(facts, "lo", None), hi=getattr(facts, "hi", None),
        container_len=getattr(facts, "container_len", None),
        container_len_ne=set(getattr(facts, "container_len_ne", None) or ())))


def _integral(m: Any) -> bool:
    return isinstance(m, (int, float)) and m == m and m not in (float("inf"), float("-inf")) and m == int(m)


def _cap_length(spec: dict, length: Any, rules: set) -> dict:
    if length.is_bottom() or (spec.get("max_len") is not None and length.lo is not None and length.lo > spec["max_len"]):
        raise UnsatisfiableConstraints(f"no {spec['kind']} has the length the guards ask for")
    if length.hi is not None and (spec.get("max_len") is None or length.hi < spec["max_len"]):
        rules.add("length")
        return dict(spec, max_len=length.hi)
    return spec


def _json_self(value: Any) -> bool:
    """True when ``value`` represents itself in ``find_inputs`` JSON without a ``$tag``."""
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is float:
        return math.isfinite(value)
    return type(value) is list and all(_json_self(v) for v in value)


def _contains_demands(guards: list, name: str, *, immutable: bool = False) -> tuple[list, list]:
    """``(positive, negative)`` literal ``c in name`` / ``c not in name`` conjuncts read directly off ``guards``
    (in guard order, unhashable literals such as ``[1, 2] in x`` included)."""
    pure = {"len", "range", "isinstance"}
    pos: list = []
    neg: list = []

    def conjuncts(node: ast.AST):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            for part in node.values:
                yield from conjuncts(part)
        else:
            yield node

    def atom(node: ast.AST, polarity: bool = True):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return atom(node.operand, not polarity)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            right = node.comparators[0]
            if isinstance(right, ast.Name) and right.id == name:
                if any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node.left)):
                    return None
                try:
                    lit = ast.literal_eval(node.left)
                except Exception:  # noqa: BLE001
                    return None
                is_in = isinstance(node.ops[0], ast.In) == polarity
                return lit, is_in
        return None

    trees = [ast.parse(g, mode="eval").body if isinstance(g, str) else g for g in guards]
    rebinds = any(isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name) and n.target.id == name
                  for t in trees for n in ast.walk(t))
    if rebinds:
        return [], []
    if not immutable:
        has_call = any(isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in pure)
                       or isinstance(n, (ast.Await, ast.Yield, ast.YieldFrom))
                       for t in trees for n in ast.walk(t))
        if has_call:
            return [], []
    for tree in trees:
        for conj in conjuncts(tree):
            got = atom(conj)
            if got is not None:
                lit, is_in = got
                (pos if is_in else neg).append(lit)
    return pos, neg


def _ordered_contains(dom: Any, pos: list) -> list:
    """Required items from ``pos`` (guard order) followed by any extra items in ``dom.value.contains`` sorted
    deterministically by :func:`canonical_key`."""
    out = list(pos)
    extra = sorted(dom.value.contains, key=lambda x: repr(canonical_key(x)))
    for item in extra:
        if not any(_same_member(item, w) for w in out):
            out.append(item)
    return out


def _coerce_list_member(elem_spec: dict, item: Any) -> Any:
    """A value in ``elem_spec``'s domain that Python ``in`` matches against ``item``; raises
    :class:`UnsatisfiableConstraints` when none exists."""
    from tharkun.typeinfer.core.kinds import strategy_constraints

    ec = strategy_constraints(elem_spec, False)
    if contains(ec, item):
        return item
    kind = elem_spec["kind"]
    if kind == "one_of":
        for arm in elem_spec["of"]:
            try:
                return _coerce_list_member(arm, item)
            except UnsatisfiableConstraints:
                continue
        raise UnsatisfiableConstraints(f"no arm of list element union admits {item!r}")
    if kind == "literal":
        for v in elem_spec["values"]:
            if v == item:
                return v
        raise UnsatisfiableConstraints(f"list literal element domain excludes {item!r}")
    if kind == "int" and _integral(item):
        cand = int(item)
        if contains(ec, cand):
            return cand
    elif kind == "float" and isinstance(item, (int, bool)):
        cand = float(item)
        if contains(ec, cand):
            return cand
    elif kind == "bool" and isinstance(item, (int, float)) and not isinstance(item, bool) and item in (0, 1):
        cand = bool(item)
        if contains(ec, cand):
            return cand
    elif kind == "complex" and isinstance(item, (int, float, bool)) and not (isinstance(item, float) and math.isnan(item)):
        cand = complex(item, 0.0)
        if contains(ec, cand):
            return cand
    raise UnsatisfiableConstraints(f"no {kind} element can match {item!r}")


def _narrow(spec: dict, dom: Any, rules: set, preds: list = (), name: str = "x",
            extra_constraints: dict | None = None) -> dict:
    """``spec`` cut to what the guard domain ``dom`` (``tharkun.solve.guard_domain.Domain``) admits; raises
    :class:`UnsatisfiableConstraints` when nothing is left.  The domain is an OVER-approximation of the values that
    take the path, so a cut never drops one of them."""
    kind, v = spec["kind"], dom.value
    if kind == "one_of":
        arms = []
        for arm in spec["of"]:
            mine: set = set()
            arm_extra: dict = {}
            try:
                arms.append(_narrow(arm, dom, mine, preds, name, arm_extra))
                rules |= mine
                if extra_constraints is not None and arm_extra:
                    extra_constraints.update(arm_extra)
            except UnsatisfiableConstraints:
                continue
        if not arms:
            raise UnsatisfiableConstraints("no arm of the union survives the guards")
        if len(arms) < len(spec["of"]):
            rules.add("arms")
        return spec if arms == spec["of"] else (arms[0] if len(arms) == 1 else dict(spec, of=arms))
    pos_mut, neg_mut = _contains_demands(list(preds), name, immutable=False) if preds else ([], [])
    if kind == "literal":
        kept = [x for x in spec["values"] if _admits(dom, x, (pos_mut, neg_mut))]
        if not kept:
            raise UnsatisfiableConstraints("no literal survives the guards")
        if len(kept) < len(spec["values"]):
            rules.add("members")
            return dict(spec, values=kept)
        return spec
    names = v.kinds.names
    if kind in _PY_TYPE and names is not None and _PY_TYPE[kind] not in names:
        raise UnsatisfiableConstraints(f"the guards admit no {kind}")
    if (pos_mut or neg_mut) and kind in ("none", "bool", "int", "float", "complex", "decimal", "fraction"):
        raise UnsatisfiableConstraints(f"membership test 'in {name}' admits no {kind}")
    if kind == "none":
        if None in v.neq:
            raise UnsatisfiableConstraints("the guards exclude None")
        return spec
    if kind == "bool":
        if v.bools is not None:
            if not v.bools:
                raise UnsatisfiableConstraints("the guards admit no bool")
            if len(v.bools) == 1:
                rules.add("members")
                return {"kind": "literal", "values": sorted(v.bools)}
        return spec
    if kind == "int":
        if v.i.is_bottom():
            raise UnsatisfiableConstraints("the guards admit no int")
        lo = max((b for b in (spec.get("min"), v.i.lo) if b is not None), default=None)
        hi = min((b for b in (spec.get("max"), v.i.hi) if b is not None), default=None)
        if dom.members is not None:
            ints = sorted({int(m) for m in dom.members if _integral(m) and (lo is None or m >= lo) and (hi is None or m <= hi)})
            if not ints:
                raise UnsatisfiableConstraints("no int is a member")
            if len(ints) <= DOMAIN_MAX_MEMBERS:
                rules.add("members")
                return {"kind": "literal", "values": ints}
        while lo is not None and lo in v.neq and (hi is None or lo < hi):
            lo += 1                                       # an excluded end is not a member (the loop ends: neq is finite)
        while hi is not None and hi in v.neq and (lo is None or lo < hi):
            hi -= 1
        if lo is not None and hi is not None and lo > hi:
            raise UnsatisfiableConstraints(f"empty int range [{lo}, {hi}]")
        out = dict(spec)
        for key, bound in (("min", lo), ("max", hi)):
            if bound is not None:
                out[key] = bound
        if out != spec:
            rules.add("bounds")
        return out
    if kind == "float":
        if v.f.is_bottom():
            raise UnsatisfiableConstraints("the guards admit no float")
        if dom.members is not None:
            nums = {float(m) for m in dom.members if isinstance(m, (int, float)) and m == m and abs(m) != float("inf")}
            listed = sorted(nums - {0.0}) + ([-0.0, 0.0] if 0.0 in nums else [])   # 0 == 0.0 == -0.0: both are admitted
            if not listed:
                raise UnsatisfiableConstraints("no float is a member")
            if len(listed) <= DOMAIN_MAX_MEMBERS:
                rules.add("members")
                return {"kind": "literal", "values": listed}
        return spec
    if kind == "str":
        if v.s.is_bottom():
            raise UnsatisfiableConstraints("the guards admit no str")
        pos_str, neg_str = _contains_demands(list(preds), name, immutable=True) if preds else ([], [])
        req = _ordered_contains(dom, pos_str)
        lacks = list(v.lacks) + [x for x in neg_str if not any(_same_member(x, w) for w in v.lacks)]
        if not all(type(c) is str for c in req) or not all(type(c) is str for c in lacks):
            raise UnsatisfiableConstraints("the guards admit no str")
        if "" in lacks:
            raise UnsatisfiableConstraints("no str satisfies '' not in x")
        if dom.members is not None:
            strs = sorted(m for m in dom.members if type(m) is str and _admits(dom, m, (pos_str, neg_str))
                          and len(m) <= spec.get("max_len", len(m)))
            if not strs:
                raise UnsatisfiableConstraints("no str is a member")
            if len(strs) <= DOMAIN_MAX_MEMBERS:
                rules.add("members")
                return {"kind": "literal", "values": strs}
        out = _cap_length(spec, v.length, rules)
        prior_subs = list(out.get("contains") or ())
        subs = _str_contains({"contains": prior_subs + req})
        if any(lack in sub for lack in lacks for sub in subs):
            raise UnsatisfiableConstraints("guards both require and forbid a substring")
        if subs:
            hi = out.get("max_len")
            merged = _shortest_superstrings(subs, hi)
            if not merged:
                raise UnsatisfiableConstraints(f"no str within max_len={hi} contains {subs!r}")
            min_req = min(len(m) for m in merged)
            if v.length.hi is not None and min_req > v.length.hi:
                raise UnsatisfiableConstraints(f"no str within guard length contains {subs!r}")
            if subs != prior_subs:
                out = dict(out, contains=subs)
            rules.add("contains")
        return out
    if kind == "list":
        out = _cap_length(spec, v.length, rules)
        req_raw = list(out.get("contains") or ()) + _ordered_contains(dom, pos_mut)
        lacks = list(v.lacks) + neg_mut
        if req_raw:
            coerced: list = []
            for item in req_raw:
                witness = _coerce_list_member(out["of"], item)
                if any(witness == lack or item == lack for lack in lacks):
                    raise UnsatisfiableConstraints(f"guards both require and forbid {item!r} in list")
                if not any(_same_member(witness, w) for w in coerced):
                    coerced.append(witness)
            if out.get("max_len") is not None and len(coerced) > out["max_len"]:
                raise UnsatisfiableConstraints(
                    f"guards require {len(coerced)} distinct elements, > max_len ({out['max_len']})")
            if coerced:
                rules.add("contains")
                if all(_json_self(x) for x in coerced):
                    out = dict(out, contains=coerced)
                elif extra_constraints is not None:
                    extra_constraints["contains"] = coerced
        return out
    if kind in ("set", "frozenset", "dict"):
        for item in pos_mut + neg_mut:
            try:
                hash(item)
            except TypeError:
                raise UnsatisfiableConstraints(f"unhashable {type(item).__name__} cannot be checked with 'in' on {kind}") from None
        return _cap_length(spec, v.length, rules)
    if kind in ("bytes", "bytearray") or (kind == "tuple" and "items" not in spec):
        if kind in ("bytes",) and v.by.is_bottom():
            raise UnsatisfiableConstraints("the guards admit no bytes")
        if kind in ("bytes", "bytearray"):
            for item in pos_mut + neg_mut:
                if not (isinstance(item, (bytes, bytearray)) or (isinstance(item, int) and 0 <= item <= 255)):
                    raise UnsatisfiableConstraints(f"the guards admit no {kind}")
        return _cap_length(spec, v.length, rules)
    return spec


def _admits(dom: Any, value: Any, extra_contains: tuple = ((), ())) -> bool:
    if dom is not None and not dom.admits(value):
        return False
    pos, neg = extra_contains
    if pos or neg:
        try:
            if any(c not in value for c in pos) or any(c in value for c in neg):
                return False
        except TypeError:
            return False
    return True


def _fits(leaf: dict, v: Any) -> bool:
    """Is the literal ``v`` a value of the scalar kind ``leaf`` (before the guards)?"""
    kind = leaf["kind"]
    if kind == "int":
        return type(v) is int and (leaf.get("min") is None or v >= leaf["min"]) and (leaf.get("max") is None or v <= leaf["max"])
    if kind == "float":
        return type(v) is float and (not leaf.get("finite") or v - v == 0.0)
    if kind in ("str", "bytes"):
        ok = type(v) is (str if kind == "str" else bytes) and len(v) <= min(leaf.get("max_len", 32), 32)
        if ok and kind == "str" and "contains" in leaf:
            ok = all(sub in v for sub in leaf["contains"])
        if ok and kind == "str" and "regex" in leaf:
            return getattr(re.compile(leaf["regex"], _regex_flags(leaf.get("flags"))), leaf.get("how", "fullmatch"))(v) is not None
        return ok
    return False


def _prefer_literals(literals: Iterable[Any], top: dict, dom: Any, extra_contains: tuple = ((), ())):
    """A ``strategy_constraints`` ``prefer`` callback: the caller's literals of the leaf's type that fit the leaf (and,
    at the top, the guards), in the caller's order, distinct, at most ``DOMAIN_LITERALS`` of a type."""
    seen: dict = {"int": [], "float": [], "str": [], "bytes": []}
    for lit in literals:
        for of in seen:
            if (type(lit) is {"int": int, "float": float, "str": str, "bytes": bytes}[of]) and len(seen[of]) < DOMAIN_LITERALS \
                    and not any(canonical_key(lit) == canonical_key(w) for w in seen[of]):
                seen[of].append(lit)
    tops = [top] + list(top.get("of") or ()) if top["kind"] == "one_of" else [top]

    def prefer(leaf: dict, of: str, convert, hashable: bool) -> list:
        at_top = any(leaf is t for t in tops)
        return [v for v in map(convert, seen.get(of, ()))
                if _fits(leaf, v) and (not at_top or _admits(dom, v, extra_contains))]

    return prefer


_RE_HOWS = ("fullmatch", "match", "search")
_RE_TRANSFORMS = ("sub", "subn", "split", "findall", "finditer")


def _flag_spec(node: Any) -> Any:
    """The ``flags`` of a ``re`` call argument: an int, or a list of flag names (``re.I | re.UNICODE``); ``None`` when
    the node is not such an expression."""
    names = {"I": "IGNORECASE", "S": "DOTALL", "M": "MULTILINE", "A": "ASCII", "U": "UNICODE", "X": "VERBOSE"}
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "re":
        name = names.get(node.attr, node.attr)
        return [name] if name in REGEX_FLAGS else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        a, b = _flag_spec(node.left), _flag_spec(node.right)
        return a + b if isinstance(a, list) and isinstance(b, list) else None
    return None


def _regex_demands(guards: list, name: str, namespace: dict) -> tuple:
    """``(specs, declined)``: the regex each guard demands of ``name`` (``re.match(P, name)``, ``re.compile(P).search(name)``,
    ``NAME.fullmatch(name)`` with ``NAME`` in ``namespace``: a top-level conjunct, or its ``is not None``), each
    ``{"regex", "how", "flags"?}``, and the NAMES of what was seen and not used: a negated match
    (``regex:negated``), a pattern that is not a literal or a ``namespace`` entry (``regex:unresolved:<text>``), a
    match on something derived from ``name`` (``regex:operand_not_the_input``), a pattern that does not compile, a
    ``sub`` / ``split`` / ``findall`` of the input (``regex:transform:<name>``: the pre-image of a normalizer is not a
    pattern of the input)."""
    specs: list = []
    declined: list = []

    def conjuncts(test):
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            for part in test.values:
                yield from conjuncts(part)
        else:
            yield test

    def call_of(test, positive=True):
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return call_of(test.operand, not positive)
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.comparators[0], ast.Constant) \
                and test.comparators[0].value is None and isinstance(test.ops[0], (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
            return call_of(test.left, positive if isinstance(test.ops[0], (ast.IsNot, ast.NotEq)) else not positive)
        if isinstance(test, ast.Call) and isinstance(test.func, ast.Name) and test.func.id == "bool" and len(test.args) == 1:
            return call_of(test.args[0], positive)
        return (test, positive) if isinstance(test, ast.Call) else (None, positive)

    def resolve(node):
        """``(pattern, flags)`` of a pattern expression, or ``None``."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value, None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "compile" \
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            return node.args[0].value, (_flag_spec(node.args[1]) if len(node.args) > 1 else None)
        ref = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        entry = namespace.get(ref) if ref is not None else None
        if isinstance(entry, str):
            return entry, None
        if isinstance(entry, dict) and isinstance(entry.get("regex", entry.get("pattern")), str):
            return entry.get("regex", entry.get("pattern")), entry.get("flags")
        return None

    def mentions(node):
        return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))

    for text in guards:
        tree = ast.parse(text, mode="eval").body if isinstance(text, str) else text
        for conjunct in conjuncts(tree):
            call, positive = call_of(conjunct)
            if call is None or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr in _RE_TRANSFORMS and any(mentions(a) for a in call.args):
                # a regex TRANSFORM of the input (a normalizer): its pre-image is not a pattern of the input
                declined.append(f"regex:transform:{call.func.attr}")
                continue
            if call.func.attr not in _RE_HOWS:
                continue
            recv = call.func.value
            if isinstance(recv, ast.Name) and recv.id == "re":
                if len(call.args) < 2:
                    continue
                pat, subject, flag_node = call.args[0], call.args[1], (call.args[2] if len(call.args) > 2 else None)
            else:
                if not call.args:
                    continue
                pat, subject, flag_node = recv, call.args[0], None
            if not mentions(subject):
                continue
            if not (isinstance(subject, ast.Name) and subject.id == name):
                declined.append("regex:operand_not_the_input")
                continue
            if not positive:
                declined.append("regex:negated")
                continue
            got = resolve(pat)
            if got is None:
                declined.append("regex:unresolved:" + ast.unparse(pat)[:40])
                continue
            pattern, flags = got
            flags = _flag_spec(flag_node) if flag_node is not None else flags
            try:
                re.compile(pattern, _regex_flags(flags))
            except (re.error, UnsatisfiableConstraints):
                declined.append("regex:does_not_compile")
                continue
            specs.append({"regex": pattern, "how": call.func.attr, **({"flags": flags} if flags else {})})
    return specs, declined


def _regex_spec(spec: Any) -> dict:
    if isinstance(spec, str):
        return {"regex": spec, "how": "fullmatch"}
    if isinstance(spec, dict) and isinstance(spec.get("regex"), str):
        return {"regex": spec["regex"], "how": spec.get("how", "fullmatch"),
                **({"flags": spec["flags"]} if spec.get("flags") else {})}
    raise UnsatisfiableConstraints(f"a regex pattern is a str or a dict with 'regex', got {spec!r}")


def _regex_hint(spec: dict, hint: dict, declined: list) -> dict:
    """``spec`` with the regex ``hint`` on its first str kind (the arm of a union too), when the strategies can sample
    the pattern's language within the row bounds (``typeinfer/core/kinds.py::ROW_MAX_TEXT``); else ``spec`` and a
    declined name.  The samples are the SAME code as every other regex draw (:func:`_regex_strategy`)."""
    from tharkun.typeinfer.core.kinds import ROW_MAX_TEXT

    if spec["kind"] == "one_of":
        arms = list(spec["of"])
        at = next((i for i, a in enumerate(arms) if a["kind"] == "str"), None)
        if at is None:
            declined.append("regex:kind_not_str")
            return spec
        arms[at] = _regex_hint(arms[at], hint, declined)
        return dict(spec, of=arms)
    if spec["kind"] != "str":
        declined.append("regex:kind_not_str")
        return spec
    flags = hint.get("flags")
    subs = list(spec.get("contains") or ())
    probe = {"type": "str", "regex": hint["regex"], "how": hint["how"],
             **({"flags": flags} if flags else {}),
             **({"contains": subs} if subs else {})}
    check = getattr(re.compile(hint["regex"], _regex_flags(flags)), hint["how"])
    # the rows draw within the row bound (the kind's constraints carry its ``length``): a pattern with no member that
    # short would make the rows' strategy refuse, so it is not a hint
    bounded = {**probe, "length": {"min": 0, "max": min(spec.get("max_len", ROW_MAX_TEXT), ROW_MAX_TEXT)}}
    try:
        members = examples(bounded, 3)
    except Exception as exc:  # noqa: BLE001 -- a construct the sampler refuses, a pattern with no short member
        declined.append("regex:not_sampleable:" + (str(exc) if isinstance(exc, UnsatisfiableConstraints) else type(exc).__name__)[:60])
        return spec
    if members and all(check(m) and all(s in m for s in subs) for m in members):
        return dict(spec, **{k: v for k, v in hint.items() if v})
    declined.append("regex:not_sampleable:members_do_not_match")
    return spec


def domain(kind: Any, *, name: str = "x", guards: Iterable[Any] = (), facts: Any = None, literals: Iterable[Any] = (),
           patterns: Iterable[Any] = (), namespace: Any = None, hints: Iterable[Any] = (),
           n: int = DOMAIN_EXAMPLES) -> Domain:
    """The generator domain of ONE parameter, from what a synthesis knows about it ([internal ref]; R1: strategies and synthesis
    have a different door from the search).

    ``kind``: a ``find_inputs`` kind (docs/INPUTS.md; a dict, or an annotation's text).  ``guards``: expressions over
    ``name`` that hold on every value that matters (the path conditions, ``bw_assume`` guards: source text or ``ast``
    nodes), read by ``tharkun.solve.guard_domain``.  ``facts``: contract facts about ``name`` (``tharkun.solve.fact_env.
    VarFacts``, or any object with ``eq`` / ``neq`` / ``lo`` / ``hi`` / ``container_len``).  ``literals``: the program's
    own constants, tried first when the kind and the guards admit them.  ``patterns``: regexes the value should match
    (a pattern text, or ``{"regex", "how", "flags"}``), and ``namespace``: the compiled patterns a guard may name
    (``{"URL_RE": {"regex": ..., "flags": ...}}``); a guard such as ``URL_RE.match(x) is not None`` or
    ``re.fullmatch(r"[a-z]+", x)`` is a regex of ``name``.  A regex is a GENERATOR, not a restriction: the str kind gets
    the pattern as a hint (``regex`` / ``how`` / ``flags``) that the strategy rows sample, through the same
    :func:`_regex_strategy` as every other regex draw, and what cannot be used is named in ``Domain.declined``.
    ``hints``: guard-shaped expressions that narrow NOTHING and are read for regexes only (a check the caller may
    relax, such as a derived backstop).
    Returns a :class:`Domain`.

    The caller owns what the guards MEAN: a domain is sound for the values that satisfy them, so a caller that wants
    every answer of a question must pass only guards every answer satisfies.  This function never runs the target and
    draws no random number: the same arguments give the same :class:`Domain` in every process, whatever its seed.
    Raises :class:`UnsatisfiableConstraints` for a kind it cannot draw and for an empty domain."""
    import copy

    from tharkun.solve.guard_domain import domain_of
    from tharkun.typeinfer.core.kinds import annotation_kind, drawable, strategy_constraints

    spec = copy.deepcopy(annotation_kind(kind) if isinstance(kind, str) else kind)
    if not isinstance(spec, dict) or not drawable(spec):
        raise UnsatisfiableConstraints(f"find_inputs cannot draw the kind {kind!r}")
    preds = list(guards)
    prior = _facts_value(facts)
    dom = domain_of(preds, name, prior=prior) if preds or prior is not None else None
    eq = getattr(facts, "eq", None)
    if dom is not None and eq is not None:                  # an exact member set is lost in the prior's interval hull
        members = frozenset(eq) if dom.members is None else frozenset(m for m in dom.members if any(m == e for e in eq))
        dom = dataclasses.replace(dom, members=members)
    rules: set = set()
    extra_constraints: dict = {}
    if dom is not None:
        narrowed = _narrow(spec, dom, rules, preds, name, extra_constraints)
    elif spec.get("kind") in ("str", "list") and spec.get("contains"):
        narrowed = spec
        rules.add("contains")
    else:
        narrowed = spec
    declined: list = []
    specs, declined = _regex_demands(preds + list(hints), name, dict(namespace or {}))
    specs = [_regex_spec(p) for p in patterns] + specs
    if specs and narrowed["kind"] != "literal":
        if any(sp != specs[0] for sp in specs[1:]):             # one pattern is sampled; the rest are named
            declined += [f"regex:conjunction:{sp['regex'][:30]}" for sp in specs[1:] if sp != specs[0]]
        hinted = _regex_hint(narrowed, specs[0], declined)
        if hinted != narrowed:
            rules.add("regex")
        narrowed = hinted
    extra_check = _contains_demands(preds, name, immutable=narrowed["kind"] == "str") if preds else ([], [])
    prefer = _prefer_literals(list(literals), narrowed, dom, extra_check)
    constraints = strategy_constraints(narrowed, False, prefer)
    if extra_constraints:
        if constraints["type"] == "list":
            constraints = dict(constraints, **extra_constraints)
        elif constraints["type"] == "one_of":
            constraints = dict(constraints, options=[
                dict(o, **extra_constraints) if o.get("type") == "list" and "contains" not in o else o
                for o in constraints["options"]])
    if dom is not None and narrowed["kind"] == "float" and "value_range" not in constraints:
        lo, hi = dom.value.f.lo, dom.value.f.hi
        if lo != -float("inf") or hi != float("inf"):
            finite = [b for b in (lo, hi) if abs(b) != float("inf")]
            if len(finite) == 2:
                rules.add("bounds")
                constraints = dict(constraints, value_range=[lo, hi])
    if constraints.get("prefer") or any(c.get("prefer") for c in _children(constraints)):
        rules.add("prefer")
    drawn = examples(constraints, max(1, n) * 2)
    values = tuple([v for v in drawn if _admits(dom, v, extra_check)][:n])
    ordered = tuple(r for r in DOMAIN_RULES if r in rules) or ("type",)
    return Domain(kind=narrowed, constraints=constraints, values=values, rules=ordered, narrowed=narrowed != spec,
                  declined=tuple(dict.fromkeys(declined)))


def _children(constraints: dict) -> list:
    out = [constraints[k] for k in _CHILDREN_ONE if isinstance(constraints.get(k), dict)]
    return out + [c for k in ("items", "options") for c in (constraints.get(k) or ()) if isinstance(c, dict)]
