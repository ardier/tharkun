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
      "type": "int" | "float" | "str" | "bytes" | "list" | "tuple" | "dict",

      # int only
      "bitmask_mask": <int>,          # only bits set in mask may be set
      "shift_direction": "left" | "right",
      "shift_bits": <int>,

      # int / float
      "value_range": [min, max],      # inclusive; None on a side means unbounded
      "values": [v1, v2, ...],        # membership; dominates every other constraint

      # str only
      "regex": "<pattern>",           # sample the language rather than filter for it
      "fullmatch": <bool>,            # default True

      # str / bytes / list / tuple / dict
      "length": {"min": a, "max": b},

      # list / tuple
      "element_constraints": {...},   # recursive

      # dict
      "key_constraints": {...},
      "value_constraints": {...},
    }

Any contradiction, empty domain, or unusable field raises
:class:`UnsatisfiableConstraints` rather than returning a strategy that cannot produce
anything — an empty domain is a fact worth surfacing, not a search to run.

Lifted from the eregion codegen project's ``hypothesis_enum`` and reimplemented here;
tharkun does not depend on eregion.
"""

from __future__ import annotations

from typing import Any

# A guard's literals are usually small; these bound the search when nothing says otherwise.
INT_DEFAULT_MIN = -(2 ** 31)
INT_DEFAULT_MAX = 2 ** 31 - 1
FLOAT_DEFAULT_MIN = float(-(2 ** 31))
FLOAT_DEFAULT_MAX = float(2 ** 31 - 1)

_VALID_TYPES = frozenset({"int", "float", "str", "bytes", "list", "tuple", "dict"})


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
    normalized = type_name.removeprefix("typing.")
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
    if normalized in {"list", "List"}:
        return st.lists(st.integers(), max_size=10)
    if normalized in {"dict", "Dict", "mapping", "Mapping"}:
        return st.dictionaries(st.text(max_size=5), st.integers(), max_size=5)
    if normalized in {"set", "Set"}:
        return st.frozensets(st.integers(), max_size=10).map(set)
    if normalized in {"tuple", "Tuple"}:
        return st.tuples(st.integers(), st.integers())
    if normalized == "object":
        return st.fixed_dictionaries({})
    if normalized in {"Any", "any"}:
        return st.one_of(st.integers(), st.text(max_size=20), st.booleans())
    return None


def _is_int(v: Any) -> bool:
    """True for a real int. ``bool`` is an ``int`` subclass and is never one here."""
    return isinstance(v, int) and not isinstance(v, bool)


def make_strategy(constraints: dict) -> Any:
    """A Hypothesis strategy for *constraints*, as narrow as the facts allow."""
    try:
        import hypothesis.strategies as st  # noqa: F401
    except ImportError as exc:                        # pragma: no cover - env dependent
        raise UnsatisfiableConstraints("hypothesis is not installed") from exc

    if not isinstance(constraints, dict):
        raise UnsatisfiableConstraints(f"constraints must be a dict, got {type(constraints).__name__}")
    kind = constraints.get("type")
    if kind not in _VALID_TYPES:
        raise UnsatisfiableConstraints(f"unknown or missing type {kind!r}")

    if kind == "int":
        return _int_strategy(constraints)
    if kind == "float":
        return _float_strategy(constraints)
    if kind == "str":
        return _str_strategy(constraints)
    if kind == "bytes":
        return _bytes_strategy(constraints)
    if kind in ("list", "tuple"):
        return _sequence_strategy(constraints, as_tuple=kind == "tuple")
    return _dict_strategy(constraints)


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
    lo = lo_default if lo is None else lo
    hi = hi_default if hi is None else hi
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

    return st.integers(min_value=lo, max_value=hi)


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

    lo, hi = _range_bounds(constraints, FLOAT_DEFAULT_MIN, FLOAT_DEFAULT_MAX, _num)
    # Bound an open side rather than leave it open: an unbounded float strategy spends
    # most of its draws in magnitudes no guard cares about. NaN and infinity are excluded
    # because they satisfy no ordered comparison and so are never a useful answer to one.
    return st.floats(min_value=float(lo), max_value=float(hi),
                     allow_nan=False, allow_infinity=False)


def _raise(msg: str):
    raise UnsatisfiableConstraints(msg)


def _str_strategy(constraints: dict) -> Any:
    import hypothesis.strategies as st

    if "values" in constraints:
        values = constraints["values"]
        if not isinstance(values, (list, tuple)) or not values:
            raise UnsatisfiableConstraints("str 'values' must be a non-empty sequence")
        return st.sampled_from([v for v in values if isinstance(v, str)] or
                               _raise("str 'values' contains no string"))

    # A regex guard is the case where filtering is hopeless: no amount of sampling
    # st.text() will produce "32hex:32hex". Sample the pattern's LANGUAGE instead.
    # fullmatch defaults True because a captured pattern is normally ^…$-anchored;
    # a caller matching only a prefix passes fullmatch=False.
    regex = constraints.get("regex")
    if regex is not None:
        if not isinstance(regex, str):
            raise UnsatisfiableConstraints(f"'regex' must be a pattern string, got {type(regex).__name__}")
        return st.from_regex(regex, fullmatch=bool(constraints.get("fullmatch", True)))

    lo, hi = _length_bounds(constraints, default_max=64)
    return st.text(min_size=lo, max_size=hi)


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


def _sequence_strategy(constraints: dict, *, as_tuple: bool) -> Any:
    import hypothesis.strategies as st

    elem = constraints.get("element_constraints")
    element_strategy = make_strategy(elem) if elem else st.integers()
    lo, hi = _length_bounds(constraints, default_max=16)
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
