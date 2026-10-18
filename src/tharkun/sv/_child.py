"""Child-interpreter side of ``tharkun.sv``: the ``_sv_verifier`` model and the run boot.

This file is executed as a script in a FRESH interpreter (``python _child.py
REQUEST.json``) so that one run of the task cannot leak state into the next.
It imports nothing from tharkun.  The request names the program, the draw
vector to replay and the output path; the child writes one JSON record:

* ``status``: ``clean`` | ``exit`` | ``raised``;
* ``exc``, ``mro``, ``msg``, ``site`` when it raised;
* ``draws``: every primitive nondeterministic value consumed, in call order;
* ``flags``: nondeterminism the model does not control (``argv``, ``time``,
  ``stdin``, ``unseeded_random_unmodelled``), which forbids a TRUE verdict.
  ``sys.stdin`` is NEVER replaced with a stand-in object here ([internal ref] owner
  ruling) -- ``stdin`` is set by a static, conservative source scan
  (``environment.references_stdin``) instead, so the real object's identity,
  type and ``.mode`` are exactly what the target program would see outside
  Tharkun. A FALSE witness may rely only on ``_sv_verifier`` draws and
  ``argv``; stdin/file CONTENT is never part of the witness vector (a
  declared/required file's REAL bytes are staged from disk via
  ``environment.install_files``, driven purely by the task's own
  ``input_files``/``required_files`` -- never by anything a witness carries).
* ``ranges``: ``[position, a, b]`` (float.hex, in call order) of every ``nondet_float_range`` call, positions in ``draws`` --
  what a producer's draw vector is pre-checked against before it costs a run (``engine.vector_admissible``);
* ``invalid_draw``: a draw fell outside the domain the helper's CODE produces (``nondet_float_range`` a value
  ``random.uniform(a, b)`` cannot return; ``nondet_float`` anything but ``inf``, ``-inf``, ``nan``, ``-0.0``;
  ``nondet_str`` with a character outside ``string.printable``) -- ``engine.py::Outcome.violated`` refuses to call this run a
  violation no matter what exception surfaced, matching ``eregion_sv.replay.violates``.
  The model REFUSES such a draw, it never clamps or repairs it, because the grading
  harness refuses it (``eregion-sv/src/eregion_sv/shim/_sv_verifier.py``): a witness
  holding one is NOT CONFIRMED there.  docs/SV.md "Draw domains" lists every kind against
  the harness's rule; ``tests/test_sv_draw_domains.py`` holds them equal.

Draw kinds (the ``sv-py-witness/1`` vector vocabulary):

* ``bool``, ``int``, ``float``, ``str`` -- one ``_sv_verifier.nondet_*`` call
  (``nondet_float_range(a, b)`` is a ``float`` draw REJECTED unless ``random.uniform(a, b)`` can return it --
  ``invalid_draw`` (below) is set and the call raises ``ValueError``, matching the grading harness's own model
  (``eregion_sv/shim/_sv_verifier.py``); ``nondet_float()`` is REJECTED unless it is one of the four values the
  helper's code produces on CPython 3.14 ([internal ref]: docs/SV.md "Draw domains: both readings");
  a run that ever sets it is never a violation, see ``violated`` below;
  ``nondet_str()`` is a ``str`` draw whose every character is in ``string.printable``,
  REFUSED the same way ([internal ref]);
  ``nondet_list``/``nondet_dict`` keep the reference loop on ``nondet_bool``);
* ``rindex`` with ``n`` -- an UNSEEDED ``random.choice``/``randrange``/``randint``
  picks index ``value`` of ``n`` possibilities;
* ``rfloat`` -- an unseeded ``random.random()`` (``uniform(a, b)`` scales it).

A float value is stored as its exact ``float.hex()`` spelling.
"""

import builtins
import importlib.machinery
import json
import io
_RESULT_FILE_IO = io.FileIO
import math
import opcode
import os
import random as _random
import string
import sys
import time as _time
import traceback
import types

# Capture interpreter identities before any target code can replace them.
_GETFRAME = sys._getframe
_IMPORT_NAME = opcode.opmap["IMPORT_NAME"]

#: the generator's real ``getstate`` (``_patch_random`` replaces the module-level one by a flagging model, [internal ref]): the
#: stationary probe's snapshot reads the generator state through this one
real_random_getstate = _random.getstate

#: The builtin classes as they are BEFORE any program code runs ([internal ref]).  A property class (``IndexError``,
#: ``AssertionError``, ...) is the builtin of that name, never a class the program defines under the same name:
#: ``class IndexError(Exception): pass; raise IndexError()`` is not an ``IndexError``.
_BUILTIN_CLASSES = {name: obj for name, obj in vars(builtins).items() if isinstance(obj, type)}
_RESOURCE_ERRORS = (RecursionError, MemoryError)
_TYPE, _TYPE_MRO = type, type.__dict__["__mro__"].__get__


def exception_mro(exc) -> list:
    """The names of the classes in ``type(exc).__mro__`` that ARE builtin classes -- the very object, not a class that
    merely carries the name or sets ``__module__ = "builtins"`` -- which is what a property class is matched
    against (``Outcome.violated``, ``DriveOutcome.violated``, the stationary proof, the grader's
    ``eregion_sv.replay.violates``).  A subclass of a builtin (``class E(IndexError)``) has the builtin in its MRO and
    counts; the program's own ``class IndexError(Exception)`` does not."""
    return [c.__name__ for c in type(exc).__mro__ if _BUILTIN_CLASSES.get(c.__name__) is c]


# MG 2026-09-29 (P0, both bugs real): this file must import nothing from
# tharkun, even inside a try (tests/test_layering.py::
# test_the_sv_child_imports_no_tharkun -- a static, eager_only=False AST scan,
# so a conditionally-executed ``tharkun.solve...`` import is caught just the
# same as an unconditional one, whether or not it ever actually runs). But the
# bare ``import environment`` below still needs ``solve/`` on sys.path to
# resolve at all, and leaving that insert in place after the import used to
# leak into the IMPORTING PROCESS for its whole remaining lifetime whenever
# this module is imported NORMALLY (``from tharkun.sv import _child``,
# nondet_harness.py, in-process) rather than run as its own disposable
# subprocess script -- shadowing the stdlib ``types`` module and the
# third-party ``attrs`` package with this repo's own
# ``tharkun/solve/types.py``/``attrs.py``, breaking any subprocess that
# inherits that sys.path (e.g. Avengers' executor workers). Confirmed:
# ``pytest -p no:xdist tests/test_nondet_harness.py
# tests/test_avengers_directed.py`` reproduced 4/4 failures before this fix.
# Fix: self-cleaning -- the insert lives only for the duration of the import
# statement itself; ``install_files``/``references_stdin`` are already bound
# to local names by then, so nothing needs sys.path to stay mutated
# afterward, in EITHER calling context.
_SOLVE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "solve")
sys.path.insert(0, _SOLVE)
try:
    from environment import (install_files, references_stack, references_stdin, refuse_threads, thread_starters,
                             THREAD_STARTERS as _THREAD_STARTERS, confined)
    from extern import native_import_reason
finally:
    sys.path.remove(_SOLVE)
    # a program that says ``import environment`` must get ITS module (or ModuleNotFoundError), never this one ([internal ref])
    sys.modules.pop("environment", None)
    sys.modules.pop("extern", None)

_DEFAULTS = {"bool": False, "int": 0, "float": 0.0, "str": "", "rindex": 0, "rfloat": 0.0, "argv": [], "argv_full": []}
#: the characters a ``nondet_str`` draw may hold: string.printable, exactly the grading harness's rule
_PRINTABLE = frozenset(string.printable)

# --- the draw domains of the float helpers ([internal ref]) -------------------------------------------------------------
# The sv-benchmarks helper (`_sv_verifier.py`, MR !1752) is
#     def nondet_float():
#         if nondet_bool(): return random.choice([inf, -inf, nan, -0.0])
#         return random.uniform(-sys.float_info.max, sys.float_info.max)
#     def nondet_float_range(a, b): return random.uniform(a, b)
# and `random.uniform(a, b)` is `a + (b - a) * self.random()` with `random()` = k * 2**-53, 0 <= k < 2**53.  Its
# documentation reads as the closed interval [a, b] (any float), a LARGER set than the code produces.  The rule (owner,
# 2026-10-06): a verdict TRUE is proven over the larger reading (the absint model, which keeps the closed interval AND the
# formula), a verdict FALSE is confirmed only with a witness value the CODE can produce, so it holds under both readings.
# `uniform(-max, max)` overflows (`b - a` is `inf`): the second branch of `nondet_float` returns `inf` (k > 0) or `nan`
# (k == 0), never a finite float, on every CPython from 3.8 to 3.14.
_R53 = 2.0 ** -53
_NK = 1 << 53
_INF_ = float("inf")


def _same_float(x, y):
    """Bit-level equality of two floats as far as a draw is concerned: ``nan`` equals ``nan``, ``-0.0`` differs from ``0.0``."""
    if x != x or y != y:
        return x != x and y != y
    return x == y and (x != 0.0 or math.copysign(1.0, x) == math.copysign(1.0, y))


def _first_reaching(f, nk, v, nondecreasing):
    """The first ``k`` in ``0 .. nk - 1`` with ``f(k) >= v`` (``<= v`` when not ``nondecreasing``) for a MONOTONE ``f`` known to reach
    ``v`` somewhere: a binary search."""
    lo, hi = 0, nk - 1
    reaches = (lambda k: f(k) >= v) if nondecreasing else (lambda k: f(k) <= v)
    while lo < hi:
        mid = (lo + hi) // 2
        if reaches(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _monotone_image_contains(f, nk, v, nondecreasing):
    """Is ``v`` among ``f(0) .. f(nk - 1)`` for a MONOTONE ``f`` (``nondecreasing`` says which way)?  Binary search for the
    first ``k`` that reaches ``v``, then an exact test of it and its successor (``-0.0`` and ``0.0`` compare equal)."""
    if v != v:
        return False
    first, last = f(0), f(nk - 1)
    if nondecreasing:
        if v < first or v > last:
            return False
    elif v > first or v < last:
        return False
    lo = _first_reaching(f, nk, v, nondecreasing)
    return _same_float(f(lo), v) or (lo + 1 < nk and _same_float(f(lo + 1), v))


def float_range_producible(a, b, v):
    """Can ``random.uniform(a, b)`` (``a + (b - a) * random()``, ``random()`` = ``k * 2**-53``, ``0 <= k < 2**53``) return the
    float ``v``?  Exact, in IEEE double arithmetic: the two roundings are monotone in ``k``, so the value set is searched,
    not sampled.  A span that overflows (``b - a`` is ``inf``) or is ``nan`` gives ``nan`` for ``k == 0`` and ``+-inf`` or
    ``nan`` for every other ``k`` (``inf * 0.0``), so only those two are tried; a ``b - a`` or ``float(b - a)`` that cannot
    be computed (an int beyond the float range) is an ``OverflowError`` in the helper: no value.  ``a`` and ``b`` are plain
    ``int``/``float``/``bool`` here (the caller decides)."""
    try:
        span = b - a
        fspan = float(span)

        def f(k):
            return a + span * (k * _R53)

        if fspan != fspan or fspan in (_INF_, -_INF_) or fspan == 0.0:
            return _same_float(f(0), v) or _same_float(f(1), v)
        return _monotone_image_contains(f, _NK, v, fspan > 0.0)
    except OverflowError:
        return False


def snap_float_range(v, a, b):
    """The value ``random.uniform(a, b)`` can return that is closest to the PROPOSAL ``v`` from inside the domain: the first producible
    value at or past ``v`` (in the direction ``uniform`` runs), the first or last one when ``v`` is beyond them (a huge proposal
    becomes the LARGEST producible value, ``1 - 2**-53`` for ``(0.0, 1.0)``, not the unproducible ``b``), the first one for ``nan``.  A
    proposal generator (the Atheris worker, the concolic stub) logs this instead of the raw value, so what the program saw and what
    the vector records are the same producible value.  Bounds that are not plain numbers are refused without running their arithmetic."""
    if not ((type(a) is int or type(a) is float or type(a) is bool) and (type(b) is int or type(b) is float or type(b) is bool)):
        raise ValueError("nondet_float_range bounds are not plain numbers")
    try:
        span = b - a
        fspan = float(span)

        def f(k):
            return a + span * (k * _R53)

        if fspan != fspan or fspan in (_INF_, -_INF_) or fspan == 0.0:
            for candidate in (f(0), f(1)):
                if _same_float(candidate, v):
                    return v
            top = f(1)
            return top if top == top else f(0)
        first, last = f(0), f(_NK - 1)
        nondecreasing = fspan > 0.0
        if v != v:
            return first
        if (v <= first) if nondecreasing else (v >= first):
            return first
        if (v >= last) if nondecreasing else (v <= last):
            return last
        return f(_first_reaching(f, _NK, v, nondecreasing))
    except OverflowError:
        pass
    lo, hi = (a, b) if a <= b else (b, a)
    return lo if v != v else min(max(v, lo), hi)


def float_range_valid(a, b, v):
    """The ``nondet_float_range(a, b)`` draw rule: a value the helper's code can produce.  A bound whose type is not EXACTLY
    ``int``, ``float`` or ``bool`` (an ``int`` or ``float`` subclass, ``Fraction``, ``Decimal``, a numpy scalar, a user object) has
    NO value in the domain: the helper's ``a + (b - a) * random()`` would run that type's own arithmetic, which a predicate
    cannot decide, and the documented closed interval it used to fall back to contains values the code never returns
    (``nondet_float_range(MyInt(0), MyInt(1)); assert x < 1.0`` was a FALSE with ``x == 1.0``, natively safe).  The draw is
    refused (``invalid_draw``: never a violation), like an out-of-domain ``nondet_str``; a TRUE is still proven over the
    documented reading (absint), so only a FALSE is lost."""
    if (type(a) is int or type(a) is float or type(a) is bool) and (type(b) is int or type(b) is float or type(b) is bool):
        return float_range_producible(a, b, v)
    return False


def float_producible(v):
    """The ``nondet_float()`` draw rule on CPython 3.14: ``inf``, ``-inf``, ``nan`` and ``-0.0`` (the first branch lists the
    four; the second, ``random.uniform(-max, max)``, overflows to ``inf`` or ``nan``)."""
    return v != v or v in (_INF_, -_INF_) or (v == 0.0 and math.copysign(1.0, v) < 0.0)


#: the draw ``nondet_float()`` takes when a vector runs out: a value of its domain (a ``0.0`` would be refused)
_FLOAT_DEFAULT = -0.0


class _State:
    vector = []
    pos = 0
    log = []
    flags = set()
    seeded = False
    invalid_draw = False
    #: the names ``import`` asked the finders for and NO finder had (``watch_imports``, [internal ref])
    absent = set()
    #: ``[position in log, lo, hi]`` (float.hex) of every ``nondet_float_range`` call: the bounds a candidate
    #: vector is pre-checked against (sv/engine.py::vector_admissible); not part of a draw, so not of a witness
    ranges = []


def _enc(kind, v):
    return float(v).hex() if kind in ("float", "rfloat") else list(v) if kind in ("argv", "argv_full") else v


def _dec(kind, v):
    if kind in ("float", "rfloat"):
        return float.fromhex(v) if isinstance(v, str) else float(v)
    if kind == "bool":
        return bool(v)
    if kind in ("int", "rindex"):
        return int(v)
    if kind in ("argv", "argv_full"):
        if type(v) is not list or not all(type(x) is str for x in v):
            _State.flags.add("invalid_draw")
            return []
        return v
    return str(v)


def _site(depth):
    try:
        f = sys._getframe(depth)
        return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"
    except ValueError:
        return "?"


_UNSET = object()


def _draw(kind, n=None, depth=3, default=_UNSET, base=None):
    value = None
    if _State.pos < len(_State.vector):
        e = _State.vector[_State.pos]
        if e.get("kind") == kind:
            value = _dec(kind, e["value"])
    if kind not in ("argv", "argv_full") or value is not None:
        _State.pos += 1
    if value is None:
        value = _DEFAULTS[kind] if default is _UNSET else default
    if kind == "str" and any(c not in _PRINTABLE for c in value):
        # [internal ref]: REFUSED, never repaired -- the grading harness raises here and marks the run invalid
        # (eregion_sv/shim/_sv_verifier.py::_draw), so a witness holding this string is NOT CONFIRMED
        # there.  Refused before it is logged, exactly like the harness.
        _State.invalid_draw = True
        raise ValueError("nondet_str draw outside string.printable")
    if kind == "rindex":
        value = min(max(int(value), 0), max(n - 1, 0))
    if kind == "rfloat" and not (0.0 <= value < 1.0):
        value = 0.0
    rec = {"kind": kind, "value": _enc(kind, value), "site": _site(depth)}
    if n is not None:
        rec["n"] = n
    if base is not None:
        # [internal ref]: a randrange(start, stop, step) draw is ``range[index]`` = ``start + step * index``; a search that knows the
        # arithmetic can pick the index that makes the call return a value the program compares with
        # (``engine.py::Candidates``, ``rindex_hits``).  Not part of the draw (the witness carries kind, value, n only).
        rec["start"], rec["step"] = base
    _State.log.append(rec)
    return value


def initial_argv(program, source):
    """A full-list draw, or an explicitly marked legacy tail, at boot.

    No draw defaults to [program]. A source mentioning argv records that example
    without consuming a later per-call draw. Boot entries are unique and first;
    malformed vectors are refused before target code can catch anything.
    """
    vector = _State.vector
    if not isinstance(vector, list) or any(
            not isinstance(e, dict) or "kind" not in e or "value" not in e
            for e in vector):
        _State.flags.add("invalid_draw")
        return [program]
    boot = [i for i, e in enumerate(vector) if e["kind"] in ("argv", "argv_full")]
    if boot and boot != [0]:
        _State.flags.add("invalid_draw")
        return [program]
    mentions = b"argv" in source if isinstance(source, bytes) else "argv" in source
    if not boot and not mentions:
        return [program]
    _State.flags.add("argv")
    kind = vector[0]["kind"] if boot else "argv_full"
    if boot and (type(vector[0]["value"]) is not list or
                 not all(type(s) is str for s in vector[0]["value"])):
        _State.flags.add("invalid_draw")
        return [program]
    value = _draw(kind, default=[] if kind == "argv" else [program])
    return [program] + list(value) if kind == "argv" else list(value)


_MAX_RANGES = 4096


def _note_range(a, b):
    """Record the bounds of the ``nondet_float_range`` call whose draw was just logged.  Best effort and strictly
    passive: only plain ``int``/``float`` bounds are recorded, so no user object is ever compared or converted
    here (a task's own ``__lt__``/``__float__`` runs exactly as often as under the grading harness's model, which
    never looks at the bounds this way), and the list is bounded.  The bounds are kept in CALL order (``uniform(a, b)`` and
    ``uniform(b, a)`` are different value sets) and only when a float holds them exactly (an int beyond 2**53 does not)."""
    if (type(a) is not int and type(a) is not float) or (type(b) is not int and type(b) is not float) or len(_State.ranges) >= _MAX_RANGES:
        return
    if (type(a) is int and abs(a) > _NK) or (type(b) is int and abs(b) > _NK):
        return
    _State.ranges.append([len(_State.log) - 1, float(a).hex(), float(b).hex()])


# --- the _sv_verifier model -------------------------------------------------

def _make_sv_verifier():
    from collections.abc import Callable, Iterable, Mapping
    from typing import Any, Literal, Union, get_args, get_origin

    m = types.ModuleType("_sv_verifier")

    def missing(name):
        # Only the interpreter's named-from-import probe may miss metadata.
        # A target's explicit lookup (including IMPORT_FROM) still taints replay.
        # Star imports are unsupported: missing exports otherwise become bare
        # NameError, bypassing this hook (even if the target later catches it).
        protocol_probe = False
        if name == "__path__":
            try:
                frame = _GETFRAME(1)
                code, offset = frame.f_code.co_code, frame.f_lasti
                instruction = code[offset] if 0 <= offset < len(code) else None
                protocol_probe = instruction == _IMPORT_NAME
            except BaseException:
                pass  # An unrecognized caller is not an import-protocol exemption.
        if not protocol_probe:
            _State.flags.add("model_error")
        raise AttributeError(f"unsupported _sv_verifier attribute: {name}")

    m.__getattr__ = missing

    def nondet_bool():
        return _draw("bool")

    def nondet_int():
        return _draw("int")

    def nondet_float():
        # [internal ref]: REJECT a draw the helper's code cannot produce (inf, -inf, nan, -0.0 on CPython 3.14), exactly like the
        # grading harness's model; a TRUE is proven over every float (the larger reading), so a FALSE can never
        # rest on a finite one
        v = _draw("float", default=_FLOAT_DEFAULT)
        if not float_producible(v):
            _State.invalid_draw = True
            raise ValueError("nondet_float draw outside what the helper produces")
        return v

    def nondet_float_range(a, b):
        # [internal ref]: REJECT a draw outside its domain, exactly like the grading harness's own model
        # (eregion_sv/shim/_sv_verifier.py) -- a silent clamp let Tharkun's own replay confirm a FALSE whose witness
        # vector held a raw draw the harness's replay refuses (invalid_draw).  [internal ref]: the domain is what
        # `random.uniform(a, b)` can return (so `y == b` is out of the domain of `(0.0, 1.0)`), see float_range_producible.
        v = _draw("float")
        _note_range(a, b)
        if not float_range_valid(a, b, v):
            _State.invalid_draw = True
            raise ValueError("nondet_float_range draw outside range")
        return v

    def nondet_str():
        return _draw("str")

    def nondet_list(nondet_elem):
        out = []
        while nondet_bool():
            out.append(nondet_elem())
        return out

    def nondet_dict(nondet_key, nondet_value):
        d = {}
        while nondet_bool():
            d[nondet_key()] = nondet_value()
        return d

    def check_type(value, hint):
        """Independent implementation of the benchmark's runtime type check."""
        def accepts(item, annotation):
            if annotation is Any:
                return True
            if annotation is None or annotation is type(None):
                return item is None
            family = get_origin(annotation)
            if family is None:
                return isinstance(item, annotation)
            parameters = get_args(annotation)
            if family is Union or family is types.UnionType:
                return any(accepts(item, alternative) for alternative in parameters)
            if family is Literal:
                return any(item == constant and type(item) is type(constant) for constant in parameters)
            if not isinstance(item, family):
                return False
            if not parameters:
                return True
            if family is tuple:
                if len(parameters) == 2 and parameters[-1] is Ellipsis:
                    return all(accepts(element, parameters[0]) for element in item)
                if parameters == ((),):
                    return len(item) == 0
                if len(item) != len(parameters):
                    return False
                return all(accepts(element, expected) for element, expected in zip(item, parameters))
            if family is Callable:
                return True
            if issubclass(family, Mapping):
                key_type, item_type = parameters
                checked_entries = (accepts(key, key_type) and accepts(element, item_type)
                                   for key, element in item.items())
                return all(checked_entries)
            if issubclass(family, Iterable):
                element_type, = parameters
                return all(accepts(element, element_type) for element in item)
            raise NotImplementedError(f"check_type: unsupported type hint {annotation!r}")

        if not accepts(value, hint):
            raise TypeError(f"expected value of type {hint!r}, got {value!r}")

    for f in (nondet_bool, nondet_int, nondet_float, nondet_float_range, nondet_str,
              nondet_list, nondet_dict, check_type):
        setattr(m, f.__name__, f)
    return m


# --- unseeded randomness is a nondet source ----------------------------------

def _range_count(rng) -> int:
    """``len(rng)`` without the C ``ssize_t`` limit ([internal ref]): ``len(range(0, 2**100 + 1))`` raises ``OverflowError``,
    but ``random.randint(0, 2**100)`` works, and a draw over it is an ordinary ``rindex`` of that many values."""
    start, stop, step = rng.start, rng.stop, rng.step
    if step > 0:
        return max(0, (stop - start + step - 1) // step)
    return max(0, (start - stop - step - 1) // -step)


def _patch_random():
    real = {name: getattr(_random, name) for name in
            ("seed", "random", "uniform", "randint", "randrange", "choice", "shuffle", "sample",
             "choices", "getrandbits", "gauss", "normalvariate", "triangular", "betavariate",
             "expovariate", "gammavariate", "lognormvariate", "vonmisesvariate", "paretovariate",
             "weibullvariate", "randbytes", "binomialvariate", "getstate", "setstate")
            if hasattr(_random, name)}

    def seed(*a, **k):
        if (a[0] if a else k.get("a")) is None:
            # seed() / seed(None) seeds from the OS: still nondeterministic, and unmodelled
            _State.flags.add("unseeded_random_unmodelled")
        _State.seeded = True
        return real["seed"](*a, **k)

    def random():
        return real["random"]() if _State.seeded else _draw("rfloat")

    def uniform(a, b):
        return real["uniform"](a, b) if _State.seeded else a + (b - a) * _draw("rfloat")

    one = 1

    def randrange(start, stop=None, step=one):
        if _State.seeded:
            return real["randrange"](start, stop, step)
        try:
            rng = range(start) if stop is None else range(start, stop, step)
            if stop is None and step is not one:
                raise TypeError("Missing a non-None stop argument")
            n = _range_count(rng)
        except (TypeError, ValueError):
            raise                           # range() refuses what randrange() refuses, with the same exception class
        except BaseException:
            # an exception OUR model raises where CPython's randrange would not (the OverflowError of [internal ref]): the run
            # says nothing about the program, so it is never a FALSE witness and never certifies anything
            _State.flags.add("model_error")
            raise
        if n == 0:
            raise ValueError("empty range for randrange()")
        return rng[_draw("rindex", n=n, base=(rng.start, rng.step))]

    def randint(a, b):
        return real["randint"](a, b) if _State.seeded else randrange(a, b + 1)

    def choice(seq):
        if _State.seeded:
            return real["choice"](seq)
        if not len(seq):
            raise IndexError("Cannot choose from an empty sequence")
        return seq[_draw("rindex", n=len(seq))]

    def _unmodelled(name):
        def f(*a, **k):
            if not _State.seeded:
                _State.flags.add("unseeded_random_unmodelled")
            return real[name](*a, **k)
        return f

    def _state_unmodelled(name):
        # [internal ref]: the draws are not a generator state, so ``getstate``/``setstate`` have nothing to save or rewind: ``st =
        # random.getstate(); a = random.random(); random.setstate(st); b = random.random()`` gives two DIFFERENT draws
        # here and the same value natively.  Such a run is never a FALSE witness, never certifies determinism.
        #
        # A ``getstate()`` that no random draw preceded reads the OS-seeded state exactly as a native run does (the real
        # generator, never advanced by the model's draws): what the program sees is the same KIND of value, so a violation
        # that depends on a constant of the generator (``st[1][0] == 2147483648``, MT's first word) replays natively and may
        # stand as a witness; but one execution says nothing about OTHER seeds, so the run carries ``random_state_read``, which
        # (like every flag) forbids a TRUE.  After a random draw the real state is no longer the native one, and any
        # ``setstate`` rewinds a generator the draws do not use: ``random_state_unmodelled`` (never a witness).
        def f(*a, **k):
            if not _State.seeded:
                advanced = name == "setstate" or any(d["kind"] in ("rindex", "rfloat") for d in _State.log)
                _State.flags.add("random_state_unmodelled" if advanced else "random_state_read")
            return real[name](*a, **k)
        return f

    _random.seed = seed
    _random.random = random
    _random.uniform = uniform
    _random.randrange = randrange
    _random.randint = randint
    _random.choice = choice
    for name in real:
        if name in ("getstate", "setstate"):
            setattr(_random, name, _state_unmodelled(name))
        elif name not in ("seed", "random", "uniform", "randrange", "randint", "choice"):
            setattr(_random, name, _unmodelled(name))
    # the instances: ``random.Random()`` seeded from the OS and ``SystemRandom`` are nondeterministic sources the model
    # does not control (siblings of the module-level ``seed(None)``); a seeded instance, and its own getstate/setstate,
    # are the real generator and behave exactly as natively.
    # Avoid wrapping the same worker twice; a foreign worker still needs this State's flags.
    if getattr(_random.Random.seed, "_flags_unseeded", None) is not _State:
        class_seed = _random.Random.seed

        def instance_seed(self, a=None, *args, **kwargs):
            if a is None:
                _State.flags.add("unseeded_random_unmodelled")
            return class_seed(self, a, *args, **kwargs)

        instance_seed._flags_unseeded = _State
        _random.Random.seed = instance_seed
    for name in ("random", "getrandbits", "randbytes"):
        base = getattr(_random.SystemRandom, name, None)
        if base is not None and getattr(base, "_flags_unseeded", None) is not _State:
            def system(self, *a, _base=base, **k):
                _State.flags.add("unseeded_random_unmodelled")
                return _base(self, *a, **k)
            system._flags_unseeded = _State
            setattr(_random.SystemRandom, name, system)


# --- other nondeterminism the model does not control: flag it ----------------

# --- running the program as a native ``__main__`` ([internal ref], [internal ref]) ------------------------------------------------

def native_sys_path(program: str) -> None:
    """``sys.path`` as ``python program`` has it: the program's own directory first, and never the directory this model
    lives in (``tharkun/sv``): ``import spec`` (or any module named like one of our files) is the program's module or a
    ``ModuleNotFoundError``, never ours."""
    own = os.path.dirname(os.path.abspath(__file__))
    here = os.path.dirname(program)
    sys.path[:] = [here] + [p for p in sys.path if p not in (own, here)]


def native_main_module(program: str) -> types.ModuleType:
    """A fresh ``__main__`` with the attribute set ``python program`` gives it: ``__package__`` None, a real
    ``SourceFileLoader``, ``__spec__`` None, ``__builtins__`` the builtins MODULE (``runpy.run_path`` gave ``__package__ ==
    ''``, ``__loader__`` None and a builtins dict)."""
    mod = types.ModuleType("__main__")
    mod.__dict__.update({"__file__": program, "__cached__": None, "__package__": None, "__spec__": None,
                         "__loader__": importlib.machinery.SourceFileLoader("__main__", program),
                         "__builtins__": builtins})
    return mod


def _watch_resources():
    """Observe caught limits too, without replacing native APIs or a line tracer."""
    flags = _State.flags
    tool = None
    owner = "tharkun.sv.resources"
    callbacks = {}

    def raised(code, offset, exc):
        for base in _TYPE_MRO(_TYPE(exc)):
            if base is _RESOURCE_ERRORS[0] or base is _RESOURCE_ERRORS[1]:
                flags.add("resource_exhausted")
                break

    def restore():
        if tool is None:
            return
        try:
            if get_tool(tool) != owner:
                flags.add("resource_unobserved")
                return
            try:
                if get_events(tool) != mask:
                    flags.add("resource_unobserved")
                set_events(tool, 0)
                for event, callback in callbacks.items():
                    if register(tool, event, None) is not callback:
                        flags.add("resource_unobserved")
            finally:
                free(tool)
        except BaseException:
            flags.add("resource_unobserved")

    try:
        monitoring = sys.monitoring
        get_tool, get_events = monitoring.get_tool, monitoring.get_events
        set_events, register, free = monitoring.set_events, monitoring.register_callback, monitoring.free_tool_id
        # HANDLED sees caught limits; UNWIND sees limits leaving a frame.
        callbacks = {monitoring.events.EXCEPTION_HANDLED: raised, monitoring.events.PY_UNWIND: raised}
        mask = 0
        for event in callbacks:
            mask |= event
        # Use only an unclaimed tool; never reset another observer's events.
        for candidate in (4, 3, 5, 2, 1, 0):
            try:
                monitoring.use_tool_id(candidate, owner)
                tool = candidate
                break
            except ValueError:
                continue
        if tool is None:
            flags.add("resource_unobserved")
        else:
            for event, callback in callbacks.items():
                register(tool, event, callback)
            set_events(tool, mask)
    except BaseException:
        flags.add("resource_unobserved")
    return restore


def run_as_main(code, mod: types.ModuleType) -> None:
    """Execute ``code`` in ``mod`` (a :func:`native_main_module`) at NATIVE recursion depth.

    The program's module frame is the first Python frame of ``python program`` (depth 1); here it sits on top of this
    child's own frames, which count against the recursion limit, so ``def f(n): return 0 if n == 0 else 1 + f(n - 1)``
    with ``f(996)`` raised ``RecursionError`` here and not natively ([internal ref]).  The limit is raised by the number of frames
    below the module frame, and ``sys.getrecursionlimit``/``setrecursionlimit`` are shown to the program as the
    native-depth equivalents for the duration of the run (a program that sets its own limit gets exactly that many frames
    of its own)."""
    depth, frame = 0, sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    real_get, real_set = sys.getrecursionlimit, sys.setrecursionlimit
    base = real_get()

    def getrecursionlimit():
        return real_get() - depth

    def setrecursionlimit(limit):
        return real_set(limit + depth)

    real_set(base + depth)
    sys.getrecursionlimit, sys.setrecursionlimit = getrecursionlimit, setrecursionlimit
    restore_resources = _watch_resources()
    try:
        exec(code, mod.__dict__)
    finally:
        try:
            restore_resources()
        finally:
            sys.getrecursionlimit, sys.setrecursionlimit = real_get, real_set
            real_set(base)


class _AbsentRecorder:
    """The LAST finder on ``sys.meta_path``: it is asked only for a module every other finder (the built-in, the frozen and the path
    finder, and any the program added before it) did not have, so a name it sees is a name ``import`` raises
    ``ModuleNotFoundError`` for.  It never finds anything."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        _State.absent.add(name)
        return None


def watch_imports():
    """Record the module names no finder has (``_State.absent``, reported as ``absent_imports``): ``import spec`` natively is a
    ``ModuleNotFoundError`` unless the program's directory or the interpreter's path has a ``spec``, and the child resolves imports
    exactly like ``python program`` (``native_sys_path``), so what it found absent is absent natively.  The determinism census takes
    it as the fact that such an ``import X`` raises on every run of this environment (``Resolver.absent_imports``).  Idempotent."""
    if not any(f is _AbsentRecorder for f in sys.meta_path):
        sys.meta_path.append(_AbsentRecorder)


def still_absent_imports() -> list:
    """The top-level names ``watch_imports`` saw absent that are STILL absent now: a program that writes ``spec.py`` and imports it
    again has code the census never read behind that ``import``, so a name that became importable is not reported.  Names with a dot
    are not reported either (the census resolves only a plain ``import X``)."""
    names = sorted(n for n in _State.absent if n and "." not in n)
    if not names:
        return []
    # the program has run: it may have rebound a builtin, patched ``sys.modules`` or ``importlib``: whatever goes wrong here, report NOTHING
    # (the census then refuses the import) rather than let a check of ours end the run
    try:
        import importlib.util

        importlib.invalidate_caches()
        out = []
        for name in names:
            try:
                found = importlib.util.find_spec(name)
            except (ImportError, ValueError, AttributeError):
                found = None
            if found is None:
                out.append(name)
        return out
    except BaseException:
        return []


def watch_threads():
    """Make starting ANY thread set ``stack_unmodelled`` ([internal ref], MG review of #153): the authority, whatever the static scan
    (``environment.stack_reason``) could not see -- a helper module of the task that starts the thread, ``exec`` /
    ``eval`` of a literal that imports ``threading``, a module imported under an alias by a computed name.

    Why a thread matters: ``run_as_main`` raises the recursion limit by the child's own frame count so that the program's
    module frame sits at native depth, but a thread starts its own stack at depth 0, so the limit it sees is ~10 frames
    too deep.  ``threading.Thread.start`` is wrapped, and so is every function a thread is created through
    (``_thread.start_new_thread`` / ``start_joinable_thread`` and the copies ``threading`` took of them at import:
    ``threading._start_joinable_thread``).  Returns a callable that undoes the patch (``nondet_harness.drive`` runs in the
    verifier's own process; the disposable children never need it)."""
    import _thread
    import threading

    saved = []

    def wrap(fn):
        def started(*a, **k):
            _State.flags.add("stack_unmodelled")
            return fn(*a, **k)
        started._stack_watch = _State
        return started

    def patch(obj, name):
        fn = getattr(obj, name, None)
        if callable(fn) and getattr(fn, "_stack_watch", None) is not _State:
            saved.append((obj, name, fn))
            setattr(obj, name, wrap(fn))

    for owner, name in thread_starters():
        patch(owner, name)

    def restore():
        for obj, name, fn in reversed(saved):
            setattr(obj, name, fn)
        saved.clear()
    return restore


def _patch_clock():
    for name in ("time", "time_ns", "perf_counter", "perf_counter_ns", "monotonic",
                 "monotonic_ns", "process_time", "process_time_ns"):
        real = getattr(_time, name, None)
        if real is None:
            continue

        def f(*a, _real=real, **k):
            _State.flags.add("time")
            return _real(*a, **k)
        setattr(_time, name, f)




def _observe_fields(namespace, demands):
    """Optional integer samples; reading them must not call target protocols."""
    samples = {}
    # ponytail: only module-rooted instances; traverse containers when demanded.
    for obj in list(namespace.values())[:1000]:
        cls = type(obj)
        if type(cls) is not type:
            continue
        attrs = type.__getattribute__(cls, "__dict__")
        name = type.__getattribute__(cls, "__name__")
        if (name not in demands or attrs.get("__module__") != "__main__"
                or type.__getattribute__(cls, "__bases__") != (object,)
                or not isinstance(attrs.get("__dict__"), types.GetSetDescriptorType)):
            continue
        fields = object.__getattribute__(obj, "__dict__")
        if type(fields) is not dict:
            continue
        for field in demands[name]:
            value = fields.get(field)
            if type(value) is int and int.bit_length(value) <= 64:
                old = samples.setdefault(name, {}).get(field, [value, value])
                samples[name][field] = [min(old[0], value), max(old[1], value)]
    return samples


def run(req):
    """Run one request in a disposable child; installed guards remain until exit."""
    program = os.path.abspath(req["program"])
    _State.vector = req.get("vector", [])
    # Normally this process is a fresh, disposable subprocess (python _child.py
    # req.json) that exits right after writing the result, so a leaked
    # sys.modules["_sv_verifier"] never mattered.  Restoring it anyway costs
    # nothing and makes main() safe to call in-process too (a caller that
    # imports and drives this module directly, rather than always through a
    # subprocess, must not leave a stub behind for whatever test runs next in
    # the same process -- see avengers_seed.py's _sv_verifier_stub_on_path,
    # which crashes on exactly this kind of leftover).
    old_sv = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = _make_sv_verifier()
    _patch_random()
    _patch_clock()
    watch_threads()
    native_sys_path(program)
    watch_imports()
    saved_main = sys.modules.get("__main__")
    res = {"status": "clean"}
    namespace = None
    try:
        with open(program, "rb") as fh:
            source = fh.read()
        # No stand-in for sys.stdin, ever ([internal ref] owner ruling): a dynamic wrapper
        # broke .mode/identity/type for target code that checks any of those. A
        # static, conservative pre-scan flags "stdin touched" (same effect on the
        # TRUE-disqualifying flags set) without touching the real object at all.
        if references_stdin(source):
            _State.flags.add("stdin")
        if native_import_reason(source, program):
            _State.flags.add("import_state_unmodelled")
        if references_stack(source):
            _State.flags.add("stack_unmodelled")
        if not isinstance(_State.vector, list) or any(
                not isinstance(e, dict) or "kind" not in e or "value" not in e
                for e in _State.vector):
            _State.flags.add("invalid_draw")
        if "invalid_draw" in _State.flags:
            res = {"status": "invalid-draw"}
        else:
            argv = initial_argv(program, source)
            if "invalid_draw" in _State.flags:
                res = {"status": "invalid-draw"}
            else:
                if "files" in req:
                    install_files(req["files"], os.getcwd(), _State.flags)
                    refuse_threads(_State.flags)
                # Keep sys.argv a plain list; initial_argv copied and recorded
                # its original value before any target mutation.
                # sys.orig_argv ([internal ref], MG review 2026-09-29): real, UNMODELLED CPython
                # interpreter metadata -- reflects how THIS interpreter process was
                # actually launched (a fresh subprocess vs. a reused forked "explore"
                # worker give genuinely DIFFERENT values, since fork() never re-execs;
                # neither is the real application's argv). _child.py never sets or
                # could soundly set it. Flagged like stdin/argv (a static, conservative
                # source scan, never a stand-in object) so a target that reads it can
                # never contribute a FALSE (unlike plain argv, this channel is not
                # modelled/replayable at all, not just untyped -- see
                # Outcome.violated()'s exclusion for the same reasoning file_unmodelled
                # already gets) or a TRUE (proof_blocker, environment.py).
                if b"orig_argv" in source:
                    _State.flags.add("orig_argv")
                sys.argv = argv
                # the program runs as ``python program`` runs it (``native_main_module``, ``run_as_main``): not through
                # ``runpy.run_path``, whose ``__main__`` has ``__package__ == ''``, no loader, a builtins dict, five extra
                # frames against the recursion limit and a read-and-write of ``sys.argv[0]`` of its own
                code = compile(source, program, "exec", dont_inherit=True)
                mod = native_main_module(program)
                sys.modules["__main__"] = mod
                namespace = mod.__dict__
                run_as_main(code, mod)
    except SystemExit as e:
        res = {"status": "exit", "code": repr(e.code)}
    except BaseException as e:  # noqa: BLE001 -- classifying how the program ended is the point
        # [internal ref] hardening (MG review of PR 115, 2026-09-29): only .filename/.lineno/.name
        # are read below, never .line (the source text) -- so lookup_lines=False, which
        # skips linecache entirely (the plain traceback.extract_tb(tb) equivalent looks
        # up every frame's source line eagerly via linecache.checkcache/updatecache,
        # which os.stat()s each frame's file -- including one outside any declared file
        # or infrastructure root -- and used to need its own exemption in
        # environment.py's sandbox). Not needing that exemption at all is safer than
        # granting it: a target that imports linecache itself and calls it directly on
        # an arbitrary path is correctly still refused/flagged like any other
        # undeclared read.
        frames = traceback.StackSummary.extract(traceback.walk_tb(e.__traceback__), lookup_lines=False)
        pdir = os.path.dirname(program)
        user = [f for f in frames if os.path.abspath(f.filename).startswith(pdir)]
        f = (user or frames or [None])[-1]
        res = {"status": "raised", "exc": type(e).__name__,
               "mro": exception_mro(e), "msg": str(e)[:300],
               "site": ({"file": os.path.basename(f.filename), "line": f.lineno,
                         "function": f.name} if f is not None else None)}
    finally:
        if saved_main is None:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = saved_main
        if old_sv is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = old_sv
    if namespace is not None and req.get("observe_fields"):
        try:
            res["class_hints"] = _observe_fields(namespace, req["observe_fields"])
        except BaseException:
            pass  # optional observations must not change the run's classification
    res["draws"] = _State.log
    res["ranges"] = _State.ranges
    res["flags"] = sorted(_State.flags)
    res["invalid_draw"] = bool(_State.invalid_draw or "invalid_draw" in _State.flags)
    res["absent_imports"] = still_absent_imports()
    return res


def main(req_path):
    with open(req_path) as fh:
        req = json.load(fh)
    res = run(req)
    with _RESULT_FILE_IO(req["out"], "w") as fh:
        fh.write(json.dumps(res).encode())


if __name__ == "__main__":
    main(sys.argv[1])
