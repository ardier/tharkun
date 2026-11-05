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

import sys as _sys

#: the module names this interpreter had before the model imported anything (``evict_import_cache``, [internal ref]): for a
#: direct ``python _child.py`` run that IS the native startup set; a forked child is handed the server-measured one
_STARTUP_MODULES = frozenset(_sys.modules)
#: ``_forkserver.py`` sets this once per server from its own probe script
NATIVE_STARTUP = None

import builtins
import importlib.machinery
import json
import io
_RESULT_FILE_IO = io.FileIO
import math
import opcode
from operator import index as _index
import os
import random as _random
import string
import _string
import sys
import time as _time
import traceback
import types

# Capture interpreter identities before any target code can replace them.
_GETFRAME = sys._getframe
_PY314 = sys.version_info >= (3, 14)
_IMPORT_NAME = opcode.opmap["IMPORT_NAME"]
_LOAD_ATTR = opcode.opmap["LOAD_ATTR"]
_CALL = opcode.opmap["CALL"]
#: the builtins whose call sites ``_mapping_observer`` checks by object (``environment.orig_argv_reason``'s ``observed``)
_VARS, _GETATTR = builtins.vars, builtins.getattr
_EXEC, _EVAL, _COMPILE = builtins.exec, builtins.eval, builtins.compile
_CODE_TYPE = types.CodeType

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
    from environment import (BOX_HOME, _patched_attributes, install_files, orig_argv_reason, references_argv, references_orig_argv, references_stack, references_stdin, refuse_threads, world_record, stdin_bytes, feed_stdin,
                             thread_starters, THREAD_STARTERS as _THREAD_STARTERS, confined, preload_infrastructure,
                             reflective_field_attribute)
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

    No draw defaults to [program]. Code naming argv records that example
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
    mentions = references_argv(source)                  # the code only: never a comment ``# argv`` (R2)
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


def observable(code, reads) -> bool:
    """Does every recorded read (``mapping_reads``) sit on an instruction of ``code`` or of its nested code that the
    observer checks -- a ``LOAD_ATTR`` for a receiver name, a ``CALL`` for a ``(callee, receiver, scope)`` entry?  A
    read the compiler placed elsewhere would go unchecked, so the run is then no witness."""
    seen, stack = set(), [code]
    while stack:
        c = stack.pop()
        for word, position in enumerate(c.co_positions()):
            op = c.co_code[word * 2]
            if op == _LOAD_ATTR or op == _CALL:
                seen.add((op, c.co_filename, *position))
        stack.extend(k for k in c.co_consts if isinstance(k, types.CodeType))
    return all((_CALL if type(entry) is tuple and not (len(entry) == 4 and entry[2] == 'format') else _LOAD_ATTR, *key) in seen
               for key, entry in reads.items() if key[0] == code.co_filename)


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
        if _State.seeded:
            return real["randint"](a, b)
        if not _PY314:
            return randrange(a, b + 1)
        a = _index(a)
        b = _index(b)
        if b < a:
            raise ValueError(f"empty range in randint({a}, {b})")
        return randrange(a, b + 1)

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


def _watch_resources(mapping_reads=(), audit_code=False):
    """Observe caught limits too, without replacing native APIs or a line tracer."""
    flags = _State.flags
    tool = None
    owner = "tharkun.sv.resources"
    callbacks = {}
    local_codes = []
    audit_active = [False]
    audit_seen = [False]

    def code_changed(event, args):
        if event == "tharkun.mapping_observer":
            audit_seen[0] = True
        # owner, 2026-10-29 audit: code-object rewriting is unmodelled (``types.CodeType`` and ``code.replace`` raise
        # ``code.__new__``; ``f.__code__ = ...`` raises ``object.__setattr__``).  ``function.__new__`` (dataclasses and
        # typing raise it) wraps an UNCHANGED code object, whose positions the observer still sees.
        if audit_active[0] and (event == "code.__new__" or (event == "object.__setattr__" and args[1] == "__code__")):
            flags.add("import_state_unmodelled")

    def raised(code, offset, exc):
        for base in _TYPE_MRO(_TYPE(exc)):
            if base is _RESOURCE_ERRORS[0] or base is _RESOURCE_ERRORS[1]:
                flags.add("resource_exhausted")
                break

    def restore():
        audit_active[0] = False
        if tool is None:
            return
        try:
            if get_tool(tool) != owner:
                flags.add("resource_unobserved")
                return
            try:
                for code in local_codes:
                    if get_local_events(tool, code) != monitoring.events.INSTRUCTION:
                        flags.add("resource_unobserved")
                    set_local_events(tool, code, 0)
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
        if mapping_reads or audit_code:
            # Replaced code metadata could hide a receiver from the source-bound instruction observer.
            sys.addaudithook(code_changed)
            sys.audit("tharkun.mapping_observer")
            if not audit_seen[0]:
                flags.add("resource_unobserved")
            audit_active[0] = True
        monitoring = sys.monitoring
        get_tool, get_events = monitoring.get_tool, monitoring.get_events
        set_events, register, free = monitoring.set_events, monitoring.register_callback, monitoring.free_tool_id
        get_local_events, set_local_events = monitoring.get_local_events, monitoring.set_local_events
        # HANDLED sees caught limits; UNWIND sees limits leaving a frame.
        callbacks = {monitoring.events.EXCEPTION_HANDLED: raised, monitoring.events.PY_UNWIND: raised}
        if mapping_reads:
            def arm(code):
                local_codes.append(code)
                set_local_events(tool, code, monitoring.events.INSTRUCTION)
            started, instruction = _mapping_observer(mapping_reads, flags, arm)
            callbacks[monitoring.events.PY_START] = started
            callbacks[monitoring.events.INSTRUCTION] = instruction
        mask = 0
        for event in callbacks:
            if event != monitoring.events.INSTRUCTION:
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


def _same_native_code(actual, expected):
    """Compare executable constants and all source/frame metadata, including nested code."""
    pending = [(actual, expected)]
    while pending:
        left, right = pending.pop()
        if _TYPE(left) is not _CODE_TYPE or _TYPE(right) is not _CODE_TYPE:
            return False
        # CPython code equality omits filename/qualname and frame layout.
        if left != right or (
                left.co_filename, left.co_qualname, left.co_stacksize, left.co_nlocals,
                left.co_varnames, left.co_freevars, left.co_cellvars) != (
                right.co_filename, right.co_qualname, right.co_stacksize, right.co_nlocals,
                right.co_varnames, right.co_freevars, right.co_cellvars):
            return False
        pending.extend((a, b) for a, b in zip(left.co_consts, right.co_consts) if _TYPE(a) is _CODE_TYPE)
    return True


def _mapping_observer(reads, flags, arm):
    """Check the actual receiver of each runtime-admitted comprehension mapping before LOAD_ATTR, and of each recorded
    ``vars(name)`` / ``getattr(name, ...)`` call before CALL (an entry ``(callee, receiver, scope)``,
    ``environment.orig_argv_reason``'s ``observed``): the callee must be the very builtin, else nothing is read; the
    receiver must not be a module (scope ``"module"``) or ``sys`` itself (scope ``"sys"``).
    A ``"format"`` entry checks the exact string receiver before LOAD_ATTR saves its bound method."""
    instructions = {}
    files = {location[0] for location in reads}
    missing = object()
    module_type = types.ModuleType
    identity = id
    enumerate_positions = enumerate
    base_exception = BaseException
    load_attr, call_op = _LOAD_ATTR, _CALL
    value_type, type_mro = _TYPE, _TYPE_MRO
    getframe = _GETFRAME
    real_vars, real_getattr, real_sys = _VARS, _GETATTR, _sys
    code_builtins = {"exec": _EXEC, "eval": _EVAL, "compile": _COMPILE}
    string_types = (str, bytes)
    string_type = str
    is_tuple = tuple
    local_types = (dict, value_type(getframe().f_locals))
    # Compile inspected alternatives before any target code can install observers or compiler hooks.
    compiled = {item: _COMPILE(item[0], item[1], item[2], dont_inherit=True)
                for entry in reads.values() if value_type(entry) is is_tuple and len(entry) == 4 and entry[2] == "code"
                for item in entry[3] if value_type(item) is is_tuple}


    def lookup(frame, name):
        local = frame.f_locals
        if value_type(local) not in local_types:
            flags.add("import_state_unmodelled")  # custom class namespaces can change a value on every read
            return missing
        if name in local:
            return local[name]
        value = frame.f_globals.get(name, missing)
        return frame.f_builtins.get(name, missing) if value is missing else value

    type_dict = value_type.__dict__["__dict__"].__get__
    object_getattribute = object.__getattribute__
    formatter_parser, field_split = _string.formatter_parser, _string.formatter_field_name_split

    def plain_attribute(obj, attribute):
        """``obj.attribute`` when reading it runs no program code (an ordinary instance-dict entry or a class value no
        descriptor wraps, on a type that keeps ``object.__getattribute__``), else ``missing``."""
        try:
            found = missing
            for base in type_mro(value_type(obj)):
                entries = type_dict(base)
                if "__getattribute__" in entries and base is not object:
                    return missing
                if found is missing and attribute in entries:
                    found = entries[attribute]
            if found is not missing:
                for base in type_mro(value_type(found)):
                    entries = type_dict(base)
                    if "__get__" in entries or "__set__" in entries or "__delete__" in entries:
                        return missing
            held = object_getattribute(obj, "__dict__")
            if value_type(held) is not dict:
                return missing
            return held[attribute] if attribute in held else found
        except base_exception:
            return missing

    def reflective_template(text):
        """Does this ACTUAL ``str.format`` template select a reflective attribute (``{0.__globals__[_State]}``), or fail
        to parse?  (the same rule as ``environment._format_selector_reason``, on C parsers the program cannot patch)"""
        pending = [text]
        try:
            while pending:
                for _, field, spec, _ in formatter_parser(pending.pop()):
                    if field is not None:
                        for is_attribute, name in field_split(field)[1]:
                            if is_attribute and reflective_field_attribute(name):
                                return True
                        if spec:
                            pending.append(spec)
        except ValueError:
            return True
        return False

    def started(code, offset):
        if code.co_filename not in files:
            return
        try:
            key = identity(code)
            if key not in instructions:
                sites = {}
                instructions[key] = (code, sites)
                path = code.co_filename
                if path in files:
                    positioned = False
                    for word, position in enumerate_positions(code.co_positions()):
                        positioned = True
                        site_offset = word * 2
                        op = code.co_code[site_offset]
                        if op == call_op:
                            name = reads.get((path, *position))
                            if value_type(name) is is_tuple and not (len(name) == 4 and name[2] == 'format'):
                                sites[site_offset] = name
                            continue
                        if op != load_attr:
                            continue
                        if "__dict__" in code.co_names and (position[0] is None or position[2] is None):
                            flags.add("import_state_unmodelled")
                        name = reads.get((path, *position))
                        if name is not None and (value_type(name) is not is_tuple or (
                                len(name) == 4 and name[2] == 'format')):
                            sites[site_offset] = name
                    if not positioned and "__dict__" in code.co_names:
                        flags.add("import_state_unmodelled")
                if sites:
                    arm(code)
        except base_exception:
            flags.add("import_state_unmodelled")

    def observe(code, offset):
        try:
            key = identity(code)
            sites = instructions[key][1]
            if offset not in sites:
                return
            name = sites[offset]
            frame = getframe(1)
            if name is None or frame.f_code is not code:
                flags.add("import_state_unmodelled")
                return
            if value_type(name) is is_tuple:
                if len(name) == 4 and name[2] == 'format':
                    path = name[0]
                    receiver = lookup(frame, path) if value_type(path) is string_type else lookup(frame, path[0])
                    if value_type(path) is is_tuple:
                        for attribute in path[1:]:
                            receiver = plain_attribute(receiver, attribute)
                    # name[3] is None: an open template (``self.text.format``) -- its ACTUAL text must not select a
                    # reflective attribute; a tuple: the scanned alternatives.
                    if value_type(receiver) is not string_type or (
                            reflective_template(receiver) if name[3] is None else receiver not in name[3]):
                        flags.add('import_state_unmodelled')
                    return
                if len(name) == 4 and name[2] == "attribute":
                    callee, receiver, attribute = lookup(frame, name[0]), lookup(frame, name[1]), lookup(frame, name[3])
                    if callee is real_getattr:
                        if receiver is real_sys:
                            flags.add("import_state_unmodelled")
                        elif value_type(attribute) is string_type:
                            if attribute == "__dict__":
                                if module_type in type_mro(value_type(receiver)):
                                    flags.add("import_state_unmodelled")
                            elif reflective_field_attribute(attribute):
                                flags.add("import_state_unmodelled")
                        elif string_type in type_mro(value_type(attribute)):
                            flags.add("import_state_unmodelled")
                    return
                if len(name) == 4 and name[2] == "code":
                    callee = lookup(frame, name[0])
                    source = lookup(frame, name[1]) if name[1] is not None else name[3][0]
                    matches = (source in name[3] if value_type(source) in string_types else
                               value_type(source) is _CODE_TYPE and any(
                                   _same_native_code(source, compiled[item]) for item in name[3]
                                   if value_type(item) is is_tuple))
                    if callee is not code_builtins[name[0]] or not matches:
                        flags.add("import_state_unmodelled")
                    return
                callee, receiver = lookup(frame, name[0]), lookup(frame, name[1])
                if callee is missing or receiver is missing:
                    flags.add("import_state_unmodelled")
                elif callee is not real_vars and callee is not real_getattr:
                    return                                  # the program calls something else: nothing reflective
                elif name[2] == "sys":
                    if receiver is real_sys:
                        flags.add("import_state_unmodelled")
                else:
                    for base in type_mro(value_type(receiver)):
                        if base is module_type:
                            flags.add("import_state_unmodelled")
                            break
                return
            value = lookup(frame, name)
            if value is missing:
                flags.add("import_state_unmodelled")
            else:
                for base in type_mro(value_type(value)):
                    if base is module_type:
                        flags.add("import_state_unmodelled")
                        break
        except base_exception:
            flags.add("import_state_unmodelled")
    return started, observe


def run_as_main(code, mod: types.ModuleType, *, mapping_reads=(), audit_code=False) -> None:
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
    restore_resources = _watch_resources(mapping_reads, audit_code)
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


def evict_import_cache(startup) -> None:
    """Make ``sys.modules`` the native startup set before the target runs ([internal ref]).

    This child is a model process: it imported its own modules (``json``, ``traceback``, ``environment``, ...) before any
    target code runs, so a target that reads the import cache sees what ``python program.py`` never would.  Every module
    that is not in ``startup``, not model-held and not a C extension is dropped, so the target's next ``import`` executes
    the source again.  Model-held: an owner of a patched attribute (``environment._patched_attributes``: the file and
    spawn model), ``random`` (the draw model), ``_sv_verifier`` and ``_posixsubprocess``.  A builtin, frozen or
    single-phase extension module stays: re-importing it yields the same object, and its state is never the target's.
    """
    held = {owner.__name__ for owner, _ in _patched_attributes()}
    held.update(("random", "_sv_verifier", "_posixsubprocess", "__main__"))
    for name, module in list(sys.modules.items()):
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if name in startup or name in held or not (isinstance(origin, str) and origin.endswith(".py")):
            continue
        del sys.modules[name]


def _watch_line(program, target):
    """Observe the original entry call from the replay harness, excluding import-time calls ([internal ref])."""
    monitoring = sys.monitoring
    path, entry, line = target["path"], target["entry"], target["line"]
    root, reached, pending, stores = None, False, False, set()
    raising_offsets, selected_error, handled_error = set(), None, None
    raising_frames = set()
    selected_class = (getattr(builtins, target["raise_site"]["class"], None)
                      if target.get("raise_site") else None)
    observed_codes = set()
    owner, tool = "tharkun.inputs.through", None

    def matches(code, identity):
        return (code.co_code.hex() == identity["code"] and code.co_linetable.hex() == identity["lines"]
                and list(code.co_varnames) == identity["varnames"] and list(code.co_cellvars) == identity["cellvars"]
                and list(code.co_freevars) == identity["freevars"] and code.co_firstlineno == identity["first_line"])

    def in_root(frame):
        while frame is not None:
            if frame is root:
                return True
            frame = frame.f_back
        return False

    def start(code, offset):
        nonlocal root, reached, pending, stores, raising_offsets, selected_error, handled_error, selected_class
        frame = _GETFRAME(1)
        if (root is None and code.co_filename == path and code.co_qualname == entry
                and frame.f_back is not None and frame.f_back.f_code.co_filename == program
                and (not target.get("caller") or frame.f_back.f_code.co_name == target["caller"])):
            root, reached, pending = frame, False, False
            selected_error = None
            handled_error = None
            raising_frames.clear()
            if target.get("local") or target.get("raise_site"):
                # Disassembly is request data from the parent, before any target import can replace dis helpers.
                unchanged = matches(code, target)
                raising_offsets = set(target["raise_site"]["offsets"]) if unchanged and target.get("raise_site") else set()
                stores = set(target["stores"]) if unchanged and target.get("local") else set()
                if target.get("raise_site", {}).get("kind") == "raise":
                    label = target["raise_site"]["class"]
                    cls = frame.f_globals.get(label, _BUILTIN_CLASSES.get(label))
                    selected_class = cls if isinstance(cls, type) and issubclass(cls, BaseException) else None
            if target.get("local"):
                monitoring.set_local_events(tool, code, monitoring.events.INSTRUCTION)
                observed_codes.add(code)
        selected = target.get("raise_site")
        if (root is not None and selected and code.co_filename == path and code.co_qualname == selected["function"]
                and in_root(frame) and matches(code, selected)):
            raising_frames.add(id(frame))

    def visited(code, lineno):
        nonlocal reached
        if root is not None and code.co_filename == path and lineno == line:
            if not target.get("raise_site") or (_GETFRAME(1) is root and selected_error is not None
                                                and handled_error is selected_error):
                reached = True

    def instruction(code, offset):
        nonlocal pending
        if _GETFRAME(1) is root:
            capture()
            pending = offset in stores

    def capture():
        nonlocal reached, pending
        if pending:
            pending = False
            try:
                value = root.f_locals[target["local"]]
                at_site = not target.get("raise_site") or (selected_error is not None and handled_error is selected_error)
                reached = (at_site and bool(root.f_back.f_globals[target["predicate"]](value))) or reached
            except Exception:
                pass                            # a missing local or failed comparison is no observation

    def end(code, offset, value):
        nonlocal root
        frame = _GETFRAME(1)
        raising_frames.discard(id(frame))
        if frame is root:
            capture()
            root = None

    def raised(code, offset, exception):
        nonlocal pending, selected_error, reached
        frame = _GETFRAME(1)
        if frame is root:
            pending = False                     # an assignment whose RHS raised never produced a local
        if (id(frame) in raising_frames and in_root(frame) and offset in raising_offsets
                and type(exception) is selected_class):
            selected_error = exception
            if target.get("exception_predicate"):
                try:
                    reached = bool(root.f_back.f_globals[target["exception_predicate"]](exception.args)) or reached
                except Exception:
                    pass  # A failed argument comparison is no native exception observation.

    def handled(code, offset, exception):
        nonlocal handled_error, pending
        if _GETFRAME(1) is root:
            pending = False
            handled_error = exception

    def unwind(code, offset, exception):
        nonlocal root, pending
        frame = _GETFRAME(1)
        raising_frames.discard(id(frame))
        if frame is root:
            root, pending = None, False

    callbacks = {monitoring.events.PY_START: start, monitoring.events.LINE: visited,
                 monitoring.events.PY_RETURN: end, monitoring.events.PY_UNWIND: end}
    if target.get("local"):
        del callbacks[monitoring.events.LINE]
        callbacks[monitoring.events.INSTRUCTION] = instruction
        callbacks[monitoring.events.RAISE] = raised
        callbacks[monitoring.events.EXCEPTION_HANDLED] = raised
        callbacks[monitoring.events.PY_UNWIND] = unwind
    if target.get("raise_site"):
        callbacks[monitoring.events.RAISE] = raised
        callbacks[monitoring.events.EXCEPTION_HANDLED] = handled
    mask = 0
    for event in callbacks:
        if event != monitoring.events.INSTRUCTION:
            mask |= event
    for candidate in (4, 3, 5, 2, 1, 0):
        try:
            monitoring.use_tool_id(candidate, owner)
            tool = candidate
            break
        except ValueError:
            continue
    if tool is not None:
        for event, callback in callbacks.items():
            monitoring.register_callback(tool, event, callback)
        monitoring.set_events(tool, mask)

    def finish():
        if tool is None or monitoring.get_tool(tool) != owner:
            return False
        observed = monitoring.get_events(tool) == mask
        monitoring.set_events(tool, 0)
        for code in observed_codes:
            observed = (monitoring.get_local_events(tool, code) == monitoring.events.INSTRUCTION) and observed
            monitoring.set_local_events(tool, code, 0)
        for event, callback in callbacks.items():
            observed = (monitoring.register_callback(tool, event, None) is callback) and observed
        monitoring.free_tool_id(tool)
        return reached and observed

    return finish


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
    if not req.get("native"):
        sys.modules["_sv_verifier"] = _make_sv_verifier()
    saved_main = sys.modules.get("__main__")
    res = {"status": "clean"}
    namespace = None
    finish_line = None
    finish_invariants = None
    if req.get("observe_invariants"):
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        try:
            from _invariant_observer import watch as watch_invariants
        finally:
            sys.path.remove(here)
    # svaudit 2026-10-29: an exception before the program's own code starts is the harness's, not the
    # program's ending -- such a run is "crashed", never a complete execution (a TRUE) or a violation
    started = False
    try:
        with open(program, "rb") as fh:
            source = fh.read()
        if req.get("native"):
            if req.get("vector") or req.get("box") or req.get("world") or req.get("observe_line"):
                raise ValueError("native observation replay does not accept execution models")
            native_sys_path(program)
            sys.argv = [program]
            mod = native_main_module(program)
            sys.modules["__main__"] = mod
            namespace = mod.__dict__
            code = compile(source, program, "exec", dont_inherit=True)
            started = True
            run_as_main(code, mod)
        else:
            # The generated flat helper declares its metadata preparation before target roots or execution models.
            if b"tharkun.avengers.flatting" in source:
                import ast
                tree = ast.parse(source, filename=program)
                if any(isinstance(node, ast.ImportFrom) and node.module == "tharkun.avengers.flatting"
                       and any(name.name == "flat_observation" for name in node.names) for node in tree.body):
                    preload_infrastructure()
            _patch_random()
            _patch_clock()
            watch_threads()
            native_sys_path(program)
            watch_imports()
            # No stand-in for sys.stdin, ever ([internal ref] owner ruling): a dynamic wrapper
            # broke .mode/identity/type for target code that checks any of those. A
            # static, conservative pre-scan flags "stdin touched" (same effect on the
            # TRUE-disqualifying flags set) without touching the real object at all.
            if references_stdin(source):
                _State.flags.add("stdin")
            # [internal ref]: a static-tier reason is what the native-equivalent child below can execute; only what eviction cannot
            # repair (a runtime-tier reason) makes this run useless as a witness
            observing = native_import_reason(source, program)
            mapping_reads = {}
            if observing and native_import_reason(source, program, runtime=True, mapping_reads=mapping_reads):
                _State.flags.add("import_state_unmodelled")
            if references_stack(source, filename=program, observed=mapping_reads):
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
                        install_files(req["files"], os.getcwd(), _State.flags, box=bool(req.get("box")),
                                      **({"world": req["world"]} if "world" in req else {}))
                        refuse_threads(_State.flags)
                        if req.get("box") and references_stdin(source) and stdin_bytes(req.get("world")) is not None:
                            world_record().touch("stdin", req["world"]["stdin"])
                    # Keep sys.argv a plain list; initial_argv copied and recorded
                    # its original value before any target mutation.
                    # sys.orig_argv ([internal ref]; owner, 2026-10-29 audit): the competition's model sets argv
                    # in-process and does not define orig_argv -- here it is THIS interpreter's launch line
                    # (a fresh subprocess and a forked worker even differ) -- so code that reads it, by any
                    # spelling, is never a witness (Outcome.violated()) nor a TRUE (proof_blocker).  The
                    # scan reads the code only (environment.orig_argv_reason: never a comment or data);
                    # the reads it cannot follow by name -- name.__dict__, vars(name), getattr(name, ...)
                    # -- are checked by object right before they run (mapping_reads, _mapping_observer).
                    if orig_argv_reason(source, program, observed=mapping_reads):
                        _State.flags.add("orig_argv")
                    sys.argv = argv
                    if observing:
                        evict_import_cache(NATIVE_STARTUP or _STARTUP_MODULES)
                    # the program runs as ``python program`` runs it (``native_main_module``, ``run_as_main``): not through
                    # ``runpy.run_path``, whose ``__main__`` has ``__package__ == ''``, no loader, a builtins dict, five extra
                    # frames against the recursion limit and a read-and-write of ``sys.argv[0]`` of its own
                    code = compile(source, program, "exec", dont_inherit=True)
                    if mapping_reads and not observable(code, mapping_reads):
                        _State.flags.add("import_state_unmodelled")
                    mod = native_main_module(program)
                    sys.modules["__main__"] = mod
                    namespace = mod.__dict__
                    if req.get("observe_line"):
                        finish_line = _watch_line(program, req["observe_line"])
                    if req.get("observe_invariants"):
                        finish_invariants = watch_invariants(program, req["observe_invariants"])
                    started = True
                    run_as_main(code, mod, mapping_reads=mapping_reads, audit_code=True)
    except SystemExit as e:
        res = ({"status": "exit", "code": repr(e.code)} if started else
               {"status": "crashed", "msg": f"before the program started: SystemExit {e.code!r}"})
    except BaseException as e:  # noqa: BLE001 -- classifying how the program ended is the point
        if not started:
            res = {"status": "crashed", "msg": f"before the program started: {type(e).__name__}: {e}"[:300]}
        else:
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
                   "mro": exception_mro(e), "msg": str(e)[:(2**20 if req.get("native") else 300)],
                   "site": ({"file": os.path.basename(f.filename), "line": f.lineno,
                             "function": f.name} if f is not None else None)}
    finally:
        if finish_invariants is not None:
            res["observations"] = finish_invariants()
        if finish_line is not None:
            res["reached"] = finish_line()
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
    world = world_record()
    if world is not None:
        if req.get("box") and stdin_bytes(req.get("world")) is not None:
            # Imported code can read fd 0 even when the main source never names stdin.
            if not any(path == "stdin" for path, _ in world.entries()) and os.lseek(0, 0, os.SEEK_CUR) > 0:
                world.touch("stdin", req["world"]["stdin"])
        res["world"] = world.entries()
    return res


def main(req_path):
    with open(req_path) as fh:
        req = json.load(fh)
    res = run(req)
    with _RESULT_FILE_IO(req["out"], "w") as fh:
        fh.write(json.dumps(res).encode())


if __name__ == "__main__":
    main(sys.argv[1])
