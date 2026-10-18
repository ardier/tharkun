"""Atheris worker for SV obligation search; all findings still need engine replay."""
import hashlib
import json
import math
import os
import string
import sys
import types

import atheris

# the draw domains of the float helpers live in ``_child.py`` (the judge of every confirmed FALSE); the insert lives only for the import
_SV_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SV_DIR)
try:
    import _child
finally:
    sys.path.remove(_SV_DIR)


#: the values ``nondet_float()`` can produce on CPython 3.14 (``_child.float_producible``, [internal ref]): ``ConsumeRegularFloat`` yields
#: finite floats only, which the child refuses, and never the ``inf`` / ``nan`` a violation here needs
_FLOAT_VALUES = (math.inf, -math.inf, math.nan, -0.0)


#: The builtin classes before any program code runs ([internal ref]; ``_child.py::exception_mro`` is the same rule, this worker
#: imports nothing of tharkun): a property class is the builtin, not a program class that shares its name.
_BUILTIN_CLASSES = {name: obj for name, obj in vars(__import__("builtins")).items() if isinstance(obj, type)}


def _clamp_float_range(v: float, a: float, b: float) -> float:
    """[internal ref]-followup (MAIN 2026-09-28, live ``bm_chaos_nondet/data-lookup`` repro): a
    ``nondet_float_range(a, b)`` proposal, snapped into the call's own domain -- a plain, side-effect-free function so it is
    directly unit-testable without a real Atheris run. ``TestOneInput``'s own ``float_range`` closure is the only caller; this
    is what it must log, never the raw pre-clamp value ``fdp.ConsumeRegularFloat()`` proposed (see :func:`TestOneInput` for
    why that mismatch mattered).

    [internal ref]: the domain is what ``random.uniform(a, b)`` can return (``_child.float_range_producible``), not the closed ``[a, b]``:
    ``b`` itself is out of it for ``(0.0, 1.0)``.  A proposal beyond the domain becomes its LARGEST value (``1 - 2**-53``), a
    proposal inside it the first producible value at or past it (``_child.snap_float_range``), so the vector this worker records
    is one ``_child.py`` confirms."""
    return _child.snap_float_range(v, a, b)


#: One printable character per input byte ([internal ref] follow-up).  A byte that already IS a ``string.printable``
#: character stays itself (so a program literal written into the corpus as ASCII bytes, and a literal
#: libFuzzer's comparison hooks splice into the input, read back as that literal); every other byte (NUL and
#: the other control characters, DEL, 0x80-0xFF) is folded onto the visible ASCII range ``0x20..0x7E``.
_BYTE_TO_CHAR = tuple(chr(b) if chr(b) in string.printable else chr(0x20 + b % 0x5F) for b in range(256))
_STR_DRAW_BYTES = 64


def _printable_str(data: bytes) -> str:
    """A ``nondet_str`` proposal built DIRECTLY from ``string.printable`` characters, one per byte of ``data``:
    the only characters the grading harness accepts for that draw (``eregion_sv/shim/_sv_verifier.py::_draw``
    marks the run invalid otherwise).  This replaces folding Atheris's unicode strings into the domain
    (``ConsumeUnicodeNoSurrogates``, then ``0x20 + ord(c) % 0x5F`` on every character outside it): there the
    first byte picked a 1-, 2- or 4-byte encoding for the whole string and a character outside the domain was
    one of a million code points, so a mutated input was less likely to read back as a target like ``"qqqqq"`` (``b"qqqqq"`` was the
    four-character string ``"qqqq"``).  One byte per character keeps libFuzzer's byte
    mutations, the corpus seeds and the comparison dictionary aligned with the characters the program sees.
    Like :func:`_clamp_float_range`, what the program sees in this worker and what the vector records are the
    SAME value, so a find replays as written."""
    return "".join(_BYTE_TO_CHAR[b] for b in data)


# The program runs FOR REAL in this process, millions of times.  The engine's own child runs it under the task's
# environment model (``solve/environment.py::install_files``: reads of declared files only, every write confined to
# a scratch copy, anything else refused and flagged), and so must this worker: without it a program that opens,
# truncates or writes a file it does not own does exactly that to the host (on sv_2027 a verify with a few
# seconds of budget truncated a file outside the scratch).  Same import discipline as ``_child.py``: the insert
# lives only for the import statement.
_SOLVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "solve")
sys.path.insert(0, _SOLVE)
try:
    from environment import install_files, refuse_threads
finally:
    sys.path.remove(_SOLVE)

with open(os.environ["THARKUN_SV_HYBRID_REQ"]) as fh:
    REQUEST = json.load(fh)
TARGET = os.path.realpath(REQUEST["program"])
sys.path.insert(0, os.path.dirname(TARGET))
with open(TARGET, encoding="utf-8") as fh:
    CODE = atheris.patch_code(compile(fh.read(), TARGET, "exec"), True)
ENV_FLAGS = set()


def _install_environment():
    """Put the process under the environment model.  The worker's cwd is its scratch directory (``sv/hybrid.py``
    starts it there): declared files are staged under it.  Called by the script entry point only: the model patches
    ``open``/``os.open``/... for the rest of the process, which an in-process import of this module (a unit test)
    must not inherit."""
    install_files(REQUEST.get("files") or [TARGET], os.getcwd(), ENV_FLAGS)
    refuse_threads(ENV_FLAGS)
CHECKS = {int(line): [compile(expr, TARGET, "eval") for expr in exprs]
          for line, exprs in REQUEST["checks"].items()}
#: top-level packages of the task-local code the program imports (sv/reach.py): instrumented while the program
#: first runs, so the fuzzer sees coverage INSIDE the library the program calls, not only in the file it was
#: handed ([internal ref]).  Imports are cached after that first run.
REACH = list(REQUEST.get("reach") or [])
COUNT = 0
FOUND = set()
BEST = {}
SEEDS = 0
_FIRST_RUN_DONE = [False]


@atheris.instrument_func
def TestOneInput(data):
    global COUNT, SEEDS
    fdp = atheris.FuzzedDataProvider(data)
    vector = []
    sv = types.ModuleType("_sv_verifier")
    hit = False

    def draw(kind, value):
        vector.append({"kind": kind, "value": float(value).hex() if kind == "float" else value})
        return value

    sv.nondet_int = lambda: draw("int", fdp.ConsumeInt(8))
    sv.nondet_bool = lambda: draw("bool", fdp.ConsumeBool())
    sv.nondet_float = lambda: draw("float", _FLOAT_VALUES[fdp.ConsumeIntInRange(0, len(_FLOAT_VALUES) - 1)])
    sv.nondet_str = lambda: draw("str", _printable_str(fdp.ConsumeBytes(_STR_DRAW_BYTES)))

    def float_range(a, b):
        # [internal ref]-followup: log the CLAMPED value (:func:`_clamp_float_range`), never the
        # raw proposal -- draw("float", ...) used to log sv.nondet_float()'s RAW,
        # pre-clamp value here, so a wide-magnitude Atheris draw (e.g. 4.15e300 for
        # nondet_float_range(0, 1)) reached a real bug INSIDE this worker (which itself
        # already used the correctly clamped value for its own control flow) but left
        # an out-of-domain value in the recorded candidate vector -- [internal ref]'s _child.py
        # fix then (correctly) refuses to confirm that vector at all, silently losing a
        # real finding. Recording the CLAMPED value means the candidate vector
        # _child.py later replays for confirmation already holds exactly what the
        # program saw, so it replays and confirms cleanly.
        return draw("float", _clamp_float_range(fdp.ConsumeRegularFloat(), a, b))

    sv.nondet_float_range = float_range

    def nondet_list(elem):
        out = []
        while len(out) < 16 and sv.nondet_bool():
            out.append(elem())
        return out

    def nondet_dict(key, value):
        out = {}
        while len(out) < 16 and sv.nondet_bool():
            out[key()] = value()
        return out

    sv.nondet_list, sv.nondet_dict = nondet_list, nondet_dict
    old_sv = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = sv

    def trace(frame, event, arg):
        nonlocal hit
        if event == "line" and frame.f_code.co_filename == TARGET:
            name = REQUEST["metrics"].get(str(frame.f_lineno))
            if name and SEEDS < 64:
                value = frame.f_locals.get(name, frame.f_globals.get(name))
                if type(value) in (int, float, bool):
                    distance = abs(value)
                    if math.isfinite(distance) and distance < BEST.get(frame.f_lineno, math.inf):
                        BEST[frame.f_lineno] = distance
                        key = hashlib.sha256(data).hexdigest()[:16]
                        path = os.path.join(sys.argv[1], key)
                        with open(path, "wb") as fh:
                            fh.write(data)
                        SEEDS += 1
            for predicate in CHECKS.get(frame.f_lineno, ()):
                try:
                    if not bool(eval(predicate, frame.f_globals, frame.f_locals)):
                        hit = True
                except Exception:
                    pass
        return trace

    try:
        if CHECKS or REQUEST["metrics"]:
            # a Python-level line tracer slows every run several-fold: only install it when there is
            # a predicate to check or a distance to keep ([internal ref]: the deepening rounds have neither --
            # a raised property exception is the hit -- and run at the fuzzer's own speed)
            sys.settrace(trace)
        try:
            if REACH and not _FIRST_RUN_DONE[0]:
                _FIRST_RUN_DONE[0] = True
                with atheris.instrument_imports(include=REACH):
                    exec(CODE, {"__name__": "__main__", "__file__": TARGET})
            else:
                exec(CODE, {"__name__": "__main__", "__file__": TARGET})
        except BaseException as exc:
            if any(c.__name__ in REQUEST["classes"] and _BUILTIN_CLASSES.get(c.__name__) is c
                   for c in type(exc).__mro__):
                hit = True
    finally:
        sys.settrace(None)
        if old_sv is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = old_sv
    if hit and COUNT < 32:
        name = hashlib.sha256(json.dumps(vector, sort_keys=True).encode()).hexdigest()[:16]
        if name not in FOUND:
            FOUND.add(name)
            path = os.path.join(REQUEST["found"], name + ".json")
            with open(path, "w") as fh:
                json.dump(vector, fh)
            COUNT += 1


if __name__ == "__main__":
    _install_environment()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()
