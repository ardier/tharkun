"""Atheris worker for SV obligation search; all findings still need engine replay."""
import ast
import hashlib
import json
import math
import os
import sys
import types

import atheris

# the draw domains of the float helpers live in ``_child.py`` (the judge of every confirmed FALSE); the insert lives only for the import
_SV_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SV_DIR)
try:
    import _child
    import _hybrid_draws          # the byte layout of every draw, shared with the seed encoder ([internal ref])
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


#: One printable character per input byte ([internal ref] follow-up): ``_hybrid_draws.BYTE_TO_CHAR`` (a byte that already IS a
#: ``string.printable`` character stays itself, so a program literal written into the corpus as ASCII bytes, and a
#: literal libFuzzer's comparison hooks splice into the input, read back as that literal; every other byte is folded
#: onto the visible ASCII range).  A ``nondet_str`` takes every input byte the program's LATER known draws do not need
#: (``_hybrid_draws.str_length``, [internal ref]: no 64-character cap, and a bool after a str keeps its byte); a str OFF that
#: known sequence is length-prefixed (``_hybrid_draws.consume_str``, [internal ref]).
_BYTE_TO_CHAR = _hybrid_draws.BYTE_TO_CHAR


def _printable_str(data: bytes) -> str:
    """A ``nondet_str`` proposal built DIRECTLY from ``string.printable`` characters, one per byte of ``data``:
    the only characters the grading harness accepts for that draw (``eregion_sv/shim/_sv_verifier.py::_draw``
    marks the run invalid otherwise).  ``_hybrid_draws.printable_str``; what the program sees in this worker and
    what the vector records are the SAME value, so a find replays as written."""
    return _hybrid_draws.printable_str(data)


# The program runs FOR REAL in this process, millions of times.  The engine's own child runs it under the task's
# environment model (``solve/environment.py::install_files``: reads of declared files only, every write confined to
# a scratch copy, anything else refused and flagged), and so must this worker: without it a program that opens,
# truncates or writes a file it does not own does exactly that to the host (on sv_2027 a verify with a few
# seconds of budget truncated a file outside the scratch).  Same import discipline as ``_child.py``: the insert
# lives only for the import statement.
_SOLVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "solve")
sys.path.insert(0, _SOLVE)
try:
    from environment import BOX_HOME, confined, feed_stdin, stdin_bytes, references_stdin, install_files, refuse_threads, reset_world
finally:
    sys.path.remove(_SOLVE)

with open(os.environ["THARKUN_SV_HYBRID_REQ"]) as fh:
    REQUEST = json.load(fh)
TARGET = os.path.realpath(REQUEST["program"])
sys.path.insert(0, os.path.dirname(TARGET))
with open(TARGET, encoding="utf-8") as fh:
    SOURCE = fh.read()
    CODE = atheris.patch_code(compile(SOURCE, TARGET, "exec"), True)
ENV_FLAGS = set()


def _install_environment():
    """Put the process under the environment model.  The worker's cwd is its scratch directory (``sv/hybrid.py``
    starts it there): declared files are staged under it.  Called by the script entry point only: the model patches
    ``open``/``os.open``/... for the rest of the process, which an in-process import of this module (a unit test)
    must not inherit.  A request with ``"box"`` ([internal ref]) installs the boxed model and fixes HOME/XDG on the FIXED,
    absent ``environment.BOX_HOME`` -- never the random scratch path, so the object state a run builds carries no
    run-varying path."""
    box = bool(REQUEST.get("box"))
    if box:
        os.environ.update({"HOME": BOX_HOME, "XDG_CONFIG_HOME": BOX_HOME + "/.config",
                           "XDG_DATA_HOME": BOX_HOME + "/.local/share", "XDG_CACHE_HOME": BOX_HOME + "/.cache",
                           "XDG_STATE_HOME": BOX_HOME + "/.local/state"})
    if not box:
        install_files(REQUEST.get("files") or [TARGET], os.getcwd(), ENV_FLAGS)
        refuse_threads(ENV_FLAGS)
CHECKS = {int(line): [compile(expr, TARGET, "eval") for expr in exprs]
          for line, exprs in REQUEST["checks"].items()}
#: top-level packages of the task-local code the program imports (sv/reach.py): instrumented while the program
#: first runs, so the fuzzer sees coverage INSIDE the library the program calls, not only in the file it was
#: handed ([internal ref]).  Imports are cached after that first run.
REACH = list(REQUEST.get("reach") or [])
#: the program's draw kinds in order, from its first run (``sv/hybrid.py``): what a str draw leaves for the later ones
KINDS = list(REQUEST.get("kinds") or [])
COUNT = 0
FOUND = set()
BEST = {}
SEEDS = 0
#: the deepest input so far (most distinct target lines run): written to ``REQUEST["frontier"]`` for the parent, which
#: hands it to the solver when coverage plateaus (``sv/hybrid.py::search``); a proposal, never a verdict
FRONTIER = [0]
_FIRST_RUN_DONE = [False]


@atheris.instrument_func
def _execute_input(data, world=None):
    global SEEDS
    reset_world()                     # [internal ref]: the touch record is this input's, like every drawn value
    fdp = atheris.FuzzedDataProvider(data)
    vector = []
    sv = types.ModuleType("_sv_verifier")
    hit = False
    raised = False
    missing = None
    bindings = {}
    keep_seed = False
    lines = set()

    def draw(kind, value):
        vector.append({"kind": kind, "value": float(value).hex() if kind == "float" else value})
        return value

    sv.nondet_int = lambda: draw("int", fdp.ConsumeInt(8))
    sv.nondet_bool = lambda: draw("bool", fdp.ConsumeBool())
    sv.nondet_float = lambda: draw("float", _FLOAT_VALUES[fdp.ConsumeIntInRange(0, len(_FLOAT_VALUES) - 1)])
    sv.nondet_str = lambda: draw("str", _printable_str(_hybrid_draws.consume_str(
        fdp, KINDS, [d["kind"] for d in vector])))

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
        nonlocal hit, keep_seed
        if event == "line" and frame.f_code.co_filename == TARGET:
            lines.add(frame.f_lineno)
            if world is not None:
                bindings.update({k: v for k, v in frame.f_locals.items() if type(v) in (str, int, bool)})
            name = REQUEST["metrics"].get(str(frame.f_lineno))
            if name and SEEDS < 64:
                value = frame.f_locals.get(name, frame.f_globals.get(name))
                if type(value) in (int, float, bool):
                    distance = abs(value)
                    if math.isfinite(distance) and distance < BEST.get(frame.f_lineno, math.inf):
                        BEST[frame.f_lineno] = distance
                        if world is None:
                            _keep_seed(data)
                        else:
                            keep_seed = True
            for predicate in CHECKS.get(frame.f_lineno, ()):
                try:
                    if not bool(eval(predicate, frame.f_globals, frame.f_locals)):
                        hit = True
                except Exception:
                    pass
        return trace

    try:
        if CHECKS or REQUEST["metrics"] or world is not None:
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
            if isinstance(exc, FileNotFoundError) and isinstance(exc.filename, (str, bytes)):
                missing = os.fsdecode(exc.filename)
            if any(c.__name__ in REQUEST["classes"] and _BUILTIN_CLASSES.get(c.__name__) is c
                   for c in type(exc).__mro__):
                hit = raised = True
    finally:
        sys.settrace(None)
        if len(lines) > FRONTIER[0] and world is None and REQUEST.get("frontier") and vector:
            FRONTIER[0] = len(lines)
            with open(REQUEST["frontier"] + ".tmp", "w") as fh:
                json.dump(vector, fh)
            os.replace(REQUEST["frontier"] + ".tmp", REQUEST["frontier"])
        if old_sv is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = old_sv
    return hit, vector, missing, bindings, keep_seed, raised


def _record_input(data, vector, world=None):
    global COUNT
    candidate = vector if world is None else {"vector": vector, "world": world.snapshot()}
    name = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()[:16]
    if COUNT < 32 and name not in FOUND:
        FOUND.add(name)
        with open(os.path.join(REQUEST["found"], name + ".json"), "w") as fh:
            json.dump(candidate, fh)
        COUNT += 1
        if world is not None:
            _keep_seed(data, world)


def _keep_seed(data, world=None):
    global SEEDS
    if SEEDS >= 64:
        return
    encoded = data if world is None else _hybrid_draws.pack_world(data, world.snapshot())
    key = hashlib.sha256(encoded).hexdigest()[:16]
    with open(os.path.join(sys.argv[1], key), "wb") as fh:
        fh.write(encoded)
    SEEDS += 1


if REQUEST.get("box"):
    # Only mocked search loads the world/drive machinery; an SV worker keeps its bare imports.
    sys.path.insert(0, os.path.dirname(os.path.dirname(_SV_DIR)))
    try:
        from tharkun.driver import _world_proposals
        from tharkun.solve.world import WorldObject, WorldRefused, WorldStore
    finally:
        sys.path.pop(0)
    WORLD_TREE = ast.parse(SOURCE, TARGET)


def _boxed_input(data):
    raw, snapshot = _hybrid_draws.unpack_world(data)
    try:
        world = WorldStore.from_snapshot(REQUEST.get("world", {}))
        if snapshot is not None:
            world.update_snapshot(snapshot)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return
    world_fdp = atheris.FuzzedDataProvider(raw)
    if references_stdin(SOURCE) and "stdin" not in world.objects:
        world.objects["stdin"] = WorldObject("stdin", world_fdp.ConsumeBytes(
            min(world_fdp.remaining_bytes(), _hybrid_draws.STR_DRAW_MAX)), kind="stream")
    targets = iter(CHECKS)
    # shortcut: at most eight file discoveries/refinements per input; broaden with a measured need.
    for _ in range(8):
        ENV_FLAGS.clear()
        with feed_stdin(stdin_bytes(world)), confined(REQUEST.get("files") or [TARGET], flags=ENV_FLAGS,
                                                     box=True, world=world, threads=False) as ctx:
            hit, vector, missing, bindings, keep_seed, raised = _execute_input(raw, world)
            if missing and os.path.isabs(missing) and os.path.commonpath((ctx.scratch, missing)) == ctx.scratch:
                missing = os.path.relpath(missing, ctx.scratch)
        if ENV_FLAGS - {"file_effect"}:
            return
        if keep_seed:
            _keep_seed(raw, world)
        if raised:
            _record_input(raw, vector, world)
            return
        if missing and missing not in world.objects:
            world.objects[missing] = WorldObject(missing, world_fdp.ConsumeBytes(
                min(world_fdp.remaining_bytes(), _hybrid_draws.STR_DRAW_MAX)))
            continue
        changed = False
        for line in targets:
            for proposal in _world_proposals(WORLD_TREE, line, bindings, world, TARGET):
                if isinstance(proposal, WorldRefused):
                    continue
                world.update_snapshot({key: obj.record() for key, obj in proposal.items()})
                changed = True
                break
            if changed:
                break
        if not changed:
            return


@atheris.instrument_func
def TestOneInput(data):
    if REQUEST.get("box"):
        _boxed_input(data)
    else:
        hit, vector, _missing, _bindings, _keep, _raised = _execute_input(data)
        if hit:
            _record_input(data, vector)


if __name__ == "__main__":
    _install_environment()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()
