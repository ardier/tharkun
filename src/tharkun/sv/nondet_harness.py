"""Runs the WHOLE program with ``_sv_verifier.nondet_*()`` reading from a drive vector,
in THIS process -- the owner's fix ([internal ref]) for a program ``sv/sites.py::build_harness``
cannot lift into a synthetic ``__sv_harness(d0, d1, ...)`` function by static rewriting
(no ``if __name__ == "__main__":`` guard -- the python-string-utils surviving-mutants
demo's own shape -- a draw inside a loop/lambda/nested def, ...). Instead of lifting
draws into parameters by static analysis, this runs the program itself, the same way
``_child.py`` already does for a replay: a position past ``len(draws)`` draws the kind's
default and is still recorded, so a caller can GROW the vector by watching what the
program actually asked for.

This reuses ``_child.py``'s own model (``_make_sv_verifier``, ``_patch_random``,
``_patch_clock``, ``_State``, ``_draw``) rather than a second copy of it -- the owner's
ruling that SV should share Tharkun's machinery, not keep a thin copy of its own. The
one real difference: ``_child.py`` throws its process away after one run, so it never
undoes what it patches; everything here runs in the SAME process as the solvers that
call it, so every patch is undone in a ``finally`` before :func:`drive` returns,
successful or not.

[internal ref]: argv_full installs any plain list of strings; stored argv draws retain
their tail meaning. The child owns this shared boot boundary, including original
snapshots and the unmodelled orig_argv refusal. sys.stdin is never replaced.

Only for EXPLORATION. Nothing here ever decides FALSE: ``driver.py``'s solvers (via
W53's assist loop) may call :func:`drive` many times to find a candidate vector, but
every finding must still replay through ``sv/engine.py::run_once`` in a genuinely fresh
child process before the engine trusts it (docs/SV.md "3a": "only a fresh-child replay
... gives FALSE").
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from tharkun.sv import _child

#: every ``random`` name ``_child.py::_patch_random`` may rebind, restored after each
#: :func:`drive` call (mirrors ``_patch_random``'s own ``real`` dict exactly, [internal ref]).
_RANDOM_NAMES = ("seed", "random", "uniform", "randint", "randrange", "choice", "shuffle",
                 "sample", "choices", "getrandbits", "gauss", "normalvariate", "triangular",
                 "betavariate", "expovariate", "gammavariate", "lognormvariate",
                 "vonmisesvariate", "paretovariate", "weibullvariate", "randbytes",
                 "binomialvariate", "getstate", "setstate")
_TIME_NAMES = ("time", "time_ns", "perf_counter", "perf_counter_ns", "monotonic",
              "monotonic_ns", "process_time", "process_time_ns")


@dataclass
class DriveOutcome:
    """One in-process run: enough for a caller to grow the drive vector, check whether
    the property was violated, and check whether ``target_line`` was reached."""
    status: str                              # "clean" | "exit" | "raised"
    exc: str | None = None
    mro: list = field(default_factory=list)
    draws: list = field(default_factory=list)      # the FULL sequence actually consumed
    flags: list = field(default_factory=list)
    reached: bool = False                    # did target_line execute, in program's own file?
    invalid_draw: bool = False               # a draw fell outside its documented domain ([internal ref])

    def violated(self, classes) -> str | None:
        from tharkun.sv.engine import NEVER_A_WITNESS_FLAGS, _resource_exhausted  # lazy: engine imports siblings

        if (self.status != "raised" or self.invalid_draw or _resource_exhausted(self)
                or any(f in self.flags for f in NEVER_A_WITNESS_FLAGS)):
            return None
        return next((c for c in classes if c in self.mro), None)


def drive(program: str, draws: list, target_line: int | None = None, *, files=None) -> DriveOutcome:
    """Execute *program* fresh, with ``_sv_verifier.nondet_*()`` reading *draws*
    (``sv-py-witness/1`` vector shape: ``{"kind": ..., "value": ...}`` dicts) in order.
    *target_line* (if given) is watched with ``sys.settrace``; ``result.reached`` is True
    the first time that line executes in *program*'s own file (not a line some function
    it calls happens to share a number with)."""
    real_path = os.path.abspath(program)
    with open(real_path, "rb") as stream:
        source = stream.read()
    if _child.native_import_reason(source, real_path, reversible=True):
        from tharkun.sv.engine import run_once

        flags = {"import_state_unmodelled", "stack_unmodelled"}
        declared = list(dict.fromkeys([real_path, *(os.path.abspath(p) for p in (files or ()))]))
        try:
            outcome = run_once(sys.executable, real_path, draws, timeout=30, files=declared)
        except OSError:
            return DriveOutcome("refused", flags=sorted(flags))
        # A disposable replay preserves native module semantics, but provides no
        # observation of target_line and cannot certify an exploratory violation.
        return DriveOutcome(outcome.status, outcome.exc, outcome.mro, outcome.draws,
                            sorted(flags | set(outcome.flags)), False, outcome.invalid_draw)

    saved_state = (_child._State.vector, _child._State.pos, _child._State.log,
                   _child._State.flags, _child._State.seeded, _child._State.invalid_draw,
                   _child._State.ranges)
    saved_random = {name: getattr(_child._random, name) for name in _RANDOM_NAMES
                    if hasattr(_child._random, name)}
    saved_random_classes = [(cls, name, getattr(cls, name))
                            for cls, names in ((_child._random.Random, ("seed",)),
                                               (_child._random.SystemRandom, ("random", "getrandbits", "randbytes")))
                            for name in names if hasattr(cls, name)]
    saved_time = {name: getattr(_child._time, name) for name in _TIME_NAMES
                 if hasattr(_child._time, name)}
    saved_sv = sys.modules.get("_sv_verifier")
    saved_main = sys.modules.get("__main__")
    saved_argv = sys.argv
    saved_syspath = list(sys.path)
    saved_trace = sys.gettrace()

    _child._State.vector = draws
    _child._State.pos = 0
    _child._State.log = []
    _child._State.flags = set()
    _child._State.seeded = False
    _child._State.invalid_draw = False
    _child._State.ranges = []

    reached = False
    unwatch_threads = None

    def tracer(frame, event, arg):
        nonlocal reached
        if (event == "line" and target_line is not None
                and os.path.abspath(frame.f_code.co_filename) == real_path
                and frame.f_lineno == target_line):
            reached = True
        return tracer

    try:
        sys.modules["_sv_verifier"] = _child._make_sv_verifier()
        _child._patch_random()
        _child._patch_clock()
        unwatch_threads = _child.watch_threads()
        mod = _child.native_main_module(real_path)
        sys.modules["__main__"] = mod
        sys.path[:0] = [os.path.dirname(real_path)]
        if target_line is not None:
            sys.settrace(tracer)
        from tharkun.solve.environment import confined

        with confined(files if files is not None else [real_path], flags=_child._State.flags, threads=False):
            try:
                # Execute the admitted snapshot; trusted import-root probes ran before confinement.
                sys.argv = _child.initial_argv(real_path, source)
                if b"orig_argv" in source:
                    _child._State.flags.add("orig_argv")
                if _child.references_stack(source):
                    _child._State.flags.add("stack_unmodelled")
                if "invalid_draw" in _child._State.flags:
                    result = DriveOutcome("invalid-draw")
                else:
                    code = compile(source, real_path, "exec", dont_inherit=True)
                    _child.run_as_main(code, mod)
                    result = DriveOutcome("clean")
            except SystemExit:
                result = DriveOutcome("exit")
            except BaseException as e:  # noqa: BLE001 -- classifying how the run ended is the point
                result = DriveOutcome("raised", type(e).__name__, _child.exception_mro(e))
            finally:
                if target_line is not None:
                    sys.settrace(saved_trace)
        result.draws = list(_child._State.log)
        result.flags = sorted(_child._State.flags)
        result.reached = reached
        result.invalid_draw = _child._State.invalid_draw
        return result
    finally:
        sys.argv = saved_argv
        sys.path[:] = saved_syspath
        if saved_main is None:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = saved_main
        if saved_sv is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = saved_sv
        for name, fn in saved_time.items():
            setattr(_child._time, name, fn)
        for name, fn in saved_random.items():
            setattr(_child._random, name, fn)
        for cls, name, fn in saved_random_classes:
            setattr(cls, name, fn)
        if unwatch_threads is not None:
            unwatch_threads()
        (_child._State.vector, _child._State.pos, _child._State.log,
         _child._State.flags, _child._State.seeded, _child._State.invalid_draw,
         _child._State.ranges) = saved_state
