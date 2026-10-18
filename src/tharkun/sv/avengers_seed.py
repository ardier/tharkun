"""``avengers`` -- SEED candidates mined from Avengers receiver synthesis ([internal ref]).

A SEED stage of :mod:`tharkun.sv.stages`.  Object-heavy Real-world tasks
(``richards``, ``raytrace``, ``go``, ``chaos``, ...) hide most of their branching
behind a class's own methods, several calls below ``main()``; the guard and
concolic stages only see the top-level draws.  This stage names every class the
target module defines and asks :func:`tharkun.avengers.synthesize_object` to
build one and drive its public methods, with the restored Pynguin backend
([internal ref] -- kept *inside* Avengers, never loaded by the SV entry point itself, see
``tests/test_sv_no_backends.py``) as one more source of constructor and setup
arguments a Hypothesis-only search would not have tried. Every LITERAL
int/float/str/bool argument any VALIDATED recipe used becomes one more
candidate value of that literal's kind.

Nothing here is a verdict, and nothing here touches the harness, the top-level
draws or the property. Like every other seed stage, a value only matters if
``engine.Candidates`` tries it at a real draw position and the ENGINE replays
the unmodified program's own ``main()`` and gets an uncaught exception of one
of the property's classes. Avengers only ever runs the target's classes inside
its own disposable worker (``IsolatedExecutor``); a recipe that merely
constructs an interesting-looking object is a hint, never a finding, exactly
like a guessed class invariant is a hint for lane SA until independently
checked -- promotion here is the same ordinary replay as everywhere else in
Avengers, and it happens again, on the real program, before this stage's
guess can mean anything.

Measured on all 80 (task, property) pairs of the SV-Benchmarks Python
Real-world set (24 tasks, [internal ref]), backend actually installed and invoked: 0
gained, 0 lost -- identical verdict everywhere, including the 30 pairs over
richards/raytrace/go/chaos this stage targets, and a real cost on 19/80 pairs
where its own search crowded out the concolic stage's run budget. See
``sv/stages.py`` (the ``REGISTRY`` entry) and ``docs/AVENGERS.md`` for the
full comparison and the keep-out-of-DEFAULT call it produced.
"""

from __future__ import annotations

import ast
import sys
import tempfile
import time
from pathlib import Path

#: at most this many classes per task -- object-heavy tasks are single files
#: with a handful of classes; this stage's cost is per-class, not per-guard
_MAX_CLASSES = 6
#: Avengers' own per-goal search budget, INCLUDING whatever the Pynguin backend
#: spends. Measured: a trivial branch-coverage threshold needs Pynguin's search
#: to actually run past its subprocess/import overhead (a few seconds isn't
#: enough -- the built-in Hypothesis-only portfolio converges to the simplest
#: value that constructs at all and never explores further). This stage runs
#: once per task, not per guard, so 15s/class is proportionate against SV's
#: 120s-900s per-task budgets, the same way concolic gets share=0.4 of what's
#: left; ``sv/stages.py`` documents the on/off measurement this was tuned for.
_PER_CLASS_SECONDS = 15.0
#: hard wall-clock ceiling for the whole stage, independent of how many
#: classes the module defines
_TOTAL_SECONDS = 30.0
_CAP_PER_KIND = 32


_stub_dir: str | None = None


def _add_to_pythonpath(directory: str) -> None:
    """Both ``sys.path`` (this process, and what ``IsolatedExecutor`` forwards
    to its own workers, ``executor.py::_invoke``) AND the ``PYTHONPATH`` env
    var (every OTHER subprocess an Avengers backend spawns on its own --
    Pynguin's runner builds its child's environment from ``os.environ``
    directly, never from this process's ``sys.path``)."""

    if directory not in sys.path:
        sys.path.insert(0, directory)
    import os
    existing = os.environ.get("PYTHONPATH", "")
    entries = existing.split(os.pathsep) if existing else []
    if directory not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([directory, *entries])


def _sv_verifier_stub_on_path() -> None:
    """Every SV task starts ``import _sv_verifier``; Avengers loads the task
    file directly (:mod:`tharkun.solve.loader`, no draw-lifting) and the
    Pynguin backend loads it AGAIN in its own subprocess to instrument it, so
    without a resolvable ``_sv_verifier`` both fail before they start. A
    zero-value stand-in is enough -- Avengers only ever runs the target's own
    classes in disposable workers, never the harness or a real draw -- and it
    is only ever installed when nothing real is already on the path, so it
    can't shadow the harness's own model
    (``src/eregion_sv/shim/_sv_verifier.py``) in a real SV run.
    """

    global _stub_dir
    if _stub_dir is not None:
        if _stub_dir:
            _add_to_pythonpath(_stub_dir)
        return
    import importlib.util
    try:
        found = importlib.util.find_spec("_sv_verifier") is not None
    except ImportError:
        found = False
    except ValueError:
        # sys.modules["_sv_verifier"] holds a module object with __spec__ is
        # None -- not a real, importable module, just a leftover raw stub some
        # OTHER in-process caller installed (sys.modules["_sv_verifier"] = a
        # bare types.ModuleType(...)) and didn't clean up.  find_spec() cannot
        # look past it and would keep raising for every later call in this
        # process, so drop the stale entry: the next `import _sv_verifier`
        # (ours below, or a real one) then goes through the normal machinery
        # instead of returning the same broken object.
        sys.modules.pop("_sv_verifier", None)
        found = False
    if found:
        _stub_dir = ""
        return
    directory = tempfile.mkdtemp(prefix="tharkun-avengers-sv-stub-")
    (Path(directory) / "_sv_verifier.py").write_text(
        "def nondet_int(*a, **k): return 0\n"
        "def nondet_bool(*a, **k): return False\n"
        "def nondet_float(*a, **k): return 0.0\n"
        "def nondet_float_range(*a, **k): return 0.0\n"
        "def nondet_str(*a, **k): return ''\n"
        "LOG = None\n"
    )
    _stub_dir = directory
    _add_to_pythonpath(_stub_dir)


def _classes(tree: ast.Module) -> list[str]:
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def _literals_in(value) -> list[tuple[str, object]]:
    """Every leaf int/float/str/bool literal a recipe argument value contains."""

    out: list[tuple[str, object]] = []
    if isinstance(value, bool):
        out.append(("bool", value))
    elif isinstance(value, int):
        out.append(("int", value))
    elif isinstance(value, float):
        out.append(("float", value))
    elif isinstance(value, str):
        out.append(("str", value))
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            out.extend(_literals_in(item))
    elif isinstance(value, dict):
        for k, v in value.items():
            out.extend(_literals_in(k))
            out.extend(_literals_in(v))
    return out


def _recipe_literals(recipe) -> list[tuple[str, object]]:
    out = list(_literals_in(list(recipe.constructor.args)))
    out.extend(_literals_in([v for _, v in recipe.constructor.kwargs]))
    for call in recipe.setup:
        out.extend(_literals_in(list(getattr(call, "args", ()))))
        out.extend(_literals_in([v for _, v in getattr(call, "kwargs", ())]))
    return out


def seed_values(program: str, *, max_classes: int = _MAX_CLASSES,
                 per_class_seconds: float = _PER_CLASS_SECONDS,
                 total_seconds: float = _TOTAL_SECONDS, log=None, files=None) -> dict[str, list]:
    """Candidate draw values, per kind, mined from Avengers receiver recipes."""

    say = log or (lambda msg: None)
    try:
        from tharkun.avengers import ObjectGoal, SaturationBudget, synthesize_object
    except ImportError:
        return {}
    try:
        from tharkun.sv.engine import read_source

        tree = ast.parse(read_source(program))
    except (OSError, SyntaxError, ValueError):
        return {}
    _sv_verifier_stub_on_path()

    out: dict[str, list] = {}
    seen: set = set()

    def add(kind: str, v) -> None:
        key = (kind, repr(v))
        if key not in seen and len(out.get(kind, [])) < _CAP_PER_KIND:
            seen.add(key)
            out.setdefault(kind, []).append(v)

    names = _classes(tree)[:max_classes]
    t0 = time.monotonic()
    tried = 0
    for name in names:
        if time.monotonic() - t0 > total_seconds:
            say(f"avengers: total budget spent after {tried}/{len(names)} classes")
            break
        tried += 1
        try:
            goal = ObjectGoal(program, name)
            report = synthesize_object(
                goal,
                objectives=True,
                files=tuple(files) if files is not None else (str(Path(program).resolve()),),
                backends=("pynguin",),
                backend_seconds=int(per_class_seconds),
                budget=SaturationBudget(max_seconds=per_class_seconds),
            )
        except Exception as error:  # a target import/construction failure is not a candidate
            say(f"avengers: {name} unavailable: {type(error).__name__}: {error}")
            continue
        for validated in report.validated:
            for kind, v in _recipe_literals(validated.recipe):
                add(kind, v)
    say(f"avengers: {tried} classes tried, {sum(len(v) for v in out.values())} candidates")
    return out
