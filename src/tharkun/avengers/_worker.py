"""Private child-interpreter entry point for :mod:`tharkun.avengers.executor`."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import pickle
import sys
from dataclasses import replace
from types import MemberDescriptorType
from typing import Any

from tharkun.avengers._wire import encode_response
from tharkun.avengers.model import (
    ConstructionRecipe,
    ConstructorCall,
    FunctionCall,
    MethodCall,
    NestedObject,
    ObjectGoal,
    ObjectReference,
    ReplayOutcome,
)


class _NestedFailure(Exception):
    """A nested recipe did not build; carries the child's own outcome."""

    def __init__(self, outcome: ReplayOutcome) -> None:
        super().__init__(outcome.detail or outcome.kind)
        self.outcome = outcome


# Marks a position where the receiver does not exist yet -- its own
# constructor arguments.  Substituting ``None`` there would silently build a
# different object than the recipe asked for.
_NO_RECEIVER = object()


def _qualified(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


_EXACT_SCALARS = (bool, int, float, str, bytes, type(None))
_EXACT_CONTAINERS = (list, tuple, dict, set, frozenset)


def _shallow_state(value: Any) -> str | None:
    """A digest of the receiver's own attributes, running no target code.

    This is the fallback for an environment without ``py_obj_flatting``.  It
    is deliberately shallow and deliberately exact-typed: only values whose
    type IS a builtin scalar are recorded, only builtin containers report a
    length (through the exact type's own C method, never a subclass override),
    and everything else contributes its type name alone.  No ``repr``, no
    ``__eq__``, no ``__len__`` dispatch, no property is read -- a digest that
    ran the object's code would not be a description of its state.

    It distinguishes fewer states than the flatting digest does. That is the
    honest trade: two states this call sees as equal may differ, so it can
    undercount state coverage, but it never invents a difference.
    """

    try:
        holder = next(
            (cls.__dict__["__dict__"] for cls in type(value).__mro__ if "__dict__" in cls.__dict__),
            None,
        )
        state = holder.__get__(value) if holder is not None else None
    except Exception:
        state = None
    fields: list[tuple[str, Any]] = []
    if isinstance(state, dict):
        for name in sorted(state):
            item = state[name]
            kind = type(item)
            if kind in _EXACT_SCALARS:
                fields.append((name, [kind.__name__, item.hex() if kind is bytes else item]))
            elif kind in _EXACT_CONTAINERS:
                try:
                    size = kind.__len__(item)
                except Exception:
                    size = None
                fields.append((name, [kind.__name__, size]))
            else:
                fields.append((name, [f"{kind.__module__}.{kind.__qualname__}"]))
    for slot in getattr(type(value), "__slots__", ()) or ():
        descriptor = type(value).__dict__.get(slot)
        if not isinstance(descriptor, MemberDescriptorType):
            continue
        try:
            item = descriptor.__get__(value)
        except Exception:
            continue
        kind = type(item)
        fields.append(
            (
                slot,
                [kind.__name__, item.hex() if kind is bytes else item]
                if kind in _EXACT_SCALARS
                else [f"{kind.__module__}.{kind.__qualname__}"],
            )
        )
    body = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "fields": sorted(fields),
    }
    try:
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return None
    return "s:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:22]


def _safe_state_digest(value: Any) -> str | None:
    """Use py_obj_flatting when installed; never call target repr as fallback."""

    try:
        import py_obj_flatting as pof
    except ImportError:
        return _shallow_state(value)
    try:
        canonical = pof.flatten(value).canonical_json()
    except Exception:
        return _shallow_state(value)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()[:24]


_sv_verifier_stub_dir: str | None = None
_source_only = False


def _sv_verifier_stub_on_path() -> None:
    """Make ``_sv_verifier`` importable for a target that does
    ``import _sv_verifier`` at module scope -- every real SV task file does,
    so without this ``load_module_for_execution`` below simply fails to load
    one at all, and a goal built from a real SV task ([internal ref], Eregion's stage;
    drive_to_line's object-guard flip) could never even be tried.  A
    zero-value stand-in is enough: this worker only ever constructs and
    drives an object in isolation, never the harness or a real draw.

    Deliberately duplicated from (not imported from)
    ``sv/avengers_seed.py``'s near-identical helper: ``avengers/`` may not
    import from ``sv/`` (LAYERS.md -- SV sits above DRIVE, not below it);
    this is small and self-contained enough that a shared module for two
    one-line callers would be more layering than the duplication it avoids.
    """
    global _sv_verifier_stub_dir
    if _sv_verifier_stub_dir:
        if _sv_verifier_stub_dir not in sys.path:
            sys.path.insert(0, _sv_verifier_stub_dir)
        return
    import importlib.util

    try:
        found = importlib.util.find_spec("_sv_verifier") is not None
    except ImportError:
        found = False
    except ValueError:
        # a stale, spec-less leftover from some OTHER in-process caller --
        # not a real, importable module; treat it as absent (see the same
        # fix in sv/avengers_seed.py::_sv_verifier_stub_on_path).
        sys.modules.pop("_sv_verifier", None)
        found = False
    if found:
        _sv_verifier_stub_dir = ""
        return
    import tempfile
    from pathlib import Path

    directory = tempfile.mkdtemp(prefix="tharkun-avengers-sv-stub-")
    (Path(directory) / "_sv_verifier.py").write_text(
        "def nondet_int(*a, **k): return 0\n"
        "def nondet_bool(*a, **k): return False\n"
        "def nondet_float(*a, **k): return 0.0\n"
        "def nondet_float_range(*a, **k): return 0.0\n"
        "def nondet_str(*a, **k): return ''\n"
        "LOG = None\n"
    )
    _sv_verifier_stub_dir = directory
    sys.path.insert(0, _sv_verifier_stub_dir)


def _load_class(goal: ObjectGoal) -> tuple[type[Any], dict[str, Any]]:
    from tharkun.solve.loader import load_module_for_execution

    _sv_verifier_stub_on_path()
    _module, namespace = load_module_for_execution(goal.source_path, source_only=_source_only)
    if not namespace:
        raise ImportError(f"could not load {goal.source_path}")
    value: Any = namespace
    for part in goal.class_name.split("."):
        value = value[part] if isinstance(value, dict) else getattr(value, part)
    if not isinstance(value, type):
        raise TypeError(f"{goal.class_name} is not a class")
    return value, namespace


def _argument(value: Any, receiver: Any) -> Any:
    if isinstance(value, ObjectReference):
        if receiver is _NO_RECEIVER:
            raise ValueError(
                "the receiver cannot appear in its own constructor arguments"
            )
        return receiver
    if isinstance(value, NestedObject):
        nested = ObjectGoal(value.source_path, value.class_name)
        built, _namespace, failure = _prepare(nested, value.recipe)
        if failure is not None:
            raise _NestedFailure(failure)
        return built
    if isinstance(value, tuple):
        return tuple(_argument(item, receiver) for item in value)
    if isinstance(value, list):
        return [_argument(item, receiver) for item in value]
    if isinstance(value, dict):
        return {
            _argument(key, receiver): _argument(item, receiver)
            for key, item in value.items()
        }
    if isinstance(value, set):
        return {_argument(item, receiver) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_argument(item, receiver) for item in value)
    return value


def _resolve_function(target: str, namespace: dict[str, Any]) -> Any:
    parts = target.split(".")
    value = namespace.get(parts[0])
    if value is None:
        raise NameError(f"{parts[0]!r} is not present in the target module")
    for part in parts[1:]:
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"{target} is not callable")
    return value


def _invoke(call: MethodCall | FunctionCall, receiver: Any, namespace: dict[str, Any]) -> Any:
    args = tuple(_argument(value, receiver) for value in call.args)
    kwargs = {name: _argument(value, receiver) for name, value in call.kwargs}
    if isinstance(call, MethodCall):
        function = getattr(receiver, call.name)
    else:
        function = _resolve_function(call.target, namespace)
    return function(*args, **kwargs)


def _exception(error: BaseException, phase: str) -> ReplayOutcome:
    if isinstance(error, _NestedFailure):
        # Report the nested failure's own kind and cause, tagged with where in
        # the outer recipe it was needed. A nested constructor that raised is
        # not "the outer setup raised"; the caller has to be able to tell.
        inner = error.outcome
        return ReplayOutcome(
            inner.kind,
            phase=f"{phase}/nested:{inner.phase}",
            detail=inner.detail,
            exception_type=inner.exception_type,
            signal=inner.signal,
        )
    return ReplayOutcome(
        "raised",
        phase=phase,
        detail=str(error)[:500],
        exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
    )


def _load(goal: ObjectGoal) -> tuple[type[Any], dict[str, Any]] | ReplayOutcome:
    """Load the target module, kept separate so it can stay outside a trace.

    Import-time lines are not what a recipe exercised -- every recipe for the
    same module would be credited with the same module body -- so the coverage
    measurement starts after this.
    """

    try:
        return _load_class(goal)
    except BaseException as error:
        return _exception(error, "load")


def _build(
    cls: type[Any], namespace: dict[str, Any], recipe: ConstructionRecipe
) -> tuple[Any | None, dict[str, Any], ReplayOutcome | None]:
    try:
        args = tuple(_argument(item, _NO_RECEIVER) for item in recipe.constructor.args)
        kwargs = {
            name: _argument(item, _NO_RECEIVER)
            for name, item in recipe.constructor.keyword_arguments.items()
        }
        value = cls(*args, **kwargs)
    except BaseException as error:
        return None, namespace, _exception(error, "construct")

    for index, call in enumerate(recipe.setup):
        try:
            _invoke(call, value, namespace)
        except BaseException as error:
            return None, namespace, _exception(error, f"setup[{index}]")
    return value, namespace, None


def _prepare(
    goal: ObjectGoal, recipe: ConstructionRecipe
) -> tuple[Any | None, dict[str, Any], ReplayOutcome | None]:
    loaded = _load(goal)
    if isinstance(loaded, ReplayOutcome):
        return None, {}, loaded
    return _build(loaded[0], loaded[1], recipe)


def _tracer(goal: ObjectGoal):
    from tharkun.trace_api import TracePolicy, Tracer

    return Tracer(TracePolicy(paths={goal.source_path}, arcs=True))


def _with_coverage(outcome: ReplayOutcome, tracer: Any, goal: ObjectGoal) -> ReplayOutcome:
    if tracer is None:
        return outcome
    lines = tracer.result.executed.get(goal.source_path, set())
    arcs = tracer.result.arcs.get(goal.source_path, set())
    return replace(outcome, lines=tuple(sorted(lines)), arcs=tuple(sorted(arcs)))


def _run(
    goal: ObjectGoal, recipe: ConstructionRecipe, *, trace: bool = False
) -> ReplayOutcome:
    loaded = _load(goal)
    if isinstance(loaded, ReplayOutcome):
        return loaded
    cls, namespace = loaded
    tracer = _tracer(goal) if trace else None

    with tracer if tracer is not None else contextlib.nullcontext():
        value, namespace, failure = _build(cls, namespace, recipe)
        if failure is not None:
            # A failed recipe still executed something; that coverage is how a
            # caller tells "raised on the way in" from "never got started".
            return _with_coverage(failure, tracer, goal)

        return_types: list[str] = []
        for index, call in enumerate(goal.calls):
            try:
                returned = _invoke(call, value, namespace)
            except BaseException as error:
                return _with_coverage(_exception(error, f"boundary[{index}]"), tracer, goal)
            return_types.append(_qualified(returned))
    return _with_coverage(
        ReplayOutcome(
            "returned",
            phase="boundary",
            return_types=tuple(return_types),
            state_digest=_safe_state_digest(value),
        ),
        tracer,
        goal,
    )


def _flatting(goal: ObjectGoal, recipe: ConstructionRecipe):
    value, _namespace, failure = _prepare(goal, recipe)
    if failure is not None:
        return failure
    try:
        from tharkun.avengers.flatting import template_from_object

        return template_from_object(value)
    except ImportError as error:
        return ReplayOutcome(
            "unavailable",
            phase="flatting",
            detail=str(error),
            exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
        )
    except BaseException as error:
        return _exception(error, "flatting")


def _ghostwriter(goal: ObjectGoal) -> Any:
    """Run Hypothesis' Ghostwriter over the target module, inside the child.

    Ghostwriter imports the module to read its signatures, which is target code
    running, so it belongs here rather than in the parent.
    """

    from tharkun.avengers._wire import GeneratedSource

    loaded = _load(goal)
    if isinstance(loaded, ReplayOutcome):
        return loaded
    try:
        from hypothesis.extra import ghostwriter
    except ImportError as error:
        return ReplayOutcome(
            "unavailable",
            phase="ghostwriter",
            detail=str(error),
            exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
        )
    cls = loaded[0]
    try:
        source = ghostwriter.magic(cls)
    except BaseException as error:
        return _exception(error, "ghostwriter")
    return GeneratedSource("ghostwriter", source)


def _hypothesis_arguments(goal: ObjectGoal, options: dict[str, Any]) -> Any:
    """Draw constructor arguments with ``st.from_type`` on each parameter.

    ``from_type`` is asked about the PARAMETERS, not about the class: drawing
    the class itself would produce an object whose constructor arguments cannot
    be recovered, and an argument-less object is not a recipe.  Resolving an
    annotation needs the module's own namespace, so this runs in the child.
    """

    from tharkun.avengers._wire import ConstructorCandidates

    loaded = _load(goal)
    if isinstance(loaded, ReplayOutcome):
        return loaded
    cls, namespace = loaded
    try:
        from hypothesis import HealthCheck, Phase, Verbosity, given, settings
        from hypothesis import strategies as st
    except ImportError as error:
        return ReplayOutcome(
            "unavailable",
            phase="hypothesis",
            detail=str(error),
            exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
        )
    from tharkun.solve.frontend.constructor import constructor_template

    template = constructor_template(goal.source_path, goal.class_name)
    if template is None:
        return ConstructorCandidates(())

    drawn: dict[str, Any] = {}
    for parameter in template.parameters:
        if parameter.kind in {"var_positional", "var_keyword"} or parameter.has_default:
            continue
        resolved = _resolve_annotation(parameter.annotation, namespace)
        if resolved is None:
            return ConstructorCandidates(())
        try:
            drawn[parameter.name] = st.from_type(resolved)
        except Exception:
            return ConstructorCandidates(())

    found: list[dict[str, Any]] = []
    limit = int(options.get("max_examples", 50))

    @settings(
        max_examples=limit,
        deadline=None,
        database=None,
        verbosity=Verbosity.quiet,
        suppress_health_check=list(HealthCheck),
        phases=(Phase.generate,),
    )
    @given(st.fixed_dictionaries(drawn))
    def probe(values: dict[str, Any]) -> None:
        if len(found) >= 4:
            return
        try:
            record = ConstructorCall(kwargs=values).record
        except TypeError:
            # Not expressible as a recipe literal; a drawn object we could not
            # write down is not a candidate, however valid it is.
            return
        try:
            cls(**values)
        except BaseException:
            return
        if record not in found:
            found.append(record)

    try:
        probe()
    except BaseException:
        pass
    return ConstructorCandidates(tuple(found))


def _resolve_annotation(annotation: str | None, namespace: dict[str, Any]) -> Any:
    """Evaluate an annotation in the target module's namespace, or ``None``.

    Only a name lookup and subscripting are evaluated; anything the module's
    own namespace plus ``typing`` cannot answer returns ``None`` rather than a
    guess.
    """

    if not annotation:
        return None
    import typing

    scope = {"typing": typing, **vars(typing), **namespace}
    try:
        return eval(annotation, {"__builtins__": __builtins__}, scope)  # noqa: S307
    except BaseException:
        return None


def _dispatch(payload: Any) -> Any:
    action = payload[0]
    if action == "batch":
        _action, items, options = payload
        trace = bool(options.get("trace"))
        outcomes = []
        for goal, recipe in items:
            if not isinstance(goal, ObjectGoal) or not isinstance(recipe, ConstructionRecipe):
                raise TypeError("invalid Avengers worker request")
            outcomes.append(_run(goal, recipe, trace=trace))
        return tuple(outcomes)
    _action, goal, recipe, options = payload
    if not isinstance(goal, ObjectGoal) or not isinstance(recipe, ConstructionRecipe):
        raise TypeError("invalid Avengers worker request")
    if action == "flatting":
        return _flatting(goal, recipe)
    if action == "ghostwriter":
        return _ghostwriter(goal)
    if action == "hypothesis-arguments":
        return _hypothesis_arguments(goal, options)
    return _run(goal, recipe, trace=bool(options.get("trace")))


def main() -> int:
    global _source_only

    if len(sys.argv) != 2:
        return 2
    response_path = sys.argv[1]
    try:
        request = sys.stdin.buffer.read()
        payload = pickle.loads(request)
        from tharkun.solve.environment import install_files, refuse_threads

        files = payload[-1].get("files")
        _source_only = files is not None
        flags = set()
        if files is not None:
            scratch = payload[-1]["scratch"]
            # Trusted stub preparation must precede the first target execution.
            _sv_verifier_stub_on_path()
            install_files(files, scratch, flags)
            refuse_threads(flags)
            os.chdir(scratch)
        # This disposable process stays confined through target finalization.
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                outcome = _dispatch(payload)
        except BaseException as error:
            outcome = _exception(error, "worker")
        encoded = encode_response(outcome)
        if flags:
            encoded = encode_response(ReplayOutcome("unmodelled", phase="environment",
                                                     detail=", ".join(sorted(flags))))
    except BaseException:
        encoded = encode_response(ReplayOutcome("worker-error", phase="worker"))
    temporary = response_path + ".tmp"
    with open(temporary, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, response_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
