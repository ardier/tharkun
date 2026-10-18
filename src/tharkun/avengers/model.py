"""Value objects shared by the Avengers object-construction pipeline.

The records in this module are intentionally stdlib-only and contain importable
names plus literal values.  They can cross a process boundary without carrying
live target objects with them.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ObjectReference:
    """A recipe argument that refers to the constructed receiver."""

    name: str = "object_0"


OBJECT = ObjectReference()

# How deep one recipe may nest another.  Composition is what lets a class whose
# __init__ needs a domain object be constructed at all, but an unbounded chain
# is a way to make a worker spend arbitrary time before the boundary call it was
# asked about, so the depth is fixed here rather than left to a caller.
MAX_NESTING = 3


@dataclass(frozen=True)
class NestedObject:
    """A recipe argument that is itself a constructed object.

    The receiver of an Avengers recipe is often not enough: a class whose
    ``__init__`` takes a domain object cannot be built from literals at all.
    ``NestedObject`` carries the whole sub-recipe, so a composed program stays
    one self-contained, data-only record that a fresh worker can replay without
    reaching back into the parent for anything.
    """

    source_path: str
    class_name: str
    recipe: "ConstructionRecipe"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", os.path.realpath(self.source_path))
        if not self.class_name:
            raise ValueError("class_name must not be empty")

    @property
    def depth(self) -> int:
        """Object nesting below this one, counting itself as one level."""

        values = [*self.recipe.constructor.args, *self.recipe.constructor.keyword_arguments.values()]
        for call in self.recipe.setup:
            values.extend(call.args)
            values.extend(call.keyword_arguments.values())
        return 1 + max((item.depth for item in values if isinstance(item, NestedObject)), default=0)

    def _argument_values(self) -> list[Any]:
        values = [
            *self.recipe.constructor.args,
            *self.recipe.constructor.keyword_arguments.values(),
        ]
        for call in self.recipe.setup:
            values.extend(call.args)
            values.extend(call.keyword_arguments.values())
        return values


def _literal_record(value: Any) -> Any:
    """Return a deterministic JSON-shaped record for a recipe literal."""

    if isinstance(value, ObjectReference):
        return {"ref": value.name}
    if isinstance(value, NestedObject):
        if value.depth > MAX_NESTING:
            raise ValueError(
                f"a composed recipe may nest {MAX_NESTING} objects deep; got {value.depth}"
            )
        return {
            "nested": {
                "source_path": value.source_path,
                "class_name": value.class_name,
                "recipe": value.recipe.record,
            }
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "+inf" if value > 0 else "-inf"}
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return {"float": "-0.0"}
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return {"tuple": [_literal_record(item) for item in value]}
    if isinstance(value, list):
        return {"list": [_literal_record(item) for item in value]}
    if isinstance(value, dict):
        items = [(_literal_record(key), _literal_record(item)) for key, item in value.items()]
        items.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":")))
        return {"dict": items}
    if isinstance(value, (set, frozenset)):
        items = [_literal_record(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"frozenset" if isinstance(value, frozenset) else "set": items}
    raise TypeError(
        "Avengers recipes accept literal constructor and call values; "
        f"got {type(value).__module__}.{type(value).__qualname__}"
    )


def _literal_value(record: Any) -> Any:
    """Rebuild the literal a :func:`_literal_record` document describes.

    This is the exact inverse of the recipe literal vocabulary and it is
    deliberately closed: a document naming a tag this function does not know
    is refused rather than passed through.  Anything that reloads a recipe --
    the on-disk store, a worker response -- reads it through here, so a stored
    or received document can only ever name values the encoder could produce.
    """

    if record is None or isinstance(record, bool):
        return record
    if isinstance(record, (int, float, str)):
        return record
    if not isinstance(record, dict) or len(record) != 1:
        raise ValueError(f"not a recipe literal record: {record!r}")
    (tag, payload), = record.items()
    if tag == "ref":
        if not isinstance(payload, str):
            raise ValueError("a receiver reference names a string")
        return ObjectReference(payload)
    if tag == "nested":
        if not isinstance(payload, dict):
            raise ValueError("a nested object record is an object")
        source_path = payload.get("source_path")
        class_name = payload.get("class_name")
        if not isinstance(source_path, str) or not isinstance(class_name, str):
            raise ValueError("a nested object names its source path and class")
        return NestedObject(
            source_path,
            class_name,
            recipe_from_record(payload.get("recipe")),
        )
    if tag == "float":
        if payload == "nan":
            return float("nan")
        if payload == "+inf":
            return float("inf")
        if payload == "-inf":
            return float("-inf")
        if payload == "-0.0":
            return -0.0
        raise ValueError(f"unknown float record {payload!r}")
    if tag == "bytes":
        if not isinstance(payload, str):
            raise ValueError("a bytes record carries a hexadecimal string")
        try:
            return bytes.fromhex(payload)
        except ValueError as error:
            raise ValueError("a bytes record carries a hexadecimal string") from error
    if tag in {"tuple", "list", "set", "frozenset"}:
        if not isinstance(payload, list):
            raise ValueError(f"a {tag} record carries a list")
        items = [_literal_value(item) for item in payload]
        if tag == "tuple":
            return tuple(items)
        if tag == "list":
            return items
        return frozenset(items) if tag == "frozenset" else set(items)
    if tag == "dict":
        if not isinstance(payload, list):
            raise ValueError("a dict record carries a list of pairs")
        result: dict[Any, Any] = {}
        for pair in payload:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("a dict record carries two-element pairs")
            result[_literal_value(pair[0])] = _literal_value(pair[1])
        return result
    raise ValueError(f"unknown recipe literal tag {tag!r}")


def _kwargs(items: Mapping[str, Any] | Iterable[tuple[str, Any]] | None) -> tuple[tuple[str, Any], ...]:
    if items is None:
        return ()
    source = items.items() if isinstance(items, Mapping) else items
    result = tuple(sorted(source, key=lambda item: item[0]))
    for name, value in result:
        if not isinstance(name, str):
            raise TypeError("keyword names must be strings")
        _literal_record(value)
    return result


@dataclass(frozen=True, init=False)
class ConstructorCall:
    """Literal arguments for one class constructor call."""

    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...]

    def __init__(
        self,
        args: Iterable[Any] = (),
        kwargs: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
    ) -> None:
        positional = tuple(args)
        for value in positional:
            _literal_record(value)
        object.__setattr__(self, "args", positional)
        object.__setattr__(self, "kwargs", _kwargs(kwargs))

    @property
    def keyword_arguments(self) -> dict[str, Any]:
        return dict(self.kwargs)

    @property
    def record(self) -> dict[str, Any]:
        return {
            "args": [_literal_record(value) for value in self.args],
            "kwargs": {name: _literal_record(value) for name, value in self.kwargs},
        }

    def render(self, class_name: str, names: Mapping[str, str] | None = None) -> str:
        values = [_render_argument(value, "object_0", names) for value in self.args]
        values.extend(
            f"{name}={_render_argument(value, 'object_0', names)}"
            for name, value in self.kwargs
        )
        return f"{class_name}({', '.join(values)})"


@dataclass(frozen=True, init=False)
class MethodCall:
    """One method invocation in a setup or boundary-call sequence."""

    name: str
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...]

    def __init__(
        self,
        name: str,
        args: Iterable[Any] = (),
        kwargs: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
    ) -> None:
        if not name or not isinstance(name, str):
            raise TypeError("method name must be a non-empty string")
        positional = tuple(args)
        for value in positional:
            _literal_record(value)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "args", positional)
        object.__setattr__(self, "kwargs", _kwargs(kwargs))

    @property
    def keyword_arguments(self) -> dict[str, Any]:
        return dict(self.kwargs)

    @property
    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": [_literal_record(value) for value in self.args],
            "kwargs": {name: _literal_record(value) for name, value in self.kwargs},
        }

    def render(
        self, receiver: str = "object_0", names: Mapping[str, str] | None = None
    ) -> str:
        values = [_render_argument(value, receiver, names) for value in self.args]
        values.extend(
            f"{name}={_render_argument(value, receiver, names)}"
            for name, value in self.kwargs
        )
        return f"{receiver}.{self.name}({', '.join(values)})"


def _render_argument(
    value: Any,
    receiver: str,
    names: Mapping[str, str] | None = None,
) -> str:
    if isinstance(value, ObjectReference):
        return receiver
    if isinstance(value, NestedObject):
        # A composed object is rendered as the variable a prelude line bound
        # it to.  Without that binding there is no runnable spelling for it,
        # so falling back to repr() would emit a program nobody can execute.
        if names is not None and _nested_identity(value) in names:
            return names[_nested_identity(value)]
        return value.recipe.render(value.class_name).splitlines()[-1]
    if isinstance(value, tuple):
        body = ", ".join(_render_argument(item, receiver, names) for item in value)
        return f"({body}{',' if len(value) == 1 else ''})"
    if isinstance(value, list):
        return f"[{', '.join(_render_argument(item, receiver, names) for item in value)}]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{_render_argument(key, receiver, names)}: "
            f"{_render_argument(item, receiver, names)}"
            for key, item in value.items()
        ) + "}"
    return repr(value)


def _nested_identity(value: "NestedObject") -> str:
    return f"{value.source_path}::{value.class_name}::{value.recipe.key}"


def _bind_nested(values: Iterable[Any]) -> tuple[dict[str, str], list[str]]:
    """Name every composed object and emit the lines that build them.

    Deepest first, so a nested object that itself needs one is already bound
    by the time its own line is written.  Identical sub-recipes share a
    binding: building the same object twice would be a different program.
    """

    names: dict[str, str] = {}
    lines: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, NestedObject):
            if _nested_identity(value) in names:
                return
            for item in value._argument_values():
                visit(item)
            name = f"nested_{len(names)}"
            names[_nested_identity(value)] = name
            body = value.recipe.constructor
            rendered = ", ".join(
                [_render_argument(item, name, names) for item in body.args]
                + [
                    f"{key}={_render_argument(item, name, names)}"
                    for key, item in body.kwargs
                ]
            )
            lines.append(f"{name} = {value.class_name}({rendered})")
            lines.extend(call.render(name, names) for call in value.recipe.setup)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)

    for value in values:
        visit(value)
    return names, lines


@dataclass(frozen=True, init=False)
class FunctionCall:
    """A free/module function call whose arguments may reference the object."""

    target: str
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...]

    def __init__(
        self,
        target: str,
        args: Iterable[Any] = (OBJECT,),
        kwargs: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
    ) -> None:
        if not target or not isinstance(target, str):
            raise TypeError("function target must be a non-empty name")
        positional = tuple(args)
        for value in positional:
            _literal_record(value)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "args", positional)
        object.__setattr__(self, "kwargs", _kwargs(kwargs))

    @property
    def keyword_arguments(self) -> dict[str, Any]:
        return dict(self.kwargs)

    @property
    def record(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "args": [_literal_record(value) for value in self.args],
            "kwargs": {name: _literal_record(value) for name, value in self.kwargs},
        }

    def render(
        self, receiver: str = "object_0", names: Mapping[str, str] | None = None
    ) -> str:
        values = [_render_argument(value, receiver, names) for value in self.args]
        values.extend(
            f"{name}={_render_argument(value, receiver, names)}"
            for name, value in self.kwargs
        )
        return f"{self.target}({', '.join(values)})"


Call = MethodCall | FunctionCall


#: What a goal asks the boundary to do.  ``returns`` is the default and the
#: original meaning.  The rest are boundary-DIRECTED: they ask the search to
#: find an object that makes the boundary misbehave in a named way, which is
#: the inverse problem -- given a behaviour, produce the program that reaches
#: it -- rather than a failure to be avoided.
EXPECTATIONS = frozenset({"returns", "raises", "crashes"})


def _parse_expectation(value: str) -> tuple[str, str | None]:
    """Split ``"raises:ValueError"`` into its kind and the type it names."""

    if not isinstance(value, str) or not value:
        raise TypeError("expect must be a non-empty string")
    kind, _, detail = value.partition(":")
    if kind not in EXPECTATIONS:
        raise ValueError(
            f"expect must name one of {sorted(EXPECTATIONS)}; got {value!r}"
        )
    if detail and kind != "raises":
        raise ValueError(f"only 'raises' takes an exception name; got {value!r}")
    return kind, detail or None


_FIELD_BOUND_OPS = frozenset({"lt", "le", "gt", "ge", "eq", "ne", "truthy", "falsy"})


@dataclass(frozen=True)
class FieldBound:
    """One requirement on a constructed object's ``<attr>`` ([internal ref] slice 3,
    [internal ref]): a comparison against a literal, or truthiness.

    A closed, safe vocabulary -- never evaluated code -- so a caller can pass
    a bound sourced from anywhere (a guard's own literal, an absint/
    guard_domain fact, a contract) without this becoming an arbitrary-code
    sink. See :attr:`ObjectGoal.field_bounds` for how it directs the search,
    and :func:`field_bounds_hold` for how it is checked against a real
    object.
    """

    attr: str
    op: str
    value: Any = None

    def __post_init__(self) -> None:
        if self.op not in _FIELD_BOUND_OPS:
            raise ValueError(f"op must be one of {sorted(_FIELD_BOUND_OPS)}; got {self.op!r}")


def materialize_recipe(recipe: "ConstructionRecipe", class_name: str,
                       module_globals: "dict[str, Any]") -> Any:
    """Replay *recipe*'s literal construction in THIS process, so a caller
    gets a live object rather than a description ([internal ref] slice 3).

    ``recipe.render(class_name)`` is this module's own executable-text
    rendering of a recipe -- every value in it is a checked JSON-safe literal
    (:func:`_literal_record`), never arbitrary code. This replays a recipe
    already run once by a disposable child (``IsolatedExecutor.replay``); a
    SECOND, in-process replay is what hands a live object back, since a
    disposable child's result cannot cross the process boundary as anything
    but the serializable :class:`ReplayOutcome`.
    """

    text = recipe.render(class_name)
    if "object_0" not in text:
        # No setup calls and no composed (nested) arguments: render() returns
        # the bare constructor expression, not an assignment (it is meant as
        # a standalone display/backend snippet, not always bound to a name).
        text = f"object_0 = {text}"
    ns = dict(module_globals)
    exec(compile(text, "<avengers-recipe-replay>", "exec"), ns)
    return ns.get("object_0")


def field_bounds_hold(obj: Any, bounds: "tuple[FieldBound, ...]") -> bool:
    """Whether every bound in *bounds* holds on *obj*, read via plain
    ``getattr`` -- never evaluated code, so a malformed bound can only make
    this return False, never raise out to its caller."""

    for b in bounds:
        try:
            v = getattr(obj, b.attr)
        except AttributeError:
            return False
        try:
            if b.op == "truthy":
                ok = bool(v)
            elif b.op == "falsy":
                ok = not bool(v)
            elif b.op == "eq":
                ok = v == b.value
            elif b.op == "ne":
                ok = v != b.value
            elif b.op == "lt":
                ok = v < b.value
            elif b.op == "le":
                ok = v <= b.value
            elif b.op == "gt":
                ok = v > b.value
            elif b.op == "ge":
                ok = v >= b.value
            else:
                ok = False
        except TypeError:
            return False
        if not ok:
            return False
    return True


@dataclass(frozen=True)
class ObjectGoal:
    """A class to construct, boundary calls, and what they should do.

    By default the calls must return normally.  ``expect`` changes that:
    ``"raises"``, ``"raises:ValueError"`` (or any suffix of the exception's
    dotted name) and ``"crashes"`` each ask for an object that drives the
    boundary into that behaviour instead.

    ``field_bounds`` ([internal ref] slice 3) directs the search toward a
    CONSTRUCTION-TIME state instead of a boundary-call outcome -- "a C whose
    .x > 3 and .items is non-empty" (the owner's own example, [internal ref]).
    Unlike ``expect``, which :meth:`satisfied` decides from a
    :class:`ReplayOutcome` alone (deliberately -- see that method's
    docstring: a worker's report must be enough, so one result can serve
    opposite goals), a field bound needs the constructed object ITSELF, which
    does not cross the disposable-child boundary a ``ReplayOutcome`` reports
    from. So ``field_bounds`` is checked with :func:`field_bounds_hold`
    against a live object a caller materializes in its OWN process (see
    ``driver.py::_avengers_object_for_bounds`` / ``HypothesisCandidatePass``'s
    search predicate) -- ``satisfied()`` keeps its existing, outcome-only
    contract unchanged; this is an ADDITIONAL check a caller with a live
    object applies alongside it, never a replacement for it.
    """

    source_path: str
    class_name: str
    calls: tuple[Call, ...] = ()
    expect: str = "returns"
    field_bounds: tuple[FieldBound, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", os.path.realpath(self.source_path))
        object.__setattr__(self, "calls", tuple(self.calls))
        object.__setattr__(self, "field_bounds", tuple(self.field_bounds))
        if not self.class_name:
            raise ValueError("class_name must not be empty")
        _parse_expectation(self.expect)
        if self.expect != "returns" and not self.calls:
            raise ValueError("a directed expectation needs a boundary call to direct")

    @classmethod
    def from_type(
        cls,
        target: type[Any],
        calls: Iterable[Call] = (),
        expect: str = "returns",
    ) -> "ObjectGoal":
        """Describe an importable class without manually locating its source."""

        if not isinstance(target, type):
            raise TypeError("target must be a class")
        source_path = inspect.getsourcefile(target) or inspect.getfile(target)
        class_name = target.__qualname__
        if "<locals>" in class_name:
            raise ValueError("target must be defined at module or class scope")
        return cls(source_path, class_name, tuple(calls), expect)

    def satisfied(self, outcome: "ReplayOutcome") -> bool:
        """Whether a replay did what this goal asked for.

        Decided in the parent, never in the worker: the worker reports what
        happened and this decides whether that is the answer, so one child
        result can serve goals that want opposite things from it.
        """

        kind, wanted = _parse_expectation(self.expect)
        if kind == "returns":
            return outcome.succeeded
        # A directed expectation is about the BOUNDARY. A constructor that
        # raises has not produced an object that misbehaves; it has failed to
        # produce an object, which is a different (and uninteresting) result.
        at_boundary = (outcome.phase or "").startswith("boundary")
        if kind == "raises":
            if outcome.kind != "raised" or not at_boundary:
                return False
            if wanted is None:
                return True
            name = outcome.exception_type or ""
            return name == wanted or name.endswith("." + wanted)
        # "crashes": the interpreter went down. The phase is the worker's,
        # because a native crash leaves no Python frame to attribute it to.
        return outcome.kind in {"crashed", "killed"}

    @property
    def key(self) -> str:
        body: dict[str, Any] = {
            "source_path": self.source_path,
            "class_name": self.class_name,
            "calls": [call.record for call in self.calls],
        }
        if self.expect != "returns":
            # Only when it is not the default, so every key minted before
            # directed goals existed still names the same goal.
            body["expect"] = self.expect
        if self.field_bounds:
            # Same reasoning: only when non-empty, so every key minted before
            # field bounds existed still names the same goal.
            body["field_bounds"] = [
                {"attr": b.attr, "op": b.op, "value": b.value} for b in self.field_bounds
            ]
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]


@dataclass(frozen=True)
class SourceArtifact:
    """Generated Python source offered as evidence by another tool."""

    tool: str
    source: str

    @property
    def key(self) -> str:
        return f"{self.tool}:{hashlib.sha256(self.source.encode('utf-8')).hexdigest()[:24]}"


@dataclass(frozen=True)
class ConstructionRecipe:
    """A constructor plus calls that prepare a fresh receiver state.

    Producer names deliberately do not live here.  The KnowledgeBase stores
    those as derivations, so the same program discovered by Pynguin and
    Hypothesis is replayed once while crediting both.
    """

    goal_key: str
    constructor: ConstructorCall = field(default_factory=ConstructorCall)
    setup: tuple[Call, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "setup", tuple(self.setup))

    @property
    def record(self) -> dict[str, Any]:
        return {
            "goal": self.goal_key,
            "constructor": self.constructor.record,
            "setup": [call.record for call in self.setup],
        }

    @property
    def key(self) -> str:
        return hashlib.sha256(
            json.dumps(self.record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]

    def render(self, class_name: str) -> str:
        """The runnable program that reaches this receiver state.

        Composed arguments are bound to ``nested_N`` variables above the
        constructor, so the whole thing is executable text rather than a
        description of one.
        """

        names, prelude = _bind_nested(self._argument_values())
        constructor = self.constructor.render(class_name, names)
        if not self.setup and not prelude:
            return constructor
        lines = [*prelude, f"object_0 = {constructor}"]
        lines.extend(call.render("object_0", names) for call in self.setup)
        return "\n".join(lines)

    def _argument_values(self) -> list[Any]:
        values = [*self.constructor.args, *self.constructor.keyword_arguments.values()]
        for call in self.setup:
            values.extend(call.args)
            values.extend(call.keyword_arguments.values())
        return values


@dataclass(frozen=True)
class ReplayOutcome:
    """What happened when a recipe and its boundary calls ran in a child."""

    kind: str
    phase: str | None = None
    detail: str | None = None
    exception_type: str | None = None
    signal: int | None = None
    return_types: tuple[str, ...] = ()
    state_digest: str | None = None
    # Lines of the target file the setup and boundary calls executed, and the
    # (from, to) transitions between them.  Empty means "not measured": the
    # worker only traces when it is asked to, because tracing costs roughly a
    # third of the replay.
    lines: tuple[int, ...] = ()
    arcs: tuple[tuple[int, int], ...] = ()
    # Whether a second, independent replay reached the same outcome.  ``None``
    # means nobody checked.  A crash that does not reproduce is not a finding,
    # so counting one would overstate what was found.
    reproduced: bool | None = None

    @property
    def succeeded(self) -> bool:
        return self.kind == "returned"

    @property
    def crash_cause(self) -> tuple[str, str] | None:
        """The (last call, terminating condition) pair this outcome groups by.

        Following Berg et al. (arXiv:2603.06107), crashes are counted by cause
        rather than by occurrence: many recipes hit one bug, and reporting one
        number per recipe would inflate it.  A normal return has no cause.
        """

        if self.succeeded:
            return None
        where = self.phase or "unknown"
        if self.signal is not None:
            return (where, f"signal:{self.signal}")
        if self.exception_type is not None:
            return (where, self.exception_type)
        return (where, self.kind)


@dataclass(frozen=True)
class ValidatedRecipe:
    """A construction recipe promoted by fresh-process replay."""

    goal: ObjectGoal
    recipe: ConstructionRecipe
    outcome: ReplayOutcome

    @property
    def program(self) -> str:
        names, _prelude = _bind_nested(self.recipe._argument_values())
        lines = self.recipe.render(self.goal.class_name).splitlines()
        # ``render`` returns a bare constructor expression only when there is
        # nothing else to write -- no setup and no composed arguments.  Every
        # other shape already binds ``object_0`` on its own line.
        if len(lines) == 1:
            lines[0] = f"object_0 = {lines[0]}"
        lines.extend(call.render("object_0", names) for call in self.goal.calls)
        return "\n".join(lines)


@dataclass(frozen=True)
class MethodCoverage:
    """Independent fresh-process outcomes for one constructed object state."""

    validated: ValidatedRecipe
    outcomes: tuple[tuple[MethodCall, ReplayOutcome], ...]

    @property
    def returned(self) -> tuple[MethodCall, ...]:
        return tuple(call for call, outcome in self.outcomes if outcome.succeeded)

    @property
    def rejected(self) -> tuple[MethodCall, ...]:
        return tuple(call for call, outcome in self.outcomes if not outcome.succeeded)


@dataclass(frozen=True)
class AvengersReport:
    """The fixed point reached for one or more object goals."""

    validated: tuple[ValidatedRecipe, ...]
    stop_reason: str
    pass_runs: Mapping[str, int]
    evidence_count: int
    unresolved: tuple[ObjectGoal, ...] = ()
    failures: tuple[ReplayOutcome, ...] = ()
    method_coverage: tuple[MethodCoverage, ...] = ()
    method_universe: tuple[str, ...] = ()
    coverage_complete: bool = True
    # Wall time from the start of the search to the first replay-validated
    # recipe.  ``None`` means no object was constructed at all, which is the
    # only honest reading: a search that never succeeded has no such time.
    time_to_first_object_s: float | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    # Distinct receiver states, call sequences the target rejected on purpose,
    # and what the validated recipes covered. Empty means the objective did not
    # run, which is not the same as "it ran and found none".
    states: tuple[Any, ...] = ()
    negatives: tuple[Any, ...] = ()
    coverage: Any = None

    @property
    def line_coverage(self) -> float | None:
        return None if self.coverage is None else self.coverage.line_coverage

    @property
    def branch_coverage(self) -> float | None:
        return None if self.coverage is None else self.coverage.branch_coverage

    @property
    def crash_causes(self) -> tuple[tuple[str, str], ...]:
        """Distinct (where, why) pairs among the failures, deduplicated.

        Counting failures rather than causes reports the size of the search,
        not the number of things wrong with the target.
        """

        causes = {item.crash_cause for item in self.failures if item.crash_cause}
        return tuple(sorted(causes))

    @staticmethod
    def _recipe_cost(item: ValidatedRecipe) -> tuple[int, int, str]:
        return (
            len(item.recipe.setup),
            len(item.recipe.constructor.args) + len(item.recipe.constructor.kwargs),
            item.recipe.key,
        )

    @property
    def best(self) -> ValidatedRecipe | None:
        if self.method_coverage:
            return min(
                self.method_coverage,
                key=lambda item: (
                    -len(item.returned),
                    *self._recipe_cost(item.validated),
                ),
            ).validated
        if not self.validated:
            return None
        return min(self.validated, key=self._recipe_cost)

    @property
    def covered_methods(self) -> tuple[str, ...]:
        return tuple(
            sorted({call.name for item in self.method_coverage for call in item.returned})
        )

    @property
    def objects(self) -> tuple[ValidatedRecipe, ...]:
        """Greedily select a small recipe portfolio with maximal method cover."""

        if not self.method_coverage:
            return (self.best,) if self.best is not None else ()
        uncovered = set(self.covered_methods)
        candidates = list(self.method_coverage)
        selected: list[ValidatedRecipe] = []
        while uncovered:
            useful = [
                (len({call.name for call in item.returned} & uncovered), item)
                for item in candidates
            ]
            gain, chosen = min(
                useful,
                key=lambda pair: (
                    -pair[0],
                    *self._recipe_cost(pair[1].validated),
                ),
            )
            if gain == 0:
                break
            selected.append(chosen.validated)
            uncovered.difference_update(call.name for call in chosen.returned)
            candidates.remove(chosen)
        return tuple(selected)


def call_from_record(record: Any) -> "Call":
    """Rebuild one boundary or setup call from its document.

    ``name`` selects a method call and ``target`` a module function; the two
    are disjoint, so a document that carries both or neither is refused.
    """

    if not isinstance(record, dict):
        raise ValueError("a call record is an object")
    has_name = "name" in record
    has_target = "target" in record
    if has_name == has_target:
        raise ValueError("a call record names exactly one of name or target")
    args = record.get("args", [])
    kwargs = record.get("kwargs", {})
    if not isinstance(args, list):
        raise ValueError("call args are a list")
    if not isinstance(kwargs, dict):
        raise ValueError("call kwargs are an object")
    values = [_literal_value(item) for item in args]
    named = {name: _literal_value(item) for name, item in kwargs.items()}
    if has_name:
        return MethodCall(record["name"], values, named)
    return FunctionCall(record["target"], values, named)


def constructor_from_record(record: Any) -> ConstructorCall:
    """Rebuild a constructor call from its document."""

    if not isinstance(record, dict):
        raise ValueError("a constructor record is an object")
    args = record.get("args", [])
    kwargs = record.get("kwargs", {})
    if not isinstance(args, list):
        raise ValueError("constructor args are a list")
    if not isinstance(kwargs, dict):
        raise ValueError("constructor kwargs are an object")
    return ConstructorCall(
        [_literal_value(item) for item in args],
        {name: _literal_value(item) for name, item in kwargs.items()},
    )


def recipe_from_record(record: Any, *, goal_key: str | None = None) -> ConstructionRecipe:
    """Rebuild a construction recipe from its document.

    ``goal_key`` overrides the stored key so a recipe learned for one goal can
    be replayed against another -- the same constructor and setup reaching a
    different boundary.  That re-keying is the whole point of keeping recipes
    around between runs.
    """

    if not isinstance(record, dict):
        raise ValueError("a recipe record is an object")
    setup = record.get("setup", [])
    if not isinstance(setup, list):
        raise ValueError("recipe setup is a list")
    stored_key = record.get("goal")
    key = goal_key if goal_key is not None else stored_key
    if not isinstance(key, str) or not key:
        raise ValueError("a recipe record names its goal")
    return ConstructionRecipe(
        key,
        constructor_from_record(record.get("constructor", {})),
        tuple(call_from_record(item) for item in setup),
    )
