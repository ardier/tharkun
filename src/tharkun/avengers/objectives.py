"""Objectives above "the object constructs": state, coverage, and bad sequences.

The default portfolio stops once a recipe replays cleanly, and then probes
public methods.  That answers "can this class be built" and "what can be called
on it", but not the three questions the object-synthesis goal actually asks:

* **state coverage** -- how many genuinely different receiver states were
  reached, not how many recipes happened to validate;
* **line and branch coverage** -- how far into the class the search got, and
  which constructor arguments would get it further;
* **deliberately invalid sequences** -- ``start(); start()`` is a result, not a
  failure, and today it is thrown away as one.

Each objective is an ordinary pass, so it competes for the portfolio's budget
like everything else and its proposals are certified by the same fresh-process
replay.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from tharkun.avengers.engine import Emission, RepeatPolicy
from tharkun.avengers.executor import IsolatedExecutor
from tharkun.avengers.model import (
    ConstructionRecipe,
    MethodCall,
    ObjectGoal,
    ReplayOutcome,
    ValidatedRecipe,
)
from tharkun.avengers.passes import (
    CANDIDATE,
    VALIDATED,
    _baseline_value,
    _call_from_values,
    _class_node,
    public_method_calls,
)
from tharkun.knowledge import Evidence, KnowledgeBase

STATE = "receiver-state"
COVERAGE = "coverage-observation"
NEGATIVE = "negative-sequence"


# --------------------------------------------------------------------------
# State coverage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReceiverState:
    """One distinct receiver state, and the cheapest recipe that reached it."""

    goal: ObjectGoal
    recipe: ConstructionRecipe
    digest: str
    #: Whether replaying the same recipe reached the same digest again.  An
    #: unstable state is still a state; its DIGEST just cannot name it, so the
    #: recipe does instead.
    stable: bool = True

    @property
    def key(self) -> str:
        """What this state is counted as.

        A stable digest is the identity: two recipes reaching it are one state.
        An unstable one is not -- ``Timer(); start()`` stores a fresh
        ``perf_counter()`` reading, so every replay of the SAME recipe digests
        differently and counting digests would report a new state per draw.
        The recipe is the honest fallback: one program, one state.
        """

        return self.digest if self.stable else f"unstable:{self.recipe.key}"

    @property
    def program(self) -> str:
        return self.recipe.render(self.goal.class_name)


@dataclass(frozen=True)
class StateDiversityPass:
    """Chase receiver states nobody has seen, not recipes nobody has tried.

    Two recipes that reach the same state are one result; a recipe that reaches
    a state no other did is a new one however ordinary it looks.  The pass
    therefore keys on the replay's state digest -- but only after checking that
    the digest REPRODUCES.  A receiver holding a timestamp digests differently
    on every replay of the same recipe, and counting those as new states each
    time reports a state space that does not exist, and sends the search
    chasing its own tail extending them.
    """

    executor: IsolatedExecutor
    max_steps: int = 2
    max_probes: int = 64
    name: str = "state-diversity"
    consumes: frozenset[str] = frozenset({VALIDATED})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        known = {
            item.value.key
            for item in knowledge.evidence(STATE)
            if isinstance(item.value, ReceiverState)
        }
        fresh: list[ValidatedRecipe] = []
        for item in delta:
            validated = item.value
            if not isinstance(validated, ValidatedRecipe):
                continue
            if validated.outcome.state_digest is None:
                # Without a digest there is no state to compare, so silently
                # treating every recipe as a new state would inflate the count.
                continue
            fresh.append(validated)
        if not fresh:
            return

        fresh = fresh[: self.max_probes]
        # One confirming replay per candidate state, batched. A digest that
        # does not survive its own repetition cannot identify anything.
        confirmations = self.executor.replay_many(
            [(item.goal, item.recipe) for item in fresh]
        )
        proposals: list[tuple[ObjectGoal, ConstructionRecipe]] = []
        for validated, again in zip(fresh, confirmations):
            digest = validated.outcome.state_digest
            state = ReceiverState(
                validated.goal,
                validated.recipe,
                digest,
                stable=again.succeeded and again.state_digest == digest,
            )
            if state.key in known:
                continue
            known.add(state.key)
            yield Emission(STATE, state.key, state)
            proposals.extend(self._extensions(validated))

        if not proposals:
            return
        proposals = proposals[: self.max_probes]
        # Extensions are proposed on their own merit -- a longer program that
        # runs is worth replaying -- rather than on their digest being novel.
        # Judging them by novelty would let an unstable digest look new every
        # time and flood the portfolio with the same program.
        outcomes = self.executor.replay_many(proposals)
        for (_goal, recipe), outcome in zip(proposals, outcomes):
            if outcome.succeeded:
                yield Emission(CANDIDATE, recipe.key, recipe)

    def _extensions(
        self, validated: ValidatedRecipe
    ) -> list[tuple[ObjectGoal, ConstructionRecipe]]:
        if len(validated.recipe.setup) >= self.max_steps:
            return []
        bare = ObjectGoal(validated.goal.source_path, validated.goal.class_name)
        return [
            (
                bare,
                ConstructionRecipe(
                    validated.goal.key,
                    validated.recipe.constructor,
                    (*validated.recipe.setup, call),
                ),
            )
            for call in public_method_calls(bare)
        ]


# --------------------------------------------------------------------------
# Line and branch coverage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageObservation:
    """What one class's validated recipes reached, together."""

    goal: ObjectGoal
    lines: tuple[int, ...]
    arcs: tuple[tuple[int, int], ...]
    class_lines: tuple[int, ...]
    class_arcs: tuple[tuple[int, int], ...]

    @property
    def key(self) -> str:
        """Content-addressed, because coverage GROWS as recipes validate.

        Keying on the goal alone would make the second, larger observation a
        different value under the same evidence identity, which the knowledge
        base refuses -- correctly, since that identity is supposed to mean one
        thing.  Each distinct reach is its own item and the report takes the
        widest.
        """

        body = json.dumps(
            {"goal": self.goal.key, "lines": self.lines, "arcs": self.arcs},
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"{self.goal.key}:{hashlib.sha256(body.encode()).hexdigest()[:16]}"

    @property
    def reach(self) -> int:
        """How much was covered -- the ordering the report picks the widest by."""

        return len(set(self.arcs) & set(self.class_arcs)) * 10_000 + len(
            set(self.lines) & set(self.class_lines)
        )

    @property
    def line_coverage(self) -> float:
        if not self.class_lines:
            return 0.0
        return len(set(self.lines) & set(self.class_lines)) / len(self.class_lines)

    @property
    def branch_coverage(self) -> float:
        if not self.class_arcs:
            return 0.0
        return len(set(self.arcs) & set(self.class_arcs)) / len(self.class_arcs)

    @property
    def uncovered_arcs(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(set(self.class_arcs) - set(self.arcs)))


def class_line_span(source_path: str, class_name: str) -> tuple[int, int] | None:
    """The first and last line of a class body, or ``None`` if not found."""

    node = _class_node(source_path, class_name)
    if node is None:
        return None
    end = getattr(node, "end_lineno", None)
    if end is None:
        end = max((getattr(child, "lineno", node.lineno) for child in ast.walk(node)))
    return (node.lineno, int(end))


def class_branch_arcs(source_path: str, class_name: str) -> tuple[tuple[int, int], ...]:
    """Both arcs out of every branch inside a class body.

    The denominator for branch coverage.  Arcs are ``(guard line, first line of
    the branch body)`` for the taken side and ``(guard line, first line after
    the statement)`` for the missed side, which is the shape the tracer records.
    """

    span = class_line_span(source_path, class_name)
    if span is None:
        return ()
    try:
        tree = ast.parse(open(source_path, encoding="utf-8", errors="ignore").read())
    except (OSError, SyntaxError, ValueError):
        return ()
    arcs: set[tuple[int, int]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            continue
        if not span[0] <= node.lineno <= span[1]:
            continue
        body = getattr(node, "body", None)
        orelse = getattr(node, "orelse", None)
        if body:
            arcs.add((node.lineno, body[0].lineno))
        if orelse:
            arcs.add((node.lineno, orelse[0].lineno))
        else:
            following = _next_line_after(tree, node)
            if following is not None:
                arcs.add((node.lineno, following))
    return tuple(sorted(arcs))


def _next_line_after(tree: ast.AST, node: ast.stmt) -> int | None:
    """The line a statement falls through to when its branch is not taken."""

    for parent in ast.walk(tree):
        for field_name in ("body", "orelse", "finalbody"):
            block = getattr(parent, field_name, None)
            if not isinstance(block, list) or node not in block:
                continue
            index = block.index(node)
            if index + 1 < len(block):
                return int(block[index + 1].lineno)
            return None
    return None


#: Returned when no literal can satisfy a guard.  Distinct from ``None``,
#: which is itself a perfectly good literal: ``logger=None`` is exactly what
#: the false side of ``if self.logger:`` needs, and conflating the two loses
#: that branch.
NO_LITERAL = object()


def _truthy_value(annotation: str | None) -> Any:
    """A literal that satisfies ``if <parameter>:`` for this annotation.

    :data:`NO_LITERAL` means nothing will do -- most importantly for a callable
    parameter, where the only truthy values are functions.  Proposing ``True``
    there would satisfy the guard and then crash on the call below it, which is
    worse than not proposing anything.
    """

    if annotation is None:
        return "x"
    text = annotation.replace("typing.", "").strip()
    if "Callable" in text or "FloatArg" in text:
        return NO_LITERAL
    for prefix, value in (
        ("bool", True),
        ("int", 1),
        ("float", 1.0),
        ("str", "x"),
        ("bytes", b"x"),
        ("list", [0]),
        ("List", [0]),
        ("dict", {"k": 0}),
        ("Dict", {"k": 0}),
        ("Mapping", {"k": 0}),
        ("set", {0}),
        ("Set", {0}),
        ("tuple", (0,)),
        ("Tuple", (0,)),
    ):
        if text.startswith(prefix):
            return value
    inner = _unwrap_optional(text)
    if inner is not None and inner != text:
        return _truthy_value(inner)
    if "str" in text:
        return "x"
    if "int" in text:
        return 1
    return NO_LITERAL


def _falsy_value(annotation: str | None) -> Any:
    """A literal that fails ``if <parameter>:`` for this annotation.

    ``None`` for anything optional -- including a callable, where ``None`` is
    the one falsy value a caller can actually write -- and the annotation's own
    zero otherwise.
    """

    if annotation is None:
        return None
    text = annotation.replace("typing.", "").strip()
    if "None" in text or text.startswith("Optional["):
        return None
    if "Callable" in text or "FloatArg" in text:
        return NO_LITERAL
    baseline = _baseline_value(annotation)
    return baseline if baseline is not None else NO_LITERAL


def _unwrap_optional(text: str) -> str | None:
    if text.startswith("Optional[") and text.endswith("]"):
        return text[len("Optional[") : -1]
    if "|" in text:
        parts = [part.strip() for part in text.split("|") if part.strip() != "None"]
        if len(parts) == 1:
            return parts[0]
    if text.startswith("Union[") and text.endswith("]"):
        parts = [part.strip() for part in text[len("Union[") : -1].split(",")]
        concrete = [part for part in parts if part != "None"]
        if concrete:
            return concrete[0]
    return None


def _self_attributes(source_path: str, lineno: int) -> tuple[str, ...]:
    """``self.X`` names the guard at *lineno* reads."""

    from tharkun.solve.guard import find_branch_at_line

    try:
        tree = ast.parse(open(source_path, encoding="utf-8", errors="ignore").read())
    except (OSError, SyntaxError, ValueError):
        return ()
    guard = find_branch_at_line(tree, lineno)
    if guard is None:
        return ()
    found: list[str] = []
    for node in ast.walk(guard):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr not in found
        ):
            found.append(node.attr)
    return tuple(found)


@dataclass(frozen=True)
class CoverageSetupPass:
    """Turn an uncovered branch into the constructor argument that reaches it.

    Two ``Timer()`` recipes may both validate and still never enter ``stop()``'s
    reporting branches, because those are gated on constructor arguments nobody
    varied.  This pass reads the guards it has not covered, and where one tests
    an attribute that IS a constructor parameter, proposes the truthy and falsy
    literal for that parameter's annotation.

    Where no literal will do -- ``callable(self.text)`` is the standing example,
    since only a function satisfies it -- the pass proposes nothing rather than
    a value that passes the guard and crashes on the line below it.
    """

    executor: IsolatedExecutor
    max_candidates: int = 32
    name: str = "coverage-setup"
    consumes: frozenset[str] = frozenset({VALIDATED})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        from tharkun.solve.frontend.constructor import constructor_template

        by_goal: dict[str, list[ValidatedRecipe]] = {}
        for item in knowledge.evidence(VALIDATED):
            if isinstance(item.value, ValidatedRecipe):
                by_goal.setdefault(item.value.goal.key, []).append(item.value)

        observed: list[tuple[str, list[ValidatedRecipe], CoverageObservation]] = []
        for key, recipes in by_goal.items():
            observation = observe_coverage(recipes[0].goal, recipes)
            if observation is not None:
                observed.append((key, recipes, observation))
        # Every observation is published before any candidate, so the candidate
        # ceiling below can never cost a goal its coverage report.
        for _key, _recipes, observation in observed:
            yield Emission(COVERAGE, observation.key, observation)

        for key, recipes, observation in observed:
            goal = recipes[0].goal
            template = constructor_template(goal.source_path, goal.class_name)
            if template is None:
                continue
            parameters = {
                parameter.name: parameter
                for parameter in template.parameters
                if parameter.kind not in {"var_positional", "var_keyword"}
            }
            proposed = 0
            for guard_line in sorted({start for start, _end in observation.uncovered_arcs}):
                for attribute in _self_attributes(goal.source_path, guard_line):
                    parameter = parameters.get(attribute)
                    if parameter is None:
                        continue
                    for value in (
                        _truthy_value(parameter.annotation),
                        _falsy_value(parameter.annotation),
                    ):
                        if value is NO_LITERAL:
                            continue
                        recipe = ConstructionRecipe(
                            key,
                            _call_from_values(template, {attribute: value}),
                            recipes[0].recipe.setup,
                        )
                        proposed += 1
                        yield Emission(CANDIDATE, recipe.key, recipe)
                        if proposed >= self.max_candidates:
                            return


def observe_coverage(
    goal: ObjectGoal, recipes: Iterable[ValidatedRecipe]
) -> CoverageObservation | None:
    """Union what a class's validated recipes executed, against its own body."""

    span = class_line_span(goal.source_path, goal.class_name)
    if span is None:
        return None
    lines: set[int] = set()
    arcs: set[tuple[int, int]] = set()
    measured = False
    for validated in recipes:
        if validated.outcome.lines or validated.outcome.arcs:
            measured = True
        lines.update(validated.outcome.lines)
        arcs.update(validated.outcome.arcs)
    if not measured:
        # Nothing was traced. Reporting 0% would read as "reached nothing".
        return None
    class_arcs = class_branch_arcs(goal.source_path, goal.class_name)
    executable = tuple(
        line for line in sorted(_executable_lines(goal.source_path)) if span[0] <= line <= span[1]
    )
    return CoverageObservation(
        goal=goal,
        lines=tuple(sorted(lines)),
        arcs=tuple(sorted(arcs)),
        class_lines=executable,
        class_arcs=class_arcs,
    )


def _executable_lines(source_path: str) -> set[int]:
    """Lines a tracer can report, taken from the module's own code objects."""

    try:
        source = open(source_path, encoding="utf-8", errors="ignore").read()
        code = compile(source, source_path, "exec")
    except (OSError, SyntaxError, ValueError):
        return set()
    found: set[int] = set()
    stack = [code]
    while stack:
        current = stack.pop()
        for _start, _end, line in current.co_lines():
            if line:
                found.add(int(line))
        stack.extend(
            constant for constant in current.co_consts if hasattr(constant, "co_lines")
        )
    return found


# --------------------------------------------------------------------------
# Deliberately invalid sequences
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NegativeSequence:
    """A call sequence that raised, and the recipe it raised from."""

    goal: ObjectGoal
    recipe: ConstructionRecipe
    calls: tuple[MethodCall, ...]
    outcome: ReplayOutcome
    shape: str

    @property
    def exception_type(self) -> str:
        return self.outcome.exception_type or ""

    @property
    def program(self) -> str:
        lines = self.recipe.render(self.goal.class_name).splitlines()
        if len(lines) == 1:
            lines[0] = f"object_0 = {lines[0]}"
        lines.extend(call.render() for call in self.calls)
        return "\n".join(lines)


def target_exceptions(source_path: str) -> frozenset[str]:
    """Exception names the target defines or raises by name.

    A ``TypeError`` from calling a method with the wrong arity is an artefact
    of the probe, not a property of the class.  A ``TypeError`` the class
    raises itself is a real precondition.  Both are spelled the same way in an
    outcome, so the module's own source decides which one this is: a class it
    defines, or a name it explicitly raises.
    """

    try:
        tree = ast.parse(open(source_path, encoding="utf-8", errors="ignore").read())
    except (OSError, SyntaxError, ValueError):
        return frozenset()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Raise) and node.exc is not None:
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return frozenset(names)


@dataclass(frozen=True)
class NegativeSequencePass:
    """Look for call sequences that a correct client would be wrong to write.

    ``start(); start()`` and a bare ``stop()`` are the canonical shapes: both
    are rejected by the class on purpose, and both are results.  The portfolio
    already produces such outcomes as failure evidence and discards them; this
    pass produces them deliberately and keeps them.

    Only exceptions the target itself defines or raises by name are kept, so a
    ``TypeError`` from probing a method with the wrong arity does not become a
    finding.  Every kept sequence is replayed a second time in its own child
    before it is published, which is both the isolation the batched scan traded
    away and the reproducibility check a crash report needs.
    """

    executor: IsolatedExecutor
    max_probes: int = 64
    name: str = "negative-sequences"
    consumes: frozenset[str] = frozenset({VALIDATED})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        seen = {item.key for item in knowledge.evidence(NEGATIVE)}
        probes: list[tuple[ObjectGoal, ConstructionRecipe, str]] = []
        for item in delta:
            validated = item.value
            if isinstance(validated, ValidatedRecipe):
                probes.extend(self._probes(validated))
        if not probes:
            return
        probes = probes[: self.max_probes]

        allowed = {
            goal.source_path: target_exceptions(goal.source_path)
            for goal, _recipe, _shape in probes
        }
        outcomes = self.executor.replay_many(
            [(goal, recipe) for goal, recipe, _shape in probes]
        )
        for (goal, recipe, shape), outcome in zip(probes, outcomes):
            if outcome.succeeded or not outcome.exception_type:
                continue
            if outcome.exception_type.split(".")[-1] not in allowed[goal.source_path]:
                continue
            key = f"{goal.key}:{recipe.key}:{outcome.exception_type}"
            if key in seen:
                continue
            confirmation = self.executor.replay(goal, recipe)
            reproduced = (
                not confirmation.succeeded
                and confirmation.exception_type == outcome.exception_type
            )
            seen.add(key)
            yield Emission(
                NEGATIVE,
                key,
                NegativeSequence(
                    goal=goal,
                    recipe=recipe,
                    calls=tuple(goal.calls),
                    outcome=replace(outcome, reproduced=reproduced),
                    shape=shape,
                ),
            )

    def _probes(
        self, validated: ValidatedRecipe
    ) -> list[tuple[ObjectGoal, ConstructionRecipe, str]]:
        source = validated.goal.source_path
        class_name = validated.goal.class_name
        bare = ObjectGoal(source, class_name)
        methods = public_method_calls(bare)
        constructor = validated.recipe.constructor
        setup = validated.recipe.setup
        probes: list[tuple[ObjectGoal, ConstructionRecipe, str]] = []

        for call in methods:
            # start(); start() -- the same transition demanded twice.
            repeated = ObjectGoal(source, class_name, (call, call))
            probes.append(
                (repeated, ConstructionRecipe(repeated.key, constructor, setup), "repeated")
            )
            # stop() with nothing before it -- a precondition simply skipped.
            if setup:
                alone = ObjectGoal(source, class_name, (call,))
                probes.append(
                    (alone, ConstructionRecipe(alone.key, constructor, ()), "unprepared")
                )
        if len(setup) > 1:
            reordered = tuple(reversed(setup))
            for call in methods:
                goal = ObjectGoal(source, class_name, (call,))
                probes.append(
                    (goal, ConstructionRecipe(goal.key, constructor, reordered), "reordered")
                )
        return probes
