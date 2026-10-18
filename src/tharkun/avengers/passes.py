"""Built-in Avengers passes for constructor and receiver-state discovery."""

from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tharkun.avengers.engine import Emission, RepeatPolicy
from tharkun.avengers.executor import IsolatedExecutor
from tharkun.avengers.model import (
    OBJECT,
    ConstructionRecipe,
    ConstructorCall,
    FunctionCall,
    MethodCall,
    NestedObject,
    ObjectGoal,
    SourceArtifact,
    ValidatedRecipe,
    field_bounds_hold,
    materialize_recipe,
)
from tharkun.knowledge import Evidence, KnowledgeBase
from tharkun.solve.frontend.constructor import (
    ConstructorParameter,
    ConstructorTemplate,
    constructor_template,
)


GOAL = "object-goal"
TEMPLATE = "constructor-template"
CANDIDATE = "candidate-recipe"
REPLAY = "replay-outcome"
VALIDATED = "validated-recipe"
FLAT_TEMPLATE = "flat-object-template"
FLAT_OUTCOME = "flatting-outcome"
SOURCE_ARTIFACT = "source-artifact"


def _goal(knowledge: KnowledgeBase, key: str) -> ObjectGoal | None:
    item = knowledge.evidence_item(GOAL, key)
    return item.value if item is not None and isinstance(item.value, ObjectGoal) else None


@dataclass(frozen=True)
class ConstructorTemplatePass:
    name: str = "constructor-template"
    consumes: frozenset[str] = frozenset({GOAL})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, _knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        for item in delta:
            goal = item.value
            if not isinstance(goal, ObjectGoal):
                continue
            template = constructor_template(goal.source_path, goal.class_name)
            if template is not None:
                yield Emission(TEMPLATE, goal.key, template, (item.identity,))


def _container_type_key(text: str) -> str | None:
    """The bare container/sequence type name ``text`` (already stripped of the
    ``typing.`` prefix) names, or ``None`` -- shared by ``_baseline_value`` and
    the [internal ref] contract-evidence hook (``_contract_nonempty_value``) so both
    read the same annotation -> type-name mapping."""
    if text in ("str", "bytes"):
        return text
    if text.startswith(("list", "List")):
        return "list"
    if text.startswith(("dict", "Dict", "Mapping")):
        return "dict"
    if text.startswith(("set", "Set")):
        return "set"
    if text.startswith(("tuple", "Tuple")):
        return "tuple"
    return None


def _baseline_value(annotation: str | None) -> Any:
    if annotation is None:
        return None
    text = annotation.replace("typing.", "").strip()
    if text in {"int", "Integer"}:
        return 0
    if text == "float":
        return 0.0
    if text == "bool":
        return False
    key = _container_type_key(text)
    if key == "str":
        return ""
    if key == "bytes":
        return b""
    if key == "list":
        return []
    if key == "dict":
        return {}
    if key == "set":
        return set()
    if key == "tuple":
        return ()
    if "None" in text or text.startswith("Optional["):
        return None
    return None


def _contract_nonempty_value(annotation: str | None) -> Any:
    """[internal ref] step 2: a non-empty seed for ``annotation``'s container type, when
    a TESTED contract shows this type's own methods can raise on an empty/
    falsy receiver (see ``solve/contracts/adapters/avengers.py``). ``None``
    when the annotation names no bare container type, or there is no such
    evidence -- in which case the caller adds nothing beyond the existing
    baseline candidate.
    """
    if annotation is None:
        return None
    key = _container_type_key(annotation.replace("typing.", "").strip())
    if key is None:
        return None
    from tharkun.solve.contracts.adapters.avengers import nonempty_seed_for_type

    return nonempty_seed_for_type(key)


def _call_from_values(
    template: ConstructorTemplate,
    values: dict[str, Any] | None = None,
) -> ConstructorCall:
    supplied = values or {}
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for parameter in template.parameters:
        if parameter.kind in {"var_positional", "var_keyword"}:
            continue
        if parameter.name in supplied:
            value = supplied[parameter.name]
        elif parameter.has_default:
            # Omitting a declared default preserves executable defaults too;
            # inserting ``None`` for a nonliteral default would change meaning.
            continue
        else:
            value = _baseline_value(parameter.annotation)
        if parameter.kind == "positional_only":
            args.append(value)
        else:
            kwargs[parameter.name] = value
    return ConstructorCall(args, kwargs)


@dataclass(frozen=True)
class BaselineCandidatePass:
    """Publish the smallest annotation/default-derived constructor call."""

    name: str = "constructor-baseline"
    consumes: frozenset[str] = frozenset({TEMPLATE})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, _knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        for item in delta:
            template = item.value
            if not isinstance(template, ConstructorTemplate):
                continue
            recipe = ConstructionRecipe(item.key, _call_from_values(template))
            yield Emission(CANDIDATE, recipe.key, recipe, (item.identity,))
            # [internal ref] step 2: an ADDITIONAL candidate, never a replacement -- the
            # baseline above is unchanged, and replay still decides both for real.
            overrides = {
                p.name: v for p in template.parameters
                if (v := _contract_nonempty_value(p.annotation)) is not None
            }
            if overrides:
                alt = ConstructionRecipe(item.key, _call_from_values(template, overrides))
                yield Emission(CANDIDATE, alt.key, alt, (item.identity,))


def _annotation_node(annotation: str | None) -> ast.expr | None:
    if annotation is None:
        return None
    try:
        return ast.parse(annotation, mode="eval").body
    except SyntaxError:
        return None


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _slice_items(node: ast.AST) -> tuple[ast.AST, ...]:
    return tuple(node.elts) if isinstance(node, ast.Tuple) else (node,)


def _strategy_for_annotation(annotation: str | None):
    """Build a finite, constructive strategy from a static annotation."""

    from hypothesis import strategies as st

    node = _annotation_node(annotation)
    bare = (_name(node) or "").removeprefix("typing.") if node is not None else ""
    if bare == "bool":
        return st.booleans()
    if bare == "int":
        from tharkun.strategies import make_strategy

        return make_strategy({"type": "int", "value_range": [-32, 32]})
    if bare == "float":
        from tharkun.strategies import make_strategy

        return make_strategy({"type": "float", "value_range": [-32.0, 32.0]})
    if bare == "str":
        from tharkun.strategies import make_strategy

        return make_strategy({"type": "str", "length": {"min": 0, "max": 16}})
    if bare == "bytes":
        from tharkun.strategies import make_strategy

        return make_strategy({"type": "bytes", "length": {"min": 0, "max": 16}})
    if bare in {"None", "NoneType"}:
        return st.none()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return st.one_of(
            _strategy_for_annotation(ast.unparse(node.left)),
            _strategy_for_annotation(ast.unparse(node.right)),
        )
    if isinstance(node, ast.Subscript):
        origin = (_name(node.value) or "").removeprefix("typing.")
        parameters = _slice_items(node.slice)
        if origin in {"Optional"} and parameters:
            return st.one_of(st.none(), _strategy_for_annotation(ast.unparse(parameters[0])))
        if origin in {"Union"}:
            return st.one_of(*(_strategy_for_annotation(ast.unparse(item)) for item in parameters))
        if origin in {"list", "List", "Sequence", "MutableSequence"}:
            element = _strategy_for_annotation(ast.unparse(parameters[0])) if parameters else st.integers()
            return st.lists(element, max_size=4)
        if origin in {"set", "Set", "frozenset", "FrozenSet"}:
            element = _strategy_for_annotation(ast.unparse(parameters[0])) if parameters else st.integers()
            base = st.sets(element, max_size=4)
            return base.map(frozenset) if "rozen" in origin else base
        if origin in {"tuple", "Tuple"}:
            if len(parameters) == 2 and isinstance(parameters[1], ast.Constant) and parameters[1].value is Ellipsis:
                return st.lists(
                    _strategy_for_annotation(ast.unparse(parameters[0])), max_size=4
                ).map(tuple)
            return st.tuples(*(_strategy_for_annotation(ast.unparse(item)) for item in parameters))
        if origin in {"dict", "Dict", "Mapping", "MutableMapping"}:
            key = _strategy_for_annotation(ast.unparse(parameters[0])) if parameters else st.text(max_size=8)
            value = _strategy_for_annotation(ast.unparse(parameters[1])) if len(parameters) > 1 else st.integers()
            return st.dictionaries(key, value, max_size=4)
    return st.one_of(
        st.none(),
        st.booleans(),
        st.integers(-8, 8),
        st.text(max_size=8),
        st.binary(max_size=8),
    )


def _call_strategy(template: ConstructorTemplate):
    from hypothesis import strategies as st

    required = tuple(
        parameter
        for parameter in template.parameters
        if parameter.kind not in {"var_positional", "var_keyword"}
        and not parameter.has_default
    )

    @st.composite
    def calls(draw):
        values = {
            parameter.name: draw(_strategy_for_annotation(parameter.annotation))
            for parameter in required
        }
        return _call_from_values(template, values)

    return calls()


@dataclass(frozen=True)
class HypothesisCandidatePass:
    """Find a constructor call whose fresh-process construction succeeds."""

    executor: IsolatedExecutor = field(default_factory=IsolatedExecutor)
    max_examples: int = 100
    backend: str = "hypothesis"
    name: str = ""
    consumes: frozenset[str] = frozenset({TEMPLATE})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", f"{self.backend}-constructor")

    def run(
        self, knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        try:
            from hypothesis import HealthCheck, find, settings
            from hypothesis.errors import NoSuchExample, Unsatisfiable
        except ImportError:
            return ()

        emissions: list[Emission] = []
        for item in delta:
            template = item.value
            goal = _goal(knowledge, item.key)
            if not isinstance(template, ConstructorTemplate) or goal is None:
                continue
            constructor_goal = ObjectGoal(goal.source_path, goal.class_name)
            # [internal ref] slice 3 ([internal ref]): when the REAL goal states field
            # bounds ("a C whose .x > 3 and .items is non-empty"), make THIS
            # search target them directly -- otherwise `find()` converges to
            # whatever constructs cheapest (typically a trivial/zero value),
            # which satisfies "constructs at all" but essentially never the
            # bounds too (measured on a toy this session: a combined bound
            # was not found in 15 independent unguided searches). A live
            # object is materialized IN-PROCESS to check bounds -- the same
            # trick `driver.py::_avengers_object_for_bounds` uses -- because
            # the bound check needs the constructed object itself, which a
            # disposable child's `ReplayOutcome` does not carry back
            # (`ObjectGoal.field_bounds`'s own docstring explains why this
            # lives beside `satisfied()` rather than inside it).
            environment_flags = set()
            mglobals: dict[str, Any] | None = None
            if goal.field_bounds:
                from tharkun.solve.loader import load_module_for_execution

                try:
                    with self.executor.confine(environment_flags):
                        _mod, mglobals = load_module_for_execution(goal.source_path)
                except ValueError:
                    continue

            def accepted(call: ConstructorCall) -> bool:
                if environment_flags - {"file_effect"}:
                    return False
                recipe = ConstructionRecipe(goal.key, call)
                outcome = self.executor.replay(constructor_goal, recipe)
                if not outcome.succeeded:
                    return False
                if goal.field_bounds:
                    if mglobals is None or goal.class_name not in mglobals:
                        return False
                    with self.executor.confine(environment_flags) as environment:
                        obj = None
                        try:
                            obj = materialize_recipe(recipe, goal.class_name, mglobals)
                            if obj is None or not field_bounds_hold(obj, goal.field_bounds):
                                return False
                        except Exception:
                            return False
                        finally:
                            obj = None
                        if environment is not None and (environment.flags - {"file_effect"}):
                            return False
                return True

            try:
                call = find(
                    _call_strategy(template),
                    accepted,
                    settings=settings(
                        max_examples=self.max_examples,
                        deadline=None,
                        database=None,
                        backend=self.backend,
                        suppress_health_check=[HealthCheck.too_slow],
                    ),
                )
            except (NoSuchExample, Unsatisfiable):
                continue
            recipe = ConstructionRecipe(goal.key, call)
            emissions.append(Emission(CANDIDATE, recipe.key, recipe, (item.identity,)))
        return emissions


def _literal(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    return ast.literal_eval(node)


def _called_name(call: ast.Call) -> str | None:
    target: ast.AST = call.func
    parts: list[str] = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
    return ".".join(reversed(parts)) if parts else None


def _literal_call(call: ast.Call, names: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    args = tuple(_literal(node, names) for node in call.args)
    kwargs = {
        keyword.arg: _literal(keyword.value, names)
        for keyword in call.keywords
        if keyword.arg is not None
    }
    return args, kwargs


def _recipe_argument(node: ast.AST, names: dict[str, Any], owner: str) -> Any:
    """Decode one Pynguin argument, replacing *owner* with ``OBJECT``."""

    if isinstance(node, ast.Name) and node.id == owner:
        return OBJECT
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.Tuple):
        return tuple(_recipe_argument(item, names, owner) for item in node.elts)
    if isinstance(node, ast.List):
        return [_recipe_argument(item, names, owner) for item in node.elts]
    if isinstance(node, ast.Set):
        return {_recipe_argument(item, names, owner) for item in node.elts}
    if isinstance(node, ast.Dict):
        return {
            _recipe_argument(key, names, owner): _recipe_argument(value, names, owner)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    return ast.literal_eval(node)


def _function_call(
    call: ast.Call,
    literals: dict[str, Any],
    objects: dict[str, ConstructionRecipe],
    module_functions: frozenset[str],
) -> tuple[str, FunctionCall] | None:
    """Recover ``module.function(object, ...)`` for one tracked object."""

    called = _called_name(call)
    if called is None:
        return None
    target = called.split(".")[-1]
    if target not in module_functions:
        return None
    referenced = {
        node.id
        for argument in (*call.args, *(item.value for item in call.keywords))
        for node in ast.walk(argument)
        if isinstance(node, ast.Name) and node.id in objects
    }
    if len(referenced) != 1:
        return None
    owner = next(iter(referenced))
    try:
        args = tuple(_recipe_argument(node, literals, owner) for node in call.args)
        kwargs = {
            keyword.arg: _recipe_argument(keyword.value, literals, owner)
            for keyword in call.keywords
            if keyword.arg is not None
        }
        return owner, FunctionCall(target, args, kwargs)
    except (ValueError, TypeError):
        return None


def _module_functions(source_path: str) -> frozenset[str]:
    try:
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return frozenset()
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


@dataclass(frozen=True)
class GeneratedSourcePass:
    """Recover concrete constructors and setup sequences from generated tests.

    Any tool that writes Python test source lands here -- Pynguin, Ghostwriter,
    CodaMosa -- because a constructor followed by method calls reads the same
    way whichever search wrote it.  One parser, so a new generator needs a way
    to produce source and nothing else.
    """

    tools: frozenset[str] = frozenset({"pynguin", "ghostwriter", "codamosa"})
    name: str = "test-source"
    consumes: frozenset[str] = frozenset({GOAL, SOURCE_ARTIFACT})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, knowledge: KnowledgeBase, _delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        goals = tuple(
            item.value for item in knowledge.evidence(GOAL) if isinstance(item.value, ObjectGoal)
        )
        artifacts = tuple(
            item for item in knowledge.evidence(SOURCE_ARTIFACT)
            if isinstance(item.value, SourceArtifact) and item.value.tool.lower() in self.tools
        )
        seen: set[tuple[str, str]] = set()
        for artifact_item in artifacts:
            try:
                tree = ast.parse(artifact_item.value.source)
            except SyntaxError:
                continue
            for goal in goals:
                for recipe in self._recipes(tree, goal):
                    identity = (goal.key, recipe.key)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    yield Emission(
                        CANDIDATE,
                        recipe.key,
                        recipe,
                        ((GOAL, goal.key), artifact_item.identity),
                    )

    @staticmethod
    def _recipes(tree: ast.Module, goal: ObjectGoal) -> Iterable[ConstructionRecipe]:
        class_leaf = goal.class_name.split(".")[-1]
        module_functions = _module_functions(goal.source_path)
        bodies = [tree.body]
        bodies.extend(
            node.body
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        for body in bodies:
            literals: dict[str, Any] = {}
            objects: dict[str, ConstructionRecipe] = {}
            for statement in body:
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                ):
                    target = statement.targets[0].id
                    if not isinstance(statement.value, ast.Call):
                        try:
                            literals[target] = ast.literal_eval(statement.value)
                        except (ValueError, TypeError):
                            pass
                        continue
                    call = statement.value
                    called = _called_name(call)
                    if called and called.split(".")[-1] == class_leaf:
                        try:
                            args, kwargs = _literal_call(call, literals)
                            recipe = ConstructionRecipe(
                                goal.key, ConstructorCall(args, kwargs)
                            )
                        except (ValueError, TypeError):
                            continue
                        objects[target] = recipe
                        yield recipe
                        continue
                elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                    call = statement.value
                else:
                    continue

                if not (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id in objects
                ):
                    recovered = _function_call(
                        call, literals, objects, module_functions
                    )
                    if recovered is None:
                        continue
                    owner, setup_call = recovered
                else:
                    owner = call.func.value.id
                    try:
                        args, kwargs = _literal_call(call, literals)
                        setup_call = MethodCall(call.func.attr, args, kwargs)
                    except (ValueError, TypeError):
                        continue
                previous = objects[owner]
                advanced = ConstructionRecipe(
                    goal.key,
                    previous.constructor,
                    previous.setup + (setup_call,),
                )
                objects[owner] = advanced
                yield advanced


def _class_node(source_path: str, class_name: str) -> ast.ClassDef | None:
    try:
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return None
    wanted = class_name.split(".")[-1]
    return next(
        (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == wanted),
        None,
    )


def _required_method_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    if positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
    required_positional = max(0, len(positional) - len(args.defaults))
    required_keywords = sum(default is None for default in args.kw_defaults)
    return required_positional + required_keywords


def _writes_receiver(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for child in ast.walk(node):
        targets: tuple[ast.AST, ...] = ()
        if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if isinstance(child, ast.Assign):
                targets = tuple(child.targets)
            else:
                targets = (child.target,)
        for target in targets:
            for part in ast.walk(target):
                if (
                    isinstance(part, ast.Attribute)
                    and isinstance(part.value, ast.Name)
                    and part.value.id == "self"
                ):
                    return True
    return False


def _public_method_calls(
    goal: ObjectGoal,
) -> tuple[tuple[bool, MethodCall], ...]:
    cls = _class_node(goal.source_path, goal.class_name)
    if cls is None:
        return ()
    all_calls: list[tuple[bool, MethodCall]] = []
    for node in cls.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") or node.name == "__init__":
            continue
        if _required_method_parameters(node) == 0 and not isinstance(node, ast.AsyncFunctionDef):
            all_calls.append((_writes_receiver(node), MethodCall(node.name)))
    return tuple(all_calls)


def public_method_calls(goal: ObjectGoal) -> tuple[MethodCall, ...]:
    """Return statically visible public methods callable without arguments."""

    return tuple(call for _writes, call in _public_method_calls(goal))


def _setup_calls(goal: ObjectGoal) -> tuple[MethodCall, ...]:
    all_calls = _public_method_calls(goal)
    state_producers = tuple(call for writes, call in all_calls if writes)
    return state_producers or tuple(call for _writes, call in all_calls)


_CONTAINER_ANNOTATIONS = frozenset(
    {
        "int", "float", "str", "bytes", "bool", "None", "object", "Any",
        "list", "dict", "set", "frozenset", "tuple", "List", "Dict", "Set",
        "FrozenSet", "Tuple", "Optional", "Union", "Sequence", "Mapping",
        "Iterable", "Callable", "Type", "bytearray", "complex",
    }
)


def _annotation_class_names(annotation: str | None) -> tuple[str, ...]:
    """Bare class names an annotation mentions, ignoring typing machinery.

    ``Box`` yields ``Box``; ``Optional[Box]`` and ``Box | None`` yield ``Box``
    too, because an optional domain object is still worth building.  Builtin
    and ``typing`` names are dropped -- those already have literal strategies.
    """

    node = _annotation_node(annotation)
    if node is None:
        return ()
    found: list[str] = []
    for child in ast.walk(node):
        name = _name(child) if isinstance(child, (ast.Name, ast.Attribute)) else None
        if name is None:
            continue
        bare = name.split(".")[-1]
        if bare in _CONTAINER_ANNOTATIONS or bare in found:
            continue
        found.append(bare)
    return tuple(found)


def resolve_annotation_class(source_path: str, annotation: str | None) -> tuple[str, str] | None:
    """Locate the class an annotation names as ``(source path, class name)``.

    Only two resolutions are attempted, and both are static: a class defined in
    the same module, and a class the module imported.  Anything else returns
    ``None`` rather than a guess -- a wrong source path would send the worker
    off to build the wrong type.
    """

    for bare in _annotation_class_names(annotation):
        if _class_node(source_path, bare) is not None:
            return (source_path, bare)
        try:
            from tharkun.solve.frontend.import_resolver import parse_imports

            imported = parse_imports(source_path)
        except Exception:
            imported = {}
        target = imported.get(bare)
        if target and _class_node(target, bare) is not None:
            return (target, bare)
    return None


@dataclass(frozen=True)
class NestedCandidatePass:
    """Build the domain objects a constructor needs, then pass them in.

    A class whose ``__init__`` takes another object cannot be reached by any
    amount of literal search.  This pass notices such a parameter, obtains a
    validated recipe for the annotated class -- from evidence already in the
    knowledge base, from the store, or by running a bounded search of its own
    -- and proposes the composed constructor call.

    ``max_depth`` bounds the recursion. Composition is worth one or two levels;
    an unbounded chain is a way to spend the whole budget below the boundary
    the caller actually asked about.
    """

    synthesize: Any = None
    store: Any = None
    max_depth: int = 1
    depth: int = 0
    max_nested_recipes: int = 3
    name: str = "nested-objects"
    consumes: frozenset[str] = frozenset({TEMPLATE})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        if self.depth >= self.max_depth:
            return
        for item in delta:
            template = item.value
            if not isinstance(template, ConstructorTemplate):
                continue
            goal = _goal(knowledge, item.key)
            if goal is None:
                continue
            substitutions: dict[str, list[NestedObject]] = {}
            for parameter in template.parameters:
                if parameter.kind in {"var_positional", "var_keyword"}:
                    continue
                target = resolve_annotation_class(goal.source_path, parameter.annotation)
                if target is None or target == (goal.source_path, goal.class_name):
                    # Self-composition would recurse on the very class being
                    # built, which is never the shortest route to it.
                    continue
                recipes = self._nested_recipes(knowledge, *target)
                if recipes:
                    substitutions[parameter.name] = [
                        NestedObject(target[0], target[1], recipe) for recipe in recipes
                    ]
            if not substitutions:
                continue
            for values in _substitution_products(substitutions):
                recipe = ConstructionRecipe(item.key, _call_from_values(template, values))
                yield Emission(CANDIDATE, recipe.key, recipe, (item.identity,))

    def _nested_recipes(
        self, knowledge: KnowledgeBase, source_path: str, class_name: str
    ) -> tuple[ConstructionRecipe, ...]:
        found: dict[str, ConstructionRecipe] = {}
        for item in knowledge.evidence(VALIDATED):
            validated = item.value
            if not isinstance(validated, ValidatedRecipe):
                continue
            if (
                validated.goal.source_path == source_path
                and validated.goal.class_name == class_name
            ):
                found.setdefault(validated.recipe.key, validated.recipe)
        if self.store is not None:
            for stored in self.store.recipes_for_type(f"{source_path}::{class_name}"):
                found.setdefault(stored.recipe.key, stored.recipe)
        if not found and self.synthesize is not None:
            for recipe in self.synthesize(source_path, class_name, self.depth + 1):
                found.setdefault(recipe.key, recipe)
        ordered = sorted(
            found.values(),
            key=lambda recipe: (
                len(recipe.setup),
                len(recipe.constructor.args) + len(recipe.constructor.kwargs),
                recipe.key,
            ),
        )
        return tuple(ordered[: self.max_nested_recipes])


def _substitution_products(
    substitutions: dict[str, list[Any]]
) -> Iterable[dict[str, Any]]:
    """Every combination of one candidate per object-valued parameter."""

    names = sorted(substitutions)
    for combination in itertools.product(*(substitutions[name] for name in names)):
        yield dict(zip(names, combination))


@dataclass(frozen=True)
class SetupSequencePass:
    """Enumerate bounded state-producing zero-argument method sequences."""

    max_steps: int = 2
    max_sequences: int = 64
    name: str = "setup-sequences"
    consumes: frozenset[str] = frozenset({CANDIDATE})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def __post_init__(self) -> None:
        if self.max_steps < 0 or self.max_sequences < 1:
            raise ValueError("setup bounds must be non-negative")

    def run(
        self, knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        emitted = 0
        for item in delta:
            recipe = item.value
            if not isinstance(recipe, ConstructionRecipe) or recipe.setup:
                continue
            goal = _goal(knowledge, recipe.goal_key)
            if goal is None:
                continue
            calls = _setup_calls(goal)
            for length in range(1, self.max_steps + 1):
                for sequence in itertools.product(calls, repeat=length):
                    candidate = ConstructionRecipe(recipe.goal_key, recipe.constructor, sequence)
                    yield Emission(CANDIDATE, candidate.key, candidate, (item.identity,))
                    emitted += 1
                    if emitted >= self.max_sequences:
                        return


@dataclass(frozen=True)
class ReplayPass:
    """Promote only recipes whose replay does what the goal asked for.

    Usually that means "returned normally". For a boundary-directed goal it
    means the boundary raised, or raised a named exception, or brought the
    interpreter down -- so the same child result is a success for one goal and
    a failure for another, and the goal decides which.
    """

    executor: IsolatedExecutor = field(default_factory=IsolatedExecutor)
    name: str = "fresh-replay"
    consumes: frozenset[str] = frozenset({CANDIDATE})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        for item in delta:
            recipe = item.value
            if not isinstance(recipe, ConstructionRecipe):
                continue
            goal = _goal(knowledge, recipe.goal_key)
            if goal is None:
                continue
            outcome = self.executor.replay(goal, recipe)
            outcome_key = f"{goal.key}:{recipe.key}"
            yield Emission(REPLAY, outcome_key, outcome, (item.identity,))
            if goal.satisfied(outcome):
                validated = ValidatedRecipe(goal, recipe, outcome)
                yield Emission(VALIDATED, outcome_key, validated, ((REPLAY, outcome_key),))


@dataclass(frozen=True)
class FlattingPass:
    """Generalize validated exemplars with the py_obj_flatting component."""

    executor: IsolatedExecutor = field(default_factory=IsolatedExecutor)
    name: str = "py-obj-flatting"
    consumes: frozenset[str] = frozenset({VALIDATED})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, _knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        from tharkun.avengers.flatting import FlatObjectTemplate

        for item in delta:
            validated = item.value
            if not isinstance(validated, ValidatedRecipe):
                continue
            result = self.executor.mine_template(validated.goal, validated.recipe)
            key = f"{validated.goal.key}:{validated.recipe.key}"
            if isinstance(result, FlatObjectTemplate):
                yield Emission(FLAT_TEMPLATE, result.key, result, (item.identity,))
            else:
                yield Emission(FLAT_OUTCOME, key, result, (item.identity,))


#: The pre-rename spelling.  ``GeneratedSourcePass`` reads more than Pynguin
#: now, but the old name is a public one and stays a permanent alias.  (The
#: obvious name, ``TestSourcePass``, is unusable: pytest collects anything
#: called ``Test*`` and then warns that it cannot.)
PynguinAstPass = GeneratedSourcePass
