"""A durable, data-only record of what Avengers has already built.

The knowledge base is per-run: a second search for the same class relearns
every recipe from scratch, and a recipe validated for one class is invisible to
the next.  This module is the persistence seam.  What it keeps is deliberately
small -- validated recipes, certified flat templates, and the negative call
sequences that were worth finding -- because those are the things a search
cannot cheaply rediscover.

The file format follows the same discipline as :mod:`tharkun.avengers._wire`:
a closed vocabulary, strict validators, a size ceiling, and no pickling.  A
store file is written by this process but may be read after target code has
run, so it is parsed as untrusted data, exactly like a worker response.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tharkun.avengers.engine import Emission, RepeatPolicy
from tharkun.avengers.model import (
    ConstructionRecipe,
    ObjectGoal,
    ReplayOutcome,
    call_from_record,
    recipe_from_record,
)
from tharkun.knowledge import KnowledgeBase

FORMAT = "tharkun-avengers-store/1"
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024

# Evidence kinds this module reads and writes.  They are re-declared here
# rather than imported from passes.py so the store does not drag the whole
# portfolio in; tests/test_avengers_store.py pins them to the same strings.
VALIDATED = "validated-recipe"
FLAT_TEMPLATE = "flat-object-template"
NEGATIVE = "negative-sequence"
CANDIDATE = "candidate-recipe"


@dataclass(frozen=True)
class StoredRecipe:
    """One recipe a previous run validated, with the class it built."""

    source_path: str
    class_name: str
    recipe: ConstructionRecipe
    state_digest: str | None = None

    @property
    def qualname(self) -> str:
        """``module path::Class`` -- the identity a later goal matches on."""

        return f"{self.source_path}::{self.class_name}"


@dataclass(frozen=True)
class StoredNegative:
    """A call sequence that raised, kept because it is a result."""

    source_path: str
    class_name: str
    recipe: ConstructionRecipe
    calls: tuple[Any, ...]
    exception_type: str
    shape: str = ""
    phase: str | None = None
    reproduced: bool | None = None


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{name} must be a string or null")


def _optional_bool(value: Any, name: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a boolean or null")


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


class TemplateStore:
    """Recipes, templates and negative sequences that outlive one search.

    Documents are sharded by the first two characters of the goal key, which
    is already a SHA-256 prefix, so a large corpus does not put thousands of
    files in one directory.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    # -- paths ---------------------------------------------------------------

    def _path(self, goal_key: str) -> Path:
        if not goal_key or "/" in goal_key or "\\" in goal_key or goal_key.startswith("."):
            raise ValueError("a goal key is a bare hexadecimal identifier")
        return self.root / goal_key[:2] / f"{goal_key}.json"

    def documents(self) -> tuple[Path, ...]:
        if not self.root.is_dir():
            return ()
        return tuple(sorted(self.root.glob("*/*.json")))

    # -- reading -------------------------------------------------------------

    def _read(self, path: Path) -> dict[str, Any] | None:
        try:
            if path.stat().st_size > MAX_DOCUMENT_BYTES:
                return None
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            return None
        if not isinstance(document, dict) or document.get("format") != FORMAT:
            return None
        return document

    def recipes_for_goal(self, goal: ObjectGoal) -> tuple[StoredRecipe, ...]:
        """Every recipe a previous run validated for exactly this goal."""

        document = self._read(self._path(goal.key))
        if document is None:
            return ()
        return self._recipes(document)

    def recipes_for_type(self, qualname: str) -> tuple[StoredRecipe, ...]:
        """Every recipe that built ``path::Class``, whatever goal found it.

        This is the cross-goal seam: a recipe learned while chasing one
        boundary is exactly what another boundary -- or another class that
        needs this one as an argument -- should start from.
        """

        found: dict[str, StoredRecipe] = {}
        for path in self.documents():
            document = self._read(path)
            if document is None:
                continue
            for stored in self._recipes(document):
                if stored.qualname == qualname:
                    found.setdefault(stored.recipe.key, stored)
        return tuple(found[key] for key in sorted(found))

    @staticmethod
    def _recipes(document: dict[str, Any]) -> tuple[StoredRecipe, ...]:
        goal = _object(document.get("goal"), "goal")
        source_path = _text(goal.get("source_path"), "goal source_path")
        class_name = _text(goal.get("class_name"), "goal class_name")
        result: list[StoredRecipe] = []
        for entry in _list(document.get("recipes", []), "recipes"):
            item = _object(entry, "recipe entry")
            try:
                recipe = recipe_from_record(item.get("recipe"))
            except ValueError:
                # One unreadable entry must not lose the rest of the file.
                continue
            result.append(
                StoredRecipe(
                    source_path=source_path,
                    class_name=class_name,
                    recipe=recipe,
                    state_digest=_optional_text(item.get("state_digest"), "state_digest"),
                )
            )
        return tuple(result)

    def negatives_for_goal(self, goal: ObjectGoal) -> tuple[StoredNegative, ...]:
        document = self._read(self._path(goal.key))
        if document is None:
            return ()
        result: list[StoredNegative] = []
        for entry in _list(document.get("negatives", []), "negatives"):
            item = _object(entry, "negative entry")
            try:
                recipe = recipe_from_record(item.get("recipe"))
                calls = tuple(call_from_record(c) for c in _list(item.get("calls", []), "calls"))
            except ValueError:
                continue
            result.append(
                StoredNegative(
                    source_path=goal.source_path,
                    class_name=goal.class_name,
                    recipe=recipe,
                    calls=calls,
                    exception_type=_text(item.get("exception_type"), "exception_type"),
                    shape=_optional_text(item.get("shape"), "shape") or "",
                    phase=_optional_text(item.get("phase"), "phase"),
                    reproduced=_optional_bool(item.get("reproduced"), "reproduced"),
                )
            )
        return tuple(result)

    def templates_for_goal(self, goal: ObjectGoal) -> tuple[dict[str, Any], ...]:
        """Flat-template payloads, in the wire's own encoding.

        They are returned as documents rather than ``FlatObjectTemplate``
        objects because ``py_obj_flatting`` is optional: a store written on a
        machine that had it must stay readable on one that does not.
        """

        document = self._read(self._path(goal.key))
        if document is None:
            return ()
        return tuple(
            _object(entry, "template entry")
            for entry in _list(document.get("templates", []), "templates")
        )

    def load(self, goal: ObjectGoal) -> tuple[Emission, ...]:
        """Prior recipes for *goal*, as candidates for this run to re-verify.

        They are published as candidates, never as validated results.  A recipe
        that worked last week is a good guess, not evidence: the target may
        have changed, so it earns its promotion from a fresh replay like any
        other proposal.
        """

        emissions: list[Emission] = []
        for stored in self.recipes_for_goal(goal):
            recipe = stored.recipe
            if recipe.goal_key != goal.key:
                recipe = ConstructionRecipe(goal.key, recipe.constructor, recipe.setup)
            emissions.append(Emission(CANDIDATE, recipe.key, recipe))
        return tuple(emissions)

    # -- writing -------------------------------------------------------------

    def save(self, goal: ObjectGoal, knowledge: KnowledgeBase) -> Path:
        """Merge this run's results for *goal* into the store."""

        path = self._path(goal.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._read(path) or {}

        recipes: dict[str, dict[str, Any]] = {}
        for entry in _list(existing.get("recipes", []), "recipes"):
            item = _object(entry, "recipe entry")
            try:
                key = recipe_from_record(item.get("recipe")).key
            except ValueError:
                continue
            recipes[key] = item
        for item in knowledge.evidence(VALIDATED):
            validated = item.value
            recipe = getattr(validated, "recipe", None)
            if not isinstance(recipe, ConstructionRecipe):
                continue
            outcome = getattr(validated, "outcome", None)
            digest = getattr(outcome, "state_digest", None) if outcome is not None else None
            recipes[recipe.key] = {
                "recipe": recipe.record,
                "state_digest": digest,
                "producers": sorted(
                    {derivation.producer for derivation in item.derivations}
                ),
            }

        templates: dict[str, dict[str, Any]] = {}
        for entry in _list(existing.get("templates", []), "templates"):
            item = _object(entry, "template entry")
            sha = item.get("recipe_sha")
            if isinstance(sha, str):
                templates[sha] = item
        for item in knowledge.evidence(FLAT_TEMPLATE):
            encoded = _encode_template(item.value)
            if encoded is not None:
                templates[encoded["recipe_sha"]] = encoded

        negatives: dict[str, dict[str, Any]] = {}
        for entry in _list(existing.get("negatives", []), "negatives"):
            item = _object(entry, "negative entry")
            key = item.get("key")
            if isinstance(key, str):
                negatives[key] = item
        for item in knowledge.evidence(NEGATIVE):
            encoded = _encode_negative(item.key, item.value)
            if encoded is not None:
                negatives[item.key] = encoded

        document = {
            "format": FORMAT,
            "goal": {
                "source_path": goal.source_path,
                "class_name": goal.class_name,
                "calls": [call.record for call in goal.calls],
                "key": goal.key,
            },
            "recipes": [recipes[key] for key in sorted(recipes)],
            "templates": [templates[key] for key in sorted(templates)],
            "negatives": [negatives[key] for key in sorted(negatives)],
        }
        _write_atomic(path, json.dumps(document, sort_keys=True, indent=1))
        return path


def _encode_template(value: Any) -> dict[str, Any] | None:
    """Re-use the worker wire's template payload so one encoding exists."""

    try:
        from tharkun.avengers._wire import _encode_template as encode
    except ImportError:  # pragma: no cover - the wire ships with the package
        return None
    try:
        return encode(value)
    except (AttributeError, TypeError, ValueError):
        return None


def _encode_negative(key: str, value: Any) -> dict[str, Any] | None:
    """Encode a ``NegativeSequence`` -- read off the object, not off a dict.

    Taken from attributes rather than by importing ``objectives``, so the store
    does not pull the objective passes in just to write one row.
    """

    recipe = getattr(value, "recipe", None)
    outcome = getattr(value, "outcome", None)
    exception_type = getattr(value, "exception_type", None)
    if not isinstance(recipe, ConstructionRecipe) or not isinstance(exception_type, str):
        return None
    if not exception_type:
        return None
    try:
        encoded_calls = [call.record for call in getattr(value, "calls", ())]
    except AttributeError:
        return None
    return {
        "key": key,
        "recipe": recipe.record,
        "calls": encoded_calls,
        "exception_type": exception_type,
        "shape": getattr(value, "shape", "") or "",
        "phase": getattr(outcome, "phase", None) if isinstance(outcome, ReplayOutcome) else None,
        "reproduced": getattr(outcome, "reproduced", None)
        if isinstance(outcome, ReplayOutcome)
        else None,
    }


def _write_atomic(path: Path, text: str) -> None:
    """Replace *path* in one step so a reader never sees a partial document."""

    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class StoreSeedPass:
    """Offer the store's prior recipes for this goal as fresh candidates.

    A recipe that worked before is a good guess, not evidence.  It is published
    as a candidate so the ordinary replay pass re-certifies it against the code
    as it stands now -- which is the point, because a stored recipe outliving a
    change to the target is exactly the case a cached "validated" would get
    wrong.
    """

    store: "TemplateStore"
    name: str = "store-seed"
    consumes: frozenset[str] = frozenset({"object-goal"})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, _knowledge: KnowledgeBase, delta: tuple[Any, ...]
    ) -> Iterable[Emission]:
        for item in delta:
            goal = item.value
            if not isinstance(goal, ObjectGoal):
                continue
            yield from self.store.load(goal)
