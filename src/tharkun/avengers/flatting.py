"""Bridge from validated Avengers objects to ``py_obj_flatting`` recipes.

``py_obj_flatting`` is a first-class part of the Tharkun ecosystem but keeps
its own distribution and safety contract.  This module is the optional seam:
Avengers bootstraps a valid exemplar; flatting turns it into an addressable,
round-trippable object template.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class FlattingUnavailable(ImportError):
    """Raised when the optional ``py_obj_flatting`` component is absent."""


def flatting_available() -> bool:
    try:
        from py_obj_flatting import recipe
    except ImportError:
        return False
    return hasattr(recipe.Recipe, "from_json") and hasattr(recipe, "mine_seeded")


def _modules():
    try:
        import py_obj_flatting as pof
        from py_obj_flatting import recipe
    except ImportError as error:
        raise FlattingUnavailable(
            "py_obj_flatting is required for object templates"
        ) from error
    if not hasattr(recipe.Recipe, "from_json") or not hasattr(recipe, "mine_seeded"):
        raise FlattingUnavailable(
            "py_obj_flatting>=0.5.0 is required for validated recipe interchange"
        )
    return pof, recipe


@dataclass(frozen=True)
class FlatObjectTemplate:
    """A mined, byte-addressable recipe and its observed exemplar shape."""

    recipe_blob: bytes
    recipe_sha: str
    total_budget: int
    n_leaves: int
    baseline_data: bytes
    exemplar_data: bytes
    exemplar_json: str
    leaf_paths: tuple[str, ...]

    @property
    def key(self) -> str:
        # A recipe SHA identifies the generalized template. Distinct receiver
        # states can intentionally mine to the same template (for example a
        # bool field observed once false and once true), so the exact exemplar
        # bytes are part of the evidence identity as well.
        return hashlib.sha256(
            self.recipe_sha.encode("ascii") + b":" + self.exemplar_data
        ).hexdigest()

    def recipe(self):
        """Load and validate the template's versioned recipe document."""

        _pof, recipe_module = _modules()
        try:
            document = self.recipe_blob.decode("ascii")
        except (AttributeError, UnicodeDecodeError) as error:
            raise TypeError("flat object template does not contain recipe JSON") from error
        value = recipe_module.Recipe.from_json(document)
        if value.sha != self.recipe_sha:
            raise ValueError("flat object template recipe SHA does not match its metadata")
        if value.prefix != 0:
            raise ValueError("flat object template recipes must start at byte zero")
        if value.total_budget != self.total_budget or value.n_leaves != self.n_leaves:
            raise ValueError("flat object template layout metadata does not match its recipe")
        expected_size = value.prefix + value.total_budget
        if self.baseline_data != bytes(expected_size):
            raise ValueError("flat object template baseline data does not match its recipe")
        if len(self.exemplar_data) != expected_size:
            raise ValueError("flat object template exemplar data does not match its recipe")
        assignments = recipe_module.decode(value, self.exemplar_data)
        if tuple(sorted(assignments)) != self.leaf_paths:
            raise ValueError("flat object template leaf paths do not match its exemplar data")
        return value

    def decode(self, data: bytes | None = None) -> dict[str, Any]:
        """Decode bytes to leaf assignments without constructing an object."""

        _pof, recipe_module = _modules()
        return recipe_module.decode(self.recipe(), self.exemplar_data if data is None else data)

    def reproduce(self, assignments: Mapping[str, Any]) -> bytes:
        """Encode concrete leaf assignments for exact round-trip replay."""

        _pof, recipe_module = _modules()
        return recipe_module.reproduce(self.recipe(), dict(assignments))

    def materialize(
        self,
        data: bytes | None = None,
        *,
        allow: Iterable[Any] = (),
    ):
        """Build an object under py_obj_flatting's resolver/effect policy."""

        _pof, recipe_module = _modules()
        context = recipe_module.RuntimeContext(allow=tuple(allow))
        plan = recipe_module.compile_recipe(self.recipe(), context)
        return plan(self.exemplar_data if data is None else data)


def template_from_object(value: Any) -> FlatObjectTemplate:
    """Flatten one exemplar and certify its recipe in this interpreter.

    Certification reconstructs the exemplar, so it runs every constructor
    named by the recipe in the caller. Use ``IsolatedExecutor.mine_template``
    when those calls need process isolation.
    """

    pof, recipe_module = _modules()
    flattened = pof.flatten(value)
    seeded = recipe_module.mine_seeded(flattened)
    mined = seeded.recipe

    # Mining may change paths (a date becomes three constructor arguments)
    # and a class recipe may not match its constructor calling convention.
    # Register every exact object type observed by the static flattening pass:
    # a nested class recipe needs its own constructor as well as the root's.
    # This direct API certifies in the current interpreter. FlattingPass calls
    # it through IsolatedExecutor's disposable worker.
    context = recipe_module.RuntimeContext(allow=flattened.runtime_types)
    materialized = recipe_module.compile_recipe(mined, context)(seeded.data)
    if not materialized.ok:
        detail = "; ".join(
            failure.detail or failure.exc or failure.kind for failure in materialized.failures
        )
        raise ValueError("mined recipe could not reconstruct its exemplar: %s" % (detail,))
    rebuilt = pof.flatten(materialized.value)
    if rebuilt.canonical_json() != flattened.canonical_json():
        raise ValueError("mined recipe reconstructed a different exemplar")
    return FlatObjectTemplate(
        # Recipe JSON is a versioned, invariant-checked interchange format.
        # Pickle would execute constructors while merely reading evidence and
        # couples the parent to the worker's exact Python/class layout.
        recipe_blob=mined.to_json().encode("ascii"),
        recipe_sha=mined.sha,
        total_budget=mined.total_budget,
        n_leaves=mined.n_leaves,
        baseline_data=bytes(mined.total_budget),
        exemplar_data=seeded.data,
        exemplar_json=flattened.canonical_json(),
        leaf_paths=tuple(sorted(seeded.assignments)),
    )
