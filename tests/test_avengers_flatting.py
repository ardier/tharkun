from __future__ import annotations

import datetime
import os

import pytest

from tharkun.avengers import ObjectGoal, synthesize_object
from tharkun.avengers.flatting import (
    FlatObjectTemplate,
    flatting_available,
    template_from_object,
)
from tharkun.avengers.passes import FLAT_TEMPLATE
from tharkun.knowledge import KnowledgeBase


pytestmark = pytest.mark.skipif(
    not flatting_available(), reason="py_obj_flatting is an optional Tharkun component"
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "avengers_target.py")


class LeafBox:
    def __init__(self, lower: int, upper: int):
        if lower >= upper:
            raise ValueError("invalid bounds")
        self.lower = lower
        self.upper = upper


class DateBox:
    def __init__(self, when: datetime.date):
        self.when = when


class InnerBox:
    def __init__(self, value: int):
        self.value = value


class OuterBox:
    def __init__(self, inner: InnerBox):
        self.inner = inner


class NoArgState:
    def __init__(self):
        self.value = 0


class NormalizingBox:
    def __init__(self, value: int):
        self.value = abs(value)


def test_valid_exemplar_becomes_an_addressable_roundtrip_template():
    template = template_from_object(LeafBox(-1, 2))
    assert template.leaf_paths == ("$.lower", "$.upper")
    assert template.recipe_blob.startswith(b"{")

    assignment = template.decode()
    assignment.update({"$.lower": -2, "$.upper": 3})
    data = template.reproduce(assignment)
    materialized = template.materialize(data, allow=(LeafBox,))

    assert materialized.ok
    assert isinstance(materialized.value, LeafBox)
    assert (materialized.value.lower, materialized.value.upper) == (-2, 3)
    assert materialized.leaves == assignment


def test_decomposed_fields_use_recipe_paths_and_reconstruct_the_exact_exemplar():
    when = datetime.date(2026, 9, 3)
    template = template_from_object(DateBox(when))

    assert template.leaf_paths == (
        "$.when.args[0]",
        "$.when.args[1]",
        "$.when.args[2]",
    )
    materialized = template.materialize(allow=(DateBox,))
    assert materialized.ok
    assert materialized.value.when == when


def test_nested_object_types_are_automatically_registered_for_certification():
    template = template_from_object(OuterBox(InnerBox(7)))

    assert template.leaf_paths == ("$.inner.value",)
    materialized = template.materialize(allow=(OuterBox, InnerBox))
    assert materialized.ok
    assert isinstance(materialized.value, OuterBox)
    assert isinstance(materialized.value.inner, InnerBox)
    assert materialized.value.inner.value == 7


def test_a_constructor_shape_that_cannot_rebuild_state_is_not_published():
    value = NoArgState()
    value.value = 7

    with pytest.raises(ValueError, match="could not reconstruct"):
        template_from_object(value)


def test_a_recipe_that_normalizes_to_different_state_is_not_published():
    value = object.__new__(NormalizingBox)
    value.value = -7

    with pytest.raises(ValueError, match="different exemplar"):
        template_from_object(value)


def test_flat_template_rejects_recipe_metadata_tampering():
    template = template_from_object(LeafBox(-1, 2))
    broken = FlatObjectTemplate(
        recipe_blob=template.recipe_blob,
        recipe_sha="0" * 64,
        total_budget=template.total_budget,
        n_leaves=template.n_leaves,
        baseline_data=template.baseline_data,
        exemplar_data=template.exemplar_data,
        exemplar_json=template.exemplar_json,
        leaf_paths=template.leaf_paths,
    )

    with pytest.raises(ValueError, match="SHA"):
        broken.recipe()


def test_flat_template_rejects_layout_metadata_tampering():
    template = template_from_object(LeafBox(-1, 2))
    broken = FlatObjectTemplate(
        recipe_blob=template.recipe_blob,
        recipe_sha=template.recipe_sha,
        total_budget=template.total_budget + 1,
        n_leaves=template.n_leaves,
        baseline_data=template.baseline_data,
        exemplar_data=template.exemplar_data,
        exemplar_json=template.exemplar_json,
        leaf_paths=template.leaf_paths,
    )

    with pytest.raises(ValueError, match="layout metadata"):
        broken.recipe()


def test_default_portfolio_publishes_flat_templates_when_component_is_present():
    knowledge = KnowledgeBase()
    report = synthesize_object(
        ObjectGoal(FIXTURE, "Box"),
        knowledge=knowledge,
        max_setup_steps=0,
        hypothesis_examples=30,
    )
    assert report.best is not None
    templates = knowledge.evidence(FLAT_TEMPLATE)
    assert templates
    assert all(isinstance(item.value, FlatObjectTemplate) for item in templates)
    assert all(item.value.key == item.key for item in templates)
    assert len({item.value.recipe_sha for item in templates}) <= len(templates)
