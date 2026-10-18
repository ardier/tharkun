"""Unused function bodies are dead; their definition-time defaults are not."""
import pytest

from tharkun.solve.census import obligation_census, site_census
from tharkun.solve.determinism import determinism_census


@pytest.mark.parametrize("header", ["def unused(x=1/0):", "def unused(*, x=1/0):", "async def unused(x=1/0):"])
def test_default_is_evaluated_even_when_function_is_unused(header):
    source = header + "\n    pass\n"
    with pytest.raises(ZeroDivisionError):
        exec(source, {})
    assert not site_census(source, ["ZeroDivisionError"]).no_site
    assert not obligation_census(source, ["ZeroDivisionError"]).complete


def test_literal_defaults_and_nested_dead_bodies():
    assert site_census("def unused(x=-1):\n    return 1/0\n", ["ZeroDivisionError"]).no_site
    assert site_census("def unused():\n    def inner(x=1/0):\n        pass\n", ["ZeroDivisionError"]).no_site


def test_unmodeled_default_scopes_cannot_hide_nondeterminism():
    # The parameter id does not shadow the builtin while the defaults evaluate.
    source = "def unused(x=id(0), id=0):\n    pass\n"
    assert not determinism_census(source).ok


def test_lambda_defaults_have_the_same_enclosing_scope_rule():
    source = "unused = lambda x=id(0), id=0: None\n"
    assert not determinism_census(source).ok
