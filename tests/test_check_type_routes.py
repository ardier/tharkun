"""The certificate routes fail closed around the newly modeled verifier helper."""
from __future__ import annotations

import ast

import pytest

from tharkun.solve.check_type_hints import closed_hint_effects
from tharkun.solve.census import SITE_KINDS, site_census
from tharkun.solve.determinism import determinism_census
from tharkun.solve.drawflow import plan_draws
from tharkun.solve.pyresolve import Resolver
from tharkun.sv import product_proof


def test_closed_hint_grammar_uses_resolved_builtin_and_typing_names():
    source = """\
import typing as t
from typing import Optional as Maybe
x = list[t.Union[int, Maybe[str]]]
"""
    resolver = Resolver(ast.parse(source), {"typing"})
    subscript = next(n for n in ast.walk(resolver.tree)
                     if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                     and n.value.id == "list")
    assert closed_hint_effects(subscript, resolver) == frozenset({"iterate"})


@pytest.mark.parametrize("hint", [
    "int", "None", "typing.Any", "typing.Union[int, str]", "typing.Optional[float]",
    "int | None", "list[int]", "tuple[int, ...]", "dict[str, list[bool]]", "set[int]",
])
def test_closed_hint_grammar_accepts_the_finite_syntax_subset(hint):
    source = f"import typing\nh = {hint}\n"
    resolver = Resolver(ast.parse(source), {"typing"})
    node = ast.parse(hint, mode="eval").body
    assert closed_hint_effects(node, resolver) is not None, hint


@pytest.mark.parametrize("source", [
    "class Custom: pass\nh = Custom\n",
    "import typing\nh = typing.Iterable[int]\n",
    "import typing\nh = typing.Literal[1]\n",
    "import typing\nh = typing.List[int, str]\n",
])
def test_custom_rebound_or_unmodeled_hints_are_not_closed(source):
    resolver = Resolver(ast.parse(source), {"typing"})
    node = ast.parse(source).body[-1].value
    assert closed_hint_effects(node, resolver) is None, source


@pytest.mark.parametrize("exc", SITE_KINDS)
@pytest.mark.parametrize("source", [
    "import _sv_verifier\n_sv_verifier.check_type(1, int)\n",
    "import _sv_verifier as sv\nsv.check_type(1, int)\n",
    "from _sv_verifier import check_type as ct\nct(1, int)\n",
])
def test_census_counts_check_type_aliases_as_possible_sites_for_each_supported_class(exc, source):
    result = site_census(source, (exc,))
    assert not result.no_site, (exc, result)
    assert result.sites and "check_type" in result.sites[0][2], (exc, result.sites)


def test_census_keeps_typeerror_outside_its_supported_site_table():
    result = site_census("import _sv_verifier\n_sv_verifier.check_type(1, int)\n", ("TypeError",))
    assert not result.no_site and "no site table" in result.reason


@pytest.mark.parametrize("exc", SITE_KINDS)
def test_census_refuses_escaped_check_type_aliases(exc):
    source = "import _sv_verifier\ncheck = _sv_verifier.check_type\ncheck(1, int)\n"
    result = site_census(source, (exc,))
    assert not result.no_site and result.reason, (exc, result)


@pytest.mark.parametrize("source", [
    "import _sv_verifier, typing\ntyping = object()\n_sv_verifier.check_type(1, typing.Any)\n",
    "import _sv_verifier\nfrom typing import Any as Hint\nHint = int\n_sv_verifier.check_type(1, Hint)\n",
    "import _sv_verifier\nint = object\n_sv_verifier.check_type(1, int)\n",
])
def test_rebound_hint_names_fail_closed_in_both_routes(source):
    # Resolver may reject a conflicting import during construction; both
    # public routes report that as a refusal rather than a certificate.
    census = site_census(source, ("AssertionError",))
    assert not census.no_site and ("does not resolve" in census.reason or
                                   "not closed" in census.reason), census
    result = determinism_census(source)
    assert not result.ok, result


@pytest.mark.parametrize("source", [
    "import _sv_verifier\n_sv_verifier.check_type(1, int)\n",
    "from _sv_verifier import check_type as check\ncheck([1, 2], list[int])\n",
    "import _sv_verifier, typing\n_sv_verifier.check_type({'x': 1}, dict[str, typing.Optional[int]])\n",
])
def test_determinism_certifies_closed_hints_and_safe_values(source):
    assert determinism_census(source).ok, determinism_census(source).reason


@pytest.mark.parametrize("source", [
    "import _sv_verifier\n_sv_verifier.check_type(object(), int)\n",
    "import _sv_verifier\nclass A:\n    def __repr__(self): return str(id(self))\n_sv_verifier.check_type(A(), int)\n",
    "import _sv_verifier\nclass A(list):\n    def __iter__(self): raise KeyError()\n_sv_verifier.check_type(A([1]), list[int])\n",
    "import _sv_verifier\nclass D(dict):\n    def items(self): raise IndexError()\n_sv_verifier.check_type(D({'x': 1}), dict[str, int])\n",
    "import _sv_verifier, typing\nclass A:\n    def __eq__(self, other): raise AssertionError()\n_sv_verifier.check_type(A(), typing.Literal[1])\n",
    "import _sv_verifier\nclass A: pass\n_sv_verifier.check_type(1, A)\n",
    "import _sv_verifier\nint = object\n_sv_verifier.check_type(1, int)\n",
    "import _sv_verifier, typing\ntyping = object()\n_sv_verifier.check_type(1, typing.Any)\n",
    "import _sv_verifier\nfrom typing import Any as Hint\nHint = int\n_sv_verifier.check_type(1, Hint)\n",
    "import _sv_verifier\n_sv_verifier.check_type({1}, set[int])\n",
    "import _sv_verifier\nhint = int\nhint = object\n_sv_verifier.check_type(1, hint)\n",
    "import _sv_verifier\n_sv_verifier.check_type([1], list[int], extra=True)\n",
    "import _sv_verifier\ncheck = _sv_verifier.check_type\ncheck(1, int)\n",
])
def test_determinism_refuses_callbacks_addresses_and_unclosed_forms(source):
    result = determinism_census(source)
    assert not result.ok, result


@pytest.mark.parametrize("source", [
    "import _sv_verifier\nclass A:\n    def __repr__(self): return str(id(self))\n_sv_verifier.check_type(A(), int)\nassert False\n",
    "import _sv_verifier\nclass A(list):\n    def __iter__(self): raise KeyError()\n_sv_verifier.check_type(A([1]), list[int])\nassert False\n",
    "import _sv_verifier, typing\nclass A:\n    def __eq__(self, other): raise AssertionError()\n_sv_verifier.check_type(A(), typing.Literal[1])\nassert False\n",
])
def test_product_proof_refuses_check_type_callback_and_address_cases(tmp_path, source):
    program = tmp_path / "check_type_unsafe.py"
    program.write_text(source)
    assert product_proof.prove(source, ("AssertionError",), program=str(program)) is None


def test_drawflow_keeps_check_type_argument_reads_out_of_the_draw_allowlist():
    source = "import _sv_verifier\nx = _sv_verifier.nondet_int()\n_sv_verifier.check_type(x, int)\n"
    result = plan_draws(source)
    assert not result.ok, result
