"""Finite correlated calls: native counterexamples precede refusal checks."""
import ast
import os
import subprocess
import sys
from textwrap import dedent

import pytest

from tharkun.solve.determinism import determinism_census
from tharkun.solve.pyresolve import Resolver
from tharkun.solve.set_singleton import singleton_receivers
from tests.test_address_provenance import renamed


CHOOSE = """\
def choose(left=10, /, *, right=30):
    remaining = ({10, 20, 30} - {left} - {right}).pop()
    return remaining
"""

WALK = """\
def moves(depth, left=10, right=30):
    result = []
    if depth > 0:
        middle = ({10, 20, 30} - {left} - {right}).pop()
        result.extend(moves(depth - 1, left, middle))
        result.append((left, right))
        result.extend(moves(depth - 1, middle, right))
    return result
def check(arguments, expected):
    assert moves(*arguments) == expected
check([0, 10, 30], [])
check([1, 10, 30], [(10, 30)])
check([2, 10, 30], [(10, 20), (10, 30), (20, 30)])
check([3, 10, 30], [(10, 30), (10, 20), (30, 20), (10, 30),
                    (20, 10), (20, 30), (10, 30)])
"""


def native(source):
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONSTARTUP"}}
    return subprocess.run([sys.executable, "-B", "-c", source], text=True,
                          capture_output=True, timeout=5, env=env)


def proved(source):
    tree = ast.parse(source)
    return singleton_receivers(Resolver(tree, ()))


@pytest.mark.parametrize("call,cardinality", [
    ("choose(10, right=10)", 2),
    ("choose(10, right=99)", 2),
    ("choose(98, right=99)", 3),
])
def test_native_bad_endpoint_rows_are_not_singletons(call, cardinality):
    source = CHOOSE + "choose()\n" + call
    observed = source.replace("}).pop()", "})\n    print(len(remaining))")
    result = native(observed)
    assert result.returncode == 0 and result.stdout == f"1\n{cardinality}\n", result.stderr
    assert not proved(source)
    assert not determinism_census(source).ok


@pytest.mark.parametrize("body,output,error", [
    ("""
        class Height:
            def __gt__(self, other):
                print('comparison')
                return False
        moves(Height())
    """, "comparison\n", None),
    ("""
        class Height:
            def __gt__(self, other): return True
            def __sub__(self, other):
                print('subtraction')
                return 0
        moves(Height())
    """, "subtraction\nsubtraction\n", None),
    ("""
        class Key:
            def __hash__(self):
                print('hash')
                raise AssertionError
        choose(Key())
    """, "hash\n", "AssertionError"),
    ("""
        class Key:
            def __hash__(self): return 10
            def __eq__(self, other):
                print('equality')
                raise AssertionError
        choose(Key())
    """, "equality\n", "AssertionError"),
    ("""
        class Arguments:
            def __iter__(self):
                print('iteration')
                return iter((10,))
        choose(*Arguments())
    """, "iteration\n", None),
    ("""
        def compare(self, other):
            print('installed comparison')
            return False
        class Height: pass
        Height.__gt__ = compare
        moves(Height())
    """, "installed comparison\n", None),
])
def test_native_operand_and_expansion_effects_refuse(body, output, error):
    source = CHOOSE + WALK + dedent(body)
    result = native(source)
    assert result.stdout == output, result.stderr
    assert (result.returncode == 0) if error is None else (error in result.stderr)
    assert not proved(source)
    assert not determinism_census(source).ok


@pytest.mark.parametrize("tail", [
    "choose(10, side_effect(), right=30)",
    "def unused(value=side_effect()): pass",
    "def ignore(value): return choose()\nignore(side_effect())",
])
def test_native_rejected_arity_and_unused_default_effects_refuse(tail):
    source = CHOOSE + dedent("""
        def side_effect():
            print('effect')
            return 0
        choose()
    """) + tail
    result = native(source)
    assert result.stdout == "effect\n"
    if tail.startswith("choose"):
        assert result.returncode != 0 and "TypeError" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
    assert not proved(source)
    assert not determinism_census(source).ok


@pytest.mark.parametrize("replacement", [
    "return ()", "return None", "return 0",
])
def test_native_recursive_return_shape_is_not_assumed(replacement):
    source = WALK.replace("    return result", "    " + replacement)
    result = native(source)
    assert result.returncode != 0 and ("TypeError" in result.stderr or "AssertionError" in result.stderr)
    assert not proved(source)
    assert not determinism_census(source).ok


def test_native_recursive_elements_can_invoke_equality():
    source = dedent("""
        class Element:
            def __eq__(self, other):
                print('element equality')
                return False
    """) + WALK.replace("result.append((left, right))", "result.append(Element())")
    result = native(source)
    assert result.returncode != 0 and result.stdout == "element equality\n"
    assert "AssertionError" in result.stderr
    assert not proved(source)
    assert not determinism_census(source).ok


@pytest.mark.parametrize("tail", [
    "alias = arguments\nalias[0] = 20",
    "alias = arguments\nalias.reverse()",
    "def change(values): values[0] = 20\nchange(arguments)",
])
def test_native_same_type_carrier_mutation_refuses(tail):
    source = CHOOSE + "arguments = [10]\n" + tail + "\nchoose(*arguments)"
    # Use two slots for reverse so the mutation has an observable effect.
    if "reverse" in tail:
        source = source.replace("[10]", "[10, 20]")
        source = source.replace("choose(*arguments)", "choose(arguments[0])")
    result = native(source + "\nprint(arguments[0])")
    assert result.returncode == 0 and result.stdout == "20\n", result.stderr
    assert not proved(source)
    assert not determinism_census(source).ok


def test_native_builder_alias_cannot_be_an_argument_carrier():
    source = CHOOSE + dedent("""
        def relay():
            values = []
            alias = values
            alias.append(10)
            return choose(*values)
        assert relay() == 20
    """)
    result = native(source)
    assert result.returncode == 0, result.stderr
    assert not proved(source)
    assert not determinism_census(source).ok


@pytest.mark.parametrize("source", [
    CHOOSE + "assert choose() == 20",
    CHOOSE + "args = [10]\nalias = args\nassert choose(*alias, right=30) == 20",
    CHOOSE + "def relay(args, target): return choose(*args, right=target)\n"
             "assert relay((10,), 30) == 20\nassert relay((30,), 10) == 20",
    WALK,
    WALK.replace("10", "-7").replace("20", "11").replace("30", "29"),
])
def test_complete_rows_keep_correlations_defaults_and_recursive_returns(source):
    for program in (source, renamed(source)):
        result = native(program)
        assert result.returncode == 0, result.stderr
        assert proved(program)
        certificate = determinism_census(program)
        assert certificate.ok, certificate.reason


@pytest.mark.parametrize("tail", [
    "def unused(value=choose(10, right=10)): pass",
    "choose(10, right=10)",
    "def relay(args): return choose(*args, right=10)\nrelay([10])",
    "def other(left=10, right=10): return ({10,20,30}-{left}-{right}).pop()\nother()",
])
def test_every_incoming_default_and_second_receiver_must_close(tail):
    source = CHOOSE + "choose()\n" + tail
    assert not proved(source)
    assert not determinism_census(source).ok


def test_both_recursive_transitions_are_checked():
    source = WALK.replace("moves(depth - 1, middle, right)", "moves(depth - 1, left, left)")
    assert not proved(source)
    assert not determinism_census(source).ok


@pytest.mark.parametrize("source", [
    CHOOSE + "def relay(choose): return choose()\nrelay(10)",
    CHOOSE + "def relay():\n    choose()\n    choose = 10\nrelay()",
    CHOOSE + "choose()\nchoose = 10",
    CHOOSE + "args = [10]\nchoose(*args)\nargs = [20]",
])
def test_scope_and_all_write_closure_cannot_borrow_other_bindings(source):
    assert not proved(source)
    assert not determinism_census(source).ok


@pytest.mark.parametrize("source", [
    WALK[:WALK.index("def check")] + "moves(0)",  # pop never reached
    CHOOSE + "alias = choose\nalias()",
    CHOOSE + "choose(**{'right': 30})",
    CHOOSE + "choose(True)",
    CHOOSE + "choose(10.0)",
    CHOOSE + "def f(): return f()\nchoose()\nf()",
    CHOOSE + "left = 10\nright = 30\nresidual = {10,20,30} - {left} - {right}\n"
             "alias = residual\nalias.pop()",
    CHOOSE + "method = ({10,20,30}-{10}-{30}).pop\nmethod()",
    CHOOSE + "choose()\nvalue = missing",
    CHOOSE + "choose()\ndef f(value: choose(10, right=10)): pass",
    CHOOSE + "choose()\nvalues = []\nvalues.append(values)",
])
def test_unknown_empty_aliased_and_unsupported_shapes_refuse(source):
    assert not proved(source)
    assert not determinism_census(source).ok


def test_inherited_literal_set_census_does_not_claim_leaf_proof():
    # Preserve this exact safe singleton case, not arbitrary named-set uses.
    source = CHOOSE + "remaining = {10,20,30}-{10}-{30}\nremaining.pop()"
    observed = source.replace("\nremaining.pop()", "\nprint(remaining.pop())")
    result = native(observed)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "20\n", (result.stdout, result.stderr)
    assert not proved(source)
    assert determinism_census(source).ok


def test_missing_entry_relation_has_no_receiver_exemption():
    assert not proved(CHOOSE)
    # The ordinary census may ignore a genuinely unused function body.
    assert determinism_census(CHOOSE).ok


@pytest.mark.parametrize("expression", [
    "left is right", "left is not right", "id(left)", "hash(left)", "repr(left)",
])
def test_native_identity_and_representation_observations_refuse(expression):
    source = CHOOSE + "left = [10]\nright = left\nchoose()\n" + expression
    # Hash needs a hashable operand; identity must never use analyzer copies.
    if expression == "hash(left)":
        source = source.replace("left = [10]", "left = (10,)")
    result = native(source)
    assert result.returncode == 0, result.stderr
    assert not proved(source)
    assert not determinism_census(source).ok


def test_finite_limits_are_refusals_not_program_preconditions():
    state_source = ("def choose(extra, left=10, right=30):\n"
                    "    return ({10,20,30}-{left}-{right}).pop()\n"
                    + "\n".join(f"choose({i})" for i in range(65)))
    edge_source = CHOOSE + "\nchoose()" * 1001
    atom_source = CHOOSE.replace("30", str(1 << 65)) + "choose()"
    universe_source = "value = ({0,1,2,3,4,5,6,7,8} - {0,1,2,3,4,5,6,7}).pop()"
    alias_source = (CHOOSE + "choose()\na0 = [10]\n"
                    + "\n".join(f"a{i} = [a{i-1}, a{i-1}]" for i in range(1, 9)))
    for source in (state_source, edge_source, atom_source, universe_source, alias_source):
        result = native(source)
        assert result.returncode == 0, result.stderr
        assert not proved(source)
        assert not determinism_census(source).ok


def test_receiver_exemption_does_not_relax_other_set_sinks():
    source = CHOOSE + "choose()\nvalues = {'alpha', 'beta'}\nsequence = list(values)"
    assert not determinism_census(source).ok


def test_a_deterministic_assertion_violation_remains_false(tmp_path):
    from tharkun.sv.engine import verify

    source = CHOOSE + "assert choose() == 10"
    result = native(source)
    assert result.returncode != 0 and "AssertionError" in result.stderr
    assert determinism_census(source).ok
    path = tmp_path / "wrong_expected.py"
    path.write_text(source)
    verdict = verify(str(path), ("AssertionError",), budget_seconds=10, jobs=1, prove=False)
    assert verdict.line() == "FALSE(AssertionError)", verdict
