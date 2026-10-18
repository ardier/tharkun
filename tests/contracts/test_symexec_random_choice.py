"""The symbolic executor reads the contract of ``random.choice`` ([internal ref] step 2, PR "symexec random.choice").

The harness lifts ``random.choice(seq)`` to ``__sv_index(kind, idx, seq)`` with ``idx`` a draw (``sv/sites.py``); the executor then picked
the element with inline code that never read the contract (the audit's ``M`` cell).  It now goes through the rule:
``_Exec._through_rule("random.choice")`` -> ``Rule.accept`` -> ``SymexecVisitor._builtin`` (a TESTED contract, else a refusal) ->
``_model_random_choice``: the base class's (an empty sequence refuses, an index is an ite chain) and ``_Runner``'s (a symbolic index is
bounded ``0 <= idx < len`` on the path, an empty sequence ends the path as ``IndexError`` does).

What this file pins, against CPython 3.14.4:

* a guard on an element the sequence holds is solved, and the model's index selects exactly that element; a guard on a value the sequence
  does not hold is not reached;
* an empty sequence is ``IndexError`` on CPython and ends the path in the executor (it never reaches what follows);
* the contract is what admits the call: untested or withdrawn, it is a refusal, not the model of old;
* NEGATIVE CONTROL: a model that always answers the first element is caught by the guard on another element.
"""

from __future__ import annotations

import ast
import random

import pytest

from tests.contracts._native import run_native
from tests.contracts.test_engine_audit_consumed import _solve, untested
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.sv import _symexec


def _sat(src: str, guard: str):
    got, h = _solve(src, guard)
    return [m for o, m in got.values() if o == "sat"], h


@pytest.mark.parametrize("seq,element", [("[10, 20, 30]", 10), ("[10, 20, 30]", 20), ("[10, 20, 30]", 30), ("(7, 8)", 8), ("[5]", 5)])
def test_a_guard_on_an_element_the_sequence_holds_is_solved(seq, element):
    sat, _h = _sat(f"import random\nx = random.choice({seq})\n", f"x == {element}")
    assert sat and sat[0]["n"] == 3, (seq, element)
    assert element in eval(seq) and all(random.choice(eval(seq)) in eval(seq) for _ in range(50))


@pytest.mark.parametrize("seq,element", [("[10, 20, 30]", 99), ("(7, 8)", 7 + 8), ("[5]", 6)])
def test_a_guard_on_a_value_the_sequence_does_not_hold_is_not_reached(seq, element):
    sat, _h = _sat(f"import random\nx = random.choice({seq})\n", f"x == {element}")
    assert not sat, (seq, element)


def test_the_solved_index_selects_the_element():
    """The model of the draw for ``x == 20`` is the index of 20: the element the index selects on CPython is 20."""
    sat, h = _sat("import random\nx = random.choice([10, 20, 30])\n", "x == 20")
    assert sat
    index = [sat[0][p] for p, k in zip(h.params, h.kinds) if k == "rindex"]
    assert index == [1], (h.params, h.kinds, sat[0])
    assert [10, 20, 30][index[0]] == 20


def test_an_empty_sequence_is_an_indexerror_on_cpython_and_ends_the_path():
    with pytest.raises(IndexError):
        random.choice([])
    assert run_native("import random\nx = random.choice([])\nraise ValueError('after')\n") == "IndexError"
    sat, _h = _sat("import random\nx = random.choice([])\n", "True")
    assert not sat


def _runner(src: str):
    tree = ast.parse("def __sv_harness():\n    pass\n")
    r = _symexec._Runner(tree, {}, {}, {}, [])
    r.run(list(ast.parse(src).body))
    return r


@pytest.mark.parametrize("qualname_state", ["untested", "withdrawn"])
def test_an_untested_or_withdrawn_contract_is_a_refusal_not_the_model_of_old(qualname_state, monkeypatch):
    src = "import random\nx = random.choice([10, 20, 30])\n"
    load_all()
    sat, _h = _sat(src, "x == 20")
    assert sat
    if qualname_state == "untested":
        monkeypatch.setitem(REGISTRY, "random.choice", untested(REGISTRY["random.choice"]))
    else:
        monkeypatch.delitem(REGISTRY, "random.choice")
    sat, _h = _sat(src, "x == 20")
    assert not sat, qualname_state


def test_negative_control_a_model_that_ignores_the_index_is_caught(monkeypatch):
    """The control: a ``random.choice`` that answers the first element whatever the draw makes ``x == 20`` unsolvable (the real model
    solves it with index 1): the tests above fail for it."""
    monkeypatch.setattr(_symexec._Runner, "_model_random_choice", lambda self, n: self.ev(n.args[2])[0])
    sat, _h = _sat("import random\nx = random.choice([10, 20, 30])\n", "x == 20")
    assert not sat
