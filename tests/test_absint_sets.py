"""``set`` and ``frozenset`` in absint (``solve/absint/sets.py``, ``heap.py::SetObj``, docs/ABSINT.md "Sets").

Four kinds of test:

* exactness: the facts a model claims are the facts the operation has (a union is at least as long as each side,
  an empty set holds nothing, ``True`` and ``1`` are one element) and what it cannot know stays UNKNOWN;
* the checks the contracts name: ``KeyError`` of ``remove`` and ``pop`` is a recorded check that is proven only
  where the element (or an element) is certainly there; an unhashable element, a mutating method on a frozenset
  and an operator on a non-set are ``TypeError``/``AttributeError`` checks, never proven away;
* named refusals (``Unsupported``: the verdict is UNKNOWN and says why);
* a differential: every generated program is also run natively over every draw that matters, and a TRUE for a
  class the program raises fails the test.  The negative control breaks the model on purpose (``discard`` that
  drops the length lower bound, ``pop`` that proves ``KeyError`` away, ``add`` that forgets the element) and
  the differential must catch each.
"""

from __future__ import annotations

import itertools
import random
import sys
import textwrap
import types

import pytest

from tharkun.solve.absint import prove

HEAD = "import _sv_verifier\ndef nd():\n    return _sv_verifier.nondet_int()\n"
ALL = ("TypeError", "AttributeError", "IndexError", "KeyError", "ZeroDivisionError", "FloatingPointError",
       "AssertionError")


def result(body: str, classes):
    return prove(HEAD + textwrap.dedent(body), list(classes), filename="<program>")


def verdict(body: str, cls="AssertionError") -> str:
    return result(body, [cls]).verdict


# ---------------------------------------------------------------------------
# construction and length
# ---------------------------------------------------------------------------


def test_a_display_collapses_equal_elements_and_its_length_is_exact_when_every_element_is_a_constant():
    assert verdict("s = {1, 2, 3}\nassert len(s) == 3\n") == "TRUE"
    assert verdict("s = {1, 1, 1}\nassert len(s) == 1\n") == "TRUE"
    assert verdict("s = {1, True, 1.0}\nassert len(s) == 1\n") == "UNKNOWN"      # a float is not a constant here
    assert verdict("s = {1, True}\nassert len(s) == 1\n") == "TRUE"             # True == 1: one element
    assert verdict("s = {'a', 'a', 'b'}\nassert len(s) == 2\n") == "TRUE"
    assert verdict("s = {None, None}\nassert len(s) == 1\n") == "TRUE"
    assert verdict("s = {nd(), nd()}\nassert len(s) >= 1\nassert len(s) <= 2\n") == "TRUE"
    assert verdict("s = {nd(), nd()}\nassert len(s) == 2\n") == "UNKNOWN"


def test_a_constructor_bounds_the_length_by_the_argument():
    assert verdict("s = set()\nassert len(s) == 0\n") == "TRUE"
    assert verdict("s = set([1, 2, 2])\nassert len(s) >= 1\nassert len(s) <= 3\n") == "TRUE"
    assert verdict("s = set([1, 2, 2])\nassert len(s) == 2\n") == "UNKNOWN"
    assert verdict("s = set(range(5))\nassert len(s) == 5\n") == "TRUE"            # a range's elements are distinct
    assert verdict("s = set((7, 8))\nassert len(s) >= 1\nassert len(s) <= 2\n") == "TRUE"
    assert verdict("s = set('abc')\nassert len(s) >= 1\nassert len(s) <= 3\n") == "TRUE"
    assert verdict("d = {1: 2, 3: 4}\ns = set(d)\nassert len(s) == 2\n") == "TRUE"   # a dict's keys are distinct
    assert verdict("s = set([])\nassert len(s) == 0\n") == "TRUE"
    assert verdict("f = frozenset([1, 2])\nassert len(f) >= 1\nassert len(f) <= 2\n") == "TRUE"
    assert verdict("f = frozenset()\nassert len(f) == 0\n") == "TRUE"
    assert verdict("s = set(set([4, 5]))\nassert len(s) <= 2\n") == "TRUE"


def test_a_comprehension_is_the_loop_it_lowers_to():
    # the loop adds one element per iteration and the number of iterations is not a bound of the length
    assert verdict("s = {x for x in [1, 2, 3]}\nassert len(s) >= 0\n") == "TRUE"
    assert verdict("s = {x for x in [1, 2, 3]}\nassert len(s) <= 3\n") == "UNKNOWN"
    assert verdict("s = {x % 2 for x in range(5)}\nassert len(s) >= 0\n") == "TRUE"
    assert verdict("s = {x for x in range(3)}\nc = s.pop()\n", "KeyError") == "TRUE"      # the first iteration is peeled
    assert verdict("s = {x for x in range(nd())}\nc = s.pop()\n", "KeyError") == "UNKNOWN"


def test_the_element_summary_is_what_the_elements_satisfy():
    assert verdict("s = {1, 2, 3}\nfor x in s:\n    assert x >= 1\n    assert x <= 3\n") == "TRUE"
    assert verdict("s = {1, 2, 3}\nfor x in s:\n    assert x > 1\n") == "UNKNOWN"
    assert verdict("s = {1, 2}\ns.add(10)\nfor x in s:\n    assert x <= 10\n") == "TRUE"
    assert verdict("s = {1, 2}\ns.add(10)\nfor x in s:\n    assert x <= 2\n") == "UNKNOWN"
    assert verdict("s = {4}\nx = s.pop()\nassert x == 4\n") == "TRUE"


# ---------------------------------------------------------------------------
# the mutating methods
# ---------------------------------------------------------------------------


def test_add_discard_and_clear_move_the_length_bounds_exactly():
    assert verdict("s = set()\ns.add(1)\nassert len(s) == 1\n") == "TRUE"
    assert verdict("s = {1}\ns.add(1)\nassert len(s) >= 1\nassert len(s) <= 2\n") == "TRUE"
    assert verdict("s = {1}\ns.add(1)\nassert len(s) == 1\n") == "UNKNOWN"          # the summary cannot tell
    assert verdict("s = {1, 2}\ns.discard(1)\nassert len(s) >= 1\nassert len(s) <= 2\n") == "TRUE"
    assert verdict("s = {1, 2}\ns.discard(1)\nassert len(s) == 1\n") == "UNKNOWN"
    assert verdict("s = {1}\ns.discard(7)\nassert len(s) <= 1\n") == "TRUE"
    assert verdict("s = {1, 2}\ns.clear()\nassert len(s) == 0\n") == "TRUE"
    assert verdict("s = {1, 2}\ns.clear()\ns.add(5)\nassert len(s) == 1\n") == "TRUE"


def test_add_forgets_nothing_it_was_given():
    assert verdict("s = {1}\ns.add(nd())\nfor x in s:\n    assert x == 1\n") == "UNKNOWN"
    assert verdict("s = set()\ns.add(2)\ns.add(3)\nfor x in s:\n    assert x >= 2\n    assert x <= 3\n") == "TRUE"


def test_remove_and_pop_keep_the_keyerror_check():
    # the element is certainly there: nothing raises
    assert verdict("s = {1}\ns.remove(1)\n", "KeyError") == "TRUE"
    assert verdict("s = {1}\ns.remove(True)\n", "KeyError") == "TRUE"             # True == 1
    assert verdict("s = set()\ns.add(5)\ns.remove(5)\n", "KeyError") == "TRUE"
    # it may be absent: a recorded, unproven check (never silently proven away)
    assert verdict("s = {1}\ns.remove(2)\n", "KeyError") == "UNKNOWN"
    assert verdict("s = {1, 2}\ns.remove(1)\n", "KeyError") == "UNKNOWN"          # the summary is [1, 2]
    assert verdict("s = {1}\ns.remove(nd())\n", "KeyError") == "UNKNOWN"
    assert verdict("s = set()\ns.remove(1)\n", "KeyError") == "UNKNOWN"
    assert verdict("s = {1}\ns.discard(2)\n", "KeyError") == "TRUE"               # discard never raises it
    # pop: proven only with a lower bound of 1
    assert verdict("s = {1}\ns.pop()\n", "KeyError") == "TRUE"
    assert verdict("s = set()\ns.pop()\n", "KeyError") == "UNKNOWN"
    assert verdict("s = {1}\ns.pop()\ns.pop()\n", "KeyError") == "UNKNOWN"
    assert verdict("s = {1, 2}\ns.pop()\ns.pop()\n", "KeyError") == "TRUE"        # {1, 2} has two elements: exactly
    assert verdict("s = {1, 2}\ns.pop()\ns.pop()\ns.pop()\n", "KeyError") == "UNKNOWN"
    assert verdict("s = {nd(), nd()}\ns.pop()\ns.pop()\n", "KeyError") == "UNKNOWN"  # they may be one element
    assert verdict("s = set()\ns.add(1)\ns.pop()\n", "KeyError") == "TRUE"
    assert verdict("s = {1}\ns.clear()\ns.pop()\n", "KeyError") == "UNKNOWN"
    assert verdict("s = {nd()}\nif nd() > 0:\n    s.clear()\ns.pop()\n", "KeyError") == "UNKNOWN"


def test_remove_and_pop_end_the_path_that_raises_so_the_rest_is_reached_only_when_the_element_was_there():
    # the path past remove(2) of {1} does not exist (a KeyError), so nothing after it can fail there
    assert verdict("s = set()\ns.remove(2)\nassert False\n") == "TRUE"
    assert verdict("s = set()\nx = s.pop()\nassert False\n") == "TRUE"
    # but the property is KeyError there, and that check is what the verdict is about
    assert verdict("s = set()\ns.remove(2)\nassert False\n", "KeyError") == "UNKNOWN"
    # absent from a non-empty set is not decided (the summary says what is in it, not what is not): the path continues
    assert verdict("s = {1}\ns.remove(2)\nassert False\n") == "UNKNOWN"


def test_pop_returns_an_element_and_removes_one():
    assert verdict("s = {3}\nx = s.pop()\nassert x == 3\nassert len(s) == 0\n") == "TRUE"
    assert verdict("s = {3, 4}\nx = s.pop()\nassert x >= 3\nassert x <= 4\nassert len(s) <= 1\n") == "TRUE"
    assert verdict("s = {3, 4}\nx = s.pop()\nassert x == 3\n") == "UNKNOWN"
    assert verdict("s = {1, 2}\nx = s.pop()\nc = 10 // x\n", "ZeroDivisionError") == "TRUE"
    assert verdict("s = {0, 2}\nx = s.pop()\nc = 10 // x\n", "ZeroDivisionError") == "UNKNOWN"


def test_a_set_alias_sees_every_update():
    assert verdict("s = {1}\nt = s\nt.add(2)\nassert len(s) >= 1\nassert len(s) <= 2\n") == "TRUE"
    assert verdict("s = {1}\nt = s\nt.clear()\nassert len(s) == 0\n") == "TRUE"
    assert verdict("s = {1}\nt = s\ns.clear()\nt.pop()\n", "KeyError") == "UNKNOWN"
    assert verdict("s = {1}\nt = s.copy()\nt.clear()\nassert len(s) == 1\n") == "TRUE"
    assert verdict("s = {1}\nt = s.copy()\nt.add(2)\nassert len(s) <= 1\n") == "TRUE"


def test_a_summary_takes_weak_updates():
    body = """
        ss = []
        for i in range(3):
            ss.append(set())
        ss[0].add(5)
        assert len(ss[1]) == 0
    """
    assert verdict(body) == "UNKNOWN"          # one of the three has the element; which one is not tracked
    assert verdict("ss = [set(), set()]\nx = ss[nd() % 2]\nx.add(5)\nassert len(ss[0]) == 0\n") == "UNKNOWN"
    assert verdict(body.replace("assert len(ss[1]) == 0", "assert len(ss[1]) <= 1")) == "TRUE"


# ---------------------------------------------------------------------------
# the methods that take iterables
# ---------------------------------------------------------------------------


def test_union_difference_intersection_and_symmetric_difference_bound_the_length():
    assert verdict("a = {1, 2}\nb = a.union([3])\nassert len(b) >= 2\nassert len(b) <= 3\n") == "TRUE"
    assert verdict("a = {1, 2}\nb = a.union()\nassert len(b) == 2\n") == "TRUE"
    assert verdict("a = {1, 2}\nb = a.difference([2])\nassert len(b) <= 2\n") == "TRUE"
    assert verdict("a = {1, 2}\nb = a.difference([2])\nassert len(b) == 1\n") == "UNKNOWN"
    assert verdict("a = {1, 2}\nb = a.difference([5, 6, 7])\nassert len(b) >= 0\nassert len(b) <= 2\n") == "TRUE"
    assert verdict("a = {1, 2}\nb = a.intersection([2, 3, 4])\nassert len(b) <= 2\n") == "TRUE"
    assert verdict("a = {1, 2}\nb = a.intersection([2])\nassert len(b) <= 1\n") == "TRUE"
    assert verdict("a = {1, 2}\nb = a.symmetric_difference([2, 3])\nassert len(b) <= 4\n") == "TRUE"
    assert verdict("a = {1, 2}\nb = a.symmetric_difference([2, 3])\nassert len(b) <= 3\n") == "UNKNOWN"   # a list may be 2 distinct
    assert verdict("a = {1, 2, 3}\nb = a.symmetric_difference({1})\nassert len(b) >= 1\n") == "TRUE"
    assert verdict("a = {1, 2, 3}\nb = a.symmetric_difference({1})\nassert len(b) >= 2\n") == "TRUE"


def test_the_retained_element_of_equal_values_is_either_one():
    # {1}.intersection([1.0]) is {1.0}: the argument's object, so the elements join both sides
    body = "a = {1}\nb = a.intersection([1.0])\nfor x in b:\n    assert CHECK\n"
    assert verdict(body.replace("CHECK", "x >= 1")) == "TRUE"
    assert verdict(body.replace("CHECK", "x > 1")) == "UNKNOWN"
    assert verdict("a = {1}\nb = a.union([2])\nfor x in b:\n    assert x >= 1\n    assert x <= 2\n") == "TRUE"


def test_the_update_methods_change_the_receiver_in_place():
    assert verdict("s = {1}\ns.update([2, 3])\nassert len(s) >= 1\nassert len(s) <= 3\n") == "TRUE"
    assert verdict("s = {1}\nt = s\ns.update([2])\nassert len(t) <= 2\n") == "TRUE"
    assert verdict("s = {1, 2, 3}\ns.difference_update([1])\nassert len(s) <= 3\n") == "TRUE"
    assert verdict("s = {1, 2, 3}\ns.intersection_update([1])\nassert len(s) <= 1\n") == "TRUE"
    assert verdict("s = {1, 2}\ns.symmetric_difference_update([2, 3])\nassert len(s) <= 4\n") == "TRUE"
    assert verdict("s = {1, 2}\ns.update()\nassert len(s) == 2\n") == "TRUE"
    assert verdict("s = {1, 2}\ns.update(s)\nassert len(s) <= 4\nassert len(s) >= 1\n") == "TRUE"
    assert verdict("s = {1, 2}\ns.difference_update(s)\nassert len(s) <= 2\n") == "TRUE"


def test_the_predicates_decide_at_the_ends():
    assert verdict("assert set().issubset([1])\nassert set().isdisjoint([1])\nassert {1}.issuperset([])\n") == "TRUE"
    assert verdict("assert not {1, 2}.issubset([1])\n") == "TRUE"           # two elements cannot fit in one
    assert verdict("assert not {1}.issuperset([1, 2])\n") == "UNKNOWN"      # a list may repeat an element
    assert verdict("assert not {1}.issuperset({1, 2})\n") == "TRUE"         # a set's elements are distinct
    assert verdict("assert {1}.issubset([1, 2])\n") == "UNKNOWN"
    assert verdict("assert {1}.isdisjoint([2])\n") == "UNKNOWN"


# ---------------------------------------------------------------------------
# operators, comparisons, membership, truth
# ---------------------------------------------------------------------------


def test_the_four_operators_bound_the_length_and_a_frozenset_left_operand_gives_a_frozenset():
    assert verdict("a = {1, 2}\nb = {2, 3}\nassert len(a | b) >= 2\nassert len(a | b) <= 4\n") == "TRUE"
    assert verdict("a = {1, 2}\nb = {2, 3}\nassert len(a & b) <= 2\n") == "TRUE"
    assert verdict("a = {1, 2}\nb = {2, 3}\nassert len(a - b) <= 2\n") == "TRUE"
    assert verdict("a = {1, 2}\nb = {2, 3}\nassert len(a ^ b) <= 4\n") == "TRUE"
    assert verdict("a = {1, 2, 3}\nb = {1}\nassert len(a - b) >= 2\n") == "TRUE"
    assert verdict("a = {1, 2, 3}\nb = {1}\nassert len(a ^ b) >= 2\n") == "TRUE"
    assert verdict("a = frozenset([1])\nb = {2}\nc = a | b\nc.add(3)\n", "AttributeError") == "UNKNOWN"
    assert verdict("a = {1}\nb = frozenset([2])\nc = a | b\nc.add(3)\n", "AttributeError") == "TRUE"
    assert verdict("a = {1}\nb = {2}\nc = a | b | {3} | {4}\nassert len(c) <= 4\n") == "TRUE"


def test_the_in_place_operators_change_the_object_and_a_frozenset_is_rebound():
    assert verdict("s = {1}\nt = s\ns |= {2}\nassert len(t) <= 2\n") == "TRUE"
    assert verdict("s = {1}\nt = s\ns |= {2}\nassert len(t) >= 1\n") == "TRUE"
    assert verdict("s = {1, 2}\nt = s\ns -= {1, 2}\nassert len(t) <= 2\n") == "TRUE"
    assert verdict("s = {1, 2}\nt = s\ns &= {1}\nassert len(t) <= 1\n") == "TRUE"
    assert verdict("s = {1, 2}\nt = s\ns ^= {5}\nassert len(t) >= 1\nassert len(t) <= 3\n") == "TRUE"
    assert verdict("f = frozenset([1])\ng = f\nf |= {2}\nassert len(g) == 1\n") == "TRUE"       # g is the old object
    assert verdict("f = frozenset([1])\nf |= {2}\nf.add(3)\n", "AttributeError") == "UNKNOWN"


def test_an_operator_on_a_non_set_is_a_typeerror_check_and_ends_the_path():
    assert verdict("s = {1}\nt = s | [1]\n", "TypeError") == "UNKNOWN"
    assert verdict("s = {1}\nt = s + {2}\n", "TypeError") == "UNKNOWN"
    assert verdict("s = {1}\nt = 5 | s\n", "TypeError") == "UNKNOWN"
    assert verdict("s = {1}\ns |= [1]\n", "TypeError") == "UNKNOWN"
    assert verdict("s = {1}\nt = s | [1]\nassert False\n") == "TRUE"       # the path past it does not exist
    assert verdict("s = {1}\nt = s | {2}\n", "TypeError") == "TRUE"


def test_comparisons_are_subset_tests_decided_at_the_ends():
    assert verdict("assert set() <= {1}\nassert set() < {1}\nassert not (set() < set())\nassert set() <= set()\n") == "TRUE"
    assert verdict("assert not ({1, 2} <= {1})\nassert not ({1} < {1})\n") == "TRUE"
    assert verdict("assert {1, 2} >= {1}\n") == "UNKNOWN"
    assert verdict("assert {1} >= set()\nassert {1} > set()\n") == "TRUE"
    assert verdict("a = {1}\nassert a <= a\n") == "UNKNOWN"
    assert verdict("t = {1} < [1]\n", "TypeError") == "UNKNOWN"
    assert verdict("t = {1} <= {2}\n", "TypeError") == "TRUE"
    assert verdict("assert {1} == {1, 2} or True\n") == "TRUE"


def test_equality_is_decided_by_length_and_never_an_error():
    assert verdict("assert set() == set()\nassert not ({1} == {1, 2})\nassert {1} != {1, 2}\n") == "TRUE"
    assert verdict("assert {1} == {1}\n") == "UNKNOWN"
    assert verdict("assert {1} != 5\n") == "UNKNOWN"
    assert verdict("t = {1} == 5\nu = {1} != [1]\n", "TypeError") == "TRUE"


def test_membership():
    assert verdict("assert 1 not in set()\nassert 1 not in frozenset()\n") == "TRUE"
    assert verdict("assert 1 in {1}\nassert True in {1}\n") == "TRUE"
    assert verdict("assert 2 in {1}\n") == "UNKNOWN"
    assert verdict("assert 2 not in {1}\n") == "UNKNOWN"                    # the summary says what is in it, not what is not
    assert verdict("s = {1}\ns.add(2)\nassert 1 in s\n") == "UNKNOWN"
    assert verdict("assert 1.0 in {1}\n") == "UNKNOWN"                      # a float is never a certain member (nan)
    assert verdict("t = [1] in {1}\n", "TypeError") == "UNKNOWN"
    assert verdict("t = {1} in {frozenset([1])}\n", "TypeError") == "TRUE"   # a set key is converted
    assert verdict("t = (1, 2) in {(1, 2)}\n", "TypeError") == "TRUE"
    assert verdict("t = (1, [2]) in {(1, 2)}\n", "TypeError") == "UNKNOWN"


def test_the_truth_of_a_set_is_its_length():
    assert verdict("assert not set()\nassert {1}\nassert not frozenset()\n") == "TRUE"
    assert verdict("s = set()\nif s:\n    assert False\n") == "TRUE"
    assert verdict("s = {1}\nif not s:\n    assert False\n") == "TRUE"
    assert verdict("s = set()\ns.add(1)\nassert s\n") == "TRUE"
    assert verdict("s = {1}\ns.discard(1)\nassert s\n") == "UNKNOWN"
    assert verdict("s = {1}\ns.discard(1)\nassert not s\n") == "UNKNOWN"
    assert verdict("s = {nd()}\ns.discard(1)\nassert len(s) <= 1\n") == "TRUE"


def test_len_is_comparable_in_the_zone():
    assert verdict("s = {1, 2}\nn = len(s)\nassert n == 2\n") == "TRUE"
    assert verdict("s = {1, 2}\nt = {1, 2}\nassert len(s) == len(t)\n") == "TRUE"
    assert verdict("s = {1, 2}\nt = {1}\nassert len(s) > len(t)\n") == "TRUE"
    assert verdict("s = {1, 2}\nc = 1 // (len(s) - 2)\n", "ZeroDivisionError") == "UNKNOWN"
    assert verdict("s = {1, 2}\nc = 1 // (len(s) - 3)\n", "ZeroDivisionError") == "TRUE"


# ---------------------------------------------------------------------------
# iteration and the builtins over a set
# ---------------------------------------------------------------------------


def test_iteration_unpacking_and_the_reducers_read_the_element_summary():
    assert verdict("s = {4, 5}\nt = 0\nfor x in s:\n    t += x\nassert t >= 0\n") == "TRUE"
    assert verdict("s = {4, 5}\nx, y = s\nassert x >= 4\nassert y <= 5\n") == "TRUE"
    assert verdict("s = {4, 5}\nassert min(s) >= 4\nassert max(s) <= 5\n") == "TRUE"
    assert verdict("s = {4, 5}\nassert sum(s) >= 4\nassert sum(s) <= 10\n") == "TRUE"
    assert verdict("s = {4, 5}\nassert any(s)\nassert all(s)\n") == "TRUE"
    assert verdict("s = {4, 5}\nassert len(sorted(s)) <= 2\nassert sorted(s)[0] >= 4\n") == "TRUE"
    assert verdict("s = {4, 5}\nassert len(list(s)) <= 2\nassert len(tuple(s)) <= 2\n") == "TRUE"
    assert verdict("s = {4, 5}\nxs = [1]\nxs.extend(s)\nassert len(xs) >= 1\n") == "TRUE"
    assert verdict("s = {4, 5}\nxs = [1]\nxs.extend(s)\nassert xs[1] >= 4\n", "AssertionError") == "UNKNOWN"
    assert verdict("s = {2, 5}\nxs = [1]\nxs.extend(s)\nc = 10 // xs[0]\n", "ZeroDivisionError") == "TRUE"
    assert verdict("s = {0, 5}\nxs = [1]\nxs.extend(s)\nc = 10 // xs[0]\n", "ZeroDivisionError") == "UNKNOWN"   # the elements join
    assert verdict("s = {1, 'a'}\nm = max(s)\n", "TypeError") == "UNKNOWN"
    assert verdict("s = {1, 2.5}\nm = sorted(s)\n", "TypeError") == "TRUE"


def test_an_empty_set_has_nothing_to_iterate_or_reduce():
    assert verdict("s = set()\nfor x in s:\n    assert False\n") == "TRUE"
    assert verdict("s = set()\nm = max(s, default=3)\nassert m == 3\n") == "TRUE"
    assert verdict("s = set()\nassert sum(s) == 0\nassert not any(s)\nassert all(s)\n") == "TRUE"
    assert verdict("s = set()\nx, y = s\nassert False\n") == "TRUE"             # a ValueError ends the path
    assert verdict("s = {1}\nx, y = s\nassert False\n") == "UNKNOWN"            # the length is not used to end it


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------


def test_an_unhashable_element_is_a_typeerror_check_and_ends_the_path():
    for body in ("s = set()\ns.add([1])\n", "s = {[1]}\n", "s = {{1}}\n", "s = {(1, [2])}\n", "s = set([[1]])\n",
                 "s = frozenset([{1}])\n", "s = {1}\ns.update([[1]])\n", "s = {1}\ns.add({1: 2})\n",
                 "s = {1}\ns.discard([1])\n", "s = {1}\ns.remove([1])\n", "s = {1}\nt = s.union([[1]])\n"):
        assert verdict(body, "TypeError") == "UNKNOWN", body
        assert verdict(body + "assert False\n") == "TRUE", body                # nothing continues past it
    for body in ("s = {1, 'a', None, 2.5, True, (1, 2), frozenset([1])}\n", "s = set()\ns.add((1, (2, 3)))\n",
                 "s = {1}\ns.update('ab', range(2), (3,), {4: 5})\n"):
        assert verdict(body, "TypeError") == "TRUE", body


def test_a_dict_values_view_hashes_but_a_keys_or_items_view_does_not():
    """``dict_values`` has no ``__eq__`` and hashes by identity; ``dict_keys`` and ``dict_items`` are set-like and are unhashable.  The
    kind matrix found ``{1, 2}.add({1: 2}.values())`` treated as a certain ``TypeError``: the path past it was dropped."""
    for call in ("s.add(v)", "s.discard(v)", "s.remove(v)"):
        body = f"s = {{1, 2}}\nv = {{1: 2}}.values()\n{call}\nc = 1 // 0\n"
        if call == "s.remove(v)":
            continue                       # a KeyError natively: not the same shape
        ns: dict = {}
        with pytest.raises(ZeroDivisionError):
            exec(textwrap.dedent(body), ns)
        assert verdict(body, "ZeroDivisionError") != "TRUE", call
    assert verdict("s = {1, 2}\nv = {1: 2}.values()\ns.add(v)\nassert len(s) >= 1\n") == "TRUE"
    for view in ("keys", "items"):
        body = f"s = {{1, 2}}\ns.add({{1: 2}}.{view}())\nc = 1 // 0\n"
        ns = {}
        with pytest.raises(TypeError):
            exec(textwrap.dedent(body), ns)
        assert verdict(body, "TypeError") == "UNKNOWN"


def test_an_instance_hashes_unless_its_class_defines_eq():
    plain = "class A:\n    pass\ns = {A(), A()}\nassert len(s) >= 1\n"
    assert verdict(plain, "TypeError") == "TRUE"
    assert verdict(plain) == "TRUE"
    eq = ("class B:\n    def __eq__(self, other):\n        return True\n"
          "s = set()\ns.add(B())\n")
    assert verdict(eq, "TypeError") == "UNKNOWN"
    assert verdict(eq + "assert False\n") == "TRUE"
    inherited = ("class B:\n    def __eq__(self, other):\n        return True\nclass C(B):\n    pass\n"
                 "s = {C()}\n")
    assert verdict(inherited, "TypeError") == "UNKNOWN"
    # a class that defines __hash__ is refused when it is defined
    hashed = ("class D:\n    def __hash__(self):\n        return 1\ns = {D()}\n")
    assert result(hashed, ["TypeError"]).verdict == "UNKNOWN"


# ---------------------------------------------------------------------------
# a tuple hashes when its parts do: a part that only MAY hash keeps the path (tharkun #152 review, the wrong TRUE)
# ---------------------------------------------------------------------------

_MAYBE = "x = [] if nd() > 0 else 1\n"               # a list for some draws, an int for the others
_REACH = "reached = True\nz = 1 // 0\n"              # past the set: only a path on which the element hashed gets here
_GROWN = "l = []\nfor i in range(nd()):\n    l.append([])\n"        # a list of lists that is empty when the draw is <= 0

# the element hashes for some draws and not for others: ``ZeroDivisionError`` is reachable (the draws where it hashes) and
# so is ``TypeError``; neither may be proven away.  Every one of these was a wrong TRUE for ``ZeroDivisionError`` when a
# tuple with a part that MAY be unhashable was dropped as if it certainly were.
TUPLE_MAY_HASH = {
    "display": _MAYBE + "s = {(x,)}\n",
    "add": _MAYBE + "s = set()\ns.add((x,))\n",
    "discard": _MAYBE + "s = {1}\ns.discard((x,))\n",
    "remove": _MAYBE + "s = {(1,)}\ns.remove((x,))\n",
    "set_of_list": _MAYBE + "s = set([(x,)])\n",
    "set_of_tuple": _MAYBE + "s = set(((x,),))\n",
    "frozenset_of_list": _MAYBE + "s = frozenset([(x,)])\n",
    "update": _MAYBE + "s = set()\ns.update([(x,)])\n",
    "union": _MAYBE + "s = set().union([(x,)])\n",
    "nested": _MAYBE + "s = {((x,),)}\n",
    "nested_inside_a_pair": _MAYBE + "s = {(1, (x, 2))}\n",
    "two_parts": _MAYBE + "s = {(1, x)}\n",
    "part_may_be_a_dict": "x = {} if nd() > 0 else 1\ns = {(x,)}\n",
    "part_may_be_a_set": "x = set() if nd() > 0 else 1\ns = {(x,)}\n",
    "remove_converts_only_its_argument": "x = set() if nd() > 0 else 1\ns = {(1,)}\ns.remove((x,))\n",
    "part_may_be_an_eq_instance": "class A:\n    def __eq__(self, o):\n        return True\nx = A() if nd() > 0 else 1\ns = {(x,)}\n",
    "part_may_be_a_keys_view": "d = {1: 2}\nx = d.keys() if nd() > 0 else 1\ns = {(x,)}\n",
    "the_tuple_itself_may_be_one_of_two": "t = ([],) if nd() > 0 else (1,)\ns = {t}\n",
    "the_tuple_itself_added": "t = ([],) if nd() > 0 else (1,)\ns = set()\ns.add(t)\n",
    "built_earlier_then_listed": _MAYBE + "t = (x,)\nxs = [t]\ns = set(xs)\n",
    "listed_with_a_hashable_one": _MAYBE + "xs = [(x,), (1,)]\ns = set(xs)\n",
    "in_a_loop": "s = set()\nfor i in range(2):\n    x = [] if nd() > i else 1\n    s.add((x,))\n",
    "in_a_comprehension": _MAYBE + "s = {(x,) for _ in range(1)}\n",
    "through_a_function": "def f(a):\n    return {(a,)}\n" + _MAYBE + "s = f(x)\n",
    # a tuple whose length is a range: it is empty (and hashes) when the loop does not run
    "a_summary_tuple_that_may_be_empty": _GROWN + "s = {tuple(l)}\n",
    "a_summary_tuple_added": _GROWN + "t = tuple(l)\ns = set()\ns.add(t)\n",
    "a_list_times_a_draw": "t = tuple([[]] * nd())\ns = {t}\n",
    "a_summary_tuple_inside_a_pair": _GROWN + "s = {(1, tuple(l))}\n",
    "a_summary_tuple_of_parts_that_may_hash": "l = []\nfor i in range(nd()):\n    l.append([] if nd() > 0 else 1)\ns = {tuple(l)}\n",
}
# the element always hashes: ``TypeError`` is proven away, ``ZeroDivisionError`` is reached
TUPLE_HASHES = {
    "a_frozenset_part_hashes": "x = frozenset() if nd() > 0 else 1\ns = {(x,)}\n",
    "constants": "s = {(1,)}\n",
    "the_empty_tuple_of_an_empty_list": "l = []\nt = tuple(l)\ns = {t}\n",       # a summary with no element at all: nothing to fail to hash
    "the_empty_tuple_of_an_empty_generator": "t = tuple(x for x in [])\ns = {t}\n",
}
# the element never hashes: the path past it does not exist, so ``ZeroDivisionError`` is proven (and ``TypeError`` is raised)
TUPLE_NEVER_HASHES = {
    "a_list_part": "s = {([],)}\n",
    "a_list_part_beside_a_maybe": _MAYBE + "s = {([], x)}\n",
    "a_non_empty_summary_of_lists": "t = tuple([[]] * 2)\ns = {t}\n",
}


def _tuple_shape(body: str):
    src = HEAD + body + _REACH
    return src, native_raises(src)


@pytest.mark.parametrize("name", sorted(TUPLE_MAY_HASH))
def test_a_tuple_with_a_part_that_may_not_hash_keeps_the_path_on_which_it_does(name):
    src, raised = _tuple_shape(TUPLE_MAY_HASH[name])
    assert {"ZeroDivisionError", "TypeError"} <= raised, (name, raised)            # both really happen: the shape is not vacuous
    assert verdict(src, "ZeroDivisionError") == "UNKNOWN", name                    # the path past the set exists
    assert verdict(src, "TypeError") == "UNKNOWN", name                            # and so does the TypeError: neither is proven away
    wrong, _ = wrong_trues([src])
    assert not wrong, wrong[0]


@pytest.mark.parametrize("name", sorted(TUPLE_HASHES))
def test_a_tuple_whose_parts_always_hash_proves_the_typeerror_away(name):
    src, raised = _tuple_shape(TUPLE_HASHES[name])
    assert "ZeroDivisionError" in raised and "TypeError" not in raised, (name, raised)
    assert verdict(src, "ZeroDivisionError") == "UNKNOWN", name
    assert verdict(src, "TypeError") == "TRUE", name
    wrong, _ = wrong_trues([src])
    assert not wrong, wrong[0]


@pytest.mark.parametrize("name", sorted(TUPLE_NEVER_HASHES))
def test_a_tuple_with_a_part_that_never_hashes_ends_the_path(name):
    src, raised = _tuple_shape(TUPLE_NEVER_HASHES[name])
    assert "TypeError" in raised and "ZeroDivisionError" not in raised, (name, raised)
    assert verdict(src, "ZeroDivisionError") == "TRUE", name                       # nothing continues past the TypeError
    assert verdict(src, "TypeError") == "UNKNOWN", name
    wrong, _ = wrong_trues([src])
    assert not wrong, wrong[0]


def test_only_remove_discard_and_in_convert_a_set_to_a_frozenset():
    assert verdict("s = {frozenset([1])}\ns.discard({1})\n", "TypeError") == "TRUE"
    assert verdict("s = {frozenset([1])}\ns.remove({1})\n", "TypeError") == "TRUE"
    assert verdict("s = {frozenset([1])}\nt = {1} in s\n", "TypeError") == "TRUE"
    assert verdict("s = {frozenset([1])}\ns.add({1})\n", "TypeError") == "UNKNOWN"
    assert verdict("s = {frozenset([1])}\nt = s.union([{1}])\n", "TypeError") == "UNKNOWN"
    assert verdict("s = {frozenset([1])}\nt = s.difference([{1}])\n", "TypeError") == "UNKNOWN"
    assert verdict("s = set()\ns.remove({1})\n", "KeyError") == "UNKNOWN"


def test_the_arguments_of_a_method_are_checked_for_arity_and_keywords():
    for body in ("s = {1}\ns.add()\n", "s = {1}\ns.add(1, 2)\n", "s = {1}\ns.add(x=1)\n", "s = {1}\ns.pop(1)\n",
                 "s = {1}\ns.clear(1)\n", "s = {1}\ns.copy(1)\n", "s = {1}\ns.update(x=[1])\n",
                 "s = {1}\ns.issubset()\n", "s = {1}\ns.issubset([1], [2])\n", "s = {1}\nt = s.symmetric_difference()\n",
                 "s = set(1, 2)\n", "s = set(x=[1])\n", "s = {1}\ns.discard()\n", "s = {1}\ns.remove(1, 2)\n"):
        assert verdict(body, "TypeError") == "UNKNOWN", body
        assert verdict(body + "assert False\n") == "TRUE", body


def test_a_non_iterable_argument_is_a_typeerror_check():
    for body in ("s = set(5)\n", "s = frozenset(None)\n", "s = {1}\ns.update(5)\n", "s = {1}\nt = s.union(1.5)\n",
                 "s = {1}\ns.difference_update(None)\n", "s = {1}\nt = s.issubset(5)\n"):
        assert verdict(body, "TypeError") == "UNKNOWN", body
        assert verdict(body + "assert False\n") == "TRUE", body


def test_a_frozenset_has_no_mutating_methods_and_a_set_has_no_other_attributes():
    for method in ("add(1)", "remove(1)", "discard(1)", "pop()", "clear()", "update([1])", "difference_update([1])",
                   "intersection_update([1])", "symmetric_difference_update([1])"):
        body = f"f = frozenset([1])\nf.{method}\n"
        assert verdict(body, "AttributeError") == "UNKNOWN", body
        assert verdict(body + "assert False\n") == "TRUE", body                 # the path ends
    assert verdict("s = {1}\ns.foo()\n", "AttributeError") == "UNKNOWN"
    assert verdict("s = {1}\nx = s.foo\n", "AttributeError") == "UNKNOWN"
    assert verdict("s = {1}\ns.foo = 1\n", "AttributeError") == "UNKNOWN"
    assert verdict("f = frozenset([1])\nt = f.union([2])\nu = f.copy()\n", "AttributeError") == "TRUE"
    assert verdict("s = {1}\nm = s.add\nm(2)\nassert len(s) <= 2\n") == "TRUE"
    assert verdict("f = frozenset([1])\nm = f.add\n", "AttributeError") == "UNKNOWN"


def test_a_frozenset_is_its_own_copy_and_its_own_constructor_result():
    # the copy of a frozenset is the same object: a later mutation of either name would be visible through both
    # (there is none to make, which is the point), and an `is` against it must not be decided False
    assert verdict("f = frozenset([1])\ng = f.copy()\nassert f is not g\n") != "TRUE"
    assert verdict("f = frozenset([1])\ng = frozenset(f)\nassert f is not g\n") != "TRUE"
    assert verdict("s = {1}\nt = s.copy()\nt.add(2)\nassert len(s) == 1\n") == "TRUE"


def test_a_frozenset_is_hashable_so_a_set_of_frozensets_holds_it():
    body = """
        f = frozenset([1, 2])
        s = {f}
        s.add(frozenset())
        assert len(s) >= 1
        assert len(s) <= 2
    """
    assert verdict(body) == "TRUE"
    assert verdict(body, "TypeError") == "TRUE"
    assert verdict("s = {1}\nt = {s}\n", "TypeError") == "UNKNOWN"


# ---------------------------------------------------------------------------
# allocation: sites, stale references, shapes
# ---------------------------------------------------------------------------


def test_a_set_allocated_in_a_loop_is_a_recent_object_and_a_summary():
    body = """
        for i in range(3):
            s = set()
            s.add(i)
            assert len(s) == 1
    """
    assert verdict(body) == "TRUE"
    body = """
        t = set()
        for i in range(3):
            s = {i}
            t |= s
        assert len(t) >= 0
    """
    assert verdict(body) == "TRUE"


def test_a_set_grown_in_a_loop_widens():
    body = """
        s = set()
        for i in range(nd()):
            s.add(i)
        assert len(s) >= 0
    """
    assert verdict(body) == "TRUE"
    body = """
        s = set()
        for i in range(nd() % 5):
            s.add(7)
        for x in s:
            assert x == 7
    """
    assert verdict(body) == "TRUE"


def test_nested_frozensets_in_a_loop_never_name_the_object_being_built():
    """``f = frozenset([f])``: the argument's element is the previous object of this very site, which the
    allocation turns into the site's summary; the new set must hold THAT, not itself."""
    import ast

    from tharkun.solve.absint import interp

    src = "f = frozenset()\nfor i in range(3):\n    f = frozenset([f])\n"
    seen = []
    real_alloc = interp.Analyzer.alloc

    def alloc(self, st, node, make, tag):
        a = real_alloc(self, st, node, make, tag)
        ob = st.heap[a]
        if ob.kind == "set":
            seen.append(a in ob.elem.refs)
        return a

    interp.Analyzer.alloc = alloc
    try:
        r = prove(src, list(ALL), filename="<program>")
    finally:
        interp.Analyzer.alloc = real_alloc
    assert seen and not any(seen), (r.verdict, r.reason)


def test_a_chain_of_operators_and_methods_allocates_one_set_per_node():
    assert verdict("a = {1}\nb = {2}\nc = {3}\nd = a | b | c\nassert len(d) <= 3\n") == "TRUE"
    assert verdict("a = {1}\nd = a.union([2]).union([3]).difference([1])\nassert len(d) <= 3\n") == "TRUE"


# ---------------------------------------------------------------------------
# a set as the source of a slice assignment (the range/dict/array sources are [internal ref]'s, tests/test_absint_slice_source.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["{0}", "frozenset([0])", "{0, 0}"])
def test_a_slice_assignment_moves_the_elements_of_a_set_into_the_list(source):
    body = f"xs = [1]\nxs[0:1] = {source}\nc = 10 // xs[0]\n"
    ns: dict = {}
    with pytest.raises(ZeroDivisionError):
        exec(textwrap.dedent(body), ns)
    assert verdict(body, "ZeroDivisionError") == "UNKNOWN"
    ok = f"xs = [1]\nxs[0:1] = {source.replace('0', '2')}\nc = 10 // xs[0]\n"
    assert verdict(ok, "ZeroDivisionError") == "TRUE"


# ---------------------------------------------------------------------------
# named refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body, why", [
    ("d = {1: 2}\ns = set(d.items())\n", "dict.items"),
    ("s = {1}\nt = s | {1: 2}.keys()\n", "view"),
    ("xs = [1]\nxs[0:1] = {1: 2}.items()\n", "dict.items"),
    ("import random\ns = {1}\nx = random.choice(s)\n", "random.choice"),
    ("s = {*[1, 2]}\n", "starred"),
    ("s = {1}\nx = isinstance(s, set)\n", "isinstance"),
    ("def f():\n    pass\ns = {f.__call__}\n", "__call__"),
    ("class A:\n    def m(self):\n        pass\ns = {A().m}\n", "bmeth"),
    ("it = iter([1])\ns = {it}\n", "iterator"),
    ("s = {1}\nx = s.__len__()\n", "reflective"),
    ("s = {1}\nx = s.__class__\n", "set"),
    ("s = {1}\nf = frozenset([1])\nt = s if nd() > 0 else f\nu = t | {2}\n", "left operand"),
    ("s = {1}\nf = frozenset([1])\nt = s if nd() > 0 else f\nu = t.union([2])\n", "allocation site"),
])
def test_what_is_not_modelled_is_refused_not_guessed(body, why):
    r = result(body, ["TypeError"])
    assert r.verdict == "UNKNOWN", (body, r.reason)
    assert r.reason.startswith("unsupported:"), (body, r.reason)
    assert why.lower().split()[0] in r.reason.lower() or why in r.reason, (body, why, r.reason)


def test_a_set_and_a_frozenset_at_one_site_do_not_join():
    from tharkun.solve.absint.heap import SetObj
    from tharkun.solve.absint.aval import BOT
    from tharkun.solve.absint.state import Incompatible, ojoin
    from tharkun.solve.absint.zone import Zone

    with pytest.raises(Incompatible):
        ojoin(SetObj.make(0, 1, BOT), Zone(), SetObj.make(0, 1, BOT, True), Zone())


def test_the_copy_of_a_set_or_a_frozenset_is_a_new_set_or_the_frozenset_itself():
    body = "s = {1}\nf = frozenset([1])\nt = s if nd() > 0 else f\nu = t.copy()\nassert len(u) <= 1\n"
    assert verdict(body) == "TRUE"


# ---------------------------------------------------------------------------
# the differential
# ---------------------------------------------------------------------------

SETUPS = [
    "s = set()\nu = set()",
    "s = {1, 2}\nu = {2, 3}",
    "s = {1}\nu = {1}",
    "s = {nd()}\nu = {nd(), 2}",
    "s = set(range(nd() % 4))\nu = {0, 1, 2}",
    "s = {nd(), nd(), 5}\nu = set()",
    "s = set([1, 1, 2])\nu = frozenset([2])",
    "s = {1.0, 2}\nu = {True, 2}",
    "s = {x for x in [3, 4, 3]}\nu = {5}",
    "s = {0}\nu = s",
    "s = {1, 2}\nu = s",
    "ss = [{1}, {2, 3}]\ns = ss[nd() % 2]\nu = ss[1]",
    "s = {-1, 1}\nu = {1}",
    "s = {nd(), 1, 2}\nu = {1, 2}",
]
OPERATIONS = [
    "s.add(nd())", "s.add(2)", "s.add(1.0)", "s.add(True)", "s.add((1, 2))", "s.add([1])", "s.add(None)",
    "s.remove(1)", "s.remove(nd())", "s.remove(2.0)", "s.remove({1})", "s.remove([1])", "s.remove(0)",
    "s.discard(1)", "s.discard(nd())", "s.discard({1})", "s.discard(0)",
    "s.pop()", "x = s.pop()", "s.pop()\ns.pop()", "s.clear()", "s.clear()\ns.add(5)", "t = s.copy()\nt.add(9)",
    "u = s.copy()", "s.update(u)", "s.update([nd(), 4])", "s.update(range(3), u)", "s.update()", "s.update('ab')",
    "s.update(5)", "s |= u", "s &= u", "s -= u", "s ^= u", "u = s | u", "u = s & u", "u = s - u", "u = s ^ u",
    "s = s | {nd()}", "s.difference_update(u)", "s.intersection_update(u)", "s.symmetric_difference_update(u)",
    "s = s.union(u, [7])", "s = s.intersection(u)", "s = s.difference(u)", "s = s.symmetric_difference(u)",
    "s.intersection_update([1, nd()])", "u.add(1)", "s = s + u", "s |= [1]", "s.foo()", "s[0]", "s.update(s)",
    "s.difference_update(s)", "s.symmetric_difference_update(s)", "s.intersection_update(s)", "s |= s", "s ^= s",
    "t = s\ns.add(nd())", "t = s\ns.clear()", "f = frozenset(s)\nf.add(1)", "s = set(u)", "s = frozenset(s)",
]
READS = [
    "c = 1 // len(s)", "c = 1 // (len(s) - 1)", "c = 1 // (len(s) - len(u))", "assert len(s) <= 2",
    "assert len(s) >= 1", "assert len(s) == len(u)", "assert 1 in s", "assert 1 not in s", "assert nd() in s",
    "c = 1 // int(1 in s)", "assert s <= u", "assert s == u", "assert s < u", "assert not s", "assert s",
    "c = 10 // s.pop()", "c = 10 // min(s)", "c = 10 // max(s)", "c = 10 // sum(s)",
    "for v in s:\n    c = 10 // (v - 1)", "x, y = s", "c = 10 // (len(sorted(s)) - 1)", "assert s.issubset(u)",
    "assert s.isdisjoint(u)", "assert s.issuperset(u)", "c = 1 // (len(s | u) - 2)", "c = 1 // (len(s & u) - 1)",
    "c = 1 // (len(s - u) - 1)", "c = 1 // (len(s ^ u) - 1)", "xs = [0, 1, 2]\nc = xs[len(s)]",
    "xs = [0, 1, 2]\nc = xs[len(s) - 1]", "assert len(s) != 0", "assert len(s) < 3", "assert u <= s",
    "assert u == s", "c = 1 // len(u)", "assert 2 in s", "assert 0 not in s",
]
DRAWS = [-3, -2, -1, 0, 1, 2, 3, 4]
CLASSES = ("ZeroDivisionError", "AssertionError", "IndexError", "KeyError", "TypeError", "AttributeError")


def native_raises(src: str) -> set[str]:
    out: set[str] = set()
    code = compile(src, "<program>", "exec")
    for draws in itertools.product(DRAWS, DRAWS):
        it = itertools.cycle(draws)
        mod = types.ModuleType("_sv_verifier")
        mod.nondet_int = lambda: next(it)
        saved = sys.modules.get("_sv_verifier")
        sys.modules["_sv_verifier"] = mod
        try:
            exec(code, {"__name__": "__main__"})
        except Exception as e:  # noqa: BLE001
            out |= {c.__name__ for c in type(e).__mro__}
        finally:
            if saved is None:
                sys.modules.pop("_sv_verifier", None)
            else:
                sys.modules["_sv_verifier"] = saved
    return out


def programs(shard: int, shards: int, per_shard: int):
    """A deterministic sample of the cross product (setups x operations x reads), one slice per shard."""
    every = [(a, b, c) for a in SETUPS for b in OPERATIONS for c in READS]
    random.Random(0).shuffle(every)
    chunk = every[shard::shards][:per_shard]
    return [HEAD + a + "\n" + b + "\n" + c + "\n" for a, b, c in chunk]


def wrong_trues(sources) -> tuple[list[str], int]:
    """Every (class, program) absint calls TRUE although a native run raises the class, and how many TRUE verdicts
    were checked (a differential that proves nothing proves nothing)."""
    wrong, proved = [], 0
    for src in sources:
        raised = native_raises(src)
        for cls in CLASSES:
            if prove(src, [cls], filename="<program>").verdict == "TRUE":
                proved += 1
                if cls in raised:
                    wrong.append(f"{cls}:\n{src}")
    return wrong, proved


SHARDS = 8


@pytest.mark.parametrize("shard", range(SHARDS))
def test_no_set_program_gets_a_wrong_true(shard):
    wrong, proved = wrong_trues(programs(shard, SHARDS, 90))
    assert not wrong, "WRONG TRUE for " + wrong[0]
    assert proved > 100


def _break(monkeypatch, name, replacement):
    from tharkun.solve.absint import sets

    monkeypatch.setattr(sets.SetsMixin, name, replacement)


def test_the_differential_catches_a_discard_that_drops_the_length_lower_bound(monkeypatch):
    """Negative control 1: ``discard`` that keeps the old lower bound (it may have removed the element)."""
    from tharkun.solve.absint.heap import SetObj
    from tharkun.solve.absint.sets import SetsMixin, _NO_HOOK
    from tharkun.solve.absint.aval import NONE

    def broken(self, st, addrs, args, kwargs, node):
        if not self._no_args(st, args, kwargs, node, "discard", 1, 1):
            return NONE, {}
        x, ok = self._hashable(st, args[0], node, allow_set=True)
        self.update_obj(st, addrs, lambda ob, strong: SetObj.make(ob.length[0], ob.length[1], ob.elem, ob.frozen))
        return NONE, dict(_NO_HOOK, TypeError="x")

    _break(monkeypatch, "sm_discard", broken)
    src = HEAD + "s = {1}\ns.discard(1)\nassert len(s) >= 1\n"
    wrong, _ = wrong_trues([src])
    assert wrong and wrong[0].startswith("AssertionError")


def test_the_differential_catches_a_pop_that_proves_keyerror_away(monkeypatch):
    """Negative control 2: ``pop`` that claims ``KeyError`` is impossible without a lower bound."""
    from tharkun.solve.absint.sets import SetsMixin

    real = SetsMixin.sm_pop

    def broken(self, st, addrs, args, kwargs, node):
        out, proven = real(self, st, addrs, args, kwargs, node)
        proven["KeyError"] = "BROKEN: claimed without a lower bound"
        return out, proven

    _break(monkeypatch, "sm_pop", broken)
    wrong, _ = wrong_trues([HEAD + "s = set()\nx = s.pop()\n"])
    assert wrong and wrong[0].startswith("KeyError")


def test_the_differential_catches_an_add_that_forgets_the_element(monkeypatch):
    """Negative control 3: ``add`` that leaves the element summary as it was."""
    from tharkun.solve.absint.heap import SetObj
    from tharkun.solve.absint.sets import SetsMixin, _NO_HOOK
    from tharkun.solve.absint.aval import NONE

    def broken(self, st, addrs, args, kwargs, node):
        if not self._no_args(st, args, kwargs, node, "add", 1, 1):
            return NONE, {}
        self.update_obj(st, addrs, lambda ob, strong: SetObj.make(max(ob.length[0], 1), None if ob.length[1] is None
                                                                  else ob.length[1] + 1, ob.elem, ob.frozen))
        return NONE, dict(_NO_HOOK, TypeError="x")

    _break(monkeypatch, "sm_add", broken)
    wrong, _ = wrong_trues([HEAD + "s = {1}\ns.add(0)\nfor x in s:\n    c = 10 // x\n"])
    assert wrong and wrong[0].startswith("ZeroDivisionError")


def test_the_differential_catches_a_remove_that_proves_a_present_element_it_cannot_know(monkeypatch):
    """Negative control 4: a membership test that calls any non-empty set a certain member."""
    from tharkun.solve.absint.sets import SetsMixin

    _break(monkeypatch, "_certain_member", lambda self, st, ob, x: ob.length[0] >= 1)
    wrong, _ = wrong_trues([HEAD + "s = {1}\ns.remove(2)\n"])
    assert wrong and wrong[0].startswith("KeyError")


_ONE_TUPLE_PER_ALTERNATIVE = {"listed_with_a_hashable_one", "the_tuple_itself_added", "the_tuple_itself_may_be_one_of_two"}


def _break_ref_hashable(monkeypatch, reading):
    """Replace what ``_ref_hashable`` answers for one object: ``reading(ob, answer) -> answer`` (the model is real otherwise)."""
    from tharkun.solve.absint.sets import SetsMixin

    real = SetsMixin._ref_hashable

    def broken(self, st, a, node, allow_set, depth):
        return reading(st.heap[a], real(self, st, a, node, allow_set, depth))

    monkeypatch.setattr(SetsMixin, "_ref_hashable", broken)


def _wrong_zero_division(body: str) -> bool:
    wrong, _ = wrong_trues([HEAD + body + _REACH])
    return any(w.startswith("ZeroDivisionError") for w in wrong)


def test_the_differential_catches_a_tuple_that_may_not_hash_read_as_one_that_never_does(monkeypatch):
    """Negative control 5 (the #152 review blocker): ``None`` (hashes for some values of its parts) collapsed to ``False``
    (certainly unhashable) drops the tuple and ends the path past ``{(x,)}``.  Every shape of ``TUPLE_MAY_HASH`` catches it
    except three whose value is two tuples (one per alternative), each decided by itself: sound with and without the fix."""
    _break_ref_hashable(monkeypatch, lambda ob, r: False if r is None else r)
    missed = [n for n, body in sorted(TUPLE_MAY_HASH.items()) if not _wrong_zero_division(body)]
    assert missed == sorted(_ONE_TUPLE_PER_ALTERNATIVE), missed


def test_the_differential_catches_a_summary_tuple_that_may_be_empty_read_as_one_that_never_hashes(monkeypatch):
    """Negative control 6: a tuple known only by a length range and an element summary hashes when it is empty, so a summary
    of lists with the lower bound 0 is a tuple that MAY hash (``tuple(l)`` of a list grown in a loop that may not run)."""
    _break_ref_hashable(monkeypatch, lambda ob, r: False if r is None and ob.kind == "tuple" and ob.items is None else r)
    for name in ("a_summary_tuple_that_may_be_empty", "a_summary_tuple_added", "a_list_times_a_draw"):
        assert _wrong_zero_division(TUPLE_MAY_HASH[name]), name


def test_the_differential_catches_a_tuple_with_no_element_read_as_one_that_never_hashes(monkeypatch):
    """Negative control 7: a summary tuple whose element summary is bottom has no element (it is the empty tuple), so nothing
    in it fails to hash; ``is_bot()`` of the kept part must not be read as 'every part was dropped'."""
    _break_ref_hashable(monkeypatch, lambda ob, r: False if ob.kind == "tuple" and ob.items is None and ob.elem.is_bot() else r)
    for name in ("the_empty_tuple_of_an_empty_list", "the_empty_tuple_of_an_empty_generator"):
        assert _wrong_zero_division(TUPLE_HASHES[name]), name
