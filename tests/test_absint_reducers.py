"""The iterable-reducing builtins in absint: ``min``/``max`` over an iterable, ``sum``, ``any``, ``all`` and the
keywords of ``sorted`` (``solve/absint/reducers.py``, docs/ABSINT.md "The iterable-reducing builtins").

Three kinds of test:

* exactness: the facts a model claims are the facts the call has (a sum of three elements in ``[1, 3]`` is in
  ``[3, 9]``, ``min`` of a list never lies outside the list's element facts, ``all([])`` is True);
* refusals: what the family does not model stays ``UNKNOWN`` (a key function, an iterator, an element that is
  an object);
* a differential: every program is also run natively over every draw that matters, and a TRUE for a class the
  program raises fails the test.
"""

from __future__ import annotations

import itertools
import sys
import textwrap
import types

import pytest

from tharkun.solve.absint import prove

HEAD = "import _sv_verifier\ndef nd():\n    return _sv_verifier.nondet_int()\n"


def verdict(body: str, classes=("AssertionError",)) -> str:
    return prove(HEAD + textwrap.dedent(body), list(classes), filename="<program>").verdict


# ---------------------------------------------------------------------------
# exactness
# ---------------------------------------------------------------------------


def test_min_and_max_of_a_list_are_elements_of_it():
    assert verdict("xs = [1, 2, 3]\nassert min(xs) >= 1\nassert min(xs) <= 3\nassert max(xs) >= 1\nassert max(xs) <= 3\n") == "TRUE"
    assert verdict("xs = [1, 2, 3]\nassert min(xs) > 1\n") == "UNKNOWN"
    assert verdict("xs = [1, 2, 3]\nassert max(xs) < 3\n") == "UNKNOWN"


def test_min_and_max_over_a_range_and_a_tuple():
    assert verdict("t = (4, 5, 9)\nassert max(t) <= 9\nassert min(t) >= 4\n") == "TRUE"
    assert verdict("n = nd()\nif 0 < n < 10:\n    m = min(range(n))\n    assert m >= 0\n    assert m < n\n") == "TRUE"


def test_an_empty_iterable_is_a_value_error_which_ends_the_path_unless_there_is_a_default():
    # ValueError is not a property class: the path past min([]) does not exist
    assert verdict("m = min([])\nassert False\n") == "TRUE"
    assert verdict("m = max([], default=7)\nassert m == 7\n") == "TRUE"
    assert verdict("m = max([], default=7)\nassert m == 8\n") == "UNKNOWN"


def test_a_default_is_joined_only_when_the_iterable_may_be_empty():
    assert verdict("xs = [5]\nm = min(xs, default=99)\nassert m == 5\n") == "TRUE"
    assert verdict("xs = [5] * nd()\nm = min(xs, default=99)\nassert m == 5\n") == "UNKNOWN"
    assert verdict("xs = [5] * nd()\nm = min(xs, default=99)\nassert m >= 5\nassert m <= 99\n") == "TRUE"


def test_min_of_floats_keeps_the_float_facts():
    body = """
        xs = [_sv_verifier.nondet_float_range(1.0, 2.0), _sv_verifier.nondet_float_range(3.0, 4.0)]
        m = min(xs)
        assert m >= 1.0
        assert m <= 4.0
    """
    assert verdict(body) == "TRUE"
    assert verdict(body.replace("assert m >= 1.0", "assert m > 1.0")) == "UNKNOWN"


def test_sum_of_integers_is_bounded_by_the_length_and_the_element_bounds():
    assert verdict("xs = [1, 2, 3]\nassert sum(xs) >= 3\nassert sum(xs) <= 9\n") == "TRUE"
    assert verdict("xs = [1, 2, 3]\nassert sum(xs) > 3\n") == "UNKNOWN"
    assert verdict("xs = [1, 2, 3]\nassert sum(xs, 10) >= 13\nassert sum(xs, start=10) <= 19\n") == "TRUE"
    assert verdict("n = nd()\nif 0 <= n < 5:\n    s = sum(range(n))\n    assert s >= 0\n    assert s <= 40\n") == "TRUE"


def test_sum_of_an_unbounded_count_is_bounded_on_the_side_the_elements_allow():
    # a non-negative element only grows the sum
    assert verdict("xs = [1] * nd()\nassert sum(xs) >= 0\n") == "TRUE"
    assert verdict("xs = [1] * nd()\nassert sum(xs) <= 100\n") == "UNKNOWN"
    # an element of either sign bounds neither side
    assert verdict("xs = [-1, 1] * nd()\nassert sum(xs) >= 0\n") == "UNKNOWN"


def test_sum_with_a_float_is_a_float_of_unknown_value_never_a_claimed_interval():
    # CPython compensates float sums (3.12+): the plain left fold is not the same number, so none is claimed
    assert verdict("xs = [0.1] * 10\nassert sum(xs) == 1.0\n") == "UNKNOWN"
    assert verdict("xs = [1, 2]\nassert sum(xs, 0.5) == 3.5\n") == "UNKNOWN"


def test_any_and_all_decide_from_the_length_and_the_truth_of_the_elements():
    assert verdict("assert all([])\nassert not any([])\nassert any([1, 2])\nassert all([1, 2])\n") == "TRUE"
    assert verdict("assert any([1, 0])\n") == "UNKNOWN"          # the elements are summarised: [0, 1]
    assert verdict("assert all([1, 0])\n") == "UNKNOWN"
    assert verdict("assert any([0, 0])\n") == "UNKNOWN"
    assert verdict("xs = [1] * nd()\nassert all(xs)\n") == "TRUE"
    assert verdict("xs = [0] * nd()\nassert not any(xs)\n") == "TRUE"
    assert verdict("xs = [1] * nd()\nassert any(xs)\n") == "UNKNOWN"          # the list may be empty


def test_sorted_takes_reverse_and_a_none_key():
    assert verdict("ys = sorted([3, 1, 2], reverse=True)\nassert len(ys) == 3\nassert min(ys) >= 1\n") == "TRUE"
    assert verdict("ys = sorted([3, 1, 2], key=None)\nassert len(ys) == 3\n") == "TRUE"
    assert verdict("ys = sorted([3, 1, 2], reverse=nd() > 0)\nassert len(ys) == 3\n") == "TRUE"


def test_sorted_dict_items_keeps_the_two_slot_types_and_live_values_at_call_time():
    body = """
        d = {"a": 1}
        view = d.items()
        alias = view
        d["a"] = 2
        pairs = sorted(alias)
        assert len(pairs) == 1
        for key, value in pairs:
            assert key == "a"
            assert isinstance(key, str)
            assert value == 2
            assert isinstance(value, int)
    """
    assert verdict(body) == "TRUE"


def test_sorted_empty_dict_items_keeps_an_empty_element_summary():
    assert verdict("pairs = sorted({}.items())\nassert len(pairs) == 0\nfor k, v in pairs:\n    assert False\n") == "TRUE"


def test_sorted_dict_items_keeps_open_primitive_slot_summaries():
    body = """
        d = {}
        d[nd()] = 1
        pairs = sorted(d.items())
        for key, value in pairs:
            assert isinstance(key, int)
            assert value == 1
    """
    assert verdict(body) == "TRUE"


def test_sorted_dict_items_is_a_snapshot_and_does_not_claim_an_order():
    after = """
        d = {"a": 1}
        pairs = sorted(d.items())
        d["a"] = 2
        assert pairs[0][1] == 1
    """
    assert verdict(after) == "TRUE"

    wrong_order = 'pairs = sorted({"b": 1, "a": 2}.items())\nassert pairs[0][0] == "b"\n'
    assert "AssertionError" in native_raises(HEAD + wrong_order)
    result = prove(HEAD + wrong_order, ["AssertionError"], filename="<program>")
    assert result.verdict != "TRUE", result.explain()


def test_sorted_dict_items_keeps_keyword_gates_and_scalar_reverse():
    assert verdict('pairs = sorted({"a": 1}.items(), key=None)\nassert len(pairs) == 1\n') == "TRUE"
    assert verdict('pairs = sorted({"a": 1}.items(), reverse=True)\nassert len(pairs) == 1\n') == "TRUE"
    for body in [
        'sorted({"a": 1}.items(), key=lambda pair: pair[0])',
        'sorted({"a": 1}.items(), reverse=object())',
    ]:
        result = prove(HEAD + body, ["TypeError"], filename="<program>")
        assert result.verdict == "UNKNOWN" and result.reason.startswith("unsupported:"), result.explain()


def test_sorted_dict_items_native_numeric_key_controls():
    native = """
        d = {True: "first", 1: "second"}
        assert len(d) == 1
        assert list(d.items()) == [(True, "second")]
        nan = float("nan")
        n = {nan: 1, 0.0: 2}
        assert sorted(n.items())[0][0] is nan
    """
    assert not native_raises(HEAD + textwrap.dedent(native))
    assert verdict('d = {True: "first", 1: "second"}\npairs = sorted(d.items())\nassert len(pairs) == 1\n') == "TRUE"
    nan_wrong_order = 'nan = float("nan")\nn = {nan: 1, 0.0: 2}\npairs = sorted(n.items())\nassert pairs[0][0] == 0.0\n'
    assert "AssertionError" in native_raises(HEAD + nan_wrong_order)
    result = prove(HEAD + nan_wrong_order, ["AssertionError"], filename="<program>")
    assert result.verdict != "TRUE", result.explain()


def test_sorted_dict_items_native_refusal_controls_do_not_make_wrong_trues():
    native = """
        d = {"a": 1}
        view = d.items()
        alias = view
        d["a"] = 2
        assert sorted(alias) == [("a", 2)]
        assert sorted({}.items()) == []
        try:
            sorted({"a": 1, 1: 2}.items())
        except TypeError:
            pass
        else:
            assert False
        class Bomb:
            def __lt__(self, other):
                raise AssertionError("dict item values must not be compared for distinct keys")
        assert sorted({"a": Bomb(), "b": Bomb()}.items())[0][0] == "a"
        assert sorted({"a": [], "b": []}.items())[0][0] == "a"
        def bad_key(pair):
            raise ZeroDivisionError("custom sorted key")
        try:
            sorted({"a": 1}.items(), key=bad_key)
        except ZeroDivisionError:
            pass
        else:
            assert False
    """
    assert native_raises(HEAD + textwrap.dedent(native)) == set()

    for body, exc in [
        ('sorted({"a": 1, 1: 2}.items())', "TypeError"),
        ('sorted({"a": [], "b": []}.items())', "TypeError"),
        ("""
         class Bomb:
             def __lt__(self, other):
                 raise AssertionError("value comparison")
         sorted({"a": Bomb(), "b": Bomb()}.items())
         """, "AssertionError"),
        ('sorted({"a": 1}.items(), key=lambda pair: 1 // 0)', "ZeroDivisionError"),
    ]:
        result = prove(HEAD + textwrap.dedent(body), [exc], filename="<program>")
        assert result.verdict == "UNKNOWN" and result.reason.startswith("unsupported:"), result.explain()


@pytest.mark.parametrize("body", [
    "d = {'a': 1}\nfor k, v in d.items():\n    pass\n",
    "d = {'a': 1}\npairs = list(d.items())\n",
])
def test_general_dict_items_iteration_and_materialization_remain_refused(body):
    result = prove(HEAD + body, ["TypeError"], filename="<program>")
    assert result.verdict == "UNKNOWN" and result.reason.startswith("unsupported:")


def test_the_reducers_leave_the_list_unchanged():
    assert verdict("xs = [1, 2]\nm = min(xs)\ns = sum(xs)\nxs.append(3)\nassert len(xs) == 3\nassert xs[0] >= 1\nassert xs[0] <= 3\n") == "TRUE"


# ---------------------------------------------------------------------------
# refusals: what is not modelled stays UNKNOWN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body", [
    "m = min([3, 1], key=abs)\n",                           # a key function is a call per element
    "m = max([3, 1], key=lambda x: -x)\n",
    "s = sorted([3, 1], key=abs)\n",
    "m = min(iter([3, 1]))\n",                              # an iterator is consumed by the call
    "m = sum(iter([3, 1]))\n",
    "m = max([3, 1], foo=1)\n",                             # a keyword the model does not take
])
def test_unmodelled_shapes_are_unknown(body):
    result = prove(HEAD + body, ["AssertionError"], filename="<program>")
    assert result.verdict == "UNKNOWN", result.reason


def test_elements_that_are_objects_are_refused():
    src = textwrap.dedent("""
        class K:
            def __init__(self, v):
                self.v = v
            def __lt__(self, other):
                return self.v < other.v
        m = min([K(1), K(2)])
    """)
    assert prove(src, ["AssertionError"], filename="<program>").verdict == "UNKNOWN"
    src = textwrap.dedent("""
        class K:
            def __bool__(self):
                return True
        assert any([K()])
    """)
    assert prove(src, ["AssertionError"], filename="<program>").verdict == "UNKNOWN"


def test_a_type_error_is_not_proven_away():
    assert prove(HEAD + "m = min([1, 'a'])\n", ["TypeError"], filename="<program>").verdict == "UNKNOWN"
    assert prove(HEAD + "s = sum(['a', 'b'])\n", ["TypeError"], filename="<program>").verdict == "UNKNOWN"
    assert prove(HEAD + "s = sum([1, None])\n", ["TypeError"], filename="<program>").verdict == "UNKNOWN"
    assert prove(HEAD + "s = sum(5)\n", ["TypeError"], filename="<program>").verdict == "UNKNOWN"
    assert prove(HEAD + "m = max([None, None])\n", ["TypeError"], filename="<program>").verdict == "UNKNOWN"
    assert prove(HEAD + "m = max([None])\n", ["TypeError"], filename="<program>").verdict == "TRUE"   # one element is never compared
    assert prove(HEAD + "m = min([1, 2.5, True])\ns = sum([1, 2.5, True])\n", ["TypeError"], filename="<program>").verdict == "TRUE"


# ---------------------------------------------------------------------------
# the differential
# ---------------------------------------------------------------------------

SOURCES = [
    "xs = [1, 2, 3]", "xs = [0, 5]", "xs = [-3, 4]", "xs = [nd(), 2]", "xs = [nd()] * 3", "xs = [0] * nd()",
    "xs = []", "xs = list(range(nd() % 4))", "xs = (4, 5)", "xs = range(nd() % 5)", "xs = [2, 2.5]",
    "xs = [nd()]", "xs = [True, 0]", "xs = [2, 2]", "xs = [2] * 3", "xs = [-1, -1]",
]
REDUCERS = [
    "min(xs)", "max(xs)", "sum(xs)", "min(xs, default=7)", "max(xs, default=-2)", "sum(xs, 3)",
    "int(any(xs))", "int(all(xs))", "len(sorted(xs, reverse=True))",
]
READS = [
    "c = 1 // {e}", "assert {e} != 0", "assert {e} >= 1", "assert {e} <= 3", "c = [1, 2][{e}]",
    "assert {e} < 5", "assert {e} > 2", "assert {e} != 4", "c = 1 // ({e} - 2)",
]
DRAWS = [-3, -2, -1, 0, 1, 2, 3, 4]
CLASSES = ("ZeroDivisionError", "AssertionError", "IndexError", "TypeError")


def native_raises(src: str) -> set[str]:
    out: set[str] = set()
    code = compile(src, "<program>", "exec")
    for draws in itertools.product(DRAWS, DRAWS):
        it = iter(draws * 3)
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


@pytest.mark.parametrize("reducer", REDUCERS)
def test_a_reducer_never_makes_a_wrong_true(reducer):
    proved = 0
    for source in SOURCES:
        for read in READS:
            src = HEAD + source + "\n" + read.format(e=reducer) + "\n"
            raised = native_raises(src)
            for cls in CLASSES:
                if prove(src, [cls], filename="<program>").verdict == "TRUE":
                    proved += 1
                    assert cls not in raised, f"WRONG TRUE for {cls}:\n{src}"
    assert proved > 0


def test_sum_bounds_are_exactly_the_extremes_of_the_sums_the_length_allows():
    """``reducers._sum_bounds``: for every length in ``[lo, hi]`` and every element in ``[a, b]`` the sum is
    inside the bounds, and a finite bound is attained (``None`` is unbounded)."""
    from tharkun.solve.absint.reducers import _sum_bounds

    big = 10**6
    for lo in range(0, 4):
        for hi in [None, *range(lo, 5)]:
            for a in [None, -2, -1, 0, 1, 2]:
                for b in [None, -2, -1, 0, 1, 2]:
                    if a is not None and b is not None and a > b:
                        continue
                    lower, upper = _sum_bounds(lo, hi, a, b)
                    counts = range(lo, (hi if hi is not None else lo + 6) + 1)
                    lows = [n * (a if a is not None else -big) for n in counts]
                    highs = [n * (b if b is not None else big) for n in counts]
                    low, high = min(lows), max(highs)
                    if lower is None:
                        assert low <= -big or hi is None, (lo, hi, a, b)
                    else:
                        assert lower == low, (lo, hi, a, b, lower, low)
                    if upper is None:
                        assert high >= big or hi is None, (lo, hi, a, b)
                    else:
                        assert upper == high, (lo, hi, a, b, upper, high)
