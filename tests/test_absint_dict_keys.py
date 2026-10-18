"""[internal ref] (go): a dict key is a set of constants when the zone bounds it, not only when the value carries literal bounds.

``EMPTY, WHITE, BLACK = 0, 1, 2; SHOW = {EMPTY: '.', WHITE: 'o', BLACK: 'x'}`` was "dict literal with a non-constant
key": a name reads as a zone variable (``IntInfo.lin``), whose bounds ``key_consts`` did not consult.  It now asks
``Analyzer.ibounds``, so a name bound to a constant is one key and a loop variable over a small range is that many;
the bounds are sound, so the set contains every value the key may take.  A differential runs random key programs
natively and fails on a TRUE for a KeyError the run raises.
"""

from __future__ import annotations

import itertools
import random
import sys
import textwrap
import types

import pytest

from tharkun.solve.absint import prove
from tharkun.sv import verify

HEAD = "import _sv_verifier\ndef nd():\n    return _sv_verifier.nondet_int()\n"


def verdict(body: str, classes=("KeyError",)) -> str:
    return prove(HEAD + textwrap.dedent(body), list(classes), filename="<program>").verdict


def native_exception(body: str, *, ints=(), bools=()):
    fake = types.ModuleType("_sv_verifier")
    int_values, bool_values = iter(ints), iter(bools)
    fake.nondet_int = lambda: next(int_values)
    fake.nondet_bool = lambda: next(bool_values)
    saved = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = fake
    try:
        exec(compile(HEAD + textwrap.dedent(body), "<native-dict-store>", "exec"), {"__name__": "__main__"})
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__
    finally:
        if saved is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = saved
    return None


def native(body: str, *, ints=(), floats=(), bools=()):
    """Run one source sample with controlled native draws; return globals and error."""
    mod = types.ModuleType("_sv_verifier")
    int_values, float_values, bool_values = iter(ints), iter(floats), iter(bools)
    mod.nondet_int = lambda: next(int_values)
    mod.nondet_float = lambda: next(float_values)
    mod.nondet_bool = lambda: next(bool_values)
    saved = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = mod
    ns = {"__name__": "__main__"}
    try:
        exec(compile(HEAD + textwrap.dedent(body), "<native-dict-display>", "exec"), ns)
    except Exception as exc:  # noqa: BLE001
        return ns, exc
    finally:
        if saved is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = saved
    return ns, None


def test_names_bound_by_a_tuple_unpack_are_dict_keys():
    body = """
        EMPTY, WHITE, BLACK = 0, 1, 2
        SHOW = {EMPTY: '.', WHITE: 'o', BLACK: 'x'}
        assert SHOW[WHITE] == 'o'
        assert len(SHOW) == 3
    """
    assert verdict(body, ("KeyError", "AssertionError")) == "TRUE"


def test_a_named_constant_key_is_one_key_and_an_absent_one_is_a_key_error():
    assert verdict("A = 1\nd = {A: 5}\nx = d[A]\n") == "TRUE"
    assert verdict("A = 1\nB = 2\nd = {A: 5}\nx = d[B]\n") == "UNKNOWN"
    assert verdict("A = 1\nd = {A: 5, A: 6}\nassert d[1] == 6\n", ("KeyError", "AssertionError")) in ("TRUE", "UNKNOWN")


def test_a_loop_variable_over_a_small_range_is_that_many_keys():
    assert verdict("d = {0: 1, 1: 2, 2: 3}\nfor i in range(3):\n    x = d[i]\n") == "TRUE"
    assert verdict("d = {0: 1, 1: 2}\nfor i in range(3):\n    x = d[i]\n") == "UNKNOWN"      # 2 is absent


def test_an_unknown_or_wide_key_is_still_not_a_constant():
    assert verdict("n = nd()\nd = {0: 1}\nx = d[n]\n") == "UNKNOWN"
    assert verdict("d = {0: 1}\nfor i in range(100):\n    x = d[i]\n") == "UNKNOWN"
    assert verdict("n = nd()\nif 0 <= n < 3:\n    d = {0: 1, 1: 2, 2: 3}\n    x = d[n]\n") == "TRUE"


def test_a_store_through_a_named_key_updates_that_key():
    body = """
        K = 4
        d = {K: 1}
        d[K] = 7
        assert d[K] == 7
    """
    assert verdict(body, ("KeyError", "AssertionError")) == "TRUE"


def test_uncertain_subscript_store_joins_values_for_every_possible_collision():
    cases = [
        ("n = nd()\nd = {0: 1, 7: 2}\nd[n] = 3\nassert d[7] == 2", [7], []),
        ("n = nd()\nd = {0: 1, 7: 2}\nd[n] = 3\nassert d[0] == 1", [0], []),
        ("import _sv_verifier\nk = True if _sv_verifier.nondet_bool() else 0\nd = {1: 1, 7: 2}\nd[k] = 3\nassert d[1] == 1", [], [True]),
        ("import _sv_verifier\nk = 1.0 if _sv_verifier.nondet_bool() else 0.0\nd = {1: 1, 7: 2}\nd[k] = 3\nassert d[1] == 1", [], [True]),
        ("import _sv_verifier\nk = 'x' if _sv_verifier.nondet_bool() else 'z'\nd = {'x': 1, 'y': 2}\nd[k] = 3\nassert d['x'] == 1", [], [True]),
        ("n = nd()\nd = {0: 1, 7: 2}\nd[n] += 1\nassert d[7] == 2", [7], []),
        ("n = nd()\nm = nd()\nk = None if n > 0 else m\nd = {None: 1}\nd[k] = 2\nassert d[None] == 1", [1, 0], []),
    ]
    for body, ints, bools in cases:
        assert native_exception(body, ints=ints, bools=bools) == "AssertionError"
        assert verdict(body, ("AssertionError",)) != "TRUE"


@pytest.mark.parametrize("body", [
    "n = nd()\nd = {0: 1, 7: 2}\nd[n] = 3\nassert d[7] == 2",
    "n = nd()\nd = {0: 1, 7: 2}\nd[n] = 3\nassert d[0] == 1",
])
def test_public_verifier_does_not_return_true_for_uncertain_store(tmp_path, body):
    source = HEAD + textwrap.dedent(body) + "\n"
    assert prove(source, ("AssertionError",)).verdict != "TRUE"
    path = tmp_path / "uncertain-store.py"
    path.write_text(source)
    for opts in ({}, {"canary_seconds": 0}):
        result = verify(str(path), ("AssertionError",), python=sys.executable,
                        budget_seconds=10, jobs=1, witness_dir=tmp_path / "witness", **opts)
        assert result.verdict != "true", (result.component, result.reason)


def test_uncertain_numeric_store_keeps_disjoint_string_and_presence_facts():
    disjoint = "n = nd()\nd = {'x': 1}\nd[n] = 2\nassert d['x'] == 1"
    present = "n = nd()\nd = {0: 1, 7: 2}\nd[n] = 3\nx = d[7]"
    assert verdict(disjoint, ("AssertionError",)) == "TRUE"
    assert verdict(present, ("KeyError",)) == "TRUE"
    assert verdict("d = {7: 2}\nd[7] = 3\nassert d[7] == 3", ("AssertionError",)) == "TRUE"


def test_native_equal_primitive_keys_keep_first_position_and_last_value():
    _, exc = native("""
        d = {True: 'bool', 1: 'int', 1.0: 'float'}
        assert len(d) == 1
        assert d[True] == d[1] == d[1.0] == 'float'
        assert type(next(iter(d))) is bool
        rev = {1.0: 'float', 1: 'int', True: 'bool'}
        assert len(rev) == 1
        assert rev[1] == rev[True] == 'bool'
        assert type(next(iter(rev))) is float
    """)
    assert exc is None


def test_native_conditional_key_collision_and_variable_reassignment():
    for draw, expected in ((0, "old"), (1, "new")):
        ns, exc = native("""
            n = nd() % 2
            d = {True: 'old', n: 'new'}
            observed = d[True]
            saved = n
            n = 9
            assert d[saved] == 'new'
        """, ints=(draw,))
        assert exc is None
        assert ns["observed"] == expected


def test_equal_uncertain_display_keys_preserve_length_and_last_value():
    for draws, expected_size, expected_value in (((0, 0), 1, "second"), ((0, 1), 2, "first")):
        ns, exc = native("""
            first, second = nd() % 2, nd() % 2
            d = {first: 'first', second: 'second'}
            size = len(d)
            value = d[0]
        """, ints=draws)
        assert exc is None
        assert ns["size"] == expected_size
        assert ns["value"] == expected_value


def test_native_possible_only_key_and_single_entry_cardinality():
    for draw in (0, 1):
        ns, exc = native("""
            n = nd() % 2
            d = {n: 'value'}
            size = len(d)
        """, ints=(draw,))
        assert exc is None
        assert ns["size"] == 1
    _, exc = native("n = nd() % 2\nd = {n: 'value'}\nvalue = d[0]", ints=(1,))
    assert isinstance(exc, KeyError)


def test_displayed_primitive_keys_do_not_imply_identity_or_exact_cardinality():
    possible = "n = nd() % 2\nd = {n: 'value'}\nx = d[0]"
    cardinality = "n = nd() % 2\nd = {n: 1}\nassert len(d) == 1"
    assert verdict(possible, ("KeyError",)) != "TRUE"
    assert verdict(cardinality, ("AssertionError",)) != "TRUE"

    same = "k = _sv_verifier.nondet_float()\nd = {k: 1}\nx = d[k]"
    fresh = "k = _sv_verifier.nondet_float()\nd = {k: 1}\nx = d[k + 0.0]"
    assert verdict(same, ("TypeError",)) == "TRUE"
    assert verdict(same, ("KeyError",)) != "TRUE"
    assert verdict(fresh, ("TypeError",)) == "TRUE"
    assert verdict(fresh, ("KeyError",)) != "TRUE"
    ns, exc = native("k = _sv_verifier.nondet_float()\nd = {k: 1}\nx = d[k]", floats=(float("nan"),))
    assert exc is None and ns["x"] == 1
    _, exc = native("k = _sv_verifier.nondet_float()\nd = {k: 1}\nx = d[k + 0.0]", floats=(float("nan"),))
    assert isinstance(exc, KeyError)


def test_finite_integer_key_may_replace_a_true_display_value():
    colliding = "n = nd() % 2\nd = {True: 'old', n: 'new'}\nassert d[True] == 'old'"
    assert isinstance(native(colliding, ints=(1,))[1], AssertionError)
    assert verdict(colliding, ("AssertionError",)) != "TRUE"
    assert verdict("n = nd() % 2\nd = {True: 'old', n: 'new'}\nx = d[True]", ("KeyError",)) == "TRUE"


def test_float_and_unbounded_integer_keys_stay_open_and_may_overwrite():
    assert verdict("n = nd()\nd = {n: 'value'}", ("TypeError",)) == "TRUE"
    assert verdict("n = nd()\nd = {n: 'value'}\nx = d[0]", ("KeyError",)) != "TRUE"
    assert verdict("d = {1.0: 'float'}\nx = d[1.0]", ("KeyError",)) != "TRUE"
    overwritten = "d = {1: 'old', 1.0: 'new'}\nassert d[1] == 'old'"
    assert verdict(overwritten, ("AssertionError",)) != "TRUE"
    ns, exc = native("d = {1: 'old', 1.0: 'new'}\nassert len(d) == 1\nassert d[True] == 'new'")
    assert exc is None and ns["d"][True] == "new"


def test_custom_and_unhashable_display_keys_remain_refused():
    unhashable = "key = []\nd = {key: 1}"
    assert isinstance(native(unhashable)[1], TypeError)
    assert verdict(unhashable, ("TypeError",)) == "UNKNOWN"
    custom = "class Key:\n    def __hash__(self): return 7\nkey = Key()\nd = {key: 1}"
    assert native(custom)[1] is None
    assert verdict(custom, ("TypeError",)) == "UNKNOWN"

    raises = "class Key:\n    def __hash__(self): raise ZeroDivisionError()\nd = {Key(): 1}"
    assert isinstance(native(raises)[1], ZeroDivisionError)
    assert verdict(raises, ("ZeroDivisionError",)) == "UNKNOWN"
    caught = "class Key:\n    def __hash__(self): raise ZeroDivisionError()\ntry:\n    d = {Key(): 1}\nexcept ZeroDivisionError:\n    assert False"
    assert isinstance(native(caught)[1], AssertionError)
    assert verdict(caught, ("AssertionError",)) == "UNKNOWN"


def test_open_display_then_uncertain_store_keeps_no_wrong_true():
    source = "n = nd()\nm = nd()\nd = {n: 2, 0: 1}\nd[m] = 3\nassert d[0] == 1"
    assert verdict(source, ("AssertionError",)) != "TRUE"
    _, exc = native(source, ints=(5, 0))
    assert isinstance(exc, AssertionError)


def test_popitem_remains_unmodelled_for_an_open_display():
    source = "n = nd()\nd = {n: 1}\npair = d.popitem()"
    ns, exc = native(source, ints=(0,))
    assert exc is None and ns["pair"] == (0, 1)
    assert verdict(source, ("KeyError",)) == "UNKNOWN"


# ---------------------------------------------------------------------------
# the differential
# ---------------------------------------------------------------------------

NAMES = ["A = 0", "A = 1", "A = 2", "A = 5", "A = nd() % 3", "A = abs(nd()) % 4"]
DICTS = ["d = {0: 1, 1: 2}", "d = {A: 9}", "d = {0: 1, 1: 2, 2: 3}", "d = {A: 9, 3: 4}", "d = {}"]
STORES = ["", "d[A] = 5", "d[1] = 6", "d[i] = 7"]
READS = ["x = d[A]", "x = d[i]", "x = d[A + 1]", "x = d[2]", "x = d[B]", "x = d.get(A, 0) + d[0]"]
DRAWS = [-4, -2, -1, 0, 1, 2, 3, 5]


def native_raises(src: str) -> set[str]:
    out: set[str] = set()
    code = compile(src, "<program>", "exec")
    for a, b in itertools.product(DRAWS, DRAWS):
        it = iter([a, b, a, b])
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


@pytest.mark.parametrize("seed", range(8))
def test_a_dict_key_program_never_makes_a_wrong_true(seed):
    rng = random.Random(1121_500 + seed)
    proved = 0
    for _ in range(30):
        lines = [rng.choice(NAMES), "B = nd() % 4", rng.choice(DICTS)]
        store, read = rng.choice(STORES), rng.choice(READS)
        body = ["for i in range(%d):" % rng.randint(1, 4)] + ["    " + l for l in (store, read) if l]
        src = HEAD + "\n".join(lines + body) + "\n"
        raised = native_raises(src)
        for cls in ("KeyError", "IndexError", "TypeError"):
            if prove(src, [cls], filename="<program>", time_budget=20.0).verdict == "TRUE":
                proved += 1
                assert cls not in raised, f"WRONG TRUE for {cls}:\n{src}"
    assert proved > 0
