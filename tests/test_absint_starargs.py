"""Narrow support for direct calls with fixed builtin ``*args`` sequences."""

from __future__ import annotations

import textwrap

from tharkun.solve.absint import prove


def _src(text: str) -> str:
    return textwrap.dedent(text).strip()


def _native(source: str):
    namespace = {}
    try:
        exec(source, namespace)
        return True, "", namespace
    except Exception as exc:
        return False, type(exc).__name__, namespace


def test_direct_call_expands_tuple_and_fixed_list_in_argument_order():
    source = _src("""
        order = 0
        def mark(value):
            global order
            order = order * 10 + value
            return value
        def total(a, b, c, d):
            return a + b + c + d
        result = total(mark(0), *(mark(1), mark(2)), mark(3))
        one = [4]
        result2 = total(*one, 1, 1, 1)
        assert result == 6 and result2 == 7
        assert order == 123
    """)
    ok, exc, _ = _native(source)
    assert ok, exc
    result = prove(source, ("AssertionError", "TypeError"), time_budget=10)
    assert result.verdict == "TRUE", result.explain()


def test_starred_list_expansion_snapshots_slots_and_borrows_contained_aliases():
    source = _src("""
        box = []
        args = [box]
        def clear_args():
            args.clear()
        def use(item, ignored):
            item.append(9)
        use(*args, clear_args())
        assert len(args) == 0
        assert len(box) == 1 and box[0] == 9
    """)
    ok, exc, _ = _native(source)
    assert ok, exc
    result = prove(source, ("AssertionError", "TypeError"), time_budget=10)
    assert result.verdict == "TRUE", result.explain()


def test_starred_list_expansion_snapshots_symbolic_length_before_later_mutation():
    source = _src("""
        items = [4, 5]
        args = [len(items)]
        def clear_items():
            items.clear()
            return 0
        def use(count, ignored):
            assert count == 2
        use(*args, clear_items())
        assert len(items) == 0
    """)
    ok, exc, _ = _native(source)
    assert ok, exc
    result = prove(source, ("AssertionError", "TypeError"), time_budget=10)
    assert result.verdict == "TRUE", result.explain()


def test_starred_arguments_keep_existing_duplicate_keyword_typeerror_routing():
    source = _src("""
        def only(value):
            return value
        caught = False
        try:
            only(*[1], value=2)
        except TypeError:
            caught = True
        assert caught
    """)
    ok, exc, _ = _native(source)
    assert ok, exc
    result = prove(source, ("TypeError", "AssertionError"), time_budget=10)
    assert result.verdict == "TRUE", result.explain()


def test_unknown_length_or_multiple_reference_starred_sequence_stays_unproven():
    source = _src("""
        import _sv_verifier
        values = [1] if _sv_verifier.nondet_bool() else []
        def accept(*items):
            return len(items)
        result = accept(*values)
        assert result >= 0
    """)
    for draw in ("True", "False"):
        native = source.replace("import _sv_verifier", "").replace(
            "_sv_verifier.nondet_bool()", draw
        )
        ok, exc, _ = _native(native)
        assert ok, exc
    # the JOINED analysis cannot fix the sequence's length: this pins that it refuses, never claims TRUE
    result = prove(source, ("AssertionError",), time_budget=10, split_draws=False)
    assert result.verdict != "TRUE", result.explain()


def test_starred_sequence_of_a_split_boolean_draw_has_a_fixed_length_per_outcome():
    # the retry (docs/ABSINT.md "Case splits on a boolean draw") analyses each outcome of the draw apart: the
    # sequence is [1] in one and [] in the other, both natively fine, so the proof is sound
    source = _src("""
        import _sv_verifier
        values = [1] if _sv_verifier.nondet_bool() else []
        def accept(*items):
            return len(items)
        result = accept(*values)
        assert result >= 0
    """)
    result = prove(source, ("AssertionError",), time_budget=10)
    assert result.verdict == "TRUE", result.explain()
    assert result.draw_splits == 1


def test_custom_iterable_starred_argument_is_not_executed_or_assumed():
    source = _src("""
        class Iterable:
            def __iter__(self):
                raise KeyError("iteration hook")
        def accept(*items):
            return len(items)
        caught = False
        try:
            accept(*Iterable())
        except KeyError:
            caught = True
        assert caught
    """)
    ok, exc, _ = _native(source)
    assert ok, exc
    result = prove(source, ("KeyError", "AssertionError"), time_budget=10)
    assert result.verdict != "TRUE", result.explain()


def test_method_starred_call_remains_refused_due_to_lookup_order():
    source = _src("""
        class Receiver:
            def take(self, item):
                return item
        receiver = Receiver()
        result = receiver.take(*[3])
        assert result == 3
    """)
    ok, exc, _ = _native(source)
    assert ok, exc
    result = prove(source, ("AssertionError",), time_budget=10)
    assert result.verdict != "TRUE", result.explain()


def test_noniterable_starred_argument_cannot_prove_typeerror_handler():
    source = _src("""
        def accept(*items):
            return len(items)
        caught = False
        try:
            accept(*1)
        except TypeError:
            caught = True
        assert caught
    """)
    ok, exc, _ = _native(source)
    assert ok, exc
    result = prove(source, ("TypeError", "AssertionError"), time_budget=10)
    assert result.verdict != "TRUE", result.explain()
