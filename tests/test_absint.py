"""tharkun.solve.absint: a sound whole-program abstract interpreter (docs/ABSINT.md).

Three kinds of test:

* the zone (DBM) domain's lattice operations;
* **targets**: SV-Benchmarks pairs absint must prove TRUE, each with a fact per check;
* **never-TRUE**: programs with a reachable property exception (mutants of the
  targets and small hand-written ones).  absint must never answer TRUE on them --
  a wrong TRUE is the one unforgivable answer.  Every soundness bug found gets a
  regression here.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.absint.zone import ZERO, Zone

FIX = Path(__file__).parent / "fixtures" / "absint"
DL = ("KeyError", "IndexError")
DT = ("TypeError", "AttributeError")
AR = ("ZeroDivisionError", "FloatingPointError")
AS = ("AssertionError",)


def _src(body: str) -> str:
    return textwrap.dedent(body)


# ------------------------------------------------------------------ the zone
def test_zone_closure_derives_transitive_bounds():
    z = Zone()
    z.add("x", "y", 0)  # x <= y
    z.add("y", ZERO, 5)  # y <= 5
    assert z.bounds("x") == (None, 5)
    z.add(ZERO, "x", -2)  # x >= 2
    assert z.bounds("y") == (2, 5)


def test_zone_detects_infeasibility():
    z = Zone()
    z.add("x", "y", -1)  # x < y
    z.add("y", "x", 0)  # y <= x
    assert z.bottom


def test_zone_join_keeps_common_bounds_and_widen_drops_unstable_ones():
    a, b = Zone(), Zone()
    a.assign_interval("x", 0, 0)
    b.assign_interval("x", 0, 1)
    j = a.join(b)
    assert j.bounds("x") == (0, 1)
    w = a.widen(j)
    assert w.bounds("x") == (0, None)


def test_zone_shift_is_exact():
    z = Zone()
    z.assign_interval("n", 1, None)
    z.assign_var("r", "n", 0)
    z.assign_var("r", "r", -1)  # r := r - 1
    assert z.entails("r", "n", -1) and z.entails("n", "r", 1)


# ------------------------------------------------------------------ targets
@pytest.mark.parametrize("task, classes", [
    ("bm_fannkuch_nondet.py", DL),
    ("bm_fannkuch_nondet.py", DT),
    ("bm_float_nondet.py", DL),
    ("bm_float_nondet.py", DT),  # slice 2: the fill loop writes every slot of [None] * n
    ("bm_float_nondet.py", AR),  # slice 2: sum-of-squares invariant, lemmas TRIG_PAIR + NORMALIZE
    ("bm_nbody_nondet.py", DL),  # slice 3: dict keys, random.choice over them, length-3 lists
    ("bm_nbody_nondet.py", DT),  # slice 3: tuple destructuring, sums of squares >= 0 (no complex **)
])
def test_real_world_targets_are_proven(task, classes):
    r = prove((FIX / "targets" / task).read_text(), classes)
    assert r.verdict == "TRUE", r.reason
    assert r.sites and all(fact and not fact.startswith("UNPROVEN") for *_, fact in r.sites)


def test_the_proof_names_the_invariant_at_each_index():
    r = prove((FIX / "targets" / "bm_fannkuch_nondet.py").read_text(), DL)
    facts = {(line, op): fact for line, _c, op, _e, fact in r.sites}
    assert "0 <= idx < len" in facts[(30, "index")]  # count[r - 1] = r


# ------------------------------------------------------------------ never TRUE
@pytest.mark.parametrize("mutant", sorted(p.name for p in (FIX / "never_true").glob("*.py")))
def test_mutants_with_a_reachable_index_error_are_never_true(mutant):
    r = prove((FIX / "never_true" / mutant).read_text(), DL)
    assert r.verdict == "UNKNOWN"


def test_float_arithmetic_proof_cites_its_lemmas():
    r = prove((FIX / "targets" / "bm_float_nondet.py").read_text(), AR)
    assert r.verdict == "TRUE" and {"TRIG_PAIR", "NORMALIZE"} <= set(r.lemmas)
    norm = [fact for line, _c, op, _e, fact in r.sites if line == 34 and op == "divide"]
    assert norm and "excludes 0" in norm[0] and "precondition checked" in norm[0]


@pytest.mark.parametrize("mutant", sorted(p.name for p in (FIX / "never_true_ar").glob("*.py")))
def test_float_mutants_with_a_possible_zero_norm_are_never_true(mutant):
    """cos(i)*0, sin/sin, a shifted argument, a reassigned i, a zeroed component..."""
    r = prove((FIX / "never_true_ar" / mutant).read_text(), AR)
    assert r.verdict == "UNKNOWN" and "unproven check" in r.reason


@pytest.mark.parametrize("mutant", sorted(p.name for p in (FIX / "never_true_dt").glob("*.py")))
def test_fill_loop_mutants_that_leave_a_none_are_never_true(mutant):
    """Segmentation: a fill loop that skips, stops early, starts at 1, overwrites with None...;
    nbody: a None body in the system, str + float."""
    r = prove((FIX / "never_true_dt" / mutant).read_text(), DT)
    assert r.verdict == "UNKNOWN" and "unproven check" in r.reason


NEVER_TRUE = {
    # the empty-range bug: materializing an element summary asserted non-emptiness
    "empty_range_list": (DL, """
        import _sv_verifier
        n = abs(_sv_verifier.nondet_int())
        xs = list(range(n))
        xs[0] = 1
    """),
    "negative_index_past_start": (DL, """
        import _sv_verifier
        xs = [1, 2, 3]
        k = _sv_verifier.nondet_int()
        if k < 3:
            print(xs[k])
    """),
    "pop_from_empty": (DL, """
        import _sv_verifier
        xs = []
        if _sv_verifier.nondet_bool():
            xs.append(1)
        xs.pop()
    """),
    "division_by_draw": (AR, """
        import _sv_verifier
        n = _sv_verifier.nondet_int()
        print(10 // n)
    """),
    "division_by_abs_draw": (AR, """
        import _sv_verifier
        n = abs(_sv_verifier.nondet_int())
        print(1 / n)
    """),
    "modulo_in_unused_expression": (AR, """
        import _sv_verifier
        n = _sv_verifier.nondet_int() % 3
        5 % n
    """),
    "none_attribute": (DT, """
        import _sv_verifier
        class P:
            def f(self):
                return 1
        p = P() if _sv_verifier.nondet_bool() else None
        p.f()
    """),
    "unset_attribute": (DT, """
        import _sv_verifier
        class P:
            def __init__(self, b):
                if b:
                    self.x = 1
        p = P(_sv_verifier.nondet_bool())
        print(p.x)
    """),
    "slots_violation": (DT, """
        class P:
            __slots__ = ('x',)
            def __init__(self):
                self.y = 1
        P()
    """),
    "wrong_arity": (DT, """
        import _sv_verifier
        def f(a, b):
            return a + b
        if _sv_verifier.nondet_bool():
            f(1)
    """),
    "raise_index_error": (DL, """
        import _sv_verifier
        if _sv_verifier.nondet_int() > 5:
            raise IndexError
    """),
    "raise_subclass_of_key_error": (DL, """
        import _sv_verifier
        class E(KeyError):
            pass
        if _sv_verifier.nondet_bool():
            raise E()
    """),
    "str_plus_int": (DT, """
        import _sv_verifier
        x = "a" if _sv_verifier.nondet_bool() else 1
        print(x + 1)
    """),
    "ordering_none": (DT, """
        import _sv_verifier
        x = None if _sv_verifier.nondet_bool() else 1
        print(x < 2)
    """),
    "loop_grows_past_bound": (DL, """
        import _sv_verifier
        xs = [0] * 10
        i = 0
        while _sv_verifier.nondet_bool():
            i += 1
        xs[i] = 1
    """),
    "slice_assignment_shrinks": (DL, """
        import _sv_verifier
        n = 1 + abs(_sv_verifier.nondet_int())
        xs = list(range(n))
        xs[0:n] = []
        print(xs[0])
    """),
    "assertion_on_draw": (AS, """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        assert x != 7
    """),
    "segment_pop_then_append": (DT, """
        import _sv_verifier
        class P:
            def f(self):
                return 1
        n = 1 + abs(_sv_verifier.nondet_int())
        xs = [None] * n
        for i in range(n):
            xs[i] = P()
        xs.pop()
        xs.append(None)
        for p in xs:
            p.f()
    """),
    "segment_insert_shifts": (DT, """
        import _sv_verifier
        class P:
            def f(self):
                return 1
        n = 1 + abs(_sv_verifier.nondet_int())
        xs = [None] * n
        for i in range(n):
            xs[i] = P()
        xs.insert(0, None)
        xs[0].f()
    """),
    "segment_loop_var_reassigned": (DT, """
        import _sv_verifier
        class P:
            def f(self):
                return 1
        n = 2 + abs(_sv_verifier.nondet_int())
        xs = [None] * n
        for i in range(n):
            i = 0
            xs[i] = P()
        xs[1].f()
    """),
    "power_zero_negative": (AR, """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        print(x ** -1)
    """),
    # [internal ref] (Astra, 2026-09-28; PR #53 revert #56): the loop-family relational
    # induction check accepted the false invariant `y <= x + 1` here. Root cause
    # predates #53: a module-scope global read (`load_name`) is a live reference
    # into the global's zone slot, not a snapshot; `cond_compare` evaluated the
    # right-hand call (`lower()`, which reassigns `y`) before using that saved
    # left operand, so the comparison silently saw the POST-call `y` instead of
    # the value Python actually saved -- and the branch it should have taken
    # (`y += 3`) was dropped during the loop's transfer function, making the
    # induction step check look inductive when it was not. Exact repro from
    # [local path] on the wrong-TRUE head,
    # six `True` draws then `False` reach `(x, y) = (6, 9)` and violate the
    # assertion. Fixed by `Analyzer.save_operand` (interp.py), which materializes
    # an operand before any operand to its right (that is not provably
    # side-effect-free) evaluates.
    "lf1049_relational_induction_saved_left_operand": (AS, """
        import _sv_verifier

        def lower():
            global y
            y -= 1
            return x

        x = y = 0
        while _sv_verifier.nondet_bool():
            if x >= 4 and y == x:
                y += 2
            else:
                y += 1
            x += 1
            if x >= 6 and y > lower():
                y += 3
        assert y <= x + 1
    """),
    # [internal ref] negative control: a minimal, loop-free case that isolates
    # `cond_compare`/`save_operand` from the loop-family induction machinery
    # entirely. Python evaluates and saves `y` (5) before calling `reset()`,
    # which reassigns the global `y` to 0 and returns 1; the saved 5 is what
    # gets compared, so `y > reset()` is True and `assert False` is reachable
    # on every execution. Before the fix, `cond_compare` re-read the (by then
    # reassigned) global instead of the saved value, concluded `5 > 1` was
    # unreachable, and dropped the branch.
    "lf1049_compare_call_mutates_left_operand": (AS, """
        import _sv_verifier

        def reset():
            global y
            y = 0
            return 1

        y = 5
        if _sv_verifier.nondet_bool() and y > reset():
            assert False
    """),
    # [internal ref] negative control, `e_BinOp` sibling of the above (same evaluate-left,
    # evaluate-right-which-may-call, then combine pattern; `save_operand` guards
    # both). Python saves `y` (5) before calling `bump()`, which reassigns the
    # global `y` to 100 and returns 1; the statement then rebinds `y` to the
    # saved 5 plus 1, i.e. 6 -- always, on every execution that takes this
    # branch -- so `assert y != 6` is a reachable violation. A left operand that
    # is not saved before the call runs would instead compute (post-call) 100 +
    # 1 = 101, concluding `y != 6` always holds and the assert is unreachable.
    "lf1049_binop_call_mutates_left_operand": (AS, """
        import _sv_verifier

        def bump():
            global y
            y = 100
            return 1

        y = 5
        if _sv_verifier.nondet_bool():
            y = y + bump()
            assert y != 6
    """),
    # PR45 soundness attack repros (MAIN, 2026-09-28): #47's list.extend contract
    # names only __iter__ as a source of raises -- extend()'s internal length-hint
    # probe of __len__ (CPython Objects/abstract.c::PyObject_LengthHint) is caught
    # and swallowed, not propagated.  The old model let that raise kill the whole
    # branch, silently pruning it as unreachable instead of continuing extend()
    # via __iter__ -- so the real ZeroDivisionError after it was never analyzed.
    "extend_hint_typeerror_from_len_is_swallowed": (AR, """
        class Box:
            def __init__(self):
                self.items = [1, 2]
            def __iter__(self):
                return iter(self.items)
            def __len__(self):
                raise TypeError("length unknown")
        items = [0]
        items.extend(Box())
        print(10 // (len(items) - 3))
    """),
    # `Cls.x = 5` replaces a __slots__ descriptor with a plain class attribute.
    # There is no instance __dict__ behind a slot, so CPython's attribute lookup
    # returns the (now non-descriptor) class value directly for every instance --
    # any value previously stored through the vanished descriptor is orphaned.
    "slot_overridden_by_class_attribute_read": (AR, """
        class Cell:
            __slots__ = ("x",)
            def __init__(self):
                self.x = 0
        c = Cell()
        Cell.x = 5
        print(10 // (c.x - 5))
    """),
    # Same override, on the write side: no descriptor left to receive the store
    # and no instance __dict__ to fall back to, so CPython raises AttributeError
    # ("... object attribute 'x' is read-only").
    "slot_overridden_by_class_attribute_write": (DT, """
        class Cell:
            __slots__ = ("x",)
            def __init__(self):
                self.x = 0
        c = Cell()
        Cell.x = 5
        c.x = 1
    """),
    # [internal ref] review (MG, PR65 head cf9a2f8): a subclass that never redeclares
    # __slots__ (the common case for an otherwise-empty subclass) still
    # inherits a real, live slot descriptor from its base -- and it can still
    # be shadowed by a class-level override, exactly like the single-class
    # case above. The bug: _all_slots used to bail to "this class has a
    # __dict__" (has_dict=True, meaning "no slots to reason about here") the
    # moment the FIRST class visited -- own or not -- lacked its own
    # __slots__, so the override-detection check never even looked past Sub
    # to see Base's slot. Confirmed wrong TRUE (mg1072 b01).
    "slot_overridden_via_subclass_without_own_slots": (AR, """
        class Base:
            __slots__ = ("x",)
            def __init__(self):
                self.x = 0
        class Sub(Base):
            pass
        s = Sub()
        Sub.x = 5
        print(10 // (s.x - 5))
    """),
    # Same bug, override landing on the BASE class instead of the subclass
    # (mg1072 b03) -- the override doesn't have to be on the same class that
    # `_all_slots` was bailing out on.
    "slot_overridden_on_base_via_subclass_instance": (AR, """
        class Base:
            __slots__ = ("x",)
            def __init__(self):
                self.x = 0
        class Sub(Base):
            pass
        s = Sub()
        Base.x = 5
        print(10 // (s.x - 5))
    """),
    # Same bug through a 3-level chain, and through two separate instances of
    # the same subclass sharing one override (mg1072 b02 folded with b07):
    # a bug that only manifests on ONE recency-tracked instance would still
    # be a wrong TRUE for the property as a whole.
    "slot_overridden_via_deep_subclass_chain": (AR, """
        class Base:
            __slots__ = ("x",)
            def __init__(self):
                self.x = 0
        class Mid(Base):
            pass
        class Sub(Mid):
            pass
        s1 = Sub()
        s2 = Sub()
        Sub.x = 9
        print(10 // (s1.x - 9) + 10 // (s2.x - 9))
    """),

    # --- [internal ref]: evaluation-order hazards through live module-global references ---
    # (soundness, REV review of [internal ref]). load_name() used to return a LIVE
    # reference to a written module global's zone slot at module scope
    # (interp.py::Analyzer.load_name); an operand read before a call that
    # rebinds that global observed the call's post-mutation value instead of
    # the value that was actually live at read time. sv_2027 @ cf5a8a1 (before
    # the fix) answered TRUE, wrongly, on 17 of these 25 -- REV's
    # [host]:[local path]
    "eo_binop_fixed": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        x = y + bump()
        assert x == 5
    """),
    "eo_compare_fixed": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        assert y == bump() + 5
    """),
    "eo_augassign": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        y += bump()
        assert y == 5
    """),
    "eo_call_args_user": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def first(a, b):
            return a
        x = first(y, bump())
        assert x == 5
    """),
    "eo_call_args_builtin": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        x = max(y, bump())
        assert x == 5
    """),
    "eo_call_args_linexpr": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def first(a, b):
            return a
        x = first(y + 1, bump())
        assert x == 6
    """),
    "eo_tuple_display": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        t = (y, bump())
        assert t[0] == 5
    """),
    "eo_list_display": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        l = [y, bump()]
        assert l[0] == 5
    """),
    "eo_dict_display": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        d = {0: y, 1: bump()}
        assert d[0] == 5
    """),
    "eo_tuple_unpack": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        a, b = y, bump()
        assert a == 5
    """),
    "eo_subscript_index": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        xs = [0, 1, 2, 3, 4, 5]
        v = xs[y] + xs[bump()]
        assert v == 5
    """),
    "eo_subscript_store": (DL, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        xs = [0, 1]
        xs[y] = bump()
    """),
    "eo_slice_bounds": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        xs = [0, 1, 2, 3, 4, 5, 6]
        s = xs[y:bump() + 7]
        assert len(s) == 2
    """),
    "eo_method_args": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        xs = []
        xs.append(y + bump())
        assert xs[0] == 5
    """),
    "eo_method_two_args": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        xs = [9]
        xs.insert(y, bump())
        assert xs[0] == 9
    """),
    "eo_boolop_value": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        z = (y == 0) and bump() == 0
        assert y == 0
    """),
    "eo_heap_store_then_call": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        t = (y,)
        bump()
        assert t[0] == 5
    """),
    "eo_name_store_then_call": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        z = y
        bump()
        assert z == 5
    """),
    "eo_range_args": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        n = 0
        for i in range(y, bump() + 1):
            n += 1
        assert n == 0
    """),
    "eo_while_test": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        k = 0
        while y < bump() + 1 and k < 1:
            k += 1
        assert k == 0
    """),
    "eo_ifexp_then_call": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        x = (y if True else 0) + bump()
        assert x == 5
    """),
    "eo_unary_then_call": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        x = -y + bump()
        assert x == -5
    """),
    "eo_floordiv_rhs_call": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        x = 10 // (y + bump() + 1)
        assert x == 1
    """),
    "eo_div_by_saved_zero": (AR, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def first(a, b):
            return a
        x = 10 // first(y, bump())
    """),
    "eo_index_saved_oob": (DL, """
        y = 9
        def bump():
            global y
            y = 0
            return 0
        xs = [0]
        def first(a, b):
            return a
        v = xs[first(y, bump())]
    """),
    # [internal ref]: abs()'s TypeError condition is "no __abs__" (src/tharkun/solve/contracts/rules_builtins.py's
    # existing "abs" contract) -- a plain object with none must be recorded as an
    # unproven check, never silently treated as safe.
    "abs_of_object_without_dunder_is_never_proven_safe": (DT, """
        class Plain:
            pass
        abs(Plain())
    """),
    # [internal ref]: the a >> k == a // 2**k tightening in _int_binop is gated on a's sign
    # being KNOWN non-negative; a possibly-negative shift operand must still get the
    # old, loose [0, a's own upper bound] treatment, not a wrong tight one.
    "shift_bound_is_not_tightened_for_unknown_sign": (DL, """
        import _sv_verifier
        n = _sv_verifier.nondet_int()
        xs = list(range(100))
        if n < 200:
            i = n >> 1
            print(xs[i])
    """),
    # [internal ref]: sorted()'s element fact must be a JOIN over every key the dict may hold,
    # never just one of them -- a violation reachable through an element sorted()
    # actually can produce (here, key 1) must stay reachable, not look proven-safe
    # because the join silently narrowed to a single key.
    "sorted_of_a_dict_keeps_every_key_reachable": (DL, """
        import _sv_verifier
        TABLE = {0: "safe", 1: "bad"}
        n = _sv_verifier.nondet_int()
        keys = sorted(TABLE)
        i = abs(n) % 2
        if keys[i] == 1:
            raise IndexError("hit the real key")
    """),
}


@pytest.mark.parametrize("name", sorted(NEVER_TRUE))
def test_programs_with_a_reachable_violation_are_never_true(name):
    classes, body = NEVER_TRUE[name]
    r = prove(_src(body), classes)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "unproven check" in r.reason, r.reason  # found the site, did not just give up


# --- [internal ref] batch 5 (the protocol-level exception-swallowing audit, docs/contracts/REPORT.md): the length probe
# of list()/list.extend()/sorted() (CPython's PyObject_LengthHint) calls __len__ and swallows exactly TypeError.
# The model of [internal ref] (PR 65) swallowed AttributeError too, restored the zone from before the call (dropping
# the callee's writes to globals) and ended the path of every conditional raise: five wrong TRUEs, all with a
# ZeroDivisionError after the call that CPython reaches.  tuple() and array.array() never call __len__, so
# probing there made a raise in __len__ wrongly end the path.
LENGTH_PROBE_NEVER_TRUE = {
    # the swallowing path keeps the side effects __len__ made before it raised
    "uncond_raise_writes_a_global": (AR, """
        g = 1
        class L:
            def __iter__(self):
                return iter([1])
            def __len__(self):
                global g
                g = 0
                raise TypeError("no length")
        xs = list(L())
        print(1 // g)
    """),
    # ... on the paths where it raised, and only there
    "cond_raise_writes_a_global": (AR, """
        import _sv_verifier
        g = 1
        class L:
            def __iter__(self):
                return iter([1])
            def __len__(self):
                global g
                if _sv_verifier.nondet_bool():
                    g = 0
                    raise TypeError("no length")
                return 1
        xs = list(L())
        print(1 // g)
    """),
    "cond_raise_writes_an_attribute": (AR, """
        import _sv_verifier
        class L:
            def __init__(self):
                self.k = 1
            def __iter__(self):
                return iter([1])
            def __len__(self):
                if _sv_verifier.nondet_bool():
                    self.k = 0
                    raise TypeError("no length")
                return 1
        l = L()
        xs = list(l)
        print(1 // l.k)
    """),
    "extend_cond_raise_writes_a_global": (AR, """
        import _sv_verifier
        g = 1
        class L:
            def __iter__(self):
                return iter([1])
            def __len__(self):
                global g
                if _sv_verifier.nondet_bool():
                    g = 0
                    raise TypeError("no length")
                return 1
        xs = []
        xs.extend(L())
        print(1 // g)
    """),
    # a TypeError raised by native code inside __len__ is swallowed too
    "extend_len_of_an_int": (AR, """
        import _sv_verifier
        class L:
            def __init__(self):
                self.items = [1]
            def __iter__(self):
                return iter(self.items)
            def __len__(self):
                return len(5)
        d = _sv_verifier.nondet_int()
        xs = []
        xs.extend(L())
        print(1 // d)
    """),
    "extend_len_with_the_wrong_arity": (AR, """
        import _sv_verifier
        class L:
            def __init__(self):
                self.items = [1]
            def __iter__(self):
                return iter(self.items)
            def __len__(self, extra):
                return 1
        d = _sv_verifier.nondet_int()
        xs = []
        xs.extend(L())
        print(1 // d)
    """),
    # array.array.extend never asks for a length hint: a raise in __len__ must not end the path
    "array_extend_never_calls_len": (AR, """
        import _sv_verifier
        import array
        class L:
            def __init__(self):
                self.items = [1]
            def __iter__(self):
                return iter(self.items)
            def __len__(self):
                raise ValueError("boom")
        d = _sv_verifier.nondet_int()
        xs = array.array('i')
        xs.extend(L())
        print(1 // d)
    """),
}


@pytest.mark.parametrize("name", sorted(LENGTH_PROBE_NEVER_TRUE))
def test_the_length_probe_never_makes_a_reachable_violation_look_unreachable(name):
    classes, body = LENGTH_PROBE_NEVER_TRUE[name]
    r = prove(_src(body), classes)
    assert r.verdict == "UNKNOWN", r.explain()


# CPython propagates every class but TypeError out of the probe, so these programs end at the probe and the
# division after it is never reached: TRUE is right (and was wrongly UNKNOWN or, for AttributeError, swallowed).
LENGTH_PROBE_PROVABLE = {
    "attributeerror_propagates": (AR, """
        class L:
            def __init__(self):
                self.items = [1]
            def __iter__(self):
                return iter(self.items)
            def __len__(self):
                raise AttributeError("no")
        xs = []
        xs.extend(L())
        print(1 // 0)
    """),
    "valueerror_propagates": (AR, """
        class L:
            def __init__(self):
                self.items = [1]
            def __iter__(self):
                return iter(self.items)
            def __len__(self):
                raise ValueError("no")
        xs = []
        xs.extend(L())
        print(1 // 0)
    """),
}


@pytest.mark.parametrize("name", sorted(LENGTH_PROBE_PROVABLE))
def test_the_length_probe_ends_the_path_for_the_classes_cpython_propagates(name):
    classes, body = LENGTH_PROBE_PROVABLE[name]
    r = prove(_src(body), classes)
    assert r.verdict == "TRUE", r.explain()


# --- [internal ref]: a reflective/unresolved object-implicit dunder must never make the state
# look unreachable (bottom) -- it must abort the proof (Unsupported -> UNKNOWN), the same
# as any other construct absint does not model. Root cause: attr_load()/call_method()
# treated class_lookup() returning None (the name is not anywhere in the MODELED class
# hierarchy) as "this receiver contributes nothing", which is only sound for a name that
# truly does not exist anywhere -- not for one of object's own always-present dunders
# (__repr__, __str__, __hash__, ...), which the real class inherits and which always
# succeeds. MG/MAIN repro (bundle [internal ref] review): a[int(s[-3], 16)] downstream of s coming
# from an unresolved __repr__()/__str__ is reachable and raises IndexError/KeyError
# natively in 12/12 runs; absint answered TRUE with 0 checks (the downstream subscript
# was never visited because the state had already gone bottom evaluating s).
REFLECTIVE_DUNDER_NEVER_TRUE = {
    # a16 (MM/MG repro, [host]:[local path]
    # a16_dunder_repr_call.py): an explicit call to the implicit object.__repr__.
    "a16_dunder_repr_call": (DL, """
        class P:
            pass

        p = P()
        s = p.__repr__()

        a = [0] * 12
        print(a[int(s[-3], 16)])
    """),
    # a18 (same directory, a18_method_wrapper.py): the implicit object.__str__ read as an
    # attribute (a method-wrapper value), then stringified -- attr_load's turn, not
    # call_method's.
    "a18_method_wrapper": (DL, """
        class P:
            pass

        p = P()
        s = str(p.__str__)

        a = [0] * 12
        print(a[int(s[-3], 16)])
    """),
}


@pytest.mark.parametrize("name", sorted(REFLECTIVE_DUNDER_NEVER_TRUE))
def test_reflective_object_dunders_abort_rather_than_look_unreachable(name):
    classes, body = REFLECTIVE_DUNDER_NEVER_TRUE[name]
    r = prove(_src(body), classes)
    assert r.verdict == "UNKNOWN", r.explain()
    assert "reflective" in r.reason and "__" in r.reason, r.reason


# Safe mirrors: the SAME shape, but the class explicitly overrides the dunder with a
# provable body -- class_lookup() now finds a real cv, so this must resolve through the
# ordinary method-call path exactly as before [internal ref] (not through the new Unsupported
# branch, which is reserved for the truly-unresolved case). This guards against the fix
# over-triggering on every dunder call, explicitly defined ones included.
REFLECTIVE_DUNDER_SAFE_MIRRORS = {
    "a16_explicit_repr_is_not_reflective": (DL, """
        class P:
            def __repr__(self):
                return "00000000"

        p = P()
        s = p.__repr__()

        a = [0] * 12
        print(a[int(s[-3], 16)])
    """),
    "a18_explicit_str_is_not_reflective": (DL, """
        class P:
            def __str__(self):
                return "00000000"

        p = P()
        s = str(p.__str__())

        a = [0] * 12
        print(a[int(s[-3], 16)])
    """),
}


@pytest.mark.parametrize("name", sorted(REFLECTIVE_DUNDER_SAFE_MIRRORS))
def test_explicitly_overridden_dunders_are_not_treated_as_reflective(name):
    classes, body = REFLECTIVE_DUNDER_SAFE_MIRRORS[name]
    r = prove(_src(body), classes)
    assert "reflective" not in r.reason, r.explain()


# --- [internal ref], widened (MG): every text-producing route to an implicit object dunder or a
# method-wrapper/bound-method value, not just the two originally-reported shapes. Same
# root cause each time (class_lookup() returning None for an unresolved-but-real name),
# reached through a different AST shape per variant -- an explicit call, an attribute read
# alone, an f-string, type()/__class__ indirection, or the builtin `object.__repr__(p)`
# unbound-call spelling. Repro: [host]:[local path] (MG). All 11
# raise IndexError natively; none may ever answer TRUE.
_P1091_K = "class P:\n    pass\np = P()\na = [0] * 12\n"
REFLECTIVE_DUNDER_WIDENED_NEVER_TRUE = {
    # a19 (MM, [host]:[local path]):
    # a class's OWN __dict__ (its namespace view -- not a dunder METHOD, but the same
    # "implicit, never in the modeled attrs table" story) subscripted for a function object,
    # whose default repr also shows an address.
    "class_dict_function_repr": (
        "class P:\n    def m(self):\n        return 1\n\n"
        "s = str(P.__dict__['m'])\n\n"
        "a = [0] * 12\nprint(a[int(s[-3], 16)])\n"
    ),
    "repr_call": _P1091_K + "s = p.__repr__()\nv = a[int(s[-3], 16)]\n",
    "str_call": _P1091_K + "s = p.__str__()\nv = a[int(s[-3], 16)]\n",
    "format_call": _P1091_K + "s = p.__format__('')\nv = a[int(s[-3], 16)]\n",
    "str_of_wrapper": _P1091_K + "s = str(p.__str__)\nv = a[int(s[-3], 16)]\n",
    "repr_of_init": _P1091_K + "s = repr(p.__init__)\nv = a[int(s[-3], 16)]\n",
    "type_repr": _P1091_K + "s = type(p).__repr__(p)\nv = a[int(s[-3], 16)]\n",
    "class_repr": _P1091_K + "s = p.__class__.__repr__(p)\nv = a[int(s[-3], 16)]\n",
    "hash_call": _P1091_K + "v = a[p.__hash__() % 13]\n",
    "reduce_ex": _P1091_K + "s = str(p.__reduce_ex__(2))\nv = a[int(s[-3], 16)]\n",
    "object_repr": _P1091_K + "s = object.__repr__(p)\nv = a[int(s[-3], 16)]\n",
    "fstr_wrapper": _P1091_K + "s = f'{p.__repr__}'\nv = a[int(s[-3], 16)]\n",
}


@pytest.mark.parametrize("name", sorted(REFLECTIVE_DUNDER_WIDENED_NEVER_TRUE))
def test_widened_reflective_dunder_variants_are_never_true(name):
    body = REFLECTIVE_DUNDER_WIDENED_NEVER_TRUE[name]
    r = prove(body, ("IndexError",), filename="t.py")
    assert r.verdict == "UNKNOWN", r.explain()


# --- [internal ref] part 2 (MG): class_lookup() misses INHERITED, non-dunder attributes of a
# builtin base a user class extends (currently only the exception family -- Exception,
# ValueError, BaseException, KeyError, ... -- recorded as the __exc_base__ marker in
# s_ClassDef rather than a real, walkable heap base). "args" and "with_traceback" are
# real, always-present attributes of any real exception instance; class_lookup() finding
# nothing for them (its walk never reaches an unmodelled builtin) must not be read as
# "this attribute does not exist" the way it would be read for an ordinary undefined name.
# Repro: [host]:[local path] (MG).
EXC_HIDDEN_ATTR_NEVER_TRUE = {
    # exc_hidden.py verbatim: a random.randint()-gated branch (never provably unreachable
    # to absint) that indexes past the end of a 3-element list using len(e.args).
    "exc_hidden_verbatim": (
        "import random\n"
        "class MyErr(Exception):\n    pass\n"
        "e = MyErr('a', 'b')\n"
        "xs = [0, 0, 0]\n"
        "x = random.randint(0, 10**12)\n"
        "h = x\n"
        "for _ in range(40):\n    h = (h * h + 12345) % 1000000007\n"
        "if h == 875841331:\n    v = xs[len(e.args) + 1]\n"
    ),
}
for _base in ("Exception", "ValueError", "BaseException", "KeyError"):
    EXC_HIDDEN_ATTR_NEVER_TRUE[f"{_base.lower()}_args"] = (
        f"class P({_base}):\n    pass\np = P('ab', 'c')\na = [0] * 3\nv = a[len(p.args) + 1]\n"
    )
EXC_HIDDEN_ATTR_NEVER_TRUE["keyerror_with_traceback"] = (
    "class P(KeyError):\n    pass\np = P('ab', 'c')\na = [0] * 3\nv = a[3 if p.with_traceback else 0]\n"
)


@pytest.mark.parametrize("name", sorted(EXC_HIDDEN_ATTR_NEVER_TRUE))
def test_exception_family_hidden_attributes_are_never_true(name):
    body = EXC_HIDDEN_ATTR_NEVER_TRUE[name]
    r = prove(body, ("IndexError",), filename="t.py")
    assert r.verdict == "UNKNOWN", r.explain()
    assert "reflective" in r.reason, r.reason


def test_exception_family_safe_mirror_stays_provable():
    """Safe mirror: a REAL, fully modelled base class (no unmodelled builtin in the MRO)
    still resolves an inherited attribute through class_lookup() exactly as before -- the
    new _mro_has_unmodelled_base() check must not fire, and precision is unchanged, when
    there is nothing unknown in the hierarchy."""
    src = (
        "class Base:\n    args = (0,)\n"
        "class MyErr(Base):\n    pass\n"
        "e = MyErr()\n"
        "xs = [0, 0, 0]\n"
        "v = xs[len(e.args)]\n"
    )
    r = prove(src, ("IndexError",), filename="t.py")
    assert r.verdict == "TRUE", r.explain()


# --- [internal ref] safe mirrors: same shape, no hazard (the mutator runs before the read it
# used to race), so these must stay provable TRUE after the materialize-on-every-read
# fix. Covers all 17 that were wrongly TRUE pre-fix, plus 3 more of eo.py's 25 whose
# mirror also happens to be clean. Not attempted for eo_subscript_store (its hazard
# operand order is reversed -- target[idx] = call() evaluates the RHS call before the
# target/index, so "call the mutator first" removes nothing) or eo_boolop_value (its
# assert checks the value of y strictly after bump() runs by construction, so there is
# no hazard-free reordering that keeps the same assertion true) -- both are already
# correctly UNKNOWN pre- and post-fix and are not evidence for or against this bug.
# 3 of the 20 remain UNKNOWN even here (eo_list_display, eo_method_two_args,
# eo_subscript_index -- excluded below) for reasons unrelated to global materialization
# (list/dict-display element-join and method-call precision): they were already UNKNOWN,
# not wrongly TRUE, before this fix, so they are not part of this regression set; only
# the ones the fixed absint successfully proves are asserted here.
EVAL_ORDER_SAFE_MIRRORS = {
    "eo_binop_fixed": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        x = y + nop()
        assert x == 5
    """),
    "eo_compare_fixed": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        assert y == nop() + 5
    """),
    "eo_augassign": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        y += nop()
        assert y == 5
    """),
    "eo_call_args_user": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        def first(a, b):
            return a
        x = first(y, nop())
        assert x == 5
    """),
    "eo_call_args_builtin": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        x = max(y, nop())
        assert x == 5
    """),
    "eo_call_args_linexpr": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        def first(a, b):
            return a
        x = first(y + 1, nop())
        assert x == 6
    """),
    "eo_tuple_display": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        t = (y, nop())
        assert t[0] == 5
    """),
    "eo_tuple_unpack": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        a, b = y, nop()
        assert a == 5
    """),
    "eo_slice_bounds": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        xs = [0, 1, 2, 3, 4, 5, 6]
        s = xs[y:nop() + 7]
        assert len(s) == 2
    """),
    "eo_method_args": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        xs = []
        xs.append(y + nop())
        assert xs[0] == 5
    """),
    "eo_range_args": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        n = 0
        for i in range(y, nop() + 1):
            n += 1
        assert n == 0
    """),
    "eo_while_test": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        k = 0
        while y < nop() + 1 and k < 1:
            k += 1
        assert k == 0
    """),
    "eo_ifexp_then_call": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        x = (y if True else 0) + nop()
        assert x == 5
    """),
    "eo_unary_then_call": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        x = -y + nop()
        assert x == -5
    """),
    "eo_floordiv_rhs_call": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        x = 10 // (y + nop() + 1)
        assert x == 1
    """),
    "eo_div_by_saved_zero": (AR, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        def first(a, b):
            return a
        x = 10 // first(y, nop())
    """),
    "eo_index_saved_oob": (DL, """
        y = 9
        def bump():
            global y
            y = 0
            return 0
        def nop():
            return 0

        bump()
        xs = [0]
        def first(a, b):
            return a
        v = xs[first(y, nop())]
    """),
    "eo_dict_display": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        d = {0: y, 1: nop()}
        assert d[0] == 5
    """),
    "eo_heap_store_then_call": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        t = (y,)
        nop()
        assert t[0] == 5
    """),
    "eo_name_store_then_call": (AS, """
        y = 0
        def bump():
            global y
            y = 5
            return 0
        def nop():
            return 0

        bump()
        z = y
        nop()
        assert z == 5
    """),
}


@pytest.mark.parametrize("name", sorted(EVAL_ORDER_SAFE_MIRRORS))
def test_eval_order_safe_mirrors_stay_provable(name):
    """[internal ref] regression: materializing a global on every read (not only inside a
    function frame) must not cost precision on a program that has no read-before-a-
    rebinding-call hazard at all -- the read already sees the settled value either way.
    """
    classes, body = EVAL_ORDER_SAFE_MIRRORS[name]
    r = prove(_src(body), classes)
    assert r.verdict == "TRUE", r.explain()
# ------------------------------------------------------------------ aliases (lane F, b01)
ALIASED_MUTATION = {
    # lane F's witness: `b *= 0` empties the list `a` names too (n = 1657 reaches it)
    "f_b01_imul_alias": """
        import _sv_verifier
        n = _sv_verifier.nondet_int()
        a = [0, 0]
        b = a
        h = 0
        for i in range(8):
            h = (h * 31 + n) % 10007
        if h == 4242:
            b *= 0
        print(a[1])
    """,
    "imul_alias_minimal": """
        import _sv_verifier
        n = abs(_sv_verifier.nondet_int())
        a = [0] * (n + 1)
        b = a
        b *= _sv_verifier.nondet_int()
        print(a[0])
    """,
    "slice_clear_alias": """
        import _sv_verifier
        a = [1, 2, 3]
        b = a
        if _sv_verifier.nondet_bool():
            b[:] = []
        print(a[0])
    """,
    "pop_alias": """
        import _sv_verifier
        a = [1]
        b = a
        if _sv_verifier.nondet_bool():
            b.pop()
        print(a[0])
    """,
    "pop_alias_in_callee": """
        import _sv_verifier
        def drain(xs):
            while _sv_verifier.nondet_bool() and len(xs) > 0:
                xs.pop()
        a = [1, 2]
        drain(a)
        print(a[1])
    """,
    "iadd_alias_then_index_past": """
        import _sv_verifier
        a = [1]
        b = a
        if _sv_verifier.nondet_bool():
            b += [2]
        print(a[1])
    """,
    "attr_alias_imul": """
        import _sv_verifier
        class C:
            def __init__(self, xs):
                self.xs = xs
        a = [0, 0]
        c = C(a)
        c.xs *= _sv_verifier.nondet_int()
        print(a[0])
    """,
}


@pytest.mark.parametrize("name", sorted(ALIASED_MUTATION))
def test_in_place_mutation_through_an_alias_is_never_true(name):
    r = prove(_src(ALIASED_MUTATION[name]), DL)
    assert r.verdict == "UNKNOWN", r.explain()


def test_in_place_extend_through_an_alias_is_seen_by_every_name():
    """`b += [2]` extends the object `a` names too: a[1] is then safe."""
    r = prove(_src("""
        a = [1]
        b = a
        b += [2]
        print(a[1])
    """), DL)
    assert r.verdict == "TRUE", r.explain()


# ------------------------------------------------------------------ unsupported is UNKNOWN
UNSUPPORTED = {
    # [internal ref]: try/except is now modelled; except* / ExceptionGroup is the unmodelled construct that aborts
    "try_star": """
        try:
            [][0]
        except* ExceptionGroup:
            pass
    """,
    "unknown_library_call": """
        import os
        os.listdir('.')
    """,
    # [internal ref]: mixed key types (str vs int) really do raise TypeError in real Python's
    # sorted() -- refuse rather than guess an order, never silently pick one side.
    "sorted_of_a_dict_with_mixed_key_types": """
        TABLE = {0: "a", "b": 1}
        sorted(TABLE)
    """,
    "recursion": """
        def f(n):
            return f(n - 1) if n else 0
        f(3)
    """,
    "eval": """
        eval("[][0]")
    """,
    "unmodelled_module_attribute": """
        import sys
        print(len(sys.argv))
    """,
    # [internal ref]: this used to be "f = sorted; print(len(f))" -- now that sorted() is
    # modelled as a genuine builtin-function value (not TOP), len() of the bare
    # value provably raises TypeError (real Python: a builtin_function_or_method
    # has no len()) and NEVER a DL-class exception, so absint now correctly (and
    # more precisely) proves this TRUE instead of refusing -- see PROVABLE's
    # "len_of_a_modelled_builtin_value_is_provably_safe". zip() stays a genuinely
    # unmodelled builtin value, keeping this test's original intent.
    "unmodelled_builtin_value": """
        f = zip
        print(len(f))
    """,
    "getattr": """
        getattr([], 'pop')()
    """,
    "user_eq": """
        class P:
            def __eq__(self, o):
                return [][0]
        P() == P()
    """,
    # unmodelled means abort, never a made-up TypeError that cuts the path
    "union_type_operator": """
        x = int | None
        [][0]
    """,
    "generic_alias_subscript": """
        t = list[int]
        [][0]
    """,
    "module_attribute_store": """
        import math
        math.tau2 = 1.0
        [][0]
    """,
    "class_operator": """
        class C:
            pass
        u = C | None
        [][0]
    """,
    "dict_union_operator": """
        d = {'a': 1} | {'b': 2}
        [][0]
    """,
    "choice_of_a_range": """
        import random
        x = random.choice(range(3))
        [][0]
    """,
    "unpack_a_dict": """
        a, b = {'a': 1, 'b': 2}
        [][0]
    """,
    "closure": """
        def outer():
            xs = []
            def inner():
                return xs[0]
            return inner()
        outer()
    """,
    # [internal ref]: abs() raises TypeError only on an argument with no __abs__ (the existing
    # "abs" contract, src/tharkun/solve/contracts/rules_builtins.py) -- a class that DEFINES __abs__ is
    # exactly as unmodelled as any other UNSUPPORTED_DUNDERS case (__abs__ is not in
    # SUPPORTED_DUNDERS), so absint must abort to UNKNOWN rather than guess whether the
    # dunder raises, returns something abs()-shaped, or does neither.
    "user_abs_dunder": """
        class Meters:
            def __abs__(self):
                return self
        m = Meters()
        abs(m)
        [][0]
    """,
}


@pytest.mark.parametrize("name", sorted(UNSUPPORTED))
def test_unmodelled_constructs_are_unknown(name):
    r = prove(_src(UNSUPPORTED[name]), DL)
    assert r.verdict == "UNKNOWN" and r.reason


# ------------------------------------------------------------------ small proofs
PROVABLE = {
    "lambda_alias_expression_body": (AS, """
        f = lambda x: x + 1
        alias = f
        assert alias(2) == 3
    """),
    "segment_fill_then_use": (DT, """
        import _sv_verifier
        class P:
            def f(self):
                return 1
        n = 1 + abs(_sv_verifier.nondet_int())
        xs = [None] * n
        for i in range(n):
            xs[i] = P()
        for p in xs:
            p.f()
        xs[n - 1].f()
        xs.append(P())
        xs[n].f()
    """),
    "loop_index_under_len": (DL, """
        import _sv_verifier
        n = 1 + abs(_sv_verifier.nondet_int())
        xs = [0] * n
        for i in range(n):
            xs[i] = i
        print(xs[n - 1], xs[-n])
    """),
    # [internal ref]: sorted() of an exact dict is a new list with the dict's own EXACT key
    # count as its length -- a loop over range(len(keys)) must be provably in bounds,
    # same as any other exactly-sized list.
    "sorted_of_a_dict_then_bounded_index": (DL, """
        TABLE = {0: "a", 1: "b", 2: "c"}
        keys = sorted(TABLE)
        for i in range(len(keys)):
            print(keys[i])
    """),
    # [internal ref]: sorted (now modelled as a builtin-function VALUE, not just a call) taken
    # by itself and passed to len() -- real Python's len() of a builtin_function_or_
    # method always raises TypeError, never a DL-class exception, so this genuinely
    # proves TRUE for DL now (see UNSUPPORTED's updated "unmodelled_builtin_value").
    "len_of_a_modelled_builtin_value_is_provably_safe": (DL, """
        f = sorted
        print(len(f))
    """),
    "division_by_positive": (AR, """
        import _sv_verifier
        n = 1 + abs(_sv_verifier.nondet_int())
        print(100 // n, 7 % n)
    """),
    "while_countdown": (DL, """
        import _sv_verifier
        n = 1 + abs(_sv_verifier.nondet_int())
        xs = list(range(n))
        r = n
        while r != 0:
            xs[r - 1] = 0
            r -= 1
    """),
    "method_calls": (DT, """
        import _sv_verifier
        class P:
            __slots__ = ('x',)
            def __init__(self, x):
                self.x = x
            def get(self):
                return self.x + 1
        p = P(_sv_verifier.nondet_int())
        print(p.get())
    """),
    # Negative controls for the NEVER_TRUE entries just above: same shape, but
    # actually safe, so the fix must not have turned "swallow the raise / read
    # the overridden class attribute" into a blanket refusal or a wrong UNKNOWN.
    "extend_hint_typeerror_from_len_is_swallowed_safe": (AR, """
        class Box:
            def __init__(self):
                self.items = [1, 2]
            def __iter__(self):
                return iter(self.items)
            def __len__(self):
                raise TypeError("length unknown")
        items = [0]
        items.extend(Box())
        print(10 // len(items))
    """),
    # [internal ref] review (MG): a plain class-level value can win over an intact slot
    # descriptor found later in the MRO (attr_load/_mro_attr_or_slot's "attr"
    # case) without the model being able to tell whether an instance's own
    # recorded value (if any) is really invisible now (written back when the
    # name was still a slot) or a genuine, still-visible value (written after
    # the override, on a hierarchy with a __dict__ somewhere) -- so both stay
    # joined, soundly but not always precisely (see slot_overridden_by_class_
    # attribute_read in NEVER_TRUE: interval-joining a 0 with this override's
    # 5 loses enough precision to make THAT case's division unprovable, by
    # design). This safe mirror instead overrides before the class is ever
    # instantiated, so there is nothing recorded to join with and the class
    # value alone -- fully precise -- governs the read.
    "slot_overridden_by_class_attribute_read_safe": (AR, """
        class Cell:
            __slots__ = ("x",)
        Cell.x = 5
        c = Cell()
        print(10 // (c.x - 3))
    """),
    "slot_write_without_override_is_safe": (DT, """
        class Cell:
            __slots__ = ("x",)
            def __init__(self):
                self.x = 0
        c = Cell()
        c.x = 1
        print(c.x)
    """),
    # Negative controls for the subclass/MRO entries just above ([internal ref] review,
    # MG): a subclass that inherits a base's slots, with no override anywhere,
    # must still prove exactly as precisely as the single-class case --
    # whether it redeclares an empty __slots__ = () or not.
    "subclass_without_own_slots_no_override_is_safe": (AR, """
        class Base:
            __slots__ = ("x",)
            def __init__(self):
                self.x = 0
        class Sub(Base):
            pass
        s = Sub()
        print(10 // (s.x - 3))
    """),
    "subclass_with_empty_slots_no_override_is_safe": (AR, """
        class Base:
            __slots__ = ("x",)
            def __init__(self):
                self.x = 0
        class Sub(Base):
            __slots__ = ()
        s = Sub()
        print(10 // (s.x - 3))
    """),
    # [internal ref]: abs(-e) == abs(e) -- provable when e's sign is already known, since abs()
    # then takes the exact (Lin-preserving) fast path both ways: abs(e) reuses e's own
    # linear form when e >= 0 is known, and abs(-e) negates -e's linear form straight
    # back to e's via the same unop(USub) round-trip. (The general, sign-unknown case
    # needs relating abs(x - y) and abs(y - x) across the zone -- see [internal ref] handoff.)
    "abs_of_negation_equals_abs_when_sign_known": (AS, """
        import _sv_verifier
        x = 5 + abs(_sv_verifier.nondet_int())
        assert abs(-x) == abs(x)
    """),
    # [internal ref]: a >> k == a // 2**k for a >= 0, as a fact the prover actually uses: once
    # the interval a right-shift produces is as tight as floor-division's (not the old
    # "[0, a's own upper bound]"), a realistic index bound becomes provable through it.
    "shift_by_constant_equals_floordiv_enables_index_proof": (DL, """
        import _sv_verifier
        n = abs(_sv_verifier.nondet_int())
        xs = list(range(100))
        if n < 200:
            i = n >> 1
            print(xs[i])
    """),
    # [internal ref] (MAIN's guarded rewrite, 2026-09-28): the owner's own phrasing -- "a // 2
    # equals a >> 1 for non-negative a" -- as a direct equality assertion, not just a
    # bound. compare_split's _shr_floordiv_identity_holds recognizes X // 2**k and
    # X >> k for the same bare-name X, with X's sign proven, and canonicalizes the
    # comparison to a trivial x == x rather than needing a shared exact Lin.
    "shift_equals_floordiv_for_same_var_when_nonneg": (AS, """
        import _sv_verifier
        a = abs(_sv_verifier.nondet_int())
        assert (a // 2) == (a >> 1)
    """),
    # Same fact, both directions of the identity (>>k == //2**k, not just //2**k ==
    # >>k) and a non-trivial k, over a variable whose non-negativity is established
    # by an offset (not by abs()) -- the sign check reads the CURRENT env value, not
    # anything abs()-specific.
    "shift_equals_floordiv_reversed_order_and_k2": (AS, """
        import _sv_verifier
        a = 3 + abs(_sv_verifier.nondet_int())
        assert (a >> 2) == (a // 4)
    """),
}


@pytest.mark.parametrize("name", sorted(PROVABLE))
def test_small_programs_are_proven(name):
    classes, body = PROVABLE[name]
    r = prove(_src(body), classes)
    assert r.verdict == "TRUE", r.explain()


# [internal ref] negative control (MAIN, 2026-09-28): compare_split's
# _shr_floordiv_identity_holds must NOT fire when X's sign isn't proven
# non-negative -- even though a // 2 == a >> 1 happens to hold for every int in
# real Python (verified by brute force, positive and negative), the rewrite
# only claims the non-negative case the ticket asked for, and must stay
# UNKNOWN rather than get lucky. Guards the gate itself, not the underlying
# arithmetic fact.
def test_shift_floordiv_identity_stays_unknown_without_a_proven_sign():
    r = prove(_src("""
        import _sv_verifier
        n = _sv_verifier.nondet_int()
        assert (n // 2) == (n >> 1)
    """), AS)
    assert r.verdict == "UNKNOWN", r.explain()


# [internal ref] negative control: a DIFFERENT variable on each side, even with matching
# bounds, must never be treated as the same term -- _shr_floordiv_identity_holds
# requires the exact same bare NAME on both sides, not just "some non-negative
# int // 2**k compared to some non-negative int >> k".
def test_shift_floordiv_identity_requires_the_same_variable():
    r = prove(_src("""
        import _sv_verifier
        a = abs(_sv_verifier.nondet_int())
        b = abs(_sv_verifier.nondet_int())
        assert (a // 2) == (b >> 1)
    """), AS)
    assert r.verdict == "UNKNOWN", r.explain()


# ------------------------------------------------------------------ site records
def test_records_cover_every_site_and_are_complete():
    r = prove((FIX / "targets" / "bm_fannkuch_nondet.py").read_text(), DL)
    assert r.complete and r.source_sha
    assert {rec["line"] for rec in r.records} >= {30, 33, 36, 40, 46, 47, 48}
    assert all(rec["status"] == "safe" and rec["fact"] for rec in r.records)


def test_a_may_raise_record_carries_a_parseable_residual_and_must_facts():
    import ast as _ast
    r = prove((FIX / "never_true" / "fk_n0.py").read_text(), DL)
    bad = [rec for rec in r.records if rec["status"] == "may_raise"]
    assert bad and r.complete
    rec = bad[0]
    _ast.parse(rec["residual"], mode="eval")
    for fact in rec["scope"]:
        _ast.parse(fact, mode="eval")
    assert set(rec["types"]) <= {n.id for n in _ast.walk(_ast.parse(rec["residual"], mode="eval"))
                                 if isinstance(n, _ast.Name)}


def test_scope_facts_hold_on_concrete_runs():
    """Must-facts: evaluate every scope fact at the site on real executions."""
    src = _src("""
        import _sv_verifier
        n = abs(_sv_verifier.nondet_int())
        k = n + 3
        print(10 // (k - n - 3))
    """)
    r = prove(src, AR)
    rec = [x for x in r.records if x["kind"] == "divide"][0]
    assert rec["status"] == "may_raise" and rec["residual"] == "(k - n - 3) == 0"
    for n in range(0, 50):
        env = {"n": n, "k": n + 3}
        assert all(eval(f, {}, env) for f in rec["scope"]), rec["scope"]
    assert "k - n <= 3" in rec["scope"] and "n - k <= -3" in rec["scope"]


def test_an_abort_is_incomplete_with_no_records():
    r = prove(_src(UNSUPPORTED["try_star"]), DL)
    assert not r.complete and r.records == []


@pytest.mark.parametrize("name", ["__builtins__", "__package__", "__spec__", "__loader__", "__cached__"])
def test_runtime_module_globals_do_not_erase_reachable_errors(name):
    result = prove(f"{name}\n[][0]\n", DL)
    assert result.verdict == "UNKNOWN", result.explain()
