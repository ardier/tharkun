"""Flow sensitivity in the drive loop: a reassignment invalidates the facts it touches.

The bug this guards against ("Witness B"): a guard establishes ``(a > 100)`` about a
parameter, the path REASSIGNS ``a = b``, and a deeper guard ``a == 7`` is then solved with
the stale ``(a > 100)`` still in its precondition. Touchstone proves the conjunction
infeasible over one namespace and the driver reports a REACHABLE line as "provably dead".

The fix threads the path condition as a ``FactState`` whose clauses are anchored to the
program point that established them, so a reassignment on the executed path can drop exactly
the clauses that named the reassigned variable — at the requires render, at the conjunction
search, and at the backtrack's conflict transport.
"""
from __future__ import annotations

import ast
import os
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import pytest

from tharkun.driver import (
    PLATEAU,
    REACHED,
    _binding_index,
    _clause_is_stale,
    _killed_between,
    drive_to_line,
)
from tharkun.hypothesis_adapter import hypothesis_available
from tharkun.lattice import TOP, Clause, Origin

requires_hypothesis = unittest.skipUnless(hypothesis_available(),
                                          "the all-parameter reach-search needs hypothesis")


def _write(src: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mod.py")
    with open(p, "w") as f:
        f.write(textwrap.dedent(src))
    return p


class TestReachProbeAgreesWithFullTrace(unittest.TestCase):
    """The fast localized reach-probe must give the SAME verdict as a whole-program trace.

    The probe (`_target_reached_probe`) instruments only the target's own code object for
    speed; it is on the hot path (reach-search and every `_flip_verifies`), so a divergence
    from the authoritative full trace would silently change which candidates verify.
    """

    def _probe_vs_trace(self, path, func, target_line, inputs, tp=None):
        from tharkun.driver import _run_traced, _target_reached_probe

        tp = tp or path
        probed = _target_reached_probe(path, func, inputs, target_line, tp,
                                       frozenset({os.path.realpath(tp)}))
        oc = _run_traced(path, func, inputs, {target_line},
                         extra_paths=frozenset({os.path.realpath(tp)}),
                         extra_snapshot_lines={os.path.realpath(tp): {target_line}})
        rp = os.path.realpath(path)
        rtp = os.path.realpath(tp)
        executed = oc.by_path.get(rtp, set()) if rtp != rp else oc.executed
        full = target_line in executed
        return probed, full

    def test_probe_matches_trace_across_reaching_and_non_reaching_inputs(self) -> None:
        # A method target reached through a heavier callee — the shape the probe exists for.
        p = _write("""
            class Codec:
                def step(self, n):
                    acc = 0
                    for i in range(n):
                        acc += i
                    if acc >= 10:
                        landed = 1          # target line
                    return acc

            def entry(n):
                return Codec().step(n)
        """)
        # target = the `landed = 1` line (8): reached when acc>=10, i.e. n large enough.
        for n in (0, 1, 4, 5, 6, 50, 100):
            probed, full = self._probe_vs_trace(p, "entry", 8, {"n": n})
            self.assertIsNotNone(probed, f"probe should localize the method line (n={n})")
            self.assertEqual(probed, full, f"probe/trace disagree at n={n}: {probed} vs {full}")

    def test_probe_returns_none_when_it_cannot_localize(self) -> None:
        from tharkun.driver import _target_reached_probe

        p = _write("def f(a):\n    if a > 0:\n        return 1\n    return 0\n")
        # A line number that is in no code object → probe abstains (caller uses full trace).
        got = _target_reached_probe(p, "f", {"a": 1}, 9999, p, frozenset({os.path.realpath(p)}))
        self.assertIsNone(got)


class TestModuleStateIsNotAFalseReach(unittest.TestCase):
    """A REACHED must survive a fresh single call — not depend on accumulated state.

    The driver runs the target in-process against `sys.modules`-cached modules, so a
    guard on a module-level counter accumulates across traces and can read as reached. A
    fresh-state re-verify at the REACHED boundary catches that and reports a plateau.
    """

    def test_a_guard_on_a_module_counter_is_a_plateau_not_a_reach(self) -> None:
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "shared_counter.py"), "w") as f:
            f.write("COUNT = [0]\n")
        p = os.path.join(d, "entry_mod.py")
        with open(p, "w") as f:
            f.write(textwrap.dedent("""
                import shared_counter
                def entry(x):
                    shared_counter.COUNT[0] += 1
                    if shared_counter.COUNT[0] >= 5:
                        y = 1
                        return y
                    return 0
            """))
        import sys as _sys
        _sys.path.insert(0, d)
        try:
            r = drive_to_line(p, "entry", 6, {"x": 0}, max_rounds=10)
        finally:
            if d in _sys.path:
                _sys.path.remove(d)
        self.assertEqual(r.status, PLATEAU)
        self.assertFalse(r.reached)
        self.assertIn("module-level", r.diagnostics)

    def test_a_pure_parameter_driven_target_still_reaches(self) -> None:
        # The fresh re-verify must AGREE for a pure function — it reaches fresh too.
        p = _write("""
            def f(a):
                if a > 100:
                    return "hit"
                return "miss"
        """)
        r = drive_to_line(p, "f", 4, {"a": 0}, max_rounds=6)
        self.assertEqual(r.status, REACHED)
        self.assertGreater(r.inputs["a"], 100)


WITNESS_B = """\
    def f(a, b):
        if a > 100:
            a = b
            if a == 7:
                return "TARGET"
        return "none"
"""


@requires_hypothesis
class TestWitnessBReaches(unittest.TestCase):
    """The reachable line is reached — and the false-dead verdict is gone."""

    def test_the_line_past_a_reassignment_is_reached_not_declared_dead(self) -> None:
        p = _write(WITNESS_B)
        r = drive_to_line(p, "f", 5, {"a": 0, "b": 0}, max_rounds=12)
        self.assertEqual(r.status, REACHED, r.diagnostics)
        self.assertTrue(r.reached)
        # And the inputs actually execute the target.
        import importlib.util

        spec = importlib.util.spec_from_file_location("wb_mod", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.f(**r.inputs), "TARGET")
        self.assertGreater(r.inputs["a"], 100)          # outer guard still satisfied
        self.assertEqual(r.inputs["b"], 7)              # reassigned-through value found

    def test_it_is_never_reported_as_dead_code(self) -> None:
        p = _write(WITNESS_B)
        r = drive_to_line(p, "f", 5, {"a": 0, "b": 0}, max_rounds=12)
        self.assertNotIn("dead code", r.diagnostics)
        self.assertNotIn("provably unreachable", r.diagnostics)




class TestSeveredGateHasASafetyNet(unittest.TestCase):
    """The severed shortcut must never lose a reach that a per-guard candidate can win.

    When the guard is NOT nested under the reassignment, setting the entry parameter
    reroutes control AROUND it — so the per-guard candidate is valid after all. The
    reach-search recovers that, but only under ``heavy``; with ``heavy=False`` the gate
    must not fire, so the cheap tier still gets its shot.
    """

    SRC = """\
    def f(a, b):
        if a > 100:
            a = b
        if a == 7:
            return "TARGET"
        return "none"
    """

    def test_reaches_via_the_cheap_tier_even_with_heavy_disabled(self) -> None:
        p = _write(self.SRC)
        r = drive_to_line(p, "f", 5, {"a": 200, "b": 0}, max_rounds=8, heavy=False)
        self.assertEqual(r.status, REACHED, r.diagnostics)
        import importlib.util

        spec = importlib.util.spec_from_file_location("sev_mod", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.f(**r.inputs), "TARGET")


class TestBindingIndex(unittest.TestCase):
    """Which lines rebind which names — the raw material of the staleness filter."""

    def _idx(self, src: str):
        return _binding_index(ast.parse(textwrap.dedent(src)))

    def test_plain_assignment(self) -> None:
        by_line, _loops, _self = self._idx("def f(a, b):\n    a = b\n")
        self.assertEqual(by_line.get(2), frozenset({"a"}))

    def test_augassign_and_self_reference_are_marked(self) -> None:
        by_line, _loops, self_rhs = self._idx("def f(a):\n    a += 1\n")
        self.assertEqual(by_line.get(2), frozenset({"a"}))
        self.assertIn("a", self_rhs.get(2, frozenset()))     # a += 1 reads a: not severing

    def test_a_reassignment_from_another_var_does_not_self_reference(self) -> None:
        _by, _loops, self_rhs = self._idx("def f(a, b):\n    a = b\n")
        self.assertNotIn("a", self_rhs.get(2, frozenset()))  # a = b severs a from entry

    def test_self_reference_through_a_call_is_marked(self) -> None:
        _by, _loops, self_rhs = self._idx("def f(a):\n    a = g(a)\n")
        self.assertIn("a", self_rhs.get(2, frozenset()))

    def test_subscript_and_attribute_targets_kill_the_base(self) -> None:
        by_line, _loops, _self = self._idx("def f(a, c):\n    a[0] = 1\n    c.x = 2\n")
        self.assertEqual(by_line.get(2), frozenset({"a"}))
        self.assertEqual(by_line.get(3), frozenset({"c"}))

    def test_for_target_and_loop_span(self) -> None:
        by_line, loops, _self = self._idx("def f(xs):\n    for a in xs:\n        pass\n")
        self.assertEqual(by_line.get(2), frozenset({"a"}))
        self.assertTrue(any(lo == 2 for lo, _hi in loops))

    def test_with_as_and_walrus(self) -> None:
        by_line, _loops, _self = self._idx(
            "def f(ctx):\n    with ctx as a:\n        b = (c := 3)\n")
        self.assertIn("a", by_line.get(2, frozenset()))
        self.assertIn("c", by_line.get(3, frozenset()))      # walrus binds c

    def test_a_comprehension_target_is_its_own_scope_not_recorded(self) -> None:
        # `[a*2 for a in items]`: the comprehension `a` is not the parameter `a`.
        by_line, _loops, _self = self._idx(
            "def f(a, items):\n    xs = [a * 2 for a in items]\n")
        self.assertEqual(by_line.get(2), frozenset({"xs"}))  # only xs, never a

    def test_a_nested_scope_binding_is_recorded_over_kill_for_soundness(self) -> None:
        # A binding in a nested function (or a class method, or a nonlocal write) IS
        # recorded. Excluding deeper scopes was UNSOUND: it made the reassignment inside a
        # class method invisible and revived the Witness B false-dead one scope down.
        # Recording it over-approximates (may drop a fact the nested local didn't really
        # invalidate) — sound, never a false-dead.
        by_line, _loops, _self = self._idx(
            "def f(a):\n    def helper():\n        a = 1\n        return a\n    return helper()\n")
        self.assertEqual(by_line.get(3), frozenset({"a"}))

    def test_a_class_method_body_binding_is_recorded(self) -> None:
        # The regression that motivated the revert: a method is a def under a class, so a
        # scope-depth gate skipped its body — reviving Witness B inside every method.
        by_line, _loops, _self = self._idx(
            "class C:\n    def m(self, a, b):\n        a = b\n        return a\n")
        self.assertEqual(by_line.get(3), frozenset({"a"}))

    def test_a_walrus_inside_a_comprehension_still_binds_outward(self) -> None:
        by_line, _loops, _self = self._idx(
            "def f(items):\n    ys = [y for x in items if (y := x + 1)]\n")
        self.assertIn("y", by_line.get(2, frozenset()))      # walrus leaks to the enclosing scope


class TestReassignmentInsideAMethodOrSibling(unittest.TestCase):
    """False-deads the adversarial audit surfaced after the first review round."""

    def test_a_reassignment_in_a_class_method_is_not_declared_dead(self) -> None:
        # The blocker sits in a method (a def under a class); the method reassigns a. A
        # scope-depth gate on _binding_index skipped method bodies, reviving Witness B.
        p = _write('''
            class C:
                def m(self, a, b):
                    if a > 100:
                        pass
                    a = b
                    if a == 7:
                        return "TARGET"
                    return "miss"
            def entry(a, b):
                return C().m(a, b)
        ''')
        r = drive_to_line(p, "entry", 8, {"a": 0, "b": 0}, max_rounds=12)
        self.assertNotIn("dead code", r.diagnostics)
        self.assertNotIn("provably unreachable", r.diagnostics)

    def test_a_satisfied_sibling_guard_is_not_asserted_as_a_must(self) -> None:
        # `if a > 100` is a SIBLING of the blocker `if a == 7`, not an encloser. A
        # different input skips it and still reaches the blocker, so asserting it certifies
        # a reachable line dead.
        p = _write('''
            def f(a, b):
                a = b
                if a > 100:
                    pass
                if a == 7:
                    return "TARGET"
                return "miss"
        ''')
        r = drive_to_line(p, "f", 7, {"a": 0, "b": 0}, max_rounds=10)   # line 7 = return body
        self.assertEqual(r.status, REACHED, r.diagnostics)
        self.assertNotIn("dead code", r.diagnostics)


class TestByteInputSynthesisIsVerified(unittest.TestCase):
    """Byte-input value generation must not emit a `bytes` that fails the guard.

    The synthesis merges every comparison atom across the guard into one last-write-wins
    buffer (ignoring and/or) and models neither signed decode nor negative targets, so a
    produced value need not satisfy the guard. A single-variable byte guard is now verified
    against the full guard, so an unsatisfiable one no longer returns a `direct` false-SAT.
    """

    def _solve(self, guard: str):
        from tharkun.solve.condition import solve_condition

        src = f"def f(data):\n    if {guard}:\n        return True\n    return False\n"
        p = _write(src)
        return solve_condition(p, 2)

    def _guard_holds(self, guard: str, value: bytes) -> bool:
        ns: dict = {"int": int, "len": len}
        exec(f"def _g(data):\n    return bool({guard})", ns)
        try:
            return bool(ns["_g"](value))
        except Exception:
            return False

    def test_a_valid_byte_guard_still_synthesizes_a_satisfying_value(self) -> None:
        for guard in ("data[0] == 48",
                      'int.from_bytes(data[0:2], "big") == 258',
                      'int.from_bytes(data[0:4], "little") == 5'):
            sol = self._solve(guard)
            self.assertEqual(sol.strategy, "direct", guard)
            v = sol.satisfying_inputs[0]["data"]
            self.assertTrue(self._guard_holds(guard, v), f"{guard} -> {v!r}")

    def test_an_unsatisfiable_byte_guard_is_not_returned_as_direct(self) -> None:
        # None of these has a satisfying bytes value; the synthesis must not claim one at
        # `direct` confidence.
        for guard in ('data[0] == 1 and int.from_bytes(data[0:2], "big") == 2',
                      "data[0] == 5 and data[0] != 5",
                      'int.from_bytes(data[0:2], "big", signed=True) == 40000'):
            sol = self._solve(guard)
            if sol.strategy == "direct" and sol.satisfying_inputs:
                v = sol.satisfying_inputs[0].get("data")
                self.assertTrue(self._guard_holds(guard, v),
                                f"direct false-SAT: {guard} -> {v!r}")


class TestCrossFunctionClausesDoNotPoison(unittest.TestCase):
    """A same-named variable in another function must not force a false dead-code verdict.

    `_satisfied_clauses` ranges over the whole file; a helper's `len(x) > 9` on an
    unrelated `x` must not be admitted into a blocker on `x == "ab"` in another function.
    """

    def test_a_helpers_len_guard_does_not_kill_an_unrelated_guard(self) -> None:
        p = _write('''
            def helper(x):
                if len(x) > 9:
                    return x
                return None

            def f(x):
                if isinstance(x, str):
                    helper("aaaaaaaaaaaa")
                    if x == "ab":
                        return "TARGET"
                return "none"
        ''')
        r = drive_to_line(p, "f", 10, {"x": 0}, max_rounds=8)
        self.assertNotIn("dead code", r.diagnostics)
        self.assertNotIn("provably unreachable", r.diagnostics)


class TestMethodMutationIsAKill(unittest.TestCase):
    """A bare ``a.append(b)`` mutates its receiver, so facts about ``a`` become stale."""

    def test_a_method_call_statement_kills_its_receiver(self) -> None:
        by_line, _loops, _self = _binding_index(
            ast.parse("def f(a, b):\n    a.append(b)\n"))
        self.assertEqual(by_line.get(2), frozenset({"a"}))

    def test_an_attribute_chain_kills_the_base(self) -> None:
        by_line, _loops, _self = _binding_index(
            ast.parse("def f(a):\n    a.data.sort()\n"))
        self.assertEqual(by_line.get(2), frozenset({"a"}))

    def test_a_module_function_call_kills_nothing(self) -> None:
        # print(x) / len(x): the func is a bare Name, not an attribute — no receiver.
        by_line, _loops, _self = _binding_index(
            ast.parse("def f(a):\n    print(a)\n"))
        self.assertEqual(by_line.get(2), None)

    def test_a_mutation_past_a_length_guard_is_not_declared_dead(self) -> None:
        # len(a) == 0 committed, then a.append(b): the length fact is stale at a[0] == 7.
        # Reachable via f([], 7); the driver must not prove it dead.
        p = _write("""
            def f(a, b):
                if len(a) == 0:
                    a.append(b)
                    if a[0] == 7:
                        return "TARGET"
                return "none"
        """)
        r = drive_to_line(p, "f", 6, {"a": [1], "b": 0}, max_rounds=8)
        self.assertNotIn("dead code", r.diagnostics)
        self.assertNotIn("provably unreachable", r.diagnostics)


class TestBackEdgeReassignmentInALoop(unittest.TestCase):
    """A reassignment on a loop's back edge invalidates a fact from before the loop."""

    @pytest.mark.wave2
    @requires_hypothesis
    def test_a_line_past_a_loop_back_edge_reassignment_is_reached(self) -> None:
        # (a > 100) is established outside the loop; `a = b` on the back edge means the
        # second iteration's `a == 5` needs b == 5, not a stale a > 100. f(101, 5) reaches.
        p = _write("""
            def f(a, b):
                if a > 100:
                    for _i in range(2):
                        if a == 5:
                            return "TARGET"
                        a = b
                return "none"
        """)
        from tharkun import driver

        original_search = driver._reach_by_search

        def longer_search(*args, **kwargs):
            kwargs["max_examples"] = 3000
            return original_search(*args, **kwargs)

        # Keep the real search and replay, with room for the background run.
        with patch.object(driver, "_reach_by_search", side_effect=longer_search):
            r = drive_to_line(p, "f", 6, {"a": 0, "b": 0}, max_rounds=32)
        self.assertEqual(r.status, REACHED, r.diagnostics)
        self.assertNotIn("dead code", r.diagnostics)

    def test_it_is_never_declared_dead_even_without_the_search(self) -> None:
        p = _write("""
            def f(a, b):
                if a > 100:
                    for _i in range(2):
                        if a == 5:
                            return "TARGET"
                        a = b
                return "none"
        """)
        r = drive_to_line(p, "f", 6, {"a": 0, "b": 0}, max_rounds=10, heavy=False)
        self.assertNotIn("dead code", r.diagnostics)     # honest plateau, not a false proof
        self.assertNotIn("provably unreachable", r.diagnostics)


class TestKilledBetween(unittest.TestCase):
    """The window rule: an EXECUTED reassignment in ``(min, max]`` invalidates."""

    def test_a_reassignment_strictly_between_the_points_kills(self) -> None:
        by_line = {3: frozenset({"a"})}
        self.assertTrue(_killed_between(by_line, [], {2, 3, 4}, 2, 4, {"a"}))

    def test_an_unexecuted_reassignment_does_not_kill(self) -> None:
        by_line = {3: frozenset({"a"})}
        self.assertFalse(_killed_between(by_line, [], {2, 4}, 2, 4, {"a"}))  # line 3 not run

    def test_a_reassignment_of_another_name_does_not_kill(self) -> None:
        by_line = {3: frozenset({"z"})}
        self.assertFalse(_killed_between(by_line, [], {2, 3, 4}, 2, 4, {"a"}))

    def test_the_low_endpoint_is_open(self) -> None:
        # A reassignment AT the establishing line is not "between" — the clause is about
        # the value after that line.
        by_line = {2: frozenset({"a"})}
        self.assertFalse(_killed_between(by_line, [], {2, 3, 4}, 2, 4, {"a"}))

    def test_a_loop_widens_the_window_to_catch_a_back_edge_reassignment(self) -> None:
        # Both endpoints inside one loop (lines 5..9): a reassignment at line 8, which by
        # raw line order sits AFTER the blocker at line 6, is still caught.
        by_line = {8: frozenset({"a"})}
        loops = [(5, 9)]
        self.assertTrue(_killed_between(by_line, loops, {5, 6, 7, 8}, 7, 6, {"a"}))

    def test_widening_catches_an_establish_outside_blocker_inside_shape(self) -> None:
        # Origin at line 2 (outside), blocker at line 4 (inside loop 3..6), kill at line 6.
        # The raw (2, 4] window misses line 6; the loop containing the blocker extends it.
        by_line = {6: frozenset({"a"})}
        loops = [(3, 6)]
        self.assertTrue(_killed_between(by_line, loops, {2, 3, 4, 6}, 2, 4, {"a"}))


class TestClauseStaleness(unittest.TestCase):
    """Cross-file clauses are kept (indeterminate); same-file ones are decided."""

    def test_a_same_file_clause_past_a_reassignment_is_stale(self) -> None:
        by_line = {3: frozenset({"a"})}
        c = Clause("(a > 100)", Origin("/m.py", 2, polarity="then"))
        self.assertTrue(_clause_is_stale(c, 4, "/m.py", {2, 3, 4}, by_line, []))

    def test_a_clause_from_another_file_is_kept(self) -> None:
        by_line = {3: frozenset({"a"})}
        c = Clause("(a > 100)", Origin("/other.py", 2, polarity="then"))
        self.assertFalse(_clause_is_stale(c, 4, "/m.py", {2, 3, 4}, by_line, []))

    def test_a_clause_with_no_origin_path_is_kept(self) -> None:
        by_line = {3: frozenset({"a"})}
        c = Clause("(a > 100)", Origin("", 2))
        self.assertFalse(_clause_is_stale(c, 4, "/m.py", {2, 3, 4}, by_line, []))


class TestVerificationIsNotRepeated(unittest.TestCase):
    """A candidate that failed verification is not re-executed for the same inputs.

    Answers the user's ask directly: once ``a == 6`` is shown not to reach under a
    committed ``a == 5``, the driver does not keep paying to re-verify it — two cheap-tier
    lanes propose the identical doomed candidate in one round, and only one runs it.
    """

    def test_the_same_candidate_is_verified_at_most_once(self) -> None:
        import tharkun.driver as drv

        p = _write("""
            def f(a):
                if a == 5:
                    if a == 6:
                        return "T"
                return "n"
        """)
        seen: list[tuple] = []
        orig = drv._flip_verifies

        def counting(source_path, func_name, blocker, inputs, candidate, guard_path=None):
            merged = {**inputs, **candidate}
            seen.append((blocker.branch_line, blocker.desired_branch,
                         tuple(sorted((k, repr(v)) for k, v in merged.items()))))
            return orig(source_path, func_name, blocker, inputs, candidate, guard_path)

        drv._flip_verifies = counting
        try:
            # target is line 5 (a leading blank line shifts every line by one): the deeper
            # guard demands a == 6, doomed under the committed a == 5, so two cheap-tier
            # lanes propose the identical candidate — the memo must run it only once.
            drive_to_line(p, "f", 5, {"a": 0}, max_rounds=8)
        finally:
            drv._flip_verifies = orig

        self.assertEqual(len(seen), len(set(seen)),
                         f"a candidate was verified more than once: {seen}")
        self.assertTrue(seen, "the doomed candidate was never verified at all")


if __name__ == "__main__":
    unittest.main()
