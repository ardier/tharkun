"""Test wrapper for the constraint -> Hypothesis toy demo.

Keeps ``demos/constraint_to_hypothesis.py`` honest: every demonstrated value
must be produced by tharkun's Hypothesis adapter (not condition_solver's static
pick) and must drive the guard's true branch when executed through the real
fixture function. Skips cleanly when the optional ``hypothesis`` extra is
absent.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import unittest

from tharkun import solve_condition
from tharkun.hypothesis_adapter import hypothesis_available

try:
    import hypothesis
except ImportError:  # pragma: no cover - guarded by hypothesis_available() below
    hypothesis = None


@contextlib.contextmanager
def _no_deadline():
    """[internal ref]: scope a `deadline=None` Hypothesis profile to one call.

    hypothesis_adapter.hypothesis_solve_branch()'s local settings(...) object
    never sets `deadline`, so it inherits the ambient default (200ms per
    example). Under host CPU contention (the full suite runs at -n48..64) a
    single one of up to 2000 predicate calls -- each loading the fixture
    module, tracing a real execution and eval()'ing the guard -- can exceed
    that per-example deadline. hypothesis.find() then raises
    DeadlineExceeded, which the adapter's broad `except Exception: return
    None` silently swallows as "no value found". That is a search-timing
    artifact of the harness, not a defect in the adapter's search or in what
    these tests check: the already-present
    suppress_health_check=[HealthCheck.too_slow] does NOT cover
    DeadlineExceeded -- that health check and the per-example deadline are
    two different Hypothesis mechanisms.

    This version of Hypothesis's `settings` object is not itself a context
    manager, so scope the change through the profile registry instead of a
    global, permanent mutation: swap in a deadline=None profile, always
    restoring whichever profile was active before.
    """
    name = "tharkun-1084-no-deadline"
    previous = hypothesis.settings.get_current_profile_name()
    hypothesis.settings.register_profile(
        name, parent=hypothesis.settings.get_profile(previous),
        deadline=None, suppress_health_check=[hypothesis.HealthCheck.too_slow],
    )
    hypothesis.settings.load_profile(name)
    try:
        yield
    finally:
        hypothesis.settings.load_profile(previous)


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO_PATH = os.path.join(_REPO_ROOT, "demos", "constraint_to_hypothesis.py")
FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "toy_constraint_hypothesis.py"
)


def _load_demo():
    spec = importlib.util.spec_from_file_location("constraint_to_hypothesis", _DEMO_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's own @dataclass (Case) can resolve its
    # annotations via sys.modules (PEP 563 string annotations need this).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestToyConstraintHypothesisDemo(unittest.TestCase):
    def test_demo_file_exists(self) -> None:
        self.assertTrue(os.path.isfile(_DEMO_PATH))
        self.assertTrue(os.path.isfile(FIXTURE))

    def test_condition_solver_tracks_constraints(self) -> None:
        # condition_solver must *track* the int range + type for classify's guard.
        sol = solve_condition(FIXTURE, 24)
        names = {v.name for v in sol.variables}
        self.assertIn("x", names)
        xc = next(v for v in sol.variables if v.name == "x")
        self.assertEqual(xc.inferred_type, "int")
        self.assertTrue(xc.concrete_values, "expected tracked boundary values")

    def test_static_pick_does_not_crack_modulo(self) -> None:
        # Honesty check: condition_solver's static value must NOT satisfy the
        # guard, so Hypothesis is genuinely the producing stage.
        mod = _load_demo()
        classify = mod._load_fixture_func("classify")
        sol = solve_condition(FIXTURE, 24)
        self.assertTrue(sol.satisfying_inputs)
        static_x = sol.satisfying_inputs[0]["x"]
        self.assertEqual(classify(static_x), "miss")

    @unittest.skipUnless(hypothesis_available(), "hypothesis not installed")
    def test_demo_runs_green(self) -> None:
        mod = _load_demo()
        with _no_deadline():
            rc = mod.main()
        self.assertEqual(rc, 0, "demo should exit 0 (all values produced + verified)")

    @unittest.skipUnless(hypothesis_available(), "hypothesis not installed")
    def test_each_case_value_comes_from_hypothesis_and_verifies(self) -> None:
        mod = _load_demo()
        for case in mod.CASES:
            with self.subTest(func=case.func_name):
                func = mod._load_fixture_func(case.func_name)
                sol = solve_condition(FIXTURE, case.guard_line)
                narrowed = mod._narrowed_from_tracked(sol, case.param)
                # [internal ref]: same deadline fix as test_demo_runs_green above, plus
                # derandom=True (a parameter hypothesis_solve_branch already exposes)
                # so this specific search is fully reproducible run to run -- it no
                # longer depends on hitting a satisfying example by random luck within
                # the 2000-example budget.
                with _no_deadline():
                    produced = mod.hypothesis_solve_branch(
                        FIXTURE, case.body_line, case.func_name,
                        max_examples=2000, narrowed=narrowed, derandom=True,
                    )
                self.assertIsNotNone(produced, "Hypothesis should produce a value")
                self.assertIn(case.param, produced)
                value = produced[case.param]
                # Verify through the real function.
                self.assertEqual(func(value), case.expected_true)
                # Verify the value came from Hypothesis, not the static pick:
                # the static pick is known to fail the guard.
                static = (sol.satisfying_inputs[0].get(case.param)
                          if sol.satisfying_inputs else None)
                if static is not None:
                    self.assertNotEqual(
                        func(static), case.expected_true,
                        "static pick unexpectedly satisfied the guard",
                    )


if __name__ == "__main__":
    unittest.main()
