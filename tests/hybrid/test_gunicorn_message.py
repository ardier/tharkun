"""Seedless static solves of gunicorn's HTTP request-parsing regex guards.

The upstream OSS-Fuzz harness (fuzz_util.py -> util functions) never calls
gunicorn/http/message.py, so its request-line / header guards sit at 0% coverage —
and every one is regex-gated (TOKEN_RE.fullmatch, VERSION_RE.fullmatch). These are
exactly the sparse guards a random ConsumeString fuzzer cannot land. tharkun solves
them statically, no seed, no execution — producing a witness that actually satisfies
the real compiled pattern. No atheris required; this is pure condition_solver.
"""
from __future__ import annotations

import os
import sys
import unittest

from tharkun.solve.condition import solve_condition

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "gunicorn_message.py")


def _line(needle: str) -> int:
    with open(FIX) as f:
        return next(i for i, l in enumerate(f, 1) if needle in l)


class TestGunicornMessageRegexGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, os.path.dirname(FIX))
        import gunicorn_message as gm  # noqa: F401
        cls.gm = gm

    def _solve(self, needle: str, var: str, branch: str):
        sol = solve_condition(FIX, _line(needle), desired_branch=branch)
        self.assertTrue(sol.satisfying_inputs, f"no solution for {needle}")
        return sol.satisfying_inputs[0].get(var), sol.strategy

    def test_method_token_solves_to_a_valid_token(self) -> None:
        # else-branch of `if not TOKEN_RE.fullmatch(method)` = the non-raise path = a match.
        v, strat = self._solve("TOKEN_RE.fullmatch(method)", "method", "else")
        self.assertIsInstance(v, str)
        self.assertIsNotNone(self.gm.TOKEN_RE.fullmatch(v), v)
        self.gm.parse_request_method(v)   # does not raise

    def test_header_name_solves_to_a_valid_token(self) -> None:
        v, strat = self._solve("TOKEN_RE.fullmatch(name)", "name", "else")
        self.assertIsInstance(v, str)
        self.assertIsNotNone(self.gm.TOKEN_RE.fullmatch(v), v)
        self.gm.parse_header_name(v)      # does not raise

    def test_http_version_solves_to_a_valid_version(self) -> None:
        v, strat = self._solve("VERSION_RE.fullmatch(version)", "version", "then")
        self.assertIsInstance(v, str)
        self.assertIsNotNone(self.gm.VERSION_RE.fullmatch(v), v)
        self.assertEqual(self.gm.parse_http_version(v), v)   # returns, does not raise


if __name__ == "__main__":
    unittest.main()
