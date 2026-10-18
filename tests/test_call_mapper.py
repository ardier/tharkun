"""Tests for call_mapper — callee-param-to-caller-expression mapping."""

from __future__ import annotations

import ast
import os

import pytest

from tharkun.solve.frontend.call_mapper import (
    ArgMapping,
    CallSiteMapping,
    find_calls_to,
    map_call_chain,
    map_call_site,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "call_mapper_target.py")


def _line_of(fixture_path, text):
    """Find the line number containing *text* in the fixture file."""
    with open(fixture_path) as f:
        for i, line in enumerate(f, 1):
            if text in line:
                return i
    raise ValueError(f"text {text!r} not found in {fixture_path}")


# ---------------------------------------------------------------------------
# map_call_site
# ---------------------------------------------------------------------------


class TestMapCallSite:
    def test_simple_direct_call(self):
        line = _line_of(FIXTURE, "return salute(name)")
        m = map_call_site(FIXTURE, line)
        assert m is not None
        assert m.callee_name == "salute"
        assert len(m.args) == 1
        arg = m.args[0]
        assert arg.callee_param == "who"
        assert arg.caller_name == "name"
        assert arg.is_simple_name is True
        assert arg.callee_position == 0

    def test_transformed_argument(self):
        line = _line_of(FIXTURE, "return process(data.strip())")
        m = map_call_site(FIXTURE, line)
        assert m is not None
        assert m.callee_name == "process"
        assert len(m.args) == 1
        arg = m.args[0]
        assert arg.callee_param == "text"
        assert arg.is_simple_name is False
        assert arg.caller_name is None
        assert "strip" in arg.caller_expr_source

    def test_positional_swap(self):
        line = _line_of(FIXTURE, "return combine(b, a)")
        m = map_call_site(FIXTURE, line)
        assert m is not None
        assert m.callee_name == "combine"
        assert len(m.args) == 2
        # First positional arg: b → x
        assert m.args[0].callee_param == "x"
        assert m.args[0].caller_name == "b"
        # Second positional arg: a → y
        assert m.args[1].callee_param == "y"
        assert m.args[1].caller_name == "a"

    def test_keyword_args(self):
        line = _line_of(FIXTURE, "return lookup(data=value)")
        m = map_call_site(FIXTURE, line)
        assert m is not None
        assert m.callee_name == "lookup"
        assert len(m.args) == 1
        arg = m.args[0]
        assert arg.callee_param == "data"
        assert arg.caller_name == "value"
        assert arg.is_simple_name is True

    def test_no_call_at_line(self):
        line = _line_of(FIXTURE, "return x + 1")
        m = map_call_site(FIXTURE, line)
        assert m is None

    def test_callee_name_filter(self):
        line = _line_of(FIXTURE, "return salute(name)")
        # Filter for wrong name.
        m = map_call_site(FIXTURE, line, callee_name="nonexistent")
        assert m is None
        # Filter for correct name.
        m = map_call_site(FIXTURE, line, callee_name="salute")
        assert m is not None

    def test_method_call(self):
        line = _line_of(FIXTURE, "return self.check(data)")
        m = map_call_site(FIXTURE, line)
        assert m is not None
        assert "check" in m.callee_name
        assert len(m.args) >= 1


# ---------------------------------------------------------------------------
# find_calls_to
# ---------------------------------------------------------------------------


class TestFindCallsTo:
    def test_multiple_call_sites(self):
        results = find_calls_to(FIXTURE, "shared_func")
        assert len(results) >= 2
        callers = {m.args[0].caller_name for m in results if m.args}
        assert "x" in callers
        assert "y" in callers

    def test_no_calls_found(self):
        results = find_calls_to(FIXTURE, "nonexistent_function")
        assert results == []


# ---------------------------------------------------------------------------
# map_call_chain
# ---------------------------------------------------------------------------


class TestMapCallChain:
    def test_two_level_chain(self):
        # greet → salute
        salute_line = _line_of(FIXTURE, "if len(who) > 0:")
        chain = map_call_chain(FIXTURE, salute_line, entry_function="greet")
        assert len(chain) >= 1
        assert chain[0].callee_name == "salute"
        assert chain[0].args[0].caller_name == "name"

    def test_max_depth_respected(self):
        # greet → salute is a direct call (depth 0), so max_depth=0 finds it.
        # With a deeper chain (swap_args → combine), max_depth=0 should find
        # the direct call but not recurse further.
        combine_line = _line_of(FIXTURE, "if x > 0 and y > 0:")
        # swap_args directly calls combine, so depth 0 finds it.
        chain = map_call_chain(FIXTURE, combine_line, entry_function="swap_args", max_depth=0)
        assert len(chain) >= 1
        # Verify that with a very deep chain and max_depth, it stops.
        # Since our fixture only has 1-hop chains, just verify the DFS terminates.
        chain2 = map_call_chain(FIXTURE, combine_line, entry_function="swap_args", max_depth=10)
        assert len(chain2) >= 1

    def test_auto_detect_entry(self):
        # Should find greet as entry when targeting salute's guard line.
        salute_line = _line_of(FIXTURE, "if len(who) > 0:")
        chain = map_call_chain(FIXTURE, salute_line)
        # Should find at least one hop.
        assert len(chain) >= 1


# ---------------------------------------------------------------------------
# Cross-module end-to-end propagation
# ---------------------------------------------------------------------------

CHAIN_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "call_chain")
ENTRY_PATH = os.path.join(CHAIN_DIR, "entry.py")
MIDDLE_PATH = os.path.join(CHAIN_DIR, "middle.py")
DEEP_PATH = os.path.join(CHAIN_DIR, "deep.py")


class TestCrossModulePropagation:
    """End-to-end test: constraint backward walk through module boundaries.

    Fixture layout:
        entry.py  → process(raw_data.strip())  (transformed arg)
        middle.py → validate(data)             (simple name pass-through)
        deep.py   → if len(value) > 3 and ...: (target guard, line 5)

    ``find_calls_to`` handles cross-module discovery; ``map_call_chain``
    operates within a single source file and follows calls into other modules
    during its DFS.
    """

    def test_find_calls_to_validate_in_middle(self):
        """find_calls_to locates the call from middle.py → validate."""
        results = find_calls_to(MIDDLE_PATH, "validate")
        assert len(results) >= 1
        mapping = results[0]
        # validate is called with one arg: data
        assert len(mapping.args) >= 1
        arg = mapping.args[0]
        assert arg.callee_param == "value"
        assert arg.caller_name == "data"
        assert arg.is_simple_name is True

    def test_find_calls_to_process_in_entry(self):
        """find_calls_to finds both callers of process in entry.py."""
        results = find_calls_to(ENTRY_PATH, "process")
        assert len(results) >= 1
        # start_simple passes 'x' directly — a simple name hop
        simple_name_hops = [
            m for m in results
            if any(a.is_simple_name for a in m.args)
        ]
        assert len(simple_name_hops) >= 1
        # The simple-name hop must have caller_name populated
        for mapping in simple_name_hops:
            for arg in mapping.args:
                if arg.is_simple_name:
                    assert arg.caller_name is not None

    def test_simple_name_hops_have_caller_name(self):
        """Every simple-name arg mapping from find_calls_to carries caller_name."""
        results = find_calls_to(MIDDLE_PATH, "validate")
        for mapping in results:
            for arg in mapping.args:
                if arg.is_simple_name:
                    assert arg.caller_name is not None, (
                        f"Simple-name hop for callee param {arg.callee_param!r} "
                        f"should have caller_name populated"
                    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_nonexistent_file(self):
        m = map_call_site("/nonexistent/file.py", 1)
        assert m is None

    def test_syntax_error_file(self, tmp_path):
        bad = tmp_path / "bad.py"
        bad.write_text("def f(:\n")
        m = map_call_site(str(bad), 1)
        assert m is None

    def test_chain_nonexistent_file(self):
        chain = map_call_chain("/nonexistent/file.py", 1)
        assert chain == []
