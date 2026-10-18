"""Hypothesis property-based search adapter (step 9 of the solver chain).

Last-resort solver: uses ``hypothesis.find(strategy, predicate)`` where the
predicate executes the target function and checks (via ``trace_api``) whether
the target branch line was hit.

Hypothesis is an optional dependency — this module degrades gracefully if
not installed.

Usage::

    from tharkun.hypothesis_adapter import hypothesis_solve_branch

    result = hypothesis_solve_branch("module.py", 42, "my_function")
    if result is not None:
        print(result)  # {"x": 5}
"""

from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
from typing import Any


def hypothesis_available() -> bool:
    """Check if ``hypothesis`` is importable."""
    try:
        importlib.import_module("hypothesis")
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Type → Hypothesis strategy mapping
# ---------------------------------------------------------------------------


def _strategy_for_type(type_name: str) -> Any:
    """Map an inferred type name to a Hypothesis strategy.

    Returns None if no strategy can be constructed.
    """
    from tharkun.solve.config import config
    from tharkun.strategies import strategy_for_type

    lo, hi = config.hypothesis_int_range
    strat = strategy_for_type(type_name, int_range=(lo, hi), max_size=50)
    if strat is not None:
        # [internal ref] step 2: a TESTED contract's evidence that an empty/zero value of
        # this type is worth sampling (see solve/contracts/adapters/draws.py) --
        # an ADDITION to the strategy's domain, never a narrowing of it.
        from hypothesis import strategies as st

        from tharkun.solve.contracts.adapters.draws import boundary_value_for_type

        boundary = boundary_value_for_type(type_name)
        if boundary is not None:
            strat = st.one_of(st.just(boundary), strat)
    return strat


def _infer_param_types(
    source_path: str,
    callable_id: str,
    lineno: int | None = None,
) -> dict[str, str]:
    """Extract parameter names and inferred types.

    Uses three sources (in priority order):
    1. Tharkun's condition solver type inference (if lineno provided)
    2. Cross-function type propagation
    3. Function signature annotations
    4. Fallback to "Any"
    """
    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {}

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == callable_id:
                func_def = node
                break
    if func_def is None:
        return {}

    result: dict[str, str] = {}

    # Start with annotations
    for arg in func_def.args.args:
        name = arg.arg
        if name in ("self", "cls"):
            continue
        if arg.annotation:
            try:
                ann = ast.unparse(arg.annotation)
            except Exception:
                ann = None
            result[name] = ann if ann else "Any"
        else:
            result[name] = "Any"

    # Try type propagation for unannotated params
    try:
        from tharkun.solve.frontend.type_propagation import propagate_types
        propagated = propagate_types(source_path, callable_id)
        for name, type_str in propagated.items():
            if name in result and result[name] == "Any":
                result[name] = type_str
    except Exception:
        pass

    # Try tharkun's guard-level type inference if lineno is provided
    if lineno:
        try:
            from tharkun.solve.condition import infer_var_type, _find_branch_at_line
            test_expr = _find_branch_at_line(tree, lineno)
            if test_expr:
                for name in result:
                    if result[name] == "Any":
                        inferred, source = infer_var_type(name, test_expr, rp, lineno)
                        if inferred != "Any":
                            result[name] = inferred
        except Exception:
            pass

    return result


def hypothesis_solve_branch(
    source_path: str,
    lineno: int,
    callable_id: str,
    *,
    max_examples: int = 200,
    timeout_s: float = 5.0,
    narrowed: Any = None,  # NarrowedConstraint from minleaf
    desired_branch: str = "then",
    derandom: bool = False,
) -> dict[str, Any] | None:
    """Attempt to solve a branch using Hypothesis search.

    Builds strategies from inferred parameter types, then uses
    ``hypothesis.find()`` with a predicate that runs the function
    and checks if the target branch line was executed.

    Args:
        source_path:  Path to the Python source file.
        lineno:       Branch line number to target.
        callable_id:  Name of the enclosing function.
        max_examples: Maximum Hypothesis examples to try.
        timeout_s:    Not enforced directly. Hypothesis's own per-example deadline is
                      disabled below (``deadline=None``) rather than left at its 200 ms
                      default: on a loaded host, a slow-but-fine example trips
                      ``DeadlineExceeded`` well before ``max_examples`` is exhausted,
                      and treating that the same as an honest "not found" throws away a
                      FALSE the search would otherwise have caught ([internal ref] part 2).

    Returns:
        Dict of satisfying inputs, or ``None`` if Hypothesis is unavailable
        or cannot find inputs.
    """
    if not hypothesis_available():
        return None

    try:
        from hypothesis import find, settings, HealthCheck, Verbosity
        from hypothesis import strategies as st
        from hypothesis.errors import NoSuchExample
    except ImportError:
        return None

    rp = os.path.realpath(source_path)
    param_types = _infer_param_types(source_path, callable_id, lineno=lineno)
    if not param_types:
        return None

    # Build a composite strategy for all parameters.
    # If minleaf provided narrowed constraints, use them for tighter strategies.
    strategies: dict[str, Any] = {}
    for name, type_name in param_types.items():
        strat = None
        # Use narrowed constraints for tighter bounds
        if narrowed is not None:
            bounds = getattr(narrowed, 'var_bounds', {}).get(name)
            narrow_type = getattr(narrowed, 'var_types', {}).get(name)
            if bounds and narrow_type in ('int', 'float'):
                lo, hi = bounds
                if narrow_type == 'int':
                    strat = st.integers(min_value=int(lo), max_value=int(hi))
                else:
                    strat = st.floats(min_value=float(lo), max_value=float(hi), allow_nan=True)
            elif narrow_type:
                strat = _strategy_for_type(narrow_type)
        if strat is None:
            strat = _strategy_for_type(type_name)
        if strat is None:
            strat = st.one_of(st.integers(), st.text(max_size=20), st.booleans())
        strategies[name] = strat

    if not strategies:
        return None

    # Build the combined strategy as a fixed_dictionaries
    combined = st.fixed_dictionaries(strategies)

    # Load the function module — prefer package import over exec
    try:
        from tharkun.branch_solver import _load_module_for_execution
        mod, module_globals = _load_module_for_execution(rp)
    except Exception:
        mod = None

    if mod is None:
        return None

    func = getattr(mod, callable_id, None)
    if func is None or not callable(func):
        return None

    # Pre-parse the guard expression for edge-level verification.
    _guard_src: str | None = None
    try:
        import ast as _ast
        _source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        _tree = _ast.parse(_source)
        from tharkun.solve.condition import _find_branch_at_line
        _guard_node = _find_branch_at_line(_tree, lineno)
        if _guard_node is not None:
            _guard_src = _ast.unparse(_guard_node)
    except Exception:
        pass

    _desired_val = desired_branch == "then"

    # Import tracer for branch verification
    from tharkun.trace_api import TracePolicy, Tracer

    def predicate(inputs: dict[str, Any]) -> bool:
        """Return True if the inputs produce the desired edge outcome.

        Evaluates the guard expression directly for exact edge verification.
        Falls back to line-hit check if guard evaluation fails.
        """
        policy = TracePolicy(paths={rp}, snapshot_lines=None)
        tracer = Tracer(policy)
        try:
            with tracer:
                func(**inputs)
        except Exception:
            pass

        # Primary: evaluate the guard against desired polarity.
        if _guard_src is not None:
            try:
                _eval_globals: dict[str, Any] = {
                    "__builtins__": {
                        "len": len, "isinstance": isinstance, "bool": bool,
                        "int": int, "str": str, "float": float,
                        "list": list, "dict": dict, "set": set, "tuple": tuple,
                        "True": True, "False": False, "None": None,
                    }
                }
                result = eval(_guard_src, _eval_globals, dict(inputs))  # noqa: S307
                return bool(result) == _desired_val
            except Exception:
                pass

        # Fallback: line-hit check (less precise).
        executed = tracer.result.executed.get(rp, set())
        return lineno in executed

    try:
        result = find(
            combined,
            predicate,
            settings=settings(
                max_examples=max_examples,
                deadline=None,
                suppress_health_check=[HealthCheck.too_slow],
                verbosity=Verbosity.quiet,
                database=None,
                derandomize=derandom,
            ),
        )
        return result
    except NoSuchExample:
        # The only outcome that legitimately means "searched and found nothing":
        # find() raises this itself once max_examples is exhausted without the
        # predicate ever returning True.  Anything else here -- DeadlineExceeded,
        # FailedHealthCheck, Flaky, a real bug in predicate() escaping the tracer's
        # own guards -- is an internal failure of the search, not a verified
        # negative, and must not be reported the same way: callers up the chain
        # (branch_solver's step-9 except, the concolic worker's per-site except,
        # the hybrid bench's campaign-level except) already treat a crashing
        # search as "unknown", never as "proved infeasible", so letting it
        # propagate here is what keeps that distinction sound.
        return None
