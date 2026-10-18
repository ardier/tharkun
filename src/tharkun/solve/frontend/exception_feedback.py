"""Exception-driven type feedback for re-solving branches.

When execution with generated inputs raises a ``TypeError``, this module
parses the error message to extract the expected type, then returns a
correction hint that the solver can use to re-generate inputs.

Usage::

    from tharkun.solve.frontend.exception_feedback import try_execute_and_correct

    correction = try_execute_and_correct(source_path, callable_id, inputs)
    if correction:
        # Re-solve with correction["param"] typed as correction["expected_type"]
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Error message patterns
# ---------------------------------------------------------------------------

# "argument of type 'int' is not iterable"
_NOT_ITERABLE = re.compile(r"argument of type '(\w+)' is not iterable")

# "expected str, got int" or "'str' expected, got 'int'"
_EXPECTED_GOT = re.compile(r"expected (\w+),? got (\w+)|'(\w+)' expected,? got '(\w+)'")

# "'int' object is not subscriptable"
_NOT_SUBSCRIPTABLE = re.compile(r"'(\w+)' object is not subscriptable")

# "'int' object has no attribute 'xxx'"
_NO_ATTRIBUTE = re.compile(r"'(\w+)' object has no attribute '(\w+)'")

# "object of type 'int' has no len()"
_NO_LEN = re.compile(r"object of type '(\w+)' has no len\(\)")


@dataclass
class TypeCorrection:
    """A correction hint from exception analysis."""
    param: str | None
    wrong_type: str
    expected_type: str
    error_message: str


def parse_type_error(error: TypeError, inputs: dict[str, Any]) -> TypeCorrection | None:
    """Parse a TypeError to extract a type correction hint.

    Args:
        error:  The TypeError that was raised.
        inputs: The inputs that caused the error.

    Returns:
        A ``TypeCorrection`` hint, or ``None`` if the error can't be parsed.
    """
    msg = str(error)

    # "expected str, got int"
    m = _EXPECTED_GOT.search(msg)
    if m:
        expected = m.group(1) or m.group(3)
        got = m.group(2) or m.group(4)
        # Try to identify which param has type `got`
        param = _identify_param(inputs, got)
        return TypeCorrection(
            param=param, wrong_type=got, expected_type=expected, error_message=msg
        )

    # "argument of type 'int' is not iterable"
    m = _NOT_ITERABLE.search(msg)
    if m:
        wrong = m.group(1)
        param = _identify_param(inputs, wrong)
        return TypeCorrection(
            param=param, wrong_type=wrong, expected_type="str", error_message=msg
        )

    # "'int' object is not subscriptable"
    m = _NOT_SUBSCRIPTABLE.search(msg)
    if m:
        wrong = m.group(1)
        param = _identify_param(inputs, wrong)
        return TypeCorrection(
            param=param, wrong_type=wrong, expected_type="list", error_message=msg
        )

    # "'int' object has no attribute 'xxx'"
    m = _NO_ATTRIBUTE.search(msg)
    if m:
        wrong = m.group(1)
        param = _identify_param(inputs, wrong)
        return TypeCorrection(
            param=param, wrong_type=wrong, expected_type="object", error_message=msg
        )

    # "object of type 'int' has no len()"
    m = _NO_LEN.search(msg)
    if m:
        wrong = m.group(1)
        param = _identify_param(inputs, wrong)
        return TypeCorrection(
            param=param, wrong_type=wrong, expected_type="str", error_message=msg
        )

    return None


def _identify_param(inputs: dict[str, Any], type_name: str) -> str | None:
    """Try to identify which input parameter has the given type."""
    type_map = {
        "int": int, "str": str, "float": float, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "bytes": bytes, "NoneType": type(None),
    }
    target_type = type_map.get(type_name)
    if target_type is None:
        return None
    for name, value in inputs.items():
        if isinstance(value, target_type):
            return name
    return None


def _make_corrected_value(expected_type: str, old_value: Any = None) -> Any:
    """Generate a default value for the expected type."""
    defaults: dict[str, Any] = {
        "str": "test",
        "int": 1,
        "float": 1.0,
        "bool": True,
        "list": [1],
        "dict": {"key": "value"},
        "tuple": (1,),
        "set": {1},
        "bytes": b"test",
        "object": None,  # Will use SimpleNamespace
    }
    val = defaults.get(expected_type)
    if val is None and expected_type == "object":
        from types import SimpleNamespace
        return SimpleNamespace()
    return val


def try_execute_and_correct(
    source_path: str,
    callable_id: str,
    inputs: dict[str, Any],
    *,
    max_retries: int = 2,
) -> dict[str, Any] | None:
    """Try executing a function with inputs. On TypeError, correct and retry.

    Args:
        source_path:  Path to the Python source file.
        callable_id:  Function name.
        inputs:       Satisfying inputs to try.
        max_retries:  Maximum correction attempts.

    Returns:
        Corrected inputs if successful, or ``None`` if correction failed.
    """
    rp = os.path.realpath(source_path)
    # Through the core's one loader (#0607): ``None`` under ``no_targets()``.
    from tharkun.solve.loader import exec_source

    module_globals = exec_source(rp)
    if module_globals is None:
        return None

    func = module_globals.get(callable_id)
    if func is None or not callable(func):
        return None

    current_inputs = dict(inputs)
    for _ in range(max_retries):
        try:
            func(**current_inputs)
            return current_inputs  # Success!
        except TypeError as e:
            correction = parse_type_error(e, current_inputs)
            if correction is None or correction.param is None:
                return None
            new_val = _make_corrected_value(correction.expected_type, current_inputs.get(correction.param))
            if new_val is None:
                return None
            current_inputs[correction.param] = new_val
        except Exception:
            # Non-TypeError — inputs are type-correct at least
            return current_inputs

    return None
