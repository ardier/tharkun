"""Native CPython helpers shared by the contract audit tests (the engine audit, the swallowing catalog)."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import types
import warnings
from pathlib import Path


_PEER_CHILD = Path(__file__).resolve().parents[2] / "src/tharkun/sv/_child.py"


def peer_runtime_check_type():
    """Load check_type from the candidate-local child runtime."""
    if not _PEER_CHILD.is_file():
        raise FileNotFoundError(f"candidate _child.py is absent: {_PEER_CHILD}")
    spec = importlib.util.spec_from_file_location("_tharkun_peer_child_check_type", _PEER_CHILD)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load peer child runtime from {_PEER_CHILD}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != _PEER_CHILD.resolve():
        raise AssertionError(f"wrong peer child module origin: {module.__file__}")
    return module._make_sv_verifier().check_type


def run_native(src: str, vector=(), *, filename: str = "<native-case>", check_type_oracle=None) -> str:
    """Run ``src`` as ``__main__`` on THIS interpreter with ``_sv_verifier`` drawing from ``vector``;
    return the class name of the exception that ended the run, or ``""`` for a clean exit."""
    draws = list(vector)
    shim = types.ModuleType("_sv_verifier")
    shim.nondet_int = lambda: draws.pop(0) if draws else 0
    shim.nondet_bool = lambda: draws.pop(0) if draws else False
    shim.nondet_list = lambda nondet_elem: [nondet_elem() for _ in range(1 if shim.nondet_bool() else 0)]
    shim.nondet_dict = lambda nondet_key, nondet_value: (
        {nondet_key(): nondet_value()} if shim.nondet_bool() else {}
    )
    if check_type_oracle is not None:
        shim.check_type = check_type_oracle
    saved = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = shim
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            exec(compile(src, filename, "exec"), {"__name__": "__main__"})
        return ""
    except BaseException as e:  # noqa: BLE001 -- every uncaught exception is the outcome
        return type(e).__name__
    finally:
        if saved is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = saved


def write_program(src: str, directory: str, name: str = "prog.py") -> str:
    path = os.path.join(directory, name)
    with open(path, "w") as fh:
        fh.write(src)
    return path


#: the planner needs an int draw to reason about: a trip count the loop body never reads
_TRIP = "    _trips = abs(_sv_verifier.nondet_int())\n    for _ in range(_trips):\n        pass\n"


def with_trip_count(src: str) -> str:
    """``src`` (in benchmark shape) with a trip-count draw first in its main block, which is what makes
    ``proof:stationary`` consider it at all (``prove.py::prove_dynamic`` needs an int draw to reason about)."""
    marker = 'if __name__ == "__main__":\n'
    head, _, tail = src.partition(marker)
    return head + marker + _TRIP + tail


def stationary(src: str, classes, python: str, *, shaped: bool = False) -> tuple[str, str]:
    """``proof:stationary`` on ``src``: ``("TRUE", reason)``, ``("found", reason)`` for a violation, or
    ``("refuses", reason)``.  ``shaped`` adds the trip-count draw first."""
    from tharkun.sv.prove import prove_dynamic

    msgs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="audit-stationary-") as tmp:
        path = write_program(with_trip_count(src) if shaped else src, tmp)
        v = prove_dynamic(path, tuple(classes), python=python, remaining=lambda: 60.0, log=msgs.append)
    if v is None:
        return "refuses", msgs[-1] if msgs else ""
    if v.verdict == "true":
        return "TRUE", v.reason
    return "found", v.reason
