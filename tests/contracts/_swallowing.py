"""Runners for the protocol-swallowing catalog (``_swallowing_cases.py``): the native CPython run
and one function per Tharkun engine, each returning a small, comparable result.

Used by ``test_swallowing_native.py`` (the pins), ``test_swallowing_engines.py`` (the matrix) and
``scripts/swallowing_matrix.py`` (the table in ``docs/contracts/REPORT.md``).  Nothing here is an
oracle: the oracle is CPython itself, run on the very program the engine was handed.
"""

from __future__ import annotations

import tempfile

from tests.contracts._native import stationary as _stationary, write_program

#: how an engine answered, from the point of view of soundness
TRUE = "TRUE"          # claims "no run raises a property class": WRONG on a violating case
FLAGS = "flags"        # models the program and reports the possible violation (sound, precise)
REFUSES = "refuses"    # does not model the program / gives up (sound, incomplete)
FOUND = "found"        # a search engine produced a violating run


# ------------------------------------------------------------------ the static engines
def census(src: str, classes) -> tuple[str, str]:
    """``proof:no-site``: TRUE only when no construct of the program can raise ``classes``."""
    from tharkun.solve.census import site_census

    c = site_census(src, tuple(classes), "<swallow-case>")
    return (TRUE if c.no_site else REFUSES), c.reason


def absint(src: str, classes) -> tuple[str, str]:
    """``proof:interval`` / ``proof:shape``: TRUE, or UNKNOWN naming an unproven check (the engine
    modelled the program and found the site: FLAGS) or an unsupported construct (REFUSES)."""
    from tharkun.solve.absint import prove

    r = prove(src, tuple(classes), time_budget=30.0)
    if r.verdict == "TRUE":
        return TRUE, r.reason
    if "unproven check" in r.reason:
        return FLAGS, r.reason
    return REFUSES, r.reason


def product(src: str, classes, python: str) -> tuple[str, str]:
    """``symbolic:product``, the exact Z3 proof for loop-free int/bool programs."""
    from tharkun.sv.product_proof import prove

    with tempfile.TemporaryDirectory(prefix="swallow-product-") as tmp:
        path = write_program(src, tmp)
        v = prove(src, tuple(classes), program=path, python=python)
    if v is None:
        return REFUSES, ""
    if v.verdict == "true":
        return TRUE, v.reason
    return FOUND, v.reason


def stationary(src: str, classes, python: str) -> tuple[str, str]:
    """``proof:stationary``: the run-based proof (a static draw-flow plan, then native probe runs), on the
    catalog program with the trip-count draw ``prove_dynamic`` needs first in its main block."""
    outcome, reason = _stationary(src, classes, python, shaped=True)
    return {"TRUE": TRUE, "found": FOUND, "refuses": REFUSES}[outcome], reason


def concolic(src: str, classes, python: str, budget: float = 20.0) -> tuple[str, str]:
    """The concolic stage (symbolic execution + ``drive_to_line``): candidates are replayed by the
    engine, so it cannot say TRUE.  Its soundness-relevant claim is ``closed`` (every site
    solved to 'no run fires it'), which stops the search early: closed on a violating program
    is reported as TRUE here."""
    from tharkun.sv import concolic as C

    with tempfile.TemporaryDirectory(prefix="swallow-concolic-") as tmp:
        path = write_program(src, tmp)
        solved = C.solve(path, tuple(classes), budget_seconds=budget, python=python)
    detail = f"ran={solved.ran} sites={sorted(solved.sites.items())} candidates={len(solved.candidates)}"
    if solved.closed:
        return TRUE, detail
    if solved.candidates:
        return FOUND, detail
    return REFUSES, detail


def child(src: str, classes, vector, python: str) -> tuple[str, str]:
    """The engine's own child run (``sv/_child.py``): the exception class it reports for the
    run, for comparison with the native one."""
    from tharkun.sv.engine import run_once

    enc = []
    for v in vector:
        enc.append({"kind": "bool" if isinstance(v, bool) else "int", "value": v})
    with tempfile.TemporaryDirectory(prefix="swallow-child-") as tmp:
        path = write_program(src, tmp)
        out = run_once(python, path, enc, timeout=60.0)
    return (out.exc or ""), out.status
