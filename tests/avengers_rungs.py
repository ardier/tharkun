"""Test helper: run ONE rung of the Avengers objective ladder at a time, so the long end-to-end searches run as parallel nodes.

``synthesize_object(..., objectives=True)`` chases three objectives after it has built the object: the states a receiver can be in
(``StateDiversityPass``), the branches of the class (``CoverageSetupPass``) and deliberately invalid call sequences
(``NegativeSequencePass``).  A test that asserts a fact about each of them used to run all three in one search, one pytest node of
25-50 s that held up the whole gate.  The objectives are independent passes over the same validated recipes, so a test node can run
the production portfolio with only ONE of them (the construction passes, the method-probe matrix and everything else are the
same) and assert that objective's part; the nodes run side by side under xdist.
"""
from __future__ import annotations

RUNGS = ("states", "coverage", "negatives")


def _objective_classes():
    from tharkun.avengers.objectives import CoverageSetupPass, NegativeSequencePass, StateDiversityPass

    return {"states": StateDiversityPass, "coverage": CoverageSetupPass, "negatives": NegativeSequencePass}


def only(rung: str, portfolio) -> tuple:
    """``portfolio`` without the objective passes of every rung but ``rung``."""
    classes = _objective_classes()
    drop = tuple(cls for name, cls in classes.items() if name != rung)
    return tuple(p for p in portfolio if not isinstance(p, drop))


def passes_for(rung: str, **default_passes_kwargs) -> tuple:
    """The production portfolio (``tharkun.avengers.api.default_passes`` with ``objectives=True``) restricted to one rung."""
    from tharkun.avengers.api import default_passes

    return only(rung, default_passes(objectives=True, **default_passes_kwargs))


def restrict_default_passes(monkeypatch, rung: str) -> None:
    """For code that builds its portfolio itself (``bench.run_one``): make ``default_passes`` return only ``rung``'s objective."""
    from tharkun.avengers import api

    real = api.default_passes
    monkeypatch.setattr(api, "default_passes", lambda **kw: only(rung, real(**kw)))
