"""Thin, per-engine adapters onto ``tharkun.solve.contracts`` ([internal ref] step 2).

Each module here is the ONE place an engine's proof/search logic touches contract
data; no engine keeps a private stdlib table once wired (``census.py``'s
``_CALL_SITES``, ``absint/interp.py``'s ``MODULES``, and similar tables are
EXTENDED through these adapters, not duplicated). There are no feature flags
([internal ref], owner rule 2026-09-28): every adapter is always on, and only ever ADDS
information gated by ``Contract.tested`` -- a withdrawn or untested contract
returns an engine to exactly what it did before the contract existed, and no
adapter removes or overrides an existing check.

Layering (``docs/LAYERS.md``): everything directly under this package is itself
``solve/`` -- stdlib only, imports only other ``solve/`` modules. The engines
that sit above ``solve/`` (``sv/_symexec.py`` for concolic, ``hypothesis_adapter.py``
/ ``hybrid/`` for draws, ``avengers/`` for object construction) call INTO
``common.py``'s pure, stdlib-only helpers for the contract data, and do their
own library-specific translation (a z3 fact, a Hypothesis strategy, ...) in
their own file, in their own layer -- never inside this package, which must
stay importable without z3/hypothesis/atheris installed.
"""

from __future__ import annotations
