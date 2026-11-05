"""The ONE seed of a Tharkun search ([internal ref]; owner 2026-10-23: "set the seed").

A search is randomized in several places: Hypothesis (``find`` / ``@given``), the libFuzzer engine under Atheris, and
whatever those children start.  Run to run, the same question could therefore answer differently (lane 1329 saw 13
versus 14 answers).  This module is the single control that makes the random choices a function of a number:

* ``find_inputs`` takes the number from ``hints.rng_seed`` (docs/INPUTS.md "Hints"), default :data:`DEFAULT_RNG_SEED`,
  and runs the whole request inside :func:`seeded`;
* every component that draws randomly asks this module for its generator, naming itself (the ``stage``), so two
  components never share a stream and one component's stream does not move when another draws more: Hypothesis
  through :func:`hypothesis_random` (``find(random=)``) and :func:`hypothesis_seed` (``@seed``), libFuzzer through
  :func:`libfuzzer_seed`, anything else through :func:`rng`;
* the number reaches a child process through the environment (:data:`ENV_RNG_SEED`), which :func:`seeded` sets for
  the call, so a child that draws (the drive worker's Hypothesis) derives the same streams.

What it does NOT fix, on purpose: a stage that stops on the wall clock (the request's ``budget_seconds``, the
per-stage shares, the hybrid fuzzer's CPU budget) stops after a different number of draws on a slower or busier
machine; the draws themselves are the same stream.  ``PYTHONHASHSEED`` of the search children is already pinned by
the engine (``sv/engine.py::run_once``, 0, and ``1..K-1`` for the replay confirmations).

Stdlib only (a SOLVE module, docs/LAYERS.md).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import random
from typing import Iterator

#: the seed of a request that names none: fixed, so an unseeded request is reproducible too
DEFAULT_RNG_SEED = 0
#: how a child process learns the seed of the request that started it
ENV_RNG_SEED = "THARKUN_RNG_SEED"

_current: list = []          # the stack of seeds of the :func:`seeded` calls in this process (innermost last)


def current() -> int:
    """The seed in force: the innermost :func:`seeded` call, else the environment (a child process), else the default."""
    if _current:
        return _current[-1]
    try:
        return int(os.environ.get(ENV_RNG_SEED, DEFAULT_RNG_SEED))
    except ValueError:
        return DEFAULT_RNG_SEED


@contextlib.contextmanager
def seeded(seed: int) -> Iterator[int]:
    """Run a block with *seed* in force, in this process and (through the environment) in every child it starts."""
    _need_seed(seed)
    before = os.environ.get(ENV_RNG_SEED)
    _current.append(seed)
    os.environ[ENV_RNG_SEED] = str(seed)
    try:
        yield seed
    finally:
        _current.pop()
        if before is None:
            os.environ.pop(ENV_RNG_SEED, None)
        else:
            os.environ[ENV_RNG_SEED] = before


def _need_seed(seed: object) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError(f"rng seed must be an int >= 0, not {seed!r}")


def derive(stage: str, seed: int | None = None) -> int:
    """A 64-bit number for *stage* under *seed* (default: :func:`current`): the SHA-256 of both, never ``hash()``, so it
    is the same in every process under every ``PYTHONHASHSEED``."""
    base = current() if seed is None else seed
    return int.from_bytes(hashlib.sha256(f"{base}|{stage}".encode()).digest()[:8], "little")


def rng(stage: str) -> random.Random:
    """A fresh ``random.Random`` for *stage* (the same stream every call, until the seed changes)."""
    return random.Random(derive(stage))


def hypothesis_random(stage: str) -> random.Random:
    """The ``random=`` argument of ``hypothesis.find`` for *stage*: ``find`` then seeds its test from it."""
    return rng(stage)


def hypothesis_seed(stage: str) -> int:
    """The number for ``hypothesis.seed(...)`` (the decorator of an ``@given`` test) for *stage*."""
    return derive(stage)


def libfuzzer_seed(stage: str) -> int:
    """The value of libFuzzer's ``-seed=`` for *stage*: an unsigned 32-bit number, never 0 (0 means "pick one")."""
    return derive(stage) % (2**32 - 1) + 1
