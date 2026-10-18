# Tharkun Documentation

Tharkun is the umbrella package for branch solving, directed reachability,
hybrid fuzzing, and Avengers object synthesis. These documents explain its
public workflows and the boundaries between its internal layers:

- [HOW_IT_WORKS.md](HOW_IT_WORKS.md)
  is the guided walkthrough. It explains the solver with examples and is the
  fastest way to understand the user-facing behavior.
- [AVENGERS.md](AVENGERS.md) documents object construction, setup sequences,
  method and free-function boundaries, and the optional `py_obj_flatting`
  bridge.
- [LAYERS.md](LAYERS.md) defines which modules solve, observe, drive, or fuzz.
- [LOWERING.md](LOWERING.md) is the contract of the exact source lowerings (comprehensions to
  loops for `solve/absint`): what they promise, refuse, and how their findings map back to the
  original program.
- [THARKUN_DESIGN.md](THARKUN_DESIGN.md)
  is the architecture document. It describes the actual module boundaries,
  runtime flow, data models, solver chain, extension points, and current gaps.
- [TESTING.md](TESTING.md) is the test-suite policy: one tier, long tests chunked into parallel nodes, and how the
  kind matrix and the Avengers searches are split.

## Recommended Reading Order

If you are new to the project:

1. Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md).
2. Read [AVENGERS.md](AVENGERS.md) if the target needs a constructed receiver.
3. Read [THARKUN_DESIGN.md](THARKUN_DESIGN.md).

If you are modifying the solver:

1. Read [THARKUN_DESIGN.md](THARKUN_DESIGN.md).
2. Read the relevant module in `src/tharkun/`.
3. Read the corresponding tests in `tests/`.

## Current Status Summary

As of this repository snapshot:

- Tharkun is a standalone Python package in `src/tharkun/`.
- The core library surface is exposed from [src/tharkun/__init__.py](../src/tharkun/__init__.py).
- Avengers is exposed as `tharkun.avengers`; it is part of Tharkun, rather
  than a separate distribution.
- `py_obj_flatting` is also a Tharkun component. It remains an independently
  packaged companion and is used only behind `tharkun.avengers.flatting`'s
  optional bridge.
- The CLI entry point is [src/tharkun/cli.py](../src/tharkun/cli.py).
- The runtime mutation solver `minleaf` is integrated and used from
  `branch_solver`.
- External fact injection for backward constraint propagation is implemented in
  `condition_solver` and threaded through `solve_branch`.
- `call_mapper` and `call_walker` exist, but cross-module backward call-chain
  walking is still incomplete relative to the original
  `SPEC_constraint_propagation.md`.
- The skill prompt exists and is still conceptually aligned with the solver,
  but some of its environment references are legacy and describe an older
  Eldarion-based integration rather than this standalone repo.

## Documentation Intent

These docs are deliberately written as design docs, not marketing docs:

- They describe what the code does now.
- They call out places where the code and the older written story drift.
- They separate stable design intent from still-open work.

That is important in this repo because the project has already gone through at
least one extraction phase, and several modules still carry the history of that
migration in their docstrings, tests, and integration seams.
