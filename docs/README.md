# Tharkun Documentation

Tharkun is the umbrella package for branch solving, directed reachability,
hybrid fuzzing, and Avengers object synthesis. These documents explain its
public workflows and the boundaries between its internal layers:

- [HOW_IT_WORKS.md](HOW_IT_WORKS.md)
  is the guided walkthrough. It explains the solver with examples and is the
  fastest way to understand the user-facing behavior.
- [AVENGERS.md](AVENGERS.md) documents object construction, setup sequences,
  method and free-function boundaries, the flattened observation of every call's
  effect on the object, and the `py_obj_flatting` bridge.
- [LAYERS.md](LAYERS.md) defines which modules solve, observe, drive, or fuzz.
- [INPUTS.md](INPUTS.md) is the contract of `tharkun.find_inputs` / `tharkun.inputs.Session`. It covers the one
  search entry above DRIVE and FUZZ ([internal ref]): the request and answer JSON (schema v1), the draw domains,
  the harness forms, the ladder and its hook point.
- [HYBRID_GENERATORS.md](HYBRID_GENERATORS.md) is limit-then-steer for the hybrid fuzzer: a `strategies` constraint dict
  limits the input domain, Atheris searches inside it, and the target line and the distance to the wanted value steer
  it. The bridge (`hybrid/domain_codec.py`), the steering signals (`hybrid/steer.py`), the three-arm runner
  (`hybrid/arms.py`) and the measured comparison on five example programs.
- [BOOTSTRAP.md](BOOTSTRAP.md) is the bootstrap loop ([internal ref]): one hub (a shared knowledge base) that typeinfer,
  Avengers and the rest publish loop facts to and read from, the driver that runs rounds over the construction
  dependency graph bottom up, the per-round growth report, and the first slice's measurement.
- [TYPEINFER.md](TYPEINFER.md) is `tharkun.typeinfer` ([internal ref]): the types of a function's parameters, from
  numbered sources, as `find_inputs` kinds; the annotation-to-kind table and the kind adapters.
- [LOWERING.md](LOWERING.md) is the contract of the exact source lowerings (comprehensions to
  loops for `solve/absint`): what they promise, refuse, and how their findings map back to the
  original program.
- [THARKUN_DESIGN.md](THARKUN_DESIGN.md)
  is the architecture document. It describes the actual module boundaries,
  runtime flow, data models, solver chain, extension points, and current gaps.
- [TESTING.md](TESTING.md) is the test-suite policy: one tier, long tests chunked into parallel nodes, and how the
  kind matrix and the Avengers searches are split.
- [THARKUN_SKILL_DESIGN.md](THARKUN_SKILL_DESIGN.md)
  documents the local Codex skill in `.agents/skills/tharkun/SKILL.md`: what
  it is trying to do, how it should interact with the library, where it is
  still useful, and where its prompt has drifted from the standalone repo.

## Recommended Reading Order

If you are new to the project:

1. Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md).
2. Read [AVENGERS.md](AVENGERS.md) if the target needs a constructed receiver.
3. Read [THARKUN_DESIGN.md](THARKUN_DESIGN.md).
4. Read [THARKUN_SKILL_DESIGN.md](THARKUN_SKILL_DESIGN.md) only if you care about Codex agent behavior or the repo-local skill.

If you are modifying the solver:

1. Read [THARKUN_DESIGN.md](THARKUN_DESIGN.md).
2. Read the relevant module in `src/tharkun/`.
3. Read the corresponding tests in `tests/`.

If you are modifying the skill:

1. Read [THARKUN_SKILL_DESIGN.md](THARKUN_SKILL_DESIGN.md).
2. Read the live skill file at [.agents/skills/tharkun/SKILL.md](../.agents/skills/tharkun/SKILL.md).
3. Cross-check the skill instructions against the current standalone package
   layout in `src/tharkun/`.

## Current Status Summary

As of this repository snapshot:

- Tharkun is a standalone Python package in `src/tharkun/`.
- The core library surface is exposed from [src/tharkun/__init__.py](../src/tharkun/__init__.py).
- Avengers is exposed as `tharkun.avengers`; it is part of Tharkun, rather
  than a separate distribution.
- `py_obj_flatting` is also a Tharkun component. It remains an independently
  packaged companion that Avengers requires ([internal ref]), used only behind
  `tharkun.avengers.flatting`.
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
