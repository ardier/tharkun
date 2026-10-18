# Tharkun Design

> *Module paths in this document predate 0.2.0. The current layout, the layer rule and the move map (every old path is a permanent shim) are in [LAYERS.md](LAYERS.md).*

## Purpose

Tharkun is a branch guard solver for Python. Its job is narrower and harder
than general test generation:

- Given a branch guard at a specific source line, determine inputs that make
  that guard evaluate the desired way.
- When a guard cannot be solved statically, collect runtime evidence that makes
  the remaining search space smaller and more concrete.
- When a guard depends on call structure, object state, or external effects,
  produce enough analysis for a higher-level orchestrator to keep moving.

The design bias is deliberate:

- Prefer explicit structural analysis over blind search.
- Prefer solver techniques that preserve closeness to the observed runtime
  state.
- Keep the library standalone. Tharkun can be called by other systems, but the
  solver logic itself should stay here.

## Non-Goals

Tharkun is not trying to be:

- a general symbolic execution engine for arbitrary Python programs,
- a full debugger UI,
- a replacement for complete end-to-end fuzzing or mutation-testing pipelines,
- a dependency of Tulkas, Glamdring, or Eregion in the architectural sense.

Those systems may orchestrate Tharkun, but the branch solving logic is meant to
live in this package.

## Design Drivers

The current architecture makes sense only if you keep the main constraints in
view:

1. Python guards are heterogeneous.
   A single guard may mix numeric constraints, object attributes, helper
   function calls, regexes, and runtime-computed intermediates.

2. Static analysis alone is not enough.
   Many guards depend on values computed earlier in the function body, values
   stored on `self`, or helper calls whose meaning only becomes obvious at
   runtime.

3. Search alone is too expensive.
   Random or property-based exploration can eventually find some paths, but it
   scales badly on compound guards if it is not fed structural hints first.

4. A branch solver must explain itself.
   The package is not just looking for a value; it also needs to say which
   technique worked, what facts were inferred, what runtime state was observed,
   and why a branch is still blocked.

5. The library has to degrade gracefully.
   CrossHair and Hypothesis are optional. Runtime execution may fail.
   Cross-module resolution may be partial. The design has to preserve useful
   intermediate results even when the full chain does not succeed.

## High-Level Architecture

At a high level, Tharkun is a layered pipeline:

```text
CLI / public API
    |
    v
branch_solver
    |
    +--> condition_solver
    +--> guard_decomposer
    +--> guard_analyzer + trace_api
    +--> attr_tracer
    +--> mock_gen
    +--> minleaf
    +--> crosshair_adapter
    +--> hypothesis_adapter
    +--> emit
```

There are two important support layers under that:

- `call_walker`, `import_resolver`, and `type_propagation` provide structural
  understanding of code and call relationships.
- `config`, `types`, `constructor`, `proxy`, and `exception_feedback` provide
  correction and refinement around the main solving path.

## Module Map

### Public surface

- [src/tharkun/__init__.py]([local path])
  exports the main datatypes and entry points.
- [src/tharkun/cli.py]([local path])
  provides the CLI commands `solve`, `decompose`, `analyze`, `branches`,
  `emit`, `mocks`, and `bench`.

### Solving core

- [src/tharkun/condition_solver.py]([local path])
  is the static guard solver and also the snapshot-assisted re-solver.
- [src/tharkun/branch_solver.py]([local path])
  is the orchestrator that sequences all per-branch techniques.
- [src/tharkun/guard_decomposer.py]([local path])
  recursively traces guard calls into helper functions.
- [src/tharkun/guard_analyzer.py]([local path])
  combines decomposition and runtime evidence to explain blockers.

### Runtime evidence and mutation

- [src/tharkun/trace_api.py]([local path])
  provides the line tracer and snapshot capture.
- [src/tharkun/minleaf/api.py]([local path])
  is the entry point for minimal leaf-value patch solving.

### Structural analysis and call reasoning

- [src/tharkun/call_walker.py]([local path])
  builds function/class indices and resolves callees.
- [src/tharkun/call_mapper.py]([local path])
  maps caller expressions to callee parameters.
- [src/tharkun/import_resolver.py]([local path])
  resolves imports to source files.
- [src/tharkun/type_propagation.py]([local path])
  propagates type hints backward through simple call edges.

### Object and environment support

- [src/tharkun/attr_tracer.py]([local path])
  finds where `self.xxx` values are assigned.
- [src/tharkun/constructor.py]([local path])
  synthesizes real instances from `__init__` AST analysis.
- [src/tharkun/proxy.py]([local path])
  uses recording proxies for usage-based type inference.
- [src/tharkun/exception_feedback.py]([local path])
  retries after type-related runtime failures.

### External boundary support

- [src/tharkun/mock_gen.py]([local path])
  detects calls that should be patched.
- [src/tharkun/crosshair_adapter.py]([local path])
  is the optional CrossHair integration.
- [src/tharkun/hypothesis_adapter.py]([local path])
  is the optional Hypothesis fallback.

### Output and benchmarking

- [src/tharkun/emit.py]([local path])
  renders solved branches as runnable tests.
- [src/tharkun/bench.py]([local path])
  is the benchmark harness.

### Shared configuration and datatypes

- [src/tharkun/config.py]([local path])
  centralizes thresholds, confidence scores, and time budgets.
- [src/tharkun/types.py]([local path])
  holds shared callable/type metadata.
- [src/tharkun/minleaf/types.py]([local path])
  holds minleaf-specific patch and fact models, including `FactEnv` and
  `VarFacts`.

## Core Data Models

### `ConditionSolution`

Defined in `condition_solver`, this is the output of static or snapshot-based
guard solving.

Important fields:

- `guard_source`: source text of the normalized guard expression,
- `variables`: list of inferred per-variable constraints,
- `satisfying_inputs`: list of input dictionaries,
- `strategy`: how the result was derived,
- `confidence`: numeric confidence score.

The design intent is that `ConditionSolution` is the smallest reusable solving
result. Anything more complex should still be expressible as “here is the guard
I solved, how I solved it, and the input maps I believe satisfy it.”

### `DecompResult`

Defined in `guard_decomposer`, this describes the internal structure of a guard
expression as a tree of leaves and calls. It exists because some guards are not
opaque formulas; they are small call graphs embedded inside a branch.

### `AnalysisResult`

Defined in `guard_analyzer`, this wraps a decomposition plus runtime snapshots
and per-blocker diagnostics. It is for explanation more than direct solving.

### `BranchSolution`

Defined in `branch_solver`, this is the unit of orchestration. It contains:

- the branch location,
- the enclosing callable,
- all attempted techniques,
- the winning inputs if any,
- mock patch requirements,
- decomposition data,
- attribute blocker analysis.

This is the key “audit trail” datatype for the full per-branch pipeline.

### `FactEnv` and `VarFacts`

Defined in `minleaf.types`, these are the project’s current shared fact model
for externally supplied constraints.

They matter in two different places:

- `condition_solver` consumes them to narrow type inference and candidate
  generation,
- `minleaf` converts them into leaf facts for patch search.

That shared fact type is what makes backward constraint propagation possible
without making the static solver depend on an external orchestrator.

## End-to-End Flow

The easiest way to understand the current design is to follow the path from
API entry to final answer.

### 1. Entry points

There are three main entry styles:

- direct library calls such as `solve_condition`, `decompose_guard`,
  `analyze_blocker`, and `solve_branch`,
- CLI commands in `cli.py`,
- higher-level automation through an external orchestrator or the local skill.

### 2. Static branch extraction

`condition_solver` starts by:

1. parsing the source file,
2. finding the branch expression at the requested line,
3. normalizing the guard when possible,
4. collecting its free variables.

This establishes the minimal static target to reason about.

### 3. Type inference and candidate generation

`condition_solver` then infers types using prioritized heuristics:

- `isinstance`,
- literal comparisons,
- method names,
- attribute patterns,
- subscripts,
- parameter annotations,
- external facts,
- fallback to `Any`.

It then generates candidate values from:

- comparisons,
- range intersections,
- string method constraints,
- length constraints,
- regex inversion,
- membership checks,
- truthiness patterns,
- scope-resolved constants.

### 4. Snapshot-assisted solving

If runtime locals are available, `solve_with_snapshot` can pin some variables
to concrete values and re-solve only for the remaining degrees of freedom.

Architecturally this is a major point: Tharkun is not “static solver first,
dynamic solver later.” It is a hybrid system in which runtime execution feeds
back into static reasoning.

### 5. Branch orchestration

`solve_branch` in `branch_solver` is the package’s main orchestrator. The real
implemented sequence is:

0. pre-step: cross-function type propagation,
1. condition solver,
2. guard decomposer,
3. attribute tracer,
4. mock detection,
5. regex inversion,
6. runtime snapshot plus snapshot re-solve,
7. state-sequence solving for `self.xxx` guards,
8. minleaf runtime patch solving,
9. CrossHair fallback,
10. Hypothesis fallback,
11. post-processing with constructor upgrade,
12. exception-driven feedback,
13. proxy-based inference as a last correction path.

This is important because the top docstring in
[branch_solver.py]([local path])
still reflects an older chain description. The code now does more than that
docstring claims.

### 6. Optional test emission

If the caller wants executable artifacts rather than just solutions,
`emit_branch_tests` turns `CoverageResult` into a Python test file.

That emitter is intentionally dumb in one useful sense: it emits from the
already computed branch solutions rather than re-solving or re-analyzing at
generation time. This keeps the pipeline reproducible.

## Static Solver Design

`condition_solver` is still the center of gravity of the project.

### Internal stages

At a high level, it runs:

1. branch lookup,
2. guard normalization,
3. free-name discovery,
4. type inference,
5. candidate extraction,
6. compound-guard repair or verification,
7. strategy/confidence assignment.

### Design choice: heuristics first, not theorem proving first

The static solver is intentionally heavy on domain heuristics because many real
Python guards are not solver-friendly formulas. Examples:

- `s.startswith("pre") and s.endswith("suf")`
- `hasattr(obj, "ready")`
- `x in VALID_SET`
- `data and len(data) > 5`
- `obj.kind == "method"`

These are cheap to solve with targeted syntactic rules and awkward to hand to a
generic symbolic engine as-is.

### External facts integration

The current implementation supports externally provided facts in several places:

- type inference from `eq`, `lo`, and `hi`,
- candidate filtering by `eq`, `neq`, `lo`, and `hi`,
- string synthesis preference for `eq`,
- int-range intersection constrained by fact bounds,
- early infeasibility return when `facts.possible` is false.

Relevant code:

- [condition_solver.py]([local path])
  for fact-based type inference,
- [condition_solver.py]([local path])
  for comparison candidate filtering,
- [condition_solver.py]([local path])
  for string synthesis with facts,
- [condition_solver.py]([local path])
  for integer range intersection with facts,
- [condition_solver.py]([local path])
  for the public `facts` parameter on `solve_condition`.

### Confidence design

Confidence values come from [config.py]([local path]).

This is a practical design, not a statistical one. The confidence score is a
structured hint about how strong the inference path was:

- direct extraction is high confidence,
- verified repaired solutions are slightly lower,
- type-inferred but unverified solutions are lower,
- snapshot-resolved or snapshot-simplified solutions are tuned separately.

The key point is that confidence is attached to strategy, not learned from a
corpus.

## Runtime Evidence Design

### `trace_api`

The tracer supports two modes:

- `sys.settrace` line tracing,
- `sys.monitoring` instruction spans when available and when no frame capture
  is needed.

The project still forces `sys.settrace` when `snapshot_lines` is requested,
because frame objects are necessary to capture `f_locals`.

Important design choices:

- paths are realpath-filtered,
- snapshots can be safe serialized values or live objects,
- trace results separate executed lines, executed spans, safe snapshots, and
  real snapshots.

This split exists because diagnostics and minleaf have different needs.

### `guard_analyzer`

`guard_analyzer` is not a solver replacement. It is an explanation layer over:

- `guard_decomposer`,
- tracer execution,
- blocker-to-snapshot matching.

Its job is to explain why a guard is still blocked and to surface the most
useful runtime facts. This keeps failure states informative instead of opaque.

## Guard Decomposition and Call Reasoning

### `call_walker`

`call_walker` is the shared structural-analysis substrate. It provides:

- module function/class indexing,
- callee resolution for names and several method-call patterns,
- return-condition extraction from helper functions,
- guard-clause handling,
- limited class and property awareness.

Both `guard_decomposer` and `call_mapper` depend on it.

### `guard_decomposer`

The decomposer recursively descends through:

- `BoolOp` trees,
- negation,
- resolvable function calls.

Its main design value is not only that it can find solvable sub-leaves, but
that it can classify why a leaf or call is still opaque:

- unresolved function,
- recursive call,
- complex return logic,
- max depth exceeded,
- unsolvable leaf.

That blocker taxonomy is what lets downstream tooling distinguish “cannot
solve” from “cannot solve yet because this call was not resolved.”

### `call_mapper`

`call_mapper` provides a backward-mapping view of call sites:

- which callee parameter receives which caller expression,
- whether the expression is a direct variable pass-through,
- what call chain connects an entry function to a target function.

This is the main bridge for backward constraint propagation.

## Minleaf Design

Minleaf is now part of the main solver chain rather than a separate prototype.

Its role is different from `condition_solver`:

- `condition_solver` synthesizes satisfying values from static code structure,
- `minleaf` mutates an already observed runtime state as little as possible.

That difference is architecturally important. Minleaf is not “another solver”
in the same sense; it is a state-local repair mechanism.

### Minleaf pipeline

The minleaf API in [minleaf/api.py]([local path])
runs:

1. environment/root consistency checks,
2. guard extraction,
3. guard normalization,
4. traced evaluation with blocking-atom identification,
5. state projection into leaf paths,
6. fact conversion into leaf facts,
7. dependency extraction,
8. BFS patch search bounded by leaf changes and candidate budgets,
9. structured diagnostics assembly.

### Why minleaf sits before heavy solvers

Minleaf narrows the problem in a way that generic search cannot:

- it knows which atoms are false,
- it knows which leaf values those atoms depend on,
- it searches for the smallest local patch that flips the outcome.

If it succeeds, the produced mutation is structurally close to the observed
state. If it fails, its diagnostics are still valuable for tighter heavy-solver
search.

## Optional Heavy Solvers

### CrossHair

CrossHair is used as an optional concolic fallback.

Current design properties:

- only activated if the dependency is importable,
- invoked as a subprocess via `crosshair cover`,
- result parsing is intentionally simple and brittle but serviceable,
- used only after lighter techniques have failed.

### Hypothesis

Hypothesis is the final fallback search layer.

Current design properties:

- builds strategies from inferred parameter types,
- can optionally use narrowed bounds from minleaf diagnostics,
- verifies success by checking whether the target line executed under the
  tracer.

This is a useful design because it constrains random search with prior
analysis, but it is still fundamentally a last-resort stochastic technique.

## Support Modules

### `type_propagation`

This module provides a cheap backward type propagation pass from simple call
edges. It is intentionally narrower than `call_mapper`:

- it only handles direct `ast.Name` pass-through cleanly,
- it skips method calls,
- it is good enough for the pre-step in `solve_branch`.

### `attr_tracer`

This module handles the object-state case:

- extract `self.xxx` targets from guards,
- find assignments to those targets,
- identify candidate setup methods.

This analysis is what makes state-sequence synthesis possible.

### `constructor`

This module upgrades placeholder object-like inputs into real instances when
possible. That matters because a static solver can often infer that “an object
with attribute `kind='method'`” is needed, but the caller may actually require
an instance of a concrete project class.

### `proxy`

Recording proxies are a fallback for cases where runtime usage tells you more
than source inference can. They are not first-line solving logic; they are a
post-failure inference repair mechanism.

### `exception_feedback`

This is a correction loop around runtime execution. Its design assumption is
simple: sometimes the best way to learn that an inferred input is wrong is to
run it and read the `TypeError`.

## Output Design

`emit.py` intentionally emits tests from `CoverageResult` without trying to be
clever. It supports:

- regular function calls,
- sequence/state setup flows,
- mock decorators,
- basic import inference.

The emitter is only as strong as the `BranchSolution` objects it receives. That
is a feature, not a bug: it keeps generation deterministic and inspectable.

## Configuration Design

`config.py` centralizes the numeric knobs:

- confidence values,
- timeouts,
- decomposition depth,
- search limits,
- default numeric ranges,
- solver budgets.

This is a practical separation:

- the solver code stays readable,
- tests can override behavior without monkeypatching random module constants,
- future orchestrators get one place to tune package behavior.

## Testing Strategy

The repository test layout mirrors the architecture:

- `tests/test_condition_solver*.py` for the static solver,
- `tests/test_guard_decomposer.py` and `tests/test_guard_analyzer.py`,
- `tests/test_branch_solver.py`,
- `tests/test_call_mapper.py`, `tests/test_call_walker.py`,
- `tests/test_crosshair_adapter.py`, `tests/test_hypothesis_adapter.py`,
- `tests/minleaf/` for the minleaf subpackage,
- `tests/fixtures/` for small targeted source examples.

This is a good design fit for Tharkun because most behavior is easiest to test
against source fixtures rather than abstract unit mocks.

## Architectural Drift and Current Gaps

This section matters because the repo has moved quickly and the written story
has not always kept up.

### 1. `branch_solver` docstring drift

The top docstring still describes an older technique chain and does not reflect
the current minleaf, CrossHair, Hypothesis, or post-processing stages.

### 2. Constraint propagation is only partly landed

The original intent in `SPEC_constraint_propagation.md` was:

- external facts into `condition_solver`,
- `facts` threaded through `solve_branch`,
- `call_mapper` for backward callee-to-caller mapping,
- working call-chain mapping across frames, including cross-module paths.

Current status:

- `facts` support is implemented and tested,
- `solve_branch(..., facts=...)` is implemented,
- `call_mapper` exists,
- `map_call_chain` is still incomplete for the intended cross-module workflow.

In other words, the package has the fact model and the mapping API, but the
end-to-end backward-walk substrate is not fully there yet.

### 3. `call_mapper` cross-module limitation

The current `call_mapper` implementation depends on `build_call_index` and
`parse_imports`, but its chain walk still begins by finding the target line in
the `source_path` AST itself. That means the intended use case of “entry file
on one side, deep target line in another module” is not fully realized yet.

This is the most important architectural limitation if Glamdring-style backward
constraint propagation is the next integration target.

### 4. Skill/environment drift

The repo-local skill still references an older Eldarion-based environment and
tool path. That does not break the library, but it does mean the skill is no
longer a perfect operational reflection of the standalone package.

That is documented in
[THARKUN_SKILL_DESIGN.md]([local path]).

### 5. Packaging/doc drift

`pyproject.toml` references `README.md`, but the repository currently relies
more on docstrings and `docs/`. The package is still usable, but the published
package story is thinner than the internal documentation story.

## Design Principles Going Forward

If the project keeps evolving, the current architecture suggests a few rules:

1. Keep the solver chain layered.
   Cheap structural reasoning should stay ahead of heavy search.

2. Keep runtime evidence explicit.
   Snapshots, blockers, and narrowed constraints should remain first-class
   outputs, not hidden implementation details.

3. Keep facts shared.
   `FactEnv` is becoming the common currency between static analysis, minleaf,
   and future backward propagation.

4. Keep call reasoning centralized.
   `call_walker` should remain the shared substrate rather than allowing each
   module to invent its own callee-resolution logic.

5. Document current behavior, not aspirational behavior.
   This repo has enough real capability now that outdated prose is more
   dangerous than missing prose.
