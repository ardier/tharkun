# Testing: critical checks and Wave 2

The default pytest selection is the critical suite: `pytest.ini` selects `not wave2`.
Most long checks remain critical and are split into independent nodes. Two real timing-sensitive
checks run in Wave 2 on each new `sv_2027` SHA: the Hypothesis loop-back-edge reach search and
physical parallel CPU accounting. Their original correctness assertions remain in place; the
search gets 3,000 examples and 32 driver rounds in a test-only wrapper. CPU accounting keeps its
4 CPU-second / less-than-8 child CPU-second contract.

`scripts/wave2_tests.py` collects the actual full, critical and marked rosters and requires
an exact disjoint union. It runs the marked roster serially and checks full JUnit identities.
A skip, missing result, changed source, admission failure or outer timeout is INCOMPLETE.
Wave 2 unit status and alerts are separate from benchmark proof regressions. A critical gate
receipt reports the critical scope and deferred roster; it must not call that the whole suite.

The watcher queues this source-complete unit phase through normal TESTQ after benchmark pairs
finish, using one shared claim per actual main SHA for both head and nightly callers. Its existing
owned cgroup launcher retains the core lease until descendants drain, including detached children.
The phase uses the exact Git archive's tests, config and source. It does not use the benchmark RAM
copy that omits tests. Repeating a SHA makes no new submission; an uncertain claim needs recovery.

## Parallel gate command

The admitted parallel gate environment already provides pytest-xdist. Use its work-stealing
scheduler with the existing positive worker grant and critical selection:

```sh
"$PY" -m pytest -q -n "$TESTQ_JOBS" --dist=worksteal -p no:cacheprovider
```

Worksteal redistributes pending nodes among workers; it does not change the test roster.
Record actual collection, workers and wall time in each gate receipt. An invocation that clears
`addopts` must explicitly add `-m 'not wave2'` for the critical gate. The Wave 2 runner clears
defaults and selects the actual marked roster itself. Plain `python -m pytest` remains supported
without xdist; `pytest.ini` does not require the plugin.

## The rule

- A test node should take at most about **20 s** on an idle swsec host.
- A test over that budget is split into nodes that each assert their part, keeping the assertions it had. The ways this is
  done in the suite today:
  - **by case**: the absint kind matrix runs as ~310 nodes of at most 80 programs, split by value kind and then by position
    inside the kind (`tests/test_absint_kind_matrix.py::NODES`; `test_the_nodes_run_exactly_the_gate_programs` pins that the
    nodes are exactly the gate's programs, once each);
  - **by arm**: the hybrid benchmark's two arms (plain Atheris and the hybrid loop) share nothing, so
    `src/tharkun/hybrid/bench.py::benchmark(parallel=True)` runs them side by side with the same per-arm budget and
    `-max_total_time` (`tests/hybrid/test_bench.py`, `tests/hybrid/test_gunicorn_hybrid.py`);
  - **by objective**: the Avengers objective ladder runs its objectives (states, branches, invalid sequences) as separate
    nodes (`tests/avengers_rungs.py`, used by `tests/test_avengers_objectives.py`, `tests/test_avengers_bench.py` and
    `tests/test_avengers_store.py`);
  - **by phase**: `tests/test_sv_avengers_stage.py` checks the seeds, the run without the stage and the run with it in three
    nodes (they were one test).
- A **soundness test** (a check that no wrong TRUE or FALSE can come out: the kind matrix, the never-TRUE corpora, the engine
  audits, `tests/test_sv_draw_domains.py`) is in the gate like every other test and is split, or given the budget its question
  needs, when it is long (the draw-domain "never FALSE" check asks whether the enumeration's own first candidates are admissible,
  which a 12 s search answers as well as a 30 s one).
- A node is not moved out of the critical suite merely because it is slow. A real randomized search or physical timing
  contract that cannot become deterministic belongs in the explicit Wave 2 roster; deterministic bookkeeping and
  soundness checks stay critical.

## Randomized and wall-clock-boxed checks: metrics, not gate assertions

No gate assertion may depend on a random search succeeding within a wall-clock budget, or on which of two racing stages
finishes first (owner, 2026-10-06: "pynguin is probabilistic, it cannot be a basis for a hard test"). A test of that kind is
rewritten one of three ways:

- **a deterministic stand-in** drives the part that needs the search's result: `tests/test_sv_avengers_stage.py::fixed_pynguin`
  replaces `PynguinBackendPass.generate` with a fixed generator that writes Pynguin-style test source with one known
  in-window argument, and the production miner, replay and seed harvest run on it (the stage's own wall-clock ceilings are
  raised to 60 s per class and 120 s in all, so a loaded host cannot make them the limiting factor: 48 concurrent pytest
  processes on a 128-core host tripped the first 10 s ones);
- **only what holds for ANY outcome** is asserted of a real search (every value it returns is well formed, every validated
  object replays as a success and the pass was dispatched once, both benchmark arms really ran Atheris with the same
  `-max_total_time`: the fairness between the arms is that equality, not a bound on their wall times);
- **the rate is a measured metric**, printed as one JSON line by a script the nightly runs, never a pass/fail:
  `scripts/pynguin_seed_hit_rate.py` (how often real Pynguin seeds a value inside the class-guard window) and
  `scripts/hybrid_bench_metric.py` (how often the hybrid arm reaches deeper than plain Atheris, on the modular guard and the
  gunicorn regex chain; the nightly runs it at `--seconds 12` per arm, the budget the old assertions had, so the rate is
  comparable -- the gate's own 6 s run asserts only that both arms ran).

A wall-clock budget that is only a CEILING for something that finishes at once (a crash reported when the child dies, a first
run that raises) is made generous instead (`test_sv_forkserver`, `test_avengers_executor`, `test_sv_environment`,
`test_sv_solver_tricks`); the one place a test needs elapsed time (`test_scheduler`) drives the ledger's own CPU clock
(`engine._cpu_seconds`, with the `cpu_time_only` seam) so no sleep or wall clock is involved.

## Files and how they relate

- `tests/test_absint_kind_matrix.py` -- the generated kind-by-consumer matrix of the abstract interpreter ([internal ref]). `PROGRAMS`
  is every program; `GATE_PROGRAMS` is the every-eighth slice the gate runs; `KINDS`, `BY_KIND` and `NODES` split it, and
  `test_no_kind_of_value_makes_a_consumers_path_infeasible` has one node per entry of `NODES`. `scripts/absint_kind_matrix.py`
  runs the whole matrix (`sample`, `run_shard`).
- `scripts/pynguin_seed_hit_rate.py`, `scripts/hybrid_bench_metric.py` -- the nightly metrics above; they import
  `src/tharkun/sv/avengers_seed.py::seed_values` and `src/tharkun/hybrid/bench.py::benchmark` and the fixtures of
  `tests/hybrid/fixtures`, and run nothing at import time.
- `tests/avengers_rungs.py` -- `only`, `passes_for`, `restrict_default_passes`: the production Avengers portfolio
  (`src/tharkun/avengers/api.py::default_passes`) with only one objective pass of `src/tharkun/avengers/objectives.py`
  (`StateDiversityPass`, `CoverageSetupPass`, `NegativeSequencePass`). Imported by the three Avengers test files above.

## Measured

Before the split (maclinux junit of `ec3cee0`, 4 loaded CPUs per run, 14,279 tests, 4,624 test-seconds): 78 tests over 10 s took
2,860 s (62% of all test time); the 16 largest held 2,168 s: eight kind-matrix shards of 140-230 s, and Avengers searches of
39-102 s. After the split no node of the kind matrix runs more than ~25 s, and the Avengers searches run as parallel nodes of at
most ~28 s; the gate's wall time before and after is in the pull request that made the change.

## Adding a test

Write it so that one node does one thing. If it needs a long search, parametrize the search by what it can be divided by (a
case, an objective, a seed, a phase) and assert each part in its own node; do not leave one 60 s node.
