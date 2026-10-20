# Tharkun — SV-COMP 2027 build

This repository is the public release of Tharkun's SV-COMP 2027 build (release
`svcomp27-0.2.1`, package version 0.2.1). It contains Tharkun's solve, knowledge, drive
and search layers, the static analyser in solve/absint, and the sv front end.
Hypothesis and z3 are required dependencies.

Touchstone and CrossHair adapters, calls, configuration and dependencies are
removed from this build. Some documents in this repository may still describe
capabilities of the regular Tharkun build that this release does not include.

## Install and run

Requires Python 3.10 or newer; competition measurements use Python 3.14.

    git clone https://github.com/ardier/tharkun.git
    cd tharkun
    python -m pip install .
    python -m tharkun.sv PROGRAM.py --property PROPERTY.prp --budget-seconds 120

The SV interface, property semantics and witness format are unchanged. FALSE
answers require replay on the original program. Static proofs, deterministic
execution and exhaustive finite exploration may establish TRUE; exhausted
search returns UNKNOWN.

The portfolio retains native integer arithmetic, static analysis, guard-value
seeding, z3 symbolic execution, snapshots, minleaf and Hypothesis search.
See [the SV front end](docs/SV.md) for stages, resource limits and witnesses.

## Development

    python -m pip install -e '.[dev]'
    python -m pytest

One suite, no slow tier: a long test is split into parallel nodes (see [docs/TESTING.md](docs/TESTING.md)).

tests/test_sv_import_graph.py traverses all SV entry points, including workers
launched as subprocess scripts, then follows lazy and relative imports through
Tharkun. It rejects either removed solver or adapter anywhere on that graph.

## Contents of this release

The release contains the SV-COMP verifier (`python -m tharkun.sv`), the
modules it uses, and the `tharkun` command-line interface (`python -m tharkun`:
`solve`, `decompose`, `analyze`, `branches`, `emit`, `mocks`, `bench`). Modules
that neither uses are left out: the MCP server, the benchmark harnesses of
Avengers and the hybrid fuzzer (`tharkun.avengers.bench`,
`tharkun.hybrid.bench`), the egglog canonicalizer shadow, the differential-kill
API, the front-end helpers `structure_infer` and `type_hints`, the command-line
entry of `tharkun.solve.absint`, and the pre-0.2.0 compatibility module paths.
Documents in `docs/` may still mention them.

## Licence

Tharkun is licensed under the European Union Public Licence, version 1.2
(EUPL-1.2); see [LICENSE](LICENSE). The output of the tool (verdicts,
witnesses, log messages) is not covered by the licence and may be used without
restriction. Copyright (c) 2026 CISPA Helmholtz Center for Information Security.
