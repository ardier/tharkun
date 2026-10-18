# Layers — where solving ends and observation begins

*The layout contract of tharkun since 0.2.0 (plan #0600 of the eregion tree).
`tests/test_layering.py` enforces every rule on this page by AST scan; if you
change the layout, change that test in the same commit.*

## The delineation rule

A function is **SOLVING** if its input is a predicate about named variables
(a guard AST, a `FactEnv`, a constraint set, a `(source_path, lineno)` anchor)
plus optional prior facts, and its output is a claim about the variables'
**value set** — refined facts, a sound domain (point / finite set / interval /
empty), a witness, or satisfiable / infeasible / unknown — computed **without an
observed output `y`** and **without executing the target**.

A function is **INVERSION** (eregion's job, never tharkun's) if its input is a
forward computation (an IR op / step / path / descriptor) plus what the inverse
already holds (`state.known`, the observed `y`), and its output is how to
recompute an input: an expression over known names, a `GeneratorSpec`, a plan,
rendered code.

Anything that **executes** the target — `sys.settrace` / `sys.monitoring`,
evaluating a guard in live `f_locals`, `hypothesis.find` with a tracer predicate,
Hypothesis, Atheris, forward replay — is **OBSERVATION / SEARCH**. It lives in the
layers above `solve/`, never inside it.

Corollaries a reviewer applies:

- (a) computing the answer is solving; turning it into a generator / range /
  list for an inverter is inversion;
- (b) search **budgets** (enumeration windows such as ±8 or ±1000) are
  inversion-side and stay with the caller, passed as parameters;
- (c) a solver answer enters a caller only as a **sound-superset domain consumed
  by intersection** or as a **witness the caller verifies** — never as a
  replacement guess;
- (d) worked example: `x*3+1 == y` with `y` observed is inversion
  (`((y)-(1))//(3)` plus a `%3==0` check); `x*3+1 > 10 and x*3+1 < 40` with no
  `y` is solving (a preimage interval, `4 <= x <= 12`).

## The layer table

Only `solve/` is a directory today (Phase A). The other layers are logical
groups of top-level modules; Phase C regroups them into directories with the
same permanent-shim discipline.

| Layer | Modules | Runs the target? | May import |
|-------|---------|------------------|------------|
| **solve** — the core | `solve/condition` (static guard solver), `solve/loop_exit`, `solve/decompose`, `solve/attrs`, `solve/guard` (extraction + shared AST helpers), `solve/normalize`, `solve/guard_eval`, `solve/fact_env` (`VarFacts`/`FactEnv`), `solve/canonicalize`, `solve/envelope` (`affine_envelope`: a proven affine bracket of an integer expression), `solve/drawflow` (`plan_draws`: which SV-COMP draws reach a program only as loop trip counts or residues), `solve/pyresolve` (`Resolver`: every name, attribute and import of a program resolved against allowlists; reflection is TOP; read-and-recurse, [internal ref], follows an import into `solve/extern` (`root_snapshots`/`find_in_roots`/`toplevel_inert`: a directory the program's OWN `sys.path.insert(...)` adds, never an ambient path) instead of refusing outright, recursively, cached and cycle-guarded, exposed via `Resolver.externs`/`Resolver.all_resolvers`), `solve/determinism` (`determinism_census`: an allowlist certificate that a run is a function of the program and its draws), `solve/census` (`site_census`: no construct a program can run raises the given exceptions; an allowlist over `solve/pyresolve`), `solve/refute` (`refute_stdlib`: a Z3-free proof that conditions over typed int / float / bool names have no model -- exact integer elimination and Fourier-Motzkin via `solve/guard_domain` (`refine` / `refine_env` / `domain_of` / `atoms`: the sound per-variable `Value` on each arm of a guard AST, eregion's `reduce_domain` and `_extract_constraint` ported onto Python source, open world, with eregion's `domain_kind` as a label), `solve/arith`, then interval propagation over `solve/values`; the dead-path pre-check), `solve/values` (`IntFact` / `FloatFact` / `StrFact` / `Kinds` / `Value`: sound per-variable value facts, their lattice operations, outward-rounded transfer functions with the exceptions they may raise, and `refine_compare`), `solve/arith` (`solve_int_guard`: a witness for an integer guard by exact linear elimination, modular and polynomial root solving; `solve_condition`'s fallback for integer names), `solve/absint/*` (a sound whole-program abstract interpreter for `! uncaught(E)`, [ABSINT.md](ABSINT.md)), `solve/lowering/*` (exact source lowerings for the analyses that do not model a construct: comprehensions to loops, with the origin of every node, stdlib only, [LOWERING.md](LOWERING.md)), `solve/rewrite_catalog`, `solve/constants`, `solve/types`, `solve/config`, `solve/loader`, `solve/module_namespace` (`module_constants`, `regex_globals`: which module-level names nothing in a program's import graph rebinds, and the compiled regexes it defines, folded statically; executes nothing), `solve/environment` (the shared environment model and `confined()`), `solve/frontend/*` (import / field / type resolution, constructor, call graph) | **No.** `solve/loader.py` is the one place a target module is *loaded* (to read a computed `re.compile`, to construct a class named in `isinstance`, and, since #1132b, to import a project module for real as a member of its own package: `load_project_module`, only under `solve/environment.py::confined`, or `import_project_isolated` in a disposable child); `no_targets()` switches even that off. `solve/module_namespace.py` reads source with `ast` and executes nothing. | stdlib only (`hypothesis` lazily, for regex witnesses) and other `solve/` modules |
| **trace** | `trace_api` | yes — observation | stdlib |
| **search** | `hypothesis_adapter`, `minleaf/*` (live-state minimal patches) | yes — live state, SMT probes, property search | solve, trace, optional deps |
| **kb** | `facts` (the strategy knowledge base — unrelated to `solve/fact_env`), `lattice`, `knowledge`, `warmup`, `egraph`, `_egraph_rules`, `egglog_bench` | no | solve |
| **drive** | `branch_solver` (the technique chain), `guard_analyzer`, `driver` (`drive_to_line`), `differential`, `mutation_input_gen`, `mock_gen`, `avengers/*` (object construction and boundary replay), `sv/*` (the SV-COMP Python verifier front end, [SV.md](SV.md)) | yes | everything below; `avengers/flatting` may lazily import the optional companion |
| **fuzz** | `redqueen`, `hybrid/*` | yes — Atheris | drive |
| **tools** | `cli`, `mcp_server`, `emit`, `bench`, `slicing_bench`, `demos/` | — | drive |

`solve/determinism` also uses the private stdlib-only `solve/set_singleton`
leaf to prove fresh singleton-set receivers from closed finite call rows. It
analyzes ASTs and never loads or runs the target.

Rules the test enforces:

- every module under `solve/` imports only `solve/` (eager or lazy);
- no `importlib`, `exec`, `__import__`, `subprocess`, `runpy`, `sys.settrace`,
  `sys.setprofile`, `sys.monitoring` under `solve/` outside `loader.py`;
- no eregion / amonhen names and no IR vocabulary (`IRExpr`, `IRFlatStep`,
  `Program`, `GeneratorSpec`, `EnumCall`) anywhere under `solve/`;
- `solve/condition.py` eagerly imports only its leaves; `trace_api` imports no
  tharkun module; the fuzzing layer has no inbound edge from outside itself;
- every `solve/` module imports in a fresh process without loading the tracer,
  the driver, the search adapters or the fuzzing layer;
- tharkun itself never imports the old module paths — those are shims for
  external consumers.

`solve/extern.py` ([internal ref], "read-and-recurse", docs/SV.md/ABSINT.md "Import boundary") and the
`solve/pyresolve.py::Resolver`/`solve/absint/interp.py::Interpreter` code that calls it read
(`open`/`ast.parse`, never `importlib`/`exec`/`__import__`) OTHER source files a PROGRAM'S OWN
`sys.path.insert(...)` points at, recursively — this is parsing more of the same program's own
text, the identical mechanism SOLVE already uses on the program handed to it, not "loading" a
target the way `solve/loader.py` does (real import machinery, to read a computed `re.compile`
or construct an `isinstance` class) — `loader.py` stays the one place SOLVE actually imports
target code for real; `extern.py` never does.

**Project import (#1132b).** A module of a package exists only inside its package: `import pkg.mod` runs
`pkg/__init__.py` first, then the submodules in the order their imports ask for them, and what `__init__` does
(`_c.LIMIT = 5`) is what `mod` sees. Three functions of `solve/loader.py` are the ONE answer for every tool that
must see the real thing: `project_root_and_module` (static: the project root and dotted name, the way Python
imports the file; `avengers/backends.py::_project_and_module` delegates to it, so Pynguin and every in-process
caller agree on what "the project" is), `load_project_module` (the real `importlib.import_module` in the calling
process, with the project root on `sys.path` for the import only, a same-named cached package evicted first and
restored on failure, the import and every later call through the returned `ConfinedModule` facade under the
caller's `confine` context) and `import_project_isolated` (the same import in a disposable child, with the
environment model installed first, answering with a JSON report). The Eregion inverter's forward replay (eregion-codegen,
through the one seam of eregion-synthesis that imports Tharkun) is the caller of the first; Eregion's synthesis
asks `module_constants` which module-level names are provable constants before it bakes one into an inverter.
Tharkun imports nothing of Eregion's. `tests/test_solve_project_loader.py` and `tests/test_module_namespace.py` pin them.

`tharkun.avengers` is Tharkun's DRIVE-layer object-construction surface. Its
recipes contain literal constructor arguments, nested sub-recipes, setup calls,
and method or module-function boundaries; replay runs target code in a
disposable child interpreter before a recipe is promoted, whether the goal asks
the boundary to return, to raise, or to bring the interpreter down.
`tests/test_layering.py` enforces the edges: nothing above DRIVE is imported,
no pre-0.2.0 shim path is used, and only `sv/` or `driver.py` may import
`avengers` from outside the package.

`driver.py`'s `_flip_avengers` ([internal ref] slice 3, [internal ref], owner ruling,
corrected 2026-09-29) is a portfolio tier for a guard on an attribute of a
parameter with no live value (minleaf's own documented blind spot): on demand
only, for the one parameter and only the field bounds that ONE guard states,
never an unconditional sweep. It calls `tharkun.avengers`'s existing public
API directly (`ObjectGoal`, `SaturationBudget`, `synthesize_object`,
`field_bounds_hold`, `materialize_recipe`) — Avengers is already Tharkun's
generic, implemented object-construction line, so there is no separate
service module between driver.py and it; the guard-to-goal translation
(`_avengers_field_bounds_from_guard` and friends) is driver.py's own AST
analysis over the program it is driving, not Avengers' concern, and lives
beside `_flip_avengers` rather than in `avengers/`. It runs unconditionally,
on demand only — no boolean gate (owner ruling, no feature flags) — bounded
per guard by `avengers_budget_seconds`; a task with no such guard spends 0
seconds here. Eregion's own [internal ref] stage (a different repo, W57) is a second,
independent caller of the same `tharkun.avengers` public API, not of
driver.py.

`tharkun.avengers.backends` runs Pynguin and Ghostwriter as subprocesses or
inside the worker, so those tools are availability-gated optional dependencies
of a DRIVE-layer pass and never of the core. Nothing outside `avengers/` may
import `backends` directly, `sv/` included: SV reaches the object-construction
surface only through `tharkun.avengers`'s public API (`synthesize_object`,
`ObjectGoal`, ...), never through the backend module or `pynguin`/`black`
themselves — `tests/test_sv_no_backends.py` enforces that the shipped `python
-m tharkun.sv` entry point never loads either. `tharkun.avengers.bench` and
`.stats` are measurement, stdlib-only, and import nothing above DRIVE either.

`py_obj_flatting` is also a Tharkun component, packaged as its official
companion distribution. Only `tharkun.avengers.flatting` crosses that
boundary, lazily and optionally, to turn a validated exemplar into an
addressable recipe; importing or using Avengers without the companion remains
supported. `tests/test_layering.py` also enforces that nothing outside
`avengers/` imports `py_obj_flatting` directly: SV, the solver and the fuzzing
layer reach a flattened object only through Avengers APIs (`synthesize_object`
/ `ObjectGoal`) and the recipes Avengers materializes.

## What eregion may import

eregion's engine packages (`eregion-commons`, `eregion-ir`,
`eregion-forward-passes`, `eregion-codegen` and the runtime helpers shipped next
to every emitted inverter) import **nothing** from tharkun. `eregion-synthesis`
reaches tharkun through **one bridge module** (its solver router's tharkun
backend), **lazily** and **optionally** — with tharkun absent or the solver
switched off, eregion runs on its own policies and domain defaults,
byte-identically. The bridge imports only public names of `tharkun.solve`:

```python
from tharkun.solve import FactEnv, VarFacts, loop_exit_factenv, solve_condition, no_targets
```

`tharkun.solve` is a lazy facade (PEP 562): `from tharkun.solve import
solve_condition` loads the static solver and nothing else. tharkun never depends
on eregion (its charter, `CLAUDE.md`).

The other direction ([internal ref] step 3) is a plugin seam, not an import: `tharkun.sv`
discovers fact producers by the entry-point group `tharkun.sv.fact_producers`
(`sv/fact_producers.py`) and runs each in a child process over plain JSON, so
tharkun never names, imports or links an eregion package; `eregion-codegen`
registers its producer in its own `pyproject.toml` and imports nothing from
tharkun to do so. With no producer installed the seam is one entry-point lookup.

## The move map (0.2.0) and the shim promise

Every path on the left is a **permanent shim**: importing it yields the moved
module itself (aliased in `sys.modules` by `tharkun/_shims.py`), so private
names and object identity — the `config` singleton included — are preserved.
eldarion, `eregion_experiments` and eregion-synthesis pin these paths and need
no edits; tharkun's own code imports the right-hand side.

| Old path | New home |
|----------|----------|
| `tharkun.condition_solver` | `tharkun.solve.condition` (loop-exit functions: `tharkun.solve.loop_exit`; `_find_branch_at_line`, `_free_names`, `_unparse`: public names in `tharkun.solve.guard`) |
| `tharkun.guard_decomposer` | `tharkun.solve.decompose` |
| `tharkun.attr_tracer` | `tharkun.solve.attrs` |
| `tharkun.constants`, `tharkun.types`, `tharkun.config` | `tharkun.solve.constants`, `tharkun.solve.types`, `tharkun.solve.config` |
| `tharkun.canonicalize`, `tharkun.rewrite_catalog` | `tharkun.solve.canonicalize`, `tharkun.solve.rewrite_catalog` |
| `tharkun.import_resolver`, `field_resolver`, `constructor`, `type_hints`, `type_propagation`, `structure_infer`, `proxy`, `exception_feedback`, `call_walker`, `call_mapper` | `tharkun.solve.frontend.<same name>` |
| `tharkun.minleaf.guard_extract` / `guard_normalize` / `guard_trace` | `tharkun.solve.guard` / `solve.normalize` / `solve.guard_eval` |
| `tharkun.minleaf.types.VarFacts` / `FactEnv` (still exported there) | `tharkun.solve.fact_env` |
| `tharkun.minleaf.types.AtomEval` / `GuardTrace` (still exported there) | `tharkun.solve.guard_eval` |
| `tharkun.branch_solver._load_module_for_execution` (still exported there) | `tharkun.solve.loader.load_module_for_execution` |

The package root `tharkun` (46 names) and the facade `tharkun.solve` are held
backward-compatible across minor versions; `tests/test_public_surface.py` pins
both and every consumer path above.

## The claim API (0.2.x, #0611)

`tharkun.solve.guard_facts(source_path, lineno, *, facts=None,
desired_branch="then", witness=False)` is the core's answer for a guard
anchor: a `GuardClaim` whose `env` is the prior `FactEnv` refined by every
atom the fact arithmetic models (`name <op> const`, `name in (...)`,
`name is None`), joined across `or` branches, with everything else left
untouched — so `sound_superset` holds by construction and `unmodelled_atoms`
says how much was left on the table. `domains` gives one `IntDomain`
(`lo` / `hi` / `values` / `excluded`, with a `kind`) per variable;
`feasible` is False when the atoms contradict. Despite the name, `values` and
`excluded` carry a *homogeneous* set of any scalar type — `str`, `bytes`,
`float`, `bool` as well as `int` — so `s == 'POST'` and `s in ('GET', 'POST')`
are claims, not gaps; `lo` / `hi` stay integer bounds. A set of mixed types, or
one containing `float('nan')`, is declined rather than approximated (set
equality does not hold for NaN). A `witness` (one satisfying
assignment from the static solver's search, possibly via `hypothesis` regex
strategies) is reported separately and never folded into a domain.
`refine` / `join` in `tharkun.solve.fact_env` are the arithmetic underneath,
and `condition._try_intersect_int_constraints` computes its interval through
the same `refine`.

## Worked examples

**A loop-exit pin (solving, eregion's live call).** `loop_exit_factenv(path,
lineno)` reads the loop enclosing `lineno`, walks every `break` and records the
constant each loop-carried variable is assigned on that exit. For a `while
True:` state machine whose every arm ends in `q = 2; break`, it returns
`FactEnv(facts={'q': VarFacts(eq={2})})`. No output was observed, nothing ran;
eregion consumes the claim by dropping tautological `q == 2` backstops from the
inverter it emits.

**A guard domain (solving).** `solve_condition(path, lineno)` on `if 10 < x*3+1
< 40:` infers `x: int` and computes satisfying values from the interval
`4 <= x <= 12`. Under `no_targets()` the answer is the same and the module at
`path` is never imported.

**An `isinstance` guard on a user class (solving, with a loader fallback).**
`solve_condition` on `if isinstance(x, Foo):` infers `x: Foo`. To hand back a
*concrete* `Foo` it may load the module through `solve/loader.py` and call the
constructor; under `no_targets()` it returns the type-inferred answer without
loading anything.

**Reaching a line (observation, above the core).** `drive_to_line` traces the
target with `trace_api`, finds the blocking guard, asks the core (and, in turn,
static solving / snapshots) how to flip it, and re-traces. Every step that runs the
target is in `driver` / `branch_solver` / `guard_analyzer`; the core only ever
sees predicates.

The SV obligation stage uses `sv/hybrid.py` as its single bridge into the FUZZ layer. Its candidates still pass through SV fresh-process replay.
