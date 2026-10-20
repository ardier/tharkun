# Changelog

All notable changes to tharkun are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Facade stability.** The package root (`from tharkun import X`, 46 names), the
solving facade (`from tharkun.solve import X`) and every pre-0.2.0 module path
listed in the move map of [docs/LAYERS.md](docs/LAYERS.md) are held
backward-compatible across minor versions: the old paths are *permanent*
aliasing shims, so deep imports — private names and object identity included —
keep working. Downstream consumers (eldarion, `eregion_experiments`,
eregion-synthesis) pin them. `tests/test_public_surface.py` is the contract.

## [Unreleased]

### Added

- [internal ref]: draws inside the function a program ends by calling, and at module level, are lifted into the SV harness, and the exact
  product proof (`symbolic:product`) accepts them. The real Tier A tasks of sv-benchmarks !1787 (`def main(): x = nondet_int();
  if not (lo <= x <= hi): return; assert ...` then `main()`) were all "no harness: no liftable draw", so the symbolic stages
  never ran. `sv/sites.py::_plan` picks the harness code: the body of the one function the program ends by calling (the
  `call_target` fact must prove the name a `user_function`; arguments, a second call, recursion, a use as a value, a generator,
  decorators, `global`, statements after the call: each a NAMED refusal, `sites.py::harness_refusal`), or the module-level
  statements when one of them draws (`SIZE = 1 + abs(nondet_int())`; refused by name where moving them would change the program),
  or the `__main__` block as before. `_symexec.py::_Runner.run_body` runs the harness as a function body (an early `return` ends
  the path). `sv/product_proof.py` / `sv/exact_subset.py`: a draw inside a def is a fresh term at every execution; chains over
  total operands, procedures (bare `return`, falling off the end), `int`/`bool`/`object` annotations, `int(x)` and the
  `__main__` guard are exact; the Z3 timeout follows the budget. At 27 s CPU per pair the 408 Tier A pairs went from 358 correct
  (50 UNKNOWN) to 383 correct (25 UNKNOWN), no wrong. Docs: [docs/SV.md](docs/SV.md) "The concolic stages" and "exact product
  proof". `tests/test_sv_entry_lift.py`, `tests/test_sv_product_proof.py`.

### Changed

- [internal ref]: add every string method, formatting syntax and regex sibling as shared rules. Both engines consume finite builtin calls; absint preserves exact string contents and lengths, and symexec models exact symbolic membership, prefix/suffix and empty-spec formatting. Other symbolic paths and oversized native folds refuse by name. Preserve CPython 3.14 conversion-before-spec ordering and malformed-call errors; commit fixed native tables, expanded runtime audit and failure classifications.

- [internal ref]: complete the whole-registry Rule migration; preserve all 234 sv_2027 contracts byte for byte, remove plain-contract registration, dispatch existing engine consumers through visitors, and derive the engine audit from visitor support plus runtime probes. Preserve a temporary set_ re-export for [internal ref]. Commit full-pipeline comparison outputs and explain volatile run counters/component races and all three campaign receipts with isolated, replay-confirmed gained-pair noise controls.

### Fixed

- Child-model soundness ([internal ref], [internal ref], [internal ref], [internal ref], [internal ref]; the grader mirrors each in `eregion-sv`). WRONG FALSE ([internal ref]): a
  program's own `class IndexError(Exception)` raised uncaught was an `IndexError` violation (the child recorded the MRO by
  name); the record now holds only the MRO classes that ARE builtin objects (`sv/_child.py::exception_mro`: the engine
  child, the stationary probe, the in-process drive, the Atheris worker, `eregion_sv.replay`). WRONG TRUE ([internal ref]):
  `random.randint(0, 2**100)`, `randrange(2**63)`, `randint(0, 2**64)` and the like died with `OverflowError`
  (`len(range(...))`) before any draw and answered "deterministic: one execution"; the draw count is arithmetic now and an
  exception OUR model raises that CPython would not sets `model_error`. WRONG TRUE and FALSE ([internal ref]): `random.getstate()`/
  `setstate()` are unmodelled: a `setstate`, or a `getstate` after a random draw, sets `random_state_unmodelled` (never a witness, never a certificate); a `getstate` before any draw reads the native OS-seeded state and sets `random_state_read` (no TRUE, a witness may stand); an OS-seeded `random.Random()` and `SystemRandom` are flagged like `random.seed()`. WRONG TRUE ([internal ref]) and
  FALSE ([internal ref]): the child ran the program through `runpy.run_path` inside `main()` -- five extra frames against the
  recursion limit (`f(994..998)` then `assert False` raised `RecursionError` in the child, `AssertionError` natively),
  `__package__ == ''`, `__loader__` None, `__builtins__` a dict, `sys.path` holding `tharkun/sv` (`import spec` found ours),
  `sys.argv` different in the probe -- it now runs a native `__main__` at native recursion depth (docs/SV.md "How it
  decides"); a program that reads the stack ABOVE its module frame (`inspect.stack()`, `traceback.extract_stack()`, a
  `sys._getframe()` walk, however the name is spelled: a string, a folded `"_get" + "frame"`, `vars(sys)`, `exec`, a computed
  `getattr`) or runs code on a thread (`threading`, `concurrent`, ...: a thread starts at recursion depth 0, so the limit
  the child raises for the main thread gives it ~10 extra frames) is flagged `stack_unmodelled`
  (`solve/environment.py::stack_reason`): never a witness, never a certificate. `tests/test_exception_identity.py`, `tests/test_random_model.py`, `tests/test_native_main.py`, the
  `eregion-sv` `tests/test_replay.py`.
- absint: a subscript of an abstract str is proven again ([internal ref], the precision #151 lost). `AbsintVisitor._abstract_string` had no case for
  `str.__getitem__`, so every `s[i]` / `s[a:b]` of a `nondet_str()` left `TypeError` and the other classes unproven (five MIN mini tasks
  lost `dynamic-typing`; slices were refused). The case reads the key (`visitor.py::str_key_parts`): an exact int / bool proves every class but the
  `IndexError` of the index; an int-bound slice proves them all; a key that runs no user code but is not an int (a float, str, None, a container, a
  slice with a float bound) is a certain `TypeError` and nothing else, so a program that raises it natively keeps `arithmetic`, `data-lookup` and
  `assertion-safety` TRUE (179 cells MG found lost against `ec3cee0`); a key that may be an int or not leaves the `TypeError` and the `IndexError`;
  an object, unknown or static key is refused. `tests/test_absint_str_getitem.py`; see [docs/ABSINT.md](docs/ABSINT.md) "A str subscript through the rule".
- [internal ref]: admit internal Ref and encoded Z3 operands by exact type without target attribute lookup; refuse unhashable string slice bounds without an internal error; preserve existing user-object conversion/format refusal diagnostics across shared string visits; refresh the genuinely consumed Match.groups absint audit cell. Fix two new wrong TRUEs from MG #151 review. Percent-format safety requires an exclusively single tuple reference; scalar/static alternatives stay unproven. Native/abstract regex attribute reads admit only declared methods or represented readonly native fields, so missing attributes refuse; Match.groups aliases select the method Rule. Commit fixed native before/after controls and complete failure classifications; finish moved-module docstring citations.
- [internal ref]: fold MG #147 follow-ups into the shared rules: keep draws.range_kwargs on the closed numeric range table, dispatch symbolic math values/calls only to their matching kinds, restore eight open/random arity twins, and move stale module citations with their files.
- [internal ref]: preserve native RegexFlag enum alternatives and finite operator semantics. On sv_2027, `assert ~re.IGNORECASE == -3` was wrongly proved TRUE although CPython returns 509; the corrected branch leaves the assertion unproven. Preserve symbolic constant/alias conversions and refuse symbolic enum bitwise results without an exact encoding. Commit native main/head controls and before-fix never-TRUE regressions.
- absint: two wrong TRUEs in the `random` models ([internal ref], [internal ref]). (1) A call with a keyword, or `seed(a, version)`, was an arity error that
  ended the path: `x = random.randint(a=0, b=5); c = 1 // 0` was TRUE for `ZeroDivisionError` and raises it; same for `uniform(a=, b=)`,
  `choice(seq=)`, `seed(a=)`, `seed(1, 2)` and `_sv_verifier.nondet_float_range(a=, b=)`. `Analyzer._bind_call` binds a call as CPython does
  (by position or name; a call that does not bind is a certain, recorded `TypeError`), `rand.py::RandomMixin` holds the models, and a
  user object or an unknown value among the arguments is a named refusal. (2) `random.uniform(a, b)` was the interval `[a, b]`; CPython
  computes `a + (b - a) * random()`, which is `inf` for `uniform(-1e308, 1e308)`, `nan` for an infinite bound, and an `OverflowError` for
  an int too large for a float (`assert y <= 1e308` was proven and fails on every draw). It is now that formula, over the analyser's
  float operators. The rest of the random family (`triangular`, `gauss`, `normalvariate`, `lognormvariate`, `expovariate`, `betavariate`,
  `gammavariate`, `vonmisesvariate`, `paretovariate`, `weibullvariate`, ...) is refused by name; `randrange`, `sample` and `shuffle` stay
  contract-driven (no value). `tests/test_absint_random.py`, `tests/test_absint_call_shapes.py` (every other model's call shapes);
  see [docs/ABSINT.md](docs/ABSINT.md) "The `random` module".
- absint: an unhandled kind of value is a named refusal, never an infeasible path ([internal ref], the class [internal ref] and [internal ref] belong to). Eleven
  operations treated a kind they do not model as "bad" and ended the path, so everything after them was unchecked: unpacking an array
  and `random.choice` of an array (now exact), a str subscripted or sliced by a non-int, `dict[slice]` (a `KeyError`, not a
  `TypeError`, since 3.12), a float times a sequence, a static object `in` a str, an attribute store on a list, tuple, dict, range,
  array, iterator or pattern, a call of an exception instance (it was the class's own value), a module attribute no table models
  (`math.foo`), a dict view whose dict was collected (a `KeyError` crash out of `prove`), and iteration over a `StringIO` (an empty loop
  whose body was never checked). `tests/test_absint_kind_matrix.py` runs every kind into every consumer (native runs in capped children),
  `tests/test_absint_kind_refusals.py` has a regression test per site; see [docs/ABSINT.md](docs/ABSINT.md) "An unhandled kind is a
  refusal".
- absint: two wrong TRUEs of sv_2027 `4e129af`, found while auditing the consumers of the heap kinds ([internal ref], [internal ref]). A slice assignment
  from a range, a dict, a dict view or an array (`xs[0:1] = range(0, 1)`) skipped the source's elements (the path went on with the
  list's old ones: `10 // xs[0]` was proven safe and raises `ZeroDivisionError`), and a bound method of an array (`m = a.append`)
  ran the list's `append`, which skips the typecode (`m(0.1); assert a[0] == 0.1` on an `'f'` array was proven and is false).
  `Analyzer.slice_store` joins the elements of every iterable source; `Analyzer._call1` dispatches a bound method by the kind of its
  receiver. `tests/test_absint_slice_source.py`, `tests/test_absint_bound_array.py`, [docs/ABSINT.md](docs/ABSINT.md) "Two wrong TRUEs".
  `scripts/absint_fuzz.py` has the alarm of its fixed copy (a generated loop could grow a list exponentially and hang a shard) and
  draws these shapes.
- File-system mutations are confined to the scratch directory, in the engine's child AND the hybrid worker (bd
  [internal ref], safety). `solve/environment.py::install_mutation_confinement`: every `os` mutator (`remove unlink
  rmdir mkdir mkfifo mknod chmod lchmod chown lchown chflags lchflags truncate utime setxattr removexattr chroot
  rename replace link symlink`, on `os` and `posix`) whose path is not under the scratch (symbolic links resolved)
  raises `OSError(EROFS)` and sets `file_effect`/`file_unmodelled`; `dir_fd` and links out of the scratch are refused;
  everything built on them (`os.makedirs`, `shutil.rmtree/move/copytree`, `pathlib.Path.unlink/rename/mkdir/...`)
  is confined with them. Before, a replay (or the Atheris worker, which runs the program for real, millions of
  times) could delete, rename or re-permission a file outside the scratch. `tests/test_sv_fs_confinement.py`, [docs/SV_IO_MODEL.md](docs/SV_IO_MODEL.md).
- Process spawning (`os.system`, `subprocess`, `os.exec*`, `os.fork`, `multiprocessing`, etc.) is also confined inside the program under test: every call raises `OSError(EPERM)` and records the same flags (`solve/environment.py::install_spawn_confinement`, safety).
- The hybrid worker runs the program under the engine's environment model ([internal ref], safety). The Atheris
  worker executes the program FOR REAL, millions of times, but without the model the engine's child runs it under
  (`solve/environment.py::install_files`: declared files only, every write confined to a scratch copy, anything else
  refused and flagged), so a program that opened, truncated or wrote a file it does not own did exactly that to the
  host: on sv_2027 a `verify` with an 8 s budget on `io.FileIO(outside, 'wb')` truncated a file outside the scratch
  (`tests/test_sv_environment.py` only passed because its 3 s budget kept the stage's own budget under the 1 s the stage
  needs to start the worker). `_hybrid_worker.py` now installs the same model (the task's declared files come through
  `Context.files`), and `AtherisBackend` starts the subprocess with stdin on `/dev/null`, like the engine's child.
  `tests/test_sv_hybrid_sandbox.py`.
- libFuzzer artifacts no longer litter the launch directory ([internal ref] hygiene, [internal ref]). Atheris/libFuzzer
  write `crash-*`, `leak-*`, `timeout-*`, `oom-*` and `slow-unit-*` (and `fuzz-N.log`) to `-artifact_prefix`, which
  defaults to the subprocess's CURRENT DIRECTORY, and the hybrid stage never set it, so they landed in whatever
  campaign results, harness, scratch or clone directory `tharkun.sv` was launched from.
  `hybrid/backend.py::AtherisBackend` -- the one place Tharkun starts Atheris -- now always passes
  `-artifact_prefix=<dir>/`: the caller's `artifact_dir`, or a private temporary directory it creates at `start()` and
  removes at `stop()`; `sv/hybrid.py` runs its worker in its own scratch directory with `artifact_dir` inside it, so
  the stage's `TemporaryDirectory` cleanup removes them. `tests/hybrid/test_backend_artifacts.py` plants a crash
  (a raising harness, and a worker-killing segfault through the SV stage) and asserts nothing is left in the cwd.

### Added

- The `call_target` fact and user classes in the symbolic executor ([internal ref], [internal ref]). `solve/call_targets.py::call_targets`
  publishes one Evidence per call site of the entry module (`target`: user class / function / method, imported, builtin,
  unknown; definition site; `proven` only from a proven `module_constant` and no local binding), keyed by `(scope, callee)` for
  consumers because the harness and the slice rewrite line numbers. Consumer 1: `sv/_symexec_objects.py` gives
  `_symexec.py::_Runner` construction, instance and class attributes, methods, `@staticmethod`/`@classmethod`, single
  inheritance with `super()` and `isinstance` over a per-path copy-on-write heap (forked and merged at every fork of the
  executor), against the LIVE classes of the harness module; everything else of the class protocol, and every skipped
  statement that may have written an object, is a named refusal (never a guess). Class support is search-only:
  `run_exact` gets no records and `product_proof.py` still refuses a `class`. Consumer 2: the stationary proof follows a trip
  value into a proven user class's `__init__` (`solve/drawflow.py::_Planner._ctor`). richards and raytrace no longer stop at
  `call Richards` / `call Canvas`. `tests/test_call_targets.py`, `tests/test_symexec_classes.py`, `tests/test_drawflow_ctor.py`,
  a negative control that re-runs the heap tests with one fix undone, `scripts/class_differential.py` (symbolic against
  native on generated programs; disabled, nothing runs it),
  [docs/SV.md](docs/SV.md) "The `call_target` fact".
- absint decides `isinstance` against user classes, `list`, `tuple` and tuples of classes ([internal ref], `Analyzer.b_isinstance`): a
  scalar by its kind, an object by its single-inheritance class chain and heap kind, a function or module as an `object`
  only; an unknown classinfo, or an object with its own `__class__`, still refuses. richards x4 moves from `isinstance`
  (line 289) to `%`-formatting (line 253); chaos's `isinstance(other, GVector)` and scimark's `isinstance(idx, tuple)` are
  modelled. See [docs/ABSINT.md](docs/ABSINT.md) "isinstance".
- absint: `State.rewrite_bounds_away` and `shift_bounds` skip a variable the zone does not have ([internal ref]): 77% of the calls of a
  chaos run named one, none changed a value, and each walked every value of every frame and object. The walk is one pass
  instead of two. chaos reaches the same point in 157 s of CPU instead of 260 s; no verdict changes. The invariant (no
  bound or term names a variable the zone lacks) is a test over the absint fixtures. See
  [docs/ABSINT.md](docs/ABSINT.md) "Where the time goes".
- absint: a call's operands are not held for the callee ([internal ref], `Analyzer.eval`/`call_func`): an expression's operands
  are held only while that expression runs (a scope per `eval`), and the operands of a user call are released for the
  callee's duration (they are in its frame, which `State.redirect` rewrites). So `d[i] = d[i].combine(d[i + 1], ...)`,
  where `combine` builds a new object of its own class at one site (chaos's de Boor loop, `p = p.inc()`), is no longer
  refused as "allocation would invalidate an in-flight heap reference"; an earlier operand that waits while a later one
  allocates at its site still is. chaos x4 moves past it and reaches the analysis time budget. See
  [docs/ABSINT.md](docs/ABSINT.md) "In-flight references".
- absint models the iterable-reducing builtins as one family ([internal ref], `solve/absint/reducers.py`): `min`/`max` over
  a list, tuple, array, range, dict or str (with `default=`, `key=None`), `sum(it, start)` (exact for integers, an
  unknown float for floats), `any`, `all`, and the `reverse=`/`key=None` keywords of `sorted`; contract-backed
  (`any` and `all` get tested contracts in `solve/contracts/builtins.py`), the same family in `sv/_symexec.py`. A
  key function, an iterator and an element that is an object are refused. `Unsupported` moves to
  `solve/absint/errors.py` (same class). See [docs/ABSINT.md](docs/ABSINT.md) "The iterable-reducing builtins".
- absint models `_sv_verifier.nondet_float_range(a, b)` with the grading shim's own domain ([internal ref], bd
  [internal ref]): the non-NaN floats in `[min(a, b), max(a, b)]`, bounds inclusive, infinities included, a
  NaN bound as `min`/`max` treat it (`a` NaN accepts nothing, `b` NaN accepts exactly `a`); a call with no
  accepted draw ends the path. Float literals are exact (`FloatFact.const`, they were the generic `FLOAT`) and
  a float comparison decides a branch when the operands' facts leave it no pair (`values.py::refine_compare`,
  NaN-aware, int against float exact). See [docs/ABSINT.md](docs/ABSINT.md) "Float draws, float literals and
  float comparisons". Never-TRUE gate and the unit suite unchanged.
- absint: a dict key is a set of constants when the zone bounds it ([internal ref], `Analyzer.key_consts`): a name bound to a constant
  (`EMPTY, WHITE, BLACK = 0, 1, 2`) or a loop variable over a small range is a key, so `SHOW = {EMPTY: '.', WHITE: 'o'}`
  (the `go` benchmark, line 20) is no longer "dict literal with a non-constant key". `go` x3 moves on to `set()` (line 147), a
  heap kind absint does not have. See [docs/ABSINT.md](docs/ABSINT.md) "Dict keys from the zone".
- Project import and proven module constants (#1132b, [internal ref]). `solve/loader.py` gains the real import of a
  project module as a member of its own package: `project_root_and_module` (static; `avengers/backends.py` delegates to
  it), `load_project_module` (the real `importlib.import_module` in the calling process, `__init__` chain in order,
  a same-named cached package evicted and restored, every call through the `ConfinedModule` facade under the caller's
  confine context) and `import_project_isolated` (the same import in a disposable child with the environment model
  installed first). `solve/environment.py::confined` is the reversible, nestable, process-global form of `install_files`
  these use: it snapshots and restores every attribute the installer rewrites (`builtins.open`, `io`, `os.*`,
  `posix.*`) and the cwd. `solve/module_namespace.py` (`module_constants`, `regex_globals`, exported by `tharkun.solve`)
  folds a program's import graph statically: a module-level name is PROVEN constant when nothing rebinds it (a second
  binding, a `global` store, a cross-module attribute store through an import alias, `setattr`/`__dict__`/`globals()`
  writes, a mutation of a mutable value, `exec`/`eval`, a dynamic import), with the evidence when it is not; a compiled
  regex is reported with its folded pattern, explicit and compiled flags and every binding. The SV scheduler gets one
  component, "static facts (module namespace)" (`sv/scheduler.py::publish_module_namespace`), that publishes both into
  the run's store as the Evidence kinds `module_constant` and `regex_pattern` (registered in `sv/contracts.py`,
  docs/SV.md "Static facts: the module namespace"); it decides nothing, costs no budget and a failing analysis publishes
  nothing. `tests/test_solve_project_loader.py`, `tests/test_sv_confined.py`, `tests/test_module_namespace.py`,
  `tests/test_sv_static_facts.py`.
- The search stages see the code a product ships ([internal ref], [internal ref]): `sv/reach.py` follows the program's
  imports to the task's own source files and builds a static call graph over them (what the program CALLS, not
  everything a package imports); `sv/hybrid.py` instruments those packages in the Atheris worker
  (`instrument_imports`), gives libFuzzer a dictionary of the guards' tokens (`startswith`/`endswith`/`split`/`in`/`==`
  arguments, pattern literal runs; 64 entries at most), and seeds the corpus with the called code's literals, the
  shortest strings its regexes accept, and those samples composed with the guards' tokens (`"a"@bc.de`). On the
  python-string-utils mutant m131 (`is_email('"a"@bcd.com')`) the stage found the kill in 8-11 s where the
  product-file-only fuzzer found nothing in 60 s; see [docs/SV.md](docs/SV.md) "What the search stages know about
  the code a product ships".
- Budget plumbing ([internal ref], [internal ref]): a stage that runs dry never ends a `tharkun.sv` run by itself.
  `sv/ledger.py::Goal` (what the run is for: one confirmed violation) and `Ledger.stopped()` are the one stop
  condition every stage reads; when the small-first enumeration drains its finite `str`/`int` candidates with budget
  left, `scheduler.py::run` hands the rest to the stages that can keep searching an unbounded draw domain
  (`stages.py::Stage.deepen`; today `hybrid.py::deepen`, Atheris streamed from a persistent corpus, its CPU on the
  ledger, no line tracer; only for a program that drew something and raised no environment flag) in rounds, until the
  goal is met or the budget/cap says stop. Before this, a product
  program over a library ended UNKNOWN after ~100 runs and ~6 s of a 900 s budget ("unbounded draw domain").
  See [docs/SV.md](docs/SV.md) "Order and the CPU-second ledger".
- Rules and visitors ([internal ref], [internal ref]): `solve/contracts/rules.py` is the shape of the whole contract
  registry -- one `Rule` class per callable (the `Contract`, the call `Shape`, the native semantics, `accept`), one
  `Visitor` per engine with a `visit_<kind>` per rule kind and a NAMED `Refused(engine, qualname, reason)` for a kind
  it has no method for (`not_applicable` for what cannot apply by design), `implements()`/`cell()` so which engine
  consumes which rule is static, `register_rule` (which also registers the Contract, so `get(qualname)` and the
  adapters keep working). Behaviour-preserving base only; the existing contracts migrate onto it in [internal ref]. See
  [docs/contracts/FORMAT.md](docs/contracts/FORMAT.md) "Rules and visitors".
- Output files ([internal ref]): `solve/absint` and `sv/_symexec` model `open(..., <constant write-only mode>)`, `with` on its
  handle and the write side of the handle (`write`, `writelines`, `flush`, `close`, `__enter__`, `__exit__`),
  from one set of contracts (`solve/contracts/fileio_.py`: `open.output`, `io.TextIOWrapper.*`,
  `io.BufferedWriter.*`). Reads stay unknown; `tell`/`seek` and attributes are not modelled; bytes values have
  no abstract model, so a binary handle is provable only for an `array` argument. absint's call string now
  names a call by its whole node: `a + b + c` with a user `__add__` was refused ("allocation would
  invalidate an in-flight heap reference", raytrace's `eye.vector + xcomp + ycomp`) because both `BinOp`s start
  at `a`. No verdict changes on the Real-world `_nondet` tasks; raytrace x4 and chaos x4 move to their next
  blockers. See [docs/ABSINT.md](docs/ABSINT.md), [docs/SV_IO_MODEL.md](docs/SV_IO_MODEL.md).

- `math` contracts ([internal ref] batch 5): every function CPython 3.14.4 defines (57) and every constant (5) has a
  tested contract, TOTAL over the seven property classes (`solve/contracts/rules_math.py`; registry 168 -> 219 with the
  audit's `protocol.length_hint`). absint models all of them (`Analyzer._math_call`), and the symbolic executor
  folds on concrete numbers, z3-encodes the 20 exact ones and refuses the rest with a reason
  (`sv/_symexec_math.py`); one hand-verified table (`solve/contracts/adapters/math_model.py`) is read by both.
  Testing found one wrong batch-1 claim (`math.log(x, 1)` raises `ZeroDivisionError`).
- The contract x engine audit ([internal ref] / [internal ref], `docs/contracts/REPORT.md` batch 5 part B): every one of
  the 219 contracts against absint, the symbolic executor, the exact product, the stationary planner, the
  census/obligations, Avengers, the draws and Eregion, MEASURED (`tests/contracts/_engine_audit.py`), frozen
  (`_engine_audit_expected.py`), embedded in the report verbatim, and backed by an end-to-end test of every
  consumed cell against CPython (`test_engine_audit_consumed.py`) plus an arity-twin soundness matrix
  (`test_engine_audit_twins.py`). Findings: Eregion consumes no contract; absint and the symbolic executor model
  55 and 18 contracts by hand; the ranked step-2 gap list is in the report.
- Draw vectors from a producer ([internal ref], 0987): a fact producer may answer with `draw_vector` facts, values
  for the task's own `_sv_verifier` draws that it thinks reach a violation. `sv/eregion_bridge.py::publish_draw_vector`
  is the door (bool/int/float/str entries only, bounded, `str` inside `string.printable`), the vectors are `Evidence`
  of the new kind `draw_vector` (never certified), and the new `producer` stage (`sv/stages.py::_producer`, in
  `DEFAULT` before the concolic stages, nothing at all with no producer installed) READS them back from the store and
  yields them for the scheduler's one replay path: on the unmodified program, in a fresh child, through `found()`.
  The program is never rewritten; a draw outside its domain is refused out of band (`invalid_draw` is set before
  the call can raise or be swallowed), the child now logs the bounds of every `nondet_float_range` call
  (`Outcome.ranges`, beside the draws, never in a witness) and `engine.py::vector_admissible` pre-checks a vector
  against them. A vector cannot make a TRUE (`prove_from_facts` reads invariant facts only). Eregion's producer
  proposes the engine's forward-verified solutions back as vectors (eregion-codegen, `tharkun_producer.py`).
  `tests/test_sv_draw_vectors.py` pins the door, the pre-check, the stage and the attacks (swallowed refusals through
  seven shapes of swallower and nine ways to reach the draw, reflection, flooding, a lying producer).
- absint first iteration on its own, widening delay 0 (A2, owner ruling 2026-10-05; [internal ref]): `interp.py::Analyzer.s_While`
  and `s_For` analyse iteration 1 from the entry state (recording), start the fixpoint from its end state and run the final
  pass from that head; a site is proven only if proven in both passes; the exit is the zero-iteration exit joined with the
  head exit and the breaks; a loop whose first iteration adds nothing is not iterated; the witness's loop-head invariant is
  entry joined with head (kept only if inductive); `WIDEN_DELAY` is now a number of plain joins and 0 (it is part of the
  fixpoint-reuse key); every pass starts like a statement (`fresh_pass`); a site unsafe only in the first iteration keeps
  its residual. Nothing lost on Real-world 80, Solve-only 65, the toys or the fixtures at any delay; delay 0 is the cheapest
  (Real-world statements -18 %, fixtures -29 %, 8,000 random programs -16 % and +389 proofs, 1 lost). See
  [docs/ABSINT.md](docs/ABSINT.md) "The first iteration on its own".
- absint fixpoint reuse (A4, owner ruling 2026-10-05; [internal ref]/`.18`): `interp.py::Analyzer._fixpoint`
  hands back the loop head of an earlier solve (`_solve_fixpoint`) when the same loop is entered again, in the same
  call string and analyzer scratch (float landmarks, allocation epoch, borrowed references, learned globals and extern
  files), from a state `state.py::State.equiv` calls equal (`leq` both ways, plus frame stack, pins and proof text).
  Only while checks are not recorded (inside an outer loop's fixpoint iterations); never in a `_call_catching` scope.
  A hit also applies what the skipped solve changed in the analyzer (temporary counter, epoch, borrowed set,
  landmarks). Every verdict and site record is byte-identical to the pre-change output on Real-world 80, Solve-only 65,
  the 542 toys and 8,000 random programs (x 4 properties, `PYTHONHASHSEED=0`); the 12 Real-world results with nested
  loops take 40 % less CPU, and scimark reaches its next refusal (generators) in 119 s instead of 185 s. See
  [docs/ABSINT.md](docs/ABSINT.md) "Fixpoint reuse".
- `solve/environment.py` ([internal ref], [internal ref]): the shared I/O model classifies a write-only `open()` as
  OUTPUT (`open_effect`: a constant mode with exactly one of `w`/`a`/`x` and no `+`; `OUTPUT_MODES`,
  `READ_MODES`, `OUTPUT_OPEN_RAISES`). `proof_blocker` no longer stops at an output open and still stops
  at every read, update mode, descriptor, rebound or attribute `open`; a program that writes a file and
  reads it back is stopped at the read. `determinism.py::check_open` uses the same classification. What
  an output open can raise (OSError, TypeError, ValueError) stays in the model; absint has no model of
  `open` and `with` and still aborts at them, so no verdict changes. Measured on the 48 Real-world
  `_nondet` pairs: raytrace x4 and chaos x4 move from the environment gate to the next absint blocker.
  See [docs/SV_IO_MODEL.md](docs/SV_IO_MODEL.md).
- `tharkun.solve.lowering` ([internal ref]): exact comprehension lowering. `lower_comprehensions` turns
  list, set and dict comprehensions into the loops they are (fresh renamed loop variables, the first
  iterable evaluated once, order-preserving hoists only, a generator expression only as the sole
  argument of `list`/`tuple`/`sorted`/`"...".join`), refuses every shape it cannot make exact, and
  marks every node of the result with the path and position of the original node it stands for.
  `solve/absint.prove` runs on the lowered program and reports on the original one, so a program with
  a comprehension is no longer an abort (a refused one still is). Measured on the SV pyperformance
  tasks: no pair changes verdict yet (hexiom now stops at `str.splitlines`, chaos at an environment
  blocker). Differential execution tests, a seeded random differential and an absint soundness fuzz
  back it. See [docs/LOWERING.md](docs/LOWERING.md).
- `sv/_symexec.py` ([internal ref] S1): `for x in range(a[, b[, step]])` with symbolic INT arguments, inline or
  held in a name, is unrolled under its guards with Python's exact `range` semantics (iteration `k`
  runs while `a + k*step` is short of `b` in the direction of `step`; a zero step is `range()`'s
  `ValueError` and ends the path; the target keeps its last value). A non-int argument, or a range used
  as anything but a loop iterable, stays `unsupported`. The single-argument fast path, which took any
  symbolic sort and left the target at its 64th value, is replaced by this one. See
  [docs/SV.md](docs/SV.md).
- `tharkun.avengers.backends` restored ([internal ref]): the owner's ruling was misread as "strip
  Pynguin/Ghostwriter from the SV build" when it meant "no OUTSIDE engine under the hood" --
  Pynguin/Ghostwriter stay available as an Avengers-internal, availability-gated backend
  (`backends`/`bench` extras); nothing outside `avengers/` imports the backend module,
  `pynguin` or `black` directly (`tests/test_sv_no_backends.py`).
  - `tharkun.sv.avengers_seed` (the `avengers` SEED stage, `sv/stages.py::REGISTRY`):
    constructs every class an SV task's module defines via
    `synthesize_object(backends=("pynguin",))` and offers back every literal argument a
    validated recipe used as a candidate value, for object-heavy tasks (richards, raytrace,
    go, chaos) whose branches sit behind a class's own methods. Measured on all 80
    (task, property) pairs of the SV-Benchmarks Python Real-world set with the backend
    actually installed and invoked: 0 gained, 0 lost, and a real cost on 19/80 pairs where
    its own search crowded out the concolic stage's run budget. Registered but out of
    `DEFAULT` per the keep-only-if-gain rule; available via `--stages ...,avengers`.
  - `tests/test_layering.py::test_nothing_outside_avengers_imports_py_obj_flatting`: SV, the
    solver and the fuzzing layer reach a flattened object only through Avengers APIs
    (`synthesize_object` / `ObjectGoal`) and the recipes Avengers materializes, never by
    importing `py_obj_flatting` directly -- was already true, now enforced.

- `tharkun.solve.guard_domain` ([internal ref], facts port slice 2): eregion's static guard facts
  (`domain_capture.reduce_domain`, `fact_repr.apply_constraint` / `merge_env`,
  `emit_bw_assume_facts._extract_constraint`) ported onto the Python AST. It is stdlib only and a
  SOLVING module.
  - `refine(test, var, prior)` returns `(on_true, on_false)` as `values.Value`, and
    `refine_env(test, env)` does the same for every name the test reads. It walks `and` / `or` / `not`
    (meet / join), chained compares, and these atoms:
    - exact int and float bounds against a number, with NaN on the false arm;
    - `==` / `in` through `Value.equal_to`, so `0 == 0.0 == False`;
    - literal tuples / lists / sets, str / bytes substrings and `range(consts)`;
    - `c in x`, `is None / True / False`, truthiness, `len(x) <op> K`;
    - lexical str / bytes bounds.

    The world is open: a type the guard does not name stays unconstrained, and a compare that would
    raise `TypeError` empties that type on both arms. A walrus to the variable drops every fact, and
    a call in the test drops the length and membership facts.
  - `domain_of` / `domains` return a `GuardDomain`: the `Value`, the exact finite set (`members`),
    the lexical bounds, eregion's `domain_kind` label, and the residual sub-guards.
  - `atoms()` is the `_extract_constraint` port for `fact_env.refine_all`. `len(x) ==` / `!=` become
    `container_len_eq` / `_ne`. `x is not True` is no longer read as `x != True`.
  - `tests/test_solve_guard_domain.py` checks random guards against concrete values (NaN, ±0.0, ±inf,
    bools, None, str, bytes, containers).
  - `scripts/guard_domain_differential.py` runs eregion's `reduce_domain` against this module over
    every guard assume in the eregion toy corpus. On 1051 cases, tharkun is never narrower. The
    report is `tests/fixtures/guard_domain_differential.json`.
- `tharkun.solve.absint` slice 2 ([internal ref] group C, lane E):
  - list segmentation: a written prefix plus a ghost counter on range loops;
  - scalar facts now live in the shared `solve/values.Value`, and `absint/values.py` is gone;
  - float arithmetic via `solve/values` transfer functions, with value terms;
  - focus/blur of summary receivers on method calls;
  - per-object sum-of-squares invariants proven with the named lemmas TRIG_PAIR and NORMALIZE
    (`absint/lemmas.py`);
  - per-site records (`AbsintResult.records`, `.complete`, `--json`) in the shape shared with the
    concolic and dead-path lanes;
  - `scripts/absint_fuzz.py`, a differential soundness fuzzer.

  Adds `bm_float_nondet` dynamic-typing and arithmetic.
- `tharkun.sv` `proof:no-site` ([internal ref]): `solve/census.py::site_census` shows statically that no operator,
  library call, `assert` or `raise` the program can run raises the property's classes, for any values. An
  allowlist over `solve/pyresolve.py`: reflection refuses. `prove.py::prove_static` runs it first, and the
  deterministic check falls back to it when the determinism certificate fails. New TRUE:
  `bm_fannkuch_nondet` and `bm_richards_nondet` arithmetic, `bm_go_nondet` assertion-safety.
- `tharkun.sv` determinism certificate ([internal ref], a soundness fix): the `deterministic` and `finite_tree`
  TRUE answers now need `solve/determinism.py::determinism_census`, and the verdict's detail cites it.
  Without it the answer is UNKNOWN. Two runs under hash seeds 0 and 1 had answered TRUE on programs that
  raise under another seed, an unseeded `random.Random()`, `SystemRandom`, `os.urandom`, `id()` or
  `datetime.now()` (lane F's probes, now `tests/fixtures/determinism/never_certified/`).
- `tharkun.sv` `proof:stationary` ([internal ref]): TRUE for programs whose unbounded draws only count the
  repetitions of a loop whose state repeats after one iteration. `solve/drawflow.py::plan_draws` shows
  statically that each draw reaches the program only as a trip count (or a residue `abs(d) % M`);
  `solve/determinism.py::determinism_census` certifies, as an allowlist over `solve/pyresolve.py`'s
  resolution, that a run is a function of the program and its draws; the probe child `sv/_probe.py` checks
  that the loop-head states after `m` and `m + 1` iterations are equal as canonical snapshots of the whole
  reachable state. `engine.verify` calls `prove.py::prove_dynamic` after the static stages; its TRUE faces
  the canary. New TRUE: `bm_spectral_norm_nondet`, `bm_unpack_sequence_nondet`, `bm_regex_v8_nondet`
  (3 properties each).

### Added


- `tharkun.sv` stage order ([internal ref] slice 3, owner ruling 2026-09-24): static proofs (lazily imported
  `tharkun.sv.prove.prove_static` and `tharkun.solve.absint.prove`) -> one symbolic run toward every
  site -> execution (+ `prove_dynamic`) -> the portfolio (seed stages; the per-site drive on the sites
  still open) -> enumeration. A proof's TRUE faces a canary of at most 60 s CPU (it was 5% of the
  budget, run after the program's first execution); a violation the canary finds wins and names the
  unsound proof. When every site is solved to "no modelled run fires it", enumeration is capped and
  the answer is UNKNOWN early.
- `tharkun.sv` symbolic execution of the whole harness (`_symexec.py::solve_targets`, [internal ref] slice 3):
  calls inlined, loops with a draw-dependent trip count unrolled to 64 under their guards, streams as
  z3 arrays at path-dependent positions, path-typed values as unions, constant-key dicts, floor `%`/`//`
  by a draw; per-site outcome sat / unsat / unreached / unsupported / unknown.
- `tharkun.sv` typing sites ([internal ref] slice 3): `TypeError` from `a <op> b` and `AttributeError` from
  `x.attr` on `None`, placed only on names assigned values of several types; `build_harness(...,
  extra_sites=...)` turns the abstract interpreter's may-raise records into solve targets and drops
  markers its safe records cover.

- `tharkun.solve.refute_stdlib(residual, scope, types)` ([internal ref], `solve/refute.py`), stdlib only. It
  proves that `residual AND all(scope)` has no model over typed `int` / `float` / `bool` names, and
  it's the Z3-free pre-check for lane B's `proof:dead-path`. Two procedures: exact integer elimination
  (`solve/arith`, which now also runs Fourier-Motzkin on its linear inequalities, so `x < y and y < x`
  and `x // 2 == 5 and x < 10` are refuted), then interval propagation over `solve/values`. The
  propagation handles IEEE floats with NaN, -0.0 and infinities: `not (x < y)` keeps NaN, `e * e` is
  never negative, and `e != e` is only NaN. It answers `(True, certificate)` only when the formula is
  proven empty. Where Python would raise while evaluating a condition, the value is left free, the
  same convention as the Z3 encoding. A random-formula test checks that no refuted formula has a
  model on a grid of ints, bools and special floats.
- `tharkun.solve.values`, requested by absint ([internal ref]). IEEE `+ - * /` are correctly rounded and therefore
  monotone, so their float corners are now exact bounds with no outward step. A sum or product of
  non-negatives keeps its lower bound at 0.0; before, it rounded to -5e-324, which made
  `(dx*dx + dy*dy) ** -1.5` "maybe complex". New transfer functions, each checked against CPython on
  random intervals: `sq` (x*x of one value, never negative), `fabs`, `floor`, `ceil`, `trunc`,
  `round_`, `round_n`, `min_` / `max_` (an int beside a float is unmodelled, since the result keeps
  its own kind), `exp`, `log` / `log10` (exact for ints of any size), `atan`, `atan2`, `tanh`,
  `hypot`, `fsum`. `draw_value(func, args) -> (Value, may_raise)` gives what a nondeterministic call may
  return: `_sv_verifier.nondet_*` as in the reference model (a float can be inf, -inf, nan or -0.0;
  `nondet_float_range` is `random.uniform`), and `random.random / uniform / randint / randrange /
  getrandbits` and the index `choice` picks, with ValueError / IndexError for empty ranges.
- `tharkun.solve.claims`: `len(x) == K` / `len(x) != K` atoms (either side) are now modelled as
  container-length facts ([internal ref], facts port slice 4). Before, they were unmodelled, so
  `len(s) == 3 and len(s) != 3` was not known to be infeasible.
- `tharkun.solve.absint` ([internal ref] group C, lane E): a sound whole-program abstract interpreter that proves
  `! uncaught(E)` statically ([docs/ABSINT.md](docs/ABSINT.md)). It is built from:
  - a relational integer zone (DBM) over variables, list lengths and range bounds, with `!=` bound
    tightening;
  - an allocation-site heap with recency;
  - inlined calls;
  - widening fixpoints with a verified narrowing step.

  Every TRUE lists its checks with the invariant fact that discharges each one
  (`python -m tharkun.solve.absint P.py --property X.prp --explain`). Anything it does not model makes it
  UNKNOWN. `tharkun.sv` runs it through `sv/prove.py` before the search, and a proof there still faces a
  canary search whose FALSE wins (`--no-prove` for ablations). Slice 1 proves `bm_fannkuch_nondet`
  data-lookup and dynamic-typing, and `bm_float_nondet` data-lookup.
- `tharkun.sv` stage 1, second step ([internal ref] slice 2): implicit violation sites (a subscript that can
  miss, a zero divisor) become solver targets via markers in the harness; draws in loops become
  indexed streams; `sv/_symexec.py` solves a site's backward slice as ONE z3 formula (loops unrolled,
  ifs merged), which answers `pick` at N = 16 and 24; the sites are solved in frontier order with
  per-site budget shares; `component=concolic:symexec` and ` solver_calls=<n>`.
- `tharkun.solve.fact_env` catches up with eregion's carrier ([internal ref], facts port slice 3).
  `VarFacts.container_len` / `container_len_ne` and the `refine` ops `container_len_eq` /
  `container_len_ne` mirror `eregion_commons.fact_repr.apply_constraint`: a second, different exact
  length, or an exact length already excluded, makes the path impossible. `join` keeps a length only
  when both paths agree on it, and an exclusion only when both paths have it. `to_value` / `from_value`
  convert soundly between `VarFacts` and `solve/values.Value`. `Value.equal_to(v)` is every value
  `== v` (`0 == 0.0 == False`), the set a guard `x == v` admits.
- `tharkun.solve.values` ([internal ref]): sound non-relational value facts, stdlib only. This is the first
  slice of moving eregion's static value facts into Tharkun, and the carrier lane E's abstract
  interpreter builds on. It adds `IntFact(lo, hi)`, `FloatFact(lo, hi, nan)` (infinities as bounds,
  NaN as a flag), `StrFact(consts | ANY)`, `Kinds` (the possible type names, None included) and the
  product `Value`, each with join / meet / widen / leq / is_bottom. Transfer functions `add sub mul
  truediv floordiv mod pow_ neg abs_ to_float sqrt sin cos` return `(fact, may_raise)`. Float results
  are rounded outward. `ZeroDivisionError` is reported exactly when the divisor may be zero, and
  `OverflowError` / `ValueError` separately. A negative base to a fractional power (a complex result)
  comes back unmodelled (`None`). `refine_compare(op, a, b)` narrows both operands for both outcomes,
  with NaN semantics and exact int / float comparison. The tests check every operation against CPython
  on random intervals: each concrete result must lie in the abstract one, or its exception must be in
  `may_raise`.
- `tharkun.sv` solving portfolio ([internal ref], `sv/stages.py`): the stages the engine runs before its
  enumeration, cheapest first, each with an attribution name. Seed stages rank candidate values right
  after each kind's head values; vector stages propose draw vectors that are replayed. Select them with
  `--stages` / `--no-stage` (`verify(stages=...)`). The concolic stage is registered unchanged. New seed
  stage `guard_values` (component `static:guard_values`, `sv/guard_values.py`): it solves every
  guard, with straight-line definitions inlined and its enclosing `if` tests conjoined, for the draws
  (`solve/arith` for integers, `solve_condition` for the rest). It executes nothing and costs
  milliseconds.

### Fixed

- absint's model of the length probe was wrong in five ways, giving seven wrong TRUEs ([internal ref] batch 5, [internal ref]). CPython's
  `PyObject_LengthHint` (`list(x)`, `sorted(x)`, `list.extend`, `+=`, `bytes`, `bytearray.extend`,
  `operator.length_hint`) calls `x.__len__` and swallows exactly a `TypeError`; the model ([internal ref]) swallowed
  `AttributeError` too, restored the state from before the call (dropping the callee's writes to globals and
  attributes), ended the path of every conditional `raise TypeError`, ended the path on a `TypeError` that native code
  raised inside `__len__` (`len(5)`, a wrong arity, a non-int result), and probed `__len__` for `tuple()` and
  `array.array`, which never ask. `Analyzer._probe_length` now follows an explicit `raise TypeError` exactly (the
  state at the raise joins the continuation), refuses when a `TypeError` can come from an unproven check, and lets every
  other class propagate; which callables probe is a table verified against CPython
  (`contracts/adapters/common.py::LENGTH_HINT_RULES`, gated on the new tested contract `protocol.length_hint`), and an
  unknown consumer of an instance defining `__len__` is refused. Seven regressions in `tests/test_absint.py` fail on
  the old tree; the catalog of 179 swallowing programs (`tests/contracts/_swallowing_cases.py`) is never TRUE on a
  violating program in any engine, and the engine's child classifies every run as CPython does. No program of the
  Real-world, Solve-only, soundness or toy corpora has the shape, so no past TRUE on them was wrong. See
  `docs/contracts/REPORT.md`, batch 5 part C.

- `tharkun.sv`: the concolic worker runs under one child's share of `--run-memory-mb`, not the whole
  of it, so the verifier's processes stay inside the task's memory limit together.
- `tharkun.sv`: the float candidate tail stops before `2.0**e` overflows; it used to raise
  `OverflowError` after about 3390 float candidates.

- `tharkun.sv` solves for violating draws before enumerating them ([internal ref] slice 2, stage 1):
  `sv/sites.py` lists the explicit violation sites and lifts the main block into a harness whose
  parameters are the draws; `sv/concolic.py` runs `driver.drive_to_line` on each site in a capped
  helper process (`sv/_concolic_worker.py`) -- from the harness, or from the function the draws feed --
  and every solved vector is replayed on the unmodified program. Every verdict names its component
  (`tharkun-sv: component=...`; `Verdict.component` / `Verdict.detail`), with the enumeration split into
  `enum:small_first` / `enum:constants` / `enum:tail`. `--no-concolic` for ablations.

### Changed

- The contract registry is rules ([internal ref], [internal ref]): every contract module but the str/re ones (`rules_bisect`, `rules_builtins`,
  `rules_builtins_more`, `rules_bytes`, `rules_collections`, `rules_copy`, `rules_dict`, `rules_fileio`,
  `rules_functools`, `rules_heapq`, `rules_io`, `rules_itertools`, `rules_list`, `rules_math`, `rules_operator`,
  `rules_protocols`, `rules_random`, `rules_set`, `rules_tuple`; the old `*_.py` modules, moved) now holds one
  `Rule` class per qualname (`MathFunction` and `MathConstant` cover the 57 functions and 5 constants, the output-handle
  methods are one class each for the text and the binary handle), with its `Contract` unchanged:
  `tests/contracts/test_registry_digest.py` freezes a canonical digest of all 232 contracts and fails on any difference.
  heapq's and bisect's rules carry the call shape the engines already trust (`adapters/common.py::CALL_SHAPE`, tested
  equal); every other rule has no verified shape and applies to no call. No engine reads a rule yet and no verdict changes.
- [internal ref]: remove the standalone RedQueen driver tier and its keyword flag, including the concolic worker opt-in. Keep live reach-search operand harvesting through `redqueen.py::harvest`.

- The census knows `math`: every function and constant has a closed-call-hooks row, so a program that calls
  `math.tan` (or reads `math.pi`) no longer keeps every class method live.
- absint's generic refusals name the callee (`call of an unknown value: collections.deque`).
- `receiver_may_raise` (Avengers, draws) counts only claims about a call's own logic as empty-value evidence,
  not a dunder's propagated exception.
- `tests/test_layering.py`: `sv/*` follows the DRIVE rule of docs/LAYERS.md (may import every layer at or
  below DRIVE, nothing above) instead of importing only itself; the child stays Tharkun-free.

### Fixed

- `tharkun.sv` parallel children ([internal ref]): a batch now fills from a single candidate chain (each popped
  vector queues its chain's next value before the next pop), so a program with one unbounded draw
  runs `--jobs` children at once instead of one. The CPU budget is also shared: each child in a batch
  may use only its share of what remains, so a batch of long runs cannot overshoot `--budget-seconds`.
  With `--jobs 1` nothing changes. On `bm_spectral_norm_nondet` at 60 CPU-seconds and 4 jobs, the
  budget now buys 13 runs in 16 s of wall clock, against 9 runs in 61 s before.

### Added

- `tharkun.solve.solve_int_guard` ([internal ref], `solve/arith.py`): a native integer solver for guards, stdlib
  only. Linear equalities are eliminated exactly (a unit coefficient is substituted; otherwise two
  coefficients are merged through their gcd with a complete integer parametrisation), `e // m` and
  `e % m` by a constant become linear brackets over an auxiliary quotient, univariate polynomial
  equalities are solved for their integer roots, and the rest is a bounded assignment. Every witness
  is checked against the guard; `unsat` is claimed only when exact elimination refutes every branch.
  `solve_condition` now takes `var_types` and falls back to it (strategy `arith`) when its own
  inference leaves a guard unsolved and every free name is a known integer (`var_types`, an `int`
  annotation in the enclosing function, integer facts, or an inferred `int`; `infeasible` only for
  the first three). `x * 7 - 11 == 3489` used to come back as `x = 0`, which does not satisfy it; now
  `x = 500`. Guards that were already solved directly are unchanged.

- `tharkun.sv` (`python -m tharkun.sv`, [internal ref]): tharkun as an SV-COMP Python verifier. It reads an
  SV-Benchmarks task (`.py` or format-2.0 `.yml`) and an `! uncaught(E)` property (`.prp`), runs the
  unmodified program in fresh child interpreters with every `_sv_verifier.nondet_*` call and every
  unseeded `random` call as a replayable draw, and prints `TRUE` (deterministic program, or a finite
  draw tree enumerated), `FALSE(E)` with an `sv-py-witness/1` witness, or `UNKNOWN`. Stdlib only;
  DRIVE layer. See [docs/SV.md](docs/SV.md).
- `tharkun.sv` search ([internal ref] slice 1, [internal ref], [internal ref]): candidate chains are walked lazily, so int,
  float and str draws are never exhausted while budget remains; the program's own literals are
  tried right after the small values; up to `--jobs` children run side by side (outcomes read in
  cost order); `--budget-seconds` counts CPU seconds of the verifier and its children.
- `tharkun.solve.affine_envelope(expr, var)` (#0911): a PROVEN affine bracket of
  an integer expression in one variable, `a * var + lo <= expr <= a * var + hi`
  for every integer `var`, computed from structure alone (no observed output,
  nothing executed). An abstract interpretation over "affine plus bounded
  residual" covers `+ - *`-by-constant, `~`, `^ & |` with a constant, `%` and
  `//` by a constant, shifts, literal-sequence subscripts, and `min` / `max` of
  one slope. For example, `x ^ c - x` lies in `[-c, c]` for `c >= 0`, because
  `x ^ c == x + c - 2 * (x & c)` and `0 <= x & c <= c`. Anything else declines
  (`None`), which is no claim: `x * x`, `**`, true division, other free names,
  calls. eregion uses it to solve a non-monotone relation `F(x) == y` exactly:
  with `a != 0`, `y` leaves a finite interval of `x` to check. Its result is
  `AffineEnvelope`, and `as_record()` gives plain data.
- `IntDomain.of` carries **typed** value sets. `eq` / `neq` facts over a
  homogeneous set of `str`, `bytes`, `float` or `bool` now become
  `values` / `excluded` instead of being filtered away, so `s == 'POST'`,
  `s in ('GET', 'POST', 'PUT')`, `b == b'\x00'` and `x == 2.5` are claims
  rather than "unknown". Integer behaviour is unchanged, and `lo` / `hi`
  remain integer bounds — a non-integer *ordering* bound is still out of
  scope, because `refine`'s strict-`<` subtracts 1. Declined by design, not
  approximated: a set of mixed types, and any set containing `float('nan')`
  (set equality does not hold for NaN).
- `try_extract_constant` folds a call whose callee is a known-pure constructor
  and whose arguments are all literals — `bytes.fromhex('00')`,
  `bytes([1, 2])`, `frozenset((1, 2))` — so a guard written against one is an
  atom rather than an unmodelled shape. Refused outright: a single integer
  argument (a size, not a value — `bytes(10**9)` would allocate a gigabyte),
  any non-literal argument, anything user-defined, a result that is not
  hashable, and a malformed literal (which raises, and is not a fact).
- Boundary-directed goals: `ObjectGoal(..., expect="raises")`,
  `expect="raises:TimerError"` or `expect="crashes"` ask the search to find an
  object that drives the boundary into that behaviour, instead of one that
  avoids it. `ObjectGoal.satisfied(outcome)` decides in the parent, so one
  child result answers a `raises` goal and fails a `returns` goal without being
  replayed twice, and a directed goal's own answers stay out of
  `report.failures`. A default goal's key is unchanged, so every recipe and
  store file minted before this still names the same goal.
- `tharkun.avengers` objectives beyond construction (`objectives=True`): state
  coverage keyed on the replay's state digest, line and branch coverage
  measured in the worker with the module load kept outside the trace, and
  deliberately invalid call sequences (`start(); start()`, a bare `stop()`)
  kept as results rather than discarded as failures. `AvengersReport` gains
  `states`, `negatives`, `coverage`, `line_coverage`, `branch_coverage`,
  `crash_causes`, `time_to_first_object_s` and `metrics`.
- `tharkun.avengers.backends`: passes that RUN Pynguin, CrossHair,
  `st.from_type` and Ghostwriter, all availability-gated, none believed without
  the usual fresh-process replay. `backend_status()` says which are usable and
  why not. Backends run after the cheap passes, because an external generator
  costs seconds before it says anything.
- `tharkun.avengers.NestedObject` and `NestedCandidatePass`: a recipe may carry
  another recipe as an argument, so a class whose `__init__` needs a domain
  object is constructible. `report.best.program` binds each composed object to
  a `nested_N` variable and stays executable Python.
- `tharkun.avengers.TemplateStore`: validated recipes, certified templates and
  negative sequences that outlive one search, read with the worker wire's
  discipline and re-offered as candidates rather than trusted as results.
- `tharkun.avengers.bench` and `tharkun.avengers.stats`: a configuration
  comparison over the ten EMSE-2023 Pynguin subjects, with stdlib
  Mann-Whitney U and Vargha-Delaney A12. `benchmarks/avengers/manifest.json`
  carries two profiles; `scripts/setup_benchmarks.py` builds one environment
  per subject.
- `python -m tharkun.avengers` with `synthesize`, `backends` and `bench`.
- `Avengers(scheduler=)` with `PortfolioOrder` (the default, unchanged
  behaviour) and `YieldOrder`; `PassMetrics` per strategy;
  `SaturationBudget.max_pass_seconds`.
- `IsolatedExecutor(interpreter=)` runs the child under another Python, and
  `replay_many()` replays a batch in one child (~3x faster), falling back to
  one child per recipe on any abnormal exit so a crash is attributed rather
  than smeared.
- `TracePolicy(arcs=True)`: `(from line, to line)` transitions, the measurement
  branch coverage counts. Forces `sys.settrace`, because `sys.monitoring`'s
  LINE callback has no frame and two live frames of one code object would
  invent arcs neither took.
- `tharkun.solve.frontend.constructor` reads the constructor a `@dataclass` (or
  attrs) decorator generates, from the annotated fields in the class body.
  Reading only `__init__` reported "takes no arguments" for every dataclass.

### Fixed

- `tharkun.mcp_server` ([internal ref]): `mcp>=1.0` now resolves to mcp 2.x, which renamed
  `mcp.server.fastmcp.FastMCP` to `mcp.server.mcpserver.MCPServer` and dropped the old
  module, so every `tests/test_mcp_server.py` test failed on import (12 failing). The
  module now imports `MCPServer` first and falls back to `FastMCP` on mcp 1.x -- both
  classes take the same `name`/`instructions=` constructor args and expose the same
  `tool()` decorator and `run()`, which is all this module uses. Verified green against
  both mcp 1.30.0 and mcp 2.2.0.

### Fixed

- `tharkun.solve.loader.file_path_to_dotted_module` took the first matching
  `sys.path` entry, so a target under the current directory -- a virtual
  environment inside a checkout -- was named relative to the checkout and
  produced a name that could not be imported. Every component must now be an
  identifier, and the longest matching entry wins.

### Added

- `tharkun.avengers`: a fixed-point portfolio for constructor discovery,
  setup-sequence search, Pynguin recipe ingestion, and fresh-process validation
  of method or module-function boundaries. `ObjectGoal.from_type()` accepts an
  installed class directly, and every recipe keeps its evidence derivations.
- With no explicit boundary, Avengers probes public zero-argument methods from
  fresh copies of each receiver state, ranks `best` by method coverage, and
  exposes a small multi-state `objects` portfolio for incompatible call
  preconditions such as `Timer.start()` and `Timer.stop()`.
- The optional `tharkun.avengers.flatting` bridge turns validated exemplars
  into byte-addressable `py_obj_flatting` recipes. Avengers and
  `py_obj_flatting` are Tharkun components with independent distribution and
  safety boundaries. It certifies exact exemplar reconstruction, including
  nested constructors, and carries recipes and child-worker results over
  bounded, data-only JSON.
- `tharkun.solve.guard_facts(..., predicate=<source text | ast.expr>)` / `guard_domains(...,
  predicate=)` and `guard_facts(..., assignment="q = 2")` (#0622): the caller hands over the
  guard (or the write to replay) itself, so no path, tree or line lookup happens at all and
  `lineno` is display only -- the anchor for a lowered module whose injected statements share
  line numbers (a pass copying one location onto everything it emits).
- `tharkun.solve.guard_facts(..., tree=<ast.Module>)` / `guard_domains(..., tree=)` (#0622): a
  caller may hand over its own parsed module instead of a path — the tree whose line numbers it
  holds — so an anchor never resolves against the wrong file; `source_path` may then be `None`
  (the witness search still needs a path).
- `tharkun.solve.guard_facts(..., assignments=True)` (#0622): an assignment statement at the
  line is replayed — its targets lose their facts and a constant right-hand side pins the target
  (strategy `"assignment"`) — so a caller can replay a whole path, guards and writes in order,
  and a guard on a variable the path later rewrites never contradicts the write.
- `tharkun.solve.guard_facts(..., assume_calls=(...))` (#0612): a bare call
  statement naming one of the given functions (e.g. `__bw_assume__(cond)`) is
  treated as a guard whose predicate is the call's first argument, and an
  `assert` test is always a guard — so a lowered branch condition can be
  solved, and a whole path of assume statements chained.
- `tharkun.solve.claims` (#0611): `guard_facts(source_path, lineno, *, facts,
  desired_branch, witness)` / `guard_domains(...)` — the sound value-set claim
  for a guard anchor (`GuardClaim` with per-variable `IntDomain`s, feasibility,
  `sound_superset`, `Provenance(backend, strategy, elapsed_ms)`); prior facts
  are refined by every comparison / membership atom the arithmetic models,
  joined across `or` branches, everything else left untouched. A witness is
  reported separately (never as a domain) and searched under `no_targets()`.
- `tharkun.solve.fact_env.refine` / `refine_all` / `join`: the pure fact
  arithmetic, mirroring eregion's `apply_constraint` / `merge_env` on the
  shared ops (property-tested on the eregion side). `condition._try_intersect_
  int_constraints` now computes its interval through `refine`, so the static
  solver and the fact domain share one interval intersection.

## [0.2.1] — 2026-10-20

### Added

- `python -m tharkun.sv --cpu-limit-seconds S --memory-limit-bytes B`: the CPU-time and memory limits of the whole run,
  as BenchExec enforces them. The verifier derives its budgets from them (`sv/__main__.py::effective_limits`): without
  `--budget-seconds` the search stops at `max(5, int(0.9 * S))` CPU seconds, without `--run-memory-mb` the program's runs
  together get `max(64, int(0.6 * B) // 2**20)` MiB. These are the values the BenchExec tool-info module of 0.2.0 passed,
  which now passes the full limits instead. `--budget-seconds` and `--run-memory-mb` keep their meaning and take
  precedence; without limits the defaults stay 600 seconds and no cap.

## [0.2.0] — 2026-09-08

The solving core is fenced into `tharkun.solve` and the layering is
machine-checked (eregion plan #0600, issues #0603–#0608).

### Added

- `tharkun.solve`: the constraint-solving core as a package with a lazy facade
  — `solve_condition`, `solve_with_snapshot`, `loop_exit_factenv`,
  `collect_loop_exit_facts`, `decompose_guard`, `FactEnv`, `VarFacts`,
  `extract_frontier`, `find_branch_at_line`, `free_names`, `normalize_guard`,
  `trace_guard`, `canonicalize`, `config`, `no_targets`, … Importing it loads
  the static solver and nothing else.
- `tharkun.solve.loader`: the one place the core imports or executes a target
  module (`import_dotted`, `load_from_file`, `exec_source`,
  `load_module_for_execution`), with a `no_targets()` context that makes the
  core answer purely statically.
- `tharkun.solve.loop_exit`: loop-exit reasoning split out of the static solver.
- `tharkun.solve.guard`: one home for "find the guard at a line" — the minleaf
  frontier extractor plus the formerly private `find_branch_at_line`,
  `free_names`, `unparse`, `unwrap_named_expr`, `try_extract_constant`.
- `tharkun.solve.fact_env`: `VarFacts` / `FactEnv` (from `minleaf.types`, which
  still exports them).
- `docs/LAYERS.md`: the delineation rule (solving vs inversion vs observation),
  the layer table, what eregion may import, the move map, worked examples.
- Layer-named extras: `search`, `smt`, `kb`, `fuzz`, `tools` (the core needs
  none). The old extras `hypothesis`, `crosshair`, `touchstone`, `egglog`,
  `mcp`, `hybrid` remain for one release as aliases.
- Contract tests: `tests/test_layering.py`, `tests/test_public_surface.py`,
  `tests/test_solve_loader.py`, `tests/test_version_sync.py`.

### Changed

- The package root is lazy (PEP 562): `import tharkun` imports no submodule;
  the 46 public names resolve on first access. `import tharkun.condition_solver`
  loads three modules instead of eighteen.
- Module layout: `condition_solver` → `solve/condition`, `guard_decomposer` →
  `solve/decompose`, `attr_tracer` → `solve/attrs`, `constants` / `types` /
  `config` / `canonicalize` / `rewrite_catalog` → `solve/`, the ten static
  front-end modules → `solve/frontend/`, `minleaf/guard_extract` /
  `guard_normalize` / `guard_trace` → `solve/guard` / `solve/normalize` /
  `solve/guard_eval`. Every old path is a permanent shim (see above).
- `branch_solver._load_module_for_execution` and `_file_path_to_dotted_module`
  delegate to `tharkun.solve.loader` (same three strategies, same order).
- `requires-python` is `>=3.10` (the code has used PEP 604 unions at runtime
  since `minleaf/types.py` landed; the `3.9` claim was false).
- `slicing_bench.default_corpus()` includes `solve/**`.

### Removed

- The last import from the core into a layer above it
  (`condition_solver` → `branch_solver`).

## [0.1.0]

Everything before the carve-out: the technique chain, the directed
reachability driver, the knowledge base, the hybrid fuzzer, the MCP server.
