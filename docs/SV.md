# tharkun as an SV-COMP Python verifier (`tharkun.sv`)

`python -m tharkun.sv` decides one property of the SV-Benchmarks Python track (SV-COMP 2027) for one
program: `CHECK( init(main()), ! uncaught(E))` for each exception class `E` in a `.prp` file. It prints
the verdict as its last line: `TRUE`, `FALSE(E)` or `UNKNOWN(reason)`. The eregion umbrella runs it as a
BenchExec solver column (the `repos/eregion-sv` submodule, tool-info `eregion_sv.tools.tharkun_sv`).

```
python -m tharkun.sv PROGRAM.py|TASK.yml --property P.prp [--python PY] [--budget-seconds S]
                     [--canary-seconds S] [--run-timeout S] [--witness-out W.json]
python -m tharkun.sv PROGRAM.py --property P.prp --replay W.json      # CONFIRMED / NOT CONFIRMED
```


### Target execution confinement

Every SV target execution phase uses the shared declared-file/scratch model and refuses target process
and thread starts. The concolic worker also covers hybrid's plateau handoff; stationary probes and their
fresh confirmations receive the same declared files. Trusted worker launches remain outside reversible
parent phases. Disposable replay, probe, Avengers and Pynguin workers retain confinement through shutdown;
Pynguin's SV path disables subprocess execution.

Reversible exploration conservatively refuses deferred finalizers/callbacks, dynamic code and type/cache
reflection, including reachable local helpers. `nondet_harness.drive` sends such a program to the existing
confined fresh child with a 30-second cap instead. It retains the observed outcome and draws, adds
`import_state_unmodelled` and `stack_unmodelled`, and never claims a target line was reached. Other reversible
routes abstain. These restrictions do not change ordinary fresh-replay admission or independent static proofs.
Concolic execution flags survive obligation-table publication; candidate vectors still require original-source
fresh replay. Trusted harness generation binds the exact generated text to the original unsliced source and
its path. The worker and driver reject changed harness bytes before execution and apply reversible admission
to that original source and its import graph, without treating the generated helper prelude as target code.
Original references or bindings to the prelude's injected globals are refused before target execution.
Pynguin collects generated tests from a separate scratch subdirectory so staged inputs are never artifacts.
The native witness oracle shares the disposable child boot and checks builtin identities after its guards
are installed. Its parent owns each scratch directory and removes it only after child termination, so
confinement remains installed through target cleanup and interpreter shutdown.
This is the existing execution model, not a general Python security sandbox.

## How it decides

The program runs **unmodified**, as `__main__`, in a fresh child interpreter per run
(`src/tharkun/sv/_child.py`, started by `src/tharkun/sv/engine.py::run_once`). The child
installs a model of `_sv_verifier` in which every `nondet_*` call is a **draw** from a replayed
vector.

**The child's `__main__` is the native one** ([internal ref], [internal ref]; `_child.py::native_sys_path`, `native_main_module`,
`run_as_main`, shared by `_probe.py`, `nondet_harness.py::drive` and, through `main`, the fork server, and mirrored by
the grader's boot in `eregion-sv/src/eregion_sv/replay.py`): `__package__` None, a real `SourceFileLoader`, `__spec__`
None, `__builtins__` the builtins module, `sys.path[0]` the program's directory and never `tharkun/sv` (so `import spec`
is the program's own module or a `ModuleNotFoundError`), `sys.argv` as natively, and the recursion limit raised by the
child's own frame count (with `sys.getrecursionlimit`/`setrecursionlimit` shown at their native-depth values) so the
program's module frame sits at native depth 1: `f(996)` reaches the `assert False` that follows it, as under
`python program.py`. Not native, by construction: (1) the Python frames BELOW the module frame (`inspect.stack()`,
`traceback.extract_stack()`, a `sys._getframe()` walk, `f_back`): a program that reads them still sees the child's own
frames; (2) a THREAD: the recursion limit the child raises is the main thread's, and a thread starts its own stack at depth
0, so a program that counts recursion depth in a thread gets ~10 more frames than natively. So
`solve/environment.py::stack_reason` (a static, over-approximate scan like `references_stdin`) flags such a program
`stack_unmodelled`: one that reads frames however the name is spelled (an attribute or import, a string or a folded
`"_get" + "frame"`, `vars(sys)`, `sys.__dict__`, `exec`/`eval`/`__import__` of anything the scan cannot read, a computed
`getattr` next to an import of `sys`/`inspect`/`traceback`) or imports a module that runs code on another stack
(`threading`, `_thread`, `concurrent`, `multiprocessing`, `asyncio`, `socketserver`) or reads frames for a living (`pdb`,
`bdb`, `trace`, `cProfile`, `profile`, `faulthandler`, `tracemalloc`, `code`). Its run is never a witness
(`engine.py::NEVER_A_WITNESS_FLAGS`, the grader's `replay.NEVER_A_VIOLATION_FLAGS`) and, carrying a flag, certifies nothing.
The scan reads the entry file only and is the cheap EARLY flag; **run-time detection is the authority**
(`_child.py::watch_threads`, installed by `main`, `_probe.py` and `nondet_harness.py::drive`, and by the grader's shim
`_sv_verifier._watch_threads`): `threading.Thread.start` and every function a thread is created through
(`_thread.start_new_thread` / `start_joinable_thread` and `threading`'s own copies of them) set `stack_unmodelled` the
moment ANY thread starts, whoever starts it (a helper module of the task, `exec("import threading ...")`, `eval`, a module
imported under an alias by a computed name, `concurrent.futures`, a `Timer`). `drive` runs in the verifier's own process and
undoes the patch after each run.
The bootstrap import cache also differs from native execution: the model has already imported `json`, `random`,
`string`, `traceback`, `types`, `time`, and `math`. `solve/extern.py::native_import_reason` refuses reads of `sys.modules`,
dynamic imports, and stdlib names shadowed by task files or packages in the script directory or explicit import roots.
It reads local imports recursively, including package initializers; dynamic code and unmodelled path changes also
refuse. Known `_sv_verifier` draw lookups through its cache entry, globals, literal import or getter stay supported.
Proof engines and the scheduler return a named
`UNKNOWN`; child replay, stationary probes and in-process drive set `import_state_unmodelled`, which cannot certify a
TRUE or a FALSE witness. Cache eviction would disturb the model's own imported objects, so these cases stay unsupported.
The file names of frames below the module frame (`warnings.warn(..., stacklevel=n)` for a large `n`) still differ.
**A property class is the BUILTIN** ([internal ref]): the child records the names of the MRO classes that
are the builtin objects (`_child.py::exception_mro`), so a program's own `class IndexError(Exception)` is no `IndexError`
and `class E(IndexError)` is. Whether SV-COMP 2027 reads `uncaught(E)` that way for a class merely *named* `E` is not
settled, so a program that binds a property exception name itself (any scope, import alias, parameter, `except ... as`,
`builtins` change, namespace `update`; a class renamed through `__name__`/`__qualname__` or made by `type(name, ...)`/`new_class`;
in the program, its declared files or the local modules it imports; unparsable source counts as binding every name;
`engine.py::shadowed_property_names`, `local_import_files`) gets no TRUE: `scheduler.run`'s `done` withholds it before any
witness is written, `engine.verify` re-checks, and the answer is `UNKNOWN` (owner 2026-10-18). FALSE is unaffected (a
built-in instance escaped). **Our model never answers for an exception CPython would not raise** ([internal ref], [internal ref]): the
random model counts a range arithmetically (`random.randint(0, 2**100)` works), and a run in which the model itself raised
(`model_error`) or that touched the module-level `random.setstate`, or `getstate` after a random draw
(`random_state_unmodelled`; the draws are not a generator state) is never a witness (`engine.py::NEVER_A_WITNESS_FLAGS`); a `getstate` before any random draw reads the OS-seeded state exactly as natively, so that flagged execution cannot certify TRUE (`random_state_read`; MT's first word `st[1][0] == 2147483648` fails natively every time, a witness replays).
Static analysis may still prove seed-independent `getstate()` shape and index invariants without executing the program;
the execution flags remain unchanged, and a flagged run never certifies determinism or a proof.

- the kinds are `bool`, `int`, `float` and `str`;
- `nondet_float_range(a, b)` is a `float` draw REJECTED unless `random.uniform(a, b)` can return it
  ([internal ref], [internal ref]: see "Draw domains: both readings" below): the call raises `ValueError` and sets
  `Outcome.invalid_draw`, matching the grading harness's own model (`eregion-sv/src/eregion_sv/shim/_sv_verifier.py::nondet_float_range`) exactly,
  instead of the earlier silent clamp -- a clamp let Tharkun's own replay confirm a FALSE
  whose witness vector held a raw draw the harness's `eregion_sv.replay` refuses to confirm
  (`invalid_draw=True` there too), a wrong FALSE;
- `nondet_float()` is a `float` draw REJECTED unless it is `inf`, `-inf`, `nan` or `-0.0` ([internal ref]), and its default
  (a vector that ran out) is `-0.0`;
- `nondet_str()` is a `str` draw whose every character must be in `string.printable`, REFUSED the same
  way ([internal ref], the second half: the child used to accept any string, so an enumeration candidate such
  as `"\u00e9"` or an Atheris unicode proposal could be confirmed here and refused by the harness);
- `nondet_list` and `nondet_dict` keep the reference loop on `nondet_bool`.

**Unseeded** `random.choice`, `randrange`, `randint`, `random` and `uniform` are draws too (`rindex`
of `n` possibilities, `rfloat`). A `randrange(start, stop, step)` draw is `range[index]` = `start + step * index` and its log
entry carries `start` and `step` (`_child.py::_draw`; the witness vector does not: kind, value, `n` only). The enumeration tries
the first 32 indices and `n - 1` (`engine.py::_RINDEX_CAP`) and, past them, the index at which the call returns an int literal
of the program or its neighbour or negation (`engine.py::harvest_constants` -> `rindex_values`, `Candidates.value`,
`rindex_hits`), so `x = random.randint(0, 2**100); assert x != 12345678901234567890` is FALSE with the index
`12345678901234567890` as its witness ([internal ref]; the grader replays it with the same arithmetic count). A hit is only a candidate: a
literal no index returns (outside the range, off the step, before the start) is never proposed, and the verdict is what the child's run does. After `random.seed(x)` the real generator runs. Nondeterminism the model does
not control is **flagged**: `sys.argv` reads, clocks, `input()`, other unseeded `random` calls.

A run that ever sets `invalid_draw` is never a violation (`engine.py::Outcome.violated`), no matter
what exception actually surfaced -- the grading harness's own replay would refuse the out-of-domain
draw before ever reaching it, so such a run can never become a real witness.

### Draw domains: Tharkun's child against the grading harness ([internal ref])

The rule is REFUSE, never clamp: wherever the harness (`eregion-sv/src/eregion_sv/shim/_sv_verifier.py`,
replayed by `eregion_sv.replay`) rejects or alters a draw, `sv/_child.py` does the same, and a run that
made a refused draw is never a violation. Audited kind by kind (the differential in
`tests/test_sv_draw_domains.py` runs the real harness against the child on every row when `eregion_sv` is
importable):

| draw | the harness | Tharkun's child | same? |
|---|---|---|---|
| `bool`, `int` | any value | any value | yes |
| `nondet_float()` | `inf`, `-inf`, `nan` or `-0.0` ([internal ref]; any float before), else the run is invalid; default `-0.0` | the same | yes |
| `nondet_float_range(a, b)` | a value `random.uniform(a, b)` returns ([internal ref]; a float in `[min(a, b), max(a, b)]`, never nan, before), else the run is invalid | the same | yes |
| `nondet_str()` | every character in `string.printable`, else invalid; refused before it is logged | the same (this fix; before it, ANY string) | yes |
| `rindex(n)` (unseeded `random.choice`/`randrange`/`randint`) | clamped into `[0, n - 1]`, the CLAMPED value is logged | the same | yes: a vector holding the raw value no longer equals the log, so `engine._replayed` and the harness's `replayed` both refuse it |
| `rfloat` (unseeded `random.random`/`uniform`) | outside `[0, 1)` becomes `0.0`, logged | the same | yes, same argument |
| a vector entry of the wrong kind | the kind's default is drawn and logged | the same | yes, same argument |
| `argv_full` (whole list, consumed first) | any plain list of strings, including empty/Unicode/NUL; malformed or misplaced boot entries refused before target execution | the same; fresh witnesses capture the original list before target mutation | yes; stored `argv` entries retain legacy tail semantics |
| a malformed vector entry | refused (a crash) | refused (`invalid_draw`) | yes |

The proposal generators are not the judge: every candidate is replayed through `_child.py`, so a generator
may propose anything. They avoid proposing what the child would refuse (it is wasted CPU, and, before the
`str` rule, a wrong FALSE): `engine.py::admissible` filters the enumeration (the seed stages' values, the
harvested program literals; `BODY["str"]` no longer holds `"\x00"` or `"\u00e9"`),
`sv/_hybrid_worker.py::_clamp_float_range` snaps Atheris's float proposals into the range's domain (`_child.snap_float_range`: a huge proposal becomes the largest value `uniform` returns, not the bound) and
`::_printable_str` builds every `nondet_str` proposal from `string.printable` characters directly (one per
input byte, `ConsumeBytes`; a byte that is already a printable character stays itself, every other byte folds
onto `0x20..0x7E`; it replaced folding Atheris's unicode strings, whose encoding byte and million-code-point
alphabet left a mutated input less likely to read back as a target such as `"qqqqq"`: `b"qqqqq"` was the string `"qqqq"`), both BEFORE logging
(so a find replays as written), and `sv/_concolic_worker.py`'s unlifted-call stub
logs an in-domain `nondet_float_range` value (the same snap). A fact producer's draw vectors ("Draw vectors from a producer", below)
are checked at the door (`sv/eregion_bridge.py::publish_draw_vector`) and by `engine.py::vector_admissible` against the
bounds the first run logged (`Outcome.ranges`, `[position, a, b]` in call order); the child refuses what still gets through. The symbolic stage (`sv/_symexec.py`) leaves strings
unconstrained; a model the child refuses simply does not confirm.

### Draw domains: both readings ([internal ref])

The sv-benchmarks helper (`_sv_verifier.py`, MR !1752) is documented one way and written another:

```python
def nondet_float():
    if nondet_bool(): return random.choice([float('inf'), float('-inf'), float('nan'), float('-0.')])
    return random.uniform(-sys.float_info.max, sys.float_info.max)
def nondet_float_range(a, b): return random.uniform(a, b)
def nondet_str(): return ''.join(random.choice(string.printable) for _ in range(abs(nondet_int())))
```

and `random.uniform(a, b)` is `a + (b - a) * random()` with `random()` = `k * 2**-53`, `0 <= k < 2**53`. The documentation reads as any
float, a float in `[a, b]`, any character. Owner ruling (2026-10-06): **a TRUE is proven over the LARGER reading, a FALSE is confirmed only with a
value the CODE can produce on CPython 3.14**, so both hold under both. A violation reachable only through a value the code never produces is
UNKNOWN, never FALSE.

| helper | TRUE is proven over (absint, the proofs) | FALSE is confirmed only for (child, harness shim) |
|---|---|---|
| `nondet_float_range(a, b)` | the documented non-NaN floats in `[min(a, b), max(a, b)]` JOINED with the formula over `random()` in `[0, 1]` (`absint/interp.py::m__sv_verifier_nondet_float_range`, `rand.py::uniform_formula`): `inf` and `nan` too when `b - a` overflows or a bound is infinite or nan | a value `a + (b - a) * (k * 2**-53)` in IEEE doubles, `0 <= k < 2**53` (`_child.float_range_producible`): `1.0` is NOT one for `(0.0, 1.0)` (its largest is `1 - 2**-53`), `5 + 5 * (1 - 2**-53)` rounds to `10.0` so `10.0` IS one for `(5, 10)`, and `uniform(-1e308, 1e308)` is `inf` or `nan` only |
| `nondet_float()` | any float (`FLOAT`) | `inf`, `-inf`, `nan`, `-0.0`: the second branch `uniform(-max, max)` overflows (`b - a` is `inf`) to `inf` (`k > 0`) or `nan` (`k == 0`), never a finite float, on every CPython from 3.8 to 3.14 (MAIN checked 100,000 draws of the real helper on each) |
| `nondet_str()` | any str (`ANYSTR`) | every character in `string.printable` (already the rule since [internal ref]) |

`_child.float_range_producible(a, b, v)` is exact, not sampled: the two roundings of `a + (b - a) * r` are monotone in `k`, so the value
set is searched by bisection (`_monotone_image_contains`, 53 steps); a span that overflows or is nan gives `nan` (`k == 0`) and
`+-inf` or `nan` (every other `k`); an int beyond the float range is an `OverflowError` in the helper (no value); a bound whose type is not exactly `int`, `float` or `bool`
(a subclass, `Fraction`, `Decimal`, a numpy scalar, a user object) has no value in the domain: the draw is refused (`invalid_draw`), never a witness, because
the formula would run that type's own arithmetic and the closed interval it used to fall back to holds values the code never returns. The harness shim holds the same predicates, and
`tests/test_sv_float_domains.py` holds them to `random.Random.uniform` itself (a subclass whose `random()` returns a chosen `k * 2**-53`), to enumeration on small monotone
functions, to the real helper's code run on this interpreter, and to each other. A proposal generator that wants a value near a bound asks
`_child.snap_float_range` (the first producible value at or past it, the largest one for a proposal beyond the domain). MIN's `cmp_g07_float_range_neg`
(`y = nondet_float_range(0.0, 1.0); assert y < 1.0`) is UNKNOWN, not FALSE: the only violation is `y == 1.0`, a value `uniform(0.0, 1.0)` never returns.

Every path that can write a confirmed FALSE ends in a `_child.py` run: `engine.py::run_once`, the fork
server (`_forkserver.py` calls `_child.main`), the stationary probe (`_probe.py` imports `_child`),
`nondet_harness.py::drive`, and the product proof's replay (`engine.py::replay`); the only caller of
`ObligationTable.violate` is the scheduler's `found`, which replays or takes the draws a `_child` run
consumed. `tests/test_sv_draw_domains.py` pins that there is no second draw model.

Each child's address space is capped (`--run-memory-mb`, [internal ref]; `engine.py::_cap_memory`). A draw that sizes a
huge allocation then ends that run with `MemoryError` inside the child, and the run counts as inconclusive,
like a timeout. Without the cap, the machine's (or BenchExec's cgroup) OOM killer would take down the
verifier itself.

`engine.py::verify` applies the competition's rules:

- **FALSE(E)**: a run ended with an uncaught instance of a property class. The run's draw vector is
  the witness (`sv-py-witness/1`), and replaying it reproduces the run exactly.
- **TRUE**: only when the explored runs cover every execution. That means either a deterministic
  program (no draw, no flag, and two runs under different hash seeds agree), or a draw tree whose
  kinds are all finite (`bool`, small `rindex`) enumerated completely with no timeout.
  **One run per draw vector stands for that vector only if runs are deterministic given the draws**,
  so the deterministic and finite-tree TRUE also need the static certificate
  `solve/determinism.py::determinism_census` ([internal ref]; the allowlist of the stationary proof below), which
  the verdict's `detail["certificate"]` cites. Without it the answer is UNKNOWN: two hash seeds are a
  sample of the executions, not all of them.
- **UNKNOWN**: everything else. A bounded search that found nothing is not a proof.

**The order of the stages ([internal ref] slice 3, owner ruling 2026-09-24).** The cheapest stage that can
DECIDE runs first; execution comes late and enumeration last (`engine.py::verify`):

1. **Static proofs** (`engine.py::_static_proof`; nothing runs), described under "Static proofs
   before the search" below.
2. **Symbolic**: ONE symbolic run of the whole harness toward every violation site
   (`concolic.py::solve` with `drive=False`, `_symexec.py::solve_targets`). Each solved vector is
   replayed on the unmodified program.
3. **Execution**: the default run and, for a program without draws, a second run under another hash
   seed (`deterministic`). Then `tharkun.sv.prove.prove_dynamic` ([internal ref]).
4. **The solving portfolio** (below): the seed stages, and the concolic stage's per-site drive, which
   now runs only on the sites that stage 2 left open.
5. **Enumeration** (the small-first search, below).
6. **The remainder** ([internal ref], `scheduler.py::run`'s `spend_remainder`): when enumeration ran dry (or
   stopped short) with budget left, the stages that can keep searching an UNBOUNDED draw domain
   (`stages.py::Stage.deepen`: today the fuzzer, `hybrid.py::deepen`) get the rest, in rounds of half
   the remaining budget, until the goal is met or the ledger says stop. A stage running dry never ends
   the run by itself; see "Order and the CPU-second ledger" below.

Enumeration is CAPPED when it can only confirm what is already known:

- after a proof, it is a **canary** of `min(canary_seconds, budget/10)` CPU seconds (60 s by
  default, `--canary-seconds`; see "Static proofs before the search" for 0). A violation the canary finds wins, since its witness is real, and
  `Verdict.detail["unsound_proof"]` names the proof that was wrong. Every run's timeout is held to what
  the cap leaves, so a program whose one execution takes minutes is decided by its proof;
- when stage 2 solved EVERY site to "no modelled run fires it" (`concolic.py::Solved.closed`), it runs
  `min(early_stop_seconds, budget/10)` CPU seconds and the answer is UNKNOWN early ("search stopped
  early"). That answer is never TRUE: the symbolic model is bounded, and the sites are only the ones
  the harness knows.

**The solving portfolio ([internal ref], `src/tharkun/sv/stages.py`).** Before any enumeration the
engine runs the stages of `stages.py::REGISTRY`, cheapest first, each with its own attribution name.
A **seed stage** returns candidate values per draw kind, which `engine.py::Candidates` ranks right after
each kind's head values. A **vector stage** proposes whole draw vectors, which the engine replays within
the stage's share of the budget. `--stages a,b` and `--no-stage NAME` (or `verify(stages=...)`) select
stages. A stage whose numbers show it adds nothing stays out of `stages.py::DEFAULT`, with the numbers
recorded next to it.

- `guard_values` (seed stage, component `static:guard_values`, `src/tharkun/sv/guard_values.py`) executes
  nothing. It finds the names bound directly to a draw, inlines straight-line definitions (a name
  assigned once in its scope, outside loops) into every `if`/`while` test and negated `assert`, conjoins
  the enclosing `if` tests, and solves the result. Integer draws go to `solve/arith.py::solve_int_guard`
  (linear elimination, `%`, `//`, `abs`, polynomial roots), in both polarities. Other kinds go to
  `solve_condition` with the draw kinds as `var_types`. It costs a parse plus milliseconds per guard,
  and it reaches draws in loops and helper functions that the harness cannot lift, because a seed
  applies at every draw position of its kind.
- `concolic` (vector stage) is described next.

**The concolic stages ([internal ref] slices 2-3, `src/tharkun/sv/concolic.py`).** Stages 2 and 4 SOLVE
for draws that reach a violation site:

1. `sites.py::build_harness` lifts the task's entry code into `__sv_harness(<draws>)`, each parameter
   annotated with its draw kind. Which code is the entry code is `sites.py::_plan`'s decision, read as the
   child runs the program (`__name__` is `"__main__"`, so the guard block is code that runs where it stands),
   in this order ([internal ref]; the real tasks of sv-benchmarks MR !1787 all end in a module-level `main()`):
   - **an entry function** (`def main(): x = nondet_int(); ...` then `main()`, at module level or under the
     guard, and no module-level statement draws): the harness body IS `main`'s body, so its early `return`
     (the precondition guard `if not (lo <= x <= hi): return`) stays a `return`, and the def, its call and
     nothing else are gone. `sites.py::_entry` decides this is exact and refuses by name, with the reason
     `concolic.py` logs (`concolic: no harness or no site (<reason>)`, from `sites.py::harness_refusal`), when:
     the call has arguments, the program does not END with the call (statements follow it, or its value is
     used: `r = main()`, `sys.exit(main())`), the function takes parameters (defaults included), is decorated,
     is a generator or declares `global`, is defined twice, is called more than once or referenced besides its
     one call (recursion included), or the `call_target` fact does not say the name is a PROVEN
     `user_function` (nothing rebinds it; no fact at all is a refusal too, since nothing says what `main`
     is). The draws in a function `main` calls are not lifted (they stay with a stub, below); when those are
     the only draws the reason says so;
   - **module-level draws** (`SIZE = 1 + abs(nondet_int())`, bm_go_nondet): a parameter is only read when the
     harness runs, so every module-level statement that is not a definition (`import`, `def`, `class` stay at
     module level, in order) moves into the harness, in order, and the names it assigns are declared `global`
     (they are the program's module globals; a draw assigned to a name a definition reads stays a global too,
     it is not made a parameter). The module-level draws are therefore part of the vector, FIRST, which they
     were not before (the worker clears the log after importing the harness). Refused by name where the move
     would change the program: a class body, default or decorator that reads a moved name while it is being
     defined, a read of the module namespace (`globals()`, `__name__`, ...), a star import under a compound
     statement, an entry function that assigns a name the harness also assigns as a global;
   - otherwise the `if __name__ == "__main__":` block, as before.
   The text is compiled once (`sites.py::_build`), so a harness the interpreter would reject is a named
   refusal, not a crashed worker. Parameters:
   - a draw `x = nondet_int()` becomes the parameter `x`, and any other draw a fresh `dK`;
   - a draw inside a loop or comprehension becomes a STREAM `sK: list[kind]`, one value per execution
     (the child's default past its end); this holds inside an entry function too (the real tasks' `while i < N:
     v = nondet_int()`);
   - draws inside lambdas and nested defs stay with a stub.

   The sites:
   - `sites.py::violation_sites` lists the explicit ones: `raise X` with `X` a property class or a
     builtin subclass of one, and `assert` for `AssertionError`, each with its enclosing guards;
   - `sites.py::implicit_checks` finds the implicit ones: `c[k]` that can miss (`IndexError`,
     `KeyError`) and `a / b`, `a // b`, `a % b` with `b == 0`. Only side-effect-free operands count,
     and the arms of a conditional expression carry its test. The harness gets
     `if <condition>: __sv_site(n)` before each such statement, unless an enclosing `try` catches the
     exception, so an implicit site is a line a solver can target. An operand may call `len`, `abs`,
     `min` or `max` (`sites.py::_PURE_CALLS`).
   - typing sites (slice 3), only on names `sites.py::typing_scope` finds assigned values of two or
     more types (`v = 0` ... `v = "tag"`) or `None`: `a <op> b` that raises `TypeError`
     (`__sv_ok_binop`, which tries the operator on one representative value per operand type) and
     `x.attr` with `x is None` (`AttributeError`). An operation over names that only ever hold
     numbers is no site.
   - `extra_sites`: the abstract interpreter's site records, on original lines. A may-raise record
     with a `residual` is one more marker, with the residual (and the record's `scope`) as its
     condition; an implicit site whose records all say safe gets no marker.

   `Harness.feeds` records which draws a main-block call passes to which function (the def-use link).
2. A helper process (`_concolic_worker.py`) runs under one child's share of the memory cap, so the
   verifier and everything it starts stay inside the task's limit together.
   - **One symbolic run toward every site** (stage 2, `_symexec.py::solve_targets`, class
     `_symexec.py::_Runner`). The harness body runs as a function body (`_Runner.run_body`): a `return` ends
     the path and what follows runs under the guard (the merge of `_exec_fn`), every other statement through
     the per-statement skip-and-poison `run`. The whole harness runs once, with the draws as z3 terms and each stream
     as a z3 array read at a position that is itself a term of the path. Calls to the program's
     functions are inlined (depth 12), and, given the `call_target` records, so are the program's own classes: construction, attributes,
     methods, `super()` and `isinstance` through a per-path heap ("The `call_target` fact", below). A loop whose trip count is a draw is unrolled `LOOP_BOUND` (64)
     times, each iteration under its guard, and the model keeps to runs that leave it within the bound.
     That covers `range(n)`, `range(a, b)` and `range(a, b, step)` ([internal ref] S1), inline or held in a
     name and iterated later (`r = range(n)` ... `for i in r`), **for int arguments only**: a Python
     int or bool, or a z3 `Int` or `Bool`. Iteration `k` runs exactly when `a + k*step` is still short
     of `b` in the direction of `step`, with the target equal to `a + k*step`, assigned inside the
     iteration (after the loop the target holds the last value it was given, or what it held before
     when the range was empty; a target that did not exist is poisoned). A zero `step` is `range()`'s
     `ValueError` at the call: the path ends before the loop, as a zero divisor does. Anything that is
     not provably an int (a float or string draw, a symbolic float, a typed union, an arity other than
     1 to 3) stays `unsupported`; a range used as anything but a loop iterable (truth, `len`, `==`,
     `r[0]`, `in`, `list`, a comprehension) is `unsupported` too, since the Python object would answer
     differently from the program. Nothing is approximated.
     **Output files ([internal ref]).** `with open(path, <constant write-only mode>) as fp:` and the write side of
     the handle (`write`, `writelines`, `flush`, `close`) run as ordinary statements: the executor reads the
     same contracts as absint (`src/tharkun/solve/contracts/rules_fileio.py`, through `src/tharkun/solve/contracts/adapters/concolic.py`), ending a path where the
     contract says the call raises (a wrong-typed argument: a `TypeError`, even on a closed file) and
     returning the length written. A str or z3-string path and text are accepted for a text mode, bytes for a
     binary one; an int path (a file descriptor), a read, an update mode, `buffering`, a handle used as a value,
     `tell`/`seek`, and a `with` body that leaves by `return`/`break`/`continue` stay `unsupported`. What the
     environment does (`OSError`) is not a property class and is assumed away: the replay of any model decides. The loop marks the run BOUNDED, so
     `run_exact` refuses it and a proof never rests on it; its `unsat` is a search hint that stops the
     search early with an UNKNOWN verdict (`scheduler.py`: `symbolic.closed`), never a TRUE. Only
     `obligation_table.py::discharge` writes a certified row.
     A value whose TYPE depends on the path is a `Union` of typed members, which is what the typing
     sites ask about. Dicts with constant keys, `in`, `.get`, `.append` and Python's floor `%` / `//`
     by a draw are modelled. The path condition carries everything a later site depends on: branch
     guards, the arms that did not die (raise or return), asserts that passed, operand types that did
     not raise. A pure builtin called on exact builtin values (`ord`, `chr`, `tuple`, `list`, `zip`,
     `enumerate`, `reversed`; `solve/contracts/adapters/concolic.py::concrete_fold`, [internal ref] batch 4) is
     evaluated natively, gated on its tested contract; `zip`/`enumerate`/`reversed` come back as a one-shot
     iterator (`_symexec.py::_LazyIter`) that only an immediate consumer (a `for`, a comprehension, `list`,
     `tuple`, `sorted`, `sum`) unwraps, and `type(x)` maps the value to its class
     (`_symexec.py::_Exec._kind_of`). `set` is not folded: its iteration order depends on the hash seed. The builtins
     the executor evaluates itself (`len`, `abs`, `min`, `max`, `any`, `all`, `isinstance`, `sum`, `range`, `sorted`,
     `int`, `float`, `divmod`, `pow`, `print`) are read through their rules ([internal ref] step 2): `_symexec.py::_Exec._through_rule`
     asks the rule to accept `sv/symexec_visitor.py::SymexecVisitor`, whose visit refuses unless the contract is TESTED
     and otherwise runs the model `_Exec._model_<name>`; an untested or withdrawn contract is a refusal, not the model of
     old. `int` and `float` of a concrete scalar, and `sum` of concrete numbers, are CPython's own (`Rule.native`:
     since 3.12 `sum` adds floats with Neumaier compensation, which the left fold the executor used did not),
     `divmod` and `pow` are the executor's `//`, `%` and `**`, and `print` evaluates its arguments and writes nothing
     the executor tracks (`file=` and `flush=` are refused). The container operations `_Runner` evaluates itself go the same way, on
     the operands already evaluated: `list.append`, `list.__getitem__` / `__setitem__`, `tuple.__getitem__`, `dict.get`, `dict.keys`,
     `dict.__getitem__` / `__setitem__` (`_Exec._model_<qualname>`, `.` read as `_`); an index or key CPython rejects ends the path,
     as before, and a concrete list store out of range now does too (it raised `IndexError` out of the executor). A list or dict is bound
     BY IDENTITY (`assign` binds the object, as CPython does) and the executor's mutations are functional, so a mutation
     (`append`, a store, `pop`, ...) is modelled only where `_Exec._mutable` finds the name this frame's own and the object held by no
     other name, caller frame or container; an aliased list is a named refusal, never a change to one of two names. The methods of
     `list`, `dict` and `tuple` (`count`, `index`, `copy`, `clear`, `extend`, `insert`, `pop`, `remove`, `reverse`, `sort`; `items`,
     `values`, `pop`, `popitem`, `setdefault`, `update`) run by CPython itself on a copy of the receiver, through their rules
     (`_Runner._native_method`, `symexec_visitor.py::NATIVE_METHODS`): concrete operands only where the method compares, a symbolic
     index or key refused, `IndexError` / `KeyError` / `ValueError` end the path as an uncaught raise does. The same for functions of
     `bisect` (`bisect_left`, `insort`), `heapq` (`heappush`, `heapify`, `nlargest`, `nsmallest`), `copy` (`copy`, `deepcopy`) and
     `itertools` (`chain`, `product`, `combinations`, `permutations`, `islice`): `_Runner._contract_call` resolves the module by
     IDENTITY (the object the name is bound to in the harness, so `import heapq as hq` works and a local called `heapq` is refused),
     `_Runner._native_module_function` runs CPython's function on concrete operands (`symexec_visitor.py::NATIVE_MODULE_FUNCTIONS`),
     a mutated first argument on a copy with the name rebound, an iterator back as a one-shot `_LazyIter` of at most 4096 items. The lifted
     `random.choice(seq)` (`sv/sites.py`: `__sv_index(kind, idx, seq)`, a draw as the index) is read through its rule too
     (`_Exec._model_random_choice`: a symbolic index is an ite chain; `_Runner._model_random_choice` bounds it `0 <= idx < len` on the path
     and ends the path on an empty sequence, as `IndexError` does; `tests/contracts/test_symexec_random_choice.py`). EVERY
     function of `math` is handled by `sv/_symexec_math.py` ([internal ref] batch 5): folded natively on concrete
     numbers, encoded in z3 where exact (`floor`, `ceil`, `trunc`, `fabs`, `copysign` on ints, `fmod`,
     `remainder`, `fma`, `isclose`, `modf`, `isqrt`, `fsum`/`prod`/`sumprod` over a list, `pow` with a small
     integer exponent, `ldexp` with a concrete shift, `gcd` with at most one unknown), and refused with a
     reason otherwise (`_symexec_math.py::REFUSED`: a transcendental or irrational function of a
     symbolic value has no exact encoding in the rational model); the callee is resolved by identity in the
     harness module's globals (`_symexec.py::_Exec._math_function`) and the constants by `ev_Attribute`. A
     statement outside the modelled subset is skipped and the names it writes are
     poisoned, so reading one later is unsupported. Each site comes back `sat` (the draws), `unsat` /
     `unreached` (no modelled run fires it), `unsupported` (a statement on its way was not modelled)
     or `unknown` (z3 gave up).
   Then, per site still open (stage 4), in frontier order (sites in the main block before sites in
   callees, fewer enclosing guards first), each with its share of the remaining budget:
   - **The site's slice** (`_symexec.py::solve_site`, sites in the main block). The site's backward
     slice keeps only the statements that define what its condition needs. The slice runs with the
     draws as z3 terms: loops over concrete ranges are unrolled (streams at the length the default run
     consumed), and an `if` on a symbolic test runs both arms and merges them with `If`. So
     `pick`'s key, built from sixteen loop draws, is one formula rather than 2**16 paths, and z3
     solves `path condition AND site condition`. Anything outside the modelled subset declines.
   - **Then Tharkun's own** `driver.drive_to_line`, first from the harness and, when the site sits in
     a function the main block calls with draws, from that function. It then maps the solved
     parameters back to the draws.
   The driver is Tharkun's portfolio: the static guard solver (`src/tharkun/solve/condition.py`),
   snapshot mutation, minleaf and a reach search, each proposal verified by traced execution.
   In this competition build the static flipper also passes the guard and path constraints to
   the native integer solver, using observed integer types to generate candidates only.
   Integer UNSAT under those observed types never certifies a Python path as dead.
   Reach-search harvests live comparison operands; there is no separate RedQueen tier.
3. For every site reached, the worker re-runs the harness and records the draws in execution order
   (the harness's `__sv_draw` and the stub `_sv_verifier` share one log), which is exactly the child's
   draw vector. The engine replays it on the unmodified program. Only a run that really raises a property
   class is a FALSE, so a wrong solve costs one run, never a wrong verdict.

Every verdict names the part of the solver that produced it, printed as `tharkun-sv: component=<c>` just
before the verdict line (`Verdict.component`, with `Verdict.detail`):

- `deterministic`: no draws; one execution decided it.
- `finite_tree`: every draw finite, and the tree was enumerated.
- `proof:<domain>`: a proof that survived its canary (`proof:no-site`, `proof:interval`,
  `proof:shape`, `proof:stationary`, ...).
- `concolic:symexec` (the symbolic run or the site's slice, solved by z3), `concolic:guard_solve`,
  `concolic:drive`: the concolic stages, named after the technique that produced the
  draws (the symbolic executor, or the strongest flipper `drive_to_line` used: static solver or
  snapshot/minleaf/reach search). The line also carries ` solver_calls=<n>` when
  they ran.
- `enum:small_first`, `enum:constants`, `enum:tail`: the enumeration, named after the most expensive
  source among the witness's values (`Candidates.source`).
- `static:guard_values`: the enumeration found it, but through a value the guard_values seed stage
  proposed.
- `none`: UNKNOWN.

`--no-concolic` switches the concolic stages off for ablations.

**Enumeration (the fallback).** The search is **generational and small-first**. A run whose forced prefix has length `L` opens, for
every later draw position, a chain of alternative values for that draw. The chain is walked **lazily**:
running one value pushes the next. Vectors run cheapest first, where a value's cost is its rank in the
kind's candidate sequence (`engine.py::Candidates`):

1. the small values (`engine.py::HEAD`: `0, 1, -1, 2, -2` for ints);
2. the **program's own constants**: every literal, with an int's neighbours and negation
   (`engine.py::harvest_constants`, [internal ref]), which are the values its guards compare with;
3. fixed interesting values (`engine.py::BODY`: powers of two, `inf`/`nan`/`-0.0`, …);
4. an **unbounded tail** of growing magnitudes (`engine.py::_tail`, [internal ref]), so an int, float or str
   draw is never exhausted while budget remains.

Up to `--jobs` vectors run **side by side** ([internal ref]). The default is the CPUs this process may use,
at most 4, which is SV-COMP's core count. A batch's outcomes are read in cost order, so for a given
`--jobs` the verdict and witness do not depend on which child finishes first. Each popped vector queues
its chain's next value before the next pop, so a batch fills even when the program has a single
unbounded draw (one chain). `--budget-seconds` counts **CPU seconds** of the verifier and its children,
which is what BenchExec limits; it also caps the wall clock. Each of a batch's children may run for its
share of the remaining budget, so a batch of long runs cannot overshoot it. `--run-memory-mb` is shared
evenly by the concurrent children.
Concurrent runs reserve that share until completion; ordinary runs have no one-second minimum.
Fork-server startup consumes the same share before the target's deadline is assigned.

## Static proofs before the search ([internal ref] group C)

FIRST, before anything runs, `engine.py::verify` calls `engine.py::_static_proof`. It imports the
provers lazily, so each is used as soon as it is merged, and a missing or crashing prover is skipped:
`tharkun.sv.prove.prove_static` (the census proofs, [internal ref]), then `tharkun.solve.absint.prove`
([ABSINT.md](ABSINT.md)), whose components are `proof:interval` and `proof:shape`. The engine calls
the abstract interpreter itself (not the `prove.py::prove` wrapper) so that it runs once and its
per-site records also reach the concolic stages (as `site_record` Evidence in the store).

A proof does not end the run: the remaining stages run as a **canary** capped at
`min(canary_seconds, budget/10)` CPU seconds. A FALSE the canary finds wins over the proof. Only when
the canary finds nothing does `verify` return the proof's TRUE, with the proof's component and its
checked sites in `Verdict.detail`. `--no-prove` switches the proofs off, for ablations.

A native proof-canary sample gets at least one second, limited by `--run-timeout`
and the whole task's unreserved CPU/wall budget; startup uses that same reservation.
The sampling cap stops new portfolio samples and enumeration batches. An admitted
batch (up to `--jobs` children), the initial run's hash/differencing checks, and
violation confirmations may finish across it, with at most one second of extra
allowance per such run. Ordinary search remains cap-clamped.

**The canary is a budget, `--canary-seconds` (owner ruling 2026-10-06, policy A).** It is the CPU time the search
may spend trying to refute a proof's TRUE, and the engine's `canary_seconds` (`sv/engine.py::verify`,
`sv/scheduler.py::run`) was always a parameter; the CLI now exposes it as `sv/__main__.py::main`'s
`--canary-seconds`, default `60.0`, which is the shipped behaviour. `0` is NO canary: `sv/scheduler.py::run`'s
`start_canary` returns the proof's verdict the moment any proof (static, product, facts, dynamic) decides TRUE, with
no further execution, and the ledger records no canary cap (`ledger.py::Ledger.capped` is never called for it).
It is a knob like `--budget-seconds`, not a feature flag: **development campaigns pass `0`** (a campaign that measures
whether a change decides pairs does not need the cost of re-checking every TRUE), **the competition and the
regression runs use the default**, since the canary is what turns an unsound proof into a FALSE instead of a wrong
TRUE (`Verdict.detail["unsound_proof"]`). `tests/test_sv_goal.py::test_canary_seconds_zero_skips_the_canary` pins
both: the default caps the search after a proof, `0` does not run it.

Every proof return records `Verdict.detail["canary"]` with `attempted`, `completed`,
`cut` and `reason`. Completed means a native run started and no run was cut by timeout,
crash or memory exhaustion; it does not certify a proof. Disabled canaries record no attempt.
The reason also distinguishes early cuts from completed runs on the proof's return path.

## `proof:no-site` ([internal ref])

The first static stage (`prove.py::prove_static`, called by `engine.py::_static_proof` before absint): `solve/census.py::site_census` finds no
construct the program can run that can raise a property class with any values. It counts every operator
that can (`/ // % **` unless the right operand is a safe literal; for data-lookup every subscript and
every `%` that may be formatting with a mapping), every library call in its tables (`divmod`, `pow`,
`math.log` with a base, the dividing `random` functions; `pop`, `popitem`, `remove`, `format`,
`re.sub`, `random.choice`, ...), every `assert`, every `raise` it cannot show is of another class, and
generator `throw`; such a callable used as a value counts too. It is sound only as an ALLOWLIST: the
program must resolve (`solve/pyresolve.py::Resolver`), so anything reflective -- a dunder attribute such
as `__getattribute__` or `__dict__`, `getattr` with a computed name, `vars`, `eval` -- refuses, as do
imports outside its tables and exception classes it has no table for (`TypeError`, `AttributeError`).
Defs never referenced are dead and skipped. It proves `bm_fannkuch_nondet`, `bm_richards_nondet`
arithmetic and `bm_go_nondet` assertion-safety (its asserts sit in `check()`, which nothing calls).
Lane F's reflection probes are regression fixtures (`tests/fixtures/census/never_no_site/`).

**Contract-derived site widening was tried and removed** ([internal ref] added it, [internal ref] proved it unnecessary and
removed it, 2026-09-28 owner review): `solve/contracts/adapters/obligations.py` briefly carried a
`contract_call_sites()` that EXTENDED `solve/census.py::_Scan.calls` from the contracts registry -- any
TESTED, non-`"never"` contract claim for a census-known class added that qualname to the site table,
call-NAME-level. This was sound by construction (it can only make more constructs count as sites) but
turned `range`/`abs`/`divmod`/`str`/`repr`/`ascii`/the builtin `format()`/`print`/`set.update`/`set.add`/
`len`/`isinstance`/`random.shuffle` (and others) into sites almost everywhere they're called at all,
regardless of the actual arguments, since each carries a TESTED claim that a hostile argument's own
dunder (or, for `print`, a hostile `file=`'s `.write()`) could raise. Measured cost on the Real-world
suite: `proof:no-site` 15 correct -> 0, only partly recovered by a same-pattern reaching-values argument
gate ([internal ref] first pass), for a net RW loss even after that recovery.

**The widening added no soundness.** Every hostile construction it could ever have caught is ALREADY
caught by census's existing, UNCONDITIONAL scan of a user override's own body, by construction: reflection
is refused (`solve/pyresolve.py::Resolver`), so any override of a dunder or of a protocol method (a hostile
`__index__`, `__abs__`, `__divmod__`, `__str__`, `__hash__`, `write`, ...) MUST be real, visible Python
source somewhere in the program -- and that def's own body is unconditionally scanned regardless of
whether the site table ever names the call that reaches it: `_dead_defs()` (`solve/pyresolve.py`) never
marks a dunder def dead, and "unknown external calls keep every class method live" (the same function,
for an ordinary method like `write` that a builtin's C implementation can invoke without any visible AST
call) means neither is ever pruned as dead code either. Confirmed empirically (not just by this argument)
against every widened builtin the Real-world suite exercised, including one level of indirection through
a helper function and a subclass override reached only via a builtin protocol (`random.shuffle` on a
`list` subclass's `__setitem__`): the OLD, unwidened census already found a real site for every one, via
the override's own body, in every case. The widening therefore only ever added a SECOND, redundant path to
the same already-caught sites, at the cost of flagging every BENIGN call that never triggers any override
at all -- pure imprecision, never soundness. It is gone; `_Scan.calls` is native-only again
(`_CALL_SITES`/`_THROW_CALLS`), matching this section's own description above. The reaching-values gate
machinery this pass motivated (`_provably_int_or_bool` and its `_numeric_gate_exempts`/`_scalar_gate_exempts`/
`_print_gate_exempts` siblings) went with it, since there is nothing left for it to gate.
`solve/contracts/adapters/obligations.py::obligation_for_call`/`_record_obligation`/`Census.obligations` --
the SEPARATE per-site witness-format safety-PREDICATE mechanism this same PR added, keyed off whatever the
(now native-only) site table finds, not off the widening -- is unaffected and stays.

## Run-based proofs: `proof:stationary` ([internal ref])

When no static stage proves the property and the first run raised no flag, `engine.py::verify` calls
`prove.py::prove_dynamic`. Its one proof, `proof:stationary`, proves programs whose unbounded draws only
say how often a loop repeats. It rests on three claims, two static and one observed:

1. `solve/drawflow.py::plan_draws` (static) classifies every draw. A *trip* draw reaches the program only
   as the trip count `c + abs(d)` of `for _ in range(...)` loops whose target is never read: through plain
   variables, `range` objects used only as loop iterables, and positional/keyword arguments of module-level
   functions called by name. It may also be the first argument of `type(x)` or `isinstance(x, <builtin
   classes>)` (`drawflow.py::_Planner._class_only_call`, [internal ref]): a trip value is always an exact int, so the
   result is a function of its CLASS only, the same for every trip count, and neither call can raise or run
   user code. The shapes come from the contracts (`solve/contracts/adapters/common.py::CLASS_ONLY_SHAPES`,
   gated on a tested contract) and a builtin name bound anywhere in the program is not trusted. No other
   builtin qualifies -- `divmod(n, K)`, `len`, `ord`, `list(range(n))` return something that varies with the
   value, and the induction below stops tracking a value the moment the planner does (the never-TRUE probe
   `divmod_gate` in `tests/test_sv_prove.py::MUTANTS` is the proof: an IndexError reached only at n == 13).
   A *quotient* draw reaches it only as `abs(d) % M`. A `nondet_bool` is finite.
   Anything else refuses, and so do introspection, reads through `open`, `id`, `hash`, `__del__`,
   imports outside `MODULE_WHITELIST`, and any use of a library module other than reading a public
   attribute. A write-only `open` (`solve/output_effects.py::MODES`: `w`/`wt`/`wb`/`a`/`at`/`ab`, the
   subset of the shared `solve/environment.py::OUTPUT_MODES` the probe replays)
   is allowed instead of refused when `solve/output_effects.py::output_only` proves the handle
   cannot be read back (below).
2. `solve/determinism.py::determinism_census` (static) certifies that a run is a function of the program
   and its draws. It is an ALLOWLIST built on `solve/pyresolve.py::Resolver`: every import is in a
   deterministic set, every attribute is one the program binds or an allowlisted type or module has,
   nothing is reflective (computed `getattr`, `vars` / `globals` / `__builtins__`, dunder attributes,
   `eval`); no `id` / `hash` / `input` / reading `open` / `__del__`; `random.Random` only with a literal
   seed and never `SystemRandom`; no set order reaches the program (a set stays in names used only
   order-blind; `pop` requires a proved singleton receiver); no address reaches it through text (every `str` / `repr` / `format` /
   `%s` / f-string of a value that might have a default `repr` must go only to `print`, a raised builtin
   exception, or a `__repr__`-like return).
3. `_probe.py` (observed) runs the program with each trip loop's iterable wrapped in a generator that
   yields the same items. With `m = max(c, 1)`, `prove.py` enumerates the draw tree over trip counts
   `c .. m + 1`, every residue and every finite value, under two hash seeds. In every run, a trip-loop
   instance with trip count `m + 1` must reach heads `m + 1` and `m + 2` with **equal canonical
   snapshots**, or leave the loop early. A snapshot (`_probe.py::snapshot`) is an address-free rendering of
   the locals of every frame out to the module frame (the loop's dead target excluded), everything they
   reach (instances, classes, functions with their defaults and closure cells, containers, `array`,
   `StringIO`), the real `random` state, and the model's draw count and flags. An object it cannot render
   (an iterator, a generator, a foreign class, a set or dict keyed by identity-hashed objects) refuses.

Why that covers every execution: between draws a run is deterministic (claim 2), and a loop body cannot see
its trip count (claim 1). So from `S(m) == S(m + 1)`, every later iteration starts from the same state,
does what iteration `m + 1` did (no violation, no draw: the draw count is part of the state), and leaves
the loop in the state trip count `m` leaves it in. Trip counts `c .. m`, trip 0 included, ran directly. A
probe run that raises a property exception is replayed on the plain child and, if confirmed, returned as
FALSE with its witness. A stationary TRUE faces the same canary as a static one.

Measured ([host], 15 min / 15 GB, 4 jobs): `bm_spectral_norm_nondet`, `bm_unpack_sequence_nondet` and
`bm_regex_v8_nondet` are TRUE (3 properties each, 6 probe runs). `bm_hexiom_nondet` is refused: it turns
`done[id][0]`, whose type the census cannot see, into text that its output comparison reads. `bm_nbody_nondet`
is refused because its `advance` loop moves the bodies, so the state does not repeat. `tests/test_sv_prove.py`
holds the draw-flow refusals, 17 stationarity mutants, the determinism allowlist cases and lane F's 24
nondeterminism probes (`tests/fixtures/determinism/never_certified/`), none of which may pass.

## Output-file effects in `proof:stationary`

`drawflow.py::screen` calls `solve/output_effects.py::output_only` before the `open` refusal above.
It admits a program whose only use of `open` is a write/append literal mode where every handle binds
exactly one local name, never escapes (is not returned, stored, or passed anywhere but
`print(file=...)` or its own discarded `write`/`close`/`flush` call), and is never read. Every import
must sit in the module's leading preamble (docstring, then `import`s only): an import after output
could read a module whose contents were just discarded. Any `try`/`try*`/`async with`, or a `with`
naming anything but a tracked handle, refuses, so no exception handler and no other context manager
can observe an I/O failure. `output_only` raising `ValueError` falls back to the ordinary `open`
refusal -- it never turns into a silent UNKNOWN.

`_probe.py::_output_open` then stands in for the real `open`, bound only into the probed run's own
module `__builtins__` (so `dir()` and friends never see it), and returns `_probe.py::_OutputFile`, a
successful-write model that stores no file contents: `write` checks the exact text/bytes type and
encodes the value (so a real `UnicodeEncodeError` still surfaces), `close`/`flush` track closed state,
and an unmodelled path or payload type raises `Refused`. `prove.py`'s `check()` treats any probe
status other than `clean`/`exit`/`raised` as "does not support the proof", so an unmodelled write
aborts the stationarity attempt rather than assuming success. This is an overapproximation in one
direction only: a real environmental failure (disk full, permission denied) truncates the execution
before the modelled continuation runs, so the proof still only claims what the successful
continuation shows. `prove.py::prove_dynamic` refuses the whole attempt when the property's classes
are not a subset of the tracked `solve/absint/interp.py::PROPERTY_EXCS`, so this licensing is never
used to discharge a property over an exception class (`OSError`, say) that a real output failure
could actually raise. A confirmed violation still replays on the plain, unmodelled child
(`engine.py::run_once`) before it is returned as FALSE.

## Output-file follow-ups ([internal ref]): preamble ordering, finally-only Try, a `nondet_str` path

Three relaxations, all additive (no existing verdict changes byte-for-byte unless the
program hits the new, wider case), targeting `bm_raytrace_nondet.py`'s own shape (MG's
soundness review of the first `output_only` cut, [internal ref]):

* `output_effects.py::output_only`'s preamble used to be "imports, then a leading
  docstring, then stop" -- `bm_raytrace_nondet.py` interleaves constant assignments
  (`EPSILON = 0.00001`, ...) between its imports, with `import _sv_verifier` LAST, so the
  first constant assign ended the preamble and every import after it (`_sv_verifier`
  itself) refused. The preamble is now every LEADING top-level statement that cannot
  itself call anything -- imports, a docstring, an undecorated `def`/`class` (defining one
  runs no code inside it), a single-target literal-constant assignment (`_cannot_call`) --
  the first statement that COULD call something ends it, unchanged.
* A `Try`/`TryStar` with `handlers == []` (only a `finally`, no `except`) cannot itself
  OBSERVE an I/O failure -- nothing catches it, so it still propagates out exactly as if
  the `try`/`finally` were not there. Only a real `except`/`except*` clause still refuses
  the whole thing, unchanged.
* A `nondet_str` draw used ONLY for truthiness and as `open()`'s own path argument
  observes nothing about the string beyond "is it empty or not" -- once `output_only` has
  already certified every `open()` in the program as write-mode-only, single-binding,
  never read back, the PATH argument's own exact bytes can never return to the program
  either. `drawflow.py::_Planner.str_restricted` (called from `plan_draws`'s `nondet_str`
  handling) checks EVERY read of the draw's own variable is one of: a truthiness position
  (`_str_truthy_or_open_path`: an `If`/`While` test, `not x`, `bool(x)`), `open()`'s own
  first argument, or a POSITIONAL argument to a call this pass can resolve -- recursively,
  INTERPROCEDURALLY, into that callee's own matching parameter (`_resolve_str_callee`):
  a plain module-level function (`prog.funcs`, mirrors `_Planner.trip_argument`'s own
  int-draw mechanism exactly), or a method call whose receiver's single, unambiguous class
  this program defines plainly (`_recv_class`: bound exactly once, directly from
  `ClassName(...)`, no inheritance beyond bare `object`, no metaclass/decorator, no
  `__getattr__`/`__getattribute__`/`__setattr__`, AND `f.attr` never the target of a plain
  `x.NAME = ...`/`del x.NAME` anywhere in the whole program (`prog.attr_stores`,
  `_resolve_str_callee`) -- MG's PR 108 review: ordinary Python (`c.run = bad`,
  `C.run = bad2`, `self.run = bad` in `__init__` or anywhere else) can replace what a
  later `c.run(...)` actually invokes with a callable this pass never looked at, and none
  of that is reflection or a receiver rebind, so `_recv_class`'s OTHER checks did not catch
  it -- the class-body definition this pass trusts is simply not what runs. `setattr`/
  `delattr` (no ordinary `Attribute` store at all) and `.__dict__` (same effect via
  subscript) are now in `FORBIDDEN_NAMES`/`FORBIDDEN_ATTRS` -- `screen()`'s own existing,
  whole-program checks already refuse the entire plan the moment either appears anywhere,
  which is the general fix (protects `trip`/`quotient` too, not just this new mechanism).
  Bounded by `_MAX_STR_DEPTH` (6 hops) so a pathological chain refuses rather than
  recursing forever, and memoized (`_Planner.str_seen`, mirroring `trip_variable`'s own
  `seen` set) so mutual recursion terminates too. `prove.py`'s `str`-kind domain collapses
  the draw to `["", "x"]` -- BOTH the empty and non-empty case, explicitly, never a single
  stand-in value (the owner's [internal ref] lesson: truthiness is itself value-dependent).

  This gets `bm_raytrace_nondet.py`'s own `args_filename` (-> `bench_raytrace`'s `filename`
  parameter -> `Canvas.write_ppm`'s own `filename` parameter -> `open()`) all the way
  through the str-draw check -- but the task's other two `nondet_int` draws,
  `args_width`/`args_height`, then refuse at `Canvas(width, height)` (a CLASS
  CONSTRUCTOR, which `trip_argument`'s existing resolution does not cover -- only a plain
  function) and, even if that were added, at `Canvas.__init__`'s own
  `array.array('B', [0] * (width * height * 3))` / `range(width * height)` (the PRODUCT of
  two independent trip draws used as another loop's own range bound -- a materially
  different, two-dimensional generalisation of the "observe repeat, generalise" stationary
  proof technique this module does not attempt). **`bm_raytrace_nondet.py` is not proven by
  this alone** -- these two blockers are real, separate, and not attempted here; recorded
  for whoever picks this up next, not mischaracterised as "just the str draw" the way an
  earlier note on this ticket did before this branch's own campaign surfaced them.

Verified: `tests/test_drawflow_str.py` (22 cases: the basic truthy/open-path shape and its
negative controls, the raytrace function+method chain and EVERY one of its own negative
controls -- inheritance, a rebound receiver, `__getattr__`, `*args`, an over-long chain,
mutual recursion, AND (MG's PR 108 review) instance/class attribute reassignment,
`__init__` shadowing, `setattr` -- each reproducing one of MG's own attack shapes
verbatim, [host]:`[local path]`), `tests/test_output_effects.py`
(preamble ordering, finally-only Try). Full suite 4968/0/37 skipped. Direct `tharkun.sv`
comparison, sv_2027 tip vs. this branch, every (task, property) pair in Real-world +
Solve-only: 0 gained/0 lost/0
new_wrong -- a real capability, safe and tested, but not yet a scored gain on this suite.

## Import boundary ([internal ref]): read-and-recurse, not allow-by-name

`solve/pyresolve.py::Resolver` (shared by `proof:no-site`'s `census.py::site_census` and
`proof:stationary`'s `determinism.py::determinism_census`, above) used to raise `Top`
unconditionally on any import outside its allowlist (`census.py::MODULES` /
`determinism.py::DET_MODULES`, both stdlib-only). A library function bundled WITH the task --
GUUT/python-string-utils-shaped, `sys.path.insert(0, "<project src dir>")` then
`from a.b import f` -- could therefore never contribute a proof either way.

`solve/extern.py` gives an import ONE more chance before `Resolver` refuses: found via a
directory the PROGRAM ITSELF adds to `sys.path` (a literal, static `sys.path.insert(0, D)`,
module-level, unconditional, before the import it enables -- never an ambient path, never
site-packages, never anything a package index could resolve differently between the analysis
run and the replay run)? `Resolver` then recursively builds a CHILD `Resolver` for that file
(cached by resolved absolute path, cycle-guarded), exposed via `Resolver.externs` (local alias
-> child `Resolver`) and `Resolver.all_resolvers()` (this file plus every recursively-read one,
transitively) -- both `site_census` and `determinism_census` now walk EVERY tree
`all_resolvers()` reaches for its own sites/effects, not only the program's own, so a library's
own top level is program code too, exactly like a C SV-COMP task's own multi-file source. A
package import runs `__init__.py`'s top level too; an INERT one (imports/defs/literal-assigns
only, `extern.py::toplevel_inert`) is treated as contributing nothing further, unopened -- this
is what makes a leaf submodule provable when the package's own `__init__.py` re-exports
siblings the program never actually loads a name from. A C extension or a
namespace-package-without-`__init__.py` refuses outright, no compromise.

`src/tharkun/solve/absint/interp.py::Analyzer.module_attr`/`src/tharkun/solve/absint/interp.py::Analyzer.call` share the identical
predicate (a SEPARATE `Resolver` built once per `Analyzer` run purely for its file-finding,
`src/tharkun/solve/absint/interp.py::Analyzer._extern_resolver`) -- see [ABSINT.md](ABSINT.md)'s own "Import
boundary" section for that gate's own plumbing (`module_key`, allocation-site
file-discrimination). A program that never establishes a root is unaffected in EITHER gate,
byte-for-byte -- confirmed by a full-budget `run-tasks` campaign against sv_2027 `effcdb1`
(branch head `f68fa74`, `20260928-222749-921165-[host]-218a532a`): RW 0 gained/0 lost/0
new_wrong (both scorings, 16 -> 16 official / 118 -> 118 adjusted); float 0/0/0 (12 -> 12);
SO 0 lost/0 new_wrong with 1 GENUINE, CONFIRMED gain (`so_base3_bug/data-lookup`: unknown ->
`false(IndexError)`, replay CONFIRMED, 22 -> 23) via `hybrid:atheris` -- a SEARCH technique,
not a static prover this change touches, so most plausibly incidental scheduling/timing
variance rather than an effect of read-and-recurse itself; none of the suite's own tasks use
`sys.path.insert`, so this change is a complete no-op for every one of them regardless.
Negative controls for every condition (a
literal-path, an unconditional-insert, a shadowing-root, an imported-code-runs-too, and a
package-semantics probe, plus a real-crash negative control) live in
`tests/test_import_boundary.py`.

## os / os.path ([internal ref]): a three-way split, not a blanket refusal

`import os` used to make `pyresolve.py::Resolver._imports` raise `Top` unconditionally (`os`
was in neither `census.py::MODULES` nor `determinism.py::DET_MODULES`) -- a program could
never get a single check proven anywhere in the same file, whatever it actually did with
`os`. Full detail (the three buckets, the `os.environ`-as-a-subscriptable-mapping trap it
avoids, the `pyresolve.py::Resolver._module_of` one-level submodule fix `os.path` needed) is
in [ABSINT.md](ABSINT.md)'s own "os / os.path" section -- summary: `os`/`os.path` are now
allowlisted in all three (`census.py::MODULES`, `determinism.py::DET_MODULES`,
`interp.py::MODULES`), split per-attribute into PURE (`os.path.join`/`dirname`/`basename`/
`splitext`/`normpath`/`abspath` and the platform constants -- exact when every argument is a
known literal, else a sound havoc string, never a refusal; NOT `realpath`, which looks pure
but dereferences symlinks against the real, live filesystem -- host-dependent, so it is
HAVOC even for an absolute literal, MG's PR 106 review), HAVOC (`getenv`/`getcwd`/`listdir`/
`urandom`/`getpid`/`cpu_count`/`realpath`/`os.path.exists`/`isfile`/`isdir` -- the call never
refuses, but its argument TYPE is still checked (a definitely-wrong type, e.g.
`os.getenv(5)`, is a checked TypeError, not a silent havoc RESULT -- also MG's PR 106
review) and `determinism.py::check_os` refuses the DETERMINISM claim unconditionally, the
same stance already taken for `os.urandom`/unseeded `random`), and REFUSED (`os.environ`
used bare, `system`, `remove`, `chdir`, `mkdir`, `fork`, `kill`, `exec*`, `putenv`,
`os.open`, ...).

Measured ([internal ref], [host]): 0/24 Real-world (pyperformance) + 0/47 Solve-only
tasks import `os` today -- this unblocks FUTURE real-world tasks and the [internal ref] GUUT-benchmark
import-boundary work (`python-string-utils`'s `manipulation.py` uses `os`), not a gain on the
CURRENT suite. Direct `python -m tharkun.sv` comparison, before (`610fcec`) vs. after, every
(task, property) pair in both sets: 0 gained/0 lost/0 new_wrong (one apparent "lost" on
`so_base3_bug`/data-lookup, `hybrid:atheris`, is pre-existing search-timing flakiness --
confirmed by re-running the SAME (after) build three times and seeing UNKNOWN/FALSE/UNKNOWN,
not something this change caused). Never-TRUE coverage: `tests/test_os_module.py` and two new
lane F fixtures, `tests/fixtures/determinism/never_certified/d27_os_getenv.py` /
`tests/fixtures/determinism/never_certified/d28_os_path_exists.py`, alongside the pre-existing `tests/fixtures/determinism/never_certified/d04_urandom.py`/`tests/fixtures/determinism/never_certified/d07_environ.py` (the
`os.environ`-read one that caught the mapping-vs-module-tag trap above).

## The program's own namespace and the import system are certifiable ([internal ref])

The child is a native `__main__` ("How it decides"), so a program that asks `__main__` WHAT it is, or asks the import system for a module
that is not there, reads values that are the same on every run and the same natively. The determinism census
(`solve/determinism.py::determinism_census`, on `solve/pyresolve.py::Resolver`) refused all of them ("imports types", "uses `__builtins__`",
"attribute `__name__` resolves to nothing known", "imports spec") and four MIN pairs (TRUE, natively verified: `ch_main_builtins_module_true`,
`cmp_m05_builtins_module`, `ch_main_no_foreign_modules_true`, `cmp_m13_import_spec`) stayed UNKNOWN. What resolves now, and what does not:

* `types` resolves with ONE attribute, `types.ModuleType` (`pyresolve.py::RESTRICTED_MODULE_ATTRS`; `determinism.py::DET_MODULES` and
  `_ADDRESS_MODULES` list it). `types.FunctionType`, `MappingProxyType`, `new_class`, ... refuse. A module object has no address model, so
  `print(types.ModuleType("m"))` still refuses.
* `__builtins__` resolves ONLY as the argument of `isinstance(__builtins__, T)` or `type(__builtins__)` in the TOP-LEVEL program
  (`Resolver._builtins_type_probe`): there it is the `builtins` module, as under `python program.py` (`_child.py::native_main_module`); an imported module
  sees a dict, so a recursively-read file refuses it. A rebinding of the name, a shadowed `isinstance`/`type`, any other argument position, an attribute,
  a subscript or a store refuse (they reach `eval` and the like).
* `.__name__`, `.__qualname__`, `.__doc__`, `.__module__` and `.__init__` (`pyresolve.py::DUNDER_OK`) need no further name in the known attributes: a
  string or a program method. Every other dunder stays reflective.
* The text of a LITERAL (`assert False, "msg"`, `str(5)`) shows no address, whatever else the file does (`determinism.py::_Census.check_text`): the
  provenance engine refuses a whole file with an `except` handler, which made every `assert x, "msg"` next to a `try` unprovable.
* `import X` of a module a run OBSERVED absent resolves (`Resolver.absent_imports`): `import spec` is a `ModuleNotFoundError` on every run in this
  environment, exactly as natively (`native_sys_path` keeps `tharkun/sv` out of `sys.path`). The child puts a recorder LAST on `sys.meta_path`
  (`_child.py::watch_imports`, `_AbsentRecorder`): it is asked only for a name every other finder did not have. At the end of the run
  `_child.py::still_absent_imports` drops every name that has become importable (a program that writes `spec.py` and imports it has code the
  census never read behind the second `import`), every dotted name, and everything if the program left the import machinery unusable. The names
  travel as `engine.py::Outcome.absent_imports` (`run_once`, `forkserver.py`); `scheduler.py::verify` keeps the union over runs (`absent_seen`) and hands it to
  `engine.py::_certify_determinism(..., absent_imports=...)`. Only a plain `import X` resolves: `from X import y`, `import X.y` and any use of the
  failed name (`X.run()`) refuse as before. Not native, and not flagged: `sys.meta_path` has one more entry than under `python program.py` (a program that
  reads it is refused by the census: `meta_path` is not in `SYS_OK`).

Tests: `tests/test_census_main_namespace.py` (the four pairs against native truth, the refusals above, the child's recorder, a module the program
creates and imports, an existing sibling `spec.py`, four reads that still find a native violation, three negative controls).

## What the search stages know about the code a product ships ([internal ref])

A multi-file task -- a `.yml` with `required_files`, or a product program such as `assert f_orig(s) == f_mut(s)`
over two copies of a library -- runs code that is not in the file the verifier was handed. Before [internal ref] the
fuzzer instrumented the program file alone, and its seeds and dictionary came from that file's own AST, so for a
product (whose file carries no literal) it was blind: libFuzzer's coverage never moved, and nothing told it that
the library compares the draw with `'"'`, `'@'` and `'..'`.

`sv/reach.py` (stdlib only, executes nothing) answers three questions:

1. **What does the program import?** `reached` follows `import`/`from` (any depth, relative imports, a package
   before its submodules) to source files under the program's directory, the directories the program adds to
   `sys.path` with a literal `sys.path.insert`/`append`, and the directories of the `.yml`'s declared files. The
   standard library and site-packages are not reached: an import that resolves to no task file is someone
   else's code.
2. **What does it call?** `called` is a static call graph over those modules (`import a.b as c`, `from a import
   f`, dotted `a.b.f`, a package `__init__` re-exporting or star-importing its submodules, a class bringing its
   methods): the definitions the program reaches and the module-level assignments of the names they read (a compiled
   pattern and the string it is built from). A package imports every module it has; the product calls
   `is_email`, `is_full_string` and two patterns. A hint, not a proof: a method called on an object is not followed.
3. **What do those definitions say about strings?** `guard_tokens` (the first argument of `startswith`, `endswith`,
   `split`, `partition`, `find`, `count`, `replace`, `strip`... in every arity form, `'x' in s`, `s == 'x'`),
   `regex_patterns` (constant-folded through module-level names, `format`, f-strings and `+`, with flags),
   `regex_tokens` (literal runs and class members of a pattern), `dictionary`.

`sv/hybrid.py` uses them for both `search` and `deepen` (`_Plan`, `_reach_plan`): the worker instruments the
top-level packages of the code the program calls (`atheris.instrument_imports(include=...)`, installed around the
program's first run, so coverage is measured INSIDE the library); libFuzzer gets a dictionary of the guards'
tokens and the patterns' literal runs (64 entries at most, guards first: libFuzzer picks an entry uniformly, so every
entry that does not matter dilutes the ones that do -- on the m131 target the 6 guard tokens found the kill in
seconds, a 180-token dictionary of every literal and class member did not in 60 s); and the corpus holds the
program's and the called code's string literals, a few SHORTEST strings each of its regexes accepts
(`_regex_samples`, through `solve/condition.py::_generate_from_regex`), and `_composed_seeds`: those samples
with the guards' tokens wrapped round the leading segment (`"a"@bc.de`), round the whole sample, and inserted at
every boundary. A regex sample satisfies the pattern but no guard, a guard token satisfies one guard and breaks
the pattern; a corpus entry that has both is zero mutations from a kill that needs both (a quoted local part in
front of a valid domain), where a fuzzer starting from either needs several lucky ones. The regex-positive
vectors of [internal ref] (`_regex_vectors`) now draw from the reached patterns too. Everything is a hint: a
candidate is replayed in a fresh child before it can mean anything. `tests/test_sv_reach.py` covers resolution,
the call graph, the tokens and patterns, the composed seeds, and the end-to-end kill of a product over a shipped
library.

## Files and how they relate

Measured from the committed source in this branch by `scripts/doc_import_map.py::compute_ast_imports`, including lazy imports.

### Subsystem: `tharkun.sv` (40 files)

- `src/tharkun/sv/__init__.py`
  - **Role**: Package root for `tharkun.sv` (DRIVE layer; docs/LAYERS.md).
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`, `src/tharkun/sv/spec.py`
  - **Is imported by (tharkun.sv)**: none
- `src/tharkun/sv/__main__.py`
  - **Role**: CLI entry point (`python -m tharkun.sv`) for verifying and replaying SV properties. It calls `spec.py::program_of` / `spec.py::read_property`, then `engine.py::verify` (or `engine.py::replay` with `--replay`).
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`, `src/tharkun/sv/spec.py`, `src/tharkun/sv/stages.py`, `src/tharkun/sv/validate.py`
  - **Is imported by (tharkun.sv)**: none
- `src/tharkun/sv/_child.py`
  - **Role**: Child-process execution model for `_sv_verifier` draws, random interposition, run classification, isolation, and replay. Imports no `tharkun` module (`tests/test_layering.py::test_the_sv_child_imports_no_tharkun`).
  - **Imports (tharkun.sv)**: none
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/_concolic_worker.py`, `src/tharkun/sv/_forkserver.py`, `src/tharkun/sv/_hybrid_worker.py`, `src/tharkun/sv/_probe.py`, `src/tharkun/sv/engine.py`, `src/tharkun/sv/nondet_harness.py`
- `src/tharkun/sv/_concolic_worker.py`
  - **Role**: Helper process for concolic solving: executes harness and drives to target sites; the one `sv/*` file that imports another Tharkun layer (`tharkun.driver`), running in its own capped process.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_child.py`, `src/tharkun/sv/_symexec.py`, `src/tharkun/sv/sites.py`
  - **Is imported by (tharkun.sv)**: none
- `src/tharkun/sv/_forkserver.py`
  - **Role**: Preloaded child forkserver: pre-initializes Python state and forks fast replay children, importing sibling `_child.py`.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_child.py`
  - **Is imported by (tharkun.sv)**: none
- `src/tharkun/sv/_hybrid_worker.py`
  - **Role**: Subprocess fuzzer worker for Atheris/libFuzzer-driven obligation search. Installs the engine child's environment model (`src/tharkun/solve/environment.py::install_files`, imported by bare name with self-cleaning `sys.path` insert) with stdin `/dev/null` (`tests/test_sv_hybrid_sandbox.py`). Imports no `tharkun` module.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_child.py`
  - **Is imported by (tharkun.sv)**: none
- `src/tharkun/sv/_probe.py`
  - **Role**: Observation child process for stationarity proofs: imports sibling `_child.py` for the same draw model, adding the loop wrapper, snapshot, quotient moduli, and output-file model (`_OutputFile`, `_output_open`) enabled through `prove.py::run_probe`. Imports no `tharkun` module (`tests/test_layering.py::test_the_sv_child_imports_no_tharkun`).
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_child.py`
  - **Is imported by (tharkun.sv)**: none
- `src/tharkun/sv/_symexec.py`
  - **Role**: Z3-based symbolic execution engine the concolic worker calls: `solve_targets` (whole-harness run) and `solve_site` (one site's backward slice); also provides `run_exact` for product proofs.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_symexec_objects.py`, `src/tharkun/sv/symexec_visitor.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/_concolic_worker.py`, `src/tharkun/sv/_symexec_math.py`, `src/tharkun/sv/_symexec_objects.py`, `src/tharkun/sv/product_proof.py`, `src/tharkun/sv/string_regex.py`, `src/tharkun/sv/symexec_visitor.py`
- `src/tharkun/sv/_symexec_math.py`
  - **Role**: Consumer of `math` contracts through `symexec_visitor.py::SymexecVisitor.visit_math_function`, which lazily calls `_symexec_math.py::call`; constants use `visit_math_constant`. Reads the concolic math adapters; covered by `tests/contracts/test_adapters_concolic_math.py`.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_symexec.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/symexec_visitor.py`
- `src/tharkun/sv/_symexec_objects.py`
  - **Role**: Model of program classes (heap, live classes, named refusals) for `_symexec.py::_Runner` ([internal ref]); relates to `src/tharkun/solve/call_targets.py` via the `call_target` fact.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_symexec.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/_symexec.py`
- `src/tharkun/sv/_vendored_py_method_lift.py`
  - **Role**: Vendored helper for Python method signature lifting and parameter binding.
  - **Imports (tharkun.sv)**: none
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/method_lift.py`
- `src/tharkun/sv/avengers_seed.py`
  - **Role**: The `avengers` seed stage ([internal ref]). Constructs classes the target module defines with `tharkun.avengers.synthesize_object` (Pynguin backend included) and offers back validated recipe arguments as candidates; the only `sv/*` module importing `tharkun.avengers` (`tests/test_layering.py::test_only_named_drive_callers_import_avengers`), never importing backends or `pynguin`/`black` directly (`tests/test_sv_no_backends.py`). Out of `stages.py::DEFAULT`. See [AVENGERS.md, Backends](AVENGERS.md#backends) for the backend measurement and keep/omit decision.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/stages.py`
- `src/tharkun/sv/concolic.py`
  - **Role**: Concolic solving driver (`solve`, `Solved`). Writes the draw-parameterized harness through `sites.py::build_harness` and starts `_concolic_worker.py` in a subprocess. Called through `scheduler.py::run` and `stages.py::_concolic`; reports named harness refusals.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`, `src/tharkun/sv/sites.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/hybrid.py`, `src/tharkun/sv/scheduler.py`, `src/tharkun/sv/stages.py`
- `src/tharkun/sv/contracts.py`
  - **Role**: Declarative portfolio stage registry, order, read/write dependencies, costs, and fact kind definitions (`KINDS`); imports nothing; covered by `tests/test_sv_contracts.py`.
  - **Imports (tharkun.sv)**: none
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/obligation_table.py`
- `src/tharkun/sv/engine.py`
  - **Role**: Verification entry point and shared verdict/replay types (`verify`, `run_once`, `Outcome`, `Verdict`, `Candidates`). `verify` delegates to `scheduler.py::run`; `run_once` starts `_child.py` in a subprocess.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_child.py`, `src/tharkun/sv/prove.py`, `src/tharkun/sv/scheduler.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/__init__.py`, `src/tharkun/sv/__main__.py`, `src/tharkun/sv/avengers_seed.py`, `src/tharkun/sv/concolic.py`, `src/tharkun/sv/eregion_bridge.py`, `src/tharkun/sv/fact_producers.py`, `src/tharkun/sv/fact_proof.py`, `src/tharkun/sv/forkserver.py`, `src/tharkun/sv/guard_values.py`, `src/tharkun/sv/ledger.py`, `src/tharkun/sv/nondet_harness.py`, `src/tharkun/sv/product_proof.py`, `src/tharkun/sv/proof.py`, `src/tharkun/sv/prove.py`, `src/tharkun/sv/reach.py`, `src/tharkun/sv/scheduler.py`, `src/tharkun/sv/solver_seeds.py`, `src/tharkun/sv/stages.py`, `src/tharkun/sv/store_witness.py`
- `src/tharkun/sv/eregion_bridge.py`
  - **Role**: Candidate fact ingestion (`publish_fact`, `publish_certified`) into Tharkun's `knowledge.py::KnowledgeBase`; reads `solve/obligations.py::Invariant`. External producers communicate through the separate process/JSON seam.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/fact_proof.py`, `src/tharkun/sv/stages.py`
- `src/tharkun/sv/exact_subset.py`
  - **Role**: Exact AST subset validator (`audit`) admitting only modeled product-proof constructs. Reads `exact_unroll.py`, contract rules (`Refused`, `get_rule`), and `ProductVisitor` for proof admission.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/exact_unroll.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/product_proof.py`
- `src/tharkun/sv/exact_unroll.py`
  - **Role**: Static, bounded exact loop unrolling for the product prover: literal tuples, ranges, alias-free lists, and counter loops; named refusal when the exact subset cannot be preserved.
  - **Imports (tharkun.sv)**: none
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/exact_subset.py`, `src/tharkun/sv/product_proof.py`
- `src/tharkun/sv/fact_producers.py`
  - **Role**: Fact producer discovery and process isolation driver; runs external invariant producers and collects candidate facts; imports `engine.py::_cap_memory` lazily.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/fact_proof.py`
- `src/tharkun/sv/fact_proof.py`
  - **Role**: Stage 3b fact-based proof consumer (`prove_from_facts`). Coordinates discovery via `fact_producers.py`, store ingestion via `eregion_bridge.py`, and independent validation via `validate.py::check_candidates_individually`.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`, `src/tharkun/sv/eregion_bridge.py`, `src/tharkun/sv/fact_producers.py`, `src/tharkun/sv/obligation_table.py`, `src/tharkun/sv/proof.py`, `src/tharkun/sv/store_witness.py`, `src/tharkun/sv/validate.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/scheduler.py`
- `src/tharkun/sv/forkserver.py`
  - **Role**: Process lifecycle manager (`ForkServer`) managing persistent preloaded forkservers (`_forkserver.py`) across threads.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/scheduler.py`
- `src/tharkun/sv/guard_values.py`
  - **Role**: The `static:guard_values` seed stage. Imports `src/tharkun/solve/arith.py` (`free_names`, `solve_int_guard`) and, lazily, `src/tharkun/solve/condition.py`.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/solver_seeds.py`, `src/tharkun/sv/stages.py`
- `src/tharkun/sv/hybrid.py`
  - **Role**: Portfolio search stage (`search`, `deepen`). Starts `src/tharkun/sv/_hybrid_worker.py` as a subprocess through `tharkun.hybrid.backend::AtherisBackend`, in the stage's own temporary directory (`cwd=`), with `-artifact_prefix` inside it (`artifact_dir=`): libFuzzer writes its `crash-*`, `leak-*`, `timeout-*`, `oom-*` and `slow-unit-*` files to the prefix, which defaults to the subprocess's current directory -- before [internal ref] they landed in whatever directory `tharkun.sv` was launched from. The backend always passes a prefix (its own private temporary directory when the caller names none, removed at `stop()`); `tests/hybrid/test_backend_artifacts.py` holds that. The worker runs the program for real, so it installs the engine child's environment model (`src/tharkun/solve/environment.py::install_files`, imported by bare name with the same self-cleaning `sys.path` insert as `src/tharkun/sv/_child.py`; declared files arrive as `Context.files` through `request.json`'s `files`), and its stdin is `/dev/null`: a program that opens, truncates or writes a file it does not own gets the same refusal there as in a replay (before this, a verify with a few seconds of budget truncated such a file on the host); `tests/test_sv_hybrid_sandbox.py` holds that. Imports `src/tharkun/sv/reach.py`, `tharkun.hybrid.backend::AtherisBackend`, and lazily `src/tharkun/solve/census.py::obligation_census`.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/concolic.py`, `src/tharkun/sv/reach.py`, `src/tharkun/sv/stages.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/stages.py`
- `src/tharkun/sv/ledger.py`
  - **Role**: CPU-second and wall-clock time ledger (`Ledger`, [internal ref] step 3a); tracks spent/remaining budgets, live forkserver CPU, and compresses stage caps. Reads clocks via `engine.py`.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/scheduler.py`
- `src/tharkun/sv/method_lift.py`
  - **Role**: Method extraction and signature lifter for candidate invariant validation.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_vendored_py_method_lift.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/validate.py`
- `src/tharkun/sv/nondet_harness.py`
  - **Role**: Standalone harness runner and classifier wrapping `_child.py` for isolated nondet draws.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_child.py`, `src/tharkun/sv/engine.py`
  - **Is imported by (tharkun.sv)**: none
- `src/tharkun/sv/obligation_table.py`
  - **Role**: Live obligation status ledger (`ObligationTable`) in a `KnowledgeBase`; imports `contracts.py::KINDS` to refuse unregistered kinds. Stages read and write open/solved rows.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/contracts.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/fact_proof.py`, `src/tharkun/sv/prove.py`, `src/tharkun/sv/scheduler.py`
- `src/tharkun/sv/product_proof.py`
  - **Role**: Exact product proof engine (`prove`, stage 1b, [internal ref]/[internal ref]). Audits AST with `exact_subset.py::audit`, evaluates with `_symexec.py::run_exact`, and solves UNSAT/SAT with Z3; accepts draws inside defs ([internal ref]). Imports `sv/sites.py::LIFTED` at module level, `sv/engine.py` and `sv/sites.py::violation_sites` lazily.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_symexec.py`, `src/tharkun/sv/engine.py`, `src/tharkun/sv/exact_subset.py`, `src/tharkun/sv/exact_unroll.py`, `src/tharkun/sv/sites.py`, `src/tharkun/sv/store_witness.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/scheduler.py`
- `src/tharkun/sv/proof.py`
  - **Role**: Proof finalization (`finish`): checks candidate invariants through `validate.py` and exports confirmed Python correctness witnesses through `witness.py` in format 2.3.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`, `src/tharkun/sv/store_witness.py`, `src/tharkun/sv/validate.py`, `src/tharkun/sv/witness.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/fact_proof.py`, `src/tharkun/sv/scheduler.py`, `src/tharkun/sv/store_witness.py`
- `src/tharkun/sv/prove.py`
  - **Role**: Proof stage orchestrator: runs `prove_static` (census and absint) and `prove_dynamic` (`proof:stationary`). `prove_dynamic` calls `src/tharkun/solve/determinism.py::determinism_census` and `src/tharkun/solve/drawflow.py::plan_draws`, then launches `_probe.py` through `run_probe`.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`, `src/tharkun/sv/obligation_table.py`, `src/tharkun/sv/store_witness.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/engine.py`, `src/tharkun/sv/scheduler.py`
- `src/tharkun/sv/reach.py`
  - **Role**: Static reachability and call-graph analysis ([internal ref]). Reads imports and call graphs (stdlib only: `ast`, `os`, `re`, `string`), extracting tokens, regex patterns, and seed dictionaries. Called by `src/tharkun/sv/hybrid.py`. Imports `src/tharkun/sv/engine.py::read_source` and lazily `src/tharkun/solve/environment.py::TaskEnvironment`.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/hybrid.py`
- `src/tharkun/sv/scheduler.py`
  - **Role**: Stage execution orchestrator (`run`, [internal ref]). Coordinates provers (`prove.py`, absint, `product_proof.py`), the obligation table (`obligation_table.py::ObligationTable`), the ledger (`ledger.py::Ledger`), the forkserver (`forkserver.py::ForkServer`), solving portfolio stages (`stages.py`), fact proofs (`fact_proof.py`), and canary runs.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/concolic.py`, `src/tharkun/sv/engine.py`, `src/tharkun/sv/fact_proof.py`, `src/tharkun/sv/forkserver.py`, `src/tharkun/sv/ledger.py`, `src/tharkun/sv/obligation_table.py`, `src/tharkun/sv/product_proof.py`, `src/tharkun/sv/proof.py`, `src/tharkun/sv/prove.py`, `src/tharkun/sv/spec.py`, `src/tharkun/sv/stages.py`, `src/tharkun/sv/store_witness.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/engine.py`
- `src/tharkun/sv/sites.py`
  - **Role**: AST analysis only (stdlib; executes nothing): finds violation sites and synthesizes draw-parameterized harnesses (`build_harness`, `harness_refusal`, `violation_sites`). Imports `src/tharkun/solve/call_targets.py::call_target_index` lazily inside `_entry` ([internal ref]). Reached via `src/tharkun/sv/concolic.py` and `src/tharkun/sv/product_proof.py`.
  - **Imports (tharkun.sv)**: none
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/_concolic_worker.py`, `src/tharkun/sv/concolic.py`, `src/tharkun/sv/product_proof.py`
- `src/tharkun/sv/solver_seeds.py`
  - **Role**: Portfolio solver seeds stage (`_solver_seeds`). Derives boundary candidates from static guard conditions and scopes via `src/tharkun/sv/guard_values.py`.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`, `src/tharkun/sv/guard_values.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/stages.py`
- `src/tharkun/sv/spec.py`
  - **Role**: Specification parser for `.prp` property files and format-2.0 `.yml` task declarations. Uses stdlib only (no YAML library).
  - **Imports (tharkun.sv)**: none
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/__init__.py`, `src/tharkun/sv/__main__.py`, `src/tharkun/sv/scheduler.py`, `src/tharkun/sv/validate.py`
- `src/tharkun/sv/stages.py`
  - **Role**: Portfolio registry (`REGISTRY`, `Context`, `Stage`, `select`). Manages stage functions (`_concolic`, `_hybrid`, `_solver_seeds`, `_guard_values`, `_avengers`, `_producer`) and lazily invokes stage modules.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/avengers_seed.py`, `src/tharkun/sv/concolic.py`, `src/tharkun/sv/engine.py`, `src/tharkun/sv/eregion_bridge.py`, `src/tharkun/sv/guard_values.py`, `src/tharkun/sv/hybrid.py`, `src/tharkun/sv/solver_seeds.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/__main__.py`, `src/tharkun/sv/hybrid.py`, `src/tharkun/sv/scheduler.py`
- `src/tharkun/sv/store_witness.py`
  - **Role**: Experimental store-derived correctness witness. Projects the obligation table's site records into witness invariants (`project`), records and binds proof routes (`record_route`, `bind_route`), and, after the frozen checker and every legacy export, writes `store-witness.yml` next to `witness.yml` (`shadow`; skipped with `--no-store-witness`). Additive: a failure here never changes a verdict or `witness.yml`.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/engine.py`, `src/tharkun/sv/proof.py`, `src/tharkun/sv/witness.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/fact_proof.py`, `src/tharkun/sv/product_proof.py`, `src/tharkun/sv/proof.py`, `src/tharkun/sv/prove.py`, `src/tharkun/sv/scheduler.py`
- `src/tharkun/sv/string_regex.py`
  - **Role**: Regex pattern constraint solver generating string models for concolic execution.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_symexec.py`
  - **Is imported by (tharkun.sv)**: none
- `src/tharkun/sv/symexec_visitor.py`
  - **Role**: Contract-rule visitor (`CallCtx`, `SymexecVisitor`) read through `_symexec.py::_Exec._visit_rule` / `_through_rule`. Dispatches modeled builtins, methods, module functions and math; calls the executor's `_model_<name>`, `_contract_fold_model`, `_contract_call_model`, and `_open_output_model` seams. Absent visits refuse.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/_symexec.py`, `src/tharkun/sv/_symexec_math.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/_symexec.py`
- `src/tharkun/sv/validate.py`
  - **Role**: Candidate invariant validator (`check_candidates`, `check_candidates_individually`, `Validation`). Imports `src/tharkun/solve/census.py::obligation_census`, `src/tharkun/solve/obligations.py` (`Invariant`, `source_points`), `src/tharkun/solve/predicates.py` (`BUILTINS`, `check_predicate`), `src/tharkun/sv/spec.py::parse_property`, and `src/tharkun/sv/witness.py::loads`.
  - **Imports (tharkun.sv)**: `src/tharkun/sv/method_lift.py`, `src/tharkun/sv/spec.py`, `src/tharkun/sv/witness.py`
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/__main__.py`, `src/tharkun/sv/fact_proof.py`, `src/tharkun/sv/proof.py`
- `src/tharkun/sv/witness.py`
  - **Role**: Python correctness-witness format 2.3 serialization and source/hash/location metadata; candidates reach it only after proof confirmation.
  - **Imports (tharkun.sv)**: none
  - **Is imported by (tharkun.sv)**: `src/tharkun/sv/proof.py`, `src/tharkun/sv/store_witness.py`, `src/tharkun/sv/validate.py`

### Solve modules SV uses

- `src/tharkun/solve/pyresolve.py`: `Resolver`, `Top`. Name and import resolution (stdlib only); `determinism.py` and `census.py` build on `pyresolve.py::Resolver`.
- `src/tharkun/solve/determinism.py`: `determinism_census`, `DET_MODULES`. Parse and census, no execution; called by `sv/prove.py::prove_dynamic` (`proof:stationary`) and certified via `sv/engine.py::_certify_determinism`.
- `src/tharkun/solve/census.py`: `site_census`, `obligation_census`. Builds on `pyresolve.py::Resolver`; `sv/prove.py::prove_static` calls `census.py::site_census`; `sv/engine.py::verify` and `sv/validate.py` call `census.py::obligation_census`.
- `src/tharkun/solve/drawflow.py`: `plan_draws`. SOLVING (stdlib, AST only); `sv/prove.py::prove_dynamic` calls `plan_draws`; internal `_Planner.screen` calls `output_effects.py::output_only`.
- `src/tharkun/solve/output_effects.py`: `output_only`, `OUTPUT_MODES`. SOLVING (stdlib, AST only); called by `drawflow.py` screen analysis.
- `src/tharkun/solve/extern.py`: Read-and-recurse file-finding (`[internal ref]`, "Import boundary" above); `pyresolve.py::Resolver` and `src/tharkun/solve/absint/interp.py::Analyzer` both call into it; imports only `ast` and `os`.
- `src/tharkun/solve/module_namespace.py`: `module_constants`, `regex_globals` (`#1132b`). Stdlib-only SOLVING (`ast`, `os`, `re`, executes nothing); reads task import graph (entry module, package `__init__` chain, local modules up to `MAX_MODULES`); `sv/scheduler.py::publish_module_namespace` calls both once per run via lazy import (so missing/crashing analysis publishes nothing) and hands records to `obligation_table.py::ObligationTable.publish_module_namespace`, which writes kinds `module_constant` and `regex_pattern` (registered in `contracts.py::KINDS`, component "static facts (module namespace)").
- `src/tharkun/solve/obligations.py`: `Invariant`, `Obligation`, `ObligationCensus`, `source_points`, `witness`. Imports `solve/predicates.py::parse_predicate`; called by `sv/engine.py::verify`, `sv/proof.py::finish`, `sv/validate.py`, and `sv/eregion_bridge.py`.
- `src/tharkun/solve/predicates.py`: `BUILTINS`, `check_predicate`, `parse_predicate`, `InvalidPredicate`. Called by `solve/obligations.py`, `solve/census.py`, and `sv/validate.py`.
- `src/tharkun/solve/environment.py`: `install_files`, `TaskEnvironment`, `open_effect`. Sandbox filesystem environment model used by `_hybrid_worker.py` and `sv/reach.py`.
- `src/tharkun/solve/arith.py`: `free_names`, `solve_int_guard`. Relational integer arithmetic used by `sv/guard_values.py`.
- `src/tharkun/solve/call_targets.py`: `call_targets`, `call_target_index`. Static facts on call targets; used lazily by `sv/sites.py::_entry` and `sv/scheduler.py::publish_module_namespace`.
- `src/tharkun/solve/absint/__init__.py`: `prove`, `AbsintResult`. Whole-program static abstract interpreter called lazily by `sv/engine.py::_static_proof` (see [ABSINT.md](ABSINT.md)).

### Verified tests and harnesses

- `tests/test_sv_stages.py`: covers the registry, the seed ranking and `guard_values`.
- `tests/test_sv_draw_domains.py` ([internal ref]): plants out-of-domain draws (a `str` outside `string.printable`, a `nondet_float_range` outside its bounds) and holds the child equal to the grading harness, row by row.
- `tests/test_sv_avengers_stage.py`: covers `avengers_seed.py`: registration outside `DEFAULT`, degrading to no candidates when Avengers or the backend is unavailable, the restored Pynguin backend actually being invoked, and (negative control) never fabricating a verdict from a recipe the real draws cannot reach.
- `tests/test_sv.py`: exercises each rule on small programs.
- `tests/test_sv_concolic.py`: covers the sites, the harness and the solve stage.
- `tests/test_layering.py::test_sv_imports_only_layers_at_or_below_drive`: applies docs/LAYERS.md's DRIVE rule to `sv/*`.
- `tests/test_layering.py::test_the_sv_child_imports_no_tharkun`: keeps the child and the probe Tharkun-free.
- `tests/test_sv_prove.py`: covers `drawflow`, `determinism` and `proof:stationary`.
- `tests/test_drawflow_str.py` ([internal ref]): covers `nondet_str`'s truthy/open-path classification and its interprocedural propagation specifically.
- `tests/test_sv_fact_producers.py`, `tests/test_eregion_bridge.py`, `tests/test_sv_fact_proof.py`, and `tests/test_sv_validate_abs_module.py`: cover fact proof and validation, using `tests/fake_producers.py` (Eregion's real producer is exercised in the external umbrella repository).
- `tests/test_sv_product_proof.py` and `tests/test_sv_entry_lift.py` ([internal ref]): cover exact product proofs, entry function lifts, and module-level draw lifts.
- `tests/test_sv_contracts.py`: covers `sv/contracts.py` component registrations, reads/writes, cost, and kind definitions.
- `tests/test_sv_ledger.py`: covers CPU-second and wall-clock accounting, caps, and allowances in `sv/ledger.py`.
- `tests/hybrid/test_backend_artifacts.py` and `tests/test_sv_hybrid_sandbox.py` ([internal ref]): cover libFuzzer artifact directory isolation, clean process cleanup, and file sandbox confinement.


## Measured (2026-09-23, [host], CPython 3.14.4, sv-benchmarks `33972db5`)

| Task / property | Verdict | Runs |
| --- | --- | --- |
| bm_float_det, all three properties | TRUE | 2 each |
| bm_float_nondet, all three properties | UNKNOWN (the draw sizes a loop: an unbounded domain) | 23 each |
| bm_chaos_nondet data-lookup | FALSE(IndexError), width 0 | 2 |
| bm_chaos_nondet arithmetic | FALSE(ZeroDivisionError), thickness 0.0 | 1 |
| bm_scimark_nondet data-lookup | FALSE(IndexError), N=1 with nz > N | 35 |
| bm_chaos_det dynamic-typing | FALSE(TypeError), `random.seed(tuple)` | 1 |

## Not built yet

See the umbrella's Beads tickets labelled `tharkun` and `sv-harness`. Not built:

- in the symbolic run: `str` / `bytes` draws and string methods, bit operations on draws (they need
  bit-vectors), loops with `break` / `continue`, `try` statements, methods of classes, and loops that
  need more than `LOOP_BOUND` (64) iterations, and a `return` inside a loop (the harness of an entry function lifts the
  loop's draws as a stream, but `_check_loop` still refuses the loop: `unsupported`, named in the log). Each is `unsupported` (or bounded) there and left to the
  drive and the enumeration; a `range` over a non-int argument or used as a value, `for` over any
  other symbolic iterable, `while t` as `while bool(t)`, and `[... for i in range(n)]` with a
  symbolic `n` for anything but its length (the list-comprehension special case still takes any
  numeric sort);
- sites in functions reached only through non-draw arguments;
- proofs for unbounded draw domains (the may-raise census, #0953);
- **ruled, not built** ([internal ref], [internal ref]): the product proof (`sv/product_proof.py`) does not admit a `while` that is not a
  counter loop, nor `try` (tuples, constant-range `for`, alias-free lists and counter `while` loops are built, [internal ref] PRs 1 to 4;
  the real Tier A tasks that still stay UNKNOWN after the entry-function lift: the boto3 loops bounded by a guarded draw or
  with a `break`, the variant loop of boto3_batch_writer, vllm's doubling loops with `&`, the `http500` tasks, whose
  `try`/`is`/`in`/string constants carry no TRUE-expected pair and stay named refusals). A loop bounded by a guarded draw
  would need copies guarded by the test plus a cut the runner proves unreachable with z3 (ruled, not built: only if it stays
  small and sound); a store through a list parameter stays refused (aliasing). A draw vector for a model whose draws sit in
  defs (the proof returns `None` there instead). The lifts of [internal ref] do not inline an entry call that has
  arguments, a program that does not END with the entry call, or an entry function the `call_target` fact does
  not prove: those are named refusals, not guesses.


### Address provenance

The determinism certificate joins source-derived value facts over names,
receiver fields, container contents, tuple positions, arguments and returns.
It checks all explicit text conversions and output, including keyword
`str(object=...)`, implicit exception text, and discarded conversions. Safe
string repetition/concatenation keeps its address-free provenance. Raw-address APIs such as `array.buffer_info` refuse, including bound aliases
obtained with literal `getattr`. Unknown
callees invalidate reachable mutable values and callback arguments rather than
invalidating unrelated scalar text throughout the program -- with one known
exception ([internal ref] follow-up): `*`/`**` call-argument expansion of a value whose
type is not provably a plain sequence/mapping/range refuses the WHOLE program
(not just the call's own reachable arguments), since Python resolves the
expansion protocol (`keys`/`__getitem__`/`__iter__`) before the callee runs and
that protocol call is not yet scoped the way an ordinary unknown callee is. This
is conservative, never a false certificate, but broader than it needs to be.

Ordinary single inheritance, module globals, tuple argument expansion and
lexical default arguments are modeled. Ambiguous class-local builtin aliases,
dynamic object hooks and unsupported binding forms refuse. Iterator/generator
objects remain distinct from their elements: copying safe elements into a list
is safe, but printing the iterator's own default representation is not.

Library resolution alone is insufficient: some otherwise deterministic
entrypoints invoke hidden string conversions. Address certification currently
admits the modeled numeric, random, regex, array, verifier, selected sys, and
StringIO/BytesIO interfaces; other library entrypoint uses refuse until they
have a provenance/effect model. Unused imports do not taint independent text.
Direct resolved `collections.defaultdict(int)` calls have a narrow model for
local aliases and subscripts with exact integer values and integer or flat
tuple-of-integer keys. Missing reads insert zero. A separate integer analysis
uses the existing all-writes fixpoint and argument binder; the broader scalar
address facts do not establish exact integer types. Factory access, methods,
escapes and other mapping mutations refuse even without a text conversion.
This integer slice also refuses source classes and unmodeled calls, whose
callbacks could mutate the carriers of integer facts.
The mapping itself, including inside another container, remains unprintable:
its factory representation is not covered by integer element provenance.
The independent programs, alpha renamings, negative controls and verifier-level
checks are in `tests/test_address_provenance.py`.

`collections.deque` has a separate closed transport model. Direct constructor
calls (including import aliases) accept no argument or an ordered literal
list/tuple carried through proven aliases. It models `append`, `popleft`,
`extend`, truth tests and length. Elements may be exact primitive values,
literal lists/tuples, or plain source records whose only method is an initializer
that assigns parameters to fields. Every call, default, field write and alias
contributes to the same provenance fixpoint. A source record retains default
identity equality and hashing; empty `set()` with `add`, `update` and membership
is modeled only in this closed deque domain. Set order remains unavailable.

An immediate, single generator passed to `extend` may select elements from a
proven list/tuple, optionally filtered by membership in a proven native set.
Its target uses the generator's scope; its first iterable uses the enclosing
scope. Custom iteration, hash/equality hooks, subclasses, constructor or method
escapes, `maxlen`, mutated source lists, set-derived sequences and unsupported
calls refuse, even without a text sink. Unknown facts never stand for a closed
empty carrier. The model does not make records or whole/nested deques printable;
their default representations still refuse. Native callback controls, scope and
alias cases, and deterministic empty-pop violations are covered in
`tests/test_deque_provenance.py`.

Fresh integer-set differences immediately consumed by zero-argument `pop` have
a separate, bounded proof in `solve/set_singleton.py`. It retains complete exact
native argument rows across direct source calls and literal list/tuple argument
expansion, using ordinary Python signature binding. Every reached row must leave
exactly one element. Defaults, arguments (including rejected arity), both recursive
call sites and all supported effects must close before any receiver is exempted;
only an exact native condition may prune a branch. Recursive list builders prove
native list returns and callback-free elements independently of general provenance.
Builders cannot escape through aliases or parameters. Unknown calls, classes,
identity observations, mutation of argument carriers, unresolved recursive cycles,
empty receiver relations and finite analysis limits refuse. The limits bound the
proof, not the target's inputs. Other set sinks and existing address/library checks
remain applicable. Native adversarial controls and renamed independent examples
are in `tests/test_set_singleton.py`.

A dict lookup whose key is a read of a list that is provably still empty
(`table[stack[-1]]`, `stack = []`, no push anywhere) never shows its key in a
`KeyError`: the key read raises the builtin `IndexError("list index out of range")`
first, a fixed text with no address. `Provenance.inert_empty_index` is the private
certificate for that one conversion in `solve/determinism.py::_Census.check_text`
(a dict-subscript key, never a raise, `print`, `str` or other sink). It is
syntactic and independent of the value facts, whose empty sets also mean
"unmodelled": the list is `name = []` as one of the function's own statements, that
binding is the only one of the identifier anywhere in the function (parameters,
defaults, decorators, nested scopes, `global`/`nonlocal`, imports, loop, `with`,
walrus and `del` targets all count), and every other mention is a truth test (through
`and`/`or`/`not` that only reach an `if`/`while`/`assert`/conditional test), a
`name[<int literal>]` load, or a zero-argument `name.pop()`. Anything that could hand
the list to code that fills it (a call argument, alias, return, container, capture,
bound method) refuses. The certificate says the key is never computed, not that the
lookup is unreachable: a run that reaches it still ends in `IndexError`, and the
verdict for that property is the run's. It is consulted only after the unsafe-binding
veto and every receiver, earlier-sibling and other-conversion check, which are not
changed. Native controls and fill-the-list adversaries are in
`tests/test_inert_empty_index.py`.

`determinism_census(..., class_only_exceptions=True)` judges exceptions by their class
alone, for the one caller whose verdict is exactly that: the deterministic TRUE of
`sv/scheduler.py` (two hash seeds, the same `status`/`exc` class), asked as a separate re-run
after the strict census refused. The default census, the FALSE-witness confirmation and every
other caller are untouched (`tests/test_address_provenance.py`'s pinned refusals, such as
`values = {}; value = values[object()]`, keep passing). Two refusals are lifted, and only
while `determinism.py::observer_reason` finds nothing in any file of the program, dead code
included, that could see an exception in flight (a `try` of any kind, a `with`, a generator,
generator expression or `await`, `__del__` / `__exit__` / `__enter__`, `sys.exc_info` /
`exception` / `excepthook` / `stderr` / `settrace` and kin, `__context__`, `__traceback__`,
`with_traceback`, and imports of traceback, logging, atexit, warnings, threading, contextlib,
signal and similar). With such a program the flag is ignored. Then the text of an uncaught
exception can only reach the stderr of the process that dies of it; the harness records the
class, its builtin MRO and the site, and keeps the message only as a diagnostic string that no
comparison reads (`_child.py`, `scheduler.py`):

* the text of the `KeyError` of `display[key]` (`Provenance.class_only_lookup`): the receiver is
  a dict display and the probe and every stored key hash and compare without running program
  code (int, bool, str, bytes, None, an instance of an identity-only class, a literal tuple of
  those), so the exception is a `KeyError` on every path;
* heaps of `(number, identity-only instance)` tuples (`Provenance._heapq_order_key`): an
  identity-only class has no dunder but `__init__` anywhere on its MRO and no metaclass,
  decorator, slots or descriptor, so `==` is `is`, `<` raises a `TypeError` naming only the class
  and hashing is the object's. Tuple comparison stops at the first slot that is not identical, so
  which pair is compared and whether a tie reaches the instances is a function of the numbers
  and of object identity, never of an address. In this mode only, `enumerate(<builtin container>)`,
  an argument-free `set()` with `add` of plain members, and a function whose last statement
  returns or raises (it cannot fall off and return `None`) are modelled.

Observed today for the reader of this section: the strict census already drops the dict-lookup
text conversion whenever the provenance is `unsafe` (a handler, a lambda, a nested def, a
decorated def or class, ...) or the receiver is not a tracked dict display (`dict()`, a
comprehension, a parameter), so the mode adds nothing there and changes nothing. Native
adversaries (an `__exit__` that suppresses on an address bit, `sys.exc_info` in `finally`, a
`__del__` that reads the exception, an overridden `sys.excepthook`, the `KeyError` chained as
`__context__`) vary between runs, are refused in both modes and never come out TRUE:
`tests/test_class_only_exceptions.py`.

## 3a: one scheduler (built, except the per-obligation loop) ([internal ref])

*Owner ruling 2026-09-28: finalizing hybrid search into one scheduler is a PREREQUISITE for Eregion
in SV-COMP ([internal ref]) -- the components must talk to each other, and the order and budget logic must be
legible in one place instead of hand-written control flow. Plan: staged plan doc, Stage 3, "3a".
Built: step 1 (`sv/scheduler.py::run`, behavior-preserving); step 3a (2026-10-05, [internal ref]/
`.59`): the obligation table is live in the `KnowledgeBase` (`sv/obligation_table.py`), the CPU-second
ledger is one object (`sv/ledger.py`), the component contracts are data that the doc table is generated
from (`sv/contracts.py`), and components read their inputs from the store and write their results to
it instead of passing them through out-parameters and `Context` fields. NOT built, ruled: step 2's
per-obligation loop (see "Order and the CPU-second ledger"), kept only if it gains.*

### Today's order, measured (`sv/engine.py::verify`)

`verify`'s own docstring already names five stages (static, symbolic, execution, portfolio,
enumeration); this is what each one actually reads, writes and may decide, read from the code, not
the docstring.

Before stage 1, `verify` computes `solve/census.py::obligation_census` once, unconditionally (an AST
walk plus `solve/pyresolve.py::Resolver`) and keeps it as `obligations`. This IS a per-task obligation
table already -- see below. Since step 3a the table lives in the run's `KnowledgeBase`
(`sv/obligation_table.py::ObligationTable`, built right after the census), and the stage descriptions
below say what each stage reads from it and writes to it; "Today's order" is the order as built.

1. **static** (`sv/engine.py::_static_proof`, gated by `prove=True`): nothing runs.
   - `sv/prove.py::prove_static` (`solve/census.py::site_census`, a different, coarser census than
     `obligation_census`): instant, no budget. May decide TRUE only (`proof:no-site`).
   - `solve/absint/__init__.py::prove` (`docs/ABSINT.md`): cost `time_budget = max(1.0, min(60.0, remaining()*0.1))`
     -- up to 10% of whatever CPU is left, capped at 60s, spent with NOTHING run. When the joined analysis
     does not prove TRUE and the source draws `nondet_bool`, it runs once more with each eligible boolean draw
     case-split (ABSINT.md "Case splits on a boolean draw"), inside the same grant and on at most
     `max(3 s, 8x)` of the joined run's CPU; the joined result is what an UNKNOWN reports. Reads the source,
     `class_hints` (from a later stage, see below) and `invariant_locations` (from `obligation_census`
     plus every `ast.While` header, via `solve/obligations.py::source_points`). Writes its per-site
     records (site, `ok`/pending, `fact`) as `site_record` Evidence (`ObligationTable.publish_site_records`;
     one generation per run), which the later stages READ from the store
     (`ObligationTable.site_records`: `concolic.solve(..., extra_sites=...)` and the seed stages through
     `Context.site_records`) -- there is no out-parameter any more.
     May decide TRUE only.
   - `_static_proof` is called a SECOND time, conditionally, from inside stage 3 (below) -- not once,
     as the docstring's stage list implies.
1b. **exact product proof** (`sv/scheduler.py::run`, `sv/product_proof.py::prove`; [internal ref]'s
    equivalent-mutant proving, general first step, owner ruling 2026-09-29), only if
    `prove and proof is None and remaining() > 1` -- absint (stage 1) already tried and failed
    (its zone domain cannot carry `abs()`'s exact piecewise-linear shape across a case split, see
    [internal ref]'s design note). Scope: EVERY `_sv_verifier` draw must be `nondet_int()` / `nondet_bool()`
    with no argument, either a bare top-level statement `name = _sv_verifier.nondet_TYPE()` or, since [internal ref],
    a call anywhere an expression may stand INSIDE a def of the program (never in a loop/lambda/nested def,
    never a loose module-level expression -- see `product_proof.py::_lift`). A top-level draw is one z3 term
    bound before the run; a draw inside a def is a FRESH term at every EXECUTION of the call
    (`_symexec.py::_Runner._fresh_draw`, `run_exact(fresh_draws=)`): a function called twice draws twice. That
    is what makes the real tasks' shape provable: `def main(): x = nondet_int(); if not (lo <= x <= hi): return;
    assert <relation>` then `main()`. The property must be exactly `AssertionError`, and the WHOLE module
    must sit inside the audited allowlist `sv/exact_subset.py::audit` (every AST node at every depth,
    every name read/bound/called; int/bool values only; no loop, attribute, container, string,
    nested def/import, builtin or `__*` rebinding -- anything not listed is
    refused, so exactness is by construction, not by a list of attacks; PR #114 reviews 1-2). [internal ref] widened
    the allowlist by four exact cases, each pinned by `tests/test_sv_product_proof.py`: a comparison CHAIN over
    total operands (`0 <= w <= 6`; an operand that can raise or constrain the path is still refused, since
    Python short-circuits and the interpreter evaluates every link), a PROCEDURE (a def with no `return` value:
    early bare `return`, falling off the end; called only as a statement, so its `None` is dropped on the spot;
    a def returning a value on some paths only is refused), the annotations `int` / `bool` / `object` on
    parameters, returns and locals (builtin names no program binding can shadow; `x: int = v` is read as
    `x = v`), `int(x)`, and the guard `if __name__ == "__main__":` (read as true, its `else` dropped, as the
    child runs the program; a program that binds `__name__` is refused). Nothing is lifted from a def's vector:
    a model of a program whose draws sit in defs has no single draw vector (which draws a path consumes, in
    what order, depends on the path), so this stage returns `None` on SAT and the search stages own the FALSE.
    The builtin calls of the subset (`abs(x)`, `int(x)`, `min(a, b)`, `max(a, b)`) are read through their contract rules ([internal ref] step 2):
    `exact_subset.py::_builtin_arity` asks the rule to accept `solve/contracts/visitors.py::ProductVisitor`, which answers the audited arity
    (`ProductVisitor.EXACT_ARITY`) only while the contract is TESTED, so an untested or withdrawn contract takes the builtin out of the
    subset (the proof then says UNKNOWN); `tests/contracts/test_product_builtin_rules.py`.
    The Z3 timeout is `min(60 s, max(10 s, 15% of the remaining budget))` and never more than what remains
    (`product_proof.py::z3_timeout_ms`, passed `remaining()` by `scheduler.py::run`), so a nonlinear postcondition such as
    `-(a // -b) * b >= a` (about 12 s) is decided at the competition's budget and a 10 s triage run never spends its whole
    budget in one solver call. ASSUMPTION of the guard rewrite: the proof reads `if __name__ == "__main__":` as true and drops
    its `else` / `elif`, which is right only if the program runs as `__main__`; `sv/_child.py` runs every run (search, replay,
    canary) with `runpy.run_path(program, run_name="__main__")`, and `tests/test_sv_product_proof.py::test_the_proof_assumes_the_child_runs_the_program_as_main`
    pins it (a dead `else` that would fail an assertion must not stop a TRUE, and the same run must not reach it). None of the 102
    Tier A tasks has a `__name__` guard at all (they end with a bare `main()` call), so for them the assumption is vacuous.
    **[internal ref] grows the subset one construct per stacked PR, each pinned by `tests/test_sv_product_proof_subset.py`**
    (an EXHAUSTIVE native differential over a finite draw domain: the proof says TRUE exactly when no vector of the
    domain makes the real program raise `AssertionError`; one refusal test per shape the audit still rejects; one
    negative control that applies the mutation of the interpreter that would make the construct unsound and must flip
    the program to a wrong TRUE). PR 1, TUPLES: a TUPLE FUNCTION (every `return` is a tuple of the same arity >= 2, no
    starred element) whose value exists only as `a, b = f(...)` into exactly that many distinct names (the unpacking
    cannot raise) or dropped by a bare call statement (`sv/exact_subset.py::_Audit._classify`, `_unpack`, `_call`); a
    tuple is never stored, compared, passed, indexed or operated on, so the elementwise `ite` merge of two returned
    tuples (`sv/_symexec.py::_ite`) is the only place a tuple value is computed with. Named refusals (each by its own
    message in `audit`, listed in the test): a tuple used as a value, a swap `a, b = b, a`, an arity mismatch, a
    starred or nested or duplicate target, a def that returns a tuple on some paths only or tuples of two arities.
    PR 2, CONSTANT-RANGE `for`: `sv/exact_unroll.py::unroll_for_loops` (called by `product_proof.py::_normalize` before the
    audit) replaces `for i in range(<constants>)` by the statements it stands for: the arguments are literal ints or
    names bound exactly once in the whole module to a foldable int expression (`K = 4`), the target is assigned before
    EVERY copy of the body and left at the last value, no `else` / `break` / `continue` (a `return` is fine: the
    interpreter's early-return merge handles it), at most 64 trips and 20,000 AST nodes. It is an exact unrolling, not a
    bound, so `run_exact` still sees no loop. Every copy keeps the original lines (the sites are the same lines) and its
    draw calls are distinct nodes (a fresh term per iteration). A `for` it does not unroll stays a `for` and
    `exact_subset.py::_Audit._stmt` refuses it by name; a `while` that is not a counter loop (PR 4, below) is refused by name.
    PR 3, ALIAS-FREE LISTS of ints / bools: a list is built by a literal (at most 256 scalar elements), `[x] * K` (`K` a
    literal or a module constant folding to 0..256, `exact_unroll.py::fold_int`) or `[e for i in range(<constant>)]`, which
    `exact_unroll.py::_expand_comprehension` (called from `unroll_for_loops`, so before the audit) replaces by the literal of
    its elements with the trip value written into each copy of `e` (one generator, no `if`, no lambda / comprehension /
    walrus in `e`; a copy per trip, so every draw in `e` is a fresh term). `exact_subset.py::_Audit._type_lists` settles
    which names hold a list per scope BEFORE any statement is checked (assigned list expressions, `list[int]` /
    `list[bool]` parameter annotations, a parameter indexed or measured in its def, a parameter handed a list at some call:
    a fixpoint over the call sites); then a list name may only be indexed (`xs[i]`), measured (`len(xs)`), stored into
    (`xs[i] = v`, `xs[i] += v`: `_Audit._subscript_target`, only a list assigned in THIS scope that is not a parameter) or
    passed to a def whose parameter is a list (`_Audit._call`); it is never a value (no alias, return, comparison, operand,
    truth test, slice, nested list, method, iteration, `del`; a list parameter is never rebound, a list name is never
    assigned a scalar or unpacked into). So no two names ever hold one list, every list has a fixed length known when it is
    built, every cell is an int or a bool, and the interpreter is exact: `_symexec.py::_Runner.ev_Subscript` and
    `_Runner._store_list` are Python's rule (`-len <= i < len` or the run ends in `IndexError`, a path end and not the
    property; a negative index counts from the end; a symbolic index selects the cell it equals under that path
    condition; the store replaces the list, never mutates it), `_Exec.st` evaluates the index of `xs[i] += v` ONCE (evaluated
    twice, a draw in the index loaded one cell and stored into another), `len` is the list's length, two arms merge a list
    elementwise (`_ite`) and two arms that leave different lengths poison the name (`_merge_if`: inexact, never a wrong
    proof). `product_proof.py::_Normalize` reads `xs: list[int] = v` as `xs = v`. Measured on the 102 Tier A assertion-safety
    tasks: proven TRUE 39 -> 43 (`vllm_block_pool_get_new_blocks`, `vllm_free_kv_cache_block_queue_append_n` / `_popleft_n`,
    `vllm_has_repeating_pattern`, and `chromium_is_eligible_dispatch` through the comprehension), their `_buggy` twins as
    replayable models, 0 contradictions. Named refusals (each by its own message, listed in the test): a list used as a
    value, a store through a parameter or into another scope's list, a slice or tuple subscript, a nested list, `append` /
    `del` / iteration / `+=` on a list, a size that is not a constant in 0..256, a rebound constant, a `list[str]`
    annotation or a list annotation on a scalar parameter, a comprehension with a filter, two generators, a draw bound or a
    lambda, `len` or a subscript of anything that is not a list name, a comparison chain over a cell.
    PR 4, COUNTER `while`: `exact_unroll.py::_fuse_counter_loops` (run by `_Unroll.generic_visit` on every statement list,
    children first, so an inner loop is plain statements by then) replaces a `while` by the deep copies of its body when
    `_counter_trips` sees the whole trip count before the program runs: the statement JUST BEFORE the loop, in the same
    statement list, binds the counter to a constant (`i = 0`), the test is `i < | <= | > | >= <constant>` with the counter on
    the left, the LAST statement of the body steps the counter by a constant (`i += 1`, `i = i + 1`, `i = i - 2`:
    `_counter_step`) in the direction the test reads (`<` / `<=` with a positive step, `>` / `>=` with a negative one), nothing
    else in the body stores, deletes or declares the counter and nothing breaks or continues (`_body_leaves_the_counter`), no
    `else`, at most 64 trips and 20,000 nodes. The trips are `range(start, stop, step)` (`stop + 1` for `<=`, `stop - 1` for
    `>=`); the copies include the step, so the counter is right at the top of every copy and is left at the first value
    failing the test, as in Python; an empty trip count removes the loop and the init stays; a `return` in the body is a
    statement of its copy (the early-return merge handles it); a draw in the body is a fresh term per copy. The constants are
    `exact_unroll.py::module_constants`, which since PR 4 also folds a def's PARAMETER every call passes the same foldable int
    (`_param_constants`: the def is a top-level plain positional def bound once and called at least once; the parameter's
    name is bound exactly once in the whole module, every load is inside that def's body and the def never assigns it;
    every call is positional and its argument at that place folds to the one same value; to a fixpoint through other
    constant names). That is what boto3's `all_not_none(xs, n)` called as `all_not_none(xs, N)` needs: `n` is `N` at every
    call, so `while i < n` has four trips. Measured on the 102 Tier A assertion-safety tasks: proven TRUE 43 -> 44
    (`boto3_all_not_none`), its `_buggy` twin a replayable model, 0 contradictions. Named refusals (`exact_subset.py::_Audit._stmt`,
    "while loop that is not a counter loop", listed in the test): a bound that is a draw, a parameter passed two different
    constants or a draw or by keyword, a parameter of a def never called, the counter stored in the body or rebound by an inner
    `for`, a step under an `if`, not last, zero, against the test, a draw or a product (`r = r * 2`), `break` / `continue` /
    `else`, an init that is not the statement just before (also the target an unrolled `for` leaves), a draw as init, no init,
    `!=`, the bound on the left, a chain, a truth test, more than 64 trips. What stays refused, by name, for the Tier A tasks:
    a loop whose bound is a guarded draw (`while j < gen_count`, boto3_dynamodb_placeholder_merge, which also stores through
    a list parameter), a variant loop (`while buffered > 0 and steps <= MAX`, boto3_batch_writer), `break` in a nested
    counter loop (boto3_collection_limit, boto3_create_tags), a doubling step (vllm_next_power_of_2, with `&`).
    Nothing may be left unmodelled ANYWHERE, not only on the path to a checked site
    (`sv/_symexec.py::run_exact`, which now fails on ANY statement `_Runner.run` skipped).
    `_Runner.ev_BoolOp` has Python operand semantics (`x or 7` is `x` when truthy). Given that, it runs the FULL inter-procedural call graph from the
    program's own entry through `sv/_symexec.py`'s existing whole-harness interpreter (`abs()` as
    `ite`, exact Python floor division / `>>`, calls inlined up to `CALL_DEPTH` -- no slicing, no
    per-statement query) and asks Z3 whether every checked `assert`/`raise` is reachable by ANY input,
    over every path at once. UNSAT -> `Verdict("true", component="symbolic:product")`, stored as
    `proof` exactly like stage 1's (subject to the same canary). SAT -> Z3's model is replayed on the
    unmodified program in a fresh child (`sv/engine.py::replay`); only a replay that really raises
    returns immediately via `found(pv, fresh=True)` (the SAME address-independence k-of-k
    confirmation every other FALSE gets, never a shortcut). Anything outside scope, or a solver
    timeout, returns `None` and changes nothing.
2. **symbolic** (`sv/concolic.py::solve` via `sv/engine.py::verify`'s local `_concolic` import), only
   if the `concolic` stage is selected and `remaining() > 1`. Cost:
   `budget_seconds = min(remaining() * concolic_share, 60.0)` (`concolic_share` default `0.4`) for one
   helper process (`_concolic_worker.py`). Reads the program, classes, the `site_record`s stage 1 stored. Writes
   `Solved(candidates, sites)`: `sites` is already a per-harness-line obligation-shaped map
   (`reached` / `unsat` / `unreached` / `open`) -- a second, narrower obligation table that never joins
   the one from `obligation_census`. Every candidate is replayed immediately in a fresh child
   (`sv/engine.py::verify`'s `replay_solved`); only a replay that actually raises decides FALSE. If
   `solved.closed` (every site solved to "no modelled run fires it"), the REMAINING ledger is
   compressed via `capped(early_stop_seconds, ...)` -- a side effect on the CPU-second ledger shared by
   every later stage, not a return value.
3. **execution**:
   - `solve/absint/contracts.py::demands` (only if `prove and proof is None`) reads the source and
     produces `field_demands`, fed into the FIRST run only as `observe_fields`.
   - the first run (`sv/engine.py::run_once` via the local `run([], observe_fields=field_demands)`
     closure): one child, no forced draws. May decide FALSE immediately
     (component `deterministic` if it drew nothing, else `enum:small_first`). A `timeout` / `crashed`
     / memory-exhausted first run ENDS THE WHOLE VERIFY CALL right here, returning whatever stage 1's
     `proof` already was (or UNKNOWN) -- stages 2's candidates aside, none of stage 3's later branches,
     the portfolio or the enumeration ever run.
   - if the first run left `class_hints` and no `proof` yet: `_static_proof` runs AGAIN, this time with
     `class_hints=first.class_hints` -- absint's second, cheaper shot, keyed on one concrete observed
     execution.
   - if the first run drew NOTHING and raised no uncontrolled-nondeterminism flag
     (`not first.draws and not first.flags`): a SECOND run under `hashseed=1`. This branch always
     RETURNS -- FALSE if the second run also violates; UNKNOWN if it times out, crashes, or disagrees
     with the first; the already-found `proof` if the two runs agree and a proof exists; else
     `solve/determinism.py::determinism_census` decides TRUE (`deterministic`) or UNKNOWN. Stage 4
     (`prove_dynamic`), the portfolio and the enumeration NEVER run on this path.
   - 3b. otherwise (the program drew something), and `prove and proof is None and not first.flags and
     remaining() > 1`: `sv/fact_proof.py::prove_from_facts` ([internal ref] step 3, "Facts from producers" below):
     every installed fact producer (`sv/fact_producers.py`; none installed costs one entry-point lookup)
     is handed the open obligations, what it proposes is published into the store
     (`sv/eregion_bridge.py`), each proposal is certified ON ITS OWN by
     `sv/validate.py::check_candidates_individually`, and TRUE is decided only when every open
     obligation has a certified invariant that IS its own invariant. Reads the census, the store's
     candidate evidence; writes candidate and certified `Evidence`; cost: at most `PRODUCER_SHARE`
     (0.15) of what remains and 30 CPU seconds for the producers AND the certification together (one
     checker run, at most `checker_seconds`, is always allowed on top), at most 32 candidates. May
     decide TRUE only; a TRUE becomes the running `proof` with the same canary cap as stage 1's.
   - then (the program drew something, or raised a flag), and `prove and proof is None and
     not first.flags and remaining() > 1`: `sv/prove.py::prove_dynamic` (`docs/SV.md`'s own
     `proof:stationary` section, above) runs. Reads program, classes, `python`, `jobs`, `run_timeout`,
     `memory_bytes`. Its own internal probe loop is bounded by `remaining()` and `MAX_RUNS`. May decide
     FALSE (replayed inside its own loop against `run_once`) or TRUE; a TRUE becomes the running
     `proof` and gets the same canary cap as stage 1's.
4. **the portfolio** (`sv/stages.py::REGISTRY`, selected by `sv/stages.py::select`): one
   `sv/stages.py::Context` (program, classes, python, memory_bytes, `first`, log, `solved` from stage
   2, `records` from stage 1) is built once and handed to every stage, in registry order:
   - a **seed stage** (`guard_values`, `comparison`, `boundaries`, `composition`, and `avengers`,
     out of `DEFAULT`) calls `seed(ctx) -> {kind: [values]}`; the values are accumulated into `seeded`,
     keyed by the stage's `component` name. Never runs the program itself; never decides by itself --
     only ranks candidates for stage 5, and only counts as a "hint" if a later run built substantially
     from its value hits.
   - a **vector stage** (`concolic` again -- narrowed this time to the harness lines stage 2 left
     `open` -- and `hybrid`, [internal ref]) gets `budget = remaining() * share` (`concolic_share` again for
     `concolic`; a fixed `0.15` for `hybrid`), clamped by the ledger's cap if one is active, and
     proposes whole vectors (`propose(ctx, budget)`); each is replayed fresh if a `proof` already
     exists, else through the fork-server-backed `explore` -- "no TRUE depends on a fork" is a global
     rule enforced here, not a per-stage one. May decide FALSE only.
   The loop stops early once `remaining() <= 1` or the ledger's cap is reached; it then falls into
   stage 5 rather than returning UNKNOWN directly.
5. **enumeration** (inline in `sv/engine.py::verify`, `Candidates`/heap based): reads `seeded` (stage
   4) and the program's own literals (`harvest_constants`) to rank candidate values, seeds the heap
   from `first.draws`, then pops cheapest-cost batches and runs them (fresh unless `finite` and no
   active `proof` requires otherwise). A violation decides FALSE, attributed to whichever seed source
   contributed the deciding value. An emptied heap with every position finite and no truncation
   re-checks `determinism_census` for a `finite_tree` TRUE; otherwise the running `proof` wins if one
   survived, else UNKNOWN (several distinct reasons).

`sv/engine.py::verify`'s local `done()` wraps every terminal `Verdict` and, ONLY on `"true"`, calls
`sv/proof.py::finish`: it reuses the SAME `obligations` `ObligationCensus` computed before stage 1,
builds the candidate `Invariant` list (per-obligation minus coverage gaps, plus any
`candidate_invariants` a proof stage attached), and independently re-checks them
(`sv/validate.py::check_candidates`), writing `witness.yml` / `obligations.yml` and recording
`open_obligations` -- the open/discharged bookkeeping the design below wants live throughout the
run, computed today only once, after the verdict, and never allowed to change it (owner ruling: no
feature flags -- a TRUE from `finish` is never downgraded).

*Before step 3a* state crossed these stages ONLY through `verify`'s own local closures over `runs`,
`proof`, `cap`, `solved`, `records`, `seeded`, `first` and `servers` -- nonlocal captures, not an object
any stage's contract names. Now the ledger is `sv/ledger.py::Ledger` (`cap`, `servers`, `remaining`,
`capped` are its methods), and what components learn about the PROGRAM lives in the store:
absint's `records` -> `site_record`, `solved` -> `site_outcome`/`stage_ran`, the first run's
`class_hints` -> `class_hints`. What stays local is what has no meaning in a store: `runs`, `proof`
(the running TRUE a canary must survive), `seeded` (values handed to one enumeration), `first` (one
`Outcome`), the fork-server pool.

### The obligation table

One table per task, seeded from `solve/census.py::obligation_census` (already computed once, eagerly,
today -- Step 1 adds no new call). Each row is keyed by an `Obligation`'s identity
(`solve/obligations.py::Obligation.identity`: source sha, AST path, exception class) and carries:

- **status**: `open`, `discharged` (naming the proof: `proof:no-site`, `proof:interval` /
  `proof:shape`, `proof:stationary`, or the finalizer's independent check), or `violated` (a witness
  that replayed FALSE in a fresh child -- `sv/scheduler.py::run`'s `found()` is the only place that
  enforces this, in two tiers ([internal ref], [internal ref]): always a fresh-child replay of the candidate
  vector, and, additionally, whenever `solve/determinism.py::determinism_census` cannot certify the
  program address-independent, `sv/engine.py::_REPLAY_CONFIRM_K` (20) consecutive fresh replays that
  must ALL reproduce the same exception before the witness is trusted. A failure whose value -- not
  just its search path -- depends on an object's address (`id()`, the default `object.__hash__`,
  address text or set/dict order over objects) can raise on this machine and not on a validator's;
  no single replay, however fresh, tells the two cases apart, and no k makes it a certainty -- only
  the census's address-independence proof does. See `_REPLAY_CONFIRM_K`'s own docstring for why k=20.
- **what components learned**, so no component repeats another's work (built, 2026-10-05;
  `sv/obligation_table.py`): `site_record` (absint's per-site records, one generation per run; the
  latest supersedes), `site_outcome` / `stage_ran` (`reached` / `unreached` / `unsat` / `open` per harness
  line, per concolic stage), `class_hints` (what the first run observed), and next to them the Facts
  earlier slices already wrote (`candidate_invariant`, `input_sensitivity`, `loop_trip_count`), the
  static read of the task's module namespace (`module_constant`, `regex_pattern`, `call_target`; "Static facts" and
  "The `call_target` fact" below) and the
  fact producers' `invariant_candidate` / `certified_invariant` (below). Every kind, with its writers and
  readers, is `sv/contracts.py::KINDS`. NOT stored, because nothing would read them: the solved draw
  vectors (replayed on the spot) and `blockers` (absint's `residual` is already inside its
  `site_record`).

**Reads and writes.** `ObligationTable` is the one door: `rows`/`open_rows`/`status` read the rows;
`discharge` and `violate` are the ONLY writers of `certified=True` in `tharkun.sv` (a scan in
`tests/test_sv_contracts.py` fails on a third -- MG's review bar); `publish_site_records`,
`publish_solved`, `publish_class_hints`, `publish_module_namespace` write what a component learned and
`site_records`, `solved`, `class_hints` read it back; `since(consumer, kind)` hands a consumer only what was published after its
own last look, which is how "absint's second try" runs once per NEW `class_hints` item
(`sv/scheduler.py::run`) -- the cost rule in "What step 1 must not change" (a whole-program prover once
per piece of global information, never once per row).

The store-derived witness foundation adds two kinds. These publication/read pairs
are additive. The store witness shadow reads actual site records and deciding routes
after the frozen finalizer has completed its check and all legacy exports.

| kind | store | writer / reader | value |
|---|---|---|---|
| `site_safety` | Fact, never certified | `ObligationTable.publish_safety` / `safety` | `SiteKey`, canonical safety term text, `proposed` or `established` status, Evidence parents |
| `proof_route` | Evidence | `ObligationTable.publish_route` / `proof_routes` | deciding route, source digest, exception classes, covered site identities, site-record parent count, whole-program coverage and projection location |

An `established` safety claim names existing Evidence parents; consumers check
their coverage. It does not discharge an obligation. Route provenance uses the
store's `Derivation(parents)` graph, and each evidence key contains the schema
version and a digest of its coverage, so a changed publication cannot overwrite
an earlier one. A whole-program route carries the source location at which its
`True` projection belongs, including when the census has no physical rows.
Its `v2` value includes `site_record_parent_count`, derived from actual distinct
site-record Evidence parents, so an empty census does not imply no checked sites.

`sv/store_witness.py::project(table)` joins the latest record generation through
the original AST map's source digest, AST path and exception. It verifies the
recorded Name/Constant operands and operator against that exact node, and
intersects must-facts across every visit to the site. A memoized implication to
`False` must return `no` before any safety claim is established: contradictory
facts remain `store_only:facts_unsatisfiable`, and unresolved consistency proves
nothing. Unknown implication results are counted as `store_only:implication_unknown`
without retrying or increasing the resource limit. An implication establishes
the full contract safety term only on an exact `yes`; a failed admission is
`store_only:domain_not_established`. Exact canonical equality between a case's
domain and the conjunction of recorded type tests selects one case for a single
confirming implication. Domains containing additional value bounds cannot match
that shortcut. If no domain matches, the witness tries `case_safety` in the
template's fixed, disjoint case order. If no single case is
implied, it projects the disjunction of cases not proved inconsistent with the
facts only when every retained case renders; otherwise it records
`store_only:case_not_singled_out`. It never prefixes facts to the full safety
term or weakens admission. Implication results are memoized in the
task KB with their canonical facts/goal, source, property, classes and resource
policy. Safety remains uncertified and does not change obligation status.

The always-on shadow runs after the frozen check and legacy YAML writes. It
writes only `store-witness.yml` and `store_*` details; projection or export errors
leave the old verdict, `proof_checked` and witness bytes intact. It installs no
signal handler and works in worker threads. Z3 queries have deterministic
resource limits; a 256-site cap and a monotonic deadline checked between sites
bound the work cooperatively. The scheduler grants at most one second and 5% of
its remaining ledger. Remaining sites are `store_only:shadow_budget`.
Only the record's original-source and flow-sensitive `builtins` proof provides identity for
rendering; these lists are intersected across visits. Missing, malformed or
insufficient lists give `store_only:builtin_identity_unproved`. The producer
excludes any name bound anywhere in the original source and all five names when
that source contains reflection; the shadow does not repeat that analysis.
Metrics count safe site groups and sites
established and rendered separately. A direct `while` test uses a `loop_invariant` from
the facts common to all visits; other supported statements use location invariants.
Whole-program coverage comes from the actual final TRUE of absint, finite tree,
the no-site census, exact product proof, certified determinism, producer proof or
stationarity. Publication occurs once in `scheduler.done`, immediately before
`finish`. The five latter provers bind private route metadata to the exact Verdict
they return, its original source digest and filename, and the requested classes;
the metadata does not enter legacy `detail` or serialization. A component name
alone cannot supply these routes, and a provisional proof superseded by canary
FALSE leaves no route. Absint parents are its latest site-record Evidence.
Producer parents are the certified-invariant Evidence captured after this
invocation's successful checks of every current row. Stationary parents are the
call-target Evidence consumed by its draw plan, captured before probing; observed
trip-count Facts are never parents. The remaining proofs have no persisted
premise Evidence and publish root routes without invented parents. Missing or
stale source binding, parent capture or publication leaves the old verdict intact.

`solve/census.py::obligation_census`'s own `$coverage` rows (an exception class or construct the
census cannot cover at all) are NOT per-site obligations; they stay a separate `coverage_gaps` tuple
(`solve/obligations.py::ObligationCensus.complete`), exactly as today.

### Component contracts

*Not to be confused with the stdlib contracts* (`src/tharkun/solve/contracts/`, `docs/contracts/`): those say
what a Python library call may raise and return, for absint and the solver. The contracts below are the
SCHEDULER'S COMPONENT contracts (`src/tharkun/sv/contracts.py`): what each stage reads from and writes to the
run's store, what it costs and what it may decide.

Only a checked proof discharges an obligation; only a fresh-child replay of a violating run marks one
violated; everything else -- a seed value, a solved-but-unreplayed vector, an absint `residual` fact,
a `class_hints` sample -- is a HINT that ranks or targets a later component's work, never a decision
by itself. This is already the rule `sv/scheduler.py::run` enforces case by case (`found()`'s
fresh-replay and, when the program is not proven address-independent, k-of-k replay requirement --
see the obligation table's `violated` status above; `sv/proof.py::finish`'s independent re-check);
the contract makes it uniform across every component instead of re-derived per call site.

<!-- BEGIN CONTRACT TABLE (generated: tharkun.sv.contracts.render_table; tests/test_sv_contracts.py) -->
| component | file::symbol | reads | writes | cost | may decide |
|---|---|---|---|---|---|
| static facts (module namespace) | `solve/module_namespace.py::module_constants`; `solve/module_namespace.py::regex_globals`; `solve/call_targets.py::call_targets`; `sv/scheduler.py::publish_module_namespace` | source (and the local modules it imports) | `module_constant`, `regex_pattern`, `call_target` | parse and fold, executes nothing, no budget | nothing (hints and proofs of constancy only) |
| census (no-site) | `sv/prove.py::prove_static` | source, classes | nothing shared | instant | TRUE only |
| absint | `solve/absint/__init__.py::prove`; `sv/engine.py::_static_proof` | source, classes, `class_hints` (the second try), invariant locations | `site_record`, `candidate_invariant` | up to 10% of remaining, capped 60s | TRUE only |
| exact product proof | `sv/product_proof.py::prove` | source, classes | nothing shared | replay at most min(30s, remaining); Z3 timeout min(60s, max(10s, 15% of remaining)) | TRUE (Z3 UNSAT) or FALSE (a model replay-confirmed; never for draws inside defs) |
| symbolic (whole-harness) | `sv/concolic.py::solve` | program, classes, `site_record` | `site_outcome`, `stage_ran` | up to `concolic_share` of remaining, capped 60s CPU | proposes only (replay decides FALSE) |
| execution (first/second run) | `sv/engine.py::run_once` | program, forced draws (none) | `class_hints`, `input_sensitivity` | one child, `run_timeout` share | FALSE (replayed run) |
| determinism certificate | `solve/determinism.py::determinism_census` | source | nothing shared | parse + census, no execution | TRUE only (given two agreeing runs) |
| facts (fact producers) | `sv/fact_proof.py::prove_from_facts`; `sv/fact_producers.py::run` | `obligation` (open rows), `invariant_candidate` | `invariant_candidate` and `draw_vector` (the producers'), `certified_invariant` | 0.15 of remaining, capped 30s CPU, producers and certification together | TRUE only, and only when every current obligation has its own invariant checked in this invocation |
| stationary | `sv/prove.py::prove_dynamic` | program, classes, `python`, `jobs`, `call_target` | `loop_trip_count`; replays its probes via `run_once` | its own loop, bounded by remaining and `MAX_RUNS` | TRUE or FALSE |
| seed stages | `sv/guard_values.py::seed_values`; `sv/solver_seeds.py::seed_values`; `sv/avengers_seed.py::seed_values` | source (static), `site_record` (comparison/boundaries/composition) | seeded values (scheduler-local) | milliseconds to seconds, static | hint only |
| producer vectors | `sv/stages.py::_producer` | `draw_vector`, the first run (for the pre-check) | nothing shared | one replay per vector, at most `MAX_PRODUCER_VECTORS` | proposes only (replay decides FALSE) |
| symbolic (narrowed) / hybrid | `sv/stages.py::_concolic`; `sv/hybrid.py::search` | `obligation` (hybrid: the open rows), `site_record`, `site_outcome` (the lines the symbolic stage left open), the first run, and for hybrid the task code the program calls (`sv/reach.py`) | `site_outcome`, `stage_ran` | `stage.share` of remaining | proposes only |
| enumeration | `sv/engine.py::Candidates` | seeded values, program literals, the first run's draws | nothing shared | the remainder of the ledger | FALSE (replayed); TRUE via `finite_tree` |
| remainder (deepen) | `sv/scheduler.py::run`; `sv/hybrid.py::deepen` | `obligation` (the precise rows), what the earlier rounds kept (`Context.state`) | nothing shared | what enumeration left, in rounds of half the remaining; a cap still bounds it | proposes only (replay decides FALSE) |
| finalizer | `sv/scheduler.py::run`; `sv/proof.py::finish` | `obligation` census, the verdict's `candidate_invariants` and bound route premises (`certified_invariant`, `call_target`) | `proof_route` (actual final TRUE), `candidate_invariant`, `obligation` (discharged, via `ObligationTable.discharge`), `witness.yml` | up to `checker_seconds` (3.0), shared with the YAML writes | never downgrades; records what stays open |
| store witness shadow | `sv/store_witness.py::project`; `sv/store_witness.py::shadow` | source identity, `site_record`, `proof_route` | `site_safety`, `store-witness.yml`, `store_*` metrics | 256 sites, 10000 Z3 resource units/query, up to 1s or 5% of remaining ledger | nothing; runs after legacy exports |
<!-- END CONTRACT TABLE -->

Approximation by PURPOSE (owner, 2026-09-29): the proof components (census, absint, determinism,
stationary, facts, the finalizer) are always conservative and rely on certified facts or on sources they
check themselves; the search components (symbolic, narrowed concolic, hybrid, seeds, enumeration, the
fact producers) are lax and may assume anything, because soundness comes from the fresh-child replay, the
only FALSE source. The table above is `sv/contracts.py::COMPONENTS` rendered by `render_table`; the order
of its rows is the stage order of `sv/scheduler.py::run`. Eregion enters as a fact PRODUCER (the facts
row, below), not as a row of its own.

### Static facts: the module namespace (#1132b, [internal ref])

The first row of the table is not a stage that searches or proves; it is a read of the task's SOURCE that
every later component and every tool above Tharkun may ask the same question of, answered once.
`sv/scheduler.py::publish_module_namespace` runs right after the obligation table is built, calls
`solve/module_namespace.py::module_constants` and `regex_globals` for the task program, and publishes their
records into the store through `sv/obligation_table.py::ObligationTable.publish_module_namespace`:

- `module_constant`, one per module-level name of the entry module, key `<module>:<name>`. The record is
  `{kind, module, name, proven, value, value_kind, value_repr, origin, bindings, rebinds, provenance}`.
  `proven` is true exactly when NOTHING in the import graph rebinds the name (a second binding, a `global`
  store from a function, `pkg.mod.NAME = v` or `setattr`/`__dict__`/`globals()` writes through an import alias,
  a mutation of a mutable value through any alias -- a set literal and a value that holds a mutable member
  (`[[0]]`, `([0],)`) included, the latter never proven --, `exec`/`eval`, a dynamic import or a reflection route
  that makes the graph incomplete) and the graph is complete. The graph is INCOMPLETE, so nothing is proven, when
  any module of it calls `exec`/`eval`, refers to `exec`/`eval`/`compile`/`globals`/`vars`/`locals`/`setattr`/`delattr`
  other than as the callee of a call (an alias, an argument, a default), imports `inspect`/`builtins`/`gc`/`ctypes`/`code`,
  reaches a frame or function globals (`sys._getframe`, `f_globals`, `__globals__`), or hands on a `__dict__` or a
  `vars(x)` instead of subscripting it (`solve/module_namespace.py::_Project._scan_reflection`; tests in
  `tests/test_module_namespace.py`). It is a scan of names, not an alias analysis: a callable obtained by a computed
  name (`getattr(sys, 'x' + 'y')`) is not seen, and neither is a function that reads a name which shadows a
  builtin and runs before the name is bound. `rebinds` is the evidence when a name is not proven. A consumer that bakes a value
  into something it emits reads `proven` and nothing else: Eregion's synthesis bakes a module global into an
  inverter only when its record is proven, and reads the live module otherwise. `value` is JSON (`null` when the value is not statically known: a name can be proven stable without
  a known value).
- `regex_pattern`, one per compiled regex of the graph, key `<module>:<line>:<col>` of the defining
  `re.compile` (an inline `re.match('lit', x)` has the same record with no `bindings`). The record is
  `{kind: "regex_global", pattern, flags, compiled_flags, module, line, col, bindings, proven, provenance}`:
  the folded pattern source, the EXPLICIT flags argument (`re.compile(pattern, flags)` rebuilds the object), what
  the compiled object reports (implicit `re.UNICODE` and inline `(?i)` bits included), every `{module, name}`
  that binds it, and `proven` that each binding is a proven module constant. A regex that is not statically
  known is absent: never stale, never guessed.

Both kinds are hints about the source, uncertified Evidence, and the component DECIDES NOTHING (it costs no
budget: it parses and folds, executes nothing, and an analysis that fails publishes nothing and the run goes
on). `solve/module_namespace.py` is the one evaluator: Eregion's synthesis calls the same functions through
`tharkun.solve` (`module_constants`, `regex_globals`), never a copy; the string-contract rules of [internal ref] are the
intended in-scheduler reader of `regex_pattern`. Cost: the component reads at most
`sv/scheduler.py::STATIC_FACTS_MAX_MODULES` (16) modules of the import graph, and a graph over the cap is
INCOMPLETE, so nothing in it is proven (a 152-module package measured 5-19 s per call, which is not "no
budget"; tools above Tharkun call the functions with the default cap of 200 themselves); at most 500 records are
published. Measured on the 164 tasks of `eregion-sv/suites`: 0.7 s together.
`tests/test_sv_static_facts.py` pins the records, the crash path and the call from `scheduler.run`;
`tests/test_module_namespace.py` pins the analysis.

### The `call_target` fact: what a call site calls ([internal ref])

The third kind the static facts component publishes, and the first fact with two consumers. A call site's callee is
a NAME, and what a name denotes is the question `module_constants` already answers for the import graph, so the
producer (`solve/call_targets.py::call_targets`) is its second reader. `sv/scheduler.py::publish_module_namespace`
calls it after the namespace records (in its own `try`: a crashed producer keeps them) and
`sv/obligation_table.py::ObligationTable.publish_call_targets` writes one Evidence per call site of the ENTRY module,
kind `call_target`, key `<module>:<line>:<col>-<end_line>:<end_col>` (the calls of a chain `a.f().g()` start at one
position). It is registered in `sv/contracts.py::KINDS` (Evidence, uncertifiable, the component "static facts") and
read back through `ObligationTable.call_targets` / `obligation_table.py::call_targets_in`.

- **The record**: `{kind, module, line, col, end_line, end_col, scope, callee, target, name, def, dispatch, proven,
  reason, provenance}`. `target` is `user_class`, `user_function`, `user_method` (`Cls.m(...)` and zero-argument
  `super().m(...)`, resolved along the single-inheritance chain), `imported` (`name` is the dotted origin),
  `builtin` or `unknown` (a local binding hides the name, a method on a receiver whose type is not statically known,
  a decorated or async `def`/`class`, a name bound by assignment, a callee that is not a name or an attribute);
  `def` is the definition site (`{module, line, col}`, the module that DEFINES it for an imported name).
- **A receiver's type** ([internal ref]): `recv.m(...)` is a `user_method` record when `recv` is a variable whose ONE binding in its
  scope is `recv = Cls(...)` for a proven user class: the record then carries `receiver` (`{name, class, line}` of that
  binding) and `m` is resolved along the class chain. "One binding" is strict: in a function, no other binding of any kind
  (parameter, assignment, `for`/`with`/`except`/`import`/`def` target, `:=`, `del`) and no nested scope declaring it
  `nonlocal`; at module level `module_constants` shows nothing but that one -- possibly conditional (the main guard) --
  binding, no nested scope declares it `global`, no star import may bind it, and the module cannot write its own namespace.
  Every read of the variable then sees an object that statement made. It types the NAME; the consumers check what a name
  cannot say (`__new__`, a metaclass, an instance attribute hiding the method).
- **The lookup key** is `(scope, callee)`: `scope` is `<main>` for module-level code (the main block included) and the
  dotted names of the enclosing defs and classes otherwise (`Canvas.__init__`), `callee` is `ast.unparse` of the
  callee. Line numbers do not survive `sv/sites.py::build_harness` (which re-emits the program with markers) or the
  analysis slice, scopes and callee text do. Two sites of one scope that spell the callee alike and resolve
  differently (a comprehension variable named like a class) make the key ambiguous, and
  `call_targets.py::call_target_index` answers `unknown` for it: a consumer never picks one.
- **When `proven`**: ONLY when the name's binding is a proven `module_constant` (nothing in the import graph rebinds
  it, the graph is complete) and no local binding of the call's own scope chain (parameter, assignment, `for`/`with`/
  `except`/`import` target, comprehension variable, lambda parameter, enclosing function, `nonlocal`) hides it; a
  builtin is proven when nothing binds the name, the graph is complete and the module cannot write its own namespace
  (`globals()`, `exec`, `setattr`, `builtins`). An unproven record keeps the declared `target` and says why in
  `reason`. `proven` is about NAMES: the class a proven name denotes can still be mutated (`Cls.m = f`), so a consumer
  that executes the class checks the live object (below) and the stationary proof checks the program's own attribute
  stores.

**Consumer 1: the symbolic executor** (`sv/_symexec_objects.py::ObjectModel`, hooked into `_symexec.py::_Runner`;
`concolic.py::solve(call_targets=, proven_classes=)` carries the records to the worker's `solve_targets`). With the
records, `_Runner` models the program's own classes, exactly or with a named `Unsupported`; without them (`run_exact`,
so every proof) a class call stays unmodelled as before, and `product_proof.py` still refuses any program with a
`class`: class support is search-only, pinned by `tests/test_symexec_classes.py`.

- **What is resolved through the fact**: `Canvas(...)` is constructed only when the record of `(scope, "Canvas")` says
  `user_class` and is proven; `Cls.m(...)` and `super().m(...)` need a proven `user_method` that names the def the
  live class resolves to; any other use of a class name (`isinstance(x, Canvas)`, `Canvas.count`) needs a proven
  class name (the callee of a proven `user_class` record, or a class-origin `module_constant` that is proven,
  `ObligationTable.proven_classes`). An unproven or missing record is a refusal that says so.
- **What a class is**: the LIVE class. `module_globals` is the harness module as the worker loaded it, so
  `module_globals["Canvas"]` is the class after every import-time statement ran. `ClassTable` checks each `def` of
  its `__dict__` against the harness AST by first line (a patched, wrapped or generated method is refused), reads
  class attributes from the live dict (an import-time `Cls.x = 5` is what the program reads), and only the harness
  AST is ever inlined.
- **Objects** are interned `Ref` handles (`_symexec.py::Ref`) to cells of a per-path copy-on-write `Heap`: a write
  replaces the cell's attribute dict, so forking a path is a shallow dict copy. The heap is forked and merged with the
  environment wherever the executor forks one (`_Runner._merge_if`, `_exec_fn` for a function with early returns,
  `_guarded` for a conditional expression or a short circuit), attribute by attribute with the same typed `ite`; an
  attribute only one path stores merges with what the class says; a value that is a different object on two paths
  (`_symexec.py::_same_object`) is poisoned and refused when read. Instance attributes shadow class attributes and
  non-data descriptors, class attribute stores go to a per-class overlay along the single-inheritance chain, a
  missing attribute is `AttributeError` and ends the path, `super()` starts after the DEFINING class of the running
  method, `isinstance` is `issubclass` on the live classes.
- **Refused, each with its own message**: a metaclass, multiple inheritance, a base that is not a class of the
  program, `__slots__`/`__new__`/`__getattr__`/`__getattribute__`/`__setattr__`/`__delattr__`/`__del__`/
  `__init_subclass__`/`__set_name__`/`__class_getitem__`, a descriptor (`property` included), a generator or async
  method, a decorated class, dunder and `__private` (name-mangled) attributes, a method used as a value, a mutable
  container (or a symbolic list, or a path-dependent value) held in an attribute, any operator, conversion or format
  applied to an object (`a + b`, `a < b`, `f"{a}"`, `len(a)`, `a == b` when the class defines `__eq__`, `if a` when it
  defines `__bool__`), keyword arguments and defaults at a call, `super()` outside an instance method, an attribute
  store through a call expression, `any`/`all` over an expression that calls code (the executor evaluates every
  element, Python stops at the first).
- **Skipped statements** (`_Runner.run` skips a statement it cannot model): such a statement may have written an
  object or a class, so unless it provably did nothing (`ObjectModel._heap_neutral`: no store, no call but a pure
  builtin, no format, no operator on an object) the heap of that path is STALE and every later use of an object or a
  class is refused with the reason of the first skipped statement; plain values keep working. A skipped statement
  that calls a class or a method also makes the sites it could reach `unsupported` instead of `unreached`
  (`ObjectModel.build_reach`).

**Consumer 2: the stationary proof** (`solve/drawflow.py::_Planner._ctor`; `sv/prove.py::prove_dynamic` reads the
records from the store it is handed, `plan_draws(..., call_targets)`). A trip value passed to `C(...)` is followed
into `C.__init__` only when the record says `user_class` AND proven, `C` is a top-level class of the program bound
once, `_Planner._plain_class` holds (no metaclass, decorator or base, no `__new__`/`__getattr__`/`__getattribute__`/
`__setattr__`: any of those could read the value too) and the class body binds `__init__` exactly once by an
undecorated `def` that no attribute store anywhere in the program (`C.__init__ = f`) can replace. A trip value passed to
`recv.m(...)` ([internal ref], `_Planner._method`) is followed into `m` the same way when the record is a PROVEN `user_method` with a
`receiver`: `C` a top-level class bound once and plain, `m` bound exactly once in its body by an undecorated instance method,
and no attribute store anywhere in the program binding the name `m` (`recv.m = f`, `C.m = f`). A builtin callee is refused by
name (`divmod`, `len`, `ord`: the result varies with the value, the `divmod_gate` mutant); every other shape is the
"unresolved callee" refusal it was, now naming why.

**What the receiver typing does and does not buy** (measured, [internal ref]). It lets the planner follow a trip value into a method
(`tests/test_stationary_receivers.py`: a stationary method on a typed receiver is PROVED through `verify`; without the fact it
is not). It does NOT make `bm_richards_nondet` provable: the plan now succeeds (`richard = Richards(); richard.run(abs(...))`,
trip loop at `Richards.run`) and the probe then says `loop [...] is not stationary after 1 iteration(s)`. The token diff of
the probe's two snapshots (scratch copy of `_probe.py` writing the tokens of each snapshot) shows why: 1356 against 1644 tokens,
6 against 12 task dicts. `Task.__init__` does `self.link = taskWorkArea.taskList; taskWorkArea.taskList = self` and nothing
ever unlinks the previous iteration's six tasks, so the reachable state grows by six objects per iteration. The benchmark's
BEHAVIOUR repeats (the old tasks are held or waiting and the scheduler skips them) but a state-equality proof cannot see that:
it would need the old suffix proven inert (a read-set argument over every attribute and container access), which is not a
narrow extension of snapshot equality, so richards stays UNKNOWN on purpose (`test_a_method_that_links_every_iteration_...`
pins the shape: planned, refused, never TRUE). go (`divmod(pos, SIZE)`) is refused by name for the same reason as the mutant,
and its trip value `SIZE` is also used as a board size (`y * SIZE + x`), so no extension of the callee rule would help.

**Guard-bounded draws are not trip values** ([internal ref] (b), ruled, not built). `n = nondet_int(); if not (lo <= n <= hi): return`
bounds the draw but does not make it a trip count, and stationarity covers trip counts `c + abs(d)` only. Measured over the
102 Tier A programs (vllm, boto3, chromium-dashboard): 94 draw ints, 0 of the 94 let a draw reach a `range()`; the windows
are `1 << 10`, `1 << 20`, `1 << 30` (not enumerable) and the values are used in arithmetic. Accepting the guard would turn the
refusal "the draw is not taken abs()" into "the draw reaches no loop", nothing more; so the refusal stays and says what it
is (`drawflow.py::plan_draws`). A narrow sound shape exists for the day it is needed -- an adjacent `if <test>: return` whose
passing implies `lo <= v <= hi`, a window of at most `MAX_FINITE` values being a finite draw (the window plus ONE outside
representative, since the guard body uses no value), a larger one a trip draw of offset `max(lo, 0)` only when `v` reaches
`range()` -- but no program of the corpora would use it.

**Evidence.** `tests/test_call_targets.py` (the producer), `tests/test_symexec_classes.py` (every modelled construct
and every named refusal, each program run symbolically and natively), `tests/test_drawflow_ctor.py` (consumer 2),
`tests/test_sv_static_facts.py` (publication), `tests/test_stationary_receivers.py` (the receiver typing end to end). `tests/test_symexec_classes.py` also carries the negative control: the
heap tests are re-run with one fix undone each (attribute stores dropped, forks sharing a heap, no merge, class
stores ignored, `super()` skipping a class) and must fail. `scripts/class_differential.py` (generated class-heavy
programs, symbolic against native execution, with broken copies of the tree as a negative control) is DISABLED by the
owner's ruling of 2026-10-06 against differential fuzzing for now: nothing runs it.

**Not built (ruled).** Mutable containers in attributes (`self.items = []` and `.append`): the aliasing of a list
needs list identity on the heap, not a refusal's worth of work. Function calls are still inlined by name without the
fact (a proven `user_function` record is published but the legacy path does not require it). Calls on an object whose
class the heap cannot name (a merged object) stay refused.

#### Files and how they relate

- `src/tharkun/solve/call_targets.py` (stdlib only, SOLVING) imports `solve/module_namespace.py` (`_Project`, `_rel`:
  its walker) and is imported by `sv/scheduler.py::publish_module_namespace` (the producer), `solve/drawflow.py`
  (`call_target_index`) and `sv/_symexec_objects.py` (`call_target_index`).
- `src/tharkun/sv/obligation_table.py` writes (`ObligationTable.publish_call_targets`) and reads
  (`ObligationTable.call_targets`, `ObligationTable.proven_classes`, `call_targets_in`) the kind; `sv/contracts.py`
  registers it. `sv/scheduler.py::run`, `sv/stages.py::_concolic` and the hybrid stage's handoff
  (`sv/hybrid.py::search`, both through `sv/stages.py::Context.call_facts`) hand the records to `sv/concolic.py::solve`,
  which writes them into the worker's request; `sv/_concolic_worker.py::_symbolic_all` passes them to
  `sv/_symexec.py::solve_targets`.
- `src/tharkun/sv/_symexec_objects.py` imports `sv/_symexec.py` (`Ref`, `Unsupported`, `_Dead`, `_typed_ite`) and
  `solve/call_targets.py`; `_symexec.py::_Runner.__init__` imports it lazily, only when records were handed in (the
  same lazy shape as `_symexec_math.py`).
- `src/tharkun/sv/prove.py::prove_dynamic` reads the records with `obligation_table.py::call_targets_in` and passes
  them to `src/tharkun/solve/drawflow.py::plan_draws`.
- `scripts/class_differential.py` imports `tharkun` (the solver, the harness builder and the producer) and, for the
  negative control, runs itself against copies of `src/tharkun` with one line broken.

### Order and the CPU-second ledger, and the scheduler's name

**Step 1's scheduler is `sv/scheduler.py::run`.** It takes over the stage loop `sv/engine.py::verify`
runs inline today: stage 1 through stage 5, and the closures that exist only to serve that loop --
the ledger (`remaining`/`cpu_seconds`/`cap`/`capped`), `run`/`explore`/`get_server` (fresh vs. forked
execution), and `replay_solved`. It returns a plain decision: a replay-confirmed `Verdict` (the same
shape `found()` produces today) or `None` for "nothing decided, here is why." `sv/engine.py::verify`
keeps everything on either side of that call unchanged: the one-time setup before it (reading the
property, building `classes`/`specification`, computing `obligations` and `invariant_locations` --
see the obligation table above), and the one-time finalization after it (`done()`'s call into
`sv/proof.py::finish` on whichever `Verdict` came back, TRUE or not). Cheapest-deciding first (owner
ruling 2026-09-24) stays the order inside `sv/scheduler.py::run`: static proofs once, covering every
class over the whole program, before anything runs a single line.

Step 1 kept the stages in their sequence and budget shares inside `sv/scheduler.py::run` and changed no
decision. **The ledger is now one object, `sv/ledger.py::Ledger`** (built, step 3a): `spent`/`remaining`
(CPU seconds AND wall-clock, whichever is tighter, over `os.times()` plus live, unreaped fork-server CPU --
`add_server`), `capped`/`at_cap`/`allowance`/`share` (a canary or early-stop compression once a proof
exists or a site set closes; a cap only ever moves earlier; what one run, or one portfolio stage, may
still use). Reads: what has been spent; writes: what a stage's completion compresses for everyone after
it. `tests/test_sv_ledger.py` pins every number it computes.

**The goal and the one stop condition ([internal ref]).** `sv/ledger.py::Goal` says what the run is for (today: ONE
confirmed violation; `wanted > 1` is ruled, not built -- a caller that wants several inputs drives
`verify` once per input), and `Ledger.stopped(reserve)` is the ONE stop condition every stage reads: the
goal is met (`scheduler.py::found` records it, only after the same replay confirmation every FALSE
goes through), or no more than `reserve` CPU seconds remain, or a cap (canary / early stop) has been
reached. A stage that runs dry -- a finite enumeration, a fuzzer that plateaued -- is NOT a reason to
stop: before [internal ref] enumeration draining its (finite, ~100-value) `str` candidates ended the run as
"unbounded draw domain" with 90% of the budget unspent. `Stage.deepen(ctx, budget, round)` is how a stage
keeps searching an unbounded domain: a generator, so each vector is replayed (fresh) the moment it
appears and the stage's helper process stops with the run. `hybrid.py::deepen` streams the Atheris
worker's finds from a corpus that persists across rounds (`Context.state`), with the fuzzer's CPU on
the ledger while it runs (`hybrid.py::_LiveCpu`) and, unlike the bounded `search`, without the
assert-fallback's "line reached" hint (and so without the worker's line tracer: a raised property
exception is the hit). It runs only for a program whose first run drew something and raised no
environment flag (`Outcome.flags`: a file effect, argv, a clock): the worker runs the program FOR REAL,
outside the engine's I/O model (the first campaign of [internal ref] caught a program that truncates a file it does not
own, with no draw; the bounded `hybrid.py::search` has the same exposure -- on sv_2027 a 8 s budget lets it
start the worker on that program and truncate the file -- noted on [internal ref], not changed here). A round in which no stage proposed anything and no CPU was
spent ends the run short of the budget ("no stage left to try"); a proof's canary and the early stop
are caps, so they still bound the remainder. `tests/test_sv_goal.py` pins all of this.

Step 2 (NOT built; ruled; kept only if it gains, per the ticket) is a loop over open rows of the obligation table:
pick the cheapest OPEN obligation and the cheapest component that has not already produced a `blocker`
for it, run one slice, write back (`discharged` / `violated` / still `open` with a new hint), repeat
until the ledger or the obligations run out. This is a genuine semantic expansion, not a refactor: see
"what step 1 must not change" below for why it needs its own soundness argument before it replaces
anything.

### The backward slot ([internal ref])

*Since the Eregion bridge ("Facts from producers", below) Eregion is a fact PRODUCER with a route to TRUE
(a nomination the checker certifies); the vector stage described here, a candidate vector a backward
inversion proposes for replay, is BUILT as the `producer` stage ("Draw vectors from a producer", below,
[internal ref]).*

Every component above is FORWARD: it either bounds every execution statically (census, absint,
stationary) or runs/enumerates from the program's start. Eregion (`sv-vision-and-terminology`: forward
first, inversion per violation point) is the first BACKWARD component: given one open obligation's
violation predicate (already a `python_expression`, `solve/obligations.py::Invariant.value`, negated
the way `sv/hybrid.py::_violation` already does for its own coverage target), it works backward from
the site toward an input, or concludes the site is unreachable under its own model.

The slot's contract is the SAME shape as `concolic` / `hybrid`'s row above: reads one obligation's
predicate plus whatever `records`/`Solved.sites` already narrowed away; writes a candidate vector (a
HINT: the engine's own fresh replay still decides FALSE) or an "unreachable for this obligation" fact
(a hint toward `discharged`, not a discharge by itself -- only a checked proof discharges, same rule as
every other component). Cost: a share of the ledger, sized the same way `hybrid`'s is today
(`stage.share` of what remains when its turn comes). This section is the slot only; [internal ref] fills it.

**The slot needs only Step 1, not Step 2.** Once `sv/scheduler.py::run` exists and holds stage 4 (the
portfolio, `sv/stages.py::REGISTRY`), a backward component is one more `Stage` entry there -- a
`propose(ctx, budget)` vector stage exactly like `concolic` and `hybrid`, added to `REGISTRY` and (once
measured) to `DEFAULT`. It needs no per-obligation loop: today's portfolio already hands every stage
the SAME whole-program `Context`, and a stage is free to iterate the obligation table itself (the way
`sv/hybrid.py::search` already iterates `obligation_census(...).obligations` internally, per-site,
inside its own `budget` share) without the scheduler doing per-obligation dispatch on its behalf. Bd
[internal ref]'s Done-when (5) -- a test-driven stub backward component proving the slot -- is
therefore a Step 1-adjacent deliverable: it can land as soon as `sv/scheduler.py::run` exists, ahead
of any decision on Step 2's per-obligation interleaving.

### Facts from producers ([internal ref] step 3: the Eregion bridge)

*Built ([internal ref], [internal ref]); ON whenever a producer is installed.* Owner rulings that
bind it (2026-09-29): the bridge is an EXTENSION of the fact store (a producer is one more writer of
`KnowledgeBase` `Evidence`, not a separate translator); a fact counts only after an independent per-candidate
check certifies it; Tharkun never depends on Eregion; Eregion's engine packages import nothing from Tharkun.

**Flow.** `sv/scheduler.py::run` stage 3b calls `sv/fact_proof.py::prove_from_facts`:

1. `sv/fact_producers.py::discover` lists the producers installed in the interpreter's environment by the
   entry-point group `tharkun.sv.fact_producers` (Tharkun names none; `eregion-codegen` registers
   `eregion` -> `eregion_codegen.tharkun_producer:produce`). `run` starts each in a CHILD process
   (`python -m tharkun.sv.fact_producers NAME`): one JSON request on stdin (the task path, the classes, the
   open obligations in the census's own shape), one JSON answer on stdout, killed at its budget, one thread
   per numeric library (measured: one inversion cost 9.7 s of CPU with every BLAS thread spinning and 0.5 s
   with one; the ledger counts CPU), memory-capped like every run. A crash, timeout or malformed answer
   yields no facts, and a producer's own text (its last stderr line, a rejected fact's contents) is never
   logged: the verifier's stdout is parsed line by line for a verdict token, so the log carries the exit
   status or the exception class only (MG review, #110).
2. `sv/eregion_bridge.py::publish_fact` turns each plain-dict fact into `Evidence` in the run's
   `KnowledgeBase` (`invariant_candidate`, keyed by location plus a digest of the predicate; a second
   producer's identical claim adds a `Derivation`, not an item; `draw_vector`, "Draw vectors from a
   producer" below). A fact that does not parse is dropped alone; `Invariant`'s own constructor validates the
   location and the predicate grammar, and the draw-vector door its own rules.
3. `prove_from_facts` reads the candidates back OUT of the store and checks each with
   `sv/validate.py::check_candidates_individually`; a proved one is published as `certified_invariant`
   (`eregion_bridge.py::publish_certified`, called from nowhere else).
4. TRUE needs: a complete census, no adapter-gap rows, at least one open obligation of the property's
   classes, the first run without nondeterminism flags, CPython 3.14 (the checker's requirement), and a
   successful independent check in this invocation of EVERY current obligation's own invariant. Old
   certified evidence and discharged row statuses do not replace a current check. A wholly discharged
   table still returns without asking producers; otherwise existing stored nominations for discharged
   rows may be rechecked alongside the open rows. A partial cover decides nothing and
   the later stages run as if this stage did not exist.

**What a producer adds.** The checker proves an invariant from the source alone, so today a producer's
contribution is the NOMINATION (which obligations to try; its own engine flags travel as `evidence`, never
as a verdict) and the checker's Z3 run is the proof. A fact that carries information the checker cannot
derive (a loop invariant, a class-state invariant) needs a verifier for its own shape first.

**Eregion's producer (`eregion_codegen/tharkun_producer.py`).** Scope: the open `ZeroDivisionError`
obligations of a task whose `if __name__ == "__main__":` block reaches the division through straight-line
assignments after `nondet_int()`/`nondet_bool()` draws. It lifts the block up to the division into a plain
function whose parameters are the draws, runs the ordinary pipeline on it
(`eregion_codegen/objects/wrapper.py::synthesize_door`, the call `ObjectWrapper` also makes) and asks the
generated inverter for the draws that make the divisor 0. A draw the engine found and forward-verified means
the obligation is violable and is proposed back as a `draw_vector` (at most 4 per obligation) for Tharkun
to replay; no draw found means the obligation is nominated. That
evidence is WEAK by design: the engine reports `incomplete_recovery:no_absence_proof` for these shapes
(only a bijective path or the sin/cos family rung ever proves an absence, [internal ref]), and for `x*x + c` it finds
no draw even when one exists, so the producer is a lax proposer (owner, 2026-09-29: search components are lax,
soundness comes from the independent check) and a wrong nomination is refused by the checker.

**Source checker spellings** (`sv/validate.py::_Checker`) added for this: `abs()` on int/float/bool, and
`import _sv_verifier [as m]` with `m.nondet_int()/nondet_bool()/nondet_float()` (the track's own spelling;
a rebinding, a call of the module, or any other attribute is refused). `tests/test_sv_validate_abs_module.py`
pairs each positive case with a violating twin.

**Ruled, not built (and why):**

- *Periodic-family intersection* (`sv/validate.py::FamilyIntersectionCandidate`; owner: "intersections are done
  by the store"). No producer can emit it honestly today: the engine's `AnswerSet` carries `complete`/`domain`
  flags, not the raw bases and period; the failing pre-state of a method (for `normalize`: all fields zero,
  because a norm is a sum of squares) is not derivable by the engine; its proof is over the REALS and the
  float-soundness lemma a verifier needs is absint's `solve/absint/lemmas.py::TRIG_PAIR`, which already decides
  `bm_float_nondet`'s arithmetic (`proof:interval`). A bridge shape without a producer would be dead code.
- *Violation hints* (eregion-codegen#127's `_VIOLATION_HINTS`): a search seed with no consumer in this path.
- *Objects through the checker.* `_Checker` models a class method modularly (slice 2b) but not a constructor
  or method CALL in the main block, nor a method `return`, nor a field whose sort is not an annotated
  `__init__` parameter copied verbatim, so no program that instantiates a class can be certified yet,
  whatever a producer says.

### What step 1 must not change

Step 1 (`sv/scheduler.py::run`, holding today's order and budget logic, with `engine.verify` calling it) must
reproduce every one of these exactly, or the full-budget campaign's 0 lost / 0 new wrong / same-scores
bar fails:

- **The two special-cased returns inside "execution" stay special-cased.** A first run that times out,
  crashes, or exhausts memory ends the WHOLE call immediately (stages 4 and 5 never run). A first run
  with zero draws and no flags takes the "confirm-and-certify" branch and RETURNS before
  `prove_dynamic`, the portfolio or the enumeration -- it does not fall into a generic per-obligation
  loop that might try them anyway.
- **Two static-proof calls, not one.** `_static_proof` runs again mid-"execution" only when the first
  run produced `class_hints`, using them as absint's second try. A scheduler must keep this exact
  trigger, not turn it into "re-run every static component whenever any new hint appears" (that would
  cost more absint calls than today, per site).
  - **A per-obligation loop must not call a whole-program static prover once per obligation.** Census
  and absint each cover every class over the whole program in ONE call; Step 2's loop must dispatch
  them once per NEW piece of global information (a fresh `class_hints` sample), exactly as today, not
  once per open row -- the current single extra call would otherwise multiply into one per obligation.
- **`concolic_share` is spent twice, from two different `remaining()` snapshots.** Stage 2 spends up
  to `concolic_share` of the budget AT THAT TIME; the portfolio's `concolic` sub-stage spends
  `concolic_share` again, of whatever is left AFTER stage 2 (and stage 3) already spent theirs. A
  ledger that computed every stage's share once, up front, from the ORIGINAL budget, would hand out
  more or less CPU than today and is a Step 2+ question (its own campaign), not Step 1.
- **"No TRUE depends on a fork" is a whole-run rule, not a per-component one.** The instant any
  component's evidence becomes a `proof`, every later run anywhere in the pipeline -- portfolio,
  enumeration, and (once built) the backward slot -- switches from the fork server to a fresh
  interpreter. A per-obligation scheduler must check the SAME global flag, not a per-obligation one:
  today's `Verdict` is whole-program (TRUE/FALSE/UNKNOWN for the whole `classes` tuple at once), and
  nothing here yet gives meaning to "obligation A is proven, obligation B is still being explored by a
  forked run" as a per-row state independent of the global one.
- **Per-obligation TRUE composition is new, not preserved.** Today, TRUE always comes from ONE
  component's whole-program claim (no-site, absint, stationary, determinism, finite-tree) --
  `verify` never concludes TRUE because every individual obligation happened to be separately
  discharged. Step 2's per-obligation loop raises the question of whether "every open obligation
  discharged, nothing else left to search" is itself a sound route to TRUE; if the design answers yes,
  that is a NEW proof rule needing the same soundness write-up as any other `proof:*` component before
  it is used for a verdict -- Step 1 must not introduce it by accident just by restructuring the loop.

### Draw vectors from a producer ([internal ref], 0987)

*Built; ON whenever a producer is installed (no flag: with none installed the store holds no vector and the stage
yields nothing).* Owner rulings that bind it (2026-10-05): the backward drive goes through the `solve_` layer
ABOVE Eregion's `invert_`, and `solve_` takes the task's draws IN PLACE: **the program is never rewritten for
replay or verdict**, and the draw values reach it only through the child's `_sv_verifier` shadow (record/replay
vectors, `sv/_child.py`). The `fp#24` guard hoist was dropped because MG and MGE found wrong answers through
`global`/`del`, `NameError` vs `UnboundLocalError`, `__main__`, `sys.modules`, frames and `__globals__`; this design
has no such surface, because the only thing that ever runs the task is the unmodified file in the child.

**The seam is the one the bridge already has.** A producer answers the facts stage's request with
`{"kind": "draw_vector", "vector": [{"kind", "value"}, ...], "site": {...}, "evidence": {...}}`
(`eregion_codegen/objects/tharkun_facts.py::draw_vector_fact`). `sv/eregion_bridge.py::publish_draw_vector`
is the door: the entries are `bool`, `int` (at most `MAX_INT_BITS` bits), `float` (a `float.hex` string) or `str`
(inside `string.printable`, at most `MAX_STR_CHARS`), nothing else (`rindex`, `rfloat` and `argv` entries, whose
value the child clamps or models specially, are refused), at most `MAX_VECTOR_ENTRIES` entries; anything else
drops that one fact. A vector is `Evidence` of kind `draw_vector`, keyed by a digest of its entries, never certified
(`sv/contracts.py::KINDS`). No `tharkun` import on the producer's side and no `eregion` import on this one.

**The `producer` stage** (`sv/stages.py::_producer`, in `DEFAULT` before the concolic stages) READS the vectors
back out of the store and yields them as `Candidate`s (component `producer:<name>`); the scheduler's portfolio
loop replays each one on the unmodified program, in a fresh child (or a fork-server child, then confirmed fresh),
through `found()`: the ONE FALSE path, with its fresh replay, its address-independence census or k-of-k
replays, and the `_replayed` check that the run consumed exactly the vector. At most `MAX_PRODUCER_VECTORS` (64)
are replayed; a vector that is the first run again (every draw its kind's default) is skipped.

**Out-of-band rejection.** A draw outside its domain is refused in the child, `Outcome.invalid_draw` is set BEFORE
the call raises, and `Outcome.violated` refuses every run that has it: a task that catches the refusal
(`try`/`except`, `contextlib.suppress`, a context manager whose `__exit__` returns true, a `finally` that returns)
and then fails an assertion is not a violation, exactly as the grading harness marks the same run invalid. Two
more layers keep such a vector from costing a run: the door (above) drops what no run could accept, and
`sv/engine.py::vector_admissible` pre-checks a vector against the domain of every `nondet_float_range(a, b)` the
FIRST run called (the child logs them beside the draws, `Outcome.ranges`, not in them, so a witness never carries
one). The call at a position is fixed by the draws before it, so those bounds hold for a vector that repeats the
first run up to that position; where the first run never reached the call, the child's own refusal is what holds.

**What a producer's answer can and cannot do.** It can cost at most 64 replays and one producer child run (the
facts stage's own budget). It cannot make a TRUE: `prove_from_facts` reads `invariant_candidate`/`certified_invariant`
only, a vector is never certified and discharges nothing, and an engine's EMPTY answer is still only the weak
nomination the independent checker decides on. It cannot make a FALSE without a native replay of the unmodified task.
Eregion's lift is therefore a LAX PROPOSER: whatever it gets wrong (a `global`, a `del`, a frame, `__main__`) costs a
replay that does not violate.

**Attack cases** (`tests/test_sv_draw_vectors.py`, each also run with the draw reached by alias, `getattr`,
`sys.modules`, `globals()`, `__import__`, `from ... import ... as`, int bounds and reversed bounds): swallowed
refusal through seven shapes of swallower, with the vector unconditional (pre-check drops it) and behind a branch the
first run did not take (the child refuses it); a non-printable `str`; a vector longer than the run draws, with the
wrong kinds, malformed, oversized or flooding; a draw reached only through `globals()`; a lying producer on a safe
program; a vector that does not discharge or certify.

**Reach** (measured 2026-10-05, CPython 3.14.4, tharkun `e4ea637` and eregion-codegen `2c9bb0d` against this change;
`reach_dv.py` per obligation row, then the stage alone per pair with `stages=("producer",)`, no symbolic stage, no
program literals, no unbounded tail):

| | Solve-only (65 pairs) | Real-world (80 pairs) |
|---|---|---|
| census rows / of them `ZeroDivisionError` | 230 / 22 | 12,416 / 112 |
| rows the producer lifts (before = after) | 8 | 0 |
| rows the engine answers | 8 | 0 |
| draw vectors proposed (before: none, the fact kind did not exist) | 2 | 0 |
| vectors that violate on native replay | 2 | 0 |
| pairs the stage decides FALSE alone, by `producer:eregion` | 1 (`so_eregion_odd_bug`) | 0 |
| pairs gained in the default portfolio | 0 (the symbolic stage and the first run decide those first) | 0 |

The producer's scope is unchanged (one `ZeroDivisionError` obligation per lift, straight-line code, `int`/`bool` draws,
and a ONE-draw lift is the only shape the engine answers), so the honest movement is the pipeline, not the score.
Verdicts against the base tree with the producer and Atheris installed (direct runs, 60 s Solve-only and 120 s
Real-world per pair): no pair lost that is not hybrid-flaky on both trees (`so_base3_bug` 2 of 18 FALSE on base against
8 of 18 on this change, `so_count_loop_bug` 13 against 14), no new wrong answer, no TRUE on an expected-false pair.
The hoist's attack programs (77 programs, 308 pairs: MG's `fp24_probe.py` and the `nondet_lift` soundness tests),
run through `tharkun.sv` against the base tree with the real producer and with a hostile producer replaying 796
out-of-range, wrong-kind and random vectors: no verdict changes, and every FALSE witness confirmed by the grading
harness's own replay. **Ruled, not built:** lifting other exception classes or a `nondet_float_range` draw (the lift
refuses them), obligations the census marks as gaps (a producer is not asked about them, as before), a multi-draw
lift (the engine returns no answer for two int parameters today).

### Files and how they relate (measured imports)

The comprehensive measured import graph across all 40 modules in `tharkun.sv` is detailed above under
[Files and how they relate](#files-and-how-they-relate).

- `sv/engine.py` -- `verify`, `_static_proof`, `run_once`, `Outcome`, `Verdict`, `Candidates`. Every
  `tharkun.*` import is LAZY (inside a function), so each prover is used as soon as it exists and a
  missing/crashing one is skipped: `solve/determinism.py::determinism_census` (`_certify_determinism`);
  `sv/prove.py::prove_static` and `solve/absint/__init__.py::prove` (`_static_proof`);
  `solve/census.py::obligation_census`, `sv/spec.py::parse_property`,
  `solve/obligations.py::source_points` (top of `verify`); `sv/proof.py::finish` (`done`);
  `sv/stages.py` as `_stages`, `sv/concolic.py` as `_concolic`,
  `solve/absint/contracts.py::demands`, `sv/prove.py::prove_dynamic`, `sv/forkserver.py::ForkServer`
  (all inside `verify` itself, at the point each stage needs them).
- `sv/stages.py` -- `REGISTRY`, `Context`, `Stage`, `select`. Its own stage functions import lazily
  too: `_concolic` imports `sv/concolic.py`; `_hybrid` imports `sv/hybrid.py::search`; `_solver_seeds`
  imports `sv/solver_seeds.py::seed_values`; `_guard_values` imports `sv/guard_values.py::seed_values`;
  `_avengers` imports `sv/avengers_seed.py::seed_values`; `_producer` imports `sv/engine.py::vector_admissible`
  and `sv/eregion_bridge.py::DRAW_VECTOR` (the vectors a producer proposed, read from the run's store).
- `sv/prove.py` -- `prove_static`, `prove`, `prove_dynamic`, `run_probe`. Imports
  `solve/determinism.py::determinism_census` and `solve/drawflow.py::plan_draws` at module level; its
  functions additionally import `sv/engine.py` (`Verdict`, `read_source`, `_cap_memory`) and
  `solve/absint/__init__.py::prove` lazily.
- `sv/product_proof.py` -- `prove`. Imports `sv/exact_subset.py` (the audit), `sv/exact_unroll.py::unroll_for_loops`
  (called by `_normalize`), `sv/_symexec.py::run_exact`, `make_var`,
  `model_value` and `sv/sites.py::LIFTED` at module level, `sv/engine.py` and `sv/sites.py::violation_sites`
  lazily; called by `sv/scheduler.py::run` (stage 1b). `sv/exact_subset.py` imports `sv/exact_unroll.py::fold_int` and
  `::module_constants` (the size of `[x] * K`) and the stdlib; `sv/exact_unroll.py` imports the stdlib only.
- `sv/sites.py` -- `build_harness`, `harness_refusal`, `violation_sites`. Imports `solve/call_targets.py::call_target_index`
  lazily (inside `_entry`); imported by `sv/concolic.py`, `sv/product_proof.py` and the tests.
- `sv/concolic.py` -- `solve`, `Solved`. Imports `sv/sites.py::build_harness` and `::harness_refusal` at module
  level (the refusal is what its log line names, [internal ref]; the `call_target` records it passes to both come from
  `sv/scheduler.py::run`); starts
  `_concolic_worker.py` as a subprocess (not a Python import).
- `sv/reach.py` -- `reached`, `called`, `dictionary`, ... Imports `sv/engine.py::read_source` lazily and
  `solve/environment.py::TaskEnvironment` lazily; called by `sv/hybrid.py`.
- `sv/hybrid.py` -- `search`, `deepen`. Imports `sv/reach.py` and `tharkun.hybrid.backend::AtherisBackend`,
  `tharkun.hybrid.fdp_codec::atheris_available`, `tharkun.hybrid.fuzzer::should_assist` and
  `sv/stages.py::Candidate` at module level; imports `solve/census.py::obligation_census` lazily
  (guarded by `ImportError`, so this stage degrades if that module is ever absent); starts
  `_hybrid_worker.py` as a subprocess through `AtherisBackend`, in the stage's own temporary directory
  (`cwd=`), with `-artifact_prefix` inside it (`artifact_dir=`): libFuzzer writes its `crash-*`, `leak-*`,
  `timeout-*`, `oom-*` and `slow-unit-*` files to the prefix, which defaults to the subprocess's CURRENT
  DIRECTORY -- before [internal ref] they landed in whatever directory `tharkun.sv` was launched from. The backend
  always passes a prefix (its own private temporary directory when the caller names none, removed at
  `stop()`); `tests/hybrid/test_backend_artifacts.py` holds that. The worker runs the program FOR REAL, so it
  installs the engine child's environment model (`solve/environment.py::install_files`, imported by bare name with the
  same self-cleaning `sys.path` insert as `_child.py`; the declared files arrive as `Context.files` through
  `request.json`'s `files`), and its stdin is `/dev/null`: a program that opens, truncates or writes a file it does not
  own gets the same refusal there as in a replay (before this, a verify with a few seconds of budget truncated such a
  file on the host). `tests/test_sv_hybrid_sandbox.py` holds that.
- `sv/guard_values.py` -- imports `solve/arith.py::free_names`/`solve_int_guard`.
- `sv/solver_seeds.py` -- imports `sv/guard_values.py::_Scope`.
- `sv/proof.py` -- `finish`. Imports `solve/obligations.py` (`Invariant`, `Obligation`,
  `source_points`, `witness`), `sv/validate.py` (`Validation`, `check_candidates`),
  `sv/witness.py::dumps`; imports `sv/engine.py::read_source_bytes` lazily.
- `sv/validate.py` -- `check_candidates`, `Validation`. Imports `solve/census.py::obligation_census`,
  `solve/obligations.py` (`Invariant`, `source_points`), `solve/predicates.py` (`BUILTINS`,
  `check_predicate`), `sv/spec.py::parse_property`, `sv/witness.py::loads`.
- `sv/witness.py` -- imports `solve/obligations.py::Invariant`.
- `solve/census.py` -- `site_census`, `obligation_census`. Imports `solve/pyresolve.py`
  (`Resolver`, `Top`); `obligation_census` additionally imports `solve/obligations.py`
  (`Invariant`, `Obligation`, `ObligationCensus`, `source_points`) and `solve/predicates.py`
  (`InvalidPredicate`, `parse_predicate`), both lazily.
- `solve/obligations.py` -- `Invariant`, `Obligation`, `ObligationCensus`, `source_points`, `witness`.
  Imports `solve/predicates.py::parse_predicate`.
- `solve/absint/__init__.py` -- `prove`, `AbsintResult` (its own module set is `docs/ABSINT.md`'s
  "Files and how they relate", not repeated here): the static prover `sv/engine.py::_static_proof`
  and `sv/prove.py::prove` call.

`sv/scheduler.py::run` (Step 1's named module, above) sits ABOVE nothing new and BELOW
`sv/engine.py::verify`: it holds the order/budget logic `verify` holds inline today and calls the same
provers/stages through the same lazy imports this list already names, so this list does not grow by a
new dependency layer -- only by one new caller (`sv/scheduler.py`) of everything already here, and one
new callee of it (`sv/engine.py::verify`, which imports `sv/scheduler.py::run` the same lazy way it
imports `sv/prove.py`, `sv/concolic.py` and `sv/stages.py` today).
