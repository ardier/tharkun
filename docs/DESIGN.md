# Tharkun — Design

> *Module paths in this document predate 0.2.0. The current layout, the layer rule and the move map (every old path is a permanent shim) are in [LAYERS.md](LAYERS.md).*

This document explains *why* tharkun is shaped the way it is. It is not an API reference;
`CLAUDE.md` already lists the modules and `docs/HOW_IT_WORKS.md` walks through the
single-guard techniques with examples. What follows is the reasoning a new maintainer
needs before changing anything: what each layer is for, what forced its shape, and where
it is known to fall short.

Every quantitative claim below is sourced from this repository — a test docstring, a
commit message, or `CLAUDE.md`. Where a number is a measurement on one target it says so;
none of them are projections.

> **Note on the older docs in this directory.** `HOW_IT_WORKS.md`, `THARKUN_DESIGN.md`
> and `THARKUN_SKILL_DESIGN.md` predate the driver, the fact store and the hybrid fuzzer.
> They remain accurate about the single-guard technique chain and are still worth reading
> for that. They do not describe the loop, and `docs/README.md` still links them by an
> absolute path from a former checkout location. Start here.

---

## 1. The problem, and the shape of the answer

Tharkun answers one question: *what inputs make this branch go the other way?* Everything
else in the repository is a consequence of taking that question seriously on real Python.

Three things about real Python force the design.

**No single technique covers the space.** Guards are heterogeneous — an integer
comparison, a regex, a `self.state == "connected"`, a `flags & Flags.NEGOTIATE_SEAL`, a
length check on a slice of a slice. A static solver cracks a large fraction and abstains
on the rest. An SMT engine decides a modeled fragment and abstains on everything outside
it — measured, **84.6% of touchstone's abstentions are unmodeled constructs**: containers,
string methods, f-strings, library calls (`redqueen.py` module docstring). So the system
is a *portfolio*, not a solver.

**One flip is almost never enough.** Flip the first blocking guard and execution proceeds
far enough to reveal a second blocker the first trace could never have shown you, because
execution did not get that far. Deep targets need a fixpoint. **The loop is the product;
a one-shot solve is its inner primitive** (`driver.py` module docstring).

**Solvers are allowed to be wrong.** This is the load-bearing decision. Every technique in
the portfolio returns a *proposal*, and `driver._flip_verifies` re-executes the program and
asks the tracer whether the guard's target side actually ran. Nothing is committed on a
solver's say-so. That single rule is what lets the portfolio contain unsound heuristics
(live operand harvesting, minleaf's mutations, a type-coerced variant of
someone else's answer) without ever emitting a wrong result: a bad proposal costs one
execution and is discarded. `tests/test_cross_file_drive.py` states the corresponding
test discipline — every assertion is checked against a real trace, never against the
driver's own claim, because *"a reachability engine that grades its own homework is worth
nothing."*

### Three scales, three entry points

| Scale | Entry point | Judge |
|---|---|---|
| one guard | `branch_solver.solve_branch(file, line, branch)` | the chain's own verification post-step |
| one line | `driver.drive_to_line(file, fn, line, seed)` | the tracer, every round |
| one fuzz target | `hybrid.HybridFuzzer(spec).run()` | Atheris's own coverage counters |

`differential.differential_kill` is the driver's sibling: the same touchstone-first,
search-behind structure with a divergence goal (`orig(*args) != mut(*args)`) instead of a
line target. It has no internal callers by design; it is consumed externally.

---

## 2. The driver loop

```
   seed inputs
       |
       v
   +-------------------------------------------------------+
   |  trace: run func(**inputs) under TracePolicy           |
   |         watching source_path (+ target_path if given)  |
   +-------------------------------------------------------+
       |
       |-- target_line executed?  ----------> REACHED (observed, not claimed)
       |
       v
   pick the blocking guard
       - lexically enclosing frames whose target side did not run
       - else: guards that RAN but did not enter their body, anywhere
         in the file (_call_chain_blockers) — the blocker may be in a CALLER
       |
       v
   build the precondition for this guard
       path prefix + banked facts + guards currently holding
       + caller-pinned constants + field bounds
       |
       v
   +-------------------------------------------------------+
   |  PORTFOLIO — each proposal verified by re-execution    |
   |  touchstone -> snapshot -> static                      |  cheap tier
   |  minleaf                                               |  live state
   |  hypothesis-over-facts -> hypothesis                   |  search
   |  backtrack / reach-search                              |  last resort
   +-------------------------------------------------------+
       |
       v
   merge the verified flip into inputs, bank the guard as a fact, re-trace
```

`drive_to_line` defaults to `max_rounds=16` and `max_backtracks=4`. It returns one of four
statuses: `REACHED` (the tracer saw the line), `PLATEAU` (no flippable frontier remained,
with diagnostics), `BUDGET`, `NO_TARGET`.

Two structural details are easy to miss and both were bugs first.

**Line numbers are per-file.** When `target_path` names a file below the entry function,
the executed-set lookup must fall back to *empty*, never to the entry file's lines —
otherwise an unrelated line with the same number reads as "the target ran", which is a
false `REACHED`, the worst possible failure mode for a reachability engine (commit
`1c73d44`). `tests/test_cross_file_drive.py::test_line_numbers_are_not_confused_across_files`
pins it.

**A cross-file target has no lexical blocker until the callee runs.** While the callee has
not been entered, its guards say only "we never got there"; the guard actually stopping us
is in the caller, which `enclosing_guards` (purely lexical) cannot see. Those rounds go
straight to the dynamic caller-side search, which scores by `_progress` — the count of
executed lines at or before the target. Crude, but monotone in the right direction:
entering a callee that leads to the target strictly increases it, so opening the next
layer is recognisable with no call-graph analysis.

---

## 3. The portfolio, and why it is ordered this way

The ordering principle is *information per unit of time*, with one exception at the top
that is structural rather than economic.

`tests/test_technique_audit.py` exists because the portfolio was audited and found
wanting. Instrumenting a real ntlm_auth run gave:

```
invoked    touchstone 1158   snapshot 1158   static 929   hypothesis 344   redqueen 30
committed  snapshot   309    static    174   everything else 0
```

That audit found two real defects (cross-file runs were reading live objects only from the
entry file, so callee blockers had no state for the techniques that need it; and RedQueen
could not invert `int.from_bytes` at all). The test file now holds one toy per technique,
each shaped so that technique should be the one to crack it. Its stated purpose is worth
quoting because it is the right standard for the whole portfolio: the point is not that a
particular member wins — the portfolio is free to find a cheaper route — but that each is
*capable* of winning when it is the only one that can.

### touchstone — the only member that can prove a branch dead

`touchstone_adapter.touchstone_solve_branch` lifts the guard into a probe and hands it to
touchstone (Z3, corroborated by cvc5, Rocq-checked core). It returns one of three verdicts:

- **satisfiable** — a *certified* assignment. This is why touchstone is first in the cheap
  tier and not merely a fallback: a certified answer is preferred to a guessed one, and it
  costs no more than the guesses do.
- **infeasible** — a *proof* that no assignment takes the branch. No other member has this
  capability; the rest can only fail to find an input. A proved-dead guard side lets the
  driver **prune** it from the worklist rather than proposing inputs for an unreachable
  path until the budget runs out.
- **unknown** — outside the modeled subset, or the solver abstained. The chain continues,
  nothing is lost.

Its limit is the modeled fragment, and that limit is large. Everything below exists to
cover it.

`driver._flip_touchstone` does one thing the adapter does not: it decides the guard under
each *likely joint typing* rather than one guessed type, using the distribution built by
`type_hints.gather_hints` / `candidate_typings`. A fuzzer field read as `int` on one seed
and `str` on another has a genuinely ambiguous type, and satisfiability tolerates a wrong
guess because execution verifies downstream. **Infeasibility is only reported when every
candidate typing agrees the branch is dead**, so ambiguity can never license a false prune.

### snapshot — the static solver, re-run against observed state

`condition_solver.solve_with_snapshot` takes the f_locals captured at the branch line and
re-solves the guard with the observed values substituted in. On the measured ntlm_auth run
this is the single highest-yielding member (309 committed flips). Its limit is that it
still reasons about source: it can only exploit the snapshot to fill in what it could not
infer statically.

### static — `condition_solver.solve_condition`

The 4,282-line workhorse: ten type-inference rules, fifteen value-generation patterns
(comparison values, range intersection, string synthesis, `len` constraints, regex
inversion, walrus, negated `isinstance`, bitmask flag guards, …). It is a pure function of
`(source, line, polarity)`, which is why it is memoizable alongside touchstone.

Bitmask support is a good illustration of what "cheap tier" work actually looks like. The
comparison extractor walked `ast.Compare` nodes, so a bare `if flags & MASK:` produced
nothing at all — and on ntlm_auth **three of five reachable frontiers were exactly that
shape, every one returning `strategy: none`** (`tests/test_bitmask_guards.py`). The guard
is trivially invertible once the mask is a number; the real work is that protocol code
*names* its masks (`Flags.EXTENDED_SESSIONSECURITY`), usually in a sibling module, so
`_resolve_int_constant` has to chase the name through the import that brought it in. The
polarity subtlety is what makes it correct: asking for the `else` edge negates the guard
to `not (flags & MASK)`, and in an if/elif chain on flags, reaching the second arm requires
*clearing* the first arm's bit. A solver that can only set bits can never leave the first
arm (commit `15305a4`).

### Live operand harvesting — input-to-state correspondence

`src/tharkun/driver.py::_harvest_operands` calls `src/tharkun/redqueen.py::harvest`
during reach-search trials and `src/tharkun/driver.py::_bank_operands` banks the
best inverted proposals. There is no standalone RedQueen driver tier or keyword
flag ([internal ref]); live harvesting remains part of reach-search. The harvester reads
comparisons the static tier cannot read, because
the demanded value does not appear in the source: a runtime-computed constant, or the
input reaching the compare *through* an unmodeled transform (`x.decode()`,
`compute_expected()`, `int.from_bytes(p[:2], "big")`).

RedQueen (NDSS'19) sidesteps modeling entirely. It never asks what a transform does; it
observes both operands of each comparison on a concrete run and asks the much weaker
question *does an input value already flow into one operand?* If it does, the comparison
itself reveals what the input must become — the other operand — and that value is copied
onto the input.

Tharkun has an edge over the paper here, and it is worth understanding why. RedQueen must
*guess* how the input was encoded (its colorization step), because it works on raw bytes
and cannot see field structure. In tharkun the driver controls typed parameters directly,
and for the fuzzer the encoder (`hybrid.fdp_codec`) is known exactly and re-verified
against the real FDP. Where RedQueen guesses a placement, tharkun knows it.

It returns a *ranked list*, unlike the cheap-tier flippers which return one candidate
each: one guard can expose several correspondences and only execution can confirm which
transform was truly identity-like. What it cannot do is recover a placement through a
non-identity-preserving transform that is not one of its shapes (identity / offset /
substring) — a checksum, a hash, base64. There, RedQueen's own colorization would also be
guessing, so it returns nothing rather than a bogus proposal. Mock the checksum instead.

### minleaf — minimal mutation of live state

`minleaf/` is the answer to the unmodeled constructs that dominate the SMT lane's
abstentions. It never translates the guard; it *evaluates* it on the real objects the
tracer captured and searches (BFS, cost = number of distinct leaf paths touched) for the
smallest set of leaf values that flips it. An unmodeled container operation, string method
or library call is a non-issue, because the real Python runs.

Its minimality objective is not aesthetic — it serves the caller. A patch structurally
close to the observed state re-encodes into a byte seed with the fewest perturbed fields,
which is exactly what a fuzzer wants.

The driver tries **both bytes policies** (`per_byte`, `atomic`). They expose different leaf
sets — `per_byte` can flip a single byte of a buffer, `atomic` can only replace the whole
thing — so a guard one cannot express the other often can. `driver._apply_patch`
materialises the patch onto deep-copied roots, rebuilding immutable sequences where
`cur[i] = v` would raise; before that existed, a perfectly good `SetOp(('data', 0), 0 -> 120)`
on a `bytes` parameter was silently dropped and the driver reported "no applicable input".

Minleaf needs concrete live state. When a parameter has never been observed with a
meaningful value, it has nothing to mutate.

### hypothesis over the accumulated facts

`driver._flip_hypothesis_facts` is not "run Hypothesis on the guard". It hands Hypothesis
the **conjunction** — every path fact banked so far plus the guard being flipped — with a
strategy per parameter built from that parameter's *live runtime type*. See §4 for why the
conjunction matters.

Two details are load-bearing:

- **Types come from the live value, not from static inference.** Static inference reads
  `p[0] == 0x7F` as "p is subscriptable" and offers a list; the running program says `p` is
  `bytes`, and only `bytes` can satisfy it.
- **The alphabet is drawn from the facts' own literals.** Uniform `binary()` has to stumble
  onto `0x7F` at index 0 — about a 1-in-256 draw, measured at **4 reached runs out of 6**
  cold — while the byte the comparison demands is written in the comparison. With the
  literals in the pool: **8 of 8, deterministic** (commit `1c73d44`). This is RedQueen's
  idea applied to a search's input distribution instead of to a single copy.

Names the clauses mention but that are not being searched over keep their observed values,
so the conjunction still means what it meant.

### hypothesis over the guard

`driver._flip_hypothesis` is the plain property search, 500 examples, last of the
per-guard tier.

CrossHair is deliberately **off** the default heavy tier (`_HEAVY_FLIPPERS =
[_flip_hypothesis]`). In the driver's portfolio it is redundant — touchstone already covers
sound SMT and the drive loop is itself concolic — and it is the slowest option (subprocess
plus timeout). `_flip_crosshair` remains available for a caller that wants it back.

### reach-search — the last resort, and the one that does not need the dataflow inverted

`driver._reach_by_search` is the answer to a guard stated in a *callee local*:
`block_length != 8 and pad`, where `block_length` is `len(data[i:i+8])` and `data` is a
slice of a caller argument. Every flipper above works on the entry function's parameters,
so none of them can say "make `block_length` 3"; mapping the local back to the parameter
needs `call_mapper` (see §7).

But a certified model still tells us *what* the guard wants, and that is enough to aim a
search. Touchstone's assignment supplies the interesting values, the live types supply the
shapes, and **execution decides**. The predicate is not the guard — which cannot even be
evaluated out at the entry point — but "did the target line run", which is the only thing
we actually care about and is never a heuristic. Length is searched as well as value,
because for this class of guard the lever is usually how *long* a parameter is rather than
what is in it.

### The verification gate, and the one thing observation does not catch

`_flip_verifies` re-runs the program with the candidate merged in and requires the
blocker's target side to execute. That subsumes almost every sanity check you would
otherwise write by hand — including "does this candidate break an outer guard we already
satisfied?", because breaking the prefix means the target side cannot execute. An earlier
version of `_ok` vetoed any candidate touching a variable an outer guard reads; it was both
redundant and *wrong* (an outer `len(c) >= 3` marked `c` locked, so a perfectly good patch
to `c[0]` was thrown away).

The one failure observation cannot catch is a candidate that flips the guard by changing a
parameter's **type**. `p[0] == 0x7F` is answered with `p = []` by the subscript rule, which
flips nothing and destroys the `bytes` every other guard was solved against. `_ok` rejects
type-changing candidates so the portfolio escalates to minleaf — with an explicit exemption
for guards that genuinely ask about the type (`isinstance` / `type` / `issubclass`), which
are the one case where changing it is the right answer.

`_variants` complements this: every candidate is offered alongside its *type-preserving
twin*, ordered twin-first. A solver answering `len(p) > 4` with the str `'xxxxx'` for a
parameter the program runs as `bytes` gets the `bytes` version tried first. The twin is a
proposal like any other — it only wins if execution confirms it.

---

## 4. The fact set, and why it must iterate

Guards are solved one at a time. That means an answer to an early guard can make a later
one **unsatisfiable**, and a naive accumulate-forever design blames the deeper guard for
the earlier guard's choice.

The canonical example — the one the cross-file fixture is built around — is **type drift**:

```python
def deep(payload):
    if len(payload) > 4:        # happy with str OR bytes
        if payload[0] == 0x7F:  # only ever true for bytes
            return "INNER-DEEP"
```

`len(payload) > 4` answered with `'xxxxx'` satisfies the first guard and kills the second,
because `'x'[0] == 0x7F` compares a str to an int. The loop then plateaus on the *second*
guard, having been defeated by the *first* guard's answer.

Tharkun closes this from both sides.

**Backtracking (the loop's side).** `driver` keeps a `trail` of committed flips. When a
blocker cannot be flipped by anything and there is a trail to unwind, the blocker's own
requirement is banked into `established` as a lasting fact, the last flip is undone, and
the earlier guard is re-solved *under the stricter fact set*. Bounded at four backtracks;
the cycle check (`seen`) still applies, so if the earlier guard can only ever answer the
same way the run plateaus rather than spinning.

**The conjunction (the solver's side).** `_flip_hypothesis_facts` hands the whole
accumulated fact set plus the current goal to Hypothesis at once. There is no ordering to
conflict with, because nothing is solved in sequence: one assignment either satisfies all
of it or does not.

### What goes into a precondition

For each blocker the driver assembles `requires` from five sources, all restricted to
clauses whose free variables the blocker's probe can actually bind (a clause naming any
other variable would be unbound for the solver, and a dropped clause shares no variable
with the blocker so it cannot affect the answer — it stays in `established` and applies
again the moment a blocker does mention that variable):

| source | function | why |
|---|---|---|
| lexically enclosing guards, in their desired polarity | `_prefix_requires` | without it, independent per-guard solving oscillates: fixing an inner guard by breaking an outer one |
| every guard deliberately satisfied so far | `_established_requires` | re-deriving facts from the current trace is not enough — once a flip moves execution elsewhere, a constraint already paid for stops being observable and would be silently dropped |
| guards currently *holding* on the executed path | `_satisfied_path_requires` | when the target sits below a call, the lexical prefix does not preserve guards in the **callers** that got us there |
| what the callers pin | `context_facts` / `harness_map.call_site_constants` | see §5 |
| a field's actual value range | `_bounds_requires` | so the answer is *encodable* back into a fixed-width fuzzer field |

Booleans are pinned as an integer comparison — `driver._pin` writes a truthy `pad` as
`pad != 0` and a falsy one as `pad == 0` — because touchstone models a bool as an int and
cannot translate `pad == True`, which would waste the fact on an "unknown" instead of
spending it on a proof.

---

## 5. The shared knowledge base — `facts.py`

This is the newest module and the one whose idea generalises furthest. Read it first.

### The observation that forced it

Every strategy derives something the others could use, and until `facts.py` none of them
could see each other. Touchstone produces certified assignments and dead-branch proofs;
reach-search harvests the operand a comparison demanded; minleaf identifies blocking atoms and
their leaf dependencies; `length_requirements` reads a width off a guard;
`call_site_constants` reports what the callers pin. Each was wired to whatever happened to
need it.

Then the cost of *not* sharing was measured. On ntlm_auth's harness, touchstone was invoked
**1158 times across 39 distinct questions** — a **96.6% repeat rate**, 29.7x per question,
**11.4 of its 11.8 seconds spent re-deciding what it had already decided** (commit
`ff78d3d`; the numbers appear in `facts.py`'s docstring and in `tests/test_facts.py`).

That is not a tuning problem. Touchstone is a *decision procedure*: given the same guard
and the same precondition it necessarily returns the same verdict. It can only produce new
information at a new frontier, or when the facts constraining it change.

### The three-way split

The core idea is a taxonomy of strategies by **what a repeat can tell us**, because that is
what determines what may be memoized, what must be re-run, and what gets a retry budget.

```
DETERMINISTIC   touchstone, static
                fully determined by (guard identity, precondition, typing).
                Re-asking is pure waste — served from the store.

STATEFUL        snapshot, minleaf, redqueen
                read live program state, which differs every round.
                A repeat is a genuinely DIFFERENT question — never memoized.

STOCHASTIC      hypothesis, hypothesis-facts, reach-search
                randomized search. A repeat may succeed where the last failed —
                recorded for provenance, never served as an answer.
```

An unknown strategy defaults to **not** memoizable. The asymmetry is deliberate: a wrong
"yes" silently serves a stale answer, a wrong "no" only costs time.

Two identity decisions matter as much as the taxonomy:

- **The precondition is part of the question, not context.** The same guard under a
  stronger precondition is a *different* question, and is exactly the case where re-asking
  is worthwhile — a guard satisfiable in isolation can be infeasible once the callers'
  pinned values are folded in. The assumed typing is part of it too, for the same reason.
- **Decisions are namespaced by strategy.** The two deciders initially shared a key space
  and the static solver was served a `TouchstoneResult`. They return different *shapes* of
  answer, so a shared key space is a collision that fails loudly if you are lucky and
  silently if you are not.

Measured with one store shared across the whole engine: **touchstone ran 40 / served 1118
(97% avoided); static ran 33 / served 896 (96% avoided)**, with ntlm_auth coverage unchanged
at 155 → 184 and des.py at 100 (commit `ff78d3d`).

### The epoch, and why the exhaustion gate is not permanent

Memoizing individual queries stops the *solver* re-running, but the **round** still pays for
the stateful and search tiers. On a minimal toy — four nested guards plus one sha256 guard
nothing can crack, so a frontier persists forever — eight rounds cost **26.7s and rounds 3-7
produced no coverage at all**. Skipping them took the same run to **7.7s with an identical
line set**; on ntlm_auth, **390.7s → 166.2s** with coverage unchanged (commit `126882f`).

So `FactStore` carries an `epoch` that increments whenever something is *genuinely learned*:
a new deterministic decision, a new `Fact`, or an explicit `bump()`. `mark_exhausted(scope)`
records the epoch at which a frontier resisted the whole portfolio; `is_exhausted(scope)` is
true only while the epoch has not moved.

**Soundness rests entirely on this being a gate rather than a verdict.** Any new decision,
fact, or seed re-opens every written-off scope at once. Correspondingly, being *served* a
remembered verdict deliberately does **not** bump the epoch — it taught us nothing — and
neither does re-adding a duplicate fact. `tests/test_facts.py::TestEpochGatedExhaustion`
pins each of those cases individually.

One hole found while building it, and it is instructive: **the corpus changes from outside
the assist**. The whole design runs Atheris in the background writing seeds into the same
directory, and those never touched the epoch — so the assist would have gone permanently
quiet in exactly the mode it exists for, while Atheris kept feeding it fresh material.
`AssistEngine.assist_once` now fingerprints the corpus and bumps when it moves
(`tests/hybrid/test_crossmodule.py::TestExternalCorpusReopensFrontiers`).

### The stochastic retry budget

Epoch-gating alone is unsound for a randomized search, and the reason is subtle: **a failed
search records nothing, so it never moves the epoch**, and the scope would be shut out
permanently on the strength of one unlucky draw. A deterministic portfolio that comes up
empty on unchanged state will come up empty again — writing it off is free. A search will
not necessarily.

So `DriveResult.used_stochastic` reports whether a randomized tier was part of what failed,
and `mark_exhausted(scope, retriable=True, budget=2)` spends a retry budget before writing
the scope off. Cost on the toy: **7.7s → 12.1s over 8 rounds**, against 26.7s before any
skipping, with an identical line set (commit `e004e8c`). Worth it for soundness.

### Constraints, not values

`Fact` records evidence — a type, a width, an observed operand. `Constraint` records
something that must **hold**, kept as an *expression* rather than a value, and it exists
because storing the value loses the knowledge.

The motivating failure (commit `21d4da9`): a search observed
`int.from_bytes(token[:2], "big") == 13987` and banked the number `13987`. Aiming the next
search with that number still failed, because the bare number asks each consumer to
rediscover the encoding by chance — while RedQueen, which can invert the `int.from_bytes`
shape directly, never received the expression at all. **The fact was routed to the one
consumer that could not use it, past the one that could.**

So a `Constraint` carries both representations, because consumers genuinely differ:
touchstone and the static solver want a precondition *string*, while RedQueen, minleaf and
a predicate compiler want to walk the AST. Text is the stored form — it is what would
serialize if the store ever crossed a process boundary for a scheduler — and the node is
derived and memoized per distinct text, so the two cannot drift.

Three projections exist, one per consumer shape: `requires_str()` for the SMT lane,
`predicate_src()` for a search, `goals_for(subject)` for RedQueen. Each filters to clauses
whose free names the consumer can actually bind — a **correctness contract, not an
optimisation**, since an unbound name makes a probe ill-formed. A dropped clause stays in
the store and applies again as soon as a consumer mentions that variable. `Constraint.names`
excludes builtins for exactly this reason: counting `int` in
`int.from_bytes(...) == 13987` made the clause look unbindable and silently vanished it
from every projection.

This is increment 1 — the type and its projections, with tests, before anything is rewired
onto them. Nothing consumes it yet; the driver still assembles `requires` from its own
`established` list (§4).

### Yield accounting

`FactStore` also records per-strategy `hits` / `misses` / `yields` (answers actually
committed) and exposes them via `report()`. This is what a power schedule would allocate
on: *a strategy invoked 1158 times for 0 committed answers has earned less energy than one
invoked 30 times for 30.* The numbers are collected. The scheduler that spends them is not
built — see §7.

---

## 6. Slicing, and what it costs

### Why it exists

A whole production function is usually outside touchstone's modeled subset. A single guard
lifted out of it, with its free variables as parameters, lands inside it far more often.
`touchstone_adapter.synthesize_probe` builds

```python
def probe(x: int, y: str):
    if <the guard>:
        return 1
    return 0
```

so that deciding `result == 0` decides guard satisfiability, and polarity is just which
postcondition you ask for (`result == 0` for "then", `result == 1` for "else").

`slicing_bench.slicing_ablation()` measures what that reduction buys, over 400 guards of
tharkun's own source: handing the engine a whole enclosing function decides **5.8%**;
handing it the guard lifted into a probe decides **64.8%** — an **11x lift**, **242 guards
rescued against 6 lost**, **median 53x fewer AST nodes** (`CLAUDE.md`). The honest framing
is in the module's own docstring: this measures *modeled-subset membership*, not a
like-for-like solve rate, and it is raw engine reach *before* the adapter's soundness
gates. The usable rate after those gates is **~32%**, all satisfiable, at **~6ms median**.

The conclusion that matters for anyone extending this: the front-end is not plumbing in
front of a solver, it is what makes the solver applicable at all.

### The two soundness gates on `infeasible`

A `satisfiable` verdict is cheap to be wrong about — execution re-verifies it downstream.
An `infeasible` verdict *prunes*, so being wrong about it silently kills live code. Two
gates therefore apply, and both downgrade `PROVED` to `unknown`:

1. **Every probe parameter must be typed.** Touchstone defaults an unannotated parameter to
   `int`, so a proved-empty over untyped parameters establishes only "no *int* makes this
   true". `isinstance(x, str)` is infeasible only for an int `x`.
2. **The guard must name no unmodeled stdlib module** (`re`, `os`, `json`, `hashlib`, …).
   Touchstone treats an unmodeled call as falsy and would otherwise certify live code dead.

And, in the driver, a third: infeasibility must be agreed by *every* candidate typing
(§3), and even then `pruned` is only recorded when nothing in the whole portfolio — cheap
or heavy — could produce a verifying flip.

### The cost: a slice discards its caller's context

This is the part that is easy to miss. Lifting a guard out of its function turns the
enclosing function's *parameters* into free variables. The solver is then free to choose a
value the program never actually supplies.

The measured case (commit `7bec5e7`). After saturation, six frontiers remained on
ntlm_auth: four unknown, two certified satisfiable and still uncovered. Checked by hand,
they turned out to be different kinds of thing entirely:

- **`des.py:162`** — satisfiable *and* genuinely reachable; a 3-byte
  `lm_challenge_response` covers it. Blocked only because `block_length` is a callee local.
- **`des.py:164`** — satisfiable but **dead here**. It needs `pad` falsy, and `pad=True` is a
  default that every `.encrypt(...)` call in the package accepts.

The second is a *false live signal*, and it is inherent to slicing: touchstone decides the
sliced guard with `pad` as a free variable, while the program never offers it a falsy one.

`harness_map.call_site_constants` is the answer. It reports the parameters of a function
that every call site in the package fixes to one value — either nobody overrides the
default, or everyone passes the same literal — and the driver folds those into the
precondition as `context_facts`. It is deliberately conservative: any call site passing
something non-constant, or that could not be matched, drops the parameter from the result
entirely. Measured on `des.py:164` this converts a false *satisfiable* into *unknown* —
the false live signal is gone, though not yet a proof (the negated `elif` prefix puts `pad`
in a boolean position touchstone still declines to translate).

---

## 7. Harness fidelity — the hybrid fuzzer

`hybrid/` runs the **unmodified** OSS Atheris harness in the background. When its coverage
plateaus, tharkun solves the frontier and recodes the solution into a byte seed Atheris
reloads (`-reload=1`) and runs past.

```
AtherisBackend (subprocess)  ---writes--->  corpus_dir  <---writes---  AssistEngine
        |                                                                  ^
        | coverage counters                                                |
        v                                                                  |
   HybridFuzzer.run() --- plateau? (should_assist) --- yes --------> assist_once()
```

`HybridFuzzer` separates the plateau decision from wall-clock time (`clock`/`sleep` are
injectable) so the orchestration is deterministically testable against a `FakeBackend`.

### The hard constraint

**The assist may only reach what the original harness can reach statically.** This is a
correctness property, not a nicety, and the reason is that there are two very different
claims hiding behind "new coverage" (`hybrid/validity.py`):

- *legitimate* — the original harness could produce this input; we merely found it in
  seconds instead of never.
- *a lie* — the original harness could never produce this input, and the coverage is an
  artifact of the rewrite.

Only the first is a coverage result. The risk is concrete: a real harness reading
`ConsumeUnicodeNoSurrogates(256)` excludes surrogate codepoints *by construction* and caps
length; a naive hand-written `decode` doing `data.decode('utf-16', 'replace')` can emit
values outside that envelope and reach guards the original cannot.

Rather than reason about the envelope, `check_reachable_by_original` **constructs the
receipt**: it runs the solution back through the original harness's own FDP codec and
checks that the real `FuzzedDataProvider` decodes those bytes to exactly that solution. If
it does, the bytes *are* the proof. If it cannot, that is reported plainly rather than
quietly counted as a win.

### `harness_map` — which field reaches which parameter

The assist originally named a single `target_func` and mapped fields to it positionally, so
frontiers in every other module the harness touched were simply invisible — measured at
**4 of 23 reachable frontier statements on ntlm_auth** (`tests/hybrid/test_crossmodule.py`).

Lifting that needs one fact the positional codec never had: which fuzzer field feeds which
callee parameter. It is recoverable from the harness itself, because a harness is written
exactly this way:

```python
fdp = atheris.FuzzedDataProvider(data)
mod_a.check_a(fdp.ConsumeUInt(4))                      # field 0 -> check_a.code
mod_c.check_c(fdp.ConsumeUInt(4), fdp.ConsumeUInt(4))  # fields 1,2 -> check_c.n, m
```

`map_harness_calls` walks the harness body in source order counting `Consume*` calls, so
the numbering matches `fdp_codec.record_schedule` exactly, and binds each field index to
the callee parameter at the same position.

It is deliberately syntactic, and deliberately refuses in three places:

- **Resolution follows the qualifier the harness wrote**, not the bare name. `mod_a.check_a`
  is resolved through the harness's *own imports*, so a same-named function in a module the
  harness never imports can never be selected. This matters on real packages — ntlm_auth
  defines seven names in more than one module. The fixture package includes a `decoy.py`
  defining `check_a` and `check_b` that the harness never imports, and
  `test_qualifier_decides_which_module_is_targeted` pins that it is never targeted.
- **An ambiguous bare call is dropped, not guessed.** A name defined in two modules and
  called without a qualifier has no sound resolution, so it yields nothing rather than
  whichever came first on disk.
- **`fixed_args` are read from the harness call site, never invented.** A harness pins some
  arguments to constants (`get_seal_key(fdp.ConsumeInt(4), fdp.ConsumeBytes(124),
  SignSealConstants.CLIENT_SEALING)`); without capturing those the call is missing a
  required argument and every branch inside it stays dark for a reason that has nothing to
  do with solving — measured, exactly the last 2 of 56 lines on ntlm_auth. `_resolve_const`
  chases a dotted name through the import that bound it, because a literal check alone is
  not enough.

A harness that stores a field in a local first, or builds a structure before passing it, is
reported with no bindings rather than guessed at.

Note the deliberate asymmetry between what is *targeted* and what is *entered*: the assist
watches every module in the package, because what the harness reaches *through* a call is
in the coverage report too (des.py is **140 statements reached only via `compute_keys`**),
but the entry function is always one the harness itself calls, so the inputs are still only
harness fields.

### `fdp_codec` — the real provider is the oracle

Atheris's byte math is quirky. Rather than reimplement it, `RecordingFDP` wraps a real
`atheris.FuzzedDataProvider`, delegates every `Consume*` call to it, and records a
`FieldRecord` per call — method, args, value, and the exact byte range consumed, measured
from `remaining_bytes()` deltas. Decoding is therefore *always exactly Atheris's*. The
front/back split (data methods consume from the front, bounded-integral methods from the
back) was verified empirically against atheris 3.0.0.

The discipline that matters: `encode_overrides` rewrites the bytes for chosen fields and
then **replays the whole buffer through a fresh real FDP**. An override that does not
reproduce — an unsupported field type, a value wider than the consumed width — is
**rejected, never emitted**. A codec never returns bytes it has not round-tripped; the same
rule applies to the hand-written `FunctionCodec`, whose judge is the harness's own
`decode`.

A rejection is not a dead end, though. `RecodeRejection` records whether the gap is a
*missing encoder* (`mechanically_possible=True` — the bytes exist, we just cannot derive
them yet) or a *genuine field limit* (the harness's own width or range excludes the value,
and no encoding would help). That turns "not recodable" into an inventory of what building
the next encoder would actually buy.

### Widths are not values

`if len(key) != 7` on `lm_hash[:7]` is satisfied for free when `lm_hash` is starved to `b''`
and unsatisfiable while it is 124 bytes. **No flipper can express that**, because width
follows from how much buffer is left when the harness reaches the field — not from anything
that can be written *into* the field (`tests/hybrid/test_width_targeting.py`).

So the requirement is read off the guard (`assist.length_requirements`: `!= 7` on the
"then" side wants width 0, which is always available since a starved field yields `b''`;
`> 4` wants 5; a slice resolves to its *base* name because it is the base whose width has
to change) and `fdp_codec.seed_with_field_width` searches buffer sizes for one where that
field lands at that width, with the real provider as the oracle.

Two things had to be right here, both found by measuring rather than reasoning (commit
`f877397`):

- **The base must come from a *pristine* seed.** Built from the widest corpus seed it
  inherits a previously solved seed's committed values — a `negotiate_flags` already set to
  `EXTENDED_SESSIONSECURITY` takes the first branch, so DES is never called however short
  `lm_hash` is, and the drive plateaus for a reason unrelated to width.
- **It must not be deduped against corpus seed sizes.** The narrow seed *is* already in the
  corpus; it is simply ranked out by "binds the most fields", which is exactly backwards for
  a guard that wants a field ABSENT. Skipping the synthesized base as a duplicate skipped
  the only base that could work.

Result: ntlm_auth, 6 rounds from one 64-byte seed, total **155 → 184** (was 182), des.py
**98 → 100**, entry functions **56/56**.

### Base seed selection, and why no single seed dominates

A seed too small for the harness starves its later fields to zero width, and a zero-width
field cannot be written to at all — on ntlm_auth's 64-byte seed that is **17 of 21 fields**,
i.e. most targets unaddressable for reasons that have nothing to do with solving. So
`widen_seed` produces a wider variant — but it is *added*, never substituted: measured,
replacing the base traded `compute_hash`'s and `des`'s lines for `compute_keys`'s, because
the wide seed starves the *early* fields the narrow one bound. Neither seed dominates. Each
frontier ranks the bases for itself and gets to try several (`max_bases=4`) before being
written off as a plateau.

### Seeds are content-addressed over the whole buffer

Seeds were named by `data.hex()[:40]` — the first *twenty* bytes. FDP consumes ints from
the **end** of the buffer, so every solution differing only in an int field collided on one
filename and all but the last were silently lost. On ntlm_auth that was all three
`negotiate_flags` solutions, and it cost the last uncovered line. `_write_seed` now uses
sha1 over the full content, which is also stable across processes (unlike `hash()`, which is
salted per process).

### When the assist actually pays

`CLAUDE.md` carries the measured table, including negative results, and it is worth
internalising before reaching for the hybrid. A guard defeats modern Atheris only when its
satisfying set is **both** sparse *and* not recoverable from the compared operands —
libFuzzer's CMP/value-profile can copy an answer straight out of a comparison it observes.
Both conditions are required, which is narrower than "modular arithmetic is hard": Luhn's
`(total+d) % 10 == 0` is a tie because it is 1-in-10, and a 32-bit `magic == 0xCAFEBABE` is
a tie because the constant is handed to the fuzzer. Expect a tie outside sparse-and-obscured
constraints, and report it as one.

---

## 8. Known gaps

Stated plainly, because each of these is a place where the code and the aspiration differ.

### `call_mapper` — the measured bottleneck

`src/tharkun/call_mapper.py` exists and works: `map_call_site`, `find_calls_to`,
`map_call_chain` extract callee-param → caller-expression mappings, and `branch_solver`
uses `find_calls_to` in a post-step to translate `best_inputs` into caller variable names.

What does not exist is the **backward direction inside the driver**: given a guard stated
in a callee local, derive the constraint on the entry function's parameter. Every flipper
works on entry parameters, so none can express "make `lm_hash` shorter than 7 bytes". The
driver knows this and says so in its plateau diagnostics (`"guard at line N depends on
non-parameter state; backward propagation (call_mapper) needed"`).

This is measured, not speculative. With cross-file targeting the assist reaches **56/56
lines of every function ntlm_auth's harness calls**, and the **8 remaining des.py frontiers
all share one shape** — a length check on a callee local:

| line | guard | where the local comes from |
|---|---|---|
| `des.py:144` | `len(key) != 8` | `DES.key56_to_key64(lm_hash[:7])` |
| `des.py:162` | `block_length != 8 and pad` | `data[i:i + 8]` |
| `des.py:211` | `len(key) != 7` | `lm_hash[:7]` |

The two partial answers already in place — width targeting (§7) and `_reach_by_search`
(§3) — attack the same class from the sides, one by building a base with the right width
and the other by searching entry parameters with execution as the predicate. Neither
inverts the dataflow.

### Operand banking from failed searches — wired, unverified

`_reach_by_search` runs the program hundreds of times, and every run passes through the
blocking comparison. `_bank_operands` harvests what each trial's comparison demanded and
records it as an `operand` Fact, so a failed search leaves evidence behind instead of
teaching nothing (which also matters for the epoch: a search that records nothing can never
re-open the frontier it just failed on).

The plumbing is in place and the call site passes the store, but per commit `e004e8c` this
has **not yet been demonstrated banking a fact on a real guard** — the attempts so far hit
guards RedQueen legitimately ignores (a bare method call rather than a comparison). Treat it
as unverified.

Half the diagnosis has since landed: `_bank_operands` records a bare `Fact("operand", …)`
holding the demanded *value*, and commit `21d4da9` established that a value is the wrong
thing to bank — the encoding is what the next consumer needs (see §5, *Constraints, not
values*). `Constraint` and its projections exist; `_bank_operands` has not been rewired
onto them.

### The power scheduler — collected, not spent

`FactStore.report()` returns per-strategy served / ran / committed, which is exactly what a
power schedule needs. Nothing consumes it. The portfolio order is currently static and
hand-justified (§3). Given the measured asymmetry — 1158 invocations for 0 committed
answers on one member, against snapshot's 309 — this is the most obvious next lever.

### `structure_infer` is not wired in

`structure_infer.infer_shapes` / `synthesize` recover an input's nested shape from how the
code uses it, and its docstring names the case it is for: the driver plateaus when a
parameter has no live value to mutate, and a synthesized shape gives the loop something to
start from. It has a test file and no callers in `src/`.

### Consolidation debt

Two duplications are known and tracked in `CLAUDE.md`'s P4 list. Minleaf's
`guard_extract.py` (depth-based innermost selection, `assert` support) and
`condition_solver._find_branch_at_line` do the same job differently; minleaf's
`candidates._false_to_true_candidates` and `condition_solver._extract_comparison_values`
share comparison-value logic. Both are still separate.

Also: `types.py` is duplicated between tharkun and tulkas and must be synced by hand.

### Scope of the driver

`drive_to_line` controls **function parameters**. Guards on module-level state, on values
constructed inside the callee from nothing the caller supplies, or on genuinely external
effects, are reported as plateaus with diagnostics rather than silently skipped. Mocking
external calls is `mock_gen`'s job and lives in the per-branch chain, not in the loop.

---

## 9. Where to look

| Question | File |
|---|---|
| How does one line get reached? | `src/tharkun/driver.py` — `drive_to_line` |
| What may be remembered, and for how long? | `src/tharkun/facts.py` |
| How is a guard handed to an SMT engine? | `src/tharkun/touchstone_adapter.py` — `synthesize_probe` |
| What does slicing buy? | `src/tharkun/slicing_bench.py` |
| How is a comparison's demanded value copied onto an input? | `src/tharkun/redqueen.py` — `harvest` |
| How is live state minimally mutated? | `src/tharkun/minleaf/api.py`, `minleaf/patch_search.py` |
| How is one guard solved statically? | `src/tharkun/condition_solver.py` — `solve_condition` |
| How does the per-branch chain compose? | `src/tharkun/branch_solver.py` — `solve_branch` |
| Which fuzzer field feeds which parameter? | `src/tharkun/hybrid/harness_map.py` |
| How do bytes become typed values and back? | `src/tharkun/hybrid/fdp_codec.py`, `hybrid/codec.py` |
| Could the original harness have produced this? | `src/tharkun/hybrid/validity.py` |
| Why does each technique exist? | `tests/test_technique_audit.py` |
| Why does the store look like this? | `tests/test_facts.py` |
| Why does the fact set iterate? | `tests/test_cross_file_drive.py` |
| Why does the assist stay inside the harness? | `tests/hybrid/test_crossmodule.py` |
| Why is width special? | `tests/hybrid/test_width_targeting.py` |

The test docstrings in that last group are the primary sources for most of this document.
When you change one of these mechanisms, the docstring explaining *why it exists* is part
of the change.
