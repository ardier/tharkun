# absint -- Tharkun's sound whole-program abstract interpreter

*Plan [internal ref] group C, lane E. Status: slice 1 (intervals, lengths and relations, fixpoints plus
inlining) and slice 2 (list segmentation, floats on the shared value facts, per-object float
invariants with named lemmas, per-site records). Later slices are listed under "Ruled, not built".*

`tharkun.solve.absint.prove(source, exc_classes)` decides the SV-COMP property
`CHECK(init(main()), ! uncaught(E))` for every `E` in `exc_classes` **without running the program**.
The proof supports the exception classes listed in `solve/absint/__init__.py::PROPERTY_CLASSES`. Any selection
containing another class (including a broader superclass) returns UNKNOWN before analysis.
It never answers FALSE; counterexamples are the search's job. It returns one of:

- **TRUE**, with a proof: every operation that can raise one of the classes has a *check*, and the
  check's fact holds in the proven invariant at that point;
- **UNKNOWN**, naming the first unproven check or the construct it does not model.

`tharkun.sv` calls it through `sv/prove.py::prove` after the deterministic check (docs/SV.md). A proof
there shrinks the search to a *canary*. A FALSE found by the canary wins over the proof, and only a
canary that finds nothing returns the proof's TRUE.

```
python -m tharkun.solve.absint PROGRAM.py --property P.prp [--explain]
```

`--explain` prints every check relevant to the property, together with the invariant fact that
discharges it. Lane F audits proofs from this output.

## What it computes

The interpreter walks the module body from the top, as `__main__`. It keeps one abstract state: a
stack of frames, a heap, and a zone.

- **Values** (`aval.py::V`) keep every *scalar* fact in Tharkun's one shared fact structure,
  `src/tharkun/solve/values.py::Value` (lane D):
  - the kinds;
  - `IntFact` constant bounds;
  - `FloatFact`, an outward-rounded interval with a NaN flag;
  - `StrFact`;
  - the bools.

  absint adds only what a per-variable fact cannot hold:
  - the int part's link to the zone: an exact linear form (`linear.py::Lin`) and bounds on a zone
    variable;
  - the float part's value term (`lemmas.py`);
  - heap references;
  - static objects: builtins, modules, exception classes.

  A `top` value exists, but *any operation on it aborts*. Float arithmetic goes through lane D's
  transfer functions (`solve/values.py::add`, `truediv`, `sqrt`, `sin`, …), which report
  `ZeroDivisionError` exactly when the divisor may be ±0.0 or int 0.
  For `**` the `ZeroDivisionError` is `0.0 ** negative` (`values.py::pow_`): the site fact names the BASE and the EXPONENT (`base in [0, inf],
  exponent in [-1.5, -1.5]: the base may be 0 and the exponent negative`; it used to print the exponent as `divisor in [-1.5, -1.5] may be 0`).
  `bm_nbody_nondet` stays UNKNOWN for `arithmetic` on two real checks, both a pair of bodies at the same point: `(dx*dx + dy*dy + dz*dz) ** (-1.5)` in
  `advance` and the division by `(dx*dx + dy*dy + dz*dz) ** 0.5` in `report_energy` (`tests/test_absint_pow_fact.py`, with native controls).
- **Integers are relational** (`zone.py::Zone`). A difference-bound matrix (DBM) holds `x - y <= c`
  over:
  - the program's int variables (`<frame>:<name>`);
  - list lengths (`L.<site>`);
  - range bounds (`R.<site>.lo/hi`);
  - statement temporaries (`$<depth>_<n>`).

  The matrix is kept closed. `!=` tightens a bound that sits exactly at the excluded value
  (`interp.py::_refine_ne`), which is what turns `while r != 1` with `r >= 1` into `r >= 2`.
- **The heap** is allocation-site based, with the *recency* abstraction (`interp.py::Analyzer.alloc`,
  `Analyzer.demote`):
  - The most recent object of a site (`(site, 'r')`) takes strong updates, and its length is a zone
    variable.
  - Older objects are joined into a summary (`(site, 's')`), which takes weak updates and whose
    length is an interval.
  - A site is `(call string, line, col, tag)`, so every inlined call context has its own objects.
- **Element and attribute summaries** carry symbolic bounds `(var, c)`, meaning `var + c`, and
  they are universal: every element satisfies them. Two rules keep them true:
  1. Before a variable that such a bound names changes, `state.py::State.rewrite_bounds_away`
     rewrites the bound to an equal variable, or to the variable's constant bound, or to infinity.
     `State.shift_bounds` handles `x := x + k` exactly.
  2. A summary is turned into zone constraints (`interp.py::Analyzer.mat`) **only on paths where
     an element provably exists**: after a successful index, `pop`, or unpack, or inside a loop body.
     Materializing the bounds of a possibly empty list would let the zone closure derive `lo <= hi`,
     i.e. non-emptiness. This was a real bug, and `tests/test_absint.py` (`empty_range_list`) is its
     regression.
- **Calls** to the program's own functions are inlined, context-sensitive by call string. Recursion
  aborts. Arity is a checked `TypeError`, and so are keyword and default binding.
- **Loops** (`interp.py::Analyzer.s_While`/`s_For`; the fixpoint is `Analyzer._solve_fixpoint`, behind the reuse of
  `Analyzer._fixpoint`, "Fixpoint reuse" below) are analysed in three parts ("The first iteration on its own", below):
  iteration 1 from the entry state, then the fixpoint of iterations 2 and on, then a final pass from that fixpoint.
  The fixpoint starts from where iteration 1 ended and iterates as follows:
  1. plain joins for `WIDEN_DELAY` updates (0 today: widen at the first update);
  2. then widen (DBM widening, then symbolic-bound and length-interval widening);
  3. then take **one decreasing step that is kept only if it is itself inductive**: the body maps
     it into itself. This makes soundness independent of whether the transfer functions are
     monotone.
  4. then, for source `for`/`while` loops only, try a *relational* strengthening: `loop_family.py`
     ([internal ref], adapted from `py-inv/eregion-ir`'s lowered-frame classifier) tags the loop's shape
     from its AST (`counter`, `accumulator`, `search`, `iterator`, plus the shared `scan`/`affine`/
     `kernel` tags) and the family only **selects which pairs of loop-body-assigned variables to
     try** a zone bound between (`a - b <= 0` or `<= 1`). The tag is a search hint, never a proof:
     every proposed bound is independently required to (a) already hold at loop entry
     (`entry.zone.entails`), (b) still cover entry once added (`entry.leq(proposed)`), and (c) be
     itself inductive (`step(proposed).leq(proposed)`) before it replaces the head. A wrong
     classification can only make step 4 try the wrong pairs and find nothing; it cannot weaken (a)-(c).
     Search is bounded to loops with at most 4 candidate variables. `COUNTER`- and `SCAN`-classified
     loops skip step 4 entirely (`_solve_fixpoint` only tries it for `ACCUMULATOR`, `AFFINE`, `ITERATOR`,
     `KERNEL` and `SEARCH`); "entry" there is iteration 1's end state.

  Checks are recorded in the first-iteration pass and in the final pass, never inside the fixpoint iterations, and a
  site is proven only if it is proven in both. A `for` over a range binds the loop variable with
  `lo <= i <= hi - 1` to the range's immutable bounds.
- **Exceptions.** A check that is not proven records `ok=False`, and the analysis continues on the
  non-raising states. A raising execution ends the program, so it cannot reach a later site. `raise`
  of a class records exactly the property classes that class is, or derives from, including user
  subclasses of `KeyError` and the like. Non-property exceptions (`ValueError`, `OverflowError`,
  `MemoryError`, `NameError`, `UnboundLocalError`) end the path.  Inside a `try` statement none of this is
  final: an unproven check is an exception in flight routed to the handlers, and a path that would end
  silently is refused ("try/except/else/finally" below, [internal ref]).

## Lists: segmentation (slice 2)

A list made by repetition (`[None] * n`) is *segmented* (`heap.py::ListObj.pre`). A zone variable
`P.<sid>` splits its indices:
- every element below `P` is in `pre`;
- every element at or above `P` is in `elem`.

A store at exactly index `P` extends the prefix (`interp.py::Analyzer.store_elem`). A `for i in range(c,
hi)` with a constant `c` carries a ghost counter `G` (`Analyzer.s_For`):
- the body sees `i = c + G` exactly;
- the exhausted exit has `G >= hi - c`.

So `for i in range(n): xs[i] = X` ends with `P = n = len(xs)`, and every element is an `X`.
Operations that shift indices, or that could leave `len < P`, collapse the segmentation: `insert`,
`pop`, `reverse`, slice assignment. In-place repetition aborts.

## Floats: value terms, focus, object invariants and lemmas (slice 2)

- **Value terms** (`lemmas.py`) name how a float was computed:
  - `trig(f, key, k)` for `k * sin/cos(x)`;
  - `rd(addr, attr, ver)`, an attribute's value;
  - `sumsq`, `sqrt`, `div`.

  Equal terms mean equal floats. So the interpreter **drops a term as soon as anything it names can
  change**:
  - an int variable is written, shifted or dies (`state.py::State.rewrite_bounds_away`,
    `State.shift_bounds`);
  - an object is refreshed, focused, blurred or demoted (`State.strip_terms`).
- **Focus/blur** (`interp.py::Analyzer.focus`, `Analyzer.blur`): a method called on an object of an
  allocation-site *summary* runs on a materialized copy `F`, which takes strong updates.
  - Every reference to the summary, including the call's arguments, may now also be `F`.
  - Afterwards `F` is joined back into the summary and its terms are dropped.
  - A method on the recent object *refreshes* its attribute versions (`Analyzer.refresh`) after
    dropping every term that names it.
- **Object invariants** (`heap.py::InstObj.qf`): for every concrete object an abstract object
  stands for, the real sum of the squares of a group of float attributes lies in `[lo, hi]`.
  - After every strong attribute store, `Analyzer.recompute_qf` re-derives it from the attributes'
    terms. A weak store drops every group that contains the attribute.
  - Joins take the hull. Widening jumps to the next *landmark*, a bound some lemma produced
    (`lemmas.py::widen_lo`), and the set of landmarks is finite.
  - `Analyzer.sumsq_lower` applies an object's invariant to `x*x + y*y + ...` over reads of that
    object's current attributes.
- **Lemmas**: each is stated once in `lemmas.py`, and every fact that uses one names it.
  - **TRIG_PAIR**: for every finite float `x`, `max(|sin x|, |cos x|) >= sqrt(2)/2 - 1e-9`.
    - Hence `a*sin(x)^2 + b*cos(x)^2 >= 0.4999 * min(a, b)`.
    - *Assumption, stated*: the platform libm's `sin`/`cos` are within 1 ulp for every finite
      argument, with full argument reduction. glibc and macOS libm are.
  - **NORMALIZE**: if `S = fl(sum a_i^2)` lies in `[2^-500, 2^500]` and `N = fl(sqrt(S))`, then
    `sum fl(a_i / N)^2` lies in `1 ± 1e-12`.
    - The precondition on `S` is checked from the interval that the `sqrt` term records, and it is
      printed in the fact.
    - A term that does not satisfy it does not match.

`bm_float_nondet` arithmetic is proven this way:
1. `__init__` establishes `x² + y² + z² ∈ [0.4999, 10.25]` via TRIG_PAIR;
2. `normalize` divides by `norm >= 0.707`, using that invariant;
3. `normalize` re-establishes the invariant: `1 ± 1e-12` via NORMALIZE, which stays inside the hull.

## Float draws, float literals and float comparisons ([internal ref])

- **`_sv_verifier.nondet_float_range(a, b)`** (`interp.py::Analyzer.m__sv_verifier_nondet_float_range`) is modelled BOTH ways
  ([internal ref]): a TRUE is proven over the larger reading, a FALSE is confirmed only with a value the helper's code can produce
  (docs/SV.md "Draw domains: both readings"). The abstract value is the JOIN of two sets:
  - the documented one, the set of draws the grading shim ACCEPTED before [internal ref] (`eregion-sv/src/eregion_sv/shim/_sv_verifier.py`):
    it draws any float `v` and calls the run INVALID, never a violation, when `isnan(v) or not min(a, b) <= v <= max(a, b)`, so the
    non-NaN floats in `[min(a, b), max(a, b)]`: bounds inclusive, an infinite bound included, `a > b` legal, `-0.0` and `0.0` both accepted when
    either lies in range. Non-constant bounds give the hull of the bounds' facts (a sound over-approximation); a bound with no usable fact
    gives every float. `min`/`max` compare with `<`/`>`, which are False for NaN, so a NaN bound is NOT symmetric: `a` NaN accepts nothing
    (`min(a, b)` is `a`, and `nan <= v` is False); `b` NaN with a number `a` accepts exactly `a` (`min` and `max` both return `a`);
  - the code's: `random.uniform(a, b)` = `a + (b - a) * random()`, `Analyzer.uniform_formula` (`rand.py`: `binop` Sub, Mult, Add over the draw `[0.0, 1.0]`, the
    outward-rounded transfer functions of `solve/values.py`). It adds what the documentation excludes: `inf` and `nan` for a span that overflows
    (`uniform(-1e308, 1e308)` is `inf` or `nan` and was proven `<= 1e308`), for an infinite bound and for a nan bound. The formula's hull contains the documented one (``a + (b - a) * [0, 1]`` runs from ``a`` to ``b``, and covers everything for a nan
    bound), so the join is the ruling stated twice, not a source of imprecision: for constant finite bounds the
    two sets have the same hull, so every proof that held still holds; for bounds that are themselves draws the formula does not relate `a` to itself and
    is wider (`a` in `[0, 1]`, `b` in `[10, 20]`: `[0, 21]`, so `x <= 20.0` is no longer proven): the price of proving a TRUE over both.
  It is not `nondet_float()`, whose model is any float. A call that certainly raises ends the path; non-numeric bounds are the `TypeError` `min` raises
  (recorded, `_num_arg`); the call binds `a` and `b` by position or name ([internal ref]).
  - A violation reachable only through the gap (`y == 1.0` for `(0.0, 1.0)`) is UNKNOWN here and not confirmable by the child: MIN's `cmp_g07_float_range_neg`.
- **Float literals are exact.** `e_Constant` gives a float literal its one-point `FloatFact` (it was the
  fully generic `FLOAT`, which is why `x = 0.5; assert x > 0.0` could not be proven). Arithmetic on
  literals goes through the outward-rounded transfer functions of `solve/values.py`, so
  `0.1 + 0.2 == 0.3` (False in IEEE-754) is still never proven.
- **Float comparisons decide a branch** (`Analyzer._prune_float_compare`, called from
  `compare_split`): when at least one operand is a float and both are purely numeric, the operands'
  facts go through `values.py::refine_compare` (NaN makes every ordered comparison and `==` False and
  `!=` True; an int is compared with a float exactly; `-0.0 == 0.0`), and a side that leaves no pair of
  values is an impossible branch. Only that: the operands are not narrowed inside the branches (a name
  keeps its fact), so `x >= 0.0 or x < 0.0` over a range that excludes NaN is still not proven. Int
  against int keeps the zone.
  An int beyond the float range sits between the finite floats and the infinity (`inf > 10 ** 400` is True): the view of
  a huge positive int is `[largest finite float, inf]` (`values.py::_cmp_view`; it used to put it AT `inf`, which pruned
  `x > n` for `x = inf`, and raised `OverflowError` for an int beyond the rounding midpoint), so the branch for an infinite
  `x` stays and one for a finite `x` is pruned (`tests/test_absint_float_range.py`, the huge-int cases; found by MG's review).
- **Why this matters**: chaos draws `nondet_float_range` bounds for its spline thickness and compares
  floats against literals; without these three pieces the analysis aborted at the call (`unsupported:
  call of an unknown value`) or lost every float comparison to the generic `FLOAT`.
- Tests: `tests/test_absint_float_range.py` holds the abstract domain to the shim's own predicate over the
  draws that matter (the bounds, both zeros, the infinities, NaN), proves nothing for a draw the shim
  accepts (the strict versions of every bound stay UNKNOWN), and runs a seeded differential: a program
  absint proves safe never raises for any accepted draw pair out of fourteen draw points.

## The iterable-reducing builtins ([internal ref])

`min`, `max`, `sum`, `any`, `all` and `sorted` read an iterable and reduce or reorder it. They are one family
(`solve/absint/reducers.py::ReducersMixin`, mixed into `interp.py::Analyzer`; the `c_<name>` convention of
"Contract-driven builtins", so the tested contracts in `solve/contracts/rules_builtins.py` give the seven classes
and each model upgrades a class to proven only for the reason it states). The source is a list, tuple, array,
range, dict (its keys) or a str; the call reads it and changes nothing; the result is a function of two facts
the analysis already holds, the bounds of the length (`len_lin`, `_range_len`, `dict_len`) and the join of the
elements (`elem_summary`).

- **`min(it)` / `max(it)`**: the result is an element, so it is the join of the elements, with their relational
  bounds when the call must have found one (`iter_elem`: the call continues only on a non-empty iterable, since an
  empty one is a `ValueError`, which is not a property class). `default=` is joined only when the length may be 0
  (its bounds are made constant: no element is known to exist). `min(a, b, ...)` keeps its numeric model.
  `key=None` is no key. Two or more elements that may not be orderable (None, a str next to a number) are a
  TypeError check that is not proven; one element is never compared.
- **`sum(it, start=0)`**: integers (and bools) are exact: the sum of `n` elements in `[a, b]`, `n` in `[lo, hi]`,
  lies in `[lo*a, hi*b]` when `a >= 0` and `b > 0` and in the corresponding extremes otherwise
  (`reducers.py::_sum_bounds`, checked exhaustively on small ranges), plus `start`; `None` is unbounded and `0`
  absorbs. A float anywhere gives an unknown float: CPython compensates float sums since 3.12, so the left fold is
  not the same number and no interval of it is claimed. `start` must be a number; a str or None element is a
  TypeError.
- **`any(it)` / `all(it)`**: a bool from the length bounds and the truth values the element fact leaves possible
  (`any([])` is False, `all([])` is True).
- **`sorted(it, reverse=<scalar>, key=None)`**: the existing list model (`c_sorted`), now taking the two keywords
  that change nothing about the result's facts.
- **Refused (`Unsupported`, the proof is UNKNOWN as before)**: a `key=` function (a call per element, in order,
  with the effects of each; not built, an exact lowering to a loop is the shape that would do it), an iterator
  (the call consumes it) or a dict view, an element that is an object or an unknown value (a hook would run per
  element: `__lt__`, `__add__`, `__bool__`), another keyword. So every class the contracts admit only through a hook
  is proven for the elements that remain: numbers, strs and None.

The symbolic executor has the same family (`sv/_symexec.py::_Exec._extreme_of` and the `any`, `all`, `sum`,
`sorted` branches of `ev_Call`) on the unrolled elements, symbolic ones included. Its claims are advisory (a vector
is replayed natively before it is believed), and it refuses the same `key=` function.

Tests: `tests/test_absint_reducers.py` (exactness, refusals, a differential that runs 17 sources x 9 reducers x 9
reads natively over every pair of draws and fails on a TRUE for a class the program raises),
`tests/contracts/test_builtins.py::TestAnyAll` (the new `any`/`all` contracts) and
`tests/test_symexec_reducers.py`. `scripts/absint_fuzz.py` draws the reducers over the list, tuple and ranges.

## Site records (shared with lanes A and B)

Every check also yields a record (`AbsintResult.records`; `--json` prints them) with these fields:
- `line col exc kind function guards`: lane A's `sv/sites.py` fields;
- `status`: `safe` or `may_raise`;
- `fact`: the proof, for a safe site;
- `operand`;
- `residual`: the violation condition, as a plain Python expression over the names at the site
  (e.g. `(norm) == 0`, `not (-len(xs) <= (i) < len(xs))`, `(p) is None`);
- `scope`: **must-facts only**. They are the zone and float bounds of the residual's names in the
  proven invariant at the site, intersected over every visit;
- `types`;
- `context`: the inlining call string;
- `loop_iteration`: `"any"`;
- `source_sha`.

`AbsintResult.complete` is True only when the analysis ran to the end. Then the records cover
every operation of the program that can raise the property's classes, because absint aborts on
anything it does not model. Lane A solves `may_raise` records for a witness, and lane B refutes
them for a certificate.

## Soundness rules (absolute: a wrong TRUE costs 32 points)

- **Every unmodelled construct aborts** (`interp.py::Unsupported`):
  - `with`/`match`, generators, lambdas;
  - comprehensions the exact lowering refuses (below: the lowering turns the others into loops
    before the analysis runs, so only a refused shape aborts);
  - nested functions and closures, recursion;
  - `eval`/`exec`/`getattr`/`setattr` (unmodelled builtins);
  - any library function outside the model table (`interp.py::MODULES`);
  - multiple inheritance, metaclasses, decorators other than `staticmethod`;
  - classes that define an implicitly called dunder (`__eq__`, `__getitem__`, `__add__`, ...;
    `interp.py::UNSUPPORTED_DUNDERS`);
  - `__str__`/`__repr__` reached by printing or formatting;
  - a dunder-shaped attribute (or `__dict__`) that `class_lookup()` does not find anywhere in
    the MODELED class hierarchy (`interp.py::_reflective_unresolved_name`, used by
    `attr_load`/`call_method`). [internal ref] (soundness): `class_lookup()` returning `None` means
    "not found in the modeled hierarchy", never "does not exist on the real object" -- every
    real class/instance also inherits `object`'s own dunders (`__repr__`, `__str__`,
    `__hash__`, ...) and every class has a real `__dict__`, none of which this analyzer
    represents as heap classes. Treating an unresolved dunder-shaped name the same as an
    ordinary undefined one silently took the state to bottom (unreachable) instead of raising
    `Unsupported`, so a REACHABLE site downstream of the dunder's (address-showing, unboundable)
    text was never visited -- a wrong TRUE with 0 checks recorded. `prove()` also cross-checks a
    0-relevant-checks TRUE against `census.py::site_census`'s OWN, independent scan as a defence
    in depth: if that scan ALSO could not resolve the program (not merely because it found an
    ordinary, absint-provably-infeasible site), the verdict is downgraded to UNKNOWN even if some
    other unmodelled construct in the future makes the same bottom-instead-of-abort mistake this
    one did;
  - any OTHER attribute `class_lookup()` does not find, when the receiver's class extends a
    builtin base this analyzer does not represent as a heap class at all
    (`interp.py::Analyzer._mro_has_unmodelled_base`, keyed on the `__exc_base__` marker
    `s_ClassDef` records for `class E(Exception): pass`-style bases -- currently the only
    builtin base a user class may extend). [internal ref] part 2 (soundness, MG): a subclass of
    `Exception`/`ValueError`/`BaseException`/`KeyError` still has real, always-present
    inherited attributes this analyzer's class model has no entry for (`args`,
    `with_traceback`, ...) -- `class_lookup()` returning `None` for one of THOSE means the
    same unknown, not absent as the dunder case above. Every builtin base carries attributes
    outside the user-class model (the exception family today; any of `int`/`str`/`list`/
    `dict` and so on a future extension might allow as a base), so the check is general --
    ANY class_lookup() miss anywhere in an MRO that passes through such a base -- rather than
    one enumerated attribute name at a time.
- A list, str, or float operation that can raise records a check, whether or not its result is
  used. Examples: `5 % n` as a bare statement, `x ** -1` with `x` possibly 0, `pop()` on a
  possibly empty list, and a slice assignment that may shrink the list.
- An in-flight value never names a variable that a callee could change:
  - lengths read by `len()` are copied into temporaries;
  - heap reads are materialized into temporaries (`interp.py::Analyzer.mat`);
  - EVERY read of a module global that some function declares `global` is materialized at read
    time (`interp.py::Analyzer.load_name`, via `mat`/`to_temp`), whether the read happens at
    module scope or inside a function frame. [internal ref] (soundness, the ROOT fix): a prior version only
    did this for reads inside a function frame (module-scope reads returned a live reference into
    the global's mutable zone slot); an operand read at module scope, followed later in the same
    statement by a call that rebinds that global, then observed the call's post-mutation value
    instead of the value that was actually live when the operand was read -- CPython evaluates
    operands strictly left-to-right, so this was a wrong TRUE (17 of 25 evaluation-order attacks in
    `tests/test_absint.py`'s `eo_`-prefixed `NEVER_TRUE` entries, e.g. `x = y + f()` where `f`
    reassigns `y`, and the `induction_call_mutates_operand`/`d1_call_args_j3` verdict-level
    probes in the eregion-sv soundness suite, e.g. `y > lower()` and `first(y, lower())`). Fixed by
    removing the function-frame-only condition; materializing a global that nothing could still
    mutate before it is used costs only precision, never soundness.
  - an operand already evaluated is ALSO explicitly materialized again before any operand to its
    right runs, unless that operand is provably side-effect-free (a bare name or a constant) --
    `Analyzer.save_operand`, used by `cond_compare` and `e_BinOp` ([internal ref], predates [internal ref]). Once
    every global read materializes at the source (`load_name`, above), `save_operand`'s own
    materialization of a global operand is redundant (the value it receives is already a fresh,
    decoupled temp) -- but the function is not soundness-critical any more for that reason alone,
    and is kept for a DIFFERENT, purely precision reason: it still snapshots into a temp BEFORE any
    call in the upcoming operand can invalidate the zone reasoning some OTHER value might be
    relying on (e.g. a heap-derived value passed through unrelated to any global), while skipping
    that extra snapshot when the upcoming operand cannot possibly call anything, which keeps an
    exact zone identity (e.g. `x * x`) that a needless re-materialize would otherwise widen away.
- The heap is garbage-collected only at module level: callers may hold in-flight references while
  a callee runs (the holders and their scopes: "In-flight references" below).
- Reaching one allocation site twice within one statement aborts, because recency redirection
  cannot reach in-flight values.
- Besides a `try` statement ("try/except/else/finally" below), the ONE exception to "an uncaught raise
  inside a call ends the path" is the length probe of `list(x)`, `sorted(x)`, `list.extend(x)` and `l += x`
  (`interp.py::Analyzer._probe_length`, reached from `iterable(..., consumer=...)`).  CPython's
  `PyObject_LengthHint` calls the object's `__len__` and swallows `TypeError` and its subclasses from it, then iterates
  with no hint; every OTHER class (`AttributeError`, `ValueError`, `KeyError`, ...) propagates (an
  earlier version also swallowed `AttributeError`, and restored the state from before the call, which lost
  the callee's writes to globals and dropped the partly raising paths: five wrong TRUEs, [internal ref] batch 5).
  An explicit `raise` of a `TypeError` is followed exactly: it is an exception in flight (`_Exc`, [internal ref]) that
  `call_func` hands to the probe instead of the caller when it leaves `__len__` into the function the probe
  runs in (`_ProbeScope.fctx`); the probe joins that state into the continuation, so side effects made before
  the raise survive and a conditionally raising `__len__` continues on both kinds of path.  A `TypeError` that
  comes from an unproven check inside the probe (an arity mismatch, `len(5)`, a non-int result) leaves no such
  state: the probe refuses.  WHICH callables probe is a fact about the callable, not the object:
  `solve/contracts/adapters/common.py::LENGTH_HINT_RULES` (`list`, `sorted`, `list.extend`, `list.__iadd__`
  probe; `tuple`, `set`, `array.array`, `array.extend` do not, and running `__len__` there would make its raise
  wrongly end the path); a callable the table does not know, given an instance whose class defines `__len__`,
  is refused. `tests/contracts/test_swallowing_native.py` pins the table against CPython.

## Comprehensions: the exact lowering ([internal ref])

`interp.py` has no model of a comprehension. `prove` therefore runs `solve/lowering`
(`lower_comprehensions`, [LOWERING.md](LOWERING.md)) on the sliced program first, builds the `Analyzer`
on the lowered tree, and puts every report back in terms of the original program. The lowering is the
SAME program (same value, exception and order of effects), so a proof of the lowered program is a proof
of the original. It refuses what it cannot make exact: a lazy generator expression (`any`, `all`, `next`,
`sum`, `min`, `max`, `zip`, ...), `*args`, `**kwargs`, f-strings, assignment expressions, comprehensions
in a loop test, a `with` item, a default value, a decorator, an augmented assignment, a lambda or a
class body, and any program that can reach its own namespace (`locals()`, `globals()`, a frame
attribute). A refused comprehension stays in the tree and absint aborts on it as before.

Findings are about the original program:

- a check lands at the line and column of the original node;
- a fact about a fresh loop variable is restored to the source's name only inside its own
  comprehension, and dropped elsewhere; a fact about an accumulator or temporary is dropped;
- no invariant is captured at a statement the lowering invented or names a variable of the lowering.

`tests/test_absint_lowering.py` pins the seam, including the case where a leaked loop variable would
make an unsafe division look safe, and `tests/test_absint_lowering_fuzz.py` runs every program absint
calls TRUE for real.

## Import boundary ([internal ref]): read-and-recurse, not allow-by-name

Before this, an import from outside `interp.py::MODULES`'s small model table (`math`,
`random`, `_sv_verifier`) was always built `top` (`src/tharkun/solve/absint/interp.py::Analyzer.module_attr`),
so any CALL through it aborted (`src/tharkun/solve/absint/interp.py::Analyzer.call`, `if fv.top: raise
Unsupported("call of an unknown value", node)`) -- a program-imported library function
could never contribute a proof, however simple.

`solve/extern.py` and `solve/pyresolve.py::Resolver` (shared with `solve/census.py` and
`solve/determinism.py`, see docs/SV.md "Import boundary") give an unresolved import ONE more
chance before refusing: is it found via a directory the PROGRAM ITSELF adds to `sys.path` -- a
literal, static `sys.path.insert(0, D)`, module-level, unconditional, before the import it
enables (never an ambient path, never site-packages)? If so, `Resolver` recursively resolves
that file (cached, cycle-guarded) the same way it resolves a stdlib entry, exposing it via
`Resolver.externs`/`Resolver.all_resolvers()`.

When SV proof is given a validated `TaskEnvironment`, its certificate, static-site and obligation
checks pass that task's declared file manifest to the resolver. As a final fallback only when
literal-root resolution finds no file, it may resolve one unique, direct `.py` sibling of the
primary file by basename; declared siblings may use the same manifest transitively under the same
check. This does not consult the ambient working directory or import path. Dotted and relative
fallback imports, packages, ambiguous names, and sibling names that collide with stdlib or modeled
modules remain unsupported. Until obligation identities include a filename, the obligation census
leaves imported-module locations as an explicit coverage gap rather than treating a primary-only
table as complete.

`Analyzer.module_attr`/`Analyzer._extern_attr` (`interp.py`) build ONE such `Resolver` per
`Analyzer` run (`Analyzer._extern_resolver`, best-effort: any failure to build it, for any
reason, degrades to the unconditionally-safe `top` fallback -- read-and-recurse simply never
applies) and, the first time a name from a resolved file is actually needed, interpret that
file's own top level for real (`Analyzer._exec_extern_toplevel`) into the SAME shared state,
under its own namespace (`Analyzer._global_key`, `FuncCtx.module_key`, `FuncObj.module_key`):
a function read in this way resolves its OWN free names against ITS OWN file's globals, never
the caller's, exactly as CPython's module-global scoping would. `module_key == ""` (the
program's own module) is byte-identical to every program's behaviour before [internal ref] -- a
program that never establishes a root is completely unaffected. Allocation SITES are
similarly file-discriminated (`Analyzer._cur_module_key`, threaded through `Analyzer.site`),
so a same-line-and-column allocation in the program and in a recursively-read file never
collide as "the same site, summarized together" -- a real, if easy to miss, soundness hazard
of representing more than one file's code in one abstract state.

`s_Expr`'s `Interpreter._is_root_establishing_insert` gives the ONE exact
`sys.path.insert(0, D)` call shape `src/tharkun/solve/extern.py` itself already trusts to establish a root a
no-op instead of "call of an unknown value" -- `sys` is not modelled at all otherwise, and
this is the only `sys.path` shape given that trust; anything else still refuses exactly as
before.

Two incidental, narrowly-scoped gaps were fixed as real prerequisites of this validating at
all (`isinstance` and every zero-arg str method were otherwise entirely unmodelled,
independent of the import boundary): `Interpreter.b_isinstance` for a bare `str`/`int`/
`float`/`bool` reference and an unambiguous receiver kind (`bool` as an `int` subclass
handled); `Interpreter.call_method`'s `_TOTAL_STR_METHODS` set (`strip`/`lstrip`/`rstrip`/
`lower`/`upper`/`title`/`capitalize`/`swapcase`, zero arguments -- total, never raise, for any
str). Neither narrows an existing capability; both were unconditional refusals before.

Validated: `string_utils.validation.is_string`/`is_full_string` (python-string-utils, the
GUUT benchmark, [internal ref]/[internal ref]) prove TRUE through the import boundary, with checks
attributed to their own file's own lines. `is_palindrome` and the regex-using functions of
the same package remain UNKNOWN -- they need `re.Pattern` method modelling (`.sub`/`.match`/
`.search`/`.findall`), a separate, substantial absint gap outside [internal ref]'s scope.

## os / os.path ([internal ref]): a three-way split, not a blanket refusal

Before this, `os` was not in `interp.py::MODULES` at all, and (independently) not in
`solve/determinism.py::DET_MODULES` or `solve/census.py::MODULES` either -- `import os`
made `solve/pyresolve.py::Resolver._imports` raise `Top` unconditionally
(`"imports os"`), before any of the three provers built on it ever looked at what the
program actually did with it. A program that wrote `import os` only to compute a
`sys.path.insert` argument (the [internal ref] import-boundary probes themselves, PR 49, bd
[internal ref]) could never get a single check proven, anywhere in the same file.

`os` and `os.path` are now allowlisted (`interp.py::MODULES`, `determinism.py::DET_MODULES`,
`census.py::MODULES`), each attribute classified into exactly one of three buckets --
documented once, in `src/tharkun/solve/determinism.py` (`OS_CONST_OK`, `OS_PATH_PURE`, `OS_HAVOC`,
`OS_PATH_HAVOC`), and mirrored by `interp.py::MODULES["os"]`/`["os.path"]`'s own
`consts`/`funcs`/`refused` entries and `m_os_*`/`m_os_path_*`:

* **pure** (`join`, `dirname`, `basename`, `splitext`, `normpath`, `abspath`, and the
  platform constants `sep`/`linesep`/`pathsep`/...): a function of its own string arguments
  only, no I/O -- `src/tharkun/solve/absint/interp.py::Analyzer._os_path_arg`/`Analyzer._os_path_unary` compute the EXACT
  result (via the analyzer's own, real `os.path`) whenever every argument is a single known
  `str` constant, else fall back to a sound havoc string (`ANYSTR`), never a refusal.
  `abspath` is exact ONLY for an already-ABSOLUTE literal (`posixpath.abspath` is just
  `normpath(path)` for one, no syscall at all); a relative one is havoc -- the real cwd is
  not, and must not be, guessed (the same principle `src/tharkun/solve/extern.py`'s condition (a) already
  applies to a `sys.path.insert` argument). `realpath` looks like the same shape but is NOT
  pure even for an absolute literal -- it dereferences symlinks against the REAL, LIVE
  filesystem (`posixpath.realpath` calls `os.readlink` in a loop), so its result for the
  exact same string can differ host to host (MG's PR 106 review, attack `o06`:
  `os.path.realpath("/var/run") == "/run"` on [host], a real symlink there). `realpath` is
  in the HAVOC bucket below, never computed exactly, on purpose.
* **havoc, category (b)** (`getenv`, `getcwd`, `listdir`, `urandom`, `getpid`, `cpu_count`,
  `realpath`, `os.path.exists`/`isfile`/`isdir`): real process/OS/filesystem state a replay
  cannot control -- the CALL never refuses, but its VALUE is an unconstrained value of the
  right type/shape (`m_os_getenv`: `str | None`, or `str | type(default)` with a default
  given; `m_os_listdir`: a `list[str]` of unknown length, `interp.py::new_list`;
  `m_os_getpid`: `int_iv(1, None)` -- a real, non-trivial gain, e.g. `1 / os.getpid()` now
  proves TRUE for ZeroDivisionError, never provable before). Every HAVOC function's
  argument TYPE is still checked (`_os_arg_type_ok`, MG's PR 106 review: the first cut
  checked arity and `top` but never the argument's type, so `os.getenv(5)`,
  `os.urandom('x')`, `os.path.isfile(None)` -- all real, native `TypeError`s -- sailed
  through to a havoc RESULT instead of a checked violation, a wrong TRUE) -- a definitely
  wrong type never refuses the CALL itself, only records the check, same as `_os_path_arg`
  above. `determinism.py::check_os` refuses the DETERMINISM claim
  unconditionally for any of these, the same conservative stance already taken for
  `os.urandom`/unseeded `random`/`id`/`hash` -- not the finer "unless it can never reach
  output" rule `determinism.py::check_text` uses for text conversions specifically.
* **refused** (`os.environ` used bare, `system`, `remove`, `chdir`, `mkdir`, `fork`, `kill`,
  `exec*`, `putenv`, `os.open`, ...): `src/tharkun/solve/absint/interp.py::Analyzer.module_attr`'s new `"refused"`
  bucket (checked before the pre-existing "known module, unmodelled name -> top" default)
  raises `Unsupported` on the ATTRIBUTE itself, not just on a call of it. `os.environ` is
  refused rather than modelled as a mapping on purpose: the `("module", "os.path")`-style
  tag that makes `.get`/further-attribute dispatch work for a real nested module does NOT
  make it subscriptable the way a real `Mapping` instance is -- `os.environ[k]` on that tag
  fell through the interpreter's subscript handling with no heap object to find, which read
  as an IMPOSSIBLE operation (state -> bottom) rather than an unproven check, turning a
  real, environment-dependent `KeyError` into a wrong "TRUE, 0 checks"
  (`tests/fixtures/determinism/never_certified/d07_environ.py` is exactly this shape, and
  caught it). `os.getenv` has no such trap (a plain function call, never subscripted) and
  is fully modelled; a real `os.environ` mapping model is future work, not shipped here.

`pyresolve.py::Resolver._module_of`/`check` gained the ONE level of submodule-attribute
nesting `os.path` needs (`os.path.NAME` off a plain `import os`): the existing single-hop
`Resolver.check`'s `Attribute` branch only ever matched `Name.attr` (`node.value` itself a
bound module alias); `os.path.dirname`'s OUTER `.dirname` attribute has an `Attribute` (not
a `Name`) as its own base, so it fell through to the flat, permissive `known_attrs`
fallback -- accidentally fine for `.join` (`str.join` also exists) but a real `Top`
("attribute ... resolves to nothing known") for `.dirname`/`.basename`/etc, caught by
`tests/test_os_module.py::test_pure_os_path_is_deterministic`. `_module_of` resolves a
`Name` via `imports.modules` as before, or recurses one `Attribute` level and admits the
dotted name only when it is itself in the Resolver's OWN `allowed` set -- `os.path` here,
nothing else, so no other module's attribute chains are affected.

Never-TRUE coverage: `tests/test_os_module.py` (unit-level, every bucket above, both
`prove()` and `determinism_census()`) and two new lane F fixtures,
`tests/fixtures/determinism/never_certified/d27_os_getenv.py` /
`tests/fixtures/determinism/never_certified/d28_os_path_exists.py` (`test_lane_f_probes_are_never_true_end_to_end`, `tests/test_sv_prove.py`)
-- alongside the pre-existing `tests/fixtures/determinism/never_certified/d04_urandom.py` and `tests/fixtures/determinism/never_certified/d07_environ.py`, both still never-TRUE
and now refusing for a more specific, on-point reason than the old blanket "imports os".

Measured: 0/24 Real-world (pyperformance) + 0/47 Solve-only tasks import `os` today (bd
[internal ref]) -- this closes a blocker for FUTURE real-world tasks and for the [internal ref]
GUUT-benchmark import-boundary work (`python-string-utils`'s `manipulation.py` uses `os`),
not a gain on the CURRENT corpus; `campaign_diff.py`-style direct `tharkun.sv` comparison
on both, before vs. after: 0 gained/lost/new_wrong (one apparent "lost" on
`so_base3_bug`/data-lookup was pre-existing hybrid:atheris search-timing flakiness,
confirmed by re-running the SAME build three times, not a regression).

## array ([internal ref]): a `ListObj` subclass, not a new heap machinery

Before this, `array` was in `determinism.py::DET_MODULES` and `census.py::MODULES` (both
already allowed `import array`) but not in `interp.py::MODULES` at all -- `array.array(...)`
resolved to `top`, and calling it raised `"call of an unknown value"`, so no program that
touched an `array` past construction could get a single check proven ([internal ref];
found by W49's gap check on #0957). This was previously listed under "Ruled, not built",
Slice 3.

**Heap shape** (`heap.py::ArrayObj`): a `ListObj` subclass, not a new representation --
indexing, slicing, `len`, and iteration all reuse `len_lin`/`index_ok`/`elem_at`/`all_elems`/
`store_elem`/`set_len` completely unmodified, because those already read only `length`/`elem`/
`pre`, which `ArrayObj` inherits verbatim (`pre` is always `None`: the `[None] * n` prefix-
segmentation proof list gets is list-only future work, not needed by any known SV-Benchmarks
array use). The one field list doesn't have is `typecode`, which gates every element WRITE
(`interp.py::Analyzer._array_store_value`, called from construction, `append`, `extend`, and
index/slice-repeat) -- real `array` enforces a `TypeError`/`OverflowError` list does not.
`state.py`'s `ojoin`/`oleq`/`_map_obj` each gained one clause mirroring `"list"`'s own (join
requires the SAME typecode; `Incompatible` otherwise), and `demote()`'s list/tuple length-
materialization clause gained `"array"` too -- an array is a real ArrayObj at every point a
list already is one, so every place the interpreter's OWN bookkeeping (not user-facing
behaviour) previously matched on `"list"`/`"tuple"` was audited for whether `"array"` needed
the same treatment; MG's PR 89 lesson (`set_bottom` on an unrecognized arity produced 9 wrong
TRUEs) is why every one of those omissions would have been a soundness bug, not just a
precision one -- `out.is_bot()` after a "some kind wasn't recognized" `bad = True` fallback
becomes `st.set_bottom()`, which is correct ONLY when the operation genuinely always raises;
`len(arr)`, iteration, and `arr *= n` do not, so those three specifically needed either an
explicit `"array"` clause (the first two) or an explicit `raise Unsupported` (the third,
mirroring list's own `"in-place list repetition"` refusal) rather than silently falling
through.

**Typecode semantics** (`interp.py::_ARRAY_INT_RANGES`, `_array_int_value`,
`_array_float_value`): only the typecodes whose C width is the same across every mainstream
platform are modelled --
`'b'`/`'B'`/`'h'`/`'H'`/`'i'`/`'I'`/`'q'`/`'Q'` (signed/unsigned 1/2/4/8-byte ints) and
`'f'`/`'d'` (IEEE-754 single/double). `'l'`/`'L'` ("long") are refused outright: CPython's OWN
array module gives them a genuinely platform-dependent itemsize (4 bytes on Windows' LLP64
`long`, 8 on 64-bit Linux/macOS) unlike every other typecode here, and this analyzer has no
way to know which platform a proof will be checked on. `'u'` (unicode char) is refused too: no
str/bytes element model exists to back it, and it does not appear in any known SV-Benchmarks
array use.

- **Integer typecodes**: a non-raising store's element is the MEET of what the value was
  already known to be and the typecode's own range -- sound (a real store that didn't raise
  really was in range), and gives a literal-range-checked store (`array('B', [0])`, or W49's
  raytrace `Canvas.plot`'s `max(0, min(255, ...))` clamp) an exact or tight post-store fact
  instead of falling back to the typecode's full range on every store.
- **`'f'`**: every finite value is rounded to the nearest representable `float32` on store.
  When the value is an EXACT known constant, the rounded value is computed with
  `struct.pack('<f', x)` / `struct.unpack('<f', ...)` -- the same IEEE-754 binary32
  conversion CPython's own `arraymodule.c` performs -- and stored as the new exact fact (so
  `array('f', [0.1])[0]` is tracked as `0.10000000149011612`, never `0.1`). Measured: CPython's
  `array` module SATURATES an out-of-range magnitude to `+-inf` rather than raising
  `OverflowError` (`array.array('f', [1e300])[0] == inf`, no exception) -- unlike
  `struct.pack('f', ...)` on the SAME input, which DOES raise. `struct`'s own `OverflowError`
  is therefore never surfaced as an array check; it would be real `struct` behaviour but not
  real `array` behaviour, and modelling it here as a check would have been a wrong claim in
  the OPPOSITE direction (claiming an exception that cannot really happen).
- **`'d'`**: a C double IS a Python float exactly -- no rounding, no overflow, ever.
- A non-exact (interval, or otherwise unresolved) float value keeps its existing bounds,
  falling back to the fully generic `FLOAT` when nothing narrower is known -- sound, if coarse.

**A pre-existing limit this does NOT change**: neither `list` nor `array` tracks an exact
per-element float value once it has passed through a container's `elem` summary (EVERY
element of a list or array is one JOINED `V`, not stored per index) -- `lst = [0.1]; lst[0]`
is already the fully generic `FLOAT` before any array is involved (confirmed:
`tests/test_array_module.py::test_f_rounding_is_never_wrongly_proven_exact`, which is the
required 'f'-rounding never-TRUE control -- it stays `UNKNOWN`, never a wrong `TRUE`, for
exactly this reason). A single-element container's `elem` IS exact end to end (no join
partner to blur it against), which is what lets `array('B', [0])`'s or
`array('i', [2147483647])`'s own construction-time exactness show up in an index read.

**Scope, matched to measured SV-Benchmarks use** (`bm_raytrace_nondet.py`,
`bm_scimark_nondet.py`, grepped directly, [internal ref]): `array.array(typecode, [0] * N)`
with a SYMBOLIC `N` (raytrace's `Canvas.bytes`), and `array(typecode, [0]) * n` (scimark's
own construction idiom, every one of its `array(...)` call sites) both resolve. Neither
benchmark's use needs a `bytes`/`str` initializer, slice assignment, `insert`/`pop`/`remove`/
`sort`/`clear`, in-place repetition, or a mixed-typecode operation -- all refuse (
`Unsupported`) rather than being guessed at. `array('d', [self.nextDouble() for i in
range(n)])` (scimark's `RandomVector`) and `[array('d', [0]) * w for y in range(h)]`
(scimark's `RandomMatrix`) are refused for an UNRELATED, pre-existing reason (a list
comprehension) before array support is ever consulted. [internal ref] lowers the comprehension (see
"Comprehensions: the exact lowering"), so the reason there is now whichever construct comes next.

Both benchmarks are still blocked from a full proof by OTHER, already-tracked gaps
independent of array: raytrace needs file writes ([internal ref]) and trip propagation; scimark needs
`sorted()` ([internal ref], since landed) and, for the non-unit-stride writes in its inner loops
(`x[i*2]`-shaped, distinct from the `x[i]`/`x[i+1]` shapes list's own segmentation already
proves), a segmented-list invariant extension that is not array-specific either. Array support
alone therefore moves neither benchmark's Real-world proof today; it removes array itself as
the NEXT blocker once those land (W49, [internal ref]).

Never-TRUE coverage: `tests/test_array_module.py` (unit-level: out-of-range index, integer
typecode overflow at every width's boundary, extend from the wrong element type, 'f' rounding
never wrongly proven exact, every refused method/operator/typecode).

### Two wrong TRUEs of the array and the slice store ([internal ref], [internal ref])

Found on sv_2027 `4e129af` while auditing every consumer of the heap kinds for the places that treat an unknown kind as
"bad, but go on" (a recorded `TypeError` and no further effect on the state):

- **[internal ref], `Analyzer.slice_store`.** The source of `xs[a:b] = src` was handled for a list or a tuple, and every other kind fell into
  the branch for a source that is NOT iterable: the `TypeError` check was recorded unproven and the path went on with the list's OLD
  elements. A range, a dict (its keys), a dict view of keys or values and an array are iterable, and their elements are in the list
  afterwards, so `xs = [1]; xs[0:1] = range(0, 1); c = 10 // xs[0]` was proven safe and raises `ZeroDivisionError`. Now an array is
  read like a list, and a range, a dict and a dict view go through `elem_summary` (the source's length is not used, as before for a
  source without a known length); `dict.items()` has no element summary and refuses. The `TypeError` check is proven only for the
  sources that are iterable.
- **[internal ref], `Analyzer._call1`.** A bound builtin method (`m = a.append`) was always run by `list_method`. For an array that skips the
  typecode of an element write (`_array_store_value`: the float32 rounding of an `'f'` array, the range of an `'i'` or `'B'`
  one, the `TypeError` of a str), so `m = a.append; m(0.1); assert a[0] == 0.1` was proven on an `'f'` array and is false (CPython
  stores 0.10000000149011612). The call is now dispatched by the kind of the receiver (an array to `array_method`, a list to
  `list_method`); a receiver that may be either, or another kind, refuses by name.

`tests/test_absint_slice_source.py` and `tests/test_absint_bound_array.py`: each regression fails on `4e129af` (27 of 40 tests) and
passes here; a native differential over every list x slice x source x read (12,600 programs, a sample of 480 per run) and every array x
call x read (624), a TRUE for a class a run raises fails the test; the negative control undoes each fix and the differential catches
it. `scripts/absint_fuzz.py` draws slice stores from every source and the array methods, direct and bound (`--slice-bias`).

## An unhandled kind is a refusal, never an infeasible path ([internal ref])

The rule behind [internal ref] and [internal ref], and the class they belong to. An operation that does not know a value's kind used to treat it as
"bad" (a recorded `TypeError` that is not proven) and, when no alternative produced a value, `set_bottom` the path: every statement
after it was unchecked, so a program that raises there was proven TRUE. The rule now: an unmodelled kind is `Unsupported` (a named
refusal, the proof is UNKNOWN and says why); a path is bottom only where the concrete semantics proves it impossible, and then the
exception that makes it so is itself a recorded check (`KeyError` of `remove`, `IndexError` of `choice` on an empty list, the `TypeError`
of `2.5 * [1]`).

**The sweep.** Every `bad = True` / `set_bottom` / out-stays-`BOT` site of `solve/absint/` was read against every kind absint has (int,
bool, float, str, None, list, tuple, array, range, dict, dict view, iterator, instance, class, function, bound method, builtin method,
static object, `re.Pattern`, `StringIO`, output file) and the sites found wrong were changed:

| site (`interp.py::`) | kind | before | now |
|---|---|---|---|
| `Analyzer.unpack` | array | `bad`, no alternative: path ends (everything after the unpack unchecked) | exact: the elements of the array |
| `Analyzer.m_random_choice` | array | the same, and the `IndexError` of an empty array never recorded | exact: the length check and the elements |
| `Analyzer.subscript_load` (str) | a non-int index, a non-int slice bound | the index type was never checked | `TypeError` check; no value comes of it |
| `Analyzer.subscript_load` (dict) | a slice (hashable since 3.12) | `TypeError`, and no `KeyError` recorded | `KeyError` check (a slice is never a key of a modelled dict) |
| `Analyzer._seq_binop`, `Analyzer.binop` | float * str/list/tuple/array, either order | no `TypeError`; no value either: the path ended | `TypeError` check |
| `Analyzer.check_in` | a static object (`len`, `int`, a module, an exception class) `in` a str | not a `TypeError` | `TypeError` check |
| `Analyzer.attr_store` | list, tuple, array, dict, dict view, range, iterator, pattern, bound method | `bad`, path ends, `TypeError`/`AttributeError` check | `AttributeError` check (unchanged for these; a `StringIO` or file handle, which takes any attribute, was the same and is now refused) |
| `Analyzer.call_static` | an exception INSTANCE (`ValueError('x')()`) | the same value as the class: a call constructed another one | not callable: `TypeError` check (the instance is the tag `("excinst", name)`) |
| `Analyzer.static_attr` | a name no module table models (`math.foo`) | the value was unknown, its existence never checked | `AttributeError` check |
| `Analyzer.gc` | a dict view | its dict was collected while the view lived: a `KeyError` out of `prove` | the dicts of a live view are reachable |
| `Analyzer.check_iterable`, `Analyzer.unpack`, `Analyzer.slice_store` | `StringIO` | iterable, element summary empty: the `for` body was an infeasible path | refused |

**Reviewed and left, each with its reason** (the exception that ends the path is a recorded check or not a property class, or the value
decides the branch): `s_Return`/`s_Break`/`s_Continue`/`s_Raise` (control flow); `load_name` (`UnboundLocalError`, `NameError`: not
property classes); `_merge_branches`; `truth_split`/`compare_split`/`_refine_is`/`_prune_float_compare` (a branch the value
decides); `binary_special` (no applicable special method: a recorded `TypeError`, exact for instances); `unop`, `binop`, `b_abs`,
`b_len`, `b_range`, `t_int`, `t_float`, `c_ord`, `c_chr`, `_math_call` (a kind with no numeric part is a certain `TypeError`, recorded
before the path ends); `call`/`_call1` (a callee that is no callable kind), `bind_args`, `call_keywords`, `instantiate` (arity,
duplicate keyword: certain `TypeError`s, recorded); `call_func` (a function that never returns); `list_method`, `array_method`,
`dict_method`, `m_array_array` (arity checks of functions that take positional arguments only, recorded; the `os` and `re` models raise `Unsupported` and never end a path); `m_random_*` were listed here and were WRONG, see "The `random` module"; `range` with step 0 (`ValueError`);
`m__sv_verifier_nondet_float_range` (every draw rejected: no valid run); `reducers.py` (`min`/`max`/`sum`/`any`/`all`: no iterable
alternative is a recorded `TypeError`; an empty iterable without a default is a `ValueError`). The symbolic
executor (`sv/_symexec.py`) produces vectors that are replayed natively before they are believed and proves nothing, so a path it
cannot follow costs a witness, never a TRUE.

**What finds the rest: the kind matrix** (`tests/test_absint_kind_matrix.py`, `scripts/absint_kind_matrix.py`). For every value of every kind
absint has, and the kinds it must refuse (set, frozenset, bytes, bytearray, a generator, a complex, user classes with every dunder
absint refuses), and every consumer (unpacking, iteration, `len`, truth, subscripts and stores, every operator, the builtins and
`math`/`random`/`re`/`io` functions with the value in every argument position, methods and attributes, calls, `with`, formatting, and
blocks whose body holds a sentinel), the program `<value>; <consumer>; reached = True; 1 // 0` runs natively. If the consumer
succeeds natively the sentinel is reached and absint must not say TRUE for `ZeroDivisionError`; if it raises a property class
natively, absint must not say TRUE for that class. A TRUE for a path the consumer made infeasible is exactly the bug, and the matrix
found the eleven above: over its 193,240 programs it reports 400 wrong TRUEs (100 consumer shapes) and 1,029 crashes on `4e129af`, and none of either on the fixed tree (193,126 checked, 114 skipped: 111 are a class whose `__getitem__` never raises `IndexError`, so the native loop never ends, and 3 are `2 ** (10 ** 30)`). A consumer with its sentinel INSIDE (a loop body, a branch) has no sentinel after it, so a body wrongly made infeasible is not hidden by an unproven check on the continuation. Each native run is a
forked child under `RLIMIT_AS`, `RLIMIT_CPU` and a wall alarm, and a program that hits a cap is SKIPPED and listed, never passed; the
runner takes at most 11 workers of 4 GB (the matrix once reached 108 GB when it ran `2 ** (10 ** 30)` in a pool worker without caps).
`tests/test_absint_kind_refusals.py` holds one regression test per changed site.

## The `random` module ([internal ref], [internal ref])

Two wrong TRUEs of sv_2027 `8fc10c8`, in `m_random_*` (now `rand.py::RandomMixin`, mixed into `interp.py::Analyzer`).

**[internal ref]: a call binds its arguments as CPython binds a Python function's.** The module functions are bound methods of one
`random.Random` instance (`Random.randint(self, a, b)`, `Random.uniform(self, a, b)`, `Random.choice(self, seq)`,
`Random.seed(self, a=None, version=2)`), so every parameter can be given by NAME, and `seed` takes a second positional `version`. The
models counted arguments with `Analyzer._arity`, which treats any keyword as an arity error: it recorded an unproven `TypeError` and the
model then ended the path, so everything after `random.randint(a=0, b=5)` was unchecked (`c = 1 // 0` after it was TRUE for
`ZeroDivisionError`; natively it raises). Same for `uniform(a=0, b=1)`, `choice(seq=[1])`, `seed(a=1)`, `seed(1, 2)`,
`seed(1, version=2)`, and `_sv_verifier.nondet_float_range(a=0.0, b=1.0)` (`m__sv_verifier_nondet_float_range`, the same pattern).
`Analyzer._bind_call(st, node, args, kwargs, what, params, required)` is CPython's binding: by position or by name, and a call that does
not bind (too many positional arguments, an unknown keyword, a parameter given twice, a required one missing) is a `TypeError` that is
certain whatever the values are: recorded as a `call-args` check and the path ends. `random()` is a C method with no parameters; the
shapes are the table `rand.py::_SHAPES`. A user object among the arguments (its `__add__`, `__sub__`, `__eq__` ... would run: `randint(0, b)` is
`randrange(0, b + 1)`, `seed` compares `version == 1`) and an unknown value are named refusals (`random.<f> of a user object`, `of an unknown
value`), never a `TypeError` that ends the path; `randint(a, b)` with `a > b` on every run ends the path (`ValueError`, not a property class).

**The siblings (every other `_arity` / `kwargs` site).** `Analyzer._arity` is right where the function takes positional arguments only
(`len`, `abs`, `ord`, `chr`, `iter`, `range`, `list`, `tuple`, `int(x)`, `float(x)`, `isinstance`, `dict.get`, `list.append/insert/pop/extend`,
`array(...)`, `array.append/extend`: a keyword is natively a `TypeError`). The `re`, `os`, `io.StringIO` and output-file models refuse an
unmodelled shape (`Unsupported`), `int` with a base and `sorted`/`min`/`max`/`sum` with keywords refuse or are exact, and `math` binds through
its own table (`src/tharkun/solve/contracts/adapters/math_model.py`). `tests/test_absint_call_shapes.py` pins that: 183 valid shapes (a keyword the function takes,
an optional argument) and 97 shapes that raise `TypeError` natively, each a fixed one-line program run natively and by absint. The only
sites found wrong were the five `random` models and `nondet_float_range`.

**[internal ref]: `uniform(a, b)` is `a + (b - a) * random()`.** It was modelled as the interval of its bounds. CPython computes `a + (b - a) *
self.random()` (`Lib/random.py::Random.uniform`), so `uniform(-1e308, 1e308)` is `inf` on every draw (`b - a` overflows), a call with an infinite
bound is `nan` (`inf - inf`, or `inf * 0.0`), and an int too large for a float is an `OverflowError` (not a property class: the path
ends). `assert y <= 1e308` after `y = random.uniform(-1e308, 1e308)` was TRUE and fails natively (MAIN's probe `g15_uniform_ok_hidden.py`).
`m_random_uniform` now evaluates the formula with the analyser's own operators over the draw `[0.0, 1.0]` (`random()`'s closed hull):
`binop(Sub)`, `binop(Mult)`, `binop(Add)`, which carry the outward-rounded transfer functions of `solve/values.py` (`inf`, `nan`, signed
zeros, int-to-float conversion). Nothing else is special: `uniform(0, 1)` is `[0.0, 1.0]`, `uniform(2, 1)` is `[1.0, 2.0]`, and a program
that stays inside a range still proves its bound.

**`randrange` and `getrandbits` (the hull of the draw).** `bm_go_nondet` hashes its board with `[random.randrange(sys.maxsize) ...]`; the
contract-driven call had no value (and left every class unproven), so the next statement refused. `Analyzer.m_random_randrange` is
the hull of `Lib/random.py::Random.randrange` (3.14: `randrange(start, stop=None, step=1)`) over the bounds of the arguments: `start`
alone is `[0, start - 1]` (a `ValueError` unless `start >= 1`); `start, stop` with a positive step is `[start, stop - 1]`, with a
negative step `[stop + 1, start]`, the two joined when the step may be either, and no value (the path ends, a `ValueError`, not a
property class) when no pair of values makes the range non-empty or the step is `0`. A float, a str, `None`, an object, and a `step` given
with `stop=None` (unless it is the very int `1`: CPython tests `step is not _ONE`, so `True` and `1.0` are the `TypeError`) are a recorded
`TypeError`; the parameters bind by name (`start=`, `stop=`, `step=`). A bool is its int. `getrandbits(k)` is `[0, 2**k - 1]` (`0` for
`k == 0`; a negative `k` is a `ValueError`; `k` is positional only, a keyword is the `TypeError`; no upper bound beyond 4096 bits).
`tests/test_absint_randrange.py` holds each hull to the native draws (400 draws all inside the hull, and `assert x != v` is never
proven for a value drawn) and one value past either end to be proven out. Negative controls (each fix undone, the tests fail): the hull including `stop` (11 cases), negative steps ignored (4), a step with `stop=None` not a
`TypeError` (4), an empty range keeping the path (16), `randrange(n)` including `n` (8), `getrandbits(k)` including `2**k` (5), an empty positive range keeping the path (2),
`getrandbits(-1)` keeping the path (1). On a tree with the set family ([internal ref]) `bm_go_nondet` now stops at
`unsupported: recursion (line 94)`, past `set()` and the zobrist table.

**The rest of the family.** Modelled (all but `getrandbits`, which has no contract, read through their rules: "Builtins the analyzer modelled by hand, read through their rules"): `random`, `randint`, `randrange`, `getrandbits`, `uniform`, `choice`, `seed`, `getstate`. Contract-driven, with no
value and every class unproven (`_contract_call`: a sound havoc, so a program that calls one is never TRUE for a property class): `sample`,
`shuffle`. Refused by name (`unsupported: call of an unknown value: random.<f>`): `triangular`, `gauss`, `normalvariate`, `lognormvariate`,
`expovariate`, `betavariate`, `gammavariate`, `vonmisesvariate`, `paretovariate`, `weibullvariate`, `binomialvariate`, `randbytes`, `choices`,
`setstate`, and the classes `Random` and `SystemRandom`. The float draws can leave the interval of their arguments (`triangular(-1e308, 1e308)` is `inf`, like `uniform`) or raise a property class
(`expovariate(0.0)`, `paretovariate(0.0)` and `weibullvariate(1.0, 0.0)` are a `ZeroDivisionError`); none has a model, and `tests/test_absint_random.py`
pins both facts (the native hazard, and the refusal). `random.choice(range(n))` and `choice(dict)` are refused too (`choice({0: 'a'})` is `'a'` and
`choice({1: 2})` is a `KeyError`: not a sequence).

**`getstate()`.** The model returns CPython 3.14's version-3 tuple and a 625-entry state tuple:
624 unsigned 32-bit words and an index in `1..624`. Seeding sets index 624; each draw advances it
before returning. The exact state words, index and Gaussian cache remain unknown. `setstate` can
install index zero and stays refused, as do module mutations and unmodelled callbacks.
`tests/test_absint_random.py` covers the structural invariant, repeated draws and state reads, and these refusals.

**Tests.** `tests/test_absint_random.py`: every valid shape continues the path; every invalid one is a certain, recorded
`TypeError`; by-name binding; `uniform` overflow, `nan` and every value the formula takes for the draws `0.0`, `0.5`, `1 - 2**-53`
(the native control is a `random.Random` whose `random()` is that draw: its own `uniform`); the refusals. Negative controls (each fix
undone, the tests fail): the old `_arity` for `random.*` (16 + 4 cases fail), `uniform` as the hull (6), `nondet_float_range` on `_arity`
(3), `randint(a > b)` keeping the path (1), user objects not refused (1), keywords bound by position (4).

## Contract-driven builtins ([internal ref] batch 4): `ord`, `chr`, `tuple`

A builtin with no `b_`/`t_` transfer function used to be `V(top=True)`, so its first call aborted
the whole proof ("call of an unknown value": `bm_richards_nondet` line 337 is `A = ord('A')`,
`bm_unpack_sequence_nondet` line 421 is `tuple(range(10))`). `interp.py::Analyzer.builtin_name` now
tags a builtin that has a contract-driven model (`c_<name>`) AND a TESTED contract
(`solve/contracts/adapters/absint.py::contract_for_builtin`) as `("cbuiltin", name)`, and
`Analyzer._contract_builtin_call` runs it:

- the contract (`solve/contracts/rules_builtins_more.py`) is the authority on WHICH of the seven property
  classes the call can raise: all seven are recorded at the call (`<name>:<class>` ops), each as the
  contract states it -- proven only for a TESTED `never`, otherwise unproven;
- the model (`c_ord`, `c_chr`, `c_tuple`) returns the abstract result and the classes it has positively
  argued cannot occur for THESE arguments; only those become proven. A class it does not mention stays as
  unproven as the contract left it, so a forgetful model costs precision, never soundness;
- a model refuses (`Unsupported`) rather than guess: an object argument to `ord`/`chr`, an unknown value,
  several possible iterables to `tuple`;
- withdrawing or un-testing the contract returns the call to `V(top=True)` and the abort.

`c_ord`: TypeError is proven only when every possible argument is a literal str of length 1, and the
result is the exact range of those code points (`ord("A")` is the constant 65). `c_chr`: TypeError is
proven only for int/bool, the six classes the contract admits only through `__index__` are proven
because no object argument reaches it, the result is `ANYSTR`. `c_tuple` shares `Analyzer._materialize`
with `t_list` (length and element summary of the source at the call; an exact tuple comes back as
itself); the six classes the contract admits only through the argument's hooks hold because a builtin
iterable runs none and an instance's `__iter__` body is ANALYSED by `iterable()`.

`zip`, `enumerate`, `reversed` and `type` have contracts but no absint model (`set` and `frozenset` have one now: "Sets",
below): the first three return lazy iterators over a LIVE container (a snapshot of the element summary would go stale after
an `append` between `reversed(xs)` and its consumption; modelling them needs a cursor-aware
`IteratorObj`). Ruled, not built.

## Builtins the analyzer modelled by hand, read through their rules ([internal ref] step 2)

`abs`, `len`, `range`, `int`, `float`, `list`, `print` and `isinstance` had a transfer function (`Analyzer.b_<name>` /
`t_<name>`) written before `solve/contracts` existed: the call never read the contract of the callable, so two sources of truth
described what it can raise (the engine-by-contract audit's `M` cell, `docs/contracts/REPORT.md` part B). They now run the way
`ord`/`chr`/`tuple` do, through the rule of the callable:

```
call_static -> b_<name> / t_<name> -> visit_rule(<name>) -> Rule.accept(AbsintVisitor)
            -> AbsintVisitor.visit_<kind> -> Analyzer._contract_builtin_call -> c_<name>
            -> Analyzer._modelled(<unchanged model>)  -> _contract_checks(<seven classes>)
```

The models are the old ones, renamed `_<name>_model` (`_abs_model`, `_len_model`, ...): the abstract result is what it was. What
changed is who states the property classes. `_contract_builtin_call` records all seven at the call (`<name>:<class>` ops), each as the
CONTRACT states it (proven only for a tested `never`), and `Analyzer._modelled` upgrades a class only on an argument the model made,
chosen by what the contract says the class is:

- a class the contract admits only through a user hook (`dunder_propagation_claims`: `__abs__`, `__len__`, `__index__`, `__iter__`, ...):
  the hook argument. For `abs`/`int`/`float`/`range`/`isinstance` it is structural: a class defining `__abs__`, `__int__`, `__float__`,
  `__index__` or `__instancecheck__` is refused where it is defined (`interp.py::UNSUPPORTED_DUNDERS`), so no instance the analysis can hold
  has one. For `len` the instance's `__len__` body is ANALYSED (`special`); for `list` the instance's `__iter__` body is analysed
  (`iterable()`) and its `__len__` probe is `protocol.length_hint`'s; for `print` a custom `__str__`/`__repr__`/`__format__` is refused
  (`check_formattable`);
- a class that is the call's own logic (`adapters/common.py::native_raise_condition`: `TypeError` for arity and argument kind; `print`'s
  `AttributeError` for a `file` without `write`): the model's own checks of the call (`_arity` records one for every call), or the argument
  the model names (`print`: stdout is the process's own because a store to a module attribute, `setattr` and `del` of one are each
  refused, and `file=` is a modelled `io.StringIO` or refused);
- an untested or unknown claim is never upgraded.

The methods of the builtin heap kinds the analyzer models by hand follow the same road ([internal ref] step 2, second PR): `list.append`, `extend`,
`insert`, `pop`, `reverse`, `dict.get`, `keys`, `values`, `items` and `io.StringIO.write`, `read`, `getvalue`, `seek`.
`Analyzer.list_method` / `dict_method` / `stringio_method` look the qualname up in `interp.py::_METHOD_HOOKS`; a method that is there goes
through `visit_method_rule` -> `AbsintVisitor.visit_<kind>` (a `MethodCtx`: state, receiver addresses, name, values) ->
`Analyzer._contract_method` -> the unchanged model (`_list_method_model`, `_dict_method_model`, `_stringio_method_model`) under `_modelled`.
`_METHOD_HOOKS` is the table of "why no user hook runs" per method, `_METHOD_NATIVE` the call's own classes a model argues without
recording a check (`StringIO.read` and `seek`: an accepted size or position is int- or bool-shaped, anything else is refused). Two
points are the contract's, not the model's: `list.pop`'s `IndexError` is the call's own logic, so it is proven only by the model's own
index check (`pop` of an empty list stays unproven); and a class the contract says `never` is VETOED when the model recorded a failing
check of the call (`proven[exc] = None`), because the `never` of `list.append`, `list.reverse`, `dict.keys`, `dict.values` and
`StringIO.getvalue` holds for the shape its tests ran: `xs.append()` raises `TypeError` and its `list.append:TypeError` op is unproven.
The read-only and shrinking list methods are modelled the same way (third PR of step 2): `copy` (a NEW list with the receiver's length and
element summary, one receiver only, like `list(x)`), `clear` (length 0, no elements), `count` (an int in `[0, len]`), `index` (an int in
`[0, len - 1]`; absent is a `ValueError`, which is no property class, so the path ends: a list that cannot hold the value ends every path),
`remove` (the length drops by one on the path that did not raise), `sort` (the elements stay, the order is unknown, segmentation collapses;
only `_elem_sortable` elements, `key=None` and a bool `reverse`), and `tuple.count` / `tuple.index` (`Analyzer.tuple_method`). An operand or
element whose `==` runs user code is refused (`custom_comparison`). A method without a rule visit (`dict.pop`, `dict.update`, ... ) is served by
its model alone, as before.

The functions of the `random` module the analyzer models by hand (`rand.py::RandomMixin`: `random`, `randint`, `randrange`, `choice`,
`uniform`, `seed`; fourth PR of step 2) follow the same road with a function of a module as the callable: `call_static` sends `random.<f>` to
`visit_rule` when `interp.py::_MODULE_HOOKS` names it, `AbsintVisitor.visit_random_<f>` runs `Analyzer._contract_module_model`, which gates on
a TESTED contract (else the named refusal `random.<f> has no tested contract`) and runs the unchanged `m_random_<f>` under `_modelled`; the
seven classes are the ops `random.<f>:<class>`. `_MODULE_HOOKS` says why no user hook runs: `RandomMixin._random_args` refuses a user object
by name (`instance_value`), so no `__index__`, `__len__`, `__getitem__` or arithmetic dunder of a user class can run. `choice`'s `IndexError`
(the empty sequence) is the call's own logic and is proven only by the model's own length check; a call that certainly raises `TypeError`
first (it does not bind, or the argument is no sequence) records that check as held, so a program doomed to a `TypeError` is still TRUE for
`IndexError`. `random.getrandbits` has no contract (nothing to read) and keeps its model; `sample` and `shuffle` have no model and stay the
generic contract-driven call. `tests/contracts/test_absint_random_rules.py`.

The implicit length of a truth value (`truth_split`) and the length probe of a consumer (`_probe_length`) are not calls the program
writes: they run `_len_model` directly, so no `len:<class>` op appears where the program never calls `len`
(`tests/contracts/test_absint_builtin_rules.py::test_the_length_of_a_truth_value_is_still_the_model_not_the_call`).

Withdrawing or un-testing a contract is no longer a silent fall back to the hand model: the call is the named refusal `<name>() has no
tested contract` (`test_withdrawing_the_contract_withdraws_the_model`). Verdicts are unchanged: 633 pairs of the Real-world A/B tiers, the
24-task Real-world set and Solve-only (`docs/contracts/evidence/1133-step2/README.md`) give the same verdict and reason before and after.

## Output files ([internal ref]): `open(..., "w")`, `with`, and the write side of the handle

`bm_raytrace_nondet` (line 213) and `bm_chaos_nondet` (line 141) write a file and never read it back. The
environment gate (`solve/environment.py::proof_blocker`) lets a write-only `open()` through
(`docs/SV_IO_MODEL.md`); `interp.py` then modelled neither `open` nor `with`, so the first one aborted the proof.
Now:

- `Analyzer.c_open` (the shape contract `open.output`, run by `_contract_builtin_call`; the gate is
  `contract_for_builtin`, which maps `open` to it through `src/tharkun/solve/contracts/adapters/common.py::BUILTIN_SHAPE_CONTRACTS`)
  allocates an `OutFileObj` (`heap.py`; one site = one `open` call = one constant mode = one kind; it carries
  only `binary`). Only the shape `environment.open_effect` calls OUTPUT is modelled; a read, an update mode,
  `buffering`, `closefd`, `opener`, a mode that is not a constant, a binary mode with a text option and an
  option of unknown type are `Unsupported`. The path must be a definite `str`: an int or bool may be a file
  DESCRIPTOR and an object may run `__fspath__`, so both refuse; a `None`/float part is a recorded, unproven
  `TypeError`. With a definite `str` path no user code runs, so the six classes the contract admits only
  through `__fspath__` are proven.
- `Analyzer.s_With` models the context manager for output handles ONLY: every item must evaluate to
  `OutFileObj`s, `__enter__` returns the handle, `__exit__` closes it and returns `None`. An exception of the
  body therefore propagates unchanged, and a `return`/`break`/`continue` through the block loses nothing.
  Any other context manager is `Unsupported`, as before. The target must be a plain name.
- `Analyzer.outfile_method` (`_outfile_call`) models `write`, `writelines`, `flush`, `close`, `__enter__`,
  `__exit__`. Every raise claim is the contract's (`src/tharkun/solve/contracts/rules_fileio.py`, read through `src/tharkun/solve/contracts/adapters/absint.py::
  output_handle_contract`); the model upgrades only what it knows of THESE arguments:
  - text `write(x)`: `TypeError` is proven iff `x` is definitely a `str` (nothing else runs: the contract says
    no callback);
  - binary `write(x)`: proven iff `x` is definitely an `array` (the one buffer provider with an abstract
    value; `bytes`, `bytearray` and `memoryview` have none yet, and an instance may define `__buffer__`, so
    objects refuse);
  - `writelines(xs)`: proven iff `xs` is a list or tuple whose every element is, by the same rule;
  - `flush`, `close`, `__enter__`, `__exit__`: every class `never`.
  A call shape the contract does not cover (a keyword, another arity), `tell`, `seek`, `read`, an attribute
  (`name`, `closed`, `buffer`) and `print(file=fp)` stay `Unsupported`.
- Whether the file is closed is NOT tracked, and need not be: what that changes (`ValueError`) and what the
  environment can do (`OSError`) are not property classes, so a path that raises them ends and the paths that
  continue are the ones modelled. A wrong-typed argument is a `TypeError` even on a closed file (the argument
  is parsed first), which is what the claims say.
- What was written never comes back (a handle opened for output has no read), so no content is tracked.

Measured on the Real-world `_nondet` tasks (48 pairs): `bm_raytrace_nondet` x4 and `bm_chaos_nondet` x4 are past
the environment gate and no verdict changes; the chains behind them are in `docs/SV_IO_MODEL.md`. The output
open itself is not what blocks them any more: `chaos` stops at `_sv_verifier.nondet_float_range` (line 245), then
`min` over a list (156), then `%`-formatting in `write_ppm` (`'%i %i\n%i\n' % (w, h, maxval)`); `raytrace` at
`math.tan` (252, C2's math module), then `try`/`finally` in `rayColour` (274), then `str.encode` and
`array.tobytes()` in `write_ppm` (bytes have no abstract value).

### Allocation sites and the call string

`raytrace` first stopped at "allocation would invalidate an in-flight heap reference" in
`Ray(eye.point, eye.vector + xcomp + ycomp)`. Diagnosis: a model gap, not a limit. `a + b + c` with a user
`__add__` is two calls whose `BinOp` nodes both START at `a`; the call string held only `(line, col)` of the
start, so both calls named one allocation site inside `__add__`, and the second allocation found the first
result still held by the outer operation (`Analyzer.demote`'s `borrowed` guard) and refused. A call is now
named by its whole node (`callstring` entries are `(line, col, end_line, end_col)`); the site records'
`context` still lists `[line, col]` pairs. `tests/test_absint_callstring.py`.

### In-flight references: holders, scopes, and a call's operands ([internal ref])

A transfer function holds values in Python locals, and `State.redirect` rewrites only the heap, the frames and the
pins when an allocation turns a site's recent object into the site's summary (`Analyzer.demote`). So the interpreter
refuses such an allocation ("allocation would invalidate an in-flight heap reference") while any local may hold the
object. `Analyzer.borrow` registers a holder for every recent object a value names, in the TOP scope of
`Analyzer.borrowed`; `demote` refuses when any scope holds the site.

The scopes follow the lifetime of the locals, so a value is held exactly while the code that produced it for its
consumer is still running:

- `eval` runs `m(e)` in a scope of its own and pops it on return: the operands an expression evaluated are dead
  when the expression has returned, and only its RESULT is registered, in the enclosing scope. (They used to stay
  registered until the end of the statement, which refused `return GVector(self.x * l, ...)` when `self` was the
  object the previous call made at that very site.)
- `call_func` pushes a scope for the callee's frame, and `gc` clears the top scope at each statement boundary.
- A call's operands (`e_Call` passes `operands=True` down `call_method`, `call`, `_call1` and `instantiate` to
  `call_func`) are in the callee's frame once it is bound, which `State.redirect` rewrites, and no Python local reads
  them after the call. For the callee's duration their holders (the current scope: the operands `e_Call`
  evaluated and the ones `bind_args` added) are not held, so the callee may allocate at their sites: `d[i] =
  d[i].combine(d[i + 1], ...)` makes a new object of the receiver's class at one site. The holders come back when the
  callee returns. `instantiate` keeps the holder of the instance it returns, which it reads after `__init__`.

What stays refused: an earlier operand of an enclosing expression that is still waiting while a later one
allocates at its site. A site is a node plus the call string, so that happens only when one call node runs again
(a loop): `lst = [p, mk(i)]; p = lst[1]` holds last iteration's `mk` object in the list builder while this
iteration's `mk` allocates at the same site. `tests/test_absint_operand_release.py` runs that shape and a
differential over a class whose methods build their own class: a TRUE the native run contradicts fails.

## Fixpoint reuse (A4)

*Built. Owner ruling 2026-10-05 ([internal ref], [internal ref]): the first build step of the loop design under
"Ruled, not built". Modelled on Mopsa's `LastFixpointCtx` (`loops.ml`), with the owner's tighter rule: equal entry
states only.*

An inner loop is solved again and again while an outer loop's fixpoint iterates (once per iteration, once more for the
decreasing step, once per relational candidate). `interp.py::Analyzer._fixpoint` is a cache in front of
`Analyzer._solve_fixpoint` (the iteration under "Loops"): when the same loop is entered again from an equal state, the
head of the earlier solve is handed back.

- **The key** is what a solve is a function of: the loop (its AST node), the call string, and the entry state, compared
  by `state.py::State.equiv`. `equiv` is "semantically equal": each state is included in the other (`State.leq` both
  ways) AND everything `leq` does not look at agrees: the frame stack (same frames, same variables), the pins, and the
  proof text (`V.why`) of every value, because a site record's fact can quote it. A bottom state equals nothing.
  The ruling named a fourth part, the float landmark set (`lemmas.py::landmark_key`): the landmarks are module state
  that GROWS during an analysis and steers float widening, so the same state widens differently after a lemma has added
  one. Reading the solve for what else it depends on found the analyzer's other scratch, and that is in the key too: the
  allocation epoch and the borrowed references of every active call and expression scope (an equal entry with a different borrowed set can
  raise "allocation would invalidate an in-flight heap reference"), and what the analysis has learned about globals and
  extern files (`Analyzer._reuse_history`; both only grow, so equal sizes mean equal contents).
- **Equal, not merely included.** The cached head is a sound invariant for any entry the cached entry includes, but it
  would be coarser than a fresh solve from the smaller entry. With equality the head is the one a fresh solve gives.
- **Only while checks are not recorded** (`Analyzer.record` is False: inside an outer loop's fixpoint iterations). The
  recording pass always solves its own loops, and its final pass over the body records exactly as before. Measured on a
  scratch variant that also reused in the recording pass: 1,052 hits instead of 1,048 on the absint fixtures, so the rule
  costs nothing.
- **Never inside a length probe** (`Analyzer._probe_length`, `Analyzer._probe_scopes` non-empty): a `raise` reached there is
  noted in the probe's scope, and a skipped solve would not note it. A bottom entry is never stored or served. A solve that learned
  something (`_reuse_history` changed) is not stored: the next run starts from different knowledge.
- **A hit does what the skipped solve did to the analyzer**, not only the head: it advances the temporary counter
  (`Analyzer.tmp`: temporaries are named `$depth_n` and a fact can quote one), restores the allocation epoch and the borrowed
  set the solve left, and merges the landmarks it added. Every later step sees what it would have seen. The head is
  returned as a copy.
- **Bounded:** `FIX_REUSE_WAYS` (8) runs per key, the oldest first.

### Measured (2026-10-05, tharkun sv_2027 72f695a against the branch, `PYTHONHASHSEED=0`)

The gate is that every site record and verdict is byte-identical to the pre-change output, with less absint CPU.
`absint.prove` ran over each corpus (all four properties, 60 s budget, no environment gate) and everything it reports
was compared (verdict, component, sites, reason, lemmas, checks, records, invariants):

| corpus | results | identical | fixpoints solved / requested | statements |
|---|---|---|---|---|
| Real-world 80 pairs | 80 | 80 | 732 / 1,110 (was 1,254) | 55,470 (was 67,668) |
| Solve-only 65 pairs | 65 | 65 | 19 / 19 (no nested loop) | 801 |
| eregion toys (542 sources x 4 properties) | 2,136 | 2,136 | 0 / 0 (no loop reaches absint) | 16,040 |
| absint fixtures (`tests/fixtures/absint`) | 116 | 116 | 1,644 / 2,692 (was 3,076) | 54,588 (was 87,884) |
| 8,000 random programs (`scripts/absint_fuzz.py` generator) x 4 | 32,000 | 32,000 | 53,740 / 56,936 (was 56,936) | 1,889,298 (was 1,941,790) |

CPU: the 12 Real-world results that hit (nbody and fannkuch, deep loop nests) take 7.9 s instead of 13.2 s, 40 % less
(statements -47 %). Real-world absint CPU in total is dominated by `bm_unpack_sequence` (about 35 s a pair, 2 loops, no
reuse), so the total does not move beyond the noise of a loaded host (236 s against 240 s); the statements executed fall
18 %. The random programs: 4.3 % less CPU, 3,196 hits.

**Scimark** (the `*args` dispatch rewritten to explicit calls, because absint refuses `*args` first): absint spends its
whole 60 s budget either way (25,300 statements, 170 loop fixpoints, none re-entered from an equal state) and stops at "analysis time budget exhausted". With a 900 s budget it runs on to its next
refusal, `generator function` at `a.indexes()` (line 108), in 185 s of CPU before and 119 s with reuse (63,113 statements
before, 40,103 after; 483 fixpoints solved before, 306 of 321 requested after). So fixpoint reuse shortens the way to that refusal by a third
but does not make scimark reachable inside the production budget (10 % of the remaining budget, at most 60 s); that
needs a cheaper iteration (the rest of the loop design) and generator support. Where the time goes there: 40 s of a 40 s
profile sit inside the outermost `for bench in benchmarks` fixpoint at about 2.4 ms a statement, so the cost is in the
operations on large states, not in solving the same loop twice.

**Hash order.** The differential has to fix `PYTHONHASHSEED`: the order of the facts a check lists depends on it (a check
that sees several heap addresses iterates a set of them), so two runs of the SAME tree differ in that order (21 of 372
results on the programs where it showed; the same facts, in another order). Found by this differential; not changed here.

## isinstance: user classes, containers and tuples of classes ([internal ref], [internal ref])

`interp.py::Analyzer.b_isinstance` decides `isinstance(x, classinfo)` per alternative of `x` and per alternative of the
classinfo. The classinfo is a builtin class (`str`, `int`, `float`, `bool`, `list`, `tuple`, `object`), a user class, or a
tuple of those (a tuple is ONE value naming all its classes: `_classinfo`); an unknown value, a scalar, an exception class
and a tuple that is a summary refuse, because they may be a TypeError or a class whose instance check this model does not
know. A classinfo that may be one of several values is decided for each (`k = int if c else A`) and the truth values are joined.

- A scalar is an exact instance of its kind (`bool` is also an `int`), of `object`, and of nothing else.
- An object is an instance of its class and of the classes along its single-inheritance chain (`s_ClassDef` refuses
  multiple inheritance, a metaclass, a decorator and every builtin base except `object` and the exceptions, so the chain
  is complete), of `list`/`tuple` by its heap kind, and of `object`. An instance or a class that carries its own
  `__class__` refuses: `isinstance` consults it.
- A function, a builtin, a module or a class object is an instance of `object` and of nothing else a classinfo can name here.

**A class statement runs once.** `Analyzer.alloc` makes the older class of a site its summary, redirects the values that name it,
and leaves the by-address references alone (`InstObj.cls`, `ClassObj.bases`), so a second run of a class statement (a loop body,
which the fixpoint runs again from a state that holds the first class) made an instance of the first class read as an instance of the
second: `isinstance(objs[0], C)` was proven for `for i in range(2): class C: ...; objs.append(C())`, where it is False. `s_ClassDef`
now refuses a class whose site already holds one (`tests/test_absint_class_in_loop.py`).

**A bound method follows its function.** `BoundMethod.func` is the address of a function object, and `State.redirect` (called when an
allocation makes the older object of a site its summary) rewrote the value references and the dict views but not that field: after
a `def` or lambda ran a second time (a loop body), the bound method of the OLDER function called the newer one. `redirect` now
rewrites `BoundMethod.func` too (`tests/test_absint_bound_method_loop.py`). The address-named fields of the heap objects are
`InstObj.cls`, `ClassObj.bases` (safe because a second class statement is refused), `BoundMethod.func` and `DictView.dicts`; every
other object holds `V` values, which `_map_values` rewrites.

richards asserts `isinstance(h, HandlerTaskRec)` four times, chaos checks `isinstance(other, GVector)` in `__add__` and
scimark `isinstance(idx, tuple)`: all were refused. `issubclass` has no contract and no model yet. richards x4 moves from
`isinstance` (line 289) to `%`-formatting (line 253, `raise Exception("Bad task id %d" % id)`; the str family, K1's [internal ref]).
Tests: `tests/test_absint_isinstance.py` (a native differential over values drawn from objects, builtins, functions and
classes against sixteen classinfo shapes, with a bug injected into the class-chain walk caught).

## Where the time goes on a heap-heavy program ([internal ref])

A profile of the chaos analysis (60 s CPU, `cProfile`) put 38 of 62 s in `State.rewrite_bounds_away` and its callers
(`kill_var`, `write_var`, `strip_terms`): each one walks every value of every frame and of every heap object
(`State._map_values`, 5.9 million `_map_obj` calls in the run) to find the bounds and float terms that name the variable.
`write_var` calls `kill_var` for the variable of EVERY non-int assignment, and 48,034 of the 62,358 calls named a variable the
zone does not have; none of them changed a value (and only 87 of the other 14,324 did).

`rewrite_bounds_away` and `shift_bounds` now return at once for a variable the zone does not have. That is exact because no
bound and no float term ever names one: a variable's bounds and terms are rewritten away before it is forgotten, and a variable
enters the zone with no value naming it. `tests/test_absint_bound_invariant.py` holds the invariant: it runs the full walk
(`_rewrite_bounds_away`, `_shift_bounds`) on a copy for every skipped call of the absint fixtures and of three heap-heavy
programs and fails if it would have changed any value. The walk itself is one pass instead of two (`_term_stripper` is
composed with the bound rewrite). Measured on chaos with the draw, reducer and operand changes: the analysis reaches the
same point (line 179) in 157 s of CPU instead of 260 s. What is left of `_map_values` is the calls for variables the zone
has; an index of the variables each object names (objects are immutable, so it could be cached on the object) is the next step
and is not built.
## The first iteration on its own, and the widening delay (A2)

*Built. Owner ruling 2026-10-05 ([internal ref]): the second build step of the loop design, after fixpoint reuse.
Modelled on Mopsa's `unroll` (`loops.ml`), N = 1, with widening delay 0.*

`interp.py::Analyzer.s_While` and `s_For` analyse every loop in three parts:

1. **Iteration 1 on its own**, from the entry state, recording as the caller does (nothing in a non-recording context).
   `continue` states join its end state; `break` states are kept.
2. **The fixpoint** (`_fixpoint`, the A4 cache in front of `_solve_fixpoint`) starts from iteration 1's END state and
   describes iterations 2 and on. If iteration 1 adds nothing (`end1 <= entry`), the entry state is the fixpoint, the
   first-iteration pass was the final pass, and nothing is iterated.
3. **The final pass** from that fixpoint records the checks of iterations 2 and on.

The loop's exit is the zero-iteration exit joined with the head exit and the `break` states of both passes (`while`/`else`
and `for`/`else` run on the join of the two normal exits; for a range `for` the zero-iteration exit has the ghost
counter at 0 and `hi <= c`). The ghost counter makes the peeled iteration exact: `i == c` there, and `i >= c + 1` in the
final pass.

- **AND, never overwrite.** `Analyzer.check` and `Analyzer._record` already accumulate, so a site is proven only if it is
  proven in BOTH passes: `x = 0; for i in range(3): y = 1 // x; x = 1` stays unproven (the first iteration divides by 0),
  and so does the mirror image (`x = 1 ... x = 0`). `tests/test_absint_peel.py` pins both for `for` and `while`.
- **The witness's loop-head invariant** is entry joined with head, since the head is visited with the entry state first
  (`Analyzer._visit_invariant`). A join of two states need not be inductive and the checker verifies preservation on its
  own, so it is kept only if one more step maps it into itself, else it is solved up to a fixpoint (and only when
  invariants are being recorded). Mutation check: the head alone claims `x >= 5` at the head of `x = 0; i = 0;
  while i < 3: x = 5; i += 1`, false at the first visit.
- **`WIDEN_DELAY` is a number of plain joins before the first widening**, 0 by default (it used to be 3, which meant
  two joins; delay 2 reproduces that schedule). It is part of the fixpoint-reuse key (`_fixpoint`), the one piece of solve
  state this step adds: it is a module constant today, but the targeted explorer will vary it per loop, and a head
  solved under one delay is not the head under another.
- **A pass starts where a statement starts** (`Analyzer.fresh_pass`): what an unfinished statement left behind (the
  allocation epoch, the borrowed references of the frame) belongs to the pass before it, exactly as `gc` clears it after
  every statement. Without it, two fuzz programs the old analysis proves were lost to "one allocation site reached
  twice within a statement" and "allocation would invalidate an in-flight heap reference".
- **A safe visit says nothing about the residual of an unsafe one** (`Analyzer._record`): a site unsafe only in the first
  iteration keeps the condition under which it raises (`(cur) is None or not hasattr(cur, 'm')`) instead of losing it.
  Before, any difference between visits turned the residual into `None`.

### Measured (2026-10-05; base tharkun `265e044` against the branch; `PYTHONHASHSEED=0`)

`absint.prove` over each corpus x four properties, 60 s budget, the first-iteration peel with each widening delay.
Statements executed are deterministic; CPU is the process time summed over the corpus.

| corpus (results) | gained / lost, delays 0 to 3 | statements: base | delay 0 | delay 1 | delay 2 | delay 3 | CPU delay 0 | CPU delay 3 |
|---|---|---|---|---|---|---|---|---|
| Real-world 80 pairs (80) | 0 / 0; 0 / 0; 0 / 0; 0 / 0 | 58,677 | 48,138 (-18.0%) | 55,421 (-5.5%) | 63,030 (+7.4%) | 70,867 (+20.8%) | 262.8 s -> 232.3 s (-11.6%) | 262.8 s -> 330.1 s (+25.6%) |
| Solve-only 65 pairs (65) | 0 / 0; 0 / 0; 0 / 0; 0 / 0 | 801 | 771 (-3.7%) | 803 (+0.2%) | 831 (+3.7%) | 859 (+7.2%) | 1.3 s -> 1.3 s (+0.9%) | 1.3 s -> 1.3 s (+2.0%) |
| eregion toys (542 x 4) (2,136) | 0 / 0; 0 / 0; 0 / 0; 0 / 0 | 16,040 | 16,040 (+0.0%) | 16,040 (+0.0%) | 16,040 (+0.0%) | 16,040 (+0.0%) | 37.0 s -> 35.9 s (-3.0%) | 37.0 s -> 66.5 s (+79.7%) |
| absint fixtures (116) | 0 / 0; 0 / 0; 0 / 0; 0 / 0 | 54,588 | 38,964 (-28.6%) | 45,124 (-17.3%) | 51,988 (-4.8%) | 59,620 (+9.2%) | 39.1 s -> 32.0 s (-18.1%) | 39.1 s -> 35.2 s (-9.9%) |
| 8,000 random programs x 4 (32,000) | 389 / 1; 389 / 0; 389 / 0; 389 / 0 | 1,889,298 | 1,589,836 (-15.9%) | 1,711,108 (-9.4%) | 1,840,724 (-2.6%) | 1,971,816 (+4.4%) | 1879.2 s -> 1809.1 s (-3.7%) | 1879.2 s -> 1336.4 s (-28.9%) |
| scimark probe (explicit calls) x 2 (2) | 0 / 0; 0 / 0; 0 / 0; 0 / 0 | 40,428 | 1,590 (-96.1%) | 2,092 (-94.8%) | 2,744 (-93.2%) | 3,496 (-91.4%) | 120.3 s -> 4.2 s (-96.5%) | 120.3 s -> 9.6 s (-92.0%) |

CPU is measured on a shared host and is noisy (the toys execute no loop at all, and their CPU still moves between -3 % and +80 %
across identical work); statements executed is the deterministic measure and is what the comparison rests on. Real-world CPU is
dominated by `bm_unpack_sequence` (about 35 s a pair, two loops, no nesting, nothing a loop change touches): without it the 74
other results take 36.4 s of CPU at the base and 33.3 s (-8.6 %) at delay 0, for 24,966 statements against 19,227 (-23.0 %).

* **Nothing is lost on Real-world 80, Solve-only 65, the toys or the fixtures, at any delay.** Delay 0 is the cheapest on every
  corpus that has a nested loop (Real-world -18.0 % statements, the fixtures -28.6 %, the random programs -15.9 %); each extra
  plain join costs statements and buys nothing there (delay 3: +20.8 % statements on Real-world).
* **The random programs** gain 389 proofs at every delay: 361 are refusals the old analysis made that the first-iteration pass and
  `fresh_pass` remove ("one allocation site reached twice within a statement" 169, "allocation would invalidate an in-flight
  heap reference" 75, "tuple() of several possible iterables" 117 -- the first iteration is exact, the head describes less) and 28
  are proofs the first iteration makes possible (a loop that runs once, a first-iteration-only fact). One proof is lost at
  delay 0 and kept at every other delay (`p00461.py`, arithmetic): widening at the first update loses an upper bound that one
  plain join keeps. The targeted explorer of the design raises the delay of the loop a check depends on and recovers it.
* **The scimark probe** (the `*args` dispatch rewritten to explicit calls, because absint refuses `*args` first) ran out of
  its 60 s budget at about 24,000 statements. With the first iteration peeled it stops after about 800 statements, in 2 s, with
  `summary join: two ranges at one site`: the iteration-1 range of an inner `for` (bounds `(1, 1)` and `(3, None)`) meets the
  summary of that site whose bounds a widened zone no longer knows (`(None, None)`), and `src/tharkun/solve/absint/state.py::ojoin` joins two ranges only
  when they are equal. That refusal is an honest UNKNOWN, not a lost proof (the base had no proof either), and a hull join of
  two interval-bounded ranges would remove it: not built here, see "Ruled, not built".
* **Records change** as the passes add up: a site's `fact` text lists the facts of both passes, and 13 Real-world and Solve-only
  records lose one `scope` fact (the first visit is part of the intersection of must-facts). No record loses its residual.

## Dict keys from the zone ([internal ref])

A dict key is a set of constants when the zone bounds it (`interp.py::Analyzer.key_consts`, at most 64 values), not only
when the value carries literal bounds. A name reads as a zone variable (`IntInfo.lin`), so `EMPTY, WHITE, BLACK = 0, 1, 2;
SHOW = {EMPTY: '.', WHITE: 'o', BLACK: 'x'}` (the `go` benchmark's line 20) and `A = 1; d = {A: 5}` were refused as
"dict literal with a non-constant key", and `d[i]` for a loop variable over `range(3)` had no key set at all. The bounds are
sound, so a singleton key stays definite and a read proves `KeyError` away when every key of a finite set is present; a store
with one constant key is a strong update. A nonsingleton builtin scalar key (int, bool, float, str or None) instead opens the
dict and contributes summarized key/value facts to `okeys`/`ovals`; float keys always use this summary, including a singleton
float fact. It does not make any possible key definite. Earlier constant keys remain present, with their values joined when a
later uncertain key may collide (`True == 1 == 1.0`). This summary loses cardinality and key identity: in particular, it
cannot prove lookup by a same-object NaN key. Float keys are never placed in the definite constant-key map. Reference/custom
hash/equality keys and dict unpacking remain refused. For a non-strong subscript store, each definitely-present key that the
incoming scalar key may equal keeps its presence but joins the incoming value; the open-key/value summary is updated as well.
The tests are `tests/test_absint_dict_keys.py` (native key semantics plus a differential over named, looped and drawn keys,
with a bug injected into the key set caught).
`go`'s next blocker behind it was `set()` (line 147): absint had no set heap kind (built by [internal ref]: "Sets", below).

## Sets ([internal ref]): `SetObj`, one kind for `set` and `frozenset`

`bm_go_nondet` stopped at `set()` (line 147, `self.hash_set = set()`) with `call of an unknown value: set`; the set contracts
(`solve/contracts/rules_set.py`) existed and nothing consumed them semantically. The set family is now one heap kind and one
mixin, built whole (every method, operator and comparison the contracts cover, not the ones one program used).

**Heap shape** (`heap.py::SetObj`, kind `"set"`): `length` is an interval `(lo, hi)` of the number of DISTINCT elements
(`hi` is `None` when unbounded; never a zone variable, so a set has no recent-object relations of its own, only the usual
strong update of a recent object and weak update of a summary through `Analyzer.update_obj`), `elem` is ONE summary every
element satisfies (`BOT`: none), and `frozen` is True for a `frozenset`. `SetObj.make` keeps the facts in agreement: no
element means length `(0, 0)` and the other way round, a length is never negative, a lower bound never exceeds the upper one.
Two equal values (`1`, `1.0`, `True`) are one element and either object may be the one a set keeps (a union keeps the
receiver's, an intersection the argument's: `{1}.intersection([1.0])` is `{1.0}`), so every operation that can bring in an
element joins the elements of every operand that can supply one. `state.py` joins, includes and widens a set by the
interval (`_len_join`/`_len_widen`) and the element summary; a set and a frozenset at one site do not join (`Incompatible`:
a named refusal), and `_map_obj` rewrites the element summary when a variable its symbolic bounds name changes.

**Construction.** `Analyzer.e_Set` (a display `{a, b}`; the length is exact when every element is an int, bool, str or None
constant, else between 1 and the number of elements; a starred element refuses), `c_set`/`c_frozenset` (the contracts `set` and
`frozenset`, run by `_contract_builtin_call`: the length is the number of distinct values the argument can yield, exact for a
set, a range or a dict's keys, at least 1 for any other iterable that yields something; `frozenset(f)` of an exact frozenset is
`f`), and the comprehension `{x for ...}`, which `solve/lowering` turns into `acc = set(); for ...: acc.add(x)`. An iterator argument
has no lower bound (CPython consumes it; absint does not track that); `dict.items()` refuses.

**Hashing is checked, not assumed** (`sets.py::SetsMixin._hashable`). Every element is hashed when it enters a set: a list, dict,
array, dict keys or items view or (non-frozen) set is a `TypeError` (a `dict.values()` view has no `__eq__` and hashes by identity), a tuple hashes when its parts do, an instance hashes unless its class
defines `__eq__` (then `__hash__` is `None`; a class that defines `__hash__` is refused when it is defined), a scalar, class,
function or static object hashes, and every other kind (a bound method, a file, a pattern) is refused rather than guessed. The
alternatives that CERTAINLY fail are removed from the element value, so the path past `s.add([1])` does not exist. An alternative
that fails only for SOME values is not removed: `sets.py::SetsMixin._ref_hashable` answers `True` (always hashes), `False` (never)
or `None` (a tuple that hashes only for some values of its parts), `_hashable` keeps the `None` alternatives and reports the
element as not certainly hashable (the `TypeError` check is recorded, never proven away). So `x = [] if nd() > 0 else 1` then
`{(x,)}` keeps the path (the draws where `x` is an int hash) and `{([], x)}` does not (a part that never hashes). A tuple known only
by a length range and an element summary follows the same rule through its length: it is certainly unhashable only when its
elements all are and it has at least one (`tuple(l)` of a list of lists grown in a loop that may not run hashes when it is
empty), and a summary with no element at all hashes. Only
`remove`, `discard` and `in` convert an unhashable SET to a frozenset first; `add`, `update`, `union`, `difference`, ... do not
(`tests/contracts/test_set_.py` pins which, on CPython 3.14.4).

**The contracts are the authority, the model argues** (as the reducers read theirs, "The iterable-reducing builtins"). Each method
`set.<name>`, each operator (`set.__or__` ... `set.__ixor__`), each ordering (`set.__lt__` ... `set.__ge__`), `set.__eq__`/`__ne__`,
`set.__contains__` and the constructors are read through `adapters/absint.py::contract_for_method` / `contract_for_builtin` (only a
TESTED contract is entered; withdrawing one makes the call `Unsupported`) and recorded as the contract's seven classes at the call
(`Analyzer._contract_checks`, op `set.<name>:<class>`). A class starts as the contract states it; the model upgrades it to proven
only for the reason it gives:
- the six classes a contract admits only by propagation from a user hook (`__hash__`, `__eq__`, `__iter__`) are proven
  whenever the arguments are of the supported kinds, because none can run: elements are scalars, tuples and frozensets of those,
  classes, functions, or instances of a class with neither hook, and an instance's `__iter__` body is analysed
  (`Analyzer.iterable`). `sets.py::SetsMixin._grounded` holds that argument to the contract: it can never prove a class the
  callable raises by its own logic, which is how a first version wrongly proved away `KeyError` of `remove` and `pop` (found by
  the differential, below);
- `TypeError` is proven when the arity is right, every argument is iterable and every element hashes (an operator: both operands
  are sets); a wrong arity or keyword, a non-iterable argument and an operator on a non-set are certain `TypeError`s and end
  the path;
- **`KeyError` of `remove` and `pop` stays a recorded check.** `pop` is proven only when the length has a lower bound of 1;
  `remove` only when the element is certainly there: the set has an element and every element is the one constant asked for
  (`True` and `1` are one; a float is never a certain member, `nan in {nan}` holds only for the same object). The path that
  continues past `remove` or `pop` had the element, so the length drops by one; a set that is certainly empty has no such
  path. Nothing decides that an element is ABSENT from a non-empty set (the summary says what is in it, never what is not):
  `s = {1}; s.remove(2); assert False` stays UNKNOWN.

**Operations and what they know** (`a`, `b` lengths `alo..ahi`, `blo..bhi`; `dlo` the fewest distinct values an iterable
yields: a set, a dict's keys and a range yield distinct ones, a list or a dict's values may repeat one):

| operation | length | elements |
|---|---|---|
| `add(x)` | `max(lo, 1) .. hi + 1` | joins `x` |
| `discard(x)` | `max(lo - 1, 0) .. hi` | unchanged |
| `remove(x)` (the path that continues) | `max(lo - 1, 0) .. hi - 1` | unchanged |
| `pop()` (the path that continues) | `max(lo - 1, 0) .. hi - 1` | returns the summary (materialised: an element exists) |
| `clear()` | `0 .. 0` | none |
| `copy()` | the receiver's; a frozenset returns itself | the receiver's |
| `update(*its)`, `a \| b`, `a \|= b` | `max(lo, dlo..) .. hi + sum(hi_i)` | joins every argument |
| `union(*its)` | the same, in a new set | joins every argument |
| `intersection(*its)`, `a & b`, `intersection_update`, `a &= b` | `0 .. min(hi, hi_i)` (no argument: a copy) | joins both sides |
| `difference(*its)`, `a - b`, `difference_update`, `a -= b` | `max(0, lo - sum(hi_i)) .. hi` | the receiver's |
| `symmetric_difference(it)`, `a ^ b`, `symmetric_difference_update`, `a ^= b` | `max(0, lo - hi_b, dlo_b - hi) .. hi + hi_b` | joins both sides |
| `issubset`, `issuperset`, `isdisjoint` | a bool; true for an empty receiver (`issubset`) or an empty argument (`issuperset`), either empty (`isdisjoint`), false when the lengths cannot fit | |
| `a < b`, `a <= b`, `a > b`, `a >= b` | a bool decided at the same ends | |
| `a == b`, `a != b` | decided when the lengths cannot match or both are empty; never an error, for any operand | |
| `x in s` | false for an empty set, true for a certain member | |
| `len(s)` | the interval, copied into one temporary as a list's length is (a comparison is decided in the zone) | |
| `if s`, `not s` | truthy exactly when `hi != 0`, falsy when `lo == 0` | |

The result of an operator or method that allocates is a NEW set at the node's allocation site (`sets.py::SetsMixin._alloc_set`;
the site's tag names where the node ends, because `a | b | c` and `a.union(b).union(c)` are nested nodes that start at one column).
The element summary the call read from the heap is not stale after the allocation: the allocation turns the site's recent object
into its summary and `State.redirect` rewrites the heap, so a reference to the old object in the value the caller holds is
rewritten to the summary here (a `f = frozenset([f])` loop would otherwise hold itself; `tests/test_absint_sets.py`
checks it). The sets of one kind that a method may be called on are ONE call (`Analyzer.call_method`), so a set made in a loop
(its recent object and its summary) allocates its result once and a mutator updates both weakly.

An operator takes `set` and `frozenset` operands only (anything else, including a plain iterable, is a `TypeError`: a set operator
is not `union`); the result has the LEFT operand's type; `a |= b` changes `a` in place and returns it, and for a frozenset it
rebinds the name to a new frozenset (`tests/contracts/test_set_.py`). A frozenset has the non-mutating methods only: a mutating
one, an unknown name or a name no set has is an `AttributeError` check (`Analyzer.call_method`, `attr_load`), a dunder-shaped name
refuses.

**Consumers.** Everything that reads an iterable now knows a set: `for`, unpacking, `list`, `tuple`, `sorted`, `min`, `max`, `sum`,
`any`, `all` (`reducers.py::ReducersMixin._reduce_source`), `xs.extend(s)`, `iter`, `print`/formatting (the elements are checked
for a custom `__repr__`), `custom_comparison` (a user `__eq__`/`__lt__` on an element refuses), `x, y = s` (a count the length
cannot have is not used to end the path: a `ValueError` is not a property class, and the model does not claim one). A slice
assignment `xs[0:1] = <set>` joins the set's elements into the list (`Analyzer.slice_store`, with the range, dict, dict view and array
sources of "Two wrong TRUEs of the array and the slice store"; `tests/test_absint_sets.py`). `random.choice(s)` refuses a set by name (it
is a `TypeError` for a non-empty one and an `IndexError` for an empty one; the random family is the next PR). A bound method
(`m = s.add`) is dispatched by the kind of its receiver in `Analyzer._call1`, with the array and list receivers of [internal ref].

**Refused (`Unsupported`, the proof is UNKNOWN and says why):** a starred element, `set(d.items())`, a set operator with a dict
view (the view's reflected operator), a set or frozenset operand that may be either kind where the result's type must be one, an
element of a kind whose hashability is not modelled, an `__len__` on an instance given to a method of the family (no verified
length-hint rule), `isinstance(s, set)`, `random.choice` of a set, a slice assignment from `dict.items()`. What is only imprecise: nothing
decides that an element is absent; a set grown in a loop widens its upper bound; `a.copy() is a` is not decided; the symbolic
executor has no set model (a listed gap in the engine audit).

**Tests and evidence.** `tests/test_absint_sets.py`: exactness of every operation, the recorded checks (`KeyError` of `remove` and `pop`,
an unhashable element, a mutating method on a frozenset, an operator on a non-set), the named refusals, allocation (a set made in a loop,
a frozenset holding the object being built), a regression test per site the kind matrix found in this model (a `dict.values()` view
hashes; a keys or items view does not; a tuple with a part that may not hash keeps the path: 36 shapes of
`TUPLE_MAY_HASH`/`TUPLE_HASHES`/`TUPLE_NEVER_HASHES`, each against the native truth), a native differential over generated set
programs (a sample runs in the default lane), and seven negative controls, each a deliberately broken model the tests must catch
(`discard` that keeps the old lower bound, `pop` that proves `KeyError` away, `add` that forgets its element, a membership test that
calls every non-empty set a certain member, and three readings of `_ref_hashable`: "may hash" read as "never hashes", the same for a
summary tuple that may be empty, and a summary tuple with no element read as unhashable).
`tests/contracts/test_set_.py` runs every contract claim on CPython 3.14.4, on both types; each set rule
holds its valid program and its wrong-arity twin (`SAMPLES` and `TWINS` of `rules_set.py`; the absint column is `C` for the whole family); `tests/test_absint_kind_matrix.py` runs sets,
frozensets and every other kind into every consumer (193,240 programs on the merged tree: 0 crashes, and the one hole it found, the
values view, is fixed above).

## The whole `math` module ([internal ref] batch 5)

`interp.py::MODULES["math"]` used to list three functions (`sin`, `cos`, `sqrt`); `math.tan` (bm_raytrace
line 252) and every other call fell to the contract fallback (`_contract_call`), whose unknown result
aborts the next use. `Analyzer._math_call` now models EVERY function of the module (57 functions; the
constants `pi`, `e`, `tau`, `inf` and `nan` are the exact floats CPython has, read through their rules since [internal ref]
step 2: `module_attr` -> `visit_rule` -> `AbsintVisitor.visit_math_constant` -> `adapters/math_model.py::constant`, a named refusal
while the contract is untested or withdrawn; `inf` and `nan` were the any-float before). It is driven by the contract
(`solve/contracts/rules_math.py`) and the one table every engine shares
(`src/tharkun/solve/contracts/adapters/math_model.py`, read here through `src/tharkun/solve/contracts/adapters/absint.py`):

- the contract supplies the seven property classes, recorded at the call as `math.<name>:<class>` ops;
  the model proves what it can for the abstract arguments: `TypeError` once the argument count and kinds
  are verified (an exact int/bool/float; an int-only function needs an int, `perm`'s `k` may be `None`),
  `ZeroDivisionError` of `log(x, base)` once the base provably is not 1, and every other class because a
  call of the verified shape on exact numbers runs no user code (`src/tharkun/solve/contracts/adapters/math_model.py::SCALAR_SHAPE_RAISES`
  is the hand-verified table: only `log` raises a property class on its own);
- the result is an interval from `solve/values.py` where a transfer exists (`sin`, `cos`, `sqrt`, `exp`,
  `log`, `log10`, `atan`, `atan2`, `tanh`, `fabs`, `hypot`), else the closed range the table states (`acos`
  in `[0, pi]`, `cosh >= 1`, `erfc` in `[0, 2]`, ...), else any float; nan only when an argument may be nan
  (`log(inf, inf)` is the one nan from numbers). `floor`/`ceil`/`trunc` of an int is the int itself (so
  it can index), of a float the rounded interval; `factorial`/`gcd`/`lcm`/`isqrt`/`comb`/`perm` carry
  their lower bound; `frexp`/`modf` return a tuple; the predicates return a bool. `log` and `log10` take an
  int exactly, as CPython does (`log(10**400)` is 921.0), not through a float conversion;
- ValueError (a domain error) and OverflowError are not property classes: a call that certainly raises one
  ends the path, a call that may continues on the other states;
- refused (`Unsupported`, with the reason): a keyword the function really takes (`isclose(rel_tol=...)`,
  `nextafter(steps=)`, `prod(start=)`), an iterable argument (`fsum`, `prod`, `dist`, `sumprod`: absint has
  no element summary of an arbitrary iterable), an argument it cannot name (`top`);
- withdrawing or un-testing a function's contract withdraws its model.

`tests/contracts/test_adapters_absint_math.py` pins this, and its containment tests are the soundness
check of the model itself: for every function and every sample inside the abstract arguments, CPython's
answer lies in the abstract answer (or CPython raises ValueError/OverflowError).

A float LITERAL is the any-float value in this analyser (`Analyzer.e_Constant`): `math.log(8.0, 2.0)`
cannot prove its base is not 1, `math.log(8.0, 2)` can. That precision gap predates the math model and is
what keeps most float proofs lemma-driven.

## A str subscript through the rule ([internal ref])

Since #151 `Analyzer.subscript_load` routes a definite-str subscript through `visit_rule("str.__getitem__")`; the abstract-str fallback
(`visitor.py::AbsintVisitor._abstract_string`) had proven-cases for `str.__mod__`, `__contains__`, the zero-argument methods, `repr`/`str`/`format`
but none for `__getitem__`, so every `s[i]` and `s[a:b]` on an abstract str (a `nondet_str()`) left the contract's classes unproven and
`dynamic-typing` was never TRUE for such a program (five of MIN's mini tasks lost it), and a slice was refused outright (`non-builtin operand`,
`non-finite slice bounds`) where sv_2027 `cfb8505` proved it. The case, for a receiver that is a definite builtin str, reads the KEY
(`visitor.py::str_key_parts`) and asks two questions: may some part of it raise `TypeError`, and may some part be an int index (`IndexError`)?

| key | `TypeError` | `IndexError` | the five hook classes (`AttributeError`, `KeyError`, `ZeroDivisionError`, `FloatingPointError`, `AssertionError`) |
|---|---|---|---|
| an exact int or bool | proven | unproven | proven |
| a slice with int, bool or absent bounds | proven | proven (a slice clamps) | proven |
| a float, str, None, a tuple/list/dict, a slice with a float/str bound | unproven (certain) | proven | proven |
| a join of the above (an int or a float) | unproven | unproven if some part is an int | proven |
| a builtin function, type, exception or module (`len`, `int`, `math`), a `range`, array or iterator | unproven (certain) | proven | proven |
| a user object (an instance may define `__index__`), an unknown value, a slice with such a bound | refused (`Unsupported`) | | |

**A certain `TypeError` ends the path** (MIN's suite v3: 28 precision regressions of #151 against `ec3cee0`, all UNKNOWN on `fca4fb9`). When every part of the key
raises it (the rows marked "(certain)": a float, str, None, a container or builtin object, a builtin function or type, a slice with a bound of such a kind)
the `TypeError` is recorded as an unproven check and the path ends, exactly as a subscript of a finite str does (`_native_string`: no output) and as
`ec3cee0` did: `s[1.5]; reached = True; 1 // 0` is TRUE for `ZeroDivisionError`, the sentinel is never reached. A key with a part that may be valid (an int or a float,
a slice whose bound may be an int) does not end the path. A slice bound that is an int, bool or None (`s[:(1 if c else None)]`) is an int bound
(`("int-slice",)`), never a `TypeError`; a slice with a bound of no int or None part (`s[:[1]]`, `s[:range(2)]`) is `("bad-slice",)`, a certain `TypeError`.
`visitor.py::NO_INDEX_KINDS` and `NO_INDEX_TAGS` list what has no `__index__` (a `re.RegexFlag` does: `s[re.I]` is valid, and `native-flag` is not on the list).

No part of a proven key runs user code (`str.__getitem__` type-checks a float, str, None or container without calling anything on it), which is
why the five hook classes are proven whatever the key is; an object key is refused because its `__index__` would run. The slice keys are
`("native-slice", slice(...))` for finitely many constant bounds, `("int-slice",)` when the bounds are exact ints that are not finitely many
(`s[:len(t)]`), and `("any-slice",)` when they run no user code but may not be ints (`s[:x]` with `x` a float): built locally by
`Analyzer.subscript_load` and passed straight to the rule, never stored. The precision of the last three rows is MG's finding on `e5580b2`:
a program that natively raises `TypeError` at such a subscript had every class unproven (179 cells against `ec3cee0` on MG's table) where sv_2027
proved everything but the `TypeError`.

`tests/test_absint_str_getitem.py` (the five MIN tasks, native controls, the negative probes) and ten negative controls (each part of the
fix undone, the tests fail).

## try/except/else/finally ([internal ref])

`interp.py::Analyzer.s_Try` models a `try` statement exactly; what it does not model is a named refusal
(`Unsupported`), never a guess.  The soundness review of the first implementation (lane TRY, 2026-10-07) found
and fixed: implicit exceptions (a `KeyError` of `d[k]`, a `ZeroDivisionError`, ...) that were never routed to the
handlers, a raising path dropped when another exit of the same try body existed, the state BEFORE a statement
handed to a handler (the callee's side effects before the raise lost), user exception classes matched by NAME,
a `break`/`continue` in a loop inside a `try/finally` routed through the finally (the continuation after the
loop lost), an exception leaving a class body dropped, and a harness draw whose refusal a handler could swallow
proven TRUE.  The rules now:

- **An exception in flight is a value.** `interp.py::_Exc` is the state at the raise (a copy the exception
  owns), the class (a builtin exception NAME, or the heap ADDRESS of a user exception class: identity, never a
  name, `test_user_exception_classes_match_by_identity_not_by_name`), the instance value, the site and the
  check the site would have recorded outside a try (`op`, `fact`, `residual`, `operand`).  It is recorded only
  where it ESCAPES (`interp.py::Analyzer._route_exc`: at module level, with the original site, op and fact, for
  the property classes it is, via `interp.py::Analyzer._record_check`), so a caught exception is not a
  violation and an uncaught one is exactly the check it always was.
- **Every raise is routed, or refused.** `interp.py::Analyzer.s_Raise` (an explicit raise: one `_Exc` per
  alternative of the value, `_raise_alternatives`/`_materialize_raise`; a raised CLASS is instantiated, its
  `__init__` runs, `test_a_raised_class_is_instantiated_and_matches_native`; a bare `raise` re-raises the
  innermost handled exception, `Analyzer._active_exc_stack`, also from a function the handler calls; inside a
  finally block it is refused).  `interp.py::Analyzer.check`: while any try statement is active
  (`Analyzer._try_stack`, of this function or a caller's), an unproven check is an exception in flight, routed
  from a copy of the state the site hands over (`st=`); a site that has no state to hand over is REFUSED by
  name ("the raising state is not available at this site"), so no check is ever silently recorded or dropped
  inside a try.  The sites that hand their state over: every subscript/index/key/slice check, `attr`,
  `attr-store`, `call`, `call-args` (user functions), `compare`, `in`, `operator`, `unary`, `binop`, `divide`,
  `iter`, `unpack`, `format`, `init-return`, `module-attr`, `pop`, `assert`.
- **Unrouted calls are refused at dispatch.** While any try is active, `call_static` refuses builtin/module
  calls, `call_method` and `_call1` refuse builtin heap methods and bound aliases, `_outfile_call` refuses
  output/context hooks, and `visit_rule` refuses contract calls and formatting. Their models retain property
  checks, not every non-property exceptional exit. This is a common boundary, not an `index`/`remove`
  blacklist: those three formerly wrong-TRUE search cases have native failing-handler controls.
  User functions, plain user methods and constructors run their actual bodies; stationary method summaries
  are bypassed inside a try. Their explicit raises and state-bearing checks route through `_Exc`, and an
  unmodeled operation or check without a raising state refuses. Builtin exception constructors are the only
  static-call exception: the explicit small constructor subset uses `BaseException_new/init` in CPython
  3.14.4 `Objects/exceptions.c`; keyword arguments route a TypeError. Specialized constructors and user
  subclasses of them refuse. Python wrapper-object identity does not identify these C slots.
  Multiple inheritance remains unsupported.
- **Implicit non-property raises also refuse.** Named boundaries cover unresolved names, destructuring,
  slices, float/builtin-sequence arithmetic, integer true division/power, possibly negative shifts,
  RegexFlag operators, oversized implicit `__len__` results, and iteration over mutable mappings/sets or opaque iterators. A preconstructed
  range/list/tuple/array can still be iterated in a try. Harness draws are static calls and refuse there;
  native scheduler replay still checks the vector. These refusals preserve UNKNOWN rather than ignore
  a handler which could raise a property exception.
- **Contract wrappers preserve routed exceptions.** An unproven model check inside a try has already
  handed its original state to the handler. When the current contract wrapper vetoes that same class
  (`_proven_classes` returns `None`), `_contract_checks` does not duplicate it using a nonexistent
  raising state. An unresolved contract class with no failing model check still refuses by name.
  Contract-driven builtins, including `range`, remain refused inside a try; prepare an iterator before
  entering the try when its construction is outside the protected region.
- **No path ends silently inside a try** (`interp.py::Analyzer.exec_block`): a statement executed while a
  try is active that ends every path without a routed exit (`Analyzer._exit_count`, bumped by every routed
  exception, return, break and continue) is refused.  Defence in depth behind the rule above, at statement
  granularity.
- **Handlers.** `interp.py::Analyzer._dispatch_handlers` matches the clauses in order against each exception
  the body raised; `_handler_targets` evaluates the clause in the exception's state (a tuple expression element
  by element: a site may not allocate twice in one statement); a clause whose value is not a single class, or
  a tuple of single classes, is refused; a clause naming a non-exception is the `TypeError` CPython raises in
  place of the exception (`except_non_exception_class_lazy`/`_hit` in `tests/contracts/_swallowing_cases.py`).
  Matching is `interp.py::Analyzer._is_subclass`: builtin against builtin by CPython's own hierarchy, a user
  class by its chain of bases (`Analyzer.subclass`) and the one builtin base `s_ClassDef` recorded
  (`__exc_base__`, read by `Analyzer._exc_base_of`; a chain naming two builtin bases is refused).
  `except E as e: BODY` is `except E: try: BODY finally: del e` (CPython): `interp.py::Analyzer._run_handler`
  runs the body inside an implicit frame whose finalizer deletes the name (`Analyzer.delete_name`) and releases
  the instance (pinned in `State.pins` while in flight, so gc keeps it) on EVERY exit, including return, break,
  continue and a raise (`test_except_as_name_is_deleted_on_every_exit_and_matches_native`).
- **Frames and exits.** `interp.py::_TryFrame` is one active try statement (or the implicit handler frame):
  the exceptions `raised` in its body, the `unhandled` ones (no clause matched; raised in a handler or in else;
  a `TypeError` of a bad clause), and the exits waiting for its finally (`pending_returns`, `pending_breaks`,
  `pending_continues`).  `interp.py::Analyzer._route_exc` delivers an exception to the innermost frame of the
  function that receives it (a body with handlers; a handler/else/finally-only frame with a finally: the
  finally runs, then it continues outward), else out of the function (`FuncCtx.raises`, delivered by
  `interp.py::Analyzer.call_func` to the caller at the call, unwound by one frame, and by `s_ClassDef` to the
  module for a class body), else it is uncaught.  `interp.py::Analyzer._route_return` goes through every
  enclosing finally of the function (`Analyzer._finally_frame`); `_route_break`/`_route_continue` only through
  the frames INSIDE their own loop (`_TryFrame.lctx`), never a try around the loop
  (`test_break_in_a_loop_inside_try_keeps_the_continuation_negative_control`).
- **finally** (`interp.py::Analyzer._finish_frame`): runs on the normal continuation and on every pending exit
  separately; an exit the finally block itself ends (its own return/break/continue/raise, routed to the
  enclosing frames) replaces the pending one; the frame is popped and the surviving exits continue outward.
  `else` runs iff the body completed; an exception in `else` or in a handler belongs to the enclosing statement.
- **Loops.** Exits routed inside the fixpoint iterations are discarded exactly like returns are
  (`interp.py::Analyzer._exit_mark`/`_exit_truncate` in `_solve_fixpoint` and `_visit_invariant`): iteration 1
  and the final pass from the head keep theirs, which cover every iteration.
- **The length probe** (`interp.py::Analyzer._probe_length`): a `TypeError` or a subclass leaving `__len__` into the
  function the probe runs in (`_ProbeScope.fctx`) is swallowed by CPython's hint by class identity; `call_func` hands it to the
  probe instead of the caller, exactly, so `lh_typeerror_swallowed_dt` is now proven (`ABSINT_PROVES`).
- **Refusals** (all named): `except*`/`ExceptionGroup` (`Analyzer.s_TryStar`, a raise or clause naming
  `ExceptionGroup`/`BaseExceptionGroup`), `raise ... from ...` (chaining effects on state), a bare raise
  outside a handler or inside a finally block, the non-property raises and the draws above, a clause that is
  not a single class, a user class with two builtin exception bases, an extern module left half-executed
  (`Analyzer._exec_extern_toplevel` restores the stacks). Write-only `with` remains modeled outside a try;
  its output hooks refuse inside a try.

Evidence: `tests/test_absint_try.py`, every construct against native CPython 3.14.4 (values AND exceptions) with
a negative control per rule (undo the rule, a test fails); the swallowing attacks of
`tests/test_sv_draw_vectors.py` (a draw inside a try is a refusal, so the vector is replayed and refused by the
child); `tests/contracts/test_swallowing_engines.py` (`except_non_exception_class_lazy`,
`lh_typeerror_swallowed_dt` proven exactly).

## Case splits on a boolean draw

The join at the end of an `if` (or the join of a returning call) keeps what both arms agree on and drops the
correlation between a harness draw and what was computed from it. SV-COMP's `type-eval-py` tasks are built on exactly
that correlation:

```python
flag = _sv_verifier.nondet_bool()
v = "s" if flag else 1
if flag:  _sv_verifier.check_type(v, str)
else:     _sv_verifier.check_type(v, int)
```

Joined, `v` is `str | int` and `check_type` refuses ("check_type value has mixed or unsupported possible types").
`_sv_verifier.nondet_bool()` takes exactly the values `False` and `True`, so an execution is one of two, and the
analysis may run the rest of the program once per value and join only where the block ends.

- **What is split** (`Analyzer._draw_to_split`, `_once_draws`, `_exec_split`): a statement of the program's own file
  that is an `Assign`/`AnnAssign` value, an `Expr` value or an `If` test, with a non-empty rest of its block, holding
  exactly ONE `nondet_bool()` call that the statement evaluates at most once and in an unconditional position: call
  callee/arguments/keyword values, operator operands, the first comparator of a comparison, the first operand of
  `and`/`or`, the test of a conditional expression, the base and index of an attribute or subscript, the elements of
  a display. The statement runs twice with that call forced to `False` / `True`, then `stmts[idx + 1:]` runs on each
  outcome, and the two end states are joined by `_exec_split` (the block's own end).
- **What is not**: a draw in a `while`/`for` header, a comprehension, a lambda, a conditional arm or the tail of a
  short circuit; two or more eligible draws in one statement; a statement inside a loop body (`lctx`), a `try` (also
  of a caller: the draw's own refusal is routed there), an extern file or a pass that does not record; a `return`
  value (no rest to keep apart). `nondet_int`/`str`/`float` draws are never split.
- **The retry** (`absint.prove`): `split_draws=None` (the default) runs the joined analysis exactly as before and only
  when it did not prove TRUE, and the source mentions `nondet_bool`, runs it once more with `split_draws=True` on at
  most `max(3 s, 8x)` of the joined run's CPU and never beyond the caller's `time_budget`. TRUE from the retry is
  returned (`AbsintResult.draw_splits` counts the splits; `explain()` says so); otherwise the JOINED run's result,
  reason and records are returned unchanged. A program the joined run proves is analysed exactly as before.
- **Budget**: `MAX_DRAW_SPLITS` (64: a full tree of depth 6). Once spent, further draws are analysed joined, the
  way they always were; a partition is never returned on its own: `_exec_split` always joins its outcomes.
- **Why it is sound.** (1) The two runs cover every value of the draw, and the join of their end states is a sound
  over-approximation of the block's end, as the join of an `if`'s arms is. (2) The force is the call node's own,
  recorded with the activation it was made in (frame depth and call string, `_activation`); `call_func` hides it from
  every callee, so a callee's own `nondet_bool()` (another node), the same node reached again in a deeper frame and a
  later call of the same function never consume it; the first evaluation consumes it; a `finally` removes it. (3)
  While a force is pending nothing is memoized: `_fixpoint` neither reads nor writes its reuse cache and the class
  contract memo stores nothing. (4) Returns, raises and the other exits of the rest of the block are appended to the
  same `fctx.returns`/`fctx.raises` lists from both outcomes, so the caller joins them exactly as it joins any two
  returns; an outcome that returns or raises while the other falls through is covered by the join of the lists and the
  end state. (5) Checks are keyed by site and AND-ed over visits, the per-site records and statement-entry facts
  intersect over visits, exactly as for a site reached in two call contexts: a site only one outcome reaches keeps that
  outcome's facts, because no execution reaches it otherwise. (6) Candidate invariants are only taken from a state that
  is not one outcome of a split: `capture_invariant` records nothing while the outcomes run apart, so a TRUE that used
  splits carries the invariant of the split statement's own entry and of the statements before and after the block,
  never a fact true of one outcome only.
- **Tests**: `tests/test_absint_draw_split.py` runs 50 adversarial programs (`tests/fixtures/absint/draw_split/`)
  natively over EVERY draw vector and fails if a property class that escapes some vector is proven TRUE in any mode
  and at budgets 0, 1, 2, 5; it also instruments the analyzer to show a callee sees no pending force and no memo is
  written while one is pending.
- **Measured** on the 704 Type-specific pairs that were UNKNOWN at `be746220` (`type-eval-py`, `typeevalpy`): the
  first refusal "check_type value has mixed or unsupported possible types" falls from 100 pairs to 24. What stays open
  after the split is symbolic equality of a result with an expression of the draws (`assert b == 2 * (n + 3)`,
  `assert a == "Hello from " + name`), decorators/closures, nested classes, `type()` calls and class hints.

## check_type with a class of the program

`_sv_verifier.check_type(value, C)` for a class `C` the program defines is, in the benchmark's runtime model, `isinstance(value, C)`
(a plain class has no `typing` origin), and it raises `TypeError` (formatting `value!r` into the message) when that is False.
`Analyzer._check_type_user_class` reads it through the decision `isinstance` already makes (`_isinstance_results`: the
single-inheritance chain of the object's class, `list`/`tuple` by heap kind, a scalar, function or class object is an instance of
nothing a user class names; an object or class carrying its own `__class__` refuses), under the same guarantees `s_ClassDef` gives
for `isinstance` (no metaclass, hence no `__instancecheck__`; no multiple inheritance; no builtin base but `object` and the
exceptions).

- **Proven** only when EVERY alternative of `value` is an instance of `C` (a subclass instance included): the check holds.
- **A possible mismatch of a scalar value** (`1`, `None`, ...) is an unproven `TypeError` check; a certain one ends the path as an
  unproven check does everywhere. A possible mismatch of an object (a `repr` of a user class runs user code that may raise any
  class, `KeyError` included) refuses by name, as the mismatch of a builtin container already did.
- **Unchanged**: a hint that is not one class address (a union of classes is a case split of the program, a `typing` form, a builtin
  other than the supported ones) keeps its refusal.
- Tests: `tests/test_absint_class_hints.py` runs 27 adversarial programs (`tests/fixtures/absint/class_hint/`) natively over every
  boolean draw vector; a metaclass with `__instancecheck__`, a `__class__` property, `a.__class__ = B`, a `__getattribute__` that
  answers `__class__`, a `repr` that raises, a rebound class name and a class object or function passed as the value are among them.

## Files and how they relate

Measured from the committed source in this branch by `scripts/doc_import_map.py::compute_ast_imports`, including lazy imports.

### Subsystem: `tharkun.solve.absint` (15 files)

- `src/tharkun/solve/absint/__init__.py`
  - **Role**: Public package entry point: `prove()` running abstract interpretation and producing verdicts; `component_for` names `proof:shape` for TypeError/AttributeError properties and `proof:interval` otherwise.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/state.py`
  - **Is imported by (tharkun.solve.absint)**: none
  - **Role**: CLI entry point (`python -m tharkun.solve.absint`) with `--property` and `--explain`.
  - **Imports (tharkun.solve.absint)**: none
  - **Is imported by (tharkun.solve.absint)**: none
- `src/tharkun/solve/absint/aval.py`
  - **Role**: Abstract value representation (`V`) combining intervals, float info, types, and zone references.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/zone.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/contracts.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/rand.py`, `src/tharkun/solve/absint/reducers.py`, `src/tharkun/solve/absint/sets.py`, `src/tharkun/solve/absint/state.py`, `src/tharkun/solve/absint/visitor.py`
- `src/tharkun/solve/absint/contracts.py`
  - **Role**: Demanded scalar and list-shape class invariants, checked independently of call contexts.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/state.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`
- `src/tharkun/solve/absint/errors.py`
  - **Role**: `Unsupported` exception definition indicating unmodelled Python constructs.
  - **Imports (tharkun.solve.absint)**: none
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/rand.py`, `src/tharkun/solve/absint/reducers.py`, `src/tharkun/solve/absint/sets.py`
- `src/tharkun/solve/absint/heap.py`
  - **Role**: Heap objects of the absint abstract state (`OutFileObj` [[internal ref]], `ListObj` with segmentation, `TupleObj`, `ArrayObj` subclassing `ListObj` with `typecode` [[internal ref]], `InstObj` with `qf`, `ClassObj`, `FuncObj` carrying `module_key` [[internal ref]], `BoundMethod`, `BuiltinMethod`, `RangeObj`, `SetObj`).
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/contracts.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/sets.py`, `src/tharkun/solve/absint/state.py`, `src/tharkun/solve/absint/visitor.py`
- `src/tharkun/solve/absint/interp.py`
  - **Role**: The absint interpreter (`Analyzer`) covering statements, expressions, calls, and loop machinery (`s_While` / `s_For`: first iteration, `_solve_fixpoint` with `_fixpoint` reuse, final pass, `fresh_pass`, `_visit_invariant`), builtin and list models, float terms, focus/blur, and site records.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/contracts.py`, `src/tharkun/solve/absint/errors.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/lemmas.py`, `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/rand.py`, `src/tharkun/solve/absint/reducers.py`, `src/tharkun/solve/absint/sets.py`, `src/tharkun/solve/absint/state.py`, `src/tharkun/solve/absint/visitor.py`, `src/tharkun/solve/absint/zone.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/__init__.py`, `src/tharkun/solve/absint/contracts.py`
- `src/tharkun/solve/absint/lemmas.py`
  - **Role**: Named numeric lemmas (`TRIG_PAIR`, `NORMALIZE`) and widening landmarks for float/numeric domains.
  - **Imports (tharkun.solve.absint)**: none
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/state.py`
- `src/tharkun/solve/absint/linear.py`
  - **Role**: Linear forms (`Lin`, `lin_bounds`, `IntInfo`) and symbolic integer bounds connected to zone DBMs.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/zone.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/rand.py`, `src/tharkun/solve/absint/state.py`, `src/tharkun/solve/absint/visitor.py`
- `src/tharkun/solve/absint/native.py`
  - **Role**: Finite builtin values for string/format/regex operations; never executes target objects.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/state.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/visitor.py`
- `src/tharkun/solve/absint/rand.py`
  - **Role**: `RandomMixin` implementing abstract transfer functions and argument binding for the `random` module.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/errors.py`, `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/state.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`
- `src/tharkun/solve/absint/reducers.py`
  - **Role**: `ReducersMixin` implementing abstract transfer functions for iterable-reducing builtins (`min`, `max`, `sum`).
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/errors.py`, `src/tharkun/solve/absint/state.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/sets.py`
- `src/tharkun/solve/absint/sets.py`
  - **Role**: `SetsMixin`: abstract set/frozenset allocation, membership, operators, hashing and method models, inherited by `interp.py::Analyzer`. `_via_rule` asks the contract rule to accept `visitor.py::AbsintVisitor`; `SetCtx.run` holds the transfer function and `serve_set_operations` installs the `visit_<kind>` methods. Constructors use builtin visits, so consumed cells are measured through `Rule.accept`. See `tests/test_absint_sets.py` and `tests/contracts/test_set_.py`.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/errors.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/reducers.py`, `src/tharkun/solve/absint/state.py`, `src/tharkun/solve/absint/visitor.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`
- `src/tharkun/solve/absint/state.py`
  - **Role**: Abstract execution state (`State`, `Frame`) and lattice operations (`vjoin`, `vleq`, `vwiden`, `ojoin`, `oleq`); includes `State.strip_terms` and `State.equiv` for equality fixpoint reuse tests.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/lemmas.py`, `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/zone.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/__init__.py`, `src/tharkun/solve/absint/contracts.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/rand.py`, `src/tharkun/solve/absint/reducers.py`, `src/tharkun/solve/absint/sets.py`, `src/tharkun/solve/absint/visitor.py`
- `src/tharkun/solve/absint/visitor.py`
  - **Role**: Contract-rule visitor (`AbsintVisitor`, `exact_int_value`, `hook_free_value`) translating tested contracts into abstract transfer operations; absent visits refuse.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/state.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/sets.py`
- `src/tharkun/solve/absint/zone.py`
  - **Role**: Difference-bound matrix (`Zone`, `ZERO`) domain tracking relational integer inequalities (`x - y <= c`).
  - **Imports (tharkun.solve.absint)**: none
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/state.py`

### External dependencies and consumers

- `src/tharkun/solve/contracts/adapters/absint.py`: `contract_for_builtin`, `raise_checks`, `simple_return_kind`, `output_handle_contract`: what `visitor.py::AbsintVisitor`, `interp.py::Analyzer.c_open`, `outfile_method`, and `s_With` read.
- `src/tharkun/solve/contracts/adapters/common.py`: `OUTPUT_HANDLE_METHOD_ARITY`, `BUILTIN_SHAPE_CONTRACTS`: shape contract arities.
- `src/tharkun/solve/contracts/adapters/math_model.py` and `src/tharkun/solve/contracts/rules_math.py`: `math_function`, `scalar_shape_raises`: what `visitor.py::AbsintVisitor.visit_math_function` and `Analyzer._math_call` read for the 57 `math` functions.
- `src/tharkun/solve/contracts/rules_builtins_more.py` and `src/tharkun/solve/contracts/rules_fileio.py`: built-in and file I/O contract specifications.
- `src/tharkun/solve/environment.py`: `open_effect`, `OUTPUT_MODES`: file system mutation classification and sandbox environment.
- `src/tharkun/solve/values.py`: `Value`, `FloatFact`, `IntFact`, `StrFact`, `Kinds`, `refine_compare`: scalar abstract domain and comparison refinement.
- `src/tharkun/solve/loop_family.py`: `classify_source_loop`, `classify_loop_family`: relational loop structure classification for widening step 4.
- `src/tharkun/solve/lowering/comprehension.py`: `lower_comprehensions`: AST transformation expanding comprehensions into explicit loops before interpretation.
- `src/tharkun/solve/lowering/provenance.py`: `path_of`, `ROLE_ATTR`: origin attribution mapping lowered nodes back to source locations.
- `src/tharkun/solve/pyresolve.py`: `Resolver`: name and import resolution used lazily by `Analyzer._extern_resolver`.
- `src/tharkun/solve/determinism.py`: `DET_MODULES`: allowlist of deterministic modules consulted during import checks.
- `src/tharkun/solve/extern.py`: read-and-recurse file finding for project imports.
- `src/tharkun/sv/prove.py`: static proof driver wrapping `tharkun.solve.absint::prove` for the SV pipeline (`src/tharkun/sv/engine.py::_static_proof`).

### Verified tests and harnesses

- `tests/test_absint.py`: target tasks that must be proven (`tests/fixtures/absint/targets/`) and mutants that must never be TRUE (`tests/fixtures/absint/never_true/`).
- `tests/test_absint_peel.py`: first-iteration loop peeling (`tests/fixtures/absint/peel/`).
- `tests/test_absint_fixpoint_reuse.py`: fixpoint cache invariance across loop solves.
- `tests/test_absint_float_range.py`: `nondet_float_range` bounds domain and float branch pruning.
- `tests/test_absint_reducers.py`: `min`, `max`, `sum` reductions over abstract containers.
- `tests/test_absint_random.py` and `tests/test_absint_call_shapes.py`: `random` models and argument shape checking.
- `tests/test_absint_kind_matrix.py` and `scripts/absint_kind_matrix.py`: kind-by-consumer matrix regression.
- `tests/test_absint_kind_refusals.py`: unhandled kind refusal assertions.
- `tests/test_absint_slice_source.py` and `tests/test_absint_bound_array.py`: array and slice store sound bounds.
- `tests/test_array_module.py`: `array.array` typecode, bounds, and indexing semantics.
- `tests/test_os_module.py`: three-way split of `os` and `os.path` operations.
- `tests/test_absint_str_getitem.py`: string indexing, slicing, and single-character domain.
- `tests/test_absint_isinstance.py`: abstract `isinstance` evaluation across class hierarchies.
- `tests/test_absint_dict_keys.py`: bounded integer key sets and constant dictionary lookups.
- `tests/test_absint_lowering.py` and `tests/test_absint_lowering_fuzz.py`: comprehension lowering interface and soundness fuzzing.
- `tests/test_absint_class_hints.py`: `check_type(value, C)` for a class of the program (`tests/fixtures/absint/class_hint/`), exhaustively checked against native runs.
- `tests/test_absint_class_in_loop.py`: a class statement that runs more than once is refused (an older instance read as an instance of the newer class).
- `tests/test_absint_bound_method_loop.py`: a bound method of an older function of a re-run `def`/lambda keeps naming that function.
- `tests/test_absint_draw_split.py`: case splits on a boolean draw (`tests/fixtures/absint/draw_split/`), exhaustively checked against native runs.
- `scripts/absint_fuzz.py`: differential soundness fuzzer against concrete CPython execution.
## Measured

See the PRs for the Real-world compares.
- Slice 1 proves `bm_fannkuch_nondet` data-lookup, dynamic-typing and arithmetic, and
  `bm_float_nondet` data-lookup.
- Slice 2 adds `bm_float_nondet` dynamic-typing and arithmetic.
- Each pair takes under 0.1 s of CPU.

## Ruled, not built

- **Next shape slice:**
  - dict key sets;
  - `random.choice` over keys;
  - tuple destructuring in `for` targets.

  Targets: nbody data-lookup and dynamic-typing.
- **Slice 3:**
  - the `array` module -- BUILT ([internal ref], above) for construction/indexing/slicing/`len`/
    `append`/`extend`/iteration and the fixed-width int + `'f'`/`'d'` typecodes; still ruled,
    not built: a `bytes`/`str` initializer, slice assignment, `insert`/`pop`/`remove`/`sort`/
    `clear`, in-place repetition, mixed-typecode operations, and the platform-dependent
    `'l'`/`'L'`/`'u'` typecodes -- none needed by a measured SV-Benchmarks use yet;
  - tuples and dicts of functions called with `*args`;
  - generators;
  - divisions guarded by a `raise`;
  - `((1 + a) * N) // N >= 1`;
  - a segmented-list invariant for non-unit-stride writes (`x[i*2]`-shaped), needed by
    scimark's inner loops independent of array support ([internal ref]'s own measurement, above).

  Target: scimark.
- **Slice 4:** whole-program object invariants. Targets: chaos, richards, raytrace, go.
- **Loop iteration and widening (ruled 2026-10-05 by the owner; fixpoint reuse and the first iteration + delay 0 are BUILT, "Fixpoint reuse" and "The first iteration on its own" above, thresholds and the explorer are not built; [internal ref], [internal ref]).**
  Designed after studying Mopsa's loop iterator (mopsa-analyzer `analyzer/languages/universal/iterators/loops.ml`
  and `analyzer/languages/universal/hooks/constant_widening_thresholds.ml`). Every loop is treated the same way:
  no shape-based special cases, no assumed termination (loop bible).
  - **First iteration on its own: BUILT** ("The first iteration on its own" above). `interp.py::Analyzer.s_While`/`s_For` analyse iteration 1 from the entry state,
    recording its checks, and starts the fixpoint from iteration 1's end state. If iteration 1 adds nothing to the
    entry state, that is the fixpoint. Exit = zero-iteration exit ⊔ head exit ⊔ breaks. A site's check is proven
    only if it is proven in BOTH the iteration-1 pass and the final pass (AND, never overwrite). The loop-head
    invariant handed to `capture_invariant` (witness.yml) is entry ⊔ head, since it must hold on every visit.
  - **Widening delay 0 by default: BUILT** (`WIDEN_DELAY = 0`, a number of plain joins; it used to be 3). Delays 0, 1, 2 and 3
    are measured on the full corpus above: 0 is the cheapest and loses nothing there.
  - **Thresholds.** A moving bound jumps to the nearest threshold that still covers the new value, and to ∞ only
    when none does. Thresholds are collected per variable, and per pair of variables, while the loop is analysed
    (including inlined callees): every evaluated condition comparing a variable with a constant (`x op c`), or two
    variables (`x op y + c`), with singleton values folded, adds c−1, c and c+1; 0 is always a threshold; at most 8
    per variable or pair. Applies to `state.py::int_widen`, `zone.py::Zone.widen` (unary AND difference entries)
    and `state.py::_len_widen`; floats keep `lemmas.py::widen_lo`/`widen_hi` landmarks. Soundness: every threshold
    covers the new value, so the widened state contains both arguments, and `back.leq(head)` still certifies the
    invariant. Termination: finitely many thresholds per entry; `MAX_ITERS` stays as a refusal.
  - **Targeted explorer.** When a property check stays unproven, find the loops it depends on (enclosing or
    preceding it on its path) and retry only those, smallest change first: two iterations on their own, then a
    higher widening delay, then whole-function and whole-program thresholds for that variable. Every retry is a
    full sound analysis; the first proof wins; when absint's budget is spent the answer is UNKNOWN.
  - **Fixpoint reuse (A4): BUILT**, see "Fixpoint reuse" below: the ruled key, equality and recording rule, plus the analyzer scratch the key had to include.
  - **Parked:** a cheap non-relational pass before the relational one. Sound (each pass is sound on its own), but
    no measured value.
  - Build order when a lane is free: fixpoint reuse (built), first iteration on its own + delay 0 (built), thresholds,
    explorer. Each ships only if the corpus gains.


## User operator and container methods

Ordinary single-inheritance classes can define arithmetic/reflected/in-place
operators, rich comparisons, item access/assignment, `__len__`, `__iter__`, and
`__call__`. Special lookup reads the class rather than the instance dictionary.
Dispatch follows subtype/reflected priority and same-type arithmetic rules;
unknown results, including `NotImplemented`, stop the proof rather than silently
skipping fallback behavior.

Every dispatch of a user instance is read through the rule of its protocol ([internal ref] step 2, `solve/contracts/rules_protocols.py`):

```
binary_special / special / iterable / _len_model / truth_split / attr_load
    -> Analyzer.visit_protocol(<qualname>, node, run, native)
    -> Rule.accept(AbsintVisitor) -> AbsintVisitor.visit_protocol_<name> (a ProtocolCtx)
    -> Analyzer._through_protocol -> run() (the unchanged dispatch, under a tap) -> _contract_checks(<seven classes>)
```

`protocol.binary_operator` (`a + b`, `1 + a`), `protocol.inplace_operator` (`a += b`: `__iadd__`, else the binary dispatch),
`protocol.comparison`, `protocol.container_getitem`, `protocol.container_len` (the call `len(x)`; the implicit length of a truth value is
`protocol.truthiness`'s and the probe of a consumer is `protocol.length_hint`'s), `protocol.iteration` and `protocol.truthiness` each record
the seven classes at the node of the dispatch as `protocol.<name>:<class>`, as the contract states them. A class the contract admits only
through the dunder (`propagates uncaught if __add__ raises C`) is proven at the dispatch because the dunder's BODY is analysed
(`interp.py::_PROTOCOL_BODY`): what it raises is recorded at the node that raises it, so the program is not TRUE for it unless that site is
proven. The native `TypeError` (no applicable method; `interp.py::_PROTOCOL_NATIVE` says what the dispatch argued) is proven when the
dispatch found support, and a failing check recorded at the node (an operator no method implements, a dunder called with the wrong
arity) VETOES it: the class CPython raises stays unproven and the other six are proven. The read of an attribute of an instance
(`protocol.attribute_access`) is on the hot path of every program with an object, so it is LEAN (`interp.py::_PROTOCOL_LEAN`): only the
classes left unproven are recorded (a clean read adds no site; the model's own `attr` check is the `AttributeError` record), and a class
that defines `__getattribute__` or `__getattr__` is refused where it is defined (`UNSUPPORTED_DUNDERS`), which is why no lookup hook can
run. An untested or withdrawn contract is the named refusal `protocol.<name> has no tested contract`. The protocols the analyzer still
refuses (`__contains__`, a custom `__bool__`, `__hash__`, the unary operators, `__int__`/`__index__`, `__enter__`/`__exit__`) are unchanged.
`tests/contracts/test_absint_protocol_rules.py`.

Each admitted operation uses the existing function transfer with its receiver
and arguments. Constructor fields and method effects remain in the existing
recency heap, with definite-field flags, alias-aware updates, type/shape joins,
and loop widening. An absent field, changed element type, or possibly failing
method body therefore remains a proof obligation. Ordinary class-attribute
writes use that heap too; protocol replacement and optional summary-class
attributes refuse. Incompatible heap joins return UNKNOWN.

`iter` over a modeled builtin iterable produces an iterator containing its source
summary. User `__iter__` must return such an iterator; `__next__` and generator
bodies remain unsupported. Iterator positions are overapproximated, so this
domain cannot prove exhaustion-sensitive claims. Collection construction and
extension execute applicable length hints; truth tests execute `__len__`. A length hint's own
`TypeError`/`AttributeError` is caught (see Soundness rules) and does not end the proof, matching
CPython's `PyObject_LengthHint`; a `len()` call written directly in source is not a hint and still
aborts the path on any raise, like any other unmodelled-propagation call.

A `__slots__` name whose class attribute is later reassigned (`Cls.x = 5`) has its slot descriptor
replaced outright: there is no instance `__dict__` behind a slot, so every instance's value for
that name becomes the (now non-descriptor) class attribute, and any value stored through the
vanished descriptor is unreachable. Reads resolve to the class attribute once it is set; writes
become an unconditional `AttributeError` (no descriptor to receive them, no `__dict__` to fall back
to) unless the model can show the class attribute was never reassigned.
Unmodeled nested comparison, slice-iteration, formatting, and callback paths
refuse, including when their errors lie outside the requested property.
Metaclasses, dynamic attribute hooks, and `setattr` remain unsupported.

`tests/test_absint_objects.py` contains independent arithmetic, container,
callable, alias, inheritance, class-field and loop examples, alpha renamings,
and failing variants. Iteration controls also request only ZeroDivisionError
and replay natively, guarding against misclassifying a method invocation as a
TypeError and then dropping its arithmetic effect.

Method binding includes keyword-only defaults, `**kwargs`, and expansion of
exact dictionaries with string keys. Unknown mapping protocols and positional
star expansion still refuse. A materialized summary receiver is included in
both positional and keyword may-alias arguments. Static/class-method
constructors refuse until their descriptor binding has an explicit transfer.

Recency reuse refuses when an earlier evaluated operand still holds the old
recent address outside the abstract state. These transient references survive
nested method and constructor calls until the enclosing statement completes.
Heap fields continue to use the existing redirects; they are not transient
roots. This conservative guard covers positional/keyword argument evaluation,
operators, subscripts and literal elements. Mutable module integer reads are
materialized before later arguments can rebind their names. Six native and
renamed negative controls exercise these evaluation-order cases.


### Inductive class contracts

A demanded receiver query can use a class contract checked independently of its
call context. The initial scalar slice proposes integer sign bounds from a
literal constructor, proves constructor establishment and preservation by every
plain receiver-only method, then caches the successful query transfers. Calls
recheck the receiver's field types, bounds and definite presence: external stores
and aliases cannot silently retain an invalid invariant. Class attribute changes
produce a different contract key.

Candidate proofs use the existing abstract transfers on an arbitrary receiver
satisfying the proposed facts. Any unsupported operation, unresolved exception
or failed preservation drops the candidate. Calls with arguments, inheritance,
global effects, receiver escapes, other field types and mutation effects use
the ordinary transfer until their required relational contracts are implemented.
Literal integer lists also propose non-emptiness and element-type contracts.
Constructor establishment, preservation through field/subscript writes, and live
alias-aware preconditions are checked with the same heap transfers. Queries must
be pure and return scalars; list identity tests refuse because the contract does
not establish distinctness. Multi-field relations and mutation summaries remain
with the ordinary transfer. Checked obligations appear in the proof-site output.

When static proof leaves an obligation open, the first existing concrete run may
report integer samples for fields read by receiver queries. The observer reads
plain instance dictionaries and skips custom dictionary descriptors; observation
failure does not alter the run classification. These samples are optional
`class_hints` inputs to absint. A suggested bound must still be established by
`__init__`, preserved by every method and satisfied by the current receiver.
Falsified or malformed samples never justify a TRUE result. SOLVE remains static:
the SV child owns observation, and the interpreter only checks proposed facts.
