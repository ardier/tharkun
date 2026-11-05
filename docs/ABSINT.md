# absint -- Tharkun's sound whole-program abstract interpreter

## Typed string-expression identity

When the program compares values for equality, pure builtin strings carry an
optional immutable typed recipe in `V.value_term`, separate from float lemma
terms and heap references. Variable binding snapshots unknown string values;
canonical str addition records its operands. Equality queries lazily hash-cons
those recipes and normalize ordered concatenation using SMT-admitted
associativity and empty-string identities. No user hooks, commutativity,
integer ring or IEEE reassociation law enters this string slice. `is` remains
separate from value equality. Exact singleton strings retain their existing
value equality and constant consumers.

A branch join uses one private choice token for all live frame values: equal
operand classes on both incoming branches remain equal after the join.
Different branch merges get distinct choices. Demotion drops value provenance from every scalar slot of the object becoming
a multi-allocation summary, so independent generated records never share one
string snapshot. Heap joins keep only common per-value provenance; widening
drops incompatible recipes. Value equality,
state ordering and fixpoint reuse include nonredundant provenance, so a new
draw cannot masquerade as an old one. There is no input length, arity or DAG
depth cap; the analyzer's existing deadline bounds query work and exhaustion
returns UNKNOWN. Generic loop induction, weak heap correlation, string methods,
bytes and other builtin theory plugins remain later precision work. Existing
scalar transfers keep their fast representation; neither egglog nor Z3 is
imported by this stdlib-only slice. The larger framework may use guarded Z3
for checks after cheap congruence; no runtime SMT fallback is installed here.

### Files and how the string slice relates

- `equality.py` owns typed recipes, generic congruence and the string plugin.
- `aval.py` carries optional provenance; `state.py` defines conservative
  per-value join/order/widen and correlated frame branch joins.
- `interp.py` drops scalar provenance at object demotion, records pure-string
  bindings/addition and consults congruence
  after the existing comparison callback and constant guards.
- `benchmarks/absint_equivalences/prove_rules.py` and `rule-proofs.json` retain
  Python-semantics SMT admission results, including all six rejected laws.
  Only congruence, string monoid and branch identities are installed here.
- `tests/test_absint_string_identity.py` compares source laws and failure twins
  against native CPython, including independent branches, mutation and loops.

## Assertion-demanded integer affine equalities

When assertions compare expressions involving multiple source names, the
existing integer difference domain carries a private exact affine equality
component. It retains arbitrary finite linear forms at assignments and list
length transfers instead of dropping sums to independent intervals. Canonical
integer and bool transfers retain their usual representation; no concrete
integers are enumerated and no float arithmetic is treated as exact integers.

Exact rational elimination projects forgotten variables. Joins take the affine
hull of both inputs, widening keeps equalities common to old and new states,
and inclusion checks the equality component as well as the existing DBM.
Exact constants and differences already established by the DBM are shared
before projection and hulls. Integer queries intersect the original bounds
with bounds of an equivalent reduced expression. Work uses the analyzer's
existing deadline, with no arity, coefficient or variable-count domain cap;
exhaustion yields UNKNOWN, never a partial safe proof.

Source assertions trigger tracking, not observed samples or proposed invariant
candidates. Programs without that demand keep the DBM path. Multi-variable
field summaries, symbolic string/sequence identity, pointer/type identity,
nonlinear and IEEE relations, variable-divisor arithmetic, indexed heap counts
and general fold induction remain named precision gaps for later mechanisms.
This slice changes no heap/State/value schema, call transfer, proof producer or
independent checker.

## Literal source export lists

Flat source stars may bind a literal list/tuple `__all__`, including private
names and duplicates, after its initializer runs normally. The consumer checks
one unconditional assignment, actual live bindings, resolver origins, and the
complete cached initializer closure for metadata changes or reflection. Empty
stars are authenticated even when no name refers to their module. Exported
values retain their existing identity; excluded names do not replace caller
bindings or builtins. Dynamic/nonisolated exports and relative/dotted/package
stars keep their existing conservative boundary. There is no export-count cut;
spent analysis budgets return UNKNOWN.

## String truth and guarded integer indices

An exact builtin string's truth test refines its immutable length observation:
true means at least one code point, false means zero. String integer indexing
uses the same length and the existing sequence-bound checker, so a nonempty
guard proves indices 0 and -1 safe and a nonnegative index below len(s) is safe.
The already evaluated receiver survives index-operand effects; later reads of
a changed binding have independent lengths. All finite Unicode lengths and
arbitrary int/bool indices are modeled without a machine-size cutoff.
Untracked/summarized provenance, custom protocols, and unresolved affine or
boolean guard relations keep their conservative result. Ordinary callable and
tested-contract checks remain mandatory; malformed keys and failure payloads
are unchanged.

## Native list truth and its current length

`truth_split` refines an evaluated exact builtin list's existing live length
cell: true means at least one item, false means zero. The receiver must denote
one recent object, so aliases share the refinement and mutations update it
normally. Element values never run hooks during this test. Every finite length
and element kind is covered; there is no size cutoff or global affine switch.
Older summary receivers, mixed values and custom subclasses retain the
existing conservative behavior. A returned `bool(xs)` does not yet retain its
relationship to `xs` after its branches merge.

## Known finite lists in native folds

`native.py::concrete` no longer rejects a known exact builtin-list length merely
because it exceeds eight. A single concrete element alternative is materialized
in linear work, with checks against the analyzer's existing CPU deadline before
and during construction. Finite integer alternatives and Cartesian products use
the same analysis deadline, without a 64-alternative ceiling. Tuple and dictionary
alternatives and contract callers share that deadline too. Optional expansion also shares
one 200,000-unit allocation/iteration work allowance across the analysis, rather than
renewing a per-operation allowance. Exhausting it discards the entire concrete result
and lets existing abstract transfers continue; it never returns a prefix. A larger
per-analysis allowance admits more work without changing an operand domain.
Singleton Cartesian pools are assembled in linear work. Deadline or memory exhaustion raises the existing
`Unsupported`, yielding UNKNOWN rather than a partial result or abstract fallback.
Unknown lengths, nonconcrete values, segmented lists, cycles and target hooks
retain their refusals; recursive builtin values and alias behavior are retained.
General abstract list lengths and scalar integer domains are unchanged.
Float singleton bounds at numerical zero cover both IEEE signs. Native
materialization includes both `0.0` and `-0.0` unless a `FloatInfo.term`
`("zero-sign", negative)` records a known sign from a source literal or concrete
native output. Unary negation flips that fact; joining different signs drops it.
`abs` records positive zero for either input zero sign. Known-sign zeros retain
one alternative, and nonzero floats keep their existing
path. Nested builtin values use the same rule and existing shared work budget;
exhaustion never selects one sign or returns a partial product. Universal
binary64 coverage and negation obligations are recorded in
`benchmarks/absint_equivalences/zero-sign-rule-proofs.json`.
If analysis stops with an unsupported construct or a spent parent budget, its UNKNOWN
diagnostics retain already observed unproven sites. Incomplete safe sites are omitted;
no partial safe records, complete-analysis flag or certificate is published.

Native callers retain their separate resource guards. `native_limit` counts
builtin container delimiters, separators and primitive representation cost
against its existing 4096-unit eager-fold work budget, including every rendered
occurrence of shared nested values. It has no container-length cutoff. Cheap
65/256-item tuples, lists and dicts, and 1024/1300-item zero tuples/lists, are
focused test points rather than semantic bounds. Cycles, traversal recursion or
memory exhaustion refuse; format width, integer representation cost, codec and regex
guards remain. The string consumer checks the existing analyzer deadline before
operand materialization, during resource traversal and between alternatives;
memory/deadline exhaustion discards all results. Native absence-of-exception
claims still require current tested, non-unknown contract claims. Long output
content may remain abstract; these focused folds claim no whole-program or
portfolio recovery.

Formatting does not equate integer magnitude or literal text digits with output
width. The same 4096-unit operand budget counts builtin representation cost;
separate accounting of bounded builtin fields counts actual aggregate output.
The stdlib format parser separates literals, names and item lookups from specs.
Static/dynamic width, numeric precision and integer base-output work are checked
before rendering each bounded field. String precision truncates rather than
expands output. Percent mapping keys and argument consumption retain native
semantics; the existing RegexFlag operator bit-work guard remains separate.
Nested replacement specs and attribute-field cost remain named UNKNOWN
boundaries. Closed builtin validation, deadlines and tested-claim authority
remain required; no target hook is called to estimate cost. Conversion errors
are left to the original native consumer, including conversion before nested
f-string specs with a test-local CPython decimal cap restored in `finally`.

## Pure source callbacks in verifier containers

`interp.py::_nondet_callback_summary` authenticates a module-level plain
zero-argument source function. Its body may contain any finite number of local
assignments, branches, returns, pass statements and docstrings. Expressions
are scalar literals, local reads, conditionals, authenticated verifier scalar
draws, or fresh records with literal string keys and scalar values. Every draw
is checked against its current binding. A local read must be definitely assigned
on every reaching path; branch continuations intersect their assigned names,
while an early return supplies no later bindings. A same-named global cannot
stand in for an unbound local. A locally shadowed callee or module
cannot authenticate as a verifier draw. There is no callback-body or container
length bound. Arbitrary calls, external reads/writes, loops, captured values,
nested mutable fields, dynamic record keys and other unproved forms refuse.

The ordinary function interpreter analyzes the body on a copied state. An
exceptional-only callback cannot eliminate the zero-invocation container path:
it refuses instead. Scalar results lose frame-local numeric correlations;
returned records immediately demote to existing heap summaries, so different
invocations cannot gain definite identity or strong cross-record updates.
The pure body cannot change caller state whether it runs zero, one or any
finite number of times. The invariant feeds both `nondet_list` elements and
`nondet_dict` callbacks; dictionary keys must be hashable scalars, and repeated
keys overwrite. Callback/hash errors inside try blocks remain refused.

`interp.py::_dict_union` reuses `_dict_unpack` for `left | right` on exact
builtin dict objects with non-float primitive keys and values. It allocates a
fresh result, copies left then right, and preserves uncertainty when an
unknown right key could replace a definite left entry. Neither operand is
mutated. Dictionary sizes are unrestricted; open mapping protocols, float or
object fields and in-place union remain UNKNOWN. No list comparison or key
membership relation is added by this change.

## Program stream buffers and facts ([internal ref])

`heap.py::StringIOObj` carries a `world.py::StreamObject` when its bytes and
cursor are exact. `interp.py::_new_stream`, `::_stringio_method_model` and
`::_print_to_file` project native StringIO/BytesIO operations onto those facts.
`write`, `print(file=)`, `getvalue`, `read`, `readline`, `seek`, `tell`, `flush`
and `close` preserve exact content/cursor when their operands are known. Joins
retain an exact stream only when both states agree; otherwise text contents and
positions widen conservatively. Unknown binary content/cursors refuse. Exact
builtin byte literals/results use a static bytes value; broader bytes protocols
and iteration remain unsupported. No user formatting or buffer callback is run.


The existing `io.BytesIO` factory contract retains its unknown raise obligations;
precise binary buffer facts do not invent a proof for those claims. Opaque bytes
remain top outside the explicitly supported stream, equality and length operations.

## File world facts ([internal ref])

`solve/absint/__init__.py::_prove_once` admits read-only builtin `open` to
`visitor.py::AbsintVisitor.visit_open_read`, using the tested `open.read` shape
in `contracts/rules_fileio.py`, then `interp.py::Analyzer.c_world_open`.
The union `open` contract and the write-only `open.output` shape retain their
own audit cells; the new consumer does not borrow an output contract for a read.
A file declared by `TaskEnvironment.files`
or an exact policy world object contributes its decoded text. An undeclared or
search-mutated object contributes arbitrary text; opening/decoding can also end
the path with OSError or a decoding error. Exception handlers around these reads
remain unsupported, so no proof silently drops their continuations. Write/read
programs and binary reads also remain UNKNOWN in this first text-file consumer.

`heap.py::ReadFileObj` tracks remaining text and closed states. `read` and
`readline` advance it, heap joins join both components, and `with` closes it.
Exact reads add an `Origin("absint", ...)` content equality only after checking
the original bytes. Search-only facts are never promoted to proof assumptions.
Finite string-to-int conversion now retains exact native integer values, which
lets the owner's prefix/parsed-integer/threshold example prove its declared-file
outcome. `lattice.py::FactState.meet` checks `Clauses.contradictions`; deeper
world contradictions are refused by `WorldObject.materialize`, and producer
failure/timeout is a refusal rather than proof of unsatisfiability.

The contract audit records `open.read` as consumed by absint. The union `open`
sample is now modelled through that shape rather than refused; its own union
contract remains unconsumed. Only these two audit files were regenerated with
`tests.contracts._frozen`; the existing union contract digest is unchanged.

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
  Full closure after widening uses row-indexed Floyd-Warshall. It considers every
  finite incoming/outgoing path for each variable and detects negative cycles;
  absent bounds remain infinity. This changes lookup cost only, retaining exact
  Python integer weights, every finite variable count and all derived bounds.
  Independent Bellman-Ford checks cover sparse/dense and disconnected graphs,
  contradictions, idempotence, sizes through 129 and weights beyond 128 bits.

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
  Nested functions and lambdas retain the fast snapshot path for immediate enclosing
  parameters which neither that scope nor descendants can rebind or delete. Otherwise,
  compiler metadata identifies each cell's lexical owner and the function references that
  existing live frame. A per-call activation number authenticates its lifetime independently
  of reused frame-depth names. Nonlocal reads, writes and handler cleanup share the owner's
  normal environment and zone; forwarded cells keep the original owner. Integer reads from
  cells, including parent locals exposed to nonlocal writes, are materialized before later
  operands can change the binding. Mutable referents retain their normal heap aliases.
  Compilation and bytecode inspection execute no target code. Every finite capture count
  and every existing value kind uses the same mechanism, governed by the run deadline.
  Implicit class cells still require a representation this model does not have. Expired
  mutable-cell owners, unfilled/deleted cells, incompatible owner joins, explicit deletion
  statements and reflective cell access remain UNKNOWN. Immutable snapshots can still
  outlive their original frame. Class-body functions keep the separate no-cell guard.
  Source-bound bindings and facts are withheld for live-closure bodies until the export
  schema represents lexical owners; every check, status and source site remains recorded.
  Existing body, string-content, non-singleton identity, scalar-summary and arithmetic precision
  gaps remain conservative; no numeric representation or input-size bound is changed.
  A capture-free lambda stored in a class has ordinary function descriptor binding: `C.f`
  leaves its arguments unchanged, while `C().f` prepends the instance. The exact-arity
  TypeError twin and variadic bound-instance counts are compared with native CPython.
  Attribute calls with starred arguments retain the separate lookup-order refusal.
  Identity with a known `None` result uses the actual evaluated operands: two known
  `None` values are identical, and any known non-None value differs from `None`.
  This applies to call results and both operand orders without invoking equality or
  truth hooks. Mixed/unknown values retain the existing conservative refinement;
  calls, effects, argument checks and tested contracts still execute abstractly first.
  Ordinary decorators evaluate in source order before defaults, then apply inside out through
  the existing abstract call/binding path. The name is stored after successful application;
  replacement values need not be callable. Function aliases, bound methods and callable
  instances use their existing transfers. Builtin descriptor syntax needs the existing source
  binding/reflection checks; mixing builtin descriptors with ordinary decorators remains UNKNOWN.
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

  Pure geometric integer loops additionally retain normal-exit facts without
  replacing those passes. `_geometric_loop_entry` authenticates an integer
  counter initially equal to 1, one doubling update (`r=r*2`, `r=2*r`, `r=r<<1`, `r=r+r`, or the augmented `r*=2`, `r<<=1`, `r+=r`: every spelling of
  "the counter becomes twice itself"), and the exact guard `r<n` (or `n>r`) or `(n//r)%2==0`. (Correction: the first implementation, by an AI lane (a5b9382b, Codex),
  was WRONG: it matched `ast.Mult` by the literal 2 and one comparison spelling, so `r << 1` lost the proof. The meaning is
  doubling, not the spelling.) Both operands must be
  existing distinct scalar cells; the bound is an integer/bool, or a float for
  the counter's power fact alone. Calls, attribute hooks, extra body effects,
  class namespaces, other seeds/guards and `else` suites keep ordinary analysis.
  The original guard/body contracts, exception checks and fixpoint budgets all
  run before `_geometric_loop_exit` records a fact. An exhausted budget gives
  UNKNOWN; no finite prefix certifies a loop.

  Input and result use fresh immutable zone snapshots in the existing integer
  value-term representation. A query must establish that the current scalar
  equals its certified result, and input-dependent queries must establish the
  same identity with the saved input. Equal intervals are insufficient. This
  proves the power mask `r&(r-1)==0`, integer ceiling bounds/minimality, and
  divisor/remainder/odd-quotient properties after normal exit. Counter/input
  reassignment and independently generated records cannot reuse a certificate.
  Float bounds, including NaN, confer no integer input relation or minimality.

  These are induction rules over all finite Python integer widths, not bounded
  unrolling or machine-bit identities. The admission script
  `benchmarks/absint_equivalences/prove_geometric_loops.py` checks binary-word
  base/step obligations and unbounded floor/mod arithmetic; its JSON includes
  rejected unguarded rules. Native tests include negative/zero/bool inputs and
  1024-bit values, plus observable 81-to-161-bit multiplication growth. Existing
  unsupported operations (including open protocols and unrelated bool ordering
  or zero/negative-mask precision) remain conservative; no general bitwise,
  nonlinear loop, termination, invariant-export or checker rule is claimed.
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

## Finite integer memberships

Stored singleton builtin integers may seed `V.int_values`, a finite set that
over-approximates their possible values independently of interval bounds and
immutable value identity. `join_into` uses the actual value's current bounds;
`vjoin` unions known alternatives, includes a plain singleton when appropriate,
and drops the coordinate for unknown alternatives or mixed kinds. Inclusion
checks this coordinate as well as the ordinary interval/zone facts. Arithmetic
creates a new value without it; same-value view changes, copies and recency
summaries preserve membership, without equating independent elements.
Seeding requires a possible supported integer comparison in the current source
scope. Demand is cached per AST scope under the run deadline; a length-only or
otherwise unobserved collection retains the original interval path.

Interval widening is unchanged. The membership coordinate retains finite
unions, so a stable list of links can keep excluding a sentinel even when its
interval widens. A growing set must still reach the ordinary verified fixpoint;
run-budget or convergence exhaustion gives UNKNOWN. There is no cardinality,
integer-width or input-size cutoff, and integer ranges are never enumerated.

After ordinary primitive comparison and contract checks, a bare-name/literal
integer comparison against a singleton can filter the actual memberships for
each branch and project their minimum/maximum into the existing zone. An empty
branch is infeasible. Work checks precede each alternative; no partial hull is
used after exhaustion. Effectful expressions, class namespaces, non-singleton
opponents and other kinds keep existing conservative transfers. This provides
ordinary index checks with facts such as `link != tail` implying a valid index,
without a queue recognizer, unrolling, a new heap shape or checker rule.
Admission: `benchmarks/absint_equivalences/integer-choice-rule-proofs.json`.

## Positional string snapshots

Known tuple items capture nonpoint exact builtin strings at construction when
existing value-equality demand is active. `new_list` creates a fresh immutable
term for each actual position after recency demotion, reuses existing terms,
and then applies the ordinary summary operation. A shared abstract `ANYSTR`
view never equates two independent draws. Literal singleton strings need no
extra term; other kinds and undemanded constructions retain their old path.

This makes repeated pointed reads, copies, slices and returned tuples available
to the existing string congruence and length observers. Tuple arity and string
length have no cutoff; each fresh capture checks the run deadline. Existing
demotion and nondeterministic callback summaries still erase scalar provenance
where independent values are merged. Nonpositional sequence summaries stand
for independent members and receive no common snapshot; mutable lists and
nonpoint index relations remain separate gaps. The existing authenticated positional
list retry also benefits when selected, with no change to its classifier or
retry policy. No new equality law, heap shape or checker rule is introduced.

## String length observations

For a captured builtin string, `_len_model` can retain the exact affine length
of its immutable `value_term`. `_string_length_form` walks concatenation recipes
iteratively, counts literal Python code points, and gives each opaque captured
value its own nonnegative length variable. Recipe keys stay alive during the
analysis so fresh draws cannot reuse an old identity. A changed binding has a
different recipe; a nontrivial join remains opaque. These observations use the
existing linear/zone representation and its optional affine equations, with no
depth, operand-count or string-size cutoff. The analyzer deadline applies before
each DAG step; exhaustion refuses the query without a partial length.

Tracking runs only on the recording path when existing source equality demand
and string provenance are present. The default numeric path, invariant iteration,
finite-string length bounds, non-string containers and user `__len__` calls keep
their existing transfers. Call-return, heap and loop provenance gaps still lose
precision conservatively. The current tested `len` contract is still required.
Admission covers a 21-bit sequence alphabet containing every Python code point,
including astral characters and isolated surrogates; UTF-8 byte counts do not
enter the model. See `benchmarks/absint_equivalences/string-length-rule-proofs.json`.

For example, a readonly closure adding captured `"tail"` to `"head"` retains an
`Add(literal("head"), literal("tail"))` recipe. Length projects this to `4+4`, so
its equality to 8 is proved. Content equality to literal `"headtail"` still loses
precision: the existing content congruence flattens concatenations and removes
empty strings, but does not fold adjacent literals. That rule is unchanged.

## Floats: value terms, focus, object invariants and lemmas (slice 2)

The numeric `float` constructor reuses `float_view` only when the existing
total-conversion guard accepts the operand. Float copies retain their facts
(including NaN/infinity and known zero signs); bool and finite bounded integer
conversion reuse the shared outward-rounded transfer. Integers that may overflow
and other inputs retain the prior conservative fallback. The overflow threshold
is CPython's conversion boundary, not a chosen integer-width limit. No-argument
`float()` yields known positive zero. Constructor chains have no length cutoff;
general IEEE arithmetic relations are a separate precision gap. The rounding
projection obligations and rejected NaN/zero-sign/integer-exactness shortcuts
are in `benchmarks/absint_equivalences/float-cast-rule-proofs.json`.

- **Exact dyadic terms** (`exactfloat.py`, V_TERMS): `float(i)` of an int with bounds inside +-2**53, a float literal such as 0.5, and `+`, `-`, `*` by a
  literal and `**2` of such values are the real number `(c + sum k*x)/2**s` itself while `|value| * 2**s < 2**53` (IEEE `+ - *` are correctly
  rounded, so a representable exact result is returned exactly): their result intervals are not rounded outward, a sum/difference keeps a term, and
  `==` between two terms is decided by the difference as an int linear form (the same real number is the same double; no NaN, -0.0 == 0.0). A
  conversion with no int variable to name gets a fresh atom. `x // x` and `x % x` of one value are 1 and 0 (the divisor 0 raises). Tests:
  `tests/test_absint_exact_floats.py`.
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
  values is an impossible branch. A pure float name in the current frame retains the refined fact when
  compared with a numeric literal. Other operands remain unchanged: an effectful comparison cannot
  refine a stale name read, and global reads inside callees are not current-frame bindings. Int
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

## Pure integer callback folds

`ReducersMixin._integer_reduce` consumes `functools.reduce` through its existing
Rule visitor. It proves a type invariant for an arbitrary number of elements,
without removing the census call site. Sources must be exact builtin integer
lists/tuples; an optional positional or keyword `initial` must also be an exact
integer. A missing initial retains the TypeError obligation unless nonemptiness
is proved; a certainly empty source with an initial returns that initial.

The callback must resolve to one plain source function or lambda with two
positional parameters. Its only body is a parameter-only expression built from
integer literals, unary `+`, `-`, `~`, and binary `+`, `-`, `*`, `&`, `|`, `^`.
No other names, calls, attributes, subscriptions, statements, decorators, open
hooks, floats, bools or iterator operands enter this transfer. Existing generic
contract handling remains the fallback outside that domain.

All integers contain the seed and are closed under these operations. The
ordinary abstract function-call machinery checks the callback on two arbitrary
integers; the result is an unconstrained integer, with no invented equality to
another fold. Parameter-only primitive expressions cannot mutate the iterable
or observe changing callback state. The tested registry contract, return and
raise claims gate this precision; withdrawal removes it. Tests in
`tests/test_absint_reduce.py` cover the invariant, native raising callbacks,
binding, empty sources, aliasing and contract withdrawal on CPython 3.14.4.

This recovers local absint arithmetic/data-lookup proofs of the public
`type-eval-py/builtins_functools` example while its census remains nonempty.
It supplies no model for `groupby` cursor sharing or non-scalar verifier
callbacks; those examples can correctly remain UNKNOWN.

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
- assertion records also carry `type_sets` and `entry_scope`: all represented kinds and
  must-facts before evaluating the assertion. Their `scope` describes the possible failing
  state, and `residual` names the failing condition. Repeated visits union the kinds and
  intersect the must-facts;
- `context`: the inlining call string;
- `loop_iteration`: `"any"`;
- `source_sha`.

`AbsintResult.complete` is True only when the analysis ran to the end. Then the records cover
every operation of the program that can raise the property's classes, because absint aborts on
anything it does not model. Lane A solves `may_raise` records for a witness, and lane B refutes
them for a certificate.

The SV run publishes these records in its existing `ObligationTable`. Guard-value seeds
solve open failure conditions together with their site bounds and relations; comparison
and boundary seeds use the failure intervals' endpoints. A proved assertion gets no
concolic solve marker. Bounds constrain these site queries only: a reassigned local is
not an entry draw, unrelated sites keep separate domains, and enumeration retains its
full input domain. A spent guidance budget supplies no further seeds and proves nothing.

## Soundness rules (absolute: a wrong TRUE costs 32 points)

An indexed store is strong only when the entire receiver value denotes one
recent object. Selecting a singleton subset of lists, dicts or arrays from a
mixed receiver does not establish which object Python will mutate. Dictionary
constant-key updates and list/array element updates all use the whole-receiver
guard; ambiguous alternatives keep their old values possible.

- **Every unmodelled construct aborts** (`interp.py::Unsupported`):
  - `with`/`match`, generators, unsupported lambda forms;
  - comprehensions the exact lowering refuses (below: the lowering turns the others into loops
    before the analysis runs, so only a refused shape aborts);
  - expired mutable-cell activations, unfilled/deleted cells, incompatible cell-owner joins, unavailable or ambiguous lexical scopes;
  - `getattr`/`setattr` and `eval`/`exec` outside the finite module-source lowering
    ([LOWERING.md](LOWERING.md), "Finite module code sources");
  - any library function outside the model table (`interp.py::MODULES`);
  - multiple inheritance, metaclasses, unrepresented descriptor/decorator composition;
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

## Refusing per check instead of per program (decided not to build: 2026-11-05, [internal ref])

The owner agreed in principle to replace "any unmodelled construct aborts the whole analysis" by "model the construct as havoc, keep analysing, prove the checks the havoc cannot reach".
This section records what such a havoc must do to keep "never a wrong TRUE", and the measurement that says what it can and cannot decide. The decision (2026-11-05, [internal ref]): the havoc is NOT built -- the measurement below shows no score gain and the wrong-TRUE risk of an unsound havoc; this section stays as the record of what a sound one would have to over-approximate and of the typed-opaque-values alternative. Nothing here changes the analyzer.

**What a sound havoc must over-approximate** (each item is an existing invariant of this analyzer, see "Soundness rules" above):
1. *Heap and aliasing*: every object reachable from the construct's operands, from any global, and from any mutable cell may change contents, length and class-visible attributes; recency/summary objects it
   could touch become unknown summaries (weak, unbounded) and any symbolic bound or zone fact that names them is rewritten away (`State.rewrite_bounds_away`) before the havoc.
2. *Locals*: only names the statement rebinds (and cells, below) change; a scalar local the statement cannot rebind keeps its fact.
3. *Cells and closures*: every mutable cell owned by an activation the construct can call into becomes unknown; an expired activation stays unreachable only if nothing unknown can hold it.
4. *Exceptions*: the construct may raise any exception class; an uncaught one ends the program, so it is a potential violation of EVERY property class unless a contract summary bounds it, and
   inside a `try` it must be routed to the handlers with the pre-state havoced (today: refused by name, "static callable raises are not routed").
5. *Callbacks*: an unmodelled callee may run any user function it can reach (a callback, a `__len__`, a `key=`) with arbitrary arguments: the sites inside those functions must be reported as unproven too.
   Analysis by inlining never sees them, so a havoc needs an independent inventory of all sites (the census, `census.py::site_census`) and every inventory site not matched by a recorded check is unproven.
   `AbsintResult.complete` no longer implies "the records cover every operation": that implication holds only because absint aborts on what it does not model.
6. *Time*: the havoc happens inside loop fixpoints too; it must be monotone and idempotent on the widening path.

**What it can decide, measured.** The frozen held-out suite (`tests/heldout/BASELINE-20261105.md`, branch 1519/heldout-suite) on main 28467f87: 82 TRUE pairs of the originals are UNKNOWN. In 21 absint ran to the end and a check
is genuinely unproven (18 assertions, 3 index/key/type sites): a havoc changes nothing there. In 61 absint aborts with `Unsupported`. By reason: `type.list` / static callable raises inside a `try` 7 (+1 `nondet_int` mfunc), `len` of an unknown value 7,
captured closure cells 6, expired closure activation 4, subscript store on an unknown value 5, `.most_common` 4, enumerate over an unmodelled callback 4, comparison with an unknown value 3, `str.__contains__` contract 3, subscript of an
unknown value 2, call of an unknown value (`next`, `ChainMap`, `MyClass`, ...) 5, `.func` / `.move_to_end` / `.append` method on an unknown value 4, `__hash__` class 1, class decorator 1, truth value of an unknown value 1, `*args` 1, others 3.
The 16 such cases of the lane-1519 library-style programs were read one by one: in every one the unmodelled construct produces or mutates a value that a property check of that pair then reads (a deque or OrderedDict
guard, a Counter lookup, a closure-built divisor, a property setter's clamp, `super().__init__`'s attributes). The slicer (`solve/slicing.py`) already removes statements no check depends on, so what is left feeds a check.
Under a sound havoc such a check reads an unknown value and stays unproven, so the verdict stays UNKNOWN: **per-check refusal alone is expected to decide almost none of these 61 TRUE pairs**; it pays only where an
unmodelled statement lies on no dependency path of a check and the slicer kept it conservatively, or where a contract summary bounds what the construct can raise.

**What would decide them: typed opaque values (proposal, needs the owner's go).** The result of a stdlib constructor the contracts layer knows (`deque`, `OrderedDict`, `Counter`, `defaultdict`, `ChainMap`, `collections.abc` views) is an
opaque heap object of a KNOWN class with unknown contents: a havoc whose class is known. Its methods are served by the contract rules (`solve/contracts/adapters/rules_collections.py` and neighbours: arity checked, which property
classes each method can raise under which argument conditions), its mutations are weak updates of the unknown contents, and `len`, truth, iteration and `in` of it return unconstrained non-negative / bool / element values with
exactly the contract's raises. That decides the sites a class's contract proves safe (a Counter or defaultdict lookup never raises KeyError, `len` of a deque cannot raise TypeError, `deque.append` cannot raise on any argument) and
leaves the data-dependent ones (`OrderedDict.move_to_end` of a possibly absent key, `popleft` of a possibly empty deque) unproven. Its soundness rests on the contracts being exact over-approximations (they are pinned against CPython
by `tests/contracts/`). Closures, `try` routing of static callable raises and expired cell activations are separate modelling tasks (they are the next three rows of the blocker table).

## Comprehensions: the exact lowering ([internal ref])

`interp.py` has no model of a comprehension. `prove` therefore runs `solve/lowering`
(`lower_comprehensions`, [LOWERING.md](LOWERING.md)) on the sliced program first, builds the `Analyzer`
on the lowered tree, and puts every report back in terms of the original program. The lowering is the
SAME program (same value, exception and order of effects), so a proof of the lowered program is a proof
of the original. It refuses what it cannot make exact: stored or partially consumed
generators (`next`, `min`, `max`, `zip`, ...), unadmitted numeric/protocol consumers, `*args`, `**kwargs`, f-strings, assignment expressions, comprehensions
in a loop test, a `with` item, a default value, a decorator, an augmented assignment, a lambda or a
class body, and any program that can reach its own namespace (`locals()`, `globals()`, a frame
attribute). A refused comprehension stays in the tree and absint aborts on it as before.

### Immediately consumed generator expressions

The comprehension normalizer also handles stable all/any/list/tuple consumers,
integer-only native sum and one pattern-unpack target; see [LOWERING.md](LOWERING.md).
all/any stop every nested loop, and unpack consumes only the source arity plus
one unless starred. Per-step StopIteration handlers preserve generator exception
conversion without catching errors from consumer truth tests. Absint permits
primitive power, bool and isinstance only across these structurally verified,
lowering-owned boundaries; user handlers, finalizers and unknown protocol calls
keep their existing conservative gates. The independent witness checker has not
acquired generator syntax support through this change.

Findings are about the original program:

- a check lands at the line and column of the original node;
- a fact about a fresh loop variable is restored to the source's name only inside its own
  comprehension, and dropped elsewhere; a fact about an accumulator or temporary is dropped;
- no invariant is captured at a statement the lowering invented or names a variable of the lowering.

`tests/test_absint_lowering.py` pins the seam, including the case where a leaked loop variable would
make an unsafe division look safe, and `tests/test_absint_lowering_fuzz.py` runs every program absint
calls TRUE for real.

### Assignment expressions in while guards

When the source contains `:=`, `prove` also applies
`solve/lowering/walrus.py::lower_walrus_loops` to the origin-aware analysis tree.
For a function-local target, `while prefix and (name := value)` keeps the prefix as its loop condition, then
assigns `name`, breaks if it is false, and executes the original body. A sole
`while (name := value)` uses a true loop condition. An `and` prefix may have any
number of operands. Evaluation order, short-circuiting, assignment before the
truth test, and original body `break`/`continue` behavior are preserved. No
temporary variable, scope, input-size bound or numeric representation is added.
The loop, body and moved operands retain their node identity and source origin;
introduced assignments/guards stand for the original assignment expression.

Module/class scopes, global/nonlocal targets, targets mentioned in nested scopes,
namespace reflection, any declared finalizer, a while-else suite, an assignment expression elsewhere in the guard, and other
assignment-expression contexts remain unchanged and reach the analyzer's general
unsupported-expression UNKNOWN rule. The normalizer does not run on source
without `:=`. `tests/test_lowering_walrus.py` compares native effects, exceptions,
binding and truth tests over sizes 0,1,2,3,17,65,129, including custom truth
protocols and integers wider than the machine representation, and checks original
site reporting and an unsafe assertion. Class namespace interception and finalizers
that rebind globals/closure cells are native refusal controls: re-reading a target
in those contexts need not yield the value of the original assignment expression.
A finalizer can also observe the extra RHS reference held by the original assignment
expression during its store, so programs declaring `__del__` are not rewritten.
The independent source checker and proof finalizer retain their existing admission rules.

## Import boundary ([internal ref]): read-and-recurse, not allow-by-name

An absint TRUE used by SV carries the raw-byte hashes of every source module
and package initializer its resolver actually parsed, plus the module names,
ordered roots and paths used for resolution. Child resolvers share this evidence;
inert initializers are included even when they need no interpreter execution.
Hashes are captured at the read, not after the proof returns. Conflicting reads
or lookup results during a proof refuse. The SV engine hashes the exact primary
bytes it decodes and passes to absint using the existing PEP 263 semantics.
Before exporting TRUE, the scheduler rechecks these bytes, import resolutions,
compiled/bytecode competitors and the existing native-import blocker. A change gives
UNKNOWN before witness export. This does not route regular packages through
the product prover's intentionally narrower flat-source shortcut; its existing
independent closure check remains in place. Proof rules and domains are unchanged.

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
Under a `try`, a builtin method whose raises are not routed is refused (`Analyzer._try_routes_builtin_raise`); `interp.py::_TRY_METHOD_ALLOWLIST` adds the one exact shape
`xs.append(v)` (one positional argument, no keywords, a direct call on a list, not a bound alias), whose `never` contract is proven for any value.
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

## Definite-key dictionary get

`Analyzer._dict_method_model` uses the existing `dict_load` presence proof to
exclude `dict.get`'s fallback when the key is definitely present. This retains
the selected value and its shallow aliases, including dataclasses-json's
`field.metadata.get('dataclasses_json', {})` mapping. All key/default argument
expressions still run before lookup. Uncertain presence keeps the previous
value/default join; a definitely absent key keeps the existing default result.
The added precision requires the current tested return contract and excludes
float values. No lookup, membership, mutation, callback, census or state
mechanism changes. `tests/test_absint_dict_get_present.py` covers aliasing,
argument effects/exceptions, uncertain keys and return-contract withdrawals.
The direct metadata check is bounded; no complete schema factory is claimed.

## Dict keys from the zone ([internal ref])

For an exact builtin `DictObj`, `Analyzer.truth_split` uses the existing
`dict_len` bounds: a must-key proves nonempty, an exact empty map proves false,
and unknown cardinality retains both branches. This makes the existing `and`,
`or`, `not` and conditional machinery select the correct mapping/effects,
including dataclasses-json `mm.py:284`'s `field.metadata or {}` fragment.
Values' truth hooks are never invoked; user/subclass truth hooks retain their
existing handling. No key-presence relation is inferred from an otherwise
unknown nonempty map, and no heap, state, comparison or callback mechanism
changes. `tests/test_absint_dict_truth.py` checks native order/alias/callback
controls and positive/negative proofs. The later `get` selector in that source
is a separate consumer; this fragment check is no complete factory proof.

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
hash/equality keys remain refused. Closed dict unpacking is described below. For a non-strong subscript store, each definitely-present key that the
incoming scalar key may equal keeps its presence but joins the incoming value; the open-key/value summary is updated as well.
The tests are `tests/test_absint_dict_keys.py` (native key semantics plus a differential over named, looped and drawn keys,
with a bug injected into the key set caught).
`go`'s next blocker behind it was `set()` (line 147): absint had no set heap kind (built by [internal ref]: "Sets", below).

### Direct primitive dictionary items loops

`Analyzer.s_For` admits one builtin items view of one exact `DictObj` as a
direct statement iterable. `iter_elem` constructs its two-field tuple from
the current dictionary at each loop entry, using the existing length, key,
value and tuple-allocation mechanisms. Keys and values must be primitive and
non-float. An exact empty dictionary has no element; unknown cardinality
never gains a definite key or key/value identity relation.

The view remains live: later value mutations join into subsequent elements,
while rebinding the dictionary name does not retarget an existing view.
Snapshotting values before the loop would be unsound and is covered by a
native counterexample that changes a later integer value to a string.
The global `check_iterable` refusal and `elem_summary` stay unchanged, so
explicit iterators and generic materializers gain no items-view admission.
Existing sorted-items behavior is unchanged. Active `try` loops retain the
existing refusal for unmodelled RuntimeError exits from size mutation.

This covers the statement shape `for k, v in kvs.items()` in dataclasses-json
`core.py:98`. It does not establish inversion or safety of the owning encoder:
membership/read-back and configured callbacks are separate consumers.
`tests/test_absint_dict_items_for.py` checks native order, aliases, live
mutation, subclass callbacks and positive/negative statement proofs.

### Closed primitive dictionary setdefault

`Analyzer._dict_setdefault_model` consumes the existing tested `dict.setdefault`
rule through the method visitor. The domain is a recent exact builtin dict,
any int/bool/str/None key summary and primitive non-float fields/defaults.
The existing single concrete tuple-key insertion path remains supported.
The default expression is evaluated by ordinary analysis even when the key is
present. Reference-bearing defaults remain refused because discarding the
unused temporary may run cleanup code. Float/custom keys, summary receivers,
stored bound methods and builtin exceptional continuations retain named refusals.

For one constant key, a definite existing entry returns its value without
overwriting it. Otherwise the value joins a possibly present, colliding entry
with the default and the existing strong subscript store guarantees that key
after success.
Possible unrelated keys remain possible; no key-identity relation or new
state field is introduced. With an uncertain key, every definite existing entry
is preserved; the return joins matching existing values and any possible default,
and only possible insertions enter the open key/value summaries. A finite set
of already-present keys cannot insert the default. The existing summary does
not track that a selected open key is now present: reading back through that
key can still be UNKNOWN. No cardinality bound is imposed; finite-key expansion
is only a precision choice, and a spent run budget remains UNKNOWN.
Binding stays positional only, and current registry
contract/return/raise withdrawals remove the corresponding proof.

The unchanged call in dataclasses-json `mm.py:301` supplies a `None` default
for optional-field options. The direct check covers bounded scalar options;
it does not prove the complete schema factory or callable defaults.
`tests/test_absint_dict_setdefault.py` covers native argument order, discarded
default cleanup, numeric collisions, hash/equality propagation for all seven
property exceptions, and positive/negative transfer and withdrawal checks.

### Closed dictionary display unpacking

`Analyzer.e_Dict` and `_dict_unpack` model `{**left, key: value, **right}`
through the existing `DictObj` fields. Computed keys and operand calls go through
ordinary `Analyzer.eval`, retaining their authentication, effects and exception
checks. Each expanded
value must identify one exact builtin dict, with int/bool/str/None keys.
Open mapping protocols, subclasses, custom hash/equality, floating keys and
unresolved/alternative receivers remain refused; no callbacks are skipped.

Operands retain source order. Later definite entries replace earlier values;
possible later keys preserve earlier key presence and join any colliding value.
Possible-only keys never become definite. Open keys/values remain summarized,
without an invented length bound or cross-dictionary key-identity relation.
Copied references are borrowed against allocation demotion and remain shallow
aliases; changing the new dict's entries does not change the source dict.
An unpacking display keeps its partial result in an existing pinned `DictObj`;
after each operand it reads that live object again. Thus ordinary calls may
mutate source mappings or shared nested values without retroactively replacing
an earlier copied entry, losing temporary lifetimes or reusing stale Python
locals. Pin shape changes are refused. Effectful displays inside an enclosing
handler scope stay refused because that exception continuation is not admitted;
handlers entirely inside an operand use the ordinary call machinery. Existing
finalizer/class refusals remain intact.
No new contract, census exception exemption, state field or checker is added.

The actual public `type-eval-py/dicts_merge` source stops at unpacking on the
base. With this transfer its abstract analysis completes, but data-lookup is
still UNKNOWN at later reads indexed by keys iterated from another dict. That
relational consumer gap remains explicit; no full-property gain is claimed.
`tests/test_absint_dict_unpack.py` records native CPython 3.14.4 overwrite,
alias, collision, evaluation-order and exception boundaries plus positive and
negative proofs. In particular, DICT_UPDATE converts an AttributeError from
the mapping/key merge into TypeError; open callback shapes remain refused.

Sources: [Python 3.14 dictionary displays](https://docs.python.org/3.14/reference/expressions.html#dictionary-displays)
and [CPython v3.14.4 dict implementation](https://github.com/python/cpython/blob/v3.14.4/Objects/dictobject.c).

The computed-key follow-up is source-matched to dataclasses-json `mm.py:363`:
its unchanged dictionary display includes `f'make_{cls.__name__.lower()}'` and
`**schema_`. On the previous transfer that display is refused by the syntax
gate; with source-defined `Record` and closed scalar bindings it proves the
selected key/value results locally. This is a consumer demonstration, not a
whole schema-factory, portfolio or checked-witness recovery claim. Source and
property names do not participate in production dispatch.

### Closed primitive dictionary updates

`Analyzer._dict_update_model` consumes the existing tested `dict.update` rule
through the method visitor. It accepts a recent exact builtin receiver, zero
or one exact builtin mapping argument, and keyword entries. Both mappings have
int/bool/str/None keys and primitive non-float values; reference-bearing values
remain refused because replacing their last reference can run cleanup code.
Summary receivers, open mappings, pair iterables and stored bound methods stay
refused. Argument expressions still run through ordinary analysis before the
transfer; builtin exceptional continuations retain the existing refusal.

The unchanged `_dict_unpack` helper preserves definite versus possible keys,
joins possible overwrites and retains scalar numeric-key collisions. Keyword
entries use the existing strong subscript store after the mapping update.
Object aliases see the mutation, source mappings remain independent, and the
tested return is `None`. Withdrawing the contract or its return proof removes
the transfer; unknown raise claims remain unproved. No new state, contract,
callback admission or cross-dictionary relation is introduced.

This covers scalar configuration updates shaped like dataclasses-json
`core.py`'s class override merge; it does not prove the complete
dataclass factory or reference-valued encoder/decoder configurations.
`tests/test_absint_dict_update.py` checks native precedence, aliasing, argument
order, partial pair failure, cleanup and all seven callback exceptions before
the positive and negative consumer checks. Unlike dictionary-display unpacking,
method `dict.update` propagates callback `AttributeError` unchanged.

### Atomic deepcopy leaves

`AbsintVisitor.visit_copy_deepcopy` and the `_copy_deepcopy_atomic` helpers
consume the existing tested rule for one positional int/bool/str/None value.
CPython 3.14.4 returns these atomic inputs unchanged before consulting a memo
or copy/pickle hook. The transfer retains the captured argument's abstract
value and existing integer relations, gated on its current tested return
claim; all seven raise claims still pass through the normal contract model.
Argument expressions and callee authentication retain ordinary analysis.

Containers, cycles, custom objects, floats, bytes, keywords and provided memos
keep the generic path. The existing refusal inside `try` remains. This is no
recursive-copy or general alias model and adds no state, contract or census
metadata. `tests/test_absint_deepcopy_atomic.py` checks native identity,
memo/hook/cycle boundaries, aliases, binding, effects and withdrawals, while
the existing list-copy tests retain their separate transfer.

The unchanged `copy.deepcopy(obj)` leaf in dataclasses-json `core.py:338` is
checked for bounded atomic inputs. No complete `_asdict` or serializer result
is claimed. The `copy.deepcopy` audit cell was already consumed through its
generic visitor and retains its existing frozen row.

The copy contracts use the broad return type `object`: hooks and supplied memo
cache entries may replace the result with an unrelated type. Atomic identity
is the closed transfer's native argument proof, gated on the tested return
claim. A supplied deepcopy memo is declared in the existing `mutates_args`
field; memo callbacks remain open, and malformed get/keep-alive operations can
raise native AttributeError or TypeError without invoking an object hook. The
atomic transfer explicitly rules out that memo access before upgrading the
native AttributeError claim. `tests/contracts/test_copy_memo_boundaries.py`
backs the mutation, cache, hook, binding and malformed-memo boundaries. This
does not admit a supplied-memo transfer or a recursive copy model.

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

## str methods with hook-free arguments (contracts-r1)

`AbsintVisitor._abstract_string` (the visit of every `str.*` rule when an operand is not finite: a `nondet_str()` receiver) proved only the
zero-argument methods. `s.startswith(p)`, `s.split(",")`, `",".join(xs)`, `s.replace(a, b)` left all seven classes of the rule unproven,
because the rules (`rules_str.py`) say what is true of them in general: the position arguments (`start`, `end`, `maxsplit`, `count`, `width`,
`tabsize`) are read by a USER `__index__`, `join` iterates a user iterable, `keepends` is read by `__bool__`, and `TypeError` is the call's own logic.
So a program that used one was never TRUE for `arithmetic`, `data-lookup` or `dynamic-typing` (36 pairs of `type-eval-py` on the pending set).

A call is **admitted** when the receiver is a definite builtin `str` (`V.s` and nothing else) and every argument has exactly the kind the method takes.
`contracts/adapters/str_shapes.py` is the engine-neutral half: `SIGNATURES` (the parameters after the receiver, which are optional, which may be keywords in CPython
3.14), `bind` (the call as CPython's argument parser binds it, or `None` for a call that does not bind), and the kinds: `s` a str, `S` a str or a tuple of str,
`c` a str or None, `i` an int or bool, `n` an int, bool or None, `L` a list or tuple of str. `visitor.py::_str_kind_ok` maps a kind to an abstract value, and
accepts nothing else: not `top`, not a float where an int is taken, not a static object, not an instance of a program class, not an `array` or set where a list is
taken. An admitted call proves all seven classes (`contract+model: ... no user hook runs`), for these methods: `startswith`, `endswith`, `count`, `find`,
`rfind`, `index`, `rindex`, `split`, `rsplit`, `strip`, `lstrip`, `rstrip`, `replace`, `removeprefix`, `removesuffix`, `join`, `zfill`, `splitlines`, `expandtabs`.
Anything else (a call that does not bind, a wrong kind, a union with an object, `startswith("a", start=1)`, `s.split(",", sep=",")`) keeps the contract's claims, exactly as before.

- **join.** The argument is a list or tuple whose element summary is `str` only (`ListObj.elem`, `pre` of a segmented list, or the items of a tuple). The summary is
  universal and the analysis keeps it true across aliases, calls, stores in containers and objects, index and slice assignment, `append`/`extend`/`insert`/`+=`/`+`
  (`tests/fixtures/absint/str_methods/sm_join_*`: each writes a non-str by one of those ways and CPython raises `TypeError`, and no property is TRUE for it). A str, dict, set, generator
  or iterator argument is valid in CPython and is refused here; so is an element of a class with a builtin base (`class S(str)`: refused where it is defined).
- **Results.** `startswith`/`endswith` are `bool`; `count`/`index`/`rindex` are at least 0 and `find`/`rfind` at least -1 (`INT_RESULT_LOWER`); `split`/`rsplit` with a
  str separator return a list of at least one part (`"".split(",") == [""]`), without a separator (None) the list may be empty (`"".split() == []`).
- **Non-property raises.** `ValueError` (empty separator, absent substring), `OverflowError` (`maxsplit`, `count` and `width` at 2**63, `tabsize` at 2**31) and `MemoryError` (a `zfill`
  or `expandtabs` width the result cannot be allocated at) are not property classes: `NON_PROPERTY_RAISES` names them per method and `tests/contracts/test_str_shapes.py` measures them on
  CPython 3.14.4. Outside a `try` they end the run, so the proof goes on past the call; **inside a `try` the rule call is refused** (`Analyzer.visit_rule`:
  "a contract rule: its non-property raises are not routed"), so `try: s.zfill(n) except Exception: 1 // 0` is never TRUE. `start`/`end` clamp and never raise.

**Safety template.** `solve/contracts/safety/rules_str.py` states the `TypeError` claim of each of these methods as cases over the exact scalar tags of the arguments (the
receiver a `str`; one case per assignment of `int`/`bool`/`float`/`str`/`NoneType` to each argument; a call raises iff some argument's tag is one its parameter does not take,
`splitlines(keepends)` taking all of them). It is registered in `safety/__init__.py::_case_builders` and `tests/contracts/test_safety_str_templates.py` holds it to CPython for
every tag assignment of every arity. The record schema binds operators only, so no site record of a method call reaches it yet; it is the half of "feeds witnesses with no rework"
that belongs to the contract.

`tests/contracts/test_str_shapes.py` is the CPython three-way test (the outcome of the real method, the rule's `native`, and the table, on both sides of every parameter's
boundary, plus a user hook raising each of the seven classes in every position that reaches one); `tests/test_absint_str_methods.py` runs 86 adversarial programs natively
over every combination of a small set of draws (2**31 and 2**63 included) and holds absint to what they do.

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
  Exact builtin tuple destructuring is admitted when every alternative has a recorded item list of
  the target's arity: iteration has no callback and the arity check cannot raise `ValueError`.
  Nested targets are checked at each level, and target assignments still run in order with their
  own routed exceptions. Lists, strings, custom iterables and mismatched/unknown tuple arities retain
  the refusal. Native controls in `tests/test_absint_try.py` cover normal/finally continuations,
  nested arity failures and a subscript target failing after an earlier name assignment.
  Native scalar float addition, subtraction and multiplication also pass this boundary when
  `_float_conversion_is_total` proves any integer operand converts: IEEE overflow returns infinity,
  and operations involving infinity/NaN return a float without a Python domain exception.
  Possible integer conversion overflow and powers retain the refusal. Native controls include
  nonfinite results, mixed bounded int/float operands and a failing overflow handler.
  A `None` alternative alongside the numeric part is a routed `TypeError`; it does not add a
  non-property exception. Guarded `math.sqrt` is admitted only for one exact native numeric operand
  with total conversion and a nonnegative numeric domain (NaN itself remains a successful NaN result).
  `tests/test_absint_float_guards.py` checks both guard branches, zero/NaN/infinity, optional floats,
  target rebinding, effectful comparisons and a custom conversion hook against native execution.
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

> **Corrected 2026-11-05 (T178, lane 1518; owner 1419: "it feels like cheating because it uses the competition function names to score
> points ... the nondet bool one is bad"; 1420, 1426: keep the split, remove the name dependence).** The first implementations of this
> section's mechanism were WRONG: an AI session (Claude Sonnet 5.5, `ab64360d`, 2026-10-16) split only the one call spelled
> `_sv_verifier.nondet_bool`, gated the retry on the source text `nondet_bool`/`nondet_int` (`_splittable`, tokenised but still a
> spelling in `b64fe40c`, Claude Opus 5.5, whose own reason was "bit-for-bit identical on the benchmark corpus"), enumerated an int only
> when a direct `nondet_int` call sat in the assignment (`_int_outcomes`), and recognised the conditional-alias cover by an attribute
> spelled `.nondet_bool()` (`_scalar_identity_cover`, `eabff14a`, opencode lane). Each keyed on a spelling instead of the value's
> domain; the sweep measured 21 of 69 TRUEs lost when only the draw's spelling changed (`nondet_int() % 2 == 0`) and 7 of 21 for a
> helper. What is true now: **a case split is decided from the value**, a `bool`-typed abstract value that may be either, derived
> from an input; the API's own name appears only to recognise an input by its origin (`_sv_verifier.nondet_*` resolved through the import
> graph) and to state its contract (`nondet_bool` returns False or True).

The join at the end of an `if` (or the join of a returning call) keeps what both arms agree on and drops the
correlation between a harness draw and what was computed from it. SV-COMP's `type-eval-py` tasks are built on exactly
that correlation:

```python
flag = _sv_verifier.nondet_bool()      # equally: _sv_verifier.nondet_int() % 2 == 0, not (...), a helper's return
v = "s" if flag else 1
if flag:  _sv_verifier.check_type(v, str)
else:     _sv_verifier.check_type(v, int)
```

Joined, `v` is `str | int` and `check_type` refuses ("check_type value has mixed or unsupported possible types").
A value that is a `bool` and may be either takes exactly two values, so an execution is one of two, and the
analysis may run the rest of the program once per value and join only where the block ends.

- **What is split** (`Analyzer._draw_to_split`, `_once_choices`, `_is_bool_choice`, `_exec_split`): a statement of the program's own file
  that is an `Assign`/`AnnAssign` value, an `Expr` value or an `If` test, with a non-empty rest of its block, holding
  exactly ONE call, comparison or `not` that derives from an input (`_has_input`: a call the import graph resolves to the verifier's
  `nondet_*` family, or a call of a function the program defines, which may return a draw) AND whose abstract value, probed on a
  copy of the state without recording (`_is_bool_choice`), is a `bool` that may be either; the statement evaluates it at most once
  and in an unconditional position: call
  callee/arguments/keyword values, operator operands, the first comparator of a comparison, the first operand of
  `and`/`or`, the test of a conditional expression, the base and index of an attribute or subscript, the elements of
  a display. The statement runs twice with that expression's value forced to `False` / `True` (`eval` overrides the value of a bool-typed node,
  `cond` empties the other branch), then `stmts[idx + 1:]` runs on each
  outcome, and the two end states are joined by `_exec_split` (the block's own end).
- **What is not**: a draw in a `while`/`for` header, a comprehension, a lambda, a conditional arm or the tail of a
  short circuit; two or more eligible draws in one statement; a statement inside a loop body (`lctx`), a `try` (also
  of a caller: the draw's own refusal is routed there), an extern file or a pass that does not record; a `return`
  value (no rest to keep apart). Values that are not bool-typed are not forced; an int drawn small is enumerated instead
  (`_int_outcomes`: an assignment whose right side derives from an input, `_has_input`, and whose value is an int with a small or
  comparison-partitionable interval).
- **The retry** (`absint.prove`): `split_draws=None` (the default) runs the joined analysis exactly as before and only
  when it did not prove TRUE, and the joined run EXECUTED a statement that evaluates an input or a program function, a list display bound
  by an assignment or a starred rest (`Analyzer.split_hint`, read by `absint/__init__.py::prove`; it replaces the token test
  `_splittable`), runs it once more with `split_draws=True` on at
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
  unproven check does everywhere. Object alternatives are checked separately: a matching instance never runs its `repr`, even
  when another alternative mismatches. Mismatching objects use the existing `check_formattable` hook check. Exact builtin
  containers are traversed through their represented elements with cycle detection; default object repr is also safe. A custom
  instance formatting hook or unknown element refuses. The six original builtin hints use the same check on mismatching
  list/dict alternatives, without inspecting a matching container's elements. There is no length or nesting cutoff.
  Their mixed-success/container-mismatch values retain UNKNOWN; this admission
  does not extend formatting proofs across an unrefined mixed normal state.
  Newly admitted mismatch formatting inside a `try` remains UNKNOWN: formatting can fail before TypeError construction
  (for example the integer string-digit limit), and those intermediate exception payloads are not modelled.
- **Unchanged**: a hint that is not one class address (a union of classes is a case split of the program, a `typing` form, a builtin
  other than the supported ones) keeps its refusal.
- **The exact container/exception hints beyond the six** (`m__sv_verifier_check_type`'s `("cbuiltin", "tuple"|"set")` and
  `("exc", <name>)` branches): the runtime decision stays `isinstance(value, <class>)`. A `tuple` hint matches a `TupleObj`
  exactly, a `set` hint matches a `SetObj` whose `frozen` is False (`frozenset` never), an exception hint matches an instance whose
  class records exactly one builtin exception base (`s_ClassDef`'s `__exc_base__`: `_exc_base_of`'s chain) or a canonical
  `ExceptionObj` whose own C-tree derives the hint. A mismatch formats `value!r` into the TypeError: a scalar repr runs no user
  code, an object's repr may, so only a fully-matching value may carry objects, and every other shape refuses (`tests/
  test_absint_builtin_class_hints.py`). The default set/`frozenset` distinction is by the `SetObj`'s own `frozen` flag; an
  `Exception` hint's `("exc", name)` tag never forms under a `typing` form or an unknown builtin, so none of those came here.
- Tests: `tests/test_absint_class_hints.py` runs 27 adversarial programs (`tests/fixtures/absint/class_hint/`) natively over every
  boolean draw vector; a metaclass with `__instancecheck__`, a `__class__` property, `a.__class__ = B`, a `__getattribute__` that
  answers `__class__`, a `repr` that raises, a rebound class name and a class object or function passed as the value are among them.

## Recursive builtin check_type hints

Inline verifier hints such as `list[list[int]]` and
`dict[str, list[tuple[int, str]]]` use the shared
`solve/check_type_hints.py::builtin_hint_shape` grammar. Absint also checks
each name's current binding before constructing an inert hint description.
Only direct verifier hint arguments receive this model; aliases that escape,
typing forms, custom classes and unsupported syntax retain their refusals.

The transfer requires exact builtin list, tuple, dict, set or frozenset heap
objects throughout the value. It checks element summaries recursively, fixed
tuple positions individually, mapping keys and values separately, and empty
containers vacuously. Bool remains an int. `tuple[()]` follows the pinned
runtime checker: its empty parameters cause success for any builtin tuple.
Unknown values, subclasses and user callbacks refuse. An uncertain or failing
check records an unproven TypeError and preserves the possible normal path;
it never treats a summary element as proof that the container is nonempty.
Checks inside try statements currently refuse.

The plain builtin-exception hint model also refuses mismatch formatting of
exception arguments: their repr can execute user code. Regression and
differential tests are in `tests/test_absint_nested_hints.py` and
`tests/test_absint_builtin_class_hints.py`.

## type(x), class identity and scalar disequality

- **`type(x)`** (`Analyzer.b_type`, the builtin tag `("builtin", "type")`): the class of each alternative of `x`, as the static class tag
  (`("type", "int" | "bool" | "float" | "str" | "list" | "dict")`, `("cbuiltin", "tuple")`) or, for an instance of a class of the
  program, that class's address. A union of alternatives is the union of their classes, so `bool` and `int` stay two classes
  (`type(True) is int` is False). It refuses what it does not name: `None`, a function, a class object (its type may be a
  metaclass), a set, an array, an iterator, a `defaultdict` (not an exact `dict`), an exception instance, an object carrying its own
  `__class__` or whose class defines one, `type(name, bases, dict)` and any keyword; any other arity is the `TypeError` of CPython.
- **Identity and equality of classes** (`Analyzer._class_identity`, used by `Is`/`IsNot` in `compare_split` and by `_decide_eq`): two
  class values are the same object when both are one builtin tag, or one class address `(site, "r")`; different objects when their
  tag sets and address sets are disjoint (a builtin class is no class of the program, and `(site, "r")` and `(site, "s")` are different
  objects: the most recent and the older ones). Anything else is not decided. A class of the program has no metaclass here, so `==`
  on classes is `is`.
- **Scalar disequality** (`_decide_eq`): a `str`, a number (`int`/`bool`/`float`) and `None` are never equal to a value of another of the
  three; `==`/`!=` on pure builtin scalars of different classes is decided. (`1 == 1.0 == True` stays undecided.)
- **Scalar union hints**: `check_type(v, A | B | ...)` for any number (>= 2) of operands, each `int`/`bool`/`float`/`str` or `None`, in
  any order (`_UNION_HINT_NAMES`; first `int | str` only, then any two names: both were decided by the hint's spelling, not by what
  `isinstance(v, <union>)` means). `int` covers `bool`, every other name covers itself, `None` covers `None`; `None | None` is a
  TypeError of the hint itself and is not admitted. An uncovered scalar (`None` included) is an unproven `TypeError` check (its repr
  runs no user code), an object or a static value refuses. Tests: `tests/test_absint_union_hints.py`.
- **Starred method calls** (`obj.f(*args)`, `mod.f(**kw)`): admitted when every argument is a name or a constant, bare or starred
  (`interp.py::_effect_free_arguments`): Python looks the attribute up before it evaluates the arguments, and nothing such an argument does
  can rebind what the attribute names. Any other argument (a call, an attribute, a subscript, a display) keeps the refusal "method call
  with *args (attribute lookup order)". The starred value itself must still be one fixed-length list or tuple
  (`call_positional_args`). Tests: `tests/test_absint_starred_method_calls.py`.
- **What this needs from the rest**: `type(x) is type(y)` is decided when each side is one class, which the boolean case split provides
  for a value drawn as `int` or `str`; a class made in a loop is refused where it is defined (`tests/test_absint_class_in_loop.py`).
- Tests: `tests/test_absint_type_call.py` (39 adversarial programs, `tests/fixtures/absint/type_call/`, natively checked over every draw
  vector): subclass versus base, bool versus int, metaclass, `__new__`, `__class__` property and reassignment, a shadowed `type`,
  three-argument `type`, a class made in a loop, defaultdict, union coverage of bool/int/float/None/objects.

## Positional string joins

String joins of exact positional string snapshots retain their value as existing string concatenation recipes after the ordinary
tested contract transfer. Empty and singleton sequences are included; mutable summaries and key-view order are not inferred.
See `tests/test_absint_positional_string_join.py`.

## List displays nothing observes as a list

A list display keeps ONE joined element (`ListObj.elem`), so `a = [func1, func2, func3]; a[1]()` calls "any of the three" however exact
the index is. A tuple display keeps one value per position (`TupleObj.items`). `frozen_list_displays` (interp.py) finds the list
displays that may be built the second way, and only in the retry (`prove(split_draws=None)` re-runs the analysis with the case splits
AND these displays; `split_draws=False` never does):

- **Admitted**: a module-level name (or inside a module-level `if`) bound ONLY by `name = [d0, d1, ...]` statements (no star item), or as a starred-rest target, in
  which every other occurrence, in every file the program reads, is `name[index]` read (not a slice, not a store, not `del`),
  `len(name)` with the builtin `len` (`len` bound nowhere), the right side of an unpacking assignment (including a star),
  or a literal builtin str receiver's `"separator".join(name)`. The starred rest is a fresh list. These reads preserve
  the list's observable behavior; variable or custom join receivers remain ordinary call arguments.
- **Starred rests**: exact source positions are retained only when the same escape audit admits the rest name and all successful source alternatives have the same known middle arity. Otherwise the ordinary list summary remains. The rest keeps native list IndexError payloads; a list-type observation withdraws positional storage. Tuple-only starred assignments also trigger the existing bounded retry. There is no size cap.
- **Withdrawn for the name**: anything else, an alias, a call argument (`f(a)`, `f(*a)`, `list(a)`), an attribute (`.append`), a loop
  or `in`, a comparison, a slice, a store, `del`, `+=`, a walrus or any other binding (parameter, def, class, import, `global`,
  `except ... as`, `for`/`with` target), a binding anywhere but the module level, and ANY mention of the name in another file
  (a sibling can mutate it through `from main import a`).
- **Withdrawn for the whole program**: any channel to a module global that does not name it, in any file: `globals()`, `vars()`,
  `locals()`, `__dict__`, `getattr`/`setattr`/`delattr`, `eval`/`exec`/`compile`, `__import__`/`importlib`, `sys.modules`, `__main__`,
  `inspect`, `gc`, `builtins`, a star import, or a sibling that imports the program by its file name.
- **Exception payloads retain the original list semantics**: out-of-range subscripts use the list contract even though the
  positions use tuple storage, so a caught `IndexError.args` has the list message. Elements keep their own heap identity: an element that is
  itself a list is the same `ListObj` the display built, so `x = a[0]; x.append(9)` is seen through `a[0]` (`fl21`-`fl24`).
- `type(a)`, `isinstance(a, ...)`, `a == ...` and every other use are not reads, so they withdraw the name and the program keeps the
  joined `ListObj`; the admitted reads cannot reach the tuple-ness.
- Tests: `tests/test_absint_frozen_lists.py` (40 adversarial programs in `tests/fixtures/absint/frozen_list/`, natively checked over every
  draw vector, an eligibility table of 37 sources, three multi-file programs with a sibling that mutates through `__main__`, star-imports over the
  name and imports the program).

## Case splits on a small int, and three exact-constant decisions

- **A small int drawn by `nondet_int`** (`Analyzer._int_outcomes`, `_exec_int_split`; the retry only, like the boolean splits). After
  a plain `name = <expression holding a nondet_int() call>` statement (outside a loop body, a try and an extern file, in a recording
  pass, with a non-empty rest of its block), the analysis' OWN value of `name` is read: when it is an int and only an int, whose
  interval holds 2 to `MAX_INT_SPLIT` (16) values (`abs(nondet_int()) % 3` is `[0, 2]`), the post-state is refined to `name == v` for
  each `v` (`cond` on a synthesized comparison; an infeasible `v` is dropped) and the rest of the block runs once per outcome, joined
  at the block end. The interval holds every value `name` can have, so the outcomes cover every execution: nothing about the
  right-hand side is trusted, only the interval, and a wider one, an unbounded one, a float, a draw inside an index expression
  (`xs[abs(nondet_int()) % 2]`), a tuple target or a statement in a loop stays joined. The budget is `MAX_DRAW_SPLITS`, shared with the
  boolean splits: an outcome beyond the first costs one. As for booleans, candidate invariants are not taken while outcomes run apart.
- **Membership in a tuple of the program** (`_tuple_membership`): `x in (1, -2)` is decided when `x` and every position are
  constants (`_decide_eq`, so `True in (1, 2)` is True and an unequal class is False); otherwise undecided.
- **Identity with the bool singletons** (`_bool_singleton_identity`): `x is True`/`x is False` is True when `x` is that constant and
  False when `x` can never be it (no such bool among its alternatives; an int, a str, None, an object are never `True`).
- **A strong store into a one-element list** (`store_elem`): `b[0] = v` / `b[-1] = v` on a RECENT list whose length is exactly 1 leaves
  `v` as its only element; any other store (two elements, a summary, an unknown index) joins as before.
- Tests: `tests/test_absint_int_split.py` (27 adversarial programs in `tests/fixtures/absint/int_split/`, run natively over every draw
  vector with ints over `range(-3, 20)`): a wider modulus, a negative remainder, two and three draws, a draw in a loop, in a function, behind
  a boolean, a float target, an index expression, the one-value IndexError and ZeroDivisionError, tuple membership across classes, a
  one-element list stored through an alias, a callee, in a loop and after an append.

## Files and how they relate

Measured from the committed source in this branch by `scripts/doc_import_map.py::compute_ast_imports`, including lazy imports.

### Subsystem: `tharkun.solve.absint` (22 files)

- `src/tharkun/solve/absint/tail_loops.py`
  - **Role**: Proof-only direct tail loops for primitive calls with an explicit base return and no frame observation; unsupported bodies retain ordinary call interpretation.
  - **Imports (tharkun.solve.absint)**: none
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`


- `src/tharkun/solve/absint/__init__.py`
  - **Role**: Public package entry point: `prove()` running abstract interpretation and producing verdicts; `component_for` names `proof:shape` for TypeError/AttributeError properties and `proof:interval` otherwise.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/state.py`
  - **Is imported by (tharkun.solve.absint)**: none
- `src/tharkun/solve/absint/__main__.py`
  - **Role**: CLI entry point (`python -m tharkun.solve.absint`) with `--property` and `--explain`.
  - **Imports (tharkun.solve.absint)**: none
  - **Is imported by (tharkun.solve.absint)**: none
- `src/tharkun/solve/absint/affine.py`
  - **Role**: Private exact affine equality basis, with projection and affine-hull joins under the analyzer deadline; enabled by assertion demand alongside the DBM.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/errors.py`, `src/tharkun/solve/absint/zone.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/zone.py`
- `src/tharkun/solve/absint/aval.py`
  - **Role**: Abstract value representation (`V`) combining intervals, float info, types, and zone references.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/zone.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/contracts.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/map_identity.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/rand.py`, `src/tharkun/solve/absint/reducers.py`, `src/tharkun/solve/absint/sets.py`, `src/tharkun/solve/absint/state.py`, `src/tharkun/solve/absint/visitor.py`
- `src/tharkun/solve/absint/contracts.py`
  - **Role**: Demanded scalar and list-shape class invariants, checked independently of call contexts.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/state.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`
- `src/tharkun/solve/absint/equality.py`
  - **Role**: Lazy typed value recipes, congruence closure and SMT-admitted string concatenation identities under the existing analysis deadline.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/errors.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/state.py`
- `src/tharkun/solve/absint/exactfloat.py`
  - **Role**: Exact dyadic float terms (`("xd", s, c, ((("iv", var), k), ...))`): a float known to be the real number `(c + sum k*x)/2**s` itself, representable in binary64; arithmetic on them, exact result intervals, and the `==` decision of two of them (V_TERMS, [internal ref]). Pure stdlib (`fractions`).
  - **Imports (tharkun.solve.absint)**: none (stdlib only).
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`
- `src/tharkun/solve/absint/errors.py`
  - **Role**: `Unsupported` exception definition indicating unmodelled Python constructs.
  - **Imports (tharkun.solve.absint)**: none
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/affine.py`, `src/tharkun/solve/absint/equality.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/map_identity.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/rand.py`, `src/tharkun/solve/absint/reducers.py`, `src/tharkun/solve/absint/sets.py`, `src/tharkun/solve/absint/visitor.py`
- `src/tharkun/solve/absint/generators.py`
  - **Role**: Compile structured source generator bodies into immutable `ContinuationGraph` and `Instruction` continuation graphs, preserving yield, branch and loop control edges; unsupported suspension shapes raise `Refused`.
  - **Imports (tharkun.solve.absint)**: none (stdlib `ast` and `dataclasses` only).
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`
  - **Calls and callers (measured AST)**: `interp.py::Analyzer._resume_generator` has one `compile_program` call site, guarded by its per-function cache. `compile_program` invokes the supplied lexical-suspension callback; the caller supplies `interp.py::_is_generator` through a lambda. The compiler makes no direct call into another absint module.
- `src/tharkun/solve/absint/heap.py`
  - **Role**: Heap objects of the absint abstract state (`OutFileObj` [[internal ref]], `ListObj` with segmentation, `TupleObj`, `ArrayObj` subclassing `ListObj` with `typecode` [[internal ref]], `InstObj` with `qf`, `ClassObj`, `GeneratorObj` and per-position `GeneratorFrame`, `FuncObj` carrying `module_key` [[internal ref]], `BoundMethod`, `BuiltinMethod`, `RangeObj`, `SetObj`).
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/contracts.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/map_identity.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/rand.py`, `src/tharkun/solve/absint/sets.py`, `src/tharkun/solve/absint/state.py`, `src/tharkun/solve/absint/visitor.py`
- `src/tharkun/solve/absint/interp.py`
  - **Role**: The absint interpreter (`Analyzer`) covering statements, expressions, calls, and loop machinery (`s_While` / `s_For`: first iteration, `_solve_fixpoint` with `_fixpoint` reuse, final pass, `fresh_pass`, `_visit_invariant`), builtin and list models, float terms, focus/blur, and site records.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/contracts.py`, `src/tharkun/solve/absint/equality.py`, `src/tharkun/solve/absint/errors.py`, `src/tharkun/solve/absint/exactfloat.py`, `src/tharkun/solve/absint/generators.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/lemmas.py`, `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/map_identity.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/polynomial.py`, `src/tharkun/solve/absint/rand.py`, `src/tharkun/solve/absint/reducers.py`, `src/tharkun/solve/absint/sets.py`, `src/tharkun/solve/absint/state.py`, `src/tharkun/solve/absint/tail_loops.py`, `src/tharkun/solve/absint/visitor.py`, `src/tharkun/solve/absint/zone.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/__init__.py`, `src/tharkun/solve/absint/contracts.py`, `src/tharkun/solve/absint/map_identity.py`, `src/tharkun/solve/absint/visitor.py`
- `src/tharkun/solve/absint/lemmas.py`
  - **Role**: Named numeric lemmas (`TRIG_PAIR`, `NORMALIZE`) and widening landmarks for float/numeric domains.
  - **Imports (tharkun.solve.absint)**: none
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/state.py`
- `src/tharkun/solve/absint/linear.py`
  - **Role**: Linear forms (`Lin`, `lin_bounds`, `IntInfo`) and symbolic integer bounds connected to zone DBMs.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/zone.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/polynomial.py`, `src/tharkun/solve/absint/rand.py`, `src/tharkun/solve/absint/state.py`, `src/tharkun/solve/absint/visitor.py`
- `src/tharkun/solve/absint/map_identity.py`
  - **Role**: Closed identity/scalar-callback map source and consumer proof, imported only when a map call is reached.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/errors.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/interp.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`
- `src/tharkun/solve/absint/native.py`
  - **Role**: Finite builtin values for string/format/regex operations; never executes target objects.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/errors.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/state.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/visitor.py`
- `src/tharkun/solve/absint/polynomial.py`
  - **Role**: Budgeted exact polynomial division returning only proved affine integer quotients.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/linear.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`
- `src/tharkun/solve/absint/rand.py`
  - **Role**: `RandomMixin` implementing abstract transfer functions and argument binding for the `random` module.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/errors.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/state.py`
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
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/equality.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/lemmas.py`, `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/zone.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/__init__.py`, `src/tharkun/solve/absint/contracts.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/rand.py`, `src/tharkun/solve/absint/reducers.py`, `src/tharkun/solve/absint/sets.py`, `src/tharkun/solve/absint/visitor.py`
- `src/tharkun/solve/absint/visitor.py`
  - **Role**: Contract-rule visitor (`AbsintVisitor`, `exact_int_value`, `hook_free_value`) translating tested contracts into abstract transfer operations; absent visits refuse.
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/errors.py`, `src/tharkun/solve/absint/heap.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/native.py`, `src/tharkun/solve/absint/state.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/sets.py`
- `src/tharkun/solve/absint/zone.py`
  - **Role**: Difference-bound matrix (`Zone`, `ZERO`) domain tracking relational integer inequalities (`x - y <= c`).
  - **Imports (tharkun.solve.absint)**: `src/tharkun/solve/absint/affine.py`
  - **Is imported by (tharkun.solve.absint)**: `src/tharkun/solve/absint/affine.py`, `src/tharkun/solve/absint/aval.py`, `src/tharkun/solve/absint/interp.py`, `src/tharkun/solve/absint/linear.py`, `src/tharkun/solve/absint/state.py`


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
- `tests/test_absint_str_methods.py`: `str` methods with hook-free arguments; 86 adversarial programs (`tests/fixtures/absint/str_methods/`) natively checked over every combination of a small set of draws. `tests/contracts/test_str_shapes.py` and `tests/contracts/test_safety_str_templates.py`: the CPython three-way tests of the admitted shapes and of the TypeError template.
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
- `tests/test_absint_int_split.py`: enumerating a small drawn int, tuple membership, bool-singleton identity and the one-element strong store (`tests/fixtures/absint/int_split/`).
- `tests/test_absint_frozen_lists.py`: list displays nothing observes as a list are built per position (`tests/fixtures/absint/frozen_list/`).
- `tests/test_absint_type_call.py`: `type(x)`, class identity, scalar disequality and scalar union hints (`tests/fixtures/absint/type_call/`).
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
The read of an attribute of a builtin scalar (`interp.py::Analyzer._attr_load_model`) is decided by the pinned exact-builtin attribute
inventory `contracts/safety/rules_protocols.py::_INSTANCE_ATTRS`: a name the inventory lists is still refused (the bound method or
data is not modelled), a name no scalar kind has is a real `AttributeError` check ([internal ref]), so `x = 5; x.foo` reaches the `attr`
check instead of the refusal "attribute foo of a builtin scalar".
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

**This check produces a claim, not a certified fact ([internal ref]).** docs/AVENGERS.md "Guessed invariants, proved facts"
makes this check one *producer* of class invariants. Its contracts serve absint's own query transfers. Only the
SV-COMP checker (`src/tharkun/sv/validate.py`) certifies a fact, and it never reads this verdict. Exactly what
`src/tharkun/solve/absint/contracts.py::infer` accepts, guesses and requires:

- **Shapes it refuses (no contract for the whole class):**
  - any base class;
  - any class attribute that is not one plain, undecorated function taking only the receiver. That excludes other
    parameters, defaults, `*args`/`**kwargs` and keyword-only parameters;
  - a class with no `__init__`;
  - an `__init__` statement other than a docstring or `self.f = <literal>`, where the literal is an `int` or a list
    of `int`s;
  - a method body that `src/tharkun/solve/absint/contracts.py::_scalar_body` rejects. It rejects a call,
    `global`/`nonlocal`, `del`, `yield`/`await`, a nested `def`/`class`/`lambda`, `with`, `try`, `import`, `is`, an
    attribute other than `self.<candidate field>`, a non-local name, and a receiver that is used bare or rebound.
- **Candidates:**
  - the sign of the literal: `> 0` gives `f >= 1`, `== 0` gives `f >= 0`, `< 0` gives `f <= -1`;
  - for a list literal, `int` elements and a length of at least `1` when it is non-empty;
  - a `class_hints` sample `[lo, hi]` replaces a scalar candidate with a one-sided bound only: `f >= lo` when
    `lo >= 0`, `f <= hi` when `hi < 0`, and no bound otherwise. The upper end of a non-negative range is never a
    candidate.
- **Obligations.** `__init__` runs from an empty receiver. Every other method runs from an arbitrary receiver that
  satisfies the candidates; that is the receiver precondition. Each run must:
  - end in a reachable (non-bottom) state;
  - pass every exception check the transfer records, so a method that may raise drops the candidate;
  - leave every candidate field definitely set and within its candidate;
  - return a scalar: not a heap reference and not an opaque value (a pattern, an exception instance).

  Any failure, `Unsupported` or `Incompatible` drops the whole contract.
- **Use.** Only methods that mutate nothing become cached queries. Outside a `try`,
  `src/tharkun/solve/absint/contracts.py::apply` rechecks the receiver precondition against the live abstract object at
  every call, so outside writes and aliases are caught where the contract is used rather than assumed away.

### Recipe field facts ([internal ref])

`src/tharkun/solve/absint/interp.py::Analyzer.run` returns the final abstract state; existing proof callers may
ignore it. Its optional `module_name` selects import semantics for the module's `__name__` guards.
`::Analyzer.exact_constant` is the existing scalar equality singleton check, shared with
`src/tharkun/avengers/flatobject.py::derived_fields` to export definite exact `InstObj` fields for a row's recipe.
No expected policy values participate. The complete module and all constructor/setup statements must be
supported; no facts escape an aborted analysis or bottom state. This reuses the instance heap, not the file-world
`FactState` mechanism. Generalizing the world-object model to every object remains deferred.

## Contracted operator callables ([internal ref])

Module callable aliases reach `AbsintVisitor.visit_operator_function`; explicit
visits take precedence over generic module fallback. The registry contract must
remain tested. The visitor returns the ordinary intrinsic abstract result and
records all seven contract raise classes, preserving every failed model check.
Arithmetic, ordering, truth and indexing reuse existing transfers; item updates
and in-place operations mutate the shared heap. Searches fold only finite
builtin values accepted by `absint.native.concrete`, which excludes target
objects and iterator cursors. Large/non-finite unsupported cases remain named
refusals. Matrix methods reuse normal reflected/in-place dispatch; list
repetition in place preserves aliases through the shared repetition shape.

Getter/caller factories have a non-null callable abstract result. Invoking that
opaque result still refuses explicitly; it never becomes a successful call or
a false TypeError. See `tests/test_absint_operator_contracts.py` for programs
that use each result, seven-class checks, stale-alias adversaries and contract
withdrawal. Scalar facts can be intervals rather than exact point values.

## Declared builtin return transfers ([internal ref])

The bool class keeps its type tag and routes calls through its tested Rule;
zero arguments produce False, otherwise the existing truth transfer produces a
bool fact. Callable reads only the abstract type call slot, including a present
but non-callable __call__ attribute, as CPython does; instance-only __call__
attributes do not create a slot. It never invokes the target or its attributes.

Bin/hex/oct require modeled builtin int/bool arguments. Finite small values fold
through the Rule's native callable; other integers produce ANYSTR. Finite fold
size is bounded, and no user conversion hook runs in the host. Unsupported
__index__ cases refuse explicitly. The seven contract raises remain authoritative
and model checks can only prove them for validated arguments. See
`tests/test_absint_builtin_contracts_missing.py`.

## HTML utility transfers ([internal ref])

Both html Rule visitors validate positional/keyword binding and accept builtin
str inputs with bool quote flags. Finite text up to 640 characters folds through
Rule.native; long/nonfinite text yields ANYSTR. The bound avoids host integer
digit-limit dependence. No target hook runs or mutates caller state on this path.
Invalid binding records TypeError; other operand kinds refuse. Seven call-qualified
checks, exact/type result use, aliases, withdrawal, open-hook refusals and reachable
failures are pinned by tests/test_absint_html_contracts.py.

## Builtin pow consumption ([internal ref], batch 7)

Pow now reads its tested Rule through an explicit absint visitor. Argument
binding supports base/exp/mod keywords and None modulus. Two-argument powers
reuse the existing binary power transfer, including analyzed source hooks and
exception checks. Three-argument builtin int/bool operands return an integer
fact; zero modulus or a possibly noninvertible negative exponent retains an
explicit ValueError check. User hooks in that form remain a named refusal.

The old numeric-only native ReturnSpec was too narrow: power hooks can return
arbitrary objects, including in Python 3.14 reflected modular dispatch. It now
says object, with builtin-number results qualified in its shape; native tests
pin direct/reflected two/three-argument returns. The frozen pow row is regenerated
(C consumption plus the reviewed semantic/source digest change). Focused tests
use actual return facts, preserve reachable assertions, withdraw the contract,
and check TypeError/ZeroDivisionError/ValueError and callback boundaries on 3.14.4.

## Subclass contract consumption ([internal ref], batch 8)

Issubclass uses its tested Rule and supplies boolean facts for modeled builtin
classes, user class inheritance and fixed nested classinfo tuples. It reuses
_classinfo validation and the existing subclass walk. Joined class values are
separate alternatives; ambiguous tuple members refuse before flattening, and
mixed class/scalar inputs refuse before losing the invalid scalar path. Custom
metaclass/subclass checks and fake class metadata remain unmodeled. All seven
contract checks remain visible, and withdrawing the contract withdraws the model.

The shared classinfo helper also accepts builtin dict; its paired isinstance
receiver-kind test now recognizes dicts. Fixed native-backed tests cover all
64 builtin class pairs, user inheritance, nested/empty tuples, dict instance
checks, joined inputs, callback and TypeError boundaries, and reachable failures.
The existing contract semantics/digest stay unchanged; its audit row is generated
by the normal frozen writer on CPython 3.14.4.

The nested-classinfo guard lives in _classinfo itself and also fixes an existing
isinstance false proof: T=int-or-str; isinstance(1,(T,)) cannot always be True.
The fixed native witness chooses str; lane1455 before/after evidence records
that clean main 4af2f810 proved it and both corrected lines leave it unproven.

## Dictionary constructor consumption ([internal ref], batch 9)

The existing dict Rule now creates a real DictObj through an explicit visitor.
Empty/keyword initialization, builtin dictionary/defaultdict copies, fixed tuple
pairs and builtin lists of tuple pairs are modeled. Tuple order is retained;
list summaries keep only universally present keys and conservative value facts.
Keyword writes reuse the existing dictionary store transfer. Values retain their
original heap references, preserving shallow aliases while the new dictionary
has separate storage. Captured fields are borrowed before allocating, using the
existing demotion guard. User mapping/iteration/hash/equality hooks and joined
sources refuse. No target operand is passed to a native host constructor.

Focused tests consume dictionary results and check contents, overrides, shared
list mutation, separate storage, nonfinite values, callback/arity/error boundaries,
withdrawal and reachable failures. The existing contract digest is unchanged;
the standard frozen writer changes its absint audit row from refusal to consumption.

## Textwrap utility transfers ([internal ref], batch 10)

Explicit Rule visits serve wrap/fill/shorten/dedent/indent. They bind the canonical
stdlib function via the existing resolve_native helper. Finite primitive operands
reuse native.concrete/product/abstract under the shared analysis deadline, with text/numeric
inputs up to 640. Native property errors and ValueError/OverflowError are recorded;
only successful alternatives contribute results. No target hook runs in the host.

For nonfinite strings, default wrapping/filling/shortening with builtin integer
width retains list[str]/str facts; possible invalid widths/placeholders keep
ValueError checks. Dedent accepts str and indent accepts str prefix with no predicate.
Custom nonfinite options and callbacks refuse. Finite wrap results retain list
length and element facts, while line order uses the existing list summary.
See test_absint_textwrap_contracts.py for result uses, errors, joins and withdrawal.

## Configured TextWrapper instances ([internal ref], batch 11)

TextWrapperObj stores the twelve constructor settings as abstract values. The
canonical constructor binds the real signature and stores values without invoking
or converting them. Heap join/order/value rewrites and GC retain settings and their
references. Field stores use strong/weak updates; bound methods use BuiltinMethod
so aliases see later mutations and keep the receiver alive after a frame returns.

Wrap/fill validate their own method call shape and tested contract, then reuse the
utility transfer with the current receiver settings. Only canonical instances and
known settings are modeled; private overrides, arbitrary attribute mutation and
unsupported callback/configuration forms refuse. Class/type facts are available
through isinstance/issubclass. Exact static class identity and attribute-expression
None identity retain the existing domain's precision limits; binding a field first
preserves its None fact. Tests verify class/result facts, settings, aliases, deferred
validation, joins, loops, lifetime and cross-kind refusal boundaries.

## Integer bit queries ([internal ref])

`contracts/rules_int.py::IntBitLength` and `IntBitCount` describe the callback-free
CPython descriptors, including TypeError conditions and native argument tuples.
`visitor.py::AbsintVisitor._scalar_query` consumes their tested contracts before
returning any fact: finite integers/bools use native results; other integers use
nonnegative intervals, bounded above by the largest absolute endpoint's bit length
when both endpoints are finite. Bit length is monotone in absolute magnitude;
population count is only bounded, never assumed monotone. Negative integers use
absolute values, not a fixed-width two's-complement representation.
`interp.py::Analyzer.call_method`, `_attr_load_model`, `_call1` and `static_attr`
serve direct, bound and unbound forms (including inherited bool methods). Bound
methods snapshot receiver bounds. Unknown receivers and calls inside exception
handlers remain refused; the transfer executes no target callbacks.

## Partition tuple results ([internal ref])

`visitor.py::AbsintVisitor._abstract_string` returns a fixed three-element tuple
of strings for `str.partition` and `str.rpartition`, even when receiver or
separator is not finite. `adapters/str_shapes.py::SIGNATURES` admits only a
positional builtin string separator; that call runs no user hooks. Empty
separators may still raise `ValueError('empty separator')`; calls inside a try
remain refused by `interp.py::Analyzer.visit_rule`, so no handler is dropped.
Finite operands keep the existing exact native tuple. Unknown components stay
unknown strings: the transfer never assumes that the separator was found.

## Integer identity and ratio queries ([internal ref])

`contracts/rules_int.py::IntConjugate`, `IntIsInteger` and `IntAsIntegerRatio`
use the same tested, callback-free descriptor contract and dispatch as the bit
queries. `visitor.py::AbsintVisitor._scalar_query` retains the receiver's
integer value for conjugation, returns True for is_integer, and constructs the
exact two-slot tuple `(value, 1)` for as_integer_ratio. This works for unknown
integers and preserves the denominator as exactly one. `interp.py::Analyzer._as_int`
normalizes bool values to integers: neither conjugate nor a ratio numerator
returns a bool. These are integer rules; float/complex methods keep their own
semantics and are not admitted by the integer transfer.

### Exception argument transfers ([internal ref])

`ExceptionConstructor` rules have an explicit absint visitor. Canonical builtin
instances retain the exact positional tuple in `ExceptionObj`; source exception
subclasses retain separate `InstObj.exc_args` descriptor storage before their own
`__init__` runs. Both participate in heap reachability, scalar freezing, recency
rewrites, joins and ordering. `.args` reads, `type(e)` and class-name observations
can therefore consume contract facts rather than opaque exception tags. The
constructor's keyword `TypeError` uses its tested `ConstantArgsRaiseClaim` to route
the exact payload to a handler. Legacy class-only checks retain opaque values.

Raising an instance, raising a class with no arguments, catching it, propagating
it through calls and a bare re-raise preserve the same modeled payload. Literal
`raise ... from None` is accepted for class-and-args observations. Cause, context,
traceback and suppression inspection remain refused, as do other explicit causes,
exception formatting, `.args` assignment and custom `.args` descriptor shadows.
No native target constructor or argument hook executes during analysis. A raise
transfer is an abstract fact; it is not a synthesized inversion proof or replay.

Primitive int/bool/float division, floor division and modulo consume typed
`NumericDivision` rules, including reflected float dispatch for an int left
operand and float right operand. Their visitor reuses existing interval transfers
and bounded canonical native slot evaluation for finite primitive sets. A possible
zero divisor routes a heap exception with its tested constant args; a definite
zero has no successful continuation. Ordinary operator sites retain the divisor
residual and operand binding for fact consumers. Non-property overflow in a try
remains a named refusal where it is not ruled out, particularly huge/unbounded
int-to-float conversion. Outside a try, the typed float division transfer
intersects each integer operand with the binary64 conversion range before
computing normal results. The largest convertible integer is
`2**1024 - 2**970 - 1`; the next midpoint rounds to infinity and raises
`OverflowError` before any zero-divisor check. An empty intersection ends the
path; a partial intersection retains every successful conversion without
assuming that the original integer was bounded. This removes raytrace's first
refusal at its unbounded canvas dimensions. With the tuple and primitive-float
changes above, the original nondeterministic task reaches the remaining
recursion refusal at line326; no new whole-task verdict is claimed. `tests/test_absint_numeric_division_contracts.py`
checks both signs of the conversion boundary, normal continuations, all three
division operators and the retained handler refusals against CPython 3.14.4.
Unsupported descriptor receivers, direct NotImplemented
results, joined numeric kinds and bound scalar-slot reads can still refuse.

At operator sites, the typed division visitor records its ZeroDivisionError
obligation once at the existing `divide` site. Direct native-slot calls record
that one obligation as `<qualname>:ZeroDivisionError`, preserving the seven
named contract checks. Contract checks can identify an obligation already recorded
or routed by the model, avoiding a second may-raise row for the same operation.
Direct native-slot calls retain their typed contract qualname through exception
routing. Operator sites retain their original binary-protocol source identity so
witness consumers check the same binding and scope as the source operation.

## Float identity and ratio queries ([internal ref])

`contracts/rules_float.py::FloatConjugate`, `FloatIsInteger` and
`FloatAsIntegerRatio` join the shared
`visitor.py::AbsintVisitor._scalar_query` consumer. Only definite native float
receivers are admitted; int/bool receivers do not acquire float descriptor
semantics. Conjugate retains the float fact; finite integrality/ratio queries
use native results. Nonfinite float sets retain boolean integrality results
and two-int ratio shapes with a positive denominator on successful paths.
NaN and infinities have no successful ratio path; their exact ValueError and
OverflowError args are documented and tested. Calls under a try stay refused
until those handlers can be modeled, so no exceptional continuation disappears.
Bound methods snapshot receiver facts without a mutable variable's value term.

## String alignment arguments and eager fold limits ([internal ref])

`adapters/str_shapes.py::SIGNATURES` admits center/ljust/rjust with builtin int
or bool widths and optional one-character builtin str fills.
`visitor.py::_str_kind_ok` checks the fill length; an unknown string fill
retains TypeError uncertainty. Thus `_abstract_string` can return a real str
fact for unknown receivers and widths without calling any index hooks.
The existing native finite-string path preserves exact values.
The existing `rules_strformat.py::native_limit` continues to bound padding/tab
width magnitudes to 4096 for eager native folds, including zfill/expandtabs.
Larger finite widths are refused; unknown widths use abstract transfers without
allocating strings. Exception handlers and custom index hooks retain their
existing conservative refusal.

The same one-character kind is consumed by
`preimage/visitor.py::_kind_ok` before native forward checks.
`contracts/safety/rules_str.py::str_method_cases` retains its exact scalar-tag
partitions and adds a `len(fill) != 1` TypeError predicate within the string-fill
case, so exception certificates do not confuse an empty/long string with a valid
fill. These consumers share the shape table instead of independent signatures.

## Range search queries ([internal ref])

`rules_range.py::RangeCount` and `RangeIndex` distinguish the native exact
int/bool arithmetic fast path from general comparison search, which may run
arbitrary equality callbacks. `visitor.py::AbsintVisitor._range_query` admits
only the arithmetic path and consumes the tested contract before returning
any result. Exact endpoints and finite needles use CPython's constant-time
search, including negative steps and ranges whose lengths exceed sys.maxsize.
Other integer facts return count in [0,1] or a nonnegative index on successful
paths. `state.py::iv_of_len` handles symbolic and frozen range endpoints.
Missing index ValueError terminates an uncaught path; try calls remain refused.
Direct, bound and unbound forms retain the immutable RangeObj receiver.
General equality needles are not run natively: their count may exceed one
and their exceptions (class and args) can propagate unchanged.

### int literal raise transfers ([internal ref])

The existing `Int` Rule supplies a tested lexical predicate and scoped non-property
raise templates. Absint consumes these for finite exact string values with a
primitive int/bool base, including keyword `base` and resolved callable aliases.
Canonical builtin parsing yields actual integer results. An invalid base/syntax
routes `ValueError` with the contract-formatted one-string args tuple; both
successful alternatives and handler continuations survive. Diagnostic and grammar
disagreement with the tested native behavior refuses analysis.

Only finite strings retained by the scalar domain, up to 640 codepoints and within
the existing Cartesian-product budget, are accepted by this transfer. This bound
cannot exceed an enabled conversion limit. The existing AST constant reader
coarsens string literals longer than 64 characters, so such literals can still
refuse before this transfer; native contract tests nevertheless pin their complete
truncated diagnostics. Unknown strings, open base coercions and process-sensitive
long conversions remain named refusals in handlers. Ordinary integer conversions
retain their existing transfer. The `int` contract now declares its process digit-
limit effect, making generic Avengers stdlib admission conservatively refuse it.

### Dictionary KeyError arguments ([internal ref])

Native dict misses and factory-None defaultdict misses now consume their tested
original-key argument relations and route KeyError with a GC-visible `(key,)`
payload. Existing hash/equality refusals stay in force for open keys; callback
errors are never relabeled as native missing-key errors. Source obligation kinds
and bindings retain their existing identities.

`collections.defaultdict.__missing__` has its own Rule and visitor for direct,
bound and explicit unbound native calls. The supported native None/list factory
states reuse the existing dictionary/list transfers. None stores the uninspected
key in KeyError args, including unhashable key aliases. The native list factory
returns a present value or inserts a fresh list for an absent scalar key, matching
3.14.4 setdefault behavior. Arbitrary factories, unsupported key protocols and
joined bound receivers remain named refusals. Native tests pin factory-before-hash
ordering and callback payloads separately. No target callback runs in the model.

### Native sequence bounds exception transfers (1455)

List/tuple indexed reads preserve the existing `index` source obligation and
consume the contract's native bounds/oversized-int payloads for caught
IndexError. An integer interval spanning both scopes carries both strings;
it cannot establish one exact message. Definitely out-of-range indexed reads
have no normal successor. String reads reuse finite native evaluation and
the existing unknown-string model, now routing primitive-int IndexError
with those contract payloads rather than an opaque exception. No target
`__index__` is executed. Slices, custom index hooks and direct list/tuple
method-call support retain their existing limits. Raise-site metadata alone
is not synthesized inversion proof.

### string.capwords (1455)

A native string helper visit validates binding and exact builtin operand kinds,
evaluates bounded finite alternatives, and retains str for coarse text. It
consumes the tested return contract, all seven property claims and the scoped
empty-separator ValueError payload. Primitive str-or-None shapes route caught
ValueError with literal args. Callback-bearing operands remain named refusals;
the analyzer never invokes their split, truth, iteration or join hooks.

### Heapq return/effect transfers (1455)

The seven added native siblings and existing heapify/heappush/heappop share
a visitor over an exact native list receiver with primitive comparable
numeric or str elements. It consumes the tested return contract and all
seven raise claims, returns None or the element abstraction, and applies
length/content mutation through existing collapse/update_obj/set_len helpers.
Pushpop on an empty heap returns the new item without inserting it. Native
empty pop/replace routes a GC-visible IndexError with contract args; its
normal successor is bottom. New siblings refuse open comparisons and mixed incomparable kinds; the original
three APIs retain their established conservative generic transfer outside the
primitive shape. Scalar element/item snapshots precede length updates, and heap
order itself is not assumed or proved. Partial mutation through a target
comparison exception is outside the admitted primitive transfer.

### Statistics primitive queries (1455)

Seventeen query functions consume their return contracts in a shared visitor
over bounded finite primitive data/configuration. Existing native concrete/
product/abstract helpers retain actual scalar or list results and all seven
property checks. Dataset length is at most8, alternative products at most64,
ints at most4096bits, text at most640codepoints, quantile output count at
most64. Finiteness failures, hooks, nonfinite floats and native exceptional
cases are named refusals. Unknown-list correlations can lose positional
precision; exact tuples retain it. Qualified StatisticsError and handler
transfers are deliberately unsupported until the exception class has its own
consumer, and are never mislabeled as builtin ValueError.

## Native divmod conditions and transfers (1455 batch21)

The existing Rule now distinguishes native payload scopes from arbitrary
__divmod__/__rdivmod__ hooks. With builtin int/bool/float operands after
successful conversion, ZeroDivisionError occurs exactly for a zero divisor,
including signed float zero, with args `('division by zero',)` on CPython
3.14.4. A huge int converted for a mixed float call can instead raise
OverflowError before that zero check. Native unsupported builtin type pairs
and positional-only binding failures have separately tested TypeError args;
these scopes never relabel callback-raised errors or their arbitrary args.

Native ints/bools produce an int pair; a float operand produces a float pair.
Open hooks can return any object, so the unscoped return type remains object
and effects declare callbacks. Bounded primitive alternatives retain exact
native results. The abstract interpreter also supports unbounded integer
results, including the contract-derived `0 <= r < divisor` relation for a
positive divisor. Aliased builtin calls and bool operands use this same
transfer; shadowed user functions keep normal source analysis. Exact native
error payloads are GC-visible and each error has one named call obligation.

Sources: [Python3.14 divmod](https://docs.python.org/3.14/library/functions.html#divmod)
and [CPython3.14 numeric dispatch](https://github.com/python/cpython/blob/3.14/Objects/abstract.c),
with integer/float slots in longobject.c and floatobject.c on that branch.
Native tests and proof controls run exclusively on CPython3.14.4.

The stationary draw planner's value-sensitive refusal remains: divmod is not
a class-only call. Its W41 never-TRUE mutant must stay unproven. Contract
raise metadata is not a synthesized proof; whole-program recursion and RNG
state can remain later blockers in bm_go_nondet.

The opaque exact-byte hint used by program buffers is discarded when joined with
an unknown or different alternative (`state.py::vjoin`); subsumption also keeps
that hint distinct from unqualified top (`state.py::vleq`). It cannot establish an
exact byte value after control-flow merging. Other byte consumers still refuse
unless explicitly modelled.
## random.seed native scopes and abstract transfer (1455 batch22)

The existing seed contract now declares version comparison/truth and seed
subclass encode/decode/iteration/hash callbacks. Their exception classes and
args propagate unchanged. Native unsupported seed types, non-version-2
bytearray hashing and selected positional binding failures have independently
tested TypeError args. Version is compared to1/2 rather than restricted to
int; 2.0 takes the version-2 path. Surrogate text can raise UnicodeEncodeError.

The general effect remains conservative: None/default uses entropy/time,
text can lazily import SHA-512 or use process-salted hashing, and open hooks
can execute user code. Exact explicit builtin int/bool/float seeds only reset
the generator state and cached Gaussian value. No unconditional purity or
recipe admission is inferred.

Absint validates closed scalar versions and supported primitive seed parts,
consumes the tested None return contract, and routes unsupported list/tuple/
dict seed errors with contract-derived args. Open versions/subclasses, bytes
outside the current value domain and surrogate encoding are named refusals.
Coarse text inside a try is refused because encoding errors are not routed.
Native unsupported-type handlers have no normal successor. A withdrawn
contract, return proof or payload relation must withdraw its corresponding
precision. Symexec consumes the same return contract as described below.

Native evidence uses CPython3.14.4:130 new boundaries plus90 existing random
native checks. Focused absint21 and154 existing contract/absint checks pass,
including the existing negative controls; none was weakened.
Sources: [Python3.14 random.seed](https://docs.python.org/3.14/library/random.html#random.seed)
and [CPython3.14 Random.seed](https://github.com/python/cpython/blob/3.14/Lib/random.py),
[CPython3.14 native seed](https://github.com/python/cpython/blob/3.14/Modules/_randommodule.c).

The symexec seed visitor now consumes the tested None return after checking
native module/function identity, valid binding, closed scalar version and an
explicit builtin or symbolic numeric seed. It leaves the user class heap
intact: later modelled RNG reads already cover every generator state. It
never executes the target's seed or changes the analyzer process's RNG.
Default/None entropy, shadowed or replaced functions, open seed/version
hooks, invalid binding and encoding gaps retain named refusals. Function and
module aliases use the same identity gate. Native replay proves the class
and instance fields remain readable and a real assertion witness survives.
Focused symexec23 + existing module57 and class67 checks pass on CPython3.14.4.

### Array numeric storage soundness (1455 batch23)

Numeric joins preserve every int/bool/float alternative before conversion to
an array's floating storage. Binary32 storage rounds both interval endpoints
and drops unrounded symbolic terms. The stdlib pack/unpack conversion uses
the same rounding; its overflow becomes infinity, matching native array
storage at the rounding threshold rather than at the largest finite value.
Known integer-to-float overflow and open instance conversion protocols are
named refusals. Construction, append, extend and indexed stores share this
one conversion path.

Native CPython3.14.4 counterexamples cover three previous wrong TRUEs: a
mixed int/float initializer lost the int alternative, float32 intervals kept
unrounded bounds, and near-overflow values were prematurely set to infinity.
The new36 storage checks and existing28 array tests pass; positive controls
retain float types, finite ranges, point rounding, underflow and saturation.
Constructor contract integration follows this shared conversion fix.

## Array constructor contract checkpoint (1455 batch23)

The constructor Rule now goes through the existing abstract array storage
transfer and requires a tested return contract. Its scoped native TypeError
args distinguish binding, Unicode codepoint type/length, and noniterable
initializer cases. OtherRaiseClaim records bad-typecode, byte alignment,
byte-range boundary and integer-to-float overflow args. Native payload scopes
exclude audit/warning intervention; arbitrary iteration, conversion, warning
and array.__new__ audit callback exceptions keep their original class+args.
The general contract declares process warning/audit dependence and callbacks,
so recipe admission remains conservative.

The abstract numeric/list/tuple/array subset excludes Unicode, bytes, native
long and open conversion protocols. The program cannot install process
hooks through an accepted call; such calls remain unsupported. A known
invalid one-codepoint typecode routes ValueError with the tested contract's
literal args. Withdrawing that relation removes handler precision. Other
constructor errors inside a try remain named refusals. Ordinary import and
function aliases work; rebinding the import alias retains the existing
origin refusal. Array tobytes and symexec construction remain future work.

Main CPython3.14.4: native114 + focused constructor16 pass; standard tooling
regenerated array.array.json. Shared storage36 and existing28 pass on both
lines. This is an incomplete checkpoint: the SV contract port and expanded
importing-file gates remain before readiness. Manager priority is to finish
parity against the landed1470 world port first; no batch23 READY is claimed.
Sources: [Python3.14 array](https://docs.python.org/3.14/library/array.html#array.array)
and [CPython3.14 array implementation](https://github.com/python/cpython/blob/3.14/Modules/arraymodule.c).


## Uninterrupted builtin-string dictionary write/read-back

`Analyzer._dict_straightline_write` proves a local source relation using existing
AST parent links and module bindings. After `d[k] = value`, an immediately
following membership check or simple read of that same dictionary/key binding
observes the write. Only `pass` and matching message-free positive membership
assertions may intervene. No identity is inferred between two `ANY_STR` values.

Admission requires one recent exact builtin `DictObj`, a builtin-string key,
callback-free existing keys, and a stable named scalar/source-function value.
The analyzer must be interpreting the unsliced, unlowered module source. Calls,
mutation, rebinding, different key names, nested blocks, effectful current-read
operands, opaque keys and broader values refuse the relation. Missing-key paths
before the write retain their own checks. The stored function's whole body and
its property goals still run through the ordinary interpreter.

The helper is consumed only by `dict_load` and the dictionary-membership branch
of `compare_split`. It adds no heap/state/abstract-value field, schema, census
shortcut or new goal class. Mapping-display/unpack and structured-callback
functions remain independent and unchanged. Native controls and focused
regressions are in `tests/test_absint_dict_readback.py`.

## Immediate scalar identity covers

An immediate module-level assignment such as
`c = identity(n) if sv.nondet_bool() else identity(s)` preserves the native
object passed through either plain source function. `_scalar_identity_cover`
proves a following `assert c is n or c is s` from that source flow; it never
infers identity from equal integer intervals or arbitrary-string summaries.

Admission requires unsliced, unlowered source, bound builtin scalar names and
an immediate single-name assignment. Each arm is a source name or one plain
source function with exactly one positional argument and the sole statement
`return parameter`. The current function object, module origin and binding
must authenticate; defaults, annotations, decorators, classes and broader
bodies refuse. The selector is a bound builtin boolean or an authenticated
zero-argument verifier boolean draw. The assignment cannot rebind a source,
callee or selector name. No statement may intervene before the assertion.

Only `cond`'s admitted module-assertion `or` case consumes the theorem.
Earlier execution and every earlier/later property goal remain in the ordinary
interpreter. No identity facts persist in abstract values, heap or state.
Nested blocks, effectful operands, independent draws, stale bindings, floats
and unsupported source shapes retain the existing refusal behavior. See
`tests/test_absint_scalar_identity.py` for the boundaries.

## Closed builtin dictionary removal

`dict.pop(key[, default])` consumes the current tested return and raise claims
for one recent, exact builtin dictionary and one constant int/bool/str/None key.
All receiver keys must be builtin non-float scalars. A successful lookup removes
the key from the current heap object and returns its original value; returned
references remain aliases, including references back to the receiver. Existing
loads, membership, length and live views read that updated heap. Branch joins
retain only keys present on every path. No persistent key fact is added.

Arguments still execute in Python order before lookup. A missing key returns
the explicit default, or records `KeyError` and ends that path when no default
was supplied. The optional default must be primitive and non-float because an
unused temporary object can run cleanup. Open or summary dictionaries, unknown
keys, subclass protocols, float results and calls under exception handlers stay
refused. The empty-dict native shortcut that skips even unhashable-key hashing
is deliberately outside this scalar-key consumer.

The direct assignment `undefined_parameters = kvs.pop(catch_all_field.name)`
from dataclasses-json is covered with a closed mapping-valued entry, mutation
through the returned alias, key removal and native negative controls. This is
statement consumer evidence, not a proof of the enclosing function or an
inversion or portfolio gain. Reference `is` comparison remains conservative;
alias effects are checked without extending comparison semantics.

## Joining a closed string-key view

`str.join` also consumes one builtin `dict.keys()` view whose live source is
one exact builtin dictionary containing only string keys. The separator must
be a definite builtin string. All argument expressions run first; the model
then checks the current source heap, including changes through aliases or
argument bodies. It neither reads nor converts dictionary values.

The result is an arbitrary string. Abstract state joins do not preserve
dictionary insertion order, so this consumer supplies no content, ordering or
length facts. It requires the current tested string return contract and leaves withdrawn or
unknown raise claims unproven. Subclass protocols, non-string keys, open
dictionaries, other view kinds, explicit iterators and handler continuations
retain their previous refusals. General iteration and string models are
unchanged; no target code or view is executed or materialized by this model.

The source-matched example is `', '.join(CREDIT_CARDS.keys())` in
python-string-utils. A statement kernel uses the six original key names and
inert values. The original regex-valued registry runs natively but its regex
prelude still refuses before this call in absint. These checks establish this
statement consumer only, not the enclosing validation function or an inversion
or portfolio gain. See `tests/test_absint_str_join_keys.py`.

## Plain automatic string format fields

When finite native evaluation cannot determine string argument contents,
`str.format` can still prove safety for finite literal templates containing
only automatic replacement fields with an empty format spec and no conversion.
Escaped braces and unused positional string arguments are allowed. Every
template alternative must parse and have enough arguments; all arguments,
including unused ones, must be definite builtin strings. `Formatter.parse`
examines only template syntax and executes no target code.

The consumer requires the current tested string return contract and leaves
untested or unknown raise claims unproven. It returns an arbitrary string with
no content or length facts. The earlier finite native path retains its exact
outputs. Explicit field indices, names, attributes, subscripts, conversions,
nonempty or nested specs, keyword arguments, open hooks and handler
continuations keep their existing conservative behavior.

The source-matched assignment is
`msg = 'Expected "str", received "{}"'.format(type_name)` in python-string-utils.
Its statement kernel declares a builtin string `type_name`; the preceding type
name lookup and enclosing exception constructor are not proved. This is
statement consumer evidence, not inversion, a native kill or a portfolio gain.
See `tests/test_absint_str_auto_format.py` for native and refusal boundaries.

## Complete finite statistics data and positional argument expansion

`AbsintVisitor.visit_statistics_query` consumes all represented finite primitive
data for the 17 existing statistics queries, without an eight-element data cap.
It checks the analysis CPU deadline before tuple-pool materialization, while
checking every primitive, after each native fold, and after result abstraction.
Memory exhaustion refuses the transfer. Cartesian enumeration shares the analysis
deadline; scalar size/finite-float guards, quantiles output-count bound, tested
contract/return requirements, and qualified exception refusals still apply.
No symbolic value receives a concrete default, and exact native float results
do not imply a theorem about arbitrary floats. Segmented lists and tuples
without recorded items retain the existing concrete-materialization refusals.

`Analyzer.call_positional_args` expands known-length builtin tuples and lists
incrementally under the same analysis deadline, without a 64-argument cap or a
length-sized repeated summary pool. Every slot and every star segment is
processed; exhaustion refuses the entire call. Tuple slots retain their values;
list summaries retain their existing loss of positional correlation. Argument
order, borrowed aliases and the callee's actual arity checks are preserved.
Unknown lengths, custom iterables and starred method-lookup cases retain their
existing refusals. Neither change supplies a new representation or proof rule.

### Integer comparison partitions for primitive type paths

The optional case-split retry can partition a drawn integer at literal comparison
boundaries when its domain is not a small finite enumeration. Every literal c
contributes boundaries c and c+1, clipped only to the already proven interval.
The intervals cover every input, including both unbounded tails; no sampled value
stands for an interval and no later predicate is assumed constant in one. The
existing conditional transfer analyzes each interval normally, and the existing
block continuation joins all outcomes. This retains primitive type correlations
across repeated comparisons and calls without adding state fields or a type cache.

Only the existing assignment/draw/loop/try/module eligibility applies. Literal
comparisons are a source-based refinement heuristic, not a restriction on allowed
integer values. The existing per-run split allowance bounds optional refinement;
when it cannot cover every interval, analysis keeps the joined domain. The shared
CPU deadline is checked during source scanning and before each outcome. Unknown
comparison boundaries and other correlations may still remain UNKNOWN.

## Nested class namespaces

Class statements at module scope or inside another class reuse the same class
construction transfer. Each body runs in its own namespace. Free names in an
inner class body or method resolve in the defining module, not in the enclosing
class namespace; bases and defaults evaluate in their actual enclosing scope.
Class statements bind their resulting class into that enclosing namespace only
after their body completes. Distinct source sites remain distinct class objects.
There is no nesting-depth or sibling-count bound in this admission rule; the
normal analysis deadline and allocation-site guards still apply.

Classes defined inside functions remain UNKNOWN because captured lexical cells
are not represented. Nested class bodies and methods needing implicit
``__class__`` cells also remain UNKNOWN under compiler-authenticated closure
guards, including an outer class cell read before it is filled. Existing multiple-inheritance,
metaclass, decorator, unsupported-dunder and repeated-class-allocation refusals
are retained. Tests compare native class namespaces, globals, header bases,
nested exception identities and false assertions, including depths and sibling
counts 0, 1, 2, 3, 9, 17 and 65.


## Complete single-inheritance lookup

Class attribute, slot and builtin-base queries traverse the complete represented
single-inheritance chain. They never report an absent attribute, missing slot or
absent builtin base after a fixed-depth prefix. Each step checks the real analysis
deadline, and an unrepresentable cycle refuses the analysis. A native depth-65
inherited attribute read with a leaf-local constructor previously dropped a
reachable assertion after the 50-class prefix; the complete walk preserves it.
Native controls cover depths 0, 1, 2, 3, 9, 17, 49, 50, 51, 65 and 129, ordinary
attributes, slots, builtin exception bases and false-assertion twins. Existing
multiple-inheritance, metaclass and unsupported-descriptor refusals still apply.


## Finite user iterators in for loops

The for-loop consumer executes represented user __iter__ and __next__ methods
through ordinary method dispatch. Its private exception frame consumes only a
StopIteration leaving __next__; body exceptions, break, continue, return and
else retain their normal routing. Every continuing abstract state is consumed
until it stops. The real deadline refuses nonterminating or expensive cases;
no iteration count is used as an input-domain bound and no prefix is a proof.
A for loop does not ask for a length hint. Existing builtin iterator paths are
preserved, and other consumers still refuse user iterator results.

Raising the canonical StopIteration class uses its native no-argument
construction and the existing opaque exception tag. Explicit specialized
constructor calls, payload reads, user StopIteration subclasses, unknown
iterator alternatives, unsupported descriptors and unrepresented method bodies
remain UNKNOWN. Tests compare native lengths 0, 1, 2, 3, 9, 17 and 65, primitive
and callable elements, inherited methods, body/next errors, pending exits,
reused/shared iterators and false assertions. No heap/state/schema is added.
## Missing scalar methods are looked up before call arguments

For a wholly represented primitive receiver, a method name absent from every
possible scalar kind is resolved through the existing attribute-load transfer
before positional or keyword arguments are evaluated. Native AttributeError
therefore reaches its handler, and argument effects that Python never evaluates
are not introduced. The guard reuses the existing builtin attribute inventory;
it is independent of the method spelling, value magnitude and argument count.

Present methods, user objects, unknown values and mixed-present scalar alternatives
retain their existing dispatch/refusal. Known missing attributes also fail before starred/keyword-spread expressions;
present starred method calls retain their existing lookup-order refusal. No numeric or container representation is widened: a
program-grown integer above machine size uses the same kind/attribute decision.
Native controls cover sizes/counts 0, 1, 2, 3, 9, 17 and 65, all five scalar kinds,
values beyond 200 bits, handler/argument effects and false assertions.


## Nonnegative integer power lower bounds

A possible zero base has lower bound zero whenever the nonnegative exponent may
be positive; zero to a certainly-zero exponent has lower bound one. A positive
integer base has lower bound at least one for every nonnegative exponent size.
The existing cheap precise bound remains the default; large bounds use these
constant facts rather than constructing huge exact powers. No exponent or base
size is rejected. Negative bases retain their existing sign-unknown summary,
and exact nonlinear identities can still remain unproved.

Native controls cover bool/int bases, exponent sizes 0, 1, 2, 3, 9, 17, 65 and 129,
cheap exponents beyond 200 bits, and false assertions. The previous lower bound
used 0**0 across a range containing positive exponents and could falsely prove
0**nondet_bool()>0. That path now remains open.

### Mixed builtin checks and starred assignment consumers

The six exact builtin `check_type` hints (`int`, `bool`, `float`, `str`, `list`,
`dict`) accept an abstract value with multiple represented kinds. Every kind
must match for the TypeError check to be proven; `bool` also matches `int`.
A possible match keeps the original normal-state over-approximation. Only a
value with no matching alternative ends that path. Unknown/custom values and
mismatching containers whose nested representation could run hooks still
refuse. This does not infer a correlation from a prior branch or refine an
argument binding after the check.

Starred assignment over exact builtin list/tuple objects uses the existing
sequence element and length summaries. Exact tuple positions supply the fixed
prefix/suffix; list summaries conservatively supply every fixed destination.
The middle becomes a fresh builtin list with all possible remaining elements
and the remaining length interval. All values are saved before any target is
assigned, including nested destinations and mutations through aliases. Empty
middles, every finite source length and target arity, and too-short sources are
handled without a size cutoff. A too-short source raises ValueError before
assigning targets. Caught unpacking exceptions remain UNKNOWN because that
ValueError is not routed here. Other iterable representations, custom iteration,
and unavailable type information remain UNKNOWN rather than assuming their
cursor or length-hint effects. The analyzer deadline bounds work, never input
sizes. Existing integer transfers retain their default representation and widen
according to the program's operations; unpacking adds no integer bound.

Files and how they relate:

- `solve/absint/interp.py::_verifier_check_type_value_kinds` and
  `m__sv_verifier_check_type` account for every represented matching/mismatching
  builtin kind without changing the value or state schema.
- `Analyzer.unpack` dispatches a starred target to `_unpack_starred`, which reuses
  `len_lin`, `all_elems`, `new_list`, and `assign` for snapshotting and assignment.
- `tests/test_absint_mixed_type_checks.py` compares native success/error paths,
  bool/int subtyping, formatting refusals, handlers, and false continuations.
- `tests/test_absint_starred_unpack.py` compares native finite sizes through 65,
  prefix/suffix arities, empty middles, alias mutation, nested targets, integer
  growth, unrepresented iterables, and false twins.

### Finite recursive call interpretation

A repeated function body is not itself a reason to refuse a call. A higher-order
function can re-enter its own body with a different callable or fewer variadic
arguments and finish normally. `Analyzer.call_func` uses the ordinary argument
binding, fresh activation/frame, closure-owner checks, body execution, exit
routing and return-state transfer for every such call. Caller locals remain in
their own frames; a re-entry never reuses its caller's local bindings.

There is no fixed call-depth or input-size cutoff. The unique-call depth removal
reuses lane 1493's qualified `abc01019` interpreter slice. Each recursive call
checks the existing analyzer deadline. A spent deadline or host interpreter
RecursionError returns UNKNOWN through the ordinary public proof boundary;
no unfinished execution or prefix is a proof. This is interpretation of finite
activations, not an inductive summary or a termination claim. Recursion that
needs a summary can still exhaust the available resources and remain UNKNOWN.
Inside an active try, a recursive call now has an additional possible
RecursionError exit, routed from a copy of the caller state through the existing
handler/finally machinery. The ordinary call is still interpreted: stack
exhaustion is never a reason to cut its continuation or certify a prefix. This
sound overapproximation may retain a runtime stack failure even for a shallow
call; it makes no assumption about available stack depth. Existing reflection,
expired closure and unsupported-operation refusals remain unchanged.

Three numeric operations also retain their non-property exits inside try.
Mixed float addition, subtraction and multiplication route possible integer
conversion OverflowError. Numeric int conversion routes NaN ValueError and
infinity OverflowError. Primitive min/max with two or more positional numeric
arguments, and abs on one numeric argument, need no user hook or non-property
raise; their existing tested contracts still gate the transfer. Numeric
alternatives may mix bool, arbitrary integers and floats, including nonfinite
floats. Other bindings, iterable extrema and custom protocols keep their
existing general refusal until their raises are routed. There is no arity or
integer-size cap. Integer-to-float interval conversion clamps both ends before
native conversion; wholly out-of-range intervals retain OverflowError and a
conservative saturated normal result, never a host conversion crash.

Heap rewrites reuse an immutable function object when defaults, keyword defaults
and captures are unchanged. Float-term stripping avoids constructing a float
view when the stored value has no term. These preserve the same abstract facts;
they neither skip a state transfer nor add a proof assumption.

Files and how they relate:

- `solve/absint/interp.py::Analyzer.call_func` admits repeated bodies, routes
  possible recursive stack exits and retains all frame/binding/closure/return transfers.
- `solve/absint/__init__.py::_prove_once` already converts resource recursion
  failure to UNKNOWN; its behavior is unchanged.
- `tests/test_absint_recursive_calls.py` compares native sizes 0,1,2,3,17,65,
  variadic/keyword re-entry, represented value kinds, frame and capture lifetime,
  200-bit operation-grown results, actual failure paths and resource controls.
- `tests/test_absint.py` keeps a genuinely unfinished recursive program in its
  refusal controls; finite recursion now has separate positive and false twins.
- `tests/test_absint_recursive_try.py` checks native handler matching, finally
  restoration, failure continuations, numeric arities through 129, wide growth,
  nonfinite conversions and contract withdrawal. Resource exhaustion remains
  UNKNOWN even on a finite native example.
- `tests/test_solve_values.py` covers integer-to-float endpoint conversion through
  20000-bit magnitudes and intervals lying wholly on either overflow side.

### Exact function values at verifier type checks

The builtin scalar/list/dict hint model recognizes exact Python function values
(including lambdas, closures and generator function definitions) as
function values, without calling their bodies. They mismatch these hints and
terminate the normal path with TypeError. Python functions have a builtin repr
that cannot invoke a user callback; its address-bearing text remains opaque,
and checker calls inside try still refuse, including caught payload reads.
Async definitions retain the existing unsupported-statement refusal. Bound methods, custom
callable objects and mismatching nested containers retain their existing repr
refusals. Function signatures and alternative counts have no domain limit;
the ordinary analysis deadline still ends an exhausted run with UNKNOWN.

### Exact polynomial floor division

`solve/absint/polynomial.py::affine_quotient` expands pure integer `+`, `-`,
`*` and unary signs into sparse polynomials, then uses exact integer polynomial
long division. It returns a linear form only when the remainder is zero and
all quotient terms are affine. No degree, input size or coefficient bound is
imposed; the current analysis CPU deadline bounds expansion and division.
Other syntax, noninteger values, nonexact division, a zero polynomial divisor,
and nonaffine quotients supply no refinement.

`Analyzer._int_binop` keeps linear constant division on its cheap coefficient
path and only expands source expressions containing multiplication. A saved
operand can have a fresh linear name after its nonlinear relationship was lost. It retains original operand evaluation and the existing
zero-divisor check. The affine result describes normal continuations only:
if `P = D*Q`, Python's `P//D == Q` for every nonzero integer `D`, of either
sign. Source names are read only after rejecting calls, attributes and other
potential effects, and every name must denote only exact integer/bool values
with an existing linear form. Float and user protocol paths keep their models.
No target executes, and no value-equivalence fact is promoted to identity.

### Stable absolute-value factors in integer quotients

After ordinary operand evaluation, primitive contracts and zero-divisor checks,
`Analyzer._product_quotient` can cancel repeated authenticated builtin `abs`
factors through the existing exact polynomial solver. Its source slice admits
only stable integer/bool names, exact integer constants and pure arithmetic.
All formal abs factors must cancel out of the affine result; none become state
variables. Rebinding, reflection, callbacks, other kinds, surviving factors and
nonaffine forms retain the existing result. There is no factor-count or width
limit; the analysis deadline yields UNKNOWN when spent. Native controls and
qualification: `docs/recon/2026-11-03/1493-product-quotient-t105.md`.

### Path-sensitive module copy identity

When an unresolved module `is`/`is not` comparison reads two names, the analyzer
tracks their object identities in a separate component of the existing zone.
Private variables denote `id` of root-module bindings as mathematical integers;
only executed plain name copies add equality edges. They never connect to
program numeric values or affine recipes, and no address is guessed or exposed.
The existing join and widening conservatively combine these relations.

Assignment saves the RHS identity before overwriting any chained targets. Root
writes and deletion discard the old slot identity; ordinary calls and imports
discard identity facts conservatively. Unknown callbacks therefore retain no
old alias assumption. These are conditional facts of the existing closed model:
user finalizers, weakref cleanup and generator protocols still refuse when the
model cannot account for their effects. The native counterguards remain active.

The source-preserving static copy pass remains the fast path. Only unresolved
module name comparisons activate zone tracking; default programs and already
proved straight copies allocate no identity variables. Function/local scopes,
non-name copy forms and facts crossing calls can remain UNKNOWN when existing
rules cannot prove them. There is no object-kind, size, chain or loop-trip cap.
CPU exhaustion ends in UNKNOWN. Source facts and invariants project only actual
source names and numeric components, never these private identity variables.

### Closed ABC classes

Canonical `abc.ABC` selects ABCMeta's frozen abstract-name calculation within
the existing single-inheritance class model. `abc.abstractmethod` marks an
exact recent source function and returns the same function; descriptor outer
staticmethod/classmethod plus inner abstractmethod is supported. Abstract
markers and class abstract sets are immutable heap metadata; conflicting joins
refuse. Concrete overrides remove inherited abstract names, while an ordinary
class using abstractmethod alone remains instantiable.

Instantiation checks the abstract set after rejecting unsupported class
machinery and before allocation, argument binding to `__init__`, or its body.
Caller argument evaluation remains first. A missing implementation routes
TypeError with opaque text, so observed exception messages refuse. There is no
method-count or inheritance-depth bound; the ordinary run budget applies.

Custom ABC hooks/metaclasses, virtual registration, mutation of ABC machinery,
private-name mangling and arbitrary abstract-marker descriptors remain UNKNOWN.
ABC's implicit attributes are an unknown MRO surface, preventing a false
AttributeError for register or internal ABC fields. The source base hierarchy
still proves nominal instance relationships; unsupported custom subclass hooks
are refused before a class can be used. No native target class is created by
the proof, and no ordinary class or numeric representation changes.

Absent ABC metadata retains ordinary class-like heap behavior; MRO deadline and
cycle guards still complete or refuse the full walk.

### Closed collections.namedtuple declarations

The factory's tested contract creates a `NamedTupleClassObj` with exact field
names and defaults, and construction binds fields into a `NamedTupleObj` tuple
payload. Field counts, constructor arities and tuple sizes have no domain cap.
Direct method syntax honors fields shadowing tuple methods, including count
and index. Existing tuple indexing, iteration and length transfers consume the payload;
class metadata survives joins, frame-value rewriting and garbage collection.
Instance and subclass checks distinguish the generated class from plain tuple.
Tuple conversion produces a plain tuple; native materialization refuses to
erase namedtuple identity during formatting or other concrete observations.

Declarations accept a known builtin string, a fixed tuple of known strings,
or anonymous literal string lists authenticated against their actual allocation
site. No re-evaluation or native target class creation occurs. Nonempty aliased
or computed lists refuse because the current list abstraction has no positions.
Tuple defaults retain their values and references; other default iterables,
unknown declaration strings, identifier normalization requiring a different
constructor binding, and callback-bearing inputs remain UNKNOWN. `rename` is
normalized only when its callback-free truth is known. Module metadata accepts
None or a known string; unmodeled class attributes remain UNKNOWN.

The generated class is sealed in this model: mutations, subclass definitions,
replacement methods, descriptors and constructor errors inside try blocks
refuse. Summary class identities cannot establish instance/subclass equality.
These refusals protect the inherited tuple protocols and native exception
payloads without claiming the rest of the generated class API is modeled.

### Nullable slice bounds

A missing bound or a value that is definitely None uses the native default
endpoint. A joined None/bool or another mixed value is not definitely None:
without a modeled normalized endpoint, retain the conservative slice-length
bounds. The fallback never treats a possible valid bool index as the default.
This applies at every finite list, tuple and array length; pure None and empty
sequence behavior is unchanged.

### Native ZIP output summaries

`ZipObj` retains live native sources and strictness within the iterator family.
Its dedicated State transport preserves its provenance. Reviewed consumers
(for, list, tuple and dict) overapproximate every remaining output, preserve
tuple columns and demote row summaries before exposing them. They infer no
rewound cursor, fixed output count, key-presence relation or fresh-row alias.
Strict exceptions occur at consumption; callbacks, unsupported attributes and
unreviewed consumers stay UNKNOWN. No arity, length or width bound is added.
See `docs/recon/2026-11-03/1493-zip-summary-t106.md`.

### Native ENUMERATE output summaries

`EnumerateObj` keeps a live source and immutable integer start bounds. Dedicated
State transport and reviewed for/list/tuple/dict consumers preserve tuple field
types, including direct composition with native ZIP. Every remaining index is
a native int at or above the captured lower bound; no cursor, rewind or exact
source-position correlation is inferred. Summary rows are demoted before use.
Opaque callbacks, unsupported native representations and other consumers remain
UNKNOWN. Sizes and signed start width have no fixed cutoff.
See `docs/recon/2026-11-03/1493-enumerate-summary-t107.md`.

### ZIP remaining counts and native ENUMERATE prefix bounds

`ZipObj.fresh` is a guarantee of no consumption, not a cursor position. Every
reviewed row consumer clears it; joins retain it only when both sides guarantee
freshness, and heap redirects preserve it. Materializing an unconsumed zip over
native sequences has the minimum of their current lengths. An existing input
iterator has lower bound zero even when the zip is fresh. Consumed zips also
keep lower bound zero. Remaining output is bounded above by each source's
current native length. Source length anchors are captured before allocation,
so heap reuse cannot bind an old source length to a new result. No size or arity
limit is introduced, and exhaustion of the existing analysis deadline is UNKNOWN.

On the existing guaranteed-element iteration path, enumerate indices over native
monotone sequence cursors are below the captured start upper bound plus the
source length. A private upper bound can constrain many rows without equating
their index values. Materializing a possibly empty enumerate uses interval
bounds only; it must not assume that any input element exists. Set/dict cursor
histories are excluded from prefix bounds: mutation can change their traversal
behavior. Unknown hooks and cursor relations retain their conservative rules.

## Callback-free tuple dictionary keys

Dictionary displays, writes, lookups and membership validate exact builtin tuple
key graphs without calling target code. Every leaf must be a builtin scalar;
unknown leaves, mutable containers, tuple subclasses and user hash/equality hooks
remain UNKNOWN. The walk is iterative, with the analysis deadline rather than an
arity or nesting bound. Finite constants use `native.concrete`'s shared per-run
work budget; exhaustion loses constant precision without excluding keys.

Known tuple keys use Python's builtin hashing/equality, including numeric key
coalescing. Unknown tuple writes conservatively join values at every potentially
equal known tuple key. No DictObj schema changes: when a constant tuple key must
be projected back to V without a heap address, it becomes TOP, never None. Thus
key iteration and joins may lose precision, while never inventing a scalar key.
Tests cover native tuple sizes 0, 1, 2, 3, 17, 65 and 129, nesting through 129,
130-bit integer leaves, bool/float/Unicode/None, aliases and weak-update negatives.
Scalar key validation retains its direct path; only tuple graphs allocate a
validation worklist and consult its deadline.

## Range allocation summaries

Repeated ranges at one allocation site already demote their immutable endpoint zone
variables to intervals. Same-step summaries now reuse the existing interval
join/widen/inclusion operations, rather than requiring identical endpoints. This
covers every endpoint magnitude and cardinality; no range elements are enumerated.
Unchanged symbolic endpoints retain the original fast path. Summary intervals do
not become zone variable names during GC.

Consumers retain their existing interval endpoint interpretation. Lost relations
can leave checks UNKNOWN. Differing steps at one summary site still refuse because
this representation stores one exact step; the constructor's existing unknown-step
and user `__index__` gaps also remain UNKNOWN. Tests compare native ranges of sizes
0, 1, 2, 3, 17, 65 and 129, both signs of step, and 130-bit endpoints/steps, including
join/widen containment and wrong-result controls.

### Alternative builtin slice receivers

A builtin slice receiver may denote several possible list, tuple or supported
numeric-array heap objects. The receiver is selected once at runtime. The
abstract transfer forks the incoming state for each alternative, runs the same
slice model from the same incoming allocation epoch, and joins the results. It
unions the resulting allocation epochs before continuing the statement. This
keeps recency and in-flight-reference guards active, while mutually exclusive
receivers no longer look like repeated allocations on one execution. Index and
bound expressions were already evaluated and are not repeated. Scalar receiver
alternatives retain their existing type/error handling. The alternative count
has no fixed cap; the analyzer deadline is checked between alternatives and
exhaustion yields UNKNOWN.

`tests/test_absint_slice_alternatives.py` compares native lengths through 129,
forward/reverse slices, lists, tuples, integer/float arrays and mixed receivers.
Mutation checks retain the independence of a list slice from its source, and
false continuations remain UNKNOWN. This changes allocation handling only: it
does not add sequence ordering, fixed-length starred-argument correlations,
new slice-bound protocols or new array typecodes.

### Primitive direct tail loops

A direct tail call with an explicit nonrecursive return can be interpreted as a
loop in the proof engine. Parameter arguments evaluate together before any
parameter changes. Each original assertion and return remains on its original
path; ordinary fallthrough returns None. The ordinary loop fixpoint and checked
narrowing must prove all iterations. There is no input, arity, numeric-width or
call-depth bound, and no termination assumption. An unfinished fixpoint or spent
resource budget yields UNKNOWN.

This interpretation is used only when all bound parameters and referenced data
globals are exact builtin primitive kinds, the global recursive callee still
names the same source function, and any allowed draw module is the authenticated
_sv_verifier. The body has no local stores, closures, user calls, frame inspection,
exception handlers/finalizers or mutable operands. An active caller try also
keeps the ordinary calls: its handlers could observe RecursionError. Removing
unobservable stack exits otherwise adds possible executions, a sound
approximation for proving exception safety. It never supplies a native FALSE
witness; coverage-guided input generation and symbolic execution still replay
violations against the original source through the shared obligation table.

Variadic/keyword-only recursive bindings, captured frames, non-tail calls,
implicit-only base exits and other source shapes keep the existing interpreter.
If that interpretation cannot finish, it returns UNKNOWN. Tests compare native
original/lowered functions at sizes and arities 0,1,2,3,17,65,129; None, bool,
arbitrary ints, finite/nonfinite floats and full Unicode strings; simultaneous
parameter updates, failed continuations, rebound callees and source handlers.
The deep safe Ion source gains an inductive proof; its unsafe counterpart stays
UNKNOWN. Native failure under an artificially raised recursion limit is not a
qualified counterexample.

## Closed identity map unpacking

A source-authenticated `map` of a closure/default-free single-parameter identity
function over a literal display (list or tuple; correction: the first implementation, by an AI lane (b50538e4), was WRONG: it required `ast.List`, so the identical tuple lost the proof) of pure module names/constants can preserve each
position for its sole unpack consumer. The result remains an `IteratorObj`;
only its private positional snapshot uses tuple storage. The callback is never
executed early. Mutable element references remain aliases, and scalar values
are captured before later rebinding. Primitive `strict` truth has no effect on
a one-source map. There is no length, arity-of-unpack or integer-width cutoff.

The whole-source binding/read audit requires one module-level (possibly `if`)
binding and one plain unpack outside loops/functions. It reuses the existing
frozen-list cross-file/global-channel exclusions. A second read, alias, hidden
global channel, unsupported binding or unavailable import closure refuses.
Reference-producing callbacks need heap reference remapping; repeated consumption needs a cursor;
effectful iterable/keyword expressions and nonliteral sources need positional
capture during argument evaluation. These gaps stay UNKNOWN. The source audit loads lazily from `map_identity.py` only on actual map calls.
No eager rewrite is exported, no target function executes in SOLVE, and no
heap/state schema changes.

Native controls exercise sizes 0, 1, 2, 3, 17, 65 and 129; primitive, Unicode,
wide-integer and nested mutable/immutable payloads; scalar rebinding, shared
objects, callback timing, strict binding, alias/second-consumer refusals and
reachable wrong assertions.

### A last `*sequence` of unknown length into a `*parameter`

`f(a, b, *seq)` where `seq` is one builtin list or tuple whose length is not a single number was refused
(`*args requires a fixed sequence length`). It is admitted when it is the LAST positional argument and
every callee is a user function (`ob.kind == "func"`) with a `*parameter` and with no named positional
parameter left for the tail: `len(posonlyargs + args) <= number of arguments before *seq`
(`Analyzer.call_with_star_tail`, fed by `call_positional_args(..., tail_out=)`, bound by `bind_args(..., star_tail=)`).
Then the arity cannot depend on the sequence's length: the named parameters are filled by the earlier
arguments, a keyword naming one of them is the same "multiple values" `TypeError` for every length,
and the `*parameter` is a tuple of length `(arguments beyond the named parameters) + [lo, hi]` (`hi` open
when the list's length is) over the join of those arguments and the elements `seq` may hold. The sequence
is snapshotted at the call (elements borrowed as in the fixed case, so an object in the tail stays aliased
with the tuple's element). A bound method, class, builtin or scalar callee, a function without a
`*parameter`, and a function whose named parameters the tail could fill keep the refusal: their arity or
protocol depends on the length. A joined sequence value of several heap objects is still refused (one exact
list or tuple is required). `tests/test_absint_starargs.py` has the native controls and the never-TRUE twins
(a tail that would bind a named parameter, wrong-typed elements, an aliased element mutated by the callee,
a callee without `*parameter`, a builtin or a class).
Measured use: `type-eval-py/args_multiple` (sv-benchmarks f0624a58) dynamic-typing goes from the refusal to
TRUE. Its assertion-safety pair and the `_unsafe` assertion pair stay UNKNOWN (`b == x + y + sum(extra)`
equates a loop fold with a builtin sum, a relation the interval domain does not have).

### Shared choices in starred calls

When callee selection and a starred argument read a shared name whose abstract
value has several possible heap receivers, the call transfer partitions the
binding by those alternatives before evaluating the original call. Each branch
starts from the same allocation epoch and borrowed-reference set; the results,
states and scratch are joined afterward. The bound name is narrowed in its
actual module, local, class or live captured scope. All operand, keyword and
callback effects still occur in their original order. There is no receiver-count
or arity cutoff; spent transfer time yields UNKNOWN.

Fixed builtin tuple slices preserve per-position values when the bound components
are exact int/bool/None alternatives. Arbitrary signed integer magnitudes,
negative steps and all finite tuple lengths use native builtin slice adjustment
on model values. A zero-step alternative has no normal continuation; active try
contexts still keep the existing named refusal. Unknown bounds retain the full
existing summary transfer, rather than truncating the domain. This supplies the
callee/argument positions that a joined element summary loses.

For a dynamic-typing query after such a shared-choice call, scalar integer stores
retain interval facts without namespace-variable relations. This is a sound
weakening, never a new assumption: all value and exception transfers still run,
and lost relations may leave a check UNKNOWN. Other queries and programs that do
not reach the shared-choice transfer keep the normal numeric representation.

`tests/test_absint_starred_correlations.py` compares native finite arities and
slice sizes 0,1,2,3,17,65,129, huge signed slice bounds, bool/None bounds, zero and
negative steps, original element aliases, wrong bindings, closures and callee /
argument mutation order. Method calls with starred arguments and opaque index
protocols retain the existing general refusals. The Scimark frontier moves beyond
slice allocation and starred length; its typing query next needs the suspended
`Array2D.indexes` generator, and incomplete numerical analyses still time out.

### Square-root exception routes

The existing tested math.sqrt transfer now admits an active try for exact
builtin int/bool/float operands, including their unions. Conversion of an
unbounded integer to float has an additional possible OverflowError exit;
negative real-domain inputs have a ValueError exit. Both use the same source
handler/finally routing as other checked operations. The normal numeric result
is still computed by the existing interval transfer; a bottom result has no
normal successor. NaN and positive infinity retain their native normal paths.
Wholly out-of-range integers use the existing conservative saturated float
abstraction plus the overflow exit, which may lose precision but never excludes
a native result or exception. The normal nonraising case keeps its fast path.

No numeric-width or magnitude bound is introduced. Custom conversion protocols,
non-numeric values, unsupported bindings and exact exception payloads retain
the adapter's existing named general refusals or unknown values. Contract
withdrawal still removes the transfer. Tests compare native scalar values,
integer bit lengths 0,1,2,3,17,65,129 and 20000-bit growth, Unicode nonnumeric
inputs, matching/unmatched handlers, finally effects and actual failure controls.


## Closed scalar map callbacks

`map_identity.model` also captures a plain source callback and scalar positional
operands for a sole module-level unpack. Construction performs no callback.
The new scalar-map branch in `Analyzer.unpack` delegates to
`map_identity.consume`, which calls the ordinary `Analyzer.call` transfer once
per operand, in native order, and snapshots each scalar result before any
assignment target is stored. Callback effects, binding checks and exceptions
therefore occur at consumption; saved callbacks and construction-time operands
survive later rebinding. No target function executes inside SOLVE.

Every finite number of int/bool/float/str/None operands and scalar results is
supported subject to the shared run budget. Ordinary callback binding handles
positional, keyword-default and variadic signatures, with primitive defaults.
The identity fast path retains its original reference payload support.
Closures or reference defaults, non-identity heap operands/results, and unknown
results remain UNKNOWN because their lifetime/remapping needs additional state.
Aliases, repeated consumption, effectful captures and caught unpack exceptions
retain the existing source/continuation refusals. Non-identity source/target
length mismatches also remain UNKNOWN: a callback's StopIteration can change
whether native unpack succeeds. On a matched unpack outside a handler, such
an early stop terminates via native ValueError before subsequent statements.

This adds no caller-return refinement or new string/nonlinear theory. The
original mixed string/None map still leaves its caller equality unproved;
string concatenation length can prove even where literal-content folding
cannot. Tests compare native values and exception order at sizes
0, 1, 2, 3, 17, 65 and 129, cover default counts through 129, retained scalar
results across callback effects, 63-to-64 and 130-to-131-bit growth, and preserve
reachable wrong assertions and unsupported continuation controls.

### Generator lexical scope

Suspension detection examines only the current function or lambda body. A nested
function, lambda or class body does not make its enclosing function a generator.
Nested defaults, decorators, class bases and class keyword expressions still run
in the enclosing scope, so a yield there keeps that enclosing function suspended.
The root function's own header belongs to its caller and is excluded. There is no
definition-count, arity or integer-width limit. Native compiler flags and normal
factory executions are compared at counts 0,1,2,3,17,65,129 and across primitive
and container arguments. Supported for consumers use the continuation transfer below. Other suspension
forms and protocols retain named UNKNOWN refusals.

### Unstarted source generator state

Calling a source generator binds its arguments immediately and allocates an
unstarted generator value. It does not run assertions, calls, handlers or yields
in the body. Normal definition-time defaults and call-time argument effects keep
their existing order, and bad signatures route TypeError at the call. The heap
object retains the defining function and bound parameters. Closure metadata stays
with the function. Mutable argument references stay live; scalar arguments retain
the existing sound interval summaries. GC, recency demotion, joins and inclusion
map these retained values like other heap fields. Creation pins them while an
allocation rewrites older references. No input size, arity or integer-width cap
is introduced.

Supported for consumers can resume this value using the transfer below. Other
consumers, element summaries, attribute reads/writes and method calls explicitly
yield UNKNOWN. They must not pretend a real generator is noniterable or a method
is missing, since that could hide an assertion in its deferred body. Type inspection also
keeps its existing unsupported-kind refusal. Mutable captured cells whose owner
frame has expired retain the existing closure refusal. This is the creation
state prerequisite, not a continuation implementation or an exhausted-prefix
proof. Resource exhaustion remains UNKNOWN.

Tests compare native unstarted frame bindings, retained aliases and default,
variadic and keyword-only parameters. A local function remains reachable after
its factory returns. Definition counts, parameter arities and argument lengths
through 129, builtin and user-object kinds, tagged module/class/function values,
nonfinite floats and wide integer bounds have native checks. Protocol tests keep
unsafe generator-body assertions unprovable through consuming builtins and source
methods returning generator objects.

### Source generator continuations in for loops

A for consumer can resume one known recent source generator, retaining a separate
local environment at each continuation position. Each yield evaluates in the
producer frame, runs the caller target assignment and body in the caller frame,
and resumes afterward. Mutable arguments and yielded objects remain live aliases.
Scalar locals use absolute interval summaries across suspension; this weakens
relations without narrowing the native values. A separate pin tracks the active
generator itself. If a nested allocation moves it to summary storage, resumption
refuses before changing a different object at the reused address. Returns and exceptions remove the
producer frame and release consumer pins. Escaping StopIteration becomes
RuntimeError, with an opaque exception value; consumer exceptions retain their
original class. Exact exception payloads and context inspection remain unknown.

The continuation graph covers statement yields, assignment from yield with the
implicit None sent by iteration, conditionals, returns, break/continue, while loops and
range-backed for loops, including else clauses and arbitrary signed range steps.
An outer range is created once. Each inner range is created where the source
reaches it, after intervening consumer mutations. Concrete trips retain exact
execution. If a continuation may either yield or stop, the existing loop solver
checks an inductive head covering every continuing iteration, then checks its
normal and break exits. Unfinished closure or spent work yields UNKNOWN for the
whole proof; an enumerated prefix cannot establish a verdict.

Other iterator cursors, yield from, suspensions inside handlers/finalizers or
complex expressions, summary/reentrant generator identities and expired mutable
closure owners keep named UNKNOWN refusals. Other consumers and direct next,
send, throw, close or introspection protocols retain their creation-stage
refusals. Existing class-attribute-shadow and string-to-float value precision can
also leave assertions unproved. No source-yield-count, trip-count, arity, numeric
width or input-size cap is introduced.

Tests compare native control flow and complete exhaustion, partial reuse, method
and __iter__ generators, consumer mutation timing, producer/caller exceptions,
late assertions and infinite-generator budget exhaustion. Native range lengths
0,1,2,3,17,65,129, signed 129-bit steps and observable 125-bit source growth are
checked. Unknown finite lengths use checked fixpoints, with native unsafe cases
kept unprovable. Coverage-guided input generation, symbolic execution and the
shared obligation table keep their existing native replay obligations.
