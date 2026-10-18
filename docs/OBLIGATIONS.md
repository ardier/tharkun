# Witness-format obligations ([internal ref], Slice 0)

`solve.census.obligation_census(source, classes, filename)` returns an untruncated
inventory, separately for each exception class. Each obligation carries an
`Invariant`: `location_invariant`, original statement location, a Python safety
expression, and `format: python_expression`. Facts use precisely the same
`Invariant` carrier. `solve.obligations.witness` groups these entries into a
format-2.3 Python `invariant_set`, without a `data_model`;
`sv.witness.dumps/loads` transports it as safe YAML. The reader and independent
checker also retain the internal 2.0 shape with its exact CPython data model.
Released WitnessLint checks schema, expression syntax and coarse locations;
its single-program hash fallback does not authenticate source identity.
Tharkun independently binds original bytes and checks invariant semantics.

Identity is `(SHA256(source UTF-8), original AST field/index path, exception)`.
The public location uses one-based Unicode character columns, a qualified
function name, and the statement pre-state. The separate AST identity retains
operations sharing that statement. Records repeated for the same AST node and
class collapse; different classes do not. The existing display census remains
limited to eight entries. Read source without newline normalization when exact
file-byte identity is needed.

Coverage refusal (unsupported exception class, import, reflection, syntax) is
an explicit `$coverage` obligation and a `coverage_gaps` diagnostic. An empty
refused scan cannot become a no-site proof. Unsupported safety rules have a
`False` placeholder plus a diagnostic gap. This is an open adapter gap, not an
assertion that unreachability is required or has been proved. Basic divisions
and sequence bounds get sufficient builtin-type-guarded conditions. Dictionary
membership and assert expressions still require source-checked purity/type
facts; no fact is guessed from a producer's display text.

`parse_predicate` checks syntax only. `check_predicate` additionally requires
source-checked definitely bound names, exact builtin types, and builtin name
identities. Only Boolean expressions, primitive constants/comparisons,
same-type integer/float `+ - *`, unary signs, exact type tests, `isinstance`,
`len`, and primitive-element membership are supported. Exact-type guards refine
conjunctions; `isinstance(x, int)` does **not** establish an exact int, because
bools and user subclasses also pass it. An unknown object is refused for
`isinstance`: its `__class__` descriptor can run user code. A bare dict/list type does not establish
pure membership: its elements can have user hooks. No eval, arbitrary calls,
attributes, subscripts, division, assignment expressions or implicit truth
conversions are accepted. Integer arithmetic is unbounded; float operations
retain Python binary64 semantics. A later Z3 adapter must use FP, not Reals.

Parsing and serialization certify neither predicates nor program safety.
Metadata, original locations, totality, truth on every reachable state and
coverage must be independently checked before a fact discharges an obligation.
There is no new TRUE route in this slice. Per the owner's correction, there is
no SMT-LIB layer: the later checker lowers the checked subset directly to Z3's
Python API. Engine domains remain native.

Checks: `python -m pytest tests/test_obligations.py` covers YAML round trips,
duplicate keys and unsafe constructors, AST identities, Unicode locations,
coverage holes, untruncated inventories, shadows and impure/partial predicates.

The shared resolver scans definition-time expressions even for unused functions.
Nonliteral defaults currently produce a coverage refusal: their enclosing/class
scope needs a dedicated model. Literal defaults remain supported, and truly dead
function bodies still do not create sites. This also closes the same omission
for the determinism census.

## Slice 1: independently checked TRUEs

Every proposed TRUE goes through the same finalizer, including
deterministic/finite routes: it always runs and always reports; owner ruling
(2026-09-28, no feature flags) is that a TRUE is never downgraded to UNKNOWN by
a runtime switch. The CLI writes an independently confirmed `witness.yml`
and the diagnostic `obligations.yml` into a fresh per-run directory under
`--proof-dir` (default `proofs`); concurrent tasks never share a file.
`proof_checked=False` explicitly identifies an unchecked candidate: it retains
in-memory candidates and open-obligation diagnostics, without `witness.yml`. Writing the witness (or failing to, e.g. a read-only proof
directory) never changes the verdict either; a write failure is only noted
(`tharkun-sv: proof: witness not written: <reason>`) with the checked/unchecked
verdict kept as-is. A genuine source change between the search and the
finalizer's read-back check is the one thing that still yields UNKNOWN --
whatever was proved may no longer describe the file on disk.

The finalizer -- reusing the obligation census already computed by `verify()`,
the independent checker, the `python -I` interpreter probe, and writing the
YAML files -- shares one wall-clock budget (`--proof-dir`'s companion budget
is `verify(checker_seconds=...)`, default 3 s). Once that budget is spent,
remaining work is skipped and reported as `proof: unchecked (budget)`, never
as a verdict change. The old strict-mode downgrade (an unchecked TRUE becomes
UNKNOWN, with its open obligations in the reason) still exists as
`tharkun.sv.proof._downgrade_if_unchecked`, an internal function tests call
directly to exercise that semantics -- it is not reachable from `verify()`,
the CLI, or any parameter.

The Python API returns the artifacts in `Verdict.detail`; passing `witness_dir`
writes the confirmed correctness witness and open-obligation diagnostics.
Every emitted FALSE requires fresh validation under the validator's runtime
conditions within the remaining task budget; its JSON witness format is unchanged. Program
source is read the way CPython reads it (PEP 263 encoding-cookie detection,
defaulting to UTF-8; exact bytes otherwise, no newline normalization) rather
than assumed to be UTF-8 -- every stage that reads or parses the task's source
file (absint, `prove_static`'s census, determinism, slicing, concolic
instrumentation, the finalizer, `--validate-witness`) goes through
`tharkun.sv.engine.read_source` ([internal ref]: a Latin-1-cookie source with a UTF-8-
byte-sequence dict key decoded differently under a stage's own strict-UTF-8
read than under CPython's real execution, proving a `KeyError` impossible that
CPython actually raises). `input_file_hashes` binds the file's raw bytes
(`tharkun.sv.engine.read_source_bytes`), not a UTF-8 re-encoding of the decoded
text -- those differ exactly when the encoding cookie isn't UTF-8.

Check a YAML witness independently with:

```
python -m tharkun.sv task.py --property property.prp --validate-witness witness.yml
```

The checker binds the exact source text as CPython would decode it (without
newline normalization) and the raw file bytes for `input_file_hashes`, the
property text, source statement coordinates and the CPython 3.14 binary64
profile. It reconstructs coverage and checks every supplied assertion and
source operation; omitted sites and tautological witnesses do not excuse an
unsafe program. Census/resolver models are an explicit shared trust boundary.
The checker never calls absint or accepts its status, prose or type premises.
Absint exports candidate exact scalar types and intervals at obligation sites;
these must follow independently from the source before they are accepted.

The initial interpreter supports acyclic scalar assignments and branches,
zero-argument nonrecursive functions, direct `_sv_verifier` scalar draws,
assertions, fixed primitive lists/tuples, and a narrow builtin set. Unsupported
syntax, effects and source models leave the proof open. Integers are unbounded;
float operations use Z3 IEEE binary64 terms, not Reals. Mixed float/int
comparisons use the exact finite value, with explicit NaN/infinity cases.
Integer true division deliberately overapproximates its result by any float;
this loses precision but cannot invent a successful proof. Floor division and
modulo include Python's negative-divisor convention. Solver failures and budget
exhaustion never establish a fact. Each check has a shared deadline.

Zero-argument callees start their local bindings unbound, independently of module
values. The whole callee is checked for unsupported declarations/effects and
builtin/imported/callable shadowing, including declarations skipped by a bare
return.

`tests/test_witness_validation.py` includes stale hashes/properties, forged
intervals, omitted sites, negative indices, NaNs, signed zero, underflow,
large integer/float comparisons, the internal unchecked-downgrade helper
(`tharkun.sv.proof._downgrade_if_unchecked`, kept only for tests -- see above)
and failed/concurrent export.

Float membership is refused by the source checker until object identity is
modeled: Python uses identity-or-equality, so a NaN object belongs to a list
containing that very object even though IEEE equality is false.

The initial checker keeps strings as concrete Python values and compares them
exactly. Z3 StringVal parses backslash-u-brace escapes and must not encode Python
string literals. Symbolic string inputs remain unsupported.

Scalar source expressions also admit primitive conditional expressions with
exact int results on both arms, checked under their separate branch conditions;
calls inside a conditional expression remain refused. This does not extend the
witness predicate grammar. The two-int/int helper subset additionally admits
one local result initialization, one ternary update (optionally under one plain
`if` without `else`), and return of that same local. The source checker re-proves
the assignments, actual caller sorts and every original statement location.
The metricsdata limit twins remain separate controls: both are type-safe, while
the zero-limit buggy twin still fails its native assertion-safety contract.

An additional pure scalar-helper body subset admits at most six positional
arguments, two scalar locals, 24 statements, two nested `if` levels and two
helper levels. Actual int/bool arguments establish unannotated parameter sorts.
Helpers returning int/bool also admit a bare `object` parameter annotation as a
nonbinding marker: every actual argument is still independently checked as exact
int/bool, and primitive parameter annotations retain exact sort matching. This
creates no object domain or free object symbols; uncalled unannotated or
object-annotated helper facts remain unchecked. Local stores start unbound,
globals come only from authenticated module literals, and every original
statement and return path is checked. Feasible None fallthrough, disagreeing
return sorts, recursive/rebound calls, declarations and effects remain refused,
including after a return. This does not expand operand-value BoolOp or predicate syntax.

Only the object-marked scalar-return bounded-helper route additionally admits
genuine unrebound `int(x)` and `max(a, b)` calls with the existing tested arities.
Whole-source identity and live contract admission are checked even for dead or
unused calls; authenticated source user helpers retain priority. Actual operands
are independently evaluated once, left to right, and must be exact ints. The
existing runtime derives int identity and ordered max directly. This does not
admit other builtin calls or expand ordinary primitive scalar/tuple helpers,
predicate syntax, helper limits, object values or annotation-based type facts.

Source `if` conditions additionally admit exact int/bool `and`, `or` and `not`.
Integer truth is nonzero, including negative and non-0/1 values. Each reached
operand is evaluated once under the preceding operands' short-circuit path;
a proved unreachable operand is skipped without evaluating its helper or draw.
Unknown reachability or exhausted checker budgets refuse. This is a condition
truth rule only: operand-value expressions retain Python's selected value and
their existing kind restrictions. Assignment, return, comparison, while,
assertion, ternary and witness-predicate boundaries are unchanged. Other truth
kinds and user truth hooks remain unsupported.

Direct source `if NAME is None` and `if NAME is not None` guards additionally
evaluate to false and true when the current, independently bound NAME has an
exact int or bool sort. The Name is evaluated once in its actual frame. None
is only the literal singleton in this guard syntax, never a modeled value or
sort. Equality, reversed/chained comparisons, nested `not`/BoolOp tests and
VALUE/assert/while/ternary/predicate uses stay refused; None-valued paths and
wrong actual helper sorts are not inferred away from annotations. The existing
scalar leaf shape admits this guard in its full body audit, including unused
helpers; generic helper limits and call rules remain unchanged. Original
statement prestates still apply. A vacuous invariant at a dead branch proves
no reachability claim, and a false branch-taken flag at a reachable join refuses.
The features num-zero task remains unsafe for AssertionError even when its
TypeError/AttributeError witness is independently checked.

Direct body statements of the authenticated plain zero-argument `main` may use
`NAME: object = RHS`, with one simple local Name target. The source checker
evaluates RHS once and keeps its independently established exact int/bool kind;
the local annotation is not evaluated and supplies no type premise. Whole-body
local masking still applies before the assignment, including after bare returns.
Other annotation forms, placements, targets and actual kinds remain unsupported.
This adds neither an object domain nor object-parameter or pure-builtin helper
admission; those boundaries remain separate.

A separate tuple wrapper admits exactly two int parameters, up to three
straight-line scalar assignments and a final pair of bool expressions. Its
result must be immediately unpacked into two fresh, distinct names. Arguments
and result elements are checked left to right once, with original statement
prestates and independently checked pure scalar callees. Stored, indexed,
aliased, nested or starred tuples, existing targets, other element kinds,
effects and unsupported bodies remain refused, including on dead paths. The
ordinary scalar helper limits and witness predicate grammar are unchanged.

The scalar source checker also covers complete `for` loops over the genuine
one-argument builtin `range`, with exact literal or independently authenticated
immutable module-int lengths 1..4. It checks every iteration, body and exhausted
head at original source locations; repeated draws stay independent, and a bare
callee return ends that path. The target must be one new scalar name. Rebound
`range`, competing target/constant bindings, symbolic or larger bounds, `else`,
nested loops, heap writes and other control effects remain refused. This is
complete finite iteration, without sampling or truncation. Witness predicates
and the restriction of `loop_invariant` to original while heads are unchanged.

## Slice 2: scalar while induction

The absint fixpoint now exports candidate `loop_invariant` entries at original
while heads. The independent checker proves the base from real entry states,
havocs every scalar written by the body, checks all body/guard operations under
the joint invariant, proves preservation on every modeled back edge, and uses
the invariant plus the negated guard at the exit. Source assignments determine
the frame; no producer-supplied frame premise is trusted. Integer/float/bool
loop-carried types must be preserved. A loop need not terminate.

This first adapter declines heap writes, for-loop iterator models,
break/continue/return and exception edges. Nested scalar while loops can use
the same induction rule. Missing or mutated invariants leave obligations open.
Stationary replay certificates are still unsupported by the YAML checker.
`tests/test_loop_witness.py` covers reachable division safety, unbounded input
trip counts, initiation/step mutations, a violating source mutant and refused
heap/control effects.

Census safety predicates are sufficient conditions, not necessary ones. A
placeholder for an unsupported adapter is omitted from the candidate facts;
it remains open unless the independent source checker covers that operation.
For example, checked builtin-list semantics exclude KeyError, and a loop's
interval can establish that the expression `i + 1` is a nonzero divisor.
Coverage gaps are never omitted or discharged by this rule.
