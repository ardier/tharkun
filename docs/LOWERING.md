# Exact lowerings ([internal ref])

Some analyses do not model a construct and abort on it. `solve/absint` aborts on every
comprehension, so a program with one gets no proof, however easy the rest is. A **lowering** rewrites
the program into one the analysis can read. This page is the contract of that rewrite: what it
promises, what it refuses, how its findings are put back in terms of the original program, and what is
measured.

The one rule: **a lowering is the same program.** The same value, the same exception (type and
message), the same ORDER of every side effect, for every input. A wrong TRUE costs 32 points, so a
rewrite that is only "the same on the cases we tried" is not allowed. Where an exact rewrite is not
possible the construct is **refused**, left exactly where it was, and the analysis aborts on it as it
did before.

Not the same thing as Eregion's `lower_comprehensions` pass
(`repos/eregion-forward-passes/src/eregion_forward_passes/passes/forward/lower_comprehensions.py`).
That pass is inversion-shaped: it rewrites into the form extraction recognizes, and it is not
semantics-preserving. Five deviations are measured and recorded on [internal ref] and in the Eregion
design doc 06 ("Ruled, not built"): the loop variable leaks, a non-sequence iterable is indexed, the
iterable is re-evaluated on every step, a nested comprehension that reuses an outer name gets the wrong
bound, and `any`/`all`/`next` over a generator lose laziness. This module is the semantics-preserving
one, and the five programs are tests here (`tests/test_lowering_comprehension.py`, `test_1_...` to
`test_5_...`). Migrating Eregion onto it is a follow-up note on [internal ref], not built.

## Files and how they relate

- `src/tharkun/solve/lowering/__init__.py`: the package surface (`lower_comprehensions`, `Lowered`,
  `Refusal`, provenance helpers). Stdlib only: nothing in the package imports the rest of tharkun, so a
  consumer inside or outside tharkun imports this package and never a copy.
- `src/tharkun/solve/lowering/comprehension.py`: `lower_comprehensions(tree) -> Lowered`. Imports
  `provenance.py` and `scopes.py`.
- `src/tharkun/solve/lowering/generator.py`: immediately consumed all/any, list/tuple,
  exact integer sum and pattern-unpack loops. Generator steps wrap StopIteration;
  consumer truth tests remain outside those handlers. Unknown inner iterator protocols
  and declared finalizers are refused.
- `src/tharkun/solve/lowering/eval_source.py`: `lower_module_code(tree) -> Lowered` expands finite
  literal-source module `eval` expressions and assignment-only `exec` statements for proof analysis; `solve/static_code.py` checks registry authority
  and supplies the frontend work budget. Neither executes evaluated code.
- `src/tharkun/solve/lowering/scopes.py`: `Names` (which names a program binds, which builtins are a
  stable read, fresh names), `rename_names` (scope-aware renaming), `reaches_the_namespace` (the
  reflection rule below), `target_names`. Imports only the standard library.
- `src/tharkun/solve/lowering/provenance.py`: `annotate`, `stand_for`, `origin_of`, `resolve`: where
  every node of a lowered program came from. Imports only the standard library.
- `src/tharkun/solve/lowering/walrus.py`: `lower_walrus_loops(Lowered)` extends the analysis tree
  in place for a sole or final `and`-guard assignment expression in a while without else.
  Assignment and its truth test stay inside each iteration; existing nodes keep their origins,
  and no names or scopes are introduced. Only function-local targets without namespace
  reflection, global/nonlocal declarations or nested-scope mentions are rewritten: a class
  namespace or a rebinding finalizer can make a target read differ from the assigned value.
  Programs declaring finalizers are also refused: the original assignment holds an extra
  RHS reference during the store. Other assignment-expression forms stay unchanged.
  `absint.prove` calls it only when the source contains `:=`.
- `src/tharkun/solve/absint/__init__.py::prove` calls `lower_comprehensions` on the sliced program and
  hands the result to `src/tharkun/solve/absint/interp.py::Analyzer(lowered=...)`, which interprets the
  lowered tree and reports on the original one. `interp.py` imports `provenance.py` for the
  node-origin attributes.
- `tests/test_lowering_comprehension.py` (differential execution: 82 hand-written cases and 40
  seeded random differentials), `tests/test_absint_lowering.py` (the seam), `tests/test_absint_lowering_fuzz.py` (a
  soundness fuzz of absint on lowered programs).
- `tests/test_lowering_walrus.py`: disposable-child native comparisons of loop guard order,
  assignment visibility, exceptions, break/continue, arbitrary guard arity and scalar/container
  truth protocols; source-origin and UNKNOWN controls for forms that are left intact.

## Immediately consumed generator expressions

Stable builtin `all` and `any` consume each produced value once and stop every
nested loop on the first decisive truth value. They return their empty-input
identity without evaluating an element. The expression is never replaced by an
eager list before these consumers. `list` and `tuple` drain fully. Generator
assignment to one tuple/list pattern stops after at most the pattern's arity
plus one elements; a starred pattern drains fully. The original assignment
consumes a buffered iterator, preserving CPython's generic-iterator unpack
messages, target-store order and native too-few/too-many errors. Arity is taken
from the source pattern, with no fixed limit.

Each generator element, filter and destructuring target has its own
StopIteration-to-RuntimeError boundary, including the original exception as
cause. The consumer's truth test runs outside that boundary. Initial outer
iterator creation also stays outside. Inner iterables must be known builtin
containers or constructors, whose GET_ITER cannot invoke a custom hook; unknown
inner protocols remain opaque. The original loop variables are renamed with the
existing comprehension scope/provenance machinery. Rebound consumer/protocol
names, assignment/suspension in the generator and declared finalizers refuse the
new lowering. Implicit-frame super() and escaped super bindings also stay opaque.
Potentially unbound captured locals also stay opaque: removing a
generator frame must not turn NameError into UnboundLocalError. Known frame and
reference-count observations, including imported aliases, are refused. Existing
unknown runtime/world-effect refusals remain.

`sum` keeps the original CPython builtin, rather than replacing its numeric
algorithm. Only effect-free integer/bool terms over builtin ranges and a literal
integer/bool start are materialized. Floating terms, custom addition, dynamic
start evaluation and unknown element types remain opaque, since moving them can
change compensation or the first error/effect. Lengths and integer magnitudes
are unrestricted; analysis work budgets still expire to UNKNOWN.

Absint recognizes only lowering-owned, structurally checked StopIteration
handlers when permitting primitive power, bool and isinstance through those
boundaries. All active frames must be such handlers, with no finally/handler
execution. Other user handlers retain complete exception-routing requirements;
unknown operands and callbacks gain no exemption. Original AST paths and source
hashes continue to identify checks. Native boundary, arity, magnitude and unsafe
twins are in `tests/test_lowering_generator.py`.

## Finite module code sources

Before slicing, `absint.prove` can expand direct, one-argument builtin `eval` and bare `exec` statements in module execution
scope. Its argument is an exact str/bytes literal or a name whose every syntactic binding resolves
through literal assignments, aliases and conditional expressions to such sources. There is no bound
on source length or alternative count: exhausting the frontend CPU budget leaves the call opaque.
Every alternative must compile without warnings in its own mode. An exec body contains only assignments
to names and pass statements, with no bound on statement count; an empty body is also supported.
Assignments that bind any dynamic-source provider, an alias it depends on, or eval/exec/compile stay opaque:
they could invalidate the original-source binding census. Value-used exec, other statements and assignment
targets stay opaque because this pass supplies no expression-result or namespace-effect interface.
Future imports, rebound
callees, namespace-writing expressions, comprehensions, function/class bodies and custom namespaces
remain opaque. Existing environment and reflection guards inspect the expanded body before any proof.

Dispatch uses exact source-value comparisons and preserves the initial source-name read, including
its NameError. Expression evaluation preserves effect order and exception payloads. Inserted nodes
report at the original code call, while a code-unit digest and inner AST path distinguish allocation
and call contexts. Comprehension lowering preserves those origins. Original source hashes and witness
locations remain unchanged. Existing value precision limits, including object identity and shifts,
remain UNKNOWN; this pass does not add arithmetic or equality rules.

This preparation is proof-only. Native search and replay still execute the original source with the
existing callee/source observers. Independent witness validation retains its separate syntax boundary;
a producer TRUE is not a checked witness merely because eval was expanded. Native differential and
negative controls live in `tests/test_absint_eval_source.py` and `tests/test_absint_exec_source.py`.

### Closed compile-to-exec snapshots

`compiled_source.py::lower_compile_snapshots` handles default three-positional-argument
`compile(source, literal_filename, 'exec')` stored in module names and used only by bare module
exec or plain aliases. Each binding must dominate every read; bindings are unconditional module
assignments from an eligible compile or another code name. Other modes/options, ambiguous scopes,
metadata reads, code-object escapes and uncertain bindings stay opaque.

The first module-code pass exposes ordinary literal-code accesses before this flow is checked.
Any remaining dynamic-code operation outside the closed compile/exec flow prevents snapshot
conversion: its source could observe or overwrite the original code names. Compiled bodies cannot
refer to those names or perform nested dynamic-code operations. These checks precede alpha-renaming.
The transformation captures the source value at compile time in collision-free private names,
then the existing module-code pass expands the saved source at exec time. Source rebinding and
code aliases therefore preserve their own snapshots; the pass never re-reads the current source.

Every alternative must validate without compiler warnings or errors before this transfer is used;
failed validation leaves the original compilation opaque, so an error is never moved to exec.
Private selectors are `Lowered.synthetic`, including after comprehension composition: they must
not export false str facts about original CodeType values. All original source positions and runtime
observers remain intact. Native snapshot, namespace-write, exception-payload, size and false controls
are in `tests/test_absint_compile_snapshots.py`. The frozen compile audit sample uses eval mode and
remains refused; this closed flow supplies no general CodeType model or checked-witness extension.

## What `lower_comprehensions` does

`lower_comprehensions(tree)` returns a **copy** of the module with list, set and dict comprehensions
turned into loops. The input is not modified. The result (`Lowered`) says how many comprehensions it
lowered, which it refused and why (`Refusal(line, col, kind, reason)`), which fresh names stand for which
original names, and which stand for none.

Each rule below is a test in `tests/test_lowering_comprehension.py`; the differential runs the original
and the lowered text with `exec` and compares a full trace (result or exception type and message, and
the log of every side effect, in order).

### Scope

A comprehension's loop variables live in its own scope: they do not leak and they do not clobber an
enclosing name. Each loop variable gets a **fresh name** (`_lw<n>_<hint>`, never colliding with any
identifier of the program, single underscore so a class body does not mangle it), and its uses are
renamed with a scope-aware renamer (`scopes.py::rename_names`): a nested comprehension or a lambda that
rebinds the name keeps its own, and the FIRST iterable of a nested comprehension is evaluated in the
enclosing scope. A comprehension in a class body is left alone: class scope resolves names differently.

### The first iterable is evaluated once, before the loop

A `for` statement evaluates its iterable once, in the enclosing scope. The later iterables of a
comprehension are evaluated per iteration of the loop above them, where the loop variables are visible
(renamed).

### Evaluation order

A comprehension is an expression; its loops are statements. Hoisting the loops in front of the
statement is exact only if everything that would have been evaluated BEFORE the comprehension still is.
So every earlier operand (and a call's callee, as the bound method it would have loaded) that is not
provably stable is first frozen into a temporary, in order. A name is a **stable read** when it names a
builtin that nothing in the program binds. A comprehension that is only conditionally evaluated (the
right of `and`/`or`, a branch of a conditional expression) becomes an `if` statement so it runs exactly
when it would have. A dict comprehension evaluates the key, then the value, then inserts: the lowering
keeps that order with temporaries (`acc[k] = v` would evaluate `v` first).

### Dict displays with `**` entries

CPython builds a display's dict first and applies each `**` operand the moment it is evaluated
(`BUILD_MAP 0`, then `DICT_UPDATE` per operand), so unpacking an earlier operand (a `TypeError` for a
non-mapping, a user mapping's `keys`/`__getitem__`) happens BEFORE a later comprehension is evaluated.
`comprehension.py::Lowering.dict_unpacking` admits a display with exactly one comprehension-bearing entry,
that entry a `**` entry following only `**` entries: the entries before it become a statement of their own,
`t = {**A, **B}` (the same unpacking in the same order, nothing else), the comprehension's statements
follow, and the display becomes `{**t, **<result>, <entries after>}`; `t` is a fresh builtin dict, so
unpacking it again has no effect, the same contents and the same order. A `key: value` entry before the
comprehension is refused (when a pair's key is hashed relative to the operands depends on how CPython chunks
the display) and so is a second comprehension-bearing entry. `tests/test_lowering_comprehension.py` runs
native differentials with a logging user mapping, a non-mapping operand and a raising comprehension.
Measured use: `type-eval-py/dicts_merge_unsafe` (sv-benchmarks f0624a58) dynamic-typing goes from the
`DictComp` refusal to TRUE; its assertion-safety pair stays UNKNOWN (the equal-values assertions need a
relation between `merged_dict[k]` and `dict2[k]` that the interval domain does not have).

### Generator expressions are lazy

Immediate all/any now use stopping loops, and pattern unpack uses its exact
consumption limit, as described above. list/tuple materialize with per-step
exception boundaries. The integer-only sum path retains the native builtin;
floating/custom arithmetic and dynamic starts remain refused. Stored generators,
next/min/max/zip/dict/set consumers and unknown inner iterator protocols stay
opaque. Existing sorted and literal join eager paths are separate.

### Refused shapes

The comprehension stays where it is, and the analysis that does not model it aborts as before. Refused:
`*args` and `**kwargs` around a comprehension; a chained comparison with a comprehension after the first
operator; f-strings; assignment expressions (inside a comprehension, or around one); `await`, `yield`;
an async comprehension; a comprehension target that is not a plain name pattern; a comprehension in a
loop test, a `with` item, a default value, a decorator, an augmented or annotated assignment, a lambda
or a class body; a set comprehension when `set` is rebound; and every generator expression that is not
the sole argument of a draining consumer. Each is a parametrized case in the tests, and each checks that
the program text is unchanged.

### A program that can reach its own namespace is left alone

A fresh name is observable if the program can read its scope, so a program that can is not lowered. The
rule (`scopes.py::reaches_the_namespace`) is exact about which loads can reach the builtin:

- a load of `locals`, `globals`, `vars`, `dir`, `eval`, `exec`, `compile`, `__import__`, `__builtins__`,
  `breakpoint` that can resolve to the **builtin**. A load inside a function that binds the name locally
  (a parameter, a loop variable) is a local read and is fine: hexiom's `for dir in DIRS` is that shape.
  A comprehension variable binds only inside the comprehension, so it does not exempt a load in the
  function around it. A module-level binding does not exempt a load either: a function can run before it
  executes. A `global` declaration of such a name is refused;
- a frame or namespace attribute: `f_locals`, `f_globals`, `f_back`, `_getframe`, `currentframe`,
  `__globals__`, `gi_frame`, `cr_frame`, `tb_frame`;
- `from x import *`, or the name `builtins` anywhere.

An attribute that happens to be called `vars` or `dir` (`o.vars`) is not a name and does not count.

## Provenance: findings are about the ORIGINAL program

Every artifact of a run (the obligation table, the site census, `witness.yml`, the records lanes A and B
read) names sites of the original program. A lowering that moved a node would make every finding
wrong. So:

- a node taken over from the original keeps its own `lineno`, `col_offset` and `end_*`, and carries the
  AST **path** it had in the original program (`_lw_path`);
- a node the lowering invents (the accumulator, a loop, a filter, a temporary) is placed at the position
  of the original node it implements, carries that node's path and a **role** (`_lw_role`:
  `acc`, `loop`, `temp`, `key`, `value`, ...). A consumer can tell "this operation is in the program"
  from "this operation implements a construct of the program";
- `origin_of(node)` returns the `Origin` (path, position, role) and `resolve(original_tree, path)` returns
  the node of the original program: a test checks that every node of a lowered program resolves.

Paths are the sequence of `(field, index)` steps from the module (`body[2].body[0].value.elt`). They are
stored on the node, so they survive `copy.copy` and `copy.deepcopy`.

## How absint uses it (`solve/absint`)

`prove` lowers the sliced program, builds the `Analyzer` on the lowered tree, and puts everything the
analyzer reports back in terms of the original program:

- a check on a lowered node is recorded at the line and column of the original node. `10 // x` inside a
  comprehension is reported at its own position;
- a fact, a residual predicate, a `scope` fact or a `types` entry that names a fresh loop variable is
  restored to the original name **only at a site inside that comprehension** (`Lowered.at_site`): at any
  other site the original name means a different variable, so the fact is dropped, not restored. A fact
  that names a synthetic name (accumulator, temporary) is dropped: it names nothing in the original
  program;
- no invariant is captured at a statement the lowering invented, and none names a variable of the
  lowering (`Analyzer.capture_invariant`);
- the loop variable of a lowered comprehension stays bound in the lowered program after the loop, but
  under its fresh name. Nothing about the original name changes; the test that keeps this honest is
  `test_the_loop_variable_does_not_leak_into_the_enclosing_scope` in `tests/test_absint_lowering.py`
  (leak the variable and the divisor becomes non-zero: a wrong TRUE).

A comprehension the lowering refuses stays in the tree and absint aborts on it: `Unsupported` and
`complete=False`, as before.

## Measured

Absint on the 24 SV pyperformance tasks (12 programs, each `_det` and `_nondet`), four properties
each, before and after the lowering (`prove` with the lowering switched off against on, 2026-10-05):

| task | comprehensions | lowered | what absint says before -> after |
| --- | --- | --- | --- |
| `bm_chaos` | 6 | 6 | stops earlier on an environment blocker (`unknown file read at line 141`): same |
| `bm_go` | 5 | 5 | `dict literal with a non-constant key (line 20)` before and after: same |
| `bm_hexiom` | 10 | 6 | `expression ListComp (line 479)` -> `str method splitlines (line 479)` |
| `bm_raytrace` | 1 | 1 | environment blocker (`unknown file read at line 213`): same |
| `bm_scimark` | 2 | 2 | `*args call (line 396)` before and after: same |
| `bm_spectral_norm` | 1 | 1 | `expression ListComp (line 31)` -> `call of an unknown value (line 41)` |
| `bm_fannkuch`, `bm_float`, `bm_nbody`, `bm_regex_v8`, `bm_richards`, `bm_unpack_sequence` | 0 | 0 | unchanged |

No pair changes verdict: every one is UNKNOWN before and after. Hexiom keeps 4 generator
expressions (not the sole argument of a draining consumer) and also stops at `str.splitlines`; chaos
and raytrace are blocked by an environment check before the analysis starts; go, scimark and the
others are blocked by other absint gaps before they reach a comprehension.

The honest reading: the lowering removes the comprehension abort and nothing else. It lets absint get
past `bm_hexiom`'s and `bm_spectral_norm`'s comprehensions to the NEXT unmodelled construct, and it
does not by itself turn any measured task from UNKNOWN to TRUE. Per pair CPU time is unchanged (0.0 to
0.5 s).

## Soundness evidence

- Differential execution of original against lowered: 82 hand-written cases (scope, order, laziness,
  every refused shape, provenance) plus a seeded random differential (40 seeds x 25 programs x 4 inputs)
  over expressions with effects and exceptions.
- A soundness fuzz of absint on lowered programs (`tests/test_absint_lowering_fuzz.py`): every program
  `prove` calls TRUE is run for real and must not raise the class. The committed test is 12 seeds x 25
  programs; run at scale on 2026-10-05 (10,000 random programs over `ZeroDivisionError`, `IndexError`,
  `KeyError`, `TypeError`: about 12,800 TRUE proofs) it found 0 wrong TRUE.
- Known and safe: CPython 3.12 and later inline comprehensions (PEP 709). In one exotic shape (a
  comprehension that BINDS a name, then a later comprehension nested in the same outer one that READS the
  same function local) it raises `UnboundLocalError` where the language reference, 3.10 and 3.11, and this
  lowering give the other behaviour. The lowering therefore has MORE behaviours there, never fewer, which
  is the safe direction for a proof. The random generator avoids that shape on purpose.

## Not built

- **`for` over a range and over an iterable for the search engine (`solve_site` in `sv/_symexec.py`)**:
  `for i in range(a, b[, step])` to a counter `while`, `for x in it` to `iter()`/`next()`, `while t` to
  `while bool(t)`. Stages S1 and S2 of the same module, next.
- **`max`/`min`/`any`/`all` over a generator** by an exact short-circuit loop.
- **Eregion migration** onto this lowering (a note on [internal ref]).

## Module copy identity facts

`copy_identity.py::annotate_module_copies` adds private must-identity metadata
to one-operation `is`/`is not` assertion comparisons of names. It keeps every
source node, operand read, source position and site key unchanged.
`compare_split` invokes it lazily once on the actual analysis tree when an
identity comparison has two name operands. It consumes the fact only in that
identity branch, independently of value equality. Programs without such an
operation do not run this pass.

Plain module name assignments preserve the copied object's token; chained
assignments share one token because the RHS is evaluated once. An effectful
RHS clears prior facts before its result is bound. Other effects, deletes,
loops, handlers and nested scopes clear facts and supply no new assertions.
Conditional branches start without assumptions about globals after their
truth conversion; only names bound on both branches survive, each with a new
token. No relation between equal-valued objects is inferred. This deliberately
loses precision across effects, control joins and other expression contexts.

These facts are consumed only by closed absint execution. Its existing hook
gates must exclude finalizers: overwriting a name can otherwise release an
object and invoke `__del__`, a weakref callback or generator cleanup that
rebinds globals. Dedicated native counterexamples retain UNKNOWN through
those existing protocol refusals.

Within that admitted model, the proof is independent of value kind, length,
magnitude and copy-chain size.
Underlying unsupported Python constructs still produce UNKNOWN before an
identity check can certify a program. The ordinary CPU deadline bounds the
pass; no size cutoff or special benchmark name appears in it.
