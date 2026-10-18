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
- `src/tharkun/solve/lowering/scopes.py`: `Names` (which names a program binds, which builtins are a
  stable read, fresh names), `rename_names` (scope-aware renaming), `reaches_the_namespace` (the
  reflection rule below), `target_names`. Imports only the standard library.
- `src/tharkun/solve/lowering/provenance.py`: `annotate`, `stand_for`, `origin_of`, `resolve`: where
  every node of a lowered program came from. Imports only the standard library.
- `src/tharkun/solve/absint/__init__.py::prove` calls `lower_comprehensions` on the sliced program and
  hands the result to `src/tharkun/solve/absint/interp.py::Analyzer(lowered=...)`, which interprets the
  lowered tree and reports on the original one. `interp.py` imports `provenance.py` for the
  node-origin attributes.
- `tests/test_lowering_comprehension.py` (differential execution: 82 hand-written cases and 40
  seeded random differentials), `tests/test_absint_lowering.py` (the seam), `tests/test_absint_lowering_fuzz.py` (a
  soundness fuzz of absint on lowered programs).

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

### Generator expressions are lazy

Lowering one to an eager loop changes WHEN its elements are evaluated relative to the code that consumes
them: `any`, `all`, `next`, `sum`, `min`, `max`, `zip`, `dict` and `set` interleave with it and can stop,
raise or have effects part-way. A generator expression is lowered only as the sole argument of a
consumer that drains it completely before doing anything else, which makes it identical to the list
comprehension: `list(g)`, `tuple(g)`, `sorted(g, ...)` (the builtins, not rebound anywhere) and
`"...".join(g)`. Every other generator expression is refused. `sum(genexp)` cannot be lowered exactly:
`sum` adds as it goes, so a float or an exception part-way differs. `max`, `min`, `any` and `all` could
be lowered by an exact loop that mimics their short circuit; that is not built.

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
