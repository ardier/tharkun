# Loop-Aware Static Resolution

Tharkun's static value resolver historically scanned only **top-level** function
statements for prior assignments (`condition_solver._resolve_scope_variables`).
A variable mutated inside a `for`/`while` body was invisible: the resolver saw
only its pre-loop init, so a guard on a loop-carried variable could not be
reasoned about statically.

This module adds **loop-aware reaching definitions**: it walks loop bodies,
collects the facts that hold at each loop *exit*, and propagates the value a
variable carries **past** the loop to any post-loop point (e.g. a `return`).

## Motivating case

```python
def main(data):
    q = 0
    while True:                 # state machine
        ...
        if c3:
            if not P1 != 0:
                assert not (...)   # bw_assume guard
                q = 2
                break              # every break sets q = 2
        ...
    return (q, b)               # <- what is q here?
```

`q` is `2` at the `return` on every path (the loop only exits via `break`, and
every `break` sets `q = 2`). The old resolver answered `q == 0` (the init). The
loop-aware resolver answers `q == 2` — the value carried *past* the loop.

## The three collection rules

`_collect_break_exits` walks a loop body depth-first, threading three accumulators
and snapshotting them at every `break`:

1. **Assert-sensitive.** `assert <cond>` statements on the path are collected
   into the exit's `asserts` list (the `bw_assume` guards that constrain it).
2. **Backward-from-break, conservative.** For each `break`, the enclosing
   `if`-guard chain (`conditions`) and the latest *constant* assignment to each
   variable (`assignments`) are recorded. Non-constant RHSs are dropped, never
   guessed; nested loops own their own breaks and are not descended; `continue`
   is ignored (it loops back, it is not an exit).
3. **Exit kind.** `_single_loop_exits` adds, per loop:
   - **break exit** → the latest guard fact (the `if`-chain that selects the break).
   - **natural exit** (`while <cond>` falls through) → the **negated** loop
     condition `not (<cond>)`. (`while True` never falls through, so it produces
     no natural exit — every exit is a break.)

Each exit is a dict: `{kind, conditions, asserts, assignments}`.

## In-order reaching definition

`_resolve_scope_variables` now walks function-level statements **in control-flow
order**, so the latest reaching definition wins — and loops participate:

- a plain assignment sets the value (constant) or clears it (non-constant);
- a loop where **every** break-exit assigns a variable the **same** constant and
  there is **no** natural fall-through carries that constant *past* the loop,
  **overriding** the pre-loop init;
- any other loop mutation (non-uniform break values, a natural fall-through, a
  non-constant assignment) marks the variable **unknown** after the loop — never
  guessed.

This is what makes `q` resolve to `2` *at the return*, not at the break.

## Soundness

The analysis only ever *narrows*; it never invents a value:

- only syntactically-present guards/asserts and **constant** assignments are recorded;
- a natural fall-through or a non-uniform break set ⇒ the variable is omitted (unknown);
- a guard value no break produces (e.g. `q == 5` when breaks only set `2`) is not
  spuriously satisfied — the resolver simply does not claim a value it cannot prove.

## Public API

```python
from tharkun import collect_loop_exit_facts, loop_exit_factenv

# Raw per-exit facts (one dict per break + natural exit):
collect_loop_exit_facts("main.py", lineno)
# -> [{'kind': 'break',   'conditions': ['q == 1', 'c3', 'not P1 != 0'],
#      'asserts': ['not (...)'], 'assignments': {'q': 2}}, ...]

# Sound FactEnv of loop-carried variable value-sets at a point:
loop_exit_factenv("main.py", lineno)
# -> FactEnv(facts={'q': VarFacts(eq={2})})        # while-True state machine
#    FactEnv(facts={'q': VarFacts(eq={2, 3})})     # multi-terminal machine
#    FactEnv(facts={})                             # natural fall-through -> unknown
```

`loop_exit_factenv` is the **bridge** into the rest of the solver: feed it back
through `solve_condition(..., facts=...)` (constraint propagation), or consume the
`eq` value-set directly. For a `while True` machine it is a single value; for a
multi-terminal machine it is the (sound) set of terminal values — a value-set,
not a single constant.

`solve_condition` itself is now loop-aware: a guard on a loop-local resolves
through `_resolve_scope_variables` automatically.

## Why the backward inverter already worked, and the forward did not

Tharkun's static resolver excludes **parameters** from assignment-resolution and
lets the guard bind them directly. In a loop-to-recursion *inverter*, the loop
state (`q`) is lifted into a recursion **parameter**, so `q == 2` resolved as a
parameter constraint. In the **forward** program `q` is a loop-mutated local, so
nothing resolved it — until this module. The two paths now agree.

## Future techniques (not yet implemented)

- **Natural-exit value narrowing.** At a natural fall-through, the negated loop
  condition plus monotonic updates often pins the carried value (`while i < n:
  i += 1` ⇒ `i == n`). Today the natural exit contributes a *condition* but not a
  resolved value, so any variable live across it is left unknown.
- **For-loop natural exit.** `for x in it` exhaustion is recorded as a `natural`
  exit kind only for `while`; `for` natural exits are conservatively treated as
  "unknown after".
- **AugAssign / compound RHS tracking.** `x += 1` currently marks `x` unknown;
  interval/monotonicity tracking would recover bounds.
- **Multi-loop / loop-nest summaries.** Each loop is summarized independently;
  facts are not yet propagated across a nest beyond the conservative "unknown".
