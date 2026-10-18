# Obligation slicing ([internal ref])

The single slicer is `solve.slicing.slice_for`. The old
`sv._symexec.slice_for` name imports it. `slice_obligation(source, entry,
filename=...)` adapts a witness-format `location_invariant`:

```python
entry = {"invariant": {
    "type": "location_invariant", "format": "python_expression",
    "location": {"file_name": "task.py", "line": 12, "column": 5,
                 "function": "check"},
    "value": "denominator != 0",
}, "exception": "ZeroDivisionError"}
result = slice_obligation(source, entry, filename="task.py")
```

The result contains source text, the ORIGINAL UTF-8 SHA256, removed statement
coordinates (one-based lines, zero-based AST byte columns), and a refusal
reason when applicable. Witness locations use one-based character columns.
The predicate is a dependency seed, never an assumption or certificate.
The source and criterion statement remain bound to the original coordinates.

## Soundness boundary

The criterion includes reaching the error site and the values in its safety
condition. All executable effects are roots: calls and their argument/global
dependencies, alias stores, potentially raising expressions, returns, raises,
handlers/finally, imports, loops and every break/continue. A fixed-point
closure keeps definitions needed by effects, the criterion, and retained
callees, including definitions textually later than their uses. Function and
control context around the anchor also survive.

Removal is limited to unread literal stores whose every binding is a literal,
and unreferenced module-level functions with inert definition headers. No
store that can overwrite an unknown object is removed: releasing the previous
value can execute a finalizer. User finalizers refuse pruning because allocation changes can affect GC scheduling.
Class namespaces and all their methods remain
intact (metaclasses/descriptors and external callbacks can observe them).
Names are scope-insensitive, so collisions retain extra statements.

Resolver refusal, reflection, user finalizers, unknown imports, non-ASCII source, and bare CR
line endings return the original. Source layouts whose blanked spans would break
statement boundaries also return the original (for example, a multiline literal
assignment ending before a semicolon sibling). Non-ASCII refusal preserves both character
and byte positions. Replacements occupy the original spans and preserve
newlines and semicolon siblings. A prefix alone cannot certify incoming
bindings, so the legacy two-argument API now retains the prefix; source-level
consumers use the validated whole-program adapter. Unsupported model behavior
continues to mean UNKNOWN.

This conservative pass preserves all reachable effects, not merely the
selected error. Therefore existing whole-program proofs may analyze its
output without treating one discharged site as complete property coverage.
The source hash on abstract-interpreter records continues to identify the
original. Search vectors still go through the engine's existing fresh-process
replay on the ORIGINAL path. No sliced execution can issue FALSE directly.

## Consumers and current ceiling

Abstract interpretation (including its loop fixpoints, `solve.absint.prove`),
concolic harness construction (`sv.concolic.solve`), and stationary-loop
planning's determinism/draw-plan step (`sv.prove.prove_dynamic`) use
`solve.slicing.analysis_slice`. It takes open witness entries via
`obligations=`; with none given, it calls `solve.census.obligation_census`
and anchors on that census's first REAL per-site entry
(`solve.slicing._first_anchorable`) -- never the census's own "$coverage"
placeholder (a whole-class coverage refusal recorded at a synthetic module
location, line 1 column 1 function `<module>`, value `False`), which does
not identify any one statement to anchor on. If every open entry for the
requested classes is that placeholder, or the chosen real entry still can't
be pinned to exactly one original statement, `analysis_slice` falls back to
the unsliced source with a reason (`solve.slicing._slice_or_fall_back`)
instead of raising: it never throws into a caller. `sv.prove.prove_dynamic`
runs `solve.determinism.determinism_census` on the ORIGINAL source, not the
slice -- determinism is a whole-program property, and the slice's one
anchor can blank away the evidence (e.g. a later sort) that makes an
order-sensitive-looking operation provably harmless; only its draw plan
(`solve.drawflow.plan_draws`) uses the sliced text. HY and standalone loop
provers can consume `slice_obligation` directly, without another slicer.

This initial implementation deliberately does not infer pure calls, heap
non-aliasing, bounded loop termination, or primitive types for deleting
arithmetic. On the big tasks most code is effectful under this boundary;
speedups must be measured, not assumed.

## Attribution and checks

The dependency fixed point and control-closure idea are adapted from
[ardier/amonhen](https://github.com/ardier/amonhen) commit
`3d721b3a11651b992d1332a0b235225e90ffd009`,
`amonhen/anchor/dataflow.py`, `amonhen/anchor/slice.py` and
`amonhen/slicer/slicer.py` (MIT, copyright Ardi Madadi, 2026).
Its observation-conditioned pruning is not used as a reachability proof.
No source repository or external dependency is modified.

The read-only forward-passes reference at `8bc300c` contains
`passes/forward/amonhen.py` as its standalone-slicer adapter;
`anchor/slice_to_anchor.py` is absent from its branches/history.

`tests/test_slicing.py` runs two non-benchmark programs in fresh processes,
an aliasing mutation and a caught exception whose removal changes the result,
plus control-flow, finalizer, header-effect, namespace, coordinate and
consumer checks. Full-suite, soundness and benchmark measurements are recorded
separately after the gates finish.
