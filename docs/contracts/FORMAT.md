# Stdlib contract format ([internal ref])

## Current entry points

The current verifier consumers and trust boundaries are in the source under `src/tharkun/sv/` and
`src/tharkun/solve/contracts/`. The registry and adapters are executable source;
`tests/contracts/frozen/<qualname>.json` records the per-callable consumer audit and digest.
Read `python -m tests.contracts._frozen summary` or `render` to inspect that inventory without
regenerating it. The batch-one descriptions and the contracts report (not part of this release) retain their historical
scope; they do not mean that the current adapters are unwired.

## Decision

Contracts are a **Python module** under `src/tharkun/solve/contracts/`, not JSON.

Why: Tharkun already models stdlib calls this way. `solve/absint/interp.py::MODULES`
and `solve/census.py::_CALL_SITES`/`_ZERO_DIV_CALLS`/`_LOOKUP_CALLS` are hand-written
Python tables (frozensets and dicts) read directly by the proofs that need them, and
`solve/values.py`'s `IntFact`/`FloatFact`/`StrFact` are frozen dataclasses in exactly
this style. A JSON file would need a second, separately-maintained schema (types,
validation, docstrings-as-comments) to say the same thing a `@dataclass(frozen=True)`
says once, and JSON can't carry the `__post_init__` soundness checks below (every
property class present, `tested=True` backed by an actual test) as anything other than
a second script. Python contracts sit next to the tables they will eventually let
`census.py`/`absint` grow into, are type-checked and lintable like the rest of the
codebase, and cost nothing extra to import (`solve/contracts` is itself under `solve/`,
so it obeys the same "stdlib only, never runs a target" rule as everything else there).

## Schema

`src/tharkun/solve/contracts/types.py` defines four frozen dataclasses:

```python
RaiseClaim(exc: str, condition: str, tested: bool = False)
ReturnSpec(type: str, shape: str = "", tested: bool = False)
Effects(mutates_receiver=False, mutates_args=(), io=False, randomness=None,
        clock_or_env=False, deterministic=True, may_execute_callback=True, notes="")
Contract(qualname: str, raises: tuple[RaiseClaim, ...], may_raise_other: bool,
         returns: ReturnSpec | None, effects: Effects, source: tuple[str, ...],
         tested: bool = False, notes: str = "")
```

### `qualname`

How a contract is looked up (`registry.get(qualname)`):

- a bare name for a builtin function: `"len"`, `"open"`, `"isinstance"`;
- `"TYPE.method"` for a method looked up on any receiver of that type: `"list.pop"`,
  `"dict.__getitem__"`, `"str.startswith"`;
- `"module.name"` for a module-level function or constant: `"math.sqrt"`,
  `"random.randint"`.

### `raises` -- TOTAL over the seven property classes

The seven classes are exactly `tharkun.solve.absint.PROPERTY_CLASSES`'s union
(`AssertionError`, `AttributeError`, `FloatingPointError`, `IndexError`, `KeyError`,
`TypeError`, `ZeroDivisionError` -- these are the classes SV-COMP's
`! uncaught(E)` property ranges over, `docs/SV.md`). **Every contract's `raises` has
exactly one `RaiseClaim` per class, no more, no fewer** (`Contract.__post_init__`
enforces this, so a contract that leaves one out fails to construct). This is the
ticket's soundness rule made mechanical: *"If a condition is not fully known, record
the class with condition 'unknown'. That is sound; a missing class is not."*

`RaiseClaim.condition` is one of:

- `"never"` -- the callable cannot raise this class, for any receiver/arguments;
- `"unknown"` -- it might; the precise condition isn't pinned down (still sound: a
  caller relying on this contract must treat it as possible, so no missing-class bug);
- a short, precise English condition on the receiver/arguments, written so a test can
  check both sides of it (e.g. `"index out of range: not -len(seq) <= index < len(seq)"`).

### `returns`

`None` when the callable's return isn't meaningfully constrained beyond "whatever it
returns" (rare -- prefer a `ReturnSpec` with an empty `shape` over `None` when the type
alone is worth recording). `type` is a typeshed-style annotation string; `shape` is the
range/invariant beyond the bare type (`"in [0, 1)"`, `"same length as seq"`).

### `effects`

Mutation, I/O, randomness (and *which* generator/state), clock/env dependence, and the
derived-but-explicit `deterministic` flag. See the field docstrings in `types.py` for
the exact meaning of each -- they're written to be read there, not duplicated here.

### `source`

At least one citation per contract: a Python 3.14 docs URL, a typeshed stub path, or a
CPython `file::function` (3.14 branch). Prefer the docs URL for raise conditions the
docs state plainly; fall back to CPython source only where the docs are vague, per the
ticket.

### `tested`

`Contract.tested` may only be `True` if every `RaiseClaim` whose condition isn't
`"unknown"`, and the `ReturnSpec` if present, are *themselves* marked `tested=True` --
enforced in `__post_init__`. `"unknown"` can never be `tested` (there is nothing
pinned-down to check); `"never"` is tested by a battery of edge cases observed not to
raise. A claim's own `tested=True` is the author's promise that `tests/contracts/` has
a passing test exercising it on CPython 3.14.4; nothing in `types.py` can verify that
promise mechanically (that's what the test suite is for), so treat a `tested=True`
with no corresponding test as a bug in the batch, not a loophole.

## Loader API

`src/tharkun/solve/contracts/registry.py`:

```python
from tharkun.solve.contracts import get, all_contracts, for_receiver, for_module

get("list.pop")            # -> Contract | None
all_contracts()            # -> tuple[Contract, ...], qualname-sorted
for_receiver("list")       # -> every "list.*" contract
for_module("math")         # -> every "math.*" contract
```

`load_all()` imports each rule module once; `rules.py::register_rule` registers each rule and its contract;
a duplicate `qualname` across modules is a `ValueError` at import time, not a silent
overwrite.

## What this format does *not* do (batch 1)

- It does not change any proof's verdict logic. `census.py` and `absint/interp.py`
  keep their own tables in batch 1; wiring them to read from `solve.contracts` instead
  is a separate, reviewed step per the ticket.
- It does not model every overload of a callable (e.g. `int(x)` vs `int(x, base=)`);
  where a callable's behavior forks on argument shape, the contract's `condition`
  strings say so per-claim rather than the schema growing an argument-shape dimension.

## `protocol.NAME` qualnames (batch 2, [internal ref])

Batch 1's `qualname`s all name a concrete callable (`"len"`, `"list.pop"`,
`"math.sqrt"`). Batch 2 adds contracts for the DISPATCH MECHANISMS behind Python's
special-method protocols -- what runs `a + b` down to `type(a).__add__`/
`type(b).__radd__`, what makes `x in obj` try `__contains__` then `__iter__` then
`__getitem__`, and so on. This matters because Tharkun's abstract interpreter
currently gives up entirely on a user class that defines these dunders (an unmodelled
protocol is `Unsupported`, `docs/LAYERS.md`'s soundness rule), so the mechanism itself
-- not any one callable -- is what needs a tested, sourced specification before the
interpreter can model it.

These live under `qualname = "protocol.<name>"` (`src/tharkun/solve/contracts/rules_protocols.py`),
reusing the exact same `Contract`/`RaiseClaim`/`ReturnSpec`/`Effects` shape and the same
`dunder_propagation_claims`/`never_claims` helpers as every other contract -- a
protocol's `raises` is just as TOTAL over the 7 property classes, for the same reason:
a user-defined dunder invoked by the mechanism can raise anything, and that must
propagate uncaught rather than being silently absent from the claim. What's different
is only the *subject*: the `condition`/`notes` text describes an ORDERED DISPATCH
sequence (which dunder is tried first, when a step is skipped, what makes the whole
thing fall through to `TypeError`) rather than a single function's argument checks.
`returns` is `None` for a protocol whose "result" is just whichever step's return value
won -- there is no fixed type to name beyond what battled it out.

`for_receiver("protocol")` lists all of them.

## The two open-world flags: `may_raise_other` and `effects.may_execute_callback`

Both flags exist for the same underlying reason and must be read together: a contract
whose mechanism invokes arbitrary, unmodeled receiver/argument code (a dunder, a
`key=`/`default=` callback, ...) cannot make a CLOSED-world claim about what that code
does, on either axis.

- **`may_raise_other`** (`Contract`, review finding on batch 2/#47): whether the
  callable can raise something OUTSIDE the 7 property classes already enumerated in
  `raises`. Per `types.py`'s docstring it must be True unless
  every non-property exception has been enumerated and ruled irrelevant, with the
  reason recorded in `source`. A callable that dispatches to an arbitrary dunder can
  never meet that bar -- the user's code can raise `ValueError`, `RuntimeError`, or any
  custom exception -- so every `dunder_propagation_claims`-based contract must set this
  True. (Getting `raises`'s 7 property claims individually correct, batch 2's earlier
  gatekeeper fix, does not settle this flag -- they are independent axes: one is "is
  each of the 7 classes claimed correctly", the other is "is anything outside the 7
  possible at all".)

- **`effects.may_execute_callback`** (`Effects`, this addendum): whether invoking this
  callable can run arbitrary receiver/argument code, full stop -- not just on the
  exception axis. When True, EVERY OTHER `Effects` field (`mutates_receiver`, `io`,
  `randomness`, `clock_or_env`, `deterministic`) describes only the callable's own
  NATIVE dispatch mechanism, never the whole operation. Confirmed empirically
  (`tests/contracts/test_protocols.py`): an `__add__` that reads `os.environ` makes
  `x + 1`'s result depend on process environment on every call, even though
  `protocol.binary_operator`'s own dispatch logic (the slot lookup, the
  `NotImplemented` retry) does none of that itself. A consumer must not read
  `deterministic=True`/`io=False`/`mutates_receiver=False` etc. as a purity or
  determinism certificate for the WHOLE operation when `may_execute_callback=True` --
  only when it is False do the other fields describe the whole operation, the same
  way `may_raise_other=False` is the only case where `raises`'s 7 claims are the whole
  exception story.

Assess the two flags independently. A callback whose exceptions propagate requires
open-world exception claims as well as `may_execute_callback=True`. Cleanup callbacks
such as an evicted object's finalizer may run arbitrary code while their exceptions
are reported as unraisable: they require the callback flag without inventing escaping
exceptions. `never_claims` describes escaping property exceptions, not absence of
callbacks. Choosing that helper is not evidence that a callback audit has happened.

The callback flag defaults to `True`, including legacy entries not yet audited for
callback freedom. Set it `False` only after recording evidence that no receiver,
argument or cleanup callback can run. Propagating callback contracts also require
`may_raise_other=True`; a registry-wide invariant checks this separately from
cleanup callbacks whose exceptions remain unraisable.

## Closed callback reachability

`closed_call_hooks(qualname)` is a separate, narrow proof contract. A non-None
result lists every ordinary user attribute an admitted native call can invoke,
and guarantees returned values come from builtin values, inputs, or their
source-defined hooks. Dunders are never excluded. None means unknown. This does
not assert safety, purity, or determinism. The CPython 3.14.4 source citations
and supported rows live in `contracts/reachability.py`.

The resolver uses it only after excluding external classes, inheritance,
reflection, decorators, annotations, and unsupported implicit protocols. An
unknown receiver includes every native type and every source-defined class;
all applicable native contracts must exist and explicitly referenced user
methods remain live. Thus a missing contract preserves all class methods.
`random.seed` retains encode/decode because isinstance can see a spoofed
`__class__`; mapping expansion and dict conversion retain keys.

## Lazy results: a contract describes the call, not the consumption of what it returns

`zip(...)`, `enumerate(...)` and `reversed(...)` (batch 4) return iterators that run the
arguments' `__next__`/`__getitem__` LATER, when something consumes them; `strict=True`'s
`ValueError` also waits for consumption. Their contracts therefore describe only what
happens when the call is made (`iter()` on each argument, `start.__index__`,
`__reversed__`/`__len__`), and say so in `effects.notes`. A consumer that models the
consumption (a `for`, `list()`) owns that half (`protocol.iteration`). The constructors
`list`/`tuple`/`set` consume at the call, so their `mutates_args` names the iterator argument
that is advanced.

## Shape contracts ([internal ref])

A contract is the union over a callable's call shapes. When one SHAPE has a much stronger, still exact
contract than the union, it is registered under its own qualname instead of weakening the union or
flipping its `tested` gate: `open.output` is the builtin `open` called with a constant write-only mode
(`solve/environment.py::OUTPUT_MODES`: exactly one of `w`/`a`/`x`, no `+`). The union `open` stays
untested (the type it returns depends on the mode); `open.output`'s `ReturnSpec` says `io.TextIOWrapper`
for a text mode and `io.BufferedWriter` for a binary one, both tested. `adapters/common.py::
BUILTIN_SHAPE_CONTRACTS` maps a builtin name to its shape contract, and `adapters/absint.py::
contract_for_builtin` reads it, so a program's `open` sites do not become contract-informed obligations
for the census (it resolves the union qualname) and the model refuses every other shape of `open` itself.

The handle `open.output` returns has its own contracts, `io.TextIOWrapper.<m>` and
`io.BufferedWriter.<m>` for `write`, `writelines`, `flush`, `close`, `__enter__`, `__exit__`
(`contracts/rules_fileio.py`). `tell` and `seek` are not contracted (a text file's `tell` is an opaque cookie
and `seek` takes one; nothing needs an exact claim yet). `ValueError` (a closed file, an encoding error)
and `OSError` (the environment) are not property classes: they are named in each contract's notes and
`may_raise_other` is True, so they never appear in `raises`.

## Hand-verified shape tables (`adapters/common.py`)

A contract is the honest union over a callable's call shapes. Where an engine needs the
stronger fact of ONE shape it reads a small, hand-verified table next to the contract, each
gated on the registry still holding a TESTED contract for the qualname and on the exact call
shape (positional count, no keywords); each entry is pinned by a test that runs CPython:

| table | meaning | consumers |
|---|---|---|
| `CALL_SHAPE` | the arity at which the precise per-qualname facts below apply | absint, concolic |
| `OUTPUT_HANDLE_METHOD_ARITY` | the positional arity (no keywords) at which a method of a write-only file handle is covered by `io.TextIOWrapper.*` / `io.BufferedWriter.*` (`write` 1, `writelines` 1, `flush` 0, `close` 0, `__enter__` 0, `__exit__` 3) | absint, concolic |
| `NO_CALLBACK_SHAPES` | this shape runs no user code and raises nothing, whatever its arguments (`type(x)`) | concolic |
| `CLASS_ONLY_SHAPES` | the result is a function of the CLASS of the first argument only (`type(x)`, `isinstance(x, <builtin classes>)`) | `solve/drawflow.py` |
| `ZERO_ARGUMENT_CONSTRUCTORS` / `SINGLETON_CONSTRUCTORS` | calling the constructor with no argument / one single-element list returns a NEW empty / one-element container | draws, Avengers |
| `adapters/concolic.py::CONCRETE_FOLD_ARITY` | the pure builtins the symbolic executor may evaluate natively on exact builtin values (`ord`, `chr`, `tuple`, `list`, `zip`, `enumerate`, `reversed`) | concolic |

A table entry is a claim about ONE shape; the contract it leans on stays the union. Withdrawing
or un-testing the contract withdraws every table entry that names it (tests in
`tests/contracts/test_adapters_*`).

## Constants, and the table the engines share for a whole module (batch 5)

A module constant is a contract too (`math.pi`, `e`, `tau`, `inf`, `nan`): `raises` is `never` for all seven
classes, `may_raise_other` False, `effects.may_execute_callback` False, `returns` the float with its value
as the shape. Reading one raises nothing and runs nothing; a program that REBINDS it (`math.pi = 3`) is
outside what a contract can see, which `effects.notes` says.

A module whose every function is contracted gets ONE hand-verified table next to the contracts,
`adapters/math_model.py` for `math`: the accepted argument count and keywords, the kind each argument
needs, the kind of the result, the closed range of a float result, and which property classes a call on
exact builtin numbers can raise (`SCALAR_SHAPE_RAISES`). absint and the symbolic executor both read that
table (through `adapters/absint.py` and `adapters/concolic.py`), so the two engines reason about a call
the same way and can exchange facts. Every entry is pinned against CPython by a test, and an entry is
usable only while the registry holds a TESTED contract for the name.

## Rules and visitors (`rules.py`, [internal ref]): the shape of the whole registry

A contract is data; which engine understands which callable used to be scattered across per-engine adapters and
hand tables. `src/tharkun/solve/contracts/rules.py` gives the registry one structure, for ALL contracts (owner:
"all contracts or none"; the existing 234 migrate onto it as one effort, [internal ref]):

* a **`Rule`** is ONE callable's rule (or one member of a parametrized family: `str.split`/`str.rsplit`, the `math`
  functions; `names` lists the qualnames the class covers and `register_rule` makes one instance per qualname). It
  holds the `Contract` (claims, claim-level `tested` flags, unchanged), the call `Shape` (arities, keywords, key
  kinds; `None` = no hand-verified shape, so `applicable()` is False for every call, the `CALL_SHAPE` discipline), the NATIVE semantics (`native(...)`: CPython itself, what the contract tests run against and what a
  candidate generator forward-checks with) and `accept(visitor, ctx)`; `kind` is the visitor dispatch key (the
  snake_case of the class name, or set explicitly). `sample`/`twin` are classmethods used by tests only.
* a **`Visitor`** is ONE engine (absint, `_symexec`, the candidate generator, the Atheris seeder, the fact publisher)
  with one `visit_<kind>(rule, ctx)` per rule kind it understands. `ctx` is the engine's own context and is not
  assumed to be a call site (a protocol rule has none). A kind with no method is a NAMED refusal,
  `Refused(engine, qualname, reason)` (the reason is data; what the engine reports at run time is that object),
  never a silent pass; a kind that cannot apply by design is declared in `not_applicable` with its reason.
* which engine consumes which rule is STATIC: `Visitor.implements()` (the kinds with a visit method) and
  `Visitor.cell(rule)` (`C` consumed, `-` not applicable by design, `R` named refusal) give the engine-by-contract
  audit table without probing; its run-time cells stay with the audit's probes.
* `register_rule` also registers each rule's `Contract` in `registry.REGISTRY`, so `get(qualname)`,
  `all_contracts()`, `for_receiver` and every adapter keep working unchanged while modules migrate one at a time.

Adding a rule is one class; supporting it in an engine is one visit method. `tests/contracts/test_rules.py`.

### Rules publish facts to the store (ruled, not built; owner 2026-10-05, MAIN-approved sketch)

A rule's identity (`Rule.qualname` plus `Rule.kind`) is the vocabulary of what the engines know about a call. Today an
engine that derives something about one (absint: a bound or a type; `_symexec`: a path constraint or a concrete
value; the census: a may-raise claim; the regex facts of `solve/module_namespace.py`) hands it to its own consumer.
Ruled, so that consumers read rule-keyed facts from the run's store instead of calling each other:

* **One canonical item per derivation.** A visit that derives something publishes it with
  `knowledge.py::KnowledgeBase.publish(kind, key, value, producer=, parents=)`: kind `rule_fact` (to be registered in
  `sv/contracts.py::KINDS`: evidence, never certifiable; writers are the visiting components, readers any), key
  `<qualname>@<site>#<claim>` -- `site` is `file:line:col` in the ORIGINAL program's coordinates (through the lowering
  provenance of `solve/lowering`), or `protocol:NAME@site` for a protocol rule, and `claim` is one of `may_raise`,
  `return_type`, `bound`, `path_constraint`, `concrete`, `regex_pattern`, ... -- and a JSON-safe value
  `{"kind": <rule kind>, "claim": ..., "data": ..., "assumes": [<contract claim id, tested flag>, ...]}`. The
  `Derivation` is `producer="<engine>:<rule kind>"` with `parents` the `rule_fact` keys the visit READ, so the chain of
  what was derived from what is recorded. `publish` already gives identity, several derivations per item (two engines
  deriving the same fact is one item with two producers) and a refusal when one identity gets two values, hence the claim
  name in the key.
* **Consumers read, they do not call.** `KnowledgeBase.evidence_since(cursor, kinds={"rule_fact"})`, filtered by rule or
  site: the census reads absint's bound at a `str.__getitem__` site, `_symexec` reads a regex pattern published for a
  `re.compile` site, the Atheris seeder reads the literals published for `str.startswith` sites, Eregion's consumer reads
  `regex_pattern`/`module_constant` facts (`sv/contracts.py` component "static facts (module namespace)").
* **Certification is unchanged and stays out of the visitors.** A `rule_fact` is a HINT (uncertified). Only
  `sv/obligation_table.py::ObligationTable.discharge` (a whole-program proof), `ObligationTable.violate` (a
  replay-confirmed violation) and `sv/eregion_bridge.py::publish_certified` after
  `sv/validate.py::check_candidates_individually` (the independent per-candidate checker) certify anything. A visitor that
  wants a fact to count nominates an `invariant_candidate` (`sv/eregion_bridge.py::publish_fact`) whose parents are its
  `rule_fact` keys, and the independent checker certifies it. What a fact stands on is the contract claim it applied --
  justified by the claim-level `tested` flag and the CPython 3.14.4 tests, recorded in `assumes` -- never by an engine's say-so.
* **Order of work.** First the behaviour-preserving refactor ([internal ref]: the registry onto `Rule`, each engine adapter becoming
  the body of a `visit_<kind>`, byte-identical verdicts on Real-world 80, Solve-only 65, the toys and the fixtures, no new
  semantics); only then the publication, kind by kind, each with its own gate.

### Migration status ([internal ref])

The existing contracts moved onto rules in behaviour-preserving steps: the `Contract` objects are byte-identical
(`tests/contracts/test_registry_digest.py` compares a canonical, hash-seed independent SHA-256 of every contract with
the digests frozen when the migration started, in `tests/contracts/frozen/<qualname>.json` (one file per contract, with its audit row; `tests/contracts/_frozen.py`); a contract that is added,
removed or changed on purpose is a digest change in the same commit) and `get(qualname)` plus every adapter work
unchanged. Migrated, one `rules_<module>.py` each: bisect, builtins, builtins_more, bytes, collections, copy, dict,
fileio, functools, heapq, io, itertools, list, math (`MathFunction`, `MathConstant`: the table-driven families),
operator, protocols, random, re, set, str, strformat, tuple. Every registry entry is now a rule;
`registry.py::load_all` imports only rule modules, and plain-contract registration is removed. Conventions of the migrated rules:

* ONE class per qualname (a family only where the engines treat the members alike: math's 57 functions, the text and
  binary handle of one output method), named after the qualname (`list.append` -> `ListAppend`, `heapq.heappop` ->
  `HeapqHeappop`, `__getitem__` -> `Getitem`); the visitor kind is the snake_case of the class name, so an engine's
  visit for it is `visit_list_append`. A kind names one class (`tests/contracts/test_rules_registry.py`).
* `SHAPE` is set only where a call shape is hand-verified (math, heapq/bisect through `CALL_SHAPE`, the output-handle
  methods); every other rule has none and applies to no call (`Rule.applicable` is False), the discipline of
  `adapters/common.py`.


The engine visits are `solve/absint/visitor.py::AbsintVisitor`,
`sv/symexec_visitor.py::SymexecVisitor`, and
`solve/contracts/visitors.py::CensusVisitor`, `solve/contracts/visitors.py::StationaryVisitor`,
`solve/contracts/visitors.py::AvengersVisitor`, `solve/contracts/visitors.py::DrawsVisitor`,
`solve/contracts/visitors.py::ProductVisitor` and `solve/contracts/visitors.py::EregionVisitor`.
The last two explicitly refuse all contracted calls. Existing adapter import paths
remain stable for SET ([internal ref]) and CTF ([internal ref]); transfer functions and tested gates are unchanged.
`tests/contracts/_engine_audit.py::measure` starts with `rules.py::Visitor.cell` and refines it with runtime probes:
C with consumption stays C; C never consumed is X; runtime refusal stays R/G;
R handled by an existing hand model is M. A runtime C without a visit fails the audit.
Samples and wrong-arity twins live on each rule through `rules.py::Rule.sample` and `rules.py::Rule.twin`.

The original [internal ref] migration pinned the canonical registry with SHA-256 `23aa2f868ef54381b48589ef49200f55aa5e1eb7c70119e487e2444000a22462`.

SET ([internal ref]) removed the temporary `solve/contracts/set_.py` view (and `rules_set.py::CONTRACTS`) and added the 24 set rules to
`solve/contracts/rules_set.py`; absint consumes them through `solve/absint/visitor.py` (`visit_set_*`). The registry checkpoint
`test_full_registry_sha256_matches_string_family_plus_set_family_checkpoint` pins SHA-256 `ebf8de2321ddb5938a6b4d0adad6a582ddcd3516c73798f2b1e9d0834e2ea87c`: the string-family
registry plus those 24 rules (every earlier digest is unchanged).
CTF ([internal ref]) keeps `sv/_symexec.py` in use and may need an ordinary merge there;
`sv/symexec_visitor.py::CallCtx` is its stable contract visit context.

## Whole string, formatting and regex family ([internal ref])

`src/tharkun/solve/contracts/rules_str.py` contains all 47 public string methods,
plus string containment, indexing and `__format__`. Percent formatting belongs to
`rules_strformat.py::StrMod`; `Format`, `Repr`, `Ascii`, `FormattedValue` and
`JoinedStr` cover builtin conversions and f-string AST operations. Regex module
functions and every public `Pattern`/`Match` method belong to `rules_re.py`.
Each has its own Rule class and visitor kind, with the existing registry API.

`solve/absint/visitor.py::AbsintVisitor._native_string` enumerates at most 64 combinations of finite,
callback-free builtin operands using `absint/native.py::concrete` and `product`.
`native.py::abstract` keeps exact string contents (and therefore exact lengths),
integer results, tuple identity and capture objects. Dict key sets must agree;
unsupported bytes results and unknown heap/object values receive named refusals.
A `finditer` result is an `IteratorObj`, not a list; consumption remains the
existing conservative iterator model. Unknown plain strings retain result types
and conservative lengths where supported. A finite percent template with
callback-free numeric/str operands proves only the tracked exception classes
whose absence is established by `_percent_safe`; unknown results have length
0..unbounded. The Richards example `"Bad task id %d" % id` reaches this seam.

`sv/symexec_visitor.py::SymexecVisitor._native_string` uses the same native Rule semantics and
`rules_strformat.py::native_limit` admission guard. Symbolic startswith/endswith
and substring membership have exact constraints with Python start/end normalization, including
negative bounds and empty prefixes. Empty-spec formatting of symbolic strings and
integers is exact; integer paths respect `sys.get_int_max_str_digits`. CPython 3.14.4 applies !s/!r/!a before evaluating nested specs; both AST executors visit the conversion rule before the spec, then FormattedValue with the already converted value. JoinedStr
concatenates only already formatted strings. Other symbolic formats and regexes
have a refusal naming the rule and the missing exact encoding.
`sv/_symexec.py::_Exec.ev_Call` checks module/builtin identities and admits native
import aliases and unbound str calls; `sv/_symexec.py::_Runner._contract_call` defers these calls
so each visitor receives evaluated operands. Target callback objects are never
executed by either native fold.

`native_limit` caps eager operand strings/specs and widths at 4096 units and
containers at 64 entries. Its conservative formatting-number guard also refuses
large numeric field/width operands (plain `%c` is a bounded single-codepoint exception). It refuses custom codecs/error handlers, regex DEBUG output and unbounded native
backtracking: regex folds admit at most one unbranched repeat, without nested
repeats or backreferences. Atomic groups and lookarounds are inspected recursively.
The visitors separately refuse scanner state and recursive/unsupported container values; the symbolic native guard also bounds container nesting. Malformed patterns/templates retain their native exceptions; these limits are
named UNKNOWN outcomes, not invented results or successful-path truncations.

### Native exception conditions

All method rules require a str receiver, with TypeError for invalid call arity,
keywords or receiver kind. Index/width/count arguments invoke `__index__` and
propagate its exceptions. `splitlines(keepends)` instead truth-tests its argument
through `__bool__`/`__len__`; a float is valid here. No string subclass or arbitrary
object is executed by an engine fold. Callback freedom in a contract is separate
from argument-kind admission in an engine.

| Operation(s) | TypeError | ValueError / other native failure | KeyError |
|---|---|---|---|
| `%`: d/i/o/u/x/X/e/E/f/F/g/G/c/r/s/a, flags `#0- +`, widths/precision, h/l/L, tuple/dict | Missing/excess arguments, incompatible conversion type, non-integer `*` width/precision, non-mapping keyed operand | Malformed/unsupported conversion; `%c` outside Unicode range raises OverflowError; decimal digit limit | Missing mapping key |
| `str.format`, `format_map` | Invalid call shape; non-mapping format_map operand; field/conversion/format callback errors | Malformed braces, invalid conversion/spec, mixed automatic/manual numbering; format_map positional fields forbidden; positional field out of range is IndexError | Missing named or item field |
| `format`, `str.__format__`, FormattedValue | Non-str spec, non-str conversion/format result; nonempty spec for default object formatter | Invalid type-specific mini-language spec; decimal digit limit | Only a dispatched target callback can supply it |
| JoinedStr | Any non-str part | None from its native joining semantics | Never |
| capitalization/casefold/case predicates, lower/upper/title/swapcase | Invalid shape/receiver | None from native character conversion | Never |
| center/ljust/rjust | Non-index width; fill is not str or its length is not 1 | Index callback may raise any class | Only an index callback |
| count/find/rfind/index/rindex | Non-str substring; start/end not index-like | index/rindex raise ValueError if absent; find/rfind return -1 | Only an index callback |
| startswith/endswith | Prefix/suffix not str or tuple; tuple members checked only until first match; bounds not index-like | Index callback may raise any class | Only an index callback |
| join | Not iterable or a non-str yielded element | Iterable callbacks may raise any class | Only an iteration callback |
| split/rsplit | Invalid separator or non-index maxsplit | Empty separator | Only an index callback |
| strip/lstrip/rstrip, removeprefix/removesuffix | Wrong chars/prefix/suffix kind (strip permits None) | None from native stripping/removal | Never |
| partition/rpartition | Non-str separator | Empty separator | Never |
| replace | Old/new not str; count not index-like | Index callback may raise any class | Only an index callback |
| expandtabs/zfill | Tabsize/width not index-like | Index callback may raise any class | Only an index callback |
| splitlines | Invalid shape or invalid truth-protocol result | Truth callbacks may raise any class | Only a truth callback |
| maketrans | One-argument input not dict; keys not int or str; multi-argument inputs not str | Unequal first/second lengths; a string key not exactly one character | Never |
| translate | No table subscription, or result not int/str/None | Ordinal outside Unicode range; other table errors propagate | LookupError (including KeyError and IndexError) from table is swallowed; original character retained |
| encode | Encoding/errors not str; invalid shape | Unknown codec/error handler LookupError; encoding failures UnicodeError | Custom codec/error callback may supply it; engines refuse custom handlers |
| containment/indexing | Non-str containment operand; key/bounds not index-like | Out-of-range integer IndexError; slice step zero ValueError | Only an index callback |
| re module / Pattern match/search/fullmatch/findall/finditer/split/sub/subn | Invalid argument shape/kind, pattern/text str-vs-bytes mismatch, invalid index/count/positions, wrong replacement kind | PatternError for malformed pattern/template; ValueError for invalid flags or nonzero flags with compiled pattern; named missing template group IndexError | Only callbacks |
| Match.group/getitem/start/end/span | Invalid call shape; group index protocols may propagate TypeError | Missing numeric/named group is IndexError; unmatched group returns None/-1, not an exception | Only a group index callback |
| Match.groups/groupdict | Too many defaults, unknown keyword or duplicate default | None from native extraction | Never |
| Match.expand | Invalid call shape/template kind | Invalid template PatternError; unknown named template group IndexError | Only a dispatched callback |
| re.escape / purge / Pattern.scanner | Invalid native argument shape/kind | escape may dispatch a subclass override; scanner consumption is separately unmodelled | Only callbacks where native dispatch invokes one |

### Fixed checks and evidence

`tests/contracts/test_string_rules.py` uses authored tables against CPython 3.14.4,
both visitors, AST entry points, native exception boundaries and callback propagation.
It includes Unicode, all percent conversions, flags/length modifiers, tuple/dict
operands, nested f-string specs, attributes/items, import aliases, iterator identity,
invalid calls and resource refusals. No new random-program generator is added.
`tests/contracts/test_engine_audit_consumed.py` retains native replay and withdrawal
checks for every C cell; the authored result guard is selected by a unique marker,
since an earlier synthetic bounds site is a distinct obligation. Regex matches and
iterators use group/position/span guards rather than non-executable object reprs.

`docs/contracts/evidence/1126/failure-classification.md` lists every failing node of
the first full suite with cause and correction (a: 1 real callback-metadata bug;
b: 93 new consumed-cell withdrawal/guard checks; c: 0 stricter-refusal changes).
`full1126-1.log` preserves that run. The final registry snapshot, full-suite log and
campaign receipt accompany the final validation record in this directory.

Non-finite formatting visits validate call shape before proving an error absent: unexpected keywords on `%`/JoinedStr and invalid FormattedValue conversion codes keep the native TypeError/KeyError checks unproven. `tests/contracts/test_string_rules.py::test_nonfinite_format_call_shape_keeps_native_errors_unproven` pins each boundary with a fixed native counterexample.

`sv/symexec_visitor.py::SymexecVisitor._native_string` checks that a symbolic f-string conversion code is a plain integer before set membership, so refusal cannot hash an unverified object. `tests/contracts/test_string_rules.py::test_symbolic_fstring_conversion_never_hashes_unverified_objects` records the boundary.

The f-string syntax rule's conversion-code context is `-1`, `115` (!s), `114` (!r) or `97` (!a), as emitted by valid Python ASTs. Malformed synthetic contexts can raise native KeyError/TypeError during conversion dispatch; the non-finite visit never proves these malformed calls safe. `tests/contracts/test_string_rules.py::test_fixed_high_unicode_symbolic_string_operations` pins exact symbolic prefix/suffix/membership semantics at U+10FFFF against native Python.

RegexFlag constants retain readonly enum alternatives alongside their integer view (`solve/absint/native.py::abstract` and `solve/absint/interp.py::Analyzer.module_attr`), so formatting never substitutes plain-int repr/str for the enum. `solve/absint/interp.py::Analyzer._flag_operator` folds only admitted finite callback-free operators, preserving IntFlag inversion and bitwise results; larger domains/attributes refuse. The numeric/enum union may lose precision on formatting, but contains every native result. `tests/contracts/test_string_rules.py::test_regexflag_identity_never_proves_plain_int_semantics` pins the four native counterexamples.

Fixed flag checks also pin arithmetic/pattern entry points, aliases and NOFLAG truth testing in `tests/contracts/test_string_rules.py::test_fixed_regexflag_arithmetic_and_pattern_entrypoints` and `tests/contracts/test_string_rules.py::test_fixed_regexflag_alias_and_zero_never_prove_wrong_assertions`.

The RegexFlag preservation is also the legacy main soundness fix [internal ref]: native `~re.IGNORECASE` is 509, while plain sv_2027 proved the false `== -3` assertion TRUE. The corrected numeric/enum overapproximation leaves it unproven. Exact main/head controls and loaded-module provenance are committed in `docs/contracts/evidence/1126/flag1143-main.json` and `flag1143-head.json`; the prior code fix is `79af6a2`, followed by a [internal ref] evidence/provenance commit without rewriting pushed history.

[internal ref] also preserves readonly enum module constants in `sv/_symexec.py::_Exec.ev_Attribute`. `sv/_symexec.py::_num` projects a RegexFlag only when a numeric operation needs an int, matching unary plus through aliases/parameters. `_Exec.ev_BinOp` refuses a symbolic enum-producing bitwise result before discarding its identity. Direct authored-AST/native controls in `tests/contracts/test_string_rules.py::test_symbolic_regexflag_module_constants_keep_enum_identity`, `test_symbolic_unary_plus_removes_regexflag_enum_identity` and `test_symbolic_regexflag_bitwise_result_refuses_without_enum_encoding` cover these sibling paths.

MG #147 follow-ups (carried by [internal ref]): `adapters/draws.py::range_kwargs` consumes only `adapters/common.py::closed_range_return`, so unrelated draws visits cannot leak boolean support facts as numeric ranges. `sv/_symexec.py::_Exec.ev_Attribute` consumes only math constants; `ev_Call` consumes only math functions and explicitly refuses constant calls as native TypeError. Function-valued attributes remain named Unsupported. `rules_builtins.py::Open` and seven `rules_random.py` classes retain fixed native TypeError arity twins. Contracts and digests are unchanged.

[internal ref] / MG #151: `solve/absint/visitor.py::AbsintVisitor._percent_safe` proves tuple operand shape only for exactly one tuple reference with every other value component bottom. Joining a scalar or static alternative never inherits the tuple's arity facts. `solve/absint/interp.py::Analyzer.static_attr` admits bare native/abstract regex method reads only when that owner-qualified Rule exists; readonly native fields must actually exist on that closed native object and be representable. Other reads refuse by name. Declared methods take precedence over field names, so Match.groups aliases stay callable while Pattern.groups is numeric data. Native flag/slice, str-type and method-tag secondary attribute paths also refuse unmodelled names; list/array paths already whitelist methods and builtin scalar reads refuse. Fixed native TypeError/AttributeError controls and valid aliases in test_string_rules.py; no new random program/native executor.

[internal ref] stack boundaries: `solve/absint/interp.py::Analyzer.subscript_load` translates an unhashable closed native slice tag into named refusal; representable bounds retain their native error obligations. `sv/symexec_visitor.py::SymexecVisitor._native_string` reuses `sv/_symexec.py::Ref._refuse` for direct modelled user-object operands, preserving the existing object-specific reason without invoking target conversions or formatting hooks. Fixed list/dict/nested-list slice bounds and conversion/format visitor siblings are checked in test_string_rules.py. Only the measured absint re.Match.groups audit cell changes R→C; native replay and contract withdrawal stay mandatory.

[internal ref] Ref admission uses exact `type(value) is Ref`, since isinstance can consult a target operand's __class__. The fixed hostile __getattribute__ check in test_string_rules.py proves the visitor refuses without any target lookup. Raw before failure and incomplete c21 validation cancellation are retained in the evidence directory.

The same no-target-lookup boundary applies to symbolic string/integer admission in SymexecVisitor._native_string: exact Z3 SeqRef/ArithRef/IntNumRef identities are admitted; unrelated objects refuse without the general isinstance-based _is_sym predicate. Fixed contains/prefix/suffix/bounds/str/format siblings retain this check.

Closed native type admission also uses identity comparisons, preserving refusal without target metaclass hash/equality/lookup. The hostile-metaclass fixed boundary check is retained in test_string_rules.py.
