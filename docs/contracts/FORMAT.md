# Stdlib contract format (#1070)

## Current entry points

Use the [proof/contract seam map](../AGENT_REFERENCE.md#proof-and-contract-fast-entry) for the
current verifier consumers and trust boundaries. The registry and adapters are executable source;
`tests/contracts/frozen/<qualname>.json` records the per-callable consumer audit and digest.
Read `python -m tests.contracts._frozen summary` or `render` to inspect that inventory without
regenerating it. The batch-one descriptions and [REPORT.md](REPORT.md) retain their historical
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

### Function-entry raising inventory

**Ruled 2026-10-29:** [Static exception sites](../EXCEPTION_SITES.md) owns the
shared ORIGINAL-source inventory and the built entry/census views.
`adapters.exception_sites.condition_table` projects
the existing `safety.cases` into JSON-safe typed domains, exact conditions and
explicit projection gaps. Safe cases are retained; English claims are not
parsed and unavailable Python projections are not weakened. The source builder
is `solve.exception_sites`; `census.site_census` keeps its audited closed-program
coverage boundary and `raise_sites.entry_raise_sites` is an unordered witness
projection. Only Eregion owns evaluation DFS and call-path contexts.

Conditional unknown rows use the complement of existing `safety.admission`,
or canonical true when no machine cases exist. `unknown: true` and null
`raises`/`condition` never denote false. The contract's other-family wildcard
has `excluded_classes` for named claim families, each independently covered by
exact cases, explicit unknown cases, or tested never claims. Unresolved dispatch
has no family exclusions. See EXCEPTION_SITES.md for receiver applicability and
the requirement to prove an unknown domain false before discharging it.

`src/tharkun/solve/raise_sites.py::entry_raise_sites` projects the shared source inventory's registry-backed
`safety.cases`, and publishes each concrete contract exception class with its
operand expressions, source span, canonical condition and checked Python predicate. An unshadowed builtin
catch superclass selects the contract subclasses it catches (`Exception` includes `TypeError`; `LookupError`
includes `KeyError` and `IndexError`). This is a static may-raise inventory, never a reachability or absence proof.
One- and two-positional-argument builtin calls retain native operand order; shadowing, starred and keyword calls
remain unresolved dispatches.

`src/tharkun/solve/contracts/safety/rules_builtins.py::classcheck_cases` supplies a scoped TypeError domain for
`isinstance` and `issubclass`: both operands must have exact type int, bool, float or str. The second operand
cannot be classinfo and neither operand can run a user hook. Valid classes, tuple/union classinfo and all other
types remain outside this template. `safety.__init__._case_builders` registers these builders next to len and int;
the inventory consumes the same condition that safety consumers use. Native parity and exclusions are pinned by
`tests/contracts/test_safety_builtin_templates.py` and `tests/test_entry_raise_sites.py`.

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

## `protocol.NAME` qualnames (batch 2, #1070)

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

## Shape contracts (#1082)

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

## Rules and visitors (`rules.py`, #1126): the shape of the whole registry

A contract is data; which engine understands which callable used to be scattered across per-engine adapters and
hand tables. `src/tharkun/solve/contracts/rules.py` gives the registry one structure, for ALL contracts (owner:
"all contracts or none"; the existing 234 migrate onto it as one effort, #1133):

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
* **Order of work.** First the behaviour-preserving refactor (#1133: the registry onto `Rule`, each engine adapter becoming
  the body of a `visit_<kind>`, byte-identical verdicts on Real-world 80, Solve-only 65, the toys and the fixtures, no new
  semantics); only then the publication, kind by kind, each with its own gate.

### Migration status (#1133)

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
`solve/contracts/visitors.py::ProductVisitor`, `solve/contracts/visitors.py::EregionVisitor` and
`solve/preimage/visitor.py::PreimageVisitor` (the backward engine, below). ProductVisitor and EregionVisitor
explicitly refuse all contracted calls. Existing adapter import paths
remain stable for SET (#1135) and CTF (#1134); transfer functions and tested gates are unchanged.
`tests/contracts/_engine_audit.py::measure` starts with `rules.py::Visitor.cell` and refines it with runtime probes:
C with consumption stays C; C never consumed is X; runtime refusal stays R/G;
R handled by an existing hand model is M. A runtime C without a visit fails the audit.
Samples and wrong-arity twins live on each rule through `rules.py::Rule.sample` and `rules.py::Rule.twin`.

The original #1133 migration pinned the canonical registry with SHA-256 `23aa2f868ef54381b48589ef49200f55aa5e1eb7c70119e487e2444000a22462`.

SET (#1135) removed the temporary `solve/contracts/set_.py` view (and `rules_set.py::CONTRACTS`) and added the 24 set rules to
`solve/contracts/rules_set.py`; absint consumes them through `solve/absint/visitor.py` (`visit_set_*`). The registry checkpoint
`test_full_registry_sha256_matches_string_family_plus_set_family_checkpoint` pins SHA-256 `ebf8de2321ddb5938a6b4d0adad6a582ddcd3516c73798f2b1e9d0834e2ea87c`: the string-family
registry plus those 24 rules (every earlier digest is unchanged).
CTF (#1134) keeps `sv/_symexec.py` in use and may need an ordinary merge there;
`sv/symexec_visitor.py::CallCtx` is its stable contract visit context.

## Whole string, formatting and regex family (#1126)

`src/tharkun/solve/contracts/rules_str.py` contains all 47 public string methods,
plus string containment, indexing and `__format__`. Percent formatting belongs to
`rules_strformat.py::StrMod`; `Format`, `Repr`, `Ascii`, `FormattedValue` and
`JoinedStr` cover builtin conversions and f-string AST operations. Regex module
functions and every public `Pattern`/`Match` method belong to `rules_re.py`.
Each has its own Rule class and visitor kind, with the existing registry API.

`solve/absint/visitor.py::AbsintVisitor._native_string` enumerates finite combinations of
callback-free builtin operands using `absint/native.py::concrete` and `product`, sharing the
analysis deadline and one per-analysis optional expansion-work allowance.
Work exhaustion discards the whole concrete result and delegates to the existing abstract
transfer; CPU or memory exhaustion refuses. No prefix can establish a property.
`native.py::abstract` joins string contents (widening to arbitrary strings when needed),
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
have no exact encoding. Since #1170 PR4 (#1189) a `_Runner` CONCRETIZES them
(`sv/symexec_visitor.py::SymexecVisitor._concretize`). Each symbolic str/int operand takes its value in a model of the
current path, the path is restricted to that value, and the call folds natively. The run is then an
under-approximation: a `sat` is a candidate replayed like any other, a site it misses is `unknown` (never
`unsat`/`unreached`, `sv/_symexec.py::solve_targets`), and `run_exact` refuses it as inexact, so no proof leans on
a dropped path. Operands that are not symbolic str/int terms (a list of symbolic parts for `join`) keep the refusal
naming the rule. Choosing the concrete value from the preimage of the site's demand, instead of the path's model, is
ruled, not built. Compiled `re.Pattern` module globals are exact constants of the run.
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

The RegexFlag preservation is also the legacy main soundness fix #1143: native `~re.IGNORECASE` is 509, while plain sv_2027 proved the false `== -3` assertion TRUE. The corrected numeric/enum overapproximation leaves it unproven. Exact main/head controls and loaded-module provenance are committed in `docs/contracts/evidence/1126/flag1143-main.json` and `flag1143-head.json`; the prior code fix is `79af6a2`, followed by a #1143 evidence/provenance commit without rewriting pushed history.

#1143 also preserves readonly enum module constants in `sv/_symexec.py::_Exec.ev_Attribute`. `sv/_symexec.py::_num` projects a RegexFlag only when a numeric operation needs an int, matching unary plus through aliases/parameters. `_Exec.ev_BinOp` refuses a symbolic enum-producing bitwise result before discarding its identity. Direct authored-AST/native controls in `tests/contracts/test_string_rules.py::test_symbolic_regexflag_module_constants_keep_enum_identity`, `test_symbolic_unary_plus_removes_regexflag_enum_identity` and `test_symbolic_regexflag_bitwise_result_refuses_without_enum_encoding` cover these sibling paths.

MG #147 follow-ups (carried by #1126): `adapters/draws.py::range_kwargs` consumes only `adapters/common.py::closed_range_return`, so unrelated draws visits cannot leak boolean support facts as numeric ranges. `sv/_symexec.py::_Exec.ev_Attribute` consumes only math constants; `ev_Call` consumes only math functions and explicitly refuses constant calls as native TypeError. Function-valued attributes remain named Unsupported. `rules_builtins.py::Open` and seven `rules_random.py` classes retain fixed native TypeError arity twins. Contracts and digests are unchanged.

#1153 / MG #151: `solve/absint/visitor.py::AbsintVisitor._percent_safe` proves tuple operand shape only for exactly one tuple reference with every other value component bottom. Joining a scalar or static alternative never inherits the tuple's arity facts. `solve/absint/interp.py::Analyzer.static_attr` admits bare native/abstract regex method reads only when that owner-qualified Rule exists; readonly native fields must actually exist on that closed native object and be representable. Other reads refuse by name. Declared methods take precedence over field names, so Match.groups aliases stay callable while Pattern.groups is numeric data. Native flag/slice, str-type and method-tag secondary attribute paths also refuse unmodelled names; list/array paths already whitelist methods and builtin scalar reads refuse. Fixed native TypeError/AttributeError controls and valid aliases in test_string_rules.py; no new random program/native executor.

#1153 stack boundaries: `solve/absint/interp.py::Analyzer.subscript_load` translates an unhashable closed native slice tag into named refusal; representable bounds retain their native error obligations. `sv/symexec_visitor.py::SymexecVisitor._native_string` reuses `sv/_symexec.py::Ref._refuse` for direct modelled user-object operands, preserving the existing object-specific reason without invoking target conversions or formatting hooks. Fixed list/dict/nested-list slice bounds and conversion/format visitor siblings are checked in test_string_rules.py. Only the measured absint re.Match.groups audit cell changes R→C; native replay and contract withdrawal stay mandatory.

#1153 Ref admission uses exact `type(value) is Ref`, since isinstance can consult a target operand's __class__. The fixed hostile __getattribute__ check in test_string_rules.py proves the visitor refuses without any target lookup. Raw before failure and incomplete c21 validation cancellation are retained in the evidence directory.

The same no-target-lookup boundary applies to symbolic string/integer admission in SymexecVisitor._native_string: exact Z3 SeqRef/ArithRef/IntNumRef identities are admitted; unrelated objects refuse without the general isinstance-based _is_sym predicate. Fixed contains/prefix/suffix/bounds/str/format siblings retain this check.

Closed native type admission also uses identity comparisons, preserving refusal without target metaclass hash/equality/lookup. The hostile-metaclass fixed boundary check is retained in test_string_rules.py.

## Preimages (#1170): the backward engine over the same rules

An anchored (`through`) `driver.preimage_candidates` query also proposes boundary members for a regex demand on
`left + separator + right`. It collects adjacent lengths from source `len(name)`
comparisons, then asks the existing regex segment generator for those lengths.
The separator must be one literal character; at most sixteen segment combinations
are tried, within the existing 4096-character native-check bound. These values
are candidates, never a narrowed domain or an absence proof. Source guards and
fresh-process replay still decide each row, and ordinary short members remain
available. The candidate walk interleaves ordinary and boundary proposals, with
the ordinary proposal first, so finite row budgets retain the short witnesses. Unanchored queries keep their existing shortest-first behavior. This carries a guard's boundary through split/concatenation rather
than sampling only the regex's shortest local and domain parts.

`solve/preimage/visitor.py::PreimageVisitor` (engine `"preimage"`) is the ONE backward-propagation capability for the
str family. For one call whose operands are all known builtin values except one, it answers which values of that
unknown operand make the result satisfy a demand. A demand (a `Want`) is a small closed vocabulary in
`solve/preimage/want.py`: `Eq`, `OneOf`, `Truth`, `IsNone`, `Compare`, `Length`, `Prefix`/`Suffix`/`Contains`, regex
`Member`, `Items` (a list/tuple result), the nominal type atoms `IsInstance` / `IsSubclass` (#1386), and `All`/`AnyOf`/`Not`. It is keyed by the rule's qualname like every other
visitor (`get_rule(qualname).accept(PreimageVisitor(), ctx)`).

It visits every `rules_str.py` method, `str.__getitem__`/`__contains__`/`__mod__`, the `rules_strformat.py`
conversions, `len`/`ord`/`chr`/`set`/`str`/`repr`/`ascii`/`format`, str `+`/`*` (`protocol.binary_operator`), the list
and tuple index rules a split result is read through, and every `rules_re.py` rule.
`tests/contracts/test_preimage_audit.py` pins that the whole family is `C` and everything else a named refusal.

A visit returns a `Preimage` with two parts:
* `transfer`: the exact demand on the unknown operand, when the vocabulary can state it. For example, `x.startswith(p)`
  wanted true is `Prefix(p)`, and `len(x) > 320` is `Length(321, None)`;
* the candidates: lazy, program literals first, then simplest first, then by length.

A visit can also return a named `Refused`. This happens when an operand is not of a builtin kind
(`adapters/str_shapes.py` kinds for the str methods; `type(x) is ...` everywhere else), when the unknown is in a
position the kind does not invert, or for a Match receiver, a callable replacement, or a format spec.

Every candidate is FORWARD-CHECKED before it is yielded: `want.holds(Rule.native(*operands))`, i.e. CPython's own
semantics on builtin values. The check is bounded by `rules_strformat.py::native_limit`; for regex calls it is bounded
by `solve/preimage/regex_members.py::check_cost` (no backreference, a work estimate under `WORK_LIMIT`; a nested unbounded repeat is
charged `2**n` per nesting level and is refused outright above `_NESTED_MAX_LEN` characters). That estimate is a worst
case over all strings of the length, so a FINITE estimate over `WORK_LIMIT` is not a size cap: `regex_members.py::admitted`
admits the check when the very same call finishes inside `PROBE_SECONDS` in a forked child killed at the deadline
(`_probe`; the result is discarded and the parent then makes the call), so python-string-utils `HTML_RE.sub('', s)` is
checked at 65 and 257 characters (estimate ~1e9 at 33). An infinite estimate (backreference, nested unbounded repeat past
`_NESTED_MAX_LEN`), a missing `fork`, a failed fork and a probe timeout are refusals (`UNCHECKED`/`None`): the check is not
made, never made wrongly. `driver.py`'s own `check_cost` guards (the callback-template path) are unchanged. So no target code, no user hook and no catastrophic backtrack runs. A candidate is a witness the caller
verifies by execution; an empty answer or a refusal is never infeasibility.

`re.sub`/`subn` (module and `re.Pattern`) with a plain-str replacement reverses EVERY replaced occurrence
(`visitor.py::_regex_sub`): for each pattern member `m` it proposes the observation with all occurrences of the
replacement text turned into `m`, then first-only and last-only, then single insertions. An observation with several
replacements (`'top-10-reasons'` from `'top 10 reasons'`, python-string-utils `slugify`) therefore has a preimage that
restores every separator. Each proposal is still FORWARD-CHECKED natively, so a proposal that does not reproduce the
observation is dropped; mixed per-occurrence members and a replacement text that also occurs unreplaced are not
searched (the all-occurrences text is one preimage, not the set of them).
`zlib.compress`/`zlib.decompress` and `base64.b64encode`/`b64decode`/`urlsafe_b64encode`/`urlsafe_b64decode` have
standard-library inverse-pair preimages (`visitor.py::_codec`, `_base64_pair`; owner 864: supplied inverses over core
stdlib facts). The forward check is the stdlib call itself (these modules are not in the contract table's native modules,
so `Rule.native` is not used): `Eq(stream)` on `compress(HOLE, level, wbits)` proposes `decompress(stream)` bounded to 16 MiB
(`_inflate`, a decompression bomb gives nothing) and keeps it only if `compress(candidate, level, wbits)` reproduces the
stream, so a stream made at another level or wbits has NO preimage rather than a wrong one (compress is deterministic, so
`decompress(stream)` is the only possible preimage). The level and wbits must be known (a refusal otherwise); when the
LEVEL is the unknown and the data known, every level (-1..9) is tried against the observation. `decompress(HOLE)` proposes
the data compressed at each level (9 first) and keeps those that decompress back. `b64encode(HOLE)` proposes the strict
decode (`validate=True`; a non-canonical text such as `b"YR=="` has no preimage); `b64decode(HOLE)` proposes the canonical
encoding as bytes and as the ASCII str (the text variants that the non-validating decoder would also accept, such as
embedded non-alphabet bytes, are not enumerated). `altchars` must be known. `bytes.decode` / `str.encode` around them are
separate steps (both have preimages). Tests: `tests/contracts/test_preimage_zlib_base64.py`.
`bytes.decode(HOLE, encoding, errors)` (`visitor.py::visit_bytes_decode`; only the receiver may be unknown) proposes
`str.encode` of each observation seed with the same codec, kept only if it DECODES BACK to the observation with the very same
`errors`. Codecs are the builtin utf-8/ascii/latin-1/utf-16/utf-32 family and handlers strict/ignore/replace/
backslashreplace/surrogateescape/surrogatepass; a registered codec or handler is user code and a named refusal, as is an
unknown or non-str `encoding`/`errors`. Under `strict` the re-encoding is the only preimage. Under `ignore`/`replace`/... it
is ONE preimage, not all of them: the invalid bytes the handler dropped or replaced are not recoverable from the text and are
not enumerated; text the codec cannot encode (`'é'` for ascii, a lone surrogate for strict utf-8) has none.
Tests: `tests/contracts/test_preimage_bytes_decode.py`.
`unicodedata.normalize(form, HOLE)` (`visitor.py::visit_unicodedata_normalize`; the form must be a literal NFC/NFD/NFKC/NFKD and
only the text may be unknown) proposes the observation, its full (compatibility) decomposition and, wherever a 1-3 character
piece of it is the normalization of other code points (`'fi'` of U+FB01, `'A'` of U+FF21, `'\u00e9'` of `'e\u0301'`; the
table is one cached scan of all code points per form, `_normalization_sources`), that source at one place and at every
place. Each proposal is FORWARD-CHECKED with `unicodedata.normalize`. normalize is idempotent, so an observation that is not
itself in the form has NO preimage and the answer is empty. The enumeration is sound, not exhaustive (the remaining
combinations of sources per position are not generated), so an empty answer for an observation already in the form never
occurs but a missing source is not a proof it does not exist. Tests: `tests/contracts/test_preimage_unicodedata_normalize.py`.
`json.loads(HOLE)` and `json.dumps(HOLE, **options)` (`visitor.py::visit_json_loads`, `visit_json_dumps`). `loads`: only the
text may be unknown and NO keyword is admitted (`cls`, `object_hook`, `parse_*`, `object_pairs_hook` are user code or change the
result types); it proposes `json.dumps` of each observation seed in the default, compact (`separators=(',', ':')`),
`ensure_ascii=False` and `indent=2` spellings, each also padded with insignificant whitespace, and keeps a text only if
`json.loads` returns a value of EXACTLY the seed's types, recursively (`1` vs `1.0` vs `True`, `list` vs `tuple`, int vs str
keys; `visitor.py::_same_json`), so `Eq(1)` is never answered with `'1.0'`, and tuples, int-keyed dicts, sets, bytes and NaN
(`nan != nan`) have no preimage; `Infinity` is the extension spelling of `inf`. `dumps`: only the object may be unknown and the
options `skipkeys`, `ensure_ascii`, `check_circular`, `allow_nan`, `indent`, `separators`, `sort_keys` must be literals;
`default` and `cls` must be absent or `None` and any other keyword is a named refusal. It proposes `json.loads` of the text and
keeps it only if dumping with the SAME options reproduces the text, so a text spaced differently from what the options write has
no preimage. One preimage, the JSON-native value: tuples, int-keyed dicts and subclasses that write the same text are not
enumerated. `want.py::is_builtin_value` now admits an exact `dict` of builtin keys and values (depth and size bounded), which
`json.loads` results need; a dict with a subclass key or non-builtin value is still not one. Tests:
`tests/contracts/test_preimage_json.py`.
`isinstance(HOLE, T)` (`visitor.py::visit_isinstance`; only the object may be unknown) with `Truth`/`Eq` of a bool demanded.
`T` is a literal builtin class, a (nested) tuple of them or an `X | Y` union, drawn from `object, bool, int, float, str, bytes,
list, tuple, dict, NoneType`. The proposals are the program's own pool literals first, then a small canonical set of exact
builtin values per class (`visitor.py::_ISINSTANCE_CANONICAL`, simplest first); each is FORWARD-CHECKED with `isinstance` itself,
so `True` is an instance of `int` and is never offered as a non-int. They are canonical values, not every instance, and
`Truth(False)` for `object` is empty. Any other class (an ABC, a user class, `set`, `complex`, `type`, an exception) is a named
refusal: building an instance runs `__init__` / `__instancecheck__` / `__subclasshook__`, which is user code. A class defined
in the same module and constructible without arguments is NOT built here; `solve/condition.py` solves such a guard at the source
level. Tests: `tests/contracts/test_preimage_isinstance.py`.
`str.encode(HOLE, codec, errors)` under a LOSSY handler (`visitor.py::visit_str_encode`, `_lossy_encode_sources`). Besides the
strict decoding of the observation, which is always proposed first, `ignore` also proposes the observation with one character
the codec cannot encode INSERTED (start, end, then the middle positions up to 32; the canonical set is `_UNENCODABLE`: a CJK and a
Cyrillic letter, `ss`, an accented letter, `EUR`, a lone surrogate, an emoji, filtered to those the codec really cannot encode, so
`latin-1` never gets an accented letter and `utf-8` only the lone surrogate); `replace` turns occurrences of the codec's marker
(`?`) into such a character (all, then each); `xmlcharrefreplace` / `backslashreplace` decode their escapes back to characters.
Every proposal is FORWARD-CHECKED by `str.encode` itself, which is what keeps this sound; the bytes do not say which characters
were dropped, so these are SOME preimages (one dropped character), not all, and a chain that needs several, or a particular one, is
not solved here. `rules_strformat.py::native_limit` now also admits the builtin handlers `xmlcharrefreplace` and `backslashreplace`
(registered or unknown handler names still give no candidate). Tests: `tests/contracts/test_preimage_str_encode_lossy.py`.
For `re.escape` with an `Eq` demand on a builtin `str` or `bytes` result, the visitor
walks the actual stdlib escape table and derives one canonical preimage. Escaped
newlines, all byte values, Unicode and arbitrary lengths use the same linear
walk. The one builtin forward check has no regex expansion or format-size window.
Bytearray, memoryview and inherited string inputs have canonical builtin witnesses;
custom `translate` results remain a named refusal. The candidate is not an exact
input-domain transfer, and an empty iterator is not an absence proof.

`solve/preimage/regex_members.py` is a deterministic member generator over `re._parser` trees. It has three modes:
* shortest first, with every branch keeping its simplest member;
* exact total length, or exact segment lengths between a literal (`split="@", segments=(64, 255)`);
* flag-differential members, built to exercise the one node a DOTALL/IGNORECASE/MULTILINE/ASCII difference changes.

It refuses backreferences, conditional groups, lookbehind and `re.LOCALE` by name, and it masks `re.DEBUG` (never
compiled with). It does not use Hypothesis' random `from_regex`; the #1170 Phase A record shows that a random sample's
length decided the K5 kills.

`solve/preimage/__init__.py::chain_preimages` composes visits backward along a straight-line chain, for example
`head.startswith('"')` ← `replace('\\ ', '')` ← `[0]` ← `split('@')`. It uses the transfers while they exist and the
candidates after that, and re-checks every yielded entry value forward through the whole chain.

Consumers:
* the driver's flipper for guards over derived locals (`driver.py::_flip_preimage`: built, #1170 PR2;
  docs/DESIGN.md §8);
* `_symexec` concretization (sat only, never unsat/unreached): built with the path's model value (#1170 PR4, #1189);
  the preimage-guided choice of the value is ruled, not built;
* hybrid whole-draw-vector seeds and deterministic regex members (built, #1188; docs/SV.md `sv/hybrid.py`);
* `tharkun.inputs`' `preimage` rung (built, #1170 PR6; docs/INPUTS.md "The ladder"):
  `driver.py::preimage_candidates` walks the paths to each `return` back to the parameters, and every row is
  replayed in a fresh child;
* facts as the candidates' prior filter (#1170 PR5): `solve/preimage/facts.py::prior_from_fact` states absint's value
  facts (`solve/values.py` `StrFact`/`IntFact`) and the fact carrier (`solve/fact_env.py::VarFacts`) as demands every
  value of the fact satisfies (a superset: a filter, never a claim). The live consumer is the drive: its own pins and
  bounds (`context_facts`, `param_bounds`) filter `_flip_preimage`'s proposals. absint's site facts reach no preimage
  hole yet: ruled, not built.
* rule-keyed hints (#1170 PR5): a committed preimage flip publishes one `rule_fact` per rule its walk went through,
  into the drive's `KnowledgeBase` (`driver.py::_publish_preimage_hints`, `solve/preimage/facts.py::rule_fact`). The
  key is `<qualname>@<file>:<line>#preimage`, it is never certified, and it has no reader yet. Registering the kind in
  `sv/contracts.py::KINDS` waits for an SV component that writes it.

### Files and how they relate

- `src/tharkun/solve/preimage/__init__.py`: facade (`preimages`, `chain_preimages`, `Step`). It imports
  `solve/preimage/want.py` eagerly, and `solve/preimage/visitor.py` and `solve/contracts/rules.py` lazily.
- `src/tharkun/solve/preimage/want.py`: the demand vocabulary. `Member.holds`/`Member.seeds` and `All.seeds` import
  `solve/preimage/regex_members.py` lazily.
- `src/tharkun/solve/preimage/regex_members.py`: regex members and the bounded native check. It imports only stdlib
  (`re`, `re._parser`, `re._constants`).
- `src/tharkun/solve/preimage/facts.py`: priors from facts (`prior_from_fact`, `prior_from_pin`, `prior_from_bounds`)
  and the `rule_fact` hint record. It imports `solve/preimage/want.py`, and lazily `solve/fact_env.py`,
  `solve/values.py` and `solve/contracts/rules.py`.
- `src/tharkun/solve/preimage/visitor.py`: `PreimageVisitor`, `PreimageCtx`, `Preimage`. It imports
  `solve/contracts/rules.py`, `solve/preimage/want.py`, `solve/preimage/regex_members.py`, and lazily
  `solve/contracts/adapters/str_shapes.py` and `solve/contracts/rules_strformat.py`.
- Tests: `tests/contracts/test_preimage_rules.py` (soundness, exact transfers, refusals, chains),
  `tests/contracts/test_preimage_audit.py`, `tests/contracts/test_preimage_facts.py`, `tests/test_regex_members.py`.


## Callable builtins (#1443, CPython 3.14.4)

`rules_builtins_missing.py` fills the callable-name gap for 26 builtins with ordinary
`Rule`/`Contract` entries. `tests/contracts/test_builtins_missing.py` checks their
native returns, call arities, validation boundaries, all seven property exceptions,
non-property hook exceptions, lazy consumption and effects on **CPython 3.14.4**.
Sources cite Python 3.14 documentation and the CPython `v3.14.4` source (3.14 branch).
`map(strict=...)` is new in 3.14; complex-valued real/imag constructor arguments are
deprecated in 3.14. These claims do not assert older-interpreter behavior.

The contract describes the CALL. `map`/`filter` acquire iterators now, while their
functions and next steps run later. `anext` invokes `__anext__` now; without a default
it returns that hook's result without checking awaitability. `classmethod`,
`staticmethod` and `property` may read arbitrary metadata descriptors now, even
though they do not invoke the wrapped function. Their metadata `AttributeError`
is suppressed, as is an attribute lookup `AttributeError` in `hasattr`.

`next`/`anext` may advance their receiver, and `memoryview` exports/aliases a buffer
(which can inhibit resizing). These effects are explicit. Recipe admission follows
`adapters/avengers.py::stdlib_admission`'s existing native-dispatch effect policy;
it does not turn `may_execute_callback=True` into a whole-operation purity proof.
`hash` depends on the process hash seed and potentially object identity; `id` on
allocation addresses (and invokes an audit hook); `super`'s zero-argument form reads
caller frames. All three declare `clock_or_env=True`, `deterministic=False`, and
remain refused as recipe callables. `super` is intentionally contracted as the
union of its forms, without a new shape-specific admission exception.

The other 18 requested names remain uncontracted with individual reasons in
`rules_builtins_missing.py`'s module docstring. They execute code, expose caller
namespaces, mutate arbitrary attributes, perform interactive I/O, or depend on
site/process configuration. Refusal tests pin every one of these names. No new
abstract-interpreter or SV transfer functions accompany this inventory: new rule
kinds retain the existing named refusal when a visitor has no implementation.

Truth conversion is itself a callback boundary: `bool` can invoke `__index__` on
an object returned by `__len__`; `issubclass` truth-tests a custom `__subclasscheck__`
result. Missing `__bases__` metadata is suppressed and becomes a native TypeError,
while an unsuppressed class-metadata exception escapes. Native tests cover these
indirect hook paths as well as exceptions raised directly by the primary hook.

On main, recipe references reach these bare-name contracts through the separately
landed #1442 `builtins.X` resolution fix (PR #62). #1443 extends that safety test
with every new admissible callable and every intentional refusal; it does not
copy or reimplement the main-line lookup fix.

## Protocols and module callable aliases (#1455)

A shared protocol contract does not automatically give operator.sub or another
module callable a registry entry. Each callable name needs its own contract; a
parametrized Rule family can retain the shared dispatch structure. Keep
comparisons' arbitrary return values, iterator consumption and in-place mutation
explicit. Identity predicates are callback-free, while methodcaller is a factory
whose result performs lookup/call later. See `rules_operator_more.py` and its
native CPython 3.14.4 tests.

## HTML utilities (#1455, CPython 3.14.4)

rules_html.py gives both public html callables their own contracts. Sources are
the Python 3.14 html documentation and CPython 3.14 Lib/html/__init__.py; native
tests run on 3.14.4. Open hooks require object returns and unknown raises. See
REPORT.md for native effects and ../ABSINT.md for the admitted return transfer.

## Textwrap utility contracts (#1455, CPython 3.14.4)

The five utility functions in rules_textwrap.py have separate callable names,
unknown seven-class raises and explicit callback effects. Wrap's open return is
list[object], since custom indentation addition can return arbitrary line values;
the four outer-join helpers return str. Configured TextWrapper instances and their two public methods now have their
own contracts and a mutable receiver model (batch 11). Python 3.14
documentation and the 3.14 source branch are cited; native tests use 3.14.4.

### Exception class and arguments (#1455)

`ConstantArgsRaiseClaim` extends `RaiseClaim` with an exact tuple of literal string
`args`. Its `tested` flag covers both the condition and the payload; `never` and
`unknown` cannot carry this claim. Existing `RaiseClaim` records remain class-only
and do not acquire an argument claim. The seven property classes are unchanged.
Because this is a claim subtype, unchanged contracts keep their canonical frozen
serialization and digests; only a contract using the new field gains that field.

The plain exception constructor rules cover canonical `BaseException`, `Exception`
and the eleven simple builtin subclasses listed in
`solve/contracts/rules_exceptions.py::PLAIN_EXCEPTION_NAMES`. Their ReturnSpec
states that `.args` is the positional tuple, retaining the original references;
construction does not format or otherwise inspect those arguments. Any keyword
raises `TypeError` with `(ClassName + '() takes no keyword arguments',)`. Source
subclass initialization can execute arbitrary code and is analyzed separately.
These claims cite the [Python 3.14 exception documentation](https://docs.python.org/3.14/library/exceptions.html#BaseException)
and [CPython's 3.14 exception implementation](https://github.com/python/cpython/blob/3.14/Objects/exceptions.c),
with boundary and adversarial tests executed on CPython 3.14.4.

The twelve `NumericDivision` rules specify the builtin int/float normal and
reflected `/`, `//` and `%` slots. `ZeroDivisionError` carries the tested literal
`('division by zero',)` on CPython 3.14.4, for every one of these slots. The zero
condition applies only after valid native receiver binding and supported numeric
operand conversion; huge int-to-float conversion can instead raise OverflowError.
A native slot with an unsupported other type returns NotImplemented. Numeric
subclass overrides are not called by an explicit builtin slot descriptor; ordinary
operator dispatch can call them and remains governed by protocol rules. See the
[Python 3.14 numeric types documentation](https://docs.python.org/3.14/library/stdtypes.html#numeric-types-int-float-complex)
and CPython's 3.14 [integer](https://github.com/python/cpython/blob/3.14/Objects/longobject.c)
and [float](https://github.com/python/cpython/blob/3.14/Objects/floatobject.c) implementations.

### Scoped non-property raises (#1455)

`Contract.other_raises` supplements the fixed seven property-class `raises` with
`OtherRaiseClaim(exc, condition, args_template, tested, case)` entries. These are
scoped known cases, not a closed list of every possible callback exception;
`may_raise_other` remains the open-world flag. Argument templates are tuples of
Python format strings. A tested contract requires its known cases to be tested.
The canonical frozen encoding omits an empty `other_raises` extension, preserving
all legacy contract bytes; nonempty cases participate fully in the digest.

`int` records exact-str cases for an invalid primitive base, invalid lexical
syntax and the process integer digit limit. `Int.literal_valid` describes the
syntax independently of that process limit: Unicode decimal digits normalize to
ASCII, non-ASCII Unicode whitespace normalizes to spaces, the six ASCII whitespace
characters may surround the number, one ASCII sign is optional, underscores occur
between digits or once after an accepted base prefix, and base zero disallows a
nonzero leading-zero decimal number. ASCII separators U+001C–U+001F and Unicode
non-decimal numeric symbols are invalid. Explicit base is zero or 2–36.

The invalid-literal args use the requested numeric base and the first 200 codepoints
of `repr(original_s)`, even if this truncates its closing quote. Invalid-base args
are the single literal `int() base must be >= 2 and <= 36, or 0`. A valid non-power-
of-two conversion whose digit count exceeds the enabled limit uses the tested
limit/count diagnostic template. Main `int` effects declare that process setting;
short finite primitive-string transfers can establish that no enabled limit is
observed. Input/base coercions and arbitrary callback raises remain open.
Sources: [Python 3.14 int](https://docs.python.org/3.14/library/functions.html#int),
[integer conversion limits](https://docs.python.org/3.14/library/stdtypes.html#integer-string-conversion-length-limitation),
and [CPython 3.14 longobject.c](https://github.com/python/cpython/blob/3.14/Objects/longobject.c).
All boundary evidence is native CPython 3.14.4.

### Scoped property-exception payloads (#1455)

`ScopedArgsRaiseClaim` keeps the ordinary property-class condition and adds tested
`RaiseArgsCase` payload relations for explicit native subsets. A case's tuple
contains literal strings or `ArgumentRef(position)` values, with receiver at
position zero. The reference preserves the original argument, including its type
and object identity; it is not formatted. Other callback payloads stay unspecified.
These are fields of the same seven property claims, not another contract registry.

Dictionary native misses carry `(key,)` after successful hashing/equality.
A dict subclass can override `__missing__`, and key hooks or a defaultdict factory
can raise KeyError with different args; those paths are outside the native payload
scope. Factory-None defaultdict lookup has the same key-miss payload. The distinct
native `defaultdict.__missing__` hook, when its factory is None, raises KeyError(key)
without hashing or testing membership, even for a present or unhashable key.

CPython 3.14.4's non-None-factory missing hook calls the factory and uses native
setdefault-style insertion: a present or reentrantly inserted value wins and is
returned. Native insertion bypasses a subclass `__setitem__`. These boundaries are
pinned natively, not inferred from another interpreter. See the
[Python 3.14 mapping documentation](https://docs.python.org/3.14/library/stdtypes.html#dict),
[defaultdict documentation](https://docs.python.org/3.14/library/collections.html#collections.defaultdict.__missing__),
and CPython's 3.14 [dict](https://github.com/python/cpython/blob/3.14/Objects/dictobject.c)
and [collections](https://github.com/python/cpython/blob/3.14/Modules/_collectionsmodule.c) implementations.

## Sequence IndexError payload scopes (1455)

list/tuple/str `__getitem__` now use tested `ScopedArgsRaiseClaim` cases. For
exact builtin receivers and primitive int/bool keys, the native `native-bounds`
case requires a Py_ssize_t-representable index outside `-len <= i < len` and
stores one string: `list index out of range`, `tuple index out of range`, or
`string index out of range`. Negative indices down to `-len` are valid. The
`oversized-int` case requires an exact int outside Py_ssize_t range and stores
`cannot fit 'int' into an index-sized integer`, before length adjustment.
Slices clamp their bounds and do not take these cases. A key's `__index__`
may raise arbitrary payloads; overflowing custom index types use their own
type name, so neither belongs to the exact-int payload scope.

Sources: [Python 3.14 sequence operations](https://docs.python.org/3.14/library/stdtypes.html#common-sequence-operations)
and the rules' explicit CPython 3.14 source-branch links. Native tests run on
CPython 3.14.4, including ssize limits, Unicode codepoint lengths and callbacks.

## string.capwords (1455)

The word helper has a Rule, native3.14.4 tests and an actual absint visitor.
Exact builtin str text and str-or-None separators have a str result. Open
split/iteration/sep truth/join callbacks are declared, and a truthy custom
separator may return any object from join. Native join attribute lookup occurs
before split argument evaluation: truthy sep1 has AttributeError; falsy sep0
reaches builtin split and has TypeError. All seven callback property classes
retain their arbitrary args. An exact empty-str separator has a tested scoped
ValueError case with args `('empty separator',)`. Formatter and Template
receiver APIs remain documented omissions for a later batch.

Sources: [Python 3.14 string helper](https://docs.python.org/3.14/library/string.html#string.capwords)
and [CPython 3.14 string package](https://github.com/python/cpython/blob/3.14/Lib/string/__init__.py).

## Heapq native sibling family (1455)

Seven missing callables now have individual contracts via one Rule family:
heapify_max/heappush_max/heappop_max, heappushpop/heappushpop_max and
heapreplace/heapreplace_max. Each declares receiver mutation and open
comparison/truth callbacks, including possible partial mutation on failure.
Bindings are positional-only native list calls. Needed incompatible builtin
element comparisons may raise TypeError; arbitrary comparison callbacks can
raise every property class with arbitrary args. Native empty pop/replace
(IndexError) scopes store `('index out of range',)`; callback payloads remain
open. Existing heappop gets the same scoped empty-list payload. Pop/pushpop/
replace return original items, with an extreme-item guarantee conditional on
the input already satisfying heap order. merge remains omitted because its
lazy consumption needs separate iterator state.

Native controls use CPython3.14.4; sources are [Python3.14 heapq](https://docs.python.org/3.14/library/heapq.html)
and [CPython3.14 native heap implementation](https://github.com/python/cpython/blob/3.14/Modules/_heapqmodule.c).

The heap Rule family stores one immutable Contract per qualname, preserving
registry object identity. The existing heapify/heappush/heappop generic adapter
remains the fallback for unsupported call shapes or element abstractions; its
applicability and havoc checks are retained. Primitive integer comparisons
prove absence of callback exceptions; a native hostile __lt__ control retains
the older per-class Unknown-vs-safe distinction.

## Statistics numeric query family (1455)

Seventeen missing summary/relation functions have one shared Rule family with
one immutable Contract per qualname: mean/fmean/geometric_mean/harmonic_mean;
median/median_low/median_high/median_grouped; mode/multimode/quantiles;
pstdev/pvariance/stdev/variance; covariance/correlation. Native3.14.4 tests
cover each result, empty-data exception class+args, arbitrary iteration-hook
args, weighted calls, tie order and a real Decimal context dependency.

All open-operand property raises remain explicitly unknown; the overall
contract's tested flag covers return evidence, not an invented never claim.
Whole-family effects conservatively declare callbacks/context dependence, so
recipe admission remains clock-or-env-refused. The absint primitive transfer
proves its own context-independent finite alternatives. NormalDist,
LinearRegression/linear_regression and kde/kde_random await receiver, result
field or callable state models. StatisticsError construction/qualified raise
transfers remain named refusals; no builtin ValueError object is substituted.

Sources: [Python3.14 statistics](https://docs.python.org/3.14/library/statistics.html)
and [CPython3.14 implementation](https://github.com/python/cpython/blob/3.14/Lib/statistics.py).

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

## Array constructor and typed storage (1455 batch23)

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

Both lines on CPython3.14.4: native114, focused constructor16, shared
storage36 and existing28 pass; standard tooling generated array.array.json.
The shared conversion fixes mixed numeric joins, rounds float32 interval
bounds and preserves finite rounding immediately below the native overflow
threshold. Constructor and shared storage are this batch; byte values and
symbolic construction are separate follow-up work. Expanded importer checks
cover the combined candidate before readiness.
Sources: [Python3.14 array](https://docs.python.org/3.14/library/array.html#array.array)
and [CPython3.14 array implementation](https://github.com/python/cpython/blob/3.14/Modules/arraymodule.c).
