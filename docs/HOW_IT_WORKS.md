# How Tharkun Works — Deep Dive with Code Examples

> *Module paths in this document predate 0.2.0. The current layout, the layer rule and the move map (every old path is a permanent shim) are in [LAYERS.md](LAYERS.md).*

Tharkun solves the hardest problem in automated test generation: **figuring out what inputs make each branch reachable.** Given any Python `if`/`while`/`for` guard, tharkun determines concrete satisfying values using 9 techniques applied in sequence.

## The Problem

Consider this function:

```python
def process_order(order, user, config):
    if user.is_verified and order.total > config.min_amount and order.currency in ("USD", "EUR"):
        apply_discount(order)
```

To test the `apply_discount` path, you need:
- `user.is_verified` → True
- `order.total` → some value > `config.min_amount`
- `order.currency` → "USD" or "EUR"

But what *is* `config.min_amount`? What type is `order`? How do you construct a `user` with `is_verified = True`?

**Existing tools** (Pynguin, Hypothesis) use random/evolutionary search. They guess thousands of inputs hoping to stumble into the right path. On compound guards like this, they rarely succeed.

**Tharkun** uses **deep per-branch analysis** — static analysis, AST decomposition, debugger-level runtime capture, and symbolic execution — to *compute* the answer.

## The 9-Step Chain

For each uncovered branch, `solve_branch()` applies techniques in order. The first technique that succeeds becomes the answer.

### Step 1: Static Condition Solver

The core engine. Parses the guard expression as AST, infers variable types from 10 rules, then computes concrete values.

```python
from tharkun import solve_condition

# Given: if x > 100 and x < 200 and x != 150
sol = solve_condition("module.py", 42)

print(sol.guard_source)        # "x > 100 and x < 200 and x != 150"
print(sol.strategy)            # "direct"
print(sol.satisfying_inputs)   # [{"x": 101}]  ← integer range intersection
```

**How it infers types** (priority order):
```python
# Rule 1: isinstance → exact type
if isinstance(items, list):      # items: list

# Rule 2: Comparison with literal → type from constant
if x > 5:                        # x: int (from 5)

# Rule 3: Method call → type from method name
if s.startswith("prefix"):       # s: str (startswith is a str method)

# Rule 4: Attribute comparison → object with that attribute
if c.kind == "method":           # c: SimpleNamespace(kind="method")

# Rule 5: hasattr → object with that attribute
if hasattr(obj, "ready"):        # obj: SimpleNamespace(ready=True)
```

**How it generates values:**
```python
# Integer range intersection
if x > 0 and x < 10 and x != 5:  →  x = 1  (intersect ranges, exclude 5)

# String synthesis
if s.startswith("pre") and s.endswith("suf"):  →  s = "presuf"

# Len constraints
if len(password) >= 8:  →  password = "xxxxxxxx"

# Regex inversion (via Hypothesis)
if re.match(r"^\d{3}-\d{2}-\d{4}$", ssn):  →  ssn = "123-45-6789"

# Or-branch exploration (one input per branch)
if x < 0 or isinstance(y, str) or z is None:
  →  [{"x": -1}, {"y": "test"}, {"z": None}]  ← three separate tests
```

**Solve rate: ~70% of all guards in production Python code.**

### Step 2: Guard Decomposer

When the guard calls other functions, traces into them recursively.

```python
from tharkun import decompose_guard

# Given: if validate(data) and len(data) > 5
result = decompose_guard("module.py", 42)

# Returns a tree:
# validate(data) [call]
#   └── data.is_valid() and len(data) > 5 [and]
#        ├── data.is_valid() [call → solvable]
#        └── len(data) > 5 [leaf → solvable]

print(f"{result.solvable_leaves}/{result.total_leaves} solvable")
```

Handles:
- `BoolOp(And/Or)` decomposition
- `UnaryOp(Not)` including negated chained comparisons
- Cross-module function resolution (follows imports)
- Guard clause extraction (`if not X: return False; return expr`)
- Cycle detection + depth limiting

### Step 3: Attribute Tracer (self.xxx Guards)

When the guard checks `self.state == "connected"`, finds where the attribute is assigned.

```python
from tharkun.attr_tracer import analyze_attr_blocker

# Given: if self.state == "connected"
results = analyze_attr_blocker("connection.py", guard_expr)

# Returns:
# AttrTraceResult(
#   target=AttrTarget(base="self", chain=("state",), full="self.state"),
#   assignments=[
#     AttrAssignment(method="connect", rhs_source="'connected'", is_init=False),
#     AttrAssignment(method="__init__", rhs_source="'disconnected'", is_init=True),
#   ]
# )
```

The state_sequence solver (Step 7) uses this to determine: **"to reach `self.state == 'connected'`, call `connect()` first."**

### Step 4: Mock Generator

Detects external calls that need mocking (HTTP, filesystem, database, checksums).

```python
from tharkun import detect_externals, generate_patches

externals = detect_externals("service.py", "fetch_data")
patches = generate_patches(externals)

# Generates:
# @patch("requests.get", return_value=Mock(status_code=200, json=Mock(return_value={})))
# def test_fetch_data(mock_requests_get):
#     ...
```

### Step 5: Regex Inversion

When the guard is `re.match(pattern, var)`, generates a matching string using `hypothesis.from_regex()`.

### Step 6: Runtime Snapshot + Re-Solve (The Debugger Trick)

**This is tharkun's strategic advantage.** When static analysis fails, tharkun executes the code with `sys.settrace` and captures `frame.f_locals` at the exact branch line — the same data a debugger sees.

```python
# The guard:
def process(data, threshold, multiplier):
    adjusted = threshold * multiplier
    if data.score > adjusted and data.category in VALID_CATEGORIES:
        ...
```

**Without snapshots:** The solver sees `data.score > adjusted` — but what is `adjusted`? It's a computed value, not a parameter. Static analysis can't determine its value.

**With snapshots:** The tracer captures `f_locals` at the branch line:
```python
{
    "data": <Data object>,
    "threshold": 50,
    "multiplier": 1.5,
    "adjusted": 75.0,          # ← computed at runtime
    "VALID_CATEGORIES": {"A", "B", "C"},
}
```

Now `solve_with_snapshot()` can:
1. **Substitute constants**: `data.score > adjusted` becomes `data.score > 75.0`
2. **Simplify compound AND**: If `data.category in {"A", "B", "C"}` is already True, remove that clause
3. **Re-solve the reduced expression**: Just need `data.score > 75.0` → `data.score = 76.0`

```python
from tharkun import solve_with_snapshot

sol = solve_with_snapshot("module.py", 42, {
    "threshold": 50,
    "multiplier": 1.5,
    "adjusted": 75.0,
    "VALID_CATEGORIES": {"A", "B", "C"},
})

print(sol.strategy)            # "snapshot_simplified"
print(sol.satisfying_inputs)   # [{"data": SimpleNamespace(score=76.0, category="A"), "adjusted": 75.0}]
```

**The key insight:** The debugger knows what variables *are* at runtime. Using that to constrain what inputs *should be* closes the feedback loop between execution and solving.

### Step 7: State Sequence Solver

For `self.xxx` guards, determines which methods must be called first.

```python
# Given:
class Connection:
    def __init__(self):
        self.state = "disconnected"

    def connect(self):
        self.state = "connected"

    def send(self, data):
        if self.state == "connected":  # ← the guard
            ...

# Tharkun generates:
# obj = Connection()
# obj.connect()          ← setup method (sets self.state = "connected")
# obj.send(data="test")  ← target method (now the guard passes)
```

### Step 9: Hypothesis Property-Based Search (Optional)

Last resort: uses `hypothesis.find(strategy, predicate)` where the predicate runs the function with the tracer and checks if the target branch was hit.

## End-to-End: From Guard to Test

```python
from tharkun import solve_uncovered_branches
from tharkun.emit import emit_branch_tests

# Solve all branches in a module
coverage = solve_uncovered_branches("service.py")
print(f"{coverage.solved_branches}/{coverage.total_branches} branches solved")

# Generate runnable pytest file
test_source = emit_branch_tests(coverage)
Path("test_service_generated.py").write_text(test_source)
```

The generated test file:
```python
"""Auto-generated branch coverage tests for service.py."""

import unittest
from unittest.mock import Mock, patch
from service import *

class TestGeneratedService(unittest.TestCase):

    @patch("requests.get", return_value=Mock(status_code=200, json=Mock(return_value={})))
    def test_fetch_data_line42_0(self, mock_requests_get):
        """Branch at line 42: response.status_code == 200"""
        result = fetch_data(url='http://test')

    def test_process_order_line15_1(self):
        """Branch at line 15: total > 100"""
        result = process_order(total=101, currency='USD')

    def test_connection_send_line30_seq_2(self):
        """State sequence for line 30: self.state == 'connected'"""
        obj = Connection()
        obj.connect()
        obj.send(data='test')
```

## Accuracy Improvements (P2)

### RecordingProxy — Infer Types from Usage

When a parameter has no type hints and the solver falls back to `Any`, tharkun can pass a `RecordingProxy` that records what the function does with the argument:

```python
from tharkun.proxy import RecordingProxy, infer_type_from_usage

proxy = RecordingProxy()
try:
    unknown_function(proxy)
except:
    pass

inferred = infer_type_from_usage(proxy)
# If the function called proxy.startswith(), proxy.upper() → inferred = "str"
# If the function called proxy.append(), proxy.sort() → inferred = "list"
# If the function called proxy.keys(), proxy.get() → inferred = "dict"
```

### Real Object Construction

When `isinstance(x, MyClass)` resolves to a class in the codebase, tharkun reads `__init__` from the AST and builds a real instance:

```python
from tharkun.constructor import construct_instance

# Instead of SimpleNamespace(), constructs:
obj = construct_instance("models.py", "UserProfile")
# → UserProfile(name="", age=0, verified=False)  ← real instance with synthesized args
```

### Cross-Function Type Propagation

When `f(x)` calls `g(x)` and `g` has annotations, tharkun propagates types back:

```python
def typed_callee(x: int, name: str) -> bool:
    return x > 0

def untyped_caller(a, b):    # a, b have no annotations
    return typed_callee(a, b)  # but we know a:int, b:str from the callee
```

### Exception-Driven Feedback

When execution with generated inputs raises `TypeError("expected str, got int")`, tharkun parses the error and corrects the input automatically:

```python
# First attempt: greet(name=42) → TypeError: can only concatenate str
# Auto-correction: greet(name="test") → works
```

## Benchmarks

Self-benchmark on tharkun's own source (19 modules, 858 branches):

```
$ tharkun bench
Module                      Branches     Solved     Rate     Time
----------------------------------------------------------------------
condition_solver                 272        268      99%   27.52s
branch_solver                     88         81      92%   27.57s
guard_decomposer                  69         60      87%    7.05s
mock_gen                          51         50      98%    3.40s
...
----------------------------------------------------------------------
TOTAL                            858        809    94.3%   95.75s
```

**94.3% solve rate** — static analysis alone, no random search.

## CLI

```bash
# Solve a single guard
tharkun solve module.py 42

# Solve with specific branch polarity
tharkun solve module.py 42 --branch else

# Decompose a compound guard
tharkun decompose module.py 42

# Analyze with runtime snapshots
tharkun analyze module.py 42

# Solve ALL branches aggressively
tharkun branches module.py

# Generate pytest tests
tharkun emit module.py -o test_generated.py

# Detect external calls needing mocks
tharkun mocks module.py

# Benchmark
tharkun bench                      # self-benchmark
tharkun bench module.py other.py   # benchmark specific files
```

---

## Structured Outputs

As of `0a93f86`, tharkun produces richer structured outputs on every solve result.

### BranchSolution Fields

`solve_branch()` returns a `BranchSolution`:

```python
from tharkun import solve_branch

sol = solve_branch("module.py", 42, desired_branch="then")

sol.guard_source            # str: "x > 100 and x < 200"
sol.lineno                  # int: 42
sol.desired_branch          # str: "then" or "else"
sol.solved                  # bool: True / False

# New structured fields:
sol.best_inputs             # dict | None: {"x": 101}
sol.best_inputs_constraints # dict | None: {"x": {"type": "int", "gte": 101, "lte": 199}}
sol.blocker_info            # dict | None: structured failure reason
```

**`best_inputs_constraints`** — a typed constraint dictionary describing what *kind* of value is needed, beyond just the specific example:

```python
# For: if len(password) >= 8 and password.isalnum()
best_inputs_constraints = {
    "password": {
        "type": "str",
        "min_len": 8,
        "pattern": "alphanumeric"
    }
}
```

**`blocker_info`** — present when `solved=False`, explaining *why* and suggesting a fix:

```python
# For: if config.rate_limit > 0  (config is an external object)
blocker_info = {
    "kind": "external_attribute",
    "detail": "config.rate_limit depends on an external configuration object",
    "suggestion": "Pass a mock or stub: config = SimpleNamespace(rate_limit=1)",
    "patch_target": "config"
}
```

### ConditionSolution Fields

`solve_condition()` returns a `ConditionSolution`:

```python
from tharkun import solve_condition

sol = solve_condition("module.py", 42)

sol.guard_source      # str: the guard expression
sol.satisfying_inputs # list[dict]: [{"x": 101}]
sol.strategy          # str: "direct", "snapshot_simplified", etc.

# New structured fields:
sol.explanation       # str: "integer range intersection: 101 satisfies x > 100 and x < 200"
sol.input_constraints # dict | None: {"x": {"type": "int", "gte": 101, "lte": 199}}
```

**Helper calls in a guard** (`solve/condition.py::_inline_pure_helpers`). A call to a function whose body is one `return <expr>`
(a docstring allowed), defined in the same module or imported from a sibling by a relative import (`from .validation import
is_string, is_full_string as full`), is replaced by that expression with the arguments substituted before the guard is solved,
so `not is_full_string(s)` is solved as `not isinstance(s, str) or s.strip() == ''` and `guard_source` shows the inlined form.
Before this the call was a free variable and the answer (`{'s': 0, 'is_full_string': 0}`) did not satisfy the guard.
Conditions: positional parameters only (no defaults, `*args`, decorators), exact arity with no keywords, `Name`/`Constant`
arguments, no lambda/comprehension/walrus in the helper, no recursion, depth 4 and 400 nodes, and a callee name that the
enclosing function rebinds is left alone; any other helper stays an opaque call. Two companions keep the answer honest:
`name.method(literals) == "literal"` (`s.strip() == ''`) is solved by the str preimage engine instead of the type default,
and an `or` guard keeps only the branch witnesses that satisfy the WHOLE guard when it can be evaluated and some do.
Tests: `tests/test_condition_helper_inlining.py`.

**Regex-membership guards** (`solve/condition.py::_regex_guard_witness`, `_resolve_compiled_pattern`). For a guard such as
`is_full_string(s) and JSON_WRAPPER_RE.match(s) is not None`, the compiled pattern is read from its definition, including
through `from .mod import PAT` / `from .mod import *` in a sibling file (AST only, one level). The witness is built by the
regex-membership preimage (`regex_members.members`, deterministic) and, when the branch body then calls `json.loads(s)`, the
texts the `json.loads` preimage writes for a few JSON values (`[]`, `{}`, `[0]`, ...) are tried FIRST, so the answer matches the
pattern and parses (`'[]'` for `validation.py` 359). Every candidate is checked by evaluating the whole guard with the pattern
bound; none that fails is returned. When the pattern is one the member generator handles and no member satisfies the guard
(`WORD_RE.match(s) is not None and len(s) > 50`), the result is `strategy="none"` with no inputs, not a type default or a
length-only guess. Patterns it refuses (bytes, backreferences, lookbehind) keep the previous route.
When the body parses with `json.loads`, `satisfying_inputs` lists the parse witness first and then a second member of the
pattern for which `json.loads` raises (`'{]'` for `validation.py` 359): it reaches the `except (TypeError, ValueError,
OverflowError)` handler (362) and its `pass` (363). Both are guard-verified; a consumer that needs the handler arm takes the second.
Tests: `tests/test_condition_regex_json.py`.

**Loop-body guards over local aliases** (`solve/loop_guard.py::solve_loop_guard`, called from `solve/condition.py::_loop_guard_solution`
before the ordinary route; `_loop_guard=False` is the recursion guard). A guard such as `head_char != tail_char` (`is_palindrome`,
`validation.py` 491) names locals, not parameters. The aliases are substituted back: straight-line `name = expr` assignments before
the loop (`string_len = len(input_string)`) and in the loop body (`head_char = input_string[index]`,
`tail_char = input_string[string_len - index - 1]`), `if flag: name = expr` rebinds in both arms (`if ignore_case: head_char =
head_char.lower()`), pre-loop `if C: return` guards required False, and `for k in range(bound)` entered. That leaves an
index-affine relation `E1(P[p]) == / != E2(P[q])` over one parameter `P`, `E1`, `E2` chains of zero-argument str methods and
`p, q` affine in `k` and `len(P)`. It is solved, not searched: z3 gives the minimal length with in-bounds, non-wrapping `p`,
`q` (distinct positions for a mismatch; the same or distinct positions for a match), and the characters come from the
`solve/preimage` str-method preimages of the value the other side produces (a mismatch falls back to a neighbouring character
whose image differs when the next code point has no preimage, as under `upper`). Each witness is a full parameter dict, checked
by evaluating the pre-loop path guards, the chosen rebinding arms, the loop entry and the aliased guard (the helper-inlined
evaluator); a witness that fails any of them is dropped. For `validation.py`: 483 then -> `input_string='a'`; 487 then ->
`ignore_case=True` (reaches 488/489); 491 then -> `'ab'`, `'aB'` (with `ignore_case`); 491 else -> `'a'`, `'aa'`, `'aA'` (with
`ignore_case`). Domain: any function whose guard is in a `for k in range(<affine in len(P)>)` body with that alias shape; length
is the optimiser's minimum for the index constraints, never a fixed size. Not solved (named): iterations after the first (a
mismatch after matching pairs), the `if flag: P = rewrite(P)` arm (`ignore_spaces=True`: needs the inverse of the rewrite), the
loop's empty arm, aliases through calls/slices/walrus. Anything outside the shape returns to the previous route.
Tests: `tests/test_condition_loop_guard.py`.

**Path-sensitive guards over copy-propagated assignments** (`solve/path_guard.py::solve_path_guard`, called from
`solve/condition.py::_path_guard_solution` after the loop-guard route; `_loop_guard=False` disables both). A guard in a function
names locals (`key = args[0]`, `m = PAT.match(key)`, `version, desc, *_ = text.split(sep="\n", maxsplit=1) + [None, None]`) and
is reached only under its enclosing `if` arms and after the early `if C: return/raise`. The path from the entry is replayed
symbolically: straight-line and tuple/starred assignments are substituted forward (an unpack also records the length it needs),
a non-ancestor `if` forks the state into its two arms (an arm that returns/raises is dead), ancestor arms and the guard with the
requested polarity become literals, in-place stores unbind the target. The conjunction (DNF over `and`/`or`/`not`) is solved
per ENTRY term: a parameter, or `param[i]` of a list parameter whose length comes from the `len(param)` literals and the
highest index. Literals solved: membership in / equality with a module constant (sets and `A | B` resolved by AST through
relative imports), `len` of a str or list, `str` predicates and `in` (the `str.*` preimages), regex (non-)membership and group
presence from `regex_members` (a group value is embedded at the group's span in a member; a non-member is a member under
different flags that the real flags reject), `X.split(sep, k)` lengths and `(D + [None, ..])[i] is None` from the split preimage,
`inspect.cleandoc(X)` through its fixed points. Each witness is a dict over entry parameters only, and is accepted only after
the WHOLE path (every literal, constants resolved from the source) evaluates as required. A literal outside this list is named
(`Unsupported`) and the ordinary route runs. Not solved (named): receiver attributes (`self.titles_re.search(text)`, built at run
time from `self.sections`), guards inside loops (loop_guard's shape only), several entry terms in one literal (`a == b`),
non-string element types, `any`/`all` comprehensions. Domain: any function whose path to the guard consists of these
statements and whose literals are of these kinds, every list length and string value the literals allow (no fixed sizes).
Tests: `tests/test_condition_path_guard.py` on the unchanged upstream package in `tests/fixtures/docstring_parser`.

*Extension (dataclasses-json):* literals `x is None` / `x is not None` over a parameter, and `isinstance(x, C)` (also the try-wrapped
`_isinstance_safe(x, C)`, recognised by its shape and resolved to `isinstance`) where `C` is a class, a tuple, a `typing` alias
(`Collection`, `Mapping`, resolved by origin) or a `typing` constant's `__args__` (`Json.__args__` of `Json = Union[dict, list, ...]`,
evaluated by AST). Names come from relative imports, absolute imports inside the package, and `from <stdlib> import X` for a short
allow-list (typing, collections.abc, datetime, uuid, decimal, enum, numbers, dataclasses ...). A True `isinstance` demand is given by
canonical instances of stdlib classes (`[]`, `{}`, `datetime`, `UUID(int=0)`, a member of a dynamic `Enum`, `Decimal(0)`, the
builtin scalars), a False one by values outside the class; a user class is never instantiated, and a stdlib predicate such as
`dataclasses._is_dataclass_instance(obj)` is a check-only literal (it never supplies a value). An `if` whose arms leave the same
bindings re-joins the paths (so a chain of `if x is not None: d['k'] = x` blocks is one state, not 2^n). Not solved (named):
`hasattr(EnumClass, name)` over a class defined in a sibling module (needs the class), type-object parameters (`cls`, `type_`),
dict/list parameters whose elements are coupled in one literal (`k in overrides`).
Tests: `tests/test_condition_path_guard_dcj.py` on the unchanged upstream package in `tests/fixtures/dataclasses_json`.

*Extension (re.finditer loops, owner 1365):* a guard inside `for m in re.finditer(P, X, flags)` binds `m` to `next(re.finditer(..))`
(the first iteration; `continue`/`break` arms of earlier `if`s are dead paths for later statements), so `chunk = m.group(0)` is a term
over `X`. `not chunk` (group(0) empty) is REFUTED when the pattern's parse-tree minimum match length is >= 1
(`regex_members.min_match_length`: an anchor or look-around counts 0, a literal/class 1, a repeat its minimum): `solve_condition`
then returns `strategy="infeasible"` with `input_constraints["refutation"]` (the pattern, flags, minimum length) and an
explanation, and `driver._flip_static` turns that into a sound infeasibility vote, so `drive_to_line` ends `plateau`
"provably unreachable". The truthy arm needs a match: `X` is lifted through `Y[m.start():]` (a regex-match slice) and
`inspect.cleandoc` to the entry (`text=':x'`). Joins of `if` arms ignore bindings nothing reads afterwards (liveness by
name), which keeps the path count at the number of live forks. Domain: patterns whose every match has a required first
character or a mandatory repeat (minimum length >= 1; any such pattern, no special sizes); a pattern that can match empty
is not refuted (its arm gets a witness). Not claimed: groups other than group(0), `if chunk == ''` spellings, `while`/`for`
over other iterables. Tests: `tests/test_continue_refutation.py`.

*Interpreter-constant guards:* a guard that reads only `sys.version_info` (and `.major/.minor/...`, `[i]`), `sys.platform`,
`sys.maxsize`, `sys.byteorder`, `os.name/sep/linesep/pathsep` and literals (`path_guard.environment_truth`; no call is made) is decided
by the interpreter it runs on (`condition._environment_solution`): the arm it takes answers `{}` (no input constraint), the other arm is
`infeasible` with `refutation.kind == "environment"`, and a path through a false such condition (an ancestor arm or an earlier
`if`) is infeasible too instead of getting a made-up witness. It is sound for this interpreter only, so the equivalence proof of
the unreachable-statement route (docs/EQUIVALENCE.md) ignores `kind == "environment"` refutations. dataclasses-json utils.py 9/35
(`sys.version_info.minor == 6`) and mm.py 142 are this shape. Tests: `tests/test_condition_env_constant.py`.

**Dead arms refuted from what the program can and cannot do** (`solve/dead_arms.py::refute_arm`, called first by `solve_condition`;
owner 1365 style: the arm's candidate is built and refuted from a constraint, the answer is `infeasible` with
`input_constraints["refutation"]` = {kind, constraint, candidate, ...}; `driver._flip_static` votes it, `drive_to_line` ends `plateau`
"provably unreachable", and the equivalence unreachable-statement proof uses it). CLOSED-WORLD over the `.py` files of the source's
directory (the package): nothing outside is assumed to write a key, mutate a table or raise a program exception.
* `dict-key-never-written`: `if 'k' in V` (`not in`: the else arm) with `V` a local dict (`V = {..}` / `dict(..)`, or `V = D[e]` over
  `D = defaultdict(dict)`): the key set that can reach `V` is collected from EVERY occurrence of `V`/`D` in the function and must be
  closed (literal-key stores, literal-key reads, literal `in` tests; one binding each). Refused on any other use: a dynamic key,
  `.update/.setdefault/.pop`, `del`, an inner dict replaced, `V`/`D` passed, aliased, returned or iterated, rebound, captured by a
  nested scope, `global/nonlocal`. dataclasses-json core.py:80 (`'mm_field' in field_metadata`; line 64 writes `'mm_fields'`).
* `callee-never-raises`: `if not rets` after `rets = []; for f in T.values(): try: rets.append(f(x)) except E: ...` with `T` a non-empty
  module-level dict literal that nothing in the package mutates, passes or rebinds, whose values are package functions. `rets` is
  empty only if every `f` raised `E`; one value whose exception-escape summary says `E` cannot escape (and every other exception leaves
  the function before the guard) makes the arm infeasible. The summary follows every call: a bare name to the caller module's
  module-level function (else the package's), `mod.f` to that module's `f`, `x.m` to the methods `m` of the caller's module when it
  defines any (else every package method), a class to its `__init__/__post_init__/__new__` and bases, a package function used as a value
  as called; `raise C` is `E` iff `C` is `E` or a package subclass, a bare/variable `raise` iff the enclosing handler covers `E`;
  `try/except` removes what its handlers cover; builtins, stdlib module functions, builtin/`re` object methods are safe; anything
  unresolved (a called variable, a method nothing defines, a `match`) or a package dunder/property that may raise `E` means "may
  raise" (no refutation). docstring_parser parser.py:23-24 (`numpydoc.parse` never raises `ParseError`; `rest.parse` and
  `google.parse` may).
**Correction (owner 1408; the T176 summary was wrong on the first try, AI-written):** it treated every builtin, standard-library and
builtin-method callee as unable to raise the caught class, so `rets = []; for f in PARSERS.values(): try: rets.append(f(s)) except ValueError`
over `int`/`float` parsers was refuted although `parse_any("x")` reaches `if not rets`. Now: for an exception class the PROGRAM defines
(docstring_parser `ParseError`) a builtin/stdlib callee can raise it only through a callback into program code, which is followed; for
a builtin or library-defined class (ValueError, `json.JSONDecodeError`) every builtin/stdlib/builtin-method callee may raise it and a
package function counts as unable to raise it only if it does nothing but return a name or a constant (`_trivial_function`), since any
operation (a subscript, an operator, a call) can raise a builtin exception. parser.py:24 (`ParseError`) is re-derived under the corrected
summary and still refutes.
Domain: exactly those shapes and summaries; a callee whose behaviour depends on code outside the package (a user-registered section,
a monkey-patched table) is outside the claim. Tests: `tests/test_dead_arms.py` (original sources, driver, equivalence, negative controls).
**`explanation`** gives human-readable rationale for the chosen strategy — useful for understanding solver behavior and debugging unsolved guards.

### Serialization

Both solution types have serializers that produce JSON-safe dicts:

```python
from tharkun.branch_solver import serialize_branch_solution, serialize_coverage_result

# Convert to JSON-safe dict
d = serialize_branch_solution(sol)
# {
#   "guard_source": "x > 100",
#   "solved": True,
#   "best_inputs": {"x": 101},
#   "best_inputs_constraints": {"x": {"type": "int", "gte": 101}},
#   "blocker_info": None,
#   "desired_branch": "then",
#   "strategy": "direct",
#   ...
# }

# Convert entire coverage result to JSON-safe dict
d = serialize_coverage_result(result)
```

---

## Progressive Solving

`solve_branches_progressive()` is a generator that yields `BranchSolution` objects *as each edge is solved*, without waiting for all edges to complete:

```python
from tharkun import solve_branches_progressive

# Process results immediately as they arrive
for sol in solve_branches_progressive("large_module.py"):
    if sol.solved:
        print(f"Line {sol.lineno} ({sol.desired_branch}): {sol.best_inputs}")
    else:
        print(f"Line {sol.lineno} UNSOLVED: {sol.blocker_info}")

# Stop early after 10 results
solutions = []
for sol in solve_branches_progressive("large_module.py"):
    solutions.append(sol)
    if len(solutions) >= 10:
        break
```

This is useful for large modules where you want to start using results before the full solve completes. The generator yields in the order branches are solved, which may not be file order.

---

## Deterministic Mode

By default, tharkun uses Hypothesis with a random seed. Set `deterministic=True` to fix the Hypothesis seed, producing reproducible results suitable for CI:

```python
from tharkun import solve_branch, solve_uncovered_branches, solve_branches_progressive

# Single branch — deterministic
sol = solve_branch("module.py", 42, deterministic=True)

# All branches — deterministic
result = solve_uncovered_branches("module.py", deterministic=True)

# Progressive — deterministic
for sol in solve_branches_progressive("module.py", deterministic=True):
    ...
```

**Trade-off:** Deterministic mode may slightly reduce solve rate (a random Hypothesis seed occasionally finds inputs that a fixed one misses), but ensures that two runs on the same file produce identical results.

---

## MCP Server

Tharkun ships an optional MCP server (built on the mcp package's server class; mcp 1.x and 2.x both supported) for use with Claude Desktop, Claude Code, or any MCP-compatible client.

**Installation:**

```bash
pip install tharkun[mcp]
```

**Start the server:**

```bash
tharkun-mcp            # stdio MCP server
# or:
python -m tharkun.mcp_server
```

**Configure in Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "tharkun": {
      "command": "tharkun-mcp"
    }
  }
}
```

**Available tools (6):**

| Tool | Parameters | Wraps |
|------|-----------|-------|
| `solve_guard` | `file`, `lineno`, `desired_branch` | `solve_branch()` |
| `find_branch_edges` | `file` | `_find_branch_edges()` |
| `solve_all_branches` | `file` | `solve_uncovered_branches()` |
| `decompose_guard` | `file`, `lineno` | `decompose_guard()` |
| `map_call_chain` | `file`, `target_line` | backward constraint propagation |
| `analyze_mocks` | `file` | `detect_externals()` |

**Note:** [Eldarion](https://github.com/ardier/eldarion) provides an MCP server that is a **superset of tharkun-mcp** — it exposes all 6 of these tools plus 31 more debugger tools. If you're using Eldarion, you only need to run `eldarion-mcp`.

---

## Branch Edge Enumeration

`_find_branch_edges()` returns all then/else edges in a file as `BranchTarget` objects:

```python
from tharkun.branch_solver import _find_branch_edges

edges = _find_branch_edges("module.py")
for edge in edges:
    print(f"line {edge.lineno:3d}  {edge.stmt_kind:10s}  {edge.desired_branch}")

# line  42  if          then
# line  42  if          else
# line  57  for         then
# line  57  for         else
# line  63  while       then
# line  63  while       else
```

`stmt_kind` values: `"if"`, `"for"`, `"while"`, `"assert"`, `"comprehension"`.

This is useful for understanding which branches exist in a file before deciding which to target — the `solve_branch()` / `solve_all_branches()` calls then solve specific edges from this list.
