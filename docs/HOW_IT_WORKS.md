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

### Step 8: CrossHair Concolic Execution (Optional)

When everything else fails, falls back to symbolic execution via CrossHair. Explores paths symbolically to find satisfying inputs.

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

By default, tharkun uses CrossHair (non-deterministic symbolic execution) and Hypothesis with a random seed. Set `deterministic=True` to skip CrossHair and fix the Hypothesis seed, producing reproducible results suitable for CI:

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

**Trade-off:** Deterministic mode may slightly reduce solve rate (CrossHair occasionally finds inputs that the static solver misses), but ensures that two runs on the same file produce identical results.

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
