# Avengers object synthesis

Tharkun is the umbrella package. Avengers is its object-construction and
call-boundary subsystem, imported as `tharkun.avengers`; it is not a separate
distribution. It combines independent constructor and setup strategies through
Tharkun's knowledge base, then replays each candidate in a fresh interpreter.

The initial portfolio combines a static constructor template, annotation-based
defaults, Hypothesis constructor search, constructor and setup sequences mined
from Pynguin test source, bounded setup-method enumeration, and authoritative
fresh-process replay. Every result retains its derivations in the shared
knowledge base, so two tools that find the same recipe add provenance rather
than duplicate work.

An `ObjectGoal` names a source file, a class, and the method or module-function
calls that must return normally. `synthesize_object` searches the default pass
portfolio and returns every replay-validated recipe. Absence of a recipe is an
unresolved search result, not an infeasibility proof.

```python
from pathlib import Path

from tharkun.avengers import MethodCall, ObjectGoal, synthesize_object

goal = ObjectGoal(
    str(Path("session.py").resolve()),
    "Session",
    calls=(MethodCall("close"),),
)
report = synthesize_object(goal, max_setup_steps=2)

if report.best is not None:
    print(report.best.program)  # constructor, setup calls, then close()
```

For an installed class, `ObjectGoal.from_type` locates the module source. This
complete example names only the desired `stop()` boundary:

```python
from codetiming import Timer
from tharkun.avengers import MethodCall, ObjectGoal, synthesize_object

goal = ObjectGoal.from_type(Timer, (MethodCall("stop"),))
report = synthesize_object(goal, max_setup_steps=2)
print(report.best.program)
```

Avengers discovers and verifies:

```python
object_0 = Timer()
object_0.start()
object_0.stop()
```

The runnable version is [`demos/avengers_codetiming.py`](../demos/avengers_codetiming.py).

If no boundary calls are supplied, the first objective is public method
coverage. Avengers tests each statically visible, zero-argument public method
independently against every validated receiver state. `best` is the state that
returns normally from the most methods; `objects` is a small greedy portfolio
whose union covers every method reached by any candidate:

```python
report = synthesize_object(ObjectGoal.from_type(Timer), max_setup_steps=2)

assert report.covered_methods == ("start", "stop")
for state in report.objects:
    print(state.program)
```

For `codetiming.Timer`, one state is `Timer()` and the other is
`Timer(); start()`: the first admits `start()`, while the second admits
`stop()`. Every probe starts from a newly replayed recipe, so one method cannot
accidentally prepare the next. Repeated setup sequences are deliberately
enumerated, and sequences that raise, crash, or time out remain failure
evidence rather than valid states. Automatic coverage is bounded to 256 child
probes by default; raise `max_method_probes` for a larger class or inspect
`coverage_complete` to distinguish saturation from a truncated probe matrix.

When the constructor and setup are already known, replay the exact program
directly. Setup runs after construction and before the goal boundary:

```python
from tharkun.avengers import (
    ConstructionRecipe,
    ConstructorCall,
    IsolatedExecutor,
    MethodCall,
    ObjectGoal,
)

goal = ObjectGoal("session.py", "Session", (MethodCall("close"),))
recipe = ConstructionRecipe(
    goal.key,
    constructor=ConstructorCall(kwargs={"token": "dev"}),
    setup=(MethodCall("open"),),
)
outcome = IsolatedExecutor().replay(goal, recipe)
assert outcome.succeeded
```

A boundary may also be a free function in the target module. `OBJECT` inserts
the constructed receiver in positional or keyword arguments:

```python
from tharkun.avengers import FunctionCall, OBJECT, ObjectGoal, synthesize_object

goal = ObjectGoal(
    "session.py",
    "Session",
    calls=(FunctionCall("validate_session", (OBJECT,), {"role": "admin"}),),
)
report = synthesize_object(goal)
```

Explicit boundary success means that all requested calls returned normally;
the return value is not treated as a predicate. A validating function should
raise when it rejects the object.

## The objective ladder

Constructing an object is the first objective, not the only one. With
`objectives=True` the portfolio pursues the rest, each as an ordinary pass
whose proposals are certified by the same fresh-process replay:

| Objective | What it means | Where it lands |
| --- | --- | --- |
| valid construction | a recipe that replays cleanly | `report.validated`, `report.best` |
| method coverage | public methods that return normally from some state | `report.covered_methods`, `report.objects` |
| state coverage | genuinely different receiver states | `report.states` |
| line / branch coverage | how far into the class body the search got | `report.line_coverage`, `report.branch_coverage` |
| invalid sequences | calls the class rejects on purpose | `report.negatives` |
| boundary behaviour | crashes, grouped by cause and re-checked | `report.crash_causes` |

```python
report = synthesize_object(ObjectGoal.from_type(Timer), objectives=True)
```

The objectives are off by default. Each one spends real child processes -- an
order of magnitude more than construction alone -- and the first question is
usually only whether the class can be built. Time to first object is unaffected
either way: the objectives consume validated recipes, so they start after
`report.time_to_first_object_s` is already recorded.

**State coverage** keys on the replay's state digest, not on recipes. Two
recipes reaching the same state are one result; a recipe reaching a state no
other did is a new one however ordinary it looks. A replay with no digest is
skipped rather than counted, which would inflate the number. The digest comes
from `py_obj_flatting` when it is installed and from a shallow, exact-typed
fallback otherwise -- one that reads no property, calls no `repr` and dispatches
no `__len__`. The fallback distinguishes fewer states, so it can undercount; it
never invents a difference.

A digest only names a state if it *reproduces*. `Timer(); start()` stores a
fresh `perf_counter()` reading, so replaying that one recipe digests
differently every time -- and counting digests would report a new state per
draw and send the search extending each of them. Every candidate state is
therefore confirmed by one repeat replay. When the two agree the digest is the
identity; when they disagree the recipe is, so one program counts as one state
and `ReceiverState.stable` says which happened. On `codetiming.Timer` that is 4
stable states out of 19.

**Branch coverage** is measured in the worker, with the target module loaded
*outside* the trace: import-time lines are not what a recipe exercised, and
counting them would credit every recipe with the same module body. Uncovered
branches feed back: where a guard tests an attribute that is also a constructor
parameter, the coverage objective proposes the truthy and falsy literal for that
parameter's annotation. Where no literal will do it proposes nothing, rather
than a value that passes the guard and crashes on the line below.

**Invalid sequences are results.** `start(); start()` and a bare `stop()` are
produced deliberately and kept. Only exceptions the target itself defines or
raises by name count, so a `TypeError` from probing the wrong arity is not a
finding, and every kept sequence is replayed a second time in its own child --
both the isolation a batched scan trades away and the reproducibility check a
crash report needs.

## Asking for misbehaviour

A goal's boundary calls must return normally by default. `expect` inverts that:
given a behaviour, find the program that reaches it.

```python
# An object whose stop() raises, rather than one whose stop() works.
goal = ObjectGoal.from_type(Timer, (MethodCall("stop"),), expect="raises:TimerError")

# An object that takes the interpreter down at the boundary.
goal = ObjectGoal(path, "Codec", (MethodCall("decode"),), expect="crashes")
```

`"returns"` (the default), `"raises"`, `"raises:<name>"` -- any suffix of the
exception's dotted name -- and `"crashes"`. A directed expectation is about the
*boundary*: a constructor that raises has not produced an object that
misbehaves, it has failed to produce an object, so it does not satisfy the goal.

The worker reports what happened and the goal decides what that means, so one
child result satisfies a `raises` goal and fails a `returns` goal without being
replayed twice. A directed goal's own answers are excluded from
`report.failures` -- filing a search's results under its failures would be a
strange way to report them.

Not finding one is still unresolved, never a proof that the behaviour is
unreachable.

```console
$ python -m tharkun.avengers synthesize timer.py Timer --call stop --expect raises
object_0 = Timer()
object_0.stop()

# boundary raised: codetiming._timer.TimerError
```

## Objects that need other objects

A class whose `__init__` takes a domain object cannot be reached by any amount
of literal search: no strategy can produce the argument. `NestedObject` carries
a whole sub-recipe as one argument value, so a composed program stays a single
self-contained record that a fresh worker replays without reaching back into the
parent:

```python
report = synthesize_object(ObjectGoal("orders.py", "Shipment", (MethodCall("volume"),)))
print(report.best.program)
# nested_0 = Box(lower=0, upper=1)
# object_0 = Shipment(box=nested_0)
# object_0.volume()
```

`NestedCandidatePass` resolves an object-valued parameter's annotation
statically -- the same module or an import, never a guess -- takes a recipe from
the knowledge base or the store, and otherwise runs a bounded sub-search of its
own. Nesting is capped: an unbounded chain is a way to spend a worker's whole
timeout below the boundary the caller actually asked about. `compose_depth=0`
turns it off.

## Backends

Each backend is a pass that runs the tool itself. Two contracts hold for all of
them. **Nothing a backend says is believed** -- it proposes a candidate or some
test source, and promotion still comes from the ordinary fresh-process replay.
**Everything is availability-gated** -- a missing tool is a missing tool, not an
error.

**A backend is an Avengers-internal implementation detail, never a direct
dependency of a caller.** Pynguin and Ghostwriter (and the `black` formatter
they share) ship as the `backends`/`bench` extras and stay behind
`tharkun.avengers.backends`; nothing outside `avengers/` imports that module,
`pynguin` or `black` directly (`tests/test_sv_no_backends.py`,
`docs/LAYERS.md`). A caller outside Avengers -- SV's `avengers` stage
included (`sv/avengers_seed.py`) -- may still ASK for a backend through the
public `synthesize_object(backends=(...))` parameter; it just never imports
the backend module or the tool itself, so a shipped `python -m tharkun.sv`
run with the stage off (the default) never loads either tool, and even with
the stage on, `pynguin`/`black` are reached only through that one API, never
directly. This is the "no outside engine under the hood" guarantee: not that
Avengers may never be told to use one, but that nothing above it can load one
by any other door.

| Backend | What it contributes | Needs |
| --- | --- | --- |
| `hypothesis-from-type` | `st.from_type` on each constructor parameter | `hypothesis` |
| `crosshair` | solved constructor arguments from `crosshair cover` | `crosshair-tool` |
| `pynguin` | test source, including its crash tests | `pynguin`, `black` |
| `ghostwriter` | test source from Hypothesis' Ghostwriter | `hypothesis`, `black` |

**SV's `avengers` stage** (`sv/avengers_seed.py`, [internal ref]) is the one caller outside
this package that requests a backend: it constructs every class the SV task's
module defines and offers back a validated recipe's literal arguments as SEED
candidates for `sv/stages.py`'s enumeration, with `backends=("pynguin",)` so
the restored backend, not just Hypothesis, gets a chance to propose them.
Measured on all 80 (task, property) pairs of the SV-Benchmarks Python
Real-world set, backend actually installed and invoked: **0 pairs gained, 0
lost** against the portfolio without it -- identical verdict on every pair,
including the 30 pairs over the object-heavy tasks it targets
(richards/raytrace/go/chaos) -- and a real cost on 19/80 pairs where its own
search (its own budget plus the Pynguin subprocess) measurably crowded out
the concolic stage's run budget within the same total time limit (e.g.
`bm_raytrace_nondet` fell from 1104/1108/1092 runs to 1 on three of its four
properties). Per the keep-only-if-gain rule the stage stays registered but
out of `stages.py::DEFAULT`; it remains available via `--stages ...,avengers`
for re-measurement as the portfolio or the corpus changes.

```python
report = synthesize_object(goal, backends=("hypothesis-from-type", "crosshair"))
```

```console
$ python -m tharkun.avengers backends
ok crosshair     available
ok flatting      available
ok hypothesis    available
-- pynguin       pynguin is not installed
```

Backends run *after* the built-in passes, deliberately: an external generator
costs seconds to tens of seconds before it says anything, and putting one first
made time to first object an order of magnitude worse than the built-in
portfolio alone. The measurement is in the benchmark below.

Any tool that writes Python test source shares one recipe miner
(`GeneratedSourcePass`), so adding a generator means producing source and
nothing else. Pynguin's *crash tests* are ingested rather than filtered out: a
sequence that killed the interpreter is exactly the boundary behaviour this
search is for.

CrossHair can also be used through Hypothesis, with no code here at all:

```python
report = synthesize_object(goal, hypothesis_backends=("hypothesis", "crosshair"))
```

when the `hypothesis-crosshair` plugin is installed.

## The scheduler and the store

`Avengers` runs the portfolio to a fixed point over shared evidence. Every pass
is measured -- activations, runs, wall time, emissions, and *novel* emissions --
and `PortfolioOrder` (the default) keeps the declared cheap-first order while
`YieldOrder` ranks by novelty per second, offering unmeasured passes first so
the ranking cannot become self-fulfilling. `SaturationBudget.max_pass_seconds`
stops one slow backend from spending the whole budget.

`TemplateStore` is the persistence seam. It keeps validated recipes, certified
flat templates and negative sequences, sharded by goal key, merged rather than
replaced, published atomically, and read with the same discipline as a worker
response -- closed vocabulary, strict validators, a size ceiling, no pickling.

```python
store = TemplateStore(Path("~/.cache/tharkun/avengers").expanduser())
report = synthesize_object(goal, store=store)
```

Reloaded recipes are published as **candidates**, never as validated results. A
recipe that worked last week is a good guess, not evidence: the target may have
changed, so it earns its promotion from a fresh replay like any other proposal.
`recipes_for_type()` crosses goals, which is what lets a discovery about one
class unblock a search for another.

## Benchmark

`tharkun.avengers.bench` compares configurations -- each backend alone, and all
of them together -- across the objective ladder:

```console
$ python scripts/setup_benchmarks.py --profile current-3.12 --smoke
$ python -m tharkun.avengers bench --profile current-3.12 --smoke --repetitions 3
```

The subjects are the ten pure-Python packages from Lukasczyk et al., *An
empirical study of automated unit test generation for Python* (EMSE 2023):
apimd, codetiming, dataclasses-json, docstring_parser, flake8, flutes, flutils,
httpie, pdir2 and python-string-utils. Two profiles: `emse-3.10` pins that
study's versions on its interpreter and is comparable to its published numbers;
`current-3.12` takes whatever installs today and is explicitly not.

One environment per subject -- the 2021 pins genuinely conflict with each other
-- and `IsolatedExecutor(interpreter=...)` runs each replay in the environment
that can import the subject.

Two statistics, always together: the Mann-Whitney U test at alpha = 0.05 and the
Vargha-Delaney A-hat-12 effect size, both stdlib-only. A significant result with
a negligible effect is a large sample noticing something that does not matter; a
large effect that is not significant is too few runs to tell. Every row records
which backends were *available*, so a configuration that scored nothing because
its tool was missing is not reported as having lost.

The smoke result is in [`benchmarks/avengers/README.md`](../benchmarks/avengers/README.md)
with its scope stated up front -- 2 packages, 4 classes, 3 repetitions, 40s, on
the profile that is *not* comparable to the published numbers. In short: the
portfolio finds significantly more valid objects and more reproducible states
than the built-in floor, `hypothesis` or `ghostwriter` alone (large effects),
costs nothing in time to first object, and beats neither `pynguin` nor
`crosshair` significantly on anything. At that scale "Avengers beats the best
single backend" is **not** established. The full corpus at the paper's budget is
the measurement; the smoke run is the harness working.

## The companion bridge

Avengers and [`py_obj_flatting`](https://github.com/ardier/py_obj_flatting) are
both parts of Tharkun. The latter is the official companion for converting a
live exemplar into a deterministic, byte-addressable object recipe. It keeps
its own distribution and safety contract. Install it in the same environment
to enable `FlattingPass` in the default Avengers portfolio and the explicit
bridge API:

```python
from tharkun.avengers import flatting_available, template_from_object

if flatting_available():
    template = template_from_object(existing_object)
    assignments = template.decode()
    rebuilt = template.materialize(
        template.reproduce(assignments),
        allow=(type(existing_object),),
    )
    assert rebuilt.ok
```

`template_from_object()` is a direct convenience API. Its flattening pass is
static, but certification reconstructs the exemplar and therefore runs the
recipe's constructors in the current process. The default `FlattingPass` uses
`IsolatedExecutor.mine_template()` so that certification happens in its
disposable worker. Use that executor path when constructors may crash or hang.
This is process-failure containment, not a security sandbox: target code in the
worker retains the user's filesystem, network, and environment access.

The dependency is optional: importing `tharkun.avengers` does not load it, and
constructor search and boundary replay remain available when it is absent. The
default portfolio detects it lazily before adding `FlattingPass`.

When the default portfolio creates a template, its recipe crosses the worker
boundary as versioned JSON. Loading that payload verifies its SHA, slot layout,
duplicated metadata, and exemplar leaf paths. The surrounding worker response
also uses a bounded, data-only JSON vocabulary, so the parent never unpickles
data produced after target code has run.

## Related work

Berg, Krodinger, Lukasczyk, Panichella, Fraser, Vanhoof and Devroey,
[*Real-World Fault Detection for C-Extended Python Projects with Automated Unit
Test Generation*](https://arxiv.org/abs/2603.06107) (arXiv:2603.06107), adapts
Pynguin to execute each generated test in an isolated subprocess, so a
segmentation fault in a C extension costs a child rather than the whole search.
Three of its decisions are adopted here:

* **Isolation as an outcome.** A crash or a hang is a `ReplayOutcome`, not an
  exception the search has to survive. Avengers already worked this way; the
  paper is the evidence that it is the right way.
* **Crash causes, not crash counts.** A crash is keyed by
  `(last phase, signal or exception type)` and deduplicated. Counting failures
  rather than causes reports the size of the search, not the target's defects.
* **Re-execution before counting.** A crash that does not reproduce in a fresh
  interpreter is not a finding; `ReplayOutcome.reproduced` records the check.

The evaluation protocol -- repeated runs compared with Mann-Whitney U at
alpha = 0.05 and the Vargha-Delaney effect size -- comes from the same line of
work.

Two differences are worth stating plainly. That paper's subjects are DS-C, 1648
modules from 21 C-extension libraries; the corpus here is the pure-Python
EMSE-2023 set, so the numbers are not comparable and the manifest says so. And
its unit is a module and its artefact a test suite, where the unit here is a
class and the artefact a *recipe*: a construction program certified by replay,
which can be stored, re-keyed onto a different boundary, and passed as an
argument when building something else.
