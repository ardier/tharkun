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

The stages of `flatobject.py::_search` share the scheduler's evidence cursors as
well as its knowledge base. Later stages consume new candidates without
replaying earlier ones: a child timeout followed by a normal return must not
replace the canonical outcome of the same replay identity. A cut remains a cut;
new strategies can still propose and validate other recipes.
The scheduler advances past irrelevant facts and when its pass runs. A budget
cut before a consumer runs leaves its relevant evidence available to the resumed search.

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
coverage. Avengers tests each statically visible public boundary of the class
-- every public method, whether it takes arguments or not, `__call__`, and
every `@property` -- independently against every validated receiver state.
`best` is the state that returns normally from the most methods; `objects` is a
small greedy portfolio whose union covers every method reached by any
candidate:

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

### Boundaries that take arguments, and properties

`passes.py::public_method_templates` reads each public boundary's kind
(`method`, `classmethod`, `staticmethod`, `property`; a `@x.setter` is not a
boundary of its own) and its parameters (`solve/frontend/constructor.py::
function_parameters`, the same reader the constructor template uses). A
property is probed as an `AttributeRead` -- `object_0.size`, the getter and
nothing else; before [internal ref] it was CALLED, which raises `TypeError` whenever
the value is not callable. A method with required parameters gets a short
list of candidate calls (`::method_call_candidates`), cheapest and likeliest
first:

1. the baseline of each parameter's type (`""`, `0`, `[]` ... -- the constructor
   baseline's values, so a setup `add("", 0.0)` and a later probe `mean("")`
   meet on the same key; "Value domains" below);
2. the truthy value of each parameter's type -- all parameters at once, then one
   at a time with the rest at their baseline (`Timers.apply(sum, "")` keeps the
   key a setup `add("", 0.0)` stored, where all-truthy `apply(sum, "0")`
   changes it);
3. values the condition solver (`solve/condition.py::solve_condition`, under
   `no_targets()`: the parent never runs the target) finds for each `if` /
   `while` in the body that tests a parameter, both branches -- `pick("fast")`
   for `if mode == "fast"`;
4. literals the class's own source uses (`::_class_constants`, docstrings
   excluded; Pynguin seeds from module constants the same way) -- the
   numerals a decoder's table spells;
5. three derandomized draws of the parameters together from their types
   (`strategies.examples`, which draws without the constants Hypothesis harvests
   from every local module, so the draws do not depend on what the process has
   imported);

at most eight per boundary.

**Whole object arguments ([internal ref], owner Q15).** Before literal calls, the probe matrix asks
`passes.py::NestedCandidatePass.method_objects` for real values of the boundary's object parameters,
including defaulted ones. It uses the same validated recipes, recipe store and bounded constructor
search as constructor composition. The sub-search constructs whole objects with their setup calls;
it does not probe their methods recursively, and repeated requests for a type reuse that search's
result. `compose_depth` also bounds this search. `probe_methods` provides the same construction path
when probing recipes a caller already holds. An explicit `objects` mapping is authoritative:
no additional slot objects are discovered, and an empty mapping requests literal probes only.
Bootstrap supplies that mapping so restart and cross-read controls cannot trigger an independent search.
Only arguments actually offered through the hub mapping count as hub reads or served slots;
locally generated stand-ins carry no hub-object attribution.
The state-producing setup alphabet also uses this composer, so a setup method can receive a real
collaborator and establish the receiver state that a later boundary needs. Bootstrap's
`nested_search=False` also disables the setup composer's independent sub-search.
Within a replay, `_worker.py::_load_class` reuses a standalone source's namespace for its receiver
and nested arguments. Loading that source again would create different class identities and reject
a real argument at `isinstance`. The cache is cleared for each replay, including each batch item.

An annotation resolves to its class statically. For an untyped argument read through attributes,
`arguments.py::parameter_targets` proposes up to three project classes that declare all of those
attributes (methods, fields and constructor assignments), fewest surplus attributes first. This is
a structural proposal, never proof of the intended type. Each proposed value goes through its real
constructor and the consuming boundary in a fresh child. An `argparse.Namespace` is built with its
public `Namespace(**fields)` constructor and the fields the method reads. No partial object state
is synthesized or patched. Several object parameters are filled together, retaining the scalar
guard and literal candidates; positional-only, static and class methods use the same path.

`duck.py::FAKE_PROJECT_CLASSES` defaults to `True` after this change (owner Q15, 2026-10-23).
Up to eight generated real argument calls precede the existing literal and observed candidates,
which remain available, including all existing fallback calls. Enum annotations keep their finite
real member domain instead of receiving collaborator stand-ins. A returning stand-in remains a measured result, counted apart by
`report.works.as_dict()["standin_only_methods"]`: these boundaries have a working probe only through
a fake argument or a receiver built from one. `duck.py::holds_fake` accepts both recipes and boundary
calls, so measurements can distinguish real object use from a namespace or `ConstFn` fallback.

A boundary that returns a generator is drained (`_worker.py::_drain`, at most
`_worker.py::_MAX_YIELDS` = 64 items, then closed), inside the same `try` and
the same trace as the call. A generator method's body -- where its mutants sit
-- runs only when it is iterated. Before [internal ref] such a boundary counted as
returned and covered without running a line of its body: about 114 sampled
GUUT mutants, such as flutes' `_ReverseReadlineFile.generator`, whose drained
body raises. Now a body that raises fails at `boundary[i]`, as a plain method
would. What it yielded is recorded in `ReplayOutcome.return_types` as
`builtins.generator[<distinct yielded types>]`, with `...` when the drain was
cut. Only an exact generator object is drained. Other lazy results (`map`,
iterator classes, coroutines, async generators) are recorded by type and not
run.

None of these is believed. The probe matrix is spent in two phases (`api.py::_probe_methods`, [internal ref]). First candidate 0 of every
boundary on EVERY state, at most three fifths of the budget: a state is the only thing that makes some boundaries return
(`Timers`' `add('', 0.0)`), so no state goes without its first probe. Then the diagonals of the (state, candidate) grid: wave `d`
replays candidate `k >= 1` of every boundary on the state ranked `d - k`, where the boundary has not worked yet, so the cheap
states get their later candidates before the dear ones and both deepen together; a boundary that worked past its entry guards from some state gets no
later candidates elsewhere. The diagonals replaced "round `k` replays candidate `k` on every state" (breadth first over STATES),
which spent all 256 probes on candidate 0 of a class with many states and boundaries (flake8 `Plugin`: 48 states, 15 boundaries)
and never reached the collaborator stand-ins at candidates 2 and 3 (`Plugin` 9 -> 12 of 15 boundaries). Spent depth first (the first version), the default
backends' extra states used codetiming `Timers`' 256 probes before the one
state that covers `count`/`mean`/... was reached (12/12 -> 4/12 boundaries on
codetiming). The states it keeps are the cheapest validated ones -- fewest
setup calls, fewest constructor arguments, then the SHORTEST program, so a
baseline state (`add('', 0.0)`) is kept before a search's arbitrary one
(Pynguin's `add('>M&7..', -1997.96)`); breaking that tie by recipe key let a
run with more Pynguin states cut the state `Timers.apply(sum, '')` returns
from. With `objectives=True` every candidate runs, because each may reach
different lines. Probes are independent fresh children, so
`probe_workers` (default 4) run at once.

The setup alphabet (`::_setup_calls`) is every state-producing call: a
zero-argument method that writes the receiver as before, and now a method
with arguments that writes it, with its first two candidate calls. Writing
the receiver includes mutating a container it holds (`self.data.append(x)`,
`self.data.setdefault(k, []).append(v)`), not only assigning `self.x`, and ([internal ref])
calling a method of the class that does (`setupstate.py::writers`: `initialize(argv)` writes
nothing itself and calls the methods that write everything). The calls that establish a field some
boundary needs go first, and a sequence whose steps feed one another before one that does not
("Collaborator setup state" below).

Measured on the ten GUUT subjects (CPython 3.14.4) before [internal ref]: of 206
classes, the zero-argument-only universe hid the main entry point of the
classes that carry the most sampled mutants -- `docstring_parser`
`GoogleParser.parse(text)` (294 of 569 mutants), python-string-utils
`__RomanNumbers.encode(n)` / `.decode(s)` (221 of 1000), flutils
`AnsiTextWrapper.wrap(text)` (155 of 1000): their method universe was empty
or held only properties.

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

The worker loads the goal's module through `solve/loader.py` and finds the
class in that module's namespace, so whatever the module itself can bind is a
valid goal: a module-PRIVATE class such as python-string-utils'
`class __ISBNChecker` (name mangling never applies at module scope) is an
ordinary global, and only `__x__` interpreter metadata is left out of the
namespace (`solve/loader.py::_is_dunder`). Target output is discarded into a
real `os.devnull` file rather than an in-memory buffer, because some modules
ask their stdout for a descriptor while they are imported (httpie's
`context.py` runs `curses.setupterm()` in a class body); with a buffer, not
one httpie class could be loaded ([internal ref]).

Explicit boundary success means that all requested calls returned normally;
the return value is not treated as a predicate. A validating function should
raise when it rejects the object.

**GUUT objects (owner ruling, 2026-10-19; ruled, not built; Eregion [internal ref]).**
Every object the GUUT mutation-testing subjects need is to go through three
stages. Codetiming, python-string-utils and dataclasses-json come first, and the
objects are the receivers, parameters and return values at each subject's public
API.

1. **A unit.** Avengers supports the object on its own, and every claim is
   replay-verified in a fresh child.
2. **Wrapped.** Eregion wraps it (`ObjectWrapper` `m`/`m_inv`, and Avengers
   initialization, reached under Eregion's `solve_`).
3. **Consumed by every component of the differential kill:** `driver.py`'s
   `_flip_avengers` tier, `sv/avengers_seed.py`, the abstract interpreter and the
   hybrid fuzzer.

Gaps are fixed here on `main`, a whole family at a time. The measurement and its
results belong to [internal ref]. Eregion's `docs/design/10_interprocedural_inversions.md`
records the ruling.

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
skipped rather than counted, which would inflate the number. The digest is
the SHA-256 of the receiver's **flattened** form (`py_obj_flatting`, see
"Flattening and observable effects"); there is no weaker stand-in. Where the
flattener is missing a replay carries the named refusal `flattening-unavailable`
and no digest, and `report.refusals` says so.

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
parameter, the coverage objective proposes the truthy and falsy value of that
parameter's type (`domains.py::Typing.truthy` / `::falsy`). Where no value will do it proposes nothing, rather
than a value that passes the guard and crashes on the line below. A callback
parameter is not such a case any more: an allowlisted builtin is a literal
(see "Callbacks" below), so `if self.logger:` gets `print`, never `True`.

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

## Parameters used only through their attributes

A parameter with no annotation, or one annotated as an attribute BAG
(`argparse.Namespace`, `types.SimpleNamespace`), is often used only by
reading attributes off it: flake8 hands nearly every class an `options` and
reads a dozen (`options.select`, `options.extend_ignore`, ...). No literal
and no annotation-resolved class can stand in for it.
`passes.py::DuckTypedCandidatePass` reads, statically, everything the goal class asks of the
parameter (`duck.py::constructor_demand`, since [internal ref]; before, `::parameter_attribute_uses`, which
`standins.py` still uses for the names alone) -- `p.attr` in `__init__` and `self.x.attr` in any method
where `__init__` stored `self.x = p`, following statically resolvable bases to the one that defines
`__init__` when the class inherits it (flake8's `Default(BaseFormatter)`; `::_class_chain`) -- with the
shape each use needs (iterated: `[]`; a text method: `""`; arithmetic or a numeric comparison: `0`;
subscripted: `{}`; tested: `False`; otherwise `None` -- each the baseline of the type the use names,
`passes.py::_fill`), and proposes the bag with exactly those attributes as a `NestedObject`
(`nested_0 = Namespace(select=[], ...)`), then the same bag with objects left unexpanded, all-`[]`,
all-`0` (an attribute the class only stores and later multiplies or compares) and all-`None`. An
attribute that is CALLED was not a bag attribute: it needed a real object, which `NestedCandidatePass`
builds when an annotation names its class -- including through a module alias the source imported
(`base_formatter.BaseFormatter`, `argparse.Namespace`; `::resolve_annotation_class`). Since [internal ref] an
untyped one is a namespace whose called attributes are `ConstFn` callables, and a collaborator's
projections are namespaces in turn ("Collaborator setup state" below). Measured on flake8 (CPython
3.14.4, before [internal ref]): `DecisionEngine` 0 -> 5/5 methods, `BaseFormatter` 0 -> 8/11, `StyleGuide` 0
-> 4/6.

**What the demand follows ([internal ref]).** `duck.py`'s reader also follows `super().m(x)` into the base class's method (the demand
of `error` in `FilenameOnly.format` includes what `SimpleFormatter.format` reads: `code`, `text`, ...), reads
`kwargs["k"]` and `kwargs.pop("k")` (no default) as a required KEY of the `**kwargs` mapping, hands `**kwargs` on to a base
constructor's own `**kwargs`, and takes the fields a base constructor stored on `self` as the subclass's
(`self.format_options = kwargs["format_options"]` in the base, `self.format_options["headers"]["sort"]` in the subclass).
`passes.py::KeywordDemandPass` (producer `keyword-demands`) proposes the constructor call with those keywords. Classes that
declare their package's re-export (`from .base import X` in `__init__`) are found by `::resolve_annotation_class`.
`passes.py::ArityFeedbackPass` (producer `arity-feedback`) reads a construction's `TypeError: ... missing 2 required positional
arguments: 'option_strings' and 'dest'` (a base the project does not define, reached through `*args`) as a demand on the call: the names
take the values text, `[]`, `0`, `None` in turn (bounded, 48 per goal) and the replay decides (`DebugAction`, `_CallbackAction`: UNSUPPORTED
-> constructed).
Measured, GUUT: `FormatterPlugin`, `HeadersFormatter`, `JSONFormatter` UNSUPPORTED -> SUPPORTED, `FilenameOnly` and
`MergedConfigParser` PARTIAL -> SUPPORTED (see the lane handoff for the table).

**Defaulted parameters ([internal ref]).** A composed call (`DuckTypedCandidatePass`,
`NestedCandidatePass`) used to omit every defaulted parameter. A parameter that
defaults to `None` is "not given", and a class built around a collaborator often
cannot be built without it: flake8's `FileProcessor(filename, options, lines=None)`
reads the file `filename` names unless it is handed `lines=[]`. Each composed call
now comes twice (`passes.py::_default_variants`): once with every default kept and
once with every `None`-defaulted parameter whose type is known given the simplest
value of that type apart from `None` (`Typing.supplied`: the annotation's
`Optional[List[str]]` gives `[]`; a type `typeinfer` reads from the body gives its
own). A default that says something (`tab_size=8`, `flag=False`) is left alone.

## Callbacks

A class that takes a function -- `codetiming.Timers.apply(func, name)`,
`Timer(logger=print)`, flutes' `MapList(func, lst)` -- had no candidate: a
recipe held data literals only, so a `Callable` parameter got `None`. Since
[internal ref] an allowlisted BUILTIN is a literal (`model.py::BUILTIN_CALLABLES`: pure
functions and types such as `len`, `sum`, `str`, `print`, `sorted`; never
`eval`, `exec`, `compile`, `open`, `__import__`, `getattr`/`setattr`,
`input`, `exit`). It is recorded by name (`{"builtin": "len"}`), decoded
only if the name is on the list, handed to the worker as the builtin itself,
and rendered as its name, so a program stays runnable text. A function that
merely shares a builtin's name is not one (identity is checked). Candidates
for a `Callable[[A], R]` come from the declared result
(`passes.py::callables_for_annotation`: `str`/`repr` for `-> str`, `print` for
`-> None`, `sum`/`len`/`max` for a number over a sequence ...), and every one
is still a proposal the replay decides.

## Value domains

Contracts and kinds decide WHAT a parameter may be; Hypothesis GENERATES the values (owner rule). Avengers has one
generator, `tharkun/strategies.py`, and no table of values ([internal ref], [internal ref]).

**What is known of a parameter** is `domains.py::Typing`: a `tharkun.typeinfer` kind, whether it was declared, and the
annotation text. `domains.py::typings(template)` reads a template's:

* an annotated parameter (an annotation, or a PEP 484 type comment: `function_parameters` puts both in `annotation`) is
  DECLARED. `::declared_kinds` hands the texts to `typeinfer.infer_params(..., declared_only=True)` as a `def` built from
  them, so a module-level alias (this module's, or one of a project module a name is imported from) is the expression it
  names, and a quoted forward reference (`'Box'`, `List['Box']`, `'LazyList[T]'`) is the annotation it quotes. Before,
  both read as untyped: dataclasses-json's `from_dict(kvs: Json)` got `None`.
* an unannotated parameter takes the kind `infer_params` reads from its default, its uses in the body and the callee
  annotations it is passed to (sources 2 to 4; the ruling-4 fallback is not used). That kind is a HINT, not a contract:
  a draw keeps the untyped union beside it, and it never refuses a mined literal.
* nothing known: the parameter is drawn from the untyped union (`None`, text, a small int, a bool, short bytes).

**A kind is constraints**: `typeinfer`'s `strategy_constraints` (the table `find_inputs` draws its rows from), cut by
`strategies.py::small` to a small domain (ints in [-32, 32], finite floats in [-32.0, 32.0], text and bytes up to 16,
collections up to 4: the search replays every draw in a child, and small values shrink and replay fast). A `Callable`
keeps no signature in a kind, so a callback is the allowlisted builtins that fit its declared result
(`passes.py::callables_for_annotation`: a `literal`). A class or `Any` is the untyped union, and so is every kind whose
values a recipe cannot hold (`model.py::_literal_record` takes `None`, `bool`, `int`, `float`, `str`, `bytes`, the
collections of those and builtins: a `Decimal`, `Fraction`, complex or `bytearray` part is drawn as nothing is known,
`domains.py::_expressible`). `Optional[X]` leads with
`None`, the cheapest value; a wider union keeps its declared order (`Json` leads with `dict`).

**Every value is a draw from them**, by `strategies.examples` (the one deterministic driver: derandomized, no example
database, and without the constants Hypothesis harvests from every local module, so a draw does not depend on what the
process has imported):

| value | is |
| --- | --- |
| the baseline (`Typing.baseline`; the constructor baseline, `passes.py::_call_from_values`) | the simplest member: `0`, `0.0`, `False`, `""`, `b""`, `[]`, `{}`, `set()`, `()`; `None` for an optional or an unknown; the first member of a union or a `Literal`; a builtin for a callback |
| the truthy value (`::truthy`; the coverage objective and the boundary candidates) | the simplest member of `strategies.py::truthy` of the constraints, which narrows a number to its positive side and a text or collection to one element: `1`, `1.0`, `True`, `"0"`, `[0]`, `{0}`; no literal for a class |
| the falsy value (`::falsy`) | `None` for an optional or an unknown, the baseline when it is falsy, no literal otherwise (a callback, a class) |
| a given default (`::supplied`) | the simplest member apart from `None`: `lines: Optional[List[str]] = None` gives `[]` |
| a search draw (`passes.py::_call_strategy`, `::_drawn_values`) | `make_strategy` / `examples` of the constraints |
| the fill of a bag attribute (`passes.py::_fill`) | the baseline of the type the use names (iterated: `list`, text: `str`, arithmetic: `int`, subscripted: `dict`, tested: `bool`) |

Replaced by this: `passes.py::_baseline_value`, `::_container_type_key`, `::_strategy_for_annotation`,
`::_annotation_accepts`, `::_unwrap_optional_text`, `::_FILLS`; `objectives.py::_truthy_value`, `::_falsy_value`,
`::_unwrap_optional` (and the `FloatArg` name check in them); `typevalues.py::_kind_baseline`; and the text readers of an
annotation in `passes.py::_annotation_class_names` and `::resolve_annotation_class`, which now read the classes a kind
names (`domains.py::object_texts`). The values changed where a table and a generator differ (a truthy text is `"0"`, where
it was `"x"`; a callback parameter's baseline is a builtin, where it was `None`; an `Optional[X]` of several types
leads with its first type), and replay is the judge.

**Still a table, and why.** `passes.py::callables_for_annotation` (a kind keeps no signature, so the result type of a
`Callable` is read from its text; the ruled replacement is the `callable` constraint with `allow` and the builtins'
contract result types); `passes.py::_use_hint` with its method tables (the shape an attribute read needs: typeinfer's
usage rules read names, not attribute chains); `solve/contracts/adapters/avengers.py::_NONEMPTY_SEED` (a contract
adapter, not Avengers'); and `typevalues.py::parses_json`, which proposes `"{}"` and `"[]"` for a parameter handed to a
function named `loads`: it is keyed by NAME, and no JSON contract exists in `solve/contracts` (a contract has no
machine-readable precondition yet), so there is nothing to derive the values from. Ruled, not built.

**Datatypes this made constructible or callable** (each was read wrongly or not at all; GUUT and the held-out set, both
modes unless noted). *Correction* = behaviour that was incorrect; *expansion* = a general rule that covers more types.

| datatype | class or boundary it unblocked | kind |
| --- | --- | --- |
| a quoted forward reference (`'LazyList[T]'`, `List['PrettyAttribute']`, `"Box"`) | GUUT `flutes.iterator:LazyList.LazyListIterator` (constructs, through a nested `LazyList`; unconfined-only, it joins the opt-out list); `pdir.api:PrettyDir` | correction: the quoted text was read as nothing |
| a module-level type alias, imported or local (`Json`, `JsonData`) | GUUT `dataclasses_json.api:DataClassJsonMixin.from_dict` (and `from_json`) | correction: the alias was read as untyped, so the baseline was `None` |
| a required `Callable[...]` parameter | GUUT `flutes.multiproc:DummyPool` `map`, `imap`, `imap_unordered`, `map_async`, `starmap`, `starmap_async`, `gather` | correction: its baseline was `None`, which is never callable; it is now a builtin that fits the declared result |
| a parameter that defaults to `None` and has a known type (`Optional[List[str]] = None`) | GUUT `flake8.processor:FileProcessor` (constructs, 17 boundaries return), `pdir.api:PrettyDir.own` | expansion: a composed call supplies the type's simplest value (`lines=[]`) |
| an attribute bag attribute the class stores and later multiplies or compares | GUUT `flake8.processor:FileProcessor` (`Namespace(indent_size=0, ...)`) | expansion: an all-`0` bag variant |
| `typing.Sequence[...]` and `Optional[Sequence[...]]` (an ABC, not `list`) | held-out `mimesis.providers.choice:Choice.__call__`, `typesystem.fields:Choice` (constructor and `validate`) | expansion: the kind table knows what the old startswith tables did not (`Sequence`, `Literal`, `frozenset`, a non-optional `Union`) |
| an unannotated parameter used as a list, text or number in the body | held-out `pytutils.lazy.lazy_import:ScopeReplacer` (constructs) | expansion: the kind `typeinfer` reads from the uses, as a hint |

Not a gain but a guard: `Decimal`, `Fraction`, complex and `bytearray` have kinds that a recipe cannot hold; the first
version of this change drew them and lost 9 of 22 validated classes of the held-out `pypara`, so they are drawn as
nothing is known (`domains.py::_expressible`, `tests/test_avengers_domains.py::test_a_type_whose_values_are_not_recipe_literals_is_drawn_as_nothing_is_known`).

**Measured** ([internal ref]; CPython 3.14.4, built-in passes, 20 s per class, main f07a69fe against this change; unconfined /
confined with the opt-out list). GUUT, 206 class rows of the 207 goals (`flutils.packages:_VersionInfo` runs at the
harness's 120 s job cap): classes validated 177 -> 179 / 178 -> 179, boundaries that returned 226 -> 252 of 410 / 222 -> 248 of 404,
GUUT mutants in returning boundaries 1,136 -> 1,252 / 1,134 -> 1,250, `__init__` mutants of validated classes 177 -> 188 /
185 -> 187. Held-out, the 157 classes of nine Pynguin-corpus projects that are not GUUT, picked before the change was
measured on them (sty, pyMonet, pypara, typesystem, pytutils, isort, mimesis, py-backwards, python-semantic-release; the
25 public classes with the most sampled mutants of each; one measurement of them found the `Decimal` fault above): validated 141 -> 142 /
135 -> 136, boundaries 408 -> 410 / 393 -> 395, mutants in returning boundaries 1,235 -> 1,262 / 1,175 -> 1,202. No project
lost a class or a boundary in any run (a class can flip between runs of one commit: `flake8.main.options:JobsArgument`).
The GUUT gain is mostly one class (`FileProcessor`, +97 of +116 mutants); the held-out gain is smaller (+27) and comes
from `Sequence`, inference and defaults. A search is slower where more recipes validate: `pypara.accounting.ledger:LedgerEntry`
36 s -> 55 s (a nested `Ledger` now resolves and is probed).

**Consumes and publishes (the shared knowledge base).** Nothing here adds an evidence kind or reads the store. The passes
it changes keep their edges: `ConstructorTemplate` evidence (`constructor-template`, with its goal) is consumed by
`BaselineCandidatePass`, `HypothesisCandidatePass`, `NestedCandidatePass` and `DuckTypedCandidatePass`, which publish
`candidate-recipe` (now also the composed calls that give a `None`-defaulted parameter its type's value, and the all-`0`
bag); `CoverageSetupPass` consumes `validated-recipe` and publishes `coverage-observation` and `candidate-recipe`. What a
parameter's type is is READ from the source on demand (`domains.py::typings`, cached per file and template in the process) and
is not published: typeinfer facts as knowledge-base facts are ruled, not built (the contracts backbone). Every candidate is
still only a proposal; promotion is the fresh replay.

**Files and how they relate.** `avengers/domains.py` imports `tharkun/strategies.py` (`small`, `truthy`, `examples`),
the `tharkun.typeinfer` facade (`infer_params`, `annotation_kind`, `strategy_constraints`, `drawable`) and
`solve/frontend/import_resolver.py::load_class_def`; lazily `passes.py::callables_for_annotation`. `passes.py`,
`objectives.py` and `typevalues.py` import it. It reads a template; it never runs the target.

## Type values, global references and ready-made receivers

A recipe is data, so a value that is a class, an enum member, a module constant or a generated dataclass cannot be
pickled into it: it is NAMED, and the worker rebuilds it ([internal ref]). `StatefulPool(pool_class=None)` is
"'NoneType' object is not callable", and dataclasses-json's `handle_from_dict(cls, kvs)` is "must be called with a
dataclass type or instance", however many strings and integers are tried.

**The vocabulary** is five frozen records in `values.py` (`::GlobalReference`, `::Subscript`, `::DataclassType`,
`::Instance`, `::FactoryCall`), each one tag of the closed recipe literal (`model.py::_literal_record`, `::_literal_value`,
`::_render_argument`, `::_bind_nested`, which also emit the `import` lines a rendered program needs):

| tag | rebuilt by the worker as |
| --- | --- |
| `{"global": "pkg.mod:Name"}` | `Name` (dotted: `Undefined.INCLUDE`) of the imported module `pkg.mod` |
| `{"subscript": [origin, [arg, ...]]}` | `origin[arg, ...]`: `typing.Optional[str]`, `list[float]` |
| `{"dataclass": {"name", "fields", "bases"}}` | `dataclasses.make_dataclass(name, fields, bases=bases)`; a field is `[name, type]` or `[name, type, default]` |
| `{"instance": {"callee", "args", "kwargs"}}` | `callee(*args, **kwargs)`; the callee must be a class |
| `{"call": {"callee", "args", "kwargs"}}` | `function(*args, **kwargs)`; the callee is a standard-library function the contract registry admits (`ast.parse("x = 1")`), never a project name (`values.py::resolve_function`) |

`_worker.py::_argument` builds them. **What may be named is closed**: the goal's own project package (and each nested
goal's), the packages outside the standard library that the goal's module imports ([internal ref]: they are already
loaded in the child), the `typing` forms a type expression is subscripted from (`values.py::TYPE_FORMS`: `Optional`, `Union`), and the
part of the standard library that **the contract registry clears**, and never a dunder attribute. There is no list of
modules here. `values.py::stdlib_refusal` asks `solve/contracts/adapters/avengers.py::stdlib_admission` for the qualname
(`module.name`): a callable is admitted when its contract is tested and declares no file, process, network, clock,
environment or randomness effect (`types.py::Effects`: `io`, `randomness`, `clock_or_env`, `deterministic`), and refused
otherwise. `values.py::global_refusal` decides; anything else raises `values.py::RefusedGlobal` in the child, and the
replay's `detail` carries the named reason (`module-outside-project:requests`, `stdlib-no-contract:os.system`,
`stdlib-effect-randomness:random.random`, `stdlib-untested-contract:open`, `dunder-attribute:...`, `unresolved-module:...`,
`unresolved-name:...`, `call-of-project-name:...`). Nothing is imported to decide it, and a record that names nothing
well-formed (`"os"`, `"a b:c"`) is refused when it is decoded. A name the registry does not cover is not "denied": it is
unknown, and the fix is the family's contracts (`solve/contracts/rules_factories.py`), not an entry in Avengers.

**A ready-made receiver.** `model.py::ConstructorCall.receiver` hands the worker an object INSTEAD of
`Class(*args, **kwargs)`: a module-level instance of the goal (a global), an instance of a concrete subclass of an
abstract goal (a `NestedObject`), an instance of a generated dataclass that subclasses the goal (an `Instance`).
`_worker.py::_build` checks it is an instance of the goal class (`receiver-not-an-instance`). It lives on the constructor
call, not the recipe, because every pass that extends a candidate copies `recipe.constructor`; it enters the record
only when set, so no earlier recipe key changes.

**The proposals** are `typevalues.py`, read statically from the project's source (nothing is imported or run); the
replay is still the only judge:

* `::class_options`: `Type[X]` (also `type[X]`, `Optional[Type[X]]`, a quoted or `ForwardRef` name, a `TypeVar`'s
  `bound=`) proposes X when concrete, then X's subclasses found by name in the project package (`::subclasses`),
  concrete ones first. `::_class_slots` reads WHERE the annotation takes a class, so a class value is found in any union
  arm (`Union[Field, Type[Schema]]`), in a module alias (`Kinds = Optional[Sequence[Type["Base"]]]`), under
  `Type[Optional[X]]`, and inside a container (`Optional[Iterable[Type[X]]]`, `Sequence[type]`, `Tuple[...]`, `Set[...]`:
  the value is that container holding one class). Three more cases are general corrections ([internal ref]):
  * **nothing constrains the class** (`Type[Any]`, a bare `type`, `Type`, an unbound `TypeVar`): `::generic_classes`, up to
    two concrete classes the goal's own module defines, then `int` and `str` (the recipe's builtin tag);
  * **the class is not the project's** (the standard library, a package, a C type): `::foreign_options` names it when a
    worker may follow it, then the project's own subclasses of it, found by the module and name each base class is imported
    from (`class HttpProtocol(asyncio.Protocol)` for `Type[asyncio.Protocol]`; a `typing.Protocol` subclass is not one);
  * **the class lives in another package the goal's module imports** (`from flake8.formatting.base import BaseFormatter`
    for `Type[BaseFormatter]` when the goal is a probe holder or any class outside that package): it may be named, because
    importing the goal already ran its package. `::_followable` (parent, static: the module's imports) and
    `_worker.py::_load_class` (child: `values.py::namespace_modules` of the loaded module) compute the same set
    (`values.py::imported_roots`: top-level packages outside the standard library). The standard library is NOT in it: it
    stays admitted by contract only.
* `::treats_as_type`: an UNANNOTATED (or `Any`) parameter that the body, or a project function it is handed to (three hops),
  uses as a class or a typing form -- `p.__origin__` / `__args__` / `__mro__` ..., `hasattr(p, "__args__")`,
  `issubclass(p, ...)`, `isinstance(p, type)`, `inspect.isclass`, `get_origin` / `get_args` / `get_type_hints` -- or one named
  `cls` / `klass` / `type_`, gets `::type_object_options`: two classes (`::generic_classes`) and two typing forms built from
  `typeinfer` kinds (`list[int]`, `Optional[str]`). `::needs_dataclass` and this share `::_reaches`, the call-graph walk.
* `::package_arguments`: what the package itself passes where the goal is constructed. `StatefulPool(pool_class, ...)`
  after `pool_class = DummyPool` (in every branch that binds the name) proposes `DummyPool`.
* `::enum_options`: an annotation naming a project `Enum` proposes its members (`Undefined.INCLUDE`). `ClassDescription`
  members whose value is a class are named the same way (`_worker.py::_describe`).
* `::needs_dataclass`: a parameter handed to `dataclasses.fields` / `is_dataclass` / `asdict` / `astuple` / `replace`, or
  asked for `__dataclass_fields__`, directly or through functions of the project (found by name, three hops), gets
  dataclass TYPES (a package `@dataclass` first, then `::dataclass_shapes`) when it is type-like (`cls`, `Type[...]`),
  else instances of them. The generated types are `make_dataclass` values whose field types are `typeinfer` kinds
  (`::_menu`, `::_kind_type`): one int field; a mixed record of an optional text, a list and a mapping; and, when the
  goal's module defines an `Optional[...]` alias (dataclasses-json's `CatchAll`), a record with a field of that type.
  Every field has a default, so `from_dict({})` builds one. A parameter handed to a function named `loads` is proposed
  `"{}"` and `"[]"`: a rule keyed by name (see "Value domains": no JSON contract exists to derive it from).
* `::abstract_options`: a parameter annotated by an abstract class or a `typing.Protocol` of the project gets its
  concrete subclasses, and for a Protocol the classes that have every public method it declares whatever their bases
  (`::implementers`, found by structure), each built by its own baseline constructor. `NestedCandidatePass` cannot: a
  nested object is built by its constructor, never from a receiver, and the abstract class does not construct. A Protocol
  goal is also a receiver by structure: `_worker.py::_satisfies` accepts an object with the protocol's public members where
  `isinstance` would raise.
* the receiver (`TypeValuePass._receiver_calls`): a module-level `NAME = Goal(...)` (`::module_instances`); for an
  abstract goal, concrete subclasses (the goal is described by a child and `_wire.ClassDescription.abstract`, computed
  since [internal ref] and read here for the first time, decides); for a goal whose methods need `self` to be a dataclass
  (`::class_needs_dataclass`), an instance of a generated dataclass that subclasses it (`DataClassJsonMixin`).

`passes.py::NestedCandidatePass` does not build an instance for `Type[X]` ([internal ref]): `_annotation_class_names` and
`resolve_annotation_class` skip the inside of `Type[...]` (`::_instance_nodes`), because that parameter takes the class X,
not an object of it; the instances it proposed there either failed the parameter's own `issubclass` check or passed a
constructor that never looked.

`typevalues.py::TypeValuePass` is the one new pass (constructor and receiver candidates, after `DuckTypedCandidatePass`);
`passes.py::method_call_candidates` calls `::method_proposals` for a boundary's own parameters, right after the
baseline call. `_worker.py::_hypothesis_arguments` records a `st.from_type` draw of an enum member, a class or a package
dataclass through the same tags (`::_expressible`: only when the name resolves back to the very object and may be
followed; a local class is not recorded, as before).

**Confinement ([internal ref]).** A global makes the worker import a module, so `executor.py::_declared_sources` walks the new
tags (and a constructor's receiver) and declares the file each global's module lives in
(`values.py::global_source_path`: path arithmetic from the source the value sits under; the standard library and a
foreign module declare nothing). `tests/test_avengers_type_values.py` pins both a declared walk and a confined replay.

**Files and how they relate.** `avengers/values.py` (stdlib only; imports `solve/loader.py` for the layout) <- `model.py`
(tags, `ConstructorCall.receiver`) <- `_worker.py` (builds, resolves, records draws) and `executor.py`
(`_declared_sources`). `avengers/typevalues.py` imports `passes.py` (templates, class resolution), `engine.py`,
`executor.py`, `model.py`, `standins.py` (`_import_map`, which also feeds `::foreign_options`) and `values.py`
(`imported_roots`; `_worker.py` calls it with `namespace_modules`); `passes.py` imports it lazily, once; `api.py::default_passes` registers the
pass. It reads the generated dataclasses' field kinds as `typeinfer` kinds and their defaults from `domains.py` (see "Value domains").

**Measured** (1226, CPython 3.14.4, the 10 GUUT projects, 207 classes, built-ins only, 20 s per class; raw rows and the
per-project tables are [internal ref]): classes
validated 175 -> 179 unconfined (177 -> 178 confined; the unconfined extra is a Hypothesis flip unrelated to 1226), boundaries
that returned 215 -> 226, GUUT mutants in returning boundaries 1,048 -> 1,136 and in `__init__` of validated classes
126 -> 177 (confined: the same). dataclasses-json: mutants in returning boundaries 56 -> 144 of the 251 class mutants
(`DataClassJsonMixin` to_dict/to_json/schema/from_json, `_CatchAllUndefinedParameters.handle_from_dict`/`handle_dump`,
`_RaiseUndefinedParameters`, `_IgnoreUndefinedParameters`, the abstract `_UndefinedParameterAction`, `Undefined`); flutes:
`StatefulPool` constructs with `pool_class` and `state_class` globals. No project lost a class or a boundary, in either
mode. `flutes.multiproc:StatefulPool` joins `benchmarks/avengers/unconfined.json` (66 entries): it now validates
unconfined, and confined every candidate is refused for the module's own `/dev/shm` reason, as for its 17 siblings.

**Measured (1306, corrections to the above).** The init re-cut (the parameter-slot probe: a one-parameter holder class whose
constructor checks the value is a T) listed 15 class-valued types, 7 GUUT (19 parameters) and 8 held-out (21), as not
constructible. Against main 0b4f1a19, same machine, both arms at once, `measure.py` built-ins 20 s per class:

| type | parameters | before | after (confined) | after (unconfined) | correction |
| --- | --- | --- | --- | --- | --- |
| GUUT `Type[Any]` (apimd, pdir2) | 6 | no | yes | yes | unconstrained class |
| GUUT `Type[TypeVar A]` / `[TypeVar T]` | 1 + 1 | no | yes | yes | unconstrained class |
| GUUT `Type[BaseFormatter]` | 1 | no | yes | yes | imported package; no instance |
| GUUT `Type[PoolState]` / `Type[PoolType]` | 1 + 1 | no | no | yes | imported package; confined: module's own `/dev/shm` (below) |
| GUUT "type or typing object" (untyped) | 8 | no | yes | yes | parameter used as a class |
| held-out `Type[Any]` (isort, typesystem) | 4 | no | yes | yes | unconstrained class |
| held-out `Type[BaseFinder]` / `[BaseProvider]` / `[RenderType]` / `[Schema]` | 1 / 1 / 2 / 8 | no | yes | yes | container; imported package; no instance |
| held-out `Type[BaseRenderer]` / `[Request]` (sanic) | 1 / 1 | not measurable | no | yes | imported package; confined: sanic's own `importlib.metadata` stat |
| held-out `Type[asyncio.Protocol]` | 3 | no | no | no | a project subclass is found; the holder has none, and the class itself is not cleared by the registry |

So 17 of 19 GUUT and 16 of 21 held-out parameters are served confined, 19 and 18 unconfined. **The confined replay does
mask the real result** for 4 of them (`PoolState`, `PoolType`, sanic's two): `tracewhy`-style flag recording shows the first
refusal is the library's own module code (`flutes.multiproc` calls `multiprocessing.Value` at import, which `mkstemp`s in
`/dev/shm`; `sanic/__init__.py` reads `importlib.metadata.version`, which stats the `sys.path` entries) before any recipe
value is built; the audit saw it as `environment|unmodelled`, and the value that fails unconfined is `None`/an instance.
That is the sandbox's (1275), not the pass's, and the opt-out list is not extended (it exists to be emptied). The
in-package places (`Application.make_formatter`, `StatefulPool(pool_class, state_class)`, `Generic.add_provider`,
`Register.set_renderfunc`, `Form(schema)`) were already served by the pass: the probe's holder sat outside the package of
the class it asked for, which the "imported package" correction closes for any goal.

Whole-class effect (A/B, 207 GUUT classes and 167 held-out classes, confined with the opt-out list and unconfined): no
class and no boundary lost anywhere (differences are the 120 s job timeouts under load and the Hypothesis-flipping
`flake8.main.options:JobsArgument`, which flips in both arms). Held-out gains: `_pytest.raises:RaisesGroup` (validated, 12
recipes, all from this pass: an exception class for a `Type[BaseException]` parameter), `_pytest._code.source:Source`
(+`getstatement`, `getstatementrange`). Fewer nested-instance recipes for a `Type[X]` parameter
(`typesystem.forms:Form`: 23 to 5 validated recipes in a single-class probe before the merge with main; the dropped ones passed an instance where a class is annotated).

**Not built here** (and measured as still blocking): `HTTPieArgumentParser`'s module-level `parser` is a receiver now,
but `parse_args` needs an `Environment` argument (method-argument objects). A queue or an in-memory stream for a
parameter the standard library types is built by the stand-ins below ([internal ref]). A parameter annotated by an alias
(`Json`) was still read as untyped here; "Value domains" reads it now ([internal ref]).

## Values for the parameters of a function ([internal ref])

`synthesize_object` asks `typevalues.py::parameter_options` for the parameters of a class's constructor and boundaries. A caller
with a FUNCTION to call (an anchor's function in a library, an entry of `find_inputs`) has no class goal and the same question.
`avengers/function_values.py::function_options(source_path, qualname)` asks it of the function `f` or `Class.method` itself,
statically (the file is parsed, nothing is imported or run), and `::build_argument(value, source_path, namespace)` rebuilds one
proposed value in the caller's own process through `_worker.py::_argument`, under the same closed vocabulary and refusals
(`values.py::global_refusal`). Nothing is a new rule. Three additions, each only for this caller (`parameter_options(wide=True)`;
the constructor and boundary menu is unchanged, `PER_PARAMETER` stays 4):

* the menu is `WIDE_PER_PARAMETER` (12) long, with `::wide_type_forms` (a mapping `dict[str, int]` and a union `int | str`, which
  `Optional` and `list` alone never make a function branch on) and `Optional` of the first dataclass type;
* the structural stand-ins (`standins.py::stand_in_options(structural=True)`: a `datetime` for a parameter read only through
  `.timestamp()`) follow the typed ones, with their trust tag, as `::structural_method_proposals` does for a boundary;
* the **classinfo role** (all callers): `typevalues.py::_is_type_use` takes a parameter that is the second argument of
  `isinstance` / `issubclass` as a class (`isinstance(o, t)` raises for any `t` that is not a class or a tuple of them).

*Files:* `avengers/function_values.py` imports `typevalues.py` (`parameter_options`), `_worker.py` (`_argument`, `_roots`),
`values.py` (`imported_roots`, `namespace_modules`) and `solve/frontend/constructor.py` (`ConstructorParameter`); `avengers/__init__.py`
re-exports `function_options` and `build_argument`. Tests: `tests/test_avengers_function_values.py`. *Measured* (dataclasses-json, 50
synthesized anchors, the fixed pool of the measuring harness alone 21 with a target; its `cls` parameter kept 27; with these
values 38; python-string-utils 60 of 62 either way; nothing lost): the harness is the lane-1386 work copy of `score-work`.
*Not built* (kinds Avengers cannot build, with the anchors they block): a class configured by the project's own decorator
(`cls.dataclass_json_config['undefined']`: a project function applied to a generated type), and an `enum.Enum`
class (no contract for the functional `enum.Enum`, `values.py::stdlib_refusal`; a `typing.NewType` has one since [internal ref] and is in the wide menu), a stdlib instance named only by
a callee's `isinstance` chain (`_encode_json_type`'s `default(value)`), a stdlib class compared against in a callee
(`issubclass(field_type, datetime)`), and a function-valued observation (no value to compare).

## Stand-ins for parameters the standard library types ([internal ref])

A parameter annotated `mp.Queue[Event]`, `typing.IO`, `io.RawIOBase` or `ast.AST`, or one the code only calls
`.seek()` on, names no class of the project for `passes.py::NestedCandidatePass` to build, and no literal reaches it:
`Proxy(queue=None)` is "'NoneType' object has no attribute 'put_nowait'" for every method. `standins.py` proposes a
value by three general rules; none of them names a type the subjects happen to need.

1. **What the parameter needs** (`standins.py::need_of`). The standard-library types its annotation names, resolved
   through the module's own imports (`import multiprocessing as mp` makes `mp.Queue` the type `multiprocessing:Queue`;
   quotes, `Optional`, `Union`, `|` and subscripts are looked through; an import under `if TYPE_CHECKING` counts), and the
   attributes the code reads off the parameter: `p.seek(...)` in the function, or `self.queue.put_nowait(...)` in another
   method of the class (`passes.py::parameter_attribute_uses`), plus the special names it relies on (`for x in p`,
   `with p`, `p[k]`, `len(p)`, and dunder attributes such as `p.__dict__`, which hold of every object), which only filter.
   A need that a plain literal already has every attribute of (`x.get`, `x.startswith`, `x.append`) gets nothing.
2. **Which concrete value satisfies it** (`standins.py::candidates`). The candidates are the value factories the
   contract registry admits (`solve/contracts/adapters/avengers.py::factories`): constructors and functions with a
   tested contract that declares no file, process, network, clock, environment or randomness effect, each with the class
   its call returns (`rules_factories.py`: `ast.parse` returns an `ast.Module`). A factory satisfies the need by, in
   order of strength, being a **subclass** of the needed type (`issubclass`, so abstract-base-class registration and
   `__subclasshook__` count: `os.PathLike` is satisfied by `pathlib.Path`), having the **name** of the annotation
   (`multiprocessing.Queue` is a function; `queue.Queue` is the class of that name) -- both **typed** matches -- or, by
   attribute names alone, its instances having **every attribute the code reads** (a **structural** match). A factory
   that lacks an attribute the code reads is dropped whatever else it matches. A structural match may be an accident (a
   `Path` for a parameter read through `.name`), so it is kept for coverage but tagged `trust=inferred-structural`
   (`values.py::TRUST_STRUCTURAL`, provenance on `Instance`/`FactoryCall`: not in the recipe record, equality or hash, so a
   typed proposal for the same call wins the dedupe), proposed only **after** every typed value, and counted apart in a
   measurement.
   Ranking: strength, then the fewest surplus public attributes, then the fewest arguments. A need a plain literal already
   satisfies (`Iterable`, `Mapping`, `Number`, `List[int]`) gets nothing: the baseline and Hypothesis candidates cover it.
   The type object is looked up only in the modules the registry imports statically (`rules_factories.py::type_named`:
   the factories' modules and `typing`, `collections.abc`, `numbers`, `os`, `abc`), never imported: a result that depended
   on what the parent had loaded would make a search irreproducible. A type whose module the registry does not hold is
   matched by name and attributes alone.
3. **How it is built** (`standins.py::value_of`). By a `values.py` record, an `Instance` of a class or a `FactoryCall`
   of a function, with the literal arguments the contract lists as valid (`io.BytesIO()` and `io.BytesIO(b"a\nb\n")`,
   `ast.parse("x = 1\n")`). The only per-type data is the contract's list of valid calls.

Every value is a proposal, and the fresh child replays it and refuses with a named reason any name the registry does not
admit. The typed ones enter through `typevalues.py::parameter_options` (so `TypeValuePass` and the boundary hook
`method_proposals` both see them, right after the type values of the parameter). The structural ones are tried after:
constructors by `typevalues.py::StructuralStandInPass` (registered after `TypeValuePass` in `api.py::default_passes`; its own
producer, `standins-structural`; a candidate combines one structural value with the typed options of the other
parameters), method arguments by `typevalues.py::structural_method_proposals`, which `passes.py::method_call_candidates`
extends after the guard values (that is the one hook line `passes.py` carries for this).

**The contract family** (`solve/contracts/rules_factories.py::StdlibFactory`, one parametrized rule, kind `stdlib_factory`):
`io.StringIO`/`BytesIO`, the four `queue` queues, `collections.OrderedDict`/`Counter`/`deque`, `ast.parse`,
`decimal.Decimal`, `fractions.Fraction`, `datetime.date`/`datetime`/`time`/`timedelta`, `uuid.UUID`,
`pathlib.PurePath`/`Path`. All seven raise claims are `unknown` (a constructor raises on a bad argument and no engine
reads a factory's raise claim); the returned class, the absence of effects and the valid calls are tested
(`tests/contracts/test_factories.py`: every listed call returns an instance of the declared class, opens no file and starts
no process or connection under an audit hook, and a call with nine arguments raises `TypeError`). The Avengers cell of
the engine audit is `C` for each (`AvengersVisitor.visit_stdlib_factory`). Adding a type is one row, then
`python -m tests.contracts._frozen write --only <qualname>`; Avengers needs no change.

**Confinement ([internal ref]).** A stand-in is standard library: `executor.py::_declared_sources` declares nothing for it, and a
confined replay of each factory is a test.

**Files and how they relate.** `solve/contracts/rules_factories.py` (the rows; stdlib only) <- registered by
`solve/contracts/registry.py::load_all`; `solve/contracts/visitors.py::AvengersVisitor.visit_stdlib_factory` <-
`solve/contracts/adapters/avengers.py` (`::stdlib_admission` decides what may be named, `::factories` lists what may be
built). `avengers/values.py` (`::stdlib_refusal`, `::FactoryCall`, `::resolve_function`) calls `stdlib_admission`;
`model.py` and `_worker.py` carry and build the `call` tag; `avengers/standins.py` calls `factories`, reads the project's
source through `typevalues.py::_module_of` and `passes.py::parameter_attribute_uses`, and is called by
`typevalues.py::parameter_options` (typed) and `::StructuralStandInPass` / `::structural_method_proposals` (structural).

**Measured** ([internal ref], CPython 3.14.4, built-in passes, 20 s per class, six jobs per arm; rows, tables and the per-class diffs are
[internal ref]). Base = development main 1eb95a9b, which
already has the typeinfer kinds of [internal ref]; after = the merge of this branch with it. *Structural-only* = classes validated
only through, and boundaries that returned only through, a stand-in tagged `trust=inferred-structural`.

| set | mode | validated | boundaries that returned | GUUT mutants in returning boundaries | structural-only (classes / boundaries) |
| --- | --- | --- | --- | --- | --- |
| GUUT, 207 classes | unconfined | 180 -> 180 | 252 -> 258 of 410 | 1,252 -> 1,350 | 0 / 2 |
| GUUT | confined + opt-outs | 180 -> 181 (the `JobsArgument` Hypothesis flip) | 252 -> 258 | 1,252 -> 1,350 | 0 / 2 |
| held out, 167 classes of 8 libraries that are not GUUT | unconfined | 134 -> 140 | 239 -> 255 of 691 | (no mutant counts) | 2 / 8 |
| held out | confined | 114 -> 120 | 214 -> 229 | | 2 / 8 |

On the main before [internal ref] (f07a69fe) the same gains were GUUT 226 -> 232 boundaries and 1,136 -> 1,234 mutants, held out 134 -> 140
and 239 -> 255 / 114 -> 120 and 216 -> 231: [internal ref] made none of them redundant and none bigger. With every backend on (Pynguin,
`st.from_type`, Ghostwriter, 60 s; run on the head before [internal ref] only) GUUT was 179 -> 179 validated, 226 -> 232 boundaries,
1,136 -> 1,234 mutants. Main with every backend on equals main with built-ins only on GUUT (179 validated, 226 boundaries), so the gain is the
same on top of all parts. No project lost a class or a boundary beyond the flip. The held-out libraries (`packaging`,
`pluggy`, `iniconfig`, `sortedcontainers`, `execnet`, `yaml`, `pygments`, `_pytest`) were fixed by a rule before any run
(`scripts/ab/heldout_inventory.py`: every class defined in the package that has a public method or a required
constructor parameter, exceptions and enums excluded, at most 30 per package by an even stride); their per-class results
were not read while the stand-ins were designed (what was read when, and the one rule a held-out program led to, is in
`1280.md`). Switching either mechanism off (measured on the head before the dunder rule of rule 1):
the stand-ins alone give held-out +2 classes and +9 boundaries, the project abstract/Protocol resolution alone +4 and +8.

**Datatypes this adds, by name**, each marked as a *correction* (a component that mishandled the type is made right) or a
*general expansion* (a type no component reached, now reached by the rules above):

| datatype | where it was measured | kind |
| --- | --- | --- |
| `multiprocessing.Queue[...]` parameter, built as a `queue` queue | GUUT flutes `ProgressBarManager.Proxy` (4 boundaries) | general expansion (name and attribute evidence; the annotation is a bound-method factory, which `resolve_annotation_class` cannot resolve) |
| `typing.IO` / a file-like read with `seek`, `tell`, `read`, built as `io.BytesIO` or `io.StringIO` (structural: `IO` is no class a factory subclasses) | GUUT flutes `_ReverseReadlineFile.generator`; held out execnet `Popen2IO`, `Message.to_io` | correction (the nested path built `typing.IO()`, an object with no working method) and expansion |
| `pathlib.PurePath` / `Path` for a parameter read through `.name` / `.parent` | GUUT flutils `_CherryPickingLoader.create_module`; held out pytest `NodeKeywords` | general expansion, by attribute names only (a structural match, not a semantic one) |
| `datetime.date` / `datetime` for a parameter read through `.isoformat` | held out yaml `SafeRepresenter.represent_date`, `represent_datetime` | general expansion (structural) |
| a project abstract class (`execnet.ExecModel`, an ABC) as a parameter type, built as a concrete subclass | held out execnet `Reply`, `WorkerPool`, `SocketIO` | correction (`NestedCandidatePass` cannot compose one: a nested object is never built from a receiver, so such a parameter only ever got `None`) |
| a project `typing.Protocol` (`execnet.ReadIO`, `WriteIO`) as a parameter type or as the goal, built as a class that has its methods | held out execnet `ReadIO`, `Message.to_io` | correction (a Protocol goal had no receiver: `inspect.isabstract` is false for it and `isinstance` raises on a non-runtime Protocol) and expansion (structural implementers) |
| `ast.AST` (`ast.parse`), `decimal.Decimal`, `fractions.Fraction`, `uuid.UUID`, `collections.OrderedDict`/`Counter`/`deque`, `datetime.time`/`timedelta`, `queue` queues other than the first | no class of either set gained | general expansion, tested on a synthetic package only (`tests/test_avengers_standins.py`) |
| the standard-library names a recipe may carry | every replay | correction: the denylist named modules (`DENIED_STDLIB`) and so let every name outside it through (`tempfile`, `webbrowser`, `urllib`, ...); the contract-derived allowlist refuses what the registry does not clear |

**Knowledge base.** What this consumes: the `GOAL` and `TEMPLATE` evidence (`ObjectGoal`, `ConstructorTemplate`) that
`TypeValuePass` already reads (`typevalues.py::TypeValuePass.run`), the project's source (static), and, outside the
knowledge base, the contract registry (read-only: `solve/contracts/adapters/avengers.py::factories`, `::stdlib_admission`).
What this publishes: `CANDIDATE` evidence only, `ConstructionRecipe`s whose parameter values are `Instance`, `FactoryCall`
or `NestedObject` records, each derived from the `TEMPLATE` it was proposed from, by producer `type-values` (typed) or
`standins-structural` (`trust=inferred-structural`). `ReplayPass` is unchanged and publishes
`REPLAY` / `VALIDATED` as before; a validated recipe may now hold a stand-in, and `NestedCandidatePass._nested_recipes`
already reads such a recipe back as a recipe for that class, so a class built once is reusable as a nested argument. A refused
name appears only as the named reason in a `REPLAY` outcome's `detail`. Not built: contract evidence flowing back from replay (a validated stand-in
call confirming or withdrawing a `ReturnSpec`), and a knowledge-base fact for "this type has no factory".

**Not built here:** a project's own factory (`def make_parser() -> argparse.ArgumentParser`) is not called: a
`FactoryCall` never names a project function, which would run project code outside the replay's phases. An untyped
parameter that is only passed on (`FlakesChecker(tree, ...)` hands `tree` to a base class) has no annotation and no
attribute read of its own; `duck.py` ([internal ref]) follows it into the project's own constructors and functions, two calls deep,
but not into a class outside the project's source. Third-party types (`requests.Response`,
`tqdm.tqdm`) are outside the project and the standard library by design.

## Collaborator setup state ([internal ref])

An object that **constructs** is not an object that **works** (`works.py::classify`: the call returned and ran past its entry
guards). The init audit (`init/INIT.md` section 5) found the commonest gap between the two to be a receiver whose own state is
`None` at the call: `'NoneType' object has no attribute 'noqa_line_for'`, `'... has no attribute 'local_configs'`,
`'... 'iter_content'`. 25 GUUT types (74 calls) and 24 held-out types (64 calls) carry it. Three causes, three general rules;
none names a type the subjects happen to need.

* The constructor was given `None` for a **collaborator** the class asks things of (`MergedConfigParser(config_finder)` calls
  `config_finder.local_configs()`; `legacy.StyleGuide(application)` reads `application.file_checker_manager` and calls
  `application.run_checks(paths)`). A real one is often out of reach (flake8's `Application.initialize([])` took 109 s wall (36 s of CPU; one run under load) against a replay's 5 s, and
  `requests.Response` is not the project's), and a literal is never callable.
* The constructor leaves a field `None` and **a setup call** establishes it (`Application.initialize(argv)` writes nothing itself
  and calls the methods that write everything), and nothing proposed that call.
* A **method argument** is a collaborator too (`Key.create_from(error)`, `HTTPBasicAuth.__call__(request)`).

**The value: `ConstFn`.** A recipe is data, so a callable is a tag: `{"const_fn": value}` is `lambda *args, **kwargs: value`
(`values.py::ConstFn`; `model.py::_literal_record`, `::_literal_value`, `::_render_argument`, `::_bind_nested`; built by
`_worker.py::_argument`; walked by `executor.py::_declared_sources`). The value is built once in the child and every call returns it,
whatever the arguments. A rendered program says what it is (`write=(lambda *args, **kwargs: None)`).

**Rule 1: what the code requires** (`duck.py::Demand`, `::constructor_demand`, `::function_demand`). A tree rooted at the
parameter, read from source and never run: the attributes read off it, the ones CALLED (`called`, and what the call's result is then
used for: `result`), the literal keys subscripted, the data shape of a value that is only iterated, compared, indexed or tested
(`passes.py::_use_hint`'s shapes). It follows the value through `self.x = p` (every other method of the class, and of the bases it
inherits its constructor from, reads `self.x`), through a local alias (`x = self.options`, `copy.deepcopy(self.options)`), through
the result of a call on it (`plugin = entry_point.load()` then `plugin(...)` or `callable(plugin)`), through a container written into
(`request.headers["k"] = v` needs `request.headers`), and through a call that passes it on into the project's own constructor or
function (`DecisionEngine(options)`, `self._check(options)`, `super().__init__(options)`, `utils.parse(options)`),
`duck.py::MAX_HOPS` = 2 calls deep, never into code outside the project's source. `p.name.startswith(x)` says `name` is text, not that
it has an attribute called `startswith` (`passes.py`'s method tables: a name a builtin value has counts as an attribute only beside
one no builtin has: `stats.count(x)` next to `stats.record(y)`).

**Rule 2: what a stand-in offers** (`duck.py::realize`, `::variants`). An object is a `SimpleNamespace` (the `argparse.Namespace` the
annotation names, at the top) whose attributes are the stand-ins of what was asked of them: a PROJECTION (`application.guide`,
`application.options`) is a namespace in turn, up to `model.py::MAX_NESTING` deep; a called attribute is a `ConstFn` returning the
stand-in for its result (`stats.count(x) > 3` returns `0`); a data attribute takes the baseline of its shape (`[]`, `""`, `0`, `{}`,
`False`, else `None`). Five variants, most specific first: every leaf by its use; objects left unexpanded (the pre-[internal ref] bag);
every leaf `[]`; every leaf `0`; every attribute `None`. A value the class uses only as DATA through `self.x` in another method
(`self.value.items()`, which the constructor's own body never shows) is the one value its use names.

**Rule 3: where it is offered.** `passes.py::DuckTypedCandidatePass` for a constructor parameter that is untyped, `Any`/`object`, or an
attribute bag; `passes.py::CollaboratorStandInPass` (its own producer, `collaborator-standins`, registered after the typed values and the
standard-library stand-ins) for one annotated by a class no pass can build (a third-party `Response`, a request: `duck.py::unbuildable`);
`duck.py::method_standins`, called from `passes.py::method_call_candidates` after the real and literal values, for the required
parameters of a method boundary that nothing can build and the body treats as a COLLABORATOR (attributes, a call, keys; a parameter used
only as data is typed by its own body, and a stand-in for it ahead of the typed candidates took `update(iterable='0')`'s place in a sorted list's setup alphabet). A parameter annotated by a project class is the
nested pass's real object; `duck.py::FAKE_PROJECT_CLASSES` (on by default since [internal ref]) also gives it a fake as a fallback. A recipe built from a fake says so
in its text and `duck.py::holds_fake` tells it: `NestedCandidatePass` ranks a real collaborator before a stand-in for one.

**Setup state** (`setupstate.py`). `::writers` is the closure of "writes the receiver" over `self.m()` calls, and `passes.py::_writing_methods`
admits a method that writes only through the methods it calls to the setup alphabet (`passes.py::_setup_calls`) when it establishes a deferred
field, so `initialize` is in it and a sorted list's `discard` (which merely reaches a writer of fields the constructor sets) is not. `::model`
reads the DEFERRED fields of a class (left `None` or unset by the constructor AND by the methods it calls, assigned by another method) and, per
method, which it establishes and which it needs, each through the methods it calls. `::ranked` puts the
setup calls that establish a field some boundary needs first, and `passes.py::SetupSequencePass` orders its bounded set of sequences
so that one whose steps feed one another (`::feed_score`) comes before one that does not. A class with no deferred field is
unchanged: both are stable sorts on a zero score.

**Stand-in states in the probe matrix.** `api.py::_recipe_cost` ranks a state built from real values before one built from a stand-in
(`duck.py::holds_fake`; likewise `model.py::AvengersReport._recipe_cost` for `best`, `flatobject.py::_cost`, `NestedCandidatePass` and the
bootstrap slot objects), and `api.py::_probe_methods` never lets a stand-in state take the last `api.py::FAKE_RESERVE` = 2 places of the first
round: that round spends states x boundaries probes, and what it leaves is all the later candidates get. Measured: held-out
`sortedcontainers.sortedlist:SortedKeyList` (16 real states, 15 boundaries, 256 probes) went from 15/15 boundaries to 13/15 when three
`key=lambda ...` states raised the states to 17 and left 1 probe for the second candidate of every boundary; with the reserve it is 15/15 again.

**Setup state on the hub.** `passes.py::SetupStatePass` (consumes `validated-recipe`) compares the flattened receiver of a validated recipe
that has setup with its twin, the same constructor without it, and publishes what the setup established as a `setup-state` fact
(`setupstate.py::SetupState`: the fields and their kinds), whose parents are the two replays. `passes.py::NestedCandidatePass` reads it
(and, for a recipe its own sub-search found, the same comparison, `api.py::_nested_synthesizer`) to hand an outer object the
collaborator whose setup state holds the projections the outer reads off it (`duck.py::constructor_demand(...).attrs`), before the
cheapest recipe; the bootstrap driver republishes it as the `setup_state` loop fact (`bootstrap.py::Bootstrap._stage_construct`;
[BOOTSTRAP.md](BOOTSTRAP.md)), and `bootstrap.py::replay` re-decides it with both replays and no proposer.

**Datatypes this adds, by name**, each marked as a *correction* or a *general expansion*:

| datatype | class or boundary it unblocked | kind |
| --- | --- | --- |
| an untyped collaborator whose attributes are CALLED | GUUT flake8 `MergedConfigParser` (`config_finder.local_configs()`), `FileChecker`, `StyleGuide`; httpie `HTTPResponse`, `HTTPRequest`; held-out pyMonet `Box.ap`, `Either.ap`, `Validation` | correction: the pass refused it (`parameter_attribute_uses` order `called`) and nothing built a real one |
| a projection of a collaborator (`application.guide.stats.record(...)`) | GUUT flake8 `legacy.StyleGuide`, `Report` | general expansion |
| a class no pass can build as a parameter type | GUUT httpie `HTTPResponse(orig: requests.Response)`, flake8 `Plugin(entry_point)` | general expansion |
| a method argument used as a collaborator | held-out pyMonet `Last.concat`, `Semigroup.fold` | general expansion |
| a value used as data only in the class's other methods | held-out pyMonet `Map.concat`, `Validation.is_success` (`len(self.errors)`) | general expansion |
| a method that writes the receiver only through the methods it calls | the `initialize(argv)` shape | general expansion |
| a collaborator whose setup state holds what the outer reads | 4 chains of the 17 audit inventories (below) | general expansion |

**What this does not do.** A fake is not a real object: `isinstance(fake, Real)` fails, a context manager or an iterable collaborator is not
built (`with p:`, `for x in p`), a result is shaped by one use and not by the data flow to every use, and a collaborator the class hands to code
outside the project is not followed. A real `flake8.main.application.Application` cannot be initialized inside a replay's budget on this host
(see above), so its boundaries stay un-run; `StyleGuide(application)` works against a fake one. Real nested setup state is rare in the corpora
(`scripts/scan_chains.py` of the lane directory [internal ref] finds 4 outer/parameter/inner chains in the 17 audit inventories where the outer reads a field the inner's
constructor leaves `None`); the fakes carry most of the measured gain.

**Knowledge base.** Consumes `GOAL`, `TEMPLATE` (the same evidence the other constructor passes read), the project's source (static) and
`validated-recipe` (`SetupStatePass`). Publishes `CANDIDATE` evidence (`duck-typed-objects`, `collaborator-standins`; the setup sequences
`setup-sequences` as before) and `setup-state` evidence (producer `setup-state`, parents the two `validated-recipe` items). Every candidate
is still a proposal; the fresh child decides.

**Files and how they relate.** `avengers/duck.py` (stdlib; lazily `passes.py` for `_use_hint`, `_class_node`, `resolve_annotation_class`,
`_fill`, and `solve/frontend/import_resolver.py`; imports `values.py`) is called by `passes.py` (`DuckTypedCandidatePass`,
`CollaboratorStandInPass`, `NestedCandidatePass`, `method_call_candidates`). `avengers/setupstate.py` (stdlib only) is called by `passes.py`
(`_public_method_calls`, `public_method_templates`, `_setup_calls`, `SetupSequencePass`, `SetupStatePass`, `NestedCandidatePass`) and `api.py`
(`_nested_synthesizer`); `bootstrap.py` reads `SetupState`. `values.py::ConstFn` <- `model.py` <- `_worker.py`, `executor.py`.

**Measured** ([internal ref]; CPython 3.14.4, built-in passes only (`--backends none`), 20 s per class, 8 jobs at once with the two arms of a class adjacent in the
job list, `py_obj_flatting` 0.5.0; *before* = development main 4abfa5c0, *after* = this branch at f02912b3 plus docs; both arms ran with the same opt-out
list, sha256 `8f7d2695...e107b`, recorded in every row; raw rows, scripts and the per-class diffs are [internal ref],
`REPRODUCE.md`). Sets: GUUT, 207 classes of ten projects; held-out A, 167 classes of eight libraries
(`packaging`, `pluggy`, `iniconfig`, `sortedcontainers`, `execnet`, `yaml`, `pygments`, `_pytest`), whose per-class results were never read while the rules
were designed; held-out B, 157 classes of the nine Pynguin-corpus projects of the init audit, whose 1282 rows WERE read (so it is not blind). *constructs* =
classes with a validated recipe; *returned* / *works* / *by-design* = boundaries by `works.py::classify` (a normal return that ran past the entry guards; a raise
the target defines or raises by name); *loose* = constructed classes with at least half their boundaries working; *strict* = every boundary works or is a by-design
raise (the audit's `works_strict`).

| set, mode | constructs | returned | works | by-design | loose | strict | with `FAKE_PROJECT_CLASSES`: constructs / works / strict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GUUT, confined + opt-outs | 181 -> 185 | 258 -> 301 | 251 -> 293 | 23 -> 24 | 75 -> 82 | 60 -> 65 | 185 / 300 / 68 |
| GUUT, unconfined | 180 -> 185 | 258 -> 304 | 251 -> 296 | 23 -> 24 | 75 -> 82 | 60 -> 65 | not run |
| held-out A, confined | 140 -> 140 | 251 -> 276 | 245 -> 269 | 39 -> 46 | 55 -> 61 | 42 -> 43 | 148 (2 classes over the job cap) / 296 / 45 |
| held-out A, unconfined | 141 -> 141 | 259 -> 284 | 253 -> 277 | 38 -> 45 | 56 -> 62 | 43 -> 44 | not run |
| held-out B, confined | 136 -> 136 | 400 -> 418 | 387 -> 405 | 64 -> 64 | 59 -> 66 | 50 -> 59 | 138 / 413 / 65 |
| held-out B, unconfined | 143 -> 142 | 418 -> 437 | 405 -> 424 | 69 -> 69 | 62 -> 70 | 51 -> 61 | not run |

No class lost a boundary in any run except two classes that flip between runs of ONE commit in both arms (`flake8.main.options:JobsArgument`:
before 1, 1, 0, 1 validated, after 0, 1, 1, 0; `pytutils.lazy.lazy_import:ScopeReplacer` unconfined: before 0, 1, 1, 1, after 1, 0, 1, 1). Per project (GUUT, confined): flake8
constructs 40 -> 44, returned 90 -> 127, works 84 -> 120, strict 8 -> 11; httpie works 49 -> 55, strict 19 -> 21; the other eight projects are unchanged. Held-out B:
pyMonet returned 72 -> 90, works 66 -> 84, strict 6 -> 15 (`Box`, `Either`, `Try`, `Last`, `Map`, `Max`, `Min`, `One`, `Validation`). Held-out A: `_pytest` works 40 -> 54,
`pygments` 9 -> 13, `yaml` 25 -> 29 (3 -> 10 by-design raises), `pluggy` 35 -> 37.

**The audit's 1282 types** (the 22 GUUT and 21 held-out project classes of the 25 + 24 types the init audit labelled 1282; the rest are standard-library classes: `Path`, `Logger`, `ArgumentParser`, `socket`, `SSLContext`):

| set | classes | works (calls of the audit's rows) | classes strict | classes strict with `FAKE_PROJECT_CLASSES` |
| --- | --- | --- | --- | --- |
| GUUT | 22 | 11 -> 33 of 60 | 5 -> 9 | 10 |
| held-out B | 21 | 2 -> 9 of 51 | 2 -> 7 | 9 |

The audit predicted, for 1282 alone, 13 GUUT types and 12 held-out types made to work (of 25 and 24 with the gap). The gap its mechanical rule called 1282 is wider than setup state:
it counts every `'NoneType' object has no attribute` on a boundary, and of the audit's calls this change leaves 23 GUUT and 33 held-out calls still failing (neither works nor by-design raises), by cause: a real file the class
reads (flake8 `FileChecker`), an `Application.initialize` that takes minutes, a method argument of a project class (`Key.create_from(error)`; they work with `FAKE_PROJECT_CLASSES`), a
`NewType('Quantity', Decimal)` parameter (pypara `SomeMoney`, `SomePrice`: 21 held-out calls), and a collaborator that flows into a container and is read later (`NumpydocParser.add_section`).

**Cost.** Wall time per class (median, total) GUUT confined 10.6 s / 3273 s -> 12.5 s / 3845 s (+17.5 %; 3905 s with `FAKE_PROJECT_CLASSES`), held-out A 18.4 s / 3751 s -> 19.3 s / 4050 s (+8 %), held-out B 14.4 s / 2968 s ->
15.1 s / 3083 s (+4 %); more candidates are replayed. With `FAKE_PROJECT_CLASSES`, two `_pytest` classes (`FixtureRequest`, `FSCollector`) exceeded the 300 s job cap that they used 125 - 224 s
of without it.

**What is a stand-in.** In the head arm, GUUT 28 classes have a validated recipe built from a stand-in (this includes [internal ref]'s attribute bags), 19 have one as their cheapest recipe and 15 are
constructed ONLY through stand-ins; held-out A 11 / 4 / 0; held-out B 5 / 0 / 0. `ReplayOutcome`, the program text and the producer name them; a consumer that needs a real receiver reads
`duck.holds_fake` (or the `fakes` flag of the loop's `construction_recipe` fact).

**Pre-existing, found on the way (not changed here).** `nested-objects` raises `ValueError: a nested object is built by its constructor, not from a ready-made receiver` on 19 held-out classes
in BOTH arms (a class whose only validated recipe is a ready-made receiver of [internal ref]): the scheduler records it as `pass-failure` evidence and the class loses its nested candidates.

## Constructors the source does not declare

`ConstructorTemplatePass` reads the constructor from the class body
(`solve/frontend/constructor.py::constructor_template`, dataclass-generated
fields included). When the class's own source does not declare one
(`::declares_constructor` is false: it inherits `__init__`, or a base or
metaclass generates it -- `NamedTuple`, `collections.namedtuple`, `Enum`, an
exception or `ModuleType` subclass), an empty static parameter list describes
the SOURCE, not the constructor, and the only candidate it yields is `Cls()`.
The pass then asks a fresh child (`IsolatedExecutor.describe_class`, worker
action `describe`, response `_wire.ClassDescription`) for
`inspect.signature(cls)` and publishes that as the template -- names, kinds,
annotations as text, whether a default exists, never the default itself --
and, for an `Enum`, one candidate `Cls(value)` per member whose value is a
recipe literal. A class nested in another class's body is found by its dotted
name (`Outer.Inner`, `solve/frontend/import_resolver.py::load_class_def`).

Both the constructor template and the method templates below read a
parameter's type from a PEP 484 TYPE COMMENT when it has no annotation
(`solve/frontend/constructor.py::function_parameters`): the per-argument
`# type: A` and the function's `# type: (A, B) -> R`, whose receiver PEP 484
leaves out; a comment of any other length is not trusted, and an annotation
always wins. The tree is parsed with `type_comments=True`
(`solve/frontend/import_resolver.py::parse_with_type_comments`, falling back
to a plain parse when a comment sits where the grammar forbids one). flake8
3.9 writes every type this way (181 signature comments, no annotations), so
before [internal ref] its 292 parameters all read as untyped.

The Hypothesis constructor search (`passes.py::_call_strategy`) draws every
required parameter and, for every DEFAULTED one, either omits it (drawn first,
so a plain "does it build" search still shrinks to `Timer()`) or draws it, so
a goal that needs a defaulted parameter varied -- a `field_bounds` on
`Timer.name`, a branch on a defaulted flag -- has a candidate that can reach it.

Measured before [internal ref] on the ten GUUT subjects (CPython 3.14.4, 206 classes):
of the classes for which Avengers produced no object, 52 declare no constructor
in their own source (enums, named tuples, inherited constructors), and 5 are
nested classes, for which no template was found at all.

## Backends

Each backend is a pass that runs the tool itself. Two contracts hold for all of
them. **Nothing a backend says is believed** -- it proposes a candidate or some
test source, and promotion still comes from the ordinary fresh-process replay.
**Everything is availability-gated** -- a missing tool is a missing tool, not an
error.

**The default portfolio runs the recipe derivers** (owner rulings 2026-10-19,
[internal ref]): with `backends` unset, `synthesize_object` adds
`api.py::DEFAULT_BACKENDS` -- `hypothesis-from-type`, then `ghostwriter`, then
`pynguin`, in that order, AFTER the built-in passes -- each only when its
interpreter can import it (`backends.py::backend_passes`; availability is
probed once per interpreter). `backends=()` runs none and an explicit list
runs exactly that list. `THARKUN_AVENGERS_BACKENDS` (comma-separated; empty
means none) overrides an UNSET choice; the test gate sets it empty
(`tests/conftest.py`) so no assertion depends on a backend's search, and the
selection itself is checked in `tests/test_avengers_default_backends.py`. A
nested sub-search (`api.py::_nested_synthesizer`, 10 s) runs none.

**SV never runs one.** Pynguin and Ghostwriter are
outside the SV-COMP allowed runtime set, so the SV path is gated twice:
`sv/avengers_seed.py` passes `backends=()` explicitly, and a CONFINED search
(an executor with `files=`, which is how SV replays) gets no default backend
whatever the caller leaves unset (`api.py::resolve_backends`).

**Pynguin runs from its OWN tool venv.** Pynguin 0.48.0 declares `pytest<9`
and `black<26`, so it cannot share the admitted test environment (pytest
9.1.1); the `avengers`/`backends` extras therefore list only what runs in the
search's own environment (`hypothesis`, and `black` for Ghostwriter), and Pynguin is the separate
`pynguin` extra / `requirements/pynguin-tool.txt`, installed in a venv of its
own together with the requirements of the code under test (Pynguin imports
the module it tests). `THARKUN_PYNGUIN_PYTHON` names that venv's `python`
(`backends.py::pynguin_interpreter`; it wins over a search's generic
`interpreter`); unset, or naming a python that cannot import pynguin, the
backend is skipped by name like any unavailable one. Only Pynguin runs there:
its test source is mined and every recipe is still replayed in the search's
own interpreter. The GUUT measurement ([internal ref]) uses one such tool venv per
subject (`av-work/venvs/pyn-<project>`), so Pynguin's pins never replace a
subject's (its `docstring-parser` 0.18 dependency vs the subject
`docstring_parser` 0.7.3).

**A backend is an Avengers-internal implementation detail, never a direct
dependency of a caller.** Pynguin ships in the `pynguin`/`bench` extras (Ghostwriter formats with `black`, listed by `avengers`/`backends`/`bench`, never by the SV path, 2026-10-28); both stay behind
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

| Backend | Default | What it contributes | Needs |
| --- | --- | --- | --- |
| `hypothesis-from-type` | yes, 1st | `st.from_type` on each constructor parameter; from the hub (below), the hub's types registered first | `hypothesis` |
| `ghostwriter` | yes, 2nd | test source from Hypothesis' Ghostwriter; from the hub (below), the tests it writes are RUN and mined | `hypothesis` and `black` (the `avengers` extra lists both) |
| `pynguin` | yes, 3rd | test source, including its crash tests | `pynguin` in its tool venv (`THARKUN_PYNGUIN_PYTHON`) |

**SV's `avengers` stage** (`sv/avengers_seed.py`, [internal ref]) constructs every class
the SV task's module defines and offers back a validated recipe's literal
arguments as SEED candidates for `sv/stages.py`'s enumeration. Until [internal ref] it
requested `backends=("pynguin",)`; it now requests none (above), and the
constructor argument its own fixture needs -- a value inside
`1_000_000 < self.value < 1_000_100` -- comes from the class's guard, solved by
`solve/condition.py` in `objectives.py::CoverageSetupPass`
(`::_guard_solved_attributes`). The measurement that kept the stage out of
`DEFAULT` was taken WITH Pynguin: on all 80 (task, property) pairs of the SV-Benchmarks Python
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

A goal replayed under `files=` (SV's stage passes the task file) runs in a
CONFINED child under the SV I/O model (`docs/SV_IO_MODEL.md`,
`solve/environment.py::install_files`): the declared files are the only
readable files (an undeclared one does not exist, so `open` raises
`FileNotFoundError`), a write inside the child's scratch directory stays there,
and a write outside it is refused and flagged. `avengers/_worker.py::main`
applies the same flag rule as every other reader of those flags
(`driver.py::_avengers_object_for_bounds`, `passes.py::HypothesisCandidatePass`,
`sv/concolic.py`): `file_effect` alone is a modelled effect and the outcome
stands, so a recipe that reads a declared file validates; any other flag
(`file_unmodelled`, `stack_unmodelled`, ...) turns the outcome into
`unmodelled`. Before [internal ref] any flag at all did, so no recipe that touched a
file -- even a declared one -- could validate. A module the worker imported
before installing the model and that bound `open` itself (`tokenize`, `bz2`,
`tarfile`) reads through the model too: `solve/environment.py::_rebind_aliases`
rebinds such aliases ([internal ref]), so `tokenize.open` of a declared file validates
and of an undeclared one raises `FileNotFoundError`.

Every child, confined or not, runs with its working directory in a scratch
directory that `executor.py::IsolatedExecutor._invoke` removes afterwards, and
with `PYTHONDONTWRITEBYTECODE=1`, as every SV child is started (`sv/engine.py`,
`sv/forkserver.py`, `sv/hybrid.py`). Before [internal ref] an unconfined replay ran in
the caller's cwd, so path-taking constructors left their drawn names (`0`,
`00`, `set.copy`, ...) there, and it cached bytecode in the target's project.
In a confined child the `.pyc` the import machinery would cache -- for the
`_sv_verifier` stand-in or any module first imported there -- is the
interpreter's write, not the target's, and before [internal ref] it turned every
confined replay into `unmodelled` on a host whose environment did not already
disable bytecode writing (the two `tests/test_sv_avengers_stage.py` reds under
[internal ref]). An unconfined replay is still unconfined: a write to an ABSOLUTE path
reaches the host.

**Replays are confined by default ([internal ref], owner ruling 2026-10-20: GUUT
replays run sandboxed).** An `IsolatedExecutor` with no `files=` still runs every
child that runs a recipe (`replay`, `replay_many`, `mine_template`) under the
same model. It declares the goal's own source and every nested recipe's source
(`executor.py::_declared_sources`, which walks the recipe's values as
`_worker.py::_argument` builds them). A target that needs a real file must have
it declared with `files=`. An installed package's own files are readable anyway,
since they sit under the interpreter's prefixes, which the model treats as
infrastructure. `IsolatedExecutor(unconfined=True)` is the explicit opt-out.

**Per-case opt-out list ([internal ref] amendment, owner ruling 2026-10-20: "per test
case opt out, but make a ticket to come back to this"; the ticket is 1275).**
`benchmarks/avengers/unconfined.json` (`executor.py::UNCONFINED_CASES`, beside the
subject manifest `__main__.py::MANIFEST`) names, per benchmark project, the
`module:qualname` of every GUUT case whose validated-class or covered-boundary
count is lower under the confined model than without it, each with one line
naming the check that refused it. `executor.py::unconfined_cases` loads it
(empty when the checkout's `benchmarks/` is absent, so an installed Tharkun
confines everything) and `executor.py::IsolatedExecutor._invoke` runs a replay
without `files=` unconfined only when its goal's case is listed (a batch: only
when every goal in it is) or `unconfined=True` was passed. The parent tags each
such outcome in `model.py::ReplayOutcome.unconfined` (the listed reason, or
`"explicit"`), so a report counts opted-out replays apart. The list exists to
be emptied: 1275 recovers each case under the model (modelled absence for
undeclared probes, `sys.path` code locations, a spawn/thread story) and
deletes its entry. Adding an entry needs the refusing check named, not a
guess, and an A/B showing the case loses coverage when confined.
**The list's presence is recorded ([internal ref]).** A missing list (a Tharkun frozen from `src/` alone) keeps
everything confined -- the safe default -- but it silently confines every case the list opts out, and A/B tables
undercount (flutes 25 -> 0, pdir2 7 -> 0 on 2026-10-21). So `executor.py::opt_out_list` reports
`{"path", "sha256"}` of the file read (`sha256` is `"absent"` when it does not exist); it is on
`model.py::AvengersReport.opt_out_list` and `bench.py::RunResult.opt_out_list`, i.e. in every report and every
benchmark row. And `executor.py::OPT_OUT_LIST_ABSENT` (`opt_out_list_absent`) is a refusal on every replay outcome that was
confined BY DEFAULT while the list was absent (set by the parent in `IsolatedExecutor._invoke`, so it reaches
`AvengersReport.refusals`): the case may be one the list would have opted out, and with no list that cannot be told. A replay
under `files=` or `unconfined=True` never consults the list and carries no such refusal.
The SV backend rule is unchanged: `api.py::resolve_backends` drops the
external backends only for an explicit `files=`. Some children are not
confined by default: the generation children (Ghostwriter, `st.from_type`
arguments, `describe`), Pynguin's own process, and the parent-side
materialization behind `IsolatedExecutor.confine`. They are not replays.
Spawning and starting threads are refused in a confined child, so a class that
needs a process pool cannot validate there.

**What the unconfined children may touch, and what survives them ([internal ref]).** On
2026-10-21 337 orphaned Pynguin trees (1,664 processes, 61 CPU-hours) were found
with their cwd in a subject's shared venv, into which fuzzed file functions had
written about 4,500 files. Pynguin runs the subject unconfined, so the rule is
the OS's, not a model's: `backends.py::PynguinBackendPass.generate` gives it a
working directory of its own inside the run's scratch (`cwd/`, apart from the
directory the tests are read from, so a stray `*.py` is never mined as a test)
and, unconfined, a disposable copy of the project for `--project-path`
(`backends.py::_disposable_project`: the module's top package, or a standalone
module with its sibling `*.py` files), so a write next to the code under test
lands in scratch. Measured on `codetiming._timer` (Pynguin 0.48.0, 8 s, seed 1,
original path vs copy): one test module each, equivalent tests. A write to an
ABSOLUTE path still reaches the host, as for any unconfined replay; under
`files=` Pynguin keeps the original project, where the environment model
refuses the write. Every child that runs target code unconfined
(`backends.py::_run_in_own_group` for Pynguin, `executor.py::IsolatedExecutor._spawn`
for the generation children) starts in a session of its own and its whole
process group is killed on a timeout, on any exception, after a clean exit
(`executor.py::kill_group`) and, for a group still running, by an `atexit` hook
(`executor.py::LIVE_GROUPS`); `solve/loader.py::import_project_isolated` does
the same. A parent that is itself SIGKILLed or SIGTERMed runs no hook, so its
children are bounded by their own timeouts only. Audit of the other children:
Ghostwriter and `st.from_type` (`generate_source`, `search_constructors`,
`describe_class`) import the target unconfined in a scratch cwd, and had the
group kill on a timeout only (now all exits); in hub mode, and for every
replay, the tests run confined by default. `hybrid/backend.py::AtherisBackend`
(SV's fuzz stage, run in the stage's scratch) still stops only its direct
child; it is not an Avengers backend and was left alone.  `tests/test_avengers_backend_sandbox.py` pins it with a fake
`pynguin`.

```python
report = synthesize_object(goal)  # built-ins, then every default backend installed
report = synthesize_object(goal, backends=("hypothesis-from-type", "ghostwriter"))
report = synthesize_object(goal, backends=())  # built-ins only
```

```console
$ THARKUN_PYNGUIN_PYTHON=/path/to/pynguin-venv/bin/python python -m tharkun.avengers backends
-- flatting      flattening-unavailable: py_obj_flatting>=0.5.0 is not installed (REQUIRED: ...)
ok ghostwriter   available
ok hypothesis    available
ok pynguin       available
```

### Hypothesis and Ghostwriter from the hub ([internal ref])

Blind, `GhostwriterPass` was almost a no-op: `ghostwriter.magic(cls)` writes the constructor test and the tests of the
class and static methods only (an instance method is `ghostwriter.fuzz(Cls.method)` and its receiver comes out as
`self=st.nothing()` with a TODO), the strategies it writes come from the goal's annotations alone, and the recipe miner
(`GeneratedSourcePass`) reads assignments of literal constructor calls, which a `@given` test does not contain. Nothing in
Tharkun called `st.register_type_strategy`.

With a `hub_scope` (`synthesize_object(hub_scope=...)`, `backend_passes(hub_scope=...)`, `bootstrap.Config.hub_strategies`
when the loop runs the backends) the two passes read the bootstrap hub ([BOOTSTRAP.md](BOOTSTRAP.md)) from the knowledge
base they run on and are no longer blind:

1. **Which types** (`hubtests.py::hub_view`): the observed `construction_recipe` facts of the scope (a type the hub can
   build; at most `MAX_RECIPES_PER_TYPE` cheapest recipes of at most `MAX_TYPES` types, the types the goal's own
   signatures name first) and, from the first validated recipe on, the goal's own `VALIDATED` recipes (the self edge).
2. **Registration** (`_hubtests.py::register_hub`, in the child): `st.register_type_strategy(T, strategy_for(T))`. The
   strategy for `T` draws one of `T`'s recipes and builds it fresh through the worker's own `_prepare`, the way a replay
   builds it (`recipe_strategy`); with `py_obj_flatting` and a mined template it also draws every leaf of the template
   from the leaf value's kind (`typeinfer/core/kinds.py::value_kind`, `strategies.py::make_strategy`), `reproduce`s the
   bytes and `materialize`s the object (`template_strategy`). Subclasses count: `from_type(Base)` resolves to a registered
   concrete subclass. The strategy is given a name that is not Python (`_named`), so Ghostwriter writes `st.from_type(T)`
   into the test source instead of an expression nobody can import.
3. **Writing** (`_hubtests.py::_ghostwriter_chunks`): one `ghostwriter.fuzz` per callable -- the constructor and every
   instance method, `self` annotated with the class through a signature stand-in (`_method_stub`) so it is drawn rather than
   `st.nothing()`. The `from-type` tool is the same registration followed by `st.from_type` on every constructor parameter
   (`HypothesisFromTypePass`, constructors only, as before).
4. **Running**: the written tests are exec'd and run under Hypothesis in the same child (`phases=(generate,)`, a seed from
   the loop's `seed_for`, `max_examples` and a time budget), their bodies observed through Hypothesis' documented
   `hypothesis.inner_test` hook; an exception from the target is an outcome of the example, not a failure of the test. The
   child is a REPLAY child: confined by default like every child that runs a recipe (`executor.py::_RECIPE_ACTIONS`;
   `_worker.py::_preload_hub_tools` imports Hypothesis (and black, when installed) before the confinement is installed, trusted preparation
   as the `_sv_verifier` stand-in is). A write outside the scratch is refused as in any replay. Hypothesis stats the source
   of every local module to harvest its literals, a probe of an undeclared path; the worker keeps the response and reports
   the count (`unmodelled_flags`) instead of discarding a run that only PROPOSES.
5. **Provenance** (`_hubtests.py::Registry`): every object a registered strategy built is remembered with the recipe that
   built it, so an argument that is a hub object, a list of them or an optional one is written as the `NestedObject` (or
   container of them) a replay can rebuild. Anything else must be a recipe literal (`_worker.py::_expressible`) or the
   record is dropped and counted (`unexpressible`). An object built from a template has no recipe and is never mined.
6. **Mining** (`hubtests.py::emit_hub_tests`): per (tool, goal, phase) -- phase `A` when the goal appears, phase `B` once
   it has validated recipes and has something they could feed (instance methods for `ghostwriter`, a slot naming its own
   type for either) -- the pass emits the test source (`SOURCE_ARTIFACT`), every construction that returned as a
   `CANDIDATE`, every method call that returned both as a `CANDIDATE` (the receiver's recipe plus the call as a setup step)
   and as an `OBSERVED_CALL`, the public fields of what a returned construction built as `OBSERVED_FIELDS`, and a
   `HUB_TESTS_RUN` marker with the counters. Each lists as `Derivation.parents` the goal and the hub facts (or validated
   recipes) whose objects the run drew. A run keeps the composed constructions (up to `KEEP_COMPOSED`, shortest first) and
   `KEEP_PLAIN` plain one: literal variants are the other passes' job and each kept candidate costs a replay.
   **Nothing is believed**: a candidate is promoted only by the ordinary fresh replay; an `OBSERVED_CALL` joins the
   boundary probes (`api.py::_with_observed_calls`, after the hub-object calls, before the literal candidates) and is a
   `call_effect` only because `_probe_methods` replayed it.

Without a `hub_scope` none of this runs: `GhostwriterPass` calls `IsolatedExecutor.generate_source`
and `HypothesisFromTypePass` calls `search_constructors`, exactly as before (`tests/test_avengers_hub_tests.py::test_without_a_hub_scope_the_backends_are_what_they_were`).

Backends run *after* the built-in passes, deliberately: an external generator
costs seconds to tens of seconds before it says anything, and putting one first
made time to first object an order of magnitude worse than the built-in
portfolio alone. The measurement is in the benchmark below.

Any tool that writes Python test source shares one recipe miner
(`GeneratedSourcePass`), so adding a generator means producing source and
nothing else. Pynguin's *crash tests* are ingested rather than filtered out: a
sequence that killed the interpreter is exactly the boundary behaviour this
search is for.

## The scheduler and the store

The scheduler is generic and lives in the KB layer: `fixpoint.py::Avengers` (with `Emission`, `RepeatPolicy`,
`FunctionPass`, `PassMetrics`, `PortfolioOrder`, `YieldOrder`, `SaturationBudget`, `SaturationReport`,
`RoundInfo`) imports only `tharkun.knowledge`, because the bootstrap driver runs it too
([BOOTSTRAP.md](BOOTSTRAP.md)). `avengers/engine.py` is a thin re-export facade over it, with no logic of its own,
so `from tharkun.avengers.engine import SaturationBudget` keeps working; new code imports `tharkun.fixpoint`.
`tests/test_layering.py::test_the_engine_is_a_facade_over_the_fixpoint` pins it.

Two things a loop that outlives one call needs, both opt-in so a single search is unchanged. `cursors=` is a
`{pass name: evidence cursor}` mapping the caller keeps across `synthesize_object` calls on one knowledge base: a
pass then sees each evidence item once over the whole run instead of being handed the earlier goals' items again
(without it, a second goal on a shared hub re-runs the first goal's work). `on_round=` is called after every outer
scan with a `RoundInfo`. The evidence ceiling and the first-result milestone count from the start of the call, and
`synthesize_object` reports one goal's recipes, failures, states and negatives (by source path and class, and from
this call's evidence) when the knowledge base is shared. `objects={method: {parameter: [NestedObject, ...]}}` puts
the calls that pass those objects ahead of a boundary's literal candidates
(`api.py::_with_object_calls`): the hub's recipe of a parameter's class is the one argument literal search cannot
make. Since [internal ref], the same probe matrix also obtains real arguments through the constructor composer when the caller
has not supplied them ("Whole object arguments" above). `probe=False` skips
the boundary-probe matrix, and `api.py::probe_methods` runs it later on recipes the caller holds: the bootstrap
driver searches first and probes from the recipes it read back from the hub, so a call effect is a hub fact derived
from a hub fact ([BOOTSTRAP.md](BOOTSTRAP.md) "The per-type loop").

`Avengers` runs the portfolio to a fixed point over shared evidence. Every pass
is measured -- activations, runs, wall time, emissions, and *novel* emissions --
and `PortfolioOrder` (the default) keeps the declared cheap-first order while
`YieldOrder` ranks by novelty per second, offering unmeasured passes first so
the ranking cannot become self-fulfilling. `SaturationBudget.max_pass_seconds`
stops one slow backend from spending the whole budget.

A scan is one delivery: every pass is handed the evidence that existed when the
scan BEGAN (`fixpoint.py::Avengers.run`), so no pass sees another's output from
the same scan and the order inside a scan cannot change what a pass is given
(semi-naive, order-independent). The first scan therefore delivers the initial
emissions only -- one goal, or the bootstrap driver's `bootstrap-start` seed
([BOOTSTRAP.md](BOOTSTRAP.md) "The team") -- and the next scan is where a pass
reads what the others found.

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
costs nothing in time to first object, and beats `pynguin`
significantly on anything. At that scale "Avengers beats the best
single backend" is **not** established. The full corpus at the paper's budget is
the measurement; the smoke run is the harness working.

## Flattening and observable effects

*([internal ref], owner 2026-10-21: "object fields being different is observable".)* A boundary call is judged by
what a caller could see of it, and the object's own fields are the first thing a caller sees. The worker
therefore flattens the receiver with `py_obj_flatting` before and after **every** boundary call
(`_worker.py::_run`, `flatting.py::observe_state`) and reports the difference.

**Per call**, a `ReplayOutcome` carries (all additive on the wire; `_wire.py` protocol `/3` also reads `/1`
and `/2`):

| field | what it says |
| --- | --- |
| `field_changes` | `FieldChange(path, change, call, before_kind, before, after_kind, after)` per leaf path that was `added`, `removed` or `changed` across the call. Paths are `py_obj_flatting`'s (`$.count`, `$.history.el#2`). A value longer than 80 characters is its `sha256:<16 hex>/len=<n>`. At most 64 are reported; `effects.field_count` is the true number. |
| `effects` | `Effects(returned, fields, output, raised_by_design, body, ...)`. `returned`: it returned normally (`None` included). `fields`: the flattened receiver differs (`None` = not observed). `output`: it printed (count, hash and a 120-character preview of stdout and stderr, not the text; `_worker.py::_Sink`). `raised_by_design`: the exception is one the target defines or raises by name. `body`: a line beyond the method's entry guards ran. |
| `opaque` | `(path, reason)` of every leaf the flattener could not read (`file`, `descriptor`, `builtin_function`, ...). A socket or an open file is a leaf like any other: named, with its reason, and **not** part of the digest. |
| `refusals` | `flattening-unavailable` or `flattening-failed:<ExceptionType>` (the flattener raised; the digest is then `flatting.py::shallow_digest`, prefixed `s:`). Empty when the receiver was flattened. |
| `state_fields` | the final receiver's leaves near the root, `(path, kind, value)`; what is published as facts (below). |
| `state_digest` | SHA-256 of the canonical flattened tree. The state-coverage objective, the store and `ReceiverState.key` read this and nothing else. |

Setup is not the call's effect: the diff is between the state **entering** the boundary call (after the
constructor and the setup calls) and the state after it. When the call raised, the diff and the output up to the
raise are still reported.

**Where a shallow digest is allowed.** Only for a receiver the flattener itself raised on (it is built not to),
marked `flattening-failed`, never silently. Everything the flattener can read is read by it; what it cannot
read is `opaque`. `py_obj_flatting` absent is not a case for a shallow digest: it is the refusal. Before [internal ref]
`_worker.py::_safe_state_digest` fell back to a shallow digest without a word, and since the package was
installed nowhere no state was ever flattened -- a violation of the no-silent-substitution rule
(docs/design/09 in the Eregion repository).

**Confinement.** A confined child (the default since [internal ref]) must not meet the flattener, or the standard library's own
import-time probes, for the first time inside the confinement. Importing every public standard-library module under
`install_files` ([internal ref]) flags exactly one cause: `sysconfig`'s import-time `is_python_build()` stats the interpreter's
directory, an undeclared path, and `environment.py::model_stat` flags it `file_unmodelled`; `ctypes`, `pydoc`, `trace`
and `zoneinfo` flag only because they import `sysconfig`, and `platform` reaches it from its functions. So the replay
is lost as `unmodelled` for the interpreter's probe, not for anything the target does. Installing `py_obj_flatting`
under tharkun main did exactly this to the shared test venv's whole held-out set (114 of 166 classes constructed
before, 0 after), because its import pulls `platform`.

The fix is trusted preparation, not a looser model. `solve/environment.py::INFRASTRUCTURE_MODULES` (`sysconfig`,
`platform`; a fixed list, add a module only with a measurement like the one above) and
`environment.py::preload_infrastructure()` import them before confinement, and `_worker.py::_prepare_confinement`
calls it, then imports the flattener and flattens one small throwaway object (whatever else the flattener reads the
first time it meets a kind of value). `model_stat` flags an undeclared stat exactly as before
(`tests/test_environment_preload.py`). Any other confined child that meets the same flag calls
`preload_infrastructure()` before `install_files`/`confined`; SV's children have not been shown to.

A consequence, measured: a target that imports `sysconfig`, `zoneinfo`, `ctypes`, `pydoc` or `trace`, or a library that does
(`packaging`), is no longer lost to that probe. On the held-out set 19 `packaging` classes that main reports
`unmodelled` construct, with or without the flattener installed.

**"Works"** (`works.py`). A call that returns is not a call that exercised the method: `if x is None: return None`
returns for `None` without running the body. Per boundary call, `works.classify` says:

| class | meaning |
| --- | --- |
| `works` | it returned, and ran past the method's entry guards (`Effects.body` is not `False`). The return value (`None` too), the field diff and the output are its effects. |
| `early-return-only` | it returned, but only through an entry guard |
| `by-design-raise` | it raised an exception the target defines or raises by name (`works.target_exceptions`): a precondition the class enforces |
| `rejected` | anything else (a probe's `TypeError`, a crash, a timeout) |

The **entry guards** are the leading statements of the method's own source that end the call: a docstring,
`assert`s, and `if` statements without an `else` whose body is ONE `return` or `raise`
(`works.py::entry_guards`). Those `if`s are guards only when something follows them: when nothing but a lone
`raise` does, they are the method's own cases (`if name in self._timings: return ...` then `raise KeyError(name)`),
and a branch that does work before it returns (`self.hits += 1`) is not an early exit either. A call ran past
the guards when a line outside them executed, or when no guard fired.
Lines are the boundary call's own: a `_LineProbe` (one `settrace` that returns no local trace function for any
other file) for an untraced replay, the parked-and-merged tracer result for a traced one. *Shortcut, recorded in
the code:* only leading guards count, and a guard written on one line (`if x: return y`) cannot be told fired from
not fired by line events and is read as not fired; upgrade to branch events if a measured class needs it.

`report.works` (`works.py::summarize`) classifies every probed boundary by the best probe any validated state
gave it and counts, per class: constructs, `works`, `early-return-only`, `by-design-raise`, and
`field_change_only` -- a working probe that returned `None` and changed the receiver, a method whose entire effect
is in the fields. These are the columns of the A/B harness.

**Published as sourced facts** (`observations.py`; the hook for the bootstrap loop [internal ref]). After a search,
`synthesize_object` publishes to its `KnowledgeBase`, producer `avengers-flattening`, with the validated recipe
as parent:

* `observed-field` `{class, field, kind, value}`: after some replay the receiver had this leaf of this type
  holding this value (at most 8 distinct values per field, then the kind alone);
* `field-effect` `{class, call, field, change, before_kind, before, after_kind, after}`: this call changed this
  field.

`observations.field_evidence(kb, source_path, class_name)` turns them into the keyword arguments of
`tharkun.typeinfer.Evidence` for **source 5** (docs/TYPEINFER.md): one row per kind a top-level field was observed
to hold, `param` = the field's name (`self.x = x` is the common constructor shape). Avengers does not import
`typeinfer` (layering), so the rows are dicts and the consumer builds `Evidence(**row)`. Wiring them into
`find_inputs` and the bootstrap loop is [internal ref].

*Built in [internal ref] (partly):* the object kind (an object **parameter** of `find_inputs`) is the validated recipe with typed leaves
("A recipe as a parameter", below), its row carries the replayable recipe and the SHA of the template mined here
(docs/INPUTS.md "Object parameters"). Taking the flattened FIELDS as the leaves where a template exists is ruled, not built.
`minleaf`'s `state_project` and SV's field samples still observe state by their own means;
moving them onto the flattened form is the same decision for those layers and is the owner's.

## A recipe as a parameter

### Pre-synthesis facts ([internal ref])

`tharkun.object_facts.explore_object(source_path, class_name, project=...,
roots=(...), seconds=20)` acquires constructor/setup facts before the first
synthesis reverse walk. It calls `flatobject.find_recipes` without boundary
calls: an unanswered callback cannot discard a useful constructor. The existing
initializer runs its built-in, Ghostwriter (where available), then Hypothesis
stages within the search budget. The export retains up to32 distinct program/state
recipes by default, including repeated setup states. Individual input-door
requests still accept at most three; consumers select or batch recorded alternatives.

`write_facts(path, entries, revision=<exact producer revision>)` atomically writes
`eregion.object_facts.v1`, producer phase `pre-synthesis`. Each entry preserves the
public `inputs.describe_object` descriptor and exemplar FlatObject records:
executable recipe records, audit programs, keys, template SHA/paths, stand-in
labels, and all typed scalar constructor/setup leaves. Exemplars are seeds,
never pins or authored domains. Nested recipes stay inline and replay bottom-up.
Source-module and conservative own-package hashes bind facts to the exact subject
side. Known nested sources are fingerprinted too; unrecorded external dependency
versions remain an environment responsibility.

`IsolatedExecutor.snapshot(goal, recipe)` uses the existing confined worker to
construct and run recorded setup only. It inventories actual dictionary fields
and inherited slots, tuple-backed namedtuple fields and exception arguments.
Properties are listed without invoking getters. Fields have
object-state addresses (`[["attr", "name"]]`), observed qualified types, declared
annotations from source when present, and public-codec values where supported.
Opaque fields keep their types and named serialization causes; replay can
reconstruct them. Field `leaf` means observed scalar; only descriptor `leaves`
are constructor/setup argument holes. Observation establishes no input lineage.

Acquisition/flattening failures remain in `refusals` beside successes. No recipe
means `DECLINED` with its actual cause and `declined_or_explicit_policy`.
Construction validation is separate from `boundary: not_requested`.
`snapshot_loadable` and conservative `fully_faithful` require every exported state
and field to be codec-serializable; `fully_faithful` also excludes stand-ins.
Neither metric proves a complete domain or all possible program states.
Recipes containing generated dataclass definitions retain the explicit policy
label `generated_class_witness` in both descriptor and record. A native replay
of such a witness is not evidence of the caller's actual class definition, and
it never counts toward `fully_faithful`.

Synthesis consumes the artifact before walking backwards. Runtime solvers may
fill recorded leaves or replay recorded construction; they never acquire a
missing skeleton. Program-produced parts stay forward computations/carried
values. Consumers bind each occurrence as outside, program-derived or class
definition; unknown lineage is a named gap. A canonical `dataclasses.Field`
requires the actual owner class/member, never an invented dataclass witness.

*([internal ref], G4; `flatobject.py`, used by `inputs.py` and nothing else.)* `find_inputs` takes an object as a parameter
(docs/INPUTS.md "Object parameters"). The object is a validated recipe -- the constructor and setup program a fresh child
certified -- with its scalar literals as typed LEAVES; the harness builds the object from a leaf assignment only where the entry
is called. `flatobject.py` is the Avengers half:

| function | what it says |
| --- | --- |
| `leaves(recipe)` | the `bool`, `int`, `float`, `str` and `bytes` literals the constructor call and each setup call pass, addressed by call site (`__init__.input_string`, `add#0.value`, `__init__[0]`) |
| `complete(recipe, source, class)` | the recipe with the constructor's scalar-literal defaults the program left out passed explicitly (`constructor_template`, nothing runs): `Box()` of `Box(limit=3)` is `Box(limit=3)` |
| `program(recipe, class, names)` / `ConstructionRecipe.statements` | imports and statements that build the object with the leaves left as names, ending in `object_0` |
| `assign(recipe, values)` | the recipe with leaf values filled in: the explicit, replayable record a row carries (`recipe_from_record` reads it, its `key` is the Avengers digest) |
| `skeleton(recipe)` | the key of the program with every leaf at its kind's default: two recipes with one skeleton are one program over different values |
| `known(kb, source, class)` | `(recipe, state digest)` for each recipe a knowledge base holds: the bootstrap hub's observed `construction_recipe` facts and any search's `validated-recipe` evidence |
| `find_recipes(source, class, kb, roots=())` | up to 3 usable recipes, cheapest first, one per skeleton and per receiver state; `known`, else ONE bounded `synthesize_object` search (built-in portfolio without the Hypothesis pass and without nested sub-searches, no backend, no probe matrix, `require_flattening=True`; when nothing validated, further stages run, each within what the earlier ones left of the time bound: Ghostwriter in hub mode where `hypothesis` and `black` are installed -- the strategies it writes for the constructor's arguments are run and what they built is a candidate recipe, replayed like any other -- then the Hypothesis pass) whose evidence stays on `kb`; `roots` (the request's, [internal ref]) go first on the search children's `PYTHONPATH` (`executor.py::IsolatedExecutor.roots`) |
| `template_of(kb, source, class, recipe)` | the `FlatObjectTemplate` mined for that validated recipe, when `FlattingPass` mined one |
| `unsupported(recipe)` | why a recipe cannot be a parameter: an argument that is itself a constructed object (nested objects: ruled, not built), a setup call of a module function |

Failures are `ObjectUnavailable(code, detail)`: `object_unresolved` (nothing validated in the budget: unresolved, not
absent), `object_recipe_unsupported`, `flattening_unavailable` (the companion is required: the search raises
`FlattingUnavailable` before it spends anything).

*Measured, GUUT classes (survey, 20 s, built-in portfolio):* recipes validate for all of them, and a template is mined
for `__ISBNChecker` (`$.input_string`), `__StringFormatter` (`$.input_string`), `_ExtendedEncoder` (six stdlib encoder
fields) and the dataclass subclasses `DataClassJsonMixin` / `_CatchAllUndefinedParameters` build; none for `Timer`
(`seed_refused: unresolved occurrence at $._start_time` and `$.logger`), `Timers` (`'Timers' does not support item
assignment`: a `UserDict` rebuilt by keyword), `_IsoField`, `_TimestampField` and `_UnionField` (`reference at
$.load_default cannot be recovered`). That is why the leaves are the recipe's arguments, which exist wherever Avengers can
build the class, and the template is reported beside them.

*Works as a question.* `works.py::guard_tests` returns the entry guards as `(kind, test)`; `find_inputs`' `{"on": "@call",
"op": "works"}` evaluates them as source text before the call (docs/INPUTS.md "Want"), the same rule `Guards.ran_past` reads
off line events.

**Reproducibility ([internal ref]).** The Hypothesis searches of the portfolio (the `find` of the constructor pass, the constructor
candidates of `_worker.py`) draw from the request's one seed (`hints.rng_seed`, `solve/seed.py`, docs/INPUTS.md "Reproducibility"):
a child inherits it as `THARKUN_RNG_SEED`. The same seed gives the same search; the recipe a search finds is cached per `Session`, so
a session that already holds a recipe answers from it whatever the seed.

## The companion bridge

Avengers and [`py_obj_flatting`](https://github.com/ardier/py_obj_flatting) are
both parts of Tharkun. The latter flattens a live object into a deterministic,
path-addressed tree and turns an exemplar into a byte-addressable object
recipe. It keeps its own distribution and safety contract, and **Avengers
requires it** (owner ruling 2026-10-21, [internal ref]): it is a declared dependency of
the `avengers`, `backends` and `bench` extras, pinned by commit
(`pyproject.toml`), and the explicit bridge API and `FlattingPass` use it:

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

Importing `tharkun.avengers` does not load it, and construction search and
boundary replay still run when it is absent -- but nothing then pretends to
observe state: every outcome carries the refusal `flattening-unavailable`,
`synthesize_object` warns once and `report.refusals` names it, and
`synthesize_object(..., require_flattening=True)` (what the GUUT A/B harness
passes) raises `FlattingUnavailable` before spending anything. The default
portfolio adds `FlattingPass` when it is importable.

When the default portfolio creates a template, its recipe crosses the worker
boundary as versioned JSON. Loading that payload verifies its SHA, slot layout,
duplicated metadata, and exemplar leaf paths. The surrounding worker response
also uses a bounded, data-only JSON vocabulary, so the parent never unpickles
data produced after target code has run.

## Guessed invariants, proved facts

**The rule (owner, 2026-10-19; [internal ref]):** Avengers' example objects guess class invariants, and an independent checker
proves them. Only proved ones become facts.

- **Producers propose** a candidate invariant `I` over a class's fields. There are three:
  - Avengers' validated receiver states (`src/tharkun/avengers/objectives.py::ReceiverState`). Ruled, not built
    ([internal ref]).
  - SV's first run. `src/tharkun/sv/_child.py::_observe_fields` samples `[min, max]` of the int fields that receiver
    queries read (`src/tharkun/solve/absint/contracts.py::demands`) and passes them on as `class_hints`.
  - absint: its inductive class-contract check (`src/tharkun/solve/absint/contracts.py::infer`) and its location
    candidates (`src/tharkun/solve/absint/interp.py::Analyzer.capture_invariant`). absint really does prove its
    contracts, but here its proof is a claim.
- **The SV-COMP checker judges.** That checker is `src/tharkun/sv/validate.py`. It is the `CHECKER` of
  `src/tharkun/sv/fact_proof.py::prove_from_facts`, and `src/tharkun/sv/proof.py::finish` uses it to set an SV
  witness's `proof_checked`.
- **Only a confirmed candidate becomes a fact.** `src/tharkun/sv/eregion_bridge.py::publish_certified` publishes it as
  `CERTIFIED_INVARIANT` evidence in `src/tharkun/knowledge.py::KnowledgeBase`.

The judge is independent for the same reason SV-COMP checks witnesses: a producer's proof is only as sound as the
producer, and a guesser is heuristic by design. The checker fails closed (`Unchecked` on anything it does not model).
It never reads a producer's reasoning, it does not use absint's transfer functions, and it re-proves each obligation
from the source with Z3. A bug in one producer therefore cannot create a fact.

### Why a proved invariant holds

The claim is that `I` holds in every state of every instance of `C` reachable through `C`'s API. The proof is by
induction over every finite sequence of calls:

- **(a) Establishment.** Every way an instance comes into existence makes `I` true: `__init__` on every path,
  classmethod and staticmethod factories, `__new__`, `copy`/`pickle` reconstruction, and `object.__new__(C)`.
- **(b) Preservation.** Every method, entered in a state satisfying `I` (the *receiver precondition*) with arguments
  of their declared sorts, leaves `I` true on every exit. That means a normal return **and** an exception exit,
  because a caller can catch the exception and keep the partly updated object.
- **(c) Closure.** Nothing but `C`'s methods writes the constrained fields; otherwise (b) does not cover the step. The
  checker must enforce each of the following, or refuse it by name:
  - writes from outside the class (module code, other classes, callers);
  - aliasing or escape of mutable fields, such as a returned list the caller mutates;
  - subclasses, overrides and dynamic dispatch;
  - reflection (`setattr`/`delattr`, `__dict__`, `__class__` assignment, `__slots__` tricks,
    `object.__setattr__`);
  - callbacks and user hooks run mid-method, while `I` may be broken;
  - descriptors and properties;
  - module reload and monkeypatching.

(a) is the base case and (b) is the step; (c) makes every step a method call.

**Examples never certify.** Exemplars are a finite sample of the reachable states, which are usually infinite. A
counter that breaks only on its 11th `up()` satisfies a bound guessed from every exemplar of a search that never calls
`up()` more than ten times. Examples only rank and propose: a candidate false on some exemplar is dropped, and one
that fits is worth a proof. The proof is what promotes it.

**What a certified `I` means.** It holds on every state reachable through the class's API, given the closure the
checker enforced and arguments of their annotated sorts. It is meant to prune and bound leaves for flattening, the
hybrid fuzzer and `drive_to_line`. It is never a kill, and never a TRUE by itself: an SV TRUE still needs every
obligation of the property checked (step 5 of `prove_from_facts`).

### What the checker can judge today

`src/tharkun/sv/validate.py::ClassInvariantCandidate(file_name, line, column, class_name, value)` poses a
class-wide fact at the original module-level class definition. `check_candidates` and
`check_candidates_individually` judge it using the same independent source interpreter as ordinary invariants.
The predicate can name only receiver fields (`self__f`) and predicate builtins. It is internal fact data:
it has no witness serialization, does not publish itself to the store, and never proves program safety by itself.
Class facts are judged separately from ordinary location/loop candidates, whose unconstrained-entry meaning stays intact.

The shared pure-data detector remains the acceptance boundary. The checker establishes the candidate after
`__init__` without assuming it, then assumes it on method entry and proves it at every modeled normal exit.
Every `def` in the class body is included. A dunder skipped by the shared detector refuses the class fact;
it is never omitted from the conjunction. No trailing `pass` marker is needed, and a scalar return expression
is evaluated before checking the exit. The field renderer now includes return-expression reads.

Every field needs an independently recovered exact scalar sort: a direct annotated `__init__` parameter or
an exact `int`, `float` or `bool` literal, including signed numeric literals. Every method must keep those sorts.
For example, `__init__: self.n = 0` and `advance: self.n = self.n + 1` establish and preserve `self__n >= 0`.
The old location-only obligation for `advance` cannot prove that bound from arbitrary negative input fields.

Receiver identity cases are exhaustive. When a method writes a field read through another instance parameter,
`validate.py::_Checker.method_cases` checks every combination in which those parameters are distinct from, or alias,
`self`. Aliased fields start with the same symbolic value, and `::_Checker.block` propagates plain field assignments
to every alias before checking later locations. Parameters reading only unaffected fields retain independent,
conservative symbols. No disjointness or alias-case limit is guessed; the existing checker budget fails closed.
This rejects the false claim `self.y == 0` for a method that saves `other.x`, increments `self.x`, then stores
`other.x - saved`: `obj.method(obj)` leaves `y == 1`.

Class facts check the field invariant and established field sorts on modeled division-by-zero exits, regardless
of the requested SV property. The exceptional state is the receiver state before the failing expression's assignment.
The normal continuation requires the divisor to be nonzero only when that expression is evaluated; a conditional
arm cannot remove its sibling's normal path. This covers integer floor division and remainder in the admitted
straight-line body and return expressions. Helpers, true division and uncovered exit shapes remain refused.
A method that divides while leaving `level = 0` can preserve that class fact even when division raises. A method
that sets `level = -1`, divides, then restores `level = 0` fails if its exception exit leaves the receiver broken.
Ordinary location invariants retain their original property-relative location claims.

| Requirement | Current boundary |
| --- | --- |
| Establishment and preservation | `__init__` from unconstrained annotated arguments; every method from the receiver precondition |
| Other construction paths | decorators, factories, `__new__`, reconstruction and reflective allocation refused |
| Exceptions | field invariant and sorts checked on modeled division-by-zero exits; uncovered exits refused |
| Method shapes | shared straight-line assignments and scalar returns; class-fact calls limited to modeled `abs`, `min`, `max`, `int` |
| Field sorts | direct annotated parameters or exact scalar literals; expressions with no independently recovered sort refused |
| Aliasing | every observable receiver/instance-parameter identity case; no mutable fields admitted |
| Actual class uses and outside writes | module construction/calls remain unmodeled, so using the class in the source still prevents whole-program certification |
| Inheritance, descriptors, reflection, callbacks | refused by the existing class and method boundaries |

These are **conditional class facts**: exact field/argument sorts and API closure remain required at every use site.
Instance parameters must also be exact instances of the modeled pure-data class; arbitrary objects whose
properties return scalar values are outside this plain-field model.
The checker does not yet discharge those use-site obligations for a program constructing and using the class.
An absint TRUE resting on its own contract therefore still has `proof_checked=False` when that use cannot be checked.
No producer verdict or sampled state enters the independent proof.

**Built:** per-location and per-class obligations; the store's certified-only door (`publish_fact` proposes,
`publish_certified` publishes checked location facts); SV field samples; absint's own contract check, described in
[ABSINT.md](ABSINT.md) "Inductive class contracts".

**Still unbuilt ([internal ref]/[internal ref]/[internal ref]/[internal ref]):**
- Avengers exemplars to candidates. Its worker reads exact scalar fields, but does not propose this class shape yet.
- Checking construction, actual argument sorts and closure at use sites.
- Broader exception kinds, method bodies and helper calls for class facts.
- `prove_from_facts` reading class obligations back and consumers using certified class facts.

**Demonstration.** `tests/test_guessed_invariants_demo.py` retains the location-only clamped `Dial`, the
11th-call `Meter` counterexample, shape refusals, the producer-claim boundary and a broken exceptional state.
`tests/test_class_invariant_candidates.py` exercises the class-wide precondition, constructor/method/closure
negative controls, return reads, aliasing, normal and exceptional continuations through conditional expressions,
property-independent exit checks, source anchors and budget expiry.

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
  `objectives.py::NegativeSequencePass.run` publishes only confirmed sequences; a timeout,
  normal return or different exception in the confirming child is not a negative finding ([internal ref]).

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

### Authored setup replay budget ([internal ref])

`src/tharkun/avengers/api.py::synthesize_object` accepts optional `setup_replay_seconds`, validated by
`src/tharkun/avengers/model.py::validate_setup_replay_seconds` (positive, finite, <=300; booleans refused).
It changes the existing isolated executor's timeout and ensures the portfolio pass budget can accommodate it.
The overall saturation/request deadline is still a separate bound. The `find_inputs` object-search path
(`src/tharkun/avengers/flatobject.py::find_recipes`) uses five seconds when the request omits the override.
Explicit executors keep their other settings; the direct API's omitted-argument behavior is unchanged.
`src/tharkun/avengers/bench.py::run_one` and `::run_benchmark` forward this optional override through the same API;
their omitted five-second executor configuration and result format stay unchanged.
An explicit saturation budget with `max_pass_seconds=None` remains unlimited; only an existing finite pass cap is raised.
