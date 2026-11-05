"""Independent, fail-closed checking of Python invariant witnesses ([internal ref]).

The producer/absint verdict is never an input. The census/resolver is a shared
trusted coverage boundary. Closed scalar dynamic-typing programs instead use
an explicit syntax boundary and always need the small source interpreter below; it does not use absint transfer functions or facts.
All scalar VCs use Z3's Python API, with unbounded ints and IEEE binary64.
"""
from __future__ import annotations

import ast
import hashlib
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

import z3

from tharkun.solve.census import obligation_census
from tharkun.solve.obligations import Invariant, source_points
from tharkun.solve.predicates import BUILTINS, check_predicate
from tharkun.sv.method_lift import _init_field_types, lift_class, lift_method
from tharkun.sv.spec import parse_property
from tharkun.sv.witness import loads


class Unchecked(ValueError):
    pass


@dataclass(frozen=True)
class Validation:
    confirmed: bool
    open_obligations: tuple
    reason: str = ""


@dataclass(frozen=True)
class ClassInvariantCandidate:
    """An internal class fact, conditional on exact field/argument sorts and API closure.

    The anchor must name the original module-level class definition. This is not
    a serializable SV witness entry or a certificate of program safety.
    Instance parameters require exact instances of the modeled pure-data class;
    arbitrary objects with properties are outside the field-read model.
    """
    file_name: str
    line: int
    column: int
    class_name: str
    value: str
    type = "class_invariant"
    function = "<module>"

    def __post_init__(self):
        Invariant(self.file_name, self.line, self.column, self.function, self.value)
        if not isinstance(self.class_name, str) or not self.class_name.isidentifier():
            raise ValueError("invalid class name")


def validate(source: str, filename: str, specification: str, text: str, *, seconds=3.0, census=None,
            source_bytes: bytes | None = None) -> Validation:
    """Check bindings, every supplied invariant, coverage, and every operation
    of an EXTERNAL witness (its exact YAML ``text``): metadata/source/property
    binding is verified before anything else touches the invariants inside.

    ``census`` lets a caller that already computed the obligation census (the
    finalizer, which is checking its own freshly built witness, not an
    external one) pass it in instead of paying to recompute it -- [internal ref]
    review: the finalizer recomputed this census a second time. An external
    caller (``--validate-witness``, a real independently supplied witness)
    leaves it None and this recomputes it as before.

    ``source_bytes``, when given, is what the expected ``input_file_hashes``
    binds against instead of a UTF-8 re-encoding of ``source`` ([internal ref]) --
    see :func:`tharkun.solve.obligations.witness`, which the producer side
    uses the same way.
    """
    _, classes = parse_property(specification)
    if census is None:
        census = obligation_census(source, classes, filename)
    try:
        if sys.version_info[:2] != (3, 14):
            raise Unchecked("checker requires CPython 3.14 semantics")
        if any("init(main())" not in line for line in specification.splitlines() if line.strip()):
            raise Unchecked("unsupported entry point")
        entries = loads(text)
        invariants = []
        digest = hashlib.sha256(source_bytes if source_bytes is not None
                                else source.encode("utf-8")).hexdigest()
        for entry in entries:
            meta = entry["metadata"]
            if set(meta) != {"format_version", "uuid", "creation_time", "producer", "task"}:
                raise Unchecked("unknown/missing metadata fields")
            expected = {"input_files": [filename], "input_file_hashes": {filename: digest},
                "specification": specification, "language": "Python"}
            if meta["format_version"] == "2.0":
                expected["data_model"] = "CPython-3.14-binary64"
            if meta["task"] != expected:
                raise Unchecked("source/property/semantics binding mismatch")
            invariants += [Invariant.from_entry(i) for i in entry["content"]]
        if not invariants:
            raise Unchecked("empty witness")
        return _check(source, filename, classes, census, invariants, seconds)
    except Exception as e:  # parsing/solver/resource failure proves nothing
        return Validation(False, census.obligations, f"{type(e).__name__}: {e}")


def check_candidates(source: str, filename: str, specification: str, census, invariants: list, *, seconds=3.0) -> Validation:
    """Independently check internal location/loop or class candidates -- same checker,
    same VCs as :func:`validate`, but for a caller (the finalizer) that just
    built these candidates itself and has no YAML text to round-trip. [internal ref]
    review: ``finish()`` used to build a witness, serialize it, and call
    ``validate()``, which immediately re-parsed that same YAML back into
    ``Invariant`` objects -- pure overhead when producer and checker are the
    same call. There is nothing to authenticate here (no external metadata to
    bind against); an externally supplied witness must go through
    :func:`validate`, which does.
    """
    _, classes = parse_property(specification)
    try:
        if sys.version_info[:2] != (3, 14):
            raise Unchecked("checker requires CPython 3.14 semantics")
        if any("init(main())" not in line for line in specification.splitlines() if line.strip()):
            raise Unchecked("unsupported entry point")
        if not invariants:
            raise Unchecked("empty witness")
        return _check(source, filename, classes, census, invariants, seconds)
    except Exception as e:
        return Validation(False, census.obligations, f"{type(e).__name__}: {e}")


@dataclass(frozen=True)
class ReachabilityCandidate:
    """[internal ref] SEV1, slice 2a shape (b) (owner, [internal ref]): Eregion can discharge
    a site when the conjunction of ``bw_assume`` facts on its inverter path back to
    ``main`` is provably UNSAT. W57 emits the producer side ([internal ref]'s
    piece D); NOT built here yet -- this dataclass fixes the input shape a producer
    should target, so it can be built in parallel.

    ``site`` -- a site identifier in the SAME shape ``solve/census.py``'s own site
    keys use (an ``ast_path`` string), not a bare line number: this shape reasons
    about a PATH, not one AST statement, so it has no single ``(file, line, column,
    function)`` the way an ``Invariant`` does.
    ``constraints`` -- ``python_expression`` strings (the SAME grammar
    ``solve/predicates.py::parse_predicate`` already accepts for ``Invariant.value``),
    ANDed together as the path's own backward-assume conjunction.
    ``sound`` -- one bool per entry of ``constraints``, in order: ``True`` iff that
    ONE constraint is a sound over-approximation of the path (never a search hint,
    never "probably true"). ALL must be ``True`` or the whole candidate stays a hint.
    ``exhaustive`` -- ``True`` iff every branch point on the path back to ``main`` is
    accounted for (no branch this path could also have taken was left out of the
    conjunction). Also required for certification.

    Certification (not built): independently re-run the conjunction (translated to
    Z3, reusing ``_Checker.expr``'s own predicate-to-Z3 machinery) and confirm
    ``solver.check() == unsat`` -- never trust a producer-supplied verdict string."""
    site: str
    constraints: tuple[str, ...]
    sound: tuple[bool, ...]
    exhaustive: bool


@dataclass(frozen=True)
class EnumerationRefutationCandidate:
    """[internal ref] SEV1, slice 2a shape (c) (W57, [internal ref]): a draw with a finite,
    fully-enumerated, CERTIFIED domain (e.g. nbody's ``rindex`` over 5 literal
    ``BODIES`` entries -- ``literal_domain_cardinality``; scimark's ``1 + abs(draw)``
    -- ``operator_inverse_cardinality``), where every value in the domain is checked
    safe at the site. PROVISIONAL: W57's exact schema was not posted as of this
    writing ([internal ref] checked 2026-09-29 ~01:30 and ~02:55); this
    dataclass is my own best-effort shape from the owner's prose description, subject
    to revision once the real schema lands -- do not build a producer against this
    without confirming it first.

    ``site`` -- same ``ast_path`` shape as :class:`ReachabilityCandidate`.
    ``domain_kind`` -- ``"literal_domain_cardinality"`` or
    ``"operator_inverse_cardinality"`` (W57's own two sub-kinds).
    ``domain`` -- every value the draw can take, as encoded ``sv-py-witness/1``
    values (a float as its ``.hex()``, matching ``_child.py``'s own draw log
    encoding) -- the CLAIMED enumeration, not yet independently confirmed complete.
    ``domain_certified`` -- ``True`` only when the enumeration's own completeness is
    independently re-derivable (a literal table read straight from the source AST,
    a module-level constant never reseeded/mutated at runtime) -- never trust the
    producer's own claim that its enumeration is exhaustive.
    ``per_value_safe`` -- one ``python_expression`` per entry of ``domain``, in the
    same order, each an independent safety claim for that one value at ``site``.

    Certification (not built): independently re-derive ``domain``'s completeness
    from the source (per ``domain_kind``), then independently check every
    ``per_value_safe`` entry."""
    site: str
    domain_kind: str
    domain: tuple
    domain_certified: bool
    per_value_safe: tuple[str, ...]


@dataclass(frozen=True)
class Family:
    """One stated periodic family in a :class:`FamilyIntersectionCandidate` --
    raw data only (bases + period), never a verdict/complete/solutions field: those
    are the producer's OWN claim (eregion_codegen/objects/wrapper.py::AnswerSet),
    which this shape must never trust directly (2026-09-29 exchange, bd
    [internal ref])."""
    fact: str        # e.g. "x == sin(i)" -- the python_expression this family satisfies
    bases: tuple      # the family's base values (before adding period*k)
    period: str       # a python_expression for the period (e.g. "2*pi")


@dataclass(frozen=True)
class FamilyIntersectionCandidate:
    """[internal ref] SEV1, slice 2a shape (d) (another lane, 2026-09-29, [internal ref]):
    two or more periodic families of values provably share no common member, so the
    joint state they would need to co-occur is unreachable (e.g. bm_float_nondet:
    sin(i) and cos(i) never both land on the same residue, so norm never hits exactly
    0). Raw data only -- see :class:`Family`.

    ``domain`` -- ``"real"`` (the producer's own proof domain, per
    eregion-codegen#122's added field) or ``"float64"``. Owner ruling: a "real"
    proof is NEVER by itself a proof for the actual IEEE-754 double-precision
    program (rounding, underflow can make a real-number emptiness untrue for
    doubles) -- certification for a ``domain="real"`` candidate additionally needs
    an independent FLOAT-SOUND argument (an interval/ULP bound, or a Z3 FP-theory
    query reusing ``_Checker``'s own existing float VC machinery), never symbolic/
    real modular arithmetic alone. A candidate with ``domain="float64"`` already
    proven float-sound by the producer still needs independent re-derivation, same
    rule as every other shape -- ``domain`` says what MATHS the claim is stated in,
    it is never itself something to trust."""
    site: str
    families: tuple[Family, ...]
    domain: str


def check_candidates_individually(source: str, filename: str, specification: str, census,
                                  candidates: list, *, seconds: float = 3.0) -> dict:
    """[internal ref] SEV1, slice 2a (owner, 2026-09-29; [internal ref]): per-CANDIDATE
    confirmation, needed before any proof component may consult an independently
    checked fact -- a single uncertified ``candidate_invariant`` may NEVER be trusted
    on its own (it is "never trusted by the witness checker" by construction, see
    absint/interp.py::capture_invariant); only a fact THIS function independently
    reconfirms may ever become ``certified=True`` anywhere downstream.

    Calls :func:`check_candidates` once per candidate, with that candidate alone.
    Location/loop claims retain their ordinary obligations. A class candidate
    checks establishment and every method's preservation under its precondition;
    it does not enter the serialized SV witness format.

    Returns ``{candidate: bool}`` (``True`` iff THAT candidate, alone, is inductively
    valid by the checker's own existing rules: a ``location_invariant`` at its own
    ``at()`` call, a ``loop_invariant`` at its own base case AND inductive step in
    :meth:`_Checker.loop`). ``seconds`` is PER candidate (the caller manages its own
    total budget by choosing how many candidates to check and how large ``seconds``
    is for each).

    Sound but INCOMPLETE by construction: a program whose safety genuinely needs
    several invariants working TOGETHER (e.g. two co-dependent loops, or a
    location_invariant that only holds because an ENCLOSING loop's own invariant is
    also assumed) will report every one of them ``False`` here, even though the
    WHOLE SET might jointly confirm through :func:`check_candidates` given all of
    them at once. That is a completeness gap (a missed certification, costing
    nothing but precision), never a soundness one (this never reports ``True`` for
    something the checker would not also accept as the sole, self-sufficient
    justification for its own location).

    ``Invariant`` location/loop claims and ``ClassInvariantCandidate`` class facts
    are implemented. Class facts pose establishment plus every method's preservation,
    conditional on exact sorts and API closure; they are not SV witness entries. Three other shapes
    have a fixed INPUT contract (their dataclasses, above) but no verification logic
    yet -- :class:`ReachabilityCandidate` (Eregion's ``bw_assume``/GUARD conjunction),
    :class:`EnumerationRefutationCandidate` (W57's exhaustive-enumeration refutation,
    provisional pending W57's own schema), :class:`FamilyIntersectionCandidate`
    (periodic-family disjointness). Each raises its own, specific
    :class:`NotImplementedError` below rather than a generic one, so a producer
    building against the dataclass shape gets an accurate "not verified yet" rather
    than "unrecognized shape". Passing anything else raises the generic form."""
    results: dict = {}
    for candidate in candidates:
        if isinstance(candidate, (Invariant, ClassInvariantCandidate)):
            v = check_candidates(source, filename, specification, census, [candidate], seconds=seconds)
            results[candidate] = v.confirmed
            continue
        if isinstance(candidate, ReachabilityCandidate):
            raise NotImplementedError(
                "check_candidates_individually: ReachabilityCandidate (bw_assume/GUARD) "
                "has a fixed input shape but no Z3 UNSAT verification yet")
        if isinstance(candidate, EnumerationRefutationCandidate):
            raise NotImplementedError(
                "check_candidates_individually: EnumerationRefutationCandidate "
                "(exhaustive-enumeration) has a PROVISIONAL input shape (W57's real "
                "schema not yet posted) and no verification yet")
        if isinstance(candidate, FamilyIntersectionCandidate):
            raise NotImplementedError(
                "check_candidates_individually: FamilyIntersectionCandidate "
                "(periodic-family disjointness) has a fixed input shape but no "
                "float-sound verification yet")
        raise NotImplementedError(
            f"check_candidates_individually: unrecognized candidate shape {type(candidate).__name__} "
            "-- see this module's Invariant/ClassInvariantCandidate/ReachabilityCandidate/"
            "EnumerationRefutationCandidate/FamilyIntersectionCandidate for the shapes this "
            "function knows about")
    return results


def _none_identity_guard(node):
    return (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
            and len(node.ops) == len(node.comparators) == 1
            and isinstance(node.ops[0], (ast.Is, ast.IsNot))
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value is None)


def _main_annassignments(tree):
    # shortcut: direct main-body Name: object = primitive RHS only; extend for a checked need.
    mains = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    call = tree.body[-1] if tree.body else None
    if (len(mains) != 1 or not isinstance(call, ast.Expr)
            or not isinstance(call.value, ast.Call) or not isinstance(call.value.func, ast.Name)
            or call.value.func.id != "main" or call.value.args or call.value.keywords):
        return set()
    main = mains[0]
    if (main.decorator_list or main.type_params or main.args.args or main.args.posonlyargs
            or main.args.kwonlyargs or main.args.vararg or main.args.kwarg):
        return set()
    return {node for node in main.body if isinstance(node, ast.AnnAssign)
            and node.simple == 1 and isinstance(node.target, ast.Name)
            and isinstance(node.annotation, ast.Name) and node.annotation.id == "object"
            and node.value is not None}


def _scalar_typing_coverage(tree):
    """Sufficient syntax boundary; argument/operation safety is still re-proved by _Checker."""
    # shortcut: scalar leaves, tuple wrappers and fixed ranges of at most four; extend for a checked proof need.
    allowed = (ast.Module, ast.Import, ast.ImportFrom, ast.alias, ast.FunctionDef,
               ast.arguments, ast.arg, ast.Lambda, ast.Assign, ast.AnnAssign, ast.Expr, ast.Return, ast.If, ast.IfExp, ast.For, ast.Assert,
               ast.Pass, ast.Constant, ast.Name, ast.Load, ast.Store, ast.Call, ast.keyword, ast.Attribute, ast.Tuple,
               ast.UnaryOp, ast.UAdd, ast.USub, ast.Not, ast.BinOp, ast.Add, ast.Sub,
               ast.Mult, ast.FloorDiv, ast.Mod, ast.LShift, ast.BitAnd, ast.BoolOp, ast.And, ast.Or,
               ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot)
    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    none_guards = {node.test for node in ast.walk(tree)
                   if isinstance(node, ast.If) and _none_identity_guard(node.test)}
    main_annotations = _main_annassignments(tree)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise Unchecked(f"unmodeled scalar typing construct {type(node).__name__}")
        if isinstance(node, ast.AnnAssign) and node not in main_annotations:
            raise Unchecked("annotated assignments require a direct main local Name: object value")
        if isinstance(node, ast.Tuple):
            owner = parent[node]
            if not (isinstance(owner, ast.Return) and owner.value is node
                    or isinstance(owner, ast.Assign) and owner.targets == [node]
                    and isinstance(owner.value, ast.Call)):
                raise Unchecked("tuple is only a helper return or an immediate unpack target")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.LShift) and not (
                isinstance(node.left, ast.Constant) and type(node.left.value) is int
                and isinstance(node.right, ast.Constant) and type(node.right.value) is int
                and 0 <= node.right.value <= 64):
            raise Unchecked("scalar typing left shifts require literal integers and counts 0..64")
        if isinstance(node, ast.Import) and any(a.name != "_sv_verifier" for a in node.names):
            raise Unchecked("scalar typing coverage requires closed int/bool imports")
        if isinstance(node, ast.ImportFrom) and (
                node.module != "_sv_verifier" or node.level
                or any(a.name not in {"nondet_int", "nondet_bool", "check_type"} for a in node.names)):
            raise Unchecked("scalar typing coverage requires closed int/bool imports")
        if isinstance(node, ast.Constant) and type(node.value) not in {int, bool}:
            inert = isinstance(parent.get(node), ast.Expr) and type(node.value) is str
            none_return = isinstance(parent.get(node), ast.FunctionDef) and node.value is None
            main_name = node.value == "__main__" and isinstance(parent.get(node), ast.Compare)
            none_guard = node.value is None and parent.get(node) in none_guards
            if not (inert or none_return or main_name or none_guard):
                raise Unchecked("scalar typing coverage requires int/bool values")
        if isinstance(node, ast.Attribute) and not (
                isinstance(parent.get(node), ast.Call) and parent[node].func is node
                and isinstance(node.value, ast.Name) and node.attr in {"nondet_int", "nondet_bool", "check_type"}):
            raise Unchecked("unmodeled scalar typing attribute")
        if isinstance(node, ast.arg) and node.annotation is not None and not (
                isinstance(node.annotation, ast.Name) and node.annotation.id in {"int", "bool", "object"}):
            raise Unchecked("scalar typing parameters require primitive annotations")
        if isinstance(node, ast.FunctionDef):
            if node not in tree.body or node.decorator_list or node.type_params:
                raise Unchecked("scalar typing functions must be plain module-level definitions")
            if node.args.args:
                if node.returns is not None and (not isinstance(node.returns, ast.Name)
                                               or node.returns.id not in {"int", "bool"}):
                    raise Unchecked("scalar typing helpers require int/bool returns")
                if node.returns is None and not any(isinstance(n, ast.Return) and n.value is not None
                                                     for n in ast.walk(node)):
                    raise Unchecked("scalar typing helpers require a valued return")
            else:
                # The module walk must actually check the property's main body, even for a True baseline.
                last = tree.body[-1]
                returned = node.body[1:] if (isinstance(node.body[0], ast.Expr)
                           and isinstance(node.body[0].value, ast.Constant)
                           and type(node.body[0].value.value) is str) else node.body
                factory = (node.name != "main" and node.returns is None and returned
                           and (len(returned) == 1 and isinstance(returned[0], ast.Return)
                                and returned[0].value is not None
                                or isinstance(returned[-1], ast.Return)
                                and isinstance(returned[-1].value, ast.Tuple)))
                if not factory and not (node.name == "main" and isinstance(last, ast.Expr)
                        and isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Name)
                        and last.value.func.id == "main" and not last.value.args and not last.value.keywords):
                    raise Unchecked("scalar typing entry requires an unconditional main() call")


def _check(source, filename, classes, census, invariants, seconds) -> Validation:
    """The shared core: re-prove every invariant against the source. Raises
    :class:`Unchecked` (or lets a solver/parse exception propagate) on any
    failure; both public entry points above catch and report it."""
    tree = ast.parse(source, filename)
    if (any(isinstance(i, ClassInvariantCandidate) for i in invariants)
            and not all(isinstance(i, ClassInvariantCandidate) for i in invariants)):
        raise Unchecked("class facts must be judged separately from location and loop invariants")
    points = source_points(tree, source, filename)
    locations = {loc for node, (_, loc) in points.items() if isinstance(node, ast.stmt)}
    if not tree.body:
        locations.add((filename, 1, 1, "<module>"))
    for inv in invariants:
        if (inv.file_name, inv.line, inv.column, inv.function) not in locations:
            raise Unchecked("invariant is not at an original statement")
        if isinstance(inv, ClassInvariantCandidate) and not any(
                isinstance(node, ast.ClassDef) and node in tree.body and node.name == inv.class_name
                and loc == (inv.file_name, inv.line, inv.column, inv.function)
                for node, (_, loc) in points.items()):
            raise Unchecked("class invariant must locate its original module-level class definition")
        if inv.type == "loop_invariant" and not any(
                isinstance(node, ast.While) and loc == (inv.file_name, inv.line, inv.column, inv.function)
                for node, (_, loc) in points.items()):
            raise Unchecked("loop invariant must locate an original while head")
    typing = bool(set(classes) & {"TypeError", "AttributeError"})
    if typing:
        if set(classes) - {"TypeError", "AttributeError"}:
            raise Unchecked("scalar typing coverage supports only TypeError/AttributeError")
        expected_gaps = {f"{exc}: no site table for {exc}" for exc in classes}
        if set(census.coverage_gaps) - expected_gaps:
            raise Unchecked("; ".join(census.coverage_gaps))
        _scalar_typing_coverage(tree)
    elif not census.complete:
        raise Unchecked("; ".join(census.coverage_gaps))
    if not typing and not census.obligations and all(
            isinstance(i, Invariant) and i.value == "True" for i in invariants):
        return Validation(True, ())
    deadline = time.monotonic() + seconds
    modes = ("strict", "primitive-exit", "closed-callable") if set(classes) == {"AssertionError"} else ("strict",)
    strict_reason = ""
    for mode in modes:
        try:
            checker = _Checker(tree, points, classes, invariants, deadline - time.monotonic(),
                               primitive_exits=mode == "primitive-exit", closed_callables=mode == "closed-callable")
            checker.block(tree.body, {"__name__": ("str", "__main__")}, z3.BoolVal(True))
            if checker.unvisited:
                raise Unchecked("invariant location not covered by source checking")
            if mode == "primitive-exit" and not checker.primitive_check_seen:
                raise Unchecked("primitive exit checking requires an authenticated int checker call")
            if mode == "closed-callable" and not checker.closed_callable_seen:
                raise Unchecked("closed callable checking requires an actual authenticated call")
            return Validation(True, ())
        except Unchecked as error:
            strict_reason = (f"{strict_reason}; {mode} retry: {error}" if strict_reason else str(error))
            if mode == modes[-1]:
                break
    # A tautological witness contributes no assumptions. Rebuild any mixed
    # cover from the source, never from published status/coverage claims.
    if (set(classes) == {"AssertionError"} and all(isinstance(i, Invariant)
            and i.type == "location_invariant" and i.value == "True" for i in invariants)
            and time.monotonic() < deadline):
        from tharkun.sv.combined_proof import cover
        if cover(source, filename, ("AssertionError",), census=census,
                 seconds=deadline - time.monotonic()) is not None:
            return Validation(True, ())
    raise Unchecked(strict_reason)


class _Checker:
    WORK_BUDGET = 10000

    #: what ``imports`` maps an ``import _sv_verifier [as m]`` alias to (a draw imported by name
    #: maps to its sort instead)
    MODULE = "<module>"
    CHECK_TYPE = "<check_type>"

    def __init__(self, tree, points, classes, invariants, seconds, *, primitive_exits=False, closed_callables=False):
        # Builtin lookup in newly created functions depends on the module's f_builtins.
        for node in ast.walk(tree):
            if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == "__builtins__"
                    or isinstance(node, ast.arg) and node.arg == "__builtins__"
                    or isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == "__builtins__"
                    or isinstance(node, ast.alias) and (node.asname or node.name.split(".")[0]) == "__builtins__"):
                raise Unchecked("rebound builtin namespace")
        self.tree, self.points, self.classes = tree, points, set(classes)
        self.main_annotations = _main_annassignments(tree)
        self.none_guards = {node.test for node in ast.walk(tree)
                            if isinstance(node, ast.If) and _none_identity_guard(node.test)}
        self.invariants = {}
        self.class_invariants = {}
        for i in invariants:
            destination = self.class_invariants if isinstance(i, ClassInvariantCandidate) else self.invariants
            destination.setdefault((i.file_name, i.line, i.column, i.function), []).append(i)
        self.unvisited = set(self.invariants) | set(self.class_invariants)
        self.deadline = time.monotonic() + max(0, seconds)
        self.functions = {}
        self.factory_helpers = set()
        self.bounded_helpers = set()
        self.tuple_helpers = set()
        self.tuple_calls = {
            node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple)
            and isinstance(node.value, ast.Call)
        }
        self.helper_calls = []
        self.imports = {}
        bindings = {}
        for node in ast.walk(tree):
            names = ()
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                names = (node.id,)
            elif isinstance(node, ast.arg):
                names = (node.arg,)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                   ast.ExceptHandler, ast.MatchAs, ast.MatchStar,
                                   ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple)):
                names = (node.name,)
            elif isinstance(node, ast.alias):
                names = (node.asname or node.name.split(".")[0],)
            elif isinstance(node, ast.MatchMapping):
                names = (node.rest,)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                names = node.names
            for name in names:
                if name is not None:
                    bindings[name] = bindings.get(name, 0) + 1
        self.constant_names = {
            s.targets[0].id for s in tree.body
            if isinstance(s, ast.Assign) and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Constant)
            and type(s.value.value) in {int, bool} and bindings[s.targets[0].id] == 1
            and "*" not in bindings
        }
        self.bindings = bindings
        self.readonly_names = {s.targets[0].id for s in tree.body
                               if isinstance(s, ast.Assign) and len(s.targets) == 1
                               and isinstance(s.targets[0], ast.Name)
                               and bindings[s.targets[0].id] == 1 and "*" not in bindings}
        self.global_values = {}
        # Shape authentication can inspect later declarations; actual calls still
        # require the binding installed by the module statement walk.
        self.helper_definitions = {node.name: node for node in tree.body
                                   if isinstance(node, ast.FunctionDef) and bindings[node.name] == 1}
        self.constants = {}  # Populated only by the actual module statement walk.
        self.calls = []
        self.serial = 0
        self.solver = z3.Solver()
        self.normal_conditions = []
        self.steps = 0
        self.work_limit = self.WORK_BUDGET
        self.class_exit = None
        self.class_guards = []
        self.range_depth = 0
        self.range_targets = {}
        self.primitive_exits = primitive_exits
        self.primitive_helpers = set()
        self.primitive_check_seen = False
        self.closed_callables = closed_callables
        self.closed_callable_seen = False
        if closed_callables:
            if self.classes != {"AssertionError"} or primitive_exits:
                raise Unchecked("closed callable checking is a separate AssertionError-only route")
            self.closed_callable_boundary()
        if primitive_exits:
            if self.classes != {"AssertionError"}:
                raise Unchecked("primitive checker exits are AssertionError-only")
            self.primitive_exit_boundary()

    def closed_callable_boundary(self):
        # shortcut: closed module aliases, primitive zero-argument leaves and hook-free callback forwarders only.
        allowed = (ast.Module, ast.Import, ast.ImportFrom, ast.alias, ast.FunctionDef, ast.arguments, ast.arg,
                   ast.Assign, ast.Expr, ast.If, ast.Assert, ast.Return, ast.Pass, ast.Constant, ast.Name,
                   ast.Load, ast.Store, ast.Call, ast.Attribute, ast.keyword, ast.UnaryOp, ast.unaryop,
                   ast.BinOp, ast.operator, ast.BoolOp, ast.boolop, ast.Compare, ast.cmpop, ast.IfExp)
        nodes = tuple(self._tuple_nodes(self.tree))
        if (any(not isinstance(n, allowed)
                or isinstance(n, ast.Constant) and type(n.value) not in {int, bool, str}
                or isinstance(n, (ast.Import, ast.ImportFrom)) and n not in self.tree.body for n in nodes)):
            raise Unchecked("closed callables require bounded loop-free source without hooks or handlers")
        imports = {a.asname or a.name.split(".")[0] for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom))
                   for a in n.names}
        from tharkun.solve.contracts.rules import get_rule
        from tharkun.solve.contracts.visitors import ProductVisitor
        rule = get_rule("_sv_verifier.check_type")
        if rule is None or rule.accept(ProductVisitor()) is not True:
            raise Unchecked("closed callable checking requires the tested checker contract")
        names, executing = set(), False
        for stmt in self.tree.body:
            declaration = isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            if declaration and executing:
                raise Unchecked("closed callable declarations must precede execution")
            if not declaration and not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)):
                executing = True
            for n in ((stmt,) if isinstance(stmt, ast.FunctionDef) else ast.walk(stmt)):
                bound = (n.name if isinstance(n, ast.FunctionDef) else n.id
                         if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) else
                         n.asname or n.name.split(".")[0] if isinstance(n, ast.alias) else None)
                if bound is not None:
                    if bound in names or bound in BUILTINS | {"__name__"}:
                        raise Unchecked("closed callable module bindings must be unique and authenticated")
                    names.add(bound)
        for function in (n for n in nodes if isinstance(n, ast.FunctionDef)):
            args = function.args
            if (function not in self.tree.body or function.decorator_list or function.type_params
                    or function.returns is not None or args.posonlyargs or args.kwonlyargs or args.vararg
                    or args.kwarg or args.defaults or args.kw_defaults
                    or len({a.arg for a in args.args}) != len(args.args)
                    or any(a.annotation is not None or a.arg in BUILTINS | imports | self.helper_definitions.keys()
                           for a in args.args)):
                raise Unchecked("closed callable requires a plain authenticated leaf or callback parameter")
            body = function.body
            if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and type(body[0].value.value) is str:
                body = body[1:]
            if len(body) != 1 or not isinstance(body[0], ast.Return):
                raise Unchecked("closed callable helper requires one valued return")
            value = body[0].value
            if args.args:
                if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                        and value.func.id in {a.arg for a in args.args} and not value.args and not value.keywords):
                    raise Unchecked("closed callback forwarder must return its parameter's zero-argument call")
            elif not (isinstance(value, ast.Constant) and type(value.value) in {int, bool, str}
                      or isinstance(value, ast.Call) and not value.args and not value.keywords
                      and (isinstance(value.func, ast.Name) and value.func.id in imports
                           or isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name)
                           and value.func.value.id in imports and value.func.attr in {"nondet_int", "nondet_bool"})):
                raise Unchecked("closed callable leaf must return a primitive literal or authenticated scalar draw")

    def closed_callable_cases(self, node, env, pc):
        self.steps += 1
        if self.steps > self.work_limit or time.monotonic() >= self.deadline:
            raise Unchecked("closed callable work budget exhausted")
        if isinstance(node, ast.IfExp):
            test = self.condition(node.test, env, pc)
            for branch, path in ((node.body, z3.And(pc, test)), (node.orelse, z3.And(pc, z3.Not(test)))):
                if not self.unsat(path):
                    for value, guarded in self.closed_callable_cases(branch, env, path):
                        if value[0] != "object":
                            raise Unchecked("conditional callable assignment requires closed functions")
                        yield value, guarded
        else:
            value = self.expr(node, env, pc)
            if value[0] not in {"int", "bool", "str", "object"}:
                raise Unchecked("closed callable assignment has an unsupported value")
            yield value, pc

    def closed_callable_value(self, call, env, pc):
        self.steps += 1
        if self.steps > self.work_limit or time.monotonic() >= self.deadline:
            raise Unchecked("closed callable work budget exhausted")
        kind, function = self.expr(call.func, env, pc)
        if (kind != "object" or not isinstance(function, ast.FunctionDef)
                or self.functions.get(function.name) is not function
                or function.name in self.helper_calls):
            raise Unchecked("call requires a currently bound finite closed source function")
        params = [a.arg for a in function.args.args]
        if len(call.args) > len(params) or any(isinstance(a, ast.Starred) for a in call.args):
            raise Unchecked("closed callable positional arity")
        local = {name: self.expr(arg, env, pc) for name, arg in zip(params, call.args)}
        for keyword in call.keywords:
            if keyword.arg not in params or keyword.arg in local:
                raise Unchecked("closed callable duplicate, expanded or unknown keyword")
            local[keyword.arg] = self.expr(keyword.value, env, pc)
        if set(local) != set(params) or any(value[0] != "object" for value in local.values()):
            raise Unchecked("closed forwarder requires its actual closed callback argument")
        self.helper_calls.append(function.name)
        self.closed_callable_seen = True
        try:
            for stmt in function.body:
                self.at(stmt, local, pc)
            value = self.expr(function.body[-1].value, local, pc)
            if value[0] not in {"int", "bool", "str"}:
                raise Unchecked("closed callable result must be an exact primitive value")
            return value
        finally:
            self.helper_calls.pop()

    def primitive_exit_boundary(self):
        # shortcut: loop-free module callers and closed primitive helper returns; no caught exits or hooks.
        allowed = (ast.Module, ast.Import, ast.ImportFrom, ast.alias, ast.FunctionDef, ast.arguments, ast.arg,
                   ast.Assign, ast.Expr, ast.If, ast.Assert, ast.Return, ast.Pass, ast.Constant, ast.Name,
                   ast.Load, ast.Store, ast.Call, ast.Attribute, ast.keyword, ast.UnaryOp, ast.unaryop,
                   ast.BinOp, ast.operator, ast.BoolOp, ast.boolop, ast.Compare, ast.cmpop, ast.IfExp)
        if any(not isinstance(n, allowed) or isinstance(n, ast.Constant) and type(n.value) not in {int, bool, str}
               or isinstance(n, ast.FunctionDef) and n not in self.tree.body
               for n in self._tuple_nodes(self.tree)):
            raise Unchecked("primitive exits require closed loop-free source without exception handlers or hooks")
        executing = False
        for stmt in self.tree.body:
            declaration = isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            if declaration and executing:
                raise Unchecked("primitive exit declarations must precede execution")
            if not declaration and not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)):
                executing = True

    def primitive_helper_shape(self, function):
        args = function.args
        self._tuple_work(len(args.args))
        params = {arg.arg for arg in args.args}
        if (function not in self.tree.body or function.decorator_list or function.type_params
                or function.returns is not None or len(params) != len(args.args)
                or any(arg.annotation is not None for arg in args.args)
                or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults or args.kw_defaults
                or self.bindings.get(function.name) != 1
                or params & (BUILTINS | self.imports.keys() | self.helper_definitions.keys())):
            raise Unchecked("primitive return helper requires plain authenticated scalar parameters")
        body = function.body
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and type(body[0].value.value) is str:
            body = body[1:]
        if (len(body) != 1 or not isinstance(body[0], ast.If) or len(body[0].body) != 1 or len(body[0].orelse) != 1
                or any(not isinstance(s, ast.Return) or s.value is None for s in body[0].body + body[0].orelse)):
            raise Unchecked("primitive helper requires one if and two valued returns")
        branch = body[0]
        literals = {s.value for s in branch.body + branch.orelse if isinstance(s.value, ast.Constant)}
        allowed = (ast.If, ast.Return, ast.Constant, ast.Name, ast.Load, ast.UnaryOp, ast.unaryop,
                   ast.BinOp, ast.operator, ast.BoolOp, ast.boolop, ast.Compare, ast.cmpop, ast.IfExp)
        for node in self._tuple_nodes(branch):
            if (not isinstance(node, allowed)
                    or isinstance(node, ast.Name) and node.id not in params
                    or isinstance(node, ast.Constant) and type(node.value) not in {int, bool}
                    and not (node in literals and type(node.value) is str)):
                raise Unchecked("primitive helper may read only its parameters and return exact literals/scalars")
        return branch

    def primitive_cases(self, call, env, pc):
        function = self.functions[call.func.id]
        branch = self.primitive_helper_shape(function)
        local = self.bind_arguments(call, function, env, pc)
        for stmt in function.body:
            self.at(stmt, local, pc)
        test = self.condition(branch.test, local, pc)
        for returned, path in ((branch.body[0], z3.And(pc, test)), (branch.orelse[0], z3.And(pc, z3.Not(test)))):
            if self.unsat(path):
                continue
            self.at(returned, local, path)
            value = self.expr(returned.value, local, path)
            if value[0] not in {"int", "bool", "str"}:
                raise Unchecked("primitive return is not an exact int/bool/literal string")
            yield value, path

    @contextmanager
    def modular_path(self):
        # Hypothetical helper/method executions cannot constrain the module path.
        previous, self.normal_conditions = self.normal_conditions, []
        try:
            yield
        finally:
            self.normal_conditions = previous

    def unsat(self, pc):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise Unchecked("checker budget exhausted")
        self.solver.set(timeout=max(1, int(left * 1000)))
        self.solver.push()
        self.solver.add(pc, *self.normal_conditions)
        try:
            return self.solver.check() == z3.unsat
        finally:
            self.solver.pop()

    def prove(self, pc, goal):
        if not self.unsat(z3.And(pc, z3.Not(goal))):
            raise Unchecked("verification condition not proved")

    def fresh(self, kind):
        self.serial += 1
        sorts = {"int": z3.IntSort(), "float": z3.Float64(), "bool": z3.BoolSort()}
        return kind, z3.Const(f"draw{self.serial}", sorts[kind])

    def source_truth(self, value):
        """Truth of independently evaluated primitive source values; no coercion or hooks."""
        kind, term = value
        if kind == "bool":
            return term
        if kind == "int":
            return term != 0
        raise Unchecked("source truth requires an exact bool or int")

    def condition(self, node, env, pc):
        """Source If truth only; and/or operand values keep their existing boundary."""
        if node in self.none_guards:
            if self.expr(node.left, env, pc)[0] not in {"int", "bool"}:
                raise Unchecked("None identity guard requires an exact int or bool")
            return z3.BoolVal(isinstance(node.ops[0], ast.IsNot))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return z3.Not(self.condition(node.operand, env, pc))
        if not isinstance(node, ast.BoolOp):
            return self.source_truth(self.expr(node, env, pc))
        terms = []
        for part in node.values:
            left = self.deadline - time.monotonic()
            if left <= 0:
                raise Unchecked("checker budget exhausted")
            self.solver.set(timeout=max(1, int(left * 1000)))
            self.solver.push()
            self.solver.add(pc, *self.normal_conditions)
            try:
                reachable = self.solver.check()
            finally:
                self.solver.pop()
            if reachable == z3.unsat:
                break  # Python never evaluates this operand: no helper, exception or draw.
            if reachable != z3.sat:
                raise Unchecked("short-circuit reachability unknown")
            term = self.condition(part, env, pc)
            terms.append(term)
            pc = z3.And(pc, term if isinstance(node.op, ast.And) else z3.Not(term))
        return (z3.And if isinstance(node.op, ast.And) else z3.Or)(*terms)

    def targeted(self, qualname):
        """Whether any supplied candidate/invariant names this function or
        method as its ``Invariant.function`` -- the SAME test the existing
        zero-argument-function registration uses implicitly (a definition
        with no relevance to any candidate is never walked): [internal ref] slice 2b
        applies it to parameterized functions and class methods too, so a
        fact is proved MODULARLY, once, for every caller (owner design,
        [internal ref]) instead of needing a call site to reach it."""
        return any(function == qualname for (_, _, _, function) in self.invariants)

    def free_params(self, args):
        """Turn a plain, fully-annotated parameter list into fresh, UNCONSTRAINED
        Z3 symbols ([internal ref] slice 2b, owner design): nothing about a specific
        call site is assumed, so this never re-derives a fact per call site.

        MG review (2026-09-29): this is NOT an unconditional "holds for
        every caller" claim -- it holds only for a caller/construction site
        that actually supplies a value of the ANNOTATED sort (Python does
        not enforce an annotation at runtime; nothing stops a real caller
        from passing a float where ``int`` is declared, or a real ``obj.x =
        1.5`` bypassing ``__init__`` entirely for a class field). Today that
        gap is VACUOUS for the original modular bodies and class methods:
        actual calls to them remain Unchecked. Scalar leaf helper calls are
        the narrow exception: every actual argument sort is checked before
        binding the helper's parameter-only environment. A field's type is
        still not checked against how an object was actually built, so a
        FUTURE consumer that trusts "this fact holds for every caller"
        literally, without independently checking that every real call site
        respects the annotated sort, would be unsound. Constraining a
        parameter by CERTIFIED facts at the function's own entry (the
        owner's stated alternative to leaving it unconstrained) is NOT
        built here either -- no caller of this checker yet supplies entry
        facts; unconstrained is the sound default for THIS checker's own
        proof obligation, conditional on the annotated sort as above.
        Every parameter must carry a plain ``int``/``float``/``bool``
        annotation naming its sort -- never guessed from a default or a call
        site, matching this module's fail-closed rule everywhere else."""
        if (args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
                or any(args.defaults) or any(args.kw_defaults)):
            raise Unchecked("only plain positional parameters, no *args/**kwargs/defaults, are supported")
        env = {}
        for a in args.args:
            if a.arg in BUILTINS or a.arg in self.imports or a.arg in self.functions or a.arg in env:
                raise Unchecked("parameter rebinds a builtin/callable/parameter name")
            if not isinstance(a.annotation, ast.Name) or a.annotation.id not in {"int", "float", "bool"}:
                raise Unchecked(f"parameter '{a.arg}' needs an int/float/bool annotation to become a free symbol")
            env[a.arg] = self.fresh(a.annotation.id)
        return env

    def parameterized(self, s):
        """Check parameterized bodies modularly at targeted definitions.

        Scalar leaves and tuple wrappers admit actual calls with checked argument sorts;
        every other parameterized call remains refused.
        """
        self._tuple_work(len(s.args.args) + len(s.body))
        if s.name in self.imports or s.name in self.functions or s.name in BUILTINS:
            raise Unchecked("rebound function name")
        self.functions[s.name] = s
        if self.primitive_exits:
            self.primitive_helper_shape(s)
            self.primitive_helpers.add(s.name)
            return
        if any(isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple) for n in ast.walk(s)):
            self.tuple_shape(s)
            self.tuple_helpers.add(s.name)
            return  # Only actual unpacking calls establish the tuple helper's prestates.
        scalar = isinstance(s.returns, ast.Name) and s.returns.id in {"int", "bool"}
        if scalar:
            try:
                self.scalar_leaf(s)
            except Unchecked:
                self.bounded_shape(s)
                self.bounded_helpers.add(s.name)
            if any(a.annotation is None or isinstance(a.annotation, ast.Name)
                   and a.annotation.id == "object" for a in s.args.args):
                return  # Only actual callers can establish these parameter sorts.
            if self.targeted(s.name):
                with self.modular_path():
                    params = self.free_params(s.args)
                    self.helper_value(s, params, z3.BoolVal(True))
            return
        if (any(a.annotation is None for a in s.args.args)
                or s.returns is None and any(isinstance(n, ast.Return) and n.value is not None
                                            for n in ast.walk(s))):
            self.bounded_shape(s)
            self.bounded_helpers.add(s.name)
            return
        if not self.targeted(s.name):
            return
        with self.modular_path():
            params = self.free_params(s.args)
            self.block(s.body, params, z3.BoolVal(True))

    def check_type_hint(self, call):
        """Authenticate a statement-only primitive check; never infer its operand's type."""
        func = call.func
        name = (func.id if isinstance(func, ast.Name) and self.imports.get(func.id) == self.CHECK_TYPE
                else func.value.id if isinstance(func, ast.Attribute) and func.attr == "check_type"
                and isinstance(func.value, ast.Name) and self.imports.get(func.value.id) == self.MODULE
                else None)
        if name is None:
            return None
        if (self.bindings.get(name) != 1 or call.keywords or len(call.args) != 2
                or isinstance(call.args[0], ast.Starred) or not isinstance(call.args[1], ast.Name)
                or call.args[1].id not in {"int", "bool", "float", "str"}
                or call.args[1].id in self.bindings):
            raise Unchecked("check_type requires an authenticated primitive hint and two positional operands")
        return call.args[1].id

    def literal_defaults(self, args):
        for default in args.defaults:
            if (isinstance(default, ast.Constant) and type(default.value) in {int, bool}
                    or isinstance(default, ast.UnaryOp) and isinstance(default.op, (ast.UAdd, ast.USub))
                    and isinstance(default.operand, ast.Constant) and type(default.operand.value) is int):
                continue
            raise Unchecked("helper defaults require immutable int/bool literals")
        return args.defaults

    def bind_arguments(self, call, function, env, pc):
        self._tuple_work(len(function.args.args) + len(call.args) + len(call.keywords))
        params = function.args.args
        names = [param.arg for param in params]
        if len(call.args) > len(params) or any(isinstance(arg, ast.Starred) for arg in call.args):
            raise Unchecked("scalar helper call arity")
        values = {name: self.expr(arg, env, pc) for name, arg in zip(names, call.args)}
        for keyword in call.keywords:
            if keyword.arg not in names or keyword.arg in values:
                raise Unchecked("scalar helper duplicate/unknown/expanded keyword")
            values[keyword.arg] = self.expr(keyword.value, env, pc)
        defaults = self.literal_defaults(function.args)
        for name, default in zip(names[len(names) - len(defaults):], defaults):
            if name not in values:
                # Immutable literals equal their definition-time values.
                values[name] = self.expr(default, env, pc)
        if set(values) != set(names):
            raise Unchecked("scalar helper missing argument")
        if any(values[param.arg][0] not in {"int", "bool"} or param.annotation is not None
               and values[param.arg][0] != param.annotation.id
               and not (isinstance(function, ast.FunctionDef) and function.name in self.bounded_helpers
                        and param.annotation.id == "object")
               for param in params):
            raise Unchecked("scalar helper argument sort does not match its annotation")
        return values

    def lambda_value(self, call, env, pc):
        # shortcut: immediate int/bool bodies without calls; extend for a checked witness need.
        function = call.func
        args = function.args
        self._tuple_work(len(args.args))
        params = {param.arg for param in args.args}
        if (args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or any(args.kw_defaults)
                or len(params) != len(args.args)
                or params & (BUILTINS | self.imports.keys() | self.functions.keys())):
            raise Unchecked("lambda requires plain distinct scalar parameters")
        self.literal_defaults(args)
        allowed = (ast.Constant, ast.Name, ast.Load, ast.UnaryOp, ast.unaryop, ast.BinOp,
                   ast.operator, ast.BoolOp, ast.boolop, ast.Compare, ast.cmpop, ast.IfExp)
        if any(not isinstance(node, allowed)
               or isinstance(node, ast.Constant) and type(node.value) not in {int, bool}
               for node in self._tuple_nodes(function.body)):
            raise Unchecked("lambda requires a primitive expression without calls or effects")
        local = {name: value for name, value in env.items() if value[0] in {"int", "bool"}}
        local.update(self.bind_arguments(call, function, env, pc))
        value = self.expr(function.body, local, pc)
        if value[0] not in {"int", "bool"}:
            raise Unchecked("lambda must return an exact int or bool")
        return value

    def factory_shape(self, function, active=()):
        # shortcut: acyclic zero-argument return expressions and readonly scalar globals.
        for _ in self._tuple_nodes(function):
            pass
        args = function.args
        if (function not in self.tree.body or function.decorator_list or function.type_params
                or function.returns is not None or args.args or args.posonlyargs or args.kwonlyargs
                or args.vararg or args.kwarg or args.defaults or args.kw_defaults
                or self.bindings.get(function.name) != 1 or function.name in active):
            raise Unchecked("factory requires a finite authenticated zero-argument helper")
        body = function.body
        if (isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and type(body[0].value.value) is str):
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            raise Unchecked("factory requires a single valued return")
        allowed = (ast.Constant, ast.Name, ast.Load, ast.Call, ast.UnaryOp, ast.unaryop,
                   ast.BinOp, ast.operator, ast.BoolOp, ast.boolop, ast.Compare, ast.cmpop)
        nodes = tuple(ast.walk(body[0].value))
        callees = {node.func for node in nodes if isinstance(node, ast.Call)}
        for node in nodes:
            if (not isinstance(node, allowed)
                    or isinstance(node, ast.Constant) and type(node.value) not in {int, bool}):
                raise Unchecked("factory return requires primitive expressions or closed callables")
            if isinstance(node, ast.Call):
                if (not isinstance(node.func, ast.Name) or node.args or node.keywords
                        or node.func.id not in self.helper_definitions):
                    raise Unchecked("factory call requires an authenticated zero-argument callee")
                self.factory_shape(self.helper_definitions[node.func.id], active + (function.name,))
            if isinstance(node, ast.Name) and node not in callees:
                if node.id in self.helper_definitions:
                    self.factory_shape(self.helper_definitions[node.id], active + (function.name,))
                elif node.id not in self.readonly_names:
                    raise Unchecked("factory reads an untrusted or mutable global")
        return body[0]

    def factory_value(self, function, pc):
        self._tuple_work(0)
        if self.functions.get(function.name) is not function or function.name in self.helper_calls:
            raise Unchecked("factory callable is unbound, rebound, recursive or too deep")
        returned = self.factory_shape(function)
        self.helper_calls.append(function.name)
        try:
            self.at(returned, self.global_values, pc)
            value = self.expr(returned.value, self.global_values, pc)
            if value[0] not in {"int", "bool", "callable"}:
                raise Unchecked("factory returned an unsupported value")
            return value
        finally:
            self.helper_calls.pop()

    def bounded_shape(self, s, active=()):
        """Closed, finite pure helpers; inspect even statements after a return."""
        self._tuple_work(len(s.args.args) + len(s.body))
        args = s.args
        if (s not in self.tree.body or s.decorator_list or s.type_params
                or args.posonlyargs or args.kwonlyargs
                or args.vararg or args.kwarg or any(args.kw_defaults)
                or s.returns is not None and not (isinstance(s.returns, ast.Name)
                                                  and s.returns.id in {"int", "bool"})
                or any(a.annotation is not None and not (isinstance(a.annotation, ast.Name)
                        and (a.annotation.id in {"int", "bool"}
                             or a.annotation.id == "object" and s.returns is not None)) for a in args.args)
                or self.bindings.get(s.name) != 1 or s.name in active):
            raise Unchecked("bounded helper signature/binding/call-depth")
        self.literal_defaults(args)
        nodes = [n for stmt in s.body for n in self._tuple_nodes(stmt)]
        params = {a.arg for a in args.args}
        locals_ = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        forbidden = (BUILTINS | {"min", "max", "abs", "range", "__builtins__"}
                     | self.imports.keys() | {n.name for n in self.tree.body if isinstance(n, ast.FunctionDef)})
        if (len(params) != len(args.args) or params & locals_
                or (params | locals_) & forbidden):
            raise Unchecked("bounded helper locals/shadowing/statement limit")
        checks = set()
        pending = [(stmt, 0) for stmt in s.body]
        while pending:
            stmt, depth = pending.pop()
            if isinstance(stmt, ast.If):
                pending.extend((child, depth + 1) for child in stmt.body + stmt.orelse)
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    raise Unchecked("bounded helper scalar assignment required")
            elif isinstance(stmt, ast.Return) and stmt.value is not None:
                pass
            elif (stmt is s.body[0] and isinstance(stmt, ast.Expr)
                  and isinstance(stmt.value, ast.Constant) and type(stmt.value.value) is str):
                continue
            elif (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                  and self.check_type_hint(stmt.value) is not None):
                checks.update(ast.walk(stmt.value.func))
                checks.add(stmt.value.args[1])
                checks.add(stmt.value)
            else:
                raise Unchecked("bounded helper unmodeled control/effect")
        allowed = (ast.Assign, ast.If, ast.Return, ast.Expr, ast.Constant, ast.Name, ast.Load,
                   ast.Store, ast.Call, ast.keyword, ast.UnaryOp, ast.unaryop, ast.BinOp, ast.operator,
                   ast.BoolOp, ast.boolop, ast.Compare, ast.cmpop, ast.IfExp)
        object_marked = any(isinstance(a.annotation, ast.Name) and a.annotation.id == "object"
                            for a in args.args)
        callees = {n.func for n in nodes if isinstance(n, ast.Call)}
        for n in nodes:
            if n in checks:
                continue
            if (not isinstance(n, allowed)
                    or isinstance(n, ast.Constant) and type(n.value) not in {int, bool}
                    and not (isinstance(s.body[0], ast.Expr) and n is s.body[0].value)
                    or isinstance(n, ast.Name) and n not in callees
                    and n.id not in params | locals_ | self.constants.keys()):
                raise Unchecked("bounded helper primitive expression required")
            if isinstance(n, ast.Call):
                if (not isinstance(n.func, ast.Name) or any(k.arg is None for k in n.keywords)
                        or n.func.id in params | locals_
                        or any(isinstance(a, ast.Starred) for a in n.args)):
                    raise Unchecked("bounded helper requires an authenticated pure callee")
                name = n.func.id
                if name in self.helper_definitions:
                    if len(n.args) > len(self.helper_definitions[name].args.args):
                        raise Unchecked("bounded helper pure callee arity")
                    self.bounded_shape(self.helper_definitions[name], active + (s.name,))
                elif object_marked and name in {"int", "max"} and name not in self.bindings:
                    from tharkun.solve.contracts.rules import get_rule
                    from tharkun.solve.contracts.visitors import ProductVisitor
                    rule = get_rule(name)
                    arity = rule.accept(ProductVisitor()) if rule is not None else None
                    if (type(arity) is not int or arity != (1 if name == "int" else 2)
                            or len(n.args) != arity):
                        raise Unchecked("bounded object helper requires a tested int/max call shape")
                else:
                    raise Unchecked("bounded helper requires an authenticated pure callee")
        return locals_

    def _tuple_work(self, count=1):
        if count > self.work_limit - self.steps or time.monotonic() >= self.deadline:
            raise Unchecked("tuple checking work/deadline budget exhausted")
        self.steps += count

    def _tuple_nodes(self, node):
        # Streaming traversal checks work before allocating an AST-node collection.
        self._tuple_work()
        yield node
        for child in ast.iter_child_nodes(node):
            yield from self._tuple_nodes(child)

    def tuple_shape(self, s):
        """A known finite straight-line bool tuple, immediately unpacked."""
        args = s.args
        self._tuple_work(len(args.args) + len(s.body))
        for _ in self._tuple_nodes(s):
            pass
        if (s not in self.tree.body or s.decorator_list or s.type_params or s.returns is not None
                or args.posonlyargs or args.kwonlyargs
                or args.vararg or args.kwarg or args.defaults or any(args.kw_defaults)
                or any(not isinstance(a.annotation, ast.Name) or a.annotation.id != "int"
                       for a in args.args) or self.bindings.get(s.name) != 1):
            raise Unchecked("tuple helper requires plain int parameters")
        body = s.body
        if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and type(body[0].value.value) is str):
            body = body[1:]
        if (not body or not isinstance(body[-1], ast.Return)
                or not isinstance(body[-1].value, ast.Tuple)
                or any(not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1
                       or not isinstance(stmt.targets[0], ast.Name) for stmt in body[:-1])):
            raise Unchecked("tuple helper requires scalar assignments and a final tuple")
        params = {a.arg for a in args.args}
        locals_ = {stmt.targets[0].id for stmt in body[:-1]}
        scope = params | locals_
        self._tuple_work(len(self.tree.body))
        forbidden = (BUILTINS | {"min", "max", "abs", "range", "__builtins__"}
                     | self.imports.keys() | {n.name for n in self.tree.body if isinstance(n, ast.FunctionDef)})
        if len(params) != len(args.args) or params & locals_ or scope & forbidden:
            raise Unchecked("tuple helper shadows a parameter/builtin/import/callee")
        self._tuple_work(len(scope) + len(self.constants))
        bound_names = scope | self.constants.keys()
        bool_names = {name for name, value in self.constants.items()
                      if value[0] == "bool" and name not in scope}

        def boolean(expr):
            return (isinstance(expr, ast.Compare)
                    or isinstance(expr, ast.Constant) and type(expr.value) is bool
                    or isinstance(expr, ast.Name) and expr.id in bool_names
                    or isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not)
                    or isinstance(expr, ast.BoolOp) and all(boolean(value) for value in expr.values))

        for stmt in body[:-1]:
            name = stmt.targets[0].id
            is_bool = boolean(stmt.value)
            bool_names.discard(name)
            if is_bool:
                bool_names.add(name)
        if not all(boolean(expr) for expr in body[-1].value.elts):
            raise Unchecked("tuple helper requires boolean result expressions")
        allowed = (ast.Constant, ast.Name, ast.Load, ast.Call, ast.keyword, ast.UnaryOp, ast.unaryop,
                   ast.BinOp, ast.operator, ast.BoolOp, ast.boolop, ast.Compare, ast.cmpop)
        for expr in [stmt.value for stmt in body[:-1]] + body[-1].value.elts:
            self._tuple_work(0)
            nodes = tuple(ast.walk(expr))
            callees = {n.func for n in nodes if isinstance(n, ast.Call)}
            for n in nodes:
                if (not isinstance(n, allowed)
                        or isinstance(n, ast.Constant) and type(n.value) not in {int, bool}
                        or isinstance(n, ast.Name) and n not in callees
                        and n.id not in bound_names):
                    raise Unchecked("tuple helper requires primitive expressions")
                if isinstance(n, ast.Call):
                    if (not isinstance(n.func, ast.Name) or any(k.arg is None for k in n.keywords)
                            or n.func.id not in self.helper_definitions or n.func.id in scope
                            or len(n.args) > len(self.helper_definitions[n.func.id].args.args)):
                        raise Unchecked("tuple helper requires an authenticated pure scalar callee")
                    for _ in self._tuple_nodes(self.helper_definitions[n.func.id]):
                        pass
                    self.bounded_shape(self.helper_definitions[n.func.id], (s.name,))
        self._tuple_work(0)
        return locals_

    def tuple_cases(self, node, env, pc):
        self._tuple_work(len(node.elts))
        values = []
        for element in node.elts:
            self._tuple_work(0)
            value = self.expr(element, env, pc)
            if value[0] not in {"int", "float", "bool"}:
                raise Unchecked("nested/mutable container")
            values.append(value)
        self._tuple_work(0)
        return "tuple", tuple(values)

    def helper_value(self, s, params, pc):
        self._tuple_work(0)
        if s.name in self.helper_calls:
            raise Unchecked("recursive/deep scalar helper call")
        self.helper_calls.append(s.name)
        try:
            if s.name in self.tuple_helpers:
                locals_ = self.tuple_shape(s)
                self._tuple_work(len(self.constants) + len(params))
                env = {name: value for name, value in self.constants.items() if name not in locals_}
                env.update(params)
                state, path = self.block(s.body[:-1], env, pc, returns=[])[0]
                self.at(s.body[-1], state, path)
                kind, values = self.expr(s.body[-1].value, state, path)
                if kind != "tuple" or any(value[0] != "bool" for value in values):
                    raise Unchecked("tuple helper requires bool results")
                self._tuple_work(0)
                return kind, values
            if s.name not in self.bounded_helpers:
                return self.leaf_value(s, params, pc)
            locals_ = self.bounded_shape(s)
            env = {name: value for name, value in self.constants.items() if name not in locals_}
            env.update(params)
            branches = []
            for _, path in self.block(s.body, env, pc, returns=branches):
                self.prove(path, z3.BoolVal(False))  # A feasible fallthrough would return None.
            if not branches or branches[0][1][0] not in {"int", "bool"}:
                raise Unchecked("bounded helper has no primitive return")
            kind = branches[0][1][0]
            if (any(value[0] != kind for _, value in branches)
                    or s.returns is not None and s.returns.id != kind):
                raise Unchecked("bounded helper return sorts disagree")
            result = branches[-1][1][1]
            for path, value in reversed(branches[:-1]):
                result = z3.If(path, value[1], result)
            return kind, result
        finally:
            self.helper_calls.pop()

    def scalar_leaf(self, s):
        """Value helpers use primitive parameters and authenticated module literals."""
        if s not in self.tree.body:
            raise Unchecked("scalar helpers must be unconditional module-level definitions")
        args = s.args
        self._tuple_work(len(args.args))
        params = {a.arg for a in args.args}
        if (args.posonlyargs or args.kwonlyargs or args.vararg
                or args.kwarg or args.defaults or any(d is not None for d in args.kw_defaults)
                or len(params) != len(args.args)
                or params & (BUILTINS | self.imports.keys() | self.helper_definitions.keys())
                or any(not isinstance(a.annotation, ast.Name)
                       or a.annotation.id not in {"int", "bool"} for a in args.args)):
            raise Unchecked("scalar helpers require plain annotated int/bool parameters")
        body = s.body
        if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and type(body[0].value.value) is str):
            body = body[1:]
        if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
            expressions = [body[0].value]
        elif (len(body) in {2, 3}
              and isinstance(s.returns, ast.Name) and s.returns.id == "bool"
              and isinstance(body[-2], ast.Assert) and body[-2].msg is None
              and isinstance(body[-1], ast.Return) and isinstance(body[-1].value, ast.Call)
              and not body[-1].value.args and not body[-1].value.keywords
              and (isinstance(body[-1].value.func, ast.Name)
                   and self.imports.get(body[-1].value.func.id) == "bool"
                   or isinstance(body[-1].value.func, ast.Attribute)
                   and isinstance(body[-1].value.func.value, ast.Name)
                   and self.imports.get(body[-1].value.func.value.id) == self.MODULE
                   and body[-1].value.func.attr == "nondet_bool")
              and (len(body) == 2 or isinstance(body[0], ast.If) and not body[0].orelse
                   and len(body[0].body) == 1 and isinstance(body[0].body[0], ast.Return)
                   and isinstance(body[0].body[0].value, ast.Constant)
                   and body[0].body[0].value.value is False)):
            # The assertion and optional early return precede the authenticated bool draw.
            expressions = [body[-2].test] + ([body[0].test] if len(body) == 3 else [])
        elif (len(body) >= 2
              and all(isinstance(branch, ast.If) and not branch.orelse
                      and len(branch.body) == 1 and isinstance(branch.body[0], ast.Return)
                      and branch.body[0].value is not None for branch in body[:-1])
              and isinstance(body[-1], ast.Return) and body[-1].value is not None
              and all(a.annotation.id == "int" for a in args.args)
              and isinstance(s.returns, ast.Name) and s.returns.id == "int"):
            expressions = [expr for branch in body[:-1]
                           for expr in (branch.test, branch.body[0].value)] + [body[-1].value]
        elif (len(body) == 3
              and all(a.annotation.id == "int" for a in args.args)
              and isinstance(s.returns, ast.Name) and s.returns.id == "int"
              and isinstance(body[0], ast.Assign) and len(body[0].targets) == 1
              and isinstance(body[0].targets[0], ast.Name)
              and isinstance(body[-1], ast.Return) and isinstance(body[-1].value, ast.Name)
              and body[-1].value.id == body[0].targets[0].id):
            # shortcut: one local result and one ternary update; extend only for checked need.
            local = body[0].targets[0].id
            if local in ({a.arg for a in args.args} | self.constants.keys() | BUILTINS
                         | self.imports.keys() | self.functions.keys()):
                raise Unchecked("scalar result local rebinds a parameter/constant/callable")
            update = body[1]
            expressions = [body[0].value]
            if isinstance(update, ast.If) and not update.orelse and len(update.body) == 1:
                expressions.append(update.test)
                update = update.body[0]
            if not (isinstance(update, ast.Assign) and len(update.targets) == 1
                    and isinstance(update.targets[0], ast.Name) and update.targets[0].id == local
                    and isinstance(update.value, ast.IfExp)):
                raise Unchecked("scalar result helper requires one ternary assignment")
            expressions.append(update.value)
        else:
            raise Unchecked("scalar helpers require a single valued return or one int conditional and fallback")
        names = params | self.constants.keys()
        if isinstance(body[0], ast.Assign):
            names.add(body[0].targets[0].id)
        allowed = (ast.Constant, ast.Name, ast.Load, ast.UnaryOp, ast.unaryop,
                   ast.BinOp, ast.operator, ast.BoolOp, ast.boolop, ast.Compare, ast.cmpop, ast.IfExp)
        none_literals = {expression.comparators[0] for expression in expressions
                         if expression in self.none_guards}
        for expression in expressions:
            for n in self._tuple_nodes(expression):
                if (not isinstance(n, allowed) or isinstance(n, ast.Name) and n.id not in names
                        or isinstance(n, ast.Constant) and type(n.value) not in {int, bool}
                        and n not in none_literals):
                    raise Unchecked("scalar helper return must use only parameters and primitive expressions")
        return body[0]

    def leaf_value(self, s, env, pc):
        returned = self.scalar_leaf(s)
        env = dict(self.constants, **env)
        if isinstance(returned, ast.Assign):
            branches = []
            for state, path in self.block(s.body[:-1], env, pc):
                self.at(s.body[-1], state, path)
                value = self.expr(s.body[-1].value, state, path)
                if value[0] != "int":
                    raise Unchecked("scalar result helper requires an exact int result")
                branches.append((path, value[1]))
            result = branches[-1][1]
            for path, value in reversed(branches[:-1]):
                result = z3.If(path, value, result)
            return "int", result
        if isinstance(s.body[-1].value, ast.Call):
            early = z3.BoolVal(False)
            for statement in s.body[:-2]:
                self.at(statement, env, pc)
                if isinstance(statement, ast.If):
                    early = self.condition(statement.test, env, pc)
                    self.at(statement.body[0], env, z3.And(pc, early))
            assertion = s.body[-2]
            active = z3.And(pc, z3.Not(early))
            self.at(assertion, env, active)
            kind, test = self.expr(assertion.test, env, active)
            if kind != "bool":
                raise Unchecked("unmodeled assertion")
            if "AssertionError" in self.classes:
                self.prove(active, test)
            continued = z3.And(active, test)
            self.at(s.body[-1], env, continued)
            value = self.expr(s.body[-1].value, env, continued)
            # A failed assertion exits the helper; keeping its caller path is conservative.
            return "bool", z3.If(early, z3.BoolVal(False), value[1])
        if isinstance(returned, ast.If):
            branches = []
            for statement in s.body[:-1]:
                self.at(statement, env, pc)
                if not isinstance(statement, ast.If):  # optional inert docstring
                    continue
                guard = self.condition(statement.test, env, pc)
                branch = statement.body[0]
                then_pc = z3.And(pc, guard)
                self.at(branch, env, then_pc)
                value = self.expr(branch.value, env, then_pc)
                if value[0] != s.returns.id:
                    raise Unchecked("scalar helper return sort does not match its annotation")
                branches.append((guard, value[1]))
                pc = z3.And(pc, z3.Not(guard))
            fallback = s.body[-1]
            self.at(fallback, env, pc)
            value = self.expr(fallback.value, env, pc)
            if value[0] != s.returns.id:
                raise Unchecked("scalar helper return sort does not match its annotation")
            result = value[1]
            for guard, value in reversed(branches):
                result = z3.If(guard, value, result)
            return s.returns.id, result
        for statement in s.body:
            self.at(statement, env, pc)
        value = self.expr(returned.value, env, pc)
        if value[0] != s.returns.id:
            raise Unchecked("scalar helper return sort does not match its annotation")
        return value

    def classdef(self, s):
        """A class definition ([internal ref] slice 2b, item 2; unified onto the
        SHARED pure-data-class detector per [internal ref]'s follow-up):
        NOT a new object model inside the checker. ``lift_class`` is a thin
        pass-through to ``py_method_lift.detect_pure_data_class`` -- the ONE
        accept/refuse decision, shared with Eregion's own forward-passes
        (its class-inlining pass AND its inversion-engine method lift), never
        re-decided here. A targeted method the shared detector REFUSES is
        Unchecked, not silently skipped -- the candidate targeting it stays
        uncertified; per the shared detector's own rule this is a PER-CLASS
        refusal (one unmodeled method anywhere in the class blocks every
        other method too), not per-method as an earlier version of this
        checker had it. A method no candidate targets is left unwalked, same
        rule as a never-called zero-argument function or a never-targeted
        parameterized one.

        MG review (2026-09-29): a class body EXECUTES at definition time --
        ``self.check_body_is_modeled`` below runs for EVERY class, targeted
        or not, BEFORE the targeted-methods check above can return early,
        because a class-body statement (an ``assert``, a division, a nested
        class, a method's own default value or decorator, the class
        statement's own decorator/bases -- all evaluated the moment the
        ``class`` statement runs) can crash the whole program regardless of
        whether any candidate names anything inside it. This is a
        DIFFERENT, complementary check from the shared detector's own
        rule: that one validates a targeted METHOD's own body shape;
        this one validates the CLASS's own top-level statements, targeted
        or not."""
        if s.name in self.imports or s.name in self.functions or s.name in BUILTINS:
            raise Unchecked("rebound class name")
        if s.decorator_list or s.bases or s.keywords or s.type_params:
            raise Unchecked("unmodeled class construct")
        self.check_body_is_modeled(s)
        location = self.points[s][1]
        claims = self.class_invariants.get(location, [])
        targeted_methods = [m.name for m in s.body if isinstance(m, ast.FunctionDef)
                            and (claims or self.targeted(f"{s.name}.{m.name}"))]
        if not targeted_methods:
            return
        info = lift_class(s)
        if info is None:
            raise Unchecked(f"cannot independently model class '{s.name}' "
                            "(py_method_lift.detect_pure_data_class refused the whole class)")
        sorts = _init_field_types(info) if claims else {}
        if claims:
            if set(sorts) != set(info.fields):
                raise Unchecked("class invariant requires independently typed scalar fields")
            allowed_names = {"self__" + field for field in sorts} | BUILTINS
            for claim in claims:
                if any(isinstance(n, ast.Name) and n.id not in allowed_names
                       for n in ast.walk(ast.parse(claim.value, mode="eval"))):
                    raise Unchecked("class invariant can name only receiver fields and predicate builtins")
            # shortcut: no helper calls or true division in class facts; extend with their value and exit models.
            if any(isinstance(n, ast.Div) or isinstance(n, ast.Call) and (
                    not isinstance(n.func, ast.Name) or n.func.id not in {"abs", "min", "max", "int"})
                   for n in ast.walk(s)):
                raise Unchecked("class invariant requires closed scalar operations with modeled exception exits")
        original_classes = self.classes
        original_exit, original_guards = self.class_exit, self.class_guards
        if claims:
            self.classes = self.classes | {"ZeroDivisionError"}
        try:
            for name in targeted_methods:
                lifted = lift_method(info, name)
                if lifted is None:
                    raise Unchecked(f"cannot independently model '{s.name}.{name}' (an un-typed field read)")
                for params, aliases in self.method_cases(info, name, lifted):
                    with self.modular_path():
                        if claims:
                            self.class_exit = lambda state, path: self.class_postcondition(claims, sorts, state, path)
                            self.class_guards = []
                        precondition = z3.BoolVal(True)
                        if claims and name != "__init__":
                            for field, kind in sorts.items():
                                if "self__" + field not in params:
                                    params["self__" + field] = self.fresh(kind)
                            precondition = z3.And(*(self.predicate(c.value, params) for c in claims))
                        body = lifted.tree.body
                        returned = body[-1] if body and isinstance(body[-1], ast.Return) else None
                        states = self.block(body[:-1] if claims and returned else body,
                                            params, precondition, aliases=aliases)
                        if claims:
                            for state, path in states:
                                if returned and returned.value is not None:
                                    self.expr(returned.value, state, path)
                                self.class_postcondition(claims, sorts, state, self.normal_path(path))
        finally:
            self.classes = original_classes
            self.class_exit, self.class_guards = original_exit, original_guards
        if claims:
            self.unvisited.discard(location)

    def class_postcondition(self, claims, sorts, state, path):
        if any(state.get("self__" + field, (None,))[0] != kind for field, kind in sorts.items()):
            self.prove(path, z3.BoolVal(False))
            return
        for claim in claims:
            self.prove(path, self.predicate(claim.value, state))

    def normal_path(self, path):
        if self.class_exit is not None:
            path = z3.And(path, *self.class_guards)
            self.class_guards.clear()
        return path

    def method_cases(self, info, name, lifted):
        """Fresh field environments for every possible receiver identity case."""
        method = info.init if name == "__init__" else info.methods[name]
        aliasable = sorted(parameter for parameter in method.instance_params if any(
            field.startswith(parameter + "__")
            and "self__" + field[len(parameter) + 2:] in lifted.field_writes
            for field in lifted.field_reads))
        # Distinct arguments and receiver aliases are both legal Python calls.
        for mask in range(1 << len(aliasable)):
            params = self.free_params(lifted.tree.args)
            aliases = {}
            for index, parameter in enumerate(aliasable):
                if not mask & (1 << index):
                    continue
                for field in lifted.field_reads:
                    if not field.startswith(parameter + "__"):
                        continue
                    receiver = "self__" + field[len(parameter) + 2:]
                    if receiver not in params:
                        params[receiver] = self.fresh(params[field][0])
                    params[field] = params[receiver]
                    aliases.setdefault(receiver, []).append(field)
            yield params, aliases

    def check_body_is_modeled(self, s):
        """MG review, blocker fix (2026-09-29): a class body is ordinary
        Python code that runs once, immediately, when the ``class``
        statement executes -- NOT a passive declaration. Before this fix,
        ``classdef`` only ever inspected TARGETED methods, silently treating
        every other class-body statement (and an untargeted method's own
        signature) as a no-op; ``class C: assert 1 == 2`` or
        ``class C: z = 1 // 0`` genuinely crash the program at that point,
        but were reported ``confirmed=True`` for an unrelated candidate
        later in the source, since nothing ever walked them.

        Every statement must be one of: an undecorated method with no
        default values (posonly/positional/keyword-only -- a default is an
        expression evaluated at class-definition time, same as a decorator
        call) and a trivial (bare ``Name``/``Constant``, or absent)
        parameter/return annotation (anything else is an expression
        evaluated at definition time too, and this checker's own
        parameter-typing rule already limits an ANNOTATION it actually uses
        to a bare sort name, so a richer one could never be modeled safely
        even for a targeted method); a docstring; ``pass``; or a value-less
        annotation (``x: int``, never evaluated at runtime, unlike
        ``x: int = ...``). A nested ``ClassDef`` is refused outright (its
        OWN body executes as part of this one) -- never recursed into,
        matching this method's own zero-recursion budget. Anything else
        raises :class:`Unchecked` -- never silently skipped."""
        for item in s.body:
            if isinstance(item, ast.FunctionDef):
                if (item.decorator_list or item.args.defaults
                        or any(d is not None for d in item.args.kw_defaults)):
                    raise Unchecked("unmodeled class-body statement (a decorator or a default "
                                    "value, both evaluated at class-definition time)")
                annotations = [a.annotation for a in
                               (item.args.posonlyargs + item.args.args + item.args.kwonlyargs)] + [item.returns]
                if any(a is not None and not isinstance(a, (ast.Name, ast.Constant)) for a in annotations):
                    raise Unchecked("unmodeled class-body statement (a non-trivial parameter/return "
                                    "annotation, evaluated at class-definition time)")
                continue
            if isinstance(item, ast.Pass):
                continue
            if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
                continue
            if isinstance(item, ast.AnnAssign) and item.value is None:
                continue
            raise Unchecked(f"unmodeled class-body statement {type(item).__name__} "
                            "(a class body executes at definition time)")

    def predicate(self, text, env):
        check_predicate(text, {n: v[0] for n, v in env.items()}, builtin_names=BUILTINS - self.imports.keys() - self.functions.keys())
        return self.expr(ast.parse(text, mode="eval").body, env, z3.BoolVal(True), predicate=True)[1]

    def at(self, s, env, pc):
        loc = self.points[s][1]
        if loc in self.invariants:
            self.unvisited.discard(loc)
        for inv in self.invariants.get(loc, []):
            self.prove(pc, self.predicate(inv.value, env))

    def block(self, body, env, pc, *, returns=None, aliases=None):
        if (self.range_depth or self.primitive_exits or self.closed_callables) and self.unsat(pc):
            return []  # Only an independently refuted range path is skipped.
        for index, s in enumerate(body):
            self.steps += 1
            if self.steps > self.work_limit or time.monotonic() > self.deadline:
                raise Unchecked("source checking budget exhausted")
            self.at(s, env, pc)
            if isinstance(s, ast.Pass):
                continue
            if isinstance(s, ast.Import) and not self.calls:
                # ``import _sv_verifier [as m]``: the track's own spelling of a draw is
                # ``_sv_verifier.nondet_int()``; only that module, only its three scalar draws
                # (see :meth:`expr`), under the same no-rebinding rule as ``from ... import``.
                for a in s.names:
                    if a.name != "_sv_verifier":
                        raise Unchecked("unmodeled import")
                    name = a.asname or a.name
                    if name in env or name in self.functions or name in self.imports or name in BUILTINS:
                        raise Unchecked("rebound imported name")
                    self.imports[name] = self.MODULE
                continue
            if isinstance(s, ast.ImportFrom) and not self.calls:
                if s.module != "_sv_verifier" or s.level:
                    raise Unchecked("unmodeled import")
                for a in s.names:
                    if a.name not in {"nondet_int", "nondet_bool", "nondet_float", "check_type"}:
                        raise Unchecked("unmodeled nondeterministic input")
                    name = a.asname or a.name
                    if name in env or name in self.functions or name in self.imports or name in BUILTINS:
                        raise Unchecked("rebound imported name")
                    self.imports[name] = self.CHECK_TYPE if a.name == "check_type" else a.name.removeprefix("nondet_")
                continue
            if isinstance(s, ast.FunctionDef) and not self.calls:
                if self.closed_callables:
                    if s.name in env or s.name in self.imports or s.name in self.functions or s.name in BUILTINS:
                        raise Unchecked("rebound closed callable definition")
                    self.functions[s.name] = s
                    continue
                none_return = (not s.args.args and isinstance(s.returns, ast.Constant)
                               and s.returns.value is None)
                scalar_return = (isinstance(s.returns, ast.Name)
                                 and s.returns.id in {"int", "bool"})
                if (s.decorator_list or s.returns is not None and not (none_return or scalar_return)
                        or s.type_params or s.name in env or s.name in BUILTINS):
                    raise Unchecked("only plain functions are supported")
                if s.args.args or scalar_return or (self.primitive_exits and any(
                        isinstance(n, ast.Return) and n.value is not None for n in ast.walk(s))) or any(
                        isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)
                                     for n in ast.walk(s)):
                    # [internal ref] slice 2b, item 1: a function WITH parameters is checked
                    # modularly at its definition; scalar leaves also model calls.
                    # A stricter arg-shape check than
                    # the zero-argument path below (annotations required) lives there.
                    self.parameterized(s)
                    continue
                if s.args.posonlyargs or s.args.kwonlyargs or s.args.vararg or s.args.kwarg:
                    raise Unchecked("only plain zero-argument functions are supported")
                if s.name in self.imports or s.name in self.functions:
                    raise Unchecked("only plain zero-argument functions are supported")
                self.functions[s.name] = s
                if any(isinstance(stmt, ast.Return) and stmt.value is not None for stmt in s.body):
                    self.factory_helpers.add(s.name)
                    try:
                        self.factory_shape(s)
                    except Unchecked:
                        try:
                            self.bounded_shape(s)
                        except Unchecked:
                            pass  # A later actual call still has to authenticate its body.
                        else:
                            self.factory_helpers.remove(s.name)
                            self.bounded_helpers.add(s.name)
                continue
            if isinstance(s, ast.ClassDef) and not self.calls:
                # [internal ref] slice 2b, item 2: see _Checker.classdef.
                if s.name in env:
                    raise Unchecked("class definition shadows an existing value")
                self.classdef(s)
                continue
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
                hint = self.check_type_hint(s.value)
                if hint is not None:
                    kind, _ = self.expr(s.value.args[0], env, pc)
                    if self.primitive_exits and hint == "int" and kind in {"int", "bool", "str"}:
                        # Contract metadata is a gate only; exact source sorts establish the hook-free exit here.
                        from tharkun.solve.contracts.rules import get_rule
                        from tharkun.solve.contracts.visitors import ProductVisitor
                        rule = get_rule("_sv_verifier.check_type")
                        if rule is None or rule.accept(ProductVisitor()) is not True:
                            raise Unchecked("primitive exit requires a tested checker contract")
                        self.primitive_check_seen = True
                        if kind == "str":
                            return []  # Uncaught TypeError, after full argument evaluation; not an AssertionError path.
                    if kind != hint and not (kind == "bool" and hint == "int"):
                        raise Unchecked("check_type operand does not match its primitive hint")
                    continue
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name) and s.value.func.id in self.functions:
                call = s.value
                function = self.functions[call.func.id]
                if (call.func.id in self.factory_helpers or call.func.id in self.bounded_helpers or isinstance(function.returns, ast.Name)
                        and function.returns.id in {"int", "bool"}):
                    self.expr(call, env, pc)
                    continue
                if call.args or call.keywords or self.calls:
                    raise Unchecked("nested/recursive function call")
                self.calls.append(call.func.id)
                try:
                    # Globals are read-only; global/nonlocal statements are refused.
                    nodes = tuple(ast.walk(function))
                    supported = (ast.Assign, ast.AugAssign, ast.Expr, ast.Return,
                                 ast.If, ast.While, ast.For, ast.Assert, ast.Pass)
                    if any(isinstance(node, ast.stmt) and node is not function
                           and not (isinstance(node, supported) or node in self.main_annotations)
                           for node in nodes):
                        raise Unchecked("unmodeled callee declaration/effect")
                    locals_ = {node.id for node in nodes if isinstance(node, ast.Name)
                               and isinstance(node.ctx, ast.Store)}
                    if locals_ & (BUILTINS | self.imports.keys() | self.functions.keys()):
                        raise Unchecked("callee shadows builtin/imported/callable name")
                    # Python locals start unbound even when the module has the same name.
                    local_env = {name: value for name, value in env.items() if name not in locals_}
                    self.block(function.body, local_env, pc)
                finally:
                    self.calls.pop()
                continue
            if isinstance(s, ast.Return) and s.value is None and self.calls:
                return []  # End this callee path; the caller continues after block().
            if isinstance(s, ast.Return) and returns is not None:
                value = self.expr(s.value, env, pc)
                if value[0] not in {"int", "bool"}:
                    raise Unchecked("bounded helper primitive return required")
                returns.append((pc, value))
                return []
            if isinstance(s, ast.If):
                test = self.condition(s.test, env, pc)
                # Acyclic source paths; neither branch's facts are assumed on its sibling.
                rest = body[index + 1:]
                return (self.block(s.body + rest, dict(env), z3.And(pc, test), returns=returns, aliases=aliases) +
                        self.block(s.orelse + rest, dict(env), z3.And(pc, z3.Not(test)), returns=returns, aliases=aliases))
            if isinstance(s, ast.While):
                head, exit_pc = self.loop(s, env, pc)
                return self.block(s.orelse + body[index + 1:], head, exit_pc)
            if isinstance(s, ast.For):
                values = self.fixed_range_values(s, env)
                states = [(env, pc)]
                self.range_depth += 1
                try:
                    for value in values:
                        following = []
                        for state, path in states:
                            self.at(s, state, path)
                            state = dict(state)
                            state[s.target.id] = "int", z3.IntVal(value)
                            following.extend(self.block(s.body, state, path))
                        states = following
                finally:
                    self.range_depth -= 1
                rest = []
                for state, path in states:
                    self.at(s, state, path)  # The exhausted iterator reaches the same original head.
                    rest.extend(self.block(body[index + 1:], state, path))
                return rest
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Tuple):
                self._tuple_work(len(s.targets[0].elts) + len(self.imports) + len(self.functions))
                forbidden = BUILTINS | self.imports.keys() | self.functions.keys()
                names = [target.id for target in s.targets[0].elts if isinstance(target, ast.Name)]
                if (len(names) != len(s.targets[0].elts) or len(set(names)) != len(names)
                        or any(name in env or self.bindings.get(name) != 1
                               or name in forbidden for name in names)
                        or not isinstance(s.value, ast.Call) or not isinstance(s.value.func, ast.Name)
                        or s.value.func.id not in self.tuple_helpers):
                    raise Unchecked("tuple unpack requires a helper call and fresh distinct names")
                kind, values = self.expr(s.value, env, pc)
                if kind != "tuple" or len(values) != len(names):
                    raise Unchecked("tuple unpack target/result arity mismatch")
                self._tuple_work(0)
                env.update(zip(names, values))
                self._tuple_work(0)
                continue
            if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
                name = s.targets[0].id
                if name in BUILTINS or name in self.imports or name in self.functions:
                    raise Unchecked("rebound builtin/callable")
                if self.closed_callables:
                    states = []
                    for value, path in self.closed_callable_cases(s.value, env, pc):
                        local = dict(env)
                        local[name] = value
                        states.extend(self.block(body[index + 1:], local, path, returns=returns))
                    return states
                if (self.primitive_exits and s in self.tree.body and isinstance(s.value, ast.Call)
                        and isinstance(s.value.func, ast.Name) and s.value.func.id in self.primitive_helpers):
                    states = []
                    for value, path in self.primitive_cases(s.value, env, pc):
                        local = dict(env)
                        local[name] = value
                        states.extend(self.block(body[index + 1:], local, path, returns=returns))
                    return states
                value = self.expr(s.value, env, pc)
                pc = self.normal_path(pc)
                env[name] = value
                if aliases:
                    env.update((alias, env[name]) for alias in aliases.get(name, ()))
                if env[name][0] == "callable":
                    raise Unchecked("returned callables require immediate invocation")
                if returns is not None and env[name][0] not in {"int", "bool"}:
                    raise Unchecked("bounded helper primitive local required")
                if not self.calls and s in self.tree.body and name in self.constant_names:
                    self.constants[name] = env[name]
                if not self.calls and s in self.tree.body and name in self.readonly_names and env[name][0] in {"int", "bool"}:
                    self.global_values[name] = env[name]
                continue
            if isinstance(s, ast.AnnAssign):
                if s not in self.main_annotations or self.calls != ["main"]:
                    raise Unchecked("annotated assignment is not an authenticated main local")
                name = s.target.id
                if name in BUILTINS or name in self.imports or name in self.functions:
                    raise Unchecked("rebound builtin/callable")
                value = self.expr(s.value, env, pc)
                if value[0] not in {"int", "bool"}:
                    raise Unchecked("annotated main local requires an actual exact int or bool")
                env[name] = value  # Local annotations are not evaluated and establish no kind.
                continue
            if isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
                env[s.target.id] = self.binary(s.op, self.expr(s.target, env, pc), self.expr(s.value, env, pc), pc, env=env)
                continue
            if isinstance(s, ast.Assert):
                test = self.expr(s.test, env, pc)
                if test[0] != "bool" or s.msg is not None:
                    raise Unchecked("unmodeled assertion")
                if "AssertionError" in self.classes:
                    self.prove(pc, test[1])
                pc = z3.And(pc, test[1])  # failing assertion exits this path
                continue
            if isinstance(s, ast.Expr):
                self.expr(s.value, env, pc)
                continue
            raise Unchecked(f"unmodeled statement {type(s).__name__} at line {s.lineno}")
        return [(env, pc)]

    def fixed_range_values(self, node, env):
        # Closed integer ranges are enumerated completely within the shared work/deadline budget.
        call = node.iter
        if (node.orelse or not isinstance(node.target, ast.Name)
                or self.bindings.get(node.target.id) != 1
                or node.target.id in env and self.range_targets.get(node.target.id) is not node
                or node.target.id == "__name__"
                or node.target.id in BUILTINS | self.imports.keys() | self.functions.keys()
                or not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                or call.func.id != "range" or "range" in self.bindings
                or len(call.args) not in {1, 2, 3} or call.keywords):
            raise Unchecked("fixed ranges require an unrebound builtin and one new scalar target")
        bounds = []
        bound_nodes = (ast.Constant, ast.Name, ast.Load, ast.UnaryOp, ast.UAdd, ast.USub,
                       ast.BinOp, ast.Add, ast.Sub, ast.Mult)
        for argument in call.args:
            nodes = tuple(self._tuple_nodes(argument))
            if any(not isinstance(n, bound_nodes)
                    or isinstance(n, ast.Constant) and type(n.value) not in {int, bool}
                    or isinstance(n, ast.Name) and (n.id not in self.constants or self.constants[n.id][0] not in {"int", "bool"})
                    for n in nodes):
                raise Unchecked("fixed ranges require closed integer bounds")
            kind, value = self.expr(argument, self.constants, z3.BoolVal(True))
            if kind == "bool":
                value = z3.If(value, z3.IntVal(1), z3.IntVal(0))
            value = z3.simplify(value)
            if kind not in {"int", "bool"} or not z3.is_int_value(value):
                raise Unchecked("fixed range bound is not a concrete integer")
            bounds.append(value.as_long())
        start, stop, step = ((0, bounds[0], 1) if len(bounds) == 1 else
                             (*bounds, 1) if len(bounds) == 2 else bounds)
        if step == 0:
            raise Unchecked("fixed range step cannot be zero")
        allowed = (ast.Assign, ast.If, ast.For, ast.Assert, ast.Return, ast.Expr, ast.Pass)
        if any(isinstance(n, ast.stmt) and (not isinstance(n, allowed)
                   or isinstance(n, ast.Return) and n.value is not None)
               for statement in node.body for n in ast.walk(statement)):
            raise Unchecked("unmodeled fixed-range body control/effect")
        self._tuple_work(0)
        distance = stop - start if step > 0 else start - stop
        count = 0 if distance <= 0 else 1 + (distance - 1) // abs(step)
        if count > self.work_limit - self.steps:
            raise Unchecked("fixed range source checking budget exhausted")
        self.range_targets[node.target.id] = node
        return range(start, stop, step)

    def loop(self, node, env, pc):
        """Initiation, preservation, transition safety and exit; no bounded unrolling."""
        invariants = self.invariants.get(self.points[node][1], [])
        if not any(i.type == "loop_invariant" for i in invariants):
            raise Unchecked("missing loop invariant")
        # Frame rule: all writes must be ordinary scalar local assignments. No
        # mutable heap, declarations, break/continue/return or exception edges yet.
        allowed = (ast.Assign, ast.AugAssign, ast.If, ast.While, ast.Assert, ast.Expr, ast.Pass)
        for stmt in node.body:
            if any(isinstance(n, ast.stmt) and not isinstance(n, allowed) for n in ast.walk(stmt)):
                raise Unchecked("unmodeled loop control/effect")
        written = {n.id for stmt in node.body for n in ast.walk(stmt)
                   if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        if any(name not in env or env[name][0] not in {"int", "float", "bool"} for name in written):
            raise Unchecked("loop-carried value is not a bound scalar")
        head = dict(env)
        for name in sorted(written):
            head[name] = self.fresh(env[name][0])
        # at() already established the base case. Entry constraints on unchanged
        # values persist; written values have fresh symbols at the arbitrary head.
        head_pc = z3.And(pc, *(self.predicate(i.value, head) for i in invariants))
        kind, guard = self.expr(node.test, head, head_pc)
        if kind != "bool":
            raise Unchecked("implicit loop truth conversion")
        for back, back_pc in self.block(node.body, dict(head), z3.And(head_pc, guard)):
            if any(back.get(name, (None,))[0] != head[name][0] for name in written):
                raise Unchecked("loop-carried type changes")
            for inv in invariants:
                self.prove(back_pc, self.predicate(inv.value, back))
        return head, z3.And(head_pc, z3.Not(guard))

    def expr(self, n, env, pc, *, predicate=False):
        if self.closed_callables:
            # ``object`` is the predicate checker's opaque sort, never an exact builtin object type.
            if isinstance(n, ast.Name) and n.id not in env and n.id in self.functions:
                return "object", self.functions[n.id]
            if (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Is, ast.IsNot))
                    and all(isinstance(x, ast.Name) and (x.id in env or x.id in self.functions)
                            for x in (n.left, n.comparators[0]))):
                a, b = self.expr(n.left, env, pc, predicate=predicate), self.expr(n.comparators[0], env, pc, predicate=predicate)
                if a[0] == b[0] == "object":
                    return "bool", z3.BoolVal((a[1] is b[1]) == isinstance(n.ops[0], ast.Is))
            if isinstance(n, ast.Call):
                if (isinstance(n.func, ast.Name) and n.func.id == "type" and len(n.args) == 1 and not n.keywords
                        and isinstance(n.args[0], ast.Name) and (n.args[0].id in env or n.args[0].id in self.functions)):
                    arg = self.expr(n.args[0], env, pc, predicate=predicate)
                    if arg[0] == "object":
                        return "type", "function"
                if (isinstance(n.func, ast.Name) and (n.func.id in env or n.func.id in self.functions)):
                    if predicate:
                        raise Unchecked("closed callable invocations are not invariant predicates")
                    return self.closed_callable_value(n, env, pc)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Lambda) and not predicate:
            return self.lambda_value(n, env, pc)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float, bool, str):
            kind = type(n.value).__name__
            value = {"int": z3.IntVal, "bool": z3.BoolVal, "str": str, "float": lambda x: z3.FPVal(x, z3.Float64())}[kind](n.value)
            return kind, value
        if isinstance(n, ast.Name) and n.id in env:
            return env[n.id]
        if (isinstance(n, ast.Name) and n.id in self.functions and n.id in self.factory_helpers
                and not predicate):
            function = self.functions[n.id]
            self.factory_shape(function)
            return "callable", function
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Call) and not predicate:
            kind, function = self.expr(n.func, env, pc)
            if kind != "callable" or n.args or n.keywords:
                raise Unchecked("immediate returned callable requires a closed zero-argument binding")
            return self.factory_value(function, pc)
        if isinstance(n, ast.IfExp) and not predicate:
            if any(isinstance(part, ast.Call) for part in ast.walk(n)):
                raise Unchecked("conditional expressions require primitive operands")
            test = self.source_truth(self.expr(n.test, env, pc))
            yes = self.expr(n.body, env, z3.And(pc, test))
            no = self.expr(n.orelse, env, z3.And(pc, z3.Not(test)))
            if yes[0] not in {"int", "bool"} or no[0] != yes[0]:
                raise Unchecked("conditional expression requires matching int/bool branches")
            return yes[0], z3.If(test, yes[1], no[1])
        if isinstance(n, (ast.List, ast.Tuple)) and not predicate:
            if isinstance(n, ast.Tuple):
                return self.tuple_cases(n, env, pc)
            vals = tuple(self.expr(e, env, pc) for e in n.elts)
            if any(v[0] not in {"int", "float", "bool"} for v in vals):
                raise Unchecked("nested/mutable container")
            return ("list" if isinstance(n, ast.List) else "tuple"), vals
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                and self.imports.get(n.func.value.id) == self.MODULE and not n.args and not n.keywords
                and not predicate and n.func.attr in {"nondet_int", "nondet_bool", "nondet_float"}):
            return self.fresh(n.func.attr.removeprefix("nondet_"))
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.functions
                and n.func.id not in env and not predicate):
            name = n.func.id
            function = self.functions[name]
            if name in self.factory_helpers:
                if n.args or n.keywords:
                    raise Unchecked("factory call requires zero arguments")
                return self.factory_value(function, pc)
            if name in self.tuple_helpers and n not in self.tuple_calls:
                raise Unchecked("tuple helper result must be immediately unpacked")
            if name in self.tuple_helpers:
                self._tuple_work(len(function.args.args) + len(n.args) + len(n.keywords))
                for _ in self._tuple_nodes(n):
                    pass
            if not (name in self.tuple_helpers or name in self.bounded_helpers or isinstance(function.returns, ast.Name)
                    and function.returns.id in {"int", "bool"}):
                raise Unchecked("only scalar leaf helpers return modeled values")
            return self.helper_value(function, self.bind_arguments(n, function, env, pc), pc)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords:
            name = n.func.id
            if name in self.imports and not n.args and not predicate:
                if self.imports[name] == self.MODULE:
                    raise Unchecked("a module is not callable")
                return self.fresh(self.imports[name])
            if (name == "int" and not predicate and name not in env
                    and name not in self.functions and name not in self.imports):
                from tharkun.solve.contracts.rules import get_rule
                from tharkun.solve.contracts.visitors import ProductVisitor
                rule = get_rule(name)
                arity = rule.accept(ProductVisitor()) if rule is not None else None
                if (type(arity) is not int or arity != 1 or len(n.args) != arity
                        or any(isinstance(a, ast.Starred) for a in n.args)):
                    raise Unchecked("int requires a tested one-argument rule")
                kind, value = self.expr(n.args[0], env, pc)
                if kind != "int":
                    raise Unchecked("int identity requires an exact int operand")
                return kind, value
            if (name in {"min", "max"} and not predicate and name not in env
                    and name not in self.functions and name not in self.imports):
                from tharkun.solve.contracts.rules import get_rule
                from tharkun.solve.contracts.visitors import ProductVisitor
                rule = get_rule(name)
                arity = rule.accept(ProductVisitor()) if rule is not None else None
                if (type(arity) is not int or arity != 2 or len(n.args) != arity
                        or any(isinstance(a, ast.Starred) for a in n.args)):
                    raise Unchecked("min/max requires a tested two-argument rule")
                # Admission only: re-evaluate both source arguments before deriving the value.
                a, b = [self.expr(arg, env, pc) for arg in n.args]
                if a[0] != "int" or b[0] != "int":
                    raise Unchecked("min/max requires exact int operands")
                first = a[1] <= b[1] if name == "min" else a[1] >= b[1]
                return "int", z3.If(first, a[1], b[1])
            if (name == "abs" and len(n.args) == 1 and not predicate and name not in env
                    and name not in self.functions and name not in self.imports):
                # abs() on the three scalar sorts: |x| for int (Python's own unbounded int), the
                # IEEE sign-bit clear for float (abs(-0.0) is 0.0, abs(nan) is nan), 0/1 for bool
                # (abs(True) is the int 1). Anything else is refused, never guessed.
                kind, v = self.expr(n.args[0], env, pc)
                if kind == "int":
                    return "int", z3.If(v >= 0, v, -v)
                if kind == "float":
                    return "float", z3.fpAbs(v)
                if kind == "bool":
                    return "int", z3.If(v, 1, 0)
                raise Unchecked("abs of an unmodeled operand")
            if name in {"len", "type", "isinstance"} and name not in env and name not in self.functions and name not in self.imports:
                if len(n.args) != (2 if name == "isinstance" else 1):
                    raise Unchecked("call arity")
                arg = self.expr(n.args[0], env, pc, predicate=predicate)
                if name == "type":
                    return "type", arg[0]
                if name == "len" and arg[0] in {"list", "tuple"}:
                    return "int", z3.IntVal(len(arg[1]))
                if name == "isinstance" and isinstance(n.args[1], ast.Name):
                    target = n.args[1].id
                    if target not in BUILTINS or target in env or target in self.functions or target in self.imports:
                        raise Unchecked("unresolved isinstance type")
                    if target in {"len", "isinstance"} and "TypeError" in self.classes:
                        self.prove(pc, z3.BoolVal(False))  # These builtin functions are not class hints.
                    return "bool", z3.BoolVal(arg[0] == target or arg[0] == "bool" and target == "int")
        if isinstance(n, ast.UnaryOp):
            kind, v = self.expr(n.operand, env, pc, predicate=predicate)
            if isinstance(n.op, ast.Not) and (kind == "bool" or not predicate):
                return "bool", z3.Not(self.source_truth((kind, v)))
            if isinstance(n.op, (ast.UAdd, ast.USub)) and kind in {"int", "float"}:
                return kind, v if isinstance(n.op, ast.UAdd) else (-v if kind == "int" else z3.fpNeg(v))
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.LShift):
            # shortcut: literal integer shifts up to 64; extend only for a checked proof need.
            if not (isinstance(n.left, ast.Constant) and type(n.left.value) is int
                    and isinstance(n.right, ast.Constant) and type(n.right.value) is int
                    and 0 <= n.right.value <= 64):
                raise Unchecked("only literal integer left shifts by 0..64 are supported")
            return "int", z3.IntVal(n.left.value << n.right.value)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd):
            if predicate or not self.classes or self.classes - {"TypeError", "AttributeError"}:
                raise Unchecked("BitAnd is source-only for TypeError/AttributeError")
            left = self.expr(n.left, env, pc, predicate=predicate)
            right = self.expr(n.right, env, pc, predicate=predicate)
            if left[0] != "int" or right[0] != "int":
                raise Unchecked("BitAnd requires actual exact int operands")
            # Retain every concrete result, without width, sign or correlation facts.
            return self.fresh("int")
        if isinstance(n, ast.BinOp):
            return self.binary(n.op, self.expr(n.left, env, pc, predicate=predicate), self.expr(n.right, env, pc, predicate=predicate), pc, env=env, predicate=predicate)
        if isinstance(n, ast.BoolOp):
            terms = []
            for part in n.values:
                kind, term = self.expr(part, env, pc, predicate=predicate)
                if kind != "bool":
                    raise Unchecked("implicit truth conversion")
                terms.append(term)
                pc = z3.And(pc, term if isinstance(n.op, ast.And) else z3.Not(term))
            return "bool", (z3.And if isinstance(n.op, ast.And) else z3.Or)(*terms)
        if isinstance(n, ast.Compare):
            terms = []
            a = self.expr(n.left, env, pc, predicate=predicate)
            for op, right in zip(n.ops, n.comparators):
                if a[0] == "type" and isinstance(op, (ast.Is, ast.IsNot)) and isinstance(right, ast.Name) and right.id in BUILTINS - env.keys() - self.functions.keys() - self.imports.keys():
                    b = ("type", right.id)
                    term = z3.BoolVal((a[1] == b[1]) == isinstance(op, ast.Is))
                else:
                    b = self.expr(right, env, pc, predicate=predicate)
                    if isinstance(op, (ast.In, ast.NotIn)) and b[0] in {"list", "tuple"}:
                        # ponytail: FP values omit object identity. Refuse float
                        # membership until identity-or-equality is modeled (NaNs).
                        if a[0] == "float" or any(v[0] == "float" for v in b[1]):
                            raise Unchecked("float membership needs object identity")
                        term = z3.Or(*(self.compare(ast.Eq(), a, x) for x in b[1]))
                        if isinstance(op, ast.NotIn):
                            term = z3.Not(term)
                    else:
                        term = self.compare(op, a, b)
                terms.append(term)
                pc = z3.And(pc, term)
                a = b
            return "bool", z3.And(*terms)
        if isinstance(n, ast.Subscript) and not predicate:
            container, values = self.expr(n.value, env, pc)
            kind, index = self.expr(n.slice, env, pc)
            if container not in {"list", "tuple"} or kind != "int" or not values or len({v[0] for v in values}) != 1:
                raise Unchecked("unmodeled subscription")
            safe = z3.And(index >= -len(values), index < len(values))
            if "IndexError" in self.classes:
                self.prove(pc, safe)
            # Continuing after a non-property exception is an overapproximation.
            index = z3.If(index < 0, index + len(values), index)
            value = values[-1][1]
            for i, (_, v) in reversed(list(enumerate(values[:-1]))):
                value = z3.If(index == i, v, value)
            return values[0][0], value
        raise Unchecked(f"unmodeled expression {ast.dump(n, include_attributes=False)[:160]}")

    def binary(self, op, a, b, pc, *, env=None, predicate=False):
        kind, x = a
        other, y = b
        if kind != other or kind not in {"int", "float"}:
            raise Unchecked("unmodeled numeric coercion")
        if isinstance(op, (ast.Add, ast.Sub, ast.Mult)):
            if kind == "float":
                fn = {ast.Add: z3.fpAdd, ast.Sub: z3.fpSub, ast.Mult: z3.fpMul}[type(op)]
                return kind, fn(z3.RNE(), x, y)
            return kind, {ast.Add: lambda: x + y, ast.Sub: lambda: x - y, ast.Mult: lambda: x * y}[type(op)]()
        if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)):
            nonzero = y != 0 if kind == "int" else z3.Not(z3.fpIsZero(y))
            if self.class_exit is not None and not predicate:
                self.class_exit(env, z3.And(pc, z3.Not(nonzero)))
                # An unevaluated conditional arm must not constrain the normal path.
                self.class_guards.append(z3.Implies(pc, nonzero))
            elif "ZeroDivisionError" in self.classes:
                self.prove(pc, nonzero)
            elif kind == "int" and not predicate:
                # Only normal continuation needs a nonzero divisor, guarded by
                # this operation's source path (including short-circuit guards).
                self.normal_conditions.append(z3.Implies(pc, nonzero))
            if kind == "int" and isinstance(op, (ast.FloorDiv, ast.Mod)):
                quotient = z3.If(y > 0, x / y, (-x) / (-y))
                return kind, quotient if isinstance(op, ast.FloorDiv) else x - y * quotient
            if kind == "float" and isinstance(op, ast.Div):
                return kind, z3.fpDiv(z3.RNE(), x, y)
            if kind == "int" and isinstance(op, ast.Div):
                # ponytail: overapproximate int true-division result; model rounding
                # and OverflowError precisely only if downstream proofs need it.
                return self.fresh("float")
        raise Unchecked("unmodeled arithmetic")

    @staticmethod
    def compare(op, a, b):
        ka, x = a
        kb, y = b
        if ka == "bool" and kb in {"int", "float"}:
            ka, x = "int", z3.If(x, 1, 0)
        if kb == "bool" and ka in {"int", "float"}:
            kb, y = "int", z3.If(y, 1, 0)
        if ka == kb == "float":
            eq = z3.fpEQ(x, y)
            table = {ast.Eq: eq, ast.NotEq: z3.Not(eq), ast.Lt: z3.fpLT(x, y), ast.LtE: z3.fpLEQ(x, y), ast.Gt: z3.fpGT(x, y), ast.GtE: z3.fpGEQ(x, y)}
        elif ka == kb and ka in {"int", "bool"}:
            if ka == "bool":
                x, y = z3.If(x, 1, 0), z3.If(y, 1, 0)
            table = {ast.Eq: x == y, ast.NotEq: x != y, ast.Lt: x < y, ast.LtE: x <= y, ast.Gt: x > y, ast.GtE: x >= y}
        elif ka == kb == "str":
            # ponytail: strings are concrete in this fragment. Use exact Python
            # equality; model sequences before adding symbolic string inputs.
            # StringVal interprets literal backslash-u-brace escapes.
            table = {ast.Eq: z3.BoolVal(x == y), ast.NotEq: z3.BoolVal(x != y)}
        elif {ka, kb} == {"int", "float"}:
            # Python compares float/int exactly, without rounding the int to float.
            f = x if ka == "float" else y
            l, r = (z3.fpToReal(x), z3.ToReal(y)) if ka == "float" else (z3.ToReal(x), z3.fpToReal(y))
            finite = z3.And(z3.Not(z3.fpIsNaN(f)), z3.Not(z3.fpIsInf(f)))
            eq = z3.And(finite, l == r)
            inf_less = z3.And(z3.fpIsInf(f), z3.fpIsNegative(f) if ka == "float" else z3.Not(z3.fpIsNegative(f)))
            inf_greater = z3.And(z3.fpIsInf(f), z3.Not(z3.fpIsNegative(f)) if ka == "float" else z3.fpIsNegative(f))
            table = {ast.Eq: eq, ast.NotEq: z3.Not(eq), ast.Lt: z3.Or(inf_less, z3.And(finite, l < r)), ast.LtE: z3.Or(inf_less, z3.And(finite, l <= r)), ast.Gt: z3.Or(inf_greater, z3.And(finite, l > r)), ast.GtE: z3.Or(inf_greater, z3.And(finite, l >= r))}
        else:
            raise Unchecked("unmodeled comparison")
        if type(op) not in table:
            raise Unchecked("unmodeled comparison operator")
        return table[type(op)]
