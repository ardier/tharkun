"""The contract x engine audit ([internal ref] batch 5, ticket [internal ref] / [internal ref]): every contract of the
registry against every engine, measured, not asserted.

``Rule.sample`` holds one valid program per contract.  Each probe below hands it to ONE
engine and reads what the engine did with the call:

* ``C`` -- consumed: the engine's handling of the call is driven by the contract (it records the contract's
  seven property classes, folds with a table gated on it, reads its range, length effect, reachability row or
  receiver evidence);
* ``M`` -- modelled by hand: the engine handles the call, but a parallel hand model decides; the contract is
  not read.  NOT consumed;
* ``R`` -- refused, with a reason that names what the engine could not do;
* ``G`` -- refused generically ("call of an unknown value"): the engine says nothing about the call.  NOT
  consumed, and the cell to close first;
* ``X`` -- dead: an adapter reads the contract but no engine calls the adapter.  NOT consumed;
* ``-`` -- not applicable: nothing for the engine to do with this callable, by design.

absint, the symbolic executor, the exact product stage and the census (with its obligations) are probed by
RUNNING them on the sample.  The stationary planner, Avengers, the draws and the Eregion bridge are read from
the one place each of them consults a contract (``adapters/common.py``'s tables, ``adapters/draws.py``,
``adapters/avengers.py``, the ``tharkun.solve`` facade), because that is the only way they ever see one.

``tests/contracts/frozen/<qualname>.json`` (one file per contract, ``_frozen.py``) freezes the measured table; ``test_engine_audit.py``
fails when an engine's behaviour changes (close a cell: regenerate that contract's file, ``python -m tests.contracts._frozen write --audit
--only <qualname>``); :func:`summary` and :func:`render` print it.  ``test_engine_audit_consumed.py`` runs every
``C`` cell end to end against CPython.
"""

from __future__ import annotations

import ast
import sys

from tharkun.solve.contracts.rules import Refused, all_rules, get_rule
from tharkun.solve.absint.interp import AbsintVisitor
from tharkun.sv.symexec_visitor import SymexecVisitor
from tharkun.solve.contracts.visitors import (
    AvengersVisitor, CensusVisitor, DrawsVisitor, EregionVisitor, ProductVisitor, StationaryVisitor,
)

from tests.contracts._shape import benchmark_shape
from tharkun.solve.contracts import all_contracts, get
from tharkun.solve.contracts.adapters import common
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES


SAMPLES = {r.qualname: r.sample(r.qualname) for r in all_rules()}
VISITORS = {
    "absint": AbsintVisitor, "symexec": SymexecVisitor, "product": ProductVisitor,
    "stationary": StationaryVisitor, "census": CensusVisitor, "avengers": AvengersVisitor,
    "draws": DrawsVisitor, "eregion": EregionVisitor,
}
ENGINES = ("absint", "symexec", "product", "stationary", "census", "avengers", "draws", "eregion")
ENGINE_TITLE = {"absint": "absint", "symexec": "symbolic", "product": "product", "stationary": "stationary",
                "census": "census", "avengers": "Avengers", "draws": "draws", "eregion": "Eregion"}
CODES = {
    "C": "consumed: the engine's handling is driven by the contract",
    "M": "modelled by hand, the contract is not read",
    "R": "refused, with a reason",
    "G": "refused generically (the engine says nothing about the call)",
    "X": "dead: an adapter reads the contract, nothing calls the adapter",
    "-": "not applicable",
}
#: the codes that mean the engine really reads the contract
CONSUMED = "C"
_GENERIC = ("call of an unknown value", "method call on an unknown value", "unknown value", "unknown argument")
_CENSUS_CLASSES = ("ZeroDivisionError", "FloatingPointError", "AssertionError", "KeyError", "IndexError")
_ALL = tuple(ALL_PROPERTY_CLASSES)


def contract_names() -> list[str]:
    return [c.qualname for c in all_contracts()]


# --------------------------------------------------------------------------- absint
def absint_cell(qualname: str, src: str) -> tuple[str, str]:
    from tharkun.solve.absint import prove

    r = prove(src, _ALL, time_budget=30.0)
    reason = r.reason
    if r.verdict == "UNKNOWN" and (reason.startswith("unsupported:") or reason.startswith("environment:")
                                   or reason.startswith("syntax") or "does not resolve" in reason):
        text = reason.split(":", 1)[1].strip() if ":" in reason else reason
        named = ": " in text.split(" (line")[0]       # "call of an unknown value: ord": names the callee
        return ("G" if text.startswith(_GENERIC) and not named else "R"), text
    ops = {s[2] for s in r.sites}
    driven = any(op.startswith(f"{op_name(qualname)}:") for op in ops)
    if not driven and (qualname.startswith("protocol.") or get_rule(qualname).kind == "math_constant"):
        # a protocol and a constant leave no op named after them: consumed when withdrawing the contract changes what absint does
        driven = _outcome_without(qualname, src) != (r.verdict, r.reason)
    return ("C" if driven else "M"), ""


def op_name(qualname: str) -> str:
    """The name absint records a contract's checks under: the qualname, except for a SHAPE contract
    (``open.output``), which the model records under the builtin it stands in for (``open``)."""
    return {v: k for k, v in common.BUILTIN_SHAPE_CONTRACTS.items()}.get(qualname, qualname)


def _outcome_without(qualname: str, src: str) -> tuple[str, str]:
    """absint's verdict and reason on ``src`` with ``qualname`` removed from the registry (restored after)."""
    from tharkun.solve.absint import prove
    from tharkun.solve.contracts.registry import REGISTRY, load_all

    load_all()
    saved = REGISTRY.pop(qualname)
    try:
        r = prove(src, _ALL, time_budget=30.0)
    finally:
        REGISTRY[qualname] = saved
    return r.verdict, r.reason


# --------------------------------------------------------------------------- the symbolic executor
#: the contracts of the write side of an output file ([internal ref]): ``open`` with a constant write-only mode and the
#: methods of the text and binary handles it returns, read by ``_symexec.py`` through ``output_open_contract`` and
#: ``output_handle_contract``
_OUTPUT_FILE_CONTRACTS = frozenset(
    rule.qualname for rule in all_rules()
    if rule.kind == "open_output" or rule.kind.startswith("output_")
)


def _contract_driven_in_symexec(qualname: str) -> bool:
    rule = get_rule(qualname)
    return rule is not None and SymexecVisitor.cell(rule) == "C"


def symexec_harness(src: str, extra: str = ""):
    """The whole-harness executor run on ``src`` (benchmark shape, plus a draw and a site); returns the runner."""
    import z3  # noqa: F401  (the executor needs it)

    from tharkun.sv import _symexec
    from tharkun.sv.sites import build_harness

    full = benchmark_shape(src + "import _sv_verifier\nn = _sv_verifier.nondet_int()\n" + extra
                           + "if n == 3:\n    raise IndexError('hit')\n")
    h = build_harness(full, ("IndexError",), None)
    if h is None:
        return None, None
    mod = {"__name__": "sv_harness_audit"}
    sys.modules.setdefault("_sv_verifier", type(sys)("_sv_verifier"))
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    tree = ast.parse(h.source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "__sv_harness")
    env = {p: _symexec._var(p, k) for p, k in zip(h.params, h.kinds)}
    runner = _symexec._Runner(tree, mod, env, {}, [s["line"] for s in h.sites])
    try:
        runner.run(list(fn.body))
    except _symexec._Dead:
        pass
    return runner, h


def symexec_cell(qualname: str, src: str) -> tuple[str, str]:
    runner, _h = symexec_harness(src)
    if runner is None:
        return "R", "no harness could be built"
    skipped = [s for s in runner.skipped if "raise" not in s.lower()]
    if not skipped:
        return ("C" if _contract_driven_in_symexec(qualname) else "M"), ""
    reason = skipped[0].split(": ", 1)[1] if ": " in skipped[0] else skipped[0]
    return "R", reason


# --------------------------------------------------------------------------- the exact product
def product_cell(qualname: str, src: str) -> tuple[str, str]:
    from tharkun.sv import exact_subset

    reason = exact_subset.audit(ast.parse(src), set())
    if reason is None:
        # admitted by the allowlist: consumed when the allowlist asks the rule, i.e. withdrawing the contract takes the call out
        from tharkun.solve.contracts.registry import REGISTRY, load_all

        load_all()
        saved = REGISTRY.pop(qualname)
        try:
            without = exact_subset.audit(ast.parse(src), set())
        finally:
            REGISTRY[qualname] = saved
        return ("C" if without is not None else "M"), ""
    return "R", reason


# --------------------------------------------------------------------------- the census and its obligations
def census_cell(qualname: str, src: str) -> tuple[str, str]:
    """The census reads a contract in two places: ``pyresolve.py`` takes a call's audited ordinary hooks from
    ``closed_call_hooks`` (a present row keeps dead-code elimination alive), and ``census.py`` asks
    ``obligation_for_call`` for an invariant or an explicit gap."""
    from tharkun.solve.census import site_census
    from tharkun.solve.contracts import closed_call_hooks

    if closed_call_hooks(qualname) is not None:
        return "C", "a closed-call-hooks row (solve/contracts/reachability.py)"
    try:
        c = site_census(src, _CENSUS_CLASSES, "<audit>")
        obligations = [ob for (_l, _c, ob) in c.obligations if ob.qualname == qualname]
    except Exception as e:  # noqa: BLE001
        return "R", f"census raised {type(e).__name__}"
    for ob in obligations:
        if ob.invariant is not None:
            return "C", f"safety obligation {ob.invariant.value}"
    for ob in obligations:
        return "R", f"obligation gap: {ob.gap.reason}"
    contract = get(qualname)
    informed = any(common.raise_condition(contract, exc) is not None for exc in _CENSUS_CLASSES)
    if informed:
        return "R", "obligations need an exactly resolved callable: a method on an unknown receiver gets none"
    return "-", "no tested non-never claim for a class the census tracks"


# --------------------------------------------------------------------------- the stationary planner
_STATIONARY_REASON = ("a trip value flowing into a value-dependent call has no sound rule short of an induction "
                      "argument (#1094)")


def stationary_cell(qualname: str) -> tuple[str, str]:
    if qualname in common.CLASS_ONLY_SHAPES:
        return "C", "drawflow._class_only_call: the result depends on the class only"
    return "R", _STATIONARY_REASON


# --------------------------------------------------------------------------- Avengers, draws, Eregion
_EVIDENCE_TYPES = ("list", "dict", "set", "tuple", "str", "bytes")


def empty_variant(src: str) -> str | None:
    """The sample with its receiver literal emptied: the empty container the evidence is about."""
    tree = ast.parse(src)
    first = tree.body[0]
    if not (isinstance(first, ast.Assign) and len(first.targets) == 1):
        return None
    value = first.value
    empty = {ast.List: "[]", ast.Dict: "{}", ast.Tuple: "()", ast.Set: "set()"}.get(type(value))
    if empty is None and isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "set":
        empty = "set()"
    if empty is None and isinstance(value, ast.Constant):
        empty = {str: '""', bytes: 'b""'}.get(type(value.value))
    if empty is None:
        return None
    first.value = ast.parse(empty, mode="eval").body
    return ast.unparse(tree) + "\n"


def _evidence_cell(qualname: str) -> tuple[str, str]:
    """The empty-value evidence Avengers and the draws read (``common.receiver_may_raise``): a contract of the
    TYPE whose own logic raises IndexError/KeyError, plus the constructors list/tuple/set.  A claim counts HERE
    only if it also holds on CPython for the sample with its receiver emptied: ``str.format`` raises IndexError
    for a missing argument, which says nothing about an empty string."""
    from tests.contracts._native import run_native

    if qualname in common.ZERO_ARGUMENT_CONSTRUCTORS:
        return "C", "the contracted constructor builds the empty / one-element container"
    owner = qualname.split(".", 1)[0]
    if "." in qualname and owner in _EVIDENCE_TYPES:
        contract = get(qualname)
        empty = empty_variant(SAMPLES[qualname])
        for exc in ("IndexError", "KeyError"):
            if common.native_raise_condition(contract, exc) is not None:
                if empty is not None and run_native(empty) == exc:
                    return "C", f"evidence: {exc} on an empty {owner}"
                return "R", f"the {exc} claim is not about an empty {owner} (the emptied sample does not raise it)"
    return "-", "no empty-value evidence"


def avengers_cell(qualname: str) -> tuple[str, str]:
    return _evidence_cell(qualname)


def draws_cell(qualname: str) -> tuple[str, str]:
    cell = _evidence_cell(qualname)
    if cell[0] in ("C", "R"):
        return cell
    if qualname in common.CLOSED_RANGE_RETURN or qualname == "heapq.heapify":
        return "X", "draws.range_kwargs / draws.heap_ordered exist, nothing calls them"
    return "-", cell[1]


def eregion_cell(qualname: str) -> tuple[str, str]:
    return "G", ("the tharkun.solve facade eregion-synthesis imports (solve_condition, guard_domain, ...) exports no "
                 "contract-driven function")


def arity_twin(src: str, qualname: str) -> str | None:
    """The rule's fixed, native-checked wrong-arity sample."""
    rule = get_rule(qualname)
    if rule is None:
        return None
    assert src == rule.sample(qualname), qualname
    return rule.twin(qualname)


# --------------------------------------------------------------------------- the table
def measure(qualname: str) -> dict[str, tuple[str, str]]:
    src = SAMPLES[qualname]
    probes = {
        "absint": absint_cell(qualname, src),
        "symexec": symexec_cell(qualname, src),
        "product": product_cell(qualname, src),
        "stationary": stationary_cell(qualname),
        "census": census_cell(qualname, src),
        "avengers": avengers_cell(qualname),
        "draws": draws_cell(qualname),
        "eregion": eregion_cell(qualname),
    }
    rule = get_rule(qualname)
    assert rule is not None, qualname
    out = {}
    for engine, (code, reason) in probes.items():
        static = VISITORS[engine].cell(rule)
        if code == "C":
            assert static == "C", (qualname, engine, "consumed outside its visitor")
        if static == "C" and code == "M":
            code, reason = "X", "visit exists but the probe never consumes it"
        if static == "R" and code in {"R", "G"}:
            refusal = rule.accept(VISITORS[engine].__new__(VISITORS[engine]))
            assert isinstance(refusal, Refused), (qualname, engine)
            reason = str(refusal) + "; " + reason
        out[engine] = (code, reason)
    return out


def render(table: dict[str, dict[str, str]]) -> str:
    """The audit as a Markdown table: one row per contract, one cell per engine."""
    head = "| contract | " + " | ".join(ENGINE_TITLE[e] for e in ENGINES) + " |"
    rows = [head, "|---|" + "---|" * len(ENGINES)]
    for q in sorted(table):
        rows.append(f"| `{q}` | " + " | ".join(table[q][e] for e in ENGINES) + " |")
    return "\n".join(rows)


def summary(table: dict[str, dict[str, str]]) -> str:
    lines = ["| engine | " + " | ".join(CODES) + " |", "|---|" + "---|" * len(CODES)]
    for e in ENGINES:
        counts = {k: sum(1 for q in table if table[q][e] == k) for k in CODES}
        lines.append(f"| {ENGINE_TITLE[e]} | " + " | ".join(str(counts[k]) for k in CODES) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    # a human report of one run of the whole audit: the counts and the reason of every refusal.  To FREEZE the table use
    # ``python -m tests.contracts._frozen write --audit`` (one file per contract).
    out = {q: measure(q) for q in contract_names()}
    codes = {q: {e: out[q][e][0] for e in ENGINES} for q in out}
    print(summary(codes))
    for q in sorted(out):
        for e in ENGINES:
            if out[q][e][0] in ("R", "G", "X"):
                print(f"{q:36s} {e:10s} {out[q][e][0]} {out[q][e][1][:120]}")
