"""The concolic assist engine: unblock the fuzzer's coverage frontier.

This is steps 2-4 of the hybrid loop.  Given the current corpus, it:

  2. finds the **high-value frontiers** — uncovered branch edges whose guard line is
     already reached (the border of coverage), ranked by how much new code they gate;
  3. **solves** each with ``driver.drive_to_line`` over the harness's *target* function,
     driven from the fields a corpus seed decodes to; and
  4. **recodes** a solution back into a byte seed and writes it to the corpus directory,
     where Atheris picks it up and keeps going past the guard.

The bytes <-> parameters bridge is pluggable (:mod:`tharkun.hybrid.codec`), because how
to cross it depends on the harness:

* an unmodified OSS harness on ``atheris.FuzzedDataProvider`` is read by recording its
  consumption schedule against the real provider (positional field->param mapping;
  ``field_to_params`` overrides it);
* a **hand-written** harness that exposes ``decode(bytes) -> {param: value}`` and
  ``encode({param: value}) -> bytes`` is used directly. Nothing is inferred — the
  harness states the mapping — which sidesteps the positional assumption and FDP's
  unencodable field classes entirely.

``codec_for`` picks between them by what the harness module offers.

Everything here needs atheris (via the codec) and the driver; import lazily.
"""

from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tharkun.driver import _run_traced, enclosing_guards  # reuse the driver's tracer
from tharkun.hybrid.codec import codec_for
from tharkun.hybrid.fdp_codec import RecodeRejection
from tharkun.hybrid.ledger import BlockerLedger, LedgerSummary


@dataclass
class HarnessSpec:
    """Points the assist engine at an unmodified OSS harness and what it exercises.

    ``harness_func`` (default ``TestOneInput``) lives in ``harness_path`` and builds an
    ``atheris.FuzzedDataProvider(data)``.  ``target_func`` in ``target_path`` is the code
    whose branches we want to cover; the harness passes the lifted fields to it.
    ``field_to_params`` maps a decoded field list to a ``{param_name: value}`` dict for
    ``target_func`` — the default is positional against the function's signature.
    """

    harness_path: str
    target_path: str
    target_func: str
    corpus_dir: str
    harness_func: str = "TestOneInput"
    field_to_params: Callable[[list[Any]], dict[str, Any]] | None = None
    # Point at the package root to attack EVERY function the harness calls, not just
    # target_func. The field->parameter bindings are recovered from the harness itself
    # (see hybrid.harness_map), which is the fact the positional codec never had.
    package_dir: str | None = None


@dataclass
class Frontier:
    """An uncovered branch edge on the border of coverage."""

    branch_line: int          # the guard line (already covered / reached)
    desired_branch: str       # "then" | "else" — the uncovered side
    body_line: int            # first line of the uncovered side's body (the drive target)
    block_size: int           # static line span of the gated body
    depth: int
    uncovered_gated: int = 0  # statements gated behind this edge NOT yet covered — the
                              # *dynamic* value: it shrinks as coverage grows, so the most
                              # valuable blocker is re-chosen each round.


@dataclass
class AssistResult:
    """What one assist round produced."""

    frontiers_found: int = 0
    seeds_written: list[str] = field(default_factory=list)   # corpus file paths
    solved: list[dict[str, Any]] = field(default_factory=list)  # solved inputs per new seed
    notes: list[str] = field(default_factory=list)
    # Solved values the codec could not write back. Each says whether an encoder is
    # derivable, so a run reports what building the next encoder would actually buy.
    rejections: list[RecodeRejection] = field(default_factory=list)
    # Cumulative worklist snapshot after this round (the ledger persists on the engine, so
    # this is the running total of blockers detected/solved/proved-impossible/exhausted).
    ledger: "LedgerSummary | None" = None

    @property
    def blocked_but_encodable(self) -> list[RecodeRejection]:
        """Rejections that a future encoder could fix (as opposed to hard field limits)."""
        return [r for r in self.rejections if r.mechanically_possible]


# ---------------------------------------------------------------------------
# Frontier detection
# ---------------------------------------------------------------------------


def _load_func(path: str, name: str):
    from tharkun.branch_solver import _load_module_for_execution

    mod, _ = _load_module_for_execution(os.path.realpath(path))
    return getattr(mod, name, None) if mod is not None else None


def _body_span(body: list[ast.stmt]) -> tuple[int, int] | None:
    if not body:
        return None
    lo = body[0].lineno
    hi = max(getattr(s, "end_lineno", s.lineno) or s.lineno for s in body)
    return (lo, hi)


def _guard_lines_of(path: str) -> set[int]:
    """Every ``if`` / ``while`` line in *path* — the points worth observing types at."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return set()
    return {n.lineno for n in ast.walk(tree) if isinstance(n, (ast.If, ast.While))}


def _body_stmt_lines(body: list[ast.stmt]) -> set[int]:
    """Statement line numbers reachable inside *body* (nested statements included)."""
    lines: set[int] = set()
    for stmt in body:
        for node in ast.walk(stmt):
            ln = getattr(node, "lineno", None)
            if ln is not None:
                lines.add(ln)
    return lines


def enclosing_function(module_path: str, line: int) -> str | None:
    """Name of the function *line* sits in, so its callers can be looked up."""
    try:
        tree = ast.parse(Path(module_path).read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return None
    best: tuple[int, str] | None = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(n, "end_lineno", n.lineno) or n.lineno
            if n.lineno <= line <= end and (best is None or n.lineno > best[0]):
                best = (n.lineno, n.name)
    return best[1] if best else None


def guard_names(module_path: str, branch_line: int) -> set[str]:
    """Names read by the guard chain enclosing *branch_line*.

    A solve returns whatever the whole drive settled on, which routinely includes fields
    the guard never reads — the driver is free to perturb anything it was handed. Those
    incidental changes still have to survive re-encoding, and a wide one (ntlm_auth's
    124-byte `exported_session_key`) fails where the guard's own 4-byte flags field would
    not. Knowing what the guard actually reads lets a failed recode retry with just that.
    """
    try:
        tree = ast.parse(Path(module_path).read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return set()
    names: set[str] = set()

    def visit(node: ast.AST, tests: list[ast.expr]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While)):
                inner = [*tests, child.test]
                if child.lineno == branch_line or any(
                    getattr(s, "lineno", -1) == branch_line for s in child.body
                ):
                    for t in inner:
                        names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
                visit(child, inner)
            else:
                visit(child, tests)

    visit(tree, [])
    return names


def length_requirements(module_path: str, branch_line: int,
                        desired: str) -> dict[str, int]:
    """Widths the guard at *branch_line* wants, keyed by the name whose length it reads.

    Reads `len(x) <op> k` out of the guard and turns it into a concrete target width for
    `x`, for the branch we are trying to take. `len(key) != 7` on the "then" side wants
    any width but 7 — 0 is the easiest to arrange and always available, since a starved
    field yields b''. `len(x) > 4` wants 5.

    Slices are followed to their base: the guard reads `len(key)` where the caller passed
    `lm_hash[:7]`, so it is `lm_hash` whose width has to change. A slice can only shorten,
    which is why asking for a SHORT base is the request that can actually be granted.
    """
    try:
        tree = ast.parse(Path(module_path).read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return {}
    node = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.If, ast.While)) and n.lineno == branch_line:
            node = n
            break
    if node is None:
        return {}

    out: dict[str, int] = {}
    for cmp_node in ast.walk(node.test):
        if not isinstance(cmp_node, ast.Compare) or len(cmp_node.ops) != 1:
            continue
        call = cmp_node.left
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "len" and call.args):
            continue
        base = call.args[0]
        while isinstance(base, (ast.Subscript, ast.Attribute)):
            base = base.value
        if not isinstance(base, ast.Name):
            continue
        k, ok = (cmp_node.comparators[0].value, True) if isinstance(
            cmp_node.comparators[0], ast.Constant) else (None, False)
        if not ok or not isinstance(k, int):
            continue
        op = cmp_node.ops[0]
        negate = desired == "else"
        want: int | None = None
        if isinstance(op, ast.NotEq):
            want = 0 if k != 0 else 1
            if negate:
                want = k
        elif isinstance(op, ast.Eq):
            want = k if not negate else (0 if k != 0 else 1)
        elif isinstance(op, (ast.Gt, ast.GtE)):
            hi = k + 1 if isinstance(op, ast.Gt) else k
            want = hi if not negate else max(hi - 1, 0)
        elif isinstance(op, (ast.Lt, ast.LtE)):
            lo = k - 1 if isinstance(op, ast.Lt) else k
            want = max(lo, 0) if not negate else lo + 1
        if want is not None:
            out[base.id] = want
    return out


def find_frontiers(target_source: str, covered_lines: set[int]) -> list[Frontier]:
    """Uncovered branch edges whose guard line is covered, ranked most-valuable first.

    An edge is a frontier when the guard executed (we reach it) but the body of one side
    never did.  Value is *dynamic*: the count of statements gated behind the uncovered
    side that are not yet covered anywhere — so as coverage grows the ranking re-orders
    itself and the assist always attacks the blocker that would open the most new code.
    Ties break toward shallower guards (cheaper to flip).
    """
    try:
        tree = ast.parse(target_source)
    except SyntaxError:
        return []

    frontiers: list[Frontier] = []

    def visit(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                if child.lineno in covered_lines:
                    for desired, body in (("then", child.body), ("else", child.orelse)):
                        span = _body_span(body)
                        if span is None:
                            continue
                        entered = any(span[0] <= ln <= span[1] for ln in covered_lines)
                        if not entered:
                            gated = _body_stmt_lines(body)
                            frontiers.append(Frontier(
                                branch_line=child.lineno,
                                desired_branch=desired,
                                body_line=span[0],
                                block_size=span[1] - span[0] + 1,
                                depth=depth,
                                uncovered_gated=len(gated - covered_lines),
                            ))
                visit(child, depth + 1)
            else:
                visit(child, depth)

    visit(tree, 0)
    # Most currently-uncovered code first (dynamic value), then shallower, then earlier.
    frontiers.sort(key=lambda f: (-f.uncovered_gated, f.depth, f.branch_line))
    return frontiers


# ---------------------------------------------------------------------------
# The assist engine
# ---------------------------------------------------------------------------


class AssistEngine:
    """Turns a corpus into new, frontier-unblocking corpus seeds."""

    def __init__(self, spec: HarnessSpec, *, max_frontiers: int = 8, max_bases: int = 4, max_rounds: int = 12) -> None:
        self.spec = spec
        self.max_frontiers = max_frontiers
        # One knowledge base for the whole engine, not one per drive: a verdict decided
        # for one frontier is still valid for the next frontier that asks the same
        # question, and across rounds the same guards come up again and again.
        from tharkun.knowledge import KnowledgeBase

        self.facts = KnowledgeBase()
        # The persistent worklist: every blocker detected across the whole run, each moved
        # to SOLVED / IMPOSSIBLE or held PENDING / EXHAUSTED. Records the engine's decisions;
        # does not make them.
        self.ledger = BlockerLedger()
        self._corpus_seen: tuple[int, int] | None = None
        # The BEFORE warmup runs once per engine, not per round: it bumps the epoch for
        # every prior it learns (a prior is new information), and re-running it each round
        # would keep re-opening the exhaustion gate it is meant to coexist with.
        self._warmed = False
        # Seeds already traced into the knowledge base (by content hash), so Atheris's
        # growing corpus is observed once each, not re-traced every round.
        self._observed_seeds: set[int] = set()
        # Guard lines per file, memoized — the observe wire re-reads them each round.
        self._guard_lines: dict[str, set[int]] = {}
        # How many different corpus seeds a single frontier may be attacked from.
        # Seeds put a target's fields in genuinely different regimes (bound vs
        # starved) and which regime a guard needs is a property of the guard.
        self.max_bases = max_bases
        self.max_rounds = max_rounds
        self._target_src = Path(os.path.realpath(spec.target_path)).read_text(
            encoding="utf-8", errors="ignore"
        )
        self._param_names = self._read_param_names()

    def _codec(self, harness_fn):
        """The bytes<->params bridge for this harness (hand-written pair, else FDP)."""
        from tharkun.branch_solver import _load_module_for_execution

        mod, _ = _load_module_for_execution(os.path.realpath(self.spec.harness_path))
        return codec_for(mod, self._param_names, self.spec.field_to_params)

    def _read_param_names(self) -> list[str]:
        tree = ast.parse(self._target_src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == self.spec.target_func:
                a = node.args
                return [p.arg for p in (a.posonlyargs + a.args)]
        return []

    def _fields_to_params(self, fields: list[Any]) -> dict[str, Any]:
        if self.spec.field_to_params is not None:
            return self.spec.field_to_params(fields)
        # Default: positional against target_func's parameters.
        return {name: fields[i] for i, name in enumerate(self._param_names) if i < len(fields)}

    def _read_corpus(self) -> list[bytes]:
        d = Path(self.spec.corpus_dir)
        if not d.is_dir():
            return []
        out = []
        for p in sorted(d.iterdir()):
            if p.is_file():
                try:
                    out.append(p.read_bytes())
                except OSError:
                    pass
        return out

    def _harness_fn(self):
        return _load_func(self.spec.harness_path, self.spec.harness_func)

    def _warm(self, targets) -> None:
        """Record static type priors once before driving any target."""
        if self._warmed:
            return
        self._warmed = True
        from tharkun.warmup import build_prior

        for module_path, func_name in targets:
            build_prior(self.facts, module_path, func_name)

    def _frontier_dead(self, path: str, fr: "Frontier") -> bool:
        """Check for an explicitly certified unreachable body in the shared store."""
        from tharkun.knowledge import program_point

        return self.facts.proved_dead(program_point(path, fr.body_line))

    @staticmethod
    def _scope_of(rp: str, fr: "Frontier", func: str) -> tuple:
        """The worklist / exhaustion key — identical shape in both assist paths."""
        return (rp, fr.branch_line, fr.desired_branch, func)

    @staticmethod
    def _blocker_label(rp: str, fr: "Frontier", func: str) -> str:
        return f"{os.path.basename(rp)} L{fr.branch_line}/{fr.desired_branch} via {func}"

    def _target_coverage(self, params: dict[str, Any]) -> set[int]:
        rp = os.path.realpath(self.spec.target_path)
        # Snapshot lines irrelevant here; we only need executed lines.
        outcome = _run_traced(rp, self.spec.target_func, params, set())
        return outcome.executed if outcome.ok else set()

    def _guards_for(self, path: str) -> set[int]:
        rp = os.path.realpath(path)
        if rp not in self._guard_lines:
            self._guard_lines[rp] = _guard_lines_of(rp)
        return self._guard_lines[rp]

    def _observe_seed_into_kb(self, data: bytes, target_path: str, target_func: str,
                              params: dict[str, Any], watch_files) -> int:
        """Trace one corpus seed and record the RUNTIME TYPES at every guard it reaches.

        This is the Atheris lane feeding the knowledge base. A seed — whether Atheris found
        it or the assist wrote it — is executed under the tracer, and at each guard line the
        real types of the locals in scope are observed into the lattice, keyed by the path
        the seed took. So Atheris's own exploration teaches the KB what actually flows
        through each guard, which then types the assist's search (``make_strategy`` reads
        ``store.at(guard)``). The loop closes: Atheris explores, the KB learns, the assist
        aims, Atheris reloads the seed.

        TYPES ONLY, never the concrete values a seed carried. At an UNCOVERED guard those are
        exactly the values that FAILED it; recorded as a membership fact they would trap the
        search in the failing set. Observed types are always sound and always help — they fix
        the case static gets flatly wrong (``token[:2]`` read as a list where bytes ran).

        Returns the number of program points observed. Idempotent-guarded by the caller on
        seed content, and a no-op (returns 0) on any trace failure — this must never be able
        to break a round.
        """
        from tharkun.knowledge import program_point
        from tharkun.lattice import TOP, path_class

        rp = os.path.realpath(target_path)
        by_file = {os.path.realpath(f): self._guards_for(f) for f in watch_files}
        by_file = {f: ls for f, ls in by_file.items() if ls}
        if not by_file:
            return 0
        own = by_file.get(rp, set())
        extra = {f: ls for f, ls in by_file.items() if f != rp}
        try:
            oc = _run_traced(rp, target_func, params, own, capture_real=True,
                             extra_paths=frozenset(extra),
                             extra_snapshot_lines=extra or None)
        except Exception:
            return 0
        if not oc.ok:
            return 0
        cls = path_class(oc.by_path)
        observed = 0
        for f, snaps in oc.real_by_path.items():
            for line, frames in snaps.items():
                if not frames:
                    continue
                state = TOP
                for name, val in frames[-1].items():
                    if val is not None:
                        state = state.with_types(name, [type(val).__name__])
                if not state.is_top():
                    self.facts.observe(program_point(f, line), cls, state)
                    observed += 1
        return observed

    def _observe_new_seeds(self, seeds, target_path: str, target_func: str,
                           params_of, watch_files) -> None:
        """Observe every corpus seed not yet traced into the KB. Content-hashed, so a seed
        is traced once however many rounds it survives in the corpus.

        ``params_of(seed)`` maps a seed entry to the kwargs its target runs with — different
        per path (single-target decodes to the target's params; the package path maps fields
        to a callee's parameters), so the caller supplies it.
        """
        for s in seeds:
            data = s["data"] if isinstance(s, dict) else s
            h = hash(data)
            if h in self._observed_seeds:
                continue
            self._observed_seeds.add(h)
            try:
                self._observe_seed_into_kb(data, target_path, target_func,
                                           params_of(s), watch_files)
            except Exception:
                pass

    def observe_new_corpus(self, *, max_seeds: int = 12) -> int:
        """Fold newly-appeared corpus seeds into the KB while Atheris runs, between assists.

        The *authoritative* reconciliation is at the top of :meth:`assist_once` — it re-reads
        the corpus, bumps the epoch, and observes every new seed before searching. This is the
        continuous supplement the design calls for: "all the while Atheris works in the
        background, and if it grows the seed queue, we update our facts." As Atheris grows the
        queue between plateaus, keep the observed-types facts warm (so the next assist starts
        from current knowledge instead of rebuilding it in a burst) and bump the epoch (so an
        EXHAUSTED scope re-opens on the background progress, not only at the next plateau).

        Cheap and safe: returns immediately when the corpus is unchanged (byte fingerprint),
        content-hashes via ``_observed_seeds`` so a seed is traced once, bounds work to
        *max_seeds* per call, and never raises — it must not be able to stall or break the
        poll loop. The package path's per-target decode is left to assist time; here only the
        single-target codec is cheap enough to run inline.
        """
        try:
            corpus = self._read_corpus()
        except Exception:
            return 0
        if not corpus:
            return 0
        fingerprint = (len(corpus), sum(len(c) for c in corpus))
        if fingerprint == self._corpus_seen:
            return 0
        self._corpus_seen = fingerprint
        self.facts.bump("corpus grew (background)")
        if self.spec.package_dir:
            return 0
        try:
            harness = self._harness_fn()
            if harness is None:
                return 0
            codec = self._codec(harness)
            fresh = [d for d in corpus if hash(d) not in self._observed_seeds][:max_seeds]
            seeds = [{"data": d, "params": codec.decode(d)} for d in fresh]
            self._observe_new_seeds(seeds, self.spec.target_path, self.spec.target_func,
                                    lambda s: s["params"], [self.spec.target_path])
            return len(seeds)
        except Exception:
            return 0

    def assist_once(self) -> AssistResult:
        """One round: recompute coverage from the corpus, solve top frontiers, write seeds."""
        from tharkun.driver import drive_to_line

        res = AssistResult()
        harness = self._harness_fn()
        if harness is None:
            res.notes.append(f"cannot load harness {self.spec.harness_func} from {self.spec.harness_path}")
            return res

        corpus = self._read_corpus()
        if not corpus:
            res.notes.append("empty corpus")
            return res

        # The corpus changes from OUTSIDE this engine: the whole design runs Atheris in
        # the background writing seeds into the same directory. A new seed is new
        # information — it can put a target's fields in a regime that makes a frontier
        # we already wrote off solvable — so it must re-open every exhausted scope.
        # Without this the assist would go permanently quiet in exactly the mode it
        # exists for, while Atheris kept feeding it fresh material.
        fingerprint = (len(corpus), sum(len(c) for c in corpus))
        if fingerprint != self._corpus_seen:
            self._corpus_seen = fingerprint
            self.facts.bump("corpus changed")

        if self.spec.package_dir:
            return self._assist_across_package(res, harness, corpus)

        # Decode each seed to fields; accumulate target coverage; remember a base seed
        # per covered guard so we can recode a solution from bytes that already reach it.
        covered: set[int] = set()
        seeds: list[dict[str, Any]] = []
        codec = self._codec(harness)
        for data in corpus:
            params = codec.decode(data)
            cov = self._target_coverage(params)
            covered |= cov
            seeds.append({"data": data, "params": params, "cov": cov})

        self._warm([(self.spec.target_path, self.spec.target_func)])
        # Feed every new corpus seed's runtime types at each guard into the KB (the Atheris
        # lane). Observed types outrank static guesses and type the assist's own search.
        self._observe_new_seeds(seeds, self.spec.target_path, self.spec.target_func,
                                lambda s: s["params"], [self.spec.target_path])
        rp_target = os.path.realpath(self.spec.target_path)
        # Retire any open blocker whose body another tactic (Atheris, an incidental reach)
        # has covered since we last looked — before spending a sound decider or static on it.
        self.ledger.reconcile_covered({rp_target: covered})
        frontiers: list = []
        dead = 0
        for f in find_frontiers(self._target_src, covered):
            scope = self._scope_of(rp_target, f, self.spec.target_func)
            label = self._blocker_label(rp_target, f, self.spec.target_func)
            if self._frontier_dead(self.spec.target_path, f):
                self.ledger.mark_impossible(scope, label=label)   # a sound decider certificate
                dead += 1
            else:
                self.ledger.note_detected(scope, label=label, value=f.uncovered_gated,
                                          body_line=f.body_line)
                frontiers.append(f)
        res.frontiers_found = len(frontiers)
        if dead:
            res.notes.append(f"pruned {dead} proved-dead frontier(s) before solving")

        # The widest corpus seed is the best base: drive_to_line solves the whole
        # enclosing guard chain from it, and wide fields leave room to re-encode the
        # solution (a short seed can leave the fields the solution touches too narrow).
        # Widen it further if the harness's own fields are still starving (see
        # widen_seed) — an unwritable zero-width field is a seed-size problem.
        widest = max(seeds, key=lambda s: len(s["data"]))
        fn = getattr(harness, "TestOneInput", None) or getattr(harness, "TestInput", None)
        if fn is not None:
            from tharkun.hybrid.fdp_codec import widen_seed

            grown = widen_seed(fn, widest["data"])
            if grown != widest["data"]:
                # Added, never substituted: a wider seed binds later fields but starves
                # earlier ones, so neither dominates (see _assist_across_package).
                gp = codec.decode(grown)
                seeds.append({"data": grown, "params": gp,
                              "cov": self._target_coverage(gp)})
                widest = max(seeds, key=lambda s: len(s["params"]))

        written: set[bytes] = set()
        for fr in frontiers[: self.max_frontiers]:
            scope = self._scope_of(rp_target, fr, self.spec.target_func)
            reaching = [s for s in seeds if fr.branch_line in s["cov"]]
            # Prefer a guard-reaching seed only if it is at least as wide as the widest.
            base = max(reaching, key=lambda s: len(s["data"]), default=None)
            if base is None or len(base["data"]) < len(widest["data"]):
                base = widest
            # Thread the persistent store, so decisions memoize across frontiers and rounds
            # (the driver's own guard-level memo makes a repeat cheap). Deliberately NOT
            # frontier-level exhaustion-gated: unlike the package path, this path keeps
            # re-attempting a resisted frontier each round, so a stochastic-only-solvable
            # guard is not frozen out on a true plateau (where nothing bumps the epoch). The
            # ledger still RECORDS the resistance as EXHAUSTED; it just does not skip.
            drive = drive_to_line(
                self.spec.target_path, self.spec.target_func, fr.body_line,
                dict(base["params"]), max_rounds=self.max_rounds,
                param_bounds=codec.bounds(base["data"]), store=self.facts,
            )
            if not drive.reached:
                # The portfolio resisted this round — record it in the worklist (retriable
                # if a stochastic search was part of the failure). Recorded, not gated.
                self.ledger.mark_exhausted(
                    scope, retriable=drive.used_stochastic,
                    label=self._blocker_label(rp_target, fr, self.spec.target_func))
                res.notes.append(f"frontier L{fr.branch_line}/{fr.desired_branch}: {drive.status}")
                continue
            rejected: list[RecodeRejection] = []
            changed = {k: v for k, v in drive.inputs.items()
                       if k in base["params"] and base["params"][k] != v}
            if not changed:
                res.notes.append(f"frontier L{fr.branch_line}: solution changed no decodable field")
                continue
            # Pass the COMPLETE field set, not just the changes: a hand-written
            # encode() builds the whole buffer and needs every field. FdpCodec diffs
            # against the seed's current values internally, so this is right for both.
            merged = {**base["params"], **changed}
            if hasattr(codec, "encode_with_rejections"):
                new_bytes = codec.encode_with_rejections(base["data"], merged, rejected)
            else:
                new_bytes = codec.encode(base["data"], merged)
            if new_bytes is None or new_bytes in written or new_bytes == base["data"]:
                res.rejections.extend(rejected)
                if rejected:
                    r = rejected[0]
                    fixable = "encoder derivable" if r.mechanically_possible else "hard field limit"
                    res.notes.append(
                        f"frontier L{fr.branch_line}: solved but not recodable — "
                        f"{r.method}: {r.reason} ({fixable})"
                    )
                else:
                    res.notes.append(f"frontier L{fr.branch_line}: recoded seed was a duplicate")
                continue
            written.add(new_bytes)
            path = self._write_seed(new_bytes)
            res.seeds_written.append(path)
            res.solved.append(dict(drive.inputs))
            self.ledger.mark_solved(
                scope, seed_hash=hashlib.sha1(new_bytes).hexdigest()[:12],
                inputs=drive.inputs,
                label=self._blocker_label(rp_target, fr, self.spec.target_func))
        res.ledger = self.ledger.summary()
        return res

    def _assist_across_package(self, res: "AssistResult", harness, corpus) -> "AssistResult":
        """Attack frontiers in every function the harness calls, best-value first.

        Same loop as the single-target path, but the frontier search runs over each
        callee's own module and each drive uses that callee's parameters — mapped from
        the fuzzer fields by :func:`~tharkun.hybrid.harness_map.map_harness_calls`.
        """
        from tharkun.driver import drive_to_line
        from tharkun.hybrid.fdp_codec import (
            encode_overrides, field_bounds, record_schedule, seed_with_field_width,
            widen_seed,
        )
        from tharkun.hybrid.harness_map import call_site_constants, map_harness_calls

        targets = map_harness_calls(self.spec.harness_path, self.spec.package_dir)
        if not targets:
            res.notes.append("no field->parameter bindings recovered from the harness")
            return res

        # Record static type hints for each function the harness calls.
        self._warm([(t.module_path, t.func) for t in targets])

        seeds = []
        for data in corpus:
            sched = record_schedule(harness, data)
            seeds.append({"data": data, "schedule": sched,
                          "fields": [r.value for r in sched]})
        # A seed too small for the harness starves its later fields to zero width, and a
        # zero-width field cannot be written to at all — on ntlm_auth's 64-byte seed that
        # is 17 of 21 fields, i.e. most targets unaddressable for reasons that have
        # nothing to do with solving. So ADD a widened variant rather than switching to
        # it: measured, replacing the base traded compute_hash's and des's lines for
        # compute_keys's, because the wide seed starves the *early* fields the narrow one
        # bound. Neither seed dominates; each target picks the one that binds its fields.
        widest = max(seeds, key=lambda s: len(s["data"]))
        grown = widen_seed(harness, widest["data"])
        if grown != widest["data"]:
            sched = record_schedule(harness, grown)
            seeds.append({"data": grown, "schedule": sched,
                          "fields": [r.value for r in sched]})

        def _bases_for(tgt) -> list[dict]:
            """Every seed worth trying for this target, best first.

            A field consumed to zero width decodes to '' / b'' / 0 and cannot be written
            back, so a seed that starves this target's fields is useless for it however
            wide it is overall. Among the seeds that do bind them, prefer the *smallest*:
            re-encoding one field shifts everything after it, and a wide seed has more
            after it to disturb — measured, preferring width lost compute_hash's lines to
            re-decode interference that the 64-byte seed did not have.

            But committing to the single best base is wrong, because "binds the most
            fields" is exactly backwards for a guard that needs a field to be SHORT or
            ABSENT. ntlm_auth's `if len(key) != 7` on `lm_hash[:7]` is satisfied for free
            on the narrow seed, where lm_hash is starved to b'', and unsatisfiable on the
            wide one, where it is 124 bytes — three reachable des.py lines sat behind that
            choice. So rank, but hand back the whole list: a frontier gets to try the
            other bases before it is written off as a plateau.
            """
            def bound(s: dict) -> int:
                return sum(1 for i in tgt.field_to_param
                           if i < len(s["schedule"]) and s["schedule"][i].nbytes > 0)

            usable = [s for s in seeds if bound(s) > 0]
            return sorted(usable, key=lambda s: (-bound(s), len(s["data"])))

        # Every module in the package is watched, not just the ones the harness names:
        # what the harness reaches THROUGH a call is in the coverage report too, and is
        # therefore fair to target (des.py is 140 statements reached only via
        # compute_keys). The entry function stays one the harness itself calls, so the
        # inputs are still only harness fields.
        watch = frozenset(str(q) for q in Path(self.spec.package_dir).rglob("*.py"))

        # Per target: what do the files it reaches still not cover, given the corpus?
        plan = []
        union_covered: dict[str, set[int]] = {}   # realpath -> covered lines, across targets
        for tgt in targets:
            covered_by_file: dict[str, set[int]] = {}
            for s in seeds:
                params = {**tgt.fixed_args,
                          **{p: s["fields"][i] for i, p in tgt.field_to_param.items()
                             if i < len(s["fields"])}}
                # The Atheris lane, per callee: a seed reaches different guards through
                # different targets, so observe once per (target, seed). Types the KB
                # learns here aim the assist's search at those callee guards.
                okey = (tgt.func, hash(s["data"]))
                if okey not in self._observed_seeds:
                    self._observed_seeds.add(okey)
                    try:
                        self._observe_seed_into_kb(s["data"], tgt.module_path, tgt.func,
                                                   params, watch)
                    except Exception:
                        pass
                for path, lines in self._coverage_by_file(
                        tgt.module_path, tgt.func, params, watch).items():
                    covered_by_file.setdefault(path, set()).update(lines)
            for path, covered in covered_by_file.items():
                union_covered.setdefault(os.path.realpath(path), set()).update(covered)
                try:
                    src = Path(path).read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for fr in find_frontiers(src, covered):
                    plan.append((fr, tgt, os.path.realpath(path)))
        dead = 0
        covered_elsewhere = 0
        alive = []
        for fr, t, fpath in plan:
            rp = os.path.realpath(fpath)
            scope = self._scope_of(rp, fr, t.func)
            label = self._blocker_label(rp, fr, t.func)
            if self._frontier_dead(fpath, fr):
                self.ledger.mark_impossible(scope, label=label)   # a sound decider certificate
                dead += 1
                continue
            self.ledger.note_detected(scope, label=label, value=fr.uncovered_gated,
                                      body_line=fr.body_line)
            # find_frontiers ran per-target, so a branch can be an uncovered frontier via
            # THIS target yet already covered via ANOTHER (union_covered). Do not drive one
            # whose body is already covered — reconcile (below) retires it off the worklist.
            if fr.body_line and fr.body_line in union_covered.get(rp, ()):
                covered_elsewhere += 1
                continue
            alive.append((fr, t, fpath))
        # Retire open blockers whose body another tactic (or another target) has covered —
        # AFTER note_detected, so its final word is RETIRED, not a re-opened PENDING.
        self.ledger.reconcile_covered(union_covered)
        plan = alive
        plan.sort(key=lambda pr: (-pr[0].uncovered_gated, pr[0].depth))
        res.frontiers_found = len(plan)
        if dead:
            res.notes.append(f"pruned {dead} proved-dead frontier(s) before solving")
        if covered_elsewhere:
            res.notes.append(f"retired {covered_elsewhere} frontier(s) already covered elsewhere")

        written: set[bytes] = set()
        for fr, tgt, fpath in plan[: self.max_frontiers]:
            # Nothing has been learned since this exact frontier last resisted the whole
            # portfolio, so re-running it cannot end differently. Epoch-gated: any new
            # decision, fact, or seed re-opens it.
            scope = (os.path.realpath(fpath), fr.branch_line, fr.desired_branch, tgt.func)
            if self.facts.is_exhausted(scope):
                self.facts.skipped["frontier"] += 1
                continue
            solved_here = False
            last_note = ""
            drive_was_stochastic = False
            # Try each base until one yields a seed. Different seeds put a target's
            # fields in genuinely different regimes (bound vs starved), and which regime
            # a guard needs is a property of the guard, not of the seed we happened to
            # rank first.
            # A length guard is decided by a field's WIDTH, which is not a value any
            # flipper can propose — it follows from how much buffer is left when the
            # harness reaches that field. So when the guard asks for a width, go and
            # build a base that has it, with the real provider as the oracle. Which
            # field the guard's local came from needs call_mapper to say, so each of
            # the target's fields is tried at the requested width; the drive and the
            # recode decide which one was right.
            width_bases: list[dict] = []
            wants = length_requirements(fpath, fr.branch_line, fr.desired_branch)
            if wants:
                want = min(wants.values())
                # Deduped only against each OTHER. Excluding sizes already in the corpus
                # would drop exactly the seed this is for: the narrow seed is in the
                # corpus, it is simply ranked out by "binds the most fields", and putting
                # it back at the front is the whole point.
                seen_sizes: set[int] = set()
                # Any seed will do as the byte pattern: what matters is the WIDTH this
                # produces, and the driver derives its way out of whatever values the
                # base happens to commit (a flags field already set to another arm of an
                # if/elif chain is cleared by the else-polarity bitmask solve). The
                # smallest is used only because the search starts small.
                pristine = min(seeds, key=lambda s2: len(s2["data"]))["data"]
                for idx in sorted(tgt.field_to_param):
                    built = seed_with_field_width(harness, pristine, idx, want)
                    if built is None or len(built) in seen_sizes:
                        continue
                    seen_sizes.add(len(built))
                    sched2 = record_schedule(harness, built)
                    width_bases.append({"data": built, "schedule": sched2,
                                        "fields": [r.value for r in sched2]})

            for base in (width_bases + _bases_for(tgt))[: self.max_bases + len(width_bases)]:
                # Fixed args first so a fuzzer field always wins the same name.
                params = {**tgt.fixed_args,
                          **{p: base["fields"][i] for i, p in tgt.field_to_param.items()
                             if i < len(base["fields"])}}
                if not params:
                    continue
                bounds = {p: b for i, p in tgt.field_to_param.items()
                          if i < len(base["schedule"])
                          for b in [field_bounds(base["schedule"][i])] if b}
                # What the CALLERS pin. A frontier that needs a value no call site ever
                # supplies is dead in context however satisfiable its guard looks alone,
                # and this is what lets the decider say so.
                owner = enclosing_function(fpath, fr.branch_line)
                ctx = (call_site_constants(self.spec.package_dir, owner) if owner else {})
                drive = drive_to_line(tgt.module_path, tgt.func, fr.body_line, dict(params),
                                      target_path=fpath, max_rounds=self.max_rounds,
                                      param_bounds=bounds, context_facts=ctx,
                                      store=self.facts)
                drive_was_stochastic |= drive.used_stochastic
                if not drive.reached:
                    last_note = (f"{tgt.func} -> {os.path.basename(fpath)}"
                                 f" L{fr.branch_line}: {drive.status}")
                    continue
                overrides = {i: drive.inputs[p] for i, p in tgt.field_to_param.items()
                             if p in drive.inputs and i < len(base["fields"])
                             and drive.inputs[p] != base["fields"][i]}
                if not overrides:
                    continue

                def _reaches(candidate: bytes, _tgt=tgt, _fpath=fpath, _fr=fr) -> bool:
                    # Execution is the judge for a recode whose mid-buffer field decoded to the
                    # solved value PLUS trailing bytes: does the reassembled seed actually run
                    # the frontier body? (A "fd://" solved into field 3 reads back as
                    # "fd://<leftover>", which still satisfies startswith — so the seed works.)
                    try:
                        sched_c = record_schedule(harness, candidate)
                        fields_c = [r.value for r in sched_c]
                        params_c = {**_tgt.fixed_args,
                                    **{p: fields_c[i] for i, p in _tgt.field_to_param.items()
                                       if i < len(fields_c)}}
                        oc = _run_traced(os.path.realpath(_tgt.module_path), _tgt.func,
                                         params_c, set(),
                                         extra_paths=frozenset({os.path.realpath(_fpath)}))
                        return oc.ok and _fr.body_line in oc.by_path.get(
                            os.path.realpath(_fpath), set())
                    except Exception:
                        return False

                rejected: list[RecodeRejection] = []
                new_bytes = encode_overrides(base["data"], base["schedule"], overrides,
                                             rejected, verify_only_targeted=True,
                                             accept=_reaches)
                if new_bytes is None and len(overrides) > 1:
                    # Retry with only the fields the guard reads. The rest were changed
                    # incidentally by the drive, and every one of them still has to
                    # survive re-encoding — so a wide field the guard never looks at can
                    # sink a solution that is otherwise perfectly encodable.
                    reads = guard_names(fpath, fr.branch_line)
                    minimal = {i: v for i, v in overrides.items()
                               if tgt.field_to_param.get(i) in reads}
                    if minimal and len(minimal) < len(overrides):
                        retry: list[RecodeRejection] = []
                        narrowed = encode_overrides(base["data"], base["schedule"], minimal,
                                                    retry, verify_only_targeted=True,
                                                    accept=_reaches)
                        if narrowed is not None:
                            new_bytes, rejected = narrowed, retry
                if new_bytes is None or new_bytes in written or new_bytes == base["data"]:
                    res.rejections.extend(rejected)
                    if rejected:
                        last_note = (f"{tgt.func} L{fr.branch_line}: "
                                     f"{rejected[0].reason[:70]}")
                    continue
                written.add(new_bytes)
                res.seeds_written.append(self._write_seed(new_bytes))
                res.solved.append({"target": tgt.func, **drive.inputs})
                self.ledger.mark_solved(
                    scope, seed_hash=hashlib.sha1(new_bytes).hexdigest()[:12],
                    inputs={"target": tgt.func, **drive.inputs},
                    label=self._blocker_label(os.path.realpath(fpath), fr, tgt.func))
                # A new seed widens what is reachable, so everything written off before
                # is worth attempting again.
                self.facts.bump("new seed")
                solved_here = True
                break
            if not solved_here:
                # A search was part of the failure, so it is owed retries before being
                # written off — a failed search records nothing, so the epoch cannot
                # distinguish "tried once and unlucky" from "provably nothing there".
                self.facts.mark_exhausted(scope, retriable=drive_was_stochastic)
                self.ledger.mark_exhausted(
                    scope, retriable=drive_was_stochastic,
                    label=self._blocker_label(os.path.realpath(fpath), fr, tgt.func))
                if last_note:
                    res.notes.append(last_note)
        res.ledger = self.ledger.summary()
        return res

    def _coverage_of(self, module_path: str, func: str, params: dict,
                     watch: frozenset[str] = frozenset()) -> set[int]:
        outcome = _run_traced(os.path.realpath(module_path), func, params, set(),
                              extra_paths=watch)
        return outcome.executed if outcome.ok else set()

    def _coverage_by_file(self, module_path: str, func: str, params: dict,
                          watch: frozenset[str]) -> dict[str, set[int]]:
        """Lines this call executes, per file — including files it merely calls into.

        A line in a transitive callee is in the coverage report exactly like one in the
        entry module, so its frontiers are just as legitimate a target. They are only
        invisible while the tracer watches a single file.
        """
        outcome = _run_traced(os.path.realpath(module_path), func, params, set(),
                              extra_paths=watch)
        return outcome.by_path if outcome.ok else {}

    def _param_bounds(self, schedule) -> dict[str, tuple[int, int]]:
        """Per-target-parameter integer bounds, from each field's FDP method.

        So the solver returns a value inside the field's domain — an unbounded
        solution would not fit back into the fixed-width field on re-encode.
        """
        out: dict[str, tuple[int, int]] = {}
        for i, name in enumerate(self._param_names):
            if i < len(schedule):
                b = field_bounds(schedule[i])
                if b is not None:
                    out[name] = b
        return out

    def _params_to_overrides(self, base: dict[str, Any], solved: dict[str, Any]) -> dict[int, Any] | None:
        """Map changed target params back to schedule field indices (positional)."""
        overrides: dict[int, Any] = {}
        for i, name in enumerate(self._param_names):
            if name in solved and i < len(base["fields"]) and solved[name] != base["fields"][i]:
                overrides[i] = solved[name]
        return overrides or None

    def _write_seed(self, data: bytes) -> str:
        d = Path(self.spec.corpus_dir)
        d.mkdir(parents=True, exist_ok=True)
        # Content-addressed name, libFuzzer's convention. It must digest the WHOLE
        # buffer: naming a seed by its first 20 bytes silently overwrote every solution
        # that differed only later on, and FDP consumes ints from the END — so on
        # ntlm_auth all three negotiate_flags solutions collided on one filename and only
        # the last survived. sha1 over the full content is deterministic (unlike hash(),
        # which is salted per process) and collides only on identical bytes.
        name = "tharkun-" + hashlib.sha1(data).hexdigest() if data else "tharkun-empty"
        path = d / name
        path.write_bytes(data)
        return str(path)
