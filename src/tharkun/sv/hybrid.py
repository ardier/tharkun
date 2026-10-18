"""Coverage search for the negation of each open SV safety obligation."""
from __future__ import annotations

import ast
import json
import os
import tempfile
import time
from pathlib import Path

from tharkun.hybrid.backend import AtherisBackend
from tharkun.hybrid.fdp_codec import atheris_available
from tharkun.hybrid.fuzzer import should_assist
from tharkun.solve.environment import subprocess_environment
from tharkun.sv import reach as _reach
from tharkun.sv.stages import Candidate

WORKER = str(Path(__file__).with_name("_hybrid_worker.py"))


def _violation(obligation):
    """Reduce a numeric nonzero obligation to its distance-zero target."""
    value = obligation.invariant.value
    if obligation.exception == "ZeroDivisionError":
        for node in ast.walk(ast.parse(value, mode="eval")):
            if (isinstance(node, ast.Compare) and len(node.ops) == 1
                    and isinstance(node.ops[0], ast.NotEq)
                    and isinstance(node.left, ast.Name)
                    and isinstance(node.comparators[0], ast.Constant)
                    and node.comparators[0].value == 0):
                return f"{node.left.id} == 0"
    return f"not ({value})"


def _assert_fallback(source, program, classes):
    """A target for Atheris when ``obligation_census`` has nothing ([internal ref]).

    ``obligation_census`` is sound only as an ALLOWLIST over the WHOLE program
    (``solve/census.py``): one reflective construct ANYWHERE -- reachable or not, and
    however far from the actual violation -- refuses the entire census for a class, not
    just the site near it. A product task's own unreachable-by-construction error path
    (``type(x).__name__`` in a validator's ``__init__``, say) is enough to leave
    ``search`` with zero obligations and no reason to ever start, even though the task's
    own top-level ``assert`` is sitting right there in the same file.

    This reads every ``ast.Assert`` directly, no resolver involved, and gives each one
    the same conservative ``value="False"`` ``solve/census.py::_safety_predicate`` itself
    falls back to for an assert it cannot turn into a checked predicate (an equality of
    two calls, here, never a supported predicate either way) -- "no known safety
    condition", not a claim. It is a HINT for where to point the fuzzer, the same as any
    census-derived obligation already is: only the engine's fresh-child replay of a found
    vector can ever call the property FALSE (docs/SV.md "3a"), so this never bears on
    soundness, only on whether the search starts at all.
    """
    if "AssertionError" not in classes:
        return []
    from tharkun.solve.obligations import Invariant, Obligation
    import hashlib
    try:
        tree = ast.parse(source, program)
    except (SyntaxError, ValueError):
        return []
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return [
        Obligation(sha, f"$fallback:{node.lineno}:{node.col_offset}", "AssertionError",
                  Invariant(program, node.lineno, node.col_offset + 1, "<module>", "False"),
                  "census refused the whole program (#1104): taken from the AST directly")
        for node in ast.walk(tree) if isinstance(node, ast.Assert)
    ]


def _regex_patterns(tree):
    """Regex pattern literals the program compiles or calls ``re.*`` with directly --
    the first argument of a ``.compile``/``.match``/``.fullmatch``/``.search``/``.sub``/
    ``.subn``/``.split``/``.findall``/``.finditer`` call, when it is a string constant.
    Good enough for ``re.compile(PATTERN, flags)`` and ``re.sub(PATTERN, ..., s)``
    directly; a pattern built from a variable is missed (conservative -- misses only
    cost a search opportunity, never a wrong verdict)."""
    names = {"compile", "match", "fullmatch", "search", "sub", "subn",
            "split", "findall", "finditer"}
    seen, out = set(), []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in names and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                and node.args[0].value not in seen):
            seen.add(node.args[0].value)
            out.append(node.args[0].value)
    return out


#: how many of the program's patterns propose a regex-positive vector each ([internal ref] had 4; the patterns of the
#: task-local code it reaches ([internal ref]) come after the program's own)
REGEX_VECTOR_PATTERNS = 8


def _reached(ctx, tree):
    """The task-local modules the program imports (``sv/reach.py``; none for a single-file task).
    Never executes anything, and never raises: a hint is optional."""
    try:
        return _reach.reached(ctx.program, getattr(ctx, "task", None), tree=tree)
    except Exception:  # noqa: BLE001
        return []


class _Plan:
    """What the program reaches and calls (``sv/reach.py``), computed once per stage call."""

    def __init__(self, ctx, tree):
        self.tree = tree
        self.modules = _reached(ctx, tree)
        self.fold = _reach.folders(self.modules, tree) if tree is not None else {}
        try:
            self.units = _reach.called(tree, self.modules) if tree is not None and self.modules else []
        except Exception:  # noqa: BLE001 -- a hint is optional
            self.units = []
        self.tokens = []                 # the guard tokens (set by ``_reach_plan``)
        names = _reach.unit_modules(self.units)
        #: the modules whose code the program calls, in discovery order
        self.called = [m for m in self.modules if m.name in set(names)]

    def patterns(self):
        """``(pattern, flags)`` for the program's own regexes (direct constant arguments, as before), then
        those it builds from module-level names / ``format`` / f-strings, then those of the code it calls."""
        seen, out = set(), []

        def add(pattern, flags):
            if pattern and (pattern, flags) not in seen:
                seen.add((pattern, flags))
                out.append((pattern, flags))

        if self.tree is not None:
            for pattern in _regex_patterns(self.tree):
                add(pattern, 0)
            for pattern, flags in _reach.regex_patterns(self.tree):
                add(pattern, flags)
        for pattern, flags in _reach.unit_patterns([u for u in self.units if u.module], self.fold):
            add(pattern, flags)
        return out


def _patterns_with_flags(ctx, tree):
    return _Plan(ctx, tree).patterns()


def _regex_vectors(ctx, tree):
    """Whole draw vectors with a generated match of one of the program's own regex
    patterns substituted in at each ``str`` draw position ([internal ref], owner idea): an HTML
    tag, a digit run -- the shapes these obligations need, and shapes
    ``sv/string_regex.py::regex_constraint``'s Z3 translation explicitly refuses for any
    pattern using ``IGNORECASE``/``MULTILINE``/``DOTALL`` (exactly the flags
    python-string-utils' own HTML patterns use), so the symbolic route can never reach
    them either.

    Reuses ``solve/condition.py::_generate_from_regex`` (branch_solver's own step 7
    regex inversion) rather than calling Hypothesis directly: it already tries
    ``re.ASCII`` first and falls back to the full Unicode pattern only if that draws
    nothing (``\\d``/``\\w``/``\\s`` are Unicode-aware, so a plain draw can return
    Devanagari digits -- a legal match but a poor witness for an ASCII-digit obligation
    like the ISBN checks), and draws under ``fullmatch=True`` so one witness per pattern
    serves ``match``/``search``/``fullmatch`` alike. Sounder and smaller than a second,
    parallel Hypothesis call here.

    ``ctx.first.draws`` gives the real draw SHAPE (kinds and count); only the ``str``
    positions are replaced, one at a time, so every other draw stays at its observed
    value -- the same substitution ``sv/engine.py::verify``'s own enumeration does with
    its small-first candidates. A proposal here is exactly as trusted as any other: the
    engine's fresh replay is the only thing that can call it FALSE."""
    from tharkun.solve.condition import _generate_from_regex
    base = [{k: v for k, v in d.items() if k != "site"} for d in ctx.first.draws]
    str_positions = [i for i, d in enumerate(base) if d.get("kind") == "str"]
    if not str_positions:
        return
    patterns = _patterns_with_flags(ctx, tree)
    if not patterns:
        return
    # Which regex the program takes (HTML_RE vs. HTML_TAG_ONLY_RE, say) is routinely
    # gated by a sibling bool draw (``keep_tag_content``); the default (unforced) run
    # always draws False (``_child.py``), so leaving bool draws at their observed value
    # would try every regex-matching string only against whichever branch the FIRST run
    # happened to take, never the other one -- [internal ref]. Few enough bool positions in these
    # product tasks (1-2) that every combination is cheap; capped so this never grows
    # unboundedly on a program with many.
    bool_positions = [i for i, d in enumerate(base) if d.get("kind") == "bool"][:4]
    bool_settings = [{}]
    for i in bool_positions:
        bool_settings = [dict(s, **{str(i): v}) for s in bool_settings for v in (False, True)]
    for pattern, flags in patterns[:REGEX_VECTOR_PATTERNS]:
        if not pattern:
            continue
        try:
            value = _generate_from_regex(pattern, flags=flags)
        except Exception:  # noqa: BLE001 -- an unusable pattern proposes nothing
            continue
        if value is None or not isinstance(value, str):
            continue
        for i in str_positions:
            for settings in bool_settings:
                vector = [dict(d) for d in base]
                vector[i] = dict(vector[i], value=value)
                for j, v in settings.items():
                    vector[int(j)] = dict(vector[int(j)], value=v)
                yield Candidate(vector, "hybrid:regex", {"pattern": pattern[:80]})


def _literal_seeds(tree, limit=32):
    """The program's own string literals, as Atheris corpus entries ([internal ref]), including
    regex patterns -- ``sv/engine.py::harvest_constants`` already harvests literals for
    the enumeration search, but caps each at 64 characters (a deliberate limit there: a
    long literal is expensive to try verbatim as a draw VALUE in the small-first search).
    A regex pattern -- exactly the kind of literal an HTML/markup obligation needs -- is
    routinely longer, so this reads strings straight from the AST instead, with a much
    looser cap (a seed FILE, unlike a candidate value, costs nothing extra to be long: it
    only shapes what libFuzzer's coverage-guided mutation starts from). Corpus seeding
    never decides anything, so this is sound by the same rule as any other candidate
    hint."""
    seen, out = set(), []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value and len(node.value) <= 512 and node.value not in seen):
            seen.add(node.value)
            out.append(node.value)
    return out[:limit]


def _obligation_tables(obligations):
    """The worker's per-line ``checks`` (predicates a run violates at that line), its distance
    ``metrics`` (numeric seeds worth keeping), and the concolic handoff's ``records``."""
    checks, metrics, records = {}, {}, []
    for o in obligations:
        line = o.invariant.line
        checks.setdefault(line, []).append(o.invariant.value)
        residual = _violation(o)
        if o.exception == "ZeroDivisionError":
            comparison = ast.parse(residual, mode="eval").body
            if isinstance(comparison, ast.Compare) and isinstance(comparison.left, ast.Name):
                metrics[line] = comparison.left.id
        records.append({"line": line, "exc": o.exception, "kind": "obligation",
                        "status": "may_raise", "residual": residual})
    return checks, metrics, records


#: regex-positive samples: DRAWN per pattern, KEPT per pattern (the shortest: a short accepted string is the one
#: that still reaches a branch after its parts are wrapped or cut), and the wall clock the drawing may take in all
REGEX_SAMPLES_DRAWN = 8
REGEX_SAMPLES_KEPT = 3
REGEX_SAMPLE_SECONDS = 3.0
#: composed seeds: the longest guard token spliced into samples, and how many composed seeds the corpus gets
COMPOSE_TOKEN_MAX = 4
COMPOSE_SEEDS = 128


def _regex_samples(patterns):
    """A few strings per pattern that the pattern accepts (the program's own regexes and those of the code it
    reaches): corpus entries that start the fuzzer INSIDE the shapes the code checks for instead of
    waiting for it to find one.  Printable text only (what a ``nondet_str`` draw can hold).  Bounded by
    ``REGEX_SAMPLE_SECONDS`` of wall clock; an unusable pattern contributes nothing."""
    try:
        from tharkun.solve.condition import _generate_from_regex
    except Exception:  # noqa: BLE001
        return []
    deadline = time.monotonic() + REGEX_SAMPLE_SECONDS
    out = []
    for pattern, flags in patterns:
        drawn = set()
        for _ in range(REGEX_SAMPLES_DRAWN):
            if time.monotonic() > deadline:
                break
            try:
                value = _generate_from_regex(pattern, flags=flags)
            except Exception:  # noqa: BLE001
                break
            if isinstance(value, str) and value and len(value) <= 128 and all(c in _reach._PRINTABLE for c in value):
                drawn.add(value)
        out += sorted(drawn, key=lambda v: (len(v), v))[:REGEX_SAMPLES_KEPT]
    return out


def _composed_seeds(samples, tokens, limit=COMPOSE_SEEDS):
    """Samples of the code's own regexes with the tokens its GUARDS compare strings with spliced in.

    A regex sample is valid for the pattern but misses every guard (``a@bc.de`` is not quoted); a guard
    token alone satisfies one guard and breaks the pattern.  A kill that needs both -- a quoted local part
    in front of a valid domain -- is a few lucky mutations away for a fuzzer that starts from either, and
    zero mutations from a corpus entry that has both.  For each sample and each short token ``t``: ``t``
    wrapped round the leading segment up to each separator (``"a"@bc.de``), round the whole sample, and
    ``t`` inserted at every boundary (start, end, either side of a non-alphanumeric).  Wraps come first
    (they satisfy two guards at once), round-robin over the samples.  Corpus entries only: a hint."""
    toks = [t for t in tokens if 0 < len(t) <= COMPOSE_TOKEN_MAX]
    per_sample = []
    for smp in samples:
        seps = [i for i, ch in enumerate(smp) if not ch.isalnum()]
        cuts = sorted({0, len(smp), *seps, *(i + 1 for i in seps)})
        wraps, inserts = [], []
        for t in toks:
            wraps += [t + smp[:i] + t + smp[i:] for i in seps]
            wraps.append(t + smp + t)
            inserts += [smp[:i] + t + smp[i:] for i in cuts]
        per_sample.append(wraps + inserts)
    out, seen = [], set(samples)
    for layer in range(max((len(x) for x in per_sample), default=0)):
        for seeds in per_sample:
            if layer < len(seeds) and seeds[layer] not in seen and len(seeds[layer]) <= 256:
                seen.add(seeds[layer])
                out.append(seeds[layer])
                if len(out) >= limit:
                    return out
    return out


def _seed_corpus(corpus, tree, plan=None):
    """The zero input, then one file each for the string literals of the program and of the task code it
    calls ([internal ref]: a product's library is where its guards compare the draw with something), and a few
    strings its regexes accept."""
    (corpus / "zero").write_bytes(bytes(64))
    seeds = list(_literal_seeds(tree)) if tree is not None else []
    if plan is not None:
        for u in plan.units:
            if u.module:
                seeds += [x for x in _reach.literals(u.node, 32) if x not in seeds and len(x) <= 512]
        samples = _regex_samples(plan.patterns())
        seeds += [x for x in samples if x not in seeds]
        seeds += [x for x in _composed_seeds(samples, plan.tokens) if x not in seeds]
    for i, lit in enumerate(seeds[:96 + COMPOSE_SEEDS]):
        try:
            (corpus / f"lit{i}").write_bytes(lit.encode("utf-8", "surrogatepass")[:256])
        except (UnicodeEncodeError, OSError):
            pass


def _reach_plan(ctx, tree, root):
    """What the fuzzer is told about the code the program reaches ([internal ref]): the packages the worker instruments,
    and a libFuzzer dictionary of the tokens its guards compare a string with (``sv/reach.py::dictionary``).
    Returns ``(plan, request_fields, libfuzzer_args)``."""
    plan = _Plan(ctx, tree)
    args = []
    units = ([_reach.Unit("", tree)] if tree is not None else []) + [u for u in plan.units if u.module]
    tokens = plan.tokens = _reach.dictionary(units, plan.fold)
    if tokens:
        path = root / "tokens.dict"
        path.write_text(_reach.render_dictionary(tokens))
        args.append(f"-dict={path}")
    return plan, {"reach": _reach.packages(plan.called)}, args


class _LiveCpu:
    """A running fuzzer's CPU for ``Ledger.add_server``: counted while the process is alive (its CPU
    is not reaped yet, so ``os.times`` cannot see it) and zero once it has been reaped (``os.times``
    then includes it).  ``close`` is the ledger's server protocol; the fuzzer is stopped by its owner."""

    def __init__(self, fuzzer):
        self._fuzzer = fuzzer

    @property
    def cpu(self) -> float:
        return self._fuzzer.cpu_seconds() if self._fuzzer.is_running() else 0.0

    def close(self) -> None:
        pass


def deepen(ctx, budget, round_=1):
    """The fuzzer as the search of an UNBOUNDED draw domain once enumeration ran dry ([internal ref]).

    ``search`` is the portfolio's bounded look (a few CPU seconds of Atheris, then the per-site solver
    on a plateau).  This is what is left to do with the rest of the budget: keep fuzzing, from the
    corpus the earlier rounds grew (``ctx.state``), and STREAM every vector the worker records as
    soon as it appears, so the engine's fresh replay (the only FALSE source) happens while the fuzzer
    keeps running and the run ends the moment a replay confirms a violation.

    The worker's hits here are RAISED property exceptions and the precise obligation predicates; the
    assert-fallback's ``False`` predicate (a hint for where to point the fuzzer: "this line was
    reached") is dropped, since every run reaches it and each hit is a replay.  Its CPU is on the
    run's ledger while it runs, and the one stop condition (``ctx.ledger.stopped``) ends it.
    Proposes vectors only: it can never decide TRUE.  It runs only for a program that drew something and raised
    no environment flag in its first run (``Outcome.flags``: a file effect, argv, a clock).
    """
    if budget <= 1 or not atheris_available():
        return
    first = getattr(ctx, "first", None)
    if first is not None and (not first.draws or first.flags):
        # nothing to vary (no draw: the run IS the program), or something the worker cannot reproduce or
        # contain: it runs the program for real, outside the engine's I/O model, so a file effect (or an
        # argv/clock read) is neither modelled nor safe to repeat millions of times
        return
    try:
        source = Path(ctx.program).read_text(encoding="utf-8")
        tree = ast.parse(source, ctx.program)
    except (OSError, ValueError, SyntaxError):
        return
    obligations = list(ctx.table.rows(ctx.classes)) if ctx.table is not None else []
    checks, metrics, _records = _obligation_tables(obligations)
    root = ctx.state.get("hybrid_root")
    if root is None:
        root = Path(ctx.cleanup.enter_context(tempfile.TemporaryDirectory(prefix="tharkun-sv-deepen-")))
        (root / "corpus").mkdir()
        plan, fields, args = _reach_plan(ctx, tree, root)
        _seed_corpus(root / "corpus", tree, plan)
        ctx.state.update(hybrid_root=root, hybrid_seen=set(), hybrid_fields=fields, hybrid_args=args)
    seen = ctx.state["hybrid_seen"]
    found = root / f"found{round_}"
    found.mkdir()
    request = root / f"request{round_}.json"
    request.write_text(json.dumps({"program": os.path.abspath(ctx.program), "classes": ctx.classes,
                                   "checks": checks, "metrics": metrics, "found": str(found),
                                   "files": list(ctx.files) if getattr(ctx, "files", None) else None,
                                   **ctx.state["hybrid_fields"]}))
    env = dict(subprocess_environment(), THARKUN_SV_HYBRID_REQ=str(request), PYTHONDONTWRITEBYTECODE="1",
               PYTHONHASHSEED="0")
    fuzzer = AtherisBackend(WORKER, str(root / "corpus"), python=ctx.python,
                            extra_args=[f"-rss_limit_mb={max(64, (ctx.memory_bytes or 2**30) // 2**20)}",
                                        *ctx.state["hybrid_args"]],
                            env=env, cwd=str(root), artifact_dir=str(root / "artifacts"))
    ledger = ctx.ledger
    if ledger is not None:
        ledger.add_server(_LiveCpu(fuzzer))
    wall_deadline = time.monotonic() + budget * 8 + 5      # a backstop for a /proc-less host or a hung child
    try:
        fuzzer.start()
        while True:
            running = fuzzer.is_running()
            for path in sorted(found.glob("*.json")):
                if path.name in seen:
                    continue
                seen.add(path.name)
                try:
                    vector = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                yield Candidate(vector, "hybrid:atheris", {"round": round_})
            if (not running or fuzzer.cpu_seconds() >= budget or time.monotonic() > wall_deadline
                    or (ledger is not None and ledger.stopped())):
                break
            time.sleep(0.2)
    except OSError as exc:
        ctx.log(f"hybrid: Atheris did not start ({exc})")
    finally:
        fuzzer.stop()
    ctx.log(f"hybrid: deepen round {round_}: Atheris cov={fuzzer.coverage()}, cpu={fuzzer.cpu_seconds():.1f}s")


def search(ctx, budget):
    """Propose vectors only. The engine decides FALSE by fresh admissible replay."""
    if budget <= 1 or not atheris_available():
        return
    try:
        source = Path(ctx.program).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return
    try:
        tree = ast.parse(source, ctx.program)
    except (SyntaxError, ValueError):
        tree = None
    # the per-site obligations come from the run's table ([internal ref] step 3a): the census was computed once,
    # before stage 1; this stage no longer walks the program a second time to rebuild it
    obligations = list(ctx.table.rows(ctx.classes))
    fallback = False
    if not obligations:
        obligations = _assert_fallback(source, ctx.program, ctx.classes)
        fallback = bool(obligations)
    # Regex-shaped candidates ([internal ref], owner idea) do not need a discharged obligation at
    # all -- they are whole vectors the engine can replay directly, same as any other
    # proposal, so they still run even when there is nothing to seed Atheris's own
    # request.json with below.
    if tree is not None:
        yield from _regex_vectors(ctx, tree)
    if not obligations:
        return
    checks, metrics, records = _obligation_tables(obligations)
    with tempfile.TemporaryDirectory(prefix="tharkun-sv-hybrid-") as tmp:
        root = Path(tmp)
        corpus, found = root / "corpus", root / "found"
        corpus.mkdir()
        found.mkdir()
        plan, fields, args = _reach_plan(ctx, tree, root)
        _seed_corpus(corpus, tree, plan)
        request = root / "request.json"
        request.write_text(json.dumps({"program": os.path.abspath(ctx.program), "classes": ctx.classes,
                                       "checks": checks, "metrics": metrics, "found": str(found), **fields,
                                       "files": list(ctx.files) if getattr(ctx, "files", None) else None}))
        env = dict(subprocess_environment(), THARKUN_SV_HYBRID_REQ=str(request), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONHASHSEED="0")
        fuzzer = AtherisBackend(WORKER, str(corpus), python=ctx.python,
                                extra_args=[f"-rss_limit_mb={max(64, (ctx.memory_bytes or 2**30) // 2**20)}",
                                            *args],
                                env=env, cwd=str(root), artifact_dir=str(root / "artifacts"))
        # [internal ref]: an earlier version of this fix also widened the fuzz duration for a
        # str-draw obligation (reserving less for the concolic handoff below, which
        # always refuses for this obligation shape -- "no harness or no site"). Measured
        # on a full-budget RW campaign: it REGRESSED bm_chaos_nondet/data-lookup -- more
        # Atheris CPU raised the odds of a raw float draw landing outside the range a
        # nondet_float_range() call at some OTHER, unrelated site clamps it into.
        # _child.py's own model then clamped a raw out-of-range draw into [a, b] (so
        # Tharkun's own replay of the witness reproduced the bug), but the independent
        # grading harness's shim REJECTS an out-of-range draw as invalid (NOT CONFIRMED)
        # instead of clamping it. ([internal ref] has since closed that mismatch: _child.py refuses
        # the same draws and the worker logs only in-domain values, docs/SV.md "Draw
        # domains"; the measurement below stands as it was taken.) Reverted: every
        # obligation, string-drawing or not, keeps the measured [internal ref] split.
        duration = min(3.0, budget / 3)
        # Budget this stage's own plateau/deadline loop by the subprocess's OWN CPU time
        # ([internal ref] part 2), not wall-clock: BenchExec measures CPU and SV-COMP ties break on
        # it, so a decision keyed on time.monotonic() gave a different answer depending on
        # how busy the shared host was at that moment (measured: SO gain +3/+2/+0 across
        # three otherwise-identical campaigns). fuzzer.cpu_seconds() is host-load-invariant:
        # a contended host makes the subprocess take longer in wall-clock terms to earn the
        # same CPU seconds, but it still earns them, so the plateau/handoff decision and the
        # budget handed to the concolic solver afterward no longer depend on load.
        #
        # A generous wall-clock backstop stays alongside it -- not to gate the normal case,
        # but because cpu_seconds() reports 0.0 (never advancing) if /proc is unavailable
        # (non-Linux dev boxes) or the subprocess is genuinely stuck, and only wall-clock can
        # catch that. It is sized well above the CPU cap so ordinary contention (the case
        # this fix targets) never trips it -- only a real hang would.
        wall_deadline = time.monotonic() + max(budget, duration) * 8 + 5
        cpu_cap = min(budget, duration)
        try:
            fuzzer.start()
            best, last_gain_cpu = 0, 0.0
            while (fuzzer.is_running() and fuzzer.cpu_seconds() < cpu_cap
                   and time.monotonic() < wall_deadline):
                time.sleep(0.2)
                cov = fuzzer.coverage()
                if cov > best:
                    best, last_gain_cpu = cov, fuzzer.cpu_seconds()
            plateau = (not fuzzer.is_running() or
                       should_assist(fuzzer.coverage(), best, fuzzer.cpu_seconds() - last_gain_cpu, 0.2))
        except OSError as exc:
            ctx.log(f"hybrid: Atheris did not start ({exc})")
            plateau = True
        finally:
            fuzzer.stop()
        used = fuzzer.cpu_seconds()
        ctx.log(f"hybrid: {len(obligations)} obligations{' (assert fallback, #1104)' if fallback else ''}, "
                f"Atheris cov={fuzzer.coverage()}, cpu={used:.1f}s, plateau={plateau}")
        for path in found.glob("*.json"):
            try:
                vector = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            yield Candidate(vector, "hybrid:atheris", {"obligations": len(obligations)})
        left = budget - used
        if plateau and left > 1:
            from tharkun.sv.concolic import solve
            solved = solve(ctx.program, ctx.classes, budget_seconds=left,
                           memory_bytes=ctx.memory_bytes, python=ctx.python,
                           extra_sites=records, site_kind="obligation", log=ctx.log, files=ctx.files, **ctx.call_facts())
            for cand in solved.candidates:
                yield Candidate(cand.vector, "hybrid:obligation", cand.detail)
