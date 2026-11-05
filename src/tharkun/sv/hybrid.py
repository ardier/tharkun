"""Coverage search for the negation of each open SV safety obligation."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from tharkun.hybrid.backend import AtherisBackend
from tharkun.hybrid.fdp_codec import atheris_available
from tharkun.hybrid.fuzzer import should_assist
from tharkun.solve import seed as _rng_seed
from tharkun.solve.environment import subprocess_environment
from tharkun.solve.preimage.regex_members import REGEX_SAMPLES_DRAWN, REGEX_SAMPLES_KEPT, REGEX_SAMPLE_SECONDS
from tharkun.sv import _hybrid_draws
from tharkun.sv import reach as _reach
from tharkun.sv.stages import Candidate

#: libFuzzer executions per CPU second of a stage's budget ([internal ref]).  A stage that stops on CPU time stops after a
#: different number of executions on every run (3200 to 5400 in 1.8 CPU s, measured on python-string-utils'
#: ``__RomanNumbers.encode``), so a find near the end of the budget came and went; ``-runs=`` counts the work instead and,
#: with a fixed ``-seed=``, the same executions happen every time.  The rate is the low end of what the fuzzer does per CPU
#: second on the measured subjects, so the count cap binds before the CPU cap, which stays as the backstop.
HYBRID_RUNS_PER_CPU_SECOND = 2000


def _executions(fuzzer) -> int:
    """The backend's libFuzzer execution count, 0 for a backend that does not count (a stub)."""
    count = getattr(fuzzer, "executions", None)
    return count() if callable(count) else 0


def runs_cap(cpu_seconds: float) -> int:
    """The libFuzzer ``-runs=`` value for a stage given *cpu_seconds* of budget."""
    return max(1, int(cpu_seconds * HYBRID_RUNS_PER_CPU_SECOND))

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


def _assert_fallback(source, program, classes, *, raises=False):
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
    if "AssertionError" not in classes and not raises:
        return []
    from tharkun.solve.obligations import Invariant, Obligation
    import hashlib
    try:
        tree = ast.parse(source, program)
    except (SyntaxError, ValueError):
        return []
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    out = []
    for node in ast.walk(tree):
        cls = "AssertionError" if isinstance(node, ast.Assert) else None
        if raises and isinstance(node, ast.Raise):
            expr = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            cls = expr.id if isinstance(expr, ast.Name) else None
        if cls not in classes:
            continue
        out.append(Obligation(sha, f"$fallback:{node.lineno}:{node.col_offset}", cls,
                              Invariant(program, node.lineno, node.col_offset + 1, "<module>", "False"),
                              "census refused the whole program (#1104): taken from the AST directly"))
    return out


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

    Shares the deterministic, shortest-first printable samples of ``_regex_samples`` with the corpus.
    A pattern sample is a proposal; the native replay still decides whether it satisfies the task.

    ``ctx.first.draws`` gives the real draw SHAPE (kinds and count); only the ``str``
    positions are replaced, one at a time, so every other draw stays at its observed
    value -- the same substitution ``sv/engine.py::verify``'s own enumeration does with
    its small-first candidates. A proposal here is exactly as trusted as any other: the
    engine's fresh replay is the only thing that can call it FALSE."""
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
        for value in _regex_samples([(pattern, flags)]):
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


# Kept at the old SV surface for callers; the implementation lives below DRIVE ([internal ref]).
#: composed seeds: the longest guard token spliced into samples, and how many composed seeds the corpus gets
COMPOSE_TOKEN_MAX = 4
COMPOSE_SEEDS = 128


def _regex_samples(patterns):
    """Compatibility entry for the shared shortest printable regex samples ([internal ref])."""
    from tharkun.solve.preimage.regex_members import samples
    return samples(patterns)


#: boundary and flag-difference members ([internal ref] PR3): at most this many per pattern, and compositions of them
BOUNDARY_SAMPLES = 24
EXTRA_COMPOSE_SEEDS = 32


def _length_constants(units) -> list[int]:
    """The integer constants the reached code compares a ``len(...)`` with (``len(input_string) > 320``,
    ``len(head) > 64``): the lengths a boundary mutant lives at."""
    out = []
    for u in units:
        for node in ast.walk(u.node):
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                sides = (node.left, node.comparators[0])
                for a, b in (sides, sides[::-1]):
                    if (isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id == "len"
                            and isinstance(b, ast.Constant) and type(b.value) is int and 0 < b.value <= 4096
                            and b.value not in out):
                        out.append(b.value)
    return out[:8]


def _boundary_samples(patterns, constants):
    """Members of each reached pattern AT the reached length constants ([internal ref] PR3; the K4 boundary mutants): of a
    total length ``c`` and ``c + 1`` for each constant, and with the stretches between a top-level literal character
    (``@``) at a pair of the constants or their neighbours (``64@255`` is a 320-character email whose parts sit at
    their own bounds).  Deterministic (``solve/preimage/regex_members.py``); each member is checked by CPython."""
    from tharkun.solve.preimage import regex_members
    if not constants:
        return []
    lens = sorted({n for c in constants for n in (c - 1, c, c + 1) if 0 < n <= _hybrid_draws.STR_DRAW_MAX})
    deadline = time.monotonic() + REGEX_SAMPLE_SECONDS
    out = []
    for pattern, flags in patterns:
        got = []
        try:
            # the stretches around a separator first (they respect the pattern's own structure), then totals
            for sep in [c for c in "@:/=;,.-_" if c in pattern][:2]:
                for a in lens:
                    for b in lens:
                        if time.monotonic() > deadline or len(got) >= BOUNDARY_SAMPLES:
                            break
                        got += list(regex_members.members(pattern, flags, how="search", split=sep,
                                                          segments=(a, b), limit=1))
            for n in lens:
                if time.monotonic() > deadline or len(got) >= 2 * BOUNDARY_SAMPLES:
                    break
                got += list(regex_members.members(pattern, flags, how="search", length=n, limit=1))
        except Exception:  # noqa: BLE001 -- a hint is optional
            pass
        out += [g for g in dict.fromkeys(got) if all(c in _reach._PRINTABLE for c in g)]
    return out


def _flag_samples(patterns):
    """For a pattern text the reached code compiles under DIFFERENT flags (an original and its mutant, two call
    sites), members built where the flags disagree (``.`` taking ``'\\n'`` for DOTALL, swapped case for IGNORECASE,
    ...: ``solve/preimage/regex_members.py::flag_variants``).  Not checked here: the program's run is the check."""
    from tharkun.solve.preimage import regex_members
    by_text: dict = {}
    for pattern, flags in patterns:
        by_text.setdefault(pattern, [])
        if flags not in by_text[pattern]:
            by_text[pattern].append(flags)
    out = []
    for pattern, flagsets in by_text.items():
        for i, fa in enumerate(flagsets):
            for fb in flagsets[i + 1:]:
                try:
                    out += [v for v in regex_members.flag_variants(pattern, fa, fb, limit=8)
                            if all(c in _reach._PRINTABLE for c in v)]
                except Exception:  # noqa: BLE001
                    pass
    return list(dict.fromkeys(out))


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
            if layer < len(seeds) and seeds[layer] not in seen and len(seeds[layer]) <= _hybrid_draws.STR_DRAW_MAX:
                seen.add(seeds[layer])
                out.append(seeds[layer])
                if len(out) >= limit:
                    return out
    return out


def _add_vectors(corpus, vectors, kinds):
    """Write whole draw vectors into the fuzzer's corpus as the exact bytes that decode back to them, those whose
    kinds are the first run's only (the worker's str lengths read the first run's ``kinds``)."""
    for vector in vectors or ():
        if kinds and [e.get("kind") for e in vector] != list(kinds):
            continue
        try:
            data = _hybrid_draws.encode_checked(vector)
        except Exception:  # noqa: BLE001 -- no atheris, a decode failure: not written
            data = None
        if data is not None:
            try:
                (corpus / f"vector{hashlib.sha256(data).hexdigest()[:16]}").write_bytes(data)
            except OSError:
                pass


def _seed_corpus(corpus, tree, plan=None, kinds=None, vectors=(), world=None):
    """The zero input, then one file each for the string literals of the program and of the task code it
    calls ([internal ref]: a product's library is where its guards compare the draw with something), a few strings its
    regexes accept, and ([internal ref] PR3) members at its length constants and where its flag sets disagree.

    [internal ref]: with the program's draw ``kinds`` known (its first run), every string becomes a WHOLE draw vector --
    the string in the first str draw, every bool both ways (at most 4 combinations), every int 0 -- written as the
    exact bytes that decode back to it (``sv/_hybrid_draws.py::encode_checked``, the real provider as the
    oracle).  A program whose kinds are unknown, or a vector that cannot be written, falls back to the raw text.

    [internal ref]: ``vectors`` (``Context.seed_vectors``, the caller's whole draw vectors) come first, each written the same
    way, and only when its kinds are the first run's (the worker's str lengths read the first run's ``kinds``)."""
    (corpus / "zero").write_bytes(bytes(64))
    _add_vectors(corpus, vectors, kinds)
    seeds = list(_literal_seeds(tree)) if tree is not None else []
    if plan is not None:
        for u in plan.units:
            if u.module:
                seeds += [x for x in _reach.literals(u.node, 32) if x not in seeds and len(x) <= 512]
        patterns = plan.patterns()
        samples = _regex_samples(patterns)
        units = [u for u in plan.units if u.module] + ([_reach.Unit("", tree)] if tree is not None else [])
        extra = _flag_samples(patterns) + _boundary_samples(patterns, _length_constants(units))
        # the short samples and THEIR compositions first, with the whole composition budget (a quoted local part
        # wrapped round the shortest email sample is the m131 kill), then the boundary / flag members and a few
        # compositions of theirs (an escaped token at the head of a bound-length email is the m125 shape)
        seeds += [x for x in samples if x not in seeds]
        seeds += [x for x in _composed_seeds(samples, plan.tokens) if x not in seeds]
        seeds += [x for x in extra if x not in seeds]
        seeds += [x for x in _composed_seeds(extra[:8], plan.tokens, limit=EXTRA_COMPOSE_SEEDS) if x not in seeds]
    files = 0
    for lit in seeds[:96 + COMPOSE_SEEDS + 4 * BOUNDARY_SAMPLES + EXTRA_COMPOSE_SEEDS]:
        for data in _seed_bytes(lit, kinds):
            try:
                (corpus / f"lit{files}").write_bytes(data)
                files += 1
            except OSError:
                pass

    if world is not None:
        for path in corpus.iterdir():
            path.write_bytes(_hybrid_draws.pack_world(path.read_bytes(), world))


def _seed_bytes(text, kinds):
    """The corpus files for one seed string: whole draw vectors when the program's draw kinds are known."""
    import itertools
    if kinds and "str" in kinds and all(k in ("str", "bool", "int") for k in kinds):
        bools = [i for i, k in enumerate(kinds) if k == "bool"]
        combos = (list(itertools.product((False, True), repeat=len(bools))) if len(bools) <= 2
                  else [(False,) * len(bools), (True,) * len(bools)])
        first = kinds.index("str")
        out = []
        for combo in combos:
            flags = dict(zip(bools, combo))
            vector = [{"kind": k, "value": text if i == first else "" if k == "str" else flags[i] if k == "bool" else 0}
                      for i, k in enumerate(kinds)]
            try:
                data = _hybrid_draws.encode_checked(vector)
            except Exception:  # noqa: BLE001 -- no atheris, a decode failure: not written
                data = None
            if data is not None:
                out.append(data)
        if out:
            return out
    try:
        return [text.encode("utf-8", "surrogatepass")[:_hybrid_draws.STR_DRAW_MAX]]
    except UnicodeEncodeError:
        return []


def _draw_kinds(ctx) -> list:
    """The program's draw kinds in call order, from its first (all-defaults) run, or [] when unknown."""
    first = getattr(ctx, "first", None)
    draws = getattr(first, "draws", None) or []
    return [d.get("kind") for d in draws if isinstance(d, dict) and isinstance(d.get("kind"), str)]


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
    # [internal ref]: only a BOXED run may deepen past a file effect -- its worker runs under the box, where a
    # declared-file read and an undeclared open's mocked answer are the model's own, reproducible state (and
    # the box fixes HOME/XDG, so no run-varying path leaks into object state).  An unboxed run's fuzzer runs
    # the program for real outside the model, so a file effect there is neither modelled nor safe to repeat
    # millions of times: the stage still stops on it, exactly as before.
    ignorable = {"file_effect", "stdin"} if getattr(ctx, "box", False) else set()
    if first is not None and (not first.draws and not getattr(ctx, "box", False) or (set(first.flags) - ignorable)):
        # nothing to vary (no draw: the run IS the program), or something the worker cannot reproduce or
        # contain: an argv/clock read, or (unboxed) any file effect
        return
    try:
        source = Path(ctx.program).read_text(encoding="utf-8")
        tree = ast.parse(source, ctx.program)
    except (OSError, ValueError, SyntaxError):
        return
    obligations = list(ctx.table.rows(ctx.classes)) if ctx.table is not None else []
    if getattr(ctx, "box", False):
        obligations += _assert_fallback(source, ctx.program, ctx.classes, raises=True)
    checks, metrics, records = _obligation_tables(obligations)
    world = ctx.table.world_store().snapshot() if getattr(ctx, "box", False) and ctx.table is not None else None
    root = ctx.state.get("hybrid_root")
    if root is None:
        root = Path(ctx.cleanup.enter_context(tempfile.TemporaryDirectory(prefix="tharkun-sv-deepen-")))
        (root / "corpus").mkdir()
        plan, fields, args = _reach_plan(ctx, tree, root)
        kinds = _draw_kinds(ctx)
        _seed_corpus(root / "corpus", tree, plan, kinds,
                     [*getattr(ctx, "seed_vectors", ()), *_feedback_vectors(ctx)], world=world)
        fields = dict(fields, kinds=kinds, frontier=str(root / "frontier.json"))
        ctx.state.update(hybrid_root=root, hybrid_seen=set(), hybrid_fields=fields, hybrid_args=args,
                         hybrid_tokens=plan.tokens, hybrid_kinds=kinds)
    seen = ctx.state["hybrid_seen"]
    (root / "frontier.json").unlink(missing_ok=True)      # a stale input is not this round's
    found = root / f"found{round_}"
    found.mkdir()
    request = root / f"request{round_}.json"
    request.write_text(json.dumps({"program": os.path.abspath(ctx.program), "classes": ctx.classes,
                                   "checks": checks, "metrics": metrics, "found": str(found),
                                   **({"box": True, "world": world or {}} if getattr(ctx, "box", False) else {}),
                                   "files": list(ctx.files) if getattr(ctx, "files", None) else None,
                                   **ctx.state["hybrid_fields"]}))
    env = dict(subprocess_environment(), THARKUN_SV_HYBRID_REQ=str(request), PYTHONDONTWRITEBYTECODE="1",
               PYTHONHASHSEED="0")
    fuzzer = AtherisBackend(WORKER, str(root / "corpus"), python=ctx.python,
                            extra_args=[f"-rss_limit_mb={max(64, (ctx.memory_bytes or 2**30) // 2**20)}",
                                        f"-seed={_rng_seed.libfuzzer_seed(f'sv:hybrid:deepen:{round_}')}",
                                        f"-runs={runs_cap(budget)}",
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
                yield _candidate(vector, {"round": round_}, getattr(ctx, "box", False))
            if (not running or fuzzer.cpu_seconds() >= budget or time.monotonic() > wall_deadline
                    or (ledger is not None and ledger.stopped())):
                break
            time.sleep(0.2)
    except OSError as exc:
        ctx.log(f"hybrid: Atheris did not start ({exc})")
    finally:
        fuzzer.stop()
    ctx.log(f"hybrid: deepen round {round_}: Atheris cov={fuzzer.coverage()}, cpu={fuzzer.cpu_seconds():.1f}s, "
            f"execs={_executions(fuzzer)}")
    # the loop goes on after the first corpus: this round's deepest input, if new, goes to the solver and what it
    # learned is in the corpus and the dictionary of the next round (inside the ledger's share, never a domain bound)
    if ctx.table is not None and not (ledger is not None and ledger.stopped(reserve=1)):
        cands, new = _handoff(ctx, root, root / "corpus", ctx.state["hybrid_kinds"], records,
                              ledger.share(0.1) if ledger is not None else budget / 4,
                              ctx.state["hybrid_args"], ctx.state["hybrid_tokens"],
                              ctx.state.setdefault("hybrid_frontiers", set()))
        yield from cands
        if new:
            ctx.log(f"hybrid: deepen round {round_}: {new} solver facts into the corpus")


def _operand_token(operand):
    """A dictionary token for a solver operand in the draw-byte layout of ``_hybrid_draws`` (a str is its own
    characters, an int the 8 little-endian bytes ``ConsumeInt(8)`` reads, a bool one byte)."""
    kind, value = operand["kind"], operand["value"]
    if kind == "str":
        return value if all(c in _reach._PRINTABLE for c in value) else None
    data = _hybrid_draws.encode([operand])
    return data.decode("latin-1") if data else None


def _operand_vectors(frontier, operands, limit=16):
    """The frontier input with each operand put in every draw position of its kind, one vector each: corpus
    seeds that try the value the solver saw where the fuzzer's own mutation would have to find it."""
    out = []
    for operand in operands:
        for i, entry in enumerate(frontier):
            if entry.get("kind") == operand["kind"] and entry.get("value") != operand["value"]:
                out.append([dict(e, value=operand["value"]) if j == i else dict(e) for j, e in enumerate(frontier)])
                if len(out) >= limit:
                    return out
    return out


def _handoff(ctx, root, corpus, kinds, records, budget, args, tokens, seen):
    """The fuzzer talking to the solver and back, once: the deepest input the worker kept (``frontier.json``,
    not handed over before) goes to the solver with the reached prefix fixed; what the solver solved, part-solved
    and compared goes into the run's store (``draw_vector`` from ``concolic:feedback``, ``guard_operand``), and
    from there into ``corpus`` (the vectors, the operands tried at the frontier's positions) and the libFuzzer
    dictionary (``args`` is updated in place).  ``(candidates, new)``: ``new`` counts what the store did not hold."""
    frontier = _read_vector(root / "frontier.json")
    key = json.dumps(frontier, sort_keys=True)
    if budget <= 1 or frontier is None or _is_first_run(ctx, frontier) or key in seen:
        return [], 0
    seen.add(key)
    from tharkun.sv.concolic import solve
    _publish(ctx, frontier, "hybrid:frontier")
    ctx.log(f"hybrid: plateau; frontier input {key[:200]} goes to the solver")
    solved = solve(ctx.program, ctx.classes, budget_seconds=budget, memory_bytes=ctx.memory_bytes,
                   python=ctx.python, extra_sites=records, site_kind="obligation", log=ctx.log,
                   files=ctx.files, seed_vector=frontier, **ctx.call_facts())
    cands = [Candidate(c.vector, "hybrid:feedback", dict(c.detail, frontier=True)) for c in solved.candidates]
    before = len(_feedback_vectors(ctx)) + len(ctx.table.operands() if ctx.table is not None else ())
    for vector in [*(c.vector for c in solved.candidates), *solved.seeds]:
        if vector and vector != frontier:
            _publish(ctx, vector, "concolic:feedback")
    if ctx.table is not None:
        for o in solved.operands:
            try:
                ctx.table.publish_operand(o["kind"], o["value"], producer="concolic:feedback")
            except (KeyError, ValueError):
                pass
    operands = ctx.table.operands() if ctx.table is not None else []
    new = len(_feedback_vectors(ctx)) + len(operands) - before
    _add_vectors(corpus, [*_feedback_vectors(ctx), *_operand_vectors(frontier, operands)], kinds)
    extra = [t for t in (_operand_token(o) for o in operands) if t]
    if extra:
        path = root / "tokens.dict"
        path.write_text(_reach.render_dictionary(list(dict.fromkeys([*tokens, *extra]))))
        args[:] = [a for a in args if not a.startswith("-dict=")] + [f"-dict={path}"]
    return cands, new


def _fuzz_round(ctx, root, corpus, args, budget, round_=0):
    """One bounded Atheris round on ``corpus``: ``(fuzzer, plateau)``.  The CPU and wall-clock reasoning of the
    first round ([internal ref], [internal ref]) is kept below; a feedback round (``round_`` > 0) is the same look with another seed."""
    (root / "frontier.json").unlink(missing_ok=True)      # a stale input is not this round's
    env = dict(subprocess_environment(), THARKUN_SV_HYBRID_REQ=str(root / "request.json"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONHASHSEED="0")
    fuzzer = AtherisBackend(WORKER, str(corpus), python=ctx.python,
                            extra_args=[f"-rss_limit_mb={max(64, (ctx.memory_bytes or 2**30) // 2**20)}",
                                        f"-seed={_rng_seed.libfuzzer_seed('sv:hybrid:search' + (f':{round_}' if round_ else ''))}",
                                        f"-runs={runs_cap(min(budget, min(3.0, budget / 3)))}",
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
    return fuzzer, plateau


#: how many times a plateau may go to the solver and back to the fuzzer in one ``search``
FEEDBACK_ROUNDS = 3


def _read_vector(path):
    try:
        vector = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return vector if isinstance(vector, list) and vector else None


def _is_first_run(ctx, vector) -> bool:
    """The fuzzer's deepest input is the all-defaults run the solver starts from anyway."""
    draws = [{"kind": d.get("kind"), "value": d.get("value")} for d in getattr(getattr(ctx, "first", None), "draws", None) or []]
    return draws == [{"kind": e.get("kind"), "value": e.get("value")} for e in vector]


def _publish(ctx, vector, producer) -> None:
    """Put a draw vector in the run's store (the ``draw_vector`` evidence a producer's vectors use): the fuzzer's
    frontier for the solver, the solver's models for the fuzzer.  A hint, never a verdict."""
    if ctx.table is None:
        return
    from tharkun.sv.eregion_bridge import publish_draw_vector
    try:
        publish_draw_vector(ctx.table.kb, {"vector": [{"kind": e["kind"], "value": e["value"]} for e in vector]},
                            producer=producer)
    except (ValueError, KeyError, TypeError):
        pass


def _feedback_vectors(ctx) -> list:
    """The solver's vectors in the run's store (``_publish(..., "concolic:feedback")``), for the fuzzer's corpus."""
    if ctx.table is None:
        return []
    from tharkun.sv.eregion_bridge import DRAW_VECTOR
    return [e.value["vector"] for e in ctx.table.kb.evidence(kind=DRAW_VECTOR)
            if any(d.producer == "concolic:feedback" for d in e.derivations)]


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
    if getattr(ctx, "box", False):
        obligations += _assert_fallback(source, ctx.program, ctx.classes, raises=True)
    if not obligations:
        return
    checks, metrics, records = _obligation_tables(obligations)
    with tempfile.TemporaryDirectory(prefix="tharkun-sv-hybrid-") as tmp:
        root = Path(tmp)
        corpus, found = root / "corpus", root / "found"
        corpus.mkdir()
        found.mkdir()
        plan, fields, args = _reach_plan(ctx, tree, root)
        kinds = _draw_kinds(ctx)
        world = ctx.table.world_store().snapshot() if getattr(ctx, "box", False) else None
        _seed_corpus(corpus, tree, plan, kinds, getattr(ctx, "seed_vectors", ()), world=world)
        fields = dict(fields, kinds=kinds)
        request = root / "request.json"
        # [internal ref]: a BOXED run (``ctx.box``) sends "box" -- the worker then runs the program under the box, where
        # an undeclared path is a mocked object (absent, unflagged) instead of a host probe or a refusal.  An
        # unboxed run (every SV-COMP one) sends no key at all, so its request is byte-identical to before.
        request.write_text(json.dumps({"program": os.path.abspath(ctx.program), "classes": ctx.classes,
                                       "checks": checks, "metrics": metrics, "found": str(found), **fields,
                                       "frontier": str(root / "frontier.json"),
                                       **({"box": True, "world": world or {}} if getattr(ctx, "box", False) else {}),
                                       "files": list(ctx.files) if getattr(ctx, "files", None) else None}))
        fuzzer, plateau = _fuzz_round(ctx, root, corpus, args, budget)
        used = fuzzer.cpu_seconds()
        ctx.log(f"hybrid: {len(obligations)} obligations{' (assert fallback, #1104)' if fallback else ''}, "
                f"Atheris cov={fuzzer.coverage()}, cpu={used:.1f}s, execs={_executions(fuzzer)}, plateau={plateau}")
        seen = set()

        def newly_found():
            for path in sorted(found.glob("*.json")):
                if path.name in seen:
                    continue
                seen.add(path.name)
                try:
                    vector = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                yield _candidate(vector, {"obligations": len(obligations)}, getattr(ctx, "box", False))

        yield from newly_found()
        left = budget - used
        # Feedback (the fuzzer and the solver talking, through the run's store): a plateau hands the deepest input
        # the fuzzer reached to the solver (``_handoff``); what it solved, part-solved and compared goes into the
        # corpus and the dictionary and the fuzzer looks again.  The rounds bound the work, not the domain.
        frontiers = ctx.state.setdefault("hybrid_frontiers", set())      # shared with deepen: one solve per input
        for round_ in range(1, 1 + FEEDBACK_ROUNDS if plateau else 1):
            cands, new = _handoff(ctx, root, corpus, kinds, records, left / 2, args, plan.tokens, frontiers)
            yield from cands
            if not new:
                break
            ctx.log(f"hybrid: feedback round {round_}: {new} solver facts into the corpus")
            fuzzer, plateau = _fuzz_round(ctx, root, corpus, args, max(0.0, left / 2), round_)
            used += fuzzer.cpu_seconds()
            ctx.log(f"hybrid: feedback round {round_}: Atheris cov={fuzzer.coverage()}, "
                    f"cpu={fuzzer.cpu_seconds():.1f}s, execs={_executions(fuzzer)}, plateau={plateau}")
            yield from newly_found()
            left = budget - used
            if not plateau:
                break
        if plateau and left > 1:
            from tharkun.sv.concolic import solve
            solved = solve(ctx.program, ctx.classes, budget_seconds=left,
                           memory_bytes=ctx.memory_bytes, python=ctx.python,
                           extra_sites=records, site_kind="obligation", log=ctx.log, files=ctx.files, **ctx.call_facts())
            for cand in solved.candidates:
                yield Candidate(cand.vector, "hybrid:obligation", cand.detail)


def _candidate(record, detail, box=False):
    """Boxed worker records carry the replay world beside the ordinary draw vector."""
    if box and isinstance(record, dict):
        return Candidate(record["vector"], "hybrid:atheris", dict(detail, world=record["world"]))
    return Candidate(record, "hybrid:atheris", detail)
