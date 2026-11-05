"""Deterministic members of a Python regex, and a bounded native membership check ([internal ref]).

Ported in spirit from the K2 ``strpath`` lessons ([internal ref] note, 2026-10-05; the Eregion kernel was deleted on the
owner's ruling that candidate generation is Tharkun's job): walk CPython's own parse tree (``re._parser``) with a CHOICE
SEQUENCE -- the simplest choice at every point (first branch, fewest repeats, first class member) gives the shortest
member, and varying one choice at a time gives its near neighbours -- so boundary mutants are cheap.  Hypothesis'
random ``from_regex`` is not used: a random sample is short or long by luck, and Phase A of [internal ref] measured that luck
deciding whether the K5 mutants die (a 1-character quoted local part is ONE specific shape).

What it offers:

* :func:`members` -- shortest-first members (``limit`` of them; every branch keeps its simplest member), optionally of
  an exact total ``length`` (each repeat's count and each branch chosen so the lengths add up; several distributions of
  the slack are tried) or with exact lengths per SEGMENT (the stretches of the top-level sequence between its literal
  ``split`` characters: ``split="@", segments=(64, 255)`` asks an email pattern for a 64-character local part and a
  255-character domain);
* :func:`flag_variants` -- members built to exercise the one node a flag changes between two flag sets (``.`` takes
  ``'\\n'`` when DOTALL differs, a cased letter takes its swapped case when IGNORECASE differs, a ``^``/``$`` sits next to
  a ``'\\n'`` when MULTILINE differs, a ``\\w``/``\\d`` takes a non-ASCII member when ASCII differs);
* :func:`native_member` / :func:`native_call` -- CPython's own ``re`` on a builtin str, admitted only when
  :func:`admitted` says the backtracking work is bounded: :func:`check_cost` under :data:`WORK_LIMIT` (the product of the
  unbounded repeats' ranges along any sequence, times the scan positions for ``search``-like methods), or a finite
  estimate over it whose very call finishes inside :data:`PROBE_SECONDS` in a killed-on-deadline forked probe; a
  backreference or a deeply nested unbounded repeat is refused.  An inadmissible check answers ``None`` -- "not checked", which a want treats as not holding.

Every member a generator yields is a CANDIDATE: :func:`members` keeps only those :func:`native_member` confirms, and a
caller forward-checks again against whatever it wants.  ``refusal(pattern, flags)`` names why a pattern is outside the
generator (a backreference, a conditional group, a lookbehind, a parse error, ``re.LOCALE``).  ``re.DEBUG`` changes no
semantics but makes compilation print: it is masked off for checking, never compiled with.

SOLVING by ``docs/LAYERS.md``: stdlib only; ``re._parser``/``re._constants`` are CPython 3.14's own parser modules.
"""

from __future__ import annotations

import functools
import os
import re
import signal
import time
from re import _constants as C
from re import _parser as P
from typing import Iterator

__all__ = ["members", "flag_variants", "native_member", "native_call", "check_cost", "refusal", "length_range",
           "WORK_LIMIT", "PROBE_SECONDS", "admitted", "samples", "REGEX_SAMPLES_DRAWN", "REGEX_SAMPLES_KEPT", "REGEX_SAMPLE_SECONDS"]

#: regex-positive samples: DRAWN per pattern, KEPT per pattern (the shortest: a short accepted string is the one
#: that still reaches a branch after its parts are wrapped or cut), and the wall clock the drawing may take in all
REGEX_SAMPLES_DRAWN = 8
REGEX_SAMPLES_KEPT = 3
REGEX_SAMPLE_SECONDS = 3.0

#: the backtracking-work bound of an admitted native check (an estimate of steps, not a soundness argument: it only keeps
#: a forward check from becoming a catastrophic backtrack; a check over the bound is simply not made)
WORK_LIMIT = 50_000_000
#: wall-clock seconds a forked probe may take to run an over-estimate check before it is killed (:func:`admitted`)
PROBE_SECONDS = 2.0
#: longest string a nested-unbounded-repeat pattern is natively checked on (2**n steps per nesting level)
_NESTED_MAX_LEN = 24
INF = float("inf")
_REPEATS = (C.MAX_REPEAT, C.MIN_REPEAT, C.POSSESSIVE_REPEAT)
#: class members in preference order: what a person writes first, then the rest of printable ASCII, then a few
#: non-ASCII letters/digits/spaces (the ones a Unicode-vs-ASCII flag tells apart)
_PREF = ("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         "._-+@!#$%&'*/=?^`{|}~\"(),:;<>[\\]" " \t\n\r\x0b\x0c" "éß٣ ")
_SEQ_ALTS = 8           # alternatives kept per (sub)sequence in shortest-first mode
_REPEAT_EXTRA = 3       # counts tried above a repeat's minimum in shortest-first mode


# --------------------------------------------------------------------------- parsing and admission

@functools.lru_cache(maxsize=512)
def _parse(pattern: str, flags: int):
    return P.parse(pattern, flags & ~re.DEBUG)


def refusal(pattern, flags: int = 0) -> str | None:
    """Why the generator cannot work on this pattern, or ``None``."""
    if type(pattern) is not str:
        return "pattern is not a str (bytes patterns are not generated)"
    if type(flags) not in (int, bool, re.RegexFlag):
        return "flags are not an int"
    if flags & re.LOCALE:
        return "re.LOCALE depends on the host locale"
    try:
        tree = _parse(pattern, int(flags))
    except (re.error, ValueError, OverflowError, RecursionError) as exc:
        return f"pattern does not parse: {type(exc).__name__}"
    bad = _unsupported(list(tree))
    return bad


def _unsupported(nodes) -> str | None:
    for op, av in nodes:
        if op in (C.GROUPREF, C.GROUPREF_IGNORE, C.GROUPREF_LOC_IGNORE, C.GROUPREF_UNI_IGNORE):
            return "backreference: a member depends on what an earlier group matched"
        if op is C.GROUPREF_EXISTS:
            return "conditional group"
        if op in (C.ASSERT, C.ASSERT_NOT) and av[0] < 0:
            return "lookbehind"
        for sub in _children(op, av):
            bad = _unsupported(sub)
            if bad:
                return bad
    return None


def _children(op, av):
    if op is C.SUBPATTERN:
        return [list(av[3])]
    if op is C.BRANCH:
        return [list(b) for b in av[1]]
    if op in _REPEATS:
        return [list(av[2])]
    if op in (C.ASSERT, C.ASSERT_NOT):
        return [list(av[1])]
    if op is C.ATOMIC_GROUP:
        return [list(av)]
    return []


def check_cost(pattern: str, flags: int, n: int, how: str = "match") -> float:
    """An upper estimate of the backtracking steps of ``<how>`` on a string of length ``n``; ``inf`` when the pattern has
    a backreference, or a nested unbounded repeat and ``n`` exceeds :data:`_NESTED_MAX_LEN` (no polynomial bound; up to
    that length an exponential ``2**n`` per nesting level is charged, so :data:`WORK_LIMIT` still gates the check).

    An unbounded repeat of ONE character class followed by a node that cannot start with a member of that class (a
    literal outside it, ``$``) never backtracks into itself, so it costs 1; any other unbounded repeat may retry each of
    its ``n + 1`` lengths.  Factors multiply along a sequence, add across branches, and a scanning method (``search``,
    ``sub``, ...) multiplies by the ``n + 1`` start positions."""
    try:
        tree = _parse(pattern, int(flags))
        gflags = int(flags) | tree.state.flags
    except (re.error, ValueError, OverflowError, RecursionError):
        return INF

    def seq(nodes, inside_unbounded, fl):
        total = 1.0
        for i, (op, av) in enumerate(nodes):
            nxt = nodes[i + 1] if i + 1 < len(nodes) else None
            total *= node(op, av, inside_unbounded, nxt, fl)
            if total == INF:
                return INF
        return total

    def node(op, av, inside_unbounded, nxt, fl):
        if op in _REPEATS:
            lo, hi, item = av
            item = list(item)
            unbounded = hi is C.MAXREPEAT or hi > 64
            if unbounded and inside_unbounded:
                # shortcut: 2**n bounds the segmentations of ONE extra nesting level; each deeper level multiplies
                # another 2**n through the sequence product.  Upgrade if a pattern needs a tighter estimate.
                return 2.0 ** n if n <= _NESTED_MAX_LEN else INF
            inner = seq(item, inside_unbounded or unbounded, fl)
            if unbounded and _deterministic(item, nxt, fl):
                return inner
            span = (n + 1) if unbounded else (hi - lo + 1)
            return span * inner
        if op is C.SUBPATTERN:
            return seq(list(av[3]), inside_unbounded, (fl | av[1]) & ~av[2])
        if op is C.BRANCH:
            return sum(seq(list(b), inside_unbounded, fl) for b in av[1])
        if op in (C.ASSERT, C.ASSERT_NOT):
            return seq(list(av[1]), inside_unbounded, fl) * (n + 1)
        if op is C.ATOMIC_GROUP:
            return seq(list(av), inside_unbounded, fl)
        if op in (C.GROUPREF, C.GROUPREF_EXISTS, C.GROUPREF_IGNORE, C.GROUPREF_LOC_IGNORE, C.GROUPREF_UNI_IGNORE):
            return INF
        return 1.0

    nodes = list(tree)
    cost = seq(nodes, False, gflags) * (n + 1)
    anchored = bool(nodes) and nodes[0][0] is C.AT and (
        nodes[0][1] is C.AT_BEGINNING_STRING or (nodes[0][1] is C.AT_BEGINNING and not gflags & re.MULTILINE))
    if how in ("search", "sub", "subn", "findall", "finditer", "split") and not anchored:
        cost *= (n + 1)          # every start position (a pattern anchored at the start fails at once elsewhere)
    return cost


def _deterministic(item, nxt, flags) -> bool:
    """Is a repeat of the single-character ``item`` followed by ``nxt`` free of backtracking into itself?"""
    if len(item) != 1 or item[0][0] not in (C.LITERAL, C.NOT_LITERAL, C.ANY, C.IN, C.CATEGORY):
        return False
    if nxt is None:
        return False
    op, av = nxt
    if op is C.AT and av in (C.AT_END, C.AT_END_STRING):
        return True
    if op is C.LITERAL:
        return not _member(chr(av), item[0], flags)
    return False


def _member(ch, node, flags) -> bool:
    op, av = node
    if op is C.LITERAL:
        return _same_char(ch, chr(av), flags)
    if op is C.NOT_LITERAL:
        return not _same_char(ch, chr(av), flags)
    if op is C.ANY:
        return ch != "\n" or bool(flags & re.DOTALL)
    if op is C.IN:
        return _in_class(ch, av, flags)
    if op is C.CATEGORY:
        return _in_category(ch, av, flags)
    return True


def _has_groupref(nodes) -> bool:
    for op, av in nodes:
        if op in (C.GROUPREF, C.GROUPREF_EXISTS, C.GROUPREF_IGNORE, C.GROUPREF_LOC_IGNORE, C.GROUPREF_UNI_IGNORE):
            return True
        if any(_has_groupref(s) for s in _children(op, av)):
            return True
    return False


@functools.lru_cache(maxsize=512)
def _compiled(pattern: str, flags: int):
    return re.compile(pattern, flags & ~re.DEBUG)


def _probe(call, seconds: float) -> bool:
    """Run ``call()`` in a forked child and report whether it finished (returned or raised) within ``seconds``.

    The child is SIGKILLed on the deadline and carries an ``RLIMIT_CPU`` backstop, so a catastrophic backtrack costs the
    deadline, never the parent.  Nothing crosses back: a ``True`` only says the very same call terminates quickly, and the
    caller then makes it for real in the parent.  No ``fork``, a failed fork or a deadline all answer ``False``."""
    if not hasattr(os, "fork"):
        return False
    try:
        pid = os.fork()
    except OSError:
        return False
    if pid == 0:                                    # child: never returns, never runs the parent's exit hooks
        code = 1
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (int(seconds) + 2, int(seconds) + 2))
            try:
                call()
            except BaseException:                   # noqa: BLE001 -- an exception is still a completed check
                pass
            code = 0
        finally:
            os._exit(code)
    deadline = time.monotonic() + seconds
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        if time.monotonic() >= deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            return False
        time.sleep(0.001)


def admitted(pattern: str, flags: int, n: int, how: str, call) -> bool:
    """May ``call()`` (a native regex check on a string of length ``n``) be made?

    Under :data:`WORK_LIMIT` the static estimate admits it.  ``inf`` (a backreference, an unbounded nested repeat past
    :data:`_NESTED_MAX_LEN`) is refused outright.  A finite estimate over the limit is only a worst case over ALL strings of
    that length -- ``HTML_RE`` is ~1e9 at 33 characters yet matches a real 257-character page in microseconds -- so the
    check is admitted when this very call finishes inside :data:`PROBE_SECONDS` in a killable child (:func:`_probe`);
    a timeout stays a refusal (the check is not made, never made wrongly)."""
    cost = check_cost(pattern, flags, n, how)
    if cost <= WORK_LIMIT:
        return True
    return cost != INF and _probe(call, PROBE_SECONDS)


def native_call(pattern, flags: int, method: str, value, *rest):
    """``re.compile(pattern, flags).<method>(value, *rest)`` with CPython, or the marker :data:`UNCHECKED` when the
    operands are not builtin (str pattern/value, int flags, str/int rest) or the work bound refuses the check.  Native
    exceptions propagate (a caller treats them as "not a preimage")."""
    if type(pattern) is not str or type(value) is not str or type(flags) not in (int, bool, re.RegexFlag):
        return UNCHECKED
    if flags & re.LOCALE or not all(type(x) in (str, int) for x in rest):
        return UNCHECKED
    run = lambda: getattr(_compiled(pattern, int(flags)), method)(value, *rest)
    if len(value) > 4096 or not admitted(pattern, int(flags), len(value), method, run):
        return UNCHECKED
    return run()


class _Unchecked:
    def __repr__(self):
        return "UNCHECKED"


UNCHECKED = _Unchecked()


def native_member(pattern, flags: int, how: str, value) -> bool | None:
    """Does ``re.compile(pattern, flags).<how>(value)`` match?  ``None`` when the check is not admitted."""
    try:
        got = native_call(pattern, flags, how, value)
    except (re.error, ValueError, TypeError, OverflowError, RecursionError):
        return None
    if got is UNCHECKED:
        return None
    return got is not None


# --------------------------------------------------------------------------- lengths

def length_range(nodes) -> tuple[int, float]:
    lo, hi = 0, 0.0
    for op, av in nodes:
        a, b = _node_len(op, av)
        lo += a
        hi += b
    return lo, hi


def min_match_length(pattern, flags: int = 0) -> int | None:
    """The length no match of *pattern* is shorter than (its parse tree's minimum: an anchor or a look-around counts 0,
    a literal/class/any 1, a repeat its minimum times its item), or ``None`` for a pattern that does not parse.  A lower
    bound only: unknown nodes count 0, so a positive answer is a proof and a 0 proves nothing."""
    if type(pattern) is not str:
        return None
    try:
        tree = _parse(pattern, int(flags))
    except (re.error, ValueError, OverflowError, RecursionError):
        return None
    return length_range(list(tree))[0]


def _node_len(op, av) -> tuple[int, float]:
    if op in (C.LITERAL, C.NOT_LITERAL, C.ANY, C.IN, C.CATEGORY):
        return 1, 1
    if op in (C.AT, C.ASSERT, C.ASSERT_NOT):
        return 0, 0
    if op is C.SUBPATTERN:
        return length_range(list(av[3]))
    if op is C.ATOMIC_GROUP:
        return length_range(list(av))
    if op is C.BRANCH:
        rs = [length_range(list(b)) for b in av[1]]
        return min(r[0] for r in rs), max(r[1] for r in rs)
    if op in _REPEATS:
        lo, hi, item = av
        a, b = length_range(list(item))
        top = INF if hi is C.MAXREPEAT else hi * b
        return lo * a, top
    return 0, INF


# --------------------------------------------------------------------------- characters

def _in_class(ch: str, items, flags: int) -> bool:
    negate = False
    hit = False
    for op, av in items:
        if op is C.NEGATE:
            negate = True
        elif op is C.LITERAL:
            hit = hit or _same_char(ch, chr(av), flags)
        elif op is C.RANGE:
            lo, hi = av
            hit = hit or lo <= ord(ch) <= hi or (flags & re.IGNORECASE and
                                                 any(len(c) == 1 and lo <= ord(c) <= hi for c in (ch.lower(), ch.upper())))
        elif op is C.CATEGORY:
            hit = hit or _in_category(ch, av, flags)
    return hit != negate


def _same_char(a: str, b: str, flags: int) -> bool:
    return a == b or bool(flags & re.IGNORECASE) and len(a.lower()) == 1 and a.lower() == b.lower()


def _in_category(ch: str, cat, flags: int) -> bool:
    ascii_only = bool(flags & re.ASCII)
    if ascii_only and not ch.isascii():
        return cat in (C.CATEGORY_NOT_DIGIT, C.CATEGORY_NOT_WORD, C.CATEGORY_NOT_SPACE)
    digit = ch.isdecimal()
    word = ch.isalnum() or ch == "_"
    space = ch.isspace() if not ascii_only else ch in " \t\n\r\x0b\x0c"
    return {C.CATEGORY_DIGIT: digit, C.CATEGORY_NOT_DIGIT: not digit, C.CATEGORY_WORD: word,
            C.CATEGORY_NOT_WORD: not word, C.CATEGORY_SPACE: space, C.CATEGORY_NOT_SPACE: not space}.get(cat, False)


def _freeze(av):
    if isinstance(av, list):
        return tuple(_freeze(x) for x in av)
    if isinstance(av, tuple):
        return tuple(_freeze(x) for x in av)
    return av


def _chars(op, av, flags: int, prefer: tuple = (), anyp: tuple = ()) -> list[str]:
    """Members of one single-character node, preferred first (``prefer`` first of all; ``anyp`` first for ``.`` only);
    memoized: a long exact-length member asks for the same class hundreds of times."""
    return list(_chars_cached(op, _freeze(av), int(flags), tuple(prefer), tuple(anyp)))


@functools.lru_cache(maxsize=4096)
def _chars_cached(op, av, flags: int, prefer: tuple, anyp: tuple) -> tuple:
    return tuple(_chars_uncached(op, av, flags, prefer, anyp))


def _chars_uncached(op, av, flags: int, prefer: tuple = (), anyp: tuple = ()) -> list[str]:
    if op is C.LITERAL:
        c = chr(av)
        out = [c]
        if flags & re.IGNORECASE and c.swapcase() != c:
            out.append(c.swapcase())
        return out
    if op is C.ANY:
        prefer = tuple(anyp) + tuple(prefer)
        test = (lambda ch: True) if flags & re.DOTALL else (lambda ch: ch != "\n")
    elif op is C.NOT_LITERAL:
        test = (lambda ch: not _same_char(ch, chr(av), flags))
    elif op is C.IN:
        test = (lambda ch: _in_class(ch, av, flags))
    elif op is C.CATEGORY:
        test = (lambda ch: _in_category(ch, av, flags))
    else:
        return []
    out = [ch for ch in prefer if test(ch)]
    out += [ch for ch in _PREF if test(ch) and ch not in out]
    if not out and op is C.IN:
        for iop, iav in av:          # a class of nothing we prefer: its own first member
            if iop is C.LITERAL and test(chr(iav)):
                out.append(chr(iav))
            elif iop is C.RANGE and test(chr(iav[0])):
                out.append(chr(iav[0]))
    return out


# --------------------------------------------------------------------------- shortest-first members

def _alts(nodes, flags: int, k: int, prefer: tuple, depth: int = 0, anyp: tuple = ()) -> list[str]:
    """Up to ``k`` strings of the sequence, shortest first (a bounded best-first product)."""
    if depth > 32:
        return []
    combos = [""]
    for op, av in nodes:
        opts = _node_alts(op, av, flags, k, prefer, depth, anyp)
        if not opts:
            return []
        merged = [(len(a) + len(b), i + j, a + b) for i, a in enumerate(combos) for j, b in enumerate(opts)]
        merged.sort(key=lambda t: (t[0], t[1]))
        seen, combos = set(), []
        for _, _, s in merged:
            if s not in seen:
                seen.add(s)
                combos.append(s)
                if len(combos) >= k:
                    break
    return combos


def _node_alts(op, av, flags, k, prefer, depth, anyp=()) -> list[str]:
    if op in (C.LITERAL, C.NOT_LITERAL, C.ANY, C.IN, C.CATEGORY):
        return _chars(op, av, flags, prefer, anyp)[:k]
    if op in (C.AT, C.ASSERT, C.ASSERT_NOT):
        return [""]
    if op is C.SUBPATTERN:
        add, rem = av[1], av[2]
        return _alts(list(av[3]), (flags | add) & ~rem, k, prefer, depth + 1, anyp)
    if op is C.ATOMIC_GROUP:
        return _alts(list(av), flags, k, prefer, depth + 1, anyp)
    if op is C.BRANCH:
        # every branch keeps its own simplest member (so a long alternative -- ``<!--.*-->`` next to ``<a>`` -- is not
        # truncated away), then the rest shortest first
        per = [_alts(list(b), flags, k, prefer, depth + 1, anyp) for b in av[1]]
        firsts = sorted(dict.fromkeys(p[0] for p in per if p), key=len)
        rest = sorted(dict.fromkeys(x for p in per for x in p[1:] if x not in firsts), key=len)
        return (firsts + rest)[:max(k, len(firsts))]
    if op in _REPEATS:
        lo, hi, item = av
        top = lo + _REPEAT_EXTRA if hi is C.MAXREPEAT else min(hi, lo + _REPEAT_EXTRA)
        inner = _alts(list(item), flags, k, prefer, depth + 1, anyp)
        if not inner:
            return [""] if lo == 0 else []
        out = []
        for count in range(lo, top + 1):
            out.append(inner[0] * count)
            if count and len(inner) > 1:
                out.append(inner[0] * (count - 1) + inner[1])
        return sorted(dict.fromkeys(out), key=len)[:k]
    return []


# --------------------------------------------------------------------------- exact-length members

def _build(nodes, flags: int, n: int, strategy: str, prefer: tuple, depth: int = 0) -> str | None:
    """One string of the sequence of exactly length ``n`` (or None), the slack spread by ``strategy``."""
    if depth > 32:
        return None
    ranges = [_node_len(op, av) for op, av in nodes]
    lo = sum(r[0] for r in ranges)
    hi = sum(r[1] for r in ranges)
    if not lo <= n <= hi:
        return None
    alloc = [r[0] for r in ranges]
    slack = n - lo
    order = list(range(len(nodes)))
    if strategy == "right":
        order.reverse()
    if strategy == "even":
        while slack:
            moved = False
            for i in order:
                if slack and alloc[i] < ranges[i][1]:
                    alloc[i] += 1
                    slack -= 1
                    moved = True
            if not moved:
                return None
    else:
        for i in order:
            take = min(slack, ranges[i][1] - alloc[i])
            take = int(take) if take != INF else slack
            alloc[i] += take
            slack -= take
    out = []
    for (op, av), m in zip(nodes, alloc):
        s = _build_node(op, av, flags, m, strategy, prefer, depth)
        if s is None:
            return None
        out.append(s)
    return "".join(out)


def _build_node(op, av, flags, m, strategy, prefer, depth) -> str | None:
    if op in (C.LITERAL, C.NOT_LITERAL, C.ANY, C.IN, C.CATEGORY):
        cs = _chars(op, av, flags, prefer)
        return cs[0] if m == 1 and cs else None
    if op in (C.AT, C.ASSERT, C.ASSERT_NOT):
        return "" if m == 0 else None
    if op is C.SUBPATTERN:
        return _build(list(av[3]), (flags | av[1]) & ~av[2], m, strategy, prefer, depth + 1)
    if op is C.ATOMIC_GROUP:
        return _build(list(av), flags, m, strategy, prefer, depth + 1)
    if op is C.BRANCH:
        for b in av[1]:
            s = _build(list(b), flags, m, strategy, prefer, depth + 1)
            if s is not None:
                return s
        return None
    if op in _REPEATS:
        lo, hi, item = av
        a, b = length_range(list(item))
        hi_count = 10**6 if hi is C.MAXREPEAT else hi
        if m == 0:
            return "" if lo == 0 else None
        if b == 0:
            return None
        # fewest repetitions whose item lengths can add up to m
        start = max(lo, 1, -(-m // int(b)) if b != INF else 1)
        for count in range(start, min(hi_count, m if a > 0 else m + 1) + 1):
            if count * a > m:
                break
            base, extra = divmod(m, count)
            parts = []
            for i in range(count):
                s = _build(list(item), flags, base + (1 if i < extra else 0), strategy, prefer, depth + 1)
                if s is None:
                    break
                parts.append(s)
            else:
                return "".join(parts)
        return None
    return None


def _segments(nodes, split: str):
    """The top-level sequence split at its literal ``split`` characters: ``[[nodes], split, [nodes], ...]``."""
    out, cur = [], []
    for op, av in nodes:
        if op is C.LITERAL and chr(av) == split:
            out.append(cur)
            out.append(chr(av))
            cur = []
        elif op is C.AT:
            continue
        else:
            cur.append((op, av))
    out.append(cur)
    return out


def samples(patterns):
    """A few strings per pattern that the pattern accepts (the program's own regexes and those of the code it
    reaches): corpus entries that start the fuzzer INSIDE the shapes the code checks for instead of
    waiting for it to find one.  Printable text only (what a ``nondet_str`` draw can hold).

    [internal ref]: the SHORTEST members, enumerated deterministically by ``solve/preimage/regex_members.py``
    (``a@aa.aa`` before anything longer), no longer Hypothesis' random draws -- Phase A measured a random sample's
    length deciding whether the m131-m133 kills happened (a 1-character quoted local part is one specific shape)."""
    import string
    import time
    deadline = time.monotonic() + REGEX_SAMPLE_SECONDS
    out = []
    for pattern, flags in patterns:
        if time.monotonic() > deadline:
            break
        try:
            got = list(members(pattern, flags, how="search", limit=REGEX_SAMPLES_DRAWN))
        except Exception:  # noqa: BLE001 -- a hint is optional
            continue
        got = [g for g in got if g and len(g) <= 128 and all(c in string.printable for c in g)]
        out += sorted(got, key=lambda v: (len(v), v))[:REGEX_SAMPLES_KEPT]
    return out


# --------------------------------------------------------------------------- the public generators

def members(pattern, flags: int = 0, *, how: str = "match", limit: int = 8, length: int | None = None,
            split: str | None = None, segments: tuple | None = None, prefer: tuple = ()) -> Iterator[str]:
    """Members of the pattern under ``how``, each confirmed by :func:`native_member`, shortest first.

    ``length``: only members of exactly that length (the slack spread left-first, right-first and evenly).
    ``split``/``segments``: exact lengths for the stretches of the top-level sequence between its literal ``split``
    characters, in order (``None`` entries are free): ``split="@", segments=(64, 255)``.  ``prefer``: characters tried first in every class (a program's own alphabet)."""
    if refusal(pattern, flags) is not None:
        return
    tree = list(_parse(pattern, int(flags)))
    gflags = int(flags) | _parse(pattern, int(flags)).state.flags
    seen = set()

    def ok(s):
        if s is None or s in seen:
            return False
        seen.add(s)
        return native_member(pattern, flags, "fullmatch" if how == "fullmatch" else how, s) is True

    count = 0
    if segments is not None:
        if type(split) is not str or len(split) != 1:
            return
        parts = _segments(tree, split)
        holes = [p for p in parts if not isinstance(p, str)]
        if len(holes) != len(segments):
            return
        for strategy in ("left", "even", "right"):
            out, it = [], iter(segments)
            for p in parts:
                if isinstance(p, str):
                    out.append(p)
                    continue
                want = next(it)
                s = (_alts(p, gflags, 1, prefer) or [None])[0] if want is None else _build(p, gflags, want, strategy, prefer)
                if s is None:
                    break
                out.append(s)
            else:
                cand = "".join(out)
                if ok(cand):
                    yield cand
                    count += 1
                    if count >= limit:
                        return
        return
    if length is not None:
        for strategy in ("left", "right", "even"):
            cand = _build(tree, gflags, length, strategy, prefer)
            if ok(cand):
                yield cand
                count += 1
                if count >= limit:
                    return
        return
    for cand in _alts(tree, gflags, max(limit * 2, _SEQ_ALTS), prefer):
        if ok(cand):
            yield cand
            count += 1
            if count >= limit:
                return


def flag_variants(pattern, flags_a: int, flags_b: int, *, limit: int = 8) -> Iterator[str]:
    """Strings built to exercise the nodes whose meaning differs between the two flag sets (DOTALL, IGNORECASE,
    MULTILINE, ASCII); NOT checked here -- a caller compares the two native results itself (``native_call``)."""
    if refusal(pattern, flags_a) is not None or refusal(pattern, flags_b) is not None:
        return
    diff = (int(flags_a) ^ int(flags_b)) & (re.DOTALL | re.IGNORECASE | re.MULTILINE | re.ASCII)
    if not diff:
        return
    seen = set()
    count = 0
    for base_flags in (int(flags_a), int(flags_b)):
        gflags = base_flags | _parse(pattern, base_flags).state.flags
        tree = list(_parse(pattern, base_flags))
        for cand in _variant_alts(tree, gflags, diff):
            if cand not in seen:
                seen.add(cand)
                yield cand
                count += 1
                if count >= limit:
                    return


def _plain(nodes):
    """The parse tree as plain lists/tuples (so a variant can be built by copying)."""
    out = []
    for op, av in nodes:
        if op is C.SUBPATTERN:
            out.append((op, (av[0], av[1], av[2], _plain(av[3]))))
        elif op is C.BRANCH:
            out.append((op, (av[0], [_plain(b) for b in av[1]])))
        elif op in _REPEATS:
            out.append((op, (av[0], av[1], _plain(av[2]))))
        elif op in (C.ASSERT, C.ASSERT_NOT):
            out.append((op, (av[0], _plain(av[1]))))
        elif op is C.ATOMIC_GROUP:
            out.append((op, _plain(av)))
        else:
            out.append((op, av))
    return out


def _force_newline(nodes, limit=16):
    """Copies of the tree in which exactly one ``.`` is the character ``'\\n'`` (and every repeat around it runs at least
    once): the members a DOTALL difference shows in."""
    out = []
    for i, (op, av) in enumerate(nodes):
        head, tail = nodes[:i], nodes[i + 1:]
        if op is C.ANY:
            out.append(head + [(C.LITERAL, 10)] + tail)
        elif op is C.SUBPATTERN:
            out += [head + [(op, (av[0], av[1], av[2], v))] + tail for v in _force_newline(av[3], limit)]
        elif op is C.BRANCH:
            for j, b in enumerate(av[1]):
                for v in _force_newline(b, limit):
                    bs = list(av[1])
                    bs[j] = v
                    out.append(head + [(op, (av[0], bs))] + tail)
        elif op in _REPEATS:
            out += [head + [(op, (max(av[0], 1), av[1], v))] + tail for v in _force_newline(av[2], limit)]
        elif op is C.ATOMIC_GROUP:
            out += [head + [(op, v)] + tail for v in _force_newline(av, limit)]
        if len(out) >= limit:
            break
    return out[:limit]


def _variant_alts(tree, flags, diff) -> list[str]:
    out = []
    if diff & re.DOTALL:
        for variant in _force_newline(_plain(tree)):
            out += [s for s in _alts(variant, flags | re.DOTALL, 8, ()) if "\n" in s][:2]
    if diff & re.IGNORECASE:
        for s in _alts(tree, flags | re.IGNORECASE, 4, ()):
            out.append(s.swapcase())
    if diff & re.MULTILINE:
        for s in _alts(tree, flags, 2, ()):
            out += ["x\n" + s, s + "\nx"]
    if diff & re.ASCII:
        out += [s for s in _alts(tree, flags & ~re.ASCII, 4, ("\u00e9", "\u0663")) if not s.isascii()]
    return out
