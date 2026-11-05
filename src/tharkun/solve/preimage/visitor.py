"""The preimage engine ([internal ref]): one more visitor over the existing contract rules.

``PreimageVisitor`` (engine ``"preimage"``) answers, for ONE call of a contracted callable whose operands are all known
builtin values except one -- the HOLE -- "which values of the hole make the call's result satisfy this :class:`Want`?".
Each ``visit_<kind>`` returns a :class:`Preimage`: an exact ``transfer`` (the demand on the hole, when the vocabulary can
state it: ``x.startswith('"')`` wanted true is ``Prefix('"')`` on ``x``) and a lazy stream of proposals.  The
:class:`Preimage` iterator FORWARD-CHECKS every proposal -- ``want.holds(native(*operands_with(proposal)))`` with the
rule's own native semantics (``Rule.native``, i.e. CPython) -- filters it by the caller's ``prior`` and yields only the
ones that pass, simplest first, at most ``budget`` of them.  A kind with no visit, a hole that is not supported for the
kind, or an operand that is not a builtin value of the kind the call takes is a NAMED refusal
(:class:`~tharkun.solve.contracts.rules.Refused`, engine ``"preimage"``), never a silent empty answer.

Soundness: a yielded value is a CANDIDATE.  The forward check proves ``want(f(candidate))`` for the rule's native
semantics on builtin operands; whether the program reaches the call with that value is for execution to judge (the
drive's verified trace, the SV fresh-child replay).  An empty iterator or a refusal says nothing about infeasibility.
Native calls run only on builtin values admitted by ``contracts/adapters/str_shapes.py`` kinds or
:func:`~tharkun.solve.preimage.want.is_builtin_value`, bounded by ``rules_strformat.py::native_limit`` (and, for the
``re`` rules, by ``regex_members.admitted``), so no target code, no user hook and no catastrophic backtrack runs here.

SOLVING by ``docs/LAYERS.md``: the input is a predicate (a Want) about the result of a named operation, the output
witnesses the caller verifies (corollaries (c), (d)).  stdlib and ``tharkun.solve`` only.
"""

from __future__ import annotations

import ast
import base64
import binascii
import functools
import itertools
import json
import re
import string
import unicodedata
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

from tharkun.solve.contracts.rules import Refused, Rule, Visitor
from tharkun.solve.preimage import regex_members as RM
from tharkun.solve.preimage.want import (ANY, All, AnyOf, CaseMap, Compare, Contains, Eq, IsNone, Items, Length, Member, Not,
                                         Prefix, Suffix, Truth, Want, is_builtin_value)

__all__ = ["HOLE", "PreimageCtx", "Preimage", "PreimageVisitor", "ENGINE", "SUPPORTED_KINDS"]

ENGINE = "preimage"


class _Hole:
    __slots__ = ()

    def __repr__(self) -> str:
        return "HOLE"


#: the marker of the unknown operand in :attr:`PreimageCtx.args`
HOLE = _Hole()

_FILLERS = "abxyz01"
_PRINTABLE = string.printable


@dataclass(frozen=True)
class PreimageCtx:
    """One call: ``args`` are the positional operands in call order (the receiver first for a method) with exactly one
    :data:`HOLE`; ``kwargs`` the keyword operands; ``want`` the demand on the result; ``prior`` what is already known of
    the hole (a filter: a proposal it rejects is not yielded); ``pool`` the program's own literals and guard tokens
    (tried first); ``op`` the operator symbol for ``protocol.binary_operator`` (``"+"`` / ``"*"``)."""

    want: Want
    args: tuple
    kwargs: tuple = ()
    prior: Want = ANY
    pool: tuple = ()
    budget: int = 64
    op: str | None = None

    @property
    def hole(self) -> int:
        idx = [i for i, a in enumerate(self.args) if a is HOLE]
        if len(idx) != 1:
            raise ValueError("a preimage context needs exactly one HOLE operand")
        return idx[0]

    def known(self, i: int) -> Any:
        return self.args[i]

    def with_value(self, value: Any) -> tuple:
        return tuple(value if a is HOLE else a for a in self.args)


class Preimage:
    """An exact ``transfer`` (or None) plus the forward-checked candidates of the hole (iterate)."""

    def __init__(self, qualname: str, ctx: PreimageCtx, transfer: Want | None,
                 proposals: Iterable[Any], native: Callable[..., Any]):
        self.qualname = qualname
        self.ctx = ctx
        self.transfer = transfer
        self._proposals = proposals
        self._native = native

    def __repr__(self) -> str:
        return f"<Preimage {self.qualname} transfer={self.transfer!r}>"

    def check(self, value: Any) -> bool:
        """``want(f(value))`` with the native semantics, on builtin operands, within the native bounds."""
        if not is_builtin_value(value):
            return False
        try:
            return self.ctx.want.holds(self._native(*self.ctx.with_value(value), **dict(self.ctx.kwargs)))
        except (TypeError, ValueError, KeyError, IndexError, AttributeError, OverflowError, re.error, RecursionError,
                LookupError, UnicodeError, MemoryError, _NotChecked):
            return False

    def __iter__(self) -> Iterator[Any]:
        seen: set = set()
        n = 0
        for value in self._proposals:
            key = (type(value), value) if _hashable(value) else (type(value), repr(value))
            if key in seen:
                continue
            seen.add(key)
            if not self.ctx.prior.holds(value) or not self.check(value):
                continue
            yield value
            n += 1
            if n >= self.ctx.budget:
                return


def _hashable(v) -> bool:
    try:
        hash(v)
        return True
    except TypeError:
        return False


#: the builtin codecs and error handlers ``bytes.decode`` is inverted for
_DECODE_CODECS = frozenset({"utf-8", "utf8", "ascii", "latin-1", "latin1", "iso-8859-1", "utf-16", "utf-16-le", "utf-16-be",
                            "utf-32", "utf-32-le", "utf-32-be"})
_DECODE_ERRORS = frozenset({"strict", "ignore", "replace", "backslashreplace", "surrogateescape", "surrogatepass"})
#: the canonical exact builtin values per class ``visit_isinstance`` proposes, simplest first (bool before int: ``isinstance(True,
#: int)`` is true, so the forward check decides each)
_ISINSTANCE_CANONICAL = {
    type(None): [None],
    bool: [False, True],
    int: [0, 1, -1, 2, 10 ** 20],
    float: [0.0, 1.5, -2.5],
    str: ["", "a", " ", "ab1"],
    bytes: [b"", b"a"],
    list: [[], [0], ["a"]],
    tuple: [(), (0,)],
    dict: [{}, {"a": 0}],
    object: [None, 0, ""],
}


def _flatten_classes(spec, depth=0):
    """The classes in an ``isinstance`` second argument (a class, a nested tuple, or ``X | Y``), or None when it is anything else."""
    import types
    if depth > 3:
        return None
    if type(spec) is type:
        return [spec]
    if spec is None:
        return None
    if type(spec) is types.UnionType:
        parts = spec.__args__
    elif type(spec) is tuple:
        parts = spec
    else:
        return None
    out = []
    for part in parts:
        got = [type(None)] if part is type(None) else _flatten_classes(part, depth + 1)
        if got is None:
            return None
        out.extend(got)
    return out


#: handlers under which ``str.encode`` loses characters the observation then cannot show
_LOSSY_ENCODE_ERRORS = ("ignore", "replace", "xmlcharrefreplace", "backslashreplace")
#: characters a codec may not encode, canonical and varied: a CJK letter and a Cyrillic letter (\w, stable under NFKD/NFKC), ss,
#: an accented letter, a currency symbol (not \w), a lone surrogate (the only thing utf-8 cannot encode), an astral emoji
_UNENCODABLE = ("\u96ea", "\u0436", "\u00df", "\u00e9", "\u20ac", "\udc80", "\U0001f600")


def _lossy_encode_sources(text: str, encoding: str, errors: str):
    """Texts that ``text.encode(encoding, errors)`` writes as ``text`` does, though they hold characters the codec cannot
    encode, built from a small canonical set (``_UNENCODABLE``) and forward-checked by the caller.

    ``ignore`` drops them, so one is INSERTED (start, end, then the middle positions up to 32); ``replace`` writes the codec's
    marker (``?``), so occurrences of the marker are turned into the character (all of them, then each one); ``xmlcharrefreplace``
    and ``backslashreplace`` write ``&#N;`` / ``\\uXXXX``-style escapes, so those are decoded back to the character. Which of
    the original characters were really there is not recoverable from the bytes: these are some preimages, not all of them."""
    unencodable = []
    for c in _UNENCODABLE:
        try:
            c.encode(encoding)
        except UnicodeEncodeError:
            unencodable.append(c)
    if errors == "ignore":
        spots = [0, len(text), *range(1, min(len(text), 32))]
        for c in unencodable[:4]:
            for i in dict.fromkeys(spots):
                yield text[:i] + c + text[i:]
    elif errors == "replace":
        for c in unencodable[:4]:
            marker = c.encode(encoding, "replace").decode(encoding)
            if marker and marker in text:
                yield text.replace(marker, c)
                for i in [m.start() for m in re.finditer(re.escape(marker), text)][:16]:
                    yield text[:i] + c + text[i + len(marker):]
    else:
        pattern = (r"&#(\d+);" if errors == "xmlcharrefreplace" else r"\\x([0-9a-fA-F]{2})|\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})")

        def back(m):
            if errors == "xmlcharrefreplace":
                return chr(int(m.group(1)))
            return chr(int(next(g for g in m.groups() if g), 16))
        try:
            decoded = re.sub(pattern, back, text)
        except (ValueError, OverflowError):
            return
        if decoded != text:
            yield decoded


def _same_json(a, b) -> bool:
    """Equal AND of the same types all the way down (``1`` is not ``1.0`` is not ``True``; NaN matches NaN)."""
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return list(a) == list(b) and all(_same_json(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(map(_same_json, a, b))
    if type(a) is float:
        return a == b or (a != a and b != b)
    return a == b


@functools.lru_cache(maxsize=4)
def _normalization_sources(form: str) -> dict:
    """``{normalized piece of 1-3 characters: [code points that normalize to it, shortest first]}`` for ``form`` over every
    code point (one scan of the Unicode table, cached per form)."""
    sources: dict = {}
    for cp in range(0x110000):
        ch = chr(cp)
        got = unicodedata.normalize(form, ch)
        if got != ch and len(got) <= 3:
            sources.setdefault(got, []).append(ch)
    return sources


#: longest data (bytes) a zlib/base64 preimage proposes or inflates: a hostile stream cannot expand past it
_CODEC_MAX = 1 << 24


def _inflate(stream: bytes, wbits: int) -> bytes:
    """``zlib.decompress(stream, wbits)`` refusing an output past :data:`_CODEC_MAX` (a decompression bomb)."""
    inflater = zlib.decompressobj(wbits)
    out = inflater.decompress(stream, _CODEC_MAX)
    if inflater.unconsumed_tail or not inflater.eof:
        raise ValueError("stream is truncated or expands past the codec bound")
    return out


class _NotChecked(Exception):
    """A native call the bounds do not admit (no forward check, so no candidate)."""


# --------------------------------------------------------------------------- operand admission

def _kind_ok(kind: str, value: Any) -> bool:
    if kind == "f":
        return type(value) is str and len(value) == 1
    if kind == "s":
        return type(value) is str
    if kind == "S":
        return type(value) is str or (type(value) is tuple and all(type(x) is str for x in value))
    if kind == "c":
        return value is None or type(value) is str
    if kind == "i":
        return type(value) in (int, bool)
    if kind == "n":
        return value is None or type(value) in (int, bool)
    if kind == "L":
        return type(value) in (list, tuple) and all(type(x) is str for x in value)
    return False


def _admit(rule: Rule, ctx: PreimageCtx) -> str | None:
    """Why the operands are outside what a native forward check may run on, or None."""
    from tharkun.solve.contracts.adapters.str_shapes import SIGNATURES, bind
    try:
        hole = ctx.hole
    except ValueError as exc:
        return str(exc)
    others = [a for i, a in enumerate(ctx.args) if i != hole] + [v for _, v in ctx.kwargs]
    q = rule.qualname
    if q.startswith("str.") and q[4:] in SIGNATURES and hole == 0:
        bound = bind(q[4:], ctx.args[1:], dict(ctx.kwargs))
        if bound is None:
            return "the call does not bind to the method's CPython 3.14 signature"
        if not all(_kind_ok(kind, value) for kind, value in bound.values()):
            return "an operand is not of the builtin kind the method takes (a user hook could run)"
        return None
    for v in others:
        if not (is_builtin_value(v) or type(v) is re.Pattern or type(v) is re.RegexFlag or v is None or type(v) is slice
                or (type(v) is dict and len(v) <= 64 and all(is_builtin_value(k) and is_builtin_value(x)
                                                              for k, x in v.items()))):
            return "an operand is not a builtin value (a user hook could run)"
        if type(v) is slice and not all(x is None or type(x) is int for x in (v.start, v.stop, v.step)):
            return "a slice bound is not an int"
    return None


def _bounded_native(rule: Rule) -> Callable[..., Any]:
    from tharkun.solve.contracts.rules_strformat import native_limit

    def call(*args, **kwargs):
        if native_limit(rule.qualname, args, kwargs):
            raise _NotChecked()
        return rule.native(*args, **kwargs)
    return call


# --------------------------------------------------------------------------- shared proposal helpers

def _filler(avoid: str = "") -> str:
    return next((c for c in _FILLERS if c not in avoid), "a")


def _str_seeds(ctx: PreimageCtx, want: Want | None = None) -> list[str]:
    """Strings that satisfy the want by construction, then the pool, then a few defaults: the starting points."""
    w = ctx.want if want is None else want
    out: list[str] = []
    for s in itertools.islice(w.seeds(), 32):
        if type(s) is str and s not in out:
            out.append(s)
    for p in ctx.pool:
        if type(p) is str and p not in out:
            out.append(p)
    for d in ("", "a", "ab", "a@b.cd", " ", "0"):
        if d not in out:
            out.append(d)
    return out


def _int_seeds(w: Want) -> list[int]:
    out: list[int] = []
    for s in itertools.islice(w.seeds(), 16):
        if type(s) is int and s not in out:
            out.append(s)
    if isinstance(w, Compare) and type(w.const) is int:
        c = w.const
        out += [v for v in (c + 1, c - 1, c, c + 2, 0, 1) if v not in out]
    elif isinstance(w, Truth):
        out += [v for v in ((1, 2) if w.flag else (0,)) if v not in out]
    elif isinstance(w, All):
        for p in w.parts:
            out += [v for v in _int_seeds(p) if v not in out]
    out += [v for v in (0, 1, 2, -1) if v not in out]
    return out


def _lengths(w: Want) -> list[int]:
    """Lengths a Length/Eq/Compare want on a length suggests (boundaries first)."""
    if isinstance(w, Length):
        lo = w.lo
        return [lo, lo + 1] + ([w.hi] if w.hi is not None and w.hi != lo else [])
    if isinstance(w, All):
        out = []
        for p in w.parts:
            out += [n for n in _lengths(p) if n not in out]
        return out
    return [n for n in _int_seeds(w) if n >= 0]


def _case_variants(s: str) -> list[str]:
    return [s, s.lower(), s.upper(), s.swapcase(), s.title(), s.capitalize()]


# --------------------------------------------------------------------------- the visitor

class PreimageVisitor(Visitor):
    """One visit per rule kind of the str family, the formatting rules, ``len``/``ord``/``chr``/``set``/``str``/
    ``repr``/``ascii``/``format``, str ``+``/``*`` (``protocol.binary_operator``) and the ``re`` module, ``Pattern`` and
    ``Match`` rules.  Everything else is the base class's named refusal."""

    engine = ENGINE

    def _refuse(self, rule: Rule, reason: str) -> Refused:
        return Refused(self.engine, rule.qualname, reason)

    def _pre(self, rule: Rule, ctx: PreimageCtx, proposals, transfer=None, native=None):
        bad = _admit(rule, ctx)
        if bad is not None:
            return self._refuse(rule, bad)
        return Preimage(rule.qualname, ctx, transfer, proposals, native or _bounded_native(rule))

    def _receiver_only(self, rule: Rule, ctx: PreimageCtx) -> Refused | None:
        try:
            hole = ctx.hole
        except ValueError as exc:
            return self._refuse(rule, str(exc))
        if hole != 0:
            return self._refuse(rule, "only the receiver may be the unknown operand of this method")
        return None

    # ---- case maps ------------------------------------------------------------------------------------------------
    def _case(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad

        def props():
            seeds = _str_seeds(ctx)
            yield from seeds                      # every seed itself first (simplest first), then its variants
            for s in seeds:
                yield from _case_variants(s)
        return self._pre(rule, ctx, props(), CaseMap(rule.qualname.split(".")[-1], ctx.want))

    visit_str_lower = visit_str_upper = visit_str_casefold = _case
    visit_str_swapcase = visit_str_title = visit_str_capitalize = _case

    # ---- predicates on the receiver -------------------------------------------------------------------------------
    _PRED_TRUE = {
        "isalnum": ("a", "0", "a0"), "isalpha": ("a", "ab"), "isascii": ("", "a"), "isdecimal": ("0", "12"),
        "isdigit": ("0", "12"), "isidentifier": ("a", "_a"), "islower": ("a", "ab"), "isnumeric": ("0", "12"),
        "isprintable": ("", "a"), "isspace": (" ", "\t"), "istitle": ("A", "Ab"), "isupper": ("A", "AB"),
    }
    _PRED_FALSE = ("", "a", " ", "A", "0", "a b", "\x00", "é")

    def _pred(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        method = rule.qualname[4:]

        def props():
            yield from self._PRED_TRUE.get(method, ())
            yield from self._PRED_FALSE
            yield from (p for p in ctx.pool if type(p) is str)
        return self._pre(rule, ctx, props())

    visit_str_isalnum = visit_str_isalpha = visit_str_isascii = visit_str_isdecimal = _pred
    visit_str_isdigit = visit_str_isidentifier = visit_str_islower = visit_str_isnumeric = _pred
    visit_str_isprintable = visit_str_isspace = visit_str_istitle = visit_str_isupper = _pred

    def _affix(self, rule: Rule, ctx: PreimageCtx):
        """startswith / endswith: the prefix (suffix) itself, padded to a start/end, and the pool around it."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        start = rule.qualname.endswith("startswith")
        affixes = ctx.args[1] if len(ctx.args) > 1 else None
        affixes = affixes if type(affixes) is tuple else (affixes,)
        bounds = ctx.args[2:]
        transfer = None
        if not bounds and not ctx.kwargs and all(type(a) is str for a in affixes):
            hit = AnyOf(tuple(Prefix(a) if start else Suffix(a) for a in affixes))
            hit = hit.parts[0] if len(hit.parts) == 1 else hit
            if ctx.want == Truth(True) or ctx.want == Eq(True):
                transfer = hit
            elif ctx.want == Truth(False) or ctx.want == Eq(False):
                transfer = Not(hit)

        def props():
            pad = _filler("".join(a for a in affixes if type(a) is str))
            lead = bounds[0] if bounds and type(bounds[0]) is int and 0 <= bounds[0] <= 64 else 0
            for a in affixes:
                if type(a) is not str:
                    continue
                yield pad * lead + a
                for p in _str_seeds(ctx, ANY):
                    yield pad * lead + (a + p if start else p + a)
            yield from _str_seeds(ctx, ANY)
        return self._pre(rule, ctx, props(), transfer)

    visit_str_startswith = visit_str_endswith = _affix

    def visit_str_contains(self, rule: Rule, ctx: PreimageCtx):
        """``sub in x`` is ``str.__contains__(x, sub)``: the receiver (container) or the item may be the hole."""
        try:
            hole = ctx.hole
        except ValueError as exc:
            return self._refuse(rule, str(exc))
        other = ctx.args[1 - hole] if len(ctx.args) == 2 else None
        if type(other) is not str:
            return self._refuse(rule, "the other operand of `in` is not a str")
        transfer = None
        if hole == 0 and ctx.want in (Truth(True), Eq(True)):
            transfer = Contains(other)
        elif hole == 0 and ctx.want in (Truth(False), Eq(False)):
            transfer = Not(Contains(other))

        def props():
            if hole == 0:
                yield other
                pad = _filler(other)
                yield pad + other + pad
                yield from _str_seeds(ctx, ANY)
                yield pad
            else:
                for i in range(len(other) + 1):
                    for j in range(i, min(len(other), i + 8) + 1):
                        yield other[i:j]
                yield _filler(other)
        return self._pre(rule, ctx, props(), transfer)

    # ---- search counts ----------------------------------------------------------------------------------------------
    def _find(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        sub = ctx.args[1] if len(ctx.args) > 1 else None
        if type(sub) is not str:
            return self._refuse(rule, "the substring is not a str")
        counting = rule.qualname.endswith("count")
        pad = _filler(sub)

        def props():
            for k in _int_seeds(ctx.want):
                if counting:
                    if 0 <= k <= 64:
                        yield (sub + pad) * k if sub else pad * max(k - 1, 0)
                elif k >= 0:
                    if k <= 4096:
                        yield pad * k + sub
                        yield pad * k + sub + pad
                else:
                    yield pad
                    yield ""
            yield from _str_seeds(ctx, ANY)
        return self._pre(rule, ctx, props())

    visit_str_find = visit_str_rfind = visit_str_index = visit_str_rindex = visit_str_count = _find

    # ---- split / partition / splitlines ----------------------------------------------------------------------------
    def _parts_for(self, ctx: PreimageCtx, sep: str | None, n_default: int = 2) -> Iterator[list[str]]:
        """Part lists the Items/Length/Eq want on a list result suggests (each part sep-free)."""
        w = ctx.want
        avoid = sep if sep else " \t\n\r\x0b\x0c"
        pad = _filler(avoid)
        counts: list[int] = []
        item_wants: dict[int, Want] = {}
        if isinstance(w, Items):
            if w.length is not None:
                counts = [n for n in _int_seeds(w.length) if 1 <= n <= 64]
            item_wants = dict(w.items)
        elif isinstance(w, Eq) and type(w.value) in (list, tuple):
            yield list(w.value)
            return
        elif isinstance(w, Length):
            counts = [n for n in _lengths(w) if 1 <= n <= 64]
        counts = counts or [n_default, 1, 3]

        def part_options(i: int, n: int) -> list[str]:
            iw = item_wants.get(i) or item_wants.get(i - n) or ANY
            opts = [s for s in _str_seeds(ctx, iw) if not (sep and sep in s) and (sep or not any(c in s for c in avoid))]
            opts = [s for s in opts if sep or s]   # whitespace split drops empty parts
            return opts[:6] or [pad]

        for n in counts:
            options = [part_options(i, n) for i in range(n)]
            # the first option of every part, then one part varied at a time
            base = [o[0] for o in options]
            yield list(base)
            for i, o in enumerate(options):
                for alt in o[1:]:
                    yield base[:i] + [alt] + base[i + 1:]

    def _split(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        args = dict(ctx.kwargs)
        sep = args.get("sep", ctx.args[1] if len(ctx.args) > 1 else None)
        if sep is not None and (type(sep) is not str or not sep):
            return self._refuse(rule, "the separator is not a non-empty str")

        def props():
            for parts in self._parts_for(ctx, sep):
                yield (sep if sep is not None else " ").join(parts)
            yield from _str_seeds(ctx, ANY)
        return self._pre(rule, ctx, props())

    visit_str_split = visit_str_rsplit = _split

    def visit_str_splitlines(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad

        def props():
            for parts in self._parts_for(ctx, "\n"):
                yield "\n".join(parts)
            yield from _str_seeds(ctx, ANY)
        return self._pre(rule, ctx, props())

    def _partition(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        sep = ctx.args[1] if len(ctx.args) > 1 else None
        if type(sep) is not str or not sep:
            return self._refuse(rule, "the separator is not a non-empty str")
        w = ctx.want
        heads: list[str] = []
        tails: list[str] = []
        if isinstance(w, Items):
            iw = dict(w.items)
            heads = _str_seeds(ctx, iw.get(0, ANY))[:6]
            tails = _str_seeds(ctx, iw.get(2, iw.get(-1, ANY)))[:6]
        elif isinstance(w, Eq) and type(w.value) is tuple and len(w.value) == 3:
            heads, tails = [w.value[0]], [w.value[2]]

        def props():
            for h in heads or ["a"]:
                for t in tails or ["b"]:
                    yield h + sep + t
            yield from _str_seeds(ctx, ANY)
        return self._pre(rule, ctx, props())

    visit_str_partition = visit_str_rpartition = _partition

    # ---- strip / remove / pad ---------------------------------------------------------------------------------------
    def _strip(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        chars = ctx.args[1] if len(ctx.args) > 1 else None
        pads = list(chars[:2]) if type(chars) is str and chars else [" ", "\t"]
        which = rule.qualname[4:]

        def props():
            seeds = _str_seeds(ctx)
            yield from seeds
            for s in seeds:
                for p in pads:
                    if which != "rstrip":
                        yield p + s
                    if which != "lstrip":
                        yield s + p
                    if which == "strip":
                        yield p + s + p
        return self._pre(rule, ctx, props())

    visit_str_strip = visit_str_lstrip = visit_str_rstrip = _strip

    def _remove(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        affix = ctx.args[1] if len(ctx.args) > 1 else None
        if type(affix) is not str:
            return self._refuse(rule, "the affix is not a str")
        prefix = rule.qualname.endswith("removeprefix")

        def props():
            seeds = _str_seeds(ctx)
            yield from (((affix + s) if prefix else (s + affix)) for s in seeds)
            yield from seeds
        return self._pre(rule, ctx, props())

    visit_str_removeprefix = visit_str_removesuffix = _remove

    def _pad(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        which = rule.qualname[4:]
        fill = ctx.args[2] if len(ctx.args) > 2 and type(ctx.args[2]) is str else ("0" if which == "zfill" else " ")

        def props():
            seeds = _str_seeds(ctx)
            yield from seeds
            for s in seeds:
                yield s.strip(fill)
                yield s.lstrip(fill)
                yield s.rstrip(fill)
                if which == "expandtabs":
                    yield re.sub(" {2,}", "\t", s)
        return self._pre(rule, ctx, props())

    visit_str_center = visit_str_ljust = visit_str_rjust = visit_str_zfill = visit_str_expandtabs = _pad

    # ---- replace -------------------------------------------------------------------------------------------------
    def visit_str_replace(self, rule: Rule, ctx: PreimageCtx):
        """Un-replace (K2's lesson): every occurrence of ``new`` back to ``old``, each one alone, and for a deletion
        ``old`` re-inserted at each boundary; the seed itself (no occurrence at all) too."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        old = ctx.args[1] if len(ctx.args) > 1 else None
        new = ctx.args[2] if len(ctx.args) > 2 else None
        if type(old) is not str or type(new) is not str:
            return self._refuse(rule, "old/new are not str")

        def props():
            seeds = _str_seeds(ctx)
            yield from seeds                      # no occurrence at all: the simplest preimage, for every seed first
            for s in seeds:
                if new:
                    yield s.replace(new, old)
                    i = s.find(new)
                    while i != -1 and i < 4096:
                        yield s[:i] + old + s[i + len(new):]
                        i = s.find(new, i + 1)
                elif old:
                    for i in range(min(len(s), 64) + 1):
                        yield s[:i] + old + s[i:]
        return self._pre(rule, ctx, props())

    # ---- indexing and slicing -------------------------------------------------------------------------------------
    def visit_str_getitem(self, rule: Rule, ctx: PreimageCtx):
        """``x[k]`` / ``x[a:b:c]``: for a wanted result ``v``, a receiver of each plausible length ``n`` whose selected
        positions spell ``v`` and whose free positions are filler (or come from a pool string)."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        key = ctx.args[1] if len(ctx.args) > 1 else None
        if type(key) not in (int, slice):
            return self._refuse(rule, "the key is not an int or a slice of ints")

        def build(v: str, n: int, pad: str) -> str | None:
            positions = range(*key.indices(n)) if type(key) is slice else [key % n] if n and -n <= key < n else None
            if positions is None or len(positions) != len(v):
                return None
            cells = [None] * n
            for pos, ch in zip(positions, v):
                cells[pos] = ch
            return "".join(c if c is not None else pad for c in cells)

        # the free positions take the filler, or one of the program's own one-character literals (a quote around a
        # sliced-off local part: ``'"' + v + '"'`` for ``x[1:-1] == v``)
        pads = list(dict.fromkeys([p for p in ctx.pool if type(p) is str and len(p) == 1][:4]))

        def props():
            for v in _str_seeds(ctx):
                lens = [len(v) + d for d in range(0, 6)] + [len(v) + 8, 2 * len(v) + 2]
                if type(key) is int:
                    lens = [max(key + 1, 1), -key] + lens
                for n in lens:
                    if 0 <= n <= 4096 and (type(key) is slice or len(v) == 1):
                        for pad in [_filler(v), *pads]:
                            got = build(v, n, pad)
                            if got is not None:
                                yield got
        return self._pre(rule, ctx, props())

    def _sequence_item(self, rule: Rule, ctx: PreimageCtx):
        """``seq[i]`` of a list/tuple result: builtin element seeds construct receiver candidates; the demand becomes
        an exact :class:`Items` demand on the sequence, which the producer of the sequence (``split``, ``partition``)
        then inverts."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        key = ctx.args[1] if len(ctx.args) > 1 else None
        if type(key) is not int:
            return self._refuse(rule, "only an int index is inverted")
        transfer = Items(None, ((key, ctx.want),))
        kind = list if rule.qualname.startswith("list.") else tuple

        def props():
            n = key + 1 if key >= 0 else -key
            if n > 64:  # rules_strformat.native_limit admits at most 64 list/tuple elements.
                return
            seeds = itertools.chain(itertools.islice(ctx.want.seeds(), 32), _str_seeds(ctx))
            for v in seeds:
                if not is_builtin_value(v):
                    continue
                cells = ["a"] * n
                cells[key] = v
                yield kind(cells)
        return self._pre(rule, ctx, props(), transfer)

    visit_list_getitem = visit_tuple_getitem = _sequence_item

    # ---- join and the formatting rules ----------------------------------------------------------------------------
    def visit_str_join(self, rule: Rule, ctx: PreimageCtx):
        try:
            hole = ctx.hole
        except ValueError as exc:
            return self._refuse(rule, str(exc))
        if hole != 1:
            return self._refuse(rule, "only the iterable of join may be the unknown operand")
        sep = ctx.args[0]
        if type(sep) is not str:
            return self._refuse(rule, "the separator is not a str")

        def props():
            for v in _str_seeds(ctx):
                yield [v]
                if sep:
                    yield v.split(sep)
                else:
                    yield list(v)
        return self._pre(rule, ctx, props())

    def _extract(self, template: str, parts: list, v: str) -> list[str]:
        """Values of the one hole field in ``template`` (literal text around ``%s``/``{}`` fields, the other fields'
        known values filled in) that render to ``v`` exactly, then the values that put ``v`` (minus the literal text it
        overlaps) next to the literal text -- for a Prefix/Suffix/Contains demand, whose seed is only a piece of the
        result."""
        rx = "".join(re.escape(p) if isinstance(p, str) else "(.*)" for p in parts)
        m = re.fullmatch(rx, v, re.DOTALL)
        out = [m.group(1)] if m else []
        k = parts.index(None)
        pre, post = "".join(parts[:k]), "".join(parts[k + 1:])
        out += _around(pre, post, v)
        return out

    def visit_str_mod(self, rule: Rule, ctx: PreimageCtx):
        """``fmt % value`` with only ``%s`` conversions and the value (or one tuple element) the hole."""
        try:
            hole = ctx.hole
        except ValueError as exc:
            return self._refuse(rule, str(exc))
        fmt = ctx.args[0]
        if hole != 1 or type(fmt) is not str:
            return self._refuse(rule, "only a str value formatted by a known template may be the unknown operand")
        pieces = re.split(r"(%s)", fmt.replace("%%", "\x00"))
        if any("%" in p for p in pieces if p != "%s") or pieces.count("%s") != 1:
            return self._refuse(rule, "the template has a conversion other than one %s")
        parts = [None if p == "%s" else p.replace("\x00", "%") for p in pieces if p != ""]

        def props():
            for v in _str_seeds(ctx):
                yield from self._extract(fmt, parts, v)
        return self._pre(rule, ctx, props())

    def visit_str_format(self, rule: Rule, ctx: PreimageCtx):
        """``template.format(*args)`` with plain ``{}`` / ``{n}`` fields only and one positional argument the hole."""
        try:
            hole = ctx.hole
        except ValueError as exc:
            return self._refuse(rule, str(exc))
        fmt = ctx.args[0]
        if hole == 0 or type(fmt) is not str or ctx.kwargs:
            return self._refuse(rule, "only a positional argument of a known template may be the unknown operand")
        try:
            fields = list(string.Formatter().parse(fmt))
        except ValueError:
            return self._refuse(rule, "the template does not parse")
        parts, auto = [], 0
        for literal, name, spec, conv in fields:
            if literal:
                parts.append(literal)
            if name is None:
                continue
            if spec or conv or (name and not name.isdigit()):
                return self._refuse(rule, "a field has a spec, a conversion or an attribute/item name")
            idx = int(name) if name else auto
            auto += 1
            if idx + 1 == hole:
                parts.append(None)
            elif 0 <= idx + 1 < len(ctx.args) and type(ctx.args[idx + 1]) is str:
                parts.append(ctx.args[idx + 1])
            else:
                return self._refuse(rule, "a known field is not a str")
        if parts.count(None) != 1:
            return self._refuse(rule, "the unknown argument is not rendered exactly once")

        def props():
            for v in _str_seeds(ctx):
                yield from self._extract(fmt, parts, v)
        return self._pre(rule, ctx, props())

    def visit_joined_str(self, rule: Rule, ctx: PreimageCtx):
        """An f-string's concatenation of already formatted parts, one of them the hole."""
        try:
            hole = ctx.hole
        except ValueError as exc:
            return self._refuse(rule, str(exc))
        if not all(type(a) is str for i, a in enumerate(ctx.args) if i != hole):
            return self._refuse(rule, "a known part is not a str")
        head = "".join(ctx.args[:hole])
        tail = "".join(ctx.args[hole + 1:])

        def props():
            for v in _str_seeds(ctx):
                if v.startswith(head) and v.endswith(tail) and len(v) >= len(head) + len(tail):
                    yield v[len(head):len(v) - len(tail)]
                yield from _around(head, tail, v)
        return self._pre(rule, ctx, props())

    def _identity_str(self, rule: Rule, ctx: PreimageCtx):
        """``format(x)`` / ``f'{x}'`` with an empty spec and no or ``!s`` conversion, and ``str(x)``: on a str hole the
        identity; a numeric text also proposes the int."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        rest = ctx.args[1:]
        if rule.qualname == "format" and rest not in ((), ("",)):
            return self._refuse(rule, "a non-empty format spec is not inverted")
        if rule.qualname == "syntax.FormattedValue" and not (rest in ((), ("",), ("", -1), ("", 115))):
            return self._refuse(rule, "a spec or a !r/!a conversion is not inverted")
        if rule.qualname == "str" and (rest or ctx.kwargs):
            return self._refuse(rule, "str() with an encoding is not inverted")

        def props():
            for v in _str_seeds(ctx):
                yield v
                if v.lstrip("-").isdigit() and len(v) < 32:
                    yield int(v)
        return self._pre(rule, ctx, props())

    visit_format = visit_formatted_value = visit_str = _identity_str

    def _literal_repr(self, rule: Rule, ctx: PreimageCtx):
        """``repr(x)`` / ``ascii(x)``: a wanted text that is a Python literal is the literal's value
        (``ast.literal_eval``: parses, never executes)."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad

        def props():
            for v in _str_seeds(ctx):
                if len(v) <= 4096:
                    try:
                        yield ast.literal_eval(v)
                    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                        pass
        return self._pre(rule, ctx, props())

    visit_repr = visit_ascii = _literal_repr

    def visit_str_encode(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        enc = ctx.args[1] if len(ctx.args) > 1 else dict(ctx.kwargs).get("encoding", "utf-8")
        if enc not in ("utf-8", "ascii", "latin-1"):
            return self._refuse(rule, "only the utf-8, ascii and latin-1 codecs are inverted")

        errors = ctx.args[2] if len(ctx.args) > 2 else dict(ctx.kwargs).get("errors", "strict")

        def props():
            for b in itertools.islice(ctx.want.seeds(), 16):
                if type(b) is bytes:
                    try:
                        text = b.decode(enc)
                    except UnicodeDecodeError:
                        continue
                    yield text
                    if errors in _LOSSY_ENCODE_ERRORS:
                        yield from _lossy_encode_sources(text, enc, errors)
            yield from _str_seeds(ctx, ANY)
        return self._pre(rule, ctx, props())

    # ---- zlib and base64: standard-library inverse pairs ------------------------------------------------------------
    # decompress inverts compress and b64decode inverts b64encode (owner 864: supplied inverses over core stdlib facts).
    # The stdlib call, not ``rule.native`` (zlib/base64 are not in the contract table's native modules), is the forward
    # check; ``Preimage.check`` re-runs it on every proposal, so an operand that is not the exact image is dropped.
    # compress(level, wbits) is deterministic, so decompress(stream) is THE preimage when one exists: the level is read
    # from the call (it must be known, only ``data`` may be the unknown) and a stream made at another level fails the
    # forward check instead of being claimed.
    def _codec(self, rule: Rule, ctx: PreimageCtx, forward, backward):
        """Hole = operand 0 (the data); ``forward`` is the stdlib call on a candidate, ``backward(seed)`` yields them."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad

        def native(*args, **kwargs):
            try:
                return forward(*args, **kwargs)
            except (zlib.error, binascii.Error):
                raise _NotChecked() from None

        def props():
            for seed in itertools.islice(ctx.want.seeds(), 16):
                if type(seed) is bytes and len(seed) <= _CODEC_MAX:
                    try:
                        yield from backward(seed)
                    except (zlib.error, binascii.Error, ValueError, TypeError):
                        continue
        return self._pre(rule, ctx, props(), None, native)

    def _zlib_arg(self, ctx: PreimageCtx, index: int, name: str, default):
        kw = dict(ctx.kwargs)
        return ctx.args[index] if len(ctx.args) > index else kw.get(name, default)

    def visit_zlib_compress(self, rule: Rule, ctx: PreimageCtx):
        if len(ctx.args) > 1 and ctx.args[1] is HOLE and type(ctx.args[0]) is bytes and len(ctx.args) <= 2:
            # the level is the unknown: it is backward-bound by trying every level against the observed stream
            def native(*args, **kwargs):
                try:
                    return zlib.compress(*args, **kwargs)
                except zlib.error:
                    raise _NotChecked() from None
            return self._pre(rule, ctx, iter((-1, *range(10))), None, native)
        level, wbits = self._zlib_arg(ctx, 1, "level", -1), self._zlib_arg(ctx, 2, "wbits", 15)
        if type(level) is not int or type(wbits) is not int:
            return self._refuse(rule, "the compression level and wbits must be known ints (only the data may be unknown)")
        return self._codec(rule, ctx, zlib.compress, lambda stream: [_inflate(stream, wbits)])

    def visit_zlib_decompress(self, rule: Rule, ctx: PreimageCtx):
        wbits = self._zlib_arg(ctx, 1, "wbits", 15)
        if type(wbits) is not int:
            return self._refuse(rule, "wbits must be a known int (only the data may be unknown)")
        # which level made the stream is not recoverable from the data, so every level is proposed (9 first: the
        # python-string-utils default); the forward check keeps those that decompress back to the observation
        return self._codec(rule, ctx, zlib.decompress,
                           lambda raw: (zlib.compress(raw, level, wbits) for level in (9, -1, 6, 1, 0, 2, 3, 4, 5, 7, 8)))

    def _base64_pair(self, rule: Rule, ctx: PreimageCtx, encode: bool, urlsafe: bool):
        extra = [a for i, a in enumerate(ctx.args) if i > 0] + [v for _, v in ctx.kwargs]
        if urlsafe and extra:
            return self._refuse(rule, "urlsafe_b64encode/decode take only the data")
        if any(type(a) not in (bytes, bool, type(None)) for a in extra):
            return self._refuse(rule, "altchars and validate must be known")
        alt = self._zlib_arg(ctx, 1, "altchars", None)
        if encode:
            fwd = base64.urlsafe_b64encode if urlsafe else base64.b64encode
            back = ((lambda text: [base64.urlsafe_b64decode(text)]) if urlsafe else
                    (lambda text: [base64.b64decode(text, alt, validate=True)]))
            return self._codec(rule, ctx, fwd, back)
        fwd = base64.urlsafe_b64decode if urlsafe else base64.b64decode
        enc = (lambda raw: [base64.urlsafe_b64encode(raw)]) if urlsafe else (lambda raw: [base64.b64encode(raw, alt)])
        # the canonical text, as bytes and as the ASCII str the function also takes
        return self._codec(rule, ctx, fwd, lambda raw: [x for e in enc(raw) for x in (e, e.decode("ascii"))])

    def visit_base64_url_safe_b64encode(self, rule: Rule, ctx: PreimageCtx):
        return self._base64_pair(rule, ctx, True, True)

    def visit_base64_url_safe_b64decode(self, rule: Rule, ctx: PreimageCtx):
        return self._base64_pair(rule, ctx, False, True)

    def visit_base64_b64encode(self, rule: Rule, ctx: PreimageCtx):
        return self._base64_pair(rule, ctx, True, False)

    def visit_base64_b64decode(self, rule: Rule, ctx: PreimageCtx):
        return self._base64_pair(rule, ctx, False, False)

    def visit_bytes_decode(self, rule: Rule, ctx: PreimageCtx):
        """``bytes.decode(encoding, errors)``: the preimage is ``str.encode`` of the observation with the same codec.

        Only the receiver may be unknown. The codec and error handler must be builtin names (a registered codec or handler
        is user code), and every candidate is forward-checked: it must DECODE BACK to the observation with the very same
        ``errors``. Under ``strict`` the re-encoding is the only preimage. Under ``ignore``/``replace``/... it is one
        preimage, not all of them: the invalid byte sequences the handler dropped or replaced are not recoverable from the
        text, so they are not enumerated (and an observation the codec cannot encode, such as 'é' for ascii, has none)."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        kw = dict(ctx.kwargs)
        enc = ctx.args[1] if len(ctx.args) > 1 else kw.get("encoding", "utf-8")
        errors = ctx.args[2] if len(ctx.args) > 2 else kw.get("errors", "strict")
        if type(enc) is not str or enc.lower() not in _DECODE_CODECS:
            return self._refuse(rule, "only the builtin utf-8/ascii/latin-1/utf-16/utf-32 codecs are inverted (a registered "
                                      "codec is user code)")
        if type(errors) is not str or errors not in _DECODE_ERRORS:
            return self._refuse(rule, "only the builtin error handlers strict/ignore/replace/backslashreplace/"
                                      "surrogateescape/surrogatepass are inverted (a registered handler is user code)")

        def props():
            for text in _str_seeds(ctx):
                for mode in dict.fromkeys(("strict", errors)):
                    try:
                        yield text.encode(enc, mode)
                    except (UnicodeEncodeError, LookupError):
                        continue
        return self._pre(rule, ctx, props())

    # ---- json: loads and dumps ---------------------------------------------------------------------------------------
    # ---- isinstance ----------------------------------------------------------------------------------------------------
    def visit_isinstance(self, rule: Rule, ctx: PreimageCtx):
        """``isinstance(HOLE, T)`` demanded true or false, for a builtin class ``T`` or a tuple / ``X | Y`` union of them.

        Only the object may be unknown. ``T`` must be one of ``object, bool, int, float, str, bytes, list, tuple, dict,
        NoneType``: each has a small canonical set of exact builtin values (``_ISINSTANCE_CANONICAL``), and every proposal is
        forward-checked with ``isinstance`` itself (so ``True`` is an ``int`` and a bool is never offered as a non-int). Any
        other class (an ABC, a user class) is a named refusal: constructing one runs ``__init__`` / ``__instancecheck__`` /
        ``__subclasshook__``, which is user code; a same-module class that is constructible without arguments is solved at the
        source level by ``solve_condition``, not here. The proposals are canonical values, not every instance."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        if len(ctx.args) != 2 or ctx.kwargs:
            return self._refuse(rule, "only isinstance(obj, T) with T a literal class or tuple of classes is inverted")
        classes = _flatten_classes(ctx.args[1])
        if classes is None or not all(c in _ISINSTANCE_CANONICAL for c in classes):
            return self._refuse(rule, "only object/bool/int/float/str/bytes/list/tuple/dict/NoneType (or a tuple or union of "
                                      "them) are inverted; another class would run user code to construct an instance")
        target = ctx.args[1]

        def props():
            for value in ctx.pool:
                yield value
            for cls in _ISINSTANCE_CANONICAL:
                yield from _ISINSTANCE_CANONICAL[cls]
        return Preimage(rule.qualname, ctx, None, props(), lambda obj, cls: isinstance(obj, cls))

    def visit_json_loads(self, rule: Rule, ctx: PreimageCtx):
        """``json.loads(HOLE)``: the JSON text of the observation, in the spellings ``json.dumps`` can write.

        Only the text may be unknown and no keyword is admitted (``cls``, ``object_hook``, ``parse_float``, ``parse_int``,
        ``parse_constant`` and ``object_pairs_hook`` are user code or change the result types). Proposed: ``json.dumps`` of
        each observation seed with the default, compact and ``ensure_ascii=False`` spellings and an indented one, each
        padded with insignificant whitespace in one variant. A proposal is kept only if ``json.loads`` gives back a value of
        EXACTLY the seed's types, recursively (``1`` vs ``1.0`` vs ``True``, ``list`` vs ``tuple``, ``str`` vs ``int``
        keys), so ``Eq(1)`` is never answered with ``'1.0'`` and a tuple or int-keyed dict, which JSON cannot return, has
        no preimage."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        if ctx.kwargs:
            return self._refuse(rule, "cls, object_hook, parse_* and object_pairs_hook are user code or change the result "
                                      "types; only json.loads(text) is inverted")

        def props():
            for seed in itertools.islice(ctx.want.seeds(), 16):
                for kwargs in ({}, {"separators": (",", ":")}, {"ensure_ascii": False}, {"indent": 2}):
                    try:
                        text = json.dumps(seed, **kwargs)
                        back = json.loads(text)
                    except (TypeError, ValueError, OverflowError, RecursionError):
                        continue
                    if len(text) > _CODEC_MAX or not _same_json(back, seed):
                        continue
                    yield text
                    yield " " + text + "\n"
        return self._pre(rule, ctx, props(), None, lambda *a, **k: json.loads(*a, **k))

    def visit_json_dumps(self, rule: Rule, ctx: PreimageCtx):
        """``json.dumps(HOLE, **options)``: ``json.loads`` of the observation text, kept only if dumping it again with the
        SAME options reproduces the text (so a text spaced or ordered differently from what these options write has no
        preimage). Only the object may be unknown; ``skipkeys``, ``ensure_ascii``, ``check_circular``, ``allow_nan``,
        ``indent``, ``separators`` and ``sort_keys`` must be literals, ``default`` and ``cls`` must be absent or ``None``
        (a hook or subclass is user code). One preimage, the JSON-native value: tuples, int-keyed dicts and subclasses
        that write the same text are not enumerated."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        if len(ctx.args) != 1:
            return self._refuse(rule, "only json.dumps(obj, <keyword options>) is inverted")
        options = dict(ctx.kwargs)
        for name, value in options.items():
            if name in ("default", "cls"):
                if value is not None:
                    return self._refuse(rule, f"{name}= is user code")
            elif name not in ("skipkeys", "ensure_ascii", "check_circular", "allow_nan", "indent", "separators", "sort_keys"):
                return self._refuse(rule, f"the option {name!r} is not one of the literal json.dumps options")

        def props():
            for seed in itertools.islice(ctx.want.seeds(), 16):
                if type(seed) is str and len(seed) <= _CODEC_MAX:
                    try:
                        yield json.loads(seed)
                    except (ValueError, RecursionError):
                        continue
        return self._pre(rule, ctx, props(), None, lambda *a, **k: json.dumps(*a, **k))

    def visit_unicodedata_normalize(self, rule: Rule, ctx: PreimageCtx):
        """``unicodedata.normalize(form, HOLE)``: the observation itself when it is already in that form, plus the sources
        that normalize to it.

        Only the text may be unknown and ``form`` must be one of NFC/NFD/NFKC/NFKD. Every proposal is forward-checked with
        ``unicodedata.normalize``, so a text that does not normalize to the observation is dropped: an observation that is
        not itself in ``form`` has no preimage at all (normalize is idempotent), and is answered with none. Proposed: the
        observation; its full (compatibility) decomposition; and, for each place a 1-3 character piece of the observation
        is the normalization of some other code point(s) (``'fi'`` of U+FB01, ``'A'`` of U+FF21, ``'\u00e9'`` of
        ``'e\u0301'``), that source substituted at one place and at every place. This is sound, not exhaustive: the
        remaining combinations of sources per position are not enumerated, so an empty answer is no proof of absence."""
        try:
            hole = ctx.hole
        except ValueError as exc:
            return self._refuse(rule, str(exc))
        form = ctx.args[0] if ctx.args else None
        if hole != 1 or len(ctx.args) != 2 or ctx.kwargs or form not in ("NFC", "NFD", "NFKC", "NFKD"):
            return self._refuse(rule, "only the text of a call with a literal form NFC/NFD/NFKC/NFKD may be the unknown")

        def native(f, text):
            return unicodedata.normalize(f, text)

        def props():
            sources = _normalization_sources(form)
            for text in _str_seeds(ctx):
                yield text
                yield unicodedata.normalize("NFKD" if "K" in form else "NFD", text)
                spots = [(i, n) for i in range(len(text)) for n in (1, 2, 3) if text[i:i + n] in sources]
                for i, n in spots[:64]:
                    for alt in sources[text[i:i + n]][:3]:
                        yield text[:i] + alt + text[i + n:]
                for pick in (0, 1):
                    out, i = [], 0
                    while i < len(text):
                        for n in (3, 2, 1):
                            alts = sources.get(text[i:i + n])
                            if alts and len(alts) > pick:
                                out.append(alts[pick])
                                i += n
                                break
                        else:
                            out.append(text[i])
                            i += 1
                    yield "".join(out)
        return self._pre(rule, ctx, props(), None, native)

    def visit_str_translate(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        table = ctx.args[1] if len(ctx.args) > 1 else None
        if type(table) is not dict:
            return self._refuse(rule, "the table is not a dict (a __getitem__ hook could run)")
        inverse = {}
        for k, v in table.items():
            if type(k) is int and type(v) is str and len(v) == 1:
                inverse.setdefault(v, chr(k))

        def props():
            for s in _str_seeds(ctx):
                yield s
                yield "".join(inverse.get(c, c) for c in s)
        return self._pre(rule, ctx, props())

    # ---- builtins -------------------------------------------------------------------------------------------------
    def visit_len(self, rule: Rule, ctx: PreimageCtx):
        """``len(x)``: strings (and lists) of the wanted lengths; exact transfer to :class:`Length`."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        w = ctx.want
        transfer = None
        if isinstance(w, Eq) and type(w.value) is int:
            transfer = Length(w.value, w.value) if w.value >= 0 else _NOTHING
        elif isinstance(w, Compare) and type(w.const) is int and w.op != "!=":
            c = w.const
            lo, hi = {"<": (0, c - 1), "<=": (0, c), ">": (c + 1, None), ">=": (c, None), "==": (c, c)}[w.op]
            transfer = Length(max(lo, 0), hi) if hi is None or hi >= max(lo, 0) else _NOTHING

        def props():
            alphabet = "".join(dict.fromkeys("".join(p for p in ctx.pool if type(p) is str) + "abcdefgh"))
            for n in _int_seeds(w):
                if 0 <= n <= 4096:
                    for p in ctx.pool:
                        if type(p) is str and p:
                            yield (p * (n // len(p) + 1))[:n]
                    yield (alphabet * (n // len(alphabet) + 1))[:n]
                    yield "a" * n
        return self._pre(rule, ctx, props(), transfer)

    def visit_set(self, rule: Rule, ctx: PreimageCtx):
        """``set(x)`` of a str: a wanted size ``n`` (a :class:`Length` want, e.g. transferred from ``len(set(x))``) is
        ``n`` distinct characters -- printable ones first, then code points past ASCII when ``n`` exceeds them."""
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        sizes = _lengths(ctx.want) if isinstance(ctx.want, (Length, All)) else []

        def props():
            for n in sizes:
                if 0 <= n <= 4096:
                    printable = _PRINTABLE[:n]
                    yield printable if len(printable) == n else printable + "".join(chr(0x100 + i)
                                                                                    for i in range(n - len(printable)))
                    yield "".join(chr(0x100 + i) for i in range(n))
        return self._pre(rule, ctx, props())

    def visit_ord(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        return self._pre(rule, ctx, (chr(k) for k in _int_seeds(ctx.want) if 0 <= k <= 0x10FFFF))

    def visit_chr(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        return self._pre(rule, ctx, (ord(s) for s in _str_seeds(ctx) if type(s) is str and len(s) == 1))

    # ---- str + and * ----------------------------------------------------------------------------------------------
    def visit_protocol_binary_operator(self, rule: Rule, ctx: PreimageCtx):
        """str ``+`` and ``*`` on exact builtin operands (no ``__add__``/``__radd__`` dispatch to user code can happen
        when both operands are exactly ``str``/``int``); ``ctx.op`` names the operator."""
        try:
            hole = ctx.hole
        except ValueError as exc:
            return self._refuse(rule, str(exc))
        if ctx.op not in ("+", "*") or len(ctx.args) != 2:
            return self._refuse(rule, "only str + and str * are inverted")
        other = ctx.args[1 - hole]
        native = (lambda a, b: _exact_binop(ctx.op, a, b))
        if ctx.op == "+":
            if type(other) is not str:
                return self._refuse(rule, "the known operand of + is not a str")
            transfer = None
            if isinstance(ctx.want, Eq) and type(ctx.want.value) is str:
                v = ctx.want.value
                ok = v.endswith(other) if hole == 0 else v.startswith(other)
                transfer = Eq(v[:len(v) - len(other)] if hole == 0 else v[len(other):]) if ok else _NOTHING

            def props():
                for v in _str_seeds(ctx):
                    if hole == 0 and v.endswith(other):
                        yield v[:len(v) - len(other)]
                    if hole == 1 and v.startswith(other):
                        yield v[len(other):]
                    yield from (_around("", other, v) if hole == 0 else _around(other, "", v))
            return self._pre(rule, ctx, props(), transfer, native)
        # *
        if hole == 0 and type(other) in (int, bool):
            def props():
                for v in _str_seeds(ctx):
                    if other > 0 and len(v) % other == 0:
                        yield v[:len(v) // other]
                    yield v
        elif hole == 1 and type(other) is str:
            def props():
                for v in _str_seeds(ctx):
                    if other and len(v) % len(other) == 0:
                        yield len(v) // len(other)
                yield from (0, 1, 2)
        else:
            return self._refuse(rule, "only str * int is inverted")
        return self._pre(rule, ctx, props(), None, native)

    # ---- the re module, Pattern and Match --------------------------------------------------------------------------
    def _regex_operands(self, rule: Rule, ctx: PreimageCtx):
        """``(pattern, flags, method, scanned_index)`` of a re call, or a refusal."""
        q = rule.qualname
        try:
            hole = ctx.hole
        except ValueError as exc:
            return self._refuse(rule, str(exc))
        kw = dict(ctx.kwargs)
        if q.startswith("re.Pattern."):
            p = ctx.args[0]
            if type(p) is not re.Pattern:
                return self._refuse(rule, "the receiver is not a compiled builtin Pattern")
            pattern, flags, method = p.pattern, p.flags, q.rsplit(".", 1)[1]
            scanned = 2 if method in ("sub", "subn") else 1
        else:
            method = q.split(".", 1)[1]
            p = ctx.args[0]
            if type(p) is re.Pattern:
                pattern, flags = p.pattern, p.flags
            else:
                pattern = p
                fi = {"match": 2, "search": 2, "fullmatch": 2, "findall": 2, "finditer": 2, "split": 3,
                      "sub": 4, "subn": 4}.get(method)
                flags = kw.get("flags", ctx.args[fi] if fi is not None and len(ctx.args) > fi else 0)
            scanned = 2 if method in ("sub", "subn") else 1
        if hole != scanned:
            return self._refuse(rule, "only the scanned string of a regex call may be the unknown operand")
        why = RM.refusal(pattern, flags)
        if why:
            return self._refuse(rule, why)
        return pattern, int(flags), method, scanned

    def _regex_native(self, rule: Rule, pattern: str, flags: int, method: str, scanned: int):
        def call(*args, **kwargs):
            rest = list(args[1:])
            value = rest[scanned - 1] if scanned - 1 < len(rest) else None
            if type(value) is not str or len(value) > 4096:
                raise _NotChecked()
            if flags & re.DEBUG and type(args[0]) is str:
                raise _NotChecked()          # compiling with re.DEBUG prints; a compiled Pattern does not
            if not RM.admitted(pattern, flags, len(value), method, lambda: rule.native(*args, **kwargs)):
                raise _NotChecked()
            return rule.native(*args, **kwargs)
        return call

    def _regex_match(self, rule: Rule, ctx: PreimageCtx):
        got = self._regex_operands(rule, ctx)
        if isinstance(got, Refused):
            return got
        pattern, flags, method, scanned = got
        if len(ctx.args) > scanned + 1 and rule.qualname.startswith("re.Pattern."):
            return self._refuse(rule, "pos/endpos are not inverted")
        hit = ctx.want in (IsNone(False), Truth(True))
        miss = ctx.want in (IsNone(True), Truth(False))
        transfer = Member(pattern, flags, method) if hit else Not(Member(pattern, flags, method)) if miss else None
        prefer = tuple(dict.fromkeys(c for p in ctx.pool if type(p) is str for c in p))[:16]

        def props():
            if not miss:
                yield from RM.members(pattern, flags, how=method, limit=16, prefer=prefer)
            for s in _str_seeds(ctx, ANY):
                yield s
            for m in itertools.islice(RM.members(pattern, flags, how=method, limit=4), 4):
                yield m[:-1]
                yield m + "\x00"
                yield "\x00" + m
        return self._pre(rule, ctx, props(), transfer, self._regex_native(rule, pattern, flags, method, scanned))

    visit_re_match = visit_re_search = visit_re_fullmatch = _regex_match
    visit_re_pattern_match = visit_re_pattern_search = visit_re_pattern_fullmatch = _regex_match

    def _regex_sub(self, rule: Rule, ctx: PreimageCtx):
        got = self._regex_operands(rule, ctx)
        if isinstance(got, Refused):
            return got
        pattern, flags, method, scanned = got
        repl = ctx.args[1]
        if type(repl) is not str or "\\" in repl:
            return self._refuse(rule, "the replacement is not a plain str (callables and group references are not "
                                      "inverted)")
        mems = list(RM.members(pattern, flags, how="search", limit=6))

        def props():
            for text in _str_seeds(ctx):
                yield text
                for m in mems:
                    if repl:
                        # Every replaced occurrence is reversed, not only the first: an observation with several
                        # replacements ('top-10-reasons' from 'top 10 reasons') has no preimage that restores one.  The
                        # all-occurrences text is yielded before the single insertions (a per-step cap would cut it);
                        # first-only and last-only keep the single-separator shapes.  ``_pre`` forward-checks each natively.
                        yield text.replace(repl, m)
                        yield text.replace(repl, m, 1)
                        head, sep, tail = text.rpartition(repl)
                        if sep:
                            yield head + m + tail
                    for i in range(min(len(text), 32) + 1):
                        yield text[:i] + m + text[i:]
        return self._pre(rule, ctx, props(), None, self._regex_native(rule, pattern, flags, method, scanned))

    visit_re_sub = visit_re_subn = visit_re_pattern_sub = visit_re_pattern_subn = _regex_sub

    def _regex_collect(self, rule: Rule, ctx: PreimageCtx):
        """split / findall / finditer: separators (split) or occurrences (findall) are members of the pattern."""
        got = self._regex_operands(rule, ctx)
        if isinstance(got, Refused):
            return got
        pattern, flags, method, scanned = got
        mems = list(RM.members(pattern, flags, how="search", limit=4)) or [""]
        pad = _filler("".join(mems))

        def props():
            counts = [n for n in _lengths(ctx.want) if 0 <= n <= 64] if isinstance(ctx.want, (Length, All)) else []
            if isinstance(ctx.want, Items) and ctx.want.length is not None:
                counts = [n for n in _int_seeds(ctx.want.length) if 0 <= n <= 64]
            for n in counts or [1, 2]:
                for m in mems:
                    if method == "split":
                        yield m.join([pad] * max(n, 1))
                    else:
                        yield pad.join([m] * n) if n else pad
            yield from _str_seeds(ctx, ANY)
        native = self._regex_native(rule, pattern, flags, method, scanned)
        if method == "finditer":
            inner = native
            native = (lambda *a, **k: list(inner(*a, **k)))
        return self._pre(rule, ctx, props(), None, native)

    visit_re_split = visit_re_findall = visit_re_finditer = _regex_collect
    visit_re_pattern_split = visit_re_pattern_findall = visit_re_pattern_finditer = _regex_collect

    def visit_re_escape(self, rule: Rule, ctx: PreimageCtx):
        bad = self._receiver_only(rule, ctx)
        if bad:
            return bad
        if isinstance(ctx.want, Eq):
            target = ctx.want.value
            if type(target) not in (str, bytes):
                return self._refuse(rule, 'a non-text escape result requires an unmodelled translate override')
            text = target.decode('latin1') if type(target) is bytes else target
            decoded = []
            position = 0
            while position < len(text):
                char = text[position]
                if char == '\\':
                    position += 1
                    if position == len(text) or ord(text[position]) not in re._special_chars_map:
                        return self._pre(rule, ctx, (), native=re.escape)
                    char = text[position]
                elif ord(char) in re._special_chars_map:
                    return self._pre(rule, ctx, (), native=re.escape)
                decoded.append(char)
                position += 1
            value = ''.join(decoded)
            if type(target) is bytes:
                value = value.encode('latin1')
            # One linear builtin operation needs no regex/format expansion window.
            return self._pre(rule, ctx, (value,), native=re.escape)

        def props():
            for v in _str_seeds(ctx):
                yield re.sub(r"\\(.)", r"\1", v)
                yield v
        return self._pre(rule, ctx, props())

    def _match_object(self, rule: Rule, ctx: PreimageCtx):
        return self._refuse(rule, "a Match object is only built by matching: ask the Pattern/re method's preimage with "
                                  "a want on the match instead")

    visit_re_match_group = visit_re_match_groups = visit_re_match_groupdict = _match_object
    visit_re_match_start = visit_re_match_end = visit_re_match_span = _match_object
    visit_re_match_getitem = visit_re_match_expand = _match_object

    def _no_string_hole(self, rule: Rule, ctx: PreimageCtx):
        return self._refuse(rule, "the result does not depend on a string operand a preimage could choose")

    visit_re_compile = visit_re_purge = visit_re_pattern_scanner = _no_string_hole
    visit_str_maketrans = visit_str_format_map = visit_str_dunder_format = _no_string_hole


def _around(pre: str, post: str, v: str) -> list[str]:
    """Hole values ``h`` for a result ``pre + h + post`` that should contain the piece ``v``: ``v`` with the literal
    text it overlaps cut off at either end, and ``v`` itself."""
    out = []
    w = v
    for k in range(min(len(pre), len(w)), 0, -1):
        if w.startswith(pre[-k:]):
            w = w[k:]
            break
    for k in range(min(len(post), len(w)), 0, -1):
        if w.endswith(post[:k]):
            w = w[:-k]
            break
    out.append(w)
    if v.startswith(pre):
        out.append(v[len(pre):])
    if post and v.endswith(post):
        out.append(v[:-len(post)])
    out.append(v)
    return out


def _exact_binop(op: str, a, b):
    if not all(type(x) in (str, int, bool) for x in (a, b)):
        raise _NotChecked()
    if op == "+":
        if type(a) is not str or type(b) is not str:
            raise TypeError("str + needs two str")
        return a + b
    if not ((type(a) is str and type(b) in (int, bool)) or (type(b) is str and type(a) in (int, bool))):
        raise TypeError("str * needs a str and an int")
    if abs(b if type(b) is not str else a) * len(a if type(a) is str else b) > 4096:
        raise _NotChecked()
    return a * b


#: the want no value satisfies (an inconsistent exact transfer: ``len(x) == -1``, ``x + 'b' == 'ac'``)
_NOTHING = AnyOf(())

#: every rule kind the preimage engine has a visit method for
SUPPORTED_KINDS = PreimageVisitor.implements()

