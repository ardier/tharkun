"""Contracts for the ``re`` module ([internal ref] batch 3) -- the census's #1 priority
(28 imports across eregion-sv's suites + Tharkun's own tests, vs. single digits for
every other batch-3 module; see the PR body / ``docs/contracts/REPORT.md``).

Module-level ``re.match``/``re.search``/``re.fullmatch``/``re.sub``/``re.split``/
``re.findall`` all compile their pattern argument on the fly (cached) and then run the
SAME underlying matcher a ``Pattern`` object's own ``.match``/``.search``/etc. methods
use -- this file contracts the module-level functions; a compiled ``Pattern``'s
methods share their raise conditions (documented once here, not duplicated).

``re.error`` (aliased ``re.PatternError`` in 3.13+) is a separate exception, not a
property class -- a malformed pattern's ``re.compile`` failure is out of scope for this
batch's 7 tracked classes (``may_raise_other=True`` covers it).

All module functions and public Pattern/Match method siblings are Rule-owned.
Finite builtin operands fold through both visitors; mutable scanner state and
unencoded symbolic regexes receive named refusals. Contract digests pin this family.
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/re.html"
_ALL_TESTED = frozenset(ALL_PROPERTY_CLASSES)


@register_rule
class ReCompile(Rule):
    """The rule of ``re.compile``."""

    SAMPLES = {'re.compile': 'import re\np = re.compile("a")\n'}
    TWINS = {'re.compile': 'import re\np = re.compile()\n'}

    names = ("re.compile",)
    CONTRACT = Contract(
        qualname="re.compile",
        raises=dunder_propagation_claims(
            "flags, combined with the internal flag bits via & -- a non-int flags "
            "argument's __and__/__rand__ is a USER method and propagates whatever "
            "it raises",
            native={"TypeError": "pattern is not str/bytes/a compiled Pattern, or flags has neither __index__ nor __and__/__rand__"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # re.error: malformed pattern; flag/pattern-type mismatch
        returns=ReturnSpec("re.Pattern", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True, notes="internally cached (re._cache) -- repeated calls with the same pattern+flags return semantically identical, though not necessarily identical-by-`is`, compiled objects"),
        notes="SOUNDNESS NOTE (gatekeeper audit): flags is combined with & rather than converted via __index__ -- confirmed empirically (a flags object with a raising __and__/__rand__ propagates that exact exception; one with only a raising __index__ is never even asked for it).",
        source=(f"{_DOCS}#re.compile", "CPython Lib/re/__init__.py::compile / Lib/re/_compiler.py"),
        tested=True,
    )


@register_rule
class ReMatch(Rule):
    """The rule of ``re.match``."""

    SAMPLES = {'re.match': 'import re\nm = re.match("a", "a")\n'}
    TWINS = {'re.match': 'import re\nm = re.match()\n'}

    names = ("re.match",)
    CONTRACT = Contract(
        qualname="re.match",
        raises=dunder_propagation_claims(
            "flags, combined via & (same as re.compile's) when pattern is not already compiled",
            native={
                "TypeError": (
                    "pattern is not str/bytes/Pattern; string is not str/bytes; the "
                    "str-vs-bytes kind of pattern and string disagree ('cannot use a "
                    "string pattern on a bytes-like object' and vice versa); or flags "
                    "has neither __index__ nor __and__/__rand__"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # re.error: malformed pattern; ValueError: flags given with an already-compiled pattern
        returns=ReturnSpec("re.Match | None", "None means no match -- calling a Match method on the result without checking for None raises AttributeError (not a re-specific exception)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="matches only at the START of the string (like Pattern.match) -- not anchored to the end; re.fullmatch is the anchored-both-ends variant.",
        source=(f"{_DOCS}#re.match", "CPython Lib/re/__init__.py::match"),
        tested=True,
    )


@register_rule
class ReSearch(Rule):
    """The rule of ``re.search``."""

    SAMPLES = {'re.search': 'import re\nm = re.search("a", "ba")\n'}
    TWINS = {'re.search': 'import re\nm = re.search()\n'}

    names = ("re.search",)
    CONTRACT = Contract(
        qualname="re.search",
        raises=dunder_propagation_claims(
            "flags, combined via & (same as re.compile's)",
            native={"TypeError": "same as re.match: bad pattern/string types, a str/bytes kind mismatch, or a bad flags value"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("re.Match | None", "scans the WHOLE string for the first match, unlike match()'s start-anchored search", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(f"{_DOCS}#re.search", "CPython Lib/re/__init__.py::search"),
        tested=True,
    )


@register_rule
class ReFullmatch(Rule):
    """The rule of ``re.fullmatch``."""

    SAMPLES = {'re.fullmatch': 'import re\nm = re.fullmatch("a", "a")\n'}
    TWINS = {'re.fullmatch': 'import re\nm = re.fullmatch()\n'}

    names = ("re.fullmatch",)
    CONTRACT = Contract(
        qualname="re.fullmatch",
        raises=dunder_propagation_claims(
            "flags, combined via & (same as re.compile's)",
            native={"TypeError": "same as re.match: bad pattern/string types, a str/bytes kind mismatch, or a bad flags value"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("re.Match | None", "the ENTIRE string must match, both ends anchored", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(f"{_DOCS}#re.fullmatch", "CPython Lib/re/__init__.py::fullmatch"),
        tested=True,
    )


@register_rule
class ReSub(Rule):
    """The rule of ``re.sub``."""

    SAMPLES = {'re.sub': 'import re\nx = re.sub("a", "b", "aa")\n'}
    TWINS = {'re.sub': 'import re\nx = re.sub()\n'}

    names = ("re.sub",)
    CONTRACT = Contract(
        qualname="re.sub",
        raises=dunder_propagation_claims(
            "repl, when it's a callable, applied to each Match (so anything "
            "Match.group()/etc. can raise -- see re.Match.group below -- propagates "
            "from inside repl); count's __index__ (a USER method)",
            native={
                "TypeError": (
                    "pattern/string type errors as above; repl is neither a str/"
                    "bytes replacement template nor callable; or count has no "
                    "__index__"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # re.error: malformed pattern or replacement backreference
        returns=ReturnSpec("str | bytes (matches the input string's type)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(f"{_DOCS}#re.sub", "CPython Lib/re/__init__.py::sub"),
        tested=True,
    )


@register_rule
class ReSplit(Rule):
    """The rule of ``re.split``."""

    SAMPLES = {'re.split': 'import re\nx = re.split(",", "a,b")\n'}
    TWINS = {'re.split': 'import re\nx = re.split()\n'}

    names = ("re.split",)
    CONTRACT = Contract(
        qualname="re.split",
        raises=dunder_propagation_claims(
            "maxsplit's __index__ (a USER method -- any exception it raises propagates uncaught)",
            native={"TypeError": "pattern/string type errors as above, or maxsplit has no __index__"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # re.error: a pattern that can match an empty string used to warn/error in older Python; 3.7+ splits on it
        returns=ReturnSpec("list[str | bytes]", "capturing groups in the pattern are INTERLEAVED into the result list, between the split pieces", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims, wrongly marking IndexError/KeyError/AttributeError/"
            "ZeroDivisionError/FloatingPointError/AssertionError as 'never' -- "
            "maxsplit's __index__ can raise any of them and it propagates uncaught. "
            "Repro: re.split('a', 'aaa', maxsplit=BadIndex()) where "
            "BadIndex.__index__ raises IndexError."
        ),
        source=(f"{_DOCS}#re.split", "CPython Lib/re/__init__.py::split"),
        tested=True,
    )


@register_rule
class ReFindall(Rule):
    """The rule of ``re.findall``."""

    SAMPLES = {'re.findall': 'import re\nx = re.findall("a", "aa")\n'}
    TWINS = {'re.findall': 'import re\nx = re.findall()\n'}

    names = ("re.findall",)
    CONTRACT = Contract(
        qualname="re.findall",
        raises=dunder_propagation_claims(
            "flags, combined via & (same as re.compile's)",
            native={"TypeError": "pattern/string type errors as above, or a bad flags value"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec(
            "list[str | bytes] for a pattern with 0 or 1 groups; list[tuple] for 2+ groups",
            "shape depends on the PATTERN's group count, not on the matched text -- a genuinely variable-shape return",
            tested=True,
        ),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(f"{_DOCS}#re.findall", "CPython Lib/re/__init__.py::findall"),
        tested=True,
    )


@register_rule
class ReEscape(Rule):
    """The rule of ``re.escape``."""

    SAMPLES = {'re.escape': 'import re\nx = re.escape("a.b")\n'}
    TWINS = {'re.escape': 'import re\nx = re.escape()\n'}

    names = ("re.escape",)
    CONTRACT = Contract(
        qualname="re.escape",
        raises=dunder_propagation_claims(
            "a str subclass's translate override (non-str buffer conversion may also execute user code)",
            native={"TypeError": "input cannot be converted to str/bytes, or its callback raises TypeError"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("Any", "native translate returns escaped str/bytes; a str subclass translate override may return any object", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(f"{_DOCS}#re.escape", "CPython Lib/re/__init__.py::escape"),
        tested=True,
    )


# ------------------------------------------------------------------
# re.Match methods
# ------------------------------------------------------------------


@register_rule
class ReMatchGroup(Rule):
    """The rule of ``re.Match.group``."""

    SAMPLES = {'re.Match.group': 'import re\nm = re.match("(a)(?P<k>b)", "ab")\nx = m.group(0)\n'}
    TWINS = {}

    names = ("re.Match.group",)
    CONTRACT = Contract(
        qualname="re.Match.group",
        raises=dunder_propagation_claims(
            "a non-int, non-str argument's __index__ (a USER method) while resolving it to a group number",
            native={
                "IndexError": (
                    "the argument (an int index OR a group name str) names a group "
                    "that does not exist in the pattern -- SAME class (IndexError) "
                    "for both an out-of-range int AND an unknown name; a name lookup "
                    "does NOT raise KeyError the way a dict miss would"
                ),
            },
            tested_native=frozenset({"IndexError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the argument's __index__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("str | bytes | None | tuple[str | bytes | None, ...]", "Multiple keys return a tuple; None for a group the pattern's overall match did not traverse (e.g. an alternation branch not taken, or an optional group)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims, wrongly marking every class but IndexError as 'never' -- "
            "an object passed as the group argument that has __index__ raising, "
            "say, KeyError propagates that KeyError unchanged, not IndexError."
        ),
        source=(f"{_DOCS}#re.Match.group", "CPython Modules/_sre/sre.c::match_group"),
        tested=True,
    )


@register_rule
class ReMatchGroups(Rule):
    """The rule of ``re.Match.groups``."""

    SAMPLES = {'re.Match.groups': 'import re\nm = re.match("(a)(?P<k>b)", "ab")\nx = m.groups()\n'}
    TWINS = {}

    names = ("re.Match.groups",)
    CONTRACT = Contract(
        qualname="re.Match.groups",
        raises=never_claims(native={"TypeError": "more than one default argument, unknown keyword or duplicate default"}, tested=_ALL_TESTED),
        may_raise_other=False,
        returns=ReturnSpec("tuple", "one entry per capturing group in the pattern, in order; unmatched groups are None, or `default` if given", tested=True),
        effects=Effects(deterministic=True),
        source=(f"{_DOCS}#re.Match.groups", "CPython Modules/_sre/sre.c::match_groups"),
        tested=True,
    )


@register_rule
class ReMatchGroupdict(Rule):
    """The rule of ``re.Match.groupdict``."""

    SAMPLES = {'re.Match.groupdict': 'import re\nm = re.match("(a)(?P<k>b)", "ab")\nx = m.groupdict()\n'}
    TWINS = {}

    names = ("re.Match.groupdict",)
    CONTRACT = Contract(
        qualname="re.Match.groupdict",
        raises=never_claims(native={"TypeError": "more than one default argument, unknown keyword or duplicate default"}, tested=_ALL_TESTED),
        may_raise_other=False,
        returns=ReturnSpec("dict[str, object]", "only NAMED groups (?P<name>...); unmatched named groups are None, or `default` if given", tested=True),
        effects=Effects(deterministic=True),
        source=(f"{_DOCS}#re.Match.groupdict", "CPython Modules/_sre/sre.c::match_groupdict"),
        tested=True,
    )


@register_rule
class ReMatchStart(Rule):
    """The rule of ``re.Match.start``."""

    SAMPLES = {'re.Match.start': 'import re\nm = re.match("(a)(?P<k>b)", "ab")\nx = m.start()\n'}
    TWINS = {}

    names = ("re.Match.start",)
    CONTRACT = Contract(
        qualname="re.Match.start",
        raises=dunder_propagation_claims(
            "a non-int, non-str argument's __index__ (same as group()'s)",
            native={"IndexError": "same as group(): int or name references a group that does not exist"},
            tested_native=frozenset({"IndexError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the argument's __index__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("int", "-1 for a group that exists in the pattern but did not participate in this match (NOT an error)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="SOUNDNESS FIX (gatekeeper): same __index__-propagation fix as group().",
        source=(f"{_DOCS}#re.Match.start", "CPython Modules/_sre/sre.c::match_start"),
        tested=True,
    )


@register_rule
class ReMatchEnd(Rule):
    """The rule of ``re.Match.end``."""

    SAMPLES = {'re.Match.end': 'import re\nm = re.match("(a)(?P<k>b)", "ab")\nx = m.end()\n'}
    TWINS = {}

    names = ("re.Match.end",)
    CONTRACT = Contract(
        qualname="re.Match.end",
        raises=dunder_propagation_claims(
            "a non-int, non-str argument's __index__ (same as group()'s)",
            native={"IndexError": "same as start()/group()"},
            tested_native=frozenset({"IndexError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the argument's __index__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("int", "-1 for a group that did not participate", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="SOUNDNESS FIX (gatekeeper): same __index__-propagation fix as group().",
        source=(f"{_DOCS}#re.Match.end", "CPython Modules/_sre/sre.c::match_end"),
        tested=True,
    )


@register_rule
class ReMatchSpan(Rule):
    """The rule of ``re.Match.span``."""

    SAMPLES = {'re.Match.span': 'import re\nm = re.match("(a)(?P<k>b)", "ab")\nx = m.span()\n'}
    TWINS = {}

    names = ("re.Match.span",)
    CONTRACT = Contract(
        qualname="re.Match.span",
        raises=dunder_propagation_claims(
            "a non-int, non-str argument's __index__ (same as group()'s)",
            native={"IndexError": "same as start()/group()"},
            tested_native=frozenset({"IndexError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the argument's __index__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("tuple[int, int]", "(start(group), end(group)); (-1, -1) for a group that did not participate", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="SOUNDNESS FIX (gatekeeper): same __index__-propagation fix as group().",
        source=(f"{_DOCS}#re.Match.span", "CPython Modules/_sre/sre.c::match_span"),
        tested=True,
    )




def _regex_contract(name, result):
    shape = "wrong receiver, argument count, keyword or duplicate argument"
    if name == "purge":
        type_error = "purge accepts no arguments"
    elif name == "Match.__getitem__":
        type_error = shape
    elif name == "Match.expand":
        type_error = shape + "; template is not str/bytes matching the subject kind"
    elif name.startswith("Pattern."):
        type_error = shape + "; subject is not a compatible str/bytes buffer; positions/count/maxsplit lack __index__"
        if name.endswith((".sub", ".subn")):
            type_error += "; replacement is neither a matching template nor callable"
    else:
        type_error = shape + "; pattern is not str/bytes/Pattern, subject kind disagrees, flags reject bit operations"
        if name == "subn":
            type_error += "; count lacks __index__ or replacement is neither a matching template nor callable"
    native = {"TypeError": type_error}
    if name in {"Match.__getitem__", "Match.expand", "subn", "Pattern.sub", "Pattern.subn"}:
        native["IndexError"] = "group key or named replacement-template group does not exist"
    callbacks = name != "purge"
    raises = (dunder_propagation_claims("flags bit operations, positions/count __index__, replacement/template/group protocols", native=native, tested_native=frozenset(native)) if callbacks
              else never_claims(native=native, tested=_ALL_TESTED))
    return Contract(
        qualname="re." + name, raises=raises,
        may_raise_other=callbacks, returns=ReturnSpec(result, tested=True),
        effects=Effects(may_execute_callback=callbacks), source=(_DOCS,),
        notes="PatternError for malformed pattern/replacement; ValueError for invalid flags or compiled pattern with nonzero flags. Missing named template group is IndexError. Iterator/scanner consumption is separate; groups preserve str/bytes kinds.",
        tested=True,
    )


@register_rule
class ReFinditer(Rule):
    names = ('re.finditer',)
    SAMPLES = {'re.finditer': 'import re\nx = re.finditer("a", "aba")\n'}
    TWINS = {}
    CONTRACT = _regex_contract('finditer', 'Iterator[re.Match]')


@register_rule
class ReSubn(Rule):
    names = ('re.subn',)
    SAMPLES = {'re.subn': 'import re\nx = re.subn("a", "b", "aba")\n'}
    TWINS = {}
    CONTRACT = _regex_contract('subn', 'tuple[str | bytes,int]')


@register_rule
class RePurge(Rule):
    names = ('re.purge',)
    SAMPLES = {'re.purge': 'import re\nx = re.purge()\n'}
    TWINS = {}
    CONTRACT = _regex_contract('purge', 'None')


@register_rule
class RePatternMatch(Rule):
    names = ('re.Pattern.match',)
    SAMPLES = {'re.Pattern.match': 'import re\nx = re.compile("(a)").match("aba")\n'}
    TWINS = {}
    CONTRACT = _regex_contract('Pattern.match', 're.Match | None')


@register_rule
class RePatternSearch(Rule):
    names = ('re.Pattern.search',)
    SAMPLES = {'re.Pattern.search': 'import re\nx = re.compile("(a)").search("aba")\n'}
    TWINS = {}
    CONTRACT = _regex_contract('Pattern.search', 're.Match | None')


@register_rule
class RePatternFullmatch(Rule):
    names = ('re.Pattern.fullmatch',)
    SAMPLES = {'re.Pattern.fullmatch': 'import re\nx = re.compile("(a)").fullmatch("a")\n'}
    TWINS = {}
    CONTRACT = _regex_contract('Pattern.fullmatch', 're.Match | None')


@register_rule
class RePatternFindall(Rule):
    names = ('re.Pattern.findall',)
    SAMPLES = {'re.Pattern.findall': 'import re\nx = re.compile("(a)").findall("aba")\n'}
    TWINS = {}
    CONTRACT = _regex_contract('Pattern.findall', 'list[str | bytes | tuple]')


@register_rule
class RePatternFinditer(Rule):
    names = ('re.Pattern.finditer',)
    SAMPLES = {'re.Pattern.finditer': 'import re\nx = re.compile("(a)").finditer("aba")\n'}
    TWINS = {}
    CONTRACT = _regex_contract('Pattern.finditer', 'Iterator[re.Match]')


@register_rule
class RePatternSplit(Rule):
    names = ('re.Pattern.split',)
    SAMPLES = {'re.Pattern.split': 'import re\nx = re.compile("(a)").split("aba")\n'}
    TWINS = {}
    CONTRACT = _regex_contract('Pattern.split', 'list[str | bytes | None]')


@register_rule
class RePatternSub(Rule):
    names = ('re.Pattern.sub',)
    SAMPLES = {'re.Pattern.sub': 'import re\nx = re.compile("(a)").sub("b", "aba")\n'}
    TWINS = {}
    CONTRACT = _regex_contract('Pattern.sub', 'str | bytes')


@register_rule
class RePatternSubn(Rule):
    names = ('re.Pattern.subn',)
    SAMPLES = {'re.Pattern.subn': 'import re\nx = re.compile("(a)").subn("b", "aba")\n'}
    TWINS = {}
    CONTRACT = _regex_contract('Pattern.subn', 'tuple[str | bytes,int]')


@register_rule
class RePatternScanner(Rule):
    names = ('re.Pattern.scanner',)
    SAMPLES = {'re.Pattern.scanner': 'import re\nx = re.compile("(a)").scanner("aba")\n'}
    TWINS = {}
    CONTRACT = _regex_contract('Pattern.scanner', 'scanner')


@register_rule
class ReMatchExpand(Rule):
    names = ('re.Match.expand',)
    SAMPLES = {'re.Match.expand': 'import re\nx = re.match("(a)", "a").expand(r"\\1!")\n'}
    TWINS = {}
    CONTRACT = _regex_contract('Match.expand', 'str | bytes')


@register_rule
class ReMatchGetitem(Rule):
    names = ('re.Match.__getitem__',)
    SAMPLES = {'re.Match.__getitem__': 'import re\nx = re.match("(a)", "a")[1]\n'}
    TWINS = {}
    CONTRACT = _regex_contract('Match.__getitem__', 'str | bytes | None')

__all__ = ['ReCompile', 'ReMatch', 'ReSearch', 'ReFullmatch', 'ReSub', 'ReSplit', 'ReFindall', 'ReEscape', 'ReMatchGroup', 'ReMatchGroups', 'ReMatchGroupdict', 'ReMatchStart', 'ReMatchEnd', 'ReMatchSpan', 'ReFinditer', 'ReSubn', 'RePurge', 'RePatternMatch', 'RePatternSearch', 'RePatternFullmatch', 'RePatternFindall', 'RePatternFinditer', 'RePatternSplit', 'RePatternSub', 'RePatternSubn', 'RePatternScanner', 'ReMatchExpand', 'ReMatchGetitem']
