"""Contracts for ``str`` methods ([internal ref]).

The receiver is already guaranteed to be a real ``str`` (a bound method looked up on
an actual str instance, not an arbitrary object), and the "identity" arguments --
``prefix``/``suffix``, ``sep``, ``chars``, ``old``/``new``, the encoding/errors names
-- are type-checked directly against ``str``/``tuple`` rather than duck-typed via a
dunder call. But every POSITION-like argument (``start``/``end`` on ``startswith``/
``endswith``/``index``, ``maxsplit`` on ``split``, ``count`` on ``replace``) goes
through ``__index__`` like any integer position elsewhere in the language, and
``__index__`` is a USER method -- SOUNDNESS FIX (found by the gatekeeper): an earlier
draft of this module claimed these were ALL "type-checked directly" and marked every
property class but ``TypeError`` as "never" for ``startswith``/``endswith``/``split``/
``replace``/``index``; a hostile ``__index__`` on any of those position arguments
propagates whatever it raises, confirmed per-method in ``tests/contracts/test_str_.py``.
``join`` (iterates the argument), ``format`` (attribute/item access on its arguments
for fields like ``{0.attr}``/``{0[key]}``), and ``__getitem__`` (``__index__`` on the
key) were already contracted with propagation from the start; ``strip`` (``chars``)
and ``encode`` (``encoding``/``errors``) remain genuinely dunder-free -- their
arguments are type-checked directly with no fallback to a protocol.

INHERITED REVIEW FIX (combined batch 2/3 handoff): ``join``'s ``raises`` correctly
used ``dunder_propagation_claims`` from the start, but its ``may_raise_other`` was
left ``False`` and its ``effects`` had no ``may_execute_callback`` flag, both wrongly
implying a closed world for a mechanism that iterates an arbitrary user iterable.
Fixed alongside the equivalent ``bytes.join`` bug in the same batch-1 vintage.

All public str methods now have a distinct Rule class. Bounded builtin operands use
CPython semantics through both engine visitors; callback-bearing operands are never
executed by solve. Contract digests deliberately change for the expanded family.
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/stdtypes.html#str"
_ALL_TESTED = frozenset(ALL_PROPERTY_CLASSES)


@register_rule
class StrJoin(Rule):
    """The rule of ``str.join``."""

    SAMPLES = {'str.join': 'x = ",".join(["a", "b"])\n'}
    TWINS = {'str.join': "x = ','.join()\n"}

    names = ("str.join",)
    CONTRACT = Contract(
        qualname="str.join",
        raises=dunder_propagation_claims(
            "the iterable's __iter__/__next__ (and each yielded element's identity check against str)",
            native={
                "TypeError": "the argument is not iterable, or yields a non-str element",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # INHERITED REVIEW FIX: the iterable's __iter__/__next__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("str", "self.join(iterable of N strs) has length sum(len(s) for s in iterable) + len(self) * (N - 1)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(_DOCS, "CPython Objects/stringlib/join.h::STRINGLIB(join)"),
        tested=True,
    )


@register_rule
class StrStartswith(Rule):
    """The rule of ``str.startswith``."""

    SAMPLES = {'str.startswith': 's = "abc"\nx = s.startswith("a")\n'}
    TWINS = {'str.startswith': "s = 'abc'\nx = s.startswith()\n"}

    names = ("str.startswith",)
    CONTRACT = Contract(
        qualname="str.startswith",
        raises=dunder_propagation_claims(
            "start's/end's __index__ (a USER method -- any exception it raises propagates uncaught)",
            native={
                "TypeError": (
                    "prefix is not a str and not a tuple of str -- a tuple is "
                    "checked left-to-right and SHORT-CIRCUITS on the first match, "
                    "so 'x'.startswith(('x', 5)) returns True without raising; or "
                    "start/end is given and has no __index__"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("bool", tested=True),
        effects=Effects(deterministic=True),
        notes=(
            "prefix is type-checked directly (str/tuple-of-str); start/end are "
            "NOT -- they go through __index__ like any position argument. "
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft claimed "
            "prefix/start/end were ALL type-checked directly and marked every "
            "class but TypeError 'never' -- start/end's __index__ can raise any "
            "of them. Repro: 'x'.startswith('x', X()) where X.__index__ raises "
            "IndexError propagates IndexError unchanged."
        ),
        source=(_DOCS, "CPython Objects/unicodeobject.c::unicode_startswith"),
        tested=True,
    )


@register_rule
class StrEndswith(Rule):
    """The rule of ``str.endswith``."""

    SAMPLES = {'str.endswith': 's = "abc"\nx = s.endswith("c")\n'}
    TWINS = {'str.endswith': "s = 'abc'\nx = s.endswith()\n"}

    names = ("str.endswith",)
    CONTRACT = Contract(
        qualname="str.endswith",
        raises=dunder_propagation_claims(
            "start's/end's __index__ (same as startswith's)",
            native={
                "TypeError": "suffix is not a str and not a tuple of str (same short-circuit as startswith); or start/end is given and has no __index__",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("bool", tested=True),
        effects=Effects(deterministic=True),
        notes="SOUNDNESS FIX (gatekeeper): same start/end __index__-propagation fix as startswith.",
        source=(_DOCS, "CPython Objects/unicodeobject.c::unicode_endswith"),
        tested=True,
    )


@register_rule
class StrSplit(Rule):
    """The rule of ``str.split``."""

    SAMPLES = {'str.split': 's = "abc"\nx = s.split(",")\n'}
    TWINS = {}

    names = ("str.split",)
    CONTRACT = Contract(
        qualname="str.split",
        raises=dunder_propagation_claims(
            "maxsplit's __index__ (a USER method)",
            native={"TypeError": "sep is given and is neither None nor a str; or maxsplit has no __index__"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # sep="" raises ValueError
        returns=ReturnSpec("list[str]", tested=True),
        effects=Effects(deterministic=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims -- maxsplit's __index__ can raise any of the 7 classes, "
            "not just TypeError. Repro: 'x'.split(maxsplit=X()) where "
            "X.__index__ raises IndexError propagates IndexError unchanged."
        ),
        source=(_DOCS, "CPython Objects/unicodeobject.c::unicode_split"),
        tested=True,
    )


@register_rule
class StrStrip(Rule):
    """The rule of ``str.strip``."""

    SAMPLES = {'str.strip': 's = "abc"\nx = s.strip()\n'}
    TWINS = {'str.strip': "s = 'abc'\nx = s.strip(0)\n"}

    names = ("str.strip",)
    CONTRACT = Contract(
        qualname="str.strip",
        raises=never_claims(
            native={"TypeError": "chars is given and is not None or a str"},
            tested=_ALL_TESTED,
        ),
        may_raise_other=False,
        returns=ReturnSpec("str", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/unicodeobject.c::unicode_strip"),
        tested=True,
    )


@register_rule
class StrReplace(Rule):
    """The rule of ``str.replace``."""

    SAMPLES = {'str.replace': 's = "abc"\nx = s.replace("a", "b")\n'}
    TWINS = {'str.replace': "s = 'abc'\nx = s.replace()\n"}

    names = ("str.replace",)
    CONTRACT = Contract(
        qualname="str.replace",
        raises=dunder_propagation_claims(
            "count's __index__ (a USER method)",
            native={"TypeError": "old or new is not a str; or count is given and has no __index__"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("str", tested=True),
        effects=Effects(deterministic=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims -- count's __index__ can raise any of the 7 classes."
        ),
        source=(_DOCS, "CPython Objects/unicodeobject.c::unicode_replace"),
        tested=True,
    )


@register_rule
class StrFormat(Rule):
    """The rule of ``str.format``."""

    SAMPLES = {'str.format': 'x = "{}".format(1)\n'}
    TWINS = {}

    names = ("str.format",)
    CONTRACT = Contract(
        qualname="str.format",
        raises=dunder_propagation_claims(
            "field access on an argument (e.g. {0.attr} -> __getattr__, {0[k]} -> __getitem__)",
            native={
                "IndexError": "a replacement field's positional index is >= the number of positional args given",
                "KeyError": "a replacement field's name is not among the keyword args given",
            },
            tested_native=frozenset({"IndexError", "KeyError"}),
        ),
        may_raise_other=True,  # malformed format string, unknown format code, unknown conversion: ValueError
        returns=ReturnSpec("str", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "https://docs.python.org/3.14/library/string.html#format-string-syntax", "CPython Objects/stringlib/unicode_format.h"),
        tested=True,
    )


@register_rule
class StrEncode(Rule):
    """The rule of ``str.encode``."""

    SAMPLES = {'str.encode': 's = "abc"\nx = s.encode()\n'}
    TWINS = {'str.encode': "s = 'abc'\nx = s.encode(0)\n"}

    names = ("str.encode",)
    CONTRACT = Contract(
        qualname="str.encode",
        raises=never_claims(
            native={"TypeError": "encoding or errors is given and is not a str"},
            tested=_ALL_TESTED,
        ),
        may_raise_other=True,  # LookupError: unknown encoding/errors handler; UnicodeEncodeError (a ValueError subclass)
        returns=ReturnSpec("bytes", tested=True),
        effects=Effects(deterministic=True),
        notes="the receiver is a real str and encoding/errors are type-checked directly, so no property class ever fires here -- only LookupError/UnicodeEncodeError, neither of which is one of the 7.",
        source=(_DOCS, "CPython Objects/unicodeobject.c::PyUnicode_AsEncodedString"),
        tested=True,
    )


@register_rule
class StrIndex(Rule):
    """The rule of ``str.index``."""

    SAMPLES = {'str.index': 's = "abc"\nx = s.index("b")\n'}
    TWINS = {'str.index': "s = 'abc'\nx = s.index()\n"}

    names = ("str.index",)
    CONTRACT = Contract(
        qualname="str.index",
        raises=dunder_propagation_claims(
            "start's/end's __index__ (a USER method)",
            native={"TypeError": "sub is not a str; or start/end is given and has no __index__"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: substring not found
        returns=ReturnSpec("int", "0 <= result < len(self)", tested=True),
        effects=Effects(deterministic=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims -- start/end's __index__ can raise any of the 7 classes."
        ),
        source=(_DOCS, "CPython Objects/stringlib/find.h"),
        tested=True,
    )


@register_rule
class StrGetitem(Rule):
    """The rule of ``str.__getitem__``."""

    SAMPLES = {'str.__getitem__': 's = "abc"\nx = s[0]\n'}
    TWINS = {}

    names = ("str.__getitem__",)
    CONTRACT = Contract(
        qualname="str.__getitem__",
        raises=dunder_propagation_claims(
            "the key argument's __index__ (for an int key)",
            native={
                "TypeError": "key is neither an int-like object nor a slice",
                "IndexError": "key is an int-like index outside [-len(self), len(self))",
            },
            tested_native=frozenset({"TypeError", "IndexError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("str", "a single-character str for an int key, or str for a slice key; a slice never raises IndexError (out-of-range bounds are clamped)", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/unicodeobject.c::unicode_subscript"),
        tested=True,
    )




def _string_contract(name, result, type_error, notes):
    callbacks = name in {"center","count","expandtabs","find","format_map","ljust","rfind","rindex","rjust","rsplit","splitlines","translate","zfill"}
    native = {"TypeError": type_error}
    if name == "format_map":
        native["KeyError"] = "missing named replacement field"
    raises = (dunder_propagation_claims("index/field/table/truth protocols when required",native=native,tested_native=frozenset(native)) if callbacks
              else never_claims(native=native,tested=frozenset({"AssertionError","AttributeError","FloatingPointError","IndexError","KeyError","TypeError","ZeroDivisionError"})))
    return Contract(
        qualname="str."+name, raises=raises, may_raise_other=callbacks or bool(notes),
        returns=ReturnSpec(result,"exact native result; finite strings retain exact length",tested=True),
        effects=Effects(may_execute_callback=callbacks), source=(_DOCS+"."+name,"CPython Objects/unicodeobject.c"),
        notes=notes+"; receiver must be str; invalid call shape raises TypeError", tested=True)


@register_rule
class StrCapitalize(Rule):
    names = ('str.capitalize',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.capitalize': 'x = "a B".capitalize()\n'}
    TWINS = {}
    CONTRACT = _string_contract('capitalize', 'str', 'wrong argument count', '')


@register_rule
class StrCasefold(Rule):
    names = ('str.casefold',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.casefold': 'x = "Straße".casefold()\n'}
    TWINS = {}
    CONTRACT = _string_contract('casefold', 'str', 'wrong argument count', '')


@register_rule
class StrCenter(Rule):
    names = ('str.center',)
    SAMPLES = {'str.center': 'x = "a".center(4, "-")\n'}
    TWINS = {}
    CONTRACT = _string_contract('center', 'str', 'width has no __index__; fillchar is not a str; wrong argument count', 'TypeError if fillchar length is not one')


@register_rule
class StrCount(Rule):
    names = ('str.count',)
    SAMPLES = {'str.count': 'x = "aba".count("a", 0, 3)\n'}
    TWINS = {}
    CONTRACT = _string_contract('count', 'int', 'sub is not str; start/end has no __index__; wrong argument count', '')


@register_rule
class StrExpandtabs(Rule):
    names = ('str.expandtabs',)
    SAMPLES = {'str.expandtabs': 'x = "a\\tb".expandtabs(4)\n'}
    TWINS = {}
    CONTRACT = _string_contract('expandtabs', 'str', 'tabsize has no __index__; wrong argument count', '')


@register_rule
class StrFind(Rule):
    names = ('str.find',)
    SAMPLES = {'str.find': 'x = "aba".find("b", 0, 3)\n'}
    TWINS = {}
    CONTRACT = _string_contract('find', 'int', 'sub is not str; start/end has no __index__; wrong argument count', 'returns -1 if absent')


@register_rule
class StrFormatMap(Rule):
    names = ('str.format_map',)
    SAMPLES = {'str.format_map': 'x = "{x:03d}".format_map({"x": 4})\n'}
    TWINS = {}
    CONTRACT = _string_contract('format_map', 'str', 'mapping has no __getitem__; wrong argument count', 'KeyError for a missing named field; ValueError for positional fields, malformed syntax/spec or mixed numbering; field attributes/items and __format__ propagate')


@register_rule
class StrIsalnum(Rule):
    names = ('str.isalnum',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.isalnum': 'x = "Ab1".isalnum()\n'}
    TWINS = {}
    CONTRACT = _string_contract('isalnum', 'bool', 'wrong argument count', '')


@register_rule
class StrIsalpha(Rule):
    names = ('str.isalpha',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.isalpha': 'x = "Ab1".isalpha()\n'}
    TWINS = {}
    CONTRACT = _string_contract('isalpha', 'bool', 'wrong argument count', '')


@register_rule
class StrIsascii(Rule):
    names = ('str.isascii',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.isascii': 'x = "Ab1".isascii()\n'}
    TWINS = {}
    CONTRACT = _string_contract('isascii', 'bool', 'wrong argument count', '')


@register_rule
class StrIsdecimal(Rule):
    names = ('str.isdecimal',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.isdecimal': 'x = "Ab1".isdecimal()\n'}
    TWINS = {}
    CONTRACT = _string_contract('isdecimal', 'bool', 'wrong argument count', '')


@register_rule
class StrIsdigit(Rule):
    names = ('str.isdigit',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.isdigit': 'x = "Ab1".isdigit()\n'}
    TWINS = {}
    CONTRACT = _string_contract('isdigit', 'bool', 'wrong argument count', '')


@register_rule
class StrIsidentifier(Rule):
    names = ('str.isidentifier',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.isidentifier': 'x = "Ab1".isidentifier()\n'}
    TWINS = {}
    CONTRACT = _string_contract('isidentifier', 'bool', 'wrong argument count', '')


@register_rule
class StrIslower(Rule):
    names = ('str.islower',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.islower': 'x = "Ab1".islower()\n'}
    TWINS = {}
    CONTRACT = _string_contract('islower', 'bool', 'wrong argument count', '')


@register_rule
class StrIsnumeric(Rule):
    names = ('str.isnumeric',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.isnumeric': 'x = "Ab1".isnumeric()\n'}
    TWINS = {}
    CONTRACT = _string_contract('isnumeric', 'bool', 'wrong argument count', '')


@register_rule
class StrIsprintable(Rule):
    names = ('str.isprintable',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.isprintable': 'x = "Ab1".isprintable()\n'}
    TWINS = {}
    CONTRACT = _string_contract('isprintable', 'bool', 'wrong argument count', '')


@register_rule
class StrIsspace(Rule):
    names = ('str.isspace',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.isspace': 'x = "Ab1".isspace()\n'}
    TWINS = {}
    CONTRACT = _string_contract('isspace', 'bool', 'wrong argument count', '')


@register_rule
class StrIstitle(Rule):
    names = ('str.istitle',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.istitle': 'x = "Ab1".istitle()\n'}
    TWINS = {}
    CONTRACT = _string_contract('istitle', 'bool', 'wrong argument count', '')


@register_rule
class StrIsupper(Rule):
    names = ('str.isupper',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.isupper': 'x = "Ab1".isupper()\n'}
    TWINS = {}
    CONTRACT = _string_contract('isupper', 'bool', 'wrong argument count', '')


@register_rule
class StrLjust(Rule):
    names = ('str.ljust',)
    SAMPLES = {'str.ljust': 'x = "a".ljust(4, "-")\n'}
    TWINS = {}
    CONTRACT = _string_contract('ljust', 'str', 'width has no __index__; fillchar is not str; wrong argument count', 'TypeError if fillchar length is not one')


@register_rule
class StrLower(Rule):
    names = ('str.lower',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.lower': 'x = "ABC".lower()\n'}
    TWINS = {}
    CONTRACT = _string_contract('lower', 'str', 'wrong argument count', '')


@register_rule
class StrLstrip(Rule):
    names = ('str.lstrip',)
    SAMPLES = {'str.lstrip': 'x = " a ".lstrip()\n'}
    TWINS = {}
    CONTRACT = _string_contract('lstrip', 'str', 'chars is neither str nor None; wrong argument count', '')


@register_rule
class StrMaketrans(Rule):
    names = ('str.maketrans',)
    SAMPLES = {'str.maketrans': 'x = str.maketrans("ab", "xy")\n'}
    TWINS = {}
    CONTRACT = _string_contract('maketrans', 'dict', 'one-argument form requires dict; three-argument form requires str; wrong argument count', 'ValueError if the first two strings differ in length, or a mapping string key has length other than one; mapping keys must be int or one-character str')


@register_rule
class StrPartition(Rule):
    names = ('str.partition',)
    SAMPLES = {'str.partition': 'x = "a:b".partition(":")\n'}
    TWINS = {}
    CONTRACT = _string_contract('partition', 'tuple[str,str,str]', 'sep is not str; wrong argument count', 'ValueError if sep is empty')


@register_rule
class StrRemoveprefix(Rule):
    names = ('str.removeprefix',)
    SAMPLES = {'str.removeprefix': 'x = "abc".removeprefix("a")\n'}
    TWINS = {}
    CONTRACT = _string_contract('removeprefix', 'str', 'prefix is not str; wrong argument count', '')


@register_rule
class StrRemovesuffix(Rule):
    names = ('str.removesuffix',)
    SAMPLES = {'str.removesuffix': 'x = "abc".removesuffix("c")\n'}
    TWINS = {}
    CONTRACT = _string_contract('removesuffix', 'str', 'suffix is not str; wrong argument count', '')


@register_rule
class StrRfind(Rule):
    names = ('str.rfind',)
    SAMPLES = {'str.rfind': 'x = "aba".rfind("a", 0, 3)\n'}
    TWINS = {}
    CONTRACT = _string_contract('rfind', 'int', 'sub is not str; start/end has no __index__; wrong argument count', 'returns -1 if absent')


@register_rule
class StrRindex(Rule):
    names = ('str.rindex',)
    SAMPLES = {'str.rindex': 'x = "aba".rindex("a", 0, 3)\n'}
    TWINS = {}
    CONTRACT = _string_contract('rindex', 'int', 'sub is not str; start/end has no __index__; wrong argument count', 'ValueError if substring is absent')


@register_rule
class StrRjust(Rule):
    names = ('str.rjust',)
    SAMPLES = {'str.rjust': 'x = "a".rjust(4, "-")\n'}
    TWINS = {}
    CONTRACT = _string_contract('rjust', 'str', 'width has no __index__; fillchar is not str; wrong argument count', 'TypeError if fillchar length is not one')


@register_rule
class StrRpartition(Rule):
    names = ('str.rpartition',)
    SAMPLES = {'str.rpartition': 'x = "a:b".rpartition(":")\n'}
    TWINS = {}
    CONTRACT = _string_contract('rpartition', 'tuple[str,str,str]', 'sep is not str; wrong argument count', 'ValueError if sep is empty')


@register_rule
class StrRsplit(Rule):
    names = ('str.rsplit',)
    SAMPLES = {'str.rsplit': 'x = "a:b:c".rsplit(":", 1)\n'}
    TWINS = {}
    CONTRACT = _string_contract('rsplit', 'list[str]', 'sep is neither str nor None; maxsplit has no __index__; wrong argument count', 'ValueError if sep is empty')


@register_rule
class StrRstrip(Rule):
    names = ('str.rstrip',)
    SAMPLES = {'str.rstrip': 'x = " a ".rstrip()\n'}
    TWINS = {}
    CONTRACT = _string_contract('rstrip', 'str', 'chars is neither str nor None; wrong argument count', '')


@register_rule
class StrSplitlines(Rule):
    names = ('str.splitlines',)
    SAMPLES = {'str.splitlines': 'x = "a\\nb".splitlines(True)\n'}
    TWINS = {}
    CONTRACT = _string_contract('splitlines', 'list[str]', 'wrong argument count; keepends __bool__ returns non-bool or __len__ returns non-int', '')


@register_rule
class StrSwapcase(Rule):
    names = ('str.swapcase',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.swapcase': 'x = "Ab".swapcase()\n'}
    TWINS = {}
    CONTRACT = _string_contract('swapcase', 'str', 'wrong argument count', '')


@register_rule
class StrTitle(Rule):
    names = ('str.title',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.title': 'x = "a b".title()\n'}
    TWINS = {}
    CONTRACT = _string_contract('title', 'str', 'wrong argument count', '')


@register_rule
class StrTranslate(Rule):
    names = ('str.translate',)
    SAMPLES = {'str.translate': 'x = "ab".translate({97: "xy", 98: None})\n'}
    TWINS = {}
    CONTRACT = _string_contract('translate', 'str', 'table has no __getitem__, or yields a value other than int, str or None; wrong argument count', 'missing table keys (LookupError) leave the character unchanged; ValueError for an integer outside Unicode range; other table callback errors propagate')


@register_rule
class StrUpper(Rule):
    names = ('str.upper',)
    SHAPE = Shape(1, 1)
    SAMPLES = {'str.upper': 'x = "ab".upper()\n'}
    TWINS = {}
    CONTRACT = _string_contract('upper', 'str', 'wrong argument count', '')


@register_rule
class StrZfill(Rule):
    names = ('str.zfill',)
    SAMPLES = {'str.zfill': 'x = "-3".zfill(4)\n'}
    TWINS = {}
    CONTRACT = _string_contract('zfill', 'str', 'width has no __index__; wrong argument count', '')


@register_rule
class StrDunderFormat(Rule):
    names = ('str.__format__',)
    SAMPLES = {'str.__format__': 'x = "ab".__format__(">4")\n'}
    TWINS = {}
    CONTRACT = _string_contract('__format__', 'str', 'format_spec is not str; wrong argument count', 'ValueError for an invalid string format specification')


@register_rule
class StrContains(Rule):
    names = ('str.__contains__',)
    SAMPLES = {'str.__contains__': 'x = "abc".__contains__("b")\n'}
    TWINS = {}
    CONTRACT = _string_contract('__contains__', 'bool', 'operand is not str; wrong argument count', '')

__all__ = ['StrJoin', 'StrStartswith', 'StrEndswith', 'StrSplit', 'StrStrip', 'StrReplace', 'StrFormat', 'StrEncode', 'StrIndex', 'StrGetitem', 'StrCapitalize', 'StrCasefold', 'StrCenter', 'StrCount', 'StrExpandtabs', 'StrFind', 'StrFormatMap', 'StrIsalnum', 'StrIsalpha', 'StrIsascii', 'StrIsdecimal', 'StrIsdigit', 'StrIsidentifier', 'StrIslower', 'StrIsnumeric', 'StrIsprintable', 'StrIsspace', 'StrIstitle', 'StrIsupper', 'StrLjust', 'StrLower', 'StrLstrip', 'StrMaketrans', 'StrPartition', 'StrRemoveprefix', 'StrRemovesuffix', 'StrRfind', 'StrRindex', 'StrRjust', 'StrRpartition', 'StrRsplit', 'StrRstrip', 'StrSplitlines', 'StrSwapcase', 'StrTitle', 'StrTranslate', 'StrUpper', 'StrZfill', 'StrDunderFormat', 'StrContains']
