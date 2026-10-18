"""String formatting rules: percent, format/repr/ascii and f-string AST operations.

CPython supplies native semantics; engines consume each Rule through their visitor.
FormattedValue applies conversion before the evaluated spec, JoinedStr joins only str parts.
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/stdtypes.html#printf-style-string-formatting"


@register_rule
class StrMod(Rule):
    """The rule of ``str.__mod__``."""

    SAMPLES = {'str.__mod__': 'x = "%d" % 3\n'}
    TWINS = {}

    names = ("str.__mod__",)
    CONTRACT = Contract(
        qualname="str.__mod__",
        raises=dunder_propagation_claims(
            "each conversion's value -- e.g. %s calls str() on it (batch 1's "
            "str() propagation applies transitively); %d/%i/%x/etc. require a "
            "real number and raise natively rather than converting",
            native={
                "TypeError": (
                    "the number of values doesn't match the number of %-specifiers "
                    "(both 'not enough arguments' and 'not all arguments converted' "
                    "are this SAME class); or a numeric specifier (%d, %f, %x, ...) "
                    "is given a value with no numeric conversion"
                ),
                "KeyError": "the MAPPING form (%(name)s % {...}) references a key the mapping does not have",
            },
            tested_native=frozenset({"TypeError", "KeyError"}),
        ),
        may_raise_other=True,  # ValueError: unsupported/malformed format character
        returns=ReturnSpec("str", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "a single non-tuple, non-mapping right-hand side is treated as ONE "
            "value, not unpacked -- '%s' % [1, 2] formats the list itself, it does "
            "NOT raise for 'too many arguments' the way '%s' % (1, 2) would."
        ),
        source=(_DOCS, "CPython Objects/unicodeobject.c::unicode_format (PyUnicode_Format)"),
        tested=True,
    )


@register_rule
class Format(Rule):
    """The rule of ``format``."""

    SAMPLES = {'format': 'x = format(3, "d")\n'}
    TWINS = {'format': 'x = format()\n'}

    names = ("format",)
    CONTRACT = Contract(
        qualname="format",
        raises=dunder_propagation_claims(
            "the object's __format__(format_spec)",
            native={
                "TypeError": (
                    "format_spec is not a str; or the object relies on the DEFAULT "
                    "object.__format__ (no custom override) and format_spec is "
                    "non-empty -- the default implementation accepts ONLY '' "
                    "(falling back to str(x)) and raises for anything else"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: a type's own __format__ rejects a spec it doesn't understand (e.g. int/str's own "Unknown format code")
        returns=ReturnSpec("str", "__format__ is REQUIRED to return a str -- a non-str return is the propagating object's own bug, not format()'s", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="this is exactly what f\"{x:spec}\" and str.format's {0:spec} fields call -- one dispatch point behind three surface syntaxes (this builtin, f-strings, str.format from batch 1).",
        source=(f"{_DOCS.rsplit('#', 1)[0]}#format", "CPython Objects/object.c::PyObject_Format"),
        tested=True,
    )


@register_rule
class Repr(Rule):
    """The rule of ``repr``."""

    SAMPLES = {'repr': 'x = repr(3)\n'}
    TWINS = {'repr': 'x = repr()\n'}

    names = ("repr",)
    CONTRACT = Contract(
        qualname="repr",
        raises=dunder_propagation_claims(
            "the object's __repr__ -- every object has ONE by default (object.__repr__, '<module.Class object at 0x...>'), so there is no 'missing protocol' case the way len()/int() have",
            native={
                "TypeError": "__repr__ is overridden and returns something that is not a str",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: __repr__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("str", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="f\"{x!r}\" is exactly repr(x), then the format-spec mini-language applied to ITS result.",
        source=("https://docs.python.org/3.14/library/functions.html#repr", "CPython Objects/object.c::PyObject_Repr"),
        tested=True,
    )


@register_rule
class Ascii(Rule):
    """The rule of ``ascii``."""

    SAMPLES = {'ascii': 'x = ascii("a")\n'}
    TWINS = {'ascii': 'x = ascii()\n'}

    names = ("ascii",)
    CONTRACT = Contract(
        qualname="ascii",
        raises=dunder_propagation_claims(
            "the object's __repr__ (ascii() is repr() plus a post-processing escape pass -- same dispatch, same propagation)",
            native={
                "TypeError": "__repr__ is overridden and returns something that is not a str -- identical condition to repr()'s",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: __repr__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("str", "identical to repr(x), except every non-ASCII character is backslash-escaped (\\xHH/\\uHHHH/\\UHHHHHHHH)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="f\"{x!a}\" is exactly ascii(x), then the format-spec mini-language applied to ITS result.",
        source=("https://docs.python.org/3.14/library/functions.html#ascii", "CPython Objects/object.c::PyObject_ASCII (wraps PyObject_Repr)"),
        tested=True,
    )




@register_rule
class FormattedValue(Rule):
    names = ("syntax.FormattedValue",)
    SHAPE = Shape(1, 3)
    SAMPLES = {"syntax.FormattedValue": 'x = f"{3!r:>4}"\n'}
    TWINS = {}
    CONTRACT = Contract(
        qualname="syntax.FormattedValue",
        raises=dunder_propagation_claims("conversion (__str__/__repr__) then __format__",
            native={"TypeError": "format_spec is not str, or conversion/format returns non-str"},
            tested_native=frozenset({"TypeError"})),
        may_raise_other=True,
        returns=ReturnSpec("str", tested=True), effects=Effects(),
        source=("https://docs.python.org/3.14/reference/lexical_analysis.html#f-strings",),
        notes="ValueError for an invalid format spec; !s, !r and !a convert before nested spec evaluation on CPython 3.14.4; formatting follows the evaluated spec",
        tested=True,
    )

    def native(self, value, spec="", conversion=-1):
        if conversion != -1:
            value = {115: str, 114: repr, 97: ascii}[conversion](value)
        return format(value, spec)


@register_rule
class JoinedStr(Rule):
    names = ("syntax.JoinedStr",)
    SHAPE = Shape(0, None)
    SAMPLES = {"syntax.JoinedStr": 'x = f"a{3}b"\n'}
    TWINS = {}
    CONTRACT = Contract(
        qualname="syntax.JoinedStr",
        raises=never_claims(
            native={"TypeError": "a part is not str"},
            tested=frozenset({"AssertionError", "AttributeError", "FloatingPointError", "IndexError", "KeyError", "TypeError", "ZeroDivisionError"})),
        may_raise_other=False, returns=ReturnSpec("str", "sum of the formatted part lengths", tested=True),
        effects=Effects(may_execute_callback=False),
        source=("https://docs.python.org/3.14/reference/lexical_analysis.html#f-strings",), tested=True,
    )

    def native(self, *parts):
        return "".join(parts)


def native_limit(qualname, args, kwargs):
    """Bound eager builtin folds before allocating their result, never invoke target hooks."""
    # ponytail: 4096-unit eager folds; add symbolic widths if larger formats need precision.
    if qualname == "str.encode" or (qualname == "str" and (len(args) > 1 or "encoding" in kwargs or "errors" in kwargs)):
        encoding = kwargs.get("encoding", args[1] if len(args) > 1 else "utf-8")
        errors = kwargs.get("errors", args[2] if len(args) > 2 else "strict")
        if type(encoding) is not str or type(errors) is not str:
            return "native TypeError: codec/error names must be str"
        if encoding not in {"utf-8", "ascii", "latin-1"} or errors not in {"strict", "ignore", "replace"}:
            return "custom codec/error handler is not modelled"
    formatting = qualname in {"str.__mod__", "str.format", "str.format_map", "str.__format__", "format", "syntax.FormattedValue"}
    def bounded(value):
        import re
        if type(value) in {str, bytes}:
            return len(value) <= 4096 and (not formatting or all(len(n) < 5 and int(n) <= 4096 for n in re.findall(r"[0-9]+", str(value))))
        if type(value) in {int, bool}:
            return not formatting or (qualname == "str.__mod__" and len(args) == 2 and args[0] == "%c") or abs(value) <= 4096
        if type(value) in {tuple, list}:
            return len(value) <= 64 and all(bounded(x) for x in value)
        if type(value) is dict:
            return len(value) <= 64 and all(bounded(k) and bounded(v) for k,v in value.items())
        return True
    if not all(bounded(x) for x in (*args,*kwargs.values())):
        return "native operand/format size exceeds 4096"
    if qualname.startswith("re.") and not qualname.startswith("re.Match.") and qualname not in {"re.escape", "re.purge"} and args:
        import re
        pattern = args[0]
        flags_at = {"re.compile": 1, "re.match": 2, "re.search": 2, "re.fullmatch": 2,
                    "re.findall": 2, "re.finditer": 2, "re.split": 3, "re.sub": 4, "re.subn": 4}.get(qualname)
        flags = kwargs.get("flags", args[flags_at] if flags_at is not None and len(args) > flags_at else 0)
        if type(pattern) is re.Pattern:
            flags, pattern = pattern.flags, pattern.pattern
        if type(flags) in {int, bool, re.RegexFlag} and flags & re.DEBUG:
            return "regex DEBUG output is not modelled"
        if type(pattern) in {str, bytes} and type(flags) in {int, bool, re.RegexFlag}:
            # ponytail: one unbranched repeat and no backreferences; use a cancellable native worker for wider regex folds.
            try:
                tree = re._parser.parse(pattern, flags)
                repeats = 0
                def safe_regex(nodes, repeated=False):
                    nonlocal repeats
                    for op, data in nodes:
                        if op in {re._constants.MAX_REPEAT, re._constants.MIN_REPEAT, re._constants.POSSESSIVE_REPEAT}:
                            repeats += 1
                            if repeated or repeats > 1 or not safe_regex(data[-1], True):
                                return False
                        elif op == re._constants.SUBPATTERN:
                            if not safe_regex(data[-1], repeated):
                                return False
                        elif op == re._constants.BRANCH:
                            if repeated or not all(safe_regex(branch, repeated) for branch in data[1]):
                                return False
                        elif op == re._constants.ATOMIC_GROUP:
                            if not safe_regex(data, repeated):
                                return False
                        elif op in {re._constants.ASSERT, re._constants.ASSERT_NOT}:
                            if not safe_regex(data[-1], repeated):
                                return False
                        elif op in {re._constants.GROUPREF, re._constants.GROUPREF_EXISTS}:
                            return False
                    return True
                if not safe_regex(tree):
                    return "native regex backtracking budget is not modelled"
            except (re.PatternError, ValueError, OverflowError):
                pass  # Native evaluation records the real malformed-pattern exception.
            except RecursionError:
                return "native regex parser recursion limit"
    if qualname in {"str.center","str.ljust","str.rjust","str.zfill","str.expandtabs"}:
        width=kwargs.get("tabsize" if qualname.endswith("expandtabs") else "width",args[1] if len(args)>1 else 8)
        if type(width) in {int,bool} and abs(width)>4096:
            return "native width exceeds 4096"
    return None

__all__ = ['StrMod', 'Format', 'Repr', 'Ascii', 'FormattedValue', 'JoinedStr']
