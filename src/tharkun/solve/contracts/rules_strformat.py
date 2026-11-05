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


def native_limit(qualname, args, kwargs, *, _work_check=None):
    """Bound eager builtin folds before allocating their result, never invoke target hooks."""
    # ponytail: 4096-unit eager folds; add symbolic widths if larger formats need precision.
    if _work_check is not None and not _work_check():
        return 'native evaluation deadline exhausted'
    import re
    # The existing RegexFlag operator consumer uses "format" for its bit-work guard.
    flag_operands = (qualname == 'format' and not kwargs and any(type(v) is re.RegexFlag for v in args)
                     and all(type(v) in {int, bool, re.RegexFlag} for v in args))
    if flag_operands and any((abs(v).bit_length() > 4096 if type(v) is re.RegexFlag else abs(v) > 4096) for v in args):
        return 'native bit operand exceeds 4096'
    if qualname == "str.encode" or (qualname == "str" and (len(args) > 1 or "encoding" in kwargs or "errors" in kwargs)):
        encoding = kwargs.get("encoding", args[1] if len(args) > 1 else "utf-8")
        errors = kwargs.get("errors", args[2] if len(args) > 2 else "strict")
        if type(encoding) is not str or type(errors) is not str:
            return "native TypeError: codec/error names must be str"
        if encoding not in {"utf-8", "ascii", "latin-1"} or errors not in {"strict", "ignore", "replace", "xmlcharrefreplace", "backslashreplace"}:  # builtin handlers only
            return "custom codec/error handler is not modelled"
    formatting = not flag_operands and qualname in {"str.__mod__", "str.format", "str.format_map", "str.__format__", "format", "syntax.FormattedValue"}
    remaining, active = 4096, set()
    refusal = "native operand/format size exceeds 4096"

    def charge(cost):
        nonlocal remaining, refusal
        if _work_check is not None and not _work_check():
            refusal = "native evaluation deadline exhausted"
            return False
        remaining -= cost
        return remaining >= 0

    def bounded(value, nested=False):
        if not charge(0):
            return False
        if type(value) in {str, bytes}:
            if len(value) > remaining:
                return False
            return charge(max(1, len(repr(value)) if nested else len(value)))
        if type(value) in {int, bool}:
            # Decimal digit upper bound without constructing an unbounded int repr.
            return charge((4 if value else 5) if type(value) is bool else abs(value).bit_length() * 30103 // 100000 + 1 + (value < 0))
        if type(value) in {tuple, list, dict}:
            marker = id(value)
            if marker in active:
                return False
            # Count delimiters, separators and each occurrence, including shared aliases.
            punctuation = 2 + 2 * max(0, len(value) - 1)
            punctuation += 2 * len(value) if type(value) is dict else int(type(value) is tuple and len(value) == 1)
            if not charge(punctuation):
                return False
            active.add(marker)
            try:
                if type(value) is dict:
                    return all(bounded(k, True) and bounded(v, True) for k, v in value.items())
                return all(bounded(x, True) for x in value)
            finally:
                active.remove(marker)
        if formatting and type(value) not in {float, type(None)}:
            return False
        return charge(len(repr(value)) if type(value) in {float, type(None)} else 1)
    try:
        if not all(bounded(x) for x in (*args, *kwargs.values())):
            return refusal
    except (RecursionError, MemoryError):
        return "native operand traversal resource exhausted"
    if formatting:
        import re
        from string import Formatter
        from _string import formatter_field_name_split

        # Check atomic builtin fields before counting their aggregate output, never build the full result here.
        remaining = 4096

        def amount(text):
            number = 0
            for digit in text:
                number = 10 * number + int(digit)
                if number > 4096:
                    return 4097
            return number

        def bit_output(value, presentation, alternate=False):
            if type(value) not in {int, bool} or presentation not in {'b', 'o', 'x', 'X'}:
                return True
            divisor = {'b': 1, 'o': 3, 'x': 4, 'X': 4}[presentation]
            size = max(1, (abs(value).bit_length() + divisor - 1) // divisor) + (value < 0) + 2 * alternate
            return size <= 4096

        def field_work(value, spec='', conversion=None):
            nonlocal refusal
            if not charge(0):
                return False
            if type(spec) is not str:
                raise TypeError('format spec is not str')
            if '{' in spec or '}' in spec:
                refusal = 'native nested format spec cost is not modelled'
                return False
            if conversion is not None:
                value = {'s': str, 'r': repr, 'a': ascii}[conversion](value)
            numeric = spec[2:] if len(spec) >= 2 and spec[1] in '<>=^' else spec
            width, dot, precision = numeric.partition('.')
            if any(amount(n) > 4096 for n in re.findall(r'\d+', width)):
                return False
            # String precision truncates; numeric precision can allocate proportional output/work.
            if dot and type(value) is not str and any(amount(n) > 4096 for n in re.findall(r'\d+', precision)):
                return False
            if not bit_output(value, spec[-1:] or 's', '#' in spec):
                return False
            if not charge(0):
                return False
            return charge(len(format(value, spec)))

        try:
            if qualname in {'str.format', 'str.format_map'} and args:
                automatic = 0
                for literal, field, spec, conversion in Formatter().parse(args[0]):
                    if not charge(len(literal)):
                        return refusal
                    if field is None:
                        continue
                    if field == '':
                        field, automatic = str(automatic), automatic + 1
                    first, rest = formatter_field_name_split(field)
                    if qualname == 'str.format_map':
                        if type(first) is not str or len(args) < 2 or type(args[1]) is not dict:
                            raise TypeError('format_map requires a closed mapping field')
                        value = args[1][first]
                    else:
                        value = args[first + 1] if type(first) is int else kwargs[first]
                    for attribute, key in rest:
                        if attribute or type(value) not in {str, bytes, list, tuple, dict}:
                            return 'native format field lookup cost is not modelled'
                        value = value[key]
                    if not field_work(value, spec, conversion):
                        return refusal
            elif qualname == 'str.__mod__' and len(args) == 2:
                template, values = args[0], args[1] if type(args[1]) is tuple else (args[1],)
                pos, used = 0, 0
                while pos < len(template):
                    start = template.find('%', pos)
                    if start < 0:
                        if not charge(len(template) - pos):
                            return refusal
                        break
                    if not charge(start - pos):
                        return refusal
                    pos = start + 1
                    if template[pos:pos + 1] == '%':
                        pos += 1
                        if not charge(1):
                            return refusal
                        continue
                    mapping = template[pos:pos + 1] == '('
                    key_start, depth = pos + 1, 0
                    if mapping:
                        while pos < len(template):
                            depth += (template[pos] == '(') - (template[pos] == ')')
                            pos += 1
                            if depth == 0:
                                break
                        if depth:
                            raise ValueError('unclosed percent mapping key')
                        key = template[key_start:pos - 1]
                    match = re.match(r'[-+#0 ]*(\*|[0-9]*)(?:\.(\*|[0-9]*))?[hlL]?([diouxXeEfFgGcrsa])', template[pos:])
                    if match is None:
                        return 'native percent format cost is not modelled'
                    operands, sizes = [], []
                    for at, number in enumerate(match.groups()[:2]):
                        if number == '*':
                            number = values[used]
                            used += 1
                            operands.append(number)
                            if type(number) not in {int, bool}:
                                raise TypeError('percent width/precision is not int')
                            size = abs(number) if at == 0 else max(0, number)
                        else:
                            size = amount(number or '0')
                        sizes.append(size)
                    value = args[1][key] if mapping else values[used]
                    if not mapping:
                        used += 1
                    if sizes[0] > 4096 or (sizes[1] > 4096 and match[3] not in 'sra'):
                        return refusal
                    if not bit_output(value, match[3], '#' in match[0]):
                        return refusal
                    pos += match.end()
                    operands.append(value)
                    if not charge(0):
                        return refusal
                    rendered = template[start:pos] % (args[1] if mapping else tuple(operands))
                    if not charge(len(rendered)):
                        return refusal
            elif args:
                conversion = ({115: 's', 114: 'r', 97: 'a'}[args[2]]
                              if qualname == 'syntax.FormattedValue' and len(args) > 2 and args[2] != -1 else None)
                if not field_work(args[0], args[1] if len(args) > 1 else '', conversion):
                    return refusal
        except (TypeError, ValueError, KeyError, IndexError, AttributeError, OverflowError):
            pass  # The unchanged native consumer records the actual conversion/lookup error.
        except (RecursionError, MemoryError):
            return 'native format resource exhausted'
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
    if _work_check is not None and not _work_check():
        return "native evaluation deadline exhausted"
    return None

__all__ = ['StrMod', 'Format', 'Repr', 'Ascii', 'FormattedValue', 'JoinedStr']
