"""String's word helper, natively checked on CPython 3.14.4 ([internal ref]).

Formatter parsing covers the original unbound method only; replacement
methods and formatting field lookup still require separate receiver models.
"""
import string

from ._helpers import dunder_propagation_claims
from .rules import Rule, Shape, register_rule
from .types import ALL_PROPERTY_CLASSES, Contract, Effects, OtherRaiseClaim, RaiseClaim, ReturnSpec


@register_rule
class StringCapwords(Rule):
    names=('string.capwords',)
    SAMPLES={names[0]:"import string\nx=string.capwords('  hello WORLD  ')\n"}
    TWINS={names[0]:"import string\nx=string.capwords()\n"}
    CONTRACT=Contract(
        qualname=names[0],
        raises=dunder_propagation_claims(
            's.split(sep), iteration of its result, sep truth and sep.join',
            native={'TypeError':'wrong binding, invalid builtin split separator after successful join lookup, or split result contains non-str elements',
                    'AttributeError':'s has no split or a truthy separator has no join'},
            tested_native=frozenset({'TypeError','AttributeError'})),
        other_raises=(OtherRaiseClaim('ValueError','exact builtin str s and exact builtin str sep equal to the empty string',
                                     ('empty separator',),True,'empty-separator'),),
        may_raise_other=True,
        returns=ReturnSpec('object','str for exact builtin str input and str-or-None separator; an open truthy separator join override can return any object',tested=True),
        effects=Effects(may_execute_callback=True,notes='open split, iteration, separator truth and join may run arbitrary target hooks'),
        source=('https://docs.python.org/3.14/library/string.html#string.capwords',
                'https://github.com/python/cpython/blob/3.14/Lib/string/__init__.py'),
        tested=True,
    )


@register_rule
class StringFormatterParse(Rule):
    """Construct a parser; syntax errors belong to its later iteration."""

    names = ('string.Formatter.parse',)
    SHAPE = Shape(max_args=2, keywords_ok=True)
    SAMPLES = {names[0]: "import string\nresult = string.Formatter().parse('hello {name}')\n"}
    TWINS = {names[0]: "import string\nresult = string.Formatter().parse()\n"}
    CONTRACT = Contract(
        qualname=names[0],
        raises=tuple(RaiseClaim(exc,
            'invalid Python argument binding, or format_string is not a str instance'
            if exc == 'TypeError' else 'never', tested=True)
            for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('Iterator[tuple[str, str | None, str | None, str | None]]',
            'fresh stateful formatteriterator; malformed replacement syntax raises ValueError only when consumed; '
            'no field lookup, conversion, or formatting is performed', tested=True),
        effects=Effects(may_execute_callback=True,
            notes='the original method ignores self and reads native str storage without user hooks; '
                  'conservative callback flag retains lifetime cleanup; iterator consumption is not modeled'),
        source=('https://docs.python.org/3.14/library/string.html#string.Formatter.parse',
                'https://github.com/python/cpython/blob/v3.14.4/Lib/string/__init__.py#L303',
                'https://github.com/python/cpython/blob/v3.14.4/Objects/stringlib/unicode_format.h#L1103'),
        tested=True,
        notes='Only the original standard-library implementation, not arbitrary parse overrides. '
              'Both self and format_string accept keyword binding; self need not be a Formatter instance.',
    )

    def native(_rule, *args, **kwargs):
        # The native Python method permits a keyword literally named self.
        return string.Formatter.parse(*args, **kwargs)
