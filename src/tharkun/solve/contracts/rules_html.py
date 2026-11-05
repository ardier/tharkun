"""Both public HTML utilities, natively checked on CPython 3.14.4 ([internal ref]).

Open operands can run replace/containment/truth hooks and return arbitrary
objects. Only exact builtin strings have the documented str result. Decimal
character references in unescape consult the process integer-string digit limit.
"""
from __future__ import annotations

from .rules import Rule, register_rule
from .types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim, ReturnSpec


_BY_QUALNAME = {
    q: Contract(
        qualname=q,
        raises=tuple(RaiseClaim(exc, 'unknown') for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('object', 'str for exact builtin str operands; open hooks can return other objects', tested=True),
        effects=Effects(may_execute_callback=True, clock_or_env=q == 'html.unescape',
                        deterministic=q != 'html.unescape',
                        notes='replace/containment/truth hooks on open operands; unescape numeric references '
                              'can depend on sys.set_int_max_str_digits'),
        source=(f'https://docs.python.org/3.14/library/html.html#{q}',
                'https://github.com/python/cpython/blob/3.14/Lib/html/__init__.py'),
        tested=True,
    )
    for q in ('html.escape', 'html.unescape')
}


@register_rule
class HtmlFunction(Rule):
    names = ('html.escape', 'html.unescape')
    SAMPLES = {'html.escape': 'import html\nx = html.escape("<&")\n',
               'html.unescape': 'import html\nx = html.unescape("&lt;&amp;")\n'}
    TWINS = {q: f'import html\nx = {q}()\n' for q in names}

    @classmethod
    def contract_for(cls, qualname):
        return _BY_QUALNAME[qualname]
