"""Text wrapping utility contracts, CPython 3.14.4 ([internal ref]).

TextWrapper construction, its two public methods and the utility functions
have separate native contracts and absint visits over shared transfers.
Open text, indentation, width and predicate operands can execute arbitrary hooks.
"""
from __future__ import annotations

from ._helpers import never_claims
from .rules import Rule, register_rule
from .types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim, ReturnSpec

_NAMES = ('wrap', 'fill', 'shorten', 'dedent', 'indent')
_BY_QUALNAME = {
    'textwrap.'+name: Contract(
        qualname='textwrap.'+name,
        raises=tuple(RaiseClaim(exc, 'unknown') for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('list[object]' if name == 'wrap' else 'str',
                           'wrap lines are str for builtin string configuration; custom indentation addition can return other elements'
                           if name == 'wrap' else 'the final builtin str.join result', tested=True),
        effects=Effects(may_execute_callback=True, clock_or_env=name in ('wrap', 'fill', 'shorten'),
                        deterministic=name not in ('wrap', 'fill', 'shorten'),
                        notes='text/string methods, width/indentation protocols, configuration truth and indent predicate can run target hooks; width error formatting '
                              'in wrap/fill/shorten also observes the process integer-string digit limit'),
        source=(f'https://docs.python.org/3.14/library/textwrap.html#textwrap.{name}',
                'https://github.com/python/cpython/blob/3.14/Lib/textwrap.py'),
        tested=True,
    ) for name in _NAMES
}


@register_rule
class TextwrapFunction(Rule):
    names = tuple(_BY_QUALNAME)
    SAMPLES = {'textwrap.wrap': 'import textwrap\nx=textwrap.wrap("hello world",5)\n',
               'textwrap.fill': 'import textwrap\nx=textwrap.fill("hello world",5)\n',
               'textwrap.shorten': 'import textwrap\nx=textwrap.shorten("hello world",8)\n',
               'textwrap.dedent': 'import textwrap\nx=textwrap.dedent("  hello\\n  world")\n',
               'textwrap.indent': 'import textwrap\nx=textwrap.indent("hello\\nworld","> ")\n'}
    TWINS = {q: f'import textwrap\nx={q}()\n' for q in names}

    @classmethod
    def contract_for(cls, qualname):
        return _BY_QUALNAME[qualname]


@register_rule
class TextwrapConstructor(Rule):
    names = ('textwrap.TextWrapper',)
    SAMPLES = {'textwrap.TextWrapper': 'import textwrap\nx=textwrap.TextWrapper(5)\n'}
    TWINS = {'textwrap.TextWrapper': 'import textwrap\nx=textwrap.TextWrapper(unknown=True)\n'}
    CONTRACT = Contract(
        qualname='textwrap.TextWrapper',
        raises=never_claims({'TypeError': 'wrong argument binding for TextWrapper constructor'},
                            tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=True,
        returns=ReturnSpec('textwrap.TextWrapper', 'new instance stores all configuration values without converting them', tested=True),
        effects=Effects(may_execute_callback=False),
        source=('https://docs.python.org/3.14/library/textwrap.html#textwrap.TextWrapper',
                'https://github.com/python/cpython/blob/3.14/Lib/textwrap.py'),
        tested=True,
    )


_METHOD_CONTRACTS = {
    'textwrap.TextWrapper.'+name: Contract(
        qualname='textwrap.TextWrapper.'+name,
        raises=tuple(RaiseClaim(exc, 'unknown') for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('object' if name == 'wrap' else 'str',
                           'canonical instances return a line list; overridden receiver methods may return any object'
                           if name == 'wrap' else 'the final builtin str.join result', tested=True),
        effects=Effects(may_execute_callback=True, clock_or_env=True, deterministic=False,
                        notes='reads mutable settings and may call text/configuration hooks; invalid width repr observes integer digit limit'),
        source=(f'https://docs.python.org/3.14/library/textwrap.html#textwrap.TextWrapper.{name}',
                'https://github.com/python/cpython/blob/3.14/Lib/textwrap.py'),
        tested=True,
    ) for name in ('wrap', 'fill')
}


@register_rule
class TextwrapMethod(Rule):
    names = tuple(_METHOD_CONTRACTS)
    SAMPLES = {q: f'import textwrap\nw=textwrap.TextWrapper(5)\nx=w.{q.rsplit(".",1)[1]}("hello world")\n' for q in names}
    TWINS = {q: f'import textwrap\nw=textwrap.TextWrapper(5)\nx=w.{q.rsplit(".",1)[1]}()\n' for q in names}

    @classmethod
    def contract_for(cls, qualname):
        return _METHOD_CONTRACTS[qualname]

    def native(self, receiver, *args, **kwargs):
        import textwrap
        return getattr(textwrap.TextWrapper, self.qualname.rsplit('.',1)[1])(receiver, *args, **kwargs)
