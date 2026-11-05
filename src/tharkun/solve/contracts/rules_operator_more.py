"""The remaining operator callable names ([internal ref]), CPython 3.14.4.

Protocol contracts describe dispatch; recipes also need a contract for each
module callable's own name. This family follows MathFunction's registry pattern.
operator.call is deliberately omitted: it directly executes an arbitrary supplied
callable, outside the recipe admission policy. Existing add/attrgetter/itemgetter
contracts stay in rules_operator.py. A methodcaller construction does not yet
invoke the named method. No new precise proof shape is claimed.
"""
from __future__ import annotations

from tharkun.solve.contracts._helpers import never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim, ReturnSpec

_DOCS = 'https://docs.python.org/3.14/library/operator.html'
_UNKNOWN = tuple(RaiseClaim(exc, 'unknown') for exc in ALL_PROPERTY_CLASSES)
_UNARY = {
    'abs': '-2', 'neg': '2', 'pos': '2', 'inv': '2', 'invert': '2',
    'index': '2', 'not_': '0', 'truth': '1', 'length_hint': '[1]',
    'is_none': 'None', 'is_not_none': 'None',
}
_BINARY = {
    'and_': '3, 1', 'concat': '[1], [2]', 'contains': '[1], 1',
    'countOf': '[1, 1], 1', 'delitem': '[1], 0', 'eq': '1, 1',
    'floordiv': '5, 2', 'ge': '2, 1', 'getitem': '[1], 0', 'gt': '2, 1',
    'iadd': '[1], [2]', 'iand': '3, 1', 'iconcat': '[1], [2]',
    'ifloordiv': '5, 2', 'ilshift': '1, 2', 'imod': '5, 2', 'imul': '[1], 2',
    'indexOf': '[0, 1], 1', 'ior': '1, 2', 'ipow': '2, 3',
    'irshift': '4, 1', 'is_': 'None, None', 'is_not': 'None, None',
    'isub': '3, 1', 'itruediv': '5, 2', 'ixor': '3, 1', 'le': '1, 2',
    'lshift': '1, 2', 'lt': '1, 2', 'mod': '5, 2', 'mul': '2, 3',
    'ne': '1, 2', 'or_': '1, 2', 'pow': '2, 3', 'rshift': '4, 1',
    'sub': '3, 1', 'truediv': '5, 2', 'xor': '3, 1',
}
_SAMPLES = {name: f'import operator\nx = operator.{name}({args})\n'
            for name, args in (_UNARY | _BINARY).items()}
_SAMPLES['setitem'] = 'import operator\na = [1]\nx = operator.setitem(a, 0, 2)\n'
_SAMPLES['methodcaller'] = "import operator\nx = operator.methodcaller('count', 'a')\n"
for _name in ('matmul', 'imatmul'):
    _SAMPLES[_name] = ('import operator\nclass Matrix:\n'
                       '    def __matmul__(self, other):\n        return 1\n'
                       f'x = operator.{_name}(Matrix(), Matrix())\n')
_IDENTITY = frozenset(('is_', 'is_not', 'is_none', 'is_not_none'))
_BOOL = _IDENTITY | {'contains', 'not_', 'truth'}
_INT = frozenset(('countOf', 'indexOf', 'index', 'length_hint'))
_MUTATES = frozenset(('contains', 'countOf', 'indexOf', 'setitem', 'delitem',
                      'iadd', 'iand', 'iconcat', 'ifloordiv', 'ilshift', 'imatmul',
                      'imod', 'imul', 'ior', 'ipow', 'irshift', 'isub', 'itruediv', 'ixor'))
_BY_QUALNAME = {}
for _name in sorted(_SAMPLES):
    _q = 'operator.' + _name
    _ret = ('bool' if _name in _BOOL else 'int' if _name in _INT else
            'None' if _name in ('setitem', 'delitem') else 'Callable' if _name == 'methodcaller' else 'object')
    _BY_QUALNAME[_q] = Contract(
        qualname=_q,
        raises=(never_claims(native={'TypeError': 'wrong argument count or unexpected keyword'},
                             tested=frozenset(ALL_PROPERTY_CLASSES)) if _name in _IDENTITY else _UNKNOWN),
        may_raise_other=True,
        returns=ReturnSpec(_ret, 'identity predicate without operand inspection' if _name in _IDENTITY else
                           'the native dispatch result; methodcaller defers method invocation', tested=True),
        effects=Effects(mutates_receiver=_name in _MUTATES, may_execute_callback=_name not in _IDENTITY,
                        notes='in-place/item operations can mutate; containment/count/index searches can consume iterators; '
                              'arbitrary operand hooks remain open except for identity predicates'),
        source=(f'{_DOCS}#{_q}', 'CPython 3.14 Modules/_operator.c', f'CPython 3.14 Lib/operator.py::{_name}'),
        tested=True,
    )


@register_rule
class OperatorFunction(Rule):
    """One rule per remaining module callable, sharing the native dispatcher family."""

    names = tuple(_BY_QUALNAME)
    SAMPLES = {'operator.'+name: text for name, text in _SAMPLES.items()}
    TWINS = {q: f'import operator\nx = {q}()\n' for q in names}

    @classmethod
    def contract_for(cls, qualname):
        return _BY_QUALNAME[qualname]
