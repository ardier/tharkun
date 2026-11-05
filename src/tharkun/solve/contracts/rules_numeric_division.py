"""Native int/float division slots and exact zero-divisor args, CPython 3.14.4 ([internal ref])."""
from .rules import Rule, Shape, register_rule
from .types import ALL_PROPERTY_CLASSES, ConstantArgsRaiseClaim, Contract, Effects, RaiseClaim, ReturnSpec

DIVISION_METHODS = {owner+'.__'+reverse+op+'__': (owner,op,bool(reverse))
                    for owner in ('int','float') for op in ('truediv','floordiv','mod') for reverse in ('','r')}
_CONTRACTS = {}
for q,(owner,op,reverse) in DIVISION_METHODS.items():
    condition = ('correctly bound receiver; other is an int (including bool)' if owner=='int' else
                 'correctly bound receiver; other is an int/float and any int-to-float conversion succeeds')
    condition += '; '+('receiver' if reverse else 'other')+' is numerically zero'
    _CONTRACTS[q] = Contract(
        qualname=q,
        raises=tuple(ConstantArgsRaiseClaim(exc,condition,tested=True,args=('division by zero',))
                     if exc=='ZeroDivisionError' else
                     RaiseClaim(exc,'wrong arity, keyword arguments, or receiver has the wrong native type',tested=True)
                     if exc=='TypeError' else RaiseClaim(exc,'never',tested=True)
                     for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec(('float' if owner=='float' or op=='truediv' else 'int')+' | NotImplementedType',
                           'numeric quotient/floor quotient/remainder; unsupported other type returns NotImplemented',tested=True),
        effects=Effects(may_execute_callback=False),
        source=('https://docs.python.org/3.14/library/stdtypes.html#numeric-types-int-float-complex',
                'https://github.com/python/cpython/blob/3.14/Objects/'+('longobject.c' if owner=='int' else 'floatobject.c')),
        tested=True,
        notes='Native slots only, including inherited slots on numeric subclasses. Operator dispatch may instead execute overridden source methods. Huge int conversion or integer true-division result can raise OverflowError before returning; allocation failure remains open.',
    )


@register_rule
class NumericDivision(Rule):
    names=tuple(DIVISION_METHODS)
    SHAPE=Shape(2,2)
    SAMPLES={q:f'x={q}('+('2.0,5' if reverse else '5.0,2')+')\n' if owner=='float' else
               f'x={q}('+('2,5' if reverse else '5,2')+')\n'
             for q,(owner,op,reverse) in DIVISION_METHODS.items()}
    TWINS={q:f'x={q}()\n' for q in names}

    @classmethod
    def contract_for(cls,qualname):return _CONTRACTS[qualname]
