"""Plain builtin exception constructors and their argument storage, CPython 3.14.4 ([internal ref])."""
from ._helpers import never_claims
from .rules import Rule, register_rule
from .types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec, ConstantArgsRaiseClaim

PLAIN_EXCEPTION_NAMES = (
    'BaseException', 'Exception', 'TypeError', 'ValueError', 'RuntimeError',
    'LookupError', 'IndexError', 'KeyError', 'AssertionError', 'ArithmeticError',
    'ZeroDivisionError', 'FloatingPointError', 'OverflowError',
)

_CONTRACTS = {name: Contract(
    qualname=name,
    raises=tuple(ConstantArgsRaiseClaim(exc, 'any keyword argument; positional arguments are stored without conversion',
                                       tested=True, args=(name+'() takes no keyword arguments',))
                 if exc == 'TypeError' else claim
                 for exc, claim in zip(ALL_PROPERTY_CLASSES, never_claims(tested=frozenset(ALL_PROPERTY_CLASSES)))),
    may_raise_other=True,
    returns=ReturnSpec(name, 'fresh canonical exception; args is the tuple of positional constructor arguments, preserving their references', tested=True),
    effects=Effects(may_execute_callback=False),
    source=('https://docs.python.org/3.14/library/exceptions.html#BaseException',
            'https://github.com/python/cpython/blob/3.14/Objects/exceptions.c'),
    tested=True,
    notes='Canonical builtin construction only. Source subclass __init__ can run arbitrary code; formatting an exception may call argument hooks. Allocation failures are outside the seven property classes.',
) for name in PLAIN_EXCEPTION_NAMES}


@register_rule
class ExceptionConstructor(Rule):
    names = PLAIN_EXCEPTION_NAMES
    SAMPLES = {name: f'x={name}("message", 7)\n' for name in names}
    TWINS = {name: f'x={name}(unknown=True)\n' for name in names}

    @classmethod
    def contract_for(cls, qualname):
        return _CONTRACTS[qualname]
