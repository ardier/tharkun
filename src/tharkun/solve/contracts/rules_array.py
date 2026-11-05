"""CPython 3.14.4 array construction; open conversion and warning hooks stay open."""
from ._helpers import dunder_propagation_claims
from .rules import Rule,Shape,register_rule
from .types import Contract,Effects,OtherRaiseClaim,RaiseArgsCase,ReturnSpec,ScopedArgsRaiseClaim


TYPECODES='bBuwhHiIlLqQfd'
BAD_TYPECODE_ARGS=('bad typecode (must be b, B, u, w, h, H, i, I, l, L, q, Q, f or d)',)
_CODE_TYPES={'NoneType':'None','int':'int','bool':'bool','float':'float','bytes':'bytes',
             'list':'list','tuple':'tuple','dict':'dict'}
_TYPE_CASES=tuple(RaiseArgsCase('typecode-'+kind,
    f'valid positional-only binding and exact builtin {kind} typecode argument',
    (f'array() argument 1 must be a unicode character, not {shown}',),True)
    for kind,shown in _CODE_TYPES.items())
_LENGTH_CASES=tuple(RaiseArgsCase('typecode-length-'+str(length),
    f'valid positional-only binding and exact builtin str typecode of {length} codepoints',
    (f'array() argument 1 must be a unicode character, not a string of length {length}',),True)
    for length in (0,2,3,4,5,6,7,8))
_ARITY_CASES=(RaiseArgsCase('arity-0','no keywords and zero positional arguments',
    ('array() takes at least 1 argument (0 given)',),True),)+tuple(
    RaiseArgsCase('arity-'+str(n),f'no keywords and {n} positional arguments',
        (f'array() takes at most 2 arguments ({n} given)',),True) for n in range(3,9))
_INIT_CASES=tuple(RaiseArgsCase('initializer-'+kind,
    f'exact typecode B, two positional arguments, no keywords and exact builtin {kind} initializer',
    (f"'{kind}' object is not iterable",),True) for kind in ('NoneType','int','bool','float','complex'))
_KEYWORDS=RaiseArgsCase('keywords','one or more keywords on the exact array.array constructor',
    ('array.array() takes no keyword arguments',),True)
_BYTE_BOUNDARIES=(
    ('b',-129,'signed char is less than minimum'),('b',128,'signed char is greater than maximum'),
    ('B',-1,'unsigned byte integer is less than minimum'),('B',256,'unsigned byte integer is greater than maximum'))


@register_rule
class ArrayArray(Rule):
    """A fresh typed array; the abstract interpreter reuses its existing storage model."""

    names=('array.array',)
    SAMPLES={'array.array':"import array\narray.array('B',[0])\n"}
    TWINS={'array.array':'import array\narray.array()\n'}
    SHAPE=Shape(min_args=1,max_args=2)
    CONTRACT=Contract(
        qualname='array.array',
        raises=tuple(ScopedArgsRaiseClaim(c.exc,c.condition,c.tested,
            args_cases=tuple(RaiseArgsCase(x.case,'Without audit/warning intervention: '+x.condition,x.args,x.tested)
                            for x in _TYPE_CASES+_LENGTH_CASES+_ARITY_CASES+_INIT_CASES+(_KEYWORDS,)))
            if c.exc=='TypeError' else c for c in dunder_propagation_claims(
                'initializer iteration, element __index__/__float__, warning handling or array.__new__ audit callbacks',
                native={'TypeError':'invalid positional-only binding or typecode form; initializer is not accepted/iterable; an item cannot be converted to the destination numeric/Unicode type; or a callback raises TypeError'},
                tested_native=frozenset({'TypeError'}))),
        may_raise_other=True,
        other_raises=(
            OtherRaiseClaim('ValueError',
                'Without audit/warning intervention: valid positional-only binding; exact str typecode is one codepoint absent from bBuwhHiIlLqQfd',
                BAD_TYPECODE_ARGS,True,'native-bad-typecode'),
            OtherRaiseClaim('ValueError',
                'Without audit/warning intervention: exact typecode d and exact bytes initializer whose byte length is not a multiple of native itemsize',
                ('bytes length not a multiple of item size',),True,'native-byte-count'),
            OtherRaiseClaim('OverflowError',
                'Without audit/warning intervention: valid f/d construction from an exact int item whose conversion to native binary64 overflows before storage',
                ('int too large to convert to float',),True,'native-int-to-float'),
        )+tuple(OtherRaiseClaim('OverflowError',
            f'Without audit/warning intervention: exact typecode {code}, no keywords and exact builtin list initializer containing the single builtin int {value}',
            (message,),True,f'native-{code}-{value}') for code,value,message in _BYTE_BOUNDARIES),
        returns=ReturnSpec('array.array','fresh storage; typecode controls element representation and itemsize; binary byte order is native',tested=True),
        effects=Effects(mutates_args=('initializer',),clock_or_env=True,deterministic=False,may_execute_callback=True,
            notes='Advances iterator initializers; does not alias copied list/array storage. Construction emits array.__new__ audit callbacks; native u consults warning filters and may call warning handlers or raise a warning as an exception. Open conversion hooks can mutate arbitrary state; numeric B/list transfers have no such hooks.'),
        source=('https://docs.python.org/3.14/library/array.html#array.array',
                'https://github.com/python/cpython/blob/3.14/Modules/arraymodule.c'),
        notes='Native integer storage raises OverflowError outside its C type range; args depend on typecode and conversion path, with byte boundary scopes above. Float32 storage rounds and can saturate to infinity; huge integer-to-float conversion can fail first. Unicode/u/w, raw-byte and platform-long constructors are outside the current abstract transfer, which refuses them. Raise-site metadata alone is not a synthesized exception proof.',
        tested=True,
    )
