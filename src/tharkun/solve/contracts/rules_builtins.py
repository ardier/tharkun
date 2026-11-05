"""Contracts for the builtin functions named in ticket [internal ref]'s scope.

Every raise condition below was checked on CPython 3.14.4 (this batch's exact
interpreter, `docs/contracts/FORMAT.md`) before being written down; several contradict
what the *docs alone* would suggest, which is why the ticket asks for the source read
alongside the tests:

* ``int(x)``/``float(x)`` do **not** fall back to ``__trunc__`` in 3.14 despite this
  still being folk wisdom from Python 2 -- confirmed both by reading
  ``Objects/abstract.c::PyNumber_Long`` (no ``__trunc__`` branch) and by
  ``int(JustTrunc())``/``float(JustTrunc())`` raising ``TypeError`` (mirrors CPython's
  own ``Lib/test/test_int.py::IntSubclassTests`` cases).
* ``int(x)`` tries ``__int__`` **before** ``__index__`` when a receiver defines both
  (``Objects/abstract.c::PyNumber_Long``); ``float(x)`` mirrors this with
  ``__float__`` before ``__index__``.
* Almost every one of these builtins dispatches to a receiver dunder via the type's C
  slot, not a ``getattr``/``hasattr`` probe, so a dunder that raises *any* exception
  (not just the one the docs mention) propagates unchanged -- verified per-callable in
  ``tests/contracts/test_builtins.py`` by handing each builtin a stub object whose
  dunder raises each of the 7 property classes in turn.

See ``docs/contracts/THIRD_PARTY_REFERENCES.md`` for the Nagini/ESBMC/Mopsa/CPython
cross-checks (none of their code is copied here).

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import Contract, Effects, OtherRaiseClaim, ReturnSpec, RaiseArgsCase, ScopedArgsRaiseClaim

_DOCS = "https://docs.python.org/3.14/library/functions.html"


@register_rule
class Len(Rule):
    """The rule of ``len``."""

    SAMPLES = {'len': 'x = len([1, 2])\n'}
    TWINS = {'len': 'x = len()\n'}

    names = ("len",)
    CONTRACT = Contract(
        qualname="len",
        raises=dunder_propagation_claims(
            "__len__",
            native={
                "TypeError": (
                    "the receiver has no __len__; its __len__ return value has no "
                    "__index__ (e.g. returns a str/float); or __len__ itself raises "
                    "TypeError"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # __len__ returning a negative int raises ValueError
        returns=ReturnSpec("int", "len(obj) >= 0 on success (a negative __len__ raises ValueError instead)", tested=True),
        effects=Effects(deterministic=True),
        source=(f"{_DOCS}#len", "CPython Objects/abstract.c::PyObject_Size"),
        tested=True,
    )


@register_rule
class Int(Rule):
    """The rule of ``int``."""

    SAMPLES = {'int': 'x = int("5")\n'}
    TWINS = {}

    names = ("int",)

    @staticmethod
    def literal_valid(s, base):
        """Exact str lexical grammar, before the separate process digit-limit check."""
        import unicodedata
        if base != 0 and not 2 <= base <= 36:
            return False
        normalized = ''.join(' ' if ord(c)>127 and c.isspace() else
                             str(unicodedata.decimal(c)) if c.isdecimal() else c for c in s)
        digits = normalized.strip(' \t\n\r\v\f')
        if digits[:1] in ('+', '-'):
            digits = digits[1:]
        if not digits:
            return False
        prefix = {'0b':2, '0o':8, '0x':16}.get(digits[:2].lower())
        if prefix is not None and base in (0, prefix):
            base, digits = prefix, digits[2:]
            if digits.startswith('_'):
                digits = digits[1:]
        elif base == 0:
            base = 10
            if digits.startswith('0') and any(c not in '0_' for c in digits):
                return False
        if not digits or digits.startswith('_') or digits.endswith('_') or '__' in digits:
            return False
        alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
        return all(c == '_' or c.isascii() and c.lower() in alphabet[:base] for c in digits)

    CONTRACT = Contract(
        qualname="int",
        raises=dunder_propagation_claims(
            "__int__ (tried first), then __index__; explicit base coercion can call base.__index__",
            native={
                "TypeError": (
                    "single-arg form: x has neither __int__ nor __index__ and is not "
                    "str/bytes/bytearray/a buffer object; or __int__ is defined but "
                    "returns something that is not an int (or subclass); or the "
                    "invoked __int__/__index__ itself raises TypeError. "
                    "__trunc__ is NOT consulted (removed; folk wisdom from Python 2). "
                    "two-arg form int(x, base): after successful base coercion, x is not "
                    "str/bytes/bytearray (input conversion dunders are not used); or base "
                    "cannot supply an integer index, returns the wrong type, or its __index__ raises TypeError; "
                    "wrong argument binding also raises TypeError"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # unparseable str, out-of-range base: ValueError
        returns=ReturnSpec("int", tested=True),
        effects=Effects(deterministic=False, clock_or_env=True,
                        notes='non-power-of-two string conversion observes sys.int_max_str_digits; user coercion callbacks remain open'),
        source=(
            f"{_DOCS}#int", "https://github.com/python/cpython/blob/3.14/Objects/longobject.c",
            "https://docs.python.org/3.14/library/stdtypes.html#integer-string-conversion-length-limitation",
            "CPython Objects/abstract.c::PyNumber_Long",
            "CPython Lib/test/test_int.py::IntSubclassTests (v3.14.4, __trunc__ non-fallback)",
        ),
        other_raises=(
            OtherRaiseClaim('ValueError', 'exact str s and exact int/bool base: base is neither 0 nor in [2,36]',
                            ('int() base must be >= 2 and <= 36, or 0',), True, 'invalid-base'),
            OtherRaiseClaim('ValueError', 'exact str s, valid exact int/bool base, and Int.literal_valid(s,base) is false',
                            ('invalid literal for int() with base {base}: {literal_repr}',), True, 'invalid-literal'),
            OtherRaiseClaim('ValueError', 'exact str s has valid lexical syntax in a non-power-of-two effective base; enabled sys.int_max_str_digits is exceeded by the digit count (sign, whitespace, underscores and prefix excluded)',
                            ('Exceeds the limit ({limit} digits) for integer string conversion: value has {digits} digits; use sys.set_int_max_str_digits() to increase the limit',), True, 'digit-limit'),
        ),
        notes='Explicit-base form does not call the string input conversion hooks, but base.__index__ can run. Literal claims are scoped to exact str and primitive bases; arbitrary callback exceptions remain open.',
        tested=True,
    )


@register_rule
class Float(Rule):
    """The rule of ``float``."""

    SAMPLES = {'float': 'x = float(3)\n'}
    TWINS = {'float': 'x = float(3, 0)\n'}

    names = ("float",)
    CONTRACT = Contract(
        qualname="float",
        raises=dunder_propagation_claims(
            "__float__ (tried first), then __index__",
            native={
                "TypeError": (
                    "x has neither __float__ nor __index__ and is not str/bytes/"
                    "bytearray; or __float__ is defined but returns something that is "
                    "not a float; or the invoked __float__/__index__ itself raises "
                    "TypeError. __trunc__ is NOT consulted (same as int(), confirmed "
                    "empirically on 3.14.4)"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # unparseable str: ValueError
        returns=ReturnSpec("float", tested=True),
        effects=Effects(deterministic=True),
        source=(f"{_DOCS}#float", "CPython Objects/floatobject.c::float_new_impl"),
        tested=True,
    )


@register_rule
class Str(Rule):
    """The rule of ``str``."""

    SAMPLES = {'str': 'x = str(3)\n'}
    TWINS = {'str': 'x = str(3, 0)\n'}

    names = ("str",)
    CONTRACT = Contract(
        qualname="str",
        raises=dunder_propagation_claims(
            "__str__/__repr__, or codec/error-handler callbacks in object/decoded-buffer forms",
            native={
                "TypeError": "wrong binding; non-str __str__/__repr__ result; failed decoded-form buffer acquisition (its error is converted), invalid encoding/errors kinds, or codec/error-handler callback propagation",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("str", tested=True),
        effects=Effects(deterministic=False,clock_or_env=True,may_execute_callback=True,notes="decoded forms can observe codec/error-handler registries"),
        notes="The object form falls back from object.__str__ to __repr__; decoded forms accept buffer providers plus encoding/errors. Failed buffer acquisition is rewritten to TypeError, while codec/error-handler exceptions propagate. str() with no arguments returns the empty str.",
        source=(f"{_DOCS}#func-str", "CPython Objects/object.c::object_str"),
        tested=True,
    )


@register_rule
class Abs(Rule):
    """The rule of ``abs``."""

    SAMPLES = {'abs': 'x = abs(-3)\n'}
    TWINS = {'abs': 'x = abs()\n'}

    names = ("abs",)
    CONTRACT = Contract(
        qualname="abs",
        raises=dunder_propagation_claims(
            "__abs__",
            native={"TypeError": "the receiver has no __abs__ (an __index__ alone is not enough)"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("int | float | complex", "same type family as the argument's __abs__ result; never negative for real numbers", tested=True),
        effects=Effects(deterministic=True),
        source=(f"{_DOCS}#abs", "CPython Objects/abstract.c::PyNumber_Absolute"),
        tested=True,
    )


@register_rule
class Round(Rule):
    """The rule of the builtin ``round``."""

    SAMPLES = {"round": "x = round(1.5)\n"}
    TWINS = {"round": "x = round()\n"}

    names = ("round",)
    # Coarse envelope: it cannot name the accepted keyword set; consumers must
    # not infer that every call in this envelope avoids TypeError.
    SHAPE = Shape(min_args=0, max_args=2, keywords_ok=True)
    CONTRACT = Contract(
        qualname="round",
        raises=dunder_propagation_claims(
            "__round__ or numeric ndigits.__index__",
            native={"TypeError": "wrong call shape, argument has no usable __round__, ndigits is not indexable, or __round__/ndigits.__index__ raises TypeError"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # one-argument/ndigits=None round of NaN/inf raises ValueError/OverflowError
        returns=ReturnSpec(
            "Any",
            "int for built-in int/bool with any supported ndigits and finite float without ndigits or with ndigits=None; float for float with integer ndigits; otherwise the custom __round__ result",
            tested=True,
        ),
        effects=Effects(
            deterministic=True,
            may_execute_callback=True,
            notes="Custom __round__ and numeric ndigits.__index__ may run arbitrary code; built-in numeric dispatch itself is deterministic.",
        ),
        source=(f"{_DOCS}#round", "CPython Python/bltinmodule.c::builtin_round_impl", "Objects/floatobject.c::float___round___impl"),
        notes=(
            "One-argument round uses ties-to-even for built-in numeric values and returns int; float with explicit "
            "integer ndigits returns float, and NaN/infinity raise ValueError/OverflowError only when ndigits is omitted or None. "
            "The structural Shape is a coarse envelope: CPython also accepts ndigits and number/ndigits keywords, but "
            "round() still raises TypeError and precise absint transfer covers only one direct positional exact built-in "
            "int, bool, or float. Custom __round__ results and __round__/ndigits.__index__ exceptions follow their hooks."
        ),
        tested=True,
    )


_DIVMOD_TYPES=('int','bool','float','str','NoneType','list','tuple','dict')
_DIVMOD_NUMERIC=frozenset(('int','bool','float'))
_DIVMOD_TYPE_CASES=tuple(
    RaiseArgsCase('native-types-'+left+'-'+right,
        f'exact builtin {left} and {right} operands; no target hook; native operation is unsupported',
        (f"unsupported operand type(s) for divmod(): '{left}' and '{right}'",),True)
    for left in _DIVMOD_TYPES for right in _DIVMOD_TYPES
    if not {left,right} <= _DIVMOD_NUMERIC)
_DIVMOD_BINDING_CASES=tuple(RaiseArgsCase('native-arity-'+str(n),
    f'no keywords and exactly {n} positional arguments; binding fails before operand protocols',
    (f'divmod expected 2 arguments, got {n}',),True) for n in range(9) if n!=2)+(RaiseArgsCase(
    'native-keywords','one or more keywords; native positional-only binding rejects them',
    ('divmod() takes no keyword arguments',),True),)


@register_rule
class Divmod(Rule):
    """The rule of ``divmod``."""

    SAMPLES = {'divmod': 'q, r = divmod(7, 2)\n'}
    TWINS = {'divmod': 'q, r = divmod()\n'}

    names = ("divmod",)
    CONTRACT = Contract(
        qualname="divmod",
        raises=tuple(ScopedArgsRaiseClaim(c.exc,c.condition,c.tested,args_cases=(
                RaiseArgsCase('native-zero','exact builtin int/bool/float operands, successful numeric conversion, divisor equals 0 or signed 0.0',
                              ('division by zero',),True),)) if c.exc=='ZeroDivisionError' else
                     ScopedArgsRaiseClaim(c.exc,c.condition,c.tested,args_cases=_DIVMOD_TYPE_CASES+_DIVMOD_BINDING_CASES)
                     if c.exc=='TypeError' else c for c in dunder_propagation_claims(
            "__divmod__/__rdivmod__ (their exceptions and arbitrary return values propagate)",
            native={
                "TypeError": "wrong positional-only binding, unsupported operand types (including complex), or an operand protocol raises TypeError",
                "ZeroDivisionError": "supported builtin numeric operands after successful conversion: divisor is zero; also propagates operand-hook ZeroDivisionError with arbitrary args",
            },
            tested_native=frozenset({"TypeError", "ZeroDivisionError"}),
        )),
        may_raise_other=True,
        returns=ReturnSpec("object", "builtin int/bool operands return tuple[int,int]; a float operand returns tuple[float,float]; an open __divmod__/__rdivmod__ can return any object", tested=True),
        effects=Effects(deterministic=True,may_execute_callback=True),
        source=(f"{_DOCS}#divmod", "https://github.com/python/cpython/blob/3.14/Objects/abstract.c", "CPython3.14 Objects/longobject.c and floatobject.c numeric slots"),
        notes="Mixed int/float conversion can raise OverflowError before a zero divisor is checked; arbitrary-precision int/int divmod has no float conversion. Float rounding/NaN/infinity semantics are native, not an exact-real identity.",
        tested=True,
    )


@register_rule
class Pow(Rule):
    """The rule of ``pow``."""

    SAMPLES = {'pow': 'x = pow(2, 3)\n'}
    TWINS = {'pow': 'x = pow()\n'}

    names = ("pow",)
    CONTRACT = Contract(
        qualname="pow",
        raises=dunder_propagation_claims(
            "__pow__",
            native={
                "TypeError": (
                    "the operand types have no __pow__/__rpow__; or the 3-argument "
                    "form's __pow__ does not accept a modulus argument"
                ),
                "ZeroDivisionError": (
                    "2-argument form ONLY: base is a numeric zero (int/float/complex "
                    "0) and the exponent is negative (or, for a complex base, has a "
                    "nonzero imaginary part). The 3-argument form NEVER raises "
                    "ZeroDivisionError despite the name -- pow(x, y, 0) and an "
                    "un-invertible pow(x, negative_y, mod) both raise ValueError "
                    "instead (confirmed on 3.14.4; see notes)"
                ),
            },
            tested_native=frozenset({"TypeError", "ZeroDivisionError"}),
        ),
        may_raise_other=True,  # pow(x, y, 0) and mod-inverse failure are ValueError
        returns=ReturnSpec("object", "numeric result for builtin numbers; a user power hook may return any object", tested=True),
        effects=Effects(deterministic=True),
        notes=(
            "pow(x, y, 0) raises ValueError ('pow() 3rd argument cannot be 0'), NOT "
            "ZeroDivisionError -- despite the name, confirmed on 3.14.4; only the "
            "2-argument pow(0-or-0.0-or-0j, negative) path raises ZeroDivisionError."
        ),
        source=("https://docs.python.org/3.14/library/functions.html#pow",
                "https://github.com/python/cpython/blob/3.14/Objects/abstract.c"),
        tested=True,
    )


@register_rule
class Min(Rule):
    """The rule of ``min``."""

    SAMPLES = {'min': 'x = min(1, 2)\n'}
    TWINS = {'min': 'x = min()\n'}

    names = ("min",)
    CONTRACT = Contract(
        qualname="min",
        raises=dunder_propagation_claims(
            "iterable's own __iter__/__next__, the key function (if given), or "
            "__lt__ (comparisons between elements)",
            native={
                "TypeError": "key= is given and is not callable",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # empty iterable with no default=: ValueError
        returns=ReturnSpec("the element type of the iterable", "the smallest element (or default= if the iterable is empty)", tested=True),
        effects=Effects(deterministic=True),
        source=(f"{_DOCS}#min", "CPython Python/bltinmodule.c::min_max"),
        tested=True,
    )


@register_rule
class Max(Rule):
    """The rule of ``max``."""

    SAMPLES = {'max': 'x = max(1, 2)\n'}
    TWINS = {'max': 'x = max()\n'}

    names = ("max",)
    CONTRACT = Contract(
        qualname="max",
        raises=dunder_propagation_claims(
            "iterable's own __iter__/__next__, the key function (if given), or "
            "__lt__ (comparisons between elements)",
            native={
                "TypeError": "key= is given and is not callable",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # empty iterable with no default=: ValueError
        returns=ReturnSpec("the element type of the iterable", "the largest element (or default= if the iterable is empty)", tested=True),
        effects=Effects(deterministic=True),
        source=(f"{_DOCS}#max", "CPython Python/bltinmodule.c::min_max"),
        tested=True,
    )


@register_rule
class Sum(Rule):
    """The rule of ``sum``."""

    SAMPLES = {'sum': 'x = sum([1, 2])\n'}
    TWINS = {'sum': 'x = sum()\n'}

    names = ("sum",)
    CONTRACT = Contract(
        qualname="sum",
        raises=dunder_propagation_claims(
            "iterable's own __iter__/__next__, and __radd__/__add__ (element + running total)",
            native={
                "TypeError": (
                    "the argument is not iterable; the iterable's elements are str "
                    "(sum() explicitly rejects strings: \"use ''.join(seq) instead\", "
                    "even with an empty iterable and default start=0, since the "
                    "check is on start's type once any element forces string mode -- "
                    "see notes); or an element/start does not support +"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("object", "start for an empty iterable; otherwise the native addition result, including lists and arbitrary custom addition results; builtin float/complex sums use their native compensated algorithm", tested=True),
        effects=Effects(deterministic=True),
        notes="sum(['a','b'], '') raises TypeError even though start='' is a str -- confirmed on 3.14.4.",
        source=(f"{_DOCS}#sum", "CPython Python/bltinmodule.c::builtin_sum_impl"),
        tested=True,
    )


@register_rule
class Any(Rule):
    """The rule of ``any``."""

    SAMPLES = {'any': 'x = any([0, 1])\n'}
    TWINS = {'any': 'x = any()\n'}

    names = ("any",)
    CONTRACT = Contract(
        qualname="any",
        raises=dunder_propagation_claims(
            "iterable's own __iter__/__next__, and each element's truth value (__bool__, else __len__) "
            "until the first truthy element",
            native={
                "TypeError": "the argument is not iterable, or not exactly one positional argument is given",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # anything a hook raises
        returns=ReturnSpec("bool", "True iff some element is truthy; False for an empty iterable; stops at the first truthy element", tested=True),
        effects=Effects(deterministic=True),
        notes="short-circuits: elements after the first truthy one are not read, so their hooks never run.",
        source=(f"{_DOCS}#any", "CPython Python/bltinmodule.c::builtin_any_impl"),
        tested=True,
    )


@register_rule
class All(Rule):
    """The rule of ``all``."""

    SAMPLES = {'all': 'x = all([1, 2])\n'}
    TWINS = {'all': 'x = all()\n'}

    names = ("all",)
    CONTRACT = Contract(
        qualname="all",
        raises=dunder_propagation_claims(
            "iterable's own __iter__/__next__, and each element's truth value (__bool__, else __len__) "
            "until the first falsy element",
            native={
                "TypeError": "the argument is not iterable, or not exactly one positional argument is given",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # anything a hook raises
        returns=ReturnSpec("bool", "True iff every element is truthy; True for an empty iterable; stops at the first falsy element", tested=True),
        effects=Effects(deterministic=True),
        notes="short-circuits: elements after the first falsy one are not read, so their hooks never run.",
        source=(f"{_DOCS}#all", "CPython Python/bltinmodule.c::builtin_all_impl"),
        tested=True,
    )


@register_rule
class Range(Rule):
    """The rule of ``range``."""

    SAMPLES = {'range': 'r = range(3)\n'}
    TWINS = {'range': 'r = range()\n'}

    names = ("range",)
    CONTRACT = Contract(
        qualname="range",
        raises=dunder_propagation_claims(
            "__index__ (each of start/stop/step)",
            native={
                "TypeError": "called with 0 or more than 3 arguments; or an argument has no __index__ (e.g. a float or str)",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # step == 0: ValueError
        returns=ReturnSpec("range", "an immutable arithmetic sequence object (not a list)", tested=True),
        effects=Effects(deterministic=True),
        source=(f"{_DOCS}#func-range", "CPython Objects/rangeobject.c::range_new"),
        tested=True,
    )


@register_rule
class Sorted(Rule):
    """The rule of ``sorted``."""

    SAMPLES = {'sorted': 'x = sorted([2, 1])\n'}
    TWINS = {'sorted': 'x = sorted()\n'}

    names = ("sorted",)
    CONTRACT = Contract(
        qualname="sorted",
        raises=dunder_propagation_claims(
            "iterable's own __iter__/__next__, the key function (if given), or "
            "__lt__ (comparisons between elements)",
            native={
                "TypeError": "the argument is not iterable; or key= is given and is not callable",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("list", "a new list, stably sorted ascending (or descending if reverse=True); input is not mutated", tested=True),
        effects=Effects(deterministic=True),
        source=(f"{_DOCS}#sorted", "CPython Objects/listobject.c::list_sort_impl (via PySequence_List + sort)"),
        tested=True,
    )


@register_rule
class Isinstance(Rule):
    """The rule of ``isinstance``."""

    SAMPLES = {'isinstance': 'x = isinstance(3, int)\n'}
    TWINS = {'isinstance': 'x = isinstance()\n'}

    names = ("isinstance",)
    CONTRACT = Contract(
        qualname="isinstance",
        raises=dunder_propagation_claims(
            "classinfo's __instancecheck__ (only for a classinfo with a custom metaclass)",
            native={
                "TypeError": (
                    "classinfo (2nd argument) is not a type, a tuple of types, or a "
                    "types.UnionType, once evaluation reaches it -- a tuple is "
                    "checked left-to-right and SHORT-CIRCUITS on the first match, so "
                    "e.g. isinstance(1, (int, 'not-a-type')) returns True without "
                    "raising (confirmed on 3.14.4)"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("bool", tested=True),
        effects=Effects(deterministic=True),
        notes="short-circuiting on a tuple classinfo means the TypeError condition depends on evaluation order, not just the classinfo's contents.",
        source=(f"{_DOCS}#isinstance", "CPython Objects/abstract.c::PyObject_IsInstance / object_isinstance"),
        tested=True,
    )


@register_rule
class Open(Rule):
    """The rule of ``open``."""

    SAMPLES = {'open': 'f = open("/dev/null")\n'}
    TWINS = {'open': 'f = open()\n'}

    names = ("open",)
    CONTRACT = Contract(
        qualname="open",
        raises=dunder_propagation_claims(
            "the path's __fspath__ (only when the path is not already str/bytes)",
            native={
                "TypeError": (
                    "the path is not str/bytes/os.PathLike (no __fspath__); "
                    "__fspath__ raises TypeError; __fspath__ returns something other "
                    "than str/bytes; or mode is not a str"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # FileNotFoundError/IsADirectoryError/PermissionError/OSError -- not property classes
        returns=ReturnSpec("io.TextIOWrapper | io.BufferedReader | io.BufferedWriter | ...", "a file object matching mode ('t'/'b')", tested=False),
        effects=Effects(io=True, deterministic=False, notes="opens a filesystem handle; success/failure and content depend on the filesystem"),
        notes=(
            "batch 1 covers only the argument-validation TypeError path; the "
            "filesystem-error paths (FileNotFoundError, IsADirectoryError, "
            "PermissionError) are OSError subclasses, not property classes, so they "
            "are out of this contract's raises table but are why may_raise_other=True."
        ),
        source=(f"{_DOCS}#open", "CPython Modules/_io/_iomodule.c::_io_open_impl"),
        tested=False,  # raises are tested; the returned object's exact type per mode is not (ReturnSpec.tested=False)
    )


@register_rule
class Print(Rule):
    """The rule of ``print``."""

    SAMPLES = {'print': 'print(1)\n'}
    TWINS = {}

    names = ("print",)
    CONTRACT = Contract(
        qualname="print",
        raises=dunder_propagation_claims(
            "file's .write() (an attribute lookup + call, not a dunder slot -- missing .write is a native AttributeError, a raising .write propagates the same class)",
            native={
                "TypeError": "sep, end, or file is given and is not None/str (sep, end) or does not support .write (file)",
                "AttributeError": (
                    "file (defaults to sys.stdout) has no attribute 'write'; "
                    "or flush=True and it has no attribute 'flush'; or its "
                    "write()/flush() call raises AttributeError"
                ),
            },
            tested_native=frozenset({"TypeError", "AttributeError"}),
        ),
        may_raise_other=True,  # RuntimeError if file omitted and sys.stdout is None
        returns=ReturnSpec("None", tested=True),
        effects=Effects(io=True, deterministic=True, notes="writes to file (default sys.stdout); deterministic given the same arguments and a well-behaved file object"),
        source=(f"{_DOCS}#print", "CPython Python/bltinmodule.c::builtin_print_impl"),
        tested=True,
    )


__all__ = ["Any", "All", "Len", "Int", "Float", "Str", "Abs", "Divmod", "Pow", "Min", "Max", "Sum", "Range", "Sorted", "Isinstance", "Open", "Print"]
