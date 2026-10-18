"""Contracts for ``bytes`` methods ([internal ref]).

``bytes.index``/``.join`` differ from ``str``/``list``/``tuple``'s comparison-based
methods in an important way: a ``bytes`` element is always an int in ``range(256)``,
compared at the C level (a memcmp-style scan), so ``index()``'s argument is
TYPE-CHECKED directly against int/bytes-like rather than duck-typed via ``__eq__`` --
an arbitrary object raises ``TypeError`` immediately, with NO chance for it to
propagate its own exception (confirmed below; contrast with ``list.index``, which
calls the arbitrary object's ``__eq__``). ``__getitem__`` still takes ``__index__``
propagation the same way ``list``/``tuple``/``str`` do, since the KEY there can be any
int-like object, not just the byte values being compared.

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/stdtypes.html#bytes"
_ALL_TESTED = frozenset(ALL_PROPERTY_CLASSES)


@register_rule
class BytesDecode(Rule):
    """The rule of ``bytes.decode``."""

    SAMPLES = {'bytes.decode': 'b = b"abc"\nx = b.decode()\n'}
    TWINS = {'bytes.decode': "b = b'abc'\nx = b.decode(0)\n"}

    names = ("bytes.decode",)
    CONTRACT = Contract(
        qualname="bytes.decode",
        raises=never_claims(
            native={"TypeError": "encoding or errors is given and is not a str"},
            tested=_ALL_TESTED,
        ),
        may_raise_other=True,  # LookupError: unknown encoding/errors; UnicodeDecodeError (a ValueError subclass)
        returns=ReturnSpec("str", tested=True),
        effects=Effects(deterministic=True),
        notes="same reasoning as str.encode: the receiver is a real bytes object and encoding/errors are type-checked directly, so no property class ever fires here.",
        source=(_DOCS, "CPython Objects/bytesobject.c::bytes_decode"),
        tested=True,
    )


@register_rule
class BytesHex(Rule):
    """The rule of ``bytes.hex``."""

    SAMPLES = {'bytes.hex': 'b = b"abc"\nx = b.hex()\n'}
    TWINS = {'bytes.hex': "b = b'abc'\nx = b.hex(0)\n"}

    names = ("bytes.hex",)
    CONTRACT = Contract(
        qualname="bytes.hex",
        raises=dunder_propagation_claims(
            "bytes_per_sep's __index__ (a USER method)",
            native={"TypeError": "sep is given and is not a single-character str/bytes, or bytes_per_sep has no __index__"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("str", "lowercase hex digits, 2 per byte, no separator by default", tested=True),
        effects=Effects(deterministic=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims with no native at all -- bytes_per_sep's __index__ can "
            "raise any of the 7 classes. Repro: b'x'.hex('-', X()) where "
            "X.__index__ raises IndexError propagates IndexError unchanged."
        ),
        source=(_DOCS, "CPython Objects/bytesobject.c::bytes_hex"),
        tested=True,
    )


@register_rule
class BytesFromhex(Rule):
    """The rule of ``bytes.fromhex``."""

    SAMPLES = {'bytes.fromhex': 'x = bytes.fromhex("ab")\n'}
    TWINS = {'bytes.fromhex': 'x = bytes.fromhex()\n'}

    names = ("bytes.fromhex",)
    CONTRACT = Contract(
        qualname="bytes.fromhex",
        raises=never_claims(
            native={"TypeError": "the argument is not str or bytes-like"},
            tested=_ALL_TESTED,
        ),
        may_raise_other=True,  # ValueError: non-hexadecimal digit found
        returns=ReturnSpec("bytes", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/bytesobject.c::bytes_fromhex"),
        tested=True,
    )


@register_rule
class BytesIndex(Rule):
    """The rule of ``bytes.index``."""

    SAMPLES = {'bytes.index': 'b = b"abc"\nx = b.index(98)\n'}
    TWINS = {'bytes.index': "b = b'abc'\nx = b.index()\n"}

    names = ("bytes.index",)
    CONTRACT = Contract(
        qualname="bytes.index",
        raises=dunder_propagation_claims(
            "start's/end's __index__ (a USER method -- sub itself never gets its "
            "__eq__ called, unlike list.index, but the slice-position arguments "
            "still go through __index__ like everywhere else)",
            native={"TypeError": "sub is neither an int nor a bytes-like object (an arbitrary object never gets its __eq__ called -- unlike list.index); or start/end has no __index__"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: not found, or an int sub outside range(256)
        returns=ReturnSpec("int", "0 <= result < len(self)", tested=True),
        effects=Effects(deterministic=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims -- start/end's __index__ can raise any of the 7 classes, "
            "independent of sub's own type-checked-not-duck-typed behavior."
        ),
        source=(_DOCS, "CPython Objects/bytesobject.c::bytes_find (index shares the scan)"),
        tested=True,
    )


@register_rule
class BytesJoin(Rule):
    """The rule of ``bytes.join``."""

    SAMPLES = {'bytes.join': 'x = b",".join([b"a"])\n'}
    TWINS = {'bytes.join': "x = b','.join()\n"}

    names = ("bytes.join",)
    CONTRACT = Contract(
        qualname="bytes.join",
        raises=dunder_propagation_claims(
            "the iterable's __iter__/__next__ (each yielded element's identity check is direct, not duck-typed)",
            native={
                "TypeError": "the argument is not iterable, or yields something that is not a bytes-like object (a str included -- bytes.join rejects str elements, unlike str.join accepting only str)",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # INHERITED REVIEW FIX: the iterable's __iter__/__next__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("bytes", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(_DOCS, "CPython Objects/bytesobject.c::bytes_join"),
        tested=True,
    )


@register_rule
class BytesGetitem(Rule):
    """The rule of ``bytes.__getitem__``."""

    SAMPLES = {'bytes.__getitem__': 'b = b"abc"\nx = b[0]\n'}
    TWINS = {}

    names = ("bytes.__getitem__",)
    CONTRACT = Contract(
        qualname="bytes.__getitem__",
        raises=dunder_propagation_claims(
            "the key argument's __index__ (for an int key)",
            native={
                "TypeError": "key is neither an int-like object nor a slice",
                "IndexError": "key is an int-like index outside [-len(self), len(self))",
            },
            tested_native=frozenset({"TypeError", "IndexError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("int (0-255) for an int key, bytes for a slice key", tested=True),
        effects=Effects(deterministic=True),
        notes="an int key returns an int (the byte value), NOT a length-1 bytes -- unlike str[i] returning a length-1 str. A slice key never raises IndexError.",
        source=(_DOCS, "CPython Objects/bytesobject.c::bytes_subscript"),
        tested=True,
    )


__all__ = ["BytesDecode", "BytesHex", "BytesFromhex", "BytesIndex", "BytesJoin", "BytesGetitem"]
