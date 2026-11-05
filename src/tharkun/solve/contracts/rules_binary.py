"""Contracts for ``binascii.hexlify`` / ``unhexlify``, ``zlib.compress`` / ``decompress`` and ``unicodedata.normalize`` (standing batch 13, lane 1493).

A bytes-like argument is read through ``PyObject_GetBuffer`` (``__buffer__``), so a Python class's ``__buffer__`` can propagate
any class from ``hexlify`` / ``compress`` / ``decompress``; ``unhexlify`` converts a ``__buffer__`` failure into its own
``TypeError`` (nothing propagates); ``normalize`` takes a ``str`` read at C level (a ``str`` subclass's overrides are never
called).  The native failures: a non-bytes-like argument is a ``TypeError``; ``binascii.Error`` (a ``ValueError``: odd-length or
non-hex) from ``unhexlify``; ``zlib.error`` (an ``Exception``, not a property class) from the zlib pair.  Exact on CPython 3.14.4
(``tests/contracts/test_binary_.py``).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

_ALL = frozenset(ALL_PROPERTY_CLASSES)


@register_rule
class BinasciiHexlify(Rule):
    """The rule of ``binascii.hexlify(data, sep=None, bytes_per_sep=1)``."""

    SAMPLES = {"binascii.hexlify": 'import binascii\nx = binascii.hexlify(b"ab")\n'}
    TWINS = {"binascii.hexlify": "import binascii\nx = binascii.hexlify()\n"}

    names = ("binascii.hexlify",)
    CONTRACT = Contract(
        qualname="binascii.hexlify",
        raises=dunder_propagation_claims(
            "data's __buffer__, or sep's __len__ / __buffer__ (user methods)",
            native={"TypeError": "data is not bytes-like (a str included); sep is not a str or bytes-like; wrong argument count or an unknown keyword; bytes_per_sep is not an int (its __index__ is used)"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: sep is not exactly one char (or two with bytes_per_sep == 2); OverflowError: too much data for the length
        returns=ReturnSpec("bytes", "the lowercase hex text, one char per nibble, with sep inserted every bytes_per_sep chars when given", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/binascii.html#binascii.hexlify", "CPython Modules/binascii.c::binascii_hexlify"),
        tested=True,
    )


@register_rule
class BinasciiUnhexlify(Rule):
    """The rule of ``binascii.unhexlify(data)``."""

    SAMPLES = {"binascii.unhexlify": 'import binascii\nx = binascii.unhexlify(b"6162")\n'}
    TWINS = {"binascii.unhexlify": "import binascii\nx = binascii.unhexlify()\n"}

    names = ("binascii.unhexlify",)
    CONTRACT = Contract(
        qualname="binascii.unhexlify",
        raises=never_claims(native={"TypeError": "data is not str, bytes, bytearray or a buffer-providing object (a __buffer__ failure is converted here, nothing propagates); wrong argument count or an unknown keyword"},
                            tested=_ALL),
        may_raise_other=True,  # binascii.Error (a ValueError): odd-length input or a non-hex character; ValueError: a non-ASCII str
        returns=ReturnSpec("bytes", "the raw bytes the hex text names", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=False),
        source=("https://docs.python.org/3.14/library/binascii.html#binascii.unhexlify", "CPython Modules/binascii.c::binascii_unhexlify"),
        tested=True,
    )


@register_rule
class ZlibCompress(Rule):
    """The rule of ``zlib.compress(data, level=-1, wbits=15)``."""

    SAMPLES = {"zlib.compress": 'import zlib\nx = zlib.compress(b"ab")\n'}
    TWINS = {"zlib.compress": "import zlib\nx = zlib.compress()\n"}

    names = ("zlib.compress",)
    CONTRACT = Contract(
        qualname="zlib.compress",
        raises=dunder_propagation_claims(
            "data's __buffer__, or level / wbits / memLevel / strategy's __index__ (user methods)",
            native={"TypeError": "data is not bytes-like (a str included); wrong argument count or an unknown keyword"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # zlib.error (an Exception): an invalid level, wbits, memLevel or strategy
        returns=ReturnSpec("bytes", "the zlib-wrapped (or raw / gzip, by wbits) compressed stream", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/zlib.html#zlib.compress", "CPython Modules/zlibmodule.c::zlib_compress"),
        tested=True,
    )


@register_rule
class ZlibDecompress(Rule):
    """The rule of ``zlib.decompress(data, wbits=15, bufsize=16384)``."""

    SAMPLES = {"zlib.decompress": 'import zlib\nx = zlib.decompress(zlib.compress(b"ab"))\n'}
    TWINS = {"zlib.decompress": "import zlib\nx = zlib.decompress()\n"}

    names = ("zlib.decompress",)
    CONTRACT = Contract(
        qualname="zlib.decompress",
        raises=dunder_propagation_claims(
            "data's __buffer__, or wbits / bufsize's __index__ (user methods)",
            native={"TypeError": "data is not bytes-like (a str included); wrong argument count or an unknown keyword"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # zlib.error: corrupted stream/window; invalid bufsize or allocation can also fail.
        returns=ReturnSpec("bytes", "the complete decompressed stream; bufsize is an initial allocation hint, not an output limit", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/zlib.html#zlib.decompress", "CPython Modules/zlibmodule.c::zlib_decompress"),
        tested=True,
    )


@register_rule
class UnicodedataNormalize(Rule):
    """The rule of ``unicodedata.normalize(form, unistr)``."""

    SAMPLES = {"unicodedata.normalize": 'import unicodedata\nx = unicodedata.normalize("NFC", "caf\\u00e9")\n'}
    TWINS = {"unicodedata.normalize": "import unicodedata\nx = unicodedata.normalize()\n"}

    names = ("unicodedata.normalize",)
    CONTRACT = Contract(
        qualname="unicodedata.normalize",
        raises=never_claims(native={"TypeError": "form or unistr is not a str; wrong argument count or an unknown keyword"}, tested=_ALL),
        may_raise_other=True,  # ValueError: form is not a known normalization form (NFC, NFKC, NFD, NFKD)
        returns=ReturnSpec("str", "the normalized string, composed or decomposed per form (form / casing / width are normalized)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=False),
        source=("https://docs.python.org/3.14/library/unicodedata.html#unicodedata.normalize", "CPython Modules/unicodedata.c::unicodedata_normalize"),
        tested=True,
    )


__all__ = ["BinasciiHexlify", "BinasciiUnhexlify", "ZlibCompress", "ZlibDecompress", "UnicodedataNormalize"]
