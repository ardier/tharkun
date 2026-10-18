"""Contracts for io.StringIO/io.BytesIO ([internal ref] batch 3).

StringIO.write requires a str instance, but normalizing a subclass can call its
__str__. BytesIO.write obtains a buffer, which can invoke Python __buffer__ and
__release_buffer__ methods. read/seek also dispatch through argument __index__.
These callback effects are distinct from native closed-file/position errors.

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/io.html"
_ALL_TESTED = frozenset(ALL_PROPERTY_CLASSES)


@register_rule
class IoStringIOWrite(Rule):
    """The rule of ``io.StringIO.write``."""

    SAMPLES = {'io.StringIO.write': 'import io\nf = io.StringIO()\nf.write("a")\n'}
    TWINS = {'io.StringIO.write': 'import io\nf = io.StringIO()\nf.write()\n'}

    names = ("io.StringIO.write",)
    CONTRACT = Contract(
        qualname="io.StringIO.write",
        raises=dunder_propagation_claims(
            "a str subclass's __str__ during normalization",
            native={"TypeError": "the argument is not a str instance, its __str__ returns a non-str, or the callback raises TypeError"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: I/O operation on closed file
        returns=ReturnSpec("int", "the number of characters written after str-subclass normalization; no partial writes", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True, notes="writes at the current stream position, which ADVANCES by the written length; overwrites existing content in place if not at the end, it does not insert"),
        source=(f"{_DOCS}#io.StringIO.write", "CPython Modules/_io/stringio.c::stringio_write"),
        tested=True,
    )


@register_rule
class IoStringIORead(Rule):
    """The rule of ``io.StringIO.read``."""

    SAMPLES = {'io.StringIO.read': 'import io\nf = io.StringIO()\nx = f.read()\n'}
    TWINS = {}

    names = ("io.StringIO.read",)
    CONTRACT = Contract(
        qualname="io.StringIO.read",
        raises=dunder_propagation_claims(
            "size's __index__ (a USER method) -- any exception it raises propagates uncaught",
            native={"TypeError": "size is given and has no __index__"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: closed file
        returns=ReturnSpec("str", "from the current position to size chars ahead, or to the end if size is negative or omitted", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True, notes="ADVANCES the stream position by the number of characters actually read"),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims -- size's __index__ can raise any of the 7 classes, not "
            "just TypeError."
        ),
        source=(f"{_DOCS}#io.StringIO", "CPython Modules/_io/stringio.c::stringio_read"),
        tested=True,
    )


@register_rule
class IoStringIOGetvalue(Rule):
    """The rule of ``io.StringIO.getvalue``."""

    SAMPLES = {'io.StringIO.getvalue': 'import io\nf = io.StringIO()\nx = f.getvalue()\n'}
    TWINS = {'io.StringIO.getvalue': 'import io\nf = io.StringIO()\nx = f.getvalue(0)\n'}

    names = ("io.StringIO.getvalue",)
    CONTRACT = Contract(
        qualname="io.StringIO.getvalue",
        raises=never_claims(tested=_ALL_TESTED),
        may_raise_other=True,  # ValueError: closed file
        returns=ReturnSpec("str", "the ENTIRE buffer's contents, regardless of the current stream position (unlike read())", tested=True),
        effects=Effects(deterministic=True, notes="does not move the stream position"),
        source=(f"{_DOCS}#io.StringIO.getvalue", "CPython Modules/_io/stringio.c::stringio_getvalue"),
        tested=True,
    )


@register_rule
class IoStringIOSeek(Rule):
    """The rule of ``io.StringIO.seek``."""

    SAMPLES = {'io.StringIO.seek': 'import io\nf = io.StringIO()\nf.seek(0)\n'}
    TWINS = {'io.StringIO.seek': 'import io\nf = io.StringIO()\nf.seek()\n'}

    names = ("io.StringIO.seek",)
    CONTRACT = Contract(
        qualname="io.StringIO.seek",
        raises=dunder_propagation_claims(
            "pos's or whence's __index__ (a USER method) -- any exception it raises propagates uncaught",
            native={"TypeError": "pos or whence has no __index__"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: closed file, negative pos with whence=0, or whence not in {0,1,2}
        returns=ReturnSpec("int", "the new absolute stream position", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims -- pos's/whence's __index__ can raise any of the 7 "
            "classes, not just TypeError."
        ),
        source=(f"{_DOCS}#io.IOBase.seek", "CPython Modules/_io/stringio.c::stringio_seek"),
        tested=True,
    )


@register_rule
class IoStringIOClose(Rule):
    """The rule of ``io.StringIO.close``."""

    SAMPLES = {'io.StringIO.close': 'import io\nf = io.StringIO()\nf.close()\n'}
    TWINS = {'io.StringIO.close': 'import io\nf = io.StringIO()\nf.close(0)\n'}

    names = ("io.StringIO.close",)
    CONTRACT = Contract(
        qualname="io.StringIO.close",
        raises=never_claims(tested=_ALL_TESTED),
        may_raise_other=False,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True, notes="idempotent -- closing an already-closed stream is fine; EVERY other method raises ValueError afterward"),
        source=(f"{_DOCS}#io.IOBase.close", "CPython Modules/_io/stringio.c::stringio_close"),
        tested=True,
    )


@register_rule
class IoBytesIOWrite(Rule):
    """The rule of ``io.BytesIO.write``."""

    SAMPLES = {'io.BytesIO.write': 'import io\nb = io.BytesIO()\nb.write(b"a")\n'}
    TWINS = {'io.BytesIO.write': 'import io\nb = io.BytesIO()\nb.write()\n'}

    names = ("io.BytesIO.write",)
    CONTRACT = Contract(
        qualname="io.BytesIO.write",
        raises=dunder_propagation_claims(
            "the argument's __buffer__ while acquiring its buffer; release may also execute __release_buffer__",
            native={"TypeError": "the argument does not support the buffer protocol (bytes/bytearray/memoryview do; str does NOT, unlike StringIO.write's str requirement)"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: closed file
        returns=ReturnSpec("int", "the number of bytes written", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True),
        notes="Accepts buffer-protocol objects, including Python __buffer__ implementations. Acquisition exceptions propagate; exceptions during __release_buffer__ are unraisable. StringIO.write accepts str instances, including subclasses, rather than arbitrary buffers.",
        source=(f"{_DOCS}#io.BytesIO.write", "CPython Modules/_io/bytesio.c::bytesio_write"),
        tested=True,
    )


__all__ = ["IoStringIOWrite", "IoStringIORead", "IoStringIOGetvalue", "IoStringIOSeek", "IoStringIOClose", "IoBytesIOWrite"]
