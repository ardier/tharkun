"""Rules for the write side of a file opened by ``open(..., "w" | "a" | "x")`` ([internal ref], [internal ref]).

A write-only ``open()`` returns an ``io.TextIOWrapper`` (a text mode) or an ``io.BufferedWriter`` (a
binary mode, default buffering); both are used through ``write``, ``writelines``, ``flush``, ``close``
and the context manager.  ``tell``/``seek`` are NOT contracted: a text file's ``tell`` is an opaque
cookie and ``seek`` takes one, and their results depend on the file system, so no exact claim is
worth recording until an engine needs one.  Reads are not part of this family (a handle opened for
output has none; ``r+``/``w+`` handles are not described here).

What the claims say, all observed on CPython 3.14.4 (``tests/contracts/test_fileio_.py``):

* ARGUMENT errors come first.  A wrong-typed argument to ``write`` is a ``TypeError`` even on a
  closed file (the argument is parsed before the file is checked); ``writelines`` checks the file
  first (a closed file is a ``ValueError`` even for a non-iterable).
* ``TextIOWrapper.write`` takes a ``str`` (a subclass is accepted and NONE of its methods is
  called); ``BufferedWriter.write`` takes any object with the buffer protocol (``bytes``,
  ``bytearray``, ``memoryview``, ``array.array``, and an object with a Python ``__buffer__``, which
  is a callback).
* ``ValueError`` (a closed file; ``UnicodeEncodeError`` for unencodable text) and ``OSError`` (the
  environment) are not property classes: ``may_raise_other`` is True and they are named in the
  conditions' notes, not in ``raises``.
* ``__exit__`` closes the file and returns ``None``: it never suppresses an exception of the body.

Migrated to rules ([internal ref], [internal ref]): ``OpenOutput`` (the shape contract of ``open``) and one class per method of the
two handle types (``OutputWrite``, ``OutputWritelines``, ``OutputFlush``, ``OutputClose``, ``OutputEnter``,
``OutputExit``, each covering the text and the binary handle); every ``Contract`` is exactly the one this module held
as plain data (``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/io.html"
_ALL = frozenset(ALL_PROPERTY_CLASSES)
_TYPE_ONLY = frozenset({"TypeError"})

_CLOSED_AND_OS = (
    "ValueError on a closed file; OSError from the environment (a full disk, a revoked descriptor); "
    "neither is a property class"
)
_ARITY = "called with other than the documented positional arguments (or with a keyword)"


def _write_side(kind: str) -> tuple[Contract, ...]:
    """The contracts of one handle kind: ``kind`` is ``TextIOWrapper`` or ``BufferedWriter``."""
    text = kind == "TextIOWrapper"
    q = f"io.{kind}"
    iobase = "CPython Modules/_io/iobase.c"
    out = []

    if text:
        write_raises = never_claims(
            native={"TypeError": f"the argument is not a str instance (a str subclass is accepted), or the call is {_ARITY}"},
            tested=_ALL,
        )
        write_returns = ReturnSpec("int", "len(text): the number of characters passed, whatever newline translation or encoding does", tested=True)
        write_callback = False
        write_notes = (
            "A str subclass is accepted and none of its methods (__str__, __len__, encode, find, "
            "__iter__, ...) is called. Encoding errors (UnicodeEncodeError, a ValueError) and a closed "
            "file (ValueError) end the call without a property class. The argument is checked BEFORE the "
            "file: a non-str argument is a TypeError even on a closed file."
        )
        write_other = True
    else:
        write_raises = dunder_propagation_claims(
            "the argument's __buffer__ while acquiring its buffer (a Python-level buffer provider)",
            native={"TypeError": f"the argument does not support the buffer protocol (str, int, list, ... do not), or the call is {_ARITY}"},
            tested_native=_TYPE_ONLY,
        )
        write_returns = ReturnSpec("int", "the number of bytes written (len of the buffer)", tested=True)
        write_callback = True
        write_notes = (
            "Accepts bytes, bytearray, memoryview, array.array (any typecode) and any object with a "
            "Python __buffer__ (a callback: acquisition exceptions propagate). A str is a TypeError, "
            "checked before the file: on a closed file a bytes-like argument is a ValueError, a str "
            "still a TypeError."
        )
        write_other = True
    out.append(Contract(
        qualname=f"{q}.write",
        raises=write_raises,
        may_raise_other=write_other,
        returns=write_returns,
        effects=Effects(mutates_receiver=True, io=True, deterministic=False, may_execute_callback=write_callback,
                        notes="appends to the file's buffer; the OS write happens at flush/close/when the buffer fills. " + _CLOSED_AND_OS),
        source=(f"{_DOCS}#io.TextIOBase.write" if text else f"{_DOCS}#io.BufferedIOBase.write",
                "CPython Modules/_io/textio.c::_io_TextIOWrapper_write_impl" if text
                else "CPython Modules/_io/bufferedio.c::_io__Buffered_write_impl"),
        notes=write_notes,
        tested=True,
    ))

    out.append(Contract(
        qualname=f"{q}.writelines",
        raises=dunder_propagation_claims(
            "the iterable's __iter__/__next__ (a Python-level iterator), or the write of an element",
            native={"TypeError": (
                "the file is open and the argument is not iterable, or an element is not "
                + ("a str" if text else "bytes-like")
                + f", or the call is {_ARITY}")},
            tested_native=_TYPE_ONLY,
        ),
        may_raise_other=True,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True, mutates_args=("lines",), io=True, deterministic=False, may_execute_callback=True,
                        notes="iterates its argument (advancing an iterator) and writes each element in order; the elements before a failing one stay written. "
                              "For an exact list or tuple of exact " + ("str" if text else "bytes") + " no user code runs. " + _CLOSED_AND_OS),
        source=(f"{_DOCS}#io.IOBase.writelines", f"{iobase}::_io__IOBase_writelines_impl"),
        notes=(
            "The file is checked FIRST: on a closed file even a non-iterable argument is a ValueError, "
            "not a TypeError. A str argument is iterated character by character (fine for a text file, "
            "a TypeError for a binary one: its elements are str)."
        ),
        tested=True,
    ))

    out.append(Contract(
        qualname=f"{q}.flush",
        raises=never_claims(native={"TypeError": f"the call is {_ARITY} (flush takes no argument)"}, tested=_ALL),
        may_raise_other=True,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True, io=True, deterministic=False, may_execute_callback=False,
                        notes="pushes the buffer to the OS. " + _CLOSED_AND_OS),
        source=(f"{_DOCS}#io.IOBase.flush", "CPython Modules/_io/textio.c::_io_TextIOWrapper_flush_impl" if text
                else "CPython Modules/_io/bufferedio.c::buffered_flush"),
        tested=True,
    ))

    out.append(Contract(
        qualname=f"{q}.close",
        raises=never_claims(native={"TypeError": f"the call is {_ARITY} (close takes no argument)"}, tested=_ALL),
        may_raise_other=True,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True, io=True, deterministic=False, may_execute_callback=False,
                        notes="flushes, then closes; idempotent (closing a closed file is fine); every later write/flush/enter "
                              "is a ValueError. A flush failure (OSError) is not a property class."),
        source=(f"{_DOCS}#io.IOBase.close", f"{iobase}::_io__IOBase_close_impl"),
        tested=True,
    ))

    out.append(Contract(
        qualname=f"{q}.__enter__",
        raises=never_claims(native={"TypeError": f"the call is {_ARITY} (__enter__ takes no argument)"}, tested=_ALL),
        may_raise_other=True,   # ValueError: I/O operation on closed file
        returns=ReturnSpec(f"io.{kind}", "the receiver itself", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=False, notes="no effect; the receiver comes back"),
        source=(f"{_DOCS}#io.IOBase.__enter__", f"{iobase}::iobase_enter"),
        tested=True,
    ))

    out.append(Contract(
        qualname=f"{q}.__exit__",
        raises=never_claims(tested=_ALL),
        may_raise_other=True,   # OSError from the flush inside close()
        returns=ReturnSpec("None", "always None: a falsy result never suppresses the body's exception", tested=True),
        effects=Effects(mutates_receiver=True, io=True, deterministic=False, may_execute_callback=False,
                        notes="calls close(), with or without a body exception, and returns None (the exception, if any, propagates unchanged)"),
        source=(f"{_DOCS}#io.IOBase.__exit__", f"{iobase}::iobase_exit"),
        tested=True,
    ))
    return tuple(out)


_OPEN_OUTPUT = Contract(
    qualname="open.output",
    raises=dunder_propagation_claims(
        "the path's __fspath__ (only when the path is not already str/bytes)",
        native={"TypeError": (
            "the path is not str, bytes or os.PathLike (or its __fspath__ raises TypeError or returns neither); "
            "or encoding, errors or newline is given and is not a str or None; or buffering is given and has no "
            "__index__; or the call has more than eight arguments, a repeated argument or an unknown keyword"
        )},
        tested_native=_TYPE_ONLY,
    ),
    may_raise_other=True,  # OSError (missing parent, a directory, permissions, file exists for 'x'), ValueError (NUL, bad mode/newline, binary + encoding), LookupError (unknown encoding)
    returns=ReturnSpec("io.TextIOWrapper | io.BufferedWriter",
                       "io.TextIOWrapper for a text mode (w, a, x, wt, ...), io.BufferedWriter for a binary one (wb, ab, xb, ...), with the default buffering",
                       tested=True),
    effects=Effects(io=True, deterministic=False, may_execute_callback=True,
                    notes="creates or truncates (w), appends to (a) or exclusively creates (x) a file; the content written never comes back to the program except through a read"),
    notes=(
        "A SHAPE contract (docs/contracts/FORMAT.md, 'Shape contracts'): the builtin `open` called with a CONSTANT "
        "write-only mode (solve/environment.py::OUTPUT_MODES). The union contract `open` (untested: the type it "
        "returns depends on the mode) is unchanged. An int path is a file DESCRIPTOR (a different call), and a "
        "mode with '+' reads: neither is this shape. buffering=0 returns io.FileIO (binary) or raises ValueError "
        "(text): not this shape either."
    ),
    source=(f"{_DOCS}#open", "CPython Modules/_io/_iomodule.c::_io_open_impl"),
    tested=True,
)

_BY_QUALNAME: dict[str, Contract] = {
    c.qualname: c for c in (_OPEN_OUTPUT, *_write_side("TextIOWrapper"), *_write_side("BufferedWriter"))}

#: the handle class a binary/text output mode produces (the receiver type of the method contracts)
OUTPUT_HANDLE_TYPES = {False: "io.TextIOWrapper", True: "io.BufferedWriter"}



#: every qualname of the family (the contracts the engines read through ``adapters/common.py``)
QUALNAMES = tuple(sorted(_BY_QUALNAME))


@register_rule
class OpenOutput(Rule):
    """The shape contract of the builtin ``open`` called with a constant write-only mode: no call shape is verified
    here (the engines classify the call through ``solve/environment.py::open_effect`` first)."""

    SAMPLES = {'open.output': 'f = open("/dev/null", "w")\n'}
    TWINS = {}

    names = ("open.output",)
    CONTRACT = _OPEN_OUTPUT


class _HandleMethod(Rule):
    """One method of the write-only handles: the same method of ``io.TextIOWrapper`` and ``io.BufferedWriter``.

    ``ARITY`` is the number of positional arguments after the receiver (``adapters/common.py::OUTPUT_HANDLE_METHOD_ARITY``
    is the engines' copy of it); the shape counts the receiver, no keywords."""

    ARITY: int = 0

    @classmethod
    def contract_for(cls, qualname: str) -> Contract:
        return _BY_QUALNAME[qualname]

    @classmethod
    def shape_for(cls, qualname: str) -> Shape:
        return Shape(min_args=cls.ARITY + 1, max_args=cls.ARITY + 1)


def _both(method: str) -> tuple[str, str]:
    return (f"io.TextIOWrapper.{method}", f"io.BufferedWriter.{method}")


@register_rule
class OutputWrite(_HandleMethod):

    SAMPLES = {'io.BufferedWriter.write': 'f = open("/dev/null", "wb")\nf.write(b"a")\n', 'io.TextIOWrapper.write': 'f = open("/dev/null", "w")\nf.write("a")\n'}
    TWINS = {'io.BufferedWriter.write': "f = open('/dev/null', 'wb')\nf.write()\n", 'io.TextIOWrapper.write': "f = open('/dev/null', 'w')\nf.write()\n"}
    names = _both("write")
    ARITY = 1


@register_rule
class OutputWritelines(_HandleMethod):

    SAMPLES = {'io.BufferedWriter.writelines': 'f = open("/dev/null", "wb")\nf.writelines([b"a", b"b"])\n', 'io.TextIOWrapper.writelines': 'f = open("/dev/null", "w")\nf.writelines(["a", "b"])\n'}
    TWINS = {'io.BufferedWriter.writelines': "f = open('/dev/null', 'wb')\nf.writelines()\n", 'io.TextIOWrapper.writelines': "f = open('/dev/null', 'w')\nf.writelines()\n"}
    names = _both("writelines")
    ARITY = 1


@register_rule
class OutputFlush(_HandleMethod):

    SAMPLES = {'io.BufferedWriter.flush': 'f = open("/dev/null", "wb")\nf.flush()\n', 'io.TextIOWrapper.flush': 'f = open("/dev/null", "w")\nf.flush()\n'}
    TWINS = {'io.BufferedWriter.flush': "f = open('/dev/null', 'wb')\nf.flush(0)\n", 'io.TextIOWrapper.flush': "f = open('/dev/null', 'w')\nf.flush(0)\n"}
    names = _both("flush")
    ARITY = 0


@register_rule
class OutputClose(_HandleMethod):

    SAMPLES = {'io.BufferedWriter.close': 'f = open("/dev/null", "wb")\nf.close()\n', 'io.TextIOWrapper.close': 'f = open("/dev/null", "w")\nf.close()\n'}
    TWINS = {'io.BufferedWriter.close': "f = open('/dev/null', 'wb')\nf.close(0)\n", 'io.TextIOWrapper.close': "f = open('/dev/null', 'w')\nf.close(0)\n"}
    names = _both("close")
    ARITY = 0


@register_rule
class OutputEnter(_HandleMethod):

    SAMPLES = {'io.BufferedWriter.__enter__': 'with open("/dev/null", "wb") as f:\n    pass\n', 'io.TextIOWrapper.__enter__': 'with open("/dev/null", "w") as f:\n    pass\n'}
    TWINS = {}
    names = _both("__enter__")
    ARITY = 0


@register_rule
class OutputExit(_HandleMethod):

    SAMPLES = {'io.BufferedWriter.__exit__': 'with open("/dev/null", "wb") as f:\n    pass\n', 'io.TextIOWrapper.__exit__': 'with open("/dev/null", "w") as f:\n    pass\n'}
    TWINS = {}
    names = _both("__exit__")
    ARITY = 3


__all__ = ["OpenOutput", "OutputWrite", "OutputWritelines", "OutputFlush", "OutputClose", "OutputEnter", "OutputExit",
           "OUTPUT_HANDLE_TYPES", "QUALNAMES"]
