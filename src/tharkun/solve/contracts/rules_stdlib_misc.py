"""Contracts for ``time`` clocks, ``base64.b64encode`` / ``b64decode`` and ``struct.pack`` / ``unpack`` / ``calcsize`` (standing batch 6, lane 1493).

A bytes-like argument is read through ``PyObject_GetBuffer``, which calls a Python class's ``__buffer__``, so such an argument can
propagate any class; ``struct`` converts most bad-argument failures into ``struct.error`` (an ``Exception``, not a property class) but an
integer format reads its argument through ``__index__`` unchanged.  Exact on CPython 3.14.4 (``tests/contracts/test_stdlib_misc.py``).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim, ReturnSpec

_ALL = frozenset(ALL_PROPERTY_CLASSES)


def _clock(name, result, note):
    qual = "time." + name
    contract = Contract(
        qualname=qual, raises=never_claims(native={"TypeError": "any argument or keyword is given"}, tested=_ALL),
        may_raise_other=False, returns=ReturnSpec(result, note, tested=True),
        effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False),
        source=(f"https://docs.python.org/3.14/library/time.html#time.{name}", "CPython Modules/timemodule.c"), tested=True)
    cls = type("Time" + "".join(p.capitalize() for p in name.split("_")), (Rule,), {
        "names": (qual,), "SAMPLES": {qual: f"import time\nx = time.{name}()\n"}, "TWINS": {qual: f"import time\nx = time.{name}(1)\n"},
        "CONTRACT": contract, "__module__": __name__})
    return register_rule(cls)


_CLOCKS = [_clock("perf_counter", "float", "seconds on a monotonic clock; only differences are meaningful"),
           _clock("monotonic", "float", "seconds on a monotonic clock that never goes backwards"),
           _clock("time", "float", "seconds since the epoch (the wall clock; can jump)"),
           _clock("time_ns", "int", "nanoseconds since the epoch"),
           _clock("thread_time", "float", "per-thread CPU time (sum of system and user time)"),
           _clock("thread_time_ns", "int", "per-thread CPU time in nanoseconds"),
           _clock("process_time", "float", "process-wide CPU time (sum of system and user time)"),
           _clock("process_time_ns", "int", "process-wide CPU time in nanoseconds"),
           _clock("monotonic_ns", "int", "nanoseconds on a monotonic clock that never goes backwards"),
           _clock("perf_counter_ns", "int", "nanoseconds on a monotonic high-resolution clock; only differences are meaningful")]


@register_rule
class TypingGetTypeHints(Rule):
    """Open-world type-hint resolution, including evaluated forward references."""

    names = ('typing.get_type_hints',)
    SAMPLES = {'typing.get_type_hints': 'import typing\ndef f(x: int): pass\nx = typing.get_type_hints(f)\n'}
    TWINS = {'typing.get_type_hints': 'import typing\nx = typing.get_type_hints()\n'}
    CONTRACT = Contract(
        qualname='typing.get_type_hints',
        raises=tuple(RaiseClaim(exc,
            'wrong argument binding, unsupported input, or an uncaught TypeError from introspection or evaluation'
            if exc == 'TypeError' else
            f'an uncaught {exc} from introspection or type-hint evaluation', tested=True)
            for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('dict', 'resolved annotations, including inherited class annotations; __no_type_check__ yields an empty dict', tested=True),
        effects=Effects(mutates_receiver=True, io=True, clock_or_env=True, deterministic=False, may_execute_callback=True,
                        notes='May populate lazy annotations, inspect namespaces and execute arbitrary code while resolving annotations.'),
        source=('https://docs.python.org/3.14/library/typing.html#typing.get_type_hints',
                'CPython v3.14.4 Lib/typing.py::get_type_hints'), tested=True,
    )


@register_rule
class DataclassesIsDataclassInstance(Rule):
    """The private instance predicate checks the instance type's fields marker."""

    names = ('dataclasses._is_dataclass_instance',)
    SAMPLES = {'dataclasses._is_dataclass_instance': 'import dataclasses\nx = dataclasses._is_dataclass_instance(1)\n'}
    TWINS = {'dataclasses._is_dataclass_instance': 'import dataclasses\nx = dataclasses._is_dataclass_instance()\n'}
    CONTRACT = Contract(
        qualname='dataclasses._is_dataclass_instance',
        raises=dunder_propagation_claims('metaclass attribute lookup',
            native={'AttributeError': 'never', 'TypeError': 'wrong argument binding or an uncaught TypeError from metaclass lookup'},
            tested_native=frozenset({'AttributeError', 'TypeError'})),
        may_raise_other=True,
        returns=ReturnSpec('bool', 'whether type(obj) has __dataclass_fields__; AttributeError from lookup is suppressed', tested=True),
        effects=Effects(may_execute_callback=True, notes='Metaclass lookup may execute arbitrary hooks; the object itself is not inspected.'),
        source=('CPython v3.14.4 Lib/dataclasses.py::_is_dataclass_instance',), tested=True,
    )


@register_rule
class JsonJSONEncoderDefault(Rule):
    """The base encoder default always fails after formatting the object's class name."""

    names = ('json.JSONEncoder.default',)
    SAMPLES = {'json.JSONEncoder.default': 'import json\ntry: json.JSONEncoder.default(None, object())\nexcept TypeError: pass\n'}
    TWINS = {'json.JSONEncoder.default': 'import json\njson.JSONEncoder.default()\n'}
    CONTRACT = Contract(
        qualname='json.JSONEncoder.default',
        raises=tuple(RaiseClaim(exc,
            'wrong argument binding, successful class-name formatting, or an uncaught TypeError from class-name lookup or formatting'
            if exc == 'TypeError' else
            f'an uncaught {exc} from class-name lookup or formatting', tested=True)
            for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('Never', 'the base implementation does not return', tested=True),
        effects=Effects(may_execute_callback=True, notes='Reads o.__class__.__name__ and formats the result; these hooks may execute arbitrary code.'),
        source=('https://docs.python.org/3.14/library/json.html#json.JSONEncoder.default',
                'CPython v3.14.4 Lib/json/encoder.py::JSONEncoder.default'), tested=True,
    )


@register_rule
class InspectGetAnnotations(Rule):
    """The inspect alias of annotationlib's open-world annotation evaluator."""

    names = ('inspect.get_annotations',)
    SAMPLES = {'inspect.get_annotations': 'import inspect\ndef f(x): pass\nx = inspect.get_annotations(f)\n'}
    TWINS = {'inspect.get_annotations': 'import inspect\nx = inspect.get_annotations()\n'}
    CONTRACT = Contract(
        qualname='inspect.get_annotations',
        raises=tuple(RaiseClaim(exc,
            'wrong argument binding, an object without annotations, or an uncaught TypeError from lookup or evaluation'
            if exc == 'TypeError' else
            f'an uncaught {exc} from annotation lookup, evaluation, format comparison or dictionary processing', tested=True)
            for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('dict', 'a fresh annotations dictionary; class annotations exclude inherited entries', tested=True),
        effects=Effects(mutates_receiver=True, mutates_args=('globals',), io=True, clock_or_env=True,
                        deterministic=False, may_execute_callback=True,
                        notes='May populate lazy annotations even with eval_str=False; string evaluation may insert __builtins__ into globals. Formats, lookup and evaluation can run arbitrary code.'),
        source=('https://docs.python.org/3.14/library/inspect.html#inspect.get_annotations',
                'https://docs.python.org/3.14/library/annotationlib.html#annotationlib.get_annotations',
                'https://github.com/python/cpython/blob/v3.14.4/Lib/annotationlib.py'), tested=True,
    )


@register_rule
class InspectGetattrStatic(Rule):
    """Static attribute lookup avoids descriptors, but not every user hook."""

    names = ('inspect.getattr_static',)
    SAMPLES = {'inspect.getattr_static': 'import inspect\nx = inspect.getattr_static(int, "real")\n'}
    TWINS = {'inspect.getattr_static': 'import inspect\nx = inspect.getattr_static()\n'}
    CONTRACT = Contract(
        qualname='inspect.getattr_static',
        raises=tuple(RaiseClaim(exc,
            'wrong argument binding, an unhashable attribute key, or an uncaught TypeError from static lookup'
            if exc == 'TypeError' else
            'no statically visible attribute and no default, or an uncaught AttributeError from static lookup'
            if exc == 'AttributeError' else
            f'an uncaught {exc} from attribute-key hashing/equality or metaclass dictionary lookup', tested=True)
            for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('object', 'the statically selected attribute or supplied default, preserving identity; descriptors are not invoked', tested=True),
        effects=Effects(may_execute_callback=True,
                        notes='Bypasses the target attribute descriptor and instance __getattr__/__getattribute__; key hashing/equality and metaclass dictionary access may still run arbitrary hooks.'),
        source=('https://docs.python.org/3.14/library/inspect.html#inspect.getattr_static',
                'https://github.com/python/cpython/blob/v3.14.4/Lib/inspect.py'), tested=True,
    )


@register_rule
class InspectGetmembersStatic(Rule):
    """Static member values with open dir, predicate and ordering callbacks."""

    names = ('inspect.getmembers_static',)
    SAMPLES = {'inspect.getmembers_static': 'import inspect\nx = inspect.getmembers_static(1)\n'}
    TWINS = {'inspect.getmembers_static': 'import inspect\nx = inspect.getmembers_static()\n'}
    CONTRACT = Contract(
        qualname='inspect.getmembers_static',
        raises=tuple(RaiseClaim(exc,
            'wrong argument binding, invalid dir output or predicate, or an uncaught TypeError from member enumeration'
            if exc == 'TypeError' else
            f'an uncaught {exc} from dir, metaclass/key lookup, predicate truth/call/result or name ordering', tested=True)
            for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('list[tuple[object, object]]', 'a fresh list of statically found (name, value) pairs sorted by name and filtered by predicate when truthy; custom dir output need not contain strings', tested=True),
        effects=Effects(may_execute_callback=True,
                        notes='Does not invoke member descriptors, but dir(), class metadata, name operations and predicate truth/call/result can run arbitrary hooks. Missing static attributes are omitted.'),
        source=('https://docs.python.org/3.14/library/inspect.html#inspect.getmembers_static',
                'https://github.com/python/cpython/blob/v3.14.4/Lib/inspect.py'), tested=True,
    )


@register_rule
class InspectSignature(Rule):
    """Open-world signature introspection, including optional annotation evaluation."""

    names = ('inspect.signature',)
    SAMPLES = {'inspect.signature': 'import inspect\nx = inspect.signature(len)\n'}
    TWINS = {'inspect.signature': 'import inspect\nx = inspect.signature()\n'}
    CONTRACT = Contract(
        qualname='inspect.signature',
        raises=tuple(RaiseClaim(exc,
            'wrong binding, a non-callable, an invalid __signature__, or an uncaught TypeError from introspection or annotation evaluation'
            if exc == 'TypeError' else
            'a callable has no supported signature, or an uncaught ValueError from introspection or annotation evaluation'
            if exc == 'ValueError' else
            f'an uncaught {exc} from introspection or annotation evaluation', tested=True)
            for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('inspect.Signature', 'the signature selected by introspection, optionally following __wrapped__', tested=True),
        effects=Effects(mutates_receiver=True, mutates_args=('globals',), io=True, clock_or_env=True, deterministic=False, may_execute_callback=True,
                        notes='Introspection can populate lazy annotations, evaluation can insert __builtins__ into globals, and hooks may run; eval_str=True evaluates string annotations in supplied or inferred namespaces with arbitrary effects.'),
        source=('https://docs.python.org/3.14/library/inspect.html#inspect.signature',
                'CPython v3.14.4 Lib/inspect.py::signature, Signature.from_callable, _signature_from_callable'),
        tested=True,
    )


@register_rule
class Base64B64encode(Rule):
    """The rule of ``base64.b64encode(s, altchars=None)``."""

    SAMPLES = {"base64.b64encode": 'import base64\nx = base64.b64encode(b"ab")\n'}
    TWINS = {"base64.b64encode": "import base64\nx = base64.b64encode()\n"}

    names = ("base64.b64encode",)
    CONTRACT = Contract(
        qualname="base64.b64encode",
        raises=dunder_propagation_claims(
            "s's __buffer__, or altchars's __len__ / __buffer__ (user methods)",
            native={
                "TypeError": "s or altchars is not bytes-like (a str included); wrong argument count or an unknown keyword",
                "AssertionError": "altchars is given and is not exactly 2 bytes long (an `assert`: absent under python -O)",
            },
            tested_native=frozenset({"TypeError", "AssertionError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("bytes", "the standard base64 text with '=' padding, or with '+' and '/' replaced by altchars", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/base64.html#base64.b64encode", "CPython Lib/base64.py::b64encode"),
        tested=True,
    )


@register_rule
class Base64B64decode(Rule):
    """The rule of ``base64.b64decode(s, altchars=None, validate=False)``."""

    SAMPLES = {"base64.b64decode": 'import base64\nx = base64.b64decode("YQ==")\n'}
    TWINS = {"base64.b64decode": "import base64\nx = base64.b64decode()\n"}

    names = ("base64.b64decode",)
    CONTRACT = Contract(
        qualname="base64.b64decode",
        raises=dunder_propagation_claims(
            "s's or altchars's __buffer__ (a user method)",
            native={
                "TypeError": "s is neither a bytes-like object nor an ASCII str, or altchars is not bytes-like; wrong argument count or an unknown keyword",
                "AssertionError": "altchars is given and is not exactly 2 bytes long (an `assert`: absent under python -O)",
            },
            tested_native=frozenset({"TypeError", "AssertionError"}),
        ),
        may_raise_other=True,  # binascii.Error (a ValueError): bad padding or, with validate, a non-alphabet byte; ValueError: a non-ASCII str
        returns=ReturnSpec("bytes", "the decoded bytes; without validate, bytes outside the alphabet are skipped", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/base64.html#base64.b64decode", "CPython Lib/base64.py::b64decode"),
        tested=True,
    )


@register_rule
class Base64UrlSafeB64encode(Rule):
    """The rule of ``base64.urlsafe_b64encode(s)``; delegates to b64encode, then replaces + -> -, / -> _."""

    SAMPLES = {"base64.urlsafe_b64encode": 'import base64\nx = base64.urlsafe_b64encode(b"ab")\n'}
    TWINS = {"base64.urlsafe_b64encode": "import base64\nx = base64.urlsafe_b64encode()\n"}

    names = ("base64.urlsafe_b64encode",)
    CONTRACT = Contract(
        qualname="base64.urlsafe_b64encode",
        raises=dunder_propagation_claims(
            "s's __buffer__ (a user method; b64encode calls it, then replace operates on the result bytes)",
            native={
                "TypeError": "s is not bytes-like (a str included); wrong argument count or an unknown keyword",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("bytes", "same as b64encode but with URL-safe alphabet (+ -> -, / -> _), no '=' padding change", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/base64.html#base64.urlsafe_b64encode", "CPython Lib/base64.py::urlsafe_b64encode"),
        tested=True,
    )


@register_rule
class Base64UrlSafeB64decode(Rule):
    """The rule of ``base64.urlsafe_b64decode(s)``; replaces - -> +, _ -> /, then delegates to b64decode."""

    SAMPLES = {"base64.urlsafe_b64decode": 'import base64\nx = base64.urlsafe_b64decode(b"YQ==")\n'}
    TWINS = {"base64.urlsafe_b64decode": "import base64\nx = base64.urlsafe_b64decode()\n"}

    names = ("base64.urlsafe_b64decode",)
    CONTRACT = Contract(
        qualname="base64.urlsafe_b64decode",
        raises=dunder_propagation_claims(
            "s's __buffer__ (a user method; replace and b64decode both read through the buffer protocol)",
            native={
                "TypeError": "s is neither a bytes-like object nor an ASCII str; wrong argument count or an unknown keyword",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # binascii.Error (a ValueError): bad padding; non-alphabet bytes are skipped (validate=False)
        returns=ReturnSpec("bytes", "the decoded bytes; unlike b64decode, does NOT accept altchars or validate", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/base64.html#base64.urlsafe_b64decode", "CPython Lib/base64.py::urlsafe_b64decode"),
        tested=True,
    )


@register_rule
class Base32B32encode(Rule):
    """The rule of ``base64.b32encode(s)``."""

    SAMPLES = {"base64.b32encode": 'import base64\nx = base64.b32encode(b"ab")\n'}
    TWINS = {"base64.b32encode": "import base64\nx = base64.b32encode()\n"}

    names = ("base64.b32encode",)
    CONTRACT = Contract(
        qualname="base64.b32encode",
        raises=dunder_propagation_claims(
            "s's __buffer__ (a user method)",
            native={
                "TypeError": "s is not bytes-like (a str included); wrong argument count or an unknown keyword",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # binascii.Error (a ValueError): non-alphabet byte in the result (should not happen)
        returns=ReturnSpec("bytes", "the base32-encoded bytes with uppercase letters", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/base64.html#base64.b32encode", "CPython Lib/base64.py::b32encode"),
        tested=True,
    )


@register_rule
class Base32B32decode(Rule):
    """The rule of ``base64.b32decode(s, casefold=False, map01=None)``."""

    SAMPLES = {"base64.b32decode": 'import base64\nx = base64.b32decode("NBSWY===")\n'}
    TWINS = {"base64.b32decode": "import base64\nx = base64.b32decode()\n"}

    names = ("base64.b32decode",)
    CONTRACT = Contract(
        qualname="base64.b32decode",
        raises=dunder_propagation_claims(
            "s's __buffer__ (a user method); map01's __len__ / __getitem__ (a user method)",
            native={
                "TypeError": "s is neither a bytes-like object nor an ASCII str; wrong argument count or an unknown keyword",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # binascii.Error (a ValueError): bad padding or non-alphabet character
        returns=ReturnSpec("bytes", "the decoded bytes; casefold=True accepts lower- and uppercase; map01 maps 0/1 chars", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/base64.html#base64.b32decode", "CPython Lib/base64.py::b32decode"),
        tested=True,
    )


@register_rule
class StructPack(Rule):
    """The rule of ``struct.pack(fmt, *values)``."""

    SAMPLES = {"struct.pack": 'import struct\nx = struct.pack("i", 1)\n'}
    TWINS = {"struct.pack": "import struct\nx = struct.pack()\n"}

    names = ("struct.pack",)
    CONTRACT = Contract(
        qualname="struct.pack",
        raises=dunder_propagation_claims(
            "an integer-format value's __index__ or a ? format value's __bool__ (user code); a float format converts a hook failure into struct.error",
            native={"TypeError": "fmt is neither str nor bytes; no arguments; or a keyword"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # struct.error: bad format character, wrong value count, value out of range or of the wrong kind
        returns=ReturnSpec("bytes", "the packed values; len == struct.calcsize(fmt)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/struct.html#struct.pack", "CPython Modules/_struct.c"),
        tested=True,
    )


@register_rule
class StructUnpack(Rule):
    """The rule of ``struct.unpack(fmt, buffer)``."""

    SAMPLES = {"struct.unpack": 'import struct\nx = struct.unpack("i", b"\\x00\\x00\\x00\\x00")\n'}
    TWINS = {"struct.unpack": "import struct\nx = struct.unpack()\n"}

    names = ("struct.unpack",)
    CONTRACT = Contract(
        qualname="struct.unpack",
        raises=dunder_propagation_claims(
            "the buffer's __buffer__ (a user method)",
            native={"TypeError": "fmt is neither str nor bytes; the buffer is not bytes-like (a str included); wrong argument count or a keyword"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # struct.error: bad format character, or a buffer whose size is not calcsize(fmt)
        returns=ReturnSpec("tuple", "one element per value of fmt", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/struct.html#struct.unpack", "CPython Modules/_struct.c"),
        tested=True,
    )


@register_rule
class StructCalcsize(Rule):
    """The rule of ``struct.calcsize(fmt)``."""

    SAMPLES = {"struct.calcsize": 'import struct\nx = struct.calcsize("i")\n'}
    TWINS = {"struct.calcsize": "import struct\nx = struct.calcsize()\n"}

    names = ("struct.calcsize",)
    CONTRACT = Contract(
        qualname="struct.calcsize",
        raises=never_claims(native={"TypeError": "fmt is neither str nor bytes; wrong argument count or a keyword"}, tested=_ALL),
        may_raise_other=True,  # struct.error: bad format character
        returns=ReturnSpec("int", "the packed size in bytes", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=False),
        source=("https://docs.python.org/3.14/library/struct.html#struct.calcsize", "CPython Modules/_struct.c"),
        tested=True,
    )


_HOOK_TE = "wrong argument count or an unknown keyword"


def _contract_rule(qual, cls_name, sample, twin, contract):
    cls = type(cls_name, (Rule,), {"names": (qual,), "SAMPLES": {qual: sample}, "TWINS": {qual: twin}, "CONTRACT": contract, "__module__": __name__})
    return register_rule(cls)


_EXTRA = [
    _contract_rule(
        "warnings.warn", "WarningsWarn", 'import warnings\nwarnings.simplefilter("ignore")\nx = warnings.warn("m")\n',
        "import warnings\nx = warnings.warn()\n",
        Contract(
            qualname="warnings.warn",
            raises=dunder_propagation_claims(
                "message's __str__ (a str subclass override, or str() of a non-str message), category's __init__ (a Warning subclass constructor), or showwarning (the module global, user-replaceable)",
                native={
                    "TypeError": "category is neither None nor a Warning subclass; stacklevel / source are not ints (stacklevel) or not a Warning-source (source); wrong argument count or an unknown keyword",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # under an 'error' warning filter the warning itself is raised (an arbitrary Warning subclass); showwarning can raise anything; RuntimeError when sys.stderr is None
            returns=ReturnSpec("None", "registers and (depending on the filter) prints the warning to stderr", tested=True),
            effects=Effects(io=True, deterministic=False, may_execute_callback=True, notes="writes to stderr when shown; reads the process-wide warning filter table"),
            source=("https://docs.python.org/3.14/library/warnings.html#warnings.warn", "CPython Lib/warnings.py::warn"),
            tested=True,
        )),
    _contract_rule(
        "dataclasses.is_dataclass", "DataclassesIsDataclass", "import dataclasses\nx = dataclasses.is_dataclass(1)\n", "import dataclasses\nx = dataclasses.is_dataclass()\n",
        Contract(
            qualname="dataclasses.is_dataclass",
            raises=dunder_propagation_claims(
                "the class's metaclass __getattribute__ / __getattr__ on __dataclass_fields__ (user code; an AttributeError there is swallowed by hasattr)",
                native={"AttributeError": "never (swallowed by the hasattr inside)", "TypeError": _HOOK_TE},
                tested_native=frozenset({"AttributeError", "TypeError"})),
            may_raise_other=True,
            returns=ReturnSpec("bool", "True for a dataclass class or an instance of one", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=True),
            source=("https://docs.python.org/3.14/library/dataclasses.html#dataclasses.is_dataclass", "CPython Lib/dataclasses.py::is_dataclass"), tested=True)),
    _contract_rule(
        "dataclasses.fields", "DataclassesFields", "import dataclasses\nP = dataclasses.make_dataclass(\"P\", [(\"x\", int)])\nx = dataclasses.fields(P)\n",
        "import dataclasses\nx = dataclasses.fields()\n",
        Contract(
            qualname="dataclasses.fields",
            raises=dunder_propagation_claims(
                "the object's __getattribute__ / __getattr__ (or a metaclass's) on __dataclass_fields__ (user code)",
                native={"AttributeError": "never (a missing __dataclass_fields__ becomes TypeError)",
                        "TypeError": "the argument is neither a dataclass class nor an instance of one; " + _HOOK_TE},
                tested_native=frozenset({"AttributeError", "TypeError"})),
            may_raise_other=True,
            returns=ReturnSpec("tuple", "the Field objects of the dataclass, in definition order (no ClassVar / InitVar pseudo-fields)", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=True),
            source=("https://docs.python.org/3.14/library/dataclasses.html#dataclasses.fields", "CPython Lib/dataclasses.py::fields"), tested=True)),
    _contract_rule(
        "inspect.cleandoc", "InspectCleandoc", 'import inspect\nx = inspect.cleandoc("  a\\n  b")\n', "import inspect\nx = inspect.cleandoc()\n",
        Contract(
            qualname="inspect.cleandoc",
            raises=dunder_propagation_claims(
                "a str subclass's expandtabs / split / lstrip overrides (user code)",
                native={"AttributeError": "doc has no expandtabs (None, an int)", "TypeError": "doc is a bytes (its split gets a str); " + _HOOK_TE},
                tested_native=frozenset({"AttributeError", "TypeError"})),
            may_raise_other=True,
            returns=ReturnSpec("str", "the docstring without its common leading whitespace and leading / trailing blank lines", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=True),
            source=("https://docs.python.org/3.14/library/inspect.html#inspect.cleandoc", "CPython Lib/inspect.py::cleandoc"), tested=True)),
    _contract_rule(
        "os.getenv", "OsGetenv", 'import os\nx = os.getenv("NOPE_X")\n', "import os\nx = os.getenv()\n",
        Contract(
            qualname="os.getenv",
            raises=dunder_propagation_claims(
                "a str subclass key's encode override (user code); a KeyError it raises is swallowed (Mapping.get)",
                native={"TypeError": "key is not a str (bytes included); " + _HOOK_TE, "KeyError": "never (Mapping.get returns the default for any KeyError, a hook's included)"},
                tested_native=frozenset({"TypeError", "KeyError"})),
            may_raise_other=True,
            returns=ReturnSpec("str | the default", "the variable's value, or default (None) when it is unset; a missing key never raises KeyError", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="reads the process environment"),
            source=("https://docs.python.org/3.14/library/os.html#os.getenv", "CPython Lib/os.py::getenv, _Environ.get"), tested=True)),
    _contract_rule(
        "os.getpid", "OsGetpid", "import os\nx = os.getpid()\n", "import os\nx = os.getpid(1)\n",
        Contract(
            qualname="os.getpid", raises=never_claims(native={"TypeError": "any argument or keyword"}, tested=_ALL), may_raise_other=False,
            returns=ReturnSpec("int", "the process id", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False),
            source=("https://docs.python.org/3.14/library/os.html#os.getpid", "CPython Modules/posixmodule.c"), tested=True)),
    _contract_rule(
        "threading.get_ident", "ThreadingGetIdent", "import threading\nx = threading.get_ident()\n", "import threading\nx = threading.get_ident(1)\n",
        Contract(
            qualname="threading.get_ident", raises=never_claims(native={"TypeError": "any argument or keyword"}, tested=_ALL), may_raise_other=False,
            returns=ReturnSpec("int", "the current thread's identifier", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False),
            source=("https://docs.python.org/3.14/library/threading.html#threading.get_ident", "CPython Modules/_threadmodule.c"), tested=True)),
    _contract_rule(
        "time.sleep", "TimeSleep", "import time\nx = time.sleep(0.001)\n", "import time\nx = time.sleep()\n",
        Contract(
            qualname="time.sleep",
            raises=dunder_propagation_claims(
                "secs's __index__ (a plain non-int/float object is converted through it; int/float subclasses pass by value)",
                native={"TypeError": "secs is neither an int, a float, nor an object with __index__ (a str / None / complex); wrong argument count or a keyword argument"},
                tested_native=frozenset({"TypeError"})),
            may_raise_other=True,  # OverflowError: too large for the C PyTime_t; ValueError: NaN or a negative value
            returns=ReturnSpec("None", "blocks for the given seconds", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True),
            source=("https://docs.python.org/3.14/library/time.html#time.sleep", "CPython Modules/timemodule.c::time_sleep"), tested=True)),
    _contract_rule(
        "datetime.datetime.now", "DatetimeNow", "import datetime\nx = datetime.datetime.now()\n", "import datetime\nx = datetime.datetime.now(1)\n",
        Contract(
            qualname="datetime.datetime.now",
            raises=dunder_propagation_claims(
                "tz's fromutc (user code; only called when tz is given)",
                native={"TypeError": "tz is neither None nor a tzinfo subclass; wrong argument count or an unknown keyword"},
                tested_native=frozenset({"TypeError"})),
            may_raise_other=True,  # whatever a user tz's fromutc raises; nothing native beyond TypeError
            returns=ReturnSpec("datetime.datetime", "the current local wall-clock time (naive) or, with tz, that timezone's wall clock", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="reads the wall clock"),
            source=("https://docs.python.org/3.14/library/datetime.html#datetime.datetime.now", "CPython Modules/_datetimemodule.c::datetime_now"), tested=True)),
    _contract_rule(
        "datetime.datetime.fromtimestamp", "DatetimeFromTimestamp", "import datetime\nx = datetime.datetime.fromtimestamp(0)\n",
        "import datetime\nx = datetime.datetime.fromtimestamp()\n",
        Contract(
            qualname="datetime.datetime.fromtimestamp",
            raises=dunder_propagation_claims(
                "timestamp's __index__ (a plain non-int/float object) or tz's fromutc (user code; both only when the argument is present)",
                native={"TypeError": "timestamp is neither an int, a float, nor an object with __index__; tz is neither None nor a tzinfo subclass; wrong argument count or an unknown keyword"},
                tested_native=frozenset({"TypeError"})),
            may_raise_other=True,  # OverflowError: out of platform range; ValueError: NaN or an impossible platform time
            returns=ReturnSpec("datetime.datetime", "the local time for the timestamp (naive) or, with tz, that timezone's time", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="reads the local timezone when tz is None"),
            source=("https://docs.python.org/3.14/library/datetime.html#datetime.datetime.fromtimestamp", "CPython Modules/_datetimemodule.c::datetime_fromtimestamp"), tested=True)),
    _contract_rule(
        "datetime.datetime.fromisoformat", "DatetimeFromIsoformat", "import datetime\nx = datetime.datetime.fromisoformat('2024-01-01')\n",
        "import datetime\nx = datetime.datetime.fromisoformat()\n",
        Contract(
            qualname="datetime.datetime.fromisoformat",
            raises=never_claims(native={"TypeError": "the argument is not a str (bytes included); wrong argument count or a keyword argument"}, tested=_ALL),
            may_raise_other=True,  # ValueError: a malformed or out-of-range ISO string (year 0, month 13, an impossible calendar date)
            returns=ReturnSpec("datetime.datetime", "the naive or timezone-aware datetime the ISO 8601 string names", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=False),
            source=("https://docs.python.org/3.14/library/datetime.html#datetime.datetime.fromisoformat", "CPython Modules/_datetimemodule.c::datetime_fromisoformat"), tested=True)),
    _contract_rule(
        "dataclasses.field", "DataclassesField1", "import dataclasses\nx = dataclasses.field()\n", "import dataclasses\nx = dataclasses.field(1)\n",
        Contract(
            qualname="dataclasses.field",
            raises=never_claims(native={"TypeError": "metadata is not a mapping (the mappingproxy conversion); wrong argument count or an unknown keyword"},
                                tested=_ALL),
            may_raise_other=True,  # ValueError: both default and default_factory are given
            returns=ReturnSpec("dataclasses.Field", "a Field spec; default_factory is stored, never called here", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=False),
            source=("https://docs.python.org/3.14/library/dataclasses.html#dataclasses.field", "CPython Lib/dataclasses.py::field"), tested=True)),
    _contract_rule(
        "ast.literal_eval", "AstLiteralEval", "import ast\nx = ast.literal_eval(\"123\")\n",
        "import ast\nx = ast.literal_eval()\n",
        Contract(
            qualname="ast.literal_eval",
            raises=never_claims(native={"TypeError": "the argument is neither a string, bytes, nor a code object; wrong argument count or a keyword"},
                                tested=_ALL),
            may_raise_other=True,  # ValueError: malformed or non-literal source, OverflowError: too large a literal
            returns=ReturnSpec("object", "the Python literal the source string evaluates to (int, float, complex, str, bytes, bool, None, list, tuple, dict, set)", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=False),
            source=("https://docs.python.org/3.14/library/ast.html#ast.literal_eval", "CPython Lib/ast.py::literal_eval"), tested=True)),
    _contract_rule(
        "ast.walk", "AstWalk", "import ast\ntree = ast.parse(\"x = 1\")\nx = ast.walk(tree)\n",
        "import ast\nx = ast.walk()\n",
        Contract(
            qualname="ast.walk",
            raises=never_claims(native={
                "AttributeError": "the argument has no _fields (is not an AST node object)",
                "TypeError": "wrong argument count or a keyword",
            }, tested=_ALL),
            may_raise_other=False,
            returns=ReturnSpec("generator", "yields every node in the tree (the root first), depth-first", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=False),
            source=("https://docs.python.org/3.14/library/ast.html#ast.walk", "CPython Lib/ast.py::walk"), tested=True)),
    _contract_rule(
        "os.getcwd", "OsGetcwd", "import os\nx = os.getcwd()\n",
        "import os\nx = os.getcwd(1)\n",
        Contract(
            qualname="os.getcwd",
            raises=never_claims(native={"TypeError": "any argument or keyword"}, tested=_ALL),
            may_raise_other=True,  # FileNotFoundError: CWD was deleted; PermissionError: no search permission; OSError: other OS errors
            returns=ReturnSpec("str", "the current working directory as a str (decoded from the OS encoding)", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False, notes="reads the process CWD from the kernel"),
            source=("https://docs.python.org/3.14/library/os.html#os.getcwd", "CPython Modules/posixmodule.c"), tested=True)),
    _contract_rule(
        "platform.system", "PlatformSystem", "import platform\nx = platform.system()\n",
        "import platform\nx = platform.system(1)\n",
        Contract(
            qualname="platform.system",
            raises=never_claims(native={"TypeError": "any argument or keyword"}, tested=_ALL),
            may_raise_other=False,
            returns=ReturnSpec("str", "the operating system name (e.g. 'Linux', 'Darwin', 'Windows'), cached after the first call", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False, notes="reads os.uname() on first call; cached thereafter"),
            source=("https://docs.python.org/3.14/library/platform.html#platform.system", "CPython Lib/platform.py::system"), tested=True)),
    _contract_rule(
        "os.listdir", "OsListdir", "import os\nx = os.listdir()\n", "import os\nx = os.listdir(1, 2)\n",
        Contract(
            qualname="os.listdir",
            raises=never_claims(native={
                "TypeError": "too many arguments, or a keyword other than path",
            }, tested=_ALL),
            may_raise_other=True,  # FileNotFoundError: missing path; NotADirectoryError: not a dir; PermissionError: no search permission; OSError: other OS errors
            returns=ReturnSpec("list", "the names of entries in the directory (sorted by the OS/LibC); None or omitted uses the current directory", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False, notes="reads a directory from the filesystem"),
            source=("https://docs.python.org/3.14/library/os.html#os.listdir", "CPython Modules/posixmodule.c::os_listdir"), tested=True)),
    _contract_rule(
        "os.chdir", "OsChdir", "import os\nold = os.getcwd()\nx = os.chdir(\"/tmp\")\nos.chdir(old)\n",
        "import os\nx = os.chdir()\n",
        Contract(
            qualname="os.chdir",
            raises=never_claims(native={
                "TypeError": "path is not a string, bytes, os.PathLike, or file descriptor; wrong argument count or a keyword",
            }, tested=_ALL),
            may_raise_other=True,  # FileNotFoundError: missing path; NotADirectoryError: not a dir; PermissionError: no search permission; OSError: other OS errors
            returns=ReturnSpec("None", "changes the process CWD; returns None", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False, notes="modifies the process CWD"),
            source=("https://docs.python.org/3.14/library/os.html#os.chdir", "CPython Modules/posixmodule.c::os_chdir"), tested=True)),
    _contract_rule(
        "os.stat", "OsStat", "import os\nx = os.stat(\".\")\n", "import os\nx = os.stat()\n",
        Contract(
            qualname="os.stat",
            raises=dunder_propagation_claims("path/command __fspath__, index and string protocol callbacks", native={
                "TypeError": "wrong argument binding or path/fd/dir_fd/follow_symlinks kinds, or protocol callback propagation",
            }, tested_native=frozenset({"TypeError"})),
            may_raise_other=True,  # FileNotFoundError: missing path; PermissionError: no permission; OSError: other OS errors (incl. OSError for fd exhaustion)
            returns=ReturnSpec("os.stat_result", "the file's stat result (mode, size, times, etc.)", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="reads file metadata from the filesystem"),
            source=("https://docs.python.org/3.14/library/os.html#os.stat", "CPython Modules/posixmodule.c::os_stat_impl"), tested=True)),
    _contract_rule(
        "os.makedirs", "OsMakedirs", "import os\nx = os.makedirs(\"/tmp/1493_test_x\", exist_ok=True)\nos.rmdir(\"/tmp/1493_test_x\")\n",
        "import os\nx = os.makedirs()\n",
        Contract(
            qualname="os.makedirs",
            raises=never_claims(native={
                "TypeError": "wrong argument count, or an unknown keyword (mode and exist_ok are the only keyword-only params)",
            }, tested=_ALL),
            may_raise_other=True,  # FileNotFoundError: parent missing; FileExistsError: path exists and exist_ok=False; PermissionError; OSError
            returns=ReturnSpec("None", "creates leaf and all parent directories; returns None", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False, notes="modifies the filesystem by creating directories"),
            source=("https://docs.python.org/3.14/library/os.html#os.makedirs", "CPython Lib/os.py::makedirs"), tested=True)),
    _contract_rule(
        "shutil.which", "ShutilWhich", "import shutil\nx = shutil.which(\"python3\")\n", "import shutil\nx = shutil.which()\n",
        Contract(
            qualname="shutil.which",
            raises=dunder_propagation_claims("path/command __fspath__, index and string protocol callbacks", native={
                "TypeError": "wrong argument binding or command/path/mode kinds, or protocol callback propagation",
            }, tested_native=frozenset({"TypeError"})),
            may_raise_other=True,  # OSError: path is not accessible; other filesystem errors reading PATH directories
            returns=ReturnSpec("object", "builtin commands return str/bytes paths, possibly relative, or None; accepted path/string subclass hooks may return another path object", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="searches the PATH for the executable"),
            source=("https://docs.python.org/3.14/library/shutil.html#shutil.which", "CPython Lib/shutil.py::which"), tested=True)),
    _contract_rule(
        "time.strftime", "TimeStrftime", 'import time\nx = time.strftime("%Y", (2020, 1, 1, 0, 0, 0, 0, 0, 0))\n',
        "import time\nx = time.strftime()\n",
        Contract(
            qualname="time.strftime",
            raises=dunder_propagation_claims(
                "tuple elements' __index__ (a user method for non-int numbers/objects)",
                native={
                    "TypeError": "format is not a str; tuple is not a 9- or 11-tuple or struct_time; an element has no __index__; wrong argument count or any keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # OverflowError if a tuple element is too large for C long; ValueError on platforms rejecting specific format codes
            returns=ReturnSpec("str", "a formatted time string according to the format specification", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="reads current time from localtime() when second argument is omitted; depends on locale and timezone"),
            source=("https://docs.python.org/3.14/library/time.html#time.strftime", "CPython Modules/timemodule.c::time_strftime"), tested=True)),
    _contract_rule(
        "pickle.dumps", "PickleDumps", 'import pickle\nx = pickle.dumps([1, 2, 3])\n',
        "import pickle\nx = pickle.dumps()\n",
        Contract(
            qualname="pickle.dumps",
            raises=dunder_propagation_claims(
                "an object's __reduce__, __reduce_ex__, __getstate__, or buffer_callback",
                native={
                    "TypeError": "missing required argument 'obj'; protocol cannot be interpreted as integer; buffer_callback is not callable; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # pickle.PicklingError: unpicklable object; ValueError: protocol must be <= 5, or buffer_callback with protocol < 5; RecursionError for deep structures
            returns=ReturnSpec("bytes", "the serialized pickle byte stream", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=True, notes="pure unless custom __reduce__, __getstate__ or buffer_callback execute user hooks"),
            source=("https://docs.python.org/3.14/library/pickle.html#pickle.dumps", "CPython Modules/_pickle.c::dump"), tested=True)),
    _contract_rule(
        "pickle.loads", "PickleLoads", 'import pickle\nx = pickle.loads(b"(lp0\\nI1\\naI2\\naI3\\na.")\n',
        "import pickle\nx = pickle.loads()\n",
        Contract(
            qualname="pickle.loads",
            raises=dunder_propagation_claims(
                "an unpickled object's __setstate__, __init__, __new__, or reduction callable",
                native={
                    "TypeError": "data is not a bytes-like object; missing argument or positional keyword; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # pickle.UnpicklingError: invalid pickle byte stream; EOFError: truncated pickle stream; AttributeError / ImportError if class not found
            returns=ReturnSpec("object", "the unpickled Python object reconstructed from the byte stream", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=True, notes="may instantiate classes and call __setstate__ or callable factories"),
            source=("https://docs.python.org/3.14/library/pickle.html#pickle.loads", "CPython Modules/_pickle.c::load"), tested=True)),
    _contract_rule(
        "contextlib.nullcontext", "ContextlibNullcontext", "import contextlib\nx = contextlib.nullcontext(42)\n",
        "import contextlib\nx = contextlib.nullcontext(1, 2)\n",
        Contract(
            qualname="contextlib.nullcontext",
            raises=never_claims(native={"TypeError": "more than 1 positional argument or an unexpected keyword argument"}, tested=_ALL),
            may_raise_other=False,
            returns=ReturnSpec("contextlib.nullcontext", "a context manager wrapping enter_result (defaults to None)", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=False),
            source=("https://docs.python.org/3.14/library/contextlib.html#contextlib.nullcontext", "CPython Lib/contextlib.py::nullcontext"), tested=True)),
    _contract_rule(
        "tempfile.gettempdir", "TempfileGettempdir", "import tempfile\nx = tempfile.gettempdir()\n",
        "import tempfile\nx = tempfile.gettempdir(1)\n",
        Contract(
            qualname="tempfile.gettempdir",
            raises=never_claims(native={"TypeError": "any positional argument or keyword argument is given"}, tested=_ALL),
            may_raise_other=False,
            returns=ReturnSpec("str", "the default directory name used for temporary files", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False, notes="returns the name of the directory used for temporary files; reads env vars on initial resolution"),
            source=("https://docs.python.org/3.14/library/tempfile.html#tempfile.gettempdir", "CPython Lib/tempfile.py::gettempdir"), tested=True)),
    _contract_rule(
        "os.replace", "OsReplace",
        'import os\nwith open("/tmp/1493_test_r1", "w") as f: f.write("a")\nos.replace("/tmp/1493_test_r1", "/tmp/1493_test_r2")\nos.remove("/tmp/1493_test_r2")\n',
        "import os\nx = os.replace()\n",
        Contract(
            qualname="os.replace",
            raises=dunder_propagation_claims(
                "src or dst's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "src or dst is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # FileNotFoundError: src does not exist; PermissionError; OSError
            returns=ReturnSpec("None", "atomically renames src to dst; returns None", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="modifies the filesystem by atomically renaming src to dst; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/os.html#os.replace", "CPython Modules/posixmodule.c::os_replace_impl"), tested=True)),
    _contract_rule(
        "types.SimpleNamespace", "TypesSimpleNamespace", "import types\nx = types.SimpleNamespace(a=1, b=2)\n",
        "import types\nx = types.SimpleNamespace(1, 2)\n",
        Contract(
            qualname="types.SimpleNamespace",
            raises=dunder_propagation_claims(
                "positional mapping's keys / __getitem__ or iterable's __iter__ (user methods)",
                native={
                    "TypeError": "more than 1 positional argument; positional argument is not an iterable or mapping; or any key is not a str",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # ValueError if iterable items are not pairs of length 2
            returns=ReturnSpec("types.SimpleNamespace", "a simple namespace object with attributes from the arguments", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=True, notes="pure unless positional mapping or iterable executes user code"),
            source=("https://docs.python.org/3.14/library/types.html#types.SimpleNamespace", "CPython Objects/namespaceobject.c"), tested=True)),
    _contract_rule(
        "linecache.getline", "LinecacheGetline", 'import linecache\nx = linecache.getline("/etc/os-release", 1)\n',
        "import linecache\nx = linecache.getline()\n",
        Contract(
            qualname="linecache.getline",
            raises=never_claims(native={
                "TypeError": "missing required arguments 'filename' and 'lineno'; more than 3 positional arguments; or an unexpected keyword argument",
            }, tested=_ALL),
            may_raise_other=False,
            returns=ReturnSpec("str", "the requested line including trailing newline, or empty string on error or EOF", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False, notes="reads file contents from filesystem or cache"),
            source=("https://docs.python.org/3.14/library/linecache.html#linecache.getline", "CPython Lib/linecache.py::getline"), tested=True)),
    _contract_rule(
        "tempfile.gettempprefix", "TempfileGettempprefix", "import tempfile\nx = tempfile.gettempprefix()\n",
        "import tempfile\nx = tempfile.gettempprefix(1)\n",
        Contract(
            qualname="tempfile.gettempprefix",
            raises=never_claims(native={"TypeError": "any positional argument or keyword argument is given"}, tested=_ALL),
            may_raise_other=False,
            returns=ReturnSpec("str", "the filename prefix used to create temporary files", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False, notes="returns the default prefix for temporary files"),
            source=("https://docs.python.org/3.14/library/tempfile.html#tempfile.gettempprefix", "CPython Lib/tempfile.py::gettempprefix"), tested=True)),
    _contract_rule(
        "os.remove", "OsRemove", 'import os\nwith open("/tmp/1493_test_remove", "w") as f: f.write("a")\nx = os.remove("/tmp/1493_test_remove")\n',
        "import os\nx = os.remove()\n",
        Contract(
            qualname="os.remove",
            raises=dunder_propagation_claims(
                "path's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "path is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # FileNotFoundError: path does not exist; PermissionError; OSError
            returns=ReturnSpec("None", "deletes the file at path; returns None", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="modifies the filesystem by deleting a file; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/os.html#os.remove", "CPython Modules/posixmodule.c::os_unlink_impl"), tested=True)),
    _contract_rule(
        "os.unlink", "OsUnlink", 'import os\nwith open("/tmp/1493_test_unlink", "w") as f: f.write("a")\nx = os.unlink("/tmp/1493_test_unlink")\n',
        "import os\nx = os.unlink()\n",
        Contract(
            qualname="os.unlink",
            raises=dunder_propagation_claims(
                "path's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "path is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # FileNotFoundError: path does not exist; PermissionError; OSError
            returns=ReturnSpec("None", "deletes the file at path; returns None", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="modifies the filesystem by deleting a file; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/os.html#os.unlink", "CPython Modules/posixmodule.c::os_unlink_impl"), tested=True)),
    _contract_rule(
        "glob.glob", "GlobGlob", 'import glob\nx = glob.glob("/tmp/*")\n', "import glob\nx = glob.glob()\n",
        Contract(
            qualname="glob.glob",
            raises=dunder_propagation_claims(
                "pathname's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "pathname is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # OSError: filesystem errors reading directories
            returns=ReturnSpec("list", "the list of paths matching pathname, sorted; may be empty", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="reads the filesystem; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/glob.html#glob.glob", "CPython Lib/glob.py::glob"), tested=True)),
    _contract_rule(
        "glob.iglob", "GlobIglob", 'import glob\nx = glob.iglob("/tmp/*")\n', "import glob\nx = glob.iglob()\n",
        Contract(
            qualname="glob.iglob",
            raises=dunder_propagation_claims(
                "pathname's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "pathname is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # OSError: filesystem errors reading directories
            returns=ReturnSpec("iterator", "an iterator yielding paths matching pathname, one by one", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="reads the filesystem; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/glob.html#glob.iglob", "CPython Lib/glob.py::iglob"), tested=True)),
    _contract_rule(
        "os.scandir", "OsScandir", 'import os\nx = os.scandir("/tmp")\n', "import os\nx = os.scandir(1, 2)\n",
        Contract(
            qualname="os.scandir",
            raises=dunder_propagation_claims(
                "path's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "too many arguments; or an unexpected keyword argument; path is not a str, bytes, os.PathLike or file descriptor",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # FileNotFoundError: path does not exist; NotADirectoryError: not a dir; PermissionError; OSError
            returns=ReturnSpec("os.DirEntry iterator", "an iterator of os.DirEntry for each entry in the directory; close() must be called (or the iterator exhausted)", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="reads the directory from the filesystem; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/os.html#os.scandir", "CPython Modules/posixmodule.c::os_scandir"), tested=True)),
    _contract_rule(
        "os.rename", "OsRename",
        'import os\nwith open("/tmp/1493_test_rn1", "w") as f: f.write("a")\nos.rename("/tmp/1493_test_rn1", "/tmp/1493_test_rn2")\nos.remove("/tmp/1493_test_rn2")\n',
        "import os\nx = os.rename()\n",
        Contract(
            qualname="os.rename",
            raises=dunder_propagation_claims(
                "src or dst's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "src or dst is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # FileNotFoundError: src does not exist; PermissionError; OSError (cross-device link, etc.)
            returns=ReturnSpec("None", "renames/moves src to dst; returns None", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="modifies the filesystem by renaming/moving a file; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/os.html#os.rename", "CPython Modules/posixmodule.c::os_rename_impl"), tested=True)),
    _contract_rule(
        "os.rmdir", "OsRmdir", 'import os\nos.mkdir("/tmp/1493_test_rmdir_x")\nx = os.rmdir("/tmp/1493_test_rmdir_x")\n',
        "import os\nx = os.rmdir()\n",
        Contract(
            qualname="os.rmdir",
            raises=dunder_propagation_claims(
                "path's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "path is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # FileNotFoundError: path does not exist; NotADirectoryError: not a dir; OSError: dir not empty; PermissionError
            returns=ReturnSpec("None", "removes the directory at path; returns None", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="modifies the filesystem by removing a directory; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/os.html#os.rmdir", "CPython Modules/posixmodule.c::os_rmdir_impl"), tested=True)),
    _contract_rule(
        "os.mkdir", "OsMkdir",
        'import os\nx = os.mkdir("/tmp/1493_test_mkdir")\nos.rmdir("/tmp/1493_test_mkdir")\n',
        "import os\nx = os.mkdir()\n",
        Contract(
            qualname="os.mkdir",
            raises=dunder_propagation_claims(
                "path's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "path is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # FileNotFoundError: parent missing; FileExistsError: path exists; PermissionError; OSError
            returns=ReturnSpec("None", "creates the directory at path; returns None", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="modifies the filesystem by creating a directory; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/os.html#os.mkdir", "CPython Modules/posixmodule.c::os_mkdir_impl"), tested=True)),
    _contract_rule(
        "os.walk", "OsWalk",
        'import os\nfor _ in os.walk("/tmp"): pass\n',
        "import os\nx = os.walk()\n",
        Contract(
            qualname="os.walk",
            raises=dunder_propagation_claims(
                "top's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "top is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # FileNotFoundError: path does not exist; PermissionError; OSError
            returns=ReturnSpec("generator", "yields (dirpath, dirnames, filenames) tuples for each directory in the tree", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="traverses the filesystem; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/os.html#os.walk", "CPython Lib/os.py::walk"), tested=True)),
    _contract_rule(
        "os.symlink", "OsSymlink",
        'import os\nwith open("/tmp/1493_test_sl", "w") as f: f.write("a")\nos.symlink("/tmp/1493_test_sl", "/tmp/1493_test_sl_link")\nos.remove("/tmp/1493_test_sl")\nos.remove("/tmp/1493_test_sl_link")\n',
        "import os\nx = os.symlink()\n",
        Contract(
            qualname="os.symlink",
            raises=dunder_propagation_claims(
                "src or dst's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "src or dst is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # FileNotFoundError: src does not exist; FileExistsError: dst exists; PermissionError; OSError; NotImplementedError on platforms without symlink support
            returns=ReturnSpec("None", "creates a symbolic link from src to dst; returns None", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="modifies the filesystem by creating a symbolic link; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/os.html#os.symlink", "CPython Modules/posixmodule.c::os_symlink_impl"), tested=True)),
    _contract_rule(
        "shutil.copy", "ShutilCopy",
        'import shutil, os\nwith open("/tmp/1493_test_cp", "w") as f: f.write("a")\nx = shutil.copy("/tmp/1493_test_cp", "/tmp/1493_test_cp_dst")\nos.remove("/tmp/1493_test_cp")\nos.remove("/tmp/1493_test_cp_dst")\n',
        "import shutil\nx = shutil.copy()\n",
        Contract(
            qualname="shutil.copy",
            raises=dunder_propagation_claims(
                "src or dst's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "src or dst is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # FileNotFoundError: src does not exist; PermissionError; OSError; SameFileError
            returns=ReturnSpec("object", "the destination path; a nondirectory destination is returned unchanged (including bytes and PathLike)", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="modifies the filesystem by copying a file; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/shutil.html#shutil.copy", "CPython Lib/shutil.py::copy"), tested=True)),
    _contract_rule(
        "time.localtime", "TimeLocaltime", "import time\nx = time.localtime(0)\n", "import time\nx = time.localtime(1, 2)\n",
        Contract(
            qualname="time.localtime",
            raises=dunder_propagation_claims(
                "secs's __index__ (a user method for non-int/float objects)",
                native={
                    "TypeError": "secs is not an int, float, or object with __index__; wrong argument count or an unknown keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # OverflowError: secs is out of platform range; ValueError: impossible time (some platforms)
            returns=ReturnSpec("time.struct_time", "the local time broken into year, month, day, hour, minute, second, weekday, yearday, isdst", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="reads local timezone when secs is omitted, or converts a universal timestamp to local time; calls __index__ on non-int/float objects"),
            source=("https://docs.python.org/3.14/library/time.html#time.localtime", "CPython Modules/timemodule.c::time_localtime"), tested=True)),
    _contract_rule(
        "time.gmtime", "TimeGmtime", "import time\nx = time.gmtime(0)\n", "import time\nx = time.gmtime(1, 2)\n",
        Contract(
            qualname="time.gmtime",
            raises=dunder_propagation_claims(
                "secs's __index__ (a user method for non-int/float objects)",
                native={
                    "TypeError": "secs is not an int, float, or object with __index__; wrong argument count or an unknown keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # OverflowError: secs is out of platform range; ValueError: impossible time
            returns=ReturnSpec("time.struct_time", "the UTC time broken into year, month, day, hour, minute, second, weekday, yearday, isdst=0", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=True, notes="UTC time conversion; pure when secs is given, always UTC; calls __index__ on non-int/float objects"),
            source=("https://docs.python.org/3.14/library/time.html#time.gmtime", "CPython Modules/timemodule.c::time_gmtime"), tested=True)),
    _contract_rule(
        "time.mktime", "TimeMktime", "import time\nx = time.mktime((2020, 1, 1, 0, 0, 0, 0, 0, 0))\n", "import time\nx = time.mktime()\n",
        Contract(
            qualname="time.mktime",
            raises=dunder_propagation_claims(
                "tuple elements' __index__ (a user method for non-int numbers/objects in the tuple)",
                native={
                    "TypeError": "argument is not a 9-tuple or struct_time; wrong argument count or keyword argument; an element has no __index__",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # OverflowError: the tuple elements produce a value that is out of range for the platform mktime; ValueError: impossible calendar date
            returns=ReturnSpec("float", "the seconds since the epoch (as a float)", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="reads the local timezone to interpret the tuple; calls __index__ on non-int elements"),
            source=("https://docs.python.org/3.14/library/time.html#time.mktime", "CPython Modules/timemodule.c::time_mktime"), tested=True)),
    _contract_rule(
        "os.link", "OsLink",
        'import os\nwith open("/tmp/1493_test_hl", "w") as f: f.write("a")\nos.link("/tmp/1493_test_hl", "/tmp/1493_test_hl2")\nos.remove("/tmp/1493_test_hl")\nos.remove("/tmp/1493_test_hl2")\n',
        "import os\nx = os.link()\n",
        Contract(
            qualname="os.link",
            raises=dunder_propagation_claims(
                "src or dst's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "src or dst is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # FileNotFoundError: src does not exist; FileExistsError: dst exists; PermissionError; OSError; NotImplementedError on platforms without hard links
            returns=ReturnSpec("None", "creates a hard link from src to dst; returns None", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="modifies the filesystem by creating a hard link; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/os.html#os.link", "CPython Modules/posixmodule.c::os_link_impl"), tested=True)),
    _contract_rule(
        "shutil.copy2", "ShutilCopy2",
        'import shutil, os\nwith open("/tmp/1493_test_cp2", "w") as f: f.write("a")\nx = shutil.copy2("/tmp/1493_test_cp2", "/tmp/1493_test_cp2_d")\nos.remove("/tmp/1493_test_cp2")\nos.remove("/tmp/1493_test_cp2_d")\n',
        "import shutil\nx = shutil.copy2()\n",
        Contract(
            qualname="shutil.copy2",
            raises=dunder_propagation_claims(
                "src or dst's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "src or dst is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # FileNotFoundError: src does not exist; PermissionError; OSError; SameFileError
            returns=ReturnSpec("object", "the destination path (preserving metadata); a nondirectory destination is returned unchanged, including bytes and PathLike", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="modifies the filesystem by copying a file with metadata; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/shutil.html#shutil.copy2", "CPython Lib/shutil.py::copy2"), tested=True)),
    _contract_rule(
        "time.asctime", "TimeAsctime", "import time\nx = time.asctime((2020, 1, 1, 0, 0, 0, 0, 0, 0))\n", "import time\nx = time.asctime(1, 2)\n",
        Contract(
            qualname="time.asctime",
            raises=never_claims(native={
                "TypeError": "more than 1 positional argument or a keyword argument; argument is not a 9-tuple or struct_time",
            }, tested=_ALL),
            may_raise_other=True,  # OverflowError on some platforms for out-of-range elements
            returns=ReturnSpec("str", "a native date/time string with unpadded year digits; its length varies with the year", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False, notes="pure when given a tuple (same output for same input); reads local time when called without arguments"),
            source=("https://docs.python.org/3.14/library/time.html#time.asctime", "CPython Modules/timemodule.c::time_asctime"), tested=True)),
    _contract_rule(
        "time.ctime", "TimeCtime", "import time\nx = time.ctime(0)\n", "import time\nx = time.ctime(1, 2)\n",
        Contract(
            qualname="time.ctime",
            raises=dunder_propagation_claims(
                "secs's __index__ (a user method for non-int/float objects)",
                native={
                    "TypeError": "secs is not an int, float, or object with __index__; wrong argument count or keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,  # OverflowError: secs is out of platform range; ValueError: impossible time
            returns=ReturnSpec("str", "a native local date/time string; its length varies with the year", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True, notes="reads local timezone (omitted secs defaults to now); calls __index__ on non-int/float objects"),
            source=("https://docs.python.org/3.14/library/time.html#time.ctime", "CPython Modules/timemodule.c::time_ctime"), tested=True)),
    _contract_rule(
        "base64.b16encode", "Base64B16encode",
        'import base64\nx = base64.b16encode(b"ab")\n',
        "import base64\nx = base64.b16encode()\n",
        Contract(
            qualname="base64.b16encode",
            raises=dunder_propagation_claims(
                "s's __buffer__ (a user method)",
                native={"TypeError": "s is not bytes-like (a str included); wrong argument count or an unknown keyword"},
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,
            returns=ReturnSpec("bytes", "hex uppercase-encoded bytes (2 chars per input byte)", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=True),
            source=("https://docs.python.org/3.14/library/base64.html#base64.b16encode", "CPython Lib/base64.py::b16encode"), tested=True)),
    _contract_rule(
        "base64.b16decode", "Base64B16decode",
        'import base64\nx = base64.b16decode(b"6162")\n',
        "import base64\nx = base64.b16decode()\n",
        Contract(
            qualname="base64.b16decode",
            raises=dunder_propagation_claims(
                "s's __buffer__ (a user method; casefold=True does not add dunder paths)",
                native={"TypeError": "s is neither a bytes-like nor an ASCII str; wrong argument count or unknown keyword argument"},
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,
            returns=ReturnSpec("bytes", "the bytes decoded from hex; casefold=False rejects lowercase", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=True),
            source=("https://docs.python.org/3.14/library/base64.html#base64.b16decode", "CPython Lib/base64.py::b16decode"), tested=True)),
    _contract_rule(
        "shutil.move", "ShutilMove",
        'import shutil, os\nwith open("/tmp/1493_test_mv", "w") as f: f.write("a")\nx = shutil.move("/tmp/1493_test_mv", "/tmp/1493_test_mv2")\nos.remove("/tmp/1493_test_mv2")\n',
        "import shutil\nx = shutil.move()\n",
        Contract(
            qualname="shutil.move",
            raises=dunder_propagation_claims(
                "src or dst's __fspath__ (a user method for os.PathLike objects)",
                native={
                    "TypeError": "src or dst is not a str, bytes, os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,
            returns=ReturnSpec("object", "the destination path, including the unchanged original PathLike or bytes destination when it is not a directory", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True,
                          notes="moves a file or directory; calls __fspath__ on path-like objects"),
            source=("https://docs.python.org/3.14/library/shutil.html#shutil.move", "CPython Lib/shutil.py::move"), tested=True)),
    _contract_rule(
        "shutil.rmtree", "ShutilRmtree",
        'import os, shutil\nos.mkdir("/tmp/1493_test_rt")\nshutil.rmtree("/tmp/1493_test_rt")\n',
        "import shutil\nx = shutil.rmtree()\n",
        Contract(
            qualname="shutil.rmtree",
            raises=dunder_propagation_claims(
                "path's __fspath__ (a user method for os.PathLike objects); onerror (user callback) if given",
                native={
                    "TypeError": "path is not a str, bytes, or os.PathLike; wrong argument count; or an unexpected keyword argument",
                },
                tested_native=frozenset({"TypeError"}),
            ),
            may_raise_other=True,
            returns=ReturnSpec("None", "removes the directory tree at path; returns None", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True,
                          notes="removes a directory tree; calls __fspath__ on path-like objects; calls onerror if given"),
            source=("https://docs.python.org/3.14/library/shutil.html#shutil.rmtree", "CPython Lib/shutil.py::rmtree"), tested=True)),
    _contract_rule(
        "atexit.register", "AtexitRegister",
        'import atexit\ndef _f(): pass\nx = atexit.register(_f)\n',
        "import atexit\nx = atexit.register()\n",
        Contract(
            qualname="atexit.register",
            raises=never_claims(native={
                "TypeError": "missing the required func argument; more than 1 positional argument; or an unexpected keyword argument",
            }, tested=_ALL),
            may_raise_other=False,
            returns=ReturnSpec("func", "the func argument (the registered callback)", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=False,
                          notes="registers func to be called at exit; never invokes func during registration"),
            source=("https://docs.python.org/3.14/library/atexit.html#atexit.register", "CPython Lib/atexit.py::register"), tested=True)),
    _contract_rule(
        "string.Formatter", "StringFormatter",
        'import string\nx = string.Formatter()\n',
        "import string\nx = string.Formatter(1)\n",
        Contract(
            qualname="string.Formatter",
            raises=never_claims(native={
                "TypeError": "any argument or keyword argument",
            }, tested=_ALL),
            may_raise_other=False,
            returns=ReturnSpec("string.Formatter", "a new Formatter instance", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=False),
            source=("https://docs.python.org/3.14/library/string.html#string.Formatter", "CPython Lib/string.py::Formatter"), tested=True)),
    _contract_rule(
        "inspect.isfunction", "InspectIsfunction",
        "import inspect\nx = inspect.isfunction(len)\n",
        "import inspect\nx = inspect.isfunction(1, 2)\n",
        Contract(
            qualname="inspect.isfunction",
            raises=never_claims(native={"TypeError": "wrong argument count or a keyword argument"}, tested=_ALL),
            may_raise_other=False,
            returns=ReturnSpec("bool", "True for a plain types.FunctionType object (a def / lambda, not a builtin or a method), else False", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=False,
                          notes="exactly isinstance(object, types.FunctionType); the C subtype fast path never consults a metaclass __instancecheck__, so no user code can run"),
            source=("https://docs.python.org/3.14/library/inspect.html#inspect.isfunction", "CPython Lib/inspect.py::isfunction"), tested=True)),
    _contract_rule(
        "inspect.iscoroutinefunction", "InspectIscoroutinefunction",
        "import asyncio, inspect\nx = inspect.iscoroutinefunction(asyncio.sleep)\n",
        "import inspect\nx = inspect.iscoroutinefunction(1, 2)\n",
        Contract(
            qualname="inspect.iscoroutinefunction",
            raises=dunder_propagation_claims(
                "a duck-typed callable's __code__ (and __name__ / __defaults__ / __kwdefaults__ / -- for the mark path -- _is_coroutine_marker property), a user method read via getattr on a non-function callable",
                native={"TypeError": "wrong argument count or a keyword argument",
                        "AttributeError": "never (the duck-typed attribute reads go through getattr with a default, so an AttributeError there means 'no such attribute')"},
                tested_native=frozenset({"TypeError", "AttributeError"}),
            ),
            may_raise_other=True,  # a user property can raise anything, a non-property class included
            returns=ReturnSpec("bool", "True for an async def function (or a function-like / method / partial wrapping one, or an object marked with markcoroutinefunction; functools.wraps copies the flag), else False", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=True,
                          notes="reads the code flags of duck-typed callables via getattr, so user properties run; plain objects and builtins are constant-time False"),
            source=("https://docs.python.org/3.14/library/inspect.html#inspect.iscoroutinefunction", "CPython Lib/inspect.py::iscoroutinefunction"), tested=True)),
    _contract_rule(
        "inspect.isgeneratorfunction", "InspectIsgeneratorfunction",
        "import inspect\nx = inspect.isgeneratorfunction(lambda: (yield))\n",
        "import inspect\nx = inspect.isgeneratorfunction(1, 2)\n",
        Contract(
            qualname="inspect.isgeneratorfunction",
            raises=dunder_propagation_claims(
                "a duck-typed callable's __code__ (and __name__ / __defaults__ / __kwdefaults__ property), a user method read via getattr on a non-function callable",
                native={"TypeError": "wrong argument count or a keyword argument",
                        "AttributeError": "never (the duck-typed attribute reads go through getattr with a default, so an AttributeError there means 'no such attribute')"},
                tested_native=frozenset({"TypeError", "AttributeError"}),
            ),
            may_raise_other=True,  # a user property can raise anything, a non-property class included
            returns=ReturnSpec("bool", "True for a def with yield (or a function-like / method / partial wrapping one), else False", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=True,
                          notes="reads the code flags of duck-typed callables via getattr, so user properties run; plain objects and builtins are constant-time False"),
            source=("https://docs.python.org/3.14/library/inspect.html#inspect.isgeneratorfunction", "CPython Lib/inspect.py::isgeneratorfunction"), tested=True)),
    _contract_rule(
        "inspect.isasyncgenfunction", "InspectIsasyncgenfunction",
        "import inspect\nx = inspect.isasyncgenfunction(print)\n",
        "import inspect\nx = inspect.isasyncgenfunction(1, 2)\n",
        Contract(
            qualname="inspect.isasyncgenfunction",
            raises=dunder_propagation_claims(
                "a duck-typed callable's __code__ (and __name__ / __defaults__ / __kwdefaults__ property), a user method read via getattr on a non-function callable",
                native={"TypeError": "wrong argument count or a keyword argument",
                        "AttributeError": "never (the duck-typed attribute reads go through getattr with a default, so an AttributeError there means 'no such attribute')"},
                tested_native=frozenset({"TypeError", "AttributeError"}),
            ),
            may_raise_other=True,  # a user property can raise anything, a non-property class included
            returns=ReturnSpec("bool", "True for an async def with yield (or a function-like / method / partial wrapping one), else False", tested=True),
            effects=Effects(deterministic=True, may_execute_callback=True,
                          notes="reads the code flags of duck-typed callables via getattr, so user properties run; plain objects and builtins are constant-time False"),
            source=("https://docs.python.org/3.14/library/inspect.html#inspect.isasyncgenfunction", "CPython Lib/inspect.py::isasyncgenfunction"), tested=True)),
    _contract_rule(
        "sys.setrecursionlimit", "SysSetrecursionlimit",
        "import sys\nx = sys.setrecursionlimit(1000)\n",
        "import sys\nx = sys.setrecursionlimit(1.5)\n",
        Contract(
            qualname="sys.setrecursionlimit",
            raises=dunder_propagation_claims(
                "limit's __index__ (a user method for int-like objects)",
                native={
                    "TypeError": "limit is not an int or int-like object with __index__ (a str / float / None)",
                    "ValueError": "limit is less than 1",
                },
                tested_native=frozenset({"TypeError", "ValueError"}),
            ),
            may_raise_other=True,  # RecursionError: the new limit is below the current recursion depth
            returns=ReturnSpec("None", "sets the interpreter's recursion depth limit; returns None", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True,
                          notes="mutates process-global interpreter state; calls __index__ on int-like objects"),
            source=("https://docs.python.org/3.14/library/sys.html#sys.setrecursionlimit", "CPython Python/sysmodule.c::sys_setrecursionlimit"), tested=True)),
    _contract_rule(
        "sys.getrecursionlimit", "SysGetrecursionlimit",
        "import sys\nx = sys.getrecursionlimit()\n", "import sys\nx = sys.getrecursionlimit(1)\n",
        Contract(
            qualname="sys.getrecursionlimit",
            raises=never_claims(native={"TypeError": "any argument or keyword"}, tested=_ALL),
            may_raise_other=False,
            returns=ReturnSpec("int", "the current recursion depth limit", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False,
                          notes="reads process-global interpreter state; changes across setrecursionlimit calls"),
            source=("https://docs.python.org/3.14/library/sys.html#sys.getrecursionlimit", "CPython Python/sysmodule.c::sys_getrecursionlimit"), tested=True)),
    _contract_rule(
        "sys._getframe", "SysGetframe",
        "import sys\nx = sys._getframe(0)\n",
        "import sys\nx = sys._getframe(1, 2)\n",
        Contract(
            qualname="sys._getframe",
            raises=dunder_propagation_claims(
                "depth's __index__ (a user method for int-like objects)",
                native={
                    "TypeError": "depth is not an int or int-like object with __index__ (a float / str)",
                    "ValueError": "call stack is not deep enough for depth",
                },
                tested_native=frozenset({"TypeError", "ValueError"}),
            ),
            may_raise_other=True,  # user __index__ code can raise anything
            returns=ReturnSpec("frame", "the frame object depth levels up from the caller (default 0); negative depths return the same current frame as depth 0", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True,
                          notes="reflects the live call stack; calls __index__ on int-like objects"),
            source=("https://docs.python.org/3.14/library/sys.html#sys._getframe", "CPython Python/sysmodule.c::sys__getframe"), tested=True)),
    _contract_rule(
        "sys.exit", "SysExit",
        "import sys\nx = None\ntry:\n    sys.exit(3)\nexcept SystemExit as e:\n    x = e.code\n",
        "import sys\nx = sys.exit(1, 2)\n",
        Contract(
            qualname="sys.exit",
            raises=never_claims(native={"TypeError": "more than one positional argument or an unexpected keyword argument"}, tested=_ALL),
            may_raise_other=True,  # always raises SystemExit (not one of the property classes); the status argument is returned as-is, no dunder paths
            returns=ReturnSpec("never", "raises SystemExit(status) immediately; never falls through", tested=True),
            effects=Effects(clock_or_env=False, deterministic=False, may_execute_callback=False,
                          notes="unwinds to the SystemExit handlers (or exits the process); stdout is flushed on the way out"),
            source=("https://docs.python.org/3.14/library/sys.html#sys.exit", "CPython Python/sysmodule.c::sys_exit"), tested=True)),
    _contract_rule(
        "importlib.import_module", "ImportlibImport_module",
        "import importlib\nx = importlib.import_module(\"json\")\n",
        "import importlib\nx = importlib.import_module(\"json\", \"pkg\", \"extra\")\n",
        Contract(
            qualname="importlib.import_module",
            raises=dunder_propagation_claims(
                "the imported module's top-level code (arbitrary user code), a finder/loader hook or user __getattr__ on the import system",
                native={
                    "TypeError": "name is a bytes (not a str); more than 2 positional arguments or an unexpected keyword argument",
                    "AttributeError": "name is not a str and has no startswith (an int / None)",
                },
                tested_native=frozenset({"TypeError", "AttributeError"}),
            ),
            may_raise_other=True,  # ImportError / ModuleNotFoundError and anything a module, finder or loader raises
            returns=ReturnSpec("module", "the imported module object; an already-imported module comes back from sys.modules", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True,
                          notes="imports and executes module code, mutates sys.modules; reads the import hooks and the file system"),
            source=("https://docs.python.org/3.14/library/importlib.html#importlib.import_module", "CPython Lib/importlib/__init__.py::import_module"), tested=True)),
    _contract_rule(
        "threading.stack_size", "ThreadingStack_size",
        "import threading\nx = threading.stack_size(0)\n",
        "import threading\nx = threading.stack_size(0.5)\n",
        Contract(
            qualname="threading.stack_size",
            raises=dunder_propagation_claims(
                "size's __index__ (a user method for int-like objects)",
                native={
                    "TypeError": "size is not an int or an int-like object with __index__ (None, a float, a str); more than 1 positional argument or an unexpected keyword argument",
                    "ValueError": "size is neither 0 (the system default) nor at least the platform minimum (53248 bytes on this Linux build)",
                },
                tested_native=frozenset({"TypeError", "ValueError"}),
            ),
            may_raise_other=True,  # user __index__ code can raise anything
            returns=ReturnSpec("int", "returns the previous thread stack size, including when size is 0; size 0 restores the system default; an omitted size also selects 0", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=True,
                          notes="mutates process-global interpreter state; calls __index__ on int-like objects"),
            source=("https://docs.python.org/3.14/library/threading.html#threading.stack_size", "CPython Modules/_threadmodule.c::thread_stack_size"), tested=True)),
    _contract_rule(
        "threading.current_thread", "ThreadingCurrent_thread",
        "import threading\nx = threading.current_thread()\n", "import threading\nx = threading.current_thread(1)\n",
        Contract(
            qualname="threading.current_thread", raises=never_claims(native={"TypeError": "any argument or keyword"}, tested=_ALL), may_raise_other=False,
            returns=ReturnSpec("threading.Thread", "the Thread object of the calling thread (an instance of the platform implementation class, not of threading.Thread)", tested=True),
            effects=Effects(clock_or_env=True, deterministic=False, may_execute_callback=False),
            source=("https://docs.python.org/3.14/library/threading.html#threading.current_thread", "CPython Modules/_threadmodule.c::thread__current_thread"), tested=True)),

]

_HASH_META = {"md5": (16, 64), "sha1": (20, 64), "sha256": (32, 64), "sha512": (64, 128)}


def _hash_rule(name):
    digest_size, block_size = _HASH_META[name]
    qual = "hashlib." + name
    cls_name = "Hashlib" + name.capitalize()
    contract = Contract(
        qualname=qual,
        raises=dunder_propagation_claims(
            "data's __buffer__ (a user method)",
            native={"TypeError": "data is not a bytes-like object (a str must be encoded); more than 1 positional argument or an unexpected keyword argument"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # usedforsecurity / the FIPS provider context can raise anything
        returns=ReturnSpec("hashlib.HASH (the " + name + " class)", f"the {name} hash object; digest_size={digest_size}, block_size={block_size}; data='' (the default) gives the empty-message state", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(f"https://docs.python.org/3.14/library/hashlib.html#hashlib.{name}", "CPython Modules/_hashopenssl.c / Modules/md5module.c"), tested=True)
    return _contract_rule(qual, cls_name, f"import hashlib\nx = hashlib.{name}(b\"ab\")\n", f"import hashlib\nx = hashlib.{name}(\"ab\")\n", contract)


_HASHES = [_hash_rule(n) for n in _HASH_META]


__all__ = ["Base64B64encode", "Base64B64decode", "StructPack", "StructUnpack", "StructCalcsize"] + [c.__name__ for c in _CLOCKS + _EXTRA]
