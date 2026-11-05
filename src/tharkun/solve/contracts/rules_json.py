"""Contracts for ``json.loads``, ``json.dumps``, ``json.load``, and ``json.dump`` (standing batches 5 and 22, lane 1493).

Both are pure Python over a C scanner / encoder and call user code through hooks: ``loads`` runs ``object_hook``, ``object_pairs_hook``,
``parse_float`` / ``parse_int`` / ``parse_constant`` and a ``cls``; ``dumps`` runs ``default``, a ``cls``, a container's ``items`` /
``__iter__``, keys' ``__lt__`` (``sort_keys``) and ``indent``'s ``__index__``.  ``load`` delegates to ``fp.read()`` + ``loads()``;
``dump`` serializes chunks to ``fp.write()``. A hook or stream method that raises propagates ANY class, so every claim is a
propagation claim; the native outcomes (bad input type, JSON syntax, circular reference) are in the conditions.  Exact on CPython 3.14.4
(``tests/contracts/test_json_.py``).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/json.html"
_SRC = "CPython Lib/json/__init__.py, Lib/json/decoder.py, Lib/json/encoder.py, Modules/_json.c"


@register_rule
class JsonLoads(Rule):
    """The rule of ``json.loads(s, **kw)``."""

    SAMPLES = {"json.loads": 'import json\nx = json.loads("[1, 2]")\n'}
    TWINS = {"json.loads": "import json\nx = json.loads()\n"}

    names = ("json.loads",)
    CONTRACT = Contract(
        qualname="json.loads",
        raises=dunder_propagation_claims(
            "a hook: object_hook, object_pairs_hook, parse_float, parse_int, parse_constant or a cls (user code)",
            native={"TypeError": "s is not a str, bytes or bytearray; wrong argument count, a positional keyword argument, or an unknown keyword; a hook or cls is not callable"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # JSONDecodeError (a ValueError), UnicodeDecodeError for bad bytes, ValueError for a 4300+ digit int, RecursionError for deep nesting
        returns=ReturnSpec("object", "dict, list, str, int, float, bool or None (or whatever a hook returns); an exact bytes / bytearray is decoded as UTF-8, UTF-16 or UTF-32", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True, notes="pure unless a hook is given"),
        source=(f"{_DOCS}#json.loads", _SRC),
        tested=True,
    )


@register_rule
class JsonDumps(Rule):
    """The rule of ``json.dumps(obj, **kw)``."""

    SAMPLES = {"json.dumps": 'import json\nx = json.dumps({"a": [1, 2.5, None]})\n'}
    TWINS = {"json.dumps": "import json\nx = json.dumps()\n"}

    names = ("json.dumps",)
    CONTRACT = Contract(
        qualname="json.dumps",
        raises=dunder_propagation_claims(
            "default, a cls, a subclass container's items / __iter__, keys' __lt__ under sort_keys, or indent's __index__ (user code)",
            native={"TypeError": "an object is not serializable and default does not handle it; a dict key is not a str, int, float, bool or None (without skipkeys); sort_keys compares unorderable keys; wrong argument count or an unknown keyword; a non-callable default or cls"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: circular reference, or nan / inf with allow_nan=False; RecursionError for deep nesting
        returns=ReturnSpec("str", "the JSON text; the same input gives the same text", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True, notes="pure unless default, cls or a subclass container's methods run user code"),
        source=(f"{_DOCS}#json.dumps", _SRC),
        tested=True,
    )


@register_rule
class JsonLoad(Rule):
    """The rule of ``json.load(fp, **kw)``."""

    SAMPLES = {"json.load": 'import io, json\nx = json.load(io.StringIO("[1, 2]"))\n'}
    TWINS = {"json.load": "import json\nx = json.load()\n"}

    names = ("json.load",)
    CONTRACT = Contract(
        qualname="json.load",
        raises=dunder_propagation_claims(
            "fp.read() or a hook: object_hook, object_pairs_hook, parse_float, parse_int, parse_constant or a cls (user code)",
            native={
                "TypeError": "fp is missing, wrong argument count, a positional keyword argument, or an unknown keyword; a hook or cls is not callable; fp.read() does not return str, bytes or bytearray",
                "AttributeError": "fp has no attribute 'read', or its read() call raises AttributeError",
            },
            tested_native=frozenset({"TypeError", "AttributeError"}),
        ),
        may_raise_other=True,  # JSONDecodeError (a ValueError), UnicodeDecodeError for bad bytes, ValueError for a 4300+ digit int, RecursionError for deep nesting, OSError from fp.read()
        returns=ReturnSpec("object", "dict, list, str, int, float, bool or None (or whatever a hook returns); an exact bytes / bytearray from fp.read() is decoded as UTF-8, UTF-16 or UTF-32", tested=True),
        effects=Effects(io=True, deterministic=False, may_execute_callback=True, notes="reads from fp stream; mutates fp stream position"),
        source=(f"{_DOCS}#json.load", _SRC),
        tested=True,
    )


@register_rule
class JsonDump(Rule):
    """The rule of ``json.dump(obj, fp, **kw)``."""

    SAMPLES = {"json.dump": 'import io, json\nfp = io.StringIO()\njson.dump({"a": [1, 2.5, None]}, fp)\n'}
    TWINS = {"json.dump": "import json\nx = json.dump()\n"}

    names = ("json.dump",)
    CONTRACT = Contract(
        qualname="json.dump",
        raises=dunder_propagation_claims(
            "fp.write(), default, a cls, a subclass container's items / __iter__, keys' __lt__ under sort_keys, or indent's __index__ (user code)",
            native={
                "TypeError": "an object is not serializable and default does not handle it; a dict key is not a str, int, float, bool or None (without skipkeys); sort_keys compares unorderable keys; wrong argument count or an unknown keyword; a non-callable default or cls",
                "AttributeError": "fp has no attribute 'write', or its write() call raises AttributeError",
            },
            tested_native=frozenset({"TypeError", "AttributeError"}),
        ),
        may_raise_other=True,  # ValueError: circular reference, or nan / inf with allow_nan=False; RecursionError for deep nesting; OSError from fp.write()
        returns=ReturnSpec("None", "always None", tested=True),
        effects=Effects(io=True, deterministic=True, may_execute_callback=True, notes="serializes obj as JSON text into fp via fp.write()"),
        source=(f"{_DOCS}#json.dump", _SRC),
        tested=True,
    )


__all__ = ["JsonLoads", "JsonDumps", "JsonLoad", "JsonDump"]
