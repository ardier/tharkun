"""Callable builtin contracts missing from the earlier batches ([internal ref]), CPython 3.14.4.

This describes the call itself, not future iterator/descriptor use. Callback flags
include conversion, metadata lookup and buffer-export hooks. Native dispatch purity
is not a promise about user hooks (types.Effects); next/anext advance state and a
memoryview exports/aliases its argument. No new proof transfer function is implied:
SHAPE remains unverified and visitors without a model keep their named refusal.
The absint followup ([internal ref]) now consumes bool/callable/bin/hex/oct through
explicit tested transfers, preserving unknown/custom-hook refusals.

compile/eval have open-world metadata in rules_builtins_code; their effects still
refuse recipe admission and no proof transfer is implied. The exec contract records
its open execution boundary and grants no recipe or proof admission.

Intentionally NO callable contract (and consequently no recipe admission) for:
* setattr, delattr: arbitrary descriptor hooks and mutation of reachable objects;
* globals, locals: reflect caller frames and expose their namespaces (locals uses
  the Python 3.13+ optimized-scope snapshot semantics, PEP 667);
* vars, dir: frame reflection in the no-argument forms; object __dict__/__dir__
  hooks in the argument forms;
* input: stdin/stdout I/O and audit/readline hooks;
* breakpoint: invokes sys.breakpointhook, consulting PYTHONBREAKPOINT by default;
* exit, quit: site-installed helpers that close stdin and raise SystemExit;
* help: site/pydoc interactive I/O, imports and object introspection;
* copyright, credits, license: site-installed printers doing output; license may
  read files and page interactively. These are site helpers, not builtin functions.

Sources: Python 3.14 library/functions.html, library/constants.html#constants-added-by-the-site-module,
CPython 3.14 Python/bltinmodule.c and Lib/_sitebuiltins.py. These 14 omissions
are intentional safety refusals, not missing pure-call models.
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/functions.html"
_ALL = frozenset(ALL_PROPERTY_CLASSES)


@register_rule
class Exec(Rule):
    """Open code execution is metadata, never a claim that executing a string is modeled."""

    names = ('exec',)
    SAMPLES = {'exec': "exec('pass')\n"}
    TWINS = {'exec': 'exec()\n'}
    CONTRACT = Contract(
        qualname='exec',
        raises=tuple(RaiseClaim(exc,
            'invalid binding, unsupported source, failed source-buffer acquisition, invalid globals/locals/closure, '
            'or an uncaught TypeError from executed code or unsuppressed compilation/namespace/audit callbacks'
            if exc == 'TypeError' else
            f'an uncaught {exc} from executed code or unsuppressed compilation/namespace/audit callbacks',
            tested=True) for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('None', 'None after normal completion, including an eval-mode code object', tested=True),
        effects=Effects(
            mutates_args=('globals', 'locals'),
            clock_or_env=True,
            deterministic=False,
            may_execute_callback=True,
            notes='Reads caller namespaces/compiler state and invokes compilation/execution audit hooks; may insert '
                  '__builtins__ into globals. Executed code and namespace/buffer/audit hooks have arbitrary effects. '
                  'Native source-buffer acquisition failures are replaced by TypeError; they do not propagate '
                  'the original exception. This is not a purity, termination, binding, or literal-expansion model.',
        ),
        source=(f'{_DOCS}#exec',
                'CPython v3.14.4 Python/bltinmodule.c::builtin_exec_impl',
                'CPython v3.14.4 Python/pythonrun.c::_Py_SourceAsString'),
        tested=True,
    )



@register_rule
class Aiter(Rule):
    """The callable rule for ``aiter``."""

    names = ('aiter',)
    SAMPLES = {'aiter': 'class AsyncIter:\n    def __aiter__(self):\n        return self\n    def __anext__(self):\n        return 1\nx = aiter(AsyncIter())\n'}
    TWINS = {'aiter': 'x = aiter()\n'}
    CONTRACT = Contract(
        qualname='aiter',
        raises=dunder_propagation_claims(
            '__aiter__',
            native={"TypeError": 'the argument is not async iterable or __aiter__ returns an object without __anext__, or wrong call shape or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('AsyncIterator[Any]', 'calls __aiter__ now; requires an object with an __anext__ slot', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#aiter", 'CPython v3.14.4 Python/bltinmodule.c::builtin_aiter'),
        tested=True,
    )


@register_rule
class Anext(Rule):
    """The callable rule for ``anext``."""

    names = ('anext',)
    SAMPLES = {'anext': 'class AsyncIter:\n    def __aiter__(self):\n        return self\n    def __anext__(self):\n        return 1\nx = anext(AsyncIter())\n'}
    TWINS = {'anext': 'x = anext()\n'}
    CONTRACT = Contract(
        qualname='anext',
        raises=dunder_propagation_claims(
            '__anext__',
            native={"TypeError": 'the argument has no __anext__ slot, or wrong call shape or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('Any', 'without default: the __anext__ result, even if not awaitable; with default: a wrapper awaitable; awaiting is a separate operation', tested=True),
        effects=Effects(
            may_execute_callback=True,
            mutates_receiver=True,
            notes='Invokes __anext__ immediately (which may mutate); awaiting the result and default exhaustion handling are separate from this call.',
        ),
        source=(f"{_DOCS}#anext", 'CPython v3.14.4 Python/bltinmodule.c::builtin_anext_impl'),
        tested=True,
    )


@register_rule
class Bin(Rule):
    """The callable rule for ``bin``."""

    names = ('bin',)
    SAMPLES = {'bin': 'x = bin(3)\n'}
    TWINS = {'bin': 'x = bin()\n'}
    CONTRACT = Contract(
        qualname='bin',
        raises=dunder_propagation_claims(
            '__index__',
            native={"TypeError": 'wrong call shape, missing __index__, or a non-int __index__ result or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('str', 'base-2 spelling, optional minus followed by 0b; no leading zeros except zero itself', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#bin", 'CPython v3.14.4 Objects/abstract.c::PyNumber_ToBase'),
        tested=True,
    )


@register_rule
class Bool(Rule):
    """The callable rule for ``bool``."""

    names = ('bool',)
    SAMPLES = {'bool': 'x = bool(3)\n'}
    TWINS = {'bool': 'x = bool(1, 2)\n'}
    CONTRACT = Contract(
        qualname='bool',
        raises=dunder_propagation_claims(
            '__bool__, __len__ or length-result __index__',
            native={"TypeError": 'wrong call shape, a non-bool __bool__ result, or a non-indexable __len__ result or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('bool', 'False for a false truth value, otherwise True; zero arguments returns False', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#bool", 'CPython v3.14.4 Objects/boolobject.c::bool_new'),
        tested=True,
    )


@register_rule
class Bytearray(Rule):
    """The callable rule for ``bytearray``."""

    names = ('bytearray',)
    SAMPLES = {'bytearray': 'x = bytearray([0, 255])\n'}
    TWINS = {'bytearray': 'x = bytearray(None, None, None, None)\n'}
    CONTRACT = Contract(
        qualname='bytearray',
        raises=dunder_propagation_claims(
            '__index__, buffer, iterable, element conversion or codec hooks',
            native={"TypeError": 'wrong call shape, text without encoding, invalid source/encoding/error types, or non-indexable iterable elements or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('bytearray', 'new mutable byte sequence; accepts a nonnegative size, a buffer, an iterable of byte integers, or text with encoding', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#bytearray", 'CPython v3.14.4 Objects/bytearrayobject.c::bytearray___init___impl'),
        tested=True,
    )


@register_rule
class Bytes(Rule):
    """The callable rule for ``bytes``."""

    names = ('bytes',)
    SAMPLES = {'bytes': 'x = bytes([0, 255])\n'}
    TWINS = {'bytes': 'x = bytes(None, None, None, None)\n'}
    CONTRACT = Contract(
        qualname='bytes',
        raises=dunder_propagation_claims(
            '__bytes__, __index__, buffer, iterable, element conversion or codec hooks',
            native={"TypeError": 'wrong call shape, text without encoding, invalid source/encoding/error types, bad __bytes__ result, or non-indexable iterable elements or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('bytes', 'immutable byte sequence; __bytes__ must return bytes (subclasses accepted); other forms accept size, buffer, iterable or encoded text', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#bytes", 'CPython v3.14.4 Objects/bytesobject.c::bytes_new'),
        tested=True,
    )


@register_rule
class Callable(Rule):
    """The callable rule for ``callable``."""

    names = ('callable',)
    SAMPLES = {'callable': 'x = callable(3)\n'}
    TWINS = {'callable': 'x = callable()\n'}
    CONTRACT = Contract(
        qualname='callable',
        raises=never_claims({"TypeError": 'not exactly one positional argument'}, tested=_ALL),
        may_raise_other=True,
        returns=ReturnSpec('bool', 'whether the type has a call slot; does not call the object or guarantee a successful call', tested=True),
        effects=Effects(
            may_execute_callback=False,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#callable", 'CPython v3.14.4 Python/bltinmodule.c::builtin_callable'),
        tested=True,
    )


@register_rule
class Classmethod(Rule):
    """The callable rule for ``classmethod``."""

    names = ('classmethod',)
    SAMPLES = {'classmethod': 'x = classmethod(3)\n'}
    TWINS = {'classmethod': 'x = classmethod()\n'}
    CONTRACT = Contract(
        qualname='classmethod',
        raises=dunder_propagation_claims(
            'wrapped-object metadata attribute lookup',
            native={"AttributeError": "never", "TypeError": 'not exactly one positional argument or metadata lookup raises TypeError or the invoked hook raises TypeError'},
            tested_native=frozenset({"AttributeError", "TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('classmethod', 'descriptor wrapping the single argument, which need not be callable; copies available __module__/__name__/__qualname__/__doc__', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Copies metadata by attribute lookup; AttributeError is suppressed, other exceptions propagate. Wrapped callable is not invoked.',
        ),
        source=(f"{_DOCS}#classmethod", 'CPython v3.14.4 Objects/funcobject.c::cm_init, functools_wraps'),
        tested=True,
    )


@register_rule
class Complex(Rule):
    """The callable rule for ``complex``."""

    names = ('complex',)
    SAMPLES = {'complex': 'x = complex("1+2j")\n'}
    TWINS = {'complex': 'x = complex(None, None, None)\n'}
    CONTRACT = Contract(
        qualname='complex',
        raises=dunder_propagation_claims(
            '__complex__, __float__ or __index__',
            native={"TypeError": 'wrong call shape, unsupported operand, invalid conversion result, or a string combined with imag or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('complex', 'complex numeric conversion, parsed string, or real + imag*1j; complex-valued real/imag arguments are deprecated in 3.14', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#complex", 'CPython v3.14.4 Objects/complexobject.c::complex_new_impl'),
        tested=True,
    )


@register_rule
class Dict(Rule):
    """The callable rule for ``dict``."""

    names = ('dict',)
    SAMPLES = {'dict': 'x = dict([(1, 2)])\n'}
    TWINS = {'dict': 'x = dict(1, 2)\n'}
    CONTRACT = Contract(
        qualname='dict',
        raises=dunder_propagation_claims(
            'mapping lookup, iteration, __hash__ or __eq__',
            native={"TypeError": 'too many positional arguments, non-iterable source/pair, or unhashable key or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('dict', 'new dictionary from mapping/iterable pairs and keyword entries; preserves insertion order', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#dict", 'CPython v3.14.4 Objects/dictobject.c::dict_update_common'),
        tested=True,
    )


@register_rule
class Filter(Rule):
    """The callable rule for ``filter``."""

    names = ('filter',)
    SAMPLES = {'filter': 'x = filter(None, [1])\n'}
    TWINS = {'filter': 'x = filter()\n'}
    CONTRACT = Contract(
        qualname='filter',
        raises=dunder_propagation_claims(
            'iterable __iter__ or __getitem__ setup',
            native={"TypeError": 'not two positional arguments or source is not iterable or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('filter', 'lazy filter iterator; only iter(iterable) runs now; predicate and element truth tests run during consumption', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#filter", 'CPython v3.14.4 Python/bltinmodule.c::filter_new'),
        tested=True,
    )


@register_rule
class Hasattr(Rule):
    """The callable rule for ``hasattr``."""

    names = ('hasattr',)
    SAMPLES = {'hasattr': 'x = hasattr(3, "real")\n'}
    TWINS = {'hasattr': 'x = hasattr()\n'}
    CONTRACT = Contract(
        qualname='hasattr',
        raises=dunder_propagation_claims(
            '__getattribute__, descriptor or __getattr__',
            native={"AttributeError": "never", "TypeError": 'wrong call shape, a non-string attribute name, or attribute lookup raises TypeError or the invoked hook raises TypeError'},
            tested_native=frozenset({"AttributeError", "TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('bool', 'True when attribute lookup succeeds; False only when lookup raises AttributeError (including in a descriptor)', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='AttributeError from any lookup stage is swallowed; all other hook exceptions propagate. This queries an explicit object, not caller frames.',
        ),
        source=(f"{_DOCS}#hasattr", 'CPython v3.14.4 Python/bltinmodule.c::builtin_hasattr_impl'),
        tested=True,
    )


@register_rule
class Getattr(Rule):
    """Attribute lookup, with an optional default for AttributeError."""

    names = ('getattr',)
    SAMPLES = {'getattr': 'x = getattr(3, "real")\n'}
    TWINS = {'getattr': 'x = getattr()\n'}
    CONTRACT = Contract(
        qualname='getattr',
        raises=tuple(RaiseClaim(exc,
            'attribute lookup raises AttributeError and no default was supplied, including an AttributeError '
            'from attaching context to that exception' if exc == 'AttributeError' else
            'wrong call shape, a non-string attribute name, or an uncaught TypeError from lookup or '
            'from attaching context to an AttributeError without a default' if exc == 'TypeError' else
            f'an uncaught {exc} from lookup or from attaching context to an AttributeError without a default',
            tested=True) for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('Any', 'the attribute value, or the supplied default when lookup raises AttributeError', tested=True),
        effects=Effects(
            mutates_receiver=True, may_execute_callback=True,
            notes='Lookup can invoke arbitrary descriptors or attribute hooks. Without a default, attaching '
                  'AttributeError context can invoke the exception\'s __setattr__ and mutate it, including when '
                  'it is the receiver; a default suppresses AttributeError.',
        ),
        source=(f'{_DOCS}#getattr', 'CPython v3.14.4 Python/bltinmodule.c::builtin_getattr',
                'CPython v3.14.4 Objects/object.c::PyObject_GetAttr, PyObject_GetOptionalAttr, _PyObject_SetAttributeErrorContext'),
        tested=True,
    )


@register_rule
class Hash(Rule):
    """The callable rule for ``hash``."""

    names = ('hash',)
    SAMPLES = {'hash': 'x = hash(3)\n'}
    TWINS = {'hash': 'x = hash()\n'}
    CONTRACT = Contract(
        qualname='hash',
        raises=dunder_propagation_claims(
            '__hash__',
            native={"TypeError": 'wrong call shape, unhashable argument, or non-int __hash__ result or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('int', 'Py_ssize_t-width hash; -1 is remapped to -2; equal numeric values have equal hashes; text/bytes hashes depend on PYTHONHASHSEED', tested=True),
        effects=Effects(
            may_execute_callback=True,
            clock_or_env=True,
            deterministic=False,
            notes='Native text/bytes hashing reads the per-process PYTHONHASHSEED secret; object identity hashes also depend on allocation. Never a process-independent deterministic recipe callable.',
        ),
        source=(f"{_DOCS}#hash", 'CPython v3.14.4 Python/bltinmodule.c::builtin_hash'),
        tested=True,
    )


@register_rule
class Hex(Rule):
    """The callable rule for ``hex``."""

    names = ('hex',)
    SAMPLES = {'hex': 'x = hex(3)\n'}
    TWINS = {'hex': 'x = hex()\n'}
    CONTRACT = Contract(
        qualname='hex',
        raises=dunder_propagation_claims(
            '__index__',
            native={"TypeError": 'wrong call shape, missing __index__, or a non-int __index__ result or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('str', 'base-16 lowercase spelling, optional minus followed by 0x', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#hex", 'CPython v3.14.4 Objects/abstract.c::PyNumber_ToBase'),
        tested=True,
    )


@register_rule
class Id(Rule):
    """The callable rule for ``id``."""

    names = ('id',)
    SAMPLES = {'id': 'x = id(3)\n'}
    TWINS = {'id': 'x = id()\n'}
    CONTRACT = Contract(
        qualname='id',
        raises=dunder_propagation_claims(
            'builtins.id audit hook',
            native={"TypeError": 'not exactly one positional argument or an audit hook raises TypeError or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('int', 'identity unique among simultaneously live objects and stable during object lifetime; CPython uses its memory address', tested=True),
        effects=Effects(
            may_execute_callback=True,
            clock_or_env=True,
            deterministic=False,
            notes='Native identity reads the object memory address (process/allocation dependent) and emits the builtins.id audit event; hooks can raise arbitrary exceptions.',
        ),
        source=(f"{_DOCS}#id", 'CPython v3.14.4 Python/bltinmodule.c::builtin_id_impl'),
        tested=True,
    )


@register_rule
class Issubclass(Rule):
    """The callable rule for ``issubclass``."""

    names = ('issubclass',)
    SAMPLES = {'issubclass': 'x = issubclass(bool, int)\n'}
    TWINS = {'issubclass': 'x = issubclass()\n'}
    CONTRACT = Contract(
        qualname='issubclass',
        raises=dunder_propagation_claims(
            'unsuppressed __subclasscheck__, checker-result truth conversion or class __bases__ lookup',
            native={"TypeError": 'wrong call shape, first argument not a class, or reached classinfo entry not a class/tuple/union or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('bool', 'class inheritance test; classinfo may be a type, tuple or union; short-circuits classinfo alternatives left to right', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#issubclass", 'CPython v3.14.4 Objects/abstract.c::PyObject_IsSubclass'),
        tested=True,
    )


@register_rule
class Iter(Rule):
    """The callable rule for ``iter``."""

    names = ('iter',)
    SAMPLES = {'iter': 'x = iter([1])\n'}
    TWINS = {'iter': 'x = iter()\n'}
    CONTRACT = Contract(
        qualname='iter',
        raises=dunder_propagation_claims(
            '__iter__',
            native={"TypeError": 'wrong call shape, one-argument source not iterable/bad __iter__ result, or two-argument first value not callable or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('Iterator[Any]', 'one argument: its iterator (__getitem__ sequence fallback); two arguments: lazy callable/sentinel iterator, with no call or comparison yet', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#iter", 'CPython v3.14.4 Python/bltinmodule.c::builtin_iter'),
        tested=True,
    )


@register_rule
class Map(Rule):
    """The callable rule for ``map``."""

    names = ('map',)
    SAMPLES = {'map': 'x = map(None, [1])\n'}
    TWINS = {'map': 'x = map()\n'}
    CONTRACT = Contract(
        qualname='map',
        raises=dunder_propagation_claims(
            'iterable __iter__ or __getitem__ setup, or strict.__bool__/__len__',
            native={"TypeError": 'fewer than two positional arguments, unknown keyword, or non-iterable source or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('map', 'lazy map iterator; acquires iterators now; function calls, __next__, and strict-length ValueError happen during consumption; strict keyword added in 3.14', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#map", 'CPython v3.14.4 Python/bltinmodule.c::map_new'),
        tested=True,
    )


@register_rule
class Memoryview(Rule):
    """The callable rule for ``memoryview``."""

    names = ('memoryview',)
    SAMPLES = {'memoryview': 'x = memoryview(b"x")\n'}
    TWINS = {'memoryview': 'x = memoryview()\n'}
    CONTRACT = Contract(
        qualname='memoryview',
        raises=dunder_propagation_claims(
            '__buffer__',
            native={"TypeError": 'wrong call shape, no buffer support, or __buffer__ does not return a memoryview or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('memoryview', 'view of the exported buffer, sharing storage; can prevent bytearray resizing until release; Python-defined __buffer__ supported since 3.12', tested=True),
        effects=Effects(
            may_execute_callback=True,
            mutates_receiver=True,
            notes='Exports and aliases the source buffer; native bytearray exports prevent resizing. Python __buffer__ may execute arbitrary code.',
        ),
        source=(f"{_DOCS}#memoryview", 'CPython v3.14.4 Objects/memoryobject.c::memoryview_impl'),
        tested=True,
    )


@register_rule
class Next(Rule):
    """The callable rule for ``next``."""

    names = ('next',)
    SAMPLES = {'next': 'x = next(iter([1]))\n'}
    TWINS = {'next': 'x = next()\n'}
    CONTRACT = Contract(
        qualname='next',
        raises=dunder_propagation_claims(
            '__next__',
            native={"TypeError": 'wrong call shape or first argument not an iterator or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('Any', 'next item; on StopIteration returns the supplied default, otherwise propagates exhaustion; advances iterator state now', tested=True),
        effects=Effects(
            may_execute_callback=True,
            mutates_receiver=True,
            notes='Native consumption mutates the iterator, including built-in iterators; user __next__ can have arbitrary effects.',
        ),
        source=(f"{_DOCS}#next", 'CPython v3.14.4 Python/bltinmodule.c::builtin_next'),
        tested=True,
    )


@register_rule
class Object(Rule):
    """The callable rule for ``object``."""

    names = ('object',)
    SAMPLES = {'object': 'x = object()\n'}
    TWINS = {'object': 'x = object(1)\n'}
    CONTRACT = Contract(
        qualname='object',
        raises=never_claims({"TypeError": 'any positional or keyword argument'}, tested=_ALL),
        may_raise_other=True,
        returns=ReturnSpec('object', 'fresh base object with no instance __dict__; takes no arguments', tested=True),
        effects=Effects(
            may_execute_callback=False,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#object", 'CPython v3.14.4 Objects/typeobject.c::object_new, object_init'),
        tested=True,
    )


@register_rule
class Oct(Rule):
    """The callable rule for ``oct``."""

    names = ('oct',)
    SAMPLES = {'oct': 'x = oct(3)\n'}
    TWINS = {'oct': 'x = oct()\n'}
    CONTRACT = Contract(
        qualname='oct',
        raises=dunder_propagation_claims(
            '__index__',
            native={"TypeError": 'wrong call shape, missing __index__, or a non-int __index__ result or the invoked hook raises TypeError'},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('str', 'base-8 spelling, optional minus followed by 0o', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#oct", 'CPython v3.14.4 Objects/abstract.c::PyNumber_ToBase'),
        tested=True,
    )


@register_rule
class Property(Rule):
    """The callable rule for ``property``."""

    names = ('property',)
    SAMPLES = {'property': 'x = property()\n'}
    TWINS = {'property': 'x = property(None, None, None, None, None)\n'}
    CONTRACT = Contract(
        qualname='property',
        raises=dunder_propagation_claims(
            'getter __doc__ attribute lookup',
            native={"AttributeError": "never", "TypeError": 'more than four arguments, bad keywords, or getter metadata lookup raises TypeError or the invoked hook raises TypeError'},
            tested_native=frozenset({"AttributeError", "TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('property', 'descriptor storing getter/setter/deleter; omitted/None doc copies getter __doc__, swallowing AttributeError; never calls the getter now', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Getter __doc__ lookup may execute code; AttributeError is suppressed. Descriptor access later is a different operation.',
        ),
        source=(f"{_DOCS}#property", 'CPython v3.14.4 Objects/descrobject.c::property_init_impl'),
        tested=True,
    )


@register_rule
class Slice(Rule):
    """The callable rule for ``slice``."""

    names = ('slice',)
    SAMPLES = {'slice': 'x = slice(3)\n'}
    TWINS = {'slice': 'x = slice()\n'}
    CONTRACT = Contract(
        qualname='slice',
        raises=never_claims({"TypeError": 'not one to three positional arguments'}, tested=_ALL),
        may_raise_other=True,
        returns=ReturnSpec('slice', 'stores start/stop/step unchanged (one argument sets stop); step zero and non-indexable components are accepted until use', tested=True),
        effects=Effects(
            may_execute_callback=False,
            notes='Native dispatch has no I/O, randomness or frame/environment dependency; any declared user hooks may have arbitrary effects.',
        ),
        source=(f"{_DOCS}#slice", 'CPython v3.14.4 Objects/sliceobject.c::slice_new'),
        tested=True,
    )


@register_rule
class Staticmethod(Rule):
    """The callable rule for ``staticmethod``."""

    names = ('staticmethod',)
    SAMPLES = {'staticmethod': 'x = staticmethod(3)\n'}
    TWINS = {'staticmethod': 'x = staticmethod()\n'}
    CONTRACT = Contract(
        qualname='staticmethod',
        raises=dunder_propagation_claims(
            'wrapped-object metadata attribute lookup',
            native={"AttributeError": "never", "TypeError": 'not exactly one positional argument or metadata lookup raises TypeError or the invoked hook raises TypeError'},
            tested_native=frozenset({"AttributeError", "TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('staticmethod', 'descriptor wrapping the single argument, which need not be callable; copies available __module__/__name__/__qualname__/__doc__', tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes='Copies metadata by attribute lookup; AttributeError is suppressed, other exceptions propagate. Wrapped callable is not invoked.',
        ),
        source=(f"{_DOCS}#staticmethod", 'CPython v3.14.4 Objects/funcobject.c::sm_init, functools_wraps'),
        tested=True,
    )


@register_rule
class Super(Rule):
    """The callable rule for ``super``."""

    names = ('super',)
    SAMPLES = {'super': 'x = super(int, 3)\n'}
    TWINS = {'super': 'x = super(1, 2, 3)\n'}
    CONTRACT = Contract(
        qualname='super',
        raises=dunder_propagation_claims(
            'proxy __class__ lookup',
            native={"AttributeError": "never", "TypeError": 'wrong call shape, first argument not a type, or second argument is not an instance/subtype (including proxy __class__) or the invoked hook raises TypeError'},
            tested_native=frozenset({"AttributeError", "TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec('super', 'MRO-search proxy; zero-argument form reads the current frame, first argument and __class__ cell; explicit arguments validate subtype relationship', tested=True),
        effects=Effects(
            may_execute_callback=True,
            clock_or_env=True,
            deterministic=False,
            notes='The union includes zero-argument caller-frame/__class__-cell reflection; explicit proxy __class__ lookup may execute code and suppresses AttributeError.',
        ),
        source=(f"{_DOCS}#super", 'CPython v3.14.4 Objects/typeobject.c::super_init_impl, supercheck'),
        tested=True,
    )
