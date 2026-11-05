"""Which exception classes a call into a (non-contract) library module can raise: the census's table, not its guess.

The first implementation of the census (written by an AI lane, owner 1408) trusted library calls BY NAME: a call
``module.f(..)`` into ``math``/``random``/``re``/``io``/``array``/``string``/``os``/``typing`` was a site only if ``f`` happened to
be in a per-class list of names, and the AssertionError list was empty -- so ``typing.assert_never(x)`` (it raises
``AssertionError`` by definition) was "no site" and the program was proven TRUE (a wrong TRUE, found by the Opus sweep on main
28467f87).  The modules with a tested contract (heapq, bisect, collections, itertools, functools) never had this problem:
``exception_sites._CONTRACT_MODULES``.

Here every callable of the remaining modules is classified by the classes of the census (ZeroDivisionError, FloatingPointError,
AssertionError, KeyError, IndexError) it CAN raise, from CPython's documented behaviour (verified natively in
``tests/test_census_library_calls.py``).  A callable that is not in the table is UNKNOWN, and an unknown callable of a trusted
module is a site for every class: nothing is trusted by name.  Raising a class by running program code (a user ``__iter__``
a function calls back, ``__floor__`` of an argument) is not a library raise: that code is scanned as program code."""
from __future__ import annotations

Z, F, A, K, I = "ZeroDivisionError", "FloatingPointError", "AssertionError", "KeyError", "IndexError"
ALL = frozenset({Z, F, A, K, I})
NONE: frozenset = frozenset()

#: module -> {callable name (dotted below the module) -> classes it can raise}
LIBRARY_CALLS: dict[str, dict[str, frozenset]] = {
    "math": {
        **{n: NONE for n in (
            "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp",
            "exp2", "expm1", "log1p", "log2", "log10", "pow", "fabs", "floor", "ceil", "trunc", "copysign", "fmod", "remainder", "hypot",
            "dist", "fsum", "sumprod", "prod", "gcd", "lcm", "isqrt", "factorial", "comb", "perm", "degrees", "radians", "isfinite", "isinf",
            "isnan", "isclose", "ldexp", "frexp", "modf", "nextafter", "ulp", "erf", "erfc", "gamma", "lgamma", "cbrt", "fma")},
        "log": frozenset({Z}),                       # log(x, 1): ZeroDivisionError
    },
    "random": {
        **{n: NONE for n in ("seed", "getstate", "randrange", "randint", "uniform", "randbytes", "random",
                             "Random", "SystemRandom")},
        "getrandbits": frozenset({K, I}),            # kept as the census always had it (a negative count is ValueError)
        "binomialvariate": frozenset({Z, A}),        # an internal assert fires for a NaN probability (found by the Opus battery)
        "setstate": frozenset({K, I}),               # state[0] / state[1] of a dict or an empty sequence
        "choice": frozenset({K, I}), "choices": frozenset({Z, K, I}), "sample": frozenset({Z, K, I}),
        "shuffle": frozenset({K, I}),                # random.shuffle({"a": 1, "b": 2}) is KeyError(1): it indexes x[i] (the first battery's
        #                                              containers had <= 1 element and missed it; wrong TRUE h1)
        **{n: frozenset({Z}) for n in ("triangular", "expovariate", "paretovariate", "weibullvariate", "betavariate", "gammavariate",
                                       "vonmisesvariate", "lognormvariate", "normalvariate", "gauss")},
    },
    "re": {
        **{n: NONE for n in ("compile", "search", "match", "fullmatch", "split", "findall", "finditer", "escape", "purge")},
        "sub": frozenset({K, I}), "subn": frozenset({K, I}),     # a template naming a missing group
    },
    "io": {n: NONE for n in ("StringIO", "BytesIO")},
    "array": {"array": NONE},
    "string": {"capwords": NONE, "Template": NONE, "Formatter": NONE},
    "os": {n: NONE for n in ("getcwd", "getenv", "getpid", "cpu_count", "urandom")} | {
        "environ.copy": NONE, "environ.get": NONE, "environ.keys": NONE, "environ.items": NONE, "environ.values": NONE,
        "environ.setdefault": NONE, "environ.update": NONE, "environ.pop": frozenset({K}), "environ.popitem": frozenset({K}),
        "path.join": NONE, "path.dirname": NONE, "path.basename": NONE, "path.isdir": NONE, "path.isfile": NONE, "path.exists": NONE,
        "path.abspath": NONE, "path.normpath": NONE, "path.splitext": NONE, "path.split": NONE, "path.expanduser": NONE,
        "path.realpath": NONE, "path.relpath": NONE, "path.isabs": NONE,
    },
    "os.path": {n: NONE for n in ("join", "dirname", "basename", "isdir", "isfile", "exists", "abspath", "normpath", "splitext", "split",
                                  "expanduser", "realpath", "relpath", "isabs")},
    "typing": {
        "assert_never": frozenset({A}),              # raises AssertionError("Expected code to be unreachable...") by definition
        "get_type_hints": ALL,                       # evaluates annotation strings: arbitrary expressions
        "NamedTuple": frozenset({K, I}), "TypedDict": frozenset({K, I}),   # a type name / field spec used as a format or a mapping
        "NamedTupleMeta": frozenset({A, K, I}),
        **{n: NONE for n in (
            "assert_type", "cast", "get_origin", "get_args", "reveal_type", "overload", "final", "no_type_check", "runtime_checkable",
            "dataclass_transform", "is_typeddict", "NewType", "TypeVar", "ParamSpec", "TypeVarTuple",
            "get_overloads", "clear_overloads", "Any", "Union", "Optional", "List", "Dict", "Set", "FrozenSet", "Tuple", "Type",
            "Callable", "Iterable", "Iterator", "Sequence", "Mapping", "MutableMapping", "MutableSequence", "Collection", "Generic",
            "Protocol", "Literal", "ClassVar", "Final", "Annotated", "TypeAlias", "Self", "Never", "NoReturn", "LiteralString",
            "Awaitable", "Coroutine", "Generator", "AsyncIterator", "AsyncIterable", "Deque", "DefaultDict", "OrderedDict", "Counter",
            "ChainMap", "Pattern", "Match", "IO", "TextIO", "BinaryIO", "Text", "AnyStr", "Hashable", "Sized", "SupportsInt",
            "SupportsFloat", "SupportsAbs", "SupportsIndex", "TYPE_CHECKING", "TypeGuard", "TypeIs", "ReadOnly", "Required", "NotRequired", "Unpack", "Concatenate", "KeysView", "ItemsView", "ValuesView", "ContextManager", "AsyncContextManager", "AbstractSet", "MutableSet", "Reversible", "SupportsBytes", "SupportsComplex", "SupportsRound", "ByteString")},
    },
    "_sv_verifier": {n: NONE for n in ("nondet_int", "nondet_bool", "nondet_str", "nondet_float", "nondet_list", "nondet_dict",
                                       "nondet_float_range")},
}
#: attributes that are values, never called: referencing them is always fine
LIBRARY_CONSTANTS: dict[str, frozenset] = {
    "math": frozenset({"pi", "e", "tau", "inf", "nan"}),
    "re": frozenset({"IGNORECASE", "I", "MULTILINE", "M", "DOTALL", "S", "VERBOSE", "X", "ASCII", "A", "UNICODE", "U", "LOCALE", "L",
                     "NOFLAG", "DEBUG", "error", "Pattern", "Match", "RegexFlag"}),
    "string": frozenset({"digits", "ascii_letters", "ascii_lowercase", "ascii_uppercase", "hexdigits", "octdigits", "punctuation",
                         "printable", "whitespace"}),
    "os": frozenset({"sep", "pathsep", "linesep", "name", "curdir", "pardir", "environ", "path"}),
    "os.path": frozenset({"sep", "pathsep"}),
    "io": frozenset({"StringIO", "BytesIO"}),
}
LIBRARY_MODULES = frozenset(LIBRARY_CALLS) | frozenset(LIBRARY_CONSTANTS)


def library_may_raise(module: str, name: str) -> frozenset | None:
    """The census classes ``module.name(..)`` can raise, or ``None`` when the callable is not in the table (UNKNOWN: a site)."""
    return LIBRARY_CALLS.get(module, {}).get(name)


def is_library_constant(module: str, name: str) -> bool:
    return name in LIBRARY_CONSTANTS.get(module, ())


# -- methods of builtin and library objects ------------------------------------------------------------------------------
#: what the Opus battery observed on the instance methods of the builtin/library types programs build (CPython 3.14, a value grid over
#: every public method): method name -> census classes some such type's method of that name raised.  A method name is judged by
#: this table over EVERY known type (the receiver's type is not always known), and a method name that no known type has and the
#: program does not define is UNKNOWN: a site.  (The census used to judge a method by a short list of names and trust the rest.)
_METHOD_OBSERVED: dict[str, frozenset] = {
    "__getitem__": frozenset({I, K}), "__setitem__": frozenset({I, K}), "__delitem__": frozenset({I, K}),
    "end": frozenset({I}), "start": frozenset({I}), "span": frozenset({I}), "group": frozenset({I}), "groups": frozenset({I}),
    "groupdict": frozenset({I}), "expand": frozenset({I}),
    "pop": frozenset({I, K}), "popitem": frozenset({K}), "popleft": frozenset({I}), "remove": frozenset({K}), "insert": frozenset({I}),
    "move_to_end": frozenset({K}), "setdefault": frozenset({K}), "__missing__": frozenset({K}),   # ChainMap().__missing__(0) is KeyError(0)
    "substitute": frozenset({K}), "format": frozenset({I, K}), "format_map": frozenset({K}), "vformat": frozenset({I, K}),
    "get_value": frozenset({I, K}), "get_field": frozenset({I, K}), "__mod__": frozenset({K, Z}),
    "shuffle": frozenset({I, K}), "binomialvariate": frozenset({Z, A}), "choice": frozenset({I, K}), "choices": frozenset({I, K, Z}), "sample": frozenset({I, K, Z}),
    "expovariate": frozenset({Z}), "paretovariate": frozenset({Z}), "weibullvariate": frozenset({Z}), "setstate": frozenset({I, K}),
    "getrandbits": frozenset({I, K}), "sub": frozenset({I, K}), "subn": frozenset({I, K}),
    "__divmod__": frozenset({Z}), "__floordiv__": frozenset({Z}), "__truediv__": frozenset({Z}), "__rtruediv__": frozenset({Z}),
    "__rfloordiv__": frozenset({Z}), "__rmod__": frozenset({Z}), "__rdivmod__": frozenset({Z}), "__pow__": frozenset({Z}),
    "__rpow__": frozenset({Z}),
    **{n: frozenset({Z}) for n in ("triangular", "betavariate", "gammavariate", "vonmisesvariate", "lognormvariate", "normalvariate", "gauss")},
}


#: class-metadata attributes CPython creates lazily (not callables; their presence in ``dir`` depends on what ran before)
_CLASS_METADATA = frozenset({"__annotate_func__", "__annotations_cache__", "__static_attributes__", "__firstlineno__", "__annotate__",
                             "__annotations__"})


def _known_method_names() -> frozenset:
    """The public/dunder names of the types the battery exercises (CPython 3.14), the universe ``_METHOD_SAFE`` must classify."""
    import array
    import collections
    import io
    import random
    import re
    import string
    types = (object, dict, list, set, frozenset, str, bytes, bytearray, tuple, range, int, float, complex, bool, memoryview,
             collections.deque, collections.OrderedDict, collections.Counter, collections.defaultdict, collections.ChainMap,
             re.Pattern, re.Match, io.StringIO, io.BytesIO, array.array, string.Template, string.Formatter, random.Random,
             random.SystemRandom, type(iter([])), type({}.items()), type({}.keys()), type({}.values()))
    return frozenset(n for t in types for n in dir(t) if n not in _CLASS_METADATA)


#: method names the battery CALLED on every such type over a grid with multi-element containers of every builtin kind and saw raise none
#: of the census classes (``tests/test_census_completeness_battery.py`` re-runs that, and asserts this set plus ``_METHOD_OBSERVED`` is
#: the whole universe: a name CPython adds later is unclassified, hence a site, until the battery has seen it).  A name in neither
#: table is UNKNOWN, never "cannot raise".
_METHOD_SAFE: frozenset = frozenset({
    'VERSION', '_MutableMapping__marker', '__abs__', '__abstractmethods__', '__add__', '__alloc__', '__and__', '__bool__',
    '__buffer__', '__bytes__', '__ceil__', '__class__', '__class_getitem__', '__complex__', '__contains__', '__copy__',
    '__deepcopy__', '__del__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__float__',
    '__floor__', '__format__', '__ge__', '__getattribute__', '__getformat__', '__getnewargs__', '__getstate__', '__gt__',
    '__hash__', '__iadd__', '__iand__', '__imul__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__',
    '__ior__', '__isub__', '__iter__', '__ixor__', '__le__', '__len__', '__length_hint__', '__lshift__', '__lt__', '__module__',
    '__mul__', '__ne__', '__neg__', '__new__', '__next__', '__or__', '__pos__', '__radd__', '__rand__', '__reduce__',
    '__reduce_ex__', '__release_buffer__', '__repr__', '__reversed__', '__rlshift__', '__rmul__', '__ror__', '__round__',
    '__rrshift__', '__rshift__', '__rsub__', '__rxor__', '__setattr__', '__setstate__', '__sizeof__', '__slots__', '__str__',
    '__sub__', '__subclasshook__', '__trunc__', '__weakref__', '__xor__', '_abc_impl', '_checkClosed', '_checkReadable',
    '_checkSeekable', '_checkWritable', '_compile_pattern', '_from_flags', '_invalid', '_keep_positive', '_notimplemented',
    '_randbelow', '_randbelow_with_getrandbits', '_randbelow_without_getrandbits', '_vformat', 'add', 'append', 'appendleft',
    'as_integer_ratio', 'bit_count', 'bit_length', 'braceidpattern', 'buffer_info', 'byteswap', 'c_contiguous', 'capitalize',
    'casefold', 'cast', 'center', 'check_unused_args', 'clear', 'close', 'closed', 'conjugate', 'contiguous', 'convert_field',
    'copy', 'count', 'decode', 'default_factory', 'delimiter', 'denominator', 'detach', 'difference', 'difference_update',
    'discard', 'elements', 'encode', 'encoding', 'endpos', 'endswith', 'errors', 'expandtabs', 'extend', 'extendleft',
    'f_contiguous', 'fileno', 'find', 'findall', 'finditer', 'flags', 'flush', 'format_field', 'from_bytes', 'from_number',
    'frombytes', 'fromfile', 'fromhex', 'fromkeys', 'fromlist', 'fromunicode', 'fullmatch', 'get', 'get_identifiers',
    'getbuffer', 'getstate', 'getvalue', 'groupindex', 'hex', 'idpattern', 'imag', 'index', 'intersection',
    'intersection_update', 'is_integer', 'is_valid', 'isalnum', 'isalpha', 'isascii', 'isatty', 'isdecimal', 'isdigit',
    'isdisjoint', 'isidentifier', 'islower', 'isnumeric', 'isprintable', 'isspace', 'issubset', 'issuperset', 'istitle',
    'isupper', 'items', 'itemsize', 'join', 'keys', 'lastgroup', 'lastindex', 'line_buffering', 'ljust', 'lower', 'lstrip',
    'maketrans', 'mapping', 'match', 'maxlen', 'most_common', 'nbytes', 'ndim', 'new_child', 'newlines', 'numerator', 'obj',
    'parents', 'parse', 'partition', 'pattern', 'pos', 'randbytes', 'randint', 'random', 'randrange', 're', 'read', 'read1',
    'readable', 'readinto', 'readinto1', 'readline', 'readlines', 'readonly', 'real', 'regs', 'release', 'removeprefix',
    'removesuffix', 'replace', 'resize', 'reverse', 'rfind', 'rindex', 'rjust', 'rotate', 'rpartition', 'rsplit', 'rstrip',
    'safe_substitute', 'scanner', 'search', 'seed', 'seek', 'seekable', 'shape', 'sort', 'split', 'splitlines', 'startswith',
    'step', 'stop', 'strides', 'string', 'strip', 'suboffsets', 'subtract', 'swapcase', 'symmetric_difference',
    'symmetric_difference_update', 'tell', 'title', 'to_bytes', 'tobytes', 'tofile', 'tolist', 'toreadonly', 'total',
    'tounicode', 'translate', 'truncate', 'typecode', 'uniform', 'union', 'update', 'upper', 'values', 'writable', 'write',
    'writelines', 'zfill'
})
KNOWN_METHOD_NAMES = _METHOD_SAFE | frozenset(_METHOD_OBSERVED)


def method_may_raise(name: str) -> frozenset | None:
    """The census classes a method called ``name`` can raise on SOME builtin/library object; ``None`` when no known type has such a
    method (the receiver is unresolved and the name unknown: a site unless the program defines the method itself)."""
    if name in _METHOD_OBSERVED:
        return _METHOD_OBSERVED[name]
    return NONE if name in _METHOD_SAFE else None


#: per-type method observations (the same battery) for the receiver kinds ``FixedTypes`` can resolve
_TYPE_OBSERVED: dict[str, dict[str, frozenset]] = {
    "list": {"__getitem__": frozenset({I}), "__setitem__": frozenset({I}), "__delitem__": frozenset({I}), "pop": frozenset({I})},
    "dict": {"__getitem__": frozenset({K}), "__delitem__": frozenset({K}), "pop": frozenset({K}), "popitem": frozenset({K}),
             "setdefault": frozenset({K})},
    "set": {"remove": frozenset({K}), "pop": frozenset({K})},
    "str": {"__getitem__": frozenset({I}), "__mod__": frozenset({K}), "format": frozenset({I, K}), "format_map": frozenset({K})},
    "bytes": {"__getitem__": frozenset({I}), "__mod__": frozenset({K})},
    "tuple": {"__getitem__": frozenset({I})},
    "int": {n: frozenset({Z}) for n in ("__divmod__", "__floordiv__", "__mod__", "__truediv__", "__rtruediv__", "__rfloordiv__", "__rmod__",
                                        "__rdivmod__", "__pow__", "__rpow__")},
    "float": {n: frozenset({Z}) for n in ("__divmod__", "__floordiv__", "__mod__", "__truediv__", "__rtruediv__", "__rfloordiv__", "__rmod__",
                                          "__rdivmod__", "__pow__", "__rpow__")},
}
_TYPES = {"list": list, "dict": dict, "set": set, "str": str, "bytes": bytes, "tuple": tuple, "int": int, "float": float}


def type_method_may_raise(kind: str, name: str) -> frozenset | None:
    """The classes ``name`` can raise on a receiver known to be of builtin type ``kind``; ``None`` when that type has no such method."""
    t = _TYPES.get(kind)
    if t is None or not hasattr(t, name):
        return None
    return _TYPE_OBSERVED.get(kind, {}).get(name, NONE)
