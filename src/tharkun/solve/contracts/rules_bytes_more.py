"""Contracts for the str-like methods of ``bytes`` (standing batch 2, lane 1493).

Every buffer-protocol argument (``sub``, ``prefix``, ``chars``, ``table``, ...) is read through ``PyObject_GetBuffer``, which in
3.12+ calls a Python class's ``__buffer__``: so an object whose ``__buffer__`` raises propagates ANY exception out of the call
(``b"a".find(X())``), and every ``start``/``end``/``maxsplit``/``count``/``width`` argument goes through ``__index__``.  A method with
such an argument therefore carries propagation claims for every property class; a method without one (``upper``, ``isdigit``...)
raises only its call-shape ``TypeError``.  Exact on CPython 3.14.4 (``tests/contracts/test_bytes_more.py`` sweeps every row).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/stdtypes.html#bytes"

# name -> (result, sample call on b"ab", native TypeError condition, other-class note or "", takes buffer/index args)
_ROWS = {
    "startswith": ("bool", 'x = b"ab".startswith(b"a")', "prefix is not bytes-like or a tuple of bytes-like; wrong argument count or keyword", "", True),
    "endswith": ("bool", 'x = b"ab".endswith(b"b")', "suffix is not bytes-like or a tuple of bytes-like; wrong argument count or keyword", "", True),
    "find": ("int", 'x = b"ab".find(b"b")', "sub is neither int nor bytes-like; wrong argument count or keyword", "-1 if absent; ValueError for an int sub outside range(256)", True),
    "rfind": ("int", 'x = b"ab".rfind(b"b")', "sub is neither int nor bytes-like; wrong argument count or keyword", "-1 if absent; ValueError for an int sub outside range(256)", True),
    "rindex": ("int", 'x = b"ab".rindex(b"b")', "sub is neither int nor bytes-like; wrong argument count or keyword", "ValueError if absent or an int sub outside range(256)", True),
    "count": ("int", 'x = b"ab".count(b"b")', "sub is neither int nor bytes-like; wrong argument count or keyword", "ValueError for an int sub outside range(256)", True),
    "split": ("list[bytes]", 'x = b"a,b".split(b",")', "sep is neither None nor bytes-like; wrong argument count", "ValueError for an empty sep", True),
    "rsplit": ("list[bytes]", 'x = b"a,b".rsplit(b",")', "sep is neither None nor bytes-like; wrong argument count", "ValueError for an empty sep", True),
    "strip": ("bytes", 'x = b"ab".strip(b"a")', "chars is neither None nor bytes-like; wrong argument count or keyword", "", True),
    "lstrip": ("bytes", 'x = b"ab".lstrip(b"a")', "chars is neither None nor bytes-like; wrong argument count or keyword", "", True),
    "rstrip": ("bytes", 'x = b"ab".rstrip(b"a")', "chars is neither None nor bytes-like; wrong argument count or keyword", "", True),
    "replace": ("bytes", 'x = b"ab".replace(b"a", b"c")', "old or new is not bytes-like; wrong argument count or keyword", "", True),
    "translate": ("bytes", 'x = b"ab".translate(None, b"a")', "table is neither None nor bytes-like; delete is not bytes-like; wrong argument count", "ValueError if the table is not 256 bytes long", True),
    "partition": ("tuple[bytes, bytes, bytes]", 'x = b"a,b".partition(b",")', "sep is not bytes-like; wrong argument count or keyword", "ValueError for an empty sep", True),
    "rpartition": ("tuple[bytes, bytes, bytes]", 'x = b"a,b".rpartition(b",")', "sep is not bytes-like; wrong argument count or keyword", "ValueError for an empty sep", True),
    "removeprefix": ("bytes", 'x = b"ab".removeprefix(b"a")', "prefix is not bytes-like; wrong argument count or keyword", "", True),
    "removesuffix": ("bytes", 'x = b"ab".removesuffix(b"b")', "suffix is not bytes-like; wrong argument count or keyword", "", True),
    "zfill": ("bytes", 'x = b"ab".zfill(4)', "width has no __index__; wrong argument count or keyword", "", True),
    "center": ("bytes", 'x = b"ab".center(4)', "width has no __index__; fillchar is not a length-1 bytes; wrong argument count or keyword", "", True),
    "ljust": ("bytes", 'x = b"ab".ljust(4)', "width has no __index__; fillchar is not a length-1 bytes; wrong argument count or keyword", "", True),
    "rjust": ("bytes", 'x = b"ab".rjust(4)', "width has no __index__; fillchar is not a length-1 bytes; wrong argument count or keyword", "", True),
    "expandtabs": ("bytes", 'x = b"a\\tb".expandtabs(4)', "tabsize has no __index__; wrong argument count", "OverflowError for an oversized tabsize or result", True),
    "splitlines": ("list[bytes]", 'x = b"a\\nb".splitlines()', "wrong argument count", "keepends goes through its __bool__ (any exception propagates)", True),
    "__contains__": ("bool", 'x = b"ab".__contains__(b"a")', "operand is neither int nor bytes-like", "ValueError for an int outside range(256)", True),
}
for _n in ("upper", "lower", "capitalize", "swapcase", "title", "isalnum", "isalpha", "isascii", "isdigit", "islower", "isspace", "istitle", "isupper"):
    _ROWS[_n] = ("bool" if _n.startswith("is") else "bytes", f'x = b"ab".{_n}()', "wrong argument count or keyword", "", False)

_NO_RAISE = frozenset(ALL_PROPERTY_CLASSES)


def _contract(name, result, type_error, note, args):
    native = {"TypeError": type_error}
    if args:
        raises = dunder_propagation_claims("a buffer argument's __buffer__ or an integer argument's __index__ (user methods)",
                                           native=native, tested_native=frozenset(native))
    else:
        raises = never_claims(native=native, tested=_NO_RAISE)
    return Contract(
        qualname="bytes." + name, raises=raises, may_raise_other=args or bool(note),
        returns=ReturnSpec(result, tested=True), effects=Effects(deterministic=not args, may_execute_callback=args),
        notes=note, source=(_DOCS + "." + name, "CPython Objects/bytesobject.c, Objects/bytes_methods.c"), tested=True)


def _make(name, row):
    result, sample, type_error, note, args = row
    qual = "bytes." + name
    cls = type("Bytes" + name.strip("_").capitalize(), (Rule,), {
        "names": (qual,), "SAMPLES": {qual: sample + "\n"}, "TWINS": {},
        "CONTRACT": _contract(name, result, type_error, note, args), "__module__": __name__})
    return register_rule(cls)


__all__ = []
for _name, _row in _ROWS.items():
    _c = _make(_name, _row)
    globals()[_c.__name__] = _c
    __all__.append(_c.__name__)
