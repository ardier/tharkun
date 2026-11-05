"""Contracts for ``os.path`` (posixpath / genericpath), standing batch 8, lane 1493.

Every function first reads its path argument through ``os.fspath``: a ``str`` or ``bytes`` (a subclass is kept as is) is returned
unchanged, anything else must have ``__fspath__``, which is user code and can raise any class; any other type is a ``TypeError``.
The string functions are then pure (no cwd, no environment, no file).  The file queries (``exists``, ``isfile``, ``isdir``,
``islink``, ``lexists``) call ``os.stat`` / ``lstat``, turn an ``OSError`` or ``ValueError`` (an embedded NUL) into ``False`` and let
everything else out, so they never raise those two but depend on the file system (``io``, not deterministic).  ``abspath``, ``realpath``,
``relpath`` (the cwd), ``expanduser`` / ``expandvars`` (the environment) and the size / time queries are NOT modelled here.
Exact on CPython 3.14.4 (``tests/contracts/test_os_path_.py``).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/os.path.html#os.path."
_SRC = "CPython Lib/posixpath.py, Lib/genericpath.py"
_FSPATH = "the path's __fspath__ (a user method), read by os.fspath"
_TE = "an argument is not a str, bytes or os.PathLike (None, an int included); str and bytes are mixed; wrong argument count or an unknown keyword"

# name -> (sample, result, TypeError text, other-class note)
_PURE = {
    "join": ('os.path.join("a", "b")', "object", _TE, "Exact builtin paths return str/bytes; accepted subclasses may override addition or slicing to return any object.", ),
    "basename": ('os.path.basename("a/b")', "str | bytes", _TE, ""),
    "dirname": ('os.path.dirname("a/b")', "str | bytes", _TE, ""),
    "split": ('os.path.split("a/b")', "tuple", _TE, ""),
    "splitext": ('os.path.splitext("a.b")', "tuple", _TE, ""),
    "splitdrive": ('os.path.splitdrive("a/b")', "tuple", _TE, ""),
    "splitroot": ('os.path.splitroot("/a/b")', "tuple", _TE, ""),
    "normpath": ('os.path.normpath("a//b/../c")', "str | bytes", _TE, ""),
    "normcase": ('os.path.normcase("a/B")', "str | bytes", _TE, ""),
    "isabs": ('os.path.isabs("/a")', "bool", _TE, ""),
    "commonprefix": ('os.path.commonprefix(["ab", "ac"])', "str | list | tuple", "the argument is not a subscriptable sequence of paths or of lists / tuples; an element is not a path", ""),
    "commonpath": ('os.path.commonpath(["/a/b", "/a/c"])', "str | bytes", "the argument is not iterable; an element is not a path; str and bytes are mixed", "ValueError: an empty sequence, or absolute and relative paths mixed"),
}
_QUERY = {n: (f'os.path.{n}("/nonexistent_tharkun")', "bool") for n in ("exists", "isfile", "isdir", "islink", "lexists")}


def _make(name, contract, sample):
    qual = "os.path." + name
    cls = type("OsPath" + name.capitalize(), (Rule,), {
        "names": (qual,), "SAMPLES": {qual: "import os\nx = " + sample + "\n"}, "TWINS": {qual: "import os\nx = os.path." + name + "()\n"},
        "CONTRACT": contract, "__module__": __name__})
    return register_rule(cls)


_RULES = []
for _n, (_sample, _result, _te, _note) in _PURE.items():
    _RULES.append(_make(_n, Contract(
        qualname="os.path." + _n,
        raises=dunder_propagation_claims(_FSPATH if _n not in ("commonprefix", "commonpath") else
                                         _FSPATH + ", or the sequence's / an element's protocol methods (user code)",
                                         native={"TypeError": _te}, tested_native=frozenset({"TypeError"})),
        may_raise_other=True,
        returns=ReturnSpec(_result, tested=True), effects=Effects(deterministic=True, may_execute_callback=True),
        notes=_note, source=(_DOCS + _n, _SRC), tested=True), _sample))
def _query_native(name):
    native = {"TypeError": "the argument is not a str, bytes or os.PathLike (None, a float; for exists / isfile / isdir an integer is a "
                           "file descriptor, for islink / lexists it is a TypeError); wrong argument count"}
    if name == "islink":
        native["AttributeError"] = "never (posixpath.islink swallows an AttributeError, a __fspath__ hook's included)"
    return native


for _n, (_sample, _result) in _QUERY.items():
    _RULES.append(_make(_n, Contract(
        qualname="os.path." + _n,
        raises=dunder_propagation_claims(_FSPATH, native=_query_native(_n), tested_native=frozenset(_query_native(_n))),
        may_raise_other=True,
        returns=ReturnSpec(_result, "False for a missing file and for any OSError or ValueError (an embedded NUL) from stat; an integer is a file descriptor", tested=True),
        effects=Effects(io=True, deterministic=False, may_execute_callback=True, notes="asks the file system"),
        source=(_DOCS + _n, _SRC), tested=True), _sample))

__all__ = [c.__name__ for c in _RULES]
for _c in _RULES:
    globals()[_c.__name__] = _c
