"""Recording proxy object for untyped parameter inference.

When the condition solver can't infer a parameter's type, it falls back to
``SimpleNamespace``.  This module provides a smarter alternative: a proxy
object that records all attribute accesses and method calls during a trial
execution, then infers the real type from the recorded usage patterns.

Usage::

    from tharkun.solve.frontend.proxy import RecordingProxy, infer_type_from_usage

    proxy = RecordingProxy()
    # Pass proxy as an argument and let the function use it...
    try:
        target_function(proxy)
    except Exception:
        pass

    type_name = infer_type_from_usage(proxy)
    # e.g. "str", "list", "dict", "object"
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Method → type mapping (what methods imply what types)
# ---------------------------------------------------------------------------

_STR_METHODS = frozenset({
    "startswith", "endswith", "upper", "lower", "strip", "lstrip", "rstrip",
    "split", "rsplit", "join", "replace", "find", "rfind", "index", "rindex",
    "count", "encode", "decode", "format", "isdigit", "isalpha", "isalnum",
    "isupper", "islower", "isspace", "title", "capitalize", "swapcase",
    "center", "ljust", "rjust", "zfill", "expandtabs", "partition",
    "rpartition", "removeprefix", "removesuffix",
})

_LIST_METHODS = frozenset({
    "append", "extend", "insert", "remove", "pop", "clear", "sort",
    "reverse", "copy",
})

_DICT_METHODS = frozenset({
    "keys", "values", "items", "get", "setdefault", "update", "pop",
    "popitem", "clear", "copy", "fromkeys",
})

_SET_METHODS = frozenset({
    "add", "discard", "remove", "pop", "clear", "union", "intersection",
    "difference", "symmetric_difference", "issubset", "issuperset",
    "isdisjoint", "update", "intersection_update", "difference_update",
    "symmetric_difference_update",
})


class RecordingProxy:
    """A proxy object that records attribute accesses and method calls.

    Attributes:
        _accessed_attrs:  Set of attribute names accessed via __getattr__.
        _called_methods:  Set of method names that were called.
        _subscript_used:  Whether __getitem__ was called.
        _iter_used:       Whether __iter__ was called.
        _len_used:        Whether __len__ was called.
        _contains_used:   Whether __contains__ was called.
        _bool_used:       Whether __bool__ was called.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_accessed_attrs", set())
        object.__setattr__(self, "_called_methods", set())
        object.__setattr__(self, "_subscript_used", False)
        object.__setattr__(self, "_iter_used", False)
        object.__setattr__(self, "_len_used", False)
        object.__setattr__(self, "_contains_used", False)
        object.__setattr__(self, "_bool_used", False)
        object.__setattr__(self, "_children", {})

    def __getattr__(self, name: str) -> "RecordingProxy":
        if name.startswith("_"):
            raise AttributeError(name)
        attrs = object.__getattribute__(self, "_accessed_attrs")
        attrs.add(name)
        children = object.__getattribute__(self, "_children")
        if name not in children:
            child = RecordingProxy()
            children[name] = child
        return children[name]

    def __call__(self, *args: Any, **kwargs: Any) -> "RecordingProxy":
        # When a proxy attribute is called, it means the parent attr was a method
        return RecordingProxy()

    def __getitem__(self, key: Any) -> "RecordingProxy":
        object.__setattr__(self, "_subscript_used", True)
        return RecordingProxy()

    def __setitem__(self, key: Any, value: Any) -> None:
        object.__setattr__(self, "_subscript_used", True)

    def __iter__(self):
        object.__setattr__(self, "_iter_used", True)
        return iter([])

    def __len__(self) -> int:
        object.__setattr__(self, "_len_used", True)
        return 0

    def __contains__(self, item: Any) -> bool:
        object.__setattr__(self, "_contains_used", True)
        return False

    def __bool__(self) -> bool:
        object.__setattr__(self, "_bool_used", True)
        return True

    def __str__(self) -> str:
        return "<RecordingProxy>"

    def __repr__(self) -> str:
        return "<RecordingProxy>"

    def __eq__(self, other: Any) -> bool:
        return False

    def __ne__(self, other: Any) -> bool:
        return True

    def __lt__(self, other: Any) -> bool:
        return False

    def __le__(self, other: Any) -> bool:
        return False

    def __gt__(self, other: Any) -> bool:
        return False

    def __ge__(self, other: Any) -> bool:
        return False

    def __hash__(self) -> int:
        return id(self)

    def __add__(self, other: Any) -> "RecordingProxy":
        return RecordingProxy()

    def __radd__(self, other: Any) -> "RecordingProxy":
        return RecordingProxy()

    def __mul__(self, other: Any) -> "RecordingProxy":
        return RecordingProxy()

    def __rmul__(self, other: Any) -> "RecordingProxy":
        return RecordingProxy()


def infer_type_from_usage(proxy: RecordingProxy) -> str:
    """Infer a Python type from the recorded usage of a RecordingProxy.

    Examines which attributes, methods, and dunder protocols were used
    and maps them to the most likely Python type.

    Returns:
        A type name string (e.g. "str", "list", "dict", "object").
    """
    attrs = object.__getattribute__(proxy, "_accessed_attrs")
    subscript = object.__getattribute__(proxy, "_subscript_used")
    iter_used = object.__getattribute__(proxy, "_iter_used")
    len_used = object.__getattribute__(proxy, "_len_used")
    contains = object.__getattribute__(proxy, "_contains_used")

    # Check method overlap with known types
    str_overlap = attrs & _STR_METHODS
    list_overlap = attrs & _LIST_METHODS
    dict_overlap = attrs & _DICT_METHODS
    set_overlap = attrs & _SET_METHODS

    # Score each type
    scores: dict[str, int] = {
        "str": len(str_overlap) * 3,
        "list": len(list_overlap) * 3,
        "dict": len(dict_overlap) * 3,
        "set": len(set_overlap) * 3,
    }

    # Protocol-based signals
    if subscript:
        scores["list"] += 2
        scores["dict"] += 2
    if iter_used:
        scores["list"] += 1
        scores["str"] += 1
    if len_used:
        scores["list"] += 1
        scores["str"] += 1
        scores["dict"] += 1
    if contains:
        scores["set"] += 1
        scores["list"] += 1
        scores["dict"] += 1

    # If "get" is in attrs and subscript, likely dict
    if "get" in attrs:
        scores["dict"] += 3

    best_type = max(scores, key=scores.get)
    if scores[best_type] > 0:
        return best_type

    # No matches to builtin types — it's a generic object
    if attrs:
        return "object"

    return "Any"


def get_accessed_attrs(proxy: RecordingProxy) -> set[str]:
    """Return the set of attribute names accessed on the proxy."""
    return set(object.__getattribute__(proxy, "_accessed_attrs"))
