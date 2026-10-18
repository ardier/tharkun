"""Shared constants used across tharkun modules."""

from __future__ import annotations

# Names that are Python builtins or well-known stdlib modules — not free
# variables in a guard expression.  Used by condition_solver, guard_decomposer,
# guard_analyzer, and branch_solver to filter out non-input names.
BUILTIN_NAMES: frozenset[str] = frozenset({
    # builtins
    "len", "isinstance", "type", "range", "int", "str", "float",
    "bool", "list", "dict", "set", "tuple", "bytes", "print",
    "True", "False", "None", "abs", "min", "max", "sum",
    "any", "all", "sorted", "reversed", "enumerate", "zip",
    "map", "filter", "hasattr", "getattr", "setattr", "callable",
    "ord", "chr", "repr", "hex", "bin", "oct", "id", "hash",
    "round", "divmod", "pow", "super", "object", "property",
    "staticmethod", "classmethod", "frozenset", "bytearray",
    "memoryview", "complex", "slice", "open", "input",
    "iter", "next", "vars", "dir", "globals", "locals",
    # well-known stdlib modules used as bare names in guards
    "math", "re", "os", "sys", "json", "collections",
    "functools", "itertools", "operator", "typing",
})
