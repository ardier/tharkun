"""Typed value-term backends, separate from Python identity and from heap refs.

Recipes capture immutable value provenance. Graphs are built only for a query.
The string laws are SMT-admitted in benchmarks/absint_equivalences/prove_rules.py.
Only the string plugin is installed; other builtin theories remain separate slices.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
import time
from .errors import Unsupported


@dataclass(frozen=True, eq=False)
class Recipe:
    kind: str
    op: str
    args: tuple = ()
    data: object = ""


def literal(value):
    kind = type(value)
    if kind is bool:
        return Recipe("bool", "literal", data=str(value))
    if kind is int:
        return Recipe("int", "literal", data=hex(value))
    if kind is float:
        return Recipe("float", "literal", data=struct.pack(">d", value).hex())
    if kind is str:
        # JSON escapes collide for surrogate-pair and single-astral strings.
        return Recipe("str", "literal", data=value)
    if kind is bytes:
        return Recipe("bytes", "literal", data=value.hex())
    if value is None:
        return Recipe("NoneType", "literal")
    raise TypeError("only exact builtin scalar literals have value terms")


@dataclass(frozen=True)
class Node:
    kind: str
    op: str
    args: tuple[int, ...]
    data: object


class Congruence:
    """Hash-consing and union-find congruence closure, with a per-query deadline."""

    def __init__(self, deadline=None):
        self.deadline = time.process_time() + 10 if deadline is None else deadline
        self.nodes = []
        self.parents = []
        self.memo = {}
        self.merges = 0

    def tick(self):
        if time.process_time() > self.deadline:
            raise Unsupported("value equality query time budget exhausted")

    def find(self, value):
        while self.parents[value] != value:
            self.parents[value] = self.parents[self.parents[value]]
            value = self.parents[value]
        return value

    def intern(self, kind, op, args=(), data=""):
        self.tick()
        node = Node(kind, op, tuple(self.find(a) for a in args), data)
        if node in self.memo:
            return self.find(self.memo[node])
        value = len(self.nodes)
        self.nodes.append(node)
        self.parents.append(value)
        self.memo[node] = value
        return value

    def union(self, left, right):
        if self.nodes[left].kind != self.nodes[right].kind:
            raise ValueError("value congruence never merges different builtin kinds")
        self._union(left, right)
        changed = True
        while changed:
            self.tick()
            changed = False
            memo = {}
            for index, node in enumerate(self.nodes):
                signature = Node(
                    node.kind,
                    node.op,
                    tuple(self.find(a) for a in node.args),
                    node.data,
                )
                if signature in memo:
                    changed = self._union(index, memo[signature]) or changed
                else:
                    memo[signature] = index
            self.memo = memo

    def _union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return False
        self.parents[max(left, right)] = min(left, right)
        self.merges += 1
        return True

    def equivalent(self, left, right):
        self.tick()
        return self.find(left) == self.find(right)


def herbrand_join(left, right, deadline=None):
    """Intersection of must-equality partitions over the shared live positions.

    Pairing the two input class numbers is the Herbrand/GVN join. This is a
    finite partition domain alongside, not a replacement for, the affine hull.
    The equality engine itself does not implement a state or loop join.
    """
    groups = {}
    result = {}
    for position in left.keys() & right.keys():
        if deadline is not None and time.process_time() > deadline:
            raise Unsupported("Herbrand equality join time budget exhausted")
        key = (left[position], right[position])
        if key not in groups:
            groups[key] = len(groups)
        result[position] = groups[key]
    return result


class Theories:
    """Admitted ordered str concatenation laws behind the typed term interface."""

    def __init__(self, backend):
        self.backend = backend
        self.materialized = {}

    def _normalize(self, node):
        b = self.backend
        n = b.nodes[node]
        if n.op == "join" and b.equivalent(*n.args):
            return n.args[0]
        if n.kind == "str" and n.op == "Add":
            parts = []
            empty = literal("").data
            for a in n.args:
                b.tick()
                child = b.nodes[a]
                if child.op == "concat":
                    parts.extend(child.args)
                elif not (child.op == "literal" and child.data == empty):
                    parts.append(a)
            if not parts:
                return b.intern("str", "literal", data=empty)
            if len(parts) == 1:
                return parts[0]
            return b.intern("str", "concat", parts)
        return node

    def build(self, recipe):
        pending = [(recipe, False)]
        while pending:
            self.backend.tick()
            current, ready = pending.pop()
            if current in self.materialized:
                continue
            if not ready:
                pending.append((current, True))
                pending.extend((arg, False) for arg in reversed(current.args))
            else:
                node = self.backend.intern(
                    current.kind,
                    current.op,
                    tuple(self.materialized[a] for a in current.args),
                    current.data,
                )
                self.materialized[current] = self._normalize(node)
        return self.materialized[recipe]

    def proves_equal(self, left, right):
        if left.kind != "str" or right.kind != "str":
            return False
        return self.backend.equivalent(self.build(left), self.build(right))
