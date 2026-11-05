"""Python string equality/concatenation as sequences of literal code points.

The 32-bit alphabet contains every Python code point, including lone surrogates.
Draws deliberately overapproximate string.printable (and Unicode): UNSAT proves
the property for the native domain too. SAT is left to native search, never
decoded into a draw or mistaken for a proof. No StringVal escape decoding or
solver string-alphabet limit participates in this representation.
"""
import z3


EDGE_METHODS = frozenset({"startswith", "endswith"})


def has_edge_calls(tree):
    import ast
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr in EDGE_METHODS for n in ast.walk(tree))


SORT = z3.SeqSort(z3.BitVecSort(32))


def is_string(value):
    return isinstance(value, z3.SeqRef) and value.sort() == SORT


def variable(name):
    return z3.Const(name, SORT)


def literal(text):
    units = [z3.Unit(z3.BitVecVal(ord(char), 32)) for char in text]
    return z3.Concat(*units) if len(units) > 1 else units[0] if units else z3.Empty(SORT)


def edge(receiver, candidates, start, end, *, suffix=False, check_work=lambda: None):
    """``(result, continuation)`` for Python's prefix/suffix test.

    Bounds are already int terms or None. Candidates are evaluated values;
    tuple entries are type checked in order, only until the first match. A bad
    entry raises TypeError unless an earlier entry matched. No user hooks run.
    ``continuation`` excludes exactly those TypeError paths, not a length bound.
    """
    size = z3.Length(receiver)
    start = 0 if start is None else start
    end = size if end is None else end
    lo = z3.If(start < 0, z3.If(start + size < 0, 0, start + size), start)
    hi = z3.If(end < 0, z3.If(end + size < 0, 0, end + size), z3.If(end > size, size, end))
    # CPython does not clamp a positive start to len: even an empty prefix
    # fails when start is beyond the original string, or start exceeds end.
    valid = z3.And(lo <= size, lo <= hi)
    span = z3.SubSeq(receiver, lo, hi - lo)
    matched = z3.BoolVal(False)
    for candidate in candidates if type(candidates) is tuple else (candidates,):
        check_work()
        if not is_string(candidate):
            return matched, matched
        test = z3.SuffixOf(candidate, span) if suffix else z3.PrefixOf(candidate, span)
        matched = z3.Or(matched, z3.And(valid, test))
    return matched, z3.BoolVal(True)
