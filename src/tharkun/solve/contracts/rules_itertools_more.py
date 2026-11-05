"""Contracts for the remaining ``itertools`` functions (standing batch 4, lane 1493).

Every function here calls ``iter()`` on its iterable arguments AT CONSTRUCTION (``combinations_with_replacement`` even drains
it into a tuple), so a non-iterable or a hostile ``__iter__`` fails at the call; a callable argument (``func``, ``predicate``) and
each element's ``__next__`` / ``__bool__`` / ``__add__`` run only when the result is consumed.  Both can raise any class, so every
claim here is a propagation claim.  Exact on CPython 3.14.4 (``tests/contracts/test_itertools_more.py``).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/itertools.html"

# name -> (sample, TypeError condition, ValueError note, returns)
_ROWS = {
    "zip_longest": ('x = list(itertools.zip_longest([1], [2, 3]))', "an argument is not iterable (at construction); an unknown keyword (only fillvalue is accepted)", "", "tuples padded with fillvalue (None) up to the longest input"),
    "cycle": ('x = list(itertools.islice(itertools.cycle([1, 2]), 3))', "the argument is not iterable (at construction); not exactly one argument", "", "the elements of the input forever (empty input: no elements)"),
    "repeat": ('x = list(itertools.repeat(1, 2))', "times is not an integer; not one or two arguments (the object itself is never inspected)", "", "the object, times times (forever without times; none for times <= 0)"),
    "compress": ('x = list(itertools.compress([1, 2], [1, 0]))', "an argument is not iterable (at construction); wrong argument count", "", "the data elements whose selector is truthy, stopping at the shorter input"),
    "tee": ('x = itertools.tee([1, 2])', "the argument is not iterable (at construction); n is not an integer; wrong argument count", "n < 0", "a tuple of n independent iterators"),
    "accumulate": ('x = list(itertools.accumulate([1, 2, 3]))', "the argument is not iterable (at construction); an unknown keyword; wrong argument count; or func / the default + rejects two elements", "", "running totals (or running func results); initial, when given, comes first"),
    "pairwise": ('x = list(itertools.pairwise([1, 2, 3]))', "the argument is not iterable (at construction); wrong argument count", "", "overlapping consecutive pairs; empty for fewer than two elements"),
    "batched": ('x = list(itertools.batched([1, 2, 3], 2))', "the argument is not iterable (at construction); n is not an integer; wrong argument count", "n < 1; strict=True and the last batch is short", "tuples of up to n elements"),
    "takewhile": ('x = list(itertools.takewhile(bool, [1, 0]))', "the iterable is not iterable (at construction); wrong argument count; the predicate / func is neither None (filterfalse only) nor callable (raised on consumption)", "", "elements while the predicate holds"),
    "dropwhile": ('x = list(itertools.dropwhile(bool, [1, 0]))', "the iterable is not iterable (at construction); wrong argument count; the predicate / func is neither None (filterfalse only) nor callable (raised on consumption)", "", "elements from the first one the predicate rejects"),
    "filterfalse": ('x = list(itertools.filterfalse(bool, [1, 0]))', "the iterable is not iterable (at construction); wrong argument count; the predicate / func is neither None (filterfalse only) nor callable (raised on consumption)", "", "elements the predicate rejects (None: falsy elements)"),
    "starmap": ('x = list(itertools.starmap(max, [(1, 2)]))', "the iterable is not iterable (at construction); an element is not iterable; wrong argument count; the predicate / func is neither None (filterfalse only) nor callable (raised on consumption)", "", "func(*element) for each element"),
    "combinations_with_replacement": ('x = list(itertools.combinations_with_replacement([1, 2], 2))', "the iterable is not iterable (at construction); r is not an integer; wrong argument count", "r < 0", "r-tuples in sorted order with repeated elements allowed"),
}


def _make(name, row):
    sample, type_error, value_error, result = row
    qual = "itertools." + name
    contract = Contract(
        qualname=qual,
        raises=dunder_propagation_claims("an iterable argument's __iter__/__next__, or a callable argument / element protocol, run at construction or consumption (user code)",
                                         native={"TypeError": type_error}, tested_native=frozenset({"TypeError"})),
        may_raise_other=True,
        returns=ReturnSpec("itertools." + name, result, tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=("ValueError: " + value_error) if value_error else "",
        source=(f"{_DOCS}#itertools.{name}", "CPython Modules/itertoolsmodule.c"),
        tested=True)
    cls = type("Itertools" + "".join(p.capitalize() for p in name.split("_")), (Rule,), {
        "names": (qual,), "SAMPLES": {qual: "import itertools\n" + sample + "\n"}, "TWINS": {}, "CONTRACT": contract, "__module__": __name__})
    return register_rule(cls)


__all__ = []
for _name, _row in _ROWS.items():
    _c = _make(_name, _row)
    globals()[_c.__name__] = _c
    __all__.append(_c.__name__)
