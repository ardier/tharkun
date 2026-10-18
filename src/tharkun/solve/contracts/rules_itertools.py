"""Contracts for ``itertools`` ([internal ref] batch 3).

The most important, easy-to-get-wrong fact this module records is that these
functions do NOT validate uniformly: some raise at CONSTRUCTION time (before any
iteration), some only when actually consumed, and among the construction-time
validators, some fully MATERIALIZE their input iterable right there (so a hostile
``__next__`` fails immediately) while others merely call ``__iter__`` once and defer
all pulling to consumption time. All three behaviors are represented here, confirmed
empirically (``tests/contracts/test_itertools_.py``) rather than assumed uniform:

* ``chain`` is fully lazy -- neither argument type nor iterability is checked until
  the chain is actually iterated.
* ``product``/``combinations``/``permutations`` MATERIALIZE their input(s) into a
  tuple at construction -- a hostile ``__next__`` (not just a hostile ``__iter__``)
  surfaces immediately, before the caller has asked for a single output value.
* ``groupby``/``islice`` validate their own arguments (and call ``__iter__`` once) at
  construction, but do not pull any elements from the input until consumed -- a
  hostile ``__iter__`` fails at construction, but a hostile ``__next__`` only fails
  once iteration actually starts.
* ``count`` takes no iterable at all; its own numeric-argument validation is eager.

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/itertools.html"


@register_rule
class ItertoolsCount(Rule):
    """The rule of ``itertools.count``."""

    SAMPLES = {'itertools.count': 'import itertools\nc = itertools.count()\n'}
    TWINS = {}

    names = ("itertools.count",)
    CONTRACT = Contract(
        qualname="itertools.count",
        raises=dunder_propagation_claims(
            "start's/step's __add__/__radd__ -- a USER method, so any exception it "
            "raises propagates uncaught, from the FIRST next() call already "
            "(confirmed empirically: the non-fast-path implementation for a step "
            "that is not already a plain int/float computes the next accumulator "
            "value on every call, including the first, rather than deferring "
            "arithmetic to the second)",
            native={
                "TypeError": (
                    "start or step fails count's own 'is this number-like' check at "
                    "CONSTRUCTION (PyNumber_Check: needs nb_index, nb_float, or "
                    "nb_int) -- an object with ONLY __add__ and none of those three "
                    "still fails this check and never gets as far as __add__ being "
                    "called at all"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: start's/step's __add__/__radd__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("itertools.count (an infinite iterator of int/float)", "start, start+step, start+2*step, ... -- NEVER terminates on its own", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "validated EAGERLY at construction -- count('a') raises before any "
            "iteration is attempted. SOUNDNESS FIX (found by the gatekeeper): an "
            "earlier draft used never_claims, missing that an object which DOES "
            "pass the number-like check (e.g. has __index__) but has a raising "
            "__add__/__radd__ propagates that exception from the FIRST next() call: "
            "count(0, X()) where X.__radd__ raises KeyError raises KeyError on "
            "next(c), not on construction and not deferred to a later call."
        ),
        source=(f"{_DOCS}#itertools.count", "CPython Modules/itertoolsmodule.c::count_new/count_next"),
        tested=True,
    )


@register_rule
class ItertoolsChain(Rule):
    """The rule of ``itertools.chain``."""

    SAMPLES = {'itertools.chain': 'import itertools\nx = list(itertools.chain([1], [2]))\n'}
    TWINS = {'itertools.chain': 'import itertools\nx = list(itertools.chain([1], [2], 0))\n'}

    names = ("itertools.chain",)
    CONTRACT = Contract(
        qualname="itertools.chain",
        raises=dunder_propagation_claims(
            "each argument's __iter__/__next__ IN TURN, ONLY as the chain is actually consumed",
            native={
                "TypeError": "an argument is not iterable -- raised lazily, only once iteration reaches that argument, NOT at construction",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: each argument's __iter__/__next__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("itertools.chain", "yields every element of the first argument, then the second, etc.", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="the ONE function in this module that validates NOTHING at construction -- chain(5) itself never raises; list(chain(5)) does.",
        source=(f"{_DOCS}#itertools.chain", "CPython Modules/itertoolsmodule.c::chain_next"),
        tested=True,
    )


@register_rule
class ItertoolsProduct(Rule):
    """The rule of ``itertools.product``."""

    SAMPLES = {'itertools.product': 'import itertools\nx = list(itertools.product([1], [2]))\n'}
    TWINS = {'itertools.product': 'import itertools\nx = list(itertools.product([1], [2], 0))\n'}

    names = ("itertools.product",)
    CONTRACT = Contract(
        qualname="itertools.product",
        raises=dunder_propagation_claims(
            "each argument's __iter__/__next__, called EAGERLY and FULLY at construction (every input is materialized into a tuple before product() returns)",
            native={
                "TypeError": "an argument is not iterable, or repeat has no __index__",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: repeat < 0
        returns=ReturnSpec("itertools.product", "the Cartesian product, as tuples, in the input iterables' order (rightmost advances fastest)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="materializes eagerly: product(BadNext()) raises at the product(...) call itself, not on first iteration -- unlike chain.",
        source=(f"{_DOCS}#itertools.product", "CPython Modules/itertoolsmodule.c::product_new"),
        tested=True,
    )


@register_rule
class ItertoolsCombinations(Rule):
    """The rule of ``itertools.combinations``."""

    SAMPLES = {'itertools.combinations': 'import itertools\nx = list(itertools.combinations([1, 2], 1))\n'}
    TWINS = {'itertools.combinations': 'import itertools\nx = list(itertools.combinations())\n'}

    names = ("itertools.combinations",)
    CONTRACT = Contract(
        qualname="itertools.combinations",
        raises=dunder_propagation_claims(
            "the iterable's __iter__/__next__, called EAGERLY and FULLY at construction (same materialization as product)",
            native={
                "TypeError": "the iterable is not iterable, or r has no __index__",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: r < 0
        returns=ReturnSpec("itertools.combinations", "r-length tuples, in sorted order, without repeated elements (by POSITION, not value)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(f"{_DOCS}#itertools.combinations", "CPython Modules/itertoolsmodule.c::combinations_new"),
        tested=True,
    )


@register_rule
class ItertoolsPermutations(Rule):
    """The rule of ``itertools.permutations``."""

    SAMPLES = {'itertools.permutations': 'import itertools\nx = list(itertools.permutations([1, 2]))\n'}
    TWINS = {'itertools.permutations': 'import itertools\nx = list(itertools.permutations())\n'}

    names = ("itertools.permutations",)
    CONTRACT = Contract(
        qualname="itertools.permutations",
        raises=dunder_propagation_claims(
            "the iterable's __iter__/__next__, called EAGERLY and FULLY at construction",
            native={
                "TypeError": "the iterable is not iterable, or r has no __index__ (message is 'Expected int as r', distinct from combinations'/product's wording)",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: r < 0
        returns=ReturnSpec("itertools.permutations", "r-length tuples (r defaults to the input's length) of all orderings, by position", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(f"{_DOCS}#itertools.permutations", "CPython Modules/itertoolsmodule.c::permutations_new"),
        tested=True,
    )


@register_rule
class ItertoolsIslice(Rule):
    """The rule of ``itertools.islice``."""

    SAMPLES = {'itertools.islice': 'import itertools\nx = list(itertools.islice([1, 2, 3], 2))\n'}
    TWINS = {'itertools.islice': 'import itertools\nx = list(itertools.islice())\n'}

    names = ("itertools.islice",)
    CONTRACT = Contract(
        qualname="itertools.islice",
        raises=dunder_propagation_claims(
            "the iterable's __iter__ ONLY at construction (obtained, not drained); its __next__ only once consumption actually starts",
            native={
                "TypeError": "the iterable is not iterable",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: a bad start/stop/step (negative, non-integer-valued, etc.)
        returns=ReturnSpec("itertools.islice", "islice(it, stop) or islice(it, start, stop[, step]) -- like a lazy slice over any iterable, not just a sequence", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="start/stop/step validation (must be None or a non-negative int) IS eager -- islice([1,2,3], 'x') raises at construction; only the underlying iterable's OWN elements are pulled lazily.",
        source=(f"{_DOCS}#itertools.islice", "CPython Modules/itertoolsmodule.c::islice_new"),
        tested=True,
    )


@register_rule
class ItertoolsGroupby(Rule):
    """The rule of ``itertools.groupby``."""

    SAMPLES = {'itertools.groupby': 'import itertools\nx = list(itertools.groupby([1, 1, 2]))\n'}
    TWINS = {'itertools.groupby': 'import itertools\nx = list(itertools.groupby())\n'}

    names = ("itertools.groupby",)
    CONTRACT = Contract(
        qualname="itertools.groupby",
        raises=dunder_propagation_claims(
            "the iterable's __iter__ ONLY at construction; its __next__, and the key function (if given), only once consumption starts",
            native={
                "TypeError": (
                    "the iterable is not iterable (checked EAGERLY, at construction); "
                    "or key is given and is not callable (checked LAZILY -- "
                    "groupby([1, 2], key=5) does NOT raise until consumed, since key "
                    "is only ever called once groupby actually reads an element)"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the iterable's __iter__/__next__ and key are arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("itertools.groupby", "(key, sub-iterator) pairs for consecutive RUNS sharing a key -- NOT a full grouping (an unsorted input yields one group per run, not per distinct key value)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="each sub-iterator shares state with groupby's own cursor -- advancing groupby before exhausting a previous sub-iterator invalidates it (a well-known gotcha, out of scope for the raise/return claims here but worth noting for a consumer).",
        source=(f"{_DOCS}#itertools.groupby", "CPython Modules/itertoolsmodule.c::groupby_new"),
        tested=True,
    )


__all__ = ["ItertoolsCount", "ItertoolsChain", "ItertoolsProduct", "ItertoolsCombinations", "ItertoolsPermutations", "ItertoolsIslice", "ItertoolsGroupby"]
