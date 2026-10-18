"""Contracts for ``functools`` ([internal ref] batch 3): ``reduce``, ``partial``,
``lru_cache``/``cache``.

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/functools.html"


@register_rule
class FunctoolsReduce(Rule):
    """The rule of ``functools.reduce``."""

    SAMPLES = {'functools.reduce': 'import functools\nx = functools.reduce(max, [1, 2])\n'}
    TWINS = {'functools.reduce': 'import functools\nx = functools.reduce()\n'}

    names = ("functools.reduce",)
    CONTRACT = Contract(
        qualname="functools.reduce",
        raises=dunder_propagation_claims(
            "function(accumulator, element), called once per element after the "
            "first (or every element, if initial is given); the iterable's own "
            "__iter__/__next__",
            native={
                "TypeError": "the iterable is empty AND no initial value was given -- the classic 'reduce() of empty iterable with no initial value'",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: function/the iterable's __iter__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("whatever function returns", "initial (unchanged) for an empty iterable WITH initial given -- function is never even called", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(f"{_DOCS}#functools.reduce", "CPython Modules/_functoolsmodule.c::functools_reduce"),
        tested=True,
    )


@register_rule
class FunctoolsPartial(Rule):
    """The rule of ``functools.partial``."""

    SAMPLES = {'functools.partial': 'import functools\nf = functools.partial(abs, -1)\n'}
    TWINS = {'functools.partial': 'import functools\nf = functools.partial()\n'}

    names = ("functools.partial",)
    CONTRACT = Contract(
        qualname="functools.partial",
        raises=dunder_propagation_claims(
            "the underlying function, called with the bound args prepended to "
            "whatever the partial object is called with -- partial(f, *a, **kw)(*more, **morekw) "
            "is exactly f(*a, *more, **kw, **morekw)",
            native={},
            tested_native=frozenset(),
        ),
        may_raise_other=True,  # REVIEW FIX: func is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec(
            "a callable (functools.partial instance)",
            "CALLING it delegates to func's own arity/type checking -- func(1)(2, 3) with too many args raises TypeError from func itself, not from partial's own logic",
            tested=True,
        ),
        effects=Effects(deterministic=True, may_execute_callback=True, notes="partial() itself never validates func's signature at construction time -- functools.partial(int, 'x', 'y') builds fine; only calling the result invokes int(...) and can fail"),
        source=(f"{_DOCS}#functools.partial", "CPython Modules/_functoolsmodule.c::partial_new/partial_call"),
        tested=True,
    )


@register_rule
class FunctoolsLruCache(Rule):
    """The rule of ``functools.lru_cache``."""

    SAMPLES = {'functools.lru_cache': 'import functools\n\n@functools.lru_cache\ndef f(x):\n    return x\n\ny = f(1)\n'}
    TWINS = {}

    names = ("functools.lru_cache",)
    CONTRACT = Contract(
        qualname="functools.lru_cache",
        raises=dunder_propagation_claims(
            "the wrapped function's own exceptions (a cache MISS calls through and "
            "propagates whatever the function raises -- a raised call is NOT "
            "cached, so the SAME failing call re-invokes the function every time); "
            "the arguments' __hash__/__eq__ (the cache key)",
            native={
                "TypeError": "maxsize is given and is neither None, an int-like object, nor a callable (the bare-@lru_cache-without-parens form); or the decorated call's arguments are unhashable",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the wrapped function and the arguments' __hash__/__eq__ are arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("a decorator (or, for the bare-@lru_cache form, the wrapped function directly)", "functools.cache is exactly lru_cache(maxsize=None)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True, notes="a HIT returns the SAME cached object every time (no re-computation, no re-raising) -- observable via `is` for a mutable return value; a negative maxsize is accepted (treated as effectively 0/unbounded-adjacent, not rejected)"),
        source=(f"{_DOCS}#functools.lru_cache", "CPython Lib/functools.py::lru_cache / _functools_lru_cache_wrapper (C-accelerated)"),
        tested=True,
    )


__all__ = ["FunctoolsReduce", "FunctoolsPartial", "FunctoolsLruCache"]
