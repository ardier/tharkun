"""Contracts for the ``random`` module ([internal ref]).

Cross-checked against ESBMC's ``src/python-frontend/models/random.py`` (Apache-2.0,
see ``docs/contracts/THIRD_PARTY_REFERENCES.md``): its own operational model raises
``IndexError`` from ``choice()``/``choice_float()``/``choice_str()``/``choice_chars()``
on an empty sequence ("matching CPython", its own comment says) and ``ValueError`` from
``sample()`` when ``k`` is negative or exceeds the population -- both confirmed
independently against real CPython 3.14.4 below.

Every function in this module reads the SAME global state: the process-wide Mersenne
Twister instance ``random`` module-level functions are bound methods of
(``random._inst``). Two calls in the same process are correlated through that shared
state (this is what ``random`` module docs mean by "not suitable for cryptographic
purposes" -- the state is a single evolving generator, not independent draws), and
``seed()`` is the only function that resets it (or, with no/None argument, reseeds it
from ``os.urandom``, which pulls from the OS's CSPRNG and the process clock).

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/random.html"
_STATE = "the random module's global Mersenne Twister state"
_ALL_TESTED = frozenset(ALL_PROPERTY_CLASSES)


@register_rule
class RandomRandom(Rule):
    """The rule of ``random.random``."""

    SAMPLES = {'random.random': 'import random\nx = random.random()\n'}
    TWINS = {'random.random': 'import random\nx = random.random(0)\n'}

    names = ("random.random",)
    CONTRACT = Contract(
        qualname="random.random",
        raises=never_claims(tested=_ALL_TESTED),
        may_raise_other=False,
        returns=ReturnSpec("float", "in [0.0, 1.0)", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.random", "CPython Modules/_randommodule.c::random_random"),
        tested=True,
    )


@register_rule
class RandomRandint(Rule):
    """The rule of ``random.randint``."""

    SAMPLES = {'random.randint': 'import random\nx = random.randint(1, 6)\n'}
    TWINS = {'random.randint': 'import random\nx = random.randint()\n'}

    names = ("random.randint",)
    CONTRACT = Contract(
        qualname="random.randint",
        raises=dunder_propagation_claims(
            "a's or b's __index__",
            native={"TypeError": "a or b has no __index__ (e.g. a str or float)"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: a > b (empty range)
        returns=ReturnSpec("int", "in [a, b], inclusive of BOTH endpoints (unlike randrange)", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.randint", "CPython Lib/random.py::Random.randint"),
        tested=True,
    )


@register_rule
class RandomRandrange(Rule):
    """The rule of ``random.randrange``."""

    SAMPLES = {'random.randrange': 'import random\nx = random.randrange(5)\n'}
    TWINS = {'random.randrange': 'import random\nx = random.randrange()\n'}

    names = ("random.randrange",)
    CONTRACT = Contract(
        qualname="random.randrange",
        raises=dunder_propagation_claims(
            "the argument(s)' __index__",
            native={"TypeError": "an argument has no __index__"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: empty range, or step == 0
        returns=ReturnSpec("int", "in range(start, stop, step) -- EXCLUDES stop, unlike randint", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.randrange", "CPython Lib/random.py::Random.randrange"),
        tested=True,
    )


@register_rule
class RandomChoice(Rule):
    """The rule of ``random.choice``."""

    SAMPLES = {'random.choice': 'import random\nx = random.choice([1, 2, 3])\n'}
    TWINS = {'random.choice': 'import random\nx = random.choice()\n'}

    names = ("random.choice",)
    CONTRACT = Contract(
        qualname="random.choice",
        raises=dunder_propagation_claims(
            "the sequence's __len__ (always) or __getitem__ (only when non-empty)",
            native={
                "TypeError": "the argument has no __len__ (is not a sequence)",
                "IndexError": "the sequence is empty",
            },
            tested_native=frozenset({"TypeError", "IndexError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("the element type", "a uniformly-chosen element; does not mutate the sequence", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.choice", "CPython Lib/random.py::Random.choice"),
        tested=True,
    )


@register_rule
class RandomUniform(Rule):
    """The rule of ``random.uniform``."""

    SAMPLES = {'random.uniform': 'import random\nx = random.uniform(0, 1)\n'}
    TWINS = {'random.uniform': 'import random\nx = random.uniform()\n'}

    names = ("random.uniform",)
    CONTRACT = Contract(
        qualname="random.uniform",
        raises=dunder_propagation_claims(
            "the arithmetic __sub__/__rsub__, __mul__/__rmul__, __add__/__radd__ "
            "needed to compute a + (b - a) * self.random() -- a PURE-PYTHON "
            "implementation, unlike math.sqrt/log's C-level __float__/__index__ "
            "conversion (confirmed by reading CPython Lib/random.py -- see notes)",
            native={"TypeError": "a or b supports none of the arithmetic operators the formula needs"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("float", "in [a, b] (or [b, a] if a > b -- NEVER raises for a > b, unlike randint/randrange)", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        notes=(
            "random.uniform is PURE PYTHON (`a + (b - a) * self.random()`), unlike "
            "math.sqrt/log's C-level PyFloat_AsDouble conversion -- a stub object "
            "with only __float__ set does NOT satisfy it (its arithmetic operators "
            "raise 'unsupported operand type(s)' instead); the propagation path is "
            "through __sub__/__rsub__, __mul__/__rmul__, __add__/__radd__, confirmed "
            "against CPython Lib/random.py::Random.uniform's actual source line."
        ),
        source=(f"{_DOCS}#random.uniform", "CPython Lib/random.py::Random.uniform"),
        tested=True,
    )


@register_rule
class RandomSeed(Rule):
    """The rule of ``random.seed``."""

    SAMPLES = {'random.seed': 'import random\nrandom.seed(1)\n'}
    TWINS = {}

    names = ("random.seed",)
    CONTRACT = Contract(
        qualname="random.seed",
        raises=never_claims(
            native={
                "TypeError": (
                    "a is given and is not one of None, int, float, str, bytes, "
                    "bytearray -- unlike most of this module, seed() type-checks "
                    "directly rather than duck-typing via a dunder, so there is no "
                    "propagation path at all"
                ),
            },
            tested=_ALL_TESTED,
        ),
        may_raise_other=False,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(
            deterministic=False,
            clock_or_env=True,
            randomness=f"RESETS {_STATE}: with a is None (the default), reseeds from os.urandom (OS CSPRNG) plus the process clock",
            notes="the only function in this module that mutates the shared generator state rather than just reading it",
        ),
        source=(f"{_DOCS}#random.seed", "CPython Lib/random.py::Random.seed"),
        tested=True,
    )


@register_rule
class RandomSample(Rule):
    """The rule of ``random.sample``."""

    SAMPLES = {'random.sample': 'import random\nx = random.sample([1, 2, 3], 2)\n'}
    TWINS = {'random.sample': 'import random\nx = random.sample()\n'}

    names = ("random.sample",)
    CONTRACT = Contract(
        qualname="random.sample",
        raises=dunder_propagation_claims(
            "the population's __len__ (always) or __getitem__ (to draw each element)",
            native={
                "TypeError": "population is not a sequence (e.g. a set or dict -- explicitly rejected, not just unsupported)",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: k negative or k > len(population)
        returns=ReturnSpec("list", "k DISTINCT elements (by position, not value), in random order; does not mutate population", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.sample", "CPython Lib/random.py::Random.sample"),
        tested=True,
    )


@register_rule
class RandomShuffle(Rule):
    """The rule of ``random.shuffle``."""

    SAMPLES = {'random.shuffle': 'import random\nxs = [1, 2, 3]\nrandom.shuffle(xs)\n'}
    TWINS = {'random.shuffle': 'import random\nrandom.shuffle()\n'}

    names = ("random.shuffle",)
    CONTRACT = Contract(
        qualname="random.shuffle",
        raises=dunder_propagation_claims(
            "the argument's __len__ (always) and __getitem__/__setitem__ (to swap elements)",
            native={
                "TypeError": "the argument has no __len__, or lacks __getitem__/__setitem__ (e.g. an immutable sequence like str or tuple)",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("None", "shuffles the argument IN PLACE (Fisher-Yates); does not return it", tested=True),
        effects=Effects(mutates_args=("x",), randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.shuffle", "CPython Lib/random.py::Random.shuffle"),
        tested=True,
    )


__all__ = ["RandomRandom", "RandomRandint", "RandomRandrange", "RandomChoice", "RandomUniform", "RandomSeed", "RandomSample", "RandomShuffle"]
