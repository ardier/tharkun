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
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseArgsCase, ReturnSpec, ScopedArgsRaiseClaim

_DOCS = "https://docs.python.org/3.14/library/random.html"
_STATE = "the random module's global Mersenne Twister state"
_ALL_TESTED = frozenset(ALL_PROPERTY_CLASSES)


@register_rule
class OsUrandom(Rule):
    """The rule of ``os.urandom(size)`` (the CSPRNG draw every random module seed falls back to)."""

    SAMPLES = {"os.urandom": "import os\nx = os.urandom(4)\n"}
    TWINS = {"os.urandom": "import os\nx = os.urandom()\n"}

    names = ("os.urandom",)
    CONTRACT = Contract(
        qualname="os.urandom",
        raises=dunder_propagation_claims(
            "size's __index__ (a plain non-int object is converted through it; an int subclass passes by value)",
            native={"TypeError": "size is neither an int nor an object with __index__; wrong argument count or an unknown keyword"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: a negative size; OSError: the OS entropy source failed
        returns=ReturnSpec("bytes", "size cryptographically-random bytes", tested=True),
        effects=Effects(randomness="os.urandom (the OS CSPRNG; no process state, not the Mersenne Twister)", deterministic=False, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/os.html#os.urandom", "CPython Modules/posixmodule.c::os_urandom"),
        tested=True,
    )


@register_rule
class UuidUuid4(Rule):
    """The rule of ``uuid.uuid4()`` (a version-4 UUID from the OS CSPRNG)."""

    SAMPLES = {"uuid.uuid4": "import uuid\nx = uuid.uuid4()\n"}
    TWINS = {"uuid.uuid4": "import uuid\nx = uuid.uuid4(1)\n"}

    names = ("uuid.uuid4",)
    CONTRACT = Contract(
        qualname="uuid.uuid4",
        raises=never_claims(native={"TypeError": "any argument or keyword (uuid4 takes none)"}, tested=_ALL_TESTED),
        may_raise_other=True,  # OSError: the OS entropy source failed
        returns=ReturnSpec("uuid.UUID", "a version-4 UUID: 122 random bits, version and variant bits fixed; pid/clock fields are absent", tested=True),
        effects=Effects(randomness="os.urandom (the OS CSPRNG; no process state)", deterministic=False, may_execute_callback=False),
        source=("https://docs.python.org/3.14/library/uuid.html#uuid.uuid4", "CPython Lib/uuid.py::uuid4"),
        tested=True,
    )


@register_rule
class RandomRandom(Rule):
    """The rule of ``random.random``."""

    SAMPLES = {'random.random': 'import random\nx = random.random()\n'}
    TWINS = {'random.random': 'import random\nx = random.random(0)\n'}

    names = ("random.random",)
    CONTRACT = Contract(
        qualname="random.random",
        raises=never_claims(native={"TypeError": "any positional or keyword argument"}, tested=_ALL_TESTED),
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
        returns=ReturnSpec("object", "a + (b - a) * random(); builtin complex and open arithmetic results are allowed, and floating-point overflow may leave the endpoint interval", tested=True),
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


_SEED_TYPE_ARGS=("The only supported seed types are:\nNone, int, float, str, bytes, and bytearray.",)
_SEED_CASES=(
    RaiseArgsCase('native-unsupported-seed',
        'valid binding, exact builtin version with no comparison hook, and seed is not an instance of None/int/float/str/bytes/bytearray',
        _SEED_TYPE_ARGS,True),
    RaiseArgsCase('native-bytearray-version',
        'valid binding, exact builtin bytearray seed and hook-free version not equal to 2',
        ("unhashable type: 'bytearray'",),True),
)+tuple(RaiseArgsCase('native-arity-'+str(n),
    f'no keywords and {n} positional arguments after the bound receiver',
    (f'Random.seed() takes from 1 to 3 positional arguments but {n+1} were given',),True)
    for n in range(3,9))


@register_rule
class RandomSeed(Rule):
    """The rule of ``random.seed``; open subclass/version hooks are not native payloads."""

    SAMPLES = {'random.seed': 'import random\nrandom.seed(1)\n'}
    TWINS = {'random.seed': 'import random\nrandom.seed([])\n'}
    names = ("random.seed",)
    CONTRACT = Contract(
        qualname="random.seed",
        raises=tuple(ScopedArgsRaiseClaim(c.exc,c.condition,c.tested,args_cases=_SEED_CASES)
                     if c.exc=='TypeError' else c for c in dunder_propagation_claims(
            "version comparison/truth hooks and seed subclass encode/decode/iteration/hash hooks",
            native={"TypeError": "invalid binding; unsupported seed type; exact bytearray with version not equal to 2; or a user hook raises TypeError"},
            tested_native=frozenset({'TypeError'}))),
        may_raise_other=True,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(
            deterministic=False, clock_or_env=True, may_execute_callback=True,
            randomness=f"RESETS {_STATE}; clears the generator's cached Gaussian value",
            notes="Exact explicit builtin int/bool/float seeds only reset generator state. None/default uses OS entropy/time. "
                  "Version-2 text/bytes can lazily import SHA-512; other versions hash text/bytes with the process hash seed. "
                  "Open subclasses and version comparisons can execute user callbacks."),
        source=(f"{_DOCS}#random.seed", "https://github.com/python/cpython/blob/3.14/Lib/random.py",
                "https://github.com/python/cpython/blob/3.14/Modules/_randommodule.c"),
        notes="version is compared to 1 and 2, not type-checked as int. Bytearray is converted only for version == 2. "
              "Exact surrogate-containing str can raise UnicodeEncodeError in version 2. Integer subclass __abs__ is bypassed, "
              "but a float subclass __hash__ or str subclass encode can raise arbitrary exceptions with arbitrary args.",
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
            "population length/index/iteration, k comparison/index, and counts iteration/arithmetic/index callbacks",
            native={
                "TypeError": "population is not a sequence; k/counts have invalid kinds, binding is invalid, or population/k/counts callbacks propagate TypeError",
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


@register_rule
class RandomChoices(Rule):
    """The rule of ``random.choices``."""

    SAMPLES = {'random.choices': 'import random\nx = random.choices([1, 2, 3], k=2)\n'}
    TWINS = {'random.choices': 'import random\nx = random.choices()\n'}

    names = ("random.choices",)
    CONTRACT = Contract(
        qualname="random.choices",
        raises=dunder_propagation_claims(
            "population, weights/cum_weights iteration/arithmetic/comparison/truth or k index callbacks",
            native={
                "TypeError": "the population has no __len__ or is not subscriptable (a set); weights and cum_weights are both given; k is not an integer; a weight is not a number; wrong argument count or an unknown keyword, or operand callback propagation",
                "IndexError": "k >= 1 and the population is empty, or an operand/weight/index callback propagates IndexError",
                "KeyError": "the population is a mapping with no integer key in range(len), or an operand/weight/index callback propagates KeyError",
            },
            tested_native=frozenset({"TypeError", "IndexError", "KeyError"}),
        ),
        may_raise_other=True,  # ValueError: weights of another length, total not > 0 or not finite
        returns=ReturnSpec("list", "k elements drawn WITH replacement; [] when k <= 0; the population is not mutated", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.choices", "CPython Lib/random.py::Random.choices"),
        tested=True,
    )


@register_rule
class RandomGetrandbits(Rule):
    """The rule of ``random.getrandbits``."""

    SAMPLES = {'random.getrandbits': 'import random\nx = random.getrandbits(8)\n'}
    TWINS = {'random.getrandbits': 'import random\nx = random.getrandbits()\n'}

    names = ("random.getrandbits",)
    CONTRACT = Contract(
        qualname="random.getrandbits",
        raises=dunder_propagation_claims(
            "k's __index__ (a user method)",
            native={"TypeError": "k is not an integer (no __index__); wrong argument count or a keyword"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: k negative; OverflowError: k does not fit a C uint64
        returns=ReturnSpec("int", "0 <= result < 2**k; 0 for k == 0", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.getrandbits", "CPython Modules/_randommodule.c::_random_Random_getrandbits_impl"),
        tested=True,
    )


_DISTRIBUTION_ARITH = (
    "the argument(s)' arithmetic (__add__/__radd__, __mul__/__rmul__, "
    "__sub__/__rsub__, __truediv__/__rtruediv__, __neg__, __pow__) "
    "that the distribution formula needs"
)
_DIST_ARGS_NATIVE_TYPE = {"TypeError": "an argument does not support the numeric conversion or arithmetic the formula needs"}


@register_rule
class RandomTriangular(Rule):
    """The rule of ``random.triangular``; catches ZeroDivisionError when high == low (returns low)."""

    SAMPLES = {'random.triangular': 'import random\nx = random.triangular(0.0, 1.0)\n'}
    TWINS = {'random.triangular': 'import random\nx = random.triangular(0, 1, 2, 3)\n'}

    names = ("random.triangular",)
    CONTRACT = Contract(
        qualname="random.triangular",
        raises=dunder_propagation_claims(
            _DISTRIBUTION_ARITH,
            native=dict(
                _DIST_ARGS_NATIVE_TYPE,
                ZeroDivisionError="never (caught internally when high == low; returns low instead)",
            ),
            tested_native=frozenset({"TypeError", "ZeroDivisionError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("object", "native interpolation result, or low when the mode calculation divides by zero; open numeric operators may return any object", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.triangular", "CPython Lib/random.py::Random.triangular"),
        tested=True,
    )


@register_rule
class RandomGauss(Rule):
    """The rule of ``random.gauss``; reads and caches a second normal variate on the Random instance (not thread-safe)."""

    SAMPLES = {'random.gauss': 'import random\nx = random.gauss(0.0, 1.0)\n'}
    TWINS = {'random.gauss': 'import random\nx = random.gauss(0, 1, 2)\n'}

    names = ("random.gauss",)
    CONTRACT = Contract(
        qualname="random.gauss",
        raises=dunder_propagation_claims(
            _DISTRIBUTION_ARITH,
            native=dict(
                _DIST_ARGS_NATIVE_TYPE,
                ZeroDivisionError="never (the internal 1.0 - random() is always > 0.0)",
            ),
            tested_native=frozenset({"TypeError", "ZeroDivisionError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("object", "mu plus a standard variate times sigma; builtin complex and open arithmetic results need not be floats", tested=True),
        effects=Effects(
            randomness=_STATE, deterministic=False,
            notes="reads and caches a second normal variate on the Random instance (self.gauss_next); not thread-safe",
        ),
        source=(f"{_DOCS}#random.gauss", "CPython Lib/random.py::Random.gauss"),
        tested=True,
    )


@register_rule
class RandomNormalvariate(Rule):
    """The rule of ``random.normalvariate``."""

    SAMPLES = {'random.normalvariate': 'import random\nx = random.normalvariate(0.0, 1.0)\n'}
    TWINS = {'random.normalvariate': 'import random\nx = random.normalvariate(0, 1, 2)\n'}

    names = ("random.normalvariate",)
    CONTRACT = Contract(
        qualname="random.normalvariate",
        raises=dunder_propagation_claims(
            _DISTRIBUTION_ARITH,
            native=_DIST_ARGS_NATIVE_TYPE,
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("object", "mu plus a standard variate times sigma; builtin complex and open arithmetic results need not be floats", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.normalvariate", "CPython Lib/random.py::Random.normalvariate"),
        tested=True,
    )


@register_rule
class RandomLognormvariate(Rule):
    """The rule of ``random.lognormvariate``; may OverflowError for very large mu/sigma."""

    SAMPLES = {'random.lognormvariate': 'import random\nx = random.lognormvariate(0.0, 1.0)\n'}
    TWINS = {'random.lognormvariate': 'import random\nx = random.lognormvariate()\n'}

    names = ("random.lognormvariate",)
    CONTRACT = Contract(
        qualname="random.lognormvariate",
        raises=dunder_propagation_claims(
            _DISTRIBUTION_ARITH,
            native=_DIST_ARGS_NATIVE_TYPE,
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # OverflowError: mu or sigma produce a value > exp709
        returns=ReturnSpec("float", "exp(normalvariate(mu, sigma)); may underflow to 0.0 or return non-finite values", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.lognormvariate", "CPython Lib/random.py::Random.lognormvariate"),
        tested=True,
    )


@register_rule
class RandomBetavariate(Rule):
    """The rule of ``random.betavariate``; delegates to gammavariate."""

    SAMPLES = {'random.betavariate': 'import random\nx = random.betavariate(1.0, 1.0)\n'}
    TWINS = {'random.betavariate': 'import random\nx = random.betavariate()\n'}

    names = ("random.betavariate",)
    CONTRACT = Contract(
        qualname="random.betavariate",
        raises=dunder_propagation_claims(
            _DISTRIBUTION_ARITH,
            native=dict(
                _DIST_ARGS_NATIVE_TYPE,
                ValueError="alpha or beta <= 0.0 (propagated from gammavariate)",
            ),
            tested_native=frozenset({"TypeError", "ValueError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("object", "native gamma-ratio result, or 0.0 on the zero-gamma shortcut; open numeric protocols may return any object", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.betavariate", "CPython Lib/random.py::Random.betavariate"),
        tested=True,
    )


@register_rule
class RandomGammavariate(Rule):
    """The rule of ``random.gammavariate``; native ValueError for alpha <= 0 or beta <= 0."""

    SAMPLES = {'random.gammavariate': 'import random\nx = random.gammavariate(1.0, 1.0)\n'}
    TWINS = {'random.gammavariate': 'import random\nx = random.gammavariate()\n'}

    names = ("random.gammavariate",)
    CONTRACT = Contract(
        qualname="random.gammavariate",
        raises=dunder_propagation_claims(
            _DISTRIBUTION_ARITH,
            native=dict(
                _DIST_ARGS_NATIVE_TYPE,
                ValueError="alpha or beta <= 0.0 (explicit guard at the top of the function)",
            ),
            tested_native=frozenset({"TypeError", "ValueError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("object", "native gamma arithmetic scaled by beta; binary64 may underflow to zero and open numeric protocols may return any object", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.gammavariate", "CPython Lib/random.py::Random.gammavariate"),
        tested=True,
    )


@register_rule
class RandomExpovariate(Rule):
    """The rule of ``random.expovariate``; ZeroDivisionError when lambd == 0."""

    SAMPLES = {'random.expovariate': 'import random\nx = random.expovariate(1.0)\n'}
    TWINS = {'random.expovariate': 'import random\nx = random.expovariate(0, 1)\n'}

    names = ("random.expovariate",)
    CONTRACT = Contract(
        qualname="random.expovariate",
        raises=dunder_propagation_claims(
            _DISTRIBUTION_ARITH,
            native=dict(
                _DIST_ARGS_NATIVE_TYPE,
                ZeroDivisionError="lambd == 0.0 (division by zero in the formula)",
            ),
            tested_native=frozenset({"TypeError", "ZeroDivisionError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("object", "-log(1-random()) / lambd; includes signed zero, complex and open arithmetic results", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.expovariate", "CPython Lib/random.py::Random.expovariate"),
        tested=True,
    )


@register_rule
class RandomParetovariate(Rule):
    """The rule of ``random.paretovariate``; ZeroDivisionError when alpha == 0."""

    SAMPLES = {'random.paretovariate': 'import random\nx = random.paretovariate(1.0)\n'}
    TWINS = {'random.paretovariate': 'import random\nx = random.paretovariate()\n'}

    names = ("random.paretovariate",)
    CONTRACT = Contract(
        qualname="random.paretovariate",
        raises=dunder_propagation_claims(
            _DISTRIBUTION_ARITH,
            native=dict(
                _DIST_ARGS_NATIVE_TYPE,
                ZeroDivisionError="alpha == 0.0 (division by zero in the exponent)",
            ),
            tested_native=frozenset({"TypeError", "ZeroDivisionError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("object", "u ** (-1.0 / alpha), u=1-random(); negative real alpha can produce values in [0,1], including underflow; open arithmetic may return any object", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.paretovariate", "CPython Lib/random.py::Random.paretovariate"),
        tested=True,
    )


@register_rule
class RandomVonmisesvariate(Rule):
    """The rule of ``random.vonmisesvariate``; uses trig and sqrt internally."""

    SAMPLES = {'random.vonmisesvariate': 'import random\nx = random.vonmisesvariate(0.0, 1.0)\n'}
    TWINS = {'random.vonmisesvariate': 'import random\nx = random.vonmisesvariate()\n'}

    names = ("random.vonmisesvariate",)
    CONTRACT = Contract(
        qualname="random.vonmisesvariate",
        raises=dunder_propagation_claims(
            _DISTRIBUTION_ARITH,
            native=_DIST_ARGS_NATIVE_TYPE,
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("object", "native circular-angle arithmetic; the small-kappa path returns a float, while open mu arithmetic may return any object", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.vonmisesvariate", "CPython Lib/random.py::Random.vonmisesvariate"),
        tested=True,
    )


@register_rule
class RandomWeibullvariate(Rule):
    """The rule of ``random.weibullvariate``; ZeroDivisionError when beta == 0."""

    SAMPLES = {'random.weibullvariate': 'import random\nx = random.weibullvariate(1.0, 1.0)\n'}
    TWINS = {'random.weibullvariate': 'import random\nx = random.weibullvariate()\n'}

    names = ("random.weibullvariate",)
    CONTRACT = Contract(
        qualname="random.weibullvariate",
        raises=dunder_propagation_claims(
            _DISTRIBUTION_ARITH,
            native=dict(
                _DIST_ARGS_NATIVE_TYPE,
                ZeroDivisionError="beta == 0.0 (division by zero in the exponent)",
            ),
            tested_native=frozenset({"TypeError", "ZeroDivisionError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("object", "alpha * (-log(u)) ** (1.0 / beta), u=1-random(); signs, zero, complex and open arithmetic results are retained", tested=True),
        effects=Effects(randomness=_STATE, deterministic=False),
        source=(f"{_DOCS}#random.weibullvariate", "CPython Lib/random.py::Random.weibullvariate"),
        tested=True,
    )


__all__ = [
    "RandomRandom", "RandomRandint", "RandomRandrange", "RandomChoice", "RandomUniform",
    "RandomSeed", "RandomSample", "RandomShuffle", "RandomChoices", "RandomGetrandbits",
    "RandomTriangular", "RandomGauss", "RandomNormalvariate", "RandomLognormvariate",
    "RandomBetavariate", "RandomGammavariate", "RandomExpovariate", "RandomParetovariate",
    "RandomVonmisesvariate", "RandomWeibullvariate",
]
