"""A stand-in for the benchmark stub's nondeterministic functions, for native control runs in tests.

Written for this repository from the documented behaviour of ``_sv_verifier`` (random draws built from a fair
coin); it is not a copy of the benchmark's file. What the tests rely on: a callback's exception propagates, a
dictionary evaluates the value callback before the key callback of each entry, and a wrong number of arguments is
a ``TypeError`` raised by the call itself.
"""
import random
import string


def nondet_bool():
    return random.random() < 0.5


def nondet_int():
    magnitude = 0
    while nondet_bool():
        magnitude += 1
    return -magnitude if nondet_bool() else magnitude


def nondet_float():
    special = (float("inf"), float("-inf"), float("nan"), -0.0)
    return random.choice(special) if nondet_bool() else random.uniform(-1e308, 1e308)


def nondet_str():
    return "".join(random.choice(string.printable) for _ in range(abs(nondet_int())))


def nondet_list(nondet_elem):
    items = []
    while nondet_bool():
        items.append(nondet_elem())
    return items


def nondet_dict(nondet_key, nondet_value):
    entries = {}
    while nondet_bool():
        value = nondet_value()       # the value is drawn first, as in ``d[key()] = value()``
        entries[nondet_key()] = value
    return entries
