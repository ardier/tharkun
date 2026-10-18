"""Defines the same names as mod_a/mod_b, but the harness never imports it.

Nothing here may ever be targeted: the assist must only reach what the original harness
can reach statically.
"""


def check_a(code):
    raise AssertionError("decoy.check_a must never be targeted")


def check_b(name):
    raise AssertionError("decoy.check_b must never be targeted")
