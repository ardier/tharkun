"""Fixture for testing external VarFacts input to condition_solver."""


def int_range(x):
    if x > 50:
        pass


def exclusion(x):
    if x != 0:
        pass


def str_len(x):
    if len(x) > 3:
        pass


def str_methods(x):
    if x.startswith("pre") and x.endswith("suf"):
        pass
