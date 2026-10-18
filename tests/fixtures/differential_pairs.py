"""Original/mutant function pairs for differential_kill tests.

Each pair shares a signature so verify_change can compare them positionally.
Line numbers are referenced by the tests' guard_line arguments, so keep the
layout stable.
"""


def add_orig(x):
    return x + x


def add_mut(x):          # equivalent to add_orig (x + x == 2 * x)
    return 2 * x


def bnd_orig(x):
    if x < 10:           # boundary; mutant flips < to <=
        return 1
    return 0


def bnd_mut(x):
    if x <= 10:
        return 1
    return 0


def none_orig(x):
    if x > 5:
        return x
    return None


def none_mut(x):         # return-value mutant: None -> 0
    if x > 5:
        return x
    return 0
