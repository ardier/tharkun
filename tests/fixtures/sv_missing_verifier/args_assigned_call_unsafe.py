# This file is part of the SV-Benchmarks collection of verification tasks:
# https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
#
# SPDX-FileCopyrightText: 2023 Paderborn University
# SPDX-FileCopyrightText: 2026 Luca Negrini (Ca' Foscari University of Venice)
#
# SPDX-License-Identifier: MIT


# A function object assigned to a variable and passed as an argument to another function.
# A function `func` is defined which takes as a parameter a variable which has a function assigned to it which it later calls.
# The function is chosen nondeterministically: 'param_func' returns a string value, 'other_func' returns an int value.
# Bug: `func` normalizes the result with `.strip()`, assuming it is always a string; when `other_func` is chosen
# the int result has no `strip` attribute and an AttributeError is raised.
import _sv_verifier


def param_func():
    return "Hello from param_func"


def other_func():
    return 42


def func(a):
    return a().strip()


if _sv_verifier.nondet_bool():
    b = param_func
else:
    b = other_func
c = func(b)

if b is param_func:
    _sv_verifier.check_type(c, str)
    assert c == "Hello from param_func"
else:
    _sv_verifier.check_type(c, int)
    assert c == 42
