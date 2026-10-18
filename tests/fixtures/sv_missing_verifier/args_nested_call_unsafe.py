# This file is part of the SV-Benchmarks collection of verification tasks:
# https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
#
# SPDX-FileCopyrightText: 2023 Paderborn University
# SPDX-FileCopyrightText: 2026 Luca Negrini (Ca' Foscari University of Venice)
#
# SPDX-License-Identifier: MIT


# Nested function calls
# Functions `nested_func` (returns a string value) and `nested_func2` (returns a nondeterministic int value).
# A function `param_func` is defined which takes a function as a parameter and calls it.
# A function `func` is defined which takes a function 'param_func' as a parameter and calls it, passing
# a nondeterministically chosen nested function (`nested_func` or `nested_func2`).
# Bug: on the int path (`use_str` false) `func` passes the result of calling `nested_func2` instead of the function
# itself, so `param_func` tries to call an int and a TypeError is raised.
import _sv_verifier

x = _sv_verifier.nondet_int()
use_str = _sv_verifier.nondet_bool()


def nested_func():
    return "Hello from nested_func"


def nested_func2():
    return x


def param_func(a):
    return a()


def func(a):
    if use_str:
        return a(nested_func)
    return a(nested_func2())


b = param_func
c = func
d = c(b)

if use_str:
    _sv_verifier.check_type(d, str)
    assert d == "Hello from nested_func"
else:
    _sv_verifier.check_type(d, int)
    assert d == x
