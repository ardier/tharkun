# This file is part of the SV-Benchmarks collection of verification tasks:
# https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
#
# SPDX-FileCopyrightText: 2023 Paderborn University
# SPDX-FileCopyrightText: 2026 Luca Negrini (Ca' Foscari University of Venice)
#
# SPDX-License-Identifier: MIT

# A function func is defined which takes as a parameter a function which it later calls.
# The parameter function is chosen nondeterministically: 'param_func' returns a string
# value, 'param_func_int' returns a nondeterministic integer value.
import _sv_verifier

n = _sv_verifier.nondet_int()


def param_func():
    return "Hello from param_func"


def param_func_int():
    return n


def func(a):
    return a()


if _sv_verifier.nondet_bool():
    b = func(param_func)
    _sv_verifier.check_type(b, str)
    assert b == "Hello from param_func"
else:
    b = func(param_func_int)
    _sv_verifier.check_type(b, int)
    assert b == n
