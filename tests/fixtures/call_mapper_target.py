"""Fixture for call_mapper tests — various call patterns in a single file."""


def greet(name):
    return salute(name)


def salute(who):
    if len(who) > 0:
        return f"Hello, {who}!"
    return "Hello!"


def transform_call(data):
    return process(data.strip())


def process(text):
    if text:
        return text.upper()
    return ""


def swap_args(a, b):
    return combine(b, a)


def combine(x, y):
    if x > 0 and y > 0:
        return x + y
    return 0


def keyword_call(value):
    return lookup(data=value)


def lookup(data):
    if data is not None:
        return True
    return False


def multi_caller_a(x):
    return shared_func(x)


def multi_caller_b(y):
    return shared_func(y)


def shared_func(z):
    if z > 10:
        return True
    return False


class Validator:
    def check(self, item):
        if item > 0:
            return True
        return False

    def run(self, data):
        return self.check(data)


def call_method(v):
    obj = Validator()
    return obj.check(v)


def no_call_here(x):
    return x + 1
