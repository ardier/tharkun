"""Entry level in the call chain fixture."""

from tests.fixtures.call_chain.middle import process


def start(raw_data):
    result = process(raw_data.strip())
    return result


def start_simple(x):
    return process(x)
