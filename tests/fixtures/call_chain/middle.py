"""Middle level in the call chain fixture."""

from tests.fixtures.call_chain.deep import validate


def process(data):
    if validate(data):
        return "ok"
    return "fail"
