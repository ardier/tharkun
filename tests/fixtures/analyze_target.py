"""Fixture for guard_analyzer tests — functions with blocking conditions."""


def validate(data):
    return len(data) > 5 and data[0] == "H"


def process(data):
    if validate(data):
        return "ok"
    return "fail"


def simple_guard(x):
    if x > 100:
        return "big"
    return "small"


class Config:
    def __init__(self):
        self.ready = False
        self.name = ""

    def is_valid(self):
        return self.ready and len(self.name) > 0

    def run(self):
        if self.is_valid():
            return "running"
        return "not ready"
