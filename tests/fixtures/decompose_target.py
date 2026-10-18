"""Fixture for guard decomposer tests — various function call patterns."""


# --- Simple boolean function ---
def is_valid(x):
    return x > 0 and x < 100


def uses_is_valid(x):
    if is_valid(x):
        pass


# --- Guard clause function ---
def check_data(data):
    if not data:
        return False
    if len(data) < 5:
        return False
    return True


def uses_check_data(data):
    if check_data(data):
        pass


# --- Nested calls ---
def inner1(x):
    return x > 0


def inner2(x):
    return isinstance(x, int)


def outer(x):
    return inner1(x) and inner2(x)


def uses_outer(x):
    if outer(x):
        pass


# --- Opaque external call ---
def uses_external(x):
    if external_lib.validate(x):
        pass


# --- Mixed: 2 solvable + 1 opaque ---
def partial_check(x, data):
    return x > 0 and len(data) >= 3 and external_validate(data)


def uses_partial(x, data):
    if partial_check(x, data):
        pass


# --- Class with method ---
class Config:
    def is_ready(self):
        return self.initialized and self.connected

    def run(self):
        if self.is_ready():
            pass


# --- If-true/return-false pattern ---
def is_positive(n):
    if n > 0:
        return True
    return False


def uses_positive(n):
    if is_positive(n):
        pass


# --- Direct boolean guard (no call) ---
def direct_guard(x, name):
    if x > 10 and name == "admin":
        pass


# --- Deeply nested ---
def level3(x):
    return x != 0


def level2(x):
    return level3(x) and x > -100


def level1(x):
    return level2(x)


def uses_deep(x):
    if level1(x):
        pass


# --- Or-based function return ---
def accepts_many(x):
    return isinstance(x, int) or isinstance(x, str)


def uses_accepts(x):
    if accepts_many(x):
        pass


# --- Guard with raise ---
def strict_check(x):
    if x is None:
        raise ValueError("null")
    if x < 0:
        raise ValueError("negative")
    return True


def uses_strict(x):
    if strict_check(x):
        pass
