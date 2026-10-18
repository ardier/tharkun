"""Native callback boundaries that previously had closed-world contracts."""
import collections
import io
import re
import sys

import pytest

from tests.contracts._propagation import PROPERTY_CLASSES
from tharkun.solve.contracts import get

Point = collections.namedtuple("Point", "x")


@pytest.mark.parametrize("name,base,method,call", [
    ("re.escape", str, "translate", lambda cls: re.escape(cls("x"))),
    ("io.StringIO.write", str, "__str__", lambda cls: io.StringIO().write(cls("x"))),
    ("io.BytesIO.write", object, "__buffer__", lambda cls: io.BytesIO().write(cls())),
    ("collections.namedtuple._replace", Point, "_make", lambda cls: Point._replace(cls(1), x=2)),
    ("collections.namedtuple._replace", Point, "__iter__", lambda cls: Point._replace(cls(1), x=2)),
])
@pytest.mark.parametrize("exc", [*PROPERTY_CLASSES.values(), ValueError])
def test_callback_exceptions_are_not_claimed_impossible(name, base, method, call, exc):
    def fail(*args):
        raise exc("user callback")
    callback_type = type("Callback", (base,), {method: fail})
    with pytest.raises(exc, match="user callback"):
        call(callback_type)
    contract = get(name)
    assert contract.effects.may_execute_callback
    assert contract.may_raise_other
    if exc.__name__ in PROPERTY_CLASSES:
        assert contract.raise_claim(exc.__name__).condition != "never"


@pytest.mark.parametrize("name,base,method,call", [
    ("re.escape", str, "translate", lambda cls: re.escape(cls("x"))),
    ("collections.namedtuple._replace", Point, "_make", lambda cls: Point._replace(cls(1))),
])
def test_callback_may_replace_the_return_type(name, base, method, call):
    sentinel = object()
    cls = type("Callback", (base,), {method: lambda *args: sentinel})
    assert call(cls) is sentinel
    assert get(name).returns.type == "Any"


@pytest.mark.parametrize("method", ["append", "appendleft"])
@pytest.mark.parametrize("maxlen", [None, 1, 2])
def test_only_eviction_releases_the_last_reference(method, maxlen):
    events = []
    class Evicted:
        def __del__(self):
            events.append("finalized")
    queue = collections.deque([Evicted()], maxlen=maxlen)
    getattr(queue, method)(0)
    assert events == (["finalized"] if maxlen == 1 else [])
    assert get("collections.deque." + method).effects.may_execute_callback


@pytest.mark.parametrize("method", ["append", "appendleft"])
def test_eviction_exception_is_unraisable_not_propagated(method, monkeypatch):
    errors = []
    monkeypatch.setattr(sys, "unraisablehook", lambda event: errors.append(event.exc_type))
    class Evicted:
        def __del__(self):
            raise IndexError("finalizer")
    queue = collections.deque([Evicted()], maxlen=1)
    assert getattr(queue, method)(0) is None
    assert errors == [IndexError]
    assert get("collections.deque." + method).raise_claim("IndexError").condition == "never"


def test_buffer_release_exception_is_unraisable_not_propagated(monkeypatch):
    errors = []
    monkeypatch.setattr(sys, "unraisablehook", lambda event: errors.append(event.exc_type))
    class Buffer:
        def __buffer__(self, flags):
            return memoryview(b"x")
        def __release_buffer__(self, view):
            raise IndexError("release")
    stream = io.BytesIO()
    assert stream.write(Buffer()) == 1
    assert stream.getvalue() == b"x"
    assert errors == [IndexError]


def test_registry_propagating_callbacks_have_both_open_world_flags():
    from tharkun.solve.contracts import all_contracts
    from tharkun.solve.contracts.types import Effects
    # Propagating callbacks can raise outside the seven property classes. Cleanup
    # callbacks are different: they may execute without their exceptions escaping.
    invalid = [c.qualname for c in all_contracts()
               if any("propagates uncaught" in claim.condition for claim in c.raises)
               and not (c.may_raise_other and c.effects.may_execute_callback)]
    assert invalid == []
    assert Effects().may_execute_callback  # Unknown effects default conservatively.
