"""User index callbacks can propagate exceptions outside the seven property classes."""
import pytest
from tharkun.solve.contracts.registry import get


class BadIndex:
    def __index__(self):
        raise ValueError("user callback")


@pytest.mark.parametrize("name,call", [
    ("str.startswith", lambda: "x".startswith("x", BadIndex())),
    ("str.endswith", lambda: "x".endswith("x", BadIndex())),
    ("str.replace", lambda: "x".replace("x", "y", BadIndex())),
    ("bytes.hex", lambda: b"x".hex("-", BadIndex())),
])
def test_other_exceptions_remain_possible(name, call):
    with pytest.raises(ValueError, match="user callback"):
        call()
    assert get(name).may_raise_other
