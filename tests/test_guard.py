

# ------------------------------------------------------------------ #0657
def test_pure_literal_calls_fold_and_impure_ones_do_not():
    """The extractor folds a call only when the callee is a known-pure
    constructor and every argument is a literal.  A size argument is refused
    outright (``bytes(10**9)`` would allocate a gigabyte), a malformed literal
    is not a fact, and nothing user-defined is ever evaluated."""
    import ast

    from tharkun.solve.guard import try_extract_constant

    def take(src):
        return try_extract_constant(ast.parse(src, mode="eval").body)

    assert take("bytes.fromhex('00')") == (b"\x00", True)
    assert take("bytearray.fromhex('0102')") == (b"\x01\x02", True)
    assert take("bytes(b'ab')") == (b"ab", True)
    assert take("frozenset((1, 2))") == (frozenset({1, 2}), True)

    for refused in ("bytes(1000000000)", "bytes.fromhex('zz')", "open('/etc/passwd')",
                    "foo.bar('x')", "bytes.fromhex(x)", "sorted([3, 1])",
                    "bytes.fromhex('00', extra=1)"):
        assert take(refused) == (None, False), refused

    # the pre-#0657 behaviour is untouched
    assert take("-100") == (-100, True)
    assert take("'GET'") == ("GET", True)
