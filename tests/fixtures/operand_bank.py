"""A guard whose demanded value exists only at runtime, behind a transform.

RedQueen harvests COMPARISONS, so the guard is one; the demanded value is computed rather
than written, so nothing static can read it out of the source; and it arrives through
`int.from_bytes`, so it must be encoded back before an input can carry it.
"""


def gate(token, other):
    expected = (0x1234 * 3 + 7) & 0xFFFF          # 13987, computed at runtime
    if int.from_bytes(token[:2], "big") == expected:
        if other == 999:
            return "BOTH"
        return "FIRST"
    return "no"
