"""A protocol-style reader whose branches are gated by bitmask flag tests."""
from flagpkg.constants import Flags, MODULE_LEVEL_MASK


def read(negotiate_flags):
    if negotiate_flags & Flags.EXTENDED_SESSIONSECURITY:
        return "extended"
    elif negotiate_flags & Flags.NEGOTIATE_SEAL:
        return "seal"
    elif negotiate_flags & MODULE_LEVEL_MASK:
        return "module-mask"
    return "none"


def read_explicit(flags):
    if flags & Flags.NEGOTIATE_UNICODE != 0:
        return "unicode-ne"
    return "no"


def read_all_bits(flags):
    if flags & Flags.NEGOTIATE_SEAL == Flags.NEGOTIATE_SEAL:
        return "seal-eq"
    return "no"


def read_cleared(flags):
    if flags & Flags.NEGOTIATE_SEAL == 0:
        return "cleared"
    return "set"
