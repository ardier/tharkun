"""Fixtures: byte-input fuzz gates that test int.from_bytes(...) / byte indexing.

These mirror the entry guards of byte-input parsers (ASN.1 DER header tag,
DNS / length-prefixed wire formats) that a fuzzer must satisfy to enter the
function body. The variable under test (``data``) is raw ``bytes`` controlled
by the fuzzer; the guard inspects individual bytes or a big/little-endian
integer lifted from a slice.
"""

from __future__ import annotations


def asn1_tag(data):
    if data[0] == 0x30:  # GUARD_index_eq
        return "sequence"
    return "other"


def asn1_tag_ge(data):
    if data[0] >= 0x20:  # GUARD_index_ge
        return "constructed-ish"
    return "primitive"


def length_prefix_big(data):
    if int.from_bytes(data[0:2], "big") > 1000:  # GUARD_fb_gt
        return "long"
    return "short"


def length_prefix_eq(data):
    if int.from_bytes(data[2:4], "big") == 256:  # GUARD_fb_eq
        return "exact"
    return "no"


def length_prefix_little(data):
    if int.from_bytes(data[0:4], "little") >= 65536:  # GUARD_fb_little
        return "wide"
    return "narrow"


def magic_index(data):
    if data[3] == 0xFF and data[0] == 0x7F:  # GUARD_index_and
        return "magic"
    return "plain"


def from_bytes_kwarg(data):
    if int.from_bytes(data[1:3], byteorder="big") < 5:  # GUARD_fb_kwarg
        return "tiny"
    return "big"


def from_bytes_no_order(data):
    # int.from_bytes default byteorder is "big" (Py3.11+); older needs explicit.
    if int.from_bytes(data[0:1], "big") == 9:  # GUARD_fb_single
        return "nine"
    return "other"
