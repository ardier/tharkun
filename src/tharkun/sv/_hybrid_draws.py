"""How the SV hybrid worker turns fuzzer bytes into ``_sv_verifier`` draws, and the inverse ([internal ref], [internal ref] PR3).

One module so the decoding ``sv/_hybrid_worker.py`` runs and the encoding ``sv/hybrid.py`` seeds the corpus with
are the SAME code: a seed is a whole DRAW VECTOR written as the exact bytes that decode back to it, checked by
decoding it again through the real ``atheris.FuzzedDataProvider`` (the oracle) before it is written.

The layout, measured on atheris 3.1.0 / CPython 3.14.4 (``tests/test_sv_hybrid_draws.py`` pins it): ``ConsumeBytes``,
``ConsumeInt`` AND ``ConsumeBool`` consume from the FRONT of the buffer, in call order; ``ConsumeIntInRange`` from
the back (``hybrid/fdp_codec.py``'s table lists ``ConsumeBool`` as back-consuming: not so on 3.1.0 -- its replay check
rejects such an encoding rather than emitting it).

* ``nondet_str`` takes ``ConsumeBytes(n)`` with ``n = min(STR_DRAW_MAX, remaining - reserve)``: every byte that is
  left except those the LATER draws of the program's known draw sequence need (``kinds``, from its first run; a later
  str reserves nothing).  So a program with one str draw sees the whole input as its string (``b"qqqqq"`` IS
  ``"qqqqq"``: a literal is zero mutations from itself), a str is no longer capped at 64 characters ([internal ref]: the K4
  mutants need 257-321), and a ``nondet_bool`` after a str still has its byte at the back ([internal ref]: before, the str ate
  every byte of an input up to 64 long, so the bool read nothing and was False);
* a str draw OFF the known sequence (the run's draws so far are not the first run's, or the first run drew no str
  there: an element of a collection that was empty in the first run, [internal ref]) cannot know what follows it, so it is
  LENGTH-PREFIXED instead: one byte ``L`` (0..255), then ``min(L, remaining)`` bytes (:func:`consume_str`), and a
  later field is never starved by it;
* each byte becomes one ``string.printable`` character (:func:`printable_str`, [internal ref]);
* ``nondet_int`` is ``ConsumeInt(8)``, ``nondet_bool`` is ``ConsumeBool()`` (one byte, its low bit).

:func:`encode` writes the fields in draw order; the str field's length comes out exactly, since ``remaining - reserve``
at that point is its own length.  Kinds it cannot write (a float, a list) make it return ``None``: the seed is not
written, never approximated.

Imports no ``tharkun`` module (the worker loads it by path, like ``_child.py``) and nothing at import time but
``string``.
"""

from __future__ import annotations

import string

__all__ = ["STR_DRAW_MAX", "FIXED_BYTES", "BYTE_TO_CHAR", "printable_str", "reserve_after", "str_length", "on_script",
           "consume_str", "encode", "decode", "encode_checked"]

#: the longest str one draw can be (the email length bound of RFC 3696 is 320; this leaves room for any guard)
STR_DRAW_MAX = 4096
#: bytes a later draw of each kind needs; a kind not listed (a float range, a list) reserves nothing
FIXED_BYTES = {"bool": 1, "int": 8}

_PRINTABLE = frozenset(string.printable)
#: One printable character per input byte ([internal ref] follow-up): a byte that IS a printable character stays itself, every
#: other byte (NUL and the other control characters, DEL, 0x80-0xFF) is folded onto the visible range 0x20..0x7E
BYTE_TO_CHAR = tuple(chr(b) if chr(b) in string.printable else chr(0x20 + b % 0x5F) for b in range(256))


def printable_str(data: bytes) -> str:
    return "".join(BYTE_TO_CHAR[b] for b in data)


def reserve_after(kinds, index: int) -> int:
    """The bytes the draws after position ``index`` of the known sequence need (later strs: none)."""
    if not kinds:
        return 0
    return sum(FIXED_BYTES.get(k, 0) for k in list(kinds)[index + 1:])


def str_length(remaining: int, kinds, index: int) -> int:
    """How many bytes the str draw at position ``index`` takes."""
    return max(0, min(STR_DRAW_MAX, remaining - reserve_after(kinds, index)))


def on_script(kinds, so_far) -> bool:
    """Is a str draw made after the draws of kinds ``so_far`` at a position where the program's first run drew a str,
    with the same draws before it?  (Unknown ``kinds``: yes, the rule is then "the whole rest".)"""
    if not kinds:
        return True
    i = len(so_far)
    return i < len(kinds) and kinds[i] == "str" and list(kinds[:i]) == list(so_far)


def consume_str(fdp, kinds, so_far) -> bytes:
    """The bytes of the str draw made after the draws of kinds ``so_far``: :func:`str_length` of them on the known
    sequence, else a one-byte length prefix and at most that many bytes ([internal ref]), so a str in a grown collection
    leaves the rest of the input to the draws after it."""
    if on_script(kinds, so_far):
        return fdp.ConsumeBytes(str_length(fdp.remaining_bytes(), kinds, len(so_far)))
    head = fdp.ConsumeBytes(1)
    return fdp.ConsumeBytes(min(head[0] if head else 0, fdp.remaining_bytes()))


def _draw(fdp, kind: str, kinds, index: int):
    if kind == "str":
        return printable_str(fdp.ConsumeBytes(str_length(fdp.remaining_bytes(), kinds, index)))
    if kind == "int":
        return fdp.ConsumeInt(8)
    if kind == "bool":
        return fdp.ConsumeBool()
    raise ValueError(f"no byte layout for a {kind} draw")


def encode(vector) -> bytes | None:
    """The bytes that decode to ``vector`` (``[{"kind", "value"}, ...]``), or ``None`` when a draw cannot be written
    (a kind without a layout, a non-printable str character, an int wider than 8 bytes, a str over
    :data:`STR_DRAW_MAX`).  Not checked here: :func:`decode` with the real provider is the check."""
    front = []
    for entry in vector:
        kind, value = entry.get("kind"), entry.get("value")
        if kind == "str":
            if type(value) is not str or len(value) > STR_DRAW_MAX or not all(c in _PRINTABLE for c in value):
                return None
            front.append(value.encode("ascii"))
        elif kind == "int":
            if type(value) is not int:
                return None
            try:
                front.append(value.to_bytes(8, "little", signed=True))
            except OverflowError:
                return None
        elif kind == "bool":
            if type(value) is not bool:
                return None
            front.append(bytes([1 if value else 0]))
        else:
            return None
    return b"".join(front)


def decode(data: bytes, kinds, atheris=None) -> list | None:
    """What the worker's draws read from ``data`` for the draw sequence ``kinds`` -- the same code path
    (``_draw``) the worker runs, on the real ``atheris.FuzzedDataProvider``; ``None`` when a kind has no layout."""
    if atheris is None:
        import atheris                                    # optional: only the hybrid stage has it
    fdp = atheris.FuzzedDataProvider(data)
    out = []
    try:
        for i, kind in enumerate(kinds):
            out.append({"kind": kind, "value": _draw(fdp, kind, kinds, i)})
    except ValueError:
        return None
    return out


def encode_checked(vector, atheris=None) -> bytes | None:
    """:func:`encode`, kept only when :func:`decode` gives back exactly the vector (the oracle's verdict)."""
    data = encode(vector)
    if data is None:
        return None
    kinds = [e["kind"] for e in vector]
    got = decode(data, kinds, atheris)
    if got is None or [(g["kind"], g["value"]) for g in got] != [(e["kind"], e["value"]) for e in vector]:
        return None
    return data


# Box-only corpus framing; the ordinary SV byte layout above is unchanged.
_WORLD_MAGIC = b"THW1"


def pack_world(data: bytes, world: dict) -> bytes:
    """Keep file bytes/facts and nondet bytes in one libFuzzer corpus entry."""
    import json
    payload = json.dumps(world, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _WORLD_MAGIC + len(payload).to_bytes(4, "little") + payload + data


def unpack_world(data: bytes):
    """Return raw nondet bytes and an optional snapshot; malformed frames are raw fuzz input."""
    import json
    if data.startswith(_WORLD_MAGIC) and len(data) >= 8:
        size = int.from_bytes(data[4:8], "little")
        if size <= 2**20 and size <= len(data) - 8:
            try:
                world = json.loads(data[8:8 + size])
                if isinstance(world, dict):
                    return data[8 + size:], world
            except (ValueError, UnicodeError, RecursionError):
                pass
    return data, None
