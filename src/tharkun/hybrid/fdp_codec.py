"""Atheris FuzzedDataProvider codec: record a consumption schedule, re-encode it.

The hybrid fuzzer keeps the *real, unmodified* OSS harness — ``TestOneInput(data:
bytes)`` that builds an ``atheris.FuzzedDataProvider(data)`` and lifts typed values
out of it.  To let tharkun solve over those typed values and feed a solution back
as a byte seed Atheris will accept, we must bridge bytes <-> typed values in a way
that is *byte-identical* to Atheris's own decoding.

Rather than reimplement Atheris's (quirky) byte math, this codec uses the real FDP
as the oracle:

  * :class:`RecordingFDP` wraps a real ``atheris.FuzzedDataProvider``, delegates every
    ``Consume*`` call to it (so decoding is always exactly Atheris's), and records a
    :class:`FieldRecord` per call — method, args, value, and the exact byte range it
    consumed, measured from ``remaining_bytes()`` deltas.

  * The front/back split is Atheris's: data methods (``ConsumeBytes``/``ConsumeInt``/
    ``ConsumeUnicode*``/``ConsumeString``) consume from the front of the buffer;
    bounded-integral methods (``ConsumeIntInRange``/``ConsumeBool``/``ConsumeFloat*``/
    ``ConsumeProbability``) consume from the back.  Empirically verified against
    atheris 3.1.0 on CPython 3.14.4.

  * :func:`encode_overrides` rewrites the bytes for chosen fields to new typed values
    and then **verifies the whole buffer by replaying it through a fresh real FDP** —
    an override that does not reproduce (an unsupported field type, a value out of the
    consumed width) is rejected, never emitted.  Execution/decoding is the judge.

Atheris is an optional dependency; this module imports it lazily.  ``tharkun.hybrid``
requires Python >=3.11 (Atheris) even though tharkun's core stays >=3.9.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable


def atheris_available() -> bool:
    try:
        importlib.import_module("atheris")
        return True
    except Exception:
        return False


# Consumption end per FDP method.  Anything not listed is treated as front-consuming
# and left un-encodable (recorded, but not a re-encode target) unless added here.
# Consumption end per FDP method (verified against atheris 3.0.0 via remaining_bytes()):
# sized/data reads take from the front, bounded-integral reads from the back.
_FRONT_METHODS = {"ConsumeBytes", "ConsumeInt", "ConsumeUInt", "ConsumeUnicode",
                  "ConsumeUnicodeNoSurrogates", "ConsumeString"}
_BACK_METHODS = {"ConsumeIntInRange", "ConsumeBool", "ConsumeFloat", "ConsumeFloatInRange",
                 "ConsumeRegularFloat", "ConsumeProbability"}


@dataclass
class FieldRecord:
    """One ``Consume*`` call captured during a replay."""

    index: int                 # position in the consumption schedule
    method: str                # e.g. "ConsumeIntInRange"
    args: tuple                # positional args passed to the method
    value: Any                 # the value the real FDP returned
    end: str                   # "front" | "back"
    nbytes: int                # bytes this call consumed
    byte_range: tuple[int, int]  # [lo, hi) absolute indices in the original buffer
    provider: int = 0          # which FuzzedDataProvider read it; each one restarts at
                               # byte 0, so fields from different providers can OVERLAP


class RecordingFDP:
    """A recording proxy over ``atheris.FuzzedDataProvider``.

    Substitute this for ``atheris.FuzzedDataProvider`` during a replay of the harness
    (see :func:`record_schedule`) to capture the consumption schedule.  Every method
    delegates to the real FDP, so the values the harness sees are exactly Atheris's.
    """

    def __init__(self, data: bytes, provider: int = 0, sink: list | None = None,
                 real_ctor=None) -> None:
        # Take the REAL constructor explicitly: record_schedule patches
        # atheris.FuzzedDataProvider to point here, so looking it up would recurse.
        if real_ctor is None:
            real_ctor = importlib.import_module("atheris").FuzzedDataProvider
        self._data = bytes(data)
        self._real = real_ctor(self._data)
        self._front = 0
        self._back = len(self._data)
        self._provider = provider
        # A shared sink keeps records in the order the harness actually made the calls,
        # even when it interleaves two providers.
        self.records: list[FieldRecord] = [] if sink is None else sink

    def _consume(self, method: str, args: tuple) -> Any:
        end = "back" if method in _BACK_METHODS else "front"
        before = self._real.remaining_bytes()
        value = getattr(self._real, method)(*args)
        after = self._real.remaining_bytes()
        n = before - after
        if end == "front":
            lo, hi = self._front, self._front + n
            self._front += n
        else:
            hi, lo = self._back, self._back - n
            self._back -= n
        self.records.append(FieldRecord(len(self.records), method, args, value, end, n,
                                        (lo, hi), self._provider))
        return value

    # --- delegated FDP surface (the commonly used subset) ---
    def ConsumeInt(self, n): return self._consume("ConsumeInt", (n,))
    def ConsumeUInt(self, n): return self._consume("ConsumeUInt", (n,))
    def ConsumeIntInRange(self, a, b): return self._consume("ConsumeIntInRange", (a, b))
    def ConsumeBool(self): return self._consume("ConsumeBool", ())
    def ConsumeBytes(self, n): return self._consume("ConsumeBytes", (n,))
    def ConsumeUnicode(self, n): return self._consume("ConsumeUnicode", (n,))
    def ConsumeUnicodeNoSurrogates(self, n): return self._consume("ConsumeUnicodeNoSurrogates", (n,))
    def ConsumeString(self, n): return self._consume("ConsumeString", (n,))
    def ConsumeFloat(self): return self._consume("ConsumeFloat", ())
    def ConsumeFloatInRange(self, a, b): return self._consume("ConsumeFloatInRange", (a, b))
    def ConsumeProbability(self): return self._consume("ConsumeProbability", ())
    def remaining_bytes(self): return self._real.remaining_bytes()

    def __getattr__(self, name: str):
        """Delegate any Consume* method we did not list explicitly.

        Without this, an unlisted reader (ConsumeRegularFloat, ConsumeIntList, ...)
        raised AttributeError inside the harness; record_schedule swallows harness
        exceptions, so the schedule was silently TRUNCATED at that call and every later
        field vanished — a wrong model that looked like a short harness.
        """
        if not name.startswith("Consume") and name != "PickValueInList":
            raise AttributeError(name)
        real_attr = getattr(self._real, name)
        if not callable(real_attr):
            return real_attr

        def _recorded(*args):
            return self._consume(name, args)

        return _recorded
    def PickValueInList(self, lst):
        # Modeled as an index-in-range consume so the record is re-encodable.
        idx = self._consume("ConsumeIntInRange", (0, max(len(lst) - 1, 0)))
        return lst[idx] if lst else None


def field_bounds(rec: FieldRecord) -> tuple[int, int] | None:
    """Inclusive integer ``(lo, hi)`` range a field's method can produce, or None.

    Lets the solver stay inside a field's actual domain so its solution is encodable
    (a ConsumeUInt(4) field holds [0, 2**32-1]; a solution outside it cannot be written
    back into 4 fixed bytes).  Width is the *consumed* byte count.
    """
    n = rec.nbytes
    if rec.method == "ConsumeUInt" and n > 0:
        return (0, 256 ** n - 1)
    if rec.method == "ConsumeInt" and n > 0:
        half = 256 ** n // 2
        return (-half, half - 1)
    if rec.method == "ConsumeIntInRange":
        lo, hi = rec.args
        return (lo, hi)
    if rec.method == "ConsumeBool":
        return (0, 1)
    return None


def replay_values(data: bytes, schedule: list[FieldRecord]) -> list[Any]:
    """Re-run *schedule*'s method sequence against a fresh real FDP over *data*.

    This is the ground-truth decoder used to verify a re-encoded buffer.
    """
    atheris = importlib.import_module("atheris")
    providers: dict[int, Any] = {}
    out: list[Any] = []
    for r in schedule:
        pid = getattr(r, "provider", 0)
        if pid not in providers:
            # Mirror the harness: a new provider restarts from byte 0.
            providers[pid] = atheris.FuzzedDataProvider(bytes(data))
        out.append(getattr(providers[pid], r.method)(*r.args))
    return out


@dataclass
class RecodeRejection:
    """A solved value the codec could not write back into its field.

    A rejection is never a wrong seed — the codec refuses rather than emit bytes it
    cannot verify.  But the *reason* matters for knowing where the ceiling is, so each
    one records whether the gap is a missing encoder (``mechanically_possible=True`` —
    the bytes exist, we just cannot derive them yet) or a genuine limit of the field
    (``False`` — the harness's own width or range excludes the value, and no encoding
    would help).  Collecting these turns "not recodable" from a dead end into an
    inventory of what building the next encoder would buy.
    """

    field_index: int
    method: str                    # the FDP method that read this field
    target_repr: str               # repr of the value the solver asked for
    reason: str
    mechanically_possible: bool
    field_bytes: int = 0           # width the field consumed


# Float readers derive their value from consumed bytes deterministically, so an encoder
# is derivable — it just has not been reverse-engineered against atheris yet.
_FLOAT_METHODS = frozenset({"ConsumeFloat", "ConsumeFloatInRange", "ConsumeRegularFloat",
                            "ConsumeProbability"})

# ConsumeProbability / ConsumeFloatInRange read 8 bytes as a little-endian u64 and
# interpolate across their range; this is the denominator.
_U64_MAX = (1 << 64) - 1

# When an exact float is not representable, offer the nearest reachable value instead of
# refusing.  Sound because the caller verifies by running the guard, not by equality.
_FLOAT_NEAREST = True



def _decode_float_bytes(method: str, args: tuple, raw: bytes) -> Any:
    """What the REAL provider returns for *raw* — the oracle for float search."""
    atheris = importlib.import_module("atheris")
    return getattr(atheris.FuzzedDataProvider(raw), method)(*args)


def _search_float_bytes(method: str, args: tuple, nbytes: int, target: float):
    """Find bytes that make *method* decode to exactly *target*, by asking the oracle.

    ``ConsumeFloat`` (10 bytes) and ``ConsumeRegularFloat`` (9) are not a single lerp:
    trailing bytes select a *range*, and the leading eight interpolate within it, with
    special cases at the extremes.  Rather than hard-code a piecewise mapping that would
    be wrong in ways only a later verification could catch — and that would silently rot
    on an atheris upgrade — we invert the function by *querying the real one*: for each
    selector the payload is monotone, so bracket the target and bisect.

    This cannot lie: every probe is the actual decoder.  It returns None when the target
    simply is not in the field's representable set (2**64 points spread over the double
    range), along with the nearest value it found, which is honest information rather
    than a fabricated seed.
    """
    payload_len = 8
    sel_len = max(nbytes - payload_len, 0)
    if sel_len > 2:                      # unknown shape; do not guess
        return None, None

    def probe(payload: int, sel: bytes) -> Any:
        return _decode_float_bytes(method, args,
                                   payload.to_bytes(payload_len, "little") + sel)

    # Sample the selector space: one byte exhaustively, two on a coarse grid (the second
    # selector moves the range by orders of magnitude, so a stride is enough to bracket).
    if sel_len == 0:
        selectors = [b""]
    elif sel_len == 1:
        selectors = [bytes([i]) for i in range(256)]
    else:
        selectors = [bytes([i, j]) for j in range(0, 256, 8) for i in (0, 64, 128, 255)]

    best_bytes = best_val = None
    best_err = float("inf")
    for sel in selectors:
        try:
            lo_v, hi_v = probe(0, sel), probe(_U64_MAX, sel)
        except Exception:
            continue
        for cand_v, cand_p in ((lo_v, 0), (hi_v, _U64_MAX)):
            if isinstance(cand_v, float) and abs(cand_v - target) < best_err:
                best_err = abs(cand_v - target)
                best_bytes = cand_p.to_bytes(payload_len, "little") + sel
                best_val = cand_v
        if not (isinstance(lo_v, float) and isinstance(hi_v, float)):
            continue
        if not (min(lo_v, hi_v) <= target <= max(lo_v, hi_v)):
            continue

        # Monotone within this selector: bisect the payload.
        lo, hi = 0, _U64_MAX
        ascending = hi_v >= lo_v
        for _ in range(72):
            if lo >= hi:
                break
            mid = (lo + hi) // 2
            v = probe(mid, sel)
            if not isinstance(v, float):
                break
            if abs(v - target) < best_err:
                best_err = abs(v - target)
                best_bytes = mid.to_bytes(payload_len, "little") + sel
                best_val = v
            if v == target:
                return best_bytes, v
            if (v < target) == ascending:
                lo = mid + 1
            else:
                hi = mid
    return (best_bytes, best_val) if best_err == 0 else (None, best_val)


def _encode_field(rec: FieldRecord, target: Any) -> tuple[bytes | None, RecodeRejection | None]:
    """Bytes that make ``rec.method(*rec.args)`` decode to *target*.

    Returns ``(bytes, None)`` on success, or ``(None, RecodeRejection)`` explaining why
    the value could not be written back — classified by whether an encoder is derivable.
    """
    def reject(reason: str, possible: bool) -> tuple[None, RecodeRejection]:
        return None, RecodeRejection(
            field_index=rec.index, method=rec.method, target_repr=repr(target)[:120],
            reason=reason, mechanically_possible=possible, field_bytes=rec.nbytes,
        )

    n = rec.nbytes
    if n == 0:
        return reject("field consumed zero bytes (nothing to overwrite)", False)

    if rec.method in ("ConsumeInt", "ConsumeUInt"):
        signed = rec.method == "ConsumeInt"
        try:
            return int(target).to_bytes(n, "little", signed=signed), None
        except (OverflowError, ValueError):
            # The harness itself chose this width; no encoding fits a larger value.
            # Reaching here means the solver proposed a value the field cannot hold,
            # which it should not: field_bounds() feeds each field's domain into the
            # driver as param_bounds. Treat this as a bounds-plumbing signal, not a
            # codec gap — no encoder could ever help.
            return reject(f"value does not fit the harness's {n}-byte "
                          f"{'signed' if signed else 'unsigned'} field — the solver was "
                          f"not given this field's bounds (see field_bounds/param_bounds)",
                          False)

    if rec.method == "ConsumeIntInRange":
        lo, hi = rec.args
        span = hi - lo + 1
        if span <= 0 or not (lo <= target <= hi):
            # Same: with param_bounds wired, the solver never proposes outside [lo, hi].
            return reject(f"value outside the harness's own range [{lo}, {hi}] — the "
                          f"solver was not given this field's bounds "
                          f"(see field_bounds/param_bounds)", False)
        # FDP returns lo + (raw mod span); pick raw = target - lo (fits the width).
        raw = int(target) - lo
        try:
            return raw.to_bytes(n, "little"), None
        except (OverflowError, ValueError):
            return reject(f"offset {raw} does not fit {n} bytes", False)

    if rec.method == "ConsumeBool":
        return bytes([1 if target else 0]), None

    if rec.method == "ConsumeBytes":
        return (bytes(target) + b"\x00" * n)[:n], None

    if rec.method in ("ConsumeUnicodeNoSurrogates", "ConsumeUnicode", "ConsumeString"):
        if not isinstance(target, str):
            return reject(f"string field given a {type(target).__name__}", False)
        if target.isascii():
            # An ODD mode byte selects one byte per codepoint — but atheris masks each to
            # SEVEN bits, so the passthrough is 1:1 only for ASCII (0xE9 would decode as
            # 0x69, turning 'é' into 'i').  Anything above 0x7F takes the wide mode.
            # Both modes are variable width; encode_overrides rebuilds a front field.
            return b"\x01" + target.encode("ascii"), None
        else:
            # Beyond ASCII, use the wide mode: an even mode byte with (mode & 3) == 0
            # makes atheris read FOUR bytes per codepoint, little-endian, masked to 21
            # bits — so every scalar value round-trips exactly (verified against atheris
            # 3.0.0 across CJK, emoji and U+10FFF0).
            if any(0xD800 <= ord(c) <= 0xDFFF for c in target):
                return reject("surrogate codepoint; ConsumeUnicodeNoSurrogates cannot "
                              "produce one, so no encoding exists", False)
            return b"\x00" + b"".join(ord(c).to_bytes(4, "little") for c in target), None

    if rec.method in ("ConsumeProbability", "ConsumeFloatInRange"):
        # Both read 8 bytes as a little-endian u64 and linearly interpolate:
        #   ConsumeProbability     -> raw / (2**64 - 1)                  in [0, 1]
        #   ConsumeFloatInRange(a,b) -> a + (b - a) * raw / (2**64 - 1)
        # Verified against atheris 3.0.0 (0/0.25/0.5/1.0 of the range land exactly).
        span_lo, span_hi = (0.0, 1.0) if rec.method == "ConsumeProbability" else rec.args[:2]
        try:
            t = float(target)
        except (TypeError, ValueError):
            return reject(f"{rec.method} given a non-float", False)
        if span_hi == span_lo:
            frac = 0.0
        else:
            frac = (t - span_lo) / (span_hi - span_lo)
        if not (0.0 <= frac <= 1.0):
            return reject(f"value outside the field's own range [{span_lo}, {span_hi}]", False)
        raw = min(max(round(frac * _U64_MAX), 0), _U64_MAX)   # clamp: float*2**64 rounds up
        return raw.to_bytes(8, "little"), None

    if rec.method in _FLOAT_METHODS:
        # No closed form for these: invert by querying the real provider (see
        # _search_float_bytes).  Honest by construction — every probe is the decoder.
        found, nearest = _search_float_bytes(rec.method, rec.args, n, float(target))
        if found is not None:
            return found, None
        # The field holds only 2**64 points across the double range, so an arbitrary
        # target usually is not one of them. Most float guards are inequalities, and the
        # caller verifies by *executing the guard*, not by comparing values — so offer
        # the nearest reachable value rather than refusing outright. The substitution is
        # always recorded, never silent.
        # Only substitute when the nearest is genuinely close. The representable points
        # are spread across +/-1.8e308, so the "nearest" to 42.5 can be 0.0 — technically
        # true, useless in practice, and it would burn a verification round.
        close = (nearest is not None and
                 abs(nearest - float(target)) <= 1e-6 * max(abs(float(target)), 1e-300))
        if _FLOAT_NEAREST and close:
            approx, _ = _search_float_bytes(rec.method, rec.args, n, float(nearest))
            if approx is not None:
                return approx, RecodeRejection(
                    field_index=rec.index, method=rec.method, target_repr=repr(target)[:120],
                    reason=f"exact value not representable; substituted the nearest "
                           f"reachable {nearest!r} (the guard, not the value, is the test)",
                    mechanically_possible=True, field_bytes=rec.nbytes,
                )
        return reject(f"{rec.method} cannot represent {target!r}; its representable set is "
                      f"2**64 points across the double range and the nearest reachable "
                      f"value is {nearest!r}", False)

    return reject(f"no encoder implemented for {rec.method}", True)


def _rebuild_buffer(
    original: bytes, schedule: list[FieldRecord], encoded: dict[int, bytes],
    rejections: list[RecodeRejection] | None = None,
) -> bytes | None:
    """Reassemble the buffer, allowing fields at either end to change width.

    An FDP buffer has two growing regions.  Front fields fill from index 0 in
    consumption order.  Back fields anchor to the *end*: the first back consume takes the
    final bytes, the next takes the bytes before those, and so on — so read
    left-to-right the back region is the back fields in **reverse** consumption order.
    An unused middle may sit between.

    Both regions are rebuilt from their field bytes, which lets either side resize: a
    longer string at the front, or a wider float at the back.  Only records from
    provider 0 lay out the buffer — a harness that builds several providers has them
    read the *same* bytes, so the layouts would contradict each other.  In that case we
    patch in place instead, and the full replay in :func:`encode_overrides` is what
    catches any interference.
    """
    # With several providers the buffer has several overlapping layouts. Lay it out for
    # the provider that owns the fields being changed: the result is still a perfectly
    # valid input, the other providers simply read different values from it. That is not
    # a lie — it is a different input — and the caller verifies the fields it targeted.
    providers = {getattr(r, "provider", 0) for r in schedule}
    owner = min({getattr(r, "provider", 0) for r in schedule if r.index in encoded},
                default=0) if len(providers) > 1 else None
    layout = [r for r in schedule
              if owner is None or getattr(r, "provider", 0) == owner]

    front = [r for r in layout if r.end == "front"]
    back = [r for r in layout if r.end == "back"]
    front.sort(key=lambda r: r.byte_range[0])
    back.sort(key=lambda r: r.index)            # consumption order

    def piece(r: FieldRecord) -> bytes:
        return encoded.get(r.index, original[r.byte_range[0]:r.byte_range[1]])


    front_bytes = b"".join(piece(r) for r in front)
    # First-consumed back field owns the final bytes, so lay them out in reverse.
    back_bytes = b"".join(piece(r) for r in reversed(back))

    # Preserve whatever unused middle the seed had, so untouched bytes stay untouched.
    old_front_end = max((r.byte_range[1] for r in front), default=0)
    old_back_start = min((r.byte_range[0] for r in back), default=len(original))
    middle = original[old_front_end:old_back_start] if old_back_start >= old_front_end else b""

    return front_bytes + middle + back_bytes


def widen_seed(harness_fn, data: bytes, *, max_size: int = 8192) -> bytes:
    """Grow *data* until the harness's fields stop starving.

    An FDP string field is greedy: ``ConsumeUnicodeNoSurrogates(256)`` takes everything
    available until its own count binds.  So on a short seed the first field swallows the
    buffer and every later field consumes **zero** bytes — and a zero-width field cannot
    be written to, because there is nothing to overwrite and nowhere to put it.

    That is not a codec gap, it is a seed that is too small for the harness the author
    wrote.  Measured on ntlm_auth's OSS harness: at 64 bytes, 17 of 21 fields are
    starved; at 4096 bytes, none are.  Growing the seed makes the other 17 addressable.

    The filler cycles the seed's own bytes, so widening is deterministic and a widened
    seed is still just an ordinary input the fuzzer could have produced itself.
    """
    def starved(buf: bytes) -> int:
        try:
            return sum(1 for r in record_schedule(harness_fn, buf) if r.nbytes == 0)
        except Exception:
            return -1

    best, best_starved = bytes(data), starved(data)
    if best_starved <= 0:                    # already saturated, or unrecordable
        return best

    pattern = bytes(data) or bytes(range(1, 251))
    size = max(len(data), 64)
    while size < max_size:
        size *= 2
        grown = (pattern * (size // len(pattern) + 1))[:size]
        n = starved(grown)
        if n < 0:
            break
        if n < best_starved:
            best, best_starved = grown, n
            if n == 0:
                break
    return best


def seed_with_field_width(harness_fn, data: bytes, field_index: int, want: int,
                          *, max_size: int = 16384) -> bytes | None:
    """A buffer on which field *field_index* consumes about *want* bytes.

    Length guards are decided by a field's WIDTH, and a field's width is decided by how
    much buffer is left when the harness reaches it — not by anything the solver can put
    in the field. `if len(key) != 7` on `lm_hash[:7]` is satisfied for free when lm_hash
    is starved to b'', and unsatisfiable while it is 124 bytes. No flipper can express
    that, because width is not a value.

    So it is searched for directly, with the real provider as the oracle: try buffer
    sizes, record the schedule each time, and keep the one whose field lands closest to
    the requested width. Truncation gives the fine control that doubling alone does not.
    """
    best: bytes | None = None
    best_err = None
    pattern = bytes(data) or bytes(range(1, 251))

    def probe(buf: bytes) -> int | None:
        try:
            sched = record_schedule(harness_fn, buf)
        except Exception:
            return None
        return sched[field_index].nbytes if field_index < len(sched) else None

    size = 8
    while size <= max_size:
        buf = (pattern * (size // len(pattern) + 1))[:size]
        got = probe(buf)
        if got is not None:
            err = abs(got - want)
            if best_err is None or err < best_err:
                best, best_err = buf, err
                if err == 0:
                    return best
        size *= 2
    return best


def encode_overrides(
    original: bytes, schedule: list[FieldRecord], overrides: dict[int, Any],
    rejections: list[RecodeRejection] | None = None,
    *, verify_only_targeted: bool = False,
    accept: "Callable[[bytes], bool] | None" = None,
) -> bytes | None:
    """Return a new buffer where each field in *overrides* decodes to its new value.

    ``overrides`` maps a :class:`FieldRecord` ``index`` to a target typed value.  When
    every override keeps its field's byte width the buffer is patched in place; when a
    front field resizes (e.g. a longer string) the buffer is rebuilt (:func:`_rebuild_buffer`).
    Either way the result is verified by replaying it through a fresh real FDP — every
    overridden field must decode to its target and every other field must be unchanged.
    Returns None if any override is unsupported or verification fails.

    Pass a list as ``rejections`` to collect :class:`RecodeRejection` records explaining
    every refusal — including whether an encoder is derivable — instead of only seeing
    ``None``.
    """
    encoded: dict[int, bytes] = {}
    substituted: dict[int, Any] = {}   # field -> what we actually encoded, when != asked
    for rec in schedule:
        if rec.index not in overrides:
            continue
        enc, rej = _encode_field(rec, overrides[rec.index])
        if enc is None:
            if rejections is not None and rej is not None:
                rejections.append(rej)
            return None
        if rej is not None:
            # A successful encode that carries a note: the exact value was not
            # representable and the nearest was used. Record it — never silent — and
            # verify against what we actually wrote.
            if rejections is not None:
                rejections.append(rej)
            try:
                substituted[rec.index] = _decode_float_bytes(rec.method, rec.args, enc)
            except Exception:
                return None
        encoded[rec.index] = enc
    if not encoded:
        if overrides:
            # Every requested index was absent from the schedule — returning the seed
            # unchanged would read as success while changing nothing.
            if rejections is not None:
                rejections.append(RecodeRejection(
                    field_index=next(iter(overrides)), method="<absent>", target_repr="",
                    reason="no recorded field with that index; the schedule may have been "
                           "truncated by a harness exception",
                    mechanically_possible=False,
                ))
            return None
        return bytes(original)

    same_width = all(
        len(encoded[r.index]) == r.byte_range[1] - r.byte_range[0]
        for r in schedule if r.index in encoded
    )
    if same_width:
        buf = bytearray(original)
        for r in schedule:
            if r.index in encoded:
                lo, hi = r.byte_range
                buf[lo:hi] = encoded[r.index]
        new = bytes(buf)
    else:
        rebuilt = _rebuild_buffer(original, schedule, encoded, rejections)
        if rebuilt is None:
            return None
        new = rebuilt

    # Verify against the real decoder — execution is the judge.
    try:
        decoded = replay_values(new, schedule)
    except Exception as exc:
        if rejections is not None:
            rejections.append(RecodeRejection(
                field_index=-1, method="<replay>", target_repr="",
                reason=f"re-decode of the rebuilt buffer raised {type(exc).__name__}",
                mechanically_possible=True,
            ))
        return None
    mismatched = False
    for rec in schedule:
        if verify_only_targeted and rec.index not in overrides:
            # Another provider's view of the same bytes may legitimately change; only the
            # fields we asked for must hold.
            continue
        want = substituted.get(rec.index, overrides.get(rec.index, rec.value))
        if decoded[rec.index] != want:
            mismatched = True
            if rejections is not None:
                # A field shifted: overriding one value perturbed how a later field reads.
                rejections.append(RecodeRejection(
                    field_index=rec.index, method=rec.method,
                    target_repr=repr(want)[:120],
                    reason=f"re-decoded to {decoded[rec.index]!r}; an earlier override "
                           f"perturbed this field's bytes",
                    mechanically_possible=True, field_bytes=rec.nbytes,
                ))
    if mismatched:
        # Exact-match failed — a mid-buffer field decodes to the solved value PLUS trailing
        # bytes the following fields need. For a prefix / substring / regex-prefix guard that
        # extra content is harmless: the solution still satisfies it. Rather than discard a
        # seed that would work, let the caller decide by EXECUTION — if ``accept(new)``
        # confirms the reassembled buffer still reaches the target, the seed is good however
        # the exact field bytes read. This is the driver's "execution is the judge" rule
        # applied to recode. With no ``accept`` the strict exact-match contract is unchanged.
        if accept is not None and accept(new):
            return new
        return None
    return new


def record_schedule(harness_fn, data: bytes) -> list[FieldRecord]:
    """Run *harness_fn(data)* with ``atheris.FuzzedDataProvider`` replaced by
    :class:`RecordingFDP`, returning the captured consumption schedule.

    *harness_fn* is the OSS harness (``TestOneInput``-style) unchanged; it constructs
    ``atheris.FuzzedDataProvider(data)`` internally.  Exceptions from the harness body
    are swallowed — a crash still yields the schedule consumed up to that point.
    """
    atheris = importlib.import_module("atheris")
    real_ctor = atheris.FuzzedDataProvider
    records: list[FieldRecord] = []
    made = [0]

    def _ctor(passed):
        # EVERY construction gets its own recorder. A real OSS-Fuzz harness often builds
        # several providers over the same buffer (ntlm_auth builds three), and each one
        # restarts at byte 0 — handing back a shared, partially-consumed recorder
        # recorded wrong values and wrong byte ranges.
        r = RecordingFDP(passed, provider=made[0], sink=records, real_ctor=real_ctor)
        made[0] += 1
        return r

    atheris.FuzzedDataProvider = _ctor
    raised: Exception | None = None
    try:
        harness_fn(data)
    except Exception as exc:
        # A crash still yields the fields consumed up to that point — but a TRUNCATED
        # schedule looks exactly like a legitimately short one, and acting on it models
        # the harness wrongly.  Record what happened so callers can tell.
        raised = exc
    finally:
        atheris.FuzzedDataProvider = real_ctor
    record_schedule.last_exception = raised   # callers can tell truncation from brevity
    return records
