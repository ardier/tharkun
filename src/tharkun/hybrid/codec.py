"""Pluggable bytes <-> parameters codecs for the assist engine.

A fuzzer hands the harness raw ``bytes``; tharkun solves over typed parameters. Some
codec has to bridge the two, and which one depends on how the harness was written:

* :class:`FdpCodec` — the harness is a real OSS-Fuzz one built on
  ``atheris.FuzzedDataProvider``. We keep it *unmodified* and recover the mapping by
  recording the FDP consumption schedule, using the real provider as the oracle. This is
  faithful but inherits FDP's quirks, and the schedule is positional, so it assumes the
  harness passes its lifted fields in order to the target.

* :class:`FunctionCodec` — the harness is hand-written with an explicit, invertible wire
  format and exposes ``decode(bytes) -> {param: value}`` plus ``encode({param: value})
  -> bytes``. Nothing is inferred: the harness *states* the mapping. That removes the
  positional assumption, the multi-provider ambiguity, and the unencodable-field classes
  in one go, at the cost of writing a few lines per target.

Both are verified the same way — encode, then decode the result and check every requested
parameter actually came back. A codec never returns bytes it has not round-tripped.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Codec(Protocol):
    """Maps a fuzzer input buffer to solver parameters and back."""

    def decode(self, data: bytes) -> dict[str, Any]:
        """Parameter values this buffer produces, keyed by target-parameter name."""

    def encode(self, base: bytes, params: dict[str, Any]) -> bytes | None:
        """A buffer that decodes to *params*, or None if it cannot be represented.

        *base* is the seed being modified, so a codec that can only patch in place has
        something to patch. A codec that can build a buffer outright may ignore it.
        """

    def bounds(self, data: bytes) -> dict[str, tuple[int, int]]:
        """Inclusive integer ranges per parameter, so the solver stays encodable."""


class FunctionCodec:
    """Codec backed by ``decode`` / ``encode`` functions the harness supplies.

    The contract is deliberately small — a hand-written harness declares its own wire
    format, so there is nothing to infer and nothing to get wrong:

        def decode(data: bytes) -> dict:   # {"magic": 1, "name": "x"}
        def encode(fields: dict) -> bytes

    ``bounds_fn`` is optional; supply it when a field has a fixed width so the solver
    does not return a value the format cannot hold.
    """

    def __init__(self, decode_fn, encode_fn, bounds_fn=None) -> None:
        self._decode = decode_fn
        self._encode = encode_fn
        self._bounds = bounds_fn

    def decode(self, data: bytes) -> dict[str, Any]:
        try:
            out = self._decode(data)
        except Exception:
            return {}
        return dict(out) if isinstance(out, dict) else {}

    def encode(self, base: bytes, params: dict[str, Any]) -> bytes | None:
        try:
            new = self._encode(dict(params))
        except Exception:
            return None
        if not isinstance(new, (bytes, bytearray)):
            return None
        new = bytes(new)
        # Round-trip check: the harness's own decode is the judge, exactly as the real
        # FuzzedDataProvider is for FdpCodec. A format that cannot represent the value
        # is rejected rather than silently emitting a buffer that decodes to something
        # else.
        back = self.decode(new)
        for k, v in params.items():
            if k not in back or back[k] != v:
                return None
        return new

    def bounds(self, data: bytes) -> dict[str, tuple[int, int]]:
        if self._bounds is None:
            return {}
        try:
            return dict(self._bounds(data) or {})
        except Exception:
            return {}


class FdpCodec:
    """Codec over an unmodified ``atheris.FuzzedDataProvider`` harness.

    Wraps the recording/re-encoding in :mod:`tharkun.hybrid.fdp_codec`. The schedule is
    positional, so ``param_names`` maps field *i* to the target's *i*-th parameter unless
    ``field_to_params`` overrides it.
    """

    def __init__(self, harness_fn, param_names: list[str], field_to_params=None) -> None:
        self._harness = harness_fn
        self._param_names = param_names
        self._field_to_params = field_to_params
        self._schedules: dict[bytes, Any] = {}

    def _schedule(self, data: bytes):
        from tharkun.hybrid.fdp_codec import record_schedule

        key = bytes(data)
        if key not in self._schedules:
            self._schedules[key] = record_schedule(self._harness, key)
        return self._schedules[key]

    def _map(self, fields: list[Any]) -> dict[str, Any]:
        if self._field_to_params is not None:
            return self._field_to_params(fields)
        return {n: fields[i] for i, n in enumerate(self._param_names) if i < len(fields)}

    def decode(self, data: bytes) -> dict[str, Any]:
        return self._map([r.value for r in self._schedule(data)])

    def encode(self, base: bytes, params: dict[str, Any]) -> bytes | None:
        from tharkun.hybrid.fdp_codec import encode_overrides

        schedule = self._schedule(base)
        current = [r.value for r in schedule]
        overrides: dict[int, Any] = {}
        for i, name in enumerate(self._param_names):
            if name in params and i < len(current) and params[name] != current[i]:
                overrides[i] = params[name]
        if not overrides:
            return None
        return encode_overrides(base, schedule, overrides, verify_only_targeted=True)

    def encode_with_rejections(self, base: bytes, params: dict[str, Any], rejections: list):
        """``encode`` that also reports why a field could not be written back."""
        from tharkun.hybrid.fdp_codec import encode_overrides

        schedule = self._schedule(base)
        current = [r.value for r in schedule]
        overrides = {i: params[n] for i, n in enumerate(self._param_names)
                     if n in params and i < len(current) and params[n] != current[i]}
        if not overrides:
            return None
        return encode_overrides(base, schedule, overrides, rejections,
                                verify_only_targeted=True)

    def bounds(self, data: bytes) -> dict[str, tuple[int, int]]:
        from tharkun.hybrid.fdp_codec import field_bounds

        schedule = self._schedule(data)
        out: dict[str, tuple[int, int]] = {}
        for i, name in enumerate(self._param_names):
            if i < len(schedule):
                b = field_bounds(schedule[i])
                if b is not None:
                    out[name] = b
        return out


def codec_for(harness_module, param_names: list[str], field_to_params=None) -> Codec:
    """Pick a codec by what the harness offers.

    A hand-written ``decode``/``encode`` pair wins when present: it states the mapping
    outright, so there is nothing to infer. Otherwise fall back to recording the FDP.
    """
    dec = getattr(harness_module, "decode", None)
    enc = getattr(harness_module, "encode", None)
    if callable(dec) and callable(enc):
        return FunctionCodec(dec, enc, getattr(harness_module, "bounds", None))

    fn = getattr(harness_module, "TestOneInput", None) or getattr(harness_module, "TestInput", None)
    return FdpCodec(fn, param_names, field_to_params)
