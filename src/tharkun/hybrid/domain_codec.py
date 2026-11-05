"""Fuzzer bytes <-> members of a ``strategies.py`` domain: the LIMIT half of limit-then-steer.

A *domain* is a ``tharkun.strategies.make_strategy`` constraint dict (owner 2026-10-23: "a compound approach that
limits the answer partially given generators and still steers them given the feedback signal").  The coverage-guided
fuzzer mutates raw bytes, so something has to turn those bytes into a member of the domain, every time.

The mechanism already exists and is not rebuilt here: Hypothesis's own ``fuzz_one_input`` (public API; its docs name
Atheris/libFuzzer as the use case) interprets a byte buffer as the choice sequence of a ``@given`` test.  Every buffer
that Hypothesis accepts decodes to a value the strategy can draw, by construction, and the fuzzer's byte mutations are
mutations of that choice sequence.  This module only wires the two together:

* the generator is ``strategies.make_strategy`` (the ONE generation subsystem; nothing here defines a strategy),
* the parameters are combined as ONE constraint dict, ``{"type": "tuple", "items": [...]}``, so the combination is also
  ``make_strategy``'s,
* the decoder is the real thing, so a decode that is not a member cannot occur; a buffer Hypothesis refuses (too short,
  filtered out) decodes to ``None`` and the caller counts it as a rejected execution.

Only decoding is built.  The reverse direction for a *new* value (a solver's proposal) needs a choice-sequence encoder per
strategy, which Hypothesis does not expose; :meth:`DomainCodec.canonical` and :meth:`DomainCodec.seeds` give the
reverse direction for values the generator itself produced, which is what seeding a corpus needs.
"""

from __future__ import annotations

import random
from typing import Any, Mapping


class DomainCodec:
    """Decode fuzzer buffers into ``{param: value}`` members of ``domains`` (one constraint dict per parameter)."""

    def __init__(self, domains: Mapping[str, dict]) -> None:
        from hypothesis import HealthCheck, given, settings

        from tharkun.strategies import make_strategy

        self.names = tuple(domains)
        if not self.names:
            raise ValueError("a domain needs at least one parameter")
        self.domains = {name: domains[name] for name in self.names}
        #: the combined domain: still a constraint dict, still drawn by strategies.make_strategy
        self.domain = {"type": "tuple", "items": [self.domains[name] for name in self.names]}
        slot: list = []

        @settings(database=None, deadline=None, suppress_health_check=list(HealthCheck))
        @given(make_strategy(self.domain))
        def _member(values: tuple) -> None:
            slot.append(values)

        self._slot = slot
        self._fuzz = _member.hypothesis.fuzz_one_input

    def decode_canonical(self, data: bytes) -> tuple[dict[str, Any], bytes] | None:
        """``(member, canonical_bytes)`` for ``data``, or None when Hypothesis rejects the buffer.

        ``canonical_bytes`` is the pruned buffer that replays the same member (Hypothesis's own return value)."""
        self._slot.clear()
        canonical = self._fuzz(bytes(data))
        if canonical is None or not self._slot:
            return None
        return dict(zip(self.names, self._slot[0])), bytes(canonical)

    def decode(self, data: bytes) -> dict[str, Any] | None:
        """The member ``data`` decodes to, or None for a rejected buffer."""
        got = self.decode_canonical(data)
        return None if got is None else got[0]

    def canonical(self, data: bytes) -> bytes | None:
        """The pruned buffer that replays ``data``'s member (None for a rejected buffer)."""
        got = self.decode_canonical(data)
        return None if got is None else got[1]

    def seeds(self, n: int, seed: int, *, tries: int = 400) -> list[bytes]:
        """Up to ``n`` canonical buffers of DISTINCT members: the simplest member first (all-zero buffer, Hypothesis's
        simplest choice sequence), then members decoded from ``random.Random(seed)`` buffers.  Deterministic in ``seed``.

        These are the generator's own draws written back as bytes: a corpus that starts inside the domain."""
        rng = random.Random(seed)
        out: list[bytes] = []
        seen: set[bytes] = set()
        candidates = (b"\x00" * 64, *(rng.randbytes(rng.randrange(8, 160)) for _ in range(tries)))
        for buf in candidates:
            got = self.decode_canonical(buf)
            if got is None or got[1] in seen:
                continue
            seen.add(got[1])
            out.append(got[1])
            if len(out) >= n:
                break
        return out
