"""Tests for the pluggable bytes<->parameters codecs.

FunctionCodec is pure stdlib and runs everywhere; the FDP selection test needs atheris.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import textwrap
import unittest

from tharkun.hybrid.codec import FunctionCodec, codec_for
from tharkun.hybrid.fdp_codec import atheris_available

requires_atheris = unittest.skipUnless(atheris_available(), "atheris not installed")


def _decode(data: bytes) -> dict:
    data = (data + b"\x00" * 5)[:max(len(data), 5)]
    return {"magic": int.from_bytes(data[:4], "little"),
            "tag": data[4],
            "name": data[5:].decode("utf-8", "replace")}


def _encode(fields: dict) -> bytes:
    return (int(fields["magic"]).to_bytes(4, "little")
            + bytes([int(fields["tag"]) & 0xFF])
            + str(fields["name"]).encode("utf-8"))


class TestFunctionCodec(unittest.TestCase):
    def setUp(self) -> None:
        self.c = FunctionCodec(_decode, _encode)

    def test_round_trip(self) -> None:
        out = self.c.encode(b"\x00" * 8, {"magic": 3405691582, "tag": 13, "name": "admin"})
        self.assertIsNotNone(out)
        self.assertEqual(self.c.decode(out),
                         {"magic": 3405691582, "tag": 13, "name": "admin"})

    def test_encode_rejects_what_the_format_cannot_hold(self) -> None:
        # 2**40 does not fit the 4-byte magic field; the round-trip check catches it
        # rather than emitting a buffer that decodes to something else.
        self.assertIsNone(self.c.encode(b"\x00" * 8,
                                        {"magic": 2 ** 40, "tag": 0, "name": ""}))

    def test_decode_of_garbage_is_empty_not_a_crash(self) -> None:
        bad = FunctionCodec(lambda d: 1 / 0, _encode)
        self.assertEqual(bad.decode(b"x"), {})

    def test_encode_failure_is_none_not_a_crash(self) -> None:
        bad = FunctionCodec(_decode, lambda f: 1 / 0)
        self.assertIsNone(bad.encode(b"\x00" * 8, {"magic": 1, "tag": 1, "name": "a"}))

    def test_encode_must_return_bytes(self) -> None:
        bad = FunctionCodec(_decode, lambda f: "not bytes")
        self.assertIsNone(bad.encode(b"\x00" * 8, {"magic": 1, "tag": 1, "name": "a"}))

    def test_bounds_optional(self) -> None:
        self.assertEqual(self.c.bounds(b"\x00" * 8), {})
        with_bounds = FunctionCodec(_decode, _encode, lambda d: {"tag": (0, 255)})
        self.assertEqual(with_bounds.bounds(b"")["tag"], (0, 255))


class TestCodecSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.d = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.d, ignore_errors=True)

    def _module(self, src: str):
        import importlib.util
        p = os.path.join(self.d, "h.py")
        with open(p, "w") as f:
            f.write(textwrap.dedent(src))
        spec = importlib.util.spec_from_file_location("_codec_fixture", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_hand_written_pair_wins(self) -> None:
        mod = self._module("""
            def decode(data):
                return {"x": data[0] if data else 0}
            def encode(fields):
                return bytes([fields["x"] & 0xFF])
            def TestOneInput(data):
                pass
        """)
        self.assertIsInstance(codec_for(mod, ["x"]), FunctionCodec)

    @requires_atheris
    def test_falls_back_to_fdp(self) -> None:
        from tharkun.hybrid.codec import FdpCodec
        mod = self._module("""
            import atheris
            def TestOneInput(data):
                fdp = atheris.FuzzedDataProvider(data)
                return fdp.ConsumeUInt(4)
        """)
        self.assertIsInstance(codec_for(mod, ["x"]), FdpCodec)


if __name__ == "__main__":
    unittest.main()
