"""The hybrid worker runs the program FOR REAL, so it runs it under the engine's environment model.

Before this, a verify with a few seconds of budget on a program that truncates a file it does not own
truncated that file on the host (``tests/test_sv_environment.py`` only passed because its 3 s budget made the
stage's own budget fall below the 1 s the stage needs to start the worker).
"""
from __future__ import annotations

import textwrap

import pytest

from tharkun.hybrid.fdp_codec import atheris_available
from tharkun.sv.engine import verify

pytestmark = pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")


def program(tmp_path, source):
    p = tmp_path / "t.py"
    p.write_text(textwrap.dedent(source))
    return str(p)


def test_a_program_that_truncates_a_file_it_does_not_own_leaves_it_alone(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"fixed")
    p = program(tmp_path, f"""
        import io
        import _sv_verifier
        if __name__ == "__main__":
            s = _sv_verifier.nondet_str()
            try:
                io.FileIO({str(outside)!r}, "wb")
            except OSError:
                raise IndexError
    """)
    v = verify(p, ("IndexError",), jobs=1, budget_seconds=10, prove=False, concolic=False, stages=["hybrid"])
    assert v.verdict != "false"                       # a refused write is not a violation of the model
    assert outside.read_bytes() == b"fixed"


def test_os_level_and_pathlib_writes_outside_the_scratch_are_refused_too(tmp_path):
    outside = tmp_path / "victim.txt"
    outside.write_text("keep")
    p = program(tmp_path, f"""
        import os
        import _sv_verifier
        if __name__ == "__main__":
            s = _sv_verifier.nondet_str()
            assert len(s) >= 0
            import pathlib
            for op in (lambda: os.open({str(outside)!r}, os.O_WRONLY | os.O_TRUNC),
                       lambda: open({str(outside)!r}, "w"),
                       lambda: pathlib.Path({str(outside)!r}).write_text("overwritten")):
                try:
                    op()
                except OSError:
                    pass
    """)
    verify(p, ("AssertionError",), jobs=1, budget_seconds=10, prove=False, concolic=False, stages=["hybrid"])
    assert outside.read_text() == "keep"


def test_a_program_that_reads_stdin_does_not_read_the_verifiers(tmp_path, monkeypatch):
    import subprocess
    import sys
    code = textwrap.dedent(f"""
        import sys
        from tharkun.sv.engine import verify
        v = verify({program(tmp_path, '''
            import _sv_verifier
            if __name__ == "__main__":
                s = _sv_verifier.nondet_str()
                assert len(s) >= 0
                data = input()
        ''')!r}, ("AssertionError",), jobs=1, budget_seconds=60, prove=False, concolic=False, stages=["hybrid"])
        print("VERDICT", v.verdict)
    """)
    # the verifier's own stdin carries a line that must never reach the program
    r = subprocess.run([sys.executable, "-c", code], input="from-the-verifiers-stdin\n", capture_output=True,
                       text=True, timeout=120)
    assert "VERDICT" in r.stdout
    assert "from-the-verifiers-stdin" not in r.stdout + r.stderr
