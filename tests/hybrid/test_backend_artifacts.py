"""libFuzzer's crash-/leak-/timeout-/oom-/slow-unit- artifacts must never land in the CWD of whoever launched a run.

``hybrid/backend.py::AtherisBackend`` is the one place Tharkun starts Atheris (the SV hybrid stage, ``HybridFuzzer``,
the benchmark): it always passes ``-artifact_prefix=<dir>/``, to the caller's directory or to a private temporary
one it removes at ``stop()``; ``sv/hybrid.py`` runs the worker in its own scratch directory.
"""
from __future__ import annotations

import glob
import os
import textwrap
import time

import pytest

from tharkun.hybrid.backend import AtherisBackend
from tharkun.hybrid.fdp_codec import atheris_available
from tharkun.sv.engine import verify

LITTER = ("crash-*", "leak-*", "timeout-*", "oom-*", "slow-unit-*", "fuzz-*.log")


def _litter(directory):
    return sorted(p for pattern in LITTER for p in glob.glob(os.path.join(str(directory), pattern)))


def test_the_command_always_names_an_artifact_prefix_ending_in_a_separator(tmp_path):
    b = AtherisBackend("h.py", str(tmp_path / "corpus"), extra_args=["-runs=1"])
    prefix = [a for a in b.command() if a.startswith("-artifact_prefix=")]
    assert len(prefix) == 1 and prefix[0].endswith(os.sep) and b.artifact_dir in prefix[0]
    mine = AtherisBackend("h.py", str(tmp_path / "corpus"), artifact_dir=str(tmp_path / "arts"))
    assert f"-artifact_prefix={tmp_path / 'arts'}{os.sep}" in mine.command()


def test_a_prefix_the_caller_already_gave_is_not_overridden(tmp_path):
    b = AtherisBackend("h.py", str(tmp_path / "c"), extra_args=["-artifact_prefix=/somewhere/else/"])
    assert [a for a in b.command() if a.startswith("-artifact_prefix=")] == ["-artifact_prefix=/somewhere/else/"]


def _crashing_harness(tmp_path):
    h = tmp_path / "harness.py"
    h.write_text(textwrap.dedent("""
        import sys
        import atheris

        def TestOneInput(data):
            if data.startswith(b"boom"):
                raise RuntimeError("planted")

        atheris.Setup(sys.argv, TestOneInput)
        atheris.Fuzz()
    """))
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "seed").write_bytes(b"boom")
    return h, corpus


def _run_until_exit(backend, seconds=20):
    backend.start()
    deadline = time.monotonic() + seconds
    while backend.is_running() and time.monotonic() < deadline:
        time.sleep(0.1)


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_a_crash_artifact_goes_to_the_private_directory_and_is_removed_at_stop(tmp_path, monkeypatch):
    h, corpus = _crashing_harness(tmp_path)
    cwd = tmp_path / "launched-from"
    cwd.mkdir()
    monkeypatch.chdir(cwd)                                       # where a lane's run used to leave them
    b = AtherisBackend(str(h), str(corpus))
    _run_until_exit(b)
    assert glob.glob(os.path.join(b.artifact_dir, "crash-*")), "the planted crash was not written where asked"
    assert _litter(cwd) == [] and _litter(tmp_path) == []
    b.stop()
    assert not os.path.exists(b.artifact_dir)


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_a_caller_owned_artifact_directory_is_kept(tmp_path, monkeypatch):
    h, corpus = _crashing_harness(tmp_path)
    monkeypatch.chdir(tmp_path)
    arts = tmp_path / "arts"
    b = AtherisBackend(str(h), str(corpus), artifact_dir=str(arts))
    _run_until_exit(b)
    b.stop()
    assert glob.glob(str(arts / "crash-*")) and _litter(tmp_path) == []


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_a_hybrid_stage_run_that_crashes_the_worker_leaves_nothing_in_its_cwd(tmp_path, monkeypatch):
    """The program segfaults on one literal it carries: libFuzzer catches the signal and writes ``crash-*``.
    The SV stage runs the worker in its own temporary directory, so it must not appear in the launch directory."""
    prog = tmp_path / "boom.py"
    prog.write_text(textwrap.dedent("""
        import ctypes
        import _sv_verifier
        if __name__ == "__main__":
            s = _sv_verifier.nondet_str()
            if s == "boom":
                ctypes.string_at(0)
            assert len(s) >= 0          # an assert gives the hybrid stage an obligation to fuzz toward
    """))
    monkeypatch.chdir(tmp_path)
    verify(str(prog), ("AssertionError",), jobs=1, budget_seconds=20, prove=False, concolic=False,
           stages=["hybrid"])
    assert _litter(tmp_path) == []
    assert _litter(os.getcwd()) == []
