"""Fuzzer backends: run the coverage-guided fuzzer in the background (step 1).

The orchestrator talks to a backend through a tiny interface — start, is_running,
coverage, stop — so the assist loop is independent of which fuzzer drives it.

:class:`AtherisBackend` runs the *real, unmodified* OSS harness as an Atheris/libFuzzer
subprocess against a corpus directory, with ``-reload=1`` so seeds the assist writes to
that directory are picked up mid-run (the same mechanism parallel libFuzzer workers use
to share finds — no restart needed).  It parses libFuzzer's ``cov:`` counter off stderr
so the orchestrator can detect a coverage plateau.

:class:`FakeBackend` scripts a coverage sequence for deterministic orchestrator tests.

**No litter.** libFuzzer writes ``crash-*``, ``leak-*``, ``timeout-*``, ``oom-*`` and ``slow-unit-*``
artifacts (and ``fuzz-N.log`` with ``-jobs``) wherever ``-artifact_prefix`` points -- by default the subprocess's
CURRENT DIRECTORY, so a run launched from a campaign's results dir, a clone or a scratch dir left them there.
:class:`AtherisBackend` therefore always passes ``-artifact_prefix=<dir>/``: the caller's ``artifact_dir`` (the
caller owns and removes it, typically a ``TemporaryDirectory`` of its own) or a private temporary directory the
backend creates at ``start()`` and removes at ``stop()``.  ``cwd`` runs the subprocess in a directory of the
caller's choosing (the SV stage runs it in its scratch directory, like every engine child).

Both backends also report ``cpu_seconds()``: the subprocess's own process CPU time
(user+system), not wall-clock -- what BenchExec measures and SV-COMP ties break on
([internal ref] part 2). A caller that budgets its own polling loop by ``cpu_seconds()`` instead
of ``time.monotonic()`` gets a decision that does not depend on how busy the host is.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from typing import Sequence

_COV_RE = re.compile(rb"cov:\s*(\d+)")


class AtherisBackend:
    """Drive an Atheris harness subprocess over a corpus directory."""

    def __init__(
        self,
        harness_path: str,
        corpus_dir: str,
        *,
        python: str | None = None,
        extra_args: Sequence[str] = (),
        reload: bool = True,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        artifact_dir: str | None = None,
    ) -> None:
        self.harness_path = os.path.realpath(harness_path)
        self.corpus_dir = os.path.realpath(corpus_dir)
        self.python = python or sys.executable
        self.extra_args = list(extra_args)
        self.reload = reload
        self.env = env
        self.cwd = cwd
        # where libFuzzer's crash-/leak-/timeout-/oom-/slow-unit- artifacts go: the caller's directory (its to
        # remove), or a private one this backend creates at start() and removes at stop()
        self._owns_artifacts = artifact_dir is None
        self.artifact_dir = os.path.realpath(artifact_dir) if artifact_dir is not None else os.path.join(
            tempfile.gettempdir(), f"tharkun-atheris-{uuid.uuid4().hex[:12]}")
        self._proc: subprocess.Popen | None = None
        self._cov = 0
        self._reader: threading.Thread | None = None
        self._cpu = 0.0

    def command(self) -> list[str]:
        """The subprocess argv (also handy to inspect/test without launching)."""
        args = [self.python, self.harness_path, self.corpus_dir]
        if self.reload:
            args.append("-reload=1")
        args.extend(self.extra_args)
        if not any(a.startswith("-artifact_prefix=") for a in self.extra_args):
            args.append(f"-artifact_prefix={self.artifact_dir}{os.sep}")
        return args

    def start(self) -> None:
        os.makedirs(self.corpus_dir, exist_ok=True)
        os.makedirs(self.artifact_dir, exist_ok=True)
        self._proc = subprocess.Popen(
            self.command(),
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,        # like the engine's child: a program reading stdin must not read ours
            env=self.env,
            cwd=self.cwd,
        )
        self._reader = threading.Thread(target=self._pump_stderr, daemon=True)
        self._reader.start()

    def _pump_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            m = _COV_RE.search(line)
            if m:
                val = int(m.group(1))
                if val > self._cov:
                    self._cov = val

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def coverage(self) -> int:
        """The highest libFuzzer ``cov:`` counter seen so far (0 until reported)."""
        return self._cov

    def cpu_seconds(self) -> float:
        """This subprocess's own CPU time (user+system) so far, via ``/proc`` on Linux.

        Reads live while the process is running; once it has exited (``stop()`` or a
        crash) the ``/proc`` entry is gone, so this returns the last live reading instead
        of silently dropping back to 0. Returns 0.0 before ``start()``, and also once the
        process has exited if no live reading was ever taken (e.g. non-Linux: there is no
        wall-clock fallback here -- a caller that needs one tracks its own clock)."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                with open(f"/proc/{self._proc.pid}/stat", "rb") as fh:
                    # ``comm`` (field 2) is parenthesized and may itself contain ')', so
                    # split after the LAST ')' rather than assume field positions from 1.
                    after = fh.read().rsplit(b")", 1)[1].split()
                # after[0] is state (stat field 3); utime/stime are fields 14/15, i.e.
                # after[11]/after[12] once state..cmajflt (fields 3-13) are skipped.
                utime, stime = int(after[11]), int(after[12])
                self._cpu = (utime + stime) / os.sysconf("SC_CLK_TCK")
            except (OSError, ValueError, IndexError):
                pass  # /proc unavailable (non-Linux) or a benign read race; keep the last value
        return self._cpu

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self.cpu_seconds()  # snapshot the last live reading -- /proc disappears once reaped
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        # Drain the stderr reader so every `cov:` line the subprocess emitted is parsed
        # before a caller reads coverage(). Without this, coverage() read right after stop()
        # races the reader thread and can miss the final counter — and it did so
        # ASYMMETRICALLY between the benchmark arms, since only one read after stop.
        if self._reader is not None:
            self._reader.join(timeout=2)
        self._proc = None
        if self._owns_artifacts:
            shutil.rmtree(self.artifact_dir, ignore_errors=True)


class FakeBackend:
    """A backend that reports a scripted coverage sequence — for tests.

    ``coverage()`` walks ``cov_sequence`` one step per call (clamping at the end), so a
    test can model 'stuck, then a jump after an assist' without a real subprocess.
    """

    def __init__(self, cov_sequence: Sequence[int]) -> None:
        self._seq = list(cov_sequence) or [0]
        self._i = 0
        self._running = True
        self.started = False
        self.stopped = False
        self.cpu = 0.0   # a test may bump this directly to script cpu_seconds()

    def start(self) -> None:
        self.started = True

    def is_running(self) -> bool:
        return self._running

    def coverage(self) -> int:
        val = self._seq[min(self._i, len(self._seq) - 1)]
        self._i += 1
        return val

    def cpu_seconds(self) -> float:
        return self.cpu

    def stop(self) -> None:
        self.stopped = True
        self._running = False
