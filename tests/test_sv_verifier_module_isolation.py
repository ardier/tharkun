"""``sys.modules["_sv_verifier"]`` isolation across in-process calls.

Flaky-test fix (2026-09-28, W45/MAIN root cause, [internal ref] follow-up).
``_child.py`` / ``_probe.py`` / ``_concolic_worker.py``'s ``main()`` each
normally run in their OWN fresh, disposable subprocess and exit right after
writing a result, so leaking ``sys.modules["_sv_verifier"]`` never mattered.
Under pytest-xdist, several tests share ONE worker process; a caller that
drives one of these in-process (or that leaves a raw, spec-less stub behind)
must not poison what the NEXT test in the same process sees.

This reproduces the exact failure a real full-budget campaign hit under load
(``ValueError: _sv_verifier.__spec__ is None`` raised by
``avengers_seed.py::_sv_verifier_stub_on_path``'s call to
``importlib.util.find_spec``) and confirms each fix: the three workers now
save/restore ``sys.modules["_sv_verifier"]`` (the same pattern
``_hybrid_worker.py``'s ``fuzz_one_input`` already used, since it is called
repeatedly in-process by Atheris), and the stub installer itself survives a
stale leftover instead of crashing on it.
"""

from __future__ import annotations

import copy
import json
import sys
import textwrap
import time
import types

import pytest

from tharkun.sv import _child, _concolic_worker, _probe


def _write_req(tmp_path, program_body, **extra):
    prog = tmp_path / "prog.py"
    prog.write_text(textwrap.dedent(program_body).lstrip("\n"))
    out = tmp_path / "out.json"
    req = {"program": str(prog), "vector": [], "out": str(out), **extra}
    req_path = tmp_path / "req.json"
    req_path.write_text(json.dumps(req))
    return str(req_path)


class TestStaleStubIsSurvived:
    """avengers_seed.py::_sv_verifier_stub_on_path must not crash on a leftover,
    spec-less _sv_verifier some OTHER in-process caller left in sys.modules."""

    def test_a_bare_module_object_left_behind_does_not_raise(self, monkeypatch):
        from tharkun.sv import avengers_seed

        monkeypatch.setattr(avengers_seed, "_stub_dir", None)
        # exactly what an unrestored `sys.modules["_sv_verifier"] = types.ModuleType(...)`
        # leaves behind: no __spec__, so importlib.util.find_spec raises ValueError
        stale = types.ModuleType("_sv_verifier")
        monkeypatch.setitem(sys.modules, "_sv_verifier", stale)
        avengers_seed._sv_verifier_stub_on_path()  # must not raise
        # and _sv_verifier must now be genuinely importable (the stale entry was
        # evicted, our file-based stub installed) -- not the leftover garbage
        import importlib

        sys.modules.pop("_sv_verifier", None)
        importlib.invalidate_caches()
        mod = importlib.import_module("_sv_verifier")
        assert mod.nondet_int() == 0
        sys.modules.pop("_sv_verifier", None)

    def test_a_second_call_after_eviction_still_works(self, monkeypatch):
        """The eviction path must leave _stub_dir in a state a LATER call can
        still use (not crash a second time, not double-install)."""
        from tharkun.sv import avengers_seed

        monkeypatch.setattr(avengers_seed, "_stub_dir", None)
        monkeypatch.setitem(sys.modules, "_sv_verifier", types.ModuleType("_sv_verifier"))
        avengers_seed._sv_verifier_stub_on_path()
        avengers_seed._sv_verifier_stub_on_path()  # cached path, must not raise either


class TestWorkerMainRestoresSysModules:
    """_child.py / _probe.py / _concolic_worker.py's main() must restore
    sys.modules["_sv_verifier"] to whatever was there before, so a caller that
    runs one of these in-process (not always via a fresh subprocess) never
    leaves a stub behind for the next test in the same xdist worker."""

    @pytest.fixture(autouse=True)
    def restore_disposable_worker_hooks(self, monkeypatch):
        import _thread
        import random
        import threading
        from tharkun.sv.nondet_harness import _RANDOM_NAMES, _TIME_NAMES

        # These entry points normally exit with their subprocess; tests share a worker.
        for obj, names in ((random, _RANDOM_NAMES), (time, _TIME_NAMES),
                           (random.Random, ("seed",)),
                           (random.SystemRandom, ("random", "getrandbits", "randbytes")),
                           (threading.Thread, ("start",)),
                           (_thread, _child._THREAD_STARTERS),
                           (threading, _child._THREAD_STARTERS)):
            for name in names:
                if hasattr(obj, name):
                    monkeypatch.setattr(obj, name, getattr(obj, name))
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.setattr(sys, "argv", list(sys.argv))
        monkeypatch.setattr(_probe.C, "_site", _probe.C._site)
        for model in (_child, _probe.C):
            for name, value in vars(model._State).items():
                if not name.startswith("__"):
                    monkeypatch.setattr(model._State, name, copy.copy(value))

    def test_child_main_restores_a_prior_sv_verifier(self, tmp_path):
        req_path = _write_req(tmp_path, "print(1)\n")
        sentinel = object()
        sys.modules["_sv_verifier"] = sentinel
        try:
            _child.main(req_path)
        finally:
            got = sys.modules.pop("_sv_verifier", None)
        assert got is sentinel

    def test_child_main_leaves_no_sv_verifier_when_none_was_present(self, tmp_path):
        req_path = _write_req(tmp_path, "print(1)\n")
        sys.modules.pop("_sv_verifier", None)
        _child.main(req_path)
        assert "_sv_verifier" not in sys.modules

    def test_probe_main_restores_a_prior_sv_verifier(self, tmp_path):
        req_path = _write_req(tmp_path, "x = 1\n", loops=[], quotients=[])
        sentinel = object()
        sys.modules["_sv_verifier"] = sentinel
        try:
            _probe.main(req_path)
        finally:
            got = sys.modules.pop("_sv_verifier", None)
        assert got is sentinel

    def test_concolic_worker_main_restores_a_prior_sv_verifier(self, tmp_path):
        prog = tmp_path / "prog.py"
        prog.write_text("if __name__ == '__main__':\n    pass\n")
        req = {"program_dir": str(tmp_path), "harness": str(prog), "sites": [],
               "source_binding": [prog.read_text(), str(prog), prog.read_text()],
               "feeds": {}, "params": [], "kinds": [], "seed": {}, "symbolic": False,
               "drive": False, "deadline": time.time() + 5,
               "out": str(tmp_path / "out.json")}
        req_path = tmp_path / "req.json"
        req_path.write_text(json.dumps(req))
        sentinel = object()
        sys.modules["_sv_verifier"] = sentinel
        try:
            _concolic_worker.main(str(req_path))
        finally:
            got = sys.modules.pop("_sv_verifier", None)
        assert got is sentinel
