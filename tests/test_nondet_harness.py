"""sv/nondet_harness.py::drive -- the in-process nondet-override harness ([internal ref]).

drive_to_line and the assist loop need this to behave EXACTLY like sv/engine.py::
run_once's fresh subprocess for the same vector (soundness: only a fresh-child replay
may ever call a candidate FALSE, so drive() proposing exploration must not silently
diverge from what a real replay would find), and must leave the process exactly as it
found it -- it runs many times in the SAME interpreter as the solvers calling it,
unlike _child.py's own throwaway subprocess.
"""
from __future__ import annotations

import os
import random
import sys
import textwrap

from tharkun.sv.engine import run_once
from tharkun.sv.nondet_harness import DriveOutcome, drive


def _prog(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_clean_run_with_no_forced_draws_uses_kind_defaults(tmp_path):
    p = _prog(tmp_path, 'clean.py', """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        b = _sv_verifier.nondet_bool()
    """)
    r = drive(p, [])
    assert r.status == "clean"
    assert [d["kind"] for d in r.draws] == ["str", "bool"]
    assert [d["value"] for d in r.draws] == ["", False]


def test_a_forced_vector_reproduces_the_violation_and_the_target_line(tmp_path):
    p = _prog(tmp_path, 'violate.py', """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        if len(s) == 3:
            raise ValueError(s)
    """)
    vec = [{"kind": "str", "value": "abc"}]
    r = drive(p, vec, target_line=5)
    assert r.status == "raised" and r.exc == "ValueError" and r.reached is True
    assert r.violated(("ValueError",)) == "ValueError"
    assert r.violated(("KeyError",)) is None


def test_target_line_not_reached_when_the_branch_is_not_taken(tmp_path):
    p = _prog(tmp_path, 'unreached.py', """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        if len(s) == 3:
            raise ValueError(s)
    """)
    r = drive(p, [{"kind": "str", "value": "no"}], target_line=5)
    assert r.status == "clean" and r.reached is False


def test_matches_run_once_for_the_same_vector(tmp_path):
    """The soundness-relevant property: drive()'s in-process run must agree with a real
    fresh-subprocess replay for the same vector -- if it silently diverged, the assist
    loop could rank a candidate as promising for a reason a real replay would not
    confirm (harmless -- replay still gates FALSE -- but the searches would disagree)."""
    p = _prog(tmp_path, 'agree.py', """
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        n = _sv_verifier.nondet_bool()
        if len(s) == 3 and n:
            raise KeyError(s)
    """)
    for vec in ([], [{"kind": "str", "value": "abc"}],
                [{"kind": "str", "value": "abc"}, {"kind": "bool", "value": True}]):
        d = drive(p, vec)
        o = run_once(sys.executable, p, vec, timeout=5)
        assert (d.status, d.exc, d.draws) == (o.status, o.exc, o.draws)


def test_the_vector_grows_past_what_was_given(tmp_path):
    p = _prog(tmp_path, 'grow.py', """
        import _sv_verifier
        a = _sv_verifier.nondet_int()
        b = _sv_verifier.nondet_int()
        c = _sv_verifier.nondet_int()
    """)
    r = drive(p, [{"kind": "int", "value": 7}])
    assert [d["value"] for d in r.draws] == [7, 0, 0]


# --- must not leak state into the rest of the process ([internal ref]: unlike _child.py's own
# throwaway subprocess, drive() runs in the SAME interpreter as its caller) -------------

def test_random_class_methods_are_restored_after_clean_and_raised_runs(tmp_path, monkeypatch):
    methods = ((random.Random, "seed"), (random.SystemRandom, "random"),
               (random.SystemRandom, "getrandbits"), (random.SystemRandom, "randbytes"))
    for cls, name in methods:
        monkeypatch.setattr(cls, name, getattr(cls, name))
    original_seed = random.Random.seed

    def seed_hook(self, *args, **kwargs):
        return original_seed(self, *args, **kwargs)

    monkeypatch.setattr(random.Random, "seed", seed_hook)
    before = tuple(getattr(cls, name) for cls, name in methods)
    for body in ("pass\n", "raise ValueError('boom')\n"):
        p = _prog(tmp_path, "class_hooks.py", body)
        drive(p, [])
        assert tuple(getattr(cls, name) for cls, name in methods) == before


def test_foreign_random_class_watch_does_not_hide_unmodelled_draws(tmp_path, monkeypatch):
    from tharkun.sv import _child

    original_seed = random.Random.seed
    seen = []

    def foreign_seed(self, *args, **kwargs):
        return original_seed(self, *args, **kwargs)

    def foreign_random(self):
        seen.append(True)
        return 0.25

    monkeypatch.setattr(random.Random, "seed", foreign_seed)
    monkeypatch.setattr(random.SystemRandom, "random", foreign_random)
    p = _prog(tmp_path, "foreign_random.py", "import random\nrandom.SystemRandom().random()\nassert False\n")
    for owner in (object(), _child._State):
        foreign_seed._flags_unseeded = owner
        seen.clear()
        out = drive(p, [])
        assert seen and "unseeded_random_unmodelled" in out.flags
        assert random.SystemRandom.random is foreign_random


def test_random_state_is_restored_after_a_call(tmp_path):
    p = _prog(tmp_path, 'rnd.py', "import _sv_verifier\n_sv_verifier.nondet_int()\n")
    random.seed(12345)
    expected = random.random()
    random.seed(12345)
    drive(p, [])
    assert random.random() == expected


def test_sys_modules_and_argv_and_path_are_restored(tmp_path):
    p = _prog(tmp_path, 'mods.py', "import _sv_verifier\n_sv_verifier.nondet_int()\n")
    argv_before = list(sys.argv)
    path_before = list(sys.path)
    main_before = sys.modules.get("__main__")
    sv_before = "_sv_verifier" in sys.modules
    drive(p, [])
    assert sys.argv == argv_before
    assert sys.path == path_before
    assert sys.modules.get("__main__") is main_before
    assert ("_sv_verifier" in sys.modules) == sv_before


def test_state_is_restored_even_when_the_program_raises(tmp_path):
    p = _prog(tmp_path, 'raises.py', "import _sv_verifier\nraise ValueError('boom')\n")
    random.seed(999)
    expected = random.random()
    random.seed(999)
    r = drive(p, [])
    assert r.status == "raised"
    assert random.random() == expected


def test_repeated_calls_do_not_interfere_with_each_other(tmp_path):
    p1 = _prog(tmp_path, 'one.py', """
        import _sv_verifier
        if _sv_verifier.nondet_bool():
            raise KeyError()
    """)
    p2 = _prog(tmp_path, 'two.py', """
        import _sv_verifier
        if _sv_verifier.nondet_bool():
            raise IndexError()
    """)
    r1a = drive(p1, [{"kind": "bool", "value": True}])
    r2 = drive(p2, [{"kind": "bool", "value": True}])
    r1b = drive(p1, [{"kind": "bool", "value": False}])
    assert r1a.exc == "KeyError" and r2.exc == "IndexError" and r1b.status == "clean"


# --- MG 2026-09-29 (P0): importing this module must never mutate sys.path --------------

def test_importing_child_does_not_mutate_sys_path():
    """_child.py used to ``sys.path.insert(0, .../solve)`` at IMPORT time,
    unconditionally -- surviving for the rest of the IMPORTING process (this test
    file's own ``from tharkun.sv.nondet_harness import ...`` above already triggered
    the import) and shadowing the stdlib ``types`` module and the third-party
    ``attrs`` package with this repo's own ``tharkun/solve/types.py``/``attrs.py``.
    Any subprocess that then inherited that corrupted sys.path (Avengers' executor
    workers) broke for real -- confirmed: ``pytest -p no:xdist
    tests/test_nondet_harness.py tests/test_avengers_directed.py`` reproduced 4/4
    failures before this fix. Fixed: ``_child.py`` imports ``environment`` via its
    real package path (``tharkun.solve.environment``) when the package is already
    resolvable (every NORMAL import, including this one), and only falls back to
    the sys.path hack when run as a bare, disposable subprocess script, where
    ``tharkun`` itself is never on the path at all."""
    from tharkun.sv import _child
    solve_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(_child.__file__))), "solve")
    assert solve_dir not in sys.path


def test_types_and_attrs_are_not_shadowed_after_importing_nondet_harness():
    """The exact failure mode MG found: a bare ``import types``/``import attrs``
    elsewhere in the same process must resolve to the real stdlib/third-party
    modules, never this repo's own ``tharkun/solve/types.py``/``attrs.py``."""
    import types as _types

    import tharkun
    # the package directory, not the word "tharkun": a stdlib that lives under a path with that word in it (the CI host's
    # ~/tharkun-ci/python/...) is not this repo's module
    package = os.path.dirname(os.path.abspath(tharkun.__file__)) + os.sep
    assert not os.path.abspath(_types.__file__).startswith(package)
    try:
        import attrs as _attrs
    except ImportError:
        return  # attrs is an optional third-party dependency; nothing to check if absent
    assert not os.path.abspath(_attrs.__file__).startswith(package)


def test_avengers_worker_still_starts_after_importing_nondet_harness():
    """The pairing test: the exact end-to-end scenario that failed (MG's repro) --
    an Avengers directed-goal search, which spawns real subprocess workers, must
    still succeed after ``nondet_harness``/``_child`` have been imported into this
    same process."""
    from tharkun.avengers import MethodCall, ObjectGoal, synthesize_object
    from tharkun.avengers.engine import SaturationBudget

    fixture = os.path.realpath(os.path.join(os.path.dirname(__file__), "fixtures",
                                            "avengers_target.py"))
    goal = ObjectGoal(fixture, "TimerLike", (MethodCall("stop"),), "raises:RuntimeError")
    report = synthesize_object(goal, max_setup_steps=1, budget=SaturationBudget(max_seconds=45))
    assert report.best is not None
    assert report.best.outcome.exception_type == "builtins.RuntimeError"
