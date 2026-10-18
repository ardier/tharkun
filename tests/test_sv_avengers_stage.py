"""The ``avengers`` SEED stage (``sv/avengers_seed.py``, ``sv/stages.py``), [internal ref].

Object-heavy tasks hide branching behind a class's own methods; this stage
constructs those classes with Avengers (backend Pynguin included) and offers
back whatever literal it needed as a candidate value. Correctness contract:
never a verdict by itself (a candidate only matters once the ENGINE replays
the real program and gets a real exception), degrades to no candidates when
Avengers/the backend cannot run, and stays out of ``DEFAULT`` until a
Real-world measurement shows it earns its keep (docs/AVENGERS.md).
"""

from __future__ import annotations

import textwrap

import pytest

from tharkun.avengers.backends import pynguin_available
from tharkun.sv import stages
from tharkun.sv.avengers_seed import seed_values
from tharkun.sv.engine import replay, verify

DL = ("ValueError",)


def _prog(tmp_path, name, src):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src).lstrip("\n"))
    return str(p)


def test_avengers_is_registered_but_not_in_default():
    names = {s.name for s in stages.REGISTRY}
    assert "avengers" in names
    assert "avengers" not in stages.DEFAULT
    assert [s.name for s in stages.select(["guard_values", "avengers"])] == ["guard_values", "avengers"]


def test_seed_values_finds_no_classes_returns_no_candidates(tmp_path):
    p = _prog(tmp_path, "noclass.py", """
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            print(x)
    """)
    assert seed_values(p) == {}


def test_seed_values_degrades_when_avengers_is_unavailable(tmp_path, monkeypatch):
    """Negative control: a broken/absent Avengers import must never raise or
    fabricate a candidate -- it is a missing tool, not an error (same contract
    as ``tharkun.avengers.backends.backend_status``)."""

    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "tharkun.avengers" or name.startswith("tharkun.avengers."):
            raise ImportError("simulated: avengers unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    p = _prog(tmp_path, "any.py", """
        class Widget:
            def __init__(self, value=0):
                self.value = value
        if __name__ == "__main__":
            Widget()
    """)
    assert seed_values(p) == {}


def test_seed_values_mines_a_literal_constructor_argument(tmp_path):
    """The base portfolio alone (no backend needed) must find *some* validated
    construction for a class with a REQUIRED annotated parameter -- unlike a
    defaulted parameter, the portfolio has to supply an explicit literal, so
    the recipe records one and this stage can harvest it."""

    p = _prog(tmp_path, "widget.py", """
        class Widget:
            def __init__(self, value: int):
                self.value = value
        if __name__ == "__main__":
            Widget(0)
    """)
    values = seed_values(p, max_classes=1, per_class_seconds=5.0, total_seconds=10.0)
    assert values.get("int")


@pytest.mark.skipif(not pynguin_available(), reason="pynguin is not installed")
def test_seed_values_actually_invokes_the_restored_pynguin_backend(tmp_path, monkeypatch):
    """Not a mock: asserts the real ``tharkun.avengers.backends`` module is
    reached with ``backends=("pynguin",)`` for every class this stage tries --
    the "no direct use outside Avengers" policy is about SV never importing
    ``backends``/``pynguin`` itself, not about the stage failing to ask
    Avengers to use it."""

    calls = []
    from tharkun.avengers import api as avengers_api

    real = avengers_api.synthesize_object

    def spy(*args, **kwargs):
        calls.append(kwargs.get("backends"))
        return real(*args, **kwargs)

    monkeypatch.setattr(avengers_api, "synthesize_object", spy)
    # avengers_seed imports synthesize_object from the tharkun.avengers facade,
    # so patch that binding too.
    import tharkun.avengers as avengers_pkg
    monkeypatch.setattr(avengers_pkg, "synthesize_object", spy)

    p = _prog(tmp_path, "widget2.py", """
        class Widget:
            def __init__(self, value=7):
                self.value = value
        if __name__ == "__main__":
            Widget()
    """)
    seed_values(p, max_classes=1, per_class_seconds=5.0, total_seconds=10.0)
    assert calls and calls[0] == ("pynguin",)


def test_avengers_stage_never_fabricates_a_verdict_on_its_own(tmp_path):
    """Negative control: a class whose recipe raises has NOTHING to do with the
    real program's draws here (the constructor is called with a literal, not a
    draw) -- adding the stage must not turn this into a false FALSE."""

    p = _prog(tmp_path, "unreachable.py", """
        import _sv_verifier

        class Bomb:
            def __init__(self, value):
                if value == 42:
                    raise ValueError("boom")

        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            Bomb(1)               # never fed by the draw -- avengers may still
            print(x)              # find value=42 raises, but it cannot matter
    """)
    v = verify(p, DL, budget_seconds=60, jobs=1, constants=False, deepen=False,
               stages=["avengers"])
    assert v.verdict != "false"


# A guard behind a class boundary that the static ``guard_values`` stage cannot see, because ``value`` is a PARAMETER of
# ``Bomb.__init__``, never a name bound directly to a draw call -- and a boundary the built-in (non-backend) portfolio does not
# find either: Hypothesis' shrinking converges to the simplest value that constructs at all (0) and stops, because plain
# construction has no branch to reward crossing the threshold.  A generator that proposes a constructor argument from the
# ``1_000_000`` literal in the source (Pynguin's fitness-guided search does) is what reaches it -- the concrete difference [internal ref]
# restores.  A NARROW window (rather than a bare threshold) keeps a generic engine's own boundary probes (``INT32_MAX`` and
# friends, tried by ``enum:small_first`` regardless of any seed stage) from accidentally satisfying it, so a pass here is
# actually attributable to the stage.  The exception itself is raised by a normal METHOD call (``band()``), not the constructor:
# a constructor that raises is always a failed construction in Avengers' goal model (never a validated recipe to harvest a
# literal from), so this fixture matches how the mechanism can actually work, not an unsupported shape.
#
# Pynguin is a randomized, time-boxed search: no gate assertion may depend on it succeeding within a wall-clock budget
# (docs/TESTING.md, "Randomized and wall-clock-boxed checks").  So the gate drives the stage with a DETERMINISTIC stand-in for
# the backend (``fixed_pynguin``: the backend's own output format, test source, with one known in-window argument), and asserts
# what holds whatever a real search finds.  How often the REAL search gets there is a measured metric, not a pass/fail:
# ``scripts/pynguin_seed_hit_rate.py`` (the nightly).
BOMB_PROGRAM = """
    import _sv_verifier

    class Bomb:
        def __init__(self, value: int):
            self.value = value

        def band(self):
            if 1_000_000 < self.value < 1_000_100:
                raise ValueError("boom")
            return "low"

    if __name__ == "__main__":
        x = _sv_verifier.nondet_int()
        Bomb(x).band()
"""

IN_WINDOW = 1_000_050
#: ceilings only: the stand-in finds the value at once (see ``fixed_pynguin``)
STAND_IN_CAPS = {"per_class_seconds": 60.0, "total_seconds": 120.0}


@pytest.fixture
def fixed_pynguin(monkeypatch):
    """The Pynguin backend replaced by a fixed generator: the same ``SourceArtifact`` (Pynguin-style test source) every time,
    with the in-window constructor argument.  The rest of the pipeline (the miner, the replay in a fresh child, the stage's
    literal harvest) is the production code.

    The stage's wall-clock caps (``SaturationBudget(max_seconds=per_class_seconds)``, ``total_seconds``) are raised to
    ``STAND_IN_CAPS``: the stand-in answers at once, so the caps are only ceilings, and a ceiling a loaded host can reach (the
    replay in a fresh child is a process start; 48 concurrent pytest processes on a 128-core host reached the old 10 s one)
    would make the assertion depend on the clock.  No gate assertion may (docs/TESTING.md).

    The replay itself has a clock too: ``IsolatedExecutor.timeout_s`` is 3 s per child, and ``synthesize_object`` builds its
    own executor with that default.  A child interpreter that needs more than 3 s to start and import (host load above 150:
    MG saw the class-guard node end ``UNKNOWN ... no stage left to try`` once, the replay timed out and no recipe validated)
    makes the miner see no object, hence no seed.  The executor ``synthesize_object`` builds is given ``STAND_IN_CAPS``'s
    ceiling too."""
    from tharkun.avengers import api, backends
    from tharkun.avengers.model import SourceArtifact
    from tharkun.sv import avengers_seed

    real_seed_values = avengers_seed.seed_values
    real_executor = api.IsolatedExecutor

    def generate(self, goal, *, seed=0):
        return (SourceArtifact("pynguin", f"import reachable as module_0\n\n\ndef test_case_0():\n"
                                          f"    int_0 = {IN_WINDOW}\n"
                                          f"    bomb_0 = module_0.{goal.class_name}(int_0)\n"
                                          f"    bomb_0.band()\n"),)

    monkeypatch.setattr(backends, "pynguin_available", lambda interpreter=None: True)
    monkeypatch.setattr(backends.PynguinBackendPass, "generate", generate)
    monkeypatch.setattr(api, "IsolatedExecutor",
                        lambda *a, **kw: real_executor(*a, **{"timeout_s": STAND_IN_CAPS["per_class_seconds"], **kw}))
    # ``sv/stages.py::_avengers`` imports ``seed_values`` when it runs, so this is the function the scheduler calls
    monkeypatch.setattr(avengers_seed, "seed_values",
                        lambda program, **kw: real_seed_values(program, **{**kw, **STAND_IN_CAPS}))


def test_avengers_seed_values_carry_the_backends_in_window_argument(tmp_path, fixed_pynguin):
    p = _prog(tmp_path, "reachable.py", BOMB_PROGRAM)
    ints = seed_values(p, max_classes=1, **STAND_IN_CAPS).get("int", [])
    assert IN_WINDOW in ints and all(isinstance(v, int) for v in ints)


@pytest.mark.skipif(not pynguin_available(), reason="pynguin is not installed")
def test_whatever_the_real_pynguin_search_finds_the_seeds_are_well_formed(tmp_path):
    """Holds for ANY outcome of the search (even none): kinds are the four literal kinds, every value has its kind's type, and
    no kind holds more than the stage's cap."""
    from tharkun.sv.avengers_seed import _CAP_PER_KIND

    p = _prog(tmp_path, "reachable.py", BOMB_PROGRAM)
    seeds = seed_values(p, max_classes=1, per_class_seconds=5.0, total_seconds=15.0)
    types = {"int": int, "float": float, "str": str, "bool": bool}
    assert set(seeds) <= set(types)
    for kind, values in seeds.items():
        assert len(values) <= _CAP_PER_KIND and all(type(v) is types[kind] for v in values)


def test_without_the_avengers_stage_the_class_guard_window_is_missed(tmp_path):
    p = _prog(tmp_path, "reachable.py", BOMB_PROGRAM)
    # prove=False: sv/prove.py's abstract interpreter is a separate, sound
    # static prover ([internal ref]) that does not model object/method flow this deep
    # and reports its own TRUE first; this test isolates the SEARCH portfolio
    # (guard_values/avengers/Candidates) the stage actually plugs into.
    without = verify(p, DL, budget_seconds=60, jobs=1, constants=False, deepen=False,
                      prove=False, stages=["guard_values"])
    assert without.verdict != "false"          # guard_values + the generic enumeration miss the window


def test_avengers_stage_seeds_a_candidate_that_reaches_a_class_guard(tmp_path, fixed_pynguin):
    p = _prog(tmp_path, "reachable.py", BOMB_PROGRAM)
    with_avengers = verify(p, DL, budget_seconds=120, jobs=1, constants=False, deepen=False,
                           prove=False, stages=["guard_values", "avengers"])
    assert with_avengers.line() == "FALSE(ValueError)"
    assert 1_000_000 < with_avengers.witness["vector"][0]["value"] < 1_000_100
    assert replay(p, with_avengers.witness, DL)[0]
