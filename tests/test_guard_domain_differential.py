"""The eregion -> tharkun differential for solve/guard_domain ([internal ref], slice 2).

``scripts/guard_domain_differential.py`` runs eregion's ``reduce_domain`` and
``guard_domain.domain_of`` over every guard assume in the eregion toy corpus,
and checks both against concrete values.  Its report is committed as
``tests/fixtures/guard_domain_differential.json``.  This test reads the
committed report.  When ``THARKUN_EREGION_ROOT`` names a py-inv umbrella
checkout, it also reruns the differential on the first toys.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "tests" / "fixtures" / "guard_domain_differential.json"


def _load():
    return json.loads(REPORT.read_text())


def test_committed_report_never_narrower():
    r = _load()
    t = r["totals"]
    assert t["cases"] > 1000 and t["from_source"] > 500
    assert r["tharkun_unsound_or_narrower"] == []
    for key in ("tharkun_unsound_cases", "narrower_cases", "narrower_undecided_cases"):
        assert key not in t, key
    # every wider case has a reason; the ones that are neither eregion being wrong
    # nor a type eregion's kind excludes are listed in full (none today)
    assert r["wider_precision_or_undecided"] == []
    assert r["random_guards"]["guards"] > 3000


def test_committed_report_lists_eregion_findings():
    r = _load()
    assert r["eregion_unsound_in_family"] == []           # the corpus: none
    rnd = r["random_guards"]
    assert set(rnd["in_family_examples"]) <= {"float_range", "RAISED"}


@pytest.mark.skipif(not os.environ.get("THARKUN_EREGION_ROOT"),
                    reason="THARKUN_EREGION_ROOT (a py-inv umbrella checkout) not set")
def test_live_differential(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_gd_diff", ROOT / "scripts" / "guard_domain_differential.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "r.json"
    rc = mod.main(["--eregion", os.environ["THARKUN_EREGION_ROOT"], "--limit", "120",
                   "--out", str(out)])
    r = json.loads(out.read_text())
    assert rc == 0 and r["tharkun_unsound_or_narrower"] == []
