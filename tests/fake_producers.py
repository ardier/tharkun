"""Throwaway fact producers for the SV fact-producer tests: a module written into
``tmp_path`` plus its own ``*.dist-info/entry_points.txt``, found by
``tharkun.sv.fact_producers.discover`` exactly the way an installed distribution's
would be (in this process and in the child it starts).  Nothing here names Eregion."""
from __future__ import annotations

import os
import textwrap

from tharkun.sv import fact_producers as fp

#: a producer that behaves like a (lax) inverter front end: it nominates the census's own
#: invariant for every ZeroDivisionError obligation it is asked about
NOMINATE_ALL = """
    def produce(request):
        return [{"kind": "invariant_candidate",
                 "site": {"file": o["file"], "line": o["line"], "column": o["column"], "function": o["function"]},
                 "value": o["value"], "evidence": {"engine": "fake"}}
                for o in request["obligations"] if o["exception"] == "ZeroDivisionError"]
"""

#: ... but only for the first obligation it is handed
NOMINATE_FIRST = """
    def produce(request):
        rows = [o for o in request["obligations"] if o["exception"] == "ZeroDivisionError"][:1]
        return [{"kind": "invariant_candidate",
                 "site": {"file": o["file"], "line": o["line"], "column": o["column"], "function": o["function"]},
                 "value": o["value"]} for o in rows]
"""


def install(tmp_path, monkeypatch, body: str, name: str = "fake", *, only: bool = False):
    """Write the producer module, register entry point ``name`` and return its ``Producer``.

    ``only`` makes THIS process's ``discover()`` return just that producer, so a test is not
    influenced by whatever else is installed in the environment (the umbrella's venv has
    Eregion's own producer; the verifier's gate environment has none)."""
    (tmp_path / "fake_producer_mod.py").write_text(textwrap.dedent(body))
    dist = tmp_path / f"fakeprod_{name}-0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: fakeprod-{name}\nVersion: 0.0\n")
    (dist / "entry_points.txt").write_text(f"[{fp.GROUP}]\n{name} = fake_producer_mod:produce\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(tmp_path), os.environ.get("PYTHONPATH", "")])))
    producer = next(p for p in fp.discover() if p.name == name)
    if only:
        monkeypatch.setattr(fp, "discover", lambda: (producer,))
    return producer
