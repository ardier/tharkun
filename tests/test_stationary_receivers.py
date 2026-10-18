"""``proof:stationary`` through a method on a typed receiver ([internal ref]; docs/SV.md "The call_target fact", consumer 2).

``richard = Richards(); richard.run(abs(_sv_verifier.nondet_int()))`` is the shape of ``bm_richards_nondet``: the trip value
reaches a METHOD, and the planner follows it there only when the ``call_target`` fact types the receiver from its one
constructor binding.  What that buys, and what it does not:

* a method whose loop repeats the same state is PROVED (``proof:stationary``) -- the program is one absint cannot decide;
* a method that grows state is planned (the callee resolves) and then REFUSED by the probe, never proved -- the shape of
  ``bm_richards_nondet`` itself: ``Task.__init__`` links every new task in front of ``taskWorkArea.taskList`` and nothing ever
  unlinks the old ones, so the reachable state differs by six task objects between any two iterations (token diff of the
  probe's two snapshots: 1356 against 1644 tokens, 6 against 12 task dicts).  Richards therefore stays UNKNOWN on purpose;
* every receiver the fact cannot type (two bindings, a parameter, a replaced method, a class with a base or a hook) is the
  "unresolved callee" refusal it always was.
"""
from __future__ import annotations

import sys
import textwrap
import time

import pytest

from tharkun.solve.call_targets import call_targets
from tharkun.solve.module_namespace import module_constants
from tharkun.sv import verify
from tharkun.sv.prove import prove_dynamic
from tharkun.sv.obligation_table import ObligationTable
from tharkun.solve.obligations import ObligationCensus
from tharkun.sv import scheduler

DL = ("KeyError", "IndexError")


def _prog(tmp_path, body, name="prog.py"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


def _prove(path, classes=DL, seconds=120):
    """``prove_dynamic`` with the facts the scheduler publishes for the program."""
    table = ObligationTable(ObligationCensus(()))
    scheduler.publish_module_namespace(table, path)
    t0 = time.monotonic()
    said = []
    v = prove_dynamic(path, classes, python=sys.executable, jobs=2, knowledge=table.kb, log=said.append,
                      remaining=lambda: seconds - (time.monotonic() - t0))
    return v, said


STATIONARY = """
    import re
    import _sv_verifier

    class Worker:
        def run(self, loops):
            for _ in range(loops):
                m = re.match("a(b)", "ab")
                g = m.group(1)
            return True

    if __name__ == "__main__":
        w = Worker()
        w.run(abs(_sv_verifier.nondet_int()))
"""


def test_a_stationary_method_on_a_typed_receiver_is_proved(tmp_path):
    p = _prog(tmp_path, STATIONARY)
    v, said = _prove(p)
    assert v is not None and v.verdict == "true" and v.component == "proof:stationary", (v, said)


def test_without_the_fact_the_same_program_is_not_proved(tmp_path):
    p = _prog(tmp_path, STATIONARY)
    v = prove_dynamic(p, DL, python=sys.executable, jobs=2, remaining=lambda: 60)
    assert v is None


def test_verify_end_to_end_proves_it_through_the_scheduler(tmp_path):
    p = _prog(tmp_path, STATIONARY)
    v = verify(p, DL, budget_seconds=60, jobs=2)
    assert v.verdict == "true" and v.component == "proof:stationary", (v.verdict, v.component, v.reason)


LEAK = """
    import _sv_verifier

    class Task:
        def __init__(self, link):
            self.link = link

    class Registry:
        head = None

    class Runner:
        def run(self, iterations):
            for i in range(iterations):
                Registry.head = Task(Registry.head)
            return True

    if __name__ == "__main__":
        r = Runner()
        r.run(abs(_sv_verifier.nondet_int()))
"""


def test_a_method_that_links_every_iteration_into_a_list_nothing_unlinks_is_planned_and_refused(tmp_path):
    """The shape of bm_richards_nondet: the callee resolves, the state grows, the probe says not stationary."""
    p = _prog(tmp_path, LEAK)
    v, said = _prove(p)
    assert v is None, v
    assert any("is not stationary after 1 iteration" in m for m in said), said


GROWS_AND_FAILS = """
    import _sv_verifier

    class Box:
        items = []

    class Worker:
        def run(self, loops):
            for _ in range(loops):
                Box.items.append(1)
            if len(Box.items) > 3:
                return {}["x"]
            return True

    if __name__ == "__main__":
        w = Worker()
        w.run(abs(_sv_verifier.nondet_int()))
"""


def test_a_method_whose_state_grows_toward_a_violation_is_never_proved_and_the_violation_is_found(tmp_path):
    p = _prog(tmp_path, GROWS_AND_FAILS)
    v, _said = _prove(p)
    assert v is None or v.verdict != "true", v
    found = verify(p, DL, budget_seconds=40, jobs=2)
    assert found.verdict == "false", (found.verdict, found.reason)


@pytest.mark.parametrize("name, body", {
    "a receiver bound twice": STATIONARY.replace("        w = Worker()\n", "        w = Worker()\n        w = Worker()\n"),
    "a receiver rebound in a branch": STATIONARY.replace("        w = Worker()\n", "        w = Worker()\n        if _sv_verifier.nondet_bool():\n            w = Other()\n").replace("    if __name__", "    class Other:\n        def run(self, loops):\n            return {}['x']\n\n    if __name__"),
    "a method replaced by an attribute store": STATIONARY.replace("        w = Worker()\n", "        w = Worker()\n        w.run = lambda loops: {}['x']\n"),
    "a class with a base": STATIONARY.replace("class Worker:", "class Base:\n        pass\n\n    class Worker(Base):"),
    "a class with __getattr__": STATIONARY.replace("        def run(self, loops):", "        def __getattr__(self, name):\n            return None\n\n        def run(self, loops):"),
    "a staticmethod": STATIONARY.replace("        def run(self, loops):", "        @staticmethod\n        def run(loops):"),
}.items(), ids=lambda x: x if isinstance(x, str) and len(x) < 60 else "")
def test_a_receiver_the_fact_cannot_type_stays_the_unresolved_callee_refusal(tmp_path, name, body):
    p = _prog(tmp_path, body)
    v, said = _prove(p)
    assert v is None or v.verdict != "true", (name, v)
    assert any("no plan" in m for m in said), (name, said)
