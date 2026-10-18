"""tharkun as an SV-COMP Python verifier (DRIVE layer; docs/SV.md).

``verify(program, classes)`` decides an ``! uncaught(E)`` property by running
the unmodified program in fresh child interpreters, with every
``_sv_verifier.nondet_*`` call and every unseeded ``random`` call under the
engine's control.  FALSE carries a replayable witness; TRUE is claimed only for
an exhaustive exploration; everything else is UNKNOWN.  ``python -m tharkun.sv``
is the command-line tool a BenchExec tool-info module calls.
"""

from tharkun.sv.engine import Outcome, Verdict, replay, run_once, verify
from tharkun.sv.spec import program_of, read_property

__all__ = ["Outcome", "Verdict", "program_of", "read_property", "replay", "run_once", "verify"]
