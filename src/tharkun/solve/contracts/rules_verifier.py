"""Contracts for the SV verifier's nondeterministic container generators."""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec

_STUB = "SV-Benchmarks python/_sv_verifier.py"
_CHECK_TYPE_UPSTREAM = "SV-Benchmarks python/_sv_verifier.py (MR1793)::check_type"


def _callback_claims(detail: str):
    return dunder_propagation_claims(
        detail,
        native={"TypeError": "invalid argument binding or a supplied callback/key hook raises TypeError"},
        tested_native=frozenset({"TypeError"}),
    )


@register_rule
class VerifierNondetList(Rule):
    """The rule of ``_sv_verifier.nondet_list``."""

    names = ("_sv_verifier.nondet_list",)
    SHAPE = Shape(0, 1, keywords_ok=True)
    SAMPLES = {
        "_sv_verifier.nondet_list": (
            "import _sv_verifier as sv\nxs = sv.nondet_list(sv.nondet_int)\n"
        )
    }
    TWINS = {
        "_sv_verifier.nondet_list": "import _sv_verifier as sv\nxs = sv.nondet_list()\n"
    }
    CONTRACT = Contract(
        qualname="_sv_verifier.nondet_list",
        raises=_callback_claims("the element callback"),
        may_raise_other=True,
        returns=ReturnSpec("list", "fresh list; length is any nonnegative integer", tested=True),
        effects=Effects(
            randomness="the SV verifier's nondeterministic draw source",
            deterministic=False,
            may_execute_callback=True,
        ),
        source=(f"{_STUB}::nondet_list", "CPython list allocation and callback propagation"),
        tested=True,
    )


@register_rule
class VerifierNondetDict(Rule):
    """The rule of ``_sv_verifier.nondet_dict``."""

    names = ("_sv_verifier.nondet_dict",)
    SHAPE = Shape(0, 2, keywords_ok=True)
    SAMPLES = {
        "_sv_verifier.nondet_dict": (
            "import _sv_verifier as sv\nd = sv.nondet_dict(sv.nondet_int, sv.nondet_int)\n"
        )
    }
    TWINS = {
        "_sv_verifier.nondet_dict": "import _sv_verifier as sv\nd = sv.nondet_dict()\n"
    }
    CONTRACT = Contract(
        qualname="_sv_verifier.nondet_dict",
        raises=_callback_claims("a key/value callback or a generated key's hash/equality"),
        may_raise_other=True,
        returns=ReturnSpec("dict", "fresh dict; generated equal keys overwrite", tested=True),
        effects=Effects(
            randomness="the SV verifier's nondeterministic draw source",
            deterministic=False,
            may_execute_callback=True,
        ),
        source=(f"{_STUB}::nondet_dict", "CPython dict insertion and equal-key overwrite"),
        tested=True,
    )


@register_rule
class VerifierCheckType(Rule):
    """The rule of ``_sv_verifier.check_type``."""

    names = ("_sv_verifier.check_type",)
    SHAPE = Shape(2, 2, keywords_ok=True)
    SAMPLES = {
        "_sv_verifier.check_type": (
            "import _sv_verifier as sv\nsv.check_type(1, int)\n"
        )
    }
    TWINS = {
        "_sv_verifier.check_type": "import _sv_verifier as sv\nsv.check_type(1, str)\n"
    }
    CONTRACT = Contract(
        qualname="_sv_verifier.check_type",
        raises=dunder_propagation_claims(
            "argument binding and the checker: plain class hints use isinstance; custom "
            "hints, metaclass __instancecheck__, value __class__ hooks, mismatch-message "
            "repr of the value and hint, and parameterized-hint iteration/mapping/equality "
            "hooks may execute user code",
            native={
                "TypeError": (
                    "invalid argument binding, an invalid hint, or a mismatch after "
                    "formatting the value representation"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("None", "returns None when the value matches", tested=True),
        effects=Effects(
            mutates_receiver=True,
            deterministic=False,
            may_execute_callback=True,
            notes=(
                "the full checker may consume an iterator supplied as value; custom hint, "
                "metaclass, __class__, repr, iteration, mapping, or equality hooks can also "
                "have external effects. A narrower model may certify only its admitted subset."
            ),
        ),
        notes=(
            "Parameterized/compound hints can invoke iteration or mapping hooks and may "
            "raise NotImplementedError for unsupported origins; custom metaclasses and "
            "value __class__ hooks can raise arbitrary exceptions."
        ),
        source=(_CHECK_TYPE_UPSTREAM, "canonical upstream checker; runtime provenance may vary"),
        tested=True,
    )
