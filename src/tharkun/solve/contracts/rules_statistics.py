"""Numeric statistics query contracts, CPython3.14.4 ([internal ref]).

The 17 scalar/list queries share finite primitive transfers. linear_regression
has a separate conservative call contract; its result fields are not modelled.
NormalDist and LinearRegression construction need receiver/result field models;
kde/kde_random need callable state (and random-state effects). Those APIs and
StatisticsError construction remain omitted. Open numeric/ordering hooks and
Decimal context dependence are conservatively effect-refused by recipe admission.
"""
from .rules import Rule, Shape, register_rule
from .types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim, ReturnSpec

NAMES=('mean','fmean','geometric_mean','harmonic_mean','median','median_low','median_high',
       'median_grouped','mode','multimode','quantiles','pstdev','pvariance','stdev','variance',
       'covariance','correlation')


def _query_contract(qualname):
    return Contract(qualname=qualname,
        raises=tuple(RaiseClaim(exc,'unknown') for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('object','scalar/element result, or list for multimode/quantiles; open numeric protocols can return other types',tested=True),
        effects=Effects(may_execute_callback=True,clock_or_env=True,deterministic=False,
                        notes='conservative open-operand effect: iteration, arithmetic, conversion, ordering and hashing can run hooks or observe Decimal context; primitive finite visitor has no context dependence'),
        notes='Empty/insufficient data and invalid numeric/configuration domains can raise statistics.StatisticsError (a ValueError subclass) or other exceptions. These qualified exception transfers remain named refusals, not fabricated builtin ValueError objects.',
        source=('https://docs.python.org/3.14/library/statistics.html#'+qualname,
                'https://github.com/python/cpython/blob/3.14/Lib/statistics.py'),tested=True)

_BY_QUALNAME={"statistics."+name:_query_contract("statistics."+name) for name in NAMES}


@register_rule
class StatisticsQuery(Rule):
    names=tuple('statistics.'+name for name in NAMES)
    SAMPLES={q:'import statistics\nx='+q+'((1,2,3)'+(',(2,4,6)' if q.rsplit('.',1)[1] in ('covariance','correlation') else '')+')\n' for q in names}
    TWINS={q:'import statistics\nx='+q+'()\n' for q in names}

    @classmethod
    def contract_for(cls,qualname):
        return _BY_QUALNAME[qualname]


@register_rule
class StatisticsLinearRegression(Rule):
    """The call contract; no unchecked tuple-field or real-arithmetic transfer."""

    names = ('statistics.linear_regression',)
    SHAPE = Shape(min_args=2, max_args=2, keywords_ok=True)
    SAMPLES = {'statistics.linear_regression': 'import statistics\nr = statistics.linear_regression([1, 2], [2, 4])\n'}
    TWINS = {'statistics.linear_regression': 'import statistics\nr = statistics.linear_regression()\n'}
    CONTRACT = Contract(
        qualname='statistics.linear_regression',
        raises=tuple(RaiseClaim(exc, 'unknown') for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('object', 'with exact list/tuple x/y containing only exact int/bool/float elements and exact bool proportional, successful calls return LinearRegression with float slope/intercept; open hooks need not produce numeric fields', tested=True),
        effects=Effects(mutates_args=(0, 1), may_execute_callback=True, clock_or_env=True, deterministic=False,
                        notes='input iteration can advance state; length, conversion, arithmetic and proportional truth can execute hooks or observe external numeric context; no whole-operation purity claim'),
        notes='Inputs x/y are positional-only; proportional is keyword-only. Size/domain failures can raise StatisticsError. A ZeroDivisionError in the final slope division is converted to StatisticsError, so arbitrary hook errors are not uniformly propagated.',
        source=('https://docs.python.org/3.14/library/statistics.html#statistics.linear_regression',
                'https://github.com/python/cpython/blob/v3.14.4/Lib/statistics.py#L744',
                'https://github.com/python/typeshed/blob/58c33f3da508976e26d681039bb9a255f9509f5c/stdlib/statistics.pyi'),
        tested=True,
    )
