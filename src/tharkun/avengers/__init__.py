"""Object construction and boundary-call orchestration for Tharkun.

The package is a DRIVE-layer facade. Importing it does not change Tharkun's
backward-compatible package-root surface.
"""

from tharkun.avengers.engine import (
    Avengers,
    Emission,
    FunctionPass,
    PassMetrics,
    PortfolioOrder,
    RepeatPolicy,
    SaturationBudget,
    SaturationReport,
    Scheduler,
    YieldOrder,
)
from tharkun.avengers.executor import IsolatedExecutor
from tharkun.avengers.flatting import (
    FlatObjectTemplate,
    FlattingUnavailable,
    flatting_available,
    template_from_object,
)
from tharkun.avengers.api import default_passes, synthesize_object
from tharkun.avengers.backends import (
    GhostwriterPass,
    HypothesisFromTypePass,
    PynguinBackendPass,
    backend_passes,
    backend_status,
)
from tharkun.avengers.objectives import (
    CoverageObservation,
    CoverageSetupPass,
    NegativeSequence,
    NegativeSequencePass,
    ReceiverState,
    StateDiversityPass,
)
from tharkun.avengers.store import StoredNegative, StoredRecipe, StoreSeedPass, TemplateStore
from tharkun.avengers.model import (
    EXPECTATIONS,
    AvengersReport,
    ConstructionRecipe,
    ConstructorCall,
    FieldBound,
    FunctionCall,
    MethodCoverage,
    MethodCall,
    NestedObject,
    OBJECT,
    ObjectGoal,
    ObjectReference,
    ReplayOutcome,
    SourceArtifact,
    ValidatedRecipe,
    field_bounds_hold,
    materialize_recipe,
)

__all__ = [
    "Avengers",
    "AvengersReport",
    "ConstructionRecipe",
    "ConstructorCall",
    "CoverageObservation",
    "CoverageSetupPass",
    "EXPECTATIONS",
    "Emission",
    "FieldBound",
    "FunctionPass",
    "FunctionCall",
    "GhostwriterPass",
    "HypothesisFromTypePass",
    "FlatObjectTemplate",
    "FlattingUnavailable",
    "IsolatedExecutor",
    "MethodCall",
    "MethodCoverage",
    "NegativeSequence",
    "NegativeSequencePass",
    "NestedObject",
    "ObjectGoal",
    "ObjectReference",
    "OBJECT",
    "PynguinBackendPass",
    "PassMetrics",
    "PortfolioOrder",
    "ReceiverState",
    "RepeatPolicy",
    "ReplayOutcome",
    "SaturationBudget",
    "SaturationReport",
    "Scheduler",
    "SourceArtifact",
    "StateDiversityPass",
    "StoreSeedPass",
    "StoredNegative",
    "StoredRecipe",
    "TemplateStore",
    "ValidatedRecipe",
    "backend_passes",
    "backend_status",
    "default_passes",
    "field_bounds_hold",
    "flatting_available",
    "materialize_recipe",
    "synthesize_object",
    "template_from_object",
    "YieldOrder",
]
