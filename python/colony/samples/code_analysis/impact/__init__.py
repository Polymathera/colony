"""Change Impact Analysis package.

Provides capability-based agents and coordinators for analyzing the ripple
effects of code changes across a codebase.
"""

from .types import (
    CodeChange,
    ChangeType,
    ImpactType,
    ImpactStep,
    ImpactSeverity,
    ChangeImpactResult,
    ChangeImpactReport,
    ImpactedComponent,
    ImpactPath,
)

from .predictor import FeedbackLoopPredictor

from .coordinator import (
    ImpactMergePolicy,
    ChangeImpactAnalysisCoordinatorCapability,
    ChangeImpactAnalysisCoordinator,
)

from .page_analyzer import (
    ChangeImpactAnalysisPolicy,
    ChangeImpactAnalysisCapability,
    ChangeImpactAnalysisAgent,
)


__all__ = [
    # Types
    "CodeChange",
    "ChangeType",
    "ImpactType",
    "ImpactStep",
    "ImpactSeverity",
    "ChangeImpactResult",
    "ChangeImpactReport",
    "ImpactedComponent",
    "ImpactPath",
    # Predictor
    "FeedbackLoopPredictor",
    # Coordinator
    "ImpactMergePolicy",
    "ChangeImpactAnalysisCoordinatorCapability",
    "ChangeImpactAnalysisCoordinator",
    # Page Analyzer
    "ChangeImpactAnalysisPolicy",
    "ChangeImpactAnalysisCapability",
    "ChangeImpactAnalysisAgent",
]
