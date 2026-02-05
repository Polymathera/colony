"""Unified hypothesis game system for validation through structured debate.

This package provides a game-theoretic framework for hypothesis validation:
- Multi-agent debate (proposer, skeptic, grounder, arbiter)
- Pluggable strategies for formation, evidence gathering, evaluation
- Hypothesis tracking with flexible scope-based sharing
- Integration with ValidationCapability for contradiction detection

Core components:
- HypothesisGameProtocol: Game capability for agents to participate
- HypothesisTrackingCapability: Cross-game hypothesis tracking
- Strategy protocols: HypothesisFormationStrategy, EvidenceGatheringStrategy, etc.

Example usage:
    ```python
    # Create game protocol with strategies
    formation = LLMHypothesisFormation(agent, temperature=0.3)
    evidence = QueryBasedEvidence(agent)
    evaluation = LLMEvaluation(agent)

    protocol = HypothesisGameProtocol(
        agent=coordinator,
        formation_strategy=formation,
        evidence_strategy=evidence,
        evaluation_strategy=evaluation,
    )

    # Form hypotheses from context
    context = HypothesisContext(
        domain=HypothesisDomain.CODE_ANALYSIS,
        subject=SubjectReference(
            subject_type=SubjectType.FILE,
            subject_id="src/auth.py",
        ),
        observations=[
            Observation(
                observation_type=ObservationType.VULNERABILITY,
                description="SQL injection risk detected",
            )
        ],
    )

    hypotheses = await protocol.form_hypotheses_from_context(context)

    # Start validation game
    game_id = await protocol.start_game(
        participants={"agent1": "proposer", "agent2": "skeptic", "agent3": "arbiter"},
        initial_data={"hypotheses": [h.model_dump() for h in hypotheses]},
    )
    ```
"""

# Types and data structures
from .types import (
    SubjectType,
    SubjectReference,
    ObservationType,
    Observation,
    HypothesisDomain,
    HypothesisContext,
    EvidenceType,
    Evidence,
    TriggerType,
    HypothesisFormationTrigger,
    EvaluationDecision,
    EvaluationResult,
    HypothesisStatus,
    HypothesisFilter,
)

# Strategy protocols and implementations
from .strategies import (
    HypothesisFormationStrategy,
    EvidenceGatheringStrategy,
    HypothesisEvaluationStrategy,
    LLMHypothesisFormation,
    RuleBasedHypothesisFormation,
    QueryBasedEvidence,
    LLMReasoningEvidence,
    CompositeEvidenceStrategy,
    LLMEvaluation,
    RuleBasedEvaluation,
)

# Tracking capability
from .tracking import (
    TrackedHypothesis,
    HypothesisTrackingCapability,
)

# Game protocol and data
from .capabilities import (
    ChallengeRecord,
    HypothesisGameData,
    HypothesisRole,
    HypothesisGameProtocol,
)

__all__ = [
    # Types
    "SubjectType",
    "SubjectReference",
    "ObservationType",
    "Observation",
    "HypothesisDomain",
    "HypothesisContext",
    "EvidenceType",
    "Evidence",
    "TriggerType",
    "HypothesisFormationTrigger",
    "EvaluationDecision",
    "EvaluationResult",
    "HypothesisStatus",
    "HypothesisFilter",
    # Strategies
    "HypothesisFormationStrategy",
    "EvidenceGatheringStrategy",
    "HypothesisEvaluationStrategy",
    "LLMHypothesisFormation",
    "RuleBasedHypothesisFormation",
    "QueryBasedEvidence",
    "LLMReasoningEvidence",
    "CompositeEvidenceStrategy",
    "LLMEvaluation",
    "RuleBasedEvaluation",
    # Tracking
    "TrackedHypothesis",
    "HypothesisTrackingCapability",
    # Game
    "ChallengeRecord",
    "HypothesisGameData",
    "HypothesisRole",
    "HypothesisGameProtocol",
]
