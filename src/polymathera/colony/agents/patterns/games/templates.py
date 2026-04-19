"""Predefined game templates for DynamicGameCapability.

Provides high-level, LLM-friendly game configurations that expand into
concrete roles, team sizes, capabilities, and protocol config. The LLM
planner chooses a template name and a scrutiny level instead of specifying
detailed game parameters.

Usage::

    from polymathera.colony.agents.patterns.games.templates import BUILTIN_TEMPLATES

    template = BUILTIN_TEMPLATES["hypothesis_validation"]
    level = template.levels["thorough"]
    # level.roles == {"proposer": 1, "skeptic": 2, "grounder": 1, "arbiter": 1}
    # level.game_config == {"max_rounds": 3, "use_llm_reasoning": True}
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TemplateLevel:
    """Configuration for a single scrutiny level within a template.

    Attributes:
        roles: Role name -> count of agents to create.
        game_config: Protocol-specific constructor kwargs.
        role_capabilities: Per-role additional capability FQNs
            (beyond DynamicGameCapability which is always included).
    """

    roles: dict[str, int]
    game_config: dict[str, Any] = field(default_factory=dict)
    role_capabilities: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class GameTemplate:
    """A predefined game configuration with multiple scrutiny levels.

    Attributes:
        name: Template identifier (e.g., "hypothesis_validation").
        game_type: Registered game type in GameProtocolRegistry
            (e.g., "hypothesis_game").
        levels: Mapping of level name -> TemplateLevel.
        description: Human-readable description for LLM planner context.
        subject_to_initial_data: Callable that maps user-provided
            ``subject`` dict to ``initial_data`` for create_game.
            If None, subject is passed through as-is.
    """

    name: str
    game_type: str
    levels: dict[str, TemplateLevel]
    description: str = ""

    def get_level(self, level_name: str) -> TemplateLevel:
        """Get a level by name, defaulting to 'standard'.

        Raises:
            ValueError: If the level doesn't exist and 'standard' doesn't either.
        """
        if level_name in self.levels:
            return self.levels[level_name]
        if "standard" in self.levels:
            return self.levels["standard"]
        raise ValueError(
            f"Unknown level {level_name!r} for template {self.name!r}. "
            f"Available: {list(self.levels.keys())}"
        )

    @property
    def available_levels(self) -> list[str]:
        return list(self.levels.keys())


# ---------------------------------------------------------------------------
# Capability FQNs (used by templates)
# ---------------------------------------------------------------------------

_REFLECTION_CAP = "polymathera.colony.agents.patterns.capabilities.reflection.ReflectionCapability"
_CRITIC_CAP = "polymathera.colony.agents.patterns.capabilities.critique.CriticCapability"
_VALIDATION_CAP = "polymathera.colony.agents.patterns.capabilities.validation.ValidationCapability"


# ---------------------------------------------------------------------------
# Built-in Templates
# ---------------------------------------------------------------------------

HYPOTHESIS_VALIDATION = GameTemplate(
    name="hypothesis_validation",
    game_type="hypothesis_game",
    description=(
        "Validate a claim via structured propose/challenge/arbitrate. "
        "Use when a finding needs adversarial scrutiny before being accepted "
        "(e.g., CRITICAL impact, security vulnerability, contract violation)."
    ),
    levels={
        "quick": TemplateLevel(
            roles={"proposer": 1, "skeptic": 1},
            game_config={"use_llm_reasoning": True},
            role_capabilities={
                "proposer": [_REFLECTION_CAP],
                "skeptic": [_CRITIC_CAP],
            },
        ),
        "standard": TemplateLevel(
            roles={"proposer": 1, "skeptic": 1, "arbiter": 1},
            game_config={"use_llm_reasoning": True},
            role_capabilities={
                "proposer": [_REFLECTION_CAP],
                "skeptic": [_CRITIC_CAP],
                "arbiter": [_VALIDATION_CAP],
            },
        ),
        "thorough": TemplateLevel(
            roles={"proposer": 1, "skeptic": 2, "grounder": 1, "arbiter": 1},
            game_config={"use_llm_reasoning": True},
            role_capabilities={
                "proposer": [_REFLECTION_CAP],
                "skeptic": [_CRITIC_CAP],
                "arbiter": [_VALIDATION_CAP],
            },
        ),
        "adversarial": TemplateLevel(
            roles={"proposer": 1, "skeptic": 3, "grounder": 2, "arbiter": 1},
            game_config={"use_llm_reasoning": True},
            role_capabilities={
                "proposer": [_REFLECTION_CAP],
                "skeptic": [_CRITIC_CAP],
                "arbiter": [_VALIDATION_CAP],
            },
        ),
    },
)


NEGOTIATED_MERGE = GameTemplate(
    name="negotiated_merge",
    game_type="negotiation",
    description=(
        "Resolve conflicting analysis results via multi-party negotiation. "
        "Use when agents disagree on severity, classification, or interpretation "
        "and confidence-based weighting is insufficient."
    ),
    levels={
        "quick": TemplateLevel(
            roles={"negotiator": 2},
            game_config={"strategy": "compromising"},
        ),
        "standard": TemplateLevel(
            roles={"negotiator": 2, "mediator": 1},
            game_config={"strategy": "compromising"},
        ),
        "thorough": TemplateLevel(
            roles={"negotiator": 3, "mediator": 1},
            game_config={"strategy": "integrative"},
        ),
        "adversarial": TemplateLevel(
            roles={"negotiator": 4, "mediator": 1},
            game_config={"strategy": "competitive", "min_acceptable_utility": 0.4},
        ),
    },
)


CONSENSUS_VOTE = GameTemplate(
    name="consensus_vote",
    game_type="consensus_game",
    description=(
        "Reach group agreement on a decision via structured voting. "
        "Use for choosing between analysis approaches, escalation decisions, "
        "or any multi-option choice requiring group input."
    ),
    levels={
        "quick": TemplateLevel(
            roles={"voter": 3, "aggregator": 1},
            game_config={"voting_method": "majority", "rounds": 1},
        ),
        "standard": TemplateLevel(
            roles={"voter": 5, "aggregator": 1},
            game_config={"voting_method": "majority", "rounds": 2},
        ),
        "thorough": TemplateLevel(
            roles={"voter": 5, "proposer_consensus": 1, "aggregator": 1},
            game_config={"voting_method": "ranked_choice", "rounds": 3},
        ),
    },
)


CONTRACT_ALLOCATION = GameTemplate(
    name="contract_allocation",
    game_type="contract_net",
    description=(
        "Allocate analysis tasks to best-fit agents via competitive bidding. "
        "Use when tasks should be matched to agents based on capability, "
        "cache affinity, or specialization."
    ),
    levels={
        "quick": TemplateLevel(
            roles={"coordinator": 1, "bidder": 2},
            game_config={},
        ),
        "standard": TemplateLevel(
            roles={"coordinator": 1, "bidder": 3, "validator": 1},
            game_config={},
        ),
        "thorough": TemplateLevel(
            roles={"coordinator": 1, "bidder": 5, "validator": 1},
            game_config={},
        ),
    },
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BUILTIN_TEMPLATES: dict[str, GameTemplate] = {
    t.name: t
    for t in [HYPOTHESIS_VALIDATION, NEGOTIATED_MERGE, CONSENSUS_VOTE, CONTRACT_ALLOCATION]
}


def get_template(name: str) -> GameTemplate:
    """Look up a built-in template by name.

    Raises:
        ValueError: If template name is not registered.
    """
    if name not in BUILTIN_TEMPLATES:
        raise ValueError(
            f"Unknown game template: {name!r}. "
            f"Available: {list(BUILTIN_TEMPLATES.keys())}"
        )
    return BUILTIN_TEMPLATES[name]


def build_initial_data_for_hypothesis(subject: dict[str, Any], creator_agent_id: str) -> dict[str, Any]:
    """Map a hypothesis_validation subject to initial_data.

    Args:
        subject: User-provided subject with keys:
            - claim (str): The claim to validate
            - evidence (list[str], optional): Supporting evidence
            - confidence (float, optional): Initial confidence
        creator_agent_id: Agent creating the hypothesis.

    Returns:
        initial_data dict for create_game.
    """
    from ..models import Hypothesis

    hypothesis = Hypothesis(
        hypothesis_id=f"hyp_{uuid.uuid4().hex[:8]}",
        claim=subject.get("claim", ""),
        test_queries=subject.get("test_queries", []),
        confidence=subject.get("confidence", 0.5),
        created_by=creator_agent_id,
    )
    return {"hypothesis": hypothesis.model_dump()}


def build_initial_data_for_negotiation(subject: dict[str, Any]) -> dict[str, Any]:
    """Map a negotiated_merge subject to initial_data.

    Args:
        subject: User-provided subject with keys:
            - issue (str): What's being negotiated
            - options (list[dict], optional): Available options
            - constraints (dict, optional): Constraints on outcome

    Returns:
        initial_data dict for create_game.
    """
    return {
        "issue": {
            "issue_id": f"neg_{uuid.uuid4().hex[:8]}",
            "description": subject.get("issue", ""),
            "parties": [],  # Filled after participant assignment
            "constraints": subject.get("constraints", {}),
            "preferences": {},
        },
        "options": subject.get("options", []),
    }


def build_initial_data_for_consensus(subject: dict[str, Any]) -> dict[str, Any]:
    """Map a consensus_vote subject to initial_data."""
    return {
        "proposal": subject.get("proposal", ""),
        "options": subject.get("options", []),
    }


def build_initial_data_for_contract(subject: dict[str, Any]) -> dict[str, Any]:
    """Map a contract_allocation subject to initial_data."""
    return {
        "tasks": subject.get("tasks", []),
        "requirements": subject.get("requirements", {}),
    }


# Template name -> initial_data builder
INITIAL_DATA_BUILDERS: dict[str, Any] = {
    "hypothesis_validation": build_initial_data_for_hypothesis,
    "negotiated_merge": build_initial_data_for_negotiation,
    "consensus_vote": build_initial_data_for_consensus,
    "contract_allocation": build_initial_data_for_contract,
}
