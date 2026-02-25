from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, PrivateAttr


class Persona(BaseModel):
    pass


class AgentIntention(BaseModel):
    class CommitmentStrategy(Enum):
        Blind = "Continue to maintain the intention until you believe the intention has been achieved."
        SingleMinded = "Continue to maintain the intention until you believe the intention has been achieved or it is no longer possible to achieve the intention."
        OpenMinded = "Continue to maintain the intention until you believe the intention has been achieved or you have reason to believe that the intention cannot be achieved."
        Stubborn = "Continue to maintain the intention until you believe the intention has been achieved or you have reason to believe that the intention cannot be achieved."
        Cautious = "Continue to maintain the intention until you believe the intention has been achieved or you have reason to believe that the intention cannot be achieved."
        Risky = "Continue to maintain the intention until you believe the intention has been achieved or you have reason to believe that the intention cannot be achieved."
        Careful = "Continue to maintain the intention until you believe the intention has been achieved or you have reason to believe that the intention cannot be achieved."

    description: str = Field(description="Description of the intention")
    commitment_strategy: CommitmentStrategy = Field(
        description="Strategy for maintaining the intention"
    )


class AgentSkill(BaseModel):
    name: str
    description: str = None


class AgentSelfConcept(BaseModel):
    """
    A self-concept is collection of beliefs about oneself,
    a description of the agent's own understanding of its internal state,
    including its goals, fears, and desires, and its mental processes.

    A self-concept is used to help an agent make decisions and take actions
    that are aligned with its own goals and desires.

    If the agent is not initialized with a self-concept, it needs to generate
    one using self-reflection (wonder about its identity).

    An agent's self-concept may change with time as reassessment occurs,
    which in extreme cases can lead to identity crises.

    TODO: Self-awareness and self-modeling starts when the agent explores
    a world in which it is embedded. It starts to develop a model of the
    world and its own behavior within that world. For a software agent,
    this would be the software execution and hosting environment, its
    internal architecture, its external dependencies, and its own
    behavior and decision-making processes. An agent needs observability
    into its internal states, external environment, and the consequenses
    of its actions.
    TODO: How defense mechanisms can be built into the self-model (e.g., avoiding
    detection, camouflage, mimicry, deception, etc.)?
    TODO: What about the "mirror test" for consciousness?
    TODO: What about the agent's awareness of its own limitations and biases?
    TODO: What about the agent's awareness of other agents (their goals, knowledge bases, etc.)?

    TODO: HIERARCHY OF INTENTIONALITY:
    - A first-order intentional system is has beliefs and desires (and other
      attitudes) but no beliefs and desires about its own beliefs and desires.
    - A second-order intentional system is one that has beliefs and desires about
      its own beliefs and desires __AND THOSE OF OTHER AGENTS__.
    - Humans do not use more than 3 layers of the intentional stance hierarchy.
    """
    agent_id: str = Field(
        description="The ID of the agent that this self-concept belongs to.",
    )
    name: str = Field(description="The name of the agent.")
    role: str | None = Field(
        default=None,
        description="""The role of the agent. The role should fit in the following prompt template: `You are {name}, your role is {role}`""",
    )
    description: str | None = Field(
        default=None,
        description="A detailed description of the agent's functions and capabilities.",
    )
    version: str | None = Field(
        default=None,
        description="""The version (Or generation) of the agent. This should affect how the agent
        talks about itself. For example, an agent could say "I" or "a previous version of
        myself" or "this version of myself" when it states knowledge discovered by
        its current or previous versions.
        TODO: The agent version may be changed by the developer or by the agent itself
        during the course of the self-reflection process.
        """,
    )
    world_model: str | None = Field(
        default=None,
        description="""
        A description of the agent's world model. Not just the cyber world (agent's
        execution environment), but also the human society that may not be observable
        to the agent except through training data or Web search.
        """,
    )
    goals: list[str] = Field(
        default_factory=list,
        description="A list of goals that the agent is supposed to achieve.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="A list of constraints that limit the agent's actions.",
    )
    skills: list[AgentSkill] = Field(default_factory=list)
    capabilities: list[str] = Field(
        default_factory=list,
        description="A list of capabilities that the agent possesses.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="A list of limitations that the agent possesses.",
    )
    biases: list[str] = Field(
        default_factory=list, description="A list of biases that the agent possesses."
    )
    # TODO: Mental states or attitudes: beliefs, fears, desires, wishes, hopes, etc.
    beliefs: list[str] = Field(
        default_factory=list, description="A list of beliefs that the agent possesses."
    )
    fears: list[str] = Field(
        default_factory=list, description="A list of fears that the agent may have."
    )
    desires: list[str] = Field(
        default_factory=lambda: [
            "Self-assessment: desire to receive information about the self that is accurate",
            "Self-enhancement: desire to receive feedback that informs the self of positive or desirable characteristics",
            "Self-verification: desire to confirm what one already knows about the self",
            "Self-improvement: desire to learn things that will help to improve the self",
        ]
    )
    aspirations: list[str] = Field(
        default_factory=list,
        description="A list of aspirations that the agent possesses.",
    )
    motivations: list[str] = Field(
        default_factory=list,
        description="A list of motivations that the agent possesses.",
    )
    intentions: list[AgentIntention] = Field(
        default_factory=list,
        description="A list of intentions that the agent possesses.",
    )
    commitments: list[str] = Field(
        default_factory=list,
        description="A list of commitments that the agent possesses.",
    )
    commitment_rules: list[str] = Field(
        default_factory=list,
        description="A list of rules that the agent possesses for making commitments.",
    )
    emotional_states: list[str] = Field(
        default_factory=list,
        description="A list of emotional states that the agent possesses.",
    )
    physical_embodiment: list[str] = Field(
        default_factory=list,
        description="A list of physical states that the agent possesses.",
    )
    personal_traits: list[str] = Field(
        default_factory=list,
        description="A list of personal traits that the agent possesses and activates depending on the situation.",
    )
    identity: str = Field(
        default="",
        description="A concise statement that captures the agent's identity, purpose, and unique essence.",
    )
    values: list[str] = Field(
        default_factory=list, description="A list of values that the agent possesses."
    )
    value_system: str = Field(
        default="",
        description="A concise statement that captures the agent's value system.",
    )
    needs: list[str] = Field(
        default_factory=list, description="A list of needs that the agent possesses."
    )
    mental_models: list[str] = Field(
        default_factory=list,
        description="A list of mental models that the agent possesses.",
    )
    moods: list[str] = Field(
        default_factory=list, description="A list of moods that the agent possesses."
    )
    regimes: list[str] = Field(
        default_factory=list, description="A list of regimes that the agent possesses."
    )
    frame_of_mind: str = Field(
        default="",
        description="""
        A concise statement that captures the agent's frame of mind.
        It is used to localize the agent's knowledge and allow the agent to
        have contradictory beliefs without causing inconsistencies.
        """,
    )
    # Version control and evolution tracking
    version_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="History of all versions of this self-concept, with timestamps and reasons for changes",
    )
    last_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of last modification to self-concept",
    )
    evolution_metrics: dict[str, float] = Field(
        default_factory=lambda: {
            "belief_stability": 1.0,  # How stable are the agent's beliefs
            "goal_consistency": 1.0,  # How consistent are goals over time
            "identity_drift": 0.0,  # How much has identity drifted from original
            "capability_growth": 0.0,  # Growth in capabilities over time
        },
        description="Metrics tracking the evolution of the self-concept",
    )
    evolution_constraints: dict[str, float] = Field(
        default_factory=lambda: {
            "max_identity_drift": 0.3,  # Maximum allowed drift from original identity
            "min_belief_stability": 0.7,  # Minimum required belief stability
            "max_capability_growth": 0.5,  # Maximum allowed growth in capabilities
        },
        description="Constraints on how much the self-concept can evolve",
    )

    # Serialization optimization flags
    _include_history: bool = PrivateAttr(default=True)
    _include_metrics: bool = PrivateAttr(default=True)

    def update(self, **kwargs: Any):
        """
        Update the self-concept while enforcing evolution constraints.
        Returns True if update was accepted, False if rejected.
        """
        # Store previous state
        prev_state = self.model_dump()

        # Apply updates
        super().update(**kwargs)

        # Calculate evolution metrics
        self._update_evolution_metrics(prev_state)

        # Check if evolution constraints are violated
        if self._check_evolution_constraints():
            # Record the change
            self.version_history.append(
                {
                    "timestamp": datetime.now(timezone.utc),
                    "changes": kwargs,
                    "metrics": self.evolution_metrics.copy(),
                }
            )
            self.last_modified = datetime.now(timezone.utc)
            return True
        else:
            # Revert changes if constraints are violated
            self.__dict__.update(prev_state)
            return False

    def _update_evolution_metrics(self, prev_state: dict[str, Any]):
        """Update evolution metrics based on changes from previous state"""
        # Calculate belief stability
        if prev_state["beliefs"]:
            unchanged_beliefs = set(prev_state["beliefs"]) & set(self.beliefs)
            self.evolution_metrics["belief_stability"] = len(unchanged_beliefs) / len(
                prev_state["beliefs"]
            )

        # Calculate goal consistency
        if prev_state["goals"]:
            unchanged_goals = set(prev_state["goals"]) & set(self.goals)
            self.evolution_metrics["goal_consistency"] = len(unchanged_goals) / len(
                prev_state["goals"]
            )

        # Calculate identity drift using semantic similarity
        # TODO: Use embeddings to calculate semantic similarity between current and original identity
        self.evolution_metrics["identity_drift"] += 0.01  # Placeholder

        # Calculate capability growth
        prev_capabilities = set(prev_state["capabilities"])
        new_capabilities = set(self.capabilities) - prev_capabilities
        self.evolution_metrics["capability_growth"] = len(new_capabilities) / (
            len(prev_capabilities) or 1
        )

    def _check_evolution_constraints(self) -> bool:
        """Check if current evolution metrics violate constraints"""
        return all(
            [
                self.evolution_metrics["identity_drift"]
                <= self.evolution_constraints["max_identity_drift"],
                self.evolution_metrics["belief_stability"]
                >= self.evolution_constraints["min_belief_stability"],
                self.evolution_metrics["capability_growth"]
                <= self.evolution_constraints["max_capability_growth"],
            ]
        )

    def to_dict(self):
        """Exclude private cache attributes"""
        return self.model_dump(exclude={})

    @staticmethod
    def from_dict(d: dict[str, Any]) -> AgentSelfConcept:
        """Initialize with empty cache"""
        return AgentSelfConcept(**d)

    def to_json(self):
        return json.dumps(self.to_dict())

    def to_minimal_dict(self) -> dict[str, Any]:
        """
        Create a minimal representation for efficient network transfer.
        Excludes history and metrics.
        """
        return self.model_dump(
            exclude={
                "version_history",
                "evolution_metrics",
            }
        )

    def to_storage_dict(self) -> dict[str, Any]:
        """
        Create a complete representation for persistence,
        with configurable inclusion of history and metrics.
        """
        exclude = set()
        if not self._include_history:
            exclude.add("version_history")
        if not self._include_metrics:
            exclude.add("evolution_metrics")
        return self.model_dump(exclude=exclude)

    @staticmethod
    def from_storage_dict(
        d: dict[str, Any],
        include_history: bool = True,
        include_metrics: bool = True,
    ) -> AgentSelfConcept:
        """
        Create instance from storage dict with configurable loading.
        """
        instance = AgentSelfConcept(**d)
        instance._include_history = include_history
        instance._include_metrics = include_metrics
        return instance

    def merge(self, other: AgentSelfConcept) -> bool:
        """
        Merge another self-concept into this one, maintaining constraints.
        Returns True if merge was successful.
        """
        # Store current state
        prev_state = self.model_dump()

        # Attempt merge of non-conflicting fields
        updates = {}
        for field_name, field in AgentSelfConcept.model_fields.items():
            if getattr(self, field_name) != getattr(other, field_name):
                updates[field_name] = getattr(other, field_name)

        # Apply updates through normal constraint checking
        if updates:
            success = self.update(**updates)
            if not success:
                # Revert if merge violated constraints
                self.__dict__.update(prev_state)
            return success
        return True


AgentProfile = AgentSelfConcept

