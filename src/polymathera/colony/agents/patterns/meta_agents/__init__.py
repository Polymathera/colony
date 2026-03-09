"""Meta-agents for validation and oversight.

Meta-agents are specialized agents that monitor and validate the work of
other agents. They enforce quality, consistency, and goal alignment.

Based on Wooldridge's organizational mechanisms and norms.

Meta-agent types:
- GroundingAgent: Validates claims against evidence
- ConsistencyAgent: Detects contradictions
- ObjectiveGuardAgent: Prevents goal drift
- ReputationAgent: Updates agent reputations

All meta-agents:
- Extend base Agent class (legacy)
- Or use EventDrivenActionPolicy with AgentCapability (new pattern)
- Subscribe to relevant blackboard events
- Publish validation results
- Trigger corrective actions when needed
"""

# Legacy agent implementations (backward compatibility)
from .grounding import GroundingAgent
from .consistency import ConsistencyAgent
from .goal_alignment import ObjectiveGuardAgent
from .reputation_agent import ReputationAgent

__all__ = [
    # Legacy agents
    "GroundingAgent",
    "ConsistencyAgent",
    "ObjectiveGuardAgent",
    "ReputationAgent",
]

