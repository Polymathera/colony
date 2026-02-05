"""Agent games framework for robust multi-agent collaboration.

This package implements game-theoretic protocols for LLM-based agents to:
- Combat hallucination through structured challenge and evidence requirements
- Prevent laziness through reputation-based task allocation
- Ensure goal alignment through objective guards
- Enable robust collaboration through formal protocols

Based on:
- Wooldridge's "An Introduction to Multi-Agent Systems"
- Shoham & Leyton-Brown's "Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations"
- Multi-agent game engine specifications in MULTI_AGENT_GAME_ENGINE.md

Core concepts:
- ACL (Agent Communication Language) messages with speech act semantics
- Game protocols as formal extensive/normal-form games
- Reputation and learning for adaptation
- Epistemic layer for knowledge tracking
- Meta-agents for validation and oversight

Game types:
- Hypothesis Game: Hallucination control through challenge-response
- Contract Net: Task allocation through competitive bidding
- Consensus Game: Agreement building through voting
- Negotiation Game: Conflict resolution through structured negotiation

All games integrate with:
- ScopeAwareResult pattern for partial knowledge
- Task Graph for coordination
- Blackboard for shared memory
- Existing agent infrastructure
"""

from .acl import ACLMessage, Performative, MessageSchema
from .state import GameState, GamePhase, GameOutcome, GameProtocolCapability
from .roles import GameRole, RoleCapabilities

__all__ = [
    # ACL
    "ACLMessage",
    "Performative",
    "MessageSchema",

    # State
    "GameState",
    "GamePhase",
    "GameOutcome",
    "GameProtocolCapability",

    # Roles
    "GameRole",
    "RoleCapabilities",
]

