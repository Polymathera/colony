"""Memory Scope ID generation utilities.

This module provides standardized scope ID generation for the memory system.
All memory scopes follow a hierarchical naming convention that enables:
- Agent-private scopes (isolated per agent instance)
- Collective scopes (shared per agent_type, survives agent termination)
- Shared scopes (games, teams, tasks with explicit lifecycle)
- Tenant and system scopes (cross-agent shared memory)

Scope IDs are just strings used as blackboard scope identifiers. The
`MemoryScope` class provides helper methods to generate consistent IDs
and query patterns.

Example:
    ```python
    from polymathera.colony.agents.patterns.memory.scopes import MemoryScope

    # Agent-private scopes
    stm_scope = MemoryScope.agent_stm(agent)
    # -> "agent:abc123:stm"

    # Collective scopes (per agent_type)
    collective = MemoryScope.collective_procedural(agent.agent_type)
    # -> "agent_type:coding_agent:collective:procedural"

    # Query patterns
    all_agent_scopes = MemoryScope.agent_all(agent)
    # -> "agent:abc123:*"
    ```
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from ..distributed.ray_utils import serving

if TYPE_CHECKING:
    from .base import Agent


class BlackboardScope(Enum):
    """Predefined namespaces for agent capabilities and action policies."""

    COLONY = "colony"
    TENANT = "tenant"
    SESSION = "session"
    AGENT = "agent"


def get_scope_prefix(scope: BlackboardScope, agent: Agent | str | None = None, **kwargs) -> str:
    """Get the prefix string for a given blackboard scope.

    Args:
        scope: The blackboard scope enum value.
        agent: The agent instance or agent ID (optional, required for AGENT scope).
        **kwargs: Additional keyword arguments for scope formatting.

    Returns:
        The prefix string for the specified scope.
    Raises:
        ValueError: If the scope is AGENT and agent is not provided.
    """
    prefix = ""
    if scope == BlackboardScope.COLONY:
        prefix = ScopeUtils.get_colony_level_scope() or "colony"
    elif scope == BlackboardScope.TENANT:
        prefix = ScopeUtils.get_tenant_level_scope() or "tenant"
    elif scope == BlackboardScope.SESSION:
        prefix = ScopeUtils.get_session_level_scope() or "session"
    elif scope == BlackboardScope.AGENT:
        if agent is None:
            raise ValueError("Agent must be provided for AGENT scope")
        prefix = ScopeUtils.get_agent_level_scope(agent) or "agent"
    else:
        raise ValueError(f"Unsupported blackboard scope: {scope}")

    if kwargs:
        prefix += ":" + ScopeUtils.format_key(**kwargs)
    return prefix


class ScopeUtils:
    """Utility functions for working with blackboard scopes."""

    @staticmethod
    def format_key(**kwargs) -> str:
        """Format a list of (namespace, value) parts into a scope ID.

        Args:
            kwargs: Dictionary of namespace to value mappings

        Returns:
            Formatted scope ID string
        """
        # Sort parts by namespace for consistency
        sorted_parts = sorted(kwargs.items(), key=lambda x: x[0])
        return ":".join(f"{ns}:{val}" for ns, val in sorted_parts)

    @staticmethod
    def pattern_key(**kwargs) -> str:
        """Format a list of (namespace, value) parts into a scope pattern.

        None values are replaced with '*' for wildcard matching.

        Args:
            kwargs: Dictionary of namespace to value mappings, where value can be None

        Returns:
            Formatted scope pattern string
        """
        if not kwargs:
            return "*"
        # Sort parts by namespace for consistency
        sorted_parts = sorted(kwargs.items(), key=lambda x: x[0])
        return ":".join(f"{ns}:{val or '*'}" for ns, val in sorted_parts)

    @staticmethod
    def parse_key(scope_id: str, key: str) -> dict[str, str]:
        """Parse a blackboard key into its component namespaces and values.

        Args:
            scope_id: Blackboard scope ID string to parse
            key: Key string to parse

        Returns:
            Dictionary of namespace to value mappings
        """
        if key.startswith(f"{scope_id}:"):
            key = key[len(scope_id) + 1:]

        parts = key.split(":")
        if len(parts) % 2 != 0:
            raise ValueError(f"Invalid blackboard key format: {key}")
        return {parts[i]: parts[i + 1] for i in range(0, len(parts), 2)}

    @staticmethod
    def parse_key_part(scope_id: str, key: str, part: str) -> str | None:
        """Parse a specific part of a blackboard key.

        Args:
            scope_id: Blackboard scope ID string to parse
            key: Key string to parse
            part: Specific part to extract

        Returns:
            Value of the specified part, or None if not found
        """
        parsed = ScopeUtils.parse_key(scope_id, key)
        if not parsed or parsed.get(part) is None:
            return None
        return parsed[part]

    @staticmethod
    def get_global_scope() -> str | None:
        """Get global scope for scope IDs."""
        return "polymathera"

    @staticmethod
    def get_tenant_level_scope() -> str | None:
        """Get tenant prefix for scope IDs."""
        syscontext = serving.require_execution_context()
        tenant_id = syscontext.tenant_id
        if not tenant_id:
            raise ValueError(
                f"tenant_id={tenant_id} must be set in execution context"
            )
        return f"{ScopeUtils.get_global_scope()}:tenant:{tenant_id}"

    @staticmethod
    def get_colony_level_scope() -> str | None:
        """Get the colony-level scope for the current execution context."""
        syscontext = serving.require_execution_context()
        colony_id = syscontext.colony_id
        if not colony_id:
            raise ValueError(
                f"colony_id={colony_id} must be set in execution context"
            )
        return f"{ScopeUtils.get_tenant_level_scope()}:colony:{colony_id}"

    @staticmethod
    def get_session_level_scope() -> str:
        """Generate scope ID for session-level scope."""
        syscontext = serving.require_execution_context()
        session_id = syscontext.session_id
        if not session_id:
            raise ValueError(
                f"session_id={session_id} must be set in execution context"
            )
        return f"{ScopeUtils.get_colony_level_scope()}:session:{session_id}"

    @staticmethod
    def get_agent_level_scope(agent: Agent | str) -> str | None:
        """Get the agent-level scope for the current execution context.

        Args:
            agent: Agent instance or agent ID
            If agent is a string, the session ID will be extracted from the execution context and used to construct the scope.
             This allows generating agent scopes even when the agent instance is not available, as long as the session context is properly set up.

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:agent:abc123"
        """
        if isinstance(agent, str):
            agent_id = agent
        else:
            serving.ensure_context(agent.agent_id, agent.syscontext)
            agent_id = agent.agent_id
        return f"{ScopeUtils.get_session_level_scope()}:agent:{agent_id}"



class MemoryScope:
    """Generates standardized memory scope IDs.

    Memory scopes follow a hierarchical naming convention:
    - Agent-private: `tenant:{tenant_id}:colony:{colony_id}:agent:{agent_id}:{level}`
    - Collective:    `tenant:{tenant_id}:colony:{colony_id}:agent_type:{agent_type}:collective:{level}`
    - Shared:        `tenant:{tenant_id}:colony:{colony_id}:{type}:{id}:{level}` (game, team, task)
    - Colony:        `tenant:{tenant_id}:colony:{colony_id}:{namespace}`
    - Tenant:        `tenant:{tenant_id}:{namespace}`
    - System:        `global:{namespace}`

    Query patterns use `*` wildcards for matching multiple scopes.
    """

    # -------------------------------------------------------------------------
    # Agent-Private Scopes (per agent instance)
    # -------------------------------------------------------------------------

    @staticmethod
    def agent_memory(agent: Agent | str) -> str:
        """Top-level agent memory scope.

        Args:
            agent: Agent instance or agent ID

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:agent:abc123:memory"
        """
        return f"{ScopeUtils.get_agent_level_scope(agent)}:memory"

    @staticmethod
    def agent_sensory(agent: Agent | str) -> str:
        """Sensory memory scope (raw observations, very short retention).

        Args:
            agent: Agent instance or agent ID

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:agent:abc123:memory:sensory"
        """
        return f"{MemoryScope.agent_memory(agent)}:sensory"

    @staticmethod
    def agent_working(agent: Agent | str) -> str:
        """Working memory scope (current task context, append-only).

        Args:
            agent: Agent instance or agent ID

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:agent:abc123:memory:working"
        """
        return f"{MemoryScope.agent_memory(agent)}:working"

    @staticmethod
    def agent_stm(agent: Agent | str) -> str:
        """Short-term memory scope (recent observations).

        Args:
            agent: Agent instance or agent ID

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:agent:abc123:memory:stm"
        """
        return f"{MemoryScope.agent_memory(agent)}:stm"

    @staticmethod
    def agent_ltm(agent: Agent | str) -> str:
        """Long-term memory (episodic, semantic, procedural).

        Args:
            agent: Agent instance or agent ID

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:agent:abc123:memory:ltm"
        """
        return f"{MemoryScope.agent_memory(agent)}:ltm"

    @staticmethod
    def agent_ltm_episodic(agent: Agent | str) -> str:
        """Long-term episodic memory (experiences, events).

        Args:
            agent: Agent instance or agent ID

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:agent:abc123:memory:ltm:episodic"
        """
        return f"{MemoryScope.agent_ltm(agent)}:episodic"

    @staticmethod
    def agent_ltm_semantic(agent: Agent | str) -> str:
        """Long-term semantic memory (facts, concepts, relationships).

        Args:
            agent: Agent instance or agent ID

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:agent:abc123:memory:ltm:semantic"
        """
        return f"{MemoryScope.agent_ltm(agent)}:semantic"

    @staticmethod
    def agent_ltm_procedural(agent: Agent | str) -> str:
        """Long-term procedural memory (skills, prompts, self-concept).

        Args:
            agent: Agent instance or agent ID

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:agent:abc123:memory:ltm:procedural"
        """
        return f"{MemoryScope.agent_ltm(agent)}:procedural"

    @staticmethod
    def agent_transfer(agent: Agent | str, transfer_name: str) -> str:
        """Transfer capability scope (for tracking transfer state).

        Args:
            agent: Agent instance or agent ID
            transfer_name: Name of the transfer (e.g., "working_to_stm")

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:agent:abc123:transfer:working_to_stm"
        """
        return f"{MemoryScope.agent_memory(agent)}:transfer:{transfer_name}"

    # -------------------------------------------------------------------------
    # Collective Scopes (per agent_type, survives agent termination)
    # -------------------------------------------------------------------------

    @staticmethod
    def collective(agent_type: str) -> str:
        """Collective memory (episodic, semantic, procedural) for an agent type.

        Aggregated experiences from all terminated agents of this type.
        Used for transfer learning to new agents.

        Args:
            agent_type: Agent type string (e.g., "coding_agent")

        Returns:
            Scope ID like "tenant:abc123:colony:def456:agent_type:coding_agent:collective"
        """
        return f"{ScopeUtils.get_colony_level_scope()}:agent_type:{agent_type}:collective"

    @staticmethod
    def collective_episodic(agent_type: str) -> str:
        """Collective episodic memory for an agent type.

        Aggregated experiences from all terminated agents of this type.
        Used for transfer learning to new agents.

        Args:
            agent_type: Agent type string (e.g., "coding_agent")

        Returns:
            Scope ID like "tenant:abc123:colony:def456:agent_type:coding_agent:collective:episodic"
        """
        return f"{MemoryScope.collective(agent_type)}:episodic"

    @staticmethod
    def collective_semantic(agent_type: str) -> str:
        """Collective semantic memory for an agent type.

        Merged knowledge graphs from terminated agents.
        Deduplicated facts and relationships.

        Args:
            agent_type: Agent type string

        Returns:
            Scope ID like "tenant:abc123:colony:def456:agent_type:coding_agent:collective:semantic"
        """
        return f"{MemoryScope.collective(agent_type)}:semantic"

    @staticmethod
    def collective_procedural(agent_type: str) -> str:
        """Collective procedural memory for an agent type.

        Best-performing prompts and skills from all agents.
        Used to initialize new agents of same type.

        Args:
            agent_type: Agent type string

        Returns:
            Scope ID like "tenant:abc123:colony:def456:agent_type:coding_agent:collective:procedural"
        """
        return f"{MemoryScope.collective(agent_type)}:procedural"

    # -------------------------------------------------------------------------
    # Shared Scopes (Games, Teams, Tasks - explicit lifecycle)
    # -------------------------------------------------------------------------

    @staticmethod
    def session() -> str:
        """Top-level scope for session-level memory.

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:memory"
        """
        return f"{ScopeUtils.get_session_level_scope()}:memory"

    @staticmethod
    def game(game_id: str, level: str = "state") -> str:
        """Game-shared scope for multi-agent games.

        All game participants share this scope. Archived to participants'
        episodic memory on game completion, then cleaned up.

        Args:
            game_id: Unique game identifier
            level: Sub-level within game scope (default: "state")

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:memory:game:game_789:state"
        """
        return f"{MemoryScope.session()}:game:{game_id}:{level}"

    @staticmethod
    def team(team_id: str, level: str = "shared") -> str:
        """Team-shared scope for agent teams/crews.

        All team members share this scope. Archived to members'
        LTM on team dissolution.

        Args:
            team_id: Unique team identifier
            level: Sub-level within team scope (default: "shared")

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:memory:team:team_456:shared"
        """
        return f"{MemoryScope.session()}:team:{team_id}:{level}"

    @staticmethod
    def task(task_id: str, level: str = "context") -> str:
        """Task-scoped memory for multi-agent task collaboration.

        Intermediate results and shared artifacts. Archived to
        agents' STM on task completion.

        Args:
            task_id: Unique task identifier
            level: Sub-level within task scope (default: "context")

        Returns:
            Scope ID like "tenant:abc123:colony:def456:session:ghi789:memory:task:task_123:context"
        """
        return f"{MemoryScope.session()}:task:{task_id}:{level}"

    # -------------------------------------------------------------------------
    # Tenant and System Scopes
    # -------------------------------------------------------------------------

    @staticmethod
    def tenant(namespace: str) -> str:
        """Tenant-level shared scope.

        Shared across all agents in a tenant. Managed by
        tenant-level MemoryManagementAgent.

        Args:
            namespace: Namespace within tenant (e.g., "shared", "policies")

        Returns:
            Scope ID like "tenant:acme_corp:shared"
        """
        return f"{ScopeUtils.get_tenant_level_scope()}:memory:{namespace}"

    @staticmethod
    def system(namespace: str) -> str:
        """System-level global scope.

        Shared across all tenants. Managed by system-level
        MemoryManagementAgent.

        Args:
            namespace: Namespace within global scope

        Returns:
            Scope ID like "global:procedures"
        """
        return f"{ScopeUtils.get_global_scope()}:memory:{namespace}"

    @staticmethod
    def colony_control_plane(namespace: str) -> str:
        """Control plane scope for system events (lifecycle, etc.).

        Used by MemoryManagementAgent to monitor agent lifecycle.

        Args:
            namespace: Namespace (e.g., "lifecycle")

        Returns:
            Scope ID like "tenant:acme_corp:control_plane:lifecycle"
        """
        return f"{ScopeUtils.get_colony_level_scope()}:control_plane:{namespace}"

    # -------------------------------------------------------------------------
    # Query Patterns (for blackboard.query(namespace=pattern))
    # -------------------------------------------------------------------------

    @staticmethod
    def agent_all(agent: Agent | str) -> str:
        """Pattern matching all scopes for an agent.

        Args:
            agent: Agent instance or agent ID

        Returns:
            Pattern like "tenant:abc123:colony:def456:session:ghi789:agent:abc123:*"
        """
        return f"{MemoryScope.agent_memory(agent)}:*"

    @staticmethod
    def agent_ltm_all(agent: Agent | str) -> str:
        """Pattern matching all LTM sub-levels for an agent.

        Args:
            agent: Agent instance or agent ID

        Returns:
            Pattern like "tenant:abc123:colony:def456:session:ghi789:agent:abc123:ltm:*"
        """
        return f"{MemoryScope.agent_ltm(agent)}:*"

    @staticmethod
    def collective_all(agent_type: str) -> str:
        """Pattern matching all collective scopes for an agent type.

        Args:
            agent_type: Agent type string

        Returns:
            Pattern like "tenant:abc123:colony:def456:agent_type:coding_agent:collective:*"
        """
        return f"{MemoryScope.collective(agent_type)}:*"

    @staticmethod
    def all_agents_scope(level: str) -> str:
        """Pattern matching a specific level across all agents.

        Args:
            level: Memory level (e.g., "stm", "ltm:episodic")

        Returns:
            Pattern like "tenant:abc123:colony:def456:session:ghi789:agent:*:stm"
        """
        return f"{ScopeUtils.get_session_level_scope()}:agent:*:memory:{level}"

    @staticmethod
    def game_all(game_id: str) -> str:
        """Pattern matching all scopes for a game.

        Args:
            game_id: Unique game identifier

        Returns:
            Pattern like "tenant:abc123:colony:def456:game:game_789:*"
        """
        return f"{MemoryScope.session()}:game:{game_id}:*"

    @staticmethod
    def team_all(team_id: str) -> str:
        """Pattern matching all scopes for a team.

        Args:
            team_id: Unique team identifier

        Returns:
            Pattern like "tenant:abc123:colony:def456:team:team_456:*"
        """
        return f"{MemoryScope.session()}:team:{team_id}:*"

    # -------------------------------------------------------------------------
    # Parsing (extract components from scope ID)
    # -------------------------------------------------------------------------

    @staticmethod
    def parse_agent_scope(scope_id: str) -> tuple[str, str] | None:
        """Parse an agent-private scope ID.

        Args:
            scope_id: Scope ID to parse

        Returns:
            Tuple of (agent_id, level) or None if not an agent scope
        """
        prefix = ScopeUtils.get_session_level_scope()
        if not scope_id.startswith(f"{prefix}:agent:"):
            return None
        scope_id = scope_id[len(f"{prefix}:") :]
        parts = scope_id.split(":", 2)
        if len(parts) < 4 or parts[2] != "memory":
            return None
        return (parts[1], ":".join(parts[3:]))

    @staticmethod
    def parse_collective_scope(scope_id: str) -> tuple[str, str] | None:
        """Parse a collective scope ID.

        Args:
            scope_id: Scope ID to parse

        Returns:
            Tuple of (agent_type, level) or None if not a collective scope
        """
        prefix = ScopeUtils.get_colony_level_scope()
        if not scope_id.startswith(f"{prefix}:agent_type:"):
            return None
        scope_id = scope_id[len(f"{prefix}:") :]
        parts = scope_id.split(":")
        if len(parts) < 4 or parts[2] != "collective":
            return None
        return (parts[1], ":".join(parts[3:]))

    @staticmethod
    def is_agent_private(scope_id: str) -> bool:
        """Check if scope is agent-private.

        Args:
            scope_id: Scope ID to check

        Returns:
            True if this is an agent-private scope
        """
        prefix = ScopeUtils.get_session_level_scope()
        return scope_id.startswith(f"{prefix}:agent:")

    @staticmethod
    def is_collective(scope_id: str) -> bool:
        """Check if scope is collective (per agent_type).

        Args:
            scope_id: Scope ID to check

        Returns:
            True if this is a collective scope
        """
        prefix = ScopeUtils.get_colony_level_scope()
        return scope_id.startswith(f"{prefix}:agent_type:") and ":collective:" in scope_id

    @staticmethod
    def is_shared(scope_id: str) -> bool:
        """Check if scope is shared (game, team, task).

        Args:
            scope_id: Scope ID to check

        Returns:
            True if this is a shared scope
        """
        prefix = MemoryScope.session()
        return (
            scope_id.startswith(f"{prefix}:game:")
            or scope_id.startswith(f"{prefix}:team:")
            or scope_id.startswith(f"{prefix}:task:")
        )

    # -------------------------------------------------------------------------
    # Session Scopes (per tenant, entries tagged with session_id)
    # -------------------------------------------------------------------------

    @staticmethod
    def session_all() -> str:
        """Pattern matching all session scopes for a tenant.

        Returns:
            Pattern like "session:acme_corp:*"
        """
        return f"{ScopeUtils.get_colony_level_scope()}:session:*:memory:*"

    @staticmethod
    def is_session_scope(scope_id: str) -> bool:
        """Check if scope is a session scope.

        Args:
            scope_id: Scope ID to check

        Returns:
            True if this is a session scope
        """
        prefix = ScopeUtils.get_colony_level_scope()
        return scope_id.startswith(f"{prefix}:session:")



