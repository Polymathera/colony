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
    stm_scope = MemoryScope.agent_stm(agent.agent_id)
    # -> "agent:abc123:stm"

    # Collective scopes (per agent_type)
    collective = MemoryScope.collective_procedural(agent.agent_type)
    # -> "agent_type:coding_agent:collective:procedural"

    # Query patterns
    all_agent_scopes = MemoryScope.agent_all(agent.agent_id)
    # -> "agent:abc123:*"
    ```
"""

from __future__ import annotations


class MemoryScope:
    """Generates standardized memory scope IDs.

    Memory scopes follow a hierarchical naming convention:
    - Agent-private: `agent:{agent_id}:{level}`
    - Collective: `agent_type:{agent_type}:collective:{level}`
    - Shared: `{type}:{id}:{level}` (game, team, task)
    - Tenant: `tenant:{tenant_id}:{namespace}`
    - System: `global:{namespace}`

    Query patterns use `*` wildcards for matching multiple scopes.
    """

    # -------------------------------------------------------------------------
    # Agent-Private Scopes (per agent instance)
    # -------------------------------------------------------------------------

    @staticmethod
    def agent_sensory(agent_id: str) -> str:
        """Sensory memory scope (raw observations, very short retention).

        Args:
            agent_id: Unique agent identifier

        Returns:
            Scope ID like "agent:abc123:sensory"
        """
        return f"agent:{agent_id}:sensory"

    @staticmethod
    def agent_working(agent_id: str) -> str:
        """Working memory scope (current task context, append-only).

        Args:
            agent_id: Unique agent identifier

        Returns:
            Scope ID like "agent:abc123:working"
        """
        return f"agent:{agent_id}:working"

    @staticmethod
    def agent_stm(agent_id: str) -> str:
        """Short-term memory scope (recent observations).

        Args:
            agent_id: Unique agent identifier

        Returns:
            Scope ID like "agent:abc123:stm"
        """
        return f"agent:{agent_id}:stm"

    @staticmethod
    def agent_ltm_episodic(agent_id: str) -> str:
        """Long-term episodic memory (experiences, events).

        Args:
            agent_id: Unique agent identifier

        Returns:
            Scope ID like "agent:abc123:ltm:episodic"
        """
        return f"agent:{agent_id}:ltm:episodic"

    @staticmethod
    def agent_ltm_semantic(agent_id: str) -> str:
        """Long-term semantic memory (facts, concepts, relationships).

        Args:
            agent_id: Unique agent identifier

        Returns:
            Scope ID like "agent:abc123:ltm:semantic"
        """
        return f"agent:{agent_id}:ltm:semantic"

    @staticmethod
    def agent_ltm_procedural(agent_id: str) -> str:
        """Long-term procedural memory (skills, prompts, self-concept).

        Args:
            agent_id: Unique agent identifier

        Returns:
            Scope ID like "agent:abc123:ltm:procedural"
        """
        return f"agent:{agent_id}:ltm:procedural"

    @staticmethod
    def agent_transfer(agent_id: str, transfer_name: str) -> str:
        """Transfer capability scope (for tracking transfer state).

        Args:
            agent_id: Unique agent identifier
            transfer_name: Name of the transfer (e.g., "working_to_stm")

        Returns:
            Scope ID like "agent:abc123:transfer:working_to_stm"
        """
        return f"agent:{agent_id}:transfer:{transfer_name}"

    # -------------------------------------------------------------------------
    # Collective Scopes (per agent_type, survives agent termination)
    # -------------------------------------------------------------------------

    @staticmethod
    def collective_episodic(agent_type: str) -> str:
        """Collective episodic memory for an agent type.

        Aggregated experiences from all terminated agents of this type.
        Used for transfer learning to new agents.

        Args:
            agent_type: Agent type string (e.g., "coding_agent")

        Returns:
            Scope ID like "agent_type:coding_agent:collective:episodic"
        """
        return f"agent_type:{agent_type}:collective:episodic"

    @staticmethod
    def collective_semantic(agent_type: str) -> str:
        """Collective semantic memory for an agent type.

        Merged knowledge graphs from terminated agents.
        Deduplicated facts and relationships.

        Args:
            agent_type: Agent type string

        Returns:
            Scope ID like "agent_type:coding_agent:collective:semantic"
        """
        return f"agent_type:{agent_type}:collective:semantic"

    @staticmethod
    def collective_procedural(agent_type: str) -> str:
        """Collective procedural memory for an agent type.

        Best-performing prompts and skills from all agents.
        Used to initialize new agents of same type.

        Args:
            agent_type: Agent type string

        Returns:
            Scope ID like "agent_type:coding_agent:collective:procedural"
        """
        return f"agent_type:{agent_type}:collective:procedural"

    # -------------------------------------------------------------------------
    # Shared Scopes (Games, Teams, Tasks - explicit lifecycle)
    # -------------------------------------------------------------------------

    @staticmethod
    def game(game_id: str, level: str = "state") -> str:
        """Game-shared scope for multi-agent games.

        All game participants share this scope. Archived to participants'
        episodic memory on game completion, then cleaned up.

        Args:
            game_id: Unique game identifier
            level: Sub-level within game scope (default: "state")

        Returns:
            Scope ID like "game:game_789:state"
        """
        return f"game:{game_id}:{level}"

    @staticmethod
    def team(team_id: str, level: str = "shared") -> str:
        """Team-shared scope for agent teams/crews.

        All team members share this scope. Archived to members'
        LTM on team dissolution.

        Args:
            team_id: Unique team identifier
            level: Sub-level within team scope (default: "shared")

        Returns:
            Scope ID like "team:team_456:shared"
        """
        return f"team:{team_id}:{level}"

    @staticmethod
    def task(task_id: str, level: str = "context") -> str:
        """Task-scoped memory for multi-agent task collaboration.

        Intermediate results and shared artifacts. Archived to
        agents' STM on task completion.

        Args:
            task_id: Unique task identifier
            level: Sub-level within task scope (default: "context")

        Returns:
            Scope ID like "task:task_123:context"
        """
        return f"task:{task_id}:{level}"

    # -------------------------------------------------------------------------
    # Tenant and System Scopes
    # -------------------------------------------------------------------------

    @staticmethod
    def tenant(tenant_id: str, namespace: str) -> str:
        """Tenant-level shared scope.

        Shared across all agents in a tenant. Managed by
        tenant-level MemoryManagementAgent.

        Args:
            tenant_id: Tenant identifier
            namespace: Namespace within tenant (e.g., "shared", "policies")

        Returns:
            Scope ID like "tenant:acme_corp:shared"
        """
        return f"tenant:{tenant_id}:{namespace}"

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
        return f"global:{namespace}"

    @staticmethod
    def control_plane(tenant_id: str | None, namespace: str) -> str:
        """Control plane scope for system events (lifecycle, etc.).

        Used by MemoryManagementAgent to monitor agent lifecycle.

        Args:
            tenant_id: Tenant ID (None for system-level)
            namespace: Namespace (e.g., "lifecycle")

        Returns:
            Scope ID like "control_plane:tenant:acme_corp:lifecycle"
        """
        if tenant_id:
            return f"control_plane:tenant:{tenant_id}:{namespace}"
        return f"control_plane:system:{namespace}"

    # -------------------------------------------------------------------------
    # Query Patterns (for blackboard.query(namespace=pattern))
    # -------------------------------------------------------------------------

    @staticmethod
    def agent_all(agent_id: str) -> str:
        """Pattern matching all scopes for an agent.

        Args:
            agent_id: Unique agent identifier

        Returns:
            Pattern like "agent:abc123:*"
        """
        return f"agent:{agent_id}:*"

    @staticmethod
    def agent_ltm_all(agent_id: str) -> str:
        """Pattern matching all LTM sub-levels for an agent.

        Args:
            agent_id: Unique agent identifier

        Returns:
            Pattern like "agent:abc123:ltm:*"
        """
        return f"agent:{agent_id}:ltm:*"

    @staticmethod
    def collective_all(agent_type: str) -> str:
        """Pattern matching all collective scopes for an agent type.

        Args:
            agent_type: Agent type string

        Returns:
            Pattern like "agent_type:coding_agent:collective:*"
        """
        return f"agent_type:{agent_type}:collective:*"

    @staticmethod
    def all_agents_scope(level: str) -> str:
        """Pattern matching a specific level across all agents.

        Args:
            level: Memory level (e.g., "stm", "ltm:episodic")

        Returns:
            Pattern like "agent:*:stm"
        """
        return f"agent:*:{level}"

    @staticmethod
    def game_all(game_id: str) -> str:
        """Pattern matching all scopes for a game.

        Args:
            game_id: Unique game identifier

        Returns:
            Pattern like "game:game_789:*"
        """
        return f"game:{game_id}:*"

    @staticmethod
    def team_all(team_id: str) -> str:
        """Pattern matching all scopes for a team.

        Args:
            team_id: Unique team identifier

        Returns:
            Pattern like "team:team_456:*"
        """
        return f"team:{team_id}:*"

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
        if not scope_id.startswith("agent:"):
            return None
        parts = scope_id.split(":", 2)
        if len(parts) < 3:
            return None
        return (parts[1], parts[2])

    @staticmethod
    def parse_collective_scope(scope_id: str) -> tuple[str, str] | None:
        """Parse a collective scope ID.

        Args:
            scope_id: Scope ID to parse

        Returns:
            Tuple of (agent_type, level) or None if not a collective scope
        """
        if not scope_id.startswith("agent_type:"):
            return None
        parts = scope_id.split(":")
        if len(parts) < 4 or parts[2] != "collective":
            return None
        return (parts[1], parts[3])

    @staticmethod
    def is_agent_private(scope_id: str) -> bool:
        """Check if scope is agent-private.

        Args:
            scope_id: Scope ID to check

        Returns:
            True if this is an agent-private scope
        """
        return scope_id.startswith("agent:")

    @staticmethod
    def is_collective(scope_id: str) -> bool:
        """Check if scope is collective (per agent_type).

        Args:
            scope_id: Scope ID to check

        Returns:
            True if this is a collective scope
        """
        return scope_id.startswith("agent_type:") and ":collective:" in scope_id

    @staticmethod
    def is_shared(scope_id: str) -> bool:
        """Check if scope is shared (game, team, task).

        Args:
            scope_id: Scope ID to check

        Returns:
            True if this is a shared scope
        """
        return (
            scope_id.startswith("game:")
            or scope_id.startswith("team:")
            or scope_id.startswith("task:")
        )

    # -------------------------------------------------------------------------
    # Session Scopes (per tenant, entries tagged with session_id)
    # -------------------------------------------------------------------------

    @staticmethod
    def session(tenant_id: str, level: str = "memory") -> str:
        """Session memory scope for a tenant.

        All sessions within a tenant share this scope. Entries are tagged
        with session_id for filtering on retrieval.

        Args:
            tenant_id: Tenant identifier
            level: Sub-level within session scope (default: "memory")

        Returns:
            Scope ID like "session:acme_corp:memory"
        """
        return f"session:{tenant_id}:{level}"

    @staticmethod
    def session_all(tenant_id: str) -> str:
        """Pattern matching all session scopes for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Pattern like "session:acme_corp:*"
        """
        return f"session:{tenant_id}:*"

    @staticmethod
    def is_session_scope(scope_id: str) -> bool:
        """Check if scope is a session scope.

        Args:
            scope_id: Scope ID to check

        Returns:
            True if this is a session scope
        """
        return scope_id.startswith("session:")
