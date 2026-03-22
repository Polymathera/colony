"""Session models for the multi-agent system.

This module defines the core data models for session management:
- Session: A conversation session with the multi-agent system
- SessionState: Lifecycle states for sessions
- SessionMetadata: Session metadata
- SessionSystemState: Distributed state for session registry
- TenantQuota: Resource quotas for a tenant
- TenantResourceUsage: Current resource usage for a tenant
- AgentRun: A tracked agent invocation within a session
- AgentRunConfig: Configuration for an agent run
- RunStatus: Lifecycle states for an agent run
- RunResourceUsage: Resource usage for a single run
- AgentRunEvent: An event during an agent run

Sessions are the user-facing abstraction for conversations. Key points:
- A session has an associated VCM branch for copy-on-write
- All agent interactions within the session use that branch
- Sessions can be forked (creating a new branch) and merged
- Sessions are NOT agent groups - they are contexts for user interactions
- Sessions track AgentRun objects for history and auditing

Tenant resource management:
- TenantQuota and TenantResourceUsage are TENANT-level concepts
- They track resources across all of a tenant's sessions and agents
- Agents work within session contexts but do NOT belong to sessions
"""

import time
import uuid
from enum import Enum
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from ...distributed.state_management import SharedState
from ...distributed.ray_utils import serving

if TYPE_CHECKING:
    from .context import SessionContextManager


class SessionState(str, Enum):
    """Lifecycle states for a session."""

    CREATED = "created"
    """Session created but not yet activated."""

    ACTIVE = "active"
    """Session is active and accepting operations."""

    SUSPENDED = "suspended"
    """Session is temporarily suspended (can be resumed)."""

    CLOSED = "closed"
    """Session is closed and cannot be resumed."""

    ARCHIVED = "archived"
    """Session is archived for historical reference."""


class SessionMetadata(BaseModel):
    """Metadata for a session.

    Contains descriptive and administrative information about a session.

    Attributes:
        syscontext: Execution context for this session
        name: Human-readable name
        description: Session description
        created_by: ID of user/system that created the session
        tags: Set of tags for categorization
        parent_session_id: Parent session if this was forked
        custom: Custom metadata for application use
    """
    syscontext: serving.ExecutionContext = Field(
        default_factory=serving.require_execution_context,
        description="Execution context for this session"
    )
    name: str | None = Field(None, description="Human-readable name")
    description: str | None = Field(None, description="Session description")
    created_by: str = Field(..., description="User/system that created the session")
    tags: set[str] = Field(default_factory=set, description="Tags for categorization")
    parent_session_id: str | None = Field(None, description="Parent session if forked")
    custom: dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class TenantQuota(BaseModel):
    """Resource quotas for a tenant.

    Defines limits on concurrent sessions, agents, and resource consumption
    to prevent resource exhaustion and ensure fair allocation across tenants.

    These are TENANT-level quotas, not session-level. A tenant's total resource
    usage across all their sessions and agents is tracked against these limits.
    """

    max_concurrent_sessions: int = Field(
        default=10,
        ge=1,
        description="Maximum concurrent active sessions for this tenant"
    )
    max_concurrent_agents: int = Field(
        default=100,
        ge=1,
        description="Maximum concurrent agents across all sessions"
    )
    max_total_cpu_cores: float = Field(
        default=10.0,
        ge=0.1,
        description="Total CPU cores available to this tenant"
    )
    max_total_memory_mb: int = Field(
        default=10240,  # 10GB
        ge=512,
        description="Total memory (MB) available to this tenant"
    )
    max_total_gpu_cores: float = Field(
        default=2.0,
        ge=0.0,
        description="Total GPU cores available to this tenant"
    )
    max_total_gpu_memory_mb: int = Field(
        default=16384,  # 16GB
        ge=0,
        description="Total GPU memory (MB) available to this tenant"
    )


class TenantResourceUsage(BaseModel):
    """Current resource usage for a tenant.

    Tracks actual consumption across all active sessions and agents.
    This is TENANT-level tracking, not session-level.
    """

    active_sessions: int = Field(default=0, ge=0, description="Number of active sessions")
    active_agents: int = Field(default=0, ge=0, description="Number of active agents")
    total_cpu_cores: float = Field(default=0.0, ge=0.0, description="CPU cores in use")
    total_memory_mb: int = Field(default=0, ge=0, description="Memory (MB) in use")
    total_gpu_cores: float = Field(default=0.0, ge=0.0, description="GPU cores in use")
    total_gpu_memory_mb: int = Field(default=0, ge=0, description="GPU memory (MB) in use")


# =============================================================================
# Agent Run Models
# =============================================================================


class RunStatus(str, Enum):
    """Lifecycle states for an agent run."""

    PENDING = "pending"
    """Run created but not started."""

    RUNNING = "running"
    """Run is currently executing."""

    COMPLETED = "completed"
    """Run finished successfully."""

    FAILED = "failed"
    """Run finished with error."""

    CANCELLED = "cancelled"
    """Run was cancelled by user/system."""

    TIMEOUT = "timeout"
    """Run exceeded timeout."""


class AgentRunConfig(BaseModel):
    """Configuration for an agent run.

    These parameters control agent execution behavior. Can be set as
    session defaults or overridden per-run.
    """

    # LLM configuration
    model_type: str | None = Field(None, description="LLM model to use")
    max_context_size: int | None = Field(None, description="Maximum context window size")

    # Tool configuration
    tools: list[str] | None = Field(None, description="Tools available to the agent")

    # Scheduling
    priority: int = Field(default=5, ge=1, le=10, description="Run priority (1-10)")

    # Output
    output_schema: dict[str, Any] | None = Field(
        None, description="Expected output schema for validation"
    )

    # VCM
    initial_vcm_pages: list[str] | None = Field(
        None, description="VCM pages to load initially"
    )

    # Metadata
    task_description: str | None = Field(None, description="Human-readable task description")
    tags: set[str] = Field(default_factory=set, description="Tags for categorization")


class RunResourceUsage(BaseModel):
    """Resource usage for a single run.

    Tracks consumption including the full tree of child agents spawned.
    """

    # Time
    wall_time_seconds: float = Field(default=0.0, ge=0.0, description="Wall clock time")

    # LLM usage
    input_tokens: int = Field(default=0, ge=0, description="Input tokens consumed")
    output_tokens: int = Field(default=0, ge=0, description="Output tokens generated")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens (input + output)")
    llm_calls: int = Field(default=0, ge=0, description="Number of LLM API calls")

    # Cost tracking (for remote LLM deployments)
    cost_usd: float = Field(default=0.0, ge=0.0, description="Estimated cost in USD")
    cache_read_tokens: int = Field(default=0, ge=0, description="Tokens read from cache (prefix cache hits)")
    cache_write_tokens: int = Field(default=0, ge=0, description="Tokens written to cache (prefix cache creation)")

    # Per-agent breakdown (agent_id → {input_tokens, output_tokens, llm_calls, cost_usd, ...})
    per_agent: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Token usage broken down by agent_id"
    )

    # Agent tree
    agents_spawned: int = Field(default=0, ge=0, description="Child agents spawned")
    child_agent_ids: list[str] = Field(
        default_factory=list, description="IDs of spawned child agents"
    )

    # Compute (aggregated from all agents in tree)
    cpu_seconds: float = Field(default=0.0, ge=0.0, description="CPU time consumed")
    memory_mb_seconds: float = Field(default=0.0, ge=0.0, description="Memory * time")
    gpu_seconds: float = Field(default=0.0, ge=0.0, description="GPU time consumed")


class AgentRunEvent(BaseModel):
    """An event that occurred during an agent run.

    Only recorded for runs with track_events=True.
    """

    event_id: str = Field(
        default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}",
        description="Unique event identifier"
    )
    timestamp: float = Field(default_factory=time.time, description="Event timestamp")
    event_type: str = Field(..., description="Event type (llm_call, tool_use, child_spawn, etc.)")
    data: dict[str, Any] = Field(default_factory=dict, description="Event data")


class AgentRun(BaseModel):
    """A tracked agent invocation within a session.

    Created when AgentHandle.run() or run_streamed() is called.
    Persisted in SessionSystemState for history and auditing.
    """

    # Identity
    run_id: str = Field(
        default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}",
        description="Unique run identifier"
    )
    session_id: str = Field(..., description="Owning session")
    tenant_id: str = Field(..., description="Owning tenant")

    # Target
    agent_id: str = Field(..., description="The agent that was invoked")
    agent_type: str | None = Field(None, description="Type of the target agent")

    # Configuration
    config: AgentRunConfig = Field(default_factory=AgentRunConfig, description="Run configuration")

    # Execution
    status: RunStatus = Field(default=RunStatus.PENDING, description="Current status")
    input_data: dict[str, Any] = Field(default_factory=dict, description="Input data")
    output_data: dict[str, Any] | None = Field(None, description="Output data (when completed)")
    error: str | None = Field(None, description="Error message (when failed)")

    # Timing
    created_at: float = Field(default_factory=time.time, description="Creation timestamp")
    started_at: float | None = Field(None, description="Start timestamp")
    completed_at: float | None = Field(None, description="Completion timestamp")
    timeout_seconds: float = Field(default=30.0, description="Timeout in seconds")

    # Resource tracking
    resource_usage: RunResourceUsage = Field(
        default_factory=RunResourceUsage, description="Resource consumption"
    )

    # Events (only populated if track_events=True)
    track_events: bool = Field(default=False, description="Whether to record events")
    events: list[AgentRunEvent] = Field(default_factory=list, description="Run events")

    # Parent run (if this is a nested call)
    parent_run_id: str | None = Field(None, description="Parent run if nested")

    def duration_seconds(self) -> float | None:
        """Get run duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    def is_terminal(self) -> bool:
        """Check if run is in a terminal state."""
        return self.status in (
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMEOUT,
        )


class Session(BaseModel):
    """A conversation session with the multi-agent system.

    Sessions are the user-facing abstraction for conversations with agents.
    They are NOT agent groups - they are contexts for user interactions.

    Key concepts:
    - A session has an associated VCM branch for copy-on-write
    - All agent interactions within the session use that branch
    - The session tracks conversation history via session_id tags
    - Sessions can be forked (creating a new branch) and merged

    One session = one working branch. If a user wants to work on multiple
    branches, they should create multiple sessions. Multi-branch operations
    (compare, merge) are done at the VCM level.

    Attributes:
        session_id: Unique identifier
        syscontext: Execution context for this session
        branch_id: Associated VCM branch for copy-on-write
        state: Current lifecycle state
        metadata: Session metadata
        created_at: Creation timestamp
        updated_at: Last activity timestamp
        expires_at: Expiration timestamp (None = no expiration)
        forked_from_session_id: Parent session if forked
        merged_into_session_id: Target session if merged
    """

    session_id: str = Field(
        default_factory=lambda: f"session_{uuid.uuid4().hex[:12]}",
        description="Unique session identifier"
    )
    syscontext: serving.ExecutionContext = Field(
        default_factory=serving.require_execution_context,
        description="Execution context for this session"
    )
    branch_id: str = Field(..., description="Associated VCM branch for copy-on-write")
    state: SessionState = Field(
        default=SessionState.CREATED,
        description="Current lifecycle state"
    )
    metadata: SessionMetadata = Field(..., description="Session metadata")

    # Timestamps
    created_at: float = Field(default_factory=time.time, description="Creation timestamp")
    updated_at: float = Field(default_factory=time.time, description="Last activity timestamp")
    expires_at: float | None = Field(None, description="Expiration timestamp")

    # Forking/merging
    forked_from_session_id: str | None = Field(
        None,
        description="Parent session if this was forked"
    )
    merged_into_session_id: str | None = Field(
        None,
        description="Target session if this was merged"
    )

    # Default run configuration for this session
    default_run_config: AgentRunConfig | None = Field(
        None,
        description="Default configuration for agent runs in this session"
    )

    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state == SessionState.ACTIVE

    def is_expired(self, current_time: float | None = None) -> bool:
        """Check if session has expired.

        Args:
            current_time: Current timestamp (uses time.time() if None)

        Returns:
            True if session has expired, False otherwise
        """
        if self.expires_at is None:
            return False
        check_time = current_time if current_time is not None else time.time()
        return check_time > self.expires_at

    def remaining_time_s(self, current_time: float | None = None) -> float | None:
        """Get remaining time until expiration in seconds.

        Args:
            current_time: Current timestamp (uses time.time() if None)

        Returns:
            Remaining seconds, or None if session doesn't expire
        """
        if self.expires_at is None:
            return None
        check_time = current_time if current_time is not None else time.time()
        remaining = self.expires_at - check_time
        return max(0.0, remaining)

    def context(self) -> "SessionContextManager":
        """Get a context manager for this session.

        All operations within the context will use this session's VCM branch
        and be tagged with this session_id.

        Example:
            ```python
            async with session.context():
                # All agent work here uses session's branch
                await agent_handle.run({"query": "analyze code"})
                # Results are on session's branch
            ```

        Returns:
            Async context manager for this session
        """
        from .context import SessionContextManager
        return SessionContextManager(self)


class SessionSystemState(SharedState):
    """Distributed state for session registry, tenant resources, and run tracking.

    Stored in StateManager for cluster-wide access. Tracks all sessions
    and provides indexing for efficient lookups. Also manages tenant-level
    resource quotas, usage tracking, and agent run history.

    Attributes:
        sessions: All sessions by session_id
        sessions_by_tenant: Index of session_ids by tenant_id
        active_session_count: Count of currently active sessions
        tenant_quotas: Resource quotas per tenant
        tenant_resource_usage: Current resource usage per tenant
        runs: All agent runs by run_id
        runs_by_session: Index of run_ids by session_id
        active_runs: Currently active runs (run_id -> session_id)
    """

    sessions: dict[str, Session] = Field(
        default_factory=dict,
        description="All sessions by session_id"
    )
    sessions_by_tenant: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Index of session_ids by tenant_id"
    )
    active_session_count: int = Field(
        default=0,
        description="Count of currently active sessions"
    )

    # Tenant resource management
    tenant_quotas: dict[str, TenantQuota] = Field(
        default_factory=dict,
        description="Resource quotas per tenant (tenant_id -> TenantQuota)"
    )
    tenant_resource_usage: dict[str, TenantResourceUsage] = Field(
        default_factory=dict,
        description="Current resource usage per tenant (tenant_id -> TenantResourceUsage)"
    )

    # Run tracking
    runs: dict[str, AgentRun] = Field(
        default_factory=dict,
        description="All runs by run_id"
    )
    runs_by_session: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Index of run_ids by session_id"
    )
    active_runs: dict[str, str] = Field(
        default_factory=dict,
        description="Active runs (run_id -> session_id)"
    )

    @classmethod
    def get_state_key(cls, app_name: str) -> str:
        """Generate state key for this session registry."""
        return f"polymathera:serving:{app_name}:sessions:system"

    def add_session(self, session: Session) -> None:
        """Add a session to the registry.

        Args:
            session: Session to add
        """
        self.sessions[session.session_id] = session

        # Update tenant index
        if session.tenant_id not in self.sessions_by_tenant:
            self.sessions_by_tenant[session.tenant_id] = []
        if session.session_id not in self.sessions_by_tenant[session.tenant_id]:
            self.sessions_by_tenant[session.tenant_id].append(session.session_id)

        # Update count
        if session.state == SessionState.ACTIVE:
            self.active_session_count += 1

    def remove_session(self, session_id: str) -> Session | None:
        """Remove a session from the registry.

        Args:
            session_id: Session ID to remove

        Returns:
            Removed session if found, None otherwise
        """
        session = self.sessions.pop(session_id, None)
        if session:
            # Update tenant index
            if session.tenant_id in self.sessions_by_tenant:
                if session_id in self.sessions_by_tenant[session.tenant_id]:
                    self.sessions_by_tenant[session.tenant_id].remove(session_id)
                if not self.sessions_by_tenant[session.tenant_id]:
                    del self.sessions_by_tenant[session.tenant_id]

            # Update count
            if session.state == SessionState.ACTIVE:
                self.active_session_count -= 1

        return session

    def update_session_state(self, session_id: str, new_state: SessionState) -> bool:
        """Update a session's state.

        Args:
            session_id: Session ID
            new_state: New state

        Returns:
            True if session was found and updated, False otherwise
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        old_state = session.state

        # Update count based on state transition
        if old_state == SessionState.ACTIVE and new_state != SessionState.ACTIVE:
            self.active_session_count -= 1
        elif old_state != SessionState.ACTIVE and new_state == SessionState.ACTIVE:
            self.active_session_count += 1

        session.state = new_state
        session.updated_at = time.time()

        return True

    def get_sessions_for_tenant(self, tenant_id: str) -> list[Session]:
        """Get all sessions for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of sessions for this tenant
        """
        session_ids = self.sessions_by_tenant.get(tenant_id, [])
        return [self.sessions[sid] for sid in session_ids if sid in self.sessions]
