"""Session manager deployment for cluster-wide session management.

The SessionManagerDeployment is a Ray Serve deployment that manages sessions
across the cluster. It provides:
- Session lifecycle (create, activate, suspend, close)
- Session discovery (by tenant, ID)
- Session forking and merging (delegates to VCM branches)
- Tenant resource quota management
- Tenant resource usage tracking
- Agent run tracking and history

Sessions are thin wrappers around VCM branches. The session manager coordinates
with VCM for copy-on-write operations. It also serves as the authority for
tenant-level resource management, tracking quotas and usage across all of a
tenant's sessions and agents. Each agent invocation is tracked as an AgentRun.
"""

import logging
import time
import uuid
from typing import Any

from ...distributed.ray_utils import serving
from ...distributed import get_initialized_polymathera
from ...distributed.state_management import StateManager
from ...deployment_names import get_deployment_names
from .models import (
    Session,
    SessionState,
    SessionMetadata,
    SessionSystemState,
    TenantQuota,
    TenantResourceUsage,
    # Run tracking
    RunStatus,
    AgentRunConfig,
    RunResourceUsage,
    AgentRunEvent,
    AgentRun,
)

logger = logging.getLogger(__name__)


# TODO: This SessionManagerDeployment does not need to be a deployment itself.
# It can be a regular class instantiated by agents directly, since it does not
# need to be distributed. Refactor later.

@serving.deployment
class SessionManagerDeployment:
    """Manages sessions and tenant resources across the cluster.

    Provides:
    - Session lifecycle (create, activate, suspend, close)
    - Session discovery (by tenant, ID)
    - Session forking and merging (delegates to VCM branches)
    - Tenant resource quota management
    - Tenant resource usage tracking (called by AgentSystemDeployment)

    Design:
    - Sessions are thin wrappers around VCM branches
    - The session manager coordinates with VCM for CoW operations
    - All page operations go through VCM branch APIs
    - Tenant quotas and usage are stored in distributed state
    - AgentSystemDeployment calls increment/decrement methods for agent resources

    Example:
        ```python
        from polymathera.colony.agents.sessions import SessionManagerDeployment

        # Get session manager handle
        session_manager = serving.get_deployment(app_name, names.session_manager)

        # Set tenant quota
        await session_manager.set_tenant_quota(
            tenant_id="my-tenant",
            quota=TenantQuota(max_concurrent_sessions=20),
        )

        # Create a session
        session = await session_manager.create_session(
            tenant_id="my-tenant",
            metadata=SessionMetadata(
                tenant_id="my-tenant",
                created_by="user-123",
            ),
        )

        # Use session in context
        from polymathera.colony.agents.sessions import session_context
        async with session_context(session):
            # All operations use session's branch
            ...

        # Close session when done
        await session_manager.close_session(session.session_id)
        ```
    """

    def __init__(
        self,
        default_session_ttl: float = 86400.0,  # 24 hours
    ):
        """Initialize session manager.

        Args:
            default_session_ttl: Default session TTL in seconds (default 24 hours)
        """
        self.default_session_ttl = default_session_ttl

        # Initialized in initialize()
        self.app_name: str | None = None
        self.state_manager: StateManager[SessionSystemState] | None = None
        self.vcm_handle = None

    @serving.initialize_deployment
    async def initialize(self):
        """Initialize session manager."""
        self.app_name = serving.get_my_app_name()
        names = get_deployment_names()
        polymathera = await get_initialized_polymathera()

        # Initialize state manager
        self.state_manager = await polymathera.get_state_manager(
            state_type=SessionSystemState,
            state_key=SessionSystemState.get_state_key(self.app_name),
        )

        # Get VCM handle for branch operations
        self.vcm_handle = serving.get_deployment(self.app_name, names.vcm)

        logger.info("SessionManagerDeployment initialized")

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    @serving.endpoint
    async def create_session(
        self,
        tenant_id: str,
        metadata: SessionMetadata | None = None,
        ttl_seconds: float | None = None,
        fork_from_session_id: str | None = None,
    ) -> Session:
        """Create a new session.

        Creates a new VCM branch for this session's copy-on-write view.

        Args:
            tenant_id: Owning tenant
            metadata: Session metadata (uses defaults if None)
            ttl_seconds: Session TTL (uses default if None)
            fork_from_session_id: Fork from existing session's branch

        Returns:
            Created session

        Raises:
            ValueError: If tenant quota exceeded or parent session not found
        """
        # Check quota
        await self._check_session_quota(tenant_id)

        session_id = f"session_{uuid.uuid4().hex[:12]}"

        # Create VCM branch for this session
        parent_branch_id = None
        forked_from = None

        if fork_from_session_id:
            parent = await self.get_session(fork_from_session_id)
            if parent:
                parent_branch_id = parent.branch_id
                forked_from = fork_from_session_id
            else:
                raise ValueError(f"Parent session {fork_from_session_id} not found")

        # Create branch via VCM
        branch = await self.vcm_handle.create_branch(
            tenant_id=tenant_id,
            parent_branch_id=parent_branch_id,
            name=f"session_{session_id}",
        )

        # Create session
        session = Session(
            session_id=session_id,
            tenant_id=tenant_id,
            branch_id=branch.branch_id,
            state=SessionState.ACTIVE,
            metadata=metadata or SessionMetadata(
                tenant_id=tenant_id,
                created_by="session_manager",
            ),
            expires_at=time.time() + (ttl_seconds or self.default_session_ttl),
            forked_from_session_id=forked_from,
        )

        # Register session and update tenant resource usage
        async for state in self.state_manager.write_transaction():
            state.add_session(session)

            # Increment tenant's active session count
            usage = self._get_or_create_usage(state, tenant_id)
            usage.active_sessions += 1

        logger.info(
            f"Created session {session_id} with branch {branch.branch_id} "
            f"for tenant {tenant_id}"
            f"{f' (forked from {fork_from_session_id})' if forked_from else ''}"
        )

        return session

    @serving.endpoint
    async def get_session(self, session_id: str) -> Session | None:
        """Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found, None otherwise
        """
        async for state in self.state_manager.read_transaction():
            return state.sessions.get(session_id)

    @serving.endpoint
    async def list_sessions(
        self,
        tenant_id: str | None = None,
        state_filter: SessionState | None = None,
        include_expired: bool = False,
        limit: int = 100,
    ) -> list[Session]:
        """List sessions with optional filters.

        Args:
            tenant_id: Filter by tenant (None = all tenants)
            state_filter: Filter by state (None = all states)
            include_expired: Include expired sessions
            limit: Maximum number of sessions to return

        Returns:
            List of matching sessions
        """
        current_time = time.time()

        async for state in self.state_manager.read_transaction():
            sessions = []

            if tenant_id:
                session_ids = state.sessions_by_tenant.get(tenant_id, [])
            else:
                session_ids = list(state.sessions.keys())

            for sid in session_ids:
                if len(sessions) >= limit:
                    break

                session = state.sessions.get(sid)
                if not session:
                    continue

                # Apply state filter
                if state_filter is not None and session.state != state_filter:
                    continue

                # Apply expiration filter
                if not include_expired and session.is_expired(current_time):
                    continue

                sessions.append(session)

            return sessions

    @serving.endpoint
    async def activate_session(self, session_id: str) -> bool:
        """Activate a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was activated, False if not found or cannot be activated
        """
        async for state in self.state_manager.write_transaction():
            session = state.sessions.get(session_id)
            if not session:
                return False

            # Can only activate from CREATED or SUSPENDED states
            if session.state not in (SessionState.CREATED, SessionState.SUSPENDED):
                logger.warning(
                    f"Cannot activate session {session_id} from state {session.state}"
                )
                return False

            state.update_session_state(session_id, SessionState.ACTIVE)

        logger.info(f"Activated session {session_id}")
        return True

    @serving.endpoint
    async def suspend_session(self, session_id: str) -> bool:
        """Suspend a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was suspended, False if not found or cannot be suspended
        """
        async for state in self.state_manager.write_transaction():
            session = state.sessions.get(session_id)
            if not session:
                return False

            # Can only suspend from ACTIVE state
            if session.state != SessionState.ACTIVE:
                logger.warning(
                    f"Cannot suspend session {session_id} from state {session.state}"
                )
                return False

            state.update_session_state(session_id, SessionState.SUSPENDED)

        logger.info(f"Suspended session {session_id}")
        return True

    @serving.endpoint
    async def close_session(
        self,
        session_id: str,
        archive: bool = True,
    ) -> bool:
        """Close a session.

        Args:
            session_id: Session identifier
            archive: If True, archive the session for historical reference

        Returns:
            True if session was closed, False if not found
        """
        async for state in self.state_manager.write_transaction():
            session = state.sessions.get(session_id)
            if not session:
                return False

            # Only decrement if session was active
            was_active = session.state == SessionState.ACTIVE

            new_state = SessionState.ARCHIVED if archive else SessionState.CLOSED
            state.update_session_state(session_id, new_state)

            # Decrement tenant's active session count
            if was_active:
                usage = self._get_or_create_usage(state, session.tenant_id)
                usage.active_sessions = max(0, usage.active_sessions - 1)

        logger.info(f"Closed session {session_id} (archived={archive})")
        return True

    @serving.endpoint
    async def extend_session_ttl(
        self,
        session_id: str,
        additional_seconds: float,
    ) -> float | None:
        """Extend session TTL.

        Args:
            session_id: Session identifier
            additional_seconds: Seconds to add to TTL

        Returns:
            New expiration timestamp, or None if session not found
        """
        current_time = time.time()

        async for state in self.state_manager.write_transaction():
            session = state.sessions.get(session_id)
            if not session:
                return None

            # Extend from current expiration or from now if already expired
            base_time = max(session.expires_at or current_time, current_time)
            session.expires_at = base_time + additional_seconds
            session.updated_at = current_time

        logger.info(
            f"Extended session {session_id} TTL by {additional_seconds}s "
            f"(new expiration: {session.expires_at})"
        )
        return session.expires_at

    # =========================================================================
    # Session Merging (Delegates to VCM)
    # =========================================================================

    @serving.endpoint
    async def merge_sessions(
        self,
        source_session_id: str,
        target_session_id: str,
        strategy: str = "last_write_wins",
    ) -> dict[str, Any]:
        """Merge source session into target session.

        Delegates to VCM branch merge.

        Args:
            source_session_id: Session to merge from
            target_session_id: Session to merge into
            strategy: Merge strategy ("last_write_wins", "first_write_wins", "fail_on_conflict")

        Returns:
            Merge result dict with success status and details
        """
        source = await self.get_session(source_session_id)
        target = await self.get_session(target_session_id)

        if not source or not target:
            return {"success": False, "error": "Session not found"}

        if source.tenant_id != target.tenant_id:
            return {"success": False, "error": "Cannot merge sessions from different tenants"}

        # Delegate to VCM branch merge
        result = await self.vcm_handle.merge_branches(
            source_branch_id=source.branch_id,
            target_branch_id=target.branch_id,
            strategy=strategy,
        )

        if result.get("success"):
            # Update session states
            async for state in self.state_manager.write_transaction():
                state.update_session_state(source_session_id, SessionState.CLOSED)
                source_session = state.sessions.get(source_session_id)
                if source_session:
                    source_session.merged_into_session_id = target_session_id

            logger.info(f"Merged session {source_session_id} into {target_session_id}")

        return result

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    @serving.endpoint
    async def enter_session(self, session_id: str) -> Session:
        """Get session for use with context manager.

        Use with session_context() for scoped operations:

            session = await session_manager.enter_session(session_id)
            async with session_context(session):
                # All operations use session's branch
                ...

        Args:
            session_id: Session identifier

        Returns:
            Session ready for use in context

        Raises:
            ValueError: If session not found or not active
        """
        session: Session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.state != SessionState.ACTIVE:
            raise ValueError(f"Session {session_id} is not active (state={session.state})")

        if session.is_expired():
            raise ValueError(f"Session {session_id} has expired")

        # Update last activity
        async for state in self.state_manager.write_transaction():
            s = state.sessions.get(session_id)
            if s:
                s.updated_at = time.time()

        return session

    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================

    @serving.endpoint
    async def get_stats(self) -> dict[str, Any]:
        """Get session manager statistics.

        Returns:
            Dictionary with session statistics
        """
        async for state in self.state_manager.read_transaction():
            # Count sessions by state
            sessions_by_state: dict[str, int] = {}
            for session in state.sessions.values():
                state_name = session.state.value
                sessions_by_state[state_name] = sessions_by_state.get(state_name, 0) + 1

            # Count sessions by tenant
            sessions_by_tenant = {
                tenant_id: len(session_ids)
                for tenant_id, session_ids in state.sessions_by_tenant.items()
            }

            return {
                "total_sessions": len(state.sessions),
                "active_session_count": state.active_session_count,
                "sessions_by_state": sessions_by_state,
                "sessions_by_tenant": sessions_by_tenant,
                "num_tenants": len(state.sessions_by_tenant),
            }

    # =========================================================================
    # Cleanup
    # =========================================================================

    @serving.endpoint
    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        current_time = time.time()
        cleaned_count = 0

        async for state in self.state_manager.write_transaction():
            expired_ids = [
                session_id
                for session_id, session in state.sessions.items()
                if session.is_expired(current_time)
            ]

            for session_id in expired_ids:
                state.remove_session(session_id)
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired sessions")

        return cleaned_count

    # =========================================================================
    # Tenant Quota Management
    # =========================================================================

    @serving.endpoint
    async def set_tenant_quota(
        self,
        tenant_id: str,
        quota: TenantQuota,
    ) -> None:
        """Set resource quota for a tenant.

        Args:
            tenant_id: Tenant identifier
            quota: Resource quota to set
        """
        async for state in self.state_manager.write_transaction():
            state.tenant_quotas[tenant_id] = quota

        logger.info(f"Set quota for tenant {tenant_id}: {quota}")

    @serving.endpoint
    async def get_tenant_quota(self, tenant_id: str) -> TenantQuota:
        """Get resource quota for a tenant.

        Returns default quota if not explicitly set.

        Args:
            tenant_id: Tenant identifier

        Returns:
            TenantQuota for the tenant
        """
        async for state in self.state_manager.read_transaction():
            return state.tenant_quotas.get(tenant_id, TenantQuota())

    @serving.endpoint
    async def get_tenant_resource_usage(self, tenant_id: str) -> TenantResourceUsage:
        """Get current resource usage for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            TenantResourceUsage for the tenant
        """
        async for state in self.state_manager.read_transaction():
            return state.tenant_resource_usage.get(tenant_id, TenantResourceUsage())

    # =========================================================================
    # Tenant Resource Tracking (called by AgentSystemDeployment)
    # =========================================================================

    @serving.endpoint
    async def increment_tenant_resources(
        self,
        tenant_id: str,
        cpu_cores: float = 0.0,
        memory_mb: int = 0,
        gpu_cores: float = 0.0,
        gpu_memory_mb: int = 0,
    ) -> None:
        """Increment resource usage for a tenant.

        Called by AgentSystemDeployment when an agent is registered.

        Args:
            tenant_id: Tenant identifier
            cpu_cores: CPU cores to add
            memory_mb: Memory (MB) to add
            gpu_cores: GPU cores to add
            gpu_memory_mb: GPU memory (MB) to add
        """
        async for state in self.state_manager.write_transaction():
            usage = self._get_or_create_usage(state, tenant_id)
            usage.active_agents += 1
            usage.total_cpu_cores += cpu_cores
            usage.total_memory_mb += memory_mb
            usage.total_gpu_cores += gpu_cores
            usage.total_gpu_memory_mb += gpu_memory_mb

        logger.debug(
            f"Incremented resources for tenant {tenant_id}: "
            f"cpu={cpu_cores}, mem={memory_mb}MB, gpu={gpu_cores}, gpu_mem={gpu_memory_mb}MB"
        )

    @serving.endpoint
    async def decrement_tenant_resources(
        self,
        tenant_id: str,
        cpu_cores: float = 0.0,
        memory_mb: int = 0,
        gpu_cores: float = 0.0,
        gpu_memory_mb: int = 0,
    ) -> None:
        """Decrement resource usage for a tenant.

        Called by AgentSystemDeployment when an agent is unregistered.

        Args:
            tenant_id: Tenant identifier
            cpu_cores: CPU cores to remove
            memory_mb: Memory (MB) to remove
            gpu_cores: GPU cores to remove
            gpu_memory_mb: GPU memory (MB) to remove
        """
        async for state in self.state_manager.write_transaction():
            usage = self._get_or_create_usage(state, tenant_id)
            usage.active_agents = max(0, usage.active_agents - 1)
            usage.total_cpu_cores = max(0.0, usage.total_cpu_cores - cpu_cores)
            usage.total_memory_mb = max(0, usage.total_memory_mb - memory_mb)
            usage.total_gpu_cores = max(0.0, usage.total_gpu_cores - gpu_cores)
            usage.total_gpu_memory_mb = max(0, usage.total_gpu_memory_mb - gpu_memory_mb)

        logger.debug(
            f"Decremented resources for tenant {tenant_id}: "
            f"cpu={cpu_cores}, mem={memory_mb}MB, gpu={gpu_cores}, gpu_mem={gpu_memory_mb}MB"
        )

    @serving.endpoint
    async def check_tenant_agent_quota(
        self,
        tenant_id: str,
        cpu_cores: float = 0.0,
        memory_mb: int = 0,
        gpu_cores: float = 0.0,
        gpu_memory_mb: int = 0,
    ) -> bool:
        """Check if tenant can spawn an agent with the given resources.

        Called by AgentSystemDeployment before spawning an agent.

        Args:
            tenant_id: Tenant identifier
            cpu_cores: CPU cores needed
            memory_mb: Memory (MB) needed
            gpu_cores: GPU cores needed
            gpu_memory_mb: GPU memory (MB) needed

        Returns:
            True if tenant has sufficient quota, False otherwise
        """
        async for state in self.state_manager.read_transaction():
            quota = self._get_quota(state, tenant_id)
            usage = state.tenant_resource_usage.get(tenant_id, TenantResourceUsage())

            # Check all resource limits
            if usage.active_agents >= quota.max_concurrent_agents:
                logger.warning(
                    f"Tenant {tenant_id} agent quota exceeded: "
                    f"{usage.active_agents}/{quota.max_concurrent_agents}"
                )
                return False

            if usage.total_cpu_cores + cpu_cores > quota.max_total_cpu_cores:
                logger.warning(
                    f"Tenant {tenant_id} CPU quota exceeded: "
                    f"{usage.total_cpu_cores + cpu_cores}/{quota.max_total_cpu_cores}"
                )
                return False

            if usage.total_memory_mb + memory_mb > quota.max_total_memory_mb:
                logger.warning(
                    f"Tenant {tenant_id} memory quota exceeded: "
                    f"{usage.total_memory_mb + memory_mb}/{quota.max_total_memory_mb}"
                )
                return False

            if usage.total_gpu_cores + gpu_cores > quota.max_total_gpu_cores:
                logger.warning(
                    f"Tenant {tenant_id} GPU quota exceeded: "
                    f"{usage.total_gpu_cores + gpu_cores}/{quota.max_total_gpu_cores}"
                )
                return False

            if usage.total_gpu_memory_mb + gpu_memory_mb > quota.max_total_gpu_memory_mb:
                logger.warning(
                    f"Tenant {tenant_id} GPU memory quota exceeded: "
                    f"{usage.total_gpu_memory_mb + gpu_memory_mb}/{quota.max_total_gpu_memory_mb}"
                )
                return False

            return True

    # =========================================================================
    # Agent Run Management
    # =========================================================================

    @serving.endpoint
    async def create_run(
        self,
        session_id: str,
        agent_id: str,
        input_data: dict[str, Any],
        config: AgentRunConfig | None = None,
        timeout: float = 30.0,
        track_events: bool = False,
        parent_run_id: str | None = None,
    ) -> AgentRun:
        """Create and register a new agent run.

        Called by AgentHandle.run() to track agent invocations.

        Args:
            session_id: Owning session
            agent_id: Target agent
            input_data: Input data for the run
            config: Run configuration (uses session defaults if None)
            timeout: Timeout in seconds
            track_events: Whether to record intermediate events
            parent_run_id: Parent run if this is a nested call

        Returns:
            Created AgentRun
        """
        session = await self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Merge config with session defaults
        effective_config = config or session.default_run_config or AgentRunConfig()

        run = AgentRun(
            session_id=session_id,
            tenant_id=session.tenant_id,
            agent_id=agent_id,
            config=effective_config,
            input_data=input_data,
            timeout_seconds=timeout,
            track_events=track_events,
            parent_run_id=parent_run_id,
        )

        async for state in self.state_manager.write_transaction():
            state.runs[run.run_id] = run

            # Update session index
            if session_id not in state.runs_by_session:
                state.runs_by_session[session_id] = []
            state.runs_by_session[session_id].append(run.run_id)

            # Track active run
            state.active_runs[run.run_id] = session_id

        logger.debug(f"Created run {run.run_id} for agent {agent_id} in session {session_id}")
        return run

    @serving.endpoint
    async def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        """Update run status.

        Called when a run completes, fails, or is cancelled.

        Args:
            run_id: Run identifier
            status: New status
            output_data: Output data (for completed runs)
            error: Error message (for failed runs)

        Returns:
            True if run was found and updated
        """
        current_time = time.time()

        async for state in self.state_manager.write_transaction():
            run = state.runs.get(run_id)
            if not run:
                return False

            # Update status
            old_status = run.status
            run.status = status
            run.output_data = output_data
            run.error = error

            # Update timing
            if status == RunStatus.RUNNING and run.started_at is None:
                run.started_at = current_time
            if status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.TIMEOUT):
                run.completed_at = current_time
                run.resource_usage.wall_time_seconds = current_time - (run.started_at or run.created_at)

                # Remove from active runs
                state.active_runs.pop(run_id, None)

            logger.debug(f"Updated run {run_id} status: {old_status} -> {status}")
            return True

    @serving.endpoint
    async def add_run_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> bool:
        """Add an event to a tracked run.

        Only works if run has track_events=True.

        Args:
            run_id: Run identifier
            event_type: Event type (llm_call, tool_use, child_spawn, etc.)
            data: Event data

        Returns:
            True if event was added
        """
        # TODO: This is too expensive because it competes with
        # all other operations by all other sessions. Store events
        # somewhere else more efficiently.
        async for state in self.state_manager.write_transaction():
            run = state.runs.get(run_id)
            if not run or not run.track_events:
                return False

            event = AgentRunEvent(event_type=event_type, data=data)
            run.events.append(event)
            return True

    @serving.endpoint
    async def update_run_resources(
        self,
        run_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        llm_calls: int = 0,
        child_agent_id: str | None = None,
    ) -> bool:
        """Update resource usage for a run.

        Called by agents to report resource consumption.

        Args:
            run_id: Run identifier
            input_tokens: Input tokens to add
            output_tokens: Output tokens to add
            llm_calls: LLM calls to add
            child_agent_id: Child agent spawned (if any)

        Returns:
            True if run was found and updated
        """
        async for state in self.state_manager.write_transaction():
            run = state.runs.get(run_id)
            if not run:
                return False

            run.resource_usage.input_tokens += input_tokens
            run.resource_usage.output_tokens += output_tokens
            run.resource_usage.total_tokens += input_tokens + output_tokens
            run.resource_usage.llm_calls += llm_calls

            if child_agent_id:
                run.resource_usage.agents_spawned += 1
                run.resource_usage.child_agent_ids.append(child_agent_id)

            return True

    @serving.endpoint
    async def get_run(self, run_id: str) -> AgentRun | None:
        """Get a run by ID.

        Args:
            run_id: Run identifier

        Returns:
            AgentRun if found, None otherwise
        """
        async for state in self.state_manager.read_transaction():
            return state.runs.get(run_id)

    @serving.endpoint
    async def get_session_runs(
        self,
        session_id: str,
        status_filter: RunStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRun]:
        """Get runs for a session.

        Args:
            session_id: Session identifier
            status_filter: Filter by status (None = all)
            limit: Maximum runs to return

        Returns:
            List of runs (most recent first)
        """
        async for state in self.state_manager.read_transaction():
            run_ids = state.runs_by_session.get(session_id, [])

            runs = []
            for run_id in reversed(run_ids):  # Most recent first
                if len(runs) >= limit:
                    break

                run = state.runs.get(run_id)
                if not run:
                    continue

                if status_filter is not None and run.status != status_filter:
                    continue

                runs.append(run)

            return runs

    @serving.endpoint
    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a running run.

        Args:
            run_id: Run identifier

        Returns:
            True if run was cancelled
        """
        async for state in self.state_manager.write_transaction():
            run = state.runs.get(run_id)
            if not run:
                return False

            if run.is_terminal():
                return False  # Already finished

            run.status = RunStatus.CANCELLED
            run.completed_at = time.time()
            run.resource_usage.wall_time_seconds = run.completed_at - (run.started_at or run.created_at)

            state.active_runs.pop(run_id, None)

            logger.info(f"Cancelled run {run_id}")
            return True

    @serving.endpoint
    async def set_session_default_config(
        self,
        session_id: str,
        config: AgentRunConfig,
    ) -> bool:
        """Set default run config for a session.

        Args:
            session_id: Session identifier
            config: Default configuration

        Returns:
            True if session was found and updated
        """
        async for state in self.state_manager.write_transaction():
            session = state.sessions.get(session_id)
            if not session:
                return False

            session.default_run_config = config
            session.updated_at = time.time()
            return True

    @serving.endpoint
    async def get_session_default_config(
        self,
        session_id: str,
    ) -> AgentRunConfig | None:
        """Get default run config for a session.

        Args:
            session_id: Session identifier

        Returns:
            Default config or None if not set
        """
        async for state in self.state_manager.read_transaction():
            session = state.sessions.get(session_id)
            if not session:
                return None
            return session.default_run_config

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _check_session_quota(self, tenant_id: str) -> None:
        """Check if tenant can create more sessions.

        Args:
            tenant_id: Tenant identifier

        Raises:
            ValueError: If tenant has exceeded session quota
        """
        async for state in self.state_manager.read_transaction():
            quota = state.tenant_quotas.get(tenant_id, TenantQuota())
            usage = state.tenant_resource_usage.get(tenant_id, TenantResourceUsage())

            if usage.active_sessions >= quota.max_concurrent_sessions:
                raise ValueError(
                    f"Tenant {tenant_id} has reached maximum sessions "
                    f"({usage.active_sessions}/{quota.max_concurrent_sessions})"
                )

    def _get_or_create_usage(
        self,
        state: SessionSystemState,
        tenant_id: str,
    ) -> TenantResourceUsage:
        """Get or create TenantResourceUsage for a tenant.

        Args:
            state: Current state
            tenant_id: Tenant identifier

        Returns:
            TenantResourceUsage for the tenant
        """
        if tenant_id not in state.tenant_resource_usage:
            state.tenant_resource_usage[tenant_id] = TenantResourceUsage()
        return state.tenant_resource_usage[tenant_id]

    def _get_quota(
        self,
        state: SessionSystemState,
        tenant_id: str,
    ) -> TenantQuota:
        """Get TenantQuota for a tenant (uses default if not set).

        Args:
            state: Current state
            tenant_id: Tenant identifier

        Returns:
            TenantQuota for the tenant
        """
        return state.tenant_quotas.get(tenant_id, TenantQuota())
