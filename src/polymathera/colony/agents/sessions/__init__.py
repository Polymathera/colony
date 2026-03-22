"""Session Management for Multi-Agent System.

This module provides session management capabilities:
- Session lifecycle (create, fork, merge, close)
- Session context manager for scoped operations
- Copy-on-write via VCM branches per session
- Tenant resource quota management
- Tenant resource usage tracking
- Agent run tracking and history

Sessions are the user-facing abstraction for conversations with agents.
Key concepts:
- A session has an associated VCM branch for copy-on-write
- All agent interactions within the session use that branch
- Sessions can be forked (creating a new branch) and merged
- TenantQuota and TenantResourceUsage track tenant-level resources
- AgentRun tracks each agent invocation with input, output, and resource usage

Example:
    ```python
    from polymathera.colony.agents.sessions import (
        SessionManagerDeployment,
        get_current_session,
    )

    with execution_context(
        ring=Ring.USER,
        colony_id="colony-456",
        tenant_id="tenant-1",
        session_id="session-789",
        run_id="run-abc",
        origin="cli",
    ):
        # Get session manager handle
        session_manager = serving.get_deployment(app_name, names.session_manager)

        # Create a session
        session = await session_manager.create_session(
            metadata=SessionMetadata(
                created_by="user-123",
            ),
        )

        # Use session in context (preferred: use session.context() method)
        async with session.context():
            # All operations use session's branch
            current = get_current_session()
            assert current.session_id == session.session_id

            # Agent work here uses session's VCM branch
            ...

        # Close session when done
        await session_manager.close_session(session.session_id)
    ```

See IMPLEMENTATION_PLAN.md for detailed design.
"""

# Models
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

# Context manager and utilities
from .context import (
    SessionContextManager,
    session_context,
    session_id_context,
    get_current_session,
    get_current_session_id,
    get_current_branch_id,
    get_current_tenant_id,
    set_current_session_id,
    reset_current_session_id,
    SessionScope,
)

# Manager deployment
from .manager import SessionManagerDeployment


__all__ = [
    # Models
    "Session",
    "SessionState",
    "SessionMetadata",
    "SessionSystemState",
    "TenantQuota",
    "TenantResourceUsage",
    # Run tracking
    "RunStatus",
    "AgentRunConfig",
    "RunResourceUsage",
    "AgentRunEvent",
    "AgentRun",
    # Context
    "SessionContextManager",
    "session_context",
    "session_id_context",
    "get_current_session",
    "get_current_session_id",
    "get_current_branch_id",
    "get_current_tenant_id",
    "set_current_session_id",
    "reset_current_session_id",
    "SessionScope",
    # Manager
    "SessionManagerDeployment",
]
