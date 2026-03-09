"""Session context manager for scoping agent interactions.

This module provides context management for sessions:
- SessionContextManager: Class-based context manager for sessions
- session_context: Function-based context manager (convenience wrapper)
- get_current_session: Get the current session from context
- get_current_session_id: Get the current session ID
- get_current_branch_id: Get the current VCM branch ID
- SessionScope: Helper class for session-scoped operations

Usage:
    session = await session_manager.enter_session(session_id)

    # Option 1: Use session.context() method (preferred)
    async with session.context():
        # All agent work here uses session's branch
        result = await agent_handle.run({"query": "analyze code"})

    # Option 2: Use session_context function
    async with session_context(session):
        # Same behavior
        ...
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, AsyncIterator, Iterator

if TYPE_CHECKING:
    from .models import Session


# Context variable for tracking current session (full Session object)
_current_session: ContextVar[Session | None] = ContextVar("current_session", default=None)

# Context variable for tracking just session_id (for distributed agents that receive session_id in requests)
# This allows agents on remote Ray nodes to propagate session_id without having the full Session object
_current_session_id: ContextVar[str | None] = ContextVar("current_session_id", default=None)


def set_current_session_id(session_id: str | None) -> Token[str | None]:
    """Set the current session ID explicitly.

    Use this when an agent receives a request with session_id from a remote caller.
    In distributed Ray systems, context variables don't cross node boundaries,
    so the receiving agent must explicitly set the session_id from the request.

    Args:
        session_id: Session ID to set (or None to clear)

    Returns:
        Token for resetting the context variable

    Example:
        ```python
        # In agent request handler:
        session_id = request.get("session_id")
        token = set_current_session_id(session_id)
        try:
            # Process request - all operations use get_current_session_id()
            await process_request(request)
        finally:
            _current_session_id.reset(token)
        ```
    """
    return _current_session_id.set(session_id)


def reset_current_session_id(token: Token[str | None]) -> None:
    """Reset the current session ID to its previous value.

    Args:
        token: Token from set_current_session_id()
    """
    _current_session_id.reset(token)


@contextmanager
def session_id_context(session_id: str | None) -> Iterator[str | None]:
    """Context manager for setting session_id in distributed agents.

    Use this when processing a request that includes session_id. All operations
    within the context will have access to the session_id via get_current_session_id().

    Args:
        session_id: Session ID from the incoming request

    Yields:
        The session_id

    Example:
        ```python
        # In agent request handler:
        with session_id_context(request.get("session_id")):
            # All memory operations will include session_id
            await memory.store(data)
        ```
    """
    token = set_current_session_id(session_id)
    try:
        yield session_id
    finally:
        reset_current_session_id(token)


def get_current_session() -> Session | None:
    """Get the current session from context.

    Returns:
        Current Session or None if not in a session context
    """
    return _current_session.get()


def get_current_session_id() -> str | None:
    """Get the current session ID from context.

    Checks both the explicit session_id (set by distributed agents) and
    the full Session object (set by Session.context()).

    This function should be used by all components that need session_id
    for traceability. It works both:
    - On the entry node where Session.context() is used
    - On remote Ray nodes where session_id is extracted from requests

    Returns:
        Current session_id or None if not in a session context
    """
    # First check explicit session_id (set by distributed agents)
    explicit_id = _current_session_id.get()
    if explicit_id:
        return explicit_id
    # Fall back to session object (set by Session.context())
    session = _current_session.get()
    return session.session_id if session else None


def get_current_branch_id() -> str | None:
    """Get the current VCM branch ID from context.

    Used by VCM operations to determine which branch to read/write.

    Returns:
        Current branch_id or None if not in a session context
    """
    session = _current_session.get()
    return session.branch_id if session else None


def get_current_tenant_id() -> str | None:
    """Get the current tenant ID from session context.

    Returns:
        Current tenant_id or None if not in a session context
    """
    session = _current_session.get()
    return session.tenant_id if session else None


class SessionContextManager:
    """Async context manager for establishing a session scope.

    This is the class-based implementation that `Session.context()` returns.
    Prefer using `session.context()` over `session_context(session)`.

    All operations within this context will:
    - Have access to the session via get_current_session()
    - Use the session's VCM branch for page operations
    - Tag events/actions with the session_id

    Example:
        ```python
        # Using Session.context() method (preferred)
        async with session.context():
            await agent_handle.run({"query": "analyze code"})

        # Or using this class directly
        async with SessionContextManager(session):
            await agent_handle.run({"query": "analyze code"})
        ```
    """

    def __init__(self, session: "Session"):
        """Initialize session context manager.

        Args:
            session: Session to make current during context
        """
        self._session = session
        self._token: Token["Session | None"] | None = None

    @property
    def session(self) -> "Session":
        """Get the session this context manages."""
        return self._session

    async def __aenter__(self) -> "Session":
        """Enter session context, making session current."""
        self._token = _current_session.set(self._session)
        return self._session

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit session context, restoring previous session."""
        if self._token is not None:
            _current_session.reset(self._token)
            self._token = None


@asynccontextmanager
async def session_context(session: "Session") -> AsyncIterator["Session"]:
    """Context manager that establishes a session scope.

    This is a convenience wrapper around SessionContextManager.
    Prefer using `session.context()` directly when you have a Session object.

    All agent interactions within this context will:
    - Have access to the session via get_current_session()
    - Use the session's VCM branch for page operations
    - Tag events/actions with the session_id

    Example:
        ```python
        async with session_context(session):
            # All agent work here uses session's branch
            await agent_handle.run({"query": "analyze code"})
            # Results are on session's branch
        ```

    Args:
        session: Session to make current

    Yields:
        The session
    """
    async with SessionContextManager(session) as s:
        yield s


class SessionScope:
    """Helper class for creating session-scoped operations.

    Used by capabilities and other components to check if they're
    running within a session context and to access session information.

    Example:
        ```python
        class MyCapability:
            async def do_work(self):
                if SessionScope.is_in_session():
                    session = SessionScope.require_session()
                    branch_id = session.branch_id
                    # Use branch for operations
                else:
                    # Use default/main branch
        ```
    """

    @staticmethod
    def is_in_session() -> bool:
        """Check if currently in a session context.

        Returns:
            True if in a session context, False otherwise
        """
        return _current_session.get() is not None

    @staticmethod
    def require_session() -> Session:
        """Get current session, raising if not in session context.

        Use this when your operation requires a session context.

        Returns:
            Current Session

        Raises:
            RuntimeError: If not in a session context
        """
        session = _current_session.get()
        if session is None:
            raise RuntimeError(
                "This operation requires a session context. "
                "Use 'async with session_context(session):' to establish one."
            )
        return session

    @staticmethod
    def get_session_or_none() -> Session | None:
        """Get current session or None (same as get_current_session).

        Convenience method for optional session handling.

        Returns:
            Current Session or None
        """
        return _current_session.get()

    @staticmethod
    def get_branch_id_or_default(default: str = "main") -> str:
        """Get current branch ID or a default value.

        Useful for code that should work both in and out of session context.

        Args:
            default: Default branch ID to use if not in session

        Returns:
            Branch ID from session or the default
        """
        session = _current_session.get()
        return session.branch_id if session else default

    @staticmethod
    def get_tenant_id_or_default(default: str = "default") -> str:
        """Get current tenant ID or a default value.

        Useful for code that should work both in and out of session context.

        Args:
            default: Default tenant ID to use if not in session

        Returns:
            Tenant ID from session or the default
        """
        session = _current_session.get()
        return session.tenant_id if session else default
