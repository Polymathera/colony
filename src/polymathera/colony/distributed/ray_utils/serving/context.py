"""Execution context for the serving framework.

Provides a unified ``ExecutionContext`` that carries identity (colony_id,
tenant_id, session_id, run_id), privilege (Ring), and audit (origin) through
the entire distributed call chain.
Context variables that carry colony_id and tenant_id through the
serving call chain. These are set at the system entry point and
automatically propagated via DeploymentRequest across Ray actor
boundaries (where Python contextvars are dropped).

The serving framework handles propagation automatically:
- ``DeploymentHandle`` captures the current ``ExecutionContext`` into
  ``DeploymentRequest`` (where Python contextvars are dropped by Ray).
- ``__handle_request__`` restores it on the replica and enforces ring-level
  access control before calling the endpoint method.
- ``asyncio.create_task`` inherits the snapshot — long-running agent loops
  retain the context of their creator.

Usage at entry points (CLI, API, background tasks)::

    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )

    # User-mode (tenant-scoped)
    with execution_context(
        ring=Ring.USER,
        colony_id="my-colony",
        tenant_id="my-tenant",
        session_id="sess-abc",
        origin="cli",
    ):
        result = await handle.start_analysis(...)

    # Kernel-mode (infrastructure)
    with execution_context(ring=Ring.KERNEL, origin="vcm_reconciler"):
        names = await handle.get_all_deployment_names()

Reading context in downstream code::

    from polymathera.colony.distributed.ray_utils.serving.context import (
        get_execution_context, require_execution_context,
        get_colony_id, get_tenant_id,
    )

    ctx = require_execution_context()  # raises if not set
    colony_id = get_colony_id()        # shorthand
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator


class Ring(IntEnum):
    """Privilege rings, modeled after CPU protection rings.

    Lower number = more privilege.  Room for intermediate levels
    (SUPERVISOR = 1, SERVICE = 2) in the future.
    """
    KERNEL = 0
    USER = 3


@dataclass(frozen=True)
class ExecutionContext:
    """Ambient execution context propagated through the serving framework.

    Frozen (immutable) — you don't mutate context, you create a new one.
    This prevents accidental modification mid-request.

    Attributes:
        ring: Privilege level. KERNEL for infrastructure, USER for tenant-scoped.
        colony_id: Colony instance identifier (required for Ring.USER).
        tenant_id: Tenant/organization identifier (required for Ring.USER).
        session_id: Optional user session identifier.
        run_id: Optional analysis run identifier.
        trace_id: Optional distributed tracing identifier.
        origin: What created this context (e.g. "cli", "api", "vcm_reconciler").
    """
    ring: Ring = Ring.USER

    # Tenant identity (required for Ring.USER, absent for Ring.KERNEL)
    colony_id: str | None = None
    tenant_id: str | None = None

    # Session identity (optional, even in Ring.USER)
    session_id: str | None = None
    run_id: str | None = None

    # Tracing
    trace_id: str | None = None

    # Audit
    origin: str | None = None

    def validate(self) -> None:
        """Validate that this context is internally consistent.

        Raises:
            RuntimeError: If Ring.USER but colony_id or tenant_id is missing.
        """
        if self.ring == Ring.USER:
            if self.colony_id is None or self.tenant_id is None:
                raise RuntimeError(
                    f"Ring.USER execution context requires colony_id and tenant_id. "
                    f"Got colony_id={self.colony_id!r}, tenant_id={self.tenant_id!r}"
                )


# ---------------------------------------------------------------------------
# Context variable
# ---------------------------------------------------------------------------

_current_context: ContextVar[ExecutionContext | None] = ContextVar(
    "serving_execution_context", default=None
)


# ---------------------------------------------------------------------------
# Public API — reading
# ---------------------------------------------------------------------------

def get_execution_context() -> ExecutionContext | None:
    """Get the current execution context, or None if not set."""
    return _current_context.get()


def require_execution_context() -> ExecutionContext:
    """Get the current execution context, raising if not set.

    Raises:
        RuntimeError: If no execution context is set.
    """
    ctx = _current_context.get()
    if ctx is None:
        raise RuntimeError(
            "No execution context set. "
            "Use 'with execution_context(ring=..., ...)' at the entry point."
        )
    return ctx


def get_colony_id() -> str | None:
    """Shorthand: get colony_id from the current execution context.
    colony_id is an identifier for grouping related context sources into one address space (e.g., virtual monorepo ID)

    Returns:
        Current colony_id or None if not in an execution context.
    """
    ctx = _current_context.get()
    return ctx.colony_id if ctx is not None else None


def get_tenant_id() -> str | None:
    """Shorthand: get tenant_id from the current execution context.
    tenant_id is used for multi-tenancy to isolate data and resources between different tenants.

    Returns:
        Current tenant_id or None if not in an execution context.
    """
    ctx = _current_context.get()
    return ctx.tenant_id if ctx is not None else None


def get_session_id() -> str | None:
    """Shorthand: get session_id from the current execution context."""
    ctx = _current_context.get()
    return ctx.session_id if ctx is not None else None


def get_run_id() -> str | None:
    """Shorthand: get run_id from the current execution context."""
    ctx = _current_context.get()
    return ctx.run_id if ctx is not None else None


def require_colony_id() -> str:
    """Get colony_id, raising if not set.

    Returns:
        Current colony_id.

    Raises:
        RuntimeError: If no execution context or colony_id is None.
    """
    ctx = require_execution_context()
    if ctx.colony_id is None:
        raise RuntimeError(
            "colony_id not set in execution context. "
            "Use Ring.USER with colony_id at the entry point."
        )
    return ctx.colony_id


def require_tenant_id() -> str:
    """Get tenant_id, raising if not set.

    Returns:
        Current tenant_id.

    Raises:
        RuntimeError: If no execution context or tenant_id is None.
    """
    ctx = require_execution_context()
    if ctx.tenant_id is None:
        raise RuntimeError(
            "tenant_id not set in execution context. "
            "Use Ring.USER with tenant_id at the entry point."
        )
    return ctx.tenant_id


# ---------------------------------------------------------------------------
# Public API — setting
# ---------------------------------------------------------------------------

@contextmanager
def execution_context(
    ring: Ring = Ring.USER,
    *,
    colony_id: str | None = None,
    tenant_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    trace_id: str | None = None,
    origin: str | None = None,
) -> Iterator[ExecutionContext]:
    """Set the execution context for the duration of the block.

    Used at three points in the serving call chain:

    1. **Entry point** (CLI/API/background task) — sets context for the caller process initially.
    2. **Proxy actor** (``handle_request``) — restores context from
       ``DeploymentRequest`` after the Ray serialization boundary drops
       contextvars.
    3. **Replica actor** (``__handle_request__``) — same restoration after
       the second Ray boundary.

    No nesting guard is needed because each Ray ``.remote()`` call starts
    a fresh asyncio task with its own ``contextvars.Context``, and concurrent
    requests on the same actor are naturally isolated.

    Args:
        ring: Privilege level.
        colony_id: Colony instance identifier.
        tenant_id: Tenant/organization identifier.
        session_id: User session identifier.
        run_id: Analysis run identifier.
        trace_id: Distributed tracing identifier.
        origin: What created this context.

    Yields:
        The ExecutionContext that was set.
    """
    ctx = ExecutionContext(
        ring=ring,
        colony_id=colony_id,
        tenant_id=tenant_id,
        session_id=session_id,
        run_id=run_id,
        trace_id=trace_id,
        origin=origin,
    )
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)


@contextmanager
def restore_execution_context(ctx: ExecutionContext) -> Iterator[ExecutionContext]:
    """Restore a previously-captured ExecutionContext.

    Used internally by ``__handle_request__`` and ``proxy.handle_request``
    to restore context from a ``DeploymentRequest``.

    Args:
        ctx: The ExecutionContext to restore.

    Yields:
        The restored context.
    """
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)
