"""Tenant/colony isolation context for the serving framework.

Context variables that carry colony_id and tenant_id through the
serving call chain. These are set at the system entry point and
automatically propagated via DeploymentRequest across Ray actor
boundaries (where Python contextvars are dropped).

The serving framework handles propagation automatically:
- DeploymentHandle captures contextvars into DeploymentRequest
- __handle_request__ restores them on the replica before calling the endpoint
- No manual propagation needed for normal @serving.endpoint calls

Usage at entry point (CLI/API)::

    from polymathera.colony.distributed.ray_utils.serving.context import isolation_context

    with isolation_context(colony_id="my-colony", tenant_id="my-tenant"):
        result = await handle.some_method(...)

Usage in downstream code::

    from polymathera.colony.distributed.ray_utils.serving.context import (
        get_colony_id, get_tenant_id,
    )

    colony = get_colony_id()   # returns "my-colony"
    tenant = get_tenant_id()   # returns "my-tenant"
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


# Isolation context variables — set at entry point, propagated via DeploymentRequest
_current_colony_id: ContextVar[str | None] = ContextVar("serving_colony_id", default=None)
_current_tenant_id: ContextVar[str | None] = ContextVar("serving_tenant_id", default=None)


def get_colony_id() -> str | None:
    """Get the current colony_id from context.
    colony_id is an identifier for grouping related context sources into one address space (e.g., virtual monorepo ID)

    Returns:
        Current colony_id or None if not in an isolation context.
    """
    return _current_colony_id.get()


def get_tenant_id() -> str | None:
    """Get the current tenant_id from context.
    tenant_id is used for multi-tenancy to isolate data and resources between different tenants.

    Returns:
        Current tenant_id or None if not in an isolation context.
    """
    return _current_tenant_id.get()


def require_colony_id() -> str:
    """Get colony_id, raising if not set.

    Returns:
        Current colony_id.

    Raises:
        RuntimeError: If not in an isolation context.
    """
    cid = _current_colony_id.get()
    if cid is None:
        raise RuntimeError(
            "colony_id not set in serving context. "
            "Use 'with isolation_context(colony_id, tenant_id):' at the entry point."
        )
    return cid


def require_tenant_id() -> str:
    """Get tenant_id, raising if not set.

    Returns:
        Current tenant_id.

    Raises:
        RuntimeError: If not in an isolation context.
    """
    tid = _current_tenant_id.get()
    if tid is None:
        raise RuntimeError(
            "tenant_id not set in serving context. "
            "Use 'with isolation_context(colony_id, tenant_id):' at the entry point."
        )
    return tid


@contextmanager
def isolation_context(colony_id: str, tenant_id: str) -> Iterator[None]:
    """Set colony_id and tenant_id for the duration of the block.

    Used at three points in the serving call chain:

    1. **Entry point** (CLI/API layer) — sets context for the caller process.
    2. **Proxy actor** (``handle_request``) — restores context from
       ``DeploymentRequest`` after the Ray serialization boundary drops
       contextvars.
    3. **Replica actor** (``__handle_request__``) — same restoration after
       the second Ray boundary.

    No nesting guard is needed because each Ray ``.remote()`` call starts
    a fresh asyncio task with its own ``contextvars.Context``, and concurrent
    requests on the same actor each have isolated contexts.

    Args:
        colony_id: Colony identifier.
        tenant_id: Tenant identifier.
    """
    colony_token = _current_colony_id.set(colony_id)
    tenant_token = _current_tenant_id.set(tenant_id)
    try:
        yield
    finally:
        _current_colony_id.reset(colony_token)
        _current_tenant_id.reset(tenant_token)
