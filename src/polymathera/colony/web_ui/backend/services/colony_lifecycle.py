"""Single entry point for colony creation.

**Discipline**: every code path that lands a row in the ``colonies``
table MUST go through :func:`provision_colony`. Going around it via a
direct ``auth_service.create_colony(...)`` (or a raw ``INSERT``)
skips the per-colony ``SessionAgent`` bootstrap that hosts the
colony-singleton capabilities (P8 ``GitHubInboundCapability``,
``InteractionLogCapability``; P9 webhook receiver fan-out; P10
``MentionRoutingCapability``). The result is a colony whose
capabilities never run — visible to operators as: nothing shows up
in the Traces / Agents / Overview tabs except per-user sessions.

This was the regression that triggered this centralization: the
signup handler was creating a "Default" colony inside
``auth_service.create_user``'s transaction without bootstrapping a
system session, and the Traces tab silently lost half its rows.

If you find yourself adding ``INSERT INTO colonies`` somewhere, OR
adding a second helper that creates colonies, STOP and call
:func:`provision_colony` instead. The whole point of this module is
that it is the ONLY way to provision a colony.
"""

from __future__ import annotations

import asyncio
import logging

from ..auth import service as auth_service
from ..chat.system_session import ensure_system_session_for_colony
from .colony_connection import ColonyConnection


logger = logging.getLogger(__name__)

# Bounded wait for the session_manager deployment's ``@on_app_ready``
# hook to finish (populates vcm_handle). Without this, every endpoint
# raises ``RuntimeError("VCM handle not yet available")`` until the
# framework finishes wiring siblings — typically a few seconds after
# Ray cluster reachability.
_SESSION_MANAGER_READY_TIMEOUT_S = 60.0
_SESSION_MANAGER_READY_POLL_S = 1.0


async def _wait_for_session_manager_ready(colony: ColonyConnection) -> bool:
    """Poll until the session_manager deployment is registered AND
    its ``@on_app_ready`` hook has finished. Returns True on ready,
    False on timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _SESSION_MANAGER_READY_TIMEOUT_S
    while True:
        try:
            sm = await colony.get_session_manager()
            if await sm.is_ready():
                return True
        except Exception:  # noqa: BLE001 — pre-registration / pre-init
            pass
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(_SESSION_MANAGER_READY_POLL_S)


async def provision_colony(
    colony: ColonyConnection,
    *,
    tenant_id: str,
    name: str,
    description: str = "",
    is_default: bool = False,
) -> dict[str, str]:
    """Land a colony row + bootstrap its always-on system SessionAgent.

    Composes two steps that callers MUST run together:

    1. ``auth_service.create_colony`` — atomic SQL insert into the
       ``colonies`` table.
    2. ``ensure_system_session_for_colony`` — best-effort Ray-side
       bootstrap of the system ``SessionAgent`` that hosts
       colony-singleton capabilities. Failures are logged but do NOT
       fail the call: the user-facing colony row is already
       persisted, the lifespan walker on the next dashboard restart
       retries the bootstrap, and the colony is otherwise functional
       in the meantime (chat sessions still work).

    Args:
        colony: The dashboard's ``ColonyConnection`` (needed for
            ``execution_context`` + session-manager handle by step 2).
        tenant_id: The tenant the new colony belongs to.
        name: Human-facing colony name (e.g. ``"Default"`` for the
            auto-created one, or whatever the user typed in the UI).
        description: Free-text description (optional).
        is_default: Whether to mark this as the tenant's default
            colony (``colonies.is_default``). Exactly one colony per
            tenant should have this; signup auto-creates one as
            default, ``POST /colonies/`` from the UI never does.

    Returns:
        ``{"colony_id": ..., "name": ..., "tenant_id": ...}`` —
        the same shape ``auth_service.create_colony`` returns, so
        callers' downstream consumers don't have to know there's an
        intervening bootstrap step.
    """

    db_pool = colony._db_pool
    if db_pool is None:
        raise RuntimeError(
            "provision_colony: no db_pool on ColonyConnection — "
            "cannot insert colony row.",
        )

    result = await auth_service.create_colony(
        db_pool,
        tenant_id=tenant_id,
        name=name,
        description=description,
        is_default=is_default,
    )

    # Wait for the session_manager's ``@on_app_ready`` to finish
    # (populates vcm_handle) before invoking endpoints that depend on
    # it. The lifespan walker is the safety net for any colony that
    # times out here — re-runs bootstrap on next restart.
    if not await _wait_for_session_manager_ready(colony):
        logger.error(
            "provision_colony: session_manager not ready within %.0fs; "
            "skipping bootstrap for colony %s — lifespan walker will "
            "retry next restart.",
            _SESSION_MANAGER_READY_TIMEOUT_S, result["colony_id"],
        )
        return result

    try:
        await ensure_system_session_for_colony(
            colony,
            tenant_id=result["tenant_id"],
            colony_id=result["colony_id"],
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "provision_colony: system-session bootstrap failed for "
            "colony %s (tenant=%s); colony-singleton capabilities "
            "will not run until the next dashboard restart.",
            result["colony_id"], result["tenant_id"],
        )

    return result


__all__ = ("provision_colony",)
