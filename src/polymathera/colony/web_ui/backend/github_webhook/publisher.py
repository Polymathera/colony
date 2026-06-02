"""Fan-out a normalized webhook event to every colony's blackboard
inside the matching tenant.

The webhook receiver runs in the dashboard process (no agent
context). Per the design decision in P9 §10 docs, v1 publishes to
EVERY colony in the tenant — no per-repo filtering against
``poll_repos`` yet. The InteractionLog write-through is naturally
scoped per-colony (each colony's system session reads its own
blackboard), so the only practical cost of broadcast-fan-out is
extra Redis writes for repos a colony doesn't care about. A
follow-up can add the ``poll_repos`` filter when a real cost
emerges.

Pattern matches ``web_ui/backend/routers/human_approval.py``: open
an ``execution_context(Ring.USER, tenant_id, colony_id)`` per
colony, construct an ``EnhancedBlackboard`` at COLONY scope,
``initialize`` → ``write`` → ``stop``. Each iteration is
independent — a single colony's failed write doesn't block the
others.
"""

from __future__ import annotations

import logging
from typing import Any

from polymathera.colony.agents.blackboard import EnhancedBlackboard
from polymathera.colony.agents.scopes import BlackboardScope, get_scope_prefix
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)


logger = logging.getLogger(__name__)


async def publish_to_tenant_colonies(
    *,
    app_name: str,
    tenant_id: str,
    colony_ids: list[str],
    key: str,
    value: dict[str, Any],
    delivery_id: str,
) -> int:
    """Write ``(key, value)`` to every colony's blackboard.

    Returns the count of colonies that received the write.
    Per-colony failures are logged at WARNING + don't abort the
    fan-out (one bad colony shouldn't block events for the others).
    ``delivery_id`` is logged so on-call can grep the dashboard logs
    for "where did webhook X go".
    """

    succeeded = 0
    for colony_id in colony_ids:
        try:
            with execution_context(
                ring=Ring.USER,
                tenant_id=tenant_id,
                colony_id=colony_id,
                origin="github_webhook_receiver",
            ):
                bb = EnhancedBlackboard(
                    app_name=app_name,
                    scope_id=get_scope_prefix(BlackboardScope.COLONY),
                    backend_type=None,
                    enable_events=True,
                )
                await bb.initialize()
                try:
                    await bb.write(
                        key, value,
                        tags={"github_webhook", "inbound"},
                        metadata={"delivery_id": delivery_id},
                    )
                finally:
                    await bb.stop()
            succeeded += 1
        except Exception:  # noqa: BLE001
            logger.exception(
                "github_webhook publisher: failed to write key=%s "
                "to colony=%s (tenant=%s, delivery=%s)",
                key, colony_id, tenant_id, delivery_id,
            )
    return succeeded


__all__ = ("publish_to_tenant_colonies",)
