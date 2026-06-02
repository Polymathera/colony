"""``/api/v1/colony-status/*`` — read-only routes powering the
ColonyStatusPanel frontend (P11).

Per design doc §15.3 the panel surfaces alerts + recent activity +
a deep-link to the GitHub Project board for the colony's design
monorepo. v1 ships THOSE three reads only — the "Next steps for you"
LLM-proposals + the "Active agent thinking" consciousness-stream
excerpts + the scheduled-mission tile (P7 was skipped) are
explicitly deferred.

All reads are server-side joins against ``interaction_log`` (the
write-through table P8b/P10/P11 InteractionLog extensions populate).
The dashboard's db_pool is the same one that mirrors the events, so
the routes don't need a Ray Serve round-trip.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import service as auth_service
from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection


logger = logging.getLogger(__name__)
router = APIRouter()


# Alert tile reads both bottleneck + inconsistency rows. event_kinds
# pinned here so a future addition (e.g. a P3+ coverage_gap protocol)
# requires touching this list explicitly — guards against the alert
# tile silently growing without UI review.
_ALERT_EVENT_KINDS: tuple[str, ...] = ("bottleneck", "inconsistency")


def _get_db_pool(colony: ColonyConnection):
    pool = colony._db_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return pool


@router.get("/colony-status/alerts")
async def colony_status_alerts(
    limit: int = Query(50, ge=1, le=200),
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Recent bottlenecks + inconsistencies for this colony.

    Joined into one tail (most-recent first) since the UI renders
    them in a single stripe; ``event_kind`` discriminates per row.
    Scoped by ``(tenant_id, colony_id)`` derived from the
    authenticated user + ``X-Colony-Id`` header (the syscontext the
    AuthMiddleware sets)."""

    from polymathera.colony.agents.patterns.capabilities.interaction_log.service import (
        fetch_recent_activity,
    )
    from polymathera.colony.distributed.ray_utils.serving.context import (
        get_colony_id, get_tenant_id,
    )

    db = _get_db_pool(colony)
    tenant_id = get_tenant_id() or user.get("tenant_id", "")
    colony_id = get_colony_id() or ""
    if not tenant_id or not colony_id:
        raise HTTPException(
            status_code=400,
            detail="tenant_id + colony_id required (set X-Colony-Id).",
        )

    # ``fetch_recent_activity`` doesn't filter by event_kind today;
    # over-fetch + post-filter. ``limit*3`` heuristic budget so we
    # don't paginate just because the user's colony is noisy with
    # non-alert events (every issue comment also lands here). If a
    # real bottleneck shows the limit is biting, add an event_kind
    # filter param to ``fetch_recent_activity`` as a follow-up.
    rows = await fetch_recent_activity(
        db, tenant_id=tenant_id, colony_id=colony_id,
        limit=max(limit * 3, limit),
    )
    alerts = [r for r in rows if r["event_kind"] in _ALERT_EVENT_KINDS]
    return {"alerts": alerts[:limit], "count": len(alerts[:limit])}


@router.get("/colony-status/recent-activity")
async def colony_status_recent_activity(
    limit: int = Query(50, ge=1, le=200),
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Recent ``interaction_log`` rows for this colony — every
    event_kind mixed. The UI groups by ``channel`` + ``event_kind``
    on the client side; this route is the raw tail."""

    from polymathera.colony.agents.patterns.capabilities.interaction_log.service import (
        fetch_recent_activity,
    )
    from polymathera.colony.distributed.ray_utils.serving.context import (
        get_colony_id, get_tenant_id,
    )

    db = _get_db_pool(colony)
    tenant_id = get_tenant_id() or user.get("tenant_id", "")
    colony_id = get_colony_id() or ""
    if not tenant_id or not colony_id:
        raise HTTPException(
            status_code=400,
            detail="tenant_id + colony_id required (set X-Colony-Id).",
        )

    rows = await fetch_recent_activity(
        db, tenant_id=tenant_id, colony_id=colony_id, limit=limit,
    )
    return {"events": rows, "count": len(rows)}


@router.get("/colony-status/project-link")
async def colony_status_project_link(
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, str | None]:
    """Resolve the colony's design-monorepo URL to a GitHub Project
    deep-link. The panel uses this for its "▶ Open Project board"
    header button (the design doc §15 explicit non-duplication call
    — GitHub Projects is the source of truth for roadmap/kanban).

    Returns ``{"project_url": None}`` when the colony has no design
    monorepo configured OR the monorepo URL is not on github.com
    (GitLab / internal forge — the panel hides the button)."""

    from polymathera.colony.distributed.ray_utils.serving.context import (
        get_colony_id,
    )
    from polymathera.colony.design_monorepo.process import (
        parse_owner_repo_from_url,
    )

    db = _get_db_pool(colony)
    tenant_id = user["tenant_id"]
    colony_id = get_colony_id() or ""
    if not colony_id:
        return {"project_url": None}

    row = await auth_service.get_design_monorepo(
        db, colony_id=colony_id, tenant_id=tenant_id,
    )
    origin_url = (row or {}).get("origin_url")
    owner_repo = parse_owner_repo_from_url(origin_url or "")
    if owner_repo is None:
        return {"project_url": None}
    # Repo-level Projects board URL. Modern GitHub serves this for
    # both Projects v2 + the legacy classic Projects. If the operator
    # uses an org-level Projects v2 board (separate URL pattern), the
    # panel still works — clicking "Open Project board" lands on the
    # repo's projects list with a one-click hop to the org board.
    return {"project_url": f"https://github.com/{owner_repo}/projects"}
