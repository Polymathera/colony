"""Virtual Context Manager (VCM) endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends

from ..dependencies import get_colony, get_redis
from ..models.api_models import PageSummary, VCMStats
from ..services.colony_connection import ColonyConnection
from ..services.redis_service import RedisService

logger = logging.getLogger(__name__)
router = APIRouter()

_DEFAULT_APP_NAME = "polymathera"


@router.get("/vcm/stats", response_model=VCMStats)
async def get_vcm_stats(
    colony: ColonyConnection = Depends(get_colony),
):
    """Get VCM statistics: total pages, loaded pages, groups, pending faults."""
    if not colony.is_connected:
        return VCMStats()

    try:
        handle = colony.get_deployment_handle(_DEFAULT_APP_NAME, "vcm")
        stats = await handle.get_stats()

        return VCMStats(
            total_pages=stats.get("total_pages", 0),
            loaded_pages=stats.get("loaded_pages", 0),
            page_groups=stats.get("page_groups", 0),
            pending_faults=stats.get("pending_faults", 0),
        )

    except Exception as e:
        logger.warning(f"Failed to get VCM stats: {e}")
        return VCMStats()


@router.get("/vcm/pages", response_model=list[PageSummary])
async def list_pages(
    colony: ColonyConnection = Depends(get_colony),
    redis_svc: RedisService = Depends(get_redis),
):
    """List virtual pages from the VCM page table.

    Reads VirtualPageTableState from Redis shared state.
    """
    if not colony.is_connected:
        return []

    try:
        # The page table is stored as shared state in Redis
        raw = await redis_svc.get_json("polymathera:serving:vcm:page_table")
        if raw is None:
            keys = await redis_svc.scan_keys("*page_table*")
            if keys:
                raw = await redis_svc.get_json(keys[0])

        if raw is None:
            return []

        pages = []
        entries = raw.get("entries", {})
        for page_id, entry in entries.items():
            pages.append(PageSummary(
                page_id=page_id,
                source=entry.get("source", ""),
                tokens=entry.get("token_count", 0),
                loaded=entry.get("loaded", False),
            ))
        return pages

    except Exception as e:
        logger.warning(f"Failed to list VCM pages: {e}")
        return []


@router.get("/vcm/working-set")
async def get_working_set(
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get the current working set of loaded pages."""
    if not colony.is_connected:
        return {"pages": []}

    try:
        handle = colony.get_deployment_handle(_DEFAULT_APP_NAME, "vcm")
        loaded = await handle.get_all_loaded_pages()
        return {"pages": loaded}

    except Exception as e:
        logger.warning(f"Failed to get working set: {e}")
        return {"pages": [], "error": str(e)}


@router.get("/vcm/pages/{page_id}")
async def get_page_detail(
    page_id: str,
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed info for a specific virtual page."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:
        handle = colony.get_deployment_handle(_DEFAULT_APP_NAME, "vcm")
        page = await handle.get_virtual_page(page_id=page_id)
        if page is None:
            return {"error": "page not found", "page_id": page_id}
        if hasattr(page, "model_dump"):
            return page.model_dump()
        return {"page_id": page_id, "raw": str(page)}

    except Exception as e:
        return {"error": str(e), "page_id": page_id}


@router.get("/vcm/pages/{page_id}/locations")
async def get_page_locations(
    page_id: str,
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get physical locations where a page is loaded."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:
        handle = colony.get_deployment_handle(_DEFAULT_APP_NAME, "vcm")
        locations = await handle.get_page_locations(virtual_page_id=page_id)
        return {
            "page_id": page_id,
            "locations": [
                loc.model_dump() if hasattr(loc, "model_dump") else str(loc)
                for loc in locations
            ],
        }

    except Exception as e:
        return {"error": str(e), "page_id": page_id}
