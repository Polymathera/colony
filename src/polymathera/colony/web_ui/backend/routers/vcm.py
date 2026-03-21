"""Virtual Context Manager (VCM) endpoints.

All data flows through the VCM deployment handle via Ray RPC.
Stats use get_stats(), page listings use list_stored_pages(),
working set uses get_all_loaded_pages().
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from ..dependencies import get_colony
from ..models.api_models import PageSummary, VCMStats
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/vcm/stats", response_model=VCMStats)
async def get_vcm_stats(
    colony: ColonyConnection = Depends(get_colony),
):
    """VCM statistics from deployment handle get_stats() RPC."""
    if not colony.is_connected:
        return VCMStats()

    try:
        stats = await colony.get_vcm().get_stats()
        pt = stats.get("page_table", {})
        storage = stats.get("storage", {})
        return VCMStats( # TODO: Add more stats
            total_pages=storage.get("total_pages", 0) or pt.get("total_pages_loaded", 0),
            loaded_pages=pt.get("total_pages_loaded", 0),
            page_groups=pt.get("num_groups", 0),
            pending_faults=pt.get("pending_faults", 0),
        )
    except Exception as e:
        logger.warning("Failed to get VCM stats: %s", e)
        return VCMStats()


@router.get("/vcm/pages", response_model=list[PageSummary])
async def list_pages(
    colony: ColonyConnection = Depends(get_colony),
    limit: int = Query(default=20000, le=50000),
    offset: int = Query(default=0, ge=0),
):
    """List stored pages via VCM deployment handle list_stored_pages() RPC."""
    if not colony.is_connected:
        return []

    try:

        summaries = await colony.get_vcm().list_stored_pages(
            limit=limit,
            offset=offset,
        )
        return [
            PageSummary(
                page_id=s["page_id"],
                source=s.get("source", ""),
                tokens=s.get("size", 0),
                loaded=True,
                files=s.get("files", []),
            )
            for s in summaries
        ]
    except Exception as e:
        logger.warning("Failed to list pages: %s", e)
        return []


@router.get("/vcm/working-set")
async def get_working_set(
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get the current working set of pages loaded in KV cache."""
    if not colony.is_connected:
        return {"pages": []}

    try:

        loaded = await colony.get_vcm().get_all_loaded_pages()
        return {"pages": loaded}
    except Exception as e:
        logger.warning("Failed to get working set: %s", e)
        return {"pages": [], "error": str(e)}


@router.get("/vcm/loaded-pages")
async def list_loaded_pages(
    colony: ColonyConnection = Depends(get_colony),
) -> list[dict[str, Any]]:
    """Get all pages currently loaded in KV cache with access stats."""
    if not colony.is_connected:
        return []

    try:

        return await colony.get_vcm().list_loaded_page_entries()
    except Exception as e:
        logger.warning("Failed to list loaded pages: %s", e)
        return []


@router.get("/vcm/pages/{page_id}/{colony_id}/{tenant_id}")
async def get_page_detail(
    page_id: str,
    colony_id: str,
    tenant_id: str,
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed info for a specific virtual page."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:

        page = await colony.get_vcm().get_virtual_page(page_id=page_id, colony_id=colony_id, tenant_id=tenant_id)
        if page is None:
            return {"error": "page not found", "page_id": page_id}
        if hasattr(page, "model_dump"):
            return page.model_dump()
        return {"page_id": page_id, "raw": str(page)}
    except Exception as e:
        return {"error": str(e), "page_id": page_id}


@router.get("/vcm/pages/{page_id}/{colony_id}/{tenant_id}/locations")
async def get_page_locations(
    page_id: str,
    colony_id: str,
    tenant_id: str,
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get physical locations where a page is loaded."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:

        locations = await colony.get_vcm().get_page_locations(page_id=page_id, colony_id=colony_id, tenant_id=tenant_id)
        return {
            "page_id": page_id,
            "colony_id": colony_id,
            "tenant_id": tenant_id,
            "locations": [
                loc.model_dump() if hasattr(loc, "model_dump") else str(loc)
                for loc in locations
            ],
        }
    except Exception as e:
        return {"error": str(e), "page_id": page_id, "colony_id": colony_id, "tenant_id": tenant_id}
