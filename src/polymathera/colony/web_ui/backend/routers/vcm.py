"""Virtual Context Manager (VCM) endpoints.

All data flows through the VCM deployment handle via Ray RPC.
Stats use get_stats(), page listings use list_stored_pages(),
working set uses get_all_loaded_pages().
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Query, UploadFile, File
from pydantic import BaseModel, Field

from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..models.api_models import PageSummary, VCMStats
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request/response models for repo mapping
# ---------------------------------------------------------------------------

class MapRepoRequest(BaseModel):
    """Map a codebase to VCM for exploration and analysis."""

    origin_url: str = Field(description="Git repo URL (https:// or file://)")
    branch: str = Field(default="main", description="Git branch")
    commit: str = Field(default="HEAD", description="Git commit SHA")
    repo_id: str | None = Field(default=None, description="Scope ID (auto-generated if None)")
    flush_threshold: int = Field(default=20, description="Page flush threshold")
    flush_token_budget: int = Field(default=4096, description="Token budget per page flush")
    pinned: bool = Field(default=False, description="Pin pages in cache")


class MapRepoResponse(BaseModel):
    """Result of mapping a codebase to VCM."""

    status: str = Field(description="mapped, already_mapped, or error")
    scope_id: str = Field(description="VCM scope identifier")
    message: str = Field(default="", description="Human-readable status message")


@router.get("/vcm/stats", response_model=VCMStats)
async def get_vcm_stats(
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
):
    """VCM statistics from deployment handle get_stats() RPC."""
    if not colony.is_connected:
        return VCMStats()

    # AuthMiddleware sets USER context; no explicit context needed
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
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
    limit: int = Query(default=20000, le=50000),
    offset: int = Query(default=0, ge=0),
):
    """List stored pages via VCM deployment handle list_stored_pages() RPC."""
    if not colony.is_connected:
        return []

    # AuthMiddleware sets USER context; no explicit context needed
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
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get the current working set of pages loaded in KV cache."""
    if not colony.is_connected:
        return {"pages": []}

    # AuthMiddleware sets USER context; no explicit context needed
    try:

        loaded = await colony.get_vcm().get_all_loaded_pages()
        return {"pages": loaded}
    except Exception as e:
        logger.warning("Failed to get working set: %s", e)
        return {"pages": [], "error": str(e)}


@router.get("/vcm/loaded-pages")
async def list_loaded_pages(
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> list[dict[str, Any]]:
    """Get all pages currently loaded in KV cache with access stats."""
    if not colony.is_connected:
        return []

    # AuthMiddleware sets USER context; no explicit context needed
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
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed info for a specific virtual page."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:
        page = await colony.get_vcm().get_virtual_page(page_id)
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
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get physical locations where a page is loaded."""
    if not colony.is_connected:
        return {"error": "not connected"}

    # AuthMiddleware sets USER context; no explicit context needed
    try:
        locations = await colony.get_vcm().get_page_locations(page_id)
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


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Repo mapping — async mapping of a codebase to VCM with progress tracking without creating a session
# ---------------------------------------------------------------------------

import uuid
import time
from typing import ClassVar

# In-memory tracker for mapping operations (v1 — not persisted across restarts)
_mapping_ops: dict[str, dict] = {}


class MappingOpStatus(BaseModel):
    """Status of an ongoing or completed mapping operation."""
    op_id: str
    status: str  # "pending", "running", "mapped", "already_mapped", "error"
    origin_url: str
    started_at: float
    completed_at: float | None = None
    message: str = ""
    scope_id: str = ""


@router.post("/vcm/map", response_model=MappingOpStatus)
async def map_repo(
    request: MapRepoRequest,
    background_tasks: BackgroundTasks,
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> MappingOpStatus:
    """Start mapping a repository to VCM. Returns immediately.

    The mapping runs in the background. Poll GET /vcm/map/operations
    to track progress. The VCM tab auto-refreshes to show new pages
    as they are created.

    Pages the codebase into VCM pages without creating a session or running
    agents. After mapping, the VCM and Page Graph tabs show the codebase
    structure. Users can explore before configuring analysis jobs.

    Idempotent — mapping the same repo again returns "already_mapped".
    """
    if not colony.is_connected:
        return MappingOpStatus(
            op_id="", status="error", origin_url=request.origin_url,
            started_at=time.time(), message="Not connected to cluster",
        )

    op_id = f"map_{uuid.uuid4().hex[:12]}"
    op = {
        "op_id": op_id,
        "status": "pending",
        "origin_url": request.origin_url,
        "started_at": time.time(),
        "completed_at": None,
        "message": f"Mapping {request.origin_url}...",
        "scope_id": "",
    }
    _mapping_ops[op_id] = op

    # Capture user context for the background task
    from polymathera.colony.distributed.ray_utils.serving.context import get_tenant_id, get_colony_id
    tenant_id = get_tenant_id() or ""
    colony_id = get_colony_id() or ""

    background_tasks.add_task(
        _run_mapping, op_id, request, colony, tenant_id, colony_id,
    )

    return MappingOpStatus(**op)


@router.get("/vcm/map/operations", response_model=list[MappingOpStatus])
async def list_mapping_operations(
    _user: dict = Depends(require_auth),
) -> list[MappingOpStatus]:
    """List all mapping operations (active and recent)."""
    return [MappingOpStatus(**op) for op in _mapping_ops.values()]


async def _run_mapping(
    op_id: str,
    request: MapRepoRequest,
    colony: ColonyConnection,
    tenant_id: str,
    colony_id: str,
) -> None:
    """Execute VCM mapping in the background."""
    op = _mapping_ops.get(op_id)
    if not op:
        return

    op["status"] = "running"
    op["message"] = f"Cloning and paging {request.origin_url}..."

    with colony.user_execution_context(
        tenant_id=tenant_id,
        colony_id=colony_id,
        origin="dashboard_map",
    ):
        # AuthMiddleware sets USER context; no explicit context needed
        try:
            from polymathera.colony.vcm.models import MmapConfig
            from polymathera.colony.vcm.sources import BuilInContextPageSourceType
            from polymathera.colony.agents import ScopeUtils

            vcm = colony.get_vcm()

            mmap_config = MmapConfig(
                flush_threshold=request.flush_threshold,
                flush_token_budget=request.flush_token_budget,
                pinned=request.pinned,
            )

            scope_id = request.repo_id or ScopeUtils.get_colony_level_scope()

            result = await vcm.mmap_application_scope(
                scope_id=scope_id,
                source_type=BuilInContextPageSourceType.FILE_GROUPER.value,
                config=mmap_config,
                origin_url=request.origin_url,
                branch=request.branch,
                commit=request.commit,
            )

            if isinstance(result, dict):
                op["status"] = result.get("status", "error")
                op["message"] = result.get("message", "")
                op["scope_id"] = result.get("scope_id", scope_id)
            else:
                op["status"] = result.status
                op["message"] = result.message
                op["scope_id"] = result.scope_id

        except Exception as e:
            logger.error("Mapping operation %s failed: %s", op_id, e)
            op["status"] = "error"
            op["message"] = str(e)

        op["completed_at"] = time.time()


@router.post("/vcm/upload-and-map", response_model=MapRepoResponse)
async def upload_and_map(
    file: UploadFile = File(...),
    flush_threshold: int = 20,
    flush_token_budget: int = 4096,
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> MapRepoResponse:
    """Upload a zip/tar.gz archive and map it to VCM.

    The archive is extracted to the cluster's shared volume and then
    mapped via mmap_application_scope with a file:// URL. This allows
    users to map local codebases without shell access to the cluster.

    Accepts: .zip, .tar.gz, .tgz
    """
    import hashlib
    import os
    import tempfile
    import zipfile
    import tarfile

    if not colony.is_connected:
        return MapRepoResponse(status="error", scope_id="", message="Not connected to cluster")

    filename = file.filename or "upload"
    if not any(filename.endswith(ext) for ext in (".zip", ".tar.gz", ".tgz")):
        return MapRepoResponse(status="error", scope_id="", message="Unsupported format. Use .zip, .tar.gz, or .tgz")

    try:
        # Read upload into temp file
        content = await file.read()
        content_hash = hashlib.sha256(content).hexdigest()[:12]
        upload_dir = f"/mnt/shared/uploads/{content_hash}"

        # Extract
        os.makedirs(upload_dir, exist_ok=True)

        if filename.endswith(".zip"):
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(upload_dir)
            os.unlink(tmp_path)
        else:
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            with tarfile.open(tmp_path, "r:gz") as tf:
                tf.extractall(upload_dir)
            os.unlink(tmp_path)

        # Map via file:// URL
        from polymathera.colony.vcm.models import MmapConfig
        from polymathera.colony.vcm.sources import BuilInContextPageSourceType
        from polymathera.colony.agents import ScopeUtils

        vcm = colony.get_vcm()
        scope_id = ScopeUtils.get_colony_level_scope()

        mmap_config = MmapConfig(
            flush_threshold=flush_threshold,
            flush_token_budget=flush_token_budget,
        )

        result = await vcm.mmap_application_scope(
            scope_id=scope_id,
            source_type=BuilInContextPageSourceType.FILE_GROUPER.value,
            config=mmap_config,
            origin_url=f"file://{upload_dir}",
            branch="main",
            commit="HEAD",
        )

        # result is MmapResult or dict from DeploymentHandle
        if isinstance(result, dict):
            status = result.get("status", "error")
            message = result.get("message", "")
            result_scope_id = result.get("scope_id", scope_id)
        else:
            status = result.status
            message = result.message
            result_scope_id = result.scope_id

        return MapRepoResponse(
            status=status,
            scope_id=result_scope_id,
            message=message or f"Uploaded and mapped from {filename}",
        )

    except Exception as e:
        logger.error("Failed to upload and map: %s", e)
        return MapRepoResponse(status="error", scope_id="", message=str(e))
