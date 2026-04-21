"""Job submission and management endpoints.

A "job" is a complete analysis workflow: create session, map repo, spawn
coordinators, monitor until completion. Jobs are submitted asynchronously
and tracked via the session system.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class AnalysisSpec(BaseModel):
    """Specification for a single analysis within a job."""

    type: str = Field(description="Analysis type: impact, compliance, intent, contracts, slicing, basic")
    coordinator_version: str = Field(default="v2", description="Coordinator version")
    max_agents: int = Field(default=10, description="Max concurrent worker agents")
    quality_threshold: float = Field(default=0.7, description="Min quality score (0-1)")
    max_iterations: int = Field(default=10, description="Max planning iterations")
    batching_policy: str = Field(default="hybrid", description="Batching: hybrid, clustering, continuous")
    extra_capabilities: list[str] = Field(default_factory=list, description="Extra capability names")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Analysis-specific params")


class PagingSpec(BaseModel):
    """Paging configuration for a job."""

    flush_threshold: int = Field(default=20)
    flush_token_budget: int = Field(default=4096)
    pinned: bool = Field(default=False)


class JobSubmitRequest(BaseModel):
    """Submit an analysis job to the running Colony."""

    origin_url: str = Field(description="Git repo URL (https:// or file://)")
    branch: str = Field(default="main")
    commit: str = Field(default="HEAD")
    repo_id: str | None = Field(default=None, description="Scope ID (auto-generated if None)")
    analyses: list[AnalysisSpec] = Field(description="Analyses to run")
    paging: PagingSpec = Field(default_factory=PagingSpec)
    timeout_seconds: int = Field(default=600, description="Max time for the job")
    budget_usd: float | None = Field(default=None, description="Max budget in USD")


class JobSubmitResponse(BaseModel):
    """Response from job submission."""

    job_id: str
    session_id: str
    status: str  # "submitted", "error"
    analyses: list[str]
    message: str = ""


class JobStatusResponse(BaseModel):
    """Current status of a submitted job."""

    job_id: str
    session_id: str
    status: str  # "running", "completed", "failed", "cancelled"
    analyses_completed: int = 0
    analyses_total: int = 0
    message: str = ""


# ---------------------------------------------------------------------------
# In-memory job tracking (v1 — no persistence across restarts)
# ---------------------------------------------------------------------------

_active_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/jobs/submit", response_model=JobSubmitResponse)
async def submit_job(
    request: JobSubmitRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> JobSubmitResponse:
    """Submit an analysis job. Returns immediately — monitoring via SSE/polling.

    The job runs asynchronously in the background:
    1. Creates a session
    2. Maps the repo to VCM (if not already mapped)
    3. Spawns coordinator agents for each analysis
    4. Coordinators run autonomously until completion or timeout
    """
    if not colony.is_connected:
        return JobSubmitResponse(
            job_id="", session_id="", status="error",
            analyses=[], message="Not connected to cluster",
        )

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    analysis_types = [a.type for a in request.analyses]

    # Create session — AuthMiddleware has already set Ring.USER context
    # with the user's tenant_id and colony_id from the JWT and header.
    try:
        sm = colony.get_session_manager()
        session = await sm.create_session()
        if isinstance(session, dict):
            session_id = session.get("session_id", "")
        else:
            session_id = session.session_id
    except Exception as e:
        logger.error("Failed to create session for job %s: %s", job_id, e)
        return JobSubmitResponse(
            job_id=job_id, session_id="", status="error",
            analyses=analysis_types, message=f"Session creation failed: {e}",
        )

    # Track job
    _active_jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "submitted",
        "analyses": analysis_types,
        "analyses_completed": 0,
        "analyses_total": len(request.analyses),
        "request": request.model_dump(),
    }

    # Launch background task for VCM mapping + agent spawning.
    # Background tasks don't inherit the request's execution context
    # (contextvars are lost), so capture tenant/colony now.
    from polymathera.colony.distributed.ray_utils.serving.context import get_colony_id
    tenant_id = user["tenant_id"]
    colony_id = get_colony_id() or ""

    background_tasks.add_task(
        _run_job, job_id, session_id, request, colony, tenant_id, colony_id,
    )

    return JobSubmitResponse(
        job_id=job_id,
        session_id=session_id,
        status="submitted",
        analyses=analysis_types,
        message=f"Job submitted. Session: {session_id}",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, _user: dict = Depends(require_auth)) -> JobStatusResponse:
    """Get current status of a submitted job."""
    job = _active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(
        job_id=job["job_id"],
        session_id=job["session_id"],
        status=job["status"],
        analyses_completed=job.get("analyses_completed", 0),
        analyses_total=job.get("analyses_total", 0),
        message=job.get("message", ""),
    )


@router.get("/jobs/", response_model=list[JobStatusResponse])
async def list_jobs(_user: dict = Depends(require_auth)) -> list[JobStatusResponse]:
    """List all tracked jobs."""
    return [
        JobStatusResponse(
            job_id=j["job_id"],
            session_id=j["session_id"],
            status=j["status"],
            analyses_completed=j.get("analyses_completed", 0),
            analyses_total=j.get("analyses_total", 0),
            message=j.get("message", ""),
        )
        for j in _active_jobs.values()
    ]


# ---------------------------------------------------------------------------
# Background job execution
# ---------------------------------------------------------------------------

async def _run_job(
    job_id: str,
    session_id: str,
    request: JobSubmitRequest,
    colony: ColonyConnection,
    tenant_id: str,
    colony_id: str,
) -> None:
    """Execute a job in the background.

    Runs in a separate asyncio task after the HTTP response is sent.
    Must set its own execution context since contextvars don't propagate
    to background tasks.

    Steps:
    1. Map repo to VCM
    2. Spawn coordinator agents for each analysis
    3. Update job status as coordinators complete
    """
    job = _active_jobs.get(job_id)
    if not job:
        return

    try:
        job["status"] = "mapping"

        # Step 1: Map repo to VCM — use the user's context
        with colony.user_execution_context(
            tenant_id=tenant_id,
            colony_id=colony_id,
            session_id=session_id,
            origin="dashboard_job",
        ):
            from polymathera.colony.vcm.models import MmapConfig
            from polymathera.colony.vcm.sources import BuilInContextPageSourceType
            from polymathera.colony.agents import ScopeUtils

            vcm = colony.get_vcm()
            scope_id = request.repo_id or ScopeUtils.get_colony_level_scope()

            mmap_config = MmapConfig(
                flush_threshold=request.paging.flush_threshold,
                flush_token_budget=request.paging.flush_token_budget,
                pinned=request.paging.pinned,
            )

            mmap_result = await vcm.mmap_application_scope(
                scope_id=scope_id,
                source_type=BuilInContextPageSourceType.FILE_GROUPER.value,
                config=mmap_config,
                origin_url=request.origin_url,
                branch=request.branch,
                commit=request.commit,
            )

            if isinstance(mmap_result, dict):
                mmap_status = mmap_result.get("status", "error")
            else:
                mmap_status = mmap_result.status

            if mmap_status not in ("mapped", "already_mapped"):
                job["status"] = "failed"
                job["message"] = f"VCM mapping failed: {mmap_status}"
                return

        # Step 2: Spawn coordinators
        job["status"] = "spawning"

        with colony.user_execution_context(
            tenant_id=tenant_id,
            colony_id=colony_id,
            session_id=session_id,
            origin="dashboard_job",
        ):
            from polymathera.colony.agents import AgentMetadata, AgentHandle
            from polymathera.colony.agents import AgentSelfConcept

            # Import the analysis registry from polymath.py
            from polymathera.colony.cli.polymath import ANALYSIS_REGISTRY, EXTRA_CAPABILITIES_REGISTRY, _resolve_class

            run_id = f"run_{uuid.uuid4().hex[:8]}"

            # Pre-create the run for token tracking
            sm = colony.get_session_manager()
            await sm.create_run(
                session_id=session_id,
                agent_id="coordinator",
                input_data={"job_id": job_id},
                run_id=run_id,
            )

            coordinator_handles: list[tuple[AnalysisSpec, AgentHandle]] = []

            for analysis in request.analyses:
                reg = ANALYSIS_REGISTRY.get(analysis.type)
                if not reg:
                    logger.warning("Unknown analysis type: %s", analysis.type)
                    continue

                coord_key = f"coordinator_{analysis.coordinator_version}"
                coord_class = reg.get(coord_key, reg.get("coordinator_v2", ""))

                self_concept_config = reg.get("self_concept", {})
                metadata = AgentMetadata(
                    role=f"{reg['label']} coordinator",
                    run_id=run_id,
                    session_id=session_id,
                    goals=[f"Run {reg['label']} on {request.repo_id or scope_id}"],
                    max_iterations=analysis.max_iterations,
                    self_concept=AgentSelfConcept(**self_concept_config) if self_concept_config else None,
                    parameters={
                        "repo_id": request.repo_id or scope_id,
                        "max_agents": analysis.max_agents,
                        "quality_threshold": analysis.quality_threshold,
                        "max_iterations": analysis.max_iterations,
                        "batching_policy": {"type": analysis.batching_policy},
                        "analysis_type": analysis.type,
                        **analysis.parameters,
                    },
                )

                # Resolve capabilities
                registry_coord_caps = [
                    cap for cap in reg.get("coordinator_capabilities", [])
                    if cap in EXTRA_CAPABILITIES_REGISTRY
                ]
                all_extra_caps = list(set(
                    ["ConsciousnessCapability"] + registry_coord_caps + analysis.extra_capabilities
                ))
                capability_paths = [
                    EXTRA_CAPABILITIES_REGISTRY[cap]["path"]
                    for cap in all_extra_caps
                    if cap in EXTRA_CAPABILITIES_REGISTRY
                ]

                agent_cls = _resolve_class(coord_class)
                cap_blueprints = [_resolve_class(path).bind() for path in capability_paths]
                bp = agent_cls.bind(
                    agent_type=coord_class,
                    metadata=metadata,
                    bound_pages=[],
                    capability_blueprints=cap_blueprints,
                )

                handle = await AgentHandle.from_blueprint(
                    agent_blueprint=bp,
                    app_name=colony.app_name,
                )
                coordinator_handles.append((analysis, handle))

            if not coordinator_handles:
                job["status"] = "failed"
                job["message"] = "No valid analyses to run"
                return

        # Step 3: Monitor coordinators (run in background)
        job["status"] = "running"
        job["message"] = f"Running {len(coordinator_handles)} analysis coordinator(s)"

        completed = 0
        for analysis_spec, handle in coordinator_handles:
            try:
                with colony.user_execution_context(
                    tenant_id=tenant_id, colony_id=colony_id,
                    session_id=session_id, origin="dashboard_job",
                ):
                    async for event in handle.run_streamed(
                        input_data={
                            "repo_id": request.repo_id or scope_id,
                            "analysis_type": analysis_spec.type,
                            **analysis_spec.parameters,
                        },
                        timeout=float(request.timeout_seconds),
                        session_id=session_id,
                        run_id=run_id,
                        namespace=analysis_spec.type,
                    ):
                        if event.event_type in ("completed", "error", "timeout"):
                            break

                completed += 1
                job["analyses_completed"] = completed
            except Exception as e:
                logger.error("Coordinator %s failed: %s", handle.agent_id, e)
                completed += 1
                job["analyses_completed"] = completed

        job["status"] = "completed"
        job["message"] = f"Completed {completed}/{len(coordinator_handles)} analyses"

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e, exc_info=True)
        job["status"] = "failed"
        job["message"] = str(e)
