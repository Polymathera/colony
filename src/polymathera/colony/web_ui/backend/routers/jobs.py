"""Job submission and management endpoints.

A "job" spawns coordinator agents within an existing session and monitors
them until completion. Content must already be mapped in VCM (POST /vcm/map)
and a session must already exist (created via the sidebar).

Jobs are submitted asynchronously and tracked via the session system.
"""

from __future__ import annotations

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

class MissionSpec(BaseModel):
    """Specification for a single mission within a job."""

    type: str = Field(description="Mission type: impact, compliance, intent, contracts, slicing, basic")
    coordinator_version: str = Field(default="v2", description="Coordinator version")
    max_agents: int = Field(default=10, description="Max concurrent worker agents")
    quality_threshold: float = Field(default=0.7, description="Min quality score (0-1)")
    max_iterations: int = Field(default=10, description="Max planning iterations")
    batching_policy: str = Field(default="hybrid", description="Batching: hybrid, clustering, continuous")
    extra_capabilities: list[str] = Field(default_factory=list, description="Extra capability names")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Mission-specific params")


class JobSubmitRequest(BaseModel):
    """Submit an mission job to the running Colony.

    Content must already be mapped in VCM before submitting a job.
    Use POST /vcm/map to map a codebase first.
    """

    session_id: str = Field(description="Session to run missions in (created via sidebar)")
    missions: list[MissionSpec] = Field(description="Analyses to run")
    timeout_seconds: int = Field(default=600, description="Max time for the job")
    budget_usd: float | None = Field(default=None, description="Max budget in USD")


class JobSubmitResponse(BaseModel):
    """Response from job submission."""

    job_id: str
    session_id: str
    status: str  # "submitted", "error"
    missions: list[str]
    message: str = ""


class JobStatusResponse(BaseModel):
    """Current status of a submitted job."""

    job_id: str
    session_id: str
    status: str  # "running", "completed", "failed", "cancelled"
    missions_completed: int = 0
    missions_total: int = 0
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
    """Submit an mission job. Returns immediately — monitoring via SSE/polling.

    Content must already be mapped in VCM.
    The job runs asynchronously in the background:
    1. Spawns coordinator agents for each mission in the given session
    2. Coordinators run autonomously until completion or timeout
    """
    if not colony.is_connected:
        return JobSubmitResponse(
            job_id="", session_id="", status="error",
            missions=[], message="Not connected to cluster",
        )

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    session_id = request.session_id
    mission_types = [a.type for a in request.missions]

    # Track job
    _active_jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "submitted",
        "missions": mission_types,
        "missions_completed": 0,
        "missions_total": len(request.missions),
        "request": request.model_dump(),
    }

    # Launch background task for agent spawning.
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
        missions=mission_types,
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
        missions_completed=job.get("missions_completed", 0),
        missions_total=job.get("missions_total", 0),
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
            missions_completed=j.get("missions_completed", 0),
            missions_total=j.get("missions_total", 0),
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

    Content must already be mapped in VCM. Steps:
    1. Spawn coordinator agents for each mission
    2. Monitor coordinators until completion or timeout
    """
    job = _active_jobs.get(job_id)
    if not job:
        return

    try:
        # Spawn coordinators
        job["status"] = "spawning"

        with colony.user_execution_context(
            tenant_id=tenant_id,
            colony_id=colony_id,
            session_id=session_id,
            origin="dashboard_job",
        ):
            from polymathera.colony.agents import AgentMetadata, AgentHandle

            # Use the merged mission registry — builtins + the
            # ``polymathera.mission_types`` entry-point group — so a
            # REST job submission can target a CPS-shipped mission
            # (e.g. an OPM-MEG coordinator) the same way the chat
            # path does. The hardcoded ``MISSION_REGISTRY`` dict
            # alone would miss every entry-point mission.
            #
            # NOTE on L4 scope: ``/api/jobs/submit`` is a REST endpoint
            # with no parent agent in scope, so it cannot reach L4
            # missions / classes that live only under a design
            # monorepo's ``.colony/``. Those route through the chat
            # path (SessionAgent + ``AgentPoolCapability.create_agent``)
            # which threads ``RepoStateProvider.discovered_extensions``
            # into ``resolve_class`` automatically. Headless L4
            # discovery for REST is a follow-up.
            from polymathera.colony.agents.mission_registry import (
                get_mission_registry,
            )
            from polymathera.colony.agents.class_resolver import resolve_class
            from polymathera.colony.cli.polymath import (
                EXTRA_CAPABILITIES_REGISTRY,
            )
            registry = get_mission_registry()

            run_id = f"run_{uuid.uuid4().hex[:8]}"

            # Pre-create the run for token tracking
            sm = await colony.get_session_manager()
            await sm.create_run(
                session_id=session_id,
                agent_id="coordinator",
                input_data={"job_id": job_id},
                run_id=run_id,
            )

            coordinator_handles: list[tuple[MissionSpec, AgentHandle]] = []

            for mission in request.missions:
                reg = registry.get(mission.type)
                if not reg:
                    logger.warning("Unknown mission type: %s", mission.type)
                    continue

                coord_key = f"coordinator_{mission.coordinator_version}"
                coord_class = reg.get(coord_key, reg.get("coordinator_v2", ""))

                # Stamp the runtime ``AgentSelfConcept`` from the
                # spec-side ``MissionSelfConcept`` via the shared
                # helper used by ``spawn_mission`` too — single
                # source of truth for the bridge so the two spawn
                # paths can't drift on what ``agent_id`` / ``name``
                # to plant on the coordinator.
                from polymathera.colony.agents.configs import (
                    build_coordinator_self_concept,
                )

                # AgentMetadata's syscontext default_factory captures
                # whatever ``with execution_context(...)`` is in scope
                # — the outer block at line 207 sets tenant/colony/
                # session/origin, but ``run_id`` is mission-scoped (it
                # was minted at line 239, after the context entered).
                # Inherit the outer context and stamp run_id explicitly
                # so the coordinator's tracing facility groups its
                # spans under the right run.
                import dataclasses
                from polymathera.colony.distributed.ray_utils.serving.context import (
                    require_execution_context,
                )
                metadata = AgentMetadata(
                    role=f"{reg['label']} coordinator",
                    syscontext=dataclasses.replace(
                        require_execution_context(), run_id=run_id,
                    ),
                    goals=[f"Run {reg['label']} mission"],
                    max_iterations=mission.max_iterations,
                    self_concept=build_coordinator_self_concept(
                        reg, mission_type=mission.type,
                    ),
                    parameters={
                        "max_agents": mission.max_agents,
                        "quality_threshold": mission.quality_threshold,
                        "max_iterations": mission.max_iterations,
                        "batching_policy": {"type": mission.batching_policy},
                        "mission_type": mission.type,
                        **mission.parameters,
                    },
                )

                # Resolve capabilities
                registry_coord_caps = [
                    cap for cap in reg.get("coordinator_capabilities", [])
                    if cap in EXTRA_CAPABILITIES_REGISTRY
                ]
                all_extra_caps = list(set(
                    ["ConsciousnessCapability"] + registry_coord_caps + mission.extra_capabilities
                ))
                capability_paths = [
                    EXTRA_CAPABILITIES_REGISTRY[cap]["path"]
                    for cap in all_extra_caps
                    if cap in EXTRA_CAPABILITIES_REGISTRY
                ]

                agent_cls = resolve_class(coord_class)
                cap_blueprints = [resolve_class(path).bind() for path in capability_paths]
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
                coordinator_handles.append((mission, handle))

            if not coordinator_handles:
                job["status"] = "failed"
                job["message"] = "No valid missions to run"
                return

        # Monitor coordinators
        job["status"] = "running"
        job["message"] = f"Running {len(coordinator_handles)} mission coordinator(s)"

        completed = 0
        for mission_spec, handle in coordinator_handles:
            try:
                with colony.user_execution_context(
                    tenant_id=tenant_id, colony_id=colony_id,
                    session_id=session_id, origin="dashboard_job",
                ):
                    async for event in handle.run_streamed(
                        input_data={
                            "mission_type": mission_spec.type,
                            **mission_spec.parameters,
                        },
                        timeout=float(request.timeout_seconds),
                        session_id=session_id,
                        run_id=run_id,
                        namespace=mission_spec.type,
                    ):
                        if event.event_type in ("completed", "error", "timeout"):
                            break

                completed += 1
                job["missions_completed"] = completed
            except Exception as e:
                logger.error("Coordinator %s failed: %s", handle.agent_id, e)
                completed += 1
                job["missions_completed"] = completed

        job["status"] = "completed"
        job["message"] = f"Completed {completed}/{len(coordinator_handles)} missions"

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e, exc_info=True)
        job["status"] = "failed"
        job["message"] = str(e)
