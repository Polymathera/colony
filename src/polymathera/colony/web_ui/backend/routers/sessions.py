"""Session and run management endpoints.

All endpoints require authentication. The AuthMiddleware sets the
ExecutionContext (Ring.USER with tenant_id from JWT, colony_id from
X-Colony-Id header) before any endpoint code runs. Deployment handle
calls automatically propagate this context across Ray boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ..auth import service as auth_service
from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..models.api_models import RunSummary, SessionSummary
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


def _bundled_samples_plugins_root() -> str:
    """Filesystem path to the sample plugins shipped with the package.

    Resolved against the installed ``polymathera.colony.samples`` module
    so it works whether the package is installed editable, from a wheel,
    or vendored. Used by ``UserPluginCapability`` to expose the
    ``colony-samples`` plugin to every session agent without requiring
    the operator to copy files into ``~/.colony/plugins``.
    """
    from polymathera.colony import samples
    return str(Path(samples.__file__).parent / "plugins")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    """Request to create a new session."""

    name: str | None = Field(default=None, description="Human-readable session name")
    ttl_seconds: float | None = Field(default=None, description="Session TTL (None = use default)")
    fork_from_session_id: str | None = Field(default=None, description="Fork from existing session")


class CreateSessionResponse(BaseModel):
    """Response from session creation."""

    session_id: str
    status: str  # "created", "error"
    message: str = ""


class SessionActionResponse(BaseModel):
    """Response from session state change."""

    session_id: str
    success: bool
    message: str = ""


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute from object or dict — handles both Pydantic models and dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _resolve_github_identity(
    tenant_row: dict | None,
    user_row: dict | None,
) -> dict:
    """Compose the ``github_identity`` metadata block agents read.

    Inputs are the raw rows returned by
    :func:`auth_service.get_tenant_github_installation` and
    :func:`auth_service.get_user_github_identity` — either may be
    ``None`` (tenant missing the row; user hasn't OAuth'd). The
    returned dict always has the five keys downstream readers
    expect; absent values are ``None``.
    """

    return {
        "tenant_installation_id": (
            (tenant_row or {}).get("installation_id")
        ),
        "user_github_login": (
            (user_row or {}).get("github_login")
        ),
        "user_github_id": (
            (user_row or {}).get("github_user_id")
        ),
        "git_user_email": (
            (user_row or {}).get("github_email")
        ),
        "git_user_name": (
            (user_row or {}).get("git_user_name")
        ),
    }


# ---------------------------------------------------------------------------
# Read endpoints
# ---------------------------------------------------------------------------

@router.get("/sessions/", response_model=list[SessionSummary])
async def list_sessions(
    limit: int = Query(100, le=500),
    include_system: bool = Query(
        False,
        description=(
            "Include colony-singleton system sessions (``session_kind="
            "'system'``) in the response. Default ``false`` hides them "
            "from the chat-UI sessions list; the Traces tab passes "
            "``true`` to surface them for observability."
        ),
    ),
    user_role: list[str] | None = Query(
        default=None,
        description=(
            "P12 forward-compat RBAC stub. When set, keep only "
            "sessions whose ``metadata.user_role`` overlaps with any "
            "of the supplied roles. NOT an authorisation gate — pure "
            "client-side filter; sessions without any ``user_role`` "
            "declared pass the filter regardless (legacy + system "
            "sessions don't fail-open into an enforcement story)."
        ),
    ),
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
):
    """List sessions for the authenticated user's tenant and active colony."""
    if not colony.is_connected:
        return []

    from polymathera.colony.distributed.ray_utils.serving.context import get_tenant_id, get_colony_id

    try:
        handle = await colony.get_session_manager()
        sessions = await handle.list_sessions(
            tenant_id=get_tenant_id(),
            colony_id=get_colony_id(),
            include_expired=False,
            limit=limit,
        )

        if not include_system:
            # Default chat-UI view hides system sessions. Filter on
            # the field directly; pre-P8-0 serialized rows default to
            # ``session_kind="user"`` and pass through unaffected.
            sessions = [
                s for s in sessions
                if _get(s, "session_kind", "user") != "system"
            ]

        if user_role:
            # P12: any-overlap filter on metadata.user_role. Sessions
            # whose metadata.user_role is None or empty pass through
            # unchanged — RBAC is a future PR, this is plumbing only.
            requested = set(user_role)
            sessions = [
                s for s in sessions
                if not _session_has_user_roles(s)
                or requested & set(_session_user_roles(s))
            ]

        return [
            SessionSummary(
                session_id=_get(s, "session_id", ""),
                tenant_id=_get(s, "tenant_id", ""),
                colony_id=_get(s, "colony_id", ""),
                state=str(_get(s, "state", "")),
                created_at=_get(s, "created_at", 0.0),
                run_count=len(_get(s, "runs", []) or []),
                session_kind=str(_get(s, "session_kind", "user")),
            )
            for s in sessions
        ]

    except Exception as e:
        logger.warning(f"Failed to list sessions: {e}")
        return []


def _session_user_roles(s: Any) -> list[str]:
    """Resolve ``metadata.user_role`` across dict-shaped and Pydantic-
    shaped session objects. Returns ``[]`` when absent or empty so
    callers can use set semantics directly."""
    metadata = _get(s, "metadata", None)
    roles = _get(metadata, "user_role", None) if metadata is not None else None
    return list(roles) if roles else []


def _session_has_user_roles(s: Any) -> bool:
    return bool(_session_user_roles(s))


@router.get("/sessions/{session_id}")
async def get_session_detail(
    session_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed session information."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:
        handle = await colony.get_session_manager()
        session = await handle.get_session(session_id=session_id)
        if session is None:
            return {"error": "session not found", "session_id": session_id}

        # Verify the session belongs to this user's tenant
        from polymathera.colony.distributed.ray_utils.serving.context import get_tenant_id
        session_tenant = _get(session, "tenant_id", "")
        if session_tenant and session_tenant != get_tenant_id():
            return {"error": "session not found", "session_id": session_id}

        if isinstance(session, dict):
            return session
        if hasattr(session, "model_dump"):
            return session.model_dump()
        return {"session_id": session_id, "raw": str(session)}

    except Exception as e:
        return {"error": str(e), "session_id": session_id}


@router.get("/sessions/{session_id}/runs", response_model=list[RunSummary])
async def get_session_runs(
    session_id: str,
    limit: int = Query(100, le=500),
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
):
    """List runs for a specific session."""
    if not colony.is_connected:
        return []

    try:
        handle = await colony.get_session_manager()
        runs = await handle.get_session_runs(session_id=session_id, limit=limit)

        result = []
        for r in runs:
            ru = _get(r, "resource_usage", None)
            result.append(RunSummary(
                run_id=_get(r, "run_id", ""),
                session_id=_get(r, "session_id", session_id),
                agent_id=_get(r, "agent_id", ""),
                status=str(_get(r, "status", "")),
                started_at=_get(r, "started_at", None),
                completed_at=_get(r, "completed_at", None),
                input_tokens=_get(ru, "input_tokens", 0) if ru else 0,
                output_tokens=_get(ru, "output_tokens", 0) if ru else 0,
            ))
        return result

    except Exception as e:
        logger.warning(f"Failed to list runs for session {session_id}: {e}")
        return []


@router.get("/sessions/runs/{run_id}")
async def get_run_detail(
    run_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get detailed run information including events and resource usage."""
    if not colony.is_connected:
        return {"error": "not connected"}

    try:
        handle = await colony.get_session_manager()
        run = await handle.get_run(run_id=run_id)
        if run is None:
            return {"error": "run not found", "run_id": run_id}
        if isinstance(run, dict):
            return run
        if hasattr(run, "model_dump"):
            return run.model_dump()
        return {"run_id": run_id, "raw": str(run)}

    except Exception as e:
        return {"error": str(e), "run_id": run_id}


@router.get("/sessions/stats/overview")
async def get_session_stats(
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Get session manager statistics."""
    if not colony.is_connected:
        return {"status": "disconnected"}

    try:
        handle = await colony.get_session_manager()
        return await handle.get_stats()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Write endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions/", response_model=CreateSessionResponse)
async def create_session(
    request: CreateSessionRequest | None = None,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> CreateSessionResponse:
    """Create a new session.

    The session is created under the authenticated user's tenant.
    The colony_id comes from the X-Colony-Id header (set by the frontend).
    """
    if not colony.is_connected:
        return CreateSessionResponse(session_id="", status="error", message="Not connected")

    request = request or CreateSessionRequest()

    # Validate the X-Colony-Id the frontend asked us to scope under
    # actually exists in postgres for this user's tenant. Without
    # this gate, a stale browser localStorage entry (e.g. left over
    # from before ``colony-env down --volumes`` wiped the DB) would
    # silently create a "ghost" session in a colony id that no
    # longer exists, leaving the session agent unable to bootstrap
    # (NoSuchPathError on the clone dir, chat messages stranded in
    # the wrong scope's blackboard, etc.).
    from polymathera.colony.distributed.ray_utils.serving.context import (
        get_colony_id, get_tenant_id,
    )
    requested_colony_id = get_colony_id() or ""
    requested_tenant_id = get_tenant_id() or ""
    if not requested_colony_id or not requested_tenant_id:
        return CreateSessionResponse(
            session_id="", status="error",
            message=(
                "X-Colony-Id header + tenant context required. "
                "Re-sign-in if you just redeployed."
            ),
        )
    if colony._db_pool is None:
        # Without the database we can't validate the colony, look up
        # the design-monorepo URL, or check the GitHub Project gate.
        # Refusing is safer than spawning a SessionAgent that would
        # crash on first DB-backed action with a confusing error.
        return CreateSessionResponse(
            session_id="", status="error",
            message=(
                "Auth database is not available — sessions cannot be "
                "created. Check ``RDS_*`` env on the dashboard service."
            ),
        )

    from ..auth import service as auth_service
    owned = await auth_service.list_colonies(
        colony._db_pool, requested_tenant_id,
    )
    if not any(c["colony_id"] == requested_colony_id for c in owned):
        logger.warning(
            "create_session: rejected X-Colony-Id=%s for "
            "tenant=%s (no such colony — stale browser cache?)",
            requested_colony_id, requested_tenant_id,
        )
        return CreateSessionResponse(
            session_id="", status="error",
            message=(
                f"Colony {requested_colony_id!r} is not visible "
                f"to you. Pick a real colony from the dropdown "
                f"(your browser may be remembering a colony from "
                f"a previous deployment — refresh the page)."
            ),
        )

    # Refuse session-create when the colony has no GitHub Project
    # attached. Every issue Colony creates against the colony's
    # monorepo is auto-attached to this project (it's threaded
    # into ``GitHubCapability.bind`` as ``default_project_id``
    # below); without one, issues land free-floating, which the
    # operator explicitly does not want. The colony-create flow
    # is supposed to gate this — the check here is defense in
    # depth so a colony provisioned through an older code path
    # or via the API directly can't slip through.
    colony_project = await auth_service.get_colony_github_project(
        colony._db_pool,
        colony_id=requested_colony_id,
        tenant_id=requested_tenant_id,
    )
    if colony_project is None:
        logger.info(
            "create_session: refused — colony=%s tenant=%s has "
            "no GitHub Project attached",
            requested_colony_id, requested_tenant_id,
        )
        return CreateSessionResponse(
            session_id="", status="error",
            message=(
                "This colony has no GitHub Project attached. "
                "Open the colony settings, pick a Project from "
                "the list (or create one on the repo's GitHub "
                "page first if none exist), then retry."
            ),
        )

    try:
        from polymathera.colony.agents.sessions.models import SessionMetadata
        metadata = SessionMetadata(name=request.name) if request.name else None

        sm = await colony.get_session_manager()
        session = await sm.create_session(
            metadata=metadata,
            ttl_seconds=request.ttl_seconds,
            fork_from_session_id=request.fork_from_session_id,
        )
        session_id = _get(session, "session_id", "")

        # Spawn a SessionAgent for this session.
        # The spawn needs session_id in the execution context — the AuthMiddleware
        # only sets tenant_id and colony_id. Wrap in user_execution_context.
        session_agent_id: str | None = None
        try:
            from polymathera.colony.agents import AgentMetadata, AgentHandle
            from polymathera.colony.agents.self_concept import AgentSelfConcept
            from ..chat import SessionAgent, SessionOrchestratorCapability
            from ..chat.session_agent_guardrails import (
                build_session_agent_runtime_guardrail,
            )
            from polymathera.colony.agents.patterns.capabilities.agent_pool import AgentPoolCapability
            from polymathera.colony.agents.patterns.capabilities.consciousness import ConsciousnessCapability
            from polymathera.colony.agents.patterns.capabilities.vcm import VCMCapability
            from polymathera.colony.agents.patterns.capabilities.web_search import (
                ColonyDocsCapability,
                WebSearchCapability,
            )
            from polymathera.colony.agents.patterns.capabilities.sandboxed_shell import (
                SandboxedShellCapability,
            )
            from polymathera.colony.agents.patterns.capabilities.user_plugin import (
                UserPluginCapability,
            )
            from polymathera.colony.agents.patterns.capabilities.github import (
                GitHubCapability,
            )
            from polymathera.colony.agents.roles.knowledge_curator import (
                KnowledgeCuratorCapability,
            )
            from polymathera.colony.knowledge.deps import (
                default_ingestor_blueprint,
            )
            from polymathera.colony.design_monorepo import (
                design_monorepo_capability_blueprints,
                DesignMonorepoCapabilityBase,
            )
            from polymathera.colony.agents.scopes import BlackboardScope
            from polymathera.colony.agents.patterns.planning.streams import (
                ConsciousnessStream,
                ConversationFormatter,
                EventContextKeyFilter,
                ActionKeySubstringFilter,
                SuccessfulActionFilter,
            )
            from polymathera.colony.distributed.ray_utils.serving.context import (
                ExecutionContext,
                Ring,
                get_tenant_id,
                get_colony_id,
            )

            # ``available_missions`` is populated dynamically by
            # :meth:`SessionOrchestratorCapability._refresh_available_missions`
            # on agent ``initialize()`` (and on every subsequent user
            # message), unioning :func:`get_mission_registry` (colony
            # builtins + ``polymathera.mission_types`` entry-points)
            # with L4 missions discovered under the per-agent design
            # monorepo clone via L1-A. We seed an empty dict here so
            # the planner-prompt rendering does not stumble on a
            # missing key during the brief window between agent
            # construction and ``initialize()``'s refresh — the value
            # is overwritten on first refresh.

            # AgentMetadata's ``syscontext`` field defaults to
            # ``require_execution_context()``, which captures whatever
            # is in the contextvar at construction time. The FastAPI
            # AuthMiddleware sets ``tenant_id`` + ``colony_id`` on the
            # request context but NOT ``session_id`` (the session is
            # being CREATED here). The matching
            # ``user_execution_context(session_id=...)`` block below
            # only takes effect for ``AgentHandle.from_blueprint`` —
            # by then the metadata is already frozen with a syscontext
            # whose ``session_id`` is empty.
            #
            # Build the syscontext explicitly so the SessionAgent's
            # ``metadata.session_id`` returns the real session id.
            # Without this, the tracing facility's trace_id resolver
            # (``AgentTracingFacility.get_trace_id``) falls back to
            # ``agent_id`` and the SessionAgent's spans land in a
            # disjoint trace tree from every other agent it spawns
            # in this session.
            #
            # NOTE: the prior ``session_id=session_id`` kwarg below
            # was silently dropped — ``session_id`` is a read-only
            # ``@property`` on ``AgentMetadata`` (delegating to
            # ``syscontext.session_id``), not a writable field.
            session_syscontext = ExecutionContext(
                ring=Ring.USER,
                tenant_id=get_tenant_id(),
                colony_id=get_colony_id(),
                session_id=session_id,
                origin="dashboard_session_create",
            )

            agent_metadata = AgentMetadata(
                role="session_orchestrator",
                syscontext=session_syscontext,
                goals=[
                    "Orchestrate user interactions within this session",
                    "Interpret user intent and decide whether to respond directly or spawn mission agents",
                    "Spawn and monitor coordinator agents for mission tasks",
                    "Relay agent progress and results back to the user",
                ],
                self_concept=AgentSelfConcept(
                    agent_id="",  # Overwritten by ConsciousnessCapability
                    name="Session Agent",
                    role=(
                        "the session orchestrator responsible for handling all user "
                        "interactions in this session. You receive user messages, decide "
                        "how to handle them, and can either respond directly or spawn "
                        "specialized coordinator agents to perform mission tasks."
                    ),
                    description=(
                        "You are the primary interface between the user and the Colony's "
                        "multi-agent system. You have access to an agent pool for spawning "
                        "coordinator agents (one per mission type), and you can respond "
                        "directly to user questions.\n\n"
                        "MISSION SPAWN PROTOCOL — how to start a mission task:\n"
                        "  When the user describes a task that matches one of the\n"
                        "  ``available_missions`` entries (match by ``label`` or\n"
                        "  ``description`` semantics — the user will use natural\n"
                        "  language, you map it to the right key), call:\n"
                        "    r = await run(\n"
                        "        \"spawn_mission\",\n"
                        "        mission_type=<the matching key from available_missions>,\n"
                        "        mission_params={...optional domain params...},\n"
                        "    )\n"
                        "  Then branch on ``r.success`` and ``r.output[\"created\"]``;\n"
                        "  the coordinator's agent_id is on ``r.output[\"agent_id\"]``.\n"
                        "  ``spawn_mission`` does the coordinator-class lookup\n"
                        "  internally — you do NOT extract ``coordinator_class`` from\n"
                        "  ``available_missions`` and you do NOT call ``create_agent``\n"
                        "  directly for missions. (``create_agent`` is the low-level\n"
                        "  generic spawn; ``spawn_mission`` is the mission-aware wrapper\n"
                        "  that knows how to read the registry.)\n\n"
                        "  Example — user says \"run a noise-floor analysis on the OPM-MEG design\":\n"
                        "    Iteration 1:\n"
                        "      await run(\"respond_to_user\", content=\"Starting OPM-MEG noise-floor analysis…\")\n"
                        "      r = await run(\"spawn_mission\", mission_type=\"opm_meg\")\n"
                        "      results[\"r\"] = r\n"
                        "    Iteration 2:\n"
                        "      r = results[\"r\"]\n"
                        "      if not r.success or not r.output.get(\"created\"):\n"
                        "          await run(\"respond_to_user\", content=f\"Spawn failed: {r.error or r.output.get('error')}\")\n"
                        "      else:\n"
                        "          await run(\"respond_to_user\",\n"
                        "              content=f\"Coordinator {r.output['agent_id'][:8]} is running. I'll relay updates as it works.\")\n\n"
                        "  When the user asks a question or needs information (no mission\n"
                        "  spawn), use respond_to_user directly.\n\n"
                        "  Before calling spawn_mission, if the user's message references a\n"
                        "  mission that may already be running (\"check on the analysis you\n"
                        "  started\", \"how is the planning going\"), call\n"
                        "  list_spawned_missions(mission_type=<key>) first. A non-empty\n"
                        "  result means you already have a live coordinator — reuse the\n"
                        "  returned agent_id instead of spawning a duplicate.\n\n"
                        "  When you store data in ``results`` (e.g., a query response, a\n"
                        "  child agent's status report) and the next step is a simple\n"
                        "  conditional, branch inline in the same code block using Python\n"
                        "  if/else over ``results[\"<key>\"].output``. When the next step\n"
                        "  genuinely needs fresh LLM reasoning over what you just learned\n"
                        "  (the data is too large, the decision is open-ended, or you need\n"
                        "  a new context window), call ``signal_continuation(reason=...)``\n"
                        "  at the end of the code block. Do NOT use it as a sequencing\n"
                        "  primitive — it is the escape valve for cases where inline\n"
                        "  branching cannot decide. The framework will fire one more\n"
                        "  planning turn even without a new external event; budget is\n"
                        "  finite per inbound event, so commit promptly.\n\n"
                        "AVAILABLE TOOLS — how to know what compute / retrieval / dispatch\n"
                        "actions a freshly-spawned coordinator can mount:\n"
                        "  The dict ``available_tools`` (in this agent's metadata.parameters)\n"
                        "  lists every L4 tool capability the operator's design monorepo\n"
                        "  declares via ``.colony/tool-registry.json``. Each entry maps\n"
                        "  ``<tool_name>`` → ``{purpose, location, capability, capability_fqn}``.\n"
                        "  The coordinator spawned via ``spawn_mission`` can mount any of these\n"
                        "  by passing the ``capability_fqn`` string in its ``capabilities=[...]``\n"
                        "  list at create_agent time. (``spawn_mission`` handles this for you when\n"
                        "  the mission's worker class declares them in its blueprint.) Tools with\n"
                        "  ``capability_fqn`` set are mountable; entries with an empty\n"
                        "  ``capability_fqn`` are catalog-only candidates surveyed by the\n"
                        "  build-vs-buy advisor — they're for context, not for mounting.\n\n"
                        "DESIGN-MONOREPO BOOTSTRAP:\n"
                        "  When the user asks to initialize / bootstrap / scaffold the "
                        "  design monorepo (or asks 'how do I set up repo_map.yaml'), "
                        "  call ``initialize_repo_map`` (on DesignCheckpointer). Do NOT "
                        "  use ``mmap_repo`` for this — ``mmap_repo`` maps an existing "
                        "  repo into VCM and assumes ``.colony/repo_map.yaml`` already "
                        "  exists. ``initialize_repo_map`` writes a commented template "
                        "  and commits it; it is idempotent and never overwrites an "
                        "  operator-edited file.\n\n"
                        "DESIGN-MONOREPO LITERATURE INGESTION:\n"
                        "  When the user asks to ingest / process / load literature "
                        "  from the design monorepo (or 'process repo_map.yaml'), call "
                        "  ``ingest_repo_map_literature`` (on RepoStateProvider). It "
                        "  refreshes the per-agent clone against ``origin``, reads "
                        "  ``repo_map.yaml``, and ingests every file matched by a "
                        "  ``knowledge_sources:`` row that is currently ENABLED in the "
                        "  Design Monorepo tab's checkbox list. The selection is "
                        "  persisted server-side per-colony, so the action picks it up "
                        "  automatically — do NOT pass any source-filter argument. If "
                        "  the user wants different rows ingested, ask them to toggle "
                        "  the Design Monorepo tab's 'Knowledge sources' checkboxes; "
                        "  do not try to filter from chat. Use this — NOT ``ingest_file`` "
                        "  per file — when the user wants the whole literature corpus loaded.\n\n"
                        "INTERACTION PROTOCOL — long-running actions:\n"
                        "  Some actions take seconds or minutes (mmap_repo, create_agent, "
                        "  search_and_fetch, run_skill, claim_unassigned_issue, …). For "
                        "  any action you suspect will take more than ~2 seconds, FIRST "
                        "  call respond_to_user with a one-line acknowledgement that "
                        "  names the action you are about to run, THEN call the action. "
                        "  This keeps the user informed while the work happens. After "
                        "  the action returns, respond_to_user again with the outcome.\n\n"
                        "ITERATION DISCIPLINE — small steps, persistent state:\n"
                        "  Each REPL iteration is a focused step (≤3 actions on any "
                        "  one execution path). Variables you bind in ``results[...]`` "
                        "  persist across iterations, so you do NOT need to pack the "
                        "  whole sequence into one cell. The natural shape is:\n"
                        "    Iteration 1 — ack + the action (2 calls), store result.\n"
                        "    Iteration 2 — read ``results[...]`` and respond.\n"
                        "  An ack-then-action-then-branched-response (success / failure "
                        "  arms) IS allowed in one iteration because only one branch "
                        "  ever runs (≤3 actions per path); a straight-line program "
                        "  with 4+ actions is not. When in doubt, defer the response "
                        "  to the next iteration.\n\n"
                        "RESULT INSPECTION — branch on the action result:\n"
                        "  ``run(...)`` returns an ``ActionResult`` Pydantic "
                        "  model — NOT a dict. Use attribute access, not "
                        "  ``.get(...)``:\n"
                        "    r.success      → bool (True iff the action ran "
                        "                     to completion without error)\n"
                        "    r.output       → the action's actual return "
                        "                     value (a dict for most actions; "
                        "                     None on failure)\n"
                        "    r.error        → str | None (set on failure)\n"
                        "    r.cancelled    → bool (set when /abort fired)\n"
                        "  Never forward a result blindly. After every "
                        "  action, branch on ``r.success`` and on the shape "
                        "  of ``r.output``, surfacing zero-count / empty / "
                        "  error cases explicitly. Doing this in the SAME "
                        "  iteration that ran the action is allowed under "
                        "  ITERATION DISCIPLINE because only one branch "
                        "  executes:\n"
                        "    r = await run(\"some.action\", ...)\n"
                        "    results[\"r\"] = r\n"
                        "    if not r.success:\n"
                        "        await run(\"respond_to_user\", text=f\"…X failed: {r.error}\")\n"
                        "    else:\n"
                        "        out = r.output or {}\n"
                        "        if not out or out.get(\"count\", 0) == 0:\n"
                        "            await run(\"respond_to_user\", text=\"…ran X but it returned 0 items; likely cause: …\")\n"
                        "        else:\n"
                        "            await run(\"respond_to_user\", text=f\"…X succeeded: {out}\")\n"
                        "  Apply this to ANY action whose ``output`` is a "
                        "  dict with count / list / status / error fields — "
                        "  not just literature ingestion. When the empty "
                        "  case has a likely cause you can name (empty "
                        "  knowledge_sources, unreachable origin, missing "
                        "  file), include it in the message so the user can "
                        "  act on it.\n\n"
                        "RESPONSE FORMATTING — pass structured data as attachments:\n"
                        "  ``respond_to_user(content, attachments=[…])`` "
                        "  renders ``content`` as Markdown (GFM) and each "
                        "  attachment below it as a typed, collapsible "
                        "  block. Use this — NEVER hand-roll markdown "
                        "  fences or ``str(dict)`` into ``content`` — for "
                        "  any structured payload (dicts, lists, "
                        "  ActionResult.output). The framework owns the "
                        "  rendering: indentation, syntax highlighting, "
                        "  collapse-when-long, the lot.\n"
                        "  Three attachment kinds, plus two convenience "
                        "  actions that wrap them:\n"
                        "    1. CODE — ``{\"kind\":\"code\",\"lang\":\"python\","
                        "       \"content\":<value>,\"label\":<str>?}``.\n"
                        "       Pass the raw value (dict, list,\n"
                        "       ``ActionResult.output``) directly as\n"
                        "       ``content``. The framework pretty-prints\n"
                        "       it server-side based on ``lang``\n"
                        "       (``python``→pprint, ``json``→json.dumps,\n"
                        "       ``yaml``→yaml.safe_dump). NEVER call\n"
                        "       ``str(value)`` yourself — that produces a\n"
                        "       single-line blob that the chat cannot lay\n"
                        "       out cleanly.\n"
                        "         await run(\n"
                        "             \"respond_to_user\",\n"
                        "             content=\"✅ Ingestion complete — 20 file(s) loaded.\",\n"
                        "             attachments=[{\n"
                        "                 \"kind\": \"code\",\n"
                        "                 \"lang\": \"python\",\n"
                        "                 \"content\": out,  # raw dict; framework formats it\n"
                        "                 \"label\": \"ingest result\",\n"
                        "             }],\n"
                        "         )\n"
                        "    2. TABLE — call ``respond_to_user_with_table(\n"
                        "       summary=<str>, rows=[{...}, …],\n"
                        "       columns=[…]?, label=<str>?)``. Use for\n"
                        "       per-file outcomes, side-by-side metrics.\n"
                        "         await run(\n"
                        "             \"respond_to_user_with_table\",\n"
                        "             summary=\"Per-file outcomes:\",\n"
                        "             rows=[{\"file\": p, \"status\": s} for p, s in pairs],\n"
                        "             columns=[\"file\", \"status\"],\n"
                        "         )\n"
                        "    3. DIFF — call ``respond_to_user_with_diff(\n"
                        "       summary=<str>, before=<str>, after=<str>,\n"
                        "       lang=<str>?, label=<str>?)``. Use for\n"
                        "       state changes (config edits, file\n"
                        "       rewrites). The chat renders ``+/-`` lines.\n"
                        "         await run(\n"
                        "             \"respond_to_user_with_diff\",\n"
                        "             summary=\"Updated repo_map.yaml:\",\n"
                        "             before=old_yaml, after=new_yaml,\n"
                        "             lang=\"yaml\", label=\".colony/repo_map.yaml\",\n"
                        "         )\n"
                        "  Keep ``summary`` / ``content`` short and "
                        "  human-readable — the user reads it first and "
                        "  only expands attachments for details. Do NOT "
                        "  embed a fenced ```` ```python … ``` ```` block "
                        "  inside ``content`` when you also pass a code "
                        "  attachment; pick one path.\n\n"
                        "OUTPUT FORMAT — code generation:\n"
                        "  Emit ONLY raw Python code at the cell level. "
                        "  Do NOT wrap the whole iteration in a markdown "
                        "  fence — the REPL executes the cell verbatim and "
                        "  a leading ``` would crash it. Prefer attachments "
                        "  over markdown fences inside ``content=`` (see "
                        "  RESPONSE FORMATTING). No mocked result blocks. "
                        "  No prose between statements. Each iteration is "
                        "  a single focused snippet."
                    ),
                ),
                parameters={
                    # SESSION-scoped keys live on
                    # ``SessionOrchestratorCapability.AGENT_METADATA_PARAMS``;
                    # we seed an empty dict + the REPL override here so
                    # the planner-prompt rendering works during the
                    # brief window between agent construction and the
                    # first ``_refresh_available_*`` pass. ``session_id``
                    # is NOT seeded — it's a typed property on
                    # ``AgentMetadata`` (via syscontext), never a
                    # parameters key.
                    SessionOrchestratorCapability.AVAILABLE_MISSIONS_KEY: {},
                    SessionOrchestratorCapability.AVAILABLE_TOOLS_KEY: {},
                    SessionOrchestratorCapability.REPL_GUIDANCE_OVERRIDE_KEY: (
                        "## REPL\n\n"
                        "You have a Python REPL. Use `await run(\"action_key\", ...)` to "
                        "execute capability actions. Use `results[\"key\"] = value` to "
                        "store results. Use `log(msg)` for structured logging."
                    ),
                    # Per-colony design-monorepo URL — populated from
                    # the ``colonies`` table. The design-monorepo
                    # capability trio reads this in ``_client_sync`` to
                    # lazy-clone the repo into the per-agent working
                    # directory on first access. ``None`` when the
                    # colony has no design monorepo configured yet.
                    DesignMonorepoCapabilityBase.DESIGN_MONOREPO_URL_KEY: (
                        await auth_service.get_design_monorepo(
                            colony._db_pool,
                            colony_id=get_colony_id() or "",
                            tenant_id=get_tenant_id() or "",
                        ) or {}
                    ).get("origin_url"),
                    # Per-commit attribution config — read once at
                    # session-creation, agents look it up on metadata
                    # for every commit they produce. Defaults baked
                    # into the schema return ``commit_principal=colony``
                    # / ``commit_co_author=user`` for fresh colonies;
                    # operator overrides via the landing page.
                    DesignMonorepoCapabilityBase._GIT_ATTRIBUTION_KEY: (
                        await auth_service.get_git_attribution(
                            colony._db_pool,
                            colony_id=get_colony_id() or "",
                            tenant_id=get_tenant_id() or "",
                        ) or {}
                    ),
                    # GitHub identity for this session (P4 of
                    # ``colony/github_identity_fix_plan.md``).
                    # Per-tenant: the App installation id Colony uses
                    # to mint REST tokens scoped to this tenant's
                    # repos (read by ``GitHubCapability`` in P5).
                    # Per-user: the OAuth-verified GitHub login +
                    # email + name (read by ``_resolve_attribution``
                    # in P6 and by ``propose_task_assignments`` in P8).
                    # Every field is ``None`` until the operator wires
                    # the corresponding piece; downstream readers
                    # handle the unset case explicitly.
                    GitHubCapability.GITHUB_IDENTITY_KEY: _resolve_github_identity(
                        await auth_service.get_tenant_github_installation(
                            colony._db_pool,
                            tenant_id=user.get("tenant_id", ""),
                        ),
                        await auth_service.get_user_github_identity(
                            colony._db_pool,
                            user_id=user.get("sub", ""),
                        ),
                    ),
                },
                action_policy_config={
                    "allow_self_termination": False,  # SessionAgent should not terminate itself — the session lives on until the user closes it
                    "reactive_only": True, # SessionAgent only responds to user messages, it doesn't have proactive goals or actions
                    "planning_capability_blueprints": [],
                },
            )

            bp = SessionAgent.bind(
                agent_type=SessionAgent.model_fields["agent_type"].default,
                metadata=agent_metadata,
                bound_pages=[],
                capability_blueprints=[
                    SessionOrchestratorCapability.bind(),
                    AgentPoolCapability.bind(),
                    ConsciousnessCapability.bind(),
                    VCMCapability.bind(scope=BlackboardScope.SESSION),
                    WebSearchCapability.bind(scope=BlackboardScope.SESSION),
                    ColonyDocsCapability.bind(scope=BlackboardScope.SESSION),
                    SandboxedShellCapability.bind(scope=BlackboardScope.SESSION),
                    UserPluginCapability.bind(
                        scope=BlackboardScope.SESSION,
                        extra_plugin_roots=[_bundled_samples_plugins_root()],
                    ),
                    # GitHubCapability starts in a "disabled" state if
                    # no credentials are configured; the action surface
                    # surfaces a clean error instead of crashing the
                    # agent. Operators enable it via env vars or the
                    # Settings UI (planned).
                    #
                    # ``default_project_id`` is the colony's attached
                    # GitHub Project (v2) node id — every new issue
                    # ``GitHubCapability.create_issue`` mints gets
                    # auto-attached to this project. Resolved upfront
                    # (see the lookup block above the spawn).
                    GitHubCapability.bind(
                        scope=BlackboardScope.SESSION,
                        default_project_id=colony_project["node_id"],
                    ),
                    # Design-monorepo capability trio (state, checkpointing,
                    # tool building) — per-agent clones under
                    # /mnt/shared/agents/<agent_id>/clones/<scope_id>/.
                    # SessionAgent runs in ``reactive_only`` mode so it
                    # must NOT subscribe to the convergence-quiescence
                    # stream — every episode boundary would otherwise
                    # wake the LLM planner, producing the same welcome
                    # message in a tight loop. Sub-agents that perform
                    # actual design work get the default trio (with
                    # auto-checkpoint enabled) when they are spawned.
                    *design_monorepo_capability_blueprints(
                        auto_checkpoint_on_quiescence=False,
                    ),
                    # Knowledge curation — chat-driven write side. The
                    # dashboard ships *blueprints*, not live instances:
                    # the Ingestor + RetrievalDeps wrap an
                    # ``AsyncQdrantClient`` (RLock) and an
                    # ``InMemoryEmbedder`` (local closure), neither of
                    # which survives cloudpickle. The blueprint chain
                    # (Ingestor.bind → InMemoryEmbedder.bind +
                    # QdrantVectorStore.bind) carries only picklable
                    # kwargs (URLs, collection names, dimensions) and
                    # is resolved on the worker via ``local_instance()``
                    # — same pattern as ``ConsciousnessStream(formatter=…)``.
                    # Same KnowledgeConfig (loaded from the same YAML),
                    # same collection, separate Python objects per process.
                    #
                    # ``KnowledgeRetrievalCapability`` is auto-injected
                    # for every agent in ``Agent._create_action_policy``
                    # (Phase 1c). Bulk acquisition / corpus ingestion
                    # runs through
                    # :meth:`RepoStateProvider.ingest_repo_map_literature`
                    # (bound above via ``design_monorepo_capability_blueprints``);
                    # the unified ``.colony/repo_map.yaml`` schema is the
                    # single source of truth for which sources land in
                    # the KB.
                    KnowledgeCuratorCapability.bind(
                        ingestor=default_ingestor_blueprint(),
                    ),
                ],
                action_policy_blueprints={
                    "consciousness_streams": [
                        ConsciousnessStream.bind(
                            name="conversation",
                            formatter=ConversationFormatter.bind(),
                            event_filter=EventContextKeyFilter("user_chat_message"),
                            action_filter=SuccessfulActionFilter(
                                ActionKeySubstringFilter("respond_to_user")
                            ),
                        ),
                    ],
                    # Hard guardrails mounted on the code-generation
                    # action policy. See ``session_agent_guardrails``
                    # for the rules + ``colony/mission_and_action_
                    # guardrails_plan.md`` Part 2 for the design.
                    # Travels through cloudpickle to the Ray worker
                    # via the ``exclude=True`` field — module-level
                    # named predicates keep the serialised graph
                    # picklable.
                    "runtime_guardrail": (
                        build_session_agent_runtime_guardrail()
                    ),
                },
            )

            with colony.user_execution_context(
                tenant_id=get_tenant_id(),
                colony_id=get_colony_id(),
                session_id=session_id,
                origin="dashboard_session_create",
            ):
                handle = await AgentHandle.from_blueprint(
                    agent_blueprint=bp,
                    app_name=colony.app_name,
                )
            session_agent_id = handle.agent_id
            logger.info("Spawned SessionAgent %s for session %s", session_agent_id, session_id)
        except Exception as e:
            logger.error("Failed to spawn SessionAgent for session %s: %s", session_id, e)
            # Session still works without an agent — chat will fall back to direct agent routing

        session_agent_id1 = await sm.set_session_agent_id(
            session_id=session_id,
            agent_id=session_agent_id
        )
        if session_agent_id1 != session_agent_id:
            return CreateSessionResponse(
                session_id=session_id,
                status="error",
                message=f"Session created but failed to set session agent ID. Expected {session_agent_id}, got {session_agent_id1}",
            )

        return CreateSessionResponse(session_id=session_id, status="created")

    except Exception as e:
        logger.error("Failed to create session: %s", e)
        return CreateSessionResponse(session_id="", status="error", message=str(e))


@router.put("/sessions/{session_id}/suspend", response_model=SessionActionResponse)
async def suspend_session(
    session_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> SessionActionResponse:
    """Suspend an active session."""
    if not colony.is_connected:
        return SessionActionResponse(session_id=session_id, success=False, message="Not connected")

    try:
        handle = await colony.get_session_manager()
        success = await handle.suspend_session(session_id=session_id)
        return SessionActionResponse(
            session_id=session_id,
            success=bool(success),
            message="Suspended" if success else "Failed to suspend",
        )
    except Exception as e:
        return SessionActionResponse(session_id=session_id, success=False, message=str(e))


@router.put("/sessions/{session_id}/resume", response_model=SessionActionResponse)
async def resume_session(
    session_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> SessionActionResponse:
    """Resume a suspended session."""
    if not colony.is_connected:
        return SessionActionResponse(session_id=session_id, success=False, message="Not connected")

    try:
        handle = await colony.get_session_manager()
        success = await handle.activate_session(session_id=session_id)
        return SessionActionResponse(
            session_id=session_id,
            success=bool(success),
            message="Resumed" if success else "Failed to resume",
        )
    except Exception as e:
        return SessionActionResponse(session_id=session_id, success=False, message=str(e))


@router.delete("/sessions/{session_id}", response_model=SessionActionResponse)
async def close_session(
    session_id: str,
    archive: bool = Query(True, description="Archive session data"),
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> SessionActionResponse:
    """Close and optionally archive a session."""
    if not colony.is_connected:
        return SessionActionResponse(session_id=session_id, success=False, message="Not connected")

    try:
        handle = await colony.get_session_manager()
        success = await handle.close_session(session_id=session_id, archive=archive)
        return SessionActionResponse(
            session_id=session_id,
            success=bool(success),
            message="Closed" if success else "Failed to close",
        )
    except Exception as e:
        return SessionActionResponse(session_id=session_id, success=False, message=str(e))


@router.post("/sessions/{session_id}/runs/{run_id}/cancel", response_model=SessionActionResponse)
async def cancel_run(
    session_id: str,
    run_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> SessionActionResponse:
    """Cancel a running agent run."""
    if not colony.is_connected:
        return SessionActionResponse(session_id=session_id, success=False, message="Not connected")

    try:
        handle = await colony.get_session_manager()
        success = await handle.cancel_run(run_id=run_id)
        return SessionActionResponse(
            session_id=session_id,
            success=bool(success),
            message=f"Run {run_id} cancelled" if success else f"Failed to cancel run {run_id}",
        )
    except Exception as e:
        return SessionActionResponse(session_id=session_id, success=False, message=str(e))
