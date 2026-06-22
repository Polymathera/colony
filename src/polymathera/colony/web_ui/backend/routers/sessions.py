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
        # Persist the auth ``sub`` on SessionMetadata.created_by so
        # PR1-B respawn can resolve the same per-user GitHub identity
        # the original session had (see
        # :func:`spawn_user_session_agent_for_session`). The field is
        # required on the model; the manager.py:218 fallback that
        # writes ``created_by="session_manager"`` is a default for
        # internal callers, not the chat-create path.
        metadata = SessionMetadata(
            name=request.name,
            created_by=user.get("sub", ""),
        )

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
        # PR1-B (R12-ROOT-CAUSE-C): spawn through the re-entrant
        # factory in chat/user_session_factory.py so the same
        # blueprint shape is used for create-session AND for the
        # dashboard's lazy respawn path when the SessionAgent dies
        # mid-session. All per-colony / per-user state the
        # blueprint needs is fetched from postgres inside the
        # factory; the user's auth ``sub`` is persisted on
        # SessionMetadata.created_by above so respawn resolves the
        # same GitHub identity.
        session_agent_id: str | None = None
        try:
            from ..chat.user_session_factory import spawn_user_session_agent_for_session
            session_agent_id = await spawn_user_session_agent_for_session(
                colony,
                session_id=session_id, # 
                tenant_id=get_tenant_id() or "",  # user.get("tenant_id", "")?
                colony_id=get_colony_id() or "",
                user_sub=user.get("sub", ""),
            )
            if session_agent_id is None:
                logger.error(
                    "spawn_user_session_agent_for_session returned None "
                    "for session %s — see prior log for the underlying"
                    " failure.", session_id,
                )
            else:
                logger.info(
                    "Spawned SessionAgent %s for session %s",
                    session_agent_id, session_id,
                )
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


# ---------------------------------------------------------------------------
# PR5-B (R12-E): operator runtime override for semantic constraints
# ---------------------------------------------------------------------------


async def _write_semantic_constraint_override(
    *,
    colony: ColonyConnection,
    session_id: str,
    constraint_id: str,
    disabled: bool,
    user: dict,
) -> SessionActionResponse:
    """Write an :class:`OperatorOverrideProtocol` record onto the
    session's default blackboard. The same key is read live by
    :meth:`SemanticConstraintGuardrail.check` on each iteration —
    no agent-side subscription / mutation step in between, so the
    override survives SessionAgent respawn without any sync logic.
    """

    import time as _time
    from polymathera.colony.agents.blackboard import EnhancedBlackboard
    from polymathera.colony.agents.blackboard.protocol import (
        OperatorOverrideProtocol,
    )
    from polymathera.colony.agents.scopes import (
        BlackboardScope,
        get_scope_prefix,
    )
    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring,
        execution_context,
    )

    if not colony.is_connected:
        return SessionActionResponse(
            session_id=session_id, success=False, message="Not connected",
        )

    try:
        # Resolve tenant/colony from the session row so the bb scope
        # matches what the SessionAgent's subscription reads.
        sm = await colony.get_session_manager()
        session = await sm.get_session(session_id=session_id)
        if session is None:
            return SessionActionResponse(
                session_id=session_id, success=False,
                message=f"Session {session_id} not found",
            )
        from polymathera.colony.agents.sessions.models import Session as SM
        if isinstance(session, dict):
            session = SM(**session)
        with execution_context(
            ring=Ring.USER,
            tenant_id=session.tenant_id,
            colony_id=session.colony_id,
            session_id=session_id,
            origin="dashboard_constraint_override",
        ):
            bb = EnhancedBlackboard(
                app_name=colony.app_name,
                scope_id=get_scope_prefix(BlackboardScope.SESSION),
                backend_type=None,
                enable_events=True,
            )
            await bb.initialize()
            await bb.write(
                OperatorOverrideProtocol.semantic_constraint_key(constraint_id),
                {
                    "disabled": disabled,
                    "set_at": _time.time(),
                    "set_by": user.get("sub", "unknown"),
                },
            )
        verb = "disabled" if disabled else "enabled"
        return SessionActionResponse(
            session_id=session_id,
            success=True,
            message=(
                f"Operator {verb} semantic constraint "
                f"{constraint_id!r} for session {session_id}"
            ),
        )
    except Exception as e:
        logger.error(
            "operator override (%s) failed: session=%s constraint=%s err=%s",
            "disable" if disabled else "enable",
            session_id, constraint_id, e,
        )
        return SessionActionResponse(
            session_id=session_id, success=False, message=str(e),
        )


class SemanticConstraintSummary(BaseModel):
    """One row of the per-session constraint catalogue returned by
    ``GET /sessions/{id}/constraints``. Carries the static metadata
    the operator needs to identify the rule + the live disabled
    state so the UI toggle is initialized correctly."""

    id: str
    rule_nl: str
    scope: str
    failure_mode: str
    disabled: bool


@router.get(
    "/sessions/{session_id}/constraints",
    response_model=list[SemanticConstraintSummary],
)
async def list_session_constraints(
    session_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> list[SemanticConstraintSummary]:
    """List the SessionAgent's active semantic constraints + which
    are currently operator-disabled. Powers the constraints toggle
    panel in the chat UI."""

    from polymathera.colony.agents.blackboard import EnhancedBlackboard
    from polymathera.colony.agents.blackboard.protocol import (
        OperatorOverrideProtocol,
    )
    from polymathera.colony.agents.scopes import (
        BlackboardScope,
        get_scope_prefix,
    )
    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring,
        execution_context,
    )
    from ..chat.session_agent_guardrails import (
        session_agent_semantic_constraints,
    )

    catalogue = session_agent_semantic_constraints()
    if not colony.is_connected:
        return [
            SemanticConstraintSummary(
                id=c.id,
                rule_nl=c.rule_nl,
                scope=c.scope.value,
                failure_mode=c.failure_mode.value,
                disabled=False,
            )
            for c in catalogue
        ]

    try:
        sm = await colony.get_session_manager()
        session = await sm.get_session(session_id=session_id)
        if session is None:
            return []
        from polymathera.colony.agents.sessions.models import Session as SM
        if isinstance(session, dict):
            session = SM(**session)
        with execution_context(
            ring=Ring.USER,
            tenant_id=session.tenant_id,
            colony_id=session.colony_id,
            session_id=session_id,
            origin="dashboard_constraint_list",
        ):
            bb = EnhancedBlackboard(
                app_name=colony.app_name,
                scope_id=get_scope_prefix(BlackboardScope.SESSION),
                backend_type=None,
                enable_events=False,
            )
            await bb.initialize()
            entries = await bb.query(
                namespace=OperatorOverrideProtocol.semantic_constraint_pattern(),
            )
        disabled_ids: set[str] = set()
        for e in entries:
            try:
                cid = OperatorOverrideProtocol.parse_semantic_constraint_key(
                    e.key,
                )
            except ValueError:
                continue
            payload = e.value if isinstance(e.value, dict) else {}
            if bool(payload.get("disabled", False)):
                disabled_ids.add(cid)
        return [
            SemanticConstraintSummary(
                id=c.id,
                rule_nl=c.rule_nl,
                scope=c.scope.value,
                failure_mode=c.failure_mode.value,
                disabled=(c.id in disabled_ids),
            )
            for c in catalogue
        ]
    except Exception as e:
        logger.error(
            "list_session_constraints failed: session=%s err=%s",
            session_id, e,
        )
        return []


@router.post(
    "/sessions/{session_id}/constraints/{constraint_id}/disable",
    response_model=SessionActionResponse,
)
async def disable_semantic_constraint(
    session_id: str,
    constraint_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> SessionActionResponse:
    """Operator-disable a semantic constraint for this session.

    The SessionAgent's runtime guardrail skips the named constraint
    (no precondition / no verifier / no advisory) for the rest of
    the session, or until ``/enable`` flips it back. Useful when a
    rule is misbehaving in ways the auto-escalation can't catch
    (wrong rule, persistent false-positives across multiple drafts).
    """

    return await _write_semantic_constraint_override(
        colony=colony,
        session_id=session_id,
        constraint_id=constraint_id,
        disabled=True,
        user=user,
    )


@router.post(
    "/sessions/{session_id}/constraints/{constraint_id}/enable",
    response_model=SessionActionResponse,
)
async def enable_semantic_constraint(
    session_id: str,
    constraint_id: str,
    user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> SessionActionResponse:
    """Re-enable a previously operator-disabled semantic constraint."""

    return await _write_semantic_constraint_override(
        colony=colony,
        session_id=session_id,
        constraint_id=constraint_id,
        disabled=False,
        user=user,
    )
