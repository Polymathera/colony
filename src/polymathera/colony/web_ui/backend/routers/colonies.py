"""Colony (workspace) management endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth.middleware import require_auth
from ..auth import service as auth_service
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


class CreateColonyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    # Repo binding (optional — user can leave blank to create a
    # "bare" colony and set the design monorepo later). All four
    # fields must be supplied together when present.
    vcs_repo_id: str | None = None
    vcs_repo_full_name: str | None = None
    default_branch: str | None = None
    # Per-commit attribution preferences. Defaults match the schema
    # (colony as principal, user as co-author).
    commit_principal: str | None = None
    commit_co_author: str | None = None
    # GitHub Project (v2) attachment. Required when a repo is
    # supplied so session-create can spawn a SessionAgent — see
    # ``routers/sessions.py`` for the session-create gate. The
    # picker on the "+ New Colony" form populates these from the
    # ``/discoverable-projects`` route; ``None`` is only valid for
    # bare colonies (no repo bound).
    github_project_node_id: str | None = None
    github_project_title: str | None = None


class ColonyInfo(BaseModel):
    colony_id: str
    name: str
    tenant_id: str
    description: str = ""
    # Per plan §3.4: a colony is bound to a VCS repo (many-to-one
    # — multiple colonies can point at the same repo). NULL during
    # the transient window between row create and repo selection.
    vcs_repo_id: str | None = None
    vcs_repo_full_name: str | None = None
    default_branch: str | None = None
    created_at: str | None = None


def _get_db_pool(colony: ColonyConnection):
    pool = colony._db_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return pool


@router.get("/colonies/", response_model=list[ColonyInfo])
async def list_colonies(
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> list[ColonyInfo]:
    """List all colonies for the current user."""
    db = _get_db_pool(colony)
    colonies = await auth_service.list_colonies(db, user["tenant_id"])
    return [ColonyInfo(**c) for c in colonies]


@router.post("/colonies/", response_model=ColonyInfo)
async def create_colony(
    request: CreateColonyRequest,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> ColonyInfo:
    """Create a new colony (workspace).

    Funnels through :func:`provision_colony` — the single colony
    creation entry point that pairs the SQL insert with the
    per-colony system ``SessionAgent`` bootstrap. See
    ``services/colony_lifecycle.py`` for the rationale.
    """
    _get_db_pool(colony)  # 503 early if the db pool is missing
    # Operator-initiated colony create with a repo bound MUST include
    # a GitHub Project node id — see ``services/colony_lifecycle.py``
    # for the rationale. Auto-discovery's walker calls
    # ``provision_colony`` directly and is intentionally allowed to
    # defer the pick.
    if request.vcs_repo_full_name and not request.github_project_node_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Pick a GitHub Project to attach to this colony "
                "before creating it. If the repository has no "
                "Projects, create one on the repo's GitHub page "
                "first (Projects tab → New project)."
            ),
        )
    from ..services.colony_lifecycle import provision_colony
    result = await provision_colony(
        colony,
        tenant_id=user["tenant_id"],
        name=request.name,
        description=request.description,
        vcs_repo_id=request.vcs_repo_id,
        vcs_repo_full_name=request.vcs_repo_full_name,
        default_branch=request.default_branch,
        commit_principal=request.commit_principal,
        commit_co_author=request.commit_co_author,
        github_project_node_id=request.github_project_node_id,
        github_project_title=request.github_project_title,
    )
    return ColonyInfo(**result)


# ---------------------------------------------------------------------------
# Per-colony design-monorepo URL — single source of truth for the
# SessionAgent's per-agent clones, the materialiser's ``Map Repo``
# trigger, and the dashboard's "Design Monorepo" tab. Auth (PAT,
# credential helper, ssh-agent, …) is operator-managed via the
# colony's git environment; we surface clone errors verbatim.
# ---------------------------------------------------------------------------


class DesignMonorepoConfig(BaseModel):
    origin_url: str | None = Field(
        default=None,
        description="Git URL. ``None`` when the colony has no design monorepo configured.",
    )
    branch: str = "main"
    commit: str = "HEAD"


class SetDesignMonorepoRequest(BaseModel):
    origin_url: str = Field(min_length=1)
    branch: str = "main"
    commit: str = "HEAD"


@router.get(
    "/colonies/{colony_id}/design-monorepo",
    response_model=DesignMonorepoConfig,
)
async def get_colony_design_monorepo(
    colony_id: str,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> DesignMonorepoConfig:
    """Return the colony's design-monorepo URL (if configured)."""
    db = _get_db_pool(colony)
    row = await auth_service.get_design_monorepo(
        db, colony_id=colony_id, tenant_id=user["tenant_id"],
    )
    if row is None:
        return DesignMonorepoConfig()
    return DesignMonorepoConfig(**row)


@router.put(
    "/colonies/{colony_id}/design-monorepo",
    response_model=DesignMonorepoConfig,
)
async def set_colony_design_monorepo(
    colony_id: str,
    request: SetDesignMonorepoRequest,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> DesignMonorepoConfig:
    """Persist the colony's design-monorepo URL.

    The dashboard's "Design Monorepo" tab calls this. The SessionAgent
    can call it from chat via the ``set_design_monorepo`` action on
    :class:`DesignMonorepoBootstrap` — both end up here.
    """

    db = _get_db_pool(colony)
    try:
        row = await auth_service.set_design_monorepo(
            db,
            colony_id=colony_id,
            tenant_id=user["tenant_id"],
            origin_url=request.origin_url,
            branch=request.branch,
            commit=request.commit,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return DesignMonorepoConfig(**row)


# ---------------------------------------------------------------------------
# Per-colony source selection — the operator's checkbox state from the
# "Design Monorepo" tab. Two independent sets, both read by the
# materialiser at action time so chat-driven ingestion (which has no
# checkboxes) honours the same selection.
# ---------------------------------------------------------------------------


class SourceSelection(BaseModel):
    """List of row names the operator has ticked. ``None`` (the
    default until the operator changes anything) means "all rows
    enabled" — same convention as
    ``materialize_*_sources(enabled_sources=...)``."""

    enabled: list[str] | None = None


@router.get(
    "/colonies/{colony_id}/enabled-vcm-sources",
    response_model=SourceSelection,
)
async def get_colony_enabled_vcm_sources(
    colony_id: str,
    _user: dict[str, Any] = Depends(require_auth),
) -> SourceSelection:
    from polymathera.colony.design_monorepo.source_selection import (
        list_enabled_vcm_sources,
    )
    return SourceSelection(enabled=await list_enabled_vcm_sources(colony_id))


@router.put(
    "/colonies/{colony_id}/enabled-vcm-sources",
    response_model=SourceSelection,
)
async def set_colony_enabled_vcm_sources(
    colony_id: str,
    request: SourceSelection,
    _user: dict[str, Any] = Depends(require_auth),
) -> SourceSelection:
    from polymathera.colony.design_monorepo.source_selection import (
        set_enabled_vcm_sources,
    )
    await set_enabled_vcm_sources(colony_id, request.enabled)
    return request


@router.get(
    "/colonies/{colony_id}/enabled-knowledge-sources",
    response_model=SourceSelection,
)
async def get_colony_enabled_knowledge_sources(
    colony_id: str,
    _user: dict[str, Any] = Depends(require_auth),
) -> SourceSelection:
    from polymathera.colony.design_monorepo.source_selection import (
        list_enabled_knowledge_sources,
    )
    return SourceSelection(
        enabled=await list_enabled_knowledge_sources(colony_id),
    )


@router.put(
    "/colonies/{colony_id}/enabled-knowledge-sources",
    response_model=SourceSelection,
)
async def set_colony_enabled_knowledge_sources(
    colony_id: str,
    request: SourceSelection,
    _user: dict[str, Any] = Depends(require_auth),
) -> SourceSelection:
    from polymathera.colony.design_monorepo.source_selection import (
        set_enabled_knowledge_sources,
    )
    await set_enabled_knowledge_sources(colony_id, request.enabled)
    return request


# ---------------------------------------------------------------------------
# Per-colony git-commit attribution (principal + optional co-author).
# Default: principal=colony, co_author=user — the persistent
# collective identity does the work on behalf of the human who
# started the session. Operator can swap to per-agent or any
# free-form agent-type label via the landing-page UI.
# ---------------------------------------------------------------------------


class GitAttributionConfig(BaseModel):
    commit_principal: str = Field(
        default="colony",
        description=(
            "Free-form identity string. Well-known: ``user`` / "
            "``colony`` / ``agent``. Anything else is treated as an "
            "agent-type label (e.g. ``session_agent``)."
        ),
    )
    commit_co_author: str | None = Field(
        default="user",
        description=(
            "Optional second identity, rendered as a "
            "``Co-Authored-By:`` trailer. Same value space as "
            "``commit_principal``. ``null`` disables the trailer."
        ),
    )
    # Per-user identity (``git_user_name`` / ``git_user_email``) is
    # OAuth-verified on ``users`` now (P1 of
    # ``colony/github_identity_fix_plan.md``); no per-colony fields.


class SetGitAttributionRequest(BaseModel):
    commit_principal: str = "colony"
    commit_co_author: str | None = "user"


@router.get(
    "/colonies/{colony_id}/git-attribution",
    response_model=GitAttributionConfig,
)
async def get_colony_git_attribution(
    colony_id: str,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> GitAttributionConfig:
    """Return the colony's per-commit attribution settings."""
    db = _get_db_pool(colony)
    row = await auth_service.get_git_attribution(
        db, colony_id=colony_id, tenant_id=user["tenant_id"],
    )
    if row is None:
        return GitAttributionConfig()
    return GitAttributionConfig(**row)


@router.put(
    "/colonies/{colony_id}/git-attribution",
    response_model=GitAttributionConfig,
)
async def set_colony_git_attribution(
    colony_id: str,
    request: SetGitAttributionRequest,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> GitAttributionConfig:
    """Persist the colony's per-commit attribution settings."""

    db = _get_db_pool(colony)
    try:
        row = await auth_service.set_git_attribution(
            db,
            colony_id=colony_id,
            tenant_id=user["tenant_id"],
            commit_principal=request.commit_principal,
            commit_co_author=request.commit_co_author,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return GitAttributionConfig(**row)


# ---------------------------------------------------------------------------
# Per-colony GitHub Project (v2) attachment.
#
# Every issue Colony creates against the colony's design monorepo is
# auto-attached to this project so the operator sees roadmap progress
# on a board, not as free-floating issues. The colony-create handler
# refuses to spawn until the operator has picked a project (see
# ``provision_colony``), and the session-create handler threads the
# project's GraphQL node id into ``GitHubCapability.bind`` as
# ``default_project_id`` (see ``routers/sessions.py``).
#
# The picker calls ``GET /discoverable-projects`` to enumerate open
# Projects v2 boards on the colony's monorepo, then ``PUT
# /github-project`` to persist the choice. ``DELETE`` is intentionally
# absent — the operator can re-PUT a different project but cannot
# clear the attachment, because a colony with no project breaks
# session-create.
# ---------------------------------------------------------------------------


class DiscoverableProjectsResponse(BaseModel):
    repo: str = Field(
        description="The owner/name probed (the colony's monorepo).",
    )
    projects: list["DiscoverableProject"] = Field(default_factory=list)
    error: str | None = Field(
        default=None,
        description=(
            "Provider-side error surfaced verbatim when projects "
            "cannot be listed (no design-monorepo on the colony, "
            "App installation missing project read scope, etc.). "
            "``None`` on a clean read even if the list is empty."
        ),
    )


class DiscoverableProject(BaseModel):
    node_id: str = Field(description="GraphQL node id (stored on the colony row).")
    title: str
    number: int | None = None
    url: str | None = None


DiscoverableProjectsResponse.model_rebuild()


class ColonyGitHubProjectConfig(BaseModel):
    node_id: str | None = Field(
        default=None,
        description=(
            "GraphQL node id of the attached Project v2, or ``None`` "
            "when the operator has not picked one yet."
        ),
    )
    title: str | None = Field(
        default=None,
        description=(
            "Human-readable title cached at set time so the picker "
            "can show it without a GraphQL round-trip. Refresh by "
            "re-listing discoverable projects."
        ),
    )


class SetColonyGitHubProjectRequest(BaseModel):
    node_id: str = Field(
        min_length=1,
        description=(
            "GraphQL node id of the Project v2 the operator picked. "
            "Must be one of the entries the ``discoverable-projects`` "
            "route returned for this colony's monorepo."
        ),
    )
    title: str | None = Field(
        default=None,
        description=(
            "Optional human-readable title to cache. Persisted "
            "verbatim — the picker should pass the title it showed "
            "the operator so the UI stays in sync without a "
            "re-discovery round-trip."
        ),
    )


@router.get(
    "/colonies/{colony_id}/discoverable-projects",
    response_model=DiscoverableProjectsResponse,
)
async def list_colony_discoverable_projects(
    colony_id: str,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> DiscoverableProjectsResponse:
    """List open Projects v2 boards on the colony's design monorepo.

    Used by the colony management UI to populate the project picker.
    The colony must already have a design-monorepo URL set (which
    auto-discovery populates at sign-in time for repos that came back
    from the user's App installation); the route surfaces a clean
    error if it doesn't.

    Auth: per-tenant App installation token, minted from the tenant's
    ``github_installation_id`` row. No per-user OAuth involvement —
    the project list is tenant-scoped, not user-scoped.
    """

    db = _get_db_pool(colony)
    tenant_id = user["tenant_id"]

    # The repo we probe is the colony's design monorepo. Without one,
    # there's nothing to enumerate — and the colony shouldn't have
    # been created in the first place per the colony-create gate, so
    # this is a clean 4xx.
    monorepo = await auth_service.get_design_monorepo(
        db, colony_id=colony_id, tenant_id=tenant_id,
    )
    if monorepo is None or not monorepo.get("origin_url"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Colony has no design-monorepo URL. Set the monorepo "
                "first, then re-list projects."
            ),
        )

    # Resolve ``owner/name`` from the design-monorepo URL using the
    # same parser ``DesignProcessCapability`` uses. github.com only —
    # GitLab/Bitbucket projects-equivalents are out of scope.
    from polymathera.colony.design_monorepo.process import (
        parse_owner_repo_from_url,
    )
    repo_full_name = parse_owner_repo_from_url(monorepo["origin_url"])
    if not repo_full_name:
        raise HTTPException(
            status_code=400,
            detail=(
                "Design-monorepo URL is not a github.com URL; "
                "Projects v2 discovery only supports github.com."
            ),
        )
    owner, name = repo_full_name.split("/", 1)

    from ..services.github_projects import (
        list_open_projects_for_repo, _ProjectsLookupFailed,
    )
    try:
        rows = await list_open_projects_for_repo(
            db, tenant_id=tenant_id, owner=owner, name=name,
        )
    except _ProjectsLookupFailed as exc:
        return DiscoverableProjectsResponse(
            repo=repo_full_name, projects=[], error=str(exc),
        )
    return DiscoverableProjectsResponse(
        repo=repo_full_name,
        projects=[DiscoverableProject(**r) for r in rows],
    )


@router.get(
    "/colonies/{colony_id}/github-project",
    response_model=ColonyGitHubProjectConfig,
)
async def get_colony_github_project(
    colony_id: str,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> ColonyGitHubProjectConfig:
    """Return the colony's attached GitHub Project (v2), if any."""

    db = _get_db_pool(colony)
    row = await auth_service.get_colony_github_project(
        db, colony_id=colony_id, tenant_id=user["tenant_id"],
    )
    if row is None:
        return ColonyGitHubProjectConfig()
    return ColonyGitHubProjectConfig(
        node_id=row["node_id"], title=row["title"],
    )


@router.put(
    "/colonies/{colony_id}/github-project",
    response_model=ColonyGitHubProjectConfig,
)
async def set_colony_github_project_route(
    colony_id: str,
    request: SetColonyGitHubProjectRequest,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> ColonyGitHubProjectConfig:
    """Attach a GitHub Project (v2) to the colony.

    The picker should always send the ``node_id`` returned by the
    ``/discoverable-projects`` route plus the ``title`` displayed to
    the operator (cached for the UI). The route does not re-validate
    against GitHub — operators who pass an arbitrary ``node_id``
    will discover the error at issue-create time when the
    ``addProjectV2ItemById`` mutation fails.
    """

    db = _get_db_pool(colony)
    try:
        row = await auth_service.set_colony_github_project(
            db,
            colony_id=colony_id,
            tenant_id=user["tenant_id"],
            node_id=request.node_id,
            title=request.title,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ColonyGitHubProjectConfig(
        node_id=row["node_id"], title=row["title"],
    )
