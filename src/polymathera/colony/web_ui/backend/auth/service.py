"""Auth + tenancy service.

SQL helpers for the schema in :mod:`schema`. No password helpers —
sign-in is VCS-OAuth-only (see ``vcs/`` + PR 3 routes). No
``authenticate_user`` for the same reason.

Discipline: routers must not call ``create_colony`` directly — go
through :func:`services.colony_lifecycle.provision_colony` so the
per-colony system-session bootstrap runs.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

logger = logging.getLogger(__name__)

# JWT configuration
_JWT_SECRET_KEY = "colony-dev-secret-change-in-production"  # TODO: Read from env
_JWT_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60  # 1 day
_REFRESH_TOKEN_EXPIRE_DAYS = 7


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------

def create_access_token(
    user_id: str,
    vcs_login: str,
    *,
    tenant_id: str = "",
    colony_id: str = "",
) -> str:
    """Create a short-lived access token.

    ``tenant_id`` + ``colony_id`` are the user's CURRENT active
    context. They're carried in the JWT so per-request handlers and
    the middleware's ``execution_context`` can scope without a DB
    round-trip. When the user switches colonies in the UI, the
    backend re-issues the JWT with the new pair.

    Empty strings mean "no active context" — happens between sign-in
    and the user's first colony landing (PR 4 discovery). Routes that
    require a tenant return 4xx in that window.
    """
    payload = {
        "sub": user_id,
        "vcs_login": vcs_login,
        "tenant_id": tenant_id,
        "colony_id": colony_id,
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=_ACCESS_TOKEN_EXPIRE_MINUTES),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _JWT_SECRET_KEY, algorithm=_JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """Create a long-lived refresh token."""
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": datetime.now(timezone.utc) + timedelta(days=_REFRESH_TOKEN_EXPIRE_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _JWT_SECRET_KEY, algorithm=_JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any] | None:
    """Decode and validate a JWT token. Returns None if invalid/expired."""
    try:
        return jwt.decode(token, _JWT_SECRET_KEY, algorithms=[_JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

async def upsert_user_from_vcs(
    db_pool,
    *,
    vcs_provider: str,
    vcs_user_id: str,
    vcs_login: str,
    vcs_email: str | None,
    name: str | None,
) -> dict[str, Any]:
    """Create-or-update a user from a verified OAuth identity. The
    natural key is ``(vcs_provider, vcs_user_id)`` (see
    ``users_vcs_identity_uq``); re-sign-in is idempotent.

    Returns ``{"user_id", "is_new"}``. ``is_new`` is True iff this
    call inserted a fresh row (signup vs. login distinction).
    """

    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM users "
            "WHERE vcs_provider = $1 AND vcs_user_id = $2",
            vcs_provider, vcs_user_id,
        )
        if existing is not None:
            await conn.execute(
                "UPDATE users SET "
                "  vcs_login = $1, "
                "  vcs_email = $2, "
                "  git_user_name = COALESCE($3, git_user_name), "
                "  vcs_last_verified_at = NOW() "
                "WHERE id = $4",
                vcs_login, vcs_email, name, existing["id"],
            )
            return {"user_id": existing["id"], "is_new": False}

        user_id = f"user_{uuid.uuid4().hex[:12]}"
        await conn.execute(
            "INSERT INTO users ("
            "  id, vcs_provider, vcs_user_id, vcs_login, vcs_email, "
            "  git_user_name, vcs_connected_at, vcs_last_verified_at"
            ") VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW())",
            user_id, vcs_provider, vcs_user_id, vcs_login, vcs_email, name,
        )
        logger.info("Created user %s (vcs=%s login=%s)",
                    user_id, vcs_provider, vcs_login)
        return {"user_id": user_id, "is_new": True}


async def get_user_by_id(db_pool, user_id: str) -> dict[str, Any] | None:
    """Get user info by ID. Returns the VCS identity fields used by
    the dashboard's ``/auth/me``-style endpoints."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, vcs_provider, vcs_login, vcs_email, "
            "git_user_name, active_colony_id, created_at "
            "FROM users WHERE id = $1",
            user_id,
        )

    if not row:
        return None

    return {
        "user_id": row["id"],
        "vcs_provider": row["vcs_provider"],
        "vcs_login": row["vcs_login"],
        "vcs_email": row["vcs_email"],
        "git_user_name": row["git_user_name"],
        "active_colony_id": row["active_colony_id"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


async def set_active_colony(
    db_pool, *, user_id: str, colony_id: str | None,
) -> None:
    """Persist the user's active colony (last-switched or
    last-discovered). ``None`` clears it. The session-create handler
    reads this to default new sessions to the user's last context;
    the UI's colony-picker PATCHes it on every switch."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET active_colony_id = $1 WHERE id = $2",
            colony_id, user_id,
        )


# ---------------------------------------------------------------------------
# user_tenants — many-to-many membership
# ---------------------------------------------------------------------------

async def upsert_user_tenant(
    db_pool, *, user_id: str, tenant_id: str, role: str,
) -> None:
    """Insert or refresh a user_tenants row. ``role`` is 'member' or
    'admin' (CHECK-constrained). Refreshes ``last_verified_at`` on
    every call so the GC pass can spot stale memberships."""
    if role not in ("member", "admin"):
        raise ValueError(f"role must be 'member' or 'admin', got {role!r}")
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO user_tenants (user_id, tenant_id, role) "
            "VALUES ($1, $2, $3) "
            "ON CONFLICT (user_id, tenant_id) DO UPDATE SET "
            "  role = EXCLUDED.role, "
            "  last_verified_at = NOW()",
            user_id, tenant_id, role,
        )


async def list_tenants_for_user(
    db_pool, *, user_id: str,
) -> list[dict[str, Any]]:
    """List every tenant the user is a member of, with their role."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT t.id AS tenant_id, t.name, t.vcs_provider, "
            "  t.vcs_org_login, ut.role "
            "FROM user_tenants ut "
            "JOIN tenants t ON t.id = ut.tenant_id "
            "WHERE ut.user_id = $1 "
            "ORDER BY ut.joined_at ASC",
            user_id,
        )
    return [
        {
            "tenant_id": row["tenant_id"],
            "name": row["name"],
            "vcs_provider": row["vcs_provider"],
            "vcs_org_login": row["vcs_org_login"],
            "role": row["role"],
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Tenants
# ---------------------------------------------------------------------------

async def upsert_tenant_from_vcs(
    db_pool,
    *,
    vcs_provider: str,
    vcs_org_id: str,
    vcs_org_login: str,
    name: str | None = None,
    github_installation_id: str | None = None,
) -> dict[str, Any]:
    """Create-or-update a tenant from a discovered VCS org. The
    natural key is ``(vcs_provider, vcs_org_id)``.

    Returns ``{"tenant_id"}``.
    """
    async with db_pool.acquire() as conn:
        existing = await conn.fetchrow(
            "SELECT id FROM tenants "
            "WHERE vcs_provider = $1 AND vcs_org_id = $2",
            vcs_provider, vcs_org_id,
        )
        if existing is not None:
            await conn.execute(
                "UPDATE tenants SET "
                "  vcs_org_login = $1, "
                "  name = COALESCE($2, name), "
                "  github_installation_id = COALESCE($3, github_installation_id) "
                "WHERE id = $4",
                vcs_org_login, name, github_installation_id, existing["id"],
            )
            return {"tenant_id": existing["id"]}

        tenant_id = f"tenant_{uuid.uuid4().hex[:12]}"
        await conn.execute(
            "INSERT INTO tenants ("
            "  id, name, vcs_provider, vcs_org_id, vcs_org_login, "
            "  github_installation_id"
            ") VALUES ($1, $2, $3, $4, $5, $6)",
            tenant_id, name or vcs_org_login, vcs_provider,
            vcs_org_id, vcs_org_login, github_installation_id,
        )
        return {"tenant_id": tenant_id}


async def get_tenant_by_installation_id(
    db_pool, *, installation_id: str | int,
) -> dict[str, Any] | None:
    """Return ``{"tenant_id"}`` for the tenant whose
    ``github_installation_id`` column matches. Used by the P9
    webhook receiver."""

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM tenants WHERE github_installation_id = $1",
            str(installation_id),
        )
    if row is None:
        return None
    return {"tenant_id": row["id"]}


async def get_tenant_github_installation(
    db_pool, *, tenant_id: str,
) -> dict[str, Any] | None:
    """Return ``{"installation_id"}`` for the tenant, or ``None`` if
    the tenant doesn't exist."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT github_installation_id FROM tenants WHERE id = $1",
            tenant_id,
        )
    if row is None:
        return None
    return {"installation_id": row["github_installation_id"]}


async def set_tenant_github_installation(
    db_pool, *, tenant_id: str, installation_id: str | None,
) -> dict[str, Any]:
    """Persist the tenant's GitHub App installation id. Raises
    :class:`KeyError` when the tenant doesn't exist."""
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE tenants SET github_installation_id = $1 WHERE id = $2",
            installation_id or None,
            tenant_id,
        )
    if result == "UPDATE 0":
        raise KeyError(f"tenant {tenant_id!r} not found")
    return {"installation_id": installation_id or None}


# ---------------------------------------------------------------------------
# Tenant — encrypted bot token (GitLab / Bitbucket — see plan §5.2)
# ---------------------------------------------------------------------------

async def set_tenant_bot_token(
    db_pool,
    *,
    tenant_id: str,
    plaintext_token: str | None,
    expires_at: Any = None,
) -> None:
    """Encrypt + persist a long-lived bot credential for a tenant.

    Used for GitLab Group Access Tokens / Bitbucket workspace API
    tokens — operator-managed credentials that aren't auto-minted from
    a deploy-wide secret (unlike GitHub installation tokens).

    ``plaintext_token=None`` clears the column (operator rotation /
    revocation). ``expires_at`` is the provider-side expiry the agent
    process pre-emptively rotates before; ``None`` means "no expiry"
    (some GitLab GATs are configured that way).

    Raises :class:`KeyError` if the tenant row doesn't exist.
    """
    from .secrets import encrypt_value

    encrypted = (
        encrypt_value(plaintext_token) if plaintext_token else None
    )
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE tenants SET "
            "  bot_token_encrypted = $1, "
            "  bot_token_expires_at = $2 "
            "WHERE id = $3",
            encrypted, expires_at, tenant_id,
        )
    if result == "UPDATE 0":
        raise KeyError(f"tenant {tenant_id!r} not found")


async def get_tenant_bot_token(
    db_pool, *, tenant_id: str,
) -> dict[str, Any] | None:
    """Fetch + decrypt a tenant's bot token.

    Returns ``{"token", "expires_at"}`` on success, ``None`` when the
    tenant has no token on file (or doesn't exist).

    Decryption failures surface as :class:`auth.secrets.SecretDecryptError`
    — caller decides whether to log and skip (treat as "no token") or
    raise (treat as a hard failure).
    """
    from .secrets import decrypt_value

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT bot_token_encrypted, bot_token_expires_at "
            "FROM tenants WHERE id = $1",
            tenant_id,
        )
    if row is None or row["bot_token_encrypted"] is None:
        return None
    return {
        "token": decrypt_value(row["bot_token_encrypted"]),
        "expires_at": row["bot_token_expires_at"],
    }


# ---------------------------------------------------------------------------
# Colonies
# ---------------------------------------------------------------------------

async def create_colony(
    db_pool,
    tenant_id: str,
    name: str,
    description: str = "",
    *,
    vcs_repo_id: str | None = None,
    vcs_repo_full_name: str | None = None,
    default_branch: str | None = None,
) -> dict[str, str]:
    """SQL-layer colony insert. Do NOT call directly from routers —
    go through :func:`services.colony_lifecycle.provision_colony` so
    the system-session bootstrap runs.
    """
    colony_id = f"colony_{uuid.uuid4().hex[:12]}"

    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO colonies ("
            "  id, name, tenant_id, description, "
            "  vcs_repo_id, vcs_repo_full_name, default_branch"
            ") VALUES ($1, $2, $3, $4, $5, $6, $7)",
            colony_id, name, tenant_id, description,
            vcs_repo_id, vcs_repo_full_name, default_branch,
        )

    return {"colony_id": colony_id, "name": name, "tenant_id": tenant_id}


async def list_colonies(db_pool, tenant_id: str) -> list[dict[str, Any]]:
    """List all colonies for a tenant."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, tenant_id, description, "
            "vcs_repo_id, vcs_repo_full_name, default_branch, created_at "
            "FROM colonies WHERE tenant_id = $1 ORDER BY created_at ASC",
            tenant_id,
        )

    return [
        {
            "colony_id": row["id"],
            "name": row["name"],
            "tenant_id": row["tenant_id"],
            "description": row["description"],
            "vcs_repo_id": row["vcs_repo_id"],
            "vcs_repo_full_name": row["vcs_repo_full_name"],
            "default_branch": row["default_branch"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
    ]


async def list_all_colonies(db_pool) -> list[dict[str, Any]]:
    """Cross-tenant variant of :func:`list_colonies`. Used by the
    dashboard's startup walker to bootstrap a system session per
    colony (P8-0)."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, tenant_id, description, "
            "vcs_repo_id, vcs_repo_full_name, default_branch, created_at "
            "FROM colonies ORDER BY tenant_id ASC, created_at ASC",
        )

    return [
        {
            "colony_id": row["id"],
            "name": row["name"],
            "tenant_id": row["tenant_id"],
            "description": row["description"],
            "vcs_repo_id": row["vcs_repo_id"],
            "vcs_repo_full_name": row["vcs_repo_full_name"],
            "default_branch": row["default_branch"],
            "created_at": (
                row["created_at"].isoformat()
                if row["created_at"] else None
            ),
        }
        for row in rows
    ]


async def upsert_tenant_repo(
    db_pool,
    *,
    tenant_id: str,
    vcs_repo_id: str,
    vcs_repo_full_name: str,
    default_branch: str,
    user_permission: str,
    has_colony_marker: bool,
) -> None:
    """Cache a repo we saw during the sign-in walker. Idempotent on
    (tenant_id, vcs_repo_id); refreshes ``last_seen_at`` + the
    ``has_colony_marker`` flag on every call so a `.colony/` added
    later flips the flag on the next sign-in."""
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO tenant_repos ("
            "  tenant_id, vcs_repo_id, vcs_repo_full_name, "
            "  default_branch, user_permission, has_colony_marker, "
            "  last_seen_at"
            ") VALUES ($1, $2, $3, $4, $5, $6, NOW()) "
            "ON CONFLICT (tenant_id, vcs_repo_id) DO UPDATE SET "
            "  vcs_repo_full_name = EXCLUDED.vcs_repo_full_name, "
            "  default_branch = EXCLUDED.default_branch, "
            "  user_permission = EXCLUDED.user_permission, "
            "  has_colony_marker = EXCLUDED.has_colony_marker, "
            "  last_seen_at = NOW()",
            tenant_id, vcs_repo_id, vcs_repo_full_name,
            default_branch, user_permission, has_colony_marker,
        )


async def list_tenant_repos_for_tenant(
    db_pool, *, tenant_id: str,
) -> list[dict[str, Any]]:
    """Return every cached repo for ``tenant_id``. Powers the
    discoverable-repos dropdown on the "+ New colony" form +
    per-colony "Design monorepo" picker."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT vcs_repo_id, vcs_repo_full_name, default_branch, "
            "user_permission, has_colony_marker, last_seen_at "
            "FROM tenant_repos WHERE tenant_id = $1 "
            "ORDER BY vcs_repo_full_name ASC",
            tenant_id,
        )
    return [
        {
            "vcs_repo_id": row["vcs_repo_id"],
            "vcs_repo_full_name": row["vcs_repo_full_name"],
            "default_branch": row["default_branch"],
            "user_permission": row["user_permission"],
            "has_colony_marker": row["has_colony_marker"],
        }
        for row in rows
    ]


async def any_colony_exists_for_repo(
    db_pool, *, tenant_id: str, vcs_repo_id: str,
) -> bool:
    """Discovery-walker gate (PR 4 will call this): does this tenant
    already have at least one colony pointed at this repo? Application-
    level uniqueness — there's intentionally no SQL UNIQUE on
    ``(tenant_id, vcs_repo_id)`` because users may explicitly create
    multiple colonies on the same repo."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM colonies "
            "WHERE tenant_id = $1 AND vcs_repo_id = $2 LIMIT 1",
            tenant_id, vcs_repo_id,
        )
    return row is not None


async def get_design_monorepo(
    db_pool, *, colony_id: str, tenant_id: str,
) -> dict[str, Any] | None:
    """Return the per-colony design-monorepo configuration row.

    Returns ``None`` for an unconfigured colony (``design_monorepo_url``
    NULL). The ``branch`` / ``commit`` columns always have defaults so
    callers do not need to coalesce.
    """

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT design_monorepo_url, design_monorepo_branch, "
            "design_monorepo_commit FROM colonies "
            "WHERE id = $1 AND tenant_id = $2",
            colony_id, tenant_id,
        )
    if row is None or row["design_monorepo_url"] is None:
        return None
    return {
        "origin_url": row["design_monorepo_url"],
        "branch": row["design_monorepo_branch"],
        "commit": row["design_monorepo_commit"],
    }


async def set_design_monorepo(
    db_pool, *,
    colony_id: str,
    tenant_id: str,
    origin_url: str,
    branch: str = "main",
    commit: str = "HEAD",
) -> dict[str, Any]:
    """Persist the colony's design-monorepo URL/branch/commit. Returns
    the newly-stored row. Raises :class:`KeyError` when the colony does
    not exist or belongs to a different tenant."""

    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE colonies SET design_monorepo_url = $1, "
            "design_monorepo_branch = $2, design_monorepo_commit = $3 "
            "WHERE id = $4 AND tenant_id = $5",
            origin_url, branch, commit, colony_id, tenant_id,
        )
    if result == "UPDATE 0":
        raise KeyError(f"colony {colony_id!r} not found for tenant {tenant_id!r}")
    return {"origin_url": origin_url, "branch": branch, "commit": commit}


async def get_git_attribution(
    db_pool, *, colony_id: str, tenant_id: str,
) -> dict[str, Any] | None:
    """Return the per-colony git-commit attribution preferences."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT commit_principal, commit_co_author FROM colonies "
            "WHERE id = $1 AND tenant_id = $2",
            colony_id, tenant_id,
        )
    if row is None:
        return None
    return {
        "commit_principal": row["commit_principal"],
        "commit_co_author": row["commit_co_author"],
    }


async def set_git_attribution(
    db_pool, *,
    colony_id: str,
    tenant_id: str,
    commit_principal: str,
    commit_co_author: str | None,
) -> dict[str, Any]:
    """Persist the colony's per-commit attribution preferences."""
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE colonies SET commit_principal = $1, "
            "commit_co_author = $2 "
            "WHERE id = $3 AND tenant_id = $4",
            commit_principal,
            commit_co_author or None,
            colony_id,
            tenant_id,
        )
    if result == "UPDATE 0":
        raise KeyError(f"colony {colony_id!r} not found for tenant {tenant_id!r}")
    return {
        "commit_principal": commit_principal,
        "commit_co_author": commit_co_author or None,
    }


async def get_colony_github_project(
    db_pool, *, colony_id: str, tenant_id: str,
) -> dict[str, str] | None:
    """Return ``{node_id, title}`` for the colony's attached GitHub
    Project (v2), or ``None`` when the operator hasn't picked one.

    The Project's GraphQL node id is what every newly-created issue
    gets stamped with (via
    ``GitHubCapability.create_issue(project_id=...)``); the title is
    a UI cache so the picker can show the human-readable name without
    a GraphQL round-trip on every render.
    """

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT github_project_node_id, github_project_title "
            "FROM colonies WHERE id = $1 AND tenant_id = $2",
            colony_id, tenant_id,
        )
    if row is None or row["github_project_node_id"] is None:
        return None
    return {
        "node_id": row["github_project_node_id"],
        "title": row["github_project_title"] or "",
    }


async def set_colony_github_project(
    db_pool, *,
    colony_id: str,
    tenant_id: str,
    node_id: str | None,
    title: str | None,
) -> dict[str, str | None]:
    """Persist the colony's attached GitHub Project. Pass
    ``node_id=None`` (with ``title=None``) to clear the attachment.

    Raises :class:`KeyError` when the colony does not exist or
    belongs to a different tenant.
    """

    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE colonies SET github_project_node_id = $1, "
            "github_project_title = $2 "
            "WHERE id = $3 AND tenant_id = $4",
            node_id, title, colony_id, tenant_id,
        )
    if result == "UPDATE 0":
        raise KeyError(
            f"colony {colony_id!r} not found for tenant {tenant_id!r}",
        )
    return {"node_id": node_id, "title": title}


# ---------------------------------------------------------------------------
# User — per-user OAuth-verified VCS identity
# ---------------------------------------------------------------------------
#
# Returned dict keys still use ``github_*`` prefixes for compatibility
# with the agent-metadata threader (``routers/sessions.py::
# _resolve_github_identity``) and the frontend hook. PR 3 will rename
# the keys to ``vcs_*`` alongside the signup-flow rewrite.

async def get_user_github_identity(
    db_pool, *, user_id: str,
) -> dict[str, Any] | None:
    """Return the user's VCS-verified identity, or ``None`` when the
    user has no VCS identity on file."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT vcs_login, vcs_user_id, vcs_email, "
            "git_user_name, vcs_connected_at, vcs_last_verified_at "
            "FROM users WHERE id = $1",
            user_id,
        )
    if row is None or row["vcs_login"] is None:
        return None
    # Returned-key shapes preserved for PR 2 — agent metadata
    # threader and frontend still read these key names. PR 3 renames.
    return {
        "github_login": row["vcs_login"],
        "github_user_id": row["vcs_user_id"],
        "github_email": row["vcs_email"],
        "git_user_name": row["git_user_name"],
        "github_connected_at": row["vcs_connected_at"],
        "github_last_verified_at": row["vcs_last_verified_at"],
    }


async def set_user_github_identity(
    db_pool, *,
    user_id: str,
    github_login: str,
    github_user_id: int | str,
    github_email: str,
    git_user_name: str | None,
) -> dict[str, Any]:
    """Persist OAuth-verified VCS identity onto the user row. ``vcs_user_id``
    is TEXT in the new schema — accept ``int`` for back-compat with the
    existing ``Connect GitHub`` callback (PR 3 will use the new
    ``upsert_user_from_vcs`` directly + drop this back-compat shim)."""
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE users SET "
            "  vcs_login = $1, "
            "  vcs_user_id = $2, "
            "  vcs_email = $3, "
            "  git_user_name = $4, "
            "  vcs_connected_at = COALESCE(vcs_connected_at, NOW()), "
            "  vcs_last_verified_at = NOW() "
            "WHERE id = $5",
            github_login, str(github_user_id), github_email, git_user_name,
            user_id,
        )
    if result == "UPDATE 0":
        raise KeyError(f"user {user_id!r} not found")
    fresh = await get_user_github_identity(db_pool, user_id=user_id)
    assert fresh is not None  # we just wrote it
    return fresh


async def delete_user_github_identity(
    db_pool, *, user_id: str,
) -> bool:
    """Disconnect a user's VCS identity (clear all VCS-side fields).
    Returns ``True`` if a row was updated; ``False`` if the user
    doesn't exist."""

    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE users SET "
            "  vcs_login = NULL, "
            "  vcs_user_id = NULL, "
            "  vcs_email = NULL, "
            "  git_user_name = NULL, "
            "  vcs_connected_at = NULL, "
            "  vcs_last_verified_at = NULL "
            "WHERE id = $1",
            user_id,
        )
    return result != "UPDATE 0"
