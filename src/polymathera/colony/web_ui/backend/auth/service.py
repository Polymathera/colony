"""Auth service: user management, password hashing, JWT tokens.

Provides the core auth operations used by the API endpoints.
All database access goes through asyncpg pool.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

logger = logging.getLogger(__name__)

# JWT configuration
_JWT_SECRET_KEY = "colony-dev-secret-change-in-production"  # TODO: Read from env
_JWT_ALGORITHM = "HS256"
_ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60  # 1 day
_REFRESH_TOKEN_EXPIRE_DAYS = 7


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Hash a password with bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------

def create_access_token(user_id: str, tenant_id: str, username: str) -> str:
    """Create a short-lived access token."""
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "username": username,
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
# User CRUD
# ---------------------------------------------------------------------------

async def create_user(db_pool, username: str, password: str) -> dict[str, str]:
    """Create a new user with a default colony.

    Returns dict with user_id, tenant_id, colony_id.
    Raises ValueError if username already exists.
    """
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    tenant_id = user_id  # tenant_id = user_id for v1
    colony_id = f"colony_{uuid.uuid4().hex[:12]}"
    pw_hash = hash_password(password)

    async with db_pool.acquire() as conn:
        # Check username uniqueness
        existing = await conn.fetchval(
            "SELECT id FROM users WHERE username = $1", username,
        )
        if existing:
            raise ValueError(f"Username '{username}' already exists")

        async with conn.transaction():
            # Create tenant first — ``users.tenant_id`` and
            # ``colonies.tenant_id`` are both FK'd to ``tenants(id)``
            # after P1 of ``colony/github_identity_fix_plan.md``.
            await conn.execute(
                "INSERT INTO tenants (id, name) VALUES ($1, $2)",
                tenant_id, username,
            )
            # Create user
            await conn.execute(
                "INSERT INTO users (id, username, password_hash, tenant_id) VALUES ($1, $2, $3, $4)",
                user_id, username, pw_hash, tenant_id,
            )
            # Create default colony
            await conn.execute(
                "INSERT INTO colonies (id, name, tenant_id, description, is_default) VALUES ($1, $2, $3, $4, $5)",
                colony_id, "Default", tenant_id, "Auto-created default workspace", True,
            )

    logger.info("Created user %s (tenant=%s) with default colony %s", username, tenant_id, colony_id)
    return {"user_id": user_id, "tenant_id": tenant_id, "colony_id": colony_id}


async def authenticate_user(db_pool, username: str, password: str) -> dict[str, str] | None:
    """Authenticate a user. Returns user info dict or None if invalid."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, username, password_hash, tenant_id FROM users WHERE username = $1",
            username,
        )

    if not row:
        return None

    if not verify_password(password, row["password_hash"]):
        return None

    return {
        "user_id": row["id"],
        "username": row["username"],
        "tenant_id": row["tenant_id"],
    }


async def get_user_by_id(db_pool, user_id: str) -> dict[str, Any] | None:
    """Get user info by ID."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, username, tenant_id, created_at FROM users WHERE id = $1",
            user_id,
        )

    if not row:
        return None

    return {
        "user_id": row["id"],
        "username": row["username"],
        "tenant_id": row["tenant_id"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


# ---------------------------------------------------------------------------
# Colony CRUD
# ---------------------------------------------------------------------------

async def create_colony(
    db_pool, tenant_id: str, name: str, description: str = "",
) -> dict[str, str]:
    """Create a new colony for a tenant."""
    colony_id = f"colony_{uuid.uuid4().hex[:12]}"

    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO colonies (id, name, tenant_id, description) VALUES ($1, $2, $3, $4)",
            colony_id, name, tenant_id, description,
        )

    return {"colony_id": colony_id, "name": name, "tenant_id": tenant_id}


async def list_colonies(db_pool, tenant_id: str) -> list[dict[str, Any]]:
    """List all colonies for a tenant."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, tenant_id, description, is_default, created_at "
            "FROM colonies WHERE tenant_id = $1 ORDER BY is_default DESC, created_at ASC",
            tenant_id,
        )

    return [
        {
            "colony_id": row["id"],
            "name": row["name"],
            "tenant_id": row["tenant_id"],
            "description": row["description"],
            "is_default": row["is_default"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
    ]


async def list_all_colonies(db_pool) -> list[dict[str, Any]]:
    """List every colony across every tenant.

    Cross-tenant variant of :func:`list_colonies`. Used by the
    dashboard's startup walker to bootstrap a system session per
    colony (P8-0). Not exposed via any route — only callable from
    server-side admin code that already has the db_pool handle.
    """

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, tenant_id, description, is_default, "
            "created_at FROM colonies "
            "ORDER BY tenant_id ASC, created_at ASC",
        )

    return [
        {
            "colony_id": row["id"],
            "name": row["name"],
            "tenant_id": row["tenant_id"],
            "description": row["description"],
            "is_default": row["is_default"],
            "created_at": (
                row["created_at"].isoformat()
                if row["created_at"] else None
            ),
        }
        for row in rows
    ]


async def get_tenant_by_installation_id(
    db_pool, *, installation_id: str | int,
) -> dict[str, Any] | None:
    """Return ``{"tenant_id"}`` for the tenant whose
    ``github_installation_id`` column matches.

    Used by the P9 webhook receiver to map an inbound webhook's
    ``installation.id`` payload field to a tenant. Returns ``None``
    when no tenant has installed the App with this id (rejected
    upstream — the webhook is fired for an installation we don't
    serve).

    ``installation_id`` accepts ``int`` or ``str``; we cast to
    ``str`` for the comparison because the column type is ``TEXT``
    (operator pastes the integer in the dashboard panel + we store
    it as a string for forward-compat).
    """

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM tenants WHERE github_installation_id = $1",
            str(installation_id),
        )
    if row is None:
        return None
    return {"tenant_id": row["id"]}


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
    """Return the per-colony git-commit attribution preferences.

    Shape: ``{"commit_principal", "commit_co_author"}``. ``None`` when
    the colony doesn't exist for this tenant. Defaults baked into the
    schema mean both fields are always populated. Per-user identity
    (``git_user_name`` / ``git_user_email``) moved to ``users`` in
    P1 of ``colony/github_identity_fix_plan.md`` (populated by the
    OAuth callback, not by this row).
    """

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
    """Persist the colony's per-commit attribution preferences.

    Returns the newly-stored row. Raises :class:`KeyError` when the
    colony does not exist or belongs to a different tenant. The
    legacy ``git_user_name`` / ``git_user_email`` per-colony fields
    are gone — per-user identity is OAuth-verified on ``users`` (P1
    of ``colony/github_identity_fix_plan.md``); when the operator
    picks ``"user"`` as principal or co-author and no user has
    connected GitHub yet, the resolver emits the commit without the
    user side rather than failing here (the per-user identity is
    only resolvable per-session, not at this colony-config edit
    time).
    """

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


async def get_default_colony(db_pool, tenant_id: str) -> dict[str, Any] | None:
    """Get the default colony for a tenant."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, name, tenant_id, description FROM colonies "
            "WHERE tenant_id = $1 AND is_default = TRUE",
            tenant_id,
        )

    if not row:
        return None

    return {
        "colony_id": row["id"],
        "name": row["name"],
        "tenant_id": row["tenant_id"],
        "description": row["description"],
    }


# ---------------------------------------------------------------------------
# Tenant — per-tenant GitHub App installation id
# ---------------------------------------------------------------------------

async def get_tenant_github_installation(
    db_pool, *, tenant_id: str,
) -> dict[str, Any] | None:
    """Return ``{"installation_id"}`` for the tenant, or ``None`` if
    the tenant doesn't exist. ``installation_id`` is ``None`` until
    a tenant admin configures it (P3 UI)."""

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
    """Persist the tenant's GitHub App installation id. ``None`` clears
    it (the tenant uninstalled the App). Raises :class:`KeyError` when
    the tenant doesn't exist."""

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
# User — per-user OAuth-verified GitHub identity
# ---------------------------------------------------------------------------

async def get_user_github_identity(
    db_pool, *, user_id: str,
) -> dict[str, Any] | None:
    """Return the user's connected GitHub identity, or ``None`` when
    the user hasn't OAuth'd yet (all GitHub-side fields are NULL)."""

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT github_login, github_user_id, github_email, "
            "git_user_name, github_connected_at, github_last_verified_at "
            "FROM users WHERE id = $1",
            user_id,
        )
    if row is None or row["github_login"] is None:
        return None
    return {
        "github_login": row["github_login"],
        "github_user_id": row["github_user_id"],
        "github_email": row["github_email"],
        "git_user_name": row["git_user_name"],
        "github_connected_at": row["github_connected_at"],
        "github_last_verified_at": row["github_last_verified_at"],
    }


async def set_user_github_identity(
    db_pool, *,
    user_id: str,
    github_login: str,
    github_user_id: int,
    github_email: str,
    git_user_name: str | None,
) -> dict[str, Any]:
    """Persist OAuth-verified GitHub identity onto the user row. Called
    by the OAuth callback (P2c). Sets ``github_connected_at`` on the
    first connect, refreshes ``github_last_verified_at`` on every
    call (lets the UI show "last verified N days ago"). Raises
    :class:`KeyError` when the user doesn't exist."""

    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE users SET "
            "  github_login = $1, "
            "  github_user_id = $2, "
            "  github_email = $3, "
            "  git_user_name = $4, "
            "  github_connected_at = COALESCE(github_connected_at, NOW()), "
            "  github_last_verified_at = NOW() "
            "WHERE id = $5",
            github_login, github_user_id, github_email, git_user_name,
            user_id,
        )
    if result == "UPDATE 0":
        raise KeyError(f"user {user_id!r} not found")
    # Return the freshly-persisted shape so the route handler can
    # echo it back to the UI without an extra SELECT.
    fresh = await get_user_github_identity(db_pool, user_id=user_id)
    assert fresh is not None  # we just wrote it
    return fresh


async def delete_user_github_identity(
    db_pool, *, user_id: str,
) -> bool:
    """Disconnect a user's GitHub identity (clear all GitHub-side
    fields). Returns ``True`` if a row was updated; ``False`` if the
    user doesn't exist."""

    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE users SET "
            "  github_login = NULL, "
            "  github_user_id = NULL, "
            "  github_email = NULL, "
            "  git_user_name = NULL, "
            "  github_connected_at = NULL, "
            "  github_last_verified_at = NULL "
            "WHERE id = $1",
            user_id,
        )
    return result != "UPDATE 0"
