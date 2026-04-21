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
