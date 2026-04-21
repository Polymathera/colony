"""Authentication endpoints: signup, login, logout, refresh, me."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ..auth.middleware import (
    ACCESS_COOKIE,
    REFRESH_COOKIE,
    require_auth,
    get_current_user,
)
from ..auth import service as auth_service
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    user_id: str
    username: str
    tenant_id: str
    default_colony_id: str | None = None
    message: str = ""


class UserInfo(BaseModel):
    user_id: str
    username: str
    tenant_id: str
    created_at: str | None = None
    colonies: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_db_pool(colony: ColonyConnection):
    """Get the PostgreSQL pool from ColonyConnection."""
    pool = colony._db_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return pool


def _set_auth_cookies(response: Response, access_token: str, refresh_token: str) -> None:
    """Set httpOnly auth cookies on the response."""
    response.set_cookie(
        key=ACCESS_COOKIE,
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=False,  # False for local dev (no HTTPS)
        max_age=24 * 60 * 60,  # 1 day
        path="/",
    )
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=refresh_token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=7 * 24 * 60 * 60,  # 7 days
        path="/api/v1/auth",  # Only sent to auth endpoints
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/auth/signup", response_model=AuthResponse)
async def signup(
    request: SignupRequest,
    response: Response,
    colony: ColonyConnection = Depends(get_colony),
) -> AuthResponse:
    """Create a new user account with a default colony."""
    db = _get_db_pool(colony)

    try:
        result = await auth_service.create_user(db, request.username, request.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Generate tokens and set cookies
    access_token = auth_service.create_access_token(
        result["user_id"], result["tenant_id"], request.username,
    )
    refresh_token = auth_service.create_refresh_token(result["user_id"])
    _set_auth_cookies(response, access_token, refresh_token)

    return AuthResponse(
        user_id=result["user_id"],
        username=request.username,
        tenant_id=result["tenant_id"],
        default_colony_id=result["colony_id"],
        message="Account created",
    )


@router.post("/auth/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    response: Response,
    colony: ColonyConnection = Depends(get_colony),
) -> AuthResponse:
    """Authenticate and get access/refresh tokens in cookies."""
    db = _get_db_pool(colony)

    user = await auth_service.authenticate_user(db, request.username, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Get default colony
    default_colony = await auth_service.get_default_colony(db, user["tenant_id"])

    # Generate tokens and set cookies
    access_token = auth_service.create_access_token(
        user["user_id"], user["tenant_id"], user["username"],
    )
    refresh_token = auth_service.create_refresh_token(user["user_id"])
    _set_auth_cookies(response, access_token, refresh_token)

    return AuthResponse(
        user_id=user["user_id"],
        username=user["username"],
        tenant_id=user["tenant_id"],
        default_colony_id=default_colony["colony_id"] if default_colony else None,
        message="Logged in",
    )


@router.post("/auth/logout")
async def logout(response: Response) -> dict[str, str]:
    """Clear auth cookies."""
    response.delete_cookie(ACCESS_COOKIE, path="/")
    response.delete_cookie(REFRESH_COOKIE, path="/api/v1/auth")
    return {"message": "Logged out"}


@router.post("/auth/refresh", response_model=AuthResponse)
async def refresh(
    request: Request,
    response: Response,
    colony: ColonyConnection = Depends(get_colony),
) -> AuthResponse:
    """Get a new access token using the refresh token cookie."""
    db = _get_db_pool(colony)

    refresh_token = request.cookies.get(REFRESH_COOKIE)
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token")

    payload = auth_service.decode_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = await auth_service.get_user_by_id(db, payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    default_colony = await auth_service.get_default_colony(db, user["tenant_id"])

    # New access token
    access_token = auth_service.create_access_token(
        user["user_id"], user["tenant_id"], user["username"],
    )
    response.set_cookie(
        key=ACCESS_COOKIE,
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=30 * 60,
        path="/",
    )

    return AuthResponse(
        user_id=user["user_id"],
        username=user["username"],
        tenant_id=user["tenant_id"],
        default_colony_id=default_colony["colony_id"] if default_colony else None,
        message="Token refreshed",
    )


@router.get("/auth/me", response_model=UserInfo)
async def get_me(
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> UserInfo:
    """Get current user info including colonies."""
    db = _get_db_pool(colony)

    user_info = await auth_service.get_user_by_id(db, user["sub"])
    if not user_info:
        raise HTTPException(status_code=401, detail="User no longer exists. Please sign up again.")

    colonies = await auth_service.list_colonies(db, user_info["tenant_id"])

    return UserInfo(
        user_id=user_info["user_id"],
        username=user_info["username"],
        tenant_id=user_info["tenant_id"],
        created_at=user_info.get("created_at"),
        colonies=colonies,
    )
