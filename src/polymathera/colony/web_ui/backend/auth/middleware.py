"""Auth middleware: JWT cookie -> ExecutionContext.

Extracts the access token from the `colony_access` httpOnly cookie,
decodes it, and sets a Ring.USER execution context with the user's
tenant_id. If no cookie or invalid token, the request proceeds
without user context (anonymous). Protected endpoints check for
the user via the `require_auth` dependency.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request, HTTPException, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .service import decode_token

logger = logging.getLogger(__name__)

# Cookie names
ACCESS_COOKIE = "colony_access"
REFRESH_COOKIE = "colony_refresh"

# Paths that don't require authentication
PUBLIC_PATHS = {
    "/api/v1/auth/login",
    "/api/v1/auth/signup",
    "/api/v1/auth/refresh",
    "/api/v1/infra/status",
    "/api/v1/infra/health",
}


class AuthMiddleware(BaseHTTPMiddleware):
    """Extract JWT from cookie and set execution context.

    If a valid access token is found, sets Ring.USER context with
    the user's tenant_id. If not, sets Ring.KERNEL for public/API
    endpoints (backward compat with dashboard monitoring endpoints).

    The actual enforcement (401 for protected endpoints) is done by
    the `require_auth` FastAPI dependency, not this middleware.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        from polymathera.colony.distributed.ray_utils.serving.context import (
            Ring, execution_context,
        )

        # Try to extract user from access token cookie
        token = request.cookies.get(ACCESS_COOKIE)
        user_payload = decode_token(token) if token else None

        if user_payload and user_payload.get("type") == "access":
            # Authenticated request — Ring.USER with tenant context
            tenant_id = user_payload.get("tenant_id", "")
            # colony_id comes from the request (header or query param),
            # not from the token — users can switch colonies
            colony_id = (
                request.headers.get("X-Colony-Id")
                or request.query_params.get("colony_id")
                or None
            )

            with execution_context(
                ring=Ring.USER,
                colony_id=colony_id,
                tenant_id=tenant_id,
                origin="dashboard",
            ):
                # Stash user info on request.state for dependencies
                request.state.user = user_payload
                return await call_next(request)
        else:
            # Anonymous request — Ring.KERNEL for monitoring endpoints
            with execution_context(ring=Ring.KERNEL, origin="dashboard"):
                request.state.user = None
                return await call_next(request)


def get_current_user(request: Request) -> dict[str, Any] | None:
    """Get the current authenticated user from request state, or None."""
    return getattr(request.state, "user", None)


def require_auth(request: Request) -> dict[str, Any]:
    """FastAPI dependency that requires authentication.

    Use in endpoint signatures:
        @router.post("/sessions/")
        async def create_session(user = Depends(require_auth)):
            tenant_id = user["tenant_id"]
    """
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user
