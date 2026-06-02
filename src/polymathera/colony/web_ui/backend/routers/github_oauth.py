"""GitHub user OAuth endpoints — the "Connect GitHub" flow.

The user clicks "Connect GitHub" in the profile UI, which hits
``GET /auth/github/connect``. Backend redirects them to GitHub's
authorisation URL. GitHub redirects back to ``GET /auth/github/callback``
with an authorisation code; backend exchanges the code for a short-
lived user-to-server token, calls ``GET /user`` + ``GET /user/emails``,
persists the verified ``(login, github_user_id, email, name)`` onto the
``users`` row, then discards the token. Colony never acts AS the user
on GitHub.

CSRF: we mint a random nonce per ``connect`` call, set it in a short-
lived ``httpOnly`` cookie (``github_oauth_state``), and embed the same
nonce in GitHub's ``state`` query parameter. The callback verifies the
two match before doing anything else. The connecting user's identity
is verified by the normal Colony JWT (``require_auth``) — the cookie
is sent automatically with the redirect back from GitHub.

See ``colony/github_identity_fix_plan.md`` §3 for the full design.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from ..auth import service as auth_service
from ..auth.github_oauth import (
    OAuthExchangeError,
    build_authorize_url,
    exchange_code_for_token,
    fetch_authenticated_identity,
)
from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()

# Cookie the connect endpoint sets and the callback reads. Short
# lived (10 minutes) since the OAuth round-trip is sub-second on the
# happy path. ``httpOnly`` so client-side JS can't forge it.
_STATE_COOKIE = "github_oauth_state"
_STATE_COOKIE_MAX_AGE_S = 600

# The callback path GitHub redirects to. Must match what the GitHub
# App registration has configured as the callback URL. The /api/v1
# prefix is added by main.py's include_router; the full path GitHub
# sees is ``<colony-host>/api/v1/auth/github/callback``.
_CALLBACK_PATH = "/api/v1/auth/github/callback"


def _get_db_pool(colony: ColonyConnection):
    pool = colony._db_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return pool


def _absolute_callback_url(request: Request) -> str:
    """Build the absolute callback URL the user is redirected back to.
    GitHub requires the exact URL the App registration has on file.
    Constructed from the request's host so multi-host deployments work
    without a static config knob."""

    return f"{request.url.scheme}://{request.url.netloc}{_CALLBACK_PATH}"


@router.get("/auth/github/connect")
async def github_connect(
    request: Request,
    user: dict[str, Any] = Depends(require_auth),
) -> RedirectResponse:
    """Start the OAuth flow. Redirects the browser to GitHub's
    authorisation URL. The user's identity is the JWT-authenticated
    Colony user; CSRF is the cookie/state nonce pair."""

    from polymathera.colony.agents.configs import get_github_auth_config
    gh = await get_github_auth_config()
    if not gh.oauth_client_id or not gh.oauth_client_secret:
        raise HTTPException(
            status_code=503,
            detail=(
                "GitHub OAuth not configured — operator must set "
                "GITHUB_APP_CLIENT_ID and GITHUB_APP_CLIENT_SECRET."
            ),
        )

    nonce = secrets.token_urlsafe(32)
    redirect_uri = _absolute_callback_url(request)
    authorize_url = build_authorize_url(
        client_id=gh.oauth_client_id,
        redirect_uri=redirect_uri,
        state=nonce,
    )
    response = RedirectResponse(url=authorize_url, status_code=302)
    response.set_cookie(
        key=_STATE_COOKIE,
        value=nonce,
        max_age=_STATE_COOKIE_MAX_AGE_S,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",  # ``lax`` so the cookie survives the GitHub redirect
        path=_CALLBACK_PATH,
    )
    logger.info(
        "github_oauth: connect initiated user=%s",
        user.get("sub", "?"),
    )
    return response


@router.get("/auth/github/callback")
async def github_callback(
    request: Request,
    code: str,
    state: str,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> RedirectResponse:
    """OAuth callback. Verifies the state/cookie pair, exchanges the
    authorisation code for a one-shot user token, pulls the verified
    identity from GitHub, writes it to the ``users`` row, and discards
    the token. Redirects the browser back to the dashboard root —
    the connect flow is full-page navigation (``window.location`` in
    :func:`startGitHubConnect`), so a JSON body would render as raw
    text instead of returning the user to the UI. The frontend
    re-mounts on this redirect; its ``useUserGitHubIdentity`` hook
    refetches and shows the connected identity."""

    cookie_nonce = request.cookies.get(_STATE_COOKIE)
    if not cookie_nonce or not secrets.compare_digest(cookie_nonce, state):
        raise HTTPException(
            status_code=400, detail="github_oauth: state mismatch",
        )

    from polymathera.colony.agents.configs import get_github_auth_config
    gh = await get_github_auth_config()
    if not gh.oauth_client_id or not gh.oauth_client_secret:
        raise HTTPException(
            status_code=503,
            detail="GitHub OAuth not configured (callback fired anyway?)",
        )

    redirect_uri = _absolute_callback_url(request)
    try:
        user_token = await exchange_code_for_token(
            client_id=gh.oauth_client_id,
            client_secret=gh.oauth_client_secret,
            code=code,
            redirect_uri=redirect_uri,
        )
        identity = await fetch_authenticated_identity(user_token=user_token)
    except OAuthExchangeError as exc:
        logger.warning("github_oauth: exchange failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not identity.get("login") or identity.get("github_user_id") is None:
        raise HTTPException(
            status_code=502,
            detail="GitHub returned no login / user id",
        )
    if not identity.get("primary_email"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Your GitHub account has no verified primary email. "
                "Add and verify one at https://github.com/settings/emails "
                "and try again."
            ),
        )

    db = _get_db_pool(colony)
    try:
        stored = await auth_service.set_user_github_identity(
            db,
            user_id=user["sub"],
            github_login=identity["login"],
            github_user_id=int(identity["github_user_id"]),
            github_email=identity["primary_email"],
            git_user_name=identity.get("name"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    response = RedirectResponse(url="/", status_code=303)
    # Clear the one-shot state cookie now that it's been consumed.
    response.delete_cookie(_STATE_COOKIE, path=_CALLBACK_PATH)
    logger.info(
        "github_oauth: connected user=%s as github_login=%s",
        user.get("sub", "?"), stored["github_login"],
    )
    return response


@router.get("/users/me/github")
async def get_github_identity(
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Return the caller's connected GitHub identity, or
    ``{"connected": False}`` when the user hasn't run the OAuth
    flow yet. Used by the profile UI to decide between the
    "Connect GitHub" button and the connected-identity display."""

    db = _get_db_pool(colony)
    identity = await auth_service.get_user_github_identity(
        db, user_id=user["sub"],
    )
    if identity is None:
        return {"connected": False}
    return {
        "connected": True,
        "github_login": identity["github_login"],
        "github_user_id": identity["github_user_id"],
        "github_email": identity["github_email"],
        "git_user_name": identity["git_user_name"],
        "github_connected_at": (
            identity["github_connected_at"].isoformat()
            if identity.get("github_connected_at") else None
        ),
        "github_last_verified_at": (
            identity["github_last_verified_at"].isoformat()
            if identity.get("github_last_verified_at") else None
        ),
    }


@router.delete("/users/me/github")
async def disconnect_github(
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, bool]:
    """Disconnect the user's GitHub identity. Clears every GitHub-side
    field on the ``users`` row. Idempotent — disconnecting an already-
    disconnected user is a no-op (returns ``cleared=False``)."""

    db = _get_db_pool(colony)
    cleared = await auth_service.delete_user_github_identity(
        db, user_id=user["sub"],
    )
    return {"cleared": cleared}
