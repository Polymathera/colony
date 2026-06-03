"""Authentication endpoints — VCS-OAuth only.

The full sign-in flow:

1. Browser hits ``GET /auth/{provider_id}/sign-in``. We mint a CSRF
   nonce, set a short-lived state cookie, and 302 to the provider's
   authorize URL via ``VcsProvider.build_authorize_url``.
2. The user approves at the provider. Provider 302s back to
   ``GET /auth/{provider_id}/callback?code=...&state=...``.
3. We verify the state-cookie/state-param pair, exchange the code
   for a user-to-server token, fetch the user's identity, and call
   :func:`sync_user_after_signin` (the walker — upserts user +
   tenants + memberships + dev licenses).
4. We mint the access + refresh JWT cookies and 303 the browser to
   the dashboard root.

There is no password sign-in. The two 410-Gone stubs left from PR 2
are gone; ``/auth/signup`` and ``/auth/login`` simply 404 now (no
route handler at all).

Provider-agnostic by design. Today only the GitHub provider is
registered; GitLab / Bitbucket slot in once their adapters land
(plan §7) — no changes here required.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from ..auth import signin_progress
from ..auth.middleware import ACCESS_COOKIE, REFRESH_COOKIE, require_auth
from ..auth import service as auth_service
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection
from ..services.user_tenant_sync import sync_user_after_signin
from polymathera.colony.vcs import OAuthExchangeError, get_provider

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AuthResponse(BaseModel):
    user_id: str
    vcs_login: str
    active_colony_id: str | None = None
    message: str = ""


class UserInfo(BaseModel):
    user_id: str
    vcs_login: str | None = None
    vcs_provider: str | None = None
    active_colony_id: str | None = None
    created_at: str | None = None
    tenants: list[dict[str, Any]] = []
    colonies: list[dict[str, Any]] = []


class SwitchActiveColonyRequest(BaseModel):
    colony_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Short-lived per-provider state cookie that the callback verifies.
# 10-minute TTL matches the user's expected click-through window.
_STATE_COOKIE_PREFIX = "vcs_oauth_state_"
_STATE_COOKIE_MAX_AGE_S = 600


def _get_db_pool(colony: ColonyConnection):
    pool = colony._db_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return pool


def _state_cookie_name(provider_id: str) -> str:
    """Per-provider state cookie. Lets the user start two parallel
    sign-in flows (e.g. GitHub + GitLab) without one's cookie
    clobbering the other's."""
    return f"{_STATE_COOKIE_PREFIX}{provider_id}"


def _callback_path(provider_id: str) -> str:
    """The callback URL path the provider redirects back to. The
    /api/v1 prefix is added by main.py's include_router; the full URL
    the provider sees is ``<host>/api/v1/auth/{provider_id}/callback``."""
    return f"/api/v1/auth/{provider_id}/callback"


def _absolute_callback_url(request: Request, provider_id: str) -> str:
    """Build the exact callback URL the provider was registered with —
    constructed from the live request's scheme + host so multi-host
    deployments work without a static knob."""
    return (
        f"{request.url.scheme}://{request.url.netloc}"
        f"{_callback_path(provider_id)}"
    )


def _set_auth_cookies(
    response: Response, access_token: str, refresh_token: str,
) -> None:
    """Set httpOnly auth cookies on ``response``."""
    response.set_cookie(
        key=ACCESS_COOKIE,
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=False,  # local dev — flip to True in prod
        max_age=24 * 60 * 60,
        path="/",
    )
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=refresh_token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=7 * 24 * 60 * 60,
        path="/api/v1/auth",
    )


# ---------------------------------------------------------------------------
# OAuth sign-in flow
# ---------------------------------------------------------------------------

@router.get("/auth/{provider_id}/sign-in")
async def vcs_sign_in(
    provider_id: str, request: Request,
) -> RedirectResponse:
    """Start the OAuth flow for ``provider_id``. 302s to the
    provider's authorize URL with a CSRF state nonce in a cookie."""

    try:
        provider = get_provider(provider_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                f"VCS provider {provider_id!r} is not registered. "
                f"Operator must set the provider's OAuth credentials "
                f"in .env (see docs/guides/github-app-setup.md §2)."
            ),
        ) from exc

    nonce = secrets.token_urlsafe(32)
    redirect_uri = _absolute_callback_url(request, provider_id)
    authorize_url = provider.build_authorize_url(
        state=nonce, redirect_uri=redirect_uri,
    )
    response = RedirectResponse(url=authorize_url, status_code=302)
    response.set_cookie(
        key=_state_cookie_name(provider_id),
        value=nonce,
        max_age=_STATE_COOKIE_MAX_AGE_S,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path=_callback_path(provider_id),
    )
    logger.info("vcs_sign_in: provider=%s initiated", provider_id)
    return response


@router.get("/auth/{provider_id}/callback")
async def vcs_callback(
    provider_id: str,
    request: Request,
    code: str,
    state: str,
    colony: ColonyConnection = Depends(get_colony),
) -> HTMLResponse:
    """OAuth callback.

    Synchronous prelude (fast — milliseconds): verify CSRF state,
    exchange code, fetch identity, upsert the user row, mint JWT
    cookies. Then spawn the rest of the sign-in walker (tenant
    discovery, repo discovery, colony provisioning, license seeding)
    as a background task, register it with :mod:`signin_progress`,
    and return an HTML loading page. The page polls
    ``GET /api/v1/auth/sign-in-progress/{nonce}`` for live status
    and redirects to ``/`` when the walker reports ``done``.

    Cookies are set on the loading-page response so the eventual
    navigation to ``/`` is already authenticated.
    """

    try:
        provider = get_provider(provider_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    cookie_nonce = request.cookies.get(_state_cookie_name(provider_id))
    if not cookie_nonce or not secrets.compare_digest(cookie_nonce, state):
        raise HTTPException(
            status_code=400,
            detail=f"OAuth state mismatch for provider={provider_id}",
        )

    redirect_uri = _absolute_callback_url(request, provider_id)
    try:
        access_token = await provider.exchange_code_for_token(
            code=code, redirect_uri=redirect_uri,
        )
        identity = await provider.fetch_user_identity(
            access_token=access_token,
        )
    except OAuthExchangeError as exc:
        logger.warning(
            "vcs_callback: provider=%s exchange failed: %s",
            provider_id, exc,
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not identity.primary_email:
        raise HTTPException(
            status_code=400,
            detail=(
                "Your VCS account has no verified primary email. "
                "Add and verify one with your provider and try again."
            ),
        )

    db = _get_db_pool(colony)

    # Upsert the user synchronously — we need the user_id for the
    # JWT we're about to mint. Fast (single SQL roundtrip).
    user_result = await auth_service.upsert_user_from_vcs(
        db,
        vcs_provider=provider.provider_id,
        vcs_user_id=identity.vcs_user_id,
        vcs_login=identity.login,
        vcs_email=identity.primary_email,
        name=identity.name,
    )
    user_id = user_result["user_id"]

    # Mint JWT cookies NOW so the eventual /-redirect is authenticated.
    # tenant_id + colony_id are empty for the initial mint — the
    # background walker will populate ``users.active_colony_id`` and
    # the frontend's post-walker ``/auth/refresh`` call re-issues the
    # JWT with the resolved tenant + colony claims.
    access_jwt = auth_service.create_access_token(
        user_id, identity.login,
        tenant_id="", colony_id="",
    )
    refresh_jwt = auth_service.create_refresh_token(user_id)

    # Track progress under the state nonce — that's the natural
    # per-flow key (already validated for CSRF + present on the
    # browser side so the loading page can pass it in).
    progress = signin_progress.start(state)

    async def _walker_bg() -> None:
        """Background-task wrapper around the walker so the HTML
        loading page can be returned immediately."""
        try:
            await sync_user_after_signin(
                colony,
                provider=provider,
                identity=identity,
                access_token=access_token,
                license_env_value=os.environ.get(
                    "COLONY_DEV_LICENSED_INSTALLATIONS",
                ),
                on_progress=lambda msg: signin_progress.emit(state, msg),
            )
            signin_progress.mark_done(state)
        except OAuthExchangeError as exc:
            signin_progress.mark_done(state, error=str(exc))
        except Exception:  # noqa: BLE001
            logger.exception(
                "vcs_callback: walker crashed for user=%s", user_id,
            )
            signin_progress.mark_done(
                state, error="Sign-in walker crashed — see dashboard logs.",
            )

    task = asyncio.create_task(_walker_bg(), name=f"signin-walker:{state[:8]}")
    signin_progress.register_task(state, task)
    _ = progress  # held reference; not used beyond ensuring start() ran

    response = HTMLResponse(content=_render_signin_loading_page(state))
    _set_auth_cookies(response, access_jwt, refresh_jwt)
    response.delete_cookie(
        _state_cookie_name(provider_id), path=_callback_path(provider_id),
    )
    logger.info(
        "vcs_callback: provider=%s user=%s login=%s walker spawned",
        provider_id, user_id, identity.login,
    )
    return response


def _render_signin_loading_page(nonce: str) -> str:
    """Render the HTML the OAuth callback returns. The page polls
    ``/api/v1/auth/sign-in-progress/{nonce}`` every 600ms, displays
    walker messages live, and on ``done`` POSTs ``/auth/refresh`` to
    re-issue the JWT with the resolved tenant/colony claims, then
    navigates to ``/``.

    Inline HTML on purpose — the dashboard's static-asset bundle is
    not yet built at this point in the flow, and the page is a one-
    shot transient. No external CSS/JS to break the auth flow."""
    # Escape the nonce for safe inline interpolation. The nonce
    # itself is generated via ``secrets.token_urlsafe`` so it's
    # already URL/HTML-safe (alphanumeric + ``-_``), but be explicit.
    safe = "".join(c for c in nonce if c.isalnum() or c in "-_")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Signing in to Colony…</title>
<style>
  body {{
    font-family: ui-monospace, SFMono-Regular, monospace;
    background: #0a0a0a; color: #e5e5e5;
    margin: 0; padding: 48px;
    min-height: 100vh; box-sizing: border-box;
  }}
  h1 {{ font-size: 18px; font-weight: 600; margin: 0 0 24px 0;
        color: #a78bfa; letter-spacing: 0.05em; }}
  #log {{
    background: #141414; border: 1px solid #2a2a2a;
    border-radius: 6px; padding: 16px; min-height: 200px;
    white-space: pre-wrap; font-size: 13px; line-height: 1.6;
  }}
  .err {{ color: #f87171; }}
  .ok  {{ color: #4ade80; }}
  a {{ color: #a78bfa; }}
</style>
</head>
<body>
<h1>SIGNING YOU IN TO COLONY</h1>
<div id="log">Connecting…</div>
<script>
(function () {{
  const nonce = "{safe}";
  const log = document.getElementById("log");
  let seen = 0;
  let pollTimer = null;

  function render(state) {{
    if (!state) return;
    const messages = state.messages || [];
    if (messages.length > seen) {{
      log.textContent = messages.join("\\n");
      seen = messages.length;
    }}
  }}

  async function poll() {{
    try {{
      const resp = await fetch(
        "/api/v1/auth/sign-in-progress/" + encodeURIComponent(nonce),
        {{ credentials: "include" }},
      );
      if (resp.status === 404) {{
        log.textContent += "\\n\\n(no progress state — already redirected?)";
        return;
      }}
      const state = await resp.json();
      render(state);
      if (state.done) {{
        clearInterval(pollTimer);
        if (state.error) {{
          log.textContent += "\\n\\nERROR: " + state.error;
          log.classList.add("err");
          return;
        }}
        // Walker finished — re-issue JWT with the now-populated
        // active_colony_id + tenant_id, then navigate to /.
        try {{
          await fetch("/api/v1/auth/refresh", {{
            method: "POST", credentials: "include",
          }});
        }} catch (e) {{
          // Best-effort — / will still load; the user can switch
          // colonies manually.
        }}
        log.textContent += "\\n\\nRedirecting…";
        window.location.assign("/");
      }}
    }} catch (e) {{
      log.textContent += "\\n\\n(polling error: " + e + ")";
    }}
  }}

  poll();
  pollTimer = setInterval(poll, 600);
}})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class SigninProgressResponse(BaseModel):
    messages: list[str] = []
    done: bool = False
    error: str | None = None
    redirect_to: str = "/"


@router.get(
    "/auth/sign-in-progress/{nonce}",
    response_model=SigninProgressResponse,
)
async def get_sign_in_progress(nonce: str) -> SigninProgressResponse:
    """Read live walker progress for the sign-in flow tracked by
    ``nonce``. The OAuth callback's loading page polls this every
    600ms; on ``done`` it re-issues the JWT (so the new tenant +
    colony claims are reflected) and navigates to ``/``.

    Returns 404 when the nonce isn't tracked (either it never was,
    the TTL purged it, or the dashboard restarted). The loading page
    surfaces that as "no progress state — already redirected?"."""

    state = signin_progress.get(nonce)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown nonce")
    return SigninProgressResponse(
        messages=list(state.messages),
        done=state.done,
        error=state.error,
        redirect_to=state.redirect_to,
    )


async def _resolve_jwt_tenant_for_active_colony(
    db, *, user_id: str, active_colony_id: str | None,
) -> str:
    """Derive the JWT's ``tenant_id`` claim from the user's active
    colony. Walks the user's tenants until it finds one that owns
    ``active_colony_id``. Falls back to the first tenant the user
    belongs to (deterministic — ``list_tenants_for_user`` orders by
    ``joined_at``) when no active colony is set or the active colony
    no longer maps to a tenant the user can see. Empty string when
    the user belongs to no tenants at all."""
    tenants = await auth_service.list_tenants_for_user(db, user_id=user_id)
    if active_colony_id:
        for t in tenants:
            for c in await auth_service.list_colonies(db, t["tenant_id"]):
                if c["colony_id"] == active_colony_id:
                    return str(t["tenant_id"])
    if tenants:
        return str(tenants[0]["tenant_id"])
    return ""


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
    """Mint a new access token from the refresh-token cookie."""
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

    # Always re-derive the JWT context from the user row's CURRENT
    # ``active_colony_id`` — this is also the path the sign-in
    # loading page calls after the background walker finishes, so we
    # must pick up the freshly-populated active colony rather than
    # the empty ``tenant_id`` the callback's initial mint had.
    colony_id = user.get("active_colony_id") or ""
    tenant_id = await _resolve_jwt_tenant_for_active_colony(
        db, user_id=user["user_id"], active_colony_id=colony_id or None,
    )

    access_token = auth_service.create_access_token(
        user["user_id"], user.get("vcs_login") or "",
        tenant_id=tenant_id, colony_id=colony_id,
    )
    response.set_cookie(
        key=ACCESS_COOKIE,
        value=access_token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=24 * 60 * 60,
        path="/",
    )

    return AuthResponse(
        user_id=user["user_id"],
        vcs_login=user.get("vcs_login") or "",
        active_colony_id=user.get("active_colony_id"),
        message="Token refreshed",
    )


@router.get("/auth/me", response_model=UserInfo)
async def get_me(
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> UserInfo:
    """Get current user info: VCS identity, tenants the user belongs
    to, and colonies in those tenants."""
    db = _get_db_pool(colony)

    user_info = await auth_service.get_user_by_id(db, user["sub"])
    if not user_info:
        raise HTTPException(
            status_code=401,
            detail="User no longer exists. Please sign in again.",
        )

    tenants = await auth_service.list_tenants_for_user(
        db, user_id=user_info["user_id"],
    )
    all_colonies: list[dict[str, Any]] = []
    for t in tenants:
        all_colonies.extend(
            await auth_service.list_colonies(db, t["tenant_id"]),
        )

    return UserInfo(
        user_id=user_info["user_id"],
        vcs_login=user_info.get("vcs_login"),
        vcs_provider=user_info.get("vcs_provider"),
        active_colony_id=user_info.get("active_colony_id"),
        created_at=user_info.get("created_at"),
        tenants=tenants,
        colonies=all_colonies,
    )


@router.patch("/auth/me/active-colony", response_model=AuthResponse)
async def switch_active_colony(
    request_body: SwitchActiveColonyRequest,
    response: Response,
    user: dict[str, Any] = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> AuthResponse:
    """Switch the user's active colony. Validates membership, updates
    ``users.active_colony_id``, AND re-issues the JWT so the new
    tenant + colony are reflected in the access token's claims.
    """
    db = _get_db_pool(colony)
    user_id = user["sub"]
    target_colony_id = request_body.colony_id

    # Validate: the colony must belong to a tenant the user is a
    # member of. Otherwise a user could switch into someone else's
    # colony by guessing the id.
    tenants = await auth_service.list_tenants_for_user(db, user_id=user_id)
    target_tenant_id: str | None = None
    for t in tenants:
        for c in await auth_service.list_colonies(db, t["tenant_id"]):
            if c["colony_id"] == target_colony_id:
                target_tenant_id = t["tenant_id"]
                break
        if target_tenant_id is not None:
            break
    if target_tenant_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Colony {target_colony_id!r} not visible to this user.",
        )

    await auth_service.set_active_colony(
        db, user_id=user_id, colony_id=target_colony_id,
    )

    # Re-issue the access JWT with the new context.
    access_token = auth_service.create_access_token(
        user_id, user.get("vcs_login") or "",
        tenant_id=target_tenant_id, colony_id=target_colony_id,
    )
    response.set_cookie(
        key=ACCESS_COOKIE, value=access_token,
        httponly=True, samesite="lax", secure=False,
        max_age=24 * 60 * 60, path="/",
    )

    return AuthResponse(
        user_id=user_id,
        vcs_login=user.get("vcs_login") or "",
        active_colony_id=target_colony_id,
        message="Active colony switched",
    )
