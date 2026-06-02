"""Per-agent git credential management — mints GitHub App installation
tokens and writes them to a file the shell credential helper reads.

The container's system git config has a credential helper that invokes
``/usr/local/bin/colony-git-credentials`` (see
``cli/deploy/docker/scripts/colony-git-credentials.sh``); that script
just prints the token from the file. This module is the writer side:
it uses the App's JWT + the tenant's installation_id to mint a token
+ writes it to disk + starts a background task that refreshes the
token every 50 minutes (well inside GitHub's 60-minute TTL).

Lifecycle:

- ``ensure_git_credentials_from_agent_metadata(agent_metadata)`` is
  the public entry point. Idempotent across calls — first caller
  starts the refresh task, subsequent calls are no-ops as long as the
  installation_id matches. Wired into
  :meth:`DesignMonorepoCapabilityBase.initialize` so every push-
  capable capability mount triggers it; the singleton manager
  collapses repeat calls.

- ``GitCredentialsManager.reset_for_tests`` lets tests tear down
  the singleton + background task without dragging in pytest-asyncio
  shutdown plumbing.

When ``installation_id`` is missing (tenant hasn't installed the App)
or the deploy-wide App credentials are absent (``GITHUB_APP_ID`` /
``GITHUB_PRIVATE_KEY_PEM`` unset), the ensure path returns silently
without writing anything. The shell helper sees an empty/missing
file and outputs nothing; git surfaces its own "Authentication failed"
error which ``_classify_git_clone_error`` reshapes into a
``GitAuthError`` with an actionable message.

P9 of ``colony/github_identity_fix_plan.md``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

from ..agents.patterns.capabilities._github.auth import (
    GitHubAppAuth,
    TokenCache,
)

logger = logging.getLogger(__name__)


_DEFAULT_CREDENTIALS_FILE = "/tmp/colony-git-credentials"

# GitHub installation tokens last 60 minutes; refresh at 50 to leave
# headroom for clock skew + a missed refresh.
_REFRESH_INTERVAL_S = 50 * 60


def _credentials_path() -> Path:
    """Resolve the path the helper script + the writer agree on. Env
    override (``COLONY_GIT_CREDENTIALS_FILE``) lets dev / tests point
    at a temp directory without changing code."""

    return Path(
        os.environ.get(
            "COLONY_GIT_CREDENTIALS_FILE", _DEFAULT_CREDENTIALS_FILE,
        ),
    )


def write_credentials_file(path: str | Path, token: str) -> None:
    """Atomically write ``token`` to ``path``.

    The write goes to a sibling tempfile + ``os.replace`` so the shell
    helper never reads a half-written token if a refresh races with a
    git invocation. The file is chmod 600 — it carries a live
    installation token.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(token)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class GitCredentialsManager:
    """Process-wide singleton owning the credential-file write loop.

    Each agent process has at most one installation_id in scope
    (tenant scoping is per-process in v1); the singleton mints once
    on first ``ensure`` + refreshes in the background until the
    process exits or ``stop`` is called.
    """

    _instance: "GitCredentialsManager | None" = None
    _instance_lock: asyncio.Lock | None = None

    @classmethod
    def get(cls) -> "GitCredentialsManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._token_cache: TokenCache | None = None
        self._installation_id: str | None = None
        self._refresh_task: asyncio.Task | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._credentials_path: Path = _credentials_path()
        self._lock = asyncio.Lock()

    async def ensure(
        self, *,
        app_id: str,
        private_key_pem: str,
        installation_id: str,
        refresh_interval_s: int = _REFRESH_INTERVAL_S,
    ) -> None:
        """Idempotent: starts the mint+refresh task on first call.

        If a subsequent call passes a different ``installation_id``
        (tenant re-installed the App, configured a new installation),
        the manager tears down the existing task + restarts on the new
        installation. Same installation = no-op.
        """

        async with self._lock:
            if (
                self._installation_id == installation_id
                and self._refresh_task is not None
                and not self._refresh_task.done()
            ):
                return  # already running for this installation
            await self._stop_locked()
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0),
            )
            auth = GitHubAppAuth(
                app_id=app_id, private_key_pem=private_key_pem,
            )
            self._token_cache = TokenCache(
                app_auth=auth,
                installation_id=installation_id,
                client=self._http_client,
            )
            self._installation_id = installation_id
            # Synchronous mint + write so the credential file is
            # present before any push runs.
            await self._mint_and_write()
            self._refresh_task = asyncio.create_task(
                self._refresh_loop(refresh_interval_s),
            )

    async def _mint_and_write(self) -> None:
        assert self._token_cache is not None
        token = await self._token_cache.get(force_refresh=True)
        write_credentials_file(self._credentials_path, token)
        logger.debug(
            "git_credentials: minted + wrote installation token "
            "for installation_id=%s",
            self._installation_id,
        )

    async def _refresh_loop(self, interval_s: int) -> None:
        try:
            while True:
                await asyncio.sleep(interval_s)
                try:
                    await self._mint_and_write()
                except Exception:
                    logger.exception(
                        "git_credentials: refresh failed; will retry "
                        "at the next interval"
                    )
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        async with self._lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except (asyncio.CancelledError, Exception):
                pass
            self._refresh_task = None
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:  # noqa: BLE001 — defensive
                pass
            self._http_client = None
        self._token_cache = None
        self._installation_id = None
        # Leave the file in place — leftover tokens expire naturally,
        # and unlinking races with concurrent readers (the shell
        # helper invoked by git during a push).

    @classmethod
    async def reset_for_tests(cls) -> None:
        """Stop the running singleton and clear it. Tests call this in
        teardown to avoid task leakage across the test session."""

        if cls._instance is not None:
            await cls._instance.stop()
        cls._instance = None


async def ensure_git_credentials_for_installation(
    installation_id: str | int | None,
) -> None:
    """Idempotently start the credential mint+refresh task for a
    specific tenant installation id.

    The non-metadata-bound primitive. Used by:

    - :func:`ensure_git_credentials_from_agent_metadata` — the
      agent-side entrypoint that pulls ``installation_id`` out of
      ``agent_metadata.parameters["github_identity"]``.
    - Dashboard routes that clone the design monorepo on behalf of
      the user (the dashboard process has no agent metadata; it
      resolves the installation id directly from
      ``auth_service.get_tenant_github_installation``).

    Silent no-op when either side is missing:

    - ``installation_id`` falsy (the tenant admin hasn't configured
      the App installation — see the Tenant GitHub Installation
      panel).
    - No deploy-wide App credentials (operator hasn't set
      ``GITHUB_APP_ID`` / ``GITHUB_PRIVATE_KEY_PEM``).

    In both cases, the credential file isn't written and git ops
    will fail with a typed ``GitAuthError`` whose message points
    the operator at the missing piece.
    """

    if not installation_id:
        return

    # Lazy import: this module is consumed by
    # ``DesignMonorepoCapabilityBase.initialize`` (every push-capable
    # capability mount), so we don't pay for the configs lookup at
    # import time.
    from ..agents.configs import get_github_auth_config
    gh = await get_github_auth_config()
    if not gh.app_id or not gh.private_key_pem:
        return

    manager = GitCredentialsManager.get()
    try:
        await manager.ensure(
            app_id=gh.app_id,
            private_key_pem=gh.private_key_pem,
            installation_id=str(installation_id),
        )
    except Exception:
        logger.exception(
            "git_credentials: failed to mint installation token; "
            "git push to the tenant's repos will fail with an auth "
            "error until this is resolved."
        )


async def ensure_git_credentials_from_agent_metadata(
    agent_metadata: Any,
) -> None:
    """Agent-side wrapper around
    :func:`ensure_git_credentials_for_installation` that pulls the
    installation id from ``metadata.parameters["github_identity"]
    ["tenant_installation_id"]`` (populated by the session-create
    handler in :mod:`colony.web_ui.backend.routers.sessions` from
    :func:`auth_service.get_tenant_github_installation`). Same
    silent-no-op behaviour."""

    params = getattr(agent_metadata, "parameters", None) or {}
    gh_identity = params.get("github_identity") or {}
    installation_id = gh_identity.get("tenant_installation_id")
    await ensure_git_credentials_for_installation(installation_id)


__all__ = (
    "GitCredentialsManager",
    "ensure_git_credentials_for_installation",
    "ensure_git_credentials_from_agent_metadata",
    "write_credentials_file",
)
