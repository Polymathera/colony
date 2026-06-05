"""``GitHubInboundCapability`` — colony-singleton poller mounted on
the system session.

One instance per (tenant, colony). Reads
``.colony/github_inbound.yaml`` from the colony's design monorepo via
GitHub's contents API (no local clone needed), runs a tick loop on
the operator-configured cadence, and emits ``GitHubEventProtocol``
writes to the colony-scoped blackboard.

The capability is best-effort at initialize time — every prerequisite
that is operator-controlled (installation_id unset, design monorepo
unconfigured, YAML absent / malformed) drops the capability into a
quiescent "no poll loop started" state with a WARNING log. The
SessionAgent stays up so other system-session capabilities (P8b
InteractionLog, P9 webhook, P10 mention routing) still run.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import httpx

from ...events import event_handler
from ....base import Agent
from ....blackboard.protocol import (
    GitHubEventProtocol,
    MonorepoCommitProtocol,
)
from ....scopes import BlackboardScope
from ....utils.postgres import get_agent_db_pool
from .._github.client import GitHubClient, GitHubError, NotFoundError
from .._github.factory import build_github_client_for_installation
from ..colony_singleton_base import ColonySingletonCapabilityBase
from .config import GitHubInboundConfig
from .cursor import bump_cursor, get_cursor
from .poller import poll_repo


logger = logging.getLogger(__name__)


# Bounds on the poll loop's tolerance for tick failures before it
# backs off. After ``_MAX_CONSECUTIVE_FAILURES`` ticks fail in a row,
# the loop sleeps ``_FAILURE_BACKOFF_SECONDS`` before retrying — a
# crude circuit breaker that avoids hammering a sick GitHub.
_MAX_CONSECUTIVE_FAILURES = 3
_FAILURE_BACKOFF_SECONDS = 300.0  # 5 minutes


class GitHubInboundCapability(ColonySingletonCapabilityBase):
    """Colony-singleton GitHub inbound poller.

    Args:
        agent: Owning ``SessionAgent`` (the system session's).
        scope: ``BlackboardScope.COLONY`` — the emitted
            ``GitHubEventProtocol`` writes are colony-scoped so user
            sessions in the same colony see them.
        config: Pre-loaded ``GitHubInboundConfig`` for test injection;
            production reads it from the design monorepo via the
            GitHub contents API.
        client: Pre-built ``GitHubClient`` for test injection;
            production constructs one from the per-tenant
            ``installation_id``.
        db_pool: Pre-built Postgres pool for test injection only.
            Production callers MUST NOT pass this — live asyncpg
            pools are not cloudpickle-serializable + would fail the
            blueprint's ``validate_serializable`` check. Production
            acquires the pool lazily via
            :func:`agents.utils.postgres.get_agent_db_pool` inside
            ``initialize()``.
        capability_key: Dispatcher key.
        app_name: Serving application name override.
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope: BlackboardScope = BlackboardScope.COLONY,
        *,
        scope_id: str | None = None,
        config: GitHubInboundConfig | None = None,
        client: GitHubClient | None = None,
        db_pool: Any = None,
        capability_key: str | None = None,
        app_name: str | None = None,
    ):
        super().__init__(
            agent=agent,
            scope=scope,
            scope_id=scope_id,
            capability_key=capability_key,
            app_name=app_name,
        )
        # ``db_pool`` is injected by tests; production leaves it None
        # and ``initialize()`` acquires the process-level pool.
        self._db_pool = db_pool
        self._config: GitHubInboundConfig | None = config
        self._client: GitHubClient | None = client
        self._http_owned: bool = client is None
        self._httpx_client: httpx.AsyncClient | None = None
        self._poll_task: asyncio.Task | None = None
        self._tenant_id: str | None = None
        self._colony_id: str | None = None
        self._design_monorepo_url: str | None = None
        # Whether the capability is currently quiesced (i.e. did not
        # successfully start a poll loop). Surface on the action /
        # status panel later; for now just a flag for tests.
        self._quiesced_reason: str | None = "not_initialized"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Wire up auth, fetch the YAML, start the poll loop.

        Returns silently (with a WARNING log + ``_quiesced_reason``
        set) on every operator-controlled prerequisite failure. The
        SessionAgent must stay up regardless."""

        await super().initialize()

        # Pull tenant + colony from the agent's syscontext-derived
        # typed properties on AgentMetadata. The system-session
        # bootstrap wraps blueprint construction in
        # ``colony.user_execution_context(tenant_id=..., colony_id=...)``
        # so ``AgentMetadata.syscontext`` captures both at metadata
        # construction time; ``metadata.tenant_id`` / ``.colony_id``
        # then resolve to those values for the agent's lifetime.
        if self._agent is None:
            self._tenant_id = ""
            self._colony_id = ""
        else:
            self._tenant_id = self._agent.metadata.tenant_id
            self._colony_id = self._agent.metadata.colony_id
        if not self._tenant_id or not self._colony_id:
            self._quiesced_reason = "no_tenant_or_colony_in_syscontext"
            logger.warning(
                "GitHubInboundCapability: tenant_id/colony_id absent "
                "from agent syscontext; poll loop NOT started.",
            )
            return

        if self._db_pool is None:
            try:
                self._db_pool = await get_agent_db_pool()
            except Exception as exc:  # noqa: BLE001
                self._quiesced_reason = "no_db_pool"
                logger.warning(
                    "GitHubInboundCapability: failed to acquire "
                    "agent-process Postgres pool (%s); poll loop "
                    "NOT started. Cursor persistence requires the "
                    "ray-worker container to have RDS_* env vars "
                    "set (see docker-compose.yml).",
                    exc,
                )
                return

        # Build the live GitHub client (auth via tenant installation
        # token). Pre-built clients (tests) skip this.
        if self._client is None:
            try:
                self._client = await self._build_live_client()
            except Exception as exc:  # noqa: BLE001
                self._quiesced_reason = "github_client_unavailable"
                logger.warning(
                    "GitHubInboundCapability: live GitHub client "
                    "unavailable (%s); poll loop NOT started. The "
                    "capability will retry on next dashboard restart "
                    "or MonorepoCommitProtocol event.",
                    exc,
                )
                return

        # Load the config — either pre-injected (tests) or fetched
        # from the design monorepo via the GitHub contents API.
        if self._config is None:
            try:
                self._design_monorepo_url = await self._fetch_design_monorepo_url()
                if not self._design_monorepo_url:
                    self._quiesced_reason = "no_design_monorepo_url"
                    logger.warning(
                        "GitHubInboundCapability: colony %s has no "
                        "design_monorepo_url configured; poll loop "
                        "NOT started.",
                        self._colony_id,
                    )
                    return
                yaml_text = await self._fetch_inbound_yaml(
                    self._design_monorepo_url,
                )
                if yaml_text is None:
                    self._quiesced_reason = "no_inbound_yaml"
                    logger.info(
                        "GitHubInboundCapability: colony %s has no "
                        ".colony/github_inbound.yaml in its design "
                        "monorepo; poll loop NOT started (silent — "
                        "operator opt-in feature).",
                        self._colony_id,
                    )
                    return
                self._config = GitHubInboundConfig.load_from_yaml_text(
                    yaml_text,
                )
            except ValueError as exc:
                # Includes the "mode: webhook needs P9" error.
                self._quiesced_reason = "config_parse_failed"
                logger.warning(
                    "GitHubInboundCapability: github_inbound.yaml "
                    "parse failed for colony %s: %s; poll loop NOT "
                    "started.",
                    self._colony_id, exc,
                )
                return
            except Exception as exc:  # noqa: BLE001
                self._quiesced_reason = "config_fetch_failed"
                logger.warning(
                    "GitHubInboundCapability: failed to fetch "
                    ".colony/github_inbound.yaml from design "
                    "monorepo for colony %s: %s; poll loop NOT "
                    "started.",
                    self._colony_id, exc,
                )
                return

        # P9: when ``mode: webhook``, the dashboard's
        # ``POST /api/v1/github/webhook`` receiver is the active
        # surface — this agent-side capability skips its poll loop
        # entirely. Quiesce with a clear reason so logs/tests can
        # distinguish "operator chose webhook" from "operator
        # mis-configured something".
        if self._config.github_inbound.mode == "webhook":
            self._quiesced_reason = "webhook_mode"
            logger.info(
                "GitHubInboundCapability: colony %s configured as "
                "``mode: webhook``; poll loop NOT started (the "
                "dashboard webhook receiver is the active surface).",
                self._colony_id,
            )
            return

        # All prerequisites satisfied — start the poll loop.
        self._quiesced_reason = None
        self._poll_task = asyncio.create_task(
            self._poll_loop(),
            name=(
                f"github_inbound_poll_loop:"
                f"{self._tenant_id}:{self._colony_id}"
            ),
        )
        logger.info(
            "GitHubInboundCapability: poll loop started for "
            "tenant=%s colony=%s, %d repo(s), interval=%ds",
            self._tenant_id, self._colony_id,
            len(self._config.github_inbound.poll_repos),
            self._config.github_inbound.poll_interval_seconds,
        )

    async def shutdown(self) -> None:
        """Cancel the poll task + close the owned httpx client."""
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._poll_task = None
        if self._client is not None and self._http_owned:
            try:
                await self._client.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Hot-reload on monorepo commit (Decision F1)
    # ------------------------------------------------------------------

    @event_handler(pattern=MonorepoCommitProtocol.event_pattern())
    async def _on_monorepo_commit(self, event, _scope) -> None:  # type: ignore[no-untyped-def]
        """Re-fetch the YAML + restart the poll loop on every
        monorepo commit.

        Operator edits to ``.colony/github_inbound.yaml`` land in the
        running cluster without a dashboard restart. Cancellation +
        re-spawn is the simplest safe approach; the next tick picks
        up from the same cursor row so no GitHub events are lost.
        """

        if self._tenant_id is None or self._colony_id is None:
            return  # capability never made it past initialize

        value = event.value or {}
        logger.info(
            "GitHubInboundCapability: MonorepoCommitProtocol fired "
            "(branch=%s sha=%s); re-loading config for colony %s",
            value.get("branch"), value.get("sha"), self._colony_id,
        )

        # Cancel the existing loop, then re-initialize.
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._poll_task = None

        # Force re-fetch of YAML on the next initialize.
        self._config = None
        await self.initialize()

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Tick once per ``poll_interval_seconds``. Per tick: walk
        each repo, run :func:`poll_repo`, bump the cursor on success.
        Best-effort per repo — one failing repo doesn't stop the
        others.
        """

        assert self._config is not None
        assert self._client is not None
        assert self._tenant_id is not None
        assert self._colony_id is not None

        interval = self._config.github_inbound.poll_interval_seconds
        repos = self._config.github_inbound.poll_repos
        consecutive_failures = 0

        try:
            while True:
                tick_had_any_success = False
                for repo in repos:
                    try:
                        await self._tick_one_repo(repo)
                        tick_had_any_success = True
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "GitHubInboundCapability: tick failed for "
                            "%s: %s",
                            repo, exc,
                        )

                if tick_had_any_success or not repos:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.warning(
                        "GitHubInboundCapability: %d consecutive "
                        "tick failures; backing off %d s",
                        consecutive_failures, _FAILURE_BACKOFF_SECONDS,
                    )
                    await asyncio.sleep(_FAILURE_BACKOFF_SECONDS)
                    consecutive_failures = 0
                else:
                    await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception(
                "GitHubInboundCapability: poll loop crashed; "
                "capability is now quiesced until restart",
            )
            self._quiesced_reason = "poll_loop_crashed"

    async def _tick_one_repo(self, repo: str) -> None:
        cursor = await get_cursor(
            self._db_pool,
            tenant_id=self._tenant_id,
            colony_id=self._colony_id,
            repo=repo,
            channel="issues",
        )
        blackboard = await self._get_colony_blackboard()
        new_last_updated, new_last_seen_id, writes = await poll_repo(
            client=self._client,
            blackboard=blackboard,
            cursor=cursor,
        )
        if new_last_updated > cursor.last_updated:
            await bump_cursor(
                self._db_pool,
                tenant_id=self._tenant_id,
                colony_id=self._colony_id,
                repo=repo,
                channel="issues",
                last_updated=new_last_updated,
                last_seen_id=new_last_seen_id,
            )
        if writes:
            logger.info(
                "GitHubInboundCapability: %s → %d blackboard write(s)",
                repo, writes,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _build_live_client(self) -> GitHubClient:
        """Construct a tenant-scoped ``GitHubClient`` via the standard
        App-installation flow.

        Mirrors :meth:`GitHubCapability._build_live_client` shape: each
        capability resolves ``installation_id`` its own way (the
        system session has no per-user ``github_identity`` metadata
        block — Postgres is the source of truth here), then delegates
        the App-creds + httpx + TokenCache + GitHubClient construction
        to :func:`build_github_client_for_installation`.
        """

        from polymathera.colony.web_ui.backend.auth import (
            service as auth_service,
        )

        tenant_row = await auth_service.get_tenant_github_installation(
            self._db_pool, tenant_id=self._tenant_id,
        )
        installation_id = (tenant_row or {}).get("installation_id")
        if not installation_id:
            raise RuntimeError(
                f"GitHubInboundCapability: tenant {self._tenant_id} "
                f"has no GitHub App installation id configured "
                f"(Tenant GitHub Installation panel).",
            )

        client, self._httpx_client = await build_github_client_for_installation(
            installation_id=str(installation_id),
            capability_name="GitHubInboundCapability",
        )
        return client

    async def _fetch_design_monorepo_url(self) -> str | None:
        from polymathera.colony.web_ui.backend.auth import (
            service as auth_service,
        )
        row = await auth_service.get_design_monorepo(
            self._db_pool,
            colony_id=self._colony_id,
            tenant_id=self._tenant_id,
        )
        return (row or {}).get("origin_url")

    async def _fetch_inbound_yaml(self, design_monorepo_url: str) -> str | None:
        """Fetch ``.colony/github_inbound.yaml`` from the design
        monorepo via the GitHub contents API.

        Returns ``None`` if the file doesn't exist (404 — operator
        hasn't opted into inbound polling). Re-raises everything else.
        """

        from polymathera.colony.design_monorepo.process import (
            parse_owner_repo_from_url,
        )

        repo = parse_owner_repo_from_url(design_monorepo_url)
        if not repo:
            raise ValueError(
                f"design_monorepo_url {design_monorepo_url!r} is not "
                f"a github.com URL; inbound polling requires github.com",
            )
        path = ".colony/github_inbound.yaml"

        try:
            data = await self._client.get(
                f"/repos/{repo}/contents/{path}",
            )
        except NotFoundError:
            return None
        except GitHubError:
            raise

        # The contents API returns base64-encoded file content (when
        # ``encoding == "base64"`` — the only encoding GitHub uses
        # for file blobs ≤ 1 MB). Decode + return as UTF-8 string.
        if not isinstance(data, dict):
            raise ValueError(
                f"GitHub contents API returned non-dict for {path}: "
                f"{type(data).__name__}",
            )
        if data.get("encoding") != "base64":
            raise ValueError(
                f"GitHub contents API returned unexpected encoding "
                f"{data.get('encoding')!r} for {path}",
            )
        raw = base64.b64decode(data["content"]).decode("utf-8")
        return raw
