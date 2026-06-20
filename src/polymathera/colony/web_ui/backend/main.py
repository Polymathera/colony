"""Colony Dashboard — FastAPI application entry point.

Usage:
    python -m colony.web_ui.backend.main          # Start the dashboard server
    colony-dashboard                                # Via installed script
"""

from __future__ import annotations

# First-party ConfigComponent registry bootstrap. Each
# ``@register_polymathera_config(path=…)`` decorator runs at module
# import time and adds its class to the global registry that
# :class:`PolymatheraConfig.__init__` snapshots when
# ``ConfigurationManager.initialize()`` first loads the YAML. Any
# component whose module is not imported BEFORE that snapshot is
# silently absent from ``cm.get_component(path)``, and downstream
# ``get_component_or_default(path, cls)`` calls fall through to bare
# defaults.
#
# Other entry-points (``polymath.py`` CLI, agent workers) reach these
# modules transitively via ``polymathera.colony.system``, but the
# dashboard's import chain doesn't — so we import them here, before
# anything else runs ``cli_main`` and constructs the manager.
from polymathera.colony.knowledge import cluster_config as _knowledge_config_register  # noqa: F401

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .config import DashboardConfig
from .services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)


async def _register_github_provider() -> None:
    """Register :class:`GitHubProvider` iff
    ``GITHUB_APP_CLIENT_ID`` + ``GITHUB_APP_CLIENT_SECRET`` are
    present. Best-effort — config-load + constructor failures log
    and the dashboard stays up with GitHub sign-in returning 404."""
    from polymathera.colony.agents.configs import get_github_auth_config
    from polymathera.colony.vcs import register_provider
    from polymathera.colony.vcs.github import GitHubProvider

    try:
        gh = await get_github_auth_config()
    except Exception:  # noqa: BLE001
        logger.warning(
            "VCS registry: failed to load GitHubAuthConfig; "
            "GitHub provider NOT registered.",
            exc_info=True,
        )
        return

    if not (gh.oauth_client_id and gh.oauth_client_secret):
        logger.info(
            "VCS registry: GITHUB_APP_CLIENT_ID / "
            "GITHUB_APP_CLIENT_SECRET unset; GitHub provider NOT "
            "registered. Sign-in via GitHub will return 404.",
        )
        return

    try:
        register_provider(GitHubProvider(
            oauth_client_id=gh.oauth_client_id,
            oauth_client_secret=gh.oauth_client_secret,
        ))
        logger.info(
            "VCS registry: registered GitHubProvider (client_id=%s…)",
            gh.oauth_client_id[:6],
        )
    except ValueError:
        logger.warning(
            "VCS registry: GitHubProvider construction failed; "
            "GitHub provider NOT registered.",
            exc_info=True,
        )


async def _register_gitlab_provider() -> None:
    """Register :class:`GitLabProvider` iff
    ``GITLAB_OAUTH_CLIENT_ID`` + ``GITLAB_OAUTH_CLIENT_SECRET`` are
    present. ``GITLAB_BASE_URL`` defaults to ``https://gitlab.com``;
    self-hosted instances override it."""
    from polymathera.colony.agents.configs import get_gitlab_auth_config
    from polymathera.colony.vcs import register_provider
    from polymathera.colony.vcs.gitlab import GitLabProvider

    try:
        gl = await get_gitlab_auth_config()
    except Exception:  # noqa: BLE001
        logger.warning(
            "VCS registry: failed to load GitLabAuthConfig; "
            "GitLab provider NOT registered.",
            exc_info=True,
        )
        return

    if not (gl.oauth_client_id and gl.oauth_client_secret):
        logger.info(
            "VCS registry: GITLAB_OAUTH_CLIENT_ID / "
            "GITLAB_OAUTH_CLIENT_SECRET unset; GitLab provider NOT "
            "registered. Sign-in via GitLab will return 404.",
        )
        return

    try:
        register_provider(GitLabProvider(
            oauth_client_id=gl.oauth_client_id,
            oauth_client_secret=gl.oauth_client_secret,
            base_url=gl.base_url,
        ))
        logger.info(
            "VCS registry: registered GitLabProvider "
            "(base=%s client_id=%s…)",
            gl.base_url, gl.oauth_client_id[:6],
        )
    except ValueError:
        logger.warning(
            "VCS registry: GitLabProvider construction failed; "
            "GitLab provider NOT registered.",
            exc_info=True,
        )


async def _register_vcs_providers() -> None:
    """Construct + register every :class:`VcsProvider` whose env-bound
    credentials are present. Called once from :func:`lifespan`. Each
    provider's registration is independent — one's failure does not
    block the others. The dashboard stays up regardless; sign-in via
    a non-registered provider responds 404."""
    await _register_github_provider()
    await _register_gitlab_provider()


class ExecutionContextMiddleware(BaseHTTPMiddleware):
    """Set a Ring.KERNEL execution context for every dashboard API request.

    The dashboard is an admin/monitoring interface that operates across
    tenants.  Individual routes that take colony_id/tenant_id as path
    params can narrow to Ring.USER if needed.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        from polymathera.colony.distributed.ray_utils.serving.context import (
            Ring, execution_context,
        )
        with execution_context(ring=Ring.KERNEL, origin="dashboard"):
            return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup/shutdown of shared services."""
    config: DashboardConfig = app.state.config

    # Initialize Colony connection (Ray cluster)
    colony = ColonyConnection(
        ray_client_address=config.ray_client_address,
        ray_dashboard_url=config.ray_dashboard_url,
        prometheus_url=config.prometheus_url,
    )
    await colony.connect()
    app.state.colony = colony
    logger.info(f"Colony connected via Ray Client: {config.ray_client_address}")

    # Initialize observability (Kafka→PG span consumer + query store)
    await colony.init_observability(
        pg_host=config.pg_host,
        pg_port=config.pg_port,
        pg_user=config.pg_user,
        pg_password=config.pg_password,
        pg_database=config.pg_database,
        kafka_bootstrap=config.kafka_bootstrap,
    )

    # Attach Kafka log handler so dashboard backend logs appear in the UI's Logs tab
    from polymathera.colony.distributed.observability.log_setup import attach_kafka_log_handler
    await attach_kafka_log_handler(kafka_bootstrap=config.kafka_bootstrap)

    # Register VCS providers based on which OAuth credentials are
    # present in env. Today: GitHub only — GitLab / Bitbucket land
    # when their providers are implemented (see
    # ``colony/vcs_native_tenancy_plan.md`` PR 7+). Registration is a
    # no-op when credentials are absent; the signup-router responds
    # 404 to ``/auth/{provider}/sign-in`` for unregistered providers.
    await _register_vcs_providers()

    # Initialize database schemas
    if colony._db_pool:
        from .auth.schema import ensure_auth_schema
        from .chat.schema import ensure_chat_schema
        from polymathera.colony.agents.patterns.capabilities.github_inbound import (
            ensure_github_inbound_schema,
        )
        from polymathera.colony.agents.patterns.capabilities.interaction_log import (
            ensure_interaction_log_schema,
        )
        from .github_webhook import ensure_github_webhook_schema
        await ensure_auth_schema(colony._db_pool)
        await ensure_chat_schema(colony._db_pool)
        # P8a: cursor table for GitHubInboundCapability. Must land
        # before the system-session bootstrap below so the capability's
        # initialize() can read the table on first tick.
        await ensure_github_inbound_schema(colony._db_pool)
        # P8b: interaction_log table for InteractionLogCapability.
        # Same lifecycle reasoning — must land before system-session
        # bootstrap.
        await ensure_interaction_log_schema(colony._db_pool)
        # P9: dedup table for the github webhook receiver.
        await ensure_github_webhook_schema(colony._db_pool)

        # Seed dev licenses from the operator's ``.env``. Each entry
        # in ``COLONY_DEV_LICENSED_INSTALLATIONS`` is upserted with
        # ``source='env_bootstrap'``; tenants not yet landed in
        # postgres get skipped (the signup walker re-runs this same
        # seeder once it lands them). Idempotent + source-precedence
        # aware — never overwrites a Marketplace/admin row.
        import os
        from .auth.license_service import seed_dev_licenses
        try:
            seeded = await seed_dev_licenses(
                colony._db_pool,
                os.environ.get("COLONY_DEV_LICENSED_INSTALLATIONS"),
            )
            if seeded:
                logger.info(
                    "lifespan: seeded %d dev license(s) from "
                    "COLONY_DEV_LICENSED_INSTALLATIONS", seeded,
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "lifespan: dev license seeding failed; "
                "non-fatal — dashboard continues.",
            )

        # Make chat store available via app state
        from .chat.store import ChatMessageStore
        app.state.chat_store = ChatMessageStore(colony._db_pool)

        # P8-0: bootstrap a system session per colony. Always-on host
        # for colony-singleton capabilities (P8: GitHub inbound +
        # InteractionLog; P9+ webhook + mention routing). Idempotent;
        # called again from ``services.colony_lifecycle.provision_colony``
        # after a new colony lands so fresh colonies get their system
        # session without a dashboard restart. Best-effort — a single
        # colony's failure does NOT prevent the dashboard from starting.
        from .chat.system_session import (
            ensure_system_sessions_for_all_colonies,
        )
        try:
            await ensure_system_sessions_for_all_colonies(colony)
        except Exception:  # noqa: BLE001
            logger.exception(
                "lifespan: system-session bootstrap failed; "
                "colony-singleton capabilities will not be running. "
                "Dashboard remains up.",
            )
    else:
        app.state.chat_store = None

    yield

    # Shutdown
    from polymathera.colony.distributed.observability.log_setup import detach_kafka_log_handler
    await detach_kafka_log_handler()
    await colony.disconnect()
    logger.info("Dashboard services shut down")


def create_app(config: DashboardConfig) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Colony Dashboard",
        description="Web dashboard for monitoring and debugging Polymathera Colony agents",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = config

    # CORS — allow localhost origins for Vite dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",  # Vite dev server
            "http://localhost:8080",  # Dashboard itself
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auth middleware — extracts JWT from cookie, sets ExecutionContext
    from .auth.middleware import AuthMiddleware
    app.add_middleware(AuthMiddleware)

    # Register API routers
    from .routers import (
        infrastructure,
        deployments,
        agents,
        sessions,
        vcm,
        metrics,
        logs,
        blackboard,
        page_graph,
        traces,
        trace_analysis,
        repo_map,
        kb,
        jobs,
        chat,
        auth,
        colonies,
        colony_status,
        human_approval,
        human_help,
        github_webhook,
        tenants,
    )
    from .routers import config as config_router
    from .streaming import sse

    app.include_router(auth.router, prefix="/api/v1", tags=["auth"])
    app.include_router(github_webhook.router, prefix="/api/v1", tags=["github-webhook"])
    app.include_router(tenants.router, prefix="/api/v1", tags=["tenants"])
    app.include_router(colonies.router, prefix="/api/v1", tags=["colonies"])
    app.include_router(colony_status.router, prefix="/api/v1", tags=["colony-status"])
    app.include_router(infrastructure.router, prefix="/api/v1", tags=["infrastructure"])
    app.include_router(deployments.router, prefix="/api/v1", tags=["deployments"])
    app.include_router(agents.router, prefix="/api/v1", tags=["agents"])
    app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
    app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    app.include_router(human_approval.router, prefix="/api/v1", tags=["human-approval"])
    app.include_router(human_help.router, prefix="/api/v1", tags=["human-help"])
    app.include_router(config_router.router, prefix="/api/v1", tags=["config"])
    app.include_router(vcm.router, prefix="/api/v1", tags=["vcm"])
    app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])
    app.include_router(logs.router, prefix="/api/v1", tags=["logs"])
    app.include_router(blackboard.router, prefix="/api/v1", tags=["blackboard"])
    app.include_router(page_graph.router, prefix="/api/v1", tags=["page-graph"])
    app.include_router(traces.router, prefix="/api/v1", tags=["traces"])
    app.include_router(trace_analysis.router, prefix="/api/v1", tags=["trace-analysis"])
    app.include_router(repo_map.router, prefix="/api/v1", tags=["repo-map"])
    app.include_router(kb.router, prefix="/api/v1", tags=["kb"])
    app.include_router(sse.router, prefix="/api/v1", tags=["streaming"])

    # Serve built frontend as static files (production mode)
    static_dir = config.static_dir
    if static_dir is None:
        # Default: look for frontend/dist relative to this package
        candidate = Path(__file__).parent.parent / "frontend" / "dist"
        if candidate.is_dir():
            static_dir = str(candidate)

    if static_dir and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
        logger.info(f"Serving frontend from {static_dir}")

    return app


def cli_main():
    """CLI entry point for the dashboard server."""
    import asyncio
    import uvicorn

    from polymathera.colony.distributed import get_initialized_polymathera

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize the global ConfigurationManager once at startup so every
    # sync ``get_*_config()`` helper used by routers / capabilities sees the
    # operator YAML at $POLYMATHERA_CONFIG and the per-field env-var
    # bindings (``RDS_*``, ``KAFKA_*``, ``RAY_*``, ...) declared on the
    # registered ConfigComponents. Without this, ``DashboardConfig.from_env``
    # would fall through to bare Pydantic defaults — e.g. ``pg_password=""``
    # — and Postgres-backed routes (auth, chat, observability) would fail
    # with "Database not available".
    asyncio.run(get_initialized_polymathera())

    config = asyncio.run(DashboardConfig.from_env())
    logger.info(f"Starting Colony Dashboard on {config.host}:{config.port}")
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    cli_main()
