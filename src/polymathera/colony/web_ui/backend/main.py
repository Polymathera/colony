"""Colony Dashboard — FastAPI application entry point.

Usage:
    python -m colony.web_ui.backend.main          # Start the dashboard server
    colony-dashboard                                # Via installed script
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import DashboardConfig
from .services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)


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

    yield

    # Shutdown
    await colony.disconnect()
    logger.info("Dashboard services shut down")


def create_app(config: DashboardConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = DashboardConfig.from_env()

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
    )
    from .streaming import sse

    app.include_router(infrastructure.router, prefix="/api/v1", tags=["infrastructure"])
    app.include_router(deployments.router, prefix="/api/v1", tags=["deployments"])
    app.include_router(agents.router, prefix="/api/v1", tags=["agents"])
    app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
    app.include_router(vcm.router, prefix="/api/v1", tags=["vcm"])
    app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])
    app.include_router(logs.router, prefix="/api/v1", tags=["logs"])
    app.include_router(blackboard.router, prefix="/api/v1", tags=["blackboard"])
    app.include_router(page_graph.router, prefix="/api/v1", tags=["page-graph"])
    app.include_router(traces.router, prefix="/api/v1", tags=["traces"])
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
    import uvicorn

    config = DashboardConfig.from_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info(f"Starting Colony Dashboard on {config.host}:{config.port}")
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    cli_main()
