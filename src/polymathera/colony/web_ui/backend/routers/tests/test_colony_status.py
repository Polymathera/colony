"""Tests for the P11 ``/api/v1/colony-status/*`` routes via FastAPI
TestClient with stubbed dependencies.

Mirrors the P9 ``test_github_webhook.py`` shape: real router + real
auth dependency overrides + AsyncMock at each load-bearing service
boundary. Verifies the route shape + the alert event_kind filter +
the project-link derivation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


pytestmark = pytest.mark.asyncio


def _build_app(
    *,
    fetch_recent_returns: list[dict] | None = None,
    design_monorepo_returns: dict | None = None,
    tenant_id: str = "t1",
    colony_id: str = "c1",
) -> tuple[FastAPI, dict[str, Any]]:
    """Build a minimal FastAPI app with the colony_status router
    and every dependency stubbed."""

    from polymathera.colony.web_ui.backend.dependencies import get_colony
    from polymathera.colony.web_ui.backend.auth.middleware import require_auth
    from polymathera.colony.web_ui.backend.routers import colony_status
    from polymathera.colony.web_ui.backend.auth import (
        service as auth_service,
    )
    from polymathera.colony.agents.patterns.capabilities.interaction_log import (
        service as interaction_log_service,
    )

    capture: dict[str, Any] = {"_originals": {}}

    app = FastAPI()
    app.include_router(colony_status.router, prefix="/api/v1")

    fake_colony = SimpleNamespace(
        app_name="test-app",
        _db_pool=object(),
    )
    app.dependency_overrides[get_colony] = lambda: fake_colony

    # ``require_auth`` returns a user dict; stub returns a tenant-id-
    # bearing one so the route reads the right tenant on the
    # project-link path.
    app.dependency_overrides[require_auth] = lambda: {
        "tenant_id": tenant_id,
        "sub": "user_x",
    }

    # Stub ``fetch_recent_activity`` + ``get_design_monorepo`` at the
    # ROUTER module's import site so the patch reaches the binding
    # the route actually calls.
    capture["_originals"]["fetch_recent_activity"] = (
        interaction_log_service.fetch_recent_activity
    )
    capture["_originals"]["get_design_monorepo"] = (
        auth_service.get_design_monorepo
    )

    interaction_log_service.fetch_recent_activity = AsyncMock(  # type: ignore[assignment]
        return_value=fetch_recent_returns or [],
    )
    auth_service.get_design_monorepo = AsyncMock(  # type: ignore[assignment]
        return_value=design_monorepo_returns,
    )

    # The route imports symbols locally; rebind on the router module
    # to make the patches reach the call sites.
    colony_status.auth_service = auth_service

    # Stub execution-context getters so the route's `get_tenant_id` /
    # `get_colony_id` reads return the test values without setting
    # up a real syscontext.
    from polymathera.colony.distributed.ray_utils.serving import context
    capture["_originals"]["get_tenant_id"] = context.get_tenant_id
    capture["_originals"]["get_colony_id"] = context.get_colony_id
    context.get_tenant_id = lambda: tenant_id  # type: ignore[assignment]
    context.get_colony_id = lambda: colony_id  # type: ignore[assignment]

    return app, capture


def _restore(capture: dict[str, Any]) -> None:
    from polymathera.colony.web_ui.backend.auth import (
        service as auth_service,
    )
    from polymathera.colony.agents.patterns.capabilities.interaction_log import (
        service as interaction_log_service,
    )
    from polymathera.colony.distributed.ray_utils.serving import context

    originals = capture["_originals"]
    interaction_log_service.fetch_recent_activity = (
        originals["fetch_recent_activity"]
    )
    auth_service.get_design_monorepo = (
        originals["get_design_monorepo"]
    )
    context.get_tenant_id = originals["get_tenant_id"]
    context.get_colony_id = originals["get_colony_id"]


# ---------------------------------------------------------------------------
# /alerts
# ---------------------------------------------------------------------------


async def test_alerts_returns_only_alert_event_kinds() -> None:
    """``/alerts`` over-fetches via ``fetch_recent_activity`` then
    post-filters to ``event_kind in ('bottleneck','inconsistency')``.
    GitHub events / mention rows are excluded."""

    app, capture = _build_app(fetch_recent_returns=[
        {
            "id": 1, "ts": "2026-06-02T10:00:00Z",
            "event_kind": "bottleneck", "channel": "internal",
            "payload": {"summary": "stalled"}, "refs": [],
            "channel_ref": "https://github.com/a/b/issues/1",
        },
        {
            "id": 2, "ts": "2026-06-02T11:00:00Z",
            "event_kind": "github_issue_event", "channel": "github",
            "payload": {}, "refs": [], "channel_ref": None,
        },
        {
            "id": 3, "ts": "2026-06-02T12:00:00Z",
            "event_kind": "inconsistency", "channel": "internal",
            "payload": {"summary": "C-11 vs D-04"}, "refs": [],
            "channel_ref": None,
        },
    ])
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/colony-status/alerts")
        assert resp.status_code == 200
        data = resp.json()
        kinds = [a["event_kind"] for a in data["alerts"]]
        assert "bottleneck" in kinds
        assert "inconsistency" in kinds
        assert "github_issue_event" not in kinds
        assert data["count"] == 2
    finally:
        _restore(capture)


async def test_alerts_empty_when_no_alerts_in_recent_activity() -> None:
    app, capture = _build_app(fetch_recent_returns=[
        {
            "id": 1, "ts": "x",
            "event_kind": "github_issue_event", "channel": "github",
            "payload": {}, "refs": [], "channel_ref": None,
        },
    ])
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/colony-status/alerts")
        assert resp.status_code == 200
        assert resp.json() == {"alerts": [], "count": 0}
    finally:
        _restore(capture)


# ---------------------------------------------------------------------------
# /recent-activity
# ---------------------------------------------------------------------------


async def test_recent_activity_returns_raw_tail() -> None:
    """``/recent-activity`` returns ``fetch_recent_activity``'s output
    unchanged — no event_kind filtering."""

    rows = [
        {
            "id": 1, "ts": "t", "event_kind": "github_issue_event",
            "channel": "github", "payload": {}, "refs": [],
            "channel_ref": None, "user_login": "alice",
        },
        {
            "id": 2, "ts": "t", "event_kind": "mention_event",
            "channel": "github", "payload": {}, "refs": [],
            "channel_ref": None, "user_login": "bob",
        },
    ]
    app, capture = _build_app(fetch_recent_returns=rows)
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/colony-status/recent-activity")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert [e["id"] for e in data["events"]] == [1, 2]
    finally:
        _restore(capture)


# ---------------------------------------------------------------------------
# /agent-diagnostics (R12 follow-up — health dashboard surface)
# ---------------------------------------------------------------------------


async def test_agent_diagnostics_returns_only_agent_diagnostic_kind() -> None:
    """``/agent-diagnostics`` over-fetches via ``fetch_recent_activity``
    then post-filters to ``event_kind == 'agent_diagnostic'``. GitHub
    events / alert rows are excluded; this is the dashboard-side mirror
    of the alerts route's post-filter pattern."""

    app, capture = _build_app(fetch_recent_returns=[
        {
            "id": 1, "ts": "2026-06-21T10:00:00Z",
            "event_kind": "agent_diagnostic", "channel": "internal",
            "payload": {
                "agent_id": "session_agent_abc",
                "kind": "session_agent_stopped",
                "stop_reason": "max_iterations_exceeded",
            },
            "refs": [
                {"kind": "diagnostic_kind", "value": "session_agent_stopped"},
                {"kind": "agent_id", "value": "session_agent_abc"},
            ],
            "channel_ref": None,
        },
        {
            "id": 2, "ts": "2026-06-21T11:00:00Z",
            "event_kind": "github_issue_event", "channel": "github",
            "payload": {}, "refs": [], "channel_ref": None,
        },
        {
            "id": 3, "ts": "2026-06-21T12:00:00Z",
            "event_kind": "agent_diagnostic", "channel": "internal",
            "payload": {
                "agent_id": "github_inbound_a1",
                "kind": "github_inbound_quiesced",
                "reason": "no_db_pool",
            },
            "refs": [
                {"kind": "diagnostic_kind", "value": "github_inbound_quiesced"},
            ],
            "channel_ref": None,
        },
    ])
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/colony-status/agent-diagnostics")
        assert resp.status_code == 200
        data = resp.json()
        kinds = [d["payload"]["kind"] for d in data["diagnostics"]]
        assert "session_agent_stopped" in kinds
        assert "github_inbound_quiesced" in kinds
        # GitHub events excluded.
        assert data["count"] == 2
        assert all(
            d["event_kind"] == "agent_diagnostic"
            for d in data["diagnostics"]
        )
    finally:
        _restore(capture)


async def test_agent_diagnostics_empty_when_no_matching_rows() -> None:
    app, capture = _build_app(fetch_recent_returns=[
        {
            "id": 1, "ts": "x",
            "event_kind": "github_issue_event", "channel": "github",
            "payload": {}, "refs": [], "channel_ref": None,
        },
    ])
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/colony-status/agent-diagnostics")
        assert resp.status_code == 200
        assert resp.json() == {"diagnostics": [], "count": 0}
    finally:
        _restore(capture)


# ---------------------------------------------------------------------------
# /project-link
# ---------------------------------------------------------------------------


async def test_project_link_returns_github_projects_url() -> None:
    """``design_monorepo_url`` set + parses as github.com → return
    the repo-level Projects URL."""

    app, capture = _build_app(design_monorepo_returns={
        "origin_url": "https://github.com/acme/widgets.git",
    })
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/colony-status/project-link")
        assert resp.status_code == 200
        assert resp.json() == {
            "project_url": "https://github.com/acme/widgets/projects",
        }
    finally:
        _restore(capture)


async def test_project_link_returns_none_when_monorepo_unconfigured() -> None:
    """No design monorepo configured → ``project_url: None``. The
    panel hides the "Open Project board" button in this case."""

    app, capture = _build_app(design_monorepo_returns=None)
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/colony-status/project-link")
        assert resp.status_code == 200
        assert resp.json() == {"project_url": None}
    finally:
        _restore(capture)


async def test_project_link_returns_none_for_non_github_url() -> None:
    """GitLab / internal forge → ``project_url: None`` (we don't try
    to guess equivalent Project board URLs on other forges)."""

    app, capture = _build_app(design_monorepo_returns={
        "origin_url": "https://gitlab.com/acme/widgets.git",
    })
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/colony-status/project-link")
        assert resp.status_code == 200
        assert resp.json() == {"project_url": None}
    finally:
        _restore(capture)
