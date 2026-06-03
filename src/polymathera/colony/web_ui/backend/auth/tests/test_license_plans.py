"""Tests for the plan-default merge helper."""

from __future__ import annotations

import pytest

from polymathera.colony.web_ui.backend.auth.license_plans import (
    PLAN_DEFAULTS,
    resolve_entitlements,
)


def test_unknown_plan_raises() -> None:
    with pytest.raises(KeyError, match="Unknown plan"):
        resolve_entitlements("nonexistent")


def test_free_plan_defaults_returned_when_no_overrides() -> None:
    result = resolve_entitlements("free")
    assert result == PLAN_DEFAULTS["free"]
    # Identity: caller mutates result; PLAN_DEFAULTS stays clean
    result["max_users"] = 999
    assert PLAN_DEFAULTS["free"]["max_users"] != 999


def test_overrides_win_over_plan_defaults() -> None:
    """Per-tenant override of a single field overrides only that
    field — every other field takes the plan default."""
    result = resolve_entitlements(
        "free",
        overrides={"max_users": 100},
    )
    assert result["max_users"] == 100
    assert result["max_colonies_per_tenant"] == PLAN_DEFAULTS["free"]["max_colonies_per_tenant"]
    assert result["features"] == PLAN_DEFAULTS["free"]["features"]


def test_enterprise_plan_has_no_caps() -> None:
    """Enterprise plan returns None for every numeric cap — callers
    treat None as ``unlimited``."""
    result = resolve_entitlements("enterprise")
    assert result["max_users"] is None
    assert result["max_colonies_per_tenant"] is None
    assert result["max_concurrent_sessions"] is None


def test_dev_plan_has_all_features() -> None:
    """Dev plan exists for ``COLONY_DEV_LICENSED_INSTALLATIONS`` —
    no caps + every feature flag, so local dev isn't quota-blocked."""
    result = resolve_entitlements("dev")
    assert result["max_users"] is None
    assert set(result["features"]) >= {
        "chat", "github_inbound", "interaction_log", "mention_routing",
    }
