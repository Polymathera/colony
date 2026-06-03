"""Tests for ``license_service.upsert_license`` / ``get_license`` /
``seed_dev_licenses``.

The source-precedence ladder is the security-critical correctness
gate — a lower-priority source MUST NOT silently overwrite a paid /
admin-set plan on a routine restart. These tests pin it.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import pytest

from polymathera.colony.web_ui.backend.auth import license_service


class _StubConn:
    """In-memory fake of an asyncpg connection — only the methods
    license_service actually calls."""
    def __init__(self, store: dict[str, dict[str, Any]]) -> None:
        self._store = store

    async def fetchrow(self, sql: str, *params: Any):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("SELECT plan, source, entitlements"):
            tenant_id = params[0]
            return self._store.get(tenant_id)
        if sql_norm.startswith("SELECT id FROM tenants"):
            installation_id = params[0]
            return self._store.get(("tenant_by_install", installation_id))
        raise AssertionError(f"unexpected fetchrow: {sql_norm}")

    async def execute(self, sql: str, *params: Any):
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("INSERT INTO licenses"):
            (tenant_id, plan, entitlements_json,
             source, valid_until) = params
            self._store[tenant_id] = {
                "plan": plan,
                "source": source,
                "entitlements": entitlements_json,
                "valid_until": valid_until,
                "updated_at": "fake-now",
            }
            return "INSERT 0 1"
        raise AssertionError(f"unexpected execute: {sql_norm}")


class _StubPool:
    """Asyncpg-pool-shaped wrapper around ``_StubConn``."""
    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}

    @asynccontextmanager
    async def acquire(self):
        yield _StubConn(self.store)


@pytest.fixture
def pool() -> _StubPool:
    return _StubPool()


# ---------------------------------------------------------------------------
# upsert_license — precedence ladder


@pytest.mark.asyncio
async def test_upsert_inserts_when_row_absent(pool: _StubPool) -> None:
    result = await license_service.upsert_license(
        pool, tenant_id="tenant_a", plan="dev", source="env_bootstrap",
    )
    assert result["applied"] is True
    assert result["row"]["plan"] == "dev"
    assert result["row"]["source"] == "env_bootstrap"


@pytest.mark.asyncio
async def test_higher_source_overwrites_lower(pool: _StubPool) -> None:
    """env_bootstrap → admin: admin outranks, write applies."""
    await license_service.upsert_license(
        pool, tenant_id="tenant_a", plan="dev", source="env_bootstrap",
    )
    result = await license_service.upsert_license(
        pool, tenant_id="tenant_a", plan="enterprise", source="admin",
    )
    assert result["applied"] is True
    assert result["row"]["plan"] == "enterprise"
    assert result["row"]["source"] == "admin"


@pytest.mark.asyncio
async def test_lower_source_does_not_overwrite_higher(pool: _StubPool) -> None:
    """admin → env_bootstrap: env_bootstrap is LOWER, write must
    skip silently (idempotent restart-time seeding must not undo a
    paid/admin plan)."""
    await license_service.upsert_license(
        pool, tenant_id="tenant_a", plan="enterprise", source="admin",
    )
    result = await license_service.upsert_license(
        pool, tenant_id="tenant_a", plan="dev", source="env_bootstrap",
    )
    assert result["applied"] is False
    # Row in the store still reflects the admin entry.
    assert result["row"]["plan"] == "enterprise"
    assert result["row"]["source"] == "admin"


@pytest.mark.asyncio
async def test_marketplace_overwrites_env_bootstrap(pool: _StubPool) -> None:
    """Real customer landed via Marketplace — the env-bootstrap seed
    was a dev-time placeholder and gets replaced."""
    await license_service.upsert_license(
        pool, tenant_id="tenant_a", plan="dev", source="env_bootstrap",
    )
    result = await license_service.upsert_license(
        pool, tenant_id="tenant_a", plan="team", source="marketplace",
    )
    assert result["applied"] is True
    assert result["row"]["plan"] == "team"


@pytest.mark.asyncio
async def test_stripe_does_not_overwrite_marketplace(pool: _StubPool) -> None:
    """Order check: marketplace outranks stripe."""
    await license_service.upsert_license(
        pool, tenant_id="tenant_a", plan="team", source="marketplace",
    )
    result = await license_service.upsert_license(
        pool, tenant_id="tenant_a", plan="business", source="stripe",
    )
    assert result["applied"] is False
    assert result["row"]["plan"] == "team"


@pytest.mark.asyncio
async def test_same_source_overwrites_self(pool: _StubPool) -> None:
    """Operator changes the plan in the admin UI from team to
    enterprise — both are source='admin'; the second write is
    applied (equal rank, not lower)."""
    await license_service.upsert_license(
        pool, tenant_id="tenant_a", plan="team", source="admin",
    )
    result = await license_service.upsert_license(
        pool, tenant_id="tenant_a", plan="enterprise", source="admin",
    )
    assert result["applied"] is True
    assert result["row"]["plan"] == "enterprise"


@pytest.mark.asyncio
async def test_unknown_plan_raises(pool: _StubPool) -> None:
    with pytest.raises(ValueError, match="Unknown plan"):
        await license_service.upsert_license(
            pool, tenant_id="tenant_a", plan="megabusiness",
            source="admin",
        )


@pytest.mark.asyncio
async def test_unknown_source_raises(pool: _StubPool) -> None:
    with pytest.raises(ValueError, match="Unknown source"):
        await license_service.upsert_license(
            pool, tenant_id="tenant_a", plan="free", source="my_made_up_source",
        )


# ---------------------------------------------------------------------------
# get_license — entitlement merge


@pytest.mark.asyncio
async def test_get_license_returns_none_when_absent(pool: _StubPool) -> None:
    assert await license_service.get_license(pool, tenant_id="tenant_a") is None


@pytest.mark.asyncio
async def test_get_license_merges_overrides_with_plan_defaults(
    pool: _StubPool,
) -> None:
    """Per-tenant overrides + plan-default fallback merged into
    ``effective_entitlements`` — the shape callers actually
    enforce against."""
    pool.store["tenant_a"] = {
        "plan": "free",
        "source": "admin",
        "entitlements": json.dumps({"max_users": 100}),
        "valid_until": None,
        "updated_at": "fake-now",
    }
    result = await license_service.get_license(pool, tenant_id="tenant_a")
    assert result is not None
    assert result["plan"] == "free"
    assert result["entitlements"] == {"max_users": 100}
    # max_users came from the override; everything else from plan default.
    assert result["effective_entitlements"]["max_users"] == 100
    assert result["effective_entitlements"]["features"] == ["chat"]


# ---------------------------------------------------------------------------
# seed_dev_licenses — parsing + skip semantics


@pytest.mark.asyncio
async def test_seed_dev_licenses_empty_env_is_no_op(pool: _StubPool) -> None:
    assert await license_service.seed_dev_licenses(pool, None) == 0
    assert await license_service.seed_dev_licenses(pool, "") == 0


@pytest.mark.asyncio
async def test_seed_dev_licenses_writes_each_entry(
    monkeypatch: pytest.MonkeyPatch, pool: _StubPool,
) -> None:
    """``12345:enterprise,67890`` upserts two rows. The second entry
    has no explicit plan ⇒ defaults to ``dev``."""
    pool.store[("tenant_by_install", "12345")] = {"id": "tenant_a"}
    pool.store[("tenant_by_install", "67890")] = {"id": "tenant_b"}

    written = await license_service.seed_dev_licenses(
        pool, "12345:enterprise,67890",
    )
    assert written == 2
    assert pool.store["tenant_a"]["plan"] == "enterprise"
    assert pool.store["tenant_a"]["source"] == "env_bootstrap"
    assert pool.store["tenant_b"]["plan"] == "dev"


@pytest.mark.asyncio
async def test_seed_dev_licenses_skips_installation_with_no_tenant(
    pool: _StubPool,
) -> None:
    """First sign-in hasn't landed the tenant yet → seeder skips +
    counts zero. The walker re-runs and writes once the tenant lands."""
    written = await license_service.seed_dev_licenses(
        pool, "12345:dev",
    )
    assert written == 0


@pytest.mark.asyncio
async def test_seed_dev_licenses_skips_unknown_plan(pool: _StubPool) -> None:
    pool.store[("tenant_by_install", "12345")] = {"id": "tenant_a"}
    written = await license_service.seed_dev_licenses(
        pool, "12345:notaplan",
    )
    assert written == 0
    assert "tenant_a" not in pool.store


@pytest.mark.asyncio
async def test_seed_dev_licenses_skips_malformed_entry(pool: _StubPool) -> None:
    """Empty installation id (``:dev``) is silently dropped — must
    not raise + must not poison the rest of the list."""
    pool.store[("tenant_by_install", "12345")] = {"id": "tenant_a"}
    written = await license_service.seed_dev_licenses(
        pool, ":dev,  ,12345:enterprise",
    )
    assert written == 1
    assert pool.store["tenant_a"]["plan"] == "enterprise"
