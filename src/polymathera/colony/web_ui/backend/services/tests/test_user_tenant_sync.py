"""Tests for the first-login walker
(``services/user_tenant_sync.sync_user_after_signin``).

The walker calls a handful of ``auth_service`` SQL helpers + the
provider's ``list_user_tenants`` + the dev-license seeder + the
colony-discovery walker. We mock all of them and assert on the
orchestration shape — that:

- The user is upserted from the identity DTO.
- Every returned ``VcsTenantRef`` lands as a ``tenants`` row.
- Memberships are upserted in lock-step.
- Colony discovery runs per tenant (PR 4).
- The dev-license seeder runs after tenants land (so freshly-landed
  tenants get their dev plan).
- The active colony is resolved (existing valid, or first available,
  or None).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from polymathera.colony.vcs import VcsTenantRef, VcsUserIdentity
from polymathera.colony.web_ui.backend.services import user_tenant_sync


def _colony(pool: object = None) -> SimpleNamespace:
    """ColonyConnection stub — the walker only reads ``_db_pool``."""
    return SimpleNamespace(_db_pool=pool if pool is not None else object())


def _make_identity(*, login: str = "alice") -> VcsUserIdentity:
    return VcsUserIdentity(
        vcs_user_id="42",
        login=login,
        name="Alice User",
        primary_email="alice@example.com",
        verified_emails=("alice@example.com",),
    )


def _make_tenant_ref(
    *, vcs_org_id: str = "1001", login: str = "polymathera-inc",
    installation_id: str | None = "100",
) -> VcsTenantRef:
    return VcsTenantRef(
        vcs_org_id=vcs_org_id,
        vcs_org_login=login,
        display_name=login,
        installation_id=installation_id,
        role_hint="member",
    )


def _provider(tenants: list[VcsTenantRef]) -> MagicMock:
    p = MagicMock()
    p.provider_id = "github"
    p.list_user_tenants = AsyncMock(return_value=tenants)
    return p


def _stub_service(monkeypatch: pytest.MonkeyPatch) -> dict[str, AsyncMock]:
    """Stub every ``auth_service`` SQL helper the walker touches +
    the PR-4 colony-discovery integration point. Returns the mock
    dict so individual tests can assert on calls."""
    stubs: dict[str, AsyncMock] = {
        "upsert_user_from_vcs": AsyncMock(return_value={
            "user_id": "user_xyz", "is_new": True,
        }),
        "upsert_tenant_from_vcs": AsyncMock(return_value={
            "tenant_id": "tenant_a",
        }),
        "upsert_user_tenant": AsyncMock(),
        "set_active_colony": AsyncMock(),
        "list_colonies": AsyncMock(return_value=[]),
        "get_user_by_id": AsyncMock(return_value={
            "user_id": "user_xyz", "active_colony_id": None,
        }),
    }
    for name, mock in stubs.items():
        monkeypatch.setattr(user_tenant_sync.auth_service, name, mock)
    # Default: discovery finds zero new colonies — keeps the existing
    # active-colony-resolution tests purely about the resolver logic.
    # Tests that exercise discovery integration patch this themselves.
    monkeypatch.setattr(
        user_tenant_sync, "discover_colonies_for_tenant",
        AsyncMock(return_value=[]),
    )
    return stubs


# ---------------------------------------------------------------------
# Happy path


@pytest.mark.asyncio
async def test_walker_upserts_user_tenants_and_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stubs = _stub_service(monkeypatch)
    monkeypatch.setattr(
        user_tenant_sync, "seed_dev_licenses",
        AsyncMock(return_value=1),
    )
    provider = _provider([
        _make_tenant_ref(vcs_org_id="1001", login="acme",  installation_id="111"),
        _make_tenant_ref(vcs_org_id="2002", login="globex", installation_id="222"),
    ])

    result = await user_tenant_sync.sync_user_after_signin(
        _colony(),
        provider=provider,
        identity=_make_identity(login="alice"),
        access_token="gho_x",
        license_env_value="111:dev",
    )

    # User upsert: identity normalised + provider id threaded.
    stubs["upsert_user_from_vcs"].assert_awaited_once_with(
        result_caller_db := stubs["upsert_user_from_vcs"].call_args.args[0],
        vcs_provider="github", vcs_user_id="42", vcs_login="alice",
        vcs_email="alice@example.com", name="Alice User",
    )
    # Two tenant upserts, in the order list_user_tenants returned them.
    assert stubs["upsert_tenant_from_vcs"].await_count == 2
    first_call_kwargs = stubs["upsert_tenant_from_vcs"].call_args_list[0].kwargs
    assert first_call_kwargs["vcs_org_id"] == "1001"
    assert first_call_kwargs["github_installation_id"] == "111"
    # Two memberships.
    assert stubs["upsert_user_tenant"].await_count == 2
    # Result reflects two tenants discovered.
    assert len(result.tenant_ids) == 2
    assert result.is_new_user is True
    assert result.user_id == "user_xyz"


@pytest.mark.asyncio
async def test_walker_seeds_dev_licenses_after_tenants_land(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seeder must run AFTER tenant upsert (so the
    installation_id → tenant lookup inside the seeder finds the
    row). The walker passes through the env-var value verbatim."""
    _stub_service(monkeypatch)
    seed = AsyncMock(return_value=1)
    monkeypatch.setattr(user_tenant_sync, "seed_dev_licenses", seed)

    await user_tenant_sync.sync_user_after_signin(
        _colony(pool="pool_sentinel"),
        provider=_provider([_make_tenant_ref()]),
        identity=_make_identity(),
        access_token="gho_x",
        license_env_value="100:enterprise",
    )

    seed.assert_awaited_once_with("pool_sentinel", "100:enterprise")


@pytest.mark.asyncio
async def test_walker_skips_license_seed_when_no_tenants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User belongs to zero Colony-installed tenants → no license to
    seed → seeder not called (saves a round trip + log spam)."""
    _stub_service(monkeypatch)
    seed = AsyncMock()
    monkeypatch.setattr(user_tenant_sync, "seed_dev_licenses", seed)

    result = await user_tenant_sync.sync_user_after_signin(
        _colony(),
        provider=_provider([]),
        identity=_make_identity(),
        access_token="gho_x",
    )
    seed.assert_not_awaited()
    assert result.tenant_ids == ()


@pytest.mark.asyncio
async def test_walker_tolerates_license_seed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing seeder MUST NOT crash the sign-in flow — best-effort
    semantics, the user can still proceed under plan='free' defaults."""
    _stub_service(monkeypatch)
    monkeypatch.setattr(
        user_tenant_sync, "seed_dev_licenses",
        AsyncMock(side_effect=RuntimeError("postgres temporarily down")),
    )

    result = await user_tenant_sync.sync_user_after_signin(
        _colony(),
        provider=_provider([_make_tenant_ref()]),
        identity=_make_identity(),
        access_token="gho_x",
    )
    assert result.user_id == "user_xyz"


# ---------------------------------------------------------------------
# Active-colony resolution


@pytest.mark.asyncio
async def test_walker_picks_existing_active_colony_when_still_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning user kept their active colony from last session AND
    they still belong to its tenant — pointer survives, no SET to NULL."""
    stubs = _stub_service(monkeypatch)
    monkeypatch.setattr(user_tenant_sync, "seed_dev_licenses", AsyncMock())
    stubs["get_user_by_id"].return_value = {
        "user_id": "user_xyz", "active_colony_id": "colony_existing",
    }
    stubs["list_colonies"].return_value = [
        {"colony_id": "colony_existing", "tenant_id": "tenant_a"},
        {"colony_id": "colony_other",    "tenant_id": "tenant_a"},
    ]

    result = await user_tenant_sync.sync_user_after_signin(
        _colony(),
        provider=_provider([_make_tenant_ref()]),
        identity=_make_identity(),
        access_token="gho_x",
    )
    assert result.active_colony_id == "colony_existing"
    # No SET — the existing pointer was valid, leave it alone.
    stubs["set_active_colony"].assert_not_awaited()


@pytest.mark.asyncio
async def test_walker_picks_first_colony_when_existing_pointer_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User's previous active_colony_id no longer points at a colony
    they can see — fall through to "first colony in first tenant"."""
    stubs = _stub_service(monkeypatch)
    monkeypatch.setattr(user_tenant_sync, "seed_dev_licenses", AsyncMock())
    stubs["get_user_by_id"].return_value = {
        "user_id": "user_xyz",
        "active_colony_id": "colony_no_longer_visible",
    }
    stubs["list_colonies"].return_value = [
        {"colony_id": "colony_new_first", "tenant_id": "tenant_a"},
    ]

    result = await user_tenant_sync.sync_user_after_signin(
        _colony(),
        provider=_provider([_make_tenant_ref()]),
        identity=_make_identity(),
        access_token="gho_x",
    )
    assert result.active_colony_id == "colony_new_first"
    stubs["set_active_colony"].assert_awaited_once_with(
        stubs["set_active_colony"].call_args.args[0],
        user_id="user_xyz", colony_id="colony_new_first",
    )


@pytest.mark.asyncio
async def test_walker_sets_active_to_none_when_no_colonies_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No colonies at all (PR 4 hasn't run discovery yet) → active is
    None + persisted as NULL so the JWT-minting caller carries through."""
    stubs = _stub_service(monkeypatch)
    monkeypatch.setattr(user_tenant_sync, "seed_dev_licenses", AsyncMock())
    stubs["list_colonies"].return_value = []  # no colonies anywhere

    result = await user_tenant_sync.sync_user_after_signin(
        _colony(),
        provider=_provider([_make_tenant_ref()]),
        identity=_make_identity(),
        access_token="gho_x",
    )
    assert result.active_colony_id is None
    stubs["set_active_colony"].assert_awaited_once_with(
        stubs["set_active_colony"].call_args.args[0],
        user_id="user_xyz", colony_id=None,
    )


# ---------------------------------------------------------------------
# Colony discovery integration (PR 4)


@pytest.mark.asyncio
async def test_walker_runs_discovery_per_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For each tenant we land, the walker must call
    ``discover_colonies_for_tenant`` exactly once, threading the
    same provider + access_token and the just-upserted tenant_id +
    tenant_ref."""
    stubs = _stub_service(monkeypatch)
    monkeypatch.setattr(user_tenant_sync, "seed_dev_licenses", AsyncMock())
    discover = AsyncMock(side_effect=[
        ["colony_a1"],
        ["colony_b1", "colony_b2"],
    ])
    monkeypatch.setattr(
        user_tenant_sync, "discover_colonies_for_tenant", discover,
    )
    # Distinct tenant_ids so each call's args are inspectable.
    stubs["upsert_tenant_from_vcs"].side_effect = [
        {"tenant_id": "tenant_a"},
        {"tenant_id": "tenant_b"},
    ]
    provider = _provider([
        _make_tenant_ref(vcs_org_id="1001", login="acme",  installation_id="111"),
        _make_tenant_ref(vcs_org_id="2002", login="globex", installation_id="222"),
    ])

    result = await user_tenant_sync.sync_user_after_signin(
        _colony(),
        provider=provider,
        identity=_make_identity(),
        access_token="gho_x",
    )

    assert discover.await_count == 2
    first_kwargs = discover.call_args_list[0].kwargs
    assert first_kwargs["tenant_id"] == "tenant_a"
    assert first_kwargs["tenant_ref"].vcs_org_login == "acme"
    assert first_kwargs["provider"] is provider
    assert first_kwargs["access_token"] == "gho_x"

    # Discovered colony ids land on the result + are surfaced in the
    # log (the route uses them for diagnostics).
    assert result.discovered_colony_ids == (
        "colony_a1", "colony_b1", "colony_b2",
    )


@pytest.mark.asyncio
async def test_walker_continues_when_discovery_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery failing for tenant A must not abort sign-in or
    block discovery for tenant B."""
    stubs = _stub_service(monkeypatch)
    monkeypatch.setattr(user_tenant_sync, "seed_dev_licenses", AsyncMock())
    discover = AsyncMock(side_effect=[
        RuntimeError("scope missing in tenant A"),
        ["colony_b1"],
    ])
    monkeypatch.setattr(
        user_tenant_sync, "discover_colonies_for_tenant", discover,
    )
    stubs["upsert_tenant_from_vcs"].side_effect = [
        {"tenant_id": "tenant_a"},
        {"tenant_id": "tenant_b"},
    ]

    result = await user_tenant_sync.sync_user_after_signin(
        _colony(),
        provider=_provider([
            _make_tenant_ref(vcs_org_id="1001"),
            _make_tenant_ref(vcs_org_id="2002"),
        ]),
        identity=_make_identity(),
        access_token="gho_x",
    )
    # Tenant B's discovery still ran + returned a colony.
    assert result.discovered_colony_ids == ("colony_b1",)
    assert len(result.tenant_ids) == 2
