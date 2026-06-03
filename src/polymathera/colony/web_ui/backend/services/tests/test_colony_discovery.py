"""Tests for ``services/colony_discovery.discover_colonies_for_tenant``.

The walker orchestrates: list repos → probe each for ``.colony/`` →
gate via ``any_colony_exists_for_repo`` → call ``provision_colony``.
We mock all four collaborators and assert on the orchestration shape.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from polymathera.colony.vcs import (
    OAuthExchangeError,
    VcsRepoRef,
    VcsTenantRef,
)
from polymathera.colony.web_ui.backend.services import colony_discovery


def _tenant() -> VcsTenantRef:
    return VcsTenantRef(
        vcs_org_id="org1", vcs_org_login="acme", display_name="Acme",
        installation_id="100", role_hint="member",
    )


def _repo(full_name: str, repo_id: str = "1") -> VcsRepoRef:
    return VcsRepoRef(
        vcs_repo_id=repo_id, full_name=full_name,
        default_branch="main", user_permission="write",
    )


def _provider(*, repos: list[VcsRepoRef], marked_repos: set[str]) -> MagicMock:
    """Build a provider whose ``list_tenant_repos`` returns ``repos``
    and whose ``repo_path_exists`` returns True only for
    ``full_name`` in ``marked_repos``."""
    p = MagicMock()
    p.provider_id = "github"
    p.display_name = "GitHub"
    p.list_tenant_repos = AsyncMock(return_value=repos)

    async def _path_exists(*, repo: VcsRepoRef, **_kw) -> bool:
        return repo.full_name in marked_repos
    p.repo_path_exists = AsyncMock(side_effect=_path_exists)
    # repo_clone_url is sync (pure formatting). MagicMock would
    # return a MagicMock — coerce to a real string so the discovery
    # walker's set_design_monorepo call receives a usable URL.
    p.repo_clone_url = lambda repo: f"https://x/{repo.full_name}.git"
    return p


def _colony_with_pool() -> SimpleNamespace:
    return SimpleNamespace(_db_pool=object())


# ---------------------------------------------------------------------
# Happy paths


@pytest.fixture(autouse=True)
def _stub_walker_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the auth-service SQL helpers the discovery walker now
    calls (``upsert_tenant_repo`` + ``set_design_monorepo`` via
    ``provision_colony``-side path). All tests get a no-op stub so
    the fake ``_db_pool=object()`` doesn't blow up on ``.acquire()``.
    Tests that care about specific call shapes patch their own stubs
    on top."""
    monkeypatch.setattr(
        colony_discovery.auth_service, "upsert_tenant_repo",
        AsyncMock(return_value=None),
    )


@pytest.mark.asyncio
async def test_discovery_provisions_one_colony_per_marked_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        colony_discovery.auth_service, "any_colony_exists_for_repo",
        AsyncMock(return_value=False),
    )
    provision = AsyncMock(side_effect=[
        {"colony_id": "colony_1", "name": "repo-a", "tenant_id": "t1"},
        {"colony_id": "colony_2", "name": "repo-c", "tenant_id": "t1"},
    ])
    monkeypatch.setattr(colony_discovery, "provision_colony", provision)

    provider = _provider(
        repos=[
            _repo("acme/repo-a", "1"),
            _repo("acme/repo-b", "2"),     # unmarked → skipped
            _repo("acme/repo-c", "3"),
        ],
        marked_repos={"acme/repo-a", "acme/repo-c"},
    )

    new_ids = await colony_discovery.discover_colonies_for_tenant(
        _colony_with_pool(),
        tenant_id="t1",
        tenant_ref=_tenant(),
        provider=provider,
        access_token="gho_x",
    )
    assert new_ids == ["colony_1", "colony_2"]
    assert provision.await_count == 2
    # repo-b was never even probed-for-Colony? No — we probe every repo,
    # we just don't provision the unmarked one. Verify probe count.
    assert provider.repo_path_exists.await_count == 3


@pytest.mark.asyncio
async def test_discovery_caches_every_repo_in_tenant_repos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The walker upserts EVERY repo it sees into ``tenant_repos``
    (regardless of ``.colony/`` marker) so the dashboard's
    ``/discoverable-repos`` route can render a complete dropdown.
    ``has_colony_marker`` reflects the probe result."""
    cache_calls: list[dict[str, object]] = []

    async def _record_upsert(_db, **kwargs):
        cache_calls.append(kwargs)
    monkeypatch.setattr(
        colony_discovery.auth_service, "upsert_tenant_repo",
        AsyncMock(side_effect=_record_upsert),
    )
    monkeypatch.setattr(
        colony_discovery.auth_service, "any_colony_exists_for_repo",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        colony_discovery, "provision_colony",
        AsyncMock(return_value={
            "colony_id": "colony_a", "name": "x", "tenant_id": "t1",
        }),
    )

    provider = _provider(
        repos=[
            _repo("acme/marked", "1"),
            _repo("acme/unmarked", "2"),
        ],
        marked_repos={"acme/marked"},
    )

    await colony_discovery.discover_colonies_for_tenant(
        _colony_with_pool(),
        tenant_id="t1", tenant_ref=_tenant(),
        provider=provider, access_token="gho_x",
    )

    # Both repos cached. Marker bit reflects probe result.
    assert {c["vcs_repo_full_name"] for c in cache_calls} == {
        "acme/marked", "acme/unmarked",
    }
    marked = next(c for c in cache_calls if c["vcs_repo_full_name"] == "acme/marked")
    unmarked = next(c for c in cache_calls if c["vcs_repo_full_name"] == "acme/unmarked")
    assert marked["has_colony_marker"] is True
    assert unmarked["has_colony_marker"] is False


@pytest.mark.asyncio
async def test_discovery_skips_repo_already_having_colony(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The walker MUST NOT auto-create a second colony on a repo
    that already has one for this tenant (per-tenant first-time-only
    discovery semantics — plan §4.3)."""

    async def _exists(*_a, **kw) -> bool:
        return kw["vcs_repo_id"] == "already_has_colony"
    monkeypatch.setattr(
        colony_discovery.auth_service, "any_colony_exists_for_repo",
        AsyncMock(side_effect=_exists),
    )
    provision = AsyncMock(return_value={
        "colony_id": "new", "name": "fresh", "tenant_id": "t1",
    })
    monkeypatch.setattr(colony_discovery, "provision_colony", provision)

    provider = _provider(
        repos=[
            _repo("acme/already", "already_has_colony"),
            _repo("acme/fresh",   "fresh_id"),
        ],
        marked_repos={"acme/already", "acme/fresh"},
    )

    new_ids = await colony_discovery.discover_colonies_for_tenant(
        _colony_with_pool(),
        tenant_id="t1", tenant_ref=_tenant(),
        provider=provider, access_token="gho_x",
    )
    assert new_ids == ["new"]
    provision.assert_awaited_once()
    # Both repos get probed: the walker now records the marker bit
    # in ``tenant_repos`` BEFORE the gate, so it has to probe every
    # repo. The gate just suppresses the provision step on repos
    # that already have a colony.
    assert provider.repo_path_exists.await_count == 2


# ---------------------------------------------------------------------
# Failure modes


@pytest.mark.asyncio
async def test_discovery_propagates_list_repos_oauth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OAuthExchangeError from ``list_tenant_repos`` propagates so the
    sign-in handler can turn it into a 4xx (token revoked / scope
    missing)."""
    monkeypatch.setattr(
        colony_discovery.auth_service, "any_colony_exists_for_repo",
        AsyncMock(return_value=False),
    )
    p = MagicMock()
    p.provider_id = "github"
    p.display_name = "GitHub"
    p.list_tenant_repos = AsyncMock(
        side_effect=OAuthExchangeError("scope missing"),
    )

    with pytest.raises(OAuthExchangeError, match="scope missing"):
        await colony_discovery.discover_colonies_for_tenant(
            _colony_with_pool(),
            tenant_id="t1", tenant_ref=_tenant(),
            provider=p, access_token="gho_x",
        )


@pytest.mark.asyncio
async def test_discovery_swallows_other_list_repos_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-OAuth exceptions from list_tenant_repos (network blip,
    JSON parse) log + return [] — don't abort the sign-in."""
    monkeypatch.setattr(
        colony_discovery.auth_service, "any_colony_exists_for_repo",
        AsyncMock(return_value=False),
    )
    p = MagicMock()
    p.provider_id = "github"
    p.display_name = "GitHub"
    p.list_tenant_repos = AsyncMock(
        side_effect=RuntimeError("upstream down"),
    )

    result = await colony_discovery.discover_colonies_for_tenant(
        _colony_with_pool(),
        tenant_id="t1", tenant_ref=_tenant(),
        provider=p, access_token="gho_x",
    )
    assert result == []


@pytest.mark.asyncio
async def test_discovery_continues_after_per_repo_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One repo's probe blowing up MUST NOT abort discovery of the
    others — best-effort per repo."""
    monkeypatch.setattr(
        colony_discovery.auth_service, "any_colony_exists_for_repo",
        AsyncMock(return_value=False),
    )
    provision = AsyncMock(return_value={
        "colony_id": "good", "name": "ok", "tenant_id": "t1",
    })
    monkeypatch.setattr(colony_discovery, "provision_colony", provision)

    repos = [_repo("acme/bad", "1"), _repo("acme/ok", "2")]
    p = MagicMock()
    p.provider_id = "github"
    p.display_name = "GitHub"
    p.list_tenant_repos = AsyncMock(return_value=repos)

    async def _path_exists(*, repo: VcsRepoRef, **_kw) -> bool:
        if repo.full_name == "acme/bad":
            raise RuntimeError("transient")
        return True
    p.repo_path_exists = AsyncMock(side_effect=_path_exists)
    p.repo_clone_url = lambda repo: f"https://x/{repo.full_name}.git"

    new_ids = await colony_discovery.discover_colonies_for_tenant(
        _colony_with_pool(),
        tenant_id="t1", tenant_ref=_tenant(),
        provider=p, access_token="gho_x",
    )
    assert new_ids == ["good"]


@pytest.mark.asyncio
async def test_discovery_no_op_when_db_pool_missing() -> None:
    """No db pool → can't write colonies → bail with [] (don't crash)."""
    colony = SimpleNamespace(_db_pool=None)
    p = MagicMock()
    p.list_tenant_repos = AsyncMock()  # must NOT be called
    result = await colony_discovery.discover_colonies_for_tenant(
        colony, tenant_id="t1", tenant_ref=_tenant(),
        provider=p, access_token="gho_x",
    )
    assert result == []
    p.list_tenant_repos.assert_not_awaited()
