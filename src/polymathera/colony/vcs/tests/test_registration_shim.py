"""Tests for the dashboard's VCS-provider-registration shim
(``main._register_vcs_providers``).

The shim is the bridge between env-bound config + the registry. It
MUST stay defensive: a missing/misconfigured provider's credentials
should leave the dashboard fully up and serving everything else,
just with that provider absent from the registry. Each provider
registers independently — one's failure doesn't block the others.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polymathera.colony.vcs import enabled_providers
from polymathera.colony.vcs.registry import reset_registry_for_testing
from polymathera.colony.web_ui.backend.main import _register_vcs_providers


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


def _stub_gh(*, client_id: str = "", client_secret: str = ""):
    return SimpleNamespace(
        oauth_client_id=client_id,
        oauth_client_secret=client_secret,
    )


def _stub_gl(
    *, client_id: str = "", client_secret: str = "",
    base_url: str = "https://gitlab.com",
):
    return SimpleNamespace(
        oauth_client_id=client_id,
        oauth_client_secret=client_secret,
        base_url=base_url,
    )


def _patch_configs(
    monkeypatch: pytest.MonkeyPatch, *, gh=None, gl=None,
    gh_raises: Exception | None = None,
    gl_raises: Exception | None = None,
) -> None:
    """Mock both ``get_github_auth_config`` and ``get_gitlab_auth_config``
    so each test pins exactly what the shim sees. Either side can be
    forced to raise to exercise the defensive paths."""
    if gh_raises is not None:
        gh_mock = AsyncMock(side_effect=gh_raises)
    else:
        gh_mock = AsyncMock(return_value=gh if gh is not None else _stub_gh())
    if gl_raises is not None:
        gl_mock = AsyncMock(side_effect=gl_raises)
    else:
        gl_mock = AsyncMock(return_value=gl if gl is not None else _stub_gl())
    monkeypatch.setattr(
        "polymathera.colony.agents.configs.get_github_auth_config",
        gh_mock,
    )
    monkeypatch.setattr(
        "polymathera.colony.agents.configs.get_gitlab_auth_config",
        gl_mock,
    )


# ---------------------------------------------------------------------
# GitHub-side behaviours


@pytest.mark.asyncio
async def test_registers_github_when_creds_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_configs(
        monkeypatch,
        gh=_stub_gh(client_id="Iv23liReal", client_secret="real-secret"),
    )
    await _register_vcs_providers()
    ids = [p.provider_id for p in enabled_providers()]
    assert "github" in ids


@pytest.mark.asyncio
async def test_skips_github_when_creds_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_configs(monkeypatch)  # both empty
    await _register_vcs_providers()
    assert enabled_providers() == []


@pytest.mark.asyncio
async def test_skips_github_when_config_load_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_configs(monkeypatch, gh_raises=RuntimeError("bad PEM"))
    await _register_vcs_providers()
    assert [p.provider_id for p in enabled_providers()] == []


# ---------------------------------------------------------------------
# GitLab-side behaviours


@pytest.mark.asyncio
async def test_registers_gitlab_when_creds_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_configs(
        monkeypatch,
        gl=_stub_gl(client_id="gl-id", client_secret="gl-secret"),
    )
    await _register_vcs_providers()
    ids = [p.provider_id for p in enabled_providers()]
    assert "gitlab" in ids


@pytest.mark.asyncio
async def test_skips_gitlab_when_creds_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_configs(monkeypatch)
    await _register_vcs_providers()
    ids = [p.provider_id for p in enabled_providers()]
    assert "gitlab" not in ids


@pytest.mark.asyncio
async def test_skips_gitlab_when_config_load_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_configs(monkeypatch, gl_raises=RuntimeError("bad config"))
    await _register_vcs_providers()
    ids = [p.provider_id for p in enabled_providers()]
    assert "gitlab" not in ids


# ---------------------------------------------------------------------
# Independence — one provider's failure doesn't block the other


@pytest.mark.asyncio
async def test_gitlab_registers_when_github_config_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitHub config crashes; GitLab still registers if its creds are
    set. Plan-mandated isolation: providers are independent."""
    _patch_configs(
        monkeypatch,
        gh_raises=RuntimeError("bad PEM"),
        gl=_stub_gl(client_id="gl-id", client_secret="gl-secret"),
    )
    await _register_vcs_providers()
    ids = [p.provider_id for p in enabled_providers()]
    assert ids == ["gitlab"]


@pytest.mark.asyncio
async def test_github_registers_when_gitlab_config_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_configs(
        monkeypatch,
        gh=_stub_gh(client_id="gh-id", client_secret="gh-secret"),
        gl_raises=RuntimeError("bad config"),
    )
    await _register_vcs_providers()
    ids = [p.provider_id for p in enabled_providers()]
    assert ids == ["github"]


@pytest.mark.asyncio
async def test_both_register_when_both_creds_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-provider deployment: both registered, order preserved
    (GitHub first since main.py registers it first → UI provider
    picker renders in that order)."""
    _patch_configs(
        monkeypatch,
        gh=_stub_gh(client_id="gh-id", client_secret="gh-secret"),
        gl=_stub_gl(client_id="gl-id", client_secret="gl-secret"),
    )
    await _register_vcs_providers()
    ids = [p.provider_id for p in enabled_providers()]
    assert ids == ["github", "gitlab"]
