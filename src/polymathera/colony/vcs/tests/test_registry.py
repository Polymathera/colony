"""Tests for the VCS provider registry."""

from __future__ import annotations

import pytest

from polymathera.colony.vcs import (
    VcsProvider,
    enabled_providers,
    get_provider,
    register_provider,
)
from polymathera.colony.vcs.registry import (
    reset_registry_for_testing,
    unregister_provider,
)


class _StubProvider:
    """Minimum surface for runtime_checkable conformance — only the
    attributes the registry actually touches."""
    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id
        self.display_name = provider_id.upper()
    def build_authorize_url(self, **_kw): return ""
    async def exchange_code_for_token(self, **_kw): return ""
    async def fetch_user_identity(self, **_kw): raise NotImplementedError
    async def list_user_tenants(self, **_kw): return []
    async def list_tenant_repos(self, **_kw): return []
    async def repo_path_exists(self, **_kw): return False
    def repo_clone_url(self, repo): return ""


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Every test starts with an empty registry. Prevents cross-test
    pollution because the registry is process-global."""
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()


def test_get_provider_raises_keyerror_when_not_registered() -> None:
    with pytest.raises(KeyError, match="No VCS provider registered"):
        get_provider("github")


def test_register_then_get_roundtrip() -> None:
    p = _StubProvider("github")
    register_provider(p)
    assert get_provider("github") is p


def test_register_overwrites_same_id() -> None:
    """A second registration with the same provider_id replaces the
    first — used by tests that want to swap a real provider for a
    mock without unregistering first."""
    first = _StubProvider("github")
    second = _StubProvider("github")
    register_provider(first)
    register_provider(second)
    assert get_provider("github") is second


def test_unregister_drops_provider() -> None:
    register_provider(_StubProvider("github"))
    unregister_provider("github")
    with pytest.raises(KeyError):
        get_provider("github")


def test_unregister_unknown_is_no_op() -> None:
    """``unregister_provider`` on an unregistered id does NOT raise —
    test cleanup that doesn't track what it registered shouldn't
    have to guard."""
    unregister_provider("nope")


def test_enabled_providers_returns_registration_order() -> None:
    """Order matters: the UI provider-picker renders these in
    operator-controlled order, which IS the registration order from
    main.lifespan."""
    g = _StubProvider("github")
    l = _StubProvider("gitlab")
    register_provider(g)
    register_provider(l)
    assert [p.provider_id for p in enabled_providers()] == ["github", "gitlab"]


def test_stub_satisfies_protocol_runtime_check() -> None:
    """`VcsProvider` is `@runtime_checkable` — the stub used here
    must satisfy isinstance() so the test fixture's mocking pattern
    keeps working without subclassing Protocol explicitly."""
    assert isinstance(_StubProvider("github"), VcsProvider)
