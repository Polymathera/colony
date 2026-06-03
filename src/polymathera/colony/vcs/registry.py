"""Process-wide registry of available VCS providers.

Populated at dashboard startup based on which env-bound credentials
are present (e.g. ``GITHUB_APP_CLIENT_ID`` + ``GITHUB_APP_CLIENT_SECRET``
present ⇒ register :class:`GitHubProvider`). The signup router reads
:func:`enabled_providers` to render the provider-picker UI; the
sign-in/callback handlers read :func:`get_provider` to dispatch.

This is a singleton mutable global by design — the providers are
deploy-wide config, not per-request state — but the test fixture in
``tests/conftest.py`` resets it between tests so cross-test
pollution can't leak.
"""

from __future__ import annotations

from .provider import VcsProvider


_REGISTRY: dict[str, VcsProvider] = {}


def register_provider(provider: VcsProvider) -> None:
    """Register ``provider`` under its ``provider_id``. Subsequent
    registrations with the same id overwrite — operators can
    swap implementations at startup if needed (rarely useful in
    production; common in tests)."""
    _REGISTRY[provider.provider_id] = provider


def unregister_provider(provider_id: str) -> None:
    """Remove ``provider_id`` from the registry. Primarily a test
    helper; production code does not unregister."""
    _REGISTRY.pop(provider_id, None)


def get_provider(provider_id: str) -> VcsProvider:
    """Return the provider registered under ``provider_id``.

    Raises :class:`KeyError` (which the router converts to a 404)
    if no provider is registered — the operator either hasn't set
    the relevant env vars or the user typed an unknown id into the
    ``/auth/{provider}/...`` URL."""
    try:
        return _REGISTRY[provider_id]
    except KeyError:
        raise KeyError(
            f"No VCS provider registered for id={provider_id!r}. "
            f"Registered: {sorted(_REGISTRY.keys()) or '[]'}",
        ) from None


def enabled_providers() -> list[VcsProvider]:
    """All registered providers, in registration order. The UI
    renders these as the provider-picker buttons; order is the order
    operators register them in (``main.lifespan``)."""
    return list(_REGISTRY.values())


def reset_registry_for_testing() -> None:
    """Test-only — drop every registered provider so each test
    starts from an empty registry. Never call from production code."""
    _REGISTRY.clear()


__all__ = (
    "enabled_providers",
    "get_provider",
    "register_provider",
    "reset_registry_for_testing",
    "unregister_provider",
)
