"""Shared factory for building a ``GitHubClient`` bound to a specific
App installation.

Two capabilities need this construction today:

- :class:`GitHubCapability` (chat-session per-user surface) — resolves
  ``installation_id`` from ``agent.metadata.parameters
  ["github_identity"]["tenant_installation_id"]`` (populated by P4
  session-create).
- :class:`GitHubInboundCapability` (system-session colony poller) —
  resolves ``installation_id`` from Postgres via
  ``auth_service.get_tenant_github_installation`` because the system
  session has no per-user ``github_identity`` metadata block.

Both call :func:`build_github_client_for_installation` to do the
App-creds + httpx + ``TokenCache`` + ``GitHubClient`` construction —
~30 lines of identical wiring that was duplicated before the P8a
refactor pulled it here.

What's NOT in this factory: ``installation_id`` resolution itself.
That logic belongs in each capability because each has a different
correct source. The factory takes ``installation_id`` as an explicit
argument so neither caller has to fight the other's source ordering.
"""

from __future__ import annotations

import httpx

from .auth import GitHubAppAuth, TokenCache
from .client import GitHubClient


def _default_httpx_client() -> httpx.AsyncClient:
    """Default timeout profile shared by both capabilities.

    The numbers come from the original ``GitHubCapability._build_live_client``
    (P5); kept identical so existing tests' timing characteristics
    are unchanged."""

    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=10.0, read=30.0, write=10.0, pool=10.0,
        ),
    )


async def build_github_client_for_installation(
    *,
    installation_id: str,
    app_id_override: str | None = None,
    private_key_pem_override: str | None = None,
    httpx_client: httpx.AsyncClient | None = None,
    capability_name: str = "GitHubClient",
) -> tuple[GitHubClient, httpx.AsyncClient]:
    """Build a :class:`GitHubClient` for the named App installation.

    Args:
        installation_id: Caller-resolved installation id. The factory
            does NOT know how to find this — it's the one piece that
            differs between callers.
        app_id_override: Test-only — overrides the App id read from
            ``GitHubAuthConfig`` (which itself reads ``GITHUB_APP_ID``
            from env). Production callers leave this ``None``.
        private_key_pem_override: Same as ``app_id_override`` for the
            RSA PEM. Production reads ``GITHUB_PRIVATE_KEY_PEM`` via
            the env-bound config.
        httpx_client: Pre-built ``httpx.AsyncClient`` for test
            injection. When ``None``, the factory builds one with the
            standard timeout profile.
        capability_name: Class name attribution for error messages,
            so the operator sees ``"GitHubInboundCapability: GITHUB_APP_ID
            … required"`` not ``"GitHubClient: …"``.

    Returns:
        Tuple ``(client, httpx_client)`` — the caller owns the
        ``httpx_client`` lifecycle and must ``await aclose()`` it in
        its ``shutdown()`` (unless it pre-built and passed one in,
        in which case the caller already owns it).
    """

    # Resolve App credentials. Either both overrides set (test
    # injection — skip the config lookup entirely) or fall through
    # to the env-bound config.
    if app_id_override and private_key_pem_override:
        app_id = app_id_override
        private_key_pem = private_key_pem_override
    else:
        # Lazy import: this factory ships in the ``_github/`` package
        # which is otherwise config-free (auth.py + client.py are
        # pure). Importing ``configs`` at module load time would
        # widen the import surface for anyone touching the package.
        # App-level credentials (``app_id`` + ``private_key_pem``) are
        # deploy-wide and come from env via ``GitHubAuthConfig``.
        from ....configs import get_github_auth_config
        gh = await get_github_auth_config()
        app_id = app_id_override or gh.app_id or None
        private_key_pem = (
            private_key_pem_override or gh.private_key_pem or None
        )

    if not app_id or not private_key_pem:
        raise RuntimeError(
            f"{capability_name}: GITHUB_APP_ID and "
            f"GITHUB_PRIVATE_KEY_PEM env vars are required.",
        )

    if httpx_client is None:
        httpx_client = _default_httpx_client()

    app_auth = GitHubAppAuth(
        app_id=app_id, private_key_pem=private_key_pem,
    )
    tokens = TokenCache(
        app_auth=app_auth,
        installation_id=installation_id,
        client=httpx_client,
    )
    return GitHubClient(tokens=tokens, client=httpx_client), httpx_client


__all__ = ("build_github_client_for_installation",)
