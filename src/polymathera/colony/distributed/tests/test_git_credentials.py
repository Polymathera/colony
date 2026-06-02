"""Tests for the per-agent git credential helper writer.

No real network: GitHub's ``/access_tokens`` exchange is stubbed via
``httpx.MockTransport`` patched onto the ``TokenCache``'s HTTP client.

See ``colony/distributed/git_credentials.py`` + P9 of
``colony/github_identity_fix_plan.md``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from polymathera.colony.distributed.git_credentials import (
    GitCredentialsManager,
    ensure_git_credentials_for_installation,
    ensure_git_credentials_from_agent_metadata,
    write_credentials_file,
)


def _make_rsa_key_pem() -> str:
    key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend(),
    )
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


def _token_handler_factory(token: str = "ghs_abc"):
    """Stub GitHub's ``POST /app/installations/<id>/access_tokens``
    endpoint. Returns a 201 with the canned token + a far-future
    expiry so the refresh-padding check in ``TokenCache.get`` sees
    the token as fresh."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.startswith("/app/installations/")
        assert request.url.path.endswith("/access_tokens")
        return httpx.Response(201, json={
            "token": token,
            "expires_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() + 60 * 60),
            ),
        })
    return handler


@pytest.fixture(autouse=True)
async def _reset_manager():
    """Teardown the singleton between tests so background refresh
    tasks don't leak across the session."""

    yield
    await GitCredentialsManager.reset_for_tests()


# ---------------------------------------------------------------------------
# write_credentials_file
# ---------------------------------------------------------------------------


def test_write_credentials_file_writes_token(tmp_path: Path) -> None:
    target = tmp_path / "creds"
    write_credentials_file(target, "ghs_xyz")
    assert target.read_text() == "ghs_xyz"


def test_write_credentials_file_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "subdir" / "creds"
    write_credentials_file(target, "ghs_xyz")
    assert target.read_text() == "ghs_xyz"


def test_write_credentials_file_is_chmod_600(tmp_path: Path) -> None:
    """The file carries a live installation token; readable only by
    the owner."""

    target = tmp_path / "creds"
    write_credentials_file(target, "ghs_xyz")
    mode = os.stat(target).st_mode & 0o777
    assert mode == 0o600


def test_write_credentials_file_overwrites_atomically(
    tmp_path: Path,
) -> None:
    """A second write replaces the first via ``os.replace`` — the
    file content is the second token, not a concatenation or partial
    state."""

    target = tmp_path / "creds"
    write_credentials_file(target, "first")
    write_credentials_file(target, "second")
    assert target.read_text() == "second"


def test_write_credentials_file_no_temp_leftovers(tmp_path: Path) -> None:
    """The atomic-write pattern uses a sibling tempfile — after the
    rename, only the target file remains."""

    target = tmp_path / "creds"
    write_credentials_file(target, "x")
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["creds"]


# ---------------------------------------------------------------------------
# GitCredentialsManager
# ---------------------------------------------------------------------------


async def test_manager_ensure_mints_and_writes_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ensure`` mints a token (HTTP round-trip stubbed) and writes
    it to the credentials file before returning."""

    target = tmp_path / "creds"
    monkeypatch.setenv("COLONY_GIT_CREDENTIALS_FILE", str(target))

    # ``GitCredentialsManager`` reads the path at __init__ time via
    # ``_credentials_path()``; force a fresh instance picking up the
    # patched env var.
    await GitCredentialsManager.reset_for_tests()

    priv = _make_rsa_key_pem()
    transport = httpx.MockTransport(_token_handler_factory("ghs_abc"))

    manager = GitCredentialsManager.get()
    # Inject our MockTransport-backed client into the manager so the
    # TokenCache hits the stub instead of real GitHub.
    manager._http_client = httpx.AsyncClient(transport=transport)
    # Build the TokenCache by hand against our http client; ``ensure``
    # would otherwise build its own fresh AsyncClient.
    from polymathera.colony.agents.patterns.capabilities._github.auth import (
        GitHubAppAuth, TokenCache,
    )
    manager._token_cache = TokenCache(
        app_auth=GitHubAppAuth(app_id="42", private_key_pem=priv),
        installation_id="777",
        client=manager._http_client,
    )
    manager._installation_id = "777"
    await manager._mint_and_write()

    assert target.read_text() == "ghs_abc"


async def test_manager_ensure_idempotent_for_same_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling ``ensure`` twice with the same installation_id is a
    no-op on the second call (no second mint round-trip)."""

    target = tmp_path / "creds"
    monkeypatch.setenv("COLONY_GIT_CREDENTIALS_FILE", str(target))
    await GitCredentialsManager.reset_for_tests()

    priv = _make_rsa_key_pem()
    mint_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        mint_count["n"] += 1
        return httpx.Response(201, json={
            "token": f"ghs_{mint_count['n']}",
            "expires_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() + 60 * 60),
            ),
        })

    manager = GitCredentialsManager.get()
    # Patch the httpx client so MockTransport intercepts the
    # token-exchange POSTs the TokenCache makes.
    original_client_cls = httpx.AsyncClient

    def _mock_client(*args, **kwargs):
        return original_client_cls(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        "polymathera.colony.distributed.git_credentials.httpx.AsyncClient",
        _mock_client,
    )

    await manager.ensure(
        app_id="42", private_key_pem=priv, installation_id="777",
        refresh_interval_s=3600,
    )
    first_count = mint_count["n"]
    assert first_count == 1
    assert target.read_text() == "ghs_1"

    # Second call with the same installation_id: no new mint.
    await manager.ensure(
        app_id="42", private_key_pem=priv, installation_id="777",
        refresh_interval_s=3600,
    )
    assert mint_count["n"] == first_count


async def test_manager_ensure_restarts_on_different_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different ``installation_id`` → tear down the running task,
    start a new one against the new installation."""

    target = tmp_path / "creds"
    monkeypatch.setenv("COLONY_GIT_CREDENTIALS_FILE", str(target))
    await GitCredentialsManager.reset_for_tests()

    priv = _make_rsa_key_pem()
    seen_installations: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        # Path looks like ``/app/installations/<id>/access_tokens``.
        parts = request.url.path.split("/")
        seen_installations.append(parts[3])
        return httpx.Response(201, json={
            "token": f"ghs_{parts[3]}",
            "expires_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(time.time() + 60 * 60),
            ),
        })

    original_client_cls = httpx.AsyncClient

    def _mock_client(*args, **kwargs):
        return original_client_cls(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        "polymathera.colony.distributed.git_credentials.httpx.AsyncClient",
        _mock_client,
    )

    manager = GitCredentialsManager.get()
    await manager.ensure(
        app_id="42", private_key_pem=priv, installation_id="777",
        refresh_interval_s=3600,
    )
    assert target.read_text() == "ghs_777"

    await manager.ensure(
        app_id="42", private_key_pem=priv, installation_id="888",
        refresh_interval_s=3600,
    )
    assert target.read_text() == "ghs_888"
    assert seen_installations == ["777", "888"]


async def test_manager_stop_cancels_refresh_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``stop`` cancels the background refresh task + closes the
    HTTP client; no lingering tasks after teardown."""

    target = tmp_path / "creds"
    monkeypatch.setenv("COLONY_GIT_CREDENTIALS_FILE", str(target))
    await GitCredentialsManager.reset_for_tests()

    priv = _make_rsa_key_pem()
    original_client_cls = httpx.AsyncClient

    def _mock_client(*args, **kwargs):
        return original_client_cls(
            transport=httpx.MockTransport(_token_handler_factory()),
        )
    monkeypatch.setattr(
        "polymathera.colony.distributed.git_credentials.httpx.AsyncClient",
        _mock_client,
    )

    manager = GitCredentialsManager.get()
    await manager.ensure(
        app_id="42", private_key_pem=priv, installation_id="777",
        refresh_interval_s=3600,
    )
    task = manager._refresh_task
    assert task is not None
    assert not task.done()

    await manager.stop()
    assert task.done() or task.cancelled()
    assert manager._refresh_task is None
    assert manager._http_client is None


# ---------------------------------------------------------------------------
# ensure_git_credentials_from_agent_metadata — public entry point
# ---------------------------------------------------------------------------


def _agent_metadata(installation_id: str | None) -> Any:
    meta = MagicMock()
    meta.parameters = {
        "github_identity": {
            "tenant_installation_id": installation_id,
        },
    }
    return meta


def _patched_github_auth_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    app_id: str = "42",
    private_key_pem: str = "",
) -> None:
    """Replace ``get_github_auth_config`` so tests don't depend on
    the live ConfigurationManager."""

    class _Stub:
        def __init__(self, app_id: str, private_key_pem: str):
            self.app_id = app_id
            self.private_key_pem = private_key_pem

    async def _fake_get():
        return _Stub(app_id=app_id, private_key_pem=private_key_pem)

    from polymathera.colony.agents import configs as configs_mod
    monkeypatch.setattr(
        configs_mod, "get_github_auth_config", _fake_get,
    )


async def test_ensure_no_op_when_no_installation_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """No installation_id in metadata → no mint, no file write."""

    target = tmp_path / "creds"
    monkeypatch.setenv("COLONY_GIT_CREDENTIALS_FILE", str(target))
    priv = _make_rsa_key_pem()
    _patched_github_auth_config(monkeypatch, private_key_pem=priv)

    await ensure_git_credentials_from_agent_metadata(
        _agent_metadata(installation_id=None),
    )
    assert not target.exists()


async def test_ensure_no_op_when_no_app_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """No deploy-wide App credentials → no mint, no file write."""

    target = tmp_path / "creds"
    monkeypatch.setenv("COLONY_GIT_CREDENTIALS_FILE", str(target))
    _patched_github_auth_config(monkeypatch, app_id="", private_key_pem="")

    await ensure_git_credentials_from_agent_metadata(
        _agent_metadata(installation_id="777"),
    )
    assert not target.exists()


async def test_ensure_no_op_when_metadata_missing_github_identity_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Metadata without the ``github_identity`` key (legacy agent
    metadata) → silent no-op, not an AttributeError."""

    target = tmp_path / "creds"
    monkeypatch.setenv("COLONY_GIT_CREDENTIALS_FILE", str(target))
    priv = _make_rsa_key_pem()
    _patched_github_auth_config(monkeypatch, private_key_pem=priv)

    meta = MagicMock()
    meta.parameters = {"design_monorepo_url": "https://x"}
    await ensure_git_credentials_from_agent_metadata(meta)
    assert not target.exists()


async def test_ensure_for_installation_no_op_when_id_falsy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Direct ``ensure_git_credentials_for_installation`` entry point
    (used by the dashboard) is the same no-op when the installation
    id is missing — falsy values short-circuit before the App config
    lookup."""

    target = tmp_path / "creds"
    monkeypatch.setenv("COLONY_GIT_CREDENTIALS_FILE", str(target))
    priv = _make_rsa_key_pem()
    _patched_github_auth_config(monkeypatch, private_key_pem=priv)

    for falsy in (None, "", 0):
        await ensure_git_credentials_for_installation(falsy)  # type: ignore[arg-type]
    assert not target.exists()


async def test_ensure_for_installation_no_op_when_no_app_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Dashboard path with installation_id set but deploy-wide App
    credentials missing → silent no-op (same shape as the agent
    path)."""

    target = tmp_path / "creds"
    monkeypatch.setenv("COLONY_GIT_CREDENTIALS_FILE", str(target))
    _patched_github_auth_config(monkeypatch, app_id="", private_key_pem="")

    await ensure_git_credentials_for_installation("777")
    assert not target.exists()
