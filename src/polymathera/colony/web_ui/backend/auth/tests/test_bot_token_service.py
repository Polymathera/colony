"""Tests for ``set_tenant_bot_token`` / ``get_tenant_bot_token``.

The discipline these helpers enforce: plaintext bot tokens NEVER hit
postgres unencrypted, and they NEVER leak back from a getter without
going through Fernet first.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from cryptography.fernet import Fernet

from polymathera.colony.web_ui.backend.auth import service as auth_service


def _set_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "COLONY_SECRETS_FERNET_KEY", Fernet.generate_key().decode(),
    )


class _StubConn:
    """In-memory ``tenants`` table — only the columns these helpers
    touch (encrypted token + expires_at + the row's existence)."""
    def __init__(self, rows: dict[str, dict[str, Any]]) -> None:
        self._rows = rows

    async def execute(self, sql: str, *params: Any) -> str:
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("UPDATE tenants SET bot_token_encrypted"):
            encrypted, expires_at, tenant_id = params
            if tenant_id not in self._rows:
                return "UPDATE 0"
            self._rows[tenant_id]["bot_token_encrypted"] = encrypted
            self._rows[tenant_id]["bot_token_expires_at"] = expires_at
            return "UPDATE 1"
        raise AssertionError(f"unexpected execute: {sql_norm}")

    async def fetchrow(self, sql: str, *params: Any) -> Any:
        sql_norm = " ".join(sql.split())
        if sql_norm.startswith("SELECT bot_token_encrypted"):
            (tenant_id,) = params
            return self._rows.get(tenant_id)
        raise AssertionError(f"unexpected fetchrow: {sql_norm}")


class _StubPool:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    @asynccontextmanager
    async def acquire(self):
        yield _StubConn(self.rows)


@pytest.fixture
def pool() -> _StubPool:
    return _StubPool()


# ---------------------------------------------------------------------
# set_tenant_bot_token


@pytest.mark.asyncio
async def test_set_stores_encrypted_value(
    monkeypatch: pytest.MonkeyPatch, pool: _StubPool,
) -> None:
    """The persisted column value MUST NOT equal the plaintext —
    proves Fernet ran before INSERT."""
    _set_key(monkeypatch)
    pool.rows["tenant_a"] = {}

    await auth_service.set_tenant_bot_token(
        pool, tenant_id="tenant_a",
        plaintext_token="glpat-secret-AAAAAAAA",
        expires_at=None,
    )
    stored = pool.rows["tenant_a"]["bot_token_encrypted"]
    assert stored is not None
    assert stored != "glpat-secret-AAAAAAAA"
    # Round-trip via the getter proves it's decryptable.
    out = await auth_service.get_tenant_bot_token(pool, tenant_id="tenant_a")
    assert out is not None
    assert out["token"] == "glpat-secret-AAAAAAAA"


@pytest.mark.asyncio
async def test_set_persists_expires_at(
    monkeypatch: pytest.MonkeyPatch, pool: _StubPool,
) -> None:
    _set_key(monkeypatch)
    pool.rows["tenant_a"] = {}
    sentinel_dt = object()  # opaque — column type is just passed through
    await auth_service.set_tenant_bot_token(
        pool, tenant_id="tenant_a",
        plaintext_token="x", expires_at=sentinel_dt,
    )
    assert pool.rows["tenant_a"]["bot_token_expires_at"] is sentinel_dt


@pytest.mark.asyncio
async def test_set_with_none_clears_token(
    monkeypatch: pytest.MonkeyPatch, pool: _StubPool,
) -> None:
    """Setting ``plaintext_token=None`` is the operator-revocation
    path. Result column MUST be NULL, not the encryption of an empty
    string."""
    _set_key(monkeypatch)
    pool.rows["tenant_a"] = {
        "bot_token_encrypted": "old-cipher",
        "bot_token_expires_at": "old-dt",
    }
    await auth_service.set_tenant_bot_token(
        pool, tenant_id="tenant_a",
        plaintext_token=None, expires_at=None,
    )
    assert pool.rows["tenant_a"]["bot_token_encrypted"] is None
    assert pool.rows["tenant_a"]["bot_token_expires_at"] is None


@pytest.mark.asyncio
async def test_set_raises_keyerror_when_tenant_missing(
    monkeypatch: pytest.MonkeyPatch, pool: _StubPool,
) -> None:
    _set_key(monkeypatch)
    with pytest.raises(KeyError, match="not found"):
        await auth_service.set_tenant_bot_token(
            pool, tenant_id="nope", plaintext_token="x",
        )


@pytest.mark.asyncio
async def test_set_raises_config_error_when_key_missing(
    monkeypatch: pytest.MonkeyPatch, pool: _StubPool,
) -> None:
    """Misconfigured deployment must fail at the write boundary, not
    silently store plaintext or skip the write."""
    monkeypatch.delenv("COLONY_SECRETS_FERNET_KEY", raising=False)
    pool.rows["tenant_a"] = {}
    from polymathera.colony.web_ui.backend.auth.secrets import SecretsConfigError
    with pytest.raises(SecretsConfigError):
        await auth_service.set_tenant_bot_token(
            pool, tenant_id="tenant_a", plaintext_token="x",
        )


# ---------------------------------------------------------------------
# get_tenant_bot_token


@pytest.mark.asyncio
async def test_get_returns_none_when_no_token(
    monkeypatch: pytest.MonkeyPatch, pool: _StubPool,
) -> None:
    _set_key(monkeypatch)
    pool.rows["tenant_a"] = {
        "bot_token_encrypted": None,
        "bot_token_expires_at": None,
    }
    assert await auth_service.get_tenant_bot_token(
        pool, tenant_id="tenant_a",
    ) is None


@pytest.mark.asyncio
async def test_get_returns_none_when_tenant_missing(
    monkeypatch: pytest.MonkeyPatch, pool: _StubPool,
) -> None:
    _set_key(monkeypatch)
    assert await auth_service.get_tenant_bot_token(
        pool, tenant_id="nope",
    ) is None


@pytest.mark.asyncio
async def test_get_raises_decrypt_error_after_key_rotation(
    monkeypatch: pytest.MonkeyPatch, pool: _StubPool,
) -> None:
    """Operator rotates the env key without re-encrypting rows —
    getter must surface a recoverable error (the caller decides
    whether to log+skip or hard-fail)."""
    _set_key(monkeypatch)
    pool.rows["tenant_a"] = {}
    await auth_service.set_tenant_bot_token(
        pool, tenant_id="tenant_a", plaintext_token="x",
    )
    # Rotate the env key.
    monkeypatch.setenv("COLONY_SECRETS_FERNET_KEY", Fernet.generate_key().decode())
    from polymathera.colony.web_ui.backend.auth.secrets import SecretDecryptError
    with pytest.raises(SecretDecryptError):
        await auth_service.get_tenant_bot_token(pool, tenant_id="tenant_a")
