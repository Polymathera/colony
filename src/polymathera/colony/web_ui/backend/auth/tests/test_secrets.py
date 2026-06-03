"""Tests for ``auth/secrets.py`` — Fernet encryption helpers."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from polymathera.colony.web_ui.backend.auth import secrets
from polymathera.colony.web_ui.backend.auth.secrets import (
    SecretDecryptError,
    SecretsConfigError,
    decrypt_value,
    encrypt_value,
    get_fernet,
)


def _set_key(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    monkeypatch.setenv("COLONY_SECRETS_FERNET_KEY", key)


def _fresh_key() -> str:
    return Fernet.generate_key().decode()


# ---------------------------------------------------------------------
# get_fernet — config validation


def test_get_fernet_raises_when_key_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COLONY_SECRETS_FERNET_KEY", raising=False)
    with pytest.raises(SecretsConfigError, match="not set"):
        get_fernet()


def test_get_fernet_raises_when_key_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_key(monkeypatch, "   ")
    with pytest.raises(SecretsConfigError, match="not set"):
        get_fernet()


def test_get_fernet_raises_when_key_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_key(monkeypatch, "not-a-real-fernet-key")
    with pytest.raises(SecretsConfigError, match="malformed"):
        get_fernet()


def test_get_fernet_constructs_when_key_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_key(monkeypatch, _fresh_key())
    f = get_fernet()
    # Smoke test the returned Fernet works.
    assert f.decrypt(f.encrypt(b"x")) == b"x"


def test_get_fernet_strips_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operators often paste keys with trailing newlines — strip them
    before constructing the primitive so the dashboard doesn't fail
    to start over an ergonomic mistake."""
    key = _fresh_key()
    _set_key(monkeypatch, f"  {key}  \n")
    get_fernet()  # would raise if the whitespace leaked into Fernet()


# ---------------------------------------------------------------------
# encrypt_value / decrypt_value — round-trip + tamper resistance


def test_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_key(monkeypatch, _fresh_key())
    plaintext = "glpat-AAAAAAAAAAAAAAAAAAAA"
    encrypted = encrypt_value(plaintext)
    assert encrypted != plaintext
    assert decrypt_value(encrypted) == plaintext


def test_encrypt_handles_unicode(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_key(monkeypatch, _fresh_key())
    plaintext = "Token-with-unicode-ünîçødé-emoji-🔒"
    assert decrypt_value(encrypt_value(plaintext)) == plaintext


def test_encrypt_is_nondeterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fernet includes a random IV — two encrypts of the same
    plaintext must produce different ciphertexts. Prevents
    ciphertext-equality from leaking plaintext equality."""
    _set_key(monkeypatch, _fresh_key())
    a = encrypt_value("same-token")
    b = encrypt_value("same-token")
    assert a != b


def test_decrypt_raises_on_tampered_ciphertext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_key(monkeypatch, _fresh_key())
    encrypted = encrypt_value("real")
    tampered = encrypted[:-2] + ("xx" if encrypted[-2:] != "xx" else "yy")
    with pytest.raises(SecretDecryptError):
        decrypt_value(tampered)


def test_decrypt_raises_after_key_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row encrypted under the OLD key is unreadable after the
    operator rotated the env var to a NEW key — surface as
    SecretDecryptError so callers can log + skip rather than crash."""
    _set_key(monkeypatch, _fresh_key())
    ciphertext = encrypt_value("plaintext-under-old-key")
    # Operator rotates the key in the env without re-encrypting rows.
    _set_key(monkeypatch, _fresh_key())
    with pytest.raises(SecretDecryptError):
        decrypt_value(ciphertext)


def test_decrypt_raises_config_error_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct error shape from SecretDecryptError so callers can
    tell "operator forgot to set the key" from "this specific row
    is bad"."""
    _set_key(monkeypatch, _fresh_key())
    ciphertext = encrypt_value("data")
    monkeypatch.delenv("COLONY_SECRETS_FERNET_KEY", raising=False)
    with pytest.raises(SecretsConfigError):
        decrypt_value(ciphertext)


# ---------------------------------------------------------------------
# get_fernet caching policy — recheck env each call


def test_get_fernet_rereads_env_on_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module deliberately does NOT cache the Fernet object —
    operators can swap keys in dev without a process restart, and
    tests can monkeypatch the env per case. Verify the policy: two
    different keys consecutively → different Fernet instances + the
    second can't decrypt the first's output."""
    key_one = _fresh_key()
    key_two = _fresh_key()
    _set_key(monkeypatch, key_one)
    encrypted = encrypt_value("hidden")

    _set_key(monkeypatch, key_two)
    # Second instance picks up the new key; can't decrypt the first.
    with pytest.raises(SecretDecryptError):
        decrypt_value(encrypted)

    # And can round-trip under the new key.
    fresh = encrypt_value("under-key-two")
    assert decrypt_value(fresh) == "under-key-two"


def test_module_exports() -> None:
    """Lock the public surface so accidental private leakage shows up
    in review."""
    assert set(secrets.__all__) == {
        "SecretDecryptError",
        "SecretsConfigError",
        "decrypt_value",
        "encrypt_value",
        "get_fernet",
    }
