"""Symmetric encryption for at-rest secrets in the dashboard's
postgres schema.

First consumer: the ``tenants.bot_token_encrypted`` column — GitLab
group access tokens / Bitbucket workspace API tokens / future
operator-supplied bot credentials whose plaintext must NOT live in
postgres unprotected. GitHub does not use this path (installation
tokens are auto-minted from the deploy-wide App private key + the
per-tenant numeric installation id — see ``_github/factory.py``).

Why Fernet (cryptography.fernet):

- AES-128-CBC + HMAC-SHA256 in one tagged primitive. Authenticated;
  decryption fails closed on tamper / key mismatch.
- Single 32-byte url-safe-base64 key. Easy to rotate (re-encrypt all
  rows once); easy to ship via env var.
- Already a transitive dep in the colony lockfile (PyJWT pulls it
  in for asymmetric JWT signing).

Key lifecycle:

- Operator generates one key per Colony deployment with
  ``python -c "from cryptography.fernet import Fernet;
  print(Fernet.generate_key().decode())"`` and stores it in
  ``cli/deploy/.env`` under ``COLONY_SECRETS_FERNET_KEY``.
- The dashboard reads it at process start via :func:`get_fernet`.
- Re-keying = generating a new key, decrypting every row with the
  old, encrypting with the new, and replacing the env value. Not in
  scope for this module (one-key-at-a-time only).

This module is dashboard-only for now. When PR 7 adds GitLab and
agents need to decrypt bot tokens during ``mint_bot_credentials``,
the key gets exported into ray-head/worker too (env-list change in
``cli/deploy/docker/docker-compose.yml`` + ``cli/deploy/config.py``).
"""

from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken


_ENV_KEY = "COLONY_SECRETS_FERNET_KEY"


class SecretsConfigError(RuntimeError):
    """``COLONY_SECRETS_FERNET_KEY`` is missing or malformed at
    process start. Surfaces with a setup-doc-pointer message so
    operators can self-resolve without reading code."""


class SecretDecryptError(RuntimeError):
    """Decryption of a stored ciphertext failed — typically because
    the key was rotated without re-encrypting the rows, or the
    ciphertext was corrupted. Distinct from :class:`SecretsConfigError`
    so callers can tell "operator hasn't configured the key" from
    "this specific row is unreadable"."""


def _generate_setup_pointer() -> str:
    return (
        "Generate one with: "
        "python -c \"from cryptography.fernet import Fernet; "
        "print(Fernet.generate_key().decode())\" "
        "and set it as COLONY_SECRETS_FERNET_KEY in "
        "cli/deploy/.env (see .env.template)."
    )


def get_fernet() -> Fernet:
    """Load + return a Fernet primed with
    ``COLONY_SECRETS_FERNET_KEY``. Raises :class:`SecretsConfigError`
    when the env var is absent or malformed.

    Not cached — each call re-reads ``os.environ`` so operators can
    swap the key in dev without a process restart (and so tests can
    monkeypatch the env var per case). The cost is microseconds per
    encrypt/decrypt; not worth caching the few-byte key state."""
    raw = os.environ.get(_ENV_KEY, "").strip()
    if not raw:
        raise SecretsConfigError(
            f"{_ENV_KEY} is not set. {_generate_setup_pointer()}",
        )
    try:
        return Fernet(raw.encode())
    except (ValueError, TypeError) as exc:
        raise SecretsConfigError(
            f"{_ENV_KEY} is malformed: {exc}. "
            f"Expected 32 url-safe base64 bytes. "
            f"{_generate_setup_pointer()}",
        ) from exc


def encrypt_value(plaintext: str) -> str:
    """Encrypt ``plaintext`` with the configured key, return a
    url-safe base64 ascii string suitable for a postgres TEXT column.

    Raises :class:`SecretsConfigError` if the key is misconfigured —
    fail fast at the call site rather than silently storing
    plaintext."""
    return get_fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a string produced by :func:`encrypt_value`. Raises
    :class:`SecretDecryptError` for any failure shape (bad MAC, key
    rotation, corrupted bytes); :class:`SecretsConfigError` if the
    key isn't configured at all.

    Returns the original utf-8 plaintext on success."""
    fernet = get_fernet()
    try:
        return fernet.decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise SecretDecryptError(
            "Failed to decrypt stored secret — likely a key rotation "
            "without re-encryption, or the ciphertext was tampered "
            "with.",
        ) from exc


__all__ = (
    "SecretDecryptError",
    "SecretsConfigError",
    "decrypt_value",
    "encrypt_value",
    "get_fernet",
)
