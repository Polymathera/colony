"""Targeted tests for ``agents/configs.py`` validators."""

from __future__ import annotations

from polymathera.colony.agents.configs import GitHubAuthConfig


def test_github_auth_config_normalises_escaped_pem() -> None:
    """Single-line PEMs with ``\\n`` escapes (the format docker-compose
    forces — see ``GitHubAuthConfig._normalize_pem``) are converted to
    real newlines so pyjwt sees a valid key."""

    escaped = (
        "-----BEGIN RSA PRIVATE KEY-----\\nBODY\\n"
        "-----END RSA PRIVATE KEY-----\\n"
    )
    cfg = GitHubAuthConfig(private_key_pem=escaped)
    assert cfg.private_key_pem == (
        "-----BEGIN RSA PRIVATE KEY-----\nBODY\n"
        "-----END RSA PRIVATE KEY-----\n"
    )


def test_github_auth_config_preserves_real_newlines_pem() -> None:
    """Multi-line PEMs (already-decoded form) pass through unchanged —
    idempotent."""

    real = (
        "-----BEGIN RSA PRIVATE KEY-----\nBODY\n"
        "-----END RSA PRIVATE KEY-----\n"
    )
    cfg = GitHubAuthConfig(private_key_pem=real)
    assert cfg.private_key_pem == real


def test_github_auth_config_leaves_empty_pem_alone() -> None:
    """Empty value (unconfigured deployment) is not touched."""

    cfg = GitHubAuthConfig(private_key_pem="")
    assert cfg.private_key_pem == ""
