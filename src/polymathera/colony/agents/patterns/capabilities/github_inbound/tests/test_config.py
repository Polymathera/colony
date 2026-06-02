"""Tests for ``.colony/github_inbound.yaml`` parsing + validation."""

from __future__ import annotations

import pytest

from polymathera.colony.agents.patterns.capabilities.github_inbound import (
    GitHubInboundConfig,
)


def test_minimal_config_uses_defaults() -> None:
    """An empty ``github_inbound`` block parses to all-defaults: mode
    poll, 60s interval, no repos."""

    text = "schema_version: 1\ngithub_inbound: {}\n"
    cfg = GitHubInboundConfig.load_from_yaml_text(text)
    assert cfg.github_inbound.mode == "poll"
    assert cfg.github_inbound.poll_interval_seconds == 60
    assert cfg.github_inbound.poll_repos == []


def test_explicit_poll_config() -> None:
    text = (
        "schema_version: 1\n"
        "github_inbound:\n"
        "  mode: poll\n"
        "  poll_interval_seconds: 30\n"
        "  poll_repos:\n"
        "    - acme/widgets\n"
        "    - acme/sprockets\n"
    )
    cfg = GitHubInboundConfig.load_from_yaml_text(text)
    assert cfg.github_inbound.poll_interval_seconds == 30
    assert cfg.github_inbound.poll_repos == ["acme/widgets", "acme/sprockets"]


def test_webhook_mode_accepted_post_p9() -> None:
    """P9 lifted the parse-time rejection of ``mode: webhook``. Now
    both modes are valid; the agent-side capability quiesces its
    poll loop on webhook + the dashboard route is the active
    surface."""

    text = (
        "schema_version: 1\n"
        "github_inbound:\n"
        "  mode: webhook\n"
        "  poll_repos:\n"
        "    - acme/widgets\n"
    )
    cfg = GitHubInboundConfig.load_from_yaml_text(text)
    assert cfg.github_inbound.mode == "webhook"


def test_schema_version_mismatch_rejected() -> None:
    text = "schema_version: 99\ngithub_inbound: {}\n"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        GitHubInboundConfig.load_from_yaml_text(text)


def test_malformed_repo_shape_rejected() -> None:
    """``poll_repos`` entries must be ``owner/repo`` — exactly one
    slash. Reject other shapes so a typo can't silently cause a
    GraphQL error every tick."""

    bad_inputs = [
        "no_slash",                # zero slashes
        "owner/repo/extra",        # two slashes
        "/repo",                   # empty owner
        "owner/",                  # empty name
    ]
    for bad in bad_inputs:
        text = (
            "schema_version: 1\n"
            "github_inbound:\n"
            "  poll_repos:\n"
            f"    - {bad}\n"
        )
        with pytest.raises(ValueError):
            GitHubInboundConfig.load_from_yaml_text(text)


def test_invalid_yaml_raises_clear_error() -> None:
    text = "this is: not: valid: yaml: ::\n"
    with pytest.raises(ValueError, match="invalid YAML"):
        GitHubInboundConfig.load_from_yaml_text(text)


def test_extra_keys_rejected() -> None:
    """Pydantic ``extra='forbid'`` — typos in optional keys (e.g.
    ``poll_intervall_seconds``) surface at parse time."""

    text = (
        "schema_version: 1\n"
        "github_inbound:\n"
        "  poll_intervall_seconds: 30\n"  # typo
    )
    with pytest.raises(ValueError):
        GitHubInboundConfig.load_from_yaml_text(text)


def test_interval_floor_enforced() -> None:
    """``poll_interval_seconds`` floored at 10s to avoid hammering
    GitHub. Below floor → validation error."""

    text = (
        "schema_version: 1\n"
        "github_inbound:\n"
        "  poll_interval_seconds: 1\n"
    )
    with pytest.raises(ValueError):
        GitHubInboundConfig.load_from_yaml_text(text)
