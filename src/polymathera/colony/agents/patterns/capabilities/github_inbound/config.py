"""``.colony/github_inbound.yaml`` schema + loader.

Both ``mode: poll`` (P8a, agent-process tick loop) and ``mode: webhook``
(P9, dashboard-process receiver) are accepted. The mode dictates which
surface fires for the colony — the agent-side ``GitHubInboundCapability``
quiesces its poll loop when ``mode: webhook``, leaving the
dashboard's ``POST /api/v1/github/webhook`` route as the active path.
The downstream blackboard protocols are mode-agnostic so subscribers
(P8b InteractionLog, future P10 mention routing) work either way.
"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = 1


class GitHubInboundSection(BaseModel):
    """The ``github_inbound:`` block inside the YAML."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["poll", "webhook"] = Field(
        default="poll",
        description=(
            "``poll`` (P8a): the colony's GitHubInboundCapability ticks "
            "every ``poll_interval_seconds`` and emits GitHubEventProtocol "
            "writes. ``webhook`` (P9, not yet shipped): the dashboard's "
            "``POST /github/webhook`` receiver normalizes signed payloads "
            "into the same protocol."
        ),
    )
    poll_interval_seconds: int = Field(
        default=60,
        ge=10,
        description=(
            "Tick interval in seconds. Operator-tunable; floor at 10s "
            "to avoid hammering the GitHub API. Default 60s — well "
            "under GitHub's 5000-point/hr ceiling for any reasonable "
            "set of repos."
        ),
    )
    poll_repos: list[str] = Field(
        default_factory=list,
        description=(
            "List of ``owner/repo`` strings to poll. Empty list = no "
            "polling (still legal; the capability stays alive but the "
            "tick is a no-op)."
        ),
    )

    @field_validator("poll_repos")
    @classmethod
    def _validate_repo_shape(cls, v: list[str]) -> list[str]:
        for repo in v:
            if not isinstance(repo, str) or "/" not in repo or repo.count("/") != 1:
                raise ValueError(
                    f"poll_repos entry {repo!r} must have shape "
                    f"'owner/repo' (exactly one '/')."
                )
            owner, name = repo.split("/", 1)
            if not owner or not name:
                raise ValueError(
                    f"poll_repos entry {repo!r} has empty owner or name."
                )
        return v


class GitHubInboundConfig(BaseModel):
    """Top-level ``.colony/github_inbound.yaml`` schema."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    github_inbound: GitHubInboundSection = Field(
        default_factory=GitHubInboundSection,
    )

    @classmethod
    def load_from_yaml_text(cls, text: str) -> "GitHubInboundConfig":
        """Parse a YAML string into a ``GitHubInboundConfig``.

        Raises:
            ValueError: malformed YAML, schema_version mismatch, mode
                set to ``webhook`` (with a P9-pointer message), or
                any pydantic validation failure.
        """

        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise ValueError(
                f"github_inbound.yaml: invalid YAML — {exc}"
            ) from exc

        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"github_inbound.yaml: unsupported schema_version "
                f"{version!r}; expected {SCHEMA_VERSION}."
            )

        return cls.model_validate(data)
