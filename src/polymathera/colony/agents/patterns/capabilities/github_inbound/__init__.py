"""GitHub inbound — poll-mode capability.

Polls GitHub via the GraphQL API on a cron-like cadence per
``.colony/github_inbound.yaml`` config and emits writes at
``GitHubEventProtocol.*`` keys on the colony-scoped blackboard.

Mounted on the system session's ``SessionAgent`` (P8-0 foundation),
NOT on per-user chat sessions — the poller is a colony-singleton,
one instance per (tenant, colony) pair.

Webhook mode is P9; this package raises a clean P9-pointer error if
the operator's YAML sets ``mode: webhook``.
"""

from __future__ import annotations

from .capability import GitHubInboundCapability
from .config import GitHubInboundConfig
from .schema import ensure_github_inbound_schema


__all__ = (
    "GitHubInboundCapability",
    "GitHubInboundConfig",
    "ensure_github_inbound_schema",
)
