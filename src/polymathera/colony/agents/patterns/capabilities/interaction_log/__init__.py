"""Cross-channel interaction log — Postgres-backed write-through for
colony-scoped blackboard events, plus read helpers for the dashboard.

P8b v1 covers ``GitHubEventProtocol.*`` only (the COLONY-scoped
inbound surface from P8a + future P9 webhooks). SessionChat
(SESSION-scoped) and ActionPolicyLifecycle (AGENT-scoped) coverage
is deferred to follow-ups — needs either cross-scope subscription
or per-scope mounting.
"""

from __future__ import annotations

from .capability import InteractionLogCapability
from .schema import ensure_interaction_log_schema


__all__ = (
    "InteractionLogCapability",
    "ensure_interaction_log_schema",
)
