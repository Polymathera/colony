"""GitHub webhook receiver — dashboard-side ingress for inbound
GitHub events.

P9 (this module) ships the receiver as a drop-in replacement for the
P8a agent-side poller. Both emit ``GitHubEventProtocol.*`` writes to
the colony blackboard so downstream subscribers (P8b InteractionLog,
future P10 mention routing) don't care which mode is active.

Key pieces:

- :func:`ensure_github_webhook_schema` — Postgres dedup table
  (``github_webhook_deliveries``).
- :mod:`.normalizer` — pure GitHub payload → ``GitHubEventProtocol``
  ``(key, value)`` translation; testable in isolation against canned
  webhook payloads.
- :mod:`.publisher` — given a tenant_id + normalized events, fan-out
  to every colony's blackboard inside that tenant.
- :mod:`colony.web_ui.backend.routers.github_webhook` — the
  ``POST /api/v1/github/webhook`` FastAPI route: HMAC verify + dedup
  + normalize + publish.
"""

from __future__ import annotations

from .schema import ensure_github_webhook_schema


__all__ = ("ensure_github_webhook_schema",)
