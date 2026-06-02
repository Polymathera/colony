"""Mention routing — colony-singleton parser that watches inbound
GitHub events for ``@colony`` / ``@polymath`` mentions and emits
:class:`MentionEventProtocol` writes for each one.

P10 v1 scope:

- Bare ``@colony`` + ``@polymath`` route to the same event surface
  (no per-handle dispatch).
- ``@colony-<name>`` / ``@polymath-<name>`` ARE captured but route
  identically to the bare form (the ``mention_kind`` payload field
  preserves the suffix so a future per-handle dispatcher can branch).
- No ``.colony/mention_routes.yaml`` (operator-extensible routing is
  follow-up).
- No LLM-driven response handler (the InteractionLog write-through
  records the mention; "Colony LLM-judges intent and responds" is a
  separate phase).
"""

from __future__ import annotations

from .capability import MentionRoutingCapability
from .parser import MENTION_RE, ParsedMention, parse_mentions


__all__ = (
    "MENTION_RE",
    "MentionRoutingCapability",
    "ParsedMention",
    "parse_mentions",
)
