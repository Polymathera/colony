"""Pure regex + extraction for ``@colony`` / ``@polymath`` mentions.

Separated from :mod:`.capability` so the regex semantics + edge cases
are testable in isolation against canned bodies.

The regex follows design doc §10 but with the alternation ordered
**longer-first** so ``@colony-roadmap`` captures the full handle
instead of stopping at ``colony``:

    \\B@(colony-\\w+|polymath-\\w+|colony|polymath)\\b

Python's ``re`` matches alternatives left-to-right and stops at the
first one that holds, so the bare ``colony`` / ``polymath`` MUST come
AFTER ``colony-\\w+`` / ``polymath-\\w+``. Otherwise the bare branch
matches the ``colony`` prefix of ``@colony-roadmap`` + the ``\\b``
check passes (because ``-`` is a non-word char and ``y``→``-`` IS a
word boundary), the regex returns ``colony``, and the suffix is
silently dropped.

- ``\\B`` (non-word-boundary BEFORE ``@``) ensures the ``@`` is the
  start of the handle, not preceded by a word character — so
  ``email@colony.com`` does NOT match (the ``@`` is preceded by ``l``,
  a word char). A literal ``@`` at the start of a string, or after
  whitespace/punctuation, IS a non-word-boundary so it matches.
- ``\\b`` (word boundary AFTER the handle) anchors the right edge so
  ``@colonyfoo`` does NOT match (no boundary between ``y`` and ``f``).
- ``colony-\\w+`` / ``polymath-\\w+`` greedy-matches whatever word
  characters follow the hyphen. ``\\w`` is ``[A-Za-z0-9_]`` — does
  NOT include ``-``, so ``@colony-roadmap-experiments`` captures
  only ``colony-roadmap``. The chained-hyphen case is rare enough
  for v1; the trailing ``-experiments`` becomes a separate token
  in the body, harmless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


MENTION_RE = re.compile(
    r"\B@(colony-\w+|polymath-\w+|colony|polymath)\b",
)


@dataclass(frozen=True)
class ParsedMention:
    """One detected mention.

    ``handle`` is the raw captured string (``colony`` /
    ``polymath`` / ``colony-<name>`` / ``polymath-<name>``); callers
    record it on the emitted event so a future per-handle dispatcher
    can branch.

    ``offset`` is the character index of the ``@`` in the source
    body — useful for surfacing the surrounding line as
    ``requested_action_hint`` in a future LLM-judge follow-up.
    """

    handle: str
    offset: int


def parse_mentions(body: str | None) -> list[ParsedMention]:
    """Find every ``@colony`` / ``@polymath`` mention in ``body``.

    Returns an empty list for:
    - ``None`` / empty body
    - body with no matches

    Order is left-to-right in source-text order. Duplicates are
    preserved (two ``@colony`` mentions in one body → two
    ``ParsedMention`` entries) so the caller's emit-per-mention
    semantics is straightforward.
    """

    if not body:
        return []
    return [
        ParsedMention(handle=m.group(1), offset=m.start())
        for m in MENTION_RE.finditer(body)
    ]
