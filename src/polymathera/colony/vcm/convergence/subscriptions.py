"""``PageSubscription`` — typed registration record for the runtime.

A capability registers a subscription with the runtime by supplying:

- ``predicate`` — what to fire on (``PageMetadataPredicate``).
- ``dispatch_scope`` + ``dispatch_key`` — where the runtime writes the
  dispatch event on the blackboard. The capability has its own
  ``@event_handler`` listening on this scope/key, so the dispatch
  flows through normal blackboard plumbing.
- ``declared_outputs`` — the set of predicates this subscription is
  expected to *write into*. Used by the dependency-aware scheduler
  (master §5.2 mechanism 2) to topo-sort within a wave: a
  subscription whose predicate matches the *output* predicate of
  another runs *after* it.
- ``tolerance`` — for numeric capabilities, the convergence damper
  (master §5.2 mechanism 3) skips dispatch when ``||new − last|| ≤
  tolerance``.
- ``capability_key`` / ``agent_id`` — identifying labels for the
  change-feed (master §5.4) and access-control checks.

A ``PageSubscription`` is immutable; the runtime tracks lifetimes
through a separate ``SubscriptionRegistry`` (next module).
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .predicates import PageMetadataPredicate


class NumericTolerance(BaseModel):
    """Damping configuration for numeric capabilities (master §5.2 mech. 3).

    Two modes:

    - ``"absolute"`` — skip dispatch when ``|new − last| ≤ value``.
    - ``"relative"`` — skip when
      ``|new − last| / max(|last|, epsilon) ≤ value``.

    ``epsilon`` guards the relative-mode division by zero. Must be
    positive.
    """

    model_config = ConfigDict(frozen=True)

    mode: Literal["absolute", "relative"] = "absolute"
    value: float = Field(gt=0.0)
    epsilon: float = Field(default=1e-12, gt=0.0)


class PageSubscription(BaseModel):
    """One registered subscription on the convergence runtime."""

    model_config = ConfigDict(frozen=True)

    subscription_id: str = Field(
        default_factory=lambda: f"sub_{uuid.uuid4().hex[:12]}",
        description="Stable id; the registry keys subscriptions by this.",
    )
    predicate: PageMetadataPredicate
    dispatch_scope: str = Field(
        description="Blackboard scope id the dispatch event is written to.",
    )
    dispatch_key: str = Field(
        description=(
            "Blackboard key (within ``dispatch_scope``) the runtime "
            "writes when the predicate matches. Capability subscribes "
            "to this key with @event_handler."
        ),
    )
    declared_outputs: tuple[PageMetadataPredicate, ...] = Field(
        default_factory=tuple,
        description=(
            "Set of predicates this subscription is expected to write "
            "into. Used by the dependency-aware scheduler — within a "
            "wave, a subscription that *reads* a predicate runs after "
            "any subscription that *writes* it."
        ),
    )
    tolerance: NumericTolerance | None = None
    capability_key: str = Field(
        default="",
        description="Identifying label of the subscribing capability.",
    )
    agent_id: str | None = Field(
        default=None,
        description="Owning agent id (for access control + the change feed).",
    )

    @property
    def dispatch_topic(self) -> str:
        """Convenience: the absolute pattern that subscribers should
        match against (``<scope>:<key>``). Matches the format the
        blackboard transport uses internally."""
        return f"{self.dispatch_scope}:{self.dispatch_key}"


__all__ = ("PageSubscription", "NumericTolerance")
