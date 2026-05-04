"""``PageSubscription`` — typed registration record for the runtime.

A capability registers a subscription with the runtime by supplying:

- ``predicate`` — what to fire on (``PageMetadataPredicate``).
- ``dispatch_scope`` — the blackboard scope the runtime writes the
  dispatch event onto (typically the subscribing capability's own
  scope). The *key* shape is fixed by
  ``ConvergenceDispatchProtocol`` — the runtime uses the
  subscription's ``subscription_id`` as the correlator. The
  capability subscribes via
  ``@event_handler(pattern=ConvergenceDispatchProtocol.dispatch_pattern())``
  and routes by the parsed ``subscription_id``.
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
    def dispatch_key(self) -> str:
        """The scope-relative key the runtime writes for this
        subscription. Derived from
        ``ConvergenceDispatchProtocol.dispatch_key(subscription_id)``;
        the protocol owns the key shape so the runtime and subscribing
        capability never disagree."""
        from ...agents.blackboard.protocol import ConvergenceDispatchProtocol

        return ConvergenceDispatchProtocol.dispatch_key(self.subscription_id)

    @property
    def dispatch_topic(self) -> str:
        """The absolute pattern subscribers should match against
        (``<scope>:<key>``). Matches the format the blackboard
        transport uses internally."""

        return f"{self.dispatch_scope}:{self.dispatch_key}"


__all__ = ("PageSubscription", "NumericTolerance")
