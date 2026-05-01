"""The convergence runtime — turns a static page graph into a live design substrate.

Public API:

- ``PageMetadataPredicate`` — typed match expression over page metadata.
- ``PageSubscription`` / ``NumericTolerance`` — subscription record.
- ``SubscriptionIndex`` — fast lookup of subscriptions by event metadata.
- ``ConvergenceDamper`` — numeric tolerance check.
- ``WriteRateLimiter`` — per-page write throttle.
- ``ConvergenceRuntime`` — pure dispatch logic.
- ``ConvergenceRuntimeDeployment`` — Ray-serving singleton wrapping the runtime.
- ``ConvergenceStatus`` / ``ConvergenceCounters`` / ``ChangeFeedEntry`` — surfaces.

See `colony_docs/markdown/apps/design_automation_architecture.md` §5 and
`colony/phase_c4_convergence_runtime_progress.md` for the discipline this
package implements.
"""

from __future__ import annotations

from .damping import ConvergenceDamper
from .deployment import ConvergenceRuntimeDeployment
from .index import SubscriptionIndex, SubscriptionRegistryFull
from .predicates import EdgeReachResolver, PageMetadataPredicate
from .rate_limit import WriteRateLimiter
from .runtime import (
    ChangeFeedEntry,
    ConvergenceCounters,
    ConvergenceRuntime,
    ConvergenceState,
    ConvergenceStatus,
)
from .subscriptions import NumericTolerance, PageSubscription


__all__ = (
    "PageMetadataPredicate",
    "EdgeReachResolver",
    "PageSubscription",
    "NumericTolerance",
    "SubscriptionIndex",
    "SubscriptionRegistryFull",
    "ConvergenceDamper",
    "WriteRateLimiter",
    "ConvergenceRuntime",
    "ConvergenceState",
    "ConvergenceStatus",
    "ConvergenceCounters",
    "ChangeFeedEntry",
    "ConvergenceRuntimeDeployment",
)
