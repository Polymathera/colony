"""Hardcoded per-plan entitlement defaults + the merge helper.

Plan structure deliberately lives in code, not in the DB: it forces
plan-shape changes to go through code review and avoids the stale-row
trap where an old tenant is billed under terms that no longer exist.
When real customers arrive (and the structure stabilises), this
becomes a versioned plan-catalog table — see plan §9.6.

The numbers below are placeholders to exercise the enforcement
plumbing; they will be revisited once we know what real customers
buy. ``dev`` is reserved for the ``COLONY_DEV_LICENSED_INSTALLATIONS``
seed (§9.4) — operators give themselves all-features-no-caps so the
local dev loop isn't quota-blocked.
"""

from __future__ import annotations

from typing import Any


# Every feature flag the plumbing currently recognises. New features
# add a string here; capability-mount checks read this list. UI hides
# capabilities whose feature isn't in the tenant's entitlements.
_ALL_FEATURES: tuple[str, ...] = (
    "chat",
    "github_inbound",
    "interaction_log",
    "mention_routing",
)


PLAN_DEFAULTS: dict[str, dict[str, Any]] = {
    "free": {
        "max_users": 3,
        "max_colonies_per_tenant": 1,
        "max_concurrent_sessions": 2,
        "max_monthly_llm_tokens": 100_000,
        "features": ["chat"],
    },
    "team": {
        "max_users": 10,
        "max_colonies_per_tenant": 5,
        "max_concurrent_sessions": 10,
        "max_monthly_llm_tokens": 1_000_000,
        "features": ["chat", "github_inbound", "interaction_log"],
    },
    "business": {
        "max_users": 50,
        "max_colonies_per_tenant": 25,
        "max_concurrent_sessions": 50,
        "max_monthly_llm_tokens": 10_000_000,
        "features": list(_ALL_FEATURES),
    },
    "enterprise": {
        # ``None`` = no cap. Enforced as ``unlimited`` by callers.
        "max_users": None,
        "max_colonies_per_tenant": None,
        "max_concurrent_sessions": None,
        "max_monthly_llm_tokens": None,
        "features": list(_ALL_FEATURES),
    },
    # Dev-only fallback for ``COLONY_DEV_LICENSED_INSTALLATIONS``. Same
    # shape as enterprise; named distinctly so production code can
    # spot dev-seeded rows (``source='env_bootstrap'`` AND
    # ``plan='dev'``) and refuse to bill them.
    "dev": {
        "max_users": None,
        "max_colonies_per_tenant": None,
        "max_concurrent_sessions": None,
        "max_monthly_llm_tokens": None,
        "features": list(_ALL_FEATURES),
    },
}


def resolve_entitlements(
    plan: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge per-tenant ``overrides`` on top of ``PLAN_DEFAULTS[plan]``.

    Row-level fields win; missing keys take plan defaults. Unknown plan
    names raise ``KeyError`` — plan strings are CHECK-constrained at
    the DB layer so this should never fire in production, but it's a
    cheap correctness gate for the merge-time caller.
    """
    if plan not in PLAN_DEFAULTS:
        raise KeyError(
            f"Unknown plan {plan!r}; valid: {sorted(PLAN_DEFAULTS)}",
        )
    merged: dict[str, Any] = dict(PLAN_DEFAULTS[plan])
    if overrides:
        merged.update(overrides)
    return merged


__all__ = (
    "PLAN_DEFAULTS",
    "resolve_entitlements",
)
