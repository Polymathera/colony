"""Tier metadata for ``ConfigComponent`` fields and components.

Five orthogonal dimensions classify each configuration value (see the system
configuration refactor plan, §4). The enums below carry the vocabulary; the
``tier_metadata`` helper packs them into a JSON-Schema-friendly dict that
attaches under the ``"polymathera"`` namespace of ``json_schema_extra``.

This module is pure metadata in this step — no loader behaviour reads it yet.
The overlay loader (plan step 7) will consume these annotations to enforce
per-tier write rules and route ``update_config`` calls through the right
``StateManager`` slot.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class Tier(str, Enum):
    """Highest layer permitted to write a value."""

    L1_OPERATOR = "L1_OPERATOR"
    L2_TENANT = "L2_TENANT"
    L3_SESSION = "L3_SESSION"
    L4_RUNTIME = "L4_RUNTIME"


class Ownership(str, Enum):
    """Who declares the value."""

    COLONY = "COLONY"
    EXTENSION = "EXTENSION"


class Mutability(str, Enum):
    """When (and how) the value may change after deploy."""

    STATIC = "STATIC"
    RELOADABLE = "RELOADABLE"
    LIVE = "LIVE"


class Persistence(str, Enum):
    """Where overlays at this tier are stored."""

    EPHEMERAL = "EPHEMERAL"
    SHARED_STATE = "SHARED_STATE"
    EXTERNAL_STORE = "EXTERNAL_STORE"


class ReadScope(str, Enum):
    """Which scope a reader observes the value in."""

    GLOBAL = "GLOBAL"
    PER_TENANT = "PER_TENANT"
    PER_SESSION = "PER_SESSION"
    PER_DEPLOYMENT = "PER_DEPLOYMENT"


METADATA_KEY = "polymathera"


def tier_metadata(
    *,
    tier: Tier | None = None,
    ownership: Ownership | None = None,
    mutability: Mutability | None = None,
    persistence: Persistence | None = None,
    read_scope: ReadScope | None = None,
) -> dict[str, Any]:
    """Pack tier metadata into a ``json_schema_extra``-compatible dict.

    Returns ``{}`` when every argument is ``None``. Otherwise returns
    ``{"polymathera": {...}}``, omitting unset keys.

    Intended use::

        Field(
            default=...,
            json_schema_extra={
                "env": "POLYMATHERA_FOO_BAR",
                **tier_metadata(tier=Tier.L2_TENANT, mutability=Mutability.RELOADABLE),
            },
        )
    """
    payload: dict[str, str] = {}
    if tier is not None:
        payload["tier"] = tier.value
    if ownership is not None:
        payload["ownership"] = ownership.value
    if mutability is not None:
        payload["mutability"] = mutability.value
    if persistence is not None:
        payload["persistence"] = persistence.value
    if read_scope is not None:
        payload["read_scope"] = read_scope.value
    return {METADATA_KEY: payload} if payload else {}
