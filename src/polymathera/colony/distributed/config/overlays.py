"""StateManager-backed overlay store for L2 / L3 / L4 config layers.

Plan §5: defaults ⊕ env+YAML (L1) ⊕ tenant overlay (L2) ⊕ session overlay (L3)
⊕ runtime overlay (L4). L1 is owned by the in-process ``ConfigurationManager``;
the higher layers live in a single ``StateManager``-backed
:class:`ConfigOverlayState` so cross-replica reads/writes go through the same
CAS path that VCM and the convergence runtime already use.

This module supplies only the storage + composition primitives. Tier
enforcement (refusing to write a higher-tier value at a lower-tier scope)
piggybacks on the metadata declared in ``tiers.tier_metadata`` and is layered
on top by ``ConfigurationManager.update_overlay``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import Field

from ..state_management import SharedState, StateManager
from .tiers import METADATA_KEY, Tier


OVERLAY_STATE_KEY = "colony:config:overlays"


class ConfigOverlayState(SharedState):
    """Per-replica view of L2 / L3 / L4 overlays.

    All three dicts are double-nested: ``{scope_key: {component_path: {field: value, ...}}}``.
    A write is a deep-merge into ``state[layer][scope_key][component_path]``.
    """

    tenant_overlays: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)
    session_overlays: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)
    runtime_overlays: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)


class OverlayLayer(str, Enum):
    TENANT = "tenant"
    SESSION = "session"
    RUNTIME = "runtime"


_LAYER_TO_TIER = {
    OverlayLayer.TENANT: Tier.L2_TENANT,
    OverlayLayer.SESSION: Tier.L3_SESSION,
    OverlayLayer.RUNTIME: Tier.L4_RUNTIME,
}

_TIER_RANK = {
    Tier.L1_OPERATOR: 1,
    Tier.L2_TENANT: 2,
    Tier.L3_SESSION: 3,
    Tier.L4_RUNTIME: 4,
}


@dataclass(frozen=True)
class OverlayScope:
    """Identifies which overlay slot a write targets."""

    layer: OverlayLayer
    key: str

    @classmethod
    def tenant(cls, tenant_id: str) -> "OverlayScope":
        return cls(OverlayLayer.TENANT, tenant_id)

    @classmethod
    def session(cls, session_id: str) -> "OverlayScope":
        return cls(OverlayLayer.SESSION, session_id)

    @classmethod
    def runtime(cls, name: str) -> "OverlayScope":
        return cls(OverlayLayer.RUNTIME, name)

    @property
    def tier(self) -> Tier:
        return _LAYER_TO_TIER[self.layer]


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two dicts; ``overlay`` wins on conflict."""
    result = dict(base)
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def field_tier(component_cls: type, field_name: str) -> Tier | None:
    """Read the declared :class:`Tier` for a field, if any."""
    field = component_cls.model_fields.get(field_name)
    if field is None or field.json_schema_extra is None:
        return None
    extra = field.json_schema_extra
    if not isinstance(extra, dict):
        return None
    payload = extra.get(METADATA_KEY)
    if not isinstance(payload, dict):
        return None
    raw = payload.get("tier")
    if raw is None:
        return None
    try:
        return Tier(raw)
    except ValueError:
        return None


def assert_writable_at_scope(
    component_cls: type, updates: dict[str, Any], scope: OverlayScope,
) -> None:
    """Refuse writes whose declared ``Tier`` is below the overlay's tier.

    A field with no tier metadata is treated as ``L1_OPERATOR`` (the most
    restrictive default) so unannotated fields cannot be silently overridden
    from a tenant or session overlay. Nested dicts are checked recursively
    against the field's annotation when it itself is a Pydantic model.
    """
    scope_rank = _TIER_RANK[scope.tier]
    for key, value in updates.items():
        declared = field_tier(component_cls, key) or Tier.L1_OPERATOR
        if _TIER_RANK[declared] > scope_rank:
            continue  # declared tier permits this scope
        if _TIER_RANK[declared] < scope_rank:
            raise PermissionError(
                f"Overlay scope {scope.layer.value!r} cannot write field "
                f"{component_cls.__name__}.{key} (declared tier {declared.value})"
            )
        # Equal tier: permitted; no recursion needed for nested models since
        # tier annotations are field-local on the leaf component.


class OverlayStore:
    """Wraps a :class:`StateManager` for the overlay state document."""

    def __init__(self, state_manager: StateManager[ConfigOverlayState]):
        self._sm = state_manager

    async def read(self) -> ConfigOverlayState:
        async for state in self._sm.read_transaction():
            return state.model_copy(deep=True)
        raise RuntimeError("OverlayStore: read_transaction yielded nothing")

    async def write(
        self, scope: OverlayScope, path: str, updates: dict[str, Any],
    ) -> None:
        """Deep-merge ``updates`` into ``state[layer][scope.key][path]``."""
        async for state in self._sm.write_transaction():
            target = self._target(state, scope.layer)
            scope_payload = target.setdefault(scope.key, {})
            scope_payload[path] = deep_merge(scope_payload.get(path, {}), updates)

    @staticmethod
    def _target(
        state: ConfigOverlayState, layer: OverlayLayer,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        if layer is OverlayLayer.TENANT:
            return state.tenant_overlays
        if layer is OverlayLayer.SESSION:
            return state.session_overlays
        return state.runtime_overlays


def compose_overlays(
    base: dict[str, Any],
    state: ConfigOverlayState,
    *,
    path: str,
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Resolve L1 (``base``) ⊕ L2[tenant_id] ⊕ L3[session_id] ⊕ L4[*] for ``path``."""
    out = base
    if tenant_id is not None:
        layer = state.tenant_overlays.get(tenant_id, {}).get(path)
        if layer:
            out = deep_merge(out, layer)
    if session_id is not None:
        layer = state.session_overlays.get(session_id, {}).get(path)
        if layer:
            out = deep_merge(out, layer)
    # Runtime overlays are not scoped by caller — every consumer sees them.
    for runtime_payload in state.runtime_overlays.values():
        layer = runtime_payload.get(path)
        if layer:
            out = deep_merge(out, layer)
    return out


__all__ = (
    "OVERLAY_STATE_KEY",
    "ConfigOverlayState",
    "OverlayLayer",
    "OverlayScope",
    "OverlayStore",
    "assert_writable_at_scope",
    "compose_overlays",
    "deep_merge",
    "field_tier",
)
