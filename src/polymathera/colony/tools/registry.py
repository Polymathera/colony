"""``ToolRegistry`` — design-time index of available adapters.

Per master §3.3, the registry is the resolution surface that maps
"I want a tool that fulfils capability X under preferences Y" to a
concrete ``ToolAdapter``. The registry is colony-generic; CPS and
per-domain registries layer on top by registering more adapters
against the same contract.

Resolution runs in two passes:

1. **Hard filter** — drop adapters whose ``ToolSpec`` violates a
   ``Preferences`` hard requirement (e.g., ``min_headless=NATIVE``
   excludes a ``GUI_PRIMARY`` adapter; ``forbid_licences={COMMERCIAL}``
   excludes Gurobi).
2. **Score rank** — among survivors, score by how well each adapter
   matches the soft preferences (preferred backend, headless tier,
   cost). The highest-scoring adapter wins; ties broken by adapter
   name for determinism.

The registry holds adapter *instances* (not classes) so adapters
that require constructor-time configuration (an HTTP client, a Ray
deployment handle, a process pool) are wired in by the deployment
layer and stay resolvable thereafter.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping

from .base import (
    Determinism,
    HITLFrequency,
    HeadlessReadiness,
    Licensing,
    Preferences,
    ToolAdapter,
    ToolSpec,
)


logger = logging.getLogger(__name__)


class ToolRegistryError(RuntimeError):
    """Base error for the registry."""


class NoAdapterAvailable(ToolRegistryError):
    """``resolve()`` could not find an adapter that satisfies the
    capability + preferences."""


class DuplicateAdapter(ToolRegistryError):
    """An adapter with the same name is already registered."""


class ToolRegistry:
    """Thread-safe registry of ``ToolAdapter`` instances."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_name: dict[str, ToolAdapter] = {}
        self._by_capability: dict[str, set[str]] = {}

    # ---- Mutation -----------------------------------------------------

    def register(self, adapter: ToolAdapter) -> None:
        """Register ``adapter``. Raises ``DuplicateAdapter`` if its
        ``spec.name`` is already taken."""

        spec = type(adapter).spec  # ClassVar
        if not isinstance(spec, ToolSpec):
            raise ToolRegistryError(
                f"{type(adapter).__name__}.spec must be a ToolSpec instance, "
                f"got {type(spec).__name__}.",
            )
        with self._lock:
            if spec.name in self._by_name:
                raise DuplicateAdapter(
                    f"Adapter named {spec.name!r} already registered.",
                )
            self._by_name[spec.name] = adapter
            for cap in spec.capabilities:
                self._by_capability.setdefault(cap, set()).add(spec.name)

    def unregister(self, name: str) -> bool:
        """Remove the adapter named ``name``. Returns True iff present."""

        with self._lock:
            adapter = self._by_name.pop(name, None)
            if adapter is None:
                return False
            spec = type(adapter).spec
            for cap in spec.capabilities:
                bucket = self._by_capability.get(cap)
                if bucket is None:
                    continue
                bucket.discard(name)
                if not bucket:
                    self._by_capability.pop(cap, None)
            return True

    def clear(self) -> None:
        with self._lock:
            self._by_name.clear()
            self._by_capability.clear()

    # ---- Lookup -------------------------------------------------------

    def get(self, name: str) -> ToolAdapter | None:
        with self._lock:
            return self._by_name.get(name)

    def list_capabilities(self) -> Iterable[str]:
        with self._lock:
            return sorted(self._by_capability.keys())

    def list_adapters(self) -> Iterable[ToolAdapter]:
        with self._lock:
            # Stable order by name for deterministic iteration.
            return [self._by_name[k] for k in sorted(self._by_name)]

    def list_adapters_for(self, capability: str) -> Iterable[ToolAdapter]:
        with self._lock:
            names = sorted(self._by_capability.get(capability, ()))
            return [self._by_name[n] for n in names]

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_name)

    # ---- Resolution ---------------------------------------------------

    def resolve(
        self,
        capability: str,
        preferences: Preferences | None = None,
    ) -> ToolAdapter:
        """Pick an adapter that fulfils ``capability`` under ``preferences``.

        Hard filter + score rank as described in the module docstring.
        Raises ``NoAdapterAvailable`` if no adapter survives the
        filter.
        """

        prefs = preferences or Preferences()
        candidates = list(self.list_adapters_for(capability))
        if not candidates:
            raise NoAdapterAvailable(
                f"No adapter registered for capability {capability!r}.",
            )
        survivors = [
            adapter for adapter in candidates
            if _passes_hard_filter(type(adapter).spec, prefs)
        ]
        if not survivors:
            raise NoAdapterAvailable(
                f"No adapter for capability {capability!r} satisfied the "
                f"preferences (filtered out {len(candidates)} candidates).",
            )
        scored = sorted(
            survivors,
            key=lambda a: (
                -_score(type(a).spec, prefs),
                type(a).spec.name,
            ),
        )
        return scored[0]

    def resolve_all(
        self,
        capability: str,
        preferences: Preferences | None = None,
    ) -> list[ToolAdapter]:
        """Like ``resolve`` but returns every survivor in score order.

        Empty list when nothing survives. Useful for the
        ``BuildVsBuyAdvisor`` (which inspects the *set* of options
        before recommending build vs. integrate)."""

        prefs = preferences or Preferences()
        candidates = list(self.list_adapters_for(capability))
        survivors = [
            adapter for adapter in candidates
            if _passes_hard_filter(type(adapter).spec, prefs)
        ]
        return sorted(
            survivors,
            key=lambda a: (
                -_score(type(a).spec, prefs),
                type(a).spec.name,
            ),
        )


# ---------------------------------------------------------------------------
# Filter + scoring
# ---------------------------------------------------------------------------


def _passes_hard_filter(spec: ToolSpec, prefs: Preferences) -> bool:
    if prefs.min_headless is not None:
        if spec.headless.order < prefs.min_headless.order:
            return False
    if prefs.max_hitl is not None:
        if spec.hitl_frequency.order > prefs.max_hitl.order:
            return False
    if prefs.required_determinism is not None:
        if spec.determinism != prefs.required_determinism:
            return False
    if prefs.required_backend is not None:
        if spec.backend != prefs.required_backend:
            return False
    if prefs.forbid_licences and spec.licensing in prefs.forbid_licences:
        return False
    cost = spec.cost_model
    if prefs.max_cpu_seconds is not None and cost.cpu_seconds > prefs.max_cpu_seconds:
        return False
    if prefs.max_gpu_seconds is not None and cost.gpu_seconds > prefs.max_gpu_seconds:
        return False
    if prefs.max_memory_gb is not None and cost.memory_gb > prefs.max_memory_gb:
        return False
    if prefs.max_dollars is not None and cost.dollars > prefs.max_dollars:
        return False
    if prefs.require_interruptible and not spec.interruptibility:
        return False
    if prefs.allowed_container_images:
        # Adapters without a container image are accepted (in-process
        # backends don't need one); only filter when the spec sets one
        # and it isn't on the allow-list.
        if spec.container_image is not None and spec.container_image not in prefs.allowed_container_images:
            return False
    if prefs.allowed_localities is not None:
        if spec.execution_locality not in prefs.allowed_localities:
            return False
    req = spec.resource_requirements
    if prefs.max_required_vcpus is not None and req.min_vcpus > prefs.max_required_vcpus:
        return False
    if prefs.max_required_memory_gb is not None and req.min_memory_gb > prefs.max_required_memory_gb:
        return False
    if prefs.max_required_wallclock_seconds is not None and req.expected_wallclock_seconds > prefs.max_required_wallclock_seconds:
        return False
    if req.gpu is not None:
        if prefs.max_required_gpu_count is not None and req.gpu.count > prefs.max_required_gpu_count:
            return False
        if prefs.allowed_required_gpu_kinds is not None and req.gpu.kind not in prefs.allowed_required_gpu_kinds:
            return False
    return True


def _score(spec: ToolSpec, prefs: Preferences) -> float:
    """Higher is better. The scoring weights are intentionally simple
    so the ordering is predictable; richer scoring is the caller's job
    (compose a custom resolver on top)."""

    score = 0.0
    # Headless: each tier above the floor adds 1.0.
    score += float(spec.headless.order)
    # HITL: each tier closer to AUTONOMOUS adds 1.0.
    score += float(_HITL_MAX_ORDER - spec.hitl_frequency.order)
    # Backend match.
    if prefs.preferred_backend is not None and spec.backend == prefs.preferred_backend:
        score += 5.0
    # Determinism: deterministic > seeded > stochastic; small.
    score += {
        Determinism.DETERMINISTIC: 0.5,
        Determinism.SEEDED: 0.25,
        Determinism.STOCHASTIC: 0.0,
    }[spec.determinism]
    # Cost: lower is better.
    if prefs.prefer_lower_cost:
        cost = spec.cost_model
        cost_penalty = (
            cost.cpu_seconds * 0.001
            + cost.gpu_seconds * 0.005
            + cost.memory_gb * 0.001
            + cost.dollars * 0.5
        )
        score -= cost_penalty
    return score


_HITL_MAX_ORDER = max(t.order for t in HITLFrequency)


__all__ = (
    "ToolRegistry",
    "ToolRegistryError",
    "NoAdapterAvailable",
    "DuplicateAdapter",
)
