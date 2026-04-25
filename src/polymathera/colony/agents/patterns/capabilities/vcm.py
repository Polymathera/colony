"""VCM capability — agent-facing facade over the ``VirtualContextManager``.

Lifts the VCM's mapping-lifecycle operations (``mmap_application_scope`` and
friends) into ``@action_executor`` methods so any agent can drive the VCM
from its action policy. Whether a given agent gets the capability is a
configuration choice (blueprint composition), not a code change.

This file is the Phase 1 surface: mapping a git repository or a blackboard
scope into VCM pages, unmapping, and inspecting mappings. Page-lifecycle
actions and filesystem watchers are added in later phases — see
``colony_docs/markdown/plans/design_VCMCapability.md``.

Typical usage::

    # Session agent: let the LLM map the repo the user just asked about
    SessionAgent.bind(
        ...,
        capability_blueprints=[
            ...,
            VCMCapability.bind(scope=BlackboardScope.SESSION),
        ],
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING
from overrides import override

from ...base import AgentCapability
from ...blackboard.protocol import VCMEventProtocol
from ...models import ActionPolicyExecutionState, AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions import action_executor

from ....vcm.models import MmapConfig, MmapResult
from ....vcm.sources import BuilInContextPageSourceType

if TYPE_CHECKING:
    from ...base import Agent
    from .... import serving

logger = logging.getLogger(__name__)


_WatchMode = Literal["reindex", "invalidate", "notify_only"]


@dataclass
class _WatchHandle:
    """Bookkeeping for one active filesystem watch.

    The watcher task reads ``stop_event`` for cancellation; every public
    VCMCapability action that touches watches manipulates this record.
    """

    watch_id: str
    scope_id: str
    paths: tuple[str, ...]
    on_change: _WatchMode
    debounce_seconds: float
    started_at: float
    stop_event: asyncio.Event
    task: asyncio.Task | None = None
    last_fired_at: float | None = None
    fire_count: int = 0


# ---------------------------------------------------------------------------
# VCMCapability
# ---------------------------------------------------------------------------

class VCMCapability(AgentCapability):
    """Agent-facing actions for the Virtual Context Manager.

    The capability is a thin wrapper around the ``VirtualContextManager``
    deployment handle obtained via ``colony.system.get_vcm()``. It does not
    cache or replicate VCM state; each action delegates to the deployment.

    Constructor kwargs are blueprint-serializable. The agent positional
    argument is injected by ``AgentCapabilityBlueprint.local_instance`` at
    capability-creation time.

    Args:
        agent: Owning agent (None in detached mode — not supported here
            because every action needs the agent's execution context).
        scope: Default blackboard scope for target mappings. Applied when a
            caller does not supply an explicit ``scope_id`` string.
        namespace: Namespace segment appended to the capability's own
            scope_id (used for the capability's blackboard writes, NOT for
            VCM mapping targets). The default groups all VCM-related
            blackboard records under ``…:vcm:…``.
        default_mmap_config: Default ``MmapConfig`` used when an action does
            not receive an explicit config. ``None`` means construct a
            fresh default ``MmapConfig()`` on every call.
        capability_key: Unique identifier for the capability in the agent's
            ``_capabilities`` dict. Defaults to ``"vcm"``.
        app_name: Optional ``serving.Application`` name override used when
            resolving the VCM deployment handle.
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.SESSION,
        namespace: str = "vcm",
        default_mmap_config: MmapConfig | None = None,
        # Watch configuration
        watch_root: str = "/mnt/shared/filesystem",
        max_concurrent_watches: int = 16,
        capability_key: str = "vcm",
        app_name: str | None = None,
    ):
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            capability_key=capability_key,
            app_name=app_name,
        )
        self._default_scope = scope
        self._default_mmap_config = default_mmap_config
        self._watch_root = watch_root
        self._max_concurrent_watches = max_concurrent_watches
        self._watches: dict[str, _WatchHandle] = {}
        self._vcm_handle: serving.DeploymentHandle | None = None

    def get_action_group_description(self) -> str:
        return (
            "Virtual Context Memory (VCM) — map application data (git "
            "repositories, blackboard scopes, memory scopes) into VCM "
            "pages so it becomes accessible to LLM workers as virtual "
            "context. Call mmap_repo when the user wants a codebase "
            "analyzed; call mmap_blackboard_scope to expose an agent's "
            "own records to other agents. munmap_scope releases the "
            "mapping."
        )

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"vcm", "context", "coordination"})

    _CUSTOM_KEY = "vcm_watches"

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> AgentSuspensionState:
        """Persist active watches across agent suspend/resume.

        Only the watch *configuration* is serialized — not the stop_event
        or the asyncio.Task, which are re-created on ``deserialize``.
        Stored under ``state.action_policy_state.custom["vcm_watches"]``,
        the designated escape hatch for policy-adjacent state.
        """
        if state.action_policy_state is None:
            state.action_policy_state = ActionPolicyExecutionState()
        state.action_policy_state.custom[self._CUSTOM_KEY] = [
            {
                "watch_id": w.watch_id,
                "scope_id": w.scope_id,
                "paths": list(w.paths),
                "on_change": w.on_change,
                "debounce_seconds": w.debounce_seconds,
            }
            for w in self._watches.values()
        ]
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> None:
        """Re-start every previously active watch.

        Missing filesystem paths or permission errors are logged but do
        not abort restoration of the other watches.
        """
        if state.action_policy_state is None:
            return
        watches = state.action_policy_state.custom.get(self._CUSTOM_KEY) or []
        for w in watches:
            try:
                await self.watch_repo(
                    scope_id=w["scope_id"],
                    paths=list(w.get("paths") or []),
                    on_change=w.get("on_change", "notify_only"),
                    debounce_seconds=float(w.get("debounce_seconds", 5.0)),
                    watch_id=w.get("watch_id"),
                )
            except Exception as e:  # pragma: no cover — defensive
                logger.warning(
                    "VCMCapability: failed to restore watch %r: %s",
                    w.get("watch_id"), e,
                )

    async def shutdown(self) -> None:
        """Stop every active watcher and wait for its task to finish.

        Called by the owning agent during teardown. Idempotent.
        """
        handles = list(self._watches.values())
        self._watches.clear()
        for h in handles:
            h.stop_event.set()
        for h in handles:
            if h.task is None:
                continue
            try:
                await asyncio.wait_for(h.task, timeout=5.0)
            except asyncio.TimeoutError:
                h.task.cancel()
            except Exception as e:  # pragma: no cover — defensive
                logger.debug(
                    "VCMCapability: watcher %r exited with %r",
                    h.watch_id, e,
                )

    # --- Internal -----------------------------------------------------------

    def _get_vcm(self):
        """Lazily resolve the VCM deployment handle.

        Caches the handle on the capability so we don't pay the
        ``get_deployment`` lookup cost on every call.
        """
        if self._vcm_handle is None:
            from ....system import get_vcm
            self._vcm_handle = get_vcm(self._app_name)
        return self._vcm_handle

    async def _emit_event(self, key: str, value: dict[str, Any]) -> None:
        """Write a VCM lifecycle event to the capability's own blackboard.

        Failures here must not break the action itself — event emission is
        advisory. Subscribers that care about guaranteed delivery should
        read the VCM's shared-state mapping records directly.
        """
        try:
            bb = await self.get_blackboard()
            await bb.write(key, value)
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(
                "VCMCapability: failed to emit event %r: %s", key, e,
            )

    @staticmethod
    def _normalize_mmap_result(result: Any, *, fallback_scope_id: str) -> dict[str, Any]:
        """Coerce an ``MmapResult`` (or the dict form returned by a Ray
        ``DeploymentHandle``) into a stable action return shape.

        Agents read this dict from the dispatcher's REPL. Keeping the
        shape uniform across success and error paths means callers don't
        have to conditional-branch on the container type.
        """
        if isinstance(result, MmapResult):
            return {
                "status": result.status,
                "scope_id": result.scope_id,
                "message": result.message,
            }
        if isinstance(result, dict):
            return {
                "status": result.get("status", "error"),
                "scope_id": result.get("scope_id", fallback_scope_id),
                "message": result.get("message", ""),
            }
        return {
            "status": "error",
            "scope_id": fallback_scope_id,
            "message": f"Unexpected VCM result type: {type(result).__name__}",
        }

    def _resolve_target_scope_id(
        self,
        *,
        scope: BlackboardScope | None,
        scope_id: str | None,
    ) -> str:
        """Decide the VCM target scope_id from the caller's arguments.

        ``scope_id`` wins if both are given. Otherwise we build one from
        ``scope`` (or the capability default) using ``get_scope_prefix``.
        """
        if scope_id:
            return scope_id
        target_scope = scope or self._default_scope
        from ...scopes import ScopeUtils
        if target_scope == BlackboardScope.COLONY:
            return ScopeUtils.get_colony_level_scope()
        if target_scope == BlackboardScope.SESSION:
            return ScopeUtils.get_session_level_scope()
        if target_scope == BlackboardScope.TENANT:
            return ScopeUtils.get_tenant_level_scope()
        if target_scope == BlackboardScope.AGENT:
            return ScopeUtils.get_agent_level_scope(self.agent)
        raise ValueError(f"Unsupported target scope: {target_scope!r}")

    # === Action Executors ===

    @action_executor(interruptible=True)
    async def mmap_repo(
        self,
        *,
        origin_url: str | None = None,
        local_repo_path: str | None = None,
        branch: str | None = None,
        commit: str | None = None,
        scope: BlackboardScope | None = None,
        scope_id: str | None = None,
        config: MmapConfig | None = None,
    ) -> dict[str, Any]:
        """Map a git repository into the VCM as pages.

        Exactly one of ``origin_url`` / ``local_repo_path`` must be set.
        ``local_repo_path`` is converted to a ``file://`` URL before it
        reaches the VCM, matching the path that the dashboard's
        upload-and-map endpoint takes.

        If ``scope_id`` is provided, it is used as-is. Otherwise the target
        scope_id is built from ``scope`` (or the capability's configured
        ``default_scope``) using the current execution context.

        The VCM records the mapping in shared state, so a second call for
        the same ``scope_id`` returns ``status="already_mapped"``.

        Args:
            origin_url: Git URL (e.g., ``https://github.com/org/repo``).
            local_repo_path: Path on the ray-head/ray-worker filesystem.
                Converted to ``file://<path>`` internally.
            branch: Git branch (optional).
            commit: Git commit SHA (optional). Pins the mapping to a
                specific revision.
            scope: Blackboard scope level for the mapping target. Ignored
                when ``scope_id`` is supplied.
            scope_id: Explicit target scope_id, overriding ``scope``.
            config: Mapping configuration (flushing, locality, pinning).
                Falls back to the capability default, then to
                ``MmapConfig()``.

        Returns:
            ``{"status", "scope_id", "message", "origin_url", "branch",
            "commit"}``. ``status`` is one of ``mapped`` /
            ``already_mapped`` / ``error``.
        """
        if bool(origin_url) == bool(local_repo_path):
            return {
                "status": "error",
                "scope_id": "",
                "message": (
                    "Exactly one of origin_url or local_repo_path must be "
                    "provided."
                ),
                "origin_url": origin_url or "",
                "branch": branch,
                "commit": commit,
            }

        effective_url = origin_url or f"file://{local_repo_path}"
        target_scope_id = self._resolve_target_scope_id(
            scope=scope, scope_id=scope_id
        )
        mmap_config = config or self._default_mmap_config or MmapConfig()

        try:
            raw = await self._get_vcm().mmap_application_scope(
                scope_id=target_scope_id,
                source_type=BuilInContextPageSourceType.FILE_GROUPER.value,
                config=mmap_config,
                origin_url=effective_url,
                branch=branch,
                commit=commit,
            )
        except Exception as e:
            logger.exception(
                "VCMCapability.mmap_repo failed for scope_id=%s url=%s",
                target_scope_id, effective_url,
            )
            return {
                "status": "error",
                "scope_id": target_scope_id,
                "message": f"VCM call raised: {e}",
                "origin_url": effective_url,
                "branch": branch,
                "commit": commit,
            }

        normalized = self._normalize_mmap_result(
            raw, fallback_scope_id=target_scope_id
        )
        normalized.update({
            "origin_url": effective_url,
            "branch": branch,
            "commit": commit,
        })
        logger.info(
            "VCMCapability.mmap_repo: scope_id=%s status=%s url=%s",
            normalized["scope_id"], normalized["status"], effective_url,
        )
        if normalized["status"] == "mapped":
            await self._emit_event(
                VCMEventProtocol.mapped_key(normalized["scope_id"]),
                {
                    "scope_id": normalized["scope_id"],
                    "source_type": BuilInContextPageSourceType.FILE_GROUPER.value,
                    "origin_url": effective_url,
                    "branch": branch,
                    "commit": commit,
                    "ts": time.time(),
                },
            )
        return normalized

    @action_executor()
    async def mmap_blackboard_scope(
        self,
        *,
        scope: BlackboardScope | None = None,
        scope_id: str | None = None,
        namespace: str | None = None,
        config: MmapConfig | None = None,
    ) -> dict[str, Any]:
        """Map a blackboard scope into the VCM.

        Useful for exposing the agent's own blackboard records (or a
        collaborating agent's scope) as virtual context so LLM workers
        can attend to them. Unlike ``mmap_repo``, this does not involve
        git — the page source is the blackboard's event stream.

        If ``scope_id`` is provided, it is used as-is. Otherwise a
        scope_id is built from ``scope`` (or the capability default), with
        ``namespace`` appended when non-None.

        Args:
            scope: Blackboard scope level. Ignored when ``scope_id`` is
                supplied.
            scope_id: Explicit target scope_id, overriding ``scope``.
            namespace: Optional sub-namespace (e.g., ``"discoveries"``).
                Ignored when ``scope_id`` is supplied.
            config: Mapping configuration. Falls back to the capability
                default, then to ``MmapConfig()``.
        """
        if scope_id:
            target_scope_id = scope_id
        else:
            base = self._resolve_target_scope_id(scope=scope, scope_id=None)
            target_scope_id = f"{base}:{namespace}" if namespace else base

        mmap_config = config or self._default_mmap_config or MmapConfig()

        try:
            raw = await self._get_vcm().mmap_application_scope(
                scope_id=target_scope_id,
                source_type=BuilInContextPageSourceType.BLACKBOARD.value,
                config=mmap_config,
            )
        except Exception as e:
            logger.exception(
                "VCMCapability.mmap_blackboard_scope failed for scope_id=%s",
                target_scope_id,
            )
            return {
                "status": "error",
                "scope_id": target_scope_id,
                "message": f"VCM call raised: {e}",
            }

        normalized = self._normalize_mmap_result(
            raw, fallback_scope_id=target_scope_id
        )
        logger.info(
            "VCMCapability.mmap_blackboard_scope: scope_id=%s status=%s",
            normalized["scope_id"], normalized["status"],
        )
        if normalized["status"] == "mapped":
            await self._emit_event(
                VCMEventProtocol.mapped_key(normalized["scope_id"]),
                {
                    "scope_id": normalized["scope_id"],
                    "source_type": BuilInContextPageSourceType.BLACKBOARD.value,
                    "ts": time.time(),
                },
            )
        return normalized

    @action_executor()
    async def munmap_scope(self, scope_id: str) -> dict[str, Any]:
        """Unmap a previously mapped scope from the VCM.

        Flushes any pending records and shuts down the page source.
        Returns ``status="not_mapped"`` if the scope was never mapped.

        Args:
            scope_id: Target scope_id as returned by ``mmap_repo`` or
                ``mmap_blackboard_scope``.
        """
        try:
            raw = await self._get_vcm().munmap_application_scope(
                scope_id=scope_id,
            )
        except Exception as e:
            logger.exception(
                "VCMCapability.munmap_scope failed for scope_id=%s", scope_id,
            )
            return {
                "status": "error",
                "scope_id": scope_id,
                "message": f"VCM call raised: {e}",
            }
        normalized = self._normalize_mmap_result(
            raw, fallback_scope_id=scope_id
        )
        logger.info(
            "VCMCapability.munmap_scope: scope_id=%s status=%s",
            normalized["scope_id"], normalized["status"],
        )
        if normalized["status"] == "unmapped":
            await self._emit_event(
                VCMEventProtocol.unmapped_key(normalized["scope_id"]),
                {"scope_id": normalized["scope_id"], "ts": time.time()},
            )
        return normalized

    @action_executor()
    async def is_scope_mapped(self, scope_id: str) -> dict[str, Any]:
        """Check whether a scope is currently mapped into the VCM.

        Args:
            scope_id: Target scope_id to check.

        Returns:
            ``{"scope_id": str, "mapped": bool, "message": str}``.
            ``message`` is populated on error; the LLM can distinguish
            "not mapped" from "lookup failed".
        """
        try:
            mapped = await self._get_vcm().is_application_scope_mapped(
                scope_id=scope_id,
            )
        except Exception as e:
            logger.exception(
                "VCMCapability.is_scope_mapped failed for scope_id=%s", scope_id,
            )
            return {"scope_id": scope_id, "mapped": False, "message": str(e)}
        return {"scope_id": scope_id, "mapped": bool(mapped), "message": ""}

    @action_executor()
    async def get_scope_status(self, scope_id: str) -> dict[str, Any]:
        """Return detailed mapping status for a scope.

        Includes the ``MmapConfig`` the scope was mapped with, the
        ``syscontext`` that mapped it, the creation timestamp, and — when
        the mapping is materialized on this replica — the current page
        count and tracked record count.

        Args:
            scope_id: Target scope_id to look up.

        Returns:
            The VCM's status dict, or ``{"status": "not_mapped",
            "scope_id": ...}`` when no mapping exists, or an error dict
            on failure.
        """
        try:
            status = await self._get_vcm().get_application_scope_mapping_status(
                scope_id=scope_id,
            )
        except Exception as e:
            logger.exception(
                "VCMCapability.get_scope_status failed for scope_id=%s", scope_id,
            )
            return {"status": "error", "scope_id": scope_id, "message": str(e)}

        if status is None:
            return {"status": "not_mapped", "scope_id": scope_id}
        return status

    @action_executor()
    async def list_mapped_scopes(self) -> dict[str, Any]:
        """List every VCM mapping visible to the caller.

        Each entry has ``scope_id`` plus the mapping's ``syscontext``
        (tenant/colony/session). This endpoint is privileged on the VCM
        side; if the caller lacks permission, the action returns an
        error rather than raising so the LLM can observe the failure.

        Returns:
            ``{"mappings": [...], "count": int, "message": str}`` on
            success; ``{"mappings": [], "count": 0, "message": "<err>"}``
            on failure.
        """
        try:
            mappings = await self._get_vcm().get_all_mapped_scopes()
        except Exception as e:
            logger.warning(
                "VCMCapability.list_mapped_scopes failed: %s", e,
            )
            return {"mappings": [], "count": 0, "message": str(e)}
        mappings = list(mappings or [])
        return {"mappings": mappings, "count": len(mappings), "message": ""}

    # --- Page lifecycle (Phase 2) ------------------------------------------

    @action_executor(interruptible=True)
    async def request_pages(
        self,
        page_ids: list[str],
        *,
        timeout_s: float = 30.0,
        priority: int = 10,
        lock_duration_s: float | None = None,
        lock_reason: str = "",
    ) -> dict[str, Any]:
        """Request that a set of pages be loaded into VCM replicas.

        Wraps the VCM's ``issue_page_fault`` + ``wait_for_pages`` flow:
        a single fault is issued for all requested page_ids at the given
        priority, then the action waits for the fault to complete (or
        times out). If ``lock_duration_s`` is set, the pages are held in
        place for that duration after loading to avoid eviction during a
        multi-step reasoning sequence.

        Args:
            page_ids: Pages that must be loaded before the action returns.
            timeout_s: Maximum wall-clock time to wait, in seconds.
            priority: Fault priority (higher = more urgent). Default 10.
            lock_duration_s: If set, lock the pages after loading for this
                many seconds. The lock is tagged with the agent's id.
            lock_reason: Human-readable reason for the lock (audited).

        Returns:
            ``{"fault_id": str | None, "loaded": bool, "page_ids": [...],
            "message": str}``. ``loaded`` is ``False`` on timeout and the
            reason is in ``message``.
        """
        if not page_ids:
            return {
                "fault_id": None, "loaded": True, "page_ids": [],
                "message": "no pages requested",
            }
        requester_id = self.agent.agent_id if self._agent is not None else "unknown"
        try:
            fault_id = await self._get_vcm().issue_page_fault(
                page_ids=list(page_ids),
                requester_id=requester_id,
                priority=priority,
                lock_duration_s=lock_duration_s,
                lock_reason=lock_reason,
            )
            loaded = await self._get_vcm().wait_for_pages(
                fault_id=fault_id, timeout_s=timeout_s,
            )
        except Exception as e:
            logger.exception(
                "VCMCapability.request_pages failed for pages=%s", page_ids,
            )
            return {
                "fault_id": None, "loaded": False,
                "page_ids": list(page_ids), "message": str(e),
            }
        # Emit a page-fault event so observers (dashboard, peer agents) see
        # the request even before the underlying pages actually load.
        await self._emit_event(
            VCMEventProtocol.page_fault_key(fault_id),
            {
                "fault_id": fault_id,
                "page_ids": list(page_ids),
                "priority": priority,
                "requester": requester_id,
                "ts": time.time(),
            },
        )
        return {
            "fault_id": fault_id,
            "loaded": bool(loaded),
            "page_ids": list(page_ids),
            "message": "" if loaded else f"timeout after {timeout_s}s",
        }

    @action_executor()
    async def lock_pages(
        self,
        page_ids: list[str],
        *,
        ttl_s: float = 300.0,
        reason: str = "",
    ) -> dict[str, Any]:
        """Lock one or more pages to prevent eviction for ``ttl_s`` seconds.

        Locks are named by the agent's ``agent_id``. Use
        ``extend_lock`` to refresh a lock, or ``unlock_pages`` to release.
        Locking is best-effort per page — the return value reports which
        pages succeeded.

        Args:
            page_ids: Pages to lock.
            ttl_s: Lock duration in seconds. Must be positive.
            reason: Human-readable reason (stored with the lock and
                visible in ``get_page_lock_info``).

        Returns:
            ``{"locked": [...], "failed": [{"page_id", "error"}, ...]}``.
        """
        if ttl_s <= 0:
            return {
                "locked": [],
                "failed": [
                    {"page_id": p, "error": "ttl_s must be positive"}
                    for p in page_ids
                ],
            }
        locker = self.agent.agent_id if self._agent is not None else "unknown"
        locked: list[str] = []
        failed: list[dict[str, str]] = []
        vcm = self._get_vcm()
        for page_id in page_ids:
            try:
                await vcm.lock_page(
                    page_id=page_id,
                    locked_by=locker,
                    lock_duration_s=ttl_s,
                    reason=reason,
                )
                locked.append(page_id)
            except Exception as e:
                logger.warning(
                    "VCMCapability.lock_pages: failed to lock %s: %s",
                    page_id, e,
                )
                failed.append({"page_id": page_id, "error": str(e)})
        return {"locked": locked, "failed": failed}

    @action_executor()
    async def unlock_pages(self, page_ids: list[str]) -> dict[str, Any]:
        """Release previously-acquired page locks.

        A page that is not locked is reported in ``already_unlocked``
        rather than as a failure (the VCM's ``unlock_page`` returns
        ``False`` in that case).

        Args:
            page_ids: Pages to unlock.

        Returns:
            ``{"unlocked": [...], "already_unlocked": [...],
            "failed": [{"page_id", "error"}, ...]}``.
        """
        unlocked: list[str] = []
        already: list[str] = []
        failed: list[dict[str, str]] = []
        vcm = self._get_vcm()
        for page_id in page_ids:
            try:
                ok = await vcm.unlock_page(page_id=page_id)
                (unlocked if ok else already).append(page_id)
            except Exception as e:
                logger.warning(
                    "VCMCapability.unlock_pages: failed to unlock %s: %s",
                    page_id, e,
                )
                failed.append({"page_id": page_id, "error": str(e)})
        return {
            "unlocked": unlocked,
            "already_unlocked": already,
            "failed": failed,
        }

    @action_executor()
    async def extend_lock(
        self, page_id: str, *, additional_s: float
    ) -> dict[str, Any]:
        """Extend an existing page lock by ``additional_s`` seconds.

        Args:
            page_id: Page whose lock to extend.
            additional_s: Seconds to add (must be positive).

        Returns:
            ``{"page_id", "extended": bool, "message": str}``. ``extended``
            is ``False`` when the page was not locked; the VCM does not
            implicitly create a new lock.
        """
        if additional_s <= 0:
            return {
                "page_id": page_id, "extended": False,
                "message": "additional_s must be positive",
            }
        try:
            ok = await self._get_vcm().extend_page_lock(
                page_id=page_id, additional_duration_s=additional_s,
            )
        except Exception as e:
            logger.exception(
                "VCMCapability.extend_lock failed for page_id=%s", page_id,
            )
            return {
                "page_id": page_id, "extended": False, "message": str(e),
            }
        return {
            "page_id": page_id, "extended": bool(ok),
            "message": "" if ok else "page was not locked",
        }

    @action_executor()
    async def get_page_graph(
        self, *, max_nodes: int = 5000
    ) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the VCM page graph.

        Wraps the VCM's ``get_page_graph_data`` endpoint, which prunes the
        graph to the highest-degree ``max_nodes`` nodes if larger than the
        cap. Intended for LLMs that want to reason about the structure of
        loaded content (clusters, centralities) and for visualisers.

        Args:
            max_nodes: Soft cap on the number of nodes returned.

        Returns:
            ``{"nodes": [...], "edges": [...], "node_count", "edge_count"}``
            on success; ``{"nodes": [], "edges": [], "message": "<err>"}``
            on failure (including the KERNEL-ring restriction — see the
            design doc §6 for context).
        """
        try:
            graph = await self._get_vcm().get_page_graph_data(
                max_nodes=max_nodes,
            )
        except Exception as e:
            logger.warning(
                "VCMCapability.get_page_graph failed: %s", e,
            )
            return {
                "nodes": [], "edges": [],
                "node_count": 0, "edge_count": 0,
                "message": str(e),
            }
        graph = dict(graph or {})
        graph.setdefault("message", "")
        return graph

    @action_executor()
    async def list_stored_pages(
        self,
        *,
        source_pattern: str | None = None,
        limit: int = 2000,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List page summaries from persistent VCM storage.

        Each entry is lightweight metadata (page_id, source, size,
        group_id, tenant_id) — the actual page content is not materialised
        here. Use ``request_pages`` to get a subset loaded into replicas.

        Args:
            source_pattern: Optional substring filter on each page's
                ``source`` field (e.g., repo URL or scope_id).
            limit: Maximum number of entries to return (default 2000).
            offset: Entries to skip (for paginated inspection).

        Returns:
            ``{"pages": [...], "count": int, "message": str}``.
        """
        try:
            pages = await self._get_vcm().list_stored_pages(
                source_pattern=source_pattern, limit=limit, offset=offset,
            )
        except Exception as e:
            logger.warning(
                "VCMCapability.list_stored_pages failed: %s", e,
            )
            return {"pages": [], "count": 0, "message": str(e)}
        pages = list(pages or [])
        return {"pages": pages, "count": len(pages), "message": ""}

    @action_executor()
    async def get_pages_for_scope(
        self,
        scope_id: str,
        *,
        include_metadata: bool = False,
    ) -> dict[str, Any]:
        """List every VCM page that was materialised from a given scope.

        This is the "which pages came from this repo" lookup — helpful
        when an agent wants to scope a subsequent ``request_pages`` or
        ``lock_pages`` call to the pages that belong to a specific
        codebase or blackboard scope.

        Args:
            scope_id: Target scope to query.
            include_metadata: If ``True``, include each page's full
                metadata dict alongside its id.

        Returns:
            ``{"scope_id": str, "pages": [...], "count": int,
            "message": str}``.
        """
        try:
            pages = await self._get_vcm().get_pages_for_scope(
                scope_id=scope_id,
                include_metadata=include_metadata,
            )
        except Exception as e:
            logger.warning(
                "VCMCapability.get_pages_for_scope failed for %s: %s",
                scope_id, e,
            )
            return {
                "scope_id": scope_id, "pages": [], "count": 0,
                "message": str(e),
            }
        pages = list(pages or [])
        return {
            "scope_id": scope_id,
            "pages": pages,
            "count": len(pages),
            "message": "",
        }

    @action_executor()
    async def get_vcm_stats(self) -> dict[str, Any]:
        """Return VCM-wide statistics (page table + storage).

        Wraps the VCM's ``get_stats`` endpoint. The endpoint runs at
        KERNEL ring; if the caller lacks permission, the action returns
        ``{"stats": None, "message": "<err>"}`` rather than raising.
        """
        try:
            stats = await self._get_vcm().get_stats()
        except Exception as e:
            logger.warning(
                "VCMCapability.get_vcm_stats failed: %s", e,
            )
            return {"stats": None, "message": str(e)}
        return {"stats": dict(stats or {}), "message": ""}

    # --- Filesystem watcher (Phase 3) --------------------------------------

    def _resolve_watch_paths(self, paths: list[str]) -> list[str]:
        """Validate and canonicalise watch paths.

        - Rejects empty lists (caller must give at least one path).
        - Expands ``~`` and resolves relative paths against ``watch_root``.
        - Drops duplicates while preserving order.
        - Does NOT stat the path at registration time. Non-existent paths
          cause the background watcher task to fail on its first iteration;
          the failure is logged and the watch stays registered so the
          caller can see it via ``list_watches()``. ``unwatch_repo()`` is
          the right way to clean up.
        """
        if not paths:
            raise ValueError("watch_repo requires at least one path")
        resolved: list[str] = []
        seen: set[str] = set()
        for p in paths:
            expanded = os.path.expanduser(p)
            if not os.path.isabs(expanded):
                expanded = os.path.join(self._watch_root, expanded)
            if expanded not in seen:
                seen.add(expanded)
                resolved.append(expanded)
        return resolved

    @action_executor()
    async def watch_repo(
        self,
        scope_id: str,
        *,
        paths: list[str] | None = None,
        on_change: _WatchMode = "notify_only",
        debounce_seconds: float = 5.0,
        watch_id: str | None = None,
    ) -> dict[str, Any]:
        """Start a filesystem watch that reacts to changes under ``paths``.

        Reactions (``on_change``):
          - ``"notify_only"`` (default) — emit a ``watch_fired`` event
            each time the watcher fires. Other agents decide what to do.
          - ``"invalidate"`` — additionally unmap ``scope_id`` so the
            next access to its pages will fault and re-fetch.
          - ``"reindex"`` — same as ``invalidate`` plus a ``reindexed``
            event. The capability does NOT automatically re-map the
            scope in v1 (that requires the original ``origin_url`` /
            ``branch`` / ``commit`` which the VCM does not currently
            surface); an observing agent should catch the event and
            decide whether to call ``mmap_repo`` again.

        Args:
            scope_id: Target scope this watch is associated with.
                Embedded in all emitted events.
            paths: Host filesystem paths to watch. Relative paths are
                resolved against the capability's ``watch_root``. At least
                one path is required.
            on_change: Reaction mode (see above).
            debounce_seconds: Minimum window over which consecutive
                filesystem events are coalesced into a single reaction.
            watch_id: Optional stable id for the watch (used by the
                resumer to restore a watch with its original identifier).
                Generated if not provided.

        Returns:
            ``{"watch_id", "scope_id", "paths", "on_change",
            "debounce_seconds", "started": bool, "message": str}``.
        """
        if len(self._watches) >= self._max_concurrent_watches:
            return {
                "watch_id": None, "scope_id": scope_id,
                "paths": list(paths or []), "on_change": on_change,
                "debounce_seconds": debounce_seconds,
                "started": False,
                "message": (
                    f"max_concurrent_watches ({self._max_concurrent_watches}) "
                    f"reached"
                ),
            }
        if on_change not in ("reindex", "invalidate", "notify_only"):
            return {
                "watch_id": None, "scope_id": scope_id,
                "paths": list(paths or []), "on_change": on_change,
                "debounce_seconds": debounce_seconds,
                "started": False,
                "message": (
                    "on_change must be 'reindex', 'invalidate', or "
                    "'notify_only'"
                ),
            }
        try:
            resolved = self._resolve_watch_paths(list(paths or []))
        except ValueError as e:
            return {
                "watch_id": None, "scope_id": scope_id,
                "paths": [], "on_change": on_change,
                "debounce_seconds": debounce_seconds,
                "started": False, "message": str(e),
            }

        wid = watch_id or f"watch_{uuid.uuid4().hex[:12]}"
        if wid in self._watches:
            return {
                "watch_id": wid, "scope_id": scope_id,
                "paths": list(resolved), "on_change": on_change,
                "debounce_seconds": debounce_seconds,
                "started": False,
                "message": f"watch_id {wid!r} already exists",
            }

        handle = _WatchHandle(
            watch_id=wid,
            scope_id=scope_id,
            paths=tuple(resolved),
            on_change=on_change,
            debounce_seconds=debounce_seconds,
            started_at=time.time(),
            stop_event=asyncio.Event(),
        )
        # Register first so the task's closure sees the handle on first
        # iteration (matters if the task fires immediately).
        self._watches[wid] = handle
        handle.task = asyncio.create_task(
            self._watch_loop(handle), name=f"vcm_watch:{wid}",
        )
        logger.info(
            "VCMCapability.watch_repo: started watch_id=%s scope_id=%s "
            "paths=%s on_change=%s",
            wid, scope_id, resolved, on_change,
        )
        return {
            "watch_id": wid, "scope_id": scope_id,
            "paths": list(resolved), "on_change": on_change,
            "debounce_seconds": debounce_seconds,
            "started": True, "message": "",
        }

    @action_executor()
    async def unwatch_repo(self, watch_id: str) -> dict[str, Any]:
        """Stop a previously started watch.

        Args:
            watch_id: Identifier returned by ``watch_repo``.

        Returns:
            ``{"watch_id", "stopped": bool, "message": str}``.
        """
        handle = self._watches.pop(watch_id, None)
        if handle is None:
            return {
                "watch_id": watch_id, "stopped": False,
                "message": "watch_id not found",
            }
        handle.stop_event.set()
        if handle.task is not None:
            try:
                await asyncio.wait_for(handle.task, timeout=5.0)
            except asyncio.TimeoutError:
                handle.task.cancel()
                logger.warning(
                    "VCMCapability.unwatch_repo: watch %r did not stop "
                    "within timeout; cancelled",
                    watch_id,
                )
            except Exception as e:
                logger.debug(
                    "VCMCapability.unwatch_repo: watcher %r exited with %r",
                    watch_id, e,
                )
        return {"watch_id": watch_id, "stopped": True, "message": ""}

    @action_executor()
    async def list_watches(self) -> dict[str, Any]:
        """List all active filesystem watches owned by this capability.

        Returns:
            ``{"watches": [...], "count": int}``. Each entry has
            ``watch_id``, ``scope_id``, ``paths``, ``on_change``,
            ``debounce_seconds``, ``started_at``, ``last_fired_at``,
            ``fire_count``.
        """
        watches = [
            {
                "watch_id": h.watch_id,
                "scope_id": h.scope_id,
                "paths": list(h.paths),
                "on_change": h.on_change,
                "debounce_seconds": h.debounce_seconds,
                "started_at": h.started_at,
                "last_fired_at": h.last_fired_at,
                "fire_count": h.fire_count,
            }
            for h in self._watches.values()
        ]
        return {"watches": watches, "count": len(watches)}

    async def _watch_loop(self, handle: _WatchHandle) -> None:
        """Background task that runs one ``watchfiles.awatch`` stream.

        Cancels cleanly when ``handle.stop_event`` is set. Each burst of
        filesystem events is debounced inside ``awatch`` (via its
        ``debounce`` parameter) and fires exactly one reaction.
        """
        try:
            from watchfiles import awatch
        except ImportError:  # pragma: no cover — watchfiles is a stdlib-adjacent dep
            logger.error(
                "VCMCapability: 'watchfiles' is not installed; "
                "watch_repo cannot run. Add it to pyproject.toml.",
            )
            return

        debounce_ms = max(1, int(handle.debounce_seconds * 1000))
        logger.debug(
            "VCMCapability._watch_loop: entering loop watch_id=%s paths=%s",
            handle.watch_id, handle.paths,
        )
        try:
            async for changes in awatch(
                *handle.paths,
                stop_event=handle.stop_event,
                debounce=debounce_ms,
            ):
                handle.last_fired_at = time.time()
                handle.fire_count += 1
                dirty_paths = sorted({p for _, p in changes})
                logger.info(
                    "VCMCapability: watch %r fired with %d change(s)",
                    handle.watch_id, len(dirty_paths),
                )
                await self._react_to_watch(handle, dirty_paths)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # pragma: no cover — defensive
            logger.exception(
                "VCMCapability._watch_loop for watch %r crashed: %s",
                handle.watch_id, e,
            )
        finally:
            logger.debug(
                "VCMCapability._watch_loop: exiting loop watch_id=%s",
                handle.watch_id,
            )

    async def _react_to_watch(
        self, handle: _WatchHandle, dirty_paths: list[str]
    ) -> None:
        """Emit the right events (and optionally unmap) for one fire."""
        base_payload = {
            "watch_id": handle.watch_id,
            "scope_id": handle.scope_id,
            "paths": dirty_paths,
            "ts": time.time(),
        }
        await self._emit_event(
            VCMEventProtocol.watch_fired_key(handle.watch_id),
            dict(base_payload),
        )
        if handle.on_change == "notify_only":
            return
        # invalidate + reindex both unmap the scope in v1 because the VCM
        # does not yet expose a per-page "evict and re-fetch" path.
        try:
            await self._get_vcm().munmap_application_scope(
                scope_id=handle.scope_id,
            )
            await self._emit_event(
                VCMEventProtocol.unmapped_key(handle.scope_id),
                {"scope_id": handle.scope_id, "ts": time.time()},
            )
        except Exception as e:
            logger.warning(
                "VCMCapability: invalidate for watch %r failed: %s",
                handle.watch_id, e,
            )
        if handle.on_change == "reindex":
            await self._emit_event(
                VCMEventProtocol.reindexed_key(handle.scope_id),
                dict(base_payload),
            )
