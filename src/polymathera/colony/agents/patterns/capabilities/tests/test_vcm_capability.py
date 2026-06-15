"""Unit tests for ``VCMCapability`` Phase 1 (mapping lifecycle).

These tests use a fake VCM deployment handle so they do not require Ray,
Redis, or a running cluster. The execution context is set via
``execution_context`` so scope-prefix helpers that read it can run.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.capabilities.vcm import VCMCapability
from polymathera.colony.agents.blackboard.protocol import VCMEventProtocol
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.vcm.models import MmapConfig, MmapResult
from polymathera.colony.distributed.ray_utils.serving.context import (
    execution_context,
    Ring,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeVCMHandle:
    """Minimal async stand-in for a Ray DeploymentHandle to VCM."""

    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.mmap_result: Any = None
        self.munmap_result: Any = None
        self.is_mapped: bool = False
        self.status_result: Any = None
        self.mapped_scopes: list[dict[str, Any]] = []
        # Phase 2
        self.fault_id: str = "fault-xyz"
        self.wait_returns: bool = True
        self.lock_raises_for: set[str] = set()
        self.unlock_returns: dict[str, bool] = {}
        self.unlock_raises_for: set[str] = set()
        self.extend_lock_returns: bool = True
        self.page_graph_data: dict[str, Any] = {
            "nodes": [], "edges": [],
            "node_count": 0, "edge_count": 0,
        }
        self.stored_pages: list[dict[str, Any]] = []
        self.pages_for_scope: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {"page_table": {}, "storage": {}}
        self.raise_on: set[str] = set()

    async def mmap_application_scope(self, **kwargs):
        self.calls.append(("mmap_application_scope", kwargs))
        if "mmap_application_scope" in self.raise_on:
            raise RuntimeError("simulated VCM failure")
        return self.mmap_result or MmapResult(
            status="mapped",
            scope_id=kwargs["scope_id"],
            message="ok",
        )

    async def munmap_application_scope(self, **kwargs):
        self.calls.append(("munmap_application_scope", kwargs))
        if "munmap_application_scope" in self.raise_on:
            raise RuntimeError("simulated VCM failure")
        return self.munmap_result or MmapResult(
            status="unmapped",
            scope_id=kwargs["scope_id"],
            message="ok",
        )

    async def is_application_scope_mapped(self, scope_id):
        self.calls.append(("is_application_scope_mapped", {"scope_id": scope_id}))
        if "is_application_scope_mapped" in self.raise_on:
            raise RuntimeError("simulated VCM failure")
        return self.is_mapped

    async def get_application_scope_mapping_status(self, scope_id):
        self.calls.append(("get_application_scope_mapping_status",
                           {"scope_id": scope_id}))
        if "get_application_scope_mapping_status" in self.raise_on:
            raise RuntimeError("simulated VCM failure")
        return self.status_result

    async def get_all_mapped_scopes(self):
        self.calls.append(("get_all_mapped_scopes", {}))
        if "get_all_mapped_scopes" in self.raise_on:
            raise RuntimeError("simulated VCM failure")
        return self.mapped_scopes

    # --- Phase 2 -----------------------------------------------------------

    async def issue_page_fault(self, **kwargs):
        self.calls.append(("issue_page_fault", kwargs))
        if "issue_page_fault" in self.raise_on:
            raise RuntimeError("simulated VCM failure")
        return self.fault_id

    async def wait_for_pages(self, **kwargs):
        self.calls.append(("wait_for_pages", kwargs))
        if "wait_for_pages" in self.raise_on:
            raise RuntimeError("simulated VCM failure")
        return self.wait_returns

    async def lock_page(self, **kwargs):
        self.calls.append(("lock_page", kwargs))
        page_id = kwargs["page_id"]
        if page_id in self.lock_raises_for:
            raise RuntimeError(f"simulated lock failure for {page_id}")
        return object()  # PageLock stand-in

    async def unlock_page(self, *, page_id):
        self.calls.append(("unlock_page", {"page_id": page_id}))
        if page_id in self.unlock_raises_for:
            raise RuntimeError(f"simulated unlock failure for {page_id}")
        return self.unlock_returns.get(page_id, True)

    async def extend_page_lock(self, *, page_id, additional_duration_s):
        self.calls.append((
            "extend_page_lock",
            {"page_id": page_id, "additional_duration_s": additional_duration_s},
        ))
        if "extend_page_lock" in self.raise_on:
            raise RuntimeError("simulated VCM failure")
        return self.extend_lock_returns

    async def get_page_graph_data(self, *, max_nodes=5000):
        self.calls.append(("get_page_graph_data", {"max_nodes": max_nodes}))
        if "get_page_graph_data" in self.raise_on:
            raise RuntimeError("simulated VCM failure")
        return self.page_graph_data

    async def list_stored_pages(self, **kwargs):
        self.calls.append(("list_stored_pages", kwargs))
        if "list_stored_pages" in self.raise_on:
            raise RuntimeError("simulated VCM failure")
        return self.stored_pages

    async def get_pages_for_scope(self, **kwargs):
        self.calls.append(("get_pages_for_scope", kwargs))
        if "get_pages_for_scope" in self.raise_on:
            raise RuntimeError("simulated VCM failure")
        return self.pages_for_scope

    async def get_stats(self):
        self.calls.append(("get_stats", {}))
        if "get_stats" in self.raise_on:
            raise RuntimeError("simulated VCM failure")
        return self.stats


class _FakeBlackboard:
    """In-memory blackboard that just records write calls.

    Used to assert that the capability emits the right VCM lifecycle
    events without requiring Redis / the real EnhancedBlackboard.
    """

    def __init__(self):
        self.writes: list[tuple[str, dict[str, Any]]] = []

    async def write(self, key, value):
        self.writes.append((key, value))


def _make_capability(
    fake_handle: _FakeVCMHandle,
    *,
    fake_blackboard: _FakeBlackboard | None = None,
) -> VCMCapability:
    """Build a VCMCapability with a synthetic agent, VCM handle and
    optional fake blackboard."""
    agent = MagicMock()
    agent.agent_id = "agent-test"
    cap = VCMCapability(agent=agent, scope=BlackboardScope.SESSION)
    # Short-circuit get_vcm() resolution.
    cap._vcm_handle = fake_handle
    # Short-circuit get_blackboard() resolution (event emission).
    if fake_blackboard is not None:
        cap._blackboard = fake_blackboard
    return cap


def _run(coro):
    """Run a single coroutine on the current thread's event loop.

    Tests that spawn filesystem watchers are declared ``async def`` and
    awaited directly by pytest-asyncio — they must not reach this
    helper. Non-watcher tests are fine here because no tasks outlive
    the call.
    """
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

def test_bind_round_trips_through_cloudpickle():
    # Ray's vendored cloudpickle — see comment in
    # test_github_capability for why standalone PyPI cloudpickle is
    # not the right import here.
    from ray import cloudpickle
    bp = VCMCapability.bind(scope=BlackboardScope.SESSION)
    raw = cloudpickle.dumps(bp)
    bp2 = cloudpickle.loads(raw)
    # AgentCapabilityBlueprint still carries the configured scope kwarg.
    assert bp2.kwargs["scope"] == BlackboardScope.SESSION


def test_action_executors_are_registered():
    import inspect
    keys = {
        m._action_key for _, m in inspect.getmembers(
            VCMCapability, predicate=inspect.isfunction
        ) if getattr(m, "_action_key", None)
    }
    # Phase 1 (mapping) + Phase 2 (page lifecycle) + Phase 3 (watcher).
    assert keys == {
        # Phase 1
        "mmap_repo", "mmap_blackboard_scope", "munmap_scope",
        "is_scope_mapped", "get_scope_status", "list_mapped_scopes",
        # Phase 2
        "request_pages", "lock_pages", "unlock_pages", "extend_lock",
        "get_page_graph", "list_stored_pages", "get_pages_for_scope",
        "get_vcm_stats",
        # Phase 3
        "watch_repo", "unwatch_repo", "list_watches",
    }


# ---------------------------------------------------------------------------
# mmap_repo
# ---------------------------------------------------------------------------

def test_mmap_repo_requires_exactly_one_source():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        # Both set → error; zero set → error.
        r1 = _run(cap.mmap_repo(
            origin_url="https://x", local_repo_path="/tmp/y"
        ))
        r2 = _run(cap.mmap_repo())
    for r in (r1, r2):
        assert r["status"] == "error"
        assert "Exactly one" in r["message"]
    assert fake.calls == []


def test_mmap_repo_delegates_origin_url_to_vcm():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.mmap_repo(
            origin_url="https://github.com/org/repo",
            branch="main",
            commit="abc123",
        ))
    assert result["status"] == "mapped"
    assert result["origin_url"] == "https://github.com/org/repo"
    assert result["branch"] == "main"
    assert result["commit"] == "abc123"
    assert len(fake.calls) == 1
    name, kwargs = fake.calls[0]
    assert name == "mmap_application_scope"
    assert kwargs["origin_url"] == "https://github.com/org/repo"
    assert kwargs["branch"] == "main"
    assert kwargs["commit"] == "abc123"
    assert kwargs["source_type"] == "codebase"
    assert isinstance(kwargs["config"], MmapConfig)


def test_mmap_repo_converts_local_path_to_file_url():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.mmap_repo(local_repo_path="/mnt/shared/repo"))
    assert result["status"] == "mapped"
    assert result["origin_url"] == "file:///mnt/shared/repo"
    _, kwargs = fake.calls[0]
    assert kwargs["origin_url"] == "file:///mnt/shared/repo"


def test_mmap_repo_uses_explicit_scope_id_when_provided():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.mmap_repo(
            origin_url="https://x", scope_id="custom:scope:42"
        ))
    assert result["scope_id"] == "custom:scope:42"
    _, kwargs = fake.calls[0]
    assert kwargs["scope_id"] == "custom:scope:42"


def test_mmap_repo_surfaces_vcm_exception_as_error_dict():
    fake = _FakeVCMHandle()
    fake.raise_on.add("mmap_application_scope")
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.mmap_repo(origin_url="https://x"))
    assert result["status"] == "error"
    assert "simulated VCM failure" in result["message"]


def test_mmap_repo_already_mapped_preserves_status():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        # MmapResult has a syscontext default that resolves via
        # require_execution_context() — construct it inside the block.
        fake.mmap_result = MmapResult(
            status="already_mapped",
            scope_id="ignored",
            message="scope exists",
        )
        cap = _make_capability(fake)
        result = _run(cap.mmap_repo(origin_url="https://x"))
    assert result["status"] == "already_mapped"
    assert result["message"] == "scope exists"


def test_mmap_repo_accepts_dict_results_from_deployment_handle():
    """Ray DeploymentHandle may return plain dicts (not MmapResult) —
    the normalizer must handle both."""
    fake = _FakeVCMHandle()
    fake.mmap_result = {
        "status": "mapped",
        "scope_id": "s:ok",
        "message": "",
    }
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.mmap_repo(origin_url="https://x"))
    assert result["status"] == "mapped"
    assert result["scope_id"] == "s:ok"


# ---------------------------------------------------------------------------
# mmap_blackboard_scope
# ---------------------------------------------------------------------------

def test_mmap_blackboard_scope_uses_blackboard_source_type():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.mmap_blackboard_scope(namespace="findings"))
    assert result["status"] == "mapped"
    _, kwargs = fake.calls[0]
    assert kwargs["source_type"] == "blackboard"
    assert kwargs["scope_id"].endswith(":findings")


# ---------------------------------------------------------------------------
# munmap / status / list
# ---------------------------------------------------------------------------

def test_munmap_scope_passes_scope_through():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        fake.munmap_result = MmapResult(
            status="unmapped", scope_id="custom:1", message=""
        )
        cap = _make_capability(fake)
        result = _run(cap.munmap_scope(scope_id="custom:1"))
    assert result["status"] == "unmapped"
    assert fake.calls[0] == (
        "munmap_application_scope", {"scope_id": "custom:1"}
    )


def test_is_scope_mapped_returns_bool_in_dict():
    fake = _FakeVCMHandle()
    fake.is_mapped = True
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.is_scope_mapped(scope_id="abc"))
    assert result == {"scope_id": "abc", "mapped": True, "message": ""}


def test_get_scope_status_reports_not_mapped_when_vcm_returns_none():
    fake = _FakeVCMHandle()
    fake.status_result = None
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.get_scope_status(scope_id="ghost"))
    assert result == {"status": "not_mapped", "scope_id": "ghost"}


def test_list_mapped_scopes_returns_count_and_empty_on_failure():
    fake = _FakeVCMHandle()
    fake.raise_on.add("get_all_mapped_scopes")
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.list_mapped_scopes())
    assert result["count"] == 0
    assert result["mappings"] == []
    assert "simulated VCM failure" in result["message"]


# ---------------------------------------------------------------------------
# Event emission (Phase 2 — wired into Phase 1 actions too)
# ---------------------------------------------------------------------------

def test_mmap_repo_emits_mapped_event_on_success():
    fake = _FakeVCMHandle()
    bb = _FakeBlackboard()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake, fake_blackboard=bb)
        _run(cap.mmap_repo(origin_url="https://x", branch="main"))
    # Exactly one write, targeting the mapped_key for the returned scope.
    assert len(bb.writes) == 1
    key, value = bb.writes[0]
    assert key.startswith("mapped:")
    parsed = VCMEventProtocol.parse_mapped_key(key)
    assert parsed == value["scope_id"]
    assert value["source_type"] == "codebase"
    assert value["origin_url"] == "https://x"
    assert value["branch"] == "main"
    assert "ts" in value


def test_mmap_repo_does_not_emit_event_on_already_mapped():
    fake = _FakeVCMHandle()
    bb = _FakeBlackboard()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        # Only "mapped" triggers emission; "already_mapped" is not new.
        fake.mmap_result = MmapResult(
            status="already_mapped", scope_id="s:x", message="",
        )
        cap = _make_capability(fake, fake_blackboard=bb)
        _run(cap.mmap_repo(origin_url="https://x"))
    assert bb.writes == []


def test_munmap_scope_emits_unmapped_event_on_success():
    fake = _FakeVCMHandle()
    bb = _FakeBlackboard()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        fake.munmap_result = MmapResult(
            status="unmapped", scope_id="s:y", message="",
        )
        cap = _make_capability(fake, fake_blackboard=bb)
        _run(cap.munmap_scope(scope_id="s:y"))
    assert len(bb.writes) == 1
    key, value = bb.writes[0]
    assert key == VCMEventProtocol.unmapped_key("s:y")
    assert value["scope_id"] == "s:y"
    assert "ts" in value


def test_mmap_blackboard_scope_emits_mapped_event_with_blackboard_source():
    fake = _FakeVCMHandle()
    bb = _FakeBlackboard()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake, fake_blackboard=bb)
        _run(cap.mmap_blackboard_scope(namespace="findings"))
    assert len(bb.writes) == 1
    _, value = bb.writes[0]
    assert value["source_type"] == "blackboard"


# ---------------------------------------------------------------------------
# Page lifecycle
# ---------------------------------------------------------------------------

def test_request_pages_issues_fault_and_waits_with_lock():
    fake = _FakeVCMHandle()
    bb = _FakeBlackboard()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake, fake_blackboard=bb)
        result = _run(cap.request_pages(
            page_ids=["p1", "p2"],
            timeout_s=5.0,
            priority=20,
            lock_duration_s=60.0,
            lock_reason="analysis",
        ))
    assert result["loaded"] is True
    assert result["fault_id"] == "fault-xyz"
    assert result["page_ids"] == ["p1", "p2"]
    # The capability must issue the fault, then wait, in order.
    seq = [name for name, _ in fake.calls]
    assert seq == ["issue_page_fault", "wait_for_pages"]
    _, fault_kwargs = fake.calls[0]
    assert fault_kwargs["page_ids"] == ["p1", "p2"]
    assert fault_kwargs["priority"] == 20
    assert fault_kwargs["lock_duration_s"] == 60.0
    assert fault_kwargs["lock_reason"] == "analysis"
    # And an event must be emitted.
    assert len(bb.writes) == 1
    key, value = bb.writes[0]
    assert key == VCMEventProtocol.page_fault_key("fault-xyz")
    assert value["page_ids"] == ["p1", "p2"]


def test_request_pages_empty_short_circuits():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.request_pages(page_ids=[]))
    assert result == {
        "fault_id": None, "loaded": True,
        "page_ids": [], "message": "no pages requested",
    }
    assert fake.calls == []


def test_request_pages_reports_timeout_and_still_emits_event():
    fake = _FakeVCMHandle()
    fake.wait_returns = False
    bb = _FakeBlackboard()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake, fake_blackboard=bb)
        result = _run(cap.request_pages(page_ids=["p1"], timeout_s=1.5))
    assert result["loaded"] is False
    assert "timeout" in result["message"]
    # Event still emitted — the fault was issued even if the wait timed out.
    assert len(bb.writes) == 1


def test_lock_pages_partitions_successes_and_failures():
    fake = _FakeVCMHandle()
    fake.lock_raises_for.add("p2")
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.lock_pages(
            page_ids=["p1", "p2", "p3"], ttl_s=30.0, reason="test",
        ))
    assert result["locked"] == ["p1", "p3"]
    assert result["failed"] == [
        {"page_id": "p2", "error": "simulated lock failure for p2"}
    ]


def test_lock_pages_rejects_non_positive_ttl():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.lock_pages(page_ids=["p1"], ttl_s=0))
    assert result["locked"] == []
    assert result["failed"] == [
        {"page_id": "p1", "error": "ttl_s must be positive"}
    ]
    # No RPC call was attempted.
    assert fake.calls == []


def test_unlock_pages_distinguishes_unlocked_from_already_unlocked():
    fake = _FakeVCMHandle()
    fake.unlock_returns = {"p1": True, "p2": False}
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.unlock_pages(page_ids=["p1", "p2"]))
    assert result["unlocked"] == ["p1"]
    assert result["already_unlocked"] == ["p2"]
    assert result["failed"] == []


def test_extend_lock_reports_false_when_page_not_locked():
    fake = _FakeVCMHandle()
    fake.extend_lock_returns = False
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.extend_lock(page_id="p1", additional_s=10.0))
    assert result == {
        "page_id": "p1", "extended": False,
        "message": "page was not locked",
    }


def test_get_page_graph_returns_graph_dict_with_message_field():
    fake = _FakeVCMHandle()
    fake.page_graph_data = {
        "nodes": [{"id": "p1"}],
        "edges": [{"src": "p1", "tgt": "p2"}],
        "node_count": 1, "edge_count": 1,
    }
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.get_page_graph(max_nodes=100))
    assert result["node_count"] == 1
    assert result["message"] == ""
    _, kwargs = fake.calls[0]
    assert kwargs["max_nodes"] == 100


def test_list_stored_pages_forwards_filter_and_pagination():
    fake = _FakeVCMHandle()
    fake.stored_pages = [{"page_id": "p1"}, {"page_id": "p2"}]
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.list_stored_pages(
            source_pattern="github.com", limit=50, offset=10,
        ))
    assert result["count"] == 2
    _, kwargs = fake.calls[0]
    assert kwargs == {
        "source_pattern": "github.com", "limit": 50, "offset": 10,
    }


def test_get_pages_for_scope_passes_scope_and_metadata_flag():
    fake = _FakeVCMHandle()
    fake.pages_for_scope = [{"page_id": "p1"}]
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.get_pages_for_scope(
            scope_id="abc", include_metadata=True,
        ))
    assert result["scope_id"] == "abc"
    assert result["count"] == 1
    _, kwargs = fake.calls[0]
    assert kwargs == {"scope_id": "abc", "include_metadata": True}


def test_get_vcm_stats_degrades_to_none_on_failure():
    fake = _FakeVCMHandle()
    fake.raise_on.add("get_stats")
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.get_vcm_stats())
    assert result == {"stats": None, "message": "simulated VCM failure"}


# ---------------------------------------------------------------------------
# VCMEventProtocol (scope_ids contain colons — parse must be colon-safe)
# ---------------------------------------------------------------------------

def test_vcm_event_protocol_round_trips_colon_bearing_scope_ids():
    scope_id = (
        "user_abc:colony:colony_def:session:session_ghi:agent:agent-xyz"
    )
    mapped = VCMEventProtocol.mapped_key(scope_id)
    assert VCMEventProtocol.parse_mapped_key(mapped) == scope_id
    unmapped = VCMEventProtocol.unmapped_key(scope_id)
    assert VCMEventProtocol.parse_unmapped_key(unmapped) == scope_id
    reindexed = VCMEventProtocol.reindexed_key(scope_id)
    assert VCMEventProtocol.parse_reindexed_key(reindexed) == scope_id


def test_vcm_event_protocol_rejects_foreign_prefix():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        VCMEventProtocol.parse_mapped_key("something_else:xyz")


# ---------------------------------------------------------------------------
# Filesystem watcher (Phase 3)
# ---------------------------------------------------------------------------

def test_watch_repo_rejects_empty_paths():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.watch_repo(scope_id="s:x", paths=[]))
    assert result["started"] is False
    assert "at least one path" in result["message"]


def test_watch_repo_rejects_unknown_on_change_value():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.watch_repo(
            scope_id="s:x",
            paths=["/tmp/repo"],
            on_change="wat",  # invalid
        ))
    assert result["started"] is False
    assert "on_change" in result["message"]


def test_watch_repo_enforces_max_concurrent_watches():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        cap._max_concurrent_watches = 1
        # Pre-populate with one watch (no task attached — we never await
        # changes; we only test the admission check).
        cap._watches["existing"] = _make_dummy_handle(scope_id="s:z")
        result = _run(cap.watch_repo(
            scope_id="s:x", paths=["/tmp/a"],
        ))
    assert result["started"] is False
    assert "max_concurrent_watches" in result["message"]


async def test_watch_repo_resolves_relative_paths_against_watch_root(tmp_path):
    """Path canonicalisation is pure string manipulation — the watcher
    task itself may crash on a non-existent path, but that crash is
    handled inside the background task and does not affect
    registration. The returned ``paths`` reflect the resolved form."""
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        cap._watch_root = str(tmp_path)
        rel_dir = tmp_path / "repo_a"
        rel_dir.mkdir()
        abs_dir = tmp_path / "abs_b"
        abs_dir.mkdir()
        result = await cap.watch_repo(
            scope_id="s:x", paths=["repo_a", str(abs_dir)],
        )
        assert result["started"] is True
        wid = result["watch_id"]
        assert result["paths"] == [str(rel_dir), str(abs_dir)]
        await cap.unwatch_repo(watch_id=wid)


def _make_dummy_handle(*, scope_id: str, on_change: str = "notify_only"):
    """Build a _WatchHandle with a no-op task for unit tests."""
    from polymathera.colony.agents.patterns.capabilities.vcm import _WatchHandle
    return _WatchHandle(
        watch_id="dummy",
        scope_id=scope_id,
        paths=("/tmp/x",),
        on_change=on_change,
        debounce_seconds=1.0,
        started_at=time.time(),
        stop_event=asyncio.Event(),
    )


async def test_list_watches_reports_active_watches(tmp_path):
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        start = await cap.watch_repo(
            scope_id="s:x",
            paths=[str(tmp_path)],
            on_change="notify_only",
        )
        listed = await cap.list_watches()
        await cap.unwatch_repo(watch_id=start["watch_id"])
    assert listed["count"] == 1
    entry = listed["watches"][0]
    assert entry["watch_id"] == start["watch_id"]
    assert entry["scope_id"] == "s:x"
    assert entry["on_change"] == "notify_only"


def test_unwatch_repo_returns_not_found_for_unknown_id():
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        result = _run(cap.unwatch_repo(watch_id="nonexistent"))
    assert result == {
        "watch_id": "nonexistent", "stopped": False,
        "message": "watch_id not found",
    }


def test_react_notify_only_emits_watch_fired_only():
    fake = _FakeVCMHandle()
    bb = _FakeBlackboard()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake, fake_blackboard=bb)
        handle = _make_dummy_handle(scope_id="s:x", on_change="notify_only")
        _run(cap._react_to_watch(handle, ["/tmp/x/a.py"]))
    assert [w[0] for w in bb.writes] == [
        VCMEventProtocol.watch_fired_key(handle.watch_id),
    ]
    # VCM is untouched in notify_only mode.
    assert fake.calls == []


def test_react_invalidate_unmaps_scope_and_emits_unmapped():
    fake = _FakeVCMHandle()
    bb = _FakeBlackboard()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        fake.munmap_result = MmapResult(
            status="unmapped", scope_id="s:x", message="",
        )
        cap = _make_capability(fake, fake_blackboard=bb)
        handle = _make_dummy_handle(scope_id="s:x", on_change="invalidate")
        _run(cap._react_to_watch(handle, ["/tmp/x/a.py"]))
    keys = [w[0] for w in bb.writes]
    assert VCMEventProtocol.watch_fired_key(handle.watch_id) in keys
    assert VCMEventProtocol.unmapped_key("s:x") in keys
    # And the capability actually called munmap on the VCM.
    assert any(c[0] == "munmap_application_scope" for c in fake.calls)


def test_react_reindex_emits_both_unmapped_and_reindexed():
    fake = _FakeVCMHandle()
    bb = _FakeBlackboard()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        fake.munmap_result = MmapResult(
            status="unmapped", scope_id="s:x", message="",
        )
        cap = _make_capability(fake, fake_blackboard=bb)
        handle = _make_dummy_handle(scope_id="s:x", on_change="reindex")
        _run(cap._react_to_watch(handle, ["/tmp/x/a.py"]))
    keys = [w[0] for w in bb.writes]
    assert VCMEventProtocol.watch_fired_key(handle.watch_id) in keys
    assert VCMEventProtocol.unmapped_key("s:x") in keys
    assert VCMEventProtocol.reindexed_key("s:x") in keys


async def test_suspension_state_round_trips_watch_configs(tmp_path):
    from polymathera.colony.agents.models import AgentSuspensionState

    watched = tmp_path / "repo"
    watched.mkdir()
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        r = await cap.watch_repo(
            scope_id="s:x", paths=[str(watched)],
            on_change="invalidate", debounce_seconds=2.5,
        )
        state = AgentSuspensionState(
            agent_id="agent-test",
            agent_type="test",
            agent_state="suspended",
            suspension_reason="unit test",
        )
        await cap.serialize_suspension_state(state)
        serialised = list(state.action_policy_state.custom["vcm_watches"])
        await cap.unwatch_repo(watch_id=r["watch_id"])
        assert len(cap._watches) == 0

        await cap.deserialize_suspension_state(state)
        assert len(cap._watches) == 1
        restored = next(iter(cap._watches.values()))
        await cap.unwatch_repo(watch_id=restored.watch_id)

    # The restored watch carries the original id and configuration.
    [entry] = serialised
    assert entry["watch_id"] == r["watch_id"]
    assert entry["scope_id"] == "s:x"
    assert entry["paths"] == [str(watched)]
    assert entry["on_change"] == "invalidate"
    assert entry["debounce_seconds"] == 2.5


async def test_shutdown_stops_every_active_watch(tmp_path):
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    dir_b = tmp_path / "b"
    dir_b.mkdir()
    fake = _FakeVCMHandle()
    with execution_context(ring=Ring.USER, tenant_id="t1", colony_id="c1",
                            session_id="s1"):
        cap = _make_capability(fake)
        await cap.watch_repo(scope_id="s:x", paths=[str(dir_a)])
        await cap.watch_repo(scope_id="s:y", paths=[str(dir_b)])
        assert len(cap._watches) == 2
        await cap.shutdown()
    assert cap._watches == {}
