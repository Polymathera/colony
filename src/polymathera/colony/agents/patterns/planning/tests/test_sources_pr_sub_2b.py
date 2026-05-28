"""Tests for the colony-side cross-process consciousness-stream
sources.

The cross-process design (canonical for this codebase): producers
publish typed records to colony-scoped blackboard protocols; per-agent
:class:`~polymathera.colony.agents.patterns.planning.sources.ColonyScopedEventSource`
subclasses subscribe via ``@event_handler`` and translate each event
into a ``record_stream_entry`` on the owning policy.

Covered:

1. :class:`VCMPageEventSource` event handler translates a
   :class:`VCMPageEventProtocol` payload into a ``vcm_update`` entry
   (with page_prefix filter).
2. :class:`MonorepoCommitEventSource` event handler translates a
   :class:`MonorepoCommitProtocol` payload into a ``monorepo_commit``
   entry (with branch + capability_fqn_prefix filters).
3. ``VirtualContextManager._publish_page_event`` writes a typed
   record to the colony blackboard via
   :class:`VCMPageEventProtocol`.
4. ``BranchScopedCapabilityBase.fire_post_commit`` writes a typed
   record to the colony blackboard via
   :class:`MonorepoCommitProtocol`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.blackboard import (
    BlackboardEvent,
    MonorepoCommitProtocol,
    VCMPageEventProtocol,
)
from polymathera.colony.agents.patterns.actions.policies import BaseActionPolicy
from polymathera.colony.agents.patterns.planning.sources import (
    MonorepoCommitEventSource,
    VCMPageEventSource,
)
from polymathera.colony.agents.patterns.planning.streams import (
    ConsciousnessStream,
    JSONStreamFormatter,
)


def _agent() -> Any:
    agent = MagicMock()
    agent.agent_id = "agent_pr_sub_2b"
    return agent


def _stream(**kw) -> ConsciousnessStream:
    return ConsciousnessStream(
        name=kw.pop("name", "s"),
        formatter=JSONStreamFormatter(section_title="## S"),
        **kw,
    )


def _accept_all(_p: Any) -> bool:
    return True


def _bb_event(key: str, value: dict[str, Any]) -> BlackboardEvent:
    return BlackboardEvent(
        event_type="write",
        key=key,
        value=value,
    )


# ---------------------------------------------------------------------------
# 0. End-to-end attach: colony-scoped subscription delivers to the policy
#    event queue when a producer writes to the colony blackboard.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attach_subscribes_colony_scope_and_delivers_to_queue() -> None:
    """attach() must (a) register the source as a capability on the
    agent, and (b) subscribe the colony-scoped protocol pattern on the
    colony blackboard so producer writes land on the policy's event
    queue. This is the cross-process delivery contract."""
    import asyncio

    from polymathera.colony.agents.blackboard import EnhancedBlackboard
    from polymathera.colony.distributed.ray_utils.serving import (
        Ring,
        execution_context,
        require_execution_context,
    )

    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ):
        # Real in-memory colony blackboard the source will subscribe on.
        colony_bb = EnhancedBlackboard(
            app_name="test_app", scope_id="colony", backend_type="memory",
            enable_events=True,
        )
        await colony_bb.initialize()

        # Stub agent + policy. add_capability / get_capabilities are
        # real enough for the source's needs. The agent's syscontext
        # must match the ambient one so get_agent_level_scope resolves.
        captured_caps: dict[str, Any] = {}
        agent = MagicMock()
        agent.agent_id = "agent_e2e"
        agent.syscontext = require_execution_context()
        agent.add_capability = MagicMock(
            side_effect=lambda cap, **kw: captured_caps.__setitem__(
                cap.capability_key, cap,
            ),
        )
        agent.get_capabilities = MagicMock(
            side_effect=lambda: list(captured_caps.values()),
        )

        policy = BaseActionPolicy(agent=agent, consciousness_streams=[])
        # Give the policy the event-queue surface attach reads.
        event_queue: asyncio.Queue = asyncio.Queue()
        policy.get_event_queue = MagicMock(return_value=event_queue)
        policy._subscribed_providers = set()
        policy._high_priority_event_queue = asyncio.Queue()

        src = VCMPageEventSource()
        # Pre-inject the colony blackboard so attach subscribes on it.
        src._colony_blackboard = colony_bb
        await src.attach(policy)

        # (a) capability registered on the agent.
        assert any(
            isinstance(c, VCMPageEventSource) for c in captured_caps.values()
        )

        # (b) producer write to the colony scope lands on the queue.
        key = VCMPageEventProtocol.event_key("added", "kb/lit/p1", 1)
        await colony_bb.write(
            key=key, value={"kind": "added", "page_id": "kb/lit/p1"},
        )
        delivered = await asyncio.wait_for(event_queue.get(), timeout=1.0)
        assert delivered.key == key


# ---------------------------------------------------------------------------
# 1. VCMPageEventSource handler — payload translation + filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVCMPageEventSourceHandler:
    async def test_handler_records_vcm_update(self) -> None:
        s = _stream(vcm_update_filter=_accept_all)
        policy = BaseActionPolicy(agent=_agent(), consciousness_streams=[s])
        src = VCMPageEventSource()
        # Bypass attach() — exercise the handler directly with the
        # policy already wired.
        src._policy = policy

        payload = {
            "kind": "added",
            "page_id": "kb/literature/k2003.pdf",
            "deployment_name": "lit_source",
        }
        event = _bb_event(
            key=VCMPageEventProtocol.event_key(
                mutation_kind="added",
                page_id="kb/literature/k2003.pdf",
                millis=1,
            ),
            value=payload,
        )
        await src._on_vcm_page_event(event, None)

        assert len(s._entries) == 1
        entry = s._entries[0]
        assert entry["kind"] == "vcm_update"
        assert entry["payload"]["kind"] == "added"
        assert entry["payload"]["page_id"] == "kb/literature/k2003.pdf"
        assert entry["payload"]["page_source"] == "lit_source"

    async def test_page_prefix_narrows(self) -> None:
        s = _stream(vcm_update_filter=_accept_all)
        policy = BaseActionPolicy(agent=_agent(), consciousness_streams=[s])
        src = VCMPageEventSource(page_prefix="kb/literature/")
        src._policy = policy

        # In prefix → recorded.
        await src._on_vcm_page_event(
            _bb_event(
                key=VCMPageEventProtocol.event_key("added", "kb/literature/p1", 1),
                value={"kind": "added", "page_id": "kb/literature/p1"},
            ),
            None,
        )
        # Out of prefix → skipped.
        await src._on_vcm_page_event(
            _bb_event(
                key=VCMPageEventProtocol.event_key("added", "kb/standards/p2", 2),
                value={"kind": "added", "page_id": "kb/standards/p2"},
            ),
            None,
        )

        assert len(s._entries) == 1
        assert s._entries[0]["payload"]["page_id"] == "kb/literature/p1"

    async def test_handler_silent_when_no_policy(self) -> None:
        src = VCMPageEventSource()
        # No policy bound — handler returns silently.
        await src._on_vcm_page_event(
            _bb_event(
                key=VCMPageEventProtocol.event_key("added", "p", 1),
                value={"kind": "added", "page_id": "p"},
            ),
            None,
        )


# ---------------------------------------------------------------------------
# 2. MonorepoCommitEventSource handler — payload translation + filters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMonorepoCommitEventSourceHandler:
    async def test_handler_records_monorepo_commit(self) -> None:
        s = _stream(monorepo_commit_filter=_accept_all)
        policy = BaseActionPolicy(agent=_agent(), consciousness_streams=[s])
        src = MonorepoCommitEventSource()
        src._policy = policy

        payload = {
            "sha": "deadbeef",
            "branch": "main",
            "message": "L2 supply_chain: bom",
            "paths": ["design/bom/bom.yaml"],
            "capability_fqn": "polymathera.cps.agents.supply_chain.SupplyChainCapability",
        }
        event = _bb_event(
            key=MonorepoCommitProtocol.event_key(branch="main", sha="deadbeef"),
            value=payload,
        )
        await src._on_monorepo_commit(event, None)

        assert len(s._entries) == 1
        entry = s._entries[0]
        assert entry["kind"] == "monorepo_commit"
        assert entry["payload"]["sha"] == "deadbeef"
        assert entry["payload"]["branch"] == "main"
        assert entry["payload"]["message"].startswith("L2 supply_chain")

    async def test_branch_filter_narrows(self) -> None:
        s = _stream(monorepo_commit_filter=_accept_all)
        policy = BaseActionPolicy(agent=_agent(), consciousness_streams=[s])
        src = MonorepoCommitEventSource(branch="main")
        src._policy = policy

        # Fork branch — rejected.
        await src._on_monorepo_commit(
            _bb_event(
                key=MonorepoCommitProtocol.event_key("fork/experiment", "x"),
                value={
                    "sha": "x", "branch": "fork/experiment",
                    "message": "m", "paths": [], "capability_fqn": "x",
                },
            ),
            None,
        )
        # main — accepted.
        await src._on_monorepo_commit(
            _bb_event(
                key=MonorepoCommitProtocol.event_key("main", "y"),
                value={
                    "sha": "y", "branch": "main",
                    "message": "m", "paths": [], "capability_fqn": "x",
                },
            ),
            None,
        )
        assert len(s._entries) == 1
        assert s._entries[0]["payload"]["sha"] == "y"

    async def test_capability_fqn_prefix_filter_narrows(self) -> None:
        s = _stream(monorepo_commit_filter=_accept_all)
        policy = BaseActionPolicy(agent=_agent(), consciousness_streams=[s])
        src = MonorepoCommitEventSource(
            capability_fqn_prefix="polymathera.cps",
        )
        src._policy = policy

        # In-prefix — accepted.
        await src._on_monorepo_commit(
            _bb_event(
                key=MonorepoCommitProtocol.event_key("main", "x"),
                value={
                    "sha": "x", "branch": "main", "message": "m",
                    "paths": [],
                    "capability_fqn": "polymathera.cps.agents.regulatory.RegulatoryCapability",
                },
            ),
            None,
        )
        # Out-of-prefix — rejected.
        await src._on_monorepo_commit(
            _bb_event(
                key=MonorepoCommitProtocol.event_key("main", "y"),
                value={
                    "sha": "y", "branch": "main", "message": "m",
                    "paths": [],
                    "capability_fqn": "polymathera.colony.foo.SomeCap",
                },
            ),
            None,
        )
        assert len(s._entries) == 1
        assert s._entries[0]["payload"]["sha"] == "x"

    async def test_handler_silent_when_no_policy(self) -> None:
        src = MonorepoCommitEventSource()
        await src._on_monorepo_commit(
            _bb_event(
                key=MonorepoCommitProtocol.event_key("main", "x"),
                value={
                    "sha": "x", "branch": "main", "message": "m",
                    "paths": [], "capability_fqn": "x",
                },
            ),
            None,
        )


# ---------------------------------------------------------------------------
# 3. VirtualContextManager._publish_page_event writes via the protocol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVCMPublishPageEvent:
    async def _vcm_stand_in(self):
        """Construct just the slice of VCM state needed to exercise
        ``_publish_page_event``: a captured ``app_name`` + a mocked
        colony-blackboard helper."""
        from polymathera.colony.vcm.manager import VirtualContextManager

        vcm = object.__new__(VirtualContextManager)
        vcm.app_name = "test_app"
        vcm._colony_blackboard = MagicMock()
        # Pre-set so the lazy helper short-circuits.
        async def _async_noop(*a, **kw):
            return None
        vcm._colony_blackboard.write = MagicMock(side_effect=_async_noop)

        async def _get_bb():
            return vcm._colony_blackboard
        vcm._get_colony_blackboard = _get_bb
        return vcm

    async def test_publish_page_event_writes_typed_record(self) -> None:
        vcm = await self._vcm_stand_in()
        event = MagicMock(
            page_id="kb/literature/k2003.pdf",
            deployment_name="lit_source",
            client_id="vllm_0",
            size=2048,
            timestamp=12345.0,
            event_type="page_loaded",
        )
        await vcm._publish_page_event("added", event)
        # Single write call to the colony blackboard.
        assert vcm._colony_blackboard.write.call_count == 1
        call = vcm._colony_blackboard.write.call_args
        key = call.kwargs["key"]
        value = call.kwargs["value"]
        assert key.startswith(VCMPageEventProtocol._PREFIX)
        # The key identifies the mutation kind; the canonical page_id
        # is carried in the value payload (consumers read it there).
        parsed = VCMPageEventProtocol.parse_event_key(key)
        assert parsed["mutation_kind"] == "added"
        assert value["kind"] == "added"
        assert value["page_id"] == "kb/literature/k2003.pdf"
        assert value["deployment_name"] == "lit_source"
        assert value["size"] == 2048

    async def test_publish_swallows_blackboard_failures(self) -> None:
        vcm = await self._vcm_stand_in()
        async def _boom(*a, **kw):
            raise RuntimeError("blackboard down")
        vcm._colony_blackboard.write = MagicMock(side_effect=_boom)
        # Must not raise.
        await vcm._publish_page_event(
            "evicted", MagicMock(page_id="p", deployment_name="d"),
        )


# ---------------------------------------------------------------------------
# 4. BranchScopedCapabilityBase.fire_post_commit writes via the protocol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFirePostCommitPublishes:
    def _capability_stand_in(self, branch: str = "main"):
        from polymathera.colony.design_monorepo.capabilities import (
            BranchScopedCapabilityBase,
        )

        class _TestCapability(BranchScopedCapabilityBase):
            @property
            def current_branch(self):  # type: ignore[override]
                return branch

        cap = object.__new__(_TestCapability)
        cap._app_name = "test_app"
        return cap

    async def test_fire_post_commit_writes_typed_record(self) -> None:
        from polymathera.colony.design_monorepo.capabilities import (
            BranchScopedCapabilityBase,
        )

        cap = self._capability_stand_in(branch="main")

        # Mock the colony blackboard so we don't hit Ray / Redis.
        bb_mock = MagicMock()
        async def _async_write(*a, **kw):
            return None
        bb_mock.write = MagicMock(side_effect=_async_write)

        async def _get_bb(self):
            return bb_mock
        BranchScopedCapabilityBase._get_colony_blackboard = _get_bb
        try:
            await cap.fire_post_commit(
                sha="abc123def456",
                message="G-1 belief_registry: record kominis",
                paths=["design/beliefs/main/beliefs.json"],
            )
        finally:
            # Restore (this test isolates by monkey-patching the
            # bound method; subsequent tests get the real one back
            # via the per-test setup above).
            del BranchScopedCapabilityBase._get_colony_blackboard

        assert bb_mock.write.call_count == 1
        call = bb_mock.write.call_args
        key = call.kwargs["key"]
        value = call.kwargs["value"]
        assert key.startswith(MonorepoCommitProtocol._PREFIX)
        parsed = MonorepoCommitProtocol.parse_event_key(key)
        assert parsed["branch"] == "main"
        assert parsed["sha"] == "abc123def456"
        assert value["sha"] == "abc123def456"
        assert value["branch"] == "main"
        assert value["message"].startswith("G-1 belief_registry")
        assert value["paths"] == ["design/beliefs/main/beliefs.json"]
        assert value["capability_fqn"].endswith("._TestCapability")

    async def test_fire_post_commit_swallows_blackboard_failures(self) -> None:
        from polymathera.colony.design_monorepo.capabilities import (
            BranchScopedCapabilityBase,
        )

        cap = self._capability_stand_in(branch="main")

        async def _get_bb(self):
            raise RuntimeError("blackboard down")
        BranchScopedCapabilityBase._get_colony_blackboard = _get_bb
        try:
            # Must not raise even when blackboard acquisition fails.
            await cap.fire_post_commit(sha="x", message="m")
        finally:
            del BranchScopedCapabilityBase._get_colony_blackboard
