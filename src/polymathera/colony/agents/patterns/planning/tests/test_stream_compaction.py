"""Tests for consciousness-stream compaction + spillover.

Covered:

1. ``BlackboardStreamLogStore`` round-trips entries + index + spans.
2. ``ConsciousnessStream`` seq-stamping + non-dropping append (compaction
   mode) vs. legacy rolling window (backward compat).
3. ``maintain`` (auto budget safety-net) compacts the oldest span,
   keeps recent raw, leaves originals losslessly in the log, and bounds
   the rendered view.
4. ``expand_span`` reverses compaction (originals back in the view).
5. ``rehydrate`` restores the view + seq counter from the durable log
   (suspend/resume + restart).
6. ``compact_now`` (agent-driven) compacts independently of budget.
7. Formatters render ``compaction_summary`` entries (interleaving +
   kind-filtering formatters both keep the condensed span visible).
8. ``VcmSpillArchive.reattach`` maps + page-faults a span (fake VCM);
   failures degrade to ``[]``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.blackboard import EnhancedBlackboard
from polymathera.colony.agents.patterns.planning.compaction import (
    ExtractiveStreamCompactor,
    KeepRecentCompactionPolicy,
    NoopSpillArchive,
    VcmSpillArchive,
    default_token_estimator,
)
from polymathera.colony.agents.patterns.planning.stream_log import (
    BlackboardStreamLogStore,
    CompactionDescriptor,
    ConsciousnessLogProtocol,
    StreamLogIndex,
)
from polymathera.colony.agents.patterns.planning.streams import (
    ConsciousnessStream,
    EventLogFormatter,
    JSONStreamFormatter,
    ToolResultFormatter,
    render_compaction_summary,
)

# asyncio_mode=auto (pyproject) runs ``async def test_*`` automatically.


async def _store() -> BlackboardStreamLogStore:
    bb = EnhancedBlackboard(
        app_name="t", scope_id="agent:x:cstream:s",
        backend_type="memory", enable_events=False,
    )
    await bb.initialize()
    return BlackboardStreamLogStore(bb)


def _stream(**kw: Any) -> ConsciousnessStream:
    kw.setdefault("formatter", JSONStreamFormatter(section_title="## S"))
    kw.setdefault("event_filter", lambda _c: True)
    return ConsciousnessStream(name="s", **kw)


async def _bind(stream: ConsciousnessStream, store: BlackboardStreamLogStore,
                *, keep_recent: int = 3) -> None:
    await stream.bind_log(
        store=store,
        compactor=ExtractiveStreamCompactor(),
        archive=NoopSpillArchive(),
        policy=KeepRecentCompactionPolicy(keep_recent=keep_recent),
        estimator=default_token_estimator(),
    )


def _feed(stream: ConsciousnessStream, n: int, pad: str = "padding text " * 3) -> None:
    for i in range(n):
        stream.consider_event({"evt": {"i": i, "pad": pad}})


# ---------------------------------------------------------------------------
# 1. Store round-trip
# ---------------------------------------------------------------------------


async def test_store_roundtrips_entries_index_and_spans() -> None:
    store = await _store()
    for seq in range(5):
        await store.append(seq, {"kind": "event", "seq": seq, "contexts": {"i": seq}})
    span = await store.read_span(1, 3)
    assert [e["seq"] for e in span] == [1, 2, 3]
    # Missing seqs are skipped, not raised.
    assert await store.read_span(10, 12) == []

    idx = StreamLogIndex(
        next_seq=5,
        compactions=[CompactionDescriptor(0, 2, "sum", {"event": 3}, 3)],
    )
    await store.write_index(idx)
    got = await store.read_index()
    assert got.next_seq == 5
    assert got.compactions[0].summary == "sum"
    assert got.covered_ranges() == [(0, 2)]


def test_log_protocol_keys_roundtrip() -> None:
    assert ConsciousnessLogProtocol.parse_entry_key(
        ConsciousnessLogProtocol.entry_key(42)
    ) == 42
    # Zero-padded ⇒ lexical order matches numeric order.
    assert ConsciousnessLogProtocol.entry_key(2) < ConsciousnessLogProtocol.entry_key(10)


# ---------------------------------------------------------------------------
# 2. Append semantics: seq + non-drop vs legacy rolling window
# ---------------------------------------------------------------------------


async def test_legacy_rolling_window_unchanged_when_compaction_disabled() -> None:
    s = _stream(max_entries=3)  # no compaction_budget_tokens ⇒ legacy
    assert s._compaction_enabled is False
    _feed(s, 10)
    assert len(s._entries) == 3                      # dropped oldest
    assert "seq" not in s._entries[0]                # no seq stamping in legacy mode


async def test_compaction_mode_stamps_seq_and_never_drops_in_memory() -> None:
    store = await _store()
    s = _stream(compaction_budget_tokens=10_000, compaction_keep_recent=3)
    await _bind(s, store)
    _feed(s, 10)
    assert [e["seq"] for e in s._entries] == list(range(10))   # monotonic, nothing dropped


# ---------------------------------------------------------------------------
# 3. Auto maintain: compact oldest, keep recent, lossless, bounded
# ---------------------------------------------------------------------------


async def test_maintain_compacts_oldest_keeps_recent_lossless() -> None:
    store = await _store()
    s = _stream(compaction_budget_tokens=40, compaction_keep_recent=3)
    await _bind(s, store, keep_recent=3)
    _feed(s, 10)
    await s.flush()
    await s.maintain()

    kinds = [e["kind"] for e in sorted(s._entries, key=s._view_sort_key)]
    assert kinds[0] == "compaction_summary"          # summary sorts first (covers oldest)
    assert kinds.count("event") == 3                 # keep_recent raw kept
    assert len(s._compactions) == 1
    desc = s._compactions[0]
    assert (desc.start_seq, desc.end_seq) == (0, 6)
    # Lossless: originals still in the durable log.
    originals = await store.read_span(0, 6)
    assert [e["seq"] for e in originals] == list(range(7))


async def test_maintain_noop_when_under_budget() -> None:
    store = await _store()
    s = _stream(compaction_budget_tokens=100_000, compaction_keep_recent=3)
    await _bind(s, store)
    _feed(s, 5)
    await s.flush()
    await s.maintain()
    assert s._compactions == []                      # never exceeded budget


# ---------------------------------------------------------------------------
# 4. expand reverses compaction
# ---------------------------------------------------------------------------


async def test_expand_span_restores_originals() -> None:
    store = await _store()
    s = _stream(compaction_budget_tokens=40, compaction_keep_recent=3)
    await _bind(s, store)
    _feed(s, 10)
    await s.flush()
    await s.maintain()
    desc = s._compactions[0]

    res = await s.expand_span(desc.start_seq, desc.end_seq)
    assert res["expanded"] == 7
    assert s._compactions == []
    raw_seqs = sorted(e["seq"] for e in s._entries if e["kind"] != "compaction_summary")
    assert raw_seqs == list(range(10))               # full history back in view


# ---------------------------------------------------------------------------
# 5. rehydrate (suspend/resume + restart)
# ---------------------------------------------------------------------------


async def test_rehydrate_restores_view_and_seq() -> None:
    store = await _store()
    s = _stream(compaction_budget_tokens=40, compaction_keep_recent=3)
    await _bind(s, store)
    _feed(s, 10)
    await s.flush()
    await s.maintain()

    # Fresh stream over the SAME durable store = "restart".
    s2 = _stream(compaction_budget_tokens=40, compaction_keep_recent=3)
    await _bind(s2, store)
    await s2.rehydrate()
    assert s2._next_seq == 10
    assert len(s2._compactions) == 1
    assert any(e["kind"] == "compaction_summary" for e in s2._entries)
    # New entries continue the seq from where we left off.
    s2.consider_event({"evt": {"i": 99}})
    assert s2._entries[-1]["seq"] == 10


# ---------------------------------------------------------------------------
# 6. agent-driven compact_now (budget-independent)
# ---------------------------------------------------------------------------


async def test_compact_now_ignores_budget() -> None:
    store = await _store()
    s = _stream(compaction_budget_tokens=10_000, compaction_keep_recent=2)
    await _bind(s, store, keep_recent=2)
    _feed(s, 6)
    await s.flush()
    desc = await s.compact_now()
    assert desc is not None
    assert desc["produced_by"] == "agent"
    assert len(s._compactions) == 1
    assert sum(1 for e in s._entries if e["kind"] == "event") == 2   # keep_recent=2


async def test_compact_now_noop_when_nothing_eligible() -> None:
    store = await _store()
    s = _stream(compaction_budget_tokens=10_000, compaction_keep_recent=5)
    await _bind(s, store, keep_recent=5)
    _feed(s, 3)
    await s.flush()
    assert await s.compact_now() is None


# ---------------------------------------------------------------------------
# 7. Formatters render compaction_summary
# ---------------------------------------------------------------------------


def _summary_entry(start: int, end: int, n: int) -> dict[str, Any]:
    return {
        "kind": "compaction_summary",
        "timestamp": 0.0,
        "payload": {"covers": [start, end], "summary": "older stuff", "entry_count": n},
    }


def test_helper_renders_summary_with_span_and_count() -> None:
    line = render_compaction_summary(_summary_entry(0, 6, 7))
    assert "seq 0–6" in line
    assert "7 entries" in line
    assert "expandable" in line


def test_generic_formatter_renders_summary_inline_in_order() -> None:
    fmt = EventLogFormatter(section_title="## S")
    entries = [
        _summary_entry(0, 4, 5),
        {"kind": "event", "seq": 5, "contexts": {"i": 5}},
    ]
    out = fmt.format(entries)
    assert "condensed history" in out
    # Summary appears before the raw event line.
    assert out.index("condensed history") < out.index('"i": 5')


def test_kind_filtering_formatter_keeps_summary_visible() -> None:
    # ToolResultFormatter normally renders ONLY tool_output entries; a
    # compacted span (mixed kinds) must still surface as a summary line.
    fmt = ToolResultFormatter(section_title="## Tools")
    out = fmt.format([_summary_entry(0, 9, 10)])
    assert "condensed history" in out
    assert out.strip() != ""


# ---------------------------------------------------------------------------
# 8. VcmSpillArchive (fake VCM)
# ---------------------------------------------------------------------------


async def test_vcm_spill_archive_reattach_maps_and_pagefaults() -> None:
    from polymathera.colony.distributed.ray_utils.serving import (
        Ring,
        execution_context,
    )

    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ):
        # Fake agent: real in-memory blackboard for the archive scope,
        # fake VCM handle recording the calls.
        archive_bb = EnhancedBlackboard(
            app_name="t", scope_id="archive", backend_type="memory",
            enable_events=False,
        )
        await archive_bb.initialize()

        loaded: list[str] = []

        vcm = MagicMock()

        async def _mmap(**_kw):
            return None

        async def _pages(_scope, **_kw):
            return [{"page_id": "p1"}, {"page_id": "p2"}]

        async def _load(*, page_id, agent_id, priority):
            loaded.append(page_id)
            return True

        vcm.mmap_application_scope = MagicMock(side_effect=_mmap)
        vcm.get_pages_for_scope = MagicMock(side_effect=_pages)
        vcm.request_page_load = MagicMock(side_effect=_load)

        agent = MagicMock()
        agent.agent_id = "agent_x"
        agent.syscontext = __import__(
            "polymathera.colony.distributed.ray_utils.serving",
            fromlist=["require_execution_context"],
        ).require_execution_context()

        async def _get_bb(scope_id, **_kw):
            return archive_bb
        agent.get_blackboard = MagicMock(side_effect=_get_bb)

        # reattach resolves the VCM via the public _handles.get_vcm.
        import polymathera.colony._handles as handles
        orig = handles.get_vcm

        async def _get_vcm(app_name=None):
            return vcm
        handles.get_vcm = _get_vcm  # type: ignore[assignment]
        try:
            archive = VcmSpillArchive(agent)
            pages = await archive.reattach(
                stream_name="s", start_seq=0, end_seq=2,
                entries=[{"kind": "event", "seq": 0, "contexts": {}}],
            )
        finally:
            handles.get_vcm = orig  # type: ignore[assignment]
        assert pages == ["p1", "p2"]
        assert loaded == ["p1", "p2"]


async def test_vcm_spill_archive_swallows_failures() -> None:
    agent = MagicMock()
    agent.agent_id = "a"

    async def _boom(app_name=None):
        raise RuntimeError("no vcm")

    # get_vcm() explodes (VCM not deployed) ⇒ reattach degrades to [].
    import polymathera.colony._handles as handles
    orig = handles.get_vcm
    handles.get_vcm = _boom  # type: ignore[assignment]
    try:
        archive = VcmSpillArchive(agent)
        assert await archive.reattach(
            stream_name="s", start_seq=0, end_seq=1, entries=[],
        ) == []
    finally:
        handles.get_vcm = orig  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 9. StreamMaintenanceCapability — planner-facing actions
# ---------------------------------------------------------------------------


async def _agent_with_compacted_stream() -> Any:
    from polymathera.colony.agents.patterns.actions.policies import (
        BaseActionPolicy,
    )
    store = await _store()
    s = _stream(compaction_budget_tokens=40, compaction_keep_recent=3)
    await _bind(s, store)
    _feed(s, 10)
    await s.flush()
    await s.maintain()
    agent = MagicMock()
    agent.agent_id = "agent_cap"
    policy = BaseActionPolicy(agent=agent, consciousness_streams=[s])
    agent.action_policy = policy
    return agent, s


async def test_capability_list_history_reports_spans() -> None:
    from polymathera.colony.agents.patterns.capabilities.stream_maintenance import (
        StreamMaintenanceCapability,
    )
    agent, _ = await _agent_with_compacted_stream()
    cap = StreamMaintenanceCapability(agent=agent, scope_id="x")
    out = await cap.list_stream_history(stream="s")
    assert out["status"] == "ok"
    assert out["condensed_spans"]
    assert out["condensed_spans"][0]["start_seq"] == 0


async def test_capability_expand_then_compact_roundtrip() -> None:
    from polymathera.colony.agents.patterns.capabilities.stream_maintenance import (
        StreamMaintenanceCapability,
    )
    agent, s = await _agent_with_compacted_stream()
    cap = StreamMaintenanceCapability(agent=agent, scope_id="x")
    span = s.history_summary()[0]

    exp = await cap.expand_stream_span(
        stream="s", start_seq=span["start_seq"], end_seq=span["end_seq"],
    )
    assert exp["status"] == "expanded"
    assert exp["expanded"] == 7
    assert s.history_summary() == []                 # span no longer condensed

    comp = await cap.compact_stream(stream="s")       # re-condense oldest
    assert comp["status"] == "compacted"
    assert s.history_summary()                        # condensed again


async def test_capability_errors_on_unknown_stream() -> None:
    from polymathera.colony.agents.patterns.capabilities.stream_maintenance import (
        StreamMaintenanceCapability,
    )
    agent, _ = await _agent_with_compacted_stream()
    cap = StreamMaintenanceCapability(agent=agent, scope_id="x")
    out = await cap.list_stream_history(stream="does_not_exist")
    assert out["status"] == "error"
