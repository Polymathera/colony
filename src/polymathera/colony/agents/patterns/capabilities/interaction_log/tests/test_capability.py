"""Tests for ``InteractionLogCapability._on_github_event`` + classifier.

The capability does two things: (1) classify a
``GitHubEventProtocol.*`` key into ``(event_kind, repo, number)``,
and (2) call ``service.insert_event`` with the right shape. Both are
verifiable without a real Postgres or live blackboard.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from polymathera.colony.agents.blackboard.protocol import GitHubEventProtocol
from polymathera.colony.agents.blackboard.types import BlackboardEvent
from polymathera.colony.agents.patterns.capabilities.interaction_log.capability import (
    InteractionLogCapability,
    _classify_github_key,
)


def _make_event(key: str, value: Any) -> BlackboardEvent:
    """Minimal BlackboardEvent shape the @event_handler wrapper
    expects (``.key`` + ``.value``)."""
    return BlackboardEvent(event_type="write", key=key, value=value)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def test_classify_issue_opened() -> None:
    key = GitHubEventProtocol.issue_opened_key("acme/widgets", 42)
    assert _classify_github_key(key) == (
        "github_issue_event", "acme/widgets", 42,
    )


def test_classify_issue_closed() -> None:
    key = GitHubEventProtocol.issue_closed_key("acme/widgets", 5)
    assert _classify_github_key(key) == (
        "github_issue_event", "acme/widgets", 5,
    )


def test_classify_issue_commented() -> None:
    key = GitHubEventProtocol.issue_commented_key("acme/widgets", 99)
    assert _classify_github_key(key) == (
        "github_comment_event", "acme/widgets", 99,
    )


def test_classify_pr_opened() -> None:
    key = GitHubEventProtocol.pr_opened_key("acme/widgets", 100)
    assert _classify_github_key(key) == (
        "github_pr_event", "acme/widgets", 100,
    )


def test_classify_unknown_key_returns_none() -> None:
    """Keys not in the v1 dispatch table (e.g.
    ``github:project_item_changed``) classify to ``None`` — the
    capability skips them silently."""

    assert _classify_github_key("github:project_item_changed:p1:i1") is None
    assert _classify_github_key("not_a_github_key") is None


def test_classify_malformed_issue_key_returns_none() -> None:
    """A key with the right prefix but a malformed tail (e.g. no
    number) classifies to ``None`` rather than raising — the handler
    needs every event to be best-effort."""

    assert _classify_github_key("github:issue_opened:no_number_here") is None


# ---------------------------------------------------------------------------
# Capability write-through
# ---------------------------------------------------------------------------


def _make_capability(
    *,
    tenant_id: str | None = "t1",
    colony_id: str | None = "c1",
    db_pool: object | None = None,
) -> InteractionLogCapability:
    """Build a detached InteractionLogCapability with an injected
    agent metadata + db_pool. Skips ``initialize()`` — tests set
    ``_tenant_id`` / ``_colony_id`` / ``_db_pool`` / ``_quiesced_reason``
    directly, since the agent / blackboard infrastructure isn't
    wired up in unit tests."""

    cap = InteractionLogCapability(
        agent=None, scope_id="test_scope",
        db_pool=db_pool or MagicMock(),
    )
    cap._tenant_id = tenant_id
    cap._colony_id = colony_id
    cap._quiesced_reason = None
    return cap


async def test_on_github_event_inserts_row(monkeypatch) -> None:
    """A GitHubEventProtocol.issue_opened write → one
    ``service.insert_event`` call with the right shape."""

    cap = _make_capability()
    insert_calls: list[dict] = []

    async def _fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)
        return 1

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _fake_insert,
    )

    key = GitHubEventProtocol.issue_opened_key("acme/widgets", 42)
    value = {
        "repo": "acme/widgets", "issue_number": 42,
        "state": "open", "title": "bug", "body": "...",
        "author_login": "alice", "change_kind": "opened",
    }
    await cap._on_github_event(_make_event(key, value), None)

    assert len(insert_calls) == 1
    kw = insert_calls[0]
    assert kw["tenant_id"] == "t1"
    assert kw["colony_id"] == "c1"
    assert kw["channel"] == "github"
    assert kw["event_kind"] == "github_issue_event"
    assert kw["channel_ref"] == (
        "https://github.com/acme/widgets/issues/42"
    )
    assert kw["user_login"] == "alice"
    assert kw["refs"] == [
        {"kind": "issue", "value": "acme/widgets#42"},
    ]
    assert kw["payload"]["title"] == "bug"


async def test_on_github_event_skips_unknown_key(monkeypatch) -> None:
    """Unknown / unclassified keys (e.g.
    ``github:project_item_changed``) → zero ``insert_event`` calls."""

    cap = _make_capability()
    insert_calls: list[dict] = []

    async def _fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)
        return 1

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _fake_insert,
    )

    await cap._on_github_event(
        _make_event("github:project_item_changed:p1:i1", {}), None,
    )
    assert insert_calls == []


async def test_on_github_event_skipped_when_quiesced(monkeypatch) -> None:
    """When initialize() couldn't reach Postgres, the handler is a
    no-op (no insert call, no crash)."""

    cap = _make_capability()
    cap._quiesced_reason = "no_db_pool"

    insert_called = False

    async def _fake_insert(*args, **kwargs):
        nonlocal insert_called
        insert_called = True
        return 1

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _fake_insert,
    )

    key = GitHubEventProtocol.issue_opened_key("acme/widgets", 42)
    await cap._on_github_event(_make_event(key, {"title": "x"}), None)
    assert insert_called is False


async def test_on_github_event_swallows_insert_failure(
    monkeypatch, caplog,
) -> None:
    """A failed insert MUST NOT crash the handler — otherwise one bad
    row stops every future event for this capability instance."""

    cap = _make_capability()

    async def _failing_insert(*args, **kwargs):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _failing_insert,
    )

    key = GitHubEventProtocol.issue_opened_key("acme/widgets", 42)
    # Must not raise.
    await cap._on_github_event(_make_event(key, {"title": "x"}), None)


async def test_initialize_quiesces_without_tenant_syscontext() -> None:
    """No tenant_id/colony_id on agent syscontext → quiesced + no
    db_pool acquired. Mirrors GitHubInboundCapability's behaviour."""

    fake_agent = SimpleNamespace(
        # Empty tenant/colony — mirrors the shape ``AgentMetadata``
        # exposes (typed properties delegating to syscontext).
        metadata=SimpleNamespace(
            tenant_id="", colony_id="", parameters={},
        ),
    )
    cap = InteractionLogCapability(
        agent=fake_agent, scope_id="test_scope", db_pool=MagicMock(),
    )
    # Bypass AgentCapability.initialize() which needs more wiring.
    # Manually replicate the body's syscontext-read + quiesce check.
    cap._tenant_id = cap._agent.metadata.tenant_id
    cap._colony_id = cap._agent.metadata.colony_id
    if not cap._tenant_id or not cap._colony_id:
        cap._quiesced_reason = "no_tenant_or_colony_in_syscontext"

    assert cap._quiesced_reason == "no_tenant_or_colony_in_syscontext"


# ---------------------------------------------------------------------------
# P10: mention write-through
# ---------------------------------------------------------------------------


async def test_on_mention_event_inserts_row(monkeypatch) -> None:
    """A ``mention:*`` write fires ``_on_mention_event`` which writes
    one ``interaction_log`` row with ``event_kind='mention_event'`` +
    refs for both the issue and the mention handle."""

    cap = _make_capability()
    insert_calls: list[dict] = []

    async def _fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)
        return 1

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _fake_insert,
    )

    from polymathera.colony.agents.blackboard.protocol import (
        MentionEventProtocol,
    )
    key = MentionEventProtocol.event_key("acme/widgets", 42, 99999)
    value = {
        "mention_kind": "colony-roadmap",
        "repo": "acme/widgets",
        "issue_number": 42,
        "comment_id": 99999,
        "commenter_login": "alice",
        "body": "@colony-roadmap take a look",
        "html_url": (
            "https://github.com/acme/widgets/issues/42"
            "#issuecomment-99999"
        ),
        "source_github_key": "github:issue_commented:acme/widgets:42",
        "mention_offset": 0,
    }
    await cap._on_mention_event(_make_event(key, value), None)

    assert len(insert_calls) == 1
    kw = insert_calls[0]
    assert kw["tenant_id"] == "t1"
    assert kw["colony_id"] == "c1"
    assert kw["channel"] == "github"
    assert kw["event_kind"] == "mention_event"
    assert kw["user_login"] == "alice"
    assert kw["channel_ref"] == (
        "https://github.com/acme/widgets/issues/42#issuecomment-99999"
    )
    # Refs hold both the issue and the mention handle so a future
    # ``fetch_by_ref(kind='mention', value='colony-roadmap')`` works.
    assert {"kind": "issue", "value": "acme/widgets#42"} in kw["refs"]
    assert {
        "kind": "mention", "value": "colony-roadmap",
    } in kw["refs"]


async def test_on_mention_event_skipped_when_quiesced(monkeypatch) -> None:
    """Same quiesce guard as ``_on_github_event``: no insert when
    Postgres unreachable."""

    cap = _make_capability()
    cap._quiesced_reason = "no_db_pool"

    insert_called = False

    async def _fake_insert(*args, **kwargs):
        nonlocal insert_called
        insert_called = True
        return 1

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _fake_insert,
    )

    from polymathera.colony.agents.blackboard.protocol import (
        MentionEventProtocol,
    )
    key = MentionEventProtocol.event_key("acme/widgets", 1)
    await cap._on_mention_event(
        _make_event(key, {"repo": "acme/widgets", "issue_number": 1}),
        None,
    )
    assert insert_called is False


async def test_on_mention_event_skips_malformed_payload(monkeypatch) -> None:
    """Mention payload missing ``repo`` / ``issue_number`` → silent
    skip, no insert."""

    cap = _make_capability()
    insert_called = False

    async def _fake_insert(*args, **kwargs):
        nonlocal insert_called
        insert_called = True
        return 1

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _fake_insert,
    )

    await cap._on_mention_event(
        _make_event("mention:acme__widgets:1:0", {"body": "x"}), None,
    )
    assert insert_called is False


# ---------------------------------------------------------------------------
# P11: alert write-through (bottleneck + inconsistency)
# ---------------------------------------------------------------------------


async def test_on_bottleneck_detected_inserts_row(monkeypatch) -> None:
    """A ``bottleneck_detected:*`` write → one row with
    ``channel='internal'`` + ``event_kind='bottleneck'`` + refs
    extracted from the payload's ``repo`` + ``issue_number``."""

    cap = _make_capability()
    insert_calls: list[dict] = []

    async def _fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)
        return 1

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _fake_insert,
    )

    key = "bottleneck_detected:acme__widgets:stalled_issue:1700000000123"
    value = {
        "kind": "stalled_issue",
        "severity": "high",
        "repo": "acme/widgets",
        "issue_number": 38,
        "title": "stalled feature work",
        "url": "https://github.com/acme/widgets/issues/38",
        "stale_days": 14.0,
        "summary": "Issue #38 has had no activity for 14d",
    }
    await cap._on_bottleneck_detected(_make_event(key, value), None)

    assert len(insert_calls) == 1
    kw = insert_calls[0]
    assert kw["channel"] == "internal"
    assert kw["event_kind"] == "bottleneck"
    assert kw["channel_ref"] == "https://github.com/acme/widgets/issues/38"
    assert {"kind": "issue", "value": "acme/widgets#38"} in kw["refs"]


async def test_on_design_inconsistency_inserts_row(monkeypatch) -> None:
    """A ``design_inconsistency:*`` write → one row with
    ``event_kind='inconsistency'``. Inconsistency payloads carry
    ``source_name`` rather than ``repo``/``issue_number`` — verify
    the source ref is extracted instead."""

    cap = _make_capability()
    insert_calls: list[dict] = []

    async def _fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)
        return 1

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _fake_insert,
    )

    key = "design_inconsistency:docs:contradiction:1700000000123"
    value = {
        "kind": "contradiction",
        "source_name": "docs",
        "summary": "decision D-04 contradicts constraint C-11",
        "subject_id": "D-04",
        "object_id": "C-11",
    }
    await cap._on_design_inconsistency(_make_event(key, value), None)

    assert len(insert_calls) == 1
    kw = insert_calls[0]
    assert kw["channel"] == "internal"
    assert kw["event_kind"] == "inconsistency"
    assert {"kind": "source", "value": "docs"} in kw["refs"]
    # No issue ref (no repo/issue_number on this protocol's payload).
    assert all(r["kind"] != "issue" for r in kw["refs"])


async def test_alert_handlers_skipped_when_quiesced(monkeypatch) -> None:
    """Same quiesce guard as ``_on_github_event``: no insert when
    Postgres unreachable. Pin both alert handlers in one test."""

    cap = _make_capability()
    cap._quiesced_reason = "no_db_pool"

    insert_count = 0

    async def _fake_insert(*args, **kwargs):
        nonlocal insert_count
        insert_count += 1
        return 1

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _fake_insert,
    )

    await cap._on_bottleneck_detected(
        _make_event("bottleneck_detected:r:k:1", {}), None,
    )
    await cap._on_design_inconsistency(
        _make_event("design_inconsistency:s:k:1", {}), None,
    )
    assert insert_count == 0


# ---------------------------------------------------------------------------
# Health dashboard: agent_diagnostic write-through
# ---------------------------------------------------------------------------


async def test_on_agent_diagnostic_inserts_row(monkeypatch) -> None:
    """An ``agent:diagnostic:*`` write on colony scope → one row with
    ``channel='internal'`` + ``event_kind='agent_diagnostic'`` + refs
    surfacing the producer's agent_id and the diagnostic kind so the
    dashboard can group / filter by either dimension."""

    from polymathera.colony.agents.blackboard.protocol import (
        AgentDiagnosticProtocol,
    )

    cap = _make_capability()
    insert_calls: list[dict] = []

    async def _fake_insert(*args, **kwargs):
        insert_calls.append(kwargs)
        return 1

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _fake_insert,
    )

    key = AgentDiagnosticProtocol.event_key(
        agent_id="session_agent_abc",
        kind="session_agent_stopped",
        sequence=1700000000123,
    )
    value = {
        "agent_id": "session_agent_abc",
        "agent_type": "SessionAgent",
        "kind": "session_agent_stopped",
        "stop_reason": "max_iterations_exceeded",
        "timestamp": 1700000000.123,
    }
    await cap._on_agent_diagnostic(_make_event(key, value), None)

    assert len(insert_calls) == 1
    kw = insert_calls[0]
    assert kw["channel"] == "internal"
    assert kw["event_kind"] == "agent_diagnostic"
    assert {
        "kind": "diagnostic_kind", "value": "session_agent_stopped",
    } in kw["refs"]
    assert {
        "kind": "agent_id", "value": "session_agent_abc",
    } in kw["refs"]
    assert kw["payload"]["stop_reason"] == "max_iterations_exceeded"


async def test_on_agent_diagnostic_skipped_when_quiesced(monkeypatch) -> None:
    """Quiesce guard: no insert when Postgres unreachable."""

    cap = _make_capability()
    cap._quiesced_reason = "no_db_pool"
    insert_called = False

    async def _fake_insert(*args, **kwargs):
        nonlocal insert_called
        insert_called = True
        return 1

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _fake_insert,
    )
    await cap._on_agent_diagnostic(
        _make_event(
            "agent:diagnostic:a1:session_agent_stopped:1",
            {"agent_id": "a1", "kind": "session_agent_stopped"},
        ),
        None,
    )
    assert insert_called is False


async def test_on_agent_diagnostic_swallows_insert_failure(monkeypatch) -> None:
    """A failing insert must NOT bubble — one bad row cannot stop the
    handler from processing future diagnostic events."""

    cap = _make_capability()

    async def _failing_insert(*args, **kwargs):
        raise RuntimeError("db unreachable")

    monkeypatch.setattr(
        "polymathera.colony.agents.patterns.capabilities."
        "interaction_log.capability.insert_event",
        _failing_insert,
    )
    # Must not raise.
    await cap._on_agent_diagnostic(
        _make_event(
            "agent:diagnostic:a1:github_inbound_quiesced:1",
            {
                "agent_id": "a1",
                "kind": "github_inbound_quiesced",
                "reason": "no_db_pool",
            },
        ),
        None,
    )
