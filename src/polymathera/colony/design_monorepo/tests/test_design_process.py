"""Tests for :class:`DesignProcessCapability` — Phase P5a surface
(``load_design_context`` + ``summarise_progress`` +
``identify_bottlenecks``) plus the two new blackboard protocols
(``BottleneckDetectedProtocol`` + ``RoadmapSyncProtocol``).

Same self-contained-git-repo + mocked-deps pattern the
``test_system_design.py`` file uses, so cross-test-tree fixture pulls
stay minimal.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import git
import pytest

from polymathera.colony.design_monorepo.process import (
    DesignProcessCapability,
)


pytestmark = pytest.mark.asyncio


def _make_repo(root: Path) -> None:
    repo = git.Repo.init(root, initial_branch="main")
    repo.config_writer().set_value("user", "email", "t@t").release()
    repo.config_writer().set_value("user", "name", "t").release()
    sentinel = root / ".init"
    sentinel.write_text("init\n", encoding="utf-8")
    repo.index.add([str(sentinel)])
    repo.index.commit("init")


def _make_capability(
    tmp_path: Path, *, github: Any = None,
) -> DesignProcessCapability:
    _make_repo(tmp_path)
    agent = MagicMock() if github is not None else None
    if agent is not None:
        # Simulate the agent's capability registry returning our
        # fake GitHubCapability when DesignProcessCapability calls
        # ``self._agent.get_capability(GitHubCapability)``.
        agent._capabilities = {"github": github}
        agent.get_capability = lambda cls: (
            github
            if any(isinstance(github, cls) for _ in [None])
            else None
        )
    cap = DesignProcessCapability(
        agent=agent, scope_id="test", working_dir=tmp_path,
    )
    return cap


@pytest.fixture
def _reset_knowledge_deps_after():
    yield
    from polymathera.colony.knowledge.deps import reset_knowledge_deps

    reset_knowledge_deps()


def _make_github_stub(
    *,
    milestones: list[dict] | None = None,
    list_milestones_ok: bool = True,
    issues: list[dict] | None = None,
    list_issues_ok: bool = True,
    default_repo: str = "acme/proj",
) -> Any:
    """A minimal duck-typed GitHubCapability for use as a sibling
    capability. Implements ``list_milestones`` + ``list_issues`` +
    ``_default_repo`` — enough surface for ``summarise_progress``
    and ``identify_bottlenecks`` to call into.

    Marked as a ``GitHubCapability`` instance via isinstance check
    sleight-of-hand (we make the fake's class identity match by
    patching the registry's class probe in test setups that need it).
    """

    from polymathera.colony.agents.patterns.capabilities.github import (
        GitHubCapability,
    )

    fake = MagicMock(spec=GitHubCapability)
    fake._default_repo = default_repo
    if list_milestones_ok:
        fake.list_milestones = AsyncMock(return_value={
            "ok": True, "message": "",
            "milestones": milestones or [],
            "count": len(milestones or []),
        })
    else:
        fake.list_milestones = AsyncMock(return_value={
            "ok": False, "message": "API error",
            "milestones": [],
        })
    if list_issues_ok:
        fake.list_issues = AsyncMock(return_value={
            "ok": True, "message": "",
            "issues": issues or [],
            "count": len(issues or []),
        })
    else:
        fake.list_issues = AsyncMock(return_value={
            "ok": False, "message": "API error",
            "issues": [],
        })
    return fake


# ---------------------------------------------------------------------------
# load_design_context — delegate parity with materialize_design_context
# ---------------------------------------------------------------------------


async def test_load_design_context_delegates_to_shared_impl(
    tmp_path: Path, monkeypatch,
) -> None:
    """``DesignProcessCapability.load_design_context`` and
    ``RepoStateProvider.materialize_design_context`` MUST call the
    same shared body and produce the same response shape. Asserted
    by patching the underlying materialiser and verifying the
    response field set."""

    cap = _make_capability(tmp_path)
    (tmp_path / ".colony").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "objectives.md").write_text(
        "# Objectives\n", encoding="utf-8",
    )
    (tmp_path / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n  - { name: default, type: git_repo }\n"
        "design_context_sources:\n"
        "  - name: docs\n    paths: ['docs/**/*.md']\n",
        encoding="utf-8",
    )

    fake_vcm = MagicMock()
    fake_vcm.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )
    fake_vcm.get_pages_for_scope = AsyncMock(return_value=[])
    fake_vcm.lock_page = AsyncMock()
    async def _stub_get_vcm():
        return fake_vcm
    from polymathera.colony import _handles as handles_mod
    monkeypatch.setattr(handles_mod, "get_vcm", _stub_get_vcm)

    fake_ingestor = MagicMock()
    from polymathera.colony.knowledge.models import (
        IngestionRecord, IngestionStatus, KnowledgeFormat,
    )
    async def _stub_ingest(path, *, tier, source_uri, **_):
        return IngestionRecord(
            source_uri=source_uri,
            detected_format=KnowledgeFormat.MARKDOWN,
            tier=tier, status=IngestionStatus.COMPLETED,
            chunks_produced=1, claims_extracted=1, document_hash="sha",
        )
    fake_ingestor.ingest_file = AsyncMock(side_effect=_stub_ingest)
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "get_default_ingestor", lambda: fake_ingestor,
    )

    fake_bb = MagicMock()
    fake_bb.write = AsyncMock()
    cap._colony_blackboard = fake_bb

    result = await cap.load_design_context(refresh=False)

    # Same shape as materialize_design_context.
    assert set(result.keys()) >= {
        "mapped", "pinned", "ingested", "total_claims",
        "failed", "count", "rows",
    }
    assert result["mapped"] == ["docs"]
    assert result["ingested"] == ["docs"]
    assert result["total_claims"] == 1
    # Two events fire (vcm + kuzu rows) — same as the RepoStateProvider
    # path.
    assert fake_bb.write.await_count == 2


async def test_load_design_context_renewer_cancelled_on_stop(
    tmp_path: Path, monkeypatch,
) -> None:
    """The base-class lifted ``stop()`` cancels the renewer for
    DesignProcessCapability just as it does for RepoStateProvider."""

    cap = _make_capability(tmp_path)
    (tmp_path / ".colony").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("# x\n", encoding="utf-8")
    (tmp_path / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n  - { name: default, type: git_repo }\n"
        "design_context_sources:\n"
        "  - name: docs\n    paths: ['docs/**/*.md']\n"
        "    pin_in_vcm: true\n",
        encoding="utf-8",
    )

    fake_vcm = MagicMock()
    fake_vcm.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )
    fake_vcm.get_pages_for_scope = AsyncMock(return_value=[
        {"page_id": "p1", "size": 100, "group_id": "g"},
    ])
    fake_vcm.lock_page = AsyncMock()
    fake_vcm.extend_page_lock = AsyncMock(return_value=True)
    async def _stub_get_vcm():
        return fake_vcm
    from polymathera.colony import _handles as handles_mod
    monkeypatch.setattr(handles_mod, "get_vcm", _stub_get_vcm)

    fake_bb = MagicMock()
    fake_bb.write = AsyncMock()
    cap._colony_blackboard = fake_bb

    await cap.load_design_context(refresh=False, include_kuzu=False)
    assert cap._design_context_renewer is not None

    await cap.stop()
    assert cap._design_context_renewer is None


# ---------------------------------------------------------------------------
# summarise_progress
# ---------------------------------------------------------------------------


async def test_summarise_progress_returns_milestone_snapshot(
    tmp_path: Path,
) -> None:
    """Reports per-milestone open/closed counts + progress_pct + a
    current_milestone selection (earliest due_on among open)."""

    github = _make_github_stub(milestones=[
        {
            "number": 1, "title": "M1-soonest", "description": "",
            "state": "open", "due_on": "2026-05-01T00:00:00Z",
            "open_issues": 4, "closed_issues": 6, "html_url": "u1",
        },
        {
            "number": 2, "title": "M2-later", "description": "",
            "state": "open", "due_on": "2026-12-01T00:00:00Z",
            "open_issues": 10, "closed_issues": 0, "html_url": "u2",
        },
        {
            "number": 3, "title": "M3-empty", "description": "",
            "state": "open", "due_on": None,
            "open_issues": 0, "closed_issues": 0, "html_url": "u3",
        },
    ])
    cap = _make_capability(tmp_path, github=github)

    result = await cap.summarise_progress(repo="acme/proj")
    assert result["error"] == ""
    assert result["current_milestone"]["title"] == "M1-soonest"
    by_title = {m["title"]: m for m in result["milestones"]}
    assert by_title["M1-soonest"]["progress_pct"] == 60.0  # 6/10
    assert by_title["M2-later"]["progress_pct"] == 0.0
    assert by_title["M3-empty"]["progress_pct"] is None  # 0/0
    assert result["totals"] == {
        "open_issues": 14, "closed_issues": 6,
        "total_issues": 20, "milestone_count": 3,
    }


async def test_summarise_progress_falls_back_when_no_due_dates(
    tmp_path: Path,
) -> None:
    """If no open milestone has a due_on, current_milestone falls back
    to the first one in the list (preserves API-side ordering)."""

    github = _make_github_stub(milestones=[
        {
            "number": 5, "title": "first-in-list", "description": "",
            "state": "open", "due_on": None,
            "open_issues": 1, "closed_issues": 0, "html_url": "u",
        },
        {
            "number": 6, "title": "second", "description": "",
            "state": "open", "due_on": None,
            "open_issues": 2, "closed_issues": 0, "html_url": "u",
        },
    ])
    cap = _make_capability(tmp_path, github=github)
    result = await cap.summarise_progress()
    assert result["current_milestone"]["title"] == "first-in-list"


async def test_summarise_progress_returns_empty_when_no_github_sibling(
    tmp_path: Path,
) -> None:
    """Detached / no sibling GitHubCapability → graceful empty +
    clear error label the planner can branch on."""

    cap = _make_capability(tmp_path)  # no github sibling
    result = await cap.summarise_progress()
    assert result["error"] == "github_capability_missing"
    assert result["milestones"] == []
    assert result["totals"]["milestone_count"] == 0


async def test_summarise_progress_handles_list_milestones_failure(
    tmp_path: Path,
) -> None:
    """Sibling exists but the API call fails → clear error + empty
    payload (no spurious current_milestone)."""

    github = _make_github_stub(list_milestones_ok=False)
    cap = _make_capability(tmp_path, github=github)
    result = await cap.summarise_progress()
    assert result["error"] == "list_milestones_failed"
    assert result["current_milestone"] is None
    assert result["milestones"] == []


# ---------------------------------------------------------------------------
# identify_bottlenecks
# ---------------------------------------------------------------------------


def _iso_n_days_ago(n: int) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(days=n)
    ).isoformat().replace("+00:00", "Z")


async def test_identify_bottlenecks_flags_stalled_open_issues(
    tmp_path: Path,
) -> None:
    """Open issues with updated_at older than the threshold land in
    ``stalled_issues``; fresh ones do not."""

    github = _make_github_stub(issues=[
        {  # Fresh — not stalled
            "number": 1, "title": "fresh", "url": "u1",
            "state": "open", "updated_at": _iso_n_days_ago(2),
        },
        {  # Stalled — beyond 14d default
            "number": 2, "title": "stale", "url": "u2",
            "state": "open", "updated_at": _iso_n_days_ago(20),
        },
        {  # Very stale — high severity (> 2x threshold)
            "number": 3, "title": "ancient", "url": "u3",
            "state": "open", "updated_at": _iso_n_days_ago(60),
        },
    ])
    cap = _make_capability(tmp_path, github=github)

    result = await cap.identify_bottlenecks(
        emit_blackboard_events=False,
    )
    assert result["error"] == ""
    assert result["stats"]["stalled_count"] == 2
    by_number = {s["issue_number"]: s for s in result["stalled_issues"]}
    assert 1 not in by_number
    assert by_number[2]["severity"] == "medium"  # >threshold, <2x
    assert by_number[3]["severity"] == "high"    # >2x threshold
    assert by_number[2]["stale_days"] > 14
    assert "ping the assignee" in by_number[2]["suggested_remedies"]


async def test_identify_bottlenecks_respects_custom_threshold(
    tmp_path: Path,
) -> None:
    """``stalled_no_activity_days`` overrides the default 14."""

    github = _make_github_stub(issues=[
        {
            "number": 1, "title": "5d", "url": "u",
            "state": "open", "updated_at": _iso_n_days_ago(5),
        },
        {
            "number": 2, "title": "10d", "url": "u",
            "state": "open", "updated_at": _iso_n_days_ago(10),
        },
    ])
    cap = _make_capability(tmp_path, github=github)
    result = await cap.identify_bottlenecks(
        stalled_no_activity_days=7, emit_blackboard_events=False,
    )
    flagged = [s["issue_number"] for s in result["stalled_issues"]]
    assert flagged == [2]  # only the 10-day one trips a 7d threshold


async def test_identify_bottlenecks_emits_event_per_stalled(
    tmp_path: Path,
) -> None:
    """One BottleneckDetectedProtocol per stalled issue with the
    right key shape + tags."""

    github = _make_github_stub(issues=[
        {
            "number": 7, "title": "stalled", "url": "u",
            "state": "open", "updated_at": _iso_n_days_ago(30),
        },
    ])
    cap = _make_capability(tmp_path, github=github)
    fake_bb = MagicMock()
    fake_bb.write = AsyncMock()
    cap._colony_blackboard = fake_bb

    result = await cap.identify_bottlenecks(emit_blackboard_events=True)
    assert len(result["stalled_issues"]) == 1
    assert fake_bb.write.await_count == 1
    call = fake_bb.write.await_args
    assert call.kwargs["key"].startswith("bottleneck_detected:")
    assert ":stalled_issue:" in call.kwargs["key"]
    assert "stalled_issue" in call.kwargs["tags"]
    # Severity tag included.
    assert "high" in call.kwargs["tags"]
    assert "detected_at" in call.kwargs["value"]


async def test_identify_bottlenecks_no_github_sibling_returns_clear_error(
    tmp_path: Path,
) -> None:
    cap = _make_capability(tmp_path)  # no github sibling
    result = await cap.identify_bottlenecks()
    assert result["error"] == "github_capability_missing"
    assert result["stalled_issues"] == []
    assert result["stats"]["github_capability_available"] is False


async def test_identify_bottlenecks_discovers_rule_claims(
    tmp_path: Path, _reset_knowledge_deps_after,
) -> None:
    """KG-based discovery of operator-authored bottleneck_rule
    claims surfaces them alongside the built-in stalled-issue
    findings (both idioms: predicate-tagged AND ``X is_a 'bottleneck rule'``)."""

    from polymathera.colony.knowledge.deps import (
        reset_knowledge_deps, set_knowledge_deps,
    )
    from polymathera.colony.knowledge.models import (
        CitationSpan, Claim, KnowledgeFormat,
    )
    from polymathera.colony.knowledge.stores.graph import InMemoryGraphStore

    reset_knowledge_deps()
    store = InMemoryGraphStore()
    set_knowledge_deps(graph_store=store)

    def _claim(s, p, o, *, source_uri):
        return Claim(
            subject=s, predicate=p, object=o,
            citation=CitationSpan(
                source_uri=source_uri,
                source_format=KnowledgeFormat.MARKDOWN,
                section_path="", char_start=0, char_end=10,
            ),
        )

    await store.add_claim(_claim(
        "rule:stalled-mention", "defines_bottleneck_rule",
        "open issue with no comment in 14d is stalled",
        source_uri="design_context://rules/docs/rules.md",
    ))
    await store.add_claim(_claim(
        "rule:deep-blocking-chain", "is_a", "bottleneck rule",
        source_uri="design_context://rules/docs/rules.md",
    ))
    # Non-rule claim — must NOT show up.
    await store.add_claim(_claim(
        "PMP-9", "is_a", "magnetometer",
        source_uri="design_context://hardware/docs/h.md",
    ))

    github = _make_github_stub(issues=[])
    cap = _make_capability(tmp_path, github=github)
    result = await cap.identify_bottlenecks(
        emit_blackboard_events=False,
    )
    rule_ids = {r["rule_id"] for r in result["rules_discovered"]}
    assert rule_ids == {"rule:stalled-mention", "rule:deep-blocking-chain"}


# ---------------------------------------------------------------------------
# BottleneckDetectedProtocol + RoadmapSyncProtocol — key/parse round-trips
# ---------------------------------------------------------------------------


def test_bottleneck_detected_protocol_round_trip() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        BottleneckDetectedProtocol,
    )

    key = BottleneckDetectedProtocol.event_key(
        repo="acme/proj", kind="stalled_issue", millis=1700000000123,
    )
    assert key.startswith("bottleneck_detected:")
    # Repo / kept-safe (no '/' collision with separator).
    parsed = BottleneckDetectedProtocol.parse_event_key(key)
    assert parsed == {
        "repo": "acme/proj",
        "kind": "stalled_issue",
        "millis": "1700000000123",
    }


def test_bottleneck_detected_protocol_rejects_alien_key() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        BottleneckDetectedProtocol,
    )

    with pytest.raises(ValueError, match="Not a BottleneckDetected"):
        BottleneckDetectedProtocol.parse_event_key("design_inconsistency:x:y:1")


def test_bottleneck_detected_protocol_patterns() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        BottleneckDetectedProtocol,
    )

    assert BottleneckDetectedProtocol.event_pattern() == (
        "bottleneck_detected:*"
    )
    assert BottleneckDetectedProtocol.event_pattern_for_repo("acme/proj") == (
        "bottleneck_detected:acme__proj:*"
    )
    assert BottleneckDetectedProtocol.event_pattern_for_kind(
        "stalled_issue",
    ) == "bottleneck_detected:*:stalled_issue:*"


def test_roadmap_sync_protocol_round_trip() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        RoadmapSyncProtocol,
    )

    key = RoadmapSyncProtocol.event_key(
        repo="acme/proj", direction="bidirectional", millis=42,
    )
    assert key.startswith("roadmap_sync:")
    parsed = RoadmapSyncProtocol.parse_event_key(key)
    assert parsed == {
        "repo": "acme/proj",
        "direction": "bidirectional",
        "millis": "42",
    }


def test_roadmap_sync_protocol_rejects_alien_key() -> None:
    from polymathera.colony.agents.blackboard.protocol import (
        RoadmapSyncProtocol,
    )

    with pytest.raises(ValueError, match="Not a RoadmapSync"):
        RoadmapSyncProtocol.parse_event_key("bottleneck_detected:x:y:1")


# Need ``Any`` import for the test helpers; pulled here to keep the
# top-of-file imports tight.
from typing import Any  # noqa: E402


# ===========================================================================
# P5b: bootstrap_roadmap_from_objectives — pure helpers + action
# ===========================================================================
#
# Pure module-level helpers (no I/O) are tested first; the action
# is exercised against mocked LLM + sibling GitHubCapability so no
# live cluster / GitHub App is needed.


# --- pure helpers ---------------------------------------------------------


def test_stable_task_id_is_deterministic_and_content_based() -> None:
    """Same (milestone, task) → same id every time. Different inputs
    → different ids. 12-hex-char hash per design-doc Q4."""

    from polymathera.colony.design_monorepo.capabilities import (
        _stable_task_id,
    )

    a = _stable_task_id("Milestone 1", "Build sensor stack")
    b = _stable_task_id("Milestone 1", "Build sensor stack")
    c = _stable_task_id("Milestone 1", "Build optics")
    d = _stable_task_id("Milestone 2", "Build sensor stack")
    assert a == b
    assert a != c
    assert a != d
    assert len(a) == 12
    assert all(ch in "0123456789abcdef" for ch in a)


def test_extract_roadmap_task_marker_finds_marker_anywhere_in_body() -> None:
    """The marker can sit anywhere in the body — operators may add
    prose, the regex still extracts it."""

    from polymathera.colony.design_monorepo.capabilities import (
        _extract_roadmap_task_marker,
    )

    body_at_end = (
        "Some description.\n\n"
        "_Part of milestone: **Milestone 1**_\n\n"
        "<!-- colony:roadmap-task: abcdef012345 -->"
    )
    assert _extract_roadmap_task_marker(body_at_end) == "abcdef012345"
    body_inline = "preface <!-- colony:roadmap-task: 0a1b2c3d4e5f --> suffix"
    assert _extract_roadmap_task_marker(body_inline) == "0a1b2c3d4e5f"
    # No marker.
    assert _extract_roadmap_task_marker("just a description") is None
    # Empty / None.
    assert _extract_roadmap_task_marker("") is None
    assert _extract_roadmap_task_marker(None) is None
    # Bad shape (non-hex / wrong length) → no match.
    assert _extract_roadmap_task_marker(
        "<!-- colony:roadmap-task: nothex! -->",
    ) is None


def test_parse_roadmap_proposal_round_trip() -> None:
    """LLM happy path: valid JSON → parsed proposal with stable ids
    auto-injected."""

    from polymathera.colony.design_monorepo.capabilities import (
        _parse_roadmap_proposal, _stable_task_id,
    )

    raw = """
    {
      "milestones": [
        {
          "title": "Milestone 1",
          "description": "first cut",
          "tasks": [
            {"title": "task A", "description": "do A", "labels": ["a"]},
            {"title": "task B", "description": "do B"}
          ]
        }
      ]
    }
    """
    proposal = _parse_roadmap_proposal(raw)
    assert proposal is not None
    assert len(proposal["milestones"]) == 1
    m = proposal["milestones"][0]
    assert m["title"] == "Milestone 1"
    assert len(m["tasks"]) == 2
    assert m["tasks"][0]["title"] == "task A"
    assert m["tasks"][0]["labels"] == ["a"]
    assert m["tasks"][0]["stable_id"] == _stable_task_id(
        "Milestone 1", "task A",
    )
    assert m["tasks"][1]["stable_id"] == _stable_task_id(
        "Milestone 1", "task B",
    )


def test_parse_roadmap_proposal_strips_code_fences() -> None:
    """LLM commonly wraps JSON in ```json fences; the parser
    handles that idiom (same as LLMClaimExtractor._parse)."""

    from polymathera.colony.design_monorepo.capabilities import (
        _parse_roadmap_proposal,
    )

    raw = (
        "```json\n"
        '{"milestones": [{"title": "M", "description": "", '
        '"tasks": [{"title": "t", "description": ""}]}]}\n'
        "```"
    )
    proposal = _parse_roadmap_proposal(raw)
    assert proposal is not None
    assert proposal["milestones"][0]["title"] == "M"


def test_parse_roadmap_proposal_rejects_invalid_shapes() -> None:
    """Malformed / shape-incompatible JSON returns None (logged at
    WARN). Six malformations covered: bad JSON, top-level non-dict,
    missing milestones, milestones not a list, milestone without
    title, milestone without tasks."""

    from polymathera.colony.design_monorepo.capabilities import (
        _parse_roadmap_proposal,
    )

    assert _parse_roadmap_proposal("not json") is None
    assert _parse_roadmap_proposal('"a string"') is None
    assert _parse_roadmap_proposal("{}") is None
    assert _parse_roadmap_proposal('{"milestones": "not a list"}') is None
    assert _parse_roadmap_proposal(
        '{"milestones": [{"title": "", "tasks": [{"title": "t"}]}]}',
    ) is None
    assert _parse_roadmap_proposal(
        '{"milestones": [{"title": "M", "tasks": []}]}',
    ) is None


def test_parse_roadmap_proposal_deduplicates_task_titles_within_milestone() -> None:
    """Per the schema constraint: task titles MUST be unique within
    a milestone. The parser drops duplicates rather than blowing
    up — keeps the action robust against a sloppy LLM."""

    from polymathera.colony.design_monorepo.capabilities import (
        _parse_roadmap_proposal,
    )

    raw = """
    {
      "milestones": [
        {
          "title": "M",
          "description": "",
          "tasks": [
            {"title": "dup", "description": "first"},
            {"title": "dup", "description": "second"}
          ]
        }
      ]
    }
    """
    proposal = _parse_roadmap_proposal(raw)
    assert proposal is not None
    assert len(proposal["milestones"][0]["tasks"]) == 1
    assert proposal["milestones"][0]["tasks"][0]["description"] == "first"


def test_render_roadmap_markdown_stamps_marker_per_task() -> None:
    """Every task line in the rendered ROADMAP.md ends with the
    stable-id marker so operator + sync can both find it visually
    + programmatically."""

    from polymathera.colony.design_monorepo.capabilities import (
        _extract_roadmap_task_marker, _render_roadmap_markdown,
    )

    proposal = {
        "milestones": [
            {
                "title": "M1", "description": "first cut",
                "tasks": [
                    {
                        "title": "task A", "description": "do A",
                        "labels": [], "stable_id": "aaaaaaaaaaaa",
                    },
                ],
            },
        ],
    }
    rendered = _render_roadmap_markdown(proposal)
    assert "# Roadmap" in rendered
    assert "## M1" in rendered
    assert "first cut" in rendered
    assert "task A" in rendered
    # Marker present + extractable.
    assert _extract_roadmap_task_marker(rendered) == "aaaaaaaaaaaa"


def test_render_issue_body_for_task_includes_marker_at_end() -> None:
    """The issue body's marker MUST be at the end so the sync
    parser finds it reliably regardless of what prose the LLM put
    in the description."""

    from polymathera.colony.design_monorepo.capabilities import (
        _extract_roadmap_task_marker, _render_issue_body_for_task,
    )

    body = _render_issue_body_for_task(
        milestone={"title": "M1"},
        task={
            "title": "task A", "description": "do A",
            "stable_id": "0123456789ab",
        },
    )
    assert "do A" in body
    assert "M1" in body
    assert _extract_roadmap_task_marker(body) == "0123456789ab"
    # Marker is the last non-empty line.
    last_nonempty = [
        line for line in body.splitlines() if line.strip()
    ][-1]
    assert "colony:roadmap-task:" in last_nonempty


# --- bootstrap_roadmap_from_objectives action ------------------------------


def _seed_design_context_repo(tmp_path: Path) -> None:
    """Bring up a tmp repo with design_context_sources + matching
    markdown files, ready for the bootstrap action to read."""

    _make_repo(tmp_path)
    (tmp_path / ".colony").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "objectives.md").write_text(
        "# Objectives\nMaximise sensitivity.\n", encoding="utf-8",
    )
    (tmp_path / "docs" / "constraints.md").write_text(
        "# Constraints\nMust fit a 30cm cube.\n", encoding="utf-8",
    )
    (tmp_path / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n  - { name: default, type: git_repo }\n"
        "design_context_sources:\n"
        "  - name: docs\n    paths: ['docs/**/*.md']\n",
        encoding="utf-8",
    )


def _stub_llm_callable(response_json: str) -> Any:
    """Build a ``build_default_llm_callable``-shaped wrapper that
    returns the canned ``response_json`` without touching the live
    LLMCluster. Patched into the deps module by tests via monkeypatch."""

    def _builder(*, max_tokens: int, temperature: float):
        async def _llm(_prompt: str) -> str:
            return response_json
        return _llm
    return _builder


async def test_bootstrap_dry_run_returns_proposal_without_writing(
    tmp_path: Path, monkeypatch,
) -> None:
    """``dry_run=True`` (the default) returns the parsed proposal +
    stats; ROADMAP.md is NOT written; no issues are created."""

    _seed_design_context_repo(tmp_path)
    github = _make_github_stub(issues=[])
    cap = _make_capability(tmp_path, github=github)

    response = (
        '{"milestones": [{"title": "M1", "description": "first",'
        '"tasks": [{"title": "build", "description": "do it"}]}]}'
    )
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_llm_callable(response),
    )

    result = await cap.bootstrap_roadmap_from_objectives()
    assert result["dry_run"] is True
    assert result["error"] == ""
    assert result["proposal"]["milestones"][0]["title"] == "M1"
    assert result["stats"]["milestone_count"] == 1
    assert result["stats"]["task_count"] == 1
    # No filesystem mutation.
    assert not (tmp_path / "ROADMAP.md").exists()
    # No GitHub issue creation.
    github.create_issue.assert_not_called()  # MagicMock — was never set up


async def test_bootstrap_dry_run_returns_clear_error_without_design_context_sources(
    tmp_path: Path, monkeypatch,
) -> None:
    """Repo without ``design_context_sources`` → clear error label
    (no LLM call attempted, no proposal returned)."""

    _make_repo(tmp_path)
    (tmp_path / ".colony").mkdir()
    (tmp_path / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n  - { name: default, type: git_repo }\n",
        encoding="utf-8",
    )

    cap = _make_capability(tmp_path, github=_make_github_stub())
    result = await cap.bootstrap_roadmap_from_objectives()
    assert result["error"] == "no_design_context_sources"
    assert result["proposal"] is None


async def test_bootstrap_apply_writes_roadmap_commits_and_creates_issues(
    tmp_path: Path, monkeypatch,
) -> None:
    """``dry_run=False`` writes ROADMAP.md + commits it + creates
    one GitHub issue per task with the stable-id marker. Verifies
    the full happy path end-to-end."""

    import git

    _seed_design_context_repo(tmp_path)
    github = _make_github_stub(issues=[])
    # Stub create_issue success per call.
    github.create_issue = AsyncMock(side_effect=lambda *, title, body, **_kw: {
        "ok": True, "message": "",
        "issue": {"number": 100, "title": title, "body": body},
    })
    cap = _make_capability(tmp_path, github=github)

    response = (
        '{"milestones": ['
        '{"title": "M1", "description": "first",'
        '"tasks": ['
        '{"title": "task A", "description": "do A", "labels": ["a"]},'
        '{"title": "task B", "description": "do B"}'
        ']}'
        ']}'
    )
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_llm_callable(response),
    )

    result = await cap.bootstrap_roadmap_from_objectives(dry_run=False)
    assert result["error"] == ""
    assert result["dry_run"] is False
    assert result["roadmap_written"] == "ROADMAP.md"
    assert result["commit_sha"]
    assert result["stats"]["issues_created_count"] == 2

    # ROADMAP.md exists with the markers.
    roadmap = (tmp_path / "ROADMAP.md").read_text(encoding="utf-8")
    assert "# Roadmap" in roadmap
    assert "## M1" in roadmap
    assert "task A" in roadmap
    assert "task B" in roadmap
    assert "<!-- colony:roadmap-task:" in roadmap

    # Commit on HEAD with the expected message.
    repo = git.Repo(str(tmp_path))
    assert repo.head.commit.hexsha == result["commit_sha"]
    assert "Bootstrap roadmap" in repo.head.commit.message

    # Each create_issue call carried the marker in the body.
    assert github.create_issue.await_count == 2
    bodies = [
        c.kwargs["body"]
        for c in github.create_issue.await_args_list
    ]
    for body in bodies:
        assert "<!-- colony:roadmap-task:" in body

    # Issues_created carries the stable_id per task.
    stable_ids = {row["stable_id"] for row in result["issues_created"]}
    assert len(stable_ids) == 2


async def test_bootstrap_apply_records_issue_failures_without_failing_overall(
    tmp_path: Path, monkeypatch,
) -> None:
    """One create_issue fails → recorded in ``issue_failures``;
    other tasks still attempt creation; ROADMAP.md still written
    + committed. Partial-success is the realistic shape."""

    _seed_design_context_repo(tmp_path)
    github = _make_github_stub(issues=[])

    async def _flaky_create_issue(*, title, body, **_kw):
        if title == "task A":
            return {"ok": False, "message": "rate limited"}
        return {
            "ok": True, "message": "",
            "issue": {"number": 100, "title": title, "body": body},
        }
    github.create_issue = AsyncMock(side_effect=_flaky_create_issue)
    cap = _make_capability(tmp_path, github=github)

    response = (
        '{"milestones": [{"title": "M1", "description": "",'
        '"tasks": ['
        '{"title": "task A", "description": "do A"},'
        '{"title": "task B", "description": "do B"}'
        ']}]}'
    )
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_llm_callable(response),
    )

    result = await cap.bootstrap_roadmap_from_objectives(dry_run=False)
    assert result["error"] == ""
    assert result["stats"]["issues_created_count"] == 1
    assert result["stats"]["issue_failure_count"] == 1
    failure = result["issue_failures"][0]
    assert failure["task"] == "task A"
    assert "rate limited" in failure["error"]
    # ROADMAP.md was still written + committed.
    assert (tmp_path / "ROADMAP.md").is_file()
    assert result["commit_sha"]


async def test_bootstrap_apply_no_github_sibling_writes_roadmap_skips_issues(
    tmp_path: Path, monkeypatch,
) -> None:
    """Operator may have configured no GitHubCapability — the
    bootstrap still writes ROADMAP.md + commits, just doesn't create
    issues. Logs a WARN."""

    _seed_design_context_repo(tmp_path)
    cap = _make_capability(tmp_path)  # no github sibling

    response = (
        '{"milestones": [{"title": "M1", "description": "",'
        '"tasks": [{"title": "task", "description": "do it"}]}]}'
    )
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_llm_callable(response),
    )

    result = await cap.bootstrap_roadmap_from_objectives(dry_run=False)
    assert result["error"] == ""
    assert (tmp_path / "ROADMAP.md").is_file()
    assert result["commit_sha"]
    assert result["issues_created"] == []
    assert result["stats"]["issues_created_count"] == 0


async def test_bootstrap_returns_clear_error_on_llm_parse_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    """Malformed LLM output → ``error='llm_proposal_parse_failed'``
    with a raw excerpt for debugging. No mutations either way."""

    _seed_design_context_repo(tmp_path)
    cap = _make_capability(tmp_path, github=_make_github_stub())

    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_llm_callable("definitely not JSON"),
    )

    result = await cap.bootstrap_roadmap_from_objectives(dry_run=False)
    assert result["error"] == "llm_proposal_parse_failed"
    assert "definitely not JSON" in result["raw_excerpt"]
    assert not (tmp_path / "ROADMAP.md").exists()


async def test_bootstrap_returns_clear_error_on_llm_timeout(
    tmp_path: Path, monkeypatch,
) -> None:
    """Live LLM hangs → asyncio.TimeoutError → clear error label,
    no mutations."""

    import asyncio as _asyncio

    _seed_design_context_repo(tmp_path)
    cap = _make_capability(tmp_path, github=_make_github_stub())

    def _builder(*, max_tokens: int, temperature: float):
        async def _hang(_prompt: str) -> str:
            await _asyncio.sleep(10.0)
            return "{}"
        return _hang
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable", _builder,
    )

    result = await cap.bootstrap_roadmap_from_objectives(
        dry_run=True, llm_timeout_s=0.05,
    )
    assert result["error"] == "llm_timeout"
    assert result["proposal"] is None


async def test_bootstrap_writes_to_custom_roadmap_path(
    tmp_path: Path, monkeypatch,
) -> None:
    """Operator passes a non-default ``roadmap_path`` (e.g. nested
    location) → action creates parent dirs + writes there."""

    _seed_design_context_repo(tmp_path)
    cap = _make_capability(tmp_path)

    response = (
        '{"milestones": [{"title": "M", "description": "",'
        '"tasks": [{"title": "t", "description": ""}]}]}'
    )
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_llm_callable(response),
    )

    result = await cap.bootstrap_roadmap_from_objectives(
        dry_run=False, roadmap_path="docs/planning/ROADMAP.md",
    )
    assert result["error"] == ""
    assert (tmp_path / "docs" / "planning" / "ROADMAP.md").is_file()


async def test_bootstrap_create_issue_calls_carry_target_project_id(
    tmp_path: Path, monkeypatch,
) -> None:
    """The action forwards ``target_project_id`` through every
    ``create_issue`` call so issues land on the operator's Project
    board (P4 auto-attach hook)."""

    _seed_design_context_repo(tmp_path)
    github = _make_github_stub(issues=[])
    github.create_issue = AsyncMock(return_value={
        "ok": True, "message": "",
        "issue": {"number": 1, "title": "x"},
    })
    cap = _make_capability(tmp_path, github=github)

    response = (
        '{"milestones": [{"title": "M", "description": "",'
        '"tasks": [{"title": "t", "description": ""}]}]}'
    )
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_llm_callable(response),
    )

    await cap.bootstrap_roadmap_from_objectives(
        dry_run=False, target_project_id="PVT_OPERATOR",
    )
    call = github.create_issue.await_args
    assert call.kwargs["project_id"] == "PVT_OPERATOR"


# ===========================================================================
# P5c: sync_roadmap_with_github — diff helpers + action
# ===========================================================================


# --- _parse_roadmap_markdown pure helper -----------------------------------


def test_parse_roadmap_markdown_round_trips_renderer_output() -> None:
    """The parser MUST round-trip whatever the renderer produces —
    they're a matched pair (sync is the second leg)."""

    from polymathera.colony.design_monorepo.capabilities import (
        _parse_roadmap_markdown, _render_roadmap_markdown,
        _stable_task_id,
    )

    proposal = {
        "milestones": [
            {
                "title": "Milestone 1", "description": "first cut",
                "tasks": [
                    {
                        "title": "task A", "description": "do A",
                        "labels": [],
                        "stable_id": _stable_task_id("Milestone 1", "task A"),
                    },
                    {
                        "title": "task B", "description": "",
                        "labels": [],
                        "stable_id": _stable_task_id("Milestone 1", "task B"),
                    },
                ],
            },
            {
                "title": "Milestone 2", "description": "",
                "tasks": [
                    {
                        "title": "task C", "description": "do C",
                        "labels": [],
                        "stable_id": _stable_task_id("Milestone 2", "task C"),
                    },
                ],
            },
        ],
    }
    text = _render_roadmap_markdown(proposal)
    parsed = _parse_roadmap_markdown(text)

    # Same milestone count + names + task stable_ids.
    assert len(parsed["milestones"]) == 2
    assert [m["title"] for m in parsed["milestones"]] == [
        "Milestone 1", "Milestone 2",
    ]
    for orig_m, parsed_m in zip(
        proposal["milestones"], parsed["milestones"], strict=True,
    ):
        assert len(parsed_m["tasks"]) == len(orig_m["tasks"])
        for orig_t, parsed_t in zip(
            orig_m["tasks"], parsed_m["tasks"], strict=True,
        ):
            assert parsed_t["title"] == orig_t["title"]
            assert parsed_t["stable_id"] == orig_t["stable_id"]
            assert parsed_t["description"] == orig_t["description"]


def test_parse_roadmap_markdown_empty_text_returns_empty_structure() -> None:
    """An empty roadmap (no file, or empty file) parses to an empty
    milestones list — symmetric with the renderer's input shape so
    the diff downstream is well-defined."""

    from polymathera.colony.design_monorepo.capabilities import (
        _parse_roadmap_markdown,
    )

    assert _parse_roadmap_markdown("") == {"milestones": []}
    assert _parse_roadmap_markdown(
        "# Roadmap\n\nsome preamble\n",
    ) == {"milestones": []}


def test_parse_roadmap_markdown_skips_task_lines_without_marker() -> None:
    """Operator may add ad-hoc bullets in ROADMAP.md without
    markers; the parser ignores those (no stable_id = nothing to
    sync). Markers are the join key."""

    from polymathera.colony.design_monorepo.capabilities import (
        _parse_roadmap_markdown,
    )

    text = (
        "## M1\n"
        "- [ ] **typed task** <!-- colony:roadmap-task: aaaaaaaaaaaa -->\n"
        "- [ ] ad-hoc bullet with no marker\n"
        "- arbitrary prose\n"
    )
    parsed = _parse_roadmap_markdown(text)
    assert len(parsed["milestones"]) == 1
    assert len(parsed["milestones"][0]["tasks"]) == 1
    assert parsed["milestones"][0]["tasks"][0]["stable_id"] == "aaaaaaaaaaaa"


# --- _diff_roadmap_vs_issues pure helper -----------------------------------


def test_diff_buckets_correctly_when_all_in_sync() -> None:
    """When ROADMAP.md and GitHub issues are perfectly aligned,
    every match lands in ``in_sync`` and other buckets are empty."""

    from polymathera.colony.design_monorepo.capabilities import (
        _diff_roadmap_vs_issues,
    )

    roadmap = {
        "milestones": [{
            "title": "M1", "description": "",
            "tasks": [
                {"title": "task A", "stable_id": "aaaaaaaaaaaa"},
                {"title": "task B", "stable_id": "bbbbbbbbbbbb"},
            ],
        }],
    }
    issues = [
        {
            "number": 1, "title": "task A", "state": "open",
            "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
        },
        {
            "number": 2, "title": "task B", "state": "closed",
            "body": "<!-- colony:roadmap-task: bbbbbbbbbbbb -->",
        },
    ]
    diff = _diff_roadmap_vs_issues(roadmap, issues)
    assert diff["roadmap_only"] == []
    assert diff["github_only"] == []
    assert diff["divergent"] == []
    assert len(diff["in_sync"]) == 2
    assert diff["untracked_issues"] == []


def test_diff_surfaces_roadmap_only_tasks() -> None:
    from polymathera.colony.design_monorepo.capabilities import (
        _diff_roadmap_vs_issues,
    )

    roadmap = {
        "milestones": [{
            "title": "M1", "description": "",
            "tasks": [
                {"title": "newly added", "stable_id": "ccccccccccccc"[:12]},
            ],
        }],
    }
    diff = _diff_roadmap_vs_issues(roadmap, [])
    assert len(diff["roadmap_only"]) == 1
    assert diff["roadmap_only"][0]["stable_id"] == "ccccccccccccc"[:12]
    assert diff["roadmap_only"][0]["milestone_title"] == "M1"
    assert diff["in_sync"] == []


def test_diff_surfaces_github_only_tasks() -> None:
    from polymathera.colony.design_monorepo.capabilities import (
        _diff_roadmap_vs_issues,
    )

    diff = _diff_roadmap_vs_issues(
        {"milestones": []},
        [
            {
                "number": 5, "title": "loner", "state": "open",
                "body": "<!-- colony:roadmap-task: deadbeefdead -->",
                "url": "u",
            },
        ],
    )
    assert len(diff["github_only"]) == 1
    assert diff["github_only"][0]["stable_id"] == "deadbeefdead"
    assert diff["github_only"][0]["issue_number"] == 5


def test_diff_surfaces_divergent_tasks_when_titles_differ() -> None:
    """Same stable_id, different titles → conflict bucket. Sync
    surfaces these for operator mediation; doesn't auto-resolve."""

    from polymathera.colony.design_monorepo.capabilities import (
        _diff_roadmap_vs_issues,
    )

    diff = _diff_roadmap_vs_issues(
        {
            "milestones": [{
                "title": "M1", "description": "",
                "tasks": [
                    {
                        "title": "title in roadmap",
                        "stable_id": "111111111111",
                    },
                ],
            }],
        },
        [{
            "number": 9, "title": "title on github (renamed)",
            "state": "open",
            "body": "<!-- colony:roadmap-task: 111111111111 -->",
            "url": "u",
        }],
    )
    assert len(diff["divergent"]) == 1
    conf = diff["divergent"][0]
    assert conf["stable_id"] == "111111111111"
    assert conf["roadmap_title"] == "title in roadmap"
    assert conf["issue_title"] == "title on github (renamed)"
    assert diff["in_sync"] == []


def test_diff_surfaces_untracked_issues_separately() -> None:
    """Issues without markers are tracked but NOT auto-imported —
    operator chose to create them outside the bootstrap flow."""

    from polymathera.colony.design_monorepo.capabilities import (
        _diff_roadmap_vs_issues,
    )

    diff = _diff_roadmap_vs_issues(
        {"milestones": []},
        [
            {
                "number": 1, "title": "no-marker", "state": "open",
                "body": "just a description, no marker",
            },
            {
                "number": 2, "title": "also no-marker", "state": "open",
                "body": "",
            },
        ],
    )
    assert len(diff["untracked_issues"]) == 2
    assert diff["github_only"] == []
    assert diff["roadmap_only"] == []


# --- _merge_github_only_into_roadmap helper --------------------------------


def test_merge_github_only_creates_untracked_milestone() -> None:
    from polymathera.colony.design_monorepo.capabilities import (
        _UNTRACKED_MILESTONE_TITLE, _merge_github_only_into_roadmap,
    )

    roadmap = {
        "milestones": [
            {"title": "M1", "description": "", "tasks": []},
        ],
    }
    github_only = [{
        "stable_id": "abcabcabcabc",
        "title": "imported",
        "issue_number": 7, "issue_state": "open",
    }]
    out = _merge_github_only_into_roadmap(roadmap, github_only)

    # Original roadmap unchanged.
    assert roadmap["milestones"][0]["tasks"] == []

    # Output has the Untracked milestone appended.
    assert len(out["milestones"]) == 2
    untracked = out["milestones"][-1]
    assert untracked["title"] == _UNTRACKED_MILESTONE_TITLE
    assert len(untracked["tasks"]) == 1
    assert untracked["tasks"][0]["stable_id"] == "abcabcabcabc"
    assert untracked["tasks"][0]["title"] == "imported"
    assert "#7" in untracked["tasks"][0]["description"]


def test_merge_github_only_reuses_existing_untracked_milestone() -> None:
    """If ``Untracked`` already exists (from a prior sync), append
    to it rather than create a duplicate."""

    from polymathera.colony.design_monorepo.capabilities import (
        _UNTRACKED_MILESTONE_TITLE, _merge_github_only_into_roadmap,
    )

    roadmap = {
        "milestones": [
            {
                "title": _UNTRACKED_MILESTONE_TITLE,
                "description": "", "tasks": [
                    {"title": "from previous sync", "stable_id": "1" * 12},
                ],
            },
        ],
    }
    out = _merge_github_only_into_roadmap(
        roadmap,
        [{
            "stable_id": "2" * 12, "title": "new",
            "issue_number": 1, "issue_state": "open",
        }],
    )
    # One milestone, two tasks (one prior + one new).
    assert len(out["milestones"]) == 1
    assert len(out["milestones"][0]["tasks"]) == 2


def test_merge_github_only_noop_when_empty() -> None:
    from polymathera.colony.design_monorepo.capabilities import (
        _merge_github_only_into_roadmap,
    )

    original = {"milestones": [{"title": "M1", "description": "", "tasks": []}]}
    out = _merge_github_only_into_roadmap(original, [])
    assert out is original  # no-op shortcut


# --- sync_roadmap_with_github action ---------------------------------------


def _seed_roadmap_repo(
    tmp_path: Path, *, roadmap_text: str | None = None,
) -> None:
    """Repo with a design_context_sources block + optionally a
    pre-existing ROADMAP.md so the sync action has something to diff."""

    _make_repo(tmp_path)
    (tmp_path / ".colony").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "x.md").write_text("# X\n", encoding="utf-8")
    (tmp_path / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n  - { name: default, type: git_repo }\n"
        "design_context_sources:\n"
        "  - name: docs\n    paths: ['docs/**/*.md']\n",
        encoding="utf-8",
    )
    if roadmap_text is not None:
        (tmp_path / "ROADMAP.md").write_text(
            roadmap_text, encoding="utf-8",
        )


async def test_sync_dry_run_returns_diff_without_writing(
    tmp_path: Path,
) -> None:
    """``dry_run=True`` returns the full diff + the planned-actions
    summary; ROADMAP.md untouched; no create_issue calls."""

    roadmap_text = (
        "# Roadmap\n\n"
        "## M1\n\n"
        "- [ ] **task A** <!-- colony:roadmap-task: aaaaaaaaaaaa -->\n"
    )
    _seed_roadmap_repo(tmp_path, roadmap_text=roadmap_text)

    # Issue exists for task A → in_sync. No other tasks/issues.
    github = _make_github_stub(issues=[{
        "number": 1, "title": "task A", "state": "open",
        "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
    }])
    cap = _make_capability(tmp_path, github=github)

    result = await cap.sync_roadmap_with_github(
        emit_blackboard_events=False,
    )
    assert result["dry_run"] is True
    assert result["direction"] == "bidirectional"
    assert result["error"] == ""
    assert result["stats"]["in_sync_count"] == 1
    assert result["stats"]["roadmap_only_count"] == 0
    assert result["stats"]["github_only_count"] == 0
    # No mutations.
    assert (tmp_path / "ROADMAP.md").read_text(encoding="utf-8") == roadmap_text
    github.create_issue.assert_not_called()


async def test_sync_bidirectional_apply_creates_issues_and_appends_to_roadmap(
    tmp_path: Path,
) -> None:
    """Apply mode: one roadmap-only task → create issue (with marker),
    one github-only issue → append to ROADMAP.md under Untracked,
    one in-sync → no action, one divergent → no auto-action."""

    import git

    # Roadmap has: aaa (in sync), bbb (roadmap-only), ccc (divergent).
    roadmap_text = (
        "# Roadmap\n\n"
        "## M1\n\n"
        "- [ ] **task A** <!-- colony:roadmap-task: aaaaaaaaaaaa -->\n"
        "- [ ] **task B (roadmap-only)** <!-- colony:roadmap-task: bbbbbbbbbbbb -->\n"
        "- [ ] **task C in roadmap** <!-- colony:roadmap-task: cccccccccccc -->\n"
    )
    _seed_roadmap_repo(tmp_path, roadmap_text=roadmap_text)

    # GitHub: aaa (in sync), ddd (github-only, has marker),
    # ccc (divergent — different title), and one untracked (no marker).
    github = _make_github_stub(issues=[
        {
            "number": 1, "title": "task A", "state": "open",
            "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
        },
        {
            "number": 4, "title": "task D (github-only)", "state": "open",
            "body": "<!-- colony:roadmap-task: dddddddddddd -->",
        },
        {
            "number": 3, "title": "task C renamed on github",
            "state": "open",
            "body": "<!-- colony:roadmap-task: cccccccccccc -->",
        },
        {
            "number": 99, "title": "untracked", "state": "open",
            "body": "no marker here",
        },
    ])
    github.create_issue = AsyncMock(side_effect=lambda *, title, body, **_: {
        "ok": True, "message": "",
        "issue": {"number": 100, "title": title, "body": body},
    })
    cap = _make_capability(tmp_path, github=github)

    result = await cap.sync_roadmap_with_github(
        dry_run=False, emit_blackboard_events=False,
    )
    assert result["error"] == ""
    assert result["stats"]["roadmap_only_count"] == 1
    assert result["stats"]["github_only_count"] == 1
    assert result["stats"]["divergent_count"] == 1
    assert result["stats"]["untracked_issue_count"] == 1
    assert result["stats"]["in_sync_count"] == 1
    # One issue created (for bbb only).
    assert result["stats"]["issues_created_count"] == 1
    created_body = github.create_issue.await_args.kwargs["body"]
    assert "bbbbbbbbbbbb" in created_body
    # Roadmap was rewritten (ddd appended).
    assert result["roadmap_written"] is True
    new_roadmap = (tmp_path / "ROADMAP.md").read_text(encoding="utf-8")
    assert "Untracked" in new_roadmap
    assert "task D (github-only)" in new_roadmap
    assert "dddddddddddd" in new_roadmap
    # Commit landed.
    repo = git.Repo(str(tmp_path))
    assert repo.head.commit.hexsha == result["commit_sha"]
    # Divergent task: no auto-action — issue still on GitHub with old
    # title, roadmap still has its own title.
    assert "task C in roadmap" in new_roadmap


async def test_sync_roadmap_to_github_only_skips_roadmap_writes(
    tmp_path: Path,
) -> None:
    """``direction='roadmap_to_github'`` creates issues for
    roadmap-only tasks but leaves ROADMAP.md untouched even when
    github-only entries exist."""

    roadmap_text = (
        "## M1\n\n"
        "- [ ] **roadmap-only** <!-- colony:roadmap-task: aaaaaaaaaaaa -->\n"
    )
    _seed_roadmap_repo(tmp_path, roadmap_text=roadmap_text)
    github = _make_github_stub(issues=[
        {
            "number": 5, "title": "github-only", "state": "open",
            "body": "<!-- colony:roadmap-task: bbbbbbbbbbbb -->",
        },
    ])
    github.create_issue = AsyncMock(return_value={
        "ok": True, "message": "", "issue": {"number": 100, "title": "x"},
    })
    cap = _make_capability(tmp_path, github=github)

    result = await cap.sync_roadmap_with_github(
        direction="roadmap_to_github", dry_run=False,
        emit_blackboard_events=False,
    )
    assert result["error"] == ""
    # Issue was created.
    assert github.create_issue.await_count == 1
    # ROADMAP.md was NOT touched.
    assert result["roadmap_written"] is False
    assert "bbbbbbbbbbbb" not in (
        tmp_path / "ROADMAP.md"
    ).read_text(encoding="utf-8")


async def test_sync_github_to_roadmap_only_skips_issue_creation(
    tmp_path: Path,
) -> None:
    """``direction='github_to_roadmap'`` updates ROADMAP.md from
    github-only issues but does NOT call create_issue for roadmap-
    only tasks."""

    roadmap_text = (
        "## M1\n\n"
        "- [ ] **roadmap-only** <!-- colony:roadmap-task: aaaaaaaaaaaa -->\n"
    )
    _seed_roadmap_repo(tmp_path, roadmap_text=roadmap_text)
    github = _make_github_stub(issues=[
        {
            "number": 5, "title": "github-only", "state": "open",
            "body": "<!-- colony:roadmap-task: bbbbbbbbbbbb -->",
        },
    ])
    github.create_issue = AsyncMock()
    cap = _make_capability(tmp_path, github=github)

    result = await cap.sync_roadmap_with_github(
        direction="github_to_roadmap", dry_run=False,
        emit_blackboard_events=False,
    )
    assert result["error"] == ""
    github.create_issue.assert_not_called()
    # ROADMAP.md got the github-only entry.
    assert result["roadmap_written"] is True
    new_roadmap = (tmp_path / "ROADMAP.md").read_text(encoding="utf-8")
    assert "bbbbbbbbbbbb" in new_roadmap
    assert "Untracked" in new_roadmap


async def test_sync_emits_roadmap_sync_blackboard_event(
    tmp_path: Path,
) -> None:
    """One :class:`RoadmapSyncProtocol` event per run, even in
    dry_run (so the dashboard knows when a sync was last attempted)."""

    _seed_roadmap_repo(tmp_path, roadmap_text=(
        "## M1\n\n- [ ] **t** <!-- colony:roadmap-task: aaaaaaaaaaaa -->\n"
    ))
    github = _make_github_stub(issues=[{
        "number": 1, "title": "t", "state": "open",
        "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
    }])
    cap = _make_capability(tmp_path, github=github)
    fake_bb = MagicMock()
    fake_bb.write = AsyncMock()
    cap._colony_blackboard = fake_bb

    await cap.sync_roadmap_with_github(emit_blackboard_events=True)
    assert fake_bb.write.await_count == 1
    call = fake_bb.write.await_args
    assert call.kwargs["key"].startswith("roadmap_sync:")
    assert ":bidirectional:" in call.kwargs["key"]
    assert "roadmap_sync" in call.kwargs["tags"]
    assert "dry_run" in call.kwargs["tags"]
    assert call.kwargs["value"]["dry_run"] is True


async def test_sync_returns_error_when_direction_invalid(
    tmp_path: Path,
) -> None:
    _seed_roadmap_repo(tmp_path)
    cap = _make_capability(tmp_path, github=_make_github_stub())
    result = await cap.sync_roadmap_with_github(
        direction="diagonal",  # nonsense
    )
    assert result["error"] == "invalid_direction"
    assert "diagonal" in result["message"]


async def test_sync_returns_clear_error_when_no_github_sibling(
    tmp_path: Path,
) -> None:
    _seed_roadmap_repo(tmp_path)
    cap = _make_capability(tmp_path)  # no github
    result = await cap.sync_roadmap_with_github()
    assert result["error"] == "github_capability_missing"


async def test_sync_no_op_when_fully_in_sync_does_not_touch_roadmap(
    tmp_path: Path,
) -> None:
    """All buckets except in_sync are empty → no ROADMAP.md
    rewrite, no create_issue calls, no commit. Verified by the
    file mtime not changing."""

    roadmap_text = (
        "# Roadmap\n\n"
        "## M1\n\n"
        "- [ ] **task A** <!-- colony:roadmap-task: aaaaaaaaaaaa -->\n"
    )
    _seed_roadmap_repo(tmp_path, roadmap_text=roadmap_text)
    orig_mtime = (tmp_path / "ROADMAP.md").stat().st_mtime
    github = _make_github_stub(issues=[{
        "number": 1, "title": "task A", "state": "open",
        "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
    }])
    github.create_issue = AsyncMock()
    cap = _make_capability(tmp_path, github=github)

    result = await cap.sync_roadmap_with_github(
        dry_run=False, emit_blackboard_events=False,
    )
    assert result["error"] == ""
    assert result["stats"]["in_sync_count"] == 1
    assert result["roadmap_written"] is False
    assert result["commit_sha"] == ""
    github.create_issue.assert_not_called()
    # Mtime unchanged.
    assert (tmp_path / "ROADMAP.md").stat().st_mtime == orig_mtime


async def test_sync_create_issue_failures_surface_in_response(
    tmp_path: Path,
) -> None:
    """Per-issue creation failures land in ``issue_failures`` but
    don't fail the overall action; the rest of the diff still
    applies."""

    roadmap_text = (
        "## M1\n\n"
        "- [ ] **task A** <!-- colony:roadmap-task: aaaaaaaaaaaa -->\n"
        "- [ ] **task B** <!-- colony:roadmap-task: bbbbbbbbbbbb -->\n"
    )
    _seed_roadmap_repo(tmp_path, roadmap_text=roadmap_text)
    github = _make_github_stub(issues=[])

    async def _flaky(*, title, body, **_):
        if title == "task A":
            return {"ok": False, "message": "rate limited"}
        return {
            "ok": True, "message": "",
            "issue": {"number": 100, "title": title},
        }
    github.create_issue = AsyncMock(side_effect=_flaky)
    cap = _make_capability(tmp_path, github=github)

    result = await cap.sync_roadmap_with_github(
        dry_run=False, emit_blackboard_events=False,
    )
    assert result["error"] == ""
    assert result["stats"]["issues_created_count"] == 1
    assert result["stats"]["issue_failure_count"] == 1
    assert result["issue_failures"][0]["task"] == "task A"
    assert "rate limited" in result["issue_failures"][0]["error"]


# ---------------------------------------------------------------------------
# P5d: propose_task_assignments
# ---------------------------------------------------------------------------


def _stub_assignment_llm_callable(
    by_title: dict[str, dict[str, str]] | None = None,
    default: dict[str, str] | None = None,
) -> Any:
    """LLM callable shaped like the one P5d's action builds.

    Pattern-matches needles in ``by_title`` against the prompt's
    ``Task:`` line *only* (not the full prompt — the system prompt
    incidentally mentions things like "SolidWorks" / "to user" and
    we'd otherwise match the system text instead of the task
    title). Falls back to ``default`` on miss.
    """

    import json as _json
    import re as _re

    fallback = default or {"assignee": "colony", "reason": "default"}

    def _builder(*, max_tokens: int, temperature: float):
        async def _llm(prompt: str) -> str:
            m = _re.search(r"^Task:\s*(.+?)$", prompt, _re.MULTILINE)
            task_title = m.group(1).strip() if m else ""
            for needle, payload in (by_title or {}).items():
                if needle in task_title:
                    return _json.dumps(payload)
            return _json.dumps(fallback)
        return _llm
    return _builder


def _stub_github_with_assignment(
    *,
    issues: list[dict],
    whoami_login: str = "colony-bot[bot]",
    whoami_ok: bool = True,
    assign_results: dict[int, dict[str, Any]] | None = None,
) -> Any:
    """A GitHubCapability fake that supports list_issues + whoami +
    assign_issue. ``assign_results`` keyed by issue number lets tests
    pin per-issue assign outcomes (default: ok)."""

    from polymathera.colony.agents.patterns.capabilities.github import (
        GitHubCapability,
    )
    fake = MagicMock(spec=GitHubCapability)
    fake._default_repo = "acme/proj"
    fake.list_issues = AsyncMock(return_value={
        "ok": True, "message": "", "issues": issues, "count": len(issues),
    })
    if whoami_ok:
        fake.whoami = AsyncMock(return_value={
            "ok": True, "message": "",
            "login": whoami_login,
            "slug": whoami_login.replace("[bot]", ""),
            "app_id": "42",
            "app_url": f"https://github.com/apps/{whoami_login.replace('[bot]', '')}",
        })
    else:
        fake.whoami = AsyncMock(return_value={
            "ok": False, "message": "App slug not configured",
            "login": None, "slug": None,
        })

    async def _assign(issue_number, assignees, *, repo=None, replace=True):
        pinned = (assign_results or {}).get(issue_number)
        if pinned is not None:
            return pinned
        return {
            "ok": True, "message": "",
            "issue": {
                "number": issue_number, "title": "t", "state": "open",
                "assignees": list(assignees),
            },
        }
    fake.assign_issue = AsyncMock(side_effect=_assign)
    return fake


async def test_propose_task_assignments_marker_path_skips_llm(
    tmp_path: Path, monkeypatch,
) -> None:
    """When the roadmap task line carries an explicit assignee marker
    (``colony:assignee: colony|user``), the action honours it and
    NEVER invokes the LLM callable for that task."""

    roadmap_text = (
        "# Roadmap\n\n"
        "## M1\n\n"
        "- [ ] **task A** "
        "<!-- colony:roadmap-task: aaaaaaaaaaaa --> "
        "<!-- colony:assignee: colony -->\n"
        "- [ ] **task B** "
        "<!-- colony:roadmap-task: bbbbbbbbbbbb --> "
        "<!-- colony:assignee: user -->\n"
    )
    _seed_roadmap_repo(tmp_path, roadmap_text=roadmap_text)
    github = _stub_github_with_assignment(issues=[
        {
            "number": 1, "title": "task A", "state": "open",
            "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
            "assignees": [], "milestone": "M1",
        },
        {
            "number": 2, "title": "task B", "state": "open",
            "body": "<!-- colony:roadmap-task: bbbbbbbbbbbb -->",
            "assignees": [], "milestone": "M1",
        },
    ])

    # The LLM must never be called — explode if it is.
    from polymathera.colony.knowledge import deps as kdeps_mod

    def _explode(*, max_tokens, temperature):  # noqa: ARG001
        async def _llm(_prompt):
            raise AssertionError("LLM should not be invoked on marker path")
        return _llm
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable", _explode,
    )

    cap = _make_capability(tmp_path, github=github)
    result = await cap.propose_task_assignments(
        user_github_login="alice",
    )
    assert result["dry_run"] is True
    assert result["error"] == ""
    by_id = {p["stable_id"]: p for p in result["proposals"]}
    assert by_id["aaaaaaaaaaaa"]["proposed_assignee"] == "colony"
    assert by_id["aaaaaaaaaaaa"]["proposed_login"] == "colony-bot[bot]"
    assert by_id["aaaaaaaaaaaa"]["source"] == "marker"
    assert by_id["bbbbbbbbbbbb"]["proposed_assignee"] == "user"
    assert by_id["bbbbbbbbbbbb"]["proposed_login"] == "alice"
    assert by_id["bbbbbbbbbbbb"]["source"] == "marker"
    assert result["stats"]["marker_count"] == 2
    assert result["stats"]["llm_count"] == 0


async def test_propose_task_assignments_issue_body_marker_overrides(
    tmp_path: Path, monkeypatch,
) -> None:
    """When the assignee marker lives in the GitHub issue body
    (operator edited it after creation), the action picks it up
    without needing to consult the roadmap."""

    _seed_roadmap_repo(tmp_path, roadmap_text=None)
    github = _stub_github_with_assignment(issues=[
        {
            "number": 1, "title": "Solder the new shield",
            "state": "open",
            "body": (
                "Some prose.\n"
                "<!-- colony:roadmap-task: aaaaaaaaaaaa -->\n"
                "<!-- colony:assignee: user -->\n"
            ),
            "assignees": [], "milestone": "M1",
        },
    ])
    from polymathera.colony.knowledge import deps as kdeps_mod

    def _explode(*, max_tokens, temperature):  # noqa: ARG001
        async def _llm(_prompt):
            raise AssertionError("LLM should not be invoked")
        return _llm
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable", _explode,
    )

    cap = _make_capability(tmp_path, github=github)
    result = await cap.propose_task_assignments(
        user_github_login="alice",
    )
    p = result["proposals"][0]
    assert p["proposed_assignee"] == "user"
    assert p["proposed_login"] == "alice"
    assert p["source"] == "marker"


async def test_propose_task_assignments_llm_path_classifies(
    tmp_path: Path, monkeypatch,
) -> None:
    """No marker → LLM classifies based on task title. The fake LLM
    returns ``user`` for the CAD task and ``colony`` for the analysis
    task — the resolved logins follow accordingly."""

    _seed_roadmap_repo(tmp_path, roadmap_text=None)
    github = _stub_github_with_assignment(issues=[
        {
            "number": 1, "title": "Model the chamber in SolidWorks",
            "state": "open",
            "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
            "assignees": [], "milestone": "M1",
        },
        {
            "number": 2, "title": "Analyse FEMM simulation results",
            "state": "open",
            "body": "<!-- colony:roadmap-task: bbbbbbbbbbbb -->",
            "assignees": [], "milestone": "M1",
        },
    ])
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_assignment_llm_callable(by_title={
            "SolidWorks": {
                "assignee": "user",
                "reason": "needs CAD software a human owns",
            },
            "FEMM": {
                "assignee": "colony",
                "reason": "pure analysis over existing results",
            },
        }),
    )

    cap = _make_capability(tmp_path, github=github)
    result = await cap.propose_task_assignments(
        user_github_login="alice",
    )
    by_id = {p["stable_id"]: p for p in result["proposals"]}
    cad = by_id["aaaaaaaaaaaa"]
    analysis = by_id["bbbbbbbbbbbb"]
    assert cad["proposed_assignee"] == "user"
    assert cad["proposed_login"] == "alice"
    assert cad["source"] == "llm"
    assert "CAD" in cad["reason"]
    assert analysis["proposed_assignee"] == "colony"
    assert analysis["proposed_login"] == "colony-bot[bot]"
    assert analysis["source"] == "llm"
    assert result["stats"]["llm_count"] == 2
    assert result["stats"]["proposed_user"] == 1
    assert result["stats"]["proposed_colony"] == 1


async def test_propose_task_assignments_skips_already_assigned_by_default(
    tmp_path: Path, monkeypatch,
) -> None:
    """An issue with non-empty ``assignees`` is skipped (default)
    even when it has a roadmap-task marker. Surfaced in ``skipped``."""

    _seed_roadmap_repo(tmp_path, roadmap_text=None)
    github = _stub_github_with_assignment(issues=[
        {
            "number": 1, "title": "Already mine",
            "state": "open",
            "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
            "assignees": ["bob"], "milestone": "M1",
        },
        {
            "number": 2, "title": "Free task",
            "state": "open",
            "body": "<!-- colony:roadmap-task: bbbbbbbbbbbb -->",
            "assignees": [], "milestone": "M1",
        },
    ])
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_assignment_llm_callable(default={
            "assignee": "colony", "reason": "default",
        }),
    )

    cap = _make_capability(tmp_path, github=github)
    result = await cap.propose_task_assignments(
        user_github_login="alice",
    )
    assert len(result["proposals"]) == 1
    assert result["proposals"][0]["issue_number"] == 2
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["issue_number"] == 1
    assert result["skipped"][0]["reason"] == "already_assigned"
    assert result["skipped"][0]["current_assignees"] == ["bob"]


async def test_propose_task_assignments_reassign_existing_includes_all(
    tmp_path: Path, monkeypatch,
) -> None:
    """``reassign_existing=True`` proposes for every roadmap-linked
    issue, including those already assigned."""

    _seed_roadmap_repo(tmp_path, roadmap_text=None)
    github = _stub_github_with_assignment(issues=[
        {
            "number": 1, "title": "Already mine",
            "state": "open",
            "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
            "assignees": ["bob"], "milestone": "M1",
        },
    ])
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_assignment_llm_callable(default={
            "assignee": "colony", "reason": "default",
        }),
    )

    cap = _make_capability(tmp_path, github=github)
    result = await cap.propose_task_assignments(
        user_github_login="alice", reassign_existing=True,
    )
    assert len(result["proposals"]) == 1
    assert len(result["skipped"]) == 0
    assert result["proposals"][0]["current_assignees"] == ["bob"]


async def test_propose_task_assignments_dry_run_does_not_call_assign(
    tmp_path: Path, monkeypatch,
) -> None:
    """``dry_run=True`` (default) MUST NOT call ``assign_issue``."""

    _seed_roadmap_repo(tmp_path, roadmap_text=None)
    github = _stub_github_with_assignment(issues=[
        {
            "number": 1, "title": "x", "state": "open",
            "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
            "assignees": [], "milestone": "M1",
        },
    ])
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_assignment_llm_callable(default={
            "assignee": "user", "reason": "default",
        }),
    )

    cap = _make_capability(tmp_path, github=github)
    result = await cap.propose_task_assignments(
        user_github_login="alice",
    )
    assert result["dry_run"] is True
    github.assign_issue.assert_not_called()
    assert "applied" not in result


async def test_propose_task_assignments_apply_calls_assign_per_proposal(
    tmp_path: Path, monkeypatch,
) -> None:
    """``dry_run=False`` calls ``assign_issue`` once per proposal with
    ``replace=True`` and the resolved login."""

    _seed_roadmap_repo(tmp_path, roadmap_text=None)
    github = _stub_github_with_assignment(issues=[
        {
            "number": 1, "title": "to colony", "state": "open",
            "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
            "assignees": [], "milestone": "M1",
        },
        {
            "number": 2, "title": "to user", "state": "open",
            "body": "<!-- colony:roadmap-task: bbbbbbbbbbbb -->",
            "assignees": [], "milestone": "M1",
        },
    ])
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_assignment_llm_callable(by_title={
            "to colony": {"assignee": "colony", "reason": "ok"},
            "to user": {"assignee": "user", "reason": "ok"},
        }),
    )

    cap = _make_capability(tmp_path, github=github)
    result = await cap.propose_task_assignments(
        user_github_login="alice", dry_run=False,
    )
    assert result["dry_run"] is False
    assert result["error"] == ""
    assert len(result["applied"]) == 2
    assert len(result["errors"]) == 0
    assert github.assign_issue.await_count == 2
    # Inspect both calls: replace=True + resolved logins.
    calls = github.assign_issue.await_args_list
    by_issue = {
        c.args[0]: (c.args[1], c.kwargs)
        for c in calls
    }
    assert by_issue[1] == (["colony-bot[bot]"], {"repo": None, "replace": True})
    assert by_issue[2] == (["alice"], {"repo": None, "replace": True})


async def test_propose_task_assignments_apply_collects_partial_failures(
    tmp_path: Path, monkeypatch,
) -> None:
    """A failed ``assign_issue`` for one issue is collected into
    ``errors`` and does NOT abort the rest of the run."""

    _seed_roadmap_repo(tmp_path, roadmap_text=None)
    github = _stub_github_with_assignment(
        issues=[
            {
                "number": 1, "title": "ok", "state": "open",
                "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
                "assignees": [], "milestone": "M1",
            },
            {
                "number": 2, "title": "boom", "state": "open",
                "body": "<!-- colony:roadmap-task: bbbbbbbbbbbb -->",
                "assignees": [], "milestone": "M1",
            },
        ],
        assign_results={
            2: {"ok": False, "message": "user does not exist"},
        },
    )
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_assignment_llm_callable(default={
            "assignee": "user", "reason": "default",
        }),
    )

    cap = _make_capability(tmp_path, github=github)
    result = await cap.propose_task_assignments(
        user_github_login="alice", dry_run=False,
    )
    assert result["stats"]["applied_count"] == 1
    assert result["stats"]["error_count"] == 1
    assert result["errors"][0]["issue_number"] == 2
    assert "user does not exist" in result["errors"][0]["error"]


async def test_propose_task_assignments_segregates_untracked_issues(
    tmp_path: Path, monkeypatch,
) -> None:
    """Issues without a roadmap-task marker are NOT classified — they
    surface in ``untracked_issues`` so the operator can decide."""

    _seed_roadmap_repo(tmp_path, roadmap_text=None)
    github = _stub_github_with_assignment(issues=[
        {
            "number": 1, "title": "tracked", "state": "open",
            "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
            "assignees": [], "milestone": "M1",
        },
        {
            "number": 99, "title": "manual", "state": "open",
            "body": "no marker here",
            "assignees": [], "milestone": None,
        },
    ])
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable",
        _stub_assignment_llm_callable(default={
            "assignee": "colony", "reason": "default",
        }),
    )

    cap = _make_capability(tmp_path, github=github)
    result = await cap.propose_task_assignments(
        user_github_login="alice",
    )
    assert len(result["proposals"]) == 1
    assert result["proposals"][0]["issue_number"] == 1
    assert len(result["untracked_issues"]) == 1
    assert result["untracked_issues"][0]["issue_number"] == 99


async def test_propose_task_assignments_errors_when_user_login_missing(
    tmp_path: Path,
) -> None:
    """``user_github_login`` is required — empty/whitespace gets a
    clean error result, not a silent assignment to an empty string."""

    _seed_roadmap_repo(tmp_path, roadmap_text=None)
    github = _stub_github_with_assignment(issues=[])
    cap = _make_capability(tmp_path, github=github)

    result = await cap.propose_task_assignments(user_github_login="")
    assert result["error"] == "user_github_login_required"
    assert "Colony commits" in result["message"]


async def test_propose_task_assignments_errors_when_whoami_fails(
    tmp_path: Path,
) -> None:
    """Whoami failure (e.g. App slug not configured) → clean error
    surface, no issue listing, no LLM call."""

    _seed_roadmap_repo(tmp_path, roadmap_text=None)
    github = _stub_github_with_assignment(
        issues=[], whoami_ok=False,
    )
    cap = _make_capability(tmp_path, github=github)
    result = await cap.propose_task_assignments(
        user_github_login="alice",
    )
    assert result["error"] == "whoami_failed"
    github.list_issues.assert_not_called()


async def test_propose_task_assignments_errors_without_github_sibling(
    tmp_path: Path,
) -> None:
    """No GitHubCapability mounted → clean error."""

    _seed_roadmap_repo(tmp_path, roadmap_text=None)
    cap = _make_capability(tmp_path, github=None)
    result = await cap.propose_task_assignments(
        user_github_login="alice",
    )
    assert result["error"] == "github_capability_missing"


async def test_propose_task_assignments_handles_llm_parse_failures(
    tmp_path: Path, monkeypatch,
) -> None:
    """LLM returns gibberish → the task lands in ``skipped`` with
    ``llm_parse_failed`` reason rather than raising or proposing a
    bogus assignee."""

    _seed_roadmap_repo(tmp_path, roadmap_text=None)
    github = _stub_github_with_assignment(issues=[
        {
            "number": 1, "title": "x", "state": "open",
            "body": "<!-- colony:roadmap-task: aaaaaaaaaaaa -->",
            "assignees": [], "milestone": "M1",
        },
    ])
    from polymathera.colony.knowledge import deps as kdeps_mod

    def _builder(*, max_tokens, temperature):  # noqa: ARG001
        async def _llm(_prompt):
            return "not json at all"
        return _llm
    monkeypatch.setattr(
        kdeps_mod, "build_default_llm_callable", _builder,
    )

    cap = _make_capability(tmp_path, github=github)
    result = await cap.propose_task_assignments(
        user_github_login="alice",
    )
    assert len(result["proposals"]) == 0
    assert len(result["skipped"]) == 1
    assert result["skipped"][0]["reason"] == "llm_parse_failed"


def test_parse_assignment_classification_rejects_unknown_assignee() -> None:
    """A JSON object with ``assignee=other`` (not colony/user) is
    rejected — keeps the binary universe enforced."""

    from polymathera.colony.design_monorepo.capabilities import (
        _parse_assignment_classification,
    )
    assert _parse_assignment_classification(
        '{"assignee": "other", "reason": "..."}',
    ) is None


def test_parse_assignment_classification_strips_code_fences() -> None:
    """LLM wrapping output in ``` blocks should not break parsing."""

    from polymathera.colony.design_monorepo.capabilities import (
        _parse_assignment_classification,
    )
    parsed = _parse_assignment_classification(
        "```json\n"
        '{"assignee": "colony", "reason": "ok"}\n'
        "```",
    )
    assert parsed == {"assignee": "colony", "reason": "ok"}


def test_extract_assignee_marker_case_insensitive() -> None:
    """The marker regex tolerates COLONY vs colony for resilience to
    operator copy-paste."""

    from polymathera.colony.design_monorepo.capabilities import (
        _extract_assignee_marker,
    )
    assert _extract_assignee_marker(
        "<!-- colony:assignee: USER -->",
    ) == "user"
    assert _extract_assignee_marker(
        "no marker here",
    ) is None
    assert _extract_assignee_marker(None) is None
