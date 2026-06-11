"""Per-primitive tests for the decompose action surface on
``DesignProcessCapability``.

Item 3 of ``colony/decompose_and_session_recovery_fixes_plan.md``:
the monolithic ``decompose_issues`` was deleted; the capability now
ships three composable primitives (``classify_issues_decomposability``,
``propose_decompositions``, ``create_decomposition``). Each is
batch-native — N=1 is the natural per-item case. The planner LLM
composes them into whatever strategy fits the data.

Helpers tested separately:
- ``_build_classify_decomposability_prompt`` — batch prompt shape
- ``_parse_classify_decomposability`` — tolerant parser, missing-entry handling
- ``_build_decomposition_prompt`` — joint-decomposition prompt shape
- ``_parse_decomposition_proposal`` — parses parent_proposals + shared_concerns
- ``_render_child_body`` / ``_render_parent_body_with_children`` — markdown helpers
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import git
import pytest

from polymathera.colony.design_monorepo.process import (
    DEFAULT_DECOMPOSITION_CRITERIA,
    DesignProcessCapability,
    _build_classify_decomposability_prompt,
    _build_decomposition_prompt,
    _parse_classify_decomposability,
    _parse_decomposition_proposal,
    _render_child_body,
    _render_parent_body_with_children,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_repo(root: Path) -> None:
    repo = git.Repo.init(root, initial_branch="main")
    repo.config_writer().set_value("user", "email", "t@t").release()
    repo.config_writer().set_value("user", "name", "t").release()
    sentinel = root / ".init"
    sentinel.write_text("init\n", encoding="utf-8")
    repo.index.add([str(sentinel)])
    repo.index.commit("init")
    bare = root.parent / f"{root.name}.git"
    if not bare.exists():
        git.Repo.init(bare, bare=True, initial_branch="main")
    if "origin" not in [r.name for r in repo.remotes]:
        repo.create_remote("origin", str(bare))
    repo.git.push("origin", "main")


def _make_capability(
    tmp_path: Path,
    *,
    github: Any = None,
    llm_response: str | None = None,
) -> DesignProcessCapability:
    _make_repo(tmp_path)
    agent = MagicMock()
    agent.metadata.parameters = {
        "design_monorepo_url": "https://github.com/acme/proj.git",
    }
    if github is not None:
        agent._capabilities = {"github": github}
        agent.get_capability = lambda cls: (
            github
            if any(isinstance(github, cls) for _ in [None])
            else None
        )
    else:
        agent._capabilities = {}
        agent.get_capability = lambda _cls: None
        agent.capability_by_class = None
    if llm_response is not None:
        fake_response = MagicMock()
        fake_response.generated_text = llm_response
        agent.infer = AsyncMock(return_value=fake_response)
    cap = DesignProcessCapability(
        agent=agent, scope_id="test", working_dir=tmp_path,
    )
    return cap


def _make_github_stub(
    *,
    issues_by_number: dict[int, dict[str, Any]] | None = None,
    create_issue_results: list[dict] | None = None,
    update_issue_body_ok: bool = True,
) -> Any:
    """Duck-typed GitHubCapability with the methods the decompose
    primitives need."""

    from polymathera.colony.agents.patterns.capabilities.github import (
        GitHubCapability,
    )

    fake = MagicMock(spec=GitHubCapability)

    async def _get_issue(number):
        data = (issues_by_number or {}).get(number)
        if data is None:
            return {"ok": False, "message": f"not found: #{number}"}
        return {"ok": True, "issue": {"number": number, **data}}
    fake.get_issue = AsyncMock(side_effect=_get_issue)

    if create_issue_results is None:
        create_issue_results = []
    iter_results = iter(create_issue_results)

    async def _create_issue(*, title, body, **kw):
        try:
            return next(iter_results)
        except StopIteration:
            return {"ok": False, "message": "no canned result"}
    fake.create_issue = AsyncMock(side_effect=_create_issue)

    fake.update_issue_body = AsyncMock(return_value={
        "ok": update_issue_body_ok,
        "message": "" if update_issue_body_ok else "patch failed",
        "issue": {"number": 0},
    })

    return fake


# ---------------------------------------------------------------------------
# Helper-level tests (pure functions)
# ---------------------------------------------------------------------------


def test_classify_prompt_includes_criteria_and_issues() -> None:
    prompt = _build_classify_decomposability_prompt(
        issues=[
            {"number": 1, "title": "Big work", "body": "Body 1"},
            {"number": 2, "title": "Tiny bug", "body": "Body 2"},
        ],
        decomposition_criteria="custom criteria text",
    )
    assert "custom criteria text" in prompt
    assert "### #1: Big work" in prompt
    assert "### #2: Tiny bug" in prompt
    assert "Body 1" in prompt
    assert "Body 2" in prompt
    assert "input order" in prompt.lower()


def test_classify_parser_preserves_input_order() -> None:
    raw = (
        '{"classifications": ['
        '{"number": 2, "decomposable": false, "kind": "bug_report", "reason": "B"},'
        '{"number": 1, "decomposable": true, "kind": "too_high_level", "reason": "A"}'
        ']}'
    )
    out = _parse_classify_decomposability(
        raw, expected_numbers=[1, 2],
    )
    assert out is not None
    assert [e["number"] for e in out] == [1, 2]
    assert out[0]["decomposable"] is True
    assert out[1]["decomposable"] is False


def test_classify_parser_fills_missing_entries_as_unclassified() -> None:
    """If the LLM forgets one of the expected numbers, the parser
    returns a placeholder entry instead of silently dropping —
    keeps the per-input alignment for the caller."""

    raw = (
        '{"classifications": ['
        '{"number": 1, "decomposable": true, "kind": "x", "reason": "y"}'
        ']}'
    )
    out = _parse_classify_decomposability(
        raw, expected_numbers=[1, 2],
    )
    assert out is not None
    assert out[0]["number"] == 1 and out[0]["decomposable"] is True
    assert out[1]["number"] == 2 and out[1]["kind"] == "missing"


def test_classify_parser_rejects_malformed_json() -> None:
    assert _parse_classify_decomposability(
        "not json", expected_numbers=[1],
    ) is None
    assert _parse_classify_decomposability(
        '{"not_classifications": []}', expected_numbers=[1],
    ) is None


def test_classify_parser_drops_extra_unknown_numbers() -> None:
    raw = (
        '{"classifications": ['
        '{"number": 1, "decomposable": true, "kind": "x", "reason": "y"},'
        '{"number": 99, "decomposable": true, "kind": "z", "reason": "w"}'
        ']}'
    )
    out = _parse_classify_decomposability(
        raw, expected_numbers=[1],
    )
    assert out is not None
    assert [e["number"] for e in out] == [1]


def test_propose_prompt_includes_max_children_and_parents() -> None:
    prompt = _build_decomposition_prompt(
        parent_issues=[
            {"number": 10, "title": "Auth", "body": "body 10"},
            {"number": 20, "title": "Billing", "body": "body 20"},
        ],
        max_children_per_parent=5,
        decomposition_criteria="custom criteria",
    )
    assert "max_children_per_parent = 5" in prompt
    assert "custom criteria" in prompt
    assert "### Parent #10: Auth" in prompt
    assert "### Parent #20: Billing" in prompt


def test_propose_parser_round_trips_single_parent_with_empty_shared_concerns() -> None:
    raw = (
        '{"parent_proposals": [{'
        '"parent_number": 10, "children": ['
        '{"title": "A", "body": "a"},'
        '{"title": "B", "body": "b"}'
        '], "reason": "splits well"'
        '}],'
        '"shared_concerns": []'
        '}'
    )
    parsed = _parse_decomposition_proposal(
        raw, expected_parents=[10], max_children_per_parent=8,
    )
    assert parsed is not None
    assert len(parsed["parent_proposals"]) == 1
    assert parsed["parent_proposals"][0]["parent_number"] == 10
    assert [c["title"] for c in parsed["parent_proposals"][0]["children"]] == ["A", "B"]
    assert parsed["shared_concerns"] == []


def test_propose_parser_surfaces_shared_concerns_for_multi_parent() -> None:
    raw = (
        '{"parent_proposals": ['
        '{"parent_number": 1, "children": ['
        '{"title": "A1", "body": "a1"}, {"title": "A2", "body": "a2"}'
        '], "reason": "r1"},'
        '{"parent_number": 2, "children": ['
        '{"title": "B1", "body": "b1"}, {"title": "B2", "body": "b2"}'
        '], "reason": "r2"}'
        '],'
        '"shared_concerns": ["common calibration procedure"]'
        '}'
    )
    parsed = _parse_decomposition_proposal(
        raw, expected_parents=[1, 2], max_children_per_parent=8,
    )
    assert parsed is not None
    assert parsed["shared_concerns"] == ["common calibration procedure"]


def test_propose_parser_caps_children_per_parent() -> None:
    children = ",".join(
        f'{{"title": "c{i}", "body": "b{i}"}}' for i in range(20)
    )
    raw = (
        '{"parent_proposals": [{'
        f'"parent_number": 1, "children": [{children}], "reason": "r"'
        '}], "shared_concerns": []}'
    )
    parsed = _parse_decomposition_proposal(
        raw, expected_parents=[1], max_children_per_parent=3,
    )
    assert parsed is not None
    assert len(parsed["parent_proposals"][0]["children"]) == 3


def test_propose_parser_fills_missing_parent_as_placeholder() -> None:
    raw = (
        '{"parent_proposals": [{'
        '"parent_number": 1, "children": ['
        '{"title": "A", "body": "a"}, {"title": "B", "body": "b"}'
        '], "reason": "r"'
        '}], "shared_concerns": []}'
    )
    parsed = _parse_decomposition_proposal(
        raw, expected_parents=[1, 2], max_children_per_parent=8,
    )
    assert parsed is not None
    # Parent 2 is missing → placeholder with empty children.
    assert parsed["parent_proposals"][1]["parent_number"] == 2
    assert parsed["parent_proposals"][1]["children"] == []
    assert "did not return" in parsed["parent_proposals"][1]["reason"].lower()


def test_render_child_body_carries_parent_of_marker() -> None:
    body = _render_child_body(
        parent_number=42,
        parent_title="Big work",
        child_body="acceptance: do X",
    )
    assert "<!-- colony:parent-of: 42" in body
    assert "Tracks #42: Big work" in body
    assert "acceptance: do X" in body


def test_render_parent_body_with_children_replaces_prior_block() -> None:
    first = _render_parent_body_with_children(
        original_body="Parent.",
        child_numbers=[1, 2],
    )
    second = _render_parent_body_with_children(
        original_body=first,
        child_numbers=[3, 4, 5],
    )
    assert second.count("## Sub-issues (decomposed)") == 1
    assert "<!-- colony:decomposed-into: 3,4,5" in second
    assert "<!-- colony:decomposed-into: 1,2" not in second


# ---------------------------------------------------------------------------
# Primitive: classify_issues_decomposability
# ---------------------------------------------------------------------------


async def test_classify_errors_without_github_sibling(
    tmp_path: Path,
) -> None:
    cap = _make_capability(tmp_path)
    result = await cap.classify_issues_decomposability(
        issue_numbers=[1, 2],
    )
    assert result["ok"] is False
    assert result["error"] == "no_github_capability"


async def test_classify_returns_empty_for_empty_input(
    tmp_path: Path,
) -> None:
    github = _make_github_stub()
    cap = _make_capability(tmp_path, github=github)
    result = await cap.classify_issues_decomposability(
        issue_numbers=[],
    )
    assert result == {"ok": True, "classifications": []}


async def test_classify_round_trips_two_issues(tmp_path: Path) -> None:
    github = _make_github_stub(
        issues_by_number={
            1: {"title": "Big work", "body": "Many subtasks"},
            2: {"title": "Tiny bug", "body": "Specific bug"},
        },
    )
    llm_response = (
        '{"classifications": ['
        '{"number": 1, "decomposable": true, "kind": "too_high_level", "reason": "covers many features"},'
        '{"number": 2, "decomposable": false, "kind": "bug_report", "reason": "already narrowly scoped"}'
        ']}'
    )
    cap = _make_capability(
        tmp_path, github=github, llm_response=llm_response,
    )
    result = await cap.classify_issues_decomposability(
        issue_numbers=[1, 2],
    )
    assert result["ok"] is True
    classifications = result["classifications"]
    assert [c["number"] for c in classifications] == [1, 2]
    assert classifications[0]["decomposable"] is True
    assert classifications[1]["decomposable"] is False


async def test_classify_uses_default_criteria_when_none_passed(
    tmp_path: Path,
) -> None:
    """When ``decomposition_criteria=None``, the helper uses the
    DEFAULT_DECOMPOSITION_CRITERIA constant. We check by sniffing
    the prompt the LLM was called with."""

    github = _make_github_stub(
        issues_by_number={1: {"title": "T", "body": "B"}},
    )
    llm_response = (
        '{"classifications": [{"number": 1, "decomposable": true, '
        '"kind": "x", "reason": "y"}]}'
    )
    cap = _make_capability(
        tmp_path, github=github, llm_response=llm_response,
    )
    await cap.classify_issues_decomposability(issue_numbers=[1])
    sent_prompt = cap._agent.infer.call_args.kwargs["prompt"]
    assert DEFAULT_DECOMPOSITION_CRITERIA in sent_prompt


# ---------------------------------------------------------------------------
# Primitive: propose_decompositions
# ---------------------------------------------------------------------------


async def test_propose_errors_without_github_sibling(
    tmp_path: Path,
) -> None:
    cap = _make_capability(tmp_path)
    result = await cap.propose_decompositions(
        parent_issue_numbers=[1],
    )
    assert result["ok"] is False
    assert result["error"] == "no_github_capability"


async def test_propose_round_trips_single_parent(tmp_path: Path) -> None:
    github = _make_github_stub(
        issues_by_number={10: {"title": "Auth", "body": "auth body"}},
    )
    llm_response = (
        '{"parent_proposals": [{"parent_number": 10, "children": ['
        '{"title": "Login", "body": "Implement login"},'
        '{"title": "Logout", "body": "Implement logout"}'
        '], "reason": "splits auth into login + logout"}],'
        '"shared_concerns": []}'
    )
    cap = _make_capability(
        tmp_path, github=github, llm_response=llm_response,
    )
    result = await cap.propose_decompositions(
        parent_issue_numbers=[10],
    )
    assert result["ok"] is True
    proposals = result["parent_proposals"]
    assert len(proposals) == 1
    assert proposals[0]["parent_number"] == 10
    assert proposals[0]["parent_title"] == "Auth"
    assert [c["title"] for c in proposals[0]["children"]] == ["Login", "Logout"]
    assert result["shared_concerns"] == []


async def test_propose_round_trips_joint_decomposition_with_shared_concerns(
    tmp_path: Path,
) -> None:
    github = _make_github_stub(
        issues_by_number={
            1: {"title": "Sensor A", "body": "..."},
            2: {"title": "Sensor B", "body": "..."},
        },
    )
    llm_response = (
        '{"parent_proposals": ['
        '{"parent_number": 1, "children": ['
        '{"title": "A1", "body": "a1"}, {"title": "A2", "body": "a2"}'
        '], "reason": "r1"},'
        '{"parent_number": 2, "children": ['
        '{"title": "B1", "body": "b1"}, {"title": "B2", "body": "b2"}'
        '], "reason": "r2"}'
        '],'
        '"shared_concerns": ["both need a calibration procedure"]}'
    )
    cap = _make_capability(
        tmp_path, github=github, llm_response=llm_response,
    )
    result = await cap.propose_decompositions(
        parent_issue_numbers=[1, 2],
    )
    assert result["ok"] is True
    assert len(result["parent_proposals"]) == 2
    assert result["shared_concerns"] == [
        "both need a calibration procedure",
    ]


# ---------------------------------------------------------------------------
# Primitive: create_decomposition
# ---------------------------------------------------------------------------


async def test_create_errors_without_github_sibling(
    tmp_path: Path,
) -> None:
    cap = _make_capability(tmp_path)
    result = await cap.create_decomposition(
        parent_issue_number=10,
        children=[{"title": "A", "body": "a"}],
    )
    assert result["ok"] is False
    assert result["error"] == "no_github_capability"


async def test_create_errors_on_empty_children(tmp_path: Path) -> None:
    github = _make_github_stub(
        issues_by_number={10: {"title": "T", "body": "B"}},
    )
    cap = _make_capability(tmp_path, github=github)
    result = await cap.create_decomposition(
        parent_issue_number=10, children=[],
    )
    assert result["ok"] is False
    assert result["error"] == "no_children_provided"


async def test_create_dry_run_does_not_mutate(tmp_path: Path) -> None:
    github = _make_github_stub(
        issues_by_number={10: {"title": "T", "body": "B"}},
    )
    cap = _make_capability(tmp_path, github=github)
    result = await cap.create_decomposition(
        parent_issue_number=10,
        children=[
            {"title": "A", "body": "a"},
            {"title": "B", "body": "b"},
        ],
        dry_run=True,
    )
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["parent_number"] == 10
    assert [w["title"] for w in result["would_create"]] == ["A", "B"]
    github.create_issue.assert_not_awaited()
    github.update_issue_body.assert_not_awaited()


async def test_create_applies_and_patches_parent(tmp_path: Path) -> None:
    github = _make_github_stub(
        issues_by_number={10: {"title": "Parent", "body": "old body"}},
        create_issue_results=[
            {"ok": True, "issue": {"number": 11}},
            {"ok": True, "issue": {"number": 12}},
        ],
        update_issue_body_ok=True,
    )
    cap = _make_capability(tmp_path, github=github)
    result = await cap.create_decomposition(
        parent_issue_number=10,
        children=[
            {"title": "A", "body": "a"},
            {"title": "B", "body": "b"},
        ],
        dry_run=False,
    )
    assert result["ok"] is True
    assert result["created_child_numbers"] == [11, 12]
    assert result["parent_patch_ok"] is True
    posted_body = github.update_issue_body.await_args.args[1]
    assert "<!-- colony:decomposed-into: 11,12" in posted_body
    assert "- [ ] #11" in posted_body
    assert "- [ ] #12" in posted_body


async def test_create_surfaces_partial_child_failure(tmp_path: Path) -> None:
    github = _make_github_stub(
        issues_by_number={10: {"title": "Parent", "body": "old body"}},
        create_issue_results=[
            {"ok": True, "issue": {"number": 11}},
            {"ok": False, "message": "API error"},
        ],
    )
    cap = _make_capability(tmp_path, github=github)
    result = await cap.create_decomposition(
        parent_issue_number=10,
        children=[
            {"title": "A", "body": "a"},
            {"title": "B", "body": "b"},
        ],
        dry_run=False,
    )
    # Partial: child A created, B failed. ``ok=False`` overall.
    assert result["ok"] is False
    assert result["created_child_numbers"] == [11]
    assert len(result["child_failures"]) == 1
    assert result["child_failures"][0]["title"] == "B"


# ---------------------------------------------------------------------------
# Integration: a fake agent composes the three primitives end-to-end
# ---------------------------------------------------------------------------


async def test_planner_can_compose_primitives_end_to_end(
    tmp_path: Path,
) -> None:
    """The point of [[primitives-not-pipelines]] is that the planner
    LLM ALONE decides how to call the primitives. This test
    demonstrates one composition: classify all → propose all → apply
    each parent. A different planner could sample, classify
    per-batch, etc. — the action layer doesn't care."""

    github = _make_github_stub(
        issues_by_number={
            1: {"title": "Auth", "body": "many auth tasks"},
            2: {"title": "Billing", "body": "many billing tasks"},
            3: {"title": "Typo fix", "body": "just a typo"},
        },
        create_issue_results=[
            {"ok": True, "issue": {"number": 100}},
            {"ok": True, "issue": {"number": 101}},
            {"ok": True, "issue": {"number": 200}},
            {"ok": True, "issue": {"number": 201}},
        ],
        update_issue_body_ok=True,
    )
    cap = _make_capability(tmp_path, github=github)

    # Fake the LLM to emit classify, then propose, then we'd
    # request_human_approval (out of scope here), then apply.
    fake_classify_resp = MagicMock()
    fake_classify_resp.generated_text = (
        '{"classifications": ['
        '{"number": 1, "decomposable": true, "kind": "too_big_for_one_pr", "reason": "many tasks"},'
        '{"number": 2, "decomposable": true, "kind": "too_big_for_one_pr", "reason": "many tasks"},'
        '{"number": 3, "decomposable": false, "kind": "already_focused", "reason": "single typo"}'
        ']}'
    )
    fake_propose_resp = MagicMock()
    fake_propose_resp.generated_text = (
        '{"parent_proposals": ['
        '{"parent_number": 1, "children": ['
        '{"title": "A1", "body": "a1"}, {"title": "A2", "body": "a2"}'
        '], "reason": "r1"},'
        '{"parent_number": 2, "children": ['
        '{"title": "B1", "body": "b1"}, {"title": "B2", "body": "b2"}'
        '], "reason": "r2"}'
        '],'
        '"shared_concerns": []}'
    )
    cap._agent.infer = AsyncMock(
        side_effect=[fake_classify_resp, fake_propose_resp],
    )

    # 1) Planner classifies all open issues.
    classify_result = await cap.classify_issues_decomposability(
        issue_numbers=[1, 2, 3],
    )
    assert classify_result["ok"] is True
    decomposable_numbers = [
        c["number"]
        for c in classify_result["classifications"]
        if c["decomposable"]
    ]
    assert decomposable_numbers == [1, 2]

    # 2) Planner proposes joint decomposition for the decomposable
    # cohort (1 and 2 are likely related).
    propose_result = await cap.propose_decompositions(
        parent_issue_numbers=decomposable_numbers,
    )
    assert propose_result["ok"] is True
    parent_proposals = propose_result["parent_proposals"]
    assert [p["parent_number"] for p in parent_proposals] == [1, 2]

    # 3) (skipping the approval round here — covered elsewhere)

    # 4) Planner applies each parent's decomposition with
    # ``create_decomposition`` per parent.
    apply_results = []
    for proposal in parent_proposals:
        result = await cap.create_decomposition(
            parent_issue_number=proposal["parent_number"],
            children=proposal["children"],
            dry_run=False,
        )
        apply_results.append(result)
    assert all(r["ok"] for r in apply_results)
    assert apply_results[0]["created_child_numbers"] == [100, 101]
    assert apply_results[1]["created_child_numbers"] == [200, 201]
