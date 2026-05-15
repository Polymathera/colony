"""Tests for PR 3 — operator-approval gate on protected branches.

Three groups:

- **Gate plumbing on `DesignCheckpointer`** — non-protected branches
  run inline; protected branches post an approval + pending-op record
  and return `ProtectedOpResult(status="pending_approval")`.
- **Response handler** — operator's `approve` / `reject` drives the
  pending op through, writes a `ProtectedOpOutcome`, surfaces planner
  context.
- **L1-F refusal** — `ProjectAuthoringCapability` refuses outright on
  a protected branch (no gating; agent must branch off and merge back).

The Web UI relay (chat → operator click → HTTP POST → blackboard
write) is not exercised here. We write the response payload directly
to the blackboard and drive the event handler manually — same shape as
``test_human_approval.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.agents.blackboard import EnhancedBlackboard
from polymathera.colony.agents.blackboard.protocol import (
    DesignMonorepoEventProtocol,
    HumanApprovalProtocol,
)
from polymathera.colony.agents.patterns.capabilities.human_approval import (
    HumanApprovalResponse,
)
from polymathera.colony.design_monorepo import (
    AgentIdentity,
    DesignCheckpointer,
    DesignMonorepoClient,
    DesignMonorepoError,
    DesignMonorepoManifest,
    ProjectAuthoringCapability,
    bootstrap_design_monorepo,
)
from polymathera.colony.design_monorepo.models import (
    PendingProtectedOp,
    ProtectedOpOutcome,
)
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _exec_ctx():
    """Session-scoped execution context. Required for the human-approval
    topic the gate writes to."""

    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


@pytest.fixture
def protected_manifest() -> DesignMonorepoManifest:
    """Override the conftest's default `manifest` to declare ``main``
    as protected — PR 3 gating tests need this on."""

    return DesignMonorepoManifest(
        tenant="acme",
        colony="acme-colony",
        program="prog-test",
        target_system="test_system",
        topology="external",
        design_repo_url="file:///tmp/never-cloned-from",
        protected_branches=("main", "release/*"),
    )


@pytest.fixture
def protected_repo(
    protected_manifest: DesignMonorepoManifest,
    identity: AgentIdentity,
    fresh_repo_dir: Path,
) -> DesignMonorepoClient:
    """A bootstrapped repo whose manifest declares ``main`` protected."""

    return bootstrap_design_monorepo(
        protected_manifest, fresh_repo_dir, identity=identity,
    )


async def _checkpointer(
    repo: DesignMonorepoClient, _exec_ctx,
) -> DesignCheckpointer:
    """Build a detached checkpointer bound to ``repo`` with an in-memory
    session blackboard wired in. Pattern mirrors
    ``test_human_approval.py``."""

    cap = DesignCheckpointer(
        agent=None, scope_id="test", working_dir=repo.working_dir,
    )
    cap._client = repo
    bb = EnhancedBlackboard(
        app_name="test_app",
        scope_id=cap.scope_id,
        backend_type="memory",
        enable_events=True,
    )
    await bb.initialize()
    cap._blackboard = bb
    return cap


async def _authoring(
    repo: DesignMonorepoClient, _exec_ctx,
) -> ProjectAuthoringCapability:
    cap = ProjectAuthoringCapability(
        agent=None, scope_id="test", working_dir=repo.working_dir,
    )
    cap._client = repo
    bb = EnhancedBlackboard(
        app_name="test_app",
        scope_id=cap.scope_id,
        backend_type="memory",
        enable_events=True,
    )
    await bb.initialize()
    cap._blackboard = bb
    return cap


# ---------------------------------------------------------------------------
# Manifest helper
# ---------------------------------------------------------------------------


def test_is_branch_protected_glob_match() -> None:
    m = DesignMonorepoManifest(
        tenant="t", colony="c", program="p", target_system="ts",
        design_repo_url="file:///x",
        protected_branches=("main", "release/*"),
    )
    assert m.is_branch_protected("main")
    assert m.is_branch_protected("release/2.0")
    assert not m.is_branch_protected("wip/feature")
    assert not m.is_branch_protected("")  # detached HEAD


# ---------------------------------------------------------------------------
# Inline (non-protected) path
# ---------------------------------------------------------------------------


async def test_commit_state_on_non_protected_branch_runs_inline(
    protected_repo: DesignMonorepoClient, _exec_ctx,
) -> None:
    """A commit on a non-protected branch must execute inline and
    return ``status="executed"``, even when the manifest declares
    protected branches."""

    cap = await _checkpointer(protected_repo, _exec_ctx)
    protected_repo.repo.git.checkout("-b", "wip/x")
    (protected_repo.working_dir / "scratch.txt").write_text(
        "hello", encoding="utf-8",
    )
    result = await cap.commit_state(
        "scratch", paths=["scratch.txt"], all_changes=False,
    )
    assert result.status == "executed"
    assert result.op_kind == "commit_state"
    assert result.target_branch == "wip/x"
    assert result.sha  # SHA is non-empty


async def test_pull_remote_ff_only_bypasses_gate_on_protected(
    protected_repo: DesignMonorepoClient, _exec_ctx,
) -> None:
    """``ff_only`` cannot rewrite or merge any existing commit — it is
    never gated, even on a protected branch. (No remote is configured,
    so the underlying pull raises; we only verify the gate decision
    by catching the failure shape — a gated call would have returned
    ``pending_approval`` without raising.)"""

    cap = await _checkpointer(protected_repo, _exec_ctx)
    with pytest.raises(DesignMonorepoError, match="Pull from"):
        await cap.pull_remote(strategy="ff_only")


# ---------------------------------------------------------------------------
# Gated path — commit_state on protected branch
# ---------------------------------------------------------------------------


async def test_commit_state_on_protected_branch_returns_pending(
    protected_repo: DesignMonorepoClient, _exec_ctx,
) -> None:
    cap = await _checkpointer(protected_repo, _exec_ctx)
    (protected_repo.working_dir / "design" / "added.txt").parent.mkdir(
        parents=True, exist_ok=True,
    )
    (protected_repo.working_dir / "design" / "added.txt").write_text(
        "x", encoding="utf-8",
    )
    result = await cap.commit_state(
        "trying a commit on main",
        paths=["design/added.txt"],
    )
    assert result.status == "pending_approval"
    assert result.op_kind == "commit_state"
    assert result.target_branch == "main"
    assert result.request_id.startswith("appr_")

    bb = await cap.get_blackboard()
    pending_raw = await bb.read(
        DesignMonorepoEventProtocol.protected_op_pending_key(result.request_id),
    )
    assert isinstance(pending_raw, dict)
    pending = PendingProtectedOp.model_validate(pending_raw)
    assert pending.op_kind == "commit_state"
    assert pending.target_branch == "main"
    assert pending.args["message"] == "trying a commit on main"

    request_raw = await bb.read(
        HumanApprovalProtocol.request_key(result.request_id),
    )
    assert isinstance(request_raw, dict)
    assert request_raw["options"] == ["approve", "reject"]


async def test_approval_executes_pending_commit(
    protected_repo: DesignMonorepoClient, _exec_ctx,
) -> None:
    cap = await _checkpointer(protected_repo, _exec_ctx)
    target_path = protected_repo.working_dir / "design" / "added.txt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("x", encoding="utf-8")
    pending_result = await cap.commit_state(
        "commit attempt", paths=["design/added.txt"],
    )
    head_before = protected_repo.repo.head.commit.hexsha

    response = HumanApprovalResponse(
        request_id=pending_result.request_id,
        choice="approve", decided_by="alice",
    )
    bb = await cap.get_blackboard()
    await bb.write(
        HumanApprovalProtocol.response_key(pending_result.request_id),
        response.model_dump(mode="json"),
    )
    fake_event = type("E", (), {})()
    fake_event.key = HumanApprovalProtocol.response_key(
        pending_result.request_id,
    )
    fake_event.value = response.model_dump(mode="json")
    result = await cap._on_protected_approval_response(fake_event, None)

    assert result is not None
    outcome_raw = await bb.read(
        DesignMonorepoEventProtocol.protected_op_outcome_key(
            pending_result.request_id,
        ),
    )
    outcome = ProtectedOpOutcome.model_validate(outcome_raw)
    assert outcome.status == "executed"
    assert outcome.op_kind == "commit_state"
    assert outcome.decided_by == "alice"
    # Commit landed: HEAD moved.
    head_after = protected_repo.repo.head.commit.hexsha
    assert head_after != head_before
    assert outcome.sha == head_after
    # Pending marker cleaned up.
    cleared = await bb.read(
        DesignMonorepoEventProtocol.protected_op_pending_key(
            pending_result.request_id,
        ),
    )
    assert cleared is None


async def test_rejection_records_outcome_and_skips_commit(
    protected_repo: DesignMonorepoClient, _exec_ctx,
) -> None:
    cap = await _checkpointer(protected_repo, _exec_ctx)
    target = protected_repo.working_dir / "design" / "added.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x", encoding="utf-8")
    pending = await cap.commit_state(
        "should be rejected", paths=["design/added.txt"],
    )
    head_before = protected_repo.repo.head.commit.hexsha

    response = HumanApprovalResponse(
        request_id=pending.request_id, choice="reject",
        decided_by="alice", note="not approved",
    )
    bb = await cap.get_blackboard()
    await bb.write(
        HumanApprovalProtocol.response_key(pending.request_id),
        response.model_dump(mode="json"),
    )
    fake_event = type("E", (), {})()
    fake_event.key = HumanApprovalProtocol.response_key(pending.request_id)
    fake_event.value = response.model_dump(mode="json")
    await cap._on_protected_approval_response(fake_event, None)

    outcome_raw = await bb.read(
        DesignMonorepoEventProtocol.protected_op_outcome_key(
            pending.request_id,
        ),
    )
    outcome = ProtectedOpOutcome.model_validate(outcome_raw)
    assert outcome.status == "rejected"
    assert outcome.error == "not approved"
    # HEAD unchanged — no commit.
    assert protected_repo.repo.head.commit.hexsha == head_before


async def test_response_ignores_pending_from_another_capability(
    protected_repo: DesignMonorepoClient, _exec_ctx,
) -> None:
    """A response targeting a pending op owned by a different
    capability scope_id must be silently ignored — multi-agent
    sessions share the human-approval topic."""

    cap = await _checkpointer(protected_repo, _exec_ctx)
    target = protected_repo.working_dir / "design" / "added.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x", encoding="utf-8")
    pending = await cap.commit_state(
        "main commit", paths=["design/added.txt"],
    )

    # Forge a pending record owned by some OTHER capability instance.
    bb = await cap.get_blackboard()
    forged = PendingProtectedOp(
        request_id="appr_other_owner",
        op_kind="commit_state",
        target_branch="main",
        args={"message": "owned by a different cap"},
        summary="not mine",
        requester_capability_scope_id="some_other_scope",
    )
    await bb.write(
        DesignMonorepoEventProtocol.protected_op_pending_key(
            "appr_other_owner",
        ),
        forged.model_dump(mode="json"),
    )

    response = HumanApprovalResponse(
        request_id="appr_other_owner", choice="approve",
    )
    fake_event = type("E", (), {})()
    fake_event.key = HumanApprovalProtocol.response_key("appr_other_owner")
    fake_event.value = response.model_dump(mode="json")
    result = await cap._on_protected_approval_response(fake_event, None)

    assert result is None  # ignored — not ours
    # And the cap-owned pending op is still there (untouched).
    owned_raw = await bb.read(
        DesignMonorepoEventProtocol.protected_op_pending_key(
            pending.request_id,
        ),
    )
    assert owned_raw is not None


# ---------------------------------------------------------------------------
# Other gated ops
# ---------------------------------------------------------------------------


async def test_merge_design_into_protected_branch_returns_pending(
    protected_repo: DesignMonorepoClient, _exec_ctx,
) -> None:
    cap = await _checkpointer(protected_repo, _exec_ctx)
    # Set up a branch with a commit to merge.
    protected_repo.repo.git.checkout("-b", "wip/contrib")
    (protected_repo.working_dir / "x.txt").write_text("x", encoding="utf-8")
    protected_repo.commit_with_identity(
        AgentIdentity(agent_id="c", role="c", colony_id="c"),
        "wip commit", all_changes=True,
    )
    protected_repo.repo.git.checkout("main")
    result = await cap.merge_design("wip/contrib")
    assert result.status == "pending_approval"
    assert result.op_kind == "merge_design"
    assert result.target_branch == "main"


async def test_push_remote_on_protected_returns_pending(
    protected_repo: DesignMonorepoClient, _exec_ctx,
) -> None:
    cap = await _checkpointer(protected_repo, _exec_ctx)
    result = await cap.push_remote()
    assert result.status == "pending_approval"
    assert result.op_kind == "push_remote"
    assert result.target_branch == "main"


async def test_pull_remote_merge_strategy_on_protected_returns_pending(
    protected_repo: DesignMonorepoClient, _exec_ctx,
) -> None:
    cap = await _checkpointer(protected_repo, _exec_ctx)
    result = await cap.pull_remote(strategy="merge")
    assert result.status == "pending_approval"
    assert result.op_kind == "pull_remote"


async def test_rebase_onto_on_protected_returns_pending(
    protected_repo: DesignMonorepoClient, _exec_ctx,
) -> None:
    cap = await _checkpointer(protected_repo, _exec_ctx)
    result = await cap.rebase_onto("HEAD")
    assert result.status == "pending_approval"
    assert result.op_kind == "rebase_onto"
    assert result.target_branch == "main"


# ---------------------------------------------------------------------------
# L1-F refusal on protected branches
# ---------------------------------------------------------------------------


async def test_l1f_write_file_refused_on_protected_branch(
    protected_repo: DesignMonorepoClient, _exec_ctx,
) -> None:
    cap = await _authoring(protected_repo, _exec_ctx)
    with pytest.raises(DesignMonorepoError, match="protected branch"):
        await cap.write_file("src/x.py", "x = 1\n")


async def test_l1f_write_file_runs_on_non_protected_branch(
    protected_repo: DesignMonorepoClient, _exec_ctx,
) -> None:
    cap = await _authoring(protected_repo, _exec_ctx)
    protected_repo.repo.git.checkout("-b", "wip/scaffold")
    payload = await cap.write_file("src/x.py", "x = 1\n")
    assert payload.action_kind == "write_file"
    assert "src/x.py" in payload.affected_paths
