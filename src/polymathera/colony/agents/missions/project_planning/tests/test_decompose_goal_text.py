"""Snapshot pin on the ``project_planning`` decompose-mode goal text.

Per ``decompose_one_and_done_and_spinner_plan.md`` Change 3 the prior
"Suggested flow (1)...(7)" recipe was deleted in favour of a
contract-shaped paragraph that names the scope contract, the primitive
menu, and the structural termination — but prescribes NO order. This
regression guard pins the absence of the rejected phrasings so a
future contributor who tries to re-introduce a recipe trips on this
test, not on a customer.
"""

from __future__ import annotations


_FORBIDDEN_PHRASES = (
    "Suggested flow",
    "step (1)",
    "step (2)",
    "step (3)",
    "step (4)",
    "step (5)",
    "step (6)",
    "step (7)",
    "loop back",
    "deferred by the user",
)


_REQUIRED_PHRASES = (
    # Scope contract
    "issue_numbers",
    "max_parents_per_run",
    # Termination contract
    "create_decomposition with dry_run=False",
    "classify_issues_decomposability returning decomposable=False",
    "request_decompose_early_stop",
    "validator will not accept signal_completion",
    # Primitive menu
    "list_issues",
    "propose_decompositions",
    "request_human_approval",
    "respond_to_user",
)


def _decompose_goal_text() -> str:
    from polymathera.colony.agents.configs import _BUILTIN_MISSIONS

    goals = _BUILTIN_MISSIONS["project_planning"]["self_concept"]["goals"]
    # The decompose-mode goal is the bullet that BEGINS with
    # "For decompose mode" — there is exactly one such goal.
    for goal in goals:
        if isinstance(goal, str) and goal.startswith("For decompose mode"):
            return goal
    raise AssertionError(
        "No 'For decompose mode' goal found in project_planning.self_concept.goals"
    )


def test_decompose_goal_text_drops_numbered_recipe() -> None:
    """The decompose-mode goal MUST NOT contain the rejected
    numbered-flow phrasing. Each forbidden phrase is the load-bearing
    sentence the primitives review correctly flagged as the recipe
    bias."""

    text = _decompose_goal_text()
    for phrase in _FORBIDDEN_PHRASES:
        assert phrase not in text, (
            f"Decompose-mode goal contains forbidden phrase {phrase!r} — "
            "the recipe creeping back in. See "
            "decompose_one_and_done_and_spinner_plan.md Change 3."
        )


def test_decompose_goal_text_states_scope_contract() -> None:
    """The replacement contract names scope, termination, and the
    primitive menu. A future surgical edit that drops one of these
    fails here loudly."""

    text = _decompose_goal_text()
    for phrase in _REQUIRED_PHRASES:
        assert phrase in text, (
            f"Decompose-mode goal is missing required phrase {phrase!r} — "
            "the contract-shaped paragraph from "
            "decompose_one_and_done_and_spinner_plan.md Change 3 has "
            "drifted."
        )


def test_decompose_goal_text_does_not_use_arrow_sequence_markers() -> None:
    """Numbered or arrow-marked sequences are the recipe shape we
    rejected. Pin absence of the obvious ordering markers (any future
    contributor reaching for ``->`` / ``→`` / ``Step 1`` is now caught
    at test time)."""

    text = _decompose_goal_text()
    for marker in ("->", "→", "Step 1", "Step 2"):
        assert marker not in text, (
            f"Decompose-mode goal contains ordering marker {marker!r} — "
            "the contract shape is a flat menu, not a sequence."
        )


# ---------------------------------------------------------------------------
# Approval-wait bullet (Bucket A.2 / Fix F3 prevention) — the bullet
# teaches the request_human_approval → wait_for_next_event → read
# planner-context binding sequence. The F3 forensic failure was the
# coordinator polling get_response 25× after state=ready already
# returned; the prevention is the actionable "advance after reading"
# contract pinned below.
# ---------------------------------------------------------------------------


def _approval_wait_goal_text() -> str:
    from polymathera.colony.agents.configs import _BUILTIN_MISSIONS

    goals = _BUILTIN_MISSIONS["project_planning"]["self_concept"]["goals"]
    for goal in goals:
        if isinstance(goal, str) and goal.startswith(
            "Post one HumanApprovalRequest",
        ):
            return goal
    raise AssertionError(
        "No 'Post one HumanApprovalRequest' goal found in "
        "project_planning.self_concept.goals — Bucket A.2's "
        "regression target has moved."
    )


_APPROVAL_WAIT_REQUIRED_PHRASES = (
    # Read-once-then-terminal contract: the planner-context binding is
    # the durable record; get_response is on-demand lookup, not a wake
    # surface that benefits from polling.
    "treat it as terminal",
    "planner-context binding persists across iterations",
    "burning an iteration",
    # Positive instruction: name the next action explicitly so the LLM
    # has somewhere to go after reading choice.
    "Your NEXT action MUST be the per-choice branch",
    # Negative instruction in the same sentence as the positive:
    # bullet N+1 uses choice, bullet N must produce it AND signal
    # "advance now". Per [[llm-prompts-must-be-actionable]].
    "do NOT call get_response for the same request_id again",
)


def test_approval_wait_goal_text_states_advance_after_choice_contract() -> None:
    """The F3 prevention contract: after reading ``response.choice``
    once for a request_id, the LLM treats the answer as terminal and
    advances to the per-choice branch. Re-polling ``get_response`` is
    explicitly forbidden — that's the precondition for the F3 forensic
    25-iteration poll-loop. Pairs with the shipped
    ``ApprovalAdvanceReflector`` (Slice E) which provides the recovery
    side when the LLM ignores this prevention."""

    text = _approval_wait_goal_text()
    for phrase in _APPROVAL_WAIT_REQUIRED_PHRASES:
        assert phrase in text, (
            f"Approval-wait goal is missing required phrase {phrase!r} — "
            "the F3 prevention contract (Bucket A.2 of "
            "agent_error_detect_prevent_recover_architecture_plan.md) "
            "has drifted."
        )
