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
