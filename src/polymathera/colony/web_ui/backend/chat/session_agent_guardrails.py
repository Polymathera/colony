"""Runtime guardrails mounted on the SessionAgent's action policy.

The SessionAgent's planner is the one most exposed to "narrative
drift" — fabricating status reports from spawn-return shapes,
claiming an apply happened before approval landed. Two hard
guardrails close those drifts deterministically:

- :class:`SemanticConstraintGuardrail` with the
  ``no_unverified_agent_state_claims`` rule — uses an LLM judge
  to evaluate whether a proposed ``respond_to_user`` content makes
  a state CLAIM about another agent that the agent didn't verify
  via ``get_agent_status`` in the SAME cell. Replaces the prior
  syntactic predicate (regex on ``agent-<hex>``) that produced
  false positives on bare references and false negatives on state
  claims that didn't include an id literal. See
  ``colony/MEMORY.md::no-syntactic-proxies-for-semantic-properties``.

- :class:`ApprovalRequiredGuardrail` — gates apply-path calls into
  mission coordinators' mutating actions on a prior
  ``HumanApprovalCapability.approval_granted`` event. ``dry_run=True``
  is always allowed.

Why this lives in its own module: the guardrail instance ships
inside ``SessionAgent.bind(...)``'s ``action_policy_blueprints``,
which travels through cloudpickle to the Ray worker. Module-level
named factories (rather than lambdas captured in closures) keep
the serialised graph small and easy to audit; tests import these
shapes too.

See ``colony/mission_and_action_guardrails_plan.md`` (Part 2) for
the broader hybrid-enforcement design,
``colony/docs/guides/action-policy-dimensions.md`` for the
operator-facing how-to, and ``semantic_constraints.py`` for the
general SemanticConstraint primitives this rule was migrated to.
"""

from __future__ import annotations

from polymathera.colony.agents.patterns.actions.code_constraints import (
    ApprovalRequiredGuardrail,
    CompositeGuardrail,
    RuntimeGuardrail,
)
from polymathera.colony.agents.patterns.actions.semantic_constraints import (
    ConstraintFailureMode,
    ConstraintScope,
    LLMJudgeVerifier,
    SemanticConstraint,
    SemanticConstraintGuardrail,
)


# ---------------------------------------------------------------------------
# Constraint catalogue
# ---------------------------------------------------------------------------


_NO_UNVERIFIED_AGENT_STATE_CLAIMS_RULE_NL = (
    "When the proposed action is respond_to_user, the content MUST "
    "NOT make any claim or imply anything about the lifecycle state "
    "of another agent (e.g. 'running', 'stopped', 'idle', 'failed', "
    "'finished', 'busy', 'has completed', 'is currently …') UNLESS "
    "the agent has called AgentPoolCapability.get_agent_status for "
    "that exact agent_id WITHIN THIS CELL and observed the state. "
    "The owner agent's own agent_id is exempt — claims about the owner agent's "
    "own state don't need verification. BARE references to another "
    "agent's id without a state claim (e.g. 'I asked agent-XYZ to "
    "help', 'the coordinator agent-XYZ was created') are allowed. "
    "Only STATE CLAIMS about OTHER agents require evidence in the "
    "form of a successful get_agent_status call this cell."
)


def _build_no_unverified_state_claims_constraint() -> SemanticConstraint:
    """The first migrated declarative constraint. Replaces the prior
    syntactic ``status_claim_rule`` that fired on ANY ``agent-<hex>``
    mention in respond_to_user content — including bare references
    that weren't state claims, and missing real state claims that
    happened to refer to agents indirectly (e.g. by capability name
    or by role).

    The LLM judge reads the proposed content + the cell's call
    history (including any prior get_agent_status calls + their
    results) and decides whether the rule is satisfied. Failure
    mode is BLOCK — state-claim fabrication is irreversible from
    the user's perspective once the message ships.
    """

    return SemanticConstraint(
        id="no_unverified_agent_state_claims",
        rule_nl=_NO_UNVERIFIED_AGENT_STATE_CLAIMS_RULE_NL,
        applies_to=["respond_to_user"],
        scope=ConstraintScope.CELL,
        failure_mode=ConstraintFailureMode.BLOCK,
        verifier=LLMJudgeVerifier(
            max_tokens=300,
            temperature=0.0,
        ),
    )


def session_agent_semantic_constraints() -> list[SemanticConstraint]:
    """The catalogue. Each entry is one declarative rule the
    framework checks at runtime AND surfaces in the planner prompt's
    "## Active semantic constraints" section.

    Operators add new rules by appending to this list. No Python
    class per rule, no test scaffolding per rule, no signature
    changes — the constraint is data."""

    return [
        _build_no_unverified_state_claims_constraint(),
    ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_session_agent_runtime_guardrail(
    *,
    approval_required_action_prefixes: list[str] | None = None,
) -> RuntimeGuardrail:
    """Build the composite guardrail the SessionAgent mounts on its
    code-generation action policy.

    Args:
        approval_required_action_prefixes: Action-key prefixes that
            require a prior ``HumanApprovalCapability.approval_granted``
            event before the apply call can dispatch. Defaults to the
            mutating actions on ``DesignProcessCapability`` (the
            project-planning mission's apply surfaces). Pass ``[]`` to
            disable the approval gate entirely.

    Returns:
        A :class:`CompositeGuardrail` running the semantic constraint
        catalogue and the approval gate in that order. Order matters:
        the semantic constraints are cheaper to evaluate for the
        common (non-applicable) action_key and are the more frequent
        check, so they front. Approval is the second line for the
        narrower set of mutating apply actions.

    The semantic catalogue lives in :func:`session_agent_semantic_constraints`;
    LLM-judge verifiers get their ``infer_fn`` bound lazily via
    :meth:`SemanticConstraintGuardrail.bind_agent` (the agent doesn't
    exist at factory time — the blueprint cloudpickles into the worker).
    """

    if approval_required_action_prefixes is None:
        approval_required_action_prefixes = [
            "DesignProcessCapability.bootstrap_roadmap_from_objectives",
            "DesignProcessCapability.sync_roadmap_with_github",
            "DesignProcessCapability.propose_task_assignments",
        ]

    return CompositeGuardrail(
        SemanticConstraintGuardrail(
            constraints=session_agent_semantic_constraints(),
        ),
        ApprovalRequiredGuardrail(
            approval_required_action_prefixes=(
                approval_required_action_prefixes
            ),
        ),
    )


__all__ = (
    "build_session_agent_runtime_guardrail",
    "session_agent_semantic_constraints",
)
