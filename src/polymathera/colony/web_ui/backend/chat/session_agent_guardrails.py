"""Runtime guardrails mounted on the SessionAgent's action policy.

The SessionAgent's planner is the one most exposed to "narrative
drift" — fabricating status reports from spawn-return shapes,
claiming an apply happened before approval landed. Two hard
guardrails close those drifts deterministically:

- :class:`ArgsAwareTemporalOrderGuardrail` — gates ``respond_to_user``
  calls whose ``content`` mentions an ``agent-<hex>`` id on a
  recent matching :meth:`AgentPoolCapability.get_agent_status` for
  THAT specific agent. Bare "Hi there!" responses pass through.

- :class:`ApprovalRequiredGuardrail` — gates apply-path calls into
  mission coordinators' mutating actions on a prior
  ``HumanApprovalCapability.approval_granted`` event. ``dry_run=True``
  is always allowed.

Why this lives in its own module: the guardrail instance ships
inside ``SessionAgent.bind(...)``'s ``action_policy_blueprints``,
which travels through cloudpickle to the Ray worker. Module-level
named functions (rather than lambdas captured in closures) keep
the serialised graph small and easy to audit; tests import these
shapes too.

See ``colony/mission_and_action_guardrails_plan.md`` (Part 2) for
the broader hybrid-enforcement design, and
``colony/docs/guides/action-policy-dimensions.md`` for the
operator-facing how-to.
"""

from __future__ import annotations

import re
from typing import Any

from polymathera.colony.agents.patterns.actions.code_constraints import (
    ApprovalRequiredGuardrail,
    ArgsAwareOrderingRule,
    ArgsAwareTemporalOrderGuardrail,
    CompositeGuardrail,
    RuntimeGuardrail,
)


# Module-level so the regex compiles once and survives cloudpickle
# round-trips by reference (the lambdas below capture the name, not
# the compiled object — cloudpickle handles that cleanly).
_AGENT_ID_RE = re.compile(r"agent-[0-9a-f]+")


# ---------------------------------------------------------------------------
# Named predicates — preferred over lambdas inside the guardrail
# instance for cloudpickle stability + readable error messages.
# ---------------------------------------------------------------------------


def _content_mentions_agent_id(params: dict[str, Any]) -> bool:
    """Return True when the proposed ``respond_to_user`` content
    references an ``agent-<hex>`` identifier. The status-claim gate
    only fires for these calls; free-form responses pass through."""

    content = params.get("content")
    if not isinstance(content, str):
        return False
    return bool(_AGENT_ID_RE.search(content))


def _prior_get_status_covers_target_agents(
    prior_params: dict[str, Any],
    target_params: dict[str, Any],
) -> bool:
    """Return True when the prior ``get_agent_status`` call's
    ``agent_ids`` cover every ``agent-<hex>`` mention in the target
    ``respond_to_user`` content. Same-agent matching prevents
    "called get_agent_status for agent-X, claimed state for
    agent-Y" loopholes."""

    content = target_params.get("content")
    if not isinstance(content, str):
        return False
    referenced = set(_AGENT_ID_RE.findall(content))
    if not referenced:
        return False
    queried = set(prior_params.get("agent_ids") or [])
    return referenced.issubset(queried)


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
        A :class:`CompositeGuardrail` running the status-claim and
        approval gates in that order. The order matters: status-claim
        is the cheaper check and the more frequent failure mode in
        traces, so it fronts. Approval is the second line.
    """

    if approval_required_action_prefixes is None:
        approval_required_action_prefixes = [
            "DesignProcessCapability.bootstrap_roadmap_from_objectives",
            "DesignProcessCapability.sync_roadmap_with_github",
            "DesignProcessCapability.propose_task_assignments",
        ]

    status_claim_rule = ArgsAwareOrderingRule(
        target_action="respond_to_user",
        applies_when=_content_mentions_agent_id,
        required_prior="get_agent_status",
        max_age_calls=20,
        prior_params_match=_prior_get_status_covers_target_agents,
        suggestion=(
            "Call AgentPoolCapability.get_agent_status with the "
            "agent_id(s) you're about to reference, then respond. "
            "The spawn_mission return only tells you whether the "
            "coordinator was created — NOT whether it is running. "
            "See docs/guides/action-policy-dimensions.md."
        ),
    )

    return CompositeGuardrail(
        ArgsAwareTemporalOrderGuardrail(rules=[status_claim_rule]),
        ApprovalRequiredGuardrail(
            approval_required_action_prefixes=(
                approval_required_action_prefixes
            ),
        ),
    )


__all__ = (
    "build_session_agent_runtime_guardrail",
)
