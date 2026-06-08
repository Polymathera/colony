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
# round-trips by reference (the closures below capture the name, not
# the compiled object — cloudpickle handles that cleanly).
_AGENT_ID_RE = re.compile(r"agent-[0-9a-f]+")


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

    The status-claim predicate is built as a closure over a
    one-element ``_speaker_ref`` dict so the SessionAgent's own
    ``agent-<hex>`` id can be excluded from the "you referenced this
    agent without status-checking it" check. The agent_id is not
    known at build time (the agent is spawned later, gets its id
    allocated at spawn); the code-generation policy calls
    :meth:`RuntimeGuardrail.bind_speaker` after init, which writes the
    id into ``_speaker_ref``. Per ``decompose_and_session_recovery_fixes_plan.md``
    item 4: the live 2026-06-07 run blocked a ``respond_to_user``
    that mentioned the SessionAgent's OWN id because the prior
    ``get_agent_status`` only queried the coordinator's id; this
    closure fixes that without touching every guardrail's check
    signature.
    """

    if approval_required_action_prefixes is None:
        approval_required_action_prefixes = [
            "DesignProcessCapability.bootstrap_roadmap_from_objectives",
            "DesignProcessCapability.sync_roadmap_with_github",
            "DesignProcessCapability.propose_task_assignments",
        ]

    speaker_ref: dict[str, str | None] = {"agent_id": None}

    def _referenced_non_self_agent_ids(content: Any) -> set[str]:
        """Extract every ``agent-<hex>`` mention in ``content`` minus
        the speaker's own id (when known). Used by both ``applies_when``
        and ``prior_params_match`` so the rule's gate AND its match
        logic share the same "what counts as a non-self reference"
        definition."""

        if not isinstance(content, str):
            return set()
        referenced = set(_AGENT_ID_RE.findall(content))
        speaker = speaker_ref["agent_id"]
        if speaker:
            referenced.discard(speaker)
        return referenced

    def _content_mentions_non_self_agent_id(
        params: dict[str, Any],
    ) -> bool:
        """The rule's ``applies_when`` gate: fires only when the
        proposed ``respond_to_user`` content references at least one
        agent_id that ISN'T the speaker's own. A response that only
        mentions the speaker (or none at all) bypasses the
        status-check requirement entirely. Per item 4: the 2026-06-07
        live run blocked a response that mentioned only the
        SessionAgent's own id because the predicate didn't know who
        the speaker was."""

        return bool(_referenced_non_self_agent_ids(params.get("content")))

    def _prior_get_status_covers_target_agents(
        prior_params: dict[str, Any],
        target_params: dict[str, Any],
    ) -> bool:
        """Return True when the prior ``get_agent_status`` call's
        ``agent_ids`` cover every NON-SELF ``agent-<hex>`` mention
        in the target ``respond_to_user`` content. Reuses the same
        non-self extraction as ``applies_when`` for symmetry."""

        referenced = _referenced_non_self_agent_ids(
            target_params.get("content"),
        )
        if not referenced:
            # ``applies_when`` should already have short-circuited;
            # belt-and-braces if this is reached anyway.
            return True
        queried = set(prior_params.get("agent_ids") or [])
        return referenced.issubset(queried)

    status_claim_rule = ArgsAwareOrderingRule(
        target_action="respond_to_user",
        applies_when=_content_mentions_non_self_agent_id,
        required_prior="get_agent_status",
        max_age_calls=20,
        prior_params_match=_prior_get_status_covers_target_agents,
        suggestion=(
            "Call AgentPoolCapability.get_agent_status with the "
            "agent_id(s) you're about to reference, then respond. "
            "The spawn_mission return only tells you whether the "
            "coordinator was created — NOT whether it is running. "
            "References to your OWN agent_id don't need a status "
            "check. See docs/guides/action-policy-dimensions.md."
        ),
    )

    class _SessionAgentSpeakerAwareGuardrail(CompositeGuardrail):
        """Plain composite + a ``bind_speaker`` that writes into the
        predicate's closure mailbox."""

        def bind_speaker(self, agent):
            super().bind_speaker(agent)
            speaker_ref["agent_id"] = (
                agent.agent_id if agent is not None else None
            )

    return _SessionAgentSpeakerAwareGuardrail(
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
