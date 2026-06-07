"""Coordinator for the ``project_planning`` mission.

The coordinator is an :class:`Agent` whose planner LLM drives the
full mission flow over the action surfaces shipped in P5:

1. Pick the right action for ``mission_params['mode']``:

   - ``bootstrap`` → :meth:`DesignProcessCapability.bootstrap_roadmap_from_objectives`
   - ``refresh``   → :meth:`DesignProcessCapability.sync_roadmap_with_github`
   - ``assignments`` → :meth:`DesignProcessCapability.propose_task_assignments`

2. Call it with ``dry_run=True``, render the proposal, and
   :meth:`HumanApprovalCapability.request_human_approval` with the
   proposal as ``extra``.
3. Wait for the user's response (the cap's event handler surfaces it
   as fresh planner context).
4. On ``approve``: re-call the action with ``dry_run=False``.
   On ``reject``: stop without writing.
5. Report the final result.

The coordinator has no bespoke state machine — every step is a
single planner action call. The self-concept guides the planner
through the sequence; the constraints enforce the "never apply
without approval" invariant.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from polymathera.colony.agents.base import Agent
from polymathera.colony.agents.configs import (
    MissionConcurrencyScope,
    MissionExecutionPolicy,
)
from polymathera.colony.agents.patterns.capabilities.github import (
    GitHubCapability,
)
from polymathera.colony.agents.patterns.capabilities.human_approval import (
    HumanApprovalCapability,
)
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.design_monorepo import (
    DesignProcessCapability,
    SystemDesignCapability,
)
from polymathera.colony.design_monorepo.blueprints import (
    design_monorepo_capability_blueprints,
)


logger = logging.getLogger(__name__)


class ProjectPlanningCoordinator(Agent):
    """LLM-planner-driven coordinator for the ``project_planning`` mission.

    Mounts the action surfaces the mission needs and lets the planner
    walk the propose → approve → apply flow. No ``run()`` method —
    fully event-driven, same shape as :class:`IntentInferenceCoordinator`
    and the other capability-only coordinators.
    """

    # Declarative spawn-gate policy (see
    # ``colony/mission_and_action_guardrails_plan.md`` Part 1). Read
    # by ``resolve_mission_execution_policy`` at spawn time.
    #
    # - One instance per SESSION (not per AGENT): if the SessionAgent's
    #   planner emits three sequential ``spawn_mission`` calls for
    #   bootstrap / refresh / assignments, only the first lands.
    # - ``chains_with_modes`` tells the gate "these three modes are
    #   parameters of ONE coordinator, not three sibling coordinators"
    # - ``return_existing`` means the second + third spawn calls get
    #   the running coordinator's ``agent_id`` back so the LLM's chain
    #   converges on a single agent instead of erroring out.
    # - ``preemptible=False`` because a mid-apply preemption would
    #   leave the GitHub roadmap in a half-written state. ``cancel``
    #   is still honoured.
    # - Approval gate covers the apply paths on all three actions; the
    #   action-level ``ApprovalRequiredGuardrail`` enforces this.
    MISSION_EXECUTION_POLICY: ClassVar[MissionExecutionPolicy] = (
        MissionExecutionPolicy(
            max_concurrent_instances=1,
            concurrency_scope=MissionConcurrencyScope.SESSION,
            on_concurrency_violation="return_existing",
            chains_with_modes=["bootstrap", "refresh", "assignments"],
            preemptible=False,
            interruptible=True,
            cancel_propagates_to_children=True,
            requires_human_approval_before=[
                "DesignProcessCapability.bootstrap_roadmap_from_objectives",
                "DesignProcessCapability.sync_roadmap_with_github",
                "DesignProcessCapability.propose_task_assignments",
            ],
            mutates_remote=True,
            max_runtime_seconds=1800.0,  # 30 minutes
            max_llm_cost_usd=5.0,
        )
    )

    async def initialize(self) -> None:
        # Mount the approval guardrail BEFORE super().initialize()
        # — the base's ``_create_action_policy`` resolves
        # ``action_policy_blueprints`` into the code-generation
        # policy at that point. Setting it later is a no-op.
        #
        # The guardrail's gated prefixes mirror this coordinator's
        # ``MISSION_EXECUTION_POLICY.requires_human_approval_before``
        # — single source of truth so the spec stays the contract
        # the runtime enforces.
        from polymathera.colony.agents.patterns.actions.code_constraints import (
            ApprovalRequiredGuardrail,
        )
        self.action_policy_blueprints["runtime_guardrail"] = (
            ApprovalRequiredGuardrail(
                approval_required_action_prefixes=(
                    self.MISSION_EXECUTION_POLICY
                    .requires_human_approval_before
                ),
            )
        )

        # The design-monorepo trio first so the per-agent clone path
        # is resolved before DesignProcessCapability /
        # SystemDesignCapability initialise (both inherit the shared
        # ``DesignMonorepoCapabilityBase`` clone-resolution path).
        self.add_capability_blueprints([
            *design_monorepo_capability_blueprints(),
            SystemDesignCapability.bind(),
            DesignProcessCapability.bind(),
            GitHubCapability.bind(scope=BlackboardScope.SESSION),
            HumanApprovalCapability.bind(scope=BlackboardScope.SESSION),
        ])
        await super().initialize()
        logger.info(
            "ProjectPlanningCoordinator %s initialised", self.agent_id,
        )
