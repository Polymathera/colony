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
from polymathera.colony.agents.patterns.capabilities.mission_status import (
    MissionStatusCapability,
)
from polymathera.colony.agents.missions.project_planning.mission_control import (
    ProjectPlanningMissionControlCapability,
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
    # - One instance per SESSION (not per AGENT). With
    #   ``max_concurrent_instances=1`` + ``return_existing``, any
    #   sibling ``spawn_mission`` against this session resolves to
    #   the running coordinator's ``agent_id`` regardless of mode —
    #   the LLM's chain converges on a single agent.
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
            preemptible=False,
            interruptible=True,
            cancel_propagates_to_children=True,
            requires_human_approval_before=[
                "DesignProcessCapability.bootstrap_roadmap_from_objectives",
                "DesignProcessCapability.sync_roadmap_with_github",
                "DesignProcessCapability.propose_task_assignments",
                # The decompose primitives are split:
                # ``classify_issues_decomposability`` and
                # ``propose_decompositions`` are READ-ONLY (no GitHub
                # mutation) and don't need approval — the planner can
                # call them freely to compose its strategy. Only the
                # APPLY primitive ``create_decomposition`` is gated.
                # Per ``decompose_and_session_recovery_fixes_plan.md``
                # item 3's primitives-not-pipelines split.
                "DesignProcessCapability.create_decomposition",
            ],
            mutates_remote=True,
            max_runtime_seconds=1800.0,  # 30 minutes
            max_llm_cost_usd=5.0,
            # Honest budget for the propose → request_approval → idle
            # poll → apply → report shape. The default of 20 is the
            # right cap for one-shot reasoning agents but starves a
            # mission that legitimately spends N iterations idle-polling
            # for the operator's approval response. The structural fix
            # is to stop counting idle-poll iterations (see
            # ``colony/agent_loop_idle_wait_separation_plan.md``); until
            # that lands, this number absorbs the polling tail.
            max_iterations=50,
        )
    )

    # ---------- Canonical mission-param + mode names ----------
    #
    # Caller-parameter names + mode values consumed by code that knows
    # this is the ``project_planning`` mission specifically: the
    # ``DecomposeCompletionValidator``, the
    # ``request_decompose_early_stop`` action, audit tools, regression
    # pins. Consumers reference these ClassVars instead of bare
    # string literals — single source of truth per
    # [[colony-scoped-params-propagation]] applied to mission-param
    # names. They live HERE (on the coordinator that owns the mission)
    # rather than on the generic ``MissionSpec`` because they describe
    # ``project_planning``-specific semantics; another mission's
    # ``MissionSpec`` instance has a different parameter vocabulary
    # and should not inherit any of these names.
    ISSUE_NUMBERS_PARAM_NAME: ClassVar[str] = "issue_numbers"
    MAX_PARENTS_PER_RUN_PARAM_NAME: ClassVar[str] = "max_parents_per_run"
    MODE_PARAM_NAME: ClassVar[str] = "mode"
    DECOMPOSE_MODE_VALUE: ClassVar[str] = "decompose"

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

        # Mount the decompose-mode completion validator. It rejects
        # ``signal_completion`` until the in-scope issue set is
        # drained (decomposed, classified non-decomposable, or
        # explicitly early-stopped via ``request_decompose_early_stop``).
        # For non-decompose modes it delegates to the default
        # ``LLMCompletionValidator`` so existing flows are untouched.
        # See ``decompose_one_and_done_and_spinner_plan.md`` Change 2.
        from polymathera.colony.agents.missions.project_planning.completion_validator import (
            DecomposeCompletionValidator,
        )
        self.action_policy_blueprints["completion_validator"] = (
            DecomposeCompletionValidator()
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
            # MissionStatusCapability exposes ``emit_mission_status`` so
            # the coordinator's planner can publish a one-line narrative
            # ("loading design context...", "classifying issues...")
            # the chat UI surfaces in place of an opaque spinner. See
            # the capability's module docstring for the design.
            MissionStatusCapability.bind(scope=BlackboardScope.SESSION),
            # ProjectPlanningMissionControlCapability owns the typed
            # ``request_decompose_early_stop`` primitive (the LLM-
            # callable contract that records an explicit user
            # acknowledgement so the DecomposeCompletionValidator can
            # accept signal_completion with remainders deferred). The
            # primitive is mission-specific so it lives on a mission-
            # specific capability, NOT on the generic
            # ``DesignProcessCapability`` (the design-process surface
            # hosts decompose OPERATIONS — create / classify / propose
            # — not mission-control concerns).
            ProjectPlanningMissionControlCapability.bind(
                scope=BlackboardScope.SESSION,
            ),
        ])
        await super().initialize()
        logger.info(
            "ProjectPlanningCoordinator %s initialised", self.agent_id,
        )
