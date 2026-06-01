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

from polymathera.colony.agents.base import Agent
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

    async def initialize(self) -> None:
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
