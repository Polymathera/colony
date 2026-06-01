"""``project_planning`` mission — propose / approve / apply roadmap edits.

The mission wraps the three action surfaces already shipped in P5:

- :meth:`DesignProcessCapability.bootstrap_roadmap_from_objectives`
  (P5b) — initial roadmap from the design-context objectives.
- :meth:`DesignProcessCapability.sync_roadmap_with_github` (P5c) —
  reconcile ``ROADMAP.md`` with the GitHub issue tracker.
- :meth:`DesignProcessCapability.propose_task_assignments` (P5d) —
  colony/user assignment proposal.

The coordinator is a pure ``Agent`` subclass — the LLM planner walks
the flow ``action(dry_run=True)`` → ``request_human_approval(extra=...)``
→ ``action(dry_run=False)`` using the actions the mounted capabilities
expose. No bespoke state machine; no separate worker class
(:class:`MissionSpec.worker` is intentionally empty for this mission).
"""

from .coordinator import ProjectPlanningCoordinator

__all__ = ["ProjectPlanningCoordinator"]
