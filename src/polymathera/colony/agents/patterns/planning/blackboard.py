"""Plan blackboard for distributed plan storage and coordination.

This module provides PlanBlackboard built on EnhancedBlackboard for:
- Storing plans with Redis persistence
- Event-driven coordination (no polling)
- Conflict detection (cache contention, dependencies)
- Approval workflows for hierarchical coordination
"""

import logging
import time
from typing import Any

from ...base import Agent
from ...blackboard.blackboard import EnhancedBlackboard
from ...blackboard.protocol import PlanProtocol
from ...scopes import BlackboardScope, get_scope_prefix
from .policies import PlanAccessPolicy
from ...models import ActionPlan, PlanStatus
from ....distributed.ray_utils import serving

logger = logging.getLogger(__name__)


class PlanBlackboard:
    """Local object with Redis backend (NOT a Ray actor)."""

    def __init__(
        self,
        agent: Agent,
        *,
        scope: BlackboardScope = BlackboardScope.COLONY,
        plan_access_policy: PlanAccessPolicy | None = None,
    ):
        """Initialize plan blackboard.

        Args:
            agent: Agent using this blackboard (for scoping and access control)
            scope: Scope for blackboard. This scope may contain multiple
                plans for different agents that need to coordinate.
            plan_access_policy: Policy for plan-level access control (optional)
        """
        self.agent = agent
        self.scope_id = f"{get_scope_prefix(scope, agent)}:action_plans"
        self.plan_access_policy = plan_access_policy
        self.blackboard: EnhancedBlackboard | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize blackboard."""
        if self._initialized:
            return

        app_name = serving.get_my_app_name()
        # Build on EnhancedBlackboard for Redis persistence
        # Note: We don't pass access_policy to EnhancedBlackboard because
        # we need plan-level semantics, not key-level semantics
        self.blackboard = EnhancedBlackboard(
            app_name=app_name,
            scope_id=self.scope_id,
            access_policy=None,  # Use plan_access_policy instead
            backend_type=None,  # Use the globally configured blackboard backend
            enable_events=True,
        )

        await self.blackboard.initialize()
        self._initialized = True

    async def get_plan(self, agent_id: str) -> ActionPlan | None:
        """Get plan for specific agent."""
        key = PlanProtocol.plan_key(agent_id, namespace="plan")
        plan_dict = await self.blackboard.read(key)
        return ActionPlan(**plan_dict) if plan_dict else None

    async def get_plan_by_id(self, plan_id: str) -> ActionPlan | None:
        """Get plan by plan_id (requires search across all plans)."""
        # TODO: Implement efficient search for plan by ID. Use RedisOM indexing.
        # Query all plans
        entries = await self.blackboard.query(
            namespace=PlanProtocol.plan_pattern(namespace="plan"),  # Query all agent_id keys
            limit=1000,
        )

        for entry in entries:
            if entry.value.get("plan_id") == plan_id:
                return ActionPlan(**entry.value)

        return None

    async def get_all_plans(self, limit: int | None = None) -> list[ActionPlan]:
        """Get all plans."""
        entries = await self.blackboard.query(
            namespace=PlanProtocol.plan_pattern(namespace="plan"),  # Query all agent_id keys
            limit=limit or 1000,  # Reasonable limit for plan discovery
        )

        return [ActionPlan(**entry.value) for entry in entries if entry.value]

    async def update_plan(self, plan: ActionPlan) -> None:
        """Update plan using plan's key method."""
        plan.updated_at = time.time()
        await self.blackboard.write(
            key=PlanProtocol.plan_key(plan.agent_id, namespace="plan"),
            value=plan.model_dump(),
            created_by=plan.agent_id,
            tags={"plan", plan.status.value},
            metadata={
                "agent_id": plan.agent_id,
                "goals": plan.goals,
                "status": plan.status.value,
            },
        )

    async def propose_plan(
        self, plan: ActionPlan, requesting_agent_id: str
    ) -> tuple[bool, str]:
        """Propose plan (with approval workflow if needed)."""
        # Check plan access policy
        if self.plan_access_policy:
            if not self.plan_access_policy.can_create_plan(requesting_agent_id, plan):
                return False, "Permission denied: cannot create plan"

        # Ensure status is set — planners may leave it as None
        if plan.status is None:
            plan.status = PlanStatus.PROPOSED

        # Store plan
        await self.blackboard.write(
            key = PlanProtocol.plan_key(plan.agent_id, namespace="plan"),
            value=plan.model_dump(),
            created_by=requesting_agent_id,
            tags={"plan", plan.status.value},
            metadata={
                "agent_id": plan.agent_id,
                "goals": plan.goals,
                "status": plan.status.value,
            },
        )

        # Check if parent approval needed
        if plan.approval_required and plan.parent_plan_id:
            # Notify parent (via blackboard events)
            await self.blackboard.write(
                key=PlanProtocol.approval_request_key(plan.plan_id, namespace="plan"),
                value={
                    "plan_id": plan.plan_id,
                    "from_agent": requesting_agent_id,
                    "parent_plan": plan.parent_plan_id,
                },
                created_by=requesting_agent_id,
                ttl_seconds=3600,  # Auto-expire # TODO: Make configurable
            )
            return False, "Awaiting parent approval"

        # Auto-approve if no parent approval needed
        plan.status = PlanStatus.APPROVED
        await self.update_plan(plan)
        return True, "Plan approved"

    async def detect_conflicts(self, plan: ActionPlan) -> list[dict]:
        """Detect conflicts."""
        entries = await self.blackboard.query(
            namespace=PlanProtocol.plan_pattern(namespace="plan"),  # Query all agent_id keys
            limit=1000,
        )

        active_plans = [
            ActionPlan(**entry.value)
            for entry in entries
            if entry.value.get("status")
            in [PlanStatus.ACTIVE.value, PlanStatus.APPROVED.value]
            and entry.value.get("agent_id") != plan.agent_id
        ]

        conflicts = []

        # Check cache conflicts
        my_ws = set(plan.cache_context.working_set)
        for other in active_plans:
            other_ws = set(other.cache_context.working_set)
            overlap = my_ws & other_ws

            if len(overlap) > len(my_ws) * 0.7:  # 70% overlap threshold
                conflicts.append(
                    {
                        "type": "cache_contention",
                        "with_agent": other.agent_id,
                        "with_plan": other.plan_id,
                        "overlap_pages": list(overlap),
                    }
                )

        return conflicts

    async def approve_plan(
        self,
        plan_id: str,
        approver_agent_id: str,
        approved: bool,
        reason: str | None = None,
    ) -> bool:
        """Approve or reject a plan.

        Args:
            plan_id: Plan ID to approve/reject
            approver_agent_id: Agent ID performing approval
            approved: True to approve, False to reject
            reason: Optional reason for decision

        Returns:
            True if approval was successful, False otherwise
        """
        # Get the plan
        plan = await self.get_plan_by_id(plan_id)
        if not plan:
            logger.warning(f"Plan {plan_id} not found for approval")
            return False

        # Check plan access policy
        if self.plan_access_policy:
            if not self.plan_access_policy.can_approve_plan(approver_agent_id, plan):
                logger.warning(
                    f"Agent {approver_agent_id} denied permission to approve plan {plan_id}"
                )
                return False

        # Update plan status
        if approved:
            plan.status = PlanStatus.APPROVED
            plan.approved_by = approver_agent_id
            logger.info(f"Plan {plan_id} approved by {approver_agent_id}")
        else:
            plan.status = PlanStatus.FAILED
            plan.metadata["rejection_reason"] = reason
            logger.info(f"Plan {plan_id} rejected by {approver_agent_id}: {reason}")

        await self.update_plan(plan)

        # Notify agent who proposed the plan
        await self._notify_agent(
            plan.agent_id,
            "plan_approval_decision",
            {
                "plan_id": plan_id,
                "approved": approved,
                "approver": approver_agent_id,
                "reason": reason,
            },
        )

        return True

    async def get_child_plans(
        self, parent_plan_id: str, requesting_agent_id: str
    ) -> list[ActionPlan]:
        """Get all child plans of a parent plan.

        Args:
            parent_plan_id: Parent plan ID
            requesting_agent_id: Agent requesting child plans

        Returns:
            List of child plans
        """
        # Query all plans with this parent
        entries = await self.blackboard.query(
            namespace=PlanProtocol.plan_pattern(namespace="plan"),  # Query all agent_id keys
            limit=1000,
        )

        # TODO: Optimize this query with indexing in future
        child_plans = []
        for entry in entries:
            plan_data = entry.value
            if plan_data.get("parent_plan_id") == parent_plan_id:
                plan = ActionPlan(**plan_data)

                # Check plan access policy
                if self.plan_access_policy:
                    if not self.plan_access_policy.can_read_plan(requesting_agent_id, plan):
                        continue  # Skip this plan

                child_plans.append(plan)

        logger.info(
            f"Found {len(child_plans)} child plans for parent plan {parent_plan_id}"
        )
        return child_plans

    async def subscribe_to_plan(
        self, plan_id: str, subscriber_agent_id: str
    ) -> bool:
        """Subscribe to plan updates and events.

        Args:
            plan_id: Plan ID to subscribe to
            subscriber_agent_id: Agent ID subscribing

        Returns:
            True if subscription successful, False otherwise
        """
        # Get the plan
        plan = await self.get_plan_by_id(plan_id)
        if not plan:
            logger.warning(f"Plan {plan_id} not found for subscription")
            return False

        # Check plan access policy
        if self.plan_access_policy:
            if not self.plan_access_policy.can_read_plan(subscriber_agent_id, plan):
                logger.warning(
                    f"Agent {subscriber_agent_id} denied permission to subscribe to plan {plan_id}"
                )
                return False

        # Subscribe to plan updates via blackboard events
        # Store subscription mapping
        subscription_key = PlanProtocol.subscription_key(plan_id, subscriber_agent_id, namespace="plan")
        await self.blackboard.write(
            key=subscription_key,
            value={
                "plan_id": plan_id,
                "subscriber": subscriber_agent_id,
                "subscribed_at": time.time(),
            },
            created_by=subscriber_agent_id,
            tags={"plan_subscription"},
        )

        logger.info(f"Agent {subscriber_agent_id} subscribed to plan {plan_id}")
        return True

    async def _notify_agent(
        self, agent_id: str, event_type: str, data: dict[str, Any]
    ) -> None:
        """Notify specific agent of event.

        Args:
            agent_id: Agent to notify
            event_type: Type of event
            data: Event data
        """
        notification_key = PlanProtocol.notification_key(agent_id, time.time(), namespace="plan")
        await self.blackboard.write(
            key=notification_key,
            value={
                "event_type": event_type,
                "data": data,
                "timestamp": time.time(),
            },
            created_by="plan_blackboard",
            ttl_seconds=3600,  # TODO: Make configurable. Auto-expire after 1 hour
            tags={"notification", event_type},
        )

    async def _notify_subscribers(
        self, plan_id: str, event_type: str, data: dict[str, Any]
    ) -> None:
        """Notify all subscribers of a plan about an event.

        Args:
            plan_id: Plan ID
            event_type: Type of event
            data: Event data
        """
        # Find all subscribers
        entries = await self.blackboard.query(
            namespace=PlanProtocol.subscription_key(plan_id, "*", namespace="plan"),
            limit=1000,
        )

        for entry in entries:
            subscriber_id = entry.value.get("subscriber")
            if subscriber_id:
                await self._notify_agent(subscriber_id, event_type, data)

        logger.info(f"Notified {len(entries)} subscribers of plan {plan_id}")

    async def _has_dependency_cycle(self, plan: ActionPlan) -> bool:
        """Check if plan creates a dependency cycle.

        Args:
            plan: Plan to check

        Returns:
            True if cycle detected, False otherwise
        """
        visited = set()
        current_path = set()

        async def visit(plan_id: str) -> bool:
            """DFS to detect cycle."""
            if plan_id in current_path:
                return True  # Cycle detected

            if plan_id in visited:
                return False  # Already checked this branch

            visited.add(plan_id)
            current_path.add(plan_id)

            # Get plan and check its dependencies
            p = await self.get_plan_by_id(plan_id)
            if p:
                for dep_id in p.depends_on:
                    if await visit(dep_id):
                        return True

            current_path.remove(plan_id)
            return False

        # Check if adding this plan creates a cycle
        for dep_id in plan.depends_on:
            if await visit(dep_id):
                logger.warning(f"Dependency cycle detected for plan {plan.plan_id}")
                return True

        return False
