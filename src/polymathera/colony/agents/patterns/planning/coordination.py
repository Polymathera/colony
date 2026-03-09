"""Multi-agent coordination mechanisms.

This module provides coordination mechanisms for multi-agent planning:
- PartialGlobalPlanning: Coordinate local plans with global context
- NegotiationProtocol: Protocol for conflict negotiation
- MarketBasedNegotiation: Market-based resource allocation
- ConsensusNegotiation: Consensus-based conflict resolution
"""

from abc import ABC, abstractmethod
from typing import Any
import logging

from ...models import ActionPlan, PlanStatus
from .policies import ConflictResolver
from .blackboard import PlanBlackboard

logger = logging.getLogger(__name__)


# ============================================================================
# Coordination Strategy Protocol
# ============================================================================


class CoordinationStrategy(ABC):
    """Strategy protocol for coordinating plans across agents."""

    @abstractmethod
    async def coordinate_plans(
        self, blackboard: PlanBlackboard, agent_id: str, local_plan: ActionPlan
    ) -> ActionPlan:
        """Coordinate local plan with global context.

        Args:
            blackboard: PlanBlackboard instance
            agent_id: ID of agent coordinating plan
            local_plan: Local plan to coordinate

        Returns:
            Revised plan after coordination
        """
        ...


# ============================================================================
# Negotiation Protocol
# ============================================================================


class NegotiationProtocol(ABC):
    """Protocol for negotiating conflicts between agents."""

    @abstractmethod
    async def negotiate(
        self,
        plan: ActionPlan,
        conflicting_agents: list[str],
        conflicts: list[dict[str, Any]],
        blackboard: PlanBlackboard,
    ) -> ActionPlan | None:
        """Negotiate to resolve conflicts.

        Args:
            plan: Plan with conflicts
            conflicting_agents: IDs of agents with conflicting plans
            conflicts: List of conflict descriptions
            blackboard: PlanBlackboard instance

        Returns:
            Revised plan after negotiation, or None if negotiation failed
        """
        ...


# ============================================================================
# Partial Global Planning Coordinator
# ============================================================================


class PartialGlobalPlanning(CoordinationStrategy):
    """Implement Partial Global Planning (PGP) pattern.

    Process:
    1. Agent creates local plan
    2. Broadcast plan to relevant agents (parent, siblings)
    3. Detect conflicts
    4. Negotiate/resolve conflicts
    5. Update plan
    6. Get approval from parent (if needed)
    """

    def __init__(
        self,
        conflict_resolver: ConflictResolver,
        negotiation_protocol: NegotiationProtocol | None = None,
    ):
        """Initialize PGP coordinator.

        Args:
            conflict_resolver: ConflictResolver for resolving conflicts
            negotiation_protocol: Protocol for negotiation (optional)
        """
        self.conflict_resolver = conflict_resolver
        self.negotiation_protocol = negotiation_protocol

    async def coordinate_plans(
        self, blackboard: PlanBlackboard, agent_id: str, local_plan: ActionPlan
    ) -> ActionPlan:
        """Coordinate local plan with system.

        Args:
            blackboard: PlanBlackboard instance
            agent_id: ID of agent coordinating plan
            local_plan: Local plan to coordinate

        Returns:
            Coordinated plan
        """
        logger.info(f"Coordinating plan {local_plan.plan_id} for agent {agent_id}")

        # Step 1: Detect conflicts
        conflicts = await blackboard.detect_conflicts(local_plan)

        if not conflicts:
            # No conflicts - propose directly
            logger.info(f"No conflicts detected for plan {local_plan.plan_id}")
            approved, message = await blackboard.propose_plan(local_plan, agent_id)
            if not approved:
                local_plan.status = PlanStatus.BLOCKED
                local_plan.blocked_reason = message
            return local_plan

        logger.info(
            f"Detected {len(conflicts)} conflicts for plan {local_plan.plan_id}"
        )

        # Step 2: Categorize conflicts by severity
        critical_conflicts = [c for c in conflicts if c.get("severity") == "critical"]
        high_conflicts = [c for c in conflicts if c.get("severity") == "high"]
        low_conflicts = [
            c for c in conflicts if c.get("severity") in ["medium", "low"]
        ]

        # Step 3: Resolve critical conflicts (must be resolved)
        if critical_conflicts:
            logger.info(
                f"Resolving {len(critical_conflicts)} critical conflicts for plan {local_plan.plan_id}"
            )

            for conflict in critical_conflicts:
                # Get all plans involved in this conflict
                involved_plans = await self._get_involved_plans(
                    conflict, local_plan, blackboard, agent_id
                )

                # Resolve conflict
                resolved_plans = await self.conflict_resolver.resolve_conflict(
                    conflict, involved_plans
                )

                # Find our plan in resolved plans
                for resolved_plan in resolved_plans:
                    if resolved_plan.plan_id == local_plan.plan_id:
                        local_plan = resolved_plan
                        break

            # Check if plan is still viable
            if local_plan.status == PlanStatus.BLOCKED:
                logger.warning(
                    f"Plan {local_plan.plan_id} blocked after critical conflict resolution"
                )
                return local_plan

        # Step 4: Negotiate high-priority conflicts
        if high_conflicts and self.negotiation_protocol:
            logger.info(
                f"Negotiating {len(high_conflicts)} high-priority conflicts for plan {local_plan.plan_id}"
            )

            # Find conflicting agents
            conflicting_agents = set()
            for conflict in high_conflicts:
                if "plan_a_id" in conflict:
                    plan_id = conflict["plan_a_id"]
                    if plan_id != local_plan.plan_id:
                        other_plan = await blackboard.get_plan(plan_id, agent_id)
                        if other_plan:
                            conflicting_agents.add(other_plan.agent_id)

                if "plan_b_id" in conflict:
                    plan_id = conflict["plan_b_id"]
                    if plan_id != local_plan.plan_id:
                        other_plan = await blackboard.get_plan(plan_id, agent_id)
                        if other_plan:
                            conflicting_agents.add(other_plan.agent_id)

            # Negotiate
            if conflicting_agents:
                negotiated_plan = await self.negotiation_protocol.negotiate(
                    local_plan, list(conflicting_agents), high_conflicts, blackboard
                )

                if negotiated_plan:
                    local_plan = negotiated_plan
                    logger.info(
                        f"Negotiation successful for plan {local_plan.plan_id}"
                    )
                else:
                    logger.warning(
                        f"Negotiation failed for plan {local_plan.plan_id}"
                    )

        # Step 5: Accept low-priority conflicts (log warnings)
        if low_conflicts:
            logger.info(
                f"Accepting {len(low_conflicts)} low-priority conflicts for plan {local_plan.plan_id}"
            )
            local_plan.metadata["low_priority_conflicts"] = low_conflicts

        # Step 6: Propose coordinated plan
        approved, message = await blackboard.propose_plan(local_plan, agent_id)
        if not approved:
            local_plan.status = PlanStatus.BLOCKED
            local_plan.blocked_reason = message
            logger.warning(
                f"Plan {local_plan.plan_id} not approved after coordination: {message}"
            )
        else:
            logger.info(f"Plan {local_plan.plan_id} successfully coordinated")

        return local_plan

    async def _get_involved_plans(
        self,
        conflict: dict[str, Any],
        local_plan: ActionPlan,
        blackboard: PlanBlackboard,
        agent_id: str,
    ) -> list[ActionPlan]:
        """Get all plans involved in a conflict.

        Args:
            conflict: Conflict description
            local_plan: Local plan
            blackboard: PlanBlackboard instance
            agent_id: ID of agent requesting plans

        Returns:
            List of involved plans
        """
        plans = [local_plan]

        # Extract other plan IDs from conflict
        for key in ["plan_a_id", "plan_b_id", "with_plan"]:
            if key in conflict:
                plan_id = conflict[key]
                if plan_id != local_plan.plan_id:
                    other_plan = await blackboard.get_plan(plan_id, agent_id)
                    if other_plan:
                        plans.append(other_plan)

        return plans


# ============================================================================
# Market-Based Negotiation
# ============================================================================


class MarketBasedNegotiation(NegotiationProtocol):
    """Market-based negotiation for resource allocation.

    Agents bid for resources (pages, compute) based on:
    - Plan priority
    - Resource value
    - Agent budget
    """

    def __init__(self, agent_budgets: dict[str, float] | None = None):
        """Initialize market-based negotiation.

        Args:
            agent_budgets: Mapping of agent_id -> budget (default: equal budgets)
        """
        self.agent_budgets = agent_budgets or {}

    async def negotiate(
        self,
        plan: ActionPlan,
        conflicting_agents: list[str],
        conflicts: list[dict[str, Any]],
        blackboard: PlanBlackboard,
    ) -> ActionPlan | None:
        """Negotiate using market-based bidding.

        Args:
            plan: Plan to negotiate for
            conflicting_agents: IDs of agents with conflicting plans
            conflicts: List of conflicts
            blackboard: PlanBlackboard instance

        Returns:
            Revised plan after negotiation, or None if failed
        """
        logger.info(
            f"Starting market-based negotiation for plan {plan.plan_id} with {len(conflicting_agents)} agents"
        )

        # For each conflict, determine who wins based on bids
        for conflict in conflicts:
            conflict_type = conflict.get("type")

            if conflict_type == "cache_contention":
                # Bid for cache pages
                my_bid = self._calculate_bid(plan, conflict.get("shared_pages", []))

                # Get bids from other agents (simplified - assume equal for now)
                other_bids = {
                    agent_id: self._calculate_bid_for_agent(agent_id, conflict)
                    for agent_id in conflicting_agents
                }

                # Check if we win
                max_other_bid = max(other_bids.values()) if other_bids else 0.0
                if my_bid < max_other_bid:
                    # We lose - need to adjust our plan
                    logger.info(
                        f"Plan {plan.plan_id} lost bid ({my_bid} < {max_other_bid}), adjusting"
                    )

                    # Adjust plan: remove contested pages or delay execution
                    plan.metadata["negotiation_result"] = "lost_bid"
                    plan.metadata["execution_delay_s"] = 10.0  # Delay execution
                else:
                    logger.info(
                        f"Plan {plan.plan_id} won bid ({my_bid} >= {max_other_bid})"
                    )
                    plan.metadata["negotiation_result"] = "won_bid"

        return plan

    def _calculate_bid(self, plan: ActionPlan, pages: list[str]) -> float:
        """Calculate bid for resources.

        Args:
            plan: Plan to bid for
            pages: Pages being bid on

        Returns:
            Bid value
        """
        # Bid based on:
        # 1. Plan priority
        # 2. Number of pages
        # 3. Agent budget

        priority = plan.metadata.get("priority", 1.0)
        budget = self.agent_budgets.get(plan.agent_id, 100.0)
        page_value = len(pages) * 10.0  # Value per page

        # Bid = priority * page_value, capped by budget
        bid = min(priority * page_value, budget)

        logger.debug(
            f"Calculated bid for plan {plan.plan_id}: {bid} (priority={priority}, pages={len(pages)}, budget={budget})"
        )
        return bid

    def _calculate_bid_for_agent(
        self, agent_id: str, conflict: dict[str, Any]
    ) -> float:
        """Calculate bid for another agent (simplified).

        Args:
            agent_id: Agent ID
            conflict: Conflict description

        Returns:
            Estimated bid value
        """
        # Simplified: assume average bid
        budget = self.agent_budgets.get(agent_id, 100.0)
        pages = conflict.get("shared_pages", [])
        return len(pages) * 10.0 * 0.5  # Assume medium priority


# ============================================================================
# Consensus-Based Negotiation
# ============================================================================


class ConsensusNegotiation(NegotiationProtocol):
    """Consensus-based negotiation for conflict resolution.

    Agents iteratively adjust their plans to reach consensus:
    - Each round, agents propose adjustments
    - Plans converge toward mutual compatibility
    - Process terminates when consensus reached or max rounds exceeded
    """

    def __init__(self, max_rounds: int = 5):
        """Initialize consensus negotiation.

        Args:
            max_rounds: Maximum negotiation rounds
        """
        self.max_rounds = max_rounds

    async def negotiate(
        self,
        plan: ActionPlan,
        conflicting_agents: list[str],
        conflicts: list[dict[str, Any]],
        blackboard: PlanBlackboard,
    ) -> ActionPlan | None:
        """Negotiate using consensus-building.

        Args:
            plan: Plan to negotiate for
            conflicting_agents: IDs of agents with conflicting plans
            conflicts: List of conflicts
            blackboard: PlanBlackboard instance

        Returns:
            Revised plan after negotiation, or None if failed
        """
        logger.info(
            f"Starting consensus negotiation for plan {plan.plan_id} with {len(conflicting_agents)} agents"
        )

        # Get all conflicting plans
        all_plans = [plan]
        for agent_id in conflicting_agents:
            # Try to get their active plan
            other_plans = await blackboard.query_plans(  # TODO: This is not implemented yet
                {"agent_id": agent_id, "status": "active"}
            )
            if other_plans:
                all_plans.extend(other_plans)

        # Iteratively adjust plans toward consensus
        for round_num in range(self.max_rounds):
            logger.info(
                f"Consensus negotiation round {round_num + 1}/{self.max_rounds}"
            )

            # Adjust our plan based on conflicts
            adjusted_plan = await self._adjust_plan(plan, all_plans, conflicts)

            # Check if conflicts resolved
            remaining_conflicts = await self._check_remaining_conflicts(
                adjusted_plan, all_plans
            )

            if not remaining_conflicts:
                logger.info(
                    f"Consensus reached for plan {plan.plan_id} in {round_num + 1} rounds"
                )
                return adjusted_plan

            # Update for next round
            plan = adjusted_plan
            conflicts = remaining_conflicts

        # Max rounds exceeded
        logger.warning(
            f"Consensus negotiation failed for plan {plan.plan_id} after {self.max_rounds} rounds"
        )
        return plan  # Return best effort

    async def _adjust_plan(
        self, plan: ActionPlan, all_plans: list[ActionPlan], conflicts: list[dict[str, Any]]
    ) -> ActionPlan:
        """Adjust plan toward consensus.

        Args:
            plan: Plan to adjust
            all_plans: All plans being negotiated
            conflicts: Current conflicts

        Returns:
            Adjusted plan
        """
        # Strategy: Make small adjustments to reduce conflicts
        # - Reduce working set overlap
        # - Stagger execution times
        # - Lower resource demands

        for conflict in conflicts:
            conflict_type = conflict.get("type")

            if conflict_type == "cache_contention":
                # Reduce cache footprint
                working_set_size = len(plan.cache_context.working_set)
                if working_set_size > 5:
                    # Remove least important pages
                    priorities = plan.cache_context.working_set_priority
                    sorted_pages = sorted(
                        plan.cache_context.working_set,
                        key=lambda p: priorities.get(p, 0.5),
                    )
                    # Keep top 80% of pages
                    keep_count = int(working_set_size * 0.8)
                    plan.cache_context.working_set = sorted_pages[:keep_count]

                    logger.info(
                        f"Reduced working set for plan {plan.plan_id} from {working_set_size} to {keep_count} pages"
                    )

            elif conflict_type == "resource_contention":
                # Add execution delay
                current_delay = plan.metadata.get("execution_delay_s", 0.0)
                plan.metadata["execution_delay_s"] = current_delay + 5.0
                logger.info(
                    f"Added 5s delay to plan {plan.plan_id} (total: {current_delay + 5.0}s)"
                )

        return plan

    async def _check_remaining_conflicts(
        self, plan: ActionPlan, all_plans: list[ActionPlan]
    ) -> list[dict[str, Any]]:
        """Check for remaining conflicts.

        Args:
            plan: Plan to check
            all_plans: All plans being negotiated

        Returns:
            List of remaining conflicts
        """
        # Simplified conflict check
        conflicts = []

        for other_plan in all_plans:
            if other_plan.plan_id == plan.plan_id:
                continue

            # Check cache overlap
            my_pages = set(plan.cache_context.working_set)
            other_pages = set(other_plan.cache_context.working_set)
            overlap = my_pages & other_pages

            if len(overlap) > 3:  # Threshold for conflict
                conflicts.append(
                    {
                        "type": "cache_contention",
                        "severity": "medium",
                        "plan_a_id": plan.plan_id,
                        "plan_b_id": other_plan.plan_id,
                        "shared_pages": list(overlap),
                    }
                )

        return conflicts

