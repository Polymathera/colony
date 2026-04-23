"""Pluggable multi-agent coordination mechanisms.

This module provides coordination mechanisms for multi-agent planning that customize planning behavior:
- PartialGlobalPlanning: Coordinate local plans with global context
- NegotiationProtocol: Protocol for conflict negotiation
- MarketBasedNegotiation: Market-based resource allocation
- ConsensusNegotiation: Consensus-based conflict resolution
- PlanCoordinationCapability: Multi-agent coordination

Per Architecture Principle #2: ONE Planner class customized via pluggable policies.

Design: Policies are stateless and receive context (page graphs, etc.) when called.
The Planner that uses these policies is responsible for providing the necessary context.

PlanCoordinationCapability provides conflict detection and resolution for multi-agent planning
over shared VCM-paged context. Used by coordinator agents that need
to ensure their plans don't conflict with sibling agents' plans.

Dual interface:
- **Programmatic API**: ``check_conflicts()``, ``resolve_conflict()`` — used
  by ``CacheAwareActionPlanner`` and ``CodeGenerationActionPolicy``.
- **LLM API**: ``@action_executor`` methods — used by ``MinimalActionPolicy``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any
from overrides import override

from ..blackboard import PlanBlackboard
from ...actions.dispatcher import action_executor
from ....base import Agent, AgentCapability
from ....scopes import BlackboardScope, get_scope_prefix
from ....models import (
    ActionPlan,
    PlanStatus,
    ConflictType,
    ConflictSeverity,
    ActionPlanConflict,
    ConflictResolutionStrategy,
    AgentSuspensionState,
)

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
        conflict_resolver: ActionPlanConflictResolver,
        negotiation_protocol: NegotiationProtocol | None = None,
    ):
        """Initialize PGP coordinator.

        Args:
            conflict_resolver: ActionPlanConflictResolver for resolving conflicts
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



# ============================================================================
# Cache-Aware Conflict Detection and Resolution
# ============================================================================


class CacheAwarePlanConflictDetector:
    """Detect conflicts between plans based on cache requirements.

    Identifies conflicts such as:
    - Resource exhaustion (total working set exceeds cache capacity)
    - Cache contention (multiple plans need same pages)
    - Sequential dependencies (plans must execute in order)
    """

    def __init__(self, cache_capacity: int, page_size: int = 40000):
        """Initialize conflict detector.

        Args:
            cache_capacity: Maximum number of pages that can be cached
            page_size: Average page size in tokens
        """
        self.cache_capacity = cache_capacity
        self.page_size = page_size

    async def detect_conflicts(self, plans: list[ActionPlan]) -> list[ActionPlanConflict]:
        """Detect conflicts between plans.

        Args:
            plans: List of plans to check for conflicts

        Returns:
            List of action plan conflicts
        """

        conflicts = []

        # Check for resource exhaustion
        all_pages = set()
        for plan in plans:
            all_pages.update(plan.cache_context.working_set)

        resource_conflict = self._check_resource_exhaustion(all_pages)
        if resource_conflict:
            conflicts.append(resource_conflict)

        # Check pairwise cache contention
        for i, plan_a in enumerate(plans):
            for plan_b in plans[i + 1 :]:
                contention = self._check_cache_contention(plan_a, plan_b)
                if contention:
                    conflicts.append(contention)

        # Calculate severity for each conflict
        for conflict in conflicts:
            conflict.severity = self._calculate_conflict_severity(conflict)

        logger.info(f"Detected {len(conflicts)} conflicts between {len(plans)} plans")
        return conflicts

    def _check_resource_exhaustion(self, total_working_set: set[str]) -> ActionPlanConflict | None:
        """Check if total working set exceeds cache capacity.

        Args:
            total_working_set: Combined working set of all plans

        Returns:
            Conflict description if exhaustion detected, None otherwise
        """

        if len(total_working_set) > self.cache_capacity:
            return ActionPlanConflict(
                type=ConflictType.RESOURCE_CONTENTION,
                description=f"Total working set ({len(total_working_set)} pages) exceeds cache capacity ({self.cache_capacity} pages)",
                severity=ConflictSeverity.HIGH,
                details={
                    "total_pages": len(total_working_set),
                    "capacity": self.cache_capacity,
                    "overflow": len(total_working_set) - self.cache_capacity,
                },
                recommended_strategy=ConflictResolutionStrategy.STAGGER_EXECUTION,
            )
        return None

    def _check_cache_contention(self, plan_a: ActionPlan, plan_b: ActionPlan) -> ActionPlanConflict | None:
        """Check for cache contention between two plans.

        Args:
            plan_a: First plan
            plan_b: Second plan

        Returns:
            Conflict description if contention detected, None otherwise
        """

        # Find overlapping pages
        pages_a = set(plan_a.cache_context.working_set)
        pages_b = set(plan_b.cache_context.working_set)
        shared_pages = pages_a & pages_b

        # Check if combined working set exceeds capacity
        combined_size = len(pages_a | pages_b)
        if combined_size > self.cache_capacity:
            return ActionPlanConflict(
                type=ConflictType.CACHE_CONTENTION,
                description=f"Plans {plan_a.plan_id} and {plan_b.plan_id} have combined working set ({combined_size} pages) exceeding cache capacity",
                plan_a_id=plan_a.plan_id,
                plan_b_id=plan_b.plan_id,
                details={
                    "shared_pages": list(shared_pages),
                    "combined_size": combined_size,
                    "capacity": self.cache_capacity,
                    "overflow": combined_size - self.cache_capacity,
                },
                recommended_strategy=ConflictResolutionStrategy.STAGGER_EXECUTION,
            )
        return None

    def _calculate_conflict_severity(self, conflict: ActionPlanConflict) -> ConflictSeverity:
        """Calculate conflict severity.

        Args:
            conflict: Conflict description

        Returns:
            Severity level: "low", "medium", "high", "critical"
        """

        conflict_type = conflict.type

        if conflict_type == ConflictType.RESOURCE_CONTENTION:
            overflow = conflict.details.get("overflow", 0)
            overflow_ratio = overflow / self.cache_capacity

            if overflow_ratio > 0.5:
                return ConflictSeverity.CRITICAL
            elif overflow_ratio > 0.25:
                return ConflictSeverity.HIGH
            elif overflow_ratio > 0.1:
                return ConflictSeverity.MEDIUM
            else:
                return ConflictSeverity.LOW

        elif conflict_type == ConflictType.CACHE_CONTENTION:
            overflow = conflict.details.get("overflow", 0)
            overflow_ratio = overflow / self.cache_capacity

            if overflow_ratio > 0.3:
                return ConflictSeverity.HIGH
            elif overflow_ratio > 0.15:
                return ConflictSeverity.MEDIUM
            else:
                return ConflictSeverity.LOW

        return ConflictSeverity.MEDIUM


class CacheAwarePlanConflictResolver:
    """Resolve conflicts between plans by modifying them.

    Resolution strategies:
    - Stagger execution (delay some plans)
    - Partition working sets (split plans)
    - Coordinate shared pages (optimize for sharing)
    """

    async def resolve_cache_contention(
        self, plan_a: ActionPlan, plan_b: ActionPlan, conflict: ActionPlanConflict
    ) -> tuple[ActionPlan, ActionPlan]:
        """Resolve cache contention between two plans.

        Args:
            plan_a: First plan
            plan_b: Second plan
            conflict: Conflict description

        Returns:
            Modified (plan_a, plan_b) tuple
        """
        logger.info(
            f"Resolving cache contention between {plan_a.plan_id} and {plan_b.plan_id}"
        )

        # Strategy: Stagger execution by adding delays
        modified_a, modified_b = await self._stagger_execution([plan_a, plan_b])

        return modified_a, modified_b

    async def _stagger_execution(self, plans: list[ActionPlan]) -> list[ActionPlan]:
        """Stagger plan execution to avoid simultaneous cache usage.

        Args:
            plans: Plans to stagger

        Returns:
            Modified plans with execution delays
        """
        # TODO: Is this even a robust strategy?

        # Add execution delay to all but first plan
        for i, plan in enumerate(plans):
            if i > 0:
                # Add metadata to indicate this plan should be delayed
                plan.metadata["execution_delay_s"] = i * 10.0  # 10 second stagger
                plan.metadata["staggered"] = True

        logger.info(f"Staggered execution of {len(plans)} plans")
        return plans

    async def _partition_working_sets(self, plans: list[ActionPlan]) -> list[ActionPlan]:
        """Partition working sets to reduce overlap.

        Args:
            plans: Plans to partition

        Returns:
            Modified plans with partitioned working sets
        """
        # Strategy: Identify shared pages and assign exclusivity
        all_pages: dict[str, list[str]] = {}  # page_id -> [plan_ids using it]

        for plan in plans:
            for page_id in plan.cache_context.working_set:
                if page_id not in all_pages:
                    all_pages[page_id] = []
                all_pages[page_id].append(plan.plan_id)

        # Mark pages as exclusive or shareable
        for plan in plans:
            exclusive = []
            shareable = []

            for page_id in plan.cache_context.working_set:
                if len(all_pages[page_id]) == 1:
                    # Only this plan uses this page
                    exclusive.append(page_id)
                else:
                    # Multiple plans use this page
                    shareable.append(page_id)

            plan.cache_context.exclusive_pages = exclusive
            plan.cache_context.shareable_pages = shareable

        logger.info(f"Partitioned working sets for {len(plans)} plans")
        return plans

    async def _coordinate_shared_pages(self, plans: list[ActionPlan]) -> list[ActionPlan]:
        """Coordinate shared page access between plans.

        Args:
            plans: Plans to coordinate

        Returns:
            Modified plans with coordinated access
        """
        # Identify frequently shared pages
        page_usage: dict[str, int] = {}
        for plan in plans:
            for page_id in plan.cache_context.working_set:
                page_usage[page_id] = page_usage.get(page_id, 0) + 1

        # Mark highly shared pages for keeping in cache
        high_usage_threshold = len(plans) // 2
        for plan in plans:
            for page_id in plan.cache_context.working_set:
                if page_usage[page_id] >= high_usage_threshold:
                    # Mark as shareable and high priority
                    if page_id not in plan.cache_context.shareable_pages:
                        plan.cache_context.shareable_pages.append(page_id)
                    plan.cache_context.working_set_priority[page_id] = 2.0  # High priority

        logger.info(f"Coordinated shared pages for {len(plans)} plans")
        return plans


# ============================================================================
# Generic Conflict Resolution
# ============================================================================


class ActionPlanConflictResolver:
    """Generic conflict resolver for plans.

    Resolves conflicts between plans using various strategies:
    - Priority-based resolution/negotiation
    - Temporal staggering
    - Resource efficiency optimization
    - Negotiation between agents
    - Escalation to higher authority
    - Partition working sets
    """

    def __init__(self, default_strategy: str = ConflictResolutionStrategy.PRIORITY):
        """Initialize conflict resolver.

        Args:
            default_strategy: Default resolution strategy to use
        """

        self.default_strategy = default_strategy
        self.cache_resolver = CacheAwarePlanConflictResolver()

    async def resolve_conflict(
        self,
        conflict: ActionPlanConflict,
        plans: list[ActionPlan],
        strategy: str | None = None,
    ) -> list[ActionPlan]:
        """Resolve a conflict between plans.

        Args:
            conflict: Conflict description from detector
            plans: Plans involved in conflict
            strategy: Resolution strategy (uses default if None)

        Returns:
            Modified plans after resolution
        """

        strategy = strategy or self.default_strategy
        conflict_type = conflict.type

        logger.info(
            f"Resolving {conflict_type} conflict using {strategy} strategy"
        )

        if strategy == ConflictResolutionStrategy.PRIORITY:
            return await self._resolve_by_priority(conflict, plans)

        elif strategy == ConflictResolutionStrategy.TEMPORAL:
            return await self._resolve_by_temporal_staggering(conflict, plans)

        elif strategy == ConflictResolutionStrategy.RESOURCE_EFFICIENCY:
            return await self._resolve_by_efficiency(conflict, plans)

        elif strategy == ConflictResolutionStrategy.NEGOTIATION:
            return await self._resolve_by_negotiation(conflict, plans)

        elif strategy == ConflictResolutionStrategy.ESCALATION:
            return await self._resolve_by_escalation(conflict, plans)

        elif strategy == ConflictResolutionStrategy.REPLAN:
            return await self._resolve_by_replanning(conflict, plans)

        else:
            logger.warning(f"Unknown resolution strategy {strategy}, using priority")
            return await self._resolve_by_priority(conflict, plans)

    async def _resolve_by_priority(
        self, conflict: ActionPlanConflict, plans: list[ActionPlan]
    ) -> list[ActionPlan]:
        """Resolve conflict by priority ordering.

        Higher priority plans execute first, lower priority plans delayed or cancelled.

        Args:
            conflict: Conflict description
            plans: Conflicting plans

        Returns:
            Modified plans
        """
        # Sort plans by priority (metadata.priority field)
        sorted_plans = sorted(
            plans,
            key=lambda p: p.metadata.get("priority", 0),
            reverse=True,  # Higher priority first
        )

        # Highest priority plan proceeds unchanged
        # Lower priority plans are delayed or suspended
        for i, plan in enumerate(sorted_plans):
            if i == 0:
                # Highest priority - no changes
                continue
            else:
                # Lower priority - add delay or suspend
                plan.metadata["delayed_by_priority"] = True
                plan.metadata["delay_reason"] = f"Conflict with higher priority plan {sorted_plans[0].plan_id}"

                # Suspend if severe conflict
                severity = conflict.severity
                if severity in [ConflictSeverity.HIGH, ConflictSeverity.CRITICAL]:
                    plan.status = PlanStatus.SUSPENDED
                    plan.blocked_reason = f"Suspended due to {severity} conflict with higher priority plan"

        logger.info(f"Resolved conflict by priority: {len(plans)} plans affected")
        return sorted_plans

    async def _resolve_by_temporal_staggering(
        self, conflict: ActionPlanConflict, plans: list[ActionPlan]
    ) -> list[ActionPlan]:
        """Resolve conflict by staggering execution over time.

        Args:
            conflict: Conflict description
            plans: Conflicting plans

        Returns:
            Modified plans with execution delays
        """
        # Use cache resolver's staggering logic
        return await self.cache_resolver._stagger_execution(plans)

    async def _resolve_by_efficiency(
        self, conflict: ActionPlanConflict, plans: list[ActionPlan]
    ) -> list[ActionPlan]:
        """Resolve conflict by optimizing resource usage.

        Args:
            conflict: Conflict description
            plans: Conflicting plans

        Returns:
            Modified plans optimized for efficiency
        """

        conflict_type = conflict.type

        if conflict_type == ConflictType.CACHE_CONTENTION:
            # Use cache-specific resolution
            return await self.cache_resolver._partition_working_sets(plans)

        elif conflict_type == ConflictType.RESOURCE_CONTENTION:
            # Generic resource optimization - share resources when possible
            for plan in plans:
                # Mark resources as shareable
                plan.metadata["resource_sharing_enabled"] = True

        logger.info("Resolved conflict by efficiency optimization")
        return plans

    async def _resolve_by_negotiation(
        self, conflict: ActionPlanConflict, plans: list[ActionPlan]
    ) -> list[ActionPlan]:
        """Resolve conflict through negotiation between agents.

        Args:
            conflict: Conflict description
            plans: Conflicting plans

        Returns:
            Modified plans after negotiation
        """
        # TODO: Implement negotiation protocol
        # For now, fall back to priority
        logger.warning("Negotiation not implemented, falling back to priority")
        return await self._resolve_by_priority(conflict, plans)

    async def _resolve_by_escalation(
        self, conflict: ActionPlanConflict, plans: list[ActionPlan]
    ) -> list[ActionPlan]:
        """Escalate conflict to higher authority.

        Args:
            conflict: Conflict description
            plans: Conflicting plans

        Returns:
            Plans marked for escalation
        """
        # Mark all plans as needing approval
        for plan in plans:
            plan.approval_required = True
            plan.metadata["escalated_conflict"] = True
            plan.metadata["conflict_details"] = conflict
            plan.status = PlanStatus.PROPOSED  # Wait for approval

        logger.info(f"Escalated conflict involving {len(plans)} plans")
        return plans

    async def _resolve_by_replanning(
        self, conflict: ActionPlanConflict, plans: list[ActionPlan]
    ) -> list[ActionPlan]:
        """Resolve conflict by requesting replanning.

        Args:
            conflict: Conflict description
            plans: Conflicting plans

        Returns:
            Plans marked for replanning
        """
        # Mark lower priority plans for replanning
        sorted_plans = sorted(
            plans,
            key=lambda p: p.metadata.get("priority", 0),
            reverse=True,
        )

        for i, plan in enumerate(sorted_plans):
            if i > 0:  # Skip highest priority plan
                plan.metadata["requires_replanning"] = True
                plan.metadata["replan_reason"] = f"Conflict: {conflict.description}"

        logger.info(f"Marked {len(plans) - 1} plans for replanning")
        return sorted_plans


# ============================================================================
# Coordination capability
# ============================================================================

class PlanCoordinationCapability(AgentCapability):
    """Multi-agent plan coordination for shared VCM context.

    Detects conflicts between an agent's plan and other agents' active
    plans (cache contention, resource exhaustion), and resolves them
    using configurable strategies (stagger, partition, negotiate).

    Usage::

        # Programmatic API
        conflicts = await cap.check_conflicts(my_plan, sibling_plans)
        resolved = await cap.resolve_conflict(my_plan, conflicts[0])

        # LLM API
        # check_plan_conflicts, get_sibling_plans, propose_plan, resolve_contention
    """

    def __init__(
        self,
        agent: Agent,
        scope: BlackboardScope = BlackboardScope.COLONY,
        namespace: str = "plan_coordination",
        cache_capacity: int = 100,
        input_patterns: list[str] | None = None,
        capability_key: str = "plan_coordination",
        app_name: str | None = None,
    ):
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            input_patterns=input_patterns or [],
            capability_key=capability_key,
            app_name=app_name,
        )
        self.cache_capacity = cache_capacity
        self._detector = None
        self._resolver = None
        self._plan_blackboard = None

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"planning"})

    def get_action_group_description(self) -> str:
        return (
            "Plan Coordination — detects and resolves conflicts between agents' plans. "
            "Checks for cache contention and resource exhaustion across sibling agents, "
            "resolves via staggering, partitioning, or negotiation. "
            "Publishes plans for colony-wide visibility."
        )

    async def _ensure_initialized(self) -> None:
        """Lazily initialize coordination components."""
        if self._detector is not None:
            return

        from .coordination import (
            CacheAwarePlanConflictDetector,
            ActionPlanConflictResolver,
        )
        from ..blackboard import PlanBlackboard

        self._detector = CacheAwarePlanConflictDetector(self.cache_capacity)
        self._resolver = ActionPlanConflictResolver(default_strategy="priority")
        self._plan_blackboard = PlanBlackboard(
            agent=self.agent,
            scope=BlackboardScope.COLONY,
            namespace="action_plans",
        )
        await self._plan_blackboard.initialize()

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        pass

    # =========================================================================
    # Programmatic API
    # =========================================================================

    async def check_conflicts(
        self, plan: ActionPlan, other_plans: list[ActionPlan] | None = None
    ) -> list[ActionPlanConflict]:
        """Check if a plan conflicts with other agents' active plans.

        Args:
            plan: The plan to check.
            other_plans: Sibling plans. If None, fetches from plan blackboard.

        Returns:
            List of conflict dicts with type, severity, description, and
            recommended_strategy.
        """
        logger.info(f"Checking conflicts for plan {plan.plan_id}")
        await self._ensure_initialized()

        if other_plans is None and self._plan_blackboard:
            all_plans = await self._plan_blackboard.get_all_plans()
            other_plans = [p for p in all_plans if p.agent_id != plan.agent_id]

        if not other_plans:
            return []

        conflicts = self._detector.detect_conflicts([plan] + other_plans)
        logger.info(f"Found {len(conflicts)} conflicts for plan {plan.plan_id}")
        return conflicts

    async def resolve_conflict(
        self, plan: ActionPlan, conflict: ActionPlanConflict
    ) -> ActionPlan | None:
        """Resolve a detected conflict.

        Args:
            plan: The plan to adjust.
            conflict: Conflict dict from check_conflicts().

        Returns:
            Adjusted plan, or None if resolution failed.
        """
        conflict_type = conflict.type
        logger.info(f"Resolving {conflict_type} conflict for plan {plan.plan_id}")

        await self._ensure_initialized()

        # Use existing conflict resolver
        strategy = conflict.recommended_strategy
        resolved_plans = await self._resolver.resolve_conflict(
            conflict=conflict,
            plans=[plan],
            strategy=strategy,
        )
        if resolved_plans:
            resolved_plan = resolved_plans[0]
            logger.info(f"Resolved conflict for plan {plan.plan_id} using {strategy} strategy")
            return resolved_plan

        logger.warning(f"Could not resolve conflict for plan {plan.plan_id}")
        return None

    # =========================================================================
    # LLM API (@action_executor)
    # =========================================================================

    @action_executor(
        planning_summary=(
            "Check if the current working set conflicts with other agents' "
            "plans in the colony. Returns conflicts with severity and resolution strategies."
        ),
    )
    async def check_plan_conflicts(self) -> dict[str, Any]:
        """Check for conflicts between this agent's plan and sibling plans.

        Returns:
            Dict with 'conflicts' list and 'has_conflicts' flag.
        """
        await self._ensure_initialized()

        if not self._plan_blackboard:
            return {"conflicts": [], "has_conflicts": False}

        my_plan = await self._plan_blackboard.get_plan(self.agent.agent_id)
        if not my_plan:
            return {"conflicts": [], "has_conflicts": False, "note": "No active plan"}

        conflicts = await self.check_conflicts(my_plan)
        return {
            "conflicts": conflicts,
            "has_conflicts": len(conflicts) > 0,
            "count": len(conflicts),
        }

    @action_executor(
        planning_summary=(
            "Get active plans of sibling agents in the colony. "
            "Useful for understanding what other agents are doing."
        ),
    )
    async def get_sibling_plans(self) -> dict[str, Any]:
        """Get plans of other agents in the colony.

        Returns:
            Dict with 'plans' list containing agent_id, goals, status,
            working_set_size for each sibling.
        """
        await self._ensure_initialized()

        if not self._plan_blackboard:
            return {"plans": [], "count": 0}

        all_plans = await self._plan_blackboard.get_all_plans()
        sibling_plans = [p for p in all_plans if p.agent_id != self.agent.agent_id]

        return {
            "plans": [
                {
                    "agent_id": p.agent_id,
                    "goals": p.goals[:3],
                    "status": p.status.value if hasattr(p.status, 'value') else str(p.status),
                    "action_count": len(p.actions),
                    "working_set_size": len(p.cache_context.working_set) if p.cache_context else 0,
                }
                for p in sibling_plans
            ],
            "count": len(sibling_plans),
        }

    @action_executor(
        planning_summary=(
            "Propose the current plan for approval by the parent agent. "
            "Publishes to the colony plan blackboard."
        ),
    )
    async def propose_plan(self, description: str) -> dict[str, Any]:
        """Publish the current plan for colony-wide visibility.

        Args:
            description: Human-readable plan description.

        Returns:
            Dict with 'proposed' flag and approval status.
        """
        await self._ensure_initialized()

        if not self._plan_blackboard:
            return {"proposed": False, "error": "No plan blackboard"}

        my_plan = await self._plan_blackboard.get_plan(self.agent.agent_id)
        if not my_plan:
            return {"proposed": False, "error": "No active plan"}

        my_plan.description = description
        approved, message = await self._plan_blackboard.propose_plan(
            my_plan, requesting_agent_id=self.agent.agent_id
        )
        return {"proposed": True, "approved": approved, "message": message}

    @action_executor(
        planning_summary=(
            "Resolve cache contention with another agent by adjusting "
            "working set overlap (stagger, partition, or negotiate)."
        ),
    )
    async def resolve_contention(
        self,
        conflicting_agent_id: str,
        strategy: str = "stagger",
    ) -> dict[str, Any]:
        """Resolve cache contention with a specific agent.

        Args:
            conflicting_agent_id: The agent causing the conflict.
            strategy: Resolution strategy ('stagger', 'partition', 'priority').

        Returns:
            Dict with 'resolved' flag and details.
        """
        await self._ensure_initialized()

        if not self._plan_blackboard:
            return {"resolved": False, "error": "No plan blackboard"}

        my_plan = await self._plan_blackboard.get_plan(self.agent.agent_id)
        their_plan = await self._plan_blackboard.get_plan(conflicting_agent_id)

        if not my_plan or not their_plan:
            return {"resolved": False, "error": "Plan not found"}

        conflicts = await self.check_conflicts(my_plan, [their_plan])
        if not conflicts:
            return {"resolved": True, "note": "No conflicts detected"}

        resolved = await self.resolve_conflict(my_plan, conflicts[0])
        if resolved:
            await self._plan_blackboard.update_plan(resolved)
            return {"resolved": True, "strategy": strategy}

        return {"resolved": False, "error": "Resolution failed"}


