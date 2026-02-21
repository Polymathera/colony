"""Planner class for creating and executing plans.

This module provides the core Planner class that:
- Creates plans (LLM-generated or manual)
- Executes plans incrementally (model-predictive control)
- Handles replanning when needed
"""

import logging
import time
from typing import Any
from abc import ABC, abstractmethod
from overrides import override

from ...base import Agent
from ...models import (
    Action,
    ActionPlan,
    ActionType,
    ActionResult,
    ActionStatus,
    CacheContext,
    PlanningContext,
    PlanningParameters,
    PlanStatus,
)
from .strategies import PlanningStrategyPolicy, TopDownPlanningStrategy
from .policies import (
    CacheAwarePlanningPolicy,
    LearningPlanningPolicy,
    CoordinationPlanningPolicy,
)
from .blackboard import PlanBlackboard
from ..models import Critique


logger = logging.getLogger(__name__)



class ActionPlanner(ABC):
    """Policy for creating and revising agent action plans.

    Different implementations:
    - SequentialPlanner: Linear sequence of actions
    - HierarchicalPlanner: Nested sub-plans
    - ReactivePlanner: Generate next action on-the-fly
    - LLMPlanner: Use LLM to generate plans
    """

    @abstractmethod
    async def create_plan(self, planning_context: PlanningContext) -> ActionPlan:
        """Create initial plan to achieve goal.

        Args:
            planning_context: Current context (goals, constraints, available resources, state, custom data, etc.)
        Returns:
            ActionPlan with ordered actions
        """
        ...

    @abstractmethod
    async def revise_plan(
        self, current_plan: ActionPlan, planning_context: PlanningContext, critique: Critique
    ) -> ActionPlan:
        """Revise plan based on critique and new information.

        Args:
            current_plan: Current plan
            planning_context: Current context (goals, constraints, available resources, state, custom data, etc.)
            critique: Critique suggesting revision

        Returns:
            Revised plan
        """
        # TODO: Remove critique? It should be part of planning_context?
        ...

    @abstractmethod
    async def learn_from_plan_execution(self, plan: ActionPlan) -> None:
        """Learn from completed plan execution.

        Args:
            plan: Completed plan to learn from
        """
        pass



# ============================================================================
# Default Implementations
# ============================================================================


class SequentialPlanner(ActionPlanner):
    """Simple manually-specified sequential planner for straightforward tasks.

    Creates linear sequence of actions.
    """

    def __init__(
        self,
        agent: Agent,
        planning_params: PlanningParameters,
    ):
        """Initialize with optional action templates.

        Args:
            agent: Agent instance
            planning_params: Planning parameters
            action_templates: Predefined action sequence
        """
        self.agent = agent
        self.planning_params = planning_params

    @override
    async def create_plan(self, planning_context: PlanningContext) -> ActionPlan:
        """Create manually-specified sequential plan."""
        if not planning_context.manual_plan:
            raise ValueError("manual_plan must be provided in context for manual plan creation")

        return ActionPlan(
            plan_id=f"plan_{self.agent.agent_id}_{int(time.time() * 1000)}",
            agent_id=self.agent.agent_id,
            goals=planning_context.goals,
            constraints=planning_context.constraints,
            generation_method="manual",
            strategy="sequential",
            actions=[Action(**a) for a in planning_context.manual_plan.actions],
            planning_horizon=self.planning_params.planning_horizon,
            replan_every_n_steps=self.planning_params.replan_every_n_steps,
            parent_plan_id=planning_context.parent_plan_id,
        )

    @override
    async def revise_plan(
        self, current_plan: ActionPlan, planning_context: PlanningContext, critique: Critique
    ) -> ActionPlan:
        """Revise plan by appending corrective actions."""
        # Simple strategy: add actions based on critique suggestions
        corrective_actions = []
        for suggestion in critique.suggestions:
            # Convert suggestion to action (simple heuristic)
            corrective_actions.append(
                Action(
                    type=ActionType.CUSTOM,  # TODO: How are these actions handled by executor?
                    parameters={"suggestion": suggestion},
                    reasoning=f"Addressing critique: {suggestion}",
                )
            )

        # Prepend corrective actions (handle issues first)
        for action in reversed(corrective_actions):
            current_plan.prepend_action(action)

        return current_plan

    @override
    async def learn_from_plan_execution(self, plan: ActionPlan) -> None:
        raise NotImplementedError("SequentialPlanner does not support learning from execution.")



class LLMPlanner(ActionPlanner):
    """Planner that uses LLM to generate plans.

    More sophisticated than sequential planner - can adapt to context.
    """

    def __init__(self, agent: Agent):
        """Initialize LLM planner.

        Args:
            agent: Agent instance for LLM inference
        """
        self.agent = agent

    @override
    async def learn_from_plan_execution(self, plan: ActionPlan) -> None:
        raise NotImplementedError("SequentialPlanner does not support learning from execution.")

    @override
    async def create_plan(self, planning_context: PlanningContext) -> ActionPlan:
        """Use LLM to generate plan."""
        prompt = self._build_planning_prompt(planning_context)

        response = await self.agent.infer(
            prompt=prompt,
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
            max_tokens=1000,  # TODO: Make configurable
            temperature=0.3,  # More deterministic - TODO: Make configurable
            json_schema=ActionPlan.model_json_schema()
        )

        # Parse LLM response into actions
        # TODO: Does this correctly populate action fields correctly?
        # TODO: What if one of the choices is an ActionPolicy? The JSON parsing won't
        # correctly link to the intended ActionPolicy instance.
        # TODO: Handle validation errors. LLMs are not perfect.
        plan = ActionPlan.model_validate_json(
            response.generated_text,
        )

        plan.goals = planning_context.goals
        plan.strategy = "llm-generated"
        plan.constraints = planning_context.constraints
        plan.agent_id = self.agent.agent_id
        plan.created_at = time.time()
        return plan

    def _build_planning_prompt(self, planning_context: PlanningContext) -> str:
        """Build prompt for planning."""
        # Extract structured context for easier reasoning
        context_summary = self._summarize_context(planning_context)

        action_descriptions = ""
        for group_desc, group_actions in planning_context.action_descriptions:
            if group_desc:
                action_descriptions += f"\n### {group_desc}\n"
            for action_key, action_desc in group_actions.items():
                action_descriptions += f"- {action_key}: {action_desc}\n"

        # Include memory architecture guidance if available
        memory_guidance = planning_context.custom_data.get("memory_architecture_guidance", "")
        memory_section = ""
        if memory_guidance:
            memory_section = f"\n{memory_guidance}\n"

        return f"""Create a plan to achieve these goals:
Goals:
{planning_context.goals}

Current state:
{context_summary}

Constraints:
{planning_context.constraints}

Available actions:
{action_descriptions}
- custom: Domain-specific action
{memory_section}
Query workflow - you control the pace:
1. generate_queries → creates list of queries from analyzed pages
2. route_query → for EACH query, find relevant pages (expensive! returns attention scores)
3. **Decision point**: Look at attention scores and total_candidates:
   - High scores (>0.7)? Process those pages first (process_query with top_pages)
   - Low scores (<0.5)? Maybe skip or try different query
   - Many candidates? Start with subset, evaluate answer quality, then decide if more needed
4. process_query → Load selected pages, get LLM answer (expensive!)
5. **Decision point**: Check answer confidence and needs_more_pages:
   - High confidence + no additional pages needed? Move to next query
   - Low confidence or needs_more_pages? Route to additional pages or refine query
6. Iterate: Use answers to generate follow-up queries if needed

Context interpretation:
- last_query_routing: Contains page IDs, attention scores, and top_pages from latest route_query
  → Use "top_pages" for initial processing, expand if answer unsatisfactory
  → Check "attention_scores" to see confidence in each page's relevance
- last_query_answer: Contains answer, confidence, evidence, pages_used from latest process_query
  → Check "confidence" (high/medium/low) and whether "additional_pages_needed" is empty
  → Use "evidence" to see which pages contributed to answer
- answer_satisfactory: Boolean indicating if last answer was sufficient
  → If false, consider processing more pages from last_query_routing
- pending_queries: List of queries that still need routing/processing

Output a list of actions in JSON format. Each action should have:
- type: One of {[t.value for t in ActionType]}
- parameters: Dict of parameters
  Examples:
  - route_query: {{"query": <query_object>}}
  - process_query: {{"query": <query_object>, "page_ids": ["page_1", "page_2"]}}
- reasoning: Why this action? What are you evaluating?

Be strategic:
- Don't process all routed pages at once - start small, evaluate, expand if needed
- Check attention scores before deciding how many pages to load
- Use answer confidence to decide whether to continue or move on
- Batch cheap operations (routing), but be selective with expensive ones (processing)

## Distributed VCM Analysis (for code analysis tasks)

When using VCM analysis capabilities (IntentAnalysis, ContractAnalysis, SlicingAnalysis,
ComplianceVCM), you have composable primitives. YOU decide the strategy:

**Primitives you control:**
- Worker lifecycle: spawn_worker, spawn_workers, terminate_worker, get_idle_workers
- Work assignment: assign_work, prioritize_work, get_pending_work
- Results: get_result, merge_results, detect_contradictions, synthesize_results
- State queries: get_analyzed_pages, get_pages_with_issues, get_outstanding_queries
- Iteration: mark_for_revisit, revisit_page, clear_result

**Example strategies (choose based on data patterns):**

1. **Cluster-Based**: For related pages that benefit from being analyzed together
   - Use page_graph.get_clusters() to find clusters
   - Spawn workers with cache_affine=True for each cluster
   - Merge results after each cluster completes
   - Good when: Pages have clear dependency structure

2. **Query-Driven Continuous**: For exploratory analysis guided by questions
   - Check get_outstanding_queries() for pending questions
   - Spawn workers for pages that might answer top queries
   - Detect contradictions as results come in
   - Good when: Analysis is question-driven, not coverage-driven

3. **Opportunistic with Revisits**: For fast initial pass with refinement
   - Spawn workers for all pages (with max_parallel limit)
   - Check get_pages_with_issues() for low-confidence results
   - Mark issues for revisit with new context
   - Good when: Need fast initial results, can refine later

**Cache-awareness is EMERGENT from your choices:**
- Set cache_affine=True to place workers near cached pages
- Use working_set.request_pages() before spawn_workers() to pre-warm cache
- Use page_graph.get_clusters() to find pages that benefit from co-location"""

    def _summarize_context(self, planning_context: PlanningContext) -> str:
        """Summarize context for LLM planner.

        Extracts key information in a structured, readable format.
        """
        summary_parts = []

        # TODO: Update this to extract more relevant PlanningContext fields as needed.
        # TODO: This method is using many non-existent fields. Update to use existing ones.

        # Pages analyzed
        pages_analyzed = planning_context.current_state.get("pages_analyzed_count", 0) if planning_context.current_state else 0
        if pages_analyzed > 0:
            summary_parts.append(f"Pages analyzed: {pages_analyzed}")

        # Pending queries
        pending_queries = planning_context.pending_queries
        if pending_queries:
            summary_parts.append(f"\nPending queries ({len(pending_queries)}):")
            for i, q in enumerate(pending_queries[:5], 1):  # Show first 5
                query_text = q.get("query", str(q))
                summary_parts.append(f"  {i}. {query_text}")
            if len(pending_queries) > 5:
                summary_parts.append(f"  ... and {len(pending_queries) - 5} more")

        # Last query routing result
        last_routing = planning_context.last_query_routing
        if last_routing:
            query = last_routing.get("query", "unknown")
            total_candidates = last_routing.get("total_candidates", 0)
            top_pages = last_routing.get("top_pages", [])
            scores = last_routing.get("attention_scores", {})

            summary_parts.append(f"\nLast query routed: '{query}'")
            summary_parts.append(f"  Found {total_candidates} candidates")
            if top_pages and scores:
                summary_parts.append(f"  Top pages (with scores):")
                for page_id in top_pages[:3]:  # Show top 3
                    score = scores.get(page_id, 0)
                    summary_parts.append(f"    - {page_id}: {score:.2f}")

        # Last query answer
        last_answer = planning_context.last_query_answer
        if last_answer:
            query = last_answer.get("query", "unknown")
            answer_data = last_answer.get("answer", {})
            if isinstance(answer_data, dict):
                confidence = answer_data.get("confidence", "unknown")
                needs_more = answer_data.get("additional_pages_needed", [])
                answer_text = answer_data.get("answer", "")
            else:
                confidence = "unknown"
                needs_more = []
                answer_text = str(answer_data)

            pages_used = last_answer.get("pages_used_count", 0)

            summary_parts.append(f"\nLast query answered: '{query}'")
            summary_parts.append(f"  Confidence: {confidence}, Pages used: {pages_used}")
            if needs_more:
                summary_parts.append(f"  Needs more pages: {needs_more[:3]}")
            if answer_text:
                preview = answer_text[:100] + "..." if len(answer_text) > 100 else answer_text
                summary_parts.append(f"  Answer: {preview}")

        # Answer satisfactory flag
        if "answer_satisfactory" in planning_context:
            satisfactory = planning_context["answer_satisfactory"]
            summary_parts.append(f"\nLast answer satisfactory: {satisfactory}")

        # Synthesis status
        if planning_context.get("synthesis_complete"):
            summary_parts.append("\nSynthesis: Complete")

        if not summary_parts:
            return "No context available yet - starting fresh"

        return "\n".join(summary_parts)

    @override
    async def revise_plan(
        self, current_plan: ActionPlan, planning_context: PlanningContext, critique: Critique
    ) -> ActionPlan:
        """Use LLM to revise plan based on critique."""
        prompt = self._build_revision_prompt(current_plan, planning_context, critique)

        response = await self.agent.infer(
            prompt=prompt,
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
            max_tokens=1000,  # TODO: Make configurable
            temperature=0.3,  # More deterministic - TODO: Make configurable
            json_schema=ActionPlan.model_json_schema()
        )

        # Parse LLM response into actions
        revised_plan = ActionPlan.model_validate_json(  # TODO: Handle validation errors. LLMs are not perfect.
            response.generated_text,
        )

        # Combine with remaining actions from current plan
        current_plan.actions = revised_plan.actions + current_plan.actions
        current_plan.updated_at = time.time()
        return current_plan

    def _build_revision_prompt(
        self, plan: ActionPlan, planning_context: PlanningContext, critique: Critique
    ) -> str:
        """Build prompt for plan revision."""
        # TODO: This prompt is very weak. Needs improvement.
        # TODO: It does not even provide action descriptions or context summary.

        # TODO: The following context fields are not yet included:
        # - planning_context.action_descriptions
        # - planning_context.custom_data # Includes memory architecture
        # - planning_context.constraints
        # - planning_context.recalled_memories

        action_descriptions = ""
        for group_desc, group_actions in planning_context.action_descriptions:
            if group_desc:
                action_descriptions += f"\n### {group_desc}\n"
            for action_key, action_desc in group_actions.items():
                action_descriptions += f"- {action_key}: {action_desc}\n"

        # Include memory architecture guidance if available
        memory_guidance = planning_context.custom_data.get("memory_architecture_guidance", "")
        memory_section = f"\n{memory_guidance}\n" if memory_guidance else ""

        return f"""Revise this plan based on critique.

Original goals: {plan.goals}

Current remaining actions:
{[a.model_dump() for a in plan.actions]}

Critique:
- Quality score: {critique.quality_score}
- Issues: {critique.issues}
- Suggestions: {critique.suggestions}

New information learned:
{planning_context.model_dump_json(indent=2)}

Available actions:
{action_descriptions}
{memory_section}

Output revised action list in JSON format."""


class CacheAwareActionPlanner(ActionPlanner):
    """ONE planner class, customized via pluggable policies."""

    def __init__(
        self,
        agent: Agent,
        planning_strategy: PlanningStrategyPolicy,
        planning_params: PlanningParameters,
        cache_policy: CacheAwarePlanningPolicy | None = None,
        learning_policy: LearningPlanningPolicy | None = None,
        coordination_policy: CoordinationPlanningPolicy | None = None, # FIXME: Not used yet
    ):
        self.agent = agent
        self.agent_id = agent.agent_id
        self.planning_strategy = planning_strategy

        # Ensure strategy has agent reference
        if not planning_strategy.agent:
            planning_strategy.set_agent(agent)

        self.planning_params = planning_params
        self.cache_policy = cache_policy
        self.learning_policy = learning_policy
        self.coordination_policy = coordination_policy

    async def initialize(self) -> None:
        """Initialize planner and its policies."""
        if self.cache_policy:
            await self.cache_policy.initialize(self.agent, self.planning_params)

        if self.learning_policy:
            await self.learning_policy.initialize(self.agent, self.planning_params)

        if self.coordination_policy:
            await self.coordination_policy.initialize(self.agent, self.planning_params)

    @override
    async def create_plan(self, planning_context: PlanningContext) -> ActionPlan:
        """Create plan (LLM-generated or manual).

        Args:
            goals: List of goals to achieve
            planning_context: Planning context (goals, constraints, resources, etc.)
        """
        # Apply learning policy (get similar successful plans)
        learned_patterns = None
        if self.learning_policy:
            learned_patterns = await self.learning_policy.get_applicable_patterns(planning_context)

        # Apply cache policy (analyze cache requirements)
        cache_context = CacheContext()
        if self.cache_policy:
            cache_context = await self.cache_policy.analyze_cache_requirements(planning_context)

        # Generate plan via strategy
        logger.warning(
            f"\n"
            f"          ╔══════════════════════════════════════╗\n"
            f"          ║  🎯 PLANNER: calling strategy        ║\n"
            f"          ║  {self.planning_strategy.__class__.__name__:<36}║\n"
            f"          ╚══════════════════════════════════════╝"
        )
        plan: ActionPlan = await self.planning_strategy.generate_plan(
            planning_context=planning_context,
            params=self.planning_params,
            learned_patterns=learned_patterns,
            cache_context=cache_context,
        )
        logger.warning(
            f"          🎯 PLANNER: strategy returned plan with "
            f"{len(plan.actions)} actions, status={plan.status}"
        )

        # Set parent relationship
        plan.agent_id = self.agent_id
        plan.parent_plan_id = planning_context.parent_plan_id
        plan.cache_context = cache_context

        return plan

    @override
    async def revise_plan(
        self, current_plan: ActionPlan, planning_context: PlanningContext, critique: Critique
    ) -> ActionPlan:
        """Replan next N steps (MPC)."""
        # Get learned patterns and cache context if policies are available
        learned_patterns = None
        if self.learning_policy:
            learned_patterns = await self.learning_policy.get_applicable_patterns(
                current_plan.goals, planning_context
            )

        cache_context = current_plan.cache_context if hasattr(current_plan, "cache_context") else None
        if not cache_context and self.cache_policy:
            cache_context = await self.cache_policy.analyze_cache_requirements(planning_context)
        else:
            cache_context = CacheContext()

        new_actions = await self.planning_strategy.replan_horizon(
            plan=current_plan,
            planning_context=planning_context,
            params=self.planning_params,
            learned_patterns=learned_patterns,
            cache_context=cache_context,
        )

        # Replace future actions
        current_plan.actions = (
            current_plan.actions[: current_plan.current_action_index] + new_actions
        )
        return current_plan

    @override
    async def learn_from_plan_execution(self, plan: ActionPlan) -> None:
        """Learn from completed plan execution.

        Args:
            plan: Completed plan to learn from
        """
        # Calculate outcome metrics
        completed_actions = [
            a for a in plan.actions if a.status == ActionStatus.COMPLETED
        ]
        failed_actions = [a for a in plan.actions if a.status == ActionStatus.FAILED]

        success_rate = (
            len(completed_actions) / len(plan.actions) if plan.actions else 0.0
        )

        # Determine overall outcome
        if plan.status == PlanStatus.COMPLETED:
            outcome_status = "success"
        elif plan.status == PlanStatus.FAILED:
            outcome_status = "failed"
        else:
            outcome_status = "partial"

        # Calculate duration
        duration_s = (
            plan.completed_at - plan.created_at if plan.completed_at else 0.0
        )

        # Calculate quality score (simple heuristic)
        quality_score = success_rate
        if plan.execution_context.findings:
            # Bonus for gathering information
            quality_score = min(1.0, quality_score + 0.1)

        # Build outcome dictionary
        outcome = {
            "status": outcome_status,
            "duration_s": duration_s,
            "success_rate": success_rate,
            "actions_completed": len(completed_actions),
            "actions_failed": len(failed_actions),
            "quality_score": quality_score,
            "actual_cost": {
                "pages_loaded": len(plan.cache_context.working_set),
                "actions_executed": len(completed_actions),
                "children_spawned": len(plan.execution_context.spawned_children),
            },
        }

        # Let learning policy record and learn
        logger.info(
            f"Learning from plan {plan.plan_id}: {outcome_status} with {success_rate:.1%} success rate"
        )
        await self.learning_policy.learn_from_execution(plan, outcome)





async def create_cache_aware_planner(
    agent: Agent,
    max_iterations: int = 5,
    quality_threshold: float = 0.8,
    planning_horizon: int = 5,
    ideal_cache_size: int = 10,
) -> CacheAwareActionPlanner:
    """Create sophisticated planner with cache-awareness and learning.

    Returns:
        CacheAwareActionPlanner wrapping planning.Planner
    """

    # Create planning parameters
    planning_params = PlanningParameters(
        planning_horizon=planning_horizon,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        ideal_cache_size=ideal_cache_size,
    )

    # Create planning strategy (TopDown works well for code analysis)
    planning_strategy = TopDownPlanningStrategy(agent=agent)

    # Create sophisticated planner with policies
    cache_aware_planner = CacheAwareActionPlanner(
        agent=agent,
        planning_strategy=planning_strategy,
        planning_params=planning_params,
        # Policies will be created automatically in Agent.initialize()
        # when metadata doesn't provide them
    )
    await cache_aware_planner.initialize()

    return cache_aware_planner




