"""Code-generation-based action policy.

Instead of selecting actions from a JSON schema, the LLM generates Python code
that composes ``@action_executor`` methods with real control flow, conditionals,
loops, and data transformation. This turns the action planning problem into a
code generation problem — something LLMs excel at.

The generated code runs in the existing ``PolicyPythonREPL`` with an enriched
namespace that provides structured access to agent capabilities:

- ``run(action_key, **params)`` — dispatch an action through the action dispatcher
- ``browse(query=None)`` — progressive capability discovery
- ``bb`` — the agent's primary blackboard
- ``results`` — dict of prior action results
- ``pages`` — current working set page IDs
- ``goals`` — agent's current goals
- ``log(msg)`` — structured logging

Usage::

    from polymathera.colony.agents.patterns.actions.code_generation import (
        CodeGenerationActionPolicy,
    )

    # Use as the agent's action policy
    agent.action_policy = CodeGenerationActionPolicy(agent=agent)

    # Or via blueprint
    policy_blueprint = CodeGenerationActionPolicy.bind(
        max_retries=2,
        code_timeout=30.0,
    )
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any

from overrides import override

from ...base import Agent
from ...blueprint import ActionPolicyBlueprint, AgentCapabilityBlueprint
from ...models import (
    Action,
    ActionType,
    ActionResult,
    ActionPolicyExecutionState,
    ActionPolicyIterationResult,
    ActionPolicyIO,
)
from .dispatcher import ActionGroup, ActionDispatcher
from .policies import (
    BaseActionPolicy,
    EventDrivenActionPolicy,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Capability browser — progressive discovery
# ---------------------------------------------------------------------------

class CapabilityBrowser:
    """Progressive capability discovery for LLM-generated code.

    Provides a hierarchical view of available capabilities that the LLM
    can query at any level of detail:

    - ``browse()`` — returns all capability groups with one-line summaries
    - ``browse("group_name")`` — returns action names + signatures for a group
    - ``browse("group_name.action_name")`` — returns full docstring + source
    - ``browse("programmatic")`` — returns programmatic API methods on all
      capabilities (not just ``@action_executor`` — the full Python methods
      that generated code can call directly)

    The ``"programmatic"`` query is unique to ``CodeGenerationActionPolicy`` —
    it exposes the richer programmatic APIs that ``CacheAwareActionPlanner``
    uses internally, making them available to LLM-generated code.
    """

    def __init__(self, dispatcher: ActionDispatcher, agent: Agent | None = None):
        self._dispatcher = dispatcher
        self._agent = agent

    async def __call__(self, query: str | None = None) -> dict[str, Any] | str:
        """Browse available capabilities.

        Args:
            query: None for top-level groups, group name for group detail,
                   "group.action" for full action detail,
                   "programmatic" for all programmatic methods on capabilities.

        Returns:
            Structured capability information.
        """
        if query is None:
            return await self._list_groups()

        if query == "programmatic":
            return await self._list_programmatic_apis()

        if "." in query:
            group_key, action_key = query.split(".", 1)
            return await self._action_detail(group_key, action_key)

        return await self._group_detail(query)

    async def _list_groups(self) -> dict[str, Any]:
        """Top-level: list all capability groups with summaries."""
        groups = {}
        for key, group in self._dispatcher.action_map.items():
            action_names = [
                name for name, ex in group.executors.items()
                if not getattr(ex, 'exclude_from_planning', False)
            ]
            groups[key] = {
                "description": group.description or "",
                "actions": action_names,
                "count": len(action_names),
            }
        # Add hint about programmatic APIs
        groups["_hint"] = {
            "description": "Use browse('programmatic') to see full Python APIs on capabilities",
            "actions": [],
            "count": 0,
        }
        return groups

    async def _list_programmatic_apis(self) -> dict[str, Any]:
        """List all public async methods on registered capabilities.

        This goes beyond ``@action_executor`` — it shows the full programmatic
        API that generated code can call directly via ``_agent.get_capability_by_type()``.
        """
        if not self._agent:
            return {"error": "No agent reference — programmatic API browsing unavailable"}

        apis: dict[str, Any] = {}
        for cap in self._agent.get_capabilities():
            cap_name = cap.__class__.__name__
            methods = {}

            for name in dir(cap):
                if name.startswith("_"):
                    continue
                attr = getattr(cap, name, None)
                if not callable(attr) or not inspect.iscoroutinefunction(attr):
                    continue
                # Skip inherited AgentCapability methods
                if name in ("initialize", "get_blackboard", "stream_events_to_queue",
                            "get_result_future", "send_request", "serialize_suspension_state",
                            "deserialize_suspension_state"):
                    continue

                sig = inspect.signature(attr)
                doc = inspect.getdoc(attr) or "(no docstring)"
                methods[name] = f"{name}{sig}\n{doc}"

            if methods:
                apis[cap_name] = {
                    "description": cap.get_action_group_description() if hasattr(cap, 'get_action_group_description') else "",
                    "methods": methods,
                    "access": f"_agent.get_capability_by_type({cap_name})",
                }

        return apis

    async def _group_detail(self, group_key: str) -> dict[str, Any]:
        """Group-level: list actions with signatures and summaries."""
        group = self._dispatcher.action_map.get(group_key)
        if not group:
            # Fuzzy match
            for key, g in self._dispatcher.action_map.items():
                if group_key.lower() in key.lower():
                    group = g
                    break
            if not group:
                return {"error": f"No capability group matching '{group_key}'"}

        actions = {}
        for name, executor in group.executors.items():
            if getattr(executor, 'exclude_from_planning', False):
                continue
            desc = await executor.get_action_description() if hasattr(executor, 'get_action_description') else str(executor)
            actions[name] = desc
        return actions

    async def _action_detail(self, group_key: str, action_key: str) -> str:
        """Action-level: full docstring and signature."""
        group = self._dispatcher.action_map.get(group_key)
        if not group:
            for key, g in self._dispatcher.action_map.items():
                if group_key.lower() in key.lower():
                    group = g
                    break
            if not group:
                return f"No capability group matching '{group_key}'"

        executor = group.executors.get(action_key)
        if not executor:
            for name, ex in group.executors.items():
                if action_key.lower() in name.lower():
                    executor = ex
                    break
            if not executor:
                return f"No action matching '{action_key}' in group '{group_key}'"

        # Get full docstring
        method = getattr(executor, 'method', None)
        if method:
            sig = inspect.signature(method)
            doc = inspect.getdoc(method) or "(no docstring)"
            return f"{method.__name__}{sig}\n\n{doc}"

        desc = await executor.get_action_description() if hasattr(executor, 'get_action_description') else str(executor)
        return desc


# ---------------------------------------------------------------------------
# Code prompt builder
# ---------------------------------------------------------------------------

def build_code_prompt(
    agent: Agent,
    state: ActionPolicyExecutionState,
    execution_history: list[dict[str, Any]],
    capability_summaries: str,
    error_feedback: str | None = None,
) -> str:
    """Build the LLM prompt for code generation.

    Args:
        agent: The agent instance.
        state: Current execution state.
        execution_history: List of prior step results.
        capability_summaries: String representation of available capabilities.
        error_feedback: If set, the previous code failed and this contains
            the error message + failed code for iterative refinement.
    """
    goals_str = "\n".join(f"- {g}" for g in (agent.metadata.goals or ["Complete the assigned task"]))

    # Build execution history section
    history_str = ""
    if execution_history:
        history_lines = []
        for i, entry in enumerate(execution_history[-10:]):  # Last 10 steps
            status = "✓" if entry.get("success") else "✗"
            history_lines.append(f"  Step {i+1} [{status}]: {entry.get('summary', 'no summary')}")
        history_str = "\n## Execution History\n" + "\n".join(history_lines)

    # Build error feedback section
    error_str = ""
    if error_feedback:
        error_str = f"""
## Previous Code Failed — Fix Required

The following code raised an error:
```python
{error_feedback}
```

Analyze the error and generate corrected code.
"""

    iteration_count = state.iteration_count

    prompt = f"""## Your Role
{agent.metadata.role or agent.agent_type}

## Goals
{goals_str}
{history_str}

## Available Capabilities
{capability_summaries}
Use `await browse()` for detailed signatures and `await browse("group_name")` for action details.

## Planning Capabilities (if registered)
If the agent has planning capabilities, you can call their programmatic APIs directly:
- `cap = _agent.get_capability_by_type(CacheAnalysisCapability)` — cache analysis
- `cap = _agent.get_capability_by_type(PlanLearningCapability)` — learned patterns from history
- `cap = _agent.get_capability_by_type(PlanCoordinationCapability)` — multi-agent conflict detection
These provide richer APIs than the @action_executor wrappers. Use `await browse()` to discover them.

## Iteration
This is iteration {iteration_count}. Generate code for the next 1-3 steps toward your goals.
{error_str}
## Instructions

Write async Python code that accomplishes the next step(s) toward your goals.

Available in your namespace:
- `await run("action_key", param1=val1, ...)` — execute a capability action, returns ActionResult
- `await browse(query=None)` — discover available capabilities (None=list groups, "group"=detail, "group.action"=full docs, "programmatic"=full Python APIs)
- `_agent` — the Agent instance (for calling programmatic APIs on capabilities directly)
- `bb` — the agent's primary blackboard (await bb.read/write/query)
- `results` — dict of prior action results by action_id
- `switch_mode("planning"|"execution")` — switch which capabilities appear in the prompt
- `pages` — current working set page IDs (list[str])
- `goals` — agent's current goals (list[str])
- `log(msg)` — structured logging
- Standard library: json, re, math, itertools, functools, collections, asyncio

Rules:
- Use `await run(...)` to invoke capability actions — do NOT call capability methods directly
- Handle errors: check `result.success` before using `result.output`
- Store important intermediate results: `results["my_key"] = value`
- When done with all goals, call `signal_complete()`

Respond with ONLY Python code (no markdown fences, no explanation).
"""
    return prompt


# ---------------------------------------------------------------------------
# CodeGenerationActionPolicy
# ---------------------------------------------------------------------------

class CodeGenerationActionPolicy(EventDrivenActionPolicy):
    """Action policy that uses LLM code generation instead of JSON action selection.

    The LLM generates Python code that composes ``@action_executor`` methods
    with real control flow, conditionals, loops, and data transformation.
    Generated code runs in the existing ``PolicyPythonREPL`` with an enriched
    namespace.

    This is a peer to ``CacheAwareActionPolicy`` — an alternative, not a
    replacement. It plugs into the same ``Agent.run_step()`` loop.

    Args:
        agent: The owning agent.
        max_retries: Maximum retries on code execution failure (iterative refinement).
        code_timeout: Timeout for each code execution in seconds.
        max_code_iterations: Maximum code generation iterations before signaling completion.

    Example::

        policy = CodeGenerationActionPolicy(agent=agent, max_retries=2)
        await policy.initialize()
        # Agent's run loop calls execute_iteration() which calls plan_step()
    """

    def __init__(
        self,
        agent: Agent,
        planning_capability_blueprints: list[AgentCapabilityBlueprint] | None = None,
        max_retries: int = 2,
        code_timeout: float = 30.0,
        max_code_iterations: int = 50,
        **kwargs,
    ):
        super().__init__(agent=agent, **kwargs)
        self.max_retries = max_retries
        self.code_timeout = code_timeout
        self.max_code_iterations = max_code_iterations
        self._planning_capability_blueprints = planning_capability_blueprints

        # Execution tracking
        self._execution_history: list[dict[str, Any]] = []
        self._code_iteration_count: int = 0
        self._last_error: str | None = None
        self._last_failed_code: str | None = None
        self._browser: CapabilityBrowser | None = None
        self._run_helper_installed: bool = False
        self._complete_signaled: bool = False

        # Mode: "planning" shows only planning capabilities in prompt,
        # "execution" shows only domain capabilities. Starts in planning.
        self._mode: str = "planning"

    @override
    async def initialize(self) -> None:
        """Initialize the policy, add planning capabilities, and set up REPL."""
        await super().initialize()

        # Ensure planning capabilities are registered — same as CacheAwareActionPlanner.
        # This gives the LLM access to cache analysis, learned patterns,
        # coordination, evaluation, and replanning in generated code.
        await self._ensure_planning_capabilities()

        # Ensure REPL exists
        if not self._action_dispatcher or not self._action_dispatcher.repl:
            logger.warning(
                "CodeGenerationActionPolicy requires REPLCapability. "
                "Ensure the agent has REPLCapability registered."
            )
            return

        self._browser = CapabilityBrowser(self._action_dispatcher, self.agent)
        await self._setup_enriched_namespace()

    async def _ensure_planning_capabilities(self) -> None:
        """Add default planning capabilities if not already registered."""
        from ..planning.capabilities import (
            CacheAnalysisCapability,
            PlanLearningCapability,
            PlanCoordinationCapability,
            PlanEvaluationCapability,
        )
        if self._planning_capability_blueprints is None:
             self._planning_capability_blueprints = [
                CacheAnalysisCapability.bind(key="cache_analysis", kwargs={}),
                PlanLearningCapability.bind(key="plan_learning", kwargs={}),
                PlanCoordinationCapability.bind(key="plan_coordination", kwargs={}),
                PlanEvaluationCapability.bind(key="plan_evaluation", kwargs={}),
            ]
        await self.use_capability_blueprints(self._planning_capability_blueprints)

    async def _setup_enriched_namespace(self) -> None:
        """Install helper functions into the REPL namespace."""
        repl = self._action_dispatcher.repl
        if not repl or self._run_helper_installed:
            return

        ns = repl._shell.user_ns if hasattr(repl, '_shell') and repl._shell else {}

        # run() — dispatch an action through the dispatcher
        async def run(action_key: str, **params) -> ActionResult:
            """Execute a capability action through the dispatcher.

            Args:
                action_key: The action key (e.g., "analyze_pages")
                **params: Parameters for the action

            Returns:
                ActionResult with success, output, error fields
            """
            import uuid
            action = Action(
                action_id=f"codegen_{uuid.uuid4().hex[:8]}",
                action_type=action_key,
                parameters=params,
            )
            result = await self._action_dispatcher.dispatch(action)

            # Track result
            self._execution_history.append({
                "action": action_key,
                "params": {k: str(v)[:100] for k, v in params.items()},
                "success": result.success,
                "summary": str(result.output)[:200] if result.output else (result.error or "no output"),
                "timestamp": time.time(),
            })

            # Mode transitions based on action results:
            # - should_replan returning should_replan=True → back to planning mode
            # - Any planning action completing → switch to execution mode
            action_lower = action_key.lower()
            if "should_replan" in action_lower and result.success:
                output = result.output if isinstance(result.output, dict) else {}
                if output.get("should_replan"):
                    self._mode = "planning"
                    log(f"Mode → PLANNING (replan triggered: {output.get('reason', '')})")

            # After planning actions succeed, switch to execution mode
            planning_actions = {"analyze_cache", "get_learned_patterns", "get_cache_optimal_batches",
                              "check_plan_conflicts", "get_sibling_plans", "evaluate_plan"}
            if any(pa in action_lower for pa in planning_actions) and result.success:
                # Stay in planning mode until the LLM explicitly starts executing
                pass
            elif self._mode == "planning" and result.success and not any(pa in action_lower for pa in planning_actions | {"should_replan", "browse"}):
                # First non-planning action → switch to execution mode
                self._mode = "execution"
                log("Mode → EXECUTION (first domain action dispatched)")

            return result

        # browse() — progressive capability discovery
        async def browse(query: str | None = None) -> dict[str, Any] | str:
            """Browse available capabilities.

            Args:
                query: None for groups, "group" for detail, "group.action" for full docs.
            """
            return await self._browser(query)

        # signal_complete() — signal that all goals are achieved
        def signal_complete():
            """Signal that all goals are achieved. Call this when done."""
            self._complete_signaled = True

        def switch_mode(mode: str):
            """Switch between planning and execution modes.

            Args:
                mode: 'planning' (shows planning capabilities in prompt) or
                      'execution' (shows domain capabilities in prompt).
            """
            if mode not in ("planning", "execution"):
                raise ValueError(f"Mode must be 'planning' or 'execution', got {mode!r}")
            self._mode = mode
            log(f"Mode → {mode.upper()} (explicit switch)")

        # log() — structured logging
        def log(msg: str):
            """Log a message visible in the execution trace."""
            logger.info(f"[CodeGen:{self.agent.agent_id}] {msg}")

        # Install into namespace
        ns["run"] = run
        ns["browse"] = browse
        ns["signal_complete"] = signal_complete
        ns["switch_mode"] = switch_mode
        ns["log"] = log
        ns["results"] = {}
        ns["pages"] = []
        ns["goals"] = list(self.agent.metadata.goals or [])

        # bb — the agent's primary blackboard (lazy, since it needs await)
        # We'll set it on first use
        ns["bb"] = None

        self._run_helper_installed = True
        logger.info("CodeGenerationActionPolicy: enriched REPL namespace installed")

    async def _update_dynamic_namespace(self) -> None:
        """Update dynamic namespace values before each code generation."""
        repl = self._action_dispatcher.repl
        if not repl:
            return

        ns = repl._shell.user_ns if hasattr(repl, '_shell') and repl._shell else {}

        # Update pages from working set if available
        try:
            from ..capabilities.working_set import WorkingSetCapability
            ws_cap = self.agent.get_capability_by_type(WorkingSetCapability)
            if ws_cap and hasattr(ws_cap, '_working_set'):
                ns["pages"] = list(ws_cap._working_set.keys()) if ws_cap._working_set else []
        except Exception:
            pass

        # Update goals
        ns["goals"] = list(self.agent.metadata.goals or [])

        # Lazy-init blackboard
        if ns.get("bb") is None:
            try:
                ns["bb"] = await self.agent.get_blackboard()
            except Exception:
                pass

    @override
    async def plan_step(
        self,
        state: ActionPolicyExecutionState
    ) -> Action | None:
        """Generate Python code via LLM and return it as an EXECUTE_CODE action.

        This is the core of the code-generation policy. Each call:
        1. Updates dynamic namespace (pages, goals, blackboard)
        2. Builds a prompt with execution history and capability summaries
        3. Calls the LLM to generate Python code
        4. Returns an EXECUTE_CODE action containing the generated code
        5. On failure, retries with error feedback (iterative refinement)
        """
        # First, process any pending events (inherited from EventDrivenActionPolicy)
        event_action = await super().plan_step(state)
        if event_action is not None:
            return event_action

        # Check completion
        if self._complete_signaled:
            state.custom["policy_complete"] = True
            return None

        # Check iteration limit
        self._code_iteration_count += 1
        if self._code_iteration_count > self.max_code_iterations:
            logger.warning(
                f"CodeGenerationActionPolicy: max iterations ({self.max_code_iterations}) reached"
            )
            state.custom["policy_complete"] = True
            return None

        # Ensure enriched namespace is ready
        await self._update_dynamic_namespace()

        # Build capability summaries filtered by current mode.
        # Planning mode: only planning capabilities (cache, learning, coordination, etc.)
        # Execution mode: only domain capabilities (analysis, synthesis, etc.)
        if self._mode == "planning":
            include_tags = frozenset({"planning"})
            exclude_tags = None
            mode_label = "PLANNING MODE — select strategy, analyze cache, check coordination"
        else:
            include_tags = None
            exclude_tags = frozenset({"planning"})
            mode_label = "EXECUTION MODE — perform domain actions toward your goals"

        try:
            action_descriptions = await self.get_action_descriptions(
                include_tags=include_tags,
                exclude_tags=exclude_tags,
            )
            cap_lines = [f"[{mode_label}]"]
            for group in action_descriptions:
                actions_str = ", ".join(list(group.action_descriptions.keys())[:5])
                more = f" (+{group.action_count - 5} more)" if group.action_count > 5 else ""
                cap_lines.append(f"- {group.group_key}: {group.group_description[:80]} [{actions_str}{more}]")
            capability_summaries = "\n".join(cap_lines)
        except Exception as e:
            capability_summaries = f"(error loading capabilities: {e})"

        # Build error feedback if previous code failed
        error_feedback = None
        if self._last_error and self._last_failed_code:
            error_feedback = f"{self._last_failed_code}\n\nError:\n{self._last_error}"

        # Build the prompt
        prompt = build_code_prompt(
            agent=self.agent,
            state=state,
            execution_history=self._execution_history,
            capability_summaries=capability_summaries,
            error_feedback=error_feedback,
        )

        # Call LLM to generate code
        try:
            response = await self.agent.infer(
                prompt=prompt,
                system_prompt=(
                    "You are a Python code generator for an autonomous agent. "
                    "Generate clean, correct async Python code that uses the provided "
                    "helper functions to accomplish the agent's goals. "
                    "Output ONLY executable Python code — no markdown, no explanation."
                ),
                max_tokens=2048,
            )
            code = self._extract_code(response)
        except Exception as e:
            logger.error(f"CodeGenerationActionPolicy: LLM inference failed: {e}")
            return None

        if not code or not code.strip():
            logger.warning("CodeGenerationActionPolicy: LLM generated empty code")
            return None

        # Clear error state on new generation
        self._last_error = None
        self._last_failed_code = None

        logger.info(
            f"CodeGenerationActionPolicy: generated code ({len(code)} chars) "
            f"for iteration {self._code_iteration_count}"
        )

        # Return as EXECUTE_CODE action
        return Action(
            action_id=f"codegen_step_{self._code_iteration_count}",
            action_type=ActionType.EXECUTE_CODE,
            parameters={"code": code},
            description=f"Code generation step {self._code_iteration_count}",
        )

    def _extract_code(self, response: Any) -> str:
        """Extract Python code from LLM response.

        Handles responses with or without markdown code fences.
        """
        if hasattr(response, 'text'):
            text = response.text
        elif hasattr(response, 'content'):
            text = response.content
        elif isinstance(response, str):
            text = response
        elif isinstance(response, dict):
            text = response.get("text", response.get("content", str(response)))
        else:
            text = str(response)

        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```python"):
            text = text[len("```python"):].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()

        return text

    @override
    async def execute_iteration(
        self,
        state: ActionPolicyExecutionState
    ) -> ActionPolicyIterationResult:
        """Execute one iteration with transaction wrapper and retry logic.

        Wraps the parent's execute_iteration with:
        1. REPL namespace snapshot (restore on failure)
        2. Retry on code execution failure (iterative refinement)
        """
        repl = self._action_dispatcher.repl if self._action_dispatcher else None

        # Snapshot namespace before execution
        namespace_snapshot = None
        if repl and hasattr(repl, '_shell') and repl._shell:
            try:
                namespace_snapshot = {
                    k: v for k, v in repl._shell.user_ns.items()
                    if not k.startswith('_') and k not in ('In', 'Out', 'get_ipython')
                }
            except Exception:
                pass

        # Execute (parent handles plan_step → dispatch)
        result = await super().execute_iteration(state)

        # Check if code execution failed
        if (result.result and not result.result.success
                and result.action_executed
                and result.action_executed.action_type == ActionType.EXECUTE_CODE):

            error_msg = result.result.error or "Unknown error"
            failed_code = result.action_executed.parameters.get("code", "")

            logger.warning(
                f"CodeGenerationActionPolicy: code execution failed: {error_msg[:200]}"
            )

            # Restore namespace on failure
            if namespace_snapshot and repl and hasattr(repl, '_shell') and repl._shell:
                try:
                    # Only restore user-defined variables, not system ones
                    for key in list(repl._shell.user_ns.keys()):
                        if not key.startswith('_') and key not in ('In', 'Out', 'get_ipython'):
                            if key not in namespace_snapshot:
                                del repl._shell.user_ns[key]
                    for key, val in namespace_snapshot.items():
                        repl._shell.user_ns[key] = val
                except Exception as e:
                    logger.warning(f"Failed to restore namespace: {e}")

            # Store error for iterative refinement on next iteration
            self._last_error = error_msg
            self._last_failed_code = failed_code

            # Return failure but don't complete — retry on next iteration
            return ActionPolicyIterationResult(
                success=False,
                policy_completed=False,
                action_executed=result.action_executed,
                result=result.result,
            )

        return result

    @classmethod
    def bind(cls, **kwargs) -> "ActionPolicyBlueprint":
        """Create a blueprint for this policy.

        Args:
            **kwargs: Constructor arguments (max_retries, code_timeout, etc.)

        Returns:
            ActionPolicyBlueprint for serializable agent configuration.
        """
        from ...blueprint import ActionPolicyBlueprint
        return ActionPolicyBlueprint(cls=cls, kwargs=kwargs)


async def create_code_generation_action_policy(
    agent: Agent,
    action_map: list[ActionGroup] | None = None,
    action_providers: list[Any] = [],
    io: ActionPolicyIO | None = None,
    max_retries: int = 2,
    code_timeout: float = 30.0,
    max_code_iterations: int = 50,
) -> "CodeGenerationActionPolicy":
    """Create a code-generation-based action policy.

    The LLM generates Python code that composes ``@action_executor`` methods
    with real control flow, instead of selecting from a JSON action schema.

    Requires ``REPLCapability`` on the agent.

    Args:
        agent: The owning agent.
        action_map: Optional pre-built action groups.
        action_providers: Additional action providers.
        io: Action policy I/O configuration.
        max_retries: Max retries on code execution failure.
        code_timeout: Timeout for each code execution.
        max_code_iterations: Max code generation iterations.

    Returns:
        CodeGenerationActionPolicy
    """
    from .code_generation import CodeGenerationActionPolicy

    action_policy = CodeGenerationActionPolicy(
        agent=agent,
        max_retries=max_retries,
        code_timeout=code_timeout,
        max_code_iterations=max_code_iterations,
        action_map=action_map,
        action_providers=action_providers,
        io=io,
    )
    await action_policy.initialize()

    return action_policy

