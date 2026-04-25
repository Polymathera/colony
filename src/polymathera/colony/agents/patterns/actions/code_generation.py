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

import ast
import asyncio
import inspect
import logging
import time
from typing import Any

from overrides import override

from ...base import Agent
from ...blackboard.protocol import ActionPolicyLifecycleProtocol
from ...blueprint import ActionPolicyBlueprint, AgentCapabilityBlueprint
from ...models import (
    Action,
    ActionType,
    ActionResult,
    ActionPolicyExecutionState,
    ActionPolicyIterationResult,
    ActionPolicyIO,
    PlanningContext,
    PlanExecutionContext,
)
from .dispatcher import ActionGroup, ActionDispatcher
from .policies import (
    BaseActionPolicy,
    EventDrivenActionPolicy,
)
from ..planning.context import PlanningContextBuilder
from ....distributed.hooks import hookable
from .code_constraints import (
    CodeGenerator,
    FreeFormCodeGenerator,
    CodeValidator,
    NoOpValidator,
    IterationShapeValidator,
    SkillLibrary,
    NoOpSkillLibrary,
    RecoveryStrategy,
    DeterministicRecovery,
    RuntimeGuardrail,
    NoGuardrail,
    CompletionValidator,
    NoOpCompletionValidator,
    LLMCompletionValidator,
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
                   full action key for full action detail,
                   "programmatic" for all programmatic methods on capabilities.

        Returns:
            Structured capability information.
        """
        if query is None:
            return await self._list_groups()

        if query == "programmatic":
            return await self._list_programmatic_apis()

        # Try as a full action key first (e.g., "CacheAnalysis.CacheAnalysis.analyze_cache").
        # Action keys contain dots so we can't split on "." to distinguish
        # group vs action queries — instead, check if the query matches any
        # registered executor key directly.
        detail = await self._action_detail_by_full_key(query)
        if detail is not None:
            return detail

        return await self._group_detail(query)

    async def _list_groups(self) -> dict[str, Any]:
        """Top-level: list all capability groups with summaries."""
        groups = {}
        for group in self._dispatcher.action_map:
            action_names = [
                name for name, ex in group.executors.items()
                if not getattr(ex, 'exclude_from_planning', False)
            ]
            groups[group.group_key] = {
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
        group = None
        for g in self._dispatcher.action_map:
            # Fuzzy match
            if g.group_key == group_key or group_key.lower() in g.group_key.lower():
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

    async def _action_detail_by_full_key(self, query: str) -> str | None:
        """Look up an action by its full compound key.

        Returns None if no match found (so the caller can try group lookup).
        """
        for group in self._dispatcher.action_map:
            executor = group.executors.get(query)
            if executor:
                return await self._format_executor(executor)
        return None

    @staticmethod
    async def _format_executor(executor) -> str:
        """Format an executor's method signature and docstring."""
        method = getattr(executor, 'method', None)
        if method:
            sig = inspect.signature(method)
            doc = inspect.getdoc(method) or "(no docstring)"
            return f"{method.__name__}{sig}\n\n{doc}"
        if hasattr(executor, 'get_action_description'):
            return await executor.get_action_description()
        return str(executor)


# ---------------------------------------------------------------------------
# Code prompt formatting — thin layer on top of PlanningContext
# ---------------------------------------------------------------------------


def _normalize_code_for_reuse(code: str) -> str:
    """Normalize code snippets so semantically identical examples dedupe."""
    text = code.strip()
    if not text:
        return ""

    try:
        return ast.unparse(ast.parse(text))
    except Exception:
        return "\n".join(
            line.strip()
            for line in text.splitlines()
            if line.strip()
        )


def _extract_run_action_keys_from_code(code: str) -> list[str]:
    """Extract literal action keys used in ``run("...")`` calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    action_keys: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "run":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            action_keys.append(first_arg.value)
    return action_keys


def _build_codegen_step_summary(
    *,
    actions_called: list[str],
    planning_action_keys: set[str],
    had_failures: bool,
    repl_success: bool,
    errors: list[str],
    mode_before: str,
    mode_after: str,
    run_call_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a structured summary for one generated-code iteration."""
    planning_actions = [key for key in actions_called if key in planning_action_keys]
    domain_actions = [key for key in actions_called if key not in planning_action_keys]

    if planning_actions and domain_actions:
        step_kind = "mixed"
    elif planning_actions:
        step_kind = "planning"
    elif domain_actions:
        step_kind = "execution"
    else:
        step_kind = "noop"

    return {
        "actions_called": list(actions_called),
        "planning_actions": planning_actions,
        "domain_actions": domain_actions,
        "step_kind": step_kind,
        "had_failures": had_failures,
        "repl_success": repl_success,
        "errors": list(errors),
        "mode_before": mode_before,
        "mode_after": mode_after,
        "run_call_trace": list(run_call_trace) if run_call_trace else [],
    }


def _format_codegen_step_history(step_info: dict[str, Any]) -> list[str]:
    """Render one stored step summary into prompt-friendly history lines."""
    ok = step_info.get("repl_success", False) and not step_info.get("had_failures", False)
    status = "✓" if ok else "✗"
    errors = [str(err) for err in step_info.get("errors", []) if err]

    planning_actions = list(step_info.get("planning_actions", []))
    domain_actions = list(step_info.get("domain_actions", []))
    actions = list(step_info.get("actions_called", []))

    step_kind = step_info.get("step_kind")
    if step_kind not in {"planning", "execution", "mixed", "noop"}:
        if planning_actions and domain_actions:
            step_kind = "mixed"
        elif planning_actions:
            step_kind = "planning"
        elif domain_actions or actions:
            step_kind = "execution"
        else:
            step_kind = "noop"

    mode_before = step_info.get("mode_before")
    mode_after = step_info.get("mode_after")
    mode_change = bool(mode_before and mode_after and mode_before != mode_after)

    lines: list[str] = []
    if planning_actions or domain_actions or actions:
        display_actions = planning_actions + [
            key for key in domain_actions if key not in planning_actions
        ]
        if not display_actions:
            display_actions = actions
        action_names = ", ".join(key.rsplit(".", 1)[-1] for key in display_actions)
        label = {
            "planning": "Planning step",
            "execution": "Execution step",
            "mixed": "Mixed step",
            "noop": "Step",
        }[step_kind]
        line = f"  [{status}] {label}: {action_names}"
        if mode_change:
            line += f" -> {mode_after.upper()}"
        lines.append(line)
    elif mode_change:
        lines.append(f"  [{status}] Mode switch: {mode_before.upper()} -> {mode_after.upper()}")
    elif ok and not errors:
        return []
    else:
        lines.append(f"  [{status}] No-op step")

    for error_msg in errors:
        lines.append(f"        Error: {error_msg[:200]}")

    return lines


def _should_store_skill_from_step_summary(step_info: dict[str, Any] | None) -> bool:
    """Store only iterations that actually performed domain work."""
    if not step_info:
        return False
    if step_info.get("had_failures"):
        return False
    return bool(step_info.get("domain_actions"))


def _select_prompt_skills(
    skills: list[Any],
    *,
    mode: str,
    planning_action_keys: set[str],
) -> list[Any]:
    """Select high-signal prior code examples for the prompt.

    Planning mode intentionally avoids code few-shot examples. Planning already
    has dedicated planning capabilities and historical plan-learning actions;
    replaying successful planning stubs tends to anchor the model into
    repeating the planning preamble instead of advancing into domain work.
    """
    if mode != "execution":
        return []

    selected: list[Any] = []
    seen_code: set[str] = set()
    for skill in skills:
        code = getattr(skill, "code", "")
        run_action_keys = _extract_run_action_keys_from_code(code)
        if not run_action_keys:
            continue
        if not any(key not in planning_action_keys for key in run_action_keys):
            continue

        normalized_code = _normalize_code_for_reuse(code)
        if not normalized_code or normalized_code in seen_code:
            continue

        seen_code.add(normalized_code)
        selected.append(skill)

    return selected

def format_planning_context_for_codegen(
    planning_context: PlanningContext,
    mode: str,
    error_history: list[dict[str, str]] | None = None,
    allow_self_termination: bool = True,
) -> str:
    """Format a ``PlanningContext`` as a code-generation prompt.

    This is a thin rendering layer — all context gathering (memories, identity,
    constraints, action descriptions, consciousness streams) is done by
    ``PlanningContextBuilder``.

    Args:
        planning_context: Structured context from ``PlanningContextBuilder``.
        mode: Current mode — ``"planning"`` or ``"execution"``.
        error_history: Accumulated list of ``{"code": ..., "error": ...}`` dicts
            from ALL prior failed attempts, not just the last one.
        allow_self_termination: If True, include signal_completion in
            the prompt instructions and namespace docs.
    """
    parts: list[str] = []

    # System prompt (agent identity from ConsciousnessCapability or metadata)
    if planning_context.system_prompt:
        parts.append(planning_context.system_prompt)

    # Goals
    goals = planning_context.goals or []
    if goals:
        parts.append("## Goals\n" + "\n".join(f"- {g}" for g in goals))

    # Constraints
    if planning_context.constraints:
        constraint_lines = [f"- {k}: {v}" for k, v in planning_context.constraints.items()]
        parts.append("## Constraints\n" + "\n".join(constraint_lines))

    # Consciousness streams — each stream is a filtered view of the agent's
    # experience (events + actions) rendered by its own formatter. Streams
    # are pre-rendered by PlanningContextBuilder.
    for section in planning_context.stream_sections:
        if section:
            parts.append(section)

    # Execution progress — show what actions each code step called,
    # Only render code steps (codegen_plan_step_*), not internal actions.
    exec_ctx = planning_context.execution_context
    step_summaries = (exec_ctx.custom_data.get("codegen_step_summaries", {})
                      if exec_ctx else {})
    if step_summaries:
        history_lines = []
        # Iterate step summaries in insertion order (Python 3.7+ dict)
        for step_id, step_info in list(step_summaries.items())[-20:]:  # TODO: Make this configurable
            history_lines.extend(_format_codegen_step_history(step_info))
        if history_lines:
            parts.append("## Execution History\n" + "\n".join(history_lines))

    if exec_ctx and exec_ctx.findings:
        findings_lines = [f"- {k}: {v}" for k, v in list(exec_ctx.findings.items())[:10]]  # TODO: Make this configurable
        parts.append("## Findings So Far\n" + "\n".join(findings_lines))

    # Recalled memories
    if planning_context.recalled_memories:
        memory_lines = []
        for mem in planning_context.recalled_memories[:10]:
            memory_lines.append(f"- [{mem.get('key', '?')}] {str(mem.get('value', ''))[:200]}")
        parts.append("## Recalled Memories\n" + "\n".join(memory_lines))

    # Memory architecture guidance
    mem_guidance = planning_context.custom_data.get("memory_architecture_guidance")
    if mem_guidance:
        parts.append(f"## Memory Architecture\n{mem_guidance}")

    # Current mode + lifecycle
    parts.append(_build_mode_section(mode, planning_context))

    # -- Available actions (explicit run() calls with exact keys + full descriptions) --
    cap_lines = []
    for group in planning_context.action_descriptions:
        cap_lines.append(f"\n### {group.group_key}")
        cap_lines.append(group.group_description)
        for action_key, action_desc in group.action_descriptions.items():
            cap_lines.append(f'\n#### `await run("{action_key}")`')
            cap_lines.append(action_desc)
    parts.append("## Available Actions\n" + "\n".join(cap_lines))

    # Error feedback — show ALL prior failed attempts so the LLM doesn't
    # repeat the same mistakes. ``raw_response`` (when present) is the
    # exact text the LLM emitted before extraction; we show it back so
    # the model sees its own malformed output instead of only the
    # post-extraction string the validator rejected.
    if error_history:
        error_lines = ["## Failed Attempts — Do NOT Repeat These\n"]
        for i, attempt in enumerate(error_history, 1):
            error_lines.append(f"### Attempt {i}")
            raw = attempt.get("raw_response")
            if raw:
                error_lines.append(
                    "**Your raw response was (truncated):**\n"
                    "```\n"
                    f"{raw}\n"
                    "```"
                )
            code = attempt.get("code") or ""
            if code and code != raw:
                error_lines.append(
                    "**After extraction the code was:**\n"
                    "```python\n"
                    f"{code}\n"
                    "```"
                )
            error_lines.append(f"**Error:** {attempt['error']}\n")
        error_lines.append(
            "Analyze ALL the above failures. Do NOT use any action key format "
            "that has already failed. Emit ONLY raw Python code "
            "— no markdown fences, no JSON output blocks, no prose "
            "between statements. Use the EXACT action keys from the "
            "Available Actions section above."
        )
        parts.append("\n".join(error_lines))

    # Completion rejection feedback
    if exec_ctx:
        rejection = exec_ctx.custom_data.get("last_completion_rejection")
        if rejection:
            rejection_lines = [
                "## Completion Rejected",
                f"Your previous signal_completion() was rejected: {rejection['reason']}",
            ]
            if rejection.get("suggestions"):
                rejection_lines.append("Suggestions:")
                for s in rejection["suggestions"]:
                    rejection_lines.append(f"- {s}")
            rejection_lines.append(
                "Do NOT call signal_completion() again until you have addressed "
                "the issues above."
            )
            parts.append("\n".join(rejection_lines))

    # -- Instructions --
    parts.append(_build_instructions_section(mode, planning_context, allow_self_termination=allow_self_termination))

    return "\n\n".join(parts)


def _build_mode_section(mode: str, planning_context: PlanningContext) -> str:
    """Build the mode lifecycle section with a worked example.

    The example uses a real action key from the current planning context so
    the LLM sees immediately usable code, not a placeholder pattern.
    """
    if mode == "planning":
        # Pick a real action key for the example
        example_key = _first_action_key(planning_context)
        example_call = f'await run("{example_key}")' if example_key else 'await run("<action_key>")'
        return f"""## Current Mode: PLANNING

You are selecting strategy before doing domain work. Available actions are
planning capabilities (cache analysis, coordination, learned patterns).

After 1–2 planning actions, call `switch_mode("execution")` to begin domain work.
Do NOT start domain actions while in planning mode.

Example of a good planning iteration:
```python
result = {example_call}
if result.success:
    results["planning"] = result.output
    switch_mode("execution")
```"""
    else:
        example_key = _first_action_key(planning_context)
        example_call = f'await run("{example_key}")' if example_key else 'await run("<action_key>")'
        return f"""## Current Mode: EXECUTION

You are performing domain work toward your goals. Available actions are domain
capabilities. Read task parameters from `params` — do NOT hardcode values.

Example of a good execution iteration:
```python
result = {example_call}
if result.success:
    results["step"] = result.output
```"""


def _build_instructions_section(mode: str, planning_context: PlanningContext, *, allow_self_termination: bool = True) -> str:
    """Build the tightened instructions section."""
    rules = [
        "1. Write 1–3 focused actions per iteration. NOT a complete program.",
        "2. Use EXACT action keys from Available Actions above with `await run(\"key\")`.",
        "3. Read task parameters from `params` — do NOT hardcode file paths, thresholds, or config values.",
        "4. Check `result.success` before using `result.output`.",
        "5. Store important results: `results[\"key\"] = value`.",
    ]
    if allow_self_termination:
        rules.append("6. Only when all goals are achieved, call `await signal_completion()` (validated before accepting).")

    namespace_items = [
        '- `await run("action_key", param1=val1, ...)` — execute a capability action, returns ActionResult',
        '- `await browse(query=None)` — discover available capabilities (None=list groups, "group"=detail, "group.action"=full docs, "programmatic"=full Python APIs)',
        "- `params` — task parameters dict (repo_id, target_files, thresholds, etc.)",
        "- `_agent` — the Agent instance (for calling programmatic APIs on capabilities directly)",
        "- `bb` — the agent's primary blackboard (await bb.read/write/query)",
        "- `results` — dict of prior action results by action_id",
        '- `switch_mode("planning"|"execution")` — switch which capabilities appear in the prompt',
        "- `pages` — current working set page IDs (list[str])",
        "- `goals` — agent's current goals (list[str])",
    ]
    if allow_self_termination:
        namespace_items.append("- `await signal_completion()` — signal all goals achieved (validated before accepting)")
    namespace_items.extend([
        "- `log(msg)` — structured logging",
        "- Standard library: json, re, math, itertools, functools, collections, asyncio",
    ])

    return (
        "## Rules\n\n" + "\n".join(rules) +
        "\n\n## Namespace\n" + "\n".join(namespace_items) +
        "\n\nRespond with ONLY Python code. No markdown fences. No explanation."
    )


def _first_action_key(planning_context: PlanningContext) -> str | None:
    """Extract the first real action key from the planning context.

    Used to build worked examples with actual keys instead of placeholders.
    """
    for group in planning_context.action_descriptions:
        for key in group.action_descriptions:
            return key
    return None


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

    The code generation process is decomposed into five independent dimensions,
    each controlled by an abstract class that library users can replace:

    1. **CodeGenerator** — controls HOW code is produced (free-form, grammar-
       constrained, skeleton-with-holes). Default: ``FreeFormCodeGenerator``.
    2. **CodeValidator** — checks generated code BEFORE execution. Default:
       ``IterationShapeValidator`` (enforces focused iterations).
    3. **SkillLibrary** — stores/retrieves successful code for reuse. Default:
       ``NoOpSkillLibrary``.
    4. **RecoveryStrategy** — handles failures (deterministic fix, LLM retry).
       Default: ``DeterministicRecovery``.
    5. **RuntimeGuardrail** — enforces constraints DURING execution (capability
       boundaries, temporal ordering). Default: ``NoGuardrail``.

    Args:
        agent: The owning agent.
        code_generator: Dimension 1 — how code is produced.
        code_validators: Dimension 2 — pre-execution validation.
        skill_library: Dimension 3 — skill storage and retrieval.
        recovery_strategy: Dimension 4 — failure recovery.
        runtime_guardrail: Dimension 5 — runtime constraints.
        max_retries: Maximum retries on code execution failure.
        code_timeout: Timeout for each code execution in seconds.
        max_code_iterations: Maximum code generation iterations before signaling completion.

    Example::

        policy = CodeGenerationActionPolicy(
            agent=agent,
            code_validators=[APIKnowledgeBaseValidator(agent)],
            skill_library=InMemorySkillLibrary(),
            runtime_guardrail=CapabilityBoundaryGuardrail(
                allowed_prefixes=["CacheAnalysis", "Analysis"],
            ),
        )
        await policy.initialize()
    """

    def __init__(
        self,
        agent: Agent,
        context_builder: PlanningContextBuilder | None = None,
        planning_capability_blueprints: list[AgentCapabilityBlueprint] | None = None,
        code_generator: CodeGenerator | None = None,
        code_validators: list[CodeValidator] | None = None,
        skill_library: SkillLibrary | None = None,
        recovery_strategy: RecoveryStrategy | None = None,
        runtime_guardrail: RuntimeGuardrail | None = None,
        completion_validator: CompletionValidator | None = None,
        max_retries: int = 2,
        code_timeout: float = 30.0,
        max_code_iterations: int = 50,
        allow_self_termination: bool = True,
        **kwargs,
    ):
        super().__init__(agent=agent, **kwargs)
        self.max_retries = max_retries
        self._allow_self_termination = allow_self_termination
        self.code_timeout = code_timeout
        self.max_code_iterations = max_code_iterations
        self._planning_capability_blueprints = planning_capability_blueprints
        self._context_builder = context_builder or PlanningContextBuilder(agent)

        # Constraint dimensions — each is a user-replaceable abstract class.
        self._code_generator = code_generator or FreeFormCodeGenerator()
        self._code_validators = code_validators or [IterationShapeValidator()]
        self._skill_library = skill_library or NoOpSkillLibrary()
        self._recovery_strategy = recovery_strategy or DeterministicRecovery()
        self._runtime_guardrail = runtime_guardrail or NoGuardrail()
        self._completion_validator = completion_validator or LLMCompletionValidator() # NoOpCompletionValidator()

        # Execution tracking — PlanExecutionContext is the structured state.
        self._execution_context = PlanExecutionContext()
        self._code_iteration_count: int = 0
        self._error_history: list[dict[str, str]] = []
        self._recovered_code: str | None = None
        self._consecutive_failures: int = 0
        self._call_history: list[str] = []
        self._run_call_trace: list[dict[str, Any]] = []
        self._had_internal_failures: bool = False
        self._internal_errors: list[str] = []
        self._browser: CapabilityBrowser | None = None
        self._run_helper_installed: bool = False
        self._complete_signaled: bool = False

        # Mode: "planning" shows only planning capabilities in prompt,
        # "execution" shows only domain capabilities. Starts in planning.
        self._mode: str = "planning"

        # In-flight LLM code-generation task. The codegen LLM call lives
        # *outside* any @action_executor (it is invoked directly from
        # plan_step), so the dispatcher cannot cancel it. We track it here
        # explicitly so /abort can cancel a long-running codegen call mid-
        # request. None when no LLM call is in flight. Cleared in finally
        # in plan_step regardless of outcome.
        self._current_codegen_task: asyncio.Task[str] | None = None
        self._codegen_cancel_requested: bool = False

    @override
    async def initialize(self) -> None:
        """Initialize the policy, add planning capabilities, and set up REPL."""
        await super().initialize()

        # Ensure planning capabilities are registered — same as CacheAwareActionPlanner.
        # This gives the LLM access to cache analysis, learned patterns,
        # coordination, evaluation, and replanning in generated code.
        # NOTE: use_capability_blueprints() nulls _action_dispatcher to force
        # rebuild with the new capabilities. We must recreate it before
        # setting up the REPL namespace.
        await self._ensure_planning_capabilities()

        # Recreate the dispatcher — _ensure_planning_capabilities() nulled it
        # via use_agent_capabilities(). _create_action_dispatcher() is idempotent
        # (no-op if dispatcher already exists).
        await self._create_action_dispatcher()

        # Ensure REPL exists
        if not self._action_dispatcher or not self._action_dispatcher.repl:
            logger.warning(
                "CodeGenerationActionPolicy requires REPLCapability. "
                "Ensure the agent has REPLCapability registered."
            )
            return

        self._browser = CapabilityBrowser(self._action_dispatcher, self.agent)
        await self._setup_enriched_namespace()

        # If no planning-tagged action groups exist, planning mode is a
        # vacuous state — the prompt filter would hide every real action
        # and the LLM could never dispatch a domain action to trigger the
        # planning→execution transition. Start directly in execution mode.
        if not self._get_planning_action_keys():
            self._mode = "execution"
            logger.info(
                "CodeGenerationActionPolicy: no planning-tagged actions registered; "
                "starting in execution mode"
            )

    async def _ensure_planning_capabilities(self) -> None:
        """Add default planning capabilities if not already registered."""
        from ..planning.capabilities import (
            CacheAnalysisCapability,
            PlanLearningCapability,
            PlanCoordinationCapability,
            PlanEvaluationCapability,
            ReplanningCapability,
        )
        if self._planning_capability_blueprints is None:
             self._planning_capability_blueprints = [
                CacheAnalysisCapability.bind(),
                PlanLearningCapability.bind(),
                PlanCoordinationCapability.bind(),
                PlanEvaluationCapability.bind(),
                ReplanningCapability.bind(),
            ]
        await self.use_capability_blueprints(self._planning_capability_blueprints)

    def _get_planning_action_keys(self) -> set[str]:
        """Return the full action keys tagged as planning actions."""
        if not self._action_dispatcher:
            return set()

        planning_action_keys: set[str] = set()
        for group in self._action_dispatcher.action_map:
            if "planning" not in group.tags:
                continue
            planning_action_keys.update(group.executors.keys())
        return planning_action_keys

    def _is_planning_action_key(self, action_key: str) -> bool:
        """Check whether an action key belongs to a planning-tagged capability."""
        planning_action_keys = self._get_planning_action_keys()
        if not planning_action_keys:
            return False
        if action_key in planning_action_keys:
            return True
        return any(key.endswith(f".{action_key}") for key in planning_action_keys)

    @override
    def get_status_snapshot(self) -> dict[str, Any]:
        """Read-only summary of policy state.

        Adds codegen-specific fields on top of the parent snapshot:
        recovery state (so ``/status`` can tell the user "regenerating
        code, attempt 2/3"), code-iteration count, and current mode.
        Safe to call concurrently with the main planning loop.
        """
        snapshot = super().get_status_snapshot()
        snapshot.update({
            "in_recovery": self._consecutive_failures > 0,
            "recovery_attempts": self._consecutive_failures,
            "max_recovery_attempts": self.max_retries,
            "code_iteration_count": self._code_iteration_count,
            "mode": self._mode,
            "complete_signaled": self._complete_signaled,
        })
        return snapshot

    @override
    async def abort_current(self, *, reason: str | None = None) -> bool:
        """Abort whatever the policy is currently doing.

        The codegen policy has TWO interruptible-state surfaces beyond what
        the dispatcher tracks, both of which must be wound down for ``/abort``
        to feel correct to the user:

        1. **In-flight LLM codegen call** — the call sits in ``plan_step``,
           NOT in an ``@action_executor``, so the dispatcher cannot see it.
           We track it explicitly in ``_current_codegen_task`` and cancel it
           here. Setting ``_codegen_cancel_requested`` first lets ``plan_step``
           tell a user-cancel apart from outer-shutdown when the
           ``CancelledError`` lands.
        2. **Recovery state** — if validation has been failing in a loop, the
           policy has ``_consecutive_failures > 0`` and an ``_error_history``
           that ``has_pending_work()`` reads to short-circuit the event wait.
           After an abort, the user is no longer interested in this attempt
           chain; resetting clears the recovery banner and lets the next
           iteration block on a fresh user event.

        Steps 1 and 2 are independent: the codegen call may not be in flight
        when /abort lands (we may already be inside ``await dispatch(...)``),
        and recovery may be active without an in-flight LLM call. We always
        do both, then delegate to ``super().abort_current()`` to also cancel
        any dispatcher-tracked action (the ``EXECUTE_CODE`` REPL run, or any
        other interruptible @action_executor invoked from generated code).

        Returns ``True`` if anything was actually interrupted — useful for
        callers (the chat handler) to know whether to send "aborted" feedback
        or "nothing to abort".
        """
        anything_aborted = False

        # 1. Cancel in-flight LLM codegen task, if any.
        codegen_task = self._current_codegen_task
        if codegen_task is not None and not codegen_task.done():
            self._codegen_cancel_requested = True
            codegen_task.cancel()
            anything_aborted = True
            logger.info(
                "CodeGenerationActionPolicy.abort_current: cancelled in-flight codegen LLM task"
            )

        # 2. Reset recovery state regardless of whether (1) fired — recovery
        # may be active without an in-flight LLM call (e.g., we already have
        # a recovered_code waiting to execute).
        prior_failures = self._consecutive_failures
        recovery_was_active = (
            self._consecutive_failures > 0
            or self._recovered_code is not None
            or bool(self._error_history)
        )
        if recovery_was_active:
            self._consecutive_failures = 0
            self._error_history.clear()
            self._recovered_code = None
            anything_aborted = True
            # Clear the recovery banner so the UI doesn't keep spinning
            # on "regenerating after invalid output". We emit a terminal
            # codegen_retry event with finished=True, succeeded=False
            # (the user aborted, not a clean recovery) — same protocol
            # the bridge uses when retries exhaust.
            await self._emit_codegen_recovery_banner(
                attempt=prior_failures,
                max_attempts=self.max_retries,
                last_error=f"aborted: {reason or 'user request'}",
                finished=True,
                succeeded=False,
            )
            logger.info(
                "CodeGenerationActionPolicy.abort_current: reset recovery state "
                "(prior failures=%d)", prior_failures,
            )

        # 3. Delegate to super so a currently-dispatched interruptible action
        # (EXECUTE_CODE in REPL, or any @action_executor with interruptible=True
        # called from generated code) is also cancelled.
        if await super().abort_current(reason=reason):
            anything_aborted = True

        return anything_aborted

    @override
    def has_pending_work(self) -> bool:
        """Tell ``EventDrivenActionPolicy`` to skip the event wait when
        a recovery iteration is in flight.

        Two situations qualify:

        - The previous iteration produced bad code that failed
          validation. ``_consecutive_failures`` is the per-attempt
          counter; while it is below ``max_retries`` we have unfinished
          work — re-prompt the LLM with the accumulated
          ``_error_history`` instead of waiting for a new user
          message that may never arrive.
        - The recovery strategy produced fixed code (``_recovered_code``)
          that still needs to be executed.
        """
        if self._recovered_code is not None:
            return True
        if (
            self._consecutive_failures > 0
            and self._consecutive_failures < self.max_retries
        ):
            return True
        return False

    async def _handle_codegen_failure(
        self, *, code: str, error: str, raw: str,
    ) -> None:
        """Centralised handling for a failed code-generation attempt.

        Three things happen here, in order:

        1. The bad attempt is recorded in ``_error_history`` so the next
           prompt iteration shows the LLM exactly what it produced
           (truncated raw output) plus the validator's error message.
           The next iteration runs without waiting for a new event
           because ``has_pending_work()`` returns True.
        2. A generic ``policy:codegen_retry:*`` lifecycle event is
           emitted on the agent's blackboard. Subscribers (the chat
           bridge, traces, log adapters) can react however they want;
           the policy itself does not know what they will do.
        3. When ``_consecutive_failures`` reaches ``max_retries``, the
           policy emits ``policy:codegen_failed:*`` with the same
           reasoning, and resets state so the agent is ready for the
           user's next prompt instead of dead-looping.

        Caller still returns ``None`` from ``plan_step``; this method
        only records and surfaces the failure.
        """
        self._consecutive_failures += 1
        # Show the LLM its own raw output so it doesn't repeat the
        # same mistake (e.g., re-emitting markdown fences after we
        # told it not to).
        self._error_history.append({
            "code": code,
            "raw_response": (raw or "")[:1500],
            "error": error,
        })

        ts_ms = int(time.time() * 1000)
        if self._consecutive_failures < self.max_retries:
            await self._emit_lifecycle_event(
                key=ActionPolicyLifecycleProtocol.codegen_retry_key(ts_ms),
                payload={
                    "attempt": self._consecutive_failures,
                    "max_attempts": self.max_retries,
                    "error": error[:300],
                    "ts": time.time(),
                },
            )
            return

        # Exhausted. Stop the loop, surface the failure, reset state.
        logger.error(
            "CodeGenerationActionPolicy: exhausted %d retries; "
            "abandoning the request to avoid a dead-loop. Last "
            "error: %s",
            self._consecutive_failures, error,
        )
        await self._emit_lifecycle_event(
            key=ActionPolicyLifecycleProtocol.codegen_failed_key(ts_ms),
            payload={
                "attempts": self._consecutive_failures,
                "max_attempts": self.max_retries,
                "error": error[:300],
                "ts": time.time(),
            },
        )
        # Reset so the agent is ready for the user's next prompt
        # instead of immediately retrying the same thing.
        self._consecutive_failures = 0
        self._error_history.clear()
        self._recovered_code = None

    async def _emit_lifecycle_event(
        self, *, key: str, payload: dict[str, Any],
    ) -> None:
        """Publish one ``ActionPolicyLifecycleProtocol`` record on the
        agent's primary blackboard.

        Subscribers — capabilities with ``@event_handler`` matching
        ``policy:*``, observers, traces — decide what to do with the
        event. The policy itself knows nothing about how the event is
        consumed: chat UIs, log adapters, dashboards are all equally
        valid downstream concerns.

        Best-effort: emission failure never breaks the action itself.
        """
        try:
            bb = await self.agent.get_blackboard()
            await bb.write(key, payload)
        except Exception as e:  # pragma: no cover — defensive
            logger.debug(
                "CodeGenerationActionPolicy: lifecycle emit failed: %s",
                e,
            )

    async def _emit_codegen_recovery_banner(
        self,
        *,
        attempt: int,
        max_attempts: int,
        last_error: str,
        finished: bool,
        succeeded: bool,
    ) -> None:
        """Emit a recovery-progress signal on the policy lifecycle bus.

        Re-uses the ``policy:codegen_retry:*`` key — chat-UI subscribers
        already render the recovery banner from these events. The
        ``finished`` + ``succeeded`` flags tell the renderer to clear /
        replace the banner instead of stacking another "attempt N/M"
        line. Without this method, the success branch in ``plan_step``
        (which fires after a recovery iteration validates) would crash
        with ``AttributeError`` and recovery itself would loop.
        """
        ts_ms = int(time.time() * 1000)
        await self._emit_lifecycle_event(
            key=ActionPolicyLifecycleProtocol.codegen_retry_key(ts_ms),
            payload={
                "attempt": attempt,
                "max_attempts": max_attempts,
                "error": (last_error or "")[:300],
                "finished": finished,
                "succeeded": succeeded,
                "ts": time.time(),
            },
        )

    async def _setup_enriched_namespace(self) -> None:
        """Install helper functions into the REPL namespace."""
        repl = self._action_dispatcher.repl
        if not repl or self._run_helper_installed:
            return

        if not repl._shell:
            raise RuntimeError("REPL shell not available for namespace enrichment")
        ns = repl._shell.user_ns

        # run() — dispatch an action through the dispatcher
        async def run(action_key: str, **params) -> ActionResult:
            """Execute a capability action through the dispatcher.

            Args:
                action_key: The action key (e.g., "analyze_pages")
                **params: Parameters for the action

            Returns:
                ActionResult with success, output, error fields
            """
            # Runtime guardrail check
            decision = await self._runtime_guardrail.check(
                action_key=action_key,
                params=params,
                call_history=self._call_history,
            )
            if not decision.allowed:
                logger.warning(f"Guardrail blocked '{action_key}': {decision.reason}")
                self._run_call_trace.append({
                    "call_index": len(self._run_call_trace),
                    "action_key": action_key,
                    "parameters": dict(params),
                    "success": False,
                    "error": f"Blocked: {decision.reason}"[:200],
                    "output_preview": "",
                    "blocked": True,
                })
                ns["_run_call_trace"] = self._run_call_trace
                return ActionResult(
                    success=False,
                    error=f"Blocked by guardrail: {decision.reason}. {decision.suggestion}",
                )

            import uuid
            action = Action(
                action_id=f"codegen_internal_{uuid.uuid4().hex[:8]}",
                agent_id=self.agent.agent_id,
                action_type=action_key,
                parameters=params,
            )
            # Publish a generic ``policy:action_started:*`` lifecycle
            # event on the agent's blackboard. Subscribers — chat UI
            # bridge, traces, log adapters — decide what to do with it.
            # The policy itself does not know what a chat or a UI is.
            _started_at = time.time()
            await self._emit_lifecycle_event(
                key=ActionPolicyLifecycleProtocol.action_started_key(
                    action.action_id,
                ),
                payload={
                    "agent_id": self.agent.agent_id,
                    "action_id": action.action_id,
                    "action_key": action_key,
                    "parameters": dict(params),
                    "started_at": _started_at,
                },
            )

            # Dispatch under a try/finally so the matching action_completed
            # event ALWAYS fires — even when dispatch raises CancelledError
            # (REPL wait_for timeout, /abort) or any other exception.
            # Without this, a started event is never paired and the chat-UI
            # action-status banner spins forever after the action that
            # actually finished but raised on the way back.
            result: ActionResult
            exc_to_reraise: BaseException | None = None
            try:
                result = await self._action_dispatcher.dispatch(action)
            except asyncio.CancelledError as e:
                # Cancellation must propagate — the REPL cell needs to
                # see it (so the LLM-generated code stops executing
                # subsequent statements). But we still need to finalise
                # the lifecycle event so the UI doesn't dangle.
                result = ActionResult(
                    success=False,
                    completed=True,
                    cancelled=True,
                    error="Action cancelled mid-flight",
                )
                exc_to_reraise = e
            except Exception as e:
                # Same intent for non-cancel failures: lifecycle must close,
                # then re-raise so the cell sees the real exception.
                result = ActionResult(
                    success=False,
                    completed=True,
                    error=f"{type(e).__name__}: {e}",
                )
                exc_to_reraise = e

            try:
                # Track result in structured PlanExecutionContext
                self._execution_context.completed_action_ids.append(action.action_id)
                self._execution_context.action_results[action.action_id] = result

                # Track internal failures so the skill library doesn't store
                # code that "executed" but had failing actions inside it.
                if not result.success:
                    self._had_internal_failures = True
                    self._internal_errors.append(result.error or "unknown error")

                resolved_action_key = str(action.action_type)
                self._call_history.append(resolved_action_key)

                # Per-call trace for timeline view annotations.
                # Stored both on the policy and in the REPL namespace so
                # _execute_repl_code can include it in the ActionResult.
                self._run_call_trace.append({
                    "call_index": len(self._run_call_trace),
                    "action_key": resolved_action_key,
                    "parameters": dict(params),
                    "success": result.success,
                    "error": (result.error or "")[:200] if not result.success else None,
                    "output_preview": str(result.output)[:200] if result.output is not None else "",
                    "blocked": False,
                })
                ns["_run_call_trace"] = self._run_call_trace

                # Mode transitions only apply when dispatch returned a real
                # result. After an exception there is no meaningful action
                # output to drive a transition.
                if exc_to_reraise is None:
                    # Mode transitions based on action results:
                    # - should_replan returning should_replan=True → back to planning mode
                    # - First domain action in planning mode → switch to execution mode
                    action_lower = resolved_action_key.lower()
                    if "should_replan" in action_lower and result.success:
                        output = result.output if isinstance(result.output, dict) else {}
                        if output.get("should_replan"):
                            self._mode = "planning"
                            log(f"Mode → PLANNING (replan triggered: {output.get('reason', '')})")

                    if self._mode == "planning" and result.success and not self._is_planning_action_key(resolved_action_key):
                        # First non-planning action → switch to execution mode
                        self._mode = "execution"
                        log("Mode → EXECUTION (first domain action dispatched)")

                # Pair the started event with a terminal one so the
                # downstream subscribers can finalise their state
                # (clear a UI banner, close a span, write a log line).
                ended_at = time.time()
                await self._emit_lifecycle_event(
                    key=ActionPolicyLifecycleProtocol.action_completed_key(
                        action.action_id,
                    ),
                    payload={
                        "agent_id": self.agent.agent_id,
                        "action_id": action.action_id,
                        "action_key": resolved_action_key,
                        "success": result.success,
                        "cancelled": result.cancelled,
                        "started_at": _started_at,
                        "ended_at": ended_at,
                        "wall_time_ms": int((ended_at - _started_at) * 1000),
                        "error": (result.error or None) if not result.success else None,
                    },
                )
            finally:
                if exc_to_reraise is not None:
                    raise exc_to_reraise

            return result

        # browse() — progressive capability discovery
        async def browse(query: str | None = None) -> dict[str, Any] | str:
            """Browse available capabilities.

            Args:
                query: None for groups, "group" for detail, "group.action" for full docs.
            """
            return await self._browser(query)

        # signal_completion() — signal that all goals are achieved (validated)
        async def signal_completion():
            """Signal that all goals are achieved. Call this when done.

            Validates completion against the configured CompletionValidator.
            If validation fails, completion is rejected and the rejection
            reason appears in the next iteration's prompt.
            """
            goals = list(self.agent.metadata.goals or [])
            repl_results = ns.get("results", {})

            validation = await self._completion_validator.validate(
                agent=self.agent,
                goals=goals,
                results=repl_results,
                execution_context=self._execution_context,
            )

            if validation.allowed:
                self._complete_signaled = True
                log(f"Completion validated: {validation.reason}")
                self._run_call_trace.append({
                    "call_index": len(self._run_call_trace),
                    "action_key": "signal_completion",
                    "success": True,
                    "error": None,
                    "output_preview": f"Completion accepted: {validation.reason}"[:200],
                    "blocked": False,
                })
                ns["_run_call_trace"] = self._run_call_trace
            else:
                log(f"Completion REJECTED: {validation.reason}")
                if validation.suggestions:
                    log(f"Suggestions: {'; '.join(validation.suggestions)}")
                self._run_call_trace.append({
                    "call_index": len(self._run_call_trace),
                    "action_key": "signal_completion",
                    "success": False,
                    "error": f"Rejected: {validation.reason}"[:200],
                    "output_preview": "; ".join(validation.suggestions)[:200] if validation.suggestions else "",
                    "blocked": False,
                })
                ns["_run_call_trace"] = self._run_call_trace
                self._execution_context.custom_data["last_completion_rejection"] = {
                    "reason": validation.reason,
                    "suggestions": validation.suggestions,
                }

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
        _LOG_MAX_CHARS = 500

        def log(msg: str):
            """Log a message visible in the execution trace."""
            truncated = msg[:_LOG_MAX_CHARS] + "..." if len(msg) > _LOG_MAX_CHARS else msg
            logger.info(f"[CodeGen:{self.agent.agent_id}] {truncated}")

        # Install into namespace
        ns["run"] = run
        ns["browse"] = browse
        if self._allow_self_termination:
            ns["signal_completion"] = signal_completion
        ns["switch_mode"] = switch_mode
        ns["log"] = log
        ns["results"] = {}
        ns["pages"] = []
        ns["goals"] = list(self.agent.metadata.goals or [])
        ns["params"] = dict(self.agent.metadata.parameters or {})

        # bb — the agent's primary blackboard (lazy, since it needs await)
        # We'll set it on first use
        ns["bb"] = None

        # Track which keys we installed so execute_iteration's namespace
        # snapshot/restore knows not to roll them back.
        self._enriched_ns_keys = frozenset(ns.keys()) - {'In', 'Out', 'get_ipython'}
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

        # Update goals and params
        ns["goals"] = list(self.agent.metadata.goals or [])
        ns["params"] = dict(self.agent.metadata.parameters or {})

        # Lazy-init blackboard
        if ns.get("bb") is None:
            try:
                ns["bb"] = await self.agent.get_blackboard()
            except Exception:
                pass

    @staticmethod
    def _build_skill_query(planning_context: PlanningContext) -> str:
        """Build a meaningful skill retrieval query from structured planning context.

        Combines goals, recent action names (from step summaries, not opaque
        IDs), and findings into a query that reflects what the agent needs to
        accomplish *right now*, not just its static goals.
        """
        parts: list[str] = []

        # Current goals
        if planning_context.goals:
            parts.append("Goals: " + "; ".join(planning_context.goals))

        # Recent execution progress — what has already been done
        exec_ctx = planning_context.execution_context
        if exec_ctx:
            # Use step summaries to get real action names, not codegen_internal_* IDs
            step_summaries = exec_ctx.custom_data.get("codegen_step_summaries", {})
            recent_actions: list[str] = []
            for step_info in list(step_summaries.values())[-3:]:  # TODO: Make this configurable
                for action_key in step_info.get("actions_called", []):
                    recent_actions.append(action_key.rsplit(".", 1)[-1])
            if recent_actions:
                parts.append("Recent actions: " + ", ".join(recent_actions))

            if exec_ctx.findings:
                finding_keys = list(exec_ctx.findings.keys())[:5]  # TODO: Make this configurable
                parts.append("Findings: " + ", ".join(finding_keys))

            # Cache state narrows to data-relevant skills
            if exec_ctx.analyzed_pages:
                parts.append(f"Analyzed {len(exec_ctx.analyzed_pages)} pages")

        return " | ".join(parts) if parts else "general agent task"

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
            logger.warning(
                "CodeGenerationActionPolicy: policy complete signaled by generated code"
            )
            state.custom["policy_complete"] = True
            return None

        # Check iteration limit (incremented after validation, not before,
        # so validation failures don't consume iterations)
        if self._code_iteration_count >= self.max_code_iterations:
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
        # Mode-specific tag filtering for action descriptions
        if self._mode == "planning":
            include_tags = frozenset({"planning"})
            exclude_tags = None
        else:
            include_tags = None
            exclude_tags = frozenset({"planning"})

        # Build full planning context via PlanningContextBuilder. This gathers memories,
        # builds identity from ConsciousnessCapability, extracts constraints,
        # and includes action descriptions.
        planning_context = await self._context_builder.get_planning_context(
            execution_context=self._execution_context,
        )
        # Override action descriptions with mode-filtered versions
        planning_context.action_descriptions = await self.get_action_descriptions(
            include_tags=include_tags,
            exclude_tags=exclude_tags,
        )

        # Format the PlanningContext for code generation, including full
        # error history so the LLM can see ALL prior failed attempts.
        prompt = format_planning_context_for_codegen(
            planning_context=planning_context,
            mode=self._mode,
            error_history=self._error_history if self._error_history else None,
            allow_self_termination=self._allow_self_termination,
        )

        # Clear completion rejection after it's been rendered into the prompt
        self._execution_context.custom_data.pop("last_completion_rejection", None)

        # --- Dimension 1: Code Generation (or use recovered code) ---
        if self._recovered_code:
            code = self._recovered_code
            self._recovered_code = None
            logger.info("CodeGenerationActionPolicy: using recovered code from previous failure")
        else:
            # Dimension 3: Skill Library — only execution mode reuses prior code.
            # Planning mode already has explicit planning capabilities and learned
            # planning patterns; replaying code examples here tends to lock the
            # model into repeating planning stubs instead of moving to execution.
            skills: list[Any] = []
            if self._mode == "execution":
                skill_query = self._build_skill_query(planning_context)
                planning_action_keys = self._get_planning_action_keys()
                retrieved_skills = await self._skill_library.retrieve(goal=skill_query, k=3)
                skills = _select_prompt_skills(
                    retrieved_skills,
                    mode=self._mode,
                    planning_action_keys=planning_action_keys,
                )
            if skills:
                skill_section = "\n\n## Relevant Examples from Prior Runs\n"
                for s in skills:
                    skill_section += f"# Goal: {s.goal}\n{s.code}\n\n"
                prompt += skill_section

            # Dimension 1: CodeGenerator. Wrap the LLM call in a tracked task
            # so /abort can interrupt it mid-request — codegen prompts can take
            # tens of seconds (especially on cold caches), and waiting that out
            # is exactly the UX problem the abort feature exists to solve.
            self._codegen_cancel_requested = False
            self._current_codegen_task = asyncio.create_task(
                self._code_generator.generate(
                    agent=self.agent,
                    prompt=prompt,
                    max_tokens=2048,
                    temperature=0.3,
                )
            )
            try:
                code = await self._current_codegen_task
            except asyncio.CancelledError:
                if self._codegen_cancel_requested:
                    # User-requested abort. Reset cancel flag, surface it as
                    # a clean "no work this iteration" — abort_current() has
                    # already cleared recovery state and emitted any chat
                    # banner, so plan_step just bails.
                    self._codegen_cancel_requested = False
                    logger.info(
                        "CodeGenerationActionPolicy: codegen LLM call cancelled by user abort"
                    )
                    return None
                # Outer cancellation (e.g., agent shutdown) — propagate.
                raise
            except Exception as e:
                logger.error(f"CodeGenerationActionPolicy: code generation failed: {e}")
                return None
            finally:
                self._current_codegen_task = None

        # Snapshot the raw output right after the LLM returns. We feed
        # it back to the LLM in the error_history when validation fails
        # so the model sees its own invalid output and (hopefully)
        # corrects course on the next attempt.
        raw_llm_output = code if isinstance(code, str) else str(code)

        if not code or not code.strip():
            logger.error(
                "CodeGenerationActionPolicy: generated empty code "
                "(raw LLM output: %r)",
                raw_llm_output[:300],
            )
            await self._handle_codegen_failure(
                code="", error=(
                    "Empty code after extraction. Your last response "
                    "contained no Python code we could parse. Emit ONLY "
                    "raw Python statements — no markdown fences, no "
                    "explanations, no mocked output."
                ),
                raw=raw_llm_output,
            )
            return None

        # --- Dimension 2: Code Validation (all validators must pass) ---
        for validator in self._code_validators:
            validation = await validator.validate(code, self.agent)
            if not validation.valid:
                logger.error(
                    "CodeGenerationActionPolicy: validation failed for "
                    "%s: %s. Raw LLM output (truncated): %r",
                    type(validator).__name__,
                    validation.errors,
                    raw_llm_output[:300],
                )
                # Dimension 4: Recovery from validation failure
                recovery = await self._recovery_strategy.recover(
                    code=code,
                    error="\n".join(validation.errors),
                    validation_result=validation,
                    attempt=self._consecutive_failures,
                    max_attempts=self.max_retries,
                )
                if recovery.recovered and recovery.code:
                    code = recovery.code
                    logger.info(f"CodeGenerationActionPolicy: {recovery.strategy_used} produced fixed code")
                    continue  # re-validate the recovered code
                error_msg = recovery.error_context or "\n".join(validation.errors)
                await self._handle_codegen_failure(
                    code=code, error=error_msg, raw=raw_llm_output,
                )
                return None

        # Clear consecutive failure count on successful generation + validation.
        # If we were recovering from a previous failure, clear the chat
        # banner and the error_history so the next user-driven iteration
        # starts fresh.
        if self._consecutive_failures > 0:
            await self._emit_codegen_recovery_banner(
                attempt=self._consecutive_failures,
                max_attempts=self.max_retries,
                last_error="",
                finished=True,
                succeeded=True,
            )
        self._consecutive_failures = 0
        self._error_history.clear()

        # Reset call history for this code execution (Dimension 5 tracking)
        self._call_history.clear()
        self._run_call_trace = []
        # Also reset in REPL namespace so _execute_repl_code sees empty trace
        # if the code block has no run() calls
        if self._action_dispatcher and self._action_dispatcher.repl:
            self._action_dispatcher.repl.namespace["_run_call_trace"] = self._run_call_trace
        self._had_internal_failures = False
        self._internal_errors.clear()

        self._code_iteration_count += 1

        logger.info(
            f"CodeGenerationActionPolicy: generated code ({len(code)} chars) "
            f"for iteration {self._code_iteration_count}"
        )

        # Return as EXECUTE_CODE action
        return Action(
            action_id=f"codegen_plan_step_{self._code_iteration_count}",
            agent_id=self.agent.agent_id,
            action_type=ActionType.EXECUTE_CODE,
            parameters={"code": code},
            code=code,
            description=f"Code generation step {self._code_iteration_count}",
        )

    @hookable
    @override
    async def execute_iteration(
        self,
        state: ActionPolicyExecutionState
    ) -> ActionPolicyIterationResult:
        """Execute one iteration with transaction wrapper and retry logic.

        This method is @hookable so memory capabilities can observe iterations.

        Wraps the parent's execute_iteration with:
        1. REPL namespace snapshot (restore on failure)
        2. Retry on code execution failure (iterative refinement)
        """
        repl = self._action_dispatcher.repl if self._action_dispatcher else None

        # Snapshot only LLM-introduced variables. Enriched namespace vars
        # (run, browse, results, params, etc.) must NOT be rolled back —
        # results accumulates across iterations, and the helpers are closures.
        exclude = getattr(self, '_enriched_ns_keys', frozenset())
        namespace_snapshot = None
        if repl and hasattr(repl, '_shell') and repl._shell:
            try:
                namespace_snapshot = {
                    k: v for k, v in repl._shell.user_ns.items()
                    if (not k.startswith('_')
                        and k not in ('In', 'Out', 'get_ipython')
                        and k not in exclude)
                }
            except Exception:
                pass

        mode_before = self._mode
        planning_action_keys = self._get_planning_action_keys()

        # Execute (parent handles plan_step → dispatch)
        result = await super().execute_iteration(state)

        # Record what this iteration did — stored in execution context so
        # format_planning_context_for_codegen can show it.
        # The code step's own ID must be in completed_action_ids so the
        # rendering loop can find its summary.
        # (Internal run() calls add codegen_internal_* IDs, but the code
        # step itself — codegen_plan_step_* — was never added, causing
        # the execution history to be completely empty.)
        if (result.action_executed
                and result.action_executed.action_type == ActionType.EXECUTE_CODE):
            step_id = result.action_executed.action_id
            self._execution_context.completed_action_ids.append(step_id)
            self._execution_context.action_results[step_id] = result.result
            summaries = self._execution_context.custom_data.setdefault(
                "codegen_step_summaries", {}
            )
            # Collect all errors: internal run() failures + REPL crash error.
            # Without the REPL error, the execution history shows [✗] with no
            # explanation — the LLM can't learn what went wrong and repeats
            # the same failing code indefinitely.
            step_errors = list(self._internal_errors)
            if result.result and not result.result.success and result.result.error:
                step_errors.append(result.result.error)

            summaries[step_id] = _build_codegen_step_summary(
                actions_called=list(self._call_history),
                planning_action_keys=planning_action_keys,
                had_failures=self._had_internal_failures,
                repl_success=result.result.success if result.result else False,
                errors=step_errors,
                mode_before=mode_before,
                mode_after=self._mode,
                run_call_trace=list(self._run_call_trace),
            )

        # Feed this iteration's action calls to every consciousness stream.
        # Each stream decides independently (via its action_filter) what to
        # record. Streams bound to this policy surface in the planning prompt.
        for call in self._run_call_trace:
            for stream in self._consciousness_streams:
                stream.consider_action(call)

        # Check if code execution failed
        if (result.result and not result.result.success
                and result.action_executed
                and result.action_executed.action_type == ActionType.EXECUTE_CODE):

            error_msg = result.result.error or "Unknown error"
            failed_code = result.action_executed.parameters.get("code", "")

            logger.warning(
                f"CodeGenerationActionPolicy: code execution failed: {error_msg[:200]}"
            )

            # Restore LLM-introduced variables on failure
            if namespace_snapshot and repl and hasattr(repl, '_shell') and repl._shell:
                try:
                    # Only restore user-defined variables, not system ones
                    for key in list(repl._shell.user_ns.keys()):
                        if (not key.startswith('_')
                                and key not in ('In', 'Out', 'get_ipython')
                                and key not in exclude
                                and key not in namespace_snapshot):
                            del repl._shell.user_ns[key]
                    for key, val in namespace_snapshot.items():
                        repl._shell.user_ns[key] = val
                except Exception as e:
                    logger.warning(f"Failed to restore namespace: {e}")

            # Dimension 4: Recovery Strategy
            self._consecutive_failures += 1
            recovery = await self._recovery_strategy.recover(
                code=failed_code,
                error=error_msg,
                validation_result=None,
                attempt=self._consecutive_failures,
                max_attempts=self.max_retries,
            )

            if recovery.recovered and recovery.code:
                # Recovery produced fixed code — use it on the next iteration
                self._recovered_code = recovery.code
                logger.info(
                    f"CodeGenerationActionPolicy: {recovery.strategy_used} "
                    f"produced recovered code, will execute next iteration"
                )
            else:
                # Accumulate error into history so the LLM sees ALL prior
                # failures and doesn't repeat the same key format mistakes.
                self._error_history.append({
                    "code": failed_code,
                    "error": recovery.error_context or error_msg,
                })

            # Return failure but don't complete — retry on next iteration
            return ActionPolicyIterationResult(
                success=False,
                policy_completed=False,
                action_executed=result.action_executed,
                result=result.result,
            )

        # Successful code execution
        if (result.result and result.result.success
                and result.action_executed
                and result.action_executed.action_type == ActionType.EXECUTE_CODE):
            self._consecutive_failures = 0
            self._error_history.clear()
            executed_code = result.action_executed.parameters.get("code", "")
            step_summaries = self._execution_context.custom_data.get("codegen_step_summaries", {})
            step_info = step_summaries.get(result.action_executed.action_id)

            # Dimension 3: Skill Library — store successful code for reuse.
            # Only store if no run() calls inside the code failed and the step
            # actually performed domain work. Planning-only stubs pollute the
            # prompt and cause the model to replay the planning preamble.
            if (
                executed_code
                and not self._had_internal_failures
                and _should_store_skill_from_step_summary(step_info)
            ):
                try:
                    planning_context = await self._context_builder.get_planning_context(
                        execution_context=self._execution_context,
                    )
                    skill_goal = self._build_skill_query(planning_context)
                    await self._skill_library.store(
                        code=executed_code,
                        goal=skill_goal,
                        result=result.result,
                        description=result.action_executed.description or "",
                    )
                except Exception as e:
                    logger.debug(f"Failed to store skill: {e}")

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
    context_builder: PlanningContextBuilder | None = None,
    code_generator: CodeGenerator | None = None,
    code_validators: list[CodeValidator] | None = None,
    skill_library: SkillLibrary | None = None,
    recovery_strategy: RecoveryStrategy | None = None,
    runtime_guardrail: RuntimeGuardrail | None = None,
    completion_validator: CompletionValidator | None = None,
    max_retries: int = 2,
    code_timeout: float = 30.0,
    max_code_iterations: int = 50,
    allow_self_termination: bool = True,
    reactive_only: bool = False,
    planning_capability_blueprints: list[Any] | None = None,
    consciousness_streams: list[Any] | None = None,
) -> CodeGenerationActionPolicy:
    """Create a code-generation-based action policy.

    The LLM generates Python code that composes ``@action_executor`` methods
    with real control flow, instead of selecting from a JSON action schema.

    Requires ``REPLCapability`` on the agent.
    Uses ``PlanningContextBuilder`` to build structured planning context
    including memories, agent identity, constraints, and action descriptions.

    Each of the six constraint dimensions accepts an implementation of the
    corresponding abstract class. Pass ``None`` for defaults::

        policy = await create_code_generation_action_policy(
            agent=agent,
            code_validators=[APIKnowledgeBaseValidator(agent)],
            runtime_guardrail=CapabilityBoundaryGuardrail(
                allowed_prefixes=["CacheAnalysis", "Analysis"],
            ),
        )

    Args:
        agent: The owning agent.
        action_map: Optional pre-built action groups.
        action_providers: Additional action providers.
        io: Action policy I/O configuration.
        context_builder: Planning context builder (default: ``PlanningContextBuilder(agent)``).
        code_generator: How code is produced (default: ``FreeFormCodeGenerator``).
        code_validators: Pre-execution validation (default: ``[IterationShapeValidator()]``).
        skill_library: Skill storage/retrieval (default: ``NoOpSkillLibrary``).
        recovery_strategy: Failure recovery (default: ``DeterministicRecovery``).
        runtime_guardrail: Runtime constraints (default: ``NoGuardrail``).
        max_retries: Max retries on code execution failure.
        code_timeout: Timeout for each code execution.
        max_code_iterations: Max code generation iterations.
        allow_self_termination: If True, generated code can signal completion (for reactive agents).
        reactive_only: If True, only include capabilities tagged "reactive" in the prompt (for reactive agents).

    Returns:
        CodeGenerationActionPolicy
    """
    action_policy = CodeGenerationActionPolicy(
        agent=agent,
        context_builder=context_builder,
        code_generator=code_generator,
        code_validators=code_validators,
        skill_library=skill_library,
        recovery_strategy=recovery_strategy,
        runtime_guardrail=runtime_guardrail,
        completion_validator=completion_validator,
        max_retries=max_retries,
        code_timeout=code_timeout,
        max_code_iterations=max_code_iterations,
        allow_self_termination=allow_self_termination,
        planning_capability_blueprints=planning_capability_blueprints,
        consciousness_streams=consciousness_streams,
        reactive_only=reactive_only,
        action_map=action_map,
        action_providers=action_providers,
        io=io,
    )
    await action_policy.initialize()

    return action_policy
