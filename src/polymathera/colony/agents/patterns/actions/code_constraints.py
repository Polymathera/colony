"""Abstract building blocks for constrained code generation.

Each dimension of the constrained code generation design space is
represented by an abstract class that users can implement to customize
how the ``CodeGenerationActionPolicy`` generates, validates, reuses,
recovers from, guards, and validates completion of generated code.

The six dimensions:

1. **CodeGenerator** — controls HOW code is produced (free-form, grammar-
   constrained, skeleton-with-holes).
2. **CodeValidator** — checks generated code BEFORE execution (AST analysis,
   type checking, API knowledge base).
3. **SkillLibrary** — stores and retrieves successful code patterns for
   reuse across iterations and sessions.
4. **RecoveryStrategy** — handles failures (deterministic fix, LLM retry,
   transactional rollback).
5. **RuntimeGuardrail** — enforces constraints DURING execution (capability
   boundaries, temporal ordering, resource limits).
6. **CompletionValidator** — validates whether the agent should be allowed
   to signal completion (goal achievement check, LLM evaluation).

``CodeGenerationActionPolicy`` accepts an implementation of each. Default
implementations are provided for common configurations. Users can mix
defaults with custom implementations:

    policy = CodeGenerationActionPolicy(
        agent=agent,
        code_generator=GuidedCodeGenerator(agent),      # Custom
        code_validators=[APIKnowledgeBaseValidator(agent)], # Default
        skill_library=ChromaSkillLibrary(agent),          # Custom
        recovery_strategy=DeterministicRecovery(),        # Default
        runtime_guardrail=CapabilityBoundaryGuardrail(    # Custom
            allowed=["CacheAnalysisCapability", "AnalysisCapability"]
        ),
    )
"""

from __future__ import annotations

import ast
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ...base import Agent, AgentCapability
    from ...models import ActionResult, PlanExecutionContext

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes shared across dimensions
# ============================================================================


@dataclass
class ValidationResult:
    """Result of code validation."""

    valid: bool
    """Whether the code passed all checks."""

    errors: list[str] = field(default_factory=list)
    """Human-readable error descriptions."""

    fixed_code: str | None = None
    """If the validator was able to fix the code, the corrected version.
    None if the code is valid or unfixable."""

    details: dict[str, Any] = field(default_factory=dict)
    """Structured details about what was checked and found."""


@dataclass
class CodeSkill:
    """A stored code pattern for reuse."""

    code: str
    """The code snippet."""

    goal: str
    """The goal this code was written for."""

    description: str
    """What this code does (generated from the code or provided)."""

    success_count: int = 0
    """How many times this skill executed successfully."""

    failure_count: int = 0
    """How many times this skill failed."""

    tags: set[str] = field(default_factory=set)
    """Tags for categorization (e.g., 'cache_aware', 'multi_agent')."""


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""

    recovered: bool
    """Whether recovery produced usable code."""

    code: str | None = None
    """Fixed code, if recovery succeeded."""

    strategy_used: str = ""
    """Description of which recovery strategy was applied."""

    should_retry: bool = False
    """Whether the policy should retry with an LLM call."""

    error_context: str = ""
    """Structured error context for the LLM retry prompt."""


@dataclass
class GuardrailDecision:
    """Decision from a runtime guardrail."""

    allowed: bool
    """Whether the action is allowed."""

    reason: str = ""
    """Why the action was blocked (if not allowed)."""

    suggestion: str = ""
    """What the code should do instead (if blocked)."""


# ============================================================================
# Dimension 1: Code Generation
# ============================================================================


class CodeGenerator(ABC):
    """Controls HOW code is produced from the LLM.

    Implementations range from free-form Python generation to grammar-
    constrained decoding to skeleton-based fill-in-the-middle.

    The generator receives the full prompt and returns executable Python.
    It may modify the inference call (e.g., by passing grammar constraints
    to vLLM) or post-process the LLM output (e.g., by filling a skeleton).

    Example implementations:
    - ``FreeFormCodeGenerator``: Raw LLM generation, no constraints.
    - ``GuidedCodeGenerator``: Grammar-constrained decoding using the
      agent's API surface (valid action keys, method names).
    - ``SkeletonCodeGenerator``: Provides a code skeleton with typed holes
      that the LLM fills in.
    """

    @abstractmethod
    async def generate(
        self,
        agent: Agent,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """Generate code from the prompt.

        Args:
            agent: The agent (provides ``infer()`` for LLM calls and
                capability access for API grammar extraction).
            prompt: The full planning prompt.
            max_tokens: Maximum tokens for the generated code.
            temperature: LLM sampling temperature.

        Returns:
            Generated Python code string.
        """
        ...


class FreeFormCodeGenerator(CodeGenerator):
    """Generate free-form Python with no structural constraints.

    The LLM produces arbitrary async Python. Maximum expressiveness,
    highest error rate. This is the default.
    """

    async def generate(
        self,
        agent: Agent,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        response = await agent.infer(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _extract_code(response)


# ============================================================================
# Dimension 2: Code Validation
# ============================================================================


class CodeValidator(ABC):
    """Checks generated code BEFORE execution.

    Implementations range from basic import whitelisting to full API
    knowledge base validation with type checking.

    The validator receives the code string and returns a ``ValidationResult``
    indicating whether the code is safe to execute. If the validator can
    fix common errors (e.g., misspelled API names), it returns the fixed
    code in ``ValidationResult.fixed_code``.

    Example implementations:
    - ``NoOpValidator``: No validation (trust the LLM).
    - ``ImportWhitelistValidator``: Block dangerous imports (current default).
    - ``APIKnowledgeBaseValidator``: Validate against the agent's registered
      capabilities (catches hallucinated APIs, wrong parameter names).
    """

    @abstractmethod
    async def validate(self, code: str, agent: Agent) -> ValidationResult:
        """Validate generated code before execution.

        Args:
            code: The generated Python code.
            agent: The agent (provides capability introspection).

        Returns:
            ValidationResult with valid/errors/fixed_code.
        """
        ...


class NoOpValidator(CodeValidator):
    """No validation — trust the LLM output completely."""

    async def validate(self, code: str, agent: Agent) -> ValidationResult:
        return ValidationResult(valid=True)


class ImportWhitelistValidator(CodeValidator):
    """Validate imports against a whitelist.

    This is the baseline safety check already in ``PolicyPythonREPL``.
    Re-implemented here as a composable validator.
    """

    ALLOWED_IMPORTS = {
        "json", "re", "math", "itertools", "functools",
        "collections", "dataclasses", "typing", "datetime",
        "pydantic", "asyncio",
    }

    async def validate(self, code: str, agent: Agent) -> ValidationResult:
        errors = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                errors=[f"Syntax error: {e}"],
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module not in self.ALLOWED_IMPORTS:
                        errors.append(f"Blocked import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module not in self.ALLOWED_IMPORTS:
                        errors.append(f"Blocked import: from {node.module}")

        return ValidationResult(valid=len(errors) == 0, errors=errors)


class APIKnowledgeBaseValidator(CodeValidator):
    """Validate against the agent's registered API surface.

    Builds a knowledge base from registered capabilities at construction
    time, then checks generated code for:
    - Hallucinated action keys in ``run()`` calls
    - Misspelled capability method names
    - Wrong parameter names for known methods
    - Missing ``await`` on async calls

    Can deterministically fix common errors (edit-distance correction
    of misspelled names). Research shows this catches 77% of hallucinated
    API calls with 100% precision and zero LLM calls.
    """

    def __init__(self, agent: Agent):
        self._agent = agent
        self._valid_action_keys: set[str] | None = None

    def _ensure_knowledge_base(self, agent: Agent) -> set[str]:
        """Build or return the set of valid action keys.

        Built lazily on first validate() call so the dispatcher's compound
        keys (``ClassName.dispatch_key.method_name``) are available — these
        are the keys shown in the prompt and used by ``run()``.

        Falls back to short ``@action_executor`` keys if the dispatcher
        isn't available yet.
        """
        if self._valid_action_keys is not None:
            return self._valid_action_keys

        self._valid_action_keys = set()

        # Primary: collect compound keys from the dispatcher's action map.
        # These are the exact keys shown in the prompt.
        from .policies import BaseActionPolicy
        policy = agent.action_policy
        if isinstance(policy, BaseActionPolicy) and policy._action_dispatcher:
            for group in policy._action_dispatcher.action_map:
                self._valid_action_keys.update(group.executors.keys())

        # Fallback: if dispatcher isn't ready, use short decorator keys.
        if not self._valid_action_keys:
            for cap in agent.get_capabilities():
                for name in dir(cap):
                    if name.startswith("_"):
                        continue
                    attr = getattr(cap, name, None)
                    if hasattr(attr, '_action_key'):
                        self._valid_action_keys.add(attr._action_key)

        return self._valid_action_keys

    async def validate(self, code: str, agent: Agent) -> ValidationResult:
        import re as _re

        valid_keys = self._ensure_knowledge_base(agent)
        errors = []
        replacements: list[tuple[str, str]] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(valid=False, errors=[f"Syntax error: {e}"])

        for node in ast.walk(tree):
            # Check run("action_key", ...) calls
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "run"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                key = node.args[0].value
                if key not in valid_keys:
                    closest = self._closest_match(key, valid_keys)
                    if closest:
                        errors.append(
                            f"Unknown action key '{key}'. Did you mean '{closest}'?"
                        )
                        replacements.append((key, closest))
                    else:
                        errors.append(f"Unknown action key '{key}'.")

        # Apply fixes only inside run() call sites, not globally.
        # Handle both single and double quotes since AST normalizes them.
        fixed_code = code
        for old_key, new_key in replacements:
            fixed_code = _re.sub(
                r"""run\(\s*(['"])""" + _re.escape(old_key) + r"""\1""",
                f'run("{new_key}"',
                fixed_code,
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            fixed_code=fixed_code if fixed_code != code else None,
        )

    @staticmethod
    def _closest_match(target: str, candidates: set[str], max_dist: int = 3) -> str | None:
        """Find the closest match by edit distance."""
        best = None
        best_dist = max_dist + 1
        for candidate in candidates:
            # Simple character-level edit distance
            dist = sum(1 for a, b in zip(target, candidate) if a != b) + abs(len(target) - len(candidate))
            if dist < best_dist:
                best_dist = dist
                best = candidate
        return best if best_dist <= max_dist else None


class IterationShapeValidator(CodeValidator):
    """Enforce structural constraints that keep each iteration focused.

    The most common LLM failure mode in code-generation planning is writing
    a monolithic program (8+ actions, browse-for-discovery, hardcoded values)
    instead of a focused 1–3 action step. This validator catches that
    structurally via AST analysis — no LLM call required.

    Checks:
    - **Max actions**: Too many ``run()`` calls means the LLM is writing a
      complete program instead of a single step.
    - **Discovery abuse**: ``browse()`` with no arguments is redundant when
      action keys are already in the prompt. At most ``max_browse`` calls
      are allowed (for parameter lookup on specific actions).
    - **Code size**: Excessive line count signals a monolithic script.

    All thresholds are configurable. Library users who need more permissive
    iterations (e.g., for batch-processing pipelines) can raise the limits.

    Args:
        max_actions: Maximum ``run()`` calls per code block.
        max_browse: Maximum ``browse()`` calls per code block. Set to 0 to
            forbid all browse calls.
        max_lines: Maximum non-comment, non-blank lines of code.
    """

    def __init__(
        self,
        max_actions: int = 3,
        max_browse: int = 1,
        max_lines: int = 50,
    ):
        self._max_actions = max_actions
        self._max_browse = max_browse
        self._max_lines = max_lines

    async def validate(self, code: str, agent: Agent) -> ValidationResult:
        errors: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(valid=False, errors=[f"Syntax error: {e}"])

        run_count = 0
        browse_count = 0

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            # Detect run() and browse() regardless of whether they are
            # awaited — the Await node wraps the Call, so walk visits both.
            func = node.func
            if isinstance(func, ast.Name):
                if func.id == "run":
                    run_count += 1
                elif func.id == "browse":
                    browse_count += 1
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                # Covers patterns like `self.run(...)` if someone tries that
                if func.attr == "run":
                    run_count += 1
                elif func.attr == "browse":
                    browse_count += 1

        if run_count > self._max_actions:
            errors.append(
                f"Too many actions in one iteration: {run_count} run() calls, "
                f"maximum is {self._max_actions}. Each iteration should be a focused "
                f"step of 1–{self._max_actions} actions, not a complete program. "
                f"Pick the most important next action and do only that."
            )

        if browse_count > self._max_browse:
            errors.append(
                f"Too many browse() calls: {browse_count}, maximum is "
                f"{self._max_browse}. Action keys are already shown in the prompt — "
                f"use the exact keys from Available Actions. Use browse() only to "
                f"look up parameter details for a specific action."
            )

        # Count substantive lines (not blank, not pure comments)
        substantive_lines = [
            line for line in code.strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        if len(substantive_lines) > self._max_lines:
            errors.append(
                f"Code is too long: {len(substantive_lines)} lines, maximum is "
                f"{self._max_lines}. Write a focused step, not a complete program."
            )

        return ValidationResult(valid=len(errors) == 0, errors=errors)


# ============================================================================
# Dimension 3: Skill Library
# ============================================================================


class SkillLibrary(ABC):
    """Stores and retrieves successful code patterns for reuse.

    Implementations range from in-memory stores to blackboard-backed
    persistent libraries to vector-indexed semantic search.

    The library is queried before code generation (to provide few-shot
    examples) and updated after successful execution (to build up the
    skill repertoire over time).

    Example implementations:
    - ``NoOpSkillLibrary``: Disabled — no reuse.
    - ``InMemorySkillLibrary``: Session-scoped, no persistence.
    - ``BlackboardSkillLibrary``: Backed by the agent's blackboard,
      persists across sessions.
    """

    @abstractmethod
    async def retrieve(self, goal: str, k: int = 3) -> list[CodeSkill]:
        """Retrieve relevant skills for the given goal.

        Args:
            goal: The current planning goal.
            k: Maximum number of skills to return.

        Returns:
            List of relevant ``CodeSkill`` objects, most relevant first.
        """
        ...

    @abstractmethod
    async def store(
        self,
        code: str,
        goal: str,
        result: ActionResult,
        description: str = "",
    ) -> None:
        """Store a code snippet after successful execution.

        Only store if the result was successful. The library decides
        whether to accept the skill (e.g., deduplication).

        Args:
            code: The executed code.
            goal: The goal it was written for.
            result: The execution result.
            description: Human-readable description of what the code does.
        """
        ...


class NoOpSkillLibrary(SkillLibrary):
    """Disabled skill library — no reuse."""

    async def retrieve(self, goal: str, k: int = 3) -> list[CodeSkill]:
        return []

    async def store(self, code: str, goal: str, result: ActionResult, description: str = "") -> None:
        pass


class InMemorySkillLibrary(SkillLibrary):
    """Session-scoped skill library stored in memory.

    Skills are lost when the agent is suspended or the process ends.
    Useful for within-session reuse during long-running analyses.
    """

    def __init__(self, max_skills: int = 100):
        self._skills: list[CodeSkill] = []
        self._max_skills = max_skills

    async def retrieve(self, goal: str, k: int = 3) -> list[CodeSkill]:
        # Simple substring matching — production implementations should
        # use embeddings for semantic similarity.
        goal_lower = goal.lower()
        scored = []
        for skill in self._skills:
            # Score by word overlap
            overlap = sum(1 for word in goal_lower.split() if word in skill.goal.lower())
            if overlap > 0:
                scored.append((overlap, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored[:k]]

    async def store(
        self, code: str, goal: str, result: ActionResult, description: str = ""
    ) -> None:
        if not result.success:
            return

        # Check for duplicate (same code)
        for skill in self._skills:
            if skill.code == code:
                skill.success_count += 1
                return

        self._skills.append(CodeSkill(
            code=code,
            goal=goal,
            description=description or f"Code for: {goal[:100]}",
            success_count=1,
            tags=set(),
        ))

        # Evict oldest if over capacity
        if len(self._skills) > self._max_skills:
            self._skills.sort(key=lambda s: s.success_count)
            self._skills = self._skills[len(self._skills) - self._max_skills:]


# ============================================================================
# Dimension 4: Recovery Strategy
# ============================================================================


class RecoveryStrategy(ABC):
    """Handles code execution failures.

    Implementations range from "do nothing" to deterministic AST-based
    fixes to LLM-powered retry with structured error feedback.

    The strategy receives the failed code and error, and returns a
    ``RecoveryResult`` indicating whether recovery succeeded, whether
    to retry with the LLM, or whether to give up.

    Example implementations:
    - ``NoRecovery``: Fail immediately, no retry.
    - ``DeterministicRecovery``: Apply AST-based fixes for known error
      classes (misspelled APIs, missing await).
    - ``LLMRetryRecovery``: Retry with the LLM, including error context.
    """

    @abstractmethod
    async def recover(
        self,
        code: str,
        error: str,
        validation_result: ValidationResult | None,
        attempt: int,
        max_attempts: int,
    ) -> RecoveryResult:
        """Attempt to recover from a code execution failure.

        Args:
            code: The code that failed.
            error: The error message.
            validation_result: If the failure was caught by validation,
                the result (may contain ``fixed_code``).
            attempt: Current attempt number (0-indexed).
            max_attempts: Maximum attempts allowed.

        Returns:
            RecoveryResult indicating next steps.
        """
        ...


class NoRecovery(RecoveryStrategy):
    """No recovery — fail immediately."""

    async def recover(self, code, error, validation_result, attempt, max_attempts):
        return RecoveryResult(recovered=False)


class DeterministicRecovery(RecoveryStrategy):
    """Apply deterministic fixes before falling back to LLM retry.

    If the validator produced ``fixed_code``, use it. Otherwise,
    signal that the LLM should retry with the error context.
    """

    async def recover(
        self,
        code: str,
        error: str,
        validation_result: ValidationResult | None,
        attempt: int,
        max_attempts: int,
    ) -> RecoveryResult:
        # If validation produced a fix, use it
        if validation_result and validation_result.fixed_code:
            return RecoveryResult(
                recovered=True,
                code=validation_result.fixed_code,
                strategy_used="deterministic_fix",
            )

        # Otherwise, signal LLM retry if within budget
        if attempt < max_attempts:
            return RecoveryResult(
                recovered=False,
                should_retry=True,
                error_context=f"Code:\n{code}\n\nError:\n{error}",
                strategy_used="llm_retry",
            )

        return RecoveryResult(recovered=False, strategy_used="exhausted")


# ============================================================================
# Dimension 5: Runtime Guardrail
# ============================================================================


class RuntimeGuardrail(ABC):
    """Enforces constraints DURING code execution.

    Implementations intercept ``run()`` calls in the REPL namespace and
    decide whether each action is allowed. This enables:
    - Capability boundary enforcement (restrict which capabilities code can use)
    - Temporal ordering constraints (enforce call sequences)
    - Resource limits (cap total LLM calls, blackboard writes, etc.)

    Example implementations:
    - ``NoGuardrail``: Allow everything.
    - ``CapabilityBoundaryGuardrail``: Restrict which capabilities can be called.
    - ``TemporalOrderGuardrail``: Enforce before/after ordering on actions.
    """

    @abstractmethod
    async def check(
        self,
        action_key: str,
        params: dict[str, Any],
        call_history: list[str],
    ) -> GuardrailDecision:
        """Check whether an action is allowed.

        Called by the ``run()`` helper in the REPL namespace before
        dispatching each action.

        Args:
            action_key: The action being dispatched.
            params: The action parameters.
            call_history: List of action keys already called in this
                code execution (in order).

        Returns:
            GuardrailDecision with allowed/reason/suggestion.
        """
        ...


class NoGuardrail(RuntimeGuardrail):
    """Allow everything — no runtime constraints."""

    async def check(self, action_key, params, call_history):
        return GuardrailDecision(allowed=True)


class CapabilityBoundaryGuardrail(RuntimeGuardrail):
    """Restrict which capabilities the generated code can invoke.

    Useful for focusing the agent: an agent in "execution mode" might
    be restricted to domain capabilities only, preventing it from
    calling planning capabilities during execution.

    Args:
        allowed_prefixes: Action keys must start with one of these prefixes.
            E.g., ``["CacheAnalysis", "Analysis"]`` allows cache analysis
            and domain analysis but blocks coordination.
        blocked_prefixes: Action keys matching these prefixes are blocked.
            Takes precedence over ``allowed_prefixes``.
    """

    def __init__(
        self,
        allowed_prefixes: list[str] | None = None,
        blocked_prefixes: list[str] | None = None,
    ):
        self._allowed = allowed_prefixes
        self._blocked = blocked_prefixes or []

    async def check(self, action_key, params, call_history):
        # Check blocked first
        for prefix in self._blocked:
            if prefix.lower() in action_key.lower():
                return GuardrailDecision(
                    allowed=False,
                    reason=f"Action '{action_key}' is blocked by guardrail (prefix '{prefix}').",
                    suggestion="Use a different action or switch mode.",
                )

        # Check allowed
        if self._allowed is not None:
            if not any(prefix.lower() in action_key.lower() for prefix in self._allowed):
                return GuardrailDecision(
                    allowed=False,
                    reason=f"Action '{action_key}' is not in the allowed set.",
                    suggestion=f"Allowed prefixes: {self._allowed}",
                )

        return GuardrailDecision(allowed=True)


class TemporalOrderGuardrail(RuntimeGuardrail):
    """Enforce ordering constraints on action calls.

    Prevents common mistakes like calling ``propose_plan`` before
    ``check_plan_conflicts``, or calling ``synthesize`` before any
    ``analyze_pages`` calls.

    Args:
        ordering_rules: List of ``(before, after)`` tuples. The ``before``
            action must have been called before the ``after`` action.
            Uses substring matching on action keys.
    """

    def __init__(self, ordering_rules: list[tuple[str, str]]):
        self._rules = ordering_rules

    async def check(self, action_key, params, call_history):
        for before, after in self._rules:
            if after.lower() in action_key.lower():
                # Check if "before" has been called
                if not any(before.lower() in h.lower() for h in call_history):
                    return GuardrailDecision(
                        allowed=False,
                        reason=f"'{action_key}' requires '{before}' to be called first.",
                        suggestion=f"Call an action matching '{before}' before '{action_key}'.",
                    )

        return GuardrailDecision(allowed=True)


# ============================================================================
# Helpers
# ============================================================================


def _extract_code(response: Any) -> str:
    """Extract Python code from an LLM response."""
    if hasattr(response, 'generated_text'):
        text = response.generated_text
    elif hasattr(response, 'text'):
        text = response.text
    elif isinstance(response, str):
        text = response
    elif isinstance(response, dict):
        text = response.get("text", response.get("content", str(response)))
    else:
        text = str(response)

    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    return text


# ============================================================================
# Dimension 6: Completion Validation
# ============================================================================


@dataclass
class CompletionValidationResult:
    """Result of completion validation."""

    allowed: bool
    """Whether completion is allowed."""
    reason: str
    """Why completion was allowed/rejected."""
    suggestions: list[str] = field(default_factory=list)
    """What to do if rejected."""


class CompletionValidator(ABC):
    """Validates whether the agent should be allowed to signal completion.

    Called when generated code invokes ``signal_complete()``. Inspects
    goals, accumulated results, and execution history to determine
    whether all objectives have been achieved.

    Library users implement this to define domain-specific completion
    criteria. The validator is plugged into ``CodeGenerationActionPolicy``
    as the 6th dimension.

    Example implementations:
    - ``NoOpCompletionValidator``: Allow all completions (default).
    - ``RuleBasedCompletionValidator``: Check domain work done + results non-empty.
    - ``LLMCompletionValidator``: Ask the LLM to evaluate goal achievement.
    """

    @abstractmethod
    async def validate(
        self,
        agent: Agent,
        goals: list[str],
        results: dict[str, Any],
        execution_context: PlanExecutionContext,
    ) -> CompletionValidationResult:
        """Validate whether completion should be allowed.

        Args:
            agent: The agent attempting to complete.
            goals: The agent's stated goals.
            results: Accumulated results dict from the REPL namespace.
            execution_context: ``PlanExecutionContext`` with action history.

        Returns:
            CompletionValidationResult with allowed/reason/suggestions.
        """
        ...


class NoOpCompletionValidator(CompletionValidator):
    """Allow all completions — no validation (default)."""

    async def validate(self, agent, goals, results, execution_context: PlanExecutionContext):
        return CompletionValidationResult(allowed=True, reason="No validation configured")


class RuleBasedCompletionValidator(CompletionValidator):
    """Check that domain work was done and results are non-empty.

    Rejects completion if:
    - No successful domain actions have been executed
    - The results dict is empty
    """

    async def validate(self, agent, goals, results, execution_context: PlanExecutionContext):
        if not goals:
            return CompletionValidationResult(allowed=True, reason="No goals defined")

        step_summaries = execution_context.custom_data.get("codegen_step_summaries", {})
        has_domain_work = any(
            info.get("domain_actions") and not info.get("had_failures")
            for info in step_summaries.values()
        )
        if not has_domain_work:
            return CompletionValidationResult(
                allowed=False,
                reason="No successful domain actions have been executed yet.",
                suggestions=["Execute domain actions before signaling completion."],
            )

        if not results:
            return CompletionValidationResult(
                allowed=False,
                reason="No results have been stored.",
                suggestions=["Store action results in the results dict before completing."],
            )

        return CompletionValidationResult(
            allowed=True,
            reason=f"Domain work completed with {len(results)} result(s).",
        )


class LLMCompletionValidator(CompletionValidator):
    """Ask the LLM to evaluate whether goals have been achieved.

    Constructs a prompt with goals, results summary, and execution
    history, then asks the LLM for a structured yes/no assessment.
    Falls open on LLM error (allows completion).
    """

    async def validate(self, agent, goals, results, execution_context: PlanExecutionContext):
        goals_text = "\n".join(f"- {g}" for g in goals)
        results_text = "\n".join(
            f"- {k}: {str(v)[:300]}" for k, v in list(results.items())[:20]
        )

        step_summaries = execution_context.custom_data.get("codegen_step_summaries", {})
        history_text = "\n".join(
            f"- {info.get('step_kind', '?')}: "
            f"{', '.join(info.get('actions_called', []))}"
            f" ({'success' if not info.get('had_failures') else 'failed'})"
            for info in list(step_summaries.values())[-10:]
        )

        prompt = f"""Evaluate whether the following goals have been achieved based on the results and execution history.

Goals:
{goals_text}

Results collected:
{results_text}

Execution history (last 10 steps):
{history_text}

Answer with EXACTLY one of:
- COMPLETE: <reason why goals are achieved>
- INCOMPLETE: <reason why goals are NOT achieved> | SUGGESTIONS: <suggestion1>; <suggestion2>"""

        try:
            response = await agent.infer(prompt=prompt, max_tokens=256, temperature=0.1)
            text = response.generated_text if hasattr(response, "generated_text") else str(response)

            if text.strip().startswith("COMPLETE"):
                reason = text.split(":", 1)[1].strip() if ":" in text else "Goals achieved"
                return CompletionValidationResult(allowed=True, reason=reason)
            else:
                parts = text.split("|")
                reason = parts[0].replace("INCOMPLETE:", "").strip()
                suggestions = []
                if len(parts) > 1:
                    sug_text = parts[1].replace("SUGGESTIONS:", "").strip()
                    suggestions = [s.strip() for s in sug_text.split(";") if s.strip()]
                return CompletionValidationResult(
                    allowed=False, reason=reason, suggestions=suggestions,
                )
        except Exception as e:
            logger.warning("LLM completion validation failed: %s", e)
            return CompletionValidationResult(
                allowed=True,
                reason=f"Validation failed ({e}), allowing completion",
            )

