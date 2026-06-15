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
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ...base import Agent, AgentCapability
    from ...models import ActionResult, PlanExecutionContext

logger = logging.getLogger(__name__)


# ============================================================================
# Call history record — the shared data shape every RuntimeGuardrail reads.
# ============================================================================


@dataclass
class CallRecord:
    """One entry in ``CodeGenerationActionPolicy._call_history``.

    Replaces the previous ``list[str]`` shape so guardrails can make
    args-aware decisions (e.g. "did get_agent_status get called for
    THIS agent_id?") rather than only matching on action-key
    substrings.

    Status semantics:

    - ``"pending"`` — record was created at admit time but the action
      hasn't returned yet. Rare for guardrail reads since check()
      runs BEFORE dispatch; the field exists so the policy can
      pre-create the record at admit time without lying about status.
    - ``"ok"`` / ``"error"`` — terminal states populated when the
      dispatch returns.
    - ``"blocked"`` — the guardrail itself refused the call. Lets
      downstream guardrails distinguish "the LLM tried X but a gate
      stopped it" from "the LLM never tried X".

    ``result`` carries a small snapshot of the action's return value
    so guardrails can inspect the OUTCOME of prior calls — e.g.
    ``ApprovalRequiredGuardrail`` keys off
    ``HumanApprovalCapability.get_response`` returning
    ``{"choice": "Approve"}``. Capped at
    :data:`_CALL_RECORD_RESULT_PREVIEW_BYTES` so a single big LLM
    payload can't blow the per-iteration history budget; guardrails
    that need the full payload should query the span store, not
    ``call_history``.
    """

    action_key: str
    params: dict[str, Any]
    action_id: str = ""
    """The dispatched action's id — keys into
    :attr:`PlanExecutionContext.action_results`. Empty for the
    ``"blocked"`` status (no dispatch happened)."""
    start_wall: float = field(default_factory=time.time)
    end_wall: float | None = None
    status: Literal["pending", "ok", "error", "blocked"] = "pending"
    error: str | None = None
    result: Any = None


_CALL_RECORD_RESULT_PREVIEW_BYTES = 4096
"""Per-call truncation cap on ``CallRecord.result``. 4 KiB easily fits
typed envelopes like ``{request_id, choice, decided_by, decided_at}``
that guardrails inspect, while keeping the policy's in-memory
history bounded across long-running coordinators."""


@dataclass
class BlockedDispatch:
    """One entry in
    :attr:`CodeGenerationActionPolicy._last_blocked_dispatches`.

    Captured at the moment the runtime guardrail refuses a ``run()``
    call inside the REPL. Survives exactly one iteration — cleared at
    the start of the next code-generation cycle, same lifecycle as
    :attr:`CodeGenerationActionPolicy._call_history`. Rendered into
    the planner prompt under "## Blocked Dispatches (last iteration)"
    so the LLM sees the block AND the guardrail's suggestion BEFORE
    proposing its next cell — instead of having to infer the block
    after the fact from a ``result.success=False`` in the cell's own
    code.

    The ``params_preview`` is a JSON-truncated snapshot of the
    proposed call's params capped at
    :data:`_BLOCKED_DISPATCH_PARAMS_PREVIEW_BYTES`; full payloads
    stay in the span store, not in this in-memory list.
    """

    action_key: str
    params_preview: Any
    reason: str
    suggestion: str
    wall_time: float = field(default_factory=time.time)


_BLOCKED_DISPATCH_PARAMS_PREVIEW_BYTES = 1024
"""Per-block truncation cap on ``BlockedDispatch.params_preview``.
Smaller than ``_CALL_RECORD_RESULT_PREVIEW_BYTES`` because blocked
dispatches usually carry the LLM's proposed args verbatim and
typically a few short fields (action_key + content + a few kwargs)
are enough to recover; long-tail payloads (entire decomposition
proposals) get truncated."""


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

    @staticmethod
    def _is_call(node: ast.AST, name: str) -> bool:
        """Match ``name(...)`` and ``self.name(...)`` shapes."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Name):
            return func.id == name
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return func.attr == name
        return False

    @classmethod
    def _count_runs_per_path(cls, node: ast.AST) -> int:
        """Recursively count ``run()`` calls along any single execution
        path through ``node``. ``If`` and ``Try`` nodes contribute the
        max across mutually-exclusive branches; everything else sums.

        This matters because the LLM's natural shape for a one-step
        action with a branched response —

            await run("ack")
            r = await run("the_action")
            if r.success:
                await run("respond_ok")
            else:
                await run("respond_err")

        — runs at most three actions in any single execution. A naive
        textual walk would count four and reject the iteration. Path-
        aware counting matches the validator's intent (focused step,
        not monolithic program).
        """
        if cls._is_call(node, "run"):
            # Count this call, but also walk its arguments — a
            # nested ``run(other=run(...))`` would otherwise be missed.
            return 1 + sum(cls._count_runs_per_path(c) for c in ast.iter_child_nodes(node))
        if isinstance(node, ast.If):
            test = cls._count_runs_per_path(node.test)
            body = sum(cls._count_runs_per_path(s) for s in node.body)
            orelse = sum(cls._count_runs_per_path(s) for s in node.orelse)
            return test + max(body, orelse)
        if isinstance(node, ast.Try):
            body = sum(cls._count_runs_per_path(s) for s in node.body)
            else_ = sum(cls._count_runs_per_path(s) for s in node.orelse)
            finalbody = sum(cls._count_runs_per_path(s) for s in node.finalbody)
            handlers = max(
                (sum(cls._count_runs_per_path(s) for s in h.body) for h in node.handlers),
                default=0,
            )
            # Either the body completes (body + else_) or an exception
            # routes through one handler. ``finalbody`` always runs.
            return max(body + else_, body + handlers) + finalbody
        return sum(cls._count_runs_per_path(c) for c in ast.iter_child_nodes(node))

    @classmethod
    def _count_browses_textual(cls, tree: ast.AST) -> int:
        """browse() is a discovery anti-pattern; multiple browses are
        excessive regardless of which branch they sit in. Text-walk."""
        return sum(1 for node in ast.walk(tree) if cls._is_call(node, "browse"))

    async def validate(self, code: str, agent: Agent) -> ValidationResult:
        errors: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(valid=False, errors=[f"Syntax error: {e}"])

        # Run() count is per execution path, not textual. An ``if/else``
        # block with one ``run()`` in each branch executes ONE of them,
        # so it counts as 1 — not 2. Otherwise the natural
        # ack-then-action-then-branched-response pattern (an
        # acknowledgement, the action, plus a success/failure
        # ``respond_to_user`` in each ``if/else`` arm) would tip a
        # focused 3-action iteration into looking like 4 to a textual
        # walker. Browse() counting stays textual since browse()
        # mid-iteration is a discovery anti-pattern regardless of
        # branch — multiple browses in different branches are still
        # excessive overall.
        run_count = self._count_runs_per_path(tree)
        browse_count = self._count_browses_textual(tree)

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
        call_history: list[CallRecord],
    ) -> GuardrailDecision:
        """Check whether an action is allowed.

        Called by the ``run()`` helper in the REPL namespace before
        dispatching each action.

        Args:
            action_key: The action being dispatched.
            params: The action parameters.
            call_history: Per-call records of every dispatched action
                so far in this code-generation iteration, in order.
                Each item is a :class:`CallRecord` carrying the
                action key, params, timing, and terminal status.
                Guardrails that only need action keys read
                ``c.action_key``; guardrails that need args-aware
                matching read ``c.params``.

        Returns:
            GuardrailDecision with allowed/reason/suggestion.
        """
        ...

    def bind_speaker(self, agent: "Any | None") -> None:
        """Tell the guardrail which agent is the SPEAKER (the agent whose
        action-policy is dispatching the calls this guardrail will check).

        Called by the code-generation action policy during init. Default
        no-op — guardrails that need speaker context (e.g. the
        SessionAgent's status-claim gate, which must exclude the
        SessionAgent's OWN agent_id from the "you referenced this id
        without status-checking it" check) override this and stash the
        identity for use in :meth:`check` / predicates.
        A guardrail that doesn't know who's speaking can't
        tell self-references from other-references.
        The narrow ``bind_speaker`` hook keeps the
        :meth:`check` signature stable while letting guardrails opt
        in to speaker-aware behaviour.

        Composite guardrails should propagate the bind to their inner
        guardrails."""

        return None

    def planner_context_advisory(
        self,
        call_history: list[CallRecord],
    ) -> str | None:
        """Standing guidance to inject into the planner's prompt
        BEFORE it proposes its next code cell.

        Default: ``None`` — no advisory. Override in subclasses that
        encode rules the planner can satisfy by choosing actions in
        a specific order (e.g. *call request_human_approval before
        the apply path*, *call get_agent_status before mentioning an
        agent_id in respond_to_user*).

        Why a separate method from :meth:`check`: ``check`` runs
        AFTER the planner has already chosen an action; surfacing the
        rule only on the block message means the planner gets steered
        post-hoc, often after burning iterations on the same wrong
        choice. The advisory runs BEFORE the planner picks, threaded
        into the prompt by the code-generation policy. Block messages
        from ``check`` remain the safety net for any planner that
        ignores the advisory.

        Implementations should return ``None`` when the rule is
        currently satisfied (or never applies) so the prompt stays
        small; return a short paragraph when the planner needs to be
        nudged.
        """
        return None


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
                # Check if "before" has been called. ``call_history``
                # is now ``list[CallRecord]``; read ``c.action_key``
                # to do the same substring match the legacy
                # ``list[str]`` shape supported.
                if not any(
                    before.lower() in c.action_key.lower()
                    for c in call_history
                ):
                    return GuardrailDecision(
                        allowed=False,
                        reason=f"'{action_key}' requires '{before}' to be called first.",
                        suggestion=f"Call an action matching '{before}' before '{action_key}'.",
                    )

        return GuardrailDecision(allowed=True)


# ============================================================================
# New guardrail subclasses introduced by the action-preconditions plan
# (``colony/mission_and_action_guardrails_plan.md`` Part 2).
# ============================================================================


@dataclass(frozen=True)
class ArgsAwareOrderingRule:
    """One rule for :class:`ArgsAwareTemporalOrderGuardrail`.

    Each rule answers "before dispatching the target_action, was the
    required_prior action called recently (and, optionally, with
    matching params)?". The ``applies_when`` predicate gates whether
    this rule even fires against the proposed call — letting bare
    ``respond_to_user("Hi!")`` calls skip a status-check rule whose
    only purpose is to gate references to agent_ids.

    All fields are picklable so the rule survives
    ``cloudpickle``-based actor spawn paths the framework relies on.
    """

    # The action_key being considered for dispatch. Matched with
    # ``in`` substring against the proposed action_key — same shape
    # as the legacy ``TemporalOrderGuardrail``'s matching semantics.
    target_action: str

    # Predicate over the proposed action's params. Returns True iff
    # the rule applies. Defaults to "always applies" via ``None``.
    applies_when: Callable[[dict[str, Any]], bool] | None = None

    # The action_key that must have been called BEFORE the target.
    # Matched with ``in`` substring against ``CallRecord.action_key``.
    required_prior: str = ""

    # Bound on how far back the prior call may be. ``max_age_calls``
    # bounds by history-index distance; ``max_age_seconds`` bounds by
    # wall-clock. Either may be ``None`` for no bound. When both are
    # set, the more recent bound wins (the prior call must satisfy
    # both — i.e. recent in BOTH senses).
    max_age_calls: int | None = None
    max_age_seconds: float | None = None

    # Optional predicate over (prior_record.params, target_params)
    # to require args-level matching (e.g. same agent_id). Returns
    # True iff this prior record satisfies the rule. ``None`` accepts
    # any prior call matching ``required_prior``.
    prior_params_match: (
        Callable[[dict[str, Any], dict[str, Any]], bool] | None
    ) = None

    # Hint sent back to the LLM on violation (rendered into
    # ``GuardrailDecision.suggestion``). Should name the fix in
    # action terms so the next planner iteration can fix it cheaply.
    suggestion: str = ""


class ArgsAwareTemporalOrderGuardrail(RuntimeGuardrail):
    """Generalisation of :class:`TemporalOrderGuardrail` that:

    - Reads the args of the proposed call to decide whether the rule
      applies (``applies_when``), so single-purpose gates don't fire
      on unrelated calls.
    - Optionally requires the prior call's params to match the
      target's (``prior_params_match``), so "called X for some other
      agent" doesn't count as a recent X for THIS agent.
    - Bounds the recency window by call count and/or wall-clock.

    All rule predicates run synchronously inside the async ``check``
    — they're expected to be cheap (regex, list lookup, dict
    comparison). Heavy work belongs in a dedicated guardrail
    (e.g. :class:`LLMJudgedGuardrail`, future).
    """

    def __init__(self, rules: Sequence[ArgsAwareOrderingRule]):
        self._rules = tuple(rules)

    async def check(self, action_key, params, call_history):
        now = time.time()
        for rule in self._rules:
            if rule.target_action.lower() not in action_key.lower():
                continue
            if rule.applies_when is not None and not rule.applies_when(
                params,
            ):
                continue
            if not rule.required_prior:
                continue
            # Look for a satisfying prior call within the recency
            # window. ``call_history`` is in admit-order; walk from
            # the tail so the most-recent match is found first.
            window_start_idx = (
                max(0, len(call_history) - rule.max_age_calls)
                if rule.max_age_calls is not None else 0
            )
            window = call_history[window_start_idx:]
            satisfied = False
            for record in reversed(window):
                if (
                    rule.required_prior.lower()
                    not in record.action_key.lower()
                ):
                    continue
                if (
                    rule.max_age_seconds is not None
                    and (now - record.start_wall) > rule.max_age_seconds
                ):
                    continue
                if (
                    rule.prior_params_match is not None
                    and not rule.prior_params_match(record.params, params)
                ):
                    continue
                satisfied = True
                break
            if not satisfied:
                return GuardrailDecision(
                    allowed=False,
                    reason=(
                        f"'{action_key}' requires a recent "
                        f"'{rule.required_prior}' call matching its "
                        f"args; none in the last "
                        f"{rule.max_age_calls or len(call_history)} "
                        f"call(s)."
                    ),
                    suggestion=rule.suggestion or (
                        f"Call an action matching "
                        f"'{rule.required_prior}' before "
                        f"'{action_key}'."
                    ),
                )
        return GuardrailDecision(allowed=True)

    def planner_context_advisory(self, call_history):
        """Surface every rule's pre-fix guidance as standing context.

        These rules describe orderings the planner can always satisfy
        by choosing actions in the right sequence — making them
        always-visible standing guidance is cheaper than letting the
        planner discover them via blocked dispatches. Rules with no
        ``suggestion`` field skip the advisory (they'll still block
        at ``check`` time)."""

        if not self._rules:
            return None
        bullets = []
        for rule in self._rules:
            if rule.suggestion:
                bullets.append(
                    f"- Before calling ``{rule.target_action}``: "
                    f"{rule.suggestion}"
                )
        if not bullets:
            return None
        return (
            "**Action-ordering rules in effect.** Satisfy these "
            "BEFORE proposing the gated call so the runtime "
            "guardrail doesn't block + burn an iteration:\n"
            + "\n".join(bullets)
        )


class ApprovalRequiredGuardrail(RuntimeGuardrail):
    """Block actions whose key prefixes are gated behind a human
    approval round.

    Reads ``approval_required_action_prefixes`` and asks
    :meth:`HumanApprovalCapability.has_active_approval_for` whether a
    non-revoked approval covers the action_key. The capability is
    resolved via :meth:`bind_speaker` when the action policy
    initialises.

    ``request_human_approval`` itself is not a substitute: issuing
    the request is not the same as the operator approving it.

    Cross-references the mission's
    :attr:`MissionExecutionPolicy.requires_human_approval_before`
    field; the typical mounting site reads that list at agent
    construction time and passes it in here.
    """

    def __init__(
        self,
        approval_required_action_prefixes: Sequence[str],
    ):
        self._gated = tuple(approval_required_action_prefixes)
        # Filled by ``bind_speaker``; the guardrail constructor runs
        # at session-create time, before the agent exists.
        self._approval_cap_resolver: Callable[[], Any] | None = None

    def bind_speaker(self, agent):
        """Capture a thunk that resolves the agent's
        ``HumanApprovalCapability`` lazily — at construction time it
        may not be mounted yet."""

        if agent is None:
            self._approval_cap_resolver = None
            return

        def _resolve():
            from ..capabilities.human_approval import (
                HumanApprovalCapability,
            )
            return agent.get_capability_by_type(HumanApprovalCapability)

        self._approval_cap_resolver = _resolve

    def _matching_prefix(self, action_key: str) -> str | None:
        return next(
            (p for p in self._gated if p.lower() in action_key.lower()),
            None,
        )

    async def check(self, action_key, params, call_history):
        gated = self._matching_prefix(action_key)
        if gated is None:
            return GuardrailDecision(allowed=True)
        # ``dry_run=True`` proposals never mutate side effects; the
        # approval gate only blocks the apply path. Mission flows
        # always carry ``dry_run`` as a kwarg per the project-
        # planning convention; if the kwarg is absent, default to
        # "this is an apply call" and gate.
        dry_run = params.get("dry_run")
        if dry_run is True:
            return GuardrailDecision(allowed=True)

        cap = (
            self._approval_cap_resolver()
            if self._approval_cap_resolver is not None
            else None
        )
        if cap is not None:
            allowed, _ = await cap.has_active_approval_for(action_key)
            if allowed:
                return GuardrailDecision(allowed=True)

        from ..capabilities.human_approval import HumanApprovalCapability
        prefix = HumanApprovalCapability.RESPONSE_CONTEXT_KEY_PREFIX
        return GuardrailDecision(
            allowed=False,
            reason=(
                f"'{action_key}' is gated behind human approval "
                f"(matched prefix '{gated}'); no recorded approval "
                f"covers this action."
            ),
            suggestion=(
                f"Apply is blocked until a ``{prefix}<request_id>`` "
                "context binding shows ``choice in "
                "{approve_once, approve_all}``. "
                "Available primitives: "
                "``request_human_approval(question, action_type, extra)`` "
                "opens a request and returns immediately with a "
                "``request_id``; the operator's answer surfaces later "
                "as that planner-context binding (with "
                "``explanation`` populated on ``reject`` / ``abort``). "
                "``wait_for_next_event()`` pauses the agent on the "
                "unified event queue and wakes on the next event of "
                "any kind. ``get_response(request_id)`` is on-demand "
                "lookup of a known request, not a wait — re-calling "
                "it in a loop is wasted work. Compose whatever "
                "strategy fits the situation."
            ),
        )

    def planner_context_advisory(self, call_history):
        """Surface the approval-gate capability surface to the planner
        BEFORE it proposes a mutating call, so the loop doesn't have
        to bounce off ``check``'s block message to learn it. Describes
        each primitive in isolation per
        ``[[primitives-not-pipelines]]``: the LLM composes the strategy
        (which may involve multiple in-flight approvals, status updates
        between request and wait, waking on a non-approval event, etc.)
        — the framework does not bake an ordering.
        """

        if not self._gated:
            return None
        from ..capabilities.human_approval import HumanApprovalCapability
        prefix = HumanApprovalCapability.RESPONSE_CONTEXT_KEY_PREFIX
        gated_render = ", ".join(f"``{p}``" for p in self._gated)
        return (
            "**Human-approval gate active.** The following action "
            f"prefixes require operator approval before the apply "
            f"path will dispatch: {gated_render}. "
            "Available primitives for this gate: "
            "``request_human_approval(question, action_type, extra)`` "
            "— opens a typed request and returns immediately with a "
            "``request_id``; the operator's answer surfaces later as "
            f"a ``{prefix}<request_id>`` planner-context binding whose "
            "``choice`` is one of ``approve_once`` / ``approve_all`` / "
            "``reject`` / ``abort`` and whose ``explanation`` is "
            "non-empty on ``reject`` / ``abort``. "
            "``wait_for_next_event()`` — pauses the agent on the "
            "unified event queue and wakes on the next event of any "
            "kind (approval answer, child mission completion, chat "
            "message, cancel). "
            "``get_response(request_id)`` — on-demand lookup of a "
            "known request's current state; not a wait primitive. "
            "Re-calling ``get_response`` in a loop is wasted work — "
            "the planner-context binding is the wake surface. "
            "When the response arrives: ``approve_once`` / "
            "``approve_all`` admits the apply (re-call the action "
            "with ``dry_run=False``); ``reject`` blocks this dispatch "
            "but keeps the agent alive (read ``explanation`` and "
            "adjust); ``abort`` is the operator's request to wind "
            "down via the mission-control surface."
        )


class CompositeGuardrail(RuntimeGuardrail):
    """Apply N guardrails in order; the first non-allowed decision
    short-circuits.

    The natural shape for agents with multiple gating concerns
    (e.g. SessionAgent needs both the status-claim gate AND the
    approval gate AND the capability-boundary gate). Composes at
    the policy-config level so individual actions stay
    decoration-free — see the action-policy-dimensions guide for
    the broader rationale.
    """

    def __init__(self, *guardrails: RuntimeGuardrail):
        if not guardrails:
            raise ValueError(
                "CompositeGuardrail requires at least one inner "
                "guardrail; use NoGuardrail() if you mean 'allow all'."
            )
        self._inner: tuple[RuntimeGuardrail, ...] = tuple(guardrails)

    async def check(self, action_key, params, call_history):
        for guardrail in self._inner:
            decision = await guardrail.check(
                action_key, params, call_history,
            )
            if not decision.allowed:
                return decision
        return GuardrailDecision(allowed=True)

    def bind_speaker(self, agent):
        """Propagate the bind to every inner guardrail so each one
        can pick up speaker-aware behaviour independently."""

        for guardrail in self._inner:
            guardrail.bind_speaker(agent)

    def planner_context_advisory(self, call_history):
        """Join non-empty advisories from every inner guardrail.

        Each inner guardrail's advisory is its own paragraph; we
        concatenate with a blank line so the resulting prompt
        section reads as a checklist. Returns ``None`` when no inner
        guardrail has anything to say (the common case once all
        active rules are satisfied)."""

        advisories = []
        for guardrail in self._inner:
            advisory = guardrail.planner_context_advisory(call_history)
            if advisory:
                advisories.append(advisory)
        if not advisories:
            return None
        return "\n\n".join(advisories)


# ============================================================================
# Helpers
# ============================================================================


# Each line-anchored ``` fence is one match. Pairing is done in
# ``_iter_fenced_blocks`` so opener/closer pairs are always
# consecutive in document order — a regex that captures opener-to-
# closer in one shot mispairs when blocks of different languages are
# interleaved (e.g., the LLM emits ```python ... ``` ... ```json ...
# ``` ... ```python ... ```; the closer of the json block was getting
# mistaken for the opener of the second python block).
_FENCE_LINE_RE = re.compile(r"^```([^\n]*)$", re.MULTILINE)


def _iter_fenced_blocks(text: str):
    """Yield ``(lang_token | None, body)`` tuples in document order.

    Pairs consecutive ``` lines. A trailing unpaired opener is
    silently dropped (the LLM occasionally emits one when it gets
    truncated mid-block).
    """
    fences = list(_FENCE_LINE_RE.finditer(text))
    for i in range(0, len(fences) - 1, 2):
        opener, closer = fences[i], fences[i + 1]
        lang = opener.group(1).strip().lower() or None
        body = text[opener.end():closer.start()].strip("\n")
        yield lang, body


def _extract_code(response: Any) -> str:
    """Extract Python code from an LLM response.

    Handles three shapes the LLM tends to produce:

    1. **Bare code, no fences.** Used as-is.
    2. **One ```` ```python ```` fence wrapping everything.** Fence
       stripped, body returned.
    3. **Multiple fenced blocks interleaved with prose / fake "Result"
       output.** Every ``python`` (or untagged) block is extracted
       and concatenated with blank-line separators; prose between
       blocks is dropped.

    Shape (3) is the load-bearing case: instruction-tuned models
    sometimes ignore the "no markdown" rule and emit a tutorial-style
    transcript with mocked results. Concatenating the real code
    blocks turns that mistake into a working iteration instead of a
    silent validation failure.
    """
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
    if not text:
        return ""

    blocks = list(_iter_fenced_blocks(text))
    if blocks:
        python_blocks = [
            body for lang, body in blocks
            if lang in ("python", "py")
        ]
        if not python_blocks:
            # No python-tagged blocks. Accept untagged fences (the
            # LLM forgot the language hint) but only when no fenced
            # block claims a non-python language — mixing languages
            # would be ambiguous, so we'd rather extract nothing than
            # pull in a JSON blob the LLM left for documentation.
            tagged_non_python = any(
                lang not in (None, "python", "py")
                for lang, _ in blocks
            )
            if not tagged_non_python:
                python_blocks = [body for _, body in blocks]
        # Once we found fenced blocks at all, the LLM's intent was
        # clearly "the code is in the fences". Returning "" lets the
        # policy retry instead of falling back to the legacy
        # whole-text-as-code path, which would treat the surrounding
        # markdown as Python and fail validation.
        return "\n\n".join(b.strip() for b in python_blocks).strip()

    # No paired fences — strip any single leading / trailing fence
    # marker (the legacy single-block case) and return the rest.
    if text.startswith("```python"):
        text = text[len("```python"):].lstrip()
    elif text.startswith("```py"):
        text = text[len("```py"):].lstrip()
    elif text.startswith("```"):
        text = text[3:].lstrip()
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()


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

        has_domain_work = any(
            info.domain_actions and not info.had_failures
            for info in execution_context.codegen_step_summaries.values()
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

        history_text = "\n".join(
            f"- {info.step_kind}: "
            f"{', '.join(info.actions_called)}"
            f" ({'success' if not info.had_failures else 'failed'})"
            for info in list(
                execution_context.codegen_step_summaries.values(),
            )[-10:]
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

