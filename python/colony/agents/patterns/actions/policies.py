"""Iterative Reasoning Loop for Agents.

This module implements the planning → action → reflection → critique cycle
that allows agents to iteratively refine their approach instead of using
finite state machines with single-shot inference.

Philosophy:
- Agents should reason iteratively, not execute predetermined sequences
- Each iteration: PLAN → ACT → REFLECT → CRITIQUE → ADAPT
- LLM-driven: Control flow determined by reasoning, not hardcoded
- Composable: Planner, Dispatcher, Critic are independent policies
- Observable: Emit events at each stage for debugging/monitoring

This replaces FSM-based agent design with true iterative reasoning.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, get_type_hints, get_origin, get_args
from contextlib import AsyncExitStack
from overrides import override
from pydantic import BaseModel, ConfigDict, Field, PydanticSchemaGenerationError, ValidationError, create_model

from ....utils import setup_logger
from ...models import (
    Action,
    ActionType,
    ActionResult,
    ActionGroupDescription,
    ActionPlan,
    ActionStatus,
    PlanningContext,
    PlanningParameters,
    PlanExecutionContext,
    AgentSuspensionState,
    ActionPolicyExecutionState,
    ActionPolicyIO,
    PolicyREPL,
    Ref,
    PlanStatus,
    ActionSharedDataDependency,
)
from ...base import Agent, ActionPolicy, ActionPolicyIterationResult, AgentCapability
from ...blackboard import BlackboardEvent
from ..hooks import hookable
from ...blackboard.backend import ConcurrentModificationError
from .repl import PolicyPythonREPL, REPLCapability, get_repl_guidance
from ..planning import (
    ActionPlanner,
    PlanBlackboard,
    PlanningStrategyPolicy,
    CacheAwarePlanningPolicy,
    LearningPlanningPolicy,
    CoordinationPlanningPolicy,
    HierarchicalAccessPolicy,
    CacheAwareActionPlanner,
    get_default_planning_strategy,
    ReplanningPolicy,
    ReplanningDecision,
    PeriodicReplanningPolicy,
)
# NOTE: Class-based ActionExecutors from executors.py are deprecated.
# Use @action_executor decorated methods on AgentCapability classes instead.
# See executors.py module docstring for migration details.


logger = setup_logger(__name__)

#
# NOTE: With ambient transactions, action executors do not need access to transaction
# handles. The dispatcher only needs to validate dependency versions and execute the
# action within the dependency transactor contexts.





class ActionExecutor(ABC):
    """Policy for executing actions.

    Different implementations based on agent type:
    - PageAnalyzerExecutor: Execute page analysis actions
    - ClusterAnalyzerExecutor: Execute cluster analysis actions
    - CoordinatorExecutor: Execute coordination actions
    """

    def __init__(self, agent: Agent):
        self.agent = agent

    @abstractmethod
    async def execute(self, action: Action) -> ActionResult:
        """Execute an action.

        Args:
            action: Action to execute

        Returns:
            Result of execution
        """
        ...

    async def get_action_description(self) -> str:
        """Get human-readable description of action or action policy to be
        used in LLM-based action selection.

        Returns:
            Description string
        """
        raise NotImplementedError(
            "get_action_description not implemented for this executor."
            " Get the docstring of the execute() method instead."
        )


class MethodWrapperActionExecutor(ActionExecutor):
    """Wraps a method decorated with @action_executor as an ActionExecutor.

    Handles:
    - Automatic input schema inference from type hints
    - Automatic output schema inference from return type
    - Parameter validation using inferred schemas
    - Output validation using inferred schemas
    """

    def __init__(
        self,
        object: Any,
        method: Callable,
        action_key: str | ActionType,
        input_schema: type[BaseModel] | None = None,
        output_schema: type[BaseModel] | None = None,
        reads: list[str] | None = None,
        writes: list[str] | None = None,
        exclude_from_planning: bool = False,
        planning_summary: str | None = None,
    ):
        self.object = object
        self.method = method
        self.action_key = str(action_key)
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.reads = reads or []
        self.writes = writes or []
        self.exclude_from_planning = exclude_from_planning
        self.planning_summary = planning_summary

    async def execute(self, action: Action, resolved_params: dict[str, Any] | None = None) -> ActionResult:
        """Execute the wrapped method.

        Args:
            action: The action to execute
            resolved_params: Pre-resolved parameters (with Refs resolved).
                If None, uses action.parameters directly.

        Returns:
            ActionResult with method return value as output
        """
        try:
            # Use resolved params if provided, otherwise use action.parameters
            params = resolved_params if resolved_params is not None else action.parameters

            # Validate against input schema if available
            if self.input_schema:
                try:
                    validated = self.input_schema(**params)
                    params = validated.model_dump()
                except ValidationError as e:
                    return ActionResult(
                        success=False,
                        completed=True,
                        error=f"Parameter validation failed: {e}"
                    )

            # Call the method with parameters as kwargs
            ret = await self.method(self.object, **params)

            # Validate output if schema available
            if self.output_schema and ret is not None:
                try:
                    # If return is not a dict, wrap it
                    if not isinstance(ret, dict):
                        validated = self.output_schema(value=ret)
                    else:
                        validated = self.output_schema(**ret)
                    ret = validated.model_dump()
                except ValidationError as e:
                    logger.warning(f"Output validation failed for {self.action_key}: {e}")

            result = ActionResult(
                success=True,
                completed=True,
                output=ret
            )
            return result
        except Exception as e:
            logger.exception(f"Failed to execute action {self.action_key}")
            result = ActionResult(success=False, completed=True, error=str(e))
            return result

    async def get_action_description(self) -> str:
        """Get human-readable description from planning_summary or the wrapped method's docstring.

        Appends a compact parameter signature so the LLM knows exact field names.
        """
        if self.planning_summary:
            desc = self.planning_summary
        else:
            docstring = inspect.getdoc(self.method)
            if not docstring:
                raise ValueError(
                    f"No docstring found for action executor {self.action_key}. "
                    "Cannot generate description."
                )
            desc = docstring

        # Append parameter schema so the LLM uses correct field names
        if self.input_schema:
            sig_parts = []
            for name, field in self.input_schema.model_fields.items():
                ann = field.annotation
                type_name = getattr(ann, '__name__', str(ann))
                if field.is_required():
                    sig_parts.append(f"{name}: {type_name}")
                else:
                    sig_parts.append(f"{name}?: {type_name}")
            if sig_parts:
                desc += f"\n  Parameters: {', '.join(sig_parts)}"
        return desc


class FunctionWrapperActionExecutor(ActionExecutor):
    """Wraps a standalone function decorated with @action_executor.

    Enables using standalone functions (not methods) as action providers,
    which is useful for composing multi-agent patterns. For example, any agent
    can spawn a negotiation game as part of its action policy execution.

    Important: If the function's first parameter is typed as `Agent`, it is
    automatically injected with `action_policy.agent` at execution time.
    This parameter is NOT exposed to the LLM planner since it cannot
    "select" an agent - the owner agent is implicit.

    Example:
        ```python
        @action_executor(writes=["game_result"])
        async def run_negotiation_game(
            owner: Agent,           # <-- Auto-injected, not in input schema
            issue: NegotiationIssue,
            ...
        ) -> CapabilityResultFuture:
            '''Spawn a negotiation game.'''
            ...

        # Add to action policy
        policy = MyPolicy(
            agent=agent,
            action_providers=[run_negotiation_game],  # Standalone function
        )
        # LLM only sees: run_negotiation_game(issue=..., ...)
        # The 'owner' is auto-filled with policy.agent
        ```
    """

    def __init__(
        self,
        func: Callable,
        action_key: str | ActionType,
        agent: Agent,
        input_schema: type[BaseModel] | None = None,
        output_schema: type[BaseModel] | None = None,
        reads: list[str] | None = None,
        writes: list[str] | None = None,
        first_param_is_agent: bool = False,
        first_param_name: str | None = None,
        exclude_from_planning: bool = False,
        planning_summary: str | None = None,
    ):
        self.func = func
        self.action_key = str(action_key)
        self.agent = agent
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.reads = reads or []
        self.writes = writes or []
        self.first_param_is_agent = first_param_is_agent
        self.first_param_name = first_param_name
        self.exclude_from_planning = exclude_from_planning
        self.planning_summary = planning_summary

    async def execute(self, action: Action, resolved_params: dict[str, Any] | None = None) -> ActionResult:
        """Execute the wrapped function.

        If the function's first parameter is an Agent, it is automatically
        injected from self.agent (the dispatcher's agent).

        Args:
            action: The action to execute
            resolved_params: Pre-resolved parameters (with Refs resolved).
                If None, uses action.parameters directly.

        Returns:
            ActionResult with function return value as output
        """
        try:
            params = resolved_params if resolved_params is not None else action.parameters

            # Validate against input schema if available
            if self.input_schema:
                try:
                    validated = self.input_schema(**params)
                    params = validated.model_dump()
                except ValidationError as e:
                    return ActionResult(
                        success=False,
                        completed=True,
                        error=f"Parameter validation failed: {e}"
                    )

            # Inject agent as first parameter if needed
            if self.first_param_is_agent and self.first_param_name:
                params = {self.first_param_name: self.agent, **params}

            # Call the function with parameters as kwargs
            ret = await self.func(**params)

            # Validate output if schema available
            if self.output_schema and ret is not None:
                try:
                    if not isinstance(ret, dict):
                        validated = self.output_schema(value=ret)
                    else:
                        validated = self.output_schema(**ret)
                    ret = validated.model_dump()
                except ValidationError as e:
                    logger.warning(f"Output validation failed for {self.action_key}: {e}")

            return ActionResult(
                success=True,
                completed=True,
                output=ret
            )

        except Exception as e:
            logger.exception(f"Error executing function {self.action_key}")
            return ActionResult(
                success=False,
                completed=True,
                error=str(e)
            )

    async def get_action_description(self) -> str:
        """Get description from planning_summary or function docstring.

        Appends a compact parameter signature so the LLM knows exact field names.
        """
        if self.planning_summary:
            desc = self.planning_summary
        else:
            docstring = inspect.getdoc(self.func)
            if not docstring:
                raise ValueError(
                    f"No docstring found for function executor {self.action_key}. "
                    "Cannot generate description."
                )
            desc = docstring

        # Append parameter schema so the LLM uses correct field names
        if self.input_schema:
            sig_parts = []
            for name, field in self.input_schema.model_fields.items():
                ann = field.annotation
                type_name = getattr(ann, '__name__', str(ann))
                if field.is_required():
                    sig_parts.append(f"{name}: {type_name}")
                else:
                    sig_parts.append(f"{name}?: {type_name}")
            if sig_parts:
                desc += f"\n  Parameters: {', '.join(sig_parts)}"
        return desc


def _infer_input_schema(func: Callable) -> type[BaseModel] | None:
    """Infer Pydantic input schema from function signature and type hints.

    Args:
        func: Function to analyze

    Returns:
        Dynamically created Pydantic model or None if no parameters
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    fields = {}
    for name, param in sig.parameters.items():
        # Skip 'self' parameter
        if name == 'self':
            continue

        # Get type hint (default to Any)
        field_type = hints.get(name, Any)

        # Handle default values
        if param.default is inspect.Parameter.empty:
            fields[name] = (field_type, ...)  # Required field
        else:
            fields[name] = (field_type, param.default)  # Optional with default

    if not fields:
        return None

    # Create dynamic Pydantic model — validate that all parameter types
    # are JSON-serializable since action executors are called by the LLM planner.
    try:
        model = create_model(f"{func.__name__}_Input", **fields)
        # Resolve forward references (e.g. TYPE_CHECKING imports like Action)
        try:
            model.model_rebuild()
        except Exception:
            pass  # Best-effort — fails if referenced types aren't importable
        return model
    except PydanticSchemaGenerationError:
        _raise_non_serializable_error(func, fields)


def _infer_input_schema_excluding_first(func: Callable) -> type[BaseModel] | None:
    """Infer input schema excluding the first parameter.

    Used for standalone action functions where the first parameter is
    an Agent that gets auto-injected (not exposed to LLM planners).

    Args:
        func: Function to analyze

    Returns:
        Dynamically created Pydantic model or None if no remaining parameters
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    fields = {}
    params = list(sig.parameters.items())

    # Skip first parameter (the Agent param that gets auto-injected)
    for name, param in params[1:]:
        # Get type hint (default to Any)
        field_type = hints.get(name, Any)

        # Handle default values
        if param.default is inspect.Parameter.empty:
            fields[name] = (field_type, ...)  # Required field
        else:
            fields[name] = (field_type, param.default)  # Optional with default

    if not fields:
        return None

    try:
        model = create_model(f"{func.__name__}_Input", **fields)
        try:
            model.model_rebuild()
        except Exception:
            pass
        return model
    except PydanticSchemaGenerationError:
        _raise_non_serializable_error(func, fields)


def _raise_non_serializable_error(
    func: Callable,
    fields: dict[str, tuple],
) -> None:
    """Identify which parameters have non-serializable types and raise a clear error.

    Called when create_model() fails with PydanticSchemaGenerationError.
    Tests each field individually to pinpoint the offending parameter(s).
    """
    bad_params = []
    for name, (field_type, *_) in fields.items():
        try:
            create_model("_TypeCheck", **{name: (field_type, ...)})
        except PydanticSchemaGenerationError:
            bad_params.append((name, field_type))

    detail = ", ".join(f"'{n}' (type: {t})" for n, t in bad_params)
    raise TypeError(
        f"@action_executor '{func.__qualname__}' has parameters with non-serializable "
        f"types: {detail}. "
        f"Since action executors are called by the LLM planner, all parameters must be "
        f"JSON-serializable Pydantic types (str, int, float, bool, list, dict, BaseModel "
        f"subclass, enum, etc.) or 'Any'. Either change the type annotation or remove the "
        f"parameter and load the data internally."
    ) from None


def _infer_output_schema(func: Callable) -> type[BaseModel] | None:
    """Infer Pydantic output schema from return type hint.

    Args:
        func: Function to analyze

    Returns:
        Return type if it's a Pydantic model, or dynamically created model, or None
    """
    try:
        hints = get_type_hints(func)
    except Exception:
        return None

    return_type = hints.get('return')

    if return_type is None or return_type is type(None):
        return None

    # If already a Pydantic model, use directly
    if isinstance(return_type, type) and issubclass(return_type, BaseModel):
        return return_type

    # For simple types, wrap in a model with 'value' field
    if return_type in (str, int, float, bool):
        return create_model(f"{func.__name__}_Output", value=(return_type, ...))

    # For generic types (list[str], Optional[X], etc.), wrap in model
    origin = get_origin(return_type)
    if origin is not None:
        # Skip dict returns — they're inherently unstructured and the schema
        # wrapper would conflict with dict unpacking in output validation.
        if origin is dict:
            return None
        return create_model(f"{func.__name__}_Output", value=(return_type, ...))

    # For complex types we don't recognize, skip validation
    return None


def action_executor(
    action_key: str | ActionType | None = None,
    *,
    input_schema: type[BaseModel] | None = None,
    output_schema: type[BaseModel] | None = None,
    reads: list[str] | None = None,
    writes: list[str] | None = None,
    exclude_from_planning: bool = False,
    planning_summary: str | None = None,
    tags: frozenset[str] | None = None,
):
    """Decorator to turn any method into an action executor.

    Automatically infers input/output schemas from type hints if not provided.

    Args:
        action_key: Key identifying the action type. If None, uses method name.
        input_schema: Optional Pydantic model for input validation.
            If None, inferred from method signature.
        output_schema: Optional Pydantic model for output validation.
            If None, inferred from return type hint.
        reads: List of scope variable names this action reads.
        writes: List of scope variable names this action writes.
        exclude_from_planning: If True, this action is not exposed to the LLM
            planner. Use this for actions that are only meant to be invoked
            programmatically in response to events (e.g., game moves in response
            to spawned agent events). Default is False.
        tags: Optional domain/modality tags for this action (e.g., frozenset({"memory", "expensive"})).
            Used for future per-action tag-based filtering and grouping.

    Example:
        ```python
        @action_executor()
        async def route_query(
            self,
            query: str,
            max_results: int = 10
        ) -> list[str]:
            '''Route query to find relevant pages.'''
            ...

        @action_executor(writes=["analysis_result"])
        async def analyze_pages(
            self,
            page_ids: list[str],
            goal: str
        ) -> AnalysisResult:
            '''Analyze pages for the given goal.'''
            ...

        # Event-driven action not visible to planner
        @action_executor(exclude_from_planning=True)
        async def submit_move(self, game_id: str, move: dict) -> None:
            '''Submit move in response to game event.'''
            ...
        ```
    """
    def decorator(func):
        # Store action key
        func._action_key = action_key or func.__name__

        # Infer schemas from type hints if not provided
        func._action_input_schema = input_schema or _infer_input_schema(func)
        func._action_output_schema = output_schema or _infer_output_schema(func)

        # Store scope read/write declarations
        func._action_reads = reads or []
        func._action_writes = writes or []

        # Store planning visibility flag
        func._action_exclude_from_planning = exclude_from_planning

        # Store concise planning summary (used instead of full docstring in prompts)
        func._action_planning_summary = planning_summary

        # Store tags for future per-action tag-based filtering
        func._action_tags = tags or frozenset()

        return func

    return decorator



class ActionGroup(BaseModel):
    """Mapping of a group of action keys to action executors sharing
    a common group description. This is needed to disambiguate actions
    from different providers of the same type but different roles (e.g.,
    multiple MemoryCapability instances at different memory levels).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    group_key: str = Field(description="Stable identifier for this action group (e.g., capability_key).")
    description: str = Field(description="Description of the action group.")
    executors: dict[str, ActionExecutor] = Field(default_factory=dict)
    tags: frozenset[str] = Field(default_factory=frozenset, description="Domain/modality/cost tags for hierarchical action scoping.")



class ActionDispatcher:
    """Dispatch and execute actions within agent context with dataflow support.

    Handles:
    - Action execution with automatic parameter resolution
    - `Ref` resolution from scope, results, capabilities, and blackboard
    - Result storage in scope for subsequent actions
    - Scope variable writes declared by action executors

    TODO: Add support for MCP tools integration.
    """

    def __init__(
        self,
        agent: Agent,
        action_policy: ActionPolicy,
        action_map: list[ActionGroup] | None = None,
        action_providers: list[Any] = [],
    ):
        """Initialize action dispatcher.

        Args:
            agent: Agent executing actions
            action_policy: Action policy managing the reasoning loop
            action_map: A list of ActionGroup instances mapping action types to executors
            action_providers: Additional objects providing action executors
                as methods decorated with @action_executor
        """
        self.agent = agent
        self.action_policy = action_policy
        self.action_map = action_map or []
        self.action_providers = action_providers
        self._repl: PolicyPythonREPL | None = None
        self._repl_discovered = False

    @property
    def repl(self) -> PolicyPythonREPL | None:
        """Get REPL if available from agent capability.

        Lazily discovers REPLCapability on first access.

        Returns:
            PolicyPythonREPL instance or None if not available
        """
        if not self._repl_discovered:
            self._repl_discovered = True
            # Try to find REPLCapability on the agent
            for cap in self.agent.get_capabilities():
                if isinstance(cap, REPLCapability):
                    self._repl = cap.repl
                    break
        return self._repl

    async def initialize(self):
        """Initialize dispatcher if needed."""
        # Create default action executors based on agent type
        self.action_map.extend(
            self._create_default_action_executors()
        )

    def _create_default_action_executors(self) -> list[ActionGroup]:
        """Create default action executors based on agent type."""
        action_groups: list[ActionGroup] = []

        for source in [self.agent, self.action_policy]:
            group = self._create_object_action_group(source)
            if group.executors:
                action_groups.append(group)

        for provider in self.action_providers:
            # Check if provider is a standalone function decorated with @action_executor
            if callable(provider) and hasattr(provider, '_action_key'):
                action_executor = self._create_function_action_executor(provider)
                if action_executor:
                    action_groups.append(ActionGroup(
                        group_key=f"func.{provider.__name__}",
                        description=f"Actions from function {provider.__name__}",
                        executors={action_executor.action_key: action_executor},
                        tags=getattr(provider, '_action_tags', frozenset()),
                    ))
            else:
                # It's an object with methods
                group = self._create_object_action_group(provider)
                if group.executors:
                    action_groups.append(group)

        return action_groups

    def _create_function_action_executor(self, func: Callable) -> FunctionWrapperActionExecutor | None:
        """Create an action executor from a standalone function decorated with @action_executor.

        This enables composing multi-agent patterns by allowing functions like
        `run_negotiation_game` to be used as action providers.

        If the function's first parameter is typed as Agent, it is automatically
        injected with self.agent at execution time (not exposed to LLM planners).

        Args:
            func: Function decorated with @action_executor

        Returns:
            FunctionWrapperActionExecutor or None if not a valid action executor
        """
        if not hasattr(func, '_action_key'):
            return None

        # Detect if first parameter is Agent type
        first_param_is_agent = False
        first_param_name = None
        input_schema = getattr(func, '_action_input_schema', None)

        sig = inspect.signature(func)
        params = list(sig.parameters.items())

        if params:
            first_name, first_param = params[0]
            try:
                hints = get_type_hints(func)
                first_type = hints.get(first_name)
                # Check if first param is Agent or a subclass
                if first_type is not None and isinstance(first_type, type) and issubclass(first_type, Agent):
                    first_param_is_agent = True
                    first_param_name = first_name

                    # Re-infer input schema excluding the first Agent parameter
                    input_schema = _infer_input_schema_excluding_first(func)
            except Exception:
                # If type hints fail, fall back to decorator-provided schema
                pass

        return FunctionWrapperActionExecutor(
            func=func,
            action_key=func._action_key,
            agent=self.agent,
            input_schema=input_schema,
            output_schema=getattr(func, '_action_output_schema', None),
            reads=getattr(func, '_action_reads', []),
            writes=getattr(func, '_action_writes', []),
            first_param_is_agent=first_param_is_agent,
            first_param_name=first_param_name,
            exclude_from_planning=getattr(func, '_action_exclude_from_planning', False),
        )

    def _create_object_action_group(self, obj: Any) -> ActionGroup:
        """Create action executors from @action_executor decorated methods.

        Respects action filters set via Agent.add_capability():
        - _action_include_filter: frozenset of action keys to include, or None for all
        - _action_exclude_filter: frozenset of action keys to exclude

        Actions not in the exposed set are skipped (not registered with dispatcher).
        They can still be invoked directly on the capability.
        """
        action_executors: dict[str, ActionExecutor] = {}

        # Get action filters if set (e.g., via Agent.add_capability)
        include_filter: frozenset[str] | None = getattr(obj, '_action_include_filter', None)
        exclude_filter: frozenset[str] = getattr(obj, '_action_exclude_filter', frozenset())

        if not hasattr(obj, '_action_dispatch_key') or not obj._action_dispatch_key:
            # Unique short ID to avoid name clashes
            obj._action_dispatch_key = uuid.uuid4().hex[:8]

        # Register methods decorated with @action_executor.
        # Walk MRO to discover inherited methods (not just immediate class).
        seen_names: set[str] = set()
        for cls in type(obj).__mro__:
            if cls is object:
                continue
            for name, method in cls.__dict__.items():
                if name in seen_names:
                    continue  # Most-derived class wins
                seen_names.add(name)
                if not hasattr(method, '_action_key'):
                    continue
                action_key = method._action_key

                # Apply action filters
                if include_filter is not None and action_key not in include_filter:
                    # include_filter is set and this action is not in it - skip
                    continue
                if action_key in exclude_filter:
                    # Action is explicitly excluded - skip
                    continue

                executor = MethodWrapperActionExecutor(
                    object=obj,
                    method=method,
                    action_key=action_key,
                    input_schema=getattr(method, '_action_input_schema', None),
                    output_schema=getattr(method, '_action_output_schema', None),
                    reads=getattr(method, '_action_reads', []),
                    writes=getattr(method, '_action_writes', []),
                    exclude_from_planning=getattr(method, '_action_exclude_from_planning', False),
                    planning_summary=getattr(method, '_action_planning_summary', None),
                )
                # We can have multiple capabilities of the same type (e.g., memory
                # capabilities) and/or capabilities of different types but with same action key.
                full_action_name = f"{obj.__class__.__name__}.{obj._action_dispatch_key}.{executor.action_key}"
                action_executors[full_action_name] = executor

        # Resolve group description: blueprint override > instance method > default
        description = getattr(obj, '_action_group_description', None)
        if description is None and hasattr(obj, 'get_action_group_description'):
            try:
                description = obj.get_action_group_description()
            except Exception:
                pass

        if description is None:
            description = f"Actions from {obj.__class__.__name__}.{obj._action_dispatch_key}"
            logger.warning(
                f"Using default action group description for {obj.__class__.__name__}. "
                "Consider implementing get_action_group_description() for better LLM planning."
            )

        # Resolve tags: blueprint override > class method > empty
        tags: frozenset[str] = getattr(obj, '_action_tags', None) or frozenset()
        if not tags and hasattr(obj, 'get_capability_tags') and callable(obj.get_capability_tags):
            tags = obj.get_capability_tags()

        # Resolve group_key: capability_key > ClassName.dispatch_key
        group_key = getattr(obj, '_capability_key', None) or getattr(obj, 'capability_key', None)
        if not group_key:
            group_key = f"{obj.__class__.__name__}.{obj._action_dispatch_key}"

        return ActionGroup(
            group_key=group_key,
            description=description,
            executors=action_executors,
            tags=tags,
        )

    def get_plannable_actions(self) -> dict[str, ActionExecutor]:
        """Get actions that are visible to the LLM planner.

        Filters out actions marked with `exclude_from_planning=True`.
        These excluded actions are only meant to be invoked programmatically
        in response to events (e.g., game moves triggered by spawned agent events).

        Returns:
            Dictionary mapping action keys to executors for plannable actions
        """
        return {
            key: executor
            for group in self.action_map
            for key, executor in group.executors.items()
            if not getattr(executor, 'exclude_from_planning', False)
        }

    async def get_action_descriptions(
        self,
        selected_groups: list[str] | None = None,
    ) -> list[ActionGroupDescription]:
        """Get human-readable description of plannable actions for LLM-based
        action selection.

        Only includes actions visible to the planner (excludes actions marked
        with `exclude_from_planning=True`).

        If REPL is available, includes REPL guidance and variable summary.

        Args:
            selected_groups: If provided, only return descriptions for groups
                whose group_key is in this list. If None, return all groups.

        Returns:
            List of ActionGroupDescription with full action descriptions.
        """
        descriptions: list[ActionGroupDescription] = []
        for group in self.action_map:
            if selected_groups is not None and group.group_key not in selected_groups:
                continue
            action_descs: dict[str, str] = {}
            for action_key, executor in group.executors.items():
                if getattr(executor, 'exclude_from_planning', False):
                    continue
                try:
                    desc = await executor.get_action_description()
                    action_descs[action_key] = desc
                except NotImplementedError:
                    # Use the docstring of the execute() method instead
                    doc = executor.execute.__doc__
                    if doc:
                        action_descs[action_key] = doc.strip()
            if action_descs:
                descriptions.append(ActionGroupDescription(
                    group_key=group.group_key,
                    group_description=group.description,
                    action_descriptions=action_descs,
                    tags=group.tags,
                    action_count=len(action_descs),
                ))

        # REPL is a meta-capability — always included regardless of selected_groups
        if self.repl:
            # Add EXECUTE_CODE action description
            descriptions.append(ActionGroupDescription(
                group_key="repl",
                group_description="Python REPL Actions",
                action_descriptions={
                    str(ActionType.EXECUTE_CODE): (
                        "Execute Python code in the REPL context. "
                        "Use for data transformations, filtering, aggregation, "
                        "or generating multiple actions programmatically."
                    ),
                },
                action_count=1,
            ))
            # Add REPL variable summary
            descriptions.append(ActionGroupDescription(
                group_key="repl.variables",
                group_description="REPL Variables",
                action_descriptions={"__repl_variables__": self.repl.get_variable_summary()},
            ))
            descriptions.append(ActionGroupDescription(
                group_key="repl.guidance",
                group_description="REPL Guidance",
                action_descriptions={"__repl_guidance__": get_repl_guidance(self.repl)},
            ))

        return descriptions

    def get_action_group_summaries(self) -> list[ActionGroupDescription]:
        """Get lightweight summaries of action groups for scope selection.

        Returns one entry per group with NO individual action details.
        Used by the scope selection phase to let the LLM choose relevant groups.

        Returns:
            List of ActionGroupDescription with empty action_descriptions
            but populated action_count, tags, and group_description.
        """
        summaries: list[ActionGroupDescription] = []
        for group in self.action_map:
            plannable = sum(
                1 for e in group.executors.values()
                if not getattr(e, 'exclude_from_planning', False)
            )
            if plannable > 0:
                summaries.append(ActionGroupDescription(
                    group_key=group.group_key,
                    group_description=group.description,
                    tags=group.tags,
                    action_count=plannable,
                ))
        return summaries

    @hookable
    async def dispatch(
        self,
        action: Action,
    ) -> ActionResult:
        """Dispatch an action with dataflow support.

        This method is @hookable so memory capabilities can observe action execution.
        The Action (first argument) has its `result` attached after execution.

        Note: This dispatcher handles Actions only. ActionPolicies (nested policies)
        are handled directly by BaseActionPolicy.execute_iteration.

        Uses PolicyPythonREPL for:
        - Action result storage (by action_id)
        - Variable storage with metadata
        - Ref resolution ($variable, $results.action_id)
        - Code execution (EXECUTE_CODE actions)

        Args:
            action: Action to execute

        Returns:
            ActionResult with execution outcome
        """

        try:
            return await self._dispatch_action(action)
        except Exception as e:
            logger.exception(f"Failed to execute action {getattr(action, 'action_id', 'unknown')}")
            result = ActionResult(success=False, completed=True, error=str(e))
            action.status = ActionStatus.FAILED
            action.result = result
            return result

    def _get_executor_for_action(self, action: Action) -> ActionExecutor | None:
        """Get the executor for a given action by exact key match."""
        action_key = str(action.action_type)
        for group in self.action_map:
            if action_key in group.executors:
                return group.executors[action_key]
        return None

    async def _dispatch_action(
        self,
        action: Action,
    ) -> ActionResult:
        """Dispatch a single action with Ref resolution and REPL integration."""
        action_key = str(action.action_type)

        # Handle EXECUTE_CODE via REPL if available
        if action_key == str(ActionType.EXECUTE_CODE) and self.repl:
            return await self._execute_repl_code(action)

        executor = self._get_executor_for_action(action)
        if executor is None:
            return ActionResult(
                success=False,
                error=f"Unknown action type: {action_key}"
            )

        # Resolve Ref values in parameters (uses REPL for variable/result resolution)
        resolved_params = await self._resolve_refs(action.parameters)

        # Execute action
        action.status = ActionStatus.IN_PROGRESS

        # Use resolved params with MethodWrapperActionExecutor
        # TODO: Allow Action.parameters to also use resolved_params in other executor types
        # Using scope._has_shared_data_dependencies() is not correct because
        # the shared-state dependencies are neither per-action, nor per-scope.
        # Rather, when a shared state (e.g., game state) changes, all actions
        # based on that new state should be part of the same transaction.
        # This requires a more global mechanism (the specific `AgentCapability`
        # responsible for the protocol maintaining the shared state, e.g., the
        # GameProtocol) to track shared-state dependencies and provide transactors.
        # if scope._has_shared_data_dependencies():
        #     result = await self._execute_with_shared_data_dependencies(
        #         action=action,
        #         scope=scope,
        #         executor=executor,
        #         resolved_params=resolved_params,
        #     )
        # else:
        if isinstance(executor, (MethodWrapperActionExecutor, FunctionWrapperActionExecutor)):
            result = await executor.execute(action, resolved_params)
        else:
            # Fallback for other executor types
            result = await executor.execute(action)

        logger.info(f"______ Action execution result: {result}")

        # Update action status
        action.status = ActionStatus.COMPLETED if result.success else ActionStatus.FAILED
        action.result = result

        # Store result in REPL
        if self.repl:
            self.repl.set_result(action.action_id, result)

            # Write declared variables to REPL
            # TODO: Allow the action itself to specify variable writes
            # (e.g., action.writes = ["var1", "var2"]) to be generated by
            # the planner.
            if hasattr(executor, 'writes') and executor.writes:
                await self._write_to_repl(action, result, executor.writes)

        # Store result in REPL as named variable if result_var is specified
        if action.result_var and self.repl:
            storage_hint = getattr(action, 'storage_hint', None)
            await self.repl.set(
                name=action.result_var,
                value=result.output,
                description=f"Result of {action.action_type} (action_id={action.action_id})",
                created_by=action.action_id,
                storage_hint=storage_hint,
            )

        # Process any pending actions from REPL
        if self.repl:
            await self._process_pending_repl_actions()

        return result

    async def _execute_repl_code(
        self,
        action: Action,
    ) -> ActionResult:
        """Execute Python code in REPL.

        Args:
            action: EXECUTE_CODE action

        Returns:
            ActionResult with execution output
        """
        if not self.repl:
            return ActionResult(
                success=False,
                error="REPL not available for EXECUTE_CODE action"
            )

        # Get code from action
        code = action.parameters.get("code", "") or action.code or ""
        if not code:
            return ActionResult(
                success=False,
                error="No code provided for EXECUTE_CODE action"
            )

        action.status = ActionStatus.IN_PROGRESS
        result_dict = await self.repl.execute(code)

        result = ActionResult(
            success=result_dict["success"],
            output=result_dict,
        )

        action.status = ActionStatus.COMPLETED if result.success else ActionStatus.FAILED
        action.result = result

        # Store result in REPL
        self.repl.set_result(action.action_id, result)

        # Store result as named variable if result_var is specified
        if action.result_var:
            storage_hint = getattr(action, 'storage_hint', None)
            await self.repl.set(
                name=action.result_var,
                value=result_dict,
                description=f"EXECUTE_CODE result (action_id={action.action_id})",
                created_by=action.action_id,
                storage_hint=storage_hint,
            )

        # Recursively execute pending actions
        await self._process_pending_repl_actions()

        return result

    async def _process_pending_repl_actions(self) -> None:
        """Execute any pending actions generated by REPL code."""
        if not self.repl:
            return

        while True:
            pending = self.repl.get_pending_actions()
            if not pending:
                break
            for pending_action in pending:
                await self.dispatch(pending_action)

    # This method is only kept for reference; see note above about shared-state dependencies.
    async def _execute_with_shared_data_dependencies(
        self,
        *,
        action: Action,
        repl: PolicyREPL,
        executor: ActionExecutor,
        resolved_params: dict[str, Any],
    ) -> ActionResult:
        """Execute an action after acquiring and validating state dependencies."""
        deps: list[ActionSharedDataDependency] = repl._get_sorted_shared_data_dependencies()

        # NOTE (assumption for shared state managers):
        # The dependency `transactor` should provide *ambient transaction* semantics for the
        # shared-state manager used by action executors. Concretely, entering the transactor
        # should ensure that shared-state reads/writes performed by executors participate in
        # the same atomic commit/rollback as the transactor exits.
        #
        # EnhancedBlackboard provides this via `async with blackboard.transaction()` which
        # makes the transaction ambient (blackboard.read/write/delete route through it).
        try:
            async with AsyncExitStack() as stack:
                conflicts: list[dict[str, Any]] = []

                for dep in deps:
                    if dep.transactor is None:
                        return ActionResult(
                            success=False,
                            completed=True,
                            error=f"Missing transactor for state dependency: {dep.data_key}",
                            metadata={"kind": "state_dependency_error", "data_key": dep.data_key},
                        )

                    # Enter dependency context (may acquire lock / open transaction)
                    handle = await stack.enter_async_context(dep.transactor)

                    # Version validation options (in increasing generality):
                    # 1) BlackboardTransaction-like handle: supports read() + version_tokens
                    #    In this case, dep.data_key is expected to be a key. We trigger a read
                    #    so the transaction captures the backend version token, then compare tokens.
                    if handle is not None and hasattr(handle, "read") and hasattr(handle, "version_tokens"):
                        try:
                            entry = await handle.read(dep.data_key)
                        except TypeError:
                            # read signature mismatch; fall back to marker extraction
                            entry = None

                        # Preferred dependency marker: persisted entry.version (backend-agnostic).
                        if entry is not None and hasattr(entry, "version") and isinstance(dep.expected_version, int):
                            current_ver = getattr(entry, "version")
                            if current_ver != dep.expected_version:
                                conflicts.append(
                                    {
                                        "data_key": dep.data_key,
                                        "expected_version": dep.expected_version,
                                        "current_version": current_ver,
                                    }
                                )
                                continue

                        # Backward-compatible marker: backend version token (string).
                        current_token = getattr(handle, "version_tokens", {}).get(dep.data_key)
                        if isinstance(dep.expected_version, str) and current_token != dep.expected_version:
                            conflicts.append(
                                {
                                    "data_key": dep.data_key,
                                    "expected_version": dep.expected_version,
                                    "current_version": current_token,
                                }
                            )
                            continue

                    # 2) Handle exposes a version marker (get_version/current_version/version)
                    current_version = self._extract_dependency_version(handle)
                    if current_version is not None and current_version != dep.expected_version:
                        conflicts.append(
                            {
                                "data_key": dep.data_key,
                                "expected_version": dep.expected_version,
                                "current_version": current_version,
                            }
                        )

                if conflicts:
                    return ActionResult(
                        success=False,
                        completed=True,
                        error="Shared state dependency conflict (stale plan). Replan on next iteration.",
                        metadata={"kind": "shared_state_dependency_conflict", "conflicts": conflicts},
                    )

                if isinstance(executor, MethodWrapperActionExecutor):
                    return await executor.execute(action, resolved_params)
                return await executor.execute(action)
        except ConcurrentModificationError as e:
            # Conflict detected by optimistic lock on commit or by a transactor raising.
            return ActionResult(
                success=False,
                completed=True,
                error=str(e),
                metadata={"kind": "shared_state_dependency_conflict", "conflicts": [{"error": str(e)}]},
            )

    def _extract_dependency_version(self, handle: Any) -> Any:
        """Extract a version marker from a dependency handle."""
        if handle is None:
            return None
        get_version = getattr(handle, "get_version", None)
        if callable(get_version):
            return get_version()
        if hasattr(handle, "current_version"):
            return getattr(handle, "current_version")
        if hasattr(handle, "version"):
            return getattr(handle, "version")
        return None

    async def _resolve_refs(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve all Ref values in parameters.

        Uses REPL for variable and result resolution.

        Args:
            params: Parameters that may contain Ref values

        Returns:
            Parameters with Refs resolved to actual values
        """
        resolved = {}
        for key, value in params.items():
            resolved[key] = await self._resolve_value(value)
        return resolved

    async def _resolve_value(self, value: Any) -> Any:
        """Resolve a single value, handling Refs recursively."""
        if isinstance(value, Ref):
            return await self._resolve_ref(value)
        elif isinstance(value, dict):
            return {k: await self._resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [await self._resolve_value(v) for v in value]
        else:
            return value

    async def _resolve_ref(self, ref: Ref) -> Any:
        """Resolve a single Ref to its value.

        Uses REPL for:
        - $variable_name: REPL namespace variable
        - $results.action_id: REPL action result storage

        Args:
            ref: Reference to resolve

        Returns:
            Resolved value

        Raises:
            ValueError: If reference cannot be resolved
        """
        parts = ref.get_parts()

        if not parts:
            raise ValueError(f"Invalid reference: {ref.path}")

        root = parts[0]
        remaining = parts[1:]

        if root == "results":
            # $results.action_id.output.field
            if not remaining:
                raise ValueError(f"Result reference missing action_id: {ref.path}")
            action_id = remaining[0]
            if not self.repl:
                raise ValueError(f"No REPL available for result reference: {ref.path}")
            result = self.repl.get_result(action_id)
            if result is None:
                raise ValueError(f"No result found for action: {action_id}")
            return self._navigate(result.model_dump(), remaining[1:])

        elif root == "agent":
            # $agent.CapabilityName.attr
            if not remaining:
                raise ValueError(f"Capability reference missing name: {ref.path}")
            cap_name = remaining[0]
            capability = self.agent.get_capability(cap_name)
            if capability is None:
                raise ValueError(f"Capability not found: {cap_name}")
            return self._navigate(capability, remaining[1:])

        elif root == "global":
            # $global.blackboard_key
            key = ".".join(remaining) if remaining else ""
            if not key:
                raise ValueError(f"Blackboard reference missing key: {ref.path}")
            board = await self.agent.get_blackboard()  # TODO: This is the agent's blackboard. Is this correct, or should there be a separate "global" blackboard for cross-agent shared state?
            entry = await board.read(key)
            return entry.value if entry else None

        elif root == "shared":
            # $shared.data_key.path
            if not remaining:
                raise ValueError(f"Shared data reference missing data key: {ref.path}")
            data_key = remaining[0]
            if not data_key:
                raise ValueError(f"Shared data reference missing key: {ref.path}")
            if not self.repl:
                raise ValueError(f"No REPL available for shared data reference: {ref.path}")
            shared_data = self.repl.get_shared(data_key)  # TODO: Add set_shared/get_shared methods to REPL
            if shared_data is None:
                raise ValueError(f"No shared data found for key: {data_key}")
            return self._navigate(shared_data, remaining[1:])
        else:
            # $variable_name - REPL namespace variable
            if not self.repl:
                raise ValueError(f"No REPL available for variable reference: {ref.path}")
            # Use namespace directly for sync access (faster than async get)
            value = self.repl.namespace.get(root)
            if value is None:
                # Try get_sync which also updates access tracking
                value = self.repl.get_sync(root)
            return self._navigate(value, remaining)

    def _navigate(self, obj: Any, path: list[str]) -> Any:
        """Navigate an object by path.

        Args:
            obj: Object to navigate
            path: List of path segments

        Returns:
            Value at path
        """
        for part in path:
            if obj is None:
                return None
            if isinstance(obj, dict):
                obj = obj.get(part)
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return None
        return obj

    async def _write_to_repl(
        self,
        action: Action,
        result: ActionResult,
        writes: list[str],
    ) -> None:
        """Write declared variables to REPL from action result.

        Args:
            result: Action result
            writes: List of variable names to write
        """
        if not result.success or result.output is None or not self.repl:
            return

        output = result.output

        for var_name in writes:
            if isinstance(output, dict) and var_name in output:
                # Output is dict with matching key
                await self.repl.set(
                    name=var_name,
                    value=output[var_name],
                    description=f"Declared output variable",  # TODO: Allow writes to have descriptions
                    created_by=f"action_type{action.action_type}:{action.action_id}",
                )
            elif len(writes) == 1:
                # Single write declaration - use entire output
                await self.repl.set(
                    name=var_name,
                    value=output,
                    description=f"Declared output variable",  # TODO: Allow writes to have descriptions
                    created_by=f"action_type{action.action_type}:{action.action_id}",
                )



class BaseActionPolicy(ActionPolicy):
    """Base class for action policies with dataflow and nested policy support.

    Provides:
    - Automatic action dispatcher creation
    - Integration with agent capabilities
    - Nested policy execution with scope inheritance
    - Dispatch with automatic Ref resolution

    Subclasses implement `plan_step` to produce the next action or child policy.
    The base `execute_iteration` handles:
    - Delegating to active child policies
    - Executing actions returned by `plan_step`
    - Setting up child policies returned by `plan_step`

    TODO: For example, we can orchestrate iterative reasoning to follow the pattern
    (PLAN → ACT → REFLECT → CRITIQUE → ADAPT) by adding AgentCapabilities that
    implement each step as an action executor, and then implementing `plan_step`
    to select the next action based on the current state. This can be enforced by:
    - Restricting available actions in the action dispatcher depending on the
      last completed step, or
    - Using an ActionPolicy subclass that implements the iterative pattern by
      overriding `execute_iteration` to enforce the sequence of steps, and
      only calling `plan_step` to get parameters for each step, or
    - Prompting the LLM planner with this workflow.

    Example:
        ```python
        class MyPolicy(BaseActionPolicy):
            io = ActionPolicyIO(
                inputs={"query": str},
                outputs={"result": dict}
            )

            async def plan_step(self, state) -> Action | None:
                # Return None when policy is complete
                if state.custom.get("done"):
                    return None

                # Return an Action to execute
                return Action(
                    action_id="analyze_001",
                    action_type="analyze",
                    parameters={"query": state.scope.get("query")}
                )

                # Or return an ActionPolicy for nested execution
                # return ChildPolicy(self.agent)
        ```
    """

    def __init__(
        self,
        agent: Agent,
        action_map: list[ActionGroup] | None = None,
        action_providers: list[Any] = [],
        io: ActionPolicyIO | None = None, # Declare I/O contract (override in subclasses)
    ):
        super().__init__(agent)
        self._action_map = action_map
        self._action_providers = action_providers
        self._action_dispatcher: ActionDispatcher | None = None
        self.io: ActionPolicyIO = io or ActionPolicyIO()

    @override
    def use_agent_capabilities(self, capabilities: list[str]) -> None:
        """Add agent capabilities as action providers.

        Args:
            capabilities: List of capability names to use. Extends existing list.
        """
        super().use_agent_capabilities(capabilities)
        # Force recreation of dispatcher and action map
        self._action_dispatcher = None

    @override
    def disable_agent_capabilities(self, capabilities: list[str]) -> None:
        """Remove agent capabilities from action providers.

        Args:
            capabilities: List of capability names to disable.
        """
        super().disable_agent_capabilities(capabilities)
        # Force recreation of dispatcher and action map
        self._action_dispatcher = None

    @override
    async def initialize(self) -> None:
        """Initialize action policy."""
        await super().initialize()

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize policy-specific state.

        Subclasses should call super() and add their own state.
        """
        # Base implementation stores scope bindings
        state = await super().serialize_suspension_state(state)
        state.action_policy_state["scope_bindings"] = {}
        state.action_policy_state["scope_results"] = {}
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        """Restore policy-specific state from suspension.

        Subclasses should call super() and restore their own state.
        """
        # Base implementation - subclasses restore from state.policy_state
        pass

    async def get_action_descriptions(
        self,
        selected_groups: list[str] | None = None,
    ) -> list[ActionGroupDescription]:
        """Get descriptions of available actions.

        Args:
            selected_groups: If provided, only return descriptions for these group keys.
        """
        await self._create_action_dispatcher()
        return await self._action_dispatcher.get_action_descriptions(selected_groups=selected_groups)

    async def get_action_group_summaries(self) -> list[ActionGroupDescription]:
        """Get lightweight group summaries for scope selection."""
        await self._create_action_dispatcher()
        return self._action_dispatcher.get_action_group_summaries()

    async def dispatch(
        self,
        action: Action,
    ) -> ActionResult:
        """Dispatch an action with dataflow support.

        Uses PolicyPythonREPL (via ActionDispatcher) for variable and result storage.

        Args:
            action: Action to execute

        Returns:
            ActionResult with execution outcome
        """
        await self._create_action_dispatcher()
        result = await self._action_dispatcher.dispatch(action)
        # Do not clear shared data dependencies here because a data
        # dependency may span multiple actions in multiple policy
        # iterations and must only be handled by the appropriate
        # AgentCapability implementing a multi-agent protocol.
        # scope._clear_shared_data_dependencies()
        return result

    async def _create_action_dispatcher(self) -> None:
        """Create action dispatcher with capability providers."""
        if self._action_dispatcher:
            return

        # Collect capability providers
        capability_providers = self.get_used_capabilities()

        self._action_dispatcher = ActionDispatcher(
            agent=self.agent,
            action_policy=self,
            action_map=self._action_map,
            action_providers=capability_providers + self._action_providers,
        )

        await self._action_dispatcher.initialize()

    @hookable
    @override
    async def execute_iteration(
        self,
        state: ActionPolicyExecutionState
    ) -> ActionPolicyIterationResult:
        """Execute one iteration of this policy.

        This method is @hookable so memory capabilities can observe iterations.

        Calls `plan_step` to get next action, then dispatches it.

        For hierarchical composition (nested policies), spawn child agents
        instead of nesting policies. Use `self.agent.spawn_child_agents()`.

        Args:
            state: Execution state for this policy (all mutable state lives here)

        Returns:
            Iteration result
        """

        # TODO: Add iteration time limit check here
        ### if len(self.iteration_history) >= self.max_iterations:
        ###     logger.info("Analysis complete (max iterations reached)")
        ###     return ActionPolicyIterationResult(
        ###         success=True,
        ###         policy_completed=True
        ###     )

        # Set up session_id context for the ENTIRE iteration
        # Session_id may be set by a previous iteration's plan_step() from event metadata
        # This ensures all memory operations, hooks, and capabilities have session_id
        from ...sessions.context import session_id_context
        current_session_id = state.custom.get("current_session_id")

        with session_id_context(current_session_id):
            # Ensure dispatcher is initialized
            await self._create_action_dispatcher()

            # Get next action from subclass (plan_step may update current_session_id)
            logger.warning(
                f"\n"
                f"    ┌────────────────────────────────────────────┐\n"
                f"    │  ⚙ EXEC_ITER: calling plan_step            │\n"
                f"    │  agent={self.agent.agent_id:<38}│\n"
                f"    └────────────────────────────────────────────┘"
            )
            next_action = await self.plan_step(state)
            logger.warning(f"    ⚙ EXEC_ITER: plan_step returned → {type(next_action).__name__}: {next_action}")

            # Re-check session_id in case plan_step updated it from a new event
            updated_session_id = state.custom.get("current_session_id")
            if updated_session_id != current_session_id:
                # Session changed mid-iteration, update context for dispatch
                # This handles the case where plan_step processes a new event with different session_id
                from ...sessions.context import set_current_session_id
                set_current_session_id(updated_session_id)

            if next_action is None:
                # Check if policy signaled completion
                if state.custom.get("policy_complete"):
                    logger.warning(f"    ⚙ EXEC_ITER: policy_complete=True → TERMINATING")
                    return ActionPolicyIterationResult(
                        success=True,
                        policy_completed=True
                    )

                # Otherwise just skip this iteration (policy continues)
                logger.warning(f"    ⚙ EXEC_ITER: next_action=None → skipping iteration")
                return ActionPolicyIterationResult(
                    success=True,
                    policy_completed=False
                )

            # dispatch is @hookable, memory captures action there
            logger.warning(
                f"\n"
                f"    ╔════════════════════════════════════════════╗\n"
                f"    ║  🚀 DISPATCHING ACTION                    ║\n"
                f"    ║  id={next_action.action_id:<40}║\n"
                f"    ║  type={next_action.action_type:<38}║\n"
                f"    ╚════════════════════════════════════════════╝"
            )
            result = await self.dispatch(next_action)
            logger.warning(f"    🚀 DISPATCH returned: success={result.success}")

            return ActionPolicyIterationResult(
                success=result.success,
                policy_completed=False,
                action_executed=next_action,
                result=result,
            )

    async def plan_step(
        self,
        state: ActionPolicyExecutionState
    ) -> Action | None:
        """Produce the next action to execute.

        Override this method to implement policy-specific planning logic.

        For hierarchical composition, spawn child agents instead of nesting
        policies. Use `self.agent.spawn_child_agents()` with appropriate
        action policies for child agents.

        Args:
            state: Execution state for this policy

        Returns:
            - Action: Execute this action
            - None: Skip this iteration. Set `state.custom["policy_complete"] = True`
              before returning None to signal that the policy is finished.

        Example:
            ```python
            async def plan_step(self, state) -> Action | None:
                phase = state.custom.get("phase", "act")

                if phase == "act":
                    action = self._get_next_action(state)
                    if action is None:
                        state.custom["policy_complete"] = True
                        return None
                    state.custom["phase"] = "process"
                    return action

                elif phase == "process":
                    # Do some processing without dispatching an action
                    self._process_results(state)
                    state.custom["phase"] = "act"
                    return None  # Skip iteration, continue policy
            ```
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement plan_step"
        )


# ============================================================================
# Event-Driven Action Policy
# ============================================================================



class EventDrivenActionPolicy(BaseActionPolicy):
    """`ActionPolicy` that processes blackboard events via capabilities.

    Subclasses subscribe to blackboard events in `initialize()`.
    Events are queued and processed by `plan_step` on each iteration.

    This bridges the async event model with the synchronous iteration model
    without modifying `Agent.run_step()` or `BaseActionPolicy.execute_iteration()`.

    **Usage**:
        ```python
        class MyPolicy(EventDrivenActionPolicy):
            async def initialize(self) -> None:
                await super().initialize()
                capability = await self.agent.get_capability(CapabilityName)
                await capability.stream_events_to_queue(self.get_event_queue())

            @override
            async def plan_step(self, state) -> Action | None:
                event: BlackboardEvent = await self.get_next_event()
                if not event:
                    return None  # No events pending
                # Parse event.value and produce action
                return Action(action_type="process_event", parameters={...})
        ```
    """

    def __init__(
        self,
        agent: Agent,
        action_map: list[ActionGroup] | None = None,
        action_providers: list[Any] = [],
        io: ActionPolicyIO | None = None, # Declare I/O contract (override in subclasses)
        **kwargs
    ):
        super().__init__(agent, action_map=action_map, action_providers=action_providers, io=io, **kwargs)
        self._event_queue: asyncio.Queue[BlackboardEvent] = asyncio.Queue()
        self._subscribed_callbacks: list[Callable] = []

    @override
    async def initialize(self) -> None:
        """Initialize event-driven policy."""
        await super().initialize()

        # Subscribe to blackboard events.
        #
        # We subscribe both:
        # - Agent capabilities (the default modular extension mechanism), and
        # - Explicit action_providers (for backwards compatibility / advanced composition).
        #
        # Deduplicate by object identity to avoid double-streaming the same capability.
        seen: set[int] = set()
        for provider in list(self.agent.get_capabilities()) + list(self._action_providers):
            if id(provider) in seen:
                continue
            seen.add(id(provider))

            if isinstance(provider, AgentCapability) and hasattr(provider, "stream_events_to_queue"):
                await provider.stream_events_to_queue(self.get_event_queue())

    def get_event_queue(self) -> asyncio.Queue[BlackboardEvent]:
        """Get the local event queue.

        Returns:
            Local asyncio.Queue of BlackboardEvents
        """
        return self._event_queue

    @hookable
    async def get_next_event(self) -> BlackboardEvent | None:
        """Get the next pending event (non-blocking).

        This method is @hookable so memory capabilities can observe events.
        The returned BlackboardEvent can be captured by sensory memory hooks.

        The plan_step method can call this method to get as many
        events as needed within a single iteration.

        Returns:
            The next pending event, or None if no events are pending
        """
        try:
            return self._event_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def _get_event_handlers(self) -> list[Callable]:
        """Get all event handlers from capabilities and action providers.
        
        Returns:
            List of event handler methods (decorated with @event_handler)
        """
        handlers = []
        
        # Check agent capabilities
        for cap in self.agent.get_capabilities():
            handlers.extend(self._get_object_event_handlers(cap))
        
        # Check action providers
        for provider in self._action_providers:
            handlers.extend(self._get_object_event_handlers(provider))
        
        return handlers
    
    def _get_object_event_handlers(self, obj: Any) -> list[Callable]:
        """Get event handlers from an object.
        
        Args:
            obj: Object to inspect for @event_handler decorated methods
        
        Returns:
            List of event handler methods
        """
        handlers = []
        for name in dir(obj):
            method = getattr(obj, name, None)
            if callable(method) and hasattr(method, '_is_event_handler'):
                handlers.append(method)
        return handlers

    @override
    async def plan_step(self, state: ActionPolicyExecutionState) -> Action | None:
        """Plan next action with event-driven context enrichment.

        Flow:
        1. Get next event from queue via get_next_event() (non-blocking)
        2. Extract session_id from event metadata and set up context
        3. If event exists, broadcast to @event_handler methods in capabilities
           and action providers
        4. Accumulate context and check for immediate actions
        5. If no immediate action, invoke LLM planner with enriched context (via super().plan_step)

        NOTE: Event handlers provide context only, not transactions.
        Transaction management belongs in action executors that need it.
        Transactions are needed for states that are shared among multiple agents.
        Any action executor can use shared state (e.g., blackboard) to coordinate
        with other agents and needs to ensure the action or sequence of actions
        based on specific shared state versions are executed atomically.

        This enables modular, extensible event handling where:
        - Game capabilities handle game events
        - Memory capabilities observe and store events
        - Custom capabilities can participate in context enrichment

        Returns:
            Action to execute, or None
        """
        # 1. Get next event
        event = await self.get_next_event()

        # 2. Extract session_id and run_id from event and store in state for distributed traceability.
        # In distributed Ray systems, context variables don't cross node boundaries,
        # so we extract these from the event metadata and propagate them explicitly.
        # Store in state so execute_iteration() can set up context around dispatch()
        from ...sessions.context import session_id_context
        event_session_id = event.metadata.get("session_id") if event else None
        if event_session_id:
            state.custom["current_session_id"] = event_session_id
        # run_id is in the event value (set by AgentHandle.run/run_streamed)
        event_value = event.value if event else None
        if isinstance(event_value, dict) and event_value.get("run_id"):
            state.custom["current_run_id"] = event_value["run_id"]
            # Also update agent metadata so child agents inherit the run_id
            self.agent.metadata.run_id = event_value["run_id"]

        # 3. Broadcast to event handlers (within session_id context)
        immediate_actions = []
        accumulated_context: dict[str, dict[str, Any]] = {}

        if event is not None:
            with session_id_context(event_session_id):
                for handler in self._get_event_handlers():
                    try:
                        result = await handler(event, self._action_dispatcher.repl)

                        if result is None:
                            continue  # Event not relevant to this handler

                        # Accumulate context from all handlers
                        # This context is available to action executors via scope
                        context = result.context
                        if context:
                            if isinstance(result.context, BaseModel):
                                context = result.context.model_dump()
                            # TODO: Handle key conflicts (namespacing?)
                            # Event handlers can also store context in agent's
                            # working memory if needed
                            accumulated_context[result.context_key] = context

                        # Collect immediate actions (rule-based decision)
                        # (Don't return yet - process all handlers)
                        if result.immediate_action:
                            immediate_actions.append(result.immediate_action)

                        # Check for terminal state
                        if result.done:
                            state.custom["policy_complete"] = True
                            return None

                    except Exception as e:
                        logger.warning(
                            f"Event handler {handler.__name__} failed: {e}",
                            exc_info=True
                        )
                        continue

                # Store accumulated context in REPL for action executors
                # Action executors can access via repl.get("event_context")
                # Event handlers can also store context in agent's working memory
                if accumulated_context and self._action_dispatcher and self._action_dispatcher.repl:
                    self._action_dispatcher.repl.set("event_context", accumulated_context)
                    await self._store_event_context(
                        accumulated_context,
                        state,
                        namespace="event_context"
                    )

                # If any handler provided immediate action, return the first one
                # (others are ignored)
                # TODO: Could be made configurable (e.g., priority-based selection or
                #       let the LLM planner choose among them)
                if len(immediate_actions) == 1:
                    return immediate_actions[0]
                elif len(immediate_actions) > 1:
                    logger.info(
                        f"Multiple immediate actions from event handlers; passing them to the LLM planner to decide among them."
                    )
                    # TODO: Ensure that these actions are not identical.
                    # Pass all to LLM planner via context
                    await self._store_event_context(
                        {
                            "immediate_actions": [action.model_dump() for action in immediate_actions],
                            "description": "Multiple immediate actions from event handlers"
                        },
                        state,
                        namespace="conflicting_immediate_actions"
                    )
                    # Let LLM planner decide among them

        # 4. Invoke LLM planner with enriched context
        # Subclasses should override this or use CacheAwareActionPolicy
        # This allows LLM to plan actions from other capabilities based on the data
        # previously injected into the policy scope and working memory by past events or
        # previous reasoning steps.
        return None

    async def _store_event_context(
        self,
        context: dict[str, Any],
        state: ActionPolicyExecutionState,
        namespace: str = "event_context",
    ) -> None:
        """Store event context in working memory for persistence.
        
        This is an optional method that can be called to persist event
        context beyond the current iteration. Most use cases should
        just use scope.bindings["event_context"] which is set automatically
        in plan_step.

        Event handlers can also store context directly in any memory level
        (e.g., STM, LTM episodic/semantic/procedural) as needed.

        Args:
            context: Context dict from event handler
            state: Policy execution state
            namespace: Namespace prefix for storage key
        """
        try:
            working_memory = self.agent.get_working_memory()
            if working_memory:
                await working_memory.store(
                    key=f"{namespace}:action_policy_iteration:{state.iteration_count}",
                    value=context,
                    tags={namespace, "planning_context"},
                    ttl_seconds=3600,  # 1 hour - TODO: Make configurable
                )
        except Exception as e:
            logger.warning(f"Failed to store event context in memory: {e}", exc_info=True)


class CacheAwareActionPolicy(EventDrivenActionPolicy):

    """Action policy class for agents that use multi-step planning.

    This agent:
    - Creates plans using configurable strategies (MPC, top-down, bottom-up)
    - Executes plans incrementally via Agent.run_step
    - Handles replanning when needed
    - Coordinates with child agents event-driven (no polling)

    Attributes:
        action_history: History of actions (for debugging/logging)
    """
    def __init__(
        self,
        agent: Agent,
        planner: ActionPlanner,
        action_map: list[ActionGroup] | None = None,
        action_providers: list[Any] = [],
        io: ActionPolicyIO | None = None,
        replanning_policy: ReplanningPolicy | None = None,
    ):
        """Initialize planning agent.

        Args:
            agent: Agent using this policy
            planner: Action planner
            action_map: List of action groups
            action_providers: Additional action providers
            io: Policy I/O contract (inputs/outputs)
            replanning_policy: Policy that decides WHEN to replan and what
                strategy to use. Defaults to PeriodicReplanningPolicy.
        """
        super().__init__(
            agent=agent,
            action_map=action_map,
            action_providers=action_providers,
            io=io,
        )
        self.planner = planner  # TODO: Unify planner with planning strategy.
        self.plan_blackboard: PlanBlackboard | None = None
        self.replanning_policy = replanning_policy

        # Stream of consciousness: actions and planning
        self.action_history: list[Action] = [] # TODO: Currently unused
        self.current_plan: ActionPlan | None = None
        self.current_plan_id: str | None = None
        self.current_action_index: int | None = None

    def get_action_group_description(self) -> str:
        return (
            "Planning & Execution Control — manages the agent's plan lifecycle. "
            "Handles plan creation, replanning on failure or periodic triggers, "
            "and plan-level coordination with child agents via blackboard events."
        )

    async def initialize(self) -> None:
        """Initialize planning agent."""

        await super().initialize()

        # Get plan blackboard
        self.plan_blackboard = await self._get_plan_blackboard()

        if self.planner is None:
            # Get planning parameters from metadata
            planning_params = PlanningParameters(
                **self.agent.metadata.parameters.get("planning_params", {})  # FIXME: Get the planning parameters properly
            )

            # Get or create planning strategy
            planning_strategy: PlanningStrategyPolicy = get_default_planning_strategy(
                planning_params, agent=self
            )

            # Ensure strategy has agent reference
            if not planning_strategy.agent:
                planning_strategy.set_agent(self.agent)

            # Create default policies if not provided in metadata
            cache_policy = CacheAwarePlanningPolicy(
                agent=self,
                cache_capacity=planning_params.ideal_cache_size
            )
            await cache_policy.initialize()
            logger.info(
                f"Created default CacheAwarePlanningPolicy for agent {self.agent.agent_id}"
            )

            # Learning policy needs blackboard for ExecutionHistoryStore
            learning_policy = LearningPlanningPolicy(
                agent=self,
                blackboard=self.plan_blackboard.blackboard
            )
            await learning_policy.initialize()
            logger.info(
                f"Created default LearningPlanningPolicy for agent {self.agent.agent_id}"
            )

            coordination_policy = CoordinationPlanningPolicy(
                cache_capacity=planning_params.ideal_cache_size
            )
            await coordination_policy.initialize()
            logger.info(
                f"Created default CoordinationPlanningPolicy for agent {self.agent.agent_id}"
            )

            self.planner = CacheAwareActionPlanner(
                agent=self.agent,
                planning_strategy=planning_strategy,
                planning_params=planning_params,
                cache_policy=cache_policy,
                learning_policy=learning_policy,
                coordination_policy=coordination_policy,
            )

        # Create default replanning policy if none provided
        if self.replanning_policy is None:
            replan_every_n = 3
            replan_on_failure = True
            if hasattr(self.planner, 'planning_params'):
                replan_every_n = self.planner.planning_params.replan_every_n_steps
                replan_on_failure = self.planner.planning_params.replan_on_failure
            self.replanning_policy = PeriodicReplanningPolicy(
                replan_every_n_steps=replan_every_n,
                replan_on_failure=replan_on_failure,
            )

        # Get or create current plan
        self.current_plan = await self.plan_blackboard.get_plan(self.agent.agent_id)
        if not self.current_plan:
            await self._create_initial_plan()

        # Sync plan ID and action index
        if self.current_plan:
            self.current_plan_id = self.current_plan.plan_id
            self.current_action_index = self.current_plan.current_action_index

    @hookable
    async def _replan_horizon(self, decision: ReplanningDecision | None = None) -> ActionPlan:
        """Replan the remaining horizon of the current plan.

        This method is @hookable so memory capabilities can observe plan revisions.
        Returns the revised plan for hook-based capture.

        Args:
            decision: Optional replanning decision with triggers and strategy info.
                Passed through to the planner via planning_context.custom_data
                so the planning strategy can adapt its revision approach.
        """
        if not self.current_plan:
            raise RuntimeError("No current plan to replan.")

        planning_context = await self._get_planning_context(
            execution_context=self.current_plan.execution_context
        )

        # Pass replanning decision info to planner via custom_data
        if decision:
            planning_context.custom_data["revision_triggers"] = [
                t.value for t in decision.triggers
            ]
            planning_context.custom_data["revision_strategy"] = decision.strategy.value
            planning_context.custom_data["revision_reason"] = decision.reason

        # Generate plan via strategy
        self.current_plan = await self.planner.revise_plan(
            current_plan=self.current_plan,
            planning_context=planning_context,
            critique=None,  # TODO: Pass actual critique if available
        )

        triggers_str = (
            [t.value for t in decision.triggers] if decision else []
        )
        strategy_str = decision.strategy.value if decision else "default"
        logger.info(
            f"Replanned horizon for agent {self.agent.agent_id}, "
            f"triggers={triggers_str}, strategy={strategy_str}, "
            f"new plan has {len(self.current_plan.actions)} actions."
        )
        self.plan_blackboard.update_plan(self.current_plan)

        return self.current_plan

    async def _gather_planning_context(self) -> list[dict[str, Any]]:
        """Gather memories for planning context via AgentContextEngine.

        If the agent has an AgentContextEngine capability, this retrieves
        relevant memories from working memory and potentially STM/LTM.

        Returns:
            List of recalled memory dicts for the planning context
        """
        # Import here to avoid circular imports
        from ..memory import AgentContextEngine, MemoryQuery, MemoryScope

        ctx_engine: AgentContextEngine = self.agent.get_capability_by_type(AgentContextEngine)
        if ctx_engine is None:
            return []

        try:
            # Gather context from all available memory scopes
            entries = await ctx_engine.gather_context(
                query=MemoryQuery(max_results=50),
                ### scopes=[
                ###     MemoryScope.agent_working(self.agent.agent_id),
                ###     MemoryScope.agent_stm(self.agent.agent_id),
                ###     MemoryScope.agent_ltm_episodic(self.agent.agent_id),
                ### ],
            )

            # Convert BlackboardEntry objects to dicts for the planning context
            recalled_memories = []
            for entry in entries:
                recalled_memories.append({
                    "key": entry.key,
                    "value": entry.value,
                    "tags": list(entry.tags) if entry.tags else [],
                    "created_at": entry.created_at,
                    "relevance": entry.metadata.get("relevance", 1.0),
                })
            return recalled_memories
        except Exception as e:
            logger.warning(f"Failed to gather planning context: {e}")
            return []

    async def _get_memory_architecture_guidance(self) -> str | None:
        """Get memory architecture guidance for inclusion in planning prompts.

        If the agent has an AgentContextEngine, generates a description of the
        agent's memory system (levels, dataflow, available actions, capacity)
        that the LLM planner can use to reason about memory as a first-class
        cognitive resource.

        Returns:
            Guidance string, or None if no context engine is available.
        """
        from ..memory import AgentContextEngine

        ctx_engine: AgentContextEngine = self.agent.get_capability_by_type(AgentContextEngine)
        if ctx_engine is None:
            return None

        try:
            return await ctx_engine.get_memory_architecture_guidance()
        except Exception as e:
            logger.warning(f"Failed to get memory architecture guidance: {e}")
            return None

    @override
    async def execute_iteration(
        self,
        state: ActionPolicyExecutionState
    ) -> ActionPolicyIterationResult:
        """Execute one planning iteration (model-predictive control)."""
        # THIS METHOD IS NOT USED - iT IS SUPPOSED TO BE REPLACED BY plan_step below.
        # Review and remove.

        # TODO: Move all policy state to the state: ActionPolicyExecutionState parameter

        # Check if plan is already complete
        # Get latest plan from blackboard
        state.current_plan = await self.plan_blackboard.get_plan(self.agent.agent_id)

        if not state.current_plan:
            # No plan - create initial plan
            await self._create_initial_plan()
            return ActionPolicyIterationResult(success=True, policy_completed=False)

        if state.current_plan.current_action_index >= len(state.current_plan.actions):
            state.current_plan.status = PlanStatus.COMPLETED
            state.current_plan.completed_at = time.time()
            await self.plan_blackboard.update_plan(state.current_plan)
            return ActionPolicyIterationResult(
                success=True,
                policy_completed=True,
                requires_termination=True,
                action_executed=None,
                result=None,
                blocked_reason=None
            )

        # Check if replanning needed via policy (MPC)
        decision = await self.replanning_policy.evaluate_replanning_need(
            plan=state.current_plan,
            last_result=None,  # execute_iteration checks before dispatch
        )
        if decision.should_replan:
            await self._replan_horizon(decision)

        # Execute next action (REPL provides dataflow via ActionDispatcher)
        next_action = state.current_plan.actions[state.current_plan.current_action_index]
        result: ActionResult = await self.dispatch(next_action)

        # Update plan context
        state.current_plan.current_action_index += 1
        state.current_plan.execution_context.completed_action_ids.append(next_action.action_id)
        state.current_plan.execution_context.action_results[next_action.action_id] = result

        # Check if this was the last action
        plan_complete = state.current_plan.current_action_index >= len(state.current_plan.actions)
        if plan_complete:
            state.current_plan.status = PlanStatus.COMPLETED
            state.current_plan.completed_at = time.time()

        # Always persist plan state (index, results) — even on failure.
        # Without this, the re-fetch from blackboard resets the index.
        await self.plan_blackboard.update_plan(state.current_plan)
        self.current_plan_id = state.current_plan.plan_id
        self.current_action_index = state.current_plan.current_action_index

        # Handle iteration result
        if not result.success:
            logger.warning(
                f"Agent {self.agent.agent_id} failed to execute plan "
                f"{state.current_plan.plan_id} iteration"
            )
            # Check if blocked (may need to wait for dependencies)
            if result.blocked_reason:
                logger.info(f"Agent {self.agent.agent_id} blocked: {result.blocked_reason}")
                # Suspend agent until dependencies resolved
                await self.agent.suspend(reason=f"Blocked: {result.blocked_reason}")
                return ActionPolicyIterationResult(
                    success=False,
                    policy_completed=False,
                    blocked_reason=result.blocked_reason
                )
            # Otherwise just log failure and continue
            return ActionPolicyIterationResult(
                success=False,
                policy_completed=False,
                action_executed=next_action,
                result=result,
            )

        # Check if plan is complete
        if state.current_plan.is_complete():
            # Learn from execution outcome
            await self.planner.learn_from_plan_execution(state.current_plan)

            logger.info(f"Agent {self.agent.agent_id} completed plan {state.current_plan.plan_id}")

        return ActionPolicyIterationResult(
            success=result.success,
            policy_completed=plan_complete,
            requires_termination=plan_complete,
            action_executed=next_action,
            result=result,
            blocked_reason=result.blocked_reason,
        )

    async def plan_step(
        self,
        state: ActionPolicyExecutionState
    ) -> Action | None:
        """Produce next action using model-predictive control planning.

        Flow:
        1. Process events via event handlers (super().plan_step)
        2. Process result of previous action (if any)
        3. Get/create plan from blackboard
        4. Check if plan complete
        5. Check if replanning needed (MPC)
        6. Return next action
        """
        # Process events first (calls event handlers, enriches context)
        logger.warning(f"      📋 PLAN_STEP: checking events  agent={self.agent.agent_id}")
        event_action = await super().plan_step(state)
        if event_action:
            logger.warning(f"      📋 PLAN_STEP: event produced immediate action → {event_action}")
            return event_action
        if state.custom.get("policy_complete"):
            logger.warning(f"      📋 PLAN_STEP: policy_complete set by event handler")
            return None

        # Get plan from blackboard
        logger.warning(f"      📋 PLAN_STEP: fetching plan from blackboard  agent_id={self.agent.agent_id}")
        state.current_plan = await self.plan_blackboard.get_plan(self.agent.agent_id)
        if state.current_plan:
            logger.warning(
                f"      📋 PLAN_STEP: got plan → {len(state.current_plan.actions)} actions, "
                f"idx={state.current_plan.current_action_index}, status={state.current_plan.status}"
            )
        else:
            logger.warning(f"      📋 PLAN_STEP: got plan → None")

        if not state.current_plan:
            # No plan - create initial plan
            logger.warning(
                f"\n"
                f"      ╔════════════════════════════════════════════╗\n"
                f"      ║  🧠 CREATING INITIAL PLAN (LLM call)      ║\n"
                f"      ║  agent={self.agent.agent_id:<38}║\n"
                f"      ╚════════════════════════════════════════════╝"
            )
            await self._create_initial_plan()
            state.current_plan = self.current_plan
            logger.warning(
                f"      🧠 PLAN CREATED: id={self.current_plan.plan_id if self.current_plan else 'NONE!'} "
                f"actions={len(self.current_plan.actions) if self.current_plan else 0}"
            )
            return None  # Plan created, will get action on next call

        # Process result of previous action (from previous iteration)
        last_action_id = state.custom.get("last_action_id")
        repl = self._action_dispatcher.repl if self._action_dispatcher else None
        last_result: ActionResult | None = None  # Hoisted for replanning policy
        if last_action_id and repl:
            last_result = repl.get_result(last_action_id)
            if last_result:
                # Update plan execution context
                state.current_plan.execution_context.completed_action_ids.append(last_action_id)
                state.current_plan.execution_context.action_results[last_action_id] = last_result

                # Handle failure
                if not last_result.success:
                    logger.warning(
                        f"Agent {self.agent.agent_id} failed to execute action {last_action_id}"
                    )
                    # Check if blocked
                    if last_result.blocked_reason:
                        logger.info(f"Agent {self.agent.agent_id} blocked: {last_result.blocked_reason}")
                        await self.plan_blackboard.update_plan(state.current_plan)
                        await self.agent.suspend(reason=f"Blocked: {last_result.blocked_reason}")
                        # Return None but don't set complete - we're suspended
                        return None

                # Persist plan state (results, completed_action_ids).
                # NOTE: current_action_index is persisted AFTER increment below.
                await self.plan_blackboard.update_plan(state.current_plan)

                # Sync local state
                self.current_plan_id = state.current_plan.plan_id
                self.current_action_index = state.current_plan.current_action_index

            state.custom["last_action_id"] = None

        # Check if plan complete
        logger.warning(
            f"      📋 PLAN_STEP: idx={state.current_plan.current_action_index} / "
            f"{len(state.current_plan.actions)} actions"
        )
        if state.current_plan.current_action_index >= len(state.current_plan.actions):
            logger.warning(f"      📋 PLAN_STEP: ★★★ PLAN COMPLETE ★★★")
            state.current_plan.status = PlanStatus.COMPLETED
            state.current_plan.completed_at = time.time()
            await self.plan_blackboard.update_plan(state.current_plan)

            # Learn from execution
            await self.planner.learn_from_plan_execution(state.current_plan)

            logger.info(f"Agent {self.agent.agent_id} completed plan {state.current_plan.plan_id}")
            state.custom["policy_complete"] = True
            return None

        # Check if replanning needed via policy (MPC)
        logger.warning(f"      📋 PLAN_STEP: evaluating replanning need")
        decision = await self.replanning_policy.evaluate_replanning_need(
            plan=state.current_plan,
            last_result=last_result,
        )
        if decision.should_replan:
            logger.warning(f"      📋 PLAN_STEP: !!! REPLANNING triggered: {decision.reason}")
            await self._replan_horizon(decision)

        # Get next action and advance index
        next_action = state.current_plan.actions[state.current_plan.current_action_index]
        state.current_plan.current_action_index += 1
        state.custom["last_action_id"] = next_action.action_id

        # Persist the incremented index so re-fetch from blackboard sees it
        await self.plan_blackboard.update_plan(state.current_plan)
        self.current_plan_id = state.current_plan.plan_id
        self.current_action_index = state.current_plan.current_action_index

        logger.warning(
            f"      📋 PLAN_STEP: returning action → id={next_action.action_id} type={next_action.action_type}"
        )
        return next_action

    async def _get_plan_blackboard(self) -> PlanBlackboard:
        """Get or create plan blackboard with access control."""
        if not hasattr(self, "_plan_blackboard_cached"):
            # Create access policy with agent hierarchy
            agent_hierarchy = await self.agent.get_agent_hierarchy()
            team_structure = await self.agent.get_team_structure()

            access_policy = HierarchicalAccessPolicy(
                agent_hierarchy=agent_hierarchy,
                team_structure=team_structure,
            )

            planning_scope_id = f"tenant:{self.agent.tenant_id}:agent:{self.agent.agent_id}:planning_scope"
            self._plan_blackboard_cached = PlanBlackboard(
                plan_access_policy=access_policy,
                scope_id=planning_scope_id,
            )
            await self._plan_blackboard_cached.initialize()
        return self._plan_blackboard_cached

    async def _build_system_prompt(self) -> str:
        """Build stable agent identity prompt.

        Uses AgentSelfConcept from ConsciousnessCapability when available,
        falling back to agent metadata, class docstring, and capabilities.
        """
        from ..capabilities.consciousness import ConsciousnessCapability

        agent = self.agent
        parts: list[str] = []

        # Try to get self-concept from ConsciousnessCapability
        consciousness: ConsciousnessCapability | None = agent.get_capability_by_type(ConsciousnessCapability)
        self_concept = await consciousness.get_self_concept() if consciousness else None

        if self_concept:
            # Build from self-concept
            identity = f"You are {self_concept.name}"
            if self_concept.role:
                identity += f", {self_concept.role}"
            parts.append(identity)

            if self_concept.description:
                parts.append(self_concept.description)

            if self_concept.identity:
                parts.append(self_concept.identity)

            if self_concept.goals:
                parts.append("Goals:\n" + "\n".join(f"- {g}" for g in self_concept.goals))

            if self_concept.constraints:
                parts.append("Constraints:\n" + "\n".join(f"- {c}" for c in self_concept.constraints))

            if self_concept.capabilities:
                parts.append(f"Capabilities: {', '.join(self_concept.capabilities)}")

            if self_concept.limitations:
                parts.append("Limitations:\n" + "\n".join(f"- {l}" for l in self_concept.limitations))

            if self_concept.world_model:
                parts.append(f"World model: {self_concept.world_model}")

            if self_concept.frame_of_mind:
                parts.append(f"Frame of mind: {self_concept.frame_of_mind}")
        else:
            # Fallback: build from agent metadata and class info
            identity = f"You are {agent.__class__.__name__}"
            if agent.metadata.role:
                identity += f" (role: {agent.metadata.role})"
            identity += f", a {agent.agent_type} agent."
            parts.append(identity)

            doc = agent.__class__.__doc__
            if doc:
                parts.append(doc.strip().split('\n\n')[0].strip())

            cap_names = agent.get_capability_names()
            if cap_names:
                parts.append(f"Your capabilities: {', '.join(cap_names)}")

        # Task parameters (always from metadata — these are per-run, not part of self-concept)
        params = agent.metadata.parameters
        if params:
            param_lines = [f"- {k}: {v}" for k, v in params.items()
                           if not k.startswith("_") and k != "planning_params"]
            if param_lines:
                parts.append("Task parameters:\n" + "\n".join(param_lines))

        return "\n\n".join(parts)

    def _get_constraints(self) -> dict[str, Any]:
        """Extract execution constraints from agent metadata."""
        constraints: dict[str, Any] = {}
        meta = self.agent.metadata
        if meta.max_iterations:
            constraints["max_iterations"] = meta.max_iterations
        params = meta.parameters
        if "max_agents" in params:
            constraints["max_parallel_workers"] = params["max_agents"]
        if "quality_threshold" in params:
            constraints["quality_threshold"] = params["quality_threshold"]
        return constraints

    async def _get_planning_context(self, execution_context: PlanExecutionContext) -> PlanningContext:
        """Build the planning context for the current planning step (initial or replanning).
        """
        goals=self.agent.metadata.goals or []

        # Recall memories for planning context
        recalled_memories = await self._gather_planning_context()

        # Get memory architecture guidance for LLM reasoning
        memory_guidance = await self._get_memory_architecture_guidance()
        custom_data: dict[str, Any] = {}
        if memory_guidance:
            custom_data["memory_architecture_guidance"] = memory_guidance

        # Create new planning context based on current state
        return PlanningContext(
            system_prompt=await self._build_system_prompt(),
            execution_context=execution_context,
            page_ids=list(self.agent.bound_pages) if hasattr(self.agent, 'bound_pages') else [],
            goals=goals,
            constraints=self._get_constraints(),
            action_descriptions=await self.get_action_descriptions(),
            action_group_summaries=await self.get_action_group_summaries(),
            recalled_memories=recalled_memories,
            custom_data=custom_data,
            # parent_plan_id=self.current_plan.parent_plan_id,
        )

    @hookable
    async def _create_initial_plan(self) -> ActionPlan:
        """Create initial plan.

        This method is @hookable so memory capabilities can observe plan creation.
        Returns the created plan for hook-based capture.
        """

        logger.warning(f"        🧠 _create_initial_plan: building planning context...")
        planning_context = await self._get_planning_context(
            execution_context=PlanExecutionContext()
        )
        logger.warning(
            f"        🧠 _create_initial_plan: context ready — "
            f"goals={planning_context.goals}, "
            f"pages={len(planning_context.page_ids)}, "
            f"actions={len(planning_context.action_descriptions)}"
        )

        logger.warning(
            f"\n"
            f"        ╔════════════════════════════════════════╗\n"
            f"        ║  🔮 CALLING planner.create_plan()      ║\n"
            f"        ║  (THIS IS THE LLM INFERENCE CALL)      ║\n"
            f"        ╚════════════════════════════════════════╝"
        )
        plan: ActionPlan = await self.planner.create_plan(planning_context)
        logger.warning(
            f"        🔮 planner.create_plan() returned: "
            f"plan_id={plan.plan_id}, actions={len(plan.actions)}, status={plan.status}"
        )
        plan.agent_id = self.agent.agent_id

        approved, msg = await self.plan_blackboard.propose_plan(plan, self.agent.agent_id)
        if approved:
            self.current_plan = plan
            # Sync plan ID and action index
            self.current_plan_id = plan.plan_id
            self.current_action_index = plan.current_action_index
            logger.info(f"Plan approved for agent {self.agent.agent_id}: {plan.plan_id}")
        else:
            # TODO: Handle plan rejection (e.g., modify and resubmit)
            # TODO: Handle pending approval properly
            logger.info(f"Plan pending approval for agent {self.agent.agent_id}: {msg}")
            self.current_plan = plan
            # Sync plan ID and action index
            self.current_plan_id = plan.plan_id
            self.current_action_index = plan.current_action_index

        return plan

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize policy-specific state.

        Args:
            state: AgentSuspensionState with all agent state serialized
        """
        state = await super().serialize_suspension_state(state)
        # Add ActionPolicy-specific state
        # Execution state
        if self.current_plan:
            state.plan_id = self.current_plan.plan_id
            state.current_action_index = self.current_plan.current_action_index
        else:
            state.plan_id = self.current_plan_id
            state.current_action_index = self.current_action_index
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        """Restore policy-specific state from suspension.

        Args:
            state: AgentSuspensionState to restore from
        """
        self.current_action_index = state.current_action_index
        self.current_plan_id = state.plan_id

    async def get_status_snapshot(self) -> dict[str, Any]:
        """Get snapshot of current status for debugging/monitoring.

        Returns:
            Status dictionary
        """
        status = {
            "current_plan_id": self.current_plan_id,
            "current_action_index": self.current_action_index,
            "total_actions_executed": len(self.action_history),
        }
        if self.current_plan:
            status.update({
                "plan_status": str(self.current_plan.status),
                "plan_actions_total": len(self.current_plan.actions),
                "plan_actions_completed": self.current_plan.current_action_index,
            })
        return status





async def create_default_action_policy(
    agent: Agent,
    action_map: list[ActionGroup] | None = None,
    action_providers: list[Any] = [],
    io: ActionPolicyIO | None = None,
    max_iterations: int = 50,
    quality_threshold: float = 0.9,
    planning_horizon: int = 5,
    ideal_cache_size: int = 10,
) -> CacheAwareActionPolicy:
    """Create sophisticated action policy with cache-awareness and learning.

    Returns:
        CacheAwareActionPolicy
    """
    from ..planning.planner import create_cache_aware_planner

    planner = await create_cache_aware_planner(
        agent=agent,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        planning_horizon=planning_horizon,
        ideal_cache_size=ideal_cache_size,
    )

    action_policy = CacheAwareActionPolicy(
        agent=agent,
        planner=planner,
        action_map=action_map,
        action_providers=action_providers,
        io=io
    )
    await action_policy.initialize()

    return action_policy

