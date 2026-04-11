
from __future__ import annotations

import inspect
import json
import traceback as traceback_mod
import uuid
from abc import ABC, abstractmethod
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
    ActionStatus,
    PolicyREPL,
    Ref,
    ActionSharedDataDependency,
)
from ...base import Agent, ActionPolicy
from ..hooks import hookable
from ...blackboard.backend import ConcurrentModificationError
from .repl import PolicyPythonREPL, REPLCapability, get_repl_guidance

logger = setup_logger(__name__)

# NOTE: Class-based ActionExecutors from executors.py are deprecated.
# Use @action_executor decorated methods on AgentCapability classes instead.
# See executors.py module docstring for migration details.

#
# NOTE: With ambient transactions, action executors do not need access to transaction
# handles. The dispatcher only needs to validate dependency versions and execute the
# action within the dependency transactor contexts.


class SchemaDetail(str, Enum):
    """Controls how action parameter schemas are rendered in planning prompts.

    COMPACT:   Type names only (e.g., ``run_context?: RunContext``).
               Smallest token footprint. Works for simple parameters.
    SELECTIVE: Compact for simple types; expands nested BaseModel fields
               inline. Best default — adds detail only where the LLM
               would otherwise guess field names.
    COMPACT_SCHEMA: Full ``model_json_schema()`` but stripped of boilerplate
               (``title``, outer ``type``) — only ``properties``, ``required``,
               and ``$defs``. ~60-70 % smaller than FULL.
    FULL:      Raw ``model_json_schema()`` output. Most precise but verbose.
    """
    COMPACT = "compact"
    SELECTIVE = "selective"
    COMPACT_SCHEMA = "compact_schema"
    FULL = "full"


def _compact_signature(schema_cls: type[BaseModel]) -> str:
    """Build ``name: Type, name?: Type`` compact signature."""
    parts = []
    for name, field_info in schema_cls.model_fields.items():
        ann = field_info.annotation
        type_name = getattr(ann, '__name__', str(ann))
        if field_info.is_required():
            parts.append(f"{name}: {type_name}")
        else:
            parts.append(f"{name}?: {type_name}")
    return ", ".join(parts)


def _selective_signature(schema_cls: type[BaseModel]) -> str:
    """Compact signature, but expand nested BaseModel fields inline."""
    parts = []
    for name, field_info in schema_cls.model_fields.items():
        ann = field_info.annotation
        type_name = getattr(ann, '__name__', str(ann))
        opt = "?" if not field_info.is_required() else ""

        # Check if annotation (or a Union member) is a BaseModel
        nested_cls = None
        for arg in get_args(ann) or [ann]:
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                nested_cls = arg
                break

        if nested_cls is not None:
            # Expand nested model fields inline
            inner = []
            for n, f in nested_cls.model_fields.items():
                f_ann = f.annotation
                f_type = getattr(f_ann, '__name__', str(f_ann))
                f_opt = "?" if not f.is_required() else ""
                inner.append(f"{n}{f_opt}: {f_type}")
            parts.append(f"{name}{opt}: {type_name} {{{', '.join(inner)}}}")
        else:
            parts.append(f"{name}{opt}: {type_name}")
    return ", ".join(parts)


def _compact_json_schema(schema_cls: type[BaseModel]) -> dict:
    """Full JSON schema with boilerplate stripped."""
    schema = schema_cls.model_json_schema()
    # Remove top-level noise — keep only what the LLM needs
    schema.pop("title", None)
    schema.pop("type", None)  # always "object"
    # Strip titles from nested $defs
    for defn in schema.get("$defs", {}).values():
        defn.pop("title", None)
    return schema


def _format_schema(schema_cls: type[BaseModel], detail: SchemaDetail) -> str:
    """Render parameter schema at the requested detail level."""
    if detail == SchemaDetail.FULL:
        return f"\n  Parameters (JSON Schema): {json.dumps(schema_cls.model_json_schema())}"
    elif detail == SchemaDetail.COMPACT_SCHEMA:
        return f"\n  Parameters (JSON Schema): {json.dumps(_compact_json_schema(schema_cls))}"
    elif detail == SchemaDetail.SELECTIVE:
        return f"\n  Parameters: {_selective_signature(schema_cls)}"
    else:  # COMPACT
        return f"\n  Parameters: {_compact_signature(schema_cls)}"


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

    async def get_action_description(self, schema_detail: SchemaDetail = SchemaDetail.SELECTIVE) -> str:
        """Get human-readable description of action or action policy to be
        used in LLM-based action selection.

        Args:
            schema_detail: How to render parameter schemas. See ``SchemaDetail``.

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
                    if isinstance(ret, self.output_schema):
                        ret = ret.model_dump()
                    elif isinstance(ret, dict):
                        validated = self.output_schema(**ret)
                        ret = validated.model_dump()
                    elif isinstance(ret, BaseModel):
                        ret = ret.model_dump()
                    # else: leave ret as-is (primitive types)
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

    @override
    async def get_action_description(self, schema_detail: SchemaDetail = SchemaDetail.SELECTIVE) -> str:
        """Get human-readable description from planning_summary or the wrapped method's docstring.

        Args:
            schema_detail: How to render parameter schemas. See ``SchemaDetail``.
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

        if self.input_schema:
            desc += _format_schema(self.input_schema, schema_detail)
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
                    if isinstance(ret, self.output_schema):
                        ret = ret.model_dump()
                    elif isinstance(ret, dict):
                        validated = self.output_schema(**ret)
                        ret = validated.model_dump()
                    elif isinstance(ret, BaseModel):
                        ret = ret.model_dump()
                    # else: leave ret as-is (primitive types)
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

    @override
    async def get_action_description(self, schema_detail: SchemaDetail = SchemaDetail.SELECTIVE) -> str:
        """Get description from planning_summary or function docstring.

        Args:
            schema_detail: How to render parameter schemas. See ``SchemaDetail``.
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

        if self.input_schema:
            desc += _format_schema(self.input_schema, schema_detail)
        return desc


def _rebuild_with_forward_refs(model: type[BaseModel], func: Callable) -> None:
    """Rebuild a dynamically-created Pydantic model, resolving forward refs.

    Forward references (string annotations like ``"Action"``) that are guarded
    behind ``TYPE_CHECKING`` aren't available at runtime in the module where
    ``func`` was defined.  We build a namespace from the func's module *plus*
    the colony.agents.models module (which exports Action, ActionResult, etc.)
    so that ``model_rebuild`` can resolve them.
    """
    import sys

    ns: dict[str, Any] = {}
    # func's own module namespace (covers most types)
    func_module = getattr(func, "__module__", None)
    if func_module and func_module in sys.modules:
        ns.update(vars(sys.modules[func_module]))
    # colony.agents.models — exports Action, ActionResult, etc. used in forward refs
    models_mod = sys.modules.get("polymathera.colony.agents.models")
    if models_mod:
        ns.update(vars(models_mod))
    try:
        model.model_rebuild(_types_namespace=ns)
    except Exception:
        pass  # Best-effort — some forward refs may still be unresolvable


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
    has_var_keyword = False
    for name, param in sig.parameters.items():
        # Skip 'self' parameter
        if name == 'self':
            continue
        # Skip *args and **kwargs — these can't be represented as fixed schema fields.
        # When **kwargs is present, the generated model is configured with extra="allow"
        # so that additional fields pass validation and flow through to the function.
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                has_var_keyword = True
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
        if has_var_keyword:
            # Function accepts **kwargs — allow extra fields so they flow through
            model.model_config["extra"] = "allow"
        # Resolve forward references that live under TYPE_CHECKING in the
        # module where `func` was defined.  We merge that module's namespace
        # with the models module (which contains Action, ActionResult, etc.)
        # so that Pydantic can rebuild the schema fully.
        _rebuild_with_forward_refs(model, func)
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
    has_var_keyword = False
    params = list(sig.parameters.items())

    # Skip first parameter (the Agent param that gets auto-injected)
    for name, param in params[1:]:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                has_var_keyword = True
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

    try:
        model = create_model(f"{func.__name__}_Input", **fields)
        if has_var_keyword:
            model.model_config["extra"] = "allow"
        _rebuild_with_forward_refs(model, func)
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



def pydantic_model_to_str(model: BaseModel | None, trunc: int = 1000) -> tuple[str, str]:
    """Helper to convert a Pydantic model to a pretty string for logging."""
    if model is None:
        return "None", "full"
    model_str = model.model_dump_json(indent=3)
    return (model_str[:trunc], "truncated") if len(model_str) > trunc else (model_str, "full" )


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
            # Prefer capability_key (semantic, stable, LLM-friendly) over random hash.
            # Non-capability objects (e.g., Agent, ActionPolicy) fall back to random hash.
            cap_key = getattr(obj, '_capability_key', None) or getattr(obj, 'capability_key', None)
            obj._action_dispatch_key = cap_key if isinstance(cap_key, str) else uuid.uuid4().hex[:8]

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
        schema_detail: SchemaDetail = SchemaDetail.SELECTIVE,
        include_tags: frozenset[str] | None = None,
        exclude_tags: frozenset[str] | None = None,
    ) -> list[ActionGroupDescription]:
        """Get human-readable description of plannable actions for LLM-based
        action selection.

        Only includes actions visible to the planner (excludes actions marked
        with `exclude_from_planning=True`).

        If REPL is available, includes REPL guidance and variable summary.

        Args:
            selected_groups: If provided, only return descriptions for groups
                whose group_key is in this list. If None, return all groups.
            schema_detail: How to render parameter schemas. See ``SchemaDetail``.
            include_tags: If provided, only include groups that have at least
                one of these tags. Used for mode filtering (e.g., ``{"planning"}``
                for planning mode, ``{"execution"}`` or no tags for execution mode).
            exclude_tags: If provided, exclude groups that have any of these tags.

        Returns:
            List of ActionGroupDescription with full action descriptions.
        """
        descriptions: list[ActionGroupDescription] = []
        for group in self.action_map:
            if selected_groups is not None and group.group_key not in selected_groups:
                continue

            # Tag-based mode filtering
            if include_tags is not None and not (group.tags & include_tags):
                continue
            if exclude_tags is not None and (group.tags & exclude_tags):
                continue
            action_descs: dict[str, str] = {}
            for action_key, executor in group.executors.items():
                if getattr(executor, 'exclude_from_planning', False):
                    continue
                try:
                    desc = await executor.get_action_description(schema_detail=schema_detail)
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
            # Capture traceback for observability — flows through to
            # TracingCapability via ActionResult.model_dump() → output_summary.
            tb_lines = traceback_mod.format_exception(type(e), e, e.__traceback__)
            user_frames = [f for f in tb_lines if "site-packages" not in f]
            result = ActionResult(
                success=False,
                completed=True,
                error=f"{type(e).__name__}: {e}",
                metadata={"traceback": "".join(user_frames[-5:])},
            )
            action.status = ActionStatus.FAILED
            action.result = result
            return result

    def _get_executor_for_action(self, action: Action) -> ActionExecutor | None:
        """Get the executor for a given action by exact key match,
        followed by fuzzy fallback.

        Tries exact match first.  If that fails, attempts suffix matching
        (the most common LLM error is emitting only the method-name suffix
        instead of the full ``ClassName.hash.method_name`` compound key).
        """
        action_key = str(action.action_type)

        # 1. Exact match
        for group in self.action_map:
            if action_key in group.executors:
                return group.executors[action_key]

        # 2. Suffix match — handles the common LLM truncation error
        resolved_key, executor = self._resolve_action_key_fuzzy(action_key)
        if executor is not None:
            logger.warning(
                f"Resolved truncated action key: '{action_key}' → '{resolved_key}'"
            )
            # Patch the action so downstream code sees the correct key
            action.action_type = resolved_key
            return executor

        return None

    def _resolve_action_key_fuzzy(
        self, raw_key: str
    ) -> tuple[str | None, ActionExecutor | None]:
        """Attempt to resolve an invalid action key via suffix or similarity matching.

        Returns (resolved_key, executor) or (None, None) if no match.
        """
        all_executors: dict[str, ActionExecutor] = {}
        for group in self.action_map:
            all_executors.update(group.executors)

        if not all_executors:
            return None, None

        # Suffix match: raw_key might be just the method name
        suffix_matches = [
            (k, e) for k, e in all_executors.items()
            if k.endswith(f".{raw_key}")
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

        # Prefix-stripped match: raw_key has a spurious prefix
        # e.g., "working.WorkingMemoryCapability.abc123.store" → try removing
        # tokens from the front until we get a match.
        parts = raw_key.split(".")
        for start in range(1, len(parts)):
            candidate = ".".join(parts[start:])
            if candidate in all_executors:
                return candidate, all_executors[candidate]

        return None, None

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
            result: ActionResult = await executor.execute(action, resolved_params)
        else:
            # Fallback for other executor types
            result: ActionResult = await executor.execute(action)

        result_str, trunc = pydantic_model_to_str(result)
        logger.info(f"______ Action execution result: {result_str} ({trunc})")  # Log a truncated version of the result for readability

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

        # Pick up per-run() call trace if the code generation policy stored it
        metadata: dict[str, Any] = {}
        run_call_trace = self.repl.namespace.get("_run_call_trace")
        if run_call_trace:
            metadata["run_call_trace"] = list(run_call_trace)

        result = ActionResult(
            success=result_dict["success"],
            output=result_dict,
            error=result_dict.get("error"),
            metadata=metadata,
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
            return self._navigate(capability, remaining[1:])  # TODO: _navigate does not handle capabilities. What the fuck?

        elif root == "global":
            # $global.blackboard_key
            key = ".".join(remaining) if remaining else ""
            if not key:
                raise ValueError(f"Blackboard reference missing key: {ref.path}")
            board = await self.agent.get_agent_level_blackboard()  # TODO: This is the agent's blackboard. Is this correct, or should there be a separate "global" blackboard for cross-agent shared state?
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
                    description="Declared output variable",  # TODO: Allow writes to have descriptions
                    created_by=f"action_type{action.action_type}:{action.action_id}",
                )
            elif len(writes) == 1:
                # Single write declaration - use entire output
                await self.repl.set(
                    name=var_name,
                    value=output,
                    description="Declared output variable",  # TODO: Allow writes to have descriptions
                    created_by=f"action_type{action.action_type}:{action.action_id}",
                )

