"""Blueprint[T] — Deferred binding for agents, capabilities, and policies.

This module implements the Blueprint pattern for Colony's agent system.
Blueprint[T] captures a class and its constructor kwargs into a serializable
object that can be transported across Ray node boundaries via cloudpickle
and instantiated later on the target node.

Same pattern as Akka's Props and Erlang/OTP's child_spec.
See capability_bind_proposal.md for full design rationale.
"""

from __future__ import annotations

from typing import TypeVar, Generic, Any, ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Agent, AgentCapability, ActionPolicy
    from ..cluster import LLMClientRequirements

T = TypeVar("T")


# ─── Error types ───────────────────────────────────────────────────────────────

class BlueprintValidationError(Exception):
    """Bind-time validation failure (invalid field name, wrong type)."""


class BlueprintSerializationError(BlueprintValidationError):
    """Argument not serializable via cloudpickle."""


class BlueprintRemoteError(Exception):
    """Remote instantiation failure with remote traceback."""

    def __init__(self, message: str, remote_traceback: str | None = None):
        super().__init__(message)
        self.remote_traceback = remote_traceback


# ─── Blueprint[T] ─────────────────────────────────────────────────────────────

class Blueprint(Generic[T]):
    """Deferred constructor: captures cls + keyword args for later instantiation.

    Serialized across Ray boundaries via cloudpickle (same as BoundDeployment).
    Subclasses add domain-specific fields (key, action filters, etc.).

    Attribute access: ``blueprint.metadata`` delegates to ``self.kwargs["metadata"]``,
    so a Blueprint[T] feels like a deferred T for read access.

    Validates at bind time:
    - For pydantic BaseModel subclasses: required fields present in kwargs
    - Serializability via ray.cloudpickle.dumps

    Invariants:
    - self.cls is the actual class (not a string path)
    - self.kwargs contains ONLY constructor kwargs for cls.__init__
    - No positional args stored (prefix_args injected at local_instance time)
    """

    __slots__ = ("cls", "kwargs")

    def __init__(self, cls: type[T], kwargs: dict[str, Any] | None = None):
        self.cls = cls
        self.kwargs = kwargs or {}
        self._validate_required_fields()

    def _deferred_fields(self) -> frozenset[str]:
        """Fields that are required on T but filled in later (e.g., by the spawn pipeline).
        Subclasses override to declare deferred fields."""
        return frozenset()

    def _validate_required_fields(self) -> None:
        """If T is a pydantic BaseModel, check that all required fields are in kwargs.
        Skips fields returned by _deferred_fields()."""
        model_fields = getattr(self.cls, "model_fields", None)
        if model_fields is None:
            return
        from pydantic.fields import PydanticUndefined
        skip = self._deferred_fields()
        missing = []
        for name, field_info in model_fields.items():
            if name in skip or name in self.kwargs:
                continue
            if field_info.default is PydanticUndefined and field_info.default_factory is None:
                missing.append(name)
        if missing:
            raise BlueprintValidationError(
                f"Required fields missing from {self.cls.__name__}.bind(): {missing}"
            )

    def __getattr__(self, name: str) -> Any:
        # Only called when normal slot/descriptor lookup fails.
        # Delegates to kwargs so blueprint.metadata works like agent.metadata.
        # Must use object.__getattribute__ to avoid recursion when self.kwargs
        # itself isn't set yet (e.g. during unpickling).
        try:
            kwargs = object.__getattribute__(self, "kwargs")
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            )
        try:
            return kwargs[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' for {self.cls.__name__} has no attribute '{name}'"
            )

    def validate_serializable(self) -> None:
        """Validate all kwargs are cloudpickle-serializable. Called at bind time.
        Recursively validates nested Blueprint objects."""
        import ray.cloudpickle as cloudpickle

        for key, value in self.kwargs.items():
            if isinstance(value, Blueprint):
                value.validate_serializable()
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Blueprint):
                        item.validate_serializable()
                    else:
                        try:
                            cloudpickle.dumps(item)
                        except Exception as e:
                            raise BlueprintSerializationError(
                                f"List item in '{key}' for {self.cls.__name__} "
                                f"not serializable: {e}"
                            ) from e
            else:
                try:
                    cloudpickle.dumps(value)
                except Exception as e:
                    raise BlueprintSerializationError(
                        f"Argument '{key}' (type={type(value).__name__}) for "
                        f"{self.cls.__name__} not serializable: {e}"
                    ) from e

    def local_instance(self, *prefix_args: Any, **extra_kwargs: Any) -> T:
        """Construct T locally. Called on the target deployment node.

        Does NOT auto-resolve nested Blueprints in kwargs — domain code
        (e.g., Agent._create_action_policy) handles that explicitly.

        Args:
            *prefix_args: Prepended positional args (e.g., agent for capabilities).
            **extra_kwargs: Merged with bound kwargs (overrides on conflict).
        """
        merged = {**self.kwargs, **extra_kwargs}
        return self.cls(*prefix_args, **merged)

    def __repr__(self) -> str:
        parts = [f"{k}={v!r}" for k, v in self.kwargs.items()]
        if parts:
            return f"Blueprint({self.cls.__name__}, {', '.join(parts)})"
        return f"Blueprint({self.cls.__name__})"


# ─── AgentCapabilityBlueprint ─────────────────────────────────────────────────

class AgentCapabilityBlueprint(Blueprint["AgentCapability"]):
    """Blueprint for an AgentCapability.
    A capability class bound with constructor arguments.

    kwargs: constructor args (scope_id, blackboard, ttl_seconds, etc.)
    Composition fields: how this capability is added to the agent (via add_capability).
    Set via .with_composition() — NOT passed to the capability constructor.
    """

    __slots__ = ("key", "include_actions", "exclude_actions", "events_only")

    def __init__(
        self,
        cls: type[AgentCapability],
        kwargs: dict[str, Any] | None = None,
        *,
        key: str | None = None,
        include_actions: list[str] | None = None,
        exclude_actions: list[str] | None = None,
        events_only: bool = False,
    ):
        super().__init__(cls, kwargs)
        self.key = key or cls.__name__
        self.include_actions = include_actions
        self.exclude_actions = exclude_actions
        self.events_only = events_only

    # TODO: Move this method to the base Blueprint class and make it more generic (e.g., with_metadata) for other use cases?
    def with_composition(
        self,
        *,
        key: str | None = None,
        include_actions: list[str] | None = None,
        exclude_actions: list[str] | None = None,
        events_only: bool | None = None,
    ) -> AgentCapabilityBlueprint:
        """Return a copy with updated composition metadata. Immutable."""
        return AgentCapabilityBlueprint(
            self.cls,
            self.kwargs,
            key=key if key is not None else self.key,
            include_actions=include_actions if include_actions is not None else self.include_actions,
            exclude_actions=exclude_actions if exclude_actions is not None else self.exclude_actions,
            events_only=events_only if events_only is not None else self.events_only,
        )

    def __repr__(self) -> str:
        parts = [f"key={self.key!r}"]
        if self.kwargs:
            parts.extend(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"AgentCapabilityBlueprint({self.cls.__name__}, {', '.join(parts)})"


# ─── ActionPolicyBlueprint ────────────────────────────────────────────────────

class ActionPolicyBlueprint(Blueprint["ActionPolicy"]):
    """Blueprint for an ActionPolicy.

    The agent and action_providers arguments are injected via local_instance
    prefix/extra args at instantiation time (not bound).
    """

    pass


# ─── AgentBlueprint ───────────────────────────────────────────────────────────

class AgentBlueprint(Blueprint["Agent"]):
    """Blueprint for an Agent class bound with constructor arguments
    with pydantic model_fields validation.

    At bind time:
    - Rejects kwargs that aren't Agent model fields
    - Rejects excluded fields (state, created_at, etc.) — runtime-managed
    - Requires all required fields except deferred ones (agent_id — filled by pipeline)
    - Validates serializability via cloudpickle
    """

    # Runtime-managed fields: never allowed in kwargs, never required at bind time.
    _EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "state",
        "created_at",
        "action_policy",
        "action_policy_state",
        "page_storage",
        "child_agents",
    })

    # Required on Agent but filled by the spawn pipeline before local_instance().
    _DEFERRED_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "agent_id",
    })

    def _deferred_fields(self) -> frozenset[str]:
        return self._EXCLUDED_FIELDS | self._DEFERRED_FIELDS

    def __init__(self, cls: type[Agent], kwargs: dict[str, Any] | None = None):
        kwargs = kwargs or {}
        model_fields = set(cls.model_fields.keys())
        allowed = model_fields - self._EXCLUDED_FIELDS
        invalid = set(kwargs.keys()) - allowed
        if invalid:
            raise BlueprintValidationError(
                f"Invalid bind() arguments for {cls.__name__}: {invalid}. "
                f"Allowed: {sorted(allowed)}"
            )
        super().__init__(cls, kwargs)

    def __getattr__(self, name: str) -> Any:
        # kwargs first, then model field defaults — so blueprint.metadata
        # returns AgentMetadata() even when metadata wasn't passed to .bind().
        try:
            kwargs = object.__getattribute__(self, "kwargs")
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            )
        try:
            return kwargs[name]
        except KeyError:
            cls = object.__getattribute__(self, "cls")
            field_info = cls.model_fields.get(name)
            if field_info is not None:
                from pydantic.fields import PydanticUndefined
                if field_info.default is not PydanticUndefined:
                    return field_info.default
                if field_info.default_factory is not None:
                    return field_info.default_factory()
            raise AttributeError(
                f"'{type(self).__name__}' for {cls.__name__} has no attribute '{name}'"
            )

    def has_deployment_affinity(self) -> bool:
        """Check if agent needs VLLM routing (has bound pages)."""
        return bool(self.bound_pages) # Will return self.kwargs.get("bound_pages")

    async def remote_instance(
        self,
        *,
        requirements: "LLMClientRequirements" | None = None,
        soft_affinity: bool = True,
        suspend_agents: bool = False,
        app_name: str | None = None,
    ) -> str:
        """Spawn this agent on a remote deployment.

        Only true deployment parameters go here — things that affect routing
        and resource allocation. Agent identity (agent_id, session_id, run_id,
        max_iterations) belongs in the blueprint's kwargs/metadata.

        Returns:
            The spawned agent_id
        """
        from ..system import get_agent_system

        agent_system = get_agent_system(app_name)
        try:
            return await agent_system.spawn_from_blueprint(
                blueprint=self,
                requirements=requirements,
                soft_affinity=soft_affinity,
                suspend_agents=suspend_agents,
            )
        except Exception as e:
            raise BlueprintRemoteError(
                f"Failed to spawn {self.cls.__name__} remotely: {e}",
                remote_traceback=getattr(e, "traceback", str(e)),
            ) from e

    def __repr__(self) -> str:
        parts = [f"{k}={v!r}" for k, v in self.kwargs.items()]
        if parts:
            return f"AgentBlueprint({self.cls.__name__}, {', '.join(parts)})"
        return f"AgentBlueprint({self.cls.__name__})"


# ─── blueprint() decorator ────────────────────────────────────────────────────

def blueprint(cls: type[T]) -> type[T]:
    """Add .bind() to any class. No-op if cls already has bind (via base class).

    For Agent/AgentCapability/ActionPolicy subclasses: returns cls as-is
    (they define .bind() on the base class).
    For other classes: adds a generic Blueprint .bind() classmethod.
    """
    if hasattr(cls, "bind"):
        return cls

    @classmethod  # type: ignore[misc]
    def bind(klass: type, **kwargs: Any) -> Blueprint:
        bp = Blueprint(klass, kwargs)
        bp.validate_serializable()
        return bp

    cls.bind = bind  # type: ignore[attr-defined]
    return cls
