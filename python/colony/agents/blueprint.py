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

    Validates at bind time:
    - Serializability via ray.cloudpickle.dumps
    - For pydantic BaseModel subclasses: kwargs are valid model fields

    Invariants:
    - self.cls is the actual class (not a string path)
    - self.kwargs contains ONLY constructor kwargs for cls.__init__
    - No positional args stored (prefix_args injected at local_instance time)
    """

    __slots__ = ("cls", "kwargs")

    def __init__(self, cls: type[T], kwargs: dict[str, Any] | None = None):
        self.cls = cls
        self.kwargs = kwargs or {}

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
    - Rejects excluded fields (agent_id, state, created_at, etc.)
    - Validates serializability via cloudpickle
    Full pydantic validation deferred to local_instance (needs agent_id).
    """

    _EXCLUDED_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "agent_id",
        "state",
        "created_at",
        "action_policy",
        "action_policy_state",
        "page_storage",
        "child_agents",
    })

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

    def has_deployment_affinity(self) -> bool:
        """Check if agent needs VLLM routing (has bound pages)."""
        return bool(self.kwargs.get("bound_pages"))

    async def remote_instance(
        self,
        *,
        requirements: Any | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        soft_affinity: bool = True,
        suspend_agents: bool = False,
        max_iterations: int | None = None,
        app_name: str | None = None,
    ) -> str:
        """Spawn this agent on a remote deployment.

        Deployment parameters (NOT Agent constructor args) go here.
        Catches remote exceptions and raises BlueprintRemoteError
        with the remote traceback attached.

        Returns:
            The spawned agent_id
        """
        from ..system import get_agent_system

        agent_system = get_agent_system(app_name)
        try:
            return await agent_system.spawn_from_blueprint(
                blueprint=self,
                requirements=requirements,
                agent_id=agent_id,
                session_id=session_id,
                run_id=run_id,
                soft_affinity=soft_affinity,
                suspend_agents=suspend_agents,
                max_iterations=max_iterations,
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
