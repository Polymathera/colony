"""Typed declaration layer for ``AgentMetadata.parameters``.

See ``colony/agent_metadata_parameter_spec_plan.md`` for the
architectural intent. Short form:

* ``AgentMetadata.parameters: dict[str, Any]`` is a free-form bag.
  Without declarations, no spawn site knows what to thread, no
  capability validates what it reads, and the LLM planner guesses
  from prose.
* This module adds a typed declaration layer ON TOP of the bag —
  the wire format on ``AgentMetadata`` is unchanged. Existing
  readers using ``params.get("key")`` keep working unmodified.
* Capabilities declare their needs via the ``AGENT_METADATA_PARAMS``
  ClassVar on ``AgentCapability``. Missions declare CALLER-scoped
  needs via ``MissionSpec.caller_parameters``.
* The registry walks the capability tree on demand, deduplicating
  identical re-declarations and raising on conflicting ones.
* :func:`inherit_scoped_parameters` is the central inheritance
  gate that ``AgentPoolCapability.create_agent`` calls for every
  spawn — fixes the bug class where every spawn site had to
  remember to copy ``design_monorepo_url`` / ``git_attribution`` /
  ``github_identity`` by hand.

Pydantic-style optionality: a :class:`ParameterSpec` is required
**iff neither ``default`` nor ``default_factory`` is set**. No
separate ``required: bool`` field (that would create the
"is required=True with default=None required?" ambiguity Pydantic
itself avoided).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator


__all__ = (
    "MISSING",
    "AgentMetadataValidationError",
    "ParameterScope",
    "ParameterSpec",
    "MetadataParameterRegistry",
    "get_metadata_parameter_registry",
    "rebuild_metadata_parameter_registry_for_testing",
    "inherit_scoped_parameters",
)


class AgentMetadataValidationError(ValueError):
    """Raised at capability ``initialize()`` when a required metadata
    parameter declared on ``AGENT_METADATA_PARAMS`` is missing from
    the agent's ``metadata.parameters``.

    Subclass of ``ValueError`` so generic ``except ValueError`` paths
    in the agent system bubble it up cleanly. Catch this specific
    type when you need to distinguish a metadata-shape error from
    other validation failures.
    """


class _Missing:
    """Private sentinel for "no default declared" on :class:`ParameterSpec`.

    ``None`` is a valid default value, so we need a sentinel distinct
    from ``None`` to express "the spec author did not write a default
    at all" — which is what makes a spec required.
    """

    _instance: ClassVar["_Missing | None"] = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISSING>"

    def __bool__(self) -> bool:
        return False


MISSING: Any = _Missing()


class ParameterScope(str, Enum):
    """How a ``metadata.parameters`` key flows at spawn time.

    See ``colony/agent_metadata_parameter_spec_plan.md`` §2.1 for the
    full table. Brief form:

    * ``COLONY`` — bound to the colony; auto-inherited parent→child.
    * ``SESSION`` — bound to the session lifecycle; auto-inherited
      parent→child within the same session.
    * ``CALLER`` — supplied by the spawn caller (LLM planner or REST
      handler); NOT inherited.
    * ``AGENT`` — per-agent, stamped by spawn machinery; NOT
      inherited (each agent gets its own).
    """

    COLONY = "colony"
    SESSION = "session"
    CALLER = "caller"
    AGENT = "agent"


class ParameterSpec(BaseModel):
    """Declaration of one ``metadata.parameters`` key."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        # Specs are declarative — the same instance is shared by the
        # registry and any consumers. Frozen prevents accidental
        # mutation that would silently break that sharing.
        frozen=True,
    )

    name: str
    scope: ParameterScope
    description: str
    json_type: Literal[
        "string", "integer", "number", "boolean", "object", "array",
    ] = "string"
    default: Any = MISSING
    default_factory: Callable[[], Any] | None = None

    @model_validator(mode="after")
    def _exactly_one_default(self) -> "ParameterSpec":
        if self.default is not MISSING and self.default_factory is not None:
            raise ValueError(
                f"ParameterSpec {self.name!r}: cannot declare both "
                f"``default`` and ``default_factory`` — pick one."
            )
        return self

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        """Make ``ParameterSpec`` JSON-safe under the typed config
        layer (``MissionRegistryConfig.model_dump_json``).

        The default Pydantic serializer can't encode our private
        ``MISSING`` sentinel (it's not a JSON type) or a
        ``default_factory`` callable. ``mode="plain"`` replaces the
        default with this explicit projection:

        * ``default is MISSING`` (the spec is required) → no
          ``default`` key in the output. Required-ness is encoded
          by absence, Pydantic-style.
        * ``default_factory`` is set → materialise the factory and
          render its result under ``default``. The callable itself
          is dropped; round-trip is lossy but inspection-faithful
          (the operator / planner sees the value that runtime
          actually applies).
        * otherwise → ``default`` is the literal declared.

        Equality / hashing are unaffected — those use field values
        directly, not serialised form.
        """

        out: dict[str, Any] = {
            "name": self.name,
            "scope": self.scope.value,
            "description": self.description,
            "json_type": self.json_type,
        }
        if self.default_factory is not None:
            out["default"] = self.default_factory()
        elif self.default is not MISSING:
            out["default"] = self.default
        return out

    @property
    def required(self) -> bool:
        """True iff this spec carries no default of any kind."""
        return self.default is MISSING and self.default_factory is None

    def resolve_default(self) -> Any:
        """Materialise the declared default.

        Calls ``default_factory()`` if set; otherwise returns
        ``default``. Raises if the spec is required (no default of
        any kind) — callers should gate on :attr:`required` first.
        """
        if self.default_factory is not None:
            return self.default_factory()
        if self.default is MISSING:
            raise RuntimeError(
                f"ParameterSpec {self.name!r} is required — "
                f"no default to resolve.",
            )
        return self.default


class MetadataParameterRegistry:
    """Process-level registry mapping ``parameters`` keys → :class:`ParameterSpec`.

    Built on demand by :func:`get_metadata_parameter_registry`
    (which walks ``AgentCapability.__subclasses__()`` recursively
    and reads each class's ``AGENT_METADATA_PARAMS`` ClassVar).
    Mission-level CALLER specs are layered on top by the
    mission-registry loader (see ``agents.mission_registry``).

    Conflict policy: re-registering the same name with a non-equal
    spec raises ``ValueError``. Two capabilities (or a capability
    plus a mission) that share a key MUST agree on every field of
    the spec.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ParameterSpec] = {}

    def register(self, spec: ParameterSpec) -> None:
        existing = self._specs.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(
                f"metadata parameter {spec.name!r} re-declared with a "
                f"different spec.\n  existing: {existing!r}\n  new:      {spec!r}",
            )
        self._specs[spec.name] = spec

    def get(self, name: str) -> ParameterSpec | None:
        return self._specs.get(name)

    def keys_by_scope(self, scope: ParameterScope) -> frozenset[str]:
        return frozenset(
            s.name for s in self._specs.values() if s.scope == scope
        )

    def all_specs(self) -> tuple[ParameterSpec, ...]:
        return tuple(self._specs.values())


_REGISTRY: MetadataParameterRegistry | None = None


def get_metadata_parameter_registry() -> MetadataParameterRegistry:
    """Return the process-wide registry, building on first call.

    The build walks every ``AgentCapability`` subclass reachable from
    the current import graph and unions their ``AGENT_METADATA_PARAMS``
    declarations. The capability tree must already be imported — the
    main agent-system entry points (``main.py``, ``cli/polymath.py``)
    do this transitively at startup.

    Specs declared by lazily-imported modules after the first build
    are not picked up automatically; call
    :func:`rebuild_metadata_parameter_registry_for_testing` in tests
    that register specs after the cache has been primed.
    """

    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def rebuild_metadata_parameter_registry_for_testing() -> MetadataParameterRegistry:
    """Drop the cache and rebuild from the current ``AgentCapability`` tree.

    Tests-only — production code uses :func:`get_metadata_parameter_registry`.
    """

    global _REGISTRY
    _REGISTRY = _build_registry()
    return _REGISTRY


def _build_registry() -> MetadataParameterRegistry:
    # Imported here, not at module top, to keep this module importable
    # before the agent system has loaded. ``base`` itself imports this
    # module (via the AgentCapability ClassVar default), so a top-level
    # import would be a cycle.
    from polymathera.colony.agents.base import AgentCapability

    registry = MetadataParameterRegistry()
    for cls in _iter_subclasses(AgentCapability):
        specs = getattr(cls, "AGENT_METADATA_PARAMS", ())
        for spec in specs:
            registry.register(spec)
    return registry


def _iter_subclasses(cls: type) -> Iterable[type]:
    """Yield every subclass of ``cls`` reachable via ``__subclasses__``.

    Depth-first; deduplicates so a diamond inheritance doesn't yield
    a class twice.
    """

    seen: set[type] = set()

    def walk(c: type) -> Iterable[type]:
        for sub in c.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            yield sub
            yield from walk(sub)

    yield from walk(cls)


def inherit_scoped_parameters(
    *,
    child_params: dict[str, Any],
    parent_params: dict[str, Any],
    registry: MetadataParameterRegistry,
) -> dict[str, Any]:
    """Inherit COLONY + SESSION scoped keys parent→child.

    The single central inheritance gate. Returns a fresh dict; does
    NOT mutate inputs.

    Semantics:

    * Only keys whose registered scope is :attr:`ParameterScope.COLONY`
      or :attr:`ParameterScope.SESSION` flow. CALLER and AGENT scoped
      keys never flow at spawn time.
    * Caller wins on collision — a key already present in
      ``child_params`` keeps the child's value. The LLM planner can
      deliberately rebind (e.g. a mission probing a different repo).
    * A parent value of ``None`` is treated as absent (not inherited).
      This matches the convention everywhere else in the codebase
      that ``params.get(key)`` returns ``None`` for "not set".
    """

    child = dict(child_params)
    if not parent_params:
        return child
    inheritable = (
        registry.keys_by_scope(ParameterScope.COLONY)
        | registry.keys_by_scope(ParameterScope.SESSION)
    )
    for key in inheritable:
        if key in child:
            continue
        value = parent_params.get(key)
        if value is None:
            continue
        child[key] = value
    return child
