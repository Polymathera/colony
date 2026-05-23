"""``ToolCapability`` — base class for agent tools.

The user-set design principle (see
``colony/STAGE_B_TOOL_FRAMEWORK_RETROFIT_PLAN.md`` Part 1):

> Every tool should appear to an agent as an :class:`AgentCapability`
> with one or more tool-specific actions that the agent can use to
> interact with the tool. The single-``invoke`` interface is rejected
> — it forces every tool through one synthetic signature and hides
> the per-tool semantics the LLM planner needs to use the tool
> correctly.

This module defines three layers:

- :class:`ToolCapability` — abstract base. Subclasses set a class-level
  :class:`~polymathera.colony.tools.ToolSpec`, override one or more
  ``@action_executor`` methods, and inherit the boilerplate that
  surfaces the spec to the LLM planner (via
  :meth:`get_action_group_description`), tags the capability as a
  tool (via :meth:`get_capability_tags`), and exposes a uniform
  ``check_preconditions`` action.
- :class:`LocalToolCapability` — convenience pass-through for in-process
  / cli-subprocess tools that don't need any extra infrastructure.
- :class:`SandboxToolCapability` — for tools whose body must run inside
  a sandboxed Docker container. Delegates container lifecycle to a
  sibling :class:`SandboxedShellCapability` instance discovered via
  the agent.

Subclasses written by tool authors look like:

.. code-block:: python

    from polymathera.colony.tools import (
        CostModel, ExecutionLocality, HITLFrequency, HeadlessReadiness,
        Licensing, ResourceRequirements, ToolSpec,
    )
    from polymathera.colony.agents.patterns.actions import action_executor
    from polymathera.colony.agents.patterns.capabilities.tool import (
        LocalToolCapability,
    )


    class MagerSumnerShieldingCapability(LocalToolCapability):
        spec = ToolSpec(
            name="mager_sumner_shielding",
            domain="em",
            capabilities=("compute_shielding_factor",),
            backend="in_process",
            execution_locality=ExecutionLocality.LOCAL,
            determinism=...,
            cost_model=CostModel(cpu_seconds=0.1),
            resource_requirements=ResourceRequirements(min_vcpus=1, min_memory_gb=0.5),
            headless=HeadlessReadiness.NATIVE,
            hitl_frequency=HITLFrequency.AUTONOMOUS,
            licensing=Licensing.MIT,
        )

        @action_executor()
        async def compute_shielding_factor(
            self,
            *,
            layers_mm: list[float],
            relative_permeability: list[float],
            inner_radius_mm: float,
        ) -> dict[str, float]:
            ...
"""

from __future__ import annotations

import logging
from abc import ABC
from typing import Any, ClassVar, TYPE_CHECKING

from overrides import override

from ...base import AgentCapability
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions import action_executor


if TYPE_CHECKING:
    from ....tools import ToolSpec
    from ...base import Agent
    from .sandboxed_shell import SandboxedShellCapability


logger = logging.getLogger(__name__)


TOOL_TAG: str = "tool"
"""Capability-tag string that classifies a capability as a tool.

Action-policy planners filter the agent's action menu by tag via
:meth:`~polymathera.colony.agents.patterns.actions.dispatcher.ActionDispatcher.get_action_descriptions`
with ``include_tags={"tool"}`` to enumerate just the tools available
to the agent — the LLM-visible discovery surface for the tool framework.
``ToolCapability`` always merges this tag into the subclass's
:meth:`get_capability_tags` result; the subclass adds domain tags
(e.g. ``{"em", "fdtd"}``) on top."""


class ToolCapability(AgentCapability, ABC):
    """Abstract base for tool capabilities.

    Subclass contract:

    - Declare a class-level :class:`~polymathera.colony.tools.ToolSpec`
      as ``spec`` — every other method on this base reads metadata off it.
      ``__init_subclass__`` enforces the declaration so a missing or
      wrong-type ``spec`` fails at import time, not at runtime.
    - Override :meth:`get_capability_tags` to return domain tags
      (e.g. ``frozenset({"em", "fdtd"})``). The ``"tool"`` tag is
      merged in automatically — do NOT add it manually.
    - Implement one or more ``@action_executor``-decorated methods
      named for what the tool *does* (``run_fdtd``, ``compute_shielding_factor``).
      The dispatcher surfaces them to the LLM planner; the planner
      sees the spec metadata inline via the action-group description.

    Subclasses MAY override :meth:`check_preconditions` to add
    environment-specific checks; the default implementation returns
    the spec's resource requirements verbatim so the planner can
    reason about whether the local cluster / HPC endpoint can fit
    the call.
    """

    spec: ClassVar["ToolSpec"]
    """Class-level frozen description of the tool. Subclasses MUST
    declare. Enforced by :meth:`__init_subclass__`."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip abstract intermediate bases (LocalToolCapability /
        # SandboxToolCapability) — they're not themselves concrete tools.
        if cls.__dict__.get("_TOOL_CAPABILITY_ABSTRACT", False):
            return
        spec = cls.__dict__.get("spec")
        if spec is None:
            # Walk MRO for a declared spec — subclasses of subclasses can
            # inherit one.
            for base in cls.__mro__[1:]:
                if base is ToolCapability:
                    break
                if "spec" in base.__dict__:
                    spec = base.__dict__["spec"]
                    break
        if spec is None:
            raise TypeError(
                f"{cls.__name__} subclasses ToolCapability but does not "
                "declare a class-level ``spec: ClassVar[ToolSpec]``. "
                "Set ``spec = ToolSpec(name=..., capabilities=(...), ...)``.",
            )
        # Local import to avoid a top-level cycle (tools/spec.py → here).
        from ....tools import ToolSpec  # noqa: PLC0415
        if not isinstance(spec, ToolSpec):
            raise TypeError(
                f"{cls.__name__}.spec must be a ToolSpec instance, "
                f"got {type(spec).__name__}.",
            )

    def __init__(
        self,
        agent: "Agent | None" = None,
        scope: BlackboardScope = BlackboardScope.AGENT,
        namespace: str | None = None,
        *,
        scope_id: str | None = None,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        # ``namespace`` defaults to the tool's spec name so two
        # ToolCapability instances mounted on the same agent (e.g. two
        # different FEM tools) get distinct blackboard scopes by
        # default. ``scope_id`` is the explicit override (preferred in
        # tests + detached use).
        if scope_id is None:
            scope_id = get_scope_prefix(
                scope, agent,
                namespace=namespace or type(self).spec.name,
            )
        # ``input_patterns=None`` triggers the base class's
        # auto-inference from ``@event_handler`` decorators on this
        # capability + its MRO. The handler picks up
        # ``AgentRunProtocol.request_pattern()`` automatically.
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=None,
            capability_key=capability_key or type(self).spec.name,
            app_name=app_name,
        )

    # ------------------------------------------------------------------
    # LLM-visible metadata
    # ------------------------------------------------------------------

    def get_capability_tags(self) -> frozenset[str]:
        """Tags exposed to the action-policy planner.

        The ``"tool"`` tag is always present so planners can filter
        for tool-shaped capabilities. Subclasses override this method
        to add domain tags (``"em"``, ``"fem"``, ``"hpc"``); the base
        merges the override into the canonical ``"tool"`` tag.
        """
        return frozenset({TOOL_TAG}) | self._domain_tags()

    def _domain_tags(self) -> frozenset[str]:
        """Subclass hook: return domain / modality tags to merge with
        the canonical ``"tool"`` tag. Default empty."""
        return frozenset()

    def get_action_group_description(self) -> str:
        """Render the ToolSpec metadata into the LLM-visible action-group
        description.

        The dispatcher surfaces this string to the planner alongside
        each action's docstring, so the planner sees the tool's cost,
        resource requirements, HITL tier, licence, and locality inline
        with the action menu — no extra ``describe_tool`` action needed.
        Subclasses MAY extend the description by overriding
        :meth:`_describe_tool_extras` rather than this method directly.
        """
        spec = type(self).spec
        cost = spec.cost_model
        req = spec.resource_requirements
        gpu = req.gpu
        gpu_summary = (
            f"{gpu.count}× {gpu.kind}"
            + (f" ({gpu.memory_gb:.0f} GB)" if gpu.memory_gb else "")
            if gpu is not None else "none"
        )
        lines = [
            f"Tool: {spec.name} (v{spec.version}, domain={spec.domain}, "
            f"backend={spec.backend}, locality={spec.execution_locality.value}).",
            f"Capabilities: {', '.join(spec.capabilities) or '<none>'}.",
            f"Licence: {spec.licensing.value} "
            f"({spec.licensing_notes or 'no notes'}).",
            f"Headless: {spec.headless.value}; HITL: {spec.hitl_frequency.value}; "
            f"determinism: {spec.determinism.value}.",
            f"Per-call cost (estimated): cpu={cost.cpu_seconds:g}s, "
            f"gpu={cost.gpu_seconds:g}s, memory={cost.memory_gb:g} GB, "
            f"dollars=${cost.dollars:g}.",
            f"Per-call minimums: {req.min_vcpus} vCPU, "
            f"{req.min_memory_gb:g} GB RAM, GPU={gpu_summary}, "
            f"wall-clock estimate={req.expected_wallclock_seconds:g}s.",
        ]
        extras = self._describe_tool_extras()
        if extras:
            lines.append(extras)
        return " ".join(lines)

    def _describe_tool_extras(self) -> str:
        """Subclass hook: extra text appended to the action-group
        description (e.g. routing / sandboxing notes specific to the
        subclass family). Default empty."""
        return ""

    # ------------------------------------------------------------------
    # Standard action: precondition check
    # ------------------------------------------------------------------

    @action_executor(
        planning_summary=(
            "Check whether the tool's per-call resource minimums are "
            "satisfied in the current environment. Returns a structured "
            "report the planner can use before submitting an expensive call."
        ),
    )
    async def check_preconditions(self) -> dict[str, Any]:
        """Return a structured snapshot of the tool's preconditions.

        Default implementation returns the spec's
        :class:`~polymathera.colony.tools.ResourceRequirements` and
        :class:`~polymathera.colony.tools.ExecutionLocality` as a dict.
        Subclasses can extend with environment-side checks (e.g.
        :class:`~polymathera.cps.tools.hpc.capability.HPCToolCapability`
        validates against the operator's ``cps.hpc.limits``).
        """
        spec = type(self).spec
        return {
            "tool": spec.name,
            "execution_locality": spec.execution_locality.value,
            "resource_requirements": spec.resource_requirements.model_dump(mode="json"),
            "ok": True,
            "warnings": [],
        }

    # ------------------------------------------------------------------
    # Suspension lifecycle (no tool-side state to persist by default)
    # ------------------------------------------------------------------

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> AgentSuspensionState:
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> None:
        return None


# ---------------------------------------------------------------------------
# Concrete intermediate bases
# ---------------------------------------------------------------------------


class LocalToolCapability(ToolCapability, ABC):
    """Convenience base for tools that run in-process (or via a
    cli_subprocess) inside the agent's own Ray actor.

    Adds nothing on top of :class:`ToolCapability` — the subclass
    declares ``spec`` and one or more ``@action_executor`` methods.
    The class exists so tool authors don't have to remember whether
    the in-process case needs any extra wiring (it doesn't), and so
    the type system signals "this tool runs locally" at the class
    name (sibling to :class:`SandboxToolCapability` and the CPS-side
    :class:`~polymathera.cps.tools.hpc.capability.HPCToolCapability`).
    """

    _TOOL_CAPABILITY_ABSTRACT: ClassVar[bool] = True


class SandboxToolCapability(ToolCapability, ABC):
    """Base for tools whose body runs inside a sandboxed Docker container.

    Tool authors set :attr:`sandbox_image_role` on the subclass; the
    base discovers the agent's mounted
    :class:`SandboxedShellCapability` lazily, lifts a container on
    first call via ``launch_container(image_role=...)``, and reuses
    it for subsequent calls (cheap). The container is stopped at
    capability suspension time so each session's containers are
    cleaned up deterministically.

    Subclasses implement their public ``@action_executor`` methods and
    call :meth:`_exec_in_sandbox` to dispatch a command.

    The shared-container strategy is intentional: tool capabilities
    typically issue many small commands per agent turn (parse, render,
    run, fetch), and a per-call container would pay launch latency on
    every call. Subclasses that need a fresh container per call can
    override :meth:`_container_id` to return ``None`` and call
    :meth:`_launch_fresh_container` themselves.
    """

    _TOOL_CAPABILITY_ABSTRACT: ClassVar[bool] = True

    sandbox_image_role: ClassVar[str]
    """Required: the image-role key registered in
    :class:`~polymathera.colony.agents.configs.DockerImageRegistryConfig`
    that the agent's :class:`SandboxedShellCapability` will launch
    when the tool first runs."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("_TOOL_CAPABILITY_ABSTRACT", False):
            return
        if not getattr(cls, "sandbox_image_role", None):
            raise TypeError(
                f"{cls.__name__} subclasses SandboxToolCapability but "
                "does not declare ``sandbox_image_role: ClassVar[str]``. "
                "Set it to the image-role key registered in "
                "``DockerImageRegistryConfig``.",
            )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._container_id_cache: str | None = None

    @override
    def _describe_tool_extras(self) -> str:
        return (
            f"Runs inside a sandboxed container of image-role "
            f"``{self.sandbox_image_role}`` (launched lazily on the "
            "first action call; reused across subsequent calls in "
            "this session)."
        )

    # ------------------------------------------------------------------
    # Subclass helpers
    # ------------------------------------------------------------------

    async def _shell(self) -> "SandboxedShellCapability":
        """Return the agent's mounted ``SandboxedShellCapability``.

        Raises :class:`RuntimeError` when the capability is detached
        (no owning agent) or when the agent hasn't mounted a
        :class:`SandboxedShellCapability` — sandboxed tools require it.
        """
        if self.is_detached:
            raise RuntimeError(
                f"{type(self).__name__}: capability is detached "
                "(agent=None); cannot dispatch into a sandbox without "
                "an agent-mounted SandboxedShellCapability.",
            )
        # Local import to avoid a top-level cycle.
        from .sandboxed_shell import SandboxedShellCapability  # noqa: PLC0415
        shell = self.agent.get_capability_by_type(SandboxedShellCapability)
        if shell is None:
            raise RuntimeError(
                f"{type(self).__name__}: the agent must mount "
                "SandboxedShellCapability before this tool can run "
                "(sandbox_image_role="
                f"{self.sandbox_image_role!r}).",
            )
        return shell

    async def _container_id(self) -> str:
        """Return the container_id this capability dispatches into,
        launching one on first call.

        Subclasses MAY override (e.g. to launch a fresh container per
        call rather than reuse). The default reuse strategy is correct
        for most tools — they issue many small commands per turn.
        """
        if self._container_id_cache is not None:
            return self._container_id_cache
        shell = await self._shell()
        launch = await shell.launch_container(image_role=self.sandbox_image_role)
        if not launch.get("started"):
            raise RuntimeError(
                f"{type(self).__name__}: launch_container failed: "
                f"{launch.get('message') or 'no message'}",
            )
        self._container_id_cache = launch["container_id"]
        return self._container_id_cache

    async def _launch_fresh_container(self) -> str:
        """Launch a fresh container for one-off use; the caller is
        responsible for cleanup via the shell capability."""
        shell = await self._shell()
        launch = await shell.launch_container(image_role=self.sandbox_image_role)
        if not launch.get("started"):
            raise RuntimeError(
                f"{type(self).__name__}: launch_container failed: "
                f"{launch.get('message') or 'no message'}",
            )
        return launch["container_id"]

    async def _exec_in_sandbox(
        self,
        command: list[str] | str,
        *,
        timeout_seconds: int = 300,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        stdin: str | None = None,
        capture_max_bytes: int = 1_000_000,
    ) -> dict[str, Any]:
        """Execute one command inside the shared container.

        Mirrors :meth:`SandboxedShellCapability.execute_command`'s
        kwarg surface (minus the container_id, which the helper supplies).
        Subclasses' ``@action_executor`` methods call this and reshape
        the result into the tool's typed return.
        """
        shell = await self._shell()
        container_id = await self._container_id()
        return await shell.execute_command(
            container_id,
            command,
            timeout_seconds=timeout_seconds,
            env=env,
            workdir=workdir,
            stdin=stdin,
            capture_max_bytes=capture_max_bytes,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Stop the shared container if one was launched."""
        if self._container_id_cache is None or self.is_detached:
            return
        try:
            shell = await self._shell()
        except RuntimeError:
            return
        try:
            await shell.stop_container(self._container_id_cache)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            logger.exception(
                "%s: stop_container raised during shutdown; container "
                "may persist on the daemon",
                type(self).__name__,
            )
        self._container_id_cache = None


__all__ = (
    "TOOL_TAG",
    "ToolCapability",
    "LocalToolCapability",
    "SandboxToolCapability",
)
