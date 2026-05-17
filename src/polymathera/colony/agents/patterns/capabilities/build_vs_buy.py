"""``BuildVsBuyCapability`` — agent-facing wrapper over ``BuildVsBuyAdvisor``.

Mountable :class:`AgentCapability` that lets an LLM-driven agent invoke
the master §4.3 six-rule build-vs-buy policy as a first-class action.

The capability is a thin shell. Pure logic lives in
:mod:`.build_vs_buy_engine`; this file is the mounting + dispatch
surface. The wrapper picks up the agent's :class:`RepoStateProvider`
(if mounted) to enable the C5 augment-vs-build refinement — without
one, the advisor still works but the AUGMENT path is unreachable.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from overrides import override

from ...base import AgentCapability
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions import action_executor
from .build_vs_buy_engine import (
    BuildVsBuyAdvisor,
    BuildVsBuyContext,
    TeamTrackRecord,
)


if TYPE_CHECKING:
    from ....tools import Licensing, ToolSpec
    from ...base import Agent


logger = logging.getLogger(__name__)


class BuildVsBuyCapability(AgentCapability):
    """Agent capability: ask the master §4.3 advisor whether to BUILD,
    BUY, AUGMENT, HYBRID, CROSS_CHECK_ONLY, or DENY a tool.

    One action — ``recommend_build_or_buy`` — surfaces the engine. The
    agent's existing :class:`RepoStateProvider` (when mounted) is
    looked up lazily and supplied to the advisor so the local-match
    AUGMENT path activates.
    """

    def __init__(
        self,
        agent: "Agent",
        scope: BlackboardScope = BlackboardScope.AGENT,
        namespace: str = "build_vs_buy",
        *,
        capability_key: str = "build_vs_buy",
        app_name: str | None = None,
    ):
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            capability_key=capability_key,
            app_name=app_name,
        )

    def get_action_group_description(self) -> str:
        return (
            "Build-vs-buy advisor (master §4.3 six-rule policy). Given "
            "a capability query and the external tools surveyed, "
            "returns one of AUGMENT / BUY / BUILD / HYBRID / "
            "CROSS_CHECK_ONLY / DENY plus a rule-by-rule trace and an "
            "effort estimate. AUGMENT activates when a writable local "
            "tool match exists via RepoStateProvider.find_existing_tool."
        )

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"build_vs_buy", "tool_selection", "planning"})

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

    # ---------------------------------------------------------------
    # Action surface
    # ---------------------------------------------------------------

    @action_executor(
        planning_summary=(
            "Ask the build-vs-buy advisor whether to AUGMENT / BUY / "
            "BUILD / HYBRID / CROSS_CHECK_ONLY / DENY a tool capability."
        ),
    )
    async def recommend_build_or_buy(
        self,
        *,
        capability_query: str,
        available_external_tools: list[dict] | None = None,
        inner_loop_call_frequency_per_workflow: int = 0,
        validation_against_gold_feasible: bool = True,
        custom_can_be_differentiable: bool = False,
        custom_problem_narrower_than_external: bool = True,
        licence_forbidden: list[str] | None = None,
        require_determinism: bool = False,
        team_custom_tools_built: int = 0,
        team_custom_tools_validated_against_gold: int = 0,
        team_average_tool_build_months: float = 3.0,
    ) -> dict[str, Any]:
        """Run the §4.3 six-rule policy.

        ``available_external_tools`` is a list of ``ToolSpec``-shaped
        dicts (the same field set as :class:`polymathera.colony.tools.ToolSpec`);
        validation runs through :meth:`ToolSpec.model_validate` so the
        LLM can pass a JSON-friendly structure.

        ``licence_forbidden`` is a list of SPDX-style licence names
        (e.g., ``["commercial", "agpl"]``); names are mapped to
        :class:`polymathera.colony.tools.Licensing` enum values and
        unknown names are ignored.

        Returns the ``BuildVsBuyVerdict`` as a dict.
        """
        # Imports kept local to avoid pulling colony.tools at module
        # import time (keeps the capability light when only its metadata
        # is being introspected, e.g. during action-policy planning).
        from ....tools import Licensing, ToolSpec

        external_specs: tuple[ToolSpec, ...] = tuple(
            ToolSpec.model_validate(entry)
            for entry in (available_external_tools or ())
        )

        forbid_set: frozenset[Licensing] = frozenset(
            Licensing(name) for name in (licence_forbidden or ())
            if name in Licensing._value2member_map_
        )

        context = BuildVsBuyContext(
            capability_query=capability_query,
            available_external_tools=external_specs,
            inner_loop_call_frequency_per_workflow=inner_loop_call_frequency_per_workflow,
            validation_against_gold_feasible=validation_against_gold_feasible,
            custom_can_be_differentiable=custom_can_be_differentiable,
            custom_problem_narrower_than_external=custom_problem_narrower_than_external,
            licence_forbidden=forbid_set,
            require_determinism=require_determinism,
            team=TeamTrackRecord(
                custom_tools_built=team_custom_tools_built,
                custom_tools_validated_against_gold=team_custom_tools_validated_against_gold,
                average_tool_build_months=team_average_tool_build_months,
            ),
        )

        advisor = BuildVsBuyAdvisor(repo_state_provider=self._resolve_repo_state_provider())
        verdict = await advisor.recommend(context)
        return verdict.model_dump(mode="json")

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------

    def _resolve_repo_state_provider(self) -> Any | None:
        """Return the agent's mounted ``RepoStateProvider`` instance, or
        ``None`` if the agent isn't bound to a design monorepo."""
        if self.agent is None:
            return None
        # Imported lazily so this module doesn't pull design_monorepo
        # transitively when the capability is only being introspected.
        from ....design_monorepo.capabilities import RepoStateProvider

        return self.agent.get_capability_by_type(RepoStateProvider)


__all__ = ("BuildVsBuyCapability",)
