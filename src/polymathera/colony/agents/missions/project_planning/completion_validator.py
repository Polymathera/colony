"""Decompose-mode completion validator + backlog tracker.

The structural drain predicate that fixes the "one-and-done" pattern:
``DecomposeCompletionValidator`` rejects ``signal_completion()`` until
every in-scope issue has been processed (decomposed, classified as
non-decomposable, or explicitly early-stopped via the typed
``request_decompose_early_stop`` primitive). The predicate is
LLM-INDEPENDENT — the LLM cannot prompt-engineer around it. For
non-decompose modes the validator delegates to the configured
fallback (typically :class:`LLMCompletionValidator`).

Design notes:

- The tracker is extracted from the validator per [[extract-dont-bloat]]
  so the validator stays an orchestrator with one read pass +
  one comparison.
- Scope is read from ``agent.metadata.parameters`` via the canonical
  :attr:`ProjectPlanningCoordinator.ISSUE_NUMBERS_PARAM_NAME` / friends — NEVER a
  string literal at this layer per [[colony-scoped-params-propagation]].
- Applied parents are read from the codegen policy's
  ``_run_call_trace`` via the typed :class:`RunCallTrace` view (Change
  4) so a writer-side schema drift surfaces as a ``ValidationError``,
  not as a silent miscount here.
- Classified-non-decomposable is read from
  ``execution_context.action_results`` (the full untruncated action
  output stored alongside the trace) so the validator sees the real
  classifications, not the truncated ``output_preview`` string.
- Action keys are referenced via canonical
  :class:`DesignProcessCapability` ClassVars — never bare strings.
- The rejection message is FACT ONLY. No procedural prose. Per
  [[primitives-not-pipelines]] the validator states the gap, the LLM
  chooses how to close it (continue decomposing, ask the user,
  request early-stop). The validator never says "next, call X".

The early-stop state is written by
``DesignProcessCapability.request_decompose_early_stop`` to a
mission-scoped blackboard key the validator reads. The LLM cannot
self-certify — the action requires a verbatim user quote authorising
the early stop, and the framework records it for audit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from overrides import override

from polymathera.colony.agents.missions.project_planning.coordinator import (
    ProjectPlanningCoordinator,
)
from polymathera.colony.agents.missions.project_planning.mission_control import (
    DecomposeEarlyStopProtocol,
)
from polymathera.colony.agents.patterns.actions.code_constraints import (
    CompletionValidationResult,
    CompletionValidator,
    LLMCompletionValidator,
)
from polymathera.colony.agents.patterns.actions.run_call_trace import (
    RunCallTrace,
)
from polymathera.colony.design_monorepo.process import DesignProcessCapability


if TYPE_CHECKING:
    from polymathera.colony.agents.base import Agent
    from polymathera.colony.agents.models import PlanExecutionContext


logger = logging.getLogger(__name__)


# Blackboard key the validator publishes on every rejection so
# postmortems can replay the drain state at each rejection point. The
# format lives next to its writer; nothing else reads it.
def _drain_state_key(agent_id: str, sequence: int) -> str:
    return (
        f"mission:project_planning:decompose:drain_state:"
        f"{agent_id}:{sequence}"
    )


# ---------------------------------------------------------------------------
# Backlog tracker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecomposeBacklogTracker:
    """In-scope / applied / classified-non-decomposable sets for one
    decompose mission run.

    Immutable snapshot — constructed fresh by the validator on every
    ``validate()`` call. Holds no references to the agent or
    execution_context; downstream tests can construct it directly to
    pin the drain predicate without spinning up an agent.
    """

    in_scope: frozenset[int]
    applied: frozenset[int]
    classified_non_decomposable: frozenset[int]
    early_stopped: bool
    max_parents_per_run: int | None

    def remaining(self) -> frozenset[int]:
        """Issues still in scope and unaddressed."""

        if self.early_stopped:
            return frozenset()
        return self.in_scope - self.applied - self.classified_non_decomposable

    def is_drained(self) -> bool:
        """True iff there is nothing left to do.

        The cap (``max_parents_per_run``) interacts as documented in
        the plan: when set and the apply count has reached the cap,
        the validator treats the run as drained for completion
        purposes — remaining entries are deferred-out-of-scope (visible
        in the mission-final summary, not blocking completion).
        """

        if self.early_stopped:
            return True
        if (
            self.max_parents_per_run is not None
            and len(self.applied) >= self.max_parents_per_run
        ):
            return True
        return not self.remaining()


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class DecomposeCompletionValidator(CompletionValidator):
    """Structural drain predicate. NOT LLM-judged.

    The contract: for ``mission_params['mode'] == 'decompose'`` the
    validator allows ``signal_completion`` only when the
    :class:`DecomposeBacklogTracker` reports the in-scope set drained
    (or the LLM has recorded a typed early-stop signal). For other
    modes (``bootstrap``, ``refresh``, ``assignments``) the validator
    delegates to a fallback — defaulting to
    :class:`LLMCompletionValidator` to preserve current behavior.

    The validator NEVER prescribes the next step. The rejection
    message is fact-only; the LLM composes its own response — continue
    decomposing the remaining set, ask the user, refine the criteria,
    or call ``request_decompose_early_stop`` with a verbatim user quote
    to record an explicit stop.
    """

    def __init__(
        self, fallback: CompletionValidator | None = None,
    ) -> None:
        self._fallback: CompletionValidator = (
            fallback if fallback is not None else LLMCompletionValidator()
        )
        self._rejection_sequence: int = 0

    @override
    async def validate(
        self,
        agent: "Agent",
        goals: list[str],
        results: dict[str, Any],
        execution_context: "PlanExecutionContext",
    ) -> CompletionValidationResult:
        params = dict(agent.metadata.parameters or {})
        mode = params.get(ProjectPlanningCoordinator.MODE_PARAM_NAME)
        if mode != ProjectPlanningCoordinator.DECOMPOSE_MODE_VALUE:
            return await self._fallback.validate(
                agent=agent,
                goals=goals,
                results=results,
                execution_context=execution_context,
            )

        tracker = await self._build_tracker(agent, execution_context, params)
        if tracker.is_drained():
            return CompletionValidationResult(
                allowed=True,
                reason=(
                    "Decompose mission scope drained "
                    f"(applied {len(tracker.applied)}, "
                    f"non_decomposable {len(tracker.classified_non_decomposable)}"
                    + (
                        ", cap reached"
                        if (
                            tracker.max_parents_per_run is not None
                            and len(tracker.applied) >= tracker.max_parents_per_run
                        )
                        else ""
                    )
                    + (", early-stop recorded" if tracker.early_stopped else "")
                    + ")."
                ),
            )

        await self._publish_drain_state(agent, tracker)
        return CompletionValidationResult(
            allowed=False,
            reason=(
                "Decompose mission scope not drained. "
                f"In-scope issues: {sorted(tracker.in_scope)}. "
                f"Applied (decomposed): {sorted(tracker.applied)}. "
                f"Classified non-decomposable: "
                f"{sorted(tracker.classified_non_decomposable)}. "
                f"Remaining: {sorted(tracker.remaining())}."
            ),
        )

    async def _build_tracker(
        self,
        agent: "Agent",
        execution_context: "PlanExecutionContext",
        params: dict[str, Any],
    ) -> DecomposeBacklogTracker:
        in_scope = await self._resolve_scope(agent, params)
        trace = self._build_trace(execution_context)
        applied = self._extract_applied(trace)
        classified_non = self._extract_classified_non_decomposable(
            trace, execution_context,
        )
        early_stopped = await self._read_early_stop(agent)
        cap = params.get(ProjectPlanningCoordinator.MAX_PARENTS_PER_RUN_PARAM_NAME)
        max_parents_per_run = int(cap) if isinstance(cap, int) else None
        return DecomposeBacklogTracker(
            in_scope=in_scope,
            applied=applied,
            classified_non_decomposable=classified_non,
            early_stopped=early_stopped,
            max_parents_per_run=max_parents_per_run,
        )

    @staticmethod
    def _build_trace(
        execution_context: "PlanExecutionContext",
    ) -> RunCallTrace:
        """Construct the typed run-call-trace view from the codegen
        policy's per-iteration custom_data. The trace is stamped into
        ``codegen_step_summaries[step_id]['run_call_trace']`` and into
        the policy's ``_run_call_trace`` field. Read from the policy's
        field via the REPL namespace if available; otherwise reconstruct
        from step summaries.
        """

        step_summaries: dict[str, Any] = execution_context.custom_data.get(
            "codegen_step_summaries", {},
        ) or {}
        entries: list[dict[str, Any]] = []
        for info in step_summaries.values():
            if not isinstance(info, dict):
                continue
            sub = info.get("run_call_trace")
            if isinstance(sub, list):
                entries.extend(item for item in sub if isinstance(item, dict))
        return RunCallTrace(entries)

    @staticmethod
    def _extract_applied(trace: RunCallTrace) -> frozenset[int]:
        """Parent issue numbers passed to ``create_decomposition`` with
        ``dry_run`` explicitly False. ``dry_run`` is the canonical
        gate — the LLM is required to pass it; an absent value is
        treated as "did not apply" (the conservative choice — drain
        condition slightly harder to meet, but no false-positive
        completion)."""

        action_key = DesignProcessCapability.CREATE_DECOMPOSITION_ACTION_KEY
        applied: set[int] = set()
        for entry in trace:
            # The dispatch-key prefix is implementation-specific
            # (``DesignProcessCapability.<dispatch_key>.<action_name>``);
            # the canonical name is always the final component.
            if not entry.action_key.endswith(action_key):
                continue
            if not entry.success or entry.blocked:
                continue
            kwargs = entry.parameters or {}
            if kwargs.get("dry_run") is not False:
                continue
            # ``create_decomposition``'s signature takes
            # ``parent_issue_number`` (the canonical name on the
            # action). Reads the LLM-provided kwarg directly; if the
            # action's signature is renamed, the audit-key constant
            # change in Change 4 catches it before this code runs.
            number = kwargs.get("parent_issue_number")
            if isinstance(number, int):
                applied.add(number)
        return frozenset(applied)

    @staticmethod
    def _extract_classified_non_decomposable(
        trace: RunCallTrace,
        execution_context: "PlanExecutionContext",
    ) -> frozenset[int]:
        """Issue numbers that ``classify_issues_decomposability``
        returned with ``decomposable=False``.

        Read from ``execution_context.action_results`` (the FULL output
        dict, not the truncated trace preview) keyed off the
        successful ``classify_issues_decomposability`` entries' action
        IDs. Output shape:
        ``{"classifications": [{"number", "decomposable", "reason"}, ...]}``.
        """

        action_key = (
            DesignProcessCapability.CLASSIFY_ISSUES_DECOMPOSABILITY_ACTION_KEY
        )
        non_decomposable: set[int] = set()
        # The trace + action_results are indexed by different keys
        # (call_index vs action_id), so we walk the trace to find
        # successful classify entries and look up their action_results
        # via the step_summaries indirection.
        step_summaries: dict[str, Any] = execution_context.custom_data.get(
            "codegen_step_summaries", {},
        ) or {}
        for info in step_summaries.values():
            if not isinstance(info, dict):
                continue
            actions_called = info.get("actions_called") or []
            action_ids = info.get("action_ids") or []
            for ak, aid in zip(actions_called, action_ids):
                if not isinstance(ak, str) or not isinstance(aid, str):
                    continue
                if not ak.endswith(action_key):
                    continue
                result = execution_context.action_results.get(aid)
                if result is None or not getattr(result, "success", False):
                    continue
                output = getattr(result, "output", None)
                if not isinstance(output, dict):
                    continue
                classifications = output.get("classifications") or []
                for item in classifications:
                    if not isinstance(item, dict):
                        continue
                    if item.get("decomposable") is False:
                        n = item.get("number")
                        if isinstance(n, int):
                            non_decomposable.add(n)
        return frozenset(non_decomposable)

    async def _resolve_scope(
        self, agent: "Agent", params: dict[str, Any],
    ) -> frozenset[int]:
        explicit = params.get(ProjectPlanningCoordinator.ISSUE_NUMBERS_PARAM_NAME)
        if isinstance(explicit, (list, tuple)):
            return frozenset(int(n) for n in explicit if isinstance(n, int))
        # Spec'd fallback: "all currently-open roadmap issues at
        # mission spawn". Snapshot was stored in agent.metadata.parameters
        # by spawn_mission OR resolved at validator-init time. Until
        # the spawn-time snapshot is wired, fall back to an EMPTY
        # scope and rely on early-stop or the LLM to recover. We
        # explicitly do NOT call GitHub here — that would re-resolve
        # mid-run, violating the snapshot-at-spawn invariant.
        return frozenset()

    async def _read_early_stop(self, agent: "Agent") -> bool:
        """True iff ``request_decompose_early_stop`` has been recorded
        for this mission instance."""

        try:
            bb = await agent.get_blackboard()
        except Exception:  # noqa: BLE001 — defensive: validator must not crash on bb hiccups
            return False
        try:
            payload = await bb.read(
                DecomposeEarlyStopProtocol.signal_key(agent.agent_id),
            )
        except Exception:  # noqa: BLE001 — read miss is "no early-stop"
            return False
        return bool(payload)

    async def _publish_drain_state(
        self, agent: "Agent", tracker: DecomposeBacklogTracker,
    ) -> None:
        """Snapshot the drain state on every rejection so postmortems
        can replay what the validator believed."""

        self._rejection_sequence += 1
        try:
            bb = await agent.get_blackboard()
        except Exception:  # noqa: BLE001
            return
        try:
            await bb.write(
                _drain_state_key(agent.agent_id, self._rejection_sequence),
                {
                    "agent_id": agent.agent_id,
                    "sequence": self._rejection_sequence,
                    "in_scope": sorted(tracker.in_scope),
                    "applied": sorted(tracker.applied),
                    "classified_non_decomposable": sorted(
                        tracker.classified_non_decomposable,
                    ),
                    "remaining": sorted(tracker.remaining()),
                    "max_parents_per_run": tracker.max_parents_per_run,
                    "early_stopped": tracker.early_stopped,
                },
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "DecomposeCompletionValidator: failed to publish drain state",
                exc_info=True,
            )


__all__ = (
    "DecomposeBacklogTracker",
    "DecomposeCompletionValidator",
)
