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
from dataclasses import dataclass, field
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
from polymathera.colony.agents.patterns.capabilities.github import (
    GitHubCapability,
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
    # Issues the classifier returned in its ``pre_filtered`` list:
    # already_decomposed parents (carry ``colony:decomposed-into``
    # marker) and children of a prior decomposition (carry
    # ``colony:parent-of`` marker). They count toward drain alongside
    # applied + classified_non_decomposable — the read-side marker
    # check is as authoritative as the writer that stamped it.
    pre_filtered: frozenset[int] = field(default_factory=frozenset)

    def remaining(self) -> frozenset[int]:
        """Issues still in scope and unaddressed."""

        if self.early_stopped:
            return frozenset()
        return (
            self.in_scope
            - self.applied
            - self.classified_non_decomposable
            - self.pre_filtered
        )

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
                    f"non_decomposable {len(tracker.classified_non_decomposable)}, "
                    f"pre_filtered {len(tracker.pre_filtered)}"
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
                f"Pre-filtered (already_decomposed): "
                f"{sorted(tracker.pre_filtered)}. "
                f"Remaining: {sorted(tracker.remaining())}."
            ),
        )

    async def _build_tracker(
        self,
        agent: "Agent",
        execution_context: "PlanExecutionContext",
        params: dict[str, Any],
    ) -> DecomposeBacklogTracker:
        in_scope = await self._resolve_scope(agent, execution_context)
        trace = self._build_trace(execution_context)
        applied = self._extract_applied(trace)
        classified_non = self._extract_classified_non_decomposable(
            trace, execution_context,
        )
        pre_filtered = self._extract_pre_filtered(execution_context)
        early_stopped = await self._read_early_stop(agent)
        cap = params.get(ProjectPlanningCoordinator.MAX_PARENTS_PER_RUN_PARAM_NAME)
        max_parents_per_run = int(cap) if isinstance(cap, int) else None
        return DecomposeBacklogTracker(
            in_scope=in_scope,
            applied=applied,
            classified_non_decomposable=classified_non,
            pre_filtered=pre_filtered,
            early_stopped=early_stopped,
            max_parents_per_run=max_parents_per_run,
        )

    @staticmethod
    def _build_trace(
        execution_context: "PlanExecutionContext",
    ) -> RunCallTrace:
        """Reconstruct the run-call trace by concatenating each step
        summary's ``run_call_trace`` in insertion order."""

        entries: list[dict[str, Any]] = []
        for info in execution_context.codegen_step_summaries.values():
            entries.extend(
                item for item in info.run_call_trace
                if isinstance(item, dict)
            )
        return RunCallTrace(entries)

    @staticmethod
    def _extract_applied(trace: RunCallTrace) -> frozenset[int]:
        """Parent issue numbers passed to ``create_decomposition`` with
        ``dry_run`` explicitly False. ``dry_run`` is the canonical
        gate — the LLM is required to pass it; an absent value is
        treated as "did not apply" (the conservative choice — drain
        condition slightly harder to meet, but no false-positive
        completion)."""

        action_key = DesignProcessCapability.create_decomposition._action_key
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

        Walks the action outputs the planner LLM produced via
        :meth:`PlanExecutionContext.iter_successful_action_outputs`
        (the trace-indexed read path is the single canonical accessor;
        the validator does not reach into ``custom_data`` or
        ``action_results`` directly). Output shape per call:
        ``{"classifications": [{"number", "decomposable", "reason"}, ...]}``.
        """

        action_key = (
            DesignProcessCapability.classify_issues_decomposability._action_key
        )
        non_decomposable: set[int] = set()
        for output in execution_context.iter_successful_action_outputs(
            action_key,
        ):
            if not isinstance(output, dict):
                continue
            for item in output.get("classifications") or []:
                if not isinstance(item, dict):
                    continue
                if item.get("decomposable") is False:
                    n = item.get("number")
                    if isinstance(n, int):
                        non_decomposable.add(n)
        return frozenset(non_decomposable)

    @staticmethod
    def _extract_pre_filtered(
        execution_context: "PlanExecutionContext",
    ) -> frozenset[int]:
        """Issue numbers the classifier reported in its ``pre_filtered``
        list — issues whose body carries the ``colony:decomposed-into``
        marker (kind=already_decomposed). The marker is stamped by
        ``create_decomposition`` on the parent; the classifier's
        structural check skips these to avoid burning an LLM call on
        a decision the operator already approved.

        These count toward drain alongside ``applied`` and
        ``classified_non_decomposable``: the structural marker check
        run inside ``classify_issues_decomposability`` is as
        authoritative as the writer (``create_decomposition``) that
        stamped the marker. Without this, a run that correctly skips
        N already-decomposed parents would fail the drain predicate —
        the validator would treat the pre-filtered issues as
        ``remaining`` and reject ``signal_completion`` indefinitely.

        Walks the same trace-indexed action outputs as
        :meth:`_extract_classified_non_decomposable`. Output shape per
        call: ``{"pre_filtered": [{"number", "kind", "reason", ...},
        ...]}``. The reason text is not consumed here; only the typed
        ``number`` field gates the drain math.
        """

        action_key = (
            DesignProcessCapability.classify_issues_decomposability._action_key
        )
        pre_filtered: set[int] = set()
        for output in execution_context.iter_successful_action_outputs(
            action_key,
        ):
            if not isinstance(output, dict):
                continue
            for item in output.get("pre_filtered") or []:
                if not isinstance(item, dict):
                    continue
                n = item.get("number")
                if isinstance(n, int):
                    pre_filtered.add(n)
        return frozenset(pre_filtered)

    async def _resolve_scope(
        self,
        agent: "Agent",
        execution_context: "PlanExecutionContext",
    ) -> frozenset[int]:
        """Return the in-scope issue-number set the validator gates
        completion on.

        Sole source of truth: the FIRST successful
        ``snapshot_open_roadmap_issues`` call recorded in the
        coordinator's run-call trace. The coordinator's LLM-generated
        code calls the snapshot as its first action per the
        ``project_planning`` decompose-mode goal block. Pinning the
        first successful call means a re-snapshot later in the run
        cannot silently extend or shrink scope.

        Raises ``RuntimeError`` when no successful snapshot is
        recorded — the rejection names the snapshot action as the
        missing prerequisite so the LLM's next iteration can recover.
        """

        action_key = (
            GitHubCapability.snapshot_open_roadmap_issues._action_key
        )
        for output in execution_context.iter_successful_action_outputs(
            action_key,
        ):
            if isinstance(output, list):
                return frozenset(n for n in output if isinstance(n, int))
        raise RuntimeError(
            f"DecomposeCompletionValidator: in-scope issue set is "
            f"unestablished — call ``GitHubCapability.{action_key}`` "
            f"as the first action of decompose mode so the validator "
            f"has a fixed in-scope set to drain against. "
            f"signal_completion() will not be accepted until the "
            f"snapshot is recorded."
        )

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
                    "pre_filtered": sorted(tracker.pre_filtered),
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
