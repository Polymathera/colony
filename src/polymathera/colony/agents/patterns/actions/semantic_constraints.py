"""Semantic-constraint primitives for agent runtime guardrails.

Declarative :class:`SemanticConstraint` records name semantic
preconditions on agent actions, paired with pluggable verifiers
(LLM-judge / typed evidence / Python predicate), per-constraint
scope (cell / session) and per-constraint failure mode (block /
advise). Replaces the syntactic predicates the legacy Python
guardrails used to encode the same rules — every new rule used to
be a new guardrail class + tests + deploy; now a rule is one record
in a catalogue and the framework does the rest.

The catalogue is also the SINGLE SOURCE OF TRUTH for the
"## Active Semantic Constraints" section in the planner prompt
(see :meth:`SemanticConstraintGuardrail.planner_context_advisory`)
— the LLM composes a strategy that respects the active constraints,
the framework does not bake an ordering. This eliminates the drift
of hand-written self_concept prose that grows out of sync with the
runtime check.

See ``colony/MEMORY.md::no-syntactic-proxies-for-semantic-properties``
for the lesson that prompted this generalization and
``colony/MEMORY.md::primitives-not-pipelines`` for the framing
principle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Sequence

from .code_constraints import GuardrailDecision, RuntimeGuardrail
from ..planning.models import CallRecord
from ...base import Agent, ActionPolicy
from ....cluster.models import InferenceRequest, InferenceResponse


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scope + failure mode
# ---------------------------------------------------------------------------


class ConstraintScope(str, Enum):
    """How far back the constraint's verifier sees agent state."""

    CELL = "cell"
    """Calls made within the CURRENT generated code block only. Right
    for rules that need fresh evidence: e.g. ``get_agent_status`` must
    have been called THIS cell before claiming the agent's state.
    Evidence from prior iterations may be stale and is rejected."""

    SESSION = "session"
    """The agent's full ``call_history`` (across iterations). Right
    for rules where evidence remains valid: e.g. an operator approval
    granted three iterations ago still satisfies the rule, an LLM
    classification of a decomposition target carries forward."""


class ConstraintFailureMode(str, Enum):
    """What the framework does when a constraint reports unsatisfied."""

    BLOCK = "block"
    """Block the action via ``GuardrailDecision(allowed=False, ...)``.
    The atomic-abort path (see ``_handle_guardrail_block``) raises
    :class:`GuardrailBlockedError`, the cell unwinds, the matching
    :class:`BlockedDispatch` carries the verdict into the next
    iteration's planner prompt. The right default for hard rules
    whose violation has irreversible side effects (state-claim
    fabrication, unauthorized apply, etc.)."""

    ADVISE = "advise"
    """Let the action through, but accumulate the verdict for the
    next iteration's planner prompt under a "Constraint Advisories
    (last iteration)" section. Right for soft / heuristic rules
    where the action's effect is reversible and a false-positive
    block would harm the workflow more than a false-negative."""


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintVerdict:
    """Verifier output. Frozen so the guardrail can pass it through
    failure-mode routing AND advisory accumulation without defensive
    copies."""

    satisfied: bool
    reason: str = ""
    suggestion: str = ""


# ---------------------------------------------------------------------------
# Scoped state — the slice of agent state a verifier sees
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScopedConstraintState:
    """Typed view the framework hands to each verifier, pre-sliced by
    the constraint's declared :class:`ConstraintScope` so the verifier
    never has to think about cell-vs-session boundaries.

    ``cell_calls`` is always a SUFFIX of ``session_calls`` (the calls
    since the current cell's plan_step boundary). Verifiers that
    declare ``scope=CELL`` read ``cell_calls``; ``scope=SESSION``
    read ``session_calls``.
    """

    cell_calls: tuple[CallRecord, ...]
    session_calls: tuple[CallRecord, ...]
    owner_agent_id: str | None
    """The owner agent's id (whose action policy is being guarded).
    ``None`` until ``bind_agent`` runs. Verifiers that need to
    exclude self-references (e.g. "don't claim state about an agent
    OTHER than yourself") read this; others ignore it."""


# ---------------------------------------------------------------------------
# Verifier ABC + three implementations
# ---------------------------------------------------------------------------


class ConstraintVerifier(ABC):
    """Pluggable mechanism for evaluating a :class:`SemanticConstraint`.

    Three default implementations:

    - :class:`LLMJudgeVerifier` — calls the agent's LLM with a structured
      prompt containing the constraint's NL rule + the proposed action +
      the scoped history. Right for novel semantic rules where structural
      checks miss either direction (false-positive / false-negative).
    - :class:`EvidenceVerifier` — satisfied iff a typed evidence record
      (a prior :class:`CallRecord`) matches a Python predicate. Right
      for "called X with these args" patterns; cheaper than LLM-judge.
    - :class:`PythonPredicateVerifier` — escape hatch when neither
      LLM-judge nor evidence-search fits the rule's shape.
    """

    @abstractmethod
    async def verify(
        self,
        *,
        constraint: "SemanticConstraint",
        action_key: str,
        action_params: dict[str, Any],
        scoped_state: ScopedConstraintState,
    ) -> ConstraintVerdict:
        """Return a verdict for the proposed action against this
        constraint, given the scoped state slice the framework already
        sliced. Must be self-contained — no reads from globals, no
        side effects (advisories are accumulated by the guardrail,
        not the verifier)."""
        ...

    @abstractmethod
    def describe_check(self) -> str:
        """One-line human-readable description of how this verifier
        evaluates the rule. Surfaced in the planner-prompt advisory
        so the LLM understands not just the rule but also what kind
        of evidence the framework looks for."""
        ...


class LLMJudgeVerifier(ConstraintVerifier):
    """LLM-judge: build a structured prompt with the rule + action +
    scoped history; ask the LLM for a typed verdict; parse.

    Fails OPEN on LLM error (allows the action) — same convention as
    :class:`LLMCompletionValidator`. An LLM hiccup must not silently
    block every action across the colony; loud-log + permissive
    default is the durable choice.

    The ``infer_fn`` is INJECTED at construction so the verifier
    doesn't reach into the agent globally. The catalogue is built
    before the agent exists (cloudpickled into the worker); the
    agent's :meth:`bind_agent` propagation calls :meth:`bind_infer`
    once the agent is alive. Tests pass a stub directly.
    """

    def __init__(
        self,
        *,
        infer_fn: Callable[..., Awaitable[InferenceResponse]] | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        cache_size: int = 128,
    ) -> None:
        self._infer_fn = infer_fn
        self._max_tokens = max_tokens
        self._temperature = temperature
        # Per-(rule, action, content, history) LRU.
        # Many judge calls on near-identical drafts can happen; the
        # cache collapses repeats to 1 LLM call. Per-verifier
        # instance, so each rule gets its own cache.
        self._cache: OrderedDict[str, ConstraintVerdict] = OrderedDict()
        self._cache_size = cache_size

    def bind_infer(
        self, infer_fn: Callable[..., Awaitable[InferenceResponse]],
    ) -> None:
        """Late-bind the LLM callable. Called from the guardrail's
        ``bind_agent`` propagation when the agent is alive."""

        self._infer_fn = infer_fn

    async def verify(
        self,
        *,
        constraint: "SemanticConstraint",
        action_key: str,
        action_params: dict[str, Any],
        scoped_state: ScopedConstraintState,
    ) -> ConstraintVerdict:
        # Observability: every judge call emits one structured INFO
        # line with constraint id, scope, proposed action, history
        # size, verdict, fallback flag, latency, and tokens. Without
        # this the only way to estimate the judge's per-action cost
        # is to grep ``Guardrail blocked`` lines (only the failures
        # surface) — every allowed action's judge call was invisible.
        start_wall = time.time()
        history = (
            scoped_state.cell_calls
            if constraint.scope == ConstraintScope.CELL
            else scoped_state.session_calls
        )

        if self._infer_fn is None:
            logger.warning(
                "LLMJudgeVerifier: no infer_fn bound; constraint %r "
                "fails open (allowed).",
                constraint.id,
            )
            verdict = ConstraintVerdict(
                satisfied=True, reason="(judge not bound — fail-open)",
            )
            self._emit_observability(
                constraint=constraint,
                action_key=action_key,
                history_len=len(history),
                verdict=verdict,
                latency_ms=(time.time() - start_wall) * 1000.0,
                tokens=None,
                fallback="unbound",
            )
            return verdict

        # Cache lookup before the LLM call. Same
        # (rule, action, content, scoped-history fingerprint) →
        # same verdict; skip the round-trip.
        cache_key = _judge_cache_key(
            constraint_id=constraint.id,
            action_key=action_key,
            action_params=action_params,
            history=history,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            self._emit_observability(
                constraint=constraint,
                action_key=action_key,
                history_len=len(history),
                verdict=cached,
                latency_ms=(time.time() - start_wall) * 1000.0,
                tokens=None,
                fallback="cache_hit",
            )
            return cached

        history_text = _render_history_for_judge(history)
        params_preview = _truncate_params_for_judge(action_params)

        owner_clause = (
            f"\nOWNER AGENT ID: {scoped_state.owner_agent_id}\n"
            if scoped_state.owner_agent_id else ""
        )
        prompt = (
            "You are a semantic constraint judge. Decide whether the "
            "proposed action satisfies the rule below, given the agent's "
            "recent action history.\n\n"
            f"RULE [{constraint.id}]: {constraint.rule_nl}\n"
            f"{owner_clause}\n"
            f"PROPOSED ACTION: {action_key}\n"
            f"PROPOSED PARAMS: {params_preview}\n\n"
            f"AGENT'S {constraint.scope.value.upper()}-SCOPED HISTORY "
            f"({len(history)} call(s); oldest first):\n{history_text}\n\n"
            "Reply with EXACTLY one JSON object on a single line:\n"
            '{"satisfied": <true|false>, "reason": "<one sentence>", '
            '"suggestion": "<one sentence — what should the agent do '
            'next so this rule is satisfied?>"}\n\n'
            "Default to satisfied=true if the rule does not apply to "
            "this action or you cannot tell from the available evidence."
        )
        tokens: int | None = None
        try:
            response = await self._infer_fn(
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            text = (
                response.generated_text
                if hasattr(response, "generated_text")
                else str(response)
            )
            if hasattr(response, "tokens_generated"):
                tokens = response.tokens_generated
        except Exception as e:
            logger.warning(
                "LLMJudgeVerifier: infer raised for constraint %r: %s. "
                "Allowing action (fail-open).",
                constraint.id, e,
            )
            verdict = ConstraintVerdict(
                satisfied=True, reason=f"judge call failed: {e!s}",
            )
            self._emit_observability(
                constraint=constraint,
                action_key=action_key,
                history_len=len(history),
                verdict=verdict,
                latency_ms=(time.time() - start_wall) * 1000.0,
                tokens=None,
                fallback=f"infer_error:{type(e).__name__}",
            )
            return verdict

        verdict = _parse_judge_response(text)
        # ``_parse_judge_response`` returns a fail-open verdict when
        # the LLM emits unparseable / malformed JSON; the reason
        # string is the canonical signal. Surface it in the fallback
        # field so a downstream grep can distinguish "judge said yes"
        # from "we couldn't read the judge's output."
        fallback: str | None = None
        if "not parseable" in verdict.reason:
            fallback = "parse_unparseable"
        elif "JSON malformed" in verdict.reason:
            fallback = "parse_malformed"

        # Cache only well-formed verdicts. Fail-open
        # parse errors are transient (LLM noise); we want a fresh
        # call next time so a temporarily-flaky LLM doesn't get
        # its bad answer pinned.
        if fallback is None:
            self._cache[cache_key] = verdict
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        self._emit_observability(
            constraint=constraint,
            action_key=action_key,
            history_len=len(history),
            verdict=verdict,
            latency_ms=(time.time() - start_wall) * 1000.0,
            tokens=tokens,
            fallback=fallback,
        )
        return verdict

    @staticmethod
    def _emit_observability(
        *,
        constraint: "SemanticConstraint",
        action_key: str,
        history_len: int,
        verdict: ConstraintVerdict,
        latency_ms: float,
        tokens: int | None,
        fallback: str | None,
    ) -> None:
        """One structured INFO line per judge call. Field order is
        stable so log adapters can parse cheaply (``key=value``,
        space-separated)."""

        logger.info(
            "[LLMJudge] constraint=%s scope=%s action=%s "
            "history_len=%d satisfied=%s latency_ms=%.1f tokens=%s "
            "fallback=%s reason=%s",
            constraint.id,
            constraint.scope.value,
            action_key,
            history_len,
            verdict.satisfied,
            latency_ms,
            tokens if tokens is not None else "n/a",
            fallback if fallback is not None else "none",
            verdict.reason[:120] if verdict.reason else "",
        )

    def describe_check(self) -> str:
        return (
            "evaluated by an LLM judge against the proposed action's "
            "params and the scoped call history"
        )


class EvidenceVerifier(ConstraintVerifier):
    """Satisfied iff at least one prior call in scope matches an
    evidence predicate.

    Cheaper than LLM-judge for rules where the agent ALREADY emits
    typed proof (a successful ``get_agent_status``, a typed approval
    record, a ``check_health`` result). The predicate gets each
    :class:`CallRecord` plus the proposed action context; the first
    matching record terminates the scan with ``satisfied=True``.

    Strictly more general than the legacy
    :class:`ArgsAwareTemporalOrderGuardrail` because the predicate
    runs over typed :class:`CallRecord` (not just action_key
    substring) and can inspect ``record.result`` to confirm the prior
    call actually SUCCEEDED with the right output — not just that it
    was attempted.
    """

    def __init__(
        self,
        *,
        evidence_predicate: Callable[
            [CallRecord, str, dict[str, Any], ScopedConstraintState],
            bool,
        ],
        description: str = "",
        unsatisfied_reason: str = "",
        unsatisfied_suggestion: str = "",
    ) -> None:
        self._predicate = evidence_predicate
        self._description = (
            description
            or "evidence search via Python predicate over scoped history"
        )
        self._unsatisfied_reason = unsatisfied_reason
        self._unsatisfied_suggestion = unsatisfied_suggestion

    async def verify(
        self,
        *,
        constraint: "SemanticConstraint",
        action_key: str,
        action_params: dict[str, Any],
        scoped_state: ScopedConstraintState,
    ) -> ConstraintVerdict:
        history = (
            scoped_state.cell_calls
            if constraint.scope == ConstraintScope.CELL
            else scoped_state.session_calls
        )
        for record in reversed(history):
            try:
                if self._predicate(
                    record, action_key, action_params, scoped_state,
                ):
                    return ConstraintVerdict(satisfied=True)
            except Exception as e:
                logger.warning(
                    "EvidenceVerifier: predicate raised for "
                    "constraint %r on record %r: %s",
                    constraint.id, record.action_key, e,
                )
        return ConstraintVerdict(
            satisfied=False,
            reason=(
                self._unsatisfied_reason
                or (
                    f"No evidence in {constraint.scope.value}-scoped "
                    f"history satisfies constraint {constraint.id!r}."
                )
            ),
            suggestion=self._unsatisfied_suggestion,
        )

    def describe_check(self) -> str:
        return self._description


class PythonPredicateVerifier(ConstraintVerifier):
    """Escape hatch — a Python callable returns a
    :class:`ConstraintVerdict` directly.

    Use only when neither LLM-judge (too expensive / no NL framing
    available) nor evidence-search (no clean evidence record) fits.
    The predicate signature mirrors :meth:`ConstraintVerifier.verify`
    inputs so a future migration to/from LLM-judge is mechanical.
    """

    def __init__(
        self,
        *,
        predicate: Callable[
            [str, dict[str, Any], ScopedConstraintState],
            ConstraintVerdict,
        ],
        description: str,
    ) -> None:
        self._predicate = predicate
        self._description = description

    async def verify(
        self,
        *,
        constraint: "SemanticConstraint",
        action_key: str,
        action_params: dict[str, Any],
        scoped_state: ScopedConstraintState,
    ) -> ConstraintVerdict:
        try:
            return self._predicate(action_key, action_params, scoped_state)
        except Exception as e:
            logger.warning(
                "PythonPredicateVerifier: predicate raised for "
                "constraint %r: %s. Allowing action (fail-open).",
                constraint.id, e,
            )
            return ConstraintVerdict(
                satisfied=True, reason=f"predicate raised: {e!s}",
            )

    def describe_check(self) -> str:
        return self._description


# ---------------------------------------------------------------------------
# Constraint record
# ---------------------------------------------------------------------------


@dataclass
class SemanticConstraint:
    """One declarative rule.

    The catalogue is just ``list[SemanticConstraint]``. Adding a new
    rule is one new record — no Python class, no test boilerplate, no
    deploy on the framework side. Fields:

    - ``id`` — short stable identifier the LLM and operator refer to.
      Surfaced in block messages, advisories, and the planner prompt.
    - ``rule_nl`` — natural-language rule. Used by LLM-judge as the
      judge prompt's RULE line AND by the advisory generator as the
      LLM-facing description.
    - ``applies_to`` — action_key substrings; the constraint is
      checked ONLY when the proposed action_key contains at least
      one match. Empty list matches nothing (an operator typo doesn't
      accidentally apply the rule everywhere).
    - ``scope`` — :class:`ConstraintScope` slice the verifier sees.
    - ``failure_mode`` — :class:`ConstraintFailureMode` on unsatisfied.
    - ``verifier`` — pluggable :class:`ConstraintVerifier`.
    """

    id: str
    rule_nl: str
    applies_to: list[str]
    verifier: ConstraintVerifier
    scope: ConstraintScope = ConstraintScope.CELL
    failure_mode: ConstraintFailureMode = ConstraintFailureMode.BLOCK
    # Cheap pre-filter run BEFORE the verifier. When
    # set, returns True iff the rule could possibly fire for this
    # action — False short-circuits to ``satisfied=True`` without
    # invoking the (expensive) verifier. The
    # ``no_unverified_agent_state_claims`` rule uses a regex match
    # for non-self ``agent-<hex>`` mentions; bare ``"Hi!"``
    # responses skip the LLM-judge entirely.
    precondition: Callable[
        [str, dict[str, Any], "ScopedConstraintState"], bool,
    ] | None = None

    def matches_action(self, action_key: str) -> bool:
        target = action_key.lower()
        return any(p.lower() in target for p in self.applies_to)


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------


class SemanticConstraintGuardrail(RuntimeGuardrail):
    """:class:`RuntimeGuardrail` wrapping a list of declarative
    :class:`SemanticConstraint` records.

    Composes with :class:`CompositeGuardrail` exactly like any other
    RuntimeGuardrail — the SessionAgent's guardrail factory mounts
    this alongside :class:`ApprovalRequiredGuardrail`, same shape,
    no new wiring.

    Per-action flow on :meth:`check`:

    1. Filter constraints to those whose ``applies_to`` matches the
       proposed ``action_key``. The common case (no match) returns
       ``allowed=True`` immediately — no slicing, no verifier calls.
    2. Build :class:`ScopedConstraintState` ONCE so all matching
       constraints share the slicing work.
    3. For each matching constraint: call the verifier; route the
       verdict by ``failure_mode``. First ``BLOCK`` wins (short
       circuits). ``ADVISE`` verdicts accumulate into
       ``_pending_advisories`` for the next iteration's prompt.

    The :meth:`planner_context_advisory` method renders BOTH the
    standing "Active Semantic Constraints" section AND the drained
    "Constraint Advisories (last iteration)" sub-section in one pass,
    so the catalogue is the SINGLE SOURCE OF TRUTH for what the
    planner sees about active constraints.
    """

    def __init__(
        self, constraints: Sequence[SemanticConstraint],
        *,
        escalation_threshold: int = 2,
    ) -> None:
        self._constraints: tuple[SemanticConstraint, ...] = tuple(
            constraints,
        )
        self._owner: Agent | None = None
        self._pending_advisories: list[
            tuple[SemanticConstraint, ConstraintVerdict]
        ] = []
        # Per-(rule, action-content) streak counter.
        # After ``escalation_threshold`` consecutive BLOCKs on the
        # SAME (rule_id, content-hash), downgrade THIS rule to
        # ADVISE for the rest of the agent's session so the LLM
        # stops retrying the same blocked draft forever. The
        # downgrade is per-rule, not global — other rules keep
        # blocking; only the stuck one yields. Run12: 46 blocks on
        # the same draft burned the iteration budget.
        self._block_streak: dict[tuple[str, str], int] = {}
        self._downgraded: set[str] = set()
        self._escalation_threshold = escalation_threshold
        # Operator runtime override. Disabled
        # constraints are skipped entirely (no precondition / no
        # verifier / no advisory). Manual counterpart to the
        # auto-downgrade above — useful when a constraint is
        # misbehaving in ways the streak detector can't catch
        # (e.g. wrong rule, off-target false-positives across
        # multiple drafts).
        # Operator-disabled constraint ids are NOT mirrored locally.
        # The session blackboard's
        # ``operator_override:semantic_constraint:<id>`` keys are the
        # single source of truth — the dashboard's POST writes there
        # and ``check()`` reads from there each iteration. No local
        # set means no rehydrate-on-respawn problem and no event
        # subscription to keep in sync.

    def bind_agent(self, agent: Agent | None) -> None:
        """Capture the owner AND late-bind any LLM-judge verifiers
        to the agent's ``infer`` method. Composite parents call this
        once at policy init."""

        self._owner = agent
        if agent is None:
            return
        for c in self._constraints:
            if isinstance(c.verifier, LLMJudgeVerifier):
                c.verifier.bind_infer(agent.infer)

    async def check(
        self,
        action_key: str,
        params: dict[str, Any],
        call_history: list[CallRecord],
        *,
        cell_start_index: int = 0,
    ) -> GuardrailDecision:
        applicable = [
            c for c in self._constraints if c.matches_action(action_key)
        ]
        if not applicable:
            return GuardrailDecision(allowed=True)

        disabled_ids = await self._read_disabled_ids()

        scoped = _build_scoped_state(
            call_history=call_history,
            cell_start_index=cell_start_index,
            owner=self._owner,
        )

        for constraint in applicable:
            # Operator override — disabled constraints
            # are skipped entirely. Checked FIRST so a disabled
            # constraint costs nothing (no precondition, no verifier,
            # no advisory).
            if constraint.id in disabled_ids:
                continue
            # Cheap precondition pre-filter. Returns
            # False → rule does not apply; skip the verifier
            # entirely (no LLM call). Defensive try/except so a
            # buggy predicate degrades to "run the verifier" rather
            # than crashing the check.
            if constraint.precondition is not None:
                try:
                    if not constraint.precondition(action_key, params, scoped):
                        continue
                except Exception as e:
                    logger.warning(
                        "SemanticConstraintGuardrail: precondition raised "
                        "for %r: %s. Falling through to verifier.",
                        constraint.id, e,
                    )

            verdict = await constraint.verifier.verify(
                constraint=constraint,
                action_key=action_key,
                action_params=params,
                scoped_state=scoped,
            )
            if verdict.satisfied:
                # Reset the streak for this (rule, content) — a
                # successful pass means the LLM course-corrected.
                self._block_streak.pop(
                    (constraint.id, _content_fingerprint(params)),
                    None,
                )
                continue

            # Pick the effective failure mode FIRST
            # (downgraded constraints get ADVISE for this call AND
            # subsequent ones), then update the streak / mark for
            # downgrade. This way the call that REACHES the
            # threshold still blocks; the FOLLOWING call is the
            # first allowed-through-advisory.
            effective_mode = (
                ConstraintFailureMode.ADVISE
                if constraint.id in self._downgraded
                else constraint.failure_mode
            )
            if (
                effective_mode == ConstraintFailureMode.BLOCK
                and constraint.id not in self._downgraded
            ):
                key = (constraint.id, _content_fingerprint(params))
                self._block_streak[key] = self._block_streak.get(key, 0) + 1
                if self._block_streak[key] >= self._escalation_threshold:
                    self._downgraded.add(constraint.id)
                    logger.warning(
                        "SemanticConstraintGuardrail: constraint %r "
                        "blocked %d consecutive times on the same "
                        "content; downgrading to ADVISE for the rest "
                        "of the session to break the death spiral.",
                        constraint.id, self._block_streak[key],
                    )

            if effective_mode == ConstraintFailureMode.BLOCK:
                return GuardrailDecision(
                    allowed=False,
                    reason=(
                        f"[{constraint.id}] {verdict.reason}"
                        if verdict.reason
                        else f"[{constraint.id}] constraint not satisfied."
                    ),
                    suggestion=verdict.suggestion,
                )
            # ADVISE (native or downgraded): surface in next iter; allow now.
            self._pending_advisories.append((constraint, verdict))

        return GuardrailDecision(allowed=True)

    def drain_pending_advisories(
        self,
    ) -> list[tuple[SemanticConstraint, ConstraintVerdict]]:
        """Atomically pull and clear ADVISE verdicts accumulated
        during the prior iteration's checks. Each verdict surfaces
        in the planner prompt EXACTLY ONCE."""

        drained = list(self._pending_advisories)
        self._pending_advisories.clear()
        return drained

    # ------------------------------------------------------------------
    # Operator runtime override — read live from the session BB
    # ------------------------------------------------------------------

    async def _read_disabled_ids(self) -> set[str]:
        """Snapshot the currently operator-disabled constraint ids
        from the session blackboard.

        The dashboard's POST writes ``operator_override:
        semantic_constraint:<id> = {"disabled": bool, ...}`` on the
        session-default scope; this method reads back the same keys.
        Reading at check-time (instead of mirroring into a local set
        and listening for change events) keeps a single source of
        truth — overrides survive SessionAgent respawn for free, and
        no in-memory state can drift from the BB.

        Read failures degrade open (returns the empty set) — losing
        operator overrides during a transient BB outage is strictly
        less bad than blocking every guardrailed action."""

        if self._owner is None:
            return set()
        # SESSION scope — same prefix the dashboard's POST writes at.
        # ``Agent.get_blackboard()`` with no ``scope_id`` defaults to
        # AGENT scope, which is a different prefix and would never see
        # the dashboard's writes.
        from ...scopes import BlackboardScope, get_scope_prefix
        from ...blackboard.protocol import OperatorOverrideProtocol
        try:
            bb = await self._owner.get_blackboard(
                scope_id=get_scope_prefix(BlackboardScope.SESSION),
            )
        except Exception as e:
            logger.warning(
                "SemanticConstraintGuardrail: get_blackboard failed "
                "while reading operator overrides: %s — treating as "
                "no overrides for this iteration.", e,
            )
            return set()
        if bb is None:
            return set()
        try:
            entries = await bb.query(
                namespace=OperatorOverrideProtocol.semantic_constraint_pattern(),
            )
        except Exception as e:
            logger.warning(
                "SemanticConstraintGuardrail: BB query for operator "
                "overrides failed: %s — treating as no overrides.", e,
            )
            return set()
        out: set[str] = set()
        for entry in entries:
            try:
                cid = OperatorOverrideProtocol.parse_semantic_constraint_key(
                    entry.key,
                )
            except ValueError:
                continue
            payload = entry.value if isinstance(entry.value, dict) else {}
            if bool(payload.get("disabled", False)):
                out.add(cid)
        return out

    def planner_context_advisory(
        self,
        call_history: list[CallRecord],
    ) -> str | None:
        """Auto-generated planner-prompt section combining:

        - The standing "Active semantic constraints" list (every
          record in the catalogue rendered with rule, scope,
          matching actions, failure mode, and check mechanism).
        - The drained "Constraint Advisories (last iteration)"
          sub-section for ADVISE verdicts the planner needs to
          react to next.

        This is the SINGLE SOURCE OF TRUTH for what the planner sees
        about active constraints — adding a new record in the catalogue
        automatically updates the prompt; the operator never edits
        the self_concept by hand for a guardrail concern.
        """

        sections: list[str] = []

        if self._constraints:
            lines = [
                "**Active semantic constraints.** Satisfy these "
                "BEFORE proposing the gated call so the runtime "
                "guardrail does not block + burn an iteration:",
            ]
            for c in self._constraints:
                applies_render = ", ".join(
                    f"``{a}``" for a in c.applies_to
                )
                scope_note = (
                    "within the SAME cell"
                    if c.scope == ConstraintScope.CELL
                    else "across iterations of the agent's session"
                )
                lines.append(
                    f"- **[{c.id}]** {c.rule_nl}\n"
                    f"  - Applies to: {applies_render}\n"
                    f"  - Scope: ``{c.scope.value}`` ({scope_note})\n"
                    f"  - On failure: ``{c.failure_mode.value}``\n"
                    f"  - Check: {c.verifier.describe_check()}"
                )
            sections.append("\n".join(lines))

        drained = self.drain_pending_advisories()
        if drained:
            advisory_lines = [
                "## Constraint Advisories (last iteration)",
                "These were ADVISE-mode constraints whose verifier "
                "reported unsatisfied during the prior cell. The "
                "actions still ran; consider whether to course-correct.",
            ]
            for c, v in drained:
                line = f"- **[{c.id}]** {v.reason}"
                if v.suggestion:
                    line += f"\n  - *Suggestion:* {v.suggestion}"
                advisory_lines.append(line)
            sections.append("\n".join(advisory_lines))

        if not sections:
            return None
        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_scoped_state(
    *,
    call_history: list[CallRecord],
    cell_start_index: int,
    owner: Agent,
) -> ScopedConstraintState:
    """Slice ``call_history`` once per :meth:`check` call so every
    applicable constraint shares the slicing work."""

    safe_start = max(0, min(cell_start_index, len(call_history)))
    owner_agent_id: str | None = None
    if owner is not None:
        owner_agent_id = owner.agent_id
    return ScopedConstraintState(
        cell_calls=tuple(call_history[safe_start:]),
        session_calls=tuple(call_history),
        owner_agent_id=owner_agent_id,
    )


def _render_history_for_judge(history: Sequence[CallRecord]) -> str:
    """Compact text rendering of recent calls for the judge prompt.

    Each line: ``[idx] action_key(params_preview) → status [result_preview]``.
    Aggressive truncation because the judge prompt fires per-action
    and bloated history multiplies token cost across every check.
    """

    if not history:
        return "(none)"
    lines = []
    for i, c in enumerate(history):
        params_preview = _truncate_params_for_judge(c.params)
        result_preview = ""
        if c.result is not None:
            r = str(c.result)
            result_preview = (
                f" [{r[:80]}" + ("…]" if len(r) > 80 else "]")
            )
        lines.append(
            f"[{i}] {c.action_key}({params_preview}) → "
            f"{c.status}{result_preview}"
        )
    return "\n".join(lines)


def _truncate_params_for_judge(params: dict[str, Any]) -> str:
    """Compact JSON params for the judge prompt; capped per-call."""

    try:
        text = json.dumps(params)
    except (TypeError, ValueError):
        text = str(params)
    if len(text) > 200:
        text = text[:200] + "…"
    return text


def _content_fingerprint(action_params: dict[str, Any]) -> str:
    """Compact hash of the proposed action params. Used by the
    streak counter to detect "same draft blocked twice in a row"
    without storing the full content."""

    try:
        blob = json.dumps(action_params, sort_keys=True, default=str)
    except (TypeError, ValueError):
        blob = str(action_params)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=12).hexdigest()


def _judge_cache_key(
    *,
    constraint_id: str,
    action_key: str,
    action_params: dict[str, Any],
    history: Sequence[CallRecord],
) -> str:
    """Stable cache key for :class:`LLMJudgeVerifier`. Hashes the
    proposed action + the scoped-history fingerprint (action keys +
    terminal status) so SAME draft + SAME history → cached verdict;
    a get_agent_status appearing in the cell flips the fingerprint
    and forces a fresh judge call."""

    try:
        params_blob = json.dumps(action_params, sort_keys=True, default=str)
    except (TypeError, ValueError):
        params_blob = str(action_params)
    history_blob = "|".join(
        f"{c.action_key}:{c.status}" for c in history
    )
    h = hashlib.blake2b(digest_size=16)
    h.update(constraint_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(action_key.encode("utf-8"))
    h.update(b"\x00")
    h.update(params_blob.encode("utf-8"))
    h.update(b"\x00")
    h.update(history_blob.encode("utf-8"))
    return h.hexdigest()


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_response(text: str) -> ConstraintVerdict:
    """Lenient JSON extraction — the LLM may wrap the verdict in
    prose, code fences, or extra fields. We accept the first
    top-level JSON object that has a ``satisfied`` key. Fails open
    on parse error (allows the action) with a loud warning, mirroring
    :class:`LLMCompletionValidator`'s convention."""

    m = _JSON_OBJECT_RE.search(text or "")
    if m is None:
        logger.warning(
            "LLMJudgeVerifier: judge output not parseable as JSON: %r",
            (text[:200] if text else ""),
        )
        return ConstraintVerdict(
            satisfied=True,
            reason="judge output not parseable; failing open",
        )
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        logger.warning(
            "LLMJudgeVerifier: judge JSON decode failed: %r",
            m.group(0)[:200],
        )
        return ConstraintVerdict(
            satisfied=True,
            reason="judge JSON malformed; failing open",
        )
    return ConstraintVerdict(
        satisfied=bool(obj.get("satisfied", True)),
        reason=str(obj.get("reason", "")),
        suggestion=str(obj.get("suggestion", "")),
    )


__all__ = (
    "ConstraintScope",
    "ConstraintFailureMode",
    "ConstraintVerdict",
    "ScopedConstraintState",
    "ConstraintVerifier",
    "LLMJudgeVerifier",
    "EvidenceVerifier",
    "PythonPredicateVerifier",
    "SemanticConstraint",
    "SemanticConstraintGuardrail",
)
