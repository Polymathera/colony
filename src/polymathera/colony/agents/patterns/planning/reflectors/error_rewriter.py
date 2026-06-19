"""Rule-based error-to-advisory reflector.

Subsumes the prior ``ErrorRewriterAdvisor``. Every iteration's
:class:`CallRecord` list is scanned against a list of
:class:`RewriteRule` instances. A rule matches when its
``matches(record)`` predicate returns ``True``; the matching rule's
``build(record)`` produces one :class:`AdvisoryEntry` the LLM sees on
the next iteration.

Rules are plugin-shape rather than a static ``(action_key,
error_pattern) → handler`` dict because the failure shapes we want to
surface aren't all framework-error-string shaped — F2's
``propose_decompositions`` fails at the domain level with
``success=True`` + ``result.ok=False``; F5's ``request_human_approval``
empty-body raises a typed exception with a known class name. Each
rule's predicate inspects whatever shape its capability emits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from ....models import ErrorContext
from ..models import (
    AdvisoryEntry,
    CallRecord,
    IterationObservation,
    ReflectMoment,
    StreamReflection,
)
from ..reflection import (
    StreamReflector,
)


_OUTER_ACTION_KEY = "<outer:execute_code>"
"""Synthetic ``CallRecord.action_key`` for outer-action failures
(the wrapping ``execute_code`` raised). Distinct prefix so rules can
opt into matching ``startswith("<outer:")`` if they want to fire only
on outer-action errors."""


@dataclass(frozen=True)
class RewriteRule:
    """One pattern → advisory mapping registered on the reflector.

    ``matches`` reads only the record (no observation / state). If a
    future rule needs cross-record context, the rule signature can be
    widened then; for now keeping it small keeps rules independently
    testable as pure functions."""

    name: str
    matches: Callable[[CallRecord], bool]
    build: Callable[[CallRecord], AdvisoryEntry | None]


class ErrorRewriterReflector(StreamReflector):
    """Iterates the registered rules over every action in the
    observation (plus the synthetic outer-error record when present),
    collecting advisories from the matching rules.

    Rules are independent — multiple rules can match a single CallRecord
    and all of their advisories surface in the next iteration's prompt.
    Order is registration order."""

    name = "error_rewriter"

    REFLECT_AT: ClassVar[frozenset[ReflectMoment]] = frozenset(
        {"iteration_boundary"},
    )

    def __init__(self, rules: list[RewriteRule] | None = None) -> None:
        self._rules: list[RewriteRule] = (
            list(rules) if rules is not None else list(DEFAULT_RULES)
        )

    def register(self, rule: RewriteRule) -> None:
        self._rules.append(rule)

    def reflect(
        self,
        *,
        entries: list[dict[str, Any]],  # noqa: ARG002
        observation: IterationObservation | None,
        moment: ReflectMoment,  # noqa: ARG002
    ) -> StreamReflection:
        if observation is None:
            return StreamReflection()

        records: list[Any] = list(observation.actions_called)
        if observation.outer_action_error is not None:
            records.append(
                _synthetic_outer_record(observation.outer_action_error),
            )
        advisories: list[AdvisoryEntry] = []
        for rec in records:
            for rule in self._rules:
                if not rule.matches(rec):
                    continue
                entry = rule.build(rec)
                if entry is not None:
                    advisories.append(entry)
        return StreamReflection(advisories=advisories)


def _synthetic_outer_record(err: ErrorContext) -> CallRecord:
    msg = err.error_details.get("message") if err.error_details else None
    ctx = err.action_context or {}
    return CallRecord(
        action_key=_OUTER_ACTION_KEY,
        params={},
        action_id=str(ctx.get("action_id") or ""),
        status="error",
        error=msg or "",
    )


# ---------------------------------------------------------------------------
# Initial rules
# ---------------------------------------------------------------------------


def _matches_propose_decomp_failure(rec: CallRecord) -> bool:
    if not rec.action_key.endswith(".propose_decompositions"):
        return False
    if rec.status != "ok" or not isinstance(rec.result, dict):
        return False
    if rec.result.get("ok") is False:
        return True
    proposals = rec.result.get("parent_proposals") or []
    return any(
        isinstance(p, dict) and p.get("error") for p in proposals
    )


def _build_propose_decomp_advisory(rec: CallRecord) -> AdvisoryEntry | None:
    result = rec.result if isinstance(rec.result, dict) else {}
    proposals = result.get("parent_proposals") or []
    successes = [
        p for p in proposals
        if isinstance(p, dict)
        and "children" in p
        and not p.get("error")
    ]
    failures = [
        p for p in proposals
        if isinstance(p, dict) and p.get("error")
    ]
    success_count = len(successes)
    failure_numbers = [
        p.get("parent_number") for p in failures
        if p.get("parent_number") is not None
    ]
    if success_count == 0 and not failure_numbers:
        body = (
            "`propose_decompositions` reported failure with no usable "
            "per-parent results. Either retry one parent at a time, "
            "or stop the run and report the failure to the operator."
        )
        next_code = (
            "# Either: retry per-parent (replace [N] with one of the\n"
            "# decomposable issue numbers from your classification):\n"
            "await run(\"DesignProcessCapability.DesignProcessCapability."
            "propose_decompositions\",\n"
            "    parent_issue_numbers=[N], include_design_context=True)\n"
            "# Or: stop the run:\n"
            "await run(\"ProjectPlanningMissionControlCapability."
            "ProjectPlanningMissionControlCapability."
            "request_decompose_early_stop\",\n"
            "    user_acknowledgement_quote=\"<the operator's exact words "
            "asking to abort>\")\n"
            "await signal_completion()"
        )
    elif success_count == 0:
        body = (
            f"`propose_decompositions` returned 0 successful proposals; "
            f"parents {failure_numbers} failed. Retry the failed "
            f"parents one at a time — single-parent cohorts avoid the "
            f"output-budget overflow that drops large cohorts."
        )
        next_code = (
            f"await run(\"DesignProcessCapability.DesignProcessCapability."
            f"propose_decompositions\",\n"
            f"    parent_issue_numbers=[{failure_numbers[0]}], "
            f"include_design_context=True)"
        )
    else:
        body = (
            f"`propose_decompositions` succeeded for "
            f"{success_count} parent(s) and failed for {failure_numbers}. "
            f"Proceed with the successful proposals via "
            f"`request_human_approval` + `create_decomposition`, and "
            f"retry the failed parents separately."
        )
        next_code = (
            "# Build the approval body from the SUCCESSFUL proposals only,\n"
            "# then retry failed parents one at a time afterwards."
        )
    return AdvisoryEntry(
        source="error_rewriter",
        kind="propose_decompositions_partial_failure",
        body=body,
        next_action_code=next_code,
    )


def _matches_human_approval_empty(rec: CallRecord) -> bool:
    if not rec.action_key.endswith(".request_human_approval"):
        return False
    if rec.status != "error":
        return False
    err = (rec.error or "").lower()
    return "empty" in err or "requesthumanapprovalempty" in err


def _build_human_approval_empty_advisory(
    rec: CallRecord,  # noqa: ARG001
) -> AdvisoryEntry | None:
    body = (
        "Your `request_human_approval` call was rejected because the "
        "question body had nothing for the operator to evaluate. "
        "Either build the body from the proposals you already computed "
        "before re-requesting approval, or — if upstream proposals "
        "failed entirely — stop the run cleanly."
    )
    next_code = (
        "# Path A: build a non-empty body from existing proposals.\n"
        "question = \"## Proposed changes\\n\" + render_proposals(\n"
        "    results[\"parent_proposals\"]\n"
        ")\n"
        "await run(\"HumanApprovalCapability.HumanApprovalCapability."
        "request_human_approval\",\n"
        "    question=question, extra={\"proposals\": "
        "results[\"parent_proposals\"]})\n"
        "\n"
        "# Path B: stop the run when proposals failed.\n"
        "await run(\"ProjectPlanningMissionControlCapability."
        "ProjectPlanningMissionControlCapability."
        "request_decompose_early_stop\",\n"
        "    user_acknowledgement_quote=\"<the operator's exact words "
        "asking to stop>\")\n"
        "await signal_completion()"
    )
    return AdvisoryEntry(
        source="error_rewriter",
        kind="human_approval_empty_body",
        body=body,
        next_action_code=next_code,
    )


def _matches_outer_key_error(rec: CallRecord) -> bool:
    if not rec.action_key.startswith("<outer:"):
        return False
    if rec.status != "error":
        return False
    return "KeyError" in (rec.error or "")


def _build_outer_key_error_advisory(
    rec: CallRecord,  # noqa: ARG001
) -> AdvisoryEntry | None:
    body = (
        "Your generated code raised a KeyError on `results` "
        "(or a similarly-keyed dict). Bare `results[<literal>]` "
        "lookups are fragile: an earlier iteration may not have "
        "stored that key, or the key may have been written under a "
        "different name. Switch to `.get(key, default)` for any "
        "dict access where you are not sure the key was set in a "
        "prior iteration. If the missing key is load-bearing for "
        "this iteration's work, log the available keys before "
        "branching so the next iteration's digest shows what IS "
        "in the dict."
    )
    next_code = (
        "# Safe-access pattern:\n"
        "rows = results.get(\"rows\")\n"
        "if rows is None:\n"
        "    log(\"results keys: \" + \", \".join(sorted(results)))\n"
        "    # decide based on what IS in results, or fall back to a\n"
        "    # query that re-establishes the data.\n"
        "    rows = []  # or fetch via the appropriate primitive\n"
    )
    return AdvisoryEntry(
        source="error_rewriter",
        kind="outer_key_error",
        body=body,
        next_action_code=next_code,
    )


DEFAULT_RULES: tuple[RewriteRule, ...] = (
    RewriteRule(
        name="propose_decompositions_partial_failure",
        matches=_matches_propose_decomp_failure,
        build=_build_propose_decomp_advisory,
    ),
    RewriteRule(
        name="request_human_approval_empty_body",
        matches=_matches_human_approval_empty,
        build=_build_human_approval_empty_advisory,
    ),
    RewriteRule(
        name="outer_key_error",
        matches=_matches_outer_key_error,
        build=_build_outer_key_error_advisory,
    ),
)


__all__ = (
    "DEFAULT_RULES",
    "ErrorRewriterReflector",
    "RewriteRule",
)
