"""Approval-advance reflector.

Subsumes the prior pair ``ApprovalResponseReadyDetector`` +
``ApprovalAdvanceAdvisor``. Fires when ``get_response`` returned
``state="ready"`` in the iteration's actions — meaning the operator has
answered. Each ready-response gets one ``advance_past_approval``
advisory naming the choice, so the LLM stops polling and branches.

Repeats every iteration the precondition persists (the LLM might
re-call ``get_response`` for the same id) — same nudging behavior the
prior detector had."""

from __future__ import annotations

from typing import Any, ClassVar

from ..models import (
    AdvisoryEntry,
    IterationObservation,
    ReflectMoment,
    StreamReflection,
)
from ..reflection import (
    StreamReflector,
)


class ApprovalAdvanceReflector(StreamReflector):
    """Per-iteration scan: one advisory per ``get_response`` returning
    ``state="ready"`` in the observation's actions."""

    name = "approval_advance"

    REFLECT_AT: ClassVar[frozenset[ReflectMoment]] = frozenset(
        {"iteration_boundary"},
    )

    def reflect(
        self,
        *,
        entries: list[dict[str, Any]],  # noqa: ARG002
        observation: IterationObservation | None,
        moment: ReflectMoment,  # noqa: ARG002
    ) -> StreamReflection:
        if observation is None:
            return StreamReflection()

        advisories: list[AdvisoryEntry] = []
        for rec in observation.actions_called:
            if not rec.action_key.endswith(".get_response"):
                continue
            if rec.status != "ok" or not isinstance(rec.result, dict):
                continue
            if rec.result.get("state") != "ready":
                continue
            response = rec.result.get("response")
            if not isinstance(response, dict):
                continue
            request_id = (
                rec.params.get("request_id") if rec.params else None
            ) or "<unknown>"
            choice = response.get("choice") or "?"
            explanation = response.get("explanation", "") or ""
            advisories.append(_build_advisory(
                request_id=request_id,
                choice=choice,
                explanation=explanation,
            ))
        return StreamReflection(advisories=advisories)


def _build_advisory(
    *,
    request_id: str,
    choice: str,
    explanation: str,
) -> AdvisoryEntry:
    expl_clause = (
        f", explanation={explanation!r}" if explanation else ""
    )
    body = (
        f"The approval response for request_id={request_id!r} is "
        f"already terminal: choice={choice!r}{expl_clause}. DO NOT "
        f"call `get_response` for this id again — it returns the same "
        f"envelope on every poll. Your next action MUST branch on "
        f"`choice`:\n"
        f"- `approve_once`: re-call the gated action with "
        f"`dry_run=False`, then proceed to the next pending decision "
        f"(or `signal_completion()` if none).\n"
        f"- `approve_all`: same as `approve_once` AND store an "
        f"`approve_all=True` flag in `results` so subsequent gated "
        f"actions skip their own approval round.\n"
        f"- `reject`: `respond_to_user` with the rejection reason, "
        f"then call your mission's terminal-stop primitive followed "
        f"by `signal_completion()`.\n"
        f"- `abort`: same as `reject`, quoting the operator's abort "
        f"explanation as the user_acknowledgement_quote."
    )
    return AdvisoryEntry(
        source="approval_advance",
        kind="advance_past_approval",
        body=body,
    )


__all__ = ("ApprovalAdvanceReflector",)
