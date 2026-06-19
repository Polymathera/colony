"""Final-iterations soft landing reflector.

Subsumes the prior ``CliffGuardAdvisor``. When the iteration budget is
within ``lead_iterations`` of ``max_iterations``, emits a
``final_iterations`` advisory telling the LLM to ship the highest-impact
action or stop cleanly.

Does NOT emit the block-streak diagnostic — that cross-agent signal is
owned by :class:`BlockStreakTracker` directly (the prior
``BlockStreakDetector`` mirrored tracker state into an in-process Signal
that no advisor consumed; retiring it is pure deletion)."""

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


_DEFAULT_LEAD = 2
"""Iterations before the cap at which the cliff-guard advisory starts
firing. Two gives the LLM one iteration to act + one to call
signal_completion cleanly."""


class CliffGuardReflector(StreamReflector):
    """Emits the final-iterations advisory when the iteration budget
    is within ``lead_iterations`` of ``max_iterations``."""

    name = "cliff_guard"

    REFLECT_AT: ClassVar[frozenset[ReflectMoment]] = frozenset(
        {"iteration_boundary"},
    )

    def __init__(
        self,
        *,
        max_iterations: int,
        lead_iterations: int = _DEFAULT_LEAD,
    ) -> None:
        self._max = max_iterations
        self._lead = lead_iterations

    def reflect(
        self,
        *,
        entries: list[dict[str, Any]],  # noqa: ARG002
        observation: IterationObservation | None,
        moment: ReflectMoment,  # noqa: ARG002
    ) -> StreamReflection:
        if observation is None:
            return StreamReflection()
        about_to_plan = observation.iter_index + 1
        remaining = self._max - about_to_plan + 1
        if not (0 < remaining <= self._lead):
            return StreamReflection()
        return StreamReflection(
            advisories=[_build_cliff_advisory(
                remaining=remaining, max_iterations=self._max,
            )],
        )


def _build_cliff_advisory(
    *,
    remaining: int,
    max_iterations: int,
) -> AdvisoryEntry:
    word = "iteration" if remaining == 1 else "iterations"
    body = (
        f"You have {remaining} {word} remaining (including this one) "
        f"before max_iterations={max_iterations} forces a hard stop. "
        f"Past that, the agent terminates with no graceful summary — "
        f"the operator sees only a partial trace.\n\n"
        f"Pick ONE:\n"
        f"(a) Ship the single highest-impact pending action this "
        f"iteration. If you have a proposal / decomposition / edit "
        f"ready to apply, apply it now.\n"
        f"(b) Stop the run cleanly: call your mission's terminal-stop "
        f"primitive (the one named in your goal block, e.g. "
        f"request_decompose_early_stop), then await signal_completion() "
        f"to close with a final summary.\n\n"
        f"Avoid investigation, status-checking, or new lines of "
        f"inquiry — there isn't budget for them."
    )
    return AdvisoryEntry(
        source="cliff_guard",
        kind="final_iterations",
        body=body,
        next_action_code=None,
    )


__all__ = ("CliffGuardReflector",)
