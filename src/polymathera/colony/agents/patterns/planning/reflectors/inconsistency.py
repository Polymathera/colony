"""Cross-action inconsistency reflector.

Subsumes the prior pair ``InconsistencyDetector`` + ``InconsistencyAdvisor``.
Fires when, in the SAME iteration, the LLM emitted a
``respond_to_user`` containing failure-language alongside a
``spawn_mission`` that returned a success outcome (``spawned`` /
``return_existing``). The advisory renders a literal correction snippet
naming the actual ``agent_id``; the diagnostic carries the underlying
pair for cross-agent monitoring."""

from __future__ import annotations

from typing import Any, ClassVar

from ..models import (
    AdvisoryEntry,
    Diagnostic,
    IterationObservation,
    ReflectMoment,
    StreamReflection,
)
from ..reflection import (
    StreamReflector,
)


_SPAWN_SUCCESS_OUTCOMES = frozenset({"spawned", "return_existing"})


_FAILURE_LANGUAGE_PHRASES: tuple[str, ...] = (
    "spawn failed",
    "failed to spawn",
    "could not spawn",
    "unknown error",
    "❌",
    "infrastructure issue",
    "coordinator failed",
    "failed multiple times",
)
"""Phrases that, in a ``respond_to_user`` content sent in the same
iteration as a successful spawn, signal the LLM misread the spawn
result. Conservative list — false positives are noisy, so phrases
must be ones a planner would only emit under the misread shape."""


class InconsistencyReflector(StreamReflector):
    """Cross-action F1 backstop. Emits one advisory + diagnostic per
    inconsistent (spawn_success, respond_to_user-failure-language) pair
    detected in the iteration's actions."""

    name = "inconsistency_spawn"

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

        spawn_calls = []
        for rec in observation.actions_called:
            if not rec.action_key.endswith(".spawn_mission"):
                continue
            if rec.status != "ok" or not isinstance(rec.result, dict):
                continue
            if rec.result.get("outcome") not in _SPAWN_SUCCESS_OUTCOMES:
                continue
            spawn_calls.append(rec)
        if not spawn_calls:
            return StreamReflection()

        failure_responses = []
        for rec in observation.actions_called:
            if not rec.action_key.endswith(".respond_to_user"):
                continue
            content = rec.params.get("content") if rec.params else None
            if not isinstance(content, str):
                continue
            lower = content.lower()
            for phrase in _FAILURE_LANGUAGE_PHRASES:
                if phrase in lower:
                    failure_responses.append((rec, phrase))
                    break
        if not failure_responses:
            return StreamReflection()

        advisories: list[AdvisoryEntry] = []
        diagnostics: list[Diagnostic] = []
        for spawn in spawn_calls:
            outcome = spawn.result.get("outcome")
            agent_id = spawn.result.get("agent_id")
            mission_type = spawn.result.get("mission_type") or ""
            for resp, phrase in failure_responses:
                content = resp.params.get("content", "")
                excerpt = content[:240] if isinstance(content, str) else ""
                advisories.append(_build_advisory(
                    iter_index=observation.iter_index,
                    outcome=outcome,
                    agent_id=agent_id,
                    mission_type=mission_type,
                    excerpt=excerpt,
                ))
                diagnostics.append(Diagnostic(
                    kind="inconsistency_spawn_misread",
                    severity="alert",
                    payload={
                        "spawn_action_id": spawn.action_id,
                        "spawn_outcome": outcome,
                        "agent_id": agent_id,
                        "mission_type": mission_type,
                        "respond_action_id": resp.action_id,
                        "matched_phrase": phrase,
                        "content_excerpt": excerpt,
                    },
                ))
        return StreamReflection(
            advisories=advisories, diagnostics=diagnostics,
        )


def _build_advisory(
    *,
    iter_index: int,
    outcome: str | None,
    agent_id: str | None,
    mission_type: str,
    excerpt: str,
) -> AdvisoryEntry:
    outcome_str = outcome or "?"
    aid = agent_id or "<unknown>"
    mt = mission_type or "?"
    excerpt = excerpt.strip()
    excerpt_clause = (
        f"\n\nYour prior message excerpt:\n> {excerpt}"
        if excerpt else ""
    )
    body = (
        f"Iter {iter_index}: you told the user the spawn failed, but "
        f"`spawn_mission` for `{mt}` returned `outcome={outcome_str!r}` "
        f"with `agent_id={aid!r}` — the spawn SUCCEEDED (or the gate "
        f"handed back a running coordinator under `return_existing` "
        f"policy). The user got a misleading message."
        f"{excerpt_clause}\n\n"
        f"Recovery: send a brief correction to the user BEFORE any "
        f"other action this iteration, naming the actual `agent_id`. "
        f"Then proceed with the existing coordinator — do NOT call "
        f"`spawn_mission` again. On the next iteration, branch on "
        f"`out['outcome']` instead of `out['created']` to avoid this "
        f"class of misread."
    )
    next_action_code = (
        f"await run(\n"
        f"    \"SessionOrchestratorCapability."
        f"SessionOrchestratorCapability.respond_to_user\",\n"
        f"    content=(\n"
        f"        \"Correction: the {mt} coordinator IS running "
        f"(agent_id {aid!r}); my prior message was incorrect. \"\n"
        f"        \"Status checks are in progress.\"\n"
        f"    ),\n"
        f")"
    )
    return AdvisoryEntry(
        source="inconsistency_spawn",
        kind="spawn_misread_correction",
        body=body,
        next_action_code=next_action_code,
    )


__all__ = ("InconsistencyReflector",)
