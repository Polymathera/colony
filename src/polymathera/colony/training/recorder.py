"""Assemble recorded spans into :class:`TrainingRecord` instances.

Two views of the same spans:

- :func:`records_from_spans` — one ``sft`` record per ``INFER`` span (a
  prompt and the text it produced). This is the canonical SFT unit:
  every LLM call, with its exact in-context prompt, is one example.
- :func:`trajectory_records_from_spans` — one structured multi-turn
  ``sft`` record per agent run, with the agent's decisions as
  ``assistant`` turns (carrying the dispatched action as ``tool_calls``)
  interleaved with the tool results as ``tool`` turns. This is the
  function-calling trajectory the ACTION spans make possible.

Both are pure transformations: callers supply the spans (from whatever
store holds them) and receive records. Fidelity follows from the
capturing deployment's ``recording_grade`` setting — this module records
what the spans contain, untruncated or not.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..distributed.observability import Span, SpanKind
from .records import ChatTurn, RecordKind, TrainingRecord


SOURCE = "span_trajectory"


def records_from_spans(spans: Iterable[Span]) -> list[TrainingRecord]:
    """One ``sft`` :class:`TrainingRecord` per usable ``INFER`` span."""

    out: list[TrainingRecord] = []
    for span in spans:
        if span.kind is not SpanKind.INFER:
            continue
        prompt = span.input_summary.get("prompt")
        response = span.output_summary.get("response")
        if not prompt or not response:
            continue
        provenance = {
            "run_id": span.run_id,
            "agent_id": span.agent_id,
            "span_ids": [span.span_id],
        }
        if span.model_name:
            provenance["model_name"] = span.model_name
        out.append(
            TrainingRecord(
                kind=RecordKind.SFT,
                messages=(
                    ChatTurn(role="user", content=str(prompt)),
                    ChatTurn(role="assistant", content=str(response)),
                ),
                source=SOURCE,
                provenance=provenance,
            ),
        )
    return out


def _children_by_parent(spans: Iterable[Span]) -> dict[str, list[Span]]:
    by_parent: dict[str, list[Span]] = {}
    for span in spans:
        if span.parent_span_id:
            by_parent.setdefault(span.parent_span_id, []).append(span)
    return by_parent


def _first_child(
    by_parent: dict[str, list[Span]], parent_id: str, kind: SpanKind,
) -> Span | None:
    children = sorted(by_parent.get(parent_id, []), key=lambda s: s.start_wall)
    return next((s for s in children if s.kind is kind), None)


def _action_tool_calls(action: Span) -> tuple[dict[str, Any], ...]:
    summary = action.input_summary
    return ({
        "action_type": summary.get("action_type"),
        "parameters": summary.get("parameters", {}),
    },)


def _action_result_text(action: Span) -> str:
    out = action.output_summary
    parts: list[str] = []
    if "success" in out:
        parts.append(f"success={out['success']}")
    if out.get("error"):
        parts.append(f"error={out['error']}")
    if out.get("output") is not None:
        parts.append(str(out["output"]))
    return "\n".join(parts) if parts else "(no output)"


def trajectory_records_from_spans(spans: Iterable[Span]) -> list[TrainingRecord]:
    """One structured multi-turn ``sft`` record per ``(run, agent)``.

    Walks ``AGENT_STEP → PLAN → (INFER, ACTION)`` in time order. The
    first step's prompt seeds the ``user`` turn; each step's INFER
    response is an ``assistant`` turn (with the dispatched action as
    ``tool_calls``), followed by the ACTION result as a ``tool`` turn.
    Trajectories with no usable step are skipped.
    """

    spans = list(spans)
    by_parent = _children_by_parent(spans)
    steps_by_owner: dict[tuple[str | None, str], list[Span]] = {}
    for step in sorted(
        (s for s in spans if s.kind is SpanKind.AGENT_STEP),
        key=lambda s: s.start_wall,
    ):
        steps_by_owner.setdefault((step.run_id, step.agent_id), []).append(step)

    out: list[TrainingRecord] = []
    for (run_id, agent_id), steps in steps_by_owner.items():
        turns: list[ChatTurn] = []
        first_prompt: str | None = None
        step_count = 0
        for step in steps:
            plan = _first_child(by_parent, step.span_id, SpanKind.PLAN)
            if plan is None:
                continue
            infer = _first_child(by_parent, plan.span_id, SpanKind.INFER)
            if infer is None:
                continue
            response = infer.output_summary.get("response")
            if not response:
                continue
            prompt = infer.input_summary.get("prompt")
            if first_prompt is None and prompt:
                first_prompt = str(prompt)
            action = _first_child(by_parent, plan.span_id, SpanKind.ACTION)
            turns.append(ChatTurn(
                role="assistant",
                content=str(response),
                tool_calls=_action_tool_calls(action) if action is not None else None,
            ))
            if action is not None:
                turns.append(ChatTurn(role="tool", content=_action_result_text(action)))
            step_count += 1

        if not turns or first_prompt is None:
            continue
        out.append(TrainingRecord(
            kind=RecordKind.SFT,
            messages=(ChatTurn(role="user", content=first_prompt), *turns),
            source=SOURCE,
            provenance={"run_id": run_id, "agent_id": agent_id, "step_count": step_count},
        ))
    return out


__all__ = ("SOURCE", "records_from_spans", "trajectory_records_from_spans")
