"""Tests for :func:`trajectory_records_from_spans`."""

from __future__ import annotations

import itertools

from polymathera.colony.distributed.observability import Span, SpanKind
from polymathera.colony.training import RecordKind, trajectory_records_from_spans

_clock = itertools.count(1)


def _span(kind, *, parent=None, run="run-1", agent="a1", **kw) -> Span:
    return Span(
        trace_id="t", agent_id=agent, name=kind.value, kind=kind,
        run_id=run, parent_span_id=parent, start_wall=float(next(_clock)), **kw,
    )


def _step(prompt, response, *, action=None, run="run-1", agent="a1") -> list[Span]:
    """A full AGENT_STEP → PLAN → (INFER [, ACTION]) subtree."""
    step = _span(SpanKind.AGENT_STEP, run=run, agent=agent)
    plan = _span(SpanKind.PLAN, parent=step.span_id, run=run, agent=agent)
    infer = _span(
        SpanKind.INFER, parent=plan.span_id, run=run, agent=agent,
    )
    infer.input_summary = {"prompt": prompt}
    infer.output_summary = {"response": response}
    spans = [step, plan, infer]
    if action is not None:
        act = _span(SpanKind.ACTION, parent=plan.span_id, run=run, agent=agent)
        act.input_summary = {"action_type": action["type"], "parameters": action["params"]}
        act.output_summary = {"success": action.get("success", True), "output": action.get("output")}
        spans.append(act)
    return spans


def test_single_step_with_action_builds_user_assistant_tool() -> None:
    spans = _step("solve it", "calling tool", action={
        "type": "run_code", "params": {"x": 1}, "output": "42",
    })
    recs = trajectory_records_from_spans(spans)
    assert len(recs) == 1
    msgs = recs[0].to_sft()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[0]["content"] == "solve it"
    assert msgs[1]["content"] == "calling tool"
    assert msgs[1]["tool_calls"][0]["action_type"] == "run_code"
    assert "42" in msgs[2]["content"]
    assert recs[0].provenance == {"run_id": "run-1", "agent_id": "a1", "step_count": 1}


def test_multi_step_trajectory_in_time_order() -> None:
    spans = _step("task", "step one", action={"type": "a", "params": {}, "output": "r1"})
    spans += _step("task", "step two", action={"type": "b", "params": {}, "output": "r2"})
    recs = trajectory_records_from_spans(spans)
    assert len(recs) == 1
    roles = [m["role"] for m in recs[0].to_sft()["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "tool"]


def test_step_without_action_has_no_tool_turn() -> None:
    recs = trajectory_records_from_spans(_step("task", "thinking only"))
    roles = [m["role"] for m in recs[0].to_sft()["messages"]]
    assert roles == ["user", "assistant"]


def test_separate_agents_get_separate_trajectories() -> None:
    spans = _step("t", "a1 says", action={"type": "x", "params": {}}, agent="a1")
    spans += _step("t", "a2 says", action={"type": "y", "params": {}}, agent="a2")
    recs = trajectory_records_from_spans(spans)
    assert {r.provenance["agent_id"] for r in recs} == {"a1", "a2"}


def test_steps_without_infer_response_skipped() -> None:
    spans = _step("task", "")  # empty response → unusable
    assert trajectory_records_from_spans(spans) == []


def test_all_records_are_sft_kind() -> None:
    recs = trajectory_records_from_spans(_step("t", "r", action={"type": "x", "params": {}}))
    assert all(r.kind is RecordKind.SFT for r in recs)
