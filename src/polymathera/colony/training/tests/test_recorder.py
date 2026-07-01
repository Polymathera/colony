"""Tests for :func:`records_from_spans`."""

from __future__ import annotations

from polymathera.colony.distributed.observability import Span, SpanKind
from polymathera.colony.training import RecordKind, records_from_spans


def _infer(
    *,
    run_id: str = "run-1",
    agent_id: str = "a1",
    prompt: str | None = "p",
    response: str | None = "r",
    model_name: str | None = None,
) -> Span:
    span = Span(
        trace_id="t", agent_id=agent_id, name="infer", kind=SpanKind.INFER,
        run_id=run_id, model_name=model_name,
    )
    if prompt is not None:
        span.input_summary = {"prompt": prompt}
    if response is not None:
        span.output_summary = {"response": response}
    return span


def test_one_sft_record_per_infer_span() -> None:
    recs = records_from_spans([_infer(prompt="hi", response="yo")])
    assert len(recs) == 1
    rec = recs[0]
    assert rec.kind is RecordKind.SFT
    assert rec.to_sft() == {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ],
    }
    assert rec.source == "span_trajectory"
    assert rec.provenance["run_id"] == "run-1"
    assert rec.provenance["agent_id"] == "a1"
    assert rec.provenance["span_ids"] == [rec_span_id(recs)]


def rec_span_id(recs) -> str:  # noqa: ANN001
    return recs[0].provenance["span_ids"][0]


def test_non_infer_spans_skipped() -> None:
    action = Span(
        trace_id="t", agent_id="a1", name="dispatch", kind=SpanKind.ACTION,
        run_id="run-1",
    )
    assert records_from_spans([action]) == []


def test_infer_without_response_skipped() -> None:
    assert records_from_spans([_infer(response=None)]) == []


def test_infer_without_prompt_skipped() -> None:
    assert records_from_spans([_infer(prompt=None)]) == []


def test_model_name_recorded_in_provenance_when_present() -> None:
    recs = records_from_spans([_infer(model_name="m-1")])
    assert recs[0].provenance["model_name"] == "m-1"


def test_model_name_absent_from_provenance_when_unset() -> None:
    recs = records_from_spans([_infer()])
    assert "model_name" not in recs[0].provenance


def test_mixed_run_keeps_only_usable_infer_spans() -> None:
    spans = [
        _infer(prompt="a", response="1"),
        Span(trace_id="t", agent_id="a1", name="plan", kind=SpanKind.PLAN,
             run_id="run-1"),
        _infer(prompt="b", response=None),
        _infer(prompt="c", response="3"),
    ]
    recs = records_from_spans(spans)
    assert [r.to_sft()["messages"][0]["content"] for r in recs] == ["a", "c"]
