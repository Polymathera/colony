"""Tests for the curation primitives."""

from __future__ import annotations

import pytest

from polymathera.colony.training import (
    ChatTurn,
    RecordKind,
    TrainingRecord,
    balance_by,
    decontaminate_against,
    dedup_near_duplicates,
    dedup_records,
    drop_near,
    record_text,
)


def _sft(text: str, *, tag: str = "") -> TrainingRecord:
    return TrainingRecord(
        kind=RecordKind.SFT,
        messages=(ChatTurn(role="user", content=text), ChatTurn(role="assistant", content="ok")),
        coverage_tag=tag,
    )


def _content(records) -> list[str]:
    return [r.to_sft()["messages"][0]["content"] for r in records]


def test_dedup_removes_exact_duplicates_keeping_first() -> None:
    out = dedup_records([_sft("a"), _sft("b"), _sft("a"), _sft("a")])
    assert _content(out) == ["a", "b"]


def test_dedup_ignores_record_id_and_provenance() -> None:
    a = _sft("x")
    b = TrainingRecord(kind=RecordKind.SFT, messages=a.messages, provenance={"run_id": "z"})
    assert len(dedup_records([a, b])) == 1


def test_decontaminate_drops_exact_held_out() -> None:
    out = decontaminate_against([_sft("a"), _sft("b"), _sft("c")], [_sft("b")])
    assert _content(out) == ["a", "c"]


def test_balance_by_to_smallest_group() -> None:
    recs = [_sft(f"x{i}", tag="big") for i in range(5)] + [_sft("y", tag="small")]
    counts: dict[object, int] = {}
    for r in balance_by(recs, key=lambda r: r.coverage_tag):
        counts[r.coverage_tag] = counts.get(r.coverage_tag, 0) + 1
    assert counts == {"big": 1, "small": 1}


def test_balance_by_with_explicit_cap() -> None:
    recs = [_sft(f"x{i}", tag="big") for i in range(5)] + [_sft(f"y{i}", tag="small") for i in range(3)]
    counts: dict[object, int] = {}
    for r in balance_by(recs, key=lambda r: r.coverage_tag, max_per_group=2):
        counts[r.coverage_tag] = counts.get(r.coverage_tag, 0) + 1
    assert counts == {"big": 2, "small": 2}


def test_record_text_joins_turns() -> None:
    assert record_text(_sft("hello")) == "hello\nok"


def test_dedup_near_duplicates_drops_similar_vectors() -> None:
    recs = [_sft("a"), _sft("b"), _sft("c")]
    vectors = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]  # a≈b, c distinct
    out = dedup_near_duplicates(recs, vectors, threshold=0.9)
    assert _content(out) == ["a", "c"]


def test_dedup_near_keeps_all_when_below_threshold() -> None:
    recs = [_sft("a"), _sft("b")]
    out = dedup_near_duplicates(recs, [[1.0, 0.0], [0.0, 1.0]], threshold=0.9)
    assert _content(out) == ["a", "b"]


def test_drop_near_removes_records_close_to_references() -> None:
    recs = [_sft("a"), _sft("b")]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    references = [[0.98, 0.02]]  # close to "a"
    out = drop_near(recs, vectors, references, threshold=0.9)
    assert _content(out) == ["b"]


def test_vector_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        dedup_near_duplicates([_sft("a")], [[1.0], [2.0]])
    with pytest.raises(ValueError, match="same length"):
        drop_near([_sft("a")], [], [[1.0]])


def test_empty_inputs() -> None:
    assert dedup_records([]) == []
    assert decontaminate_against([], [_sft("a")]) == []
    assert balance_by([], key=lambda r: r.coverage_tag) == []
    assert dedup_near_duplicates([], []) == []
