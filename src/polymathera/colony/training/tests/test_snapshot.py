"""Tests for :func:`assemble_snapshot`."""

from __future__ import annotations

import pytest

from polymathera.colony.training import (
    ChatTurn,
    RecordKind,
    SnapshotError,
    TrainingRecord,
    assemble_snapshot,
)


def _sft(text: str, *, tag: str = "") -> TrainingRecord:
    return TrainingRecord(
        kind=RecordKind.SFT,
        messages=(ChatTurn(role="user", content=text), ChatTurn(role="assistant", content="ok")),
        coverage_tag=tag,
    )


def test_concatenates_all_records_without_ratios() -> None:
    snap = assemble_snapshot({"v1": [_sft("a"), _sft("b")], "v2": [_sft("c")]})
    assert snap.record_count == 3
    assert snap.sources == ("v1", "v2")
    assert snap.coverage_manifest["sources"] == {"v1": 2, "v2": 1}
    assert snap.coverage_manifest["kinds"] == {"sft": 3}


def test_coverage_tags_counted() -> None:
    snap = assemble_snapshot(
        {"v1": [_sft("a", tag="cell:1"), _sft("b", tag="cell:1"), _sft("c", tag="cell:2")]},
    )
    assert snap.coverage_manifest["coverage_tags"] == {"cell:1": 2, "cell:2": 1}


def test_content_hash_is_stable_and_ignores_record_id() -> None:
    # Same training content, independently constructed (different record_ids).
    a = assemble_snapshot({"v1": [_sft("x"), _sft("y")]})
    b = assemble_snapshot({"v1": [_sft("x"), _sft("y")]})
    assert a.content_hash == b.content_hash
    # Different content → different hash.
    c = assemble_snapshot({"v1": [_sft("x"), _sft("z")]})
    assert c.content_hash != a.content_hash


def test_mix_ratios_downsample_to_proportions_without_upsampling() -> None:
    sources = {
        "big": [_sft(f"b{i}") for i in range(100)],
        "small": [_sft(f"s{i}") for i in range(10)],
    }
    # 1:1 mix is limited by the small source (10), so 10 from each.
    snap = assemble_snapshot(sources, mix_ratios={"big": 1.0, "small": 1.0})
    assert snap.coverage_manifest["sources"] == {"big": 10, "small": 10}
    assert snap.record_count == 20
    assert snap.mix_ratios == {"big": 1.0, "small": 1.0}


def test_mix_ratios_skewed() -> None:
    sources = {
        "big": [_sft(f"b{i}") for i in range(100)],
        "small": [_sft(f"s{i}") for i in range(10)],
    }
    # 9:1 → small (10) supports total 100: 90 big + 10 small.
    snap = assemble_snapshot(sources, mix_ratios={"big": 9.0, "small": 1.0})
    assert snap.coverage_manifest["sources"] == {"big": 90, "small": 10}


def test_unknown_mix_ratio_source_rejected() -> None:
    with pytest.raises(SnapshotError, match="unknown sources"):
        assemble_snapshot({"v1": [_sft("a")]}, mix_ratios={"v2": 1.0})


def test_empty_sources_rejected() -> None:
    with pytest.raises(SnapshotError, match="at least one source"):
        assemble_snapshot({})


def test_all_zero_ratios_rejected() -> None:
    with pytest.raises(SnapshotError, match="positive weight"):
        assemble_snapshot({"v1": [_sft("a")]}, mix_ratios={"v1": 0.0})
