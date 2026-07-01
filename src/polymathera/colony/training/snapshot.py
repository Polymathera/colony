"""Assemble training records from several sources into one snapshot.

A snapshot is an immutable, content-addressed bundle of training
records drawn from one or more named sources (e.g. dataset versions),
optionally mixed to target proportions. The content hash identifies the
training *content* — it excludes per-record ids and provenance — so the
same data assembled twice yields the same hash, which is what makes a
snapshot a stable, deduplicable dataset version.

This is a pure transformation. Loading records from storage and
persisting the snapshot belong to the caller.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from .records import TrainingRecord
from .sampling import evenly_spaced


class SnapshotError(ValueError):
    """Raised on an ill-formed assembly request."""


class Snapshot(BaseModel):
    """An assembled, content-addressed set of training records."""

    model_config = ConfigDict(frozen=True)

    content_hash: str
    record_count: int
    records: tuple[TrainingRecord, ...]
    sources: tuple[str, ...]
    mix_ratios: dict[str, float] | None = None
    coverage_manifest: dict[str, Any]


def _target_counts(
    sources: Mapping[str, Sequence[TrainingRecord]],
    mix_ratios: Mapping[str, float],
) -> dict[str, int]:
    """Per-source counts honoring ``mix_ratios`` without up-sampling.

    The total is the largest size at which every source can supply its
    share, so the result respects the requested proportions exactly
    (subject to integer rounding) and never repeats a record.
    """
    total = sum(mix_ratios.values())
    caps = [
        len(sources[vid]) * total / ratio
        for vid, ratio in mix_ratios.items()
        if ratio > 0
    ]
    budget = min(caps) if caps else 0.0
    return {
        vid: int((ratio / total) * budget) if ratio > 0 else 0
        for vid, ratio in mix_ratios.items()
    }


def _content_hash(records: Sequence[TrainingRecord]) -> str:
    payload = json.dumps(
        [r.content_payload() for r in records],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assemble_snapshot(
    sources: Mapping[str, Sequence[TrainingRecord]],
    *,
    mix_ratios: Mapping[str, float] | None = None,
) -> Snapshot:
    """Combine ``sources`` (``name → records``) into one snapshot.

    With ``mix_ratios`` (``name → weight``), each source is down-sampled
    to its share; without it, every record is kept. Sources are
    concatenated in iteration order.
    """

    if not sources:
        raise SnapshotError("at least one source is required")

    if mix_ratios is not None:
        unknown = set(mix_ratios) - set(sources)
        if unknown:
            raise SnapshotError(f"mix_ratios names unknown sources: {sorted(unknown)}")
        if all(r <= 0 for r in mix_ratios.values()):
            raise SnapshotError("mix_ratios must have at least one positive weight")
        counts = _target_counts(sources, mix_ratios)
        selected: list[TrainingRecord] = []
        for name in sources:
            if name in counts:
                selected.extend(evenly_spaced(sources[name], counts[name]))
        per_source = {name: counts.get(name, 0) for name in sources}
    else:
        selected = [rec for name in sources for rec in sources[name]]
        per_source = {name: len(sources[name]) for name in sources}

    coverage_manifest = {
        "sources": per_source,
        "kinds": dict(Counter(rec.kind.value for rec in selected)),
        "coverage_tags": dict(
            Counter(rec.coverage_tag for rec in selected if rec.coverage_tag)
        ),
    }
    return Snapshot(
        content_hash=_content_hash(selected),
        record_count=len(selected),
        records=tuple(selected),
        sources=tuple(sources),
        mix_ratios=dict(mix_ratios) if mix_ratios is not None else None,
        coverage_manifest=coverage_manifest,
    )


__all__ = ("Snapshot", "SnapshotError", "assemble_snapshot")
