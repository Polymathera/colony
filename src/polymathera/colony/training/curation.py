"""Curation primitives for training-record sets.

Domain-agnostic, pure transformations used to clean and shape a dataset
before a snapshot is assembled. Compose them as needed (they are
primitives, not a fixed pipeline):

- :func:`dedup_records` / :func:`decontaminate_against` — exact, by
  record content.
- :func:`balance_by` — cap each group of a chosen dimension.
- :func:`dedup_near_duplicates` / :func:`drop_near` — near-match, over
  vectors the caller supplies.

Near-match needs vectors. Embedding is the caller's job — production
embedders are async and back onto an embedding service / Qdrant — so
these functions take pre-computed vectors and stay pure and sync. Use
:func:`record_text` to get the text to embed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Sequence

from .records import TrainingRecord
from .sampling import evenly_spaced

Vector = Sequence[float]


def record_content_hash(record: TrainingRecord) -> str:
    """Stable hash of a record's training content (ignores id/provenance)."""
    payload = json.dumps(record.content_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_text(record: TrainingRecord) -> str:
    """The record's text, for embedding — every turn's content joined."""
    parts = [
        turn.content
        for turns in (record.messages, record.prompt, record.chosen or (), record.rejected or ())
        for turn in turns
    ]
    return "\n".join(parts)


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def dedup_records(records: Iterable[TrainingRecord]) -> list[TrainingRecord]:
    """Records with exact duplicates removed, keeping the first."""
    seen: set[str] = set()
    out: list[TrainingRecord] = []
    for record in records:
        digest = record_content_hash(record)
        if digest in seen:
            continue
        seen.add(digest)
        out.append(record)
    return out


def decontaminate_against(
    records: Iterable[TrainingRecord], held_out: Iterable[TrainingRecord],
) -> list[TrainingRecord]:
    """Records whose content is not present in ``held_out`` — so an
    eval/gold set never leaks into training."""
    held_hashes = {record_content_hash(r) for r in held_out}
    return [r for r in records if record_content_hash(r) not in held_hashes]


def balance_by(
    records: Iterable[TrainingRecord],
    key: Callable[[TrainingRecord], object],
    *,
    max_per_group: int | None = None,
) -> list[TrainingRecord]:
    """Cap each ``key`` group. ``max_per_group=None`` balances every group
    down to the smallest group's size; otherwise each is capped at
    ``max_per_group``. Over-full groups are evenly down-sampled."""
    groups: dict[object, list[TrainingRecord]] = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    if not groups:
        return []
    cap = max_per_group if max_per_group is not None else min(len(g) for g in groups.values())
    out: list[TrainingRecord] = []
    for group in groups.values():
        out.extend(evenly_spaced(group, cap))
    return out


def dedup_near_duplicates(
    records: Sequence[TrainingRecord],
    vectors: Sequence[Vector],
    *,
    threshold: float = 0.97,
) -> list[TrainingRecord]:
    """Greedily drop records whose vector is within ``threshold`` cosine
    of an already-kept record. ``vectors[i]`` is the vector for
    ``records[i]``."""
    if len(records) != len(vectors):
        raise ValueError("records and vectors must be the same length")
    kept: list[TrainingRecord] = []
    kept_vectors: list[Vector] = []
    for record, vector in zip(records, vectors):
        if any(_cosine(vector, kv) >= threshold for kv in kept_vectors):
            continue
        kept.append(record)
        kept_vectors.append(vector)
    return kept


def drop_near(
    records: Sequence[TrainingRecord],
    vectors: Sequence[Vector],
    reference_vectors: Sequence[Vector],
    *,
    threshold: float = 0.97,
) -> list[TrainingRecord]:
    """Drop records whose vector is within ``threshold`` cosine of any
    ``reference_vectors`` entry — near-match decontamination against an
    embedded held-out set. ``vectors[i]`` is the vector for
    ``records[i]``."""
    if len(records) != len(vectors):
        raise ValueError("records and vectors must be the same length")
    out: list[TrainingRecord] = []
    for record, vector in zip(records, vectors):
        if any(_cosine(vector, rv) >= threshold for rv in reference_vectors):
            continue
        out.append(record)
    return out


__all__ = (
    "balance_by",
    "decontaminate_against",
    "dedup_near_duplicates",
    "dedup_records",
    "drop_near",
    "record_content_hash",
    "record_text",
)
