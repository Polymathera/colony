"""JSONL (de)serialization for :class:`TrainingRecord`.

One record per line, with sorted keys so the encoding is stable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from .records import TrainingRecord


def records_to_jsonl(records: Iterable[TrainingRecord]) -> str:
    return "".join(
        json.dumps(r.model_dump(mode="json"), sort_keys=True) + "\n" for r in records
    )


def records_from_jsonl(text: str) -> list[TrainingRecord]:
    return [
        TrainingRecord.model_validate_json(line)
        for line in text.splitlines()
        if line.strip()
    ]


__all__ = ("records_from_jsonl", "records_to_jsonl")
