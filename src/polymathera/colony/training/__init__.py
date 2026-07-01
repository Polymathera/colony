"""Data models for fine-tuning chat LLMs from recorded interactions.

Exposes :class:`TrainingRecord` — a single conversational record that
projects to the three trainer-native dataset views (supervised
fine-tuning, preference optimization, and prompt-only reinforcement
learning) without reprocessing.
"""

from .records import (
    ChatTurn,
    RecordKind,
    TrainingRecord,
    TrainingRecordError,
)
from .curation import (
    balance_by,
    decontaminate_against,
    dedup_near_duplicates,
    dedup_records,
    drop_near,
    record_content_hash,
    record_text,
)
from .recorder import records_from_spans, trajectory_records_from_spans
from .serialization import records_from_jsonl, records_to_jsonl
from .snapshot import Snapshot, SnapshotError, assemble_snapshot


__all__ = (
    "ChatTurn",
    "RecordKind",
    "Snapshot",
    "SnapshotError",
    "TrainingRecord",
    "TrainingRecordError",
    "assemble_snapshot",
    "balance_by",
    "decontaminate_against",
    "dedup_near_duplicates",
    "dedup_records",
    "drop_near",
    "record_content_hash",
    "record_text",
    "records_from_jsonl",
    "records_from_spans",
    "records_to_jsonl",
    "trajectory_records_from_spans",
)
