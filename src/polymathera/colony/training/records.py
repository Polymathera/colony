"""A common record for fine-tuning chat LLMs.

One :class:`TrainingRecord` holds a single interaction in exactly one
training shape and projects to the corresponding trainer-native view:

- ``sft`` — a full chat trajectory → ``{"messages": [...]}`` for
  supervised fine-tuning.
- ``preference`` — a prompt plus a chosen/rejected completion pair →
  ``{"prompt": ..., "chosen": ..., "rejected": ...}`` for preference
  optimization (e.g. DPO).
- ``rl_prompt`` — a prompt only → ``{"prompt": ...}`` for prompt-only
  reinforcement learning (e.g. GRPO), where the reward comes from a
  reward function rather than the dataset.

Each record is constrained to be well-formed for its ``kind`` at
construction, so a record never carries half-populated, ambiguous
views. The three projection methods emit plain dicts in the
``role``/``content`` chat format trainers consume directly.

The schema is domain-agnostic. Producer-specific disposition —
``source`` (where the record came from), ``reward_source`` (what
produced ``verifiable_reward``), and ``coverage_tag`` (an opaque
position in the producer's input space) — are open strings, so
callers use their own vocabularies without this module enumerating
them.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1"


class TrainingRecordError(ValueError):
    """Raised when a record is projected to a view it does not carry."""


class RecordKind(str, Enum):
    """The training shape a record carries."""

    SFT = "sft"
    PREFERENCE = "preference"
    RL_PROMPT = "rl_prompt"


class ChatTurn(BaseModel):
    """One turn in a chat trajectory."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str
    tool_calls: tuple[dict[str, Any], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain ``role``/``content`` dict; ``tool_calls`` only when set."""

        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            out["tool_calls"] = [dict(tc) for tc in self.tool_calls]
        return out


def _turns(turns: tuple[ChatTurn, ...]) -> list[dict[str, Any]]:
    return [t.to_dict() for t in turns]


class TrainingRecord(BaseModel):
    """A single interaction recorded for fine-tuning, in one shape."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(
        default_factory=lambda: f"tr_{uuid.uuid4().hex[:16]}",
    )
    kind: RecordKind

    #: SFT view — the full trajectory. Set iff ``kind == SFT``.
    messages: tuple[ChatTurn, ...] = ()

    #: Preference / RL view — the prompt turns. Set iff ``kind`` is
    #: ``PREFERENCE`` or ``RL_PROMPT``.
    prompt: tuple[ChatTurn, ...] = ()

    #: Preference view — the completion pair. Set iff ``kind ==
    #: PREFERENCE``.
    chosen: tuple[ChatTurn, ...] | None = None
    rejected: tuple[ChatTurn, ...] | None = None

    #: Optional verifiable signal carried alongside the record (e.g.
    #: for filtering, weighting, or reward-model targets). Not part of
    #: any projection — RL reward comes from a reward function.
    verifiable_reward: float | None = None

    #: Open-set tags interpreted by the producer, never by this module.
    source: str = ""
    reward_source: str = ""
    coverage_tag: str = ""

    provenance: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="after")
    def _well_formed_for_kind(self) -> "TrainingRecord":
        if self.kind is RecordKind.SFT:
            if not self.messages:
                raise TrainingRecordError("sft record requires non-empty messages")
            if self.prompt or self.chosen is not None or self.rejected is not None:
                raise TrainingRecordError(
                    "sft record must not carry prompt/chosen/rejected",
                )
        elif self.kind is RecordKind.PREFERENCE:
            if not self.prompt:
                raise TrainingRecordError("preference record requires a prompt")
            if not self.chosen or not self.rejected:
                raise TrainingRecordError(
                    "preference record requires both chosen and rejected",
                )
            if self.messages:
                raise TrainingRecordError(
                    "preference record must not carry messages",
                )
        else:  # RL_PROMPT
            if not self.prompt:
                raise TrainingRecordError("rl_prompt record requires a prompt")
            if (
                self.messages
                or self.chosen is not None
                or self.rejected is not None
            ):
                raise TrainingRecordError(
                    "rl_prompt record must not carry messages/chosen/rejected",
                )
        return self

    def to_sft(self) -> dict[str, list[dict[str, Any]]]:
        """``{"messages": [...]}``. Raises if this is not an SFT record."""

        if self.kind is not RecordKind.SFT:
            raise TrainingRecordError(
                f"to_sft on a {self.kind.value!r} record (no messages)",
            )
        return {"messages": _turns(self.messages)}

    def to_dpo(self) -> dict[str, list[dict[str, Any]]]:
        """``{"prompt", "chosen", "rejected"}``. Raises if not a
        preference record."""

        if self.kind is not RecordKind.PREFERENCE:
            raise TrainingRecordError(
                f"to_dpo on a {self.kind.value!r} record (no preference pair)",
            )
        assert self.chosen is not None and self.rejected is not None  # validator
        return {
            "prompt": _turns(self.prompt),
            "chosen": _turns(self.chosen),
            "rejected": _turns(self.rejected),
        }

    def to_grpo(self) -> dict[str, list[dict[str, Any]]]:
        """``{"prompt": [...]}``. Raises if the record has no prompt
        (i.e. it is an SFT record)."""

        if self.kind is RecordKind.SFT:
            raise TrainingRecordError("to_grpo on an 'sft' record (no prompt)")
        return {"prompt": _turns(self.prompt)}

    def content_payload(self) -> dict[str, Any]:
        """The record's training content, excluding the per-record id and
        provenance. Two records with the same payload are the same data —
        the basis for content hashing and de-duplication."""

        return self.model_dump(mode="json", exclude={"record_id", "provenance"})


__all__ = (
    "SCHEMA_VERSION",
    "ChatTurn",
    "RecordKind",
    "TrainingRecord",
    "TrainingRecordError",
)
