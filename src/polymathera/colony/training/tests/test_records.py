"""Tests for the :class:`TrainingRecord` schema + projections."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from polymathera.colony.training.records import (
    ChatTurn,
    RecordKind,
    TrainingRecord,
    TrainingRecordError,
)

# Construction-time well-formedness is enforced by a pydantic model
# validator, so an ill-formed record raises ``ValidationError`` (a
# ``ValueError``) wrapping the ``TrainingRecordError`` message; the
# projection methods raise ``TrainingRecordError`` directly.


def _u(text: str) -> ChatTurn:
    return ChatTurn(role="user", content=text)


def _a(text: str) -> ChatTurn:
    return ChatTurn(role="assistant", content=text)


# ---- ChatTurn -------------------------------------------------------------


def test_chat_turn_to_dict_omits_tool_calls_when_absent() -> None:
    assert _u("hi").to_dict() == {"role": "user", "content": "hi"}


def test_chat_turn_to_dict_includes_tool_calls_when_present() -> None:
    turn = ChatTurn(
        role="assistant", content="", tool_calls=({"name": "f", "args": {}},),
    )
    assert turn.to_dict() == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": "f", "args": {}}],
    }


# ---- SFT ------------------------------------------------------------------


def test_sft_projects_to_messages() -> None:
    rec = TrainingRecord(
        kind=RecordKind.SFT, messages=(_u("q"), _a("r")), source="agent_trajectory",
    )
    assert rec.to_sft() == {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "r"},
        ],
    }


def test_sft_record_id_defaults_and_is_prefixed() -> None:
    rec = TrainingRecord(kind=RecordKind.SFT, messages=(_u("q"), _a("r")))
    assert rec.record_id.startswith("tr_")


def test_sft_other_projections_raise() -> None:
    rec = TrainingRecord(kind=RecordKind.SFT, messages=(_u("q"), _a("r")))
    with pytest.raises(TrainingRecordError):
        rec.to_dpo()
    with pytest.raises(TrainingRecordError):
        rec.to_grpo()


def test_sft_without_messages_rejected() -> None:
    with pytest.raises(ValidationError):
        TrainingRecord(kind=RecordKind.SFT)


def test_sft_with_prompt_rejected() -> None:
    with pytest.raises(ValidationError):
        TrainingRecord(
            kind=RecordKind.SFT, messages=(_a("r"),), prompt=(_u("q"),),
        )


# ---- Preference -----------------------------------------------------------


def test_preference_projects_to_dpo() -> None:
    rec = TrainingRecord(
        kind=RecordKind.PREFERENCE,
        prompt=(_u("q"),),
        chosen=(_a("good"),),
        rejected=(_a("bad"),),
        source="span_feedback",
    )
    assert rec.to_dpo() == {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "good"}],
        "rejected": [{"role": "assistant", "content": "bad"}],
    }


def test_preference_projects_to_grpo_prompt_only() -> None:
    rec = TrainingRecord(
        kind=RecordKind.PREFERENCE,
        prompt=(_u("q"),),
        chosen=(_a("good"),),
        rejected=(_a("bad"),),
    )
    assert rec.to_grpo() == {"prompt": [{"role": "user", "content": "q"}]}


def test_preference_to_sft_raises() -> None:
    rec = TrainingRecord(
        kind=RecordKind.PREFERENCE,
        prompt=(_u("q"),),
        chosen=(_a("good"),),
        rejected=(_a("bad"),),
    )
    with pytest.raises(TrainingRecordError):
        rec.to_sft()


def test_preference_missing_rejected_rejected() -> None:
    with pytest.raises(ValidationError):
        TrainingRecord(
            kind=RecordKind.PREFERENCE, prompt=(_u("q"),), chosen=(_a("good"),),
        )


# ---- RL prompt ------------------------------------------------------------


def test_rl_prompt_projects_to_grpo() -> None:
    rec = TrainingRecord(
        kind=RecordKind.RL_PROMPT,
        prompt=(_u("solve"),),
        verifiable_reward=1.0,
        reward_source="sim",
    )
    assert rec.to_grpo() == {"prompt": [{"role": "user", "content": "solve"}]}
    assert rec.verifiable_reward == 1.0


def test_rl_prompt_other_projections_raise() -> None:
    rec = TrainingRecord(kind=RecordKind.RL_PROMPT, prompt=(_u("solve"),))
    with pytest.raises(TrainingRecordError):
        rec.to_sft()
    with pytest.raises(TrainingRecordError):
        rec.to_dpo()


def test_rl_prompt_with_messages_rejected() -> None:
    with pytest.raises(ValidationError):
        TrainingRecord(
            kind=RecordKind.RL_PROMPT, prompt=(_u("q"),), messages=(_a("r"),),
        )


# ---- Round-trip + open-set fields -----------------------------------------


def test_round_trip_preserves_record() -> None:
    rec = TrainingRecord(
        kind=RecordKind.SFT,
        messages=(_u("q"), _a("r")),
        source="custom_source",
        reward_source="physics_floor",
        coverage_tag="cell:7",
        provenance={"run_id": "run-1", "agent_type": "X"},
    )
    again = TrainingRecord.model_validate(rec.model_dump())
    assert again == rec
    # Open-set fields pass through verbatim — this module never enumerates them.
    assert again.reward_source == "physics_floor"
    assert again.coverage_tag == "cell:7"
