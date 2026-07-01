"""Tests for ``vllm_max_lora_rank`` — rounding adapter ranks up to a
vLLM-supported ``max_lora_rank``."""

from __future__ import annotations

import pytest

from polymathera.colony.cluster.config import LoRAAdapterConfig, vllm_max_lora_rank


def _adapter(rank: int) -> LoRAAdapterConfig:
    return LoRAAdapterConfig(
        adapter_id=f"a{rank}", adapter_name="org/adapter",
        base_model_name="org/base", rank=rank,
    )


@pytest.mark.parametrize(
    "ranks,expected",
    [
        ([8], 8),       # exact supported value
        ([16], 16),
        ([24], 32),     # rounds up to the next supported value
        ([1], 8),       # below the smallest supported value
        ([8, 200], 256),  # max across adapters drives it
        ([256], 256),   # ceiling (field caps rank at 256)
    ],
)
def test_rounds_up_to_supported_rank(ranks: list[int], expected: int) -> None:
    assert vllm_max_lora_rank([_adapter(r) for r in ranks]) == expected
