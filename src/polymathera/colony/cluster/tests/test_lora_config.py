"""Tests for ``vllm_max_lora_rank`` — rounding adapter ranks up to a
vLLM-supported ``max_lora_rank``."""

from __future__ import annotations

import pytest

from polymathera.colony.cluster.config import (
    LLMDeploymentConfig,
    LoRAAdapterConfig,
    round_up_lora_rank,
    vllm_max_lora_rank,
)


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


@pytest.mark.parametrize("needed,expected", [(1, 8), (16, 16), (17, 32), (300, 256)])
def test_round_up_lora_rank(needed: int, expected: int) -> None:
    assert round_up_lora_rank(needed) == expected


def test_hot_add_capacity_config_defaults() -> None:
    # Hot-add is off by default (backward compatible with static-only).
    default = LLMDeploymentConfig(model_name="org/base")
    assert default.max_lora_slots == 0 and default.max_lora_rank == 16
    # A serving fleet reserves slots + a rank ceiling for runtime adds.
    fleet = LLMDeploymentConfig(model_name="org/base", max_lora_slots=8, max_lora_rank=64)
    assert fleet.max_lora_slots == 8 and fleet.max_lora_rank == 64


def test_hot_add_capacity_reachable_from_yaml_config() -> None:
    # Activation path: cluster YAML → from_model_registry → LLMDeploymentConfig.
    # Without plumbing here, max_lora_slots stayed 0 and hot-add was disabled.
    cfg = LLMDeploymentConfig.from_model_registry(
        model_name="org/base", max_lora_slots=8, max_lora_rank=64,
    )
    assert cfg.max_lora_slots == 8 and cfg.max_lora_rank == 64


def test_vllm_provider_reachable_from_yaml_config() -> None:
    # The YAML remote-deployment dataclass carries provider "vllm" + base_url,
    # so VllmRemoteDeployment can actually be configured.
    from polymathera.colony.cli.polymath import RemoteDeploymentYAMLConfig

    rd = RemoteDeploymentYAMLConfig(provider="vllm", base_url="http://fleet:8000/v1")
    assert rd.provider == "vllm" and rd.base_url == "http://fleet:8000/v1"
