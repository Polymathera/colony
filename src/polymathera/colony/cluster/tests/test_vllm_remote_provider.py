"""The self-hosted vLLM remote provider: registry resolution + config."""

from __future__ import annotations

import pytest

from polymathera.colony.cluster.remote_config import RemoteLLMDeploymentConfig
from polymathera.colony.cluster.remote_registry import get_remote_llm_deployment_class


def test_vllm_provider_resolves_to_remote_deployment_class() -> None:
    cls = get_remote_llm_deployment_class("vllm")
    from polymathera.colony.cluster.remote_deployment import RemoteLLMDeployment
    from polymathera.colony.cluster.vllm_remote_deployment import VllmRemoteDeployment

    assert cls is VllmRemoteDeployment
    assert issubclass(cls, RemoteLLMDeployment)


def test_vllm_config_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        RemoteLLMDeploymentConfig(model_name="adapter-x", provider="vllm")


def test_vllm_config_with_base_url_ok() -> None:
    config = RemoteLLMDeploymentConfig(
        model_name="adapter-x", provider="vllm",
        base_url="http://vllm.internal:8000/v1",
    )
    assert config.base_url.endswith("/v1")
    assert config.get_deployment_name() == "remote-adapter-x"
