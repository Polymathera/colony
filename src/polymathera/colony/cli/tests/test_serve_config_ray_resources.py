"""Regression: a vLLM deployment's ``ray_resources`` (its GPU-type pin) must
survive the YAML → ``LLMDeploymentConfig`` round-trip.

``load_config_from_yaml`` filters each YAML dict to
``VLLMDeploymentYAMLConfig.__dataclass_fields__`` — any key not declared on that
dataclass is silently dropped. The CPS multi-GPU-type serving fleet relies on
``ray_resources`` reaching the deployment's ``ray_actor_options["resources"]`` so
each model's actor is placed on the worker group advertising the matching custom
resource. Lock both links so a future refactor can't silently drop the field
(which would unpin every model and scatter actors across GPU types).
"""

from __future__ import annotations

from polymathera.colony.cli.polymath import load_config_from_yaml
from polymathera.colony.cluster.config import LLMDeploymentConfig


def test_ray_resources_survives_yaml_load(tmp_path) -> None:
    # Link 1: the __dataclass_fields__ filter must keep ray_resources.
    cfg = tmp_path / "fleet.yaml"
    cfg.write_text(
        "cluster:\n"
        "  app_name: polymathera\n"
        "  vllm_deployments:\n"
        "    - model_name: meta-llama/Llama-3.1-8B\n"
        "      tensor_parallel_size: 1\n"
        "      num_replicas: 1\n"
        "      ray_resources:\n"
        "        accelerator_a10g: 1.0\n",
        encoding="utf-8",
    )
    loaded = load_config_from_yaml(str(cfg))
    vd = loaded.cluster.vllm_deployments[0]
    assert vd.ray_resources == {"accelerator_a10g": 1.0}


def test_ray_resources_threads_into_deployment_config() -> None:
    # Link 2: from_model_registry (**overrides) must carry it onto the config,
    # from where add_deployments_to_app merges it into ray_actor_options.
    dep = LLMDeploymentConfig.from_model_registry(
        model_name="meta-llama/Llama-3.1-8B",
        tensor_parallel_size=1,
        ray_resources={"accelerator_a10g": 1.0},
    )
    assert dep.ray_resources == {"accelerator_a10g": 1.0}


def test_ray_resources_defaults_to_none() -> None:
    # Absent in YAML → None → no "resources" added to ray_actor_options (the
    # deployment places on any GPU node, preserving today's single-type behavior).
    dep = LLMDeploymentConfig.from_model_registry(model_name="meta-llama/Llama-3.1-8B")
    assert dep.ray_resources is None


def test_gateway_targets_one_port_per_base_model() -> None:
    # polymath serve binds one gateway per base model at base_port + index — the
    # port assignment the CPS k8s manifests (serving_config.gateway_endpoints)
    # must match, so a client reaches each base model at its own base_url.
    from polymathera.colony.cli.polymath import _resolve_gateway_targets

    configs = [
        LLMDeploymentConfig(model_name="meta-llama/Llama-3.1-8B"),
        LLMDeploymentConfig(model_name="meta-llama/Llama-3.1-70B"),
    ]
    targets = _resolve_gateway_targets(configs, None, 8000)
    assert [p for _, p in targets] == [8000, 8001]
    assert [name for name, _ in targets] == [c.get_deployment_name() for c in configs]
    # --deployment pins a single model to the base port (back-compat).
    assert _resolve_gateway_targets(configs, "vllm-x", 8000) == [("vllm-x", 8000)]
