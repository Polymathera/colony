# Colony operator configuration

This directory ships the operator-facing configuration surface for Colony.

- **[`example.yaml`](example.yaml)** — a ready-to-customize template. Copy it,
  edit it, and pass to the CLI:

  ```bash
  colony-env up --workers 3 --config /path/to/your/colony.yaml
  ```

  The example provides a working baseline and adds commented blocks for every typed-component   override slot.

- This README — the override chain, registered components, and how to extend
  the config surface from an external package.

---

## Override chain

For every field of every registered `ConfigComponent`, the resolution order is
(lowest → highest precedence):

1. **Pydantic field default** declared on the component class. Source of truth
   for built-in defaults; no separate defaults file.
2. **Operator YAML** loaded by `ConfigurationManager` from `--config`.
3. **Environment variable** declared by the field's `json_schema_extra["env"]`
   binding. (Field-level — see each component class's source for which env
   vars it binds.) The catch-all
   `POLYMATHERA_<dotted_path_with_underscores>_<field>` always works for any
   registered field.

**Tier overlays** (L2 tenant / L3 session / L4 runtime) layer on top of L1
(defaults + YAML + env) at read time via `cm.get_component_for(path,
tenant_id=..., session_id=...)`. Writes go through `cm.update_overlay(path,
updates, scope=OverlayScope.tenant(tid))` and are tier-checked against each
field's declared `Tier` (see `distributed/config/tiers.py`).

There is no fallback location search — `--config` is the single explicit file
path. Everything goes through this YAML.

---

## Registered components

| YAML path | Component | Source |
|---|---|---|
| `agent_system` | `AgentSystemConfig` | `agents/config.py` |
| `analysis_registry` | `AnalysisRegistryConfig` | `agents/configs.py` |
| `capabilities.github` | `GitHubAuthConfig` | `agents/configs.py` |
| `capabilities.web_search` | `WebSearchConfig` | `agents/configs.py` |
| `custom_deployments` | `CustomDeploymentsConfig` | `deployments/configs.py` |
| `deployment_names` | `DeploymentNames` | `deployment_names.py` |
| `distributed.observability` | `ObservabilityConfig` | `distributed/configs.py` |
| `llms.inference.cluster.embedding` | `STEmbeddingDeploymentConfig` | `cluster/embedding/st_embedding.py` |
| `memory.chroma` | `ChromaConfig` | `agents/configs.py` |
| `plugins` | `PluginsConfig` | `agents/configs.py` |
| `polymathera_cluster` | `PolymatheraClusterConfig` | `system.py` |
| `sandbox_images` | `SandboxImagesConfig` | `agents/configs.py` |
| `task_images` | `TaskImagesConfig` | `agents/configs.py` |
| `web_ui` | `WebUIConfig` | `web_ui/backend/configs.py` |

Plus infrastructure-level components (`redis`, `kafka`, `aws`, `security`,
`monitoring`, storage backends, etc.) — see `distributed/configs.py` and
`distributed/config/configs.py`. Operators rarely override these from YAML;
the deployment image's `.env` populates them via env-var bindings.

The legacy top-level keys consumed by `cli/polymath.py:TestConfig` (`cluster`,
`vcm`, `paging`, `analyses`, `hierarchy`, `output_dir`, `timeout_seconds`,
`budget_usd`, `working_dir`, `origin_url`, `branch`, `commit`) coexist in the
same YAML; the dataclass parser and the typed manager both read it.

---

## Extending the config surface

External packages (e.g. `polymathera-cps`) add new `ConfigComponent`s and
custom-deployment handlers without touching public colony code. Two
mechanisms:

### Register components via entry-point group

Declare in your package's `pyproject.toml`:

```toml
[tool.poetry.plugins."polymathera.config_components"]
my_extension = "polymathera.cps.config:register_components"
```

`polymathera.cps.config:register_components` is a callable that, when
invoked, imports the modules whose `@register_polymathera_config()`-decorated
classes you want registered. `ConfigurationManager.initialize()` walks the
group at startup; failures are isolated and logged.

Inside your module:

```python
from polymathera.colony.distributed.config import (
    ConfigComponent, register_polymathera_config, Tier, tier_metadata,
)

@register_polymathera_config(path="my_ext.thing")
class MyThingConfig(ConfigComponent):
    knob: int = Field(
        default=42,
        json_schema_extra={
            "env": "MY_EXT_THING_KNOB", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )
```

Operators set it via:

```yaml
my_ext:
  thing:
    knob: 99
```

### Register a custom deployment handler

For long-running externally-managed resources (HPC stacks, AWS-CDK,
Slurm clusters):

```python
from polymathera.colony.deployments import (
    DeploymentContext, register_custom_deployment,
)

@register_custom_deployment("aws_cdk_hpc")
class AwsCdkHpc(ConfigComponent):
    name = "aws_cdk_hpc"

    async def provision(self, ctx: DeploymentContext) -> None:
        # ... bring up resources ...
        await ctx.write_runtime_overlay(
            "my_ext.hpc_endpoints",
            {"scheduler_url": "https://...", "auth_token": "..."},
        )

    async def query_state(self, ctx): ...
    async def tear_down(self, ctx): ...
```

Operator YAML wires it:

```yaml
custom_deployments:
  deployments:
    cps_hpc_aero:
      handler: aws_cdk_hpc
      auto_provision: true
      params: { stack_name: my-stack, region: us-west-2 }
```

After `provision()`, any consumer reading `my_ext.hpc_endpoints` via
`cm.get_component_for(...)` observes the runtime values — no restart needed.

### Register a new LLM deployment provider

Same pattern; the registry lives at `cluster/remote_registry.py`:

```python
from polymathera.colony.cluster.remote_registry import register_remote_llm_provider
from polymathera.colony.cluster.remote_deployment import RemoteLLMDeployment

@register_remote_llm_provider("my_provider")
class MyProviderDeployment(RemoteLLMDeployment):
    ...
```

Operator YAML uses the new `provider` value:

```yaml
cluster:
  remote_deployments:
    - model_name: "my-model"
      provider: "my_provider"
      api_key_env_var: "MY_API_KEY"
```
