# Third-Party API-Based Embedding Deployments

## Context

Colony's `EmbeddingDeployment` uses vLLM with `task="embed"` — requires a GPU. For the `colony-env` local deployment (no GPUs), we need API-based embedding deployments that are **drop-in replacements**. The `LLMCluster.embed()` endpoint already delegates to `embedding_deployment_handle.embed(texts)` and doesn't care about the backing implementation.

Three providers: **OpenAI**, **Google Gemini**, **OpenRouter** (OpenAI-compatible API).

---

## Architecture

Follows the existing remote LLM deployment pattern exactly:

```
GPU path:   LLMDeploymentConfig → EmbeddingDeployment (vLLM)
API path:   RemoteEmbeddingConfig → RemoteEmbeddingDeployment subclasses
                                     ├── OpenAICompatibleEmbeddingDeployment (OpenAI + OpenRouter)
                                     └── GeminiEmbeddingDeployment (Google)
```

Both paths register as deployment name `"embedding"` — mutually exclusive. `LLMCluster.embed()` is unchanged.

OpenAI and OpenRouter share a single class (`OpenAICompatibleEmbeddingDeployment`) since OpenRouter uses the OpenAI-compatible API with a different `base_url`. No code duplication.

---

## New Files (2)

### 1. `colony/python/colony/cluster/remote_embedding_config.py`

Config class + embedding pricing table.

```python
class RemoteEmbeddingConfig(BaseModel):
    model_name: str = "text-embedding-3-small"
    provider: Literal["openai", "gemini", "openrouter"]
    api_key_env_var: str  # Default set by model_post_init based on provider
    dimensions: int | None = None  # For OpenAI text-embedding-3-* dimension reduction
    max_batch_size: int = 2048     # OpenAI: 2048, Gemini: 100
    max_concurrent_requests: int = 10
    num_replicas: int = 1
    min_replicas: int | None = None
    max_replicas: int | None = None
    target_queue_length: int = 5
```

Provider-aware defaults in `model_post_init`:
- `openai` → `api_key_env_var="OPENAI_API_KEY"`
- `gemini` → `api_key_env_var="GOOGLE_API_KEY"`, `max_batch_size=100`
- `openrouter` → `api_key_env_var="OPENROUTER_API_KEY"`

Pricing table (per 1M tokens):
- `text-embedding-3-small`: $0.02
- `text-embedding-3-large`: $0.13
- `text-embedding-ada-002`: $0.10
- Gemini models: free

### 2. `colony/python/colony/cluster/remote_embedding_deployment.py`

Base class + two provider subclasses (all in one file — embedding is simple enough that 3 files would be bloat).

**Base class** (`RemoteEmbeddingDeployment`):
- `@serving.deployment` with autoscaling
- `embed(texts: list[str]) -> list[list[float]]` — handles batching transparently
- `get_state() -> LLMClientState` — basic metrics for monitoring
- Concurrency control via `asyncio.Semaphore`
- `@inference_circuit` for fault tolerance
- Abstract: `_initialize_client()`, `_call_embed_api(texts) -> list[list[float]]`

**`OpenAICompatibleEmbeddingDeployment`** (handles both OpenAI and OpenRouter):
- Uses `openai.AsyncOpenAI` (lazy import)
- Sets `base_url=OPENROUTER_BASE_URL` when `provider == "openrouter"`
- Passes `dimensions` param if configured
- Tracks token usage from `response.usage.prompt_tokens`

**`GeminiEmbeddingDeployment`**:
- Uses `google.genai.Client` (lazy import of `google-genai` SDK)
- Async via `client.aio.models.embed_content()`
- No token usage in response (Gemini embedding is free)

---

## Files to Modify (5)

### 1. `colony/python/colony/cluster/config.py`
- Import `RemoteEmbeddingConfig`
- Add field: `remote_embedding_config: RemoteEmbeddingConfig | None = None`
- In `add_deployments_to_app()` (after line 577): dispatch by provider to select deployment class, register as name `"embedding"`, validate mutual exclusivity with `embedding_config`

### 2. `colony/python/colony/cluster/cluster.py`
- Line ~268: extend embedding handle discovery to check both config fields:
  ```python
  if self.config.embedding_config or self.config.remote_embedding_config:
  ```

### 3. `colony/python/colony/cluster/__init__.py`
- Add eager imports: `RemoteEmbeddingConfig`, `RemoteEmbeddingDeployment`, `OpenAICompatibleEmbeddingDeployment`, `GeminiEmbeddingDeployment`
- Add to `__all__`

### 4. `colony/python/colony/cli/polymath.py`
- Add `RemoteEmbeddingYAMLConfig` dataclass (after `RemoteDeploymentYAMLConfig`)
- Add `remote_embedding_config` field to `LLMClusterYAMLConfig`
- Parse in `load_config_from_yaml()` from `cluster_raw.get("remote_embedding_config")`
- Convert to `RemoteEmbeddingConfig` in `run_integration_test()`
- Pass to `ClusterConfig(remote_embedding_config=...)`
- Display line for remote embedding in deployment panel
- Update `generate_sample_config()` YAML template
- Include embedding API key env var in passthrough collection

### 5. `colony/python/colony/cli/deploy/config.py`
- Add `"OPENAI_API_KEY"` and `"GOOGLE_API_KEY"` to `api_key_env_vars` default list

Also update `docker-compose.yml` to pass through:
```yaml
- OPENAI_API_KEY=${OPENAI_API_KEY:-}
- GOOGLE_API_KEY=${GOOGLE_API_KEY:-}
```

---

## YAML Config Examples

**OpenAI** (most common):
```yaml
cluster:
  remote_embedding_config:
    model_name: "text-embedding-3-small"
    provider: "openai"
    # api_key_env_var defaults to OPENAI_API_KEY
    # dimensions: 512  # Optional: reduce for efficiency
```

**Gemini** (free tier):
```yaml
cluster:
  remote_embedding_config:
    model_name: "text-embedding-004"
    provider: "gemini"
    # api_key_env_var defaults to GOOGLE_API_KEY
    # max_batch_size defaults to 100 for gemini
```

**OpenRouter**:
```yaml
cluster:
  remote_embedding_config:
    model_name: "openai/text-embedding-3-small"
    provider: "openrouter"
    # api_key_env_var defaults to OPENROUTER_API_KEY
```

---

## Implementation Order

1. `remote_embedding_config.py` — config + pricing (no deps)
2. `remote_embedding_deployment.py` — base + providers (depends on 1)
3. `config.py` — add field + dispatch (depends on 1, 2)
4. `cluster.py` — one-line change
5. `__init__.py` — imports
6. `polymath.py` — YAML integration
7. `deploy/config.py` + `docker-compose.yml` — API key passthrough

---

## Verification

1. **Import test**: `python -c "from colony.cluster import RemoteEmbeddingConfig, OpenAICompatibleEmbeddingDeployment"` succeeds without openai/google-genai installed
2. **YAML test**: Create config with `remote_embedding_config`, verify it parses and creates correct `ClusterConfig`
3. **Integration test**: `colony-env run` with OpenAI embedding configured — verify `LLMCluster.embed()` returns vectors
4. **Mutual exclusivity**: Setting both `embedding_config` and `remote_embedding_config` raises `ValueError`
