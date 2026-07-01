# Training data & fine-tuned serving

Colony provides domain-agnostic building blocks for turning recorded agent runs into LLM training data, collecting human feedback on them, and serving fine-tuned LoRA adapters. These are primitives — a deployment composes them. The pipeline that *uses* them to train models (data lake, trainers, registry) is product-side and lives outside Colony.

## The training record

`colony.training.TrainingRecord` is one conversational example in exactly one of three shapes, with a projection to each trainer-native view:

| `kind` | Holds | Projection |
|--------|-------|------------|
| `sft` | `messages` (a chat trajectory) | `to_sft()` → `{"messages": [...]}` |
| `preference` | `prompt` + `chosen`/`rejected` | `to_dpo()` → `{"prompt", "chosen", "rejected"}` |
| `rl_prompt` | `prompt` only | `to_grpo()` → `{"prompt": [...]}` |

A record is validated to be well-formed for its `kind` at construction, so it never carries a half-populated view. `source`, `reward_source`, and `coverage_tag` are open strings — callers use their own vocabularies. `records_to_jsonl` / `records_from_jsonl` serialize one record per line.

## Capturing records from spans

The observability tracer already records every agent step as a span. Recording-grade capture and two recorders turn those into records.

**`TracingConfig.recording_grade`** (default off). Off keeps tracing cheap (input/output summaries truncated to ~1.5 KB). On lifts truncation so action and inference spans carry their full payload — the fidelity a training recorder needs. Set it on deployments whose traces feed training.

Two recorders (pure functions over spans):

- `records_from_spans(spans)` — one `sft` record per `INFER` span (a prompt→completion pair, the canonical SFT unit).
- `trajectory_records_from_spans(spans)` — one structured `sft` record per `(run, agent)`: the agent's decisions as `assistant` turns carrying the dispatched action as `tool_calls`, interleaved with the tool results as `tool` turns.

## Human feedback

Reviewers rate individual spans up/down (with an optional note) from the **Traces** tab, and rate agent responses from the **chat** panel. Both land in one store, `span_feedback`:

- Traces tab → `POST /traces/{trace_id}/spans/{span_id}/feedback`. The spans GET returns each span's feedback inline.
- Chat thumb → `POST /chat/sessions/{session_id}/messages/{message_id}/feedback`. The thumb is recorded against the **INFER span that produced the message** (resolved server-side), so chat and trace feedback feed one signal.

A rating must be `up` or `down`; re-rating updates in place (one rating per author per span).

## Curation primitives

Pure transformations over record lists, composed as needed:

- `dedup_records` — drop exact duplicates (by content, ignoring per-record ids/provenance).
- `decontaminate_against(records, held_out)` — drop records present in a held-out/gold set, so eval data never leaks into training.
- `dedup_near_duplicates(records, vectors)` / `drop_near(records, vectors, reference_vectors)` — near-match versions; the caller supplies vectors (embedding is the caller's job).
- `balance_by(records, key)` — cap each group of a chosen dimension.
- `assemble_snapshot(sources, mix_ratios=…)` — combine sources into one content-addressed `Snapshot` (hash + coverage manifest). The hash excludes ids/provenance, so identical data yields the same version.

## Serving a fine-tuned adapter (multi-LoRA)

Give a deployment `LoRAAdapterConfig`s via `LLMDeploymentConfig.lora_adapters` and `VLLMDeployment` enables vLLM multi-LoRA (`enable_lora`, `max_loras`, `max_lora_rank`) at init, resolves each adapter to a `LoRARequest` (downloading S3 weights when configured), and serves a request against `request.requirements.lora_adapter_id`. A request for an adapter the deployment doesn't serve fails loudly rather than silently using the base model. Routing already selects deployments that carry the requested adapter.

For a model served **outside** the cluster — a fine-tuned adapter behind a self-hosted, OpenAI-compatible vLLM — use the `vllm` remote provider: `RemoteLLMDeploymentConfig(provider="vllm", base_url="http://…/v1", model_name=…)`. It is the remote-client counterpart to the in-cluster `VLLMDeployment` (prefix caching is server-side; no per-token cost). The provider registry resolves `"vllm"` to `VllmRemoteDeployment`, alongside `anthropic` and `openrouter`.

## Gated self-improvement (closing the loop)

`promotion_gate(candidate, incumbent, *, min_gain, guards)` decides whether a freshly trained model should replace the incumbent: it must beat the incumbent's mean reward by `min_gain` **and** pass every guard. The shipped guards are `collapse_guard` (output diversity dropped — `distinct_ratio`) and `reward_hack_guard` (degenerate outputs scoring reward — `degenerate_ratio`); the guard set and the degeneracy predicate are injectable. `summarize_eval(completions, rewards)` builds the `EvalSummary` the gate reads.

`run_flywheel(assemble, train, evaluate, publish, …)` chains the loop (assemble → train → evaluate → **gate** → publish) with those steps as injectable seams, so it is generic and testable; `publish` runs only when the gate promotes. This is the primitive a domain (e.g. CPS) wires its own corpus, trainer, evaluator, and serving catalog into.
