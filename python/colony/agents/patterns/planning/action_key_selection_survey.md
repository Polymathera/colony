# Survey: Strategies for Precise LLM Action Key Selection in Agentic Systems

**Date**: 2026-03-06
**Context**: Colony agents use compound action keys like `ConsciousnessCapability.b79b5858.consciousness_update_self_concept`. LLMs frequently emit **truncated keys** (e.g., `consciousness_update_self_concept`) or **malformed keys** (e.g., `working.WorkingMemoryCapability.6cbcb1af.working_memory_store`), causing "Unknown action type" errors.

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Taxonomy of Strategies](#2-taxonomy-of-strategies)
3. [Category A: Prompt Engineering Strategies](#3-category-a-prompt-engineering-strategies)
4. [Category B: Structured Output & Constrained Generation](#4-category-b-structured-output--constrained-generation)
5. [Category C: Action Space Reduction](#5-category-c-action-space-reduction)
6. [Category D: Post-Processing & Error Recovery](#6-category-d-post-processing--error-recovery)
7. [Category E: Alternative Action Representations](#7-category-e-alternative-action-representations)
8. [Category F: Fine-Tuning & Specialized Models](#8-category-f-fine-tuning--specialized-models)
9. [Category G: Hybrid & Multi-Stage Approaches](#9-category-g-hybrid--multi-stage-approaches)
10. [LLM-Specific Considerations](#10-llm-specific-considerations)
11. [Benchmarks & Evaluation](#11-benchmarks--evaluation)
12. [Recommendations for Colony](#12-recommendations-for-colony)
13. [References](#13-references)

---

## 1. Problem Analysis

### 1.1 Failure Modes

Our system exhibits three distinct failure modes when LLMs select action keys:

| Failure Mode | Example | Frequency |
|---|---|---|
| **Suffix truncation** | `consciousness_get_self_concept` instead of `ConsciousnessCapability.b79b5858.consciousness_get_self_concept` | High |
| **Prefix mangling** | `working.WorkingMemoryCapability.6cbcb1af.working_memory_store` (added spurious `working.` prefix) | Medium |
| **Hallucination** | Inventing a plausible action key that doesn't exist | Low |

### 1.2 Root Causes

1. <mark>**Tokenization mismatch**: Compound keys span many tokens. The LLM has no built-in concept that the entire dotted string is an atomic identifier.</mark>
2. <mark>**Attention dilution**: In long prompts with many action descriptions, the model's attention to the exact prefix weakens.</mark>
3. <mark>**Pattern completion bias**: LLMs are trained on code where method names (the suffix) are the primary identifier. The `ClassName.hash.method` pattern is unusual, and the model defaults to the most "meaningful" portion.</mark>
4. <mark>**Context window effects**: Tool selection accuracy degrades as the number of tools and overall context length grows. Studies show accuracy drops 10-30% when going from 10 to 100+ tools.</mark>
5. <mark>**Hash opacity**: The `b79b5858` segment is a meaningless hash to the LLM — it has no semantic content to anchor on, making it easy to drop or mangle.</mark>

### 1.3 Why This Is Hard

Unlike simple enum classification (pick from 3-5 options), our system has:
- **Dynamic action sets**: Actions change based on agent capabilities
- **Compound keys**: Not simple names but `Prefix.Hash.Suffix` patterns
- **Prompt-based selection**: We generate plans in JSON, not using native function calling
- **Multi-model support**: Must work across Claude, GPT-4, open-source models

---

## 2. Taxonomy of Strategies

```
                        ┌─── Prompt Engineering (format, examples, reinforcement)
                        │
                        ├─── Structured Output (JSON schema, grammar, enum)
                        │
                        ├─── Action Space Reduction (retrieval, filtering, pagination)
Precise Action     ─────┤
Key Selection           ├─── Post-Processing (fuzzy match, retry, self-correction)
                        │
                        ├─── Alternative Representations (numeric IDs, aliases, code)
                        │
                        ├─── Fine-Tuning (LoRA, tool-calling specialization)
                        │
                        └─── Hybrid / Multi-Stage (2-stage selection, chain-of-thought)
```

Each category is analyzed below with concrete techniques, trade-offs, and applicability to Colony.

---

## 3. Category A: Prompt Engineering Strategies

### A1. XML-Structured Action Descriptions (Implemented)

**Technique**: Wrap action descriptions in XML tags with explicit `key=` attributes, so the LLM sees the key as a clearly delimited attribute rather than inline text.

```xml
<action-group key="ConsciousnessCapability.b79b5858">
  <description>Agent self-awareness and metacognition</description>
  <action key="ConsciousnessCapability.b79b5858.consciousness_update_self_concept">
    Update the agent's self-concept...
  </action>
</action-group>
```

**Status**: Already implemented as `XMLPromptFormatting`. Partially effective — reduced but did not eliminate the issue.

**Why partial**: XML tags provide structural cues, but the LLM still needs to *copy* a long compound string token-by-token. The attention mechanism can lose track mid-string.

**Source**: <mark>Anthropic recommends XML tags for structuring Claude prompts. Claude models are specifically tuned to parse XML structure.</mark>

### A2. Explicit Copy Instructions

**Technique**: Add explicit meta-instructions that emphasize exact copying.

```
CRITICAL: The action_type value MUST be copied EXACTLY from the key= attribute
of an <action> element above. Include ALL parts: ClassName.hash.method_name.
Do NOT abbreviate, truncate, or modify the key in any way.
Incorrect: "consciousness_get_self_concept"
Correct: "ConsciousnessCapability.b79b5858.consciousness_get_self_concept"
```

**Effectiveness**: Moderate improvement (10-20% error reduction in practice). Works better with Claude and GPT-4 than with smaller models.

**Trade-offs**:
- (+) Zero implementation cost
- (+) Works with any LLM
- (-) Uses extra tokens
- (-) Not reliable under high cognitive load (complex plans)
- (-) LLMs can "forget" instructions in long contexts

### A3. Few-Shot Examples with Full Keys

**Technique**: Provide 1-3 examples of correctly formatted plan output showing full action keys.

```
Example plan output:
{
  "actions": [
    {
      "action_type": "WorkingMemoryCapability.6cbcb1af.working_memory_store",
      "description": "Store analysis results",
      "parameters": {"key": "analysis", "value": "..."}
    }
  ]
}
```

**Effectiveness**: High for the specific action keys shown in examples. Limited for unseen keys (the LLM generalizes the *pattern* but may still truncate novel keys).

**Trade-offs**:
- (+) Strong signal for pattern following
- (+) Works well with all LLMs (in-context learning is universal)
- (-) Token cost scales with number of examples
- (-) Examples may become stale if action set changes
- (-) Risk of the LLM over-fitting to example keys

**Source**: Anthropic's "Tool Use Examples" beta feature (2025) formalizes this pattern — providing exemplar tool calls so the model learns correct usage patterns beyond schema alone.

### A4. Key Echo / Confirmation Pattern

**Technique**: After listing actions, add a confirmation block that echoes all valid keys in a compact format.

```
Valid action_type values (copy exactly):
1. ConsciousnessCapability.b79b5858.consciousness_update_self_concept
2. ConsciousnessCapability.b79b5858.consciousness_get_self_concept
3. WorkingMemoryCapability.6cbcb1af.working_memory_store
...

Use ONLY values from this list. The action_type field must match one of these EXACTLY.
```

**Effectiveness**: Moderate-high. The redundancy helps the LLM "see" the full key twice (once in description, once in the echo list), reinforcing the complete form.

**Trade-offs**:
- (+) Simple to implement
- (-) Doubles the token cost of action keys
- (-) Still relies on the LLM to copy correctly

### A5. <mark>Semantic Key Design</mark>

**Technique**: Redesign the key format to be more "LLM-friendly" — shorter, more meaningful, and without opaque hashes.

Current: `ConsciousnessCapability.b79b5858.consciousness_update_self_concept`
Proposed alternatives:
- `consciousness.update_self_concept` (drop hash, use short prefix)
- `CONSCIOUSNESS__update_self_concept` (uppercase prefix, double underscore separator)
- `cap:consciousness:update_self_concept` (uniform prefix scheme)

<mark>**Effectiveness**: High — reduces the fundamental difficulty. Shorter, more regular keys are easier for LLMs to reproduce.</mark>

**Trade-offs**:
- (+) Addresses <mark>the root cause (key complexity)</mark>
- (+) Fewer tokens per key
- (-) Requires changes to `ActionDispatcher` key generation
- (-) Hash serves a disambiguation purpose (multiple instances of same capability class)
- (-) Breaking change if keys are persisted/logged

### A6. Structured Prompt with Numbered References

**Technique**: Assign each action a short numeric ID and instruct the LLM to use the ID, with a lookup table that maps IDs to full keys.

```
## Available Actions

| # | Action Key | Description |
|---|-----------|-------------|
| 1 | ConsciousnessCapability.b79b5858.consciousness_update_self_concept | Update self-concept |
| 2 | ConsciousnessCapability.b79b5858.consciousness_get_self_concept | Get self-concept |
| 3 | WorkingMemoryCapability.6cbcb1af.working_memory_store | Store to memory |

Output format: "action_type": <number from # column above>
```

The framework then maps the number back to the full key.

**Effectiveness**: Very high. <mark>LLMs are excellent at producing single integers. Eliminates the string-copying problem entirely.</mark>

**Trade-offs**:
- (+) Near-perfect accuracy for key selection
- (+) Fewer output tokens
- (-) Numbering must be consistent between prompt and parsing
- (-) Loses human readability in raw plan output
- (-) Numbers are positional — adding/removing actions shifts all IDs
- (-) The plan output becomes less self-documenting

---

## 4. Category B: Structured Output & Constrained Generation

### B1. Native Function Calling / Tool Use APIs

**Technique**: Instead of prompt-based action selection, use the LLM provider's native function calling API (OpenAI `tools`, Anthropic `tool_use`). The API constrains the model to select from defined tool names.

**How it works**:
- OpenAI: Define tools with `name` field; model outputs `tool_calls[].function.name` which is constrained to registered names
- Anthropic: Define tools with `name` field; model outputs `tool_use` blocks with constrained tool names
- Tool names have strict format requirements (e.g., OpenAI: `^[a-zA-Z0-9_-]{1,64}$`)

**Effectiveness**: Very high for tool *name* selection (the API constrains it). However:
- Our keys contain dots (`.`) which violate OpenAI's name format
- We use free-form JSON plans, not single tool calls
- Our planner generates *multiple* actions in one response (a plan), not one tool call at a time

**Trade-offs**:
- (+) Near-perfect name accuracy when applicable
- (+) Provider handles the constrained generation internally
- (-) Name format restrictions (no dots, max 64 chars for OpenAI)
- (-) Doesn't support generating a *list* of tool calls in one response (varies by provider)
- (-) Ties the system to a specific provider's API format
- (-) Can't use with open-source models served via vLLM (unless they support the tool calling protocol)

<mark>**Applicability to Colony**: Would require restructuring the planning interface from "generate a plan JSON" to "make a sequence of tool calls." This is a fundamental architectural change. However, a hybrid approach (use native tool calling for single-action selection, prompt-based for plan generation) could work.</mark>

**Source**: OpenAI Structured Outputs, Anthropic Tool Use API

### B2. JSON Schema with Enum Constraint

**Technique**: Use OpenAI's Structured Outputs (`strict: true`) or equivalent to constrain `action_type` to a JSON schema `enum` of valid keys.

```json
{
  "type": "object",
  "properties": {
    "action_type": {
      "type": "string",
      "enum": [
        "ConsciousnessCapability.b79b5858.consciousness_update_self_concept",
        "ConsciousnessCapability.b79b5858.consciousness_get_self_concept",
        "WorkingMemoryCapability.6cbcb1af.working_memory_store"
      ]
    }
  },
  "required": ["action_type"],
  "additionalProperties": false
}
```

**How it works**: The provider uses constrained decoding (vocabulary masking + grammar) to ensure the output conforms to the schema. With `strict: true`, the output is **guaranteed** to be valid.

**Effectiveness**: 100% accuracy for key selection when the schema is properly constructed. OpenAI reports "our evaluations demonstrated a 100% score on our structured outputs benchmark."

**Trade-offs**:
- (+) Mathematically guaranteed correct keys
- (+) Available from OpenAI, Anthropic (JSON mode), and via constrained decoding libraries for open-source models
- (-) Schema must be regenerated per request (dynamic action set)
- (-) First request with a new schema has compilation latency (OpenAI caches schemas)
- (-) Plan output structure must conform to JSON Schema (nested arrays of actions with enum fields)
- (-) Anthropic's JSON mode is less strict than OpenAI's — no enum guarantee
- (-) Large enums (100+ keys) can cause compilation delays with some constrained decoding engines

**Applicability to Colony**: Very promising. The plan output is already JSON. We could define a JSON schema per planning request with the current action keys as an enum. This requires provider-level support but works with OpenAI and open-source models (via Outlines/XGrammar).

> **NOTE**: The `Agent.infer` method already supports a parameter `json_schema` which can be used to pass the schema, and it is used throughout the codebase. It is supported by our vLLM and OpenRouter deployments. When you created the `AnthropicLLMDeployment`, you didn't hook up the `json_schema` parameter to the underlying Anthropic API call. To use this feature, you would need to modify the planning code to generate a JSON schema with the current action keys and pass it to `infer`.


**Source**: OpenAI Structured Outputs (2024), JsonSchemaBench (2025)

### B3. Grammar-Constrained Decoding (Outlines, XGrammar, llguidance)

**Technique**: For self-hosted open-source models (via vLLM, llama.cpp), use grammar-constrained decoding to enforce output structure at the token level.

**Libraries**:
- **Outlines** (dottxt): Compiles JSON Schema / regex into FSM, masks invalid tokens at each generation step. Pioneered the FSM approach.
- **XGrammar**: Split tokens into context-independent and context-dependent sets. Achieves CFG-level expressiveness with FSM-level performance. Up to 100x faster than traditional grammar methods.
- **llguidance** (Microsoft): Enforces arbitrary CFG on LLM output. ~50μs CPU time per token for 128k tokenizer. Negligible startup.
- **lm-format-enforcer**: Character-level enforcement, supports multiple backends.
- **vLLM structured decoding**: Integrates Outlines and XGrammar natively.

**How it works**: At each decoding step, compute the set of valid next tokens based on the grammar/schema and mask all others. The LLM can only produce tokens that lead to valid output.

**Effectiveness**: 100% structural compliance. However, quality of *content* within the structure may degrade — the model is forced into valid forms but may choose semantically wrong options from the valid set.

**Trade-offs**:
- (+) Guaranteed valid output
- (+) Works with any model (it's a decoding-time constraint)
- (+) No prompt changes needed
- (-) Only works with self-hosted models (need access to logits)
- (-) Complex schemas can cause slow compilation (Outlines: 40s to 10min for large schemas)
- (-) XGrammar and vLLM solve the compilation issue but require specific infrastructure
- (-) May degrade generation quality (perplexity increases under tight constraints)

<mark>**Applicability to Colony**: Directly applicable when using open-source models via vLLM. Colony already uses vLLM. This is the **strongest guarantee** for self-hosted models.</mark>

**Source**: Outlines (2023), XGrammar (2024), llguidance (2024), vLLM Blog (2025), JsonSchemaBench (2025)

### B4. Logit Bias / Token Masking

**Technique**: For API-based models that expose logit bias (OpenAI), bias the model toward tokens that appear in valid action keys.

**How it works**: Tokenize all valid action keys, compute the set of tokens that could appear in any valid key, and apply negative logit bias to all other tokens when generating the `action_type` field.

**Effectiveness**: Moderate. Coarse-grained — biases toward the right token *vocabulary* but doesn't enforce the right *sequence*.

**Trade-offs**:
- (+) Works with API models that expose logit bias
- (-) Coarse (token-level, not sequence-level)
- (-) Difficult to implement correctly (must track position in the key)
- (-) OpenAI limits logit bias to 300 tokens
- (-) Anthropic doesn't expose logit bias

**Applicability to Colony**: Limited. Too coarse for compound keys.

---

## 5. Category C: Action Space Reduction

### C1. <mark>Retrieval-Augmented Tool Selection (Tool RAG)</mark>

**Technique**: Instead of including all actions in the prompt, use embedding-based retrieval to select only the most relevant actions for the current task/query.

**How it works**:
1. <mark>Embed all action descriptions into a vector store</mark>
2. <mark>Given the agent's current goal/context, retrieve top-K most relevant actions</mark>
3. <mark>Include only those K actions in the **planning prompt**</mark>

> **NOTE**: This only replaces the first stage of Colony's two-stage selection (scope selection remains), but it can dramatically reduce the number of actions the LLM must choose from in the second stage.

**Results**:
- Anthropic's RAG-MCP: Boosted tool selection accuracy from 13% to 43% on large toolsets (500+ tools) with ~50% prompt token reduction
- AWS Bedrock + S3 Vectors: 82.3% accuracy (up from 75.8%), 91.9% recall, 21% faster latency
- Red Hat Tool RAG: "Next breakthrough in scalable AI agents"

**Trade-offs**:
- (+) Dramatically reduces prompt size
- (+) Higher accuracy by reducing choice confusion
- (+) Scales to thousands of tools
- (-) Retrieval may miss relevant actions (recall < 100%)
- (-) Requires embedding infrastructure
- (-) Latency from retrieval step
- (-) Less useful when the total action set is small (< 30 actions)

**Applicability to Colony**: Applicable when agents have many capabilities. Colony agents currently have 10-30 actions — borderline for retrieval benefit. Would become critical if the action set grows.

**Source**: RAG-MCP (2025), Tool-to-Agent Retrieval (2025), ToolSEE, AWS Blog (2025)

### C2. Anthropic Tool Search Tool (Dynamic Discovery)

**Technique**: Anthropic's advanced tool use feature where tools are marked `defer_loading: true` and Claude discovers them on-demand via a built-in search mechanism.

**How it works**:
1. Only a "tool search" meta-tool is included in the initial prompt
2. Claude decides when it needs a tool and searches for it
3. Matching tool definitions are loaded into context just-in-time
4. Uses regex or BM25 matching (custom embedding search also supported)

**Results**:
- 85% reduction in token usage
- Opus 4 accuracy improved from 49% to 74% on MCP evaluations
- Opus 4.5 improved from 79.5% to 88.1%

**Trade-offs**:
- (+) Massive context savings
- (+) Claude-native, well-tuned
- (-) Anthropic-specific (not portable)
- (-) Adds a tool-search round-trip before each tool use
- (-) Doesn't solve the key *accuracy* problem — just reduces choice set size

**Applicability to Colony**: Could be layered on top of existing capability system. More relevant for reducing prompt size than fixing key accuracy directly.

**Source**: Anthropic Advanced Tool Use (2025), Claude API Docs

### C3. Category/Group Filtering

**Technique**: Present action groups (capabilities) first, let the LLM select a group, then present only that group's actions.

**Step 1 prompt**: "Which capability group is relevant? Options: Consciousness, WorkingMemory, Planning, ..."
**Step 2 prompt**: "Select an action from ConsciousnessCapability: [list of 3-5 actions]"

**Effectiveness**: High. Reduces the action set at each step. The LLM selects from a small set at each stage.

**Trade-offs**:
- (+) Small choice sets = high accuracy
- (+) Works with any LLM
- (-) Two LLM calls instead of one (latency, cost)
- (-) The group selection step itself can fail
- (-) Doesn't work well when the LLM needs to select multiple actions from different groups in one plan

**Applicability to Colony**: Colony already has a 2-stage action selection mechanism (scope selection). This pattern is proven in the codebase. Could be extended to make the group→action selection more explicit.

---

## 6. Category D: Post-Processing & Error Recovery

### D1. Fuzzy / Suffix Matching Fallback

**Technique**: When the exact key isn't found, try matching the LLM's output against known keys using suffix matching, Levenshtein distance, or other similarity measures.

```python
def resolve_action_key(raw_key: str, valid_keys: list[str]) -> str | None:
    # Exact match
    if raw_key in valid_keys:
        return raw_key
    # Suffix match (handles truncation)
    suffix_matches = [k for k in valid_keys if k.endswith(f".{raw_key}")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    # Levenshtein distance
    from difflib import get_close_matches
    close = get_close_matches(raw_key, valid_keys, n=1, cutoff=0.6)
    if close:
        return close[0]
    return None
```

**Effectiveness**: High for suffix truncation (the most common failure mode). Risky for hallucinated keys (may match the wrong action).

**Trade-offs**:
- (+) Catches the most common error (truncation) with high confidence
- (+) Zero prompt overhead
- (+) Works with any LLM
- (-) Ambiguous when multiple keys share the same suffix
- (-) Silent error correction may mask deeper issues
- (-) Levenshtein can match semantically wrong keys
- (-) Should be a safety net, not a primary strategy

**Applicability to Colony**: **Strongly recommended as a safety net** regardless of which primary strategy is chosen. Suffix matching alone would fix most observed errors.

### D2. Retry with Error Feedback

**Technique**: When action key parsing fails, re-prompt the LLM with the error message and list of valid keys.

```
Your previous response contained an invalid action_type: "consciousness_get_self_concept"
This is not a valid action key. Valid keys are:
- ConsciousnessCapability.b79b5858.consciousness_get_self_concept
- ConsciousnessCapability.b79b5858.consciousness_update_self_concept
- ...
Please regenerate the plan with corrected action_type values.
```

**Effectiveness**: High on retry (LLMs are good at self-correction when given specific feedback). Studies show 80-90% recovery rate on first retry.

**Trade-offs**:
- (+) High recovery rate
- (+) Works with any LLM
- (-) Additional LLM call (latency, cost)
- (-) May need multiple retries
- (-) Should have a retry limit to prevent infinite loops
- (-) Increases overall plan generation time

**Applicability to Colony**: Good fallback strategy. Can be combined with fuzzy matching (try fuzzy first, retry only if fuzzy also fails).

**Source**: CRITIC (2023), Instructor library self-correction, LangGraph retry patterns

### D3. Self-Verification / Chain-of-Verification

<mark>**Technique**: After generating a plan, ask the LLM to verify each `action_type` against the available actions list.</mark>

```
Verify: Is "consciousness_get_self_concept" in the available actions list?
If not, what is the correct full action key?
```

**Effectiveness**: Moderate. The LLM may make the same mistake twice, or may "verify" an incorrect key as correct.

**Trade-offs**:
- (+) Can catch errors before execution
- (-) Additional LLM call
- (-) Not reliable — the same model that made the error may "verify" it as correct
- (-) Better to use a different strategy (fuzzy matching) for verification

**Source**: Self-Verification Prompting (2023), S²R (2025)

---

## 7. Category E: Alternative Action Representations

### E1. Numeric ID Mapping

**Technique**: Assign each action a sequential integer ID. The LLM outputs the number; the framework resolves it to the full key.

```
Actions:
[1] Update self-concept (consciousness)
[2] Get self-concept (consciousness)
[3] Store to working memory
[4] Recall from working memory
...

Output: "action_id": 1
→ Resolved to: ConsciousnessCapability.b79b5858.consciousness_update_self_concept
```

**Effectiveness**: Very high. Single-token output. No string copying required.

**Trade-offs**:
- (+) Near-perfect accuracy
- (+) Minimal output tokens
- (+) Works with any LLM
- (-) Numbers are meaningless — plan output is not human-readable
- (-) Numbering is positional and fragile
- (-) LLM reasoning about action sequences is harder with opaque IDs
- (-) Multiple numbering schemes in one prompt can confuse the LLM

### E2. Short Alias Mapping

**Technique**: Generate short, unique, human-readable aliases for each action. The LLM uses the alias; the framework resolves to the full key.

```xml
<action key="ConsciousnessCapability.b79b5858.consciousness_update_self_concept"
        alias="update_self_concept">
  Update the agent's self-concept...
</action>
```

Output: `"action_type": "update_self_concept"`
→ Resolved to: `ConsciousnessCapability.b79b5858.consciousness_update_self_concept`

**Effectiveness**: High. Short names are easy for LLMs to reproduce. Risk of alias collision (two capabilities with same method name).

**Trade-offs**:
- (+) Human-readable aliases
- (+) Short, easy to reproduce
- (+) The alias IS the semantically meaningful part the LLM naturally wants to output
- (-) Must handle collisions (e.g., two capabilities both having `store` method)
- (-) Alias generation adds complexity
- (-) The plan output uses aliases, not full keys (less self-documenting)

**Applicability to Colony**: **Very promising.** The method name suffix (`consciousness_update_self_concept`) is already the natural alias. The main risk is collision when two capabilities have the same method name — can be resolved by prefixing with a short capability name.

### E3. CodeAct: Code Instead of JSON

<mark>**Technique**: Instead of having the LLM output JSON with action keys, have it output executable code (Python) that calls action functions directly.</mark>

```python
# Instead of: {"action_type": "ConsciousnessCapability.b79b5858.consciousness_update_self_concept", ...}
# The LLM outputs:
await agent.consciousness.update_self_concept(new_concept="...")
result = await agent.working_memory.store(key="analysis", value=result)
```

**Effectiveness**: Very high. Code is natural for LLMs (trained on massive code corpora). Method names in code are short and familiar. Autocomplete-like behavior naturally produces valid identifiers.

**Results**: CodeAct achieves up to 20% higher success rate across 17 LLMs compared to JSON/text action formats (ICML 2024).

**Trade-offs**:
- (+) Leverages the LLM's strongest capability (code generation)
- (+) Natural composability (variables, loops, error handling)
- (+) Shorter action identifiers (method names, not compound keys)
- (+) Self-debugging via error messages
- (-) Requires a code execution sandbox
- (-) Security concerns (arbitrary code execution)
- (-) Harder to validate and audit than declarative plans
- (-) Significant architectural change

**Source**: CodeAct (ICML 2024), AgentScript (2025), Anthropic Programmatic Tool Calling (2025)

**Applicability to Colony**: Interesting for future exploration but requires fundamental redesign of the planning → execution pipeline.

> **NOTE**: Colony already support a Python REPL as an agent capability.


### E4. Anthropic Programmatic Tool Calling

**Technique**: Anthropic's advanced tool use feature where Claude orchestrates tools through code rather than individual API round-trips.

**How it works**: Claude writes code that calls multiple tools, processes their outputs, and controls what information enters its context window. This combines the benefits of CodeAct with native tool use.

**Trade-offs**:
- (+) Combines code generation with tool calling guarantees
- (+) Claude handles the code execution internally
- (-) Anthropic-specific
- (-) Beta feature, may change

**Source**: Anthropic Advanced Tool Use (2025)

---

## 8. Category F: Fine-Tuning & Specialized Models

### F1. Tool-Calling Fine-Tuned Models

**Technique**: Use models specifically fine-tuned for function/tool calling.

**Models**:
- **Gorilla** (Berkeley): Fine-tuned LLaMA for API calling. Surpasses GPT-4 on writing API calls. Trained on API documentation. Limited generalization to unseen APIs.
- **NexusRaven-13B** (Nexusflow): Based on CodeLLaMA-13B. Matches GPT-3.5 in zero-shot function calling. With retrieval augmentation, surpasses GPT-4. 30% higher success rate than GPT-4 on specific domains.
- **ToolLLaMA** (Tsinghua): Fine-tuned on ToolBench dataset (16,000+ real-world APIs). Demonstrates comparable performance to ChatGPT. Good generalization to unseen APIs.

**Trade-offs**:
- (+) Much better tool selection accuracy than base models
- (+) Can handle large tool sets
- (-) Requires hosting specialized models
- (-) May lag behind frontier models on general reasoning
- (-) Training data becomes stale as APIs evolve

**Applicability to Colony**: Relevant if Colony moves to self-hosted models. Could fine-tune a Colony-specific adapter.

### F2. LoRA / Adapter Fine-Tuning

**Technique**: Fine-tune a LoRA adapter on Colony's specific action key format using examples of correct plan generation.

**Data**: Generate synthetic (prompt, correct_plan) pairs from real agent runs.

**Trade-offs**:
- (+) Highly effective for specific formats
- (+) LoRA is lightweight (small adapter, base model unchanged)
- (-) Requires training infrastructure
- (-) Must retrain when action set changes
- (-) Not applicable to API-based models (Claude, GPT-4)

### F3. AVATAR: Contrastive Reasoning for Tool Use

**Technique**: Optimizes LLM agents for tool usage via contrastive reasoning. Trains the model to distinguish between correct and incorrect tool calls using positive/negative examples.

**Source**: AVATAR (Stanford, NeurIPS 2024)

---

## 9. Category G: Hybrid & Multi-Stage Approaches

### G1. Two-Stage Selection (Category → Action)

**Technique**: Split action selection into two stages:
1. **Stage 1**: Select the capability/group (short name, high accuracy)
2. **Stage 2**: Select the specific action within that group (small set, high accuracy)

**Implementation**:
```
Stage 1: "Which capability groups are needed for this plan?"
→ LLM outputs: ["consciousness", "working_memory"]

Stage 2 (per group): "From ConsciousnessCapability, which actions?"
→ LLM outputs: ["update_self_concept", "get_self_concept"]

Framework resolves: ConsciousnessCapability.b79b5858.update_self_concept
```

**Effectiveness**: Very high. Each stage has a small choice set (5-10 options). The framework handles key construction.

**Trade-offs**:
- (+) Small choice sets → high accuracy
- (+) LLM only needs to produce short, meaningful names
- (+) Framework constructs the compound key (no string copying)
- (-) Multiple LLM calls (2 per plan generation)
- (-) Stage 1 errors cascade
- (-) More complex orchestration

<mark>**Applicability to Colony**: **Highly recommended.** Colony already has group-level structure (action groups / capabilities). This aligns naturally with the existing `_format_action_descriptions` which iterates over `ActionGroupDescription` objects.</mark>

> **NOTE**: We already do this 2-stage selection.



### G2. Plan-then-Validate Pipeline

<mark>**Technique**: Generate the plan, then validate all action keys. For invalid keys, attempt fuzzy resolution, then retry if necessary.</mark>

```
1. Generate plan (may have key errors)
2. For each action_type in plan:
   a. Exact match? → OK
   b. Suffix match? → Resolve and warn
   c. Fuzzy match (>0.8 similarity)? → Resolve and warn
   d. No match? → Collect error
3. If errors:
   a. Re-prompt with specific error feedback
   b. Or: reject plan and regenerate
```

**Effectiveness**: Very high when combining fuzzy matching with retry. Catches nearly all errors.

**Trade-offs**:
- (+) Robust — multiple fallback layers
- (+) Works with any LLM
- (+) The primary generation can use any format
- (-) Validation adds latency
- (-) Retry doubles cost for failed plans

### G3. AutoTool: Graph-Based Tool Selection

**Technique**: Build a directed graph from historical agent trajectories, where nodes are tools and edges represent transition probabilities. Use the graph to predict the next likely tool, bypassing LLM inference.

**Results**: Reduces LLM calls by 15-25%, token consumption by 10-40%.

**Trade-offs**:
- (+) Reduces LLM inference cost
- (+) Leverages historical patterns
- (-) Requires trajectory data
- (-) Cold start problem
- (-) May miss novel tool combinations

**Source**: AutoTool (2025)

### G4. Dynamic ReAct: Search-and-Load

**Technique**: Extension of ReAct that handles large tool sets by dynamically loading relevant tools at each reasoning step.

**How it works**: At each step, the agent searches for relevant tools based on the current observation, loads them, and selects from the loaded subset. Achieves intelligent tool selection with minimal computational overhead.

**Source**: Dynamic ReAct (2025)

### G5. Execution Sketch Constraint

**Technique**: Generate an "execution sketch" that constrains the action space at each step to a subset of relevant tools. From the paper on "Robust and Efficient Tool Orchestration via Layered Execution Structures with Reflective Correction" (2025).

---

## 10. LLM-Specific Considerations

### Claude (Anthropic)

| Aspect | Details |
|--------|---------|
| **XML parsing** | Claude is specifically tuned for XML. XML-structured prompts work well for structural cues. |
| **Native tool use** | Anthropic's tool use API constrains tool names. Tool names must match `^[a-zA-Z0-9_-]+$`. |
| **Tool Search Tool** | Dynamic tool discovery reduces context bloat. 85% token reduction. |
| **Programmatic tool calling** | Code-based orchestration (beta). |
| **JSON mode** | Supports JSON output but without strict schema enforcement (less strict than OpenAI). |
| **Best strategy** | XML prompting + short aliases + fuzzy fallback. Or native tool use if feasible. |
| **Known weakness** | May add creative prefixes to keys (seen: `working.WorkingMemory...`). |

### GPT-4 / GPT-4o (OpenAI)

| Aspect | Details |
|--------|---------|
| **Structured Outputs** | `strict: true` with JSON Schema. 100% schema compliance. Enum enforcement. |
| **Function calling** | Native tool names constrained by schema. Tool names: `^[a-zA-Z0-9_-]{1,64}$`. |
| **Best strategy** | JSON Schema enum constraint (guaranteed correct). Or structured outputs with `strict: true`. |
| **Known weakness** | Schema compilation latency on first request with new schema. Large enums may be slow. |
| **Logit bias** | Supports up to 300 tokens. Coarse but usable for simple constraints. |

### Gemini (Google)

| Aspect | Details |
|--------|---------|
| **Function calling** | Supports function declarations. Less mature than OpenAI/Anthropic. |
| **Structured output** | JSON mode available. Enum support improving. |
| **Best strategy** | Function calling with short names + alias resolution. |

### Open-Source (Llama 3, Mistral, Qwen, DeepSeek)

| Aspect | Details |
|--------|---------|
| **Constrained decoding** | Full access to logits → can use Outlines, XGrammar, llguidance for guaranteed compliance. |
| **Tool calling** | Varies by model. Some (Llama 3.1+, Qwen 2.5) have built-in tool calling. Others require prompting. |
| **Fine-tuning** | Can fine-tune LoRA adapters for Colony's specific key format. |
| **Best strategy** | Grammar-constrained decoding (XGrammar/vLLM) for guaranteed keys. Or fine-tuned adapter. |
| **Known weakness** | Smaller models (7B-13B) are significantly worse at tool selection than frontier models. |

### Scaling Behavior

Research and benchmarks show:
- **10 tools**: Most models achieve >90% accuracy
- **30 tools**: Accuracy drops to 70-85% for frontier models
- **100+ tools**: Without retrieval/filtering, accuracy drops to 40-60%
- **500+ tools**: Naive prompting drops to <20%. RAG-based approaches recover to 40-70%

---

## 11. Benchmarks & Evaluation

### Berkeley Function Calling Leaderboard (BFCL)

The de facto standard for evaluating function calling accuracy. V4 evaluates:
- Simple function calls (single tool, single call)
- Multiple function calls (select from many tools)
- Parallel function calls (multiple tools in one response)
- Java, JavaScript, Python function calls
- Multi-turn conversations
- Relevance detection (knowing when NOT to call a tool)

**Key findings**: Top models ace single-turn calls but struggle with memory, dynamic decision-making, and long-horizon reasoning.

### ToolBench

16,000+ real-world APIs organized hierarchically (domain → category → tool → API). Evaluates multi-step tool usage scenarios.

### API-Bank

Evaluates LLMs on API call generation, including parameter filling and API selection.

### Key Metric for Colony

<mark>The most relevant metric is **exact key match accuracy** — does the LLM output the *exact* action key from the available set? This is binary (match or no match) and directly measurable from agent runs.</mark>

Recommended evaluation:
```python
# For each plan generated:
for action in plan.actions:
    if action.action_type in valid_action_keys:
        exact_matches += 1
    elif resolve_fuzzy(action.action_type, valid_action_keys):
        fuzzy_matches += 1  # Would have been caught by fallback
    else:
        misses += 1  # True failure
```

---

## 12. Recommendations for Colony

Based on the survey, here are recommended strategies ordered by **implementation complexity** and **expected impact**.

### Tier 1: Immediate (Low effort, High impact)

These should be implemented first as they provide the best ROI.

#### 1a. Fuzzy/Suffix Matching Fallback (D1)

Add a post-processing step in `ActionDispatcher.dispatch()` that tries suffix matching when exact key lookup fails. This catches the most common error (truncation) with zero prompt overhead.

```python
# In ActionDispatcher or action resolution:
if action_type not in self._action_map:
    # Try suffix match
    suffix_matches = [k for k in self._action_map if k.endswith(f".{action_type}")]
    if len(suffix_matches) == 1:
        action_type = suffix_matches[0]
        logger.warning(f"Resolved truncated action key: {action_type}")
```

**Expected impact**: Catches 70-80% of observed errors.

#### 1b. Explicit Copy Instructions (A2)

Add a 2-3 line instruction in the prompt emphasizing exact key copying with a positive/negative example.

**Expected impact**: Additional 10-15% error reduction.

#### 1c. Short Alias Mapping (E2)

**This is the most impactful single change.** Generate short aliases from action keys and let the LLM use them. The framework resolves aliases to full keys.

```python
class AliasPromptFormatting(PromptFormattingStrategy):
    """Assign short aliases. LLM outputs alias, framework resolves to full key."""

    def _generate_alias(self, full_key: str, all_keys: list[str]) -> str:
        """Extract shortest unambiguous suffix."""
        parts = full_key.split(".")
        suffix = parts[-1]  # e.g., "consciousness_update_self_concept"
        # Check if suffix is unique across all keys
        if sum(1 for k in all_keys if k.endswith(f".{suffix}")) == 1:
            return suffix
        # Prefix with group name for disambiguation
        return f"{parts[0].lower()}.{suffix}"
```

**Expected impact**: 80-90% error reduction. The LLM naturally wants to output the method name — let it.

### Tier 2: Medium effort, High impact

#### 2a. Two-Stage Selection (G1)

For the planning prompt, split into:
1. Goal analysis + capability group selection
2. Per-group action selection

This aligns with Colony's existing `ActionGroupDescription` structure.

**Expected impact**: Near-perfect for the capability/group selection stage. Overall 90%+ accuracy.

#### 2b. JSON Schema Enum Constraint (B2)

When using OpenAI models: use Structured Outputs with the action keys as an enum.
When using vLLM: use XGrammar/Outlines for the same effect.

Requires generating the JSON schema dynamically per planning request.

**Expected impact**: 100% key accuracy (guaranteed by constrained decoding).

#### 2c. Numeric ID Mapping (E1)

As a `PromptFormattingStrategy` variant: assign each action a number, LLM outputs the number, framework resolves.

**Expected impact**: Near-perfect accuracy. But reduced plan readability.

### Tier 3: High effort, Specialized

#### 3a. Native Function Calling Integration (B1)

Restructure the planning interface to use native tool calling APIs when available. Requires significant architectural changes.

#### 3b. CodeAct / Programmatic Tool Calling (E3, E4)

Replace JSON plan output with code-based action specification.

#### 3c. Tool RAG (C1)

For large action sets (100+ tools), use embedding-based retrieval to filter actions before prompting.

#### 3d. Fine-Tuning (F1, F2)

For self-hosted models, fine-tune a LoRA adapter on Colony's action key format.

### Strategy Matrix by LLM

| Strategy | Claude | GPT-4 | Open-Source (vLLM) |
|---|---|---|---|
| XML formatting (A1) | Good | Moderate | Moderate |
| Explicit instructions (A2) | Good | Good | Poor (small models) |
| Few-shot examples (A3) | Good | Good | Moderate |
| Short aliases (E2) | **Best** | **Best** | **Best** |
| Numeric IDs (E1) | Good | Good | Good |
| JSON Schema enum (B2) | N/A | **Best** | **Best** (XGrammar) |
| Native function calling (B1) | Good | **Best** | Varies |
| Fuzzy fallback (D1) | Universal safety net | Universal safety net | Universal safety net |
| Two-stage (G1) | Good | Good | Good |
| Constrained decoding (B3) | N/A | N/A | **Best** |

### Recommended Implementation Order

1. **Now**: Add fuzzy/suffix matching fallback (D1) — safety net for all strategies
2. **Now**: Add explicit copy instructions (A2) — low-effort prompt improvement
3. **Next**: Implement `AliasPromptFormatting` strategy (E2) — highest single-strategy impact
4. **Next**: Implement `NumericIDPromptFormatting` strategy (E1) — alternative to aliases
5. **Later**: Implement JSON Schema enum constraint (B2) for OpenAI/vLLM
6. **Later**: Implement two-stage selection (G1) for large action sets
7. **Future**: Evaluate CodeAct approach (E3) for next-gen planning

---

## 13. References

### Academic Papers
- Wang et al., "Executable Code Actions Elicit Better LLM Agents" (CodeAct), ICML 2024
- Qin et al., "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs", ICLR 2024
- Patil et al., "Gorilla: Large Language Model Connected with Massive APIs", NeurIPS 2024
- Patil et al., "Berkeley Function Calling Leaderboard", ICLR 2024 / ongoing
- Gou et al., "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing", ICLR 2024
- "AVATAR: Optimizing LLM Agents for Tool Usage via Contrastive Reasoning", NeurIPS 2024
- "S²R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning", 2025
- "NexusRaven: A Commercially-Permissive Language Model for Function Calling", 2024
- "AutoTool: Efficient Tool Selection for Large Language Model Agents", 2025
- "AutoTool: Dynamic Tool Selection and Integration for Agentic Reasoning", 2025
- "Tool-to-Agent Retrieval: Bridging Tools and Agents for Scalable LLM Multi-Agent Systems", 2025
- "Dynamic ReAct: Scalable Tool Selection for Large-Scale MCP Environments", 2025
- "Robust and Efficient Tool Orchestration via Layered Execution Structures with Reflective Correction", 2025
- "Generating Structured Outputs from Language Models: Benchmark and Studies" (JsonSchemaBench), 2025
- "XGrammar: Flexible and Efficient Structured Generation", 2024
- "LLM-Based Agents for Tool Learning: A Survey", Data Science and Engineering, 2025
- "Tool and Agent Selection for Large Language Model Agents in Production: A Survey", 2025
- "CompactPrompt: A Unified Pipeline for Prompt and Data Compression", 2025
- Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools", NeurIPS 2023
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023

### Industry Documentation & Blog Posts
- [Anthropic: Introducing Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use)
- [Anthropic: Tool Search Tool Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- [Anthropic: How to Implement Tool Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use)
- [OpenAI: Structured Model Outputs](https://platform.openai.com/docs/guides/structured-outputs)
- [OpenAI: Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [vLLM Blog: Structured Decoding Introduction](https://blog.vllm.ai/2025/01/14/struct-decode-intro.html)
- [Outlines (dottxt): Grammar-Constrained Generation](https://github.com/dottxt-ai/outlines)
- [XGrammar: Flexible and Efficient Structured Generation](https://arxiv.org/abs/2411.15100)
- [llguidance (Microsoft)](https://github.com/guidance-ai/llguidance)
- [AWS: Optimize Agent Tool Selection Using S3 Vectors](https://aws.amazon.com/blogs/storage/optimize-agent-tool-selection-using-s3-vectors-and-bedrock-knowledge-bases/)
- [Red Hat: Tool RAG — Next Breakthrough in Scalable AI Agents](https://next.redhat.com/2025/11/26/tool-rag-the-next-breakthrough-in-scalable-ai-agents/)
- [RAG-MCP: Taming Tool Bloat in the MCP Era](https://www.dakshineshwari.net/post/rag-mcp-taming-tool-bloat-in-the-mcp-era)
- [LangChain: State of Agent Engineering](https://www.langchain.com/state-of-agent-engineering)
- [Instructor: Self-Verification and Structured Output](https://python.useinstructor.com/examples/self_critique/)
- [LlamaIndex: Building Better Tools for LLM Agents](https://www.llamaindex.ai/blog/building-better-tools-for-llm-agents-f8c5a6714f11)
- [BFCL V4 Live Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [Microsoft LLMLingua: Prompt Compression](https://github.com/microsoft/LLMLingua)
- [Constrained Decoding Guide](https://www.aidancooper.co.uk/constrained-decoding/)
- [Structuring Enums for Flawless LLM Results](https://ohmeow.com/posts/2024-07-06-llms-and-enums.html)

### Frameworks & Libraries
- [Outlines](https://github.com/dottxt-ai/outlines) — FSM-based constrained generation
- [XGrammar](https://github.com/mlc-ai/xgrammar) — High-performance grammar-constrained decoding
- [llguidance](https://github.com/guidance-ai/llguidance) — CFG enforcement, ~50μs per token
- [lm-format-enforcer](https://github.com/noamgat/lm-format-enforcer) — Character-level format enforcement
- [Instructor](https://python.useinstructor.com/) — Pydantic-based structured output extraction with self-correction
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Retry and error correction patterns
- [AgentScript](https://github.com/AgentScript-AI/agentscript) — CodeAct Agent SDK with AST-based execution
