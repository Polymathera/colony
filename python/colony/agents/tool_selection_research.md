# Tool Selection in LLM Agent Frameworks: Research Report

**Problem Statement:** When an LLM is given a list of actions with compound keys like `ClassName.hash.method_name`, it often emits only the short suffix (`method_name`) or a slightly wrong key. How do popular frameworks solve this?

---

## 1. OpenAI Function Calling / tool_choice

### Mechanism
OpenAI uses **native function calling** built into the model's training. Tools are passed as a JSON schema in the `tools` parameter of the API request. Each tool has a `name`, `description`, and `parameters` schema.

The model outputs a structured `tool_calls` array where each entry contains the exact `function.name` and `function.arguments` (as a JSON string). This is NOT free-text generation — the model is fine-tuned to emit valid tool call structures.

### Name Constraints
- Function names must match regex `^[a-zA-Z0-9_-]{1,64}$`
- No dots, no compound keys — flat namespace only
- This is a **hard API constraint**, so the problem of compound keys simply cannot arise

### Structured Outputs (strict mode)
- Since late 2024, OpenAI supports `"strict": true` on function definitions
- With strict mode, the model's output is **constrained decoding** — it uses a grammar/mask during generation to guarantee valid JSON matching the schema
- The function name itself is also guaranteed to match one of the defined tools

### tool_choice parameter
- `"auto"` — model decides whether/which tool to call
- `"required"` — model must call at least one tool
- `{"type": "function", "function": {"name": "specific_tool"}}` — forces a specific tool
- `"none"` — model must not call tools

### Large Tool Sets
- OpenAI recommends **fewer than 20 tools** for best performance; quality degrades significantly beyond ~50
- No built-in multi-stage selection — frameworks must implement their own
- Common workaround: use an initial LLM call to select a category/subset, then a second call with only relevant tools

### Error Recovery
- No built-in fuzzy matching — if the model hallucinates a function name (rare with native calling but possible), the API returns it as-is and the application must handle the error
- Applications typically re-prompt the model with an error message indicating the tool was not found

---

## 2. Anthropic Claude Tool Use API

### Mechanism
Similar to OpenAI — tools are passed as structured JSON in the `tools` parameter. Each tool has `name`, `description`, and `input_schema`. Claude outputs a `tool_use` content block with the exact tool `name` and `input` (as structured JSON).

### Name Constraints
- Tool names must match `^[a-zA-Z0-9_-]{1,64}$`
- Same flat namespace, no dots allowed
- The API enforces this at request validation time

### Constrained Selection
- Claude is trained to emit exact tool names from the provided set
- The `tool_choice` parameter supports:
  - `{"type": "auto"}` — model decides
  - `{"type": "any"}` — must use a tool
  - `{"type": "tool", "name": "specific_tool"}` — forces a specific tool
  - `{"type": "none"}` — disables tool use (since 2025)

### Large Tool Sets
- Anthropic documentation suggests keeping tool count manageable (under ~20-30 for best results)
- Performance degrades with many tools, especially if descriptions are similar
- Recommendation: use clear, distinct tool names and descriptions
- No built-in hierarchical/multi-stage selection

### Error Recovery
- If the model emits an invalid tool name (very rare), the application receives it and must handle it
- Best practice: return a `tool_result` with `is_error: true` and a message explaining the valid tools
- The model then self-corrects on the next turn

---

## 3. LangChain / LangGraph

### Mechanism
LangChain uses **model-native tool calling** as the primary mechanism (since ~v0.2). It wraps OpenAI/Anthropic/Google tool calling APIs through a unified interface.

Tools are defined via:
- `@tool` decorator on Python functions
- Subclassing `BaseTool`
- Pydantic models as `args_schema`

LangChain converts these to the provider's native tool format and lets the model handle selection.

### Legacy: ReAct Agent (text parsing)
The older ReAct agent pattern used **regex parsing** of free-text output:
```
Thought: I need to search for X
Action: search_tool
Action Input: "X"
```
This was parsed with regex like `Action:\s*(.*?)[\n]` — extremely brittle and prone to exactly the problem you describe (model emitting wrong action names).

### How Tool Names Work
- Tool names come from the Python function name or explicit `name` parameter
- LangChain sanitizes names to match provider constraints (e.g., replacing dots with underscores)
- For compound names, LangChain recommends using underscores: `class_name_method_name`

### Large Tool Sets — Multi-stage Selection
LangChain/LangGraph supports several strategies:

1. **Dynamic tool selection**: Use a retrieval step (semantic search over tool descriptions) to select a subset of tools before passing them to the model. LangGraph's `ToolNode` can be configured with dynamic tool lists.

2. **Tool namespacing via Toolkits**: Group related tools into "toolkits" (e.g., `SQLDatabaseToolkit`, `GmailToolkit`). The agent first selects a toolkit, then tools within it.

3. **Semantic routing**: Use embeddings of tool descriptions to find the most relevant tools for a given query, then only pass those to the model.

### Error Recovery
- `ToolNode` in LangGraph has `handle_tool_errors=True` — catches exceptions and returns error messages to the model
- `InvalidToolCall` objects capture malformed tool calls (wrong name, invalid JSON args)
- The model sees the error and retries
- No fuzzy matching by default — exact name match required

---

## 4. CrewAI

### Mechanism
CrewAI uses a **custom ReAct-style text parsing** approach (not native function calling by default). The LLM outputs text in a structured format:

```
Thought: ...
Action: tool_name
Action Input: {"param": "value"}
```

CrewAI parses this with regex to extract the tool name and input.

### Tool Name Matching
- Tools are registered with a `name` attribute (simple string, typically snake_case)
- CrewAI does **case-insensitive matching** and strips whitespace
- As of 2024-2025, CrewAI added **fuzzy matching**: if the exact tool name isn't found, it searches for the closest match using string similarity
- The tool name in the prompt is listed clearly, and the system prompt explicitly tells the model the exact names available

### Error Recovery
- If the tool name doesn't match, CrewAI returns an error message to the model: "Tool 'X' not found. Available tools: [list]"
- The model then retries with the correct name
- CrewAI has a configurable `max_retry_limit` for tool execution errors
- Recent versions added `force_tool_output` to require tool use

### Large Tool Sets
- No built-in multi-stage selection
- CrewAI mitigates by assigning specific tools to specific agents (each agent gets only its relevant tools)
- The `tools` parameter on `Agent` and `Task` controls which tools are available

### Native Function Calling Mode
- CrewAI added support for native function calling (via LiteLLM) as an alternative to text parsing
- When enabled, it uses the provider's native tool calling API instead of regex parsing
- This largely eliminates the name-matching problem

---

## 5. AutoGPT / BabyAGI / MetaGPT

### AutoGPT
- Uses **OpenAI function calling** as the primary mechanism (since v0.5+)
- Earlier versions used a JSON-based command format parsed from text output
- Commands had simple names like `web_search`, `write_file` — flat namespace
- Error recovery: re-prompts with error message
- No fuzzy matching

### MetaGPT
- Uses a **role-based action selection** pattern
- Each role has a fixed set of `Action` classes it can perform
- Action selection is done by the LLM outputting a structured response indicating which action to take
- Actions have simple class names, not compound keys
- Uses Pydantic models to constrain outputs
- Multi-stage by design: roles have a small action set (typically 2-5), so the selection problem is minimal

### BabyAGI
- Minimal tool selection — primarily uses a fixed pipeline (task creation → prioritization → execution)
- Not really applicable to the multi-tool selection problem

---

## 6. Microsoft Semantic Kernel

### Mechanism
Semantic Kernel uses **native function calling** through its connector layer (OpenAI, Azure OpenAI, etc.).

### Plugin/Function Naming
- Functions are organized into **Plugins** (like namespaces)
- Each function has a compound name: `PluginName-FunctionName` (hyphen-separated)
- When sent to the model, these are typically flattened: `PluginName_FunctionName` or `PluginName-FunctionName`
- The kernel maintains a registry mapping these compound names back to implementations

### Auto Function Calling
- `FunctionChoiceBehavior.Auto()` — lets the model choose from all registered functions
- `FunctionChoiceBehavior.Required()` — must call a function
- `FunctionChoiceBehavior.None()` — no function calling
- Can filter to specific functions: `FunctionChoiceBehavior.Auto(filters={"included_functions": ["Plugin-Func1"]})`

### Large Tool Sets — Function Filtering
- **Function Filters**: Pre-filter which functions are advertised to the model based on context
- **Auto-invocation with limits**: `maximum_auto_invoke_attempts` prevents infinite tool loops
- Semantic Kernel explicitly supports the pattern of advertising a large plugin set but filtering dynamically

### Error Recovery
- Built-in retry logic with `FunctionInvocationFilter`
- If the model calls a non-existent function, the kernel returns an error and the model retries
- No fuzzy matching — exact compound name required

---

## 7. DSPy

### Mechanism
DSPy takes a fundamentally different approach — it uses **typed signatures** and **structured output generation** rather than tool calling.

### Tool Selection via `dspy.ReAct`
- The `ReAct` module wraps tools and uses structured generation
- Tools are passed as a list, and DSPy generates the tool selection as a **typed field** (essentially an enum)
- The signature includes `action: Literal["tool1", "tool2", "tool3"]` — constraining the model to valid options

### Constrained Generation
- DSPy can use **guided generation** (via Outlines, SGLang, or structured output APIs) to constrain the model's output to valid values
- For tool selection, this means the model literally cannot emit an invalid tool name — it's grammar-constrained
- This is the most robust approach to the exact problem described

### Large Tool Sets
- DSPy's optimization (prompt tuning) can help with large tool sets by finding few-shot examples that help the model select correctly
- The `dspy.Suggest` and `dspy.Assert` primitives can add runtime constraints
- No built-in multi-stage selection, but the signature system makes it easy to build

---

## 8. Gorilla LLM

### Mechanism
Gorilla is specifically fine-tuned for API calling. It uses a **custom training approach**:

- Trained on large datasets of API documentation (TorchHub, TensorHub, HuggingFace, etc.)
- The model outputs structured API calls in a specific format
- Uses **retrieval-augmented generation (RAG)** — retrieves relevant API docs before generating the call

### Key Innovation: AST-based Validation
- Gorilla validates generated API calls by parsing them as ASTs (Abstract Syntax Trees)
- If the call doesn't parse correctly, it's rejected and regenerated
- This catches malformed function names, wrong signatures, etc.

### Berkeley Function Calling Leaderboard (BFCL)
- Gorilla project created the BFCL benchmark for evaluating function calling
- Tests models on: simple calls, multiple calls, parallel calls, relevance detection
- Measures both **name accuracy** and **argument accuracy** separately

### Large Tool Sets
- Gorilla's RAG approach is specifically designed for large API sets (thousands of APIs)
- First retrieves relevant APIs, then generates calls against the subset
- This is essentially multi-stage selection: retrieval → generation

---

## 9. ToolLLM / ToolBench

### Mechanism
ToolLLM (from Tsinghua/THU) is a framework for tool learning with LLMs:

### DFSDT (Depth-First Search Decision Tree)
- Instead of linear tool selection, ToolLLM uses a **tree-based search**
- The model explores multiple tool selection paths simultaneously
- If one path fails, it backtracks and tries another
- This is a fundamentally different approach — treating tool selection as search, not classification

### API Retriever
- For large tool sets (16,000+ APIs in ToolBench), uses a **trained retriever**
- The retriever is a separate model that selects 5-10 relevant APIs from the full set
- Only the selected APIs are presented to the LLM
- This is explicit multi-stage selection: retrieval → selection → execution

### RapidAPI Hub Integration
- ToolBench covers 16,000+ real-world APIs from RapidAPI
- APIs are organized into categories → tools → endpoints (3-level hierarchy)
- The retriever operates at the endpoint level but uses category information

### Error Recovery
- DFSDT naturally handles errors — failed paths are pruned from the search tree
- Multiple attempts at different tool selections are tried automatically

---

## 10. Instructor Library

### Mechanism
Instructor uses **Pydantic models** to constrain LLM outputs to valid structured data.

### Tool Selection as Enum
```python
class ToolChoice(str, Enum):
    SEARCH = "search_database"
    CALCULATE = "calculate_result"
    FETCH_URL = "fetch_url"

class AgentAction(BaseModel):
    tool: ToolChoice
    arguments: dict
```

- By using Pydantic enums, the tool name is constrained to valid values
- Instructor uses the model's native structured output / function calling to enforce this
- With OpenAI's strict mode or Anthropic's tool use, the enum values are guaranteed

### Retry Logic
- Instructor has built-in `max_retries` with **Tenacity** integration
- If validation fails (including wrong tool name), it re-prompts with the validation error
- `Instructor.from_openai(client, max_retries=3)` — automatic retry on validation failure

### Validation Hooks
- Pydantic validators can check tool names, arguments, etc.
- `@field_validator("tool")` can implement fuzzy matching, alias resolution, etc.
- This is where you could add custom logic to map `method_name` → `ClassName.hash.method_name`

### Large Tool Sets
- No built-in multi-stage selection
- But the Pydantic model approach makes it easy to dynamically generate enum types based on available tools

---

## Summary: Key Strategies

### Strategy 1: Native Function Calling (Most Common)
**Used by:** OpenAI, Anthropic, LangChain (modern), Semantic Kernel, CrewAI (optional)

- Model is fine-tuned to emit structured tool calls
- Tool names constrained to `[a-zA-Z0-9_-]{1,64}` — **no dots allowed**
- The model "sees" the exact tool list and is trained to emit exact matches
- **Limitation:** Degrades with 50+ tools; no dots/compound keys in names

### Strategy 2: Constrained/Guided Decoding (Most Robust)
**Used by:** DSPy, OpenAI strict mode, Instructor

- Grammar-based constraints during token generation ensure the output is literally a valid tool name
- Impossible to emit wrong name — the decoding is masked to valid tokens
- **Best solution for the compound key problem** — if you can constrain decoding to emit only valid compound keys
- Requires model/API support for constrained decoding

### Strategy 3: Retrieval-Based Selection (Best for Scale)
**Used by:** Gorilla, ToolLLM/ToolBench, LangChain (dynamic tools)

- Separate retrieval step selects relevant tools from a large set
- Only 5-20 tools presented to the model at a time
- **Best for 100+ tools** — decouples discovery from selection
- Two-stage: semantic search over tool descriptions → LLM selection from subset

### Strategy 4: Text Parsing with Error Recovery (Least Robust)
**Used by:** CrewAI (default), LangChain (legacy ReAct), AutoGPT (legacy)

- LLM outputs text, parsed with regex
- Most prone to the exact problem described (wrong name, partial name, etc.)
- Mitigated with: fuzzy matching, error messages listing valid tools, retries
- **Not recommended for compound keys**

### Strategy 5: Hierarchical/Multi-Stage Selection
**Used by:** ToolBench (categories → tools → endpoints), Semantic Kernel (plugins → functions)

- First select category/plugin, then select specific tool
- Reduces the selection space at each stage
- Natural fit for compound keys: first select `ClassName`, then `method_name`

---

## Recommendations for the Compound Key Problem

Given the specific problem of `ClassName.hash.method_name` keys:

### Option A: Flatten + Alias Table (Quick Fix)
- Convert compound keys to flat names: `ClassName_hash_method_name` or just `ClassName_method_name`
- Use native function calling where the name regex is enforced
- Maintain a mapping table from flat name → compound key
- Risk: name collisions if method_name repeats across classes

### Option B: Two-Stage Selection (Most Robust for Large Sets)
- Stage 1: LLM selects the `ClassName` (or category)
- Stage 2: LLM selects the `method_name` from that class's methods
- Dramatically reduces selection space and eliminates hash issues
- Natural fit for compound keys

### Option C: Constrained Decoding (Most Robust for Small-Medium Sets)
- Use structured output / enum constraints to limit the model to valid compound keys
- Works perfectly when the set is small enough (<50 tools)
- DSPy's approach or Instructor's Pydantic enum approach

### Option D: Fuzzy Matching + Retry (Pragmatic Fallback)
- Accept the model's output and fuzzy-match against valid keys
- If `method_name` is emitted, search for compound keys ending in `.method_name`
- If ambiguous, re-prompt with: "Multiple tools match 'method_name': [list]. Please specify the full key."
- Combine with retry logic (max 2-3 attempts)

### Option E: Semantic Retrieval Pre-filter
- Embed all tool descriptions
- Before each LLM call, retrieve the top-K most relevant tools
- Only advertise those to the LLM
- Eliminates the large-set problem entirely
