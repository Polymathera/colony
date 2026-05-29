# Consciousness Streams

A **consciousness stream** is a filtered, ordered record of one slice of an agent's experience -- the events it receives and the actions it takes -- rendered into the LLM planning prompt. An agent's action policy can maintain as many streams as it needs, and each stream decides independently what to capture and how to present it.

Streams are defined in `polymathera.colony.agents.patterns.planning.streams` and are consumed by `EventDrivenActionPolicy` and its subclasses (including `CodeGenerationActionPolicy`).

## Why Streams Instead of a Single Event History?

A conversational session agent's experience is fundamentally a chat transcript: user messages interleaved with the agent's own replies, rendered so the LLM can reason about and continue the conversation. An analysis coordinator's experience is not a transcript at all -- it is a collection of worker-result events, synthesis actions, game moves, each requiring its own presentation. A monitoring agent might stream telemetry events but never actions.

Baking any one of these shapes into the framework privileges it over the others. Chat agents would get a `ConversationFormatter`; analysis agents would get an `EventHistoryFormatter`; every new agent type would push on the framework until the core policy code accumulated agent-specific hacks. The same problem shows up at the write side: if the policy has a single `event_history`, then adding agent replies to it means the policy has to scan the action dispatcher's call trace for a specific action key (`respond_to_user`) -- which is exactly the kind of domain-specific leak that the framework should not contain.

Consciousness streams invert the relationship. The policy does not decide what to record or how to present it. It just feeds every event and every action call to all registered streams. Each stream is a small, composable object that answers three questions:

1. **Which events does this stream care about?** (event filter)
2. **Which action calls does this stream care about?** (action filter)
3. **How should the recorded entries be rendered into a prompt section?** (formatter)

## Anatomy of a Stream

A `ConsciousnessStream` is fully defined by its three pluggable pieces plus a rolling window:

```python
class ConsciousnessStream:
    def __init__(
        self,
        name: str,
        formatter: ConsciousnessStreamFormatter,
        event_filter:  Callable[[dict[str, Any]], bool] | None = None,
        action_filter: Callable[[dict[str, Any]], bool] | None = None,
        max_entries: int = 20,
    ):
        ...

    def consider_event(self, contexts: dict[str, Any]) -> None: ...
    def consider_action(self, call: dict[str, Any]) -> None: ...
    def consider_tool_output(self, payload: dict[str, Any]) -> None: ...
    def consider_vcm_update(self, payload: dict[str, Any]) -> None: ...
    def consider_monorepo_commit(self, payload: dict[str, Any]) -> None: ...
    def consider_domain_state(self, payload: dict[str, Any]) -> None: ...
    def render(self) -> str: ...
```

A stream holds one filter *per entry kind* (`event`, `action`, `tool_output`, `vcm_update`, `monorepo_commit`, `domain_state`); a kind with no filter is silently ignored, so a stream only records the kinds it opts into. The `event` and `action` kinds are fed directly by the policy (after each event-handler round and each dispatched action); the other four kinds are fed by **stream sources** (see [Stream Sources](#stream-sources) below) via `policy.record_stream_entry(kind, payload)`. The stream consults the matching per-kind filter to decide whether to append; old entries are dropped once `max_entries` is exceeded. At prompt-build time, the policy asks each stream to `render` itself and drops the resulting markdown section directly into the planning prompt.

### Filters

A filter is any picklable callable with a specific signature. Stock implementations:

| Filter | Signature | Purpose |
|--------|-----------|---------|
| `EventContextKeyFilter(*keys)` | `(contexts: dict) -> bool` | Accept events whose accumulated context contains any of the given `context_key` values. |
| `ActionKeySubstringFilter(*substrings)` | `(call: dict) -> bool` | Accept action calls whose `action_key` contains any of the given substrings. |
| `SuccessfulActionFilter(inner)` | `(call: dict) -> bool` | Wraps another action filter and additionally requires `call["success"]` to be truthy. |

Custom filters are ordinary callables -- classes, top-level functions, or lambdas, as long as they are picklable for transport through Ray. `AnyOf(...)` and `AllOf(...)` style composition is just stacked `and`/`or` over the underlying callables.

!!! tip "Design note: why classes, not closures"
    The stock filters are classes with `__init__` arguments rather than closures so that an entire stream blueprint can be serialized via cloudpickle and shipped across Ray boundaries without capturing surrounding scope. If you write a custom filter, prefer a top-level class or function for the same reason.

### Formatters

A `ConsciousnessStreamFormatter` is an abstract class whose `format(entries)` method renders the recorded entries into a markdown section. Each captured entry is a plain dict:

```python
# Event entry
{"kind": "event", "timestamp": ..., "contexts": {<context_key>: <context_dict>}}

# Action entry
{"kind": "action", "timestamp": ..., "call": {"action_key": ..., "output_preview": ..., "success": ..., ...}}
```

Two stock formatters ship with the framework:

- **`ConversationFormatter`** -- renders a chat thread. Event entries with a configured `user_context_key` become `**User**: <message>`; action entries become `**You (Agent)**: <output>`. Suitable for session agents.
- **`JSONStreamFormatter`** -- renders a flat bullet list with the event or action key and a truncated value. A reasonable default when no domain-specific formatter is needed.

Domain agents should subclass `ConsciousnessStreamFormatter` to render streams in whatever form makes sense for their task. Formatters are constructed via `ConsciousnessStreamFormatter.bind(**kwargs)`, which returns a `Blueprint` that travels through agent configuration and is resolved locally when the agent materializes.

## How Streams Wire Into the Planning Prompt

Streams live on the `EventDrivenActionPolicy`. The flow is:

1. **Configure**: The agent's `action_policy_blueprints` dict supplies a `consciousness_streams` entry -- a list of `ConsciousnessStream` blueprints.
2. **Resolve**: `Agent._initialize_action_policy` walks the blueprints, calls `local_instance()` on each, and passes the resulting list to `create_default_action_policy`.
3. **Capture**:
    - After event handlers run inside `EventDrivenActionPolicy.plan_step`, the policy calls `stream.consider_event(accumulated_context)` for every stream.
    - After code execution inside `CodeGenerationActionPolicy.execute_iteration`, the policy iterates its `_run_call_trace` and calls `stream.consider_action(call)` for every stream.
4. **Render**: `PlanningContextBuilder.get_planning_context` calls `stream.render()` on each stream and stores the resulting markdown sections in `PlanningContext.stream_sections`.
5. **Format**: `format_planning_context_for_codegen` inserts every section into the prompt verbatim, between the goals/constraints block and the available-actions block.

No part of the policy or the prompt formatter knows about chat threads, worker results, or any other domain-specific concept. That knowledge lives entirely in the stream objects supplied by the agent.

```mermaid
flowchart LR
    EV[Blackboard Event] --> H[Event Handlers]
    H -->|accumulated_context| P[EventDrivenActionPolicy]
    A[Dispatched Action] --> P
    P -->|consider_event| S1[Stream: conversation]
    P -->|consider_event| S2[Stream: worker_results]
    P -->|consider_action| S1
    P -->|consider_action| S2
    S1 -->|render| PC[PlanningContext.stream_sections]
    S2 -->|render| PC
    PC --> PR[Planning Prompt]
```

## Stream Sources

The `event` and `action` kinds are fed by the policy itself, but the richer kinds (`tool_output`, `vcm_update`, `monorepo_commit`, `domain_state`) come from **stream sources**. A source is any object implementing `StreamEventSource` (in `polymathera.colony.agents.patterns.planning.sources`):

```python
class StreamEventSource(ABC):
    async def attach(self, policy: "BaseActionPolicy") -> None: ...
    async def detach(self, policy: "BaseActionPolicy") -> None: ...
```

`attach(policy)` arranges for the source to call `policy.record_stream_entry(kind, payload)` whenever it has something to feed; `record_stream_entry` fans the payload to every mounted stream's `consider_<kind>` method. The policy keeps a list of attached sources and invokes each source's `attach` from `attach_pending_sources()` (called during `initialize`, and re-callable when an agent registers more sources afterward). The agent never has to know which source feeds which kind — it attaches sources and mounts streams independently.

Sources fall into two families:

### In-process sources (direct feed)

These observe facts that are already local to the agent's own process and feed them synchronously:

| Source | Feeds kind | What it observes |
|--------|-----------|------------------|
| `AccumulatedContextSource` | `event` | The policy's existing event-handler accumulated context (sentinel — no new hook). |
| `ActionCallSource` | `action` | The policy's existing dispatched-action feed (sentinel — no new hook). |
| `ToolResultSource` | `tool_output` | Installs a post-dispatch hook; when an action returns a typed `ToolResult`-shaped value, builds a `tool_output` payload. |

`attach_colony_standard_sources(policy)` wires these three in one call; `colony_basic_stream()` returns a catch-all stream that accepts every kind, so the pair is a one-line starting point for any Colony agent.

### Cross-process sources (colony blackboard)

Some experience originates in *other* processes — a VCM replica reconciling a page-graph mutation, a peer agent committing to the design monorepo on a shared branch. A process-local listener cannot see those events (it lives in a different Ray actor). So cross-process sources ride the same blackboard-protocol idiom every other cross-process event in the colony uses:

1. **Producers `await blackboard.write(key, value)`** to a **colony-scoped** `BlackboardProtocol`. `VirtualContextManager._publish_page_event` writes `VCMPageEventProtocol`; `BranchScopedCapabilityBase.fire_post_commit` writes `MonorepoCommitProtocol`. The blackboard's Redis-backed pub/sub fans the write to every subscribed agent regardless of process or replica.
2. **Consumers are `ColonyScopedEventSource` subclasses** — both an `AgentCapability` (so the agent's event-dispatch loop discovers their `@event_handler` method) and a `StreamEventSource` (so they slot into `attach_source`). Their `attach` binds the agent, registers the capability with `add_capability(..., events_only=True)`, and overrides `stream_events_to_queue` to subscribe the protocol's `event_pattern()` on the **colony** scope (not the agent's own scope). The `@event_handler` method translates each blackboard write into a `record_stream_entry(kind, payload)` call.

| Source | Feeds kind | Subscribes to |
|--------|-----------|---------------|
| `VCMPageEventSource` | `vcm_update` | `VCMPageEventProtocol` (colony scope) |
| `MonorepoCommitEventSource` | `monorepo_commit` | `MonorepoCommitProtocol` (colony scope) |

```mermaid
flowchart LR
    subgraph proc1[VCM replica / committing agent process]
        PRD[Producer] -->|blackboard.write| BB[(Colony-scoped<br/>BlackboardProtocol)]
    end
    subgraph proc2[Subscribing agent process]
        BB -->|@event_handler| SRC[ColonyScopedEventSource]
        SRC -->|record_stream_entry| POL[ActionPolicy]
        POL -->|consider_*| STR[Stream]
    end
```

`ColonyScopedEventSource` is a **public extension point** (exported from the `sources` module). Downstream packages subclass it to surface their own cross-process events — e.g. CPS's `BudgetStateEventSource` feeds `domain_state` from budget-tree transitions published under CPS's `BudgetStateProtocol`. To add a new cross-process kind:

1. Define a colony-scoped `BlackboardProtocol` subclass with an `event_key(...)` / `event_pattern()` pair.
2. Make the producer `await blackboard.write(...)` after its state change.
3. Subclass `ColonyScopedEventSource`, set `_PATTERN = MyProtocol.event_pattern()`, and decorate one `@event_handler(pattern=MyProtocol.event_pattern())` method that calls `self._policy.record_stream_entry("<kind>", payload)`.

The colony blackboard handle a source subscribes on is resolved through the inherited `AgentCapability._get_colony_blackboard()`, a small helper on the capability base that calls `get_blackboard(scope_id=ScopeUtils.get_colony_level_scope())` once and buffers the result on the instance. (`get_blackboard` builds a fresh `EnhancedBlackboard` per call — it is not pooled downstream — so the per-instance buffer is what avoids rebuilding on every publish/subscribe.) The same helper backs `BranchScopedCapabilityBase.fire_post_commit` and CPS's budget-state publishing, so all colony-scoped pub/sub shares one resolution path.

## Example 1 -- Session Agent (Conversation Stream)

A session agent's entire experience is the user chat thread plus its own replies. One stream suffices: capture the `user_chat_message` event (emitted by `SessionOrchestratorCapability.handle_user_message`) and successful calls to the `respond_to_user` action, render both as a `**User** / **You (Agent)**` transcript.

```python
from polymathera.colony.agents.patterns.planning.streams import (
    ConsciousnessStream,
    ConversationFormatter,
    EventContextKeyFilter,
    ActionKeySubstringFilter,
    SuccessfulActionFilter,
)

bp = SessionAgent.bind(
    metadata=agent_metadata,
    capability_blueprints=[...],
    action_policy_blueprints={
        "consciousness_streams": [
            ConsciousnessStream.bind(
                name="conversation",
                formatter=ConversationFormatter.bind(),
                event_filter=EventContextKeyFilter("user_chat_message"),
                action_filter=SuccessfulActionFilter(
                    ActionKeySubstringFilter("respond_to_user")
                ),
            ),
        ],
    },
)
```

The resulting planning prompt contains a section like:

```markdown
## Conversation

**User**: Can you run an impact analysis on the auth module?
**You (Agent)**: I'll spawn an ImpactAnalysisCoordinator for the auth module...
**User**: Focus on session token handling specifically.
```

The session agent's own replies are captured automatically by the same stream, because `respond_to_user` is a dispatched action and the stream's action filter accepts it. The policy never has to scan its own call trace for a specific action key.

## Example 2 -- Analysis Coordinator (Two Streams)

An analysis coordinator watches worker result events and also performs synthesis actions. It wants the LLM planner to see worker results as a compact list and synthesis actions as a summarized history, cleanly separated in the prompt. Two streams:

```python
from polymathera.colony.agents.patterns.planning.streams import (
    ConsciousnessStream,
    JSONStreamFormatter,
    EventContextKeyFilter,
    ActionKeySubstringFilter,
    SuccessfulActionFilter,
)

streams = [
    ConsciousnessStream.bind(
        name="worker_results",
        formatter=JSONStreamFormatter.bind(section_title="## Worker Results"),
        event_filter=EventContextKeyFilter("worker_result", "worker_failed"),
        action_filter=None,  # no actions in this stream
        max_entries=50,
    ),
    ConsciousnessStream.bind(
        name="synthesis",
        formatter=JSONStreamFormatter.bind(section_title="## Synthesis Progress"),
        event_filter=None,  # no events in this stream
        action_filter=SuccessfulActionFilter(
            ActionKeySubstringFilter("synthesize", "finalize")
        ),
        max_entries=20,
    ),
]

bp = AnalysisCoordinator.bind(
    metadata=coordinator_metadata,
    capability_blueprints=[...],
    action_policy_blueprints={"consciousness_streams": streams},
)
```

The prompt now contains two independent sections in the order the streams were declared -- `## Worker Results` filled by events, `## Synthesis Progress` filled by successful synthesis actions -- without any domain-specific code in the policy.

## Example 3 -- Custom Formatter (Game State Transitions)

A game-playing agent wants to render each recorded move as a state transition with the move number, the move itself, and its evaluation. Subclass `ConsciousnessStreamFormatter`:

```python
from polymathera.colony.agents.patterns.planning.streams import (
    ConsciousnessStream,
    ConsciousnessStreamFormatter,
    EventContextKeyFilter,
)

class GameMoveFormatter(ConsciousnessStreamFormatter):
    def __init__(self, section_title: str = "## Game Moves"):
        self._section_title = section_title

    def format(self, entries):
        if not entries:
            return ""
        lines = [self._section_title, ""]
        for i, entry in enumerate(entries, start=1):
            if entry["kind"] != "event":
                continue
            ctx = entry["contexts"].get("game_move", {})
            move = ctx.get("move", "?")
            eval_ = ctx.get("evaluation", "?")
            lines.append(f"{i}. {move} (eval: {eval_})")
        return "\n".join(lines)

stream = ConsciousnessStream.bind(
    name="game_moves",
    formatter=GameMoveFormatter.bind(),
    event_filter=EventContextKeyFilter("game_move"),
    max_entries=30,
)
```

Any agent can ship its own formatters alongside its capabilities. The framework treats them as opaque blueprints; only the agent knows what a "game move" means.

## Serialization and Transport

Streams are configured via `Blueprint` objects because agent configuration crosses Ray actor boundaries. The rules:

- **`ConsciousnessStream.bind(**kwargs)`** -- returns a `Blueprint[ConsciousnessStream]`. Kwargs are validated via cloudpickle at bind time.
- **`ConsciousnessStreamFormatter.bind(**kwargs)`** -- same, for formatters. A formatter blueprint passed as the `formatter` kwarg of a stream is resolved by `ConsciousnessStream.__init__` automatically.
- **Filters** are plain callables, not blueprints. They must be picklable (top-level classes or functions).

On the remote node, `Agent._initialize_action_policy` resolves each blueprint in `action_policy_blueprints` -- including every element of list-valued entries like `consciousness_streams` -- before handing the fully materialized list to the policy constructor.

!!! info "Why separate from `action_policy_config`"
    `action_policy_config` lives on `AgentMetadata` and is JSON-serialized into Redis for durability. Blueprints are cloudpickle-only. Keeping them in a separate `action_policy_blueprints` field (with `exclude=True` on the Pydantic model) avoids JSON-serialization errors while letting blueprints still travel through `AgentBlueprint` via cloudpickle.

## Compaction & Spillover

By default a stream keeps a rolling window of `max_entries` and silently drops the oldest — fine for short-lived agents, lossy for long-lived ones. Set `compaction_budget_tokens` on a stream's `bind()` to switch it into **compaction mode**, where the stream is treated as an *infinite, linear* history and the prompt renders a bounded *view* over it:

```python
ConsciousnessStream.bind(
    name="design_reasoning",
    formatter=EventLogFormatter.bind(section_title="## Design reasoning"),
    filters={...},
    compaction_budget_tokens=4000,   # enable; keep the rendered view under ~4k tokens
    compaction_keep_recent=12,       # never auto-compact the 12 most-recent raw entries
)
```

### The model

- **Durable log = source of truth.** Every recorded entry is appended, in order, to a per-`(agent, stream)` durable log (`StreamLogStore`; default `BlackboardStreamLogStore` — a non-evicting, events-off blackboard scope). Entries carry a monotonic `seq` and are *never* dropped from the log. This is the "infinite linear" history; it survives suspend/resume and restart.
- **The view is a projection.** The in-memory `_entries` becomes the *hot view*: recent raw entries plus `compaction_summary` stand-ins for older spans. `render()` stays synchronous over this view.
- **Compaction is reversible.** Compacting a span `[start_seq, end_seq]` records a `CompactionDescriptor` (the LLM-produced summary + the span it covers) in the log index and replaces those raw entries in the view with one synthesized `compaction_summary`. The originals stay in the log, so `expand_span(start, end)` brings them back verbatim — lossy in the view, lossless in the log. (Implementation note: descriptors live in the index, *not* in the raw seq space, so a late-created summary covering an old span still sorts correctly — by its span's `start_seq` — even after an arbitrary expand.)
- **Spillover = the same log.** "Spilling" an entry just means it left the view; it remains in the log, range-addressable via `read_span`. Reversible compaction *is* spillover with a summary stand-in.

### Triggers

- **Automatic safety-net.** After every iteration, `BaseActionPolicy.execute_iteration` flushes new entries to the log and runs `stream.maintain()`: while the rendered view exceeds `compaction_budget_tokens`, the `CompactionPolicy` (default `KeepRecentCompactionPolicy`) selects the oldest span beyond `compaction_keep_recent` and the `StreamCompactor` (default `LLMStreamCompactor`, via `agent.infer`) condenses it. Token counting reuses the cluster's `TokenizerProtocol` (`TiktokenTokenizer`). Bounded prompts without any agent action.
- **Agent-driven.** `StreamMaintenanceCapability` exposes planner-facing actions: `compact_stream`, `expand_stream_span` (optionally `reattach_to_context=True` to page the original span back into the *real* LLM context window via the VCM), and `list_stream_history`. It is **auto-mounted** by `Agent._create_action_policy` whenever the agent has ≥1 compaction-enabled stream (idempotent, same pattern as the `REPLCapability` / `KnowledgeRetrievalCapability` auto-installs) — so enabling `compaction_budget_tokens` is the only operator action needed; the capability is *not* added to agents without compacted streams, keeping their action surface clean.

### Swap points (every alternative is an ABC)

| ABC | Default | Alternative |
|-----|---------|-------------|
| `StreamLogStore` | `BlackboardStreamLogStore` (non-evicting blackboard scope) | Redis-Streams / SQLite / WAL backing |
| `CompactionPolicy` | `KeepRecentCompactionPolicy` (oldest beyond a recent window) | relevance-ranked / time-based |
| `StreamCompactor` | `LLMStreamCompactor` (`agent.infer`) | `ExtractiveStreamCompactor` (no-LLM digest) — the "keep only the most relevant" arm vs. the "summarize" arm |
| `SpillArchive` | `VcmSpillArchive` (mmap span + page-fault) / `NoopSpillArchive` | direct-S3, etc. |
| token estimator | reused `TokenizerProtocol` / `TiktokenTokenizer` | `HuggingFaceTokenizer` (model-exact) |

The policy builds the default collaborators in `_init_stream_logs()` (from the live agent) and injects them via `stream.bind_log(...)`; swapping an implementation is a change there, not in the stream. Compaction config (`compaction_budget_tokens`, `compaction_keep_recent`) travels in the serializable `bind()` blueprint; the runtime collaborators do not (they need the live agent).

Streams without `compaction_budget_tokens` are entirely unaffected — the legacy rolling window is unchanged.

## Design Principles

1. **No domain knowledge in the policy.** `EventDrivenActionPolicy` and `CodeGenerationActionPolicy` do not know what a chat message, a worker result, or a game move is. They only feed events and actions to whatever streams are attached.
2. **Each stream owns its presentation.** The formatter is part of the stream, not a framework-level concept. Two agents that both consume `user_chat_message` events can render them completely differently.
3. **Filters decide membership; formatters decide shape.** These two concerns are independent -- the same `ConversationFormatter` can consume different filters on different agents; the same `EventContextKeyFilter` can feed different formatters.
4. **Streams are declarative.** An agent configures its streams at `bind()` time. The prompt shape is a consequence of the declared streams, not of imperative code sprinkled through the policy.
5. **Add a stream, never patch the policy.** When a new agent type needs a new view of its experience, the answer is always a new stream (or a new formatter), never a new branch in the policy or the prompt formatter.

## Further Reading

- Streams module: `polymathera.colony.agents.patterns.planning.streams`
- Sources module: `polymathera.colony.agents.patterns.planning.sources` (`StreamEventSource`, `ColonyScopedEventSource`, `VCMPageEventSource`, `MonorepoCommitEventSource`)
- Cross-process protocols: `polymathera.colony.agents.blackboard.protocol` (`VCMPageEventProtocol`, `MonorepoCommitProtocol`)
- Compaction/spillover: `polymathera.colony.agents.patterns.planning.stream_log` (`StreamLogStore`, `StreamLogIndex`, `CompactionDescriptor`) + `…planning.compaction` (`CompactionPolicy`, `StreamCompactor`, `SpillArchive`, token estimator) + `…capabilities.stream_maintenance.StreamMaintenanceCapability`
- Used by: `polymathera.colony.agents.patterns.actions.policies.EventDrivenActionPolicy`, `polymathera.colony.agents.patterns.actions.code_generation.CodeGenerationActionPolicy`
- Rendered by: `polymathera.colony.agents.patterns.planning.context.PlanningContextBuilder`
- Prompt integration: `polymathera.colony.agents.patterns.actions.code_generation.format_planning_context_for_codegen`
