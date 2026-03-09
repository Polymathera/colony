# Plan: Colony Observability & Tracing System (LangSmith-Inspired)

## Context

Colony is a multi-agent framework where agents execute via `_run_agent_loop` → `run_step` → `execute_iteration` → `plan_step` → `dispatch`. The system has rich infrastructure (VCM, blackboard, sessions, capabilities, hook system) but **no structured observability**. Current debugging relies on `logger.warning()` ASCII boxes with no correlation IDs, no trace hierarchy, no real-time streaming to the dashboard.

**Goal**: Build a LangSmith-inspired but Colony-native observability system that captures fine-grained execution traces (spans) from agents, streams them through Kafka, stores them durably in PostgreSQL, and renders them in a real-time trace waterfall view in the dashboard. The system must be:
- **Zero-change to agent code** — uses the existing `@hookable` + `AgentCapability` hook system
- **Optional** — enabled via config, disabled by default for zero overhead
- **Colony-specific** — captures VCM page operations, blackboard interactions, cache-aware planning, capability compositions
- **Open-source friendly** — clean API, easy to extend, well-documented

---

## Part 1: Data Model

### 1.1 Trace Hierarchy

The trace boundary is the **session**, not the run. An agent's lifetime can span multiple runs within the same session, and data flows across runs via blackboard and VCM. A session-level trace captures all of this.

```
Session = Trace (trace_id = session_id)
  ├── Run Span (one per AgentRun — groups work done for one user request)
  │     ├── Agent Step Span (one iteration of run_step)
  │     │     ├── Plan Span (execute_iteration → plan_step)
  │     │     │     └── Infer Span (LLM call with VCM pages)
  │     │     ├── Action Span (dispatch:analyze_code)
  │     │     │     ├── Infer Span
  │     │     │     └── Blackboard Write Span
  │     │     └── Page Request Span (VCM page load)
  │     └── Agent Step Span #2 ...
  ├── Run Span (next user request in same session)
  │     └── ... (can reference data written by previous run)
  └── Child Agent Trace Link (child agent in same session)
```

This subsumes per-run tracing: each run is a top-level span within the session trace. Cross-run correlations (agent reading blackboard data from a previous run) are visible within the same trace tree.

<mark>A **Trace** is the tree of all spans within one `Session`. The `trace_id` = `session_id` (reuse existing session identity). A **Span** is one unit of work: an LLM call, action dispatch, planning step, VCM operation, etc.</mark>

The `TracingCapability` lives on the agent (which persists across runs within a session), so trace context is naturally maintained at the session level. When a new run arrives via `get_next_event`, the capability creates a new `RUN` span under the session trace rather than starting a fresh trace.

### 1.2 Span Model

**New file**: `colony/python/colony/agents/observability/models.py`

<mark>`SpanKind` enum may appear restrictive. But `SpanKind.CUSTOM` allows user-defined spans with the other enums representing built-in span types.</mark>

```python
class SpanKind(str, Enum):
    """Colony-specific span types (controlled vocabulary)."""
    RUN           = "run"            # One AgentRun (user request)
    AGENT_STEP    = "agent_step"     # One iteration of run_step
    PLAN          = "plan"           # Planning via LLM (plan_step)
    ACTION        = "action"         # Action dispatch
    INFER         = "infer"          # LLM inference call
    PAGE_REQUEST  = "page_request"   # VCM page load
    BLACKBOARD_OP = "blackboard_op"  # Blackboard read/write
    CHILD_SPAWN   = "child_spawn"    # Spawning child agent
    EVENT_PROCESS = "event_process"  # Processing blackboard event
    CAPABILITY    = "capability"     # Capability method execution
    CUSTOM        = "custom"         # User-defined spans

class SpanStatus(str, Enum):
    RUNNING = "running"
    OK      = "ok"
    ERROR   = "error"

class Span(BaseModel):
    span_id: str          # "span_{uuid8}"
    trace_id: str         # = session_id (the trace boundary)
    parent_span_id: str | None  # For nesting
    run_id: str | None          # Which AgentRun this span belongs to (for filtering)

    # Identity
    agent_id: str
    name: str             # Human-readable: "plan_step", "dispatch:analyze_code", etc.
    kind: SpanKind

    # Timing
    start_time: float     # time.monotonic() for duration calc
    end_time: float | None
    start_wall: float     # time.time() for wall clock display

    # Status
    status: SpanStatus    # RUNNING, OK, ERROR
    error: str | None

    # Data (JSON-serializable, truncated for large payloads)
    input_summary: dict[str, Any]   # Summarized inputs (not full prompt text)
    output_summary: dict[str, Any]  # Summarized outputs

    # LLM-specific (only for kind=INFER)
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    model_name: str | None
    context_page_ids: list[str] | None  # VCM pages in KV context

    # Metadata
    tags: list[str]
    metadata: dict[str, Any]  # Extensible: capability_name, action_id, page_id, etc.
```

### 1.3 `TracingConfig`

**New file**: `colony/python/colony/agents/observability/config.py`

```python
@dataclass
class TracingConfig:
    enabled: bool = False              # Master switch
    kafka_bootstrap: str = "kafka:9092"  # Kafka broker address
    kafka_topic: str = "colony.spans"    # Span topic
    sample_rate: float = 1.0           # 0.0-1.0 sampling
    max_input_chars: int = 500         # Truncate large inputs in span data
    max_output_chars: int = 500        # Truncate large outputs
    flush_interval: float = 0.5        # Background flush interval (seconds)
    flush_batch_size: int = 50         # Max spans per Kafka produce batch
    capture_infer_inputs: bool = False # Include full prompt text (expensive)
    capture_action_results: bool = True
```

Wired into `AgentSystemConfig` as `tracing: TracingConfig = field(default_factory=TracingConfig)`.

---

## Part 2: Storage Backend Analysis

### Why Kafka

Colony is designed for 1,000s to 100,000s of agents. At that scale, the span ingestion pipeline needs:

1. **High-throughput ordered append** — agents produce spans continuously; the pipeline must keep up without backpressure affecting agent execution
2. **Durable log** — spans are the debugging record; losing them on a process restart defeats the purpose
3. **Fan-out to multiple consumers** — the dashboard SSE stream, the PostgreSQL sink, and future consumers (OTLP exporter, analytics) all need the same span data independently
4. **Horizontal scalability** — partitioned by `trace_id` (= `session_id`) so throughput scales linearly

Kafka (KRaft mode, no Zookeeper) is purpose-built for exactly this. It's the same infrastructure LangSmith uses internally for trace ingestion.

**"But isn't Kafka heavy?"** — No. In KRaft mode it's a single container:

```yaml
# Added to colony-env docker-compose.yaml
kafka:
  image: bitnami/kafka:3.7
  environment:
    KAFKA_CFG_NODE_ID: 0
    KAFKA_CFG_PROCESS_ROLES: controller,broker
    KAFKA_CFG_CONTROLLER_QUORUM_VOTERS: 0@kafka:9093
    KAFKA_CFG_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
    KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
    KAFKA_CFG_CONTROLLER_LISTENER_NAMES: CONTROLLER
    KAFKA_CFG_AUTO_CREATE_TOPICS_ENABLE: "true"
    KAFKA_CFG_LOG_RETENTION_HOURS: 24
    KAFKA_CFG_LOG_RETENTION_BYTES: 1073741824  # 1GB
  volumes:
    - kafka-data:/bitnami/kafka
```

~300MB RAM, single process, auto-creates topics. Users run `colony-env up` and Kafka starts alongside Redis and PostgreSQL — completely transparent.

### Options Evaluated

| Option | Real-time streaming | Structured queries | Durability | Scalability | In stack? | Complexity |
|--------|-------------------|--------------------|-----------|------------|----------|-----------|
| **Apache Kafka** | Excellent (consumer groups, partitions) | None (need separate query layer) | Excellent (replicated log) | Excellent (horizontal partitioning) | Added to docker-compose | Low (KRaft, single container) |
| **Redis Streams** | Excellent (XREAD blocking) | Poor (no SQL, no indexes) | Configurable (AOF/RDB; ElastiCache on AWS) | Good (ElastiCache clustered on AWS) | Yes | Low |
| **PostgreSQL** | Poor (LISTEN/NOTIFY 8KB limit, polling) | Excellent (SQL, B-tree/GIN indexes) | Excellent (WAL, ACID) | Good (proven at scale) | Yes | Low |
| **Kafka + PostgreSQL** | Excellent | Excellent | Excellent | Excellent | Kafka added to compose | Medium |
| **Redis Streams + PostgreSQL** | Excellent | Excellent | Excellent | Good | Yes | Medium |
| **OpenTelemetry → OTLP backend** | Depends on backend | Depends | Depends | Excellent | No | Medium (new dep + backend) |
| **ClickHouse** | Good (Kafka engine) | Excellent (OLAP, columnar) | Excellent | Excellent | No | High (new dep) |
| **TimescaleDB** | Moderate (via PG LISTEN) | Excellent (SQL + time-series hypertables) | Excellent | Good | Could use existing PG | Medium |
| **Ray Object Store** | Poor (no streaming primitive) | Poor | Volatile | Good (distributed) | Yes | Low |
| **Append-only JSONL files** | Poor (tail -f) | Poor (grep) | Good (filesystem) | Poor | Trivial | Trivial |
| **SQLite (embedded)** | Poor | Moderate (SQL, single-writer) | Good (file) | Poor (single process) | Trivial | Low |
| **Jaeger/Zipkin** | Good (via OTLP) | Moderate (trace-oriented queries) | Good | Good | No | Medium |
| **Loki (Grafana)** | Good (LogQL tailing) | Moderate (label-based) | Good | Good | No | Medium |

### Why Kafka wins

- **vs Redis Streams**: Redis Streams provides similar streaming semantics (ordered append, consumer groups, blocking reads). On AWS with ElastiCache, durability and scalability are not issues. However, Kafka provides native partitioned fan-out to independent consumers (PG sink, SSE, future OTLP exporter) without consumers competing for the same Redis resources as blackboard events and state management. At Colony's target scale (100K+ agents), separating the span pipeline from the operational Redis workload is the right architectural boundary.
- **vs PostgreSQL only**: No real-time streaming primitive. LISTEN/NOTIFY has an 8KB payload limit and no ordering guarantees at scale. Would require polling for the dashboard waterfall.
- **vs ClickHouse**: Excellent for OLAP on spans but adds a heavy dependency when PostgreSQL handles the query patterns we need (filter by trace/run/kind/agent, aggregate tokens).
- **vs OpenTelemetry/OTLP**: Protocol standard, not a storage backend. Best used as a future *exporter* — another Kafka consumer that converts Colony spans to OTel spans.
- **vs Jaeger/Zipkin**: We're building a Colony-native trace UI. Their storage backends (Cassandra/ES) are heavier than PG.

### Architecture: Kafka + PostgreSQL

```
TracingCapability → Kafka topic: "colony.spans"
                        │
                        ├──→ Consumer 1: Dashboard SSE (real-time waterfall)
                        ├──→ Consumer 2: PostgreSQL sink (durable storage + queries)
                        └──→ Consumer 3: (future) OTLP exporter
```

**Hot path** (real-time): Kafka for ingestion and dashboard streaming
- `TracingCapability` produces span records to `colony.spans` topic (partitioned by `trace_id`)
- Dashboard backend consumes from Kafka for real-time SSE to the frontend
- Ordering guaranteed within a partition (= within a session/trace)

**Cold path** (durable queries): PostgreSQL for historical storage and queries
- A Kafka consumer process sinks spans into the `spans` table
- Enables: `SELECT * FROM spans WHERE kind='infer' AND input_tokens > 1000 ORDER BY start_wall`
- The consumer runs as a background task in the dashboard backend (or as a separate service)

**Future extensibility**:
- Add OTLP exporter as another Kafka consumer — pipe to Jaeger/Tempo/Datadog with zero changes to agents
- Add ClickHouse consumer for OLAP analytics on span data
- Each consumer is independent — adding one doesn't affect others

### Kafka Topics

| Topic | Partition Key | Retention | Purpose |
|-------|---------------|-----------|---------|
| `colony.spans` | `trace_id` (= `session_id`) | 24h / 1GB | Span events from all agents |
| `colony.traces` | `trace_id` | 24h / 100MB | Trace lifecycle events (started, completed, errored) |

### PostgreSQL Schema

```sql
CREATE TABLE spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,           -- = session_id
    parent_span_id TEXT,
    run_id TEXT,
    agent_id TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_wall TIMESTAMPTZ NOT NULL,
    end_wall TIMESTAMPTZ,
    duration_ms DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT,
    input_summary JSONB,
    output_summary JSONB,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    model_name TEXT,
    context_page_ids TEXT[],
    tags TEXT[],
    metadata JSONB
);

CREATE INDEX idx_spans_trace ON spans (trace_id, start_wall);
CREATE INDEX idx_spans_run ON spans (run_id);
CREATE INDEX idx_spans_agent ON spans (agent_id);
CREATE INDEX idx_spans_kind ON spans (kind);
```

---

## Part 3: Instrumentation — `TracingCapability`

### 3.1 Design: Out-of-Line via Hook System

**Key insight**: Colony already has `@hookable` decorators on every critical method and a capability system that auto-registers hooks. We create a `TracingCapability` that registers AROUND hooks on all hookable join points. Zero changes to agent code.

**Reference patterns**:
- `MemoryCapability` (`colony/agents/patterns/memory/capability.py`) — demonstrates how an `AgentCapability` registers AFTER hooks, manages background tasks, initializes storage
- `@register_hook` decorator (`colony/agents/patterns/hooks/decorator.py`) — AROUND hook `proceed` callback pattern
- `session_id_context` (`colony/agents/sessions/context.py`) — contextvars pattern for context propagation

**New file**: `colony/python/colony/agents/observability/capability.py`

`TracingCapability(AgentCapability)`:
- On `initialize()`: registers AROUND hooks on all hookable methods via Pointcut patterns
- Sets `trace_id = session_id` from the agent's session context
- Each AROUND hook: creates a Span, sets it as current via contextvars, calls `proceed()`, ends the span
- Background `asyncio.Task` flushes buffered spans to Kafka

### 3.2 Context Propagation

**New file**: `colony/python/colony/agents/observability/context.py`

Follows the exact pattern of `colony/agents/sessions/context.py`:

```python
_current_span: ContextVar[Span | None] = ContextVar("current_span", default=None)
_current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)

def get_current_span() -> Span | None: ...
def set_current_span(span: Span) -> Token: ...

@contextmanager
def span_context(span: Span) -> Iterator[Span]:
    """Set span as current, restore previous on exit."""
    token = _current_span.set(span)
    try:
        yield span
    finally:
        _current_span.reset(token)
```

### 3.3 Hook Registration

The `TracingCapability` registers these AROUND hooks (priority=-100, lowest = outermost wrapper):

| Pointcut | SpanKind | What it captures |
|----------|----------|-----------------|
| `Pointcut.pattern("*.run_step")` | `AGENT_STEP` | One agent loop iteration |
| `Pointcut.pattern("*.execute_iteration")` | `PLAN` | Planning + dispatch cycle |
| `Pointcut.pattern("*.plan_step")` | `PLAN` | LLM-based planning |
| `Pointcut.pattern("*.dispatch")` | `ACTION` | Action execution |
| `Pointcut.pattern("*.infer")` | `INFER` | LLM inference (captures tokens, page IDs) |
| `Pointcut.pattern("*.request_page")` | `PAGE_REQUEST` | VCM page load |
| `Pointcut.pattern("*.get_next_event")` | `EVENT_PROCESS` | Blackboard event dequeue |

Each AROUND hook follows the same pattern:
```python
async def _trace_around(self, ctx: HookContext, proceed: Callable) -> Any:
    parent = get_current_span()
    span = Span(
        span_id=generate_span_id(),
        trace_id=self._trace_id,  # = session_id, set once on init
        parent_span_id=parent.span_id if parent else None,
        run_id=self._current_run_id,
        agent_id=self.agent.agent_id,
        name=ctx.join_point,
        kind=self._resolve_kind(ctx.join_point),
        start_time=time.monotonic(),
        start_wall=time.time(),
        status=SpanStatus.RUNNING,
        input_summary=self._summarize_input(ctx),
    )
    with span_context(span):
        try:
            result = await proceed()
            span.status = SpanStatus.OK
            span.output_summary = self._summarize_output(ctx.join_point, result)
            if span.kind == SpanKind.INFER and hasattr(result, 'usage'):
                span.input_tokens = result.usage.input_tokens
                span.output_tokens = result.usage.output_tokens
            return result
        except Exception as e:
            span.status = SpanStatus.ERROR
            span.error = f"{type(e).__name__}: {str(e)[:200]}"
            raise
        finally:
            span.end_time = time.monotonic()
            self._buffer.append(span)
```

### 3.4 Run Boundary Detection

When `get_next_event` delivers a new run request, the AROUND hook detects the `run_id` change:
```python
# In the EVENT_PROCESS hook:
event = result  # the blackboard event
if event and hasattr(event, 'metadata') and 'run_id' in event.metadata:
    self._current_run_id = event.metadata['run_id']
    # Create a RUN span as child of the session trace root
    run_span = Span(kind=SpanKind.RUN, trace_id=self._trace_id, run_id=self._current_run_id, ...)
    self._run_span_stack.append(run_span)
```

### 3.5 Cross-Agent Trace Propagation

When a parent spawns a child, `trace_id` (= `session_id`) + `parent_span_id` propagate via agent metadata:
1. `CHILD_SPAWN` action's AROUND hook captures current trace context
2. Injects `_trace_id` and `_parent_span_id` into the child agent's metadata dict
3. Child's `TracingCapability` reads these on `initialize()` and sets them as initial trace context

No modification to spawn code needed — the hook intercepts the dispatch and enriches the arguments.

### 3.6 Activation

In `AgentManagerBase.start_agent()`, after `agent.initialize()`, check config:
```python
if self._tracing_config and self._tracing_config.enabled:
    from colony.agents.observability.capability import TracingCapability
    tracing_cap = TracingCapability(agent=agent, config=self._tracing_config)
    await tracing_cap.initialize()  # Creates SpanProducer internally
    agent.add_capability(tracing_cap)
```

This keeps `TracingCapability` fully external — zero changes to agent classes or existing capabilities.

---

## Part 4: Storage Implementation

### 4.1 Span Producer (Agent → Kafka)

**New file**: `colony/python/colony/agents/observability/producer.py`

```python
class SpanProducer:
    """Produces span records to Kafka."""

    def __init__(self, kafka_bootstrap: str, topic: str = "colony.spans"):
        self._producer: AIOKafkaProducer  # aiokafka async producer
        self._topic = topic

    async def start(self) -> None:
        """Initialize and start the Kafka producer."""
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode(),
            key_serializer=lambda k: k.encode(),
            acks=1,          # Leader ack (fast, durable enough for spans)
            linger_ms=50,    # Batch for throughput
            batch_size=65536,
        )
        await self._producer.start()

    async def send_spans(self, spans: list[Span]) -> None:
        """Send batch of spans. Key = trace_id for partition affinity."""
        for span in spans:
            await self._producer.send(
                self._topic,
                key=span.trace_id,
                value=span.model_dump(),
            )

    async def stop(self) -> None:
        await self._producer.stop()
```

Uses `aiokafka` — the standard async Python Kafka client. Added to requirements.

### 4.2 Span Consumer (Kafka → PostgreSQL)

**New file**: `colony/python/colony/agents/observability/consumer.py`

```python
class SpanConsumer:
    """Consumes spans from Kafka and sinks to PostgreSQL."""

    def __init__(self, kafka_bootstrap: str, db_pool: asyncpg.Pool,
                 topic: str = "colony.spans", group_id: str = "colony-pg-sink"):
        ...

    async def run(self) -> None:
        """Consume loop: read batch → INSERT INTO spans."""
        consumer = AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=self._kafka_bootstrap,
            group_id=self._group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
        )
        await consumer.start()
        try:
            async for msg in consumer:
                span_data = json.loads(msg.value)
                await self._upsert_span(span_data)  # INSERT ... ON CONFLICT DO UPDATE
        finally:
            await consumer.stop()
```

Runs as a background task in the dashboard backend or as a standalone service.

### 4.3 Span Query Store (PostgreSQL → Dashboard)

**New file**: `colony/python/colony/agents/observability/store.py`

```python
class SpanQueryStore:
    """Read-side: query spans from PostgreSQL for the dashboard API."""

    def __init__(self, db_pool: asyncpg.Pool):
        ...

    async def get_spans(self, trace_id: str, run_id: str | None = None,
                        kind: str | None = None, limit: int = 5000) -> list[dict]:
        """Query spans for a trace with optional filters."""

    async def list_traces(self, limit: int = 100) -> list[dict]:
        """List all traces (sessions with spans), ordered by most recent."""

    async def get_trace_summary(self, trace_id: str) -> dict:
        """Aggregate stats: span count, total tokens, error count, duration."""
```

### 4.4 Background Flushing in `TracingCapability`

`TracingCapability` buffers spans in-memory and flushes to Kafka in a background task:
- Every `flush_interval` (default 0.5s): batch send up to `flush_batch_size` spans via `SpanProducer`
- On agent stop/suspend: flush remaining buffer synchronously
- Failures are logged but don't crash the agent (best-effort)
- Kafka's `linger_ms=50` + `batch_size=64KB` further optimizes network usage

### 4.5 PostgreSQL Migration

**New file**: `colony/python/colony/agents/observability/migrations.py`

Auto-creates the `spans` table on first connection if it doesn't exist. Uses `asyncpg`.

---

## Part 5: Dashboard Backend — Trace API + SSE

### 5.1 Connections

**Modified file**: `colony/web_ui/backend/services/colony_connection.py`
- Add `self._db_pool` (asyncpg pool to `postgres:5432`) for span queries
- Add `self._kafka_consumer` for SSE streaming
- Expose via `colony.get_db_pool()` and `colony.get_kafka_consumer()`

### 5.2 New Router

**New file**: `colony/web_ui/backend/routers/traces.py`

```
GET  /api/v1/traces?session_id=...&limit=100
     → List traces (from PostgreSQL via SpanQueryStore)

GET  /api/v1/traces/{trace_id}/spans?run_id=...&kind=...
     → All spans for a trace, with optional filters (PostgreSQL)

GET  /api/v1/stream/traces/{trace_id}
     → SSE stream of spans (real-time via Kafka consumer)
```

SSE endpoint:
1. First: query PostgreSQL for all existing spans (initial load)
2. Then: consume from Kafka topic `colony.spans` with a dedicated consumer group (`colony-sse-{trace_id}`), filtering by `trace_id` key
3. Yield matching spans as SSE events
4. Frontend accumulates spans and rebuilds tree in real-time

Note: Each SSE connection creates a Kafka consumer with `auto_offset_reset="latest"` after the initial PG load. The consumer is cleaned up on SSE disconnect.

### 5.3 Kafka → PostgreSQL Sink

The dashboard backend also runs the `SpanConsumer` (from Part 4.2) as a background task on startup. This ensures spans flow from Kafka to PostgreSQL regardless of whether anyone is viewing the dashboard.

### 5.4 Register Router

**Modified file**: `colony/web_ui/backend/main.py` — register traces router, start PG sink consumer on startup.

---

## Part 6: Dashboard Frontend — Trace View

### 6.1 New Types

**Modified file**: `frontend/src/api/types.ts`

```typescript
interface TraceSpan {
  span_id: string;
  trace_id: string;       // = session_id
  parent_span_id: string | null;
  run_id: string | null;
  agent_id: string;
  name: string;
  kind: string;           // "run", "infer", "action", "plan", etc.
  start_wall: number;
  duration_ms: number | null;
  status: "running" | "ok" | "error";
  error: string | null;
  input_summary: Record<string, unknown>;
  output_summary: Record<string, unknown>;
  input_tokens: number | null;
  output_tokens: number | null;
  cache_read_tokens: number | null;
  model_name: string | null;
  context_page_ids: string[] | null;
  tags: string[];
  metadata: Record<string, unknown>;
}

interface TraceSummary {
  trace_id: string;       // = session_id
  agent_id: string;
  status: string;
  start_time: number;
  span_count: number;
  run_count: number;
  total_tokens: number;
}
```

### 6.2 New Hooks

**New file**: `frontend/src/api/hooks/useTraces.ts`

- `useTraces(sessionId)` — List traces / summary (REST, PG-backed)
- `useTraceSpans(traceId, filters?)` — Fetch all spans for a trace (REST, PG-backed)
- `useTraceStream(traceId)` — SSE → accumulates spans into `Map<span_id, TraceSpan>`

### 6.3 Trace Waterfall View

**New file**: `frontend/src/components/traces/TraceView.tsx`

```
┌─────────────────────────────────────────────────────────────┐
│ Session: session_abc  Agent: coordinator  Status: ● Active  │
│ Duration: 45.2s  Spans: 147  Tokens: 12,841  Runs: 3        │
│ Filter: [Run ▾] [Kind ▾] [Status ▾]                         │
├──────────────────────────────────┬──────────────────────────┤
│  WATERFALL (left panel)          │  DETAIL (right panel)    │
│                                  │                          │
│  ▼ Run: "analyze auth module"    │  Name: dispatch:analyze  │
│    ▼ agent_step #1 ██████ 4.2s   │  Kind: action            │
│      ▼ execute_iteration         │  Agent: analyzer-01      │
│        plan_step ███ 1.2s        │  Duration: 2.8s          │
│        ▼ dispatch:analyze_code   │  Status: OK              │
│          infer ████████ 2.1s     │                          │
│            512 in / 89 out tkns  │  Input:                  │
│          page_request █ 0.1s     │  { action: "analyze",    │
│        dispatch:write_result     │    target: "auth/..." }  │
│          blackboard_write        │                          │
│    ▼ agent_step #2 █████ 3.1s    │  Output:                 │
│      ...                         │  { result: "Found 3..." }│
│  ▼ Run: "fix token validation"   │                          │
│    (reads data from run above)   │  VCM Pages: [page-1, .] │
│    ▼ agent_step #1 ...           │                          │
│                                  │                          │
│  ▼ child:analyzer-02 ███ 5.0s    │                          │
│    (child agent spans inline)    │                          │
├──────────────────────────────────┴──────────────────────────┤
│  ■ run  ■ infer  ■ action  ■ plan  ■ page_request  ■ other │
└─────────────────────────────────────────────────────────────┘
```

Components:
- **`TraceWaterfall`**: Tree of spans as nested rows with timing bars, color-coded by `SpanKind`
  - Each row: indent (depth), icon (by kind), name, duration bar, token count
  - Timing bars proportional to root span duration
  - Color coding by SpanKind
  - Click row → detail panel
  - Real-time: new spans animate in as SSE delivers them
- **`SpanDetail`**- **`SpanDetail`**: Right panel — input/output, tokens, cache stats, VCM page IDs, errors
  - Input/output (collapsible JSON)
  - LLM-specific: tokens, model, cache stats, VCM page IDs
  - Error (red highlight)
  - Tags and metadata
- **`TraceHeader`**: Summary bar + filter dropdowns (filter by `run_id`, `kind`, `agent_id`)

### 6.4 Traces List Tab

**New file**: `frontend/src/components/traces/TracesTab.tsx`

```
┌───────────────────────────────────────────────────────────┐
│ Session: [session_abc ▾]    Filter: [kind ▾] [status ▾]   │
├───────────────────────────────────────────────────────────┤
│ trace_id    agent       status  spans  tokens  duration   │
│ run_abc123  coordinator ● ok    47     3241    12.3s      │
│ run_def456  analyzer    ● ok    23     1890    8.7s       │
│ run_ghi789  coordinator ◉ run   12     450     ...        │
└───────────────────────────────────────────────────────────┘
```

Lists all session-level traces.
Click a trace → opens waterfall view.
Filter by run within.

### 6.5 Integration Points

- **Sessions tab**: Add "View Trace" button on each session/run row
- **AppShell**: Add "Traces" tab between Sessions and VCM

---

## Part 7: Colony-Specific Enrichments

| Feature | Span Type | What it captures |
|---------|-----------|-----------------|
| **VCM Context** | `INFER` | `context_page_ids` — which pages were in KV cache during inference |
| **Cache Efficiency** | `INFER` | `cache_read_tokens` / `cache_write_tokens` — prefix cache hits |
| **Blackboard Ops** | `BLACKBOARD_OP` | scope, key, operation type, value size |
| **Action Details** | `ACTION` | action_id, action_type, owning capability, result |
| **Planning Details** | `PLAN` | capabilities in scope, action groups selected, generated plan |
| **Cross-Run Data Flow** | `BLACKBOARD_OP` | Visible in session trace: run A writes, run B reads same key |
| **Child Correlation** | `CHILD_SPAWN` | Links parent trace to child trace via `metadata.child_trace_id` |

---

## Files Summary

### New Files (11)
| File | Purpose |
|------|---------|
| `colony/agents/observability/__init__.py` | Package init, public API |
| `colony/agents/observability/models.py` | Span, SpanKind, SpanStatus |
| `colony/agents/observability/config.py` | TracingConfig dataclass |
| `colony/agents/observability/context.py` | contextvars for current span/trace |
| `colony/agents/observability/capability.py` | TracingCapability (hook-based instrumentation) |
| `colony/agents/observability/producer.py` | SpanProducer (agent → Kafka) |
| `colony/agents/observability/consumer.py` | SpanConsumer (Kafka → PostgreSQL sink) |
| `colony/agents/observability/store.py` | SpanQueryStore (PostgreSQL → dashboard queries) |
| `colony/agents/observability/migrations.py` | PostgreSQL schema auto-creation |
| `web_ui/backend/routers/traces.py` | REST + SSE endpoints |
| `web_ui/frontend/src/api/hooks/useTraces.ts` | TanStack Query + SSE hooks |
| `web_ui/frontend/src/components/traces/TracesTab.tsx` | Trace list + waterfall view |

### Modified Files (7)
| File | Change |
|------|--------|
| `colony/agents/config.py` | Add `tracing: TracingConfig` to `AgentSystemConfig` |
| `colony/agents/base.py` | Auto-add TracingCapability in `start_agent()` when enabled |
| `web_ui/backend/main.py` | Register traces router, start PG sink consumer |
| `web_ui/backend/services/colony_connection.py` | Add asyncpg pool + Kafka consumer |
| `web_ui/frontend/src/api/types.ts` | Add TraceSpan, TraceSummary types |
| `web_ui/frontend/src/components/layout/AppShell.tsx` | Add Traces tab |
| `colony-env/docker-compose.yaml` | Add Kafka service (bitnami/kafka KRaft) |

---

## Implementation Order

| Step | What | Depends On |
|------|------|-----------|
| 1 | Data models + config | — |
| 2 | Context propagation (contextvars) | 1 |
| 3 | Kafka docker-compose + `SpanProducer` + `SpanConsumer` | 1 |
| 4 | `SpanQueryStore` + PG migration | 1 |
| 5 | `TracingCapability` (hooks + session-level trace + flush to Kafka) | 1, 2, 3 |
| 6 | Wire into `AgentSystemConfig` + auto-add | 5 |
| 7 | Backend router (REST + SSE) + PG/Kafka connections | 3, 4 |
| 8 | Frontend types + hooks | 7 |
| 9 | Frontend `TracesTab` + `TraceView` | 8 |
| 10 | Integration (Sessions → Trace link, AppShell tab) | 9 |

Steps 1-4 are largely independent. Step 5 depends on 1-3.

---

## Design Decisions & Rationale

### Why out-of-line (hooks) instead of inline (@traceable decorators)?

Colony already has a battle-tested hook system with `@hookable` on every critical method. Using AROUND hooks means zero changes to agent code — `TracingCapability` is the only addition. The MemoryCapability already uses this exact pattern.
- **Zero changes to agent code** — adding `TracingCapability` is the only change
- **Respects existing patterns** — `MemoryCapability` already uses hooks this way
- **Easy to disable** — remove the capability or set `enabled=False`
- **Composable** — other capabilities can add their own tracing hooks

The `@traceable` pattern (LangSmith-style) would require modifying every method signature and adding decorator imports throughout — invasive and fragile.

### Why trace_id = session_id (not run_id)?

A session is the natural trace boundary: agents persist across runs, data flows between runs via blackboard/VCM, and a user's full interaction within a session is one logical "trace." Each run becomes a top-level span within the session trace. This strictly subsumes per-run tracing while enabling cross-run correlation.

### Why Kafka + PostgreSQL?

Colony is designed for 1,000s to 100,000s of agents. The span pipeline must scale accordingly.

- **Kafka** is purpose-built for high-throughput, durable, ordered event streams with fan-out to multiple consumers. It's what LangSmith uses. In KRaft mode it's a single container (~300MB) added to docker-compose — completely transparent to users who just run `colony-env up`.
- **PostgreSQL** provides the structured query layer Kafka lacks. Historical span queries (`SELECT ... WHERE kind='infer' AND input_tokens > 1000`), aggregations for trace summaries, and durable storage that survives topic retention expiry.
- **Not Redis Streams**: Kafka provides native partitioned fan-out to independent consumers without competing for the same Redis resources used by blackboard events and state management. Separating the span pipeline from the operational Redis workload is the right architectural boundary at scale.
- **Not PostgreSQL-only**: No real-time streaming primitive — would require polling for the dashboard waterfall.

### Future: OpenTelemetry

The Kafka consumer architecture makes it straightforward to add an `OTLPExporterConsumer` that converts Colony Spans to OTel Spans and sends to Jaeger/Tempo/Datadog — just another consumer on the `colony.spans` topic, zero changes to agents or existing consumers.

---

## Verification

```bash
colony-env down && colony-env up --workers 3 && colony-env run --local-repo /home/anassar/workspace/agents/crewAI/ --config my_analysis.yaml --verbose
```

After the run starts, open `http://localhost:8080`:
1. **Traces tab**: Shows session-level traces
2. Click a trace → **Waterfall view** with run spans as top-level groups
3. Spans stream in real-time as the agent works
4. Click a span → **Detail panel** with inputs/outputs, tokens, VCM pages
5. Multiple runs within a session visible in the same trace tree
6. Filter by run, kind, agent within the waterfall
7. Cross-run data flow visible (run A writes blackboard, run B reads it)
