# Observability

Colony provides built-in distributed observability for long-running multi-agent sessions. Both **traces** (structured span data) and **logs** (standard Python logging) are durably persisted via Kafka and PostgreSQL, enabling post-mortem debugging even after agents have stopped.

## Architecture

<style>
/* ── Observability diagram ── */
.obs-svg { width: 100%; max-width: 820px; margin: 0 auto; display: block; }
.obs-svg text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.obs-svg .title  { font-size: 13.5px; font-weight: 600; fill: #1e1b4b; }
.obs-svg .body   { font-size: 11px;   fill: #374151; }
.obs-svg .detail { font-size: 10px;   fill: #6b7280; }
.obs-svg .mono   { font-size: 10px;   fill: #57534e; font-family: "SF Mono", "Fira Code", monospace; }
.obs-svg .box    { rx: 8; ry: 8; stroke-width: 1.4; }
.obs-svg .arrow       { stroke-width: 1.6; fill: none; }
.obs-svg .arrow-label { font-size: 10px; fill: #6b7280; font-weight: 500; }
@keyframes obsDash { to { stroke-dashoffset: -14; } }
.obs-svg .flow { stroke-dasharray: 7 4; animation: obsDash 1s linear infinite; }
/* Light mode box fills */
.obs-svg .r-agent { fill: #f5f3ff; stroke: #8b5cf6; }
.obs-svg .r-cap   { fill: #ede9fe; stroke: #a78bfa; }
.obs-svg .r-kafka { fill: #fffbeb; stroke: #f59e0b; }
.obs-svg .r-dash  { fill: #ecfdf5; stroke: #10b981; }
.obs-svg .r-pg    { fill: #eff6ff; stroke: #3b82f6; }
.obs-svg .r-table { fill: #dbeafe; stroke: #3b82f6; }
/* Dark mode */
[data-md-color-scheme="slate"] .obs-svg .title  { fill: #e0e7ff; }
[data-md-color-scheme="slate"] .obs-svg .body   { fill: #cbd5e1; }
[data-md-color-scheme="slate"] .obs-svg .detail { fill: #94a3b8; }
[data-md-color-scheme="slate"] .obs-svg .mono   { fill: #a8a29e; }
[data-md-color-scheme="slate"] .obs-svg .arrow-label { fill: #94a3b8; }
[data-md-color-scheme="slate"] .obs-svg .r-agent { fill: #1e1b4b; stroke: #6d28d9; }
[data-md-color-scheme="slate"] .obs-svg .r-cap   { fill: #2e1065; stroke: #7c3aed; }
[data-md-color-scheme="slate"] .obs-svg .r-kafka { fill: #422006; stroke: #d97706; }
[data-md-color-scheme="slate"] .obs-svg .r-dash  { fill: #052e16; stroke: #059669; }
[data-md-color-scheme="slate"] .obs-svg .r-pg    { fill: #172554; stroke: #2563eb; }
[data-md-color-scheme="slate"] .obs-svg .r-table { fill: #1e3a5f; stroke: #3b82f6; }
</style>

<div>
<svg class="obs-svg" viewBox="0 0 820 340" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="obs-ah" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <path d="M0,0 L10,3.5 L0,7 Z" fill="#7c3aed"/>
    </marker>
    <marker id="obs-ah-amber" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <path d="M0,0 L10,3.5 L0,7 Z" fill="#f59e0b"/>
    </marker>
    <marker id="obs-ah-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <path d="M0,0 L10,3.5 L0,7 Z" fill="#10b981"/>
    </marker>
    <marker id="obs-ah-blue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <path d="M0,0 L10,3.5 L0,7 Z" fill="#3b82f6"/>
    </marker>
  </defs>

  <!-- ══════════ AGENT / DEPLOYMENT ══════════ -->
  <rect class="box r-agent" x="20" y="20" width="250" height="130"/>
  <text class="title" x="145" y="42" text-anchor="middle">Agent / Deployment</text>

  <rect class="box r-cap" x="36" y="54" width="218" height="36"/>
  <text class="body" x="145" y="70" text-anchor="middle">AgentTracingFacility</text>
  <text class="detail" x="145" y="83" text-anchor="middle">SpanProducer</text>

  <rect class="box r-cap" x="36" y="98" width="218" height="36"/>
  <text class="body" x="145" y="114" text-anchor="middle">Python logging.*</text>
  <text class="detail" x="145" y="127" text-anchor="middle">KafkaLogHandler</text>

  <!-- ══════════ KAFKA ══════════ -->
  <rect class="box r-kafka" x="340" y="20" width="140" height="130"/>
  <text class="title" x="410" y="42" text-anchor="middle">Kafka</text>

  <rect class="box r-kafka" x="352" y="54" width="116" height="30" style="fill:#fef3c7"/>
  <text class="mono" x="410" y="73" text-anchor="middle">colony.spans</text>

  <rect class="box r-kafka" x="352" y="98" width="116" height="30" style="fill:#fef3c7"/>
  <text class="mono" x="410" y="117" text-anchor="middle">colony.logs</text>

  <!-- ══════════ DASHBOARD BACKEND ══════════ -->
  <rect class="box r-dash" x="20" y="200" width="250" height="120"/>
  <text class="title" x="145" y="222" text-anchor="middle">Dashboard Backend</text>

  <text class="body" x="40" y="245">SpanConsumer</text>
  <text class="body" x="40" y="262">LogConsumer</text>
  <text class="body" x="40" y="286">SpanQueryStore</text>
  <text class="body" x="40" y="303">LogQueryStore</text>

  <!-- ══════════ POSTGRESQL ══════════ -->
  <rect class="box r-pg" x="560" y="200" width="240" height="120"/>
  <text class="title" x="680" y="222" text-anchor="middle">PostgreSQL</text>

  <rect class="box r-table" x="576" y="236" width="100" height="28"/>
  <text class="mono" x="626" y="254" text-anchor="middle">spans</text>

  <rect class="box r-table" x="688" y="236" width="100" height="28"/>
  <text class="mono" x="738" y="254" text-anchor="middle">logs</text>

  <text class="detail" x="626" y="280" text-anchor="middle">trace_id, agent_id,</text>
  <text class="detail" x="626" y="292" text-anchor="middle">tokens, duration</text>
  <text class="detail" x="738" y="280" text-anchor="middle">session_id, level,</text>
  <text class="detail" x="738" y="292" text-anchor="middle">message, context</text>

  <!-- ═══ ARROWS: Agent → Kafka ═══ -->
  <line class="arrow flow" x1="254" y1="72" x2="340" y2="72" stroke="#7c3aed" marker-end="url(#obs-ah)"/>
  <text class="arrow-label" x="297" y="65" text-anchor="middle">spans</text>

  <line class="arrow flow" x1="254" y1="116" x2="340" y2="116" stroke="#7c3aed" marker-end="url(#obs-ah)"/>
  <text class="arrow-label" x="297" y="109" text-anchor="middle">logs</text>

  <!-- ═══ ARROWS: Kafka → Dashboard (downward) ═══ -->
  <path class="arrow flow" d="M 390,150 L 390,170 Q 390,180 380,180 L 180,180 Q 170,180 170,190 L 170,200" stroke="#f59e0b" marker-end="url(#obs-ah-amber)"/>
  <text class="arrow-label" x="290" y="176" text-anchor="middle">consume</text>

  <!-- ═══ ARROWS: Dashboard → PostgreSQL ═══ -->
  <line class="arrow" x1="270" y1="248" x2="560" y2="248" stroke="#10b981" marker-end="url(#obs-ah-green)"/>
  <text class="arrow-label" x="415" y="242" text-anchor="middle">INSERT</text>

  <line class="arrow" x1="270" y1="290" x2="560" y2="290" stroke="#3b82f6" marker-end="url(#obs-ah-blue)"/>
  <text class="arrow-label" x="415" y="284" text-anchor="middle">SELECT (query)</text>

  <!-- ═══ ARROW: Kafka → PostgreSQL (direct label) ═══ -->
  <path class="arrow flow" d="M 480,85 L 560,85 Q 580,85 600,100 L 680,200" stroke="#f59e0b" style="stroke-dasharray:4 3" marker-end="url(#obs-ah-amber)"/>
  <text class="arrow-label" x="570" y="125" text-anchor="middle" transform="rotate(40,570,125)">durable</text>

</svg>
</div>

## Traces

Traces capture structured execution spans (LLM calls, agent steps, tool invocations) with parent-child relationships, token counts, and timing data. See the [`AgentTracingFacility`](../../src/polymathera/colony/agents/observability/facility.py) for details.

**Pipeline:** `AgentTracingFacility` → `SpanProducer` → Kafka (`colony.spans`) → `SpanConsumer` → PostgreSQL `spans` table → `SpanQueryStore` → Dashboard Traces tab.

## Logs

Every Python log record emitted under the `polymathera.colony` namespace is captured, enriched with execution context, and durably stored.

### How it works

1. **`KafkaLogHandler`** — A standard Python `logging.Handler` attached to the `polymathera.colony` root logger during deployment initialization. It intercepts all log records without requiring changes to existing log calls.

2. **Context enrichment** — Each log record is enriched with the current `ExecutionContext` (`tenant_id`, `colony_id`, `session_id`, `run_id`, `trace_id`) when available. This enables filtering logs by session or correlating them with traces.

3. **Async batching** — Log records are queued in-process and flushed to Kafka asynchronously in batches (default: 50 records every 2 seconds). The `emit()` method never blocks the caller.

4. **`LogConsumer`** — Runs in the dashboard backend container. Reads from the `colony.logs` Kafka topic and batch-inserts into PostgreSQL.

5. **`LogQueryStore`** — Provides filtered, paginated queries over persisted logs:
   - Filter by session, run, trace, actor class, log level
   - Full-text search in messages (case-insensitive)
   - Time range queries
   - Aggregate statistics (error counts, actor summaries)

### Querying logs

The dashboard exposes persistent log endpoints:

```
GET /api/v1/logs/persistent?session_id=X&level=WARNING&limit=100
GET /api/v1/logs/persistent?run_id=Y&search=timeout
GET /api/v1/logs/persistent?actor_class=StandaloneAgentDeployment&since=1712000000
GET /api/v1/logs/persistent/stats?session_id=X
GET /api/v1/logs/persistent/actors
```

These work **even after the application stops** — logs are durably stored in PostgreSQL as long as the dashboard and database containers are running.

### Log record schema

Each log record stored in PostgreSQL contains:

| Field | Type | Description |
|-------|------|-------------|
| `log_id` | TEXT | Unique identifier |
| `timestamp` | TIMESTAMPTZ | When the log was emitted |
| `level` | TEXT | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `logger_name` | TEXT | Python logger name (e.g., `polymathera.colony.agents.base`) |
| `message` | TEXT | Log message |
| `module` | TEXT | Python module name |
| `func_name` | TEXT | Function that emitted the log |
| `line_no` | INTEGER | Source line number |
| `pid` | INTEGER | Process ID |
| `actor_class` | TEXT | Deployment class name (e.g., `StandaloneAgentDeployment`) |
| `node_id` | TEXT | Ray node ID |
| `tenant_id` | TEXT | Tenant ID (from execution context) |
| `colony_id` | TEXT | Colony ID (from execution context) |
| `session_id` | TEXT | Session ID (from execution context) |
| `run_id` | TEXT | Run ID (from execution context) |
| `trace_id` | TEXT | Trace ID (for correlation with spans) |
| `exc_info` | TEXT | Exception traceback if present |

### Indexes

Logs are indexed for fast queries on common access patterns:

- `(session_id, timestamp DESC)` — all logs for a session, newest first
- `(run_id, timestamp DESC)` — all logs for a specific run
- `(actor_class, timestamp DESC)` — all logs from a deployment type
- `(level, timestamp DESC)` — find errors/warnings quickly
- `(trace_id, timestamp DESC)` — correlate logs with traces
- `(timestamp DESC)` — global time-ordered access

### Setup

The log pipeline is automatic. When `KAFKA_BOOTSTRAP` is set in the environment (which it is in all Docker containers), every deployment attaches the `KafkaLogHandler` during initialization. No configuration required.

To disable the log pipeline, unset the `KAFKA_BOOTSTRAP` environment variable.

## Infrastructure requirements

Both traces and logs require:

- **Kafka** — Message broker for reliable delivery and replay
- **PostgreSQL** — Durable storage and indexed queries
- **Dashboard container** — Runs the Kafka consumers that sink to PostgreSQL

All three are included in the default `colony-env` Docker Compose setup.
