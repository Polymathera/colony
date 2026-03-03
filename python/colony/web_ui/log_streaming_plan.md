# Plan: Log Streaming with Per-Actor Views

## Context

Phase 2 dashboard items (2.1–2.5, Prometheus streaming) are all complete. The remaining gap is **log streaming**: the dashboard currently has two basic REST endpoints (`/logs/actors`, `/logs/file`) that proxy to the Ray Dashboard, but no real-time streaming and no frontend LogViewer.

**Problem**: In the terminal, all logs from all actors are mixed together, making it hard to debug specific components. Lines from different actors are prefixed like `(AgentSystemDeployment pid=500, ip=172.18.0.6)` but interleaved. The dashboard should provide **separate log views per actor** with real-time streaming.

**Ray Log API** (port 8265):
- `GET /api/v0/logs?node_id=<id>` — list available log files on a node
- `GET /api/v0/logs/stream?actor_id=<id>&interval=0.5` — **real-time HTTP chunked streaming** per actor
- `GET /api/v0/logs/file?actor_id=<id>&lines=1000` — fetch last N lines
- Actor metadata via `ray.util.state.list_actors(detail=True)` — returns `actor_id`, `class_name`, `node_id`, `pid`, `repr_name`

---

## Design

### Backend

#### 1. Actor discovery endpoint — `GET /logs/sources`

Uses `ray.util.state.list_actors(detail=True)` to list all live actors with metadata. Returns a list of log sources the frontend can select from.

**File**: `colony/web_ui/backend/routers/logs.py` (extend existing)

```python
@router.get("/logs/sources")
async def list_log_sources(colony: ColonyConnection = Depends(get_colony)):
    """List available log sources (actors) with metadata for per-actor log views."""
    import ray.util.state
    actors = ray.util.state.list_actors(detail=True, limit=200)
    sources = []
    for a in actors:
        if a.state != "ALIVE":
            continue
        sources.append({
            "actor_id": a.actor_id,
            "class_name": a.class_name,
            "node_id": a.node_id,
            "pid": a.pid,
            "repr_name": a.repr_name or a.class_name,
        })
    return sources
```

#### 2. SSE log stream endpoint — `GET /stream/logs`

Proxies Ray Dashboard's HTTP chunked streaming as SSE events. Each SSE event is a batch of log lines (or a single line).

**File**: `colony/web_ui/backend/streaming/sse.py` (extend existing)

```python
@router.get("/stream/logs")
async def stream_logs(
    actor_id: str = Query(None),
    node_id: str = Query(None),
    pid: int = Query(None),
    lines: int = Query(200, ge=1, le=2000),
    colony: ColonyConnection = Depends(get_colony),
) -> StreamingResponse:
    """SSE stream of logs for a specific actor/worker."""
    # Build URL to Ray Dashboard's streaming endpoint
    params = {"lines": lines, "interval": "0.5"}
    if actor_id:
        params["actor_id"] = actor_id
    elif pid and node_id:
        params["pid"] = str(pid)
        params["node_id"] = node_id

    async def _generate():
        async with colony._http_client.stream(
            "GET", f"{colony.ray_dashboard_url}/api/v0/logs/stream", params=params
        ) as resp:
            async for chunk in resp.aiter_text():
                if chunk.strip():
                    # Send each line as an SSE event
                    for line in chunk.splitlines():
                        yield f"data: {json.dumps({'line': line})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream", ...)
```

**Key**: `httpx.AsyncClient.stream()` keeps the connection open to Ray Dashboard and forwards chunks as SSE events.

#### 3. Fetch recent logs (non-streaming) — update existing `GET /logs/file`

Already exists but uses `node_id + filename`. Update to also accept `actor_id` directly (simpler for frontend).

### Frontend

#### 4. Log types — `types.ts`

```typescript
export interface LogSource {
  actor_id: string;
  class_name: string;
  node_id: string;
  pid: number;
  repr_name: string;
}
```

#### 5. Hooks — `useLogStream.ts` (new)

- `useLogSources()` — TanStack Query hook polling `/logs/sources` every 10s
- `useLogStream(actorId)` — SSE EventSource hook connecting to `/stream/logs?actor_id=<id>`, returns `lines: string[]` with a rolling buffer (max 5000 lines)

#### 6. LogsTab component — `components/logs/LogsTab.tsx` (new)

Layout:
```
┌─────────────────────────────────────────────┐
│ [Source selector dropdown]   [Level filter] │
│ [Auto-scroll toggle]   [Clear]              │
├─────────────────────────────────────────────┤
│ 2025-03-03 10:30:45 INFO  Spawning agent... │
│ 2025-03-03 10:30:45 INFO  Loading pages...  │
│ 2025-03-03 10:30:46 DEBUG Cache hit for...  │
│ ...                                         │
│ (auto-scroll to bottom)                     │
└─────────────────────────────────────────────┘
```

- **Source selector**: dropdown populated from `useLogSources()`, grouped by `class_name`. Options: "All Sources" + each actor (showing `class_name (pid=N)`)
- **Level filter**: multi-select for DEBUG/INFO/WARNING/ERROR (parsed from log lines)
- **Log display**: monospace `<pre>` with syntax-colored levels (green=INFO, yellow=WARNING, red=ERROR, gray=DEBUG)
- **Auto-scroll**: toggle, on by default — scrolls to bottom on new lines
- **Clear**: clears the buffer
- Lines parsed by regex: `(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\S+) - (\w+) - (.*)` to extract timestamp, logger name, level, message

#### 7. Wire up — `AppShell.tsx`

Add "Logs" tab between "Blackboard" and "Metrics".

---

## Files Modified

| File | Change |
|------|--------|
| `backend/routers/logs.py` | Add `GET /logs/sources` (actor discovery) |
| `backend/streaming/sse.py` | Add `GET /stream/logs` (SSE proxy to Ray log streaming) |
| `frontend/src/api/types.ts` | Add `LogSource` type |
| `frontend/src/api/hooks/useLogStream.ts` | New: `useLogSources()`, `useLogStream()` |
| `frontend/src/components/logs/LogsTab.tsx` | New: per-actor log viewer |
| `frontend/src/components/layout/AppShell.tsx` | Add "Logs" tab |

**No Colony core files modified.** All data comes from Ray Dashboard API.

## Implementation Order

1. Backend: `GET /logs/sources` actor discovery
2. Backend: `GET /stream/logs` SSE proxy
3. Frontend: types + hooks
4. Frontend: LogsTab component
5. Wire up AppShell

## Verification

1. `colony-env down && colony-env up --workers 3 && colony-env run --local-repo /home/anassar/workspace/agents/crewAI/ --config my_analysis.yaml --verbose`
2. Open `http://localhost:8080`, go to Logs tab
3. Source dropdown should list actors (AgentSystemDeployment, SessionManagerDeployment, VCM, etc.)
4. Selecting an actor shows its logs streaming in real-time
5. Level filter hides/shows lines by severity
6. Auto-scroll keeps the view at the bottom as new lines arrive
