# Plan: Colony Web Dashboard — Design & Implementation

## Context

Developing and debugging Colony agents from the CLI is hitting its limits. We need a web-based dashboard for monitoring deployments, visualizing agent state, viewing logs/metrics, and interacting with agents in real-time. The dashboard should be a high-quality, extensible development tool that grows with the agent system.

**Reference**: Google ADK Web UI (Angular 21 + Material) — we take inspiration from its tabbed layout and real-time streaming patterns, but use React + Vite for a lighter, faster development experience.

## Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Frontend | React 19 + Vite 6 + TypeScript | Fast dev cycle, rich 3D/chart ecosystem |
| UI Components | shadcn/ui (Radix + Tailwind CSS) | High-quality, accessible, composable |
| 3D Graph | react-three-fiber + @react-three/drei | Page graph visualization |
| Charts | recharts | Token usage, metrics dashboards |
| Server State | TanStack Query v5 | Polling, caching, real-time sync |
| Layout | react-resizable-panels | ADK-style resizable panes |
| Global State | zustand | Minimal UI preferences |
| Backend | FastAPI + uvicorn | Async, Pydantic model reuse, WebSocket + SSE |
| Connection | Ray Client API (port 10001) + Redis (6379) + Postgres (5432) | Direct access to Colony deployments inside Docker |

## Architecture

The backend runs **inside Docker** as a new `dashboard` service in `docker-compose.yml`, with direct access to the `colony-net` network. The frontend is built and served as static files by FastAPI. For development, the Vite dev server runs on the host with a proxy to the backend.

```
Host Machine                          Docker Network (colony-net)
+--------------------+                +--------------------------------+
| Vite Dev (5173)    |---proxy------->| dashboard (8080)               |
| (HMR, hot reload)  |               |   FastAPI backend               |
+--------------------+                |   Serves built frontend (prod)  |
                                      |     |                          |
     Browser (localhost:5173 or 8080) |     +---> ray-head (6379/10001)|
                                      |     +---> redis (6379)         |
                                      |     +---> postgres (5432)      |
                                      +--------------------------------+
```

### Key data flow

1. **REST**: Frontend → FastAPI router → `get_deployment(app_name, name)` → `DeploymentHandle` → Ray actor RPC → response → JSON
2. **Real-time**: Colony `EventBus` (Redis pub/sub) → `event_bridge.py` → WebSocket → Frontend
3. **Metrics**: Prometheus (port 9090 on ray-head) → FastAPI proxy → recharts
4. **Logs**: Ray log API (port 8265) → SSE endpoint → `LogViewer` component

### How the backend connects to Colony

The dashboard backend imports Colony's serving framework directly (same Docker image, same `PYTHONPATH`):

```python
from colony.distributed.ray_utils.serving import get_deployment
from colony.system import get_session_manager, get_vcm, get_agent_system

# get_deployment() uses ray.get_actor() to find named proxy actors
# Works via Ray Client: ray.init(address="ray://ray-head:10001")
handle = get_deployment("polymath-test", "agent_system")
agents = await handle.list_all_agents()
```

Key Colony APIs consumed by the dashboard:
- `ApplicationRegistry` (SharedState in Redis) — list apps and deployments
- `AgentSystemDeployment` — `list_all_agents()`, `get_agent_info()`, `spawn_from_blueprint()`
- `SessionManagerDeployment` — `list_sessions()`, `get_session_runs()`, `get_run()`, `get_stats()`
- `VirtualPageTableState` (SharedState in Redis) — page table, working set, page groups
- `RedisBackend` — blackboard entries with indexed queries
- Redis pub/sub — real-time blackboard events
- Prometheus HTTP API — pre-existing metrics (blackboard ops, cache hit rate, Redis latency, etc.)

## Directory Structure

```
colony/web_ui/
├── pyproject.toml                      # Backend package (depends on colony)
├── backend/
│   ├── __init__.py
│   ├── main.py                         # FastAPI app, lifespan (startup/shutdown)
│   ├── config.py                       # DashboardConfig dataclass
│   ├── dependencies.py                 # FastAPI DI: colony connection, redis, db
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── infrastructure.py           # GET /infra/status, /infra/containers, /infra/redis
│   │   ├── deployments.py              # GET /deployments/, /deployments/{app}/{name}/health
│   │   ├── agents.py                   # GET /agents/, /agents/{id}, /agents/{id}/capabilities
│   │   ├── sessions.py                 # GET /sessions/, /sessions/{id}/runs, /sessions/runs/{id}
│   │   ├── vcm.py                      # GET /vcm/pages, /vcm/working-set, /vcm/stats
│   │   ├── page_graph.py               # GET /graph/{scope_id} (nodes+edges JSON)
│   │   ├── blackboard.py               # GET /blackboard/entries, /blackboard/scopes
│   │   ├── metrics.py                  # GET /metrics/prometheus, /metrics/tokens
│   │   └── commands.py                 # POST /commands/spawn, /commands/send-task
│   ├── services/
│   │   ├── __init__.py
│   │   ├── colony_connection.py        # ray.init + get_deployment() + reconnection logic
│   │   ├── redis_service.py            # Async redis client for blackboard/metrics
│   │   ├── db_service.py               # asyncpg for VCM page metadata
│   │   └── log_service.py              # Ray log API proxy
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── websocket.py               # WebSocket manager + broadcast
│   │   ├── sse.py                     # SSE endpoints for logs, metrics
│   │   └── event_bridge.py            # Redis pub/sub → WebSocket forwarding
│   └── models/
│       ├── __init__.py
│       └── api_models.py             # Response models (reuse Colony Pydantic models)
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts                 # Proxy /api → backend:8080
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx                     # Root: QueryClientProvider + Router + AppShell
│       ├── config.ts                   # Runtime backend URL
│       ├── api/
│       │   ├── client.ts               # fetch wrapper with base URL
│       │   ├── types.ts                # TS types matching backend models
│       │   ├── websocket.ts            # WS hook + context provider
│       │   └── hooks/                  # TanStack Query hooks per domain
│       │       ├── useDeployments.ts
│       │       ├── useAgents.ts
│       │       ├── useSessions.ts
│       │       ├── useVCM.ts
│       │       ├── usePageGraph.ts
│       │       ├── useMetrics.ts
│       │       └── useInfrastructure.ts
│       ├── components/
│       │   ├── layout/
│       │   │   ├── AppShell.tsx       # Sidebar + tabbed main area + status bar
│       │   │   ├── TabBar.tsx
│       │   │   └── StatusBar.tsx      # Connection health, cluster info
│       │   ├── shared/
│       │   │   ├── DataTable.tsx      # Reusable sortable/filterable table
│       │   │   ├── JsonViewer.tsx     # Collapsible JSON tree
│       │   │   ├── MetricCard.tsx     # Stat card with trend spark
│       │   │   ├── LogViewer.tsx      # Virtualized log display
│       │   │   └── Badge.tsx          # Status badges
│       │   ├── dashboard/             # Overview tab
│       │   ├── agents/                # Agents tab (list + detail)
│       │   ├── sessions/              # Sessions tab (sessions + runs)
│       │   ├── vcm/                   # VCM tab (page table + working set)
│       │   ├── graph/                 # Page Graph tab (3D force graph)
│       │   ├── observability/         # Logs + Metrics tab
│       │   └── interact/              # Command/chat interface tab
│       └── store/
│           └── index.ts               # zustand UI state
```

## Backend API Design

All routes under `/api/v1/`. Colony Pydantic models reused directly as FastAPI response models.

### REST Endpoints

| Route | Source | Colony API |
|-------|--------|------------|
| `GET /infra/status` | Docker SDK, Redis ping, Postgres ping | Direct connections |
| `GET /infra/redis`  | `redis.info()` | Direct Redis |
| `GET /deployments/` | `ApplicationRegistry` (Redis shared state) | `StateManager` read |
| `GET /deployments/{app}/{name}/health` | `DeploymentProxyInfo` | Via proxy actor |
| `GET /agents/` | `AgentSystemDeployment` | `handle.list_all_agents()` |
| `GET /agents/{id}` | `AgentSystemDeployment` | `handle.get_agent_info(agent_id=id)` |
| `GET /sessions/` | `SessionManagerDeployment` | `handle.list_sessions(tenant_id=...)` |
| `GET /sessions/{id}/runs` | `SessionManagerDeployment` | `handle.get_session_runs(session_id=id)` |
| `GET /sessions/runs/{run_id}` | `SessionManagerDeployment` | `handle.get_run(run_id=run_id)` |
| `GET /sessions/stats` | `SessionManagerDeployment` | `handle.get_stats()` |
| `GET /vcm/pages` | `VirtualPageTableState` | Redis shared state read |
| `GET /vcm/working-set` | `VirtualPageTableState.client_pages` | Redis shared state read |
| `GET /vcm/stats` | `VirtualContextManager` | Deployment handle |
| `GET /graph/{scope_id}` | `PageStorage` | Load NetworkX graph → JSON `{nodes, edges}` |
| `GET /blackboard/entries` | `RedisBackend` | Direct Redis query on indexed entries |
| `GET /metrics/prometheus` | Prometheus HTTP API | Proxy `ray-head:9090/metrics` |
| `GET /metrics/tokens` | `RunResourceUsage` | Aggregate from session runs |
| `POST /commands/spawn` | `AgentSystemDeployment` | `handle.spawn_from_blueprint(...)` |
| `POST /commands/send-task` | Agent handles | Via deployment system |

### Streaming Endpoints

| Endpoint | Protocol | Source |
|----------|----------|--------|
| `WS /ws/events` | WebSocket | Redis pub/sub `{app}:blackboard:events:*` |
| `WS /ws/logs/{actor}` | WebSocket | Ray log API at `ray-head:8265` |
| `GET /stream/metrics` | SSE | Poll Prometheus every 5s |

## Frontend Layout

Tabbed interface inspired by ADK:

```
┌──────────────────────────────────────────────────────────┐
│ Colony Dashboard                          [● connected]  │
├──────────────────────────────────────────────────────────┤
│ [Overview] [Agents] [Sessions] [VCM] [Graph] [Logs] [⌘] │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  (Tab content with resizable panels)                     │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ Ray: connected │ 3 agents │ 12 pages │ 0 errors          │
└──────────────────────────────────────────────────────────┘
```

**Tabs**:
- **Overview**: Deployment cards (health, replicas), resource summary, active run count
- **Agents**: Master-detail — agent list table (filterable by type/capability) + detail panel (state, capabilities, action history, token usage)
- **Sessions**: Session list + expandable run details (events timeline, resource usage, child agents)
- **VCM**: Page data table (sortable, paginated) + working set grid (which pages on which replicas)
- **Graph**: Full-width 3D force-directed page graph (react-three-fiber). Nodes colored by status (loaded=green, cached=blue, unloaded=gray). Click to inspect. Highlight working set.
- **Observability**: Split pane — log stream (filterable by actor/level) + metrics charts (recharts)
- **Interact** (⌘): Chat-like command interface. Send tasks to agents, see streamed responses.

## Docker Compose Integration

Add `dashboard` service to `colony/python/colony/cli/deploy/docker/docker-compose.yml`:

```yaml
  dashboard:
    image: colony:local
    container_name: colony-dashboard
    command: >
      python -m colony.web_ui.backend.main
    ports:
      - "${COLONY_DASHBOARD_UI_PORT:-8080}:8080"
    environment:
      - RAY_ADDRESS=ray-head:6379
      - RAY_CLIENT_ADDRESS=ray://ray-head:10001
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - RDS_HOST=postgres
      - RDS_PORT=5432
      - RDS_USER=colony
      - RDS_PASSWORD=colony_dev
      - RDS_DB_NAME=colony
      - RAY_DASHBOARD_URL=http://ray-head:8265
      - PROMETHEUS_URL=http://ray-head:9090
    depends_on:
      ray-head:
        condition: service_healthy
    networks:
      - colony-net
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/v1/infra/status')"]
      interval: 10s
      timeout: 5s
      retries: 5
```

## CLI Integration

Add to `colony/python/colony/cli/deploy/cli.py`:

```python
@app.command()
def dashboard(port: int = 8080):
    """Open the Colony dashboard in the browser."""
    import webbrowser
    webbrowser.open(f"http://localhost:{port}")
```

The dashboard container starts automatically with `colony-env up`. The CLI command just opens the browser.

## Phased Implementation

### Phase 1 — MVP (this PR)
**Goal**: Working dashboard showing deployment status, agents, sessions, VCM pages.

**Backend**:
1. Project scaffolding: `pyproject.toml`, `main.py` with FastAPI + lifespan
2. `colony_connection.py`: Connect to Ray cluster via `ray.init()`, cache deployment handles
3. `routers/infrastructure.py`: `GET /infra/status` (Redis ping, Ray status)
4. `routers/deployments.py`: `GET /deployments/` from `ApplicationRegistry`
5. `routers/agents.py`: `GET /agents/`, `GET /agents/{id}` from `AgentSystemDeployment`
6. `routers/sessions.py`: Session/run endpoints from `SessionManagerDeployment`
7. `routers/vcm.py`: Page table and working set from `VirtualPageTableState`
8. Static file serving for built frontend

**Frontend**:
1. Vite + React + shadcn/ui + Tailwind scaffolding
2. `AppShell` with tab navigation and status bar
3. Overview tab: deployment cards, resource summary
4. Agents tab: agent list table + detail panel (JSON viewer)
5. Sessions tab: session list + run timeline
6. VCM tab: page data table + working set grid
7. TanStack Query hooks with 5s polling

**Docker**:
1. `dashboard` service in docker-compose.yml
2. Vite proxy config for dev mode (`npm run dev` on host)

### Phase 2 — Real-Time + Observability
- WebSocket `event_bridge.py` for blackboard events
- Log streaming via SSE from Ray log API
- Prometheus metrics panel with recharts
- Token usage breakdown charts (by model, by agent)
- Run event timeline visualization

### Phase 3 — Page Graph Visualization
- Graph data API (NetworkX → JSON nodes/edges)
- react-three-fiber 3D force-directed graph with d3-force-3d
- Working set overlay (loaded/cached/unloaded colors)
- LOD rendering for large graphs (10k+ nodes): instanced meshes, frustum culling
- Server-side initial layout via `networkx.spring_layout()`

### Phase 4 — Agent Interaction
- Command/chat interface with streamed responses
- Blueprint builder form UI
- Action history timeline
- Blackboard explorer with tag filtering

### Phase 5 — Advanced
- Tauri desktop packaging (~5MB, lighter than Electron)
- Remote cluster support (connect to AWS Ray clusters)
- Plugin/widget system for custom panels
- Cost tracking visualization (`RunResourceUsage.cost_usd`)

## Files to Modify

| File | Change |
|------|--------|
| `colony/cli/deploy/docker/docker-compose.yml` | Add `dashboard` service |
| `colony/cli/deploy/cli.py` | Add `dashboard` command |
| `colony/web_ui/` (new directory) | Entire dashboard codebase |

## Verification

1. `colony-env up` starts dashboard container alongside Ray + Redis
2. `http://localhost:8080` loads the dashboard UI
3. Overview tab shows deployment health indicators
4. Agents tab lists agents when a run is active
5. Sessions tab shows active sessions and their runs
6. VCM tab shows page table entries
7. Status bar shows connection health (green dot)
8. FastAPI auto-docs at `http://localhost:8080/docs`


