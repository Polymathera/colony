# Plan: Phase 3 Page Graph 3D + Phase 4/5 UI Placeholders

## Context

Phases 1-2 of the Colony Web Dashboard are complete (MVP tabs, real-time metrics/logs, VCM grid, agent hierarchy, blackboard). The user wants:

1. **Phase 3**: Real, working 3D force-directed page graph visualization backed by VCM graph data
2. **Phase 4/5 placeholders**: Functional-looking but not backend-connected UI for future features (agent chat, blueprint builder, blackboard tag filtering, remote cluster, plugins)

**Problem**: The VCM `PageStorage` stores page graphs as `nx.DiGraph` but the `VirtualContextManager` has no `@serving.endpoint` to expose graph data. We need to add Colony endpoints, a backend router, and a 3D frontend component.

---

## Part 1: Phase 3 — Page Graph 3D Visualization

### 1.1 Colony Layer: New VCM Endpoints

**File**: `colony/python/colony/vcm/manager.py`

Add two `@serving.endpoint` methods:

**`get_page_graph_groups()`** — Lists `(tenant_id, group_id, scope_id)` tuples from `VirtualPageTableState.mapped_scopes` so the frontend knows which graphs are available.

**`get_page_graph_data(tenant_id, group_id, max_nodes=5000)`** — Loads the `nx.DiGraph` from `PageStorage`, computes 3D positions with `nx.spring_layout(dim=3)`, returns JSON:
```json
{
  "nodes": [{"id": "page-1", "x": 0.3, "y": -0.5, "z": 0.1}, ...],
  "edges": [{"source": "page-1", "target": "page-2", "weight": 0.8, "confidence": 0.9, "relationship_types": ["dependency"]}, ...],
  "node_count": 150, "edge_count": 450
}
```

Key design choices:
- Server-side `spring_layout(dim=3, iterations=50)` avoids needing `d3-force-3d` client-side dependency
- `max_nodes=5000` with degree-based pruning prevents massive payloads
- Delegates to existing `PageStorage.load_page_graph()` (already cached)

### 1.2 Backend: New Router

**New file**: `colony/web_ui/backend/routers/page_graph.py`

Two routes following existing `vcm.py` pattern:
- `GET /vcm/page-graph/groups` → `colony.get_vcm().get_page_graph_groups()`
- `GET /vcm/page-graph?tenant_id=...&group_id=...&max_nodes=5000` → `colony.get_vcm().get_page_graph_data(...)`

**Modify**: `colony/web_ui/backend/main.py` — register `page_graph.router`

### 1.3 Frontend: 3D Graph Tab

**New dependencies** (package.json):
- `three` ^0.170.0
- `@react-three/fiber` ^8.17.0
- `@react-three/drei` ^9.117.0

**New types** (api/types.ts): `PageGraphGroup`, `PageGraphNode`, `PageGraphEdge`, `PageGraphData`

**New hook** (api/hooks/usePageGraph.ts): `usePageGraphGroups()`, `usePageGraph(tenantId, groupId)`

**New component** (components/graph/PageGraphTab.tsx):
```
┌────────────────────────────────────────────────────┐
│ [Group selector ▾]  Nodes: 150  Edges: 450         │
├────────────────────────────────────────────────────┤
│                                                    │
│       3D force-directed graph                      │
│       (react-three-fiber Canvas)                   │
│                                                    │
│       ● Green = loaded page                        │
│       ○ Gray = unloaded page                       │
│       — Lines = edges (opacity = weight)           │
│       Orbit/zoom/pan via OrbitControls             │
│                                                    │
├────────────────────────────────────────────────────┤
│ [Selected node detail: page_id, source, tokens,    │
│  connected edges, loaded status]                   │
└────────────────────────────────────────────────────┘
```

- **Nodes**: `<instancedMesh>` of spheres — green if loaded (cross-ref with `useLoadedPageEntries()`), gray otherwise. Size proportional to token count (from `useVCMPages()`)
- **Edges**: `<Line>` segments from `@react-three/drei`, opacity = weight × confidence
- **Camera**: `<OrbitControls>` from drei (orbit, zoom, pan)
- **Node click**: Raycasting on instancedMesh → show detail panel with page metadata + connected edges
- **Positions**: From server-side `spring_layout` (no client physics needed)

---

## Part 2: Phase 4/5 — UI Placeholders

All frontend-only. No backend/Colony changes. Hardcoded placeholder data with "Coming Soon" badges on disabled actions.

### 2.1 Interact Tab (Agent Chat + Blueprint Builder)

**New file**: `components/interact/InteractTab.tsx`

Two sub-views toggled by buttons:

**Chat sub-view**:
- Chat-like message list with placeholder messages:
  - System: "Connected to Colony cluster. 3 agents available."
  - User: "Analyze the authentication module"
  - Agent: "I'll analyze the authentication module. Loading 12 pages..."
- Agent selector dropdown (placeholder options)
- Message input + send button (disabled, "Coming Soon")

**Blueprint Builder sub-view**:
- Form: Agent Type (dropdown), Capabilities (multi-select chips), Instructions (textarea), Max Concurrent Runs (number)
- JSON preview panel showing resulting blueprint structure
- "Deploy Agent" button (disabled, "Coming Soon" badge)

### 2.2 Enhanced Blackboard (Tag Filtering)

**Modify**: `components/blackboard/BlackboardTab.tsx`

Add to existing entries section:
- Text search input above the entries table
- Clickable tag chips in entries → clicking adds tag as filter
- Active filter chips row with × to remove
- Client-side filter: entries where key/value matches text AND tags include ALL selected tags

### 2.3 Settings Tab (Remote Cluster + Plugins)

**New file**: `components/settings/SettingsTab.tsx`

Three sections:

**Remote Cluster Connection** (placeholder):
- Form: Cluster Address, Auth Token, Namespace — all disabled
- "Connect" button (disabled, "Coming Soon")
- Status: "Local Cluster" with green indicator

**Plugin System** (placeholder):
- List with toggle switches (all disabled):
  - Code Analysis, Knowledge Graph, Prometheus Metrics (enabled)
  - Custom Agent Templates, External LLM Providers (disabled)
- "Coming Soon" badge on disabled toggles

**Cluster Info** (real data):
- App name, node count from existing `useHealthStatus()` hook
- Config fields (max workers, auto-scaling) — disabled placeholder

---

## Files Summary

### New Files (5)
| File | Purpose |
|------|---------|
| `web_ui/backend/routers/page_graph.py` | Graph data API router |
| `web_ui/frontend/src/api/hooks/usePageGraph.ts` | TanStack Query hooks |
| `web_ui/frontend/src/components/graph/PageGraphTab.tsx` | 3D graph visualization |
| `web_ui/frontend/src/components/interact/InteractTab.tsx` | Chat + blueprint placeholder |
| `web_ui/frontend/src/components/settings/SettingsTab.tsx` | Remote cluster + plugins placeholder |

### Modified Files (6)
| File | Change |
|------|--------|
| `colony/python/colony/vcm/manager.py` | Add 2 `@serving.endpoint` methods |
| `web_ui/backend/main.py` | Register page_graph router |
| `web_ui/frontend/package.json` | Add three.js dependencies |
| `web_ui/frontend/src/api/types.ts` | Add graph interfaces |
| `web_ui/frontend/src/components/blackboard/BlackboardTab.tsx` | Add tag/text filtering |
| `web_ui/frontend/src/components/layout/AppShell.tsx` | Add 3 new tabs |

---

## Implementation Order

| Step | What | Files |
|------|------|-------|
| 1 | VCM graph endpoints | `colony/vcm/manager.py` |
| 2 | Backend router + registration | `backend/routers/page_graph.py` (new), `backend/main.py` |
| 3 | npm dependencies | `frontend/package.json` |
| 4 | Types + hooks | `frontend/src/api/types.ts`, `frontend/src/api/hooks/usePageGraph.ts` (new) |
| 5 | PageGraphTab component | `frontend/src/components/graph/PageGraphTab.tsx` (new) |
| 6 | BlackboardTab tag filtering | `frontend/src/components/blackboard/BlackboardTab.tsx` |
| 7 | InteractTab placeholder | `frontend/src/components/interact/InteractTab.tsx` (new) |
| 8 | SettingsTab placeholder | `frontend/src/components/settings/SettingsTab.tsx` (new) |
| 9 | Wire all tabs in AppShell | `frontend/src/components/layout/AppShell.tsx` |

Steps 6-8 are independent. Step 5 depends on 1-4. Step 9 depends on 5-8.

---

## Tab Order (Final)

```typescript
const TABS: Tab[] = [
  { id: "overview",   label: "Overview" },
  { id: "agents",     label: "Agents" },
  { id: "sessions",   label: "Sessions" },
  { id: "vcm",        label: "VCM" },
  { id: "graph",      label: "Page Graph" },
  { id: "blackboard", label: "Blackboard" },
  { id: "interact",   label: "Interact" },
  { id: "logs",       label: "Logs" },
  { id: "metrics",    label: "Metrics" },
  { id: "settings",   label: "Settings" },
];
```

---

## Verification

```bash
colony-env down && colony-env up --workers 3 && colony-env run --local-repo /home/anassar/workspace/agents/crewAI/ --config my_analysis.yaml --verbose
```

After paging, open `http://localhost:8080`:
1. **Page Graph tab**: Group selector lists graphs. 3D canvas shows nodes/edges. Orbit/zoom works. Click node shows detail.
2. **Blackboard tab**: Search input + clickable tag filters on entries.
3. **Interact tab**: Chat messages visible. Blueprint form interactive. Actions show "Coming Soon".
4. **Settings tab**: Cluster form visible (disabled). Plugin toggles visible. Real cluster info displayed.
