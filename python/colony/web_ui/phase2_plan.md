# Plan: Colony Web Dashboard — Phase 2

## Context

Phase 1 (MVP) is complete: dashboard backend with all routers using deployment handles, frontend with 5 tabs (Overview, Agents, Sessions, VCM, Metrics). All direct backend access (Redis, PostgreSQL) has been removed — everything flows through Colony deployment handles.

Phase 2 implements the 5 features requested by the user, plus fixes a critical bug in the metrics router.

## Architecture Constraint

**ALL data flows through Colony deployment handles via Ray RPC.** Dashboard NEVER accesses Colony backends directly.

---

## 2.1 Fix Token Usage Bug + By-Agent Aggregation

**Bug**: `metrics.py:48` uses `getattr(r, "resources", None)` — the actual field on `AgentRun` (`models.py:283`) is `resource_usage`. All token counts are always 0.

**Backend fix** (`colony/web_ui/backend/routers/metrics.py`):
- Change `getattr(r, "resources", None)` → `getattr(r, "resource_usage", None)`
- Add `by_agent` dict to the response: group runs by `agent_id`, sum token fields per agent

**Frontend** (`MetricsTab.tsx`):
- The pie chart already groups by `agent_id` — will show real data once bug is fixed
- Add a session filter dropdown (data from `useSessions()`)

## 2.2 VCM Page Size Histogram

**No backend changes.** `GET /vcm/pages` already returns `PageSummary.tokens` (size).

**Frontend** (`VCMTab.tsx`):
- Add recharts `BarChart` histogram below stats cards
- Bin pages by token count client-side: 1, 2-10, 11-50, 51-100, 101-500, 501-1k, 1k-5k, 5k+
- X-axis: bin label, Y-axis: page count

## 2.3 VCM Page Grid View with KV Cache Overlay

**New Colony endpoint** — `VirtualContextManager` (`colony/vcm/manager.py`):
```python
@serving.endpoint
async def list_loaded_page_entries(self) -> list[dict[str, Any]]:
```
Iterates `self.page_table.state.entries`, returns per-page: `{page_id, size, tenant_id, total_access_count, locations: [{deployment_name, client_id, access_count, last_access_time, load_time}]}`.

**Extend existing `list_stored_pages()`** — add `metadata_json` parsing to include `files` list (FileGrouper pages store file paths there). Update `list_page_summaries()` in `page_storage.py` to also select `metadata_json`, parse JSON, extract `files` key.

**Backend** (`routers/vcm.py`):
- New route `GET /vcm/loaded-pages` → `vcm.list_loaded_page_entries()`
- Update `GET /vcm/pages` response model to include optional `files: list[str]`

**Frontend** (`VCMTab.tsx`):
- Toggle: Table View | Grid View
- Grid: CSS grid of small colored squares, one per page
- Color dropdown: "Loaded status" (green/gray), "Access frequency" (heat), "Last access" (bright→dim)
- Hover tooltip: page_id, source, tokens, file paths, loaded?, access count
- New hook `useLoadedPageEntries()` in `useVCM.ts`

**Types** (`types.ts`):
- `PageSummary` — add `files?: string[]`
- New: `PageLoadedEntry { page_id, size, tenant_id, total_access_count, locations: PageLocationSummary[] }`
- New: `PageLocationSummary { deployment_name, client_id, access_count, last_access_time, load_time }`

## 2.4 Agent Hierarchy View

**No new Colony endpoints needed.** `get_agent_info()` already returns `AgentRegistrationInfo` which has `metadata.parent_agent_id`, `metadata.role`, `capability_names`.

**Backend** (`routers/agents.py`):
- New route `GET /agents/hierarchy` — calls `list_all_agents()` + `get_agent_info()` for each, returns flat list including `parent_agent_id` and `role`
- Response model: `AgentHierarchyNode { agent_id, agent_type, state, role, parent_agent_id, capability_names, bound_pages, tenant_id }`

**Frontend** (`AgentsTab.tsx`):
- Toggle: List View | Hierarchy View
- Hierarchy: tree built from `parent_agent_id` links, rendered as indented list with expand/collapse
- Each node: agent_id (truncated), role badge, state badge, capability count
- Click → existing detail panel
- New hook `useAgentHierarchy()` in `useAgents.ts`

**Types** (`types.ts`):
- New: `AgentHierarchyNode { agent_id, agent_type, state, role, parent_agent_id, capability_names: string[], bound_pages: string[], tenant_id }`

## 2.5 Blackboard Scope Observer

**Problem**: Blackboards are created per-agent at runtime. No central registry exists. Need endpoints on `AgentSystemDeployment` to discover and query them.

**Key insight**: Redis keys follow pattern `{app_name}:blackboard:{scope}:{scope_id}`. Can discover scopes by scanning keys, then create `EnhancedBlackboard` instances to get stats and entries.

**New Colony endpoints** (`colony/agents/system.py` — `AgentSystemDeployment`):

```python
@serving.endpoint
async def get_blackboard_scopes(self) -> list[dict[str, Any]]:
    """Discover active blackboard scopes via Redis key scan."""
    # 1. Get RedisClient via get_polymathera().get_redis_client()
    # 2. SCAN for keys matching "{app_name}:blackboard:*"
    # 3. Parse scope/scope_id from key patterns
    # 4. For each scope, create EnhancedBlackboard and call get_statistics()
    # Returns: [{scope, scope_id, entry_count, oldest_entry_age, newest_entry_age, backend_type}]

@serving.endpoint
async def get_blackboard_entries(
    self, scope: str, scope_id: str, limit: int = 100
) -> list[dict[str, Any]]:
    """List entries in a specific blackboard scope."""
    # Create EnhancedBlackboard(scope=scope, scope_id=scope_id), call query(limit=limit)
    # Returns: [{key, value, version, created_by, updated_at, tags}]
```

**Backend** — new router `routers/blackboard.py`:
- `GET /blackboard/scopes` → `agent_system.get_blackboard_scopes()`
- `GET /blackboard/scopes/{scope}/{scope_id}/entries?limit=100` → `agent_system.get_blackboard_entries()`

**Frontend**:
- New tab "Blackboard" in `AppShell.tsx`
- `BlackboardTab.tsx`: scope cards (name, entry count, age) → click → entry table (key, value preview, created_by, version, tags)
- `useBlackboard.ts`: `useBlackboardScopes()`, `useBlackboardEntries(scope, scopeId)`
- Types: `BlackboardScopeSummary`, `BlackboardEntryInfo`

---

## Implementation Order

| Step | What | Files Modified |
|------|------|---------------|
| 1 | Fix `resource_usage` bug | `backend/routers/metrics.py` |
| 2 | Add by-agent aggregation | `backend/routers/metrics.py` |
| 3 | Page size histogram | `frontend/.../vcm/VCMTab.tsx` |
| 4 | `list_loaded_page_entries()` endpoint | `colony/vcm/manager.py` |
| 5 | Extend `list_stored_pages()` with files | `colony/vcm/page_storage.py`, `colony/vcm/manager.py` |
| 6 | Page grid view UI | `frontend/.../vcm/VCMTab.tsx`, `frontend/src/api/hooks/useVCM.ts`, `frontend/src/api/types.ts` |
| 7 | Agent hierarchy endpoint | `backend/routers/agents.py`, `backend/models/api_models.py` |
| 8 | Agent hierarchy UI | `frontend/.../agents/AgentsTab.tsx`, `frontend/src/api/hooks/useAgents.ts`, `frontend/src/api/types.ts` |
| 9 | Blackboard Colony endpoints | `colony/agents/system.py` |
| 10 | Blackboard router | `backend/routers/blackboard.py` (new), `backend/main.py` |
| 11 | Blackboard UI | `frontend/.../blackboard/BlackboardTab.tsx` (new), `frontend/src/api/hooks/useBlackboard.ts` (new) |
| 12 | Wire up tab + router | `backend/main.py`, `frontend/.../layout/AppShell.tsx` |

## Colony Files Modified

| File | Endpoint Added |
|------|---------------|
| `colony/vcm/manager.py` | `list_loaded_page_entries()` |
| `colony/vcm/page_storage.py` | extend `list_page_summaries()` to include `files` from `metadata_json` |
| `colony/agents/system.py` | `get_blackboard_scopes()`, `get_blackboard_entries()` |

## Verification

1. `colony-env down && colony-env up --workers 3 && colony-env run ...`
2. After paging, refresh `http://localhost:8080`
3. **Metrics tab**: token bars show non-zero values; pie chart shows real per-agent data
4. **VCM tab**: histogram visible with page size distribution; grid view shows pages with color coding; hover shows file paths
5. **Agents tab**: hierarchy toggle shows tree with parent→child arrows, role badges
6. **Blackboard tab**: lists scopes with entry counts; clicking shows entries with keys/values
