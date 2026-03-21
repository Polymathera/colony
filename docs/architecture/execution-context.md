# Execution Context

Colony is a shared-nothing distributed system where multiple tenants run workloads on the same cluster. The **execution context** is the ambient credential that every operation carries, answering three questions: *who* is running, *what scope* they're in, and *what privilege* they have.

!!! tip "OS analogy"
    Think of the execution context like a CPU's privilege ring register combined with a process's UID/GID. Every instruction (`@serving.endpoint` call in Colony) runs at a specific ring level, and the system enforces access control based on that level.

## The `ExecutionContext` Object

```python
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring, ExecutionContext, execution_context,
)

# User-mode context (tenant-scoped)
with execution_context(
    ring=Ring.USER,
    colony_id="acme-monorepo",
    tenant_id="acme-corp",
    session_id="sess-abc123",
    run_id="run-def456",
    origin="cli",
):
    result = await handle.start_analysis(config)

# Kernel-mode context (infrastructure)
with execution_context(ring=Ring.KERNEL, origin="vcm_reconciler"):
    names = await handle.get_all_deployment_names()
```

The context is a **frozen dataclass** — immutable once created. You don't mutate it; you create a new one for a new scope (e.g., when a kernel task iterates over tenants).

| Field | Type | Description |
|-------|------|-------------|
| `ring` | `Ring` | Privilege level: `KERNEL` (0) or `USER` (3) |
| `colony_id` | `str \| None` | Colony instance (required for `Ring.USER`) |
| `tenant_id` | `str \| None` | Organization/tenant (required for `Ring.USER`) |
| `session_id` | `str \| None` | User session (optional) |
| `run_id` | `str \| None` | Analysis run (optional) |
| `trace_id` | `str \| None` | Distributed tracing ID (optional) |
| `origin` | `str \| None` | What created this context (audit) |

---

## Privilege Rings

<style>
/* ── Execution Context diagrams ── */
.ec-svg text { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif; }
.ec-svg .r-kernel  { fill: #eff6ff; stroke: #3b82f6; }
.ec-svg .r-user    { fill: #f5f3ff; stroke: #8b5cf6; }
.ec-svg .r-outer   { fill: #f9fafb; stroke: #9ca3af; }
.ec-svg .r-req     { fill: #ecfdf5; stroke: #10b981; }
.ec-svg .r-ray     { fill: #fffbeb; stroke: #f59e0b; }
.ec-svg .t-title   { fill: #1e1b4b; }
.ec-svg .t-body    { fill: #374151; }
.ec-svg .t-muted   { fill: #6b7280; }
.ec-svg .t-kernel  { fill: #1e40af; }
.ec-svg .t-user    { fill: #5b21b6; }
.ec-svg .t-green   { fill: #064e3b; }
.ec-svg .t-amber   { fill: #92400e; }
[data-md-color-scheme="slate"] .ec-svg .r-kernel  { fill: #172554; stroke: #2563eb; }
[data-md-color-scheme="slate"] .ec-svg .r-user    { fill: #2e1065; stroke: #7c3aed; }
[data-md-color-scheme="slate"] .ec-svg .r-outer   { fill: #1f2937; stroke: #6b7280; }
[data-md-color-scheme="slate"] .ec-svg .r-req     { fill: #052e16; stroke: #059669; }
[data-md-color-scheme="slate"] .ec-svg .r-ray     { fill: #451a03; stroke: #d97706; }
[data-md-color-scheme="slate"] .ec-svg .t-title   { fill: #c4b5fd; }
[data-md-color-scheme="slate"] .ec-svg .t-body    { fill: #d1d5db; }
[data-md-color-scheme="slate"] .ec-svg .t-muted   { fill: #9ca3af; }
[data-md-color-scheme="slate"] .ec-svg .t-kernel  { fill: #93c5fd; }
[data-md-color-scheme="slate"] .ec-svg .t-user    { fill: #c4b5fd; }
[data-md-color-scheme="slate"] .ec-svg .t-green   { fill: #6ee7b7; }
[data-md-color-scheme="slate"] .ec-svg .t-amber   { fill: #fcd34d; }
</style>

<div style="margin:1.5rem 0;">
<svg class="ec-svg" viewBox="0 0 560 280" xmlns="http://www.w3.org/2000/svg">
  <!-- Outer ring label -->
  <rect class="r-outer" x="10" y="10" width="540" height="260" rx="16" stroke-width="2"/>
  <text class="t-muted" x="280" y="36" text-anchor="middle" font-size="12" font-weight="600">PRIVILEGE RINGS</text>

  <!-- Ring 3: USER -->
  <rect class="r-user" x="40" y="50" width="480" height="200" rx="12" stroke-width="2"/>
  <text class="t-user" x="60" y="72" font-size="11" font-weight="700">RING 3 &mdash; USER</text>
  <text class="t-muted" x="60" y="88" font-size="10">Requires colony_id + tenant_id</text>

  <!-- Ring 0: KERNEL (nested inside) -->
  <rect class="r-kernel" x="80" y="100" width="400" height="130" rx="10" stroke-width="2"/>
  <text class="t-kernel" x="100" y="122" font-size="11" font-weight="700">RING 0 &mdash; KERNEL</text>
  <text class="t-muted" x="100" y="138" font-size="10">No tenant context required</text>

  <!-- Kernel endpoints -->
  <rect class="r-req" x="100" y="150" width="160" height="32" rx="6" stroke-width="1.2"/>
  <text class="t-green" x="180" y="170" text-anchor="middle" font-size="10" font-weight="600">get_deployment_names()</text>
  <rect class="r-req" x="280" y="150" width="180" height="32" rx="6" stroke-width="1.2"/>
  <text class="t-green" x="370" y="170" text-anchor="middle" font-size="10" font-weight="600">reconcile_page_state()</text>
  <rect class="r-req" x="100" y="192" width="160" height="32" rx="6" stroke-width="1.2"/>
  <text class="t-green" x="180" y="212" text-anchor="middle" font-size="10" font-weight="600">health_check()</text>
  <rect class="r-req" x="280" y="192" width="180" height="32" rx="6" stroke-width="1.2"/>
  <text class="t-green" x="370" y="212" text-anchor="middle" font-size="10" font-weight="600">get_replica_count()</text>

  <!-- Syscall arrow -->
  <line x1="520" y1="85" x2="520" y2="165" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#ec-arrow)"/>
  <text class="t-muted" x="526" y="128" font-size="9" font-style="italic">syscall</text>

  <defs>
    <marker id="ec-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="#7c3aed"/>
    </marker>
  </defs>
</svg>
</div>

Colony uses two privilege rings, modeled after CPU protection rings:

| Ring | Value | Tenant context | Use case |
|------|-------|---------------|----------|
| **KERNEL** | 0 | Not required | VCM reconciliation, health checks, autoscaling, deployment management |
| **USER** | 3 | Required (`colony_id` + `tenant_id`) | Agent execution, analysis runs, session management |

### Calling Rules

| Caller | Target | Allowed? |
|--------|--------|----------|
| USER | USER | Yes (tenant context validated) |
| USER | KERNEL | Yes (syscall pattern) |
| KERNEL | KERNEL | Yes |
| KERNEL | USER | **No** &mdash; must enter USER context first |
| No context | Any | **Error** |

!!! note "Syscall pattern"
    User-mode code can call kernel-mode endpoints freely &mdash; just like user processes making system calls. The kernel endpoint simply doesn't validate tenant fields. The execution context is still propagated for audit and tracing.

---

## Context Propagation

Python `contextvars` don't survive Ray `.remote()` calls. The serving framework bridges this gap by serializing the `ExecutionContext` into `DeploymentRequest` and restoring it on the other side.

<div style="margin:1.5rem 0;">
<svg class="ec-svg" viewBox="0 0 700 240" xmlns="http://www.w3.org/2000/svg">
  <!-- Caller process -->
  <rect class="r-user" x="10" y="30" width="180" height="180" rx="10" stroke-width="1.5"/>
  <text class="t-user" x="100" y="52" text-anchor="middle" font-size="11" font-weight="700">Caller Process</text>
  <text class="t-body" x="100" y="80" text-anchor="middle" font-size="10">contextvars:</text>
  <text class="t-body" x="100" y="95" text-anchor="middle" font-size="10" font-weight="600">ExecutionContext</text>
  <rect class="r-req" x="30" y="110" width="140" height="28" rx="5" stroke-width="1"/>
  <text class="t-green" x="100" y="129" text-anchor="middle" font-size="9" font-weight="600">[A] Capture into request</text>
  <text class="t-muted" x="100" y="162" text-anchor="middle" font-size="9">DeploymentHandle</text>
  <text class="t-muted" x="100" y="175" text-anchor="middle" font-size="9">reads ctx, puts in</text>
  <text class="t-muted" x="100" y="188" text-anchor="middle" font-size="9">DeploymentRequest</text>

  <!-- Ray boundary 1 -->
  <rect class="r-ray" x="205" y="90" width="60" height="60" rx="6" stroke-width="1.5" stroke-dasharray="4 2"/>
  <text class="t-amber" x="235" y="118" text-anchor="middle" font-size="8" font-weight="700">Ray</text>
  <text class="t-amber" x="235" y="130" text-anchor="middle" font-size="8">.remote()</text>

  <!-- Proxy actor -->
  <rect class="r-kernel" x="280" y="30" width="150" height="180" rx="10" stroke-width="1.5"/>
  <text class="t-kernel" x="355" y="52" text-anchor="middle" font-size="11" font-weight="700">Proxy Actor</text>
  <rect class="r-req" x="295" y="75" width="120" height="28" rx="5" stroke-width="1"/>
  <text class="t-green" x="355" y="94" text-anchor="middle" font-size="9" font-weight="600">[B] Restore context</text>
  <text class="t-muted" x="355" y="120" text-anchor="middle" font-size="9">Routes request</text>
  <text class="t-muted" x="355" y="133" text-anchor="middle" font-size="9">to replica queue</text>
  <text class="t-muted" x="355" y="175" text-anchor="middle" font-size="9">contextvars fresh</text>
  <text class="t-muted" x="355" y="188" text-anchor="middle" font-size="9">(restored from req)</text>

  <!-- Ray boundary 2 -->
  <rect class="r-ray" x="445" y="90" width="60" height="60" rx="6" stroke-width="1.5" stroke-dasharray="4 2"/>
  <text class="t-amber" x="475" y="118" text-anchor="middle" font-size="8" font-weight="700">Ray</text>
  <text class="t-amber" x="475" y="130" text-anchor="middle" font-size="8">.remote()</text>

  <!-- Replica actor -->
  <rect class="r-user" x="520" y="30" width="170" height="180" rx="10" stroke-width="1.5"/>
  <text class="t-user" x="605" y="52" text-anchor="middle" font-size="11" font-weight="700">Replica Actor</text>
  <rect class="r-req" x="535" y="75" width="140" height="28" rx="5" stroke-width="1"/>
  <text class="t-green" x="605" y="94" text-anchor="middle" font-size="9" font-weight="600">[C] Restore + enforce</text>
  <text class="t-muted" x="605" y="120" text-anchor="middle" font-size="9">Validates ring vs</text>
  <text class="t-muted" x="605" y="133" text-anchor="middle" font-size="9">endpoint requirement</text>
  <text class="t-muted" x="605" y="155" text-anchor="middle" font-size="9">Then calls</text>
  <text class="t-body" x="605" y="170" text-anchor="middle" font-size="10" font-weight="600">method(*args)</text>
  <text class="t-muted" x="605" y="188" text-anchor="middle" font-size="9">Resets on exit</text>

  <!-- Arrows -->
  <line x1="190" y1="120" x2="205" y2="120" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#ec-arrow)"/>
  <line x1="265" y1="120" x2="280" y2="120" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#ec-arrow)"/>
  <line x1="430" y1="120" x2="445" y2="120" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#ec-arrow)"/>
  <line x1="505" y1="120" x2="520" y2="120" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#ec-arrow)"/>

  <!-- Title -->
  <text class="t-title" x="350" y="18" text-anchor="middle" font-size="13" font-weight="700">Request Lifecycle: Context Propagation</text>
</svg>
</div>

### Injection Points

| Point | Location | What happens |
|-------|----------|-------------|
| **A** | `DeploymentHandle.__getattr__` | Reads `ExecutionContext` from contextvars, serializes into `DeploymentRequest` |
| **B** | `proxy.handle_request` | Restores `ExecutionContext` from request (for proxy-local logging/metrics) |
| **C** | `__handle_request__` on replica | Restores context, validates caller ring vs endpoint ring, calls method |

### Why `asyncio.create_task` Works

Python's `asyncio.create_task` copies the current `contextvars.Context` at task creation time ([PEP 567](https://peps.python.org/pep-0567/)). So when `start_agent` (running inside an `ExecutionContext`) does `asyncio.create_task(self._run_agent_loop(...))`, the agent loop inherits the full context &mdash; including ring, tenant, and session &mdash; for its entire lifetime.

---

## Endpoint Ring Declaration

Every `@serving.endpoint` declares its ring level:

```python
from polymathera.colony.distributed.ray_utils.serving import context as ctx

@serving.endpoint(ring=ctx.Ring.KERNEL)
async def get_all_deployment_names(self) -> list[str]:
    """Infrastructure endpoint -- no tenant context needed."""
    ...

@serving.endpoint  # defaults to Ring.USER
async def start_agent(self, blueprint: AgentBlueprint) -> str:
    """Tenant-scoped endpoint -- requires colony_id + tenant_id."""
    ...
```

The ring is stored as `func.__endpoint_ring__` by the decorator and collected into `self._endpoint_rings` during replica initialization. The `__handle_request__` method reads it to enforce access control before calling the actual method.

---

## Background Tasks

Background tasks that run outside `@serving.endpoint` (periodic reconciliation, health checks, resource monitors) must set their own execution context. These are kernel-mode by definition:

```python
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring, execution_context,
)

async def _periodic_reconciliation_loop(self):
    while True:
        await asyncio.sleep(30)
        with execution_context(ring=Ring.KERNEL, origin="vcm_reconciler"):
            await self._reconcile_page_state()
```

### `@periodic_health_check` Methods

Periodic health checks are routed through `__handle_request__` with a kernel-mode `ExecutionContext`. This means the health check method runs with proper context and can make downstream deployment calls without crashing.

---

## Ring Transitions

### USER to KERNEL (automatic)

When user-mode code calls a kernel endpoint, the existing `ExecutionContext` is propagated as-is. The kernel endpoint simply doesn't validate tenant fields. No explicit transition needed.

### KERNEL to USER (explicit)

A kernel task that needs to operate per-tenant must explicitly create a USER context:

```python
# Inside a kernel-mode background task
with execution_context(ring=Ring.KERNEL, origin="garbage_collector"):
    tenants = await admin_handle.list_all_tenants()
    for tenant in tenants:
        with execution_context(
            ring=Ring.USER,
            colony_id=tenant.colony_id,
            tenant_id=tenant.tenant_id,
            origin="garbage_collector:per_tenant",
        ):
            await handle.cleanup_expired_caches()
```

The `execution_context` manager uses token-based reset, so nesting works correctly &mdash; the inner context fully replaces the outer one, and the outer is restored on exit.

---

## Reading Context in Downstream Code

Any code running inside an execution context can read it:

```python
from polymathera.colony.distributed.ray_utils import serving

# Full context object
ctx = serving.require_execution_context()
print(ctx.ring, ctx.colony_id, ctx.tenant_id)

# Shorthand accessors
colony_id = serving.get_colony_id()
tenant_id = serving.get_tenant_id()
session_id = serving.get_session_id()

# Strict accessors (raise if None)
colony_id = serving.require_colony_id()
tenant_id = serving.require_tenant_id()
```

### The `check_isolation` Decorator

Agents use `@check_isolation` to verify that the ambient execution context matches the agent's identity &mdash; catching context leaks or cross-tenant contamination:

```python
@check_isolation
async def run_step(self) -> None:
    # This will raise RuntimeError if:
    # - ctx.tenant_id != self.tenant_id
    # - ctx.colony_id != self.colony_id
    ...
```
