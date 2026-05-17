# Tool capabilities

Colony's tool framework treats every tool as a normal
[`AgentCapability`](agent-system.md) subclass. The LLM-driven action
policy discovers tool actions the same way it discovers any other
capability action: through the agent's mounted capability set. There
is no separate `ToolRegistry`, no `ToolAdapter` ABC, no `invoke()`
shoehorn. One framework, one discovery path.

## The design principle

To fold the tool surface into the existing capability machinery:

> Every tool should appear to an agent as an `AgentCapability` with
> one or more tool-specific actions that the agent can use to interact
> with the tool. The `AgentCapability` should handle all the details
> of how to interact with the tool (e.g., through a CLI, through a
> REST API, through a Python SDK, etc.).

The `ToolSpec` metadata is a class-level
[`ClassVar`](#toolspec) on each tool capability, surfaced to the LLM
planner via the action-group description.

## Class hierarchy

```
AgentCapability                                     (colony/agents/base.py)
    │
    └── ToolCapability                              (colony/agents/patterns/capabilities/tool.py)
            │   spec: ClassVar[ToolSpec]
            │   get_capability_tags() → {"tool", ...domain}
            │   get_action_group_description() → renders spec metadata
            │   @action_executor check_preconditions()
            │
            ├── LocalToolCapability                 (colony/agents/patterns/capabilities/tool.py)
            │       In-process / cli_subprocess tools.
            │       No extra wiring; subclass adds @action_executor methods.
            │
            ├── SandboxToolCapability               (colony/agents/patterns/capabilities/tool.py)
            │       Delegates to agent-mounted SandboxedShellCapability.
            │       sandbox_image_role: ClassVar[str]
            │       _exec_in_sandbox(command, …) helper
            │       Lazy-launches a shared container; cleans up on shutdown.
            │
            └── HPCToolCapability                   (cps/src/polymathera/cps/tools/hpc/capability.py)
                    Dispatches to AWS Batch via the CPS HPC REST API.
                    _build_input_artifacts() / _parse_output_artifacts() hooks
                    _submit_hpc_job(args) helper (submit → upload → poll → fetch)
                    check_preconditions() overrides to validate vs operator limits
```

Every concrete tool subclasses one of the three intermediate bases,
declares its `spec`, and ships one or more `@action_executor` methods
named for what the tool **does** (`run_em_fdtd`,
`compute_shielding_factor`, `search_knowledge`). The LLM planner sees
them in the normal action menu — filtered by the canonical `"tool"`
tag when the planner asks for "what tools do I have?".

## `ToolSpec`

Lives at [`colony/tools/spec.py`](../../src/polymathera/colony/tools/spec.py).
Frozen Pydantic model carrying:

- `name`, `version`, `domain`, `backend`, `capabilities` (tuple of
  capability keys), `inputs_schema`, `outputs_schema` — identity +
  documentation.
- `determinism`, `cost_model`, `licensing`, `licensing_notes`,
  `headless`, `hitl_frequency`, `interruptibility` — Appendix-C / -D
  metadata the planner reasons about.
- `execution_locality` (`LOCAL` / `HPC` / `CUSTOMER_SITE`) and
  `resource_requirements` (`min_vcpus`, `min_memory_gb`, optional
  `GpuRequirement`, `expected_wallclock_seconds`) — the dispatch
  shape.
- `references`, `extra` — free-form.

`ToolCapability` enforces `spec` declaration at subclass creation via
`__init_subclass__`. A subclass without one fails at import time, not
at runtime.

## `get_action_group_description` — the LLM-visible card

Per the convention shared by every `AgentCapability`, the action
dispatcher surfaces `get_action_group_description()` to the planner
alongside each action's docstring. The `ToolCapability` base
overrides it to fold the spec metadata into the description string —
the planner sees cost, resource requirements, HITL tier, licence,
and locality inline with the action menu. No separate `describe_tool`
action is needed; the metadata is part of the action's planner-time
context.

`SandboxToolCapability` and `HPCToolCapability` extend the
description via the `_describe_tool_extras` hook (the shared image
role / "runs on AWS Batch" notes).

## `check_preconditions` — the standard preflight action

Every tool capability inherits a `check_preconditions`
`@action_executor` that returns a structured snapshot of the tool's
operating state:

```json
{
  "tool": "openems_fdtd",
  "execution_locality": "hpc",
  "resource_requirements": {"min_vcpus": 16, "min_memory_gb": 64.0, ...},
  "ok": true,
  "warnings": []
}
```

`HPCToolCapability` overrides it to additionally validate the spec's
`ResourceRequirements` against the operator's `cps.hpc.limits` and
warn on each cap violation. The planner reads `ok` + `warnings` to
decide whether to dispatch the call or ask for operator approval.

## The `"tool"` tag

`ToolCapability.get_capability_tags()` always merges the canonical
`"tool"` tag into the subclass's tag set. The action dispatcher
already supports tag-based filtering of the LLM's action menu via
`get_action_descriptions(include_tags=..., exclude_tags=...)`. A
planner that wants "just the tools" calls
`get_action_descriptions(include_tags={"tool"})`; a planner that
wants "the HPC tools only" calls
`get_action_descriptions(include_tags={"tool", "hpc"})`.

Subclasses add domain tags via `_domain_tags()`:

```python
class OpenEMSFdtdCapability(HPCToolCapability):
    spec = ToolSpec(...)
    def _domain_tags(self) -> frozenset[str]:
        return frozenset({"em", "fdtd"})
```

## Discovery flow

1. **Catalog.** The L4 design monorepo's
   `.colony/tool-registry.json` lists each available tool as a
   [`ToolEntry`](../../src/polymathera/colony/design_monorepo/models.py):
   `name`, `purpose`, `location`, `capability` (search key, denormalised
   from `spec.capabilities[0]`), `capability_fqn` (the import path of
   the implementing `ToolCapability` subclass), `extra`.
2. **L1-A discovery.** `discover_tools(repo_root)` reads the catalog
   via `load_registry` and returns `dict[str, ToolEntry]` on
   `DiscoveredExtensions.tools`. No file imports happen at discovery
   time — the FQN is resolved later at mount time.
3. **Session refresh.** `SessionOrchestratorCapability._refresh_available_tools`
   projects the catalog into the planner-visible
   `agent.metadata.parameters["available_tools"]`. Entries with an
   empty `capability_fqn` are catalog-only stubs (build-vs-buy
   candidates) and are omitted from the planner-visible dict.
4. **Mount.** When `spawn_mission` (or `AgentPoolCapability.create_agent`)
   includes a tool's `capability_fqn` in the new agent's `capabilities=[...]`
   list, `class_resolver.resolve_class` imports the class; the
   blueprint constructor binds it to the agent; the dispatcher
   registers its actions.

The catalog read at step 2 is `O(JSON file size)`; the class
import happens only at mount time. The L4 author updates a single
JSON file to make a new tool discoverable; no parallel registration
ceremony.

## Catalog ↔ spec invariant

`ToolEntry.capability` is a denormalised cache of
`ToolCapability.spec.capabilities[0]` — kept on the catalog so the
search index in [`design_monorepo/registry.py`](../../src/polymathera/colony/design_monorepo/registry.py)
doesn't have to import every capability class to find a match. The
denormalisation is policed at registration time: `upsert_tool`
imports `entry.capability_fqn`, reads `cls.spec.capabilities`, and
raises `ToolEntrySpecMismatch` if `entry.capability` isn't in there.
The on-disk index cannot drift silently from the live spec.

Entries with an empty `capability_fqn` are exempt from the
validation — they're catalog-only stubs for the build-vs-buy
advisor.

## Sample subclasses

### `LocalToolCapability` — in-process tool

```python
from polymathera.colony.agents.patterns.actions import action_executor
from polymathera.colony.agents.patterns.capabilities.tool import LocalToolCapability
from polymathera.colony.tools import (
    CostModel, HITLFrequency, HeadlessReadiness, Licensing,
    ResourceRequirements, ToolSpec,
)


class MagerSumnerShieldingCapability(LocalToolCapability):
    spec = ToolSpec(
        name="mager_sumner_shielding",
        domain="em",
        capabilities=("compute_shielding_factor",),
        backend="in_process",
        cost_model=CostModel(cpu_seconds=0.1),
        resource_requirements=ResourceRequirements(min_vcpus=1, min_memory_gb=0.5),
        headless=HeadlessReadiness.NATIVE,
        hitl_frequency=HITLFrequency.AUTONOMOUS,
        licensing=Licensing.MIT,
    )

    def _domain_tags(self) -> frozenset[str]:
        return frozenset({"em", "magnetic_shielding"})

    @action_executor()
    async def compute_shielding_factor(
        self, *,
        layers_mm: list[float],
        relative_permeability: list[float],
        inner_radius_mm: float,
    ) -> dict[str, float]:
        ...
```

### `SandboxToolCapability` — runs inside a Docker container

```python
class FemmShieldingCapability(SandboxToolCapability):
    spec = ToolSpec(name="femm_2d_shielding", ...)
    sandbox_image_role = "femm"

    @action_executor()
    async def run_femm_2d(self, *, geometry_step: str) -> dict[str, Any]:
        return await self._exec_in_sandbox(
            ["femm-batch", "/mnt/inputs/geometry.step"],
            timeout_seconds=600,
        )
```

The sandbox image is launched lazily on the first action call and
reused across subsequent calls in the session; `shutdown` stops it.

### `HPCToolCapability` — dispatches to AWS Batch

```python
class OpenEMSFdtdCapability(HPCToolCapability):
    spec = ToolSpec(
        name="openems_fdtd",
        capabilities=("run_em_fdtd",),
        backend="http_api",
        execution_locality=ExecutionLocality.HPC,
        resource_requirements=ResourceRequirements(
            min_vcpus=8, min_memory_gb=64,
            gpu=GpuRequirement(kind="a100", count=1, memory_gb=40),
            expected_wallclock_seconds=21600,
        ),
    )

    def _hpc_domain_tags(self) -> frozenset[str]:
        return frozenset({"em", "fdtd"})

    def _build_input_artifacts(self, call_args):
        deck = self._render_simulation_deck(call_args)
        return ((
            InputArtifactDescriptor(key="sim.xml", size_bytes=len(deck)),
            deck.encode(),
            "application/xml",
        ),)

    def _parse_output_artifacts(self, artifacts):
        return parse_openems_h5(artifacts["fields.h5"])

    @action_executor()
    async def run_em_fdtd(
        self, *,
        geometry_step: str,
        excitation_freq_hz: float,
        mesh_resolution_m: float = 1e-3,
    ) -> dict[str, Any]:
        result = await self._submit_hpc_job(
            {"freq_hz": excitation_freq_hz, "mesh_resolution": mesh_resolution_m},
        )
        return result
```

The `HPCClient` is auto-built from the operator's `cps.hpc.endpoint`
config via a process-wide `functools.cache`-keyed singleton; tests
inject a fake via the constructor's `client=...` kwarg.

## Boundary with the existing `SandboxedShellCapability`

`SandboxedShellCapability` is the agent-facing container-execution
surface. `SandboxToolCapability` is a tool-author convenience that
wraps it — every sandboxed tool capability delegates container
lifecycle through the agent's mounted `SandboxedShellCapability`.
The shared container is launched lazily on first call, reused, and
stopped at `shutdown`. There is one image-routing path, owned by
`SandboxedShellCapability`'s `SandboxImagesConfig`.

`HPCToolCapability` delegates the equivalent to AWS Batch; the
JobDefinition image is owned by the [CDK
stack](../guides/operations/aws-hpc-deployment.md) (CPS-side), not
by colony.

## Why not `invoke()`

The pre-retrofit `ToolAdapter.invoke(call: ToolCall) -> ToolResult`
shoehorn forces every tool through one synthetic signature. The LLM
planner can't reason about a tool's actual inputs / outputs from a
generic `parameters: dict` — it sees only "call this tool, somehow".

With each tool's surface as concrete `@action_executor` methods, the
planner sees real typed signatures, real docstrings, real return
shapes. The dispatcher's action-description rendering already knows
how to surface them. The `ToolCapability` base layers metadata
(spec, tags, preconditions) on top of that proven surface.

## Related

- [`guides/operations/aws-hpc-deployment.md`](../guides/operations/aws-hpc-deployment.md)
  — operator-side AWS Batch + REST stack.
- [`design-monorepo-extensions.md`](design-monorepo-extensions.md) —
  L1-A `.colony/*` discovery (incl. `discover_tools`).
- [`agent-system.md`](agent-system.md) — `AgentCapability` lifecycle.
- [`STAGE_B_TOOL_FRAMEWORK_RETROFIT_PLAN.md`](../../STAGE_B_TOOL_FRAMEWORK_RETROFIT_PLAN.md)
  — the retrofit plan + 11-checkpoint history.
