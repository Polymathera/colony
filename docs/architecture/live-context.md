# Live Context — Watchers + Convergence Runtime

This page documents the **always-live design context** machinery (master design doc §5): the chain that lets a change in an upstream context source (e.g., a git repository) ripple through subscribed `AgentCapabilities` until the VCM contents (e.g., a design and its associated context such as requirements, specifications, codebases, etc.) eventually converge to a quiescent state.

The chain is built from four layers, each in its own colony package:

| Layer | Package | Role |
|-------|---------|------|
| 1. Source watchers | [polymathera.colony.vcm.watchers](../../src/polymathera/colony/vcm/watchers/) | Detect upstream mutations and emit `PageChangeEvent`s |
| 2. VCM watch bridge | [polymathera.colony.vcm.manager.VirtualContextManager._start_watch_bridge](../../src/polymathera/colony/vcm/manager.py) | Drains each non-static source's `watch()` and feeds events directly into the runtime via the deployment handle (KERNEL ring) |
| 3. Convergence runtime | [polymathera.colony.vcm.convergence](../../src/polymathera/colony/vcm/convergence/) | Subscriptions, topological dispatch, quiescence detection, cycle break, rate limit, damping |
| 4. Agent surface | [polymathera.colony.agents.patterns.capabilities.ConvergenceCapability](../../src/polymathera/colony/agents/patterns/capabilities/convergence.py) | `subscribe_pattern`, `unsubscribe`, `dispatch_change`, `wait_for_quiescence`, `get_convergence_status`, `get_change_feed`, `detect_cycle` |

All four layers are implemented and wired end-to-end at cluster boot. The chain runs whenever:

1. `VCMConfig.add_deployments_to_app` registers `ConvergenceRuntimeDeployment` (it always does, unconditionally).
2. The VCM materializes a scope mapping for any non-static `ContextPageSource` — `_start_watch_bridge` drains the source's `watch()` iterator and calls `ConvergenceRuntimeDeployment.feed_page_event` directly via its deployment handle. There is exactly **one** chain per source: `GitRepoContextPageSource.watch()` runs both a `LocalFsWatcher` (for in-tree edits) and a `GitRemoteWatcher` (for upstream commits) and merges them; `BlackboardContextPageSource.watch()` drains the source's own record-event loop.

Layer 1 (HTTP webhook receiver) is the only piece deferred — the `WebhookEventBuilder` translator exists; the route lives in Phase C6.

End-to-end smoke tests live in [`vcm/convergence/tests/test_chain_smoke.py`](../../src/polymathera/colony/vcm/convergence/tests/test_chain_smoke.py) — they cover `LocalFsWatcher`, `SourcePollWatcher`, and the formal `ContextPageSource.watch()` contract.

---

## 1. The `ContextPageSource.watch()` contract

`ContextPageSource` ([context_page_source.py](../../src/polymathera/colony/vcm/sources/context_page_source.py)) carries a `watch()` method on the base ABC plus a class-level `static: ClassVar[bool] = True` flag. The default `watch()` raises `NotImplementedError`. Per the docstring:

> Sources that can detect upstream changes and translate them into page-graph mutations declare `static = False` and override `watch()`. Sources that cannot — a one-shot static dump, an archived corpus snapshot — leave `static = True`; the convergence runtime refuses to subscribe to a static source as a live page-graph input and instead relies on whatever bulk re-ingestion path the source provides.

### Current state

| Source | `static` | `watch()` shape | Live-event production |
|--------|----------|----------------|-----------------------|
| [`GitRepoContextPageSource`](../../src/polymathera/colony/samples/paging/git_repo_page_source.py) | `False` | Merges a `LocalFsWatcher` (in-tree edits) and a `GitRemoteWatcher` (upstream commits) rooted at the cloned working tree, into one async iterator | Filesystem events on tracked files (those in `file_to_page`) become `PageChangeEvent`s with `data_type="design_monorepo_file"` and the affected page's id |
| [`BlackboardContextPageSource`](../../src/polymathera/colony/agents/blackboard/paging/blackboard_page_source.py) | `False` | Yields from an internal `asyncio.Queue` populated by the source's record-event loop | Each blackboard write/update/delete diffs the page graph; new pages emit `PageAdded`, retired pages emit `PageInvalidated`, page reassignments emit `PageReplaced` |

The two shapes differ but the *contract* is uniform: every non-static source exposes mutations as an `AsyncIterator[PageChangeEvent]`. The runtime side does not know or care whether the iterator delegates to a sidecar watcher (`LocalFsWatcher`) or pulls from the source's own internal event loop.

#### Limitations to know about

- **`GitRepoContextPageSource` only watches files in `file_to_page` at the time `watch()` starts.** A file added to the working tree after the page graph was built fires no events; detecting it requires a graph rebuild, which is a separate pass.
- **Multi-replica VCMs duplicate events.** Every replica that materializes a non-static scope mapping starts its own watcher; events get N-fold duplicated into the convergence runtime. The runtime's per-page rate-limiter absorbs transient bursts, but a leader-election story is the durable fix. The same caveat applies to `BlackboardContextPageSource`.

### The bridge: `VirtualContextManager._start_watch_bridge`

When the VCM materializes a scope mapping, it inspects `source.static`. If the source is non-static, the VCM starts a long-running task with the canonical shape:

```python
async for event in source.watch():
    await convergence_runtime.feed_page_event(event=event, source_id=source.scope_id)
```

The convergence runtime handle is resolved once per VCM replica via `get_convergence_runtime(self.app_name)` and reused across all bridges. `feed_page_event` is a `Ring.KERNEL` endpoint — it is the privileged ingestion path between sibling deployments and bypasses any blackboard mediation. The bridge task's lifetime is owned by the `MappedScope` — `_shutdown_mapped_scope` cancels it on unmap.

This means: any future `ContextPageSource` subclass that sets `static = False` and implements `watch()` is automatically wired into the convergence chain. No special hook on the source, no new code in the bridge, no additional registration call.

---

## 2. Watcher transports (master §5.6)

Three transport classes plus a webhook payload translator, all in [vcm/watchers/](../../src/polymathera/colony/vcm/watchers/):

- **`LocalFsWatcher`** — `watchdog`-based filesystem watcher with a debounce window; falls back to mtime-poll when `watchdog` is unavailable.
- **`GitRemoteWatcher`** — periodic `git fetch` + `git diff --name-only` against a local clone; covers remote-driven push fallback when no webhook is available.
- **`SourcePollWatcher`** — generic interval poll over any `ContextPageSource`'s `get_all_mapped_pages()` snapshot; the catch-all transport for sources behind APIs (arXiv RSS, supplier catalogues) with no push notification.
- **`WebhookEventBuilder`** — translates a Gitea / GitLab / GitHub git-push webhook payload into a sequence of `PageChangeEvent`s. (HTTP receiver is a Phase C6 concern; the translator lives here so the watcher contract stays in one place.)
- **`CompositeWatcher`** — merges N child watchers into one async iterator. Used when a single source needs more than one watch transport against the same backing store (e.g., `GitRepoContextPageSource` couples a `LocalFsWatcher` and a `GitRemoteWatcher` against the cloned working tree).

All set `static = False` and emit `PageChangeEvent`s. Watchers are **not** subclasses of `ContextPageSource` — they are sidecar classes that operate alongside a source. This decouples watch lifecycle from page-source lifecycle and lets multiple watchers cover one source (LocalFs *and* GitRemote on the same working tree, for example).

### Bridge: `GitRepoContextPageSource.watch`

The page source itself is the bridge. When a working tree is mapped into the VCM as a `GitRepoContextPageSource` (via `mmap_application_scope` with `source_type="codebase"`), the VCM's `_start_watch_bridge` drains the source's `watch()` iterator and feeds each event into the runtime. `watch()` runs both watchers (LocalFs + GitRemote) inside the source itself — via `CompositeWatcher` — and merges them into one stream. There is no separate registration call from any capability.

There used to be a parallel `DesignMonorepoWatcher` registered through `ConvergenceRuntimeDeployment.register_design_monorepo`; that produced duplicate watchers when the same working tree was both registered AND mapped. It was removed — the sole place a working tree's filesystem + remote are watched is `GitRepoContextPageSource.watch()`. See `colony_docs/markdown/convergence_flow_review.md` §P0 for the rationale.

---

## 3. The convergence runtime

[`vcm/convergence/`](../../src/polymathera/colony/vcm/convergence/) implements the dispatch loop the master doc §5.2 describes:

```
PageChangeEvent  ─┐
                  │  VCM watch bridge → ConvergenceRuntimeDeployment.feed_page_event
                  ▼
       ConvergenceRuntime.feed_event
                  │
                  ▼
         SubscriptionIndex.match    ─── PageMetadataPredicate, EdgeReachResolver
                  │
                  ▼
        WriteRateLimiter (per page)
                  │
                  ▼
         ConvergenceDamper.skip?     ─── numeric tolerance check
                  │
                  ▼
        dispatch_callback(sub, event)
                  │
                  ▼
        EnhancedBlackboard.write(
            scope=sub.dispatch_scope,
            key=ConvergenceDispatchProtocol.dispatch_key(sub.subscription_id),
        )
                  │
                  ▼
   ConvergenceCapability's @event_handler(pattern=
        ConvergenceDispatchProtocol.dispatch_pattern()) picks it up
```

Module map:

- [`runtime.py`](../../src/polymathera/colony/vcm/convergence/runtime.py) — `ConvergenceRuntime` (pure dispatch logic), `ConvergenceState`, `ConvergenceStatus`, `ConvergenceCounters`, `ChangeFeedEntry`. Tracks the current episode, applies damping, rate-limits per-page writes, enforces a per-episode budget, detects quiescence.
- [`subscriptions.py`](../../src/polymathera/colony/vcm/convergence/subscriptions.py) — `PageSubscription`, `NumericTolerance`. The dispatch *key* is derived from `ConvergenceDispatchProtocol` + the subscription_id; callers do not pick it.
- [`predicates.py`](../../src/polymathera/colony/vcm/convergence/predicates.py) — `PageMetadataPredicate` (typed match expression over page metadata) + `EdgeReachResolver` for graph-aware predicates.
- [`index.py`](../../src/polymathera/colony/vcm/convergence/index.py) — `SubscriptionIndex` (fast lookup by event metadata).
- [`damping.py`](../../src/polymathera/colony/vcm/convergence/damping.py) — numeric-tolerance check that suppresses dispatches inside a configured tolerance.
- [`rate_limit.py`](../../src/polymathera/colony/vcm/convergence/rate_limit.py) — `WriteRateLimiter` (per-page write throttle).
- [`deployment.py`](../../src/polymathera/colony/vcm/convergence/deployment.py) — `ConvergenceRuntimeDeployment`, the Ray-serving singleton wrapping the runtime; exposes `feed_page_event` (KERNEL ring, called by VCM's watch bridge), `subscribe`/`unsubscribe`, the read-side polling surfaces (`get_status`, `get_change_feed`, `wait_for_quiescence`), and emits `ConvergenceQuiescenceProtocol` events on the colony scope.

### Mechanism scope (what's in v1, what's not)

| Mechanism | In v1 | Notes |
|---|---|---|
| Predicate dispatch (`PageMetadataPredicate`) | ✓ | Justifies the runtime's existence — typed metadata matching that `@event_handler` glob patterns can't express. |
| Per-page rate limit (`WriteRateLimiter`) | ✓ | Catches runaway loops modifying the same page. |
| Episode budget (default 1000 dispatches) | ✓ | Catches runaway loops at the episode level. |
| Quiescence detection + emit | ✓ | Wakes `wait_for_quiescence` callers and writes `ConvergenceQuiescenceProtocol` events on the colony scope (consumer: `DesignCheckpointer` auto-tagging an `auto_quiescence_<iso8601>` checkpoint, master §8.1). |
| Numeric damping (`NumericTolerance`) | ✓ | Asymptotic-convergence absorber for capabilities producing scalar outputs (optimization loops, error-budget reconciliation, confidence-interval narrowing). A subscription with `tolerance=NumericTolerance(...)` is skipped when `event.extra["value"]` is within tolerance of the previous run for the same `(subscription_id, page_id)`. The producing capability is responsible for publishing the scalar in `extra["value"]`. |
| Topo-sort within an episode | ✗ | Cut. Was an optimization (order subscribers within a wave by declared output predicates), not correctness; the system converges through repeated waves without it. Re-add when there's a real subscription topology that benefits. |
| Cycle detection + leader-pick break | ✗ | Cut. Hostile to legitimate iterative design loops (requirements ↔ code ↔ simulation). The pathology it was nominally targeting (a single subscription thrashing the same page) is already caught by the per-page rate limit + episode budget. |
| `convergence:status` blackboard mirror | ✗ | Cut. UI polls `get_status()`. |
| `convergence:change_feed` blackboard mirror | ✗ | Cut. UI polls `get_change_feed(limit)`. |

### `ConvergenceDispatchProtocol`

[`ConvergenceDispatchProtocol`](../../src/polymathera/colony/agents/blackboard/protocol.py) defines the key shape between the runtime (sole writer) and the subscribing capability (sole reader):

- `dispatch_key(subscription_id) -> "convergence:dispatch:<sub_id>"`
- `dispatch_pattern() -> "convergence:dispatch:*"`
- `parse_dispatch_key(key) -> subscription_id`

### `ConvergenceQuiescenceProtocol`

[`ConvergenceQuiescenceProtocol`](../../src/polymathera/colony/agents/blackboard/protocol.py) defines the colony-scope event the runtime emits at each episode boundary:

- `quiescence_key(episode_id) -> "convergence:quiescence:<episode_id>"`
- `quiescence_pattern() -> "convergence:quiescence:*"`
- `parse_quiescence_key(key) -> episode_id`

The payload is a serialized `ConvergenceCounters`. The reference consumer is [`DesignCheckpointer`](../../src/polymathera/colony/design_monorepo/capabilities.py) — its `@event_handler(pattern=ConvergenceQuiescenceProtocol.quiescence_pattern())` tags an `auto_quiescence_<iso8601>` checkpoint when the working tree has uncommitted changes, giving the master §8.1 `restore_checkpoint(id=auto_quiescence_<timestamp>)` crash-recovery primitive a real producer.

---

## 4. The agent-facing surface

[`ConvergenceCapability`](../../src/polymathera/colony/agents/patterns/capabilities/convergence.py) gives any agent the primitives master §3.4 / §5.4 specifies — `subscribe_pattern`, `unsubscribe`, `dispatch_change`, `wait_for_quiescence`, `get_convergence_status`, `get_change_feed`:

```python
class MyCoordinator(Agent):
    async def initialize(self) -> None:
        self.add_capability_blueprints([ConvergenceCapability.bind()])
        await super().initialize()
        # Register a subscription; the runtime dispatches onto our scope.
        cc = self.get_capability(ConvergenceCapability)
        await cc.subscribe_pattern(
            predicate=PageMetadataPredicate.equals(
                "data_type", "design_monorepo_file",
            ),
        )
```

The capability owns the receive side via:

```python
@event_handler(pattern=ConvergenceDispatchProtocol.dispatch_pattern())
async def _on_dispatch(self, event, repl) -> EventProcessingResult | None:
    sub_id = ConvergenceDispatchProtocol.parse_dispatch_key(event.key)
    if sub_id not in self._owned_subscription_ids:
        return None
    page_event = PageChangeEvent.model_validate(event.value)
    return EventProcessingResult(
        context_key=event.key,
        context={"subscription_id": sub_id, "page_event": page_event.model_dump(mode="json")},
    )
```

The capability tracks its own subscription ids so a clean shutdown unregisters them automatically — agents that suspend or terminate do not leak subscriptions. Subscription ids are checkpointed in `serialize_suspension_state` / `deserialize_suspension_state`.

---

## 5. Concrete consumers in colony

### `DesignCheckpointer` quiescence handler

[`DesignCheckpointer`](../../src/polymathera/colony/design_monorepo/capabilities.py) consumes `ConvergenceQuiescenceProtocol` events. When an episode settles with uncommitted changes in the working tree, the capability commits and tags an `auto_quiescence_<iso8601>` checkpoint, giving master §8.1's `restore_checkpoint(id=auto_quiescence_<timestamp>)` crash-recovery primitive a real producer. The behavior is opt-out via the `auto_checkpoint_on_quiescence=False` constructor flag for agents that need fully manual checkpointing.

### How a downstream capability wires damping

To consume the runtime's numeric damping in a domain-specific way, a downstream capability:

1. Constructs a `PageSubscription` with `tolerance=NumericTolerance(...)` matched to the engineering tolerance of the value being tracked.
2. Ensures the *producer* of the matching `PageChangeEvent`s populates `extra["value"]` with the new scalar (the runtime's damper reads it from there).
3. Adds an `@event_handler(pattern=ConvergenceDispatchProtocol.dispatch_pattern())` that filters by its own `subscription_id`, validates the payload, and republishes a typed event under a domain-defined `BlackboardProtocol` so further downstream consumers don't have to know about the convergence runtime's wire mechanics.

Downstream domain packages (e.g. CPS-domain capabilities for budget reconciliation in error-budget designs) implement this pattern in their own repos.

---

## Outstanding work

What's wired:

- `ConvergenceRuntimeDeployment` is registered by `VCMConfig.add_deployments_to_app` and starts unconditionally with the rest of the VCM subsystem.
- `GitRepoContextPageSource` declares `static = False` and `watch()` runs `LocalFsWatcher` + `GitRemoteWatcher` against the working tree (merged via `CompositeWatcher`). The VCM's `_start_watch_bridge` drains them and feeds events directly into the runtime. There is no separate registration call — mapping the working tree as a `GitRepoContextPageSource` is the registration.
- `BlackboardContextPageSource` declares `static = False` and overrides `watch()` to yield `PageChangeEvent`s as its event loop processes live writes. The VCM's `_start_watch_bridge` automatically drains the iterator into the runtime — the same generic bridge applies to any future non-static source.
- End-to-end smoke tests in [`test_chain_smoke.py`](../../src/polymathera/colony/vcm/convergence/tests/test_chain_smoke.py) exercise watcher → runtime → subscription dispatch with no Ray.

What's still deferred:

- **HTTP webhook receiver (Phase C6).** `WebhookEventBuilder` translates Gitea/GitLab/GitHub push payloads into `PageChangeEvent`s today; the HTTP route that takes a payload and calls into the builder lives with the Web UI work in C6.
- **Edge events from `BlackboardContextPageSource`.** Today the source emits `page_added` / `page_replaced` / `page_invalidated` only. It does not yet produce `page_graph_edge_added` / `page_graph_edge_removed` events; those need IngestionPolicy hooks to surface relationship changes between pages.
- **Tokenized-content edit_diff.** `PageReplaced` events from the blackboard source carry no `edit_diff`. Producing one requires the IngestionPolicy to retain enough of the previous flush to diff against.
- **Multi-replica leader election.** Today every VCM replica that materializes a non-static scope spins up its own watch bridge, so events fan in to the runtime N times per change. The runtime's per-page rate-limiter absorbs transient bursts; a leader election among VCM replicas is the durable fix.

---

## What this means for application code

- **Subscribe early.** Calling `ConvergenceCapability.subscribe_pattern(...)` from a coordinator's `initialize()` is the canonical pattern. Once the agent is up, mutations on watched sources flow back through the capability's `@event_handler(pattern=ConvergenceDispatchProtocol.dispatch_pattern())` and become planner context bindings.
- **The dispatch key is not a parameter.** It is owned by `ConvergenceDispatchProtocol` and derived from the subscription_id the runtime returns. No caller picks the key; that prevents the LLM action surface from carrying a free-form string the planner can not meaningfully fill in.
- **`dispatch_change` is for synthesis.** Use it when a capability *itself* derives a graph mutation (a deduplication step that retracts a citation, a coordinator that confirms a hypothesis), rather than waiting for the source to surface it. Tests use it to inject events without standing up watchers.

---

## Pointers

- Master design doc §5 ("the always-live design context"), §5.2 (convergence-runtime mechanics), §5.6 (the immutability gap and watcher transports) — [`colony_docs/markdown/apps/design_automation_architecture.md`](../../../colony_docs/markdown/apps/design_automation_architecture.md).
- Phase plan + progress — `colony/phase_c4_convergence_runtime_progress.md`.
