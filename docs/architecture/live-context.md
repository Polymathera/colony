# Live Context — Watchers + Convergence Runtime

This page documents the **always-live design context** machinery (master design doc §5): the chain that lets a change in an upstream source ripple through subscribed capabilities and converge to a quiescent design state.

The chain is built from four layers, each in its own colony package:

| Layer | Package | Role |
|-------|---------|------|
| 1. Source watchers | [polymathera.colony.vcm.watchers](../../src/polymathera/colony/vcm/watchers/) | Detect upstream mutations and emit `PageChangeEvent`s |
| 2. Page-event topic | `EnhancedBlackboard` (colony scope), key prefix `vcm:page_events:*` | Transport — decouples watchers from runtime initialisation order |
| 3. Convergence runtime | [polymathera.colony.vcm.convergence](../../src/polymathera/colony/vcm/convergence/) | Subscriptions, topological dispatch, quiescence detection, cycle break, rate limit, damping |
| 4. Agent surface | [polymathera.colony.agents.patterns.capabilities.ConvergenceCapability](../../src/polymathera/colony/agents/patterns/capabilities/convergence.py) | `subscribe_pattern`, `unsubscribe`, `dispatch_change`, `wait_for_quiescence`, `get_convergence_status`, `get_change_feed`, `detect_cycle` |

All four layers are implemented and wired end-to-end at cluster boot. The chain runs whenever:

1. `VCMConfig.add_deployments_to_app` registers `ConvergenceRuntimeDeployment` (it always does, unconditionally).
2. A capability inheriting from `_DesignMonorepoCapabilityBase` initialises against a working tree — its `initialize()` registers a `DesignMonorepoWatcher` with the convergence runtime, idempotently keyed by the absolute working-tree path so N agents binding to the same monorepo produce one watcher.
3. The VCM materialises a scope mapping for any non-static `ContextPageSource` (e.g., `BlackboardContextPageSource`) — `_start_watch_bridge` automatically drains the source's `watch()` iterator into the colony scope's `vcm:page_events:*` topic.

Layer 1 (HTTP webhook receiver) is the only piece deferred — the `WebhookEventBuilder` translator exists; the route lives in Phase C6.

End-to-end smoke tests live in [`vcm/convergence/tests/test_chain_smoke.py`](../../src/polymathera/colony/vcm/convergence/tests/test_chain_smoke.py) — they cover `LocalFsWatcher`, `SourcePollWatcher`, and the formal `ContextPageSource.watch()` contract.

---

## 1. The `ContextPageSource.watch()` contract

`ContextPageSource` ([context_page_source.py](../../src/polymathera/colony/vcm/sources/context_page_source.py)) carries a `watch()` method on the base ABC plus a class-level `static: ClassVar[bool] = True` flag. The default `watch()` raises `NotImplementedError`. Per the docstring:

> Sources that can detect upstream changes and translate them into page-graph mutations declare `static = False` and override `watch()`. Sources that cannot — a one-shot static dump, an archived corpus snapshot — leave `static = True`; the convergence runtime refuses to subscribe to a static source as a live page-graph input and instead relies on whatever bulk re-ingestion path the source provides.

### Current state

| Source | `static` | `watch()` shape | Live-event production |
|--------|----------|----------------|-----------------------|
| [`FileGrouperContextPageSource`](../../src/polymathera/colony/samples/paging/file_grouper_page_source.py) | `False` | Yields from a wrapped `LocalFsWatcher` rooted at the cloned working tree | Filesystem events on tracked files (those in `file_to_page`) become `PageChangeEvent`s with `data_type="design_monorepo_file"` and the affected page's id |
| [`BlackboardContextPageSource`](../../src/polymathera/colony/agents/blackboard/paging/blackboard_page_source.py) | `False` | Yields from an internal `asyncio.Queue` populated by the source's record-event loop | Each blackboard write/update/delete diffs the page graph; new pages emit `PageAdded`, retired pages emit `PageInvalidated`, page reassignments emit `PageReplaced` |

The two shapes differ but the *contract* is uniform: every non-static source exposes mutations as an `AsyncIterator[PageChangeEvent]`. The runtime side does not know or care whether the iterator delegates to a sidecar watcher (`LocalFsWatcher`) or pulls from the source's own internal event loop.

#### Limitations to know about

- **`FileGrouperContextPageSource` only watches files in `file_to_page` at the time `watch()` starts.** A file added to the working tree after the page graph was built fires no events; detecting it requires a graph rebuild, which is a separate pass.
- **Multi-replica VCMs duplicate events.** Every replica that materialises a non-static scope mapping starts its own watcher; events get N-fold duplicated on the page-event topic. The runtime's per-page rate-limiter absorbs transient bursts, but a leader-election story is the durable fix. The same caveat applies to `BlackboardContextPageSource`.

### The bridge: `VirtualContextManager._start_watch_bridge`

When the VCM materialises a scope mapping, it inspects `source.static`. If the source is non-static, the VCM starts a long-running task with the canonical shape:

```python
async for event in source.watch():
    await publisher.publish(event)
```

`publisher` is a `PageEventPublisher` against the colony scope's `vcm:page_events:*` topic, where the convergence runtime forwarder reads. The bridge task lifetime is owned by the `MappedScope` — `_shutdown_mapped_scope` cancels it on unmap. One shared `EnhancedBlackboard` handle backs all bridges in a VCM replica; it is released when the last scope is unmounted.

This means: any future `ContextPageSource` subclass that sets `static = False` and implements `watch()` is automatically wired into the convergence chain. No special hook on the source, no new code in the bridge, no additional registration call.

---

## 2. Watcher transports (master §5.6)

Three transport classes plus a webhook payload translator, all in [vcm/watchers/](../../src/polymathera/colony/vcm/watchers/):

- **`LocalFsWatcher`** — `watchdog`-based filesystem watcher with a debounce window; falls back to mtime-poll when `watchdog` is unavailable.
- **`GitRemoteWatcher`** — periodic `git fetch` + `git diff --name-only` against a local clone; covers remote-driven push fallback when no webhook is available.
- **`SourcePollWatcher`** — generic interval poll over any `ContextPageSource`'s `get_all_mapped_pages()` snapshot; the catch-all transport for sources behind APIs (arXiv RSS, supplier catalogues) with no push notification.
- **`WebhookEventBuilder`** — translates a Gitea / GitLab / GitHub git-push webhook payload into a sequence of `PageChangeEvent`s. (HTTP receiver is a Phase C6 concern; the translator lives here so the watcher contract stays in one place.)

All four set `static = False` and emit `PageChangeEvent`s. Watchers are **not** subclasses of `ContextPageSource` — they are sidecar classes that operate alongside a source. This decouples watch lifecycle from page-source lifecycle and lets multiple watchers cover one source (LocalFs *and* GitRemote on the same working tree, for example).

### Publisher

[`PageEventPublisher`](../../src/polymathera/colony/vcm/watchers/publisher.py) is the small adapter that takes events from a watcher loop and writes them to the colony scope's `vcm:page_events:*` topic. The convergence runtime's forwarder consumes from that topic — never from a watcher directly.

### Bridge: `DesignMonorepoWatcher`

[`design_monorepo/watcher.py`](../../src/polymathera/colony/design_monorepo/watcher.py) glues a `DesignMonorepoClient` to the runtime by spinning up `LocalFsWatcher` + `GitRemoteWatcher` against the working tree and feeding both through one `PageEventPublisher`. It is the canonical "high-level live source" composition.

The watcher's lifetime is owned by `ConvergenceRuntimeDeployment`. Capabilities that bind to a working tree call `register_design_monorepo(working_dir)` on the runtime (transparently — `_DesignMonorepoCapabilityBase.initialize()` does this on every agent that uses any of the three design-monorepo capabilities). The runtime dedups by absolute path, so N agents on the same program produce one watcher; the watcher shuts down on cluster shutdown.

---

## 3. The convergence runtime

[`vcm/convergence/`](../../src/polymathera/colony/vcm/convergence/) implements the dispatch loop the master doc §5.2 describes:

```
PageChangeEvent  ─┐
                  │  forwarder (in ConvergenceRuntimeDeployment)
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
        EnhancedBlackboard.write to sub.dispatch_scope/sub.dispatch_key
                  │
                  ▼
   subscriber's @event_handler picks it up and fires
```

Module map:

- [`runtime.py`](../../src/polymathera/colony/vcm/convergence/runtime.py) — `ConvergenceRuntime` (pure dispatch logic), `ConvergenceState`, `ConvergenceStatus`, `ConvergenceCounters`, `ChangeFeedEntry`. Tracks the current episode, detects quiescence, breaks cycles when the episode budget is exhausted.
- [`subscriptions.py`](../../src/polymathera/colony/vcm/convergence/subscriptions.py) — `PageSubscription`, `NumericTolerance`.
- [`predicates.py`](../../src/polymathera/colony/vcm/convergence/predicates.py) — `PageMetadataPredicate` (typed match expression over page metadata) + `EdgeReachResolver` for graph-aware predicates.
- [`index.py`](../../src/polymathera/colony/vcm/convergence/index.py) — `SubscriptionIndex` (fast lookup by event metadata).
- [`damping.py`](../../src/polymathera/colony/vcm/convergence/damping.py) — numeric-tolerance check that suppresses dispatches inside a configured tolerance.
- [`rate_limit.py`](../../src/polymathera/colony/vcm/convergence/rate_limit.py) — `WriteRateLimiter` (per-page write throttle).
- [`deployment.py`](../../src/polymathera/colony/vcm/convergence/deployment.py) — `ConvergenceRuntimeDeployment`, the Ray-serving singleton wrapping the runtime; runs the forwarder task that consumes `vcm:page_events:*` from the colony blackboard.

The runtime emits three surfaces back onto the colony blackboard for the SessionAgent's UI panel (master §5.4): a `ConvergenceStatus` snapshot, per-episode quiescence markers, and the change feed.

---

## 4. The agent-facing surface

[`ConvergenceCapability`](../../src/polymathera/colony/agents/patterns/capabilities/convergence.py) gives any agent the seven primitives master §3.4 / §5.4 specify:

```python
class MyCoordinator(Agent):
    async def initialize(self) -> None:
        self.add_capability_blueprints([ConvergenceCapability.bind()])
        await super().initialize()
        # Register a subscription; the runtime dispatches onto our scope.
        cc = self.get_capability(ConvergenceCapability)
        await cc.subscribe_pattern(
            predicate=PageMetadataPredicate.equals("data_type", "design_monorepo_file"),
            dispatch_key="convergence:design_change",
        )
```

The capability tracks its own subscription ids so a clean shutdown unregisters them automatically — agents that suspend or terminate do not leak subscriptions. Subscription ids are checkpointed in `serialize_suspension_state` / `deserialize_suspension_state`.

---

## Outstanding work

What's wired:

- `ConvergenceRuntimeDeployment` is registered by `VCMConfig.add_deployments_to_app` and starts unconditionally with the rest of the VCM subsystem.
- `_DesignMonorepoCapabilityBase.initialize()` calls `register_design_monorepo(working_dir)` on the runtime, which spins up a `DesignMonorepoWatcher` (idempotent per absolute working-tree path).
- `BlackboardContextPageSource` declares `static = False` and overrides `watch()` to yield `PageChangeEvent`s as its event loop processes live writes. The VCM's `_start_watch_bridge` automatically drains the iterator into the colony scope's `vcm:page_events:*` topic — the same generic bridge applies to any future non-static source.
- End-to-end smoke tests in [`test_chain_smoke.py`](../../src/polymathera/colony/vcm/convergence/tests/test_chain_smoke.py) exercise watcher → runtime → subscription dispatch with no Ray.

What's still deferred:

- **HTTP webhook receiver (Phase C6).** `WebhookEventBuilder` translates Gitea/GitLab/GitHub push payloads into `PageChangeEvent`s today; the HTTP route that takes a payload and calls into the builder lives with the Web UI work in C6.
- **Edge events from `BlackboardContextPageSource`.** Today the source emits `page_added` / `page_replaced` / `page_invalidated` only. It does not yet produce `page_graph_edge_added` / `page_graph_edge_removed` events; those need IngestionPolicy hooks to surface relationship changes between pages.
- **Tokenized-content edit_diff.** `PageReplaced` events from the blackboard source carry no `edit_diff`. Producing one requires the IngestionPolicy to retain enough of the previous flush to diff against.

---

## What this means for application code

- **Subscribe early.** Calling `ConvergenceCapability.subscribe_pattern(...)` from a coordinator's `initialize()` is the canonical pattern. Once the agent is up, mutations on watched sources flow back as `@event_handler(pattern="convergence:...")` dispatches.
- **`@event_handler` is the receive end.** The runtime writes to `sub.dispatch_scope/sub.dispatch_key`; the capability's existing event-handler machinery picks it up. No special hook on the capability side.
- **`dispatch_change` is for synthesis.** Use it when a capability *itself* derives a graph mutation (a deduplication step that retracts a citation, a coordinator that confirms a hypothesis), rather than waiting for the source to surface it. Tests use it to inject events without standing up watchers.

---

## Pointers

- Master design doc §5 ("the always-live design context"), §5.2 (convergence-runtime mechanics), §5.6 (the immutability gap and watcher transports) — [`colony_docs/markdown/apps/design_automation_architecture.md`](../../../colony_docs/markdown/apps/design_automation_architecture.md).
- Phase plan + progress — `colony/phase_c4_convergence_runtime_progress.md`.
- VCM page-source overview — [Virtual Context Memory](virtual-context-memory.md).
- Agent-facing capability — referenced from [Agent System](agent-system.md) and [Action Policies](action-policies.md).
