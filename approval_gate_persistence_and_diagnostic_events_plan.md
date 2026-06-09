# Approval-Gate Persistence + Agent Diagnostic Events Plan

## Context

The 2026-06-08 03:07–03:16 live run exercised the new decompose primitives end-to-end: `SessionAgent` picked `mode=decompose`, coordinator called `classify_issues_decomposability` → `propose_decompositions` → `request_human_approval` → poll → user clicked APPROVE TWICE. Zero GitHub mutations landed because every `create_decomposition(dry_run=False)` was blocked with *"no successful approval signal in this iteration"*.

**Root cause** ([code_generation.py:1775](colony/src/polymathera/colony/agents/patterns/actions/code_generation.py#L1775)): `self._call_history.clear()` wipes the policy's per-iteration call history at the start of every iteration. The `ApprovalRequiredGuardrail._default_is_approval_granted` predicate ([code_constraints.py:1262](colony/src/polymathera/colony/agents/patterns/actions/code_constraints.py#L1262)) scans `_call_history` for a successful `get_response` returning `choice='approve'`. The approval observed in iteration N is gone by iteration N+1 — but the gated apply naturally lands in N+1 (or later). The approval gate is per-iteration; the approval flow is inherently multi-iteration. The two are incompatible.

**Two items in this plan**, both shipped together:

1. **Approval-gate persistence** — read approval state from the blackboard (where it already lives durably), not from the iteration-local call history. Plus a 3-choice approval model (reject / approve once / approve all of this type in this session) matching Claude Code's tool-permission shape.
2. **Agent diagnostic event protocol** — a general typed-event channel for "a subsystem detected a noteworthy internal pattern; other agents need to know" (guardrail block streaks, LLM failure streaks, polling timeouts, capability init failures, etc.). Same mechanism the framework already uses for business-logic events (HumanApprovalResponse, RoadmapSync, BottleneckDetected). First producer: the action policy on a guardrail-block streak. First consumer: SessionAgent reports the stuck coordinator to the user instead of going silent.

Out of scope (flagged explicitly): the `IterationShapeValidator`'s 50-line cap. Originally I flagged it as a problem because the band-aid version of the fix would force same-cell `get_response → create_decomposition`. Under the durable fix (item 1) the same-cell pattern is unnecessary; the cap stays unchanged.

Per [[surface-assumptions-in-plans]], each item has an explicit **Assumptions** subsection naming the load-bearing choices.

---

## Item 1 — Approval-gate persistence with action-type scoping

**Symptom**: gate blocks approved actions because the approval signal lives in iteration-local state.

**Band-aid version (REJECTED)**: stop clearing `_call_history` at iteration boundaries. **Why band-aid**: the call history is iteration-scoped on purpose (per-iteration `args`-aware guardrail predicates need it scoped that way); broadening its lifetime to "agent lifetime" breaks every other guardrail's recency semantics. Also doesn't survive suspend/resume.

**Band-aid version 2 (REJECTED)**: add a `_approved_request_ids: set[str]` field on the policy that's NEVER cleared. **Why band-aid**: in-memory, lost on suspend/resume. Solves the symptom, leaves the architectural mismatch: the approval is persistent state and should be read from where it lives.

**Durable fix**: query the blackboard directly. The approval response already lives at `HumanApprovalProtocol.response_key(request_id)` for the lifetime of the session's namespace. Plus extend the model to support Claude Code–style action-type scoping with three choices:

- `reject` — block this action.
- `approve_once` — allow this single dispatch of the named `action_type`.
- `approve_all` — allow ALL future dispatches of the named `action_type` in this session, until explicitly revoked.

Mechanism:

- `HumanApprovalRequest` gains an `action_type: str | None` field. When set, the chat UI renders the 3-choice approval card. When `None` (default), the existing 2-choice (`approve` / `reject`) UI renders for backwards compatibility.
- `choice` values extend: `"reject" | "approve_once" | "approve_all"` for action-typed requests; `"reject" | "approve"` (unchanged) for the legacy untyped form.
- New helper on `HumanApprovalCapability`:
  ```python
  async def has_active_approval_for(self, action_key: str) -> tuple[bool, str | None]:
      """Return (True, request_id) if a non-revoked approval covers
      ``action_key`` (substring match on action_type), else (False, None).
      Order of consultation:
        1. approve_all responses whose action_type matches → allow (do NOT consume).
        2. unconsumed approve_once responses whose action_type matches → allow + mark consumed.
        3. Legacy untyped approve responses → allow (compat).
      """
  ```
- Consumption tracking for `approve_once`: a separate blackboard write at `HumanApprovalProtocol.consumption_key(request_id)` (or a `consumed_at: datetime` field on the response). Idempotent — re-marking a consumed approval is a no-op.
- `ApprovalRequiredGuardrail.check()` becomes async-aware of the approval capability. Two access paths considered:
  - (a) **Capability handle injection**: the guardrail is constructed with a callable that resolves to the agent's `HumanApprovalCapability` lazily (so cloudpickle survives). The guardrail calls `cap.has_active_approval_for(action_key)` directly.
  - (b) **Blackboard query helper**: the guardrail takes a blackboard scope-id and queries `human_approval:response:*` keys directly. Avoids the capability handle.
  - **Going with (a)**: cleaner separation, the capability already knows how to read its own persistent state.
- Approval prefixes stay the same (`requires_human_approval_before=[...]`); the matching is on `action_type` AS WELL AS the `action_key`. When `action_type` is provided, it MUST match the action being checked (substring match on the `action_key`); when absent, the legacy approve-anywhere semantic applies.

Coordinator self-concept changes: the planner is instructed to call `request_human_approval(action_type="create_decomposition", question=..., extra=...)` so the user sees the right action_type label on the approval card. The same applies to the other gated mission actions (`bootstrap_roadmap_from_objectives`, `sync_roadmap_with_github`, `propose_task_assignments`).

Frontend changes: the chat UI's approval card renders 3 buttons when `request.action_type is not None`. Existing 2-button rendering stays when `action_type is None`.

**Assumptions** (per [[surface-assumptions-in-plans]]):

- *The blackboard is the source of truth for approval state.* The capability's in-memory `_responses: dict[str, HumanApprovalResponse]` is a CACHE; the canonical state is on the blackboard. **Extensibility risk**: if a future capability needs to introspect the cache without going through the capability API, it'd miss state. Mitigation: the helper is the only sanctioned read path. **Code understandability risk**: a future reader could be tempted to read `cap._responses` directly. Documenting the cache-vs-canonical distinction in the field docstring.

- *`action_type` is a single free-text string per request, not a list.* Claude Code's tool-permission model is per-tool. **Extensibility risk**: an approval that should gate two action types (e.g. "create child issues" + "update parent body" — both implied by approving a decomposition) needs a multi-`action_type` primitive. NOT shipping that yet; flag as future work. The current `create_decomposition` action does both internally, so one `action_type="create_decomposition"` is enough for our case.

- *Substring matching on `action_type` vs `action_key`.* The guardrail's gated prefixes use case-insensitive substring matching today (e.g. `"DesignProcessCapability.create_decomposition"` matches any `action_key` containing that substring). Extending the same shape to `action_type`. **Modularity risk**: the user-visible `action_type` is now both a UI label AND a match string. If the user reads `"create_decomposition"` on the card and approves, they're implicitly approving any action key containing that substring. The naming convention has to be careful. Mitigation: `action_type` defaults to the gated prefix from the policy's config, so it's already structurally aligned.

- *`approve_all` is session-scoped, not colony- or tenant-scoped.* Aligned with the blackboard's `human_approval` namespace scope (session). **Extensibility risk**: a tenant operator approving "all `create_decomposition` in this colony forever" is a different feature; flag as future work. For now, a new session means re-prompting.

- *Consumption of `approve_once` is recorded on the blackboard, not in-memory.* Survives suspend/resume. **Performance risk**: every gated dispatch's guardrail check does a blackboard read. The capability's in-memory cache absorbs the common case; consumption writes are rare (once per `approve_once`). Acceptable.

- *Backwards compatibility for the 2-choice case.* When `action_type` is `None`, the legacy `approve` choice still unlocks the legacy gate. Existing missions (bootstrap / refresh / assignments coordinator self-concept) don't break. **Code understandability risk**: two parallel choice vocabularies (`approve` vs `approve_once`/`approve_all`) is confusing for a reader; the distinction is purely `action_type`-presence. Mitigation: a `is_typed_request()` predicate on the request model so reading code is unambiguous.

- *Guardrail gets the capability handle via a lazy factory, not a direct reference.* The guardrail is built at session-create time (before any agent is constructed); the capability is mounted on the agent at spawn time. The factory takes `agent: Agent` at `bind_speaker` time (which we already wired in item 4 of the prior plan!) and resolves the capability then. **Modularity risk**: this couples the approval guardrail to the existence of `HumanApprovalCapability` on the speaker. If a future guardrail config gates an action on an agent that doesn't mount the approval capability, every check returns "no approval" → blocked. Mitigation: document the requirement on `ApprovalRequiredGuardrail`'s docstring; surface a clearer error message when the capability is missing.

**Files touched**:
- `colony/src/polymathera/colony/agents/patterns/capabilities/human_approval.py` — `action_type` field, `has_active_approval_for` helper, consumption tracking.
- `colony/src/polymathera/colony/agents/patterns/actions/code_constraints.py` — `ApprovalRequiredGuardrail.check()` becomes capability-aware; `_default_is_approval_granted` is deleted (no longer how approvals are detected).
- `colony/src/polymathera/colony/agents/patterns/actions/code_generation.py` — wire the capability factory into the guardrail at policy init (same hook as `bind_speaker`).
- `colony/src/polymathera/colony/web_ui/backend/chat/session_agent_guardrails.py` — pass the approval capability factory into `ApprovalRequiredGuardrail`.
- `colony/src/polymathera/colony/agents/missions/project_planning/coordinator.py` — self-concept text update.
- `colony/src/polymathera/colony/agents/configs.py` — coordinator self-concept text update.
- Frontend (TBD path) — render 3-choice card when `action_type` is present.
- Tests: extend `test_human_approval.py` + `test_runtime_guardrails.py` + integration test that exercises approve_all → second dispatch unblocked without re-prompting.

---

## Item 2 — `AgentDiagnosticProtocol` for cross-agent visibility

**Symptom**: when the coordinator spent 5+ iterations bouncing off the approval gate, the `SessionAgent` watching it had no signal. The user clicked approve twice; chat went silent; coordinator stopped with `policy_completed` because the LLM eventually gave up. The `SessionAgent` should have noticed the streak and told the user *"the coordinator is stuck on the approval gate; this is a known framework bug; aborting"* — or after item 1, *"your approval went through, the coordinator is applying now."*

**Band-aid version (REJECTED)**: special-case the `SessionAgent` to introspect its spawned children's call histories. Couples the `SessionAgent` to internal action-policy state of other agents; doesn't generalise to LLM failure streaks, polling timeouts, etc.

**Durable fix**: a new typed protocol `AgentDiagnosticProtocol` on the agent's own blackboard scope, sibling of the existing `HumanApprovalProtocol` / `RoadmapSyncProtocol` / `BottleneckDetectedProtocol`. Same mechanism, applied to internal-state events instead of business-logic events.

Surface:

```python
class AgentDiagnosticProtocol:
    """Typed events on the agent's own scope for cross-agent visibility
    of internal failure patterns. Producers: action policies, capabilities.
    Consumers: parents, observers via @event_handler."""

    @staticmethod
    def event_key(agent_id: str, kind: str, sequence: int) -> str:
        return f"agent:diagnostic:{agent_id}:{kind}:{sequence}"

    @staticmethod
    def event_pattern(agent_id: str | None = None) -> str:
        return (
            f"agent:diagnostic:{agent_id or '*'}:*:*"
        )
```

Event kinds in v1:
- `guardrail_block_streak` — payload: `{action_key, count, last_reason, last_suggestion, first_blocked_at, last_blocked_at}`. Emitted by the action policy when the SAME action_key has been blocked K times in a row (default K=3, debounced so a single transient block doesn't fire). Reset to 0 when the action_key successfully dispatches or when a different action_key is blocked.

Future kinds (NOT shipped here, naming-only so the contract is clear): `code_validation_streak`, `llm_call_failure_streak`, `polling_timeout`, `capability_init_failure`, `budget_threshold_crossed`. Each is added when first needed; the protocol shape is uniform.

Producer wiring (action policy):
- `CodeGenerationActionPolicy._track_block_streak(action_key, decision)` — internal helper called from the `run()` block-capture path. Maintains `self._block_streak_action_key` and `self._block_streak_count`. When count crosses the threshold, emits a `guardrail_block_streak` event.
- Debouncing: the threshold (default 3) is configurable on the policy; the first emission resets the count to prevent spam (re-fires after 3 more consecutive blocks).

Consumer wiring (`SessionAgent`):
- A new `@event_handler(pattern=AgentDiagnosticProtocol.event_pattern())` on `SessionOrchestratorCapability` (or a new `AgentDiagnosticsCapability` if it grows).
- The handler does NOT decide what to do — it surfaces a structured `EventProcessingResult` whose `context_key` / `context` lands in the `SessionAgent`'s next planner iteration. The LLM planner sees *"diagnostic event: coordinator X has been blocked Y times on action Z"* and decides whether to tell the user, re-request approval, abort, or wait.

**Assumptions** (per [[surface-assumptions-in-plans]]):

- *The diagnostic events go on the AGENT'S OWN blackboard scope, not a shared diagnostics scope.* This means the `SessionAgent`'s handler has to be scoped to listen for events on its child coordinator's blackboard. **Modularity risk**: a parent already has a known relationship with its child (via `parent_agent_id` on `AgentMetadata`); the handler builds the pattern for the child's scope at spawn time. **Extensibility risk**: a non-parent observer (e.g. a future "colony health monitor" agent) needs the same hook to subscribe to ANY agent's diagnostics; the wildcard `AgentDiagnosticProtocol.event_pattern(agent_id=None)` handles that.

- *Event kinds are open-ended strings, not a closed enum.* New kinds are added without touching every reader. **Code understandability risk**: a reader can't enumerate all possible kinds from a single place; mitigated by a registry-style constant module that documents known kinds. Adding a new kind = adding a constant + updating docs.

- *Debouncing is per-policy, not per-event-handler.* The producer decides the threshold. **Performance risk**: a misconfigured low threshold spams the blackboard. Mitigation: hard floor of K≥3, soft cap of K≤10. Tunable on the policy.

- *The handler converts the event into planner context, not into a direct LLM call.* The `SessionAgent`'s planner is in the loop; the handler doesn't bypass it. **Modularity risk**: an aggressive future handler that wants to react immediately (e.g. abort the child agent without LLM mediation) would need a different shape; flag as future work.

- *Diagnostic events live forever in the namespace (no TTL).* Aligned with how `human_approval` and `roadmap_sync` events behave today. **Performance risk**: long-running sessions accumulate diagnostic events. Mitigation: the namespace is session-scoped, so the events die when the session does. Per-event TTL is future work if a tenant runs many sessions.

- *The action policy is the right producer surface, not the runtime guardrail.* The guardrail returns a `GuardrailDecision`; the policy decides whether to emit. **Code understandability risk**: a future guardrail-specific event might want guardrail-internal info (e.g. which rule fired in a composite). The policy can pass the decision verbatim; the handler reads what it needs.

**Files touched**:
- `colony/src/polymathera/colony/agents/blackboard/protocol.py` — `AgentDiagnosticProtocol` class + event-kind constants.
- `colony/src/polymathera/colony/agents/patterns/actions/code_generation.py` — `_track_block_streak` + event emission.
- `colony/src/polymathera/colony/web_ui/backend/chat/session_agent.py` — `@event_handler` on `SessionOrchestratorCapability` for diagnostic events from spawned children.
- Tests: a new `test_agent_diagnostic_protocol.py` covering protocol key shape + producer-side debouncing; integration test where N consecutive blocks → event fires → SessionAgent handler receives it.

---

## Sequencing

Item 1 and item 2 land together. Implementation order:

1. **Item 1a — backend approval-state plumbing**: `action_type` field + `has_active_approval_for` helper + guardrail capability injection. Tests at this layer first.
2. **Item 1b — coordinator self-concept update**: planner instructions to pass `action_type` to request_human_approval.
3. **Item 1c — frontend 3-choice card**: when the backend changes are tested, wire the UI.
4. **Item 2 — diagnostic protocol**: shipped after item 1's plumbing because the producer hook lives in the policy that item 1 also touches; doing them in sequence reduces merge churn.

Each step: green test suite + commit-worthy slice before moving on.

---

## Verification gate

- `python -m pytest colony/src/polymathera/colony/agents colony/src/polymathera/colony/web_ui colony/src/polymathera/colony/design_monorepo colony/src/polymathera/colony/cli -q --tb=line --ignore=colony/src/polymathera/colony/agents/blackboard/tests/test_integrations.py` — 1419/1419 baseline holds or grows.
- `colony-env down && colony-env up --workers 3 --config ~/workspace/polymathera_inc/colony/configs/example.yaml`
- Repeat the user's chat input: *"Let's refine the roadmap. Most GitHub issues seem too high-level and we probably need to create sub-issues to break them down into more manageable tasks."*
- Expect:
  - Coordinator runs classify → propose → request_human_approval (with `action_type="create_decomposition"`).
  - Chat UI shows the 3-choice card: Reject / Approve once / Approve all `create_decomposition` in this session.
  - User clicks "Approve all" → first `create_decomposition` lands on GitHub → second `create_decomposition` ALSO lands without re-prompting.
  - Or: user clicks "Approve once" → first `create_decomposition` lands → second `create_decomposition` blocked with a NEW approval card.
  - No `_call_history`-clearing failures. No silence on the SessionAgent side.
- Negative scenario: deliberately deny approval → coordinator gets a `guardrail_block_streak` after K consecutive blocks (if it tries to apply anyway) → SessionAgent's planner sees the diagnostic event and tells the user the coordinator was rejected, then aborts.

---

## What this plan does NOT include

- IterationShapeValidator's 50-line cap (item #5 from the prior analysis) — dissolved under this plan's design; not changed.
- Multi-action_type per approval request (flag as future work in item 1).
- Tenant- or colony-scoped `approve_all` (currently session-scoped only).
- Per-event TTL on diagnostic events (currently lives for session lifetime).
- A direct-LLM-call escalation handler that bypasses the SessionAgent's planner (currently planner-mediated only).
- Refactoring the existing untyped approve/reject in older missions to use action_type — backwards compatible; can migrate per-mission later.

---

## Awaiting review

I will not touch code until you confirm:
- The 3-choice approval model (`reject` / `approve_once` / `approve_all`) is what you want — particularly the session-scoped lifetime of `approve_all`.
- Diagnostic-event protocol naming + the producer/consumer split.
- Sequencing: items 1+2 together vs. 1 first then 2 as a follow-up.
