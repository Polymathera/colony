# Action Policy Dimensions

`CodeGenerationActionPolicy` — Colony's default agent action policy — is composed of six pluggable dimensions. Each dimension is an abstract base class with a small surface; you swap or extend any of them without rewriting the others. The dimensions are defined in [`colony/agents/patterns/actions/code_constraints.py`](../../src/polymathera/colony/agents/patterns/actions/code_constraints.py) and consumed by [`code_generation.py`](../../src/polymathera/colony/agents/patterns/actions/code_generation.py).

This guide formalises:

- The shape and purpose of each dimension.
- How to choose between writing a **soft** guideline in a docstring vs. a **hard** rule in a guardrail/validator.
- The "composite action over precondition" pattern: when the cleanest fix to a precondition is to delete the precondition by fusing the two actions.

> If you're new to Colony, start with [Concepts](../getting-started/concepts.md) and [Registering a Mission](registering-a-mission.md). Come back here once you need to add a new agent or harden an existing one's action surface.

## The six dimensions

| # | Dimension | When to override | Cost of a bad choice |
|---|---|---|---|
| 1 | `CodeGenerator` | Constrained decoding, skeleton fill-in, agent-specific prompt scaffolding | Low — affects code quality, not correctness |
| 2 | `CodeValidator` | Block code before execution (AST checks, API hallucination correction, monolithic-iteration prevention) | Medium — bad validator over-rejects, under-rejects |
| 3 | `SkillLibrary` | Reuse successful code across iterations / sessions | Low — falls back to generation when no match |
| 4 | `RecoveryStrategy` | Handle execution failure (deterministic fix, LLM retry, rollback) | Medium — bad strategy loops or gives up too early |
| 5 | `RuntimeGuardrail` | Enforce constraints **during** execution — capability boundaries, temporal ordering, args-aware precondition checks | **High** — bad guardrail silently corrupts agent behaviour |
| 6 | `CompletionValidator` | Gate `signal_complete()` on goal achievement | Medium — bad validator lets the agent declare victory early |

Each dimension's default is conservative (`NoGuardrail`, `NoOpValidator`, `NoRecovery`, etc.) so the framework runs out of the box; agents opt into stricter shapes as their action surface grows.

## Soft vs. hard enforcement: pick the right tool

Colony agents can be constrained at three layers, weakest to strongest:

1. **Docstring / prompt convention** — the action's `__doc__` says "call X before this." The LLM reads it on iteration 1 and usually obeys. **Cheap, fast, can be ignored by the LLM under pressure.**
2. **`CodeValidator`** — AST checks before execution. Catches structural mistakes (too many `run()` calls, hallucinated action keys, blocked imports). **Deterministic, runs once per code block, no per-action overhead.**
3. **`RuntimeGuardrail`** — checked between every `run()` call inside the same code block. Sees action keys, params, and the full call history. **Deterministic, runs per dispatch.**

The rule of thumb: **hard-enforce when the operator-facing cost of a wrong call is high or the failure is invisible.** Soft for quality-of-life and stylistic concerns. Anything that mutates external state, makes user-facing claims, or burns budget belongs behind a hard guardrail.

| Concern | Where to put it |
|---|---|
| "Each iteration is a focused 1–3 actions, not a monolithic program" | `CodeValidator` (`IterationShapeValidator`) — structural, AST |
| "Don't call `apply` without `approval_granted`" | `RuntimeGuardrail` — depends on call history |
| "Don't claim agent state without `get_agent_status` for that agent" | `RuntimeGuardrail` — depends on call history + args |
| "This action is the most natural choice when X" | Docstring — too soft to enforce |
| "Pick the cheaper LLM model for trivial tool replies" | Docstring + model-selection in `CodeGenerator` |

## The composite-action pattern

Often the cleanest way to enforce a precondition is to **delete the precondition by fusing the two actions**. Instead of:

```python
# Soft: the LLM must remember to call get_agent_status first.
@action_executor()
async def respond_to_user(self, *, content: str): ...

@action_executor()
async def get_agent_status(self, *, agent_ids: list[str]): ...
```

expose **one** action that does both:

```python
@action_executor()
async def report_mission_status_to_user(
    self, *, agent_id: str, audience: Literal["chat", "log"] = "chat",
) -> dict[str, Any]:
    """Fetch the running coordinator's state, format it, then
    send the report to the chat.

    This action exists so the LLM doesn't have to remember the
    "check before report" sequence: the sequence is the action's
    body. The bare ``respond_to_user(content=...)`` action is
    still available for free-form messages that don't claim
    state.
    """
    state = await self.get_agent_status(agent_ids=[agent_id])
    return await self.respond_to_user(
        content=self._render_status_report(state),
    )
```

The LLM's planner now picks `report_mission_status_to_user` for status updates because the docstring is the most specific match. The two-step dance disappears, and the `respond_to_user` action retains its general utility for messages that don't reference agent state. **Composability beats convention.**

Use this pattern whenever a precondition can be eliminated by fusion. Reach for `RuntimeGuardrail` when fusion is impossible (e.g. the action's content is free-form LLM-authored text that might or might not make a state claim, or when the precondition crosses capability boundaries).

## Writing a `RuntimeGuardrail`

A guardrail's `check()` method receives the action about to be
dispatched and the call history so far. It returns a
`GuardrailDecision(allowed, reason, suggestion)`. The
`suggestion` field goes back to the LLM as a hint when blocked,
so the next iteration can fix the call cheaply.

```python
class StatusClaimGuardrail(RuntimeGuardrail):
    """Block respond_to_user calls whose content references an
    agent_id unless get_agent_status was called recently for that
    agent."""

    AGENT_ID_RE = re.compile(r"agent-[0-9a-f]+")

    async def check(self, action_key, params, call_history):
        if "respond_to_user" not in action_key:
            return GuardrailDecision(allowed=True)
        content = params.get("content", "")
        agent_ids = self.AGENT_ID_RE.findall(content)
        if not agent_ids:
            return GuardrailDecision(allowed=True)  # no claim

        recent_status_calls = [
            c for c in call_history[-10:]
            if "get_agent_status" in c
        ]
        if not recent_status_calls:
            return GuardrailDecision(
                allowed=False,
                reason=(
                    f"Content references {agent_ids} but "
                    "get_agent_status was not called recently."
                ),
                suggestion=(
                    "Call AgentPoolCapability.get_agent_status "
                    f"with agent_ids={agent_ids} before "
                    "reporting state."
                ),
            )
        return GuardrailDecision(allowed=True)
```

Mount it via the agent's action-policy config:

```python
SessionAgent.bind(
    ...,
    action_policy_config={
        "runtime_guardrail": StatusClaimGuardrail(),
    },
)
```

For multiple gates, wrap in `CompositeGuardrail`:

```python
runtime_guardrail = CompositeGuardrail(
    StatusClaimGuardrail(),
    ApprovalRequiredGuardrail(),
    CapabilityBoundaryGuardrail(
        allowed_prefixes=["SessionOrchestrator", "AgentPool"],
    ),
)
```

Gates run in order; the first to return `allowed=False` short-circuits.

## Writing a `CodeValidator`

Validators check the **whole code block** before it executes — they don't see the call sequence, they see the AST. Use them for:

- Import safety (already covered by `ImportWhitelistValidator`).
- Structural shape — too many actions per iteration, browse abuse, monolithic-program detection (`IterationShapeValidator`).
- API correctness — hallucinated action keys, edit-distance correction (`APIKnowledgeBaseValidator`).

A validator returns `ValidationResult(valid, errors, fixed_code, details)`. If `fixed_code` is non-None, the `RecoveryStrategy` may apply it deterministically. If `valid=False` and no fix is available, execution is rejected and the LLM retries with the error context.

Multiple validators run in sequence; any failure short-circuits.

## Choosing between `CodeValidator` and `RuntimeGuardrail`

| If the rule... | Use |
|---|---|
| ...depends only on the code's AST | `CodeValidator` |
| ...depends on which action is about to run + its params | `RuntimeGuardrail` |
| ...depends on what was already called in this code block | `RuntimeGuardrail` |
| ...needs to consult an external state (DB, blackboard) | `RuntimeGuardrail` |
| ...can be deterministically fixed (rename, reformat) | `CodeValidator` with `fixed_code` |
| ...is "agent shouldn't have generated this in the first place" | `CodeValidator` |
| ...is "agent should re-think after seeing what's already happened" | `RuntimeGuardrail` |

When in doubt, prefer `CodeValidator` for shape and `RuntimeGuardrail` for sequence.

## Reference

- Module: [`agents/patterns/actions/code_constraints.py`](../../src/polymathera/colony/agents/patterns/actions/code_constraints.py)
- Consumer: [`agents/patterns/actions/code_generation.py`](../../src/polymathera/colony/agents/patterns/actions/code_generation.py)
- Defaults: [`agents/patterns/actions/defaults.py`](../../src/polymathera/colony/agents/patterns/actions/defaults.py)
- Mission-level guardrails (concurrency, preemption, etc.): [`mission_and_action_guardrails_plan.md`](../../mission_and_action_guardrails_plan.md)
- Tracing for action / guardrail decisions: [`observability_design_notes.md`](../../observability_design_notes.md)
