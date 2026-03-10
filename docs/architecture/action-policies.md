# Action Policies

Action policies are the decision-making core of Colony agents. They implement the principle that the LLM is the planner -- the framework provides context and action descriptions, the LLM chooses what to do, and the framework executes and feeds back results.

## The LLM as Planner

Colony does not use rigid plan graphs, state machines, or rule-based orchestration. Instead:

1. The framework gathers context (goals, constraints, execution history, available actions)
2. The LLM reasons about what to do next
3. The framework executes the chosen action
4. Results feed back into context for the next iteration

A "plan" in Colony is the LLM's current thinking plus execution history -- not a fixed sequence. The LLM can revise or abandon its plan at any point based on new information.

## ActionPolicy Base Class

`polymathera.colony.agents.base.ActionPolicy` defines the contract:

```python
class ActionPolicy(ABC):
    async def execute_iteration(
        self, state: ActionPolicyExecutionState
    ) -> ActionPolicyIterationResult:
        """Execute one iteration of the policy loop."""
        ...

    async def serialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> AgentSuspensionState:
        """Serialize policy state for suspension."""
        ...

    async def deserialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> None:
        """Restore policy state from suspension."""
        ...
```

The policy manages which `AgentCapability` instances are active via `use_agent_capabilities()` and `disable_agent_capabilities()`. Active capabilities provide action executors that the policy can invoke.

## Two-Phase Action Selection

Action selection follows a two-phase process to manage the combinatorial space:

### Phase 1: Action Selection
The LLM receives descriptions of all available actions (from active capabilities) and chooses which action to take. Actions are typed via `ActionType` -- planning, reasoning, tool usage, communication, memory management, orchestration, and output.

### Phase 2: Parameterization
Once an action type is selected, the LLM receives the specific action's JSON schema and fills in parameters. This separation prevents the LLM from being overwhelmed by the full parameter space of all actions simultaneously.

## ActionPolicy I/O Contract

The policy operates on structured input and produces structured output:

**Input** (`ActionPolicyInput`):

- `goals`: What the agent is trying to achieve
- `constraints`: Boundaries on behavior
- `initial_context`: Starting context for the task
- `action_descriptions`: Available actions from active capabilities

**Output** (`ActionPolicyOutput`):

- `success`: Whether the policy achieved its goals
- `final_result`: The produced result
- `exported_results`: Results to share with other agents
- `learned_patterns`: Patterns discovered during execution

## Model-Predictive Control

`CacheAwareActionPolicy` (in `polymathera.colony.agents.patterns.actions.policies`) uses Model-Predictive Control (MPC) for plan execution:

```mermaid
graph LR
    Plan["Create/Revise Plan"] --> Execute["Execute Next Actions"]
    Execute --> Evaluate["Evaluate Results"]
    Evaluate -->|"Conditions changed"| Plan
    Evaluate -->|"On track"| Execute
    Evaluate -->|"Done"| Complete["Complete"]
```

1. **Plan**: The LLM creates or revises a plan based on current context
2. **Execute**: Execute only the next few actions (not the full plan)
3. **Evaluate**: Check results against expectations
4. **Adapt**: If conditions changed, revise the plan; otherwise continue

This accounts for the nonstationary nature of multi-agent environments -- other agents may change shared state, new information may invalidate assumptions, and resource availability fluctuates.

## CacheAwareActionPolicy

The primary policy implementation, extending `EventDrivenActionPolicy`:

```python
class CacheAwareActionPolicy(EventDrivenActionPolicy):
    """Action policy with multi-step planning.

    - Creates plans using configurable strategies (MPC, top-down, bottom-up)
    - Executes plans incrementally via Agent.run_step
    - Handles replanning when needed
    - Coordinates with child agents event-driven (no polling)
    """
```

Key features:

- **Configurable planning strategies**: MPC (default), top-down decomposition, bottom-up aggregation
- **Event-driven coordination**: No polling for child agent results; events trigger re-evaluation
- **Cache context in plans**: Every plan includes working set information, access patterns, page graph summary, prefetch hints, and shareable vs. exclusive page designations
- **Sub-plan generation**: JIT sub-plan creation when executing composite actions, with arbitrary depth and maintained position in the plan tree

!!! info "Cache-awareness is emergent"
    Cache-awareness is not a property of individual primitives. It is an emergent property of the LLM planner composing primitives with knowledge of current cache state. The most efficient strategy depends on the data being analyzed at runtime, which only the LLM can assess.

## Replanning

Replanning is triggered by:

- **Plan exhaustion**: All actions in the current plan have been executed
- **Failure**: An action fails or produces unexpected results
- **New information**: Events from other agents or blackboard changes invalidate assumptions
- **Resource changes**: VCM page availability changes

Replanning strategies:

- **Revision**: Modify the existing plan to account for new information
- **Backtracking**: Undo recent actions and try a different approach
- **Escalation**: Request help from a supervisor agent
- **Re-creation**: Discard the plan entirely and create a new one

!!! warning "Cache-conscious revision"
    When revising plans, the policy preserves cache locality when possible. Abandoning a plan may mean abandoning cached pages, so the cost of re-planning is weighed against the cost of cache misses.

## Hierarchical Planning

Plans can be hierarchical -- higher-level plans use high-level actions that encapsulate lower-level plans:

1. A parent agent creates a high-level plan with composite actions
2. When executing a composite action, a sub-plan is generated JIT
3. The sub-plan may itself contain composite actions, creating arbitrary depth
4. The policy maintains its position in the plan tree for context

This allows natural decomposition of complex tasks without requiring the LLM to plan everything up front.

## DataflowReference

Actions can reference results from previous actions, planning context, capability state, or blackboard entries via typed `DataflowReference` objects. This enables dataflow between actions without manual state threading.

## Blueprint Pattern

Policies are configured via `ActionPolicyBlueprint`, created through the `bind()` class method:

```python
blueprint = CacheAwareActionPolicy.bind(
    planning_strategy="mpc",
    max_iterations=50,
)
```

Blueprints are validated for serializability at creation time. The `agent` reference is injected at instantiation time, not bound in the blueprint.
