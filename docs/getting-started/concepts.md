# Key Concepts

A quick overview of Colony's core concepts before diving into the architecture.

## Agents

An **Agent** is the fundamental unit of work. Each agent has:

- **State**: `RUNNING`, `IDLE`, or `STOPPED`
- **Capabilities**: pluggable modules that add behavior (memory, games, reflection, etc.)
- **ActionPolicy**: decides what to do next (usually LLM-driven)
- **Blackboard access**: shared state with other agents

Agents don't hardcode their behavior. The LLM planner is given context and decides "what's next?" at every step.

```python
from polymathera.colony.agents.base import Agent, AgentCapability, ActionPolicy

class Agent:
    @hookable
    async def run_step(self) -> None:
        """One iteration of the agent's reasoning loop."""
        result = await self.action_policy.execute_iteration(self._build_state())
        await self._apply_iteration_result(result)

    def add_capability(self, capability: AgentCapability, *,
                       include_actions: list[str] | None = None,
                       exclude_actions: list[str] | None = None) -> None: ...

    def get_capability(self, name: str) -> AgentCapability | None: ...
```

## `AgentCapabilities`

Capabilities are pluggable modules attached to agents. They:

- Export **action executors** (`@action_executor`) — actions the LLM planner can choose
- Register **hooks** that react to agent events (action completion, errors, etc.)
- Declare **hookable methods** that other capabilities can intercept

Examples: `MemoryCapability`, `ReflectionCapability`, `HypothesisGameProtocol`, `CriticCapability`.

```python
class MyCapability(AgentCapability):
    @action_executor()
    async def analyze(self, query: str, max_depth: int = 5) -> dict:
        """Auto-discovered by ActionPolicy — the LLM planner can invoke this."""
        ...

    @hookable
    async def process(self, data: dict) -> Result:
        """Other capabilities can intercept this via hooks."""
        ...
```

## `ActionPolicy`

The ActionPolicy controls the agent's decision loop. Colony's main implementation is `CacheAwareActionPolicy`, which:

1. Gathers context (memory, blackboard state, VCM pages)
2. Presents available actions to the LLM
3. LLM selects an action and parameters
4. Executes the action
5. Feeds results back and loops

This follows a **Model-Predictive Control** pattern: execute part of the plan, re-evaluate, adapt.

```python
class ActionPolicy(ABC):
    async def execute_iteration(
        self, state: ActionPolicyExecutionState
    ) -> ActionPolicyIterationResult:
        """Execute one iteration of the policy loop."""
        ...
```

## Virtual Context Memory (VCM)

VCM manages extremely long context across a GPU cluster. Think of it as **virtual memory for LLMs**:

- **Pages**: chunks of context that can be loaded/unloaded from GPU KV caches
- **Page Table**: tracks which pages are where (like an OS page table)
- **Page Faults**: accessing an unloaded page increases its priority for loading
- **Working Set**: the set of pages an agent is currently using

## Blackboard

The **Blackboard** is a shared, observable, transactional key-value store (Redis-backed). It serves as:

- The single source of truth for all agent state
- An event bus (state changes emit events)
- A coordination mechanism between agents

All agent state lives in blackboards. No out-of-band state in instance variables.

```python
board = EnhancedBlackboard(app_name="my-app", scope=BlackboardScope.SHARED, scope_id="team-1")
await board.initialize()

await board.write("results", my_data, created_by="agent-123", tags={"analysis"})
value = await board.read("results")

# Subscribe to changes
board.subscribe(on_change, filter=KeyPatternFilter("*:results"))
```

## Memory System

Colony's memory system is a **hierarchy of MemoryCapabilities**, each managing a different abstraction level:

- **Sensory memory**: raw events and observations
- **Working memory**: high-turnover, context-relevant items
- **Short-term memory**: decaying items that may transfer to long-term
- **Long-term memory**: persistent storage (episodic, semantic, procedural)

Each capability handles ingestion, storage, retrieval, and maintenance for its scope.

```python
# Query memory using semantic search, logical filtering, or both
result = await memory.recall(MemoryQuery(
    query="What authentication approach was used?",
    tag_filter=TagFilter(all_of={"action", "success"}),
    max_results=10,
))
```

## Games

**Game-theoretic protocols** are used for multi-agent coordination and error correction:

- **Hypothesis games**: one agent proposes, others refute or refine
- **Contract Net**: agents bid to take subtasks based on reputation
- **Negotiation**: agents with conflicting constraints exchange offers
- **Consensus**: agents vote with evidence; meta-agent aggregates

These aren't just coordination tools — they're error correction mechanisms that combat LLM failure modes.

## Sessions and Runs

A **Session** groups related work. Each call to `agent.run()` creates an **`AgentRun`** tracked within the session, with configuration, I/O history, resource usage, and intermediate events.
