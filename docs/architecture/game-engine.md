# Game Patterns

!!! bug "Merge this with `docs/design-insights/game-theoretic-correctness.md`"
    Or keep them separate, but minimize overlap and add cross-references.

Colony's game patterns provide structured multi-agent deliberation through formal game-theoretic protocols. Games serve as error correction mechanisms -- they combat hallucination, laziness, goal drift, and miscommunication by forcing agents into adversarial or cooperative interactions with defined rules and payoffs.

## Why Games?

LLM agents suffer from predictable failure modes. The game patterns map each failure mode to a game-theoretic countermeasure:

| Failure Mode | Game Countermeasure |
|-------------|---------------------|
| **Hallucination** | Every claim is a `ScopeAwareResult` with confidence, evidence, and missing context. No single agent's output is final. |
| **Laziness** | Contract Net task allocation with reputation-based bid selection. Agents compete for tasks. |
| **Goal drift** | `ObjectiveGuardAgent` checks each new draft against the original goal. Plans and intentions are explicit (BDI). |
| **Miscommunication** | Meta-agent normalizes vocabulary and detects mismatched assumptions across agents. |

!!! bug "Miscommunication agent is not implemented yet"



## Four Game Types

### Hypothesis Game

One agent proposes a hypothesis; others refute or refine it. Continues until the hypothesis is accepted (sufficient evidence), rejected (contradicting evidence), or refined into a new hypothesis.

```python
# Launch a complete hypothesis validation game
@action_executor(writes=["game_future"])
async def run_hypothesis_game(
    *,
    owner: Agent,
    hypothesis: Hypothesis,
    num_skeptics: int = 2,
    num_grounders: int = 1,
    use_llm_reasoning: bool = False,
    game_id: str | None = None,
) -> CapabilityResultFuture: ...
```

Roles: `HypothesisProposerAgent`, `HypothesisSkepticAgent`, `HypothesisGrounderAgent`, `HypothesisArbiterAgent`, coordinated by `HypothesisCoordinatorAgent`.

### Bidding / Contract Game

!!! bug "Explain and Justify Contract Net Protocol"
    Why do agents need to bid for work? Why is this better than a central planner assigning tasks? What are the failure modes this prevents? Can agents learn from bidding outcomes to improve future performance? What agent aspect does learning vary in this case (agent capabilities, strategies, or experience, or parameters)?

Multiple agents bid to take a subtask (Contract Net Protocol). The supervisor evaluates bids based on capability match, reputation, current load, and cache affinity. The winning bidder commits to delivery.

```python
class TaskBid(BaseModel):
    bid_id: str
    bidder_id: str
    task_id: str
    estimated_cost_tokens: int
    estimated_duration_seconds: float
    estimated_quality_gain: float          # 0.0 to 1.0
    rationale: str
    capabilities_match: list[str]
    past_performance: dict[str, float]

class ContractAward(BaseModel):
    task_id: str
    winner_id: str
    winning_bid: TaskBid
    selection_reasoning: str

class ContractNetGameCapability(GameProtocolCapability[ContractGameData, ContractNetGameRole]):
    """Phases: ANNOUNCE → BID → AWARD → EXECUTE → VALIDATE → TERMINAL"""
```

### Negotiation Game

Agents with conflicting constraints exchange offers until they reach agreement or escalate to an arbiter. Used for resource allocation, page assignment, and conflicting analysis results.

```python
@action_executor(writes=["game_future"])
async def run_negotiation_game(
    *,
    owner: Agent,
    issue: NegotiationIssue,
    strategies: dict[str, NegotiationStrategy] | None = None,
    num_participants: int = 2,
    use_llm_reasoning: bool = False,
    game_id: str | None = None,
) -> CapabilityResultFuture: ...
```

Roles: `NegotiationParticipantAgent`, `NegotiationMediatorAgent`, coordinated by `NegotiationCoordinatorAgent`.

### Consensus Game
Agents vote or provide evidence; a meta-agent aggregates using configurable voting rules. Used for final decisions where multiple independent analyses must be reconciled.

## Roles

Games define fixed roles with specific permissions:

| Role | Responsibility |
|------|---------------|
| **Proposer** | Initiates hypotheses, claims, or offers |
| **Skeptic** | Challenges claims, demands evidence, identifies weaknesses |
| **Grounder** | Connects claims to specific evidence in context pages |
| **Arbiter** | Makes final decisions when agents cannot agree |
| **Planner** | Decomposes tasks and coordinates agent assignments |

Roles map to move permissions -- a Skeptic can `challenge` and `request_evidence` but cannot `propose`. This prevents agents from stepping outside their designated function.

## `GameProtocolCapability`

`polymathera.colony.agents.patterns.games.state.GameProtocolCapability` is the base class for game implementations. It extends `AgentCapability` with:

- **State machine**: Game phases (setup, active, terminal) with validated transitions
- **Move validation**: Moves checked against role permissions before application
- **Shared scope**: All participants share a blackboard namespace via `scope_id=game_id`
- **Memory integration**: Terminal game states are captured via `MemoryProducerConfig` for episodic memory

```python
class GameProtocolCapability(AgentCapability, ABC, Generic[TGameData, TRole]):
    """Base class for game protocol implementations.

    All game participants share the same scope_id (typically the game_id),
    enabling them to see each other's moves and events via the shared blackboard.
    """

    role_permissions: RolePermissions = RolePermissions()

    def __init__(
        self,
        *,
        agent: Agent,
        game_id: str | None = None,
        game_type: str,
        role: str | None = None,
        use_llm_reasoning: bool = False,
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 500,
    ): ...

    @abstractmethod
    async def apply_move(self, state: GameState, move: ACLMessage) -> GameState: ...

    @abstractmethod
    def is_terminal(self, state: GameState) -> bool: ...

    @abstractmethod
    async def compute_outcome(self, state: GameState) -> GameOutcome: ...
```

### `RolePermissions`

Declarative permission matrix that maps `(role, phase)` pairs to allowed performatives:

```python
class RolePermissions:
    def allows(self, role: str, phase: GamePhase, performative: Performative) -> bool:
        """Fail-closed: no entry means no permission."""
        ...

    def add(self, role: str, phase: GamePhase, performatives: set[Performative]) -> "RolePermissions":
        """Add permissions (returns self for chaining)."""
        ...
```

### `GameState`

```python
class GameState(BaseModel):
    game_id: str
    game_type: str
    conversation_id: str
    participants: list[str]
    roles: dict[str, str]                # agent_id -> role
    phase: GamePhase                     # SETUP, ACTIVE, TERMINAL, etc.
    history: list[ACLMessage]            # Full move history
    game_data: dict[str, Any]            # Game-specific state
    outcome: GameOutcome | None = None
    started_at: float
    ended_at: float | None = None
```

### `GameOutcome`

```python
class GameOutcome(BaseModel):
    outcome_type: str
    success: bool
    result: Any | None = None
    rounds_played: int
    messages_exchanged: int
    duration_seconds: float | None = None
    consensus_level: float | None = None
    conflicts_resolved: int = 0
    lessons_learned: list[str] = []
```

## Agent Communication Language (ACL)

!!! bug "Is ACL Still Required?"
    The ACL message field is populated, but may not be required anymore.

Messages between agents are not plain strings. Each message has structure:

- **Illocutionary force**: The intent -- `inform`, `request`, `propose`, `promise`, `challenge`, `accept`, `reject`
- **Content**: The payload (claim, evidence, offer, etc.)
- **Preconditions**: What must be true for this message to be valid
- **Expected effects**: What the sender expects to change

This follows the FIPA Agent Communication Language model, adapted for LLM agents where the "content" is often natural language with structured metadata.

## Hybrid Architecture

The game patterns use a hybrid deliberative-reactive architecture:

- **Deliberative core**: The LLM handles planning, explanation, and reasoning about game state
- **Reactive rules**: Automated triggers that surround the LLM core:
    - Auto-invoke validators on every move
    - Auto-escalate on low confidence
    - Auto-trigger conflict resolution when contradictions are detected

Each agent's mental state is represented partly in prompts (natural language reasoning) and partly in structured state on the blackboard:

- **Beliefs**: References to blackboard entries the agent considers true
- **Goals**: Explicit `Goal` objects with success criteria
- **Intentions**: Current plans and sub-tasks

## Advanced Mechanisms

### No-Regret Learning

!!! bug "Add Code Sample, Guidance and Results"

Algorithms like Exp3/EXP4 adjust the mixture over agents and strategies based on quality metrics across games. Over time, the system learns which agents perform well at which roles and which strategies succeed in which contexts.

### Targeted Learning

!!! bug "Add Code Sample, Guidance and Results"

`TargetedLearningManager` clusters past tasks by similarity and learns per-cluster tuning parameters. Different types of analysis tasks may require different game configurations, and the system adapts automatically.

### VCG-Style Incentives

!!! bug "Add Code Sample, Guidance and Results"

Vickrey-Clarke-Groves mechanism design rewards agents for their marginal contribution to global performance. This discourages free-riding and encourages honest reporting of capabilities and confidence levels.

### Social Choice Theory

!!! bug "Add Code Sample, Guidance and Results"

Voting rules aggregate evaluator rankings for consensus games. Arrow's impossibility theorem informs rule selection -- no voting rule is perfect, so the choice depends on which properties matter most for the specific decision.

### Coalition Games

!!! bug "Add Code Sample, Guidance and Results"

Agents can form stable, high-value coalitions for complex tasks. Coalition value is approximated empirically over time. Stable coalitions persist across tasks; unstable ones are dissolved and reformed.

### Epistemic Logic

!!! bug "Add Code Sample, Guidance and Results"

Agent mental states are structured and inspectable:

- Beliefs and knowledge are classified as "agent's belief" vs. "common knowledge"
- Only `common=True` propositions appear in final confirmed reports
- This prevents individual agent biases from propagating to system output

## Integration with Action Policies

!!! bug "Game invocation from action policies"
    Currently, games are started by an **owner agent** using actions such as `run_negotiation_game` which allows the owner agent to specify the game parameters and spawn participant agents. However, these participants have exclusive roles as game players and are disposed of after the game concludes. This is a limitation -- ideally, any agent should be able to participate in games as part of its normal reasoning process, and the insights from games should inform the agent's ongoing decision-making rather than being siloed in a separate "game mode". So, we need an **ad-hoc game invocation mechanism** that allows existing agents to decide to enter a game protocol at any point during their action policy execution, and to have the game outcomes directly influence their reasoning and planning.

Games are invoked by action policies just like any other `AgentCapability`. The `CacheAwareActionPolicy` can delegate to a game protocol when it detects a situation requiring multi-agent deliberation:

- Conflicting analysis results from child agents trigger a hypothesis game
- Resource contention triggers a negotiation game
- Task decomposition triggers a bidding game
- Final synthesis triggers a consensus game

The game events flow into the planning context, and the LLM planner decides how to respond -- games inform the LLM's reasoning rather than bypassing it.
